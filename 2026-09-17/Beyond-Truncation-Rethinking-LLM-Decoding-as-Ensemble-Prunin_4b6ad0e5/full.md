# Beyond Truncation: Rethinking LLM Decoding as Ensemble Pruning

Dunyao Xue<sup>1</sup>, Chengshuo Du<sup>1</sup>, Zhengbo Wang<sup>1</sup>, Wenlin Dai<sup>1,2,†</sup>, Cheng Meng<sup>1,3,†</sup>

<sup>1</sup>Institute of Statistics and Big Data, Renmin University of China, Beijing, China <sup>2</sup>Big Data and Responsible Artificial Intelligence for National Governance, Renmin University of China, Beijing, China <sup>3</sup>Center for Applied Statistics, Institute of Statistics and Big Data, Renmin University of China, Beijing, China {xuedunyao1202,duchengshuo,zhengbowang,wenlin.dai,chengmeng}@ruc.edu.cn

## Abstract

We introduce Mahalanobis-Ensemble Decoding (ME-Decoding), a novel Large Language Model (LLM) decoding framework that frames candidate token selection as ensemble pruning. Existing selection strategies rely predominantly on scalar probabilities, ignoring geometric semantic relationships and causing candidate redundancy. Meanwhile, current geometry-aware methods often require complex optimization or directly reweighting the original token probabilities, leading to significant computational overhead or inference instability. To address this, we formulate decoding as a subset optimization problem using a Mahalanobis distancedriven objective to enhance semantic diversity while preserving high probabilities. Specifically, we dynamically discount redundant generation paths using a token similarity matrix, constructed via an adaptive-bandwidth kernel over token embeddings. We further devise an efficient greedy selection algorithm with nearlinear complexity in the candidate size under early stopping, while establishing its theoretical approximation guarantees. This renders ME-Decoding a robust, plug-and-play module with negligible inference overhead. Extensive experiments across diverse reasoning and generation tasks demonstrate that our method consistently achieves strong performance.

## 1 Introduction

Large Language Models (LLMs) have demonstrated strong capabilities in mathematical reasoning, instruction following, and open-ended generation (Brown et al., 2020; Touvron et al., 2023; Chowdhery et al., 2023; OpenAI, 2023). Besides model scaling and post-training alignment, the inference-time decoding strategy also plays a crucial role in determining generation quality. At each step, an autoregressive LLM produces a next-token distribution, from which the decoder selects or samples the next token. Deterministic methods such as greedy decoding and beam search (Holtzman et al., 2020) favor high-probability continuations but often produce repetitive or overly conservative outputs. Sampling-based methods introduce stochasticity and improve generation diversity, but they may also assign non-negligible probability to low-quality tokens, increasing inference uncertainty and potentially leading to illogical continuations or hallucinations.

To control this quality–diversity trade-off, many decoding methods perform probability-based truncation and reshaping of the next-token distribution. Classical approaches such as Top-k and nucleus sampling restrict sampling to high-probability tokens (Fan et al., 2018; Holtzman et al., 2020), while later methods such as Min-p, entropy-aware sampling, and p-less adapt the truncation rule according to model confidence or distributional statistics (Hewitt et al., 2022; Nguyen et al., 2025; Tan et al., 2026). Despite their effectiveness, these methods mainly operate on token probabilities and treat candidates as independent categorical outcomes. As a result, they overlook the semantic geometry among tokens, which may retain redundant candidates and limit the effectiveness of the selected sampling support (Yang et al., 2026).

Although recent geometry-aware methods incorporate token-space structure, they continue to face significant challenges. For example, Top-W formulates a Wasserstein-regularized distributionmatching problem (Davoodi et al., 2026), which requires approximating the Wasserstein objective and careful hyperparameter tuning, while not explicitly optimizing redundancy within the selected token support. Alternatively, CraEG penalizes tokens in crowded regions via a lightweight reweighting mechanism (Yang et al., 2026). However, the approach still relies on an auxiliary decoding method for post-processing, making its behavior dependent on the properties of the downstream method. These limitations motivate our key question:

Table 1: Comparison of different decoding methods.
<table><tr><td>Method</td><td>Decoding Strategy</td><td>Aware1</td><td>Selection</td><td>Geometry Adaptive Redundancy Control²</td></tr><tr><td>Greedy</td><td>Determin.</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Top-k</td><td>Prob. Trunc.</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Min-p</td><td>Prob. Trunc.</td><td>x</td><td>x</td><td>x</td></tr><tr><td>p-less</td><td>Prob. Trunc.</td><td>x</td><td>√</td><td>x</td></tr><tr><td>Top-W</td><td>Dist. Match.</td><td>√</td><td>√</td><td>x</td></tr><tr><td>CraEG</td><td>Reweighting</td><td>√</td><td>x</td><td>√</td></tr><tr><td>Ours</td><td>Ensemble</td><td>√</td><td>√</td><td>√</td></tr></table>

<sup>1</sup> Embedding Geometry: Uses token embedding geometry in the decoding process.  
<sup>2</sup> Redundancy Control: Explicitly controls redundancy among candidate tokens.

How can we design a simple and efficient framework that selects a compact yet informative token set while preserving high-confidence candidates?

In this work, we address this question from the perspective of ensemble pruning, a paradigm that entails selecting a compact subset of learners from a larger pool to optimize predictive outcomes. This paradigm closely mirrors the token selection problem in LLM decoding. Extensive research in ensemble learning shows that effective ensembles should balance individual accuracy with collective complementarity (Krogh and Vedelsby, 1994; Opitz and Maclin, 1999). Consequently, simply selecting the highest-scoring learners may introduce redundancy and hurt overall performance (Zhou et al., 2002; Li et al., 2012). Similarly, LLM decoding encounters an analogous challenge of token redundancy. Building on this insight, we introduce ME-Decoding, a novel framework that incorporates token geometry to reformulate decoding as a subset maximization problem driven by the Mahalanobis distance. By approximately solving this objective with an efficient greedy procedure, ME-Decoding mitigates redundancy within the selected token set during generation with negligible inference overhead, thereby improving token selection quality while maintaining reasoning performance.

Our contributions are summarized as follows:

• We reformulate LLM decoding through the novel lens of ensemble pruning.

• We define a Mahalanobis distance-driven objective to adaptively select a compact, informative candidate set while preserving token probabilities.

• We devise an efficient greedy selection algorithm with candidate-linear complexity under early stopping, while establishing its trajectory unimodality and theoretical approximation guarantees.

• Extensive experiments across diverse reasoning and generation tasks show that our method consistently outperforms strong existing baselines.

## 2 Motivation

## 2.1 Background of LLM Decoding

Large Language Models (LLMs) generate text autoregressively by predicting the next token from the preceding context. At each step, an LLM outputs a probability distribution $\mathcal { P } \in \Delta ^ { | \Omega | - 1 }$ over the full vocabulary Ω. To avoid sampling from the unreliable long tail, modern decoding methods usually first form a candidate pool ${ \nu } _ { t } \subseteq \Omega .$ , then select a final subset $S \subseteq \mathcal { V } _ { t }$ and sample from the renormalized distribution:

$$
p _ { S } ( i ) = \frac { p _ { i } { \bf 1 } \{ i \in S \} } { \sum _ { j \in S } p _ { j } } .
$$

This process can be viewed as a subset selection problem aiming to preserve the original distribution under a probability-based criterion:

$$
\begin{array} { r l } & { S ^ { \star } = \arg \underset { S \subseteq \mathcal { V } _ { t } } { \operatorname* { m i n } } f ( \pmb { p } _ { S } ) , } \\ & { } \\ & { \mathrm { s . t . } \quad S = \{ i : p _ { i } \geq \tau ( \mathscr { P } ) \} , } \end{array}
$$

where f(·) denotes a generalized decoding objective and $\tau ( \mathcal { P } )$ is a probability-based truncation threshold.

Most prevailing decoding methods follow this probability-driven paradigm. Top-k and nucleus sampling truncate the distribution using fixed probability or cumulative-mass thresholds, while adaptive methods such as p-less and entropy-aware sampling adjust the threshold based on distributional statistics. Despite different truncation rules, these methods remain probability-centric and treat candidate tokens as independent outputs. They therefore overlook semantic dependencies among tokens, motivating us to incorporate inter-token semantics into the subset selection objective.

![](images/f8f93c2d7d330e74ad72f867ff1df7118302390867abc9e44cc1169181f31154.jpg)  
Figure 1: Comparison between probability-based truncation methods (left) and our ME-Decoding framework (right). ME-Decoding extends standard probability-only token selection by incorporating embedding-based similarity into the MES score and selecting a compact geometry-aware subset for sampling.

## 2.2 Connection to Ensemble Pruning

Ensemble pruning aims to select a compact subensemble that preserves or improves the predictive performance of the full ensemble. Classical error analysis (Breiman, 2001) characterizes ensemble performance as a trade-off between individual accuracy and inter-model diversity. Following this idea, Zhang et al. (2006) formulate pruning as a quadratic subset-selection problem. Given a binary error indicator matrix E, they construct the error co-occurrence matrix $\pmb { C } = \pmb { E } ^ { \top } \pmb { E }$ , where diagonal entries measure individual errors and off-diagonal entries measure shared failures. After normalization, the pruning objective can be written as

$$
\operatorname* { m i n } _ { s } s ^ { \top } \widehat C s \quad \mathrm { s . t . } \sum _ { i } s _ { i } = N , s _ { i } \in \{ 0 , 1 \} .\tag{1}
$$

This objective selects a size-N sub-ensemble by jointly penalizing individual inaccuracy and pairwise redundancy.

This paradigm naturally parallels token selection in LLM decoding. Candidate tokens can be viewed as ensemble members: their probabilities measure individual confidence, while their embeddingspace similarities measure semantic redundancy. Therefore, an ideal decoding subset should retain high-likelihood tokens while suppressing redundant candidates, mirroring the accuracy–diversity trade-off in ensemble pruning.

## 2.3 Generalizing Ensemble Pruning to LLM Decoding via Mahalanobis Distance

The quadratic objective in Eq. 1 leverages secondorder statistics to penalize redundant candidates. Motivated by this, we aim to extend this diversityaware framework to LLM decoding. To achieve this, we observe that for any representation vector p and a symmetric positive definite matrix, this quadratic structure naturally corresponds to the Mahalanobis distance.

In machine learning, the Mahalanobis distance is widely used to construct discriminative representations, such as in metric learning, out-of-distribution detection, and representation regularization (Weinberger and Saul, 2009; Lee et al., 2018; Wan et al., 2018; Kang et al., 2026). Formally, given two vectors x, $\pmb { y } \in \mathbb { R } ^ { d }$ and a symmetric positive definite matrix $M \succ 0$ , it is defined as

$$
\mathcal { D } _ { M } ( \pmb { x } , \pmb { y } ; M ) : = \sqrt { ( \pmb { x } - \pmb { y } ) ^ { \top } M ^ { - 1 } ( \pmb { x } - \pmb { y } ) } .
$$

In particular, setting $\mathbf { \nabla } _ { \mathbf { y } } = \mathbf { 0 }$ and letting M be the token similarity matrix K gives the Mahalanobis norm

$$
\| \pmb { x } \| _ { K } : = \sqrt { \pmb { x } ^ { \top } K ^ { - 1 } \pmb { x } } .
$$

Mathematically, by explicitly incorporating the inverse matrix $\pmb { K } ^ { - 1 }$ , the Mahalanobis norm inherently downweights correlated directions and separates information along non-redundant axes. This geometric property aligns perfectly with our overarching goal: to suppress repetitive generation paths and enhance token diversity during decoding.

## 3 Method: ME-Decoding

In this section, we propose Mahalanobis-Ensemble Decoding (ME-Decoding), a geometry-aware decoding framework inspired by ensemble pruning. ME-Decoding selects a compact token subset by optimizing a Mahalanobis-style objective that combines token probabilities with embedding-based similarities. The overall structure of ME-Decoding is illustrated on the right side of Figure 1.

## 3.1 Mahalanobis-Ensemble Score.

Inspired by ensemble pruning, we view the candidate tokens at each decoding step as a pool of weak semantic predictors. Instead of retaining a fixed number of high-probability tokens, our goal is to construct a compact token ensemble that preserves high-confidence candidates while reducing redundancy in the token embedding space. At each decoding step, ME-Decoding is applied to a probability-based candidate pool $\begin{array} { r } { \partial _ { t } \subseteq \Omega ; } \end{array}$ ; for simplicity, we omit t and write V hereafter.

We first define the Mahalanobis-Ensemble $E n \cdot$ ergy (MEE) for a selected token subset $s \subseteq \nu$ as

$$
\mathrm { M E E } ( \mathcal { S } ) = p _ { S } ^ { \top } K _ { \mathcal { S } } ^ { - 1 } p _ { \mathcal { S } } ,
$$

where $p _ { \mathcal { S } }$ denotes the vector of token probabilities indexed by S, and $\pmb { K } _ { \mathcal { S } }$ is the token similarity matrix restricted to S. The inverse matrix $\pmb { K } _ { S } ^ { - 1 }$ discounts redundant tokens and rewards subsets with high collective quality.

However, unlike standard ensemble pruning, the optimal subset size in decoding is unknown a priori. This requires us to maximize the MEE of the selected tokens while avoiding unnecessarily large subsets, thereby filtering out redundant tokens that contribute only marginally to the overall MEE. To this end, we define the Mahalanobis-Ensemble Score (MES) as follows:

$$
\mathrm { M E S } _ { \lambda } ( \mathcal { S } ) = \frac { \mathrm { M E E } ( \mathcal { S } ) } { \left[ 1 + \lambda H _ { 2 } ^ { T } ( \pmb { p } ) \right] ^ { | \mathcal { S } | } } ,
$$

where $\begin{array} { r } { H _ { 2 } ^ { T } ( p ) = 1 - \sum _ { i \in \mathcal { V } } p _ { i } ^ { 2 } } \end{array}$ is the order-2 Tsallis entropy measuring the uncertainty of the nexttoken distribution, and $\lambda > 0$ controls the entropydependent size penalty.

Intuitively, the numerator rewards tokens with large non-redundant contributions, while the denominator discourages excessive subset expansion. When the distribution is flat, the unregularized MEE may keep increasing by accumulating weak marginal gains from many uncertain candidates. The entropy-dependent penalty raises the inclusion threshold, retaining only tokens with sufficiently large non-redundant contributions. Consequently, the decoding objective is given by:

$$
S ^ { \star } = \arg \operatorname* { m a x } _ { S \subseteq \mathcal { V } } \mathrm { M E S } _ { \lambda } ( { S } ) .\tag{2}
$$

By optimizing the MES, we can effectively extract a highly representative token subset from the candidate distribution.

## 3.2 Construction of Similarity Matrix

To measure token similarity in the embedding space while maintaining numerical stability, we construct K using a Gaussian kernel on normalized token embeddings. Let $e _ { i } \in \mathbb { R } ^ { d }$ denote the normalized embedding of token i. We define

$$
C _ { i j } = 1 - e _ { i } ^ { \top } e _ { j } , \qquad K _ { i j } = \exp \{ - C _ { i j } / \epsilon \} .\tag{3}
$$

Here, $C _ { i j }$ measures the semantic distance between tokens, and $\epsilon > 0$ is a bandwidth parameter controlling the smoothness of the kernel.

In our implementation, the bandwidth ϵ is chosen adaptively according to the probability-weighted semantic dispersion of the candidate tokens:

$$
\epsilon = \frac { 1 } { 2 } \sum _ { i , j \in \mathcal { V } } p _ { i } p _ { j } C _ { i j } ,
$$

where $\textstyle \sum _ { i , j \in \mathcal { V } } p _ { i } p _ { j } C _ { i j }$ is the expected pairwise semantic distance between tokens sampled from the candidate distribution. When the candidate tokens are semantically concentrated, the adaptive bandwidth becomes smaller, yielding a localized kernel and making the selection more probability-driven. Conversely, when the candidates are semantically dispersed, the bandwidth becomes larger, preserving semantic relations over a broader range and making the selection more geometry-aware. Thus, the adaptive bandwidth balances probability and semantic structure according to the local geometry of the decoding distribution.

## 3.3 Optimization Strategy

Directly solving the maximization problem formulated in Eq. 2 is an NP-hard combinatorial task. To maintain computational feasibility, we use a greedy forward-selection procedure and maintain the inverse Cholesky factor of $\pmb { K } _ { \mathcal { S } } ^ { - 1 }$ incrementally (Golub and Van Loan, 2013), which avoids repeated matrix inversions during subset construction.

The greedy selection in Algorithm 1 is conceptually related to Orthogonal Matching Pursuit (OMP) (Pati et al., 1993). At each step, $( p _ { j } - \pmb { \alpha } _ { j } ^ { \top } z )$ represents the conditional residual score of token j after accounting for its correlation with the selected tokens, while $r _ { j }$ normalizes it by the corresponding conditional variance. The marginal contribution is therefore measured by $\left\lceil r _ { j } ( p _ { j } - \pmb { \alpha } _ { j } ^ { \top } z ) \right\rceil ^ { 2 }$ . The algorithm selects the token with the largest contribution and accepts it only when the penalized MES objective improves, favoring high-probability tokens while discounting redundant candidates.

Algorithm 1 Greedy Algorithm for ME-Decoding   
1: Input: Candidate token pool V with $N$   
$| \nu | ;$ candidate token scores $\pmb { p } ~ \in ~ \mathbb { R } ^ { N }$ ; nor  
malized candidate token embeddings $\textbf { \em E } =$   
$[ e _ { 1 } , \ldots , e _ { N } ] ^ { \top } \in \mathbb { R } ^ { N \times d } ;$ MES penalty λ.   
2: Initialize ${ \mathcal { S } } = { \mathcal { D } } .$   
3: Set $c _ { \lambda }  1 + \lambda H _ { 2 } ^ { T } ( p ) .$   
4: Select the first token $j _ { 1 } = \arg \operatorname* { m a x } _ { i } p _ { i }$ and   
update $S \gets \{ j _ { 1 } \}$   
5: Compute $K _ { \mathcal { S } , \mathcal { S } }$ using Eq. 3.   
6: $\pmb { R }  ( \pmb { K } _ { S , S } ^ { 1 / 2 } ) ^ { - 1 } , \quad z  \pmb { R } p _ { S } .$   
7: $\begin{array} { r } { \mathrm { M E S } ^ { g }  \| z \| _ { 2 } ^ { 2 } / c _ { \lambda } ^ { | S | } . } \end{array}$   
8: for $t = 1$ to $N - 1$ do   
9: for $j \in \{ 1 , \dots , N \} \setminus S$ do   
10: Compute $\beta _ { j }  [ K _ { i j } ] _ { i \in \mathcal { S } }$ using Eq. 3, and   
set $b _ { j }  { \pmb K } _ { j j } .$   
11: $\begin{array} { r } { \pmb { \alpha } _ { j } \overset { \cdot } {  } \pmb { R } \pmb { \beta } _ { j } , \quad r _ { j } \gets ( b _ { j } - \| \pmb { \alpha } _ { j } \| _ { 2 } ^ { 2 } ) ^ { - 1 / 2 } . } \end{array}$   
12: Compute the MEE after adding token $j \colon$   
13: $c _ { j } \gets \hat { \| } z \| _ { 2 } ^ { 2 } + \Big [ r _ { j } ( p _ { j } - \pmb { \alpha } _ { j } ^ { \top } z ) \Big ] ^ { 2 } .$   
14: end for   
15: Select $j _ { \star } = \arg \operatorname* { m a x } _ { j \notin S } c _ { j } ,$ let $c ^ { \star } = c _ { j _ { \star } }$   
16: Compute the candidate MES score:   
17: $\mathrm { M E S } _ { \mathrm { c a n d } } ^ { g }  c ^ { \star } / c _ { \lambda } ^ { | S | + 1 } .$   
18: if $\mathrm { M E S } _ { \mathrm { c a n d } } ^ { g } \leq \mathrm { M E S } ^ { g }$ then   
19: break   
20: end if   
21: $\gamma \gets - r _ { j _ { \star } } R ^ { \top } { \alpha } _ { j _ { \star } }$   
22: $v _ { j _ { \star } }  r _ { j _ { \star } } ( p _ { j _ { \star } } - \alpha _ { j _ { \star } } ^ { \top } z )$   
23: Update R and $_ { z : }$   
$\begin{array} { r } { R  [ \begin{array} { c c } { R } & { \mathbf { 0 } } \\ { \gamma ^ { \top } } & { r _ { j _ { \star } } } \end{array} ] , \quad z  [ \begin{array} { c c } { z } \\ { v _ { j _ { \star } } } \end{array} ] . } \end{array}$   
24: $\begin{array} { r } { \mathcal { S }  \mathcal { S } \cup \{ j _ { \star } \} , \quad \mathrm { M E S } ^ { g }  \mathrm { M E S } _ { \mathrm { c a n d } } ^ { g } . } \end{array}$   
25: end for   
26: Output: Selected token set ${ \mathcal { S } } .$

## 3.4 Theoretical Properties

Furthermore, to rigorously validate the effectiveness of our proposed strategy, we establish theoretical properties of the greedy algorithm, showing its unimodality along the search path and deriving an approximation bound to the global optimum.

The following theorem states that, under a conditioning assumption on the token-similarity matrix, the greedy MES sequence cannot increase again once it stops increasing.

Theorem 3.1 (Greedy Unimodality of Mahalanobis-Ensemble Decoding). Let $| \nu | = N$ and $\mathrm { M E S } _ { t } ^ { g } = \mathrm { M E S } _ { \lambda } ( S _ { t } )$ , where $\{ S _ { t } \} _ { t = 0 } ^ { N }$ is the greedy path generated by Algorithm 1. For any $T \subseteq$ $\nu ,$ let ${ \pmb { K } } _ { T } = ( K _ { i j } ) _ { i , j \in { T } }$ and define $\kappa _ {  { N } } (  { \boldsymbol { K } } ) =$ max $T \subseteq \nu , | T | \leq N \ \frac { \lambda _ { \operatorname* { m a x } } ( { \dot { K } } _ { T } ) } { \lambda _ { \operatorname* { m i n } } ( K _ { T } ) }$ . If $\begin{array} { l c l } { \kappa _ { N } ( { \pmb K } ) } & { \leq } & { 1 + } \end{array}$ $\lambda H _ { 2 } ^ { T } ( p )$ , oncefor some $t < N ,$

$$
\mathrm { M E S } _ { t + 1 } ^ { g } \leq \mathrm { M E S } _ { t } ^ { g } ,
$$

then for all $s \geq t ,$ $\mathrm { M E S } _ { s + 1 } ^ { g } \ \leq$ MES<sup>g</sup><sub>s</sub>. Consequently, with $\tau = \mathrm { m i n } \{ t : \bar { \mathrm { M E S } } _ { t + 1 } ^ { g } \leq \mathrm { M E S } _ { t } ^ { g } \}$ , we have

$$
\mathrm { M E S } _ { \tau } ^ { g } = \operatorname* { m a x } _ { 0 \leq r \leq N } \mathrm { M E S } _ { r } ^ { g } .
$$

Theorem 3.1 justifies the early stopping rule in Algorithm 1: the greedy search can terminate once the MES score drops, without traversing all candidates. Appendix B.3 complements this sufficient-condition result by disabling early stopping and recording complete greedy trajectories on Qwen2.5-1.5B. All 512 trajectories on each of GSM8K and GPQA are unimodal, providing empirical support for the stopping behavior characterized by Theorem 3.1. The next theorem compares the stopped greedy value with the global optimum, showing that it provides an effective approximation to the optimal subset.

Theorem 3.2 (Approximation Guarantee for Mahalanobis-Ensemble Decoding). Let

$$
\mathrm { M E S } _ { \lambda } ^ { \star } = \operatorname* { m a x } _ { S \subseteq \gamma } \mathrm { M E S } _ { \lambda } ( S ) , \qquad | \mathcal { V } | = N .
$$

Let $\{ { \cal S } _ { t } \} _ { t = 0 } ^ { N }$ be the greedy path generated by Algorithm $^ { l , }$ and let τ be the stopping time defined in Theorem 3.1. Under the assumptions of Theorem 3.1, we have

$$
\begin{array} { r } { \mathrm { M E S } _ { \tau } ^ { g } \geq \left( 1 - \exp \{ - \lambda _ { \operatorname* { m i n } } ( K , N ) \} \right) \mathrm { M E S } _ { \lambda } ^ { \star } , } \end{array}
$$

where $\begin{array} { r } { \lambda _ { \operatorname* { m i n } } ( K , r ) \ \triangleq \ \operatorname* { m i n } _ { T \subseteq \mathcal { V } , \ | T | = r } \lambda _ { \operatorname* { m i n } } ( K _ { T } ) } \end{array}$ denotes the smallest eigenvalue among all $r \times r$ principal submatrices ofK.

Theorem 3.2 establishes the relation between the optimal greedy value and the globally optimal MES value. The approximation factor depends on the restricted minimum eigenvalue of $\kappa$ , which reflects the non-redundancy and conditioning of the selected token representations.

<table><tr><td></td><td colspan="3">Qwen3-4B-Inst.</td><td colspan="3">Phi-4-mini-Inst.</td><td colspan="3">Mistral-7B-Inst.</td><td></td></tr><tr><td>Method</td><td> $T = 1 . 0$ </td><td> $T = 1 . 5$ </td><td> $T = 2 . 0$ </td><td> $T = 1 . 0$ </td><td> $T = 1 . 5$ </td><td> $T = 2 . 0$ </td><td> $T = 1 . 0$ </td><td> $T = 1 . 5$ </td><td> $T = 2 . 0$ </td><td>Avg.</td></tr><tr><td>Min-p</td><td>70.74</td><td>62.02</td><td>54.06</td><td>79.68</td><td>70.81</td><td>30.86</td><td>48.67</td><td>36.32</td><td>17.97</td><td>52.35</td></tr><tr><td>Top-p</td><td>68.01</td><td>55.19</td><td>21.00</td><td>80.06</td><td>11.75</td><td>0.30</td><td>48.90</td><td>23.05</td><td>0.45</td><td>34.30</td></tr><tr><td>p-less</td><td>78.92</td><td>72.71</td><td>63.15</td><td>84.05</td><td>83.09</td><td>69.52</td><td>54.06</td><td>50.80</td><td>46.47</td><td>66.97</td></tr><tr><td>Top-H</td><td>76.80</td><td>67.17</td><td>58.53</td><td>81.65</td><td>73.77</td><td>34.65</td><td>52.01</td><td>46.78</td><td>22.52</td><td>57.10</td></tr><tr><td>Top-W</td><td>77.71</td><td>76.65</td><td>74.98</td><td>82.64</td><td>83.24</td><td>82.18</td><td>53.98</td><td>53.15</td><td>51.02</td><td>70.62</td></tr><tr><td>Ours</td><td>80.67</td><td>79.76</td><td>80.06</td><td>84.08</td><td>84.23</td><td>82.41</td><td>55.34</td><td>53.45</td><td>53.90</td><td>72.66</td></tr></table>

Table 2: GSM8K accuracy (%) across different temperatures and decoding methods. The Avg. column reports the average accuracy over all models and temperatures. The best result is in bold and the second best is underlined.
<table><tr><td></td><td colspan="3">Qwen3-4B-Inst.</td><td colspan="3">Phi-4-mini-Inst.</td><td colspan="3">Mistral-7B-Inst.</td><td></td></tr><tr><td>Method</td><td> $T = 1 . 0$ </td><td> $T = 1 . 5$ </td><td> $T = 2 . 0$ </td><td> $T = 1 . 0$ </td><td> $T = 1 . 5$ </td><td> $T = 2 . 0$ </td><td> $T = 1 . 0$ </td><td> $T = 1 . 5$ </td><td> $T = 2 . 0$ </td><td>Avg.</td></tr><tr><td>Min-p</td><td>34.38</td><td>32.59</td><td>35.04</td><td>27.01</td><td>29.02</td><td>26.56</td><td>26.12</td><td>28.12</td><td>25.22</td><td>29.34</td></tr><tr><td>Top-p</td><td>34.60</td><td>35.49</td><td>30.58</td><td>31.25</td><td>27.01</td><td>15.40</td><td>25.89</td><td>26.41</td><td>14.06</td><td>26.74</td></tr><tr><td>p-less</td><td>35.27</td><td>34.82</td><td>35.49</td><td>33.26</td><td>33.26</td><td>27.90</td><td>27.68</td><td>27.01</td><td>24.78</td><td>31.08</td></tr><tr><td>Top-H</td><td>34.38</td><td>33.93</td><td>34.15</td><td>34.38</td><td>33.93</td><td>29.91</td><td>28.12</td><td>27.23</td><td>23.88</td><td>31.10</td></tr><tr><td>Top-W</td><td>34.82</td><td>36.83</td><td>35.27</td><td>31.70</td><td>29.46</td><td>31.47</td><td>27.23</td><td>29.02</td><td>28.23</td><td>31.56</td></tr><tr><td>Ours</td><td>35.49</td><td>35.49</td><td>35.71</td><td>34.38</td><td>36.16</td><td>33.93</td><td>28.35</td><td>28.79</td><td>28.35</td><td>32.96</td></tr></table>

Table 3: GPQA accuracy (%) across different temperatures and decoding methods. The Avg. column reports the average accuracy over all models and temperatures. The best result is in bold and the second best is underlined

## 4 Experiments

## 4.1 Experimental Details

Models and Baselines. We evaluate ME-Decoding on three instruction-tuned language models: Qwen3-4B-Instruct (Yang et al., 2025), Phi-4-mini-Instruct (Abouelenin et al., 2025), and Mistral-7B-Instruct (Jiang et al., 2023). We compare our method with several representative decoding methods, including Min-p (Nguyen et al., 2025), Top-p (Holtzman et al., 2020), p-less (Tan et al., 2026), Top-H (Baghaei Potraghloo et al., 2025), and Top-W (Davoodi et al., 2026). For a fair comparison, all methods are evaluated under the same prompts, temperature settings, maximum generation length, and stopping criteria. Unless otherwise specified, we evaluate all methods at temperatures $T \in \{ 1 . 0 , 1 . 5 , 2 . 0 \}$ All reported experiments use $N = 5 1 2 .$ . We additionally implement CraEG (Yang et al., 2026) and combine its post-softmax reweighting with p-less, following its plug-in formulation. Repeated results for selected methods are reported in Appendix B.1.

Benchmarks and Metrics. We consider two types of tasks. First, for reasoning benchmarks, we evaluate on GSM8K (Cobbe et al., 2021) and GPQA (Rein et al., 2024), where performance is measured by answer accuracy after extracting the final answer. These benchmarks test whether a decoding method can maintain reliable reasoning performance under different sampling temperatures. Second, for instruction-following and chat-style generation, we evaluate on AlpacaEval (Li et al., 2023) and MT-Bench (Zheng et al., 2023). For AlpacaEval, we report the candidate win-rate (%), and for MT-Bench, we report the average judge score. Since these tasks involve open-ended generation, we use DeepSeek-V4-Pro (DeepSeek-AI, 2026) as the judge to compare the generated responses under a fixed evaluation protocol. For reproducibility, AlpacaEval compares each response with a fixed reference for the same prompt, whereas MT-Bench independently scores each response on a 1–10 scale. Appendix B.7 reports an additional evaluation using GLM-5.2 as a second judge, together with paired prompt-level bootstrap confidence intervals.

ME-Decoding Configuration. ME-Decoding selects a compact token subset by optimizing a Mahalanobis-style objective that balances token probability and embedding geometry. For Top-W and ME-Decoding, both of which require an explicit candidate pool, we use the same top-N pool with $\begin{array} { r } { \begin{array} { r c l } { N } & { = } & { 5 1 2 } \end{array} } \end{array}$ Except for the candidate-pool size, Top-W uses its default hyperparameters. Top-p, Min-p, and Top-H follow Davoodi et al. (2026), and p-less is evaluated under its original parameter-free rule. ME-

Decoding uses λ = 0.9 by default, following Appendix B. The official implementation is available at https://github.com/sapphirexdy/ME\_decoding.

Temperature Placement. At temperature T, every method operates on the same distribution $p _ { T } =$ softmax $( z / T )$ . ME-Decoding uses $p _ { T }$ for candidate scoring and samples from the renormalized selected support, while each baseline applies its truncation or reweighting rule to the same temperatureadjusted distribution.

## 4.2 Main Results

Reasoning Benchmarks. We first evaluate ME-Decoding on GSM8K and GPQA across three models and three temperatures. The results are reported in Tables 2 and 3. Overall, ME-Decoding achieves the best average performance across both datasets. Appendix B.1 reports three-seed repetitions as mean ± sample standard deviation and includes CraEG combined with p-less following its plug-in formulation.

Figure 2 further shows the accuracy–temperature curves of different decoding methods. Compared with existing baselines, ME-Decoding maintains a more stable accuracy profile as the temperature increases. Higher temperatures enlarge the sampling space, making probability-only truncation methods more prone to noisy tokens. By incorporating token geometry, ME-Decoding better preserves high-confidence candidates while filtering out uninformative ones, leading to a better balance between exploration and reasoning reliability.

Instruction-following and Chat. We further evaluate ME-Decoding on instruction-following and chat-style generation tasks using MT-Bench and AlpacaEval. Experiments are conducted on Qwen3-4B, Phi-4-mini, and Mistral-7B under $T \in$ {1.0, 1.5, 2.0}. Figure 3 summarizes the results. The left and middle panels report the average AlpacaEval win rate and MT-Bench score over the three models at each temperature, while the right panel presents the overall average rank across all models, temperatures, and benchmarks. Detailed numerical results are provided in the Appendix.

As shown in Figure 3, ME-Decoding achieves strong performance on both open-ended benchmarks. It obtains the best averaged AlpacaEval win rate and MT-Bench score. The overall rank comparison further shows that ME-Decoding attains the best average rank among all compared decoding methods, indicating its stable advantage across models, temperatures, and benchmarks. These results suggest that ME-Decoding is not limited to accuracy-oriented reasoning tasks, but also improves open-ended generation by selecting a compact and informative token set. The same conclusion holds under a second automatic judge. Paired prompt-level bootstrap intervals against Top-W, pless, and Top-H are positive for both benchmarks and both judges, as reported in Appendix B.7.

## 4.3 Analysis

Complexity Analysis. We analyze the time complexity of ME-Decoding in Algorithm 1. Let N be the candidate-pool size, d the embedding dimension, and τ the number of selected tokens before early stopping. The kernel computation costs $\mathcal { O } ( N \tau d )$ , and the greedy evaluation costs $\mathcal { O } ( N \tau ^ { 3 } )$ leading to $\mathcal { O } \big ( N \tau ( d + \tau ^ { 2 } ) \big )$ . Since early stopping usually yields a small τ with $\tau \ll N$ , the practical cost scales nearly linearly with N and approaches $O ( N d )$ when τ is treated as a small constant. A detailed analysis is provided in Appendix B.4.

Efficiency Analysis. To compare inference-time efficiency, we use a CPU-only synthetic logits benchmark that isolates sampling and logitsprocessing overhead. We measure the average time for filtering logits and sampling one token under two settings: Medium with vocabulary size 32,000, embedding dimension 256, and top- $N = 5 1 2 ;$ and Large with vocabulary size 128,000, embedding dimension 1024, and $\mathrm { t o p } { - } N = 2 0 4 8$ . All experiments use batch size 1, one CPU thread, temperature 1.0, 50 warm-up steps, and 500 measured steps over 10 repeats. Results are reported in Table 4.

As shown in Table 4, ME-Decoding introduces moderate overhead compared with purely probability-based methods due to its use of token embeddings, but remains highly efficient. It is about 2.7× faster than Top-W in the Medium setting and 7.4× faster in the Large setting. In the Large setting, its overhead is also comparable to Top-p and Top-H, demonstrating that ME-Decoding can exploit token-geometry information with substantially lower cost than existing geometry-aware decoding methods. Appendix B.4 additionally reports end-to-end GPU latency and shows that model inference remains the dominant cost under the measured settings.

Diversity Analysis. We analyze the accuracy– diversity trade-off on GSM8K with Qwen3-4B. For each method, we randomly sample 500 questions and generate 10 responses per question. Accuracy is computed over all generations, and diversity is measured by Distinct-1/2 (Li et al., 2016) and Self-BLEU (Zhu et al., 2018). We further average the min-max normalized Distinct-1, Distinct-2, and 1 − Self-BLEU within each temperature as an aggregated diversity score.

![](images/90c151b1e27f75b66f4e60bde801dd4b8a2544e7258f8b2fe8df93d1cb2443d9.jpg)

![](images/f1d0aa719fd9ca1a78e4b1c631f8ce8ea398d5238bafc49e394f76144febe2af.jpg)

Figure 2: Accuracy vs. temperature curves of different decoding methods on GSM8K and GPQA.  
![](images/ce444594b6baa4f7d8408c97ca8f234957be55b50e7f9aecc7b48689d3f47f39.jpg)  
Figure 3: Open-ended generation performance and overall ranking comparison. The left and middle panels report AlpacaEval win rate and MT-Bench score across different temperatures, averaged over three models and aggregated over 3 runs. The right panel summarizes the average rank of each decoding method across all models, temperatures, and benchmarks, with bold numbers indicating the best overall rank.

Figure 4 shows a clear accuracy–diversity tradeoff. More exploratory methods, such as Top-p, achieve higher diversity, especially at high temperatures, but suffer from notable accuracy degradation. In contrast, ME-Decoding consistently achieves the highest accuracy and maintains stronger reasoning performance at comparable diversity levels. This suggests that the proposed Mahalanobis-Ensemble objective improves token selection reliability by constructing a compact and informative sampling support, rather than simply increasing output randomness. Detailed results are provided in Table 14.

To further test whether ME-Decoding constructs an informative geometry-aware support rather than merely truncating more aggressively, we compare it with probability-only controls calibrated on independent prompts to match its average support size, post-pruning entropy, or support-level pairwise cosine. At each step with $| S | \ge 2 ,$ , Cos. is the mean cosine over all unordered pairs of L2- normalized selected-token embeddings; we then average it across eligible steps. Table 5 reports results on Qwen2.5-1.5B at $T = 1 . 0$ over three generation seeds.

![](images/35d2fe486fcfe44ca3e2071c830b704e1ff16a91d9864373764c530b1c335cde.jpg)  
Figure 4: Accuracy–diversity trade-off on GSM8K using Qwen3-4B. Each method generates 10 responses for 500 sampled questions under each temperature.

<table><tr><td>Setting</td><td>Statistic</td><td>Top-p</td><td>Min-p</td><td>p-less</td><td>Top-H</td><td>Top-W</td><td>ME-Decoding</td></tr><tr><td rowspan="2">Large</td><td>Mean s/token</td><td>0.01407</td><td>0.00196</td><td>0.00314</td><td>0.01493</td><td>0.13337</td><td>0.01799</td></tr><tr><td>Standard Deviation</td><td>0.00006</td><td>0.00002</td><td>0.00000</td><td>0.00009</td><td>0.00098</td><td>0.00560</td></tr><tr><td>Medium</td><td>Mean s/token Standard Deviation</td><td>0.00335 0.00003</td><td>0.00054 0.00001</td><td>0.00080 0.00001</td><td>0.00330 0.00005</td><td>0.00487 0.00003</td><td>0.00179 0.00010</td></tr></table>

Table 4: Average CPU sampling overhead per token on synthetic logits. Large: vocabulary size 128K, embedding dimension 1024, top-N = 2048. Medium: vocabulary size 32K, embedding dimension 256, top-N = 512.

<table><tr><td>Dataset</td><td>Method</td><td>Acc. (%)</td><td>Avg. |S|</td><td>Cos.</td><td>MES</td></tr><tr><td rowspan="4">GSM8K</td><td>ME-Decoding</td><td> ${ \bf 6 3 . 2 0 \pm 0 . 5 7 }$ </td><td>1.076</td><td>0.151</td><td>0.756</td></tr><tr><td>Size-matched</td><td> $6 1 . 7 9 \pm 1 . 2 6$ </td><td>1.067</td><td>0.2100.707</td><td></td></tr><tr><td>Entropy-matched</td><td> $6 1 . 7 9 \pm 0 . 9 9$ </td><td>1.070</td><td></td><td>0.2080.708</td></tr><tr><td>Cosine-matched</td><td> $6 2 . 7 0 \pm 0 . 3 5$ </td><td>1.014</td><td>0.198</td><td>0.715</td></tr><tr><td rowspan="4">GPQA</td><td>ME-Decoding</td><td> $\mathbf { 3 2 . 6 6 \pm 1 . 0 5 }$ </td><td>1.134</td><td>0.243</td><td>0.589</td></tr><tr><td>Size-matched</td><td> $2 8 . 4 7 \pm 1 . 1 7$ </td><td>1.106</td><td></td><td>0.2890.558</td></tr><tr><td>Entropy-matched</td><td> $2 8 . 4 7 \pm 1 . 1 7$ </td><td>1.106</td><td></td><td>0.2890.558</td></tr><tr><td>Cosine-matched</td><td> $2 9 . 2 5 \pm 0 . 9 1$ </td><td>1.084</td><td></td><td>0.2590.537</td></tr></table>

Table 5: ME-Decoding and approximately matched probability-only controls on Qwen2.5-1.5B at T = 1.0. Accuracy is reported as mean ± sample standard deviation over three seeds.

Across both datasets, ME-Decoding achieves the highest accuracy and MES under approximately matched support size, entropy, or pairwise cosine. This result is consistent with our design goal of selecting a high-confidence support whose members provide complementary geometric information, and supports the effectiveness of directly optimizing MES. Appendix B.6 reports the complete diagnostics, including retained probability mass and entropy before pruning. Appendix B.2 further isolates the contribution of correctly aligned token geometry using identity- and permuted-kernel controls.

## 5 Conclusion

In this paper, we introduce ME-Decoding, a decoding framework that reformulates LLM token selection as ensemble pruning. ME-Decoding maximizes the Mahalanobis-Ensemble Score (MES) to select a compact token subset, combining token probabilities with an embedding-based similarity matrix to discount semantic redundancy. We further develop an efficient greedy algorithm with theoretical guarantees, including greedy-trajectory unimodality and an approximation bound. Experiments on reasoning and open-ended generation tasks show that ME-Decoding improves over representative probability-based and geometry-aware baselines with low inference overhead.

These results demonstrate the potential of ensemble pruning as a principled perspective for constructing compact, diverse, and reliable token candidate sets. Future work includes developing simpler and more effective objectives under this perspective, and extending ME-Decoding to broader scenarios such as long-form generation, multi-sample reasoning, and verification-augmented decoding.

## Limitations

Although ME-Decoding introduces an ensemblepruning perspective for LLM decoding and achieves strong performance on several tasks, it still has several limitations. First, its performance depends on the construction of the similarity matrix, which controls the trade-off between token confidence and redundancy. While we provide a practical geometry-based kernel, more effective and adaptive similarity designs may further improve performance. The current kernel uses static token embeddings as a soft redundancy signal and may not capture all context-dependent semantic distinctions. Contextualized or hidden-stateconditioned kernels are therefore a useful direction for future work. Second, although ME-Decoding consistently achieves higher accuracy than competing methods under comparable diversity levels, its output diversity is not always the highest among all decoding strategies. This suggests that the current kernel construction may still be conservative in promoting diverse generations. How to further improve generation diversity while preserving the accuracy gains of ME-Decoding remains an important direction for future work. Third, determining the optimal subset size remains challenging, as in existing truncation-based decoding methods. The proposed MES criterion provides a principled selection rule, but it may not be optimal for all token distributions. Exploring adaptive subset-size rules and alternative objectives is another promising direction.

## Ethical Considerations

This work proposes ME-Decoding, a geometryaware decoding framework that reformulates candidate token selection from the perspective of ensemble pruning. By selecting compact yet informative token sets, ME-Decoding aims to improve generation quality and reasoning performance without modifying model parameters or requiring additional training. A potential positive impact is that such plug-and-play inference-time methods may improve the usability of existing language models with limited computational overhead, thereby reducing deployment costs and the need for repeated model retraining.

At the same time, improved decoding and reasoning performance may also amplify downstream risks associated with generative systems. These risks include misinformation generation, hallucinated but plausible outputs, biased or harmful content, and the automation of low-quality or malicious text at scale. Since ME-Decoding operates at inference time and can be applied to a wide range of pretrained language models, its broader impact depends strongly on the deployment context, access control, and downstream use cases.

We encourage practitioners to use ME-Decoding together with established responsible deployment practices, including safety evaluation, bias and toxicity assessment, hallucination analysis, usage policies, and rate limits in open-ended settings. For high-stakes applications such as medical, legal, financial, or educational decision-making, model outputs should be carefully verified by qualified human experts. In addition, when ME-Decoding is applied to models trained on copyrighted, private, or sensitive data, developers should follow appropriate data governance, documentation, and licensing practices.

All datasets, models, and evaluation tools used in this work are publicly available. We follow their respective licenses and terms of use, and use them solely for research evaluation purposes. Our use of these artifacts is consistent with their intended use as benchmarks, pretrained models, and evaluation tools for research. We do not redistribute or repurpose any artifact beyond the scope allowed by its original access conditions, and we cite the original creators of all benchmarks, models, and baseline methods used in our experiments.

We do not collect any new user data or personally identifying information. The datasets used in our experiments are publicly available benchmarks released for research evaluation. We use them only under their standard evaluation settings and do not attempt to identify individuals or recover private information from any dataset.

## Acknowledgments

This work was supported by the Outstanding Innovative Talents Cultivation Funded Programs 2026 of Renmin University of China and the National Natural Science Foundation of China under Grant No. 12571301.

## References

Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany Awadalla, Nguyen Bach, Jianmin Bao, Alon Benhaim, Martin Cai, Vishrav Chaudhary, Congcong Chen, Dong Chen, Dongdong Chen, Junkun Chen, Weizhu Chen, Yen-Chun Chen, Yi-ling Chen, Qi Dai, Xiyang Dai, Ruchao Fan, and 55 others. 2025. Phi-4-mini technical report: Compact yet powerful multimodal language models via mixture-of-LoRAs. arXiv preprint arXiv:2503.01743.

Erfan Baghaei Potraghloo, Seyedarmin Azizi, Souvik Kundu, and Massoud Pedram. 2025. Top-H decoding: Adapting the creativity and coherence with bounded entropy in text generation. In Advances in Neural Information Processing Systems, volume 38, pages 28482–28513.

Leo Breiman. 2001. Random forests. Machine Learning, 45(1):5–32.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, and 12 others. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, pages 1877–1901.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, and 48 others. 2023. PaLM: Scaling language modeling with pathways. Journal ofMachine Learning Research, 24(240):1–113.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Abhimanyu Das and David Kempe. 2018. Approximate submodularity and its applications: Subset selection, sparse approximation and dictionary selection. Journal ofMachine Learning Research, 19(3):1–34.

Arash Gholami Davoodi, Navid Rezazadeh, Seyed Pouyan Mousavi Davoudi, and Pouya Pezeshkpour. 2026. Geometry-aware decoding with Wassersteinregularized truncation and mass penalties for large language models. In Proceedings ofthe 43rd Inter national Conference on Machine Learning.

DeepSeek-AI. 2026. DeepSeek-V4: Towards highly efficient million-token context intelligence. https://huggingface.co/deepseek-ai/ DeepSeek-V4-Pro. Technical report.

Angela Fan, Mike Lewis, and Yann Dauphin. 2018. Hierarchical neural story generation. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 889–898.

Gene H Golub and Charles F Van Loan. 2013. Matrix Computations. JHU Press.

John Hewitt, Christopher D Manning, and Percy Liang. 2022. Truncation sampling as language model desmoothing. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2022, pages 3414– 3427.

Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. 2020. The curious case of neural text degeneration. In International Conference on Learning Representations.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7b. Preprint, arXiv:2310.06825.

Xinlai Kang, Dunyao Xue, Zhengbo Wang, Chengshuo Du, Xinghao Chen, Hang Zhou, Hanting Chen, and Cheng Meng. 2026. Breaking the echo chamber: A dynamic ensemble pruning perspective on MoE. In Proceedings ofthe 43rd International Conference on Machine Learning.

Anders Krogh and Jesper Vedelsby. 1994. Neural network ensembles, cross validation, and active learning. In Advances in Neural Information Processing Systems, volume 7, pages 231–238.

Kimin Lee, Kibok Lee, Honglak Lee, and Jinwoo Shin. 2018. A simple unified framework for detecting outof-distribution samples and adversarial attacks. In Advances in Neural Information Processing Systems, volume 31, pages 7167–7177.

Jiwei Li, Michel Galley, Chris Brockett, Jianfeng Gao, and William B Dolan. 2016. A diversity-promoting objective function for neural conversation models. In Proceedings ofthe 2016 conference ofthe North American chapter of the association for computational linguistics: human language technologies, pages 110–119.

Nan Li, Yang Yu, and Zhi-Hua Zhou. 2012. Diversity regularized ensemble pruning. In Joint European conference on machine learning and knowledge discovery in databases, pages 330–345. Springer.

Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, Ishaan Gulrajani, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Alpacaeval: An automatic evaluator of instruction-following models. https://github.com/tatsu-lab/alpaca\_eval.

Nhat Minh Nguyen, Andrew Baker, Clement Neo, Allen G. Roush, Andreas Kirsch, and Ravid Shwartz-Ziv. 2025. Turning up the heat: Min-p sampling for creative and coherent LLM outputs. In International Conference on Learning Representations.

OpenAI. 2023. GPT-4 technical report. arXiv preprint arXiv:2303.08774.

David Opitz and Richard Maclin. 1999. Popular ensemble methods: An empirical study. Journal of artificial intelligence research, 11:169–198.

Yagyensh Chandra Pati, Ramin Rezaiifar, and Perinkulam Sambamurthy Krishnaprasad. 1993. Orthogonal matching pursuit: Recursive function approximation with applications to wavelet decomposition. In Proceedings of 27th Asilomar conference on signals, systems and computers, pages 40–44. IEEE.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. 2024. GPQA: A graduate-level google-proof q&a benchmark. In Proceedings of the First Conference on Language Modeling.

Runyan Tan, Shuang Wu, and Phillip Howard. 2026. p-less sampling: A robust hyperparameter-free approach for LLM decoding. In International Conference on Learning Representations.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. LLaMA: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Weitao Wan, Yuanyi Zhong, Tianpeng Li, and Jiansheng Chen. 2018. Rethinking feature distribution for loss functions in image classification. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 9117–9126.

Kilian Q Weinberger and Lawrence K Saul. 2009. Distance metric learning for large margin nearest neighbor classification. Journal ofMachine Learning Research, 10(9):207–244.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Yixin Yang, Qingxiu Dong, and Zhifang Sui. 2026. Decoding in geometry: Alleviating embedding-space crowding for complex reasoning. arXiv preprint arXiv:2601.22536.

Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chenghua Huang, Chengxing Xie, et al. 2026. Glm-5: from vibe coding to agentic engineering. arXiv preprint arXiv:2602.15763.

Yi Zhang, Samuel Burer, and W. Nick Street. 2006. Ensemble pruning via semi-definite programming. Journal ofMachine Learning Research, 7(48):1315– 1338.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-judge with mt-bench and chatbot arena. In Advances in Neural Information Processing Systems, volume 36, pages 46595–46623. Curran Associates, Inc.

Zhi-Hua Zhou, Jianxin Wu, and Wei Tang. 2002. Ensembling neural networks: many could be better than all. Artificial intelligence, 137(1-2):239–263.

Yaoming Zhu, Sidi Lu, Lei Zheng, Jiaxian Guo, Weinan Zhang, Jun Wang, and Yong Yu. 2018. Texygen: A benchmarking platform for text generation models. In The 41st international ACM SIGIR conference on research & development in information retrieval, pages 1097–1100.

## A Proofs of Theoretical Results

Proof of Theorem 3.1. Let $\mathrm { M E E } ( S ) = p _ { S } ^ { \top } K _ { S } ^ { - 1 } p _ { S }$ and $c _ { \lambda } = 1 + \lambda H _ { 2 } ^ { T } ( p )$ . Write $F _ { t } = \mathrm { M E E } ( S _ { t } )$ and $\Delta _ { t } = \mathrm { M E E } ( S _ { t + 1 } ) - \mathrm { M E E } ( S _ { t } )$ . Then the greedy objective is defined as $\begin{array} { r } { \mathrm { \ M E S } _ { t } ^ { g } = \frac { F _ { t } } { c _ { \mathrm { \lambda } } ^ { t } } } \end{array}$

We first show that the restricted condition number controls the conditional residual correlations. Fix any subset $S \subseteq \mathcal { V }$ and two indices $i , k \notin S$ . Define the conditional residual covariance matrix of $( i , k )$ given $S$ as the Schur complement:

$$
C _ { \{ i , k \} | S } = K _ { \{ i , k \} } - K _ { \{ i , k \} , S } K _ { S } ^ { - 1 } K _ { S , \{ i , k \} } .
$$

Write

$$
C _ { \{ i , k \} | S } = \binom { v _ { i } } { c _ { k i | S } } \frac { c _ { k i | S } } { v _ { k } } ) ,
$$

where $v _ { i } = K _ { i i } - K _ { i S } K _ { S } ^ { - 1 } K _ { S i } , v _ { k } = K _ { k k } - K _ { k S } K _ { S } ^ { - 1 } K _ { S k } , \mathrm { a n d } c _ { k i | S } = K _ { k i } - K _ { k S } K _ { S } ^ { - 1 } K _ { S i }$ . The corresponding conditional residual correlation is

$$
\rho _ { k i | S } = \frac { c _ { k i | S } } { \sqrt { v _ { i } v _ { k } } } .
$$

Since $C _ { \{ i , k \} | S }$ is the Schur complement of $K _ { S }$ in $K _ { S \cup \{ i , k \} }$ , its eigenvalues are bounded by those of $K _ { S \cup \{ i , k \} }$ . More precisely,

$$
\lambda _ { \operatorname* { m i n } } ( \boldsymbol K _ { S \cup \{ i , k \} } ) \leq \lambda _ { \operatorname* { m a x } } ( \boldsymbol C _ { \{ i , k \} | S } ) \leq \lambda _ { \operatorname* { m a x } } ( \boldsymbol K _ { S \cup \{ i , k \} } ) .
$$

These inequalities follow from the variational characterization of eigenvalues and the Schur-complement residualization argument. Therefore,

$$
\kappa ( \boldsymbol { C } _ { \{ i , k \} | S } ) \leq \kappa ( \boldsymbol { K } _ { S \cup \{ i , k \} } ) \leq \kappa _ { N } ( \boldsymbol { K } ) .
$$

On the other hand, for any $2 \times 2$ positive definite matrix

$$
C = \left( \begin{array} { c c } { { v _ { i } } } & { { c } } \\ { { c } } & { { v _ { k } } } \end{array} \right) ,
$$

with correlation $\textstyle \rho = { \frac { c } { \sqrt { v _ { i } v _ { k } } } }$ , we have:

$$
| \rho | \leq { \frac { \kappa ( C ) - 1 } { \kappa ( C ) + 1 } } ,
$$

which gives

$$
\kappa ( C ) \geq { \frac { 1 + | \rho | } { 1 - | \rho | } } .
$$

Applying this to $C _ { \{ i , k \} | S }$ gives

$$
\frac { 1 + | \rho _ { k i | S } | } { 1 - | \rho _ { k i | S } | } \leq \kappa ( { \pmb C } _ { \{ i , k \} | S } ) \leq \kappa _ { N } ( { \pmb K } ) .
$$

By the assumption $\kappa _ { N } ( { \pmb K } ) \leq c _ { \lambda }$ , we obtain

$$
{ \frac { 1 + | \rho _ { k i | S } | } { 1 - | \rho _ { k i | S } | } } \leq c _ { \lambda } .\tag{4}
$$

Next, we prove the one-step growth control of greedy marginal gains. Fix a greedy step t, let $S = S _ { t } ,$ and let i be the element added at step $t , \mathrm { i . e . , } S _ { t + 1 } = S _ { t } \cup \{ i \}$ . For any remaining token $k \not \in S \cup \{ i \}$ , define the standardized residual scores

$$
z _ { i } = \frac { p _ { i } - { K _ { i S } K _ { S } ^ { - 1 } p _ { S } } } { \sqrt { K _ { i i } - { K _ { i S } K _ { S } ^ { - 1 } K _ { S i } } } } ,
$$

and define $z _ { k }$ analogously. By the Schur-complement formula, $\Delta \mathrm { M E E } ( i \mid S ) = z _ { i } ^ { 2 }$ and $\Delta \mathrm { M E E } ( k \mid S ) =$ $z _ { k } ^ { 2 } .$ . Since i is selected greedily, $z _ { k } ^ { 2 } \le z _ { i } ^ { 2 } . \mathrm { I f } z _ { i } = 0 .$ , then all remaining marginal gains are zero and the conclusion is immediate. Otherwise, define $\begin{array} { r } { r _ { k } = \frac { z _ { k } } { z _ { i } } } \end{array}$ , where $| r _ { k } | \le 1$ . After adding i, the marginal gain of k is

$$
\Delta \mathrm { M E E } ( k \mid S \cup \{ i \} ) = \frac { ( z _ { k } - \rho _ { k i | S } z _ { i } ) ^ { 2 } } { 1 - \rho _ { k i | S } ^ { 2 } } .
$$

Hence

$$
\frac { \Delta \mathrm { M E E } ( k \mid S \cup \{ i \} ) } { \Delta \mathrm { M E E } ( i \mid S ) } = \frac { ( r _ { k } - \rho _ { k i \mid S } ) ^ { 2 } } { 1 - \rho _ { k i \mid S } ^ { 2 } } \leq \frac { ( 1 + | \rho _ { k i \mid S } | ) ^ { 2 } } { 1 - \rho _ { k i \mid S } ^ { 2 } } = \frac { 1 + | \rho _ { k i \mid S } | } { 1 - | \rho _ { k i \mid S } | } \leq c _ { \lambda } ,\tag{5}
$$

where the last inequality follows from (4). Taking the maximum over all remaining k gives

$$
\Delta _ { t + 1 } \leq c _ { \lambda } \Delta _ { t } .\tag{6}
$$

Now suppose $\mathrm { M E S } _ { t + 1 } ^ { g } \leq \mathrm { M E S } _ { t } ^ { g }$ . Since $\begin{array} { r } { \mathrm { M E S } _ { t } ^ { g } = \frac { F _ { t } } { c _ { \lambda } ^ { t } } } \end{array}$ , this is equivalent to

$$
\frac { F _ { t + 1 } } { c _ { \lambda } ^ { t + 1 } } \leq \frac { F _ { t } } { c _ { \lambda } ^ { t } } .
$$

Using $F _ { t + 1 } = F _ { t } + \Delta _ { t }$ , we get $F _ { t } + \Delta _ { t } \leq c _ { \lambda } F _ { t }$ . Define $\begin{array} { r } { R _ { t } = \frac { \Delta _ { t } } { F _ { t } } } \end{array}$ . Then

$$
\mathrm { M E S } _ { t + 1 } ^ { g } \leq \mathrm { M E S } _ { t } ^ { g } \quad \Longleftrightarrow \quad R _ { t } \leq c _ { \lambda } - 1 .\tag{7}
$$

Using (6), we have

$$
\begin{array} { r } { R _ { t + 1 } = \frac { \Delta _ { t + 1 } } { F _ { t + 1 } } = \frac { \Delta _ { t + 1 } } { F _ { t } + \Delta _ { t } } } \\ { \leq \frac { c _ { \lambda } \Delta _ { t } } { F _ { t } + \Delta _ { t } } = \frac { c _ { \lambda } R _ { t } } { 1 + R _ { t } } . } \end{array}
$$

The function $\begin{array} { r } { h ( R ) = \frac { c _ { \lambda } R } { 1 + R } } \end{array}$ is increasing for $R \geq 0$ . Therefore, if $R _ { t } \le c _ { \lambda } - 1$ , then

$$
R _ { t + 1 } \leq h ( R _ { t } ) \leq h ( c _ { \lambda } - 1 ) = c _ { \lambda } - 1 .
$$

By induction, $R _ { s } \le c _ { \lambda } - 1$ for all $s \geq t .$ . Using (7), we obtain

$$
\mathrm { M E S } _ { s + 1 } ^ { g } \leq \mathrm { M E S } _ { s } ^ { g } , \qquad \forall s \geq t .
$$

Finally, define

$$
\tau = \operatorname* { m i n } \{ t \in \{ 0 , \dots , N - 1 \} : \mathrm { M E S } _ { t + 1 } ^ { g } \leq \mathrm { M E S } _ { t } ^ { g } \} ,
$$

the sequence increases strictly before $\tau$ and is nonincreasing after τ. Hence

$$
\mathrm { M E S } _ { \tau } ^ { g } = \operatorname* { m a x } _ { 0 \leq r \leq N } \mathrm { M E S } _ { r } ^ { g } .
$$

This completes the proof.

Lemma A.1 (Approximation Guarantee for Greedy Selection). Let $f : 2 ^ { U } \to \mathbb { R } _ { \geq 0 }$ be a normalized $( f ( \varnothing ) = 0 )$ , nonnegative, monotone set function, and let $\mathrm { O P T } = \operatorname* { m a x } _ { | S | \leq k } f ( S )$ denote the maximum value obtained by any set of size at most k. Let S be the set selected by the Greedy algorithm. Then, the solution satisfies the following approximation guarantee:

$$
f ( S ) \geq \left( 1 - e ^ { - \gamma _ { U , k } } \right) \cdot \mathrm { O P T } ,\tag{8}
$$

where $\gamma _ { U , k }$ is the submodularity ratio of f. Formally, $\gamma _ { U , k }$ is defined as the minimum ratio of the marginal gain ofa set to the marginal gain ofits individual elements:

$$
\gamma _ { U , k } = \operatorname* { m i n } _ { \substack { L \subseteq U , A : | A | \leq k , L \cap A = \emptyset } } \frac { \sum _ { x \in A } \left( f ( L \cup \{ x \} ) - f ( L ) \right) } { f ( L \cup A ) - f ( L ) } .\tag{9}
$$

The ratio is taken over pairs $( L , A )$ such that $f ( L \cup A ) > f ( L )$ ; ifthe denominator is zero, the ratio is defined as 1.

Proof. This is the standard approximation guarantee for greedy maximization of a nonnegative monotone set function with submodularity ratio ${ \gamma } _ { U , k }$ . Proof can be found in Das and Kempe (2018). □

While Lemma A.1 offers a general guarantee, the ratio $\gamma _ { \nu , k }$ is typically intractable. We further provide a concrete bound for our Mahalanobis-Ensemble Energy (MEE) by considering the specific objective

$$
f ( S ) = \mathrm { M E E } ( S ) = p _ { S } ^ { \top } K _ { S } ^ { - 1 } p _ { S } .
$$

The following lemma bounds $\gamma _ { \nu , k }$ via the restricted minimum eigenvalue of the kernel matrix $\kappa$

Lemma A.2 (Schur Complement Preserves the Minimum Eigenvalue). Let $\ b { K } \in \mathbb { R } ^ { n \times n }$ be a symmetric positive definite matrix. Write

$$
K = \binom { K _ { 1 1 } } { s ^ { \top } } \quad K _ { n n } \Big ) , \quad K _ { 1 1 } \in \mathbb { R } ^ { ( n - 1 ) \times ( n - 1 ) } , ~ s \in \mathbb { R } ^ { ( n - 1 ) \times 1 } , ~ K _ { n n } > 0 .
$$

Define the Schur complement

$$
\pmb { K } ^ { \prime } = \pmb { K } _ { 1 1 } - \frac { 1 } { K _ { n n } } \pmb { s s } ^ { \top } .
$$

Then

$$
\lambda _ { \operatorname* { m i n } } ( K ) \leq \lambda _ { \operatorname* { m i n } } ( K ^ { \prime } ) .
$$

Proof. Let $\lambda _ { 1 } ^ { \prime } = \lambda _ { \operatorname* { m i n } } ( K ^ { \prime } )$ , and choose a nonzero eigenvector $\boldsymbol { e } ^ { \prime } \in \mathbb { R } ^ { n - 1 }$ such that $K ^ { \prime } e ^ { \prime } = \lambda _ { 1 } ^ { \prime } e ^ { \prime }$ . We construct $\boldsymbol { e } = { \binom { e ^ { \prime } } { e _ { n } } }$ , where $\begin{array} { r } { e _ { n } = - \frac { 1 } { K _ { n n } } \pmb { s } ^ { \top } e ^ { \prime } } \end{array}$ . Using the block form of $\kappa$ , we have:

$$
K e = \binom { K _ { 1 1 } e ^ { \prime } + s e _ { n } } { s ^ { \top } e ^ { \prime } + K _ { n n } e _ { n } } .
$$

By the definition of $e _ { n }$ , the lower block becomes $s ^ { \top } e ^ { \prime } + K _ { n n } e _ { n } = 0$ . For the upper block, we have $\begin{array} { r } { K _ { 1 1 } e ^ { \prime } + s e _ { n } = K _ { 1 1 } e ^ { \prime } - \frac { 1 } { K _ { n n } } s s ^ { \top } e ^ { \prime } = \left( K _ { 1 1 } - \frac { 1 } { K _ { n n } } s s ^ { \top } \right) e ^ { \prime } = K ^ { \prime } e ^ { \prime } = \lambda _ { 1 } ^ { \prime } e ^ { \prime } } \end{array}$ . Therefore, $K e = \binom { \lambda _ { 1 } ^ { \prime } e ^ { \prime } } { 0 }$

By the Rayleigh quotient, we know $\begin{array} { r } { \lambda _ { \operatorname* { m i n } } ( K ) = \operatorname* { m i n } _ { x \neq 0 } \frac { x ^ { \top } K x } { x ^ { \top } x } \leq \frac { e ^ { \top } K e } { e ^ { \top } e } } \end{array}$ . Since $e ^ { \top } K e = \lambda _ { 1 } ^ { \prime } \| e ^ { \prime } \| _ { 2 } ^ { 2 }$ and $\begin{array} { r } { e ^ { \top } e = \| e ^ { \prime } \| _ { 2 } ^ { 2 } + e _ { n } ^ { 2 } \geq \| e ^ { \prime } \| _ { 2 } ^ { 2 } } \end{array}$ , we can bound the quotient as follows:

$$
\lambda _ { \operatorname* { m i n } } ( K ) \leq \frac { e ^ { \top } K e } { e ^ { \top } e } = \frac { \lambda _ { 1 } ^ { \prime } \| e ^ { \prime } \| _ { 2 } ^ { 2 } } { \| e ^ { \prime } \| _ { 2 } ^ { 2 } + e _ { n } ^ { 2 } } \leq \lambda _ { 1 } ^ { \prime } .
$$

Since $\pmb { K } ^ { \prime } \succ 0$ , we have $\lambda _ { 1 } ^ { \prime } > 0$ , which implies $\lambda _ { 1 } ^ { \prime } = \lambda _ { \operatorname* { m i n } } ( K ^ { \prime } )$ . This proves the claim.

Lemma A.3 (Spectral Bound on Submodularity Ratio). Assume that $\kappa \succ 0$ is a correlation matrix. For $f ( S ) = p _ { S } ^ { \intercal } K _ { S } ^ { - 1 } p _ { S }$ and disjoint sets $L , A ,$ let $K _ { L \cup A }$ be partitioned naturally into blocks $K _ { L } , K _ { A } , K _ { L A } , K _ { A L } .$ . We define the residual statistics p˜ and K<sup>˜</sup> (Schur complement) as:

$$
\tilde { p } = p _ { A } - K _ { A L } K _ { L } ^ { - 1 } p _ { L } , \quad \tilde { K } = K _ { A } - K _ { A L } K _ { L } ^ { - 1 } K _ { L A } .
$$

The submodularity ratio ${ \mathrm { \Omega } } \gamma _ { \nu } , k ,$ which involves minimizing the term $\frac { \tilde { p } ^ { \top } [ \mathrm { d i a g } ( \tilde { K } ) ] ^ { - 1 } \tilde { p } } { \tilde { p } ^ { \top } \tilde { K } ^ { - 1 } \tilde { p } }$ , is boundedfrom below by the eigenvalues of K:

$$
\gamma \nu , k \geq \operatorname* { m i n } _ { \stackrel { L \subseteq \mathcal { V } , \stackrel { A \subseteq \mathcal { V } \backslash L } } { | A | \leq k } } \lambda _ { \operatorname* { m i n } } ( K , | L \cup A | ) \geq \lambda _ { \operatorname* { m i n } } ( K , N ) ,
$$

where $\lambda _ { \operatorname* { m i n } } ( K , k ) \triangleq$ min $S { : } | S | { = } k \ \lambda _ { \operatorname* { m i n } } \left( K _ { S } \right)$ denotes the smallest eigenvalue among all $k \times k$ principal submatrices ofK, and $N = | \dot { \mathcal { V } } |$

Proof. Consider disjoint L and A. Using the block inverse formula for $\pmb { K } _ { L \cup A } ^ { - 1 }$ , one obtains the decomposition

$$
f ( L \cup A ) = p _ { L } ^ { \top } K _ { L } ^ { - 1 } p _ { L } \ + \ \tilde { p } ^ { \top } \tilde { K } ^ { - 1 } \tilde { p } ,
$$

hence

$$
f ( L \cup A ) - f ( L ) = \tilde { p } ^ { \top } \tilde { K } ^ { - 1 } \tilde { p } .
$$

For a singleton $a \in A .$ , the same calculation with $| { \cal A } | = 1$ gives

$$
f ( L \cup \{ a \} ) - f ( L ) = \frac { \tilde { p } _ { a } ^ { 2 } } { \tilde { K } _ { a a } } .
$$

Therefore, for any $L \subseteq \mathcal { V }$ and A with $| A | \le k$ and $A \cap L = \emptyset$ , the ratio appearing in the definition of the submodularity ratio becomes

$$
\frac { \sum _ { a \in A } \big ( f ( L \cup \{ a \} ) - f ( L ) \big ) } { f ( L \cup A ) - f ( L ) } = \frac { \tilde { p } ^ { \top } \big [ \mathrm { d i a g } ( \tilde { K } ) \big ] ^ { - 1 } \tilde { p } } { \tilde { p } ^ { \top } \tilde { K } ^ { - 1 } \tilde { p } } .
$$

Then, let $D = \mathrm { d i a g } ( \tilde { \cal K } )$ and define the correlation matrix $\tilde { \cal K } _ { \rho } = D ^ { - 1 / 2 } \tilde { \cal K } D ^ { - 1 / 2 } . \mathrm { L e t } v = D ^ { - 1 / 2 } \tilde { p } .$ Then

$$
\tilde { p } ^ { \top } D ^ { - 1 } \tilde { p } = \| \pmb { v } \| _ { 2 } ^ { 2 } , \qquad \tilde { p } ^ { \top } \tilde { \pmb { K } } ^ { - 1 } \tilde { p } = \pmb { v } ^ { \top } \tilde { \pmb { K } } _ { \rho } ^ { - 1 } \pmb { v } .
$$

Hence the ratio equals $\frac { \pmb { v } ^ { \top } \pmb { v } } { \pmb { v } ^ { \top } \tilde { \pmb { K } } _ { \rho } ^ { - 1 } \pmb { v } }$ . By Rayleigh–Ritz, $\begin{array} { r } { \pmb { v } ^ { \top } \tilde { \pmb { K } } _ { \rho } ^ { - 1 } \pmb { v } \leq \lambda _ { \operatorname* { m a x } } ( \tilde { \pmb { K } } _ { \rho } ^ { - 1 } ) \pmb { v } ^ { \top } \pmb { v } = \frac { 1 } { \lambda _ { \operatorname* { m i n } } ( \tilde { \pmb { K } } _ { \rho } ) } \pmb { v } ^ { \top } \pmb { v } , } \end{array}$ so

$$
\frac { \tilde { p } ^ { \top } D ^ { - 1 } \tilde { p } } { \tilde { p } ^ { \top } \tilde { K } ^ { - 1 } \tilde { p } } \geq \lambda _ { \operatorname* { m i n } } ( \tilde { K } _ { \rho } ) .
$$

We can eliminate the elements of L one by one via residualization; each elimination replaces the current covariance by the covariance of residuals. By Lemma $\mathsf { A } . 2 ,$ each such residualization step cannot decrease the smallest eigenvalue. After eliminating all of $L ,$ we obtain the Schur complement $\tilde { \kappa }$ corresponding to conditioning on $L ,$ and thus

$$
\lambda _ { \operatorname* { m i n } } ( K _ { L \cup A } ) \leq \lambda _ { \operatorname* { m i n } } ( \tilde { K } ) .
$$

Finally, normalizing $\tilde { \kappa }$ to unit variance gives ${ \tilde { \cal K } } _ { \rho } ,$ and $\lambda _ { \operatorname* { m i n } } ( \tilde { K } ) \leq \lambda _ { \operatorname* { m i n } } ( \tilde { K } _ { \rho } )$ . Combining the last two displays:

$$
\lambda _ { \operatorname* { m i n } } ( \tilde { K } _ { \rho } ) \geq \lambda _ { \operatorname* { m i n } } ( K _ { L \cup A } ) \geq \lambda _ { \operatorname* { m i n } } ( K , | L \cup A | ) .
$$

By definition of $\gamma _ { \nu , k }$ , we conclude

$$
\gamma \nu , \boldsymbol { k } \geq \operatorname* { m i n } _ { \boldsymbol { L } \subseteq \nu , \boldsymbol { A } \subseteq \nu \setminus \boldsymbol { L } } \lambda _ { \operatorname* { m i n } } ( \boldsymbol { K } , | \boldsymbol { L } \cup \boldsymbol { A } | ) .
$$

Since $K _ { L \cup A }$ is a principal submatrix of K, Cauchy’s interlacing theorem gives

$$
\lambda _ { \operatorname* { m i n } } ( K _ { L \cup A } ) \geq \lambda _ { \operatorname* { m i n } } ( K ) .
$$

In our notation, $\lambda _ { \operatorname* { m i n } } ( { \pmb K } ) = \lambda _ { \operatorname* { m i n } } ( { \pmb K } , N )$ , which naturally completes the proof.

Lemma A.4 (Approximation Bound for Greedy MEE). Assume that $\kappa \succ 0$ is a correlation matrix defined on ground set $\mathcal { V } \left( \left| \mathcal { V } \right| = N \right)$ . Let $S _ { k }$ be the greedy set after k steps. Then the approximation bound holds:

$$
f ( S _ { k } ) \geq ( 1 - \exp \{ - \lambda _ { \operatorname* { m i n } } ( K , N ) \} ) \operatorname* { m a x } _ { \stackrel { S \subseteq \mathcal { V } } { | S | \leq k } } f ( S ) .
$$

Proof. Let $\begin{array} { r } { \mathrm { O P T } \triangleq \operatorname* { m a x } _ { S \subseteq \mathcal { V } , | S | \leq k } \mathrm { M E E } ( S ) } \end{array}$

First, since the Mahalanobis-Ensemble Energy objective $\mathrm { M E E } ( S ) = p _ { S } ^ { \top } K _ { S } ^ { - 1 } p _ { S }$ is a normalized, nonnegative, and monotone set function, we can apply the general greedy guarantee from Lemma A.1. This yields:

$$
\begin{array} { r } { \mathrm { M E E } ( S _ { k } ) \geq \left( 1 - e ^ { - \gamma \nu , k } \right) \cdot \mathrm { O P T } , } \end{array}
$$

where $\gamma _ { \nu , k }$ is the submodularity ratio of MEE on the ground set $\nu .$

Next, we lower bound the submodularity ratio $\gamma _ { \nu , k }$ . By Lemma A.3, the submodularity ratio satisfies:

$$
\gamma \nu , \boldsymbol { k } \geq \operatorname* { m i n } _ { \boldsymbol { L } \subseteq \nu , \boldsymbol { A } \subseteq \nu \setminus \boldsymbol { L } } \lambda _ { \operatorname* { m i n } } ( \boldsymbol { K } , | \boldsymbol { L } \cup \boldsymbol { A } | ) .
$$

Since $L \cup A \subseteq \mathcal { V }$ , we have $| L \cup A | \leq N .$ . By the interlacing theorem, every principal submatrix of K has a minimum eigenvalue at least $\lambda _ { \operatorname* { m i n } } ( { K } , N )$ . Hence:

$$
\gamma \nu , \mathrm { { } } k \geq \lambda _ { \operatorname* { m i n } } ( K , N ) .
$$

Substituting this spectral lower bound back into the greedy approximation guarantee gives:

$$
\mathrm { M E E } ( S _ { k } ) \geq \left( 1 - e ^ { - \lambda _ { \operatorname* { m i n } } ( K , N ) } \right) \cdot \mathrm { O P T } .
$$

This completes the proof.

Proof of Theorem 3.2. Let $\mathrm { M E E } ( S ) = p _ { S } ^ { \top } K _ { S } ^ { - 1 } p _ { S }$ and $c _ { \lambda } = 1 + \lambda H _ { 2 } ^ { T } ( p )$ . The objective can be written as $\begin{array} { r } { \mathrm { M E S } _ { \lambda } ( S ) = \frac { \mathrm { M E E } ( S ) } { c _ { \lambda } ^ { | S | } } } \end{array}$ . Define the greedy approximation factor as $\alpha _ { { \cal N } } = 1 - \exp \{ - \lambda _ { \operatorname* { m i n } } ( { \cal K } , { \cal N } ) \}$ .

By Lemma A.4, for every $k = 1 , \ldots , N$ , the greedy set $S _ { k }$ satisfies:

$$
\operatorname { M E E } ( S _ { k } ) \geq \alpha _ { N } \operatorname* { m a x } _ { S \subseteq \mathcal { V } \mid S \mid \leq k } \operatorname { M E E } ( S ) .\tag{10}
$$

Fix any $k \in \{ 1 , \ldots , N \}$ . Dividing both sides of (10) by $c _ { \lambda } ^ { k }$ gives:

$$
\begin{array} { r l } { \displaystyle \mathrm { M E S } _ { \lambda } ( S _ { k } ) = \frac { \mathrm { M E E } ( S _ { k } ) } { c _ { \lambda } ^ { k } } \geq \alpha _ { N } \frac { \operatorname* { m a x } _ { S \subseteq \mathcal { V } , | S | \leq k } \mathrm { M E E } ( S ) } { c _ { \lambda } ^ { k } } } \\ { \displaystyle \geq \alpha _ { N } \frac { \operatorname* { m a x } _ { S \subseteq \mathcal { V } , | S | = k } \mathrm { M E E } ( S ) } { c _ { \lambda } ^ { k } } } \\ { \displaystyle = \alpha _ { N } \underset { S \mid = k } { \operatorname* { m a x } } \frac { \mathrm { M E E } ( S ) } { c _ { \lambda } ^ { | S | } } = \alpha _ { N } \underset { | S | = k } { \operatorname* { m a x } } \mathrm { M E S } _ { \lambda } ( S ) . } \end{array}
$$

Taking the maximum over all subset sizes $k = 1 , \ldots , N$ on both sides, we obtain the global bound for the greedy trajectory:

$$
\operatorname* { m a x } _ { 1 \leq k \leq N } \mathrm { M E S } _ { \lambda } ( S _ { k } ) \geq \alpha _ { N } \operatorname* { m a x } _ { 1 \leq k \leq N } \operatorname* { m a x } _ { S \subseteq \mathcal { V } } \mathrm { M E S } _ { \lambda } ( S ) .\tag{11}
$$

By Theorem 3.1, the greedy saturation stopping point τ achieves the exact maximum of the entire greedy sequence, meaning $\mathrm { M E S } _ { \tau } ^ { g } = \mathrm { m a x } _ { 0 \leq r \leq N }$ MES<sup>g</sup>. Since $\mathrm { M E S } _ { r } ^ { g } = \mathrm { M E S } _ { \lambda } ( S _ { r } )$ , we have:

$$
\mathrm { M E S } _ { \tau } ^ { g } \geq \operatorname* { m a x } _ { 1 \leq k \leq N } \mathrm { M E S } _ { \lambda } ( S _ { k } ) .
$$

Combining this with (11) and substituting the definition of $\alpha _ { N }$ , we arrive at our final conclusion:

$$
\mathrm { M E S } _ { \tau } ^ { g } \geq ( 1 - \exp \{ - \lambda _ { \operatorname* { m i n } } ( K , N ) \} ) \operatorname* { m a x } _ { \emptyset \neq S \subseteq \mathcal { V } } \mathrm { M E S } _ { \lambda } ( S ) .
$$

This completes the proof.

## B Additional Experiments

This section provides repeated evaluations, controlled ablations, diagnostic analyses, and efficiency measurements. All experiments in this work were conducted on a single NVIDIA GeForce RTX 4090 GPU. Unless stated otherwise, repeated generation results use three shared seeds and report the mean ± sample standard deviation.

## B.1 Additional Results for Reasoning Benchmarks

Tables 6 and 7 repeat the reasoning evaluation over three seeds and report CraEG combined with p-less. This composition follows CraEG’s plug-in role as a post-softmax reweighting module rather than a standalone support-selection rule, while the tables use CraEG as the concise method name.

Across the nine model–temperature settings, ME-Decoding achieves the highest average accuracy on both GSM8K and GPQA. It leads in eight of the nine settings on each benchmark, supporting stable improvements across seeds and temperatures.

## B.2 Ablation Studies

Probability and Kernel Controls. We use Greedy decoding to test whether repeatedly selecting the locally most probable token is sufficient. At $T ~ = ~ 1 . 0$ , ME-Decoding improves GSM8K accuracy from 61.41% to $6 3 . 2 0 \pm 0 . 5 7 \%$ and GPQA accuracy from 32.37% to 32.66±1.05% on Qwen2.5-1.5B. Greedy is deterministic, whereas ME-Decoding results average three generation seeds.

We then compare the semantic kernel with two controlled alternatives. The identity kernel $K = I$ removes pairwise geometry, reducing the MEE numerator to $\textstyle \sum _ { i \in S } p _ { i } ^ { 2 }$ . The permuted kernel $K _ { \mathrm { p e r m } } = P K _ { \mathrm { t r u e } } P ^ { \intercal }$ preserves the eigenvalues, condition number, entry multiset, and numerical scale of the semantic kernel while disrupting its alignment with token identities. All variants use the same prompts, probabilities, generation seeds, and pruning hyperparameters.

<table><tr><td>Dataset</td><td>True K</td><td>Permuted K</td><td> $K = I$ </td></tr><tr><td>GSM8K</td><td>62.32</td><td>61.15</td><td>60.85</td></tr><tr><td>GPQA</td><td>32.64</td><td>32.09</td><td>29.81</td></tr></table>

Table 8: Accuracy (%) averaged over $T \in$ {1.0, 1.5, 2.0} in the Qwen2.5-1.5B kernel-control experiment. Both controls reduce accuracy relative to the correctly aligned semantic kernel.

The correctly aligned semantic kernel achieves the highest average accuracy on both datasets, while removing or permuting the geometry reduces performance.

Component Ablations. We examine the contribution of the adaptive bandwidth ϵ and the kernel matrix K. Removing the adaptive bandwidth degrades performance, especially at higher temperatures, which indicates that an appropriately scaled kernel is important for robust optimization. Replacing K with the identity matrix removes the embedding-based similarity structure and also reduces accuracy.

<table><tr><td rowspan="2">Method</td><td colspan="3">Qwen3-4B-Instruct</td><td colspan="3">Phi-4-mini-Instruct</td><td colspan="3">Mistral-7B-Instruct</td><td rowspan="2">| Avg.</td></tr><tr><td> $T = 1 . 0$ </td><td> $T = 1 . 5$ </td><td> $T = 2 . 0$ </td><td> $T = 1 . 0$ </td><td> $T = 1 . 5$ </td><td> $T = 2 . 0$ </td><td> $T = 1 . 0$ </td><td> $T = 1 . 5$ </td><td> $T = 2 . 0$ </td></tr><tr><td>p-less</td><td> $7 8 . 8 2 _ { \pm 0 . 2 4 }$ </td><td> $7 2 . 5 3 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $6 4 . 4 7 _ { \pm 1 . 0 0 }$ </td><td> $8 2 . 9 9 _ { \pm 0 . 7 6 }$ </td><td> $8 2 . 9 9 _ { \pm 0 . 3 9 }$ </td><td> $7 1 . 1 4 { \scriptstyle \pm 0 . 9 9 }$ </td><td> $5 3 . 5 3 { \scriptstyle \pm 0 . 7 3 }$ </td><td> $5 1 . 9 6 _ { \pm 0 . 7 4 }$ </td><td> $4 6 . 0 7 _ { \pm 1 . 3 0 }$ </td><td>|67.17</td></tr><tr><td>Top-W</td><td> $7 7 . 6 3 { \scriptstyle \pm 0 . 3 5 }$ </td><td> $7 6 . 6 5 { \scriptstyle \pm 0 . 2 3 }$ </td><td> $7 5 . 5 4 { \scriptstyle \pm 0 . 7 2 }$ </td><td> $8 3 . 2 4 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $8 3 . 5 2 _ { \pm 0 . 2 3 }$ </td><td> $8 2 . 9 9 { \scriptstyle \pm 0 . 1 8 }$ </td><td> $5 3 . 0 7 _ { \pm 0 . 4 0 }$ </td><td> $5 3 . 5 0 { \scriptstyle \pm 1 . 0 5 }$ </td><td> $5 2 . 4 4 { \scriptstyle \pm 1 . 2 6 }$ </td><td>70.95</td></tr><tr><td>CraEG</td><td> $7 2 . 2 3 { \scriptstyle \pm 1 . 7 1 }$ </td><td> $6 5 . 0 7 { \scriptstyle \pm 1 . 0 0 }$ </td><td> $5 8 . 1 2 { \scriptstyle \pm 1 . 3 2 }$ </td><td> $8 3 . 5 2 { \scriptstyle \pm 0 . 5 5 }$ </td><td> $8 1 . 9 1 { \scriptstyle \pm 1 . 0 7 }$ </td><td> $6 5 . 3 8 { \scriptstyle \pm 1 . 2 6 }$ </td><td> $5 2 . 7 2 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $5 0 . 8 2 { \scriptstyle \pm 0 . 5 7 }$ </td><td> $4 0 . 5 6 { \scriptstyle \pm 1 . 6 4 }$ </td><td>63.37</td></tr><tr><td>Ours</td><td> $\mathbf { 8 0 . 0 4 } _ { \pm 0 . 6 9 }$ </td><td> ${ \bf 7 9 . 5 6 _ { \pm 0 . 3 5 } }$ </td><td> $\mathbf { 8 0 . 0 4 } _ { \pm 0 . 8 0 }$ </td><td> $\mathbf { 8 3 . 9 0 _ { \pm 0 . 2 4 } }$ </td><td> ${ \bf 8 3 . 9 5 { \scriptstyle \pm 0 . 3 1 } }$ </td><td> $8 2 . 4 4 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $\mathbf { 5 4 . 1 8 _ { \pm 1 . 0 3 } }$ </td><td> ${ \bf 5 3 . 5 5 { \scriptstyle \pm 0 . 1 2 } }$ </td><td> ${ \bf 5 3 . 9 8 _ { \pm 0 . 5 7 } }$ </td><td>72.40</td></tr></table>

Table 6: GSM8K flexible-extract accuracy (%) over three generation seeds. Subscripts report sample standard deviations, and the Avg. column averages the nine model–temperature means.
<table><tr><td rowspan="2">Method</td><td colspan="3">Qwen3-4B-Instruct</td><td colspan="3">Phi-4-mini-Instruct</td><td colspan="3">Mistral-7B-Instruct</td><td rowspan="2">Avg.</td></tr><tr><td> $T = 1 . 0$ </td><td> $T = 1 . 5$ </td><td> $T = 2 . 0$ </td><td> $T = 1 . 0$ </td><td> $T = 1 . 5$ </td><td> $T = 2 . 0$ </td><td> $T = 1 . 0$ </td><td> $T = 1 . 5$ </td><td> $T = 2 . 0$ </td></tr><tr><td>p-less</td><td> $3 5 . 5 7 { \scriptstyle \pm 2 . 2 9 }$ </td><td> $3 5 . 8 6 { \scriptstyle \pm 1 . 5 7 }$ </td><td> $3 3 . 8 5 { \scriptstyle \pm 1 . 3 5 }$ </td><td> $3 2 . 8 1 { \scriptstyle \pm 1 . 7 7 }$ </td><td> $3 2 . 6 6 { \scriptstyle \pm 0 . 7 8 }$ </td><td> $3 0 . 0 6 { \scriptstyle \pm 2 . 7 9 }$ </td><td> $2 8 . 4 2 _ { \pm 1 . 4 4 }$ </td><td> $2 8 . 7 2 { \scriptstyle \pm 1 . 2 3 }$ </td><td> $2 7 . 6 8 { \scriptstyle \pm 1 . 3 9 }$ </td><td>31.74</td></tr><tr><td>Top-W</td><td> $3 5 . 4 9 { \scriptstyle \pm 0 . 4 5 }$ </td><td> $3 4 . 9 0 { \scriptstyle \pm 0 . 7 8 }$ </td><td> $3 5 . 8 6 { \scriptstyle \pm 0 . 2 6 }$ </td><td> $3 1 . 6 2 { \scriptstyle \pm 0 . 7 8 }$ </td><td> $3 0 . 5 8 { \scriptstyle \pm 0 . 7 7 }$ </td><td> $3 0 . 6 5 { \scriptstyle \pm 1 . 4 9 }$ </td><td> $2 8 . 7 9 { \scriptstyle \pm 1 . 0 2 }$ </td><td> $2 9 . 3 9 { \scriptstyle \pm 0 . 3 4 }$ </td><td> $2 9 . 3 9 { \scriptstyle \pm 1 . 4 9 }$ </td><td>31.85</td></tr><tr><td>CraEG</td><td> $3 5 . 7 1 { \scriptstyle \pm 1 . 2 4 }$ </td><td> $3 4 . 0 8 _ { \pm 2 . 1 7 }$ </td><td> $3 3 . 9 3 _ { \pm 0 . 9 7 }$ </td><td> $3 2 . 2 2 _ { \pm 1 . 4 2 }$ </td><td> $3 2 . 1 4 { \scriptstyle \pm 0 . 8 0 }$ </td><td> $2 9 . 5 4 { \scriptstyle \pm 0 . 6 8 }$ </td><td> $2 8 . 6 5 { \scriptstyle \pm 0 . 6 8 }$ </td><td> $2 8 . 7 2 _ { \pm 2 . 0 3 }$ </td><td> $2 8 . 2 0 { \scriptstyle \pm 1 . 2 3 }$ </td><td>31.46</td></tr><tr><td>Ours</td><td> $\mathbf { 3 6 . 6 8 _ { \pm 0 . 1 3 } }$ </td><td> $\mathbf { 3 6 . 7 6 _ { \pm 0 . 6 8 } }$ </td><td> $\mathbf { 3 6 . 6 1 _ { \pm 0 . 5 9 } }$ </td><td> $\mathbf { 3 4 . 3 0 _ { \pm 0 . 6 8 } }$ </td><td> $\mathbf { 3 4 . 4 5 _ { \pm 0 . 7 8 } }$ </td><td> $\mathbf { 3 4 . 0 8 _ { \pm 0 . 5 6 } }$ </td><td> $\mathbf { 2 9 . 0 9 _ { \pm 1 . 0 1 } }$  </td><td> $\mathbf { 2 9 . 8 4 _ { \pm 0 . 3 4 } }$ </td><td> $2 9 . 3 2 _ { \pm 0 . 4 6 }$ </td><td>33.46</td></tr></table>

Table 7: GPQA accuracy (%) over three generation seeds. Subscripts report sample standard deviations, and the Avg. column averages the nine model–temperature means.

<table><tr><td>Dataset</td><td>T</td><td>ME-Decoding</td><td>Without €</td><td> $K = I$ </td></tr><tr><td rowspan="3">GSM8K</td><td>1.0</td><td>84.08</td><td>-2.20</td><td rowspan="3">-0.15 -0.83 -1.36</td></tr><tr><td>1.5</td><td>84.23</td><td>-4.62</td></tr><tr><td>2.0</td><td>82.41</td><td>-8.19</td></tr><tr><td rowspan="3">GPQA</td><td>1.0</td><td>34.38</td><td>-0.22</td><td>-0.67</td></tr><tr><td>1.5</td><td>36.16</td><td>-1.56</td><td>-2.90</td></tr><tr><td>2.0</td><td>33.93</td><td>-0.22</td><td>-0.89</td></tr></table>

Table 9: Component ablations under different temperatures. The ME-Decoding column reports absolute accuracy, while the remaining columns report changes relative to ME-Decoding.

submatrices.

## B.4 Efficiency and Complexity

We benchmark 200 GSM8K problems, generating 64 tokens per problem with $T = 1 . 5 .$ , batch size 1, and $N = 5 1 2 .$ . Each method therefore generates 12,800 tokens. ME-Decoding uses a custom Triton kernel, and all methods use identical filtering and sampling infrastructure.

## B.3 Empirical Validation of Theorem 3.1

The condition in Theorem 3.1 is sufficient rather than necessary for trajectory unimodality. To examine whether the predicted behavior occurs in practice, we disable early stopping and record the complete greedy MES trajectory for a 512-trajectory probe from each benchmark using Qwen2.5-1.5B at $T = 1 . 0 .$

<table><tr><td>Dataset</td><td>Unimodal Trajectories</td><td>Rate</td></tr><tr><td>GSM8K</td><td>512/512</td><td>100%</td></tr><tr><td>GPQA</td><td>512/512</td><td>100%</td></tr></table>

Table 10: Empirical unimodality of complete greedy MES trajectories.

The observation supports the practical relevance of the stopping behavior without replacing the theorem’s sufficient-condition analysis. The approximation result has a separate scope and depends on restricted minimum eigenvalues of selected kernel

<table><tr><td>Model</td><td>Method</td><td>Model Inference (s)</td><td>Decoding and Sampling (s)</td><td>Other (s)</td><td>Total (s)</td><td>ms/token</td></tr><tr><td rowspan="5">Qwen2.5-1.5B</td><td>Top-p</td><td>233.20</td><td>4.29</td><td>0.99</td><td>238.48</td><td>18.63</td></tr><tr><td>Min-p</td><td>233.19</td><td>2.17</td><td>0.57</td><td>235.93</td><td>18.43</td></tr><tr><td>p-less</td><td>233.23</td><td>2.17</td><td>0.63</td><td>236.03</td><td>18.44</td></tr><tr><td> $\mathrm { T o p } { \cdot } H$ </td><td>233.56</td><td>4.68</td><td>1.00</td><td>239.25</td><td>18.69</td></tr><tr><td>ME-Decoding</td><td>242.85</td><td>5.64</td><td>1.03</td><td>249.52</td><td>19.49</td></tr><tr><td rowspan="5">Qwen3-4B</td><td>Top-p</td><td>358.20</td><td>2.89</td><td>0.50</td><td>361.58</td><td>28.25</td></tr><tr><td>Min-p</td><td>356.42</td><td>1.97</td><td>0.55</td><td>358.94</td><td>28.04</td></tr><tr><td>p-less</td><td>356.63</td><td>1.97</td><td>0.59</td><td>359.18</td><td>28.06</td></tr><tr><td>Top-H</td><td>359.14</td><td>2.91</td><td>0.63</td><td>362.68</td><td>28.33</td></tr><tr><td>ME-Decoding</td><td>360.29</td><td>3.01</td><td>1.15</td><td>364.45</td><td>28.47</td></tr></table>

Table 11: End-to-end GPU timing breakdown.

Table 11 shows that ME-Decoding increases total latency by approximately 5.8% on Qwen2.5- 1.5B and 1.5% on Qwen3-4B relative to the fastest baseline, with model inference remaining the dominant cost.

Detailed Complexity Analysis. Let N denote the size of the candidate token pool, d denote the token embedding dimension, and τ denote the number of tokens selected before the early stopping criterion is triggered. ME-Decoding does not explicitly construct the full $N \times N$ similarity matrix. The adaptive bandwidth can also be computed without any pairwise construction. Since we use normalized token embeddings and $C _ { i j } = 1 - e _ { i } ^ { \top } e _ { j }$ , we have

$$
\epsilon = \frac { 1 } { 2 } \sum _ { i , j \in \mathcal { V } } p _ { i } p _ { j } C _ { i j } = \frac { 1 } { 2 } \left( 1 - \left\| \sum _ { i \in \mathcal { V } } p _ { i } e _ { i } \right\| _ { 2 } ^ { 2 } \right) ,
$$

where p is normalized over the candidate pool. Therefore, computing ϵ only requires a probabilityweighted average of token embeddings, with complexity O(N d).

After obtaining ϵ, kernel entries are dynamically computed according to Eq. 3. Whenever a new token is selected, the algorithm computes and caches its similarities to all candidate tokens, which costs $O ( N d )$ per selected token and therefore $O ( N \tau d )$ in total. For the greedy selection stage, at the t-th step, the selected set has size t, and the algorithm evaluates at most $N - t$ remaining candidates. For each candidate j, the dominant operation is computing $\pmb { \alpha } _ { j } = \mathbf { R } \beta _ { j }$ , where $\mathbf { R } \in \mathbb { R } ^ { t \times t }$ and $\beta _ { j } \in \mathbb { R } ^ { t }$ which costs $O ( t ^ { 2 } )$ . Hence, the total greedy evaluation cost before stopping is

$$
\sum _ { t = 1 } ^ { \tau - 1 } { O } \big ( ( N - t ) t ^ { 2 } \big ) \leq \sum _ { t = 1 } ^ { \tau - 1 } { O } ( N t ^ { 2 } ) = O ( N \tau ^ { 3 } ) .
$$

After each selected token, updating R and z through the block update costs $O ( t ^ { 2 } )$ at step t, giving an additional $O ( \tau ^ { 3 } )$ cost, which is dominated by $O ( N \tau ^ { 3 } )$ since $\tau \leq N$ . Therefore, the overall time complexity of ME-Decoding is

$$
O ( N d + N \tau d + N \tau ^ { 3 } ) = O \bigl ( N \tau ( d + \tau ^ { 2 } ) \bigr ) .
$$

The memory overhead is $O ( N \tau )$ for cached kernel entries and $O ( \tau ^ { 2 } )$ for maintaining R. In contrast, explicitly constructing the full kernel matrix would require $O ( N ^ { 2 } d )$ time and $O ( N ^ { 2 } )$ memory. Since the early stopping rule usually yields $\tau \ll N$ , ME-Decoding avoids the quadratic dependence on the candidate pool size and remains efficient in practice.

## B.5 Hyperparameter Analysis

We study the sensitivity of ME-Decoding to the hyperparameter λ, which controls the compactness of the selected token set. A larger λ imposes a stronger penalty on the subset size and therefore encourages a tighter candidate set, while a smaller λ allows more tokens to be retained. Table 12 reports the accuracy of ME-Decoding under different values of λ across three models, three temperatures, and two reasoning datasets. The results show that ME-Decoding is relatively stable over a range of λ values and consistently achieves strong performance under different settings. Among the tested values, $\lambda = 0 . 9$ obtains the best average accuracy on both GSM8K and GPQA. Therefore, unless otherwise specified, we use $\lambda = 0 . 9$ as the default setting in the main experiments.

## B.6 Diversity Analysis

Support-Level Diversity Diagnostics. Table 13 shows that ME-Decoding achieves the highest accuracy and MES on both datasets, while retaining a compact support and comparable probability mass. None of the approximately matched probabilityonly controls reproduces both results, indicating that the gain is not explained solely by stronger truncation or one support statistic.

<table><tr><td rowspan="2">λ</td><td colspan="3">Qwen3-4B-Inst.</td><td colspan="3">Phi-4-mini-Inst.</td><td colspan="3">Mistral-7B-Inst.</td><td rowspan="2"> $\operatorname { A v g } .$ </td></tr><tr><td>T = 1.0</td><td>T = 1.5</td><td>T = 2.0</td><td>T = 1.0</td><td>T = 1.5</td><td>T = 2.0</td><td>T = 1.0</td><td>T = 1.5</td><td>T = 2.0</td></tr><tr><td colspan="10">GSM8K</td></tr><tr><td>1.0</td><td>81.20</td><td>81.35</td><td>79.76</td><td>83.78</td><td>84.38</td><td>82.03</td><td>52.84</td><td>52.77</td><td>53.45</td><td>72.39</td></tr><tr><td>0.9</td><td>80.67</td><td>79.76</td><td>80.06</td><td>84.08</td><td>84.23</td><td>82.41</td><td>55.34</td><td>53.45</td><td>53.90</td><td>72.66</td></tr><tr><td>0.8</td><td>80.44</td><td>80.06</td><td>80.44</td><td>83.70</td><td>84.08</td><td>81.65</td><td>55.19</td><td>53.45</td><td>53.53</td><td>72.50</td></tr><tr><td>0.7</td><td>79.68</td><td>80.14</td><td>79.98</td><td>84.15</td><td>83.55</td><td>82.11</td><td>53.45</td><td>54.51</td><td>53.75</td><td>72.37</td></tr><tr><td colspan="10">GPQA</td></tr><tr><td>1.0</td><td>34.82</td><td>34.15</td><td>34.82</td><td>35.04</td><td>35.27</td><td>35.71</td><td>27.90</td><td>28.79</td><td>27.23</td><td>32.64</td></tr><tr><td>0.9</td><td>35.49</td><td>35.49</td><td>35.71</td><td>34.38</td><td>36.16</td><td>33.93</td><td>28.35</td><td>28.79</td><td>28.35</td><td>32.96</td></tr><tr><td>0.8</td><td>36.16</td><td>36.38</td><td>35.27</td><td>35.27</td><td>33.26</td><td>34.15</td><td>27.90</td><td>25.67</td><td>27.68</td><td>32.42</td></tr><tr><td>0.7</td><td>36.38</td><td>35.04</td><td>34.60</td><td>33.71</td><td>31.92</td><td>33.93</td><td>27.90</td><td>29.24</td><td>26.34</td><td>32.12</td></tr></table>

Table 12: Detailed hyperparameter selection results for ME-Decoding. Accuracy is reported in percentage. The highlighted row corresponds to the selected setting, λ=0.9, which achieves the best average performance on both GSM8K and GPQA.

Detailed Diversity Results. Table 14 reports the detailed numerical results corresponding to the diversity analysis in the main text. For each temperature, we report Acc, Distinct-1, Distinct-2, 1 − Self-BLEU, and the aggregated Diversity Avg. score. The diversity score is computed by minmax normalizing the three diversity metrics within each temperature and then taking their average. Although Top-p often obtains the highest diversity score, especially at high temperature, its accuracy drops substantially. In contrast, ME-Decoding maintains the strongest accuracy across temperatures, showing that it provides a better accuracy– diversity trade-off for reasoning-oriented generation.

## B.7 Additional Results for Open-Ended Generation

In the main text, we report the averaged instruction-following and chat performance over three instruction-tuned models. Here, we provide the corresponding per-model visualizations and detailed numerical results. Figure 5 shows the performance curves on MT-Bench and AlpacaEval across temperatures for each model. Figure 6 further summarizes the relative ranking of each decoding method under every model-temperature setting, averaged over MT-Bench and AlpacaEval. Lower ranks indicate better performance. The heatmap shows that ME-Decoding obtains competitive ranks in multiple settings and achieves the best overall average rank, suggesting that its improvement is stable across models and temperatures. Tables 17 and 18 report the exact MT-Bench judge scores and AlpacaEval candidate win-rates, respectively. ME-Decoding achieves the highest overall averages, with 7.17 on MT-Bench and 14.80% on AlpacaEval, although individual baselines lead in several settings. All results are evaluated using the same judge-based evaluation protocol as in the main text and aggregated over 3 runs.

Evaluation Robustness. AlpacaEval-style evaluation compares each generated response with a fixed reference for the same prompt, while MT-Bench-style evaluation independently assigns a score from 1 to 10. All methods use matched samples at $T \in \{ 1 . 0 , 1 . 5 , 2 . 0 \}$ over three generation runs. We evaluate the same outputs with DeepSeek-V4-Pro and GLM-5.2 (Zeng et al., 2026). Table 15 shows that ME-Decoding achieves the highest result under both judges and evaluation protocols. Table 16 further reports positive paired 95% confidence intervals against Top-W, p-less, and Top-H, supporting the consistency of the gains beyond aggregate means. Additional length, style, and refusal diagnostics indicate that the gains are not explained by verbosity or conservative refusal behavior.

<table><tr><td>Dataset</td><td>Method</td><td>Acc. (%)</td><td>Mass</td><td>Cos.</td><td>Avg. |S|</td><td> $H _ { \mathrm { b e f o r e } }$ </td><td> $H _ { \mathrm { a f t e r } }$ </td><td>MES</td></tr><tr><td rowspan="7"></td><td>ME-Decoding</td><td> ${ \bf 6 3 . 2 0 \pm 0 . 5 7 }$ </td><td>0.846</td><td>0.151</td><td>1.076</td><td>0.614</td><td>0.052</td><td>0.756</td></tr><tr><td>Top-W</td><td> $6 1 . 6 4 \pm 1 . 1 9$ </td><td>0.841</td><td>0.208</td><td>1.093</td><td>0.692</td><td>0.057</td><td>0.688</td></tr><tr><td>p-less</td><td> $6 1 . 3 3 \pm 0 . 2 3$ </td><td>0.842</td><td>0.179</td><td>1.164</td><td>0.659</td><td>0.075</td><td>0.689</td></tr><tr><td>GSM8K Top-H</td><td> $6 1 . 1 6 \pm 0 . 4 2$ </td><td>0.847</td><td>0.241</td><td>2.240</td><td>0.736</td><td>0.191</td><td>0.660</td></tr><tr><td>Approx. size-matched</td><td> $6 1 . 7 9 \pm 1 . 2 6$ </td><td>0.850</td><td>0.210</td><td>1.067</td><td>0.615</td><td>0.044</td><td>0.707</td></tr><tr><td>Approx. entropy-matched</td><td> $6 1 . 7 9 \pm 0 . 9 9$ </td><td>0.851</td><td>0.208</td><td>1.070</td><td>0.614</td><td>0.045</td><td>0.708</td></tr><tr><td>Approx. cosine-matched</td><td> $6 2 . 7 0 \pm 0 . 3 5$ </td><td>0.840</td><td>0.198</td><td>1.014</td><td>0.598</td><td>0.010</td><td>0.715</td></tr><tr><td rowspan="7">GPQA</td><td>ME-Decoding</td><td> $\mathbf { 3 2 . 6 6 \pm 1 . 0 5 }$ </td><td>0.759</td><td>0.243</td><td>1.134</td><td>0.983</td><td>0.091</td><td>0.589</td></tr><tr><td>Top-W</td><td> $2 7 . 6 9 \pm 0 . 9 1$ </td><td>0.692</td><td>0.185</td><td>1.213</td><td>1.454</td><td>0.124</td><td>0.498</td></tr><tr><td>p-less</td><td> $2 8 . 2 1 \pm 1 . 0 5$ </td><td>0.770</td><td>0.192</td><td>1.265</td><td>1.002</td><td>0.144</td><td>0.553</td></tr><tr><td>Top-H</td><td> $2 7 . 5 2 \pm 0 . 6 6$ </td><td>0.778</td><td>0.184</td><td>2.877</td><td>1.277</td><td>0.431</td><td>0.475</td></tr><tr><td>Approx. size-matched</td><td> $2 8 . 4 7 \pm 1 . 1 7$ </td><td></td><td>0.7600.289</td><td>1.106</td><td>0.947</td><td>0.069</td><td>0.558</td></tr><tr><td>Approx. entropy-matched</td><td> $2 8 . 4 7 \pm 1 . 1 7$ </td><td></td><td>0.7600.289</td><td>1.106</td><td>0.947</td><td>0.069</td><td>0.558</td></tr><tr><td>Approx. cosine-matched</td><td> $2 9 . 2 5 \pm 0 . 9 1$ </td><td>0.737</td><td>0.259</td><td>1.084</td><td>1.018</td><td>0.055</td><td>0.537</td></tr></table>

Table 13: Complete support diagnostics on Qwen2.5-1.5B at $T = 1 . 0 .$ Cosine similarity is computed within the selected support at each decoding step and then averaged across steps. The approximately matched Min-p controls are calibrated on independent prompts. Accuracy reports the mean ± sample standard deviation over three seeds.

Table 14: Accuracy and diversity comparison under different temperatures. Diversity Avg. is computed by first min-max normalizing Distinct-1, Distinct-2, and $1 - \mathrm { S e l f - B L E U }$ within each temperature, and then averaging the three normalized scores.
<table><tr><td>T</td><td>Method</td><td>Acc ↑</td><td>Distinct-1 ↑</td><td>Distinct-2 ↑</td><td>1 - Self-BLEU ↑</td><td>Diversity Avg. ↑</td></tr><tr><td rowspan="5">1.0</td><td>Top-p</td><td>0.8484</td><td>0.0050</td><td>0.0695</td><td>0.2570</td><td>0.9842</td></tr><tr><td>Min-p</td><td>0.8596</td><td>0.0050</td><td>0.0705</td><td>0.2595</td><td>1.0000</td></tr><tr><td>p-less</td><td>0.8906</td><td>0.0046</td><td>0.0445</td><td>0.0607</td><td>0.1012</td></tr><tr><td>Top-H</td><td>0.8832</td><td>0.0045</td><td>0.0418</td><td>0.0614</td><td>0.0043</td></tr><tr><td>Top-W</td><td>0.8996</td><td>0.0049</td><td>0.0480</td><td>0.0686</td><td>0.3550</td></tr><tr><td rowspan="4"></td><td>ME-Decoding (λ = 0.9)</td><td>0.9030</td><td>0.0049</td><td>0.0472</td><td>0.0588</td><td>0.3294</td></tr><tr><td>ME-Decoding (λ = 0.8)</td><td>0.9008</td><td>0.0049</td><td>0.0504</td><td>0.0813</td><td>0.4039</td></tr><tr><td>ME-Decoding (λ = 0.7)</td><td>0.9024</td><td>0.0049</td><td>0.0528</td><td>0.1014</td><td>0.4652</td></tr><tr><td>Top-p</td><td>0.6794</td><td>0.0224</td><td>0.1503</td><td>0.4136</td><td>1.0000</td></tr><tr><td rowspan="5">1.5</td><td>Min-p</td><td>0.7802</td><td>0.0054</td><td>0.0849</td><td>0.3421</td><td>0.3884</td></tr><tr><td>p-less</td><td>0.8682</td><td>0.0046</td><td>0.0546</td><td>0.1464</td><td>0.0783</td></tr><tr><td>Top-H</td><td>0.8224</td><td>0.0047</td><td>0.0585</td><td>0.1974</td><td>0.1434</td></tr><tr><td>Top-W</td><td>0.8982</td><td>0.0049</td><td>0.0542</td><td>0.1148</td><td>0.0516</td></tr><tr><td>ME-Decoding (λ = 0.9)</td><td>0.9026</td><td>0.0052</td><td>0.0526</td><td>0.0735</td><td>0.0112</td></tr><tr><td></td><td>ME-Decoding (λ = 0.8)</td><td>0.9058</td><td>0.0051</td><td>0.0564</td><td>0.1046</td><td>0.0528</td></tr><tr><td rowspan="4">Top-p</td><td>ME-Decoding (λ = 0.7)</td><td>0.8996</td><td>0.0053</td><td>0.0592</td><td>0.1268</td><td>0.0879</td></tr><tr><td></td><td>0.1886</td><td>0.2494</td><td></td><td></td><td></td></tr><tr><td>Min-p</td><td>0.6600</td><td>0.0068</td><td>0.7750 0.1122</td><td>0.9054 0.4487</td><td>1.0000</td></tr><tr><td>p-less</td><td>0.8128</td><td>0.0049</td><td>0.0659</td><td>0.2454</td><td>0.1833 0.0786</td></tr><tr><td rowspan="5">2.0</td><td>Top-H</td><td>0.7394</td><td>0.0053</td><td>0.0784</td><td>0.3209</td><td>0.1149</td></tr><tr><td>Top-W</td><td>0.8906</td><td>0.0047</td><td>0.0543</td><td>0.1397</td><td>0.0310</td></tr><tr><td>ME-Decoding (λ = 0.9)</td><td>0.9068</td><td>0.0050</td><td>0.0499</td><td>0.0668</td><td>0.0004</td></tr><tr><td>ME-Decoding (λ = 0.8)</td><td>0.9036</td><td>0.0051</td><td>0.0553</td><td>0.0959</td><td>0.0146</td></tr><tr><td>ME-Decoding (λ = 0.7)</td><td>0.9034</td><td>0.0052</td><td>0.0573</td><td>0.1187</td><td>0.0247</td></tr></table>

<table><tr><td>Judge</td><td>Method</td><td>AlpacaEval (%)</td><td>MT-Bench</td></tr><tr><td rowspan="6">DeepSeek-V4-Pro</td><td>Top-p</td><td> $1 3 . 0 4 \pm 2 . 0 7$ </td><td> $6 . 3 4 \pm 0 . 8 9$ </td></tr><tr><td>Min-p</td><td> $1 4 . 1 4 \pm 0 . 6 0$ </td><td> $6 . 8 6 \pm 0 . 4 1$ </td></tr><tr><td>Top-H</td><td> $1 4 . 3 6 \pm 0 . 5 8$ </td><td> $6 . 8 8 \pm 0 . 3 3$ </td></tr><tr><td> $\scriptstyle p - \mathrm { l e s s }$ </td><td> $1 4 . 3 3 \pm 0 . 0 7$ </td><td> $6 . 9 4 \pm 0 . 1 1$ </td></tr><tr><td> $\mathsf { \bar { T o p } } \mathrm { - } W$ </td><td> $1 4 . 1 4 \pm 0 . 0 4$ </td><td> $7 . 0 8 \pm 0 . 0 2$ </td></tr><tr><td>MÉ-Decoding</td><td> ${ \bf 1 4 . 8 0 \pm 0 . 3 0 }$ </td><td> ${ \bf 7 . 1 7 \pm 0 . 0 3 }$ </td></tr><tr><td rowspan="4">GLM-5.2</td><td>Top-H</td><td> $9 . 4 8 \pm 0 . 7 8$ </td><td> $5 . 4 2 \pm 0 . 2 3$ </td></tr><tr><td>p-less</td><td> $1 0 . 1 8 \pm 0 . 2 4$ </td><td> $5 . 5 4 \pm 0 . 0 5$ </td></tr><tr><td> $\mathsf { \bar { T o p } } \mathrm { - } W$ </td><td> $1 0 . 1 1 \pm 0 . 2 0$ </td><td> $5 . 5 8 \pm 0 . 0 3$ </td></tr><tr><td>ME-Decoding</td><td> ${ \bf 1 0 . 6 3 \pm 0 . 1 6 }$ </td><td> ${ \bf 5 . 6 4 \pm 0 . 0 1 }$ </td></tr></table>

Table 15: Open-ended evaluation under two automatic judges. Results average temperature-level aggregates over three generation runs.

<table><tr><td>Judge</td><td>Baseline</td><td>∆ AlpacaEval (pp)</td><td>95% CI</td><td>∆ MT-Bench</td><td>95% CI</td></tr><tr><td rowspan="3">DeepSeek-V4-Pro</td><td>Top-W</td><td>+0.656</td><td>[0.320, 0.980]</td><td>+0.092</td><td>[0.018,0.169]</td></tr><tr><td>p-less</td><td>+0.469</td><td>[0.115,0.810]</td><td>+0.234</td><td>[0.158, 0.305]</td></tr><tr><td>Top-H</td><td>+0.440</td><td>[0.076, 0.799]</td><td>+0.297</td><td>[0.221, 0.373]</td></tr><tr><td rowspan="3">GLM-5.2</td><td>Top-W</td><td>+0.529</td><td>[0.239,0.819]</td><td>+0.069</td><td>[0.018,0.120]</td></tr><tr><td>p-less</td><td>+0.456</td><td>[0.138, 0.782]</td><td>+0.107</td><td>[0.052, 0.163]</td></tr><tr><td>Top-H</td><td>+1.157</td><td>[0.824, 1.481]</td><td>+0.221</td><td>[0.162, 0.281]</td></tr></table>

Table 16: Paired-bootstrap gains of ME-Decoding. Resampling preserves prompt, model, temperature, and generation-run matching.

![](images/f793cde086bdd637818fec28fb2a313de253032df91a331bb46bd97c2551814c.jpg)

![](images/9d2eaba91a11abd069bbcd99537a77928c1ec6434805cdc9c0b026d11e137642.jpg)

![](images/e1987e84df8ca1aec2b5d70eaf913b113704e61b256c8c20fe52abde8987faa6.jpg)

Top-p Min-p Top-H p-less Top-W ME-Decoding  
![](images/873abf321271a540c350cbc45b726908b333523a743674bfd3ed329daaaf27e2.jpg)

![](images/18b477cb894dc3904d5cd82acf9fff6038e0550cea8e25d1b29d9183ea73f565.jpg)

![](images/f5aa7b755a3d6d8dfeca97305591d53a9df55bd4fe57af846bc298daef228eb9.jpg)  
Figure 5: Detailed instruction-following and chat performance across temperatures. Top: MT-Bench judge scores. Bottom: AlpacaEval candidate win-rate. Results are reported for three instruction-tuned models and aggregated over 3 runs.

<table><tr><td></td><td colspan="3">Qwen3-4B-Inst.</td><td colspan="3">Phi-4-mini-Inst.</td><td colspan="3">Mistral-7B-Inst.</td><td></td></tr><tr><td>Method</td><td> $T = 1 . 0$ </td><td> $T = 1 . 5$ </td><td> $T = 2 . 0$ </td><td> $T = 1 . 0$ </td><td> $T = 1 . 5$ </td><td> $T = 2 . 0$ </td><td> $T = 1 . 0$ </td><td> $T = 1 . 5$ </td><td> $T = 2 . 0$ </td><td>Avg.</td></tr><tr><td>Min-p</td><td>8.71</td><td>8.73</td><td>8.51</td><td>6.19</td><td>5.63</td><td>4.61</td><td>6.69</td><td>6.55</td><td>6.10</td><td>6.86</td></tr><tr><td>Top-p</td><td>8.76</td><td>8.69</td><td>8.47</td><td>5.79</td><td>4.85</td><td>2.37</td><td>6.63</td><td>6.27</td><td>5.19</td><td>6.34</td></tr><tr><td>p-less</td><td>8.70</td><td>8.62</td><td>8.75</td><td>5.91</td><td>5.91</td><td>5.15</td><td>6.42</td><td>6.47</td><td>6.53</td><td>6.94</td></tr><tr><td>Top-H</td><td>8.70</td><td>8.65</td><td>8.62</td><td>5.97</td><td>5.79</td><td>4.63</td><td>6.59</td><td>6.70</td><td>6.25</td><td>6.88</td></tr><tr><td>Top-W</td><td>8.62</td><td>8.72</td><td>8.64</td><td>6.12</td><td>5.96</td><td>6.16</td><td>6.44</td><td>6.55</td><td>6.51</td><td>7.08</td></tr><tr><td>Ours</td><td>8.84</td><td>8.76</td><td>8.87</td><td>5.93</td><td>5.94</td><td>6.05</td><td>6.78</td><td>6.72</td><td>6.68</td><td>7.17</td></tr></table>

Table 17: Detailed MT-Bench judge scores. Results are reported across three instruction-tuned models and three temperatures, aggregated over 3 runs. The Avg. column gives the unweighted average over all model-temperature combinations. Best results are shown in bold, and second-best results are underlined.

<table><tr><td></td><td colspan="3">Qwen3-4B-Inst.</td><td colspan="3">Phi-4-mini-Inst.</td><td colspan="3">Mistral-7B-Inst.</td><td></td></tr><tr><td>Method</td><td> $T = 1 . 0$ </td><td> $T = 1 . 5$ </td><td> $T = 2 . 0$ </td><td> $T = 1 . 0$ </td><td> $T = 1 . 5$ </td><td> $T = 2 . 0$ </td><td> $T = 1 . 0$ </td><td> $T = 1 . 5$ </td><td> $T = 2 . 0$ </td><td>Avg.</td></tr><tr><td>Min-p</td><td>31.72</td><td>31.21</td><td>29.80</td><td>2.90</td><td>2.69</td><td>1.82</td><td>9.17</td><td>9.13</td><td>8.78</td><td>14.14</td></tr><tr><td>Top-p</td><td>32.08</td><td>31.28</td><td>27.00</td><td>2.73</td><td>2.40</td><td>0.17</td><td>8.43</td><td>8.49</td><td>4.82</td><td>13.04</td></tr><tr><td>p-less</td><td>31.93</td><td>31.68</td><td>32.09</td><td>2.73</td><td>2.94</td><td>2.65</td><td>8.30</td><td>8.59</td><td>8.05</td><td>14.33</td></tr><tr><td>Top-H</td><td>32.85</td><td>32.23</td><td>30.45</td><td>2.73</td><td>3.02</td><td>2.15</td><td>8.49</td><td>8.84</td><td>8.47</td><td>14.36</td></tr><tr><td>Top-W</td><td>32.26</td><td>32.05</td><td>31.76</td><td>2.61</td><td>2.57</td><td>2.32</td><td>7.56</td><td>7.95</td><td>8.22</td><td>14.14</td></tr><tr><td>Ours</td><td>34.00</td><td>31.93</td><td>33.42</td><td>2.90</td><td>3.31</td><td>2.65</td><td>8.36</td><td>8.20</td><td>8.43</td><td>14.80</td></tr></table>

Table 18: Detailed AlpacaEval candidate win-rate (%). Results are reported across three instruction-tuned models and three temperatures, aggregated over 3 runs. The Avg. column gives the unweighted average over all model temperature combinations. Best results are shown in bold, and second-best results are underlined.

Rank by Model and Temperature  
![](images/2e70900a9ba0bd409b2d4dfe831373a8b04eeba6560c2104fed8005ee7a1a15c.jpg)  
Figure 6: Detailed average-rank comparison on open-ended generation benchmarks. The heatmap reports the rank of each decoding method under each model-temperature setting, averaged over AlpacaEval and MT-Bench.