# Collapsibility of Performance Metrics in Clinical Predictive AI

João Matos University of Oxford

Ben Van Calster KU Leuven

Richard D. Riley University of Birmingham

Paula Dhiman University of Oxford

Gary S. Collins University of Birmingham

## Abstract

Background: Population level assessments of predictive artificial intelligence (AI) can conceal performance disparities across subgroups. Fairness evaluations commonly rely on performance analyses across subgroups. However, some performance metrics are non-collapsible, meaning that the overall population performance value does not equal the weighted average of subgroup specific values. This can mislead fairness evaluation.

Objective: To examine the collapsibility properties of commonly reported performance metrics in predictive AI, with a focus on the area under the receiver operating characteristic curve (AUC, also known as c-statistic).

Methods: We investigate the collapsibility of fifteen performance metrics, either by expressing each metric as a linear combination of its stratum specific values or, where non-collapsible, by providing a counterexample inspired by Simpson’s paradox as a formal disproof. We focus particularly on the AUC, a commonly reported metric of model discrimination, and formally characterise its expression under mixtures of subpopulations.

Results: Five performance metrics (AUC, calibration intercept, calibration slope, expected calibration error, and Nagelkerke $R ^ { 2 } )$ are shown to be non-collapsible, and ten (O:E ratio, logloss, Brier score, accuracy, F1-score, true positive rate, true negative rate, positive predictive value, negative predictive value, and net benefit) are shown to be collapsible. The AUC is shown to be non-collapsible because it decomposes into within- and cross-group AUC terms when subpopulations coexist, such that its overall value may fall outside the range of subgroup specific AUCs.

Conclusions: Non-collapsibility of performance metrics has important consequences for reporting, model appraisal, and fairness evaluation. It can generate spurious differences between subgroup and overall performance, which may mislead fairness evaluations. Explicitly acknowledging and reporting the collapsibility properties of performance metrics improves both the interpretability and transparency of fairness assessments.

## Box 1: Glossary of key terms used throughout the manuscript

<table><tr><td>AUC: area under the receiver operating characteristic curve, also known as c-statistic [1]. Calibration: the extent of the agreement between the probability estimates and the observed event</td></tr><tr><td>proportions [2]. Collapsibility: a metric is said to be collapsible over a stratifying variable  $g \left( \mathrm { e . g . } \right.$  sex) if its marginal value</td></tr><tr><td>equals the weighted average of the stratum specific values (e.g., male, female), with weights depending on the marginal distribution of g [3].</td></tr><tr><td>Cross-group discrimination:  $\mathrm { \ x A U C _ { i , j } , }$  probability that an individual with the event from subgroup ¿ is ranked higher than an individual without the event from subgroup j [4]. Discrimination: extent to which the model gives higher probability estimates for individuals with the</td></tr><tr><td>event compared to those without the event, quantified by the area under the receiver operating characteristic curve (AUC) [5].</td></tr><tr><td>Fairness evaluation: assessing the nature and magnitude of any potential differential model performance across individuals or subgroups defined by a sensitive attribute. It can be conducted in different ways, including disaggregated performance evaluation; stratification of visual tools such as calibration plots; computation of fairness metrics.</td></tr><tr><td>Fairness metric: a metric that quantifies the extent to which a model&#x27;s output does not discriminate (in the societal sense, based on a given notion of fairness) against individuals or subgroups defined by a sensitive attribute [6].</td></tr><tr><td>Jensen&#x27;s Gap: difference between the two sides of Jensen&#x27;s inequality, i.e., the deviation of the overall metric from the linear interpolation of subgroup values [7].</td></tr><tr><td>Outcome prevalence (π): proportion of individuals experiencing an outcome of interest.</td></tr><tr><td>Simpson&#x27;s Paradox: phenomenon in which a trend observed within multiple subgroups reverses or disappears when subgroups are combined [8].</td></tr><tr><td>Subgroup proportion  $( p ^ { g } ) \colon$  proportion of individuals belonging to subgroup g within the population. Within-group discrimination  $\mathrm { ( A U C _ { g } ) } { \ : }$  probability that an individual with the event from subgroup g is</td></tr></table>

## 1 Introduction

Evaluating the performance of a clinical prediction model at the population level can conceal important differences in model performance across subgroups. To address this, fairness assessments often report performance within relevant subgroups defined by a sensitive attribute, such as sex or ethnicity, or compute explicit fairness metrics [6, 9]. Even so, interpreting subgroup specific results $( \mathrm { e . g . }$ , male vs female, or across ethnicities) is not always straightforward. For instance, the area under the receiver operating characteristic curve (AUC, also known as c-statistic) may vary across subgroups, with a model having an overall AUC of 0.85 but subgroup AUCs of 0.75 and 0.80 in male and female subgroups, respectively.

Such patterns reflect a key mathematical property of the metric itself: collapsibility. A metric is collapsible over a variable g (e.g., sex) if its marginal value equals the weighted average of the subgroup specific values, with weights determined by the marginal distribution of g (e.g., male, female). When collapsibility holds, this aggregation artefact cannot occur. The phenomenon is closely related to, though not identical to Simpson’s paradox [8, 10], whereby associations observed within subgroups disappear or reverse when data is aggregated [11]. In extreme cases, marginal and conditional association metrics may have opposite directions, a pattern also referred to as the reversal paradox [12], aggregation bias, or the amalgamation paradox [13].

In the context of performance metrics, collapsibility means that the overall value can be expressed as an appropriate weighted average of the subgroup specific values. If a metric is collapsible, the overall value $M _ { \mathrm { o v e r a l l } }$ is therefore directly linked to the subgroup values $M _ { g }$ . By contrast, when a metric is non-collapsible, this relationship breaks down, making it more difficult to interpret subgroup and overall performance jointly, and potentially distorting fairness evaluations.

This motivates a broader examination of collapsibility in the context of fairness, extending beyond the AUC example provided above. Most fairness metrics are parity based, defined as differences in subgroup specific values of model performance [6]. Applying such metrics without understanding and accounting for collapsibility can lead to misleading conclusions about subgroup and overall performance. A deeper understanding of collapsibility is therefore essential for fairness evaluation.

The aim of this paper is to examine the collapsibility properties of 15 commonly reported performance metrics for predictive artificial intelligence (AI) models with binary outcomes, with a particular focus on the AUC.

## 2 Collapsibility of Performance Metrics

## 2.1 Definition

Let $( Y , g )$ denote samples with a binary outcome $Y \in \{ 0 , 1 \}$ and a sensitive attribute g (e.g., $g \in \{ A , B \} )$ . Let pˆ denote the model estimated probabilities for each sample. A performance metric M is collapsible [3] over an attribute $g$ if

$$
M _ { o v e r a l l } = \frac { \sum _ { g } \left\{ w _ { g } \times M ( \hat { p } , Y , g ) \right\} } { \sum _ { g } w _ { g } } ,\tag{1}
$$

where $w _ { g }$ denote the weights used to recover the overall performance value $( \mathrm { i . e . }$ , metric specific weights). These weights are determined by the specific metric under consideration and by the way it decomposes across subgroups; for example, they may correspond to subgroup proportions or subgroup specific outcome prevalences.

A metric is collapsible over a variable $g$ if it can be written as a linear combination of its subgroup specific components. By definition, such a metric satisfies the following condition:

$$
\operatorname* { m i n } _ { g } M _ { g } \leq M _ { \mathrm { o v e r a l l } } \leq \operatorname* { m a x } _ { g } M _ { g } ,\tag{2}
$$

meaning that overall metric values are bounded by subgroup specific values.

If no metric specific weights exist, consistent with the decomposition of $M ,$ for which equation 1 holds, then M is non-collapsible with respect to g. In such cases, a Simpson’s paradox type aggregation artefact may occur, although this is neither necessary nor sufficient for non-collapsibility. When this artefact occurs, condition 2 is violated.

Because even collapsible metrics, such as the arithmetic mean, can exhibit Simpson’s paradox when subgroup comparisons are stratified by a confounding variable [11], we refer to the aforementioned phenomenon here as an “aggregation artefact”. The term refers to situations in which the overall value $M _ { \mathrm { o v e r a l l } }$ lies outside the range spanned by the subgroup values $M _ { g }$ , such that the relationship observed within subgroups is reversed or distorted after aggregation.

For non-collapsible metrics, the discrepancy between $M _ { \mathrm { o v e r a l l } }$ and the weighted combination of subgroup values is captured by Jensen’s gap. If no metric specific weights make the overall value equal to the weighted average of the subgroup values, then the aggregation relationship is non-linear, and may be convex or concave. For $g \in { \bar { \{ A , B \} } }$ , this discrepancy can be written as

$$
J _ { \mathrm { g a p } } = w _ { A } M _ { A } + ( 1 - w _ { A } ) M _ { B } - M _ { \mathrm { o v e r a l l } } ,\tag{3}
$$

where $w _ { A } \in [ 0 , 1 ]$ is the weight that would recover $M _ { \mathrm { o v e r a l l } }$ from $M _ { A }$ and $M _ { B }$ , and $J _ { \mathrm { g a p } } \in \mathbb { R }$ Figure 1 illustrates this contrast, with $J _ { \mathrm { g a p } } = 0$ for collapsible metrics (left), and $J _ { \mathrm { g a p } } \neq 0$ for non-collapsible metrics (right).

![](images/4e3ed9a64e0deb84d58907cdcc33178fb38a45166381bbfd497e52172f0469f5.jpg)

![](images/facd9a129a57310798bff8e971f18818412e9eabe2bd98aab8647fc18919c26d.jpg)  
Figure 1: Collapsible vs non-collapsible metrics as functions of a weight $w _ { g } .$ . Left: collapsible metric, $M _ { o v e r a l l }$ can be recovered from linear combination of group-specific values $M _ { A }$ and $M _ { B } .$ . Right: non-collapsible metric, curved black line $( M _ { \mathrm { o v e r a l l } } )$ deviates from the linear interpolation $( M ^ { * } )$ . This could be concave (black) or convex (grey). The vertical red arrow denotes Jensen’s gap for the concave case.

Whilst a metric can be shown to be collapsible if Equation 1 holds, non-collapsibility can be established by counterexample: if one can find a case in which Equation 2 is violated, then the metric is not collapsible.

## 2.2 Collapsible and non-collapsible metrics

We examined the collapsibility of performance metrics that are considered “recommended” [5, 6], together with metrics that are widely used but not necessarily fit for purpose, such as the F1-score. These metrics are summarised in Table 1, and Table 2 summarises their collapsibility properties, with full proofs provided in the Appendix.

Collapsibility varies across metrics depending on how their weights are computed, a property that is critical to preventing aggregation artifacts. We distinguish between strict collapsibility, where weights depend exclusively on subgroup properties, such as the number of events per subgroup, and general collapsibility, where the marginal value equals the weighted average based on any identifiable weight, such as the number of true positives according to the model.

Metrics satisfying strict collapsibility include logloss, the Brier score, accuracy, recall, specificity, and net benefit. These aggregate as linear weighted averages based on subgroup outcome prevalence $( \pi _ { g } )$ or subgroup proportion $( p ^ { g } )$ .

Metrics satisfying general collapsibility include the O:E ratio, F1-score, PPV, and NPV, which remain collapsible when weighted by the model-dependent terms (e.g., true positives).

The AUC, calibration intercept, calibration slope, expected calibration error (ECE), and Nagelkerke’s R<sup>2</sup> are non-collapsible; their overall value cannot be expressed as a weighted average of the subgroup values.

Table 1: Performance metrics for evaluating prediction models with binary outcomes [5].
<table><tr><td>Measure</td><td>Description</td></tr><tr><td>Discrimination</td><td></td></tr><tr><td>AUC / c-statistic</td><td>Probability that an individual with the event has a higher estimated probability than an individual without the event.</td></tr><tr><td>Calibration</td><td></td></tr><tr><td>O:E ratio</td><td>Ratio of the number of individuals with an event to the expected number of individuals with an event according to the model.</td></tr><tr><td>Calibration intercept</td><td>Indicates whether probabilities are on average underestimated (intercept &gt; 0), overestimated (&lt;0), or correct (0).</td></tr><tr><td>Calibration slope</td><td>Indicates whether probabilities are too extreme (slope &gt; 1), too modest (&lt;1), or perfect (1).</td></tr><tr><td>Expected Calibration Error (EČE)</td><td>Weighted mean absolute difference between average probability and observed proportion from risk groups.</td></tr><tr><td>Overall Performance</td><td></td></tr><tr><td>Logloss / Cross-entropy Brier score</td><td>Negative loglikelihood.</td></tr><tr><td></td><td>Average squared difference between outcome labels and estimated probabilities.</td></tr><tr><td>Nagelkerke R-squared</td><td>Cox-Snell R-squared scaled (loglikelihood-based R-squared) to the value for a perfect model.</td></tr><tr><td>Classification (Summary Measures)</td><td></td></tr><tr><td>Classification accuracy</td><td>Proportion of correctly classified individuals.</td></tr><tr><td>F1-score</td><td>Harmonic mean of precision and sensitivity.</td></tr><tr><td>Classification (Partial Measures)</td><td></td></tr><tr><td>Sensitivity / Recall / TPR</td><td>Proportion of individuals with an event that are correctly classified.</td></tr><tr><td>Specificity / TNR</td><td>Proportion of individuals without an event that are correctly classified.</td></tr><tr><td>PPV / Precision</td><td>Proportion of high-risk individuals that have an event.</td></tr><tr><td>NPV</td><td>Proportion of low-risk individuals that do not have an event.</td></tr><tr><td>Clinical Utility</td><td></td></tr><tr><td>Net Benefit</td><td>Net proportion of true positives, equivalent to sensitivity in the absence</td></tr></table>

Table 2: Collapsibility properties of common performance measures.
<table><tr><td>Measure</td><td>Collapsibility</td><td>Intuition</td><td> $\mathbf { W e i g h t s } , w _ { g }$ </td></tr><tr><td colspan="4">Discrimination AUC Non-collapsible</td></tr><tr><td></td><td></td><td>cross-group pairings, which cannot be expressed as a convex combination of subgroup  $\mathbf { A U C s } .$ </td><td></td></tr><tr><td colspan="4">Calibration</td></tr><tr><td>O:E ratio</td><td>Collapsible (model-dependent)</td><td>Ratio of sums; expected counts decompose linearly.</td><td> $E _ { g } / \sum _ { h } E _ { h }$ </td></tr><tr><td>Cal. Intercept</td><td>Non-collapsible</td><td>Regression coefficient, estimated globally, not an average of subgroup intercepts.</td><td></td></tr><tr><td>Cal. Slope</td><td>Non-collapsible</td><td>Regression coefficient, estimated globally, not an average of subgroup</td><td></td></tr><tr><td>ECE</td><td>Non-collapsible</td><td>slopes. Absolute deviations inside bins prevent linear aggregation across groups.</td><td></td></tr><tr><td colspan="4">Overall Performance</td></tr><tr><td>Logloss</td><td>Collapsible</td><td>Log-likelihood contributions add over individuals.</td><td> $p ^ { g }$ </td></tr><tr><td>Brier score Nagelkerke  $R ^ { 2 ^ { \bullet } }$ </td><td>Collapsible Non-collapsible</td><td>Squared errors add over individuals Nonlinear rescaling of likelihood ratio; not decomposable into group averages.</td><td> $p ^ { g }$ </td></tr><tr><td colspan="4">Classification (Summary Measures)</td></tr><tr><td>Accuracy</td><td>Collapsible</td><td>Correct classifications add linearly across individuals.</td><td> $p ^ { g }$   $2 T P _ { g } + F P _ { g } + F N _ { g }$ </td></tr><tr><td>F1-score</td><td>Collapsible (model-dependent)</td><td>Ratio of sums with group-specific denominators; can be written as weighted sum of subgroup F1.</td><td> $\begin{array} { r } { \overline { { \sum _ { h } ( 2 T P _ { h } + F P _ { h } + F N _ { h } ) } } } \end{array}$ </td></tr><tr><td colspan="4">Classification (Partial Measures)</td></tr><tr><td>TPR</td><td>Collapsible</td><td>Denominator is number of positives, which decomposes via subgroup prevalences.</td><td> $\frac { p ^ { g } \pi _ { g } } { \pi }$ </td></tr><tr><td>TNR</td><td>Collapsible</td><td>Denominator is number of negatives, which decomposes via subgroup negative prevalences.</td><td> $\frac { p ^ { g } ( 1 - \pi _ { g } ) } { 1 - \pi }$ </td></tr><tr><td>PPV</td><td>Collapsible (model-dependent)</td><td>Denominator is number of predicted positives, which decomposes linearly.</td><td> $\frac { p ^ { g } \hat { \pi } _ { g } } { \hat { \pi } }$ </td></tr><tr><td>NPV</td><td>Collapsible (model-dependent)</td><td>Denominator is number of predicted negatives, which decomposes linearly.</td><td> $\frac { p ^ { g } ( 1 - \hat { \pi } _ { g } ) } { 1 - \hat { \pi } }$ </td></tr><tr><td colspan="4">Clinical Utility</td></tr><tr><td>Net Benefit</td><td>Collapsible</td><td>Defined directly in terms of collapsible counts (TP, FP).</td><td> $p ^ { g }$ </td></tr></table>

$p ^ { g } = N _ { g } / N$ is the subgroup proportion; $\pi _ { g } = N _ { g } ^ { + } / N _ { g }$ is the subgroup specific prevalence of the outcome, with $\pi = N ^ { + } / N$ overall; $\hat { \pi } _ { g } = \hat { N } _ { g } ^ { + } / N _ { g }$ is the subgroup specific predicted positive rate, with $\begin{array} { r } { \hat { \pi } = \hat { N } ^ { + } / N _ { o v e r a l l } ; E _ { g } = \sum _ { i \in g } \hat { p } _ { i } } \end{array}$ is the expected number of events in subgroup g; $T P _ { g } , F P _ { g } , F N _ { g }$ are subgroup specific confusion matrix counts.

## 3 AUC mixture decomposition and non-collapsibility

The AUC is a commonly reported prediction model performance metric [5, 14], quantifying how well a model ranks individuals who experience the event above those who do not [1]. Because it evaluates all event and non-event pairs, it includes pairs that span different subgroups in fairness sensitive settings, a property previously examined through the notion of cross-AUC (xAUC) [4]. The $\mathrm { \ x A U C _ { i , j } }$ represents the probability that an individual with the event from subgroup i is ranked higher than an individual without the event from subgroup j [4]. Both within-group and crossgroup AUC are expected to influence the overall AUC. Given the central role of AUC in evaluating discrimination, we derive its decomposition under mixtures of subpopulations and formally prove that it is non-collapsible.

## 3.1 Problem setting and notation

Consider a dataset of i.i.d. samples $( X , G , Y )$ , where X denotes the predictor vector, G is the subgroup indicator taking one of K possible values in $\{ 1 , \ldots , K \}$ (sensitive attribute), and $Y \in \{ 0 , 1 \}$ is the binary outcome. Let

$$
p _ { g } = P ( G = g ) , \qquad p _ { y } ^ { g } = P ( G = g , Y = y ) ,
$$

and define the subgroup outcome prevalence

$$
\pi _ { g } = P ( Y = 1 \mid G = g ) = \frac { p _ { 1 } ^ { g } } { p _ { g } } .
$$

By the law of total probability,

$$
p _ { y } = \sum _ { g = 1 } ^ { K } p _ { y } ^ { g } , \qquad p _ { g } = p _ { 0 } ^ { g } + p _ { 1 } ^ { g } , \qquad p _ { 1 } ^ { g } = p _ { g } \pi _ { g } , \quad p _ { 0 } ^ { g } = p _ { g } ( 1 - \pi _ { g } ) .
$$

A prediction model

$$
h : \mathcal { X } \to [ 0 , 1 ] , \qquad \hat { p } = h ( X )
$$

produces estimated probabilities ${ \hat { p } } .$

For a decision threshold $\tau \in [ 0 , 1 ] ,$ , group- and class-conditional cumulative distributions can be defined as:

$$
F _ { g , y } ( \tau ) = P \big ( \hat { p } \leq \tau ~ | ~ g , Y = y \big ) , ~ g \in \{ A , \ldots , K \} , ~ y \in \{ 0 , 1 \} ,
$$

and true positive rate (TPR) and false positive rate (FPR) functions can be defined as follows (and as illustrated in Figure 2):

$$
\mathrm { T P R } _ { g } ( \tau ) = 1 - F _ { g , 1 } ( \tau ) , \quad \mathrm { F P R } _ { g } ( \tau ) = 1 - F _ { g , 0 } ( \tau ) ,
$$

Evaluating across all subgroups, as typically done when evaluating performance at population level, yields the overall model estimated probability distributions

$$
F _ { y } ( t ) = P ( \hat { p } \leq \tau \mid Y = y ) , \quad y \in \{ 0 , 1 \} .\tag{4}
$$

The AUC is then defined as [15]:

$$
\mathrm { A U C } = \int ( 1 - F _ { 1 } ( \tau ) ) \mathrm { d } F _ { 0 } ( \tau ) .\tag{5}
$$

## 3.2 Mixture decomposition of overall AUC and non-collapsibility

The overall AUC can also be expressed as the probability that the prediction model assigns a higher estimated probability to a randomly selected individual with the event than to a randomly selected individual without the event [16]:

$$
\mathrm { A U C _ { o v e r a l l } } = P ( \hat { p } _ { 1 } > \hat { p } _ { 0 } ) ,\tag{6}
$$

![](images/9b0880e67127608140d25c0bb7fbfcd9f4dc16eef5a64f80172713e6e2d2d071.jpg)  
Figure 2: Conditional estimated probability densities and shaded error-rates for $^ 2$ groups. For both Group A and for Group $B ,$ the solid line denotes the distribution $f _ { 0 } = F _ { 0 } ^ { \prime }$ (individuals without the event) and the dashed line, the distribution $f _ { 1 } = F _ { 1 } ^ { \prime }$ (individuals with the event). At a common threshold τ (set to 0.4 in this example), the red-hatched area shows $\mathrm { F P R } _ { g }$ and the black-hatched area shows $\mathrm { T P R } _ { g }$ [1]

where $\hat { p } _ { 1 }$ and $\hat { p } _ { 0 }$ denote the estimated probabilities for independent draws from the event and nonevent individuals, respectively.

Conditioning on subgroup membership, let $\hat { p } _ { 1 } ^ { i }$ denote the estimated probability for an individual with the event from subgroup $i ,$ and let $\hat { p } _ { 0 } ^ { j }$ denote the estimated probability for an individual without the event from subgroup j, where $i , j \in \{ 1 , \ldots , K \}$ . Assuming independence between sampled pairs, we obtain

$$
{ \begin{array} { r l } & { { \mathrm { A U C } } _ { \mathrm { o v e r a l l } } = \displaystyle \sum _ { i , j = 1 } ^ { K } P ( G = i \mid Y = 1 ) P ( G = j \mid Y = 0 ) P ( { \hat { p } } _ { 1 } ^ { i } > { \hat { p } } _ { 0 } ^ { j } ) } \\ & { \quad \quad \quad = \displaystyle \sum _ { i , j = 1 } ^ { K } { \frac { p _ { 1 } ^ { i } } { p _ { 1 } } } { \frac { p _ { 0 } ^ { j } } { p _ { 0 } } } { \mathrm { A U C } } _ { i , j } , } \end{array} }\tag{7}
$$

This decomposition can be represented as

$$
\begin{array} { r } { \mathrm { A U C } _ { \mathrm { o v e r a l l } } = \left( \frac { p _ { 1 } ^ { g _ { 1 } } } { p _ { 1 } } \quad \cdot \cdot \cdot \quad \frac { p _ { 1 } ^ { g _ { K } } } { p _ { 1 } } \right) \mathbf { M } _ { \mathrm { A U C } } \left( \begin{array} { c } { \frac { p _ { 0 } ^ { g _ { 1 } } } { p _ { 0 } } } \\ { \vdots } \\ { \frac { p _ { \tiny { 0 } } ^ { g _ { K } } } { p _ { 0 } } } \end{array} \right) , } \end{array}\tag{8}
$$

where ${ \bf M } _ { \mathrm { A U C } }$ is a $K \times K$ matrix of within- and cross-group AUCs with entries $\mathrm { A U C } _ { i , j } = P ( \hat { p } _ { 1 } ^ { i } > \hat { p } _ { 0 } ^ { j } )$ The diagonal entries correspond to within-group $\mathbf { A U C s } ,$ while the off-diagonal terms capture crossgroup rankings.

For two groups A and B, the decomposition reduces to

$$
\boxed { \mathrm { A U C } _ { \mathrm { o v e r a l l } } = \frac { p _ { 1 } ^ { A } p _ { 0 } ^ { A } } { p _ { 1 } p _ { 0 } } \mathrm { A U C } _ { A } + \frac { p _ { 1 } ^ { A } p _ { 0 } ^ { B } } { p _ { 1 } p _ { 0 } } \mathrm { x A U C } _ { A , B } + \frac { p _ { 1 } ^ { B } p _ { 0 } ^ { A } } { p _ { 1 } p _ { 0 } } \mathrm { x A U C } _ { B , A } + \frac { p _ { 1 } ^ { B } p _ { 0 } ^ { B } } { p _ { 1 } p _ { 0 } } \mathrm { A U C } _ { B } }\tag{9}
$$

with

$$
\begin{array} { r } { \mathbf { M } _ { \mathbf { A U C } } = \left( \begin{array} { c c } { \mathrm { A U C } _ { A } } & { \mathrm { x A U C } _ { A , B } } \\ { \mathrm { x A U C } _ { B , A } } & { \mathrm { A U C } _ { B } } \end{array} \right) . } \end{array}\tag{10}
$$

In the special case where $\pi _ { g } ~ = ~ \pi , ~ \mathrm { i . e . }$ ., each subgroup has the same outcome prevalence, the decomposition in (7) simplifies further. Substituting $p _ { 1 } ^ { i } = p ^ { i } \pi , p _ { 0 } ^ { i } = p ^ { i } ( 1 - \pi ) , p _ { 1 } = \pi$ , and $p _ { 0 } = 1 - \pi .$ , we obtain $\begin{array} { r } { \frac { p _ { 1 } ^ { i } p _ { 0 } ^ { j } } { p _ { 1 } p _ { 0 } } = p ^ { i } p ^ { j } } \end{array}$ , which yields

$$
\mathrm { A U C } _ { \mathrm { o v e r a l l } } = \sum _ { i , j \in \{ A , B \} } p ^ { i } p ^ { j } \mathrm { A U C } _ { i , j }\tag{11}
$$

and, noting that $p ^ { B } = 1 - p ^ { A }$

$$
\operatorname { A U C } _ { \mathrm { o v e r a l l } } = \left( { p ^ { A } } \right) ^ { 2 } \operatorname { A U C } _ { A } + \left( 1 - { p ^ { A } } \right) ^ { 2 } \operatorname { A U C } _ { B } + p ^ { A } ( 1 - p ^ { A } ) \big ( \operatorname { x A U C } _ { A , B } + \operatorname { x A U C } _ { B , A } \big ) .\tag{12}
$$

Thus, when subgroups have equal outcome prevalence, the overall AUC reduces to an average of within-group and cross-group AUC terms weighed by the subgroup proportions $p ^ { A }$

This mixture decomposition highlights that the overall AUC depends not only on within-group AUCs but also on cross-group terms (xAUC, of equation 9), subgroup proportions $( p _ { g } )$ , and subgroup specific outcome prevalences $( \pi _ { g } ) [ 1 7 ]$ . Consequently, the overall AUC may lie outside the convex bounds of the subgroup AUCs, violating the collapsibility condition 2. This property has direct implications for fairness evaluation, as discussed in Section 4.

## 3.3 AUC collapsibility special cases

Despite non-collapsibility being observed in general for the AUC, there are specific scenarios when collapsibility may occur. A collapsible form of the AUC would depend solely on within-group AUC terms. Transitioning from equation 7 to equation 1 requires an appropriate combination of subgroup AUC and cross-group xAUC terms, such as the ones that we suggest and explore below.

Scenario (i): $\mathbf { x A U C } _ { i , j } = \mathbf { A U C } _ { i }$

Let the cross-group AUC equal the within-group AUC for each subgroup g:

$$
\mathrm { x A U C } _ { A , B } = \mathrm { A U C } _ { A } , \quad \mathrm { x A U C } _ { B , A } = \mathrm { A U C } _ { B } .
$$

Applying equation 9 for two groups A and B, without loss of generality, it can be shown that:

$$
\begin{array} { r l r } {  { \mathrm { A U C } _ { \mathrm { o v e r a l l } } = \frac { p _ { 1 } ^ { A } p _ { 0 } ^ { A } } { p _ { 1 } p _ { 0 } } \mathrm { { A U C } } _ { A } + \frac { p _ { 1 } ^ { A } p _ { 0 } ^ { B } } { p _ { 1 } p _ { 0 } } \mathrm { { x A U C } } _ { A , B } + \frac { p _ { 1 } ^ { B } p _ { 0 } ^ { A } } { p _ { 1 } p _ { 0 } } \mathrm { { x A U C } } _ { B , A } + \frac { p _ { 1 } ^ { B } p _ { 0 } ^ { B } } { p _ { 1 } p _ { 0 } } \mathrm { { A U C } } _ { B } } } \\ & { } & \\ & { } & { \quad = \frac { p _ { 1 } ^ { A } ( p _ { 0 } ^ { A } + p _ { 0 } ^ { B } ) } { p _ { 1 } p _ { 0 } } \mathrm { { A U C } } _ { A } + \frac { p _ { 1 } ^ { B } ( p _ { 0 } ^ { A } + p _ { 0 } ^ { B } ) } { p _ { 1 } p _ { 0 } } \mathrm { { A U C } } _ { B } } \\ & { } & \\ & { } & { \quad = \sum _ { g } \frac { \pi _ { g } } { \pi } p ^ { g } \mathrm { { A U C } } _ { g } . } \end{array}\tag{13}
$$

Thus, if cross-group AUC values equal the within-group AUC values, i.e., individuals without the event are identically distributed across subgroups, then $\mathrm { { A U C } _ { \mathrm { { o v e r a l l } } } }$ can be written as a linear combination, achieved with $\begin{array} { r } { w _ { g } = \frac { \pi _ { g } } { \pi } p ^ { g } \left( 1 \right) } \end{array}$ .

Scenario (ii): $\mathbf { x A U C } _ { i , j } = \mathbf { A U C } _ { i }$ and $\pi _ { g } = \pi .$

In addition to the cross-group AUCs equalling the within-group $\mathbf { A U C s } .$ , suppose that the subgroup outcome prevalences also match $( \mathrm { i } . \mathrm { e } . , \pi _ { g } = \pi )$ . Then, $\mathrm { A U C } _ { \mathrm { o v e r a l l } }$ can be expressed as:

$$
\mathrm { A U C _ { o v e r a l l } } = \sum _ { g } p ^ { g } \mathrm { A U C } _ { g } ,\tag{14}
$$

These special cases are presented here as theoretical exceptions, and in practice the AUC should generally be treated as non-collapsible in the population of interest. In Section 4, we explore scenarios in which $\mathrm { x A U C } _ { i , j } \ne \mathrm { A U C } _ { i } .$ , and examine how these may lead to counter-intuitive fairness evaluations. This raises deeper questions about the AUC itself, particularly its sensitivity to case-mix differences and its relationship to performance metrics beyond discrimination, such as calibration.

## 3.4 AUC estimation under normal assumptions

To support the forthcoming examples, we derive the closed-form expression for the AUC when the overall distribution arises from a mixture of subpopulations (equation 7), under normality assumptions.

There are multiple ways of estimating the AUC, including the Mann–Whitney statistic, kernel smoothing, normal assumptions, and empirical transformations to normality [16, 18]. Here, for illustrative purposes, we derive a closed-form expression for the overall AUC as defined by equation 7, under a truncated normal approximation. Without loss of generality, suppose that within each group $g \in \{ A , B \}$ , the estimated probabilities for those with and without the event are approximately truncated normal on [0, 1]:

$$
\begin{array} { r l } { \hat { p } _ { 0 } ^ { A } \sim \mathcal { T N } _ { [ 0 , 1 ] } ( \mu _ { 0 } ^ { A } , ( \sigma _ { 0 } ^ { A } ) ^ { 2 } ) , } & { \quad \hat { p } _ { 1 } ^ { A } \sim \mathcal { T N } _ { [ 0 , 1 ] } ( \mu _ { 1 } ^ { A } , ( \sigma _ { 1 } ^ { A } ) ^ { 2 } ) , } \\ { \hat { p } _ { 0 } ^ { B } \sim \mathcal { T N } _ { [ 0 , 1 ] } ( \mu _ { 0 } ^ { B } , ( \sigma _ { 0 } ^ { B } ) ^ { 2 } ) , } & { \quad \hat { p } _ { 1 } ^ { B } \sim \mathcal { T N } _ { [ 0 , 1 ] } ( \mu _ { 1 } ^ { B } , ( \sigma _ { 1 } ^ { B } ) ^ { 2 } ) . } \end{array}
$$

![](images/e17e9e28d427125801c3856de728b7800976e87c95b6eae688e827d77647d600.jpg)

![](images/932a38b6d876932deca88ac9cf9d4503ee249fb621e1dd36e4d36baf7701d04a.jpg)  
Figure 3: Group-conditional densities for those without the event $( p _ { 0 } ^ { g } .$ , solid) and those with the event $( p _ { 1 } ^ { \breve { g } }$ , dashed). Nearby text labels indicate the corresponding mean and standard deviation $\mu _ { y } ^ { g } , \sigma _ { y } ^ { g }$

Then for any pair (i, j) the component AUC admits the closed-form:

$$
\mathrm { A U C } _ { i , j } = P \big ( \hat { p } _ { 1 } ^ { i } > \hat { p } _ { 0 } ^ { j } \big ) = \Phi \Big ( \frac { a _ { i j } } { \sqrt { 1 + b _ { i j } ^ { 2 } } } \Big ) ,\tag{15}
$$

where a is the separation coefficient and b the symmetry coefficient,

$$
a _ { i j } = \frac { \mu _ { 1 } ^ { i } - \mu _ { 0 } ^ { j } } { \sigma _ { 1 } ^ { i } } , \qquad b _ { i j } = \frac { \sigma _ { 0 } ^ { j } } { \sigma _ { 1 } ^ { i } } .
$$

By the mixture decomposition 7, the overall AUC becomes

$$
\mathrm { A U C } _ { \mathrm { o v e r a l l } } = \sum _ { i , j \in \{ A , B \} } { \frac { p _ { 1 } ^ { i } } { p _ { 1 } } } { \frac { p _ { 0 } ^ { j } } { p _ { 0 } } } \Phi \Bigl ( { \frac { a _ { i j } } { \sqrt { 1 + b _ { i j } ^ { 2 } } } } \Bigr )\tag{16}
$$

Although assuming that the model’s estimated probabilities follow a normal distribution may not accurately reflect real-world scenarios, we adopt this assumption in what follows for illustration and interpretability.

## 4 AUC non-collapsibility and fairness implications

Non-collapsibility can be a practical concern for fairness assessments. A collapsible metric always remains within the range defined by the subgroup minimum and maximum (equation 2); in contrast, a non-collapsible metric, such as the AUC, can breach these bounds, manifesting as an aggregation artefact (examples in Figure 5). Collapsible metrics also permit straightforward reasoning from subgroup to overall level. For the AUC in particular, the overall value entangles within-group and cross-group pairwise comparisons, which complicates interpretation and communication among statisticians, clinicians, and policymakers. This section examines specific ways in which AUC non-collapsibility can affect fairness evaluation.

## 4.1 Counter-intuitive fairness scenarios

Jensen’s gap, $J _ { g a p }$ , arises from the non-collapsibility of the AUC and quantifies the deviation of the overall metric from the linear combination of subgroup metrics. Here we use $J _ { g a p }$ to illustrate and examine two counter-intuitive, yet plausible, fairness scenarios (Figure 4).

## (a) Subgroup parity satisfied, but overall AUC is lower:

![](images/38a36903c969b0bd380d6ed847dd56c47a201c0ff49af44a1975cad5265ce3ce.jpg)

(b) $\mathbf { A U C } _ { B }$ and $\mathbf { A U C } _ { \mathrm { o v e r a l l } }$ match, but $\mathbf { A U C } _ { A }$ is higher:

![](images/edbf931856094dc5105c5ae81586a4c15a3664625e22c1bc150087d9b2068563.jpg)  
Figure 4: Two scenarios illustrating relationships between subgroup AUCs and the overall AUC.

Scenario (i): $\mathbf { A U C } _ { A } = \mathbf { A U C } _ { B }$

Figure 5a shows that subgroup parity may hold $( A U C _ { A } = 0 . 8 6$ and $A U C _ { B } = 0 . 8 6 )$ while the overall AUC is substantially lower $( \dot { A } U \dot { C } _ { \mathrm { o v e r a l l } } = 0 . 7 8 )$ . This apparent paradox is explained by the cross-group AUC terms: $\mathrm { _ { x A U C } } _ { A , B }$ is disproportionately low (0.36), whereas $\Pi \mathrm { \Pi } \mathrm { C } _ { B , A }$ is very high (0.99), leading to an overall AUC that is lower than both subgroups’ (similar) performance. Mathematically, when subgroup parity is satisfied, i.e., $A U C _ { A } = A { \bar { U } } C _ { B } \dot { : }$

$$
\begin{array} { r l } & { A U C _ { A } - A U C _ { \mathrm { o v e r a l l } } = A U C _ { A } - \left( w _ { A } A U C _ { A } + w _ { B } A U C _ { B } - J _ { g a p } \right) } \\ & { \qquad = A U C _ { A } - \left( w _ { A } A U C _ { A } + ( 1 - w _ { A } ) A U C _ { A } - J _ { g a p } \right) = J _ { g a p } . } \end{array}\tag{17}
$$

Scenario (ii): $\mathbf { A U C } _ { o v e r a l l } = \mathbf { A U C } _ { B }$

Figure 5b illustrates a setting in which one subgroup’s AUC $( A U C _ { B } = 0 . 7 6 )$ is similar to the overall AUC $( A U C _ { \mathrm { o v e r a l l } } = 0 . 7 6 )$ but the other subgroup AUC $( A U C _ { A } = 0 . 9 2 )$ is markedly higher. Once again, the difference arises from the cross-group AUC terms, which are further weighted by subgroup proportions and subgroup outcome prevalence differences. Formally, when one subgroup metric equals the overall metric, i.e., $A U C _ { A } ^ { - } = A U C _ { \mathrm { o v e r a l l } } ;$

$$
A U C _ { A } - A U C _ { B } = \frac { A U C _ { A } - A U C _ { \mathrm { o v e r a l l } } + S } { w _ { B } } = \frac { J _ { g a p } } { w _ { B } } .\tag{18}
$$

These examples illustrate that the counter-intuitive behaviours are directly tied to the non-collapsibility of the AUC, as captured by $J _ { g a p } .$ The resulting differences are not artefacts of estimation but rather consequences of the decomposition in equation 7. Examining the distribution of the estimated probabilities generated by the model in both scenarios reveals substantial differences between groups A and B that account for the observed AUC values. The following section investigates these distributions, why $J _ { g a p }$ may be non-zero, and explores how its magnitude reflects the interaction between case-mix differences and other dimensions of model performance.

![](images/3995654e588bc0445d7a1da328702fd96a2496be6801a324225c9013f3ec4894.jpg)

![](images/8b4b440258abd12716c6a56eb41ed456e4e13421a7e419c60b65882686158b7f.jpg)  
(a) Simulated scenario in which subgroup parity is satisfied $( \mathrm { A U C _ { A } } = \mathrm { A U C _ { B } } )$ but overall AUC is lower than both subgroup AUC values. Estimated probability distributions follow $\mu _ { 0 } ^ { A } = 0 . 3 ,$ $\mu _ { 1 } ^ { A } = 0 . 4 5 , \mu _ { 0 } ^ { B } = 0 . 5 ,$ , and $\hat { \mu } _ { 1 } ^ { B } = 0 . 6 5$

![](images/8d5d203f1bb3abd970bb0eac2ea96a3f8f8cbda4e5ced2fa13571a331afe89ca.jpg)

![](images/59752c9daed7bc4017ae22f90456e0f02d136ee6262c8fe4461f7a777f4aeedb.jpg)  
(b) Simulated scenario in which $\mathrm { A U C _ { B } } = \mathrm { A U C _ { o v e r a l l } }$ , but $\mathrm { A U C _ { A } }$ falls outside the range. Estimated probability distributions follow $\mu _ { 0 } ^ { A } = 0 . 3 , \mu _ { 1 } ^ { A } = 0 . 5 , \mu _ { 0 } ^ { B } = 0 . 5 1$ , and $\mu _ { 1 } ^ { B } = 0 . 6 1$  
Figure 5: Illustrative cases comparing subgroup and overall AUC relationships. On the left panel, AUC component values are shown in the form of a matrix, $\mathrm { M _ { A U C } }$ , with $\mathrm { A U C } _ { \mathrm { o v e r a l l } }$ for the pooled sample in the centre. On the right panel, the distributions of the estimated probability by outcome and subgroup are shown. Within each subgroup $g \in \{ A , B \}$ the estimated probabilities for non-events and events are drawn from truncated normal distributions on $[ 0 , 1 ]$ , using 100,000 samples in total. For both scenarios, $p ^ { A } = 0 . 3 , \pi ^ { A } = \pi ^ { B } = \pi = 0 . 5$ , and the estimated probabilities follow $\sigma = 0 . 1$ for all strata. The parameters defined here are common to both scenarios but $\mu _ { 0 } ^ { A } , \mu _ { 1 } ^ { A } , \mu _ { 0 } ^ { B } , \mu _ { 1 } ^ { B }$ are specific to each scenario above.

An interactive web application for exploring the proposed concepts and the mentioned scenarios is available at https://collapsibility.streamlit.app/. The corresponding Python code, used to implement the app and produce all demonstrations presented in this work, is available at https: $/ / { \tt g } \dot { \bf 1 }$ thub.com/joamats/collapsibility.

## 4.2 Drivers of non-collapsible behaviour

To understand why the AUC may fail to be collapsible, and how this leads to the counter-intuitive scenarios above, we examine how differences in the distributions of the estimated model probabilities across subgroups influence both the within-group AUC and the cross-group xAUC terms.

For illustration, suppose that there are two groups, A and $B ,$ defined by a sensitive attribute $^ { g , }$ and assume normality (truncated to [0, 1], by definition) of the estimated probabilities with equal standard deviations across all subgroup-outcome strata [19]:

$$
\sigma _ { 0 } ^ { A } = \sigma _ { 1 } ^ { A } = \sigma _ { 0 } ^ { B } = \sigma _ { 1 } ^ { B } .
$$

Under this assumption the symmetry coefficient $b _ { i j }$ appearing in equation 15 becomes $b _ { i j } = 1$ . The component AUC therefore simplifies to

$$
\mathrm { A U C } _ { i , j } = \Phi \left( { \frac { a _ { i j } } { \sqrt { 2 } } } \right) = \Phi \left( { \frac { \mu _ { 1 } ^ { i } - \mu _ { 0 } ^ { j } } { \sqrt { 2 } \sigma } } \right) ,\tag{19}
$$

where $\sigma$ denotes the common standard deviation.

As noted in subsection 3.3, collapsibility can be achieved via equality between certain within-group and cross-group AUC terms, such as $\mathrm { A U C } _ { i } = \mathrm { x A U C } _ { i , j }$ . Under the simplified expression above, this condition becomes

$$
\mu _ { 1 } ^ { i } - \mu _ { 0 } ^ { i } = \mu _ { 1 } ^ { i } - \mu _ { 0 } ^ { j } \quad \Longleftrightarrow \quad \mu _ { 0 } ^ { i } = \mu _ { 0 } ^ { j } .
$$

Alternatively, if collapsibility were to arise through $\mathrm { A U C } _ { j } = \mathrm { x A U C } _ { i , j }$ , then

$$
\mu _ { 1 } ^ { i } - \mu _ { 0 } ^ { j } = \mu _ { 1 } ^ { j } - \mu _ { 0 } ^ { j } \quad \Longleftrightarrow \quad \mu _ { 1 } ^ { i } = \mu _ { 1 } ^ { j } .
$$

Hence deviations from collapsibility can be attributable to differences in the mean estimated probabilities across subgroups, specifically to contrasts of the form $\mu _ { 1 } ^ { A } - \mu _ { 1 } ^ { B }$ and $\mu _ { 0 } ^ { A } - \mu _ { 0 } ^ { B }$ . The quantity $\mu _ { 1 } ^ { A } - \mu _ { 1 } ^ { B }$ has been referred to in the literature as “balance for the positive class” and $\mu _ { 0 } ^ { A } - \mu _ { 0 } ^ { B }$ as “balance for the negative class” [20].

This raises the question of why the means of the estimated probabilities should differ between subgroups at all. Several mechanisms may lead to such differences: two that are common in a fairness evaluation are highlighted below.

## Driver (i): Differences in subgroup outcome prevalence under fair mean calibration.

Assume the prediction model is well-calibrated within each subgroup in the sense of calibration-inthe-large, meaning the average estimated probability matches the outcome prevalence [2]. When within-group discrimination is identical, differences in subgroup outcome prevalence $\pi _ { A }$ and $\pi _ { B }$ will induce corresponding differences in the mean estimated probabilities, since a group with a higher outcome prevalence will, on average, have higher estimated probabilities. These shifts in $\mu _ { 1 } ^ { g }$ and $\mu _ { 0 } ^ { g }$ propagate into the xAUC terms, potentially creating cross-group asymmetries even when the withingroup AUCs remain comparable. Under fairness evaluation involving two or more heterogeneous subpopulations, such xAUC differences may have counter-intuitive effects on the overall AUC.

## Driver (ii): Differences in subgroup calibration under equal subgroup outcome prevalence.

Now suppose the subgroups share a common outcome prevalence but differ in their mean calibration. For instance, systematic over- or underestimation in one subgroup would shift either $\mu _ { 1 } ^ { g }$ or $\mu _ { 0 } ^ { g }$ relative to the proportion of observed outcomes. Such differences again alter $\mathrm { _ { x A U C } } _ { i , j }$ even when the within-group discrimination remains similar.

These two mechanisms motivate a closer examination of how case-mix differences and miscalibration differences respectively contribute to AUC behaviour, as detailed below.

## 4.2.1 The role of case-mix differences toward naturally occurring subgroup differences

Differences in AUC across subgroups do not necessarily stem from shortcomings in the model’s ability to discriminate, nor do they necessarily imply a lack of fairness. They may instead reflect differences in the underlying case-mix [21], a phenomenon broadly referred to as spectrum bias [22], whereby the performance of a prediction model varies when applied to populations with differing distributions of patient characteristics.

Here we focus on two quantities that can vary across subgroups defined by a sensitive attribute and affect the mixture decomposition of the overall AUC: subgroup proportions and subgroup specific outcome prevalences. Subgroup proportions reflect population composition, whereas differences in outcome prevalence may arise from underlying case-mix differences in baseline characteristics. Both influence the mixture decomposition of the overall AUC and thereby the magnitude of noncollapsibility effects.

Let $p ^ { g }$ denote the proportion of individuals in subgroup g, and $\pi _ { g }$ the outcome prevalence within that subgroup. Differences in these quantities influence the mixture decomposition of the overall AUC and thereby contribute to the magnitude of non-collapsibility effects.

## (i) Different subgroup proportions:

Subgroup proportions $p ^ { g }$ do not affect the AUC or xAUC terms, but they do determine the relative influence of these components in the overall mixture expression (equation 16). Consequently, even if a model behaves identically across subgroups, different proportions of the subgroups can impact on the overall AUC.

## (ii) Different subgroup specific outcome prevalences:

Although the AUC is invariant to outcome prevalence in a homogeneous population, differences in $\pi _ { g }$ across subgroups influence the mixture decomposition in two important ways. First, as discussed in Section 3.3, varying $\pi _ { g }$ affects the means of the estimated probability distributions – potentially to different extents, depending on how much subgroup specific calibration-in-the-large follows such differences or not. These mean shifts alter the relative positions of the subgroup estimated probability distributions and therefore modify the xAUC terms, even when the within-group discrimination remains unchanged. Second, subgroup outcome prevalences determine the mixture weights $\frac { p _ { 1 } ^ { g } } { p _ { 1 } }$ and $\frac { p _ { 0 } ^ { g } } { p _ { 0 } }$ , which govern the contribution of each subgroup-outcome stratum to the overall AUC. Differences in $\pi _ { g }$ therefore amplify or attenuate the influence of particular strata on the mixture.

![](images/dea489abe80e5afb50caf4a635005337b09e36622793fbd5211f6f28afe505b5.jpg)  
Figure 6: Overall AUC of a single model applied to subgroups A and B as the sample proportion $p _ { A }$ varies from 0 to 1. Curves show different prevalences in $\ A , \pi _ { A } \in \{ 0 . 2 , 0 . 3 , \ldots , 0 . 8 \}$ , with fixed $\pi _ { B } = 0 . 5 .$ . There are 20,000 samples in total for each scenario. For each subgroup $g \in \{ A , B \}$ estimated probabilities for non-events and events were drawn from truncated Normal([0, 1]) with means $( \mu _ { 0 } ^ { A } , \mu _ { 1 } ^ { A } ) \ : = \ : ( 0 . 6 , 0 . 7 )$ and $( \mu _ { 0 } ^ { B } , \mu _ { 1 } ^ { B } ) = ( 0 . 2 5 , 0 . 3 0 )$ , and fixed $\sigma = 0 . 1$ . Endpoints $p _ { A } = 0$ , 1 recover $\mathrm { A U C _ { B } }$ and $\mathrm { A U C _ { A } }$ , respectively.

Figure 6 illustrates these effects, showing how variations in subgroup proportions and subgroup specific prevalences jointly influence the overall AUC under a fixed underlying model. These observations highlight that case-mix differences, including variations in subgroup proportions and subgroup specific outcome prevalences, can produce non-collapsible behaviour that may inflate “fairness gaps” that do not necessarily translate to unfairness in the model. Rather, such differences may reflect population heterogeneity and should be interpreted with care in fairness evaluations.

This perspective aligns closely with earlier work by van Klaveren and colleagues, who proposed a concordance based metric designed to disentangle the contribution of case-mix heterogeneity from true differences in discriminative ability [23]. It is also consistent with propensity based standardisation approaches, which aim to disentangle differences in discrimination due to case-mix from differences due to model generalisability across samples [24]. Furthermore, Vergouwe and colleagues demonstrate that shifts in the distribution of patient characteristics can affect model performance, even when the underlying model remains unchanged [21]. Notably, they also document a scenario in which more severe case-mix changes induce systematic miscalibration, which naturally raises the question of how calibration and discrimination interact. We therefore now examine the role of miscalibration differences in shaping observed AUC behaviour.

## 4.2.2 The role of miscalibration differences

The AUC is a rank based metric, invariant to any strictly monotone transformation of the estimated probabilities, and is therefore generally, considered independent of calibration [25]. This interpretation warrants some clarification. Pencina and colleagues have noted that miscalibration can affect the empirical AUC in practice [26]. More recently, Sadatsafavi and colleagues introduced a “model-based ROC curve”, which separates the effects of case-mix heterogeneity and miscalibration [27]. These contributions motivate a closer examination of how miscalibration interacts with case-mix to affect discrimination, particularly in fairness contexts where subgroup differences are of key interest.

In heterogeneous populations, the overall AUC reflects not only within-group discrimination but also cross-group estimated probability comparisons. Consequently, even equal within-group AUCs can yield a higher, or lower, overall AUC if groups differ in mean calibration or outcome prevalence. This conflation complicates fairness evaluations, since variations in overall AUC may arise from calibration discrepancies across subgroups, rather than genuine differences in discrimination. The form of calibration relevant here corresponds to calibration at the level of covariate patterns, whereby estimated probabilities match observed event rates for each subgroup specific covariate pattern. As discussed by Van Calster and colleagues [2], this “strong calibration” is rarely achievable, yet systematic differences in estimated probabilities propagate directly into the AUC decomposition. These considerations further underscore the importance of evaluating discrimination and calibration jointly [5], including subgroup performance alongside overall performance.

## 4.3 Clinical illustration: EuroSCORE II

Here, we use the European System for Cardiac Operative Risk Evaluation (EuroSCORE) II, a clinical prediction model for in-hospital mortality after cardiac surgery [28], as the setting for three hypothetical fairness evaluations stratified by sex. The numerical values are illustrative and are not derived from an empirical analysis of EuroSCORE II. Box 2 illustrates how differences in overall, subgroup, and cross-group AUCs may arise without necessarily indicating model unfairness.

## Box 2: Illustrative fairness evaluation scenarios using EuroSCORE II

Apparent subgroup disparity: Assume that EuroSCORE II has higher discrimination among male patients than among female patients $\mathsf { \Gamma } ( \mathrm { A U C } _ { M } = 0 . 8 8$ versus $\mathrm { A U C } _ { F } = \mathbf { \bar { 0 } } . 7 3 )$ , while $\mathrm { A U C _ { g l o b a l } } = 0 . 8 5$ The cross-group discrimination values are also asymmetric, with $\mathrm { x A U C } _ { M , F } = 0 . 6 7$ and $\mathrm { \ x A U C } _ { F , M } = 0 . 9 2$ Here, $x A \bar { U } C _ { M , F } = 0 . 6 7$ indicates that a male patient who dies is $\mathrm { ^ { * } r a n k e d ^ { * } }$ above a female patient who survives only $67 \%$ of the time, whereas $x A U \dot { C } _ { F , M } = 0 . 9 2$ indicates that a female patient who dies is “ranked” above a male patient who survives 92% of the time. These within- and cross-group differences identify potential fairness concerns that warrant further investigation, but they do not by themselves establish that the model is unfair.

Improved subgroup performance but lower overall AUC: Discrimination among female patients increases to $\mathrm { A U C } _ { F } = 0 . 8 0$ , while discrimination among male patients remains $\mathrm { A U } \bar { \mathrm { C } } _ { M } = 0 . \bar { 8 8 }$ . Nevertheless, the overall AUC decreases from 0.85 to 0.84. The corresponding cross-group values are $\mathrm { x A U C } _ { M , F } = 0 . 4 4$ and $\mathrm { x A U C } _ { F , M } = 0 . 9 9$ . While $\mathrm { x A U C } _ { F , M } = 0$ .99 indicates almost perfect ranking of female patients who die above male patients who survive, $\mathrm { x A U C } _ { M , F } = 0 . 4 4$ indicates that male patients who die are ranked above female survivors less often than expected by chance. The lower overall AUC should therefore not be interpreted as evidence that the model has become worse overall, as it reflects changes in cross-group rankings rather than within-group performance alone.

## Subgroup parity without improvement in overall AUC:

Within-group discrimination is equalised $( \mathrm { A U C } _ { M } = \mathrm { A U C } _ { F } = 0 . 8 6 ) .$ , while $\mathrm { { A U C } _ { \mathrm { { g l o b a l } } } }$ remains 0.84. However, the cross-group AUCs remain markedly different, with $\mathrm { x A U C } _ { M , F } = 0 .$ .53 and $\mathrm { x A U C } _ { F , M } =$ 0.98, despite identical subgroup AUCs. Specifically, $x A U C _ { M , F } = 0 .$ 53 indicates that male patients who die are only slightly more likely than chance to be “ranked” above female survivors, whereas $x A U C _ { F , M } = 0 . 9 8$ indicates near-perfect ranking of female patients who die above male survivors. Equality of subgroup AUCs therefore does not imply equality across all dimensions of discrimination fairness.

## 5 Discussion

## 5.1 Key finding: Non-collapsibility matters for fairness checks

When a metric is non-collapsible, several interpretation complications arise. The relationship between subgroup and overall performance can become obscured, as overall metrics can fail to reflect subgroup behaviour faithfully or to act as a linear combination of their constituent parts. This mismatch can create natural inconsistencies between subgroup level and population level results, potentially undermining confidence in fairness evaluations.

In some settings, non-collapsibility can also introduce spurious disparities between subgroups and overall estimates, generating differences driven by subgroup proportions or subgroup outcome prevalence, rather than by genuine model behaviour. It can also lead to counter-intuitive consequences for fairness interventions, since improvements in subgroup metrics do not necessarily translate into, and may even diminish, overall performance. Finally, because the overall AUC includes both within-group and cross-group pairwise comparisons, it may conflate discrimination with calibration differences across subgroups, further complicating interpretation in heterogeneous populations.

## 5.2 Recommendations

The findings presented in this paper suggest several recommendations for researchers and practitioners evaluating clinical prediction models, particularly in fairness contexts (Box 3). First, before conducting subgroup analyses, it is important to establish whether the metrics of interest are collapsible. When a metric is non-collapsible, discrepancies between subgroup level and overall performance may arise naturally, even when the model behaves consistently across groups, due to case-mix differences. Second, metric estimates should always be interpreted in relation to subgroup proportions, subgroup outcome prevalences, and complementary performance domains, such as calibration performance when examining the AUC, and vice-versa. Third, transparent reporting of subgroup composition and outcome distributions is therefore essential for evaluating both model performance and fairness, and adherence to TRIPOD+AI remains fundamental [14]. Finally, sufficient sample size is essential for subgroup analyses [29, 30, 31], as even collapsible metrics are subject to sampling variability and small samples can lead to chance differences between subgroup and overall performance that are difficult to interpret.

## Box 3: Key considerations and recommendations

Collapsibility properties of performance metrics should be considered when conducting subgroup analyses of performance of predictive AI.

The AUC, calibration intercept, the calibration slope, ECE, and Nagelkerke $\mathrm { R } ^ { 2 }$ are non-collapsible metrics, whereas the O:E ratio, logloss, Brier score, accuracy, F1-score, TPR, TNR, PPV, NPV, and net benefit are collapsible.

The AUC is a non-collapsible metric because under mixtures of subpopulations it decomposes into withinand cross-group AUC components. Its overall value cannot always be recovered as a weighted average of within-group AUCs. This may lead to counter-intuitive scenarios such as each subgroup AUC being smaller than the overall AUC.

AUC collapsibility holds when the distribution of estimated probabilities among individuals without the event are identical across subgroups. Non-collapsibility is therefore driven by distributional variations in estimated probabilities, arising from case-mix variation or miscalibration differences across subgroups.

Fairness evaluations based on differences between subgroups and overall performance, or across subgroups, are complex, as these quantities are not necessarily directly comparable, and differences may arise naturally.

Transparent reporting is therefore essential. In the context of non-collapsible metrics, we underscore the importance of reporting population characteristics, outcome distributions, and fairness-related design choices. Adherence to TRIPOD+AI promotes transparency and reproducibility in fairness evaluations [14].

## Acknowledgements

JM is funded by a Clarendon Fund Scholarship at the University of Oxford. GSC and RDR are supported by the National Institute for Health and Care Research (NIHR) Biomedical Research Centre: Birmingham at University Hospitals Birmingham NHS Foundation Trust and the University of Birmingham. GSC and RDR are NIHR Senior Investigators. BVC is supported by Research Foundation – Flanders (FWO) grant G097322N, Kom Op Tegen Kanker grant 13583, and KU Leuven Internal Funds grant C24M/20/064. The funders of the study had no role in study design, data collection, data analysis, interpretation of data, writing of the report, or decision to submit the article for publication. The views expressed are those of the authors and not necessarily represent the views of the funding agencies.

## References

[1] Janssens ACJ, Martens FK. Reflection on modern methods: Revisiting the area under the ROC Curve. International journal of epidemiology. 2020;49(4):1397-403.

[2] Van Calster B, Nieboer D, Vergouwe Y, De Cock B, Pencina MJ, Steyerberg EW. A calibration hierarchy for risk models was defined: from utopia to empirical data. Journal of clinical epidemiology. 2016;74:167-76.

[3] Huitfeldt A, Stensrud MJ, Suzuki E. On the collapsibility of measures of effect in the counterfactual causal framework. Emerging themes in epidemiology. 2019;16(1):1.

[4] Kallus N, Zhou A. The fairness of risk scores beyond classification: Bipartite ranking and the xauc metric. Advances in neural information processing systems. 2019;32.

[5] Van Calster B, Collins GS, Vickers AJ, Wynants L, Kerr KF, Barreñada L, et al. Evaluation of performance measures in predictive artificial intelligence models to support medical decisions: overview and guidance. The Lancet Digital Health. 2025.

[6] Matos J, Van Calster B, Celi LA, Dhiman P, Gichoya JW, Riley RD, et al. Critical Appraisal of Fairness Metrics in Clinical Predictive AI. The Lancet Digital Health. 2026.

[7] Denny M. The fallacy of the average: on the ubiquity, utility and continuing novelty of Jensen’s inequality. Journal of Experimental Biology. 2017;220(2):139-46.

[8] Blyth CR. On Simpson’s paradox and the sure-thing principle. Journal of the American Statistical Association. 1972;67(338):364-6.

[9] van der Meijden SL, Wang Y, Ng MY, Arbous SM, Geerts BF, Steyerberg EW, et al. Navigating fairness in artificial intelligence-based prediction models: theoretical constructs and practical applications. The Lancet Digital Health. 2026.

[10] Hernán MA, Clayton D, Keiding N. The Simpson’s paradox unraveled. International journal of epidemiology. 2011;40(3):780-5.

[11] Simpson EH. The interpretation of interaction in contingency tables. Journal of the Royal Statistical Society: Series B (Methodological). 1951;13(2):238-41.

[12] Messick DM, Van de Geer JP. A reversal paradox. Psychological Bulletin. 1981;90(3):582.

[13] Good IJ, Mittal Y. The amalgamation and geometry of two-by-two contingency tables. The Annals of Statistics. 1987:694-711.

[14] Collins GS, Moons KG, Dhiman P, Riley RD, Beam AL, Van Calster B, et al. TRIPOD+ AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. bmj. 2024;385:e078378.

[15] Gonçalves L, Subtil A, Oliveira MR, de Zea Bermudez P. ROC curve estimation: An overview. REVSTAT-Statistical journal. 2014;12(1):1-20.

[16] Faraggi D, Reiser B. Estimation of the area under the ROC curve. Stat Med. 2002 Oct;21(20):3093-106.

[17] Sudjianto A, Liu AJ. Decomposing Global AUC into Cluster-Level Contributions for Localized Model Diagnostics. arXiv preprint arXiv:250807495. 2025.

[18] Zhou XH, McClish DK, Obuchowski NA. Statistical Methods in Diagnostic Medicine. Wiley Series in Probability and Statistics. Wiley; 2009. Available from: https://books.google. com/books?id=ijN\_Dlx7wmoC.

[19] Austin PC, Steyerberg EW. Interpreting the concordance statistic of a logistic regression model: relation to the variance and odds ratio of a continuous explanatory variable. BMC medical research methodology. 2012;12(1):82.

[20] Kleinberg J, Mullainathan S, Raghavan M. Inherent trade-offs in the fair determination of risk scores. arXiv preprint arXiv:160905807. 2016.

[21] Vergouwe Y, Moons KG, Steyerberg EW. External validity of risk models: use of benchmark values to disentangle a case-mix effect from incorrect coefficients. American journal of epidemiology. 2010;172(8):971-80.

[22] Tseng AS, Shelly-Cohen M, Attia IZ, Noseworthy PA, Friedman PA, Oh JK, et al. Spectrum bias in algorithms derived by artificial intelligence: a case study in detecting aortic stenosis using electrocardiograms. European Heart Journal-Digital Health. 2021;2(4):561-7.

[23] van Klaveren D, Gönen M, Steyerberg EW, Vergouwe Y. A new concordance measure for risk prediction models in external validation settings. Statistics in medicine. 2016;35(23):4136-52.

[24] de Jong VM, Hoogland J, Moons KG, Riley RD, Nguyen TL, Debray TP. Propensity-based standardization to enhance the validation and interpretation of prediction model discrimination for a target population. Statistics in Medicine. 2023;42(19):3508-28.

[25] Martens FK, Tonk EC, Kers JG, Janssens ACJ. Small improvement in the area under the receiver operating characteristic curve indicated small changes in predicted risks. Journal of clinical epidemiology. 2016;79:159-64.

[26] Pencina MJ, Fine JP, D’Agostino Sr RB. Discrimination slope and integrated discrimination improvement–properties, relationships and impact of calibration. Statistics in Medicine. 2017;36(28):4482-90.

[27] Sadatsafavi M, Saha-Chaudhuri P, Petkau J. Model-based ROC curve: examining the effect of case mix and model calibration on the ROC plot. Medical Decision Making. 2022;42(4):487-99.

[28] Nashef SA, Roques F, Sharples LD, Nilsson J, Smith C, Goldstone AR, et al. Euroscore ii. European journal of cardio-thoracic surgery. 2012;41(4):734-45.

[29] Riley RD, Collins GS, Whittle R, Archer L, Snell KI, Dhiman P, et al. A decomposition of Fisher’s information to inform sample size for developing or updating fair and precise clinical prediction models for individual risk—part 1: binary outcomes. Diagnostic and prognostic research. 2025;9(1):14.

[30] Riley RD, Collins GS, Archer L, Whittle R, Legha A, Kirton L, et al. A decomposition of Fisher’s information to inform sample size for developing or updating fair and precise clinical prediction models—part 2: time-to-event outcomes. Diagnostic and Prognostic Research. 2025;9(1):33.

[31] Whittle R, Riley RD, Archer L, Collins GS, Legha A, Snell KI, et al. A decomposition of Fisher’s information to inform sample size for developing or updating fair and precise clinical prediction models–part 3: continuous outcomes. Diagnostic and Prognostic Research. 2026;10(1):11.

## A Appendix

## A.1 Collapsibility Proofs

We investigate the performance measures summarised in Table 1.

## A.1.1 O:E Ratio

The observed-to-expected ratio (O:E) compares the number of observed outcomes to the expected number of outcomes under the model.

$$
\mathrm { O : E } \triangleq \frac { O } { E } , \qquad O = \sum _ { i = 1 } ^ { N } y _ { i } , \quad E = \sum _ { i = 1 } ^ { N } \hat { p } _ { i } ,
$$

with

$$
\left( \operatorname { O } : \operatorname { E } \right) _ { g } \triangleq \frac { O _ { g } } { E _ { g } } , \quad O _ { g } = \sum _ { i \in g } y _ { i } , \quad E _ { g } = \sum _ { i \in g } \hat { p } _ { i } .
$$

Substituting group-wise counts:

$$
{ \begin{array} { r l } & { \mathbf { 0 } : \operatorname { E } = { \frac { O _ { A } + O _ { B } } { E _ { A } + E _ { B } } } } \\ & { \qquad = { \frac { 1 } { E _ { A } + E _ { B } } } { \Big [ } \underbrace { O _ { A } } _ { E _ { A } { \mathrm { ~ ( O : E ) } } _ { A } } + \underbrace { O _ { B } } _ { E _ { B } { \mathrm { ~ ( O : E ) } } _ { B } } { \Big ] } } \\ & { \qquad = { \frac { E _ { A } } { E _ { A } + E _ { B } } } \left( \mathbf { 0 } : \mathbf { E } \right) _ { A } + { \frac { E _ { B } } { E _ { A } + E _ { B } } } \left( \mathbf { 0 } : \mathbf { E } \right) _ { B } \ \boxed { \mathbf { \Sigma } } } \end{array} }
$$

Thus the O:E ratio satisfies 1 with weights

$$
w _ { g } = \frac { E _ { g } } { \sum _ { h } E _ { h } } .
$$

Therefore, O:E is collapsible with weights proportional to the group-expected counts under the model.

## A.1.2 Calibration Intercept

Calibration intercept α is usually defined in terms of:

$$
\mathrm { l o g i t } ( y ) = \alpha + 1 \times \mathrm { l o g i t } ( \hat { p } ) .
$$

Let groups $g \in \{ A , B \}$ have sizes $n _ { g }$ and weights $w _ { g } = n _ { g } / n$ with $n = n _ { A } + n _ { B }$ . For observations $( y _ { i } ^ { g } , \bar { p } _ { i } ^ { g } )$ with $0 < \hat { p } _ { i } ^ { g } < 1$ , define

$$
\begin{array} { c c c } { { \displaystyle \mathrm { e x p i t } ( t ) = \frac { 1 } { 1 + e ^ { - t } } , } } & { { \displaystyle \mathrm { l o g i t } ( p ) = \log \Big ( \frac { p } { 1 - p } \Big ) , } } & { { \displaystyle u _ { i } ^ { g } ( \alpha ) = \mathrm { e x p i t } \big ( \alpha + \log \mathrm { i t } ( \hat { p } _ { i } ^ { g } ) \big ) , } } \\ { { } } & { { } } & { { f _ { g } ( \alpha ) = \displaystyle \frac { 1 } { n _ { g } } \sum _ { i = 1 } ^ { n _ { g } } u _ { i } ^ { g } ( \alpha ) . } } \end{array}
$$

The calibration intercept $\alpha _ { g }$ solves

$$
f _ { g } ( \alpha _ { g } ) = \bar { y } _ { g } , \qquad \bar { y } _ { g } = \frac { 1 } { n _ { g } } \sum _ { i = 1 } ^ { n _ { g } } y _ { i } ^ { g } .
$$

For the pooled sample one can define

$$
f ( \alpha ) = \frac { 1 } { n } \sum _ { g \in \{ A , B \} } \sum _ { i = 1 } ^ { n _ { g } } u _ { i } ^ { g } ( \alpha ) = w _ { A } f _ { A } ( \alpha ) + w _ { B } f _ { B } ( \alpha ) ,
$$

and

$$
\bar { y } = \frac { 1 } { n } \sum _ { g , i } y _ { i } ^ { g } = w _ { A } \bar { y } _ { A } + w _ { B } \bar { y } _ { B } .
$$

Lemma 1 (estimating equation). The pooled calibration intercept $\alpha _ { \mathrm { a l l } }$ is the unique solution of

$$
\begin{array} { r } { f ( \alpha ) = \bar { y } . } \end{array}
$$

Proof. The log-likelihood is

$$
\ell ( \boldsymbol { \alpha } ) = \sum _ { g , i } \Big \{ y _ { i } ^ { g } \log u _ { i } ^ { g } ( \boldsymbol { \alpha } ) + ( 1 - y _ { i } ^ { g } ) \log \big ( 1 - u _ { i } ^ { g } ( \boldsymbol { \alpha } ) \big ) \Big \} ,
$$

and the score, the derivative of the log-likelihood with respect to $\alpha ,$ is

$$
S ( \alpha ) = \ell ^ { \prime } ( \alpha ) = \sum _ { g , i } \big ( y _ { i } ^ { g } - u _ { i } ^ { g } ( \alpha ) \big ) ,
$$

since $\begin{array} { r } { \frac { \partial u _ { i } ^ { g } ( \alpha ) } { \partial \alpha } = u _ { i } ^ { g } ( \alpha ) \big ( 1 - u _ { i } ^ { g } ( \alpha ) \big ) } \end{array}$ . The maximum likelihood estimator satisfies the score equation $S ( \alpha ) = 0 ;$ , that is

$$
\frac { 1 } { n } \sum _ { g , i } u _ { i } ^ { g } ( \alpha ) = \frac { 1 } { n } \sum _ { g , i } y _ { i } ^ { g } \quad \Longleftrightarrow \quad f ( \alpha ) = \bar { y } .
$$

Uniqueness follows because $\ell ( \alpha )$ is strictly concave in α and admits a unique maximiser, equivalently a unique root of $S ( \alpha ) = 0$ □

Lemma 2 (monotonicity). Each $f _ { g }$ is continuous and strictly increasing; hence $f$ is continuous and strictly increasing.

Proof. $\begin{array} { r } { f _ { g } ^ { \prime } ( \alpha ) = \frac { 1 } { n _ { q } } \sum _ { i } u _ { i } ^ { g } ( \alpha ) \{ 1 - u _ { i } ^ { g } ( \alpha ) \} > 0 } \end{array}$ . Linearity gives the result for $f .$

Proposition (convex bounds). Let $\alpha _ { - } = \operatorname* { m i n } \{ \alpha _ { A } , \alpha _ { B } \}$ and $\alpha _ { + } = \operatorname* { m a x } \{ \alpha _ { A } , \alpha _ { B } \}$ . Then the overall intercept satisfies

$$
\alpha _ { \mathrm { a l l } } \in [ \alpha _ { - } , \alpha _ { + } ] .
$$

Proof. Assume, without loss of generality, that $\alpha _ { A } \leq \alpha _ { B }$ . Then

$$
f ( \alpha _ { A } ) = w _ { A } f _ { A } ( \alpha _ { A } ) + w _ { B } f _ { B } ( \alpha _ { A } ) \leq w _ { A } \bar { y } _ { A } + w _ { B } \bar { y } _ { B } = \bar { y } ,
$$

since $f _ { B }$ is increasing and $\alpha _ { A } \leq \alpha _ { B } \Longrightarrow f _ { B } ( \alpha _ { A } ) \leq f _ { B } ( \alpha _ { B } )$ . Similarly,

$$
\begin{array} { r } { f ( \alpha _ { B } ) = w _ { A } f _ { A } ( \alpha _ { B } ) + w _ { B } f _ { B } ( \alpha _ { B } ) \ge w _ { A } \bar { y } _ { A } + w _ { B } \bar { y } _ { B } = \bar { y } , } \end{array}
$$

since $f _ { A }$ is increasing and $\alpha _ { B } \geq \alpha _ { A }$ . By Lemma 2 and the intermediate value theorem, there is a unique $\alpha _ { \mathrm { a l l } } \ \in \ \left[ \alpha _ { A } , \alpha _ { B } \right]$ with $f ( \alpha _ { \mathrm { a l l } } ) = \bar { y }$ . Hence, $\begin{array} { l l l l } { f ( \alpha _ { A } ) } & { \leq } & { \bar { y } } & { \leq } & { f ( \alpha _ { B } ) } \end{array}$ , and for every $\alpha _ { A } < \alpha _ { \mathrm { a l l } } < \alpha _ { B }$ we have $f ( \alpha _ { A } ) \ \leq \ f ( \alpha _ { \mathrm { a l l } } ) \ \leq \ f ( \alpha _ { B } )$ □

Remark. The only property established is the bounding relation $\alpha _ { \mathrm { a l l } } \in [ \alpha _ { A } , \alpha _ { B } ]$ . With this, we are not claiming that the calibration intercept is necessarily collapsible: in general, $\alpha _ { \mathrm { a l l } }$ is not expected to be expressed as a convex or linear combination of $\alpha _ { A }$ and $\alpha _ { B }$

## A.1.3 Calibration Slope

The calibration slope $\beta$ is estimated from the regression

$$
\begin{array} { r } { \mathrm { l o g i t } ( y _ { i } ) = \alpha + \beta \log \mathrm { i t } ( \hat { p } _ { i } ) + \varepsilon _ { i } , } \end{array}
$$

with ideal calibration corresponding to $\beta = 1$

The calibration slope β is estimated as a regression coefficient from a logistic regression model fitted to the entire dataset. Logistic regression coefficients are generally non-collapsible, meaning that the coefficient estimated from the pooled data is not, in general, a weighted average of coefficients estimated within subgroups. Consequently, the calibration slope is also expected to be non-collapsible.

## A.1.4 Expected Calibration Error (ECE)

The expected calibration error measures the average absolute deviation between predicted and observed outcome rates over bins of estimated probability.

$$
\mathrm { E C E } \triangleq \sum _ { b = 1 } ^ { B } \frac { n _ { b } } { N } \Big | \frac { 1 } { n _ { b } } \sum _ { i \in b } y _ { i } - \frac { 1 } { n _ { b } } \sum _ { i \in b } \hat { p } _ { i } \Big | ,
$$

$$
\mathrm { w i t h ~ \ E C E } _ { g } \triangleq \sum _ { b = 1 } ^ { B } \frac { n _ { g b } } { N _ { g } } \Big | \frac { 1 } { n _ { g b } } \sum _ { i \in ( g , b ) } y _ { i } - \frac { 1 } { n _ { g b } } \sum _ { i \in ( g , b ) } \hat { p } _ { i } \Big | .
$$

Although ECE is an average across bins, the absolute value inside the bin definition prevents linear aggregation across groups:

$$
\mathrm { E C E } \neq \sum _ { g } \frac { N _ { g } } { N } \mathrm { E C E } _ { g } \quad \mathrm { i n ~ g e n e r a l } .
$$

Thus ECE is expected to be non-collapsible.

## A.1.5 Logloss / Cross-Entropy

Logloss (cross-entropy) is the empirical average of the per-instance negative log-likelihood; it is collapsible.

$$
\mathrm { L o g L o s s } \triangleq \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \Big [ - y _ { i } \log \hat { p } _ { i } - ( 1 - y _ { i } ) \log ( 1 - \hat { p } _ { i } ) \Big ] ,
$$

and

$$
\operatorname { L o g L o s s } _ { g } \triangleq \frac { 1 } { N _ { g } } \sum _ { i \in g } \Big [ - y _ { i } \log \hat { p } _ { i } - ( 1 - y _ { i } ) \log ( 1 - \hat { p } _ { i } ) \Big ] .
$$

Partitioning by groups:

$$
{ \begin{array} { r l } { { \mathrm { L o g L o s s } } = { \cfrac { 1 } { N } } \displaystyle \sum _ { g } \sum _ { i \in g } { \Big [ } - y _ { i } \log { \hat { p } } _ { i } - ( 1 - y _ { i } ) \log ( 1 - { \hat { p } } _ { i } ) { \Big ] } } & { } \\ { = \displaystyle \sum _ { g } { \frac { N _ { g } } { N } } \underbrace { { \frac { 1 } { N _ { g } } } \sum _ { i \in g } { \Big [ } - y _ { i } \log { \hat { p } } _ { i } - ( 1 - y _ { i } ) \log ( 1 - { \hat { p } } _ { i } ) { \Big ] } } _ { \mathrm { L o g L o s s } _ { g } } = \sum _ { g } p ^ { g } { \mathrm { L o g L o s s } } _ { g } } & { { \boxed { 1 \times { \mathord { \operatorname { l o g s s } } _ { g } } } } } \end{array} }
$$

Thus Logloss satisfies 1 with weights $w _ { g } = p ^ { g } = N _ { g } / N$

## A.1.6 Brier Score

The Brier score is the average squared error; it is collapsible.

$$
{ \mathrm { B r i e r } } \triangleq { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } \left( { \hat { p } } _ { i } - y _ { i } \right) ^ { 2 } \quad { \mathrm { a n d } } \quad { \mathrm { B r i e r } } _ { g } \triangleq { \frac { 1 } { N _ { g } } } \sum _ { i \in g } \left( { \hat { p } } _ { i } - y _ { i } \right) ^ { 2 } .
$$

Partitioning by groups:

$$
{ \mathrm { B r i e r } } = { \frac { 1 } { N } } \sum _ { g } \sum _ { i \in g } \left( { \hat { p } } _ { i } - y _ { i } \right) ^ { 2 } = \sum _ { g } { \frac { N _ { g } } { N } } \underbrace { { \frac { 1 } { N _ { g } } } \sum _ { i \in g } \left( { \hat { p } } _ { i } - y _ { i } \right) ^ { 2 } } _ { \mathrm { B r i e r } _ { g } } = \sum _ { g } p ^ { g } \ { \mathrm { B r i e r } } _ { g } \quad \boxed { 1 \leq N }
$$

Thus the Brier score satisfies 1 with weights $w _ { g } = p ^ { g }$

## A.1.7 Nagelkerke $R ^ { 2 }$

Nagelkerke’s $R ^ { 2 }$ (for binary outcomes) rescales Cox–Snell’s $R ^ { 2 }$ to [0, 1]:

$$
R _ { \mathrm { N } } ^ { 2 } = { \frac { 1 - \exp { \Big ( } { \frac { 2 } { N } } \left( \ell _ { 0 } - \ell _ { 1 } \right) { \Big ) } } { 1 - \exp { \Big ( } { \frac { 2 } { N } } \ell _ { 0 } { \Big ) } } } , \quad \ell _ { 1 } = \sum _ { i = 1 } ^ { N } \log p _ { \theta } ( y _ { i } \mid x _ { i } ) , \quad \ell _ { 0 } = \sum _ { i = 1 } ^ { N } \log p _ { \theta _ { 0 } } ( y _ { i } ) .
$$

Here $\ell _ { 1 }$ denotes the log-likelihood of the fitted model with parameter $\theta ,$ and $\ell _ { 0 }$ the log-likelihood of the null model (intercept-only, with parameter $\theta _ { 0 } )$ . The term $p _ { \theta } ( y _ { i } \mid x _ { i } )$ is the model-estimated probability of observing outcome $y _ { i }$ given covariates $x _ { i }$ under parameter θ. Group-wise versions are then:

$$
R _ { \mathrm { N } , g } ^ { 2 } = \frac { 1 - \exp { \left( \frac { 2 } { N _ { g } } \left( \ell _ { 0 g } - \ell _ { 1 g } \right) \right) } } { 1 - \exp { \left( \frac { 2 } { N _ { g } } \ell _ { 0 g } \right) } } , \quad \ell _ { 1 } = \sum _ { g } \ell _ { 1 g } , ~ \ell _ { 0 } = \sum _ { g } \ell _ { 0 g } , ~ N = \sum _ { g } N _ { g } .
$$

In general,

$$
R _ { \mathrm { N } } ^ { 2 } = \frac { 1 - \exp \Bigl ( \frac { 2 } { N } \sum _ { g } ( \ell _ { 0 g } - \ell _ { 1 g } ) \Bigr ) } { 1 - \exp \Bigl ( \frac { 2 } { N } \sum _ { g } \ell _ { 0 g } \Bigr ) } \neq \sum _ { g } w _ { g } \frac { 1 - \exp \Bigl ( \frac { 2 } { N _ { g } } \bigl ( \ell _ { 0 g } - \ell _ { 1 g } \bigr ) \Bigr ) } { 1 - \exp \Bigl ( \frac { 2 } { N _ { g } } \ell _ { 0 g } \Bigr ) } .
$$

Because the mapping $( \ell _ { 0 } , \ell _ { 1 } , N ) \mapsto R _ { \mathrm { N } } ^ { 2 }$ applies nonlinear exponentials with different normalisations $( 1 / N \mathrm { v s } 1 / N _ { g } )$ to sums of log-likelihoods, the overall index is not a convex combination of group indices except under potentially restrictive special cases. Therefore, Nagelkerke $R ^ { 2 }$ does not satisfy 1 in general and is expected to be non-collapsible. If an amalgamation paradox is observed to occur, the measure is proven to be non-collapsible.

## A.1.8 Classification accuracy

Classification accuracy is collapsible with weights given by the group sizes.

and

$$
\begin{array} { l } { \displaystyle \mathrm { A c c } \triangleq \frac { T P + T N } { N } , \qquad N = T P + T N + F P + F N , } \\ { \displaystyle \mathrm { A c c } _ { g } \triangleq \frac { T P _ { g } + T N _ { g } } { N _ { g } } . } \end{array}
$$

Substituting group-wise counts:

$$
\begin{array} { r l } & { \mathrm { A c c } = \frac { ( T P _ { A } + T N _ { A } ) + ( T P _ { B } + T N _ { B } ) } { N _ { A } + N _ { B } } } \\ & { \quad \quad = \frac { 1 } { N _ { A } + N _ { B } } \Big [ \underbrace { T P _ { A } + T N _ { A } } _ { N _ { A } \mathrm { A c c } _ { A } } + \underbrace { T P _ { B } + T N _ { B } } _ { N _ { B } \mathrm { A c c } _ { B } } \Big ] } \\ & { \quad \quad = \frac { N _ { A } } { N _ { A } + N _ { B } } \mathrm { A c c } _ { A } + \frac { N _ { B } } { N _ { A } + N _ { B } } \mathrm { A c c } _ { B } \ \sqsupset \  } \end{array}
$$

Thus accuracy satisfies 1 with weights $w _ { g } = N _ { g } / N = p ^ { g }$

## A.1.9 F1-score

F1-score is the harmonic mean of sensitivity and precision, which can also be defined as:

$$
\mathrm { F } 1 \triangleq \frac { 2 T P } { 2 T P + F P + F N } \quad \mathrm { a n d } \quad \mathrm { F } 1 _ { g } \triangleq \frac { 2 T P _ { g } } { 2 T P _ { g } + F P _ { g } + F N _ { g } } .
$$

Substituting group-wise counts and writing the overall F1 as a ratio of linear sums:

$$
\begin{array} { r l } & { \mathrm { F 1 } = \frac { 2 T P _ { A } } { \left( 2 T P _ { A } + F P _ { A } + F N _ { A } \right) } + \frac { 2 T P _ { A } } { \left( 2 T P _ { B } + F P _ { B } + F N _ { B } \right) } } \\ & { = \underbrace { \left( \frac { 2 T P _ { A } } { 2 T P _ { A } + F P _ { A } + F N _ { A } } \right) } _ { F P _ { A } } \cdot \frac { 2 T P _ { A } + F N _ { A } + F N _ { A } } { \sum _ { g } \left( 2 T P _ { g } + F P _ { g } + F N _ { g } \right) } } \\ & { \quad + \underbrace { \left( \frac { 2 T P _ { B } } { 2 T P _ { B } + F N _ { B } } \right) } _ { F 1 _ { B } } \cdot \frac { 2 T P _ { B } } { \sum _ { d } \left( 2 T P _ { f } + F P _ { g } + F N _ { B } \right) } } \\ & { = \underbrace { \sum _ { g } 2 T P _ { g } + F N _ { g } } _ { g \in \{ A , B \} } + \frac { 2 T P _ { B } } { F 1 _ { B } } \mathrm { F 1 } _ { g } } \end{array}
$$

Thus F1 satisfies 1 with weights $w _ { g } = \frac { 2 T P _ { g } + F P _ { g } + F N _ { g } } { \sum _ { h } ( 2 T P _ { h } + F P _ { h } + F N _ { h } ) } .$

## A.1.10 Sensitivity / Recall / TPR

Sensitivity (TPR) is collapsible with weights proportional to the group-specific prevalence of the outcome.

$$
\begin{array} { r } { \mathrm { T P R } \triangleq \displaystyle \frac { T P } { N ^ { + } } , \qquad N ^ { + } = T P + F N , } \\ { \mathrm { a n d } \quad \mathrm { T P R } _ { g } \triangleq \displaystyle \frac { T P _ { g } } { N _ { g } ^ { + } } , \quad N _ { g } ^ { + } = T P _ { g } + F N _ { g } . } \end{array}
$$

Substituting group-wise counts:

$$
\begin{array} { r l } & { \mathrm { T P R } = \frac { T P _ { A } + T P _ { B } } { N _ { A } ^ { + } + N _ { B } ^ { + } } } \\ & { \quad \quad = \frac { 1 } { N _ { A } ^ { + } + N _ { B } ^ { + } } \Big [ \underbrace { T P _ { A } } _ { N _ { A } ^ { + } \mathrm { ~ T P R } _ { A } } + \underbrace { T P _ { B } } _ { N _ { B } ^ { + } \mathrm { ~ T P R } _ { B } } \Big ] } \\ & { \quad \quad = \frac { N _ { A } ^ { + } } { N ^ { + } } \mathrm { T P R } _ { A } + \frac { N _ { B } ^ { + } } { N ^ { + } } \mathrm { T P R } _ { B } \ \sqcap \ } \end{array}
$$

Thus TPR satisfies 1 with weights

$$
w _ { g } = \frac { N _ { g } ^ { + } } { N ^ { + } } = \frac { p ^ { g } \pi _ { g } } { \pi } ,
$$

where $p ^ { g } = N _ { g } / N , \pi _ { g } = N _ { g } ^ { + } / N _ { g }$ and $\pi = N ^ { + } / N .$

## A.1.11 Specificity / TNR

Specificity (TNR) is collapsible with weights proportional to the group-specific rate of actual negatives.

$$
\begin{array} { r } { \mathrm { T N R } \triangleq \displaystyle \frac { T N } { N ^ { - } } , \qquad N ^ { - } = T N + F P , } \\ { \mathrm { a n d } \quad \mathrm { T N R } _ { g } \triangleq \displaystyle \frac { T N _ { g } } { N _ { g } ^ { - } } , \quad N _ { g } ^ { - } = T N _ { g } + F P _ { g } . } \end{array}
$$

Substituting group-wise counts:

$$
\begin{array} { r l } & { \mathrm { T N R } = \frac { T N _ { A } + T N _ { B } } { N _ { A } ^ { - } + N _ { B } ^ { - } } } \\ & { \quad \quad = \frac { 1 } { N _ { A } ^ { - } + N _ { B } ^ { - } } \Big [ \underbrace { T N _ { A } } _ { N _ { A } ^ { - } \mathrm { ~ T N R } _ { A } } + \underbrace { T N _ { B } } _ { N _ { B } ^ { - } \mathrm { ~ T N R } _ { B } } \Big ] } \\ & { \quad \quad = \frac { N _ { A } ^ { - } } { N ^ { - } } \mathrm { T N R } _ { A } + \frac { N _ { B } ^ { - } } { N ^ { - } } \mathrm { T N R } _ { B } \ \sqcap \ \Omega } \end{array}
$$

Thus TNR satisfies 1 with weights

$$
w _ { g } = \frac { N _ { g } ^ { - } } { N ^ { - } } = \frac { p ^ { g } ( 1 - \pi _ { g } ) } { 1 - \pi } .
$$

## A.1.12 PPV / Precision

PPV is collapsible with weights proportional to the group-specific rate of predicted positives.

$$
\begin{array} { r } { \mathrm { P P V } \triangleq \displaystyle \frac { T P } { \hat { N } ^ { + } } , \qquad \hat { N } ^ { + } = T P + F P , } \\ { \mathrm { a n d } \quad \mathrm { P P V } _ { g } \triangleq \displaystyle \frac { T P _ { g } } { \hat { N } _ { g } ^ { + } } , \quad \hat { N } _ { g } ^ { + } = T P _ { g } + F P _ { g } . } \end{array}
$$

Substituting group-wise counts:

$$
\begin{array} { r l } & { \mathrm { P P V } = \frac { T P _ { A } + T P _ { B } } { \hat { N } _ { A } ^ { + } + \hat { N } _ { B } ^ { + } } } \\ & { \quad \quad = \frac { 1 } { \hat { N } _ { A } ^ { + } + \hat { N } _ { B } ^ { + } } \Big [ \underbrace { T P _ { A } } _ { \hat { N } _ { A } ^ { + } \mathrm { P P V } _ { A } } + \underbrace { T P _ { B } } _ { \hat { N } _ { B } ^ { + } \mathrm { P P V } _ { B } } \Big ] } \\ & { \quad \quad \quad = \frac { \hat { N } _ { A } ^ { + } } { \hat { N } ^ { + } } \mathrm { P P V } _ { A } + \frac { \hat { N } _ { B } ^ { + } } { \hat { N } ^ { + } } \mathrm { P P V } _ { B } \ \boxtimes } \end{array}
$$

Thus PPV satisfies 1 with weights

$$
w _ { g } = \frac { \hat { N } _ { g } ^ { + } } { \hat { N } ^ { + } } = \frac { p ^ { g } \hat { \pi } _ { g } } { \hat { \pi } } ,
$$

where $\hat { \pi } _ { g } = \hat { N } _ { g } ^ { + } / N _ { g }$ is the group-specific predicted positive rate, and $\hat { \pi } = \hat { N } ^ { + } / N$

## A.1.13 NPV

NPV is collapsible with weights proportional to the group-specific rate of predicted negatives.

$$
\begin{array} { r } { \mathrm { N P V } \triangleq \displaystyle \frac { T N } { \hat { N } ^ { - } } , \qquad \hat { N } ^ { - } = T N + F N , } \\ { \mathrm { a n d } \quad \mathrm { N P V } _ { g } \triangleq \displaystyle \frac { T N _ { g } } { \hat { N } _ { g } ^ { - } } , \quad \hat { N } _ { g } ^ { - } = T N _ { g } + F N _ { g } . } \end{array}
$$

Substituting group-wise counts:

$$
\begin{array} { r l } & { \mathrm { N P V } = \frac { T N _ { A } + T N _ { B } } { \hat { N } _ { A } ^ { - } + \hat { N } _ { B } ^ { - } } } \\ & { \quad \quad = \frac { 1 } { \hat { N } _ { A } ^ { - } + \hat { N } _ { B } ^ { - } } \Big [ \underbrace { T N _ { A } } _ { \hat { N } _ { A } ^ { - } \mathrm { ~ N P V } _ { A } } + \underbrace { T N _ { B } } _ { \hat { N } _ { B } ^ { - } \mathrm { ~ N P V } _ { B } } \Big ] } \\ & { \quad \quad \quad = \frac { \hat { N } _ { A } ^ { - } } { \hat { N } ^ { - } } \mathrm { ~ N P V } _ { A } + \frac { \hat { N } _ { B } ^ { - } } { \hat { N } ^ { - } } \mathrm { ~ N P V } _ { B } \perp } \end{array}
$$

Thus NPV satisfies 1 with weights

$$
w _ { g } = \frac { \hat { N } _ { g } ^ { - } } { \hat { N } ^ { - } } = \frac { p ^ { g } ( 1 - \hat { \pi } _ { g } ) } { 1 - \hat { \pi } } ,
$$

where $\hat { \pi } _ { g } = \hat { N } _ { g } ^ { + } / N _ { g }$ and $\hat { \pi } = \hat { N } ^ { + } / N$

## A.1.14 Net Benefit

Net Benefit $( N B )$ is often used in decision curve analysis as a summary of clinical usefulness. It combines true positives and false positives into a single measure, weighted by the relative harms of false positives at a given threshold, t. Here we show that $N B$ is a collapsible measure.

$$
\begin{array} { r } { \mathrm { N B } \triangleq \displaystyle \frac { T P } { N } - c \displaystyle \frac { F P } { N } , \qquad c = t ( 1 - t ) , } \\ { \mathrm { a n d } \quad \mathrm { N B } _ { g } \triangleq \displaystyle \frac { T P _ { g } } { N _ { g } } - c \displaystyle \frac { F P _ { g } } { N _ { g } } . \qquad } \end{array}
$$

Substituting group-wise counts into the definition:

$$
\begin{array} { r l } & { \mathrm { N B } = \frac { T P _ { A } + T P _ { R } } { N } - c \frac { F P _ { A } + F P _ { R } } { N } } \\ & { \quad = \frac { 1 } { N } \Big [ ( T P _ { A } - c F P _ { A } ) + ( T P _ { B } - c F P _ { B } ) \Big ] } \\ & { \quad = \frac { 1 } { N } \Big [ \underbrace { T P _ { A } - c F P _ { A } } _ { N _ { A } \mathrm { N B } _ { A } } + \underbrace { T P _ { B } - c F P _ { B } } _ { N _ { B } \mathrm { N B } _ { B } } \Big ] } \\ & { \quad = \frac { N _ { A } } { N } \mathrm { N B } _ { A } + \frac { N _ { B } } { N } \mathrm { N B } _ { B } } \\ & { \quad = p ^ { A } \mathrm { N B } _ { A } + p ^ { B } \mathrm { N B } _ { B } = \underbrace { p ^ { \prime } } _ { g \in \{ A , B \} } p ^ { \prime } \mathrm { N B } _ { g } \subseteq } \end{array}
$$

Thus NB satisfies 1 with weights $w _ { g } = p ^ { g }$ , proving that NB is collapsible.

## A.2 Non-collapsibilty Counterexamples

## A.2.1 Calibration Slope

We here present a counterexample showing that the calibration slope is not a collapsible measure with respect to a grouping variable. We use two subpopulations of equal size $( n = 1 , 0 0 0$ in each subgroup). For subgroup A a single predictor $X _ { A }$ is drawn from a standard normal distribution,

$$
X _ { A } \sim { \mathcal { N } } ( 0 , 1 ) ,
$$

and the event probability is defined by a logistic model,

$$
P ( Y _ { A } = 1 \mid X _ { A } = x ) = \frac { 1 } { 1 + \exp ( - 2 x ) } .
$$

Binary outcomes $Y _ { A }$ are then sampled from this Bernoulli model. For subgroup B the same logistic relationship is used but the predictor distribution is shifted,

$$
X _ { B } \sim { \mathcal { N } } ( 2 , 1 ) , \qquad P ( Y _ { B } = 1 \mid X _ { B } = x ) = { \frac { 1 } { 1 + \exp ( - 2 x ) } } ,
$$

and binary outcomes $Y _ { B }$ are drawn accordingly.

Pooling the two groups changes the distribution of the linear predictor and the outcome in a nonlinear way, producing a overall calibration slope that is not a weighted average of the subgroup-specific slopes.

Table 3: Toy example, calibration slope computed separately by subgroup and for the pooled sample.
<table><tr><td>Subgroup</td><td>n</td><td>Calibration slope</td></tr><tr><td>A</td><td>1,000</td><td>0.934</td></tr><tr><td>B</td><td>1,000</td><td>0.963</td></tr><tr><td>Population</td><td>2,000</td><td>0.979</td></tr></table>

If the calibration slope were collapsible then the overall slope would equal a simple weighted average of the subgroup slopes, where the weights depend only on quantities such as sample size or outcome prevalence. Table 3 shows that this is not the case: the overall slope (0.979) exceeds both subgroup values (0.934 and 0.963), so it cannot be written as a linear combination of them.

## A.2.2 Nagelkerke $R ^ { 2 }$

We now show by counterexample that Nagelkerke’s $R ^ { 2 }$ is not a collapsible measure. Data are generated similarly as in the calibration slope example. When the two groups are pooled, this heterogeneity in X affects the null model likelihood and the fitted model likelihood in a nonlinear way, which leads to a overall Nagelkerke $R ^ { 2 }$ that is not a weighted average of the subgroup $R ^ { 2 }$ values.

Table 4: Toy example, Nagelkerke $R ^ { 2 }$ computed separately by subgroup and for the pooled sample.
<table><tr><td>Subgroup</td><td> $n$ </td><td> $\mathrm { R ^ { 2 } }$ </td></tr><tr><td>A</td><td>1,000</td><td>0.481</td></tr><tr><td>B</td><td>1,000</td><td>0.440</td></tr><tr><td>Population</td><td>2,000</td><td>0.623</td></tr></table>

If Nagelkerke $R ^ { 2 }$ were collapsible with respect to the grouping variable then the overall $R ^ { 2 }$ would be a simple weighted average of the subgroup $R ^ { 2 }$ values (weights depending only on sample sizes or outcome prevalences). Table 4 shows this is not true for the toy data: the overall $R ^ { 2 }$ (0.623) is larger than both subgroup values (0.481 and 0.440).

## A.2.3 Expected Calibration Error (ECE)

We present a toy example showing how calibration assessed by the ECE can differ between subgroups and the population. We use two subpopulations of equal size $( n = 1 , 0 0 0$ in each subgroup). For subgroup A a single predictor $X _ { A }$ is drawn from a normal distribution,

$$
X _ { A } \sim { \mathcal { N } } ( 0 , 1 ) ,
$$

and the event probability is defined by a logistic model,

$$
P ( Y _ { A } = 1 \mid X _ { A } = x ) = \frac { 1 } { 1 + \exp { ( - \left( 0 . 7 x \right) ) } } .
$$

Binary outcomes $Y _ { A }$ are then sampled from this Bernoulli model. For subgroup $B$ the same logistic relationship is used but the predictor distribution is shifted

$$
X _ { B } \sim \mathcal { N } ( 2 , 1 ) , \qquad P ( Y _ { B } = 1 \mid X _ { B } = x ) = \frac { 1 } { 1 + \exp \big ( - ( 0 . 7 x ) \big ) } ,
$$

and binary outcomes $Y _ { B }$ are drawn accordingly. Estimated probabilities $\hat { p }$ are given by the model logistic probabilities and ECE is computed using quantile (equal-sized) bins with 10 bins.

Pooling the two groups changes the empirical relationship between estimated probabilities and observed outcomes in a nonlinear way.

Table 5: Toy example, ECE computed separately by subgroup and for the pooled sample.
<table><tr><td>Subgroup</td><td>n</td><td>ECE</td></tr><tr><td>A</td><td>1,000</td><td>0.0378</td></tr><tr><td>B</td><td>1,000</td><td>0.0379</td></tr><tr><td>Population</td><td>2,000</td><td>0.0297</td></tr></table>

If the ECE were collapsible then the overall ECE would equal a simple weighted average of the subgroup ECEs with weights depending only on quantities such as sample size or outcome prevalence. Table 5 shows that this is not the case: the overall ECE (0.0297) differs from the subgroup values (0.0378 and 0.0379), so it cannot be written as a simple linear combination of them.