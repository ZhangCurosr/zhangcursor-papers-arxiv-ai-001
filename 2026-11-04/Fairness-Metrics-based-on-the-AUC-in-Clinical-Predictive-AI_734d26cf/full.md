# Fairness Metrics based on the AUC in Clinical Predictive AI

JOÃO MATOS<sup>✉</sup>, Centre for Statistics in Medicine, University of Oxford, UK

BEN VAN CALSTER, Department of Development and Regeneration, KU Leuven, Belgium

RICHARD D. RILEY, Department of Applied Health Sciences, University of Birmingham, UK and National Institute for Health and Care Research (NIHR) Biomedical Research Centre: Birmingham, University Hospitals Birmingham NHS Foundation Trust and University of Birmingham, UK

PAULA DHIMAN, Centre for Statistics in Medicine, University of Oxford, UK

GARY S. COLLINS, Department of Applied Health Sciences, University of Birmingham, UK and National Institute for Health and Care Research (NIHR) Biomedical Research Centre: Birmingham, University Hospitals Birmingham NHS Foundation Trust and University of Birmingham, UK

Fairness evaluation of clinical prediction models for binary outcomes often involves fairness metrics that compare the area under the receiver operating characteristic curve (AUC, or c-statistic) across subgroups. However, the AUC has been shown to be a non-collapsible metric, meaning that its subgroup specific and overall values are not directly comparable as often implicitly assumed. This raises important questions about how existing AUC-based fairness metrics should be used and interpreted. We examine eight AUC-based fairness metrics proposed in the literature, including: within-group AUC parity; cross-group AUC (xAUC) parity; gap-to-overall AUC; equity-scaled AUC; minimum-based metrics; xAUC<sup>1</sup>; xAUC<sup>0</sup>; and xAUC<sup>1</sup> or xAUC<sup>0</sup> parity. We characterise their estimands, relate them to the decomposition of overall AUC (into within- and cross-group components), and critically appraise them. We then illustrate their behaviour using simulations based on the EuroSCORE II, a clinical prediction model for in-hospital mortality. Based on our formal analysis, simulations, and clinical examples, we recommend reporting an AUC matrix, containing the overall AUC and all within- and cross-group AUC components, alongside population characteristics. When fairness metrics are needed, we recommend reporting the within-group AUC parity, with cross-group AUC parity providing supplementary information but requiring caution due to its potential conflict with calibration. Metrics involving xAUC<sup>1</sup> and xAUC<sup>0</sup> have merit, summarising within- and cross-group AUCs into single values, but their interpretation is non-trivial, and they should be used with caution. Gap-to-overall AUC, equity-scaled AUC, and minimum-based metrics compare quantities that should not be compared and it is unclear how to operationalise them, making them not fit for purpose and generally not recommended.

CCS Concepts: • Computing methodologies → Artificial intelligence; • Applied computing → Health informatics.

Additional Key Words and Phrases: AUC, Clinical Prediction Models, Collapsibility, Fairness Metrics

## ACM Reference Format:

João Matos, Ben Van Calster, Richard D. Riley, Paula Dhiman, and Gary S. Collins. 2026. Fairness Metrics based on the AUC in Clinical Predictive AI. 1, 1 (November 2026), 18 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn

Authors’ Contact Information: João Matos, joao.matos@ndorms.ox.ac.uk, Centre for Statistics in Medicine, University of Oxford, Oxford, UK; Ben Van Calster, Department of Development and Regeneration, KU Leuven, Leuven, Belgium; Richard D. Riley, Department of Applied Health Sciences, University of Birmingham, Birmingham, UK and National Institute for Health and Care Research (NIHR) Biomedical Research Centre: Birmingham, University Hospitals Birmingham NHS Foundation Trust and University of Birmingham, Birmingham, UK; Paula Dhiman, Centre for Statistics in Medicine, University of Oxford, Oxford, UK; Gary S. Collins, Department of Applied Health Sciences, University of Birmingham, Birmingham, UK and National Institute for Health and Care Research (NIHR) Biomedical Research Centre: Birmingham, University Hospitals Birmingham NHS Foundation Trust and University of Birmingham, Birmingham, UK

## 1 Introduction

Clinical prediction models using artificial intelligence (AI) promise to improve patient outcomes and reduce health care costs by informing clinical decision making and risk communication [28]. When developing or evaluating a clinical prediction model, assessing its performance is essential to determine its potential value in clinical practice [6, 18, 29].

Evaluating model performance at the target population level can conceal important diferences in model behaviour across subgroups, raising fairness concerns [24]. It is therefore common to report model performance within relevant subgroups defined by sensitive attributes such as sex or ethnicity. Although variation in model performance across subgroups is expected, understanding the nature and magnitude of diferences is central to fairness evaluation [20].

A fairness metric quantifies the extent to which a model’s output does or does not discriminate, according to a particular notion of fairness, against individuals or subgroups defined by a sensitive attribute [19]. Most fairness metrics are a comparison of model performance across subgroups, for example through diferences, ratios, or other summary functions of subgroup specific performance [10]. Their interpretation therefore depends on the mathematical properties of the underlying performance metric. One mathematical property that is particularly relevant when comparing performance across subgroups is collapsibility [16]. A performance metric � is collapsible if its overall value can be expressed as the weighted average of its subgroup (�) specific values. If a metric is collapsible, the overall value $M _ { \mathrm { o v e r a l l } }$ is directly related to the subgroup specific values $M _ { g } ;$ when a metric is non-collapsible, such relationship breaks down, making subgroup and overall values dificult to interpret jointly and potentially misleading fairness assessments.

One important aspect of prediction model performance for which collapsibility is particularly relevant is discrimina tion, which quantifies the extent to which a model assigns higher estimated probabilities to individuals who experience the event than to those who do not [12, 26, 29]. The area under the receiver operating characteristic curve (AUC, also known as the c-statistic) is the most widely reported measure of discrimination for clinical prediction models for binary outcomes [7, 29]. The AUC has been shown to be non-collapsible [16]. Even though numerous fairness metrics tha are derived directly from subgroup specific and overall AUC values have been proposed, the implications of AUC non-collapsibility for the interpretation of AUC-based fairness metrics have not been systematically examined.

In this paper, we examine AUC-based fairness metrics for clinical prediction models with binary outcomes. We review eight AUC-based fairness metrics, clarify their target estimands, relate them to the decomposition of the AUC into within- and cross-group components, and illustrate their behaviour using simulations based on the European System for Cardiac Operative Risk Evaluation (EuroSCORE) II [23]. Our aim is to clarify when these metrics are informative and how they should be interpreted when evaluating fairness in clinical predictive AI.

## 2 Non-collapsibility of the AUC

The AUC is a non-collapsible metric [16]. The overall AUC cannot, in general, be expressed as a weighted average of subgroup specific AUCs because the AUC considers all event and non-event pairs, including those from diferent subgroups. Consider a population partitioned into � mutually exclusive subgroups. Following the notation of Kallus and Zhou [14], let $\mathrm { A U C } _ { i , j }$ denote the probability that an individual with the event from subgroup � is assigned, by the prediction model, a higher estimated probability than an individual without the event from subgroup �. When $i = j ,$ $\mathrm { A U C } _ { i , i }$ corresponds to the usual within-group AUC, whereas for $i \neq j , \mathrm { A U C } _ { i , j }$ is the cross-group AUC (xAUC). The overall AUC can therefore be decomposed [16] as:

$$
\mathrm { A U C } _ { \mathrm { o v e r a l l } } = \sum _ { i , j = 1 } ^ { K } \frac { \mathcal { P } _ { 1 } ^ { i } } { \mathcal { P } ^ { 1 } } \frac { \mathcal { P } _ { 0 } ^ { j } } { \mathcal { P } 0 } \mathrm { A U C } _ { i , j } ,\tag{1}
$$

where $\boldsymbol { p } _ { 1 } ^ { i }$ and $p _ { 0 } ^ { i }$ denote the numbers (or proportions) of event and non-event observations in subgroup �, respectively, and $\begin{array} { r } { p _ { 1 } = \sum _ { i } p _ { 1 } ^ { i } } \end{array}$ and $\begin{array} { r } { p _ { 0 } = \sum _ { i } p _ { 0 } ^ { i } } \end{array}$ denote the corresponding totals. For two subgroups, � and �, Equation (1) reduces to

$$
\mathrm { A U C } _ { \mathrm { o v e r a l l } } = \frac {  { p _ { 1 } } ^ { A }  { p _ { 0 } } ^ { A } } {  { p _ { 1 } }  { p _ { 0 } } } \mathrm { A U C } _ { A } + \frac {  { p _ { 1 } } ^ { A }  { p _ { 0 } } ^ { B } } {  { p _ { 1 } }  { p _ { 0 } } } \mathrm { x A U C } _ { A , B } + \frac {  { p _ { 1 } } ^ { B }  { p _ { 0 } } ^ { A } } {  { p _ { 1 } }  { p _ { 0 } } } \mathrm { x A U C } _ { B , A } + \frac {  { p _ { 1 } } ^ { B }  { p _ { 0 } } ^ { B } } {  { p _ { 1 } }  { p _ { 0 } } } \mathrm { A U C } _ { B }\tag{2}
$$

Equation (1) shows that the overall AUC is a weighted combination of within- and cross-group AUCs. Consequently, the overall AUC is generally not determined by the within-group AUCs alone, hence why it is non-collapsible [16].

## 3 Review of AUC-based fairness metrics

We now briefly examine eight AUC-based fairness metrics that have been proposed in the literature and summarised by [19], through the lens of the AUC decomposition (presented in Section 2). These metrics are based on diferent components of the AUC and therefore capture diferent aspects of model discrimination. For consistency with the notation introduced above, we write AUC for the overall AUC, $\operatorname { A U C } _ { g }$ for the within-group AUC in subgroup �, and $\mathrm { \bf x A U C } _ { i , j }$ for the cross-group AUC comparing individuals with the event from subgroup � against individuals without the event from subgroup �. We let $\hat { p } _ { y } ^ { g }$ denote the estimated probability assigned by a prediction model for binary outcome $y \in \{ 0 , 1 \}$ for an arbitrary individual in subgroup $g \in \{ A , B \}$

We only consider two subgroups (� and �) for simplicity. However, the metrics presented in this section can be extended to $n > 2$ subgroups by computing pairwise diferences across all $\textstyle { { \binom { n } { 2 } } = { \frac { n ( n - 1 ) } { 2 } } }$ distinct subgroup pairs.

A fairness metric � may be operationalised using a pre-specified threshold � to determine whether a model satisfies the fairness notion at hand. For a metric � where larger values indicate greater fairness, this may be defined as $M \geq \theta .$ Conversely, for a metric where smaller values indicate greater fairness, this may be defined as $M \leq \theta .$ . The choice of � depends on the metric and application. For example, a parity-based metric, where smaller values indicate greater fairness, may use $\theta = 0 . 0 5$ as the maximum acceptable diference.

## 3.1 Within-group AUC parity

Within-group AUC parity calculates the AUC within each subgroup and then estimates their absolute diference [5]:

$$
\mathrm { A U C } _ { \mathrm { p a r i t y } } = \left| \mathrm { A U C } _ { A } - \mathrm { A U C } _ { B } \right| .\tag{3}
$$

A multiplicative variant has also been proposed [2],

$$
\mathsf { A U C } _ { \mathrm { r a t i o } } = \frac { \mathsf { A U C } _ { A } } { \mathsf { A U C } _ { B } } .\tag{4}
$$

Both metrics quantify asymmetries in within-group discrimination only. For the diference-based metric (equation 3), the “optimal value” is 0, whereas for the ratio-based one (equation 4) a value of 1 indicates equal performance.

## 3.2 Cross-group AUC (xAUC) parity

Cross-group AUC (i.e., ranking of individuals with an event from one subgroup versus individuals without an event from the other subgroup) is calculated for each subgroup, the absolute diference is calculated across subgroups [14]:

$$
\begin{array} { r } { \mathrm { \ x A U C } _ { \mathrm { p a r i t y } } = \left| \mathrm { \underline { { x A U C } } } _ { A , B } - \mathrm { \ x A U C } _ { B , A } \right| . } \end{array}\tag{5}
$$

A value of $" 0 \ "$ means that the model is fair with respect to this metric $( \mathrm { i . e . } ,$ , no cross-group discrimination asymmetry).

## 3.3 Gap to overall AUC

It has been proposed to compare within-group AUC with the overall AUC across subgroups [1]. This metric calculates the AUC in the overall sample, in the subgroup, and then takes the absolute diference:

$$
\mathrm { A U C _ { g a p } } = \left| \mathrm { A U C } - \mathrm { A U C } _ { g } \right| .\tag{6}
$$

A value of $^ { \ast } 0 ^ { \ast }$ means that the model is fair with respect to this metric.

## 3.4 Equity-scaled AUC

Luo et al. [15] proposed penalising the overall AUC according to the within-group AUC parity:

$$
\mathrm { \Delta \ A U C _ { E S } = \frac { A U C } { 1 + A U C _ { p a r i t y } } . }\tag{7}
$$

When $\mathrm { A U C _ { E S } } = \mathrm { A U C } \left( { \mathrm { i . e . , A U C _ { p a r i t y } } } = 0 \right)$ , the model is considered fair with respect to this metric.

## 3.5 Minimum-based metrics

Minimum-based metrics can evaluate discrimination according to the worst-performing subgroup [9, 22], for example:

$$
\mathrm { A U C } _ { \operatorname* { m i n } } = \operatorname* { m i n } \left\{ \mathrm { A U C } _ { A } , \mathrm { A U C } _ { B } \right\} .\tag{8}
$$

Analogous definitions may be constructed using other AUC components. A model is considered fair when the minimum value of a metric of interest $( \mathrm { e . g . } , A U C _ { g } )$ is above a pre-specified threshold (e.g., min $\left\{ { \cal A } U C _ { g } \right\} \ge 0 . 8 )$

## 3.6 xAUC<sup>1</sup>

Also referred to as “Pairwise Ranking Fairness” (PRF), xAUC<sup>1</sup> quantifies the probability that an individual with the event from subgroup � is ranked above a randomly selected individual without the event from the population [8, 14]:

$$
\mathrm { x A U C } _ { A } ^ { 1 } = P ( \hat { p } _ { 1 } ^ { A } > \hat { p } _ { 0 } ) ,\tag{9}
$$

$$
\mathrm { x A U C } _ { B } ^ { 1 } = P ( \hat { p } _ { 1 } ^ { B } > \hat { p } _ { 0 } ) .\tag{10}
$$

A higher xAUC<sup>1</sup> indicates that individuals with the event from subgroup � are more likely to be assigned higher estimated probabilities than individuals without the event in the overall population. A value of 0.5 corresponds to chancelevel ranking, while values approaching 1 indicate better discrimination. Fairness is achieved when $\mathrm { x A U C } _ { A } ^ { 1 } = \mathrm { x A U C } _ { B } ^ { 1 } ,$ indicating that individuals with the event are ranked relative to the non-event population equally well across subgroups

## 3.7 xAUC<sup>0</sup>

Also referred to as “Subgroup $\mathrm { A U C ^ { \prime \prime } \ ( s A U C ) , x A U C ^ { 0 } }$ quantifies the probability that a randomly selected individual without the event from subgroup � is ranked below any individual with the event from the population [14, 21]:

$$
\mathrm { x A U C } _ { A } ^ { 0 } = P ( \hat { p } _ { 1 } > \hat { p } _ { 0 } ^ { A } ) ,\tag{11}
$$

$$
\mathrm { x A U C } _ { B } ^ { 0 } = P ( \hat { p } _ { 1 } > \hat { p } _ { 0 } ^ { B } ) .\tag{12}
$$

Similarly to $\mathrm { \bf x A U C ^ { 1 } }$ (above), fairness is achieved when $\mathrm { x A U C } _ { A } ^ { 0 } = \mathrm { x A U C } _ { B } ^ { 0 } ,$ , indicating that individuals without the event are ranked relative to the event population equally well across subgroups.

## 3.8 $\mathbf { x A U C } ^ { 1 }$ and $\mathbf { x A U C } ^ { 0 }$ parity

These metrics estimate absolute diferences in xAUC<sup>1</sup> or $\mathbf { \nabla _ { X A U C } } ^ { 0 }$ , respectively, across subgroups [8, 21]:

$$
\mathrm { \ x A U C _ { p a r i t y } ^ { 1 } } = \left| \mathrm { x A U C _ { \it A } ^ { 1 } } - \mathrm { x A U C _ { \it B } ^ { 1 } } \right| ,\tag{13}
$$

$$
\mathrm { x A U C _ { p a r i t y } ^ { 0 } } = \left| \mathrm { x A U C _ { \it A } ^ { 0 } } - \mathrm { x A U C _ { \it B } ^ { 0 } } \right|\tag{14}
$$

Fairness is achieved when $\mathrm { \Delta x A U C _ { p a r i t y } ^ { 1 } = 0 ~ o r ~ x A U C _ { p a r i t y } ^ { 0 } = 0 . }$

## 4 Relationships between AUC-based fairness metrics

Several of the metrics presented above are either related to each other or to the overall AUC. For example, $\mathrm { \tt x A U C ^ { 1 } }$ and $\mathrm { \mathbf { x } A U C } ^ { 0 }$ can be expressed in terms of within-group and cross-group discrimination. For two subgroups � and �,

$$
\mathrm { x A U C } _ { A } ^ { 1 } = \frac { p _ { 0 } ^ { A } } { p _ { 0 } } \mathrm { A U C } _ { A } + \frac { p _ { 0 } ^ { B } } { p _ { 0 } } \mathrm { x A U C } _ { A , B } ,\tag{15}
$$

$$
\mathrm { x A U C } _ { B } ^ { 1 } = \frac { p _ { 0 } ^ { A } } { p _ { 0 } } \mathrm { x A U C } _ { B , A } + \frac { p _ { 0 } ^ { B } } { p _ { 0 } } \mathrm { A U C } _ { B } ,\tag{16}
$$

$$
\mathrm { x A U C } _ { A } ^ { 0 } = \frac { p _ { 1 } ^ { A } } { p _ { 1 } } \mathrm { A U C } _ { A } + \frac { p _ { 1 } ^ { B } } { p _ { 1 } } \mathrm { x A U C } _ { B , A } ,\tag{17}
$$

$$
\mathrm { x A U C } _ { B } ^ { 0 } = \frac { p _ { 1 } ^ { A } } { p _ { 1 } } \mathrm { x A U C } _ { A , B } + \frac { p _ { 1 } ^ { B } } { p _ { 1 } } \mathrm { A U C } _ { B } .\tag{18}
$$

These four quantities can in turn form a linear decomposition of the overall AUC,

$$
\mathrm { A U C } = \frac { p _ { 1 } ^ { A } } { p _ { 1 } } \mathrm { x A U C } _ { A } ^ { 1 } + \frac { { p _ { 1 } ^ { B } } } { p _ { 1 } } \mathrm { x A U C } _ { B } ^ { 1 } = \frac { p _ { 0 } ^ { A } } { p _ { 0 } } \mathrm { x A U C } _ { A } ^ { 0 } + \frac { p _ { 0 } ^ { B } } { p _ { 0 } } \mathrm { x A U C } _ { B } ^ { 0 } ,\tag{19}
$$

providing an alternative representation of the overall model discrimination. Full proofs can be found in the Appendix. Unlike the commonly reported within-group $\mathrm { A U C , x A U C ^ { 1 } }$ and $\mathrm { \tt x A U C } ^ { 0 }$ incorporate both within- and cross-group discrimination. An alternative way to interpret these decompositions is through a part-whole perspective. $\mathrm { \ x A U C } _ { A } ^ { 1 }$ and $\mathrm { x A U C } _ { B } ^ { 1 }$ (and analogously $\mathrm { \mathbf { x } } \mathrm { A U C } _ { A } ^ { 0 }$ and $\mathrm { \ x A U C } _ { B } ^ { 0 } )$ form mutually exclusive and collectively exhaustive contributions to the overall AUC, a property that conventional within-group AUCs lack since they omit the cross-group components.

Figure 1 summarises AUC components and AUC-based fairness metrics using AUC matrices. For a binary sensitive attribute, an AUC Matrix partitions all pairwise comparisons according to the subgroup membership of the individual with the event (rows) and the individual without the event (columns) [4]. Each cell represents the probability that a randomly selected individual with an event from one subgroup has a higher estimated probability than a randomly selected individual without an event from the same or the other subgroup. The diagonal cells correspond to the within-group AUCs $( { \mathrm { A U C } } _ { A }$ and $\operatorname { A U C } _ { B } )$ and the of-diagonal cells to the cross-group $\mathsf { A U C s } \left( \mathrm { x A U C } _ { A , B } \right.$ and $\mathrm { x A U C } _ { B , A } )$

## 5 Critical appraisal of AUC-based fairness metrics

In this section, we critically appraise each of the metrics in Section 3, in light of their formalisation and relationships with one another. Table 1 summarises the estimand targeted by each metric, our critical appraisal, and the corresponding

Individuals without an event from subgroup  
![](images/3952ae66a59d8aab582f3c2d1961fb08dabe8b0293734de8a0fe2f0bf0ae61b5.jpg)

![](images/99166d0e024709f0d79f6bf677843c5cf97bfb790e5f44c04d55148ab95245d0.jpg)

$$
\bigsqcup \mathrm { \ A U C _ { p a r i t y } } = \big | \mathrm { A U C } _ { A } - \mathrm { A U C } _ { B } \big |
$$

$$
\square \mathrm { \ x A U C _ { p a r i t y } } = \left| \mathrm { x A U C _ { A , B } } - \mathrm { x A U C _ { B , A } } \right|
$$

![](images/06562123409b6131a01908b8fc7109bb919edc3536dfb15f5274da5af31ef854.jpg)

$$
\begin{array} { r } { \bigsqcup _ { \mathrm { \scriptsize ~ X A U C } _ { A } ^ { 1 } } = \frac { p _ { 0 } ^ { A } } { p _ { 0 } } \operatorname { A U C } _ { A } + \frac { p _ { 0 } ^ { B } } { p _ { 0 } } \mathrm { \ x A U C _ { A , B } } } \end{array}
$$

$$
\begin{array} { r } { \bigsqcup _ { \mathrm { \scriptsize ~ x A U C } _ { B } ^ { 1 } } = \frac { p _ { 0 } ^ { A } } { p _ { 0 } } \mathrm { x A U C _ { B , A } } + \frac { p _ { 0 } ^ { B } } { p _ { 0 } } \mathrm { A U C } _ { B } } \end{array}
$$

![](images/f0b48c0ce37dcbdec3a93c04d35462e3f2445b5f55cfbaf4f4ac9f82fc597e4c.jpg)  
Fig. 1. Relationship between diferent AUC metrics (for binary outcome $y \in \{ 0 , 1 \}$ and assuming two subgroups � and �).

$$
\begin{array} { r } { \bigsqcup _ { \mathrm { \bf ~ x A U C } _ { A } ^ { 0 } } = \frac { p _ { 1 } ^ { A } } { p _ { 1 } } \mathrm { A U C } _ { A } + \frac { p _ { 1 } ^ { B } } { p _ { 1 } } \mathrm { x A U C } _ { \mathrm { B , A } } } \end{array}
$$

$$
\begin{array} { r } { \bigsqcup _ { \mathrm { \scriptsize ~ x A U C } _ { B } ^ { 0 } } = \frac { p _ { 1 } ^ { A } } { p _ { 1 } } \mathrm { x A U C _ { A , B } } + \frac { p _ { 1 } ^ { B } } { p _ { 1 } } \mathrm { A U C } _ { B } } \end{array}
$$

recommendation. Recommendations are colour coded as recommended, use with caution, and generally not recommended. The table is intended as a practical guide for selecting AUC-based fairness metrics that align with an intended fairness objective whilst avoiding misleading interpretations. In general, we recommend reporting the overall AUC and the complete AUC matrix in Figure 1, which makes all within-group and cross-group AUC components explicit.

## 5.1 Within-group AUC parity

Within-group AUC parity [5] compares the AUC calculated within each subgroup. It is appropriate when the fairness objective is to assess whether the model discriminates equally well within each subgroup. As demonstrated by the decomposition of the AUC, within-group AUCs ignore cross-group components, which may difer substantially even when subgroup specific AUCs are identical. Consequently, similar values of $\mathrm { { A U C } } _ { A }$ and $\mathrm { A U C } _ { B }$ may coexist with pronounced asymmetries in $\mathrm { _ { x A U C } } _ { A , B }$ and $\mathrm { _ { x A U C } } _ { B , A }$ , as well as with apparently paradoxical behaviour of the overall AUC due to non-collapsibility [16]. Conversely, substantial diferences in within-group AUCs do not necessarily imply disparities in cross-group AUCs. This metric is recommended as the primary fairness metric for discrimination performance and should be reported alongside overall AUC. As the within-group AUC parity does not capture crossgroup ranking disparities, the xAUC parity can provide important additional information, with cautious interpretation.

## 5.2 Cross-group AUC (xAUC) parity

Cross-group AUC (xAUC) parity is relevant when fairness concerns how individuals from diferent groups are ranked relative to one another (e.g., patients are ranked based on estimated probabilities or a binary decision is made after applying a single threshold). This metric provides information beyond the within-group AUC parity, which is important to comprehensively evaluate fairness in discrimination performance. However, as discussed by [16], xAUC parity may be influenced by diferences in outcome prevalence or calibration across subgroups. Consequently, improvements in xAUC parity may come at the expense of calibration disparity. Calibration has been recommended as a key domain of model performance [29] and fairness evaluation [17, 19], unless only patient ranking matters [14]. The xAUC parity should be used with caution and reported alongside within-group AUC parity and overall AUC. Interpreting this metric in isolation may be misleading because of its interplay with calibration and case-mix diferences across subgroups.

## 5.3 Gap to overall AUC

The diference between the overall AUC and a subgroup specific AUC compares quantities that estimate fundamentally diferent aspects of discrimination. Subgroup specific AUC values quantify only within-group discrimination, whereas the overall AUC combines within-group and cross-group components. As these quantities are generally not directly comparable, this type of metric is generally not recommended for fairness checks.

## 5.4 Equity-Scaled AUC

Equity-Scaled AUC divides overall AUC by within-group AUC parity. The numerator (overall AUC) depends on both within- and cross-group AUCs, whereas the denominator depends only on within-group AUC parity. The metric implicitly assumes that fairness (“equity”) is captured by within-group parity alone, ignoring xAUCs. If used in isolation, this metric is dificult to interpret and susceptible to manipulation. This metric is generally not recommended.

## 5.5 Minimum-based fairness metrics

Minimum-based metrics focus exclusively on the worst-performing subgroup and aim to align with fairness definitions based on maximising the minimum subgroup performance [22]. However, it is unclear how to interpret when reported in isolation, or how to act upon disparities flagged by this metric alone. This type of metric is generally not recommended.

## 5.6 xAUC<sup>1</sup>

$\mathrm { \ x A U C } _ { g } ^ { 1 }$ measures discrimination for individuals with the event $( Y = 1 )$ in subgroup �, by comparing their estimated probabilities against individuals without the event in the overall population. $\mathrm { \ x A U C } _ { g } ^ { 1 }$ incorporates both within- and cross-group discrimination through a linear decomposition (Section 4). Consequently, it is less susceptible to Simpsontype paradoxes arising from ignoring cross-group AUCs. Whether this metric qualifies as a fairness metric on its own, or as an ancillary metric for fairness evaluation, is unclear. For example, for two subgroups, a comparison between $\mathrm { \mathbf { x } } \mathrm { A U C } _ { A } ^ { 1 }$ and $\mathrm { \bf x A U C } _ { B } ^ { 1 }$ would be necessary to draw any fairness conclusion, and $\mathrm { \ x A U C } _ { A } ^ { 1 }$ or $\mathrm { \underline { { x A U C } } } _ { B } ^ { 1 }$ do not say much about fairness on their own. This metric should be used with caution for each subgroup, when fairness is viewed from the perspective of individuals who do experience the event, and both within- and cross-group discrimination are relevant.

## 5.7 xAUC<sup>0</sup>

$\mathrm { \bf x A U C } ^ { 0 }$ measures discrimination for individuals without the event $( Y = 0 )$ in subgroup �. Like $\mathrm { \bf x A U C ^ { 1 } }$ , it combines within- and cross-group discrimination into a linear decomposition of the overall AUC. This metric should be used with caution when fairness is viewed from the perspective of individuals who do not experience the event, and both within- and cross-group discrimination are relevant.

## 5.8 $\mathbf { x A U C ^ { 1 } }$ and $\mathbf { x A U C } ^ { 0 }$ Parity

These metrics should be used with caution whenever the use o $\mathbf { \dot { x } } \mathbf { A } \mathbf { U } \mathbf { C } ^ { 1 }$ or $\mathrm { \bf x A U C } ^ { 0 }$ is deemed justifiable and relevant. Fairness conclusions based on the $\mathrm { \Delta x A U C _ { p a r i t y } ^ { 1 } }$ may be particularly relevant in positively polar decisions (i.e., patients benefit from selection, e.g., being selected for an organ transplant). An eligible patient from subgroup � should always be ranked (by the model) above an ineligible patient from any subgroup [25], regardless of subgroup membership. Contrary to $\mathrm { \Delta x A U C _ { p a r i t y } ^ { 1 } , x A U C _ { p a r i t y } ^ { 0 } }$ is particularly relevant in negatively polar decisions (i.e, patients benefit from exclusion, e.g., quarantining patients at high infectious risk). An ineligible patient from subgroup � should always be ranked (by the model) below an eligible patient from any subgroup [25], regardless of subgroup membership.

## 6 Clinical example

The European System for Cardiac Operative Risk Evaluation (EuroSCORE) II [23] is a clinical prediction model developed using data from 22,381 patients across 154 hospitals in 43 countries. It uses demographic and clinical predictors to estimate the probability of in-hospital mortality following cardiac surgery and achieved an AUC of 0.81. Although the model has been widely validated [3, 11, 13], its fairness has been rarely assessed [27]. As a publicly-available model, EuroSCORE II may be used for, among others, three potential use cases, which we summarise below:

(1) Risk estimation (continuous scale): The model output can be used to communicate risk to patients and caregivers, benchmark institutional outcomes, or stratify populations’ risk.

(2) Binary decision: A threshold (set by clinicians, hospitals, or health authorities) is applied, to facilitate surgical decision-making or serve as a quality control benchmark to stratify outcomes between low- and high-risk interventions. Decision thresholds are often arbitrarily set by optimising a statistical criterion.

(3) Patient ranking: In certain contexts, only the ordering of patients matters and the absolute magnitude of estimated risk is irrelevant. For example, a prioritisation system for surgical waiting lists might rank patients according to their estimated risk and relative ordering.

Table 1. Summary of AUC-based fairness metrics, their meanings, critical appraisal, and recommendations.
<table><tr><td>Metric</td><td>Formula</td><td>Meaning</td><td>Critical appraisal and recommendation</td></tr><tr><td>Within-group AUC parity [2, 5]</td><td> $| \mathrm { A U C } _ { A } - \mathrm { A U C } _ { B } |$ </td><td>Absolute difference in within-group discrimination</td><td>Recommended. Appropriate as the primary fairness metric for discrimination performance. Should be reported alongside overall AUC. Can be used on its own, with cross-group AUC parity providing important additional information for a more comprehensive evaluation of fairness in model discrimination.</td></tr><tr><td>Cross-group AUC parity [14]</td><td> $\left| \mathbf { x A U C } _ { A , B } - \mathbf { x A U C } _ { B , A } \right|$ </td><td>Absolute difference in cross-group discrimination</td><td>Use with caution. Provides important additional information on xAUC disparities, but it may be influenced by differences in outcome prevalence and calibration across subgroups. Should be reported alongside within-group  $\mathrm { A U C }$  parity and overall AUC. Interpretation in isolation may be misleading.</td></tr><tr><td>Gap to overall AUC [1]</td><td> $\left| { \mathrm { A U C } } - { \mathrm { A U C } } _ { g } \right|$ </td><td>Absolute difference between overall and within-group discrimination</td><td>Not recommended. Compares quantities estimating fundamentally different aspects of discrimination. Interpretation is generally convoluted.</td></tr><tr><td>“Equity-Scaled” AUC [15]</td><td> $\frac { \mathrm { A U C } } { 1 + \mathrm { A U C } _ { \mathrm { p a r i t y } } }$ </td><td>Overall discrimination penalised by within-group discrimination parity</td><td>Not recommended. Combines quantities targeting different aspects of discrimination. It is difficult to interpret and offers limited added value when other metrics are already reported.</td></tr><tr><td>Minimum-based metrics [9, 22]</td><td> $\operatorname* { m i n } \{ { \mathrm { A U C } } _ { A } , { \mathrm { A U C } } _ { B } \}$ </td><td>Worst-performing subgroup discrimination</td><td>Not recommended. Aligns with fairness objectives based on maximising worst-case subgroup performance but it is unclear how to interpret on its own or act upon potential disparities.</td></tr><tr><td> $\mathrm { \ x A U C ^ { 1 } \ [ 8 , 1 4 ] }$ </td><td> $P ( \hat { p } _ { 1 } ^ { g } > \hat { p } _ { 0 } )$ </td><td>Probability that an individual in subgroup q with the event is ranked above any individual without the event</td><td>Use with caution. Interpretation is non-trivial. Combines within-group and cross-group discrimination for a subgroup g in a single metric, focusing on individuals with the event. Contains more information than the within-group AUC alone but it is still a summary rather than the full AUC matrix.</td></tr><tr><td> $\mathrm { x A U C ^ { 0 } \ [ 1 4 , 2 1 ] }$ </td><td> $P ( \hat { p } _ { 1 } > \hat { p } _ { 0 } ^ { g } )$ </td><td>Probability that an individual in subgroup g without the event is ranked below any individual with the event</td><td>Use with caution. Interpretation is non-trivial. Combines within-group and cross-group discrimination for a subgroup g in a single metric, focusing on individuals without the event. Contains more information than the within-group AUC alone but it does not replace the full AUC matrix.</td></tr><tr><td> $\mathrm { \ x A U C ^ { 1 } \ o r }$  xAUC0 Parity  $[ 8 , 1 4 , 2 1 ]$ </td><td> $\left| \mathbf { x A U C } _ { A } ^ { 1 } - \mathbf { x A U C } _ { B } ^ { 1 } \right|$   $\left| \mathbf { x A U C } _ { A } ^ { 0 } - \mathbf { x A U C } _ { B } ^ { 0 } \right|$ </td><td>Absolute difference ir  $\mathbf { \Omega } _ { 1 } \mathbf { x } \mathbf { A } \mathbf { U } \mathbf { C } ^ { 1 }$  or  $\mathrm { \tt x A U C } ^ { 0 }$  across subgroups</td><td>Use with caution. Interpretation is difficult. Could be helpful when a single fairness metric is needed to summarise the differences observed in the AUC matrix.</td></tr></table>

In this section, we consider these three use cases as context for the interpretation of AUC-based fairness metrics, in three simulated performance scenarios. The Python code used to generate the simulated data, AUC-based fairness metrics, and figures presented in this section is publicly available as an executable Jupyter notebook on Google Colab.

## 6.1 Simulations and discrimination performance scenarios

We simulate three scenarios with varying discrimination performance across subgroups, defined by patient sex at birth (binary sensitive attribute). Simulations were conducted by generating estimated probabilities distributions separately for patients with and without events within each sex subgroup. For each scenario, we fixed the proportion of male and female patients $( p ^ { M } = 0 . 8 , p ^ { F } = 0 . 2 )$ and the outcome prevalences $( \pi ^ { M } = 0 . 0 2 , \pi ^ { F } = 0 . 0 3 )$ , following the patient characteristics reported in [27]. Estimated probability distributions were simulated using truncated normal distributions with support on the interval [0, 1]. Discrimination was controlled by specifying the means of the distributions patients not experiencing an event $( \mu _ { 0 } ^ { F }$ and $\mu _ { 0 } ^ { M } )$ and experiencing an event $( \mu _ { 1 } ^ { F }$ and $\mu _ { 1 } ^ { M } )$ . Simulation parameters are summarised in Table 2. The resulting AUC metrics are reported in Table 3. Each scenario is examined below.

Scenario (i): Substantial within-group AUC disparity. Suppose the model’s overall AUC is 0.85, with within-group AUCs of 0.88 for male patients and 0.73 for female patients. However, there is a diference in cross-group AUC in the opposite direction, with $\mathrm { x A U C } _ { M , F } = 0 . 6 7$ and $\mathrm { x A U C } _ { F , M } = 0 . 9 2$ . Metrics are summarised in Figure 2a.

Scenario (ii): Improving worst-performing subgroup performance. The model is updated to improve discrimination in female patients, resulting in $\mathrm { A U C } _ { \mathrm { f e m a l e } }$ increasing from 0.73 to 0.80, whilst performance in male patients remains unchanged $( \mathrm { A U C } _ { \mathrm { m a l e } } = 0 . 8 8 )$ . Despite this improvement, the overall AUC decreases from 0.85 to 0.84, which is explained by a more pronounced disparity in cross-group AUC components: $\mathrm { x A U C } _ { M , F } = 0 . 4 4$ and $\mathrm { x A U C } _ { F , M } = 0 . 9 9$ (see Figure 2b).

Scenario (iii): Degrading best-performing subgroup performance. Suppose AUC parity is now achieved, such that the model appears fair from the standpoint of within-group AUC parity $( \mathrm { A U C _ { M } } = \mathrm { A U C _ { F } } = 0 . 8 6 )$ . Improving $\mathrm { A U C _ { F } }$ came at the cost of degrading $\mathrm { { A U C } _ { M } }$ . Despite within-group AUC parity, there still is a disparity in cross-group AUC: $\mathrm { x A U C } _ { M , F } = 0 . 5 3$ and $\mathrm { x A U C } _ { F , M } = 0 . 9 8$ , which results in an overall AUC of 0.84 (see Figure 2c).

Table 2. Simulation parameters.
<table><tr><td>Scenario</td><td> $\boldsymbol { p } ^ { M }$ </td><td> $\pi ^ { M }$ </td><td> $\pi ^ { F }$ </td><td> $\mu _ { 0 } ^ { M }$ </td><td> $\mu _ { 1 } ^ { M }$ </td><td> $\mu _ { 0 } ^ { F }$ </td><td> $\mu _ { 1 } ^ { F }$ </td><td>σ</td></tr><tr><td>(i)</td><td>0.8</td><td>0.02</td><td>0.03</td><td>0.015</td><td>0.031</td><td>0.025</td><td>0.034</td><td>0.01</td></tr><tr><td>(ii)</td><td>0.8</td><td>0.02</td><td>0.03</td><td>0.015</td><td>0.031</td><td>0.033</td><td>0.045</td><td>0.01</td></tr><tr><td>(iii)</td><td>0.8</td><td>0.02</td><td>0.03</td><td>0.016</td><td>0.031</td><td>0.030</td><td>0.045</td><td>0.01</td></tr></table>

Legend: $\boldsymbol { p } ^ { M }$ denotes the proportion of male patients in the population. $\pi ^ { M }$ and $\pi ^ { F }$ are the outcome prevalence rates for male and female patients, respectively. $\mu _ { 0 } ^ { M }$ and $\mu _ { 1 } ^ { M }$ denotes mean of the (truncated normal) distribution of estimated probabilities for male patients without and with an event, respectively; �<sup>�</sup> an $\lvert \mu _ { 1 } ^ { F }$ the mean of the distribution of estimated probabilities for female patients without and with an event. � denotes the standard deviation of each simulated normal distribution.

Table 3. AUC metrics and AUC-based fairness metrics, across subgroups.
<table><tr><td>Scenario</td><td> $\mathrm { A U C _ { g l o b a l } }$ </td><td> $\mathrm { A U C } _ { M }$ </td><td> $\mathrm { A U C } _ { F }$ </td><td> $\mathrm { x A U C } _ { M , F }$ </td><td> $\mathrm { x A U C } _ { F , M }$ </td><td> $\mathrm { x A U C } _ { M } ^ { 1 }$ </td><td> $\mathrm { \bf x A U C } _ { F } ^ { 1 }$ </td><td> $\mathrm { { A U C } _ { \mathrm { { p a r i t y } } } }$ </td><td> $\mathrm { \Delta x A U C _ { p a r i t y } }$ </td><td> $\underline { { \mathrm { x A U C } _ { \mathrm { p a r i t y } } ^ { 1 } } }$ </td></tr><tr><td>(i)</td><td>0.85</td><td>0.88</td><td>0.73</td><td>0.67</td><td>0.92</td><td>0.84</td><td>0.88</td><td>0.15</td><td>0.25</td><td>0.04</td></tr><tr><td>(ii)</td><td>0.84</td><td>0.88</td><td>0.80</td><td>0.44</td><td>0.99</td><td>0.79</td><td>0.95</td><td>0.08</td><td>0.55</td><td>0.16</td></tr><tr><td>(iii)</td><td>0.84</td><td>0.86</td><td>0.86</td><td>0.53</td><td>0.98</td><td>0.80</td><td>0.96</td><td>0.00</td><td>0.45</td><td>0.16</td></tr></table>

## 6.2 Fairness interpretation across use cases and scenarios

6.2.1 Model use case (1): Risk estimation. Let us assume that the developed model is used to communicate risk. The model estimated probabilities are directly used, and no decision thresholds are applied. We consider three scenarios.

Scenario (i): Substantial AUC disparity. A diference in within-group AUCs exists $( \mathrm { A U C _ { p a r i t y } } = | 0 . 8 8 - 0 . 7 3 | = 0 . 1 5 ) .$ However, when examining xAUC components, the picture difers: $\mathrm { x A U C } _ { M , F } = 0 . 6 7$ and $\mathrm { x A U C } _ { F , M } = 0 . 9 2$ , which means that female patients with an event would be better separated from male patients without an event $( \mathrm { x A U C } _ { F , M } )$ , than the other other way around $( \mathrm { x A U C } _ { M , F } )$ , resulting in a cross-group disparity of 0.25 $( \mathrm { x A U C _ { p a r i t y } } = | 0 . 6 7 - 0 . 9 2 | = 0 . 2 5 )$ Here, $\mathrm { { A U C } _ { \mathrm { { p a r i t y } } } }$ does not provide a complete picture of unfairness, and ${ \mathrm { x A U C } } _ { \mathrm { p a r i t y } }$ is informative.

![](images/c5f03e5cac0a46275392d656ba325a84c7182c320616ef2a0fbcc46d66d627f7.jpg)

![](images/9fc70b2d970aca02d578589c319e43edeb0f312902c54a87f9c581beb0bf509b.jpg)  
(a) Simulated scenario illustrating substantial subgroup AUC disparity, reflecting an initial setup for the EuroSCORE case study. Estimated probability distributions follow $\mu _ { 0 } ^ { M } = 0 . 0 \dot { 1 } 5 , \mu _ { 1 } ^ { M } = 0 . 0 3 1 , \mu _ { 0 } ^ { F } = 0 . 0 2 5$ , and $\mu _ { 1 } ^ { F } = 0 . 0 3 4$ . Labels of 0 denote patients without an event, labels of 1 denote patients with an event.

![](images/5324fb067e924c69936f747a5ca8a427a375aea720f77dbb5c32367e424989b7.jpg)

![](images/281bc89ccb73a81322a552253c5c2d614280d8385f43de3389fb58479921cae3.jpg)  
(b) Simulated scenario illustrating a counter-intuitive “level-up” efect, in which improved subgroup performance does not change the overall $\mathsf { A U C }$ . Estimated probability distributions follow $\mu _ { 0 } ^ { M } = 0 . 0 1 5 , \mu _ { 1 } ^ { M } = 0 . 0 3 1 , \mu _ { 0 } ^ { F } = 0 . 0 3 3 , \mathrm { a n d } \mu _ { 1 } ^ { F } = 0 . 0 4 5 .$

![](images/fa5f4e502bb0eb42ced17fe4cb37a5f6afaa6c71d772eb93810bfe7744c6041d.jpg)

![](images/44b8b9d13226f8f8d4f2226bdd184d6db6e65fa683baa2de351829a326adcdfc.jpg)  
(c) Simulated scenario in which subgroup parity is achieved $( \mathrm { A U C _ { A } } = \mathrm { A U C _ { B } } )$ while the overall AUC falls outside the subgroup AUC range. Estimated probability distributions follow $\mu _ { 0 } ^ { M } = 0 . 0 1 6 , \mu _ { 1 } ^ { M } = 0 . 0 3 1 , \ : \bar { \mu } _ { 0 } ^ { F } = 0 . 0 3 0 ,$ and $\mu _ { 1 } ^ { F } = 0 . 0 4 5 .$

Fig. 2. Illustrative cases comparing subgroup and overall AUC relationships. Within each subgroup � $\in M , F$ (male, female, respectively) the estimated probabilities for non-events and events are drawn from truncated normal distributions on [0, 1], using 100,000 samples in total. For all three scenarios, ${ p ^ { M } } = 0 . 8 , \pi ^ { M } = 0 . 0 2 , \pi ^ { F } = 0 . 0 3$ , and the estimated probabilities follow $\sigma = 0 . 0 1$ for all strata. $\mu _ { 0 } ^ { M }$ $\mu _ { 1 } ^ { M } , \mu _ { 0 } ^ { F }$ , and $\mu _ { 1 } ^ { F }$ are specific to each scenario. Labels of 0 denote patients without an event, labels of 1 denote patients with an event.

When examining $\mathrm { \bf x A U C ^ { 1 } }$ metrics, the interplay between within- and cross-group discrimination is summarised: $\mathrm { x A U C } _ { M } ^ { 1 } = 0 . 8 4$ and $\mathrm { x A U C } _ { F } ^ { 1 } = 0 . 8 8$ , resulting in a less pronounced disparity according to these metrics: $\mathrm { \ x A U C _ { p a r i t y } ^ { 1 } = }$ $\lvert 0 . 8 4 - 0 . 8 8 \rvert = 0 . 0 4$ . This results from the fact that male patients are better of according to within-group AUCs, whereas female patients are better of in terms of cross-group AUC. Here, $\mathrm { \Delta x A U C _ { p a r i t y } ^ { 1 } }$ is not exactly zero, meaning that the model is not considered “perfectly fair” according to this metric, which happens because the “advantage” that female patients have in cross-group discrimination outweighs the “advantage” that male patients have in within-group discrimination.

Key takeaways: Within-group AUC disparity can co-exist with cross-group AUC disparity, underscoring the importance of reporting both within-group AUC parity and cross-group AUC parity metrics

Scenario (ii): Improving worst-performing subgroup performance. In this new scenario, the diference in within-group AUC has decreased, thanks to increase in $\mathrm { A U C } _ { F }$ from 0.73 to 0.80, while $\mathrm { A U C } _ { M }$ was kept at 0.88. As depicted on the right-hand side of Figure 2b, this is explained by the fact that the distribution of model estimated probabilities for male patients did not change, only female patients’ distribution did. Nonetheless, this resulted in changes in cross-group AUC values that afected both subgroups’ performance. $\mathrm { x A U C } _ { M , F }$ further deteriorated, decreasing from $0 . 6 7 ,$ , in Scenario (i), to 0.44 here, whereas $\mathrm { x A U C } _ { F , M }$ improved, going from 0.92, in Scenario (i), to 0.99 here. As a result, overall AUC degraded slightly, going from 0.85 in Scenario (i), to 0.84, meaning that the model discrimination improvement in terms of AUC� did not compensate for losses in cross-group discrimination (in which both $\mathrm { x A U C } _ { M , F }$ and $\mathbf { x A U C } _ { F , M }$ changed).

As a result ofthese model changes, $\mathrm { \ x A U C } _ { M } ^ { 1 }$ declines from 0.84 in Scenario (i) to 0.79 here, highlighting the contribution of both $\mathsf { A U C } _ { M }$ and $\mathrm { x A U C } _ { M , F }$ to $\mathrm { \ x A U C } _ { M } ^ { 1 }$ . The verified decline in $\mathrm { \ x A U C } _ { M } ^ { 1 }$ also shows that, even though the model estimated probabilities for male patients did not change, discrimination declined for these patients, overall, due to the contribution of cross-group ranking. On the other hand, $\mathrm { \bf x A U C } _ { F } ^ { 1 }$ improved, going from 0.88, in Scenario (i), to 0.95 in this Scenario (ii), which stems from the improved $\mathrm { x A U C } _ { F , M }$ and $\mathrm { A U C } _ { F }$

In terms of fairness metrics, the improved $\mathrm { A U C } _ { F }$ results in a lower $\mathrm { A U C _ { p a r i t y } } = \left| 0 . 8 8 - 0 . 8 0 \right| = 0 . 0 8 ,$ , meaning that the model in Scenario (ii) is fairer than the model in Scenario (i) in terms of within-group discrimination. However, $\mathrm { x A U C } _ { p a r i t y } = | 0 . 4 4 - 0 . 9 9 | = 0 . 5 5$ , which is substantially higher than the 0.25 disparity in cross-group discrimination from Scenario (i), meaning that this new model is more unequal in terms of cross-group discrimination, to a greater extent than the fairness improvement in within-group discrimination. The $\mathrm { \bf x A U C ^ { 1 } }$ metrics show this: $\mathrm { \Delta x A U C _ { p a r i t y } ^ { 1 } = }$ $\left| 0 . 7 9 - 0 . 9 5 \right| = 0 . 1 6$ , an increase in disparity when compared to Scenario (i) (0.04). This parity metric indicates greater inequality across male and female patients, when taking into consideration both within- and cross-group discrimination.

An important artefact in this Scenario (ii) is the apparent improvement in “subgroup AUCs” (i.e., improvement of female patients’ within-group AUC), but a decline in overall AUC. This almost paradoxical finding is a consequence of the AUC non-collapsibility [16], and highlights the importance of considering and reporting cross-group AUCs. The phenomena we describe here complicate interpretation and communication with stakeholders. What was intended to be a “level-up” strategy, improving performance (measured in terms of within-group AUC) of the worse-of subgroup (female patients), without compromising the better-of subgroup (male patients), inadvertently produced a “level-down” efect for male patients through changes in cross-group discrimination.

Key takeaways: Improving the performance of a worse-performing subgroup in terms of within-group AUC, without any model change to the model estimated probabilities of the better-performing subgroup, may nonetheless result in a deterioration of the unchanged subgroup’s cross-group AUC. If such deterioration is in greater magnitude than the within-group improvement, a (seemingly paradoxical) degradation of overall AUC can occur.

Scenario (iii): Degrading best-performing subgroup performance. In this final scenario, within-group AUC parity is achieved $( \mathrm { A U C _ { p a r i t y } } ~ = ~ \vert 0 . 8 6 - 0 . 8 6 \vert ~ = ~ 0 )$ thanks to a model modification that shifts the estimated probabilities distributions of both male and female patients (in Scenario (ii) only female patients’ distribution had been afected). As a result, $\mathrm { A U C } _ { M }$ declines to 0.86, from 0.88 in Scenario (i) and (ii), equalising $\mathrm { A U C } _ { F }$ at 0.86, which raises from 0.80 in Scenario (ii). This model modification also results in an increase of $\mathbf { x A U C } _ { M , F }$ from 0.44 in Scenario (ii) to 0.53 here, and a slight decrease in $\mathrm { x A U C } _ { F , M }$ from 0.99 in Scenario (ii) to 0.98 here, making $\mathrm { x A U C _ { p a r i t y } } = | 0 . 5 3 - 0 . 9 8 | = 0 . 4 5$ . Overall AUC is kept at 0.84, which is below both $\mathrm { A U C } _ { M } = \mathrm { A U C } _ { F } = 0 . 8 6$ , an “artefact” of AUC’s non-collapsibility [16] that is possible here due to cross-group AUC values $( \mathrm { x A U C } _ { M , F } = 0 . 5 3$ and $\mathrm { x A U C } _ { F , M } = 0 . 9 8 )$ , which together outweigh the within-group values $( \mathrm { A U C } _ { M } = \mathrm { A U C } _ { F } = 0 . 8 6 )$

Additionally, $\mathrm { x A U C } _ { M } ^ { 1 }$ increased from 0.79 in Scenario (ii) to 0.80 here, and $\mathrm { \bf x A U C } _ { F } ^ { 1 }$ from 0.95 in Scenario (ii) to 0.96 here, resulting in $\mathrm { x A U C _ { p a r i t y } ^ { 1 } = 0 . 1 6 . }$ , the same exact value achieved in Scenario (ii). The fact that $\mathrm { \Delta x A U C _ { p a r i t y } ^ { 1 } }$ did not change from Scenario (ii) to (iii) underscores the importance of considering both within- and cross-group metrics to understand sources of potential unfairness. Despite the apparent deterioration captured by $\mathrm { A U C } _ { M } \left( \mathrm { A U C } _ { M } = 0 . 8 8 \right.$ in Scenario (ii), to $\mathrm { A U C } _ { M } = 0 . 8 6 \mathrm { h e r e } )$ , male patients achieve better performance in terms of model discrimination, overall, because of the cross-group AUC contributions $( \mathrm { x A U C } _ { M , F } = 0 .$ .44 in Scenario (ii), raised to $\mathrm { x A U C } _ { M , F } = 0 . 5 3$ here).

Key takeaways: Within-group AUC parity can co-exist with cross-group AUC disparity. If cross-group AUC disparity is suficiently large, the resulting overall AUC could be lower (or higher) than both subgroups’ within-group AUC values. This artefact is created by AUC non-collapsibility and may complicate communication with stakeholders.

6.2.2 Model use case (2): Binary decision. Let us assume now that this last model – Scenario (iii) – is used for binary decisions, i.e., a threshold is applied to model estimated probabilities. Two plausible strategies could follow.

Strategy (i): One single threshold is applied to all patients. When a model is used to support decision-making, all patients (regardless of their subgroups) use a common decision threshold. With such strategy, relying on within-group discrimination metrics $( \mathrm { A U C _ { g } ) }$ alone is insuficient. Cross-group AUC terms are as important as within-group AUC values because together they determine how estimated probabilities are ordered when a single threshold is applied.

In this Scenario (iii) model, as shown in Figure ${ \mathrm { 2 c } } ,$ even when within-group AUCs are identical $( \mathrm { A U C } _ { M } = \mathrm { A U C } _ { F } =$ 0.86), $\mathrm { \ x A U C } _ { M , F } = 0 . 5 3$ versus $\mathrm { x A U C } _ { F , M } = 0 . 9 8$ indicate that it is only slightly above that chance (53% probability) that male patients with an event are assigned estimated probabilities that are higher than the ones assigned to female patients without an event, whereas female patients with an event will have a higher estimated probability than male patients without an event most times (98% probability). In settings where the decision is polar $( \mathrm { i . e . , }$ patients benefit from selection [25]), this asymmetry disadvantages male patients, as female patients are far more likely to exceed the decision threshold. With a single decision threshold, this asymmetry would result in unequal downstream decision (which could be captured by threshold-dependent fairness metrics [19]).

Two main mechanisms can explain how within-group AUC parity can coexist with cross-group AUC disparities. First, diferences in outcome prevalence across subgroups may reflect diferences in underlying risk, leading to shifts in the distributions of estimated probabilities. For example, female patients have a higher outcome prevalence $( \pi ^ { F } = 0 . 0 3 )$ than male patients $( \pi ^ { M } = 0 . 0 2 )$ , which may result in systematically higher estimated probabilities. Such diferences can be monitored using “mean score parity” metrics $[ 1 6 ,$ 19]. Second, diferences in calibration across subgroups can lead to asymmetric cross-group rankings, even when within-group AUCs are identical. For instance, if risk is systematically underestimated for male patients but not for female patients, the resulting diferences in estimated probabilities may introduce cross-group AUC disparities [16].

Key takeaways: Applying a single decision threshold to a model with $\mathrm { A U C } _ { \mathrm { p a r i t y } } = 0$ but $\mathrm { x A U C _ { p a r i t y } \ \gg }$ 0 is expected to result in unequal downstream decisions, underscoring the importance of reporting within- and cross-group discrimination to prevent downstream unequal decisions.

Strategy (ii): Subgroup specific thresholds are defined and applied. When cross-group AUC disparities exist $( \mathrm { x A U C _ { p a r i t y } = }$ 0.45) but no within-group AUC disparities exist $( \mathrm { A U C _ { p a r i t y } } = 0 )$ , as seen in Scenario (iii), subgroup specific thresholds should be considered. If distinct decision thresholds are specified for each subgroup, cross-group comparisons no longer directly influence decision-making and the within-group AUC parity becomes the only consequential fairness metric in this context. Fairness considerations in terms of discrimination only concern within-group ${ \mathrm { A U C s } } ,$ as cross-group “competition” is eliminated and cross-group ranking ceases to be a consideration. Subgroup specific thresholds (such as “race corrections” [30]) are not uncommon in medicine.

Key takeaways: Subgroup specific decision thresholds should be considered when $\mathrm { \Delta x A U C _ { p a r i t y } }$ ≫ 0. With subgroup specific decision thresholds, cross-group ranking ceases to be a consideration, and fairness in terms of discrimination solely depends on $\mathrm { { A U C } _ { \mathrm { { p a r i t y } } } }$

6.2.3 Model use case (3): Patient ranking. Finally, let us assume that the model is being used to rank patients. In settings where only the ordering of individuals matters, such as surgery prioritisation, absolute risk magnitudes become irrelevant. Decisions in this context are inherently polar, as patients are “competing” against each other (and against the “system”) to get an estimated probability that places them on top of the list [25]. Traditional within-group AUC parity-based fairness analyses are insuficient to assess ranking/discrimination fairness, and xAUC components are equally relevant. Metrics that combine both within- and cross-group AUCs, such as the $\mathrm { \bf x A U C ^ { 1 } }$ and $\mathrm { \ x A U C } ^ { 0 }$ , are therefore well suited to summarise fairness in terms of discrimination in this setting (Figure 2).

## 7 Conclusions and Recommendations

In this paper, we examined eight AUC-based fairness metrics, characterised their estimands, critically appraised their interpretation using a clinical example, and provided guidance on how to use and interpret these metrics in the context of diferent types of fairness assessments. The use and interpretation of these metrics are inherently context-dependent, and no single metric can fully capture the complexity of fairness in model discrimination when subgroups coexist. Therefore, we recommend reporting the AUC matrix as a whole, containing the overall AUC and all within- and cross-group AUC components, to allow discrimination performance to be evaluated comprehensively. When fairness metrics are needed (for example, when more than two subgroups coexist, resulting in large AUC matrices, or when a summary measure of fairness is preferable), we recommend within-group AUC parity as the primary fairness metric for discrimination performance. Cross-group AUC parity can provide important additional information on cross-group ranking disparities but should be used with caution due to its potential conflict with calibration and case-mix diferences across subgroups. Transparent reporting of population characteristics [16] and adherence to TRIPOD+AI [7] are essential for meaningful fairness evaluations of clinical predictive AI. Although this paper focused on discrimination and AUC-based fairness metrics, their interplay with calibration is important and warrants further investigation, especially regarding the potential trade-ofs between discrimination and calibration fairness

## Generative AI Usage Statement

Generative AI technologies were used to assist with text review, LaTeX debugging, and Python code generation during the preparation of this manuscript. Specifically, ChatGPT Edu was employed between May 2025 and November 2026, using diferent versions of large language models (LLMs) over time. These tools supported manuscript preparation but were not used to draft the manuscript itself. No sensitive, proprietary, or human-subject data were shared with LLMs. All AI-generated outputs were critically reviewed and validated by the authors before incorporation, who take full responsibility for the content and integrity of this work.

## Acknowledgments

JM is funded by a Clarendon Fund Scholarship at the University of Oxford. GSC and RDR are supported by the National Institute for Health and Care Research (NIHR) Biomedical Research Centre: Birmingham at University Hospitals Birmingham NHS Foundation Trust and the University of Birmingham; and an NIHR-MRC Better Methods Better Research Grants (MR/Z503873/1). GSC and RDR are NIHR Senior Investigators. BVC is supported by the Research Foundation – Flanders (FWO) grant G097322N, Kom Op Tegen Kanker grant 13583, and KU Leuven Internal Funds grant C24M/20/064. The funders of the study had no role in study design, data collection, data analysis, interpretation of data, writing of the report, or decision to submit the article for publication. The views expressed are those of the authors and not necessarily represent the views of the funding agencies

## Ethical Considerations Statement

This study aims to improve evaluation of fairness in clinical prediction models. We highlight the limitations of existing AUC-based fairness metrics and provide recommendations for their appropriate use and interpretation. Misinterpretation or inappropriate application of these metrics could lead to misleading conclusions about model performance and fairness, potentially afecting healthcare decisions and exacerbating existing health inequalities. Our recommendations aim to support more transparent and comprehensive fairness assessments, while recognising that statistical fairness metrics alone cannot establish whether a clinical prediction model is fair or safe for deployment.

## Code Availability

The Python code used to reproduce the simulations, analyses, and figures presented in the EuroSCORE II case study (Section 6) is publicly available as a Jupyter notebook on Google Colab. The code is distributed under the MIT Licence.

## References

[1] Isabela Albuquerque, Jessica Schrouf, David Warde-Farley, Taylan Cemgil, Sven Gowal, and Olivia Wiles. 2024. Evaluating model bias requires characterizing its mistakes. arXiv preprint arXiv:2407.10633 (2024). doi:10.48550/arXiv.2407.10633

[2] Thomas F Byrd, Bronwyn Southwell, Adarsh Ravishankar, Travis Tran, Abhinab Kc, Tom Phelan, Genevieve B Melton-Meaux, Michael G Usher, Daren Scheppmann, Sean Switzer, et al. 2023. Validation of a proprietary deterioration index model and performance in hospitalized adults. JAMA Network Open 6, 7 (2023), e2324176–e2324176. doi:10.1001/jamanetworkopen.2023.24176

[3] John Chalmers, Mark Pullan, Brian Fabri, James McShane, Matthew Shaw, Neeraj Mediratta, and Michael Poullis. 2013. Validation of EuroSCORE II in a modern cohort of patients undergoing cardiac surgery. European Journal ofCardio-Thoracic Surgery 43, 4 (2013), 688–694. doi:10.1093/ejcts/ezs406

[4] Trenton Chang, Michael W Sjoding, and Jenna Wiens. 2022. Disparate censorship & undertesting: A source of label bias in clinical machine learning In Machine Learning for Healthcare Conference. PMLR, 343–390. https://pubmed.ncbi.nlm.nih.gov/37152303/

[5] Victoria Cheng, Vinith M Suriyakumar, Natalie Dullerud, Shalmali Joshi, and Marzyeh Ghassemi. 2021. Can you fake it until you make it? impacts of diferentially private synthetic data on downstream classification fairness. In Proceedings ofthe 2021 ACM Conference on Fairness, Accountability, and Transparency. 149–160. doi:10.1145/3442188.3445879

[6] Gary S Collins, Paula Dhiman, Jie Ma, Michael M Schlussel, Lucinda Archer, Ben Van Calster, Frank E Harrell, Glen P Martin, Karel GM Moons, Maarten Van Smeden, et al. 2024. Evaluation of clinical prediction models (part 1): from development to external validation. bmj 384 (2024) doi:10.1136/bmj-2023-074819

[7] Gary S Collins, Karel GM Moons, Paula Dhiman, Richard D Riley, Andrew L Beam, Ben Van Calster, Marzyeh Ghassemi, Xiaoxuan Liu, Johannes B Reitsma, Maarten Van Smeden, et al. 2024. TRIPOD+ AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. bmj 385 (2024). doi:10.1136/bmj-2023-078378

[8] Sen Cui, Weishen Pan, Changshui Zhang, and Fei Wang. 2023. Bipartite ranking fairness through a model agnostic ordering adjustment. IEEE Transactions on Pattern Analysis and Machine Intelligence (2023). doi:10.48550/arXiv.2307.14668

[9] Emily Diana, Wesley Gill, Michael Kearns, Krishnaram Kenthapadi, and Aaron Roth. 2021. Minimax group fairness: Algorithms and experiments. In Proceedings ofthe 2021 AAAI/ACM Conference on AI, Ethics, and Society. 66–76. doi:10.1145/3461702.3462523

[10] Jack Gallifant, Danielle S Bitterman, Leo Anthony Celi, Judy W Gichoya, Joao Matos, Liam G McCoy, and Robin L Pierce. 2024. Ethical debates amidst flawed healthcare artificial intelligence metrics. NPJ Digital Medicine 7, 1 (2024), 243. doi:10.1038/s41746-024-01242-1

[11] Antonio Garcia-Valentin, Carlos A Mestres, Eduardo Bernabeu, José A Bahamonde, Iván Martín, Cristina Rueda, Alberto Domenech, Jamit Valencia, Delfina Fletcher, Facundo Machado, et al. 2016. Validation and quality measurements for EuroSCORE and EuroSCORE II in the Spanish cardiac surgical population: a prospective, multicentre study. European Journal ofCardio-Thoracic Surgery 49, 2 (2016), 399–405. doi:10.1093/ejcts/ezv090

[12] A Cecile JW Janssens and Forike K Martens. 2020. Reflection on modern methods: Revisiting the area under the ROC Curve. International journal of epidemiology 49, 4 (2020), 1397–1403. doi:10.1093/ije/dyz274

[13] Mehmet Kalender, Taylan Adademir, Mehmet Tasar, Ata Niyazi Ecevit, Okay Guven Karaca, Salih Salihi, Fuat Buyukbayrak, and Mehmet Ozkokeli. 2014. Validation of EuroSCORE II risk model for coronary artery bypass surgery in high-risk patients. Kardiochirurgia i Torakochirurgia Polska/Polish Journal ofThoracic and Cardiovascular Surgery 11, 3 (2014), 252–256. https://pubmed.ncbi.nlm.nih.gov/26336431/

[14] Nathan Kallus and Angela Zhou. 2019. The fairness of risk scores beyond classification: Bipartite ranking and the xauc metric. Advances in neural information processing systems 32 (2019). doi:10.48550/arXiv.1902.05826

[15] Yan Luo, Yu Tian, Min Shi, Louis R Pasquale, Lucy Q Shen, Nazlee Zebardast, Tobias Elze, and Mengyu Wang. 2024. Harvard glaucoma fairness: a retinal nerve disease dataset for fairness learning and fair identity normalization. IEEE Transactions on Medical Imaging (2024). doi:10.1109/TMI.2024.3377552

[16] João Matos, Ben Van Calster, Richard D. Riley, Paula Dhiman, and Gary S. Collins. 2026. Collapsibility of Performance Metrics in Clinical Predictiv AI. arXiv preprint arXiv:2608.30568 (2026). doi:10.48550/arXiv.2608.30568

[17] João Matos, Paula Dhiman, and Gary S. Collins. 2026. Clarifying the SOFA-2 Score. JAMA 335, 15 (04 2026), 1359–1360. doi:10.1001/jama.2026.0310

[18] João Matos, Jack Gallifant, Anand Chowdhury, Nicoleta Economou-Zavlanos, Marie-Laure Charpignon, Judy Gichoya, Leo Anthony Celi, Lama Nazer, Heather King, and An-Kwok Ian Wong. 2024. A clinician’s guide to understanding bias in critical clinical prediction models. Critical Care Clinics 40, 4 (2024), 827–857. doi:10.1016/j.ccc.2024.05.011

[19] João Matos, Ben Van Calster, Leo Anthony Celi, Paula Dhiman, Judy Wawira Gichoya, Richard D Riley, Chris Russell, Sara Khalid, and Gary S Collins. 2026. Critical Appraisal of Fairness Metrics in Clinical Predictive AI. The Lancet Digital Health (2026). doi:10.1016/j.landig.2026.10100

[20] Melissa D McCradden, Shalmali Joshi, Mjaye Mazwi, and James A Anderson. 2020. Ethical limitations of algorithmic fairness solutions in health care machine learning. The Lancet Digital Health 2, 5 (2020), e221–e223. doi:10.1016/s2589-7500(20)30065-0

[21] Felix Meissen, Svenja Breuer, Moritz Knolle, Alena Buyx, Ruth Müller, Georgios Kaissis, Benedikt Wiestler, and Daniel Rückert. 2024. (Predictable) performance bias in unsupervised anomaly detection. Ebiomedicine 101 (2024). doi:10.1016/j.ebiom.2024.105002

[22] Brent Mittelstadt, Sandra Wachter, and Chris Russell. 2023. The Unfairness of Fair Machine Learning: Levelling down and strict egalitarianism by default. (2023). arXiv:2302.02404 [cs.AI] https://arxiv.org/abs/2302.02404

[23] Samer AM Nashef, François Roques, Linda D Sharples, Johan Nilsson, Christopher Smith, Antony R Goldstone, and Ulf Lockowandt. 2012. Euroscor ii. European journal ofcardio-thoracic surgery 41, 4 (2012), 734–745. doi:10.1093/ejcts/ezs04

[24] Luke Oakden-Rayner, Jared Dunnmon, Gustavo Carneiro, and Christopher Ré. 2020. Hidden stratification causes clinically meaningful failures in machine learning for medical imaging. Proc. ACM Conf. Health Inference Learn. 2020 (April 2020), 151–159. doi:10.1145/3368555.3384468

[25] Jessica K. Paulus and David M. Kent. 2020. Predictably unequal: Understanding and addressing concerns that algorithmic clinical prediction may increase health disparities. npj Digital Medicine 3, 1 (Jul 2020). doi:10.1038/s41746-020-0304-9

[26] Michael J Pencina and Ralph B D’Agostino, Sr. 2015. Evaluating discrimination of risk prediction models: The C statistic. JAMA 314, 10 (Sept. 2015), 1063. doi:10.1001/jama.2015.11082

[27] Martin Silverborn, Susanne Nielsen, and Martin Karlsson. 2023. The performance of EuroSCORE II in CABG patients in relation to sex, age, and surgical risk: a nationwide study in 14,118 patients. Journal ofCardiothoracic Surgery 18, 1 (2023), 40. doi:10.1186/s13019-023-02141-4

[28] Eric J Topol. 2019. High-performance medicine: the convergence of human and artificial intelligence. Nature medicine 25, 1 (2019), 44–56. doi:10.1038/s41591-018-0300-7

[29] Ben Van Calster, Gary S Collins, Andrew J Vickers, Laure Wynants, Kathleen F Kerr, Lasai Barreñada, Gael Varoquaux, Karandeep Singh, Karel GM Moons, Tina Hernandez-Boussard, et al. 2025. Evaluation of performance measures in predictive artificial intelligence models to support medical decisions: overview and guidance. The Lancet Digital Health (Dec. 2025). doi:10.1016/j.landig.2025.100916

[30] Darshali A Vyas, Leo G Eisenstein, and David S Jones. 2020. Hidden in plain sight—reconsidering the use of race correction in clinical algorithms New England Journal ofMedicine 383, 9 (2020), 874–882. doi:10.1056/NEJMms2004740

## A Relations between $\mathrm { \tt X A U C ^ { 1 } } .$ x $\mathrm { U C } ^ { 0 }$ and AUC

Starting from the overall AUC,

$$
\mathrm { A U C } = P ( \hat { \rho } _ { 1 } > \hat { p } _ { 0 } ) = \sum _ { i \in \{ A , B \} } \sum _ { j \in \{ A , B \} } P ( g = i \mid Y = 1 ) P ( g = j \mid Y = 0 ) P ( \hat { p } _ { 1 } ^ { i } > \hat { p } _ { 0 } ^ { j } ) .
$$

Collapsing over the negative group yields $\mathrm { \bf x A U C ^ { 1 } }$

$$
\begin{array} { r l } & { \mathrm { A U C } = \displaystyle \sum _ { i \in \{ A , B \} } P ( g = i \mid Y = 1 ) \sum _ { j \in \{ A , B \} } P ( g = j \mid Y = 0 ) P ( \hat { p } _ { 1 } ^ { i } > \hat { p } _ { 0 } ^ { j } ) } \\ & { \quad \quad = \displaystyle \sum _ { i \in \{ A , B \} } P ( g = i \mid Y = 1 ) P ( \hat { p } _ { 1 } ^ { i } > \hat { p } _ { 0 } ) } \\ & { \quad \quad = \displaystyle \sum _ { i \in \{ A , B \} } P ( g = i \mid Y = 1 ) \mathrm { x A U C } _ { i } ^ { 1 } . } \end{array}\tag{20}
$$

Collapsing over the positive group yields xAUC<sup>0</sup>:

$$
\begin{array} { r l } & { \mathrm { A U C } = \displaystyle \sum _ { j \in \{ A , B \} } P ( g = j \mid Y = 0 ) \sum _ { i \in \{ A , B \} } P ( g = i \mid Y = 1 ) P ( \hat { p } _ { 0 } ^ { j } < \hat { p } _ { 1 } ^ { i } ) } \\ & { \quad \quad = \displaystyle \sum _ { j \in \{ A , B \} } P ( g = j \mid Y = 0 ) P ( \hat { p } _ { 1 } > \hat { p } _ { 0 } ^ { j } ) } \\ & { \quad \quad = \displaystyle \sum _ { j \in \{ A , B \} } P ( g = j \mid Y = 0 ) \mathrm { x A U C } _ { j } ^ { 0 } . } \end{array}\tag{21}
$$

Substituting $P ( g = i \mid Y = 1 ) = p _ { 1 } ^ { i } / p _ { 1 }$ and $P ( g = j \mid Y = 0 ) = p _ { 0 } ^ { j } / p _ { 0 }$ gives the concise decomposition:

$$
\mathrm { A U C } = \frac { p _ { 1 } ^ { A } } { p _ { 1 } } \mathrm { { x A U C } } _ { A } ^ { 1 } + \frac { \dot { p } _ { 1 } ^ { B } } { \dot { p } _ { 1 } } \mathrm { { x A U C } } _ { B } ^ { 1 } = \frac { \dot { p } _ { 0 } ^ { A } } { \dot { p } _ { 0 } } \mathrm { { x A U C } } _ { A } ^ { 0 } + \frac { \dot { p } _ { 0 } ^ { B } } { \dot { p } _ { 0 } } \mathrm { { x A U C } } _ { B } ^ { 0 }\tag{22}
$$

Comparing Equation 2 with the $\mathrm { \tt x A U C ^ { 1 } }$ decomposition (20) yields

$$
\mathrm { x A U C } _ { A } ^ { 1 } = \frac { p _ { 0 } ^ { A } } { p _ { 0 } } \mathrm { A U C } _ { A } + \frac { p _ { 0 } ^ { B } } { p _ { 0 } } \mathrm { x A U C } _ { A , B }\tag{23}
$$

$$
\mathrm { x A U C } _ { B } ^ { 1 } = \frac {  { p _ { 0 } ^ { A } } } {  { p _ { 0 } } } \mathrm { x A U C } _ { B , A } + \frac {  { p _ { 0 } ^ { B } } } {  { p _ { 0 } } } \mathrm { A U C } _ { B }\tag{24}
$$

and, similarly, for $\mathrm { \mathbf { x } A U C } ^ { 0 }$ decomposition 21:

$$
\mathrm { x A U C } _ { A } ^ { 0 } = \frac { p _ { 1 } ^ { A } } { p _ { 1 } } \mathrm { A U C } _ { A } + \frac { p _ { 1 } ^ { B } } { p _ { 1 } } \mathrm { x A U C } _ { B , A }\tag{25}
$$

$$
\mathrm { x A U C } _ { B } ^ { 0 } = \frac {  { p _ { 1 } ^ { A } } } {  { p _ { 1 } } } \mathrm { x A U C } _ { A , B } + \frac {  { p _ { 1 } ^ { B } } } {  { p _ { 1 } } } \mathrm { A U C } _ { B }\tag{26}
$$