# LongAgent: History-Guided Agentic Search for Longitudinal Outcome Prediction

Siyao Wang<sup>1,2⋆</sup>, Florian Guitton<sup>1</sup>, Shuojie Fu<sup>1</sup>, Guanyu Tao<sup>1</sup>, Kai Sun<sup>1</sup>, and Wenjia Bai<sup>1,2,3</sup>

<sup>1</sup> Data Science Institute, Imperial College London, London, UK

<sup>2</sup> Department of Computing, Imperial College London, London, UK

3 Department of Brain Sciences, Imperial College London, London, UK s.wang18@imperial.ac.uk

Abstract. Extracting informative representations from longitudinal data that can predict future outcomes remains a critical challenge in medicine. Medical datasets are inherently heterogeneous, consisting of a large number of variables collected from diferent sources, sampled with diferent temporal spacings, and representing diferent aspects of human health status. This requires identifying those variables with predictive value, processing longitudinal information, and integrating multiple variables for outcome prediction. Here, we propose a novel agent-based approach, LongAgent, that can autonomously search over combinations of variable sets, temporal windows and longitudinal aggregation functions, and identify candidates with promising predictive performance. LongAgent utilises a history memory of previous searches and numerical evidence to guide subsequent exploration. On synthetic data, LongAgent achieves a mean prediction RMSE of 1.7376 and improves over the strongest nonagent baseline by 0.0151 (95% CI: [0.0045, 0.0260]; p = 0.0273). On a real clinical dataset, it performs comparably to the best baseline.

Keywords: Agentic search · Predictive modelling · Longitudinal data · Longitudinal outcome prediction

## 1 Introduction

In medical research, longitudinal outcome prediction involves collecting multimodal measurements of an individual, monitoring their longitudinal changes, and developing a predictive model that can learn multimodal features from the data for predicting future outcomes [2]. Typically, features, also known as predictors, are extracted by selecting a subset of recorded variables, followed by choosing an empirical time window and an aggregation function that can be applied to longitudinal measurements. Given the complexity of clinical longitudinal datasets and the exploratory nature of medical research, feature extraction is often performed in a hand-crafted manner and is time-consuming.

Repeated monitoring of outcomes adds another layer of complexity. In longitudinal studies, the outcome may be recorded at multiple time points for each participant. Predictors must therefore be constructed only from measurements preceding each observed outcome to avoid information leakage [2]. However, the appropriate duration of this preceding time window is often unknown. Dunn et al. found that the performance of predictive models using wearable device measurements varied with the monitoring time window and its proximity to the outcome date [6]. FIDDLE and flexible-window electronic health record (EHR) methods explore diferent ways to define the time window [8, 18]. Based on the time window, diferent aggregation functions have been explored to integrate longitudinal measurements to form predictive features [3, 7, 13].

Recently, AI agents, implemented as large language models (LLMs), have demonstrated great potential to either perform predefined workflows or discover new knowledge [4,12,19]. For longitudinal medical data analysis, PHIA has been proposed, which combines code generation and information retrieval to answer questions about wearable device data [15]. FeatEHR-LLM utilises an agent to generate executable feature extraction code for analysing irregular time series EHR data [10]. CoDaS coordinates hypothesis generation, analysis, grounding, and validation for discovering wearable device biomarkers [11]. LLM-FE proposes validation-guided evolution to search over a space of feature transformation programs to process tabular data [1].

These studies demonstrate the potential of LLM agents but primarily focus on generating code for feature transformation and data analysis. The proposed method, LongAgent, focuses on search for longitudinal outcome predictors. It uses accumulated evidence to guide the search for a combination of variable subset, time window and aggregation function that leads to predictive features for future outcomes. Our main contributions are:

1. We formulate the discovery of predictive features as a budgeted search problem, over a search space of variable subsets, time windows, and aggregation functions.

2. We introduce a history-guided search agent that combines search status, candidate history, coverage summary, and transition evidence, and uses a two-stage decision process to decide the next move.

3. We evaluate LongAgent against three non-agent search policies on a large number of synthetic datasets constructed with diverse settings and a real longitudinal cohort, and demonstrate LongAgent outperforms or on par with other search policies.

## 2 Methods

Problem formulation. Consider a longitudinal medical dataset containing repeated measurements and outcomes from a cohort of participants. The objective is to extract predictive features from these measurements and fit a model onto the features to predict each participant’s outcome.

Denote the number of subjects as $N _ { ☉ }$ , the number of recorded variables as $M .$ and the dataset of all variables as $X = \{ x _ { i , j } | 1 \leq i \leq N , 1 \leq j \leq M \}$ , where i denotes the subject index and $j$ denotes the variable index. Each variable $x _ { i , j }$ is measured multiple times in the longitudinal study with $x _ { i , j } [ t ]$ denoting the measurement at time t. Note that the time interval between measurements can be irregular for diferent subjects and for diferent variables. For each subject, the outcome is monitored longitudinally. Denote the set of outcomes as $Y =$ $\{ y _ { i } \mid 1 \le i \le N \}$ , where $y _ { i }$ denotes a sequence of outcomes for subject $i ,$ and $y _ { i } [ \tau ] \in$ R denotes the outcome observed at time τ.

The proposed AI agent searches over a space of candidates, defined by a variable subset, an outcome-relative time window, and an aggregation function. Each candidate c is formulated as $c = ( S , w , f )$ , where $S \subseteq \{ 1 , \dots , M \}$ denotes the indices of the selected variables, w denotes the time window preceding an outcome, and $f$ denotes an aggregation function applied to measurements within that window. Applying candidate c to the longitudinal measurements of subject i yields a feature vector or predictor $z _ { i }$ for the selected variables $S { \mathrm { : } }$

$$
\begin{array} { r l } & { z _ { i , j } = f \big ( \{ x _ { i , j } [ t ] \mid \tau - w < t \leq \tau \} \big ) , \qquad j \in S , } \\ & { \quad z _ { i } = \big ( z _ { i , j } \big ) _ { j \in S } } \end{array}\tag{1}
$$

Let $Z = \{ z _ { i } \mid 1 \leq i \leq N \}$ denote the set of features or predictors. The performance of candidate c can be evaluated by fitting a model onto $\{ Z , Y \}$ and computing the prediction error $l ( c )$ . The objective is to find the optimal candidate c that minimises $l ( c )$

Iterative search. Figure 1 illustrates the procedure. The agent starts the search from an initial candidate $c ^ { ( 0 ) } = ( S ^ { ( 0 ) } , w ^ { ( 0 ) } , f ^ { ( 0 ) } )$ , where ${ \bf \it \Delta } _ { w } \mathrm { ( 0 ) }$ and $f ^ { ( 0 ) }$ are predefined initial choices for the time window and aggregation function. Applying $w ^ { ( 0 ) }$ and $f ^ { ( 0 ) }$ to each recorded variable constructs one feature per variable. The variables are ranked by the absolute Pearson correlation between these features and the outcome, and $S ^ { ( 0 ) }$ is initialised with the highest-ranked variable.

After initialisation, the agent performs iterative search under an evaluation budget $B ,$ i.e. the maximum number of search steps. At iteration $n ,$ it sets either the current candidate or the previously best performing candidate to be the base candidate $b ^ { ( n ) } = ( S _ { b } ^ { ( n ) } , w _ { b } ^ { ( n ) } , f _ { b } ^ { ( n ) } )$ , and proposes an action $a ^ { ( n ) } \in \left\{ \begin{array} { l l } { \begin{array} { r l r l } \end{array} } \end{array} \right.$ {add, drop, replace, update, stop} to modify the base candidate. For the non-stop actions, applying $a ^ { ( n ) }$ to the base candidate $b ^ { ( n ) }$ produces the next candidate:

$$
c ^ { ( n ) } = \left\{ \begin{array} { l l } { ( S _ { b } ^ { ( n ) } \cup \{ k \} , w _ { b } ^ { ( n ) } , f _ { b } ^ { ( n ) } ) , } & { a ^ { ( n ) } = \mathrm { a d d } ( k ) , } \\ { ( S _ { b } ^ { ( n ) } \setminus \{ k \} , w _ { b } ^ { ( n ) } , f _ { b } ^ { ( n ) } ) , } & { a ^ { ( n ) } = \mathrm { d r o p } ( k ) , } \\ { ( ( S _ { b } ^ { ( n ) } \setminus \{ k \} ) \cup \{ k ^ { \prime } \} , w _ { b } ^ { ( n ) } , f _ { b } ^ { ( n ) } ) , } & { a ^ { ( n ) } = \mathrm { r e p l a c e } ( k , k ^ { \prime } ) , } \\ { ( S _ { b } ^ { ( n ) } , w ^ { \prime } , f ^ { \prime } ) , } & { a ^ { ( n ) } = \mathrm { u p d a t e } ( w ^ { \prime } , f ^ { \prime } ) . } \end{array} \right.\tag{2}
$$

The first three actions add, drop, or replace a recorded variable in $S .$ The update action changes the time window $w _ { \mathrm { i } }$ , the aggregation function $f ,$ or both.

![](images/e9c8ad182a87bc487506e972c6066433fee60b1b42144156c5cdf876db27c4c6.jpg)  
Fig. 1. History-guided agentic search. A candidate $c = ( S , w , f )$ specifies a variable subset, an outcome-relative window, and an aggregation function. The evaluator fits a prediction model onto features extracted using $^ { c , }$ and records the loss in the search history. The agent integrates four sources of evidence: search status, candidate history, coverage summary, and transition evidence, and then selects a base candidate and the next move. The agent either continues the search or stops when the budget is exhausted.

Previously evaluated candidates are excluded from the space of search, ensuring that $c ^ { ( n ) }$ has not been evaluated before. The stop action terminates the search without generating the next candidate.

History-guided search agent. At iteration $n ,$ the agent receives history information $p ^ { ( n ) }$ coming from four sources $p ^ { ( n ) } = [ s ^ { ( n ) } , h ^ { ( n - 1 ) } , g ^ { ( n - 1 ) } , e ^ { ( n - 1 ) } ]$ , where:

1. Search status and constraints $s ^ { ( n ) }$ : the current candidate, remaining budget, permitted variables, windows and aggregation functions, and legal candidate updates. It also includes the variable ranking used in initialisation.

2. Candidate history $h ^ { ( n - 1 ) }$ : an ordered sequence $\{ c ^ { ( m ) } , l ( c ^ { ( m ) } ) \} _ { m = 0 } ^ { n - 1 }$ of previously evaluated candidates and their prediction errors.

3. Coverage summary $g ^ { ( n - 1 ) }$ : the variables, window and aggregation combinations that have already been evaluated, which helps the agent define the unexplored space.

4. Transition evidence $e ^ { ( n - 1 ) }$ : the previous transitions from base candidates to modified candidates, along with the change of prediction performance as evidence.

LongAgent is implemented as an LLM, queried in two stages using diferent prompt templates. In the first stage, LongAgent selects a base candidate $b ^ { ( n ) }$ and an action type $a ^ { ( n ) }$ based on the evidence $p ^ { ( n ) }$ . In the second stage, LongAgent is provided with the same evidence, including $p ^ { ( n ) }$ , and the selected pair $( \bar { b ^ { ( n ) } } , a ^ { ( n ) } )$

from the first stage and a list of parameters for action $a ^ { ( n ) }$ . It then selects the parameter for $a ^ { ( n ) }$ from the list. The selected action, equipped with the selected parameter, is applied to modify $b ^ { ( n ) }$ and generate the next candidate $c ^ { ( n ) }$ , which is evaluated and added to the search history. When the LLM selects the stop action or the evaluation budget is exhausted, the method returns $\hat { c } =$ arg min $\mathfrak { l } _ { c \in \mathcal { E } _ { B } } l ( c )$ , where $\mathcal { E } _ { B }$ denotes the set of all candidates evaluated within budget B.

## 3 Experiments

Synthetic longitudinal datasets. Synthetic data are generated in two stages: longitudinal measurements followed by outcome generation.

Stage 1: Longitudinal measurement generation. For subject i and variable $j ,$ the measurement at time t is simulated as:

$$
x _ { i , j } [ t ] = \lambda _ { j } \cdot u _ { i } + d _ { i , j } \cdot t + c _ { i , j } \cdot \mathrm { s i n } \Bigg ( \frac { 2 \pi h ( t ) } { 2 4 } \Bigg ) + \epsilon _ { i , j } [ t ] .\tag{3}
$$

Here, $u _ { i }$ denotes a subject-level efect shared across variables, $\lambda _ { j }$ denotes the variable-level loading, $d _ { i , j } \cdot t$ models linear temporal drift at time $t ,$ and $c _ { i , j }$ sin $\mathsf { \Omega } _ { 1 } ( 2 \pi h ( t ) / 2 4 )$ models the daily cycle with amplitude $c _ { i , j } . h ( t )$ denotes the hour of day, and $\epsilon _ { i , j } [ t ] \sim \mathcal { N } ( 0 , \sigma _ { X } ^ { 2 } )$ denotes Gaussian measurement noise.

Stage 2: Outcome generation. For subject i and variable $j ,$ , let $r _ { i , j } [ \tau ]$ denote the aggregated measurement within the outcome-relative window:

$$
r _ { i , j } [ \tau ] = f ^ { \star } \big ( \{ x _ { i , j } [ t ] ~ | ~ \tau - w ^ { \star } < t \leq \tau \} \big ) .\tag{4}
$$

Here, $w ^ { \star }$ denotes the temporal window and $f ^ { \star }$ denotes the aggregation function. The outcome $y _ { i } ^ { ( s ) } [ \tau ]$ is generated from the combined efects of $r _ { i , j } [ \tau ]$ 2 $j = 1 , \dots , M$ , as follows:

$$
\begin{array} { r l } & { y _ { i } ^ { ( s ) } [ \tau ] = \displaystyle \sum _ { j \in S _ { \mathrm { l i n } } ^ { ( s ) } } \beta _ { j } \cdot r _ { i , j } [ \tau ] + \displaystyle \sum _ { j \in S _ { \mathrm { n o n } } ^ { ( s ) } } \gamma _ { j } \cdot | r _ { i , j } [ \tau ] | + \displaystyle \sum _ { ( j , k ) \in \mathcal { P } _ { \mathrm { i n t } } ^ { ( s ) } } \theta _ { j k } \cdot r _ { i , j } [ \tau ] \cdot r _ { i , k } [ \tau ] } \\ & { ~ + ~ \delta _ { u } ^ { ( s ) } \cdot u _ { i } + \delta _ { b } ^ { ( s ) } \cdot b _ { i } + \xi _ { i } ^ { ( s ) } [ \tau ] . } \end{array}\tag{5}
$$

Here, the superscript $s \in \{ 1 , \ldots , 6 \}$ identifies one of the six outcome generation settings (Table 1), each defined by a diferent combination of terms. $S _ { \mathrm { l i n } } ^ { ( s ) }$ and $S _ { \mathrm { n o n } } ^ { ( s ) }$ denote the sets of variables with linear and absolute value efects, with coeficients $\beta _ { j }$ and $\gamma _ { j } . \mathcal { P } _ { \mathrm { i n t } } ^ { ( s ) }$ denotes the set of interacting variable pairs, and $\theta _ { j k }$ denotes the corresponding interaction coeficient for pair $( j , k )$ $u _ { i }$ and $b _ { i }$ denote shared and independent subject-level efects, and $\delta _ { u } ^ { ( s ) }$ and $\delta _ { b } ^ { ( s ) }$ denote their coeficients. Finally, $\xi _ { i } ^ { ( s ) } [ \tau ] \sim \mathcal { N } ( 0 , \sigma _ { Y } ^ { 2 } )$ denotes the Gaussian outcome noise.

Table 1. Six synthetic outcome generation settings, using diferent combinations of linear terms, absolute value terms, interaction pairs, and subject-level efect coeficients in Eq. 5. | · | denotes the cardinality of a set, i.e. the number of variables.
<table><tr><td>Setting  $| S _ { \mathrm { l i n } } ^ { ( s ) } |$ </td><td> $\vert S _ { \mathrm { n o n } } ^ { ( s ) } \vert$   $| \mathcal { P } _ { \mathrm { i n t } } ^ { ( s ) }$  t1  $\delta _ { u } ^ { ( s ) }$   $\delta _ { b } ^ { ( s ) }$ </td></tr><tr><td>Linear 6</td><td>0 0 0 0</td></tr><tr><td>Nonlinear</td><td>2 4 0 0</td></tr><tr><td>Interaction</td><td>0 2 0 2 0 0</td></tr><tr><td>Shared factor</td><td>6 0 0 κ 0</td></tr><tr><td>Independent subject 6</td><td>0 0 0 κ</td></tr><tr><td>Combined</td><td>2 2 1 0 κ</td></tr></table>

Clinical longitudinal dataset. We use digital home monitoring data from IDEA-FAST, a prospective multinational study that followed over 1,000 participants for up to 24 weeks [14]. All data were pseudonymised and collected with ethical approval and written consent. Longitudinal measurements were recorded using a chest-worn VitalPatch and a lower-back Axivity AX6 inertial sensor, with 62 variables in total, consisting of 44 heart-rate variability, 4 VitalPatch summary, 7 activity, and 7 sit-to-stand or stand-to-sit variables. In terms of outcomes, 32,128 fatigue scores were recorded for 1,084 participants, which were normalised to z-scores.

Implementation. Each synthetic dataset contains 500 subjects, 100 variables measured every 30 minutes for 15 days, and one daily outcome per subject. We independently sample $u _ { i }$ and $b _ { i }$ from $\mathcal { N } ( 0 , 1 )$ , and $\lambda _ { j } , d _ { i , j } , c _ { i , j }$ , and $\epsilon _ { i , j } [ t ]$ from $\mathcal { N } ( 0 , 1 )$ . The magnitude of the active coeficient is set to $\kappa \in \{ 1 , 0 . 5 , 0 . 2 5 \}$ , and $| \beta _ { j } | = | \gamma _ { j } | = | \theta _ { j k } | = \kappa$ . For each term, the coeficient is assigned randomly to either +κ or −κ. σ<sub>X</sub> is set to 1 and $\sigma _ { Y } ~ \in ~ \{ 0 . 5 , 1 \}$ . We consider six outcome generation settings, 3 coeficient magnitudes for κ and 2 outcome noise levels for $\sigma _ { Y }$ . For each combination, we generate 10 datasets using diferent random seeds, yielding 360 synthetic datasets in total.

For synthetic data, the search space of time windows is {8, 12, 16, 20, 24, 32, 48} hours and the space of aggregation functions is {mean, median, p<sub>95</sub>, p<sub>05</sub>}, p denoting percentile. For IDEA-FAST, the space of windows is {4, 8, 12, 16, 20, 24, 32, 48} hours and the same space of aggregation functions is used. Synthetic and IDEA-FAST searches are initialised with $( w ^ { ( 0 ) } , f ^ { ( 0 ) } ) = ( 1 2 \mathrm { h }$ , mean) and (8h, mean), respectively. In both experiments, candidates contain at most ten variables, with a budget of $B = 3 0$ evaluations. Synthetic experiments use DeepSeek-V4-Flash [5] at temperature 0.01 with a 1,024-token limit and thinking disabled. For privacy reasons, IDEA-FAST experiments use gpt-oss-120b [16] at temperature 0.01.

Evaluation protocol and baseline methods. For both synthetic and real datasets, data is split subject-wise into 80% for training and 20% for test. The loss l(c) is computed using five-fold cross-validation on the training set. The evaluator fits an elastic net model $( \alpha = 0 . 1$ and equal $\ell _ { 1 }$ and $\ell _ { 2 }$ mixing, with $\ell _ { 1 }$ ratio = 0.5) and assesses the prediction performance in terms of the root mean squared error (RMSE). After the final candidate cˆ is selected, the evaluator fits the model to the full training set and evaluates the prediction performance on the test set, in terms of the RMSE, mean absolute error (MAE), and the coeficient of determination $( R ^ { 2 } )$ . The prediction target is the simulated outcome for synthetic datasets, and fatigue z-scores for IDEA-FAST.

We compare LongAgent with three baselines. Greedy variable selection retains fixed time window and aggregation function, and adds variables in decreasing order of absolute Pearson correlation [9, 17]. Random candidate search randomly samples a legal, previously unevaluated one-step neighbour of the current candidate. Rule-based candidate search follows the correlation ranking before exploring time window and aggregation function that reduce validation loss of the evaluator. Confidence intervals are computed by resampling the ten seed blocks. Statistical comparison of two methods is performed using the two-sided Wilcoxon test on seed-level mean RMSE diferences.

## 4 Results

Table 2. Comparison of LongAgent with baseline methods on 360 synthetic datasets. Values are mean $\pm$ standard deviation across datasets; $p \textmd { - }$ values compare RMSE between LongAgent and each baseline.
<table><tr><td>Method</td><td>RMSE↓</td><td>MAE↓</td><td> $R ^ { 2 } \uparrow$ </td><td>p</td></tr><tr><td>Greedy variable selection</td><td> $1 . 7 8 7 1 \pm 1 . 0 0 6 0$ </td><td> $1 . 3 3 5 7 \pm 0 . 6 8 5 7$ </td><td> $0 . 3 1 6 3 \pm 0 . 1 7 5 6$ </td><td>0.0020</td></tr><tr><td>Random candidate search</td><td> $1 . 8 0 9 4 \pm 1 . 0 1 3 9$ </td><td> $1 . 3 5 5 6 \pm 0 . 6 9 5 7$ </td><td> $0 . 2 9 9 9 \pm 0 . 1 7 4 3$ </td><td>0.0020</td></tr><tr><td>Rule-based search</td><td> $1 . 7 5 2 7 \pm 1 . 0 0 2 4$ </td><td> $1 . 3 0 9 1 \pm 0 . 6 8 1 7$ </td><td> $0 . 3 3 6 4 \pm 0 . 1 9 1 6$ </td><td>0.0273</td></tr><tr><td>LongAgent</td><td> $\mathbf { 1 . 7 3 7 6 \pm 1 . 0 0 0 3 }$ </td><td>1.2984 ± 0.6802</td><td> $\mathbf { 0 . 3 4 8 2 \pm 0 . 1 9 4 3 }$ </td><td></td></tr></table>

Results on synthetic data. Table 2 compares LongAgent with the three baseline methods, averaged across 360 synthetic datasets. LongAgent achieves the lowest RMSE (1.7376) and MAE (1.2984), and the highest $R ^ { \bar { 2 } }$ (0.3482) among all methods. Rule-based search is the second strongest method, which explores the same candidate space as LongAgent with a deterministic search policy. However, it follows the correlation ranking in search, while LongAgent leverages a rich history of evidence to guide the next move. Using two-sided Wilcoxon test, LongAgent outperforms rule-based search with statistical significance $\left( p < 0 . 0 5 \right)$ . Ablation study. Table 3 investigates the efects of diferent evidence sources in the agent history. The basic history configuration contains search status and candidate history. By adding coverage summary to history, RMSE decreases from 1.2982 to 1.2918, and MAE decreases from 0.9773 to 0.9730, with $R ^ { 2 }$ increasing, all indicating better prediction performance. By further adding transition

Table 3. Performance of LongAgent under three history configurations on 30 synthetic datasets. Values are mean ± standard deviation across datasets; $^ { 6 6 } + \ '$ denotes evidence added cumulatively across rows, and p-values compare each augmented configuration with the basic history configuration.
<table><tr><td>History configuration</td><td>RMSE↓</td><td>MAE↓</td><td> $R ^ { 2 } \uparrow$ </td><td> $p$ </td></tr><tr><td>Basic history</td><td> $1 . 2 9 8 2 \pm 0 . 4 5 2 5$ </td><td> $0 . 9 7 7 3 \pm 0 . 3 1 8 4$ </td><td> $0 . 3 9 9 3 \pm 0 . 2 0 2 9$ </td><td></td></tr><tr><td>+ Coverage summary</td><td> $1 . 2 9 1 8 \pm 0 . 4 5 5 8$ </td><td> $0 . 9 7 3 0 \pm 0 . 3 2 0 7$ </td><td> $0 . 4 0 5 9 \pm 0 . 2 0 4 3$ </td><td>0.3223</td></tr><tr><td>+ Transition evidence</td><td> $\mathbf { 1 . 2 8 4 1 \pm 0 . 4 5 9 6 }$ </td><td> $\mathbf { 0 . 9 6 7 0 \pm 0 . 3 2 3 4 }$ </td><td> $\mathbf { 0 . 4 1 4 5 \pm 0 . 2 0 3 6 }$ </td><td>0.0488</td></tr></table>

evidence to history, LongAgent achieves the lowest RMSE $( 1 . 2 8 4 1 \pm 0 . 4 5 9 6 )$ and MAE $( 0 . 9 6 7 0 \pm 0 . 3 2 3 4 )$ , and the highest $R ^ { 2 } ~ ( 0 . 4 1 4 5 \pm 0 . 2 0 \dot { 3 } 6 )$ . Its RMSE is significantly lower than that obtained with basic history $\left( p < 0 . 0 5 \right)$ , suggesting that transition evidence, such as past loss changes, provides informative signals for LongAgent to plan subsequent candidate updates.

Table 4. Comparison of LongAgent with baseline methods on a real clinical dataset, IDEA-FAST. Values report test performance on the subject-held-out split.
<table><tr><td>Method</td><td>RMSE↓</td><td>MAE↓</td><td> $R ^ { 2 } \uparrow$ </td></tr><tr><td>Greedy variable selection</td><td>0.9957</td><td>0.8197</td><td>0.0173</td></tr><tr><td>Random candidate search</td><td>1.0012</td><td>0.8256</td><td>0.0063</td></tr><tr><td>Rule-based search</td><td>0.9939</td><td>0.8186</td><td>0.0208</td></tr><tr><td>LongAgent</td><td>0.9937</td><td>0.8187</td><td>0.0212</td></tr></table>

Results on IDEA-FAST. The real longitudinal data experiment compares LongAgent with the same three baseline methods on the IDEA-FAST dataset, with results reported in Table 4. LongAgent achieves the lowest RMSE (0.9937) and highest $R ^ { \bar { 2 } }$ (0.0212), outperforming the other methods. In terms of MAE, LongAgent only underperforms rule-based search by 0.0001. In terms of RMSE, LongAgent outperforms rule-based search by 0.0002, indicating comparable performance between the two methods on IDEA-FAST.

## 5 Discussion and Conclusions

We present a novel agent-based method, LongAgent, that can perform automated analysis for longitudinal medical data and discover predictive features for outcomes. LongAgent performs history-guided search over variable subsets, outcome-relative time windows, and aggregation functions. It constructs a rich history consisting of four sources of evidence to guide the search of the agent. Across 360 synthetic datasets, LongAgent achieved the lowest mean RMSE compared to three baseline methods. The ablation study demonstrates the usefulness of history guidance. On a real clinical dataset, the performance of LongAgent was slightly better or comparable to the strongest baseline. Current limitations include the use of a single shared time window and aggregation function for extracting features from selected variables. Future work will investigate variablespecific temporal representation learning and extend the method to additional longitudinal datasets with heterogeneous and missing data.

Acknowledgements The IDEA-FAST project has received funding from the Innovative Medicines Initiative 2 Joint Undertaking under grant agreement No. 853981. This Joint Undertaking receives support from the European Union’s Horizon 2020 research and innovation programme and EFPIA and associated partners. This communication reflects the view of the authors and neither IMI nor the European Union and EFPIA are liable for any use that may be made of the information contained herein. W.B. acknowledges the support of EPSRC CVD-Net Programme Grant (EP/Z531297/1) and BHF New Horizons Grant (NH/F/23/70013).

## References

1. Abhyankar, N., Shojaee, P., Reddy, C.K.: LLM-FE: Automated feature engineering for tabular data with LLMs as evolutionary optimizers. Transactions on Machine Learning Research (2026)

2. Cascarano, A., Mur-Petit, J., Hernández-González, J., Camacho, M., et al.: Machine and deep learning for longitudinal biomedical data: A review of methods and applications. Artificial Intelligence Review 56 (2023)

3. Christ, M., Braun, N., Neufer, J., Kempa-Liehr, A.W.: Time series FeatuRe extraction on basis of scalable hypothesis tests (tsfresh - a Python package). Neurocomputing 307 (2018)

4. Collaco, B.G., Haider, S.A., Prabha, S., Gomez-Cabello, C.A., Genovese, A., et al.: The role of agentic artificial intelligence in healthcare: a scoping review. npj Digital Medicine 9(1) (2026)

5. DeepSeek-AI: DeepSeek-V4 preview release. DeepSeek API Documentation (2026)

6. Dunn, J., Kidzinski, L., Runge, R., Witt, D., Hicks, J.L., et al.: Wearable sensors enable personalized predictions of clinical laboratory measurements. Nature Medicine 27(6) (2021)

7. Fulcher, B.D., Jones, N.S.: hctsa: A computational framework for automated timeseries phenotyping using massive feature extraction. Cell Systems 5(5), 527–531 (2017)

8. Gupta, M., Poulain, R., Phan, T.L.T., Bunnell, H.T., Beheshti, R.: Flexiblewindow predictions on electronic health records. AAAI Conference on Artificial Intelligence 36(11) (2022)

9. Guyon, I., Elisseef, A.: An introduction to variable and feature selection. Journal of Machine Learning Research 3 (2003)

10. Karami, H., Atienza, D., Thiran, J.P., Ionescu, A.: FeatEHR-LLM: Leveraging large language models for feature engineering in electronic health records. arXiv preprint arXiv:2604.22534 (2026)

11. Kim, Y., Rahman, S., Schmidgall, S., Park, C., Heydari, A.A., et al.: CoDaS: AI co-data-scientist for biomarker discovery via wearable sensors. arXiv preprint arXiv:2604.14615 (2026)

12. Lu, C., Lu, C., Lange, R.T., Yamada, Y., Hu, S., et al.: Towards end-to-end automation of AI research. Nature 651 (2026)

13. Lubba, C.H., Sethi, S.S., Knaute, P., Schultz, S.R., Fulcher, B.D., Jones, N.S.: catch22: CAnonical time-series CHaracteristics. Data Mining and Knowledge Discovery 33 (2019)

14. Maetzler, W., Avey, S., Pilotto, A., et al.: IDEA-FAST clinical study protocol: Identifying digital end-points of fatigue, sleep quality and daytime sleepiness in N = 2000. Digital Health 12 (2026)

15. Merrill, M.A., Paruchuri, A., Rezaei, N., Kovacs, G., Perez, J., et al.: Transforming wearable data into personal health insights using large language model agents. Nature Communications 17 (2026)

16. OpenAI: gpt-oss-120b and gpt-oss-20b model card. arXiv preprint arXiv:2508.10925 (2025)

17. Saeys, Y., Inza, I., Larrañaga, P.: A review of feature selection techniques in bioinformatics. Bioinformatics 23 (2007)

18. Tang, S., Davarmanesh, P., Song, Y., Koutra, D., Sjoding, M.W., Wiens, J.: Democratizing EHR analyses with FIDDLE: A flexible data-driven preprocessing pipeline for structured clinical data. Journal of the American Medical Informatics Association 27 (2020)

19. Wang, Z., Cai, L., Low, C.H., Liu, H., Wu, J., et al.: 3DMedAgent: Unified perception-to-understanding for 3d medical analysis. In: International Conference on Machine Learning (2026)