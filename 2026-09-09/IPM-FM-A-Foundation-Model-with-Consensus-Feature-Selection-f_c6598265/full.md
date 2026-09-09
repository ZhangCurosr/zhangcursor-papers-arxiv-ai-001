# IPM-FM: A Foundation Model with Consensus Feature Selection for Industrial Process Monitoring

Liang Cao

Weide Liu

Yan Qin

Department of Chemical and Biological Engineering

University of British Columbia

School of Computing and Artificial Intelligence School of Automation

Vancouver, BC, Canada

Jiangxi University of Finance and Economics Chongqing University

Nanchang, China

Chongqing, China

Jun Cheng Weisi Lin Bhushan Gopaluni Institute for Infocomm ResearchCollege of Computing and Data ScienceDepartment of Chemical and Biological Engineering A\*STAR Nanyang Technological University University of British Columbia Singapore Singapore Vancouver, BC, Canada

Abstract—Industrial process monitoring is fundamental to the safety and economic performance of modern process plants. Current practice remains a one-task-one-model paradigm that is label-inefficient and prone to degradation under operating drift. Foundation models have reshaped language, vision, and generic time-series forecasting, but it has not been adapted to industrial process monitoring. This setting poses domain-specific challenges, including safety-critical decisions and asymmetric sampling between process variables and laboratory measurements. We propose the industrial process monitoring foundation model (IPM-FM). It first learns general-purpose representations from unlabeled industrial process data through self-supervised pretraining, then adapts to specific monitoring tasks using a small amount of task-labeled data, and finally produces calibrated predictions through an uncertainty-aware prediction head. IPM-FM integrates a self-supervised Informer backbone with a multicriteria consensus feature selector, a recursive lag-feature regression head, and a calibrated Monte Carlo dropout uncertainty module. On a seven-year hydrotreater dataset for diesel flashpoint soft sensing, IPM-FM attains an RMSE of 2.99, R<sup>2</sup> of 0.50, and 97% coverage of its 95% predictive interval, outperforming the strongest classical and from-scratch sequence baselines by 8.3% and 14.6% in RMSE respectively, supporting the viability of a unified pretraining–adaptation framework for industrial process monitoring.

Index Terms—Foundation model, industrial process monitoring, soft sensing, self-supervised pretraining, causal feature selection, uncertainty quantification.

## I. INTRODUCTION

Modern industrial plants increasingly rely on dense sensor networks [1]. Soft sensing infers hard-to-measure quality variables from cheap online measurements [2], [3]. Recent automated soft-sensor design tools have further highlighted the need for deployable machine-learning workflows in industrial applications [4]. Fault detection flags significant deviations from a learned normal operating region. Fault diagnosis isolates the root cause among known failure modes, while prognostic methods estimate the remaining useful life of critical equipment. Although these tasks consume the same multivariate process time series, current practice addresses each one with a dedicated model trained from scratch on its own labeled history.

This one-task-one-model paradigm carries three fundamental limitations. First, it is data inefficient because task labels are scarce and expensive while modern neural models are labelhungry. Second, it offers little transferability because knowledge extracted from one task or plant is rarely reused, even though industrial signals share common dynamical primitives such as oscillations, step responses, and slow drifts. Third, it degrades quickly under operating drift because models fitted to a specific operating window fail when feedstock, catalyst activity, or ambient conditions shift, and each drift then forces an expensive retraining cycle on every affected task. This issue is very common in multimode industrial processes, where adaptive monitoring across operating modes is required [5]. Addressing these issues requires decoupling representation learning from task supervision so that a single learned representation can serve many monitoring tasks.

Such decoupling is the contribution of foundation models in language and vision, where a single backbone is pretrained on massive unlabeled data with self-supervised objectives and then adapted to heterogeneous downstream tasks with minimal labels [6], [7]. The same framework has recently been extended to generic time-series data through large pretrained backbones, with notable examples including Chronos [8] and TimesFM [9] for zero-shot forecasting and Moirai [10] and MOMENT [11] for general-purpose representation. Earlier representation learners such as TS2Vec [12] together with taskspecific backbones such as PatchTST [13] and TimesNet [14] have further pushed the state of the art on standard forecasting and classification benchmarks.

Complementary progress in geometric representation learning has produced important methodological advances for extracting structured and interpretable primitives from complex physical observations. In particular, BPNet and its extension provide an advanced framework for Bezier primitive segmen-´ tation and decomposition directly from irregular 3D point clouds, demonstrating that deep networks can learn compact, primitive-level abstractions from unstructured geometric data [15], [16]. Nevertheless, these advances are designed for generic time-series forecasting or geometric data abstraction rather than industrial process monitoring. They treat data either as generic numerical sequences or as geometric point sets, rather than as signals from physically coupled actuators and sensors inside a controlled plant.

Transferring the foundation-model to industrial process monitoring therefore faces three domain-specific challenges. First, industrial plants carry dozens to hundreds of sensor channels, only a task-dependent subset of which carries useful signal. Spurious channels become a dominant failure mode whenever raw multivariate input is fed directly into a shared backbone. Second, monitoring decisions are safety-critical and calibrated uncertainty becomes a hard requirement rather than an optional feature [17], [18]. Third, the sampling rate between process variables and laboratory measurements is severely mismatched. Laboratory information must therefore be effectively exploited to keep the downstream model up to date.

Motivated by these gaps, we propose industrial process monitoring foundation model (IPM-FM), a model that addresses all three challenges within a single unified architecture. We validated the framework for diesel flash-point soft sensing at a commercial hydrotreating unit. IPM-FM outperforms the strongest classical and from-scratch sequence baselines by 8.3% and 14.6% RMSE respectively. The main contributions of this paper are summarized as follows:

• We propose IPM-FM, the first foundation-model framework tailored to industrial process monitoring, which decouples representation learning from task supervision and unifies multiple monitoring tasks under a shared selfsupervised backbone.

• We design a multi-criteria consensus feature selector that fuses tree-based, spectral, and causal evidence, with a theoretical guarantee of multiplicative suppression of spurious channels.

• We introduce a recursive lag-feature mechanism that exploits sparse but highly informative laboratory measurements, together with a calibrated MC dropout wrapper that delivers predictive intervals suitable for risk-aware deployment.

## II. IPM-FM FRAMEWORK

Figure 1 illustrates the overall architecture of IPM-FM, which is organized into a two-stage pipeline. In the pretraining stage, a shared backbone is trained on unlabeled multivariate process data using self-supervised objectives that do not require any monitoring labels. In the adaptation stage, the pretrained backbone is combined with a consensus feature selection module, a recursive lag-feature builder, and a calibrated uncertainty head to produce a downstream monitor for a specific task. Switching among monitoring tasks is achieved by replacing only the lightweight task head while preserving the pretrained backbone and the consensus feature adaptation module, which yields the cross-task reusability characteristic of foundation-model pipelines.

## A. Notation and Problem Setup

Let $\mathbf { X } \in \mathbb { R } ^ { T \times D }$ denote a segment of multivariate process data with $T$ time steps and D sensor channels. Different downstream monitoring tasks correspond to different target spaces: for soft sensing, a scalar quality variable $y \in \mathbb { R }$ is associated with the final time step; for fault detection and diagnosis, a categorical label $c \in \{ 1 , \ldots , C \}$ is attached instead; for anomaly detection, no target label is required at inference. The pretraining set $\mathcal { D } _ { \mathrm { p r e } } \doteq \{ \mathbf { X } ^ { ( i ) } \} _ { i = 1 } ^ { N _ { \mathrm { p r e } } }$ consists of unlabeled segments, while the downstream set $\mathcal { D } _ { \mathrm { d o w n } }$ contains aligned input-target pairs for the task of interest, with $\left| \mathcal { D } _ { \mathrm { d o w n } } \right| \ll N _ { \mathrm { p r e } } .$ The goal is to learn a backbone $g _ { \phi }$ pretrained on ${ \mathcal { D } } _ { \mathrm { p r e } }$ and a lightweight task head $h _ { \psi }$ such that $h _ { \psi } ( g _ { \phi } ( \mathbf { X } ) , \mathbf { l } )$ approximates the downstream target with calibrated uncertainty, where l collects the most recent laboratory measurements. In this paper we instantiate the regression setting for soft sensing, writing $\hat { y } ~ = ~ f _ { \theta } ( { \bf X } , \mathbf { l } )$ with $\theta \ = \ \left( \phi , \psi \right)$ ; the generalization to classification and anomaly heads is straightforward and is treated as future work.

## III. SELF-SUPERVISED PRETRAINING

## A. Backbone Architecture

The backbone is an encoder architecture derived from the Informer [19], which itself follows the standard transformer attention construction [20], chosen for its favorable tradeoff between long-range modeling capacity and computational cost on industrial sequences. Informer-based architectures have also shown promise for interpretable industrial soft-sensor design with long process sequences [21]. The encoder ingests a segment $\mathbf { X } \in \mathbf { \overline { { \mathbb { R } } } } ^ { T \times D }$ and produces a sequence of latent representations $\mathbf { H } \in \mathbb { R } ^ { T ^ { \prime } \times d _ { \mathrm { m o d e l } } }$ through stacked ProbSparse self-attention and distillation layers. Figure 2 shows the overall encoder structure used as the IPM-FM backbone. The same backbone is shared across all downstream monitoring tasks.

1) ProbSparse Self-Attention: Standard self-attention computes:

$$
\operatorname { A t t n } ( \mathbf { Q } , \mathbf { K } , \mathbf { V } ) = \operatorname { s o f t m a x } \left( { \frac { \mathbf { Q } \mathbf { K } ^ { \top } } { \sqrt { d _ { k } } } } \right) \mathbf { V } ,\tag{1}
$$

which scales quadratically with sequence length. To reduce the cost on long industrial sequences, the ProbSparse mechanism restricts the interaction to the top-u queries selected according to a sparsity measure:

$$
M _ { \mathrm { s p } } ( \mathbf { q } _ { i } ) = \operatorname* { m a x } _ { j } \mathbf { A } _ { i j } - \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \mathbf { A } _ { i j } ,\tag{2}
$$

where $\begin{array} { r } { \mathbf { A } = \mathbf { Q } \mathbf { K } ^ { \top } / \sqrt { d _ { k } } , } \end{array}$ , n is the sequence length, and $u =$ ⌈κ log n⌉ with $\kappa = 3$ . Only a sparse subset of dominant query– key interactions is retained, which sharply reduces memory and FLOPs on industrial sequences with hundreds to thousands of time steps.

![](images/277c07def7844a9e4071dd8c1c68a127949c85a745ddceb0263527fea9a4878a.jpg)  
Fig. 1. Two-stage IPM-FM pipeline: self-supervised pretraining of a shared backbone, followed by consensus feature selection, recursive lag features, and a calibrated uncertainty head for downstream adaptation.

![](images/4004e81a562ffb4915f9eaa886c6c96aeac6a001fc8b9cd03b8579cc308ae529.jpg)  
Fig. 2. Informer-based encoder backbone of IPM-FM, with stacked Prob-Sparse self-attention and distillation layers for long industrial sequences.

2) Distillation Layer: To compress the sequence length and reduce memory consumption, each encoder stage concludes with a distillation layer

$$
\begin{array} { r } { \mathbf { X } _ { \mathrm { o u t } } = \mathbf { M a x P o o l } \big ( \mathbf { E L U } ( \mathbf { W } * \mathbf { X } _ { \mathrm { i n } } + \mathbf { b } ) \big ) , } \end{array}\tag{3}
$$

which halves the sequence length while preserving the most informative activations.

## B. Self-Supervised Objectives

Pretraining is driven by two complementary objectives that do not require any task labels and therefore produce a representation that is reusable across monitoring tasks.

1) Masked Segment Reconstruction: Given an input segment X, we randomly mask a fraction r of non-overlapping sub-segments of length ℓ and task a lightweight reconstruction head to predict the masked values from the surrounding context:

$$
\mathcal { L } _ { \mathrm { r e c } } = \frac { 1 } { | \mathcal { M } | } \sum _ { ( t , d ) \in \mathcal { M } } \big ( \hat { \mathbf { X } } _ { t , d } - \mathbf { X } _ { t , d } \big ) ^ { 2 } ,\tag{4}
$$

where M is the index set of masked positions.

2) Operating-Regime Contrastive Learning: Industrial processes typically operate in a small number of recurrent regimes separated by transitions, and the ability to distinguish regimes is valuable for every downstream monitoring task. We exploit this structure with a contrastive objective that pulls together representations of segments from the same operating regime and pushes apart representations from different regimes. Regimes are obtained by applying k-means clustering to daily aggregated process statistics, which is a lightweight pseudo-labeling step that does not rely on any downstream label. The information noise-contrastive estimation loss is:

$$
\mathcal { L } _ { \mathrm { c o n t } } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \frac { \exp ( \sin ( \mathbf { z } _ { i } , \mathbf { z } _ { i } ^ { + } ) / \tau ) } { \sum _ { k \ne i } \exp ( \sin ( \mathbf { z } _ { i } , \mathbf { z } _ { k } ) / \tau ) } ,\tag{5}
$$

where $\mathbf { z } _ { i }$ is the pooled representation of segment i, ${ \bf z } _ { i } ^ { + }$ is a positive segment from the same regime, and τ is a temperature hyperparameter.

3) Total Pretraining Loss: The total pretraining loss is a weighted sum

$$
\mathcal { L } _ { \mathrm { p r e } } = \mathcal { L } _ { \mathrm { r e c } } + \lambda _ { \mathrm { c o n t } } \mathcal { L } _ { \mathrm { c o n t } } ,\tag{6}
$$

where $\lambda _ { \mathrm { c o n t } }$ is selected on a held-out validation split.

## IV. DOWNSTREAM TASK ADAPTATION

During downstream adaptation, the pretrained backbone is connected to three task-specific modules. A consensus feature selection module identifies the subset of sensor channels relevant to the target monitoring task. A recursive lag-feature builder augments the encoder embedding with prior laboratory measurements, and a calibrated uncertainty head produces predictive intervals or class probabilities depending on the task.

## A. Multi-Criteria Consensus Feature Selection

Industrial plants carry dozens to hundreds of sensor channels, many of which are irrelevant or redundant for any specific monitoring target. We propose a consensus framework that fuses three complementary families of feature selection methods.

1) Tree-Based Importance: Tree-based ensembles including LightGBM [22], Extra Trees [23], Random Forest [24], and CatBoost [25] rank features by impurity decrease:

$$
I _ { j } = \sum _ { t \in \mathrm { I r e e s } } \sum _ { s \in S _ { j } ^ { t } } p ( s ) \cdot \Delta s ,\tag{7}
$$

where $S _ { j } ^ { t }$ is the set of nodes split on feature $j$ in tree $t , p ( s )$ is the fraction of samples reaching node $s ,$ and $\Delta s$ is the impurity decrease.

2) Spectral Similarity: Power spectral densities are computed via Welch’s method, and the similarity between feature i and the monitoring target y is scored as

$$
\rho _ { \mathrm { s p e c } } ( i ) = \frac { \sum _ { f } ( P _ { i } ( f ) - \bar { P } _ { i } ) ( P _ { y } ( f ) - \bar { P } _ { y } ) } { \sqrt { \sum _ { f } ( P _ { i } ( f ) - \bar { P } _ { i } ) ^ { 2 } \sum _ { f } ( P _ { y } ( f ) - \bar { P } _ { y } ) ^ { 2 } } } .\tag{8}
$$

This criterion captures dynamic similarity that is invariant to time delays and phase shifts.

3) Causal Discovery: Three causal discovery algorithms are applied: the PC algorithm for constraint-based conditional independence testing, FCI for handling latent confounders, and DirectLiNGAM for exploiting non-Gaussianity [26], [27]. Causality analysis has been used in industrial inferential sensing and stable soft-sensor modeling to improve feature relevance, interpretability, and robustness under changing operating conditions [28], [29]. Process-knowledge-guided causal discovery can further reduce incorrect causal relations when applying causal discovery algorithms to industrial process data [30]. The Markov Blanket of the monitoring target $y$ is used as the causal feature set.

$$
\mathbf { M B } ( y ) = \operatorname { P a r e n t s } ( y ) \cup \operatorname { C h i l d r e n } ( y ) \cup \operatorname { C o - p a r e n t s } ( y ) .\tag{9}
$$

4) Consensus Scoring: Each method m produces a score $S _ { m } ( f _ { i } )$ that is min-max normalized to $\bar { S } _ { m } ( f _ { i } ) \in [ 0 , 1 ]$ . The weighted consensus score is

$$
C ( f _ { i } ) = \sum _ { m \in \mathcal { M } } w _ { m } \cdot \bar { S } _ { m } ( f _ { i } ) \cdot \mathbb { I } ( \bar { S } _ { m } ( f _ { i } ) > \tau _ { m } ) ,\tag{10}
$$

with a voting count $\begin{array} { r } { V ( f _ { i } ) ~ = ~ \sum _ { m } \mathbb { I } ( \bar { S } _ { m } ( f _ { i } ) > \tau _ { m } ) } \end{array}$ and an agreement bonus

$$
B ( f _ { i } ) = \left\{ \begin{array} { l l } { 0 . 1 5 } & { V ( f _ { i } ) = 3 } \\ { 0 . 0 5 } & { V ( f _ { i } ) = 2 . } \\ { 0 } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{11}
$$

A feature is selected if and only if

$$
C ( f _ { i } ) + B ( f _ { i } ) \geq \tau _ { \mathrm { c o n s } } \ \land \ V ( f _ { i } ) \geq V _ { \operatorname* { m i n } } .\tag{12}
$$

We use $\tau _ { \mathrm { c o n s } } = 0 . 3 0 , \tau _ { m } = 0 . 3 0$ for every m, and $V _ { \mathrm { m i n } } = 2 ,$ with category weights $w = ( 0 . 4 0 , 0 . 3 0 , 0 . 3 0 )$ for tree-based, spectral, and causal criteria respectively.

Let $f _ { i }$ be a spurious feature and assume each method category independently evaluates $f _ { i }$ with false-positive rate $\alpha _ { m } ~ = ~ \mathbb { P } ( \bar { S } _ { m } ( f _ { i } ) ~ > ~ \tau _ { m } ~ |$ spurious). Under the consensus requirement $V ( f _ { i } ) \geq V _ { \mathrm { m i n } } = 2 .$ , the probability of selecting a spurious feature is bounded by

$$
\mathbb { P } ( \mathrm { s e l e c t ~ } f _ { i } \mid \mathrm { s p u r i o u s } ) \leq \sum _ { k = 2 } ^ { 3 } { \binom { 3 } { k } } \bar { \alpha } ^ { k } ( 1 - \bar { \alpha } ) ^ { 3 - k } ,\tag{13}
$$

where $\bar { \alpha } = \operatorname* { m a x } _ { m } \alpha _ { m }$ . The three method categories operate on distinct principles, and their errors on spurious features are approximately independent. A confounder that reduces tree impurity is unlikely to also match spectral signatures and pass conditional independence tests. Consequently, the consensus mechanism achieves multiplicative suppression of spurious features relative to any single-criterion method.

## B. Recursive Lag-Feature Mechanism

Industrial laboratory measurements are sampled far less frequently than process variables, yet they are precisely the ground-truth signal the model is trying to predict. Successive measurements of the same quality variable are also strongly autocorrelated, which makes the most recent laboratory value the single most informative input for predicting the next one. Failing to expose this signal to the model would discard the very information that distinguishes process monitoring from generic time-series forecasting.

We therefore augment the encoder embedding with a recursive lag vector before it enters the regression head. For each downstream query timestamp $t _ { q } ,$ the lag-feature builder retrieves the $N _ { \mathrm { l a g } }$ most recent laboratory measurements strictly before $t _ { q }$ and emits the vector

$$
\mathbf { l } ( t _ { q } ) = \left[ \tilde { y } _ { 1 } , \Delta _ { 1 } , m _ { 1 } , \ldots , \tilde { y } _ { N _ { \mathrm { l a g } } } , \Delta _ { N _ { \mathrm { l a g } } } , m _ { N _ { \mathrm { l a g } } } \right] \in \mathbb { R } ^ { 3 N _ { \mathrm { l a g } } } ,\tag{14}
$$

where $\tilde { y } _ { i }$ is the standardized i-th most recent lab value, $\Delta _ { i }$ is the elapsed time in hours, and $m _ { i } \in \{ 0 , 1 \}$ is a presence mask that fires only when the lag is within the maximum-staleness budget $G _ { \mathrm { m a x } }$

The mechanism is termed recursive because, when fresh laboratory measurements are temporarily unavailable, the model’s own previous predictions are fed back into the lag builder as surrogate inputs, allowing continuous operation between sampling events. We refer to inference with real lab values as the closed-loop mode and inference with model-generated surrogates as the open-loop mode.

## C. Calibrated Uncertainty Head

The adaptation head is a fully connected projection with two MC-dropout layers [31]. During inference, dropout remains active and M stochastic forward passes are executed end-to-end, yielding an empirical distribution $\{ \hat { y } _ { m } \} _ { m = 1 } ^ { M }$ . The predictive mean and variance are

$$
\mu _ { \mathrm { p r e d } } ( \mathbf { x } ) = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \hat { y } _ { m } , \qquad \sigma _ { \mathrm { p r e d } } ^ { 2 } ( \mathbf { x } ) = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \bigl ( \hat { y } _ { m } - \mu _ { \mathrm { p r e d } } \bigr ) ^ { 2 } .\tag{15}
$$

In practice the raw MC-dropout variance often under- or over-estimates the true error magnitude, which leaves the resulting predictive intervals miscalibrated. We therefore introduce a scalar temperature $\gamma > 0$ that rescales the predictive standard deviation, so that the 95% predictive interval under a Gaussian approximation becomes

$$
\begin{array} { r } { \mathbf { P I } _ { 9 5 } = \left[ \mu _ { \mathrm { p r e d } } - 1 . 9 6 \gamma \sigma _ { \mathrm { p r e d } } , ~ \mu _ { \mathrm { p r e d } } + 1 . 9 6 \gamma \sigma _ { \mathrm { p r e d } } \right] . } \end{array}\tag{16}
$$

The optimal temperature $\gamma ^ { * }$ is obtained by minimizing the Gaussian negative log-likelihood on a held-out calibration split, which admits the closed-form solution

$$
\gamma ^ { * } = \sqrt { \frac { 1 } { N _ { \mathrm { c a l } } } \sum _ { j = 1 } ^ { N _ { \mathrm { c a l } } } { \frac { ( y _ { j } - \mu _ { j } ) ^ { 2 } } { \sigma _ { j } ^ { 2 } } } } ,\tag{17}
$$

where $\{ ( y _ { j } , \mu _ { j } , \sigma _ { j } ) \} _ { j = 1 } ^ { N _ { \mathrm { c a l } } }$ are the labels, predictive means, and predictive standard deviations on the calibration split. This adapts the temperature-scaling principle of [18] from classification to regression. The same dropout-based machinery extends to classification monitoring heads by applying MCdropout to the softmax output.

For the soft sensing instantiation considered in this paper, the downstream loss combines prediction and regularization terms

$$
\mathcal { L } _ { \mathrm { f t } } ( \theta ) = \frac { 1 } { N _ { \mathrm { d o w n } } } \sum _ { j = 1 } ^ { N _ { \mathrm { d o w n } } } ( \hat { y } _ { j } - y _ { j } ) ^ { 2 } + \lambda \lVert \theta _ { \mathrm { h e a d } } \rVert _ { 2 } ^ { 2 } ,\tag{18}
$$

When IPM-FM is adapted to classification-based monitoring tasks, the squared error is replaced by a cross-entropy term acting on the same backbone features.

## V. EXPERIMENTAL VALIDATION

## A. Primary Downstream Task: Quality-Variable Soft Sensing

Among the downstream tasks covered by IPM-FM, soft sensing of safety-critical quality variables is arguably the most label-scarce and most directly coupled to online control; it therefore serves as a stringent primary benchmark for the framework. The specific task chosen is diesel flash-point soft sensing at a commercial hydrotreating unit of the Parkland refinery.

Flash point is a safety-critical property measured offline every 4–8 hours. We collected a seven-year dataset spanning June 2017 to June 2024, containing 6,157 quality samples. Each sample is aligned with 24 online process variables sampled every 10 min from the furnace, reactor, fractionator, and hydrogen system. Missing values were filled by interpolation, and all process variables were standardized. Figure 3 shows the laboratory flash point together with a representative subset of selected process variables. We adopt a temporal split: the period from June 2017 to March 2024 is used for unlabeled pretraining and supervised fine-tuning, the period from April 2024 to June 2024 is used as the test set. Performance is reported as RMSE, MAE, MAPE, $R ^ { 2 }$ , and 95% predictive-interval coverage.

![](images/97caaa417f72224a08fe84684525a5efe96897c857d97eaa624e3a54edff895a.jpg)

![](images/0e1c2b519ac291235e801cfc7624dfe438f93934f5a1d71a5dfe68e6c4b6837f.jpg)  
Fig. 3. Top panel: normalized flash-point trajectory over the seven-year dataset; bottom panel: consensus feature-selection scores per process tag, with green bands marking the selected channels.

## B. Baselines

We compare IPM-FM against two categories of baselines. The first category consists of classical machine learning methods, including decision tree, elastic net, orthogonal matching pursuit (OMP), partial least squares (PLS), K-neighbors, AdaBoost, random forest, gradient boosting, support vector machine (SVM) with an RBF kernel, LightGBM, extra trees, XGBoost, and CatBoost. The second category consists of sequence modeling baselines, including LSTM, Transformer, and Informer. The sequence baselines are run for 5 random seeds and we report mean ± standard deviation; classical learners are likewise run for 5 seeds where stochasticity exists. All sequence baselines use the same input format and training budget as IPM-FM. In particular, the Informer baseline uses the same encoder backbone family as IPM-FM but is trained from scratch without consensus feature selection or selfsupervised pretraining.

## C. Overall Performance

Table I summarizes the comparison across all baselines on the steady-state held-out window. IPM-FM achieves the lowest RMSE, MAE, and MAPE and the highest $R ^ { 2 }$ of any method tested. Compared to the strongest classical regressor (Partial Least Squares, RMSE 3.26), IPM-FM achieves an 8.3% RMSE reduction and a 22% increase in $R ^ { 2 }$ . Compared to the strongest from-scratch sequence baseline (LSTM with no FS and no pretraining, RMSE 3.50), it delivers a 14.6% RMSE reduction. Figure 4 visualizes the IPM-FM mean prediction together with its γ-calibrated 95% predictive interval and the per-sample residuals.

![](images/1ec7b0929966ff22c9ea988098ddd634683048bc91a9816626cf8abb4bad7cd1.jpg)

Fig. 4. IPM-FM flash-point prediction (red) with γ-calibrated 95% predictive interval against laboratory measurements (black)  
![](images/2445388c2bb1debf855206c9313f78f5ba2c6aca56f43a13a23999484727869c.jpg)  
Fig. 5. Closed-loop (red) and open-loop recursive (purple) flash-point predictions on test set.

The recursive lag mechanism supports two evaluation regimes. Closed-loop inference uses the actual laboratory measurement as the lag input. Open-loop recursive inference uses only training labels to initialize the lag history and passes the model’s own predictions forward through the lag builder. Figure 5 compares the two regimes on the test window, showing that the open-loop mode tracks the laboratory trajectory with only a small RMSE penalty over closed-loop inference.

## D. Ablation Study

We ablate each component of IPM-FM to isolate its individual contribution. Table II reports the results. Removing the consensus feature selection causes the largest degradation (RMSE 2.99 → 3.72, $\Delta R ^ { 2 } = - 0 . 2 6 )$ , confirming that multicriteria channel selection is the primary driver of performance. Removing the recursive lag features causes the second-largest degradation (RMSE 2.99 → 3.22), demonstrating that the autoregressive coupling to prior laboratory values cannot be replaced by process-variable information alone. Removing self-supervised pretraining increases the RMSE to 3.11, indicating that the masked reconstruction and operating-regime contrastive objectives provide transferable representations for downstream regression.

Among the consensus-criterion ablations, removing the treebased or spectral criterion degrades RMSE to 3.08 and 3.09 , respectively, while removing the causal criterion slightly improves RMSE to 2.94. This suggests that tree-based and spectral evidence provide the strongest selection signals in this dataset, whereas the causal-discovery output is partly redundant with them for this particular soft-sensing task. We retain the causal criterion because it adds process-knowledge consistency and interpretability, even when its marginal effect on RMSE is dataset-dependent. Overall, the ablation results show that IPM-FM benefits primarily from consensus feature selection and recursive lag features, with self-supervised pretraining and MC-dropout calibration improving representation quality and uncertainty reliability, respectively.

## VI. CONCLUSION

This paper addressed the label inefficiency and limited transferability of the prevailing one-task-one-model practice in industrial process monitoring by proposing IPM-FM, a foundation-model framework that decouples representation learning from task supervision. Experiments on a seven-year commercial hydrotreater dataset confirmed that the framework adapts to a label-scarce soft sensing task with calibrated predictions. IPM-FM attained an RMSE of 2.99 and an R<sup>2</sup> of 0.50, while holding 97% empirical coverage on its 95% predictive interval. It also outperformed the strongest classical and from-scratch sequence baselines by 8.3% and 14.6% RMSE. The methodological core consists of a consensus feature selector that suppresses spurious channels through crosscriterion agreement, a recursive lag mechanism that brings sparse laboratory values into inference, and a regression-side adaptation of temperature scaling for MC-dropout intervals. The unified pretraining–adaptation pipeline gives operating plants a route to reuse representation learning across the soft sensing, fault detection, and prognostic tasks they currently maintain in isolation.

## REFERENCES

[1] S. Yin, S. X. Ding, X. Xie, and H. Luo, “A review on basic data-driven approaches for industrial process monitoring,” IEEE Transactions on Industrial Electronics, vol. 61, no. 11, pp. 6418–6428, 2014.

[2] P. Kadlec, B. Gabrys, and S. Strandt, “Data-driven soft sensors in the process industry,” Computers & Chemical Engineering, vol. 33, no. 4, pp. 795–814, 2009.

[3] L. Cao, J. Wang, J. Su, Y. Luo, Y. Cao, R. D. Braatz, and B. Gopaluni, “Comprehensive analysis on machine learning approaches for interpretable and stable soft sensors,” IEEE Transactions on Instrumentation and Measurement, vol. 74, pp. 1–17, 2025.

[4] L. Cao, J. Su, E. Conde, L. C. Siang, Y. Cao, and R. B. Gopaluni, “A novel automated soft sensor design tool for industrial applications based on machine learning,” Control Engineering Practice, vol. 160, p. 106322, 2025.

[5] L. Cao, X. Ji, Y. Cao, and R. B. Gopaluni, “Adaptive process monitoring for multimode industrial processes through machine learning,” IEEE Journal of Emerging and Selected Topics in Industrial Electronics, vol. 6, no. 4, pp. 1819–1827, 2025.

[6] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pretraining of deep bidirectional transformers for language understanding,” in Proceedings ofthe Annual Conference ofthe North American Chapter of the Association for Computational Linguistics, 2019, pp. 4171–4186.

[7] K. He, X. Chen, S. Xie, Y. Li, P. Dollar, and R. Girshick, “Masked´ autoencoders are scalable vision learners,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 16000–16009.

[8] A. F. Ansari, L. Stella, C. Turkmen, and others, “Chronos: Learning the language of time series,” arXiv preprint arXiv:2403.07815, 2024.

[9] A. Das, W. Kong, R. Sen, and Y. Zhou, “A decoder-only foundation model for time-series forecasting,” in Proceedings of the International Conference on Machine Learning, 2024, pp. 10148–10167.

TABLE I  
FLASH-POINT SOFT SENSING PERFORMANCE OF IPM-FM AGAINST CLASSICAL AND FROM-SCRATCH SEQUENCE BASELINES.
<table><tr><td>Model</td><td>RMSE</td><td>MAE</td><td>MAPE(%)</td><td> $R ^ { 2 }$ </td></tr><tr><td>Decision Tree</td><td> $6 . 5 4 \pm 0 . 1 4$ </td><td> $5 . 1 6 \pm 0 . 1 1$ </td><td> $7 . 6 6 \pm 0 . 1 6$ </td><td> $\cdot 1 . 3 6 \pm 0 . 1 0$ </td></tr><tr><td>Elastic Net</td><td> $4 . 5 1 \pm 0 . 0 0$ </td><td> $3 . 7 1 \pm 0 . 0 0$ </td><td> $5 . 5 6 \pm 0 . 0 0$ </td><td> $- 0 . 1 2 \pm 0 . 0 0$ </td></tr><tr><td>OMP</td><td> $4 . 4 7 \pm 0 . 0 0$ </td><td> $3 . 5 4 \pm 0 . 0 0$ </td><td> $5 . 2 7 \pm 0 . 0 0$ </td><td> $- 0 . 1 0 \pm 0 . 0 0$ </td></tr><tr><td>Gradient Boosting</td><td> $4 . 2 8 \pm 0 . 0 2$ </td><td> $3 . 0 9 \pm 0 . 0 1$ </td><td> $4 . 6 7 \pm 0 . 0 1$ </td><td> $- 0 . 0 1 \pm 0 . 0 1$ </td></tr><tr><td>K-Neighbors</td><td> $4 . 2 7 \pm 0 . 0 0$ </td><td> $3 . 3 8 \pm 0 . 0 0$ </td><td> $4 . 9 5 \pm 0 . 0 0$ </td><td> $- 0 . 0 1 \pm 0 . 0 0$ </td></tr><tr><td>XGBoost</td><td> $4 . 2 3 \pm 0 . 0 0$ </td><td> $3 . 2 4 \pm 0 . 0 0$ </td><td> $4 . 8 8 \pm 0 . 0 0$ </td><td> $0 . 0 1 \pm 0 . 0 0$ </td></tr><tr><td>LightGBM</td><td> $4 . 1 3 \pm 0 . 0 0$ </td><td> $3 . 1 7 \pm 0 . 0 0$ </td><td> $4 . 7 7 \pm 0 . 0 0$ </td><td> $0 . 0 6 \pm 0 . 0 0$ </td></tr><tr><td>CatBoost</td><td> $4 . 0 8 \pm 0 . 1 0$ </td><td> $3 . 1 8 \pm 0 . 1 0$ </td><td> $4 . 8 0 \pm 0 . 1 5$ </td><td> $0 . 0 8 \pm 0 . 0 5$ </td></tr><tr><td>AdaBoost</td><td> $3 . 9 1 \pm 0 . 0 3$ </td><td> $3 . 0 5 \pm 0 . 0 3$ </td><td> $4 . 5 1 \pm 0 . 0 5$ </td><td> $0 . 1 5 \pm 0 . 0 1$ </td></tr><tr><td>Transformer</td><td> $3 . 9 0 \pm 0 . 1 7$ </td><td> $3 . 1 4 \pm 0 . 1 6$ </td><td> $4 . 5 7 \pm 0 . 2 3$ </td><td> $0 . 1 6 \pm 0 . 0 7$ </td></tr><tr><td>Extra Trees</td><td> $3 . 7 6 \pm 0 . 0 2$ </td><td> $2 . 8 2 \pm 0 . 0 4$ </td><td> $4 . 2 4 \pm 0 . 0 5$ </td><td> $0 . 2 2 \pm 0 . 0 1$ </td></tr><tr><td>Random Forest</td><td> $3 . 7 5 \pm 0 . 0 3$ </td><td> $2 . 7 9 \pm 0 . 0 2$ </td><td> $4 . 2 1 \pm 0 . 0 3$ </td><td> $0 . 2 2 \pm 0 . 0 1$ </td></tr><tr><td>SVM (RBF)</td><td> $3 . 7 2 \pm 0 . 0 0$ </td><td> $2 . 7 1 \pm 0 . 0 0$ </td><td> $4 . 0 3 \pm 0 . 0 0$ </td><td> $0 . 2 4 \pm 0 . 0 0$ </td></tr><tr><td>Informer (scratch)</td><td> $3 . 6 8 \pm 0 . 1 8$ </td><td> $2 . 9 1 \pm 0 . 2 2$ </td><td> $4 . 2 4 \pm 0 . 3 0$ </td><td> $0 . 2 5 \pm 0 . 0 7$ </td></tr><tr><td>LSTM</td><td> $3 . 5 0 \pm 0 . 1 8$ </td><td> $2 . 7 3 \pm 0 . 1 6$ </td><td> $3 . 9 8 \pm 0 . 2 2$ </td><td> $0 . 3 2 \pm 0 . 0 7$ </td></tr><tr><td>Partial Least Squares</td><td> $3 . 2 6 \pm 0 . 0 0$ </td><td> $2 . 4 6 \pm 0 . 0 0$ </td><td> $3 . 6 5 \pm 0 . 0 0$ </td><td> $0 . 4 1 \pm 0 . 0 0$ </td></tr><tr><td>IPM-FM (ours)</td><td> ${ \bf 2 . 9 9 \pm 0 . 0 3 }$ </td><td> ${ \bf 2 . 1 4 \pm 0 . 0 3 }$ </td><td> ${ \bf 3 . 2 1 \pm 0 . 0 4 }$ </td><td> $\mathbf { 0 . 5 0 \pm 0 . 0 1 }$ </td></tr></table>

TABLE II  
COMPONENT-WISE ABLATION OF IPM-FM.
<table><tr><td>Variant</td><td>RMSE</td><td>MAE</td><td> $R ^ { 2 }$ </td></tr><tr><td>Full IPM-FM</td><td>2.99</td><td>2.14</td><td>0.50</td></tr><tr><td>w/o causal criterion</td><td>2.94</td><td>2.11</td><td>0.52</td></tr><tr><td>w/o tree-based criterion</td><td>3.08</td><td>2.19</td><td>0.48</td></tr><tr><td>w/o spectral criterion</td><td>3.09</td><td>2.15</td><td>0.47</td></tr><tr><td>w/o pretraining</td><td>3.11</td><td>2.23</td><td>0.47</td></tr><tr><td>w/o recursive lag features</td><td>3.22</td><td>2.42</td><td>0.43</td></tr><tr><td>w/o consensus FS (full module)</td><td>3.72</td><td>2.84</td><td>0.24</td></tr></table>

[10] G. Woo, C. Liu, A. Kumar, C. Xiong, S. Savarese, and D. Sahoo, “Unified training of universal time series forecasting transformers,” in Proceedings of the International Conference on Machine Learning, 2024, pp. 53140–53164.

[11] M. Goswami, K. Szafer, A. Choudhry, Y. Cai, S. Li, and A. Dubrawski, “MOMENT: A family of open time-series foundation models,” in Proceedings of the International Conference on Machine Learning, 2024, pp. 16115–16152.

[12] Z. Yue, Y. Wang, J. Duan, T. Yang, C. Huang, Y. Tong, and B. Xu, “TS2Vec: Towards universal representation of time series,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2022, pp. 8980–8987.

[13] Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, “A time series is worth 64 words: Long-term forecasting with transformers,” in Proceedings of the International Conference on Learning Representations, 2023.

[14] H. Wu, T. Hu, Y. Liu, H. Zhou, J. Wang, and M. Long, “TimesNet: Temporal 2D-variation modeling for general time series analysis,” in Proceedings of the International Conference on Learning Representations, 2023.

[15] R. Fu, C. Wen, Q. Li, X. Xiao, and P. Alliez, “BPNet: Bezier primitive´ segmentation on 3D point clouds,” in Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, 2023, pp. 754– 762.

[16] R. Fu, Q. Li, C. Wen, N. An, and F. Tang, “A novel framework for learning Bezier decomposition from 3D point clouds,” ´ IEEE Transactions on Circuits and Systems for Video Technology, vol. 35, no. 5, pp. 4329– 4340, 2025.

[17] J. Gawlikowski, C. R. N. Tassi, M. Ali, and others, “A survey of uncertainty in deep neural networks,” Artificial Intelligence Review, vol. 56, pp. 1513–1589, 2023.

[18] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in Proceedings of the International Confer-

ence on Machine Learning, 2017, pp. 1321–1330.

[19] H. Zhou, S. Zhang, J. Peng, S. Zhang, J. Li, H. Xiong, and W. Zhang, “Informer: Beyond efficient transformer for long sequence timeseries forecasting,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2021, pp. 11106–11115.

[20] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,” in Proceedings of the Advances in Neural Information Processing Systems, 2017, pp. 5998–6008.

[21] L. Cao, X. Ji, Y. Cao, Y. Luo, Y. Wang, L. C. Siang, J. Li, and R. B. Gopaluni, “Interpretable industrial soft sensor design based on Informer and SHAP,” IFAC-PapersOnLine, vol. 58, no. 14, pp. 73–78, 2024.

[22] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, “LightGBM: A highly efficient gradient boosting decision tree,” in Proceedings of the Advances in Neural Information Processing Systems, 2017, pp. 3146–3154.

[23] P. Geurts, D. Ernst, and L. Wehenkel, “Extremely randomized trees,” Machine Learning, vol. 63, no. 1, pp. 3–42, 2006.

[24] L. Breiman, “Random forests,” Machine Learning, vol. 45, no. 1, pp. 5– 32, 2001.

[25] L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, and A. Gulin, “CatBoost: Unbiased boosting with categorical features,” in Proceedings of the Advances in Neural Information Processing Systems, 2018, pp. 6639–6649.

[26] M. J. Vowels, N. C. Camgoz, and R. Bowden, “D’ya like DAGs? A survey on structure learning and causal discovery,” ACM Computing Surveys, vol. 55, no. 4, pp. 1–36, 2022.

[27] S. Shimizu, T. Inazumi, Y. Sogawa, A. Hyvarinen, Y. Kawahara, T.¨ Washio, P. O. Hoyer, and K. Bollen, “DirectLiNGAM: A direct method for learning a linear non-Gaussian structural equation model,” Journal of Machine Learning Research, vol. 12, no. 10, pp. 1225–1248, 2011.

[28] L. Cao, F. Yu, F. Yang, Y. Cao, and R. B. Gopaluni, “Data-driven dynamic inferential sensors based on causality analysis,” Control Engineering Practice, vol. 104, p. 104626, 2020.

[29] F. Yu, Q. Xiong, L. Cao, and F. Yang, “Stable soft sensor modeling based on causality analysis,” Control Engineering Practice, vol. 122, p. 105109, 2022.

[30] L. Cao, J. Su, Y. Wang, Y. Cao, L. C. Siang, J. Li, J. Saddler, and B. Gopaluni, “Causal discovery based on observational data and process knowledge in industrial processes,” Industrial & Engineering Chemistry Research, vol. 61, no. 38, pp. 14272–14283, 2022.

[31] Y. Gal and Z. Ghahramani, “Dropout as a Bayesian approximation: Representing model uncertainty in deep learning,” in Proceedings of the International Conference on Machine Learning, 2016, pp. 1050–1059.