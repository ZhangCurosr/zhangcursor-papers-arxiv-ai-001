# CoMPASS: Collaborative Molecular Property Prediction via Adaptive Small–Large Model Synergy

Wentao Li, Jiangjie Qiu, Yijun Li, Leyi Zhao, Xiaonan Wang<sup>∗</sup>

Beijing Key Laboratory of Artificial Intelligence for Advanced Chemical Engineering Materials State Key Laboratory of Chemical Engineering and Low-Carbon Technology Department of Chemical Engineering, Tsinghua University Beijing 100084, China

## Abstract

Accurate molecular property prediction requires both statistical reliability and chemical reasoning. Graph neural networks can be calibrated directly on labeled assays but remain limited by the coverage of their training data. Large language models (LLMs) can compare molecular evidence and articulate chemical rationales, yet are unreliable as standalone quantitative predictors. The central challenge is therefore to determine when an LLM should influence a calibrated model and by how much. Here we present CoM-PASS, a retrieval-calibrated framework for small–large model collaboration. CoMPASS retains a graph attention network (GAT) as the predictive anchor, retrieves locally relevant training molecules, provides attention-grounded evidence to an LLM, and converts its proposal into a bounded correction through an agreement-aware gate. Across six classification and two regression benchmarks, CoMPASS improves the GAT anchor in regions of correctable uncertainty while limiting LLM intervention in high-confidence regimes. Ablations show that the gains arise from validation-calibrated retrieval and bounded fusion rather than prompting alone. These results suggest that generative reasoning should augment calibrated prediction through evidence-grounded, controlled corrections rather than direct output replacement. Code is available at https://github.com/littlepeachs/CoMPASS.

## Introduction

Molecular property prediction is useful only when its numerical outputs are reliable enough to guide downstream decisions. In drug discovery and toxicology, a model must assign a probability or continuous property value that can support screening, prioritization, and risk assessment (Cherkasov et al. 2014; Vamathevan et al. 2019; Wu et al. 2018; Stokes et al. 2020). This requirement creates a dual demand: the predictor must be statistically calibrated on labeled assays, but it should also reason over chemical evidence when local data are sparse or ambiguous.

Graph neural networks (GNNs), including graph attention networks (GATs), address the supervised prediction side of this problem by learning directly from molecular graphs and task-specific labels (Duvenaud et al. 2015; Kearnes et al. 2016; Gilmer et al. 2017; Kipf and Welling 2017; Veličković et al. 2018; Schütt et al. 2018; Klicpera, Groß, and Günnemann 2020; Xiong et al. 2020). Their reliability, however, is bounded by the coverage of the training distribution. Near decision boundaries, rare scafolds, sparse activity regions, or noisy multi-task assays, a compact graph model may return a confident prediction even when the local evidence is weak or conflicting.

LLMs provide a complementary form ofevidence processing. Recent scientific and chemistry-oriented LLMs can follow structured instructions, compare molecular descriptions, and articulate rationales (Taylor et al. 2022; Bran et al. 2024; Christofidellis et al. 2023; Grattafiori et al. 2024; Gemma Team 2024; Qwen Team 2025; Zhao et al. 2025; Zhang et al. 2024). Yet a fluent chemical rationale is not the same as a calibrated quantitative prediction. If an LLM is allowed to freely replace the output of a supervised predictor, retrieved molecules become anecdotes rather than statistical evidence, and the final score can drift away from the assay-calibrated model.

The central challenge is therefore to decide when an LLM should influence a calibrated molecular predictor and by how much. CoMPASS answers this question by treating largemodel reasoning as a controlled correction rather than direct output replacement. The GAT remains the predictive anchor. Retrieval supplies locally relevant training molecules and a validation-calibrated numerical estimate. The LLM receives a structured packet containing the target molecule, GAT prediction, attention evidence, retrieved analogues, and conflict summaries. Its proposal is then converted into a bounded prediction shift through a deterministic agreement-aware gate. Thus every LLM-mediated prediction can be traced to four explicit signals: the small model, the retrieved neighborhood, the LLM proposal, and the gate that mediates the correction.

Contributions. (1) We formulate small–large collaboration for molecular property prediction as evidence-grounded correction of a calibrated graph predictor. (2) We introduce retrieval-calibrated base prediction, where retrieved neighbors serve both as prompt evidence and validationfitted numerical calibration. (3) We define an agreementaware gate that limits LLM intervention in high-confidence regimes while allowing correction in uncertain regions. (4) We validate the mechanism across classification and regression benchmarks using ablations, confidence-bin analysis, backbone swaps, and success/failure cases.

![](images/ed8f19ecf6f000380e969ac066cdc38bd8a9568b5e92302c3fe0f785e648dadd.jpg)  
Figure 1: CoMPASS overview. A GAT produces the base prediction and attention evidence. Retrieval supplies top-K calibration molecules. The LLM proposes an evidence-conditioned correction, and a bounded gate controls how much that proposal changes the final prediction.

## Method

## Problem Setup and Small-Model Evidence

Given a molecule represented by a SMILES string s and graph G, the goal is to predict a property y. Classification tasks are evaluated by ROC-AUC, and regression tasks by RMSE. CoMPASS decomposes prediction into four stages. First, a GAT produces a supervised prediction, confidence, attention evidence, and graph representation:

$$
\begin{array} { r } { ( \hat { y } _ { s } , c _ { s } , a _ { s } , h _ { s } ) = f _ { s } ( G ) . } \end{array}\tag{1}
$$

For classification, confidence is prediction sharpness,

$$
\begin{array} { r } { c _ { s } ^ { \mathrm { c l s } } = \operatorname* { m a x } ( \hat { y } _ { s } , 1 - \hat { y } _ { s } ) . } \end{array}\tag{2}
$$

For regression, confidence is derived from Monte Carlo dropout uncertainty (Gal and Ghahramani 2016),

$$
c _ { s } ^ { \mathrm { r e g } } = \frac { 1 } { 1 + u _ { s } } ,\tag{3}
$$

where $u _ { s }$ is predictive standard deviation on the standardized target scale. Attention is not used only for visualization: it becomes a communication channel that tells the LLM which atoms or functional groups the graph model considered important.

## Retrieval-Calibrated Base Prediction

For each target molecule, the retriever returns K calibration molecules from the training split only:

$$
\begin{array} { r } { \mathcal { R } _ { K } ( s ) = \mathrm { T o p K } _ { i \in \mathcal { D } _ { \mathrm { t r a i n } } } \mathrm { s i m } ( s , s _ { i } ) . } \end{array}\tag{4}
$$

Similarity combines Morgan fingerprint overlap, graphembedding cosine similarity, and attention overlap:

$$
\sin ( i , j ) = \omega _ { m } m _ { i j } + \omega _ { q } q _ { i j } + \omega _ { a } r _ { i j } .\tag{5}
$$

The query molecule is excluded by SMILES match, and validation/test molecules are never inserted into the retrieval

database. Retrieved labels form a similarity-weighted local estimate:

$$
\tilde { w } _ { i } = \mathrm { m a x } ( \mathrm { s i m } ( s , s _ { i } ) , 0 ) + \epsilon ,\tag{6}
$$

$$
w _ { i } = \tilde { w } _ { i } \Big / \sum _ { j \in \mathcal { R } _ { K } } \tilde { w } _ { j } ,\tag{7}
$$

$$
\hat { y } _ { r } = \sum _ { i \in \mathcal { R } _ { K } ( s ) } w _ { i } y _ { i } .\tag{8}
$$

For classification, ${ \hat { y } } _ { r }$ is a probability-like label aggregate; for regression, it is a local property estimate on the standardized target scale.

CoMPASS forms a retrieval-calibrated base prediction,

$$
\hat { y } _ { b } = ( 1 - \lambda ) \hat { y } _ { s } + \lambda \hat { y } _ { r } ,\tag{9}
$$

where λ is selected on validation data for each dataset and retrieval size:

$$
\lambda ^ { \star } = \arg \operatorname* { m i n } _ { \lambda \in \Lambda } \mathcal { L } _ { \mathrm { v a l } } \big ( ( 1 - \lambda ) \hat { y } _ { s } + \lambda \hat { y } _ { r } , y \big ) .\tag{10}
$$

For classification, ${ \mathcal L } _ { \mathrm { v a l } }$ is negative ROC-AUC; for regression, it is RMSE. This makes retrieval more than prompt context: local examples also provide a validation-calibrated numerical correction.

## Structured LLM Proposal

The prompt presents the property, target SMILES, GAT prediction and confidence, attention-highlighted functional groups, retrieved molecules with labels and GAT predictions, retrieval conflict, attention consistency, and required output format. The LLM returns

$$
\begin{array} { r } { ( \hat { y } _ { \ell } , c _ { \ell } , d _ { \ell } , g _ { \ell } , r _ { \ell } ) = \mathrm { L L M } ( P _ { s } ) , } \end{array}\tag{11}
$$

where $\hat { y } _ { \ell }$ is the numeric proposal, $d _ { \ell }$ is the correction direction, $g _ { \ell }$ is an optional suggested gate, and $r _ { \ell }$ is the rationale.

The parser uses the last explicit “Final Prediction” field to avoid echoed template text. If parsing fails, CoMPASS falls back to the small-model prediction. Classification outputs are clipped to [0, 1], and regression outputs are clipped to the observed standardized range.

## Agreement-Aware Bounded Gate

The final prediction interpolates between the calibrated base and the LLM proposal:

$$
\hat { y } = ( 1 - g ) \hat { y } _ { b } + g \hat { y } _ { \ell } .\tag{12}
$$

The gate $g$ is deterministic and bounded. It uses normalized uncertainty, retrieval conflict, attention consistency, and directional agreement. Retrieval conflict is

$$
\rho = \mathrm { s t d } _ { j \in \mathcal { R } _ { K } ( s ) } ( y _ { j } ) , \qquad \bar { \rho } = \mathrm { c l i p } ( 2 \tilde { \rho } , 0 , 1 ) ,\tag{13}
$$

where $\tilde { \rho } = \rho$ for classification and $\tilde { \rho } = \rho / \sigma _ { y }$ for regression. Attention consistency is the mean attention-overlap similarity:

$$
A _ { s } = \frac 1 K \sum _ { j \in \mathcal { R } _ { K } } \mathrm { s i m } _ { \mathrm { a t t } } ( s , s _ { j } ) .\tag{14}
$$

Let $\Delta _ { r } = \hat { y } _ { r } - \hat { y } _ { s }$ and $\Delta _ { \ell } = \hat { y } _ { \ell } - \hat { y } _ { b }$ . Directional agreement is

$$
\delta _ { \ell r } = \left\{ \begin{array} { l l } { \mathrm { a g r e e , } } & { \Delta _ { r } \Delta _ { \ell } > 0 , } \\ { \mathrm { w e a k , } } & { \left| \Delta _ { r } \right| < \xi \mathrm { o r } \left| \Delta _ { \ell } \right| < \xi , } \\ { \mathrm { d i s a g r e e , } } & { \mathrm { o t h e r w i s e . } } \end{array} \right.\tag{15}
$$

The implemented gate is

$$
g = \mathrm { c l i p } ( s _ { \delta _ { \ell r } } \Phi _ { \Theta } ( g _ { \ell } , { \bf e } _ { s } ) , f _ { \delta _ { \ell r } } , c _ { \delta _ { \ell r } } ) ,\tag{16}
$$

where

$$
\mathbf { e } _ { s } = [ 1 , 1 - c _ { s } , \bar { u } _ { s } , \bar { \rho } , 1 - A _ { s } ] ^ { \top } .\tag{17}
$$

The normalized uncertainty term is $\bar { u } _ { s }$ = $\mathrm { c l i p } ( u _ { s } / \log 2 , 0 , 1 )$ for classification entropy and $\bar { u } _ { s } ~ = ~ \mathrm { c l i p } { ( u _ { s } , 0 , 1 ) }$ for standardized regression dropout uncertainty. Here $\Phi _ { \Theta }$ is a fixed linear evidence map blended with the optional LLM-recommended gate. The floor $f _ { \delta } ,$ cap $c _ { \delta }$ , and scale $s _ { \delta }$ depend on agreement: agreeing retrieval and LLM shifts receive the largest cap, weak agreement receives a smaller cap, and disagreement receives the most conservative cap. Thus the LLM always participates through a capped correction rather than direct label replacement.

## Experimental Setup

## Datasets, Metrics, and Splits

We evaluate eight benchmarks spanning target activity, pharmacokinetics, toxicity, and physicochemical regression. BACE, BBBP, ClinTox, HIV, Tox21, ESOL, and Lipo are standard MoleculeNet-style benchmarks (Wu et al. 2018); CYP450 is a five-task cytochrome P450 inhibition benchmark commonly used for ADME modeling (Huang et al. 2021). Six tasks use ROC-AUC; ESOL and Lipo use original-scale RMSE. All runs use fixed DeepChem train/validation/test splits. For multi-task datasets, predictions, labels, and weights are stored as task vectors; retrieval produces a label vector, validation fitting respects task weights, and final metrics average valid task entries.

CoMPASS inference for one test molecule   
1. Run the GAT: obtain $\hat { y } _ { s } , c _ { s } , a _ { s } , h _ { s } .$   
2. Retrieve $\mathcal { R } _ { K } ( s )$ from the training split and compute ${ \hat { y } } _ { r }$   
3. Form $\hat { y } _ { b } = ( \hat { 1 } - \lambda ^ { \star } ) \hat { y } _ { s } + \lambda ^ { \star } \hat { y } _ { r }$   
4. Prompt the LLM with target, GAT evidence, and re  
trieved cases.   
5. Parse ${ \hat { y } } _ { \ell } , g _ { \ell } , r _ { \ell } ;$ fall back to $\hat { y } _ { s }$ if parsing fails.   
6. Compute agreement $\delta _ { \ell r }$ and bounded gate $g .$   
7. Return $\hat { y } = ( 1 - g ) \hat { y } _ { b } + g \hat { y } _ { \ell }$ and the full evidence trace.

Table 1: Inference summary. CoMPASS makes every LLMmediated prediction auditable through the GAT output, retrieved cases, LLM proposal, and gate.

## Configurations and Selection Protocol

The main configuration uses Llama-3-8B and evaluates $K \in \{ 4 , 6 , 8 \}$ . Retrieval weight $\lambda ^ { \star }$ and the reported retrieval size are selected on validation data. Test labels are used only for final evaluation. The main results average three independent seeds, with GAT training for up to 30 epochs with early stopping, inference batch size 32, learning rate $1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 5 }$ , Morgan radius 2 with 2048 bits, and LLM decoding temperature 0.7. Additional experiments compare Qwen2.5-7B, ChemDFM, and ChemLLM, and replace the GAT anchor with a GraphGPS-style graph transformer (Rampášek et al. 2022) to test whether the collaboration interface is tied to one graph architecture.

## Results

## Main Results: Recoverable Anchor Errors

Table 3 and Table 4 show that CoMPASS consistently improves the paired GAT anchor, but it does not claim universal dominance over all graph baselines. Its value is clearest where the anchor leaves retrievable local error. BACE, BBBP, Clin-Tox, and HIV show large improvements because retrieved analogues expose boundary or calibration mistakes. CYP450 and Tox21 have smaller gains because the GAT anchor is already strong and the gate mainly limits harm. Regression follows the same pattern: CoMPASS improves over our GAT runs on ESOL and Lipo, while the claim remains calibrated correction rather than state-of-the-art regression.

Bootstrap intervals in Table 5 are positive on BACE, BBBP, ClinTox, HIV, ESOL, and Lipo. Tox21 is positive but small, and CYP450 is best read as a low-headroom setting where bounded gating preserves the anchor. This supports the central claim: CoMPASS helps when local evidence reveals correctable GAT uncertainty, and its conservative design matters when headroom is limited.

## Retrieval Size Is a Calibration Axis

Figure 2 shows that retrieval size is not a cosmetic hyperparameter. BACE, ClinTox, and Tox21 peak at $K = 6 .$ , while BBBP, HIV, ESOL, and Lipo peak at $K = 8$ . More neighbors help when analogues form a coherent local neighborhood, but can dilute signal when the property varies sharply.

<table><tr><td>Dataset</td><td>Type</td><td>Tasks</td><td>Train/Valid/Test</td><td>Label statistic</td><td>Metric</td></tr><tr><td>BACE</td><td>Class.</td><td>1</td><td>1210/151/152</td><td>0.426 positive</td><td>ROC-AUC</td></tr><tr><td>BBBP</td><td>Class.</td><td>1</td><td>1631/204/204</td><td>0.822 positive</td><td>ROC-AUC</td></tr><tr><td>ClinTox</td><td>Class.</td><td>2</td><td>1184/148/148</td><td>0.507 positive</td><td>ROC-AUC</td></tr><tr><td>CYP450</td><td>Class.</td><td>5</td><td>13040/1690/1690</td><td>0.305 positive</td><td>ROC-AUC</td></tr><tr><td>HIV</td><td>Class.</td><td>1</td><td>32896/4112/4112</td><td>0.037 positive</td><td>ROC-AUC</td></tr><tr><td>Tox21</td><td>Class.</td><td>12</td><td>6258/782/783</td><td>0.071 positive</td><td>ROC-AUC</td></tr><tr><td>ESOL</td><td>Reg.</td><td>1</td><td>902/113/113</td><td>-2.867 train mean</td><td>RMSE</td></tr><tr><td>Lipo</td><td>Reg.</td><td>1</td><td>3360/420/420</td><td>2.163 train mean</td><td>RMSE</td></tr></table>

Table 2: Dataset statistics. Counts are fixed DeepChem splits. Classification statistics are positive rates. Regression statistics are train-label means on the original property scale.
<table><tr><td>Model</td><td>Method</td><td>BACE (152)</td><td>BBBP (204)</td><td>ClinTox (148)</td><td>CYP450 (1690)</td><td>HIV (4112)</td><td>Tox21 (783)</td></tr><tr><td rowspan="4"></td><td>Gimlet</td><td>0.696</td><td>0.594</td><td></td><td>0.713</td><td>0.662</td><td>0.612</td></tr><tr><td>KVPLM</td><td>0.513</td><td>0.602</td><td></td><td>0.592</td><td>0.612</td><td>0.492</td></tr><tr><td>MoMu</td><td>0.666</td><td>0.498</td><td></td><td>0.580</td><td>0.503</td><td>0.576</td></tr><tr><td>Galactica-125M</td><td>0.445</td><td>0.605</td><td></td><td>0.537</td><td>0.367</td><td>0.496</td></tr><tr><td rowspan="2">MolRAG</td><td>Galactica-1.3B</td><td>0.565</td><td>0.539</td><td></td><td>0.469</td><td>0.339</td><td>0.495</td></tr><tr><td>Struct-CoT</td><td>0.626</td><td>0.572</td><td>一</td><td>0.584</td><td>0.595</td><td>0.566</td></tr><tr><td rowspan="3">Graph-based Networks</td><td>Sim-CoT</td><td>0.723</td><td>0.541</td><td></td><td>0.723</td><td>0.644</td><td>0.639</td></tr><tr><td>GAT</td><td>0.756</td><td>0.652</td><td>0.893</td><td>0.877</td><td>0.715</td><td>0.735</td></tr><tr><td>GIN</td><td>0.701</td><td>0.658</td><td>一</td><td>0.821</td><td>0.753</td><td>0.740</td></tr><tr><td rowspan="2">Our runs</td><td>Graphormer</td><td>0.776</td><td>0.702</td><td>一</td><td>0.844</td><td>0.745</td><td>0.759</td></tr><tr><td>CoMPASS Gain vs GAT</td><td>0.852 +0.096</td><td>0.732 +0.080</td><td>0.937 +0.044</td><td>0.880 +0.003</td><td>0.755 +0.040</td><td>0.744 +0.009</td></tr></table>

Table 3: Experimental results on classification test datasets. Values are ROC-AUC, where higher is better. External GIMLET, KVPLM, MoMu, Galactica, GIN, and Graphormer values are reproduced from Zhao et al. (2023); four-shot MolRAG Struct-CoT and Sim-CoT values are reproduced from Xian et al. (2025). Only GAT and CoMPASS are paired under the fixed split and evaluator. The best value in each dataset column is bolded and the second-best value is underlined.

CoMPASS therefore treats K as a validation-calibrated design axis: it determines whether the LLM sees a focused local counterfactual set or a noisier neighborhood summary.

## Ablations: Retrieval Carries Signal, Gating Controls It

Table 6 separates three efects. First, LLM-only correction without calibrated retrieval is weak, so the LLM should not replace the GAT. Second, retrieval carries meaningful signal, but fixed or partial retrieval variants are weaker than the full interface. Third, attention evidence helps ground the LLM when retrieved molecules contain mixed evidence. The useful signal is therefore not the raw LLM estimate alone; it is the agreement-filtered correction after retrieval calibration.

This ablation also clarifies the role of the gate. Retrievalonly variants can already improve regression and some classification tasks, which means local label evidence is genuinely informative. However, prompt evidence and LLM rationales become useful only when the final score is capped by agreement and uncertainty. CoMPASS should therefore be viewed as a calibrated decision rule for incorporating generative reasoning, not as a prompt engineering method.

## Backbone and Anchor Robustness

Table 7 shows that the collaboration interface is not tied to one LLM. Llama-3-8B is strongest on BACE and BBBP, while chemistry-oriented LLMs are competitive on regression. This suggests a practical deployment path: the same GAT and retriever can route dificult examples to diferent LLMs according to endpoint type or local availability. Table 8 shows that the interface can also plug into an attentionbearing graph transformer. The efect is largest on BACE and smaller on BBBP/ESOL, consistent with the idea that collaboration helps most when the anchor leaves retrievable headroom.

## Where Corrections Help

The confidence-binned analysis in Figure 3 connects the metric gains to the gate design. Low-confidence GAT predictions have more headroom and benefit most from retrieved evidence and LLM interpretation. High-confidence predictions have less headroom and greater over-correction risk. This pattern explains why an unrestricted LLM mixture is not desirable: the large model should be heard most clearly when the anchor is uncertain, and should be capped when the anchor is confident or retrieval evidence is inconsistent.

<table><tr><td colspan="7">Classification</td><td colspan="3">Regression</td></tr><tr><td>K=4</td><td>0.048</td><td>0.042</td><td>-0.001</td><td>-0.000</td><td>0.021</td><td>0.006</td><td>0.042</td><td>0.014</td><td rowspan="3">Impr ovr oAT -0.08 -0.06 -0.04 -0.02 -0.00</td></tr><tr><td>K=6</td><td>0.096 best</td><td>0.054</td><td>0.044 best</td><td>0.001</td><td>0.028</td><td>0.009 best</td><td>0.048</td><td>0.014</td></tr><tr><td>K=8</td><td>0.094</td><td>0.080 best</td><td>0.019</td><td>0.003 best</td><td>0.040 best</td><td>0.008</td><td>0.074 best</td><td>0.027 best</td></tr><tr><td></td><td>BACE</td><td>BBBP</td><td>ClinTox</td><td>CYP450 Dataset</td><td>HIV</td><td>Tox21</td><td>ESOL</td><td>Lipo</td></tr></table>

Figure 2: Main-result heatmap over retrieval size. Cells show improvement over the GAT baseline in each dataset’s metric units: ROC-AUC points for classification and original-scale RMSE reduction for regression.

<table><tr><td>Model</td><td>Method</td><td>ESOL RMSE↓</td><td>Lipo RMSE↓</td></tr><tr><td rowspan="3">Llama3-8B</td><td>1-shot Sim-CoT</td><td>4.142</td><td>1.271</td></tr><tr><td>2-shot Sim-CoT</td><td>3.499</td><td>1.168</td></tr><tr><td>4-shot Sim-CoT</td><td>3.281</td><td>1.125</td></tr><tr><td>Pre-train. Methods</td><td>Gimlet</td><td>1.132</td><td>1.345</td></tr><tr><td rowspan="2">Graph Networks</td><td>GAT</td><td></td><td></td></tr><tr><td>GIN</td><td>1.275 1.243</td><td>0.816 0.781</td></tr><tr><td rowspan="2">Our runs</td><td>CoMPASS</td><td>1.201</td><td>0.790</td></tr><tr><td>RMSE reduction vs GAT</td><td>+0.074</td><td>+0.026</td></tr></table>

Table 4: Regression test results. Values are original-scale RMSE, so lower is better. Llama3-8B Sim-CoT values at 1, 2, and 4 shots are reproduced from Xian et al. (2025), and external Gimlet and GIN values are reproduced from Zhao et al. (2023); only GAT and CoMPASS are paired under the fixed split and evaluator. The best value in each dataset column is bolded and the second-best value is underlined.

## Case and Failure Analysis

Figure 4 illustrates the two correction regimes. In classification, successful cases often involve boundary movement: the GAT score lies near the 0.5 threshold, retrieved analogues provide a directional signal, and the gate permits enough movement to change the decision. In regression, there is no discrete threshold; the correction usually moves partway toward a local physicochemical property scale. The point is not that the LLM invents a label, but that it interprets structured evidence and proposes a direction that survives bounded fusion.

The lower half of Figure 4 reveals the main limitation. Retrieval can be misleading: a correct GAT prediction can be pulled across the threshold, or a regression estimate can move farther from the label. The nonzero gate floor is intentional because a zero floor would make the large model decorative, but the cap is equally important because it prevents uncontrolled override. Future gains are therefore more likely to come from better conflict detection and more adaptive caps than from simply increasing the LLM weight. Showing both successful and failed corrections is central to the CoMPASS claim: the method is not a promise that LLMs always help, but a controlled interface that makes their contribution and risk auditable.

<table><tr><td>Dataset</td><td>Gain</td><td>Bootstrap interval</td></tr><tr><td>BACE</td><td>+0.096</td><td>[0.038, 0.155]</td></tr><tr><td>BBBP</td><td>+0.080</td><td>[0.041, 0.121]</td></tr><tr><td>ClinTox</td><td>+0.044</td><td>[0.018, 0.081]</td></tr><tr><td>CYP450</td><td>+0.003</td><td>[-0.000, 0.006]</td></tr><tr><td>HIV</td><td>+0.040</td><td>[0.022, 0.060]</td></tr><tr><td>Tox21</td><td>+0.009</td><td>[0.001, 0.016]</td></tr><tr><td>ESOL</td><td>+0.074</td><td>[0.018, 0.134]</td></tr><tr><td>Lipo</td><td>+0.026</td><td>[0.013, 0.040]</td></tr></table>

Table 5: Molecule-bootstrap gain intervals. Positive gain means ROC-AUC increase for classification and originalscale RMSE reduction for regression.

## Operational Interpretation

Multi-task endpoints. ClinTox, CYP450, and Tox21 contain multiple assay tasks with missing-label masks. CoM-PASS keeps the GAT prediction as a task vector and computes retrieval-calibrated label aggregates with the same evaluator masks used by the benchmark. The LLM emits a scalar correction because the prompt asks it to reason over the molecule-level evidence packet rather than produce one free-form answer per assay. This scalar shift is applied to the retrieval-calibrated task vector and clipped to the valid range. The gate is therefore molecule-specific rather than task-specific, while metric computation remains task-aware through the original masks. This design avoids asking the LLM to hallucinate labels for missing tasks and keeps the supervised graph model responsible for task-specific calibration.

<table><tr><td>Variant</td><td>BACE</td><td>BBBP</td><td>ESOL</td><td>Lipo</td></tr><tr><td>GAT only</td><td>0.756</td><td>0.652</td><td>1.275</td><td>0.816</td></tr><tr><td>LLM + ĠAT</td><td>0.759</td><td>0.655</td><td>1.247</td><td>0.818</td></tr><tr><td>GAT + Attention Retrieval</td><td>0.781</td><td>0.663</td><td>1.244</td><td>0.802</td></tr><tr><td>GAT + Similarity Retrieval</td><td>0.796</td><td>0.662</td><td>1.226</td><td>0.809</td></tr><tr><td>LLM + GAT + Retrieval (fixed weight)</td><td>0.819</td><td>0.687</td><td>1.216</td><td>0.804</td></tr><tr><td>LLM + GAT + Retrieval (no attention)</td><td>0.781</td><td>0.661</td><td>1.230</td><td>0.812</td></tr><tr><td>Full CoMPASS</td><td>0.852</td><td>0.732</td><td>1.201</td><td>0.790</td></tr></table>

Table 6: Complete ablation study. BACE and BBBP report ROC-AUC, where higher is better; ESOL and Lipo report original scale RMSE, where lower is better. The best value in each dataset column is bolded and the second-best value is underlined.

<table><tr><td>Dataset</td><td>GAT L3</td><td>Qwen CDFM CLLM</td></tr><tr><td>BACE 0.756</td><td>0.852</td><td>0.849 0.843 0.848</td></tr><tr><td>BBBP</td><td>0.652 0.732</td><td>0.717 0.701 0.707</td></tr><tr><td>ESOL</td><td>1.275 1.201</td><td>1.231 1.183 1.210</td></tr><tr><td>Lipo</td><td>0.816 0.790</td><td>0.798 0.800 0.789</td></tr></table>

Table 7: LLM backbone comparison. BACE/BBBP use ROC-AUC; ESOL/Lipo use RMSE. L3 denotes Llama-3-8B, CDFM denotes ChemDFM, and CLLM denotes ChemLLM.

<table><tr><td>Data</td><td>Metric</td><td>GPS</td><td>CoMPASS-GPS</td><td>Gain</td></tr><tr><td>BACE</td><td>ROC</td><td>0.770</td><td>0.868</td><td>+0.098</td></tr><tr><td>BBBP</td><td>ROC</td><td>0.718</td><td>0.730</td><td>+0.012</td></tr><tr><td>ESOL</td><td>RMSE</td><td>1.107</td><td>1.103</td><td>+0.005</td></tr></table>

Table 8: GraphGPS anchor swap. Only the graph anchor is changed; retrieval, prompting, and gating remain unchanged.

Audit trail. Each CoMPASS prediction stores the target SMILES, GAT score, confidence, attention summary, retrieved molecules, retrieved labels, retrieval weights, LLM proposal, gate value, final prediction, and rationale. This trace is notjust a logging convenience. It allows a reviewer or practitioner to distinguish three cases that have the same final metric efect but diferent operational meaning: a retrievaldriven correction, an LLM proposal suppressed by disagreement, and an LLM proposal accepted because it agrees with local evidence. The trace also enables the failure gallery in Figure 4; without it, a molecular LLM system can look persuasive while hiding whether the large model actually improved the calibrated prediction.

Why not prompt-only RAG? Prompt-only molecular RAG treats retrieved examples mainly as context for a generator. CoMPASS instead gives retrieval a numerical role before the LLM is consulted. This matters because a few retrieved analogues can be chemically plausible but statistically misleading. Validation-fitted blending estimates how much local labels should move the anchor on each dataset, while the gate determines how much of the LLM’s interpretation of that evidence should survive. The ablation table shows the consequence: prompting and retrieval are useful only when the final prediction remains bounded by calibrated evidence.

![](images/13ee3c29416c081d4ed345daaf58ab39153739e8676dbe6d3925c369b152022b.jpg)  
Figure 3: Confidence-binned behavior. Mean correction benefit is largest in lower-confidence GAT bins and shrinks as the GAT becomes more confident.

## Related Work

Molecular representation learning. Molecular property prediction has long used graph and fingerprint representations (Cherkasov et al. 2014; Rogers and Hahn 2010; Wu et al. 2018; Mayr et al. 2016). Modern GNNs learn taskspecific representations (Duvenaud et al. 2015; Kearnes et al. 2016; Gilmer et al. 2017; Kipf and Welling 2017; Veličković et al. 2018; Xu et al. 2019; Ying et al. 2021; Schütt et al. 2018; Klicpera, Groß, and Günnemann 2020), while molecular pretraining and contrastive learning improve transfer across endpoints (Hu et al. 2020; Rong et al. 2020; Liu et al. 2022; Wang et al. 2022; You et al. 2020).

LLMs and retrieval. LLMs have been explored for scientific reasoning and chemistry workflows (Taylor et al. 2022; Bran et al. 2024; Christofidellis et al. 2023; Zhao et al. 2025; Zhang et al. 2024). Retrieval-augmented generation grounds LLM outputs in external examples (Lewis et al. 2020; Guu et al. 2020; Karpukhin et al. 2020; Izacard and Grave 2021;

![](images/b387369a0ddabcd9fdecd5634aefc5d0217396dbfd651959e986c575923e8e9e.jpg)  
Figure 4: Molecule-level success and failure cases. Top: successful corrections show boundary movement or regression error reduction. Bottom: failure cases show over-correction from misleading retrieval or a nonzero gate floor.

Borgeaud et al. 2022). CoMPASS difers from LLM-driven molecular RAG by making the graph model the primary predictor and using retrieval as both prompt evidence and numeric calibration.

Small–large collaboration. Small–large model collaboration appears in speculative decoding, mixture-of-experts, and adaptation (Leviathan, Kalman, and Matias 2023; Fedus, Zoph, and Shazeer 2022; Xu et al. 2024). CoM-PASS contributes a molecular prediction instance of this paradigm where attention evidence, retrieval calibration, and uncertainty-aware gating mediate the collaboration. The broader lesson is that large models can improve prediction systems when their role is formalized as bounded, evidenceconditioned correction.

## Conclusion

CoMPASS argues for a restrained role for LLMs in molecular property prediction. The large model should not replace the supervised graph predictor; it should interpret retrieved analogues and graph-attention evidence when the small model may be locally miscalibrated. Across eight benchmarks, this interface improves the paired GAT anchor while exposing where headroom is small. Ablations sharpen the conclusion: retrieval and validation-fitted local correction carry much of the signal, and the LLM becomes useful when its proposal is agreement-filtered and bounded.

## Limitations

CoMPASS adds LLM inference, so it is most appropriate for ofline screening, triage, and mechanistic analysis rather than billion-scale virtual-library scoring. It also depends on retrieval quality: meaningful analogues expose local GAT errors, while misleading neighborhoods can induce overcorrection. Finally, results are reported on fixed benchmark splits rather than repeated resampling protocols, so the conclusions should be read as evidence for the joint retrievalplus-gating interface, not as a claim that prompting alone solves molecular prediction.

## References

Borgeaud, S.; Mensch, A.; Hofmann, J.; Cai, T.; Rutherford, E.; Millican, K.; van den Driessche, G.; Lespiau, J.-B.; Damoc, B.; Clark, A.; et al. 2022. Improving Language Mod-

els by Retrieving from Trillions of Tokens. In International Conference on Machine Learning, 2206–2240.

Bran, A. M.; Cox, S.; Schilter, O.; Baldassari, C.; White, A. D.; and Schwaller, P. 2024. ChemCrow: Augmenting large-language models with chemistry tools. Nature Machine Intelligence, 6: 525–535.

Cherkasov, A.; Muratov, E. N.; Fourches, D.; Varnek, A.; Baskin, I. I.; Cronin, M.; Dearden, J.; Gramatica, P.; Martin, Y. C.; Todeschini, R.; et al. 2014. QSAR modeling: where have you been? Where are you going to? Journal ofMedicinal Chemistry, 57(12): 4977–5010.

Christofidellis, D.; Giannone, G.; Born, J.; Winther, O.; Laino, T.; and Manica, M. 2023. Unifying molecular and textual representations via multi-task language modelling. In Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, 6140–6157.

Duvenaud, D. K.; Maclaurin, D.; Aguilera-Iparraguirre, J.; Gómez-Bombarelli, R.; Hirzel, T.; Aspuru-Guzik, A.; and Adams, R. P. 2015. Convolutional Networks on Graphs for Learning Molecular Fingerprints. In Advances in Neural Information Processing Systems, volume 28.

Fedus, W.; Zoph, B.; and Shazeer, N. 2022. Switch transformers: Scaling to trillion parameter models with simple and eficient sparsity. Journal of Machine Learning Research, 23(120): 1–39.

Gal, Y.; and Ghahramani, Z. 2016. Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. In Proceedings of the 33rd International Conference on Machine Learning, volume 48 of Proceedings of Machine Learning Research, 1050–1059.

Gemma Team. 2024. Gemma 2: Improving Open Language Models at a Practical Size. arXivpreprint arXiv:2408.00118.

Gilmer, J.; Schoenholz, S. S.; Riley, P. F.; Vinyals, O.; and Dahl, G. E. 2017. Neural message passing for quantum chemistry. In International Conference on Machine Learning, 1263–1272.

Grattafiori, A.; Dubey, A.; Jauhri, A.; Pandey, A.; Kadian, A.; Al-Dahle, A.; Letman, A.; Mathur, A.; Schelten, A.; Vaughan, A.; et al. 2024. The Llama 3 Herd of Models. arXiv preprint arXiv:2407.21783.

Guu, K.; Lee, K.; Tung, Z.; Pasupat, P.; and Chang, M.-W. 2020. REALM: Retrieval-Augmented Language Model Pre-Training. In International Conference on Machine Learning, 3929–3938.

Hu, W.; Liu, B.; Gomes, J.; Zitnik, M.; Liang, P.; Pande, V.; and Leskovec, J. 2020. Strategies for pre-training graph neural networks. arXiv preprint arXiv:1905.12265.

Huang, K.; Fu, T.; Gao, W.; Zhao, Y.; Roohani, Y.; Leskovec, J.; Coley, C. W.; Xiao, C.; Sun, J.; and Zitnik, M. 2021. Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development. In Advances in Neural Information Processing Systems, volume 34, 937– 949.

Izacard, G.; and Grave, E. 2021. Leveraging Passage Retrieval with Generative Models for Open Domain Question

Answering. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics, 874–880.

Karpukhin, V.; Oguz, B.; Min, S.; Lewis, P.; Wu, L.; Edunov, S.; Chen, D.; and Yih, W.-t. 2020. Dense Passage Retrieval for Open-Domain Question Answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, 6769–6781.

Kearnes, S.; McCloskey, K.; Berndl, M.; Pande, V.; and Riley, P. 2016. Molecular Graph Convolutions: Moving Beyond Fingerprints. Journal of Computer-Aided Molecular Design, 30(8): 595–608.

Kipf, T. N.; and Welling, M. 2017. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations.

Klicpera, J.; Groß, J.; and Günnemann, S. 2020. Directional Message Passing for Molecular Graphs. In International Conference on Learning Representations.

Leviathan, Y.; Kalman, M.; and Matias, Y. 2023. Fast inference from transformers via speculative decoding. arXiv preprint arXiv:2211.17192.

Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V.; Goyal, N.; Küttler, H.; Lewis, M.; Yih, W.-t.; Rocktäschel, T.; et al. 2020. Retrieval-augmented generation for knowledgeintensive NLP tasks. In Advances in Neural Information Processing Systems, volume 33, 9459–9474.

Liu, S.; Wang, H.; Liu, W.; Lasenby, J.; Guo, H.; and Tang, J. 2022. Pre-training Molecular Graph Representation with 3D Geometry. In International Conference on Learning Representations.

Mayr, A.; Klambauer, G.; Unterthiner, T.; and Hochreiter, S. 2016. DeepTox: Toxicity Prediction using Deep Learning. Frontiers in Environmental Science, 3: 80.

Qwen Team. 2025. Qwen2.5 Technical Report. arXiv preprint arXiv:2412.15115.

Rampášek, L.; Galkin, M.; Dwivedi, V. P.; Luu, A. T.; Wolf, G.; and Beaini, D. 2022. Recipe for a General, Powerful, Scalable Graph Transformer. In Advances in Neural Information Processing Systems, volume 35, 14501–14515.

Rogers, D.; and Hahn, M. 2010. Extended-connectivity fingerprints. Journal of Chemical Information and Modeling, 50(5): 742–754.

Rong, Y.; Bian, Y.; Xu, T.; Xie, W.; Wei, Y.; Huang, W.; and Huang, J. 2020. Self-Supervised Graph Transformer on Large-Scale Molecular Data. In Advances in Neural Information Processing Systems, volume 33, 12559–12571.

Schütt, K. T.; Sauceda, H. E.; Kindermans, P.-J.; Tkatchenko, A.; and Müller, K.-R. 2018. SchNet: A Deep Learning Architecture for Molecules and Materials. Journal ofChemical Physics, 148(24): 241722.

Stokes, J. M.; Yang, K.; Swanson, K.; Jin, W.; Cubillos-Ruiz, A.; Donghia, N. M.; MacNair, C. R.; French, S.; Carfrae, L. A.; Bloom-Ackermann, Z.; et al. 2020. A Deep Learning Approach to Antibiotic Discovery. Cell, 180(4): 688–702.e13.

Taylor, R.; Kardas, M.; Cucurull, G.; Scialom, T.; Hartshorn, A.; Saravia, E.; Poulton, A.; Kerkez, V.; and Stojnic, R. 2022. Galactica: A large language model for science. arXiv preprint arXiv:2211.09085.

Vamathevan, J.; Clark, D.; Czodrowski, P.; Dunham, I.; Ferran, E.; Lee, G.; Li, B.; Madabhushi, A.; Shah, P.; Spitzer, M.; et al. 2019. Applications of machine learning in drug discovery and development. Nature Reviews Drug Discovery, 18(6): 463–477.

Veličković, P.; Cucurull, G.; Casanova, A.; Romero, A.; Lio, P.; and Bengio, Y. 2018. Graph attention networks. In International Conference on Learning Representations.

Wang, Y.; Wang, J.; Cao, Z.; and Barati Farimani, A. 2022. Molecular contrastive learning of representations via graph neural networks. Nature Machine Intelligence, 4(3): 279– 287.

Wu, Z.; Ramsundar, B.; Feinberg, E. N.; Gomes, J.; Geniesse, C.; Pappu, A. S.; Leswing, K.; and Pande, V. 2018. MoleculeNet: a benchmark for molecular machine learning. Chemical Science, 9(2): 513–530.

Xian, Z.; Gu, J.; Li, L.; and Liang, S. 2025. MolRAG: Unlocking the Power of Large Language Models for Molecular Property Prediction. In Proceedings ofthe 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 15513–15531.

Xiong, Z.; Wang, D.; Liu, X.; Zhong, F.; Wan, X.; Li, X.; Li, Z.; Luo, X.; Chen, K.; Jiang, H.; et al. 2020. Pushing the boundaries of molecular representation for drug discovery with the graph attention mechanism. Journal of Medicinal Chemistry, 63(16): 8749–8760.

Xu, C.; Xu, Y.; Wang, S.; Liu, Y.; Zhu, C.; and McAuley, J. 2024. Small models are valuable plug-ins for large language models. In Findings of the Association for Computational Linguistics: ACL 2024, 283–294.

Xu, K.; Hu, W.; Leskovec, J.; and Jegelka, S. 2019. How Powerful are Graph Neural Networks? In International Conference on Learning Representations.

Ying, C.; Cai, T.; Luo, S.; Zheng, S.; Ke, G.; He, D.; Shen, Y.; and Liu, T.-Y. 2021. Do transformers really perform badly for graph representation? In Advances in Neural Information Processing Systems, volume 34, 28877–28888.

You, Y.; Chen, T.; Sui, Y.; Chen, T.; Wang, Z.; and Shen, Y. 2020. Graph contrastive learning with augmentations. In Advances in Neural Information Processing Systems, volume 33, 5812–5823.

Zhang, D.; Liu, W.; Tan, Q.; Chen, J.; Yan, H.; Yan, Y.; Li, J.; Huang, W.; Yue, X.; Ouyang, W.; et al. 2024. Chem-LLM: A Chemical Large Language Model. arXiv preprint arXiv:2402.06852.

Zhao, H.; Liu, S.; Ma, C.; Xu, H.; Fu, J.; Deng, Z.; Kong, L.; and Liu, Q. 2023. GIMLET: A Unified Graph-Text Model for Instruction-Based Molecule Zero-Shot Learning. In Advances in Neural Information Processing Systems, volume 36, 5850–5887.

Zhao, Z.; Ma, D.; Chen, L.; Sun, L.; Li, Z.; Xia, Y.; Chen, B.;Xu, H.; Zhu, Z.; Zhu, S.; et al. 2025. Developing ChemDFM

as a Large Language Foundation Model for Chemistry. Cell Reports Physical Science, 6: 102523.