# EEG-AS: Instance-Level Foundation Model Selection for EEG Foundation Models via Behavior Reconstruction

Yunzhen Zhang<sup>1∗</sup>, Ruoxi Piao<sup>1∗</sup>, Hasan Onur Keles<sup>2†</sup>, Mustafa Misir<sup>1†</sup>

<sup>1</sup>Duke Kunshan University <sup>2</sup>Ankara University

## Abstract

Electroencephalography (EEG) is a non-invasive technique for measuring neural activity and has been widely used in neuroscience applications. Recent advances in EEG foundation models have enabled strong performance across diverse neural decoding tasks. However, no single foundation model consistently performs best across datasets or individual EEG instances, while instance-level model selection remains largely unexplored. To address this limitation, we formulate EEG foundation model selection as an instance-level Algorithm Selection (AS) problem. We propose EEG-AS, an instance-level algorithm selection framework that characterizes each EEG instance using inference-available latent EEG embeddings, handcrafted neurophysiological features, and an anchor foundation model. During training, EEG-AS learns to reconstruct unavailable foundation-model behaviors from privileged prediction tokens conditioned on an anchor foundation model, while during inference it estimates these behaviors without executing the entire model portfolio, enabling eficient selection from seven EEG foundation models. Experiments on seven public EEG benchmarks demonstrate that EEG-AS substantially narrows the gap between the Single Best Solver (SBS) and the oracle upper bound for each instance. These results highlight the efectiveness of instancelevel AS for adaptive deployment of EEG foundation models.

## Introduction

Electroencephalography (EEG) foundation models (FMs) have recently emerged as a promising paradigm for learning transferable neural representations from large-scale unlabeled EEG data (Kuruppu et al. 2026). General-purpose models have demonstrated strong performance across a wide range of downstream EEG applications(Xiong et al. 2026). As the number of available EEG FMs continues to grow, attention is shifting from developing stronger individual models to efectively utilizing existing ones.

Current practice typically deploys a single FM according to its average performance on a benchmark dataset. This Single Best Solver (SBS) strategy is implicitly based on the assumption that one model is uniformly preferable for all EEG instances within a dataset. However, our empirical analysis reveals that this assumption does not hold. Although a FM may achieve the highest average accuracy, its performance varies substantially across individual EEG instances. Different FMs correctly classify diferent instances, revealing substantial instance-level complementarity beyond their dataset-level competition. This observation is further supported by the large performance gap between the SBS and the Virtual Best Solver (VBS) / Oracle upper bound observed across all evaluated datasets, suggesting promising potential for adaptive model deployment. These observations echo the common Algorithm Selection (AS) premise that algorithm performance is instance-dependent (Kerschke et al. 2019). Since EEG inference is ultimately performed on individual trials, windows, or recordings, this heterogeneity motivates adapting model deployment at the same granularity.

These observations raise a fundamental research question: Can we automatically identify the most suitable FM for each EEG instance? We cast this problem as a per-instance AS task, where one FM is selected from a portfolio for each EEG instance (Kerschke et al. 2019). We propose EEG-AS. Our key insight is that the behaviors of unexecuted candidate FMs can be inferred from an inference-available EEG representation together with the prediction behavior of a single inference-available anchor FM. During training, EEG-AS learns the relationship among diferent FMs using privileged prediction tokens generated ofline. During inference, it reconstructs the unavailable model behaviors conditioned on the anchor model and selects the most suitable FM without exhaustively evaluating the entire portfolio. Experiments on seven public EEG benchmarks demonstrate that EEG-AS achieves the highest average performance among deployable methods, obtaining the best result on five datasets while remaining competitive on the others and outperforming SBS on all seven datasets.

The main contributions of this work are summarized as:

• We formulate EEG FM deployment as an instancelevel AS problem. To the best of our knowledge, this is the first work on adaptive deployment of EEG FMs at the instance level, moving beyond the conventional singlemodel deployment. Two traditional AS models are built.

• We propose EEG-AS, a novel instance-level selection framework that reconstructs unavailable FM behaviors from an EEG representation and a single inferenceavailable anchor model. This enables efective model selection without exhaustively executing the entire FM port-

folio during inference.

• We conduct comprehensive experiments on seven public EEG benchmarks covering diverse downstream applications. The experimental results demonstrate that EEG-AS improves average selection accuracy from 53.7% for SBS to 66.9%, closes 44.0% of the SBS-to-Oracle performance gap and the 83.7% Oracle upper bound. It also provides extensive analyses of instance-level complementarity among EEG FMs, highlighting the potential of AS for adaptive EEG FM deployment.

## Background

## EEG Foundation Models

Recent EEG foundation models (FMs) leverage selfsupervised pretraining on large-scale unlabeled EEG data to learn transferable neural representations for downstream EEG applications. Existing EEG FMs difer substantially in architectural design and pretraining strategy, reflecting diverse approaches to modeling the spatial, temporal, and semantic characteristics of EEG signals. Comprehensive benchmarks such as EEG-FM-Bench (Xiong et al. 2025, 2026) have standardized the evaluation of these models across diverse EEG datasets, providing a unified benchmark for comparing their downstream performance.

## Automated Algorithm Selection (AS)

AS has been applied to various problem domains such as Boolean satisfiability, combinatorial optimization and machine learning. Within the scope of Automated Machine Learning (AutoML), AS focuses on a critical meta-learning task. Traditionally, AS is implemented through performance prediction models that identify the best expected candidate algorithm for a given problem instance. Many studies have focused on utilizing handcrafted features to build these models. To eliminate the need of domain expertise for feature engineering, representation learning has been investigated. Convolutional Neural Networks (CNNs) (Loreggia et al. 2016) is an early example in this direction. Following that, considering the structural nature of the target problems, Graph Neural Networks (GNNs) have been adopted to learn instance characterizations from graph data (Cao et al. 2025). More recently, advances in Transformer architectures and large-scale embedding models have enabled the automatic extraction of high-dimensional latent representations (Wang et al. 2026).

## Method

## Problem Formulation

Following the standard Automated Algorithm Selection (AS) formulation (Rice 1976; Kerschke et al. 2019), we specify EEG Foundation Model (FM) selection as an per-instance AS problem over a portfolio of pretrained FMs. Let $\mathcal { M } =$ $\{ m _ { 1 } , \hdots , m _ { K } \}$ denote a portfolio of K frozen EEG FMs Although all models are evaluated on the same downstream EEG task within each dataset, their performances may vary substantially across diferent EEG instances due to their distinct architectures and learned representations.

For an EEG instance $x _ { i }$ and a $\mathrm { F M } m _ { k } \in \mathcal { M } , \mathrm { l e t } \rho ( m _ { k } , x _ { i } )$ is the utility of model $m _ { k }$ on instance $x _ { i } .$ . As the performance upper bound, Virtual Best Solver (VBS) / Oracle represents the theoretical performance limit, i.e. choosing the absolute best FM for each individual instance (Kerschke et al. 2019):

$$
m ^ { * } ( x _ { i } ) = \arg \operatorname* { m a x } _ { m _ { k } \in \mathcal { M } } \rho ( m _ { k } , x _ { i } ) .\tag{1}
$$

The goal is to learn a selector that predicts the best FM using only information available during inference. Specifically, the selector produces a relevance score $s ( x _ { i } , m _ { k } ) =$ $f _ { \theta } ( x _ { i } , m _ { k } )$ for each FM.

and chooses the one with the highest score:

$$
{ \hat { m } } ( x _ { i } ) = \arg \operatorname* { m a x } _ { m _ { k } \in \mathcal { M } } s ( x _ { i } , m _ { k } ) .\tag{2}
$$

The objective is to predict the Oracle model assignment $m ^ { * } ( x _ { i } )$ as accurately as possible without evaluating the entire FM portfolio for each EEG instance. At the same time, the Single Best Solver (SBS) is required to be outperformed. In this work, SBS is one FM delivering the best overall performance across the entire instance set. During training, additional information from FMs may be utilized to learn model suitability, while inference-time selection only relies on information available before executing the candidate FM.

## EEG-AS: Overall Framework

EEG-AS aims to perform eficient instance-level FM selection by reconstructing unavailable FM behaviors from a lightweight anchor model. As shown in Fig. 1, EEG-AS has three components: an EEG representation module, a conditional token predictor, and a token-based model selector.

Given an EEG instance, EEG-AS first extracts an inference-available representation. During training, prediction tokens from all FMs are available as privileged information. During inference, only the token from an anchor model is observed. The conditional token predictor reconstructs the missing FM tokens, and the reconstructed token matrix is passed to the selector to predict the most suitable FM for the current instance. The framework is optimized by combining token reconstruction and model selection objectives.

EEG Representation. To characterize EEG instances without evaluating the entire FM portfolio, EEG-AS relies on an inference-available representation $z _ { x } .$ The framework is agnostic to the choice of representation, provided that it can be extracted without executing the candidate FMs. Such representations may be obtained from pretrained EEG encoders, handcrafted EEG features, or a combination of both.

Let $e _ { x } = f _ { \mathrm { e m b } } ( x ) \in \mathbb { R } ^ { d _ { e } }$ is the embedding extracted by an inference-time EEG encoder, and $h _ { x } = \mathbf { \bar { f } } _ { \mathrm { h c } } ( x ) \in \mathbb { R } ^ { d _ { h } }$ denote handcrafted EEG features describing complementary signal characteristics. The final EEG representation is obtained by $z _ { x } = [ e _ { x } ; h _ { x } ]$ . In this work, $f _ { \mathrm { e m b } }$ is instantiated using a frozen BIOT encoder (Yang, Westover, and Sun 2023).

Privileged Token Representation. During training, EEG-AS has access to the prediction outputs of all foundation models (FMs) as privileged information. (Vapnik and Vashist 2009) Rather than using scalar performance scores, we represent the behavior of each FM by its prediction token.

![](images/93593c5d2831afcbc797a3b742f88b7d9c5cd7a6e135c6134e6839f1c55093b7.jpg)  
Figure 1: Overview of the EEG-AS. During training, privileged prediction tokens are available. During inference, only the anchor model is executed.

For an EEG instance $x ,$ the k-th FM produces a prediction token $t _ { k } = \phi _ { k } ( x ) \in \mathbb { R } ^ { C }$ where $\phi _ { k } ( \cdot )$ denotes the frozen FM, C is the number of target classes, and $t _ { k }$ is the predicted class probability vector. The prediction tokens from all FMs are organized into a privileged token matrix

$$
T = [ t _ { 1 } ; \dots ; t _ { K } ] \in \mathbb { R } ^ { K \times C } .\tag{3}
$$

Compared with scalar performance scores, prediction tokens preserve the complete prediction distributions of FMs, providing richer instance-specific information for modeling their behaviors and supporting downstream model selection.

Conditional Token Predictor. During inference, only the anchor model token $t _ { a }$ is available. To recover the unavailable FM behaviors, we introduce a conditional token predictor.

Given the EEG representation $z _ { x }$ and the observed anchor token $t _ { a } .$ , the predictor estimates the prediction tokens of all non-anchor foundation models:

$$
\hat { T } _ { \setminus a } = P _ { \psi } ( z _ { x } , t _ { a } ) ,\tag{4}
$$

where ${ \hat { T } } _ { \backslash a }$ contains the reconstructed tokens of all foundation models except the anchor model.

The reconstructed token matrix is defined as

$$
\tilde { t } _ { k } = \Big \{ t _ { a } , \quad k = a , \mathrm { w h e r e } \tilde { T } = [ \tilde { t } _ { 1 } ; \ldots ; \tilde { t } _ { K } ] .\tag{5}
$$

Unlike static representations shared across samples, the predicted tokens are conditioned on the current EEG instance, enabling sample-specific approximation of FM behaviors.

Token-based Model Selector. Given the reconstructed token matrix ${ \tilde { T } } _ { * }$ , the token-based model selector predicts the relevance of each FM.

To model interactions between EEG characteristics and FM behaviors, the selector employs a cross-attention mechanism (Vaswani et al. 2017). Each FM slot constructs a modelspecific query by combining the EEG representation with its token representation. These queries attend over the reconstructed token embeddings, allowing each model slot to aggregate information from the FM portfolio.

The selector outputs relevance scores

$$
s = f _ { \theta } ( z _ { x } , \tilde { T } ) \in \mathbb { R } ^ { K } .\tag{6}
$$

where $s _ { k }$ is the predicted relevance score of the k-th FM.

Joint Optimization. The conditional token predictor is first pretrained using the privileged tokens and is subsequently jointly optimized with the token-based model selector. The token reconstruction objective is defined as

$$
\mathcal { L } _ { \mathrm { t o k e n } } = \frac { 1 } { K - 1 } \sum _ { k \neq a } \| \hat { t } _ { k } - t _ { k } \| _ { 2 } ^ { 2 } .\tag{7}
$$

For an EEG instance $x$ with ground-truth label $y ,$ the relevance of the k-th FM is defined according to its prediction correctness:

$$
r _ { k } = { \bf 1 } \left( \arg \operatorname* { m a x } _ { c } { t _ { k } [ c ] } = y \right) .\tag{8}
$$

Since multiple FMs may correctly predict the same instance, pairwise ranking constraints are constructed only between relevant and irrelevant models:

$$
{ \mathcal { P } } ( x ) = \{ ( i , j ) \mid r _ { i } > r _ { j } \} .\tag{9}
$$

The selector is optimized using the pairwise logistic ranking loss (Burges et al. 2005):

$$
\mathcal { L } _ { \mathrm { s e l } } = \sum _ { ( i , j ) \in \mathcal { P } ( x ) } \log \left( 1 + \exp \bigl ( - \sigma \bigl ( s _ { i } - s _ { j } \bigr ) \bigr ) \right) .\tag{10}
$$

The overall training objective is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { s e l } } + \lambda \mathcal { L } _ { \mathrm { t o k e n } } , } \end{array}\tag{11}
$$

where λ balances the token reconstruction and model selection objectives.

Inference. At inference time, the privileged token matrix T is unavailable since obtaining it requires executing the entire FM portfolio. Instead, EEG-AS only requires the anchor FM during selection.

Given an unseen EEG instance, we first extract its EEG representation $z _ { x }$ and execute the anchor model to obtain the observed token $t _ { a }$ . The conditional token predictor then generates the missing non-anchor tokens. The reconstructed token matrix $\tilde { T }$ is constructed by combining the observed anchor token with the predicted tokens. The token-based selector then produces relevance scores and chooses the FM with the highest score. Therefore, EEG-AS performs eficient instance-level FM selection without exhaustive portfolio evaluation. If the selected model difers from the anchor model, the selected model is subsequently executed for the final downstream prediction.

## Experimental Setup

Datasets and Evaluation Protocol. Following the EEG-FM-Bench evaluation protocol (Xiong et al. 2025, 2026), we evaluate EEG-AS on seven publicly available EEG benchmark datasets: ADFTD (Miltiadous et al. 2023), BCIC-2a (Brunner et al. 2008), MIMUL-11 (Jeong et al. 2020), SEED-V (Li et al. 2019), SEED-VII (Jiang et al. 2025), THINGS-EEG-2 (Giford et al. 2022), Workload (Zyma et al. 2019). These datasets covering diverse downstream tasks and recording conditions (Table 1). For each dataset, we adopt the oficial train/validation/test split and follow the standardized downstream evaluation protocol provided by EEG-FM-Bench (Xiong et al. 2025, 2026).

<table><tr><td>Dataset</td><td>Task</td><td>Classes Subjects Channels</td><td></td></tr><tr><td>ADFTD</td><td>Clinical</td><td>3</td><td>88 19</td></tr><tr><td>BCIC-2a</td><td>Motor Img.</td><td>4</td><td>9 22</td></tr><tr><td>MIMUL-11</td><td>Motor Img.</td><td>11</td><td>25 60</td></tr><tr><td>SEED-V</td><td>Emotion</td><td>5</td><td>16 62</td></tr><tr><td>SEED-VII</td><td>Emotion</td><td>7</td><td>20 60</td></tr><tr><td>THINGS-EEG-2</td><td>Visual</td><td>2</td><td>10 64</td></tr><tr><td>Workload</td><td>Cognitive</td><td>2</td><td>36 30</td></tr></table>

Table 1: Overview of the EEG benchmark datasets

Foundation Model Portfolio. We construct the FM portfolio using seven representative EEG FMs, including BIOT(Yang, Westover, and Sun 2023), EEGPT(Wang et al. 2024), BENDR(Kostas, Aroca-Ouellette, and Rudzicz 2021), CBraMod(Wang et al. 2025), CSBrain(Zhou et al. 2026), LaBraM(Jiang, Zhao, and Lu 2024), and REVE(El Ouahidi et al. 2026). All FMs are frozen and are evaluated under the unified EEG-FM-Bench protocol. These models difer substantially in architecture design, pretraining strategy, and representation learning paradigm, constituting a diverse candidate portfolio for instance-level AS.

For each EEG instance, each FM produces a class probability vector, which is used as its prediction token. These tokens provide the foundation-model behavior information required for EEG-AS training.

During inference, EEG-AS requires only one anchor model to obtain the observed prediction token. In our experiments, BIOT is selected as the anchor model and is also used to provide the EEG encoder representation. The remaining FM behaviors are reconstructed by EEG-AS without executing the full portfolio.

Evaluation Metrics. We evaluate EEG-AS from the perspective of selection efectiveness. Accuracy is the main measure used to assess selection efectiveness:

$$
\mathrm { A c c } _ { \mathrm { s e l } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \rho ( \hat { m } ( x _ { i } ) , x _ { i } ) ,\tag{12}
$$

where $\hat { m } ( x _ { i } )$ denotes the FM selected by EEG-AS for instance $x _ { i } .$ , and $\rho ( m , x ) \in \{ 0 , 1 \}$ indicates whether model m correctly classifies instance x. This measure is bounded by the Virtual Best Solver (VBS) where the best FM is chosen for each instance. The AS performance potential can be further realized by checking the VBS-SBS gap:

$$
\mathrm { G a p } _ { \mathrm { V B S } } = \mathrm { A c c } _ { \mathrm { V B S } } - \mathrm { A c c } _ { \mathrm { S B S } } .\tag{13}
$$

A larger gap indicates greater instance-level diversity, ofering more opportunities for AS.

Baselines and Reference Methods. In addition to the performances of SBS and VBS / Oracle, two traditional AS models are built by using Multi-layer Perceptron (MLP) and Random Forest (RF), utilizing $z _ { x }$ . Additionally, we incorporate two reference scenarios for evaluation including:

Global Token Predictor This baseline represents unavailable foundation-model outputs using learnable global token representations. Unlike EEG-AS, the predicted tokens are independent of EEG instances.

– Privileged Teacher. Privileged Teacher receives the complete prediction token matrix generated by all foundation models and adopts the same token-based selection mechanism as EEG-AS. Since complete token access is unavailable during deployment, Teacher provides a privileged upper bound rather than a practical inference-time method.

Implementation Details. All experiments are implemented in PyTorch. The seven EEG foundation models (FMs) remain frozen throughout training and evaluation. The EEG representation $z _ { x } ,$ , including the BIOT embedding and handcrafted EEG features, is precomputed and kept fixed. We choose BIOT as the default anchor FM because it provides a lightweight and practical deployment choice. Compared with other EEG foundation models, BIOT has a relatively small parameter size (3.4M parameters), making it suitable as an inference-time anchor while maintaining competitive selection performance. Therefore, EEG-AS only optimizes the conditional token predictor and the token-based selector.

The conditional token predictor and token-based selector are implemented with lightweight neural architectures. EEG-AS is optimized through the aforementioned hybridized loss L with λ = 1.0. All trainable components are optimized using AdamW. All experiments are repeated with three random seeds. The reported results are their averages.

## Results

## Overall Selection Performance

Table 2 summarizes the accuracy of EEG-AS on seven EEG benchmarks. EEG-AS consistently outperforms the Single Best Solver (SBS) on all datasets, improving the average performance from 53.7% to 66.9%. This demonstrates the efectiveness of instance-level foundation model (FM) selection over using a single globally optimal FM.

Compared with conventional feature-based selectors, EEG-AS achieves higher average accuracy than MLP (64.8%) and Random Forest (62.7%), and also surpasses the Global Token Predictor (65.4%). EEG-AS achieves the best deployable performance on five out of seven datasets, with the largest improvement observed on BCIC-2a, where it exceeds the strongest deployable baseline by 4.9 percentage points. The Privileged Teacher obtains 69.0% by accessing complete FM prediction tokens, while the Oracle reaches 83.7%, indicating remaining room for improving FM behavior reconstruction under practical deployment constraints.

## Ablation Studies

To better understand the proposed framework, we conduct a series of ablation studies to answer four key questions:

(1) Do FM embeddings and handcrafted EEG features provide complementary information for AS?

(2) Does conditioning on an anchor FM improve the reconstruction of unavailable FM behaviors?

(3) Which supervision target is most efective for learning the reconstructed FM behaviors?

(4) How does the choice of the anchor FM afect the AS performance?

Unless otherwise specified, all variants follow the same training protocol as EEG-AS, and results are averaged over the seven benchmark datasets.

Input Representation. To investigate whether FM representations provide complementary information beyond handcrafted EEG features, we compare EEG-AS under three input configurations: removing handcrafted features (w/o Handcrafted Features), removing the foundation-model embedding (w/o Encoder Embedding), and using both representations (EEG-AS).

As summarized Table 3, removing the FM embedding decreases the accuracy from 66.9% to 66.3%, with a more noticeable degradation on BCIC-2a (56.0% to 48.6%). This suggests that pretrained EEG representations provide useful information for capturing instance-level characteristics that are dificult to describe using handcrafted features alone.

Meanwhile, removing handcrafted features leads to a larger overall drop when using only the encoder embedding (64.4%). The degradation is more pronounced on SEED-V (43.0% to 39.6%) and Workload (91.8% to 89.6%), indicating that handcrafted EEG descriptors still capture complementary task-specific information. By combining both representations, EEG-AS achieves the best average performance (66.9%), demonstrating that FM embeddings and handcrafted features provide complementary signals for FM selection.

Anchor Conditioning To evaluate the importance of instance-specific anchor conditioning, we compare EEG-AS with two variants: removing the anchor token (w/o Anchor Token) and replacing it with a learnable global token (Global Token Predictor). All variants use the same EEG representation and token selector.

Table 4 shows that removing the anchor token decreases the average selection accuracy from 66.9% to 66.3%, while replacing it with a global token further reduces the performance to 65.9%. The improvement is particularly evident on BCIC-2a and SEED-V, where instance-specific anchor conditioning provides more efective guidance for distinguishing FM behaviors. Although some individual datasets show comparable or slightly lower performance, EEG-AS achieves the best overall performance across the benchmark suite.

These results indicate that EEG representations alone or dataset-level priors are insuficient to capture instancedependent FM behaviors that an inference-available anchor FM provides valuable instance-level contextual information for reconstructing unavailable foundation-model behaviors, thereby improving downstream foundation-model selection.

Supervision Target To investigate the impact of the reconstruction target, we compare token-level supervision with correctness supervision while keeping the rest of EEG-AS unchanged.

Table 5 shows that Token Supervision achieves higher overall performance than Correctness Supervision, improving the dataset-level average from 65.4% to 66.9% and the instance-level average from 74.1% to 74.6%. The improvement is particularly noticeable on BCIC-2a and SEED-V, where reconstructing prediction tokens provides richer information for distinguishing foundation-model behaviors.

This advantage comes from preserving the complete prediction distribution of each FM, including confidence and class preference information, whereas binary correctness labels discard such fine-grained details. These results validate token-level behavior reconstruction as a more informative supervision target for instance-level FM selection.

Anchor Foundation Model. To evaluate the robustness of EEG-AS to the choice of anchor FM, we replace the anchor FM while keeping the remaining components unchanged. Table 6 shows that EEG-AS maintains stable performance across diferent anchor choices, with dataset-level accuracy ranging from 65.7% to 67.1%. No single anchor consistently dominates, indicating that the proposed conditional token reconstruction mechanism can efectively exploit instancelevel information from diferent FMs.

Although REVE achieves the highest dataset average and EEGPT obtains the highest instance average, BIOT provides comparable performance and serves as a practical default anchor because it simultaneously provides the EEG representation and anchor token required by EEG-AS.

<table><tr><td>Dataset Method</td><td>ADFTD</td><td>BCIC-2a</td><td>MIMUL-11</td><td>SEED-V</td><td>SEED-VII</td><td>THINGS-EEG-2</td><td>Workload</td><td>Dataset Avg.</td><td>Instance Avg.</td></tr><tr><td rowspan="2">SBS</td><td>REVE</td><td>REVE</td><td>CSBrain</td><td>EEGPT</td><td>EEGPT</td><td>REVE</td><td>CSBrain</td><td></td><td></td></tr><tr><td>60.2±0.0</td><td>31.9±0.0</td><td>67.9±0.0</td><td> $3 2 . 5 { \pm } 0 . 0 $ </td><td>29.0±0.0</td><td> $8 9 . 1 { \pm } 0 . 0 $ </td><td> $6 5 . 6 { \pm } 0 . 0 $ </td><td> $5 3 . 7 { \pm } 0 . 0 $ </td><td>68.8</td></tr><tr><td>MLP AS (Ours)</td><td> $7 7 . 2 { \pm } 0 . 7 $ </td><td> $5 0 . 0 { \pm } 3 . 8 $ </td><td> $7 2 . 8 { \pm } 0 . 3 $ </td><td> $3 8 . 9 { \pm } 0 . 5 $ </td><td> $3 2 . 5 { \pm } 1 . 5 $ </td><td> $9 0 . 8 { \pm } 0 . 1 $ </td><td> $9 1 . 3 { \pm } 0 . 9 $ </td><td> $6 4 . 8 { \pm } 1 . 1 $ </td><td>73.7</td></tr><tr><td>RF AS (Ours)</td><td> $7 5 . 3 { \pm } 0 . 7 $ </td><td> $4 6 . 3 { \pm } 1 . 8 $ </td><td>71.0±0.9</td><td> $3 8 . 7 { \pm } 0 . 9 $ </td><td> $3 2 . 5 { \pm } 2 . 2 $ </td><td> $8 9 . 5 { \pm } 0 . 1 $ </td><td> $8 5 . 8 { \pm } 3 . 4 $ </td><td> $6 2 . 7 { \pm } 1 . 4 $ </td><td>72.3</td></tr><tr><td>Global Token Predictor</td><td> $7 8 . 2 { \pm } 0 . 7 $ </td><td> $5 1 . 1 { \pm } 2 . 0 $ </td><td>74.2±0.4</td><td> $3 9 . 7 { \pm } 0 . 0 $ </td><td> $3 2 . 2 \pm 3 . 0 $ </td><td> ${ \bf 9 0 . 9 2 0 . 1 }$ </td><td> $9 1 . 3 { \pm } 0 . 9 $ </td><td> $6 5 . 4 \pm 1 . 0$ </td><td>74.3</td></tr><tr><td>Privileged Teacher</td><td>83.5±0.0</td><td> $5 1 . 4 { \pm } 1 . 3 $ </td><td>78.7±0.5</td><td> $4 2 . 7 { \pm } 0 . 3 $ </td><td> $4 3 . 5 { \pm } 0 . 8 $ </td><td> $9 2 . 0 { \pm } 0 . 1 $ </td><td> $9 0 . 7 { \pm } 1 . 9 $ </td><td>69.0±0.7</td><td>77.7</td></tr><tr><td>EEG-AS (Ours)</td><td>79.5±0.7</td><td> ${ \pm } 6 . 0 { \pm } 1 . 5 $ </td><td>73.5±0.4</td><td>43.0±1.2</td><td> $3 3 . 7 { \pm } 1 . 3 $ </td><td> $9 0 . 5 { \pm } 0 . 2 $ </td><td> ${ \bf 9 1 . 8 { \pm 0 . 0 } }$ </td><td> ${ \bf 6 6 . 9 2 0 . 7 }$ </td><td>74.6</td></tr><tr><td>Oracle (VBS)</td><td>84.0±0.0</td><td> $8 0 . 2 { \pm } 0 . 0 $ </td><td>90.9±0.0</td><td> $5 9 . 6 { \pm } 0 . 0 $ </td><td> $7 6 . 1 { \pm } 0 . 0 $ </td><td> $9 6 . 9 { \pm } 0 . 0 \ $ </td><td> $9 8 . 4 { \pm } 0 . 0 $ </td><td> $8 3 . 7 { \pm } 0 . 0 $ </td><td>88.9</td></tr></table>

Table 2: Overall AS accuracy (%) on seven EEG benchmarks. Learning-based selectors are reported as mean±stdev. over three random seeds, while SBS and Oracle are deterministic reference methods. Dataset Avg. is the macro-average across datasets; Instance Avg. aggregates over all evaluation instances. The best deployable method is highlighted in bold.
<table><tr><td>Method</td><td>ADFTD BCIC-2a MIMUL-11 SEED-V SEED-VII THINGS-EEG-2 Workload |Dataset Avg. Instance</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>w/o Handcrafted Features</td><td> $7 7 . 6 { \pm } 0 . 7 ~ 4 8 . 3 { \pm } 1 . 5$ </td><td> $7 2 . 9 { \pm } 0 . 9 $   $3 9 . 6 { \pm } 0 . 2 $ </td><td>32.5±0.8</td><td> $9 0 . 5 { \pm } 0 . 0 $ </td><td> $8 9 . 6 { \pm } 0 . 9 $ </td><td> $6 4 . 4 { \pm } 0 . 7 $ </td><td>73.6</td></tr><tr><td>w/o Encoder Embedding</td><td> $7 8 . 6 { \pm } 0 . 9 \ 4 8 . 6 { \pm } 1 . 3$ </td><td> ${ \bf 7 4 . 4 \pm 0 . 6 }$ </td><td> ${ \bf 4 4 . 0 { \pm } 0 . 7 }$   ${ \bf 3 6 . 5 { \pm 0 . 4 } }$ </td><td> $\mathbf { 9 0 . 7 \pm 0 . 2 }$ </td><td> $9 1 . 3 { \pm } 1 . 9 $ </td><td> $6 6 . 3 { \pm } 0 . 9$ </td><td>75.0</td></tr><tr><td>EEG-AS (Ours)</td><td> $\mathbf { 7 9 . 5 \pm 0 . 7 \ 5 6 . 0 \pm 1 . 5 }$ </td><td> $7 3 . 5 { \pm } 0 . 4 $ </td><td> $4 3 . 0 { \pm } 1 . 2 $ </td><td> $3 3 . 7 { \pm } 1 . 3 $ </td><td> $9 0 . 5 { \pm } 0 . 2 $   ${ \bf 9 1 . 8 { \pm 0 . 0 } }$ </td><td> ${ \bf 6 6 . 9 2 0 . 7 }$ </td><td>74.6</td></tr></table>

Table 3: Ablation study on EEG representations in EEG-AS. w/o Handcrafted Features removes handcrafted EEG descriptors, while w/o Encoder Embedding removes the FM embedding. EEG-AS uses both representations as the default configuration. Selection accuracy (%) is mean±std over seeds.

The unified deployment results in Table 7 further confirm this observation: replacing BIOT with REVE leads to comparable selection accuracy (66.9% vs. 67.0%), demonstrating that EEG-AS is robust to diferent anchor configurations.

## Analysis

Reconstructed Foundation-model Behavior. To verify whether EEG-AS can recover unavailable foundation-mode (FM) behaviors from an inference-available anchor FM, we evaluate the reconstructed tokens using cosine similarity, Spearman rank correlation between FM rankings, and Top-1 agreement between the reconstructed and true best FMs. Table 8 and Fig. 2 summarize the results. Overall, EEG-AS achieves an average cosine similarity of 0.66, a ranking correlation of 0.67, and a Top-1 agreement of 0.53, demonstrating that the conditional token predictor captures meaningful FM behaviors rather than learning arbitrary representations.

The reconstruction quality varies across datasets. SEED-V and MIMUL-11 show the highest reconstruction fidelity, while THINGS-EEG-2 remains more challenging as the relationships among FM behaviors are dataset dependent.

Moreover, Top-1 agreement is consistently lower than token similarity and ranking correlation, suggesting that reconstructing the general behavior of FMs is easier than precisely identifying the optimal FM for each instance.

Understanding Performance Gaps. Although EEG-AS efectively reconstructs unavailable FM behaviors, the selection gains vary across datasets. We identify three factors that limit the final AS performance.

First, the achievable improvement is bounded by the selection headroom between SBS and Oracle. For example, on THINGS-EEG-2, SBS already achieves 89.1% accuracy while Oracle reaches 96.9%, leaving limited room for instance-level selection improvements. Therefore, the modest gain on this dataset is mainly due to the limited selection margin rather than inefective behavior reconstruction.

![](images/5d8ba8c35048198e857e2e4e5eb7743ba4b8536a0fbdf9db0fc27e2e6c32ae37.jpg)  
Figure 2: Token-reconstruction metrics on each dataset (mean±std over seeds: cosine similarity $R _ { D }$ , Spearman rank correlation $\rho _ { D }$ , and Top-1 agreement of arg $\operatorname* { m a x } _ { m } P ( y \mid z )$

Second, accurate token reconstruction does not necessarily guarantee optimal FM selection. On SEED-VII, EEG-AS achieves strong reconstruction quality $( R _ { D } = 0 . 7 4 , \rho _ { D } =$ 0.76), but the selection accuracy remains limited. Moreover, the Privileged Teacher using true FM tokens only achieves 43.5%, far below the Oracle performance of 76.1%. These results reveal that FM selection is not solely a representation reconstruction problem, but a decision alignment problem.

Finally, FM diversity afects the efectiveness of selection. When multiple FMs exhibit highly similar prediction behaviors, their tokens provide limited discriminative information for identifying the optimal model.

Overall, these findings suggest that efective FM selection requires not only accurate behavior reconstruction, but also suficient selection headroom and decision-relevant diversity

<table><tr><td> $\widehat { \mathbf { M e t h o d } } ^ { \mathbf { D a t a s e t } }$ </td><td>ADFTD BCIC-2a MIMUL-11 SEED-V SEED-VII THINGS-EEG-2 Workload Dataset Avg. Inst. Avg.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>w/o Anchor Token</td><td> $7 7 . 8 \pm 0 . 5 ~ 5 4 . 3 \pm 3 . 0$ </td><td></td><td> $7 2 . 5 { \pm } 0 . 5 $ </td><td> $4 1 . 9 { \pm } 1 . 8 $ </td><td>35.0±1.7</td><td> ${ \bf 9 0 . 6 { \pm 0 . 1 } }$ </td><td> $9 1 . 8 { \pm } 1 . 6 $ </td><td> $6 6 . 3 { \pm } 1 . 3 $ </td><td>74.3</td></tr><tr><td>Global Token Predictor</td><td> $7 8 . 6 { \pm } 0 . 9 \ 5 4 . 3 { \pm } 1 . 7$ </td><td></td><td> $7 2 . 9 { \pm } 0 . 7 $ </td><td> $4 0 . 2 { \pm } 1 . 0 3 3 . 6 { \pm } 1 . 9$ </td><td></td><td> ${ \bf 9 0 . 6 { \pm 0 . 1 } }$ </td><td> $9 1 . 3 { \pm } 0 . 9 $ </td><td> $6 5 . 9 { \pm } 1 . 0 $ </td><td>74.1</td></tr><tr><td>EEG-AS (Ours)</td><td> $7 9 . 5 { \pm } 0 . 7 ~ 5 6 . 0 { \pm } 1 . 5$ </td><td></td><td> ${ \bf 7 3 . 5 { \pm 0 . 4 } }$ </td><td> ${ \bf 4 3 . 0 { \pm 1 . 2 } }$ </td><td> $3 3 . 7 { \pm } 1 . 3 $ </td><td> $9 0 . 5 { \pm } 0 . 2 $ </td><td> ${ \bf 9 1 . 8 { \pm 0 . 0 } }$ </td><td> ${ \bf 6 6 . 9 \pm 0 . 7 }$ </td><td>74.6</td></tr></table>

Table 4: Ablation study on the efectiveness of instance-specific anchor conditioning. w/o Anchor Token and Global Token Predictor reconstruct all foundation-model tokens without the true BIOT token, while EEG-AS conditions token reconstruction on the instance BIOT token and retains the observed anchor token during selection.
<table><tr><td> $\widehat { \mathbf { M e t h o d } } ^ { \mathbf { D a t a s e t } }$ </td><td>ADFTD BCIC-2a MIMUL-11 SEED-V SEED-VII THINGS-EEG-2 Workload Dataset Avg. Inst. Avg.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Correctness Supervision</td><td>|79.5±0.9  $4 9 . 7 { \pm } 1 . 0 $ </td><td> $7 2 . 7 { \pm } 0 . 2 $ </td><td>39.8±1.7</td><td> ${ \bf 3 4 . 5 { \pm 0 . 4 } }$ </td><td> ${ \bf 9 0 . 6 { \pm 0 . 1 } }$ </td><td> $9 0 . 7 { \pm } 0 . 9$ </td><td> $6 5 . 4 { \pm } 0 . 7 $ </td><td>74.1</td></tr><tr><td>Token Supervision (EEG-AS)79.5±0.7</td><td> ${ \pm } 6 . 0 { \pm } 1 . 5 $ </td><td> $7 3 . 5 { \pm } 0 . 4 $ </td><td>43.0±1.2</td><td> $3 3 . 7 { \pm } 1 . 3 $ </td><td> $9 0 . 5 { \pm } 0 . 2 $ </td><td> ${ \bf 9 1 . 8 { \pm 0 . 0 } }$ </td><td> ${ \bf 6 6 . 9 2 0 . 7 }$ </td><td>74.6</td></tr></table>

Table 5: Efect of supervision target for conditional FM behavior reconstruction. Correctness Supervision trains the predictor to estimate FM correctness, while Token Supervision reconstructs the full prediction tokens used in EEG-AS.
<table><tr><td> $\scriptstyle \sum _ { \mathbf { A n c h o r } } \mathbf { D a t a s e t }$ </td><td>ADFTD BCIC-2a MIMUL-11 SEED-V SEED-VII THINGS-EEG-2 Workload Dataset Avg.</td><td></td><td></td><td></td><td></td><td></td><td></td><td>Inst. Avg.</td></tr><tr><td>BENDR</td><td> $7 9 . 8 { \pm } 0 . 7 $   $5 3 . 2 { \pm } 2 . 8 $ </td><td> $7 4 . 6 { \pm } 0 . 4 $ </td><td> $4 1 . 8 { \pm } 0 . 5 $ </td><td> $3 3 . 5 { \pm } 2 . 5 $ </td><td> ${ \bf 9 1 . 5 { \pm 0 . 3 } }$ </td><td> $9 1 . 8 { \pm } 0 . 0 $ </td><td> $6 6 . 6 { \pm } 1 . 0 $ </td><td>75.1</td></tr><tr><td>CBraMod</td><td> $\mathbf { 8 0 . 4 \pm 1 . 5 \ 5 4 . 9 \pm 2 . 2 }$ </td><td> $7 3 . 0 { \pm } 0 . 3 $ </td><td> $4 1 . 9 { \pm } 0 . 4 $ </td><td> $3 2 . 5 { \pm } 1 . 5 $ </td><td> $9 0 . 9 { \pm } 0 . 3 $ </td><td> $8 9 . 6 { \pm } 2 . 5 $ </td><td> $6 6 . 2 { \pm } 1 . 2 $ </td><td>74.4</td></tr><tr><td>CSBrain</td><td>78.5±0.2 56.0±0.9</td><td> ${ \bf 7 5 . 1 \pm 0 . 6 }$ </td><td> $\pm 2 . 5 { \pm } { \bf 0 . 8 }$ </td><td> $3 2 . 9 { \pm } 1 . 9 $ </td><td> $9 0 . 8 { \pm } 0 . 1 $ </td><td> $9 0 . 2 { \pm } 2 . 8 $ </td><td> $6 6 . 6 { \pm } 1 . 0 $ </td><td>74.9</td></tr><tr><td>EEGPT</td><td> $7 8 . 2 { \pm } 0 . 2 \ 5 2 . 3 { \pm } 0 . 5$ </td><td> $7 3 . 9 { \pm } 1 . 2 $ </td><td> $4 1 . 5 { \pm } 0 . 5 $ </td><td> $\mathbf { 3 8 . 6 { \pm 0 . 9 } }$ </td><td> $9 1 . 4 { \pm } 0 . 2 $ </td><td> $9 0 . 7 { \pm } 1 . 9 $ </td><td> $6 6 . 7 { \pm } 0 . 8 $ </td><td>75.3</td></tr><tr><td>LaBraM</td><td> $7 8 . 2 { \pm } 0 . 9 \ 5 1 . 4 { \pm } 1 . 3$ </td><td> $7 3 . 3 { \pm } 0 . 7 $ </td><td> $4 1 . 6 { \pm } 1 . 3 $ </td><td> $3 3 . 8 { \pm } 2 . 6 $ </td><td> $9 0 . 6 { \pm } 0 . 3 $ </td><td> $9 1 . 3 { \pm } 0 . 9 $ </td><td> $6 5 . 7 { \pm } 1 . 1 $ </td><td>74.2</td></tr><tr><td>REVE</td><td> $7 8 . 4 \pm 0 . 4$   $5 5 . 5 { \pm } 1 . 8 $ </td><td> $7 2 . 8 { \pm } 0 . 9 $ </td><td> $4 1 . 8 { \pm } 0 . 8 $ </td><td> $3 7 . 4 { \pm } 0 . 7 $ </td><td> $9 1 . 3 { \pm } 0 . 1 $ </td><td> ${ \bf 9 2 . 3 { \pm 1 . 9 } }$ </td><td> ${ \bf 6 7 . 1 { \pm 0 . 9 } }$ </td><td>75.0</td></tr><tr><td>BIOT (Ours)</td><td> $8 0 . 2 { \pm } 0 . 7 $   $5 4 . 3 { \pm } 0 . 0 $ </td><td> $7 3 . 2 { \pm } 0 . 4 $ </td><td> $4 2 . 3 { \pm } 0 . 4 $ </td><td> $3 2 . 9 { \pm } 1 . 3 $ </td><td> $9 0 . 5 { \pm } 0 . 1 $ </td><td> $9 1 . 3 { \pm } 2 . 5 $ </td><td> $6 6 . 4 { \pm } 0 . 8 $ </td><td>74.3</td></tr></table>

Table 6: Robustness of EEG-AS to diferent anchor FMs. Each variant replaces only the inference-available anchor token while keeping the EEG representation, conditional token predictor, token-based selector, and training protocol unchanged.
<table><tr><td>Anchor</td><td>Dataset | ADFTD BCIC-2a MIMUL-11 SEED-V SEED-VII THINGS-EEG-2 Workload Dataset Avg. Inst. Avg.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BIOT</td><td> $7 9 . 5 { \pm } 0 . 7 ~ 5 6 . 0 { \pm } 1 . 5$ </td><td> $7 3 . 5 { \pm } 0 . 4 $ </td><td> $4 3 . 0 { \pm } 1 . 2 $ </td><td> $3 3 . 7 { \pm } 1 . 3 $ </td><td> $9 0 . 5 { \pm } 0 . 2 $ </td><td> ${ \bf 9 1 . 8 { \pm 0 . 0 } }$ </td><td> $6 6 . 9 { \pm } 0 . 7 \ $ </td><td>74.6</td></tr><tr><td>REVE</td><td> $7 7 . 9 { \pm } 0 . 0 \ 4 8 . 9 { \pm } 2 . 0 $ </td><td> $\mathbf { 7 8 . 4 \pm 0 . 4 }$ </td><td> ${ \bf 4 4 . 3 } \pm { \bf 0 . 7 }$ </td><td> $3 7 . 5 { \pm } 1 . 3 $ </td><td> ${ \bf 9 1 . 2 \pm 0 . 1 }$ </td><td> $9 0 . 7 { \pm } 1 . 9 $ </td><td> ${ \bf 6 7 . 0 { \pm } 0 . 9 }$ </td><td>76.3</td></tr></table>

Table 7: Unified deployment with diferent anchor foundation models. Each variant uses the same foundation model for both EEG representation extraction and anchor token generation.

<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=2> $R _ { D }$      $\mathrm { R a n k i n g } \rho _ { D }$  ${ \mathrm { T o p } } { \cdot } 1 \ { \mathrm { A g r } } .$ </td></tr><tr><td rowspan=1 colspan=1>ADFTD</td><td rowspan=1 colspan=2> $0 . 6 7 { \pm } 0 . 0 1$   $0 . 7 1 { \pm } 0 . 0 1$   $0 . 6 1 { \pm } 0 . 0 3$ </td></tr><tr><td rowspan=1 colspan=1>BCIC-2a</td><td rowspan=1 colspan=2> $0 . 5 6 { \pm } 0 . 0 2$   $0 . 6 3 { \pm } 0 . 0 2$   $0 . 5 9 { \pm } 0 . 0 1$ </td></tr><tr><td rowspan=1 colspan=1>MIMUL-11</td><td rowspan=2 colspan=2>0.79±0.00  $0 . 6 9 { \pm } 0 . 0 0$   $0 . 5 5 { \pm } 0 . 0 2$  $0 . 8 1 { \pm } 0 . 0 0$   $0 . 7 1 { \pm } 0 . 0 2$ </td></tr><tr><td rowspan=1 colspan=1>SEED-V</td><td rowspan=1 colspan=1>0.02 0.60±0.00</td></tr><tr><td rowspan=1 colspan=1>SEED-VII</td><td rowspan=2 colspan=2> $0 . 7 6 { \pm } 0 . 0 2$   $0 . 5 4 { \pm } 0 . 0 3$  $0 . 3 8 { \pm } 0 . 0 3$   $0 . 4 3 { \pm } 0 . 0 2$   $0 . 3 6 { \pm } 0 . 0 1$  $0 . 6 9 { \pm } 0 . 0 2$   $0 . 7 3 { \pm } 0 . 0 1$   $0 . 4 8 { \pm } 0 . 0 6$ </td></tr><tr><td rowspan=1 colspan=1>THINGS-EEG-2Workload</td></tr><tr><td rowspan=1 colspan=1>Average</td><td rowspan=1 colspan=2>0.66        0.67        0.53</td></tr></table>

Table 8: Token reconstruction analysis. $R _ { D } \colon$ mean cosine similarity between predicted and true tokens over unavailable FMs. ρ<sub>D</sub>: mean per-instance Spearman correlation of FM rankings by $P ( y \mid z )$ . Top-1: agreement of the highest- $\mathbf { \nabla } \cdot P ( y )$ model under true vs. reconstructed tokens $( = 1 / \bar { | \mathcal { M } | } )$

among candidate FMs.

In this paper, we propose EEG-AS, an instance-level Algorithm Selection (AS) framework for EEG foundation models (FMs). By reconstructing unavailable FM behaviors from an inference-available anchor model, EEG-AS enables adaptive selection of FMs for diferent EEG instances.

## Conclusion

Experiments on seven EEG benchmarks demonstrate that EEG-AS consistently outperforms conventional selection strategies as well as standalone FMs. Further analysis shows that the reconstructed tokens capture meaningful FM behaviors, while the remaining gap depends on the dificulty of identifying decision-relevant model behaviors.

In the future, we will investigate cost-aware FM selection by jointly considering model accuracy, inference time, and computational resources.

## References

Brunner, C.; Leeb, R.; Müller-Putz, G. R.; Schlögl, A.; and Pfurtscheller, G. 2008. BCI Competition 2008–Graz Data Set A.

Burges, C. J. C.; Shaked, T.; Renshaw, E.; Lazier, A.; Deeds, M.; Hamilton, N.; and Hullender, G. 2005. Learning to Rank using Gradient Descent. In Proceedings of the 22nd International Conference on Machine Learning (ICML), 89– 96. ACM.

Cao, S.; Wu, H.; Wang, J. B.; Yuan, Y.; and Misir, M. 2025. MC-GNNAS-Dock: Multi-criteria gnn-based algorithm selection for molecular docking. In Pacific Rim International Conference on Artificial Intelligence (PRICAI), 686–696. Springer.

El Ouahidi, Y.; Lys, J.; Thölke, P.; Farrugia, N.; Pasdeloup, B.; Gripon, V.; Jerbi, K.; and Lioi, G. 2026. REVE: A foundation model for EEG-adapting to any setup with large-scale pretraining on 25,000 subjects. Advances in Neural Information Processing Systems (NeurIPS), 38: 22541–22577.

Giford, A. T.; Dwivedi, K.; Roig, G.; and Cichy, R. M. 2022. A Large and Rich EEG Dataset for Modeling Human Visual Object Recognition. NeuroImage, 264: 119754.

Jeong, J.-H.; Cho, J.-H.; Shim, K.-H.; Kwon, B.-H.; Lee, B.- H.; Lee, D.-Y.; Lee, D.-H.; and Lee, S.-W. 2020. Multimodal Signal Dataset for 11 Intuitive Movement Tasks from Single Upper Extremity During Multiple Recording Sessions. GigaScience, 9(10): giaa098.

Jiang, W.-B.; Liu, X.-H.; Zheng, W.-L.; and Lu, B.-L. 2025. SEED-VII: A Multimodal Dataset of Six Basic Emotions With Continuous Labels for Emotion Recognition. IEEE Transactions on Afective Computing, 16(2): 969–983.

Jiang, W.-B.; Zhao, L.-M.; and Lu, B.-L. 2024. Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI. In International Conference on Learning Representations (ICLR).

Kerschke, P.; Hoos, H. H.; Neumann, F.; and Trautmann, H. 2019. Automated algorithm selection: Survey and perspectives. Evolutionary Computation, 27(1): 3–45.

Kostas, D.; Aroca-Ouellette, S.; and Rudzicz, F. 2021. BENDR: Using Transformers and Large Scale Self-Supervised Learning for Generalizable Neural Decoding. Frontiers in Human Neuroscience, 15: 653659.

Kuruppu, G.; Wagh, N.; Kremen, V.; and Varatharajah, Y. 2026. EEG foundation models: a critical review of current progress and future directions. Journal of Neural Engineering, 23: 021001.

Li, T.-H.; Liu, W.; Zheng, W.-L.; and Lu, B.-L. 2019. Classification of Five Emotions from EEG and Eye Movement Signals: Discrimination Ability and Stability over Time. In 2019 9th International IEEE/EMBS Conference on Neural Engineering (NER), 804–807. IEEE.

Loreggia, A.; Malitsky, Y.; Samulowitz, H.; and Saraswat, V. A. 2016. Deep Learning for Algorithm Portfolios. In Proceedings of the 13th Conference on Artificial Intelligence (AAAI), 1280–1286.

Miltiadous, A.; Tzimourta, K. D.; Afrantou, T.; Ioannidis, P.; Grigoriadis, N.; Tsipouras, M. G.; Glavas, E.; Giannakeas, N.; and Tzallas, A. T. 2023. A Dataset of Scalp EEG Recordings of Alzheimer’s Disease, Frontotemporal Dementia and Healthy Subjects from Routine EEG. Data, 8(5): 95.

Rice, J. R. 1976. The Algorithm Selection Problem. Advances in Computers, 15: 65–118.

Vapnik, V.; and Vashist, A. 2009. A New Learning Paradigm: Learning Using Privileged Information. Neural Networks, 22(5–6): 544–557.

Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, Ł.; and Polosukhin, I. 2017. Attention Is All You Need. In Advances in Neural Information Processing Systems, volume 30, 5998–6008. Curran Associates, Inc.

Wang, G.; Liu, W.; He, Y.; Xu, C.; Ma, L.; and Li, H. 2024. EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals. Advances in Neural Information Processing Systems, 37: 39249–39280.

Wang, J.; Zhao, S.; Luo, Z.; Zhou, Y.; Jiang, H.; Li, S.; Li, T.; and Pan, G. 2025. Cbramod: A criss-cross brain foundation model for eeg decoding. In International conference on learning representations, volume 2025, 75310–75346.

Wang, J. B.; Cao, S.; Wu, H.; Yuan, Y.; and Mısır, M. 2026. Molecular embedding-based algorithm selection in proteinligand docking. Journal ofCheminformatics.

Xiong, W.; Li, J.; Li, J.; and Zhu, K. 2025. EEG-FM-Bench: A Comprehensive Benchmark for the Systematic Evaluation of EEG Foundation Models. arXiv preprint arXiv:2508.17742.

Xiong, W.; Li, J.; Li, J.; Zhu, K.; and Jiang, C. 2026. EEG-FM-Bench: A Comprehensive Benchmark for the Systematic Evaluation and Diagnostic Analyses of EEG Foundation Models. In ICML, Poster.

Yang, C.; Westover, M.; and Sun, J. 2023. Biot: Biosignal transformer for cross-data learning in the wild. Advances in Neural Information Processing Systems (NeurIPS), 36: 78240–78260.

Zhou, Y.; Wu, J.; Ren, Z.; Yao, Z.; Lu, W.; Peng, K.; Zheng, Q.; Song, C.; Ouyang, W.; and Gou, C. 2026. Csbrain: A cross-scale spatiotemporal brain foundation model for eeg decoding. Advances in Neural Information Processing Systems (NeurIPS), 38: 87150–87195.

Zyma, I.; Tukaev, S.; Seleznov, I.; Kiyono, K.; Popov, A.; Chernykh, M.; and Shpenkov, O. 2019. Electroencephalograms During Mental Arithmetic Task Performance. Data, 4(1): 14.