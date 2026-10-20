# CODA: Cohort- and Drift-aware Foundation Model for Multimodal Clinical Reasoning

Changshuo Liu School of Computing National University of Singapore

Wenqiao Zhang School of Software Technology Zhejiang University

Jiaqi Zhu<sup>∗</sup> School of Computing National University of Singapore

Xiaokui Xiao School of Computing National University of Singapore

Beng Chin Ooi School of Software Technology Zhejiang University

## Abstract

Recent advances in multimodal foundation models have demonstrated strong performance across diverse clinical tasks. However, existing approaches predominantly operate at the individual sample level and fail to model the interplay between population-level cohort patterns and patient-specific variation that underlie real world clinical reasoning. In practice, clinicians routinely reason through patient cohorts sharing similar demographic, physiological, or pathological characteristics, while accounting for patient-specific deviations from the group prototype. Motivated by this gap, we propose CODA, a cohort- and drift-aware foundation model for multimodal clinical reasoning. CODA explicitly integrates cohort structure into the modeling pipeline through two complementary mechanisms: cohort-aware population stratification and drift-aware individual adaptation. In particular, we leverage reinforcement learning to discover clinically meaningful patient cohorts, and further model each patient’s deviation from cohort prototypes as a structured drift signal encoding fine-grained individual heterogeneity. These cohort- and drift-aware representations are incorporated into the foundation model to enable reasoning that is simultaneously informed by population-level patterns and sensitive to individual-level variation. Extensive experiments on real-world electronic health record (EHR) benchmarks demonstrate that CODA achieves consistent effectiveness and strong generalizability across diverse clinical tasks, including closed-QA, open-QA, and report generation.

## 1 Introduction

Generative AI has rapidly evolved from text-only large language models (LLMs) [1, 2] to large vision–language models (LVLMs) [3], enabling unified reasoning over both textual and visual modalities. This paradigm shift is particularly consequential in the medical domain, where clinical decision-making inherently relies on heterogeneous data sources, including medical images, EHR data, and free-text reports [4, 5, 6, 7]. By integrating these modalities within a single framework, recent approaches have demonstrated strong capabilities in medical question answering [8], multimodal understanding [9], and medical decision support [10]. These advances have driven notable progress in applications such as medical knowledge acquisition and education, treatment planning, and patient outcome prediction [11, 12].

![](images/65915ff6bbbfa6762f51895f55b9bc603bcfdfdfd595bcb9e89c4603f5402ae6.jpg)  
Figure 1: Rethinking Clinical Reasoning: From instance-level predictions to cohort-aware modeling.

Despite these advances, existing medical foundation models [13, 14, 15] are developed in a taskcentric manner, where reasoning is performed on individual queries with limited consideration of broader clinical context. However, clinicians rarely reason about a patient in isolation. Instead, they implicitly compare the current case with previously encountered patients, drawing on experience with recurring patterns across similar populations [16]. Without this kind of contextual grounding, medical foundation models struggle to capture meaningful clinical variation, often leading to fragmented or overly simplified reasoning.

A natural way to incorporate such context is through patient cohorts [17]. By grouping patients with similar clinical profiles, such as demographics or specific health conditions, cohorts help guide how individual cases are interpreted and managed. Figure 1 illustrates the transition from conventiona instance-level reasoning to cohort-aware clinical reasoning. Take the age cohorts as an example, a sudden episode of chest pain may be treated as a potential medical emergency for elderly patients with a history of heart disease, often requiring immediate hospital evaluation, whereas the same symptoms for young and healthy individuals may be attributed to benign causes such as muscle strain or stress. This type of cohort-based reasoning allows clinical decisions to be grounded in population-level patterns.

Furthermore, relying on cohort structure alone is insufficient to support accurate clinical decisions. Even within the same cohort, patients may differ in ways that lead to different clinical judgments. For example, two elderly patients with similar cardiovascular risk factors may both have chest pain, yet one may show stable vital signs and mild symptoms that allow for conservative monitoring, while the other may exhibit severe pain that demands immediate intervention. Such differences in symptoms and clinical signals often determine the final intervention, yet are not well captured by coarse cohort patterns. As a result, effective clinical decision-making requires jointly modeling cohort structure and individual heterogeneity, a capability that remains underexplored in existing approaches.

To this end, we propose CODA<sup>2</sup>, a COhort- and Drift-Aware foundation model that explicitly characterizes both cohort structure and individual heterogeneity. Our framework consists of two key components. First, we formulate cohort construction as a reinforcement learning process that automatically partitions patients into clinically meaningful groups. This process builds a cohort tree, where each internal node corresponds to a feature-based status and action, and each leaf represents a cohort. A cohort encoder is then employed to encode these cohorts, and the resulting embeddings are injected into the model input to provide cohort-level context. Second, we move beyond static cohorts and introduce a continuously evolving cohort prototype space, where cohort representations are dynamically updated. For each patient, we further compute a cohort-drift signal that captures how the individual deviates from the corresponding cohorts. This signal reflects patient-specific characteristics that are not explained by shared cohort patterns, and is incorporated into the model to refine its reasoning at the individual level. These cohort-aware and drift-aware representations enable more precise and context-aware clinical reasoning. We summarize our main contributions as follows:

• We propose a cohort- and drift-aware foundation model for multimodal clinical reasoning that jointly incorporates population-level cohort structure and patient-specific heterogeneity.

• We present a reinforcement learning-based cohort discovery and dynamic prototype updates strategy, enabling adaptive and data-driven cohort modeling.

• We explicitly model the individual deviations from cohort prototypes, and incorporate such signals into the backbone through drift-conditioned adapters.

• Extensive experiments demonstrate that CODA consistently outperforms both general-purpose and medical-specific baselines across closed-QA, open-QA, and report generation tasks on diverse real-world EHR datasets, highlighting its effectiveness in enhancing clinical reasoning.

## 2 Preliminary

## 2.1 Multimodal Clinical Data Representation in Foundation Models

Modern foundation models encode heterogeneous clinical data by projecting multimodal inputs into a unified token embedding space. Given the input $\mathbf { x } = \{ \boldsymbol { \mathcal { T } } , \boldsymbol { \mathcal { X } } \}$ and ground truth output sequence $\mathbf { y } _ { \mathrm { : } }$ where $\mathcal { T } \in \mathbb { R } ^ { H \times W \times 3 }$ represents a medical image and $\mathcal { X } = \{ \hat { x _ { 1 } } , x _ { 2 } , \ldots , x _ { n } , \mathbf { q } \}$ denotes n serialized clinical records and the question ${ \bf q } ,$ existing LVLMs first encode the visual and textual inputs separately, and then fuse them into a shared embedding space as

$$
{ \bf Z } ^ { ( 0 ) } = [ f _ { I  t } ( \mathcal { E } _ { I } ( \mathcal { T } ) ) ; \mathcal { E } _ { t } ( \mathcal { X } ) ] \in \mathbb { R } ^ { ( L _ { I } + L _ { t } ) \times d _ { t } }\tag{1}
$$

where $\mathcal { E } _ { I } ( \cdot )$ is a vision encoder and $\mathcal { E } _ { t } ( \cdot )$ is the textual encoder. $L _ { I }$ and $L _ { t }$ denote the numbers of visual and textual tokens, respectively. $f _ { I  t }$ maps the visual embeddings into the shared embedding space, and $d _ { t }$ denotes the dimensions of the shared embedding space.

The fused embedding is then processed by $L _ { b }$ Transformer blocks $\mathcal { F } , \mathbf { Z } ^ { ( \ell ) } = \mathcal { F } ^ { ( \ell ) } \big ( \mathbf { Z } ^ { ( \ell - 1 ) } \big ) , l =$ $1 , \ldots , L _ { b }$ yielding the final contextualized representation $\mathbf { Z } ^ { ( L _ { b } ) }$ . Conditioned on $\mathbf { Z } ^ { ( L _ { b } ) }$ , the model generates the output sequence autoregressively with learnable parameters $\theta$ as

$$
p _ { \theta } ( \mathbf { y } \mid \mathbf { x } ) = \prod _ { i = 1 } ^ { | \mathbf { y } | } p _ { \theta } \left( y _ { i } \mid y _ { < i } , \mathbf { Z } ^ { ( L _ { b } ) } \right)\tag{2}
$$

## 2.2 Reinforcement Learning

Reinforcement learning (RL) is a general framework designed for sequential decision-making, typically formulated as a Markov Decision Process (MDP) defined as

$$
\mathcal { M } = ( \mathcal { S } , \mathcal { A } , P , r , \gamma ) , \quad s \in \mathcal { S } , \ a \in \mathcal { A }\tag{3}
$$

where $s$ denotes the state space, A denotes the action space, $P$ denotes the transition dynamics. r is the reward function, and $\gamma \in [ 0 , 1 ]$ is the discount factor. Given a state s, a policy $\pi _ { \phi } ( \boldsymbol { a } \mid \boldsymbol { s } )$ specifies a distribution over the action space. The policy induces a trajectory $\tau$ with probability $p ( \tau )$

$$
\tau = ( s _ { 1 } , a _ { 1 } , s _ { 2 } , a _ { 2 } , \ldots , s _ { T } ) , \quad p ( \tau ) = p ( s _ { 1 } ) \prod _ { k = 1 } ^ { T } \pi _ { \phi } ( a _ { k } \mid s _ { k } ) P ( s _ { k + 1 } \mid s _ { k } , a _ { k } )\tag{4}
$$

The objective of $\mathtt { R L }$ is to learn a policy that maximizes the expected cumulative reward as

$$
\operatorname* { m a x } _ { \pi _ { \phi } } \mathbb { E } _ { \tau \sim p _ { \tau } ( \cdot | \pi _ { \phi } ) } \left[ \sum _ { k = 1 } ^ { T } \gamma ^ { k - 1 } r ( s _ { k } , a _ { k } ) \right] ,\tag{5}
$$

where $p _ { \tau } ( \cdot \mid \pi _ { \phi } )$ denotes the trajectory distribution induced by policy $\pi _ { \phi }$

![](images/ea17f69eb994238b9862fed2cbf5d5670ed8053580481b1707858057bdffbf9c.jpg)  
Figure 2: Overview of CODA for cohort- and drift-aware multimodal clinical reasoning.

## 3 Methodology

## 3.1 CODA Overview

As illustrated in Figure 2, CODA adopts a two-stage formulation to enable cohort-aware clinical reasoning. In the first stage, we learn a hierarchical cohort tree $\tau$ that explicitly partitions the patient population $\mathcal { P }$ into a clinically meaningful cohort set C. In the cohort tree, each internal node parameterizes a feature-based state s and partition action $^ { a , }$ while each leaf node represents a cohort $c \in { \mathcal { C } } .$ . These partition actions are optimized with reinforcement learning, allowing the cohort structure to be shaped not only by clinical heterogeneity but also by its utility for downstream generation. With the learned cohorts in ${ \mathcal { C } } _ { : }$ , we derive a cohort-aware representation for each sample and inject it into the backbone, thereby providing explicit population-level context rather than relying on the model to infer such structure implicitly from its parameters alone. In the second stage, building upon this cohort-level grounding, we maintain dynamically updated cohort prototypes $\mathbf { h } _ { c }$ during training and characterize each sample by its cohort-relative deviation $\Delta ( { \bf x } , c )$ . This cohort-relative deviation captures fine-grained patient-specific variability beyond shared cohort characteristics, enabling the model to reason jointly over stable population patterns and individualized clinical departures. The cohort- and drift-aware framework is encapsulated in the following integrated objective:

$$
\begin{array} { r l } { \mathcal { I } ( \theta , \pi _ { \phi } ) = \displaystyle \sum _ { \mathbf { x } , \mathbf { y } } \bigg [ \underbrace { \mathbb { E } _ { \tau \sim p _ { \tau } ( \cdot | \pi _ { \phi } ) } [ \mathcal { R } ( \mathbf { x } , \tau ) ] + \mathcal { H } ( p _ { \pi _ { \phi } } ( \cdot | \mathbf { x } ) ) } _ { \mathrm { C o h o r t - a w a r e ~ P o p u l a t i o n ~ S t r a t i t i c a t i o n } } } & { + \underbrace { \log p _ { \theta } \big ( \mathbf { y } \mid \mathbf { x } , \Delta ( \mathbf { x } ; c ) \big ) } _ { \mathrm { D r i f t - a w a r e ~ I n d i v i d u a l ~ A d a p t a t i o n } } \bigg ] } \end{array}\tag{6}
$$

where θ denotes the learnable model parameters and $\pi _ { \phi }$ is the cohort-partitioning policy, $\mathcal { R } ( \mathbf { x } , \tau )$ is the trajectory-level cohort discovery reward, $\mathcal { H } ( p _ { \pi _ { \phi } } ( \cdot | \mathbf { \sigma _ { x } } ) )$ ) is the policy entropy for cohort partition exploration.

## 3.2 Cohort-aware Population Stratification

To formalize the context-dependency inherent in clinical heuristics, CODA eschews the conventional assumption of a homogeneous patient distribution. Instead, CODA introduces a hierarchical stratification process that partitions the patient population space P into a set of clinically distinct cohorts $\mathcal { C } .$ This process is governed by two synergistic components, i.e., an RL-based cohort discovery mechanism and a cohort context injection schema that injects cohort-level context into the model.

RL-based Cohort Discovery. We formulate cohort discovery as a hierarchical policy learning problem, where a binary-tree-structured policy $\pi _ { \phi }$ recursively partitions the patient population $\bar { \mathcal P }$ into a set of cohorts C. Formally, we define a cohort tree as a hierarchical binary tree $\dot { \mathcal { T } } = ( V , E )$ where $V = \{ v _ { 1 } , \cdots , v _ { N } \}$ is the set of nodes and $E \subseteq V \times V$ is the set of directed edges representing parent–child relationships. Each node $v \in V$ corresponds to a subset of patients $\bar { \mathcal { P } _ { v } } \subseteq \bar { \mathcal { P } }$ , with the root node representing the entire population. Internal nodes correspond to decision points that further partition the patient subset, while leaf nodes represent terminal cohorts corresponding to clinically homogeneous patient groups. Each node v is further associated with (i) a state representation $s _ { v }$ that summarizes the statistical and structural properties of the corresponding patient subset, and (ii) a partition action $a _ { v }$ induced by the policy, which determines how the node is expanded or terminated.

The cohort tree thereby induces a recursive partition of the population space, where each root-to-leaf path encodes a sequence of feature-based decisions that uniquely determines a cohort.

Specifically, for each sample $\mathbf { x } ,$ we extract a clinical feature $\mathbf { u } = g ( \mathbf { x } )$ , where $g ( \cdot )$ maps serialized EHR records into a vector of statistical features capturing patient demographics, clinical conditions, and laboratory signals. Each node v is represented by a compact state vector:

$$
s _ { v } = [ \mu ( \mathbf { U } _ { \mathcal { P } _ { v } } ) ; \sigma ( \mathbf { U } _ { \mathcal { P } _ { v } } ) ; | \mathcal { P } _ { v } | ; d _ { v } ] ,\tag{7}
$$

where $\mathbf { U } _ { \mathcal { P } _ { v } } = \{ \mathbf { u } _ { i } \} _ { i \in \mathcal { P } _ { \ i } }$ denotes the collection of feature vectors of patients in node v. $\mu ( \cdot )$ and $\sigma ( \cdot )$ denote the mean and standard deviation, respectively. $| \mathcal { P } _ { v } |$ is the number of patients and $d _ { v }$ is the depth of node $v .$

At each node $v ,$ the cohort policy $\pi _ { \phi }$ induces a stochastic action $a _ { v }$ over the patient subset $\mathcal { P } _ { v }$ to specify whether to stop, the selected medical feature, and the corresponding split threshold, where action $a _ { v }$ is defined as:

$$
a _ { v } = ( z _ { v } , j _ { v } , \mathrm { t } _ { v } ) \sim \pi _ { \phi } ( \cdot \mid s _ { v } ) .\tag{8}
$$

where $z _ { v } \in \{ 0 , 1 \}$ is a binary stopping variable, with $z _ { v } = 1$ indicating that v becomes a terminal cohort. $j _ { v }$ is the selected feature, and $t _ { v }$ is the split threshold. When $z _ { v } = 0$ , the node is expanded by partitioning $\mathcal { P } _ { v }$ into two disjoint subsets:

$$
\mathcal { P } _ { v _ { L } } = \{ i \in \mathcal { P } _ { v } ~ | ~ u _ { i , j _ { v } } \leq \mathrm { t } _ { v } \} , ~ \mathcal { P } _ { v _ { R } } = \mathcal { P } _ { v } ~ \backslash ~ \mathcal { P } _ { v _ { L } } ,\tag{9}
$$

where $v _ { L }$ and $v _ { R }$ denote the left and right child nodes, respectively.

The reward function is designed to encourage each split to produce more compact and balanced child cohorts while penalizing unnecessary tree depth:

$$
r _ { v } = \Big ( \boldsymbol { \sigma } ( \mathbf { U } _ { \mathcal { P } _ { v } } ) - \sum _ { k \in \{ L , R \} } \frac { \left| \mathcal { P } _ { v _ { k } } \right| } { \left| \mathcal { P } _ { v } \right| } \boldsymbol { \sigma } ( \mathbf { U } _ { \mathcal { P } _ { v _ { k } } } ) \Big ) + \left( 1 - \frac { \mid \left| \mathcal { P } _ { v _ { L } } \right| - \left| \mathcal { P } _ { v _ { R } } \right| \mid } { \left| \mathcal { P } _ { v } \right| } \right) - \frac { d _ { v } } { d _ { \operatorname* { m a x } } } ,\tag{10}
$$

where $d _ { \mathrm { m a x } }$ is a normalization constant denoting the maximum depth. Here, the first term measures the reduction in intra-cohort variance after splitting, the second term discourages highly imbalanced partitions, and the last term penalizes overly deep trees.

Cohort Context Encoding. For each cohort $c ,$ we derive a representative embedding via a similarityaware aggregation over its members, which is then mapped to a cohort embedding e and injected into the multimodal input sequence for the patients $\mathcal { P } _ { c }$ within cohort c to condition generation:

$$
{ \bf Z } ^ { ( 0 ) } = [ f _ { I  t } ( \mathcal { E } _ { I } ( \mathcal { Z } ) ) ; { \bf e } _ { c } ; \mathcal { E } _ { t } ( \mathcal { X } ) ] , \quad { \bf e } _ { c } = f _ { c } ( \sum _ { i \in \mathcal { P } _ { c } } \frac { \exp ( \cos ( { \bf h } _ { i } , \bar { \bf h } _ { c } ) ) } { \sum _ { j \in \mathcal { P } _ { c } } \exp ( \cos ( { \bf h } _ { j } , \bar { \bf h } _ { c } ) ) } { \bf h } _ { i } ) ,\tag{11}
$$

where $f _ { c }$ is a learnable mapping function, $\mathbf { h } _ { i } = f _ { \mathbf { u } \to \mathbf { h } } ( \mathbf { u } _ { i } )$ is the learned patient representation based on statistical feature $\mathbf { u } _ { i }$ via function $f _ { { \bf u } \to { \bf h } }$ , and $\mathbf { h } _ { c }$ is the centroid of cohort c in the feature space.

To jointly optimize cohort discovery and cohort-aware generation, we define a unified objective that integrates policy learning and downstream generation:

$$
\mathcal { L } _ { \mathrm { c o h o r t } } = \mathbb { E } _ { \tau \sim p _ { \tau } ( \cdot | \pi _ { \phi } ) } \left[ \sum _ { v \in V } \left( - \log \pi _ { \phi } ( a _ { v } \mid \mathbf { s } _ { v } ) R _ { v } - \mathcal { H } ( \pi _ { \phi } ( \cdot \mid \mathbf { s } _ { v } ) ) \right) \right] + \lambda _ { g } \mathcal { L } _ { \mathrm { g e n } } ,\tag{12}
$$

where $R _ { v } = r _ { v } - \mathcal { L } _ { \mathrm { g e n } }$ denotes the generation-aware reward and $\mathcal { H } ( \pi _ { \phi } ( \cdot ) )$ is the entropy of the policy that encourages exploration. $\mathcal { L } _ { \mathrm { g e n } }$ is the downstream generation loss with $\lambda _ { g }$ balancing policy learning and generation performance.

In this way, a cohort tree $\tau$ and cohort embeddings $\mathbf { e } _ { c }$ are jointly learned. The resulting structure captures population-level heterogeneity via hierarchical partitioning, while the learned embeddings encode cohort-specific context for clinical reasoning. Through the proposed generation-aware objective, cohort discovery is directly aligned with multimodal reasoning, enabling the model to move beyond purely structure-driven partitioning toward reasoning-aware population stratification.

## 3.3 Drift-aware Individual Adaptation

While the cohort tree $\tau$ provides population-level structure, clinical reasoning further requires capturing patient-specific deviations from cohort prototypes. To this end, we introduce a drift-aware individual adaptation mechanism that explicitly models the cohort-relative deviation of each sample and integrates it into the generation process.

Cohort Prototype Representation. For each cohort c, we maintain a learnable prototype representation $\mathbf { h } _ { c }$ , which summarizes the central clinical characteristics of the cohort and is initialized using the previously learned cohort embedding $\mathbf { e } _ { c }$ . During training, these prototypes are dynamically updated in an online manner based on the incoming samples. Specifically, for a sample x of patient p assigned to cohort c with feature representation $\mathbf { h } _ { p } ,$ , we update the corresponding prototype using an exponential moving average mechanism:

$$
\mathbf { h } _ { c } \gets ( 1 - \beta w ( p , c ) ) \mathbf { h } _ { c } + \beta w ( p , c ) \frac { \mathbf { h } _ { p } } { \| \mathbf { h } _ { p } \| _ { 2 } } ,\tag{13}
$$

where $w ( p , c ) \in [ 0 , 1 ]$ denotes the assignment confidence of patient $p$ to cohort $c ,$ and the feature is ℓ -normalized to stabilize prototype updates. $\mathbf { h } _ { c }$ replaces $\mathbf { e } _ { c }$ in $\mathbf { Z } ^ { ( 0 ) }$ and serves as a conditioning signal in the multimodal input sequence, analogous to $\mathbf { e } _ { c }$

Drift-aware Deviation Modeling. While the cohort tree $\tau$ captures population-level structure, clinical reasoning further requires modeling patient-specific deviations from cohort prototypes. To this end, we introduce a unified drift-aware deviation mechanism that explicitly characterizes individual departures from cohort-level patterns and injects them into the generation process.

Given a sample x with feature representation $\mathbf { h } _ { p }$ and assigned cohort $c ,$ we compute its cohort-relative deviation as $\Delta ( \mathbf { x } , c ) = \mathbf { h } _ { p } - \mathbf { h } _ { c } .$ , which captures patient-specific variation beyond the cohort prototype. We first transform this deviation into a compact drift-aware fusion code:

$$
\mathbf { d } _ { \mathbf { x } } = \mathrm { L N } ( \mathbf { W } _ { d } [ \Delta ( \mathbf { x } , c ) ; \mathbf { h } _ { p } ; \mathbf { h } _ { c } ; \Delta ( \mathbf { x } , c ) \odot \mathbf { h } _ { c } ] + \mathbf { b } _ { d } ) ,\tag{14}
$$

where $[ \cdot ; \cdot ]$ denotes concatenation, $\odot$ denotes element-wise interaction, $\operatorname { L N } ( \cdot )$ denotes layer normalization, $\mathbf { W } _ { d }$ is a learnable weight matrix and ${ \mathbf { b } } _ { d }$ is a learnable bias.

The drift code is then used to modulate the hidden representation at transformer block ℓ through a residual adapter. Compared to a simple additive bias, which applies a uniform shift, our formulation introduces a multiplicative interaction between the deviation and the hidden representation. This enables context-dependent, feature-wise modulation, allowing the model to adapt differently across features and tokens.

$$
\begin{array} { r } { \tilde { \mathbf { Z } } ^ { ( \ell ) } = \mathbf { Z } ^ { ( \ell ) } + \mathbf { W } _ { \ell } ^ { o } \left( ( 1 + \operatorname { t a n h } ( \mathbf { W } _ { \ell } ^ { \gamma } \mathbf { d } _ { \mathbf { x } } ) ) \odot \operatorname { G E L U } ( \mathbf { W } _ { \ell } ^ { \alpha } \mathbf { Z } ^ { ( \ell ) } ) + \mathbf { W } _ { \ell } ^ { \beta } \mathbf { d } _ { \mathbf { x } } \right) , } \end{array}\tag{15}
$$

where $\mathbf { W } _ { \ell } ^ { o }$ is a learnable output projection that maps the adapted signal back to the hidden space. The term $\mathrm { G E L U } ( \mathbf { W } _ { \ell } ^ { \alpha } \mathbf { Z } ^ { ( \ell ) } )$ corresponds to a nonlinear transformation of the transformer output, capturing content-dependent features. The modulation term $1 + \operatorname { t a n h } ( \mathbf { W } _ { \ell } ^ { \gamma } \mathbf { d } _ { \mathbf { x } } )$ acts as a feature-wise scaling factor conditioned on the cohort-relative deviation, where tanh(·) ensures bounded and stable modulation. Finally, $\mathbf { W } _ { \ell } ^ { \beta } \mathbf { d } _ { \mathbf { x } }$ introduces an additive bias term that shifts the representation based on the deviation signal.

Cohort-aware Generation. The adapted hidden representation $\tilde { \mathbf { Z } } ^ { ( L _ { b } ) }$ at the final transformer block $L _ { b }$ is then used for autoregressive generation:

$$
p _ { \theta } ( \mathbf { y } \mid \mathbf { x } ) = \prod _ { i = 1 } ^ { | \mathbf { y } | } p _ { \theta } \left( y _ { i } \mid y _ { < i } , { \tilde { \mathbf { Z } } } ^ { ( L _ { b } ) } \right) .\tag{16}
$$

To facilitate individual deviation modeling and enforce inter-cohort separability, we introduce a drift loss that explicitly encodes both intra-cohort consistency and inter-cohort discrimination by minimizing the deviation from the assigned cohort prototype while imposing a margin-based constraint against alternative cohorts.

$$
\mathcal { L } _ { \mathrm { d r i f t } } = \mathcal { L } _ { \mathrm { g e n } } + \lambda _ { d } ( \| \boldsymbol { \Delta } ( \mathbf { x } , c ) \| _ { 2 } ^ { 2 } + \operatorname* { m a x } ( 0 , m - \frac { \mathbf { h } _ { p } ^ { \top } \mathbf { h } _ { c } } { \| \mathbf { h } _ { p } \| \| \mathbf { h } _ { c } \| } + \frac { \mathbf { h } _ { p } ^ { \top } \mathbf { h } _ { c ^ { \prime } } } { \| \mathbf { h } _ { p } \| \| \mathbf { h } _ { c ^ { \prime } } \| } ) )\tag{17}
$$

Table 1: Comparison of CODA with baselines under Closed-QA settings.
<table><tr><td rowspan="3">Domain</td><td rowspan="3">Model</td><td colspan="8">ID</td><td colspan="4">OOD</td><td rowspan="2" colspan="2">Avg.</td></tr><tr><td colspan="2">Admissions</td><td colspan="2">IE</td><td colspan="2">POE</td><td colspan="2">Services</td><td colspan="2">Transfers</td><td colspan="2">Obesity</td></tr><tr><td>ACC</td><td>MCC</td><td>ACC</td><td>MCC</td><td>ACC</td><td>MCC</td><td>ACC</td><td>MCC</td><td>ACC</td><td>MCC</td><td>ACC</td><td>MCC</td><td>ACC</td><td>MCC</td></tr><tr><td rowspan="9">General Domain</td><td>LLaVA 1.5-7B</td><td>40.00</td><td>20.06</td><td>19.44</td><td>-7.48</td><td>57.00</td><td>42.82</td><td>63.59</td><td>51.45</td><td>60.60</td><td>47.61</td><td>61.90</td><td>47.97</td><td>50.42 33.74</td></tr><tr><td>LLaMA 3-11B</td><td>32.80</td><td>10.74</td><td>26.52</td><td>4.85 35.40</td><td>14.55</td><td>43.64</td><td>25.32</td><td>37.00</td><td>16.29</td><td>54.12</td><td>39.58</td><td>38.25</td><td>18.56</td></tr><tr><td>InternVL 3-8B</td><td>28.20</td><td>5.06</td><td>18.94</td><td>-6.52</td><td>46.80 29.13</td><td>42.39</td><td>23.35</td><td>27.00</td><td>2.58</td><td>70.92</td><td>59.61</td><td>39.04</td><td>18.87</td></tr><tr><td>Qwen 3-VL-8B</td><td>36.40</td><td>15.35</td><td>38.38</td><td>17.64</td><td>40.40 20.49</td><td>60.60</td><td>48.07</td><td>53.80</td><td>38.63</td><td>69.67</td><td>58.10</td><td>49.88</td><td>33.05</td></tr><tr><td>Gemma 3-12B</td><td>18.80</td><td>-8.06</td><td>19.95</td><td>-6.83</td><td>35.20 13.55</td><td>58.85</td><td>45.20</td><td>50.00</td><td>33.42</td><td>71.07</td><td>60.24</td><td>42.31</td><td>22.92</td></tr><tr><td>Grok 4.2</td><td>23.60</td><td>-1.58</td><td>31.57</td><td>8.56</td><td>35.20 13.43</td><td>25.19</td><td>0.17</td><td>40.60</td><td>20.45</td><td>56.92</td><td>39.69</td><td>35.51</td><td>13.45</td></tr><tr><td>Claude 4.5 Haiku</td><td>35.60</td><td>14.09</td><td>29.29</td><td>6.51</td><td>40.40</td><td>20.78 42.64</td><td>23.54</td><td>42.60</td><td>23.51</td><td>47.90</td><td>32.61</td><td>39.74</td><td>20.17</td></tr><tr><td>GPT 5.4 Mini</td><td>34.60</td><td>12.60</td><td>42.39</td><td>23.08 44.40</td><td>26.02</td><td>43.14</td><td>24.17</td><td>48.40</td><td>31.14</td><td>59.72</td><td>45.92</td><td>45.44</td><td>27.15</td></tr><tr><td>LLaVA-Med v1.5-7B</td><td>21.00</td><td>3.83</td><td>25.00</td><td>2.21</td><td>58.00</td><td>45.01</td><td>44.89</td><td>27.44</td><td>34.40</td><td>17.29</td><td>58.48</td><td>45.85</td><td>40.29</td><td>23.60</td></tr><tr><td rowspan="9">Medical Domain</td><td>Med-Flamingo-7B</td><td>17.40</td><td>-3.86</td><td>8.08</td><td>-3.11</td><td>21.80</td><td>3.18</td><td>25.69</td><td>4.28</td><td>27.80 1.84</td><td>17.42</td><td>3.60</td><td>19.70</td><td>0.99</td></tr><tr><td>HuatuoGPT-Vision-7B</td><td>44.00</td><td>26.00</td><td>40.66</td><td>20.78</td><td>48.40</td><td>32.11</td><td>60.10 46.94</td><td>57.80</td><td>44.60</td><td>67.96</td><td>56.04</td><td>53.15</td><td>37.74</td></tr><tr><td>HealthGPT-14B</td><td>39.60</td><td>19.53</td><td>25.51</td><td>1.53 38.20</td><td>18.04</td><td>33.42</td><td>12.10</td><td>42.00</td><td>23.55</td><td>55.68</td><td>42.71</td><td>39.07</td><td>19.58</td></tr><tr><td>Lingshu-7B</td><td>44.40</td><td>26.26</td><td>45.45</td><td>28.06 50.60</td><td>34.10</td><td>78.55</td><td>71.56</td><td>57.00</td><td>44.26</td><td>70.63</td><td>60.52</td><td>57.77</td><td>44.13</td></tr><tr><td>Hulu-Med-7B</td><td>43.60</td><td>25.12</td><td>26.01</td><td>2.39</td><td>53.20 37.75</td><td>60.60</td><td>47.84</td><td>36.00</td><td>14.56</td><td>65.47</td><td>53.56</td><td>47.48</td><td>30.20</td></tr><tr><td>CODA-Qwen3-8B</td><td>76.60</td><td>68.77</td><td>92.42</td><td>89.93</td><td>82.80 77.05</td><td>93.52</td><td>91.39</td><td>75.20</td><td>66.89</td><td>72.32</td><td>61.28</td><td>82.14</td><td>75.88</td></tr><tr><td>CODA-Llama3-8B</td><td>76.60</td><td>68.77</td><td>92.93</td><td>90.59</td><td>82.00</td><td>76.00</td><td>92.27</td><td>89.71 74.80</td><td>66.31</td><td>72.90</td><td>61.71</td><td>81.92</td><td>75.52</td></tr><tr><td></td><td>72.80</td><td>63.75</td><td>92.17</td><td>89.59</td><td>82.40</td><td>76.53</td><td>85.29 80.55</td><td>70.20</td><td>60.61</td><td>72.16</td><td>60.98</td><td>79.17</td><td>72.00</td></tr><tr><td>CODA-Qwen3-8B w/o LoRA</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

where $m > 0$ is the cohort margin, $c ^ { \prime }$ denotes the cohort with the second highest confidence to $p ,$ and $\lambda _ { d }$ controls the trade-off between generation performance and cohort-relative deviation modeling.

Unlike standard generation, where the model relies solely on input features, our formulation conditions the generation process on both the cohort prototype $\mathbf { h } _ { c }$ and the cohort-relative deviation $\Delta ( { \bf x } , c )$ This allows the model to first anchor predictions to population-level patterns, and then refine them based on patient-specific variations. Overall, this design enables a unified integration of cohort-level regularities and individual-level deviations, leading to context-aware and fine-grained generation.

## 4 Experiments

## 4.1 Experimental Setup

Datasets. We conduct extensive experiments on four real-world EHR benchmarks, i.e., EHR-Bench [18], DIYHealth-Bench [19], MIMIC-CXR [20], and PadChest [21]. Specifically, the admissions, services, ingredient events (IE), and provider order entry (POE) tasks from EHR-Bench, together with MIMIC-CXR, are treated as in-distribution (ID) datasets, while the Transfers task from EHR-Bench, Obesity from DIYHealth-Bench, and PadChest are used as out-of-distribution (OOD) datasets. The model is trained on the ID training datasets and evaluated on both ID and OOD settings. Through these datasets, we aim to provide a comprehensive assessment of the effectiveness and generalizability of CODA in answering practical medical questions and generating clinical reports.

Baselines. We evaluate CODA against a broad set of baselines, including state-of-the-art generalist models, (e.g., LLaVA 1.5 [22], Llama 3 [23], InternVL 3 [24], Qwen3 VL [25], Gemma 3 [26], Grok [27], Claude 4.5 Haiku [28] and GPT 5.4 Mini [29]), as well as medical-specific models (e.g., LLaVA-Med v1.5 [8], Med-Flamingo [13], HuatuoGPT-Vision [14], HealthGPT [9], Lingshu [30], and Hulu-Med [31]. All models are evaluated under closed-form multiple-choice question answering (Closed-QA), open-ended question reasoning (Open-QA), and radiology report generation.

Training Strategy. Our training pipeline consists of three stages. (i) Cross-Modal Alignment. To adapt the visual and textual alignment to the medical domain, we train the vision projector of the backbone using a mixture of medical VQA and QA datasets, including MedQuAD [32], PubMedVision [33], SLAKE [34], and VQA-RAD [35]. (ii) Cohort-aware Population Stratification. With the backbone frozen, we train only the RL-related modules on a subset (15%) of the indistribution data to learn a hierarchical cohort structure. (iii) Drift-aware Individual Adaptation. We fine-tune only the LoRA modules together with deviation-aware components to model patient-specific variations conditioned on cohort context.

Evaluation Metrics. For Closed-QA, we adopt Accuracy (ACC) and Matthews Correlation Coefficient (MCC) [36]. ACC quantifies the proportion of correctly predicted answers, reflecting exact match performance. MCC provides a balanced evaluation by capturing the agreement between predicted and true labels across all classes. For Open-QA, we use BLEU-1 [37] and F1-BioBERT (F1-Bio) [38]. BLEU-1 evaluates lexical overlap based on unigram precision, serving as a signal of surface-level correctness. F1-Bio measures semantic consistency by computing similarity in a contextual embedding space derived from BioBERT [39], enabling robust evaluation beyond surface level matching. For report generation, we employ F1-RadGraph (F1-Rad) [40] and F1-Bio. F1-Rad assesses the correctness of clinical entities and their relations by comparing structured representations extracted from RadGraph [40], capturing clinical fidelity at a fine-grained level. Please refer to Appendix C for more details about the datasets, model implementation, and evaluation metrics.

Table 2: Comparison of CODA with baselines under Open-QA and report generation settings.
<table><tr><td rowspan="3" colspan="2">Domain</td><td colspan="10"></td><td colspan="6">OOD</td><td colspan="4">Avg.</td></tr><tr><td></td><td colspan="2">Admissions</td><td>IE</td><td></td><td>POE</td><td></td><td>Services</td><td></td><td>MIMIC-CXR</td><td></td><td>Transfers</td><td></td><td>Obesity</td><td></td><td>PadChest</td><td>Open-QA</td><td></td><td></td><td>Report</td></tr><tr><td>Fl-Bio</td><td>BLEU-1</td><td>F1-Bio</td><td>BLEU-1</td><td>Fl-Bio</td><td>BLEU-1</td><td>F1-Bio</td><td>BLEU-1</td><td>F1-Rad</td><td>F1-Bio</td><td>F1-Bio</td><td>BLEU-1</td><td>F1-Bio</td><td>BLEU-1</td><td>F1-Rad</td><td>Fl-Bio</td><td>Fl-Bio</td><td>BLEU-1</td><td>F1-Rad</td><td>Fl-Bio</td></tr><tr><td rowspan="8">General Domain</td><td>LLaVA 1.5-7B LLaMA 3-11B</td><td></td><td>36.91</td><td>75.16</td><td>29.21</td><td>75.66</td><td>33.51</td><td>75.08</td><td>34.12</td><td>1.79 68.12</td><td>75.79</td><td>41.27</td><td>72.23</td><td>9.99</td><td>11.75</td><td>68.78</td><td>74.98</td><td>30.83</td><td></td><td>6.77 68.45</td></tr><tr><td></td><td>75.95 74.68</td><td>35.65</td><td>72.73</td><td>24.38</td><td>74.83</td><td>36.18</td><td>74.97</td><td>34.34</td><td>4.38 65.01</td><td>74.88</td><td>38.01</td><td>70.51</td><td>8.68</td><td>7.54</td><td>62.28</td><td>73.77</td><td>29.54</td><td></td><td>63.64</td></tr><tr><td>InternVL 3-8B</td><td>72.20</td><td>24.43</td><td>73.91</td><td>25.01</td><td>72.61</td><td>23.90</td><td>72.39</td><td>23.83 6.38</td><td>61.62</td><td>73.96</td><td>26.16</td><td>67.90</td><td>5.34</td><td>9.32</td><td>61.45</td><td>72.16</td><td>21.45</td><td>5.96</td><td>61.54</td></tr><tr><td>Qwen 3-VL-8B</td><td>68.82</td><td>18.31</td><td>68.77</td><td>17.14</td><td>68.47</td><td>16.82</td><td>68.07 17.09</td><td>8.23</td><td>63.63</td><td>72.23</td><td>26.96</td><td>68.95</td><td>6.40</td><td>8.15</td><td>66.42</td><td>69.22</td><td>17.12</td><td>7.85 8.19</td><td>65.03</td></tr><tr><td>Gemma 3-12B</td><td>72.33</td><td>26.56</td><td>71.86</td><td>25.30</td><td>73.57</td><td>30.07</td><td>74.57 33.98</td><td>10.27</td><td>68.75</td><td>74.72</td><td>25.58</td><td>66.93</td><td>4.07</td><td>12.31</td><td>65.35</td><td>72.33</td><td>24.26</td><td>11.29</td><td>67.05</td></tr><tr><td>Grok 4.2</td><td>69.61</td><td>19.71</td><td>68.97</td><td>19.71</td><td>68.61</td><td>17.82</td><td>69.65 19.86</td><td>13.52</td><td>72.62</td><td>71.21</td><td>25.32</td><td>67.63</td><td>5.29</td><td>12.74</td><td>70.31</td><td>69.28</td><td>17.95</td><td>13.13</td><td>71.46</td></tr><tr><td>Claude 4.5 Haiku</td><td>65.12</td><td>15.36</td><td>66.50</td><td>17.24</td><td>65.31</td><td>15.38</td><td>65.57 15.18</td><td>10.54</td><td>67.75</td><td>67.78</td><td>20.84</td><td>64.80</td><td>7.67</td><td>8.54</td><td>61.97</td><td>65.85</td><td>15.28</td><td>9.54</td><td>64.86</td></tr><tr><td>GPT 5.4 Mini</td><td>68.86</td><td>23.15</td><td>65.55</td><td>26.80 69.59</td><td></td><td>23.77 68.84</td><td>22.84</td><td>15.88</td><td>72.30</td><td>71.24</td><td>30.65</td><td>65.78</td><td>7.07</td><td>10.35</td><td>68.39</td><td>68.31</td><td>22.38</td><td>13.11</td><td>70.34</td></tr><tr><td>LLaVA-Med v1.5-7B Med-Flamingo-7B</td><td></td><td></td><td>75.27</td><td>6.30</td><td>75.59</td><td>10.09</td><td>74.44</td><td>7.72</td><td>0.80</td><td>67.11</td><td>76.85</td><td>11.76</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="7">Medical Domain</td><td></td><td>73.85 58.64</td><td>8.71 11.86</td><td>55.51</td><td>11.36</td><td>58.68</td><td>13.04</td><td>58.61</td><td>11.56 0.14</td><td>40.54</td><td>62.24</td><td>9.51</td><td>74.94 57.95</td><td>15.14 4.91</td><td>3.85 2.63</td><td>66.79 52.08</td><td>75.16 58.61</td><td>9.95 10.37</td><td>2.33 1.38</td><td>66.95 46.31</td></tr><tr><td>HuatuoGPT-Vision-7B</td><td>74.33</td><td>33.93</td><td>74.51</td><td>33.42</td><td>74.01</td><td>32.45</td><td>74.09</td><td>32.50 7.77</td><td>70.45</td><td>74.69</td><td>33.66</td><td>72.56</td><td>10.46</td><td>7.60</td><td>64.60</td><td>74.03</td><td>29.40</td><td>7.68</td><td>67.53</td></tr><tr><td>HealthGPT-14B</td><td>69.91</td><td>29.27</td><td>70.67</td><td>27.16</td><td>71.03</td><td>26.87</td><td>72.15</td><td>28.22 9.05</td><td>73.04</td><td>66.58</td><td>27.69</td><td>66.49</td><td>7.49</td><td>2.03</td><td>68.48</td><td>69.47</td><td>24.45</td><td>5.54</td><td>70.76</td></tr><tr><td>Lingshu-7B</td><td>76.17</td><td>38.17</td><td>74.68</td><td>35.01</td><td>75.88</td><td>33.47</td><td>76.26</td><td>33.23</td><td>16.18 78.19</td><td>76.11</td><td>35.19</td><td>72.95</td><td>11.16</td><td>1.23</td><td>68.85</td><td>75.34</td><td>31.04</td><td>8.71</td><td>73.52</td></tr><tr><td>Hulu-Med-7B</td><td>76.20</td><td>34.79</td><td>74.43</td><td>14.55</td><td>75.47</td><td>30.74</td><td>75.62</td><td>27.64</td><td>16.31 75.84</td><td>76.06</td><td>26.06</td><td>73.79</td><td>12.45</td><td>5.52</td><td>69.15</td><td>75.26</td><td>24.37</td><td>10.91</td><td>72.49</td></tr><tr><td>CODA-Qwen3-8B</td><td>80.82</td><td>52.45 46.38</td><td>82.21</td><td>51.69</td><td>81.02</td><td>51.97</td><td>80.21 49.19</td><td>17.49</td><td>79.03 78.56</td><td>79.57 77.49</td><td>46.42 46.19</td><td>75.37 74.95</td><td>15.65 14.52</td><td>25.52 24.09</td><td>73.93 71.77</td><td>79.87 77.22</td><td>44.56 40.63</td><td>21.51 20.70</td><td>76.48</td></tr><tr><td>CODA-Llama3-8B</td><td>78.49</td><td>50.50</td><td>78.74 81.06</td><td>46.11 48.43</td><td>77.30 80.47</td><td>46.05 51.64</td><td>76.35 79.54</td><td>44.51 48.37</td><td>17.31 16.76</td><td></td><td></td><td></td><td></td><td>23.32</td><td></td><td></td><td>43.04</td><td></td><td>75.17</td></tr><tr><td></td><td>CODA-Qwen3-8B w/o LoRA</td><td>80.29</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>78.08</td><td>78.58</td><td>44.45</td><td>74.15</td><td>14.82</td><td></td><td>71.64</td><td>79.02</td><td></td><td>20.04</td><td>74.86</td></tr></table>

Table 3: Ablation study results of CODA. "w/o Cohort" removes the Cohort Context Encoding, while "w/o Deviation" removes Drift-aware Deviation Modeling.
<table><tr><td rowspan="3">Model</td><td colspan="6">ID</td><td colspan="6">OOD</td></tr><tr><td colspan="2">Closed-QA</td><td colspan="2">Open-QA</td><td colspan="2">Report</td><td colspan="2">Closed-QA</td><td colspan="2">Open-QA</td><td colspan="2">Report</td></tr><tr><td>ACC</td><td>MCC</td><td>F1-Bio</td><td>BLEU-1</td><td>F1-Rad</td><td>F1-Bio</td><td>ACC</td><td>MCC</td><td>F1-Bio</td><td>BLEU-1</td><td>F1-Rad</td><td>F1-Bio</td></tr><tr><td>Vanilla Backbone</td><td>43.95</td><td>25.39</td><td>68.53</td><td>17.34</td><td>8.23</td><td>63.63</td><td>61.74</td><td>48.36</td><td>70.59</td><td>16.68</td><td>8.15</td><td>66.42</td></tr><tr><td>w/o Cohort</td><td>85.58</td><td>80.69</td><td>80.11</td><td>49.18</td><td>15.73</td><td>78.58</td><td>72.56</td><td>62.19</td><td>76.82</td><td>28.83</td><td>24.60</td><td>72.51</td></tr><tr><td>w/o Deviation</td><td>85.56</td><td>80.73</td><td>80.10</td><td>48.86</td><td>16.33</td><td>78.94</td><td>73.05</td><td>62.95</td><td>76.65</td><td>28.82</td><td>25.19</td><td>72.41</td></tr><tr><td>CODA</td><td>86.34</td><td>81.78</td><td>81.06</td><td>51.32</td><td>17.49</td><td>79.03</td><td>73.76</td><td>64.08</td><td>77.47</td><td>31.03</td><td>25.52</td><td>73.93</td></tr></table>

## 4.2 Main Results

The experimental results for Closed-QA, Open-QA, and report generation are presented in Tables 1 and 2. We consider three variants: (i) CODA with a Qwen3 backbone, (ii) CODA with an LLaMA3 backbone, and (iii) CODA with a Qwen3 backbone but without LoRA. We summarize the key findings as follows. (i) State-of-the-art performance. Three variants of CODA consistently outperform all baselines across all tasks, yielding average gains of 42.18% in ACC for Closed-QA, 6.01% in F1-Bio for Open-QA, and 63.82% in F1-Rad for report generation. These results demonstrate the effectiveness of CODA and suggest that explicitly modeling cohort structure and cohort drift provides a principled mechanism for capturing both population-level patterns and individual-level variations, resulting in consistently superior performance. (ii) Out-of-Distribution Generalization. Despite being trained solely on ID data, CODA generalizes well to OOD settings, as evidenced by its strong performance on the Obesity dataset, which represents a home-care scenario with limited EHR availability, as well as on PadChest, where the scale of EHR data is substantially smaller. These results indicate that CODA is robust to both data scarcity and domain shift, maintaining reliable performance across diverse clinical environments. (iii) Robustness across backbones. CODA consistently improves performance across diverse backbone architectures (e.g., Qwen3-VL and LLaMA-based models), demonstrating that the proposed cohort-aware mechanism is model-agnostic and broadly applicable. (iv) Limitations of existing Med-LVLMs. Existing Med-LVLMs are typically trained on large-scale corpora and a diverse set of tasks, largely mirroring general LVLM training paradigms with domain-specific data substitution. However, such models often overlook clinically meaningful reasoning processes and fail to incorporate domain-specific inductive biases or mechanisms. As a result, when data scale is no longer the primary bottleneck, their performance remains constrained by a lack of clinically grounded modeling.

## 4.3 Ablation Study and In-Depth Analysis

Component-wise Ablation Study. To assess the contributions of key components in CODA, we conduct an ablation study, with results reported in Table 3 and full results in Appendix E.1. Removing the Cohort Context Encoding leads to consistent performance degradation in both in-distribution and out-of-distribution settings, particularly for Closed-QA and report generation tasks. This suggests that explicitly modeling cohort structure is essential for capturing population-level regularities. Removing deviation modeling results in additional performance drops, especially on Open-QA tasks, indicating that cohort-level representations alone are insufficient to capture fine-grained, patientspecific variations required for accurate clinical reasoning. Overall, the full model (CODA) achieves the best performance across all settings, demonstrating that the joint modeling of cohort structure and individual-level deviations is critical for robust generalization.

![](images/1cdd857963dafb5b74b14fdb0417e7053d64f0aee2dc45cd7a0252ca34e2aef6.jpg)  
(a) Results of Closed-QA.

![](images/5ac1e4fed2c50a5ff31012d1007836aa4748517a15ef7b9537c2f59bac643884.jpg)  
(b) Results of Open-QA.

![](images/cedc6eb2584c3b4ac6b3f88f0d69679207e2c32dda0147ff1fa673e26724a711.jpg)  
(c) Results of Report Generation.

Figure 3: Comparisons with fine-tuned baselines.  
Table 4: Efficiency comparison of CODA against different baselines.
<table><tr><td>Model</td><td>Tokens (tokens/sample)</td><td>Latency (s/sample)</td><td>Token Throughput (tokens/s)</td><td>Peak GPU Mem. (GB)</td><td>FLOPs (G/token)</td></tr><tr><td>LLaVA 1.5-7B</td><td>187.31</td><td>8.62</td><td>21.73</td><td>13.27</td><td>13.60</td></tr><tr><td>LLaVA-Med v1.5-7B</td><td>69.56</td><td>3.21</td><td>21.67</td><td>14.21</td><td>14.60</td></tr><tr><td>Hulu-Med-7B</td><td>72.75</td><td>5.79</td><td>28.95</td><td>15.65</td><td>14.40</td></tr><tr><td>Lingshu-7B</td><td>167.62</td><td>5.79</td><td>25.18</td><td>15.65</td><td>14.40</td></tr><tr><td>InternVL 3-8B</td><td>417.55</td><td>19.53</td><td>21.38</td><td>15.00</td><td>14.80</td></tr><tr><td>Qwen 3-VL-8B</td><td>545.65</td><td>23.15</td><td>23.57</td><td>16.47</td><td>15.03</td></tr><tr><td>CODA-8B</td><td>160.51</td><td>7.95</td><td>20.19</td><td>17.25</td><td>15.69</td></tr></table>

Comparison with Fine-tuned Baselines. To fully validate the effectiveness of CODA, we compare CODA-Qwen3-8B with two representative baselines, Lingshu-7B and InternVL 3-8B, with results presented in Figure 3 and full results in Appendix E.2. CODA consistently achieves the best performance across all tasks and metrics. These results demonstrate that CODA not only improves predictive accuracy but also enhances semantic consistency and clinical fidelity, highlighting the effectiveness of its cohort-aware and deviation-aware modeling in complex medical reasoning tasks. Interpretation Study. We visualize the learned patient representations of CODA on OOD tasks using t-SNE [41]. For clarity, we present results from the

![](images/8ed7f22e4c3c965066ae466adf17b8edcb22bdc1fb0e955529cc272ea7407be0.jpg)  
Figure 4: Visualization of embeddings within different cohorts.

top-10 largest cohorts. As shown in Figure 4, samples from the same cohort form compact clusters, while different cohorts are largely well separated, indicating intra-cohort consistency and inter-cohort separability. We also observe mild overlap among cohorts 27, 28, and 29, which can be attributed to their shared ancestor nodes in the cohort hierarchy, leading to partially aligned representations. Overall, the learned embeddings exhibit coherent and discriminative geometric structures, where samples are grouped according to shared population-level patterns, suggesting that cohort semantics are effectively captured in the embedding space.

Efficiency Study. To evaluate the computational efficiency of CODA, we compare it with representative general and medical baselines. The results are summarized in Table 4. Despite introducing cohort-aware population stratification and drift-aware individual adaptation, CODA maintains competitive inference efficiency while achieving substantially stronger clinical reasoning performance. Specifically, CODA achieves 7.95 s/sample and 20.19 tokens/s throughput, remaining comparable to widely used LLaVA-based baselines while exhibiting substantially lower sample latency than models with higher output complexity, such as InternVL3-8B and Qwen3-VL-8B. In addition, CODA maintains a practical memory footprint for standard single-GPU deployment, while the additional FLOPs/token introduced by cohort-aware conditioning remain modest. Overall, the results demonstrate that CODA achieves a favorable effectiveness–efficiency trade-off, enabling enhanced population-grounded clinical reasoning without sacrificing practical inference efficiency.

## 4.4 Expert Review

To assess the practical significance of CODA, we conduct an expert review involving both GPT-5.4 and human experts. Specifically, we randomly sample 250 Open-QA instances from the OOD Transfers task and 250 report generation instances from the OOD PadChest task. We compare against five strong baselines, including Grok4.2, Hulu-Med, Qwen3 VL, Lingshu, and LLaVA 1.5.

For each sample, experts are asked to select their first preference, i.e., the response that is most accurate and clinically reliable, from six candidate answers. The results are presented in Figure 5. CODA is consistently chosen as the first preference by both human experts and GPT-5.4, demonstrating its superior clinical utility and robustness. Among the baselines, Grok-

![](images/1ff9c777b57dc58fd61b2aafd8a69e38045dd77b7ad4da5985de890b94adea50.jpg)  
(a) Results of human review on the first preference.

![](images/e21a4e097bd18525e2825bfa211864bc007e749c37e272ff3512e11f68d8c1fa.jpg)  
(b) Results of GPT-5.4 review on the first preference.  
Figure 5: Expert review results on the first preference.

4.2 and Lingshu are ranked as the second-best models by human experts and GPT-5.4, respectively. Despite these variations, all baselines remain substantially behind CODA, indicating that the proposed cohort- and drift-aware mechanisms provide consistent and reliable improvements.

We provide additional in-depth analyses on few-shot settings, hyperparameter sensitivity, and case studies in Appendices E.3, E.4, and F, further supporting the generality and reliability of CODA.

## 5 Conclusions

Generative AI has significantly advanced AI for healthcare. However, current medical foundation models largely overlook the interplay between population-level cohort patterns and patient-specific variation that underlies real-world clinical reasoning. In this paper, we propose CODA, a cohortand drift-aware medical foundation model that explicitly discovers patient cohorts and integrates them into the reasoning pipeline. CODA introduces cohort-aware population stratification and drift-aware individual adaptation, providing a principled and effective framework for enhancing medical reasoning. Extensive experiments demonstrate that CODA consistently outperforms both general-purpose and medical-specific baselines across diverse clinical tasks, including closed-QA, open-QA, and report generation. Despite its effectiveness, CODA currently leverages structured feature representations and summary statistics for cohort construction, which may introduce additional preprocessing overhead in certain scenarios. Developing more automated and end-to-end approaches for cohort discovery and integration remains an important direction for future work.

## References

[1] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in Neural Information Processing Systems, 33:1877–1901, 2020.

[2] OpenAI. Gpt-4 technical report, 2023.

[3] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

[4] Gang Chen, Changshuo Liu, Gene Anne Ooi, Marcus Tan, Zhongle Xie, Jianwei Yin, James Wei Luen Yip, Wenqiao Zhang, Jiaqi Zhu, and Beng Chin Ooi. Generative ai for healthcare: Fundamentals, challenges, and perspectives. arXiv preprint arXiv:2510.24551, 2025.

[5] Jiarui Ye and Hao Tang. Multimodal large language models for medicine: A comprehensive survey. arXiv preprint arXiv:2504.21051, 2025.

[6] Béria Chingnabé Kalpélbé, Angel Gabriel Adaambiik, and Wei Peng. Vision language models in medicine. arXiv preprint arXiv:2503.01863, 2025.

[7] Jiaqi Zhu, Naili Xing, Hexiang Pan, Haotian Gao, Jianwei Yin, Xiaokui Xiao, and Beng Chin Ooi. Logictree-rag: Logic tree-guided retrieval-augmented generation for long-form patent drafting. arXiv preprint arXiv:2609.30943, 2026.

[8] Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. Llava-med: Training a large language-and-vision assistant for biomedicine in one day. Advances in Neural Information Processing Systems, 36:28541–28564, 2023.

[9] Tianwei Lin, Wenqiao Zhang, Sijing Li, Yuqian Yuan, Binhe Yu, Haoyuan Li, Wanggui He, Hao Jiang, Mengze Li, Xiaohui Song, et al. HealthGPT: A medical large vision-language model for unifying comprehension and generation via heterogeneous knowledge adaptation. In Proceedings ofForty-second International Conference on Machine Learning (ICML), 2025.

[10] Jianing Qiu, Kyle Lam, Guohao Li, Amish Acharya, Tien Yin Wong, Ara Darzi, Wu Yuan, and Eric J Topol. Llm-based agentic systems in medicine and healthcare. Nature Machine Intelligence, 6(12):1418–1420, 2024.

[11] Karan Singhal, Tao Tu, Juraj Gottweis, Rory Sayres, Ellery Wulczyn, Mohamed Amin, Le Hou, Kevin Clark, Stephen R Pfohl, Heather Cole-Lewis, et al. Toward expert-level medical question answering with large language models. Nature medicine, 31(3):943–950, 2025.

[12] Kai Zhang, Rong Zhou, Eashan Adhikarla, Zhiling Yan, Yixin Liu, Jun Yu, Zhengliang Liu, Xun Chen, Brian D Davison, Hui Ren, et al. A generalist vision–language foundation model for diverse biomedical tasks. Nature medicine, 30(11):3129–3141, 2024.

[13] Michael Moor, Qian Huang, Shirley Wu, Michihiro Yasunaga, Yash Dalmia, Jure Leskovec, Cyril Zakka, Eduardo Pontes Reis, and Pranav Rajpurkar. Med-flamingo: a multimodal medical few-shot learner. In Machine Learningfor Health (ML4H), pages 353–367. PMLR, 2023.

[14] Junying Chen, Chi Gui, Ruyi Ouyang, Anningzhe Gao, Shunian Chen, Guiming Hardy Chen, Xidong Wang, Ruifei Zhang, Zhenyang Cai, Ke Ji, et al. Huatuogpt-vision, towards injecting medical visual knowledge into multimodal llms at scale. arXiv preprint arXiv:2406.19280, 2024.

[15] Andrew Sellergren, Sahar Kazemzadeh, Tiam Jaroensri, Atilla Kiraly, Madeleine Traverse, Timo Kohlberger, Shawn Xu, Fayaz Jamil, Cían Hughes, Charles Lau, et al. Medgemma technical report. arXiv preprint arXiv:2507.05201, 2025.

[16] Eta S Berner. Clinical decision support systems, volume 233. Springer, 2007.

[17] Xiaofeng Wang and Michael W Kattan. Cohort studies: design, analysis, and reporting. Chest, 158(1):S72– S78, 2020.

[18] Yusheng Liao, Chaoyi Wu, Junwei Liu, Shuyang Jiang, Pengcheng Qiu, Haowen Wang, Yun Yue, Shuai Zhen, Jian Wang, Qianrui Fan, et al. Ehr-r1: A reasoning-enhanced foundational language model for electronic health record analysis. arXiv preprint arXiv:2510.25628, 2025.

[19] Changshuo Liu, Wu Junran, Zhongle Xie, Wenqiao Zhang, Kaiping Zheng, Jiaqi Zhu, Qingpeng Cai, Ooi Gene Anne, Marcus Chun Jin Tan, Jianwei Yin, et al. Diyhealth suite: Dataset, model, and benchmark for health management at home.

[20] Alistair EW Johnson, Tom J Pollard, Seth J Berkowitz, Nathaniel R Greenbaum, Matthew P Lungren, Chih-ying Deng, Roger G Mark, and Steven Horng. Mimic-cxr, a de-identified publicly available database of chest radiographs with free-text reports. Scientific data, 6(1):317, 2019.

[21] Aurelia Bustos, Antonio Pertusa, Jose-Maria Salinas, and Maria De La Iglesia-Vaya. Padchest: A large chest x-ray image dataset with multi-label annotated reports. Medical image analysis, 66:101797, 2020.

[22] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in Neural Information Processing Systems, 36:34892–34916, 2023.

[23] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv e-prints, pages arXiv–2407, 2024.

[24] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

[25] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[26] Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, et al. Gemma 3 technical report. arXiv preprint arXiv:2503.19786, 2025.

[27] xAI. Grok 4 technical overview. https://x.ai/blog, 2025. Technical report.

[28] AI Anthropic. The claude 3 model family: Opus, sonnet, haiku. Claude-3 Model Card, 1(1):4, 2024.

[29] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

[30] Weiwen Xu, Hou Pong Chan, Long Li, Mahani Aljunied, Ruifeng Yuan, Jianyu Wang, Chenghao Xiao, Guizhen Chen, Chaoqun Liu, Zhaodonghui Li, et al. Lingshu: A generalist foundation model for unified multimodal medical understanding and reasoning. arXiv preprint arXiv:2506.07044, 2025.

[31] Songtao Jiang, Yuan Wang, Sibo Song, Tianxiang Hu, Chenyi Zhou, Bin Pu, Yan Zhang, Zhibo Yang, Yang Feng, Joey Tianyi Zhou, et al. Hulu-med: A transparent generalist model towards holistic medical vision-language understanding. arXiv preprint arXiv:2510.08668, 2025.

[32] Asma Ben Abacha and Dina Demner-Fushman. A question-entailment approach to question answering. BMC bioinformatics, 20(1):511, 2019.

[33] Junying Chen, Chi Gui, Ruyi Ouyang, Anningzhe Gao, Shunian Chen, Guiming Chen, Xidong Wang, Zhenyang Cai, Ke Ji, Xiang Wan, et al. Towards injecting medical visual knowledge into multimodal llms at scale. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 7346–7370, 2024.

[34] Bo Liu, Li-Ming Zhan, Li Xu, Lin Ma, Yan Yang, and Xiao-Ming Wu. Slake: A semantically-labeled knowledge-enhanced dataset for medical visual question answering. In 2021 IEEE 18th international symposium on biomedical imaging (ISBI), pages 1650–1654. IEEE, 2021.

[35] Jason J Lau, Soumya Gayen, Asma Ben Abacha, and Dina Demner-Fushman. A dataset of clinicall generated visual questions and answers about radiology images. Scientific data, 5(1):180251, 2018.

[36] Brian W Matthews. Comparison of the predicted and observed secondary structure of t4 phage lysozyme. Biochimica et Biophysica Acta (BBA)-Protein Structure, 405(2):442–451, 1975.

[37] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Proceedings ofthe 40th annual meeting ofthe Associationfor Computational Linguistics, pages 311–318, 2002.

[38] Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. Bertscore: Evaluating text generation with bert. In International Conference on Learning Representations.

[39] Jinhyuk Lee, Wonjin Yoon, Sungdong Kim, Donghyeon Kim, Sunkyu Kim, Chan Ho So, and Jaewoo Kang. Biobert: a pre-trained biomedical language representation model for biomedical text mining. Bioinformatics, 36(4):1234–1240, 2020.

[40] Saahil Jain, Ashwin Agrawal, Adriel Saporta, Steven QH Truong, Du Nguyen Duong, Tan Bui, Pierre Chambon, Yuhao Zhang, Matthew P Lungren, Andrew Y Ng, et al. Radgraph: Extracting clinical entities and relations from radiology reports. arXiv preprint arXiv:2106.14463, 2021.

[41] Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of machine learning research, 9(Nov):2579–2605, 2008.

[42] Renqian Luo, Liai Sun, Yingce Xia, Tao Qin, Sheng Zhang, Hoifung Poon, and Tie-Yan Liu. Biogpt: generative pre-trained transformer for biomedical text generation and mining. Briefings in bioinformatics, 23(6):bbac409, 2022.

[43] Yizhen Luo, Jiahuan Zhang, Siqi Fan, Kai Yang, Massimo Hong, Yushuai Wu, Mu Qiao, and Zaiqing Nie. Biomedgpt: An open multimodal large language model for biomedicine. IEEE Journal of Biomedical and Health Informatics, 2024.

[44] Hongyang Yang, Xiao-Yang Liu, and Christina Dan Wang. Fingpt: Open-source financial large language models. arXiv preprint arXiv:2306.06031, 2023.

[45] Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652, 2021.

[46] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

[47] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

[48] Xiaole Tang, Jiayi Xu, Xiang Gu, Yan Yang, and Jian Sun. Binding multiple modalities via multimodal wasserstein barycenter. arXiv preprint arXiv:2609.33800, 2026.

[49] Jiaqi Zhu, Shaofeng Cai, Fang Deng, Beng Chin Ooi, and Junran Wu. Do LLMs understand visual anomalies? uncovering LLM’s capabilities in zero-shot anomaly detection. In Proceedings ofthe 32nd ACM International Conference on Multimedia, pages 48–57, 2024.

[50] Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. Text embeddings by weakly-supervised contrastive pre-training. arXiv preprint arXiv:2212.03533, 2022.

[51] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 conference on empirical methods in natural language processing (EMNLP), pages 6769–6781, 2020.

[52] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledgeintensive nlp tasks. Advances in neural information processing systems, 33:9459–9474, 2020.

[53] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaugh lin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.

[54] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

[55] Yu Gu, Robert Tinn, Hao Cheng, Michael Lucas, Naoto Usuyama, Xiaodong Liu, Tristan Naumann, Jianfeng Gao, and Hoifung Poon. Domain-specific language model pretraining for biomedical natural language processing. ACM Transactions on Computingfor Healthcare (HEALTH), 3(1):1–23, 2021.

[56] Kexin Huang, Jaan Altosaar, and Rajesh Ranganath. Clinicalbert: Modeling clinical notes and predicting hospital readmission. arXiv preprint arXiv:1904.05342, 2019.

[57] Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, et al. Large language models encode clinical knowledge. Nature, 620(7972):172–180, 2023.

[58] Hongbo Zhang, Junying Chen, Feng Jiang, Fei Yu, Zhihong Chen, Guiming Chen, Jianquan Li, Xiangbo Wu, Zhang Zhiyi, Qingying Xiao, et al. Huatuogpt, towards taming language model to be a doctor. In Findings of the association for computational linguistics: EMNLP 2023, pages 10859–10885, 2023.

[59] Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. Advances in neural information processing systems, 33:21271–21284, 2020.

[60] Antti Tarvainen and Harri Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. Advances in neural information processing systems, 30, 2017.

[61] Florian Schroff, Dmitry Kalenichenko, and James Philbin. Facenet: A unified embedding for face recognition and clustering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 815–823, 2015.

## Appendix

• A. Notations . . 15   
• B. Related Work . . . 16   
• C. Implementation Details . . 16   
C.1. Datasets . 16   
C.2. Model Implementation Details . 17   
C.3. Experimental Environment . 17   
C.4. Evaluation Metrics . 18   
• D. Cohort Structure . 19   
• E. Supplementary Experimental Results . . 20   
E.1. Supplementary Ablation Study . . . 20   
E.2. Supplementary Comparisons with Fine-tuned Baselines . . . 20   
E.3. Comparison with Few-shot Baselines . . . 21   
E.4. Hyperparameter Study . . 21   
• F. Case Study . 23   
• G. Broader Impacts . .25

## A Notations

To provide a comprehensive overview of the notations used throughout the paper, we present a summary of notations in Table 5 as a quick reference to facilitate the understanding and recall of each symbol.

Table 5: Notations.
<table><tr><td>Notation</td><td>Description</td></tr><tr><td>X</td><td>Input instance, consisting of clinical records, a question and an image.</td></tr><tr><td> $\mathbf { y }$ </td><td>The ground truth output sequence.</td></tr><tr><td> $x _ { i }$ </td><td>The i-th clinical records in x.</td></tr><tr><td> $\mathbf { q }$ </td><td>The question in x.</td></tr><tr><td> $\mathcal { T }$ </td><td>Input image of height H, width W, and three RGB channels.</td></tr><tr><td> $_ { x }$ </td><td>The textual input, including clinical records and a question.</td></tr><tr><td> $\mathcal { Z } ^ { ( 0 ) }$ </td><td>Initial multimodal embedding, formed by visual and text embeddings.</td></tr><tr><td> $\mathcal { Z } ^ { ( \ell ) }$ </td><td>Output of l-th transformer block.</td></tr><tr><td> $L _ { b }$ </td><td>The number of transformer blocks in the backbone.</td></tr><tr><td> $f _ { I  t }$ </td><td>The function that maps the visual embeddings into the shared textual embedding space.</td></tr><tr><td> $\mathcal { E } _ { I } ( \cdot )$ </td><td>The visual encoder that maps images to embeddings.</td></tr><tr><td> $\mathcal { E } _ { t } ( \cdot )$ </td><td>The textual encoder that maps words to embeddings.</td></tr><tr><td> $L _ { I }$ </td><td>The number of visual tokens in image Z, similar to  $L _ { t } .$ </td></tr><tr><td> $d _ { t }$ </td><td>The dimensions of shared textual embedding space.</td></tr><tr><td> $p _ { \theta }$ </td><td>The conditional distribution parameterized by a large language model with parameters θ.</td></tr><tr><td> $\mathcal { M }$ </td><td>Markov decision process (MDP).</td></tr><tr><td>S, A, P, r and γ</td><td>State space, action space, transition dynamics, reward function and discount factor, respectively.</td></tr><tr><td> $\pi _ { \phi }$ </td><td>Policy,  $\pi _ { \phi } ( \boldsymbol { a } \mid \boldsymbol { s } )$  parameterized by  $\phi ,$  which in this work corresponds to the cohort-partitioning policy.</td></tr><tr><td> $\tau$ </td><td>Trajectory generated by the policy, i.e., a sequence of states and actions.</td></tr><tr><td> $\mathcal { F } ^ { ( \ell ) }$ </td><td>The l-th Transformer block.</td></tr><tr><td> $\tau$ </td><td>The learned hierarchical cohort tree.</td></tr><tr><td> $\mathcal { P }$ </td><td>The set of patients.</td></tr><tr><td> $\mathcal { C }$ </td><td>The set of cohorts.</td></tr><tr><td> $\mathbf { h } _ { C }$ </td><td>The cohort prototypes of cohorts.</td></tr><tr><td> $\Delta ( \mathbf { x } , c )$ </td><td>The deviation between input x and cohort c.</td></tr><tr><td> $\theta$ </td><td>The learnable model parameters.</td></tr><tr><td> $p _ { \tau } ( \cdot \mid \pi _ { \phi } )$ </td><td>The trajectory distribution induced by policy  $\pi _ { \phi }$ </td></tr><tr><td> $\mathcal { R } ( \mathbf { x } , \tau )$ </td><td>The trajectory-level cohort discovery reward</td></tr><tr><td> $\mathcal { H } ( \boldsymbol { p } )$ </td><td>Entropy of a distribution p.</td></tr><tr><td> $p _ { \pi _ { \phi } } ( \cdot \mid \mathbf { x } )$ </td><td>Policy distribution conditioned on input x.</td></tr><tr><td> $g ( \cdot )$ </td><td>The function that maps serialized EHR records into a vector of statistical features.  $\tau .$ </td></tr><tr><td> $V$ </td><td>The set of nodes in  $\tau ,$  similar to  $E .$  which is the set of edges in</td></tr><tr><td> $\mu ( \cdot )$ </td><td>The mean function.</td></tr><tr><td> $\sigma ( \cdot )$ </td><td>The standard deviation function.</td></tr><tr><td> $d _ { v }$ </td><td>Depth of node v.</td></tr><tr><td> $\mathbf { U } _ { \mathcal { P } _ { v } }$ </td><td>Set of statistical features of patients  $\mathcal { P } _ { v }$ </td></tr><tr><td> $z _ { v }$ </td><td>Binary stopping variable for node v.</td></tr><tr><td> $j _ { v }$ </td><td>The selected feature for node v.</td></tr><tr><td> $\mathbf { t } _ { v }$ </td><td>The split threshold for node v.</td></tr><tr><td> $d _ { \mathrm { m a x } }$ </td><td>The maximum depth.</td></tr><tr><td> $\mathbf { e } _ { c }$ </td><td>Cohort embedding that will be injected into the multimodal input.</td></tr><tr><td> $\mathcal { P } _ { c }$ </td><td>The patients within cohort c.</td></tr><tr><td> $\mathcal { L } _ { \mathrm { g e n } }$ </td><td>Downstream generation loss.</td></tr><tr><td> $R _ { v }$ </td><td>The generation-aware reward.</td></tr><tr><td> $\mathbf { h } _ { p }$ </td><td>Learned representation of patient  $p .$ </td></tr><tr><td> $\mathbf { W } _ { \ell } ^ { \alpha }$ </td><td>Learnable weight matrix, similar to  $\mathbf { W } _ { \ell } ^ { \beta }$  Wγ, Wi and</td></tr></table>

## B Related Work

Foundation Models in General Domain. The rapid development of LLMs has significantly reshaped modern AI, enabling strong general-purpose reasoning and coherent, human-like interactions. Beginning with generative pre-trained Transformers (GPT) [1, 2], it has been widely proven that scaling model size and training data leads to substantial improvements in general-purpose question answering across a broad range of tasks [42, 43, 44]. Subsequent advances further enhanced vision-language alignment and multimodal understanding through instruction tuning [45, 22, 46] and reinforcement learning from human feedback (RLHF) [47]. Recent studies explore multimodal alignment via Wasserstein barycenters [48] and fine-grained image-text alignment for zero-shot anomaly detection [49]. Beyond generative modeling, representation learning approaches, including embedding models such as E5 [50] and dense retrieval frameworks like DPR [51], have supported retrieval augmented generation (RAG) [52] by providing high-quality semantic representations. These approaches allow LLMs to incorporate external knowledge, thereby improving factual consistency and mitigating hallucinations. More recently, advanced LLMs, such as Claude [28], GPT-5 [53] and Gemini [54], have demonstrated strong performance in complicated multimodal understanding and generation tasks. Yet, the general-purpose LLMs are not readily applicable to specialized domains such as healthcare. In such high-stakes settings, effective reasoning requires not only general linguistic competence but also a deep understanding of domain-specific knowledge and complex clinical data.

Foundation Models in Medical Domain. Following the success of general-purpose LLMs, extensive efforts have been made to adapt these models to the medical domain. Early efforts focus on domain-specific pretraining, where models such as BioBERT [39], PubMedBERT [55] and ClinicalBERT [56] are trained on large-scale biomedical corpora to learn medical domain knowledge and preserve medical semantics. More recently, Med-PaLM [57] and HuatuoGPT [58] are proposed to extend LLMs to clinical applications, including medical question answering, report generation, and medical decision support. At the same time, multimodal medical LVLMs are developed to integrate heterogeneous clinical data. For example, LLaVA-Med [8] and Med-Flamingo [13] enables joint reasoning over radiology images and medical reports. Building upon these advances, subsequent models, exemplified by HealthGPT [9] and Hulu-Med [31], further scale medical corpora and task diversity to enhance the medical understanding of complex clinical tasks involving both structured and unstructured data. Despite these improvements, existing models mainly operate at the individual instance level, treating each patient independently and relying on implicit medical knowledge learned by model parameters. However, real-world clinical reasoning is inherently cohort-based, where clinicians often compare patients within similar populations and reason about deviations from the typical patterns. The lack of explicit modeling of cohort structure and patient-level derivation limits the ability of current foundation models to capture clinically meaningful heterogeneity.

## C Implementation Details

## C.1 Datasets

The experiments are conducted on four real-world EHR benchmarks: EHR-Bench [18], DIYHealth-Bench [19], MIMIC-CXR [20], and PadChest [21]. The task description for each dataset is introduced in Table 6. EHR-Bench and DIYHealth-Bench provide raw QA pairs. For the MIMIC-CXR and PadChest datasets, we further preprocess the EHR records and radiology reports into unified QA-pair formats to ensure consistency with EHR-Bench and DIYHealth-Bench. To balance evaluation reliability and computational efficiency, datasets containing more than 2,500 samples are split by reserving 500 samples for testing and using the remaining samples for training. For smaller datasets, we adopt a 4:1 training-to-test split ratio.

Table 6: Task descriptions across datasets.
<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Task Description</td></tr><tr><td rowspan=1 colspan=1>Admissions</td><td rowspan=1 colspan=1>Predict the next admission type from EHR records (Closed-QA) and provide clinical reasoning (Open-QA)</td></tr><tr><td rowspan=1 colspan=1>Services</td><td rowspan=1 colspan=1>Predict the next clinical department from EHR records (Closed-QA) and provide clinical reasoning (Open-QA).</td></tr><tr><td rowspan=1 colspan=1>Ingredient Events (IE)</td><td rowspan=1 colspan=1>Predict the next administered medication or substance from EHR records (Closed-QA) and provide clinical reasoning (Open-QA).</td></tr><tr><td rowspan=1 colspan=1>Provider Order Entry (POE)</td><td rowspan=1 colspan=1>Predict the next clinical order type from EHR records (Closed-QA) and provide clinical reasoning (Open-QA).</td></tr><tr><td rowspan=1 colspan=1>Transfers</td><td rowspan=1 colspan=1>Predict the next care location from EHR records (Closed-QA) and provide clinical reasoning (Open-QA)</td></tr><tr><td rowspan=1 colspan=1>Obesity</td><td rowspan=1 colspan=1>Assess obesity risk from self-reported EHR records (Closed-QA) and provide supporting analysis (Open-QA).</td></tr><tr><td rowspan=1 colspan=1>MIMIC-CXR</td><td rowspan=1 colspan=1>Generate radiology reports from EHR records and chest X-ray images.</td></tr><tr><td rowspan=1 colspan=1>PadChest</td><td rowspan=1 colspan=1>Generate radiology reports from a few EHR records and chest X-ray images</td></tr></table>

## C.2 Model Implementation Details

We adopt Qwen3-8B and LLaMA3-8B as backbone models. The projector $f _ { I  t }$ is implemented as a two-layer MLP. Both g(·) and f are instantiated as MLPs, and $f _ { { \bf u } \to { \bf h } }$ is an MLP-based encoder. The maximum depth d<sub>max</sub> is set to 100. We set $\lambda _ { g } = 0 . 3$ and $\lambda _ { d } = 1 0 ^ { - 4 }$ across all experiments. The coefficient β is set to 0.01 following prior work [59, 60], and the margin m is set to 0.2 following prior study [61]. The assignment confidence of patient p to cohort c is computed based on the similarity between $\mathbf { h } _ { p }$ and $\mathbf { h } _ { c } .$

We present the detailed configurations for the training process. The specific settings are listed in Table 7. These hyperparameters are set up following prior studies [9, 22, 8, 19]. The architectural parameters are listed in Table 8.

Table 7: Summary of configurations for the training process.
<table><tr><td>Configuration</td><td>Stage (i)</td><td>Stage (ii)</td><td>Stage (iii)</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Adapter LR</td><td>1</td><td>5e-5</td><td>1e-4</td></tr><tr><td>Learning Rate</td><td>2e-5</td><td>1e-4</td><td>2e-5</td></tr><tr><td>Weight Decay</td><td>1</td><td>0.01</td><td>0.01</td></tr><tr><td>Dropout Rate</td><td>0.05</td><td>0.1</td><td>0.1</td></tr><tr><td>LR Scheduler</td><td>Warmup</td><td>Constant</td><td>Cosine w/ Warmup</td></tr><tr><td>Max Sequence Length</td><td>32k</td><td>32k</td><td>32k</td></tr></table>

Table 8: Architectural parameters of CODA.
<table><tr><td rowspan=1 colspan=3>Module                  Parameter                  Value / Shape</td></tr><tr><td rowspan=1 colspan=1>Tokenizer</td><td rowspan=1 colspan=1>Vocabulary size</td><td rowspan=1 colspan=1>151,670</td></tr><tr><td rowspan=1 colspan=1>Backbone (Qwen3-VL)</td><td rowspan=1 colspan=1>Hidden size# Transformer layersEmbedding matrixSelf-attn projectionsMLP</td><td rowspan=1 colspan=1>409636(151670, 4096)q: (4096, 4096), k/v: (4096, 1024), o: (4096, 4096)gate/up: (4096, 12288), down: (12288, 4096)</td></tr><tr><td rowspan=1 colspan=1>LoRA</td><td rowspan=1 colspan=1>Rank rMatrix AMatrix B</td><td rowspan=1 colspan=1>16q/k/v/o/gate/up: (16, 4096), down: (16, 12288)q/o: (4096,16), k/v: (1024,16),gate/up: (12288,16), down: (4096,16)</td></tr><tr><td rowspan=1 colspan=1>Stage ii</td><td rowspan=1 colspan=1>Cohort embedding dimPrefix lengthPrefix projectorCohort encoder hidden dim</td><td rowspan=1 colspan=1>10248(4096, 4096) → (4096, 8 × 4096)4096</td></tr><tr><td rowspan=1 colspan=1>Stage iii</td><td rowspan=1 colspan=1>Drift dimAdapter bottleneck dimAdapter (down / up)Drift generatorGating module</td><td rowspan=1 colspan=1>1024256(4096, 256), (256, 4096)(1024, 512)(1024, 4096)</td></tr><tr><td rowspan=1 colspan=1>Cohort Memory</td><td rowspan=1 colspan=1># CohortsPrototype matrix</td><td rowspan=1 colspan=1>34(34, 1024)</td></tr></table>

## C.3 Experimental Environment

All experiments are conducted on a server equipped with an AMD EPYC 9334 CPU (32 cores, 2.7 GHz), 128 GB RAM, and an NVIDIA H100 NVL GPU with CUDA 12.9. The system runs Ubuntu 24.04 with Linux kernel 6.8.0-79-generic.

## C.4 Evaluation Metrics

## C.4.1 Evaluation Metrics for Closed-QA

• Accuracy. Accuracy (ACC) evaluates the overall prediction performance by computing the fraction of correctly classified instances among all samples. It provides a coarse-grained measure of classification effectiveness, especially under balanced label distributions.

$$
A C C = { \frac { T P + T N } { T P + T N + F P + F N } }\tag{18}
$$

where TP, TN, FP, and FN are the numbers of true positives, true negatives, false positives, and false negatives, respectively.

• Matthews Correlation Coefficient. The Matthews Correlation Coefficient (MCC) provides a holistic evaluation of classification performance by capturing the correlation between predicted and true labels through the full confusion matrix. In contrast to accuracy, MCC remains robust under class imbalance, as it simultaneously considers all types of prediction outcomes and their interactions across classes. As a normalized correlation measure, MCC ranges from −1 to 1, where 1 indicates perfect alignment, 0 corresponds to chance-level prediction, and −1 reflects complete disagreement.

$$
\mathrm { M C C } = \frac { N \sum _ { k } C _ { k k } - \sum _ { k } T _ { k } P _ { k } } { \sqrt { \left( N ^ { 2 } - \sum _ { k } P _ { k } ^ { 2 } \right) \left( N ^ { 2 } - \sum _ { k } T _ { k } ^ { 2 } \right) } }\tag{19}
$$

where $C \in \mathbb { R } ^ { K \times K }$ denotes the confusion matrix, and $C _ { k l }$ represents the number of samples whose groundtruth class is k and predicted class is l. $\textstyle . \ T _ { k } = \sum _ { l } C _ { k l }$ and $\begin{array} { r } { \dot { P _ { k } } = \sum _ { l } C _ { l k } } \end{array}$ denote the total number of true and predicted instances for class $k ,$ respectively. $\begin{array} { r } { \overline { { N } } { } ^ { \circ } = \sum _ { k } \sum _ { l } C _ { k l } } \end{array}$ is the total number of samples, and K is the number of classes.

## C.4.2 Evaluation Metrics for Open-QA

• F1-BioBERT. F1-BioBERT evaluates semantic consistency between generated answers and ground-truth responses by leveraging contextual embeddings from BioBERT [39]. Instead of relying on exact lexical matching, it computes similarity at the representation level, enabling robust assessment of paraphrased or semantically equivalent expressions in medical text.

Specifically, precision and recall are defined based on semantic matching between predicted and reference tokens (or spans) in the embedding space, and the final score is computed as their harmonic mean:

$$
\mathrm { F l - B i o B E R T } = \frac { 2 \cdot \mathrm { P r e c i s i o n } \cdot \mathrm { R e c a l l } } { \mathrm { P r e c i s i o n } + \mathrm { R e c a l l } } .\tag{20}
$$

Here, matches are determined according to similarity scores derived from BioBERT representations, with higher scores indicating stronger semantic alignment.

• BLEU-1. BLEU-1 evaluates lexical fidelity by measuring the modified unigram precision between the generated text and the reference. Compared with higher-order BLEU variants, BLEU-1 focuses on word-level overlap and is therefore suitable for assessing whether key lexical content is preserved.

$$
{ \bf B L E U - 1 } = { \bf B P \cdot \exp \left( \log p _ { 1 } \right) } ,\tag{21}
$$

where $p _ { 1 }$ is the modified unigram precision:

$$
p _ { 1 } = \frac { \sum _ { w \in \mathcal { V } } \operatorname* { m i n } \left( \mathrm { C o u n t } _ { \mathrm { p r e d } } ( w ) , \mathrm { C o u n t } _ { \mathrm { r e f } } ( w ) \right) } { \sum _ { w \in \mathcal { V } } \mathrm { C o u n t } _ { \mathrm { p r e d } } ( w ) } .\tag{22}
$$

Here, $\nu$ denotes the unigram vocabulary, while $\mathrm { C o u n t _ { p r e d } } ( w )$ and $\mathrm { C o u n t } _ { \mathrm { r e f } } ( \omega )$ denote the frequency of unigram w in the generated and reference texts, respectively. BP is the brevity penalty that discourages overly short generations.

## C.4.3 Evaluation Metrics for Report Generation

• F1-RadGraph. F1-RadGraph evaluates the quality of generated radiology reports by measuring the overlap between predicted and reference clinical graphs constructed using RadGraph [40]. Specifically, it assesses the correctness of extracted medical entities and their associated relations, thereby capturing both factual accuracy and clinical consistency at the structured level. The metric is computed as the harmonic mean of precision and recall over matched graph elements:

$$
\mathrm { F l - R a d G r a p h } = \frac { 2 \cdot \mathrm { P r e c i s i o n } \cdot \mathrm { R e c a l l } } { \mathrm { P r e c i s i o n } + \mathrm { R e c a l l } }\tag{23}
$$

where precision and recall are defined based on the matching between predicted and ground-truth Rad-Graph annotations, with true positives corresponding to correctly identified entities and relations, and false positives/negatives representing mismatched or missing clinical concepts.

## D Cohort Structure

The learned cohort tree provides an interpretable hierarchical decomposition of the patient space, where each internal node corresponds to a clinically meaningful decision criterion, and each leaf node defines a distinct cohort, as shown in Figure 6. At the top levels, the model prioritizes global clinical signals such as cardiac-related markers, care context (e.g., discharge disposition and observation status), and multimorbidity, indicating that population stratification is first driven by overall severity and healthcare utilization. As the hierarchy progresses, the tree further branches along complementary axes, including chronic disease patterns (e.g., cardiometabolic conditions) and imaging-derived phenotypes (e.g., pleural effusion and pneumothorax), demonstrating that the model disentangles heterogeneous patient populations using semantically coherent, multimodal criteria. At deeper levels, the structure shifts toward fine-grained refinement, where demographic attributes (e.g., age and gender) and laboratory-derived signals (e.g., hematologic and metabolic abnormalities) are used to differentiate patients with subtle physiological variations. Notably, certain features recur across different branches, reflecting context-dependent decision boundaries rather than fixed global rules. In addition, the tree exhibits adaptive depth, with simpler cohorts identified early and more complex populations refined through deeper branches.

Overall, this hierarchy reveals that the proposed method learns clinically grounded and interpretable cohort structures, organizing patients into meaningful subgroups that support coherent and reliable downstream reasoning.

![](images/11be6374394921291840e3fb4ae06a073028b222d7fcb9a718ae77e2cd02b1d6.jpg)  
Figure 6: Illustration of Cohort Structure.

## E Supplementary Experimental Results

## E.1 Supplementary Ablation Study

Tables 9 and 10 provide extended ablation results across closed-QA, open-QA, and report generation settings. Consistent with the main paper, removing either cohort-aware population stratification or drift-aware individual adaptation leads to performance degradation across both in-distribution and out-of-distribution tasks, demonstrating that the two components are complementary and jointly contribute to robust multimodal clinical reasoning.

Under closed-QA settings, CODA achieves the best overall performance across all tasks and evaluation metrics. Similar trends are observed in open-QA and report generation settings. CODA consistently achieves the strongest performance across both language-level and clinical evaluation metrics. Overall, the results demonstrate that population-level cohort structure and patient-specific deviation modeling are both critical for reliable multimodal clinical reasoning.

Table 9: Ablation Study of CODA under Closed-QA settings.
<table><tr><td rowspan="3">Model</td><td colspan="8"></td><td colspan="4">OOD</td><td rowspan="2" colspan="2">Avg.</td></tr><tr><td colspan="2">Admissions</td><td colspan="2">E</td><td colspan="2">POE</td><td colspan="2">Services</td><td colspan="2">Transfers</td><td colspan="2">Obesity</td></tr><tr><td>ACC</td><td>MCC</td><td>ACC</td><td>MCC</td><td>ACC</td><td>MCC</td><td>ACC</td><td>MCC</td><td>ACC</td><td>MCC</td><td>ACC</td><td>MCC</td><td>ACC</td><td>MCC</td></tr><tr><td>LoRA Only</td><td>74.60</td><td>65.78</td><td>91.92</td><td>89.04</td><td>81.20</td><td>74.83</td><td>92.02</td><td>89.42</td><td>73.60</td><td>64.65</td><td>70.30</td><td>58.48</td><td>80.61</td><td>73.70</td></tr><tr><td>w/o Cohort</td><td>75.20</td><td>66.57</td><td>92.17</td><td>89.57</td><td>82.20</td><td>76.25</td><td>92.77</td><td>90.38</td><td>74.20</td><td>65.16</td><td>70.92</td><td>59.21</td><td>81.24</td><td>74.52</td></tr><tr><td>w/o Deviation</td><td>75.80</td><td>67.37</td><td>92.17</td><td>89.51</td><td>82.00</td><td>76.00</td><td>92.27</td><td>90.06</td><td>74.40</td><td>65.70</td><td>71.70</td><td>60.20</td><td>81.39</td><td>74.81</td></tr><tr><td>CODA</td><td>76.60</td><td>68.77</td><td>92.42</td><td>89.93</td><td>82.80</td><td>77.05</td><td>93.52</td><td>91.39</td><td>75.20</td><td>66.89</td><td>72.32</td><td>61.28</td><td>82.14</td><td>75.88</td></tr></table>

Table 10: Ablation Study of CODA under Open-QA and report generation settings.
<table><tr><td rowspan="3">Model</td><td colspan="10">ID POE</td><td colspan="6">OOD</td><td colspan="4">Avg.</td></tr><tr><td>Admissions</td><td></td><td></td><td>IE</td><td colspan="2"></td><td colspan="2"></td><td colspan="2">MIMIC-CXR</td><td colspan="2"></td><td colspan="2">Obesity</td><td colspan="2">PadChest</td><td colspan="2"></td><td colspan="2">Open-QA</td><td colspan="2">Report</td></tr><tr><td>F1-Bio</td><td>BLEU-1</td><td></td><td>F1-Bio</td><td>BLEU-1</td><td>F1-Bio</td><td>BLEU-1</td><td>F1-Bio</td><td>BLEU-1</td><td>F1-Rad</td><td>F1-Bio</td><td>F1-Bio</td><td>BLEU-1</td><td>F1-Bio</td><td>BLEU-1</td><td>F1-Rad</td><td></td><td>F1-Bio</td><td>F1-Bio</td><td>BLEU-1</td><td>F1-Rad F1-Bio</td></tr><tr><td>LoRA Only</td><td>79.06</td><td>46.09</td><td>80.94</td><td>47.20</td><td>80.16</td><td>50.27</td><td>78.87</td><td></td><td>46.24</td><td>15.46</td><td>78.74</td><td>78.61</td><td>44.18</td><td>74.61</td><td>13.44</td><td>24.68</td><td>72.24</td><td></td><td></td><td></td><td></td></tr><tr><td>w/o Cohort</td><td>79.71</td><td>48.02</td><td>81.10</td><td>49.08</td><td>80.44</td><td>52.03</td><td>79.20</td><td>47.60</td><td>15.73</td><td></td><td>78.58</td><td>78.58</td><td>43.66</td><td>75.07</td><td>14.01</td><td>24.60</td><td>72.51</td><td>78.71 79.02</td><td>41.24 42.40</td><td>20.07 20.17</td><td>75.49 75.55</td></tr><tr><td>w/o Deviation</td><td>79.80</td><td>48.64</td><td>81.18</td><td>48.99</td><td>80.06</td><td>50.50</td><td>79.34</td><td>47.32</td><td>16.33</td><td></td><td>78.94</td><td>78.55</td><td>43.68</td><td>74.76</td><td>13.96</td><td>25.19</td><td>72.41</td><td>78.95</td><td>42.18</td><td>20.76</td><td>75.67</td></tr><tr><td>CODA</td><td>80.82</td><td>52.45</td><td>82.21</td><td>51.69</td><td>81.02</td><td>51.97</td><td>80.21</td><td>49.19</td><td></td><td>17.49</td><td>79.03</td><td>79.57</td><td>46.42</td><td>75.37</td><td>15.65</td><td>25.52</td><td>73.93</td><td>79.87</td><td>44.56</td><td>21.51</td><td>76.48</td></tr></table>

## E.2 Supplementary Comparisons with Fine-tuned Baselines

Tables 11 and 12 provide the full comparisons between CODA and representative LoRA fine-tuned multimodal medical foundation models under closed-QA, open-QA, and report generation settings. CODA consistently achieves the best overall performance across both in-distribution and out-of-distribution tasks.

Table 11: Comparison of CODA with LoRA fine-tuned baselines under Closed-QA settings.
<table><tr><td>Model</td><td colspan="8"></td><td colspan="4">OOD</td><td rowspan="2" colspan="2">Avg.</td></tr><tr><td></td><td colspan="2">Admissions</td><td colspan="2">IE</td><td colspan="2">POE</td><td colspan="2">Services</td><td colspan="2">Transfers</td><td colspan="2">Obesity</td></tr><tr><td></td><td>ACC</td><td>MCC</td><td>ACC</td><td>MCC</td><td>ACC</td><td>MCC</td><td>ACC</td><td>MCC</td><td>ACC</td><td>MCC</td><td>ACC</td><td>MCC</td><td>ACC</td><td>MCC</td></tr><tr><td>InternVL 3-8B</td><td>76.00</td><td>67.98</td><td>91.18</td><td>88.51</td><td>81.20</td><td>74.93</td><td>93.27</td><td>91.02</td><td>74.60</td><td>65.75</td><td>71.23</td><td>59.52</td><td>81.25</td><td>74.62</td></tr><tr><td>Lingshu-7B</td><td>76.20</td><td>68.24</td><td>91.67</td><td>88.85</td><td>81.20</td><td>74.96</td><td>93.27</td><td>91.07</td><td>74.40</td><td>65.52</td><td>71.38</td><td>60.24</td><td>81.35</td><td>74.81</td></tr><tr><td>CODA-Qwen3-8B</td><td>76.60</td><td>68.77</td><td>92.42</td><td>89.93</td><td>82.80</td><td>77.05</td><td>93.52</td><td>91.39</td><td>75.20</td><td>66.89</td><td>72.32</td><td>61.28</td><td>82.14</td><td>75.88</td></tr></table>

Table 12: Comparison of CODA with LoRA fine-tuned baselines under Open-QA and report generation settings.
<table><tr><td rowspan="3">Model</td><td colspan="10">ID</td><td colspan="7">OOD</td><td colspan="3">Avg.</td></tr><tr><td>Admissions</td><td></td><td></td><td>E</td><td colspan="2">POE</td><td colspan="2">Services</td><td colspan="2">MIMIC-CXR</td><td colspan="2">Transfers</td><td colspan="2">Obesity</td><td colspan="2">PadChest</td><td colspan="2"></td><td colspan="2">Open-QA</td><td colspan="2">Report</td></tr><tr><td>F1-Bio</td><td>BLEU-1</td><td>F1-Bio</td><td>BLEU-1</td><td>F1-Bio</td><td></td><td>BLEU-1</td><td>F1-Bio</td><td>BLEU-1</td><td>F1-Rad</td><td>F1-Bio</td><td>F1-Bio</td><td>BLEU-1</td><td>F1-Bio</td><td>BLEU-1</td><td>F1-Rad</td><td>F1-Bio</td><td>F1-Bio</td><td></td><td>BLEU-1</td><td>F1-Rad F1-Bio</td></tr><tr><td>InternVL 3-8B</td><td>80.73</td><td>52.10</td><td>81.54</td><td>50.22</td><td>80.91</td><td>51.51</td><td>80.18</td><td>49.39</td><td>16.59</td><td>78.69</td><td>79.14</td><td>46.00</td><td>72.17</td><td>11.21</td><td>23.85</td><td></td><td>72.05</td><td>79.11</td><td>43.74</td><td>20.22</td><td>75.37</td></tr><tr><td>Lingshu-7B CODA</td><td>80.60 80.82</td><td>48.92 52.45</td><td>82.16 82.21</td><td>50.76 51.69</td><td>80.99 81.02</td><td>51.65 51.97</td><td>80.01 80.21</td><td>49.03 49.19</td><td>16.89 17.49</td><td>78.88 79.03</td><td>78.82 79.57</td><td>46.64 46.42</td><td>73.72 75.37</td><td>12.07 15.65</td><td>25.52</td><td>22.58</td><td>71.90 73.93</td><td>79.38 79.87</td><td>43.44 44.56</td><td>19.74 21.51</td><td>75.39 76.48</td></tr></table>

## E.3 Comparison with Few-shot Baselines

To further evaluate CODA against strong baselines, we augment several representative models with few-shot demonstrations. The results are reported in Tables 13 and 14. Due to the rich and heterogeneous nature of EHR data, each sample is relatively long, posing challenges for models with limited context length constraints. While InternVL 3 and Hulu-Med benefit from few-shot demonstrations, LLaVA 1.5 and Lingshu exhibit performance degradation, suggesting insufficient capability to effectively process long-context inputs. In particular, the extreme degradation observed in LLaVA 1.5 under the 2-shot setting can also be attributed to its 4K context window, which is insufficient to accommodate both demonstrations and the query. Overall, these results further highlight the effectiveness of CODA.

Table 13: Comparison of CODA with few-shot baselines under Closed-QA settings.
<table><tr><td rowspan="3">Model</td><td colspan="8"></td><td colspan="4">OOD</td><td rowspan="2" colspan="2">Avg.</td></tr><tr><td colspan="2">Admissions</td><td colspan="2">IE</td><td colspan="2">POE</td><td colspan="2">Services</td><td colspan="2">Transfers</td><td colspan="2">Obesity</td></tr><tr><td>ACC</td><td>MCC</td><td>ACC</td><td>MCC</td><td>ACC</td><td>MCC</td><td>ACC</td><td>MCC</td><td>ACC</td><td>MCC</td><td>ACC</td><td>MCC</td><td>ACC</td><td>MCC</td></tr><tr><td>LLaVA 1.5-7B 1-shot</td><td>35.60</td><td>17.73</td><td>20.20</td><td>0.94</td><td>41.80</td><td>26.70</td><td>55.36</td><td>41.83</td><td>54.60</td><td>39.33</td><td>58.79</td><td>44.27</td><td>44.39</td><td>28.46</td></tr><tr><td>LLaVA 1.5-7B 2-shot</td><td>10.60</td><td>5.68</td><td>8.59</td><td>5.72</td><td>14.20</td><td>11.59</td><td>18.20</td><td>16.18</td><td>46.20</td><td>34.57</td><td>49.61</td><td>35.99</td><td>24.57</td><td>18.29</td></tr><tr><td>InternVL 3-8B 1-shot</td><td>38.00</td><td>17.64</td><td>29.04</td><td>5.55</td><td>45.40</td><td>27.36</td><td>55.36</td><td>40.53</td><td>38.20</td><td>17.96</td><td>53.65</td><td>37.40</td><td>43.28</td><td>24.41</td></tr><tr><td>InternVL 3-8B 2-shot</td><td>42.60</td><td>23.74</td><td>30.30</td><td>6.82</td><td>43.40</td><td>24.72</td><td>56.86</td><td>42.46</td><td>42.60</td><td>23.92</td><td>53.50</td><td>36.94</td><td>44.88</td><td>26.43</td></tr><tr><td>lingshu-7B 1-shot</td><td>38.80</td><td>19.01</td><td>34.85</td><td>14.62</td><td>49.80</td><td>33.25</td><td>69.08</td><td>58.83</td><td>49.20</td><td>33.08</td><td>65.01</td><td>51.82</td><td>51.12</td><td>35.10</td></tr><tr><td>lingshu-7B 2-shot</td><td>45.60</td><td>27.99</td><td>38.13</td><td>17.71</td><td>48.40</td><td>31.26</td><td>70.32</td><td>60.57</td><td>51.00</td><td>34.97</td><td>68.58</td><td>56.05</td><td>53.67</td><td>38.09</td></tr><tr><td>Hulu-Med-7B 1-shot</td><td>43.40</td><td>24.78</td><td>30.81</td><td>7.86</td><td>57.60</td><td>43.66</td><td>65.84</td><td>54.53</td><td>49.20</td><td>32.28</td><td>60.19</td><td>47.34</td><td>51.17</td><td>35.07</td></tr><tr><td>Hulu-Med-7B 2-shot</td><td>49.60</td><td>33.02</td><td>34.09</td><td>12.33</td><td>59.60</td><td>46.31</td><td>68.58</td><td>58.59</td><td>49.40</td><td>32.60</td><td>63.61</td><td>50.78</td><td>54.15</td><td>38.94</td></tr><tr><td>CODA-Qwen3-8B</td><td>76.60</td><td>68.77</td><td>92.42</td><td>89.93</td><td>82.80</td><td>77.05</td><td>93.52</td><td>91.39</td><td>75.20</td><td>66.89</td><td>72.32</td><td>61.28</td><td>82.14</td><td>75.88</td></tr></table>

Table 14: Comparison of CODA with few-shot baselines under Open-QA and report generation settings.
<table><tr><td rowspan="3">Model</td><td colspan="10">In-distribution Task</td><td colspan="6">Out-of-distribution Task</td><td colspan="4">Avg.</td></tr><tr><td>Admissions</td><td></td><td></td><td>IE</td><td></td><td>POE</td><td></td><td>Services</td><td></td><td>MIMIC-CXR</td><td></td><td>Transfers</td><td></td><td>Obesity</td><td></td><td>PadChest</td><td></td><td>Open-QA</td><td></td><td>Report</td><td></td></tr><tr><td>F1-Bio</td><td>BLEU-1</td><td>F1-Bio</td><td></td><td>BLEU-1</td><td>F1-Bio</td><td>BLEU-1</td><td>F1-Bio</td><td>BLEU-1</td><td>F1-Rad</td><td>F1-Bio</td><td>F1-Bio</td><td>BLEU-1</td><td>F1-Bio</td><td>BLEU-1</td><td>F1-Rad</td><td>F1-Bio</td><td>F1-Bio</td><td>BLEU-1</td><td>F1-Rad</td><td>F1-Bio</td></tr><tr><td>LLaVA 1.5-7B 1-shot</td><td>72.42</td><td>23.91</td><td>68.45</td><td>17.87</td><td></td><td>22.52</td><td></td><td>22.99</td><td>4.17</td><td></td><td>68.19</td><td>76.07</td><td>73.11</td><td>11.48</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>51.98</td><td></td><td></td><td></td><td>71.84</td><td></td><td>72.93</td><td></td><td></td><td></td><td>54.40</td><td>68.85</td><td>28.96 23.58</td><td></td><td></td><td>9.25</td><td>67.83</td><td>72.47</td><td>21.29</td><td>6.71</td><td>68.01</td></tr><tr><td>LLaVA 1.5-7B 2-shot</td><td>74.40</td><td>6.14</td><td>50.39</td><td>3.98</td><td>51.28</td><td>5.24</td><td>52.83</td><td>6.56</td><td>2.44</td><td></td><td></td><td>75.44</td><td>67.65</td><td>9.44</td><td></td><td>4.69</td><td>45.58</td><td>57.16</td><td>9.16</td><td>3.57</td><td>49.99</td></tr><tr><td>InternVL 3-8B 1-shot</td><td></td><td>27.94 30.00</td><td>75.70 76.24</td><td>28.50 30.14</td><td>74.89</td><td>28.06</td><td>74.92</td><td>28.32</td><td>9.18</td><td>67.77</td><td></td><td>30.32</td><td>69.46</td><td>7.04</td><td>8.63</td><td></td><td>63.21 63.76</td><td>74.13 74.76</td><td>25.03 26.62</td><td>8.91</td><td>65.49</td></tr><tr><td>InternVL 3-8B 2-shot</td><td>75.48 77.09</td><td>38.61</td><td>75.65</td><td>30.27</td><td>75.69 76.54</td><td>30.34 33.60</td><td>75.61 76.76</td><td>30.27 34.11</td><td>9.32 11.98</td><td>68.25 75.19</td><td>76.11</td><td>31.96 32.17</td><td>69.45 73.04</td><td>6.98 10.67</td><td>8.31 7.85</td><td></td><td>67.36</td><td>76.01</td><td>29.91</td><td>8.82 9.92</td><td>66.01 71.27</td></tr><tr><td>lingshu-7B 1-shot lingshu-7B 2-shot</td><td>77.55</td><td>39.27</td><td>76.18</td><td>30.57</td><td>76.74</td><td>34.79</td><td>77.21</td><td>34.89</td><td>11.53</td><td></td><td>74.36</td><td>76.97 77.25 33.63</td><td>72.88</td><td>10.54</td><td>7.67</td><td></td><td>66.75</td><td>76.30</td><td>30.61</td><td>9.60</td><td>70.55</td></tr><tr><td>Hulu-Med-7B 1-shot</td><td>76.93</td><td>31.16</td><td>75.42</td><td>24.08</td><td>76.10</td><td>29.40</td><td>76.56</td><td>29.89</td><td>13.86</td><td></td><td>75.77</td><td>76.91 26.78</td><td>72.87</td><td>10.58</td><td>8.86</td><td></td><td>64.90</td><td>75.80</td><td>25.32</td><td>11.36</td><td>70.34</td></tr><tr><td>Hulu-Med-7B 2-shot</td><td>77.04</td><td>32.19</td><td>75.83</td><td>23.07</td><td>76.25</td><td>30.06</td><td>76.76</td><td>29.04</td><td>14.50</td><td>76.32</td><td>77.01</td><td>29.95</td><td>72.85</td><td>10.36</td><td></td><td>8.29</td><td>65.52</td><td>75.96</td><td>25.78</td><td>11.40</td><td>70.92</td></tr><tr><td>CODA-Qwen3-8B</td><td>80.82</td><td>52.45</td><td>82.21</td><td>51.69</td><td>81.02</td><td>51.97</td><td>80.21</td><td>49.19</td><td>17.49</td><td>79.03</td><td>79.57</td><td>46.42</td><td>75.37</td><td>15.65</td><td>25.52</td><td></td><td>73.93</td><td>79.87</td><td>44.56</td><td>21.51</td><td>76.48</td></tr></table>

## E.4 Hyperparameter Study

We conduct a sensitivity study on two key hyperparameters, $\lambda _ { g }$ and $\lambda _ { d } ,$ , with results shown in Figure 7. As observed in Figure 7(a–c), performance consistently peaks at moderate values of $\lambda _ { g }$ (around 0.3–0.5), highlighting the importance of balancing the generation objective and cohort-aware signals. Both smaller and larger values lead to performance degradation, corresponding to the under- and over-dominance of the generation objective, respectively.

In contrast, Figure 7(d–f) shows that the model is relatively insensitive to $\lambda _ { d } ,$ with stable performance across several orders of magnitude. The best results are typically achieved around $1 0 ^ { - 4 } \mathrm { o r } 1 0 ^ { - 1 }$ , suggesting that the drift loss primarily acts as a regularizer.

Overall, these results demonstrate that our method achieves robust performance without requiring precise hyperparameter tuning.

![](images/ad4679fbf0f7f1185a89cb798217b9b289360320ac55cccf0e3f75153ea85278.jpg)  
(a) Hyperparameter study of $\lambda _ { g }$ under Closed-QA setting.

![](images/09edac930896a58497ae78378f5d0121bd7b860c282a748c344983d37a23bb48.jpg)  
(b) Hyperparameter study of $\lambda _ { g }$ under Open-QA setting.

![](images/802b735e8dc9f4061983934a27a6404b50e056bcd3b021143c2afc335e44e55d.jpg)  
(c) Hyperparameter study of $\lambda _ { g }$ under Report Generation setting.

![](images/02958c70f98db048da6cbe21a15aa724ac4fc5fa97e1210a9d682724600f4bbb.jpg)  
(d) Hyperparameter study of $\lambda _ { d }$ under Closed-QA setting.

![](images/8152a2ae7799dfd42da21eb7ec46fd5247ce8741c9290b1bed651238a8923eef.jpg)  
(e) Hyperparameter study of $\lambda _ { d }$ under Open-QA setting.

![](images/a51fe544ed9b258653fe0cd0c3fdea5058335b754329099eeb85ef6e17cae718.jpg)  
(f) Hyperparameter study of $\lambda _ { d }$ under Report Generation setting.  
Figure 7: Hyperparameter study on $\lambda _ { g }$ and $\lambda _ { d }$

## F Case Study

![](images/93453fe232b908df9a39ea4d6ebe8f796d3e1f457a878037180bd8aabcf27db4.jpg)  
Figure 8: A case study of Transfers task under Closed-QA setting.

![](images/05c1e4a53b57c21c067e9af0ea984bce5818584b4e8294ccfda0962c0ed6bcb0.jpg)  
Figure 9: A case study of PadChest dataset under Report Generation setting.

## G Broader Impacts

Current medical foundation models overlook patient-specific characteristics, which are critical for accurate clinical decision-making. By explicitly modeling cohort structure and individual-level deviations, CODA has the potential to support clinicians in delivering more personalized care, particularly in settings with heterogeneous patient populations, and to enhance transparency by exposing cohort-level patterns and patient-specific factors that influence model predictions.

At the same time, several risks should be considered. First, cohort construction may inadvertently encode biases present in the underlying data, potentially leading to disparate performance across demographic groups. Second, reliance on structured features and summary statistics may limit applicability in scenarios where such information is incomplete or noisy. Third, as with other generative models, there remains a risk of producing plausible but incorrect outputs, which could be misinterpreted in high-stakes clinical contexts.

To mitigate these risks, future work should incorporate fairness-aware cohort construction, uncertainty estimation, and human-in-the-loop validation to ensure safe deployment. In addition, rigorous evaluation across diverse populations and clinical settings is necessary to assess generalizability and prevent unintended harms. Overall, while CODA offers a step toward more interpretable and patient-aware medical AI, careful consideration of these societal implications is essential for responsible use.