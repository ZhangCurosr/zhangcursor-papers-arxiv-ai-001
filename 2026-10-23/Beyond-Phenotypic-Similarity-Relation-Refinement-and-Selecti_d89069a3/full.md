# Beyond Phenotypic Similarity: Relation Refinement and Selective Population Graph Learning for Multimodal Brain Disorder Diagnosis

Yinyu Lan

School of Medicine and Health, Harbin Institute of Technology

Harbin 150001, China

lanyinyu19@mails.ucas.ac.cn

Abstract—Population graph learning supports brain disorder diagnosis by sharing information across subjects. Its effectiveness, however, depends on whether the relationships guiding this exchange are relevant to diagnosis. Similarity in demographic or acquisition attributes does not necessarily indicate diagnostic agreement, and heterogeneous subjects may require different representation updates within the same population graph. We propose PopDRP, a framework that combines diagnosis-guided relation refinement with hierarchical selective state-space learning. Using only training labels, Contrastive Affinity Refinement calibrates attribute-based affinities according to diagnostic agreement. Neural Graph Rectification then refines these relationships using fused functional-connectivity and non-imaging representations, guided by graph reconstruction and dual-view alignment objectives. On the resulting graph, a Selective Pyramid State-Space backbone integrates input-dependent state updates, graph diffusion, and hierarchical pooling and unpooling to learn population representations at multiple resolutions. Together, these components connect the refinement of inter-subject relationships with selective representation learning over the population graph. Experiments on ABIDE and ADHD-200 under stratified tenfold cross-validation show improved accuracy and AUC over the compared baselines. Ablation studies further support the contributions of relation refinement and the selective hierarchical backbone to classification performance.

Index Terms—Population graph learning, brain disorder diagnosis, multimodal learning, graph structure refinement, selective state-space models.

## I. INTRODUCTION

Population graph learning provides a way to incorporate inter-subject information into brain disorder diagnosis [1], [2]. In this formulation, each node represents a subject, node features describe individual measurements such as resting-state functional connectivity, and edges determine how information is shared across the cohort. Non-imaging attributes, including age, sex, and acquisition site, provide additional cues for constructing these relationships [2], [3]. The value of a population graph therefore depends not only on the quality of individual features, but also on whether the relationships connecting subjects support the classification task.

A central modeling difficulty is that attribute similarity and diagnostic relevance are not interchangeable. Two subjects may share demographic characteristics or an acquisition site while having different diagnoses. Attribute-based affinities consequently provide useful relational priors, but their implications for information sharing must be considered in relation to the task. Existing research has already moved beyond fixed graph construction: EV-GCN learns adaptive inter-subject associations [3], while MM-GTUNets incorporates label-informed affinity modeling and hierarchical graph encoding [4]. Building on these advances, we investigate how diagnosis-guided affinities can be further refined using multimodal representations and incorporated into a selective hierarchical population encoder.

![](images/bdf6112905b100b607cd4c978a1ebf7dca21f3a6180be2f640b726c8f02f5f69.jpg)  
Fig. 1. Multimodal brain disease diagnosis based on population graph.

Relation refinement and population representation learning are closely connected. Edge weights determine which subjects influence one another, while the encoder determines how the resulting information is transformed. A refined graph provides a basis for information exchange, but heterogeneous subject features still motivate input-dependent representation updates. Selective state-space models offer a mechanism for conditioning state transformations on the input, and their adaptation to graph learning has motivated investigation into graph-aware information processing [5]–[7]. Hierarchical graph processing introduces a further consideration: representations should integrate information at different population resolutions while retaining features needed for individual predictions [4], [8]. These considerations motivate a framework that combines feature-conditioned relation refinement with selective state updates throughout a graph hierarchy.

We present PopDRP to implement this design through three connected stages. First, Contrastive Affinity Refinement (CAR) uses diagnostic agreement among training subjects to calibrate non-imaging affinities, producing a diagnosis-guided population prior. Second, Neural Graph Rectification (NGR) combines this prior with fused imaging and non-imaging features to learn refined edge weights. Graph reconstruction and dual-view alignment constrain this process. Third, the Selective Pyramid State-Space (SPSS) backbone processes the refined graph through input-dependent state updates and topology-aware diffusion within a pooling and unpooling hierarchy. Skip connections retain subject-level information as graph resolution changes. The resulting architecture links the refinement of population relationships to selective representation learning across the hierarchy.

The contributions of this work are threefold:

• We develop a population graph refinement pipeline that combines training-label-informed affinities with featureconditioned edge learning. This allows multimodal subject representations to refine the relationships used for downstream classification.

• We introduce a Selective Pyramid State-Space backbone that integrates input-dependent state updates and graph diffusion within a hierarchical encoder–decoder. It learns population representations at multiple resolutions while retaining information for subject-level prediction.

• We evaluate PopDRP on ABIDE and ADHD-200 using stratified ten-fold cross-validation. Comparisons with established baselines and component ablations assess the effectiveness of the framework and the contributions of its principal components.

## II. RELATED WORK

## A. Population Graph-Based Brain Disease Diagnosis

Parisot et al. pioneered population graph learning for brain disease diagnosis via handcrafted non-imaging similarity and GCN-based classification [1], [2]. Follow-up works optimized this framework via multi-scale graph convolution [9], ROIaware graph construction [10], multimodal fusion, graph transformer and adversarial contrastive learning [3], [4], [11], [12]. Even so, most methods build population graphs with fixed handcrafted similarity metrics on raw non-imaging data, suffering severely from missing data, site effects and label noise.

## B. State-Space Models and Graph Mamba

SSMs outperform attention models for long-range sequence modeling with linear computational complexity. Classic S4 and Mamba capture long-range dependencies via continuous latent states and input-selective gating [5], [13]. Extended graph Mamba methods serialize graph nodes into sequences to fit Mamba blocks for graph learning [6], [7], yet they only support single-scale modeling without hierarchical graph pooling. To tackle this issue, our SPSS backbone embeds selective SSM units into a U-Net pyramid structure and adopts graph diffusion for topology-aware state propagation, supporting joint multi-scale and long-range representation learning for population graphs.

## C. Graph Denoising and Structure Learning

Graph structure learning jointly optimizes graph topology and node embeddings [14]. Existing graph rectification methods refine graph edges relying on node features or inherent topology [15], [16], which are tailored for homogeneous graphs and fail to introduce diagnostic label supervision. Besides, population graphs face unique noise from defective non-imaging attributes and site confounding factors. Our NGR module addresses this gap by modeling edge feature interactions and leveraging clinical label priors from CAR-optimized graphs under a minimum-sufficient-representation objective.

## III. METHOD

## A. Problem Formulation

Let $\mathbf { \mathcal { D } } \mathbf { \Psi } = \mathbf { \Psi } ( \mathbf { X } _ { i } , \mathbf { p } i , y _ { i } )  \} i = 1 ^ { N }$ be a cohort of N subjects, where $\mathbf { X } _ { i } \in \mathbb { R } ^ { D }$ is the vectorized upper-triangular functional connectivity feature, $\mathbf { p } _ { i } \in \mathbb { R } ^ { S }$ denotes non-imaging attributes (e.g., age, gender, and site), and $y _ { i } \in 0 ,$ 1 is the diagnostic label. We first construct a population graph $\mathcal { G } = ( \gamma , \mathcal { E } , \mathbf { A } )$ from multimodal imaging and non-imaging information, where each subject is represented as a node and edges encode inter-subject relationships. Brain disease diagnosis is then formulated as node classification over the learned population graph, $\mathrm { i . e . , } ( \mathbf { X } , \mathbf { P } ) \to \mathcal { G } \to { \hat { \mathbf { y } } }$

## B. Overview of PopDRP

PopDRP consists of four stages: (1) multimodal preprocessing (RFE for imaging, VAE for non-imaging), (2) Contrastive Affinity Refinement (CAR) to build a clinically informed graph, (3) Neural Graph Rectification (NGR) to denoise the graph via minimum-sufficient representation, and (4) Selective Pyramid State-Space (SPSS) backbone for hierarchical representation, followed by MLP classification.

## C. Multimodal Feature Preprocessing

1) Imaging Feature Encoding: We vectorize the uppertriangular part of each connectivity matrix and apply recursive feature elimination (RFE) with a ridge classifier to select the top D discriminative features using only training data:

$$
\mathbf { X } = \mathrm { R F E } \left( \mathbf { X } _ { \mathrm { r a w } } , \mathbf { y } _ { \mathrm { t r a i n } } , D \right) , \quad \mathbf { X } \in \mathbb { R } ^ { N \times D } .\tag{1}
$$

2) Phenotypic Feature Encoding: A variational autoencoder (VAE) is trained unsupervised on $\mathbf { P } ~ \in ~ \mathbb { R } ^ { N \times S }$ . The encoder produces $\mu _ { i } , \log \pmb { \sigma } _ { i } ^ { 2 } = \mathrm { M L P } _ { \mathrm { e n c } } ( \mathbf { p } _ { i } )$ , and we sample $\mathbf { z } _ { i } ~ = ~ \pmb { \mu } _ { i } + \pmb { \sigma } _ { i } \odot \pmb { \epsilon }$ via reparameterization. The decoder $\hat { \bf p } _ { i } = { \bf M L P _ { \mathrm { d e c } } } ( { \bf z } _ { i } )$ reconstructs the input. The VAE loss is:

$$
\mathcal { L } _ { \mathrm { V A E } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \| \hat { \mathbf { p } } _ { i } - \mathbf { p } _ { i } \| _ { 2 } ^ { 2 } + \beta \cdot D _ { \mathrm { K L } } \left( \mathcal { N } ( \mu _ { i } , \sigma _ { i } ^ { 2 } ) \| \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) \right) .\tag{2}
$$

After pretraining, latent embeddings $\mathbf { Z } \ = \ [ \mathbf { z } _ { 1 } , \ldots , \mathbf { z } _ { N } ] ^ { \top } \ \in$ $\mathbb { R } ^ { N \times D ^ { \bullet } }$ serve as denoised non-imaging features; the encoder is frozen thereafter.

![](images/ac2438d84582e036bf66a29462382c89ed37b16169438f70697d1df38e492cd0.jpg)  
Fig. 2. (A) The framework of PopDRP. (B) Contrastive Affinity Refinement (CAR) module. (C) Neural Graph Rectification (NGR) module. (D) Architecture of the Selective Pyramid State-Space (SPSS).

## D. Contrastive Affinity Refinement (CAR)

For each non-imaging attribute s, we decompose pairwise similarity into three matrices using only training labels. For continuous attributes (threshold t):

$$
\mathbf { P } _ { i j } ^ { ( s ) } = \mathbf { 1 } [ i , j \in \mathcal { V } _ { \mathrm { t r a i n } } , | x _ { i } ^ { ( s ) } - x _ { j } ^ { ( s ) } | < t , y _ { i } = y _ { j } ] ,\tag{3}
$$

$$
\mathbf { N } _ { i j } ^ { ( s ) } = \mathbf { 1 } [ i , j \in \mathscr { V } _ { \mathrm { t r a i n } } , | x _ { i } ^ { ( s ) } - x _ { j } ^ { ( s ) } | < t , y _ { i } \neq y _ { j } ] ,\tag{4}
$$

$$
{ \bf C } _ { i j } ^ { ( s ) } = { \bf 1 } [ i , j \in \mathcal { V } _ { \mathrm { v a l } } \cup \mathcal { V } _ { \mathrm { t e s t } } , | x _ { i } ^ { ( s ) } - x _ { j } ^ { ( s ) } | < t ] .\tag{5}
$$

For categorical attributes, replace $| x _ { i } ^ { ( s ) } - x _ { j } ^ { ( s ) } | < t$ with $x _ { i } ^ { ( s ) } =$ $x _ { j } ^ { ( s ) }$ . CAR computes a refined affinity per attribute:

$$
\tilde { \mathbf { A } } ^ { ( s ) } = \alpha _ { s } ( \mathbf { P } ^ { ( s ) } - \mathbf { N } ^ { ( s ) } ) + \beta _ { s } + \gamma _ { s } \mathbf { C } ^ { ( s ) } ,\tag{6}
$$

with $\alpha _ { s } , \gamma _ { s } = \mathrm { S o f t p l u s } ( \cdot ) > 0$ . Learnable importance weights $w _ { s } = \mathrm { s o f t m a x } ( \hat { w } _ { s } )$ aggregate:

$$
{ \bf A } _ { \mathrm { c a r } } = \sigma \left( \frac { 1 } { \tau } \sum _ { s = 1 } ^ { S } w _ { s } \tilde { \bf A } ^ { ( s ) } \right) , \quad \tau = | \hat { \tau } | > 0 .\tag{7}
$$

The final symmetric adjacency with self-loops is:

$$
\hat { \mathbf { A } } _ { \mathrm { c a r } } = \frac { 1 } { 2 } ( \mathbf { A } _ { \mathrm { c a r } } + \mathbf { A } _ { \mathrm { c a r } } ^ { \top } ) , \hat { \mathbf { A } } _ { \mathrm { c a r } } [ i , i ] = 1 .\tag{8}
$$

E. Graph Construction and Neural Graph Rectification (NGR) We construct fused node features $\mathbf { H } = [ \mathbf { X } , \mathbf { Z } ] \in \mathbb { R } ^ { N \times 2 D }$ A Gaussian RBF similarity graph is:

$$
\mathbf { A } _ { \mathrm { s i m } } = \exp \left( - \frac { \| \mathbf { h } _ { i } - \mathbf { h } _ { j } \| _ { 2 } ^ { 2 } } { 2 \sigma ^ { 2 } } \right) .\tag{9}
$$

The target graph combines similarity and clinical prior: ${ \bf A } _ { \mathrm { t a r } } =$ $\mathbf { A } _ { \mathrm { s i m } } \odot \hat { \mathbf { A } } _ { \mathrm { c a r } }$ . NGR learns a minimum sufficient representation $\mathbf { U } = f ( \mathbf { H } )$ by optimizing:

$$
\mathcal { L } _ { \mathrm { g r a p h } } = \underbrace { \mathbf { M } \mathbf { S } \mathbf { E } ( \mathbf { A } _ { \mathrm { d e n } } , \mathbf { A } _ { \mathrm { t a r } } ) } _ { \mathcal { L } _ { \mathrm { r e c o n } } } + \underbrace { \left( 1 - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { \mathbf { U } _ { i } \cdot \mathbf { U } _ { \mathrm { r e f } , i } } { \| \mathbf { U } _ { i } \| \| \mathbf { U } _ { \mathrm { r e f } , i } \| } \right) } _ { \mathcal { L } _ { \mathrm { r e c o n } } } ,\tag{10}
$$

where $\mathbf { U } _ { \mathrm { r e f } }$ is a second encoding with detached gradients. The rectified graph $\mathbf { A } _ { \mathrm { d e n } }$ is predicted from U via:

$$
\begin{array} { r } { \mathbf { A } _ { \mathrm { d e n } } ^ { i j } = \sigma \big ( \mathbf { M } \mathbf { L } \mathbf { P } _ { \mathrm { e d g e } } \big ( \big [ \mathbf { U } _ { i } ; \mathbf { U } _ { j } ; | \mathbf { U } _ { i } - \mathbf { U } _ { j } | ; \mathbf { U } _ { i } \odot \mathbf { U } _ { j } ; \mathbf { A } _ { \mathrm { t a r } , i j } \big ] \big ) \big ) , } \end{array}\tag{11}
$$

then symmetrized, masked by $\mathbf { A } _ { \mathrm { t a r } }$ , and filled with self-loops.

## F. Selective Pyramid State-Space (SPSS)

The rectified graph $\mathbf { A } _ { \mathrm { d e n } }$ serves as computation basis for SPSS, the core module of PopDRP. SPSS is designed following three neuroscience-derived principles: local message passing captures within-subsystem brain region interactions; selective state-space recurrence models cross-population longrange sequential dependencies; hierarchical pyramid pooling/unpooling constructs multi-scale representations matching hierarchical human brain functional organization.

1) SPSS Architecture: SPSS adopts a U-Net structure with L encoder layers, one bottleneck layer, and L decoder layers. Denote input node feature $\mathbf { H } ^ { ( 0 ) } \ = \ \mathbf { H }$ and original graph ${ \mathcal G } ^ { ( 0 ) } = ( \dot { \mathcal { V } } ^ { ( 0 ) } , { \mathcal E } ^ { ( 0 ) } )$ built on $\mathbf { A } _ { \mathrm { d e n } }$ . For each encoder layer $\ell = 0 , \ldots , L - 1$ , node features are updated via GraphStateCell before pooling:

$$
\mathbf { H } ^ { ( \ell ) } \gets \mathrm { G r a p h S t a t e C e l l } _ { \mathrm { d o w n } } ^ { ( \ell ) } \left( \mathbf { H } ^ { ( \ell ) } , \mathcal { G } ^ { ( \ell ) } \right) ,\tag{12}
$$

$$
\mathbf { H } ^ { ( \ell + 1 ) } , \mathcal { G } ^ { ( \ell + 1 ) } , \mathbf { P } ^ { ( \ell ) } \gets \mathrm { T o p K P o o l } \left( \mathbf { H } ^ { ( \ell ) } , \mathcal { G } ^ { ( \ell ) } \right) ,\tag{13}
$$

where $\mathbf { P } ^ { ( \ell ) }$ stores indices of preserved nodes. TopKPool retains the top-r ratio nodes by projection score to allocate computation to informative nodes. After L pooling steps, the coarsened graph contains global population structural information, processed by bottleneck GraphStateCell:

$$
\mathbf { H } ^ { ( L ) } \gets \mathrm { G r a p h S t a t e C e l l } _ { \mathrm { b o t t l e n e c k } } \left( \mathbf { H } ^ { ( L ) } , \mathcal { G } ^ { ( L ) } \right) .\tag{14}
$$

Decoders restore graph resolution via unpooling and lateral skip GraphStateCell fusion:

$$
\mathbf { H } _ { \mathrm { u p } } ^ { ( \ell ) } \gets \mathrm { U n p o o l } \left( \mathbf { H } ^ { ( \ell + 1 ) } , \mathbf { P } ^ { ( \ell ) } \right) ,\tag{15}
$$

$$
\mathbf { H } ^ { ( \ell ) } \gets \mathrm { G r a p h S t a t e C e l l } _ { \mathrm { l a t e r a l } } ^ { ( \ell ) } \left( \mathbf { H } ^ { ( \ell ) } , \mathcal { G } ^ { ( \ell ) } \right) ,\tag{16}
$$

$$
\mathbf { H } ^ { ( \ell ) } \gets \mathrm { G r a p h S t a t e C e l l } _ { \mathrm { u p } } ^ { ( \ell ) } \left( \mathbf { H } _ { \mathrm { u p } } ^ { ( \ell ) } + \mathbf { H } ^ { ( \ell ) } , \mathcal { G } ^ { ( \ell ) } \right) .\tag{17}
$$

Unpool maps coarse features back to original node positions with zero padding for pruned nodes; lateral skip connections recover fine-grained local structures lost during pooling.

2) Selective State-Space with Graph Diffusion (GraphState-Cell): GraphStateCell is SPSS’s fundamental computation unit, integrating graph attention convolution, input-selective SSM recurrence, and sparse graph diffusion to align hidden states with graph topology. Layer-normalized input first passes a TransformerConv + ReLU block:

$$
\mathbf { H } _ { \mathrm { c o n v } } = \mathrm { R e L U } \left( \mathrm { T r a n s f o r m e r C o n v } \left( \mathrm { L a y e r N o r m } ( \mathbf { H } ) , \mathbf { A } _ { \mathrm { d e n } } \right) \right)\tag{18}
$$

a) State-space projection: Convolution outputs are linearly split into SSM input branch U and residual branch R:

$$
[ \mathbf { U } , \mathbf { R } ] = \mathbf { H } _ { \mathrm { c o n v } } \mathbf { W } _ { \mathrm { i n } } , \quad \mathbf { U } , \mathbf { R } \in \mathbb { R } ^ { N \times d _ { \mathrm { i n n e r } } } .\tag{19}
$$

b) State Parameter Discretization: Unlike conventional fixed-parameter SSMs, SPSS generates input-adaptive transition parameters ∆, B, C:

$$
[ \Delta ^ { \prime } , \mathbf { B } , \mathbf { C } ] = \mathbf { U } \mathbf { W } _ { x } , \quad \Delta = \mathrm { S o f t p l u s } ( \Delta ^ { \prime } \mathbf { W } _ { \Delta } ) .\tag{20}
$$

Diagonal state matrix $\textbf { A } = \mathbf { \Gamma } - \exp ( \mathbf { A } _ { \mathrm { l o g } } )$ uses learnable weights to stabilize long-range state propagation.

c) Recurrent State Update: Hidden states are iteratively updated for K steps with zero initialization $\mathbf { S } _ { 0 } = \mathbf { 0 }$

$$
\begin{array} { r } { \mathbf { S } _ { k } = \exp ( \Delta \otimes \mathbf { A } ) \odot \mathbf { S } _ { k - 1 } + \mathbf { U } \otimes \mathbf { B } , } \end{array}\tag{21}
$$

where ⊗ denotes outer product and ⊙ element-wise multiplication. Iterative refinement accumulates multi-step contextual information per node.

d) Sparse Graph Diffusion: Post-recurrence hidden states diffuse over graph edges for neighborhood aggregation:

$$
\begin{array} { r } { \tilde { \bf S } _ { K } = \tilde { \bf A } _ { \mathrm { s y m } } \cdot \mathrm { f l a t t e n } ( { \bf S } _ { K } ) , } \end{array}\tag{22}
$$

with symmetric normalized adjacency matrix $\begin{array} { r l } { \tilde { \mathbf { A } } _ { \mathrm { s y m } } } & { { } = } \end{array}$ $\mathbf { D } ^ { - 1 / 2 } \bar { ( } \mathbf { A } _ { \mathrm { d e n } } + \mathbf { I ) } \mathbf { D } ^ { - 1 / 2 }$ . Sparse matrix multiplication avoids $\mathcal { O } ( N ^ { 2 } )$ dense memory cost. Preliminary SSM output fuses diffused states, projection C and channel bias D:

$$
\begin{array} { r } { \mathbf { Y } _ { \mathrm { s s m } } = ( \tilde { \mathbf { S } } _ { K } \odot \mathbf { C } ) + \mathbf { U } \cdot \mathbf { D } , } \end{array}\tag{23}
$$

$\mathbf { D } \in \mathbb { R } ^ { d _ { \mathrm { i n n e r } } }$ is learnable channel bias.

e) Residual gating: A temperature gate balances longrange SSM and local residual signals:

$$
\mathbf { G } = \sigma \left( \frac { [ \mathbf { Y } _ { \mathrm { s s m } } , \mathbf { R } ] \mathbf { W } _ { \mathrm { g a t e } } } { \tau _ { g } } \right) ,\tag{24}
$$

$$
\mathbf { Y } _ { \mathrm { o u t } } = \left( \mathbf { Y } _ { \mathrm { s s m } } + \mathbf { G } \odot \mathrm { R e L U } ( \mathbf { R } ) \right) \mathbf { W } _ { \mathrm { o u t } } .\tag{25}
$$

Standard residual connection with dropout yields final cell output:

$$
\mathbf { Y } _ { \mathrm { c e l l } } = \mathbf { H } + \mathrm { D r o p o u t } ( \mathbf { Y } _ { \mathrm { o u t } } ) .\tag{26}
$$

Conventional SSMs use shared fixed (∆, A, B, C) for all samples, leading to uniform feature compression that fails to model heterogeneous brain connectivity across neurodevelopmental disorder subjects. SPSS dynamically computes parameters per input, enabling subject-adaptive feature selection for diverse multimodal brain graphs.

## G. Classification and Training Objective

SPSS output embeddings $\mathbf { H } _ { \mathrm { f i n a l } } \in \mathbb { R } ^ { N \times d _ { \mathrm { o u t } } }$ feed a two-layer BN-ReLU MLP classifier:

$$
\hat { \mathbf { y } } = \mathrm { S o f t m a x } \left( \mathbf { W } _ { 2 } \cdot \mathrm { B N } \left( \mathrm { R e L U } \left( \mathbf { W } _ { 1 } \mathbf { H } _ { \mathrm { f i n a l } } + \mathbf { b } _ { 1 } \right) \right) + \mathbf { b } _ { 2 } \right) .\tag{27}
$$

Total training loss combines supervised cross-entropy and graph smoothness regularization:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = - \frac { 1 } { | \mathcal { Z } _ { \mathrm { t r a i n } } | } \sum _ { i \in \mathcal { T } _ { \mathrm { t r a i n } } } \sum _ { c = 0 } ^ { 1 } \mathbb { I } [ y _ { i } = c ] \log \hat { y } _ { i , c } + \lambda _ { \mathrm { g r a p h } } \cdot \mathcal { L } _ { \mathrm { g r a p h } } ,\tag{28}
$$

where $\mathcal { L } _ { \mathrm { g r a p h } }$ follows Eq. (10), and $\lambda _ { \mathrm { g r a p h } }$ weights regularization strength. All model parameters are end-to-end optimized via Adam.

## IV. EXPERIMENTS

## A. Datasets and Preprocessing

We evaluate PopDRP on two widely used public neurodevelopmental disorder datasets.

ABIDE Dataset: The Autism Brain Imaging Data Exchange (ABIDE) dataset [17] aggregates rs-fMRI and nonimaging data from 17 international imaging sites. Following standard preprocessing with the Configurable Pipeline for the Analysis of Connectomes (CPAC), we obtain AAL atlas-based region-averaged time series for 116 regions of interest (ROIs) from 871 subjects (468 healthy controls, 403 individuals with

TABLE I  
HYPERPARAMETER SETTINGS FOR THE ABIDE AND ADHD-200 DATASETS.
<table><tr><td>Hyperparameter</td><td>ADHD-200</td><td>ABIDE</td></tr><tr><td>SPSS Depth</td><td>4</td><td>1</td></tr><tr><td>State dimension</td><td>64</td><td>64</td></tr><tr><td>Mask rate</td><td>0.2</td><td>0.8</td></tr><tr><td>Hidden dimension</td><td>64</td><td>512</td></tr><tr><td>Learning rate</td><td>10⁻4</td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>Edge dropout</td><td>0.3</td><td>0.3</td></tr><tr><td>Weight decay</td><td> $5 \times 1 0 ^ { - 4 }$ </td><td> $5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Epoch</td><td>500</td><td>500</td></tr></table>

ASD). Functional connectivity matrices are computed as Pearson correlation coefficients between ROI time series, followed by Fisher-Z transformation. Age, gender, and scanning site are used as non-imaging attributes.

ADHD-200 Dataset: The ADHD-200 consortium dataset [18] provides rs-fMRI from 582 subjects (364 healthy controls, 218 individuals with ADHD) across multiple sites. Data are preprocessed using the Athena pipeline with nuisance signal removal. Functional connectivity matrices are computed identically to ABIDE using the AAL atlas. Age, gender, and site serve as non-imaging attributes.

For both datasets, we apply Fisher-Z normalization to connectivity values, extract the upper triangular portion of each $1 1 6 \times 1 1 6$ connectivity matrix as a 6670-dimensional feature vector, and apply RFE to select the D = 500 most discriminative features. Stratified 10-fold cross-validation is employed, with an additional 10% of training data held out as a validation set for each fold.

## B. Evaluation Metrics and Baselines

We evaluate performance using five metrics: classification accuracy (ACC), sensitivity (SEN), specificity (SPE), area under the ROC curve (AUC), and F1-score. We compare PopDRP against three categories of methods:

We compare PopDRP against three categories of baselines. Traditional machine learning methods: SVM with RBF kernel and MLP applied to the RFE-selected connectivity features. Brain-graph-based methods: Brain-GNN [10], DGCN [19], AL-NEGAT [20], and A-GCL [12], which construct individual-level brain graphs and apply GNNs for classification. Population-graph-based methods: Pop-GCN [2], GATE [21], EV-GCN [3], MMGL [11], and MM-GTUNets [4], which model inter-subject relationships as population graphs. To ensure consistency, the same data splits and evaluation protocol are adopted.

## C. Implementation Details

The VAE used for non-imaging feature encoding includes a two-layer MLP encoder and a two-layer MLP decoder. We pre-train this VAE for a maximum of 3000 epochs and adopt early stopping with a patience value of 100. The CAR module applies learnable affine transformations $\alpha _ { s } , \beta _ { s } , \gamma _ { s }$ to each attribute across three contrastive decomposition channels for positive, negative and cross-split signals, while the learnable edge temperature τ is initialized to 1.0. Neural Graph Rectification (NGR) uses a shared two-layer node encoder to produce dual-view representations, along with a two-layer edge decoder. GELU is used as the activation function for both modules. The dual-view learning objective combines MSE loss for reconstruction and cosine similarity for alignment, and it requires no extra hyperparameters. The hyperparameter settings for the experiment are shown in Table I.

## D. Performance Comparison

As shown in Table II, traditional machine learning methods including SVM and MLP achieve relatively weak overall performance across all metrics. These methods lack the ability to model complex pairwise relationships between subjects, which limits their effectiveness for brain disease diagnosis tasks. Methods built upon brain graphs deliver better results than classic machine learning approaches, yet they still cannot fully exploit cross-subject information for accurate prediction.

Recent population-graph methods generally outperform earlier solutions, as they explicitly model connections among different subjects and make full use of population-level data. Among these competitors, MM-GTUNets obtains competitive results and ranks second on most evaluation metrics. Even so, our PopDRP achieves the best performance in most cases on both datasets. It yields higher accuracy, sensitivity and AUC values than all compared methods, and maintains a favorable specificity level. The consistent improvements demonstrate that the designed graph refinement and multi-scale representation learning modules effectively capture reliable populationlevel patterns, leading to more robust and accurate diagnosis.

## E. Ablation Study

We conduct ablation experiments on the ADHD-200 and ABIDE datasets to quantify the contribution of each key component in our PopDRP framework. The detailed results are presented in Table III.

We first remove the NGR module and keep all other components unchanged. The model achieves lower accuracy and AUC on both datasets, which verifies that neural graph rectification effectively eliminates spurious edges and improves the quality of learned graph structures. When the CAR module is replaced by simple averaging over non-imaging similarities of individual attributes, overall performance drops noticeably. This outcome indicates that contrastive affinity refinement can produce more reliable graph priors compared with naive aggregation strategies.

To examine the effect of the SPSS backbone, we substitute it with a standard graph transformer. The resulting performance experiences a substantial decline, demonstrating that the proposed selective pyramid state-space design is essential for capturing multi-scale topological features and long-range node dependencies. We also test the model by directly feeding raw non-imaging features instead of embeddings learned by VAE. The degraded performance shows that the VAE module learns denoised and compact representations that benefit downstream diagnosis.

TABLE II  
PERFORMANCE COMPARISON WITH THREE CATEGORIES OF BASELINES ON ABIDE AND ADHD-200 DATASETS (%). THE BEST RESULTS ARE MARKED IN BOLD AND THE SECOND-BEST ARE UNDERLINED.
<table><tr><td rowspan="2">Category</td><td rowspan="2">Method</td><td colspan="4">ABIDE</td><td colspan="4">ADHD-200</td></tr><tr><td>ACC</td><td>SEN</td><td>SPE</td><td>AUC</td><td>ACC</td><td>SEN</td><td>SPE</td><td>AUC</td></tr><tr><td rowspan="2">Traditional ML</td><td>SVM</td><td>63.21 (0.25)</td><td>57.05 (1.05)</td><td>68.50 (1.55)</td><td>67.43 (1.16)</td><td>64.67 (0.92)</td><td>33.95 (2.18)</td><td>83.05 (0.65)</td><td>66.30 (0.26)</td></tr><tr><td>MLP</td><td>63.71 (0.36)</td><td>56.28 (0.40)</td><td>70.06 (0.35)</td><td>68.48 (0.25)</td><td>65.30 (0.35)</td><td>40.30 (3.55)</td><td>80.20 (0.45)</td><td>67.33 (0.42)</td></tr><tr><td rowspan="4">Brain-graph</td><td>Brain-GNN</td><td>69.76 (3.80)</td><td>67.47 (3.10)</td><td>73.28 (3.26)</td><td>72.50 (3.10)</td><td>65.26 (3.60)</td><td>68.59 (3.30)</td><td>63.05 (4.00)</td><td>66.02 (5.50)</td></tr><tr><td>DGCN</td><td>71.83 (2.90)</td><td>70.90 (2.59)</td><td>71.80 (3.10)</td><td>72.45 (2.98)</td><td>68.81 (3.69)</td><td>69.08 (4.07)</td><td>69.53 (4.71)</td><td>70.05 (4.57)</td></tr><tr><td>AL-NEGAT</td><td>73.17 (3.32)</td><td>78.18 (2.66)</td><td>73.28 (3.14)</td><td>75.02 (2.56)</td><td>69.15 (3.82)</td><td>69.28 (4.32)</td><td>72.26 (3.92)</td><td>68.25 (3.69)</td></tr><tr><td>A-GCL</td><td>79.04 (2.40)</td><td>81.42 (3.03)</td><td>80.95 (3.19)</td><td>82.86 (2.91)</td><td>80.11 (4.30)</td><td>82.04 (4.58)</td><td>80.08 (4.10)</td><td>78.78 (4.39)</td></tr><tr><td rowspan="6">Population-graph</td><td>Pop-GCN</td><td>68.43 (0.86)</td><td>78.32 (0.89)</td><td>57.51 (5.99)</td><td>73.90 (3.08)</td><td>75.45 (0.32)</td><td></td><td>90.11 (0.22)</td><td>81.72 (1.40)</td></tr><tr><td>ĠATE</td><td></td><td></td><td></td><td></td><td>72.20 (0.23)</td><td>50.99 (1.96)</td><td></td><td></td></tr><tr><td></td><td>74.65 (2.50)</td><td>75.59 (2.43)</td><td>76.87 (2.27)</td><td>76.87 (2.27)</td><td></td><td>77.27 (5.20)</td><td>72.40 (3.78)</td><td>74.61 (3.30)</td></tr><tr><td>EV-GCN MMGL</td><td>80.95 (0.27) 80.34 (0.18)</td><td>83.74 (0.61)</td><td>77.69 (0.18) 77.16 (3.15)</td><td>82.37 (0.29)</td><td>80.95 (0.96) 77.59 (3.64)</td><td>60.33 (7.64)</td><td>93.11 (0.13)</td><td>87.15 (1.15)</td></tr><tr><td>MM-GTUNets</td><td>82.92 (0.54)</td><td>80.25 (0.21) 84.22 (0.81)</td><td>81.43 (0.64)</td><td>79.83 (1.06) 88.21 (0.61)</td><td>82.68 (0.60)</td><td>76.21 (2.98) 77.58 (2.11)</td><td>74.57 (1.25) 85.76 (0.46)</td><td>78.47 (2.29)</td></tr><tr><td>PopDRP</td><td>84.18 (0.66)</td><td>89.14 (0.44)</td><td>81.59 (1.30)</td><td>90.55 (0.61)</td><td>85.79 (1.11)</td><td>89.96 (1.56)</td><td>84.46 (1.10)</td><td>90.71 (0.72) 92.37 (1.22)</td></tr></table>

TABLE III  
ABLATION STUDY ON ADHD-200 AND ABIDE (%).
<table><tr><td rowspan="2">Variant</td><td colspan="2">ADHD-200</td><td colspan="2">ABIDE</td></tr><tr><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td></tr><tr><td>Full PopDRP</td><td>85.79 (1.11)</td><td>92.37 (1.22)</td><td>84.18 (0.66)</td><td>90.55 (0.61)</td></tr><tr><td>w/o NGR</td><td>82.38 (1.02)</td><td>90.18 (1.26)</td><td>80.11 (0.67)</td><td>86.25 (0.59)</td></tr><tr><td>w/o CAR</td><td>81.57 (1.40)</td><td>89.42 (1.35)</td><td>79.03 (0.68)</td><td>85.84 (0.67)</td></tr><tr><td>w/o SPSS</td><td>72.54 (0.00)</td><td>71.30 (0.68)</td><td>70.97 (0.13)</td><td>71.81 (0.39)</td></tr><tr><td>w/o VAE</td><td>69.61 (0.99)</td><td>88.98 (1.25)</td><td>80.73 (0.52)</td><td>87.33 (0.57)</td></tr><tr><td>w/o Graph Loss</td><td>80.73 (1.18)</td><td>88.48 (1.34)</td><td>80.80 (0.58)</td><td>87.79 (0.57)</td></tr></table>

Finally, we remove the graph loss term from the overall optimization objective. The slight performance reduction confirms that this regularization term helps constrain graph learning and further enhances model robustness. Compared with all simplified variants, the complete PopDRP system obtains the best results across all metrics, proving the necessity and mutual complementarity of all designed modules.

## F. Hyperparameter Sensitivity

We conducted a systematic hyperparameter sensitivity analysis on four key parameters: SPSS depth $L ,$ state dimension $d _ { \mathrm { s t a t e } } , \mathrm { N G R }$ mask rate, and hidden dimension. Each parameter was swept independently while holding all others at their default values, using 10-fold cross-validation on both datasets. Figure 4 presents the results for ADHD-200 and ABIDE separately.

The analysis reveals dataset-specific optimal configurations. On ADHD-200, accuracy increases monotonically with SPSS depth, from 84.2% at L = 1 to 85.8% at L = 4, suggesting that deeper state-space pyramids benefit datasets with lower site heterogeneity. The state dimension $d _ { \mathrm { s t a t e } } = 6 4$ provides the best accuracy-efficiency trade-off (85.1%), while smaller hidden dimensions (64) outperform larger ones (84.3% vs. 79.6% at 512), indicating that ADHD’s relatively homogeneous acquisition does not require excessive model capacity.

![](images/e5a18ea11eca3363557aff08f4d902081611c2225d94a70d6f0c4951ee0ddcd2.jpg)  
Fig. 3. t-SNE visualization of SPSS output embeddings. Diagnosis-colored panels show clear HC/Patient separation; site-colored panels show intermingled site labels, indicating partial mitigation of site effects.

Mask rates of 0.2 provide a modest 0.5 pp gain over the nomask baseline.

In striking contrast, ABIDE exhibits the opposite trend for SPSS depth: accuracy decreases from 83.3% at $L \ = \ 1$ to 80.8% at L = 4, indicating that deeper pyramid pooling destroys diagnostically valuable cross-site signal in the highly heterogeneous 17-site dataset. Larger hidden dimensions (512) yield the best ABIDE performance (84.2%), and higher mask rates (0.6–0.8) provide 0.6–0.8 pp gains, consistent with the hypothesis that stronger regularization benefits multi-site robustness. State dimension $d _ { \mathrm { s t a t e } } = 6 4$ is the only parameter whose optimal value is consistent across both datasets.

![](images/a4c62d6a3229ef22100bd247e20c25b03584f0baef98eabbef8583d2efedfeb6.jpg)

Fig. 4. Hyperparameter sensitivity analysis under 10-fold cross-validation (upper: ADHD-200, lower: ABIDE). Solid lines and shaded bands denote ACC; dashed lines denote AUC. ADHD and ABIDE exhibit opposite trends for depth and hidden dimension, reflecting their different site heterogeneity profiles.  
![](images/6ba048ffba47bd267e60f7442f6e4f51d4a082c65fd5506e342998528bb245d6.jpg)  
(a) ADHD-200: connections predominantly involve prefrontal-striatalcingulate circuits.

![](images/f2761f6933b81a99fb125b9d68e4691fe83bcb4c6f639cdb60be28e0d0d80396.jpg)  
(b) ABIDE: connections are dominated by cerebellar and limbic pathways.

Fig. 5. Top-10 PopDRP-selected functional connectivity features.

## G. Visualization and Interpretability

We provide four complementary visualizations to assess whether PopDRP learns clinically meaningful population structures and to validate the RFE-selected features against established neuroscience findings.

PopDRP-selected brain connectome. Figure 5 shows the top-10 functional connectivity features identified by PopDRP on the glass brain for both datasets. On ADHD-200 (Figure 5a) the selected features concentrate on prefrontal, striatal and cingulate structures—the list contains IFG–amygdala, IFG–SMA and MFG–anterior-cingulate pairs, along with a cingulate– caudate connection, which is the classic fronto-striatal ADHD profile. These fronto-striatal and fronto-limbic pathways are well-established biomarkers of ADHD [22], reflecting the disorder’s core deficits in cognitive control, impulse regulation, and emotion processing. On ABIDE (Figure 5b) six of the ten features are cerebellar or vermian (two intravermis and four intra-cerebellar pairs) and three further involve orbitofrontal cortex, i.e. orbitofrontal–pallidum, orbitofrontal– precuneus and orbitofrontal–SMA. Cerebellar structural and functional abnormalities are among the most consistently replicated neuroimaging findings [23] in ASD, and the frontostriatal (orbitofrontal-pallidal) circuit is centrally implicated in the repetitive behaviors and social cognition deficits characteristic of the disorder. The alignment between PopDRP-selected features and established neuroscience literature confirms that PopDRP’s feature selection captures diagnostically and biologically meaningful signal.

t-SNE embedding visualization. Figure 3 presents t-SNE projections of the SPSS output embeddings. Each row corresponds to one dataset, with the left panel colored by diagnosis and the center panel colored by scanning site; the right panel provides a site-color legend. On both datasets, the diagnosiscolored panels show clear separation between healthy controls and patients, confirming that the SPSS backbone learns diagnostically discriminative representations. The site-colored panels reveal that subjects from different sites are intermingled within each diagnostic cluster rather than forming site-specific subclusters.

## H. Limitations and Future Work

Several limitations motivate future research directions. First, while PopDRP currently models two diagnostic categories per dataset, extending to multi-class diagnosis and dimensional symptom severity prediction would increase clinical utility. Second, the population graph is constructed from a fixed set of non-imaging attributes; incorporating dynamic or longitudinal data could further personalize diagnosis. Third, all performance assessments in this paper are conducted via internal cross-validation. Additional validation strategies such as testing on independent external cohorts may offer more robust insights into the model’s cross-site transferability.

## V. CONCLUSION

This work proposes PopDRP, a progressive population graph learning framework for brain disease diagnosis following a disentangle–rectify–perceive pipeline. CAR extracts clinically relevant non-imaging similarities free of confounds; NGR eliminates false graph edges via minimal-sufficient representation; the SPSS backbone captures multi-scale hierarchical features by integrating selective state-space modeling and U-Net graph pooling. Evaluated via 10-fold cross-validation on ABIDE and ADHD-200, PopDRP attains SOTA diagnostic accuracies, with ablation experiments verifying each module’s efficacy. It further delivers interpretable results including structured embeddings to link model outputs to clinical population patterns, alongside biologically meaningful diseaserelated brain connectivity biomarkers.

## REFERENCES

[1] S. Parisot, S. I. Ktena, E. Ferrante, M. Lee, R. Guerrero, B. Glocker, and D. Rueckert, “Spectral graph convolutions for population-based disease prediction,” Medical Image Computing and Computer-Assisted Intervention (MICCAI), pp. 177–185, 2017.

[2] ——, “Disease prediction using graph convolutional networks: Application to autism spectrum disorder and alzheimer’s disease,” Medical Image Analysis, vol. 48, pp. 117–130, 2018.

[3] Y. Huang and A. C. Chung, “Disease prediction with edge-variational graph convolutional networks,” Medical Image Analysis, vol. 77, p. 102375, 2022.

[4] L. Cai, W. Zeng, H. Chen, H. Zhang, Y. Li, Y. Feng, H. Yan, L. Bian, W. T. Siok, and N. Wang, “Mm-gtunets: Unified multi-modal graph deep learning for brain disorders prediction,” IEEE Transactions on Medical Imaging, 2025.

[5] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” arXiv preprint arXiv:2312.00752, 2023.

[6] C. Wang, O. Tongo et al., “Graph-mamba: Towards long-range graph sequence modeling with selective state spaces,” arXiv preprint arXiv:2402.00789, 2024.

[7] A. Behrouz and F. Hashemi, “Graph mamba: Towards learning on graphs with state space models,” in Proceedings of the 30th ACM SIGKDD conference on knowledge discovery and data mining, 2024, pp. 119– 130.

[8] R. Guo, X. Tian, H. Lin, S. McKenna, H.-D. Li, F. Guo, and J. Liu, “Graph-based fusion of imaging, genetic and clinical data for degenerative disease diagnosis,” IEEE/ACM Transactions on Computational Biology and Bioinformatics, vol. 21, no. 1, pp. 57–68, 2023.

[9] A. Kazi, S. Shekarforoush, S. A. Krishna, H. Burwinkel, G. Vivar, K. Kortum, S.-A. Ahmadi, and N. Navab, “Inceptiongcn: receptive field¨ aware graph convolutional network for disease prediction,” International Conference on Information Processing in Medical Imaging (IPMI), pp. 73–85, 2019.

[10] X. Li, Y. Zhou, N. C. Dvornek, M. Zhang, S. Gao, J. Zhuang, D. Scheinost, L. H. Staib, P. Ventola, and J. S. Duncan, “Brain-gnn: Interpretable brain graph neural network for fmri analysis,” Medical Image Analysis, vol. 74, p. 102233, 2021.

[11] S. Zheng, Z. Zhu, Z. Liu, Z. Guo, Y. Liu, Y. Yang, and Y. Zhao, “Multimodal graph learning for disease prediction,” IEEE Transactions on Medical Imaging, vol. 41, no. 9, pp. 2207–2216, 2022.

[12] S. Zhang, X. Chen, X. Shen, B. Ren, Z. Yu, H. Yang, X. Jiang, D. Shen, Y. Zhou, and X.-Y. Zhang, “A-gcl: Adversarial graph contrastive learning for fmri analysis to diagnose neurodevelopmental disorders,” Medical Image Analysis, vol. 90, p. 102932, 2023.

[13] A. Gu, K. Goel, and C. Re, “Efficiently modeling long sequences´ with structured state spaces,” International Conference on Learning Representations (ICLR), 2022.

[14] Y. Zhu, W. Xu, J. Zhang, Q. Liu, S. Wu, and L. Wang, “A survey on deep graph structure learning,” IEEE Transactions on Neural Networks and Learning Systems, 2021.

[15] J. Song, J. Park, and E. Yang, “Tam: topology-aware margin loss for class-imbalanced node classification,” in International conference on machine learning. PMLR, 2022, pp. 20 369–20 383.

[16] J. Song, S. Kim, and E. Yang, “Rge: A repulsive graph rectification for node classification via influence,” in International Conference on Machine Learning. PMLR, 2023, pp. 32 331–32 348.

[17] A. Di Martino, C.-G. Yan, Q. Li, E. Denio, F. X. Castellanos, K. Alaerts, J. S. Anderson, M. Assaf, S. Y. Bookheimer, M. Dapretto et al., “The autism brain imaging data exchange: towards a large-scale evaluation of the intrinsic brain architecture in autism,” Molecular Psychiatry, vol. 19, no. 6, pp. 659–667, 2014.

[18] P. Bellec, C. Chu, F. Chouinard-Decorte, Y. Benhajali, D. S. Margulies, and R. C. Craddock, “The adhd-200 consortium: A model to advance the translational potential of neuroimaging in clinical neuroscience,” Frontiers in Systems Neuroscience, vol. 11, p. 62, 2017.

[19] K. Zhao, B. Duka, H. Xie, D. J. Oathes, V. D. Calhoun, and Y. Zhang, “A dynamic graph convolutional neural network framework reveals new insights into connectome dysfunctions in ADHD,” NeuroImage, vol. 246, p. 118774, 2022. [Online]. Available: https://doi.org/10.1016/j.neuroimage.2021.118774

[20] Y. Chen, J. Yan, M. Jiang et al., “Adversarial learning based node-edge graph attention networks for autism spectrum disorder identification,” IEEE Transactions on Neural Networks and Learning Systems, vol. 35, no. 6, pp. 7275–7286, 2022.

[21] L. Peng, N. Wang, J. Xu, X. Zhu, and X. Li, “Gate: Graph cca for temporal self-supervised learning for label-efficient fmri analysis,” IEEE Transactions on Medical Imaging, vol. 42, no. 2, pp. 391–402, 2022.

[22] L. J. Norman, G. Sudre, J. Price, and P. Shaw, “Subcortico-cortical dysconnectivity in adhd: a voxel-wise mega-analysis across multiple cohorts,” American Journal of Psychiatry, vol. 181, no. 6, pp. 553–562, 2024.

[23] M. Assaf, K. Jagannathan, V. D. Calhoun, L. Miller, M. C. Stevens, R. Sahl, J. G. O’Boyle, R. T. Schultz, and G. D. Pearlson, “Abnormal functional connectivity of default mode sub-networks in autism spectrum disorder patients,” Neuroimage, vol. 53, no. 1, pp. 247–256, 2010.