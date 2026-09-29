# Addressing Spatial Indistinguishability in Spatiotemporal Prediction via Optimal Transport-Guided Masking

Guangyu Wang<sup>a</sup>, Jiawei Tong<sup>b,∗</sup>

<sup>a</sup>Data Science and Artificial Intelligence, Dongbei University ofFinance and Economics, Dalian, Liaoning, China

<sup>b</sup>Institute of Systems, Molecular & Integrative Biology, University of Liverpool, Liverpool, United Kingdom

## Abstract

Spatiotemporal prediction aims to learn discriminative representations from correlated temporal signals over spatial structures for accurate future inference. A central challenge is spatial indistinguishability: diferent nodes may share similar historical patterns yet evolve toward divergent futures, severely degrading forecasting performance in real-world sensor networks. Existing embedding-based and graph neural network (GNN)-based approaches can partially detect such ambiguous nodes but rely on historical similarity, struggling to capture future behavioral divergence. We propose STOT (SpatioTemporal Optimal Transport), a self-supervised framework that resolves spatiotemporal ambiguity via structured masking guided by optimal transport. Our key idea treats indistinguishability as a disambiguation problem: future states are inferred by exploiting concurrent spatial correlations and their time-varying similarity. We design a similarity-aware metric for dynamic inter-node relationships and an optimal transport-based masking strategy to emphasize ambiguous positions during pre-training. A batch consistency constraint preserves semantic coherence, while a random-walk masking mechanism promotes structured context exploration. Experiments on six real-world datasets show that STOT performs competitively with stateof-the-art baselines on the evaluated benchmarks and improved interpretability through transport-plan visualizations.

Keywords: Spatiotemporal Prediction, Masked Autoencoder, Deep Learning,

## 1. Introduction

Spatiotemporal data prediction through sensor networks is a classic multivariate time series forecasting task [1], with broad applications in earth system forecasting [2], trafic flow prediction [3], and energy prediction [4]. Prediction accuracy plays a crucial role in efective decision-making and minimizing socioeconomic impacts across these tasks. However, for traditional machine learning methods such as Support Vector Regression [5], Random Forest [6], and Gradient Boosting Decision Trees [7], achieving high prediction accuracy remains challenging because of the complex spatiotemporal relationships and nonlinear patterns inherent in these systems.

Notably, deep learning advances in spatiotemporal prediction have revolutionized this field with increasingly efective solutions, especially after identifying spatial indistinguishability as the key insight for model design. In the early stages, although Recurrent Neural Network (RNN) and Convolutional Neural Network (CNN)-based methods showed promise in capturing temporal and spatial patterns [8], their limitations in incorporating topological structures still hindered prediction performance. To address this limitation, Graph Neural Networks (GNNs) overcame topological limitations by capturing complex relationships between spatial nodes [9], yet their architectural complexity with uncertain performance gains still posed significant challenges for spatiotemporal prediction. A key breakthrough came with the specific illustration of spatial indistinguishability [10] – nodes with similar historical patterns exhibiting divergent future trends – which directly led to a paradigm shift in spatiotemporal prediction. This insight inspired new directions in model design, where embeddingbased methods [11] tackled spatial indistinguishability more directly through simpler architectures by constructing spatially distinguishable embeddings, avoiding the performance uncertainties of complex GNN-based methods [12].

Currently, predicting spatially indistinguishable nodes poses a critical challenge for accurate spatiotemporal prediction, as these nodes exhibit similar historical patterns but divergent future trends. This challenge is particularly significant given that such nodes dominate over 50% of sensor networks in real-world applications [13]. Although existing embedding-based methods [11] have attempted to improve performance by distinguishing node identities, this approach fails to capture the future behavioral characteristics of these nodes: while distinguishing identities may prevent interference with stable nodes, it does not directly contribute to predicting their future states. Such limitations persist in other approaches as well; GNN-based methods isolate these nodes through topological structures, yet still fail to capture their inherent predictive characteristics. We provide a more detailed discussion of their limitations in Section 2.

In this paper, we propose a novel theoretical framework to address the fundamental challenge of predicting spatially indistinguishable nodes in spatiotemporal systems. Our key insight is that while these nodes may exhibit weak temporal autocorrelation, their future states can be efectively inferred through concurrent spatial correlations. This theoretical foundation is supported by empirical observations across various spatiotemporal domains, where local perturbations propagate through the system with spatially and temporally varying efects, creating predictable patterns of influence. Leveraging this theoretical framework, we introduce several methodological innovations. First, we develop a quantitative metric for dynamically evaluating spatial indistinguishability, providing a rigorous foundation for identifying and analyzing these challenging nodes. Second, inspired by recent advances in self-supervised learning, particularly masked autoencoders (MAE), we design a specialized pre-training paradigm that focuses on reconstructing the states of spatially indistinguishable nodes using information from their distinguishable counterparts. To enhance the efectiveness of this approach, we incorporate two complementary constraints: a batch consistency policy that maintains semantic coherence across training instances and a random walk mechanism that ensures comprehensive exploration of the representation space. These components are unified within an optimal transport framework, leading to our proposed method: SpatioTemporal Optimal Transport (STOT). The main contributions of this paper can be summarized as follows:

• We present a systematic theoretical analysis of spatial indistinguishability in spatiotemporal prediction, revealing the critical importance of predicting indistin-

guishable nodes’ future states.

• We introduce a novel timestep-level metric for quantifying spatial indistinguishability, enabling a more precise characterization of spatiotemporal dynamics.

• We develop an optimal transport constraint framework that ensures both semantic continuity and diversity through batch consistency and random walk strategies.

• We validate our theoretical framework through extensive experiments on six realworld datasets, demonstrating competitive or superior performance on the evaluated benchmarks and providing insights into spatiotemporal prediction mechanisms.

## 2. Related Work

This section reviews the mainstream approaches for spatiotemporal prediction and discusses their limitations in accurately predicting future states of spatially indistinguishable nodes.

## 2.1. Graph-based Methods

Graph-based methods leverage the topological relationships among sensor nodes to represent spatial interactions in time series data, enabling the design of advanced spatiotemporal models. For instance, STGCN [14] integrates graph convolutional networks with gated temporal convolutions to capture comprehensive spatiotemporal features, while STNN [15] further refines this approach by incorporating spatial and temporal attention mechanisms to address region-based dependencies and external factors. Recently, a prevalent view has emerged that predefined graphs may introduce biases or prove inadequate in certain scenarios. In response, GTS [16] proposes learning latent graph structures jointly with spatiotemporal dynamics, and DFDGCN [17] combines static graphs with dynamically learned adjacency matrices. More recently, LvSTformer [18], a transformer-based approach, has been introduced to capture evolving multi-scale spatiotemporal patterns across sensors. From a data-driven perspective, the superior modeling of spatiotemporal relationships by latent graph structures cannot be solely attributed to capturing time-varying interaction patterns. We argue that the key lies in addressing spatially indistinguishable nodes. In predefined graphs, these nodes are connected to others, and their inherently unpredictable future patterns can induce information confusion during message passing. Latent graphs, by contrast, efectively sever these problematic connections, which leads to consistently better performance, a view supported by [13]. Clearly, GNN-based methods struggle to predict the future states of spatially indistinguishable nodes when they remain connected to other nodes [19], and empirical evidence shows that disconnecting these nodes enhances overall predictive accuracy.

## 2.2. Embedding-based Methods

Embedding-based methods address spatial indistinguishability by designing spatially distinguishable models. Further examining the mechanisms of representative approaches, STID [11] augments nodes with identity information in both temporal and spatial dimensions, enabling node diferentiation. STID provides a more direct solution that achieves similar results to models that learn latent graphs, disconnecting nodes that are spatially indistinguishable from each other. Building upon this, STAEformer [20] further integrates these identity-aware embeddings with transformer architectures to enhance performance. Another line of work focuses more on capturing location-specific temporal dynamics. ST-WA [21] learns location-specific encodings to capture unique patterns across time series, while ST-Norm [10] achieves a similar efect through carefully designed normalization methods, eliminating the need for additional embeddings.

More recently, HiMNet [22] learns heterogeneous meta-parameters from clustered spatial and temporal embeddings, which captures node-level behavioral diferences without relying on explicit identity inputs. A parallel and more recent trend concerns generative modeling for spatiotemporal systems. For instance, $\mathrm { N e t - E v ^ { 2 } }$ [23] simulates network event evolution with a generative framework, and Difusion-TS [24] provides an interpretable difusion model for general time series generation.

Embedding-based methods inherently rely on node diferentiation, similar to latent graph learning, as they disconnect spatially indistinguishable nodes from others. However, they fail to efectively predict the future states of these spatially indistinguishable nodes.

## 2.3. Pre-trained Methods

Pre-trained methods enable models to receive longer historical inputs, which has received increasing attention in spatiotemporal forecasting, as extended input sequences enhance a model’s robustness to sudden events and improve its generalization to unseen data [25]. To obtain superior hidden representations, researchers have explored using pre-training techniques on time series data. One promising technique is the masked autoencoder (MAE), which masks out random input patches and reconstructs the orig inal input based on the unmasked patches. This approach has been successfully applied in various domains, with masked pre-training yielding significant improvements in downstream tasks in computer vision [26] and natural language processing [27]. Several studies have explored the application of MAE in spatiotemporal forecasting. For instance, STEP [28] employs MAE to reconstruct long historical time series, enhancing the model’s prediction capabilities. However, Ti-MAE [29] noted the inconsistency between contrastive learning during pre-training and downstream prediction tasks and proposed using MAE as an auxiliary task to address this issue. Similarly, PITS [30] tackled this problem by combining MAE with comparative learning to avoid conflicts between pre-training and downstream tasks. Building upon these works, STD-MAE [25] considered the practical challenges of spatiotemporal forecasting and employed two MAEs in both temporal and spatial dimensions to reconstruct historical sequences, concatenating the spatiotemporal representations as inputs for the downstream predictor. Taking a diferent approach, SimMTM [31] connects masked modeling with manifold learning to learn deeper representations. However, these methods overlook the inherent nature of spatiotemporal forecasting and fail to strengthen train ing for spatial indistinguishability, diminishing pre-training performance and reducing eficiency. In contrast, our proposed method, STOT, introduces a carefully designed spatial indistinguishability-guided masking strategy during pre-training. By reconstructing indistinguishable nodes through distinguishable ones, STOT more efectively captures the complex interactions within spatiotemporal data.

## 3. Problem Formulation

We formulate the spatiotemporal system as a graph $G \ = \ ( V , E , A )$ , where $V \ =$ $\{ \nu _ { i } \} _ { i = 1 , 2 , \ldots , N }$ represents the set of N nodes in the system. The edge set $E = \{ e _ { i j } \}$ captures the connectivity between node pairs, with the adjacency matrix $A \in \{ 0 , 1 \} ^ { N \times N }$ encoding these relationships such that $A _ { i j } = 1$ if nodes $\nu _ { i }$ and $\nu _ { j }$ are connected $( e _ { i j } \in E )$ , and $A _ { i j } = 0$ otherwise. The system dynamics are characterized by a feature matrix $X \in$ $\mathbb { R } ^ { N \times T _ { l o n g } \times C }$ , where $T _ { l o n g }$ represents the temporal dimension of historical observations in the pre-training stage, which is much longer than the historical window T used in endto-end prediction and C is the feature dimension. At each timestep $t ,$ the state of the system is captured by $\boldsymbol { x } _ { t } \in \mathbb { R } ^ { N }$

## 4. Methodology

## 4.1. Overall Architecture of STOT

As shown in Figure 1, STOT contains two stages: pre-training (a)-(e) and downstream prediction (f). In pre-training, $T _ { l o n g }$ timesteps pass through an embedding layer that combines raw features with temporal and spatial positional encodings, and then through a Transformer encoder-decoder with an SI-Mask targeting spatially indistinguishable positions; the learned SI-enhanced representations are integrated with downstream predictors. Unlike random-mask MAE models, STOT needs only one pre-training phase without contrastive learning, achieving comparable representation quality more eficiently than STD-MAE [25] and PITS [30].

## 4.2. Embedding Layer

The embedding layer produces position-aware representations through patch segmentation, high-dimensional feature projection, and temporal/spatial positional embeddings.

![](images/5a312259e0a9d20fe095b87554448d0b305ffba93a207f44251fc914876000fe.jpg)  
Figure 1: Architecture of STOT: (a) the embedding layer aggregates raw long-term trafic data with temporal and spatial positional features; (b) the encoder learns spatially indistinguishable node representations through masked reconstruction; (c) the decoder reconstructs masked positions from SI-enhanced representations; (d) mask generation uses SI-Mask, Batch Consistency, and Random Walk, where SI-Mask guides spatial indistinguishability and the latter two constrain representation continuity and richness; (e) patch embedding segments spatiotemporal data into local-pattern units; and (f) the STOT-enhanced downstream predictor can be integrated with various downstream architectures.

## 4.2.1. Patch Segmentation

Following [32], the input $X _ { p r e } ~ \in ~ \mathbb { R } ^ { T _ { l o n g } \times N \times C }$ is divided into P non-overlapping patches of length $T _ { p } = T _ { l o n g } / P \mathrm { ; }$

$$
X _ { p a t c h } = [ \mathcal { P } _ { 0 } ( X _ { p r e } ) ; \mathcal { P } _ { 1 } ( X _ { p r e } ) ; . . . ; \mathcal { P } _ { P - 1 } ( X _ { p r e } ) ] ,\tag{1}
$$

where $\mathcal { P } _ { i } ( \cdot )$ extracts the i-th segment, yielding $X _ { p a t c h } \in \mathbb { R } ^ { T _ { p } \times N \times ( C P ) }$

## 4.2.2. Positional Embeddings

Temporal Positional Embedding (TPE) encodes temporal position $p$ via sinusoidal functions:

$$
E _ { t } ( p , q , k ) = \left\{ \begin{array} { l l } { \sin ( p / 1 0 0 0 0 ^ { 4 k / D } ) , } & { \mathrm { i f } k \mathrm { i s e v e n } } \\ { \cos ( p / 1 0 0 0 0 ^ { 4 k / D } ) , } & { \mathrm { i f } k \mathrm { i s o d d } } \end{array} , \right.\tag{2}
$$

yielding $E _ { t } \in \mathbb { R } ^ { T _ { p } \times N \times d / 2 }$ . Spatial Positional Embedding (SPE) follows the same formulation with spatial position q in place of $p ,$ yielding $E _ { s } \in \mathbb { R } ^ { T _ { p } \times N \times d / 2 }$

## 4.2.3. Embedding Output

Raw data is projected via a fully connected layer $E _ { F } = \mathsf { F C } ( X _ { p a t c h } ) \in \mathbb { R } ^ { P \times N \times d }$ . The final embedding concatenates all three components:

$$
Z = E _ { F } \parallel E _ { t } \parallel E _ { s } \in \mathbb { R } ^ { T _ { p } \times N \times 2 d } .\tag{3}
$$

## 4.3. Spatiotemporal Masked Pre-training

Following the MAE architecture, the encoder masks spatially indistinguishable positions and the decoder reconstructs them; our masking strategy targets such positions, balances semantic richness with computational eficiency, and learns transferable spatiotemporal representations.

## 4.3.1. Encoder Layer

As shown in Figure 1(b), the encoder masks positions with high spatial indistinguishability. Unlike prior dataset-level measurements [13], our dynamic assessment captures time-varying node relationships and guides mask generation through an optimal transport (OT) formulation.

Similarity Matrix Computation We compute cosine similarity matrices for both historical and future timesteps:

$$
S _ { t , i , j } ^ { P } = \frac { \boldsymbol { X } _ { t , i } ^ { P } \cdot \boldsymbol { X } _ { t , j } ^ { P } } { \| \boldsymbol { X } _ { t , i } ^ { P } \| \| \boldsymbol { X } _ { t , j } ^ { P } \| } , \quad S _ { t , i , j } ^ { F } = \frac { \boldsymbol { X } _ { t , i } ^ { F } \cdot \boldsymbol { X } _ { t , j } ^ { F } } { \| \boldsymbol { X } _ { t , i } ^ { F } \| \| \boldsymbol { X } _ { t , j } ^ { F } \| } , \quad i , j \in \{ 1 , \dots , N \} .\tag{4}
$$

Spatial Indistinguishability Computation Given thresholds $\epsilon _ { u }$ (high similarity) and $\epsilon _ { l }$ (low similarity), the indistinguishability score for node i at timestep t is:

$$
I _ { t , i } = \frac { \sum _ { j \neq i } \mathcal { k } ( S _ { t , i , j } < \epsilon _ { l } ) } { \sum _ { j \neq i } \mathcal { k } ( S _ { t , i , j } > \epsilon _ { u } ) + 1 } ,\tag{5}
$$

which is high when node i’s similarity to others is uniformly distributed. We adopt this ratio between confidently dissimilar and confidently similar neighbors because it is directional, growing precisely when a node lacks reliable similar anchors yet is surrounded by dissimilar ones, which is the configuration that makes its future hard to infer. A dispersion summary such as the entropy of the similarity distribution is direction-agnostic and assigns the same value to an anchor-rich node and to an anchorpoor one, so it cannot separate easy nodes from genuinely ambiguous ones. The same construction also keeps the score insensitive to the exact thresholds, since it reflects the gap between the two groups rather than their absolute counts, and Figure 2 confirms that nodes with $I _ { t , i } > 0$ remain consistently harder to predict as ϵ varies. The overall score aggregates across both time periods and nodes:

$$
I _ { i } = \alpha \cdot \frac { 1 } { T _ { p } } \sum _ { t = 1 } ^ { T _ { p } } I _ { t , i } ^ { P } + ( 1 - \alpha ) \cdot \frac { 1 } { T _ { p r e d } } \sum _ { t = 1 } ^ { T _ { p r e d } } I _ { t , i } ^ { F } , \quad T s c o r e = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } I _ { i } ,\tag{6}
$$

where $\alpha \in [ 0 , 1 ]$ balances historical and future contributions.

Generating SI-Mask Position Directly selecting the top-K highest T score positions has two drawbacks: dispersed within-batch masks hinder coherent learning, and fixed masks for identical inputs limit semantic diversity. We address both by formulating mask generation as an OT problem.

Positions with high Tscore are assigned lower transport cost via a log transformation, and a batch consistency penalty discourages overly dispersed masks:

$$
C _ { b a s e } = - \log ( T s c o r e + 1 ) , \quad C _ { f i n a l } = C _ { b a s e } + \lambda \cdot \left| C _ { b a s e } - \frac { 1 } { B } \sum _ { b = 1 } ^ { B } C _ { b } \right| ,\tag{7}
$$

where λ controls the batch consistency weight.

The Sinkhorn algorithm [33] solves the OT problem by iteratively normalizing rows

and columns to produce a doubly stochastic transport matrix (Pseudocode 1):

Algorithm 1 Sinkhorn Algorithm for Mask Position Generation   
Require: Cost matrix $C _ { f i n a l } ,$ number of iterations K, regularization parameter ϵ   
Ensure: Optimal transport plan M (mask positions)   
1: Initialize $M ^ { ( 0 ) } = \exp ( - C _ { f i n a l } / \epsilon )$   
2: for k = 1 to K do   
3: $Q ^ { ( k ) } = M ^ { ( k - 1 ) }$ ▷ Store previous iteration   
4: $M ^ { ( k ) } = \mathrm { D i a g } ( u ^ { ( k ) } ) \cdot Q ^ { ( k ) } \cdot \mathrm { D i a g } ( \nu ^ { ( k ) } )$ ▷ Update transport plan   
5: $u ^ { ( k ) } = a \oslash ( Q ^ { ( k ) } \nu ^ { ( k - 1 ) } )$ ▷ Row normalization   
6: $\nu ^ { ( k ) } = b \oslash ( ( Q ^ { ( k ) } ) ^ { T } u ^ { ( k ) } )$ ▷ Column normalization   
7: end for   
8: return $M ^ { ( K ) }$ ▷ Final transport plan as mask positions

Mask positions are selected by choosing the $M _ { t o t a l } = \lfloor T _ { p } \cdot r _ { m a s k } \rfloor$ entries that receive the largest transport mass per sample:

$$
{ M } _ { b , s } = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f } \ s \in \mathrm { t o p k } ( M _ { b } , M _ { t o t a l } ) } \\ { 0 , } & { \mathrm { o t h e r w i s e } } \end{array} , \right.\tag{8}
$$

where topk(· $M _ { t o t a l } )$ returns the indices of the $M _ { t o t a l }$ largest transport values for sample b. The cost is evaluated per sample and candidate timestep, so $C _ { f i n a l } \in \mathbb { R } ^ { B \times T _ { p } }$ with $C _ { f i n a l , b , }$ <sub>s</sub> the cost of sample b at timestep $s ,$ and $M _ { b }$ is the b-th row of the transport plan $M ^ { ( K ) }$ returned by Algorithm 1, holding the mass that sample b assigns to each candidate position. Since $M ^ { ( 0 ) } = \exp ( - C _ { f i n a l } / \epsilon )$ maps low cost to high mass, selecting the largest-mass entries of $M _ { b }$ and setting ${ \mathcal M } _ { b , s } = 1$ turns the transport plan into the binary SI-Mask, while the batch-consistency term and column normalization keep this selection coupled across samples.

Random Walk Mechanism Controlled by $w _ { r a t i o } ,$ , random walk replaces $N _ { r e p l a c e } =$ $\left\lfloor M _ { t o t a l } \cdot w _ { r a t i o } \right\rfloor$ positions in the OT-generated mask by sampling a new timestep $t _ { n e w }$ from unmasked positions $\mathcal { T } \backslash M _ { b a t c h }$ and releasing an existing mask position $p _ { o l d } ;$ to maintain batch consistency, the operation is applied uniformly across samples:

$$
\begin{array} { r } { M _ { b , p _ { o l d } } = 0 , \quad M _ { b , t _ { n e w } } = 1 , \quad \forall b \in \{ 1 , \ldots , B \} , } \end{array}\tag{9}
$$

yielding the final binary mask $\boldsymbol { \mathcal { M } } = [ \mathcal { M } _ { b , 1 } , \ldots , \mathcal { M } _ { b , T _ { p } } ]$ that combines OT-guided positions with random walk diversity.

Encoder with Mask Given the masked input where [MAS K] tokens replace positions with $\mathcal { M } _ { b , t } = 1$

$$
\tilde { x } _ { b , t } = \left\{ \begin{array} { l l } { [ M A S K ] , } & { \mathrm { i f } \ M _ { b , t } = 1 } \\ & { \mathrm { i f } \ M _ { b , t } = 0 } \end{array} , \right.\tag{10}
$$

the encoder applies standard Transformer layers—multi-head self-attention, feed-forward networks, and layer normalization—to produce the SI-enhanced representation after L layers:

$$
\mathcal { R } ^ { S I } = \mathrm { E n c o d e r S t a c k } _ { L } ( \tilde { X } _ { p a t c h } , M ) \in \mathbb { R } ^ { B \times T _ { p } \times D } .\tag{11}
$$

## 4.3.2. Decoder Layer

The decoder uses the same Transformer architecture as the encoder; after padding to uniform dimensionality, $L _ { d }$ decoder layers reconstruct masked positions from $\mathcal { R } ^ { S I }$ and yield $\hat { X } _ { p a t c h } \in \mathbb { R } ^ { B \times T _ { p } \times D }$ with reconstructed values $Y _ { p r e }$ where $\mathcal { M } _ { b , t } = 1$

## 4.4. Downstream Spatiotemporal Prediction

The pre-trained encoder processes $T _ { l o n g }$ timesteps to produce $\mathcal { R } ^ { S I }$ , which is fused with the short-term representation from downstream predictor $F _ { 0 }$ via an MLP:

$$
\mathcal { R } ^ { f u s e d } = \mathbf { M } \mathbf { L } \mathbf { P } ( [ \mathcal { R } ^ { S I } ; F _ { 0 } ( T ) ] ) .\tag{12}
$$

In this work we use GWNet [34] as the primary predictor, with additional experiments on LSTM and ASTGCN [35] to validate generalizability across architectures. Since this fusion runs after representation learning, where the pre-trained encoder already supplies informative spatial indistinguishability representations, we keep it as a simple MLP that preserves their integrity while remaining accurate, following classic models such as STAEformer [20].

## 4.5. Loss Function

STOT uses two losses across its two training phases, with the encoder frozen during prediction:

$$
\mathcal { L } _ { p r e } = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \sum _ { t = 1 } ^ { T _ { p } } \boldsymbol { M } _ { b , t } \Vert Y _ { p r e } ^ { b , t } - X _ { p a t c h } ^ { b , t } \Vert _ { 2 } ^ { 2 } , \quad \mathcal { L } _ { p r e d } = \sqrt { \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \Vert \boldsymbol { F } _ { 0 } ( \mathcal { R } ^ { f u s e d } ) - Y \Vert _ { 2 } ^ { 2 } } ,\tag{13}
$$

where $\mathcal { L } _ { p r e }$ is the masked reconstruction loss over pre-training and $\mathcal { L } _ { p r e d }$ is the RMSE for downstream forecasting.

## 5. Experiments

We answer seven research questions: (I) STOT’s performance against SOTA methods; (II) the efect of each mask pre-training component; (III) performance under high spatial indistinguishability; (IV) hyperparameter sensitivity across datasets; (V) computational eficiency versus other pre-training models; (VI) interpretability from OT visualization; and (VII) the efectiveness of the random walk mechanism compared with variants that change its unit or position.

## 5.1. Experimental Settings

## 5.1.1. Datasets

Table 1: Dataset description and statistics.
<table><tr><td>Datasets</td><td>#Sensors</td><td>#Edges</td><td>#Time Period</td><td>#Type</td><td>#Place</td></tr><tr><td>PeMS03</td><td>358</td><td>574</td><td>2018/09–2018/11</td><td>Flow</td><td>North Central California, USA</td></tr><tr><td>PeMS04</td><td>307</td><td>340</td><td>2018/01-2018/02</td><td>Flow</td><td>San Francisco Bay Area, USA</td></tr><tr><td>PeMS07</td><td>883</td><td>866</td><td>2017/05–2017/08</td><td>Flow</td><td>Los Angeles area, USA</td></tr><tr><td>PeMS08</td><td>170</td><td>295</td><td>2016/07-2016/08</td><td>Flow</td><td>San Bernardino area, USA</td></tr><tr><td>PeMS-BAY</td><td>325</td><td>2369</td><td>2017/01–2017/06</td><td>Speed</td><td>San Francisco Bay Area, USA</td></tr><tr><td>METR-LA</td><td>207</td><td>1515</td><td>2012/03-2012/06</td><td>Speed</td><td>Los Angeles area, USA</td></tr></table>

We conducted extensive experiments using six real-world trafic datasets, which are classic and widely used examples of spatiotemporal data [36, 37, 38], with details provided in Table 1. The datasets are briefly described as follows: PeMS03 provides trafic flow data from 358 sensors in California’s District 3, spanning September to November 2018; PeMS04 provides trafic flow data from 307 sensors in California’s District 4, collected between January and February 2018; PeMS07 provides trafic flow measurements from 883 sensors in California’s District 7, covering May to August 2017; PeMS08 provides trafic flow data from 170 sensors in California’s District 8, obtained during July and August 2016; METR-LA provides trafic speed measurements from 207 sensors in Los Angeles, California, collected between March and June 2012; and PeMS-BAY provides trafic speed data from 325 sensors in the San Francisco Bay Area, California, spanning January to June 2017. The trafic flow and speed datasets are partitioned into training, validation, and test subsets to facilitate model evaluation. For the trafic flow data, the ratio of the training, validation, and test subsets is 6:2:2, while for the trafic speed data, the ratio is 7:1:2.

## 5.1.2. Baselines

We compare STOT with 19 representative baselines covering classical statistical methods and recent SOTA deep learning models: ARIMA [39] and VAR [40]; spatiotemporal graph neural networks including DCRNN [9], STGCN [14], ASTGCN [35], GWNet [34], STGODE [41], DSTAGNN [42], ASTGNN [43], and AGCRN [44]; spatially enhanced or embedding/normalization-based models including ST-WA [21], ST-Norm [10], and STID [11]; transformer-based or attention-driven PDFormer [45] and STAEformer [20]; pre-training and masked-reconstruction STEP [28] and STD-MAE [25]; and heterogeneity or decomposition models HimNet [22] and STDN [46], ensuring comprehensive and fair assessment across modeling paradigms.

## 5.1.3. Parameter Settings

Following the BasicTS heterogeneity analysis protocol [13], we set the similarity thresholds to $\epsilon _ { u } = 0 . 9$ and $\epsilon _ { l } = 0 . 5$ . We also use a batch size of 8, 100 pre-training epochs, and at most 300 forecasting epochs. The specific hyperparameters are summarized in Table 2.

We further examined whether the low similarity threshold afects the validity of the spatial indistinguishability score. Because the score in Equation 5 is computed as a ratio between confidently dissimilar and confidently similar neighbors, its value depends on the overall separation of the similarity distribution and stays stable when the thresholds are perturbed within a reasonable range. To verify this property on a concrete model, we conducted a control analysis on the GWNet backbone [34]. For each node sample, we computed $I _ { t , i }$ and split the samples into one set with $I _ { t , i } = 0$ and one set with $I _ { t , i } ~ > ~ 0$ , then compared their forecasting MAE distributions under $\epsilon _ { l } \in \{ 0 . 4 0 , 0 . 4 5 , 0 . 5 0 , 0 . 5 5 \}$ }. As shown in Figure 2, the set with $I _ { t , i } > 0$ consistently attains larger MAE, which indicates that the score captures prediction dificulty and that the default threshold remains reliable.

Table 2: Detailed configurations of STOT and main baselines.
<table><tr><td>Setting</td><td>STOT</td><td>STD-MAE</td><td>STEP</td><td>ST-WA</td><td>STID</td><td>ST-Norm</td></tr><tr><td>Learning Rate</td><td>0.001</td><td>0.001</td><td>0.001</td><td>0.001</td><td>0.002</td><td>0.002</td></tr><tr><td>Hidden Dimension</td><td>96</td><td>96</td><td>96</td><td>128</td><td>32</td><td>一</td></tr><tr><td>Number of Heads</td><td>4</td><td>4</td><td>4</td><td>一</td><td>一</td><td>一</td></tr><tr><td>Similarity Thresholds  $( \epsilon _ { u } , \epsilon _ { l } )$ </td><td>(0.9, 0.5)</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td></tr><tr><td>Mask Ratio</td><td>0.25</td><td>0.25</td><td>0.75</td><td>一</td><td>一</td><td>一</td></tr><tr><td>Sinkhorn Temperature €</td><td>0.8</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td></tr><tr><td>Sinkhorn Iteration Count K</td><td>3</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td></tr><tr><td>Batch Consistency Weight λ</td><td>0.7</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td></tr></table>

![](images/bf66812e3337617c05f8d15432b43b317308cadb18ac163d9a1ca8509ab1eb52.jpg)  
Figure 2: KDE distributions of node-sample forecasting MAE grouped by the spatial indistinguishability score in Equation 5. Dashed lines indicate the group-wise mean MAE.

## 5.1.4. Evaluation Metrics

We use three standard metrics (y<sub>i</sub>: ground truth, ˆy<sub>i</sub>: prediction): $\begin{array} { r } { \mathbf { M } \mathbf { A } \mathbf { E } = \frac { 1 } { n } \sum | y _ { i } - \hat { y } _ { i } | . } \end{array}$ $\begin{array} { r } { \mathbf { R M S E } = \sqrt { \frac { 1 } { n } \sum ( y _ { i } - \hat { y } _ { i } ) ^ { 2 } } } \end{array}$ , and $\begin{array} { r } { \mathbf { M A P E } = \frac { 1 } { n } \sum \left| \frac { y _ { i } - \hat { y } _ { i } } { y _ { i } } \right| \times 1 0 0 } \end{array}$

## 5.1.5. Implementation Details

We conducted experiments on 8 RTX A6000 GPUs and implemented our proposed model based on BasicTS [13], which provides a unified framework for data loading and evaluation metrics to ensure fair comparisons. The shared training protocol and method-specific hyperparameters are detailed in the Parameter Settings subsection and summarized in Table 2. For a fair comparison, all pre-training methods, including STEP, STD-MAE, and STOT, were run with the same pre-training length of $T _ { l o n g } = 2 8 8$ steps, even though their original papers adopt longer sequences.

## 5.2. Main Results (RQ1)

## 5.2.1. Comparison Results

Tables 3 and 4 report six-dataset prediction results, with bold values marking the best performance; STOT consistently outperforms baselines on trafic speed and flow, and Figure 3 shows that it surpasses STID and ST-Norm across 1- to 12-step forecasts on PeMS03/04/07/08, demonstrating its advantage in handling spatial indistinguishability. The observations are that conventional statistical methods struggle with nonlinear and non-stationary data, yielding lower accuracy than deep learning approaches; GNN-based methods such as AGCRN capture spatiotemporal correlations but sufer from RNN-based multi-step error accumulation, while attention-based PDFormer mitigates this issue yet still underperforms because of insuficient spatial modeling. Their message passing also aggregates signals over fixed neighborhoods, so the confusion carried by spatially indistinguishable nodes propagates to neighbors and reinforces ambiguous representations, which is the core dificulty that our masking strategy targets. Spatially-aware models such as STID and ST-Norm generalize better by directly addressing spatial distinguishability, outperforming GNN-based methods in accuracy and eficiency, while pre-trained STEP and STD-MAE learn from long historical sequences and focus on hard-to-predict nodes, with STD-MAE’s dual autoencoders further exploiting spatial structure to achieve near-optimal results. STOT’s gains come from long-term inputs with temporal periodic features for richer spatiotemporal representations, adaptive spatial representations learned through mutual reconstruction guided by spatial distinguishability, and OT-based batch consistency that improves semantic coherence and generalization. The gain is smaller on PeMS-BAY because its nodes share very similar long-term levels and form few strongly correlated pairs, so the dataset carries less spatial indistinguishability for the optimal transport masking to exploit.

Table 3: Performance comparison on PeMS03, PeMS04, PeMS07, and PeMS08 benchmarks.
<table><tr><td rowspan="2">Model</td><td colspan="3">PeMS03</td><td colspan="3">PeMS04</td><td colspan="3">PeMS07</td></tr><tr><td>MAE</td><td>RMSE MAPE</td><td></td><td>MAE</td><td>RMSE MAPE</td><td></td><td>MAE</td><td>RMSE MAPE</td><td></td></tr><tr><td>ARIMA [39]</td><td>35.31</td><td>47.59</td><td>33.78</td><td>33.73</td><td>48.80</td><td>24.18</td><td>38.17</td><td>59.27</td><td>19.46</td></tr><tr><td>VAR [40]</td><td>23.65</td><td>38.26</td><td>24.51</td><td>23.75</td><td>36.66</td><td>18.09</td><td>75.63</td><td>115.24</td><td>32.22</td></tr><tr><td>DCRÑN [9]</td><td>18.18</td><td>30.31</td><td>18.91</td><td>24.70</td><td>38.12</td><td>17.12</td><td>25.30</td><td>38.58</td><td>11.66</td></tr><tr><td>STGCN [14]</td><td>17.49</td><td>30.12</td><td>17.15</td><td>22.70</td><td>35.55</td><td>14.59</td><td>25.38</td><td>38.78</td><td>11.08</td></tr><tr><td>ASTGCN [35]</td><td>17.69</td><td>29.66</td><td>19.40</td><td>22.93</td><td>35.22</td><td>16.56</td><td>28.05</td><td>42.57</td><td>13.92</td></tr><tr><td>GWNet [34]</td><td>19.85</td><td>32.94</td><td>19.31</td><td>25.45</td><td>39.70</td><td>17.29</td><td>26.85</td><td>42.78</td><td>12.12</td></tr><tr><td>STGODÈ [41]</td><td>16.50</td><td>27.84</td><td>16.69</td><td>20.84</td><td>32.82</td><td>13.77</td><td>22.99</td><td>37.54</td><td>10.14</td></tr><tr><td>DSTAGNN [42]</td><td>15.57</td><td>27.21</td><td>14.68</td><td>19.30</td><td>31.46</td><td>12.70</td><td>21.42</td><td>34.51</td><td>9.01</td></tr><tr><td>ST-WA [21]]</td><td>15.17</td><td>26.63</td><td>15.83</td><td>19.06</td><td>31.02</td><td>12.52</td><td>20.74</td><td>34.05</td><td>8.77</td></tr><tr><td>ASTGNN [43]</td><td>15.07</td><td>26.88</td><td>15.80</td><td>19.26</td><td>31.16</td><td>12.65</td><td>22.23</td><td>35.95</td><td>9.25</td></tr><tr><td>AGCRN [44]</td><td>16.06</td><td>28.49</td><td>15.85</td><td>19.83</td><td>32.26</td><td>12.97</td><td>21.29</td><td>35.12</td><td>8.97</td></tr><tr><td>ST-Norm [10]</td><td>15.32</td><td>25.93</td><td>14.37</td><td>19.21</td><td>32.30</td><td>13.05</td><td>20.59</td><td>34.86</td><td>8.61</td></tr><tr><td>STID [11]</td><td>15.30</td><td>27.26</td><td>16.47</td><td>18.40</td><td>29.98</td><td>12.93</td><td>19.61</td><td>32.83</td><td>8.33</td></tr><tr><td>STEP * [28]</td><td>14.22</td><td>24.55</td><td>14.42</td><td>18.20</td><td>29.71</td><td>12.48</td><td>19.32</td><td>32.19</td><td>8.12</td></tr><tr><td>PDFormer [45]</td><td>14.94</td><td>25.39</td><td>15.82</td><td>18.32</td><td>29.97</td><td>12.10</td><td>19.83</td><td>32.87</td><td>8.53</td></tr><tr><td>STAEformer [20]</td><td>15.35</td><td>27.55</td><td>15.18</td><td>18.22</td><td>30.18</td><td>11.98</td><td>19.14</td><td>32.60</td><td>8.01</td></tr><tr><td>HimNet [22]</td><td>15.34</td><td>27.50</td><td>N/A</td><td>18.24</td><td>30.12</td><td>N/A</td><td>19.30</td><td>32.77</td><td></td></tr><tr><td>STDN [46]</td><td>15.54</td><td>27.28</td><td>N/A</td><td>18.40</td><td>30.22</td><td>N/A</td><td>20.65</td><td>34.77</td><td>N/A N/A</td></tr><tr><td>STD-MAE* [25]</td><td>14.01</td><td>25.79</td><td>14.30</td><td>18.10</td><td>29.68</td><td>11.97</td><td>19.25</td><td>32.27</td><td>8.04</td></tr><tr><td>STOT (Our Study)</td><td>13.91</td><td>25.71</td><td>13.56</td><td>18.08</td><td>29.67</td><td>11.89</td><td>18.95</td><td>31.63</td><td>7.98</td></tr></table>

Table 4: Performance comparison on PeMS08, METR-LA, and PeMS-BAY benchmarks.
<table><tr><td rowspan="2">Model</td><td colspan="3">PeMS08</td><td colspan="3">METR-LA</td><td colspan="3">PeMS-BAY</td></tr><tr><td>MAE</td><td>RMSE MAPE</td><td></td><td></td><td></td><td>MAE RMSE MAPE</td><td></td><td></td><td>MAE RMSE MAPE</td></tr><tr><td>ARIMA [39]</td><td>31.09</td><td>44.32</td><td>22.73</td><td>5.15</td><td>10.45</td><td>12.70</td><td>2.33</td><td>4.76</td><td>5.40</td></tr><tr><td>VAR [40]]</td><td>23.46</td><td>36.33</td><td>15.42</td><td>5.28</td><td>9.06</td><td>12.50</td><td>2.24</td><td>4.96</td><td>4.83</td></tr><tr><td>DCRÑN [9]</td><td>17.86</td><td>27.83</td><td>11.45</td><td>3.59</td><td>7.61</td><td>10.44</td><td>1.96</td><td>4.59</td><td>4.68</td></tr><tr><td>STGCN [14]</td><td>18.02</td><td>27.83</td><td>11.40</td><td>3.60</td><td>7.50</td><td>10.56</td><td>1.99</td><td>4.51</td><td>4.66</td></tr><tr><td>ASTGCÑ [35]</td><td>18.61</td><td>28.16</td><td>13.08</td><td>3.57</td><td>7.19</td><td>10.32</td><td>1.86</td><td>4.07</td><td>4.27</td></tr><tr><td>GWNet [34]</td><td>19.13</td><td>31.05</td><td>12.68</td><td>4.90</td><td>9.70</td><td>14.75</td><td>2.71</td><td>6.25</td><td>6.69</td></tr><tr><td>STGODÉ [41]</td><td>16.81</td><td>25.97</td><td>10.62</td><td>4.73</td><td>7.60</td><td>11.71</td><td>1.77</td><td>3.89</td><td>4.02</td></tr><tr><td>DSTAGNN [42]</td><td>15.67</td><td>24.77</td><td>9.94</td><td>3.32</td><td>6.68</td><td>9.31</td><td>1.89</td><td>4.11</td><td>4.26</td></tr><tr><td>ST-WA [21]</td><td>15.41</td><td>24.62</td><td>9.94</td><td>3.65</td><td>7.56</td><td>10.35</td><td>2.01</td><td>4.57</td><td>4.70</td></tr><tr><td>ASTGNN [43]</td><td>15.98</td><td>25.67</td><td>9.97</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>AGCRN [44]</td><td>15.95</td><td>25.22</td><td>10.09</td><td>3.68</td><td>7.74</td><td>10.82</td><td>2.01</td><td>4.64</td><td>4.71</td></tr><tr><td>ST-Norm [10]</td><td>15.39</td><td>24.80</td><td>9.91</td><td>3.56</td><td>7.48</td><td>10.20</td><td>1.92</td><td>4.49</td><td>4.54</td></tr><tr><td>STID [11]</td><td>14.20</td><td>23.29</td><td>9.33</td><td>3.55</td><td>7.55</td><td>10.94</td><td>1.90</td><td>4.41</td><td>4.44</td></tr><tr><td>STEP* [28]</td><td>14.00</td><td>23.41</td><td>9.50</td><td>3.37</td><td>6.99</td><td>9.61</td><td>1.79</td><td>4.20</td><td>4.18</td></tr><tr><td>PDFormer [45]</td><td>13.58</td><td>23.51</td><td>9.05</td><td>3.62</td><td>7.47</td><td>10.91</td><td>1.91</td><td>4.43</td><td>4.51</td></tr><tr><td>STAEformèr [20]</td><td>13.46</td><td>23.25</td><td>8.88</td><td>3.34</td><td>7.04</td><td>9.78</td><td>1.85</td><td>4.31</td><td>4.33</td></tr><tr><td>HimNet [22]</td><td>13.56</td><td>23.36</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>STDN [46]</td><td>14.25</td><td>24.39</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>STD-MAE* [25]</td><td>13.45</td><td>22.47</td><td>8.78</td><td>3.42</td><td>7.08</td><td>9.62</td><td>1.79</td><td>4.21</td><td>4.17</td></tr><tr><td>STOT (Our Study)</td><td>13.37</td><td>21.90</td><td>8.49</td><td>3.27</td><td>6.75</td><td>9.15</td><td>1.83</td><td>4.20</td><td>4.16</td></tr></table>

\* Due to the use of fewer pre-training steps, these metrics may difer from those reported in their paper.

## 5.2.2. Reconstruction Results and Visualization

Figure 5 visualizes reconstruction accuracy on a randomly selected PeMS node, where STOT closely tracks the ground truth even under abrupt changes, such as the

![](images/0430292e255ec3d6742125eecc324381c8d6adde8e8f4af47abaeba89eb7669b.jpg)  
Figure 3: Multi-step prediction comparison on PeMS03, PeMS04, PeMS07, and PeMS08.

significant PeMS03 fluctuation during $t \in [ 8 5 0 , 1 1 0 0 ]$ , because its adaptive masking emphasizes spatially indistinguishable, high-variation timesteps.

## 5.2.3. Forecasting Results and Visualization

Figure 6 shows forecasting accuracy on a randomly selected node, where predicted and true values align closely; the gain over the baseline without pre-training confirms that pre-training focuses the model on challenging nodes and improves its capture of complex spatiotemporal patterns.

## 5.3. Ablation Study (RQ2)

This section first evaluates the mask pre-training strategy in Section 5.3.1, then assesses STOT’s generalization ability in Section 5.3.2.

## 5.3.1. Mask Ablation

To investigate each component of the designed mask pre-training strategy, we define five variants: w/o Random walk mask removes the post-OT random walk step and reduces mask-generation randomness; w/o Optimal transport constraints directly uses the most spatially indistinguishable positions as masks; w/o Batch consistency constraints allows masks at any position without penalty; w/o Spatial indistinguishability guidance removes this guidance and generates fully random masks; and w/o Mask applies no masked pre-training.

![](images/ab0102ff155a0fea0fbb93a88d3e3a26ecd8d5e47813a63134a875424a8f3d74.jpg)  
Figure 4: Mask pre-training component ablation on PeMS04 and PeMS08.

The ablation results in Figure 4 show that removing the final random walk reduces mask diversity and creates static masks for identical sequences, hindering spatial dependency capture and confirming its role in spatial understanding; OT-based softening with a temperature parameter enables simultaneous temporal and spatial learning, replacing separate pre-training stages with an eficient single-step process; removing batch consistency causes severe degradation, which may indicate that semantic continuity is needed for coherent representations, accuracy, and robustness; eliminating spatial consistency supervision yields fully random masks and what may be trivial semantics, possibly reducing pre-training efectiveness and underscoring the need for guided mask generation; and removing masking strategies makes the model depend entirely on the predictor, highlighting the role of pre-training in accuracy, robustness, and generalization.

![](images/484e7bc6f7763f9c9b3e8e0e0ab7a5dd0a0fbe5b315415bced91b05eec5ff3de.jpg)

![](images/2f4548262e5573c889ffec45808e69a6682b4b90f43669bef1dd6c7341d85f3b.jpg)

![](images/e90429665825c3b9c4f84d60798af2ab16465a747da5db8cc700157688777777.jpg)

![](images/b093c4ab2b95391849d62b2fa3ad501e9baa6e630eb9be2e558ee95d3f2435fa.jpg)  
Figure 5: STOT reconstruction visualization on four datasets during pre-training.

## 5.3.2. Predictor Ablation

To demonstrate that STOT enhances diverse backbones, we use it as a pre-trained feature extractor for LSTM, ASTGCN, and GWNet. The results in Figure 7 show that STOT benefits downstream predictors despite architectural diferences: before STOT, their performance difers substantially, whereas after integration the gap narrows, indicating stronger feature extraction, deeper spatiotemporal dependency modeling, and more accurate and robust prediction.

## 5.4. High Spatial Indistinguishability Prediction Performance (RQ3)

To evaluate performance under high spatial indistinguishability, we test STOT on PeMS04-s, a PeMS04 subset whose nodes have above-average spatial indistinguishability (Table 5). Table 6 summarizes the averaged results from three random experiments for all models, where STOT consistently outperforms baselines in longer-step prediction, captures long-term dependencies better, and maintains robust, stable performance under high indistinguishability, demonstrating reliable modeling of complex spatiotemporal dynamics when spatial patterns are dificult to discern.

![](images/7d7bb6378b58bd2079cac1a30cf71ebf843c909928fb2962bbccae68bf352212.jpg)

![](images/a4c0dbac53422cc3088fbb74954a3dac3cba7c4b2187acfc554e286d97a1444a.jpg)

![](images/16c8382ae8de0706b455badaf8b2abdf33331425d9da98e0f8d4c5c9a0874c92.jpg)

![](images/282edb74f7b0ac560311f263bcbebd649fc912aa2b8ff7e0c41361aec3e45776.jpg)  
Figure 6: STOT forecasting visualization on four datasets after pre-training.

Table 5: Dataset segmentation details.
<table><tr><td>Datasets</td><td>#Sensors #Edges</td><td></td><td>#Training</td><td>#Validation</td><td>#Testing</td></tr><tr><td>PeMS04 (IN)</td><td>307</td><td>340</td><td>[10173, 12, 307,1]</td><td>[3375, 12, 307,1]</td><td>[3375, 12, 307,1]</td></tr><tr><td>PeMS04 (OUT)</td><td>307</td><td>340</td><td>[10173, 12, 307,1]</td><td>[3375, 12, 307,1]</td><td>[3375, 12, 307,1]</td></tr><tr><td>PeMS04-s (IN)</td><td>161</td><td>188</td><td>[10173, 12, 161,1]</td><td>[3375, 12, 161,1]</td><td>[3375, 12, 161,1]</td></tr><tr><td>PeMS04-s (OUT)</td><td>161</td><td>188</td><td>[10173, 12, 161,1]</td><td>[3375, 12, 161,1]</td><td>[3375, 12, 161,1]</td></tr></table>

## 5.5. Hyperparameter Sensitivity (RQ4)

We investigate STOT’s sensitivity to learning rate, hidden dimension, and encoder depth. For Learning Rate, we test $\{ 1 \mathrm { e } ^ { - 3 } , 3 \mathrm { e } ^ { - 3 } , 5 \mathrm { e } ^ { - 3 } , 1 \mathrm { e } ^ { - 2 } \}$ , and Figure 8(a) shows that accuracy decreases as the rate rises from $\mathrm { l e } ^ { - 3 }$ to $\mathrm { l e } ^ { - 2 }$ , with $\mathrm { l e } ^ { - 3 }$ best balancing convergence speed and stability. For Hidden Dimension, we evaluate {32, 48, 96, 128}, and Figure 8(b) shows peak performance at 96, while larger dimensions slightly degrade results, suggesting that overly high-dimensional representations may introduce noise. For Number of Encoder Layers, we vary depth over {1, 2, 3, 4}, and Figure 8(c) shows that 2 layers perform best, indicating suficient capacity for essential spatiotemporal dependencies while deeper encoders provide limited additional benefit.

![](images/697a57378cdcdfd9aca32221dfbdb9ad299b6f9114eac3f6d02035ff840d44cd.jpg)  
Figure 7: Mask pre-training component ablation on PeMS03 and PeMS07.

## 5.6. Computation Cost (RQ5)

Although STOT’s autoencoder-based spatiotemporal representation learning may raise eficiency concerns, its dynamic mask strategy mitigates them. Figure 9 reports per-sample pre-training and forecasting time on four datasets, showing that STOT is more eficient than other pre-training models, reduces spatial representation learning cost, and runs nearly twice as fast as STD-MAE while balancing competitive performance on the evaluated benchmarks and practical feasibility. For reproducibility, the timing comparison in Figure 9 was measured on a single NVIDIA RTX A6000 GPU with a batch size of 8 for every model, matching the hardware reported in our experimental settings.

We further analyzed how mask generation scales with the number of sensor nodes, since the Sinkhorn solver is only one substep of the full mask generation pipeline. Table 7 reports, for each dataset, the node count together with the time of the complete mask generation and of the Sinkhorn substep alone, measured per batch on a single A6000 GPU. The full mask generation grows with the node count and reaches about 5.04 ms per batch on PeMS07 with 883 nodes, while the Sinkhorn substep stays close to 0.214 to 0.216 ms across all datasets. This stability arises because the optimal transport is solved on the temporal patch dimension after the node dimension has already

![](images/c4420852c67182212bb810694e41888357b85e2cc97ae54816789f63a7f6f970.jpg)

![](images/8bf4ac57dc3aa55c50d1bd9050ed28b83d9896e945885eddb0ed85c4cbfa9910.jpg)

![](images/0ebb4474bebedc780e08035b9f502b1f8edb0afcf7cb0e475be1fae8a9117300.jpg)

![](images/a865fcb2f34148c8ff4a4100dd22d6048aa96975cc8b3ef6e8eb13d59494939b.jpg)

![](images/2ef3106e46885b6386bfa7d1a7d32f8c70fe8ec6c5cb4bb2037755d435d3118e.jpg)

![](images/3d67725cf6e5fab15d303dd77bef8d79362c0620f28e771e25b7c06c8073a50f.jpg)

![](images/77665fb41f4c5c005dc8dee64777b90a40bac653ea98cfe0b86a28e827d3595e.jpg)

![](images/6356086f1ca7a86965c635119acad0ef8079f6f1f2b504b3317ea837ea5c85cf.jpg)

![](images/1a5d3c195dda0e3b71e1b90d9696f60093ac34df7a1f28edf710fc88856b8461.jpg)  
Figure 8: STOT parameter sensitivity on PeMS04 under MAE, RMSE, and MAPE.

![](images/bb5421563bf37ab2cfcb2c0f5af71a75cec9dbe2f02a21345ebffe0ab7138af7.jpg)  
Figure 9: Time cost of diferent approaches.

Table 6: High spatial indistinguishability prediction on PeMS04.
<table><tr><td>Model</td><td>Metrics</td><td>#1 step</td><td>#3 steps</td><td>#6 steps</td><td>#12 steps</td></tr><tr><td rowspan="3">STID</td><td>MAE</td><td> $1 6 . 8 8 { \pm } 0 . 0 3 $ </td><td> $1 8 . 1 2 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $1 9 . 1 2 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $2 1 . 1 0 { \pm } 0 . 2 5 $ </td></tr><tr><td>RMSE</td><td> $2 7 . 2 9 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $2 9 . 3 1 { \pm } 0 . 0 3 $ </td><td> $3 0 . 8 2 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $3 3 . 3 2 { \scriptstyle \pm 0 . 2 7 }$ </td></tr><tr><td>MAPE (%)</td><td> $1 1 . 1 9 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $1 2 . 0 7 { \scriptstyle \pm 0 . 2 3 }$ </td><td> $1 2 . 5 4 { \pm } 0 . 1 1 $ </td><td> $1 3 . 7 6 { \pm } 0 . 0 5$ </td></tr><tr><td rowspan="3">ST-Norm</td><td>MAE</td><td> $1 6 . 2 6 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $1 8 . 2 3 { \pm } 0 . 0 3 $ </td><td> $1 9 . 5 8 { \pm } 0 . 0 7$ </td><td> $2 1 . 6 6 { \pm } 0 . 1 4$ </td></tr><tr><td>RMSE</td><td> $2 6 . 2 0 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $3 0 . 0 0 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $3 2 . 3 8 { \pm } 0 . 0 7$ </td><td> $3 5 . 3 7 { \scriptstyle \pm 0 . 1 7 }$ </td></tr><tr><td>MAPE (%)</td><td> $7 . 1 8 { \pm } 0 . 2 1$ </td><td> $8 . 0 1 { \pm } 0 . 2 3 $ </td><td> $8 . 4 1 { \pm } 0 . 2 7 $ </td><td> $9 . 5 5 { \pm } 0 . 3 2 $ </td></tr><tr><td rowspan="3">STEP</td><td>MAE</td><td> $1 6 . 8 8 { \pm } 0 . 0 2 $ </td><td> $1 8 . 1 3 { \pm } 0 . 0 1$ </td><td> $1 9 . 1 3 { \pm } 0 . 0 0 $ </td><td> $2 0 . 8 8 { \pm } 0 . 1 0 $ </td></tr><tr><td>RMSE</td><td> $2 7 . 2 5 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $2 9 . 3 1 { \pm } 0 . 0 1$ </td><td> $3 0 . 8 5 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $3 3 . 1 1 { \pm } 0 . 0 7 $ </td></tr><tr><td>MAPE (%)</td><td> $1 1 . 6 6 { \pm } 0 . 1 6$ </td><td> $1 2 . 4 6 { \pm } 0 . 1 0 $ </td><td> $1 2 . 9 7 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $1 4 . 5 0 { \pm } 0 . 3 5 $ </td></tr><tr><td rowspan="3">STD-MAE</td><td>MAE</td><td> $1 6 . 8 6 { \pm } 0 . 0 3$ </td><td> $1 8 . 1 0 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $1 9 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $2 0 . 4 0 { \scriptstyle \pm 0 . 0 3 }$ </td></tr><tr><td>RMSE</td><td> $2 7 . 3 0 { \pm } 0 . 0 3 $ </td><td> $2 9 . 2 7 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $3 0 . 6 8 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $3 2 . 6 2 { \scriptstyle \pm 0 . 1 1 }$ </td></tr><tr><td>MAPE (%)</td><td> $1 1 . 3 4 { \pm } 0 . 3 7$ </td><td> $1 2 . 1 9 { \scriptstyle \pm 0 . 2 2 }$ </td><td> $1 2 . 9 5 { \pm } 0 . 1 6$ </td><td> $1 3 . 8 6 { \pm } 0 . 1 7 $ </td></tr><tr><td rowspan="3">STOT</td><td>MAE</td><td> $1 6 . 5 4 { \pm } 0 . 0 1$ </td><td> $1 7 . 7 3 { \pm } 0 . 0 1$ </td><td> $1 8 . 5 7 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $1 9 . 8 9 { \scriptstyle \pm 0 . 0 5 }$ </td></tr><tr><td>RMSE</td><td> $2 6 . 8 5 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $2 8 . 8 7 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $3 0 . 2 3 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $3 1 . 9 7 { \scriptstyle \pm 0 . 1 1 }$ </td></tr><tr><td>MAPE (%)</td><td> $1 0 . 7 9 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $1 1 . 7 5 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $1 2 . 1 7 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $1 3 . 3 0 { \pm } 0 . 0 9$ </td></tr></table>

been aggregated into the T score, so the solver cost is largely independent of the network size. These measurements indicate that the masking stays practical for large-scale urban sensor networks.

Table 7: Mask generation time and the Sinkhorn substep time, in milliseconds per batch, across datasets with diferent numbers of nodes.
<table><tr><td>Dataset</td><td>Nodes</td><td>Mask generation</td><td>Sinkhorn</td><td></td></tr><tr><td>PeMS08</td><td>170</td><td> $1 . 3 4 6 6 \pm 0 . 0 8 5 2$ </td><td> $0 . 2 1 3 8 \pm 0 . 0 0 3 6$ </td><td rowspan="5"></td></tr><tr><td>METR-LA</td><td>207</td><td> $1 . 4 6 4 4 \pm 0 . 1 2 3 8$ </td><td> $0 . 2 1 5 0 \pm 0 . 0 0 3 5$ </td></tr><tr><td>PeMS04</td><td>307</td><td> $1 . 3 8 3 7 \pm 0 . 0 8 9 1$ </td><td> $0 . 2 1 3 7 \pm 0 . 0 0 0 9$ </td></tr><tr><td>PeMS-BAY</td><td>325</td><td> $1 . 3 8 9 0 \pm 0 . 0 9 6 3$ </td><td> $0 . 2 1 4 9 \pm 0 . 0 0 3 4$ </td></tr><tr><td>PeMS03</td><td>358</td><td> $1 . 4 2 9 4 \pm 0 . 0 5 5 6$ </td><td> $0 . 2 1 5 4 \pm 0 . 0 0 2 9$ </td></tr><tr><td>PeMS07</td><td>883</td><td> $5 . 0 4 1 9 \pm 0 . 3 5 1 4$ </td><td> $0 . 2 1 5 7 \pm 0 . 0 0 3 8$ </td></tr></table>

## 5.7. OT Constraint Visualization (RQ6)

To interpret STOT’s OT efectiveness, we visualize the cost matrix, time-varying features, OT-generated masks, and batch consistency efects. Figure 10(a) and Figure 10(b) show the cost matrix before and after batch consistency constraints, with input timesteps on the horizontal axis, a batch size of 8 on the vertical axis, and color intensity indicating the cost magnitude that controls the likelihood of masking each position.

The original cost matrix in Figure 10(a) has a highly concentrated color distribution; after applying batch consistency in Figure 10(b), the distribution remains concentrated but gains more overlapping regions. These regions encourage masks at similar, though not identical, positions within a batch, which may enhance semantic understanding and improve downstream performance.

Figure 10(c) shows that spatial indistinguishability has clear temporal periodicity: the lowest costs occur during morning and evening peaks, indicating the most severe indistinguishability and the greatest prediction challenge, while weaker of-peak fluctuations are less likely to be bottlenecks. This illustrates how SI-guided masking emphasizes challenging peak-hour segments, helping STOT learn richer spatiotemporal dependencies and improve accuracy.

Figure 10(d) shows that transport-matrix peaks align with the low-cost regions in Figure 10(c), validating the Sinkhorn algorithm and matching our design expectations. The peak positions also vary slightly across batches, with three distinct peaks in the time range [40, 60], which may reflect STOT’s batch-adaptive mask adjustment for more accurate prediction.

The transport-matrix heatmap in Figure 10(e) intuitively shows preferred mask locations, where masks are continuous within batches and periodic across timesteps. Figure 10(f) further shows high inter-batch correlation among generated masks, which may facilitate optimization and enhance overall performance.

Together, these visualizations support the efectiveness of STOT’s OT process: transport peaks align with low-cost regions, mask positions adjust dynamically across batches, masks remain continuous and periodic, and high inter-batch correlation improves optimization and prediction accuracy. These findings validate STOT’s design choices, clarify its internal behavior, and motivate future improvements for spatiotemporal prediction.

We further quantified how the batch consistency weight λ in Equation 7 trades of consistency against diversity. For a fixed training batch we regenerated the masks across a range of λ without retraining, measured consistency as the average pairwise

![](images/835cb86e4f70fb94442045bd14f84e27fec4be1a68dc8e635ef19e67dc8e1149.jpg)  
(a)

![](images/f645015aec76025fe6a967f52badbe755267e4cce8f741b5cb358c445e6ba443.jpg)  
(b)

![](images/b08c2bebc2f695c165c44dbec3a904978a89354eb60bd599bcc8446f9dad9dde.jpg)

![](images/7751866725ac64705b2be28db7c1951c43b8116208ef02c62c7c5c21dd27e89d.jpg)

![](images/901af806ea371fbea7ef95c9c9879ea3850905d93f88a04648cfb6b53bd4e9c9.jpg)

![](images/a4f01fa46d0cf11c5cf1401a8595c8be9579f06a4a2bd858f9ec15703ca57278.jpg)  
Figure 10: Visualization of OT and batch consistency: (a) and (b) cost matrices before and after batch consistency for the first 72 timesteps (6 patches), (c) and (d) the final cost matrix and OT matrix, (e) transportmatrix heatmap, and (f) batch-mask correlation matrix showing the efect of batch consistency on mask generation.

Jaccard similarity among the batch masks, used its complement as diversity, and combined the normalized diversity with the mean Tscore of the selected patches into a balance score. As reported in Figure 11, the balance score stays near its peak while λ remains small and declines once λ exceeds 0.7, while consistency keeps rising and diversity keeps shrinking. The default $\lambda = 0 . 7$ therefore sits at the point where the masks stay diverse enough to cover varied scenarios while the consistency term already aligns them across the batch, and this behavior holds on all six datasets.

![](images/88448ad1188340952ca5e02e21c8ee679fe1d952baa3cac67078ac0990f82141.jpg)  
Figure 11: Consistency, diversity, and balance scores as functions of the batch consistency weight λ on the six datasets. The dashed line marks the default λ = 0 7.

## 5.8. Random Walk Mechanism for Mask Generation (RQ7)

![](images/ad256103d199dea1feb00cc2538e0dcc4e4f822c01573fbfba7da1054b71bb5c.jpg)  
Figure 12: Comparison of random walk mechanism variants on PeMS04.

To assess the random walk design, we test alternative implementations that operate

on diferent positions or aspects and could substitute for the original mechanism:

• w/o Patch-based: The smallest unit for the random walk is a single time step, as opposed to a patch of time steps.

• w/o Batch-based: The random walk does not apply simultaneously to the entire batch, but rather to individual samples within the batch.

• w/o After OT: The random walk is performed before the OT process, instead of after it.

Figure 12 reports training and testing losses for diferent random walk designs, leading to the following conclusions:

• w/o Patch-based: Applying random walk at the timestep level simplifies intertimestep semantics and makes deep representation learning harder; however, because batch consistency remains, optimization is not strongly afected, and the model resembles the T-MAE design in STD-MAE, achieving relatively good results among variants.

• w/o Batch-based: Disrupting batch consistency within one optimization step produces fragmented and possibly less meaningful semantics, so this design may underperform the Patch-based approach that preserves semantic continuity.

• w/o After OT constraint: Performing random walk before OT makes randomness depend on cost-matrix softening; temperature adjustment partly mitigates this, but mask-generation randomness is still greatly reduced, possibly leading to redundant semantics and the lowest performance.

Overall, all three variants converge to nearly identical training losses, which may indicate that they can almost fully reconstruct the sequence, while their performance diferences arise from where random walk is applied. Our design avoids the above issues and enables comprehensive spatiotemporal modeling.

## 6. Conclusion

This paper introduced STOT, a self-supervised spatiotemporal prediction framework that improves representation learning under the common but underexplored challenge of spatial indistinguishability. Instead of treating masking as random or heuristic, STOT uses an optimal transport (OT)-based strategy to construct training signals aligned with intrinsic spatial structure. Across multiple real-world benchmarks, STOT improves over strong baselines, suggesting that explicitly modeling ambiguity among spatially similar regions may ofer a forecasting advantage.

A key strength of our approach is that the OT-based masking mechanism provides both efectiveness and interpretability. Empirically, the strongest gains appear during peak hours, where demand patterns are volatile and prediction becomes most dificult, suggesting that the proposed masking strategy is especially beneficial when the model must discriminate among multiple plausible spatial explanations. The transport matrices ofer an interpretable lens into how the model perceives ”substitutable” or low-cost spatial regions, and the observed alignment between low-cost regions and high transport mass supports the intuition that OT can serve as a principled tool for constructing challenging but semantically meaningful self-supervised objectives. The batch consistency constraint further strengthens the learned representations by encouraging coherent semantics while allowing suficient diversity, helping to stabilize training and improve generalization. Since the reconstruction objective draws on a longer context and a direct target, it is easier to optimize than forecasting, so the pre-training losses of diferent variants settle at close values, and the downstream gains instead come from masking that concentrates the reconstruction on the spatially indistinguishable nodes. However, our work has several limitations. The computational overhead introduced by OT can become a bottleneck when the number of spatial units grows or when extremely long sequences are used, raising scalability concerns for real-world deployments on city-scale or nation-scale grids. Additionally, STOT focuses primarily on endogenous spatiotemporal signals and may not fully capture the causal drivers of variation when exogenous variables (e.g., weather, holidays, accidents) dominate the dynamics. Finally, the current analysis does not fully characterize when OT-based masking is most beneficial, leaving open questions about robustness under distribution shifts. In addition, the optimal transport objective is easier to optimize under a moderate mask ratio, so STOT does not exploit the high mask ratios that benefit single-stage masking methods such as STEP [28], which leaves room for transport formulations that stay stable under heavier masking.

Beyond performance, STOT ofers two practical contributions: it provides a general framework for self-supervised objectives that address spatial ambiguity in trafic and environmental monitoring, with OT-based masking usable as a plug-in strategy under limited labels; and its transport matrices act as interpretable diagnostics for spatial similarity, data quality assessment, and decision policy design. Future work includes faster OT approximations for large-scale graphs, exogenous covariates for robustness, and extensions to anomaly detection, imputation, and cross-city transfer.

## CRediT authorship contribution statement

Guangyu Wang: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing – original draft, Visualization. Jiawei Tong: Conceptualization, Methodology, Writing – review & editing, Supervision, Project administration.

## Acknowledgements

The authors would like to thank Chen Yu for language polishing and proofreading, and Yujie Chen for assistance with manuscript formatting and typesetting.

## References

[1] J. Wang, J. Ji, Z. Jiang, L. Sun, Trafic Flow Prediction Based on Spatiotemporal Potential Energy Fields, IEEE Transactions on Knowledge and Data Engineering 35 (2023) 9073–9087.

[2] L. Xu, N. Chen, Z. Chen, C. Zhang, H. Yu, Spatiotemporal forecasting in earth system science: Methods, uncertainties, predictability and future directions, Earth-Science Reviews 222 (2021) 103828.

[3] Q. Lv, L. Liu, R. Yang, Y. Wang, Multimodal urban trafic flow prediction based on multi-scale time series imaging, Pattern Recognition 164 (2025) 111499.

[4] Z. Jiang, C. Liu, A. Akintayo, G. P. Henze, S. Sarkar, Energy prediction using spatiotemporal pattern networks, Applied Energy 206 (2017) 1022–1039.

[5] M. Awad, R. Khanna, M. Awad, R. Khanna, Support vector regression, Eficient learning machines: Theories, concepts, and applications for engineers and system designers (2015) 67–80.

[6] S. J. Rigatti, Random forest, Journal of Insurance Medicine 47 (2017) 31–39.

[7] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, T.-Y. Liu, Lightgbm: A highly eficient gradient boosting decision tree, Advances in neural information processing systems 30 (2017).

[8] S. Pal, L. Ma, Y. Zhang, M. Coates, Rnn with particle flow for probabilistic spatio-temporal forecasting, in: International Conference on Machine Learning, PMLR, 2021, pp. 8336–8348.

[9] Y. Li, R. Yu, C. Shahabi, Y. Liu, Difusion convolutional recurrent neural network: Data-driven trafic forecasting, in: International Conference on Learning Representations, 2018.

[10] J. Deng, X. Chen, R. Jiang, X. Song, I. W. Tsang, St-norm: Spatial and temporal normalization for multi-variate time series forecasting, in: Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, 2021, pp. 269–278.

[11] Z. Shao, Z. Zhang, F. Wang, W. Wei, Y. Xu, Spatial-temporal identity: A simple yet efective baseline for multivariate time series forecasting, in: Proceedings of the 31st ACM International Conference on Information & Knowledge Management, 2022, pp. 4454–4458.

[12] G. Zou, Z. Zhou, R. Weibel, Y. Li, T. Wang, Z. Liu, W. Ding, C. Fu, Multi-graph spatio-temporal network for trafic accident risk forecasting, Pattern Recognition 172 (2026) 112784.

[13] Z. Shao, F. Wang, Y. Xu, W. Wei, C. Yu, Z. Zhang, D. Yao, T. Sun, G. Jin, X. Cao, G. Cong, C. S. Jensen, X. Cheng, Exploring progress in multivariate time series forecasting: Comprehensive benchmarking and heterogeneity analysis, IEEE Transactions on Knowledge and Data Engineering 37 (2025) 291–305. doi:10.1109/TKDE.2024.3484454.

[14] B. Yu, H. Yin, Z. Zhu, Spatio-temporal graph convolutional networks: a deep learning framework for trafic forecasting, in: Proceedings of the 27th International Joint Conference on Artificial Intelligence, 2018, pp. 3634–3640.

[15] Z. He, C.-Y. Chow, J.-D. Zhang, Stnn: A spatio-temporal neural network for trafic predictions, IEEE Transactions on Intelligent Transportation Systems 22 (2021) 7642–7651.

[16] C. Shang, J. Chen, J. Bi, Discrete graph structure learning for forecasting multiple time series, 2021. arXiv:2101.06861.

[17] Y. Li, Z. Shao, Y. Xu, Q. Qiu, Z. Cao, F. Wang, Dynamic frequency domain graph convolutional network for trafic forecasting, 2023. arXiv:2312.11933.

[18] J. Lin, Q. Ren, X. Lv, H. Xu, Y. Liu, When multi-view meets multi-level: A novel spatio-temporal transformer for trafic prediction, Information Fusion 117 (2025) 102801.

[19] K. N. Kumar, D. Roy, T. A. Suman, C. Vishnu, C. K. Mohan, Tsanet: Forecasting trafic congestion patterns from aerial videos using graphs and transformers, Pattern Recognition 155 (2024) 110721.

[20] H. Liu, Z. Dong, R. Jiang, J. Deng, J. Deng, Q. Chen, X. Song, Spatio-temporal adaptive embedding makes vanilla transformer sota for trafic forecasting, in: Proceedings of the 32nd ACM International Conference on Information and Knowledge Management, 2023, pp. 4125–4129.

[21] R.-G. Cirstea, B. Yang, C. Guo, T. Kieu, S. Pan, Towards spatio-temporal aware trafic time series forecasting, in: 2022 IEEE 38th International Conference on Data Engineering (ICDE), IEEE, 2022, pp. 2900–2913.

[22] Z. Dong, R. Jiang, H. Gao, H. Liu, J. Deng, Q. Wen, X. Song, Heterogeneityinformed meta-parameter learning for spatiotemporal time series forecasting, in: Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2024, pp. 631–641.

[23] G. Wang, Z. Wang, Net-ev<sup>2</sup>: A generative simulator for network event evolution, in: Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, ACM, 2026, pp. 4836–4847. doi:10.1145/3770855. 3817972.

[24] X. Yuan, Y. Qiao, Difusion-TS: Interpretable difusion for general time series generation, in: The Twelfth International Conference on Learning Representations, 2024.

[25] H. Gao, R. Jiang, Z. Dong, J. Deng, Y. Ma, X. Song, Spatial-temporal-decoupled masked pre-training for spatiotemporal forecasting, 2024.

[26] H. Bao, L. Dong, S. Piao, F. Wei, Beit: Bert pre-training of image transformers, in: International Conference on Learning Representations, 2021.

[27] Z. Lan, M. Chen, S. Goodman, K. Gimpel, P. Sharma, R. Soricut, Albert: A lite bert for self-supervised learning of language representations, arXiv preprint arXiv:1909.11942 (2019).

[28] Z. Shao, Z. Zhang, F. Wang, Y. Xu, Pre-training enhanced spatial-temporal graph neural network for multivariate time series forecasting, in: Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD ’22, ACM, 2022, p. 1567–1577.

[29] Z. Li, Z. Rao, L. Pan, P. Wang, Z. Xu, Ti-mae: Self-supervised masked time series autoencoders, 2023. arXiv:2301.08871.

[30] S. Lee, T. Park, K. Lee, Learning to embed time series patches independently, 2024. arXiv:2312.16427.

[31] J. Dong, H. Wu, H. Zhang, L. Zhang, J. Wang, M. Long, Simmtm: A simple pre-training framework for masked time-series modeling, in: Advances in Neural Information Processing Systems, 2023.

[32] Y. Nie, N. H. Nguyen, P. Sinthong, J. Kalagnanam, A time series is worth 64 words: Long-term forecasting with transformers, 2023. arXiv:2211.14730.

[33] A. Genevay, G. Peyre, M. Cuturi, Learning generative models with sinkhorn´ divergences, in: International Conference on Artificial Intelligence and Statistics, PMLR, 2018, pp. 1608–1617.

[34] Z. Wu, S. Pan, G. Long, J. Jiang, C. Zhang, Graph wavenet for deep spatialtemporal graph modeling., in: IJCAI, 2019.

[35] S. Guo, Y. Lin, N. Feng, C. Song, H. Wan, Attention based spatial-temporal graph convolutional networks for trafic flow forecasting, in: Proceedings of the AAAI conference on artificial intelligence, volume 33, 2019, pp. 922–929.

[36] T. Afrin, N. Yodo, A survey of road trafic congestion measures towards a sustainable and resilient transportation system, Sustainability (2020) 4660.

[37] R.-G. Cirstea, T. Kieu, C. Guo, B. Yang, S. J. Pan, Enhancenet: Plugin neural networks for enhancing correlated time series forecasting, in: 2021 IEEE 37th International Conference on Data Engineering (ICDE), IEEE, 2021, pp. 1739– 1750.

[38] J. Choi, H. Choi, J. Hwang, N. Park, Graph neural controlled diferential equations for trafic forecasting, in: AAAI, 2022.

[39] B. M. Williams, P. K. Durvasula, D. E. Brown, Urban freeway trafic flow prediction: application of seasonal autoregressive integrated moving average and exponential smoothing models, Transportation Research Record 1644 (1998) 132–141.

[40] S. R. Chandra, H. Al-Deek, Predictions of freeway trafic speeds and volumes using vector autoregressive models, Journal of Intelligent Transportation Systems 13 (2009) 53–72.

[41] Z. Fang, Q. Long, G. Song, K. Xie, Spatial-temporal graph ode networks for trafic flow forecasting, in: Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, 2021, pp. 364–373.

[42] S. Lan, Y. Ma, W. Huang, W. Wang, H. Yang, P. Li, Dstagnn: Dynamic spatialtemporal aware graph neural network for trafic flow forecasting, in: International conference on machine learning, PMLR, 2022, pp. 11906–11917.

[43] S. Guo, Y. Lin, H. Wan, X. Li, G. Cong, Learning dynamics and heterogeneity of spatial-temporal graph data for trafic forecasting, IEEE Transactions on Knowledge and Data Engineering (2021).

[44] L. Bai, L. Yao, C. Li, X. Wang, C. Wang, Adaptive graph convolutional recurrent network for trafic forecasting, Advances in Neural Information Processing Systems 33 (2020) 17804–17815.

[45] J. Jiang, C. Han, W. X. Zhao, J. Wang, Pdformer: Propagation delay-aware dynamic long-range transformer for trafic flow prediction, in: AAAI, AAAI Press, 2023.

[46] L. Cao, B. Wang, G. Jiang, Y. Yu, J. Dong, Spatiotemporal-aware trendseasonality decomposition network for trafic flow forecasting, Proceedings of the AAAI Conference on Artificial Intelligence 39 (2025) 11463–11471.