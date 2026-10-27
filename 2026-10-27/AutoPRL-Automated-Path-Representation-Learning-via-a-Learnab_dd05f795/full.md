# AutoPRL: Automated Path Representation Learning via a Learnable View Generator

Sean Bin Yang, Member, IEEE, Ying Sun, Yunyao Cheng, Hao Miao, Member, IEEE, Jilin Hu<sup>\*</sup>, Member, IEEE, Bin Yang, Senior Member, IEEE, Torben Bach Pedersen, Senior Member, IEEE, and Christian S. Jensen, Fellow, IEEE

Abstract—Self-supervised learning has emerged as a powerful paradigm for Path Representation Learning (PRL) in road networks. PRL aims to learn informative representations of paths traversed by moving objects, such as vehicles, where each path consists of a sequence of connected road segments. These representations are essential for various downstream tasks, including travel-time estimation and path ranking. However, existing methods often suffer from ineffective view generation and inefficient training procedures due to their reliance on heuristic augmentations and computationally intensive contrastive learning. To address these issues, we propose AutoPRL, an effective and efficient self-supervised PRL framework that integrates three key components. The Automatic Data Augmentation (AutoDA) module employs variational autoencoders to generate semantically meaningful and diverse feature-level path views by modeling each topology-aware node embedding with a Gaussian latent distribution, thereby mitigating information loss and enhancing representation quality. Second, the Cross-View Decoupled Contrastive (CVDC) learning mechanism improves training efficiency by disentangling the optimization of positive and negative pairs across node- and path-level views, enabling more stable contrastive learning. Third, an Adaptive Co-Training (ACT) module dynamically adjusts uncertainty-based loss weights to balance multiple objectives, ensuring efficient and scalable training. Extensive experiments on three real-world datasets demonstrate that AutoPRL achieves strong and consistent performance in both effectiveness and efficiency across multiple downstream tasks.

Index Terms—Self-Supervised Learning, Path Representation Learning, Automatic Data Augmentation.

## I. INTRODUCTION

W <sup>ITH</sup> <sup>the</sup> <sup>rapid</sup> <sup>proliferation</sup> <sup>of</sup> <sup>GPS-enabled</sup> <sup>devices</sup>and ubiquitous mobile computing services, massive and ubiquitous mobile computing services, massive volumes of path data are being continuously collected to capture the mobility patterns of vehicles. Effective analysis of such data is fundamental to a broad range of intelligent transportation applications, including travel time estimation [1]–[6], trajectory mining [7]–[14], and path recommendation [2]–[4], [15]. Path representation learning (PRL), an essential and popular path data analytics technique, aims at learning a generic path representation that can support various downstream tasks, rather than being tailored to a specific application [2]–[4], [16]–[18].

![](images/1c6c82e3c7f529e58235cc4aa14bcb4d3701cf4bc0d30862e5a12a0807bfe509.jpg)  
Fig. 1: Architectures of the conventional contrastive paradigm and our AutoPRL for path representation learning. In (a), heuristic augmentation directly perturbs the discrete path sequence. In (b), AutoDA generates feature-level views as continuous embedding sequences from topology-aware node embeddings, rather than discrete node sequences on the roadnetwork graph. Contrast denotes the contrastive loss.

Due to its significance, substantial research efforts have been devoted to developing effective and efficient methods for path representation learning. Among them, contrastive learningbased approaches have garnered considerable attention due to their ability to enhance representation generalization by pulling positive samples closer while pushing negative samples apart in the embedding space [2]–[4], [19]–[22].

More specifically, as illustrated in Figure 1 (a), most existing PRL methods adopt a contrastive learning paradigm comprising three stages: (1) Given an input path (e.g., $\langle v _ { 1 } , v _ { 2 } , v _ { 3 } , v _ { 4 } , v _ { 5 } \rangle )$ , augmented views are generated using heuristic, prior-based strategies such as random node drop, resulting in path view 1 $( \langle v _ { 1 } , v _ { 2 } , v _ { 3 } \rangle )$ and path view 2 $( \langle v _ { 1 } , v _ { 2 } , v _ { 3 } , v _ { 5 } \rangle )$ ; (2) These path views are encoded via a shared path encoder to obtain latent representations; (3) A contrastive loss is applied to align representations of positive pairs while pushing apart negatives. Despite their simplicity, such pre-defined augmentations are inherently limited: they often lead to significant semantic information loss and lack adaptability to the structural characteristics of individual paths. As a result, the learned representations are not only suboptimal but also suffer from poor generalization across tasks and domains.

Different from existing PRL methods that mainly rely on pre-defined positive samples or manually designed structurelevel perturbations, AutoPRL adapts learnable view generation to the PRL setting and generates probabilistic feature-level path views. It automatically generates semantically consistent feature-level path views in the latent space based on topologyaware node embeddings derived from the road network. The generated views are continuous embedding sequences rather than discrete topology-level perturbations. In addition, AutoPRL adopts a cross-view decoupled contrastive learning mechanism to improve training efficiency and stability, and an adaptive co-training strategy to dynamically balance multiple learning objectives.

These limitations motivate the design of AutoPRL, an effective and efficient framework for learning generalizable path representations through an automatic and learnable view generator. AutoPRL addresses three key challenges:

Challenge I: How to effectively construct path views with minimal information loss? To derive effective and generalizable path representations, existing methods often rely on heuristic, prior-driven view generation strategies. However, such approaches—e.g., random node drop (as illustrated in Figure 1(a), where the generated path view retains only a partial segment of the original path)—can result in significant information loss, thereby limiting the representational capacity of the path encoder and leading to suboptimal performance.

Although recent advances in automatic and learnable view generation have demonstrated effectiveness in graph and vision domains, these techniques are not directly applicable to path representation learning due to the unique spatial dependencies inherent in paths. To this end, as shown in Figure 1(b), AutoPRL generates semantically consistent path views with minimal information loss by leveraging two independent variational autoencoders that explicitly account for spatial dependencies. Specifically, it models each topology-aware node embedding with a Gaussian latent distribution and performs sampling from these distributions to generate diverse yet semantically consistent feature-level views for each input path. The generated views are continuous embedding sequences in the latent space rather than discrete paths on the road-network graph. This probabilistic design introduces adaptive featurelevel variations without directly modifying the underlying road-network topology. This process constitutes an automatic data augmentation module, referred to as AutoDA, which enables AutoPRL to learn more effective and generalizable path representations for various downstream tasks.

Challenge II: How to efficiently conduct contrastive learning for PRL? Prevailing PRL frameworks commonly adopt SimCLR-style contrastive objectives [3], [10], which rely on a large number of negative samples to achieve discriminative representations. Although effective, these methods incur significant computational and memory overhead due to their dependence on large batch sizes, thereby posing challenges for scalability and efficiency. To facilitate more efficient contrastive training, we adapt the decoupled contrastive learning principle to PRL and develop a CVDC learning mechanism that enhances training efficiency by disentangling the optimization of positive and negative sample pairs. By eliminating the competition between positives and negatives within a shared contrastive space, CVDC enables more stable and computationally efficient training of the path encoder.

Challenge III: How to efficiently optimize multi-objective learning in a dynamic setting? Unlike existing fixed-weight multi-objective optimization strategies, where the contribution of each objective remains static throughout training, path representation learning often involves objectives whose gradients vary dynamically during training. This dynamic behavior necessitates an adaptive optimization mechanism that can effectively balance competing objectives in real time to ensure stable and efficient encoder learning. To mitigate this, AutoPRL incorporates an adaptive loss co-training module that learns uncertainty-based weights for each objective. This dynamic weighting strategy facilitates efficient multi-objective optimization and enhances both the stability and efficiency of path encoder training.

In summary, the AutoPRL framework consists of three components: AutoDA, CVDC, and ACT modules.

First, we introduce AutoDA, an automatic and learnable path view generator that adaptively generates semantically meaningful views conditioned on the input path through Gaussian latent modeling, thereby mitigating information loss associated with heuristic augmentations. Second, we propose a CVDC mechanism that separately optimizes cross-node and crosspath objectives. This decoupling strategy allows the encoder to simultaneously capture local structural dependencies among nodes and global semantic patterns across entire paths. By decoupling the competition between positive and negative samples in the shared contrastive space, CVDC facilitates more stable optimization and enhances computational efficiency in path encoder training. Finally, the ACT module dynamically balances multiple learning objectives via uncertainty-based weighting, enhancing both training efficiency and model generalization.

The main contributions of this paper are summarized as follows:

• We propose AutoPRL, an automated path representation learning framework that adapts learnable view generation to PRL. Compared with previous PRL methods that mainly rely on pre-defined positive samples or heuristic augmentations, AutoPRL generates semantically consistent feature-level path views through Gaussian latent modeling, reducing information loss and improving generalization.

• We adapt decoupled contrastive learning to PRL and formulate a cross-view contrastive learning component with cross-path and cross-node objectives. This component captures both global path-level semantics and local nodelevel consistency while improving training efficiency.

• We incorporate an adaptive co-training strategy to jointly train the view generator and path encoder in an end-toend manner. This strategy introduces uncertainty-based adaptive weighting into PRL to dynamically balance multiple loss components, reducing the need for manual weight tuning and improving training stability.

• We conduct extensive experiments on three real-world datasets across multiple downstream tasks to assess the effectiveness and generality of AutoPRL.

## II. RELATED WORK

## A. Trajectory Representation Learning

With the remarkable success of self-supervised learning, the pursuit of generic trajectory representations has received increasing attention, as such representations can support a wide range of downstream applications in urban computing. A variety of methods have been proposed to learn vector representations of trajectories. Existing trajectory representation learning (TRL) approaches can be broadly categorized into spatial trajectory–based (i.e., path-based) methods and spatiotemporal trajectory–based methods.

Spatial Trajectory-based Methods. Spatial Trajectory Representation Learning (STRL), also known as path representation learning (PRL), focuses solely on spatial information to derive a generic path representation. In this setting, existing works are primarily divided into two classes, recurrent neural network (RNN) based methods and Transformer based methods. PIM [3] proposes a first RNN-based mutual information maximization framework to learn generic path representations through curriculum negative sampling. To improve upon this, TPR [4] treats temporal information as weak labels to construct diverse contrastive samples and proposes a weakly supervised contrastive learning method with a curriculum learning strategy based on LSTMs. However, RNN-based methods struggle to capture long-range path dependencies and generally achieve significantly lower accuracy than Transformer-based methods. Building on this, LightPath [2] proposes a scalable and lightweight path representation learning framework that employs varying mask ratios for path augmentation and adopts relational reasoning within a Transformer architecture to learn generic path representations. More recently, MM-Path [17] and Path-LLM [16] extend PRL into the multi-modality domain by incorporating image and textual modalities, respectively. However, these PRL methods still rely on pre-defined augmentation strategies, which limits their generalization ability.

Trajectory-based Methods. In contrast, trajectory representation learning (TRL) aims to learn generic representations by jointly modeling spatial and temporal information, typically through Transformer architectures. Toast [23] introduces a traffic context–aware skip-gram module and a trajectoryenhanced Transformer module to improve trajectory representation learning. START [10] incorporates temporal regularities and travel semantics via a random masking strategy and further enhances learning with data augmentation techniques such as trajectory trimming, temporal shifting, and contrastive learning. TrajRL [24] proposes a self-supervised trajectory representation learning model that captures multi-scale spatiotemporal correlations from both road segments and segment sequences. It further incorporates multi-faceted temporal information, including time intervals, to model context-aware temporal patterns within trajectories. RED [1] leverages comprehensive trajectory information and adopts a road-aware masking strategy to learn more effective trajectory representations. We adopt LightPath, Toast, START, TrajRL and RED as baselines since they are Transformer-based methods, and additionally include PIM as a baseline because it represents a classical path representation learning approach.

## B. Contrastive Learning

In recent years, contrastive learning has made significant strides in unsupervised and self-supervised learning. Its core objective is to maximize similarity between positive pairs while minimizing similarity between negative pairs, using augmented views of the same data. Several frameworks have been proposed, including SimCLR [25], BYOL [26], and MoCo [27]. However, a key challenge remains: model collapse, where embeddings degenerate into constant values, undermining feature learning. To address this, two main strategies have emerged. The first relies on negative samples, which requires large batch sizes and extensive GPU memory (e.g., PIM [3]). The second uses momentum encoders with stop-gradient and sliding average mechanisms to preserve representational diversity. Studies suggest that stop-gradient, rather than the sliding average, plays a central role in preventing collapse. SimSiam [25], and its variant LightPath [2], leverage stopgradient with projection layers, achieving strong performance without large batch sizes or extensive negative sampling. This makes SimSiam-style approaches efficient for downstream tasks. Nonetheless, contrastive learning methods still face challenges in scalability and resistance to collapse, warranting further research into more efficient and effective solutions.

## III. PRELIMINARIES

We proceed to present the necessary preliminaries and then define the problem addressed.

## A. Basic Definition

Definition 1 (Road Network). A road network is represented as a graph $\mathcal { G } = ( \nu , \mathcal { E } )$ , where V is a set of nodes $v _ { i } .$ , each representing a road intersection, and $\mathcal { E } \subseteq \mathcal { V } \times \mathcal { V }$ is a set of edges $e _ { i } ,$ with each edge $e _ { i } = ( v _ { i } , v _ { j } )$ denoting a directed road segment from node $v _ { i }$ to $v _ { j }$

Definition 2 (Path). A path $p = \langle v _ { 1 } , v _ { 2 } , \cdot \cdot \cdot , v _ { n } \rangle$ is defined as a sequence of connected nodes, where each $v _ { i } \in \mathcal V$ represents a node in the path. Specifically, for each adjacent pair of edges, $\boldsymbol { e } _ { i } = ( v _ { i } , v _ { j } )$ and $\boldsymbol { e } _ { i + 1 } = ( v _ { j } , v _ { k } )$ share a common node $v _ { j } .$ where $i , j , k \le n$

Definition 3 (Node Embedding Vector). In a road network graph ${ \mathcal { G } } ,$ , the embedding of a node is denoted by an embedding vector $h _ { n } \in \mathbb { R } ^ { d _ { n } }$ , where $d _ { n }$ represents the dimensionality of the node embedding. This vector is typically obtained using graph representation learning models such as Node2vec [28].

Definition 4 (Path View). Given a path $p _ { i } = \langle v _ { 1 } , v _ { 2 } , \cdot \cdot \cdot , v _ { n } \rangle$ a path view refers to an augmented representation of $p _ { i }$ used as a positive sample in contrastive learning. Depending on the augmentation mechanism, a path view can be either a discrete node sequence generated by heuristic structure-level augmentation (e.g., random node dropping) or a continuous nodeembedding sequence generated by feature-level augmentation. In this work, AutoDA generates the latter type of path view in the latent embedding space.

![](images/8147fd89f44cf369d8f037379eca77365c4a2f7bc9adf61056d65943a97d85d4.jpg)  
Fig. 2: An example of heuristic path view generation.

## B. Problem Definition

Given a set of paths $\mathcal { P } = \left\{ p _ { 1 } , p _ { 2 } , \cdot \cdot \cdot , p _ { | \mathcal { P } | } \right\}$ and a road network graph ${ \mathcal { G } } ,$ AutoPRL aims to learn a generic embedding vector, which can be defined as $P E _ { \psi } ( p _ { i } )  \mathbb { R } ^ { d _ { p } }$ for each path $p _ { i } \in \mathcal { P }$ , where $d _ { p }$ is the dimensionality of the path embedding vector.

## C. Module Background

Transformer Module. Given an input path token sequence represented by matrix $Z \in \mathbb { R } ^ { n \times d _ { n } }$ , where n is the length of the sequence, and $d _ { n }$ is the dimensionality of each embedding vector, the Transformer module starts by adding a position embedding to Z to preserve the order of tokens. It then transforms Z to three distinct matrices: the query matrix $Q = Z W ^ { Q } \in \mathbb { R } ^ { n \times d _ { n } }$ , the key matrix $K = Z W ^ { \mathbf { \bar { K } } } \in \mathbb { R } ^ { n \times d _ { n } }$ and the value matrix $V = Z \dot { W } ^ { V } \in \mathbb { R } ^ { n \times d _ { n } }$ , where $W ^ { Q } , W ^ { K }$ and $W ^ { V }$ are learnable weight matrices in $\mathbb { R } ^ { d _ { n } \times d _ { n } }$ . The selfattention mechanism computes the attention correlation matrix as

$$
{ \mathrm { A t t e n t i o n } } ( Q , K , V ) = { \mathrm { s o f t m a x } } \left( { \frac { Q K ^ { \top } } { \sqrt { d _ { n } } } } \right) V \in \mathbb { R } ^ { n \times d _ { n } }\tag{1}
$$

where each element $\alpha _ { j , k }$ in the softmax matrix reflects the attention score between j-th and k-th tokens in the matrix Z.

## D. Solution Overview

Figure 3 shows an overview of the proposed AutoPRL, which consists of three modules: (1) AutoDA, (2) CVDC Learning, and (3) ACT. The details of these modules are provided in Sections IV, V, and VI.

![](images/20e07dbb5609c959e1d1ab589706cc89f57104173c27fd1740247c9e8ac31274.jpg)  
Fig. 3: Solution Overview.

## IV. AUTODA

Figure 4 provides an overview of the AutoPRL framework, which consists of three modules: AutoDA for view generation, CVDC for contrastive representation learning, and ACT for adaptive optimization.

We first detail the AutoDA module. Given an input path and the corresponding road-network context, AutoDA first maps the path into topology-aware spatial embeddings. It then employs two Gaussian-distribution-based VNAEs to model sequential correlations among nodes and generate two semantically consistent feature-level path views. These generated views are subsequently used by the CVDC module for contrastive learning. The intuition is that path augmentation should be dynamic and data-driven, rather than relying on static heuristic transformations.

## A. Spatial Embedding

As shown in Figure 5, topology-constrained spatial embedding is fundamental to constructing effective initial path representation vectors. By incorporating the structural properties of the road network, this approach ensures that each node embedding is informed by its local topology, promoting semantic consistency among connected nodes and encouraging similar embeddings for spatially and functionally related segments.

Motivated by recent advances in graph embedding learning [28], [29], we aim to learn high-quality node embedding vectors that effectively capture the structural properties of road networks. Given a road network represented as a graph $\mathcal { G } = ( \nu , \mathcal { E } )$ , where V denotes intersections and E denotes road segments, we adopt the Node2vec algorithm [28] to encode topological relationships among nodes. Node2vec simulates biased random walks to explore local and global neighborhood structures, and learns embedding vectors by preserving node co-occurrence patterns in a low-dimensional latent space, which can be defined as:

$$
\vec { Z } _ { \mathcal { G } } = \mathrm { N o d e 2 v e c } ( \mathcal { G } ( \mathcal { V } , \mathcal { E } ) )\tag{2}
$$

where $\vec { Z } _ { \mathcal { G } } ~ = ~ \{ \vec { z } _ { v } ~ | ~ v ~ \in ~ \mathcal { V } \}$ denotes the topology-aware node embedding matrix learned from the road network, and $\begin{array} { r l r } { \vec { z } _ { v } } & { { } \in } & { \mathbb { R } ^ { d _ { \mathcal { G } } } } \end{array}$ is the embedding of node v. Given a path $p = \langle v _ { 1 } , v _ { 2 } , v _ { 5 } , v _ { 4 } \rangle \in \mathcal { P }$ , the corresponding initial path node sequence embedding is defined as $\vec { Z } _ { p } = \{ \vec { z } _ { 1 } , \vec { z } _ { 2 } , \vec { z } _ { 5 } , \vec { z } _ { 4 } \}$ , where each ⃗z<sub>i</sub> corresponds to the embedding of node $v _ { i }$ and $\vec { Z } _ { p }$ denotes the set of path sequence embeddings.

## B. Automatic Data Augmentation

To facilitate efficient and automatic path representation learning, we introduce an automatic and data-driven framework for generating contrastive views. Unlike existing methods that rely on manually crafted rules or domain-specific heuristics, our approach learns feature-level augmentation patterns directly from topology-aware node embeddings.

At the core of our method is the AutoDA module, which employs a variational node autoencoder as a generative model for producing diverse and semantically meaningful feature-level path views based on learned Gaussian distributions. These views retain information encoded in the input embeddings while providing sufficient variability for contrastive learning.

![](images/b9a00c2fa0b6aca5743b80828831898352cecab9ff9f2fc124c4482a62991a5b.jpg)  
Fig. 4: Overall architecture of AutoPRL. The framework consists of three modules: AutoDA for automatic path view generation, CVDC for contrastive representation learning with cross-path and cross-node objectives, and ACT for adaptive multi-objective optimization.

a) Variational Node Encoder.: Inspired by AutoST [30], the AutoDA module adopts two distinct Variational Node AutoEncoders (VNAEs), i.e., variational LSTMs [31], to generate two augmented path views. The two VNAEs have the same architecture but use non-shared parameters, enabling them to capture different latent variations of the same input path.

Given the initial path embedding sequence $\vec { Z } _ { p _ { i } }$ , we feed it into two VNAE encoders to obtain two latent hidden representations:

$$
\begin{array} { r } { \vec { H } ^ { ( 1 ) } = V N A E _ { 1 } ( \vec { Z } _ { p _ { i } } , \theta _ { 1 } ) , } \\ { \vec { H } ^ { ( 2 ) } = V N A E _ { 2 } ( \vec { Z } _ { p _ { i } } , \theta _ { 2 } ) , } \end{array}\tag{3}
$$

where $\theta _ { 1 }$ and $\theta _ { 2 }$ denote the learnable parameters of the two VNAE encoders, and $\vec { H } ^ { ( 1 ) }$ and $\vec { H } ^ { ( 2 ) }$ denote the corresponding hidden node representations.

To model the latent node distribution of each view, we use an MLP to parameterize a Gaussian distribution from the hidden representation $\vec { H } ^ { ( k ) }$ , where $k \in \{ 1 , 2 \}$ . Specifically, the MLP predicts the mean vector $\mu _ { ( k ) }$ and standard-deviation vector $\sigma _ { ( k ) }$ , yielding the node-level approximate posterior $q _ { \phi } ( Z ^ { ( k ) } \ \bar { \ } ^ { ' } H ^ { ( k ) } ) \ = \ \mathcal { N } ( \mu _ { ( k ) } , \mathrm { d i a g } ( \sigma _ { ( k ) } ^ { 2 } ) )$ . We then sample node-level latent embeddings from this distribution to generate diverse yet semantically consistent augmented path views.

b) Rationale of VAE-based View Generation.: The VNAE module is not used merely as an encoding architecture, but as a probabilistic view generator for path augmentation. Compared with deterministic encoders, the VNAE learns a continuous latent distribution and generates augmented path views through latent sampling. This property is particularly suitable for self-supervised PRL, where the diversity and semantic consistency of contrastive views directly affect the quality of learned representations. We adopt Gaussian latent distributions because they provide a tractable and stable way to model the uncertainty of road-network paths. Paths traversed by moving objects may exhibit stochastic variations caused by route choices, traffic conditions, and local roadnetwork structures, while still preserving their underlying route semantics. The Gaussian latent space enables efficient reparameterized sampling and stable optimization, allowing the model to generate diverse yet semantically consistent path views.

c) Automatic View Generation: The decoders of the two VNAE modules produce two semantically diverse node-level augmentations, denoted as $p _ { i } ^ { ( 1 ) }$ and $p _ { i } ^ { ( 2 ) }$ . Specifically, $p _ { i } ^ { ( 1 ) }$ and $p _ { i } ^ { ( 2 ) }$ are continuous node-embedding sequences in the latent space, rather than discrete node sequences mapped back to the original road-network graph. Thus, AutoDA performs feature-level augmentation on topology-aware node embeddings instead of explicit structure-level augmentation. Here, “topology-aware” refers to the road-network structural information encoded in the input node embeddings and does not imply that AutoDA generates topologically valid discrete path perturbations.

![](images/7ef7ab505bcf8805a2ad88de5543dcf1166d05207abeb4e478b6c9ec3b1251a6.jpg)  
Fig. 5: Automatic path view generation in AutoDA. The input path is first mapped into topology-aware spatial embeddings. Two VNAEs then encode the path into hidden representations, from which MLPs parameterize Gaussian latent distributions. By sampling from these distributions, AutoDA generates two semantically consistent feature-level views, $p _ { i } ^ { ( 1 ) }$ and $p _ { i } ^ { ( 2 ) }$

In contrast to recent path contrastive learning approaches that rely on heuristic or stochastic augmentation techniques—such as path node masking or dropping [2], [10]—AutoDA introduces a principled, learnable augmentation mechanism by modeling sequential dependencies directly within the data. This adaptive and data-driven strategy generates augmented views that preserve the underlying semantic structure while introducing meaningful variation, thereby enhancing spatial path representation. As a result, AutoDA offers taskirrelevant, data-driven self-supervised signals that improve the effectiveness, expressiveness, and generalizability of the learned path representations.

## C. Objective Optimization for Augmentation

To effectively train the learnable VNAEs, we introduce two objectives that constrain the latent variable Z during the learning process, aiming to align the distribution of reconstructed representations with that of the original input embedding H. We optimize the variational lower bound $\mathcal { L } _ { L B }$ using gradient-based learning. Since direct sampling from the approximate posterior introduces stochasticity into the latent variable, we adopt the reparameterization trick to enable gradient propagation through the sampling process. Specifically, for a node-level latent variable with approximate posterior $q _ { \phi } ( Z | H ) = \mathcal { N } ( \mu , \mathrm { d i a g } ( \sigma ^ { 2 } ) )$ , we rewrite the sampling operation as

$$
Z = \mu + \sigma \odot \varepsilon , \qquad \varepsilon \sim { \mathcal { N } } ( 0 , I ) ,\tag{4}
$$

where $\mu$ and $\sigma$ denote the mean and standard deviation of the approximate posterior, respectively. This formulation separates the stochastic component ε from the learnable distribution parameters, thereby enabling end-to-end gradient-based optimization.

We apply a denoising autoencoding criterion to the VNAEs by introducing corrupted input with Gaussian noise, $\tilde { z } = z +$ $\varepsilon _ { \mathrm { n 0 i s e } } .$ , where $\bar { \varepsilon } _ { \mathrm { n o i s e } } \sim \mathcal { N } ( 0 , \bar { \sigma } _ { \mathrm { n o i s e } } ^ { 2 } I )$

$$
\mathcal { L } _ { L B } = \mathcal { L } _ { L B } ^ { ( 1 ) } + \mathcal { L } _ { L B } ^ { ( 2 ) }\tag{5}
$$

In particular:

$$
\begin{array} { r l r } & { } & { \mathcal { L } _ { L B } ^ { ( 1 ) } = - D _ { K L } \left( q _ { \phi } \left( Z ^ { ( 1 ) } \mid H ^ { ( 1 ) } \right) \parallel p _ { s } \left( Z ^ { ( 1 ) } \right) \right) } \\ & { } & { + \mathbb { E } _ { q _ { \phi } \left( Z ^ { ( 1 ) } \mid H ^ { ( 1 ) } \right) } \left[ \log p _ { \theta } \left( H ^ { ( 1 ) } \mid Z ^ { ( 1 ) } \right) \right] } \\ & { } & { \mathcal { L } _ { L B } ^ { ( 2 ) } = - D _ { K L } \left( q _ { \phi } \left( Z ^ { ( 2 ) } \mid H ^ { ( 2 ) } \right) \parallel p _ { s } \left( Z ^ { ( 2 ) } \right) \right) } \\ & { } & { + \mathbb { E } _ { q _ { \phi } \left( Z ^ { ( 2 ) } \mid H ^ { ( 2 ) } \right) } \left[ \log p _ { \theta } \left( H ^ { ( 2 ) } \mid Z ^ { ( 2 ) } \right) \right] } \end{array}\tag{6}
$$

where, for $k \in \{ 1 , 2 \}$ , the first term is the KL regularization term that aligns the approximate posterior $q _ { \phi } ( Z ^ { ( k ) } \mid H ^ { ( k ) } )$ with the progress-based prior $p _ { s } ( Z ^ { ( k ) } )$ , and the second term is the reconstruction likelihood that encourages the latent representation to preserve the semantic and structural information encoded in the input embeddings $H ^ { ( k ) }$

We then define the reconstruction objective as $\mathcal { L } _ { R e c } ( \mathcal { L } _ { R e c } ^ { ( 1 ) } , \mathcal { L } _ { R e c } ^ { ( 2 ) } )$ , which evaluates how well the latent variable $Z _ { ( \cdot ) }$ preserves the semantic and topology-aware information encoded in the input node embeddings.

$$
\mathcal { L } _ { R e c } = \mathcal { L } _ { R e c } ^ { ( 1 ) } + \mathcal { L } _ { R e c } ^ { ( 2 ) }\tag{7}
$$

where

$$
\begin{array} { r } { \mathcal { L } _ { R e c } ^ { ( 1 ) } = \displaystyle \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { M } ( H _ { i j } ^ { ( 1 ) } - H _ { i j \prime } ^ { ( 1 ) } ) ^ { 2 } } \\ { \mathcal { L } _ { R e c } ^ { ( 2 ) } = \displaystyle \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { M } ( H _ { i j } ^ { ( 2 ) } - H _ { i j \prime } ^ { ( 2 ) } ) ^ { 2 } } \end{array}\tag{8}
$$

where $H _ { i j } ^ { ( k ) }$ and $H _ { i j ^ { \prime } } ^ { ( k ) }$ represent the original and reconstructed node embeddings of the j-th node within path i, where $k \in$ {1, 2}.

![](images/ee5c0c352a547b07bb187c77c8f656523eb7c7b85aaf28fc0070455c7818810f.jpg)  
Fig. 6: Illustration of the progress-based prior. The approximate posterior is aligned with a dynamic Gaussian prior through KL regularization. During training, the mean of the prior gradually shifts from $\pmb { \mu } _ { s _ { 1 } }$ to $\mu _ { s _ { T } } .$ which improves the alignment between the prior and the evolving latent space.

a) Progress-based Prior.: Moreover, we introduce a progress-based prior distribution for the latent variables. Instead of using a fixed standard normal prior $\mathcal { N } ( \mathbf { 0 } , I )$ throughout training, we employ a dynamic Gaussian prior whose mean evolves with the training progress. Specifically, at training stage s, the prior is defined as

$$
p _ { s } ( Z ) = \mathcal { N } ( \pmb { \mu _ { s } } , I ) ,\tag{9}
$$

where $\pmb { \mu } _ { s }$ denotes the progress-dependent prior mean and I is the identity covariance matrix.

The progress-based prior serves as an adaptive regularization strategy for variational learning. A fixed standard normal prior imposes the same latent regularization throughout training, which may be restrictive while the latent representation space is still evolving. In contrast, we gradually adjust the prior mean from $\mu _ { s _ { 1 } }$ at the initial stage to $\pmb { \mu } _ { s _ { T } }$ at the final stage. This progressive adjustment allows the prior to better follow the evolving latent space and provides smoother KL regularization during optimization.

For the learned approximate posterior, we write

$$
\begin{array} { r } { q _ { \phi } ( Z \mid H ) = \mathcal { N } \left( \pmb { \mu } _ { Z } , \pmb { \Sigma } _ { Z } \right) , \qquad \pmb { \Sigma } _ { Z } = \mathrm { d i a g } \left( \pmb { \sigma } _ { Z } ^ { 2 } \right) , } \end{array}\tag{10}
$$

where $\pmb { \mu } _ { Z }$ and $\sigma _ { Z }$ denote the mean and standard-deviation vectors of the approximate posterior, respectively.

The KL regularization term can therefore be written as

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { K L } = D _ { K L } \left( q _ { \phi } ( Z \mid H ) \| p _ { s } ( Z ) \right) } \\ { \displaystyle = D _ { K L } \left( { \mathcal { N } } ( { \pmb \mu } _ { Z } , { \pmb \Sigma } _ { Z } ) \| { \mathcal { N } } ( { \pmb \mu } _ { s } , I ) \right) } \\ { \displaystyle = \frac { 1 } { 2 } \left[ \mathrm { t r } ( { \pmb \Sigma } _ { Z } ) + ( { \pmb \mu } _ { s } - { \pmb \mu } _ { Z } ) ^ { T } ( { \pmb \mu } _ { s } - { \pmb \mu } _ { Z } ) - D - \log \left| { \pmb \Sigma } _ { Z } \right| \right] , } \end{array}\tag{11}
$$

where D denotes the latent dimensionality and $\operatorname { t r } ( \cdot )$ denotes the matrix trace.

## V. CROSS-VIEW DECOUPLED CONTRASTIVE LEARNING

To ensure the learning of generic path representations that generalize across diverse downstream applications, we adopt decoupled contrastive learning to define the overall training objectives. As illustrated in Figure 4, given the two distinct path views generated by the AutoDA module, we first describe the path encoder and then present the formulation of the crosspath and cross-node decoupled contrastive losses. The key intuition is that contrastive learning should enable efficient training while maintaining strong performance on downstream tasks.

## A. Path Encoder

To further capture the semantic information of a given path, we employ a Transformer-based path encoder, as defined in Eq. 1. Formally, as shown in Figure 4, given two distinct node sequence embeddings for a path $p _ { i }$ , denoted as $p _ { i } ^ { ( 1 ) } = \langle \vec { z } _ { 1 } ^ { ( 1 ) } , \vec { z } _ { 2 } ^ { ( 1 ) } , \cdot \cdot \cdot , \vec { z } _ { | p _ { i } | } ^ { ( 1 ) } \rangle$ and $\overline { { p _ { i } ^ { ( 2 ) } } } = \langle { \bar { z } _ { 1 } ^ { ( 2 ) } , \bar { z } _ { 2 } ^ { ( 2 ) } , \cdot \cdot \cdot , \bar { z } _ { \vert p _ { i } \vert } ^ { ( 2 ) } } \rangle$ generated from the learnable view generator, we first enrich both sequences with positional encodings to preserve the order of nodes. These position-enhanced sequences are then fed into the Transformer encoder to produce high-level node-level embeddings, denoted as $\vec { p _ { i } ^ { ( 1 ) } } = \mathrm { \dot { T r a n s f o r m e r } } ( p _ { i } ^ { ( 1 ) } )$ and $\bar { p } _ { i } ^ { ( 2 ) } =$ Transformer $\cdot ( p _ { i } ^ { ( 2 ) } )$ . To obtain the final path-level embeddings, we apply an aggregation function over the transformed edge embeddings in each view, resulting in two path embeddings, $Z _ { i } ^ { p 1 } = \vec { H } _ { \vec { p _ { i } } } ^ { ( 1 ) } \in \mathbb { R } ^ { d _ { p } }$ and $\dot { Z } _ { i } ^ { p 2 } = \vec { H } _ { \vec { p _ { i } } } ^ { ( \overline { { 2 } } ) } \in \mathbb { R } ^ { d _ { p } }$

## B. Cross-Path Decoupled Contrastive Learning

Given a batch of N samples (e.g., paths), $\left\{ p _ { 1 } , p _ { 2 } , \cdots , p _ { N } \right\}$ let $\vec { H } _ { \vec { p _ { i } } } ^ { ( 1 ) }$ and $\vec { H } _ { \vec { m } _ { i } } ^ { ( 2 ) }$ denote two embeddings of the generated view $p _ { i } ^ { ( 1 ) }$ and $p _ { i } ^ { ( 2 ) }$ of the i-th path $p _ { i }$ . We define the complete set of generated views in the batch as $B ~ = ~ \{ p _ { i } ^ { ( k ) } | k ~ \in$ $\{ 1 , 2 \} , i \in [ 1 , N ] \}$ , resulting in a total of 2N views. Following the SimCLR framework [25], contrastive learning is formulated as a classification task for each generated view $p _ { i } ^ { ( k ) }$ , where the objective is to correctly identify its corresponding positive pair $\mathbf { \Delta } _ { p _ { i } ^ { ( u ) } } ^ { \mathbf { \check { \alpha } } ( u ) }$ (with $u \ne k )$ among all other views in B. All remaining $2 N - 2$ views are treated as negative samples. For each view $p _ { i } ^ { ( k ) }$ , we use a cross-entropy objective to compute the contrastive loss $\mathcal { L } _ { i } ^ { ( k ) }$ . The objective pulls the positive pair, i.e., two views generated from the same path, closer in the embedding space, while pushing negative pairs, i.e., views generated from different paths, apart. The overall SimCLR loss across the batch is $\begin{array} { r } { \mathcal { L } _ { S i m C L R } = \sum _ { k \in \{ 1 , 2 \} , i \in [ 1 , N ] } \mathcal { L } _ { i } ^ { ( k ) } } \end{array}$

$$
\mathcal { L } _ { i } ^ { ( k ) } = - \left( \left. \vec { H } _ { \vec { p } _ { i } } ^ { ( 1 ) } , \vec { H } _ { \vec { p } _ { i } } ^ { ( 2 ) } \right. / t \right) + \log \left( \exp \left( \left. \vec { H } _ { \vec { p } _ { i } } ^ { ( 1 ) } , \vec { H } _ { \vec { p } _ { i } } ^ { ( 2 ) } \right. / t \right) \right.
$$

$$
+ \sum _ { \stackrel { j = 1 } { j \neq i } } ^ { N } \sum _ { l = 1 } ^ { 2 } \exp \left( \left. \vec { H } _ { \vec { p _ { i } } } ^ { ( k ) } , \vec { H } _ { \vec { p _ { j } } } ^ { ( l ) } \right. / t \right) \nonumber\tag{9}
$$

where $\exp ( \cdot )$ denotes the exponential operation, $\langle \cdot , \cdot \rangle$ denotes the inner product, and t is the temperature parameter.

However, the self-supervised contrastive learning objective in Eq. 9 relies on a large batch size to provide a sufficiently diverse set of negative samples. When the batch size is small, the training process suffers from significant negative-positive coupling (NPC) fluctuations, which can degrade representation quality and training stability. Motivated by decoupled contrastive learning techniques [32], we argue that effective and efficient path embedding learning in real-world scenarios should support contrastive learning under small-batch settings. This requires a learning strategy that can mitigate NPC effects while preserving discriminative power in low-sample regimes.

In this paper, we adapt the decoupled contrastive learning principle to path-level representation learning and formulate a Cross-Path Decoupled Contrastive loss (CPDCL) to better capture global semantic information (i.e., path-level) while addressing the limitations of conventional contrastive learning in small-batch settings. Specifically, CPDCL removes the positive pair term—i.e., exp $\left( \left. \vec { H } _ { \vec { p _ { i } } } ^ { ( 1 ) } , \vec { H } _ { \vec { p _ { i } } } ^ { ( 2 ) } \right. / t \right)$ —from the denominator of the standard SimCLR loss $\left( \operatorname { E q . } ^ { \prime } 9 \right)$ , thereby decoupling the gradient flow between positive and negative pairs. As illustrated in Figure 4, for each path $p _ { i }$ in the batch, the overall CPDCL objective is $\begin{array} { r }  \mathcal { L } _ { C P D C L } = \sum _ { k \in \{ 1 , 2 \} , i \in [ [ 1 , N ] \} \mathcal { L } _ { C P D C L , i } ^ { k } , } \end{array}$ where $\mathcal { L } _ { C P D C L , i } ^ { k }$ is

$$
\begin{array} { l } { \mathcal { L } _ { C P D C L , i } ^ { ( k ) } = - \log \frac { { \exp \left( \left. \vec { H } _ { \vec { p _ { i } } } ^ { ( 1 ) } , \vec { H } _ { \vec { p _ { i } } } ^ { ( 2 ) } \right. / t \right) } } { \sum _ { j = 1 , j \neq i } ^ { N } \sum _ { l = 1 } ^ { 2 } \exp \left( \left. \vec { H } _ { \vec { p _ { i } } } ^ { ( k ) } , \vec { H } _ { \vec { p _ { j } } } ^ { ( l ) } \right. / t \right) } } \\ { = - \frac { \langle \vec { H } _ { \vec { p _ { i } } } ^ { ( 1 ) } , \vec { H } _ { \vec { p _ { i } } } ^ { ( 2 ) } \rangle } { t } + \log \underset { j = 1 , j \neq i } { \overset { N } { \sum } } \underset { l = 1 } { \overset { 2 } { \sum } } \exp \left( \frac { \left. \vec { H } _ { \vec { p _ { i } } } ^ { ( k ) } , \vec { H } _ { \vec { p _ { j } } } ^ { ( l ) } \right. } { t } \right) } \end{array}\tag{12}
$$

Furthermore, we extend the CPDCL loss to a weighted variant by introducing a weighting function over the positive pairs. This generalization allows the model to dynamically adjust the contribution of each positive pair during training. The overall weighted loss is defined as:

$$
\begin{array} { c } { { \mathcal { L } _ { C P D C L , i } ^ { ( k ) } = - w \left( \vec { H } _ { \vec { p _ { i } } } ^ { ( 1 ) } , \vec { H } _ { \vec { p _ { i } } } ^ { ( 2 ) } \right) \left( \left. \vec { H } _ { \vec { p _ { i } } } ^ { ( 1 ) } , \vec { H } _ { \vec { p _ { i } } } ^ { ( 2 ) } \right. / t \right) + } } \\ { { \log \sum _ { j = 1 , j \neq i } ^ { N } \displaystyle \sum _ { l = 1 } ^ { 2 } \exp \left( \left. \vec { H } _ { \vec { p _ { i } } } ^ { ( k ) } , \vec { H } _ { \vec { p _ { j } } } ^ { ( l ) } \right. / t \right) } } \end{array}\tag{13}
$$

where we intuitively define a weighting function $\begin{array} { r l r } { w ( \vec { H } _ { \vec { p } _ { i } } ^ { ( 1 ) } , \vec { H } _ { \vec { p } _ { i } } ^ { ( 2 ) } ) } & { { } = } & { \vec { 2 } \mathrm { ~ - ~ } \frac { \exp \left( \left. \vec { H } _ { \vec { p } _ { i } } ^ { ( 1 ) } , \vec { H } _ { \vec { p } _ { i } } ^ { ( 2 ) } \right. / \sigma \right) ^ { - } } { E _ { i } \left[ \exp \left( \left. \vec { H } _ { \vec { p } _ { i } } ^ { ( 1 ) } , \vec { H } _ { \vec { p } _ { i } } ^ { ( 2 ) } \right. / \sigma \right) \right] } } \end{array}$ for the positive pairs in $\mathcal { L } _ { C P D C L }$ based on a negative von Mises-Fisher distribution, where $\begin{array} { r l r } { E [ w ] } & { { } = } & { 1 } \end{array}$ . Intuitively, this weighting emphasizes harder positive pairs—those with lower similarity—by assigning them higher weights, thus providing stronger learning signals. Notably, the original CPDCL loss is a special case of the weighted CPDCL loss: as $\sigma  \infty$ the weight approaches 1, and $w \mathcal { L } _ { C P D C L } \to \mathcal { L } _ { C P D C L }$

The design of the weighting function is motivated by the imbalance between easy and hard positive pairs in contrastive learning. Positive pairs with high similarity already provide limited optimization signals, while harder positive pairs with lower similarity are more informative for improving representation alignment. The proposed weighting function assigns larger weights to harder positive pairs and smaller weights to easier ones, thereby encouraging the model to focus on under-aligned positive pairs during training. Since the expected weight satisfies $E [ w ] = 1$ , the weighted objective preserves the overall loss scale and avoids introducing an additional bias in expectation. Moreover, the original CPDCL is recovered when $\sigma \to \infty$ , indicating that the weighted variant is a generalized form of CPDCL. This design improves training stability while maintaining the decoupling advantage of CPDCL.

## C. Cross-Node Decoupled Contrastive Learning

In addition to decoupled contrastive learning over global path-level embedding learning, we further incorporate local contrastive signals by introducing a Cross-Node Decoupled Contrastive loss (CNDCL). This formulation captures finegrained differences between positive and negative edge-level (node embedding) samples, serving as a strong regularization that improves the embedding capacity of the path encoder. As illustrated in Figure 4, CNDCL complements the global cross-path loss by enforcing consistency at a more granular level, thereby enhancing the generalizability of the learned path embeddings.

Following the formulation of the CPDCL, the objective of the weighted CNDCL is to maximize the similarity between corresponding edge embeddings from the two augmented views $\vec { p } _ { i } ^ { ( 1 ) }$ and $\bar { p _ { i } ^ { ( 2 ) } }$ , while minimizing their similarity to negative edge samples. This encourages local structural alignment across views and improves the model’s ability to capture fine-grained variations. Formally, CNDCL is reformulated as $\begin{array} { r } { \mathcal { L } _ { C N D C L } = \sum _ { i = 1 } ^ { N } \sum _ { h = 1 } ^ { | p _ { i } | } \sum _ { k = 1 } ^ { 2 } \mathcal { L } _ { C N D C L , i , h } ^ { ( k ) } : } \end{array}$ , where $\mathcal { L } _ { C N D C L , i , h } ^ { ( k ) }$ denotes the decoupled contrastive loss for the $h -$ th edge (or node embedding) in the k-th view of path $p _ { i }$ and $\mathcal { L } _ { C N D C L , i , h } ^ { ( k ) }$ is

$$
\begin{array} { c } { \displaystyle \mathcal { L } _ { C N D C L , i , h } ^ { ( k ) } = - \sum _ { h = 1 } ^ { \lfloor p _ { i } \rfloor } w \Big ( \bar { p } _ { i , h } ^ { ( 1 ) } , \bar { p } _ { i , h } ^ { ( 2 ) } \Big ) \left( \langle \bar { p } _ { i , h } ^ { ( 1 ) } , \bar { p } _ { i , h } ^ { ( 2 ) } \rangle / t \right) } \\ { \displaystyle + \log \sum _ { j = 1 } ^ { N } \sum _ { h = 1 } ^ { \lfloor p _ { i } \rfloor } \sum _ { l = 1 } ^ { 2 } \exp \left( \langle \bar { p } _ { i , h } ^ { ( k ) } , \bar { p } _ { j , h } ^ { ( l ) } \rangle / t \right) } \end{array}\tag{11}
$$

## VI. ADAPTIVE CO-TRAINING

To efficiently train both the view generator and path encoder in an end-to-end manner, we jointly optimize four key loss components: the lower bound and reconstruction losses from the view generator $( \mathcal { L } _ { L B }$ and $\mathcal { L } _ { R e c } )$ , the global-level contrastive loss $( \mathcal { L } _ { C P D C L } )$ , and the local-level contrastive loss $( \mathcal { L } _ { C N D C L } )$ . A common approach is to combine these objectives through a weighted sum:

$$
\mathcal { L } _ { T o t a l } = \lambda _ { 1 } \mathcal { L } _ { L B } + \lambda _ { 2 } \mathcal { L } _ { R e c } + \lambda _ { 3 } \mathcal { L } _ { C P D C L } + \lambda _ { 4 } \mathcal { L } _ { C N D C L }\tag{14}
$$

where $\lambda _ { 1 } , \lambda _ { 2 } , \lambda _ { 3 } ,$ , and $\lambda _ { 4 }$ are manually selected hyperparameters. These weights are typically chosen based on empirical tuning or heuristic rules. However, this manual process is both time-consuming and sub-optimal, especially in complex multi-objective settings where the relative importance of each loss can vary dynamically over the course of training. The underlying intuition is that multiple training losses should be adaptively weighted in response to task uncertainty and training dynamics, thereby promoting balanced optimization and enhancing overall performance.

To address this limitation, we introduce an adaptive cotraining strategy that learns to weight each loss automatically based on task-specific uncertainty. Inspired by Ref. [33], we formulate the total loss as:

$$
\begin{array} { c } { { \displaystyle \mathcal { L } _ { T o t a l } = \frac { 1 } { 2 \sigma _ { 1 } ^ { 2 } } \mathcal { L } _ { L B } + \frac { 1 } { 2 \sigma _ { 2 } ^ { 2 } } \mathcal { L } _ { R e c } + \frac { 1 } { 2 \sigma _ { 3 } ^ { 2 } } \mathcal { L } _ { C P D C L } + } } \\ { { \displaystyle \frac { 1 } { 2 \sigma _ { 4 } ^ { 2 } } \mathcal { L } _ { C N D C L } + ( \log \sigma _ { 1 } \sigma _ { 2 } \sigma _ { 3 } \sigma _ { 4 } ) } } \end{array}\tag{15}
$$

where $\sigma _ { i }$ is a learnable uncertainty parameter for the $i -$ th loss component. A larger $\sigma _ { i }$ reduces the weight of the corresponding loss, while the logarithmic regularization term prevents $\sigma _ { i }$ from increasing without bound.

This uncertainty-aware formulation offers two key benefits: (i) dynamic adaptation: as training progresses and the difficulty of each objective changes, the model adjusts its focus accordingly—allocating more learning capacity to losses that offer more stable and informative gradients; (ii) improved stability and performance: by eliminating manual tuning and mitigating loss-scale imbalance, this strategy leads to more stable convergence and better generalization across both global and local levels of path embedding learning.

## VII. COMPLEXITY ANALYSIS

In this section, we provide a theoretical complexity analysis of AutoPRL. Let N denote the batch size, M the number of nodes in a path, $d _ { n }$ the node embedding dimension, $d _ { h }$ the VNAE hidden dimension, $d _ { z }$ the latent dimension, $d _ { p }$ the path embedding dimension, and K the number of Transformer layers. The complexity of AutoPRL mainly comes from the VNAE-based view generator, the Transformer-based path encoder, and the decoupled contrastive objectives.

For the VNAE-based view generator, the LSTM encoder and decoder process node-embedding sequences of length M. The encoder has time complexity $O ( N M ( d _ { n } d _ { h } + d _ { h } ^ { 2 } ) )$ , while the decoder has complexity $O ( N M ( d _ { z } d _ { h } + d _ { h } ^ { 2 } + d _ { h } d _ { n } ) )$ ). The linear layers for estimating the mean, variance, and latentto-hidden mapping add $O ( N d _ { h } d _ { z } )$ overhead, which is minor compared with the recurrent computation. Thus, the overall time complexity of the VNAE module is $O ( N M ( d _ { n } d _ { h } \ +$ $d _ { z } d _ { h } + d _ { h } ^ { 2 } + d _ { h } d _ { n } ) )$

For the Transformer-based path encoder, each view is prepended with a CLS token and processed by K Transformer layers. The complexity of one Transformer layer is dominated by self-attention and feed-forward networks, i.e., $O ( N M ^ { 2 } d _ { p } \dot { + } N M d _ { p } ^ { 2 } )$ . Since AutoPRL encodes two views, the total Transformer encoding complexity is $O ( 2 K N ( M ^ { 2 } d _ { p } +$ $M d _ { p } ^ { 2 } ) \rangle$ ), which dominates the runtime when M or $K$ is large.

For the decoupled contrastive objectives, the global pathlevel loss costs $O ( N ^ { 2 } d _ { p } )$ , while the local node-level loss costs $O ( M N ^ { 2 } d _ { p } )$ . Therefore, the total contrastive-learning complexity is $\dot { O } ( ( M + 1 ) N ^ { 2 } d _ { p } )$ . The ACT module only introduces three learnable scalar weights and thus has negligible runtime overhead.

In terms of parameter memory complexity, the VNAE module requires $O ( d _ { n } d _ { h } + d _ { z } d _ { h } + d _ { h } ^ { 2 } + d _ { h } d _ { z } )$ parameters, and the Transformer path encoder requires $O ( K d _ { p } ^ { 2 } )$ parameters. The contrastive objectives introduce no additional trainable parameters, while ACT adds only three scalars. Therefore, the total parameter memory complexity is $O ( d _ { n } d _ { h } + d _ { z } d _ { h } +$ $d _ { h } ^ { 2 } + d _ { h } d _ { z } + K d _ { p } ^ { 2 } )$ . Overall, the computational and memory costs of AutoPRL are dominated by the VNAE view generator and Transformer encoder, whereas the additional overhead of CVDC and ACT is limited.

## VIII. EXPERIMENTAL EVALUATION

To demonstrate the effectiveness of AutoPRL, we conduct experiments on three real-world trajectory datasets. Our focus is on five key downstream tasks: travel time estimation, path ranking, most similar path search, path recommendation, and path imputation. For each dataset, we compare AutoPRL’s performance with state-of-the-art baselines.

## A. Experimental Setup

1) Dataset: We evaluate our approach using three realworld datasets: Aalborg (Denmark) [15], Chengdu (China) [34] and Harbin (China) [35]. The Aalborg road network contains 10,017 nodes and 11,597 edges, the Chengdu road network comprises 6,632 nodes and 17,038 edges, and the Harbin road network contains 8,497 nodes and 14,497 edges. We also collect three large-scale GPS trajectory datasets corresponding to these networks. Specifically, we use 33,264 path trajectories in the Aalborg dataset, 54,491 paths in the Chengdu dataset, and 259,615 paths in the Harbin dataset after map matching [4], [36]–[38]. We also remove the paths with fewer than 20 nodes. The underlying road network graphs for three cities are obtained from OpenStreetMap<sup>1</sup>, ensuring real-world topological consistency and scalability for our experiments.

2) Downstream Tasks: Travel Time Estimation: For each path, we extract the corresponding travel time (in minutes) from the GPS trajectory data. The goal is to apply a regression model that predicts the travel time based on the learned path embedding vectors. To evaluate the performance of travel time estimation, we adopt three widely used metrics: Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and Root Mean Square Error (RMSE).

Path Ranking: Given a set of candidate paths—typically sharing the same source and destination—each path is assigned a ranking score within the range [0, 1], reflecting its relative preference based on historical trajectory data. Following prior studies [2], [15], we compute these scores using similarity measurements derived from real-world trajectories. To evaluate the performance of path ranking, we adopt three widely used metrics: Mean Absolute Error (MAE), the Kendall rank correlation coefficient (τ), and the Spearman rank correlation coefficient (ρ). These metrics collectively assess both the accuracy of the predicted ranking scores and the consistency of the predicted orderings relative to the ground truth.

Most Similar Path Search: Given a query path and a path database, the goal is to retrieve the path that is most similar to the query based on the learned path embeddings. We use Mean Rank as the evaluation metric, where a lower value indicates better retrieval performance.

Path Recommendation: Given a set of candidate paths, the goal is to identify the path preferred by users based on historical trajectory data. We formulate this task as a binary classification problem, where the actually traveled path is regarded as a positive sample and alternative paths are regarded as negative samples. We evaluate the performance using Accuracy, Precision, and F1 score.

Path Imputation: Given an incomplete path with a portion of its segments masked, the goal is to recover the missing path segments using the learned path representations. We randomly mask 30% of the path segments and use the original path as the ground truth. We adopt Recall@3 (R@3), Recall@5 (R@5), and Mean Average Precision (MAP) as the evaluation metrics.

3) Baselines: We evaluate AutoPRL with 13 baselines, consisting of 9 unsupervised learning-based methods and 4 supervised learning-based methods. The unsupervised learning based methods are: Node2vec [28], VGAE-AMI [39], DVGA [40], Toast [23], PIM [3], START [10], LightPath [2], TrajRL [24], and RED [1]. The supervised methods take into account the labels from a specific downstream task to obtain path representations, which are: PathRank [15], HierETA [41], CompactETA [42], and HMTRL [19].

For all unsupervised learning methods, we first train the path encoder using unlabeled trajectory data—specifically, 30K unlabeled samples from the Aalborg dataset, 50K from the Chengdu dataset and 200K from the Harbin dataset—to learn path embedding vectors. These embeddings are then used as input to a regression model trained on a smaller, labeled dataset (e.g., 12K labeled paths from Aalborg) to predict travel time and path ranking scores. For the regression model, we adopt the ensemble-based Gradient Boosting Regressor (GBR) [43] as the prediction model since our tasks are regression tasks. In contrast, for all supervised learning methods, the path encoder is trained directly using the limited labeled training data (e.g., the 12K labeled samples from the Aalborg dataset), without access to the larger unlabeled set.

4) Implementation Details: We implement AutoPRL using PyTorch 1.12.0. We use a machine running a Linux server with 40 Intel(R) Xeon(R) W-2155 CPU @ 3.30GHz and four Quadro RTX 8000 GPU cards. We set the embedding size $d _ { p }$ to 128 in all methods. We optimize the learnable view generator and path encoder using the AdamW optimizer, paired with a cosine decay learning rate schedule. The training batch size is set to 64. The base learning rate is set to 1e-5 for the Aalborg and Chengdu datasets, and 1e-6 for the Harbin dataset. The source code is available at https://github.com/Sean-Bin-Yang/ Path-Representation-Learning-Library/tree/master/AutoPRL.

## B. Main Results

The main experimental results are reported in Table I for travel time estimation, Table II for path ranking, Table III for most similar path search, Table IV for path recommendation, and Table V for path imputation.

Performance at Travel Time Estimation. Table I reports the overall performance of our AutoPRL and the baseline methods across the three real-world datasets, evaluated using various metrics. Specifically, we select 30K unlabeled paths from the Aalborg dataset, 50K from Chengdu (CD), and 200K from Harbin. However, only 12K labeled paths are available for each dataset. Thus, we use 30K, 50K, and 200K unlabeled paths from the Aalborg, Chengdu (CD), and Harbin datasets, respectively, to train the path encoder for unsupervised methods. In contrast, supervised approaches can only utilize the 12K labeled paths for each dataset. Overall, Table I shows that AutoPRL achieves the best observed performance across all metrics on the three real-world datasets. The improvement is most pronounced on Aalborg, followed by Harbin, while the gains on Chengdu are relatively modest. In particular, compared with the strongest baseline, the improvements in MAE and RMSE on Chengdu are 1.10% and 0.48%, respectively.

TABLE I: Overall performance on travel time estimation. The best results are highlighted in bold, and the second-best result are underlined.
<table><tr><td colspan="2" rowspan="2">Method</td><td colspan="3">Aalborg</td><td colspan="3">Chengdu</td><td colspan="3">Harbin</td></tr><tr><td>MAE↓</td><td>MAPE(%)↓</td><td>RMSE↓</td><td>MAE↓</td><td>MAPE(%)↓</td><td>RMSE↓</td><td>MAE↓</td><td>MAPE(%)↓</td><td>RMSE↓</td></tr><tr><td rowspan="4">Suppvrsed</td><td>PathRank</td><td>2.913</td><td>18.713</td><td>4.832</td><td>5.802</td><td>34.014</td><td>8.894</td><td>4.127</td><td>14.339</td><td>5.301</td></tr><tr><td>HierETA</td><td>3.097</td><td>18.620</td><td>5.058</td><td>5.060</td><td>28.820</td><td>7.921</td><td>4.897</td><td>16.506</td><td>5.961</td></tr><tr><td>CompactETA</td><td>3.030</td><td>19.726</td><td>4.791</td><td>4.832</td><td>28.678</td><td>7.447</td><td>4.372</td><td>14.974</td><td>5.362</td></tr><tr><td>HMTRL</td><td>3.515</td><td>19.955</td><td>5.792</td><td>4.861</td><td>28.529</td><td>7.473</td><td>4.704</td><td>15.543</td><td>5.803</td></tr><tr><td rowspan="11">Unsrsed</td><td>Node2vec</td><td>5.701</td><td>35.485</td><td>8.165</td><td>5.997</td><td>35.919</td><td>8.653</td><td>5.399</td><td>18.709</td><td>6.773</td></tr><tr><td>VGAE-AMI</td><td>4.986</td><td>32.421</td><td>7.458</td><td>5.454</td><td>34.192</td><td>7.860</td><td>5.193</td><td>17.824</td><td>6.359</td></tr><tr><td>DVGA</td><td>5.022</td><td>33.128</td><td>7.846</td><td>5.601</td><td>34.892</td><td>8.011</td><td>5.241</td><td>18.198</td><td>6.514</td></tr><tr><td>Toast</td><td>3.439</td><td>21.311</td><td>5.555</td><td>4.918</td><td>28.567</td><td>7.513</td><td>4.627</td><td>16.101</td><td>5.803</td></tr><tr><td>PIM</td><td>3.471</td><td>21.314</td><td>5.640</td><td>4.924</td><td>28.553</td><td>7.493</td><td>4.538</td><td>15.713</td><td>5.682</td></tr><tr><td>START</td><td>2.965</td><td>18.432</td><td>4.979</td><td>5.174</td><td>30.769</td><td>7.726</td><td>4.402</td><td>15.262</td><td>5.503</td></tr><tr><td>LightPath</td><td>2.920</td><td>18.254</td><td>4.932</td><td>4.850</td><td>28.387</td><td>7.406</td><td>4.221</td><td>14.962</td><td>5.310</td></tr><tr><td>RED</td><td>2.457</td><td>14.907</td><td>3.809</td><td>4.845</td><td>28.380</td><td>7.296</td><td>4.097</td><td>14.214</td><td>5.153</td></tr><tr><td>TrajRL</td><td>2.243</td><td>13.125</td><td>3.632</td><td>4.724</td><td>27.960</td><td>7.108</td><td>3.994</td><td>13.872</td><td>5.107</td></tr><tr><td>AutoPRL</td><td>1.971</td><td>12.867</td><td>3.382</td><td>4.672</td><td>27.318</td><td>7.074</td><td>3.883</td><td>13.421</td><td>4.896</td></tr><tr><td>Imp.</td><td>12.13%</td><td>1.97%</td><td>6.88%</td><td>1.10%</td><td>2.30%</td><td>0.48%</td><td>2.78%</td><td>3.25%</td><td>4.13%</td></tr></table>

TABLE II: Overall performance on path ranking. The best results are highlighted in bold, and the second-best results are underlined.
<table><tr><td colspan="2" rowspan="2">Method</td><td colspan="3">Aalborg</td><td colspan="3">Chengdu</td><td colspan="3">Harbin</td></tr><tr><td>MAE↓</td><td>TAU(τ)↑</td><td>RHO(ρ)↑</td><td>MAE↓</td><td>TAU(τ)↑</td><td>RHO(ρ)↑</td><td>MAE↓</td><td>TAU(τ)↑</td><td>RHO(ρ)↑</td></tr><tr><td rowspan="4">Supvrsed</td><td>PathRank</td><td>0.159</td><td>0.727</td><td>0.751</td><td>0.123</td><td>0.703</td><td>0.742</td><td>0.145</td><td>0.821</td><td>0.832</td></tr><tr><td>HierETA</td><td>0.152</td><td>0.757</td><td>0.767</td><td>0.178</td><td>0.735</td><td>0.774</td><td>0.133</td><td>0.846</td><td>0.879</td></tr><tr><td>CompactETA</td><td>0.169</td><td>0.698</td><td>0.709</td><td>0.195</td><td>0.661</td><td>0.705</td><td>0.160</td><td>0.796</td><td>0.833</td></tr><tr><td>HMTRL</td><td>0.163</td><td>0.729</td><td>0.741</td><td>0.149</td><td>0.779</td><td>0.789</td><td>0.133</td><td>0.838</td><td>0.874</td></tr><tr><td rowspan="11">Unurvised</td><td>Node2vec</td><td>0.253</td><td>0.632</td><td>0.637</td><td>0.246</td><td>0.464</td><td>0.505</td><td>0.211</td><td>0.468</td><td>0.512</td></tr><tr><td>VGAE-AMI</td><td>0.194</td><td>0.662</td><td>0.685</td><td>0.212</td><td>0.543</td><td>0.577</td><td>0.188</td><td>0.494</td><td>0.524</td></tr><tr><td>DVGA</td><td>0.211</td><td>0.653</td><td>0.669</td><td>0.220</td><td>0.522</td><td>0.549</td><td>0.195</td><td>0.475</td><td>0.501</td></tr><tr><td>Toast</td><td>0.138</td><td>0.768</td><td>0.779</td><td>0.131</td><td>0.734</td><td>0.752</td><td>0.161</td><td>0.840</td><td>0.878</td></tr><tr><td>PIM</td><td>0.141</td><td>0.764</td><td>0.777</td><td>0.125</td><td>0.725</td><td>0.745</td><td>0.119</td><td>0.879</td><td>0.911</td></tr><tr><td>START</td><td>0.178</td><td>0.839</td><td>0.859</td><td>0.228</td><td>0.638</td><td>0.679</td><td>0.151</td><td>0.859</td><td>0.890</td></tr><tr><td>LightPath</td><td>0.118</td><td>0.781</td><td>0.791</td><td>0.195</td><td>0.711</td><td>0.748</td><td>0.122</td><td>0.881</td><td>0.913</td></tr><tr><td>RED</td><td>0.109</td><td>0.819</td><td>0.823</td><td>0.129</td><td>0.797</td><td>0.833</td><td>0.127</td><td>0.858</td><td>0.865</td></tr><tr><td>TrajRL</td><td>0.089</td><td>0.855</td><td>0.871</td><td>0.114</td><td>0.810</td><td>0.839</td><td>0.116</td><td>0.896</td><td>0.929</td></tr><tr><td>AutoPRL</td><td>0.073</td><td>0.890</td><td>0.893</td><td>0.103</td><td>0.836</td><td>0.845</td><td>0.105</td><td>0.939</td><td>0.958</td></tr><tr><td>Imp.</td><td>17.98%</td><td>4.09%</td><td>2.53%</td><td>9.65%</td><td>3.21%</td><td>0.72%</td><td>9.48%</td><td>4.80%</td><td>3.12%</td></tr></table>

The relatively larger improvement on Aalborg may be related to its dataset characteristics. Specifically, Aalborg contains the largest road network among the three datasets, with 10,017 nodes, while only 30K unlabeled paths are used for representation learning. In comparison, Chengdu and Harbin use 50K and 200K unlabeled paths over road networks with 6,632 and 8,497 nodes, respectively. This suggests relatively sparser trajectory coverage with respect to the road-network size on Aalborg. Under such a setting, pre-defined node masking or dropping may provide less effective augmentation, whereas AutoDA learns feature-level variations directly from topology-aware node embeddings without explicitly removing path nodes. This difference may leave greater room for representation improvement on Aalborg. We regard this as a dataset-specific empirical observation rather than a universal advantage of AutoPRL.

Specifically, we observe the following: First, the graph representation learning-based approach Node2vec is less effective in capturing spatial correlations in paths. In contrast, AutoPRL effectively models spatial dependencies through its self-attention mechanism, resulting in better empirical performance. Second, AutoPRL achieves better empirical performance than generative self-supervised learning methods such as Toast and RED, which mainly consider local trajectory information during path encoder training. In comparison, AutoPRL integrates both global (see Section V-B) and local (see Section V-C) information. AutoPRL also achieves better overall empirical performance than contrastive learning-based methods such as PIM, START, and LightPath. In addition, all these unsupervised baselines optimize multiple losses using fixed weights, whereas we propose an adaptive cotraining module that dynamically optimizes multiple losses. Finally, PathRank, HierETA, CompactETA, and HMTRL are four supervised learning methods that exhibit relatively lower performance due to the limited availability of labeled training data, as labeling is both time-consuming and costly. This paper considers a scenario where labeled data is scarce.

TABLE III: Evaluation on Most Similar Path Search Task.
<table><tr><td rowspan=2 colspan=1>Method</td><td rowspan=1 colspan=3>Chengdu</td><td rowspan=1 colspan=3>Harbin</td></tr><tr><td rowspan=1 colspan=1>10K</td><td rowspan=1 colspan=1>30K</td><td rowspan=1 colspan=1>50K</td><td rowspan=1 colspan=1>60K</td><td rowspan=1 colspan=1>80K</td><td rowspan=1 colspan=1>100K</td></tr><tr><td rowspan=1 colspan=1>Toast</td><td rowspan=1 colspan=1>5.233</td><td rowspan=1 colspan=1>7.254</td><td rowspan=1 colspan=1>9.283</td><td rowspan=1 colspan=1>7.674</td><td rowspan=1 colspan=1>9.231</td><td rowspan=1 colspan=1>11.643</td></tr><tr><td rowspan=1 colspan=1>PIM</td><td rowspan=1 colspan=1>4.107</td><td rowspan=1 colspan=1>6.788</td><td rowspan=1 colspan=1>8.328</td><td rowspan=1 colspan=1>5.962</td><td rowspan=1 colspan=1>7.479</td><td rowspan=1 colspan=1>9.896</td></tr><tr><td rowspan=1 colspan=1>START</td><td rowspan=1 colspan=1>2.765</td><td rowspan=1 colspan=1>4.371</td><td rowspan=1 colspan=1>5.287</td><td rowspan=1 colspan=1>4.102</td><td rowspan=1 colspan=1>5.997</td><td rowspan=1 colspan=1>8.122</td></tr><tr><td rowspan=1 colspan=1>LightPath</td><td rowspan=1 colspan=1>2.431</td><td rowspan=1 colspan=1>3.289</td><td rowspan=1 colspan=1>5.364</td><td rowspan=1 colspan=1>3.562</td><td rowspan=1 colspan=1>5.296</td><td rowspan=1 colspan=1>7.432</td></tr><tr><td rowspan=1 colspan=1>RED</td><td rowspan=1 colspan=1>1.453</td><td rowspan=1 colspan=1>1.578</td><td rowspan=1 colspan=1>1.690</td><td rowspan=1 colspan=1>2.590</td><td rowspan=1 colspan=1>3.102</td><td rowspan=1 colspan=1>3.457</td></tr><tr><td rowspan=1 colspan=1>TrajRL</td><td rowspan=1 colspan=1>1.326</td><td rowspan=1 colspan=1>1.467</td><td rowspan=1 colspan=1>1.582</td><td rowspan=1 colspan=1>2.601</td><td rowspan=1 colspan=1>3.229</td><td rowspan=1 colspan=1>3.757</td></tr><tr><td rowspan=1 colspan=1>AutoPRL</td><td rowspan=1 colspan=1>1.217</td><td rowspan=1 colspan=1>1.307</td><td rowspan=1 colspan=1>1.389</td><td rowspan=1 colspan=1>2.571</td><td rowspan=1 colspan=1>2.991</td><td rowspan=1 colspan=1>3.212</td></tr></table>

Performance at Path Ranking. The results for path ranking are presented in Table II. AutoPRL achieves the best performance across all evaluation metrics, underscoring the effectiveness of its integrated advanced technologies. Specifically, AutoPRL incorporates three key components—AutoDA, crossview decoupled contrastive learning, and the Adaptive Co-Training module—which work in concert to enhance generalization of the learned path embeddings. In particular, the AutoDA module in AutoPRL automatically generates diverse path views based on a Gaussian distribution. This data-driven augmentation strategy enables the model to explore a richer variety of views during training, improving generality and reducing reliance on predefined augmentation schemes, which are often suboptimal and inflexible. Moreover, the crossview decoupled contrastive learning mechanism facilitates a more comprehensive understanding of path representations by decoupling information across views. This allows AutoPRL to effectively capture both global and local dependencies, akin to its performance gains in travel time estimation, while avoiding the limitations of traditional contrastive learning methods that rely solely on static augmentation strategies. Additionally, the adaptive co-training module dynamically adjusts the weighting of multiple loss components during training, effectively balancing them based on the learning dynamics.

Performance at Most Similar Path Search. The results for most similar path search are reported in Table III. AutoPRL achieves the lowest Mean Rank on both the Chengdu and Harbin datasets across all database sizes, showing consistent performance as the search space increases. These results indicate that the path representations learned by AutoPRL effectively preserve path-level semantic similarity. By capturing both global path semantics and local structural information, AutoPRL learns discriminative representations that are effective for similarity-based path retrieval.

TABLE IV: Evaluation on Path Recommendation Task.
<table><tr><td rowspan=2 colspan=2>Method</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>Aalborg</td><td rowspan=1 colspan=2>Chengdu</td></tr><tr><td rowspan=1 colspan=1>Acc.</td><td rowspan=1 colspan=1>Pre.</td><td rowspan=1 colspan=1>F1</td><td rowspan=1 colspan=1>Acc.</td><td rowspan=1 colspan=1>Pre.</td><td rowspan=1 colspan=1>F1</td></tr><tr><td rowspan=1 colspan=2>Toast</td><td rowspan=1 colspan=1>0.692</td><td rowspan=1 colspan=1>0.711</td><td rowspan=1 colspan=1>0.738</td><td rowspan=1 colspan=1>0.774</td><td rowspan=1 colspan=1>0.785</td><td rowspan=1 colspan=1>0.806</td></tr><tr><td rowspan=1 colspan=2>PIM</td><td rowspan=1 colspan=1>0.722</td><td rowspan=1 colspan=1>0.739</td><td rowspan=1 colspan=1>0.762</td><td rowspan=1 colspan=1>0.785</td><td rowspan=1 colspan=1>0.799</td><td rowspan=1 colspan=1>0.813</td></tr><tr><td rowspan=1 colspan=2>START</td><td rowspan=1 colspan=1>0.751</td><td rowspan=1 colspan=1>0.770</td><td rowspan=1 colspan=1>0.787</td><td rowspan=1 colspan=1>0.804</td><td rowspan=1 colspan=1>0.826</td><td rowspan=1 colspan=1>0.853</td></tr><tr><td rowspan=1 colspan=2>LightPath</td><td rowspan=1 colspan=1>0.752</td><td rowspan=1 colspan=1>0.768</td><td rowspan=1 colspan=1>0.790</td><td rowspan=1 colspan=1>0.817</td><td rowspan=1 colspan=1>0.832</td><td rowspan=1 colspan=1>0.849</td></tr><tr><td rowspan=1 colspan=2>RED</td><td rowspan=1 colspan=1>0.785</td><td rowspan=1 colspan=1>0.811</td><td rowspan=1 colspan=1>0.830</td><td rowspan=1 colspan=1>0.844</td><td rowspan=1 colspan=1>0.860</td><td rowspan=1 colspan=1>0.879</td></tr><tr><td rowspan=1 colspan=2>TrajRL</td><td rowspan=1 colspan=1>0.789</td><td rowspan=1 colspan=1>0.820</td><td rowspan=1 colspan=1>0.835</td><td rowspan=1 colspan=1>0.850</td><td rowspan=1 colspan=1>0.862</td><td rowspan=1 colspan=1>0.880</td></tr><tr><td rowspan=1 colspan=2>AutoPRL</td><td rowspan=1 colspan=1>0.791</td><td rowspan=1 colspan=1>0.823</td><td rowspan=1 colspan=1>0.841</td><td rowspan=1 colspan=1>0.858</td><td rowspan=1 colspan=1>0.869</td><td rowspan=1 colspan=1>0.881</td></tr></table>

Performance at Path Recommendation. The results for path recommendation are presented in Table IV. AutoPRL achieves the best observed performance across Accuracy, Precision, and F1 score on both the Aalborg and Chengdu datasets. These results indicate that the learned path representations can be effectively applied to the classification-based path recommendation task. By capturing informative path semantics, AutoPRL provides discriminative representations for distinguishing userpreferred paths from alternative candidates.

TABLE V: Evaluation on Path Imputation Task.
<table><tr><td rowspan=1 colspan=1></td><td rowspan=2 colspan=1>Method</td><td rowspan=1 colspan=3>Aalborg</td><td rowspan=1 colspan=2>Chengdu</td></tr><tr><td></td><td rowspan=1 colspan=1>R@3</td><td rowspan=1 colspan=1>R@5</td><td rowspan=1 colspan=1>MAP</td><td rowspan=1 colspan=1>R@3</td><td rowspan=1 colspan=1>R@5</td><td rowspan=1 colspan=1>MAP</td></tr><tr><td rowspan=1 colspan=1>Toast</td><td rowspan=1 colspan=1>0.656</td><td rowspan=1 colspan=1>0.705</td><td rowspan=1 colspan=1>0.697</td><td rowspan=1 colspan=1>0.770</td><td rowspan=1 colspan=1>0.797</td><td rowspan=1 colspan=1>0.800</td></tr><tr><td rowspan=1 colspan=1>PIM</td><td rowspan=1 colspan=1>0.679</td><td rowspan=1 colspan=1>0.716</td><td rowspan=1 colspan=1>0.663</td><td rowspan=1 colspan=1>0.792</td><td rowspan=1 colspan=1>0.837</td><td rowspan=1 colspan=1>0.845</td></tr><tr><td rowspan=1 colspan=1>START</td><td rowspan=1 colspan=1>0.708</td><td rowspan=1 colspan=1>0.725</td><td rowspan=1 colspan=1>0.692</td><td rowspan=1 colspan=1>0.834</td><td rowspan=1 colspan=1>0.860</td><td rowspan=1 colspan=1>0.863</td></tr><tr><td rowspan=1 colspan=1>LightPath</td><td rowspan=1 colspan=1>0.714</td><td rowspan=1 colspan=1>0.756</td><td rowspan=1 colspan=1>0.736</td><td rowspan=1 colspan=1>0.830</td><td rowspan=1 colspan=1>0.855</td><td rowspan=1 colspan=1>0.823</td></tr><tr><td rowspan=1 colspan=1>RED</td><td rowspan=1 colspan=1>0.733</td><td rowspan=1 colspan=1>0.778</td><td rowspan=1 colspan=1>0.746</td><td rowspan=1 colspan=1>0.862</td><td rowspan=1 colspan=1>0.881</td><td rowspan=1 colspan=1>0.863</td></tr><tr><td rowspan=1 colspan=1>TrajRL</td><td rowspan=1 colspan=1>0.742</td><td rowspan=1 colspan=1>0.781</td><td rowspan=1 colspan=1>0.775</td><td rowspan=1 colspan=1>0.876</td><td rowspan=1 colspan=1>0.899</td><td rowspan=1 colspan=1>0.881</td></tr><tr><td rowspan=1 colspan=1>AutoPRL</td><td rowspan=1 colspan=1>0.757</td><td rowspan=1 colspan=1>0.802</td><td rowspan=1 colspan=1>0.796</td><td rowspan=1 colspan=1>0.883</td><td rowspan=1 colspan=1>0.924</td><td rowspan=1 colspan=1>0.907</td></tr></table>

Performance at Path Imputation. The results for path imputation are reported in Table V. AutoPRL achieves the best observed performance across Recall@3, Recall@5, and MAP on both the Aalborg and Chengdu datasets. These results indicate that the learned path representations effectively preserve local structural and contextual information that is useful for recovering missing path segments. In particular, the node- and path-level contrastive objectives enable AutoPRL to capture both fine-grained local dependencies and global path semantics, resulting in effective representations for path imputation.

## C. Effect of Pre-training

In this experiment, we investigate the effect of pretraining. We employ AutoPRL as a pre-training method for the Transformer-based PathRank. We ensure that this PathRank framework adopts the same settings as AutoPRL for a fair comparison but the loss terms are changed to mean squared error (MSE) for both tasks. Note that AutoPRL is not tied to PathRank; the pre-trained path encoder can also initialize other supervised models for task-specific fine-tuning. In particular, we first train AutoPRL in an unsupervised manner, and then use the learned path encoder to initialize the Transformerbased PathRank. Specifically, PathRank takes path sequences as input and outputs both travel time and the corresponding path ranking score. Figure 7 reports the performance of PathRank with and without pre-training across the two downstream tasks on the three datasets. When employing both non-pre-trained and pre-trained PathRank with labeled data volumes of 2K, 4K, 6K, 8K, 10K, and 12K, we observe that (1) with pre-training, PathRank consistently outperforms the non-pre-trained version at every labeled data scale. Moreover, PathRank with pre-training achieves the same performance as non-pre-trained PathRank, but with significantly less labeled data. For example, to match the performance of non-pretrained PathRank trained on 12K labeled samples for path ranking, pre-trained PathRank only requires 5K labeled paths for the Aalborg dataset, slightly over 4K for Chengdu, and 8K for Harbin. (2) PathRank with pre-training achieves better performance than the non-pre-trained version, demonstrating its potential as a pre-training method to enhance supervised approaches.

![](images/9cf9332e70fa28a4d2d77cde5b234a1add306c5703d76e1875a470abb7fbb97e.jpg)  
(a) Travel Time Estimation

![](images/07cab35dc6732bcaa0eb7b8e62f1b4771af814c5c838403fc88819ae7f4c8433.jpg)  
(b) Path Ranking

![](images/384ecec5bf555a85be1ea1c1436ca36a52baee7ad694722d7935ab3d9bf46deb.jpg)  
(c) Travel Time Estimation

![](images/5011df3dea1ba0c0abdca4b3f0447f03140cb398368aa85ab11fff734a4f4df8.jpg)

![](images/59b5036b97f107db91b6909ed1e035bc94189adbfdcdf9c098e4708a6989a9a5.jpg)  
(a) Aalborg (TTE)  
(b) Aalborg (PR)

Fig. 7: Effect of Pre-training.  
(d) Path Ranking  
![](images/b37336ae84649d961b14a48655697c8fa157ca7aeee4d831044b76cbc85db1b4.jpg)

(f) Path Ranking  
![](images/722496a90e4ea787c1242149a3aef07d944a0d0a6aaebab498b5e16d5b31754c.jpg)

## D. Ablation Study

AutoPRL Performance with/without AutoDA Module. To evaluate the impact of the automatic view generator, we apply the same augmentation strategies as in START to generate path views, resulting in the w/o AutoDA model. The results across both tasks and three datasets (Figure 8) highlight the effectiveness of the AutoDA module. For instance, in Figure 8(a), AutoPRL improves the MAE from 2.865 to 1.971 for the travel time estimation task on the Aalborg dataset. This substantial reduction in MAE demonstrates the effectiveness of the learned feature-level augmentation compared with predefined augmentation strategies. Furthermore, similar performance gains are consistently observed on the Chengdu and Harbin datasets, reinforcing the general applicability of the AutoDA module in diverse real-world scenarios.

(e) Travel Time Estimation  
Fig. 8: Ablation Study.  
![](images/9f0cb17d67e8c14ba8003563fbd64c49d0b66fa1123078e09dd76de8f96b39a9.jpg)

![](images/d6d7d3f2d35e240cde7ac323efc794a017c038c15c5edb753cc9a0c76a0767a1.jpg)

![](images/c857fcdac7dbe10d18ce2030c633daafd73319ac208918790c3fad08f0aed7dc.jpg)  
(d) Chengdu (PR)

![](images/8de0c7331d6e26c755511830384ce5109ca087f735a0cb49d385974c85173bfb.jpg)  
(e) Harbin (TTE)

![](images/399381d24e9e3ceac4a80c90af479bef695364a756e68ff71bc454c865fdc89a.jpg)  
(f) Harbin (PR)

AutoPRL Performance with/without Progress-Based Distribution (PPD) for AutoDA Module. To assess the impact of the progress-based distribution strategy in training the AutoDA module, we use a general fixed prior distribution, resulting in the w/o PPD variant. Figure 8 shows that the progressbased distribution improves the ranking score (τ) from 0.798 to 0.836 on the Chengdu dataset for the path ranking task. This improvement highlights the benefits of adaptively adjusting the distribution of augmented views during training to better match the model’s learning progression. Additionally, similar performance enhancements are observed on the Aalborg and Harbin datasets, further demonstrating that the progress-based strategy consistently improves representation quality and generalization across diverse datasets.

AutoPRL Performance with/without Cross-View Decoupled Contrastive Loss. To assess the impact of cross-path decoupled contrastive loss (CPDCL) and cross-node decoupled contrastive loss (CNDCL), we conduct experiments considering each loss independently. (1) With only CPDCL (i.e., w/o CNDCL), the MAE for AutoPRL increases from 3.883 to 4.326 for travel time estimation on the Harbin dataset (Figure 8). (2) Conversely, with only CNDCL (i.e., w/o CPDCL), the MAE increases from 3.883 to 4.460. This indicates that CPDCL has a larger impact than CNDCL, though both contribute to improved AutoPRL performance. These findings suggest that modeling cross-path dependencies captures broader structural relationships within paths, which are essential for accurate travel time estimation. Furthermore, incorporating both CPDCL and CNDCL ensures a comprehensive understanding of path and node-level information, further boosting the model’s effectiveness.

AutoPRL Performance with/without Multi-loss Optimization. To evaluate the effect of the adaptive loss co-training (ACT) strategy, we train both the view generator and path encoder with fixed loss weights, similar to existing methods that simply sum the losses without adaptive optimization (denoted as w/o ACT). Figure 8 shows the results across both tasks and three datasets, demonstrating that AutoPRL with ACT achieves superior performance compared to AutoPRL without ACT, confirming the effectiveness of this strategy. This improvement stems from the ACT module’s ability to dynamically balance the learning of different components, effectively mitigating the dominance of any single loss during training. Additionally, ACT enables better convergence and generalization across datasets with varying characteristics, further enhancing the generality of AutoPRL.

Epoch  
Epoch  
![](images/0419a03fb9da96d74e4198583b7cf88b6885bd7b83a961b372e748375a3a357f.jpg)  
Temperature  
(a) TTE (Aalborg)

![](images/0bf2b84240d650030e7ae03673934ece4822eba86d14996535d57db923dd4fff.jpg)  
Temperature  
(b) Path Ranking (Aalborg)

![](images/788a8a6c345da76a2a28375803d2b5a5186d16c66cf50846119a36f9538e787e.jpg)  
(c) TTE (CD)

![](images/66e5d579d0cc4aaf4acac5fcaa6c0cbe627b4c4f05b92e6ab589ac5a67d5e1fc.jpg)  
Temperature  
(d) Path Ranking (CD)

![](images/ab0dc2a0d34f3f095dda3b23d677a7e10327ad597af6e9b58392470118b2c602.jpg)  
Temperature  
(e) TTE (Harbin)

![](images/48224d56cb028a0e749f7edf2badb11b48159eb04b6090207a9a44b712a8f627.jpg)  
Temperature  
(e) Path Ranking (Harbin)

Fig. 9: Effect of Temperature t  
![](images/18e6eebeef43ae064eda5caf322636bc06dfb486417b93a70f165bd68419ee18.jpg)  
(a) Travel Time Estimation

![](images/2822d31e0b72099406ff89fc8df893b985b817374e538d6df2f1f760a0705eba.jpg)  
(b) Path Ranking

![](images/0e870a121d441173a0fceaade38e32bf916958db4ab2e8df859172e89931c5ec.jpg)  
(c) Travel Time Estimation

![](images/d2079db42fd8cb82fb9e9f46609dc907ec156d8ba70853d54ad7b8ced3b0b0c7.jpg)  
(d) Path Ranking

![](images/56edf5e01458b851d03ba14e17e18d6cbbfcd6c85ce5a423c109a20e389ff6ed.jpg)  
(e) Travel Time Estimation

![](images/13b0c049f924bf36687b8bedf56ad2b4073e146fd98baf91c2c6e752aa569df8.jpg)  
(f) Path Ranking  
Fig. 10: Scalability Performance vs. Cardinality

![](images/856984fd157e9d47009d9d84651da3cb418bae0787b712690a669a6672d8b70e.jpg)  
(a) Training Time

![](images/88d81deb02e165b2ab8cbe74aabbbc5d733d7644a581ec6829a9851e70858d0f.jpg)  
(b) Inference Time

![](images/7c97a3b406b47176844b9c26933dd96cc038af3ee9e165373a26140d953898f0.jpg)

![](images/d5e640837b7e32ef654bc6c791dd241577f808160eb0aa7b86f658c5e84ba6f5.jpg)  
(d) Training Convergence (CD)

![](images/b7a1576140cf9850085a3624a00a89ef16638a3469be685491c0e1ec7d34ef41.jpg)  
(e) Training Convergence (Harbin)  
Fig. 11: (a) Model training time (in minutes) per epoch; (b) model inference time (in ms) per path; and (c) – (e) training convergence of AutoPRL on three datasets.

AutoPRL Performance with Different Generative Modeling Paradigms. To verify the effectiveness of the proposed VAE-based view generator, we compare AutoPRL with a diffusion-based generative variant, denoted as AutoPRL-DGM. As shown in Table VI, the original VAE-based design achieves slightly better observed performance than AutoPRL-DGM on both travel-time estimation and path ranking across the Aalborg and Chengdu datasets. Although the absolute performance differences are relatively modest, these results suggest that the VNAE-based formulation provides a stable and effective choice for path view generation in our setting.

## E. Parameter Study

Effect of Temperature. Further, we evaluate the sensitivity of AutoPRL to assess its performance under varying temperature conditions in the decoupled contrastive loss. Specifically, we conduct a parameter study across three datasets, evaluating model performance for $t \in \{ 0 . 0 3 , 0 . 0 5 , 0 . 1 , 0 . 5 , 5 \}$ . The results, presented in Figure 9, indicate that the best performance is achieved at t = 0.05 on the Aalborg and Chengdu datasets, while t = 0.1 performs best on the Harbin dataset. These findings suggest that optimal temperature values vary across datasets, reflecting their differing complexity levels. We also observe that when t = 5, AutoPRL exhibits reduced performance, as the contrastive learning component applies less pressure to distinguish between positive and negative samples, leading to collapsed embeddings and diminished discriminative power. Overall, we conclude that the temperature parameter should be tuned separately for each dataset, as optimal values may vary depending on dataset complexity and characteristics.

![](images/60942d9961c63e5766945bc34690c2fa224f132f44ec4626bd927528105fbe29.jpg)  
Dimension

![](images/cd6f421fe9a6345024e8559f76f036c944ad0ba1322f0eb7a4687b487f9c5ac0.jpg)  
Dimension  
Fig. 12: Effect of Embedding Dimension.

Effect of Dimension. To examine the influence of the path embedding size $d _ { p }$ , we conduct a sensitivity analysis on the traveltime estimation task using the Aalborg and Harbin datasets. We compare AutoPRL with representative baselines, including RED, TrajRL, and LightPath, under $d _ { p } \in \{ 3 2 , 6 4 , 1 2 8 , 2 5 6 \}$ As shown in Figure 12, AutoPRL consistently achieves the lowest MAE across all embedding sizes on both datasets.

TABLE VI: Ablation study on travel-time estimation and path ranking under different generative modeling paradigms.
<table><tr><td rowspan=3 colspan=1>Method</td><td rowspan=1 colspan=6>Travel Time Estimation</td><td rowspan=1 colspan=6>Path Ranking</td></tr><tr><td rowspan=1 colspan=3>Aalborg</td><td rowspan=1 colspan=3>Chengdu</td><td rowspan=1 colspan=3>Aalborg</td><td rowspan=1 colspan=3>Chengdu</td></tr><tr><td rowspan=1 colspan=1>MAE</td><td rowspan=1 colspan=1>MAPE</td><td rowspan=1 colspan=1>RMSE</td><td rowspan=1 colspan=1>MAE</td><td rowspan=1 colspan=1>MAPE</td><td rowspan=1 colspan=1>RMSE</td><td rowspan=1 colspan=1>MAE</td><td rowspan=1 colspan=1>TAU(τ)</td><td rowspan=1 colspan=1>RHO(ρ)</td><td rowspan=1 colspan=1>MAE</td><td rowspan=1 colspan=1>TAU(τ)</td><td rowspan=1 colspan=1>RHO(ρ)</td></tr><tr><td rowspan=1 colspan=1>AutoPRL-DGM</td><td rowspan=1 colspan=1>1.976</td><td rowspan=1 colspan=1>12.901</td><td rowspan=1 colspan=1>3.396</td><td rowspan=1 colspan=1>4.683</td><td rowspan=1 colspan=1>27.447</td><td rowspan=1 colspan=1>7.091</td><td rowspan=1 colspan=1>0.079</td><td rowspan=1 colspan=1>0.870</td><td rowspan=1 colspan=1>0.878</td><td rowspan=1 colspan=1>0.112</td><td rowspan=1 colspan=1>0.821</td><td rowspan=1 colspan=1>0.830</td></tr><tr><td rowspan=1 colspan=1>AutoPRL-GMGP</td><td rowspan=1 colspan=1>1.971</td><td rowspan=1 colspan=1>12.867</td><td rowspan=1 colspan=1>3.382</td><td rowspan=1 colspan=1>4.672</td><td rowspan=1 colspan=1>27.318</td><td rowspan=1 colspan=1>7.074</td><td rowspan=1 colspan=1>0.073</td><td rowspan=1 colspan=1>0.890</td><td rowspan=1 colspan=1>0.893</td><td rowspan=1 colspan=1>0.103</td><td rowspan=1 colspan=1>0.836</td><td rowspan=1 colspan=1>0.845</td></tr></table>

In addition, $d _ { p } ~ = ~ 1 2 8$ generally yields the best or nearbest performance, while increasing the dimension to 256 does not further improve accuracy and introduces additional computational cost. Following RED and our sensitivity results, we set $d _ { p } = 1 2 8$ as the default embedding size for all methods to ensure a fair comparison and a good balance between effectiveness and efficiency.

## F. Efficiency Study

We then report the model evaluations in terms of scalability and efficiency.

Model Scalability. In this section, we explore the scalability (i.e., training data size) performance of AutoPRL in comparison with LightPath and RED, using the Aalborg, Chengdu (CD), and Harbin datasets. Figure 10 illustrates the experimental results, where 20%, 40%, 60%, 80%, and 100% represent the proportion of the training dataset used. The first observation is that performance improves with increasing data cardinality, as larger training datasets enhance model capabilities. Second, AutoPRL demonstrates consistent performance improvements over Toast, PIM, LightPath, START, RED and TrajRL. Moreover, the performance of AutoPRL in both travel time estimation and path ranking tasks across the Aalborg, Chengdu, and Harbin datasets shows a consistent improvement as cardinality increases. Overall, our AutoPRL model demonstrates strong potential for scalability, supporting large-scale applications across diverse datasets and tasks.

Runtime Efficiency. In this section, we investigate the runtime efficiency during both the training and inference phases. Figures 11(a) and 11(b) illustrate the corresponding results. Overall, AutoPRL achieves competitive runtime efficiency compared with the baselines. Specifically, AutoPRL is faster than RED in both training and inference phases across all datasets, with more evident improvements on Aalborg and Chengdu during training. Although lightweight methods such as LightPath can achieve lower runtime costs due to their simpler input processing strategies, AutoPRL maintains comparable efficiency while delivering strong overall performance. During inference, we measure the inference time for each path embedding. As observed, AutoPRL and LightPath require approximately the same amount of time per path embedding, but are slightly faster than RED.

Training Efficiency. To further evaluate the efficiency of AutoPRL, we examine its convergence behavior with and without the ACT strategy (refer to Section VIII-D) on the three datasets. Figures 11(c), 11(d), and 11(e) present the corresponding results. The number of epochs for both w/ ACT and w/o ACT is set to 600. We observe the following: (1) both models quickly reduce their loss during the initial epochs;

and (2) AutoPRL w/ ACT consistently maintains a lower loss than AutoPRL w/o ACT, indicating that AutoPRL achieves superior training efficiency, converging faster and more stably to better performance with reduced computational cost. This improvement can be attributed to the adaptive optimization of multiple losses under the ACT strategy, which facilitates effective coordination between the view generator and path encoder.

Batch-Size Sensitivity Analysis. To evaluate the sensitivity of CVDC to batch size, we compare AutoPRL-DCLW with a standard SimCLR-based variant, AutoPRL-SimCLR, under batch sizes of 16, 32, 64, and 128 on the travel-time estimation task using the Aalborg and Harbin datasets. As shown in Table VII, AutoPRL-DCLW consistently achieves lower MAE than AutoPRL-SimCLR across different batch sizes. Moreover, AutoPRL-DCLW converges within 400 epochs on Aalborg and 500 epochs on Harbin, whereas AutoPRL-SimCLR requires 500 and 700 epochs, respectively. These results demonstrate that CVDC is less sensitive to batch size and enables more stable and efficient training than the standard SimCLR objective.

## G. Robustness to Noisy Representations

To evaluate the robustness of the learned path representations, we perturb the learned path embedding $\mathbf { h } _ { p }$ by adding Gaussian noise:

$$
\tilde { \mathbf { h } } _ { p } = \mathbf { h } _ { p } + \boldsymbol { \epsilon } , \qquad \boldsymbol { \epsilon } \sim \mathcal { N } \left( \mathbf { 0 } , \sigma _ { \mathrm { n o i s e } } ^ { 2 } I \right) ,\tag{16}
$$

where $\sigma _ { \mathrm { n o i s e } } = 0 . 0 3$ controls the perturbation magnitude. The noise is independently sampled for each embedding dimension and is applied to the learned path representations during downstream evaluation, while the path encoder and downstream prediction settings remain unchanged. We use the same noise magnitude and evaluation protocol for all compared methods to ensure a controlled comparison.

As reported in Table VIII, AutoPRL exhibits only marginal performance degradation under noisy representations. For example, the MAE on Aalborg travel-time estimation increases slightly from 1.971 to 1.985, whereas most baseline methods experience more noticeable performance drops.

The results indicate that the path representations learned by AutoPRL are relatively robust to representation-level perturbations. Together with the results from the preceding experiments, this provides additional empirical evidence that AutoPRL can learn stable and informative path representations under moderate perturbations.

TABLE VII: Batch-size Sensitivity Analysis on Travel-Time Estimation Comparing AutoPRL-SimCLR and AutoPRL-DCLW.
<table><tr><td rowspan=3 colspan=1>TrainingEpochs</td><td rowspan=1 colspan=8>Aalborg</td><td rowspan=1 colspan=8>Harbin</td></tr><tr><td rowspan=1 colspan=4>AutoPRL-SimCLR</td><td rowspan=1 colspan=4>AutoPRL-DCLW</td><td rowspan=1 colspan=4>AutoPRL-SimCLR</td><td rowspan=1 colspan=4>AutoPRL-DCLW</td></tr><tr><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1>128</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1>128</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1>128</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1>128</td></tr><tr><td rowspan=1 colspan=1>400</td><td rowspan=1 colspan=1>2.844</td><td rowspan=1 colspan=1>2.657</td><td rowspan=1 colspan=1>2.539</td><td rowspan=1 colspan=1>2.340</td><td rowspan=1 colspan=1>2.112</td><td rowspan=1 colspan=1>2.044</td><td rowspan=1 colspan=1>1.971</td><td rowspan=1 colspan=1>1.954</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td></tr><tr><td rowspan=1 colspan=1>500</td><td rowspan=1 colspan=1>2.435</td><td rowspan=1 colspan=1>2.216</td><td rowspan=1 colspan=1>2.161</td><td rowspan=1 colspan=1>2.064</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>6.158</td><td rowspan=1 colspan=1>5.725</td><td rowspan=1 colspan=1>5.384</td><td rowspan=1 colspan=1>5.008</td><td rowspan=1 colspan=1>4.001</td><td rowspan=1 colspan=1>3.964</td><td rowspan=1 colspan=1>3.883</td><td rowspan=1 colspan=1>3.775</td></tr><tr><td rowspan=1 colspan=1>600</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>5.574</td><td rowspan=1 colspan=1>4.977</td><td rowspan=1 colspan=1>4.472</td><td rowspan=1 colspan=1>4.258</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td></tr><tr><td rowspan=1 colspan=1>700</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>4.476</td><td rowspan=1 colspan=1>4.433</td><td rowspan=1 colspan=1>4.212</td><td rowspan=1 colspan=1>3.979</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr></table>

TABLE VIII: Performance on Travel-Time Estimation and Path Ranking under Noisy Representations.

<table><tr><td rowspan=3 colspan=1>Method</td><td rowspan=1 colspan=6>Travel Time Estimation</td><td rowspan=1 colspan=6>Path Ranking</td></tr><tr><td rowspan=1 colspan=3>Aalborg</td><td rowspan=1 colspan=3>Chengdu</td><td rowspan=1 colspan=3>Aalborg</td><td rowspan=1 colspan=3>Chengdu</td></tr><tr><td rowspan=1 colspan=1>MAE</td><td rowspan=1 colspan=1>MAPE</td><td rowspan=1 colspan=1>RMSE</td><td rowspan=1 colspan=1>MAE</td><td rowspan=1 colspan=1>MAPE</td><td rowspan=1 colspan=1>RMSE</td><td rowspan=1 colspan=1>MAE</td><td rowspan=1 colspan=1>TAU(τ)</td><td rowspan=1 colspan=1>RHO(ρ)</td><td rowspan=1 colspan=1>MAE</td><td rowspan=1 colspan=1>TAU(τ)</td><td rowspan=1 colspan=1>RHO(ρ)</td></tr><tr><td rowspan=1 colspan=1>Toast</td><td rowspan=1 colspan=1>3.439</td><td rowspan=1 colspan=1>21.311</td><td rowspan=1 colspan=1>5.555</td><td rowspan=1 colspan=1>4.918</td><td rowspan=1 colspan=1>28.567</td><td rowspan=1 colspan=1>7.513</td><td rowspan=1 colspan=1>0.138</td><td rowspan=1 colspan=1>0.768</td><td rowspan=1 colspan=1>0.779</td><td rowspan=1 colspan=1>0.131</td><td rowspan=1 colspan=1>0.734</td><td rowspan=1 colspan=1>0.752</td></tr><tr><td rowspan=1 colspan=1>Toast w/ noise</td><td rowspan=1 colspan=1>4.426</td><td rowspan=1 colspan=1>21.906</td><td rowspan=1 colspan=1>6.066</td><td rowspan=1 colspan=1>5.622</td><td rowspan=1 colspan=1>29.134</td><td rowspan=1 colspan=1>8.233</td><td rowspan=1 colspan=1>0.207</td><td rowspan=1 colspan=1>0.672</td><td rowspan=1 colspan=1>0.683</td><td rowspan=1 colspan=1>0.206</td><td rowspan=1 colspan=1>0.692</td><td rowspan=1 colspan=1>0.703</td></tr><tr><td rowspan=1 colspan=1>PIM</td><td rowspan=1 colspan=1>3.471</td><td rowspan=1 colspan=1>21.314</td><td rowspan=1 colspan=1>5.640</td><td rowspan=1 colspan=1>4.924</td><td rowspan=1 colspan=1>28.553</td><td rowspan=1 colspan=1>7.493</td><td rowspan=1 colspan=1>0.141</td><td rowspan=1 colspan=1>0.764</td><td rowspan=1 colspan=1>0.777</td><td rowspan=1 colspan=1>0.125</td><td rowspan=1 colspan=1>0.725</td><td rowspan=1 colspan=1>0.745</td></tr><tr><td rowspan=1 colspan=1>PIM w/ noise</td><td rowspan=1 colspan=1>4.003</td><td rowspan=1 colspan=1>21.904</td><td rowspan=1 colspan=1>5.982</td><td rowspan=1 colspan=1>5.427</td><td rowspan=1 colspan=1>28.994</td><td rowspan=1 colspan=1>7.89</td><td rowspan=1 colspan=1>0.192</td><td rowspan=1 colspan=1>0.727</td><td rowspan=1 colspan=1>0.738</td><td rowspan=1 colspan=1>0.169</td><td rowspan=1 colspan=1>0.674</td><td rowspan=1 colspan=1>0.685</td></tr><tr><td rowspan=1 colspan=1>START</td><td rowspan=1 colspan=1>2.965</td><td rowspan=1 colspan=1>18.432</td><td rowspan=1 colspan=1>4.979</td><td rowspan=1 colspan=1>5.174</td><td rowspan=1 colspan=1>30.769</td><td rowspan=1 colspan=1>7.726</td><td rowspan=1 colspan=1>0.178</td><td rowspan=1 colspan=1>0.839</td><td rowspan=1 colspan=1>0.859</td><td rowspan=1 colspan=1>0.228</td><td rowspan=1 colspan=1>0.638</td><td rowspan=1 colspan=1>0.679</td></tr><tr><td rowspan=1 colspan=1>START w/ noise</td><td rowspan=1 colspan=1>3.441</td><td rowspan=1 colspan=1>18.901</td><td rowspan=1 colspan=1>5.551</td><td rowspan=1 colspan=1>5.607</td><td rowspan=1 colspan=1>31.114</td><td rowspan=1 colspan=1>8.011</td><td rowspan=1 colspan=1>0.225</td><td rowspan=1 colspan=1>0.795</td><td rowspan=1 colspan=1>0.803</td><td rowspan=1 colspan=1>0.265</td><td rowspan=1 colspan=1>0.603</td><td rowspan=1 colspan=1>0.617</td></tr><tr><td rowspan=1 colspan=1>LightPath</td><td rowspan=1 colspan=1>2.920</td><td rowspan=1 colspan=1>18.254</td><td rowspan=1 colspan=1>4.932</td><td rowspan=1 colspan=1>4.850</td><td rowspan=1 colspan=1>28.387</td><td rowspan=1 colspan=1>7.406</td><td rowspan=1 colspan=1>0.118</td><td rowspan=1 colspan=1>0.781</td><td rowspan=1 colspan=1>0.791</td><td rowspan=1 colspan=1>0.195</td><td rowspan=1 colspan=1>0.711</td><td rowspan=1 colspan=1>0.748</td></tr><tr><td rowspan=1 colspan=1>LightPath w/ noise</td><td rowspan=1 colspan=1>3.232</td><td rowspan=1 colspan=1>18.673</td><td rowspan=1 colspan=1>5.412</td><td rowspan=1 colspan=1>5.365</td><td rowspan=1 colspan=1>28.387</td><td rowspan=1 colspan=1>7.798</td><td rowspan=1 colspan=1>0.156</td><td rowspan=1 colspan=1>0.736</td><td rowspan=1 colspan=1>0.744</td><td rowspan=1 colspan=1>0.248</td><td rowspan=1 colspan=1>0.672</td><td rowspan=1 colspan=1>0.681</td></tr><tr><td rowspan=1 colspan=1>RED</td><td rowspan=1 colspan=1>2.457</td><td rowspan=1 colspan=1>14.907</td><td rowspan=1 colspan=1>3.809</td><td rowspan=1 colspan=1>4.845</td><td rowspan=1 colspan=1>28.380</td><td rowspan=1 colspan=1>7.296</td><td rowspan=1 colspan=1>0.109</td><td rowspan=1 colspan=1>0.819</td><td rowspan=1 colspan=1>0.823</td><td rowspan=1 colspan=1>0.129</td><td rowspan=1 colspan=1>0.797</td><td rowspan=1 colspan=1>0.833</td></tr><tr><td rowspan=1 colspan=1>RED w/ noise</td><td rowspan=1 colspan=1>2.527</td><td rowspan=1 colspan=1>14.999</td><td rowspan=1 colspan=1>3.903</td><td rowspan=1 colspan=1>4.987</td><td rowspan=1 colspan=1>28.510</td><td rowspan=1 colspan=1>7.374</td><td rowspan=1 colspan=1>0.127</td><td rowspan=1 colspan=1>0.790</td><td rowspan=1 colspan=1>0.805</td><td rowspan=1 colspan=1>0.144</td><td rowspan=1 colspan=1>0.762</td><td rowspan=1 colspan=1>0.779</td></tr><tr><td rowspan=1 colspan=1>TrajRL</td><td rowspan=1 colspan=1>2.243</td><td rowspan=1 colspan=1>13.125</td><td rowspan=1 colspan=1>3.632</td><td rowspan=1 colspan=1>4.724</td><td rowspan=1 colspan=1>27.960</td><td rowspan=1 colspan=1>7.108</td><td rowspan=1 colspan=1>0.089</td><td rowspan=1 colspan=1>0.855</td><td rowspan=1 colspan=1>0.871</td><td rowspan=1 colspan=1>0.114</td><td rowspan=1 colspan=1>0.810</td><td rowspan=1 colspan=1>0.839</td></tr><tr><td rowspan=1 colspan=1>TrajRL w/ noise</td><td rowspan=1 colspan=1>2.357</td><td rowspan=1 colspan=1>13.396</td><td rowspan=1 colspan=1>3.829</td><td rowspan=1 colspan=1>4.810</td><td rowspan=1 colspan=1>28.128</td><td rowspan=1 colspan=1>7.278</td><td rowspan=1 colspan=1>0.097</td><td rowspan=1 colspan=1>0.823</td><td rowspan=1 colspan=1>0.841</td><td rowspan=1 colspan=1>0.128</td><td rowspan=1 colspan=1>0.782</td><td rowspan=1 colspan=1>0.820</td></tr><tr><td rowspan=1 colspan=1>AutoPRL</td><td rowspan=1 colspan=1>1.971</td><td rowspan=1 colspan=1>12.867</td><td rowspan=1 colspan=1>3.382</td><td rowspan=1 colspan=1>4.672</td><td rowspan=1 colspan=1>27.318</td><td rowspan=1 colspan=1>7.074</td><td rowspan=1 colspan=1>0.073</td><td rowspan=1 colspan=1>0.890</td><td rowspan=1 colspan=1>0.893</td><td rowspan=1 colspan=1>0.103</td><td rowspan=1 colspan=1>0.836</td><td rowspan=1 colspan=1>0.845</td></tr><tr><td rowspan=1 colspan=1>AutoPRL w/ noise</td><td rowspan=1 colspan=1>1.985</td><td rowspan=1 colspan=1>12.875</td><td rowspan=1 colspan=1>3.392</td><td rowspan=1 colspan=1>4.690</td><td rowspan=1 colspan=1>27.338</td><td rowspan=1 colspan=1>7.094</td><td rowspan=1 colspan=1>0.078</td><td rowspan=1 colspan=1>0.881</td><td rowspan=1 colspan=1>0.890</td><td rowspan=1 colspan=1>0.106</td><td rowspan=1 colspan=1>0.833</td><td rowspan=1 colspan=1>0.840</td></tr></table>

Original Path  
Random Dropping View  
AutoDA View  
![](images/00543b1e3a5da651d88457306f1f14837f68adbfad9a9f6d01327eb0ea9eb674.jpg)  
Fig. 13: Case study of path view generation. We compare the original path, a random dropping view, and an AutoDA-generated view that is projected onto the road network only for visualization. Random dropping may remove important intermediate nodes and break local path continuity, while AutoDA better preserves the overall route semantics.

## H. Visualization of Generated Path Views

To provide more direct evidence for the semantic consistency of the generated views, we conduct a qualitative case study of path view generation. Figure 13 compares the original path, a random dropping view, and an AutoDA-generated view projected onto the road network only for visualization. Random dropping may remove important intermediate nodes and break local path continuity, leading to a view that deviates from the original route semantics. In contrast, the AutoDAgenerated view better preserves the overall route semantics while still introducing moderate variation.

It is worth noting that the projection in Figure 13 is used only for visualization. During training, AutoDA operates on continuous topology-aware node-embedding sequences in the latent space and does not require mapping the generated views back to discrete road-network nodes.

## IX. CONCLUSION

We propose AutoPRL, an effective and efficient framework for path representation learning that leverages decoupled contrastive learning. Unlike existing contrastive learning methods that rely on pre-defined augmentation strategies and often suffer from information loss, AutoPRL first introduces a learnable path view generator parameterized by Gaussian distributions to automatically create diverse, semantically meaningful views. To improve efficiency, we design cross-view decoupled contrastive strategies that disentangle positive and negative pair optimization at both node and path levels, and integrate an Adaptive Co-Training (ACT) module that dynamically assigns uncertainty-based weights to each objective. Extensive experiments across multiple downstream tasks on three realworld datasets demonstrate that AutoPRL achieves strong performance and consistent empirical improvements over the considered baselines.

## ACKNOWLEDGEMENTS

This work was partially supported by the National Natural Science Foundation of China (Nos. 62472174 and 62402082), the ECNU Multifunctional Platform for Innovation (No. 001), and the Chongqing Municipal Education Commission (No. KJQN202400637).

## REFERENCES

[1] S. Zhou, S. Shang, L. Chen, C. S. Jensen, and P. Kalnis, “RED: effective trajectory representation learning with comprehensive information,” PVLDB, vol. 18, no. 2, pp. 80–92, 2024.

[2] S. B. Yang, J. Hu, C. Guo, B. Yang, and C. S. Jensen, “Lightpath: Lightweight and scalable path representation learning,” in KDD, 2023, pp. 2999–3010.

[3] S. B. Yang, C. Guo, J. Hu, J. Tang, and B. Yang, “Unsupervised path representation learning with curriculum negative sampling,” in IJCAI, 2021, pp. 3286–3292.

[4] S. B. Yang, C. Guo, J. Hu, B. Yang, J. Tang, and C. S. Jensen, “Weaklysupervised temporal path representation learning with contrastive curriculum learning,” in ICDE, 2022, pp. 2873–2885.

[5] S. B. Yang, Y. Sun, J. Hu, Z. Xu, K. Torp, H. Lu, B. Yang, and C. S. Jensen, “REFINE: trajectory representation learning via closedloop transcription,” in KDD, 2026, pp. 6010–6020.

[6] S. B. Yang, Y. Sun, Y. Cheng, Y. Lin, T. Kristian, and J. Hu, “Spatiotemporal trajectory foundation model-recent advances and future directions,” arXiv preprint arXiv:2511.20729, 2025.

[7] X. Han, R. Cheng, C. Ma, and T. Grubenmann, “Deeptea: Effective and efficient online time-dependent trajectory outlier detection,” PVLDB, vol. 15, no. 7, pp. 1493–1505, 2022.

[8] Y. Chang, E. Tanin, G. Cong, C. S. Jensen, and J. Qi, “Trajectory similarity measurement: An efficiency perspective,” PVLDB, vol. 17, no. 9, pp. 2293–2306, 2024.

[9] L. Deng, Y. Zhao, J. Chen, S. Liu, Y. Xia, and K. Zheng, “Learning to hash for trajectory similarity computation and search,” in ICDE, 2024, pp. 4491–4503.

[10] J. Jiang, D. Pan, H. Ren, X. Jiang, C. Li, and J. Wang, “Self-supervised trajectory representation learning with temporal regularities and travel semantics,” in ICDE, 2023, pp. 843–855.

[11] C. S. Jensen, B. Yang, C. Guo, J. Hu, and K. Torp, “Routing with massive trajectory data,” in ICDE, 2024, pp. 5542–5547.

[12] C. Guo, R. Xu, B. Yang, Y. Ye, T. Kieu, Y. Zhao, and C. S. Jensen, “Efficient stochastic routing in path-centric uncertain road networks,” PVLDB, vol. 17, no. 11, pp. 2893–2905, 2024.

[13] J. Xu, Z. Bao, and H. Lu, “A framework to support continuous range queries over multi-attribute trajectories,” TKDE, vol. 35, no. 2, pp. 1119– 1133, 2023.

[14] S. B. Yang, H. Miao, Z. Xu, J. Hu, X. Wang, H. Lu, B. Yang, and C. S. Jensen, “Dgcpath: Distribution-aware generative contrastive framework for self-supervised path representation learning,” in IJCAI/ECAI, 2026, pp. 3169–3177.

[15] S. B. Yang, C. Guo, and B. Yang, “Context-aware path ranking in road networks,” TKDE, vol. 34, no. 7, pp. 3153–3168, 2022.

[16] Y. Wei, Y. Lin, H. Gao, R. Xu, S. B. Yang, and J. Hu, “Path-llm: A multi-modal path representation learning by aligning and fusing with large language models,” in WWW, 2025, pp. 2289–2298.

[17] R. Xu, H. Cheng, C. Guo, H. Gao, J. Hu, S. B. Yang, and B. Yang, “Mm-path: Multi-modal, multi-granularity path representation learning,” in KDD, 2025, pp. 1703–1714.

[18] S. B. Yang, Y. Sun, Z. Xu, T. Kieu, J. Hu, B. Yang, K. Torp, H. Lu, and T. B. Pedersen, “When correlations mislead: Confounder-aware multiview urban region representation learning,” in ICDE, 2027.

[19] H. Liu, J. Han, Y. Fu, Y. Li, K. Chen, and H. Xiong, “Unified route representation learning for multi-modal transportation recommendation with spatiotemporal pre-training,” VLDB J., vol. 32, no. 2, pp. 325–342, 2023.

[20] Z. Ma, Z. Tu, X. Chen, Y. Zhang, D. Xia, G. Zhou, Y. Chen, Y. Zheng, and J. Gong, “More than routing: Joint GPS and route modeling for refine trajectory representation learning,” in WWW, 2024, pp. 3064– 3075.

[21] H. Zhou, W. Huang, Y. Chen, T. He, G. Cong, and Y. S. Ong, “Road network representation learning with the third law of geography,” in NeurIPS, 2024.

[22] K. He, X. Chen, S. Xie, Y. Li, P. Dollar, and R. B. Girshick, “Masked´ autoencoders are scalable vision learners,” in CVPR, 2022, pp. 15 979– 15 988.

[23] Y. Chen, X. Li, G. Cong, Z. Bao, C. Long, Y. Liu, A. K. Chandran, and R. Ellison, “Robust road network representation learning: When traffic patterns meet traveling semantics,” in CIKM, 2021, pp. 211–220.

[24] H. Xia, X. Zhang, Y. Cao, L. Cao, Y. Yu, and J. Dong, “Selfsupervised trajectory representation learning with multi-scale spatiotemporal feature exploration,” in ICDE, 2025, pp. 779–792.

[25] X. Chen and K. He, “Exploring simple siamese representation learning,” in CVPR, 2021, pp. 15 750–15 758.

[26] J. Grill, F. Strub, F. Altche, C. Tallec, P. H. Richemond, E. Buchatskaya,´ C. Doersch, B. A. Pires, Z. Guo, M. G. Azar, B. Piot, K. Kavukcuoglu,<sup>´</sup> R. Munos, and M. Valko, “Bootstrap your own latent - A new approach to self-supervised learning,” in NeurIPS , 2020.

[27] K. He, H. Fan, Y. Wu, S. Xie, and R. B. Girshick, “Momentum contrast for unsupervised visual representation learning,” in CVPR, 2020, pp. 9726–9735.

[28] A. Grover and J. Leskovec, “node2vec: Scalable feature learning for networks,” in KDD, 2016, pp. 855–864.

[29] Y. Ye, X. Lian, and M. Chen, “Efficient exact subgraph matching via gnn-based path dominance embedding,” PVLDB, vol. 17, no. 7, pp. 1628–1641, 2024.

[30] Q. Zhang, C. Huang, L. Xia, Z. Wang, Z. Li, and S. Yiu, “Automated spatio-temporal graph contrastive learning,” in WWW, 2023, pp. 295– 305.

[31] S. Liang, Z. Pan, W. Liu, J. Yin, and M. de Rijke, “A survey on variational autoencoders in recommender systems,” ACM Comput. Surv., vol. 56, no. 10, p. 268, 2024.

[32] C. Yeh, C. Hong, Y. Hsu, T. Liu, Y. Chen, and Y. LeCun, “Decoupled contrastive learning,” in ECCV, vol. 13686, 2022, pp. 668–684.

[33] A. Kendall, Y. Gal, and R. Cipolla, “Multi-task learning using uncertainty to weigh losses for scene geometry and semantics,” in CVPR, 2018, pp. 7482–7491.

[34] H. Yuan, G. Li, and Z. Bao, “Route travel time estimation on A road network revisited: Heterogeneity, proximity, periodicity and dynamicity,” PVLDB, vol. 16, no. 3, pp. 393–405, 2022.

[35] X. Li, G. Cong, A. Sun, and Y. Cheng, “Learning travel time distributions with deep generative model,” in WWW, 2019, pp. 1017–1027.

[36] C. Yang and G. Gidofalvi, “Fast map matching, an algorithm integrating´ hidden markov model with precomputation,” Int. J. Geogr. Inf. Sci., vol. 32, no. 3, pp. 547–570, 2018.

[37] X. Li, K. Zhao, G. Cong, C. S. Jensen, and W. Wei, “Deep representation learning for trajectory similarity computation,” in ICDE, 2018, pp. 617– 628.

[38] C. Han, S. B. Yang, and J. Hu, “Diffmm: Efficient method for accurate noisy and sparse trajectory map matching via one step diffusion,” in AAAI, 2026, pp. 14 783–14 791.

[39] D. Li, D. Li, and G. Lian, “Variational graph autoencoder with adversarial mutual information learning for network representation learning,” ACM Trans. Knowl. Discov. Data, vol. 17, no. 3, pp. 45:1–45:18, 2023.

[40] D. Fan and C. Gao, “Learning network representations with disentangled graph auto-encoder,” CoRR, vol. abs/2402.01143, 2024.

[41] Z. Chen, X. Xiao, Y. Gong, J. Fang, N. Ma, H. Chai, and Z. Cao, “Interpreting trajectories from multiple views: A hierarchical self-attention network for estimating the time of arrival,” in KDD, 2022, pp. 2771– 2779.

[42] K. Fu, F. Meng, J. Ye, and Z. Wang, “Compacteta: A fast inference system for travel time prediction,” in KDD, 2020, pp. 3337–3345.

[43] S. Peter, F. Diego, F. A. Hamprecht, and B. Nadler, “Cost efficient gradient boosting,” in NeurIPS, 2017, pp. 1551–1561.