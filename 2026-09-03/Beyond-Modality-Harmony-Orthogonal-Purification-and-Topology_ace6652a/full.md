# Beyond Modality Harmony: Orthogonal Purification and Topology-Guided MoE for Conflict-Aware Multimodal Recommendation

Jialin Liu camilla.liu@my.cityu.edu.hk City University of Hong Kong Hong Kong, Hong Kong

Zhaorui Zhang<sup>∗</sup>   
zhaorui.zhang@polyu.edu.hk   
The Hong Kong Polytechnic   
University   
Hong Kong, Hong Kong

Ray C. C. Cheung r.cheung@cityu.edu.hk City University of Hong Kong Hong Kong, Hong Kong

## Abstract

Multimodal Recommender Systems (MRSs) typically rely on a flawed “modality harmony” assumption, presuming that multimodal features are inherently beneficial and strictly aligned with users’ collaborative interaction patterns. However, modality-topology conflicts are ubiquitous in real-world scenarios due to deceptive visual clickbaits and mismatched semantics. Blindly integrating these noisy modalities inevitably pollutes the pristine collaborative space, causing severe representation distortion. To address this, we propose Orthogonal purification and topology-guided MoE for conflict-aware multimodal Recommendation (OrthoRec). At its core, OrthoRec introduces Collaborative-Guided Orthogonal Purification (CGOP), which geometrically decouples multimodal features into directions parallel and orthogonal to a pure collaborative anchor. By adaptively truncating the orthogonal noise with an energy-preserving normalization, CGOP rectifies deceptive semantic directions while preserving the modality’s intrinsic representation capacity. Furthermore, we design a Topology-Aware Routing Mixture-of-Experts (TAR-MoE). Guided by the collaborative topology, TAR-MoE employs decoupled sigmoid gating to break the zero-sum bottleneck of traditional softmax attention, autonomously determining the injection scale for each purified modality. Finally, a safe-SSL objective is introduced to dynamically penalize the forced contrastive alignment of contradictory pairs. Experiments on three real-world Amazon datasets show that OrthoRec consistently outperforms competitive recent baselines and exhibits improved robustness under modality noise and item sparsity. Our code is available at https://github.com/Camilla-jl/Orthorec.

## CCS Concepts

• Information systems → Recommender systems.

## Keywords

Multimodal Recommendation, Graph Neural Networks, Orthogonal Purification, Mixture-of-Experts, Contrastive Learning

![](images/e768b40a17ad08f0031e89cf5c1b8323a3fdfa789f0d0ceaa4c5be95c27d98a6.jpg)  
Figure 1: Illustration of bottlenecks in current multimodal recommendation scenarios. (a) Impact of modality noise on user intent: blindly absorbing deceptive visual clickbaits distorts the true collaborative preference, leading to inaccurate recommendations. (b) Limitation of traditional modality fusion: softmax-based attention forces a zero-sum competition, heavily suppressing crucial textual cues when visual features are prominent.

## ACM Reference Format:

Jialin Liu, Zhaorui Zhang, and Ray C. C. Cheung. 2026. Beyond Modality Harmony: Orthogonal Purification and Topology-Guided MoE for Conflict-Aware Multimodal Recommendation. In Proceedings ofthe 34th ACM International Conference on Multimedia (MM ’26), November 10–14, 2026, Rio de Janeiro, Brazil. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/ 3767308.3836499

## 1 Introduction

Multimodal recommender systems (MRSs) have emerged as the foundational infrastructure for personalized information retrieval [6, 16, 30, 42]. By incorporating visual and textual modalities, MRSs can efectively break the bottleneck of data sparsity and cold-start problems inherent in traditional collaborative filtering (CF) [12, 19, 39]. Current state-of-the-art MRSs encompass a variety of approaches, such as graph-based message passing (e.g., LATTICE [37], MMGCN [30]) and contrastive learning paradigms (e.g., MENTOR [32], BM3 [42]). The core philosophy behind these models is to integrate multimodal semantics with structural collaborative signals [5, 23], aiming to construct a comprehensive and expressive latent representation for each user and item.

However, the prosperity of existing architectures implicitly relies on a fragile utopian assumption: “Modality Harmony” [13, 41].

They presume that multimodal features are inherently flawless, universally beneficial, and strictly aligned with users’ collaborative interaction patterns. In reality, this assumption is fundamentally flawed [27, 36]. Real-world recommender systems are inundated with modality-topology conflicts, driven by visual clickbaits, exaggerated textual claims, or mismatched modality semantics [9]. Such deceptive multimodal signals introduce false-positive edges into the user-item graph, fundamentally distorting the latent representation space [3, 17].

As illustrated in Figure 1, blindly absorbing these deceptive features triggers two critical bottlenecks in existing architectures. First, destructive interference via blind alignment: By forcefully aligning (via Contrastive Learning [2, 35]) or fusing (via GNNs [7, 31]) these deceptive modalities with the topological graph, existing models inevitably pollute the pristine CF space [38]. The user’s true preference representation collapses into the spurious visual appeal, leading to severe latent space distortion [20, 28]. Second, the zero-sum game in fusion: Traditional fusion mechanisms typically rely on softmax-based attention [25] or heuristic summation [30]. This enforces a zero-sum competition among modalities [18] (e.g., increasing the visual weight strictly decreases the tex tual weight), completely ignoring the fact that modalities should be decoupled and independently evaluated based on the topological consensus of diferent items. To date, the question of how to systematically disentangle deceptive modality noise from true semantic intent [24, 38] while preserving the modality’s intrinsic representational capacity remains largely under-explored.

To bridge this gap, we propose Orthogonal Purification and Topology-Guided MoE for Conflict-Aware Multimodal Recommendation (OrthoRec), a novel framework that transcends the modality harmony illusion via geometric decoupling. Our core insight is to treat the pure crowd-wisdom interaction patterns (i.e., the topological CF embeddings) as a trustworthy collaborative anchor. Rather than arbitrarily masking or discarding multimodal features, OrthoRec introduces a Collaborative-Guided Orthogonal Purification (CGOP) mechanism. We geometrically project the raw visual and textual features into directions parallel and orthogonal to the collaborative anchor. The parallel component retains the safe, consensus-aligned semantics, while the orthogonal component harboring both exploratory semantics and deceptive clickbait noise is adaptively scaled by a topology-aware conflict gate. Crucially, this purification is energy-preserving, since we dynamically restore the rectified vector’s $L _ { 2 }$ norm. This ensures that we only correct the deceptive “semantic direction” without artificially crippling the feature’s in herent representation capacity.

Furthermore, to eliminate the zero-sum fusion bottleneck, we design a Topology-Aware Routing Mixture-of-Experts (TAR-MoE). Instead of using raw modalities to compute attention, TAR-MoE uses the topological collaborative anchor as the routing condition, employing decoupled sigmoid gating to independently determine the optimal injection scale for each purified modality. Coupled with a safe-SSL objective that dynamically down-weights the contrastive penalty for inherently contradictory modalities, OrthoRec efectively prevents latent space distortion.

The main contributions of this work are summarized as follows:

• We explicitly challenge the “modality harmony” assumption by identifying the modality-topology conflict and the destructive interference caused by deceptive multimodal noise in MRSs.

• We propose an energy-preserving Orthogonal Purification (CGOP) mechanism. It provides a simple geometric approach to disentangle and rectify deceptive modality noise while alleviating magnitude decay.

• We introduce TAR-MoE with decoupled gating to break the zero-sum fusion bottleneck, and safe-SSL to mitigate the latent distortion caused by forced contrastive alignment.

• Experiments on three real-world Amazon datasets show that OrthoRec consistently outperforms competitive recent baselines, and exhibits improved robustness under injected modality noise and item sparsity.

## 2 Related Work

## 2.1 Graph-based Multimodal Recommendation

Multimodal Graph Neural Networks (GNNs) capture complex useritem connectivity and multimodal signals by constructing modalityspecific graphs [11, 26, 30, 31] or mining latent semantic structures [5, 37]. However, these methods typically rely on early feature concatenation or heuristic summation, operating under a strict “modality harmony” assumption. This blind fusion makes them highly vulnerable to destructive interference from modality-topology conflicts (e.g., visual clickbaits). In contrast, OrthoRec abandons naive fusion by introducing a collaborative-guided orthogonal purification (CGOP) mechanism, geometrically filtering out deceptive modality noise prior to graph aggregation.

## 2.2 Contrastive Alignment in Recommendation

Contrastive learning (CL) is widely adopted to mitigate data sparsity by maximizing mutual information between multimodal views and ID embeddings [10, 23, 42], or by distilling modality-invariant representations via cross-modal alignment [32, 41]. While efective, existing CL methods blindly pursue semantic alignment across all items. When an item’s visual and textual modalities are inherently contradictory, forcing alignment inevitably leads to latent space distortion, collapsing representations into spurious features [20, 28, 34]. To address this, our proposed safe-SSL utilizes a geometric conflict score to dynamically down-weight the alignment penalty for contradictory pairs, ensuring a safe optimization space.

## 2.3 Feature Denoising and Disentanglement

To combat ubiquitous data noise, recent studies explore disentangled representation learning [15, 38] and robust multimodal denoising [4, 27, 36]. Nevertheless, existing techniques face two critical dilemmas: (1) heuristic masking or edge dropping [21] directly discards features, causing severe magnitude decay and representation capacity loss, and (2) traditional attention mechanisms [25] rely on zero-sum softmax gating, restricting the independent utilization of multiple modalities [18]. OrthoRec addresses these bottlenecks by combining an energy-preserving orthogonal truncation (which rectifies deceptive semantic directions without shrinking the $L _ { 2 }$ norm) with TAR-MoE [14, 22], which employs decoupled sigmoid gating to autonomously determine modality injection scales free from zero-sum constraints.

## 3 Methodology

## 3.1 Problem Formulation

Let U and I denote the sets of users and items, respectively. The historical user-item interactions are formally represented by a bipartite graph $G = ( { \mathcal { U } } \cup { \mathcal { I } } , { \mathcal { E } } )$ , where an edge $( u , i ) \in \mathcal { E }$ indicates that user � has interacted with item �. For multimodal side information, each item $i \in \mathcal { I }$ is associated with raw visual and textual features extracted from pre-trained encoders, denoted as $\mathbf { x } _ { V } ^ { ( i ) } \in \mathbb { R } ^ { d _ { V } }$ and $\mathbf { x } _ { T } ^ { ( i ) } \in \mathbb { R } ^ { d _ { T } }$ . To project these heterogeneous modalities into a unified latent space, we employ modality-specific linear transformations:

$$
\mathbf { e } _ { m } ^ { ( i ) } = \mathbf { W } _ { m } \mathbf { x } _ { m } ^ { ( i ) } + \mathbf { b } _ { m } , \quad m \in \{ V , T \}\tag{1}
$$

where $\mathbf { W } _ { m } \in \mathbb { R } ^ { d \times d _ { m } }$ and $\mathbf { b } _ { m } \in \mathbb { R } ^ { d }$ are trainable weight matrices and biases, transforming the raw features into �-dimensional dense embeddings $\mathbf { e } _ { V } ^ { ( i ) }$ and ${ \bf e } _ { T } ^ { ( i ) }$ . Given the interaction graph $\mathcal { G }$ and the unified multimodal features $\{ { \bf e } _ { V } ^ { ( i ) } , { \bf e } _ { T } ^ { ( i ) } \}$ , our ultimate goal is to learn robust, noise-resistant representations for users and items. These purified representations are then utilized to accurately predict the unobserved interaction probability $\hat { y } _ { u , i }$

## 3.2 Collaborative Anchor Extraction

Most existing multimodal recommenders adopt an early-fusion strategy, blindly injecting visual and textual features into the graph message passing at the very beginning [5, 30, 37]. However, we argue that this conventional practice inevitably corrupts the topological structure with modality noise (e.g., visual clickbaits or mismatched semantics) [27]. Before aligning or fusing any heterogeneous modalities, it is imperative for the model to first establish a pristine, trustworthy structural reference.

To achieve this, we exclusively perform graph convolutions on the pure ID embeddings over G, isolating them from any multimodal interference. We initialize trainable ID embeddings ${ \bf e } _ { u } ^ { ( 0 ) } \in \mathbb { R } ^ { d }$ and ${ \bf e } _ { i } ^ { ( 0 ) } \in \mathbb { R } ^ { d }$ for users and items. Following the standard Light-GCN architecture [7], the message passing paradigm at the �-th layer is defined as:

$$
\mathbf { e } _ { u } ^ { ( l ) } = \sum _ { i \in N _ { u } } \frac { 1 } { \sqrt { | N _ { u } | | N _ { i } | } } \mathbf { e } _ { i } ^ { ( l - 1 ) } , \quad \mathbf { e } _ { i } ^ { ( l ) } = \sum _ { u \in N _ { u } } \frac { 1 } { \sqrt { | N _ { i } | | N _ { u } | } } \mathbf { e } _ { u } ^ { ( l - 1 ) }\tag{2}
$$

where $N _ { u }$ and ${ \cal N } _ { i }$ denote the first-order topological neighbors of user � and item �, respectively. The symmetric normalization term $1 / \sqrt { | N _ { u } | | N _ { i } | }$ serves to discount the impact of high-degree nodes, preventing representation over-smoothing. After � layers of propagation, we aggregate the embeddings across all layers to obtain the final ID-based representation:

$$
{ \bf e } _ { i } ^ { I D } = \frac { 1 } { L + 1 } \sum _ { l = 0 } ^ { L } { \bf e } _ { i } ^ { ( l ) }\tag{3}
$$

We define this resulting representation ${ \bf e } _ { i } ^ { I D }$ as the collaborative anchor $( \mathbf { e } _ { C F } \in \mathbb { R } ^ { d } )$ . Following LGMRec [5], we additionally enhance the anchor with a global hypergraph embedding: ${ \mathbf { e } } _ { C F } \gets { \mathbf { e } } _ { C F } + \alpha \cdot$ $\mathbf { e } ^ { G H E }$ , where $\mathbf { e } ^ { G H E }$ is the normalized hypergraph-propagated embedding and � is the hypergraph enhancement weight. This design inherits the ability to capture global structural dependencies, while our contributions operate on top of this anchor. By distilling the pure “crowd-wisdom” interaction patterns, ${ \bf e } _ { C F }$ serves as a robust collaborative consensus. It provides the essential geometric refer ence axis required to detect and rectify deceptive modality noise in the subsequent orthogonal purification stage.

## 3.3 Collaborative-Guided Orthogonal Purification

Given the reliable collaborative anchor $\mathbf { e } _ { C F } { \mathrm { , } }$ , we discard the naive assumption that raw multimodal features $\mathbf { e } _ { m } \left( m \in \{ T , V \} \right)$ are flawless. In real-world scenarios plagued by visual clickbaits or exaggerated textual claims, $\mathbf { e } _ { m }$ often encapsulates deceptive noise that strictly conflicts with the user’s true behavioral intent [27]. To address this, we propose Collaborative-Guided Orthogonal Purification (CGOP), a module that geometrically decouples $\mathbf { e } _ { m }$ into a safe consensus direction and an exploratory/noisy direction.

Geometric Decoupling. Taking the collaborative embedding ${ \bf e } _ { C F }$ as the structural reference axis, we project the raw multimodal feature $\mathbf { e } _ { m }$ onto it to extract the consistent parallel component:

$$
\mathbf { e } _ { m } ^ { \parallel } = \left( \frac { \mathbf { e } _ { m } \cdot \mathbf { e } _ { C F } } { | | \mathbf { e } _ { C F } | | _ { 2 } ^ { 2 } } \right) \mathbf { e } _ { C F }\tag{4}
$$

where (·) denotes the inner product and $| | \cdot | | _ { 2 }$ is the $L _ { 2 }$ norm. This parallel component precisely captures the modality semantics that strictly align with the topological behaviors verified by the crowd. Consequently, the residual forms the orthogonal component:

$$
\mathbf { e } _ { m } ^ { \perp } = \mathbf { e } _ { m } - \mathbf { e } _ { m } ^ { \parallel }\tag{5}
$$

By definition, the orthogonal component $\mathbf { e } _ { m } ^ { \perp }$ lies in a null space mathematically independent of the collaborative graph. It harbors modality-specific unique information, encompassing both beneficial long-tail exploratory semantics and deceptive clickbait noise.

Conflict-Aware Adaptive Truncation. To selectively filter out the deceptive noise within the orthogonal space, we compute a cosine-based conflict score $c _ { m } \in \left[ - 1 , 1 \right]$ ]:

$$
c _ { m } = \frac { \mathbf { e } _ { m } \cdot \mathbf { e } _ { C F } } { | | \mathbf { e } _ { m } | | _ { 2 } | | \mathbf { e } _ { C F } | | _ { 2 } }\tag{6}
$$

A lower (or negative) $c _ { m }$ indicates a severe modality-topology conflict. In practice, we center $c _ { m }$ by subtracting its batch mean $\bar { c } _ { m }$ before gating, so that the gate responds to an item’s relative conflict severity rather than the global modality gap between pre-trained features and the CF space. Crucially, this conflict score acts as a universal indicator of semantic trustworthiness. It not only governs the local feature truncation within CGOP but also serves as the fundamental guidance for the global safe alignment (detailed in Section 3.5). We dynamically scale the orthogonal space via a soft gating function $\gamma ( c _ { m } ) = \sigma ( g _ { \theta } ( c _ { m } ) )$ , where $g _ { \boldsymbol { \theta } } ( \cdot )$ is a modalityshared lightweight two-layer MLP and �(·) is the sigmoid activation. The temporarily truncated feature is thus formulated as:

$$
\tilde { \mathbf { e } } _ { t m p } = \mathbf { e } _ { m } ^ { \parallel } + \gamma ( c _ { m } ) \cdot \mathbf { e } _ { m } ^ { \perp }\tag{7}
$$

Energy-Preserving Normalization. In high-dimensional representation learning, the magnitude $( \mathrm { i } . \mathrm { e } . , L _ { 2 }$ norm) of an embedding vector typically correlates with its feature expressiveness and confidence [28]. Directly outputting the truncated feature $\tilde { \mathbf { e } } _ { t m p }$ would lead to severe magnitude decay, artificially shrinking the modality’s intrinsic representation capacity. To preserve the representation magnitude, we introduce an energy-preserving normalization:

![](images/a2b42574dcb9fb8274bd4b6d7aa68888652724372eab99ada64e9531a73fb672.jpg)  
Figure 2: The overall architecture of OrthoRec. It consists of four main modules: (1) collaborative anchor extraction; (2) energy-preserving orthogonal purification to filter deceptive noise; (3) topology-aware routing MoE for decoupled fusion; and (4) safe-SSL and optimization, which prevents latent space distortion.

$$
\tilde { \mathbf { e } } _ { m } = \frac { \tilde { \mathbf { e } } _ { t m p } } { | | \tilde { \mathbf { e } } _ { t m p } | | _ { 2 } } \cdot | | \mathbf { e } _ { m } | | _ { 2 }\tag{8}
$$

Through this geometric operation, CGOP rectifies the noisy modality’s deceptive semantic direction while retaining its representa tional energy. In essence, it reorients the modality vector away from topological conflicts without artificially shrinking its magnitude. Having obtained the purified multimodal features $\tilde { \mathbf { e } } _ { V }$ and $\tilde { \mathbf { e } } _ { T }$ the next critical step is to dynamically aggregate them with the collaborative anchor ${ \bf e } _ { C F }$ to form a unified item representation.

## 3.4 Topology-Aware Routing MoE (TAR-MoE)

To perform the aforementioned aggregation, traditional methods typically rely on heuristic summation or softmax-based attention mechanisms [25]. However, softmax inherently traps modalities in a zero-sum game [18], assigning a higher weight to the visual feature strictly suppresses the textual counterpart. This rigid competition contradicts real-world recommendation scenarios where an item might ofer equally crucial visual and textual cues, demanding high injection scales for both simultaneously.

To break this zero-sum bottleneck, we propose TAR-MoE equipped with decoupled sigmoid gating [14]. We conceptualize the fusion process as a non-mutually-exclusive routing mechanism, where the local and global graph topology $( { \bf e } _ { C F } )$ serves as the ultimate condition to independently evaluate each modality. The routing weight $g _ { m } \in ( 0 , 1 )$ for each modality is calculated as:

$$
g _ { m } = \sigma \big ( \mathrm { M L P } _ { r o u t e \_ m } ( \mathbf e _ { C F } ) / \tau _ { r } \big ) , \quad m \in \{ T , V \}\tag{9}
$$

where $\tau _ { r }$ is a routing temperature and �(·) is the sigmoid function. The final unified item representation $\mathbf { e } _ { f i n a l }$ is then dynamically aggregated via:

$$
{ \bf e } _ { f i n a l } = { \bf e } _ { C F } + \gamma \left( g _ { T } \cdot \tilde { { \bf e } } _ { T } + g _ { V } \cdot \tilde { { \bf e } } _ { V } \right)\tag{10}
$$

where $\gamma$ is a global modality injection scale balancing the purified multimodal signals against the collaborative anchor (note that � is distinct from the item-wise truncation gate $\gamma ( c _ { m } )$ in CGOP). In practice, the purified features are passed through identity-initialized expert layers and $L _ { 2 } \cdot$ normalized before injection, which stabilizes the injected signal magnitudes. Unlike previous attention mechanisms computed from the raw, potentially noisy modalities themselves, our decoupled routing strategy empowers the pure graph structure to autonomously determine the injection scale for each purified signal. This design allows the model to fully capitalize on all beneficial multimodal features without destructive interference.

## 3.5 Conflict-Aware Safe Contrastive Learning

While TAR-MoE efectively fuses multimodal features for the primary recommendation task, it is also crucial to align the underlying semantic spaces of diferent modalities. To this end, cross-modal contrastive learning (SSL) is widely adopted as an auxiliary task to maximize the mutual information between paired texts and images [23].

The Latent Distortion Dilemma. Traditional SSL blindly pursues semantic alignment across all items. However, this introduces a critical vulnerability: False Positive Alignment. When an item’s visual and textual modalities are inherently contradictory (e.g., a factual text paired with a deceptive clickbait image), forcing them to align with each other inevitably leads to latent space distortion [28].

Safe-SSL Objective. To ensure a safe optimization boundary, we propose to dynamically penalize the contrastive alignment for inherently contradictory modality pairs. Rather than introducing external heuristics, we directly reuse $\mathrm { C G O P } { \mathrm { { s } } }$ own conflict estimates inside the contrastive objective. We seamlessly reuse the centered geometric conflict scores $\hat { c } _ { T } ,$ �ˆ<sub>�</sub> (i.e., $c _ { m }$ after subtracting its batch mean, as detected during the CGOP stage in Section 3.3). We design a dynamic penalty weight $\lambda _ { s a f e } ^ { ( i ) }$ for each item �:

$$
\lambda _ { s a f e } ^ { ( i ) } = \sigma \left( \frac { \hat { c } _ { T } ^ { ( i ) } + \hat { c } _ { V } ^ { ( i ) } } { \tau _ { s a f e } } \right)\tag{11}
$$

where $\sigma ( \cdot )$ is the sigmoid function and $\tau _ { s a f e }$ is a temperature hy perparameter controlling the penalty sharpness.

Since �(·) is bounded, $\lambda _ { s a f e } ^ { ( i ) }$ lies within (0, 1). If both modalities are relatively harmonious with the collaborative anchor $( \hat { c } _ { T } , \hat { c } _ { V } > 0 )$ $\lambda _ { s a f e }$ approaches 1 (full alignment). Conversely, if either modality conflicts with the topology substantially more than average (e.g., $\hat { c } _ { V } \ll 0$ due to visual noise), $\lambda _ { s a f e }$ decays smoothly towards 0.

The standard InfoNCE loss is then reformulated into our safe-SSL loss:

$$
\mathcal { L } _ { S a f e - S S L } = - \sum _ { i \in \mathcal { B } } \lambda _ { s a f e } ^ { ( i ) } \ln \frac { \exp ( \sin ( \tilde { \bf e } _ { T } ^ { ( i ) } , \tilde { \bf e } _ { V } ^ { ( i ) } ) / \tau _ { s s l } ) } { \sum _ { j \in \mathcal { B } } \exp ( \sin ( \tilde { \bf e } _ { T } ^ { ( i ) } , \tilde { \bf e } _ { V } ^ { ( j ) } ) / \tau _ { s s l } ) }\tag{12}
$$

where B denotes the mini-batch, sim $( \cdot , \cdot )$ is the cosine similarity function, and $\tau _ { s s l }$ is the contrastive temperature. The denominator sums over all in-batch negative samples �.

By integrating this dynamic penalty, safe-SSL acts as an automatic safety valve. It instructs the model to confidently reject the alignment of toxic pairs, thereby protecting the purified semantic space from optimization distortion.

## 3.6 Model Optimization

The overall framework is optimized end-to-end in a multi-task learning manner. For the primary recommendation task, we employ the widely-used Bayesian Personalized Ranking (BPR) loss, which encourages the model to rank observed positive items higher than unobserved negative ones:

$$
\mathcal { L } _ { B P R } = - \sum _ { ( u , i , j ) \in O } \ln \sigma ( \hat { y } _ { u , i } - \hat { y } _ { u , j } )\tag{13}
$$

where $O = \{ ( u , i , j ) \mid i \in J _ { u } ^ { + } , j \in J \backslash J _ { u } ^ { + } \}$ denotes the set ofpairwise training triplets, with ${ \cal T } _ { u } ^ { + }$ representing the items interacted by user �, and � sampled as an unobserved negative item. In practice we additionally apply a small negative margin � inside the sigmoid $( \mathrm { i . e . , } \sigma ( \hat { y } _ { u , i } - \hat { y } _ { u , j } - m ) )$ to encourage a larger separation, where � is grid-searched on the validation set. The predicted preference score is calculated via the inner product $\begin{array} { r } { \hat { y } _ { u , i } = \mathbf { e } _ { u } ^ { \top } \mathbf { e } _ { f i n a l , i } . } \end{array}$ where $\mathbf { e } _ { u }$ is the final user representation obtained from the graph encoder.

To jointly optimize the primary recommendation task and the conflict-aware semantic alignment, the overall objective function is formulated as:

$$
\mathcal { L } _ { t o t a l } = \mathcal { L } _ { B P R } + \lambda _ { c l } \mathcal { L } _ { S a f e - S S L } + \lambda _ { r e g } \| \Theta \| _ { 2 } ^ { 2 }\tag{14}
$$

where $\lambda _ { c l }$ is a hyperparameter controlling the safe contrastive learning strength (the safe-SSL weight, reported as the CL weight in

Table 1: Statistics of the experimental datasets.
<table><tr><td>Dataset</td><td>#Users</td><td>#Items</td><td>#Inters.</td><td>Sparsity</td></tr><tr><td>Baby</td><td>19,445</td><td>7,050</td><td>160,792</td><td>99.88%</td></tr><tr><td>Sports</td><td>35,598</td><td>18,357</td><td>296,337</td><td>99.95%</td></tr><tr><td>Clothing</td><td>39,387</td><td>23,033</td><td>278,677</td><td>99.97%</td></tr></table>

Section $4 . 6 ) , \lambda _ { r e g }$ is the weight for $L _ { 2 }$ regularization to prevent overfitting, and Θ denotes all trainable parameters in the model.

## 4 Experiments

We conduct extensive experiments on three real-world datasets to comprehensively evaluate the proposed OrthoRec framework.

## 4.1 Experimental Setup

Datasets. We evaluate our model on three widely-used public datasets from the Amazon Product Reviews [16]: Baby, Sports and Outdoors, and Clothing, Shoes and Jewelry. Detailed statistics are presented in Table 1. To ensure a fair comparison, we follow the same data processing and filtering settings as in prior works that use these common benchmark datasets [41, 42]. To represent the visual and textual modalities, we utilize pre-trained BEiT [29] and BGE [1] sentence embeddings, respectively, across all datasets.

Evaluation Metrics. We use an 8:1:1 train/validation/test split for all interactions. To evaluate the ranking performance, we use two standard metrics: Recall@� and Normalized Discounted Cumulative Gain (NDCG@�), where � ∈ {10, 20}. We report the average metrics across all test users.

Baselines. We compare OrthoRec with three groups of representative baselines: General CF Models: MF-BPR [19], LightGCN [7], SimGCL [35] and LayerGCN [40], which solely rely on useritem interaction structures. Graph-based Multimodal Models: VBPR [6], MMGCN [30], GRCN [31], DualGNN [26], LATTICE [37] and LGMRec [5]. These models incorporate multimodal features into graph propagation using early fusion, modality-aware graphs, or structural refinement. Contrastive Learning & Denoising Models: SLMRec [23], BM3 [42], FREEDOM [41], MMGCL [8] and DA-MRS [33]. These SOTA models use cross-modal contrastive learning or denoising-and-alignment objectives to purify multimodal semantics and user feedback.

Implementation Details. OrthoRec is implemented in PyTorch. The latent dimension � is set to 64 on Baby and to 128 on Sports and Clothing. We optimize via Adam (batch size 4096) with early stop ping based on validation Recall@20, using a patience of 150 epochs. Key hyperparameters (e.g., learning rate, �<sub>2</sub> weight, $\lambda _ { c l } )$ are tuned via grid search on the validation set, and the main hyperparameters’ sensitivity is reported in Section 4.6.

## 4.2 Overall Performance (RQ1)

Table 2 summarizes the comparison. (1) OrthoRec improves consistently over all baselines, ranking first on every metric of every dataset, with gains up to 11.28%. At �=20 the NDCG gain exceeds the Recall gain throughout, so purification not only retrieves more relevant items but ranks them higher, as expected when deceptive directions are removed. (2) Blind fusion can be actively harmful. Several multimodal baselines fall below the strongest pure-CF model despite access to strictly more information. Multimodal signal is thus not free, and the “Modality Harmony” assumption is violated often enough to cost these models the benefit of their features, the premise CGOP targets by truncating deceptive directions without shrinking representational capacity. (3) The strongest baselines hedge against noise. FREEDOM and LGMRec lead the field, both through structural rather than semantic treatment: a frozen denoised item-item graph and a global hypergraph. Their edge over unguarded early fusion confirms that noise must be handled, while their gap to OrthoRec shows structural hedging is no substitute for filtering the features themselves and routing them without zero-sum competition, as corroborated in Section 4.3.

Table 2: Overall performance comparison on three datasets. The best results are highlighted in bold, and the strongest baseline results are underlined. “Improv.” denotes OrthoRec’s relative improvement over the best baseline.
<table><tr><td rowspan="2">Datasets Models</td><td colspan="4">Baby</td><td colspan="4">Sports</td><td colspan="4">Clothing</td></tr><tr><td>R@10</td><td>R@20</td><td>N@10</td><td>N@20</td><td>R@10</td><td>R@20</td><td>N@10</td><td>N@20</td><td>R@10</td><td>R@20</td><td>N@10</td><td>N@20</td></tr><tr><td>MF-BPR</td><td>0.0357</td><td>0.0575</td><td>0.0192</td><td>0.0249</td><td>0.0432</td><td>0.0653</td><td>0.0241</td><td>0.0298</td><td>0.0187</td><td>0.0279</td><td>0.0103</td><td>0.0126</td></tr><tr><td>LightGCN</td><td>0.0479</td><td>0.0754</td><td>0.0257</td><td>0.0328</td><td>0.0569</td><td>0.0864</td><td>0.0311</td><td>0.0387</td><td>0.0340</td><td>0.0526</td><td>0.0188</td><td>0.0236</td></tr><tr><td>SimGCL</td><td>0.0513</td><td>0.0804</td><td>0.0273</td><td>0.0350</td><td>0.0601</td><td>0.0919</td><td>0.0327</td><td>0.0414</td><td>0.0356</td><td>0.0549</td><td>0.0195</td><td>0.0244</td></tr><tr><td>LayerGCN</td><td>0.0529</td><td>0.0820</td><td>0.0281</td><td>0.0355</td><td>0.0594</td><td>0.0916</td><td>0.0323</td><td>0.0406</td><td>0.0371</td><td>0.0566</td><td>0.0200</td><td>0.0247</td></tr><tr><td>VBPR</td><td>0.0423</td><td>0.0663</td><td>0.0223</td><td>0.0284</td><td>0.0558</td><td>0.0856</td><td>0.0307</td><td>0.0384</td><td>0.0281</td><td>0.0415</td><td>0.0158</td><td>0.0192</td></tr><tr><td>MMGCN</td><td>0.0378</td><td>0.0615</td><td>0.0200</td><td>0.0261</td><td>0.0370</td><td>0.0605</td><td>0.0193</td><td>0.0254</td><td>0.0218</td><td>0.0345</td><td>0.0110</td><td>0.0142</td></tr><tr><td>DualGNN</td><td>0.0448</td><td>0.0716</td><td>0.0240</td><td>0.0309</td><td>0.0568</td><td>0.0859</td><td>0.0310</td><td>0.0385</td><td>0.0454</td><td>0.0683</td><td>0.0241</td><td>0.0299</td></tr><tr><td>SLMRec</td><td>0.0529</td><td>0.0775</td><td>0.0290</td><td>0.0353</td><td>0.0663</td><td>0.0990</td><td>0.0365</td><td>0.0450</td><td>0.0452</td><td>0.0675</td><td>0.0247</td><td>0.0303</td></tr><tr><td>GRCN</td><td>0.0536</td><td>0.0829</td><td>0.0286</td><td>0.0363</td><td>0.0609</td><td>0.0925</td><td>0.0323</td><td>0.0408</td><td>0.0431</td><td>0.0661</td><td>0.0229</td><td>0.0278</td></tr><tr><td>LATTICE</td><td>0.0547</td><td>0.0850</td><td>0.0292</td><td>0.0370</td><td>0.0620</td><td>0.0953</td><td>0.0335</td><td>0.0421</td><td>0.0492</td><td>0.0733</td><td>0.0268</td><td>0.0330</td></tr><tr><td>FREEDOM</td><td>0.0627</td><td>0.0992</td><td>0.0330</td><td>0.0424</td><td>0.0717</td><td>0.1089</td><td>0.0385</td><td>0.0481</td><td>0.0628</td><td>0.0941</td><td>0.0341</td><td>0.0420</td></tr><tr><td>BM3</td><td>0.0564</td><td>0.0883</td><td>0.0301</td><td>0.0383</td><td>0.0656</td><td>0.0980</td><td>0.0355</td><td>0.0438</td><td>0.0422</td><td>0.0621</td><td>0.0231</td><td>0.0281</td></tr><tr><td>MMGCL</td><td>0.0522</td><td>0.0779</td><td>0.0288</td><td>0.0357</td><td>0.0660</td><td>0.0992</td><td>0.0359</td><td>0.0445</td><td>0.0433</td><td>0.0667</td><td>0.0239</td><td>0.0291</td></tr><tr><td>LGMRec</td><td>0.0639</td><td>0.0989</td><td>0.0337</td><td>0.0430</td><td>0.0719</td><td>0.1068</td><td>0.0387</td><td>0.0477</td><td>0.0555</td><td>0.0828</td><td>0.0302</td><td>0.0371</td></tr><tr><td>DA-MRS</td><td>0.0561</td><td>0.0895</td><td>0.0302</td><td>0.0388</td><td>0.0608</td><td>0.0952</td><td>0.0322</td><td>0.0410</td><td>0.0518</td><td>0.0779</td><td>0.0278</td><td>0.0344</td></tr><tr><td>OrthoRec</td><td>0.0695</td><td>0.1057</td><td>0.0375</td><td>0.0466</td><td>0.0762</td><td>0.1158</td><td>0.0421</td><td>0.0514</td><td>0.0658</td><td>0.0955</td><td>0.0352</td><td>0.0436</td></tr><tr><td>Improv.</td><td>+8.76%</td><td>+6.55%</td><td>+11.28%</td><td>+8.37%</td><td>+5.98%</td><td>+6.34%</td><td>+8.79%</td><td>+6.86%</td><td>+4.78%</td><td>+1.49%</td><td>+3.23%</td><td>+3.81%</td></tr></table>

## 4.3 Ablation Study (RQ2)

To validate our design choices, we compare OrthoRec against four variants: (1) w/o CGOP Truncation disables the adaptive truncation gate (� = 1), (2) w/o TAR-MoE (softmax) replaces the decoupled Sigmoid routing with standard softmax, (3) w/o safe-SSL degrades our conflict-aware penalty to standard InfoNCE, and (4) w/o SSL removes the self-supervised objective entirely. Based on Figure 3, we conclude:

(1) Decoupled routing avoids zero-sum competition bottlenecks. Replacing TAR-MoE with softmax is the single most damag ing ablation on every dataset, and especially severe on Clothing. The efect stems from the constraint itself rather than injection magnitude: re-tuning � for the softmax variant on a magnitude-matched grid selects the same scale OrthoRec already uses. Forcing heterogeneous modalities into mutually exclusive distributions thus limits joint expressiveness, whereas independent routing is crucial for robust fusion. (2) Adaptive truncation and decoupled routing are complementary. Routing controls how much purified signal enters, truncation controls which directions. Disabling truncation isolates the directional efect alone, which is modest on curated benchmark features. This ablation is thus a lower bound on CGOP’s contribution: uncorrupted features carry little deceptive component to remove, and directional filtering matters once conflicts intensify, as Section 4.4 confirms. (3) Cross-modal alignment is indispensable and must be conflict-aware. Removing SSL entirely (w/o SSL) drops R@20 by 14.1–19.7% across all datasets, so alignment carries a large share of the gain. Degrading it to standard InfoNCE (w/o safe-SSL) separates how alignment is applied from whether it is: on Baby the unguarded objective recovers none of the gap, so forcing contradictory pairs together is as damaging as no alignment at all. Elsewhere it operates near its safety margin at the tuned weight, with Section 4.6 showing sharp degradation once the weight exceeds that margin.

![](images/40ff620983774b2fa51446b6f91876f75a4cd05c2fb9c7c81afab131a5059ef0.jpg)  
Figure 3: Ablation study on OrthoRec’s key design choices.

![](images/d2e3cb3246b6fa0787b427665453ab79183d28a5affb71d47ded9071c88f08b3.jpg)  
Figure 4: Performance evaluation under varying visual noise ratios.

## 4.4 Robustness against Modality Noise (RQ3)

To evaluate robustness against modality-topology conflicts, we inject synthetic noise by randomly shufling visual features for a proportion $( p \in \{ 0 \% 1 0 \% 2 0 \% 3 0 \% \}$ ) of items during both training and inference. Figure 4 compares OrthoRec against LayerGCN (pure CF), FREEDOM (graph-based), and LGMRec (deep-fusion). The trajectories expose a utilization-robustness trade-of that OrthoRec substantially alleviates: (1) A flat curve measures modality reliance, not robustness. LayerGCN is perfectly horizontal because it uses no visual features, yet trails OrthoRec by 0.020– 0.033 Recall@20 at every ratio. FREEDOM is likewise near-flat (≤ 0.8% degradation) as its frozen item-item graph weights the visual channel by only 0.1. Flatness here comes from declining to use the modality, and reading it as noise tolerance would reward the very conservatism our method avoids. (2) Unfiltered deep fusion converts modality signal into a liability. LGMRec integrates modalities aggressively and is competitive on clean data, but degrades on every dataset and stays below OrthoRec at every ratio, by a margin that holds essentially constant rather than shrinking. As the two models see identical features and difer mainly in whether the signal is purified and decoupled, naïve early fusion evidently amplifies toxic noise instead of filtering it. (3) OrthoRec exploits semantics without inheriting their fragility. OrthoRec attains the highest clean-data accuracy on all three datasets and, against the comparable deep-fusion baseline, both starts higher and falls more slowly (10.6% relative R@20 drop at 30% noise versus 11.6%). Its residual degradation is the expected cost of genuinely using the visual channel; what CGOP buys is a bound on corruption propagation, since truncating the anchor-orthogonal component removes the directions along which shufled features would otherwise pollute the collaborative space. The trade-of curve thus shifts rather than tilts. As this protocol shufles visual features, robustness to naturally occurring corruption is left for future work.

![](images/90348e13d0b35c142877d367b16c5149caeaa27aed8e9c5e751d8dac97b87e73.jpg)  
Figure 5: Recommendation performance across item popularity groups.

## 4.5 Item Popularity and Sparsity Analysis (RQ4)

To evaluate sparsity handling, we partition test items by training frequency: Cold (≤ 5), Normal (6 ∼ 20), and Popular (> 20). Figure 5 compares OrthoRec against LGMRec and LayerGCN, revealing key insights: (1) Complementary strengths across popularity groups. OrthoRec’s most pronounced advantage lies in the Cold group (about +41% R@20 over LGMRec), while the two models stay within 4% on Popular items, where abundant collaborative edges already yield high-quality CF representations. The gain is thus concentrated where collaborative evidence is scarce rather than spread uniformly across popularity levels. (2) Safe and robust utilization for cold items. For Cold items with minimal structural edges, unfiltered early fusion is most vulnerable, as deceptive signals cannot be counterbalanced by reliable collaborative evidence, which is exactly where purification is most valuable. This matches the routing behaviour: since TAR-MoE conditions the injection scale on topological consensus, an item whose evidence is too thin to corroborate its features receives a conservative injection rather than an unverified one. On Baby, the gains on Normal and Popular items show that this conservatism is confined to the regime where the anchor cannot yet certify the signal. (3) The indispensable role of multimodal semantics. The pure CF model (LayerGCN) consistently underperforms, nearly collapsing on Cold items, confirming that purified multimodal semantics remain essential to overcome structural sparsity.

## 4.6 Hyperparameter Sensitivity (RQ5)

To evaluate OrthoRec’s robustness, we investigate its sensitivity to four key hyperparameters in Figure 6:

Impact of $\tau _ { r } .$ . OrthoRec is stable across $\tau _ { r } \in [ 0 . 0 1 , 1 . 0 ]$ , with fluctuation below 4%. Since $\tau _ { r }$ only sharpens the sigmoid on the routing logits, the gates keep comparable relative preferences over a wide range, so no delicate tuning is required.

Impact of �. � controls the global hypergraph enhancement injected into the collaborative anchor. Baby and Sports are stable across the range, while Clothing requires $\alpha \ge 0 . 3 .$ This follows from the anchor’s role: Clothing is the sparsest dataset, so local edges alone cannot yield a trustworthy geometric reference and the global hypergraph must supply the missing structure before purification can decide which directions to keep. The dependence thus confirms the mechanism rather than revealing fragility, and $\alpha = 0 . 5$ sits safely inside the stable region on all three datasets.

![](images/b56745768fba6efbd2c9294adf4ea3dfd5a8fd0e279d608c06c9b2eb3cba782a.jpg)  
Figure 6: OrthoRec’s hyperparameter sensitivity on three datasets.

Impact of $\gamma .$ . Performance improves sharply once � exceeds 1, then flattens, peaking within $\gamma \in \left[ 1 . 2 5 , 1 . 5 \right]$ . Multimodal semantics thus matter, yet excessive amplification introduces noise that overpowers collaborative signals.

Impact of $\lambda _ { c l }$ . A moderate contrastive signal $( \lambda _ { c l } \in [ 1 0 ^ { - 5 } , 1 0 ^ { - 4 } ] )$ yields the best alignment, and removing it $\left( \lambda _ { c l } = 0 \right)$ is clearly worse everywhere, matching Section 4.3. Pushing past this range $( \lambda _ { c l } \ge 5 \times$ 10<sup>−4</sup>) degrades Clothing markedly, the “Latent Distortion Dilemma” the conflict score is designed to contain. The best range is broad and consistent across datasets, so the sensitivity reflects alignment pressure rather than a brittle setting.

## 4.7 Qualitative Analysis (RQ6)

To understand OrthoRec’s inner workings, we visualize the latent distributions and gating weights in Figure 7. (1) Heterogeneous modality-topology conflicts. Raw conflict scores $c _ { m } =$ cos $\left( \mathbf { e } _ { m } , \mathbf { e } _ { C F } \right)$ (top row) reveal severe geometric divergence between pre-trained semantics and the CF space. Textual scores concentrate near −0.99, while visual scores spread from below −0.8 up to +0.56 on Clothing, confirming that raw modalities sufer varying spatial misalignments and rendering static fusion suboptimal. (2) Intelligent suppression via decoupled gating. The bottom row plots TAR-MoE routing weights $( g _ { T } , g _ { V } )$ . Softmax MoE enforces $g _ { T } + g _ { V } = 1 ,$ so a toxic modality can only be suppressed by promoting another rather than being filtered on its own merits, consistent with the degradation in Figure 3. Our decoupled sigmoid gating instead acts as an independent filter: facing extreme textual con flict it suppresses text $( g _ { T } \sim 0 . 0 2 )$ while visual weights span up to 0.83. The learned gate sums stay well below the softmax diagonal (averaging 0.24–0.40 across datasets), with >99% of Clothing items satisfying $g _ { T } + g _ { V } < 0 . 5$ . Breaking the zero-sum constraint thus grants the freedom to suppress multiple noisy modalities at once.

Case Study. Figure 8 compares both models on Clothing for User #23578. Although Clothing shows the highest visual scores overall, this user sits in the negative tail: their images conflict sharply with the collaborative topology and act as deceptive noise. Such per-item variation is what item-wise truncation and topologyconditioned routing exist to handle, since a dataset-level weight would inherit the average rather than this user’s conflict. LGM-Rec’s unfiltered fusion overfits to appealing but irrelevant visual semantics and misses both targets, whereas CGOP truncates the toxic components, shielding the user’s true intent and retrieving both targets in the top-20.

![](images/214780bc935e6d376e45fa62aa39a473c8e48a9288639ae4b4bffa686659a7c9.jpg)  
Figure 7: Visualizations of (a) latent conflict scores and (b) TAR-MoE routing weights across the three datasets.

![](images/50d1170f450961a3f49a0d0f66bc6e81adef57945d418a0eda53fba9d25e4d09.jpg)  
Figure 8: Case study of User #23578.

## 5 Conclusion

We propose OrthoRec to alleviate the modality-topology conflicts that distort latent spaces in multimodal recommendation. By unifying Collaborative-Guided Orthogonal Purification (CGOP) for geometric noise filtration, decoupled TAR-MoE routing, and safe-SSL, OrthoRec delivers consistent accuracy improvements over competitive baselines and improved robustness under modality noise and item sparsity within the evaluated settings.

## Acknowledgments

This work was supported by the Strategic Support Fund of City University of Hong Kong under Project 7020230, and by the National Natural Science Foundation of China under Grant 62302420.

## References

[1] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024. Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. arXiv preprint arXiv:2402.03216 4, 5 (2024).

[2] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geofrey Hinton. 2020. A simple framework for contrastive learning of visual representations. In International conference on machine learning. 1597–1607.

[3] Chen Gao, Yu Zheng, Wenjie Wang, Fuli Feng, Xiangnan He, and Yong Li. 2024. Causal inference in recommender systems: A survey and future directions. ACM Transactions on Information Systems 42, 4 (2024), 1–32.

[4] Xiaobao Guo, Alex Kot, and Adams Wai-Kin Kong. 2023. Pace-Adaptive and Noise Resistant Contrastive Learning for Multimodal Feature Fusion. IEEE Transactions on Multimedia 25 (2023), 9437–9448

[5] Zhiqiang Guo, Jianjun Li, Guohui Li, Chaoyang Wang, Si Shi, and Bin Ruan. 2024. Lgmrec: Local and global graph learning for multimodal recommendation. In Proceedings ofthe AAAI conference on artificial intelligence, Vol. 38. 8454–8462.

[6] Ruining He and Julian McAuley. 2016. VBPR: visual bayesian personalized ranking from implicit feedback. In Proceedings ofthe AAAI conference on artificial intelligence, Vol. 30.

[7] Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang. 2020. Lightgcn: Simplifying and powering graph convolution network for recommendation. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval. 639–648.

[8] Yuezihan Jiang, Changyu Li, Gaode Chen, Peiyi Li, Qi Zhang, Jingjian Lin, Peng Jiang, Fei Sun, and Wentao Zhang. 2024. Mmgcl: Meta knowledge-enhanced multi-view graph contrastive learning for recommendations. In Proceedings of the 18th ACM conference on recommender systems. 538–548.

[9] Yangqin Jiang, Lianghao Xia, Wei Wei, Da Luo, Kangyi Lin, and Chao Huang. 2024. Difmm: Multi-modal difusion model for recommendation. In Proceedings of the 32nd ACM international conference on multimedia. 7591–7599.

[10] Xixun Lin, Rui Liu, Yanan Cao, Lixin Zou, Qian Li, Yongxuan Wu, Yang Liu, Dawei Yin, and Guandong Xu. 2025. Contrastive modality-disentangled learning for multimodal recommendation. ACM Transactions on Information Systems 43, 3 (2025), 1–31.

[11] Jialin Liu, Zhaorui Zhang, and Ray CC Cheung. 2026. MoToRec: Sparse-Regularized Multimodal Tokenization for Cold-Start Recommender. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 40. 15324–15332.

[12] Jialin Liu, Zhaorui Zhang, and Ray CC Cheung. 2026. SGA-GNN: Semantic-Guided Adaptive Graph Neural Network for Cold-Start Multimodal Recommendation. In ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 1266–1270.

[13] Qidong Liu, Jiaxi Hu, Yutian Xiao, Xiangyu Zhao, Jingtong Gao, Wanyu Wang, Qing Li, and Jiliang Tang. 2024. Multimodal recommender systems: A survey. Comput. Surveys 57, 2 (2024), 1–17.

[14] Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, and Ed H Chi. 2018. Modeling task relationships in multi-task learning with multi-gate mixture-ofexperts. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining. 1930–1939.

[15] Jianxin Ma, Chang Zhou, Peng Cui, Hongxia Yang, and Wenwu Zhu. 2019. Learning disentangled representations for recommendation. Advances in neural information processing systems 32 (2019).

[16] Julian McAuley, Christopher Targett, Qinfeng Shi, and Anton Van Den Hengel. 2015. Image-based recommendations on styles and substitutes. In Proceedings ofthe 38th international ACM SIGIR conference on research and development in information retrieval. 43–52.

[17] Shanlei Mu, Yaliang Li, Wayne Xin Zhao, Jingyuan Wang, Bolin Ding, and Ji-Rong Wen. 2022. Alleviating spurious correlations in knowledge-aware recommendations through counterfactual generator. In Proceedings of the 45th international ACM SIGIR conference on research and development in information retrieval. 1401– 1411.

[18] Xiaokang Peng, Yake Wei, Andong Deng, Dong Wang, and Di Hu. 2022. Balanced multimodal learning via on-the-fly gradient modulation. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 8238–8247.

[19] Stefen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme. 2012. BPR: Bayesian personalized ranking from implicit feedback. arXiv preprint arXiv:1205.2618 (2012).

[20] Joshua Robinson, Ching-Yao Chuang, Suvrit Sra, and Stefanie Jegelka. 2020. Contrastive learning with hard negative samples. arXiv preprint arXiv:2010.04592 (2020).

[21] Yu Rong, Wenbing Huang, Tingyang Xu, and Junzhou Huang. 2019. Dropedge: Towards deep graph convolutional networks on node classification. arXiv preprint arXiv:1907.10903 (2019).

[22] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geofrey Hinton, and Jef Dean. 2017. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538 (2017).

[23] Zhulin Tao, Xiaohao Liu, Yewei Xia, Xiang Wang, Lifang Yang, Xianglin Huang, and Tat-Seng Chua. 2022. Self-supervised learning for multimedia recommenda tion. IEEE Transactions on Multimedia 25 (2022), 5107–5116.

[24] Yonglong Tian, Chen Sun, Ben Poole, Dilip Krishnan, Cordelia Schmid, and Phillip Isola. 2020. What makes for good views for contrastive learning? Advances in neural information processing systems 33 (2020), 6827–6839.

[25] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems 30 (2017).

[26] Qifan Wang, Yinwei Wei, Jianhua Yin, Jianlong Wu, Xuemeng Song, and Liqiang Nie. 2021. Dualgnn: Dual graph neural network for multimedia recommendation. IEEE Transactions on Multimedia 25 (2021), 1074–1084.

[27] Qi Wang, Anbiao Wu, Ye Yuan, Yishu Wang, Guangqing Zhong, Xuefeng Gao, and Chenghu Yang. 2024. Noise-resistant graph neural networks for sessionbased recommendation. In Asia-Pacific Web (APWeb) and Web-Age Information Management (WAIM) Joint International Conference on Web and Big Data. 144– 160.

[28] Tongzhou Wang and Phillip Isola. 2020. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In International conference on machine learning. 9929–9939.

[29] Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, et al. 2023. Image as a foreign language: Beit pretraining for vision and vision-language tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 19175–19186.

[30] Yinwei Wei, Xiang Wang, Liqiang Nie, Xiangnan He, Richang Hong, and Tat-Seng Chua. 2019. MMGCN: Multi-modal Graph Convolution Network for Personalized Recommendation of Micro-video. In Proceedings ofthe 27th ACM International Conference on Multimedia. 1437–1445.

[31] Yinwei Wei, Xiang employment Wang, Liqiang Nie, Xiangnan He, and Tat-Seng Chua. 2020. Graph-Refined Convolutional Network for Multimedia Recom mendation with Implicit Feedback. In Proceedings ofthe 28th ACM International Conference on Multimedia. 3541–3549.

[32] Jinfeng Xu, Zheyu Chen, Shuo Yang, Jinze Li, Hewei Wang, and Edith CH Ngai. 2025. Mentor: multi-level self-supervised learning for multimodal recommendation. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39. 12908–12917.

[33] Guipeng Xv, Xinyu Li, Ruobing Xie, Chen Lin, Chong Liu, Feng Xia, Zhanhui Kang, and Leyu Lin. 2024. Improving multi-modal recommender systems by denoising and aligning multi-modal content and user feedback. In Proceedings of the 30th ACM SIGKDD conference on knowledge discovery and data mining. 3645–3656.

[34] Yuhao Yang, Chao Huang, Lianghao Xia, Chunzhen Huang, Da Luo, and Kangyi Lin. 2023. Debiased contrastive learning for sequential recommendation. In Proceedings ofthe ACM web conference 2023. 1063–1073.

[35] Junliang Yu, Hongzhi Yin, Xin Xia, Tong Chen, Lizhen Cui, and Quoc Viet Hung Nguyen. 2022. Are graph augmentations necessary? simple graph contrastive learning for recommendation. In Proceedings ofthe 45th international ACM SIGIR conference on research and development in information retrieval. 1294–1303.

[36] Chunkai Zhang, Wenjing Zheng, Quan Liu, Junli Nie, and Hanyu Zhang. 2022. SEDGN: Sequence enhanced denoising graph neural network for session-based recommendation. Expert Systems with Applications 203 (2022), 117391.

[37] Jinghao Zhang, Yanqiao Zhu, Qiang Liu, Shu Wu, Shuhui Wang, and Liang Wang. 2021. Mining Latent Structures for Multimedia Recommendation. In Proceedings ofthe 29th ACM International Conference on Multimedia. 3872–3880.

[38] Yu Zheng, Chen Gao, Xiang Li, Xiangnan He, Yong Li, and Depeng Jin. 2021. Disentangling User Interest and Conformity for Recommendation with Causal Embedding. In Proceedings ofthe Web Conference 2021. 2980–2991.

[39] Yujia Zheng, Siyi Liu, Zekun Li, and Shu Wu. 2021. Cold-start Sequential Recommendation via Meta Learner. In Proceedings ofthe Thirty-Fifth AAAI Conference on Artificial Intelligence. 4706–4713.

[40] Xin Zhou, Donghai Lin, Yong Liu, and Chunyan Miao. 2023. Layer-refined graph convolutional networks for recommendation. In 2023 IEEE 39th International Conference on Data Engineering (ICDE). 1247–1259.

[41] Xin Zhou and Zhiqi Shen. 2023. A Tale of Two Graphs: Freezing and Denoising Graph Structures for Multimodal Recommendation. In Proceedings ofthe 31st ACM International Conference on Multimedia. 935–943.

[42] Xin Zhou, Hongyu Zhou, Yong Liu, Zhiwei Zeng, Chunyan Miao, Pengwei Wang, Yuan You, and Feijun Jiang. 2023. Bootstrap Latent Representations for Multi modal Recommendation. In Proceedings of the ACM Web Conference 2023. 845–854.