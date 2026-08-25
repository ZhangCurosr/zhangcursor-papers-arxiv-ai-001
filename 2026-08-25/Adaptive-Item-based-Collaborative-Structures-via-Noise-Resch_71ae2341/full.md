# Adaptive Item-based Collaborative Structures via Noise Rescheduling in Difusion for Generative Recommendation

JIAQI WANG<sup>∗</sup>, Tongji University, China

TIANYING LIU<sup>∗</sup> and HENG CHANG<sup>†</sup>, Huawei Technologies Co., Ltd., China

JIHONG GUAN<sup>†</sup> and WENGEN LI, Tongji University, China

SHUIGENG ZHOU, Fudan University, China

Discrete Difusion Models (DDMs) have recently been introduced to recommendation systems, modeling user history as a token generation process via iterative denoising. However, while efective at capturing user-level sequential patterns, these methods often fail to explicitly integrate item-based collaborative filtering information, a critical component for accurate recommendation. This deficiency manifests in two key aspects: (1) the item representation is often semantic-focused, lacking collaborative priors for difusion training; and (2) the denoising process employs a uniform noise schedule, treating all tokens indiscriminately and ignoring item-level adaptive structural dependencies. To bridge this gap, we propose ANR-DifRec, a unified framework designed to encode item-based collaborative structures into discrete difusion for generative recommendation. First, we explicitly incorporate an item co-occurrence matrix to guide semantic ID generation, providing a structured collaborative prior for discrete difusion training. Second, we introduce an item-based adaptive noise rescheduling mechanism that dynamically adjusts denoising weights according to both local contextual recoverability and behavior-aware item dependencies. Specifically, the proposed strategy jointly models intra-item structural context and inter-item collaborative signals, enabling structure-aware denoising during difusion training. Extensive experiments on multiple benchmarks demonstrate that our method consistently outperforms state-of-the-art generative recommendation models. Code: https://github.com/CalmaQi/ANR-DifRec.

CCS Concepts: • Information systems → Recommender systems.

Additional Key Words and Phrases: Generative Recommendation, Discrete Difusion Model

## ACM Reference Format:

Jiaqi Wang, Tianying Liu, Heng Chang, Jihong Guan, Wengen Li, and Shuigeng Zhou. 2026. Adaptive Item-based Collaborative Structures via Noise Rescheduling in Difusion for Generative Recommendation. In Proceedings of (Conference acronym ’XX). ACM, New York, NY, USA, 23 pages. https://doi.org/XXXXXXX.XXXXXXX

## 1 Introduction

The recent surge in Large Language Models (LLMs) has revolutionized natural language processing [1, 25, 44, 52], inspiring a paradigm shift in recommender systems towards generative recommendation [18, 24, 26, 27, 38]. Unlike

Authors’ Contact Information: Jiaqi Wang, wangjq@tongji.edu.cn, Tongji University, Shanghai, China; Tianying Liu, tianying\_liu@outlook.com; Heng Chang, changh.heng@gmail.com, Huawei Technologies Co., Ltd., Shanghai, China; Jihong Guan, jhguan@tongji.edu.cn; Wengen Li, lwengen@tongji.edu. cn, Tongji University, Shanghai, China; Shuigeng Zhou, sgzhou@fudan.edu.cn, Fudan University, Shanghai, China.

![](images/c0c7221aa6c4f0a71965671f5cc5e9da7251fa2ac2849d76ee400d0e8c0fb382.jpg)  
Fig. 1. Illustration of the proposed item-based adaptive noise rescheduling strategy. (a) Existing difusion-based recommendation models employ a uniform noise schedule, assigning identical denoising weights to masked tokens regardless of their contextual dependencies. (b) Our method dynamically adjusts the denoising weight for each token by jointly considering two complementary signals: (1) local recoverability estimation, and (2) behavior-aware dependency modeling. The final denoising weight is obtained by combining these two signals, enabling structure-aware difusion training.

traditional discriminative approaches that predict scores for a candidate set via point-wise ranking [4, 40], the generative paradigm reconceptualizes it as a sequence-to-sequence generation task, which is analogous to next-token prediction in autoregressive language modeling paradigms

The paradigm of generative recommendation generally hinges on two fundamental pillars: item tokenization [19] and generation [53]. The first component, item tokenization, maps each item to a sequence of Semantic IDs (SIDs). This is typically achieved through methods such as hierarchical clustering or residual quantization (e.g., RQ-VAE), which capture semantic relationships within the ID structure. After the SIDs are generated, a generative model is applied to predict the target item tokens based on the user history SID sequence.

Discrete difusion provides a natural generation paradigm for SID-based recommendation [28, 35]. Compared with autoregressive models that generate SID tokens in a fixed left-to-right order, discrete difusion reconstructs masked tokens from partially corrupted sequences, enabling bidirectional context modeling and adaptive-order generation. This property is particularly suitable for recommendation, where the tokens in an SID jointly specify an item and may exhibit diferent semantic granularity and predictability.

While the adaptation of discrete difusion models to generative recommendation has yielded promising results, we argue that a fundamental bottleneck remains: the failure to explicitly encode the item-based collaborative filtering information within the generative framework. Current DDM-based approaches primarily focus on modeling user-based sequential patterns via iterative token denoising. However, they largely overlook the intrinsic item-based structural relationships, the core signal in classic collaborative filtering. This issue creates a disconnect between the generic Manuscript submitted to ACM

generation paradigm and the specific requirements of recommendation, manifesting in two critical limitations where item-level structures are neglected in both item representation and the training objective.

The first limitation stems from the lack of collaborative signals in item tokenization. Existing discrete difusion-based approaches predominantly rely on semantic embeddings (e.g., derived from textual descriptions) to construct SIDs. While efective at capturing content similarity, these representations are oblivious to the collaborative item relationships that are pivotal for SID-based recommendation [42, 53]. For instance, two semantically distinct items may frequently co-occur in user histories, a pattern that standard SIDs fail to encode. Consequently, the generative model lacks a structured collaborative prior, forcing it to infer these item associations solely from sparse sequence contexts without support from the item representation.

Nevertheless, such a collaborative prior is insuficient, since diferent masked tokens may receive diferent amounts of contextual support under diferent corruption patterns. The second limitation stems from the uniform discrete difusion training process, which cannot explicitly model item-based collaborative signals. As illustrated in Figure 1, existing DDM-based recommendation models generally adopt a uniform noise schedule, assigning the same noise level and denoising weight $( \mathrm { e . g . , } w _ { 1 } = w _ { 2 } = \cdot \cdot \cdot = w _ { 9 } )$ to all masked tokens regardless of their context. However, recommendation sequences naturally exhibit uneven information density. The recoverability of a masked token largely depends on the availability of related collaborative contexts, including both neighboring items in the sequence and informative tokens within the same semantic ID. Under a uniform schedule, structurally supported tokens and context-isolated tokens are treated equally during training, making it dificult for the model to efectively exploit item-level collaborative structures and behavioral dependencies.

To bridge the gap, we propose ANR-DifRec, a unified framework designed to systematically integrate item-based collaborative structures into generative recommendation. To address the first challenge and provide a structural prior for the adaptive noise scheduler, we enhance Semantic ID construction by explicitly incorporating collaborative structural signals. Specifically, we construct an item-item co-occurrence matrix from historical interaction sequences and apply matrix factorization to derive structural item representations. These collaborative embeddings are then fused with pre-trained textual semantic features and discretized into hierarchical token sequences. By mapping items into this structurally-aware semantic token space, the resulting SIDs capture both semantic similarity and implicit transition patterns, thereby providing more informative conditions for the discrete difusion denoising process.

To address the second challenge, we propose an item-based adaptive noise rescheduling mechanism for difusionbased recommendation. As illustrated in Figure 1, instead of applying identical denoising weights to all masked positions, our method dynamically adjusts the denoising weight of each token according to two complementary signals: (1) Local Recoverability Estimation, which estimates how easily a masked token can be recovered based on geometric dependencies from surrounding intra-item and inter-item contexts; and (2) Behavior-aware Dependency Modeling, which leverages attention-based interaction modeling to adaptively capture collaborative dependencies in the user behavior sequence. By jointly integrating these two signals, our method assigns larger denoising weights to tokens supported by more informative structural and behavioral contexts, thereby providing fine-grained guidance for the difusion denoising process. For clarity, the denoising weight shown in Figure 1(b) is computed for the masked token in Item 3. The figure demonstrates how surrounding unmasked contexts and behavior-aware item dependencies jointly influence the adaptive denoising weight assignment for the token. The denoising weight can be interpreted as an inverse efective noise level. Tokens receiving larger denoising weights are considered easier to recover, corresponding to lower noise levels. In contrast, tokens with sparse contextual information are assigned smaller weights, indicating higher noise levels. In summary, our main contributions are formulated as follows:

Manuscript submitted to ACM

• We propose an item-based adaptive noise rescheduling strategy in discrete difusion that enables dynamic learning of item-to-item collaborative structures for generative recommendation.

• We introduce an explicit item co-occurrence–guided semantic ID generation mechanism, which serves as a lightweight collaborative prior to stabilize difusion training.

• Extensive experiments across five diverse, real-world benchmark datasets demonstrate that ANR-DifRec consistently and significantly outperforms state-of-the-art autoregressive and difusion-based generative recommenda tion models.

## 2 Related Works

## 2.1 Generative Recommendation

Unlike traditional discriminative approaches that formulate recommendations as a ranking task over a predefined candidate set [12, 13, 20, 23, 37, 47, 50, 54], generative recommendation reframes the problem as a sequence-to-sequence generation task [5, 7, 26, 36, 51]. In this paradigm, predicting the next item is isomorphic to next-token prediction in language modeling, unlocking the potential to leverage Large Language Models (LLMs) for recommendation.

Recent research has predominantly shifted towards generative recommendation with semantic IDs [18, 22, 41] to bypass the ineficiency of generating lengthy natural language text. These works typically employ residual quantization (e.g., RQ-VAE [21]) or hierarchical clustering to convert atomic item IDs into highly compressed, discrete token sequences. TIGER [33] establishes this paradigm by generating tuple-based SIDs, enabling Transformers to predict items autoregressively. VQ-Rec [15] parallelly enriches this approach by deriving discrete codes from item text to bridge content semantics. Building on these foundations, subsequent works such as LC-Rec [53] and LETTER [39] refine the tokenization mechanism through learnable codebooks and hierarchical regularization. Most recently, advanced frameworks like OneRec [6] and RPG [17] unify multi-modal signals and optimize retrieval strategies for eficient generation. Despite these advancements, the dominant autoregressive (AR) models rely heavily on a strict left-to-right generation order. This AR nature introduces two critical bottlenecks: (1) Unidirectional Bias, which restricts the model’s ability to leverage bidirectional contextual semantics crucial for user intent understanding; and (2) Error Propagation, where a misprediction at an early hierarchical token (e.g., the coarse-grained cluster center) may propagate errors to subsequent token generations, leading to completely irrelevant item retrievals.

To circumvent the limitations of autoregressive decoding, LLaDA-Rec [35] and DifGRM [28] introduce the discrete difusion model to recommendation, enabling a non-autoregressive, bidirectional generation process. However, these DDM-based methods inherently rely on a uniform noise strategy, where tokens are masked stochastically without accounting for the density of surrounding contextual information. This denoising process fails to distinguish between easy-to-recover tokens and hard ones, severely limiting its ability to capture collaborative filtering signals through structure-aware guidance. Furthermore, relying solely on semantic-based tokenization lacks structured collaborative priors, further hindering the robust modeling of inter-item dependencies under noisy contexts.

## 2.2 Discrete Difusion Models

Denoising Difusion Probabilistic Models (DDPMs) [14, 46] have achieved state-of-the-art performance in continuous state-space generation tasks like image synthesis. To address the discrete and categorical nature of language and recommendation data, Discrete Difusion Models [34, 43, 45, 48, 55] operate directly on categorical variables [8]. Manuscript submitted to ACM

Adaptive Item-based Collaborative Structures via Noise Rescheduling in Difusion for Generative Recommendation 5

D3PM [3] explicitly generalizes the difusion framework to discrete state spaces using transition matrices, introducing absorbing states (e.g., [MASK]) to model the controlled corruption and progressive denoising processes.

Building on these foundations, recent works have significantly scaled up this paradigm to the LLM era. LLaDA [31] demonstrates that masked difusion models trained from scratch can achieve performance highly competitive with strong autoregressive methods. Parallelly, frameworks like bd3lm [2] and Dream [49] further improve generation eficiency and quality through block-wise token updates and refined loss reweighting objectives. In the context of recommendation systems, the discrete difusion model ofers a compelling alternative to AR models [28, 35]. By training a bidirectional Transformer to reconstruct masked tokens from a partially corrupted sequence, it inherently learns to utilize global user history. However, SIDs in recommendation possess both intra-item and inter-item structures, making them fundamentally diferent from natural language text. Our work builds upon the discrete difusion foundation but fundamentally diverges by introducing an item-based adaptive noise rescheduling mechanism. By integrating dynamic attention and structural priors, our approach is specifically tailored for collaborative SIDs, distinguishing it from general text difusion models.

## 3 Preliminaries and Background

## 3.1 Problem Formulation

Consider a standard recommendation scenario involving a user set U and an item set I. For each user $u \in \mathcal { U } ,$ , we denote their interaction sequence as $\mathcal { H } _ { u } = \{ v _ { 1 } , v _ { 2 } , \dotsc , v _ { | \mathcal { H } _ { u } | } \}$ , ordered chronologically, where $\upsilon _ { k } \in \mathcal { I }$ . The primary objective is to estimate the probability distribution of the subsequent item $v _ { n e x t }$ given the historical context $\mathcal { H } _ { u }$

Within the generative recommendation paradigm, items are not treated as atomic ID embeddings but are represented as sequences of discrete tokens. Specifically, each item � is tokenized into a semantic ID, denoted as a vector of length $L \colon \mathbf { x } _ { v } = \left[ x _ { v , 1 } , x _ { v , 2 } , \ldots , x _ { v , L } \right]$ , where each $x _ { v , l }$ represents a discrete token derived from a quantization codebook. Consequently, a user’s interaction history can be flattened into a unified token sequence $S _ { u } ,$ , formed by concatenating the SIDs of interacted items:

$$
S _ { u } = [ \mathbf { x } _ { v _ { 1 } } \oplus \mathbf { x } _ { v _ { 2 } } \oplus \cdot \cdot \cdot \oplus \mathbf { x } _ { v _ { | \mathcal { H } _ { u } | } } ] ,\tag{1}
$$

where ⊕ denotes sequence concatenation. The next-item prediction task is thus reframed as maximizing the conditional likelihood of the target item’s token sequence ${ \bf x } _ { n }$ given the context $S _ { u } \mathrm { : }$

$$
\Theta = \underset { \theta } { \operatorname { a r g m a x } } P _ { \theta } ( \mathbf { x } _ { n } \mid S _ { u } ) ,\tag{2}
$$

where � represents the learnable parameters of the model.

## 3.2 Discrete Difusion Modeling

Diverging from the conventional Autoregressive approach that generates tokens from left to right, Discrete Difusion operates as a non-autoregressive refinement process. It models generation as a reverse denoising trajectory, initiating from a completely corrupted state and progressively recovering the clean data. Let $\mathbf { x } _ { n } ^ { T }$ be the fully corrupted state of the next item SID consisting entirely of [MASK] tokens. Inference is discretized into � steps. Starting from $\mathbf { x } _ { n } ^ { T }$ , the Transformer encoder denoising network iteratively predicts the clean tokens. At each step � (from � down to 1), the denoising network takes the current state $\mathbf { x } _ { n } ^ { t }$ of the next item and the unmasked user interaction history $S _ { H }$ as input to predict the clean tokens for all masked positions. In each iteration, the model predicts the probability distribution $P _ { \theta } ( \mathbf { x } _ { n } ^ { 0 } | \mathbf { x } _ { n } ^ { t } , S _ { H } )$ . Let $n _ { t } \in [ 1 , L ]$ denote the number of tokens to be kept at step �. We select the top- $\cdot n _ { t }$ tokens with the Manuscript submitted to ACM highest prediction confidence and re-mask the remaining $L - n _ { t }$ positions to form the input for the next step $\mathbf { x } _ { n } ^ { t - 1 }$ Formally, the generation process can be viewed as maximizing the conditional likelihood over the refinement trajectory:

![](images/c838409a584f2e0d83a05673e22a15adbebe08c65b446531ff4ee8c1b5e97022.jpg)  
Fig. 2. The architecture of ANR-DifRec. (a) SID generation via fusion of semantic and collaborative signals. (b) Discrete difusion training enhanced by item-based adaptive noise rescheduling, enabling structure-aware denoising to capture collaborative filtering signals. (c) Constrained discrete difusion inference with validity filtering

$$
P _ { \theta } ( \mathbf { x } _ { n } \mid S _ { H } ) = \prod _ { t = 1 } ^ { T } \prod _ { l = 1 } ^ { L } \left\{ { P _ { \theta } ( x _ { n , l } \mid \mathbf { x } _ { n } ^ { t } , S _ { H } ) , } \quad \mathrm { i f } x _ { n , l } ^ { t } = [ \mathsf { M A S K } ] , \right.\tag{3}
$$

Here, $x _ { n , l } ^ { t }$ denotes the token at position � in the state $\mathbf { x } _ { n } ^ { t }$

## 4 Methodology

In this section, we present our proposed framework. As illustrated in Figure 2, the framework consists of three components: (1) Item Co-occurrence-Guided Semantic ID Generation (Section 4.1): we construct SIDs by fusing item textual information with collaborative signals; (2) Item-based Discrete Difusion Training (Section 4.2): we introduce an item-based adaptive noise rescheduling mechanism and employ a Transformer encoder to model the generation process; (3) Constrained Discrete Difusion Inference (Section 4.3): we employ iterative denoising with beam search augmented by an SID constraint module to ensure the validity of the generated token sequence.

## 4.1 Item Co-occurrence-Guided SID Generation

4.1.1 Item Representation Fusion. Let I denote the set of items. For each item $i \in \mathcal { I }$ , we construct a continuous representation by jointly encoding semantic and collaborative information. We first obtain a semantic embedding $\mathbf { e } _ { i } ^ { \mathrm { s e m } } \in \mathbb { R } ^ { d _ { s } }$ using a pre-trained sentence embedding encoder. To fuse collaborative signals, we construct an item co-occurrence matrix $\mathbf { C } \in \mathbb { R } ^ { | \mathcal { I } | \times | \mathcal { I } | }$ <sup>|</sup>, where $\mathbf { C } _ { i j }$ denotes the number of times items � and � co-occur in user interaction Manuscript submitted to ACM

Adaptive Item-based Collaborative Structures via Noise Rescheduling in Difusion for Generative Recommendation 7

sequences. To extract low-dimensional collaborative features, we perform truncated singular value decomposition (SVD) on C:

$$
\mathbf { C } \approx \mathbf { U } _ { k } \boldsymbol { \Sigma } _ { k } \mathbf { V } _ { k } ^ { \top } ,\tag{4}
$$

where $\mathbf { U } _ { k } \in \mathbb { R } ^ { | \mathcal { I } | \times k }$ contains the top-� left singular vectors. The collaborative embedding of item � is given by:

$$
\mathbf { e } _ { i } ^ { \mathrm { c o l l } } = \mathbf { U } _ { k } ( i , : ) \Sigma _ { k } .\tag{5}
$$

The final fused item representation is obtained by concatenation:

$$
\mathbf { e } _ { i } = \big [ \mathbf { e } _ { i } ^ { \mathrm { s e m } } ; \mathbf { e } _ { i } ^ { \mathrm { c o l l } } \big ] \in \mathbb { R } ^ { d _ { s } + d _ { c } } .\tag{6}
$$

This fusion embedding preserves both semantic similarity and item collaborative proximity.

4.1.2 RQ-KMeans-based SID Generation. Given the fused item representations $\{ \mathbf { e } _ { o } \} _ { v \in \mathcal { I } }$ , we apply Residual Quantized K-Means (RQ-KMeans) to obtain SIDs. Each item is encoded as a sequence of � discrete tokens. RQ-KMeans performs multi-stage vector quantization in a residual manner. At stage $l = 1 ,$ , K-Means is applied to cluster $\mathbf { e } _ { v }$ into $\left| C _ { 1 } \right|$ centroids, producing the first token:

$$
\begin{array} { r } { x _ { v , 1 } = \underset { \mathbf { c } \in C _ { 1 } } { \arg \operatorname* { m i n } } \ : \| \mathbf { e } _ { v } - \mathbf { c } \| _ { 2 } ^ { 2 } . } \end{array}\tag{7}
$$

The residual is then computed as: $\mathbf { r } _ { v , 1 } = \mathbf { e } _ { v } - \mathbf { c } _ { x _ { v , 1 } }$ . Subsequent stages recursively quantize the residual:

$$
x _ { v , l } = \underset { \mathbf { c } \in C _ { l } } { \arg \operatorname* { m i n } } \ : \| \mathbf { r } _ { v , l - 1 } - \mathbf { c } \| _ { 2 } ^ { 2 } , \quad \mathbf { r } _ { v , l } = \mathbf { r } _ { v , l - 1 } - \mathbf { c } _ { x _ { v , l } . }\tag{8}
$$

After � stages, each item � is represented as a SID:

$$
\mathrm { S I D } ( v ) = ( x _ { v , 1 } , x _ { v , 2 } , \ldots , x _ { v , L } ) .\tag{9}
$$

By incorporating collaborative signals into the input space of RQ-KMeans, the resulting SIDs preserve both semantic and co-occurrence structures, yielding more informative corrupted contexts and an efective collaborative prior for discrete difusion training.

## 4.2 Item-based Discrete Difusion Training

Given a user interaction history consisting of $N - 1$ items, each represented by � SID tokens, the user history token sequence is denoted by:

$$
S _ { H } = ( x _ { 1 , 1 } , \ldots x _ { 1 , L } , \ldots , x _ { N - 1 , 1 } \ldots x _ { N - 1 , L } ) ,\tag{10}
$$

The difusion training process consists of two stages: forward masking and reverse reconstruction. In the forward process, tokens in the input sequence are independently replaced with a [MASK] token with probability $\tau \in ( 0 , 1 ]$ . In the reverse process, the model learns to reconstruct the original sequence from the noisy state by predicting masked tokens. We adopt two masking learning strategies for discrete difusion training: user-history level masking and next-item level masking.

4.2.1 User-History Level Masking. User-history level masking aims to capture contextual dependencies among all tokens in the user history. During training, for each batch, we sample a masking ratio $\tau \sim \mathcal { U } ( 0 , 1 ]$ . Each token in $S _ { H }$ is independently masked with probability �, producing a partially corrupted sequence $S _ { H } ^ { \tau }$ . Let $\mathbb { I } [ S _ { H , i } ^ { \tau } = [ \mathsf { M A S K } ] ]$ ] be an indicator function denoting whether the �-th token in $S _ { H }$ is masked. The reconstruction loss is defined as:

$$
\mathcal { L } _ { \mathrm { U H } } = - \mathbb { E } _ { \tau , S _ { H } , S _ { H } ^ { \tau } } \left[ \frac { 1 } { \tau } \sum _ { i = 1 } ^ { ( N - 1 ) \times L } \mathbb { I } [ S _ { H , i } ^ { \tau } = [ \mathsf { M A S K } ] ] \log p _ { \theta } ( S _ { H , i } \mid S _ { H } ^ { \tau } ) \right] .\tag{11}
$$

4.2.2 Item-based Adaptive Noise Rescheduling. To efectively capture item-based collaborative filtering signals during the generative process, we propose an item-based adaptive noise rescheduling strategy for user-history level masking. Unlike uniform denoising objectives, our mechanism quantifies the context of each token by jointly considering its structural hierarchy and dynamic user behavior. Specifically, the contextual dependency between token � and token � is modeled as the combination of a local recoverability estimation and a behavior-aware item dependency score.

First, we define a static structural dependency matrix $\mathbf { W } ^ { s t r u c t }$ based on symmetric geometric decay:

$$
W _ { i j } ^ { s t r u c t } = \phi \big ( | I ( i ) - I ( j ) | \big ) + \mathbb { I } [ I ( i ) = I ( j ) ] \cdot \psi ( | T ( i ) - T ( j ) | ) ,\tag{12}
$$

where $I ( i )$ denotes the item index associated with token �, and $T ( i )$ denotes its intra-item token position. The functions $\phi ( d )$ and $\psi ( d )$ are geometric decay kernels defined as

$$
\begin{array} { r } { p ( 1 - p ) ^ { d - 1 } , \quad d \geq 1 , } \end{array}
$$

which model inter-item and intra-item contextual recoverability, respectively. Here, � is the distance, and $\boldsymbol { p } \in ( 0 , 1 ]$ controls the decay sharpness: a larger � emphasizes nearby contextual information, while a smaller � captures broader contextual dependencies. We set $\phi ( 0 ) = 0$ and $\psi ( 0 ) = 0$ to avoid self-dependency.

However, purely relying on static structural context cannot adaptively capture user-specific behavioral dependencies. To address this limitation, we further introduce behavior-aware item dependency modeling based on item-level attention. Specifically, let $\mathbf { e } _ { m } \in \mathbb { R } ^ { d }$ denote the representation of the �-th item in the user sequence, obtained by mean pooling the hidden states of its corresponding tokens from the last Transformer layer. To avoid degenerate optimization, we detach $\mathbf { e } _ { m }$ from the computational graph when computing the reweighting coeficients. This design prevents the model from trivially reducing the objective by manipulating the adaptive weights. We compute the item-level attention score between item � and item � as:

$$
A _ { m n } ^ { i t e m } = \frac { \exp ( \mathbf { e } _ { m } ^ { \top } \mathbf { W } _ { Q } ^ { \top } \mathbf { W } _ { K } \mathbf { e } _ { n } / \sqrt { d } ) } { \sum _ { k } \exp ( \mathbf { e } _ { m } ^ { \top } \mathbf { W } _ { Q } ^ { \top } \mathbf { W } _ { K } \mathbf { e } _ { k } / \sqrt { d } ) } ,\tag{13}
$$

where ${ \bf W } _ { Q }$ and $\mathbf { W } _ { K }$ are learnable projection matrices. The item-level dependency score is then projected back to token pairs according to their corresponding item indices:

$$
A _ { i j } = A _ { I ( i ) , I ( j ) } ^ { i t e m } .\tag{14}
$$

Finally, the adaptive contextual dependency weight is obtained by combining the structural context with the behavior-aware dependency score:

$$
W _ { i j } = W _ { i j } ^ { s t r u c t } + \alpha A _ { i j } ,\tag{15}
$$

where $\alpha \geq 0$ is a learnable scaling parameter. This formulation encourages stronger denoising guidance when two tokens are either structurally proximate or semantically correlated through user behavioral patterns. To estimate the denoising importance of each masked token, we aggregate contextual contributions from all unmasked tokens:

$$
\mathbf { W } _ { i } = \sum _ { j = 1 } ^ { ( N - 1 ) \times L } \mathbb { I } [ S _ { H , j } ^ { \tau } \neq [ \mathsf { M A S K } ] ] \cdot W _ { i j } .\tag{16}
$$

Manuscript submitted to ACM

Adaptive Item-based Collaborative Structures via Noise Rescheduling in Difusion for Generative Recommendation 9

The reweighted denoising objective is formulated as:

$$
\mathcal { L } _ { 1 } = - \mathbb { E } _ { \tau , S _ { H } , S _ { H } ^ { \tau } } \left[ \frac { 1 } { \tau } \sum _ { i = 1 } ^ { ( N - 1 ) \times L } \mathbf { W } _ { i } \cdot \mathbb { I } [ S _ { H , i } ^ { \tau } = \left[ \mathsf { M A S K } \right] ] \log p _ { \theta } ( S _ { H , i } \mid S _ { H } ^ { \tau } ) \right] .\tag{17}
$$

This adaptive noise rescheduling strategy redistributes denoising importance according to both local contextual recoverability and behavior-aware item dependencies. Tokens surrounded by richer unmasked contexts or strongly correlated items receive larger denoising weights, corresponding to lower efective noise levels during training. In contrast, tokens with sparse contextual support are assigned smaller weights. By explicitly aligning denoising supervision with item-level collaborative structures, the proposed strategy provides fine-grained and structure-aware guidance for discrete difusion training.

4.2.3 Next-Item Level Masking. Next-item level masking focuses on predicting the target item, conditioned on the fully observed user history. We define $\mathbf { x } _ { n } = \left( x _ { n , 1 } , \ldots , x _ { n , L } \right)$ as the SIDs of the next item. Given mask ratio �, each token in ${ \bf x } _ { n }$ is independently masked with probability �, yielding $\mathbf { x } _ { n } ^ { \tau } .$ The partially masked target item tokens are concatenated with the unmasked user history $S _ { H }$ and fed into the model. The next-item prediction loss is defined as:

$$
\mathcal { L } _ { 2 } = - \mathbb { E } _ { \tau , S _ { H } , \mathbf { x } _ { n } , \mathbf { x } _ { n } ^ { \tau } } \left[ \frac { 1 } { \tau } \sum _ { i = 1 } ^ { L } \mathbb { I } [ x _ { n , i } ^ { \tau } = [ \mathsf { M A S K } ] ] \log \rho _ { \theta } ( x _ { n , i } \mid S _ { H } , \mathbf { x } _ { n } ^ { \tau } ) \right] .\tag{18}
$$

4.2.4 Overall Training Objective. The final training objective combines the two masking strategies:

$$
\begin{array} { r } { \mathcal { L } = \lambda \mathcal { L } _ { 1 } + \mathcal { L } _ { 2 } , } \end{array}\tag{19}
$$

where � balances user history modeling and target item generation.

## 4.3 Constrained Discrete Difusion Inference

At inference time, our model generates the next item conditioned on the observed user history by progressively denoising a fully masked SID.

4.3.1 Initialization. Given a user history sequence $S _ { H }$ , we initialize the target item SID as a fully masked sequence $\mathbf { x } _ { n } ^ { T } = ( x _ { n , 1 } ^ { T } , \ldots , x _ { n , L } ^ { T } ) , \quad x _ { n , m } ^ { T } = [ \mathsf { M A S K } ]$ , and concatenate it with $S _ { H }$ as the input to the denoising network, implemented as a Transformer encoder.

4.3.2 Iterative Denoising with Beam Search. Inference proceeds over � difusion steps. At each step �, the model predicts token distributions for the masked positions:

$$
p _ { \theta } ( x _ { n , m } \mid S _ { H } , \mathbf { x } _ { n } ^ { t } ) .\tag{20}
$$

Following LLaDA-Rec [35], we adopt a beam search strategy to maintain multiple candidate partially denoised sequences. At each step, candidate sequences in the beam are expanded by unmasking a subset of positions and selecting highprobability tokens. The beam retains the top-� candidates according to their cumulative log-likelihoods

4.3.3 SID-Validity Constrained Decoding. In discrete difusion, SIDs may be revealed in an arbitrary order. Enforcing validity solely at the level of individual positions is insuficient, as independently valid token assignments may form an invalid combination. To guarantee that the final generated SID corresponds to a real item, we enforce an SID-validity constraint during decoding.

During inference, we maintain a candidate item set $C ^ { t } \subseteq { \cal { I } }$ at difusion step �, representing all items that remain consistent with the currently revealed tokens. Formally, let $O ^ { t } = \{ ( m , x _ { n , m } ) \mid x _ { n , m } \neq [ \mathsf { M A S K } ] \}$ denote the set of observed (position, token) pairs at step �. The candidate set is defined as:

$$
C ^ { t } = \left\{ i \in { \cal I } \big | \forall ( m , x _ { n , m } ) \in { \cal O } ^ { t } , { \mathrm { S I D } } ( i ) [ m ] = x _ { n , m } \right\} .\tag{21}
$$

When unmasking a token at position �, the valid candidate token set is restricted to:

$$
\begin{array} { r } { \chi _ { m } ^ { t } = \left\{ x \big | \exists i \in C ^ { t } \mathrm { ~ s u c h ~ t h a t ~ } \mathrm { S I D } ( i ) [ m ] = x \right\} . } \end{array}\tag{22}
$$

The predictive distribution is renormalized over $X _ { m } ^ { t }$ before token selection. After assigning a value to position �, the candidate set is updated accordingly.

4.3.4 Final Prediction. After all � tokens are unmasked, each beam candidate corresponds to a valid SID. The candidate with the highest overall likelihood is selected, and its corresponding item is returned as the final recommendation.

To consolidate the aforementioned components and provide a clear implementation overview, we summarize the overall workflow of ANR-DifRec. The complete training pipeline is outlined in Algorithm 1. Correspondingly, the constrained iterative decoding process used during the inference phase is detailed in Algorithm 2.

```latex
Algorithm 1 Training Process of ANR-DifRec
Require: User interaction sequences H, Items I, Difusion steps � , Masking ratio distribution U (0, 1], Hyperparameter
$\lambda , p .$
Ensure: Trained network parameters �.
1: Ofline Preparation:
2: Compute Item-Item co-occurrence matrix $\mathbf { C } .$
3: Apply SVD to obtain collaborative embeddings and fuse with semantic embeddings.
4: Generate semantic IDs using RQ-KMeans
5: Online Training:
6: while not converged do
7: Sample a batch of user interaction histories $s _ { H }$ and target items $\mathbf { x } _ { n } .$
8: Sample masking ratios $\tau _ { 1 } , \tau _ { 2 } \sim \mathcal { U } ( 0 , 1 ] .$
9: // User-History Level Masking
10: Mask tokens in $s _ { H }$ with probability $\tau _ { 1 } .$
11: Extract item-level hidden states e from the last Transformer layer.
12: Compute dynamic attention scores $A _ { i j }$ using e.
13: Compute structural recoverability weights $\bar { W } _ { i j } ^ { s t r u c t }$ based on geometric decay $\mathit { \Delta } \mathcal { P }$ (Eq. 12).
14: Calculate context-adaptive weights $W _ { i j } = W _ { i j } ^ { s i r u c t } + \alpha A _ { i j } .$
15: Compute reweighted history loss $\mathcal { L } _ { 1 }$ (Eq. 17).
16: // Next-Item Level Masking
17: Mask tokens in target item ${ \bf x } _ { n }$ with probability $\tau _ { 2 } .$
18: Compute target item prediction loss $\mathcal { L } _ { 2 }$ given unmasked $s _ { H }$ (Eq. 18).
19: // Parameter Update
20: Compute total loss $\begin{array} { r } { \mathscr { L } = \lambda \mathscr { L } _ { 1 } + \mathscr { L } _ { 2 } . } \end{array}$
21: Update network parameters � via gradient descent.
22: end while
```

## 5 Experiments

## 5.1 Experimental Setup

Manuscript submitted to ACM

Algorithm 2 Inference Process with Constrained Decoding   
Require: User history SIDs $S _ { H } ,$ , Trained model ${ \mathit { p } } _ { \theta } ,$ Beam size �, Difusion steps �.   
Ensure: Recommended target item $i ^ { * } .$   
1: Initialize candidate target item $\mathbf { x } _ { n } ^ { T }$ as fully [MASK] tokens.   
2: Initialize beam with sequence $\mathbf { x } _ { n } ^ { T }$ and cumulative log-likelihood 0.   
3: Initialize valid candidate item set $C ^ { T }$   
4: for step $t = T$ to 1 do   
5: Predict: Pass $[ S _ { H } ; \mathbf { x } _ { n } ^ { t } ]$ into $\hbar \theta$ to get token distributions .   
6: for each sequence in beam do   
7: Identify the position � to unmask.   
8: Constrain: Filter invalid tokens by updating valid token set $X _ { m } ^ { t }$ based on $C ^ { t } .$   
9: Expand sequence with top probability valid tokens.   
10: end for   
11: Prune: Retain top-� sequences based on cumulative log-likelihood.   
12: Update Valid Set: $C ^ { t - 1 }$ ← items consistent with newly unmasked tokens.   
13: end for   
14: Return item $i ^ { * }$ corresponding to the top-1 generated valid SID sequence.

Table 1. Statistics of the used datasets. “Avg.Len” indicates the average number of interactions per user.
<table><tr><td>Dataset</td><td>#Users</td><td>#Items</td><td>#Interactions</td><td>Sparsity</td><td>Avg.Len</td></tr><tr><td>Scientific</td><td>50,985</td><td>25,848</td><td>412,947</td><td>99.969%</td><td>8.10</td></tr><tr><td>Instrument</td><td>57,439</td><td>24,587</td><td>511,836</td><td>99.964%</td><td>8.91</td></tr><tr><td>Game</td><td>94,762</td><td>25,612</td><td>814,586</td><td>99.966%</td><td>8.60</td></tr><tr><td>Steam</td><td>39,795</td><td>9,265</td><td>2,949,605</td><td>99.200%</td><td>74.12</td></tr><tr><td>MovieLens</td><td>6,040</td><td>3,883</td><td>1,001,456</td><td>95.730%</td><td>165.57</td></tr></table>

5.1.1 Datasets. We evaluate our method on five real-world datasets across diferent domains. From the Amazon collection [16], we select three subsets: Scientific, Musical Instruments, and Video Games. In addition to these productbased datasets, we include MovieLens [10], which contains user ratings alongside movie genres and titles, and Steam [32], a dataset comprising user-item interactions and textual reviews from the Steam gaming platform. These datasets vary in scale and domain, providing a comprehensive benchmark for recommendations. We follow the standard data split as in previous generative recommendation works [15, 33], where each user’s historical reviews are treated as interaction records and arranged in chronological order, with the earliest review appearing first. The detailed statistics of these datasets are summarized in Table 1. We evaluate our model using the widely adopted leave-one-out protocol [35]. Formally, for a sequence of � interactions, we reserve the �-th item for final performance evaluation and the (� − 1)-th item for validation. All prior interactions from 1 to � − 2 are used to optimize the model.

5.1.2 Evaluation Metrics. To quantify the recommendation quality, we employ two standard ranking-based metrics consistently used in prior literature [35]: Recall@� and Normalized Discounted Cumulative Gain (NDCG@�). In our experiments, we report the performance across � ∈ {1, 5, 10}. Note that NDCG@1 yields the same numerical value as Recall@1.

5.1.3 Baselines. To comprehensively evaluate our proposed model, we compare it against 16 representative baselines. Based on their item representation strategies, we categorize these models into two main groups: traditional Item ID-based methods and recent Semantic ID-based generative methods.

Manuscript submitted to ACM

Item ID-based Baselines. This group treats items as atomic IDs. The selected baselines encompass both classical sequential models and generative approaches:

• GRU4Rec [13] models chronological user action sequences using Gated Recurrent Units.

• SASRec [20] captures sequential dynamics through a unidirectional, left-to-right Transformer architecture with self-attention.

• BERT4Rec [37] utilizes a bidirectional Transformer trained via a Cloze objective to predict masked item IDs.

• FMLP-Rec [54] achieves eficient sequence encoding by replacing standard self-attention mechanisms with learnable filter-enhanced MLPs.

• LRURec [50] employs linear recurrent units to maintain linear-time training and inference complexity while overcoming the non-linearity limitations of traditional linear transitions.

• DreamRec [47] and DifuRec [23] formulate recommendation as a continuous denoising generation process by representing item IDs as continuous embeddings.

• PreferGrow [18] is a discrete difusion-based model that directly captures ranking dynamics by fading and growing user preference ratios over the discrete item corpus.

Semantic ID-based Baselines. This group represents items as Semantic IDs to enable generative recommendation. The evaluated models include:

• VQ-Rec [15] facilitates sequence-to-sequence recommendation by deriving discrete item codes from textual representations via Vector Quantization.

• TIGER [33] constructs tuple-based semantic IDs using RQ-VAE and predicts the next item autoregressively.

• TIGER-SAS [33] integrates the semantic IDs generated by TIGER into the standard SASRec architecture.

• LETTER [39] and LC-Rec [53] enhance generative recommendation through learnable tokenizers, allowing for the joint optimization of the codebook and the downstream recommendation objective.

• RPG [17] introduces a lightweight recommendation model to parallelly generate long, unordered semantic IDs.

• DifGRM [28] is a difusion-based model that replaces the traditional autoregressive decoder with a masked discrete difusion framework.

• LLaDA-Rec [35] serves as a state-of-the-art discrete difusion model, generating SIDs non-autoregressively through a mask-and-predict denoising framework

5.1.4 Implementation Details. SID Generation. Each item is represented by concatenating textual and collaborative embeddings. Specifically, we leverage a pre-trained Sentence-T5 model [30] to extract semantic features from item metadata (e.g., titles and descriptions). Concurrently, to capture structural interaction patterns, we build an item co-occurrence matrix and apply SVD to derive a 64-dimensional collaborative embedding for each item. The combined embeddings are then discretized into SIDs using RQ-KMeans with � = 4 levels and a codebook size of � = 256 per level.

Discrete Difusion Model. The discrete difusion model is configured as a bidirectional Transformer encoder (8 attention heads, 256-dimensional hidden states). We deploy a 4-layer architecture for the Scientific, Instrument, Steam, and MovieLens while extending it to 6 layers for the Game dataset to accommodate its larger scale. The scaling factor � in Eq. 19 is determined through a grid search over {1, 2, 3, 4, 5}. � is determined to be 3, 5, 2, 3, and 3 for the Scientific, Instrument, Game, MovieLens, and Steam datasets, respectively. The symmetric-geometric noise rescheduling sharpness � is determined through a grid search over {0.1, 0.5, 0.9}. We train the Transformer encoder for 150 epochs with an Manuscript submitted to ACM

Table 2. Performance comparison on three Amazon datasets. R@� and N@� denote Recall@� and NDCG@�, respectively. The best results are highlighted in bold, and the second-best results are underlined. The improvements of our model over the best baseline are statistically significant (paired t-test, � < 0.05).
<table><tr><td rowspan="2">Category</td><td rowspan="2">Model</td><td colspan="5">Scientific</td><td colspan="5">Instrument</td><td colspan="5">Game</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>N@5</td><td>N@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>N@5</td><td>N@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>N@5</td><td>N@10</td></tr><tr><td rowspan="9">Item ID based</td><td>GRU4Rec [2015]</td><td>0.0071</td><td>0.0184</td><td>0.0272</td><td>0.0128</td><td>0.0156</td><td>0.0094</td><td>0.0297</td><td>0.0453</td><td>0.0196</td><td>0.0246</td><td>0.0149</td><td>0.0461</td><td>0.0712</td><td>0.0307</td><td>0.0387</td></tr><tr><td>SASRec [2018]</td><td>0.0063</td><td>0.0240</td><td>0.0379</td><td>0.0152</td><td>0.0197</td><td>0.0089</td><td>0.0331</td><td>0.0525</td><td>0.0211</td><td>0.0273</td><td>0.0128</td><td>0.0516</td><td>0.0823</td><td>0.0323</td><td>0.0421</td></tr><tr><td>BERT4Rec [2019]</td><td>0.0045</td><td>0.0157</td><td>0.0264</td><td>0.0100</td><td>0.0134</td><td>0.0065</td><td>0.0255</td><td>0.0412</td><td>0.0160</td><td>0.0211</td><td>0.0082</td><td>0.0315</td><td>0.0530</td><td>0.0199</td><td>0.0267</td></tr><tr><td>FMLP-Rec [2022]</td><td>0.0046</td><td>0.0181</td><td>0.0300</td><td>0.0113</td><td>0.0151</td><td>0.0086</td><td>0.0299</td><td>0.0496</td><td>0.0193</td><td>0.0257</td><td>0.0099</td><td>0.0395</td><td>0.0649</td><td>0.0246</td><td>0.0328</td></tr><tr><td>LRURec [2024]</td><td>0.0049</td><td>0.0169</td><td>0.0267</td><td>0.0110</td><td>0.0141</td><td>0.0071</td><td>0.0272</td><td>0.0431</td><td>0.0172</td><td>0.0223</td><td>0.0134</td><td>0.0480</td><td>0.0753</td><td>0.0308</td><td>0.0396</td></tr><tr><td>DreamRec [2023a]</td><td>0.0052</td><td>0.0184</td><td>0.0299</td><td>0.0118</td><td>0.0155</td><td>0.0069</td><td>0.0245</td><td>0.0423</td><td>0.0157</td><td>0.0214</td><td>0.0125</td><td>0.0381</td><td>0.0611</td><td>0.0253</td><td>0.0326</td></tr><tr><td>DiffuRec [2023]</td><td>0.0050</td><td>0.0190</td><td>0.0310</td><td>0.0119</td><td>0.0158</td><td>0.0077</td><td>0.0283</td><td>0.0465</td><td>0.0179</td><td>0.0237</td><td>0.0111</td><td>0.0425</td><td>0.0709</td><td>0.0268</td><td>0.0359</td></tr><tr><td>PreferGrow [2025]</td><td>0.0065</td><td>0.0172</td><td>0.0283</td><td>0.0101</td><td>0.0125</td><td>0.0094</td><td>0.0265</td><td>0.0397</td><td>0.0181</td><td>0.0224</td><td>0.0180</td><td>0.0521</td><td>0.0784</td><td>0.0351</td><td>0.0436</td></tr><tr><td>VQ-Rec [2023]</td><td>0.0076</td><td>0.0248</td><td>0.0385</td><td>0.0162</td><td>0.0206</td><td>0.0099</td><td>0.0345</td><td>0.0532</td><td>0.0222</td><td>0.0282</td><td>0.0150</td><td>0.0497</td><td>0.0769</td><td>0.0325</td><td>0.0412</td></tr><tr><td rowspan="9">Semantic ID based</td><td>TIGER [2023]</td><td>0.0084</td><td>0.0282</td><td>0.0446</td><td>0.0183</td><td>0.0236</td><td>0.0105</td><td>0.0359</td><td>0.0566</td><td>0.0233</td><td>0.0300</td><td>0.0166</td><td>0.0529</td><td>0.0823</td><td>0.0348</td><td>0.0442</td></tr><tr><td>TIGER-SAS [2023]</td><td>0.0067</td><td>0.0221</td><td>0.0356</td><td>0.0144</td><td>0.0187</td><td>0.0102</td><td>0.0342</td><td>0.0521</td><td>0.0223</td><td>0.0280</td><td>0.0170</td><td>0.0548</td><td>0.0847</td><td>0.0360</td><td>0.0457</td></tr><tr><td>LETTER [2024a]</td><td>0.0082</td><td>0.0273</td><td>0.0423</td><td>0.0179</td><td>0.0227</td><td>0.0114</td><td>0.0362</td><td>0.0562</td><td>0.0239</td><td>0.0303</td><td>0.0169</td><td>0.0552</td><td>0.0863</td><td>0.0362</td><td>0.0462</td></tr><tr><td>LC-Rec [2024]</td><td>0.0091</td><td>0.0280</td><td>0.0434</td><td>0.0186</td><td>0.0235</td><td>0.0119</td><td>0.0379</td><td>0.0587</td><td>0.0251</td><td>0.0318</td><td>0.0165</td><td>0.0567</td><td>0.0891</td><td>0.0366</td><td>0.0471</td></tr><tr><td>RPG [2025]</td><td>0.0087</td><td>0.0257</td><td>0.0395</td><td>0.0174</td><td>0.0218</td><td>0.0118</td><td>0.0362</td><td>0.0545</td><td>0.0241</td><td>0.0300</td><td>0.0209</td><td>0.0579</td><td>0.0853</td><td>0.0397</td><td>0.0485</td></tr><tr><td>DiffGRM [2026b]</td><td>0.0092</td><td>0.0279</td><td>0.0441</td><td>0.0189</td><td>0.0232</td><td>0.0125</td><td>0.0382</td><td>0.0593</td><td>0.0255</td><td>0.0316</td><td>0.0205</td><td>0.0602</td><td>0.0925</td><td>0.0401</td><td>0.0483</td></tr><tr><td>LLaDA-Rec [2025]</td><td>0.0098</td><td>0.0310</td><td>0.0474</td><td>0.0203</td><td>0.0256</td><td>0.0128</td><td>0.0406</td><td>0.0623</td><td>0.0268</td><td>0.0337</td><td>0.0207</td><td>0.0623</td><td>0.0942</td><td>0.0415</td><td>0.0517</td></tr><tr><td>ANR-DiffRec</td><td>0.0122</td><td>0.0327</td><td>0.0494</td><td>0.0226</td><td>0.0279</td><td>0.0135</td><td>0.0410</td><td>0.0629</td><td>0.0273</td><td>0.0343</td><td>0.0214</td><td>0.0653</td><td>0.0987</td><td>0.0434</td><td>0.0542</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

early stopping mechanism. The AdamW [29] optimizer is used with a batch size of 1024, where the learning rate and weight decay are fine-tuned within {0.005, 0.003, 0.001} and {0.05, 0.005, 0.001}, respectively.

Our framework is implemented in PyTorch and executed on a workstation featuring the Intel Xeon Gold 6226R CPU and two NVIDIA RTX 4090 GPUs.

## 5.2 Main Results

5.2.1 Overall Performance. The experimental results on three Amazon datasets are summarized in Table 2. Additionally, to further validate the generalization of our method across diferent domains, we conduct experiments on the MovieLens and Steam datasets, as presented in Table 3. For these two datasets, we benchmark our method against a selective subset of the most competitive generative state-of-the-art models (i.e., RPG and LLaDA-Rec) alongside recently proposed strong baselines (PreferGrow and DifGRM). By comparing our proposed model with various baselines, we can draw the following conclusions:

• Superiority of ANR-DifRec: Our model consistently achieves the best performance across all datasets and evaluation metrics. Notably, ANR-DifRec significantly outperforms the strongest generative baseline based on the discrete difusion model, LLaDA-Rec, across most metrics. For example, on the Scientific dataset, ANR-DifRec delivers a substantial improvement in Recall@1. This performance gain validates the efectiveness of our noise rescheduling strategy, which enables the model to capture item-based collaborative filtering signals more precisely.

• Semantic ID vs. Item ID: SID-based generative methods (e.g., VQ-Rec, TIGER, and RPG) consistently outperform traditional item ID-based models such as SASRec and BERT4Rec across multiple benchmarks. This suggests that Semantic IDs provide a more informative representation space than independent atomic item IDs. Furthermore, incorporating collaborative structural signals via our item co-occurrence modeling further improves the ability of SIDs to capture item relationships and sequential transition patterns.

Table 3. Performance comparison on MovieLens and Steam. The best results are highlighted in bold, and the second-best results are underlined. The improvements of our model over the best baseline are statistically significant (paired t-test, � < 0.05).
<table><tr><td>Dataset</td><td>Model</td><td>NDCG@5</td><td>NDCG@10</td><td>Recall@5</td><td>Recall@10</td></tr><tr><td rowspan="5">MovieLens</td><td>PreferGrow [2025]</td><td>0.0910</td><td>0.1177</td><td>0.1409</td><td>0.2237</td></tr><tr><td>RPG [2025]</td><td>0.1197</td><td>0.1440</td><td>0.1750</td><td>0.2503</td></tr><tr><td>DiffGRM [2026b]</td><td>0.1038</td><td>0.1260</td><td>0.1541</td><td>0.2235</td></tr><tr><td>LLaDA-Rec [2025]</td><td>0.1359</td><td>0.1628</td><td>0.1980</td><td>0.2820</td></tr><tr><td>ANR-DiffRec</td><td>0.1468</td><td>0.1738</td><td>0.2141</td><td>0.2977</td></tr><tr><td rowspan="5">Steam</td><td>PreferGrow [2025]</td><td>0.0406</td><td>0.0508</td><td>0.0615</td><td>0.0935</td></tr><tr><td>RPG [2025]</td><td>0.0388</td><td>0.0490</td><td>0.0579</td><td>0.0898</td></tr><tr><td>DiffGRM [2026b]</td><td>0.0378</td><td>0.0479</td><td>0.0565</td><td>0.0870</td></tr><tr><td>LLaDA-Rec [2025]</td><td>0.0391</td><td>0.0502</td><td>0.0607</td><td>0.0914</td></tr><tr><td>ANR-DiffRec</td><td>0.0411</td><td>0.0526</td><td>0.0627</td><td>0.0991</td></tr></table>

• Efectiveness of Discrete Difusion: Among SID-based approaches, DDM-based models (LLaDA-Rec, DifGRM and ANR-DifRec) demonstrate superior ranking capabilities compared to autoregressive-based generators. This suggests that the bidirectional attention is more robust for recommendation tasks. Our model further improves upon this by adaptively rescheduling the noise level based on the geometric properties of the context, improving item retrieval accuracy.

5.2.2 Performance on Sequence Length. To investigate the robustness of our model against varying amounts of user historical information, we categorize the users into four groups based on their interaction sequence lengths: short (< 5), medium (5 ∼ 10), long (10 ∼ 20), and extremely long (> 20). We evaluate NDCG@5 and Recall@5 of ANR-DifRec alongside two strong baselines: the autoregressive state-of-the-art RPG, and the difusion-based LLaDA-Rec.

As illustrated in Figure 3, ANR-DifRec consistently outperforms the baselines across all length groups. Notably, RPG experiences a performance drop on extremely long sequences (> 20). This degradation could be partially attributed to the unidirectional bias, where the AR model over-emphasizes recent items while forgetting early context. In contrast, difusion-based models exhibit better resilience due to their bidirectional attention. Furthermore, ANR-DifRec surpasses LLaDA-Rec on short sequences (< 5). Interestingly, the short-sequence group tends to achieve relatively strong results across diferent models, suggesting that these users may have more concentrated and less noisy preferences rather than representing purely cold-start cases. This is reasonable for Amazon datasets, where the average sequence length is only around 8, making the < 5 group still close to the dominant interaction range.

5.2.3 Performance across Item Popularity. Popularity imbalance remains a fundamental challenge in recommendation systems, as user interactions are typically concentrated on a small subset of highly frequent items while many others sufer from limited exposure. To better understand how our method behaves under diferent popularity regimes, we conduct a fine-grained analysis on the Scientific and Instrument dataset by partitioning items according to their interaction frequency.

Following standard practices, we sort all items based on their interaction frequencies in the training set and divide them into three disjoint groups: Head (the top 20% most popular items), Mid (the next 60% items), and Tail (the bottom 20% infrequent items). We then compute NDCG@5 and Recall@5 for interactions in the test set where the target item belongs to these respective groups. We compare ANR-DifRec with the strongest autoregressive model RPG, and the discrete difusion model LLaDA-Rec.

As illustrated in Figure 4, ANR-DifRec consistently outperforms LLaDA-Rec across head, mid, and tail item groups, with the clearest gains appearing on mid-frequency items. This suggests that our method is particularly efective at Manuscript submitted to ACM

![](images/d3495db7c043a0f4d255c0eb49cdacbc03f3f9d5f2319fb0f1287c8c92506bfd.jpg)

Fig. 3. Performance comparison across diferent user interaction sequence length groups on the Scientific and Instrument dataset.  
![](images/447198a01b2d4d3b099d7a1c79657752d9e7f49b3552268551985f7fa3a000fb.jpg)  
Fig. 4. Performance comparison across Head, Mid, and Tail item groups on the Scientific and Instrument dataset.

modeling items with moderate interaction frequency, where both collaborative dependency and transition diversity are important. For tail items, the improvements are relatively smaller, since extremely sparse items still provide limited behavioral evidence for reliable generation. Nevertheless, on the Instruments dataset, ANR-DifRec still achieves better tail performance than LLaDA-Rec, indicating that our item co-occurrence-guided SID construction may help alleviate the representation isolation issue of infrequent items. While RPG maintains a marginal lead in the extreme tail of Instruments, this is likely due to its longer semantic IDs over-specializing to rare patterns. In contrast, ANR-DifRec prioritizes a more balanced representation, ensuring robust generalization across the entire popularity spectrum rather than over-fitting to the extreme tail

## 5.3 Ablation Study

We conduct extensive ablation studies to analyze the contribution of each component in our framework, as summarized in Table 4. We compare our model against the following variants: (1) w/o Schedule, which replaces our item-based noise rescheduling strategy with a standard uniform noise schedule; (2) w/o II, which removes the SVD-based collaborative features, relying solely on text semantics in SID generation; (3) w/o Schedule and II, which simultaneously removes both the adaptive noise scheduling mechanism and the collaborative structural features used in SID construction; and (4) RQ-VAE, which replaces our RQ-KMeans tokenizer with RQ-VAE. As shown in Table 4, we observe:

• Impact of item-based adaptive noise rescheduling: w/o Schedule leads to the most significant performance drop across all datasets (e.g., NDCG@1 drops by over 15% on Scientific). This confirms that our noise rescheduling is the core engine of ANR-DifRec, enabling the model to adaptively learn item-based collaborative filtering signals that uniform noise scheduling cannot capture.

• Benefit of item co-occurrence-guided SID generation: The performance decline in w/o II highlights the necessity of fusing item co-occurrence information. While semantic features provide content understanding, the collaborative signals are crucial for capturing the structural behavior of users.

• Superiority of RQ-KMeans: Our RQ-KMeans tokenizer (Full) consistently outperforms the RQ-VAE variants. This may be partly because clustering-based discretization avoids the instability that can occur when training neural tokenizers from scratch.

• Synergy of two components: The performance drops to its lowest point when both modules are removed (w/o Schedule and II), indicating that the two contributions are mutually reinforcing rather than redundant.

5.3.1 Efect ofNoise Rescheduling. To further investigate the efectiveness of the proposed adaptive noise rescheduling mechanism, we compare ANR-DifRec with several variants in Figure 5. The evaluated variants include: (1) w/o LRS, which removes the geometric contextual recoverability modeling and only retains behavior-aware dependency modeling; (2) w/o BDM, which removes the adaptive item-level dependency modeling and only preserves the local recoverability estimation; (3) w/o Inter-Item/Intra-Item, which separately removes inter-item or intra-item contextual dependencies from the local recoverability estimation module.

From the results, removing local recoverability estimation causes the most significant performance degradation, indicating that contextual recoverability serves as the primary source of denoising guidance in our adaptive rescheduling framework. By explicitly modeling the availability of surrounding unmasked contexts, the model can better distinguish easy-to-recover tokens from highly uncertain ones, thereby allocating denoising supervision more efectively. Without this mechanism, the difusion process degenerates toward a nearly uniform denoising strategy, substantially weakening the ability to exploit structural collaborative signals.

Manuscript submitted to ACM

Table 4. Ablation study results on three Amazon datasets. “II” denotes item co-occurrence information, and “Schedule” denotes the difusion adaptive noise rescheduling strategy.
<table><tr><td>Dataset</td><td>Method</td><td>NDCG@1</td><td>NDCG@5</td><td>NDCG@10</td><td>Recall@5</td><td>Recall@10</td></tr><tr><td rowspan="5">Scientific</td><td>RQ-VAE</td><td>0.0120</td><td>0.0221</td><td>0.0269</td><td>0.0320</td><td>0.0487</td></tr><tr><td>w/o ⅡI</td><td>0.0109</td><td>0.0217</td><td>0.0268</td><td>0.0314</td><td>0.0479</td></tr><tr><td>w/o Schedule</td><td>0.0099</td><td>0.0205</td><td>0.0259</td><td>0.0312</td><td>0.0477</td></tr><tr><td>w/o Schedule and II</td><td>0.0097</td><td>0.0203</td><td>0.0253</td><td>0.0310</td><td>0.0474</td></tr><tr><td>Full</td><td>0.0122</td><td>0.0226</td><td>0.0279</td><td>0.0327</td><td>0.0494</td></tr><tr><td rowspan="5">Instrument</td><td>RQ-VAE</td><td>0.0128</td><td>0.0270</td><td>0.0341</td><td>0.0408</td><td>0.0626</td></tr><tr><td>w/oⅡI</td><td>0.0130</td><td>0.0265</td><td>0.0339</td><td>0.0403</td><td>0.0627</td></tr><tr><td>w/o Schedule</td><td>0.0127</td><td>0.0259</td><td>0.0331</td><td>0.0401</td><td>0.0612</td></tr><tr><td>w/o Schedule and II</td><td>0.0127</td><td>0.0258</td><td>0.0329</td><td>0.0391</td><td>0.0610</td></tr><tr><td>Full</td><td>0.0135</td><td>0.0273</td><td>0.0343</td><td>0.0410</td><td>0.0629</td></tr><tr><td rowspan="5">Game</td><td>RQ-VAE</td><td>0.0208</td><td>0.0426</td><td>0.0532</td><td>0.0641</td><td>0.0973</td></tr><tr><td>w/oⅡI</td><td>0.0209</td><td>0.0423</td><td>0.0530</td><td>0.0633</td><td>0.0964</td></tr><tr><td>w/o Schedule</td><td>0.0206</td><td>0.0413</td><td>0.0523</td><td>0.0628</td><td>0.0957</td></tr><tr><td>w/o Schedule and II</td><td>0.0205</td><td>0.0409</td><td>0.0514</td><td>0.0610</td><td>0.0936</td></tr><tr><td>Full</td><td>0.0214</td><td>0.0434</td><td>0.0542</td><td>0.0653</td><td>0.0987</td></tr></table>

![](images/0961bad2743eeffdf60ab4fce74f3c585aca68dd2b6475c47455f9d602e3936c.jpg)  
Fig. 5. Ablation study on diferent noise rescheduling strategies.

Removing behavior-aware dependency modeling also leads to a noticeable performance drop, although the degradation is smaller than removing local recoverability estimation. This suggests that adaptive item-level dependency modeling provides important complementary information beyond static structural priors. In particular, the behavior-aware dependency module enables the model to dynamically capture user-specific collaborative patterns, helping the denoising process focus on semantically correlated items within diferent behavioral contexts.

We further analyze the contributions of inter-item and intra-item dependencies within local recoverability estimation. Removing inter-item contextual modeling results in a significantly larger decline than removing intra-item dependencies, demonstrating that collaborative relationships across items constitute the dominant source of contextual recoverability in recommendation sequences. Since user behaviors are inherently driven by sequential item interactions, neighboring and correlated items provide strong structural cues for recovering masked tokens. In contrast, intra-item dependencies mainly refine the semantic consistency within each semantic ID, yielding a relatively smaller but still consistent contribution.

![](images/b38108c69239efb06a050f2d6ffacceff6981dabe08aaec1f7cc602a2f7b6ec3.jpg)  
Fig. 6. Ablation study on diferent forms of collaborative signal construction.

5.3.2 Efect of Collaborative Signals. To evaluate how diferent forms of collaborative signal construction afect RQ-KMeans quantization, we compare several representative strategies, including SVD-based matrix factorization, Node2Vec [9], and a graph-based LightGCN [11] encoder.

In Figure 6, when introducing collaborative signals, both SVD-based and LightGCN-based approaches achieve clear improvements over the w/o II across datasets. However, their performance is largely comparable on both Scientific and Instrument, suggesting that first-order co-occurrence signals already provide a strong inductive bias for constructing SID representations, while higher-order propagation does not always translate into additional gains under our discrete difusion framework. In contrast, Node2Vec performs noticeably worse than both SVD and LightGCN. We attribute this to its reliance on homogeneous random walks, which tend to dilute fine-grained user-item interaction structure and are less aligned with the bipartite nature of recommendation data. Overall, these results indicate that incorporating explici collaborative structure is beneficial, while the specific choice between SVD and LightGCN has limited impact in our setting. This suggests that the efectiveness of SID construction is primarily driven by the presence of collaborative information itself rather than the complexity of the encoder used to extract it.

## 5.4 Hyper-parameter Analysis

We investigate the impact of two critical hyperparameters on model performance: the dimensionality of collaborative embeddings and the sharpness of the geometric distribution used in our noise rescheduling.

5.4.1 Impact of Collaborative Embedding Dimension. We first vary the dimension of the SVD-decomposed item cooccurrence features within {32, 64, 128, 256}. As illustrated in Figure 7, the recommendation performance initially improves as the dimension increases from 32 to 64. This indicates that a suficient embedding size is necessary to encode the complex structural dependencies between items. However, further increasing the dimension to 128 or 256 yields negligible gains or even slight performance degradation. This suggests that 64 dimensions ofer an optimal balance between representation capacity and noise suppression, efectively capturing collaborative signals without overfitting to sparse co-occurrence patterns. Consequently, we set the dimension to 64.

5.4.2 Impact of Sharpness in Noise Rescheduling. Additionally, we perform a sensitivity analysis on the sharpness parameter � within the geometric distribution employed for our noise rescheduling, varying it across {0.1, 0.5, 0.9}. As shown in Figure 8, the model exhibits robust performance across diferent settings. Based on these observations, we set � = 0.1 for all datasets.

Manuscript submitted to ACM

Adaptive Item-based Collaborative Structures via Noise Rescheduling in Difusion for Generative Recommendation19

![](images/bc3157e84a3b8c55f15a6b3a8f9b478071a457ca1685e584cd9d02cd1687d0a0.jpg)

![](images/621d0a69d06fe6e2ea82e9d95de62cbb3da803e83e9f0a830dc08aabdbac5dd5.jpg)  
Fig. 7. Hyper-parameter study with diferent dimensionalities for collaborative embedding.

![](images/e2079fcee643890254c9e6d52006a7d6308f7cbf1374ef6df4a58cb6445fe47f.jpg)

![](images/5c638c8a8b815d8ef337b973692a075dfd04dcdf86df1774c46ea618136f97f9.jpg)  
Fig. 8. Hyper-parameter study of the sharpness in the geometric distribution.

## 5.5 Eficiency Analysis

5.5.1 Complexity Analysis. We briefly analyze the computational complexity of the proposed ANR-DifRec , focusing on our primary contribution: the item-based adaptive noise rescheduling mechanism. For a user history of length � with SIDs of length �, the total sequence length is $M = \left( N - 1 \right) \times L$

The standard bidirectional Transformer encoder incurs a self-attention complexity of $O ( M ^ { 2 } d )$ , where � is the hidden state dimension. In our proposed rescheduling module, computing the structural geometric decay $w _ { i j } ^ { s t r u c t }$ requires $O ( M ^ { 2 } )$ operations. Concurrently, extracting the attention scores $A _ { i j }$ directly reuses the final hidden states from the Transformer, incurring $O ( M ^ { 2 } d )$ complexity. Therefore, the overall time complexity of ANR-DifRec remains strictly bounded by $O ( M ^ { 2 } d )$ . It operates in the same asymptotic complexity class as standard Transformer-based baselines (e.g., BERT4Rec, LLaDA-Rec), guaranteeing that our fine-grained denoising guidance is achieved without compromising training scalability.

5.5.2 Training Eficiency. We analyze the training cost from two aspects: SID generation and discrete difusion model optimization. As shown in Table 5, RQ-VAE-based SID generation requires expensive pretraining. In contrast, our item co-occurrence-guided SID generation incurs only a fraction of the computational cost, including the construction of the co-occurrence matrix, its SVD decomposition, and the RQ-KMeans clustering. For our model training, the proposed item-based adaptive noise rescheduling introduces marginal overhead (around 1%) compared to LLaDA-Rec [35].

Table 5. Training eficiency comparison in Scientific.
<table><tr><td>Stage</td><td>Method</td><td>Time/Epoch (s)</td><td>Total (h)</td><td>Cost</td></tr><tr><td>Item Tokenization</td><td>RQ-VAE Ours</td><td>0.648 一</td><td>1.799 0.106</td><td>1.000× 0.059×</td></tr><tr><td rowspan="3">Generative Training</td><td>LLaDA-Rec</td><td>87.402</td><td>3.641</td><td>1.000×</td></tr><tr><td>ANR-DiffRec</td><td>88.351</td><td>3.681</td><td>1.011×</td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

5.5.3 Inference Eficiency. We further analyze the eficiency of the proposed constrained difusion inference. Although the decoding process maintains a candidate item set to ensure valid SID generation, the efective token vocabulary shrinks rapidly at each difusion step. As shown in Table 6, We scale the beam size from 1 to 50 to evaluate the total inference time on the Scientific test set. While the total overhead for both models scales linearly with the beam size their execution times remain nearly identical across all configurations. Our method achieves a comparable per-user latency to LLaDA-Rec with Beam=50 (10.17 ms vs. 10.06 ms). This close match indicates that ANR-DifRec mitigates potential computational burdens, introducing negligible inference overhead compared to LLaDA-Rec while achieving superior recommendation performance.

Table 6. Inference eficiency comparison with diferent beam sizes in Scientific.
<table><tr><td>Method</td><td>Metric</td><td>Beam=1</td><td>Beam=10</td><td>Beam=20</td><td>Beam=30</td><td>Beam=40</td><td>Beam=50</td></tr><tr><td>LLaDA-Rec</td><td>Total Time (s) Time / User (ms)</td><td>0m 13s 0.25</td><td>1m 46s 2.08</td><td>3m 22s 3.96</td><td>5m 04s 5.96</td><td>6m 46s 7.96</td><td>8m 33s 10.06</td></tr><tr><td>ANR-DiffRec</td><td>Total Time (s) Time / User (ms)</td><td>0m 13s 0.25</td><td>1m 46s 2.08</td><td>3m 24s 4.00</td><td>5m 06s 6.00</td><td>6m 50s 8.04</td><td>8m 39s 10.18</td></tr></table>

## 5.6 Case Studies

To rigorously evaluate the efectiveness of incorporating item co-occurrence features in the SID generation, we examine three randomly sampled items (#710, #6278, and #219) with distinct characteristics. For each item, we retrieve the Top-20 nearest neighbors based on continuous embedding similarity and discrete SIDs overlap, respectively. We then verify how many of these neighbors appear in the ground-truth global co-occurrence matrix. As detailed in Table 7, our method yields consistent improvements across all cases. Embedding space refinement: after integrating co-occurrence features, the embedding space becomes more collaborative. For instance, in Item #710, the number of co-occurring neighbors in the top-20 list increases from 3 to 5. Significant SID refinement: the most notable gains are observed in the SID space. For instance, in Item #6278, the number of co-occurring neighbors found by SID retrieval tripled (from 4 to 12) after feature integration. This confirms that our quantizer, when guided by the co-occurrence feature, successfully encodes collaborative information into the SIDs. Boldface indicates the gain in co-occurring hits.

Table 7. Case study on three sampled items. We compare the number of ground-truth co-occurring items found within the Top-20 nearest neighbors in both the embedding space and the SID space. “Pre” and “Post” denote before and after integrating the item co-occurrence features.
<table><tr><td rowspan="2">Item</td><td rowspan="2">Metric Space</td><td colspan="3">Co-occur Hits@20</td><td rowspan="2">Co-occurring Neighbors (Post-integration)</td></tr><tr><td>Pre</td><td>Post</td><td>Gain (↑)</td></tr><tr><td rowspan="2">Item #710</td><td>Embedding</td><td>3</td><td>5 +2</td><td></td><td>{757, 732, 9894, 750, 1104}</td></tr><tr><td>SIDs</td><td>1</td><td>8</td><td>+7</td><td>{732, 719, 750, 1104, 755, 758, 751, 757}</td></tr><tr><td rowspan="2">Item #6278</td><td>Embedding SIDs</td><td>5</td><td>9</td><td>+4</td><td>{9430, 3671, 23939, 2229, 3021, 17400, 24692, 21922, 17350}</td></tr><tr><td></td><td>4</td><td>12</td><td>+8</td><td>{3671, 3717, 879, 2371, 2250, 2230, 1613, 2268, 1169, 998, 2229, 1774}</td></tr><tr><td rowspan="2">Item #219</td><td>Embedding</td><td>4</td><td>8</td><td>+4</td><td>{975, 971, 20665, 977, 214, 13070, 9644, 6499}</td></tr><tr><td>SIDs</td><td>3</td><td>7</td><td>+4</td><td>{220, 214, 516, 6499, 13148, 13070, 971}</td></tr></table>

Manuscript submitted to ACM

## 6 Conclusion

This paper presents ANR-DifRec, a novel generative recommendation framework that integrates item-based collaborative structures into discrete difusion models. The core of our contribution is the item-based adaptive noise rescheduling strategy, which provides structure-aware denoising guidance by adaptively adjusting the noise level according to intra-item and inter-item contexts. Complementary to this, we introduce an item co-occurrence-guided SID generation mechanism that injects item co-occurrence priors for difusion training. Empirical results on various benchmarks confirm the efectiveness of our approach, demonstrating that aligning the denoising process with the collaborative signals in recommendation data significantly improves generative performance. In future work, we aim to further advance generative recommendation models by integrating multimodal information.

## References

[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774 (2023).

[2] Marianne Arriola, Aaron Gokaslan, Justin T Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. 2025. Block difusion: Interpolating between autoregressive and difusion language models. arXiv preprint arXiv:2503.09573 (2025).

[3] Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. 2021. Structured denoising difusion models in discrete state-spaces. Advances in neural information processing systems 34 (2021), 17981–17993.

[4] Jiawei Chen, Hande Dong, Xiang Wang, Fuli Feng, Meng Wang, and Xiangnan He. 2020. Bias and debias in recommender system: a survey and future directions (2020). arXiv preprint arXiv:2010.03240 (2020).

[5] Zeyu Cui, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. 2022. M6-rec: Generative pretrained language models are open-ended recommender systems. arXiv preprint arXiv:2205.08084 (2022)

[6] Jiaxin Deng, Shiyao Wang, Kuo Cai, Lejian Ren, Qigen Hu, Weifeng Ding, Qiang Luo, and Guorui Zhou. 2025. Onerec: Unifying retrieve and rank with generative recommender and iterative preference alignment. arXiv preprint arXiv:2502.18965 (2025)

[7] Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge, and Yongfeng Zhang. 2022. Recommendation as language processing (rlp): A unified pretrain, personalized prompt & predict paradigm (p5). In Proceedings ofthe 16th ACM conference on recommender systems. 299–315.

[8] Marjan Ghazvininejad, Omer Levy, Yinhan Liu, and Luke Zettlemoyer. 2019. Mask-predict: Parallel decoding of conditional masked language models. arXiv preprint arXiv:1904.09324 (2019)

[9] Aditya Grover and Jure Leskovec. 2016. node2vec: Scalable feature learning for networks. In Proceedings ofthe 22nd ACM SIGKDD internationa conference on Knowledge discovery and data mining. 855–864.

[10] F Maxwell Harper and Joseph A Konstan. 2015. The movielens datasets: History and context. Acm transactions on interactive intelligent systems (tiis) 5, 4 (2015), 1–19.

[11] Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang. 2020. Lightgcn: Simplifying and powering graph convolution network for recommendation. In Proceedings ofthe 43rd International ACM SIGIR conference on research and development in Information Retrieval. 639–648.

[12] Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. 2017. Neural collaborative filtering. In Proceedings ofthe 26th international conference on world wide web. 173–182

[13] Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. 2015. Session-based recommendations with recurrent neural networks. arXiv preprint arXiv:1511.06939 (2015).

[14] Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising difusion probabilistic models. Advances in neural information processing systems 33 (2020), 6840–6851.

[15] Yupeng Hou, Zhankui He, Julian McAuley, and Wayne Xin Zhao. 2023. Learning vector-quantized item representation for transferable sequential recommenders. In Proceedings ofthe ACM Web Conference 2023. 1162–1171.

Manuscript submitted to ACM

[16] Yupeng Hou, Jiacheng Li, Zhankui He, An Yan, Xiusi Chen, and Julian McAuley. 2024. Bridging language and items for retrieval and recommendation arXiv preprint arXiv:2403.03952 (2024)

[17] Yupeng Hou, Jiacheng Li, Ashley Shin, Jinsung Jeon, Abhishek Santhanam, Wei Shao, Kaveh Hassani, Ning Yao, and Julian McAuley. 2025. Generating long semantic ids in parallel for recommendation. In Proceedings ofthe 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2. 956–966.

[18] Guoqing Hu, An Zhang Liu, Wenyu Mao, Jiancan Wu, Xun Yang, Xiang Li, Lantao Hu, Han Li, Kun Gai, Xiang Wang, et al. 2025. Fading to Grow: Growing Preference Ratios via Preference Fading Discrete Difusion for Recommendation. arXiv preprint arXiv:2509.26063 (2025).

[19] Wenyue Hua, Shuyuan Xu, Yingqiang Ge, and Yongfeng Zhang. 2023. How to index item ids for recommendation foundation models. In Proceedings ofthe Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region. 195–204.

[20] Wang-Cheng Kang and Julian McAuley. 2018. Self-attentive sequential recommendation. In 2018 IEEE international conference on data mining (ICDM). IEEE, 197–206.

[21] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. 2022. Autoregressive image generation using residual quantization. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 11523–11532

[22] Wuchao Li, Rui Huang, Haijun Zhao, Chi Liu, Kai Zheng, Qi Liu, Na Mou, Guorui Zhou, Defu Lian, Yang Song, et al. 2025. DimeRec: a unified framework for enhanced sequential recommendation via generative difusion models. In Proceedings ofthe Eighteenth ACM International Conference on Web Search and Data Mining. 726–734.

[23] Zihao Li, Aixin Sun, and Chenliang Li. 2023. Difurec: A difusion model for sequential recommendation. ACM Transactions on Information Systems 42, 3 (2023), 1–28.

[24] Xinyu Lin, Haihan Shi, Wenjie Wang, Fuli Feng, Qifan Wang, See-Kiong Ng, and Tat-Seng Chua. 2025. Order-agnostic identifier for large language model-based generative recommendation. In Proceedings ofthe 48th international ACM SIGIR conference on research and development in information retrieval. 1923–1933.

[25] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. 2024 Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437 (2024).

[26] Chengyi Liu, Xiao Chen, Shijie Wang, Wenqi Fan, and Qing Li. 2026. Continuous-time Discrete-space Difusion Model for Recommendation. In Proceedings ofthe Nineteenth ACM International Conference on Web Search and Data Mining. 406–415.

[27] Enze Liu, Bowen Zheng, Cheng Ling, Lantao Hu, Han Li, and Wayne Xin Zhao. 2025. Generative recommender with end-to-end learnable item tokenization. In Proceedings ofthe 48th International ACM SIGIR Conference on Research and Development in Information Retrieval. 729–739.

[28] Zhao Liu, Yichen Zhu, Yiqing Yang, Xiao Lv, Guoping Tang, Rui Huang, Qiang Luo, Ruiming Tang, and Guorui Zhou. 2026. Difgrm: Difusion-based generative recommendation model. In Proceedings ofthe ACM Web Conference 2026. 5853–5864.

[29] Ilya Loshchilov and Frank Hutter. 2017. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017)

[30] Jianmo Ni, Gustavo Hernandez Abrego, Noah Constant, Ji Ma, Keith Hall, Daniel Cer, and Yinfei Yang. 2022. Sentence-t5: Scalable sentence encoders from pre-trained text-to-text models. In Findings of the association for computational linguistics: ACL 2022. 1864–1874.

[31] Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. 2025. Large language difusion models. arXiv preprint arXiv:2502.09992 (2025).

[32] Apurva Pathak, Kshitiz Gupta, and Julian McAuley. 2017. Generating and personalizing bundle recommendations on steam. In Proceedings of the 40th international ACM SIGIR conference on research and development in information retrieval. 1073–1076

[33] Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Hulikal Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Tran, Jonah Samost, et al. 2023. Recommender systems with generative retrieval. Advances in Neural Information Processing Systems 36 (2023), 10299–10315.

[34] Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis Titsias. 2024. Simplified and generalized masked difusion for discrete data Advances in neural information processing systems 37 (2024), 103131–103167.

[35] Teng Shi, Chenglei Shen, Weijie Yu, Shen Nie, Chongxuan Li, Xiao Zhang, Ming He, Yan Han, and Jun Xu. 2025. LLaDA-Rec: Discrete Difusion for Parallel Semantic ID Generation in Generative Recommendation. arXiv preprint arXiv:2511.06254 (2025).

[36] Zihua Si, Zhongxiang Sun, Jiale Chen, Guozhang Chen, Xiaoxue Zang, Kai Zheng, Yang Song, Xiao Zhang, Jun Xu, and Kun Gai. 2024. Generative retrieval with semantic tree-structured identifiers and contrastive learning. In Proceedings ofthe 2024 Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region. 154–163.

[37] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer. In Proceedings ofthe 28th ACM international conference on information and knowledge management. 1441–1450

[38] Jiakai Tang, Sunhao Dai, Teng Shi, Jun Xu, Xu Chen, Wen Chen, Jian Wu, and Yuning Jiang. 2025. Think before recommend: Unleashing the latent reasoning power for sequential recommendation. arXiv preprint arXiv:2503.22675 (2025).

[39] Wenjie Wang, Honghui Bao, Xinyu Lin, Jizhi Zhang, Yongqi Li, Fuli Feng, See-Kiong Ng, and Tat-Seng Chua. 2024. Learnable item tokenization for generative recommendation. In Proceedings ofthe 33rd ACM International Conference on Information and Knowledge Management. 2400–2409.

[40] Wenjie Wang, Yiyan Xu, Fuli Feng, Xinyu Lin, Xiangnan He, and Tat-Seng Chua. 2023. Difusion recommender model. In Proceedings ofthe 46th international ACM SIGIR conference on research and development in information retrieval. 832–841.

[41] Ye Wang, Jiahao Xun, Minjie Hong, Jieming Zhu, Tao Jin, Wang Lin, Haoyuan Li, Linjun Li, Yan Xia, Zhou Zhao, et al. 2024. Eager: Two-stream generative recommender with behavior-semantic collaboration. In Proceedings ofthe 30th ACM SIGKDD Conference on Knowledge Discovery and Manuscript submitted to ACM

Data Mining. 3245–3254.

[42] Longtao Xiao, Haozhao Wang, Cheng Wang, Linfei Ji, Yifan Wang, Jieming Zhu, Zhenhua Dong, Rui Zhang, and Ruixuan Li. 2025. Unger: Generative recommendation with a unified code via semantic and collaborative integration. ACM Transactions on Information Systems 44, 2 (2025), 1–31.

[43] Zhihui Xie, Jiacheng Ye, Lin Zheng, Jiahui Gao, Jingwei Dong, Zirui Wu, Xueliang Zhao, Shansan Gong, Xin Jiang, Zhenguo Li, et al. 2025. Dream-coder 7b: An open difusion language model for code. arXiv preprint arXiv:2509.01142 (2025).

[44] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388 (2025).

[45] Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke Shen, Yunhai Tong, and Mengdi Wang. 2025. Mmada: Multimodal large difusion language models arXiv preprint arXiv:2505.15809 (2025).

[46] Ling Yang, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao, Wentao Zhang, Bin Cui, and Ming-Hsuan Yang. 2023. Difusion models: A comprehensive survey of methods and applications. ACM computing surveys 56, 4 (2023), 1–39.

[47] Zhengyi Yang, Jiancan Wu, Zhicai Wang, Xiang Wang, Yancheng Yuan, and Xiangnan He. 2023. Generate what you prefer: Reshaping sequential recommendation via guided difusion. Advances in Neural Information Processing Systems 36 (2023), 24247–24261.

[48] Jiacheng Ye, Shansan Gong, Jiahui Gao, Junming Fan, Shuang Wu, Wei Bi, Haoli Bai, Lifeng Shang, and Lingpeng Kong. 2025. Dream-VL & Dream-VLA: Open Vision-Language and Vision-Language-Action Models with Difusion Language Model Backbone. arXiv preprint arXiv:2512.22615 (2025).

[49] Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. 2025. Dream 7b: Difusion large language models. arXiv preprint arXiv:2508.15487 (2025).

[50] Zhenrui Yue, Yueqi Wang, Zhankui He, Huimin Zeng, Julian McAuley, and Dong Wang. 2024. Linear recurrent units for sequential recommendation. In Proceedings ofthe 17th ACM international conference on web search and data mining. 930–938.

[51] Jiaqi Zhai, Lucy Liao, Xing Liu, Yueming Wang, Rui Li, Xuan Cao, Leon Gao, Zhaojie Gong, Fangda Gu, Michael He, et al. 2024. Actions speak louder than words: Trillion-parameter sequential transducers for generative recommendations. arXiv preprint arXiv:2402.17152 (2024).

[52] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. 2023. A survey of large language models. arXiv preprint arXiv:2303.18223 1, 2 (2023)

[53] Bowen Zheng, Yupeng Hou, Hongyu Lu, Yu Chen, Wayne Xin Zhao, Ming Chen, and Ji-Rong Wen. 2024. Adapting large language models by integrating collaborative semantics for recommendation. In 2024 IEEE 40th International Conference on Data Engineering (ICDE). IEEE, 1435–1448

[54] Kun Zhou, Hui Yu, Wayne Xin Zhao, and Ji-Rong Wen. 2022. Filter-enhanced MLP is all you need for sequential recommendation. In Proceedings of the ACM web conference 2022. 2388–2399.

[55] Fengqi Zhu, Rongzhen Wang, Shen Nie, Xiaolu Zhang, Chunwei Wu, Jun Hu, Jun Zhou, Jianfei Chen, Yankai Lin, Ji-Rong Wen, et al. 2025. LLaDA 1.5: Variance-Reduced Preference Optimization for Large Language Difusion Models. arXiv preprint arXiv:2505.19223 (2025).