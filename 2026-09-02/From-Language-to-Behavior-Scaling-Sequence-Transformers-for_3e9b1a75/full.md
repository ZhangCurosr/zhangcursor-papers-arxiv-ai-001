# From Language to Behavior: Scaling Sequence Transformers for Industrial Recommendation Ranking with Rec-Native Designs

Jie Chen, Xiangqian Yu, Yanchao Lian, Tan Lu, Run Yang, Zhengchun Shang,

Xing Wang, Cheng Chen, Ke Hu, Qiang Li<sup>\*</sup>, Tianjiu Yin, Xiaobing Liu

ByteDance

{chenjie.996, yuxiangqian.0, lianyanchao, lutan, yangrun,austin.shang, xing.wang, chencheng.kit, xiaohe.00, huangtao.ranger, yintianjiu, will.liu}@bytedance.com

## ABSTRACT

Scaling Transformers has driven large gains in language modeling, but transplanting this to behavior-sequence modeling in production ranking is challenging: recommendation difers in signal quality, where behavior sequences are noisy, temporally irregular, and sparsely supervised, and in computation asymmetry, where each request scores many candidates against one shared user history under tight latency budgets. We propose ReST, a recommendation-native Transformer scaling framework. For signal quality, it introduces a sequence encoder with dual-gated attention, rotary positional and temporal embedding, stabilized residual normalization, and training-only auxiliary objectives. For computation asymmetry, it factorizes ranking into a heavy reusable encoder and a lightweight cross decoder with projection-free KV attention and token-specific parameterization, coupling user-level shared-prefix training with shared-prefix serving for compute-once, decode-many-times rank ing. Across industrial and public benchmarks, ReST achieves higher accuracy and scales more consistently along sequence length, depth, and width, where LLM-style Transformer blocks saturate. A oneweek online A/B test on a production advertising platform improves online AUC by 1.31% and lifts a core revenue metric by 11.93% within a 50 ms P99 budget; ReST has since been fully deployed in production, showing that behavior-sequence scaling remains a promising, under-exploited axis for production ranking.

## CCS CONCEPTS

• Information systems → Recommender systems.

## KEYWORDS

Recommender System, Large Recommendation Model, Scaling Laws

## 1 INTRODUCTION

Transformers have become the dominant scaling recipe in language modeling, where increasing model capacity, data, and training com pute yields substantial and consistent gains [13, 17, 29]. Recommendation ranking appears to ofer a similar opportunity, as user behavior sequences capture evolving intent beyond static user fea tures. This promise has motivated a long line of behavior-sequence models, from target-aware attention [41] and long-sequence retrieval [3, 23] to recent large recommendation models [37]. However, directly scaling Transformer sequence models for production discriminative ranking remains challenging, because recommenda tion difers from language modeling in two fundamental ways.

The first obstacle lies in signal quality. Unlike natural language, user behavior sequences exhibit three distinct characteristics that make naive scaling statistically ineficient. (i) Behavior reliability. Behavior tokens are implicit feedback: clicks and views may reflect stable preference, but also accidental exposure, exploratory browsing, or biased feedback [15, 32]. Standard self-attention aggregates them through a shared softmax without an explicit mechanism to distinguish reliable from unreliable behavioral evidence. (ii) Temporal structure. User behaviors occur in irregular physical time—adjacent actions may be seconds or days apart, so equal ordinal distances imply very diferent recency—and purely ordinal encoding is insuficient for multi-scale recency [19]. (iii) Supervision density. Decoder-only language models receive dense next-token supervision, whereas industrial ranking provides only sparse, delayed labels at the user and candidate level [4, 18]. This sparse discriminative supervision provides a less direct training signal to the sequence stack, and the dificulty is further amplified in hybrid rankers: the main discriminative loss can be shortcut through the strong non-sequential DLRM branch, leaving the sequence module weakly supervised even when the history carries useful intent—a sequence-starvation efect that worsens as the module scales. Consequently, naively scaling a vanilla Transformer yields diminishing returns [11].

The second obstacle lies in computation asymmetry in targetconditioned ranking. Unlike language modeling, a ranking request pairs one user history with � candidate targets [7, 37]. The two sides benefit from scaling diferently: sequence-side computation is shared across candidates, so investing more compute there can be amortized over all � targets; target-conditioned interaction, however, depends on each candidate and is instantiated � times per request, making per-candidate compute the latency bottleneck while still requiring enough capacity to discriminate fine-grained candidates. An LLM-style decoder-only Transformer places both roles in a single stack and couples computation with parameters, making it hard to serve these divergent needs separately. Scaling the sequence model under a production latency budget therefore requires breaking this symmetry.

We propose ReST, a Recommendation-native Scalable Transformer framework that redesigns the Transformer scaling recipe for behavior-sequence modeling in production ranking. To address the signal quality challenge, ReST introduces a recommendation-native sequence encoder block for behavior data, combining dual-gated attention, rotary positional and temporal embedding (RoPE + RoTE), stabilized residual normalization, and training-only auxiliary objectives that strengthen sequence-side supervision. To address computation asymmetry, ReST factorizes ranking into a compute-heavy sequence encoder �, which contextualizes each user history once and produces reusable memory, and a lightweight cross decoder �, which scores many candidates through projection-free key/value attention and token-specific query parameterization. We further co-design the training and serving stack around this boundary: userlevel shared-prefix training amortizes prefix computation across samples from the same user, while shared-prefix serving reuses encoder states across candidates within a request. Empirically, ReST restores consistent scaling along behavior sequence length, model depth, and hidden dimension, reduces redundant training and serving computation, and satisfies the production P99 latency budget in online deployment.

Our contributions are summarized as follows:

1. Rec-native Transformer for sequence scaling. We identify and empirically diagnose a sequence-starvation efect in hybrid production rankers. We develop a recommendation-native sequence encoder that combines dual-gated attention for noisy behaviors, rotary positional and temporal embedding for irregular temporal structure, and SRN for stable depth scaling, together with two training-only auxiliary objectives that mitigate sequence starvation.

2. Asymmetric architecture with system co-design. We exploit the one-history-to-many-candidates structure of production ranking by factorizing computation into a reusable, compute-heavy sequence encoder and a lightweight candidate-aware cross decoder. Projection-free key/value attention and token-specific parameterization preserve candidate-side capacity with low activated FLOPs, while user-level shared-prefix training and shared-prefix serving translate this architectural asymmetry into computation reuse.

3. Large-scale empirical validation and online deployment. Across a large-scale industrial advertising dataset and public bench marks, ReST achieves better accuracy–eficiency trade-ofs than LLM-style and recommendation baselines, and continues to scale in sequence length, depth, and width where LLM-style blocks saturate. A one-week online A/B test improves online AUC by 1.31% and lifts a core revenue metric by 11.93% within the 50 ms P99 latency budget, leading to full production deployment.

## 2 METHOD

## 2.1 Problem Formulation

We study conversion rate (CVR) prediction in large-scale advertising ranking, where industrial rankers combine chronological behavior sequences with non-sequential user/context features [22, 41]. Let U and A denote the user and ad sets. For a user � $\in { \mathcal { U } } ,$ we observe a behavior sequence $S _ { u } = [ ( \mathbf { s } _ { 1 } , t _ { 1 } ) , \dots , ( \mathbf { s } _ { L } , t _ { L } ) ]$ , where $\mathbf { s } _ { i }$ is the feature vector of the �-th interaction and $t _ { i }$ its physical timestamp, together with non-sequential features $\mathbf { u } _ { d } .$

In production ranking, these shared user-side inputs are paired with � candidate ads $\{ a _ { 1 } , \ldots , a _ { N } \} \subseteq { \mathcal { A } }$ . For each candidate $^ { a , }$ the ranker predicts

$$
\boldsymbol { \hat { y } } = f _ { \theta } ( S _ { u } , \mathbf { u } _ { d } , a ) \approx \boldsymbol { P } ( \boldsymbol { y } = 1 \mid S _ { u } , \mathbf { u } _ { d } , a ) ,\tag{1}
$$

where $y \in \{ 0 , 1 \}$ is the binary conversion label. The model is trained with the CVR loss $\mathcal { L } _ { \mathrm { c e } } = \mathbb { E } _ { \mathcal { D } } \big [ \ell _ { \mathrm { B C E } } ( \hat { y } , y ) \big ]$ , where $\ell _ { \mathrm { B C E } }$ is the standard binary cross-entropy over the training set D.

This work focuses on scaling the sequence modeling component under this discriminative formulation, while keeping the nonsequential ranking stack and candidate scoring protocol intact.

## 2.2 Overview

To address the signal quality and computation asymmetry challenges, we propose ReST, a recommendation-native scalable Transformer for behavior-sequence modeling in recommendation ranking (Figure 1). ReST adopts an asymmetric encoder–decoder architecture: a compute-heavy rec-native sequence Transformer encoder � builds robust, reusable memory once per user behavior history (Sec. 2.3), while a lightweight cross decoder � performs candidatespecific decoding from this memory (Sec. 2.4). We further introduce two training-only auxiliary objectives to mitigate sequence starvation (Sec. 2.5), together with user-level shared-prefix training and shared-prefix serving (Sec. 2.6), enabling compute-once, decodemany-times ranking under production latency budgets.

## 2.3 Rec-Native Sequence Encoder (� )

We start from a LLaMA-style [29] Transformer block with causal self-attention and SwiGLU [27] feed-forward layers, but adapt it to the distinct properties of recommendation behavior sequences. Naively scaling such blocks is hindered by noisy interactions, irregular temporal gaps, and unstable optimization as depth grows under sparse supervision. We therefore introduce three recommendationnative sequence modeling designs: Dual-Gated Attention for noisy behaviors, Rotary Positional and Temporal Embedding for ordinal and temporal structure, and Stabilized Residual Normalization for stable depth scaling.

Dual-Gated Attention (DGA). Implicit feedback in recommendation is inherently noisy [32, 41]: accidental clicks, weak intent, and exploratory actions may not faithfully reflect stable user preference. This noise afects attention in two stages. Before aggregation, an individual behavior token may be unreliable evidence; after aggregation, the resulting behavioral context may still be only weakly useful for updating the representation. We therefore design Dual Gated Attention (DGA) to make denoising explicit on both sides of the attention operator. A value gate performs pre-aggregation filtering over behavior features, while an output gate performs post-aggregation context modulation. Given an input sequence $\bar { \mathbf { X } } \in \mathbb { R } ^ { \bar { L } \times \bar { d } }$

$$
\mathbf { H } = \mathrm { A t t e n t i o n } \Big ( \mathbf { X } \mathbf { W } _ { Q } , \mathbf { X } \mathbf { W } _ { K } , \sigma ( \mathbf { X } \mathbf { W } _ { g v } ) \odot ( \mathbf { X } \mathbf { W } _ { V } ) \Big ) ,\tag{2}
$$

$$
\mathrm { D u a l G a t e d A t t n } ( \mathbf { X } ) = \bigl ( \mathbf { H } \odot \sigma ( \mathbf { X } \mathbf { W } _ { g o } ) \bigr ) \mathbf { W } _ { o } ,\tag{3}
$$

where $\mathrm { A t t e n t i o n } ( \mathbf { Q } , \mathbf { K } , \mathbf { V } ) = \mathrm { s o f t m a x } ( \mathbf { Q } \mathbf { K } ^ { \top } / \sqrt { d } + \mathbf { M } _ { \mathrm { c a u s a l } } ) \mathbf { V } , e$ � is the sigmoid, and ⊙ is element-wise multiplication. The value gate $\mathbf { W } _ { g v }$ attenuates unreliable interaction features before they contribute to any context, while the output gate $\mathbf { W } _ { g o }$ controls how much the aggregated behavioral context updates each token. Decoupling token-level reliability filtering from context-level modulation yields robustness to noisy histories without modifying the attention soft max, preserving compatibility with eficient attention kernels.

Building on gated-attention designs that primarily modulate the aggregated attention output [14, 25, 37], DGA additionally gates the value stream, suppressing noisy behavior features before they enter the aggregated context. HSTU uses an unbounded SiLU output gate [37]; in our setting, we observe that such gates can exhibit attention-output divergence at scale and therefore adopt bounded sigmoid gates for stable model scaling. Our ablation study shows that the value and output gates are complementary, and Appendix A.1 further validates the denoising efect of DGA.

![](images/ecdf74f2284c5d31c5001823e82f79f126fcf7ce8ea4de4109ef7d4e96f2e8a9.jpg)  
Figure 1: ReST architecture: a compute-heavy sequence encoder � and a lightweight cross decoder �.

Rotary Positional and Temporal Embedding (RoPE + RoTE). Language tokens are evenly spaced in an ordinal sequence. Modern LLMs commonly use RoPE [28] to encode relative ordinal positions through rotation, a formulation compatible with FlashAttentionstyle kernels [8]. User behaviors, however, live in irregular physical time, where user intent evolves with elapsed time: two clicks seconds apart and two clicks days apart should be treated diferently even when they occupy adjacent sequence positions. Prior time-aware recommenders encode time through relative interval embeddings, temporal biases, or bucketized features [6, 19, 37], which are less aligned with length extrapolation and rotary-friendly eficient attention. We therefore design a rotary temporal embedding that preserves the rotary formulation while supporting multigranularity time intervals by splitting attention heads into two complementary groups.

We propose Rotary Temporal Embedding (RoTE), which rotates queries and keys by the discretized physical timestamp under a head-specific time granularity $\tau _ { h } .$ . To jointly model ordinal order and physical time, we split attention heads into two groups: heads in ${ \mathcal { H } } _ { \mathrm { p o s } }$ use standard RoPE [28] for relative ordinal positions, while heads in ${ \mathcal { H } } _ { \mathrm { t i m e } }$ use RoTE for relative time intervals. Small $\tau _ { h }$ values at the second/minute scale capture fine temporal proximity, whereas large $\tau _ { h }$ values at the day/week scale capture coarse recency. For each sequence, we first shift timestamps by its start time, $\tilde { t } _ { i } = t _ { i } - t _ { 1 }$ and then define $b _ { i } = \lfloor \tilde { t } _ { i } / \tau _ { h } \rfloor$ . This shift avoids unnecessarily large absolute bucket indices without changing pairwise time intervals. For head ℎ, the attention score between query ${ \bf q } _ { h }$ at position � and key $\mathbf { k } _ { h }$ at position � is

$$
\mathrm { S c o r e } _ { h } ( \mathbf { q } _ { h } , \mathbf { k } _ { h } ) = \left\{ \begin{array} { l l } { \left( \mathcal { R } _ { \Theta } ( n ) \mathbf { q } _ { h } \right) ^ { \top } \big ( \mathcal { R } _ { \Theta } ( m ) \mathbf { k } _ { h } \big ) , } & { h \in \mathcal { H } _ { \mathrm { p o s } } , } \\ { \left( \mathcal { R } _ { \Theta } ( b _ { n } ) \mathbf { q } _ { h } \right) ^ { \top } \big ( \mathcal { R } _ { \Theta } ( b _ { m } ) \mathbf { k } _ { h } \big ) , } & { h \in \mathcal { H } _ { \mathrm { t i m e } } , } \end{array} \right.\tag{4}
$$

where $\mathcal { R } _ { \Theta }$ is the block-diagonal rotary matrix and $b _ { i }$ bucketizes timestamps at granularity $\tau _ { h } .$ . Because both position and time are encoded through the same rotary mechanism, RoTE unifies ordinal and temporal proximity at multiple granularities within a single attention operator, which is naturally compatible with FlashAttention. Appendix A.2 derives the relative-time property of RoTE.

Stabilized Residual Normalization (SRN). Pure-attention stacks have a theoretical bias toward token uniformity or rank collapse with depth, although skip connections and MLPs mitigate this degeneration [9]; related analyses connect over-mixing and representational collapse to depth in decoder-only LLMs [2]. More generally, the training objective strongly afects over-smoothing and the utility of deeper Transformer layers [34]. In discriminative recommendation ranking, however, the sequence encoder is trained from sparse labels, and the main loss can even be shortcut by strong non-sequential DLRM features. As a result, directly deepening a Transformer, even with the LLM-style Pre-LN configuration, often yields limited gains in recommendation scaling [11].

We propose a simple and efective SRN to address this depthscaling bottleneck with two minimal modifications. First, it adjusts the LN placement in a Mix-LN-style depth-dependent manner [20] to regulate backward gradient flow across layers. Second, it adds a learnable scalar initialized to a small value to each residual branch to control how much each layer updates the forward representation [1]. For the �-th layer (where SubLayer ∈ {Atn, FFN}),

$$
\begin{array} { r } { \mathbf x _ { l + 1 } = \left\{ \begin{array} { l l } { \mathrm { L N } \big ( \mathbf x _ { l } + \boldsymbol \alpha _ { l } \cdot \mathrm { S u b L a y e r } ( \mathbf x _ { l } ) \big ) , } & { l \leq L _ { \mathrm { s w i t c h } } , } \\ { \mathbf x _ { l } + \boldsymbol \alpha _ { l } \cdot \mathrm { S u b L a y e r } \big ( \mathrm { L N } ( \mathbf x _ { l } ) \big ) , } & { l > L _ { \mathrm { s w i t c h } } , } \end{array} \right. } \end{array}\tag{5}
$$

where $L _ { \mathrm { s w i t c h } }$ is roughly 25% of the total depth, and $\alpha _ { l }$ is initialized to 0.01. This small initialization makes the deep encoder start near an identity mapping, so residual updates are introduced gradually as they receive suficient training signal. Together with depthdependent LN placement, this forward/backward control provides a more stable residual path under sparse, shortcut-prone supervision and leads to more consistent gains under depth scaling.

## 2.4 Lightweight Cross Decoder (�)

The cross decoder � uses a small set of semantic query tokens {[CLS], context, user, ad} to attend to the reusable sequence memory from � . Since $C$ runs for every candidate, we keep its activated computation small while preserving candidate-side capacity through two design choices.

Projection-Free Key/Value. Because � already produces contextualized memory, � directly uses $\textbf { H } = \ T ( S _ { u } )$ as both keys and values, removing $\mathbf { W } _ { K } , \mathbf { W } _ { V }$ , and the value gate $\mathbf { W } _ { g v }$ from repeated candidate-side computation. Given candidate query tokens Z, the resulting cross-attention is

$$
\mathrm { C r o s s A t t n } ( \mathbf { Z } , \mathbf { H } ) = \left[ { \ s o f t m a x } \left( \frac { ( \mathbf { Z } \mathbf { W } _ { Q } ) \mathbf { H } ^ { \top } } { \sqrt { d _ { h } } } \right) \mathbf { H } \odot \sigma ( \mathbf { Z } \mathbf { W } _ { g o } ) \right] \mathbf { W } _ { o } ,\tag{6}
$$

where $\mathbf { W } _ { g o }$ is the output-gate projection.

Token-Specific Parameterization for Query. The query tokens in � play diferent semantic roles, so we assign each token type its own parameters for candidate-side transformations such as ${ \bf W } _ { Q }$ $\mathbf { W } _ { g o }$ in the cross attention, and FFN sublayers. For a transformation $\mathcal { F }$ , TSP applies

$$
\mathcal { F } _ { i } ( \mathbf { x } _ { i } ) = \mathcal { F } ( \mathbf { x } _ { i } ; \mathbf { W } _ { i } ) , \qquad i = 1 , \dotsc , K .\tag{7}
$$

Each token activates only its own parameter set, increasing decoder capacity without materially increasing activated FLOPs.

## 2.5 Auxiliary Supervision for Sequence Scaling

Unlike LLMs, industrial recommendation rankers are hybrid systems that fuse non-sequential DLRM features [22] with user behavior sequences [41] in the final prediction head. This creates a training imbalance we call sequence starvation: the dense, memorizationheavy non-sequential branch shortcuts the main CVR objective and leaves the sequence branch weakly supervised, hindering ef fective scaling of the sequence encoder. We address this with two training-only auxiliary objectives: an auxiliary sequence CVR loss that directly supervises the sequence branch, and an alignment loss that regularizes sequence/non-sequence representations.

Auxiliary Sequence CVR Supervision. Let $\mathbf { H } _ { \mathrm { s e q } } \in \mathbb { R } ^ { K \times d }$ denote the final states of all � sequence-conditioned query tokens. To provide a direct supervised signal to the sequence side, we attach a lightweight auxiliary sequence CVR head to $\mathrm { H } _ { \mathrm { s e q } }$ and the candidate ad features. Unlike the main prediction head, this auxiliary head cannot rely on non-sequential user/context DLRM features. It therefore applies the same BCE supervision to a sequence-conditioned CVR prediction:

$$
\begin{array} { r } { \hat { y } _ { \mathrm { s e q } } = \sigma \big ( g _ { \mathrm { s e q } } ( \mathbf { H } _ { \mathrm { s e q } } , a ) \big ) , \qquad \mathcal { L } _ { \mathrm { s e q } } = \mathbb { E } _ { \mathcal { D } } \left[ \ell _ { \mathrm { B C E } } ( \hat { y } _ { \mathrm { s e q } } , y ) \right] . } \end{array}\tag{8}
$$

As shown in Figure 2, even with a small auxiliary weight $\lambda _ { \mathrm { { s e q } } } =$ 0.1, the auxiliary sequence CVR loss $\mathcal { L } _ { \mathrm { s e q } }$ increases the average gradient norm of the sequence encoder by more than 3×, indicating that direct sequence-side supervision substantially strengthens the optimization signal received by the sequence branch. Appendix A.3 provides additional details.

![](images/2d4713a978b99d03e658724fdab6192248ac74fbd26b2f1f50a0ec3348cb6617.jpg)  
Figure 2: Gradient norm of sequence-encoder parameters with and without auxiliary sequence CVR supervision.

Seq/Non-Seq Alignment. Beyond direct label supervision, we regularize the relationship between sequential and non-sequential user representations. Inspired by vision–language representation alignment [26, 38], we treat behavior sequences and non-sequential features as complementary modalities that describe the same user. We therefore align the behavior-sequence and non-sequential representations with a sigmoid contrastive objective, which is less sensitive to batch size than softmax-based contrastive losses and better suited to scaling recommendation models. Concretely, for user �, let $\mathbf { H } _ { \mathrm { s e q } , i }$ denote the sequence-encoder output and let $\mathbf { h } _ { \mathrm { s e q } , i } = \mathbf { H } _ { \mathrm { s e q } , i } [ \mathsf { C L S } ]$ denote its [CLS] state, which we use as the sequence-level representation. We feed $\mathbf { h } _ { \mathrm { s e q } , i }$ and the non-sequential representation into separate two-layer MLP projection heads followed by $\ell _ { 2 }$ normalization, obtaining $\mathbf { z } _ { \mathrm { s e q } , i }$ and $\mathbf { z } _ { \mathrm { n } \mathrm { s } , i } ,$ respectively. These projection heads are used only during training and removed at inference. We instantiate the alignment loss as:

$$
\mathcal { L } _ { \mathrm { a l i g n } } = - \frac { 1 } { B ^ { 2 } } \sum _ { i = 1 } ^ { B } \sum _ { j = 1 } ^ { B } \log \sigma \Big ( y _ { i j } \cdot \big ( \mathbf { z } _ { \mathrm { s e q } , i } ^ { \top } \mathbf { z } _ { \mathrm { n s } , j } / \tau - b \big ) \Big ) ,\tag{9}
$$

where $y _ { i j } { = } { + } 1$ when the sequence and non-sequence representations come from the same user, and −1 otherwise, with a positive learnable temperature � and a learnable bias �. The overall training objective is:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { c e } } + \lambda _ { \mathrm { s e q } } \mathcal { L } _ { \mathrm { s e q } } + \lambda _ { \mathrm { a l i g n } } \mathcal { L } _ { \mathrm { a l i g n } } .\tag{10}
$$

## 2.6 System Co-Design for Computation Reuse

To translate ReST’s architectural separation into actual computation reuse, we co-design the system stack around the shared encoder outputs via the following two shared-prefix mechanisms.

User-Level Shared-Prefix Training (ULT). In standard instancelevel training, samples from the same user often share long overlapping behavior prefixes, causing the sequence encoder $T$ to repeatedly process nearly identical histories. ULT groups samples from the same user within a time window. For a user with $M _ { u }$ samples, we run � once over the union behavior sequence and apply prefix-valid causal masks so that each sample can only attend to its own valid history. This preserves the original chronological training semantics while reducing the encoder computation from $M _ { u } \cdot \mathrm { F L O P s } ( T )$ to FLOPs(� ).

Shared-Prefix Serving. At serving time, a ranking request scores � candidates against the same user history. Inspired by the M FALCON-style reuse principle [37], we compute the encoder memory $\mathbf { H } = T ( S _ { u } )$ once per request and reuse it as shared prefix states. Candidate-side query tokens are then batched and decoded by � against the same memory. The per-request computation changes from $N { \cdot } \mathrm { F L O P s } ( T ) { + } N { \cdot } \mathrm { F L O P s } ( C )$ down to $\mathrm { F L O P s } ( T ) + { N } { \cdot } \mathrm { F L O P s } ( C )$ Because � is designed to be much lighter than�, shared-prefix reuse decouples sequence-side scaling from repeated per-candidate latency, making larger sequence encoders practical under production constraints.

## 3 EXPERIMENTS

We conduct comprehensive experiments to evaluate ReST’s efectiveness, scalability, eficiency, and deployability. Specifically, we answer five research questions: (RQ1): Does ReST outperform other sequence models under production FLOPs budgets? (RQ2): Does ReST scale better in length, depth, and width than LLM-style baselines? (RQ3): How much does each ReST design component contribute? (RQ4): Do user-level shared-prefix training (ULT) and shared-prefix serving improve training and serving eficiency? (RQ5): Does ReST improve online business metrics under the P99 latency budget?

## 3.1 Experimental Setup

We primarily evaluate ReST on a large-scale industrial dataset from TikTok Shop Ads. All data are strictly anonymized and de-identified before use, containing no personally identifiable information. We keep the ranking stack, behavior sequence, and query tokens fixed across models, and replace only the sequence-modeling component. Due to confidentiality constraints, we report the relative AUC improvement (AUC Δ) over a highly optimized production DLRM baseline for the industrial evaluation. We also evaluate on three public benchmarks: MovieLens-1M and MovieLens-20M [12], and Amazon-Books [21]. To align these public benchmarks with our primary industrial CVR task, we cast their standard sequential recommendation protocols into a pointwise prediction task via chronological splitting. For these public datasets, we report absolute AUC and log loss under identical evaluation protocols for all models, with additional dataset and implementation details provided in the Appendix.

We group baselines into three families. (1) Target-aware attention baselines include DIN [41] and Stacked Target-to-History Cross Attention (STCA) [10]. (2) Causal sequence-model baselines in clude a LLaMA-style Transformer [29] and HSTU [37]. (3) Trans., a strong encoder–decoder baseline, uses a modernized bidirectional encoder following ModernBERT [33], with Pre-LN, GeGLU, and RoPE. We equip Trans. with token-specific parameterization (TSP) and configure it to approximately match the parameter count of ReST. For a fair comparison, all models use the same sequence preprocessing pipeline and receive the same preprocessed behavior sequence as input. Reported FLOPs and parameter counts cover only the sequence-modeling component. Unless otherwise specified, all sequence-modeling baselines use the auxiliary sequence CVR loss during training to mitigate sequence starvation.

![](images/34e8c6d13a165fcf7b496aa39daa5774565c1e3ccac2a32de1bf131bf25759bc.jpg)  
Figure 3: Accuracy vs FLOPs on the internal dataset. ReST achieves favorable accuracy-FLOPs scaling.

## 3.2 Overall Comparison (RQ1)

Table 1 reports the main comparison on the large-scale industrial dataset. We replace the sequence-modeling component in the production DLRM ranker and roughly align FLOPs across the main sequence-scaling models. ReST consistently outperforms all baseline families at comparable FLOPs, reaching the best AUC lift under both Base and Large budgets and +0.92% in the Large setting. In this production ranking setting, an ofline AUC Δ of 0.05% is considered practically meaningful and can translate into measurable online gains. Among the baselines, LLaMA and HSTU outperform DIN and STCA, confirming the value of richer sequence-side self-attention. HSTU is weaker than LLaMA, suggesting that scaling recipes for generative recommendation may not transfer directly to discriminative ranking. The modern encoder–decoder Trans. further improves over LLaMA and is approximately parameter-matched to the full ReST, yet still trails it by 0.20% AUC in the Large setting. Table 3 provides a second parameter-controlled comparison: the parameter-rich cross decoder � contributes +0.05% AUC, while the encoder-only ReST, whose parameter count is close to that of LLaMA, retains approximately +0.87% AUC lift and outperforms LLaMA by 0.17%. Together, these comparisons show that ReST’s gains arise from its recommendation-native sequence modeling and asymmetric architecture rather than parameter count alone.

We then report results on three public datasets in Table 2. Consistent with the industrial findings, ReST achieves the best AUC and log loss across all datasets, while maintaining comparable FLOPs. This suggests that the benefits of ReST are not merely artifacts of large-scale industrial data, but also transfer to standard public benchmarks.

## 3.3 Scaling Analysis (RQ2)

Overall scaling. Figure 3 visualizes the joint scaling behavior when layer depth, hidden dimension, and sequence length are scaled together. We focus this scaling analysis on scalable self-attention-style sequence models, since their sequence-side computation can be cached, making them the most relevant family for long-context scaling under production ranking constraints. As a descriptive summary over the tested compute range, we fit $\Delta \mathrm { A U C } ( F ) = a \cdot F ^ { b }$ where � denotes FLOPs, and obtain $( a , b ) = ( 0 . 6 2 6 , 0 . 1 0 1 )$ for ReST, (0.491, 0.090) for LLaMA, and (0.461, 0.088) for HSTU, with log space $R ^ { 2 } = 0 . 9 9 1 , 0 . 9 3 5 .$ , and 0.924, respectively. ReST attains con sistently higher AUC gains across the tested compute regime, suggesting that rec-native designs yield a more favorable empirical compute-performance trend in our setting. This advantage becomes clearer in the axis-wise scaling analysis below: when scaling con text length, depth, or width independently, LLM-style sequence models show earlier diminishing returns, whereas ReST continues to convert additional sequence-side computation into accuracy gains.

Table 1: Comparison with the production DLRM baseline on the large-scale industrial dataset.
<table><tr><td>Setting</td><td>Metric</td><td>DLRM</td><td>DIN</td><td>STCA</td><td>LLaMA</td><td>HSTU</td><td>Trans.</td><td>ReST</td></tr><tr><td rowspan="3">Base</td><td>AUC ∆</td><td>一</td><td>+0.23%</td><td>+0.38%</td><td>+0.53%</td><td>+0.51%</td><td>+0.57%</td><td>+0.67%</td></tr><tr><td>FLOPs (G)</td><td></td><td>0.07</td><td>1.66</td><td>1.37</td><td>1.70</td><td>1.52</td><td>1.52</td></tr><tr><td>Params (M)</td><td>-</td><td>0.12</td><td>2.97</td><td>0.86</td><td>0.66</td><td>5.36</td><td>5.69</td></tr><tr><td rowspan="3">Large</td><td>AUC ∆</td><td>一</td><td>+0.27%</td><td>+0.46%</td><td>+0.70%</td><td>+0.61%</td><td>+0.72%</td><td>+0.92%</td></tr><tr><td>FLOPs (G)</td><td>-</td><td>1.18</td><td>31.98</td><td>30.21</td><td>32.88</td><td>32.38</td><td>32.40</td></tr><tr><td>Params (M)</td><td>一</td><td>0.46</td><td>15.05</td><td>3.43</td><td>1.98</td><td>21.33</td><td>22.64</td></tr></table>

Table 2: Public benchmarks on ML-1M, ML-20M, and Amazon-Books.

<table><tr><td>Dataset</td><td>Metric</td><td>DIN</td><td>STCA</td><td>LLaMA</td><td>HSTU</td><td>Trans.</td><td>ReST</td></tr><tr><td>ML-1M</td><td>AUC</td><td>0.8064</td><td>0.8074</td><td>0.8087</td><td>0.8079</td><td>0.8090</td><td>0.8099</td></tr><tr><td></td><td>Log loss</td><td>0.5346</td><td>0.5372</td><td>0.5334</td><td>0.5343</td><td>0.5329</td><td>0.5317</td></tr><tr><td>ML-20M</td><td>AUC</td><td>0.8070</td><td>0.8079</td><td>0.8088</td><td>0.8082</td><td>0.8103</td><td>0.8135</td></tr><tr><td></td><td>Log loss</td><td>0.5329</td><td>0.5333</td><td>0.5317</td><td>0.5322</td><td>0.5302</td><td>0.5292</td></tr><tr><td>Books</td><td>AUC</td><td>0.7521</td><td>0.7527</td><td>0.7541</td><td>0.7566</td><td>0.7556</td><td>0.7571</td></tr><tr><td></td><td>Log loss</td><td>0.4345</td><td>0.4357</td><td>0.4305</td><td>0.4242</td><td>0.4244</td><td>0.4221</td></tr><tr><td></td><td>FLOPs (M)</td><td>1.90</td><td>24.10</td><td>21.13</td><td>29.77</td><td>23.32</td><td>23.41</td></tr><tr><td></td><td>Params (M)</td><td>0.01</td><td>0.11</td><td>0.03</td><td>0.02</td><td>0.17</td><td>0.18</td></tr></table>

![](images/9a8550b27a482e25c59a849ed6280283f23662f8bdab0df4878eb225a3aeb63f.jpg)  
(a) Sequence Scale

![](images/919ecce376afd4335dbb695a053eedd6de7d8fa519cd56d2c4f36b160eb675c1.jpg)  
(b) Depth Scale

![](images/e09165b1412ee2fcb08006fed71aaa945d9d3d8023c0f2c83650c3e3ce152e9f.jpg)  
(c) Width Scale  
Figure 4: Scaling behavior along three axes: (a) sequence length, (b) model depth, and (c) hidden dimension. ReST consistently outperforms competing sequence models across the explored budgets.

Sequence length scale. Figure 4(a) varies the available behavior context from 100 to 16k tokens while keeping depth and hidden dimension fixed. We evaluate each context-length setting on recent data with suficient history coverage and report each model’s

AUC lift over the production baseline evaluated on the same samples. Without auxiliary sequence CVR supervision, LLaMA exhibits the sequence-starvation efect discussed in Sec. 2.5: extending the available context yields almost no additional lift. With auxiliary sequence CVR supervision, all sequence models show stronger gains at larger context budgets, with the ordering ReST > LLaMA > HSTU. Across the evaluated settings, ReST continues to improve up to 16k tokens, achieving an additional AUC lift of approximately 0.4% over the evaluated context range. This represents the largest observed increase among the compared models and suggests that rec-native designs enable more efective use of longer user histories. Model depth scale. Figure 4(b) varies the number of layers while keeping sequence length and hidden dimension fixed. For ReST, � and � depths are scaled 1:1. As depth increases, ReST improves steadily, with AUC gain rising from +0.59% to +0.76%. In contrast,

Table 3: Ablation: architecture and rec-native block designs.
<table><tr><td>Group</td><td>Variant</td><td>AUC ∆</td><td>Param ∆</td><td>FLOPsΔ</td></tr><tr><td rowspan="5">Architecture</td><td>T</td><td>base</td><td></td><td></td></tr><tr><td>+ MoE in T</td><td>+0.00%</td><td>+500%</td><td>+0.2%</td></tr><tr><td>+ C (w/o TSP)</td><td>+0.03%</td><td>+100.3%</td><td>+0.2%</td></tr><tr><td>+ C (w/o PF-KV)</td><td>+0.05%</td><td>+473.2%</td><td>+6.7%</td></tr><tr><td>+ C</td><td>+0.05%</td><td>+473.2%</td><td>+0.2%</td></tr><tr><td rowspan="4">Gated Attn</td><td>w/o gate</td><td>base</td><td>一</td><td>一</td></tr><tr><td>+ O gate</td><td>+0.03%</td><td>+7.6%</td><td>+3.7%</td></tr><tr><td>+ O &amp; QKV gate</td><td>+0.06%</td><td>+11.4%</td><td>+14.9%</td></tr><tr><td>+ O &amp;V gate</td><td>+0.06%</td><td>+8.9%</td><td>+7.5%</td></tr><tr><td rowspan="4">Norm</td><td>Pre-LN</td><td>base</td><td>一</td><td>一</td></tr><tr><td>Post-LN</td><td>+0.00%</td><td>≈ 0</td><td>≈ 0</td></tr><tr><td>Mix-LN</td><td>+0.02%</td><td>≈0</td><td>≈ 0</td></tr><tr><td>SRN</td><td>+0.08%</td><td>≈ 0</td><td>≈0</td></tr><tr><td rowspan="4">Pos./Temporal</td><td>w/o pos</td><td>base</td><td>一</td><td>一</td></tr><tr><td>RoPE</td><td>+0.04%</td><td>≈ 0</td><td>≈ 0</td></tr><tr><td>RoPE + RoTE (sec)</td><td>+0.08%</td><td>≈ 0</td><td>≈ 0</td></tr><tr><td> $\mathbf { R o P E } + \mathbf { R o T E }$ </td><td>+0.12%</td><td>≈ 0</td><td>≈0</td></tr></table>

LLaMA and HSTU improve at shallow depths but exhibit dimin ishing returns beyond 8 and 16 layers, respectively. This pattern suggests that simply stacking deeper generic sequence blocks is insuficient for recommendation ranking: under sparse supervision and noisy behavior data, additional depth may saturate before being efectively optimized. ReST benefits more consistently from depth, and the component ablation in Sec. 3.4 further confirms that SRN is especially important for deeper models.

Hidden dimension scale. Figure 4(c) scales the hidden dimension � ∈ {128, 256, 512} with depth and sequence length fixed. Both baselines saturate early: HSTU and LLaMA flatten around �=256, and increasing to �=512 yields almost no additional AUC despite a substantial increase in FLOPs. ReST follows a diferent trend, continuing to improve from +0.67% to +0.80% and remaining the only tested model with monotonic gains at the largest width. The combination of gated attention, temporal encoding, and stabilized residual learning allows ReST to benefit from larger hidden dimensions rather than merely increasing arithmetic cost.

## 3.4 Ablation Studies (RQ3)

Architecture and Compute Allocation. We study where additional capacity should be allocated in ReST. As shown in the Architecture group of Table 3, simply adding MoE-based capacity [30] to the sequence encoder � increases parameters by 500% but yields no measurable AUC gain, suggesting that encoder capac ity alone is not the limiting factor. In contrast, adding a lightweight cross decoder � improves AUC by +0.03% with only +0.2% extra FLOPs, showing the value of candidate-conditioned readout. Tokenspecific parameterization further raises the gain to +0.05%, while projection-free KV keeps the FLOPs overhead at +0.2% rather than +6.7%. Thus, ReST benefits from allocating capacity to the heterogeneous candidate-conditioned decoder while keeping repeated per-candidate arithmetic nearly unchanged.

Table 4: Ablation: auxiliary objectives.
<table><tr><td>Backbone</td><td> $\mathcal { L } _ { \mathrm { s e q } } \left( \mathrm { B a s e } \right)$ </td><td> $\mathcal { L } _ { \mathrm { { s e q } } } \left( \mathrm { L a r g e } \right)$ </td><td> $\mathcal { L } _ { \mathrm { a l i g n } }$ </td></tr><tr><td>DIN</td><td>+0.02%</td><td>+0.02%</td><td>一</td></tr><tr><td>STCA</td><td>+0.13%</td><td>+0.20%</td><td>一</td></tr><tr><td>LLaMA</td><td>+0.16%</td><td>+0.22%</td><td>+0.05%</td></tr><tr><td>HSTU</td><td>+0.10%</td><td>+0.15%</td><td>+0.05%</td></tr><tr><td>ReST</td><td>+0.07%</td><td>+0.12%</td><td>+0.06%</td></tr></table>

Gated Attention. The Gated Attention group of Table 3 shows that an output gate alone already yields a +0.03% gain. Adding gates to all Q/K/V branches increases AUC to +0.06%, but also incurs a large FLOPs overhead (+14.9%). In contrast, combining the output gate with a value gate reaches the same +0.06% gain at roughly half the extra compute (+7.5% FLOPs). Appendix A.1 further shows that combining the output and value gates yields better performance under noisy behavior perturbations. We therefore adopt Output + Value gating as our Dual-Gated Attention design for a better accuracy–eficiency trade-of.

Normalization and Residual Connections. We then evaluate how normalization placement and residual scaling afect optimization in a fixed 16-layer setting. As shown in the Norm group of Table 3, the LLaMA-style Pre-LN design serves as the local baseline; switching to Post-LN brings no gain, while Mix-LN provides a modest +0.02% improvement, suggesting that normalization placement matters for deeper recommendation models. Adding scaled residual learning on top of this design, SRN achieves the largest gain of +0.08%, substantially outperforming the other variants at the same depth. This indicates that training deeper sequence models on sparse recommendation data benefits from both appropriate normalization placement and controlled residual updates.

Positional and Temporal Encoding. The Positional and Temporal Encoding group in Table 3 starts from a no-position baseline. Adding RoPE improves AUC by +0.04%, confirming the benefit of ordinal positional information. Adding single-granularity RoTE with one-second buckets on top of RoPE raises the gain to +0.08%, showing that physical time provides complementary signal beyond ordinal position. The default multi-granularity RoTE, which assigns distinct temporal granularities to diferent temporal attention heads, further raises the gain to +0.12%, demonstrating the benefit of modeling temporal proximity at multiple resolutions. Importantly, all variants already receive the same delta-time features as token-level side information, so these gains reflect the additional value of injecting physical time directly into the attention mechanism rather than merely exposing the model to temporal features. Further details of RoTE are provided in Appendix A.2.

Auxiliary Objectives. Table 4 summarizes the efects of the two training-only auxiliary objectives. Auxiliary sequence CVR supervision improves all sequence-aware backbones, with larger gains in the Large setting: STCA improves from +0.13% to +0.20%, LLaMA from +0.16% to +0.22%, HSTU from +0.10% to +0.15%, and ReST from +0.07% to +0.12%. This supports our motivation that larger sequence modules require stronger direct sequence-side supervision to avoid being shortcut by the dense non-sequential branch. The smaller marginal gain on ReST suggests that its rec-native encoder is less dependent on the auxiliary objective, although auxiliary supervision remains beneficial. On top of auxiliary sequence CVR supervision, the alignment objective further improves LLaMA, HSTU, and ReST by +0.05%, +0.05%, and +0.06%, respectively, indicating that sequence/non-sequence alignment provides a stable additional regularization signal across scalable sequence models. Since these objectives are architecture-agnostic and introduce no serving-time overhead, we enable auxiliary sequence CVR supervision for all sequence-scaling baselines in the main comparison.

## 3.5 System Eficiency and Online A/B Test Result (RQ4 & RQ5)

To translate the reuse boundary of ReST into actual system eficiency, we implement the training and serving stack around the shared sequence encoder. On the training side, our baseline is already a highly optimized pipeline with fused attention/GEMM kernels, mixed-precision training, and padding removal for variable length sequences. On top of this strong baseline, user-level sharedprefix training (ULT) further improves end-to-end training throughput by 5.8× by removing redundant prefix computation across samples from the same user, showing that the gain comes from computation reuse rather than from low-level kernel optimization alone. On the serving side, request-level shared-prefix serving is the key enabler for online sequence scaling: � is executed once per request and reused across all candidates, yielding up to a 20× reduction in the per-request inference cost of the sequence-modeling component. This reuse is critical for deploying ReST online under strict production latency budgets and substantially narrows the deployment gap between large sequence models and industrial ranking systems.

Table 5: Online A/B results.
<table><tr><td>AUC∆</td><td>Advv Lift</td><td>p-value</td></tr><tr><td>+1.31%</td><td>+11.93%</td><td>&lt; 0.01</td></tr></table>

To validate the practical impact of ReST, we conduct a one-week online A/B test on TikTok Shop Ads, one of the largest advertising platforms. Users are randomly assigned to control and treatment groups, with the treatment group receiving 20% ofproduction trafic. The control is a strong production DLRM ranker, while the treatment replaces its sequence modeling component with ReST. We scale ReST to a deployable configuration that satisfies the 50 ms P99 latency budget. We use a revenue-related business metric, Advertiser Value (Advv), as the primary online metric, report streaming AUC for online prediction quality, and monitor end-to-end P99 latency and other production guardrails. As shown in Table 5, this latency-constrained configuration improves AUC by +1.31% and lifts Advv by a statistically significant +11.93% (� < 0.01), while satisfying all other monitored production guardrails. ReST has been fully deployed in production, showing that scaling behaviorsequence modeling through rec-native designs can translate ofline gains into substantial business value in a realistic large-scale production setting.

## 4 RELATED WORK

Deep Learning Recommendation Models (DLRMs). Modern industrial ranking systems are largely built on the DLRM paradigm [5, 22], which combines large embedding tables with relatively lightweight interaction modules. A long line of work has refined this paradigm from complementary angles. DCN-V2 [31] strengthens explicit feature crossing; DIN [41] introduces target-aware attention for user-behavior modeling; and SIM/SDIM [3, 23] extend behavior modeling to much longer histories through retrieval-based approximation. Together, these methods establish behavior sequences as highly informative signals for ranking, but typically keep sequence modeling compact or heavily compressed within a larger ranking stack rather than treating it as a primary scaling axis.

Large Recommendation Models (LRMs). Recent work on large recommendation models can be viewed through the lens of what is being scaled. One line scales the joint interaction between behavior sequences and dense non-sequential features: RankMixer [42] scales dense feature interaction through hardware-aware token mixing; OneTrans [39] and InterFormer [36] integrate behavior sequences with non-sequential features; and FAT [35] and MTmix-Att [24] scale heterogeneous feature interaction through field-aware or mixture-of-experts parameterization. These methods strengthen the joint feature-interaction stack rather than isolating the behavior sequence as the main scaling object, making them complementary to our focus. A second line scales sequence modeling more directly. Building on sequence Transformers such as SASRec [16], STCA [10] strengthens target-to-history cross attention, LLaMAstyle decoder-only baselines [11, 29] enlarge sequence-model capacity with generic self-attention blocks, and HSTU [37] demonstrates strong scaling in a generative recommendation formulation. Yet these recipes do not fully address behavior-sequence scaling in discriminative ranking, where noisy and sparsely supervised histories must be scored against many candidates under strict latency constraints. ReST addresses these limitations by making the behavior sequence a reusable primary scaling axis through rec-native designs.

## 5 CONCLUSION

In this paper, we introduced ReST, a recommendation-native framework for scaling behavior-sequence Transformers in industrial ranking under strict latency constraints. By addressing the signal quality and computation asymmetry challenges through rec-native sequence modeling, asymmetric architecture design, and sharedprefix computation reuse, ReST mitigates sequence starvation and scales more efectively along behavior sequence length, depth, and width than LLM-style blocks. This enables larger sequence models to achieve substantial business gains under practical latency budgets, leading to full production deployment. Our results suggest that, with rec-native designs, behavior-sequence scaling remains an under-exploited axis for industrial ranking and can translate into substantial online business value.

## ACKNOWLEDGMENTS

We thank Chenzhi Zhou for his valuable PMO support, especially in facilitating cross-functional collaboration and production deployment.

## REFERENCES

[1] Thomas Bachlechner, Bodhisattwa Prasad Majumder, Henry Mao, Gary Cottrell, and Julian McAuley. 2021. ReZero is all you need: fast convergence at large depth. In Proceedings ofthe Thirty-Seventh Conference on Uncertainty in Artificial Intelligence (UAI) (Proceedings ofMachine Learning Research, Vol. 161), Cassio de Campos and Marloes H. Maathuis (Eds.). PMLR, 1352–1361.

[2] Federico Barbero, Álvaro Arroyo, Xiangming Gu, Christos Perivolaropoulos, Michael M. Bronstein, Petar Veličković, and Razvan Pascanu. 2025. Why Do LLMs Attend to the First Token?. In Conference on Language Modeling (COLM).

[3] Yue Cao, Xiaojiang Zhou, Jiaqi Feng, Peihao Huang, Yao Xiao, Dayao Chen, and Sheng Chen. 2022. Sampling Is All You Need on Modeling Long-Term User Behaviors for CTR Prediction. In Proceedings ofthe 31st ACM International Conference on Information & Knowledge Management. 2974–2983.

[4] Olivier Chapelle. 2014. Modeling Delayed Feedback in Display Advertising. In Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 1097–1105.

[5] Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, Rohan Anil, Zakaria Haque, Lichan Hong, Vihan Jain, Xiaobing Liu, and Hemal Shah. 2016. Wide & deep learning for recommender systems. In Proceedings ofthe 1st workshop on deep learning for recommender systems. 7–10.

[6] Junsu Cho, Dongmin Hyun, Seongku Kang, and Hwanjo Yu. 2021. Learning Het erogeneous Temporal Patterns of User Preference for Timely Recommendation. In Proceedings ofthe Web Conference 2021. 1274–1283.

[7] Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep Neural Networks for YouTube Recommendations. In Proceedings ofthe 10th ACM Conference on Recommender Systems. 191–198.

[8] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. FlashAttention: Fast and Memory-Eficient Exact Attention with IO-Awareness. In Advances in Neural Information Processing Systems (NeurIPS), Vol. 35. Curran Associates, Inc., 16344–16359.

[9] Yihe Dong, Jean-Baptiste Cordonnier, and Andreas Loukas. 2021. Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth. In Proceedings of the 38th International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 139). PMLR, 2793–2803.

[10] Lin Guan, Jia-Qi Yang, Zhishan Zhao, Beichuan Zhang, Bo Sun, Xuanyuan Luo, Jinan Ni, Xiaowen Li, Yuhang Qi, Zhifang Fan, et al. 2026. Make it long, keep it fast: End-to-end 10k-sequence modeling at billion scale on Douyin. In Proceedings ofthe ACM Web Conference 2026. 7989–7998.

[11] Wei Guo, Hao Wang, Luankang Zhang, Jin Yao Chin, Zhongzhou Liu, Kai Cheng, Qiushi Pan, Yi Quan Lee, Wanqi Xue, Tingjia Shen, et al. 2024. Scaling new fron tiers: Insights into large recommendation models. arXiv preprint arXiv:2412.00714 (2024).

[12] F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems 5, 4 (2015), 1–19.

[13] Jordan Hofmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. 2022. Training Compute-Optimal Large Language Models. In Advances in Neural Information Processing Systems (NeurIPS), Vol. 35. Curran Associates, Inc., 30016–30030.

[14] Weizhe Hua, Zihang Dai, Hanxiao Liu, and Quoc V. Le. 2022. Transformer Quality in Linear Time. In Proceedings ofthe 39th International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 162). PMLR, 9099–9117.

[15] Thorsten Joachims, Adith Swaminathan, and Tobias Schnabel. 2017. Unbiased Learning-to-Rank with Biased Feedback. In Proceedings ofthe Tenth ACM International Conference on Web Search and Data Mining. 781–789.

[16] Wang-Cheng Kang and Julian McAuley. 2018. Self-attentive sequential recommendation. In 2018 IEEE International Conference on Data Mining (ICDM). IEEE, 197–206.

[17] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jefrey Wu, and Dario Amodei. 2020. Scaling Laws for Neural Language Models. arXivpreprint arXiv:2001.08361 (2020).

[18] Sofia Ira Ktena, Alykhan Tejani, Lucas Theis, Pranay Kumar Myana, Deepak Dilipkumar, Ferenc Huszár, Steven Yoo, and Wenzhe Shi. 2019. Addressing Delayed Feedback for Continuous Training with Neural Networks in CTR Prediction. In Proceedings of the 13th ACM Conference on Recommender Systems. 187–195.

[19] Jiacheng Li, Yujie Wang, and Julian McAuley. 2020. Time interval aware selfattention for sequential recommendation. In Proceedings of the 13th International Conference on Web Search and Data Mining. 322–330.

[20] Pengxiang Li, Lu Yin, and Shiwei Liu. 2025. Mix-LN: Unleashing the Power of Deeper Layers by Combining Pre-LN and Post-LN. In The Thirteenth International Conference on Learning Representations (ICLR).

[21] Julian McAuley, Christopher Targett, Qinfeng Shi, and Anton van den Hengel. 2015. Image-based recommendations on styles and substitutes. In Proceedings ofthe 38th international ACM SIGIR conference on research and development in

information retrieval. 43–52.

[22] Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang, Narayanan Sundaraman, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu, Alisson G Azzolini, et al. 2019. Deep learning recommendation model for personalization and recommendation systems. arXiv preprint arXiv:1906.00091 (2019).

[23] Qi Pi, Guorui Zhou, Yujing Zhang, Zhe Wang, Lejian Ren, Ying Fan, Xiaoqiang Zhu, and Kun Gai. 2020. Search-based user interest modeling with lifelong sequential behavior data for click-through rate prediction. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management. 2685–2692.

[24] Xianyang Qi, Yuan Tian, Zhaoyu Hu, Zhirui Kuai, Chang Liu, Hongxiang Lin, and Lei Wang. 2025. MTmixAtt: Integrating Mixture-of-Experts with Multi-Mix Attention for Large-Scale Recommendation. arXiv preprint arXiv:2510.15286 (2025).

[25] Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, Songlin Yang, Rui Men, Le Yu, Fei Huang, Suozhi Huang, Dayiheng Liu, Jingren Zhou, and Junyang Lin. 2025. Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free. In Advances in Neural Information Processing Systems, Vol. 38. Curran Associates, Inc., 100092–100118.

[26] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning Transferable Visual Models From Natural Language Supervision. In Proceedings ofthe 38th International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 139). PMLR, 8748–8763.

[27] Noam Shazeer. 2020. GLU Variants Improve Transformer. arXiv preprint arXiv:2002.05202 (2020).

[28] Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. 2024. RoFormer: Enhanced Transformer with Rotary Position Embedding. Neurocomputing 568 (2024), 127063.

[29] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 (2023).

[30] Lean Wang, Huazuo Gao, Chenggang Zhao, Xu Sun, and Damai Dai. 2024. Auxiliary-loss-free load balancing strategy for mixture-of-experts. arXiv preprint arXiv:2408.15664 (2024).

[31] Ruoxi Wang, Rakesh Shivanna, Derek Z. Cheng, Sagar Jain, Dong Lin, Lichan Hong, and Ed H. Chi. 2021. DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-Scale Learning to Rank Systems. In Proceedings of the Web Conference 2021. 1785–1797.

[32] Wenjie Wang, Fuli Feng, Xiangnan He, Liqiang Nie, and Tat-Seng Chua. 2021. Denoising Implicit Feedback for Recommendation. In Proceedings of the 14th ACM International Conference on Web Search and Data Mining. 373–381.

[33] Benjamin Warner, Antoine Chafin, Benjamin Clavié, Orion Weller, Oskar Hallström, Said Taghadouini, Alexis Gallagher, Raja Biswas, Faisal Ladhak, Tom Aarsen, Grifin Thomas Adams, Jeremy Howard, and Iacopo Poli. 2025. Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efi cient, and Long Context Finetuning and Inference. In Proceedings ofthe 63rd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, 2526–2547.

[34] Fuzhao Xue, Jianghai Chen, Aixin Sun, Xiaozhe Ren, Zangwei Zheng, Xiaoxin He, Yongming Chen, Xin Jiang, and Yang You. 2023. A Study on Transformer Configuration and Training Objective. In Proceedings ofthe 40th International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 202). PMLR, 38913–38925.

[35] Bencheng Yan, Yuejie Lei, Zhiyuan Zeng, Zheye Deng, Di Wang, Kaiyi Lin, Pengjie Wang, Chuan Yu,Jian Xu, and Bo Zheng. 2026. From Scaling to Structured Expressivity: Rethinking Transformers for CTR Prediction. In Proceedings ofthe 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining. ACM, 8358–8368.

[36] Zhichen Zeng, Xiaolong Liu, Mengyue Hang, Xiaoyi Liu, Qinghai Zhou, Chaofei Yang, Yiqun Liu, Yichen Ruan, Laming Chen, Yuxin Chen, Yujia Hao, Jiaqi Xu, Jade Nie, Xi Liu, Buyun Zhang, Wei Wen, Siyang Yuan, Hang Yin, Xin Zhang, Kai Wang, Wen-Yen Chen, Yiping Han, Huayu Li, Chunzhi Yang, Bo Long, Philip S. Yu, Hanghang Tong, and Jiyan Yang. 2025. InterFormer: Efective Heterogeneous Interaction Learning for Click-Through Rate Prediction. In Proceedings ofthe 34th ACM International Conference on Information and Knowledge Management (CIKM). ACM, 6225–6233.

[37] Jiaqi Zhai, Lucy Liao, Xing Liu, Yueming Wang, Rui Li, Xuan Cao, Leon Gao, Zhaojie Gong, Fangda Gu, Jiayuan He, Yinghai Lu, and Yu Shi. 2024. Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations. In Proceedings ofthe 41st International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 235). PMLR, 58484–58509.

[38] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. 2023. Sigmoid Loss for Language Image Pre-Training. In Proceedings ofthe IEEE/CVF

international conference on computer vision. 11975–11986.

[39] Zhaoqi Zhang, Haolei Pei, Jun Guo, Tianyu Wang, Yufei Feng, Hui Sun, Shaowei Liu, and Aixin Sun. 2026. OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender. In Proceedings of the ACM Web Conference 2026. ACM, 8162–8170.

[40] Wayne Xin Zhao, Shanlei Mu, Yupeng Hou, Zihan Lin, Yushuo Chen, Xingyu Pan, Kaiyuan Li, Yujie Lu, Hui Wang, Changxin Tian, et al. 2021. RecBole: Towards a Unified, Comprehensive and Eficient Framework for Recommendation Algorithms. In Proceedings ofthe 30th ACM International Conference on Information & Knowledge Management. 4653–4664.

[41] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep interest network for click-through rate prediction. In Proceedings ofthe 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 1059–1068

[42] Jie Zhu, Zhifang Fan, Xiaoxie Zhu, Yuchen Jiang, Hangyu Wang, Xintian Han, Haoran Ding, Xinmin Wang, Wenlin Zhao, Zhen Gong, Huizhi Yang, Zheng Chai, Zhe Chen, Yuchao Zheng, Qiwei Chen, Feng Zhang, Xun Zhou, Peng Xu, Xiao Yang, Di Wu, and Zuotao Liu. 2025. RankMixer: Scaling Up Ranking Models in Industrial Recommenders. In Proceedings ofthe 34th ACM International Conference on Information and Knowledge Management. ACM, 6309–6316.

## A APPENDIX

## A.1 Denoising Analysis of Dual-Gated Attention

To further validate the denoising role of Dual-Gated Attention (DGA), we conduct a controlled noise-injection study. Specifically, we inject Gaussian noise into behavior tokens at diferent noise levels and compare standard attention, output-gated attention, and DGA under the same training and evaluation protocol. This setup isolates whether the attention block can remain robust when a growing fraction of behavior evidence becomes unreliable.

Figure 5 shows that DGA is consistently more robust as token noise increases. Standard attention degrades most rapidly because noisy behavior features directly enter the value stream and are ag gregated into the contextual representation. Output-gated attention is more stable, but still cannot prevent corrupted token features from being mixed during aggregation. By contrast, DGA gates both the value stream before aggregation and the output stream after aggregation, allowing the model to suppress unreliable evidence before it contaminates the sequence context. This result supports our design choice that value and output gates are complementary for noisy recommendation sequences.

## A.2 Relative-Time Property of RoTE

RoTE follows the same principle as RoPE: query and key vectors are rotated according to their timestamp buckets, but their dot product cancels the shared coordinate frame and depends only on the relative rotation. We provide a compact derivation for a temporal head. As defined in the main text, timestamps are shifted by the sequence start time before bucketization: $\tilde { t } _ { i } = t _ { i } - t _ { 1 }$ and $b _ { i } = \lfloor \tilde { t } _ { i } / \tau _ { h } \rfloor$ . This keeps bucket indices within the time span of each sequence rather than the range of absolute wall-clock timestamps. Let $b _ { n }$ and $b _ { m }$ denote the resulting buckets under head-specific granularity $\tau _ { h } .$ . The RoTE attention score can be written as

$$
\begin{array} { r l } { \mathrm { S c o r e } _ { \mathrm { R o T E } } ( \mathbf { q } , \mathbf { k } ) = ( \mathcal { R } _ { \Theta } ( b _ { n } ) \mathbf { q } ) ^ { \top } ( \mathcal { R } _ { \Theta } ( b _ { m } ) \mathbf { k } ) } & { } \\ & { = \mathbf { q } ^ { \top } \mathcal { R } _ { \Theta } ( b _ { n } ) ^ { \top } \mathcal { R } _ { \Theta } ( b _ { m } ) \mathbf { k } } \\ & { = \mathbf { q } ^ { \top } \mathcal { R } _ { \Theta } ( - b _ { n } ) \mathcal { R } _ { \Theta } ( b _ { m } ) \mathbf { k } } \\ & { = \mathbf { q } ^ { \top } \mathcal { R } _ { \Theta } ( b _ { m } - b _ { n } ) \mathbf { k } . } \end{array}\tag{11}
$$

![](images/d1f82e3e1b5c373795a456273f0160d5223605fee554ab1fcc863ad1c770ec7f.jpg)  
Figure 5: Dual-Gated Attention is more robust as token noise increases, supporting its value-side filtering and output-side modulation design.

The derivation uses two basic properties of rotation matrices:

$$
\mathscr { R } _ { \Theta } ( x ) ^ { \top } = \mathscr { R } _ { \Theta } ( - x ) , \qquad \mathscr { R } _ { \Theta } ( x ) \mathscr { R } _ { \Theta } ( y ) = \mathscr { R } _ { \Theta } ( x + y ) .
$$

Therefore, the temporal head does not depend on the shifted timestamp buckets $b _ { n }$ and $b _ { m }$ separately, but only on their relative bucket diference $b _ { m } - b _ { n } .$ . Because subtracting the sequence start time cancels in pairwise diferences, it does not change the elapsed time represented between two tokens.

This derivation establishes relative-time dependence, not a monotonic or one-to-one mapping from elapsed time to attention score: rotary features are periodic, and the resulting score also depends on the query and key contents. Events within the same bucket are intentionally indistinguishable at that granularity. Assigning diferent $\tau _ { h }$ values to diferent heads exposes relative time intervals at complementary resolutions while retaining compatibility with the rotary computation used by eficient attention kernels. In our industrial implementation, we reserve one attention head for RoPE and assign the remaining heads to head-specific temporal granularities $\tau _ { h } ~ \in$ {second, minute, hour, day, week, quarter, year}, enabling diferent temporal heads to capture recency at complementary resolutions, from fine-grained second/minute patterns to long-horizon quarter/year trends.

## A.3 Gradient Analysis for Auxiliary Sequence CVR Supervision

To further diagnose the efect of auxiliary sequence CVR supervision, we monitor the gradient norm received by the Transformer sequence encoder during training. As discussed in Section 2.5, the main CVR objective is optimized after sequence representations are fused with strong non-sequential DLRM features, so the prediction head can partially reduce the loss through the non-sequential branch. This shortcut can leave the sequence encoder with weak gradients, especially when the sequence module is scaled up.

Figure 2 reports the average gradient norm of projection matrices across all attention layers in the sequence encoder. Specifically, at each training checkpoint, we compute the gradient norms of the �, �, and � projection matrices in every sequence-encoder attention layer and report their average across layers. Even with a small auxiliary weight $\lambda _ { \mathrm { s e q } } = 0 . 1$ , auxiliary sequence CVR supervision increases the average gradient norm by more than 3× (from 2.35 to 8.50), indicating that direct sequence-side supervision substantially strengthens the optimization signal received by the sequence branch. We observe a similar trend for the FFN parameters, suggesting that the auxiliary objective improves gradient flow throughout the sequence encoder rather than only in the attention projections.

## A.4 Implementation Details

Public Benchmark Settings. All public benchmarks are used only for ofline academic benchmarking; models are implemented with RecBole [40] and trained using Adam. We tune the learning rate and weight decay over $\{ 1 0 ^ { - 4 } , 5 { \times } \bar { 1 } 0 ^ { - 4 } , 1 0 ^ { - 3 } \}$ , use a batch size of 4,096, and train for up to 20 epochs with early stopping on validation AUC. We apply 5-core filtering to each dataset, retaining only users and items with at least five interactions. Table 6 summarizes the resulting dataset statistics.

Table 6: Public dataset overview.
<table><tr><td>Statistic</td><td>ML-1M</td><td>ML-20M</td><td>Books</td></tr><tr><td>Users</td><td>6K</td><td>138K</td><td>604K</td></tr><tr><td>Items</td><td>4K</td><td>18K</td><td>368K</td></tr><tr><td>Interactions</td><td>1M</td><td>20M</td><td>9M</td></tr></table>

Evaluation of Public Benchmarks. Ratings of 4 or above are treated as positive samples and lower ratings as negative samples. No additional negative sampling is performed; the observed lowrating interactions serve as negative instances. For each user, interactions are sorted chronologically and split into train/validation/test sets with an 80%/10%/10% ratio. We report AUC and binary log loss under this labeled binary prediction setting.

Model Configuration of Public Benchmarks. The maximum behavior sequence length is 200 and the item embedding dimension is 32. Transformer-based models use two sequence-modeling layers and, for encoder–decoder architectures, two additional decoder layers, with two attention heads and a feed-forward inner dimension of 128. The prediction head is a three-layer MLP with hidden sizes [256, 128, 64]. We apply a dropout rate of 0.25 to the embedding layer, hidden layers, and attention weights.