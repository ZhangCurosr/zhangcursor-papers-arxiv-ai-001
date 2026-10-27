# Cross-Domain Sequential Recommendation with Representation-Level User Transfer

Abdulaziz Samra   
Skolkovo Institute of Science and   
Technology   
Moscow, Russian Federation   
Higher School of Economics   
Moscow, Russian Federation   
abdulaziz.samra@skoltech.ru   
Nikita Sukhorukov   
AXXX   
Moscow, Russian Federation   
Skolkovo Institute of Science and   
Technology   
Moscow, Russian Federation   
Higher School of Economics   
Moscow, Russian Federation   
niksukhorukov7@gmail.com Alexey Vasilev Sber AI Lab   
Moscow, Russian Federation   
Higher School of Economics   
Moscow, Russian Federation alexxl.vasilev@yandex.ru   
Alexey Grishanov   
Sber AI Lab   
Moscow, Russian Federation   
grishanov.av@phystech.edu   
Evgeny Frolov   
AXXX   
Moscow, Russian Federation   
Higher School of Economics   
Moscow, Russian Federation   
evfro@live.ru

## Abstract

Cross-domain sequential recommendation aims to improve nextitem prediction in a target domain by leveraging user behavior from auxiliary source domains. However, real-world deployment faces two critical challenges: business constraints that isolate interaction data across domains, and user heterogeneity, where many users lack cross-domain activity. This makes existing cross-domain methods dificult to apply. In this paper, we propose a simple and practi cal framework for cross-domain sequential recommendations that transfers information exclusively through learned user representations, without sharing raw interaction data or item embeddings across domains. We introduce a lightweight adapter that projects the source-domain user representation into the target-domain embedding space and integrates it with the target interaction sequence. Importantly, when source data is unavailable (i.e., the user observed only in the target domain), the model seamlessly defaults to a robust target-only sequential recommender, retaining benefits from joint training across all users. Extensive experiments were conducted under two setups; one targets benchmarking against previous baselines, while the other focuses on realistic and scalable conditions. Both sets of experiments demonstrate consistent performance gains for shared and unshared users.

## CCS Concepts

• Information systems → Recommender systems; Collaborative filtering; Recommender systems; • Computing methodologies → Neural networks.

Keywords   
recommender systems, cross-domain recommendations, sequential   
recommendations, federated learning ACM Reference Format:   
Abdulaziz Samra, Nikita Sukhorukov, Alexey Grishanov, Alexey Vasilev, and Evgeny Frolov. 2026. Cross-Domain Sequential Recommendation with Representation-Level User Transfer. In Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM ’26), November 07–11, 2026, Rome, Italy. ACM, New York, NY, USA, 11 pages. https://doi.org/10.1145/3799682.3841135

## 1 Introduction

Sequential recommender systems have become a central component of modern information platforms, enabling personalized prediction ofa user’s next interaction based on their historical behavior. Recent advances in deep sequential models, such as Transformer-based architectures, have significantly improved recommendation quality within a single domain [14, 24, 26, 35]. However, in many real-world scenarios, user interaction histories in an individual domain are sparse or incomplete, limiting the efectiveness of even the most expressive sequential models.

Cross-domain sequential recommendation (CDSR) addresses this limitation by transferring knowledge across domains in which the same users interact with diferent item spaces, such as e-commerce, media consumption, or online services [7]. By leveraging behavioral signals from a source domain, CDSR methods seek to improve recommendation performance in a target domain, particularly for users with limited interaction histories. Existing CDSR approaches commonly rely on shared item spaces, joint training on merged interaction data, or pretraining and fine-tuning strategies. While efective in controlled settings, these methods often require strong assumptions that are dificult to satisfy in practice, such as user/item alignment across domains or unrestricted access to raw interaction data [18, 30, 34]. Moreover, parameter-transfer approaches may suffer from negative transfer or catastrophic forgetting when domain distributions difer substantially [13].

Practical deployment of CDSR systems, therefore, faces two fundamental challenges. First, interaction data transfer across domains is often prohibited due to storage or data ownership concerns, making centralized modeling of raw interaction sequences from diferent domains infeasible [25, 32]. We formalize this setting as a vertical federated learning (VFL) problem [28], where domains possess distinct feature spaces (item sets) but partially overlap ping users, and only intermediate representations rather than raw interaction sequences may be exchanged. Second, user overlap between domains is typically partial, and many users may have no interaction history in the source domain at all [36]. Efective CDSR models must consequently operate under restricted data sharing assumptions and remain robust to missing source-domain information.

To address these challenges, we propose FLADRec (Federated Learning Adapted Recommender), a representation-level crossdomain sequential recommendation framework. Our key idea is to transfer cross-domain knowledge exclusively through learned user representations, rather than raw interactions or shared item embeddings. In the proposed framework, the source-domain encoder produces a fixed-dimensional user representation from its interaction sequence.

To enable efective transfer, we introduce a lightweight fusing mechanism that conditions target-domain sequence modeling on a projected source-domain user representation. The transferred representation is aligned to the target embedding space and injected directly into the target sequence, allowing the target encoder to adapt its internal dynamics based on cross-domain user preferences. This design enforces strict separation between domains while en abling early and flexible integration of source-domain information.

Importantly, the proposed framework naturally supports users without source-domain interaction histories. In such cases, the model reduces to a standard single-domain sequential recommender, while still benefiting from joint training with users for whom crossdomain information is available. As a result, the approach improves recommendation quality not only for shared users, but also for users observed exclusively in the target domain.

We conduct extensive experiments on multiple cross-domain sequential recommendation benchmarks, demonstrating that the proposed method consistently outperforms strong single-domain and cross-domain baselines.

Our main contributions are:

• We propose FLADRec, a framework for CDSR based purely on representation-level transfer. FLADRec transfers only fixed-dimensional user embeddings, without merging raw interaction sequences across domains, enabling deployment under vertical federated learning constraints.

• We design an efective conditional fusion mechanism that integrates projected source-domain user representations into target-domain sequence modeling, enabling adaptive preference transfer under strict data separation.

• We demonstrate the capability of FLADRec to enhance recommendations for all target users, including those with no source-domain history through implicit regularization, even without explicit cross-domain transfer at inference.

## 2 Proposed Approach

Figure 1 illustrates the proposed cross-domain sequential recommendation framework, highlighting the forward pass of user representations through domain-specific encoders and the crossdomain fusion mechanism.

## 2.1 Problem Description

The core objective is to transfer user preference signals from a source domain to a target domain while respecting practical constraints on data sharing. In particular, we assume that raw interaction sequences cannot be exchanged across domains due to distributed storage or even privacy restrictions; instead, only learned user representations are allowed to be transferred.

Formally, given a user � with an interaction sequence in chronological order (from old to new) in the source domain:

$$
{ \cal S } _ { u } ^ { S } = \{ i _ { 1 } ^ { u , S } , \dots , i _ { n } ^ { u , S } \} ,
$$

and a sequence in chronological order in the target domain:

$$
{ \cal S } _ { u } ^ { T } = \{ i _ { 1 } ^ { u , T } , \dots , i _ { m } ^ { u , T } \} ,
$$

the task is to predict the users’ � next target-domain item $i _ { m + 1 } ^ { u , T }$ Importantly, while $\mathcal { S } _ { u } ^ { S }$ may be absent for users exclusive to the target domain, our framework seamlessly handles both shared and nonshared users.

## 2.2 Encoder-Agnostic User Representations

To ensure the generality of the framework, we do not assume a specific architecture for the sequential encoders. Any model capable of producing a fixed-dimensional representation of a user’s interaction history can be employed, including state-of-the-art sequential recommenders such as SASRec [14], BERT4Rec [24], or other Transformer- or RNN-based architectures [12].

In the source domain, the item sequence $\mathcal { S } _ { u } ^ { S }$ is first mapped to embeddings $\{ \mathbf { e } _ { 1 } ^ { u , S } , \ldots , \mathbf { e } _ { n } ^ { u , S } \}$ using a (frozen) source item embedding matrix. These embeddings are processed by the source encoder, yielding a compact source-domain user representation $\mathbf { e } _ { u } ^ { S }$

Similarly, in the target domain, the item sequence $\mathcal { S } _ { u } ^ { T }$ is mapped to embeddings $\{ \mathbf { e } _ { 1 } ^ { u , T } , \bar { \dots } , \mathbf { e } _ { m } ^ { u , T } \}$ }, which serve as inputs to the targetside processing pipeline.

## 2.3 Fusing layer for cross-domain transfer

To inject source-domain knowledge into the target-domain sequence modeling, we introduce a Fusing Layer, as shown in Fig. 1. This layer operates solely on the transferable source user representation $\mathbf { e } _ { u } ^ { S } ,$ thereby ensuring that user interactions remain isolated and independent across diferent domains.

Specifically, the source representation is first normalized using Layer Normalization and then projected through a multi-layer perceptron (MLP) to obtain $\mathbf { e } _ { u } ^ { S , p }$ which lies in the same latent space as the target item embeddings. This projection enables alignment between source-domain user semantics and target-domain item representations.

![](images/a712aa03e72f808cd304965f3501a646f04603568fbe90699d273ac0c00c2778.jpg)  
Figure 1: The FLADRec architecture. The Source Domain (Green, upper) and Target Domain (Red, lower) streams process user sequences in parallel. Fusing Layer maps the source sequence representation $\mathbf { e } _ { u } ^ { S }$ to the target space as $\mathbf { e } _ { u } ^ { S , p }$ . This projected vector is then fused element-wise with the target item embeddings $\{ \mathbf { e } _ { 1 } ^ { u , T ^ { - } } , \dots , \mathbf { e } _ { m } ^ { u , T } \}$ , acting as a global user context. The fused embeddings $\{ \mathbf { e } _ { \dots } ^ { u , f T } \}$ are fed into the target encoder for final prediction $\mathbf { e } _ { u } ^ { T } .$ . Icons denote trainable (Flame) vs. frozen (Snowflake) parameters. Note: If a user has no source history, the fusion step is skipped.

The projected source user representation is then added element wise to each target item embedding:

$$
\mathbf { e } _ { k } ^ { f T } = \mathbf { e } _ { k } ^ { u , T } + \mathbf { e } _ { u } ^ { S , p } , ~ k = 1 , \ldots , m ,\tag{1}
$$

resulting in a sequence offused target item embeddings $\{ \mathbf { e } _ { 1 } ^ { f T } , \hdots , \mathbf { e } _ { m } ^ { f T } \}$ that are enriched with cross-domain user information.

These fused embeddings are subsequently passed to the target encoder, which produces the final target-domain user representation $e _ { u } ^ { T }$ used for next-item prediction.

## 2.4 Handling missing source-domain histories

A key property of the proposed framework is its ability to jointly improve performance for users with and without source-domain interaction histories.

For users lacking source-domain data, the model naturally reduces to a standard single-domain sequential recommender, as the fusion step is skipped and the target encoder operates on unmodified target item embeddings. Crucially, however, these users still benefit indirectly from the training procedure.

During joint training, the target encoder is exposed to both: fused sequences, where target items are conditioned on transferred source preferences, and unfused sequences, corresponding to users without source histories.

As a result, the target encoder learns a representation space that is simultaneously compatible with conditioned and unconditioned inputs. This shared optimization encourages the model to learn more robust and generalizable sequential patterns in the target domain, improving its ability to model user behavior even when no source-domain information is available.

From a representation learning standpoint, the presence of fused sequences acts as a form of implicit regularization, guiding the target encoder toward preference-aware dynamics without imposing additional constraints at inference time. Consequently, the framework not only supports the partial-overlap scenario, but can also improve recommendation quality for users observed exclusively in the target domain.

## 2.5 Vertical Federated Learning

Although users’ embeddings transferred from the source are valuable to enhance recommendations in the target domain, they are originally encoded to solve a diferent task (usually next-item prediction in the source domain). However, these embeddings may not be optimal for the task of cross-domain knowledge transfer. In our proposed solution, we fine-tune the source encoder to generate task-specific embeddings for fusion in the target domain using a vertical federated learning scheme. Vertical Federated Learning (VFL) refers to the federated learning paradigm in which multiple parties share the same set of users but contain diferent feature spaces [10, 28], which are the item catalogs in our case.

Let $f _ { S } ( \cdot ; \theta _ { S } )$ denote the source-domain sequential encoder and $f _ { T } ( \cdot ; \theta _ { T } )$ the target-domain recommendation model. For a minibatch of users $\mathcal { U } _ { b }$ sampled from the target dataset, we first encode the source interaction sequences for shared users. The source encoder produces user embeddings

$$
\mathbf { e } _ { u } ^ { S } = f _ { S } ( S _ { u } ^ { S } ; \theta _ { S } ) , \quad u \in \mathcal { U } _ { b } \cap \mathcal { U } _ { S } ,\tag{2}
$$

which are transferred to the target domain and used as auxiliary inputs for the target model. The target model optimizes a cross

entropy loss <sup>1</sup>

$$
\mathcal { L } _ { T } = \mathcal { L } \big ( f _ { T } ( S _ { u } ^ { T } , \mathbf { e } _ { u } ^ { S } ; \boldsymbol { \theta } _ { T } ) \big ) ,\tag{3}
$$

where $\mathcal { S } _ { u } ^ { T }$ denotes the interaction sequence of user � in the target domain. During backpropagation, gradients with respect to the transferred embeddings are first computed on the target side

$$
{ \bf g } _ { u } = \frac { \partial \mathcal { L } _ { T } } { \partial { \bf e } _ { u } ^ { S } } ,\tag{4}
$$

and then explicitly propagated back to the source encoder by treating $\mathbf { g } _ { u }$ as an external gradient signal. Using the chain rule, gradients of source encoders w.r.t. the target loss are computed as:

$$
\nabla _ { \boldsymbol { \theta _ { S } } } \mathcal { L } _ { T } = \mathbf { g } _ { u } \cdot \frac { \partial f _ { S } ( S _ { u } ^ { S } ; \boldsymbol { \theta _ { S } } ) } { \partial \boldsymbol { \theta _ { S } } } .\tag{5}
$$

This gradient-bridging mechanism enables the source encoder to be optimized directly with respect to the target-domain objective, without sharing raw interaction data. Source and target parameters $( \theta _ { S } , \theta _ { T } )$ are updated jointly using independent optimizers, while batches without shared users revert to standard target-only training process. This procedure allows the learned source representations to align with the target recommendation task under a distributed cross-domain training protocol. The overall workflow is illustrated in Figure 2.

![](images/e76aa8972bcf7c6feac151a88c2cd465db2abf77ae680a3807b183b5acef6120.jpg)  
Figure 2: Vertical federated learning protocol for crossdomain sequential recommendation. Throughout, raw interaction data and item IDs remain private to each domain; only embeddings and their gradients are exchanged.

Table 1: Statistics of the CDSR benchmark datasets adopted from the same experimental setup from prior work [4].
<table><tr><td>Dataset</td><td>Users</td><td>Items</td><td>Inter.</td><td>Val Seq.</td><td>Test Seq.</td><td>Density</td></tr><tr><td>Food</td><td>7144</td><td>11837</td><td>83663</td><td>2837</td><td>2419</td><td>0.098%</td></tr><tr><td>Kitchen</td><td></td><td>16258</td><td>89885</td><td>4307</td><td>4725</td><td>0.077%</td></tr><tr><td rowspan="2">Beauty Electronics</td><td rowspan="2">4474</td><td>10379</td><td>50329</td><td>2086</td><td>1875</td><td>0.108%</td></tr><tr><td>14188</td><td>63800</td><td>2388</td><td>2599</td><td>0.100%</td></tr><tr><td>Movie</td><td>28350</td><td>35712</td><td>347650</td><td>11728</td><td>10935</td><td>0.034%</td></tr><tr><td>Book</td><td></td><td>90958</td><td>403147</td><td>16622</td><td>17415</td><td>0.015%</td></tr></table>

## 3 Experiment Setup

In this section we describe the experimental setup. We evaluate our method in two complementary settings to balance reproducibility against prior work and realism in deployment conditions.

## 3.1 Data and Splitting

We conduct our benchmarking experiments on three datasets used by previous work [4]. These datasets are derived from the Amazon reviews datasets<sup>2</sup> [11, 20], representing three pairs of domains: Food-Kitchen (FK), Beauty-Electronics (BE), and Movie-Book (MB). These datasets serve as established testbeds that enable direct comparison with published baselines under identical preprocessing, user filtering, and evaluation protocols (e.g., sampled metrics with 999 negatives, last-item split, and full user overlap). While essential for reproducibility, such benchmarks exhibit known limitations: (i) they are not native cross-domain datasets, interactions originate from a single platform (Amazon), and domain separation is imposed post hoc based on product categories, (ii) artificially high user overlap (only shared users retained), (iii) evaluation via negative sampling—which distorts ranking quality [5, 17], and (iv) nontemporal or inconsistent data splits that risk temporal leakage [9]. We elaborate further on these pitfalls in Section 7.2, recommending critical guidelines to advance benchmarking of CDR models.

To address these shortcomings and assess performance under realistic conditions, we additionally evaluate on two industrialscale, multi-domain datasets: TenRec [31] and Zvuk&MegaMarket (Zvuk&MM) [23]. Both capture authentic cross-domain user behavior across distinct services (e.g., music streaming and e-commerce) with naturally partial user overlap and million-scale item catalogs. Crucially, we employ full-catalog ranking without negative sampling and enforce strict temporal splits, thereby eliminating key biases inherent in benchmarking evaluations. As summarized in Tables 1 and 2, this dual-evaluation strategy allows us to validate competitiveness against prior approaches on standard benchmarks, and demonstrate robustness and scalability in settings that reflect real-world cross-domain recommendation challenges.

3.1.1 Benchmarking Datasets. For Amazon-based benchmarks, we follow ABXI [4] exactly: items with fewer than five interactions are filtered out, user histories are truncated to the most recent 50 interactions, and only users appearing in both domains are retained, resulting in 100% user overlap by construction. The last interaction per user is used for testing, the penultimate for validation, and the remainder for training.

Table 2: Statistics of the large-scale CDSR datasets used in the scalability-oriented evaluation. We consider a unidirectional transfer setting, where the first domain in each pair serves as the source and the second as the target.
<table><tr><td>Domains</td><td>Users</td><td>Items</td><td>Inter.</td><td>Shared</td><td>Overlap</td></tr><tr><td>QK-Article</td><td>449,288</td><td>180,951</td><td>14,887,703</td><td>224,644</td><td>50.0%</td></tr><tr><td>→ QK-Video</td><td>449,288</td><td>694,609</td><td>11,784,281</td><td></td><td></td></tr><tr><td>Zvuk</td><td>95,952</td><td>822,274</td><td>38,723,635</td><td>47,976</td><td>50.0%</td></tr><tr><td> MegaMarket</td><td>95,952</td><td>453,367</td><td>4,754,096</td><td></td><td></td></tr></table>

3.1.2 Scalability-OrientedDatasets. We filter users with fewer than 10 interactions. To model realistic partial overlap, we randomly subsample non-shared users to achieve a 50% shared-user ratio across domains, therefore reflecting typical cross-platform scenarios.

Zvuk&MegaMarket. For the Zvuk&MM dataset, we employ a global temporal split strategy. The final week of interaction data is reserved for testing, the penultimate week is utilized for validation, and the remaining historical data comprises the training set. This split ensures there are no data temporal leakage [9].

TenRec. As the TenRec dataset lacks explicit timestamps and defines item consumption order solely at the user level, we adopt a user-stratified splitting approach. Each user’s history is partitioned sequentially: the initial 80% of interactions are allocated for training, the subsequent 10% for validation, and the final 10% for testing.

## 3.2 Training

Learning Objective. We formulate the recommendation task as next-item prediction, where the model predicts a user’s next interaction given their historical sequence. For benchmarking datasets, we optimize models using the full Cross-Entropy (CE) loss [15]. However, for large-scale datasets with extensive item catalogs, full CE becomes computationally infeasible. Therefore, for scalability experiments, we adopt the Scalable Cross-Entropy loss [22] during both the pre-training and adaptation stages.

Initialization and Federated Adaptation. The source and target encoder weights are initialized from the best pretrained singledomain SASRec models to leverage domain-specific sequential patterns prior to adaptation. To enable cross-domain knowledge transfer without mixing user interactions across domains, we employ vertical federated learning, where the source encoder is fine-tuned using gradients propagated from the target domain.

Optimization and Training Protocol. All models are trained using the AdamW optimizer with a batch size of 256. Training proceeds until convergence, with early stopping applied using a patience of 10 epochs based on validation NDCG@10.

The source code and implementation details are publicly available in an anonymous repository<sup>3</sup>.

## 3.3 Evaluation and Metrics

We evaluate performance using two standard metrics for sequential recommendation: Normalized Discounted Cumulative Gain (NDCG@�) and Hit Ratio (HR@�) with � ∈ {10, 50, 100}. We also report Coverage@� (COV@�) to measure the ratio of the whole item catalog covered by recommended items, providing insight into recommendation diversity and long-tail item promotion.

Following established benchmarking practice for the Amazon datasets [4], we compute metrics via sampled evaluation with 999 negatives per positive item to enable fair comparison with prior work under identical conditions.

For industrial datasets, to ensure a rigorous assessment, metrics are calculated via full-ranking over the entire item catalog. This rigorous evaluation eliminates sampling bias [5, 17] and provides a more accurate assessment of model capabilities in real-world scenarios where systems must rank millions of items.

To ensure robust and reproducible results, we conduct hyperparameter optimization on the validation split using the Optuna framework with 30 trials per model configuration, optimizing for NDCG@10. Following hyperparameter selection, all models are retrained on the combined training and validation sets to maximize data utilization before final evaluation on the held-out test set.

## 3.4 Baselines

To comprehensively evaluate the proposed framework, we compare its performance against both strong single-domain and representative cross-domain sequential recommendation models. All baselines are implemented under the same data splits, evaluation protocols, and privacy constraints where applicable.

Single-domain: we utilize SASRec [14] trained exclusively on the target-domain data as our primary single-domain baseline. We also report BERT4Rec results [24]. For each of these SD models, two settings were reported: (1) Single-Target (ST), where hyperparameters are chosen according to the score computed separately within each individual domain, and (2) Dual-Target (DT), where hyperparameters are selected based on the combined scores obtained across both domains [4].

Cross-domain: FedDCSR [33], which enables federated knowledge transfer between domains through disentangled sequence representation learning. CD-SASRec [2] aggregates source-domain sequences into target encoding via dual self-attention. CD-ASR [1] uses separate self-attention encoders for each domain. MGCL [27] integrates graphical and sequential information under diferent views and strengthens the profiling via user-to-user contrastive learning on views. C<sup>2</sup>DSR [6] leverages domain-specific and domaininvariant contrastive tasks. DREAM [29] transfers knowledge from domain-specific to cross-domain representations with contrastive learning. ABXI [4] uses task-guided LoRA (Low-Rank Adaptation) modules to extract domain-invariant user interests<sup>4</sup>.

## 4 Results

We present a comprehensive evaluation of FLADRec on both standard benchmark datasets and large-scale industrial settings, comparing against strong single-domain and cross-domain baselines under consistent experimental conditions.

Table 3: Benchmark results against models running under limited access to source interaction data
<table><tr><td colspan="2"></td><td colspan="2">Single Domain</td><td colspan="4">Cross-Domain (Disjoint-data)</td></tr><tr><td>Domain</td><td>Metric</td><td>SASRec-1</td><td>BERT4Rec-1</td><td>CD-SASRec</td><td>FedDCSR</td><td>CD-ASR</td><td>FLADRec</td></tr><tr><td rowspan="4">Food</td><td>HR@5</td><td>0.1930</td><td>0.1819</td><td>0.1797</td><td>0.1337</td><td>0.1976</td><td>0.2303</td></tr><tr><td>HR@10</td><td>0.2611</td><td>0.2528</td><td>0.2454</td><td>0.2274</td><td>0.2727</td><td>0.3043</td></tr><tr><td>NDCG@10</td><td>0.1561</td><td>0.1462</td><td>0.1421</td><td>0.1214</td><td>0.1616</td><td>0.1851</td></tr><tr><td>MRR</td><td>0.1332</td><td>0.1230</td><td>0.1197</td><td>0.0994</td><td>0.1368</td><td>0.1486</td></tr><tr><td rowspan="4">Kitchen</td><td>HR@5</td><td>0.1241</td><td>0.1114</td><td>0.1119</td><td>0.0987</td><td>0.1345</td><td>0.2241</td></tr><tr><td>HR@10</td><td>0.1851</td><td>0.1685</td><td>0.1757</td><td>0.1454</td><td>0.1995</td><td>0.2891</td></tr><tr><td>NDCG@10</td><td>0.1040</td><td>0.0926</td><td>0.0946</td><td>0.0754</td><td>0.1107</td><td>0.1926</td></tr><tr><td>MRR</td><td>0.0900</td><td>0.0810</td><td>0.0821</td><td>0.0668</td><td>0.0941</td><td>0.1668</td></tr><tr><td rowspan="4">Beauty</td><td>HR@5</td><td>0.1837</td><td>0.1687</td><td>0.1605</td><td>0.1358</td><td>0.1661</td><td>0.2747</td></tr><tr><td>HR@10</td><td>0.2597</td><td>0.2438</td><td>0.2530</td><td>0.2091</td><td>0.2550</td><td>0.3616</td></tr><tr><td>NDCG@10</td><td>0.1523</td><td>0.1404</td><td>0.1380</td><td>0.1061</td><td>0.1424</td><td>0.2135</td></tr><tr><td>MRR</td><td>0.1295</td><td>0.1197</td><td>0.1162</td><td>0.0872</td><td>0.1211</td><td>0.1696</td></tr><tr><td rowspan="4">Electronics</td><td>HR@5</td><td>0.1345</td><td>0.1277</td><td>0.1290</td><td>0.1180</td><td>0.1355</td><td>0.2062</td></tr><tr><td>HR@10</td><td>0.1894</td><td>0.1832</td><td>0.1842</td><td>0.1708</td><td>0.1938</td><td>0.2627</td></tr><tr><td>NDCG@10</td><td>0.1111</td><td>0.1053</td><td>0.1069</td><td>0.0939</td><td>0.1122</td><td>0.1776</td></tr><tr><td>MRR</td><td>0.0982</td><td>0.0930</td><td>0.0948</td><td>0.0825</td><td>0.0987</td><td>0.1575</td></tr><tr><td rowspan="4">Movie</td><td>HR@5</td><td>0.2258</td><td>0.2329</td><td>0.2347</td><td>0.2154</td><td>0.2352</td><td>0.2678</td></tr><tr><td>HR@10</td><td>0.2961</td><td>0.3105</td><td>0.3117</td><td>0.2886</td><td>0.3052</td><td>0.3487</td></tr><tr><td>NDCG@10</td><td>0.1647</td><td>0.1927</td><td>0.1940</td><td>0.1866</td><td>0.1956</td><td>0.2260</td></tr><tr><td>MRR</td><td>0.1874</td><td>0.1696</td><td>0.1709</td><td>0.1743</td><td>0.1592</td><td>0.1977</td></tr><tr><td rowspan="4">Book</td><td>HR@5</td><td>0.1357</td><td>0.1638</td><td>0.1710</td><td>0.1265</td><td>0.1622</td><td>0.1859</td></tr><tr><td>HR@10</td><td>0.1789</td><td>0.2152</td><td>0.2253</td><td>0.1578</td><td>0.2118</td><td>0.2122</td></tr><tr><td>NDCG@10</td><td>0.1007</td><td>0.1378</td><td>0.1434</td><td>0.1030</td><td>0.1372</td><td>0.1716</td></tr><tr><td>MRR</td><td>0.1147</td><td>0.1243</td><td>0.1285</td><td>0.0929</td><td>0.1244</td><td>0.1661</td></tr></table>

## 4.1 Benchmarking results

In these experiments, we split the compared baselines according to whether each domain has access to the interaction data of the other domain, yielding two groups. The first group comprises singledomain baselines trained independently on each domain’s data, namely SASRec-1 and BERT4Rec-1, as well as the cross-domain models CD-SASRec, FedDCSR, and CD-ASR. In contrast, the second group consists of the same single-domain baselines trained on the union of the two domains’ datasets, denoted as SASRec-2 and BERT4Rec-2. In addition, the other cross-domain models –MGCL, C<sup>2</sup>DSR, DREAM, and ABXI– explicitly incorporate mixed raw interaction data from both domains within their pipelines. We recall that our model belongs to the first category.

Table 3 shows the performance comparison of our proposed approach against the first family ofbaselines on benchmark datasets (Food-Kitchen, Beauty-Electronic, and Movie-Book datasets). FLADRec achieves the best overall performance across all dataset pairs and evaluation metrics, achieving the best score in all domains and metrics, with a single exception for the HR@10 on the Book dataset, where it places second. These figures highlight the superiority of FLADRec over other models under the condition of limited access to remote interaction data.

Table 4 reports the comparison between FLADRec and the second group of baselines. Although ABXI achieves marginally higher scores on several benchmarks, FLADRec remains broadly competitive despite operating under a more strict regime that prevent interaction-level data sharing. Notably, the BERT4Rec-2 and SASRec-2 baselines perform slightly worse than the top-performing methods (ABXI and FLADRec), indicating that training strong single-domain models on mixed datasets is a viable compromise when joint training is permissible; however, this approach forgoes additional gains available via adaptation mechanisms (LoRA in ABXI and source-encoder fine-tuning in FLADRec).

## 4.2 Scalability Evaluation

We evaluated the computational eficiency of FLADRec on the benchmark datasets used in our main experiments. As shown in Table 6, FLADRec achieves substantial speedups over ABXI, even though it requires 4 stages of training - pre-training two SASRec models and fine-tuning for each transfer.

Given this significant computational boost, which ranges from 3.68× to 11.57× across diferent domain pairs, we further assess FLADRec’s scalability by evaluating its performance on two largescale, real-world cross-domain datasets. Table 5 reports the results. We compare against SASRec, which serves both as a strong singledomain baseline and as the pretrained initialization for our method.

Across both TenRec and Zvuk&MegaMarket, FLADRec consistently outperforms SASRec across all ranking metrics and cutofs.

Table 4: Benchmark results against models running under full access to source interaction data
<table><tr><td colspan="2"></td><td colspan="2">Single Domain</td><td colspan="3">Cross-Domain (Pooled-data)</td><td>Ours</td></tr><tr><td>Domain</td><td>Metric</td><td>SASRec-2</td><td>BERT4Rec-2</td><td>MGCL</td><td>C²DSR</td><td>DREAM</td><td>ABXI FLADRec</td></tr><tr><td rowspan="4">Food</td><td>HR@5</td><td>0.2313</td><td>0.2223</td><td>0.1932</td><td>0.1984</td><td>0.2158</td><td>0.2498</td><td>0.2303</td></tr><tr><td>HR@10</td><td>0.2854</td><td>0.2956</td><td>0.2673</td><td>0.2574</td><td>0.2771</td><td>0.3175</td><td>0.3043</td></tr><tr><td>NDCG@10</td><td>0.1797</td><td>0.1727</td><td>0.1523</td><td>0.1546</td><td>0.1698</td><td>0.1973</td><td>0.1851</td></tr><tr><td>MRR</td><td>0.1535</td><td>0.1427</td><td>0.1260</td><td>0.1311</td><td>0.1441</td><td>0.1679</td><td>0.1486</td></tr><tr><td rowspan="4">Kitchen</td><td>HR@5</td><td>0.1510</td><td>0.1363</td><td>0.1467</td><td>0.1263</td><td>0.1377</td><td>0.1737</td><td>0.2241</td></tr><tr><td>HR@10</td><td>0.2168</td><td>0.2055</td><td>0.2157</td><td>0.1879</td><td>0.2045</td><td>0.2410</td><td>0.2891</td></tr><tr><td>NDCG@10</td><td>0.1248</td><td>0.1116</td><td>0.1203</td><td>0.1051</td><td>0.1138</td><td>0.1415</td><td>0.1926</td></tr><tr><td>MRR</td><td>0.1062</td><td>0.0948</td><td>0.1017</td><td>0.0903</td><td>0.0956</td><td>0.1206</td><td>0.1668</td></tr><tr><td rowspan="4">Beauty</td><td>HR@5</td><td>0.2292</td><td>0.1970</td><td>0.1364</td><td>0.1835</td><td>0.2090</td><td>0.2807</td><td>0.2747</td></tr><tr><td>HR@10</td><td>0.3262</td><td>0.3077</td><td>0.2109</td><td>0.2645</td><td>0.3043</td><td>0.3835</td><td>0.3616</td></tr><tr><td>NDCG@10</td><td>0.1866</td><td>0.1679</td><td>0.1162</td><td>0.1519</td><td>0.1742</td><td>0.2245</td><td>0.2135</td></tr><tr><td>MRR</td><td>0.1530</td><td>0.1363</td><td>0.1001</td><td>0.1290</td><td>0.1447</td><td>0.1846</td><td>0.1696</td></tr><tr><td rowspan="4">Electronics</td><td>HR@5</td><td>0.1481</td><td>0.1519</td><td>0.1537</td><td>0.1288</td><td>0.1216</td><td>0.1659</td><td>0.2062</td></tr><tr><td>HR@10</td><td>0.2169</td><td>0.2246</td><td>0.2159</td><td>0.1859</td><td>0.1817</td><td>0.2389</td><td>0.2627</td></tr><tr><td>NDCG@10</td><td>0.1236</td><td>0.1275</td><td>0.1273</td><td>0.1081</td><td>0.1023</td><td>0.1385</td><td>0.1776</td></tr><tr><td>MRR</td><td>0.1058</td><td>0.1101</td><td>0.1118</td><td>0.0960</td><td>0.0895</td><td>0.1200</td><td>0.1575</td></tr><tr><td rowspan="4">Movie</td><td>HR@5</td><td>0.2303</td><td>0.2317</td><td>0.2097</td><td>0.2299</td><td>0.2507</td><td>0.2859</td><td>0.2678</td></tr><tr><td>HR@10</td><td>0.3067</td><td>0.3095</td><td>0.2851</td><td>0.3003</td><td>0.3255</td><td>0.3682</td><td>0.3487</td></tr><tr><td>NDCG@10</td><td>0.1903</td><td>0.1925</td><td>0.1726</td><td>0.1911</td><td>0.2082</td><td>0.2388</td><td>0.2260</td></tr><tr><td>MRR</td><td>0.1673</td><td>0.1697</td><td>0.1509</td><td>0.1700</td><td>0.1848</td><td>0.2118</td><td>0.1977</td></tr><tr><td rowspan="4">Book</td><td>HR@5</td><td>0.1356</td><td>0.1547</td><td>0.1248</td><td>0.1316</td><td>0.1469</td><td>0.1973</td><td>0.1859</td></tr><tr><td>HR@10</td><td>0.1830</td><td>0.2063</td><td>0.1668</td><td>0.1767</td><td>0.1973</td><td>0.2571</td><td>0.2122</td></tr><tr><td>NDCG@10</td><td>0.1146</td><td>0.1302</td><td>0.1043</td><td>0.1123</td><td>0.1237</td><td>0.1669</td><td>0.1716</td></tr><tr><td>MRR</td><td>0.1034</td><td>0.1176</td><td>0.0946</td><td>0.1025</td><td>0.1118</td><td>0.1502</td><td>0.1661</td></tr></table>

On TenRec, FLADRec achieves relative improvements in the range (19-24%) in diferent metrics and cutofs. On Zvuk&MegaMarket, improvements are more moderate in NDCG (7.4–9.8%) but remain substantial in HitRate, reaching up to 20.4% at HR@50.

Importantly, FLADRec also enhances recommendation diversity when generating a small number of candidates: it increases Coverage@10 by +6.6%, indicating broader exposure to relevant items early in the ranked list. This balance between accuracy and diversity is critical in industrial settings where long-tail item promotion directly impacts user engagement. However, steady declines in coverage are observed at larger cutofs (@50 and @100), indicating that the recommendations become less diverse when many candidates are generated.

These gains were observed under partial user overlap, full-catalog ranking without negative sampling, and strict temporal splits (only Zvuk&MegaMarket contains timestamps), thereby avoiding evaluation biases common in benchmark settings. The consistent improvements indicate that FLADRec enhances recommendation quality in large item spaces with naturally partial user overlap.

Overall, these results support the benchmark evaluation by demonstrating that FLADRec is efectively built on a strong, widely used single-domain foundation (SASRec) and scales to realistic deployment settings.

## 5 Ablation Study

This section concentrates on the ablation of the components of the proposed approach. In the following ablations, we breakdown the metric computation into three subsets of users (all/shared/unshared). This helps us to understand the efect of each factor in our pipeline on recommendation quality.

## 5.1 Efect of cross-domain embedding transfer

To rigorously evaluate the contribution of source-domain representations to target-domain prediction, we conduct controlled ablation studies where we modify or eliminate the cross-domain signal for shared users. As shown in Figure 1, our framework transfers knowledge exclusively through the representation of the source user $\mathbf { e } _ { u } ^ { S }$ at the output of the source encoder. We test three regimes:

(1) Full transfer, corresponding to the complete FLADRec model; (2) Random injection, where randomly sampled embedding is injected for all users from the same distribution of $\mathbf { e } _ { u } ^ { S } ;$

(3) No transfer, where the fusing layer is bypassed entirely during inference, reducing the model to using only targetdomain sequences for prediction.

However, unlike training a completely isolated single-domain model from scratch, the target encoder’s weights still retain knowledge transferred from the source domain during joint training. This condition isolates the efect of explicit embedding transfer while preserving implicit knowledge transfer through model weights.

Table 5: Results of FLADRec model on large-scale datasets. NDCG, HitRate, Coverage (COV) at cutof � = 10, 50, 100 and relative improvements w.r.t. the SASRec baseline are reported.
<table><tr><td>Dataset</td><td>Metric</td><td>SASRec</td><td>FLADRec (Ours)</td><td>Imp.</td></tr><tr><td rowspan="8">QK-article QK-video</td><td>NDCG@10</td><td>0.0225</td><td>0.0283</td><td>+24.1%</td></tr><tr><td>NDCG@50</td><td>0.0382</td><td>0.0469</td><td>+22.8%</td></tr><tr><td>NDCG@100</td><td>0.0469</td><td>0.0569</td><td>+21.3%</td></tr><tr><td>HR@10</td><td>0.0423</td><td>0.0532</td><td>+25.8%</td></tr><tr><td>HR@50</td><td>0.1155</td><td>0.1395</td><td>+20.8%</td></tr><tr><td>HR@100</td><td>0.1691</td><td>0.2017</td><td>+19.3%</td></tr><tr><td>COV@10</td><td>0.1682</td><td>0.1794</td><td>+6.6%</td></tr><tr><td>COV@50</td><td>0.2634</td><td>0.2601</td><td>-1.2%</td></tr><tr><td rowspan="8">Zvuk Megamarket</td><td>COV@100</td><td>0.3179</td><td>0.3009</td><td>-5.3%</td></tr><tr><td>NDCG@10</td><td>0.0583</td><td>0.0626</td><td>+7.4%</td></tr><tr><td>NDCG@50</td><td>0.0674</td><td>0.0739</td><td>+9.6%</td></tr><tr><td>NDCG@100</td><td>0.0711</td><td>0.0781</td><td>+9.8%</td></tr><tr><td>HR@10</td><td>0.0897</td><td>0.1063</td><td>+18.5%</td></tr><tr><td>HR@50</td><td>0.1306</td><td>0.1573</td><td>+20.4%</td></tr><tr><td>HR@100</td><td>0.1532</td><td>0.1830</td><td>+19.5%</td></tr><tr><td>COV@10</td><td>0.1320</td><td>0.1320</td><td>0.0%</td></tr><tr><td></td><td>COV@50</td><td>0.2321</td><td>0.2135</td><td>-8.7%</td></tr><tr><td>COV@100</td><td></td><td>0.2960</td><td>0.2600</td><td>-13.8%</td></tr></table>

Table 6: Training time (minutes) comparison between ABXI and FLADRec under identical hardware conditions.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">ABXI</td><td colspan="3">FLADRec</td><td rowspan="2">Speedup</td></tr><tr><td>Pretrain</td><td>Adapt</td><td>Total</td></tr><tr><td>Food Kitchen</td><td>42.0</td><td>0.39 1.67</td><td>0.86 0.71</td><td>3.63</td><td>11.57×</td></tr><tr><td>Beauty Electronics</td><td>17.0</td><td>1.18 0.33</td><td>0.94 0.55</td><td>3.00</td><td>5.67×</td></tr><tr><td>Movie Book</td><td>65.6</td><td>5.29 5.5</td><td>2.45 4.58</td><td>17.82</td><td>3.68×</td></tr></table>

The ablation study results for the tuned source encoder are presented in Table 7. The full-transfer (full) setting achieves the best performance in both the source-tuned and source-frozen scenarios. When the source encoder is tuned, random embeddings (rand) slightly outperform the no-embedding baseline (none) for the shared subset of users, likely because they introduce userdiscriminative signals that complement the adapted encoder weights. Not surprisingly, the metrics in the (full) and (none) settings for the unshared users are identical, since no source embeddings exist to transfer for these users, whereas injecting random source embeddings for unshared users degrades performance. Conversely, when the source encoder is frozen, random embeddings perform worse than no embeddings, as the fixed representation space cannot adapt to the injected noise, leading to uncontrolled perturbations.

Table 7: NDCG@10 comparison across user subsets (ALL / Shared / Unshared). Relative gains w.r.t. baseline are reported.
<table><tr><td colspan="3"></td><td colspan="3">User subset</td></tr><tr><td>Model</td><td>Regime</td><td>Metric</td><td>ALL</td><td>Shared</td><td>Unshared</td></tr><tr><td>SASRec</td><td>-</td><td>value</td><td>0.02248</td><td>0.00976</td><td>0.01272</td></tr><tr><td rowspan="4">FLADRec (Tuned Source)</td><td>full</td><td>value</td><td>0.02832</td><td>0.01231</td><td>0.01601 +25.8</td></tr><tr><td></td><td>gain,% value</td><td>+26.0 0.02790</td><td>+26.3 0.01226</td><td>0.01564</td></tr><tr><td>rand</td><td>gain,%</td><td>+24.5</td><td>+25.7</td><td>+23.0</td></tr><tr><td>none</td><td>value gain,%</td><td>0.02814</td><td>0.01213</td><td>0.01601</td></tr><tr><td rowspan="5">FLADRec (Frozen Source)</td><td>full</td><td>value</td><td>+25.2 0.02689</td><td>+24.4 0.01158</td><td>+25.8 0.01531</td></tr><tr><td></td><td>gain,%</td><td>+19.6</td><td>+18.8</td><td>+20.3</td></tr><tr><td>rand</td><td>value</td><td>0.02621</td><td>0.01133</td><td>0.01488</td></tr><tr><td></td><td>gain,%</td><td>+16.6</td><td>+16.2</td><td>+17.0</td></tr><tr><td>none</td><td>value gain,%</td><td>0.02689 +19.6</td><td>0.01158 +18.8</td><td>0.01531 +20.3</td></tr></table>

## 5.2 Ablation of source encoder training

To investigate the necessity of source encoder adaptation in crossdomain transfer, we conduct a controlled ablation study where we freeze the source encoder weights during training (see Fig. 1). In this configuration, the source representation remains fixed as initially pretrained on its native task, while only the target encoder and fusing layer parameters are updated. This experimental setup isolates the contribution ofsource encoder fine-tuning to the overall performance gain.

As shown in Table 7, freezing the source encoder leads to consistent performance degradation across all user subsets and transfer variants. Compared to the tuned-source setting, the frozensource configuration yields lower NDCG@10 for both shared and unshared users, with particularly pronounced drops for shared users under full transfer. These results indicate that projecting fixed, pretrained source representations into the target space—even with an adaptive fusion layer—is insuficient for optimal cross-domain learning.

Overall, this ablation confirms that source encoder adaptation is a critical component ofFLADRec, enabling the source representations to align more efectively with the target-domain semantics and thereby enhancing cross-domain transfer.

Finally, a comparison between the performance improvements obtained by varying the transfer regime (full, random, or none) and those resulting from modifying the source encoder status (trainable versus frozen) indicates that most of performance gain can be attributed to the joint training process of the proposed architecture.

## 6 Related Work

Single-domain recommender systems have achieved significant progress through deep learning architectures like Transformers[14,

24] and self-supervised learning techniques [26, 35], which efictively capture the dynamic user preferences.

For the case of many domains, particularly with partial user overlap, sequential cross-domain recommendation have emerged. These method concentrate on transferring knowledge from source domains to improve recommendation quality in the target do main [7] for the next-item prediction task.

A common line of work assumes the presence of users shared across domains and transfers knowledge through shared user representations. Models such as CD-SASRec [2], DCDIR [3], and MIFN [19] employ domain-aware encoders or attention mechanisms to fuse sequences from multiple domains. More recent CDSR approaches focus on transferring knowledge at the representation level, by aligning latent user embeddings via contrastive objectives or partially sharing model parameters across domains. C<sup>2</sup>DSR [6] utilized contrastive learning through domain-specific and domain-invariant tasks. ABXI [4] uses task-guided LoRA (Low-Rank Adaptation) modules to identify domain-invariant user interests and finetune the models in both source and target domains. AutoCDSR [13] utilizes self-attention to optimize the cross-domain knowledge transfer and combat the negative transfer caused by imbalanced signals from domains. These methods typically rely on joint training over all domains, including the usage of merged interaction sequences across domains and exploit shared users to bridge domain gaps.

The increasing focus on privacy has spurred research into federated recsys, where models are trained across decentralized data sources without exchanging raw user interactions. PPCDR [25] proposes a scheme that shares user embeddings protected by local diferential privacy technique. However, this method provides learning schemes for domains with full user overlap, which rarely happens in real-world services. FedCDR [21] addresses cross-domain recommendation in federated settings but focuses on rating prediction rather than sequential next-item prediction, limiting its applicability to modern sequential recommendation systems.

Our approach bridges critical gaps between cross-domain recommendation and privacy-preserving learning. Unlike prior CDSR methods that require sharing raw sequences or item mappings, our framework transfers knowledge exclusively through distilled user representations, never exchanging raw interactions or embedding tables between domains. Unlike methods that assume full user overlap, our architecture naturally accommodates partial overlap through a fusion mechanism that seamlessly converts to singledomain recommendation when source data is missing.

Our lightweight adapter enables cross-domain knowledge transfer while maintaining strict domain separation, making it deployable in federated learning scenarios where existing methods fail. Furthermore, our framework uniquely benefits all users, not just those with cross-domain activity. Through joint training with shared parameters, target-domain-only users indirectly benefit from the regularization efect of cross-domain signals, achieving performance gains. This dual benefit represents a practical advance toward deployable cross-domain recommendation systems that respect privacy boundaries while maximizing utility.

## 7 Discussion

## 7.1 Privacy Analysis

FLADRec was essentially designed to achieve cross-domain knowledge transfer between diferent subsidiaries (services) under a shared ecosystem. The federated training process operates in a distributed storage environment with mild privacy restrictions. Thus, the proposed architecture lack a dedicated mechanism to prevent user history reconstruction attacks. However, we obtain a representation-transfer mechanism that naturally operates efectively whether representations are exchanged via secure channels (VFL) or directly (non-VFL), without architectural modification. To secure the training process where gradients are transferred back to the source domain, stronger privacy guarantees, such as formal diferential privacy, can be integrated into the federated training protocol.

## 7.2 Toward A Fair, Sustainable CDR Benchmarking Protocol

In this section, we identify and analyze five prevalent pitfalls in benchmarking pipelines for cross-domain recommendation models, which substantially undermine their practical reliability in industrial settings when compared with single-domain baselines. As illustrated in our scalability experiments (Section 4.2), we took steps to avoid as many of these pitfalls as possible.

7.2.1 Origin of Data. Large companies such as Amazon and Douban release anonymized user-item datasets split into categories, which researchers repurpose to simulate cross-domain recommendation. However, these datasets originate from a single platform, where a shared environment and shared models have access to all data simultaneously. Recently, more "honest" datasets have been released, notably the MegaMarket–Zvuk pair [23] and TenRec [31] are collected from genuinely separate platforms and therefore produced by independent models.

7.2.2 Domain Overlap Ratio. Most CDR research assumes shared users or items across domains and links their representations accordingly. In practice, overlap is limited, creating non-uniform conditions under which models must handle shared and non-shared users diferently. Most researchers in the CDR field run experiments on datasets with unrealistically high or even full user overlap, and insuficiently discuss the disparity in quality improvement between shared and unshared users. Notably, recent work has begun addressing quality fairness between shared and non-shared users [8].

7.2.3 Dataset Size and Scalability. Many recent CDR papers still rely on small, heavily processed datasets (<300k interactions), especially when controlling the user overlap ratio. Replicating most such models in real-world settings is typically computationally infeasible. Since recommender systems research is inherently practical, comparisons against baselines evaluated only on small datasets are of limited value, even when those baselines appear to outperform others in that restricted regime.

7.2.4 Sampled Metric Evaluation. Sampled metrics have grown popular because they enable fast computation by ranking relevant items against a random subset of negatives rather than the full catalog. However, this approach can distort model comparisons, and several works have proposed corrections for this bias [16].

7.2.5 Temporal Leakage in Train/Test Split. Sampling each user’s most recently interacted item is a common evaluation practice in recommender systems. It is, however, frequently applied to non-temporal or temporally inconsistent data splits, introducing a risk of temporal leakage [9]. Nevertheless, temporal leakage across multiple domains remains insuficiently investigated to date.

Specifically, the Zvuk/MegaMarket experiment satisfies all the criteria discussed above, but to a lesser extent with respect to the overlap ratio, as a 50% user overlap remains relatively high. In addition, the TenRec experiment also does not meet the requirement of a strict temporal split, since the dataset originally lacks absolute timestamps.

Nonetheless, we argue that enforcing the optimal condition for each of the aforementioned considerations is overly demanding in the context of fast-paced academic research, even though it would promote stronger alignment with industrial applications. We further argue that the partial overlap and scalability conditions are the most critical aspects to prioritize in future CDR research.

## 8 Conclusion and future works.

In this work, we introduced a practical framework for cross-domain sequential recommendation that transfers knowledge through learned user representations, without sharing raw interaction data or item embeddings across domains. Our SASRec-based architecture, combined with a lightweight fusing layer and vertical federated fine-tuning of the source encoder, enables efective alignment of source and target user preferences under realistic privacy and deployment constraints. Extensive experiments across five datasets, including two industrial-scale benchmarks and three established academic benchmarks, demonstrate consistent improvements over strong single-domain and cross-domain baselines, for both shared and target-only users, confirming that representation-level transfer not only avoids negative transfer but also benefits the entire user population. Future work includes extending the framework to multi-source and multi-task settings, exploring alternative fusion mechanisms and backbone architectures, and integrating privacypreserving mechanisms to protect the exchanged data during learning.

## Acknowledgments

This work is an output of a research project (HSE-BR-2025-019) implemented as part of the Basic Research Program at the HSE University.

## GenAI Usage Disclosure

During the preparation of this manuscript, the authors used Claude to rephrase parts of the text and to improve the readability of the open-source code of this work. All experimental design, data collection, statistical analysis, and interpretation were performed entirely by the authors without the use of GenAI. The authors reviewed and edited all AI-assisted text and code modifications and take full responsibility for the final content of the publication.

## References

[1] Nawaf Alharbi and Doina Caragea. 2021. Cross-domain attentive sequentia recommendations based on general and current user preferences (cd-asr). In IEEE/WIC/aCM international conference on web intelligence and intelligent agent technology. 48–55.

[2] Nawaf Alharbi and Doina Caragea. 2021. Cross-domain self-attentive sequential recommendations. In Proceedings ofInternational Conference on Data Science and Applications: ICDSA 2021, Volume 2. Springer, 601–614.

[3] Ye Bi, Liqiang Song, Mengqiu Yao, Zhenyu Wu, Jianming Wang, and Jing Xiao. 2020. A Heterogeneous Information Network based Cross Domain Insurance Recommendation System for Cold Start Users. In Proceedings ofthe 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (Virtual Event, China) (SIGIR ’20). Association for Computing Machinery, New York, NY, USA, 2211–2220. doi:10.1145/3397271.3401426

[4] Qingtian Bian, Marcus de Carvalho, Tieying Li, Jiaxing Xu, Hui Fang, and Yiping Ke. 2025. ABXI: Invariant Interest Adaptation for Task-Guided Cross-Domain Sequential Recommendation. In Proceedings ofthe ACM on Web Conference 2025 (Sydney NSW, Australia) (WWW ’25). Association for Computing Machinery, New York, NY, USA, 3183–3192. doi:10.1145/3696410.3714819

[5] Rocío Cañamares and Pablo Castells. 2020. On Target Item Sampling in Ofline Recommender System Evaluation. In Proceedings of the 14th ACM Conference on Recommender Systems (Virtual Event, Brazil) (RecSys ’20). Association for Computing Machinery, New York, NY, USA, 259–268. doi:10.1145/3383313. 3412259

[6] Jiangxia Cao, Xin Cong, Jiawei Sheng, Tingwen Liu, and Bin Wang. 2022. Contrastive Cross-Domain Sequential Recommendation. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management (Atlanta, GA, USA) (CIKM ’22). Association for Computing Machinery, New York, NY, USA, 138–147. doi:10.1145/3511808.3557262

[7] Shu Chen, Zitao Xu, Weike Pan, Qiang Yang, and Zhong Ming. 2024. A survey on cross-domain sequential recommendation. In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence (Jeju, Korea) (IJCAI ’24). Article 884, 10 pages. doi:10.24963/ijcai.2024/884

[8] Weixin Chen, Yuhan Zhao, Li Chen, and Weike Pan. 2025. Leave no one behind: Fairness-aware cross-domain recommender systems for non-overlapping users. In Proceedings ofthe Nineteenth ACM Conference on Recommender Systems. 226– 236.

[9] Danil Gusak, Anna Volodkevich, Anton Klenitskiy, Alexey Vasilev, and Evgeny Frolov. 2025. Time to Split: Exploring Data Splitting Strategies for Ofline Evaluation of Sequential Recommenders. In Proceedings ofthe Nineteenth ACM Conference on Recommender Systems. 874–883.

[10] Stephen Hardy, Wilko Henecka, Hamish Ivey-Law, Richard Nock, Giorgio Patrini, Guillaume Smith, and Brian Thorne. 2017. Private federated learning on vertically partitioned data via entity resolution and additively homomorphic encryption. ArXiv abs/1711.10677 (2017). https://api.semanticscholar.org/CorpusID:26233593

[11] Ruining He and Julian McAuley. 2016. Ups and Downs: Modeling the Visual Evolution of Fashion Trends with One-Class Collaborative Filtering. In Proceedings ofthe 25th International Conference on World Wide Web (Montréal, Québec, Canada) (WWW ’16). International World Wide Web Conferences Steering Committee, Republic and Canton of Geneva, CHE, 507–517. doi:10. 1145/2872427.2883037

[12] Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. 2016. Session-based Recommendations with Recurrent Neural Networks. arXiv:1511.06939 [cs.LG] https://arxiv.org/abs/1511.06939

[13] Clark Mingxuan Ju, Leonardo Neves, Bhuvesh Kumar, Liam Collins, Tong Zhao, Yuwei Qiu, Qing Dou, Sohail Nizam, Sen Yang, and Neil Shah. 2025. Revisiting Self-attention for Cross-domain Sequential Recommendation. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (Toronto ON, Canada) (KDD ’25). Association for Computing Machinery, New York, NY, USA, 1094–1105. doi:10.1145/3711896.3737108

[14] Wang-Cheng Kang and Julian McAuley. 2018. Self-Attentive Sequential Recommendation. arXiv:1808.09781 [cs.IR] https://arxiv.org/abs/1808.09781

[15] Anton Klenitskiy and Alexey Vasilev. 2023. Turning Dross Into Gold Loss: is BERT4Rec really better than SASRec?. In Proceedings of the 17th ACM Conference on Recommender Systems (Singapore, Singapore) (RecSys ’23). Association for Computing Machinery, New York, NY, USA, 1120–1125. doi:10.1145/3604915. 3610644

[16] Walid Krichene and Stefen Rendle. 2020. On sampled metrics for item recommendation. In Proceedings ofthe 26th ACM SIGKDD international conference on knowledge discovery & data mining. 1748–1757.

[17] Walid Krichene and Stefen Rendle. 2022. On sampled metrics for item recommendation. Commun. ACM 65, 7 (June 2022), 75–83. doi:10.1145/3535335

[18] Qidong Liu, Xiangyu Zhao, Yejing Wang, Zijian Zhang, Howard Zhong, Chong Chen, Xiang Li, Wei Huang, and Feng Tian. 2025. Bridge the Domains: Large Language Models Enhanced Cross-domain Sequential Recommendation. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (Padua, Italy) (SIGIR ’25). Association for Computing Machinery, New York, NY, USA, 1582–1592. doi:10.1145/3726302. 3729911

[19] Muyang Ma, Pengjie Ren, Zhumin Chen, Zhaochun Ren, Lifan Zhao, Peiyu Liu, Jun Ma, and Maarten de Rijke. 2022. Mixed Information Flow for Cross-Domain Sequential Recommendations. ACM Trans. Knowl. Discov. Data 16, 4, Article 64 (Jan. 2022), 32 pages. doi:10.1145/3487331

[20] Julian McAuley, Christopher Targett, Qinfeng Shi, and Anton van den Hengel. 2015. Image-Based Recommendations on Styles and Substitutes. In Proceedings ofthe 38th International ACM SIGIR Conference on Research and Development in Information Retrieval (Santiago, Chile) (SIGIR ’15). Association for Computing Machinery, New York, NY, USA, 43–52. doi:10.1145/2766462.2767755

[21] Wu Meihan, Li Li, Chang Tao, Eric Rigall, Wang Xiaodong, and Xu Cheng-Zhong. 2022. FedCDR: Federated Cross-Domain Recommendation for Privacy-Preserving Rating Prediction. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management (Atlanta, GA, USA) (CIKM ’22). Association for Computing Machinery, New York, NY, USA, 2179–2188. doi:10. 1145/3511808.3557320

[22] Gleb Mezentsev, Danil Gusak, Ivan Oseledets, and Evgeny Frolov. 2024. Scalable Cross-Entropy Loss for Sequential Recommendations with Large Item Catalogs. In Proceedings ofthe 18th ACM Conference on Recommender Systems. 475–485.

[23] Valeriy Shevchenko, Nikita Belousov, Alexey Vasilev, Vladimir Zholobov, Artyom Sosedka, Natalia Semenova, Anna Volodkevich, Andrey Savchenko, and Alexey Zaytsev. 2024. From Variability to Stability: Advancing RecSys Benchmarking Practices. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (Barcelona, Spain) (KDD ’24). Association for Computing Machinery, New York, NY, USA, 5701–5712. doi:10.1145/3637528. 3671655

[24] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer. In Proceedings ofthe 28th ACM International Conference on Information and Knowledge Management (Beijing, China) (CIKM ’19). Association for Computing Machinery, New York, NY, USA, 1441–1450. doi:10.1145/3357384.3357895

[25] Changxin Tian, Yuexiang Xie, Xu Chen, Yaliang Li, and Xin Zhao. 2024. Privacypreserving Cross-domain Recommendation with Federated Graph Learning. ACM Trans. Inf. Syst. 42, 5, Article 135 (May 2024), 29 pages. doi:10.1145/3653448

[26] Xu Xie, Fei Sun, Zhaoyang Liu, Shiwen Wu, Jinyang Gao, Bolin Ding, and Bin Cui. 2021. Contrastive Learning for Sequential Recommendation. arXiv:2010.14395 [cs.IR] https://arxiv.org/abs/2010.14395

[27] Zitao Xu, Shu Chen, Weike Pan, and Zhong Ming. 2025. A multi-view graph contrastive learning framework for cross-domain sequential recommendation. ACM Transactions on Recommender Systems 3, 4 (2025), 1–28

[28] Qiang Yang, Yang Liu, Tianjian Chen, and Yongxin Tong. 2019. Federated machine learning: Concept and applications. ACM Transactions on Intelligent Systems and

Technology (TIST) 10, 2 (2019), 1–19.

[29] Xiaoxin Ye, Yun Li, and Lina Yao. 2023. DREAM: Decoupled Representation via Extraction Attention Module and Supervised Contrastive Learning for Cross Domain Sequential Recommender. In Proceedings ofthe 17th ACM Conference on Recommender Systems (Singapore, Singapore) (RecSys ’23). Association for Computing Machinery, New York, NY, USA, 479–490. doi:10.1145/3604915. 3608780

[30] Mingjia Yin, Hao Wang, Wei Guo, Yong Liu, Zhi Li, Sirui Zhao, Zhen Wang, Defu Lian, and Enhong Chen. 2024. Learning Partially Aligned Item Representation for Cross-Domain Sequential Recommendation. arXiv:2405.12473 [cs.IR] https: //arxiv.org/abs/2405.12473

[31] Guanghu Yuan, Fajie Yuan, Yudong Li, Beibei Kong, Shujie Li, Lei Chen, Min Yang, Chenyun Yu, Bo Hu, Zang Li, Yu Xu, and Xiaohu Qi. 2022. Tenrec: a large-scale multipurpose benchmark dataset for recommender systems. In Proceedings of the 36th International Conference on Neural Information Processing Systems (New Orleans, LA, USA) (NIPS ’22). Curran Associates Inc., Red Hook, NY, USA, Article 834, 14 pages.

[32] Hao Zhang, Mingyue Cheng, Qi Liu, Junzhe Jiang, Xianquan Wang, Rujiao Zhang, Chenyi Lei, and Enhong Chen. 2025. A Comprehensive Survey on Cross-Domain Recommendation: Taxonomy, Progress, and Prospects. arXiv:2503.14110 [cs.IR] https://arxiv.org/abs/2503.14110

[33] Hongyu Zhang, Dongyi Zheng, Xu Yang, Jiyuan Feng, and Qing Liao. 2024. FedDCSR: Federated Cross-domain Sequential Recommendation via Disentangled Representation Learning. Society for Industrial and Applied Mathematics, 535–543. doi:10.1137/1.9781611978032.62

[34] Chuang Zhao, Hongke Zhao, Ming HE, Jian Zhang, and Jianping Fan. 2023. Crossdomain recommendation via user interest alignment. In Proceedings ofthe ACM Web Conference 2023 (Austin, TX, USA) (WWW’23). Association for Computing Machinery, New York, NY, USA, 887–896. doi:10.1145/3543507.3583263

[35] Kun Zhou, Hui Wang, Wayne Xin Zhao, Yutao Zhu, Sirui Wang, Fuzheng Zhang, Zhongyuan Wang, and Ji-Rong Wen. 2020. S3-Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management (Virtual Event, Ireland) (CIKM ’20). Association for Computing Machinery, New York, NY, USA, 1893–1902. doi:10.1145/3340531.3411954

[36] Yongchun Zhu, Zhenwei Tang, Yudan Liu, Fuzhen Zhuang, Ruobing Xie, Xu Zhang, Leyu Lin, and Qing He. 2022. Personalized Transfer of User Preferences for Cross-domain Recommendation. In Proceedings ofthe Fifteenth ACM International Conference on Web Search and Data Mining (Virtual Event, AZ, USA) (WSDM ’22). Association for Computing Machinery, New York, NY, USA, 1507–1515. doi:10.1145/3488560.3498392