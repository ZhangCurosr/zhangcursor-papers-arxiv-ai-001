# Don’t Rewrite the SID Map: Topology Preservation for Generative Retrieval in Dynamic E-Commerce Search

Jianbo Zhu<sup>✉</sup> Taobao & Tmall Group of Alibaba Hangzhou, China zhujianbo.zjb@taobao.com

Mingmin Jin Taobao & Tmall Group of Alibaba Hangzhou, China jimmy.jmm@taobao.com

Xing Fang   
Nankai University   
Tianjin, China   
Taobao & Tmall Group of Alibaba   
Hangzhou, China   
fangxing.fx@taobao.com

Jing Wang Taobao & Tmall Group of Alibaba Hangzhou, China jing.wangj1@taobao.com

Jiaqi Ye Taobao & Tmall Group of Alibaba Hangzhou, China guike.yjq@taobao.com

Xiaopeng Chu University of Science and Technology of China Hefei, China cxphhh@mail.ustc.edu.cn

Bokang Wang   
Taobao & Tmall Group of Alibaba Hangzhou, China   
wangbokang.wbk@taobao.com

Xiang Zhang Taobao & Tmall Group of Alibaba Hangzhou, China heiqiang.zx@taobao.com

## Abstract

Generative retrieval assigns each item a discrete semantic identifier (SID) and retrieves items by autoregressively generating SID sequences. In dynamic environments such as e-commerce, new items arrive continuously and business signals keep evolving, requiring periodic updates to the SID-to-Item mapping and model to maintain retrieval timeliness. Each mapping update, however, can invalidate search paths already learned by the retriever model, and the quantitative relationship between the degree of mapping change and retrieval quality has not been systematically characterized. We study this problem with extensive controlled experiments over 67 days of large-scale e-commerce search logs and obtain three core findings. (1) SID semantic drift causes severe retrieval degradation that may be masked by large beam sizes: mapping updates reduce HitRate by 41.78-62.05% at beam size 1 but only 8.33-10.85% at beam size 1000, as redundant candidate generation conceals the true severity of drift. (2) The fraction of changed SIDs is strictly monotonically related to retrieval quality: as the remapped fraction of old-item SIDs increases from 0% to 72.75%, HitRate decreases monotonically, with a drop of up to 42.66% at beam size 1. Under the same warm-start training procedure, full remapping remains substantially worse than the preservation policy, indicating that the model has internalized the implicit topology of the SID space and that short-budget continued training alone cannot repair topology disruption. (3) The SID topology-preserving principle makes even the simplest warm-start strategy highly efective, and models finetuned from previous checkpoints outperform naive model souping: fixing old-item SIDs, inserting new-items, and warm-starting from

![](images/5e5669ce654e06eec4fc921d45f933022ee4276a48a32802bc5deb4b73861720.jpg)

the previous model improves old-item HitRate by approximately 10% and new-item HitRate by approximately 109%, validated across two non-overlapping update cycles. These findings distill into a concise deployment principle: insert new-items, freeze old-items; warm-start when possible, otherwise freeze; never change SIDs the model has already learned. A three-week online A/B test of this principle on the mobile e-commerce platform TmallAPP shows UCTCVR +0.29% and GMV +1.16%.

## CCS Concepts

• Information systems → Information retrieval; Retrieval models and ranking.

## Keywords

Generative Retrieval, Semantic Identifiers, E-Commerce Search, Semantic Drift, Dynamic Update

## ACM Reference Format:

Jianbo Zhu, Xing Fang, Xiaopeng Chu, Mingmin Jin, Jing Wang, Bokang Wang, Jiaqi Ye, and Xiang Zhang. 2018. Don’t Rewrite the SID Map: Topology Preservation for Generative Retrieval in Dynamic E-Commerce Search. In Proceedings of Make sure to enter the correct conference title from your rights confirmation email (Conference acronym ’XX). ACM, New York, NY, USA, 10 pages. https://doi.org/XXXXXXX.XXXXXXX

## 1 Introduction

Generative retrieval reformulates item retrieval as a sequence generation task: given a user query, a language model autoregressively generates a short discrete token sequence, known as a semantic identifier (SID), which is resolved to one or more items through an external mapping. Since the diferentiable search index [24] was proposed, this paradigm has demonstrated advantages in many scenarios [2, 3, 6, 8, 21, 30], ofering end-to-end training, elimination of large-scale vector index maintenance, and natural integration of multi-objective ranking signals.

Beyond the initial item SID construction, a more pressing challenge in dynamic settings is how to maintain the SID-to-Item mapping over time. In e-commerce search, a highly dynamic retrieval environment and the item set expands daily: new items are listed, prices fluctuate, and conversion rates evolve, all of which require periodic updates to the retriever model and SID-to-Item mapping. In production-grade systems in particular, SID encodings often incorporate a business-value dimension [15], making the mapping unavoidably time-varying. After the model is trained on a historical mapping, it may generate the same SID at inference time, yet that SID now resolves to a diferent item under the updated mapping—a phenomenon we call semantic drift. A deeper issue is that prior work has predominantly pursued collision-free SID assignments; in dynamic settings, however, the continuously growing item set inevitably introduces collisions with existing items. Forcing a full remapping to eliminate these collisions changes the SIDs of many existing items, triggering severe semantic drift.

In practice, semantic drift is easy to overlook. At a large beam size, the model can still recover target items by exploring many al ternative branches even when high-probability paths have become stale; ofline evaluation at a single wide beam may therefore suggest that a mapping update is safe, while latency-constrained serving at a narrow beam sufers a far larger loss. Meanwhile, retraining is often assumed to repair the refreshed mapping. However, this assumption neglects the structural knowledge the model has already encoded during training, particularly the learned topology of the SID space.

These considerations raise some deployment-oriented questions. How much damage does a SID mapping refresh cause, and how does the observed damage depend on beam size? Does retrieval quality vary systematically with the fraction of old items whose SIDs change? Can warm-start training repair the damage from remapping, or should the system preserve the old SID topology and only insert new items?

Rather than proposing another algorithmic remedy, we take a step back and answer these questions empirically through extensive controlled experiments over 67 days of large-scale e-commerce search logs. Unlike prior studies that evaluate a single proposed method against baselines, we systematically vary the mapping policy as an experimental factor, spanning the full spectrum from complete remapping to complete freezing. By independently controlling the training strategy, this design enables a clean disentanglement of mapping-induced damage from training-induced repair. Our contributions are:

• Damage and masking efect of semantic drift. We systematically quantify the degradation caused by SID mapping evolution and reveal that wide-beam evaluation masks the true severity of drift. Specifically, while mapping updates cause a notable reduction in HitRate and MRR at beam size 1000, the degradation is substantially more severe at beam size 1.

• Monotonic efect and topology irreparability. We discover a strict monotonic negative relationship between the fraction of changed old-item SIDs � and retrieval quality. Under the same warm-start budget, full remapping remains substantially worse than the preservation policy, indicating that the model has internalized the implicit topology of the SID space and that the simple warm-start regime does not repair topology disruption.

• Topology-preservation principle and optimal update policy. We find that topology preservation is a more fundamental determinant than retraining alone, and propose a simple incremental warm-start strategy feasible at industrial scale (freeze old items, insert new items, warm-start the model periodically). Experiments show approximately 9% improvement on old items and 109% on new items; a three-week online A/B test yields UCTCVR +0.29% and GMV +1.16%.

Four findings are particularly counter-intuitive. First, evaluations conducted with wide beams can mask the vast majority of the actual drift severity, meaning that many production systems may be operating under a false sense of safety. Second, the relationship between remapping fraction and quality loss is strictly monotonic, leaving no useful middle ground between full remapping and com plete freezing. Third, warm-start training, the default repair strategy assumed by most continual-learning approaches, cannot recover from topology disruption: Full-Remapping after warm-start still underperforms a no-retrain Frozen baseline. Fourth, rather than creating a ’model soup’ by mixing the old and new models, it is more efective to directly use the new model trained from the old one. Together, these findings suggest that the problem prior work has been solving, namely how to repair mapping-induced damage, is better addressed by not creating the damage in the first place.

## 2 Related Work

## 2.1 Generative Retrieval

Diferentiable Search Index [24] and GENRE [2] established the paradigm of directly generating document or entity identifiers, replacing nearest-neighbor search with autoregressive sequence generation. Since then, generative retrieval has been applied across document and passage retrieval [2, 24], podcast discovery [6], recommendation ranking [8, 21, 22], and e-commerce search [3, 15, 27, 30]. TIGER [21] demonstrated that residual-quantized semantic identifiers can efectively capture item semantics for retrieval, and since its introduction, a wave of work has adopted semantic IDs in ecommerce. OneRec [8] and OneSearch [3] propose an end-to-end framework. In e-commerce search scenarios, CQ-SID [30] introduces query-item contrastive learning and EG-GRPO for searchrelevant clusters. Ling et al. [15] encode business-value ranking into SID assignments, and Zhan et al. [27] propose a unified generative framework that integrates retrieval and pre-ranking within a single model. These eforts focus on initial model design; the update-time implications of evolving item collections remain open.

## 2.2 Semantic Identifier Construction

The quality of semantic identifiers is a first-order determinant of generative retrieval performance. RQ-VAE-based methods [3, 21, 30] produce hierarchical codes through residual vector quantization of item embeddings. Alternative approaches include RQ-KMeans [8], multi-identifier pretraining [28], and self-improving tokenization [5]. A key concern is how quantization distorts neighborhood structure: topology-aware tokenization [16], qualification aware collision handling [12], and topology-preserving tokenization [29] all indicate that identifier structure afects downstream retrieval. Hybrid semantic/hash identifiers [17] and constrained decoding [23] address serving eficiency, while FORGE [10] provides a systematic benchmark for evaluating SID construction methods. DREAM [11] further identifies premature static SID commitment as a cold-start bottleneck and proposes counterfactual contrastive learning for progressive refinement. These eforts focus on constructing higher-quality initial SIDs; however, even when the initial design is efective, the impact of subsequent mapping evolution on retrieval quality remains insuficiently characterized.

## 2.3 Dynamic Updates and Continual Learning

DSI++ [18] frames document addition as continual learning and documents catastrophic forgetting during sequential updates. CLEVER [4] and PromptDSI [14] address dynamic corpora under limited replay. More recent work directly targets semantic ID staleness: Baikalov et al. [1] propose bijective alignment mappings to mitigate collaborative-SID drift, Feng et al. [9] introduce drift-confidence-based selective tokenization to minimize unnecessary code changes. Alternative paradigms avoid weight updates entirely: Den et al. [7] reframe incremental retrieval as an in-context generation problem, Mekonnen et al. [19] attach parametric memory heads to frozen models, and Huynh et al. [13] use dynamically expandable LoRA experts. These methods mitigate mapping staleness through increasingly sophisticated algorithmic designs such as alignment solvers, selective reassignment, and parameter-eficient adapters. However, most of these approaches introduce additional architectural complexity.

Although recent methods have recognized the issue of drift and proposed algorithmic remedies, no study has treated the mapping policy as an independent experimental factor for disentangling the damage caused by mapping changes themselves from the efectiveness of repair algorithms. Nor has the monotonic relationship between the fraction of changed mappings and retrieval quality been established. More fundamentally, a simpler yet prerequisite question, whether the mapping should be changed at all, has been largely overlooked.

## 3 Method

We study how to update a generative retrieval system deployed in ecommerce search when the item set expands and business-value signals shift. During long-term operation in real-world environments, the deployed retriever gradually becomes outdated as new items arrive and the commercial importance of existing items changes. Directly refreshing the Item-SID mapping to catch up with changes will destroy the autoregressive decoding path learned by the model, resulting in the decline of retrieval quality. Our goal is to identify update policies that accommodate new items and evolving value signals while keeping the learned mappings for existing items as stable as possible, avoiding a full rebuild-and-retrain pipeline.

## 3.1 Problem Formulation

Let I denote the item set. Each item $i \in \mathcal { I }$ is assigned a three-token semantic identifier

$$
\begin{array} { r } { \mathrm { S I D } ( i ) = \left[ s _ { 1 } , s _ { 2 } , s _ { 3 } \right] . } \end{array}\tag{1}
$$

Given the query � and user profile �, the retrieval model $f _ { \theta }$ generates the SID of the target item autoregressively and is trained by minimizing

$$
\mathcal { L } ( \boldsymbol { q } , \boldsymbol { u } , i ) = - \sum _ { k = 1 } ^ { 3 } \log P _ { \boldsymbol { \theta } } ( s _ { k } \mid s _ { < k } , \boldsymbol { q } , \boldsymbol { u } ) .\tag{2}
$$

At inference, beam search produces the top-K SID sequences, and a serving-time mapping $M : \mathrm { S I D }  I$ resolves each sequence to one or more items.

The retriever is inherently tied to the SID space through its token embeddings and output distributions. Let $M _ { t _ { 0 } }$ be the mapping used to label the training data and $M _ { t _ { 1 } }$ the mapping at inference time. A drift event occurs when $M _ { t _ { 0 } } \neq M _ { t _ { 1 } }$ . We measure its extent over old items as

$$
\gamma = \delta ( M _ { t _ { 0 } } , M _ { t _ { 1 } } ) = \frac { 1 } { | \mathcal { T } | } \sum _ { i \in \mathcal { I } } \mathbb { I } \big [ M _ { t _ { 0 } } ^ { - 1 } ( i ) \neq M _ { t _ { 1 } } ^ { - 1 } ( i ) \big ] .\tag{3}
$$

In the experiments, we denote the observed remapped fraction as $\gamma$ evaluated over the old-item set.

The central tension is stability versus freshness. Several routine operations can trigger mapping drift: refreshing value-based positions to track shifting market conditions, retraining an RQ-VAE or other item-to-SID clustering model after a large batch of new items alters the codebook, or more generally any procedure that reassigns existing items to diferent SIDs. Each of these invalidates part of the mapping learned by $f _ { \theta } .$ . Unlike flat class labels, each SID in a generative retrieval model corresponds to a sequence of autoregressive decoding decisions: the model first chooses $s _ { 1 }$ , then � conditioned on $s _ { 1 } ,$ and so on until the full identifier is produced. Every Item-SID association is therefore not a single label but a learned decoding path through a tree-structured output space. When a mapping update reassigns items to diferent SIDs, it does not merely relabel training data; it redistributes output probability across previously learned paths, potentially invalidating entire branches of the decoding tree. Topology preservation, keeping item-to-SID assignments unchanged, ensures that these learned paths remain valid. A wide beam may partially compensate for drift by exploring additional branches, whereas a narrow beam commits earlier and is more vulnerable.

## 3.2 SID Construction

The first two tokens (� , � ) form a semantic prefix obtained by quantizing item-content embeddings with an RQ-VAE trained under the CQ-SID query-item (QI) contrastive framework [30]. The QI objective augments the standard VQ-VAE reconstruction and commitment losses with a bidirectional InfoNCE term that aligns itemcontent and query representations in a shared embedding space, so that the resulting codebook captures search-relevant semantic clusters rather than content-only neighborhoods. The codebook size is set to 8192x8192.

![](images/2c63268b05b7a3edbd944b400d4ada84b89e8878f515e005822cfcf96bb184a3.jpg)  
Figure 1: Framework overview. Top: Three-level SID construction, two semantic layers from RQ-VAE with QI contrastive learning produce prefix $( s _ { 1 } , s _ { 2 } ) ;$ a business-value layer appends $s _ { 3 }$ by ranking items with engagement signals. Bottom: Retrieval pipeline from embedding through SID construction, LLM autoregressive generation, beam search, and SID-to-item map resolution. There are two divergent types of update paths: the red paths rewrite old item SIDs and use a soup model (combining old and new models), which degrades metrics; the green paths freeze old SIDs, insert new items, and directly apply the new model fine-tuned from the old one, improving performance on both old and new items.

Within each semantic prefix, the third token $s _ { 3 }$ encodes a businessvalue prior. Following [15], we compute

$$
\boldsymbol { v } ( i ) = \frac { 0 . 1 \cdot \mathrm { C l i c k } ( i ) + 0 . 3 \cdot \mathrm { C a r t } ( i ) + 1 . 0 \cdot \mathrm { B u y } ( i ) } { \mathrm { E x p } ( i ) + \epsilon } , \quad \epsilon = 1 0 ,\tag{4}
$$

where Exp(�) denotes the exposure count of item $i ,$ and Click, Cart, and Buy denote click, add-to-cart and purchase counts, respectively. Items within each prefix are ranked by �(�), so that the third-level token directly reflects commercial importance.

Hybrid collision-aware assignment. In a collection of production items with continuous ingestion, the number of items under each semantic prefix grows monotonically over time. Maintaining a bijective item-to-token mapping in a bounded vocabulary is therefore infeasible: as new items accumulate, collisions (multiple items sharing the same identifier) become inevitable regardless of the initial codebook size. Rather than attempting to eliminate all collisions through an ever-expanding vocabulary, we adopt a hybrid collisionaware (HC) design that concentrates exact resolution where it matters most. For each prefix $P = \left( { { s _ { 1 } } , { s _ { 2 } } } \right)$ , items are sorted by � (�) and the top � items receive unique third-level tokens, $s _ { 3 } ( i _ { j } ) = j$ for $j \in \{ 1 , \ldots , K \}$ ; remaining items share bucket identifiers in fixedsize groups. We set $K = 1 0 0$ and allocate up to 1024 additional collision buckets for the tail. As a reference, the full-sparse (FS) design assigns one unique identifier per item, requiring up to 8192 third-level buckets. HC preserves exact resolution for the highvalue head while gracefully accommodating the long tail.

Non-invasive new-item insertion. Once a new item accumulates suficient exposure to become eligible for the item pool, it first receives a semantic prefix from the RQ-VAE quantizer and is then inserted into the value ranking of its corresponding cluster. The item is inserted directly at its value-rank position, regardless of whether that position falls in the top-� head or the collision-bucketed tail. The top-� dedicated assignment is maintained only in the trainingtime SID set; at insertion time, a new item landing in the head may share a previously dedicated token with an existing item, and no old-item SID is modified. This ensures that new-item insertion has zero impact on the existing SID space.

## 3.3 Retriever Model

The retriever is built on Qwen2.5-0.5B [20] and training follows the three-stage pipeline of CQ-SID [30]:

(1) Item→SID. The model learns to generate item SIDs conditioned on item content alone, establishing the basic mapping from content representations to the SID space.

(2) Query→SID. Query text is incorporated as an input to train the model to generate target item SIDs. This is achieved by leveraging query-SID associations derived from the majority of users’ query-item interaction sequences.

(3) User+Query→SID. User profile and interaction history are prepended to the query, enabling personalized retrieval that conditions SID generation on both user context and query intent.

## 3.4 Update Policies

Each update cycle poses two independent decisions: which items receive new SIDs (the mapping policy) and whether the model is retrained to accommodate the updated mapping (the training strategy). We deliberately focus on simple, interpretable options rather than sophisticated continual-learning methods, for two reasons. First, production systems favor reliable, debuggable solutions over algorithmic complexity. Second, because the autoregressive retriever learns query-to-SID-path associations, any mapping change structurally invalidates part of these learned associations; the mapping policy is therefore expected to be the dominant factor, while the training procedure plays a supporting role.

Mapping policies. We consider three policies that span the stability– freshness spectrum. Full remapping (Full-Re) reassigns third-level tokens to all items within each prefix using the latest value-rank; a large fraction of terminal token assignments change. Frozen mapping keeps every old item’s SID unchanged; eligible new items receive an available unique or shared-bucket token under their RQ VAE prefix without modifying any existing assignment. Thresholded remapping interpolates between the two: an old item is remapped only when its rank displacement exceeds a threshold �,

$$
\vert \mathrm { r a n k } _ { t _ { 1 } } ( i ) - \mathrm { r a n k } _ { t _ { 0 } } ( i ) \vert > t .\tag{5}
$$

Varying � from 0 to ∞ traces a continuum from Full-Re to Frozen mapping, allowing us to study the dose–response relationship between the remapping fraction � and retrieval quality.

Training strategies. Given any mapping policy, the model either remains unchanged (no-retrain), or is continued from the previous checkpoint (warm-start) and trained on accumulated new query– SID pairs labeled with the updated mapping. Warm-start training allows the model to learn associations involving newly inserted items and to reinforce existing ones under the current mapping. During warm-start updates, only retriever model training stage 3 in Section 3.3 is re-executed.

The two dimensions combine independently, allowing diferent mapping policies to be evaluated with or without warm-start training. Figure 1 contrasts the two extremes: Full-Re (red path) rewrites the terminal token assignments consumed by the existing decoder, causing topology disruption and degrading both HitRate and MRR; Frozen mapping with warm-start training (green path) extends the mapping with new items only and leverages warm-start training to improve retrieval quality for both old and new items.

## 4 Experiments

## 4.1 Experimental Setup

Datasets. All experiments use query–item interaction logs from a large-scale e-commerce search platform. The observation win dow spans 67 days. The first 30 days train the initial model, the following 30 days form the first update and evaluation cycle, and the final 7 days form a subsequent non-overlapping update cycle initialized from the Day-30 model. Training interactions are split strictly before their evaluation events.

Implementation Details. The full-sparse (FS) codebook has size 8192 × 8192 × 8192 across the three SID levels; the hybrid collisionaware is 8192×8192×1124. The RQ-VAE is trained with exponential moving average (EMA) updates and a codebook reset strategy to stabilize quantizer training. The LLM retriever is trained on approximately 90 million logs for the 30-day cycle and 20 million for the 7-day cycle. All training runs on 64 GPUs with batch size 64 and gradient accumulation over 4 steps (efective batch size 256). The initial model is trained for 10000 steps at learning rate $4 \times 1 0 ^ { - 5 } \colon$ warm-start updates use a lower learning rate of $1 \times 1 0 ^ { - 5 }$ , with 5000 steps for the 30-day cycle and 1200 steps for the 7-day cycle.

Evaluation. For each query, beam search generates candidate SID sequences, the current map resolves them to items, and HitRate records whether the target item is present. We report HR@� for � ∈ {1, 10, 100, 1000} and MRR@1000. Day-1 denotes the first day following a model trained on the prior 30 days of data, which corresponds to the 31st day of the overall cycle. Subsequent ref erences such as Day-30 and Day-37 are calculated relative to this Day-1 baseline. The same SID-expansion and metric computation are used across temporal variants. The experiments are organized around SID design, mapping drift, change-rate sensitivity, model adaptation, weight mixing, and cycle-to-cycle reproducibility.

## 4.2 Collisions Are Not the Bottleneck

Before investigating mapping drift, we first verify that the SID design itself, specifically the controlled collisions inherent in productionscale systems, does not already dominate retrieval quality.

We compare the full-sparse (FS) and hybrid-collision (HC) designs using the Day-1 evaluation. As shown in Table 1, HC consistently outperforms FS across all beam sizes, achieving HR@1 = 0.0589 (+0.86%), HR@1000 = 0.7183 (+0.11%), and MRR = 0.1199 (+1.52%). The table further reports the number of resolved items per query. Together with the uniformly higher HitRate and MRR, these results suggest that the collision mechanism efectively concentrates probability mass on relevant SID paths, while the bucket structure for tail items recovers additional candidates once those paths are selected. Moreover, the median number of retrieved items for HC at beam 1 is identical to that of FS (both 1) and only marginally higher at beam 1000 (986 vs. 966, the quantity is less than 1,000 because certain retrieved SIDs have no associated items.), indicating that controllable collisions do not cause the model to indiscriminately favor large clusters.

## 4.3 Wide Beams Mask Mapping Drift

Next, we keep the item set fixed and investigate whether updating each item’s third-level SID token (based on its business value observed on Day 30) afects retrieval performance. We decouple two independently refreshable components: Map, the serving-time SID-to-item lookup table, and Feat, the SID representations within the model’s input interaction sequences. In the result tables, a check mark indicates that a component is refreshed to its latest version, whereas a cross indicates that it remains frozen.

Table 1: Day1 comparison of FS and HC identifiers: HitRate, MRR, and resolved items per query.
<table><tr><td colspan="7">Retrieval Quality</td><td colspan="3">Resolved Items</td></tr><tr><td>SID</td><td>HR@1</td><td>HR@10</td><td>HR@100</td><td>HR@1000</td><td>MRR</td><td>Avg@B1</td><td>Med@B1</td><td>Avg@B1000</td><td>Med@B1000</td></tr><tr><td>FS</td><td>0.0584</td><td>0.2409</td><td>0.5134</td><td>0.7175</td><td>0.1181</td><td>1</td><td>1</td><td>925</td><td>966</td></tr><tr><td>HC</td><td>0.0589</td><td>0.2457</td><td>0.5215</td><td>0.7183</td><td>0.1199</td><td>1</td><td>1</td><td>1200</td><td>986</td></tr></table>

Table 2: Old-item metrics on Day30 without adding new items. ✓: refreshed; ×: frozen.
<table><tr><td>SID</td><td>Map</td><td>Feat</td><td>HR@1</td><td>HR@10</td><td>HR@100</td><td>HR@1000</td><td>MRR</td></tr><tr><td>FS</td><td>X</td><td>X</td><td>0.0585</td><td>0.2324</td><td>0.4927</td><td>0.7002</td><td>0.1158</td></tr><tr><td>FS</td><td>√</td><td>√</td><td>0.0319</td><td>0.1268</td><td>0.3262</td><td>0.6333</td><td>0.0647</td></tr><tr><td>FS</td><td>√</td><td>X</td><td>0.0222</td><td>0.0982</td><td>0.2890</td><td>0.6242</td><td>0.0492</td></tr><tr><td>HC</td><td>X</td><td>X</td><td>0.0584</td><td>0.2344</td><td>0.5005</td><td>0.7035</td><td>0.1164</td></tr><tr><td>HC</td><td>√</td><td>√</td><td>0.0340</td><td>0.1394</td><td>0.3535</td><td>0.6449</td><td>0.0706</td></tr><tr><td>HC</td><td>√</td><td>X</td><td>0.0226</td><td>0.1056</td><td>0.3172</td><td>0.6373</td><td>0.0519</td></tr></table>

Table 2 compares the fully frozen baseline (where both the map and feature representations are kept static) against two partialrefresh configurations. Refreshing both components significantly degrades performance, with HC HR@1 dropping from 0.0584 to 0.0340 (-41.78%) and MRR falling from 0.1164 to 0.0706 (-39.35%).

Notably, refreshing only the map while keeping input-sequence features frozen is even more detrimental than refreshing both. The two configurations difer in where the inconsistency lies. When both are refreshed (✓/✓), the model’s conditioning context and the resolution table are mutually consistent, so the freshly generated SIDs and the map that resolves them describe the same value-rank; the residual loss stems only from the mismatch against the model’s learned weights. When only the map is refreshed (✓/×), the model still emits SIDs consistent with the stale input features, but the serving map now resolves those SIDs under the new value-rank—so a decoding path that was coherent under the old features is repointed to diferent items, compounding a feature–map mismatch on top of the weight–map mismatch. The additional degradation therefore isolates the cost of internal inconsistency between the input representation and the resolution table, and reinforces our central argument: it is the reassignment of learned paths, not the freshness of the value signal per se, that drives drift.

The same comparison reveals substantially smaller degradation at beam size 1000: the HR@1000 decreases are 8.33% and 9.41% for the two HC refresh configurations. Across FS and HC, the narrowbeam HR@1 degradation is 41.78–62.05%, whereas the wide-beam degradation is only 8.33-10.85%. MRR, which averages over all ranks within beam 1000, shows an intermediate degradation of 39.35– 55.41%, confirming that drift primarily damages high-confidence paths. The observations are consistent with wide beams recovering some stale paths through redundant exploration and thereby concealing part of the failure mode.

Since HC consistently outperforms FS, we adopt HC as the default SID design for all subsequent experiments. Having established that controlled collisions do not degrade retrieval quality, the remaining source of performance loss in dynamic settings is the mapping update itself, which we investigate next.

## 4.4 Remapping Rate Predicts Degradation

Before studying how remapping afects old items, we first confirm whether these two strategies will have diferent impacts on new project retrieval.

We examine new-item retrieval without model retraining. Table 3 compares Full-Re (all items remapped by Day-30 value) with Frozen (old SIDs frozen, new items inserted by value). The two policies produce nearly identical metrics across all beam sizes and MRR, indicating that new-item retrieval is insensitive to how old-item SIDs are represented in the input features.

Table 3: New-item HitRate and MRR (Day30).
<table><tr><td>Policy</td><td>HR@1</td><td>HR@10</td><td>HR@100</td><td>HR@1000</td><td>MRR</td></tr><tr><td>Full-Re</td><td>0.0187</td><td>0.0828</td><td>0.2499</td><td>0.5133</td><td>0.0417</td></tr><tr><td>Frozen</td><td>0.0180</td><td>0.0835</td><td>0.2507</td><td>0.5127</td><td>0.0415</td></tr></table>

We then vary threshold � to control the remapping fraction � of the old items. The SID map and input sequence features are refreshed consistently. Table 4 shows a strictly monotonic pattern: as � decreases from 72.75% to 0%, HitRate and MRR increase at every beam size. Even the moderate threshold � = 10 remains 12.4% below complete freezing at beam size 1. Within the tested policies, partial remapping does not yield a useful stability-freshness compromise.

Under full remapping (Full-Re), all metrics degrade monotonically: HR@1 drops by 42.66% and MRR by 40.34%, while HR@1000 exhibits the smallest decline of only 8.35%, consistent with the masking efect observed in Section 4.3.

Notably, inserting new items with frozen old SIDs (Frozen, � = 0) raises old-item HR@1000 from 0.7035 (Table 2, without new items) to 0.7043, confirming that adding new-item signals to the feature sequence benefits old-item retrieval as well.

Table 4: Old-item metrics versus the remapped fraction �.
<table><tr><td>Policy</td><td>γ</td><td>HR@1</td><td>HR@10</td><td>HR@100</td><td>HR@1000</td><td>MRR</td></tr><tr><td>Full-Re</td><td>72.75%</td><td>0.0332</td><td>0.1368</td><td>0.3523</td><td>0.6455</td><td>0.0692</td></tr><tr><td>t = 10</td><td>29.08%</td><td>0.0507</td><td>0.1996</td><td>0.4332</td><td>0.6568</td><td>0.1004</td></tr><tr><td>t = 100</td><td>7.36%</td><td>0.0553</td><td>0.2196</td><td>0.4694</td><td>0.6712</td><td>0.1097</td></tr><tr><td>t = 500</td><td>1.65%</td><td>0.0557</td><td>0.2217</td><td>0.4749</td><td>0.6769</td><td>0.1106</td></tr><tr><td>Frozen</td><td>0%</td><td>0.0579</td><td>0.2343</td><td>0.5011</td><td>0.7043</td><td>0.1160</td></tr></table>

![](images/08e91df1b71346bbe1ec89754b6ba415edfbe2f09ee1200ba18a3064c548de9c.jpg)  
Figure 2: Monotonic remapping efect. Metric decline rate (%) relative to the frozen baseline as the remapping threshold becomes more aggressive.

## 4.5 Warm-Start Does Not Repair Remapping

We now warm-start both Full-Re and Frozen from the same Day-1 model and train on the same accumulated Day1–30 data. Only the mapping policy difers. Table 5 shows higher observed HitRate and MRR for Frozen on both old and new items. At beam size 1, Frozen HitRate improves over Full-Re by 36.70% on old items and 13.94% on new items; MRR also shows a large gap (35.50%/+9.56% for old/new items). For old items, Full-Re after training still underperforms the no-retrain Frozen baseline at beam size 1000 (0.6923 vs. 0.7043 on Table 4) and on MRR (0.0941 vs. 0.1160).

Compared with the no-retrain policy, Frozen raises old-item HR@1000 from 0.7043 to 0.7272 and MRR from 0.1160 to 0.1276; new-item HR@1000 rises from 0.5127 to 0.6200 and MRR from 0.0415 to 0.0779. These observations are consistent with mapping preservation protecting established paths while warm-start training learns new Query-SID associations.

Table 5: Warm-start comparison after the 30-day cycle.
<table><tr><td>Items</td><td>Policy</td><td>HR@1</td><td>HR@10</td><td>HR@100</td><td>HR@1000</td><td>MRR</td></tr><tr><td rowspan="2">Old</td><td>Full-Re</td><td>0.0466</td><td>0.1867</td><td>0.4309</td><td>0.6923</td><td>0.0941</td></tr><tr><td>Frozen</td><td>0.0637</td><td>0.2597</td><td>0.5355</td><td>0.7272</td><td>0.1276</td></tr><tr><td rowspan="2">New</td><td>Full-Re</td><td>0.0330</td><td>0.1474</td><td>0.3684</td><td>0.6193</td><td>0.0711</td></tr><tr><td>Frozen</td><td>0.0376</td><td>0.1590</td><td>0.3833</td><td>0.6200</td><td>0.0779</td></tr></table>

The results support a structural rather than purely optimizationbased explanation. With Frozen and Full-Re, the initialization and accumulated training data are held fixed, and the mapping policy is the principal experimental diference. Frozen preserves established Query2SID associations, whereas Full-Re rewrites many SID assignments. A finite warm-start run must therefore relearn many old SID decisions while also learning new items. This interpretation is consistent with two empirical observations: degradation grows monotonically with $\gamma ,$ as expected if a larger remapped fraction invalidates more learned SID decisions; and the smaller degradation at a large beam is consistent with alternative-path exploration compensating for some mapping errors.

![](images/a70a20b44348f188d709ddd31b5424d66280aa1f335be38f25a25ccc796107f5.jpg)

![](images/3638335fcc6f5e0cb529113a2e9d346c629d444a07714c0981f9f7d5bd9de959.jpg)  
Figure 3: Warm-start training improvement over frozen baseline (Day-30). Relative change (%) for Full-Re and Frozen compared with the frozen (no-retraining) policy.

The efect of the warm-start training can be seen from Figure 3. For old items (left), Full-Re degrades HR@1 by −19.5% and MRR by −18.9%, while Frozen improves all three metrics. For new items (right), Frozen outperforms Full-Re on every metric, with HR@1 gaining +108.9% over frozen versus +83.3% for Full-Re (the noretrain baseline corresponds to � = 1.0 in Table 6).

Scope ofthis claim. In practical industrial environments, simple and reliable methods are generally preferred. Accordingly, our warm-start mechanism adopts a straightforward training strategy: it loads the parameters of the previously trained model and then fine-tunes them on the newly arrived data. Therefore, we do not claim that topological changes are inherently irreversible; a more robust continual learning framework may partially recover performance. Nevertheless, in the present context, preserving the original topology remains the optimal solution.

## 4.6 No Benefit from Weight Mixing

Warm-start training naturally raises a follow-up question at the parameter level. The previous checkpoint $\theta _ { o l d }$ encodes stable associations for old items, whereas the warm-started checkpoint $\theta _ { n e w }$ additionally encodes fresh associations for newly inserted items. A standard, training-free way to trade of this stability–plasticity tension is to interpolate the two checkpoints at inference time, in the spirit of model-soup [25] and WiSE-FT-style weight averaging [26]; such interpolation is attractive in production because it needs no extra training and is tuned by a single scalar. We test whether weight mixing ofers any additional benefit—that is, whether a useful middle ground exists at the parameter level. For Frozen policy, we interpolate the old and newly trained parameters at inference time:

$$
\theta ( \alpha ) = \alpha \theta _ { \mathrm { o l d } } + ( 1 - \alpha ) \theta _ { \mathrm { n e w } } .\tag{6}
$$

Table 6 reports the 30-day cycle. Old-item quality changes moderately across interpolation weights, whereas new-item quality depends strongly on the new checkpoint. A plausible explanation is that unchanged old-item SIDs are represented in both checkpoints, while new Query-SID associations appear only during the update; prediction-distribution analysis is needed to verify this hypothesis. The pure new checkpoint (� = 0) has the highest or near-highest observed value at every reported beam and on MRR for both item groups. At beam size 1, it raises new-item HitRate from 0.0180 to 0.0376 relative to the frozen old model, representing a more than twofold improvement. The tested interpolation settings provide no observed benefit.

Table 6: Weight-mixing ablation for Frozen after 30 days.
<table><tr><td>Items</td><td>α</td><td>HR@1</td><td>HR@10</td><td>HR@100</td><td>HR@1000</td><td>MRR</td></tr><tr><td rowspan="4">Old</td><td>0.0</td><td>0.0637</td><td>0.2597</td><td>0.5355</td><td>0.7272</td><td>0.1276</td></tr><tr><td>0.1</td><td>0.0631</td><td>0.2590</td><td>0.5321</td><td>0.7270</td><td>0.1276</td></tr><tr><td>0.5</td><td>0.0628</td><td>0.2502</td><td>0.5228</td><td>0.7193</td><td>0.1243</td></tr><tr><td>1.0</td><td>0.0579</td><td>0.2343</td><td>0.5011</td><td>0.7043</td><td>0.1160</td></tr><tr><td rowspan="4">New</td><td>0.0</td><td>0.0376</td><td>0.1590</td><td>0.3833</td><td>0.6200</td><td>0.0779</td></tr><tr><td>0.1</td><td>0.0339</td><td>0.1503</td><td>0.3744</td><td>0.6153</td><td>0.0727</td></tr><tr><td>0.5</td><td>0.0233</td><td>0.1089</td><td>0.3141</td><td>0.5752</td><td>0.0534</td></tr><tr><td>1.0</td><td>0.0180</td><td>0.0835</td><td>0.2507</td><td>0.5127</td><td>0.0415</td></tr></table>

Table 7 reports the same ablation in a subsequent seven-day cycle initialized from the Day-30 model and can be seen in Figure 3. The qualitative pattern is similar: new-item quality depends strongly on the new checkpoint, while old-item quality is comparatively stable. At $\alpha = 0 . 1 ,$ a few old-item metrics (HR@10, HR@100, MRR) marginally exceed the � = 0 values. This suggests that, over a shorter cycle, retaining a small portion of the old model’s weights can slightly improve performance on older items. However, the gain is marginal and almost flat, likely because the data available within a short cycle is less diverse, whereas the old model has already learned from a large amount of historical data. Therefore, this limited improvement does not contradict the overall conclusion that directly using the new checkpoint is optimal or near-optimal.

Table 7: Weight-mixing ablation for Frozen after an additional 7-day cycle (Day37).
<table><tr><td>Items</td><td>α</td><td>HR@1</td><td>HR@10</td><td>HR@100</td><td>HR@1000</td><td>MRR</td></tr><tr><td rowspan="4">Old</td><td>0.0</td><td>0.0687</td><td>0.2754</td><td>0.5490</td><td>0.7328</td><td>0.1357</td></tr><tr><td>0.1</td><td>0.0686</td><td>0.2756</td><td>0.5491</td><td>0.7329</td><td>0.1356</td></tr><tr><td>0.5</td><td>0.0674</td><td>0.2714</td><td>0.5443</td><td>0.7306</td><td>0.1335</td></tr><tr><td>0.9</td><td>0.0647</td><td>0.2631</td><td>0.5348</td><td>0.7248</td><td>0.1291</td></tr><tr><td rowspan="4">New</td><td>0.0</td><td>0.0492</td><td>0.2114</td><td>0.4468</td><td>0.6551</td><td>0.1021</td></tr><tr><td>0.1</td><td>0.0466</td><td>0.2043</td><td>0.4434</td><td>0.6535</td><td>0.0983</td></tr><tr><td>0.5</td><td>0.0341</td><td>0.1549</td><td>0.3795</td><td>0.6126</td><td>0.0747</td></tr><tr><td>0.9</td><td>0.0301</td><td>0.1319</td><td>0.3269</td><td>0.5619</td><td>0.0644</td></tr></table>

## 4.7 Online A/B Test

We deployed generative retrieval in the recall stage of the search system on TmallAPP, a major mobile e-commerce platform, where it works together with other recall channels. The generative retriever undergoes warm-start training on 7-days basis. The initial SIDs are constructed using a hybrid-collision method and updated daily. We adopt a topology-preserving policy that maintains the SIDs of existing items unchanged. For newly added items, the first two layers of the SID are generated via RQ-VAE, while the third layer is determined based on business value ranking. The system operates with dynamic beam sizes in [50, 200, 1000], maintaining an average end-to-end latency of approximately 50 ms. Compared with the baseline trained 2 months ago that freezes old items and inserts new items without cyclic training, a three-week online A/B test conducted on 20% of live trafic validated the efectiveness of the proposed policies, demonstrating consistent gains in key metrics: UCTCVR increased by 0.29% and GMV by 1.16%.

![](images/b7ca8f983d1840fe173487ef95010d05710d9dcfd4e35000b1873227074333d9.jpg)

![](images/43e6feb533139f50233629d67812b6426a7b28cdb86ba517621ff71dfe0ec1d5.jpg)  
Figure 4: Asymmetric sensitivity to model mixing: HR@1000. Day-30 (left) and Day-37 (right) weight-mixing ablations. The horizontal axis is the new-model weight (1 − �). Solid lines show HR@1000 (left axis), dashed lines show MRR (right axis). Both metrics rise monotonically with the new-model weight, confirming that the pure new checkpoint (1 − � = 1) is optimal or near-optimal.

## 5 Discussion

Our item SID deliberately embeds a time-varying business-value dimension (the third token � ), which makes value-rank refresh a frequent and large-magnitude source of drift; in this sense the severity we report is characteristic of value-aware SID schemes. The underlying mechanism, however, is not specific to the value dimension: any operation that reassigns existing items to diferent SIDs—refreshing value positions, retraining the RQ-VAE codebook after a large ingestion batch, or de-duplicating collisions—invalidates part of the learned decoding paths, as formalized by $\delta ( M _ { t _ { 0 } } , M _ { t _ { 1 } } )$ in Eq 3. More generally, the drift magnitude is governed less by the presence of a value dimension per se than by how extensively the SID-assignment model is retrained: fully retraining the quantizer/codebook produces an entirely new SID space and hence the largest drift, whereas fine-tuning only a subset of parameters reassigns far fewer identifiers and yields proportionally smaller drift. Across this spectrum the direction of the efect and the preservation principle are unchanged. We therefore position topology preservation as a general operating principle whose urgency scales with how much of the mapping a given update reassigns.

The proposed policy, which involves freezing existing items, inserting new ones, and employing warm-starts, is intentionally designed for operational simplicity. We contend that this simplicity is a distinct advantage rather than a limitation. In production retrieval systems, update policies must inherently be reliable, easily debuggable, and broadly applicable across diverse item categories. Moreover, real-world deployments typically involve a temporal decoupling between serving model updates and item index table refreshes. During this transition period, any misalignment between the updating model and the item index can cause inference outputs to fail when mapping back to their corresponding items. This operational constraint also motivates our strategy of preserving the original SIDs for existing items. Therefore, the primary contribution of this work extends beyond the policy itself; it provides robust empirical evidence that such a straightforward approach is practically suficient.

## 6 Conclusion

We investigate strategies for updating generative retrieval systems in e-commerce environments where items increase and businessvalue signals continuously evolve. Our experiments reveal that full identifier remapping induces severe semantic drift, whereas wide-beam evaluation demonstrates significantly less degradation. Across all tested update policies, both HitRate and MRR decrease monotonically as the proportion ofremapped identifiers for existing items increases. Under identical warm-start conditions, retaining original identifiers for existing items while selectively inserting only eligible new ones yields higher HitRate than full remapping—a trend that remains consistent across subsequent dependent update cycles. Furthermore, models fine-tuned from previous checkpoints outperform naive model souping. These findings distill into a concise operational principle: insert new items but freeze existing ones; apply warm-starting when feasible, otherwise freeze the model; and strictly avoid altering the semantic identifiers (SIDs) the model has already learned. A three-week online A/B test on the mobile e-commerce platform TmallAPP search system validates this principle with UCTCVR +0.29% and GMV +1.16%.

## References

[1] Vladimir Baikalov, Iskander Bagautdinov, and Sergey Muravyov. 2026. Mitigating Collaborative Semantic ID Staleness in Generative Retrieval. In Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval. ACM, 3602–3608. doi:10.1145/3805712.3809877

[2] Nicola De Cao, Gautier Izacard, Sebastian Riedel, and Fabio Petroni. 2021. Au toregressive Entity Retrieval. arXiv:2010.00904 [cs.CL] https://arxiv.org/abs 2010.00904

[3] Ben Chen, Xian Guo, Siyuan Wang, Zihan Liang, Yue Lv, Yufei Ma, Xinlong Xiao, Bowen Xue, Xuxin Zhang, Ying Yang, et al. 2025. Onesearch: A preliminary exploration of the unified end-to-end generative framework for e-commerce search. arXiv preprint arXiv:2509.03236 (2025).

[4] Jiangui Chen, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Wei Chen, Yixing Fan, and Xueqi Cheng. 2023. Continual Learning for Generative Retrieval over Dynamic Corpora. In Proceedings ofthe 32nd ACM International Conference on Information and Knowledge Management. ACM, 306–315. doi:10.1145/3583780. 3614821

[5] Runjin Chen, Mingxuan Ju, Ngoc Bui, Dimosthenis Antypas, Stanley Cai, Xiaopeng Wu, Leonardo Neves, Zhangyang Wang, Neil Shah, and Tong Zhao. 2024. Enhancing Item Tokenization for Generative Recommendation through Self-Improvement. arXiv:2412.17171 [cs.LG] https://arxiv.org/abs/2412.17171

[6] Edoardo D’Amico, Marco De Nadai, Praveen Chandar, Divita Vohra, Shawn Lin, Max Lefarov, Paul Gigioli, Gustavo Penha, Ilya Kopysitsky, Ivo Joel Senese, Darren Mei, Francesco Fabbri, Oguz Semerci, Yu Zhao, Vincent Tang, Brian St. Thomas, Alexandra Ranieri, Matthew N. K. Smith, Aaron Bernkopf, Bryan Leung, Ghazal Fazelnia, Mark VanMiddlesworth, Timothy Christopher Heath, Petter Pehrson Skiden, Alice Y. Wang, Doug J. Cole, Andreas Damianou, Maya Hristakeva, Reid Wilbur, Tarun Chillara, Vladan Radosavljevic, Pooja Chitkara, Sainath Adapa, Juan Elenter, Bernd Huber, Jacqueline Wood, Saaketh Vedantam, Jan Stypka, Sandeep Ghael, Martin D. Gould, David Murgatroyd, Yves Raimond, Mounia Lalmas, and Paul N. Bennett. 2026. Deploying Semantic ID-based Generative Retrieval for Large-Scale Podcast Discovery at Spotify. arXiv:2603.17540 [cs.IR] https://arxiv.org/abs/2603.17540

[7] Yu-Chen Den, Yung-Yu Shih, Zhi Rui Tam, Kuan-Yu Chen, Pu-Jen Cheng, Yun-Nung Chen, and Eugene Yang. 2026. ICICLE: Expanding Retrieval with In-Context Documents. arXiv:2605.26902 [cs.IR] https://arxiv.org/abs/2605.26902

[8] Jiaxin Deng, Shiyao Wang, Kuo Cai, Lejian Ren, Qigen Hu, Weifeng Ding, Qiang Luo, and Guorui Zhou. 2025. Onerec: Unifying retrieve and rank with generative recommender and iterative preference alignment. arXiv preprint arXiv:2502.18965 (2025).

[9] Yuebo Feng, Jiahao Liu, Mingzhe Han, Dongsheng Li, Hansu Gu, Peng Zhang, Tun Lu, and Ning Gu. 2026. Drift-Aware Continual Tokenization for Generative Recommendation. arXiv:2603.29705 [cs.IR] https://arxiv.org/abs/2603.29705

[10] Kairui Fu, Tao Zhang, Shuwen Xiao, Ziyang Wang, Xinming Zhang, Chenchi Zhang, Yuliang Yan, Junjun Zheng, Xiangheng Kong, Shengyu Zhang, Kun Kuang, and Yuning Jiang. 2026. FORGE: Forming Semantic Identifiers for Generative Retrieval in Industrial Datasets. arXiv:2509.20904 [cs.IR] https://arxiv.org/abs 2509.20904

[11] Liwei Guan, Huanjie Wang, Hongwei Zhang, Linxun Chen, and Zhaojie Liu. 2026. DREAM: Dynamic Refinement of Early Assignment Mappings. arXiv:2606.06947 [cs.IR] https://arxiv.org/abs/2606.06947

[12] Zheng Hu, Yuxin Chen, Yongsen Pan, Xu Yuan, Yuting Yin, Daoyuan Wang, Boyang Xia, Zefei Luo, Hongyang Wang, Songhao Ni, Dongxu Liang, Jun Wang, Shimin Cai, Tao Zhou, Fuji Ren, and Wenwu Ou. 2026. Stop Treating Collisions Equally: Qualification-Aware Semantic ID Learning for Recommendation at Industrial Scale. arXiv:2603.00632 [cs.IR] https://arxiv.org/abs/2603.00632

[13] Tuan-Luc Huynh, Thuy-Trang Vu, Weiqing Wang, Trung Le, Dragan Gašević, Yuan-Fang Li, and Thanh-Toan Do. 2026. MixLoRA-DSI: Dynamically Expandable Mixture-of-LoRA Experts for Rehearsal-Free Generative Retrieval over Dynamic Corpora. arXiv:2507.09924 [cs.IR] https://arxiv.org/abs/2507.09924

[14] Tuan-Luc Huynh, Thuy-Trang Vu, Weiqing Wang, Yinwei Wei, Trung Le, Dragan Gasevic, Yuan-Fang Li, and Thanh-Toan Do. 2025. PromptDSI: Prompt-based Rehearsal-free Continual Learning for Document Retrieval. arXiv:2406.12593 [cs.IR] https://arxiv.org/abs/2406.12593

[15] Gui Ling, Zhihong Chen, Yu Li, Tong Xiong, Kunhai Lin, Kaixuan Zhang, Yuliang Yan, Dan Ou, Haihong Tang, and Bo Zheng. 2026. Beyond Semantic IDs: Encoding Business-Value Ranking into Document Identifiers for Generative Retrieval. arXiv:2607.11392 [cs.IR] https://arxiv.org/abs/2607.11392

[16] Yaokun Liu, Yifan Liu, Zhenrui Yue, Gyuseok Lee, Zelin Li, Ruichen Yao, and Dong Wang. 2026. Topology-Aware Tokenization for Generative Recommendation. arXiv:2607.18600 [cs.IR] https://arxiv.org/abs/2607.18600

[17] Ziwei Liu, Yejing Wang, Wanyu Wang, Wang Zejian, Qidong Liu, Zijian Zhang, Chong Chen, Wei Huang, and Xiangyu Zhao. 2026. The Best of the Two Worlds: Harmonizing Semantic and Hash IDs for Sequential Recommendation. arXiv:2512.10388 [cs.IR] https://arxiv.org/abs/2512.10388

[18] Sanket Vaibhav Mehta, Jai Gupta, Yi Tay, Mostafa Dehghani, Vinh Q. Tran, Jinfeng Rao, Marc Najork, Emma Strubell, and Donald Metzler. 2023. DSI++: Updating Transformer Memory with New Documents. arXiv:2212.09744 [cs.CL] https://arxiv.org/abs/2212.09744

[19] Kidist Amde Mekonnen, Yubao Tang, and Maarten de Rijke. 2026. A Parametric Memory Head for Continual Generative Retrieval. In Proceedings ofthe 49th International ACM SIGIR Conference on Research and Development in Information Retrieval. ACM, 1347–1358. doi:10.1145/3805712.3809725

[20] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. 2025. Qwen2.5 Technical Report. arXiv:2412.15115 [cs.CL] https://arxiv.org/abs/2412.15115

[21] Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Hulikal Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Tran, Jonah Samost, et al. 2023. Recommender systems with generative retrieval. Advances in Neural Information Processing Systems 36 (2023), 10299–10315.

[22] Anima Singh, Trung Vu, Nikhil Mehta, Raghunandan Keshavan, Maheswaran Sathiamoorthy, Yilin Zheng, Lichan Hong, Lukasz Heldt, Li Wei, Devansh Tandon, Ed H. Chi, and Xinyang Yi. 2024. Better Generalization with Semantic IDs: A Case Study in Ranking for Recommendations. arXiv:2306.08121 [cs.IR] https: //arxiv.org/abs/2306.08121

[23] Zhengyang Su, Isay Katsman, Yueqi Wang, Ruining He, Lukasz Heldt, Raghu nandan Keshavan, Shao-Chuan Wang, Xinyang Yi, Mingyan Gao, Onkar Dalal, Lichan Hong, Ed H. Chi, and Ningren Han. 2026. Vectorizing the Trie: Eficient Constrained Decoding for LLM-based Generative Retrieval on Accelerators. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2. ACM, 8009–8020. doi:10.1145/3770855.3818506

[24] Yi Tay, Vinh Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta, Zhen Qin, Kai Hui, Zhe Zhao, Jai Gupta, et al. 2022. Transformer memory as a diferentiable search index. Advances in neural information processing systems 35 (2022), 21831–21843.

[25] Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, and Ludwig Schmidt. 2022. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. arXiv:2203.05482 [cs.LG] https://arxiv.org/abs/2203.05482

[26] Mitchell Wortsman, Gabriel Ilharco, Jong Wook Kim, Mike Li, Simon Kornblith, Rebecca Roelofs, Raphael Gontijo-Lopes, Hannaneh Hajishirzi, Ali Farhadi, Hongseok Namkoong, and Ludwig Schmidt. 2022. Robust fine-tuning ofzero-shot

[27] Tianyu Zhan, Gui Ling, Tong Xiong, Kunhai Lin, Yang Wang, Kaixuan Zhang, Zhihong Chen, Yuliang Yan, Dan Ou, Shengyu Zhang, Haihong Tang, and Bo Zheng. 2026. TSGR: Taobao Search Generative Retrieval. arXiv:2607.18796 [cs.IR] https://arxiv.org/abs/2607.18796

models. arXiv:2109.01903 [cs.CV] https://arxiv.org/abs/2109.01903

[28] Bowen Zheng, Enze Liu, Zhongfu Chen, Zhongrui Ma, Yue Wang, Wayne Xin Zhao, and Ji-Rong Wen. 2025. Pre-training Generative Recommender with Multi Identifier Item Tokenization. arXiv:2504.04400 [cs.IR] https://arxiv.org/abs/2504. 04400

[29] Ziyu Zheng, Zhengshun Du, Yaming Yang, Bin Tong, Guan Wang, Meng Yan, Ziyu Guan, and Wei Zhao. 2026. TopoGR: Revealing and Preserving Latent Structure of Semantic ID in Generative Recommendation. arXiv:2607.25216 [cs.IR] https: //arxiv.org/abs/2607.25216

[30] Jianbo Zhu, Xing Fang, Jing Wang, Mingmin Jin, Bokang Wang, Guangxin Song, Zhenyu Xie, and Junjie Bai. 2026. Eficient Generative Retrieval for E-commerce Search with Semantic Cluster IDs and Expert-Guided RL. arXiv:2605.14434 [cs.IR] https://arxiv.org/abs/2605.14434