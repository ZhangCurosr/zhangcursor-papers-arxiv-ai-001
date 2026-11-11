# SeqFold: Timestamp-Addressable Compressed States for User Action Sequence Backbones

Mianwei Zhou, Duo Zhang, Henry Tsai, Yuqi Zheng, Jian Li, Juexin Lin,

Sujay Khandagale, Vijay Rajaram, Dong Jiao, Tao Cheng

mizhou,duozhang,mitsai,yuqizheng,jili31,julin9,sukhanda,vijay,dojiao,taocheng@coupang.com

Coupang

Mountain View, United States

## Abstract

Existing sequential user modeling methods mainly focus on improving the backbone encoder or learning task-specific user rep resentations for a single prediction objective. We instead study how to export the multi-layer user states of a frozen long-history backbone as compact, timestamp-addressable representations for downstream training backfill and serving. In this setting, the user state at a given timestamp summarizes the user’s actions before that time, so downstream systems need only the state aligned to each target timestamp rather than the full hidden-state timeline. To this end, we present SeqFold, a compressed-state interface that combines learned multi-channel fusion, channel-specific temporal decay, and Lookup-Free Quantization (LFQ) to produce compact discrete codes from frozen backbone states. Task-specific decoders then map these shared codes into flexible embedding spaces for downstream models. To make timestamp-addressable export practical at scale, we further develop a fused streaming implementation that updates the fusion recurrence after every action while materi alizing states only at target timestamps. On large-scale proprietary datasets, SeqFold reduces per-state representation storage by 3,400× while preserving most of the downstream ranking performance of the uncompressed model, showing that SeqFold is an efective inter face for reusing long-history sequence backbones in downstream models.

## Keywords

user behavior modeling, sequence models, representation compression, vector quantization, recommender systems

## 1 Introduction

Large-scale e-commerce platforms increasingly rely on user behavior sequence models to capture long-term preferences, evolving purchase intent, and cross-surface behavior patterns. A user action sequence can contain thousands of heterogeneous actions, such as search queries, product impressions, clicks, add-to-cart events, and purchases. Transformer-based architectures are well suited to this setting because they can jointly encode semantic, contextual, and temporal dependencies across long user histories [9, 10, 17]. In practical large-scale systems, these sequence models often serve as reusable user representation backbones.

While this backbone reuse is attractive, it creates a mismatch between what the upstream sequence model produces and what downstream systems actually need. A frozen long-history backbone can produce rich multi-layer states over a user’s full action sequence, but downstream applications do not directly need that entire hidden-state timeline. Instead, for downstream ranking, both ofline backfill and online serving need only the user state aligned to each target timestamp, that is, a summary of the user’s actions before that time. Passing the raw backbone states through this interface is nevertheless impractical. In our main experiments, each user sequence contains at most 2,000 actions and the backbone uses 4 Transformer layers with hidden dimension 256, yielding up to approximately 2.05 × 10<sup>6</sup> floating-point values in the full hidden-state timeline for a single user sequence. Delivering such a high-dimensional payload to downstream models would create severe storage, transmission, and memory costs.

![](images/4a7fba569d2890ee64c98ee23c3532debfb561dc982ec1a4700ae0c55b927390.jpg)  
Figure 1: SeqFold as a timestamp-addressable interface between a frozen action sequence backbone and downstream applications. Given a target timestamp, SeqFold returns timestamp-aligned compact codes that downstream tasks decode with task-specific decoders for downstream training backfill and serving.

To address this bottleneck, we study the problem of timestampaddressable compressed-state export for downstream training backfill and serving. Given a user history and a target timestamp, the goal is to export a compact representation of the user state available before that time. As illustrated in Figure 1, our focus is not on accelerating the backbone itself, but on the interface between an upstream frozen long-history backbone and downstream consumers that need compact timestamp-aligned user states. We therefore propose SeqFold, a timestamp-addressable compressed-state interface for this setting. Given a target timestamp, SeqFold returns compact codes derived from the frozen backbone and aligned to that timestamp. Task-specific decoders then map these shared codes into continuous representations for downstream models. SeqFold instantiates this interface with learned multi-channel fusion, channel-specific temporal decay, and Lookup-Free Quantization (LFQ) [23], producing compact discrete codes for flexible downstream use.

A timestamp-addressable interface changes not only what must be stored, but also what must be materialized during training backfill and serving. Let � denote batch size, � sequence length, � hidden dimension, and let � denote the number of fusion channels in SeqFold. A direct tensorized implementation of SeqFold’s action-level recurrence would construct a channel-specific state for every action prefix, expanding the activation footprint from [�, � , �] to [�, � , �, �] and leading to severe memory pressure. Yet the target workload does not require this dense prefix timeline: it needs only the latest causally accessible state at each target timestamp. We therefore develop a fused streaming implementation that scans each sequence chronologically, updates the fusion recurrence after every action, and materializes states only at target timestamps. During compressor training, periodic export points allow nearby target timestamps to share the latest causally valid exported state while improving eficiency.

In summary, the primary contributions of this paper are as follows:

• We formulate downstream reuse of long-history backbone states as timestamp-addressable compressed-state export, where downstream training backfill and online serving consume a compact causal state aligned to each target timestamp rather than a single static user embedding.

• We propose SeqFold, which instantiates this interface with learned multi-channel fusion, temporal decay, and LFQ encoding to compress multi-layer user representations into compact discrete codes.

• We develop a fused streaming export mechanism that decouples state update from state materialization, preserving action-level recurrence while avoiding dense prefixstate tensors.

• We show on proprietary datasets that SeqFold provides a strong quality–storage trade-of and that its timestampstate generation scales eficiently with sequence length and export sparsity.

## 2 Related Work

## 2.1 Sequential User Modeling and Long-History Retrieval

Sequential recommendation models encode user histories to predict future actions. Early systems relied on recurrent or convolutional architectures such as GRU4Rec [5] and Caser [18]. Later work introduced target-aware interest modeling, including DIN [25] and DIEN [24], while Transformer-based backbones such as SASRec [9], BERT4Rec [17], and TiSASRec [10] improved long-range dependency modeling. For very long histories, industrial systems often combine sequence modeling with retrieval or multi-interest summarization, for example SIM [13] and MIND [12]. These methods mainly improve the backbone encoder or produce task-specific user representations for a particular prediction objective. In contrast, SeqFold assumes a frozen backbone and studies how to export its multi-layer causal states as timestamp-addressable compressed states for downstream reuse.

## 2.2 Reusable User Representations for Downstream Tasks

A related line of work studies how upstream representation models can serve multiple downstream retrieval or ranking stages. Deep structured semantic models [6] and large-scale recommender stacks such as YouTube’s two-stage architecture [2] motivate the practical value of shared user representations, but they typically expose a single static or request-time embedding rather than a causal state aligned to an arbitrary target timestamp. SeqFold is closest in spirit to this reuse setting, but it exports compact states indexed by timestamp so that ofline backfill and online serving can both consume a causal state aligned to the target time without replaying or storing the full hidden-state timeline.

## 2.3 Long-Context Eficiency and State Summarization

A separate line of work reduces the cost of processing long contexts within the backbone itself. Eficient attention methods such as Linformer [21], Longformer [1], and FlashAttention [4] improve the asymptotic or practical cost of attention, while recurrenceand memory-based variants such as Transformer-XL [3] and Compressive Transformer [14] carry information across longer hori zons. Other approaches summarize or compress long sequences inside the model, for example Perceiver IO [7], TokenLearner [16], VQL [11], and SlimPer [20]. These approaches are relevant because SeqFold also targets long histories, but they optimize a diferent stage of the pipeline: they redesign the backbone or its intermediate computation, whereas SeqFold operates after a frozen backbone has produced hidden states and focuses on storing and serving timestamp-aligned compressed states.

## 2.4 Discrete Representation Learning for Compact State Interfaces

Discrete latent methods provide tools for aggressively compressing continuous representations. VQ-VAE [19] and VQ-VAE-2 [15] learn discrete codebooks for neural latent spaces, product quantization [8] compresses vectors for large-scale similarity search, and Lookup-Free Quantization (LFQ) [23] replaces explicit codebook lookup with direct sign-based indexing. SeqFold applies LFQ in a diferent regime: it serializes multi-layer user states into compact causal codes that remain addressable by timestamp, while allowing each downstream system to decode the same shared codes into taskspecific continuous spaces.

## 3 Problem Setup and Interface Requirements

## 3.1 Frozen User Action Sequence Backbone

We assume a frozen causal user action sequence backbone that is trained on historical action data and refreshed periodically. Each action combines heterogeneous contextual and behavioral signals, and the backbone maps every action to a hidden state while respecting chronological causality. As illustrated in Figure 2, historical actions are updated layer by layer through causal self-attention, while the current action context queries only the preceding history through causal cross-attention to produce the final current-action state for label prediction. In our main experiments, the backbone stacks 4 Transformer layers with a 256-dimensional hidden space, and each user sequence contains at most 2,000 actions.

![](images/02e25f42e4cbf5a443c3388506e636f8489b30c3e0cf9f16d1748cfc595b9af5.jpg)  
Figure 2: Two-stream action sequence backbone. At each layer, historical action states are updated by causal selfattention, while the current action context attends only to preceding historical states through causal cross-attention. The resulting current-action state is propagated upward through the stacked layers, and the final current-action state is used to predict the action label.

To train this backbone, we adopt a two-stream setup inspired by XLNet [22]. Let $a _ { i }$ denote the �-th action in a user’s chronological sequence. We decouple each action into a context part and a label part, and predict the current action label from its concurrent context together with the historical actions that precede it:

$$
P ( \mathrm { L a b e l } ( a _ { n + 1 } ) \mid \mathrm { C o n t e x t } ( a _ { n + 1 } ) , \mathrm { L a b e l } ( a _ { 1 : n } ) , \mathrm { C o n t e x t } ( a _ { 1 : n } ) ) .
$$

In the rest of the paper, we use $a _ { i }$ for the action itself and $A _ { i } ^ { \mathrm { f u l l } }$ and $A _ { i } ^ { \mathrm { c t x } }$ for the corresponding backbone-produced hidden states under the full-information and context-only views, respectively. In practice, the main target is action-type prediction, and additional auxiliary objectives share the same backbone. The goal of this paper is not to redesign that backbone, but to expose its multi-layer causal states to downstream systems through a compact timestampaddressable interface.

## 3.2 Downstream Records and

## Timestamp-Addressable Export

Downstream applications such as search, recommendation, and query autocomplete do not need the backbone’s full hidden-state timeline. Instead, each ofline training example or online serving request needs a compact causal state aligned to its target timestamp. A historical record must use only the user state available before its target event, and an online request must consume the latest causally valid state at request time.

Exporting the raw stacked hidden states is nevertheless impractical at this interface. For a history length �, � backbone layers, and hidden dimension �, one user’s complete multi-layer state contains $N \times M \times D$ floating-point values. In our main experiments, $N \leq 2 , 0 0 0 , M = 4$ , and $D \ : = \ : 2 5 6 .$ , so one user’s complete multilayer state can contain up to $2 . 0 4 8 \times 1 0 ^ { 6 }$ floating-point values. For many lightweight downstream models, a standard record contains only a few hundred dense and sparse features, so injecting the full backbone state would overwhelm storage, transmission, and training-time memory budgets. The desired interface must therefore satisfy three requirements: it must be causal with respect to target time, compact enough for large-scale storage and delivery, and reusable across both backfill and online serving.

## 4 SeqFold as a Compressed State Interface

## 4.1 Architecture Overview

SeqFold sits between a frozen long-history action sequence backbone and the downstream systems that consume its states. The backbone first computes layer-wise history states, and SeqFold compresses those states after the backbone instead of rewriting the backbone attention blocks. It then exports a compact causal state indexed by target timestamp in place of the full per-action hidden-state timeline. For any requested time �, it summarizes only the actions before � into reusable codes, enabling backfill and online serving to consume the same timestamp-addressable interface without replaying or storing the full continuous sequence. As illustrated in Figure 3, this interface has two operational phases: ofline compression training and downstream serving. The bottom zoom-in further expands the training-only layer-wise learning path for one representative backbone layer.

During the Training Phase, the parameters of the pre-trained action sequence backbone are frozen, and SeqFold is trained to map the backbone history states into compact discrete codes through Lookup-Free Quantization (LFQ) [23]. The compressed states are then passed through a training-time LFQ decoder and a layer-wise cross-attention module. This training-only path is supervised with the same upstream action-prediction objective used to train the backbone and is used only to supervise the compressor; it is not the task-specific decoder used during downstream serving. The output of this phase is a compact matrix of quantized indices that summarizes the user behavior profile across multiple backbone layers.

During the Downstream Serving Phase, downstream models consume these shared codes at the requested serving or training timestamp rather than the original continuous hidden states. Each downstream task then learns its own task-specific LFQ decoder that maps the LFQ codes into a continuous space layer-by-layer before cross-attention scoring against its task-specific targets. The decoded embedding dimension need not match the original backbone dimension, allowing the same exported codes to support diferent downstream capacity budgets.

To illustrate the compression eficiency, consider a sequence with average history length �<sup>¯</sup> , � backbone layers, hidden dimension $D , C$ fusion vectors per layer, and $b _ { \mathrm { L F Q } }$ bits per code. The raw, uncompressed user representation at one target timestamp requires $\bar { N } \times M \times D$ floating-point values, whereas SeqFold requires only $M \times C \times b _ { \mathrm { L F O } } / 8$ bytes after quantization. In our main experiments, �<sup>¯</sup> ≈ 1,700, $M = 4 , D = 2 5 6 , C = 6 4 _ { \cdot }$ , and $b _ { \mathrm { L F Q } } = 6 4$ , so the raw representation requires $1 , 7 0 0 \times 4 \times 2 5 6 = 1 , 7 4 0$ ,800 floating-point values, while SeqFold requires only $4 \times 6 4 \times 6 4 / 8 = 2 , 0 4 8$ bytes. The resulting gap can be substantial even for moderate layer counts and hidden dimensions, making compact timestamp-addressable export attractive for downstream consumption.

![](images/64faa4006d5cde83558830956e3cd029c5c78ff96a8f700e5ff395ce5b4f6511.jpg)  
Figure 3: SeqFold compression training and downstream serving. A frozen multi-layer action sequence backbone first produces layer-wise history states, which SeqFold compresses into shared LFQ bit codes. During compression training, a training-only layer-wise learning path (bottom zoom-in; one representative layer shown) decodes the compressed states and applies cross attention to the current action states to supervise the compressor with the upstream prediction loss. During serving, each downstream application reuses the shared codes and learns its own task-specific LFQ decoder with a configurable output dimension for layer-wise cross-attention.

In the subsequent sections, we first describe one concrete instantiation of this timestamp-addressable interface. We then analyze the bottlenecks that arise from dense tensor materialization and present a fused streaming implementation that supports sparse timestamp export in long-history settings.

## 4.2 A Concrete Instantiation

We next describe one concrete instantiation of the SeqFold interface. Its core mechanism adaptively gates, decays, and accumulates variable-length sequence embeddings into a fixed number of com pressed discrete latent vectors. Table 1 summarizes the key notation used below.

Formally, for a sequence of length �, SeqFold computes gated action contributions $V _ { i , c } ;$ , continuous fusion states $S _ { i , c } { : }$ , binary LFQ codes $Q _ { i , c } \in \{ 0 , 1 \} ^ { b _ { \mathrm { L F Q } } }$ , and a cross-attended representation $\hat { A } _ { i + 1 } ^ { \mathrm { c t x } }$ as follows:

$$
V _ { i , c } = \sigma \left( f _ { c } \left( A _ { i } ^ { \mathrm { f u l l } } \right) \right) \cdot A _ { i } ^ { \mathrm { f u l l } } ,\tag{1}
$$

$$
S _ { i , c } = \sum _ { k \le i } \left( e ^ { - \lambda _ { c } ( t _ { i } - t _ { k } ) } V _ { k , c } \right) ,\tag{2}
$$

$$
\begin{array} { r } { Q _ { i , c } = \mathrm { L F Q E n c o d e } \left( \mathrm { R M S N o r m } \left( S _ { i , c } \right) \right) , } \end{array}\tag{3}
$$

$$
\hat { A } _ { i + 1 } ^ { \mathrm { c t x } } = \mathrm { A t t n } \left( A _ { i + 1 } ^ { \mathrm { c t x } } , \mathrm { L F Q D e c o d e } \left( Q _ { i , 1 : C } \right) \right) .\tag{4}
$$

Table 1: Key notation used in the SeqFold instantiation.
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $\overline { { A _ { i } ^ { \mathrm { f u l l } } } }$ </td><td>Full-information hidden state of action i</td></tr><tr><td> $A _ { i } ^ { \mathrm { { { \dot { c } t x } } } }$ </td><td>Context-only representation for action i</td></tr><tr><td> $c$ </td><td>Fusion channel index</td></tr><tr><td> $\lambda _ { c }$ </td><td>Temporal decay rate for channel c</td></tr><tr><td> $b _ { \mathrm { L F Q } }$ </td><td>Number of bits in each LFQ code</td></tr><tr><td> $S _ { i , c }$ </td><td>Continuous fused state after the first i actions in channel c</td></tr><tr><td> $Q _ { i , c }$ </td><td>LFQ code after the first i actions in channel c</td></tr></table>

Here $\hat { A } _ { i + 1 } ^ { \mathrm { c t x } }$ denotes the cross-attended downstream representation obtained after the context-only query representation attends to the decoded compressed history.

4.2.1 Action Gating and Time-Aware Fusion. As expressed in Equation 1, the compression pipeline begins with dynamic gating. The scalar gate $\sigma \left( f _ { c } \left( A _ { i } ^ { \mathrm { f u l l } } \right) \right)$ acts as a data-dependent routing weight that determines the contribution of the �-th action to the �-th fusion space. Because these channel gates are parameterized independently with sigmoids rather than normalized with a softmax across channels, a single action can contribute to multiple fusion channels when it contains information relevant to multiple semantic aspects.

To capture temporal dynamics in long-term user behaviors, these gated representations $V _ { i , c }$ are aggregated across the entire historical horizon to construct the raw fusion vectors $S _ { i , c } ,$ as detailed in Equation 2. We introduce a time-dependent exponential decay factor $e ^ { - \lambda _ { c } ( t _ { i } - t _ { k } ) }$ , where $t _ { k }$ and $t _ { i }$ denote the respective action timestamps, and $\lambda _ { c }$ is a channel-specific, learnable decay coeficient. This decay mechanism lets more recent actions contribute more strongly to the current state, while historical actions are discounted according to elapsed time. By condensing a sequence of thousands of historical actions into a fixed set of � fusion vectors, SeqFold reduces downstream cross-attention complexity from �(�) to �(�), where $C \ll N$

4.2.2 Fusion Vector Compression and Downstream Decoding. To reduce storage and network transmission cost, the continuous fusion vectors $S _ { i , c }$ undergo standardization and quantization as formalized in Equation 3. We first apply Root Mean Square Normalization (RMSNorm) to stabilize the variance across the learned representation channels. The normalized vectors are then passed to a Lookup-Free Quantization (LFQ) module, which maps continuous vectors directly to discrete binary codes via a sign-based bit assignment strategy. The resulting discrete representation $Q _ { i , c } \in \{ 0 , 1 \} ^ { b _ { \mathrm { L F Q } } }$ can be serialized compactly; in the reference setting used in our exper iments, $b _ { \mathrm { L F Q } } = 6 4$ , so each code is stored as one 64-bit unsigned integer.

During downstream consumption (Equation 4), the target ap plication loads the compact LFQ codes $Q _ { i , 1 : C }$ that summarize the state after the first � actions, so the next target representation $A _ { i + 1 } ^ { \mathrm { c t x } }$ attends only to causally preceding history. These codes are mapped back to continuous space using an LFQDecode network, whose output dimension can be chosen to match the downstream model The downstream model then uses a cross-attention layer (Atn) so that its task-specific targets can query the reconstructed user behavior profile.

## 5 Eficient Timestamp-State Generation

The compressed-state interface above determines which states downstream records address, but it also changes how those states should be materialized. This section focuses on the systems realiza tion of timestamp-addressable export.

## 5.1 Bottlenecks of Dense State Materialization

The formulation of SeqFold in Section 4.2 defines a causal compressed state after every action prefix, so every historical action must update the running state. A direct tensorized implementation of this recurrence, however, can be prohibitively expensive because it materializes all prefix states explicitly. Let � denote the training batch size, � the historical sequence length, � the action embedding dimension, and � the number of fusion channels. The hidden states from the action backbone have shape [�, �, �]. A naive implementation of Equations 1–2 expands each action representation across the fusion channels and materializes all prefix states in a tensor with shape [�, � , �, �]. In the long-history scalability benchmark studied later $( N = 1 0 ^ { 4 } , D = 2 5 6$ , and $C = 6 4 )$ this tensor alone contains 10<sup>4</sup> × 64 × 256 floating-point values per sample per layer, requiring approximately 0.66 GB in FP32 before accounting for other activations or backward storage.

Yet backfill and online serving do not need this dense prefix timeline. Downstream records request states only at target timestamps, each consuming the latest causally available state before its query time. Materializing every action-prefix state is therefore unnecessary for the interface that SeqFold targets.

Compressor training likewise need not materialize the full prefix timeline. Training supervision is attached to downstream prediction records rather than to individual history actions, so each query needs only the latest exported state available before its efective query time. In one practical grouped-export variant with interval �, evenly spaced checkpoints are materialized, and queries between two consecutive checkpoints share the most recent exported state. Figure 4 illustrates this training-time pattern using checkpoints $\tau _ { 1 } , \tau _ { 2 } , \tau _ { 3 } , . . . .$ This introduces only a bounded visibility delay while avoiding dense prefix-state materialization and enabling more eficient batching. The key systems distinction is therefore between state update and state materialization: the recurrence must still be evaluated after every action, but states need to be materialized only at requested timestamps.

![](images/b56841d35346a8196413744f60ba9c7eab846e8677ca4467b52dafa588273852.jpg)  
Figure 4: Fused streaming compression with sparse timestamp export. SeqFold updates the fusion recurrence after every action, but materializes compressed states only at requested timestamps. The right panel illustrates the grouped training special case, where evenly spaced checkpoints $\tau _ { 1 } , \tau _ { 2 } , . . .$ are exported and queries between two consecutive checkpoints reuse the most recent exported state. More generally, the requested export set can be arbitrary, including backfill timestamps or the latest online serving timestamp.

## 5.2 Fused Streaming Realization with Sparse Timestamp Export

SeqFold addresses these bottlenecks by decoupling state update from state materialization.

5.2.1 Streaming Kernel. The summation in Equation 2 can be implemented as an equivalent streaming recurrence:

$$
\begin{array} { r } { S _ { i , c } = \alpha _ { i , c } S _ { i - 1 , c } + \beta _ { i , c } A _ { i } ^ { \mathrm { { f u l l } } } , } \end{array}\tag{5}
$$

Algorithm 1 Fused streaming SeqFold export   
Require: Action states $\{ A _ { i } ^ { \mathrm { f u l l } } \} _ { i = 1 } ^ { N } ,$ , action timestamps $\{ t _ { i } \} _ { i = 1 } ^ { N }$ , requested   
timestamp set $\mathcal { T } _ { u } ^ { \mathrm { r e q } }$ , decay factors $\alpha _ { i , c } ,$ routing weights $\beta _ { i , c }$   
Ensure: Exported timestamp codes $Q _ { u }$   
1: Initialize $S _ { c } \gets 0$ for all $c = 1$ : � and $Q _ { u } \gets 0$   
2: for � = 1 to � do   
3: for � = 1 to � do   
4: $S _ { c } \gets \alpha _ { i , c } S _ { c } + \beta _ { i , c } A _ { i } ^ { \mathrm { f u l l } }$   
5: end for   
6: for each requested timestamp $\tau \in \mathcal { T } _ { u } ^ { \mathrm { r e q } }$ such that � = max $\{ j : t _ { j } <$   
$\tau \}$ do   
7: $Q _ { \tau , c } \gets$ LFQEncode(RMSNorm $\left( S _ { c } \right) )$ for all $c = 1 : C$   
8: $Q _ { u } \gets Q _ { u } \cup \{ Q _ { \tau , c } \} _ { c = 1 : C }$   
9: end for   
10: if training and � is a checkpoint position then   
11: Store an FP32 checkpoint of $S _ { c }$ for the backward reverse scan   
12: end if   
13: end for   
14: return $Q _ { u }$

where $\alpha _ { i , c }$ is the decay factor between consecutive actions and $\beta _ { i , c } ~ = ~ \sigma ( f _ { c } ( A _ { i } ^ { \mathrm { { f u l l } } } ) )$ is the action-dependent routing weight. The recurrence is evaluated after every action, so no historical action is skipped.

We implement this recurrence as a fused streaming kernel rather than as a sequence of standard framework tensor operators. Although Equation 5 admits exact blockwise scan formulations, we prioritize a streaming design because our target workload requires sparse timestamp export without dense prefix-state materialization. The kernel performs the required chronological scan once along the action axis while exposing parallelism over channel–embedding tiles. Routing, temporal decay, state accumulation, and selected timestamp export are fused into one pass, avoiding both the dense [�, �, �, �] prefix-state tensor and repeated memory trafic from materializing a separate state after every action. SeqFold therefore updates the running state after every action but materializes states only at requested timestamps. This separation is the systems counterpart of our timestamp-addressable abstraction: SeqFold preserves action-level causal evolution while exposing only the timestamped states that downstream records actually address.

For training, the backward pass follows the same streaming principle. Instead of saving the full [�, � , �, �] prefix-state tensor, it stores final states and periodic FP32 checkpoints, then reconstructs local states through a reverse scan. The auxiliary workspaces still in clude channel-dependent gradient bufers, but the execution avoids materializing a full $C \times D$ fusion state for every action prefix.

5.2.2 Timestamp Export Policies. Formally, let $\mathcal { T } _ { u } ^ { \mathrm { r e q } }$ be the set of target timestamps requested by downstream records for user �. For a requested timestamp $\tau \in \mathcal { T } _ { u } ^ { \mathrm { r { \dot { \mathrm { e q } } } } }$ , SeqFold exports the state after the latest action that strictly precedes �:

$$
i ( \tau ) = \operatorname* { m a x } \{ i : t _ { i } < \tau \} , \qquad Q _ { \tau , c } = \mathrm { L F Q E n c o d e } \left( \mathrm { R M S N o r m } \left( S _ { i ( \tau ) , c } \right) \right)\tag{6}
$$

This construction enforces the causal constraint required by back fill and online serving. A historical training record receives the user state available before its target event, while online serving can export the latest state at request time. Importantly, the number of requested states $V _ { u } = | \mathcal { T } _ { u } ^ { \mathrm { r e q } } |$ is determined by downstream data demand rather than by the length of the behavior sequence, allowing the same user history to support dense diagnostic export, sparse backfill, or single-state online serving without changing the recurrence. The continuous states materialized by the compressor scale as $O ( V _ { u } C D )$ before quantization and as $O ( V _ { u } C )$ after LFQ encoding, instead of �(���) when every action prefix is exported.

Grouped training export is one special case of this interface and is the case illustrated in Figure 4. When query timestamps are dense during compressor training, we can choose evenly spaced export checkpoints with interval �. In the figure notation, these checkpoints are denoted by $\tau _ { 1 } , \tau _ { 2 } , . . . ,$ and queries between two consecutive checkpoints reuse the most recent exported state. In our implementation, we use zero-based action indexing, so the corresponding requested export set can be written as $\mathcal { T } _ { u } ^ { \mathrm { r e \bar { q } } } = \{ G -$ $1 , 2 G - 1 , \ldots \}$ , leading to:

$$
\hat { A } _ { i } ^ { \mathrm { c t x } } = \mathrm { A t t n } \left( A _ { i } ^ { \mathrm { c t x } } , \mathrm { L F Q D e c o d e } \left( Q _ { \lfloor i / G \rfloor - 1 , 1 : C } ^ { \prime } \right) \right) ,\tag{7}
$$

where actions are zero-indexed, $Q _ { g } ^ { \prime }$ denotes the code exported at the �-th group boundary, and the first group uses an empty initial state. The group size � changes export granularity and trainingstate visibility, but it does not approximate the fusion recurrence: every action still updates the running state through Equation 5. For backfill and online serving, the same fused scan exports precisely the requested target timestamps in $\mathcal { T } _ { u } ^ { \mathrm { r e q } }$

The resulting implementation has three practical properties:

• No dense prefix-state tensor: The forward pass does not materialize the dense [�, �, �, �] prefix-state tensor produced by a tensorized implementation. It keeps only tiled running fusion states during the scan and writes selected outputs.

• Output proportional to requested timestamps: The number of materialized compressed states is controlled by the export set, ranging from every action prefix during diagnostic runs to sparse backfill timestamps or the latest online serving state.

• Controlled backward memory: The backward pass stores final states and periodic FP32 checkpoints to reconstruct the reverse scan. Its auxiliary workspaces still contain channeldependent bufers for accumulating gradients, but they avoid storing the full $C \times D$ fusion state for every action prefix.

## 6 Experiments

## 6.1 Experimental Setup

We evaluate SeqFold on proprietary datasets. Unless otherwise specified, the main experiments use a four-layer backbone with hidden dimension 256, with the compression module applied at every layer. In this main experimental setting, each user sequence contains at most 2,000 actions, with an average length of approximately 1,700. The scalability study benchmarks timestamp-state generation up to 10,000 actions. SeqFold compresses each layer into $C = 6 4$ fusion codes, each containing 64 LFQ bits, for a fixed storage cost of

$$
4 ~ { \mathrm { l a y e r s } } \times 6 4 ~ { \mathrm { c o d e s } } \times 6 4 ~ { \mathrm { b i t s } } = 2 , 0 4 8 ~ { \mathrm { b y t e s } }
$$

per user-state snapshot.

Throughout this section, storage denotes the serialized feature payload required to describe one user at one target timestamp during online serving or backfill. This timestamped definition is important for downstream training: each historical record must be paired with the causal user state available before its target event. In the main comparison with recent raw-action baselines, each record consumes the latest available snapshot before the target event rather than an online request-time state. This protocol reflects downstream settings in which signals attached to the most recent actions may be delayed or not yet finalized when features are materialized. Storage does not include temporary checkpoints materialized during compressor training. Total backfill volume still scales with the number of requested timestamps for every method, while Table 2 compares the payload size of one record.

Due to data confidentiality, we report relative changes rather than absolute metric values. Unless otherwise specified, the metric in this section is Query Precision-Recall AUC (PRAUC) on the upstream action-prediction task used to train the compressor. Here, each target event defines one query, and PR-AUC is computed at the query level and then averaged across queries. The reference configuration is stated in each table; positive values indicate an improvement over that reference, whereas negative values indicate a degradation. The downstream transfer subsection reports each task’s own metric.

## 6.2 Comparison with Compression Baselines

Table 2 compares SeqFold with the uncompressed action sequence model and several compression strategies. The full model retains every action representation from every backbone layer and therefore provides an approximate upper bound on predictive quality. It improves the evaluation metric by 2.11% over SeqFold, but requires an average of 6,963,200 bytes per user-state snapshot. In contrast, SeqFold uses only 2,048 bytes, corresponding to a 3,400× reduction in storage. This result shows that most of the predictive information in the multi-layer action history can be retained without transmitting the complete continuous hidden states.

The continuous Fusion Model isolates the efect of quantization. It uses the same learned fusion mechanism as SeqFold but stores 64 dense 256-dimensional vectors at every layer. Removing LFQ improves the metric by 0.72%, while increasing storage from 2,048 to 262,144 bytes. Thus, LFQ provides an additional 128× reduction over continuous fusion for a 0.72% relative metric decrease. This trade-of is important when the representation must be stored and delivered to multiple downstream applications.

SeqFold also consistently outperforms compression methods that do not learn which historical information should be preserved. Window pooling reduces the sequence length by aggregating neighboring actions, but remains more than 53× larger than SeqFold and trails it by 1.10%–1.38%. The single-vector pooling baseline averages the action states from the final backbone layer only and stores one 256-dimensional FP32 vector (256×4 = 1,024 bytes). It therefore has a slightly smaller footprint than SeqFold, but loses 2.00%, suggesting that one last-layer global vector does not provide enough capacity to represent the diverse interests and temporal patterns in a long user history.

Finally, retaining only 64 raw actions does not ofer a competitive quality–storage trade-of in the evaluated setting. Sampling 64 actions and selecting the most recent 64 actions both require 262,144 bytes—the same storage as continuous fusion—yet underperform SeqFold by 2.37% and 3.98%, respectively. Sampling can discard sparse but informative behaviors, whereas recent-window truncation relies more heavily on immediate recency. Under the per-record snapshot protocol used here, this dependence becomes a liability relative to a summary that aggregates information across the full history. This result supports SeqFold in the timestamped export setting studied here; performance may difer under other online serving assumptions. SeqFold avoids both limitations by allowing all actions to contribute through learned, channel-specific gating and time-aware decay. Overall, the results indicate that the gain comes not only from reducing sequence length, but from learning a structured multi-channel summary before applying aggressive discrete quantization.

## 6.3 Scalability of Timestamp-State Generation

We isolate the timestamp-state generation operator from the backbone and downstream decoders to measure the system-level efect of fused streaming. The benchmark uses � = 64 fusion channels, action dimension $D = 2 5 6 ,$ , bfloat16 inputs, and forward–backward execution. Let � denote batch size, � denote the per-user sequence length, and � denote the total number of exported states in a batch. We compare SeqFold’s Triton fused streaming kernel with an optimized tensorized PyTorch baseline that evaluates the same linear recurrence using chunked cumulative products and cumulative sums. This baseline avoids Python-level loops and represents a strong tensorized implementation of exact blockwise scan using standard framework operators, but it still materializes a state for every action prefix before selecting requested timestamps; in our setting, each action update already spans a wide �×� state (64×256 in the main configuration), so we prioritize a streaming design that directly supports sparse timestamp export rather than additional sequence-axis scan orchestration. In this subsection, we benchmark up to 10,000 actions with $B = 1 0$ so that both implementations remain runnable while keeping the same $C = 6 4 , D = 2 5 6 .$ , bfloat16, and forward–backward protocol. Latency denotes operator execution time and memory denotes peak allocated memory.

Table 3 extends the sequence-length study from 2K actions to long histories. After reducing batch size to keep both implementations runnable, the fused kernel remains close to linear in sequence length, whereas the cumsum baseline grows substantially faster. At 10,000 actions, fused streaming is 19.6× faster and still uses less peak memory, showing that timestamp-state generation remains practical even at 10,000 actions.

Table 4 fixes the action workload at 10,000 actions and varies only the number of requested timestamps. The fused kernel must still scan the entire sequence, but sparse export sharply reduces its materialization and backward costs: decreasing $V / ( B N )$ from 1.0 to 0.0156 reduces latency from 51.70 ms to 18.45 ms and peak memory from 35.0 GB to 2.0 GB. In contrast, the cumsum baseline remains near 1 second and close to 20 GB because it still constructs the full prefix-state timeline before selecting the requested timestamps. At the sparsest setting, fused streaming is 52.2× faster and uses

Table 2: Comparison with full-sequence and sequence-compression baselines. Performance is reported relative to SeqFold: Fusion + LFQ. Storage is measured per user-state snapshot at one serving or backfill timestamp; values marked “avg.” depend on the historical sequence length.
<table><tr><td>Model</td><td>Relative Metric Change</td><td>Storage per Snapshot (Bytes)</td></tr><tr><td>Full Action Sequence Model</td><td>+2.11%</td><td>6,963,200 avg.</td></tr><tr><td>Fusion Model (without LFQ)</td><td>+0.72%</td><td>262,144</td></tr><tr><td>SeqFold: Fusion + LFQ</td><td>0.00%</td><td>2,048</td></tr><tr><td>32-Action Window Pooling</td><td>-1.10%</td><td>217,600 avg.</td></tr><tr><td>64-Action Window Pooling</td><td>-1.38%</td><td>108,800 avg.</td></tr><tr><td>Pooling to One Vector (Last Layer)</td><td>-2.00%</td><td>1,024</td></tr><tr><td>Sample 64 Actions</td><td>-2.37%</td><td>262,144</td></tr><tr><td>Last 64 Actions</td><td>-3.98%</td><td>262,144</td></tr></table>

Table 3: Scaling to long histories under dense export of all action prefixes. We use $B \ : = \ : 1 0 , C \ : = \ : 6 4 , D \ : = \ : 2 5 6 ,$ , bfloat16, and one forward–backward pass. Memory is reported as fused/cumsum peak memory in GB.
<table><tr><td>N</td><td>Fused ms</td><td>Cumsum ms</td><td>Speedup</td><td>Mem. GB</td></tr><tr><td>2048</td><td>10.50</td><td>113.66</td><td>10.8×</td><td>7.2 / 9.6</td></tr><tr><td>4096</td><td>20.97</td><td>262.77</td><td>12.5×</td><td>14.4 / 19.3</td></tr><tr><td>8192</td><td>42.34</td><td>721.42</td><td>17.0×</td><td>28.7 / 38.6</td></tr><tr><td>10000</td><td>51.77</td><td>1014.48</td><td>19.6×</td><td>35.0 / 47.1</td></tr></table>

Table 4: Efect of export density at 10,000 actions with batch size 10. Memory is reported as fused/cumsum peak memory in GB.
<table><tr><td>V/(BN)</td><td>V</td><td>Fused ms</td><td>Cumsum ms</td><td>Mem. GB</td></tr><tr><td>1.0000</td><td>100,000</td><td>51.70</td><td>1017.15</td><td>35.0 / 47.1</td></tr><tr><td>0.2500</td><td>25,000</td><td>26.44</td><td>971.16</td><td>9.9 / 21.9</td></tr><tr><td>0.0625</td><td>6,250</td><td>19.98</td><td>966.45</td><td>3.6 / 19.9</td></tr><tr><td>0.0156</td><td>1,560</td><td>18.45</td><td>963.29</td><td>2.0 / 19.8</td></tr></table>

Table 5: Efect of the number of fusion vectors. The continuous Fusion Model without LFQ is used to isolate the efect of fusion capacity, and the 64-vector configuration is used as the reference.
<table><tr><td>Number of Fusion Vectors</td><td>Relative Metric Change</td></tr><tr><td>8</td><td>-1.16%</td></tr><tr><td>16</td><td>-1.13%</td></tr><tr><td>32</td><td>-0.72%</td></tr><tr><td>64</td><td>0.00%</td></tr><tr><td>128</td><td>0.08%</td></tr></table>

9.9× less peak memory, highlighting the systems benefit of sparse timestamp-addressable export for long histories.

## 6.4 Ablation Study

6.4.1 Number ofFusion Vectors. The number of fusion vectors, �, controls the capacity of the compressed sequence representation. As shown in Table 5, using only 8 or 16 vectors degrades the metric by 1.16% and 1.13%, respectively. Increasing � to 32 recovers a substantial fraction of this loss, reducing the gap to 0.72%. The improvement continues up to 64 vectors, whereas doubling the capacity from 64 to 128 yields only a marginal additional gain of 0.08%. This trend indicates that very small fusion spaces create an information bottleneck, but the benefit largely saturates beyond 64 vectors. We therefore choose � = 64 as the default, since it provides a favorable balance between representation capacity, downstream attention cost, and storage.

Table 6: Efect of applying fusion to diferent numbers of backbone layers. The one-layer configuration uses only the last layer, while the four-layer configuration uses all backbone layers and serves as the reference.
<table><tr><td>Number of Fusion Layers</td><td>Relative Metric Change</td></tr><tr><td>1</td><td>-0.97%</td></tr><tr><td>2</td><td>-0.52%</td></tr><tr><td>3</td><td>-0.44%</td></tr><tr><td>4</td><td>0.00%</td></tr></table>

6.4.2 Number ofFusion Layers. Table 6 studies whether it is suficient to compress only the final backbone layer. Using the last layer alone trails the full four-layer configuration by 0.97%. Adding the second and third fusion layers progressively narrows the gap to 0.52% and 0.44%, while applying fusion at all four layers achieves the best result. In the proposed architecture, each fusion module summarizes the hidden states available at a diferent depth of the backbone and is consumed by the corresponding downstream crossattention layer. The result suggests that compressed representations from diferent depths provide complementary signals to the downstream model. Based on this observation, we apply SeqFold at every backbone layer in the default configuration.

6.4.3 Temporal Decay. Table 7 compares fixed and learned temporal weighting. A fixed one-year retention factor of 0.3 slightly reduces the metric by 0.02%, while a factor of 0.6 yields only a 0.03% improvement. In contrast, learning the decay rates improves the metric by 0.18% over the no-decay variant. These results indicate that a single hand-tuned decay schedule is too restrictive: diferent fusion channels benefit from adapting their temporal horizons to the data. We therefore use channel-specific learned decay rates in SeqFold.

Table 7: Efect of temporal decay. Performance is reported relative to the variant without temporal decay. A constant factor specifies the remaining weight after one year.
<table><tr><td>Temporal Decay</td><td>Relative Metric Change</td></tr><tr><td>No Time Decay</td><td>0.00%</td></tr><tr><td>Constant (0.3 after one year)</td><td>-0.02%</td></tr><tr><td>Constant (0.6 after one year)</td><td>0.03%</td></tr><tr><td>Learned Decay</td><td>0.18%</td></tr></table>

Table 8: Efect of the LFQ code length. Performance is relative to the default 64-bit configuration. Storage is measured per user-state snapshot.
<table><tr><td>Bits per Code</td><td>Storage (Bytes)</td><td>Relative Change</td></tr><tr><td>16</td><td>512</td><td>-0.99%</td></tr><tr><td>32</td><td>1,024</td><td>-0.36%</td></tr><tr><td>64</td><td>2,048</td><td>0.00%</td></tr><tr><td>96</td><td>3,072</td><td>-0.05%</td></tr></table>

6.4.4 LFQ Quantization Capacity. We next vary the capacity of the LFQ representation while keeping the fusion architecture fixed. This experiment measures how aggressively the continuous fusion state can be discretized before quantization becomes the dominant information bottleneck

Table 8 shows a clear quality–storage trade-of. Reducing the code length from 64 to 16 bits lowers the per-snapshot footprint from 2,048 to 512 bytes, but incurs a 0.99% performance loss, confirming that overly aggressive quantization removes useful infor mation from the learned fusion vectors. A 32-bit code recovers most of this loss while retaining a 2× storage reduction relative to the default, trailing the 64-bit code by only 0.36%. Increasing the code length beyond 64 bits provides no further benefit: the 96-bit configuration consumes 50% more storage and performs slightly worse by 0.05%. The diference between 64 and 96 bits is small, and there is no measured benefit from the larger code. We therefore use 64-bit LFQ codes as the default operating point.

## 6.5 Downstream Transfer Across Decoder Dimensions

We next evaluate whether the compact LFQ codes transfer to downstream search and query autocomplete (QAC) models under different decoder capacities. Table 9 reports relative changes in the primary metrics for both tasks as the projection dimension varies from 32 to 128. Relative changes are measured against task-specific baselines that do not use an exported SeqFold user-state embedding as input.

Table 9 shows positive gains for both search and QAC at every tested projection dimension. On search, the gain increases from 0.24% at dimension 32 to 0.61% at dimension 128. QAC shows a similar pattern, improving from 0.10% at dimension 32 to 0.60% at dimension 128, with most of the gain already realized by dimension 64. These results suggest that the LFQ codes remain useful across downstream capacity budgets, allowing each application to choose its decoder dimension without changing the shared upstream exported representation.

Table 9: Transfer of SeqFold codes on search and QAC across projection dimensions.
<table><tr><td>Task</td><td>Projection Dimension</td><td>Relative Change</td></tr><tr><td>Search</td><td>32</td><td>0.2433%</td></tr><tr><td>Search</td><td>64</td><td>0.4908%</td></tr><tr><td>Search</td><td>128</td><td>0.6149%</td></tr><tr><td>QAC</td><td>32</td><td>0.0966%</td></tr><tr><td>QAC</td><td>64</td><td>0.5311%</td></tr><tr><td>QAC</td><td>128</td><td>0.5954%</td></tr></table>

## 7 Conclusion

We presented SeqFold, a framework for compressing multi-layer, long-horizon action representations into compact discrete codes. Its learned channel routing and time-aware fusion combine semantic and temporal signals, while LFQ provides a compact interface that downstream applications can decode into task-specific representation spaces. A fused streaming implementation supports long-sequence training by maintaining a bounded continuous state and batching attention over periodically exported checkpoints.

Experiments on proprietary datasets show that SeqFold reduces the average representation size by 3,400× relative to the full action sequence while remaining within a 2.11% relative metric gap. It provides a better quality–storage trade-of than the evaluated pooling, sampling, and truncation baselines; the fused streaming kernel scales to long histories and benefits from sparse timestamp export; and the learned codes transfer to downstream search and query autocomplete across decoder capacities. Ablations further show that multi-channel and multi-layer fusion matter, learned temporal decay outperforms fixed decay, and gains saturate around 64 channels and 64 bits per code. These results support SeqFold as a shared interface between large user sequence models and downstream applications that consume compact timestamp-aligned states; future work will study additional downstream tasks, adaptive checkpoint intervals, and task-aware compression budgets.

## 8 Ethical Considerations

The experiments in this paper use proprietary datasets of user actions and report only aggregate results. We do not include useridentifying examples or release raw behavioral data in the manuscript. Because compact exported user states can still encode sensitive behavioral signals, practical deployments of SeqFold should follow existing privacy, retention, and access-control policies for user representations. In addition, downstream ranking or recommendation models built on user histories may inherit or amplify historical biases present in the underlying data. We therefore view SeqFold as a systems interface for more eficient state reuse, not as a substitute for application-level auditing, fairness evaluation, or governance before production deployment.

## References

[1] Iz Beltagy, Matthew E. Peters, and Arman Cohan. 2020. Longformer: The Long-Document Transformer. arXiv preprint arXiv:2004.05150 (2020).

[2] Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep Neural Networks for YouTube Recommendations. In Proceedings ofthe 10th ACM Conference on Recommender Systems. 191–198.

[3] Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V. Le, and Ruslan Salakhutdinov. 2019. Transformer-XL: Attentive Language Models beyond a Fixed-Length Context. Advances in Neural Information Processing Systems 32 (2019).

[4] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. FlashAttention: Fast and Memory-Eficient Exact Attention with IO-Awareness. In Advances in Neural Information Processing Systems, Vol. 35. 16344–16359.

[5] Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. 2016. Session-based Recommendations with Recurrent Neural Networks. In International Conference on Learning Representations.

[6] Po-Sen Huang, Xiaodong He, Jianfeng Gao, Li Deng, Alex Acero, and Larry Heck. 2013. Learning Deep Structured Semantic Models for Web Search Using Clickthrough Data. In Proceedings ofthe 22nd ACM International Conference on Information & Knowledge Management. 2333–2338.

[7] Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock, Evan Shel hamer, Olivier Hénaf, Matthew Botvinick, Andrew Zisserman, Oriol Vinyals, and João Carreira. 2022. Perceiver IO: A General Architecture for Structured Inputs and Outputs. In International Conference on Learning Representations.

[8] Hervé Jégou, Matthijs Douze, and Cordelia Schmid. 2011. Product Quantization for Nearest Neighbor Search. IEEE Transactions on Pattern Analysis and Machine Intelligence 33, 1 (2011), 117–128.

[9] Wang-Cheng Kang and Julian McAuley. 2018. Self-Attentive Sequential Recommendation. In 2018 IEEE International Conference on Data Mining. IEEE, 197–206

[10] Jiacheng Li, Yujie Wang, and Julian McAuley. 2020. Time Interval Aware Self Attention for Sequential Recommendation. In Proceedings ofthe 13th International Conference on Web Search and Data Mining. 322–330.

[11] Kaiyuan Li, Yongxiang Tang, Yanhua Cheng, Yong Bai, Yanxiang Zeng, Chao Wang, Xialong Liu, and Peng Jiang. 2025. VQL: An End-to-End Context-Aware Vector Quantization Attention for Ultra-Long User Behavior Modeling. arXiv:2508.17125 [cs.IR] https://arxiv.org/abs/2508.17125

[12] Chuan Ma, Phani Kone, Alex Clearwater, Qi Pi, Weijie Bian, Xiaoqiang Zhu, and Kun Gai. 2019. Multi-Interest Network with Dynamic Routing for Recommenda tion at Tmall. In Proceedings ofthe 28th ACM International Conference on Information and Knowledge Management. ACM, 2105–2113. doi:10.1145/3357384.3357951

[13] Qi Pi, Weijie Bian, Guorui Zhou, Xiaoqiang Zhu, Yujing Zhang, Zhe Wang, Lejian Ren, Ying Fan, and Kun Gai. 2020. Search-based User Interest Modeling with Lifelong Sequential Behavior Data for Click-Through Rate Prediction. In Proceedings of the 29th ACM International Conference on Information and Knowledge Management. 2685–2692.

[14] Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, Chloe Hillier, and Timothy P. Lillicrap. 2020. Compressive Transformers for Long-Range Sequence Modelling. International Conference on Learning Representations (2020).

[15] Ali Razavi, Aaron van den Oord, and Oriol Vinyals. 2019. Generating Diverse High-Fidelity Images with VQ-VAE-2. Advances in Neural Information Processing Systems 32 (2019).

[16] Michael S. Ryoo, AJ Piergiovanni, Anurag Arnab, Mostafa Dehghani, and Anelia Angelova. 2021. TokenLearner: What Can 8 Learned Tokens Do for Images and Videos?. In Advances in Neural Information Processing Systems, Vol. 34. 11237– 11251.

[17] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer. In Proceedings ofthe 28th ACM International Conference on Information and Knowledge Management. 1441–1450.

[18] Jiaxi Tang and Ke Wang. 2018. Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding. In Proceedings ofthe Eleventh ACM International Conference on Web Search and Data Mining. 565–573.

[19] Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. 2017. Neural Discrete Representation Learning. In Advances in Neural Information Processing Systems, Vol. 30.

[20] Siqi Wang, Xianjie Chen, Shaofeng Deng, Albert Chen, Romil Shah, Jiawei Huang, Zhaoqin Wang, Zhang Zhang, Yiqun Liu, Meilei Jiang, et al. 2026. SlimPer: Make Personalization Model Slim and Smart. arXiv:2607.12281 https://arxiv.org/abs/ 2607.12281

[21] Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, and Hao Ma. 2020. Linformer: Self-Attention with Linear Complexity. arXiv preprint arXiv:2006.04768 (2020).

[22] Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Russ R Salakhutdinov, and Quoc V Le. 2019. Xlnet: Generalized autoregressive pretraining for language understanding. Advances in neural information processing systems 32 (2019).

[23] Lijun Yu, José Lezama, Nitesh Bharadwaj Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Agrim Gupta, Xiuye Gu, Alexander G Haupt mann, et al. 2024. Language model beats difusion-tokenizer is key to visual generation. In International Conference on Learning Representations, Vol. 2024. 765–783.

[24] Guorui Zhou, Na Mou, Ying Fan, Qi Pi, Weijie Bian, Chang Zhou, Xiaoqiang Zhu, and Kun Gai. 2019. Deep Interest Evolution Network for Click-Through Rate Prediction. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 33. 5941–5948.

[25] Guorui Zhou, Chengru Song, Xiaoqiang Zhu, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep Interest Network for Click-Through Rate Prediction. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 1059–1068.