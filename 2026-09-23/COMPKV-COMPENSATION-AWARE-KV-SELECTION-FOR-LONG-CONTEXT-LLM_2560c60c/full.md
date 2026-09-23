# COMPKV: COMPENSATION-AWARE KV SELECTION FOR LONG-CONTEXT LLM INFERENCE

Zhen Huang<sup>1∗</sup>, Ruizhe Yao<sup>2∗</sup>, Danyi Liu<sup>1∗</sup>, Xinrui Chen<sup>1</sup>, Shuwei Li<sup>1</sup>, Siru Zhong<sup>1</sup>, Zijian Cao<sup>3</sup>, Yushan Lai<sup>1</sup>, Mingming Guo<sup>1</sup>, Weijie Zheng<sup>2†</sup>, Haohuan Fu<sup>1†</sup>

<sup>1</sup>Tsinghua University <sup>2</sup>Harbin Institute of Technology, Shenzhen <sup>3</sup>Northeastern University zhengweijie@hit.edu.cn haohuan@tsinghua.edu.cn

## ABSTRACT

Despite their strong performance, large language models (LLMs) are bottlenecked by KV cache memory traffic during long-context inference. Sparse attention is widely used to accelerate LLM inference by computing exact attention over a selected subset of tokens. To recover the contribution of tokens excluded from exact attention, recent methods apply coarse-grained compensation to the omitted attention tail. However, existing methods typically select tokens based on attention mass and only then compensate for the unselected tokens. This decoupled design overlooks their interaction: selection should prioritize tokens that would leave the largest compensation error if omitted. To address this limitation, we introduce CompKV, the first compensation-aware sparse attention framework that divides tokens into blocks and explicitly optimizes selection for the downstream compensation mechanism. Our theoretical analysis shows that the residual left by block-level mean compensation is governed by both block attention mass and within-block logit variation. We approximate this residual using compact blocklevel statistics, yielding a deployable selection criterion. We further develop an efficient asynchronous implementation. Experiments on RULER and LongBench-Pro show that CompKV performs best among the evaluated sparse baselines while delivering up to a 6.85× self-attention speedup over full attention.

## 1 INTRODUCTION

Large language models increasingly operate over long and evolving contexts for document understanding, software engineering, and agentic reasoning (Grattafiori et al., 2024; Yang et al., 2025; Jimenez et al., 2024; Bai et al., 2024; Zhou et al., 2024). During autoregressive decoding, however, the KV cache grows with context length, and dense attention reads the entire cache at every step, making long-context inference increasingly bottlenecked by KV cache memory traffic.

To alleviate this bottleneck, query-aware sparse attention reduces this traffic by reading only a small set of tokens selected for the current query (Xiao et al., 2024a; Ribar et al., 2024; Wang et al., 2026). More recent methods divide tokens into blocks (Tang et al., 2024; Liu et al., 2025; Deng et al., 2026) and further compensate for the attention contributions of omitted blocks using compact token summaries (Hooper et al., 2025; Yang et al., 2026; Fan et al., 2026). Despite recent progress in compensation, selection and reconstruction remain largely decoupled: blocks are ranked by predicted relevance or attention mass before reconstruction is considered. This design overlooks how well each block can be reconstructed by the compensator. A high-mass block may be easy to reconstruct, whereas a lower-mass block with greater within-block logit variation may leave a larger error. Therefore, selection should prioritize the blocks that have a larger estimated reconstruction error.

Motivated by this mismatch, we introduce CompKV, a training-free, compensation-aware KV selection method that prioritizes blocks that are difficult to reconstruct, as shown in Figure 1. To the best of our knowledge, CompKV is the first sparse attention framework to make block selection explicitly aware of the error induced by the downstream compensation mechanism. We formulate full block attention and its summary-based approximation as distributions whose KL divergence reduces to a log-partition gap. For block-mean reconstruction, our analysis reveals that the residual contributed by an omitted block depends jointly on its attention mass and withinblock logit variation. Based on this result, CompKV estimates the post-compensation residual and then converts compensation-aware selection into a query-dependent block-ranking problem.

![](images/601b07a42c744bb7a0f7f7763bbc80927ba999048a77a8c86ee9e8ddce151fe6.jpg)  
Figure 1: Comparison between full attention (a), attention-mass selection (b), and CompKV (c) on two illustrative blocks B1 and BX. Mass-based selection may spend exact reads on blocks already well approximated, whereas CompKV targets blocks with the largest estimated reconstruction error.

CompKV maintains compact mean and grouped-variance statistics for each KV block to estimate its post-compensation residual. For each query, these statistics approximate the block attention mass and within-block logit variation, which are combined into a selection score. Blocks with the largest scores receive exact attention, while the rest contribute through their mean summaries; both contributions are jointly normalized to produce the final output. For efficient long-context inference, we further develop an asynchronous CPU-offload implementation that overlaps summary transfer and compensation with block selection, CPU-side KV gathering, and selected-KV transfer.

We evaluate CompKV in terms of both accuracy and efficiency. For accuracy, we benchmark CompKV on RULER and LongBench-Pro using Llama-3.1-8B-Instruct, Qwen3-8B, and Qwen3- 32B. Across both benchmarks, CompKV achieves the highest average scores among the evaluated sparse methods for all three models. For efficiency, we benchmark the CPU-offload single-layer pipeline on an NVIDIA H100 GPU. CompKV achieves the lowest mean decoding-step latency among the evaluated methods for all nine combinations of 32K-128K context lengths and 512-2,048 token budgets, delivering up to 6.85× speedups over full attention.

## Our contributions are:

• We identify the mismatch between selection and compensation and present the first formulation that explicitly models selection with respect to compensation. Our analysis shows that the postcompensation residual is jointly governed by attention mass and within-block logit variation.

• Based on this analysis, we introduce CompKV, the first training-free, compensation-aware KV selection method that constructs a query-dependent score from compact block statistics, along with an efficient asynchronous implementation to accelerate long-context inference.

• We comprehensively evaluate the accuracy and efficiency of CompKV. Across three widely used models and two long-context benchmarks, CompKV achieves the highest average scores among the evaluated baselines while delivering up to 6.85× self-attention speedups over full attention.

## 2 RELATED WORK

KV Cache Reduction for Long-Context Inference. To accelerate long-context inference, eviction methods bound the cache using recency, accumulated attention, heavy-hitter statistics, or promptside importance estimates (Xiao et al., 2024b; Liu et al., 2023; Zhang et al., 2023; Li et al., 2024; Cai et al., 2025; Feng et al., 2025). Complementary approaches reduce retained-state costs through lowrank compression (Saxena et al., 2024; Singhania et al., 2024; Chang et al., 2025), quantization (Liu et al., 2024; Hooper et al., 2024; Kang et al., 2024; Li et al., 2025), or memory management (Kwon et al., 2023; Qu et al., 2025). These methods primarily reduce the amount or representation cost of retained KV states. In contrast, our work focuses on deciding which KV blocks should be read exactly for each query under a fixed per-step read budget.

Query-Aware and Block-Level Sparse Attention. Query-aware sparse attention saves all KVs and conditions exact KV reads on the current decoding query. For example, Quest (Tang et al., 2024) ranks pages with query-dependent upper bounds derived from compact key statistics, while Infini-Gen (Lee et al., 2024), ShadowKV (Sun et al., 2025), SpeCache (Jie et al., 2025) and RetrievalAttention (Liu et al., 2025) retrieve candidate blocks using different scoring mechanisms. However, these methods primarily rank blocks according to their predicted attention mass for the current query.

Tail Compensation and Reconstruction. Tail-compensation methods approximate omitted KV states or their aggregate attention contribution. SparQ (Ribar et al., 2024) reallocates estimated omitted mass to a running mean value, RESA (Yang et al., 2026) reconstructs the tail from a lowrank logit prior, and ResKV (Zhan et al., 2026) retains compact moment or residual summaries for compressed caches. These methods partly reconstruct the tail without extra exact KV reads.

Nevertheless, existing block selection and tail compensation are largely treated as separate stages, as mentioned in Section 1. In contrast, CompKV explicitly selects blocks according to their estimated post-compensation residual, rather than simply those with the largest predicted attention mass.

## 3 COMPENSATION-AWARE SELECTION FORMULATION

In this section, we formulate sparse attention selection and derive a compensation-aware selection criterion. We define Mean compensation as replacing every logit in an unselected block by its block mean. We use Mean compensation because its compact summaries yield an exact logpartition identity, making the interaction between selection and compensation analytically tractable. The compensation-aware formulation itself is not restricted to Mean compensation. Based on this, we analyze the residual introduced by compensation and show that the optimal selection priority depends jointly on a block’s attention mass and the variation of its within-block attention logits.

## 3.1 MEAN COMPENSATION MODEL AND SELECTION OBJECTIVE

Attention and Mean Compensation. At each decoding step, we consider one KV head shared by G query heads under grouped-query attention (Ainslie et al., 2023). Let $q _ { g } , k _ { j } \in \mathbb { R } ^ { d }$ denote the query of head $g \in \{ 1 , \ldots , G \}$ and the key at token position j, respectively, where d is the head dimension. The corresponding attention logit is $z _ { g , j } = \mathbf { \mathbf { \mathbf { q } } } _ { g } \mathbf { \mathbf { k } } _ { j } ^ { \top } / \sqrt { d } .$ Following Tang et al. (2024), we divide the KV cache into contiguous blocks of a fixed size, starting from the first token. Let B denote the set of nonempty blocks, where each block $b \in B$ contains |b| tokens. Its unnormalized attention weight is $\begin{array} { r } { Z _ { g , b } = \sum _ { j \in b } e ^ { z _ { g , j } } } \end{array}$ , giving the full attention probability $P _ { g } = \mathrm { s o f t m a x } _ { j } ( z _ { g , j } )$

Given a selected block set $\cal { S } \subseteq \cal { B }$ shared by the $G$ query heads, for each token position $j \in b ,$ , the logit under Mean compensation is

$$
\hat { z } _ { g , j } = \left\{ \begin{array} { l l } { z _ { g , j } , } & { b \in \mathcal { S } , } \\ { \bar { z } _ { g , b } , } & { b \notin \mathcal { S } , } \end{array} \right.\tag{1}
$$

where $\begin{array} { r } { \bar { z } _ { g , b } = | b | ^ { - 1 } \sum _ { j \in b } z _ { g , j } } \end{array}$ is the block-mean logit. Therefore, applying softmax jointly over all token positions defines the compensated distribution $\widehat { P } _ { g } ( S ) = \operatorname { s o f t m a x } _ { j } ( \widehat { z } _ { g , j } )$

Optimization objective. For Mean compensation, we define the selection loss as ${ \mathcal { L } } ( S ) ~ =$ $\begin{array} { r } { \sum _ { g = 1 } ^ { G } D _ { \mathrm { K L } } \Big ( \widehat { P } _ { g } ( S ) \lVert P _ { g } \Big ) } \end{array}$ , summing over the query heads that share the selected set. With a budget of $K \leq | { \dot { B } } |$ exact block reads, our objective is

$$
\operatorname* { m i n } _ { { \cal S } \subseteq { \cal B } , | { \cal S } | = { \cal K } } { \mathscr { L } ( { \cal S } ) } .\tag{2}
$$

## 3.2 DERIVING THE COMPENSATION-AWARE SELECTION CRITERION

For each query head $^ { g , }$ define the attention mass and logit variance of block b as

$$
p _ { g , b } = \frac { Z _ { g , b } } { \displaystyle \sum _ { c \in \mathcal { B } } Z _ { g , c } } , \qquad \sigma _ { g , b } ^ { 2 } = \frac { 1 } { | b | } \displaystyle \sum _ { j \in b } ( z _ { g , j } - \bar { z } _ { g , b } ) ^ { 2 } ,\tag{3}
$$

where c ranges over all available blocks and $\sigma _ { g , b } \geq 0$ denotes the corresponding standard deviation. To solve Equation 2, we expand the KL divergence defined in Section 3.1, yielding

$$
\mathcal { L } ( \boldsymbol { S } ) = \sum _ { g = 1 } ^ { G } \log \frac { \displaystyle \sum _ { b \in \boldsymbol { B } } Z _ { g , b } } { \displaystyle \sum _ { b \in \boldsymbol { S } } Z _ { g , b } + \sum _ { b \notin \boldsymbol { S } } | b | e ^ { \bar { z } _ { g , b } } } = - \sum _ { g = 1 } ^ { G } \log \left[ 1 - \sum _ { b \notin \boldsymbol { S } } p _ { g , b } \left( 1 - \frac { | b | e ^ { \bar { z } _ { g , b } } } { Z _ { g , b } } \right) \right] .\tag{4}
$$

Appendix A.1 provides the derivation. The ratio $| b | e ^ { \bar { z } _ { g , b } } / Z _ { g , b }$ measures the fraction of block $b \mathbf { \hat { s } }$ unnormalized attention weight recovered by Mean compensation. Appendix B.1 provides empirical support for this selection principle: ranking blocks by their exact compensation residuals reduces average attention KL and output error relative to ranking by exact attention mass.

A second-order Taylor expansion around the block-mean logit yields

$$
p _ { g , b } \left( 1 - \frac { | b | e ^ { \bar { z } _ { g , b } } } { Z _ { g , b } } \right) = p _ { g , b } ( \frac { 1 } { 2 } \sigma _ { g , b } ^ { 2 } + O ( \sigma _ { g , b } ^ { 3 } ) ) .\tag{5}
$$

Thus, the leading contribution of each block is proportional to its attention mass multiplied by its logit variance. Appendix A.2 derives this expansion and specifies the expansion regime.

Let $R _ { g } ( S )$ denote the inner sum in Equation 4. Applying $- \log ( 1 - R _ { g } ) = R _ { g } + O ( R _ { g } ^ { 2 } )$ to each head in Equation 4 then yields

$$
\mathcal { L } ( \boldsymbol { S } ) = \frac { 1 } { 2 } \sum _ { g = 1 } ^ { G } \sum _ { b \not \in \boldsymbol { S } } p _ { g , b } ( \sigma _ { g , b } ^ { 2 } + O \big ( \sigma _ { g , b } ^ { 3 } \big ) ) .\tag{6}
$$

The remainder collects the blockwise expansion errors and the outer-logarithm remainder, which is absorbed into the cubic term in the small-variance regime. Appendix A.3 provides the derivation.

The second-order term is additive across unselected blocks. Since the sum over all blocks is inde pendent of S, minimizing the second-order term of Equation 6 is equivalent to

$$
\operatorname* { m a x } _ { \substack { s \subseteq B , | S | = K } } \sum _ { b \in S } \sum _ { g = 1 } ^ { G } p _ { g , b } \sigma _ { g , b } ^ { 2 } .\tag{7}
$$

Thus, selecting the K blocks with the largest $\textstyle \sum _ { g = 1 } ^ { G } p _ { g , b } \sigma _ { g , b } ^ { 2 }$ minimizes the second-order objective. The resulting score combines a block’s attention mass with the variation that Mean compensation leaves unreconstructed. A block with nearly constant logits contributes little to the loss, even when its attention mass is large. Appendix A.5 also proves that removing Mean compensation from our KL objective recovers the standard Top-K selection rule based on attention mass for a single query head. Section 4.1 estimates both factors from compact block statistics.

## 4 COMPKV

Building on the criterion derived in Section 3, CompKV turns the oracle selection rule into a practical decoding pipeline. It first estimates block attention mass and logit variance from compact statistics, uses their product to allocate exact reads, and represents the remaining blocks with mean summaries under a shared normalization. To realize it efficiently, we develop an implementation that overlaps compensation with KV retrieval and exact attention.

![](images/2ef73f5b965305088e9b036032187cf5e44bfd4af839523cd91039fba0f856c1.jpg)  
Figure 2: End-to-end CompKV decoding, where $X = | \boldsymbol { B } |$ is the number of available blocks. Stage 1 ranks blocks with Equation 10. Stage 2 evaluates selected blocks exactly, compensates the remaining blocks with their means, and normalizes all contributions jointly.

## 4.1 BLOCK SCORING FROM COMPACT STATISTICS

Grouped key-variance estimation. Computing the exact score in Equation 7 requires reading the keys of every candidate block. CompKV estimates this score from the mean key and grouped key variances stored for each block. The mean key $\begin{array} { r } { \bar { \pmb { k } } _ { b } = | b | ^ { - 1 } \sum _ { j \in b } \pmb { k } _ { j } } \end{array}$ gives the block-mean logit $\bar { z } _ { g , b } = \mathbf { { q } } _ { g } \bar { \mathbf { { k } } } _ { b } ^ { \top } / \sqrt { d }$ defined in Section 3.1. We partition the d key coordinates into r disjoint groups $\mathcal { G } _ { t } ,$ indexed by $t = 0 , \ldots , r - 1$ . The vector $\bar { \mathbf Ḋ \mathbf Ḋ b Ḍ Ḍ } \in \mathbb { R } ^ { r }$ stores the average coordinate variance in each group. At each decoding step, query head g estimates the block’s logit variance as

$$
\widehat { \sigma } _ { g , b } ^ { 2 } = \frac { 1 } { d } \sum _ { t = 0 } ^ { r - 1 } \left( \sum _ { i \in \mathcal { G } _ { t } } q _ { g , i } ^ { 2 } \right) ( \bar { \mathbf { D } } _ { b } ) _ { t } , \qquad ( \bar { \mathbf { D } } _ { b } ) _ { t } = \frac { 1 } { | b | | \mathcal { G } _ { t } | } \sum _ { j \in b } \sum _ { i \in \mathcal { G } _ { t } } ( k _ { j , i } - \bar { k } _ { b , i } ) ^ { 2 } .\tag{8}
$$

Here, $q _ { g , i } , k _ { j , i }$ , and $\bar { k } _ { b , i }$ denote the ith coordinates of $q _ { g } , k _ { j }$ , and $\bar { k } _ { b } .$ , respectively. The parameter r controls the granularity of the stored variances. Appendix A.4 specifies the coordinate groups and derives this estimate from the key covariance within a block along the query.

Attention-mass estimation. The log-partition expansion in Equation 23 gives log $\left| b \right| + \bar { z } _ { g , b } +$ $\sigma _ { g , b } ^ { 2 } / 2$ as the second-order approximation to log $Z _ { g , b }$ . Substituting $\widehat { \sigma } _ { g , b } ^ { 2 }$ and normalizing the estimated block weights within each query head gives the attention-mass estimate $\widehat { p } _ { g , b } \colon$

$$
\widehat { p } _ { g , b } = \frac { \displaystyle \left| b \right| \exp \left( \bar { z } _ { g , b } + \frac { 1 } { 2 } \widehat { \sigma } _ { g , b } ^ { 2 } \right) } { \displaystyle \sum _ { c \in \mathcal { B } } \left| c \right| \exp \left( \bar { z } _ { g , c } + \frac { 1 } { 2 } \widehat { \sigma } _ { g , c } ^ { 2 } \right) } .\tag{9}
$$

![](images/19b0ec5bd234c3d8c22b476059d16b334f69dd925c968087a1fe56d57d667f07.jpg)  
Figure 3: Implementation of CPU-offload CompKV. (a) Dataflow across the main stream, auxiliary stream, and CPU. (b) Asynchronous schedule: after mean values and selection are ready, Mean compensation runs on an auxiliary stream while the main path calculates the exact attention. The branches join at the final merge. Dashed arrows indicate dependencies; durations are schematic.

The variance correction accounts for within-block logit variation when estimating the block’s unnormalized attention weight. A detailed derivation is provided in Appendix A.4.

Compensation-aware block scoring. Replacing the mass and variance in Equation 7 with these estimates defines the CompKV score $S _ { b }$ for block b:

$$
S _ { b } = \sum _ { g = 1 } ^ { G } \widehat { p } _ { g , b } \widehat { \sigma } _ { g , b } ^ { 2 } .\tag{10}
$$

The two uses of $\widehat { \sigma } _ { g , b } ^ { 2 }$ follow from the preceding derivations: the correction inside $\widehat { p } _ { g , b }$ estimates the block’s attention mass, and the outer variance factor captures the leading dependence of the compensation residual on logit variation. CompKV computes $\widehat { \sigma } _ { g , b } ^ { 2 }$ only once and then reuses it. To evaluate the effectiveness of the score, we measure how closely it follows exact residual selection using Spearman correlation and residual capture. The results are shown in Appendix B.2.

## 4.2 BLOCK SELECTION AND COMPENSATED DECODING

Figure 2 shows the end-to-end decoding workflow of CompKV. Mean compensation additionally stores $\begin{array} { r } { \bar { \pmb { v } } _ { b } = | b | ^ { - 1 } \sum _ { j \in b } \pmb { v } _ { j } } \end{array}$ , where ${ \pmb v } _ { j } \in \mathbb { R } ^ { d }$ . The complete summary $\{ | b | , \bar { k } _ { b } , \bar { \mathbf { D } } _ { b } , \bar { v } _ { b } \}$ contains $2 d + r + 1$ scalars per block and is updated incrementally as new K/V states enter the active block.

The mandatory set $\mathcal { F }$ contains the first sink block and the two most recent blocks, counting any overlap once. These blocks count toward the total budget, and Equation 10 ranks the other blocks:

$$
S = { \mathcal { F } } \cup \mathrm { T o p K } _ { K - \mid { \mathcal { F } } \mid } \left\{ S _ { b } : b \in { \mathcal { B } } \setminus { \mathcal { F } } \right\} , \qquad | S | = K .\tag{11}
$$

Here, TopK returns the blocks with the highest scores. Selected blocks use their original K/V states, while unselected blocks use their mean summaries. Both contributions are normalized together:

$$
\widehat { \mathbf { o } } _ { g } = \frac { \displaystyle \sum _ { b \in S } \displaystyle \sum _ { j \in b } e ^ { z _ { g , j } } \pmb { v } _ { j } + \displaystyle \sum _ { b \notin S } \vert b \vert e ^ { \bar { z } _ { g , b } } \bar { \pmb { v } } _ { b } } { \displaystyle \sum _ { b \in S } Z _ { g , b } + \displaystyle \sum _ { b \notin S } \vert b \vert e ^ { \bar { z } _ { g , b } } } .\tag{12}
$$

## 4.3 EFFICIENT IMPLEMENTATION OF COMPKV

In practical LLM deployments, the KV cache may need to be stored outside GPU memory. Therefore, CompKV keeps the complete BF16 KV cache and FP32 block summaries in pinned CPU memory. Each step uses a fused GPU kernel to update the current block’s statistics and pack the new K/V, followed by one device-to-host (D2H) transfer and CPU writeback. The summary buffer separates selection statistics from mean values. The main CUDA stream transfers the selection region and runs two kernels for block statistics and normalized Top-K scoring, while an auxiliary stream independently transfers mean values. The selector reuses grouped variance projection in both factors of Equation 10 to estimate both block mass and residual magnitude.

After selection, the CPU gathers the selected K/V into a contiguous pinned buffer for one host-todevice (H2D) transfer and FlashInfer exact attention. Meanwhile, the auxiliary stream computes Mean compensation once the mean values and selected-block mask are ready, overlapping this computation with CPU gathering, selected-KV transfer, and exact attention. The main stream waits for compensation immediately before merging both contributions under the shared normalization in Equation 12. Figure 3 shows the schedule; Appendix E.1 details implementation and measurement.

Table 1: RULER accuracy (%). Bold values mark the best sparse result per model and column, including ties. AVG annotations show drops (↓) from Full in percentage points.
<table><tr><td>Method</td><td>SG1</td><td>SG2</td><td>SG3 MK1 MK2 MK3 MV</td><td></td><td></td><td></td><td>MQ</td><td colspan="2">VT CWE FWE QA1 QA2 AVG</td><td></td><td></td><td></td></tr><tr><td>Llama-3.1-8B-Instruct</td><td colspan="10"></td></tr><tr><td>Full</td><td>100.0 100.0</td><td></td><td>100.0 100.0</td><td></td><td>99.8</td><td>99.2 94.9</td><td>99.1</td><td>99.1</td><td>9.9</td><td>93.3 81.4</td><td>54.4 87.0</td><td></td></tr><tr><td>Quest</td><td>100.0</td><td>99.6</td><td>99.6</td><td>99.8</td><td>92.0</td><td>22.4 86.9</td><td>96.8</td><td>94.4</td><td>1.7</td><td>80.7</td><td>76.2 50.8 77.0↓10.0</td><td></td></tr><tr><td>InfLLM</td><td>100.0</td><td>95.4</td><td>78.2</td><td>91.0</td><td>92.0</td><td>38.0 64.0</td><td>83.0</td><td>90.6</td><td>2.7</td><td>84.4</td><td>80.0</td><td>51.4 73.1↓13.9</td></tr><tr><td>Quest+RESA 100.0</td><td></td><td>99.4</td><td>99.6</td><td>99.8</td><td>93.8</td><td>29.4 91.8</td><td>97.2</td><td>97.1</td><td>0.1</td><td>82.5</td><td>75.4</td><td>50.6 78.2↓8.8</td></tr><tr><td>CompKV</td><td>100.0</td><td>99.6</td><td>99.8</td><td>98.6</td><td>97.8</td><td>73.0 91.2</td><td>97.1</td><td>97.2</td><td>0.5</td><td>91.7</td><td>80.8</td><td>54.0 83.2↓3.8</td></tr><tr><td></td><td colspan="10"></td></tr><tr><td>Full</td><td>100.0 100.0</td><td></td><td>100.0</td><td>98.6</td><td>97.4</td><td>Qwen3-8B 98.8 96.2</td><td></td><td>97.5 100.0</td><td>82.7</td><td>92.3</td><td>71.8</td><td>54.0 91.5</td></tr><tr><td>Quest</td><td>100.0 100.0</td><td></td><td>97.6</td><td>98.2</td><td>77.8</td><td>3.4 82.0</td><td>95.3</td><td>99.7</td><td>34.8</td><td>93.3</td><td>64.0</td><td>49.0 76.5↓15.0</td></tr><tr><td>InfLLM</td><td>100.0</td><td>94.0</td><td>76.0</td><td>94.0</td><td>89.6</td><td>42.8 81.9</td><td>85.9</td><td>99.8</td><td>20.0</td><td>88.4</td><td>70.6</td><td>52.6 76.6↓14.9</td></tr><tr><td>Quest+RESA 100.0 100.0</td><td></td><td></td><td>97.2</td><td>97.6</td><td>75.8</td><td>4.8 85.6</td><td>95.7</td><td>99.6</td><td>21.1</td><td>93.9</td><td>65.2</td><td>48.6 75.8↓15.7</td></tr><tr><td>CompKV</td><td>100.0 100.0</td><td></td><td>100.0</td><td>98.6</td><td>96.4</td><td>75.0 95.7</td><td>97.2</td><td>99.8</td><td>35.2</td><td>94.9</td><td>71.8</td><td>53.4 86.0↓5.5</td></tr><tr><td></td><td colspan="10"></td></tr><tr><td>Full</td><td>100.0</td><td>98.4</td><td>100.0</td><td>99.8</td><td></td><td>Qwen3-32B 99.8 100.0 99.4 100.0</td><td></td><td>99.7</td><td>87.5</td><td>93.3</td><td>80.0</td><td>60.4 93.7</td></tr><tr><td>Quest</td><td>100.0</td><td>99.2</td><td>94.4</td><td>100.0</td><td>35.2</td><td>2.4 95.8</td><td>99.8</td><td>98.4</td><td>35.5</td><td>93.1</td><td>74.8</td><td>55.8 75.7↓18.0</td></tr><tr><td>InfLLM</td><td>100.0</td><td>95.8</td><td>82.4</td><td>93.4</td><td>91.6</td><td>68.0 85.3</td><td>90.1</td><td>99.4</td><td>34.9</td><td>93.3</td><td>81.0</td><td>57.0 82.5↓11.2</td></tr><tr><td>Quest+RESA</td><td>97.6</td><td>98.0</td><td>93.0</td><td>100.0</td><td>38.2</td><td>3.2 94.8</td><td>99.4</td><td>98.1</td><td>39.5</td><td>91.3</td><td>74.6</td><td>55.4 75.6↓ 18.1</td></tr><tr><td>CompKV</td><td>100.0</td><td></td><td>99.2 100.0</td><td>99.8</td><td>99.4</td><td>93.0 98.5</td><td>99.6</td><td>99.2</td><td>56.0</td><td>96.7</td><td></td><td>81.4 60.2 91.0↓ 2.7</td></tr></table>

## 5 EXPERIMENTS

## 5.1 SETUP

Models. To comprehensively evaluate our proposed CompKV, we consider three models that are widely used in sparse attention studies: Llama-3.1-8B-Instruct (Grattafiori et al., 2024), Qwen3-8B, and Qwen3-32B (Yang et al., 2025). Qwen3-8B and Qwen3-32B are used to assess generalization across model scales, while Llama-3.1-8B-Instruct is included to further examine whether CompKV generalizes across different model families and architectural designs.

Methods. To verify the effectiveness of CompKV, we compare it against Quest (Tang et al., 2024), InfLLM (Xiao et al., 2024a), and Quest+RESA with λ = 0.25 (Yang et al., 2026). Quest and InfLLM are state-of-the-art training-free, query-aware baselines for block selection. CompKV differs from these methods in two aspects: how blocks are selected for exact attention computation and how the unselected blocks are handled. RESA is a plug-in compensation method that can be integrated with different sparse-attention methods; following its original evaluation setup, we use Quest+RESA as the RESA baseline. As defined in Section 4.1, r denotes the number of variance groups, and we set r = 4 for all main CompKV results. Full attention is included as the dense reference.

Benchmarks and metrics. RULER (Hsieh et al., 2024) provides controlled retrieval, tracing, and aggregation tasks to test whether CompKV preserves long-context capabilities under tight exact-read budgets. LongBench-Pro (Chen et al., 2026) assesses whether these gains extend to more realistic tasks requiring either localized evidence retrieval or global context integration. We evaluate RULER at 32K and LongBench-Pro at context lengths up to 128K using their respective official scorers.

Additional configuration. All evaluations are conducted on NVIDIA H100 80GB HBM3 GPUs with a batch size of one. For a consistent comparison, all four sparse methods partition consecutive tokens into blocks of size 16 and retain the first block as an attention sink together with the two most recent blocks as a local attention window. These mandatory blocks are always included in the selected set and count toward the same per-step exact-read budget. Appendix C provides further details on the complete experimental setup and evaluation configurations.

Table 2: LongBench-Pro scores. Mod. and Extr. abbreviate Moderate and Extreme. Bold marks the best sparse result per model and column. AVG arrows show point drops from Full.
<table><tr><td rowspan="2">Method</td><td colspan="4">Difficulty</td><td colspan="5">Context length</td><td rowspan="2">AVG</td></tr><tr><td>Easy</td><td>Mod.</td><td>Hard</td><td>Extr.</td><td>8k</td><td>16k</td><td>32k</td><td>64k</td><td>128k</td></tr><tr><td colspan="10">Llama-3.1-8B-Instruct</td></tr><tr><td>Full</td><td>28.30</td><td>18.88</td><td>25.61</td><td>25.72</td><td>30.46</td><td>25.55</td><td>25.73</td><td>22.59</td><td>21.98</td><td>25.26</td></tr><tr><td>Quest</td><td>28.30</td><td>17.71</td><td>22.66</td><td>24.75</td><td>28.09</td><td>25.54</td><td>24.47</td><td>21.55</td><td>21.75</td><td>24.28↓0.98</td></tr><tr><td>InfLLM</td><td>27.56</td><td>17.69</td><td>21.84</td><td>23.72</td><td>28.31</td><td>24.53</td><td>24.07</td><td>20.72</td><td>20.35</td><td>23.60↓1.66</td></tr><tr><td>Quest+RESA</td><td>27.51</td><td>18.09</td><td>24.38</td><td>24.54</td><td>27.95</td><td>24.34</td><td>25.94</td><td>21.73</td><td>21.54</td><td>24.30↓0.96</td></tr><tr><td>CompKV</td><td>27.91</td><td>18.05</td><td>23.94</td><td>24.83</td><td>28.63</td><td>25.36</td><td>25.30</td><td>21.50</td><td>21.39</td><td>24.44↓0.82</td></tr><tr><td colspan="9">Qwen3-8B</td></tr><tr><td>Full</td><td>39.63</td><td>29.36</td><td>32.36</td><td>31.62</td><td>38.68</td><td>35.86</td><td>34.88</td><td>29.53</td><td>32.21</td><td>34.23</td></tr><tr><td>Quest</td><td>34.54</td><td>27.14</td><td>28.68</td><td>28.89</td><td>33.83</td><td>29.54</td><td>30.50</td><td>27.84</td><td>31.19</td><td>30.58↓3.65</td></tr><tr><td>InfLLM</td><td>36.18</td><td>29.09</td><td>29.82</td><td>28.83</td><td>35.01</td><td>30.87</td><td>33.43</td><td>27.13</td><td>32.31</td><td>31.75↓2.48</td></tr><tr><td>Quest+RESA</td><td>34.54</td><td>27.82</td><td>28.63</td><td>28.46</td><td>33.90</td><td>29.96</td><td>31.37</td><td>27.25</td><td>30.50</td><td>30.60↓3.63</td></tr><tr><td>CompKV</td><td>37.17</td><td>30.09</td><td>30.68</td><td>29.89</td><td>35.74</td><td>32.82</td><td>33.77</td><td>28.17</td><td>33.19</td><td>32.74↓1.49</td></tr><tr><td colspan="9">Qwen3-32B</td></tr><tr><td>Full</td><td>50.88</td><td>39.11</td><td>39.17</td><td>33.63</td><td>48.20</td><td>44.45</td><td>41.56</td><td>38.72</td><td>37.09</td><td>42.00</td></tr><tr><td>Quest</td><td>44.93</td><td>33.93</td><td>35.76</td><td>31.43</td><td>41.67</td><td>35.15</td><td>39.00</td><td>36.22</td><td>36.06</td><td>37.62↓4.38</td></tr><tr><td>InfLLM</td><td>48.07</td><td>36.44</td><td>37.18</td><td>31.80</td><td>44.63</td><td>38.75</td><td>42.02</td><td>36.54</td><td>36.15</td><td>39.62↓2.38</td></tr><tr><td>Quest+RESA</td><td>43.61</td><td>31.93</td><td>35.11</td><td>29.69</td><td>41.87</td><td>35.24</td><td>38.35</td><td>35.07</td><td>30.28</td><td>36.16↓5.84</td></tr><tr><td>CompKV</td><td>49.56</td><td>35.23</td><td>38.62</td><td>32.36</td><td>45.49</td><td>39.52</td><td>41.09</td><td>38.28</td><td>37.15</td><td>40.30↓1.70</td></tr></table>

## 5.2 ACCURACY RESULTS

We evaluate how well CompKV preserves task quality by comparing overall scores, task-level behavior, and sensitivity to the KV budget. With r = 4, CompKV achieves the highest AVG for all three models on RULER at a 32K length and a 512-token budget (Table 1), as well as on LongBench-Pro (Table 2). It also obtains the best or tied-best sparse scores in most task-level comparisons.

The gains are particularly pronounced on MK3, where the accuracy of CompKV is significantly higher than that of other baselines. This improvement suggests that compensation-aware selection helps preserve the information needed for difficult retrieval tasks. However, on Llama-3.1, Full attention scores only 9.9% on CWE, and all evaluated sparse methods remain low, which means the aggregate improvements coexist with tasks that remain challenging even for the dense model.

The advantage persists as the exact-read budget varies. We further evaluate CompKV with token budgets ranging from 128 to 1024 for both 8B models on RULER. Results are shown in Appendix D. The curves show that several baselines approach Full as the budget increases, but CompKV remains the best across most budgets, tasks and models.

## 5.3 EFFICIENCY EVALUATION

To further evaluate the efficiency of our CompKV implementation, we follow the profiling setup of Quest (Tang et al., 2024) and measure the complete CPU-offload attention step on a single NVIDIA H100 GPU with a batch size of one. We benchmark Full, Quest, InfLLM, and CompKV with $r \in \{ 4 , 1 2 8 \}$ under the same evaluation setting. We do not include Quest+RESA in the latency evaluation because its original paper reports higher latency than Quest. The evaluation covers endpoint lengths of 32K, 64K, and 128K and selected-token budgets of 512, 1,024, and 2,048. CompKV $( r = 4 )$ achieves the lowest mean latency in all nine settings, with up to 6.85× speedups over Full. Figure 4 shows the 512-token budget; Appendix E.2 gives the full protocol and results.

![](images/420cd1f6c8fc8a76937e78dd8e4290b6164c7ea9c8df096c91a6389e396489fb.jpg)  
32K

![](images/a7dbf6aa5ba6a13dd3eb8bd045154a9b1a9085d36d36000b364e0d59582ad2fc.jpg)  
64K

![](images/8a3a47a186acb238a63fa43709c7eaa0eec845f697e37ffe827fbc786cdead85.jpg)  
128K  
Figure 4: CPU-offload attention latency. Mean host wall time per single-layer attention step at a 512-token budget. Lengths denote the endpoints of the 256-step decode windows.

## 5.4 ABLATION STUDIES

Ablation objectives. We examine Mean compensation, the outer variance factor, and the secondorder mass correction to determine how tail reconstruction and residual-aware scoring contribute to the final task quality. We also vary the number of variance groups r to assess how compact block statistics affect accuracy and metadata cost.

Factor ablations. Table 3 compares four configurations with $r \ = \ 4 .$ Here, $\widehat { p } ^ { ( 2 ) }$ includes the variance correction in Equation 9, while ${ \widehat { p } } ^ { ( 1 ) }$ uses meankey logits alone. Adding Mean compensation improves AVG by 0.48–2.47 points, while the outer variance factor provides a further 0.35–1.19 points. The second-order mass correction contributes an additional 1.96–3.39 points over its first-order counterpart. Overall, the results indicate that compensation, residual-aware selection, and second-order mass estimation all contribute consistently, with CompKV achieving the highest AVG in all four settings.

Variance-group granularity. Figure 5 shows that accuracy generally improves with finer variance groups, with $r = 1 2 8$ achieving the highest AVG at both budgets. Our default $r = 4$ trails this endpoint by 1.87 and 1.08 points at budgets 256 and $5 1 { \bar { 2 } } ,$ respectively, while requiring only 1/32 of its variance metadata. This trade-off motivates the use of compact grouped statistics in the main experiments.

Table 3: Factor ablations on RULER. Values are 13-task AVG (%) for Llama-3.1-8B-Instruct and Qwen3-8B with $r = 4 .$ . Column labels denote selected-token budgets.
<table><tr><td rowspan="2">Variant</td><td colspan="2">Llama</td><td colspan="2">Qwen</td></tr><tr><td>256</td><td>512</td><td>256</td><td>512</td></tr><tr><td>CompKV</td><td>78.6</td><td>83.2</td><td>80.9</td><td>86.0</td></tr><tr><td> $\widehat { p } ^ { ( 2 ) }$  only</td><td>75.6</td><td>81.8</td><td>79.2</td><td>83.5</td></tr><tr><td>p(2) + Mean</td><td>78.1</td><td>82.8</td><td>79.7</td><td>85.1</td></tr><tr><td> $\widehat { p } ^ { ( 1 ) } \widehat { \sigma } ^ { 2 } + \mathrm { M e a n }$ </td><td>76.1</td><td>81.2</td><td>77.5</td><td>83.8</td></tr></table>

![](images/f87e1577357c389d466c32cf5613977b458c4828d1db5cfecbe1ce0eea279a34.jpg)  
Figure 5: Variance-group granularity on RULER. Curves show the 13-task AVG for Llama-3.1-8B-Instruct at selected-token budgets of 256 and 512.

## 6 CONCLUSION

In this paper, we identify the mismatch between KV block selection and downstream compensation and introduce CompKV, a compensation-aware method that selects blocks according to their estimated post-compensation residual. Our analysis shows that, under Mean compensation, this residual is jointly governed by block attention mass and within-block logit variation, which CompKV estimates from compact block statistics. Across three models and two long-context benchmarks, CompKV achieves the highest average scores among the evaluated sparse methods, while its asynchronous implementation delivers up to 6.85× self-attention speedups over full attention.

## REFERENCES

Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit Sanghai. GQA: Training generalized multi-query transformer models from multi-head checkpoints. In The 2023 Conference on Empirical Methods in Natural Language Processing, 2023. URL https://openreview.net/forum?id=hmOwOZWzYE.

Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. LongBench: A bilingual, multitask benchmark for long context understanding. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3119–3137, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.172. URL https://aclanthology.org/2024.acl-long.172/.

Zefan Cai, Yichi Zhang, Bofei Gao, Yuliang Liu, Yucheng Li, Tianyu Liu, Keming Lu, Wayne Xiong, Yue Dong, Junjie Hu, and Wen Xiao. PyramidKV: Dynamic KV cache compression based on pyramidal information funneling. In Second Conference on Language Modeling, 2025. URL https://openreview.net/forum?id=ayi7qezU87.

Chi-Chih Chang, Wei-Cheng Lin, Chien-Yu Lin, Chong-Yan Chen, Yu-Fang Hu, Pei-Shuo Wang, Ning-Chi Huang, Luis Ceze, Mohamed S. Abdelfattah, and Kai-Chiang Wu. Palu: KV-cache compression with low-rank projection. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=LWMS4pk2vK.

Ziyang Chen, Xing Wu, Junlong Jia, Chaochen Gao, Qi Fu, Debing Zhang, and Songlin Hu. Longbench pro: A more realistic and comprehensive bilingual long-context evaluation benchmark, 2026. URL https://arxiv.org/abs/2601.02872.

Keqi Deng, Shaoshi Ling, Ruchao Fan, and Jinyu Li. Unique: Universal top-k sparse attention for training-free inference and sparsity-aware training, 2026. URL https://arxiv.org/abs/ 2605.27740.

Qihang Fan, Huaibo Huang, Zhiying Wu, Bingning Wang, and Ran He. Flashprefill v2: Blocksparse prefill attention for long-context llm serving, 2026. URL https://arxiv.org/abs/ 2608.19758.

Yuan Feng, Junlin Lv, Yukun Cao, Xike Xie, and S Kevin Zhou. Ada-KV: Optimizing KV cache eviction by adaptive budget allocation for efficient LLM inference. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview. net/forum?id=tcisuhGsQZ.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783.

Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W. Mahoney, Yakun Sophia Shao, Kurt Keutzer, and Amir Gholami. Kvquant: Towards 10 million context length llm inference with kv cache quantization. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 1270–1303. Curran Associates, Inc., 2024. doi: 10.52202/ 079017-0040. URL https://proceedings.neurips.cc/paper\_files/paper/ 2024/file/028fcbcf85435d39a40c4d61b42c99a4-Paper-Conference.pdf.

Coleman Richard Charles Hooper, Sebastian Zhao, Luca Manolache, Sehoon Kim, Michael W. Mahoney, Sophia Shao, Kurt Keutzer, and Amir Gholami. Multipole attention for efficient long context reasoning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=5Qe7AGO3Eq.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, and Boris Ginsburg. RULER: What’s the real context size of your long-context language models? In First Conference on Language Modeling, 2024. URL https://openreview.net/forum? id=kIoBbc76Sy.

Shibo Jie, Yehui Tang, Kai Han, Zhi-Hong Deng, and Jing Han. Specache: Speculative key-value caching for efficient generation of LLMs. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=PQIrsaIQdn.

Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R Narasimhan. SWE-bench: Can language models resolve real-world github issues? In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview. net/forum?id=VTF8yNQM66.

Hao Kang, Qingru Zhang, Souvik Kundu, Geonhwa Jeong, Zaoxing Liu, Tushar Krishna, and Tuo Zhao. Gear: An efficient kv cache compression recipe for near-lossless generative inference of llm, 2024. URL https://arxiv.org/abs/2403.05527.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Principles, SOSP ’23, pp. 611–626, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9798400702297. doi: 10.1145/3600006.3613165. URL https: //doi.org/10.1145/3600006.3613165.

Wonbeom Lee, Jungi Lee, Junghwan Seo, and Jaewoong Sim. Infinigen: efficient generative inference of large language models with dynamic kv cache management. In Proceedings of the 18th USENIX Conference on Operating Systems Design and Implementation, OSDI’24, USA, 2024. USENIX Association. ISBN 978-1-939133-40-3.

Xing Li, Zeyu XING, Yiming Li, Linping Qu, Hui-Ling Zhen, Yiwu Yao, Wulong Liu, Sinno Jialin Pan, and Mingxuan Yuan. KVTuner: Sensitivity-aware layer-wise mixed-precision KV cache quantization for efficient and nearly lossless LLM inference. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id= zDwipF6h06.

Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. SnapKV: LLM knows what you are looking for before generation. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id=poE54GOq2l.

Di Liu, Meng Chen, Baotong Lu, Huiqiang Jiang, Zhenhua Han, Qianxi Zhang, Qi Chen, Chengruidong Zhang, Bailu Ding, Kai Zhang, Chen Chen, Fan Yang, Yuqing Yang, and Lili Qiu. Retrievalattention: Accelerating long-context LLM inference via vector retrieval. In The Thirtyninth Annual Conference on Neural Information Processing Systems, 2025. URL https: //openreview.net/forum?id=8z3cOVER4z.

Zichang Liu, Aditya Desai, Fangshuo Liao, Weitao Wang, Victor Xie, Zhaozhuo Xu, Anastasios Kyrillidis, and Anshumali Shrivastava. Scissorhands: Exploiting the persistence of importance

hypothesis for LLM KV cache compression at test time. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id= JZfg6wGi6g.

Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. KIVI: A tuning-free asymmetric 2bit quantization for KV cache. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pp. 32332–32344. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/v235/liu24bz.html.

Guanqiao Qu, Qiyuan Chen, Wei Wei, Zheng Lin, Xianhao Chen, and Kaibin Huang. Mobile edge intelligence for large language models: A contemporary survey. IEEE Communications Surveys & Tutorials, 27(6):3820–3860, 2025.

Luka Ribar, Ivan Chelombiev, Luke Hudlass-Galley, Charlie Blake, Carlo Luschi, and Douglas Orr. SparQ attention: Bandwidth-efficient LLM inference. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 42558–42583. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/v235/ribar24a.html.

Utkarsh Saxena, Gobinda Saha, Sakshi Choudhary, and Kaushik Roy. Eigen attention: Attention in low-rank space for KV cache compression. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 15332–15344, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-emnlp.899. URL https://aclanthology.org/ 2024.findings-emnlp.899/.

Prajwal Singhania, Siddharth Singh, Shwai He, Soheil Feizi, and Abhinav Bhatele. Loki: Low-rank keys for efficient sparse attention. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id=raABeiV71j.

Hanshi Sun, Li-Wen Chang, Wenlei Bao, Size Zheng, Ningxin Zheng, Xin Liu, Harry Dong, Yuejie Chi, and Beidi Chen. ShadowKV: KV cache in shadows for high-throughput long-context LLM inference. In Forty-second International Conference on Machine Learning, 2025. URL https: //openreview.net/forum?id=oa7MYAO6h6.

Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, and Song Han. Quest: queryaware sparsity for efficient long-context llm inference. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org, 2024.

Georgios Tzachristas, Lei Deng, Ioannis Tzachristas, Gong Zhang, and Renhai Chen. A mathematical theory of top-k sparse attention via total variation distance. In 2026 IEEE International Symposium on Information Theory (ISIT), pp. 1–6. IEEE, 2026.

Yifei Wang, Yueqi Wang, Zhenrui Yue, Huimin Zeng, Yong Wang, Ismini Lourentzou, Zhengzhong Tu, Xiangxiang Chu, and Julian McAuley. FASA: FREQUENCY-AWARE SPARSE ATTEN-TION. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=FnSgecCEwg.

Chaojun Xiao, Pengle Zhang, Xu Han, Guangxuan Xiao, Yankai Lin, Zhengyan Zhang, Zhiyuan Liu, and Maosong Sun. InfLLM: Training-free long-context extrapolation for LLMs with an efficient context memory. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024a. URL https://openreview.net/forum?id=bTHFrqhASY.

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. In The Twelfth International Conference on Learning Rep resentations, 2024b. URL https://openreview.net/forum?id=NG7sS51zVF.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report, 2025. URL https://arxiv. org/abs/2505.09388.

Weihao Yang, Hao Huang, Ningke Li, Shihao Wang, Darong Yang, Yanqi Pan, Wen Xia, Shiyi Li, and Xiangyu Zou. RESA: Bringing back what sparse attention ignores with residual estimation. In The Fourteenth International Conference on Learning Representations, 2026. URL https: //openreview.net/forum?id=ktcq26hMCH.

Yuhang Zhan, Lisi Chen, and Shuo Shang. Reskv: Reconstructing omitted attention contributions for fixed-budget kv cache compression, 2026. URL https://arxiv.org/abs/2607. 29591.

Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Re, Clark Barrett, Zhangyang Wang, and Beidi Chen. H2o: Heavyhitter oracle for efficient generative inference of large language models. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/ forum?id=RkRrPp7GKO.

Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. Webarena: A realistic web environment for building autonomous agents. In B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun (eds.), International Conference on Learning Representations, volume 2024, pp. 15585–15606, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/file/ 4410c0711e9154a7a2d26f9b3816d1ef-Paper-Conference.pdf.

## A DETAILED COMPKV DERIVATIONS

## A.1 KL IDENTITY FOR MEAN COMPENSATION

Mean compensation replaces the exact weight of each unselected block with its mean-based weight. Consequently, the selection loss admits both a log-partition form and a form that isolates the unreconstructed weight of each omitted block (Equation 4):

$$
\begin{array} { r l } & { \mathcal { L } ( S ) = \displaystyle \sum _ { g = 1 } ^ { G } \log \frac { \displaystyle \sum _ { b \in \mathcal { B } } Z _ { g , b } } { \displaystyle \sum _ { b \in \mathcal { S } } Z _ { g , b } + \displaystyle \sum _ { b \notin \mathcal { S } } | b | e ^ { \bar { z } _ { g , b } } } } \\ & { \quad \quad = - \displaystyle \sum _ { g = 1 } ^ { G } \log \left[ 1 - \displaystyle \sum _ { b \notin \mathcal { S } } p _ { g , b } \left( 1 - \frac { | b | e ^ { \bar { z } _ { g , b } } } { Z _ { g , b } } \right) \right] . } \end{array}
$$

To establish this identity, fix a query head g and a selected block set S. Write the full and compensated normalization constants as

$$
D _ { g } = \sum _ { b \in \mathcal { B } } Z _ { g , b } , \qquad { \widehat { D } } _ { g } ( \mathcal { S } ) = \sum _ { b \in \mathcal { S } } Z _ { g , b } + \sum _ { b \not \in \mathcal { S } } | b | e ^ { { \overline { { z } } } _ { g , b } } .\tag{13}
$$

The token probabilities are $P _ { g } ( j ) = e ^ { z _ { g , j } } / D _ { g }$ and $\widehat { P } _ { g } ( j ; S ) = e ^ { \widehat { z } _ { g , j } } / \widehat { D } _ { g } ( S )$ . Substituting them into the KL divergence gives

$$
\begin{array} { r } { D _ { \mathrm { K L } } \Big ( \widehat { P } _ { g } ( \boldsymbol { S } ) \| P _ { g } \Big ) = \displaystyle \sum _ { b \in \mathcal { B } } \sum _ { j \in b } \widehat { P } _ { g } ( j ; \boldsymbol { S } ) \left[ \widehat { z } _ { g , j } - z _ { g , j } + \log \frac { D _ { g } } { \widehat { D } _ { g } ( \boldsymbol { S } ) } \right] } \\ { = \log \frac { D _ { g } } { \widehat { D } _ { g } ( \boldsymbol { S } ) } + \displaystyle \sum _ { b \not \in \mathcal { S } } \sum _ { j \in b } \widehat { P } _ { g } ( j ; \boldsymbol { S } ) \left( \bar { z } _ { g , b } - z _ { g , j } \right) . } \end{array}\tag{14}
$$

The second equality follows from $\hat { z } _ { g , j } = z _ { g , j }$ in selected blocks and normalized probabilities.

Within each unselected block, every token has the same compensated probability. The remaining term therefore vanishes:

$$
\sum _ { b \notin S } \sum _ { j \in b } \widehat { P } _ { g } ( j ; \mathcal { S } ) \left( \bar { z } _ { g , b } - z _ { g , j } \right) = \frac { 1 } { \widehat { D } _ { g } ( \mathcal { S } ) } \sum _ { b \notin S } e ^ { \bar { z } _ { g , b } } \sum _ { j \in b } ( \bar { z } _ { g , b } - z _ { g , j } ) = 0 ,\tag{15}
$$

because $\bar { z } _ { g , b }$ is the arithmetic mean of the original logits in block b. Summing the resulting logpartition ratio over the G query heads yields

$$
\mathcal { L } ( \boldsymbol { S } ) = \sum _ { g = 1 } ^ { G } \log \frac { D _ { g } } { \widehat { D } _ { g } ( \boldsymbol { S } ) } = \sum _ { g = 1 } ^ { G } \log \frac { \displaystyle \sum _ { b \in \mathcal { B } } Z _ { g , b } } { \displaystyle \sum _ { b \in \mathcal { S } } Z _ { g , b } + \sum _ { b \not \in \mathcal { S } } \vert b \vert e ^ { \bar { z } _ { g , b } } } .\tag{16}
$$

To obtain the residual form, write the compensated normalization constant as the full weight minus the unreconstructed weight:

$$
\widehat { D } _ { g } ( \boldsymbol { S } ) = D _ { g } - \sum _ { b \not \in \boldsymbol { S } } \left( Z _ { g , b } - | b | e ^ { \bar { z } _ { g , b } } \right) .\tag{17}
$$

Dividing by $D _ { g }$ and using $p _ { g , b } = Z _ { g , b } / D _ { g }$ gives

$$
\frac { \widehat { D } _ { g } ( S ) } { D _ { g } } = 1 - \sum _ { b \notin S } p _ { g , b } \left( 1 - \frac { | b | e ^ { \bar { z } _ { g , b } } } { Z _ { g , b } } \right) .\tag{18}
$$

Substitution into the log-partition ratio establishes

$$
\mathcal { L } ( \mathcal { S } ) = - \sum _ { g = 1 } ^ { G } \log \left[ 1 - \sum _ { b \notin \mathcal { S } } p _ { g , b } \left( 1 - \frac { | b | e ^ { \bar { z } _ { g , b } } } { Z _ { g , b } } \right) \right] ,\tag{19}
$$

which is the second equality in Equation 4.

## A.2 EXPANSION OF THE BLOCKWISE COMPENSATION RESIDUAL

The residual form above shows that an omitted block contributes to the loss only through the weight that Mean compensation fails to recover. In the small-variance regime, this unreconstructed fraction has the expansion in Equation 5:

$$
p _ { g , b } \left( 1 - \frac { | b | e ^ { \bar { z } _ { g , b } } } { Z _ { g , b } } \right) = p _ { g , b } \left( \frac { 1 } { 2 } \sigma _ { g , b } ^ { 2 } + O ( \sigma _ { g , b } ^ { 3 } ) \right) .
$$

To establish this expansion, we hold the maximum block size fixed and let ma $\mathrm { x } _ { g , b } \sigma _ { g , b }  0 .$ , where $\sigma _ { g , b }$ is the standard deviation of the logits within block b for query head $g$ . We start from the block’s exact weight relative to its mean-based weight.

Averaging the second-order Taylor expansion of the exponential over a block gives

$$
\frac { Z _ { g , b } } { | b | e ^ { \bar { z } _ { g , b } } } = \frac { 1 } { | b | } \sum _ { j \in b } e ^ { z _ { g , j } - \bar { z } _ { g , b } } = 1 + \frac { 1 } { 2 } \sigma _ { g , b } ^ { 2 } + O ( \sigma _ { g , b } ^ { 3 } ) .\tag{20}
$$

The linear term vanishes because $\begin{array} { r } { \sum _ { j \in b } ( z _ { g , j } - \bar { z } _ { g , b } ) = 0 } \end{array}$ , and the quadratic term averages to $\sigma _ { g , b } ^ { 2 } / 2$ The fixed maximum block size makes the averaged third-order remainder $O ( \sigma _ { g , b } ^ { 3 } )$

The deviation of this ratio from one is $O ( \sigma _ { g , b } ^ { 2 } )$ , so taking its reciprocal gives

$$
\frac { | b | e ^ { \bar { z } _ { g , b } } } { Z _ { g , b } } = \frac { 1 } { 1 + \frac { 1 } { 2 } \sigma _ { g , b } ^ { 2 } + O ( \sigma _ { g , b } ^ { 3 } ) } = 1 - \frac { 1 } { 2 } \sigma _ { g , b } ^ { 2 } + O ( \sigma _ { g , b } ^ { 3 } ) .\tag{21}
$$

The reciprocal expansion introduces a quadratic remainder of order $O ( \sigma _ { q , b } ^ { 4 } )$ , which combines with the existing cubic remainder. Subtracting the reciprocal from one and multiplying by $p _ { g , b }$ yields

$$
p _ { g , b } \left( 1 - \frac { | b | e ^ { \bar { z } _ { g , b } } } { Z _ { g , b } } \right) = p _ { g , b } ( \frac { 1 } { 2 } \sigma _ { g , b } ^ { 2 } + O ( \sigma _ { g , b } ^ { 3 } ) ) .\tag{22}
$$

Taking the logarithm of Equation 20 also gives the log-partition expansion used for attention-mass estimation:

$$
\log Z _ { g , b } = \log | b | + \bar { z } _ { g , b } + \frac { 1 } { 2 } \sigma _ { g , b } ^ { 2 } + O ( \sigma _ { g , b } ^ { 3 } ) .\tag{23}
$$

## A.3 SECOND-ORDER EXPANSION OF THE SELECTION LOSS

The preceding subsection expands the residual of a single omitted block. We now combine these residuals within each query head and account for the outer logarithm in the exact selection loss. This gives the second-order approximation in Equation 6:

$$
\mathcal { L } ( \mathcal { S } ) = \frac { 1 } { 2 } \sum _ { g = 1 } ^ { G } \sum _ { b \notin \mathcal { S } } p _ { g , b } \left( \sigma _ { g , b } ^ { 2 } + O \big ( \sigma _ { g , b } ^ { 3 } \big ) \right) .
$$

To derive this expression, hold the maximum block size fixed and set $\varepsilon = \operatorname* { m a x } _ { g , b } \sigma _ { g , b }  0$ . Define the total normalized residual for head $g$ as

$$
R _ { g } ( S ) = \sum _ { b \not \in S } p _ { g , b } \left( 1 - \frac { | b | e ^ { \bar { z } _ { g , b } } } { Z _ { g , b } } \right) .\tag{24}
$$

Summing the blockwise expansion from Appendix A.2 gives

$$
R _ { g } ( S ) = \frac { 1 } { 2 } \sum _ { b \notin S } p _ { g , b } \sigma _ { g , b } ^ { 2 } + O \left( \sum _ { b \notin S } p _ { g , b } \sigma _ { g , b } ^ { 3 } \right) .\tag{25}
$$

Since $\begin{array} { r l r } { \sum _ { b \notin S } p _ { g , b } \sigma _ { g , b } ^ { 3 } } & { \le } & { \varepsilon \sum _ { b \notin S } p _ { g , b } \sigma _ { g , b } ^ { 2 } , } \end{array}$ we have $\begin{array} { r c l } { R _ { g } } & { = } & { O ( \sum _ { b \notin \mathcal { S } } p _ { g , b } \sigma _ { g , b } ^ { 2 } ) } \end{array}$ Moreover, $\begin{array} { r } { \sum _ { b \notin \boldsymbol { S } } p _ { g , b } \overset { \cdot } { \boldsymbol { \ b } } \leq 1 } \end{array}$ implies $R _ { g }  0$ . Expanding the outer logarithm therefore yields

$$
\begin{array} { l } { \displaystyle - \log ( 1 - R _ { g } ) = \frac { 1 } { 2 } \sum _ { b \notin \mathcal { S } } p _ { g , b } \sigma _ { g , b } ^ { 2 } } \\ { \displaystyle \qquad + O \left( \sum _ { b \notin \mathcal { S } } p _ { g , b } \sigma _ { g , b } ^ { 3 } + \left[ \sum _ { b \notin \mathcal { S } } p _ { g , b } \sigma _ { g , b } ^ { 2 } \right] ^ { 2 } \right) . } \end{array}\tag{26}
$$

The quadratic remainder can be absorbed into the cubic term. By the Cauchy–Schwarz inequality,

$$
\begin{array} { r l } { \displaystyle \left[ \sum _ { b \notin \mathcal { S } } p _ { g , b } \sigma _ { g , b } ^ { 2 } \right] ^ { 2 } \leq \left( \sum _ { b \notin \mathcal { S } } p _ { g , b } \right) \left( \sum _ { b \notin \mathcal { S } } p _ { g , b } \sigma _ { g , b } ^ { 4 } \right) } & { } \\ { \displaystyle } & { \leq \varepsilon \sum _ { b \notin \mathcal { S } } p _ { g , b } \sigma _ { g , b } ^ { 3 } . } \end{array}\tag{27}
$$

Substituting this bound and summing over the query heads gives

$$
\begin{array} { l } { \displaystyle \mathcal { L } ( \mathcal { S } ) = \frac { 1 } { 2 } \sum _ { g = 1 } ^ { G } \sum _ { b \notin \mathcal { S } } p _ { g , b } \sigma _ { g , b } ^ { 2 } + O \left( \sum _ { g = 1 } ^ { G } \sum _ { b \notin \mathcal { S } } p _ { g , b } \sigma _ { g , b } ^ { 3 } \right) } \\ { \displaystyle \quad = \frac { 1 } { 2 } \sum _ { g = 1 } ^ { G } \sum _ { b \notin \mathcal { S } } p _ { g , b } \left( \sigma _ { g , b } ^ { 2 } + O ( \sigma _ { g , b } ^ { 3 } ) \right) , } \end{array}\tag{28}
$$

which establishes Equation 6. The remainder bounds hold uniformly over the selected sets throughout the stated small-variance regime with a fixed maximum block size.

The leading term depends on $s$ only through the unselected blocks. Because its sum over all blocks is independent of S, minimizing the unselected contribution under a budget of K exact block reads is equivalent to

$$
\operatorname* { m a x } _ { \substack { s \subseteq B , | S | = K } } \sum _ { b \in S } \sum _ { g = 1 } ^ { G } p _ { g , b } \sigma _ { g , b } ^ { 2 } .
$$

Thus, the second-order objective selects the K blocks with the largest $\textstyle \sum _ { g = 1 } ^ { G } p _ { g , b } \sigma _ { g , b } ^ { 2 } .$ , as stated in Equation 7.

## A.4 GROUPED VARIANCE AND ATTENTION-MASS ESTIMATION

The oracle selection rule above requires each block’s exact attention mass and logit variance, which would require reading its keys. CompKV instead estimates the variance from grouped key statistics. The first quantity to derive is Equation 8:

$$
\widehat { \sigma } _ { g , b } ^ { 2 } = \frac { 1 } { d } \sum _ { t = 0 } ^ { r - 1 } \left( \sum _ { i \in \mathcal { G } _ { t } } q _ { g , i } ^ { 2 } \right) ( \bar { \mathbf { D } } _ { b } ) _ { t } , \qquad ( \bar { \mathbf { D } } _ { b } ) _ { t } = \frac { 1 } { \left| b \right| \left| \mathcal { G } _ { t } \right| } \sum _ { j \in b } \sum _ { i \in \mathcal { G } _ { t } } ( k _ { j , i } - \bar { k } _ { b , i } ) ^ { 2 } .
$$

To obtain this estimate, let $\Sigma _ { k , b } \in \mathbb { R } ^ { d \times d }$ denote the key covariance within block $b .$ Its definition and projection along the query give

$$
\pmb { \Sigma } _ { k , b } = \frac { 1 } { | b | } \sum _ { j \in b } ( \pmb { k } _ { j } - \bar { \pmb { k } } _ { b } ) ^ { \top } ( \pmb { k } _ { j } - \bar { \pmb { k } } _ { b } ) , \qquad \sigma _ { g , b } ^ { 2 } = \frac { 1 } { d } \pmb { q } _ { g } \pmb { \Sigma } _ { k , b } \pmb { q } _ { g } ^ { \top } .\tag{29}
$$

The projection follows from $z _ { g , j } - \bar { z } _ { g , b } = q _ { g } ( k _ { j } - \bar { k } _ { b } ) ^ { \top } / \sqrt { d } .$

For the settings where r divides $d / 2 ,$ , we divide the ordered RoPE frequency pairs into r equal, consecutive groups, keeping both coordinates of each pair together. Each group $\mathcal { G } _ { t }$ contains $d / r$ coordinates. $\mathrm { A t } r = d $ , each group contains one coordinate. The statistic stored for each group is the average of its diagonal covariance entries:

$$
( \bar { \mathbf { D } } _ { b } ) _ { t } = \frac { 1 } { | \mathcal { G } _ { t } | } \sum _ { i \in \mathcal { G } _ { t } } ( \Sigma _ { k , b } ) _ { i i } = \frac { 1 } { | b | | \mathcal { G } _ { t } | } \sum _ { j \in b } \sum _ { i \in \mathcal { G } _ { t } } ( k _ { j , i } - \bar { k } _ { b , i } ) ^ { 2 } .\tag{30}
$$

We approximate $\Sigma _ { k , b }$ by a diagonal matrix, assigning $( \bar { \mathbf { D } } _ { b } ) _ { 1 }$ <sub>t</sub> to every diagonal entry whose coordinate belongs to $\mathcal { G } _ { t }$ . Projecting this approximation along the query yields

$$
\widehat { \sigma } _ { g , b } ^ { 2 } = \frac { 1 } { d } \sum _ { t = 0 } ^ { r - 1 } \sum _ { i \in \mathcal { G } _ { t } } q _ { g , i } ^ { 2 } ( \bar { \mathbf { D } } _ { b } ) _ { t } = \frac { 1 } { d } \sum _ { t = 0 } ^ { r - 1 } \left( \sum _ { i \in \mathcal { G } _ { t } } q _ { g , i } ^ { 2 } \right) ( \bar { \mathbf { D } } _ { b } ) _ { t } .\tag{31}
$$

Together, these expressions recover both parts of Equation 8. Each group’s average key variance is weighted by the sum of squared query coordinates in that group, with attention scaling $\dot { 1 } / d$

The same variance estimate also determines the attention-mass estimate in Equation 9. Exponenti ating the log-partition expansion in Equation 23 gives

$$
Z _ { g , b } = \left| b \right| \exp \left( \bar { z } _ { g , b } + \frac { 1 } { 2 } \sigma _ { g , b } ^ { 2 } + O ( \sigma _ { g , b } ^ { 3 } ) \right) .\tag{32}
$$

Retaining the second-order terms and replacing $\sigma _ { g , b } ^ { 2 }$ with $\widehat { \sigma } _ { g , b } ^ { 2 }$ gives the estimated block weight $\vert b \vert \exp ( \bar { z } _ { g , b } + \widehat { \sigma } _ { q , b } ^ { 2 } / 2 )$ . Since the full attention mass is $\begin{array} { r } { p _ { g , b } = Z _ { g , b } / \sum _ { c \in \mathcal { B } } Z _ { g , c } . } \end{array}$ , normalizing these estimated weights over all available blocks yields

$$
\widehat { p } _ { g , b } = \frac { | b | \exp \Big ( \bar { z } _ { g , b } + \frac { 1 } { 2 } \widehat { \sigma } _ { g , b } ^ { 2 } \Big ) } { \displaystyle \sum _ { c \in \mathcal { B } } | c | \exp \Big ( \bar { z } _ { g , c } + \frac { 1 } { 2 } \widehat { \sigma } _ { g , c } ^ { 2 } \Big ) } .\tag{33}
$$

This recovers Equation 9, with normalization performed separately for each query head.

## A.5 COMPARISON WITH UNCOMPENSATED SPARSE ATTENTION

For comparison, Drop removes unselected blocks and renormalizes over the tokens in a nonempty selected set S, giving $P _ { g } ^ { \mathrm { D r o p } } ( S )$ . Unselected positions in the original token space receive zero probability. The ratio between the approximate and full probabilities is constant on selected positions:

$$
D _ { \mathrm { K L } } \big ( P _ { g } ^ { \mathrm { D r o p } } ( S ) \| P _ { g } \big ) = \log \frac { \displaystyle \sum _ { c \in \mathcal { B } } Z _ { g , c } } { \displaystyle \sum _ { b \in S } Z _ { g , b } } = - \log \left( \sum _ { b \in \mathcal { S } } p _ { g , b } \right) .\tag{34}
$$

This identity agrees with the KL analysis of Top-k sparse attention (Tzachristas et al., 2026). For one head and a fixed block budget, retaining the blocks with the largest attention mass exactly minimizes this KL divergence. With a shared selected set across heads, the objective sums the headwise logarithmic terms. When the omitted attention mass is small for each head, its first-order additive score is $\sum _ { g } p _ { g , b }$

For the same nonempty selected set, adding positive mean-compensation weights increases the normalization constant. The inequality $| b | e ^ { \bar { z } _ { g , b } } \leq Z _ { g , b }$ also ensures that this constant does not exceed that of full attention. Consequently,

$$
0 \leq D _ { \mathrm { K L } } \Big ( \widehat { P } _ { g } ( S ) \| P _ { g } \Big ) \leq D _ { \mathrm { K L } } \big ( P _ { g } ^ { \mathrm { D r o p } } ( S ) \| P _ { g } \big ) .\tag{35}
$$

The second inequality is strict whenever an unselected block exists. The first is also strict if at least one unselected block has nonconstant logits. This comparison holds for a fixed selected set; the two operators can induce different optimal block sets under the same exact-read budget.

## B ATTENTION FIDELITY OF RESIDUAL-AWARE SELECTION

## B.1 EXACT RESIDUAL VERSUS ATTENTION MASS

We compare exact mass-based and residual-based selection on Llama-3.1-8B-Instruct across all 13 RULER tasks, randomly sampling 64 prompts per task with seed 42 (832 total). We collect full-attention states under greedy decoding with batch size one at some layers. Both selectors use identical Q/K/V states, Mean compensation, and a 512-token budget with 16-token blocks; the first block and two most recent blocks are mandatory and included in the budget.

The selectors rank blocks by $\begin{array} { r } { \sum _ { g } p _ { g , b } \mathrm { o r } \sum _ { g } \rho _ { g , b } } \end{array}$ , sharing each selected set across heads, where

$$
\rho _ { g , b } = p _ { g , b } \left( 1 - \frac { | b | e ^ { \bar { z } _ { g , b } } } { Z _ { g , b } } \right)\tag{36}
$$

is the exact compensation residual in Equation 4. For each head h, we measure

$$
E _ { \mathrm { K L } , h } = D _ { \mathrm { K L } } ( \widehat { P } _ { h } \| P _ { h } ) , \qquad E _ { \mathrm { o u t } , h } = \frac { \| \widehat { P } _ { h } V _ { h } - P _ { h } V _ { h } \| _ { 2 } } { \| P _ { h } V _ { h } \| _ { 2 } } .\tag{37}
$$

KL compares post-softmax probabilities; relative $L _ { 2 }$ compares outputs after multiplication by $V _ { h }$ We plot $\Delta E _ { h } = E _ { h } ^ { ( \mathrm { m a s s } ) } - E _ { h } ^ { ( \mathrm { r e s } ) }$ , so positive values favor residual-based selection. Each point averages positions within prompts, prompts within tasks, and then all 13 tasks equally.

Averaged over the eight layers and 32 query heads, residual-based selection reduces KL by 0.660% and relative $L _ { 2 }$ by 0.582%. Both metrics improve in every layer-level mean, with improvements in most layer–head pairs. Thus, prioritizing the contribution left unreconstructed by Mean compensation lowers average attention and output error relative to mass-based selection.

Although the local gains are modest, attention-output errors propagate through subsequent layers and decoding steps and may be amplified, while reducing them may help limit accumulated deviations.

## B.2 FIDELITY OF PRACTICAL COMPKV SCORES

On the same states and budget, we compare CompKV $( r \in \{ 4 , 1 2 8 \} )$ with estimated-mass selection, $\begin{array} { r } { \sum _ { g } \widehat { p } _ { g , b } . } \end{array}$ , using the $r = 4$ mass estimate in Equation 9. This baseline retains the second-order mass correction and removes the external variance factor in Equation 10.

For each KV group, let $\begin{array} { r } { T _ { b } = \sum _ { q } \rho _ { g , b } } \end{array}$ , with $\rho _ { g , b }$ defined in Equation 36. Spearman correlation compares scores with $T _ { b }$ over non-mandatory blocks. Let $S _ { m }$ and $ { \boldsymbol { S } } _ { \rho }$ contain the 29 non-mandatory blocks selected by method m and exact residual ranking. Residual capture is

$$
{ \mathrm { C a p t u r e } } ( m ) = { \frac { \sum _ { b \in S _ { m } } T _ { b } } { \sum _ { b \in S _ { \rho } } T _ { b } } } .\tag{38}
$$

![](images/7c81000627c0cf0e58de41292cfaecdaa2ba42f7c82f45e01b84bff55049282c.jpg)  
Figure 6: Per-head error reduction from residual-based selection. Mass-minus-residual differences in attention KL (left) and relative output $L _ { 2 }$ (right), using the same Mean compensator.

We use the averaging in Appendix B.1. Layer curves additionally average KV heads, and overall results also average layers.

Figure 7 reports both metrics. Overall Spearman correlations are 0.8511, 0.8604, and 0.8782 for estimated mass, CompKV $( r = 4 )$ , and CompKV $( r = 1 2 8 )$ , respectively, with capture rates of 81.17%, 81.53%, and 85.61%. CompKV (r = 4) improves residual capture over estimated-mass selection in seven of eight layer means. Spearman correlations were saved as online scalars.

These results show that highly compressed block summaries preserve much of the residual-selection signal. With only four grouped-variance statistics per block, CompKV achieves strong ranking agreement and captures over 81% of the residual removable by the exact reference. Its modest improvements over estimated-mass selection use the same summary statistics, showing the benefit of compensation-aware scoring within a fixed metadata budget. Although a gap to exact selection remains, this provides a practical balance between summary compactness and selection fidelity.

![](images/cd14832444c7a237ce9139d836caa9d756a249e9f1d720254ed2b1dab6e32c16.jpg)

(b) Spearman, CompKV (r = 4)  
![](images/a22e675e45569d60391e4ed76da39cfd6967efea9f0a91d94a40df62580cbe78.jpg)  
Figure 7: Fidelity of practical CompKV scores. (a) residual capture at budget 512, normalized by exact residual selection. (b) CompKV (r = 4) Spearman correlations by layer and KV head. Mandatory blocks are excluded from both metrics.

## C EVALUATION CONFIGURATION

## C.1 MODEL CONFIGURATIONS

• Llama-3.1-8B-Instruct has 32 Transformer layers, 32 query heads, 8 key–value (KV) heads, a head dimension of 128, and a native context length of 131,072 tokens.

• Qwen3-8B has 36 Transformer layers, 32 query heads, 8 KV heads, a head dimension of 128, and a native context length of 32,768 tokens.

• Qwen3-32B has 64 Transformer layers, 64 query heads, 8 KV heads, a head dimension of 128, and a native context length of 32,768 tokens.

## C.2 EVALUATION BENCHMARKS

RULER. RULER (Hsieh et al., 2024) is a synthetic long-context benchmark organized around retrieval, multi-hop tracing, aggregation, and question answering. We evaluate all 13 tasks in its 32K configuration, using 500 samples per task and 6,500 samples in total. Retrieval is measured by eight Needle-in-a-Haystack tasks: niah single 1/2/3, niah multikey 1/2/3, niah multivalue, and niah multiquery. vt evaluates variable tracking, cwe and fwe evaluate common- and frequent-word aggregation, and qa 1 and qa 2 evaluate question answering over the SQuAD and HotpotQA documents provided for these two tasks, respectively.

LongBench-Pro. LongBench-Pro (Chen et al., 2026) is constructed from natural long documents and contains 1,500 bilingual samples across 11 primary tasks and 25 secondary tasks. Its primary tasks are Retrieval & Ranking, Sequencing & Structure Reconstruction, Evidence-Grounded QA, Summarization & Synthesis, Attribution & Citation Alignment, Aggregation & Clustering, Consistency & Compliance Checking, Structured & Numeric Reasoning, Version & Code Diff Analysis, Rule Induction & In-Context Learning, and Dialogue Memory & Long-Horizon Tracking. We use the 625 English samples in the 8k, 16k, 32k, 64k, and 128k buckets, with 125 samples in each bucket. The selected samples cover both Full (global integration) and Partial (localized retrieval) context requirements and retain the Easy, Moderate, Hard, and Extreme difficulty labels.

## C.3 EVALUATION DETAILS

InfLLM configuration. We adapt InfLLM to 16-token blocks and grouped-query attention. Each query head accumulates causal attention probabilities and represents each block by the FP32 mean of its four highest-weight post-RoPE keys. Representatives are frozen when blocks leave the recent window, before selection. Query–representative scores are summed within each KV group to obtain a shared selected set. We retain the model’s positional encoding, including YaRN for Qwen3 on LongBench-Pro, to compare KV selection and compensation under the positional semantics of the same Full attention reference. InfLLM’s distant-position remapping changes both retrieval scores and attention logits. Retaining the configured positions keeps this factor consistent across methods and between dense prefill and decoding, allowing us to evaluate its representative-based selection within our common decode setting. Original Q/K/V states are BF16; representatives, accumulated weights, scores, and attention accumulation use FP32.

Prompt formatting. Llama-3.1-8B-Instruct uses the official raw RULER prompt, in which the input is followed directly by the answer prefix. For Qwen3-8B and Qwen3-32B, the RULER input is placed in a user message and the answer prefix in the final assistant message, with generation continuing from that prefix by setting continue final message=true. For LongBench-Pro, we concatenate each context with the official question nonthinking field using four newline characters and then render the model-native chat template. Llama uses its instruct template, while both Qwen3 models use their assistant-prefill template with the generation prompt enabled.

No-thinking policy. We evaluate both Qwen3 checkpoints in their supported non-thinking mode on RULER and LongBench-Pro, setting enable thinking=False in the chat template and add special tokens=False during tokenization. This establishes a direct-answer setting for comparing KV-selection methods under fixed output budgets. In preliminary runs, some methods still generated thinking markers despite this configuration. We therefore additionally suppress the <think> and </think> token IDs, 151667 and 151668, at every decoding step before token selection. This policy applies identically to Full attention, all sparse baselines, and CompKV.

Context handling. RULER prompts are left-truncated only after reserving the task-specific completion length. On LongBench-Pro, both Qwen3 models use YaRN with a factor of 4 and an original maximum position of 32,768, giving an effective context length of 131,072 tokens, while Llama use its native context window. If a rendered prompt plus 1,024 completion tokens exceeds the model context, we retain equal-length prefix and suffix segments through middle truncation.

Sampling and stopping. RULER uses deterministic greedy decoding without sampling filters. Its official generation limits are 128 tokens for the retrieval tasks, 30 for vt, 120 for cwe, 50 for fwe, and 32 for each question-answering task. LongBench-Pro uses stochastic decoding: both Qwen3 models set (temperature, top p, top k, min p) = (0.7, 0.8, 20, 0), while Llama uses (0.6, 0.9, 0, None). All three models use max new tokens=1024, one beam, repetition penalty 1.0, and EOS-only stopping. RULER uses seed 42 for its single deterministic decode. Each LongBench-Pro case is decoded for three rounds with seeds 42, 43, and 44.

LongBench-Pro budgets. The two token budgets are (256, 512) for 8k and 16k, (512, 1024) for 32k, (1024, 2048) for 64k, and (2048, 4096) for 128k. For all sparse methods, these values cap the exact KV tokens read per decoding step, including the sink and recent blocks.

Scoring metrics. The official RULER scorer applies all-reference substring matching to the retrieval, tracing, and aggregation tasks and any-reference substring matching to the two questionanswering tasks. The LongBench-Pro scorer selects the metric associated with each secondary task, including NDCG, Pairwise Accuracy, Accuracy, Summary, F1, and SubEM. Summary uses Qwen3- Embedding-8B as specified by the benchmark.

Score aggregation. For RULER, we first average over the 500 samples within each task and then report the unweighted mean of the 13 task scores. For LongBench-Pro, we average the scores from the three rounds for each sample and budget tier, average the two tiers with equal weight, and then take the arithmetic mean over all 625 samples to obtain the AVG score reported in Table 2. Difficulty and length-bucket scores apply the same aggregation to their corresponding sample subsets.

## D ACCURACY ACROSS KV BUDGETS

Figure 8 compares per-step exact-read budgets of 128–1,024 tokens. On MK3, CompKV reaches 73.0% for Llama and 75.0% for Qwen with 512 tokens, exceeding the strongest baselines at 1,024 tokens (70.4% and 62.4%, respectively). At 128 tokens, Qwen’s MV and MQ scores exceed the strongest baseline by 37.2 and 31.0 percentage points, respectively. Several baselines approach Full on FWE, MV, and MQ as budgets increase, narrowing the gaps between methods.

![](images/507ddfc3ab8661188cdb5c313d2c0ab949516ca98a4fd8885a4c2f51cc9396d3.jpg)  
Figure 8: Accuracy across KV budgets on selected RULER tasks: (a–c) Llama-3.1-8B-Instruct and (d–h) Qwen3-8B. CompKV uses r = 4. Dashed gray lines denote Full attention.

## E CPU-OFFLOAD IMPLEMENTATION AND DETAILED LATENCY RESULTS

## E.1 IMPLEMENTATION DETAILS

The CPU-offload adapter stores BF16 KV states and FP32 summaries in pinned CPU memory, while retaining a 16-token active block and reusable workspaces on the GPU. Each append updates the active block’s mean keys, mean values, and grouped population variances on the GPU, then copies the current KV states and updated summary to the CPU. Completed blocks’ summaries remain unchanged. GPU scoring reuses projected variances for mass estimation and residual scoring, aggregating scores across query heads sharing a KV head. Stable Top-K selection includes one sink block and two recent blocks within the budget and returns block IDs in position order.

The CPU gathers selected KV blocks into a contiguous pinned buffer for one H2D transfer and exact attention with FlashInfer, passing the valid token count explicitly for a partial final block. An auxiliary stream transfers mean values and computes Mean compensation once the selected-block mask is ready, independently of the selected-KV retrieval and exact-attention path. CUDA events synchronize the two branches before merging their contributions under the shared normalization in Equation 12. Intermediate computations use FP32, and the final output is BF16.

## E.2 DETAILED LATENCY RESULTS

Each case runs the pretrained Llama-3.1-8B-Instruct on one NVIDIA H100 80GB HBM3 GPU with four NUMA-local CPU threads and batch size one. Q/K/V and outputs use BF16; summaries and scores use FP32. Following the input construction and collection schedule in Quest’s official profiling implementation (Tang et al., 2024), we use standard-normal random embeddings matched across methods by a saved RNG state. We prefill $L - 2 5 6$ tokens and decode 256 steps, defining L as the window endpoint. Steps 193–208 and layers 3–32 yield 480 calls per case. Sparse methods use 16-token blocks with one sink and two recent blocks included in the budget. Full transfers all valid CPU-resident KV at each step. Results are shown in Table 4.

A synchronized host timer starts after Q/K/V are ready and stops after the attention output and al state updates complete. It includes KV/summary updates, transfers, selection, CPU gathering, exact attention, compensation, merging, and InfLLM probability maintenance. Input generation, prefill, compilation, projections, RoPE, and MLP are excluded. Primary timing runs without a profiler. We report the mean and sample SD of all 480 calls. Speedups are ratios of reported means.

Table 4: CPU-offload attention-step latency, reported as mean ± sample SD (ms) over 480 calls per case. Lengths denote decode-window endpoints. Full is shared across the three budget rows at each length. Bold marks the lowest mean in each row.
<table><tr><td>Endpoint Budget</td><td></td><td>Full</td><td>Quest</td><td>InfLLM</td><td> $\mathrm { C o m p K V } \left( \mathrm { r } { = } 4 \right)$ </td><td>CompKV (r=128)</td></tr><tr><td rowspan="3">32K</td><td>512</td><td></td><td> $0 . 5 9 5 \pm 0 . 0 1 3$ </td><td> $0 . 9 1 9 \pm 0 . 0 7 9$ </td><td> $\mathbf { 0 . 5 2 7 \pm 0 . 0 0 7 }$ </td><td> $0 . 6 7 5 \pm 0 . 0 0 6$ </td></tr><tr><td>1024</td><td> $2 . 5 3 6 \pm 0 . 0 0 8$ </td><td> $0 . 7 5 0 \pm 0 . 0 1 3$ </td><td> $1 . 0 8 4 \pm 0 . 5 9 6$ </td><td> $\mathbf { 0 . 6 6 5 \pm 0 . 0 1 7 }$ </td><td> $0 . 8 0 6 \pm 0 . 0 1 6$ </td></tr><tr><td>2048</td><td></td><td> $0 . 9 7 2 \pm 0 . 0 2 5$ </td><td> $1 . 2 9 3 \pm 0 . 0 5 1$ </td><td> $\mathbf { 0 . 8 9 1 \pm 0 . 0 2 4 }$ </td><td> $1 . 0 3 8 \pm 0 . 0 1 5$ </td></tr><tr><td rowspan="3">64K</td><td>512</td><td></td><td> $0 . 9 4 4 \pm 0 . 0 1 8$ </td><td> $1 . 5 5 6 \pm 0 . 0 4 1$ </td><td> $\mathbf { 0 . 8 4 0 \pm 0 . 0 0 6 }$ </td><td> $1 . 1 3 4 \pm 0 . 0 1 0$ </td></tr><tr><td>1024</td><td> $4 . 9 9 5 \pm 0 . 0 0 7$ </td><td> $1 . 0 9 9 \pm 0 . 0 1 6$ </td><td> $1 . 7 0 6 \pm 0 . 0 3 4$ </td><td> $\mathbf { 0 . 9 1 6 \pm 0 . 0 0 7 }$ </td><td> $1 . 2 0 6 \pm 0 . 0 1 9$ </td></tr><tr><td>2048</td><td></td><td> $1 . 3 3 1 \pm 0 . 0 2 0$ </td><td> $1 . 9 3 5 \pm 0 . 0 5 6$ </td><td> $\mathbf { 1 . 1 1 2 \pm 0 . 0 2 1 }$ </td><td> $1 . 3 9 1 \pm 0 . 0 1 3$ </td></tr><tr><td rowspan="3">128K</td><td>512</td><td></td><td> $1 . 6 2 5 \pm 0 . 0 2 4$ </td><td> $2 . 8 7 7 \pm 0 . 0 4 3$ </td><td> $\mathbf { 1 . 4 5 6 \pm 0 . 0 0 7 }$ </td><td> $2 . 0 4 8 \pm 0 . 0 3 3$ </td></tr><tr><td>1024</td><td> $9 . 9 6 6 \pm 0 . 0 1 5$ </td><td> $1 . 7 9 5 \pm 0 . 0 2 0$ </td><td> $2 . 9 9 6 \pm 0 . 0 2 0$ </td><td> $\mathbf { 1 . 5 3 6 \pm 0 . 0 1 0 }$ </td><td> $2 . 1 1 9 \pm 0 . 0 1 1$ </td></tr><tr><td>2048</td><td></td><td> $2 . 0 3 5 \pm 0 . 0 2 9$ </td><td> $3 . 2 6 9 \pm 0 . 0 5 6$ </td><td> $\mathbf { 1 . 6 1 8 \pm 0 . 0 1 8 }$ </td><td> $2 . 2 0 0 \pm 0 . 0 2 1$ </td></tr></table>