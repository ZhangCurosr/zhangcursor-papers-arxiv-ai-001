# DRAMA: Dissecting Attention Redundancy for Accelerating Multimodal Diffusion Large Language Models

Yi-Hsin Hung¹, Fangfu Liu2, Shao-Yuan Lo¹, Chu-Song Chen¹

1National Taiwan University, 2Tsinghua University

{r14922a09, sylo, chusong}@csie.ntu.edu.tw, liuff19@mails.tsinghua.edu.cn

## Abstract

Multimodal diffusion large language models (dLLMs) generate text through iterative masked denoising with bidirectional attention, offering a compelling alternative to autoregressive vision-language modeling. However, each denoising step requires a full forward pass over long multimodal sequences in which visual tokens dominate the sequence length, and a fixed step budget further compounds this quadratic per-step cost. Existing inference acceleration approaches for multimodal dLLMs characterize attention redundancy implicitly, motivating designs from isolated attention visualizations rather than from measurement. We close this gap with DRAMA, a training-free acceleration framework grounded in a systematic dissection of attention redundancy in multimodal dLLMs. By probing attention tensors along spatial and temporal axes, we reveal two persistent redundancy structures: (i) attention mass consistently concentrates on a sparse subset of visual tokens, and (ii) attention distributions rapidly stabilize across denoising steps after an initial formation phase. These findings motivate two complementary mechanisms: Spatial-Redundancy-Aware Visual Token Pruning (SVTP) and Sufficiency-Guided Adaptive Stopping (SGAS). Together, DRAMA achieves an average 18.67× inference speedup across six vision-language benchmarks without training, architectural modification, or attention caching, while maintaining competitive accuracy. Our results establish a strong, measurement-grounded acceleration baseline for multimodal dLLM inference. The code is available at https://github.com/CindyHung20/DRAMA.

## 1 Introduction

Diffusion large language models (dLLMs) [38, 3] generate text by iteratively denoising a fully masked sequence using bidirectional self-attention, enabling parallel token generation and global context reasoning as an alternative to left-to-right autoregressive decoding. Models such as LLaDA [38] and Gemini Diffusion [11] have demonstrated competitive performance with autoregressive counterparts Recent extensions, including LLaDA-V [52], LaViDa [28], and MMaDA [51], further bring this paradigm to vision-language tasks by conditioning the denoising process on visual inputs.

While effective, the iterative nature of dLLMs incurs substantial computational cost. This issue is further amplified in multimodal settings. Each denoising step requires a full forward pass through all L transformer layers over the concatenated sequence [v; c; xt] of visual tokens v, text condition c, and noisy output xt, where the visual tokens alone contribute 2-3k tokens per image and dominates the total sequence length N. With a fixed budget of T steps, total cost scales as $\mathcal { O } ( \breve { T } \cdot L \cdot N ^ { 2 } )$ , where the visual token count $N _ { v }$ is the dominant term in N. Crucially, the visual prefix is never masked and remains static across all steps, yet its quadratic attention cost is repeatedly incurred at every iteration.

![](images/380fd3293127337e6626e2c35e5e72e74f65363051e4b4b28c03ca8d32279dfe.jpg)  
Figure 1: Accuracy-latency Pareto frontier. DRAMA variants (diamonds) dominate the Pareto front, achieving higher accuracy at lower latency than all fixed-budget baselines (triangles).

A concurrent line of work has begun extending acceleration techniques to multimodal dLLMs [50. 7, 48]. Their empirical speedups, together with attention visualizations, suggest the presence of exploitable redundancy in this regime. However, these studies characterize such attention redundancy either implicitly through downstream performance gains or isolated examples. Consequently, their designs are rarely accompanied by a systematic measurement of the attention dynamics they exploit

In this work, we instead ask how redundancy itself can be systematically exploited to derive an analysis-driven, training-free acceleration framework whose budget is set by measurement rather than by tuning. To answer this, we systematically dissect attention redundancy in multimodal dLLMs along two orthogonal axes: a spatial axis examining how attention mass is distributed over visual tokens, and a temporal axis examining how attention evolves across consecutive denoising steps. Quantified across six vision-language benchmarks, our dissection consistently exposes two analysis-grounded redundancy structures, one per axis.

These findings motivate DRAMA (Dissecting attention Redundancy for Accelerating MultimodAl diffusion large language models), a training-free acceleration framework whose two complementary mechanisms each translate one identified redundancy structure into a dedicated strategy. Exploiting the spatial redundancy, Spatial-Redundancy-Aware Visual Token Pruning (SVTP) was designed to retain only dominant visual patches before denoising begins, with no per-step overhead. Sufficiency-Guided Adaptive Stopping (SGAS) exploits the inter-step convergence by terminating denoising once a sufficient decoded sequence has stabilized. By integrating these strategies, DRAMA achieves an average 18.67× inference speedup across six vision-language benchmarks [22, 33, 13, 54, 8, 26] on LLaDA-V without any training, architectural modifications, or attention caching, as shown in Tab. 1 and Fig. 1.

Our contributions are summarized as follows:

• We close the gap between observing and exploiting attention redundancy in multimodal dLLMs through the first systematic, two-axis dissection of redundancy structures, providing a quantitative basis for deciding what can be removed and how much.

• We propose DRAMA, a training-free framework that accelerates multimodal dLLMs via Spatial-Redundancy-Aware Visual Token Pruning and Sufficiency-Guided Adaptive Stopping, grounded in the discovered redundancy structures.

• We demonstrate that DRAMA achieves an average 18.67× inference speedup across six benchmarks with competitive performance, providing a strong acceleration baseline for multimodal dLLM inference.

## 2 Related Work

Diffusion Large Language Models. Diffusion large language models (dLLMs) have emerged as an alternative to autoregressive LLMs, offering parallel decoding capabilities and bidirectional attention mechanisms that enable more effective bidirectional reasoning [38, 42]. Following the pioneering work of LLaDA [38], subsequent efforts including Mercury Coder [24], Gemini Diffusion [11] and Seed Diffusion [42] have demonstrated the commercial viability of this paradigm. Recent works have extended dLLMs to multimodal settings, including MMaDA [51], LLaDA-V [52], and LaViDa [28]. Despite these advances, dLLMs face several fundamental challenges compared to autoregressive models: a persistent perplexity gap [2, 42], inability to naturally generate variablelength sequences [2], and high inference latency due to iterative denoising [42]. While dLLMs theoretically offer higher parallelism than autoregressive models, their iterative nature often results in slower practical inference. Our work addresses this gap by proposing interpretable attention analysis-based acceleration strategies specifically designed for multimodal dLLMs.

Inference Acceleration for Diffusion Language Models. Existing acceleration methods for dLLMs fall into two categories. Since bidirectional attention precludes the standard AR KV cache, cachingbased approaches [27, 17, 36, 21, 35, 34, 37, 47, 41, 9] reduce redundant computation by reusing intermediate states across denoising steps: dKV-Cache [35] and dLLM-Cache [34] adaptively cache key-value or feature states based on token-level dynamics, Elastic-Cache [37] triggers layer-aware cache refresh via attention-drift detection, and Fast-dLLM [46] combines block-wise approximate KV caching with confidence-aware parallel decoding. Sampling-optimization approaches alter the generation procedure itself via parallel [6, 18, 45, 12, 23, 5] or speculative [49, 1, 15, 16] decoding, distillation [10, 14], and hybrid modeling methods [46, 44, 19]. A concurrent line of work [50, 7, 48] has begun extending acceleration to the multimodal setting. While their empirical gains signal the presence of exploitable redundancy, this redundancy is characterized only implicitly, inferred from downstream speedups or illustrated on isolated visualizations, leaving open how much of it can actually be removed, and where.

## 3 Method

## 3.1 Dissecting Spatial-Axis Attention Redundancy

Multimodal dLLMs based on LLaVA-NeXT [32] partition each image into $N _ { p }$ patches, yielding 2–3k visual tokens that dominate the sequence length and, consequently, the attention cost at every denoising step. Since this visual prefix is static across all T steps, any spatial redundancy in attention is replicated T-fold. We dissect the attention tensors along the spatial dimension at three granularities: individual tokens, image patches, and their cross-step evolution. Each of them reveals a consistent, exploitable structure that generalizes across all evaluated benchmarks. We use MMMU [54] for illustration throughout this section, and the full per-benchmark results are in the Appendix Section E.

Token-Level Attention Mass Concentration. We first examine the finest granularity: the distribution of attention mass over individual visual key tokens, stratified by the modality of query-key interaction. For each query token, we sort its attention weights over the modality-specific key set and compute cumulative coverage at varying retention fractions (Fig. 2(c)). The two cross-modal pairs involving visual keys, namely output→image (t2v) and image→image (v2v), exhibit strikingly concentrated profiles: the top 1% of visual key tokens already capture \~47% of a query's attention mass, and the top 10% suffice for ～78% coverage. By contrast, text-key interactions (t2t and v2t) follow a markedly flatter curve, with the top 1% accounting for only \~26% and \~33%, respectively. This asymmetry confirms that the extreme sparsity is specific to the visual modality: the model concentrates its attention budget on a small set of visually salient key positions while treating the remaining visual tokens as near-irrelevant context.

Patch-Level Non-Uniformity. If salient tokens were spread uniformly across patches, each patch would receive ～1 $/ N _ { p }$ of the output→image attention, leaving no efficient patch-level pruning strategy. Instead, the distribution is highly skewed within individual samples: the most-attended patch captures \~44% of aggregate attention, which is 1.7× the uniform baseline, and the top two patches suffice for ～68% coverage, while the remaining patches contribute marginally (Fig. 2(a)). Patches are ranked per-sample before averaging, so these fractions reflect genuine per-sample concentration rather than a cross-sample aggregation artefact. The dominant patch's attention share is consistent across the dataset (see Appendix Section E.1), confirming a stable structural property of the model. This cross-patch skewness implies the effective visual receptive field is far smaller than the full token set, motivating our spatial-redundancy-aware visual token pruning (Section 3.3.1).

![](images/2f37e6649e542a2cef3082633377f8326b28eba2f29290b27470479360b151f7.jpg)  
Figure 2: Spatial redundancy analysis. (a) Per-sample attention share of each image patch ranked by dominance. (b) Per-sample attention share distribution of the most-attended patch across the dataset. (c) Cumulative attention coverage by top-k% visual key tokens, stratified by query-key modality pair. (d) Minimum visual tokens needed for 90% attention coverage (κ(0.9)) across denoising steps

Step-Invariant Spatial Redundancy. A critical question for acceleration design is whether the spatial redundancy structure changes as denoising progresses. We track the per-query metric $\kappa _ { i } ( 0 . 9 )$ the minimum number of visual key tokens needed to capture 90% of an output query's attention mass, averaged across all layers and queries at each denoising step (Fig. 2(d)). Across all benchmarks, κ(0.9) remains remarkably stable: it accounts for only \~27% of the total visual token budget at every step, with a relative variation below 5% from the first step to the last. The near-flat trajectory demonstrates that the degree of spatial concentration is step-invariant. Moreover, the identity of the salient positions is also preserved: the inter-step attention divergence measured in Section 3.2 drops to near zero during the plateau phase, and Pinsker's inequality bounds each token's weight change accordingly (formal derivation in Section 3.3.2), confirming that a patch selection made before denoising remains valid across all T steps.

## 3.2 Dissecting Temporal-Axis Attention Redundancy

The spatial analysis dissects redundancy across tokens, identifying which visual positions can be pruned. Here, we dissect redundancy across denoising steps, examining whether the full attention distribution changes enough between consecutive steps to justify repeating the $\mathcal { O } ( L \cdot N ^ { 2 } )$ forward pass. When it does not, successive steps recompute near-identical outputs at full cost, constituting temporal redundancy. We quantify this via the divergence of per-query attention distributions at consecutive steps across the full trajectory. Inter-Step Distributional Divergence. Following [39], we adopt the Jensen-Shannon divergence (JSD), a symmetric, [0, ln 2]-bounded variant of KL divergence, as our inter-step attention redundancy metric to measure how attention distributions change between consecutive steps. For each layer l, head h, and query position i, the attention weights at denoising step t form a probability distribution over all N key positions:

$$
\mathbf { a } _ { l , h , i } ^ { ( t ) } = \mathrm { S o f t m a x } \left( \frac { \mathbf { q } _ { l , h , i } ^ { ( t ) } \cdot \mathbf { K } _ { l , h } ^ { ( t ) \top } } { \sqrt { d _ { k } } } \right) \in \Delta ^ { N - 1 } ,\tag{1}
$$

where $\Delta ^ { N - 1 }$ denotes the probability simplex. The JSD between consecutive-step attention distributions at the same query position is

$$
\begin{array} { r } { \mathrm { J S D } \Big ( \mathbf { a } ^ { ( t ) } \left\| \mathbf { a } ^ { ( t - 1 ) } \right) = \frac { 1 } { 2 } \mathrm { K L } \Big ( \mathbf { a } ^ { ( t ) } \left\| \mathbf { m } \right) + \frac { 1 } { 2 } \mathrm { K L } \Big ( \mathbf { a } ^ { ( t - 1 ) } \left\| \mathbf { m } \right) , } \end{array}\tag{2}
$$

where $\begin{array} { r } { \mathbf { m } = \frac { 1 } { 2 } ( \mathbf { a } ^ { ( t ) } + \mathbf { a } ^ { ( t - 1 ) } ) } \end{array}$ is the midpoint distribution. Unlike [39], which compares attention across spatial positions within a single step, we compare the same (layer, head, query) triplet across two consecutive steps. We aggregate into a mean per-layer divergence and its normalized redundancy

![](images/6bb6f0dd8965b35fb17260ac465c53d90113250a8113646010218ad4fc20cd8e.jpg)

![](images/d494e90e48a581d61a5cd0ebeab504a4324bd9c5ff92a1c316312bfa31010d38.jpg)  
Denoising Steps

![](images/4edb9556b788f740e7db21cfccdb7d536514c8fa929bb91667bd3a33f299dc72.jpg)  
Figure 3: Temporal redundancy analysis across the denoising trajectory (averaged over six benchmarks). (a) Per-layer inter-step JSD $( { \bar { J } } _ { l } )$ between consecutive denoising steps; peaks highlight steps of greatest attention change. (b) Redundancy heatmap $R _ { l } ( t )$ across layers and steps; bright regions indicate high redundancy $( R _ { l }  1 )$ . (c) Distribution of actual stopping steps of our proposed DRAMA, whose concentration aligns with the plateau onset in (a).

complement:

$$
\bar { J } _ { l } ( t ) = \frac { 1 } { H \cdot N } \sum _ { h = 1 } ^ { H } \sum _ { i = 1 } ^ { N } \mathrm { J S D } \left( \mathbf { a } _ { l , h , i } ^ { ( t ) } \parallel \mathbf { a } _ { l , h , i } ^ { ( t - 1 ) } \right) , \qquad R _ { l } ( t ) = 1 - \frac { \bar { J } _ { l } ( t ) } { \ln 2 } .\tag{3}
$$

$R _ { l } ( t )  1$ indicates near-identical consecutive distributions.

Phase-Structured Denoising Dynamics. Fig. 3(a) plots ${ \bar { J } } _ { l } ( t )$ across the full denoising trajectory for each transformer layer, averaged over all evaluation benchmarks. Rather than declining monotonically, the inter-step divergence reveals a characteristic three-phase temporal structure:

(i) Structure formation $( \bar { J } _ { l }$ peak). The JSD peaks sharply as the sequence transitions from fully masked to carrying initial content. Although only ${ \sim } N _ { g } / \bar { T }$ tokens are revealed per step, they represent the first semantic signal in an otherwise content-agnostic attention landscape. Naturally resulting in a maximized relative information gain and inducing the largest attention redistribution in the entire trajectory.

(ii) Stable plateau $( \bar { J } _ { l } \to 0 )$ . Once sufficient context has been established, each subsequent unmasking becomes marginal, and the redundancy score rises to $R _ { l } ( t ) > 0 . 9 9$ across all layers (Fig. 3(b), yellow region). Because $\mathbf { O } _ { l } ^ { ( t ) } = \mathbf { A } _ { l } ^ { ( t ) } \mathbf { V } _ { l } ^ { ( t ) }$ , near-identical attention maps $( \mathbf { A } _ { l } ^ { ( t ) } \approx \mathbf { A } _ { l } ^ { ( t - 1 ) } )$ combined with only marginally updated values indicate approximately stationary outputs $\mathbf { O } _ { l } ^ { ( t ) } \approx \mathbf { O } _ { l } ^ { ( t - 1 ) }$ during this denoising phase. This plateau spans the majority of the trajectory and constitutes the primary locus of temporal redundancy: the model pays full $\mathcal { O } \dot { ( } L \cdot N ^ { 2 } )$ cost per step for diminishing informational return.

(iii) Terminal refinement $( \bar { J } _ { l }$ secondary rise). A secondary JSD rise emerges as each remaining step must resolve a growing fraction of the remaining ambiguity: with ${ \sim } N _ { g } / \bar { T }$ tokens unmasked per step but progressively fewer masked positions left, each unmasking carries disproportionate semantic weight, requiring fine-grained disambiguation that redistributes attention in the deeper layers.

## 3.3 DRAMA

## 3.3.1 Spatial-Redundancy-Aware Visual Token Pruning.

Given the extreme spatial concentration identified in Section 3.1, a natural strategy is to retain only the most attended visual tokens at each denoising step. However, scoring by attention from the current step requires a forward pass that has already been computed. A cross-step reuse variant that scores tokens using the previous step's attention would circumvent this, but still introduces per-step overhead across all $T \cdot L$ layer passes and faces a cold-start problem at $t = 1$ . Crucially, both strategies overlook a key finding from our analysis: the identity of the dominant patches is content-dependent yet step-invariant (Section 3.1), meaning a single, well-chosen selection made before denoising can remain valid across all $T$ steps. This motivates a fundamentally different approach: rather than relying on attention signals that are either unavailable or redundantly recomputed, we score patches by their cross-modal relevance to the input question, performing the selection at the beginning of the denoising process at zero marginal cost.

![](images/833d9305f70ff6051d1d7ba1a9a5a93c4b204052c29f12a70fdd2de11343da6a.jpg)  
Figure 4: Overview of DRAMA. DRAMA applies Spatial-Redundancy-Aware Visual Token Pruning (SVTP) to select a compact patch subset once before denoising. Sufficiency-Guided Adaptive Stopping (SGAS) terminates the denoising trajectory early.

Specifically, the frozen vision encoder and the dLLM's token embedding layer project visual patches and question tokens into a shared representation space, providing a natural relevance signal without any additional computation. We score each patch i by the similarity between its mean-pooled embedding $\bar { \bf p } _ { i }$ and the mean-pooled question representation q in this space, and construct the active set $\mathcal { P }$ by retaining the global low-resolution overview base patch together with the highest-scoring sub-patches. At every denoising step t and layer $l ,$ attention is computed over the filtered key-value set:

$$
\hat { \mathbf { A } } _ { l } ^ { ( t ) } = \mathrm { S o f t m a x } \left( \frac { \mathbf { Q } \mathbf { K } [ \mathcal { P } \cup \mathcal { T } ] ^ { \top } } { \sqrt { d _ { k } } } \right) { \mathbf { V } } [ \mathcal { P } \cup \mathcal { T } ] , \mathrm { w h e r e } \mathcal { P } = \{ \boldsymbol { 0 } \} \cup \left\{ i \mid s _ { i } \geq s _ { ( | \mathcal { P } | ) } \right\} ,\tag{4}
$$

with $s _ { i } = \sin ( \bar { \bf p } _ { i } , \bar { \bf q } )$ and $\mathcal { T } = \mathcal { T } _ { c } \cup \mathcal { T } _ { g }$ denoting all text token indices.

This reduces the effective key-value length from N to $| { \mathcal { P } } | { \cdot } M + | { \mathcal { T } } | ;$ since visual tokens constitute over 90% of the sequence $( N _ { v } \gg N _ { c } + N _ { q } )$ , even moderate pruning yields a substantial per-step speedup that compounds across all $T$ steps. The entire scoring procedure involves only $N _ { p }$ inner products in $\mathbb { R } ^ { D }$ and is performed once before the first denoising step, introducing no per-step overhead or attention caching.

## 3.3.2 Sufficiency-Guided Adaptive Stopping.

The plateau established in Section 3.2 shows that, beyond the structure-formation phase, consecutive denoising steps produce near-identical attention distributions across all layers. A Pinsker-type bound on the resulting layer-output drift (full derivation in $\mathsf { A p p }$ . D) then propagates this stationarity through the residual stream and the output projection to the decoded token sequence, allowing us to elevate the decoded text to a token-space sufficiency surrogate. This surrogate supplants any runtime monitoring ${ \bar { J } } _ { l } ( t )$ , which would otherwise re-introduce the per-layer attention caching we seek to eliminate, with a stopping rule defined in decoded sequence space.

Stop predicate. We frame the stopping rule as the first-hit time of a conjunction of two failure-modeorthogonal predicates,

$$
\begin{array} { r l } & { t ^ { \star } = \operatorname* { m i n } \big \{ t \geq T _ { \operatorname* { m i n } } : t \in \mathcal { E } , \mathrm { ~ S u f f } ( \mathbf { x } _ { t } ) \wedge \mathrm { C o h } ( \mathbf { x } _ { t } ) \big \} , } \\ & { } \\ { \mathrm { w h e r e } \mathrm { S u f f } ( \mathbf { x } _ { t } ) : = \mathbb { K } \big [ \mathrm { P A R S E } ( \mathbf { x } _ { t } ) \in \mathcal { V } _ { t a s k } \big ] , \mathrm { ~ C o h } ( \mathbf { x } _ { t } ) : = \mathbb { K } \big [ r ( \mathbf { x } _ { t } ) \in \mathcal { R } _ { \theta ( t ) } \big ] , } \end{array}\tag{5}
$$

with $\pmb { r } ( \mathbf x _ { t } ) = ( r ^ { \mathrm { d i v } } , r ^ { \mathrm { r e p } } , r ^ { \mathrm { c o n } } )$ the diversity, peak-frequency, and consecutive-repetition statistics of $\mathbf { x } _ { t } , \mathcal { R } _ { \theta ( t ) }$ their admissible region under the threshold schedule $\mathbf { \theta } \mathbf { \theta } \mathbf { \theta } \theta ( t ) , \mathcal { V } _ { t a s k }$ the set of structurally valid task answers, and $\mathcal { E }$ the granularity-controlled evaluation set; the procedure returns $\mathbf { x } _ { t ^ { \star } }$ , defaulting to the full-budget output $\mathbf { x } _ { T }$ when the predicate never triggers, so the worst-case behaviour gracefully recovers the unaccelerated baseline. The two predicates discharge complementary failure modes: Suff alone admits parseable answers embedded in degenerate near-mask sequences, while Coh alone admits fluent but task-irrelevant continuations, so their conjunction rules out both pathologies by construction, requiring neither retraining nor an auxiliary verifier. The concrete forms of PARSE, r, and the per-variant schedules $\pmb \theta ( t )$ are detailed in App. B.3 due to space.

Design space. Eq. (5) exposes two orthogonal design axes whose Cartesian product instantiates the $2 \times 2$ design space ablated in Section 4.3. Evaluation granularity controls when the predicate is queried along the trajectory: a checkpoint schedule evaluates only on a sparse set C, amortising decoding overhead over a coarse partition, whereas a fine-grained schedule evaluates at every step beyond $T _ { \mathrm { m i n } }$ to admit the earliest viable stop. Threshold scheduling controls how strictly the predicates are evaluated: a progressive schedule tightens $\pmb \theta ( t )$ early and relaxes it later, encoding the structure-formation prior surfaced in Section 3.2, whereas a uniform schedule applies fixed thresholds throughout, encoding the absence of such a prior. The two axes are independent by construction, with the first governing temporal density and the second semantic strictness, so the four resulting variants form a $2 \times 2$ instantiation whose two axes correspond to the two redundancy structures measured in Section 3.2 and 3.1. The full procedure (Algorithm 1, deferred to App. B.3) and the empirical stopping distribution under each variant (Fig. 5) concentrate near the plateau onset predicted in Section 3.2, closing the loop between dissection and design.

## 4 Experiments

## 4.1 Experimental Setup

Datasets and Baselines. We evaluate on six vision-language benchmarks spanning diverse task formats: AI2D [22], MMBench [33], MME [13], MMMU [54], MM-Star [8], and SEED-Bench [26]. For SEED-Bench, we evaluate on a randomly sampled subset of 2,000 samples on LLaDA-V and 1,000 samples on LaViDa, following LLaDA-V's own protocol; all other benchmarks are evaluated in full. We compare against LLaDA-V [52] with fixed denoising budgets of $T \in \{ 1 6 , 3 2 , 6 4 \}$ steps, representing the natural baseline of uniform step reduction. The 64-step configuration serves as the unaccelerated reference throughout. We further provide comparisons of LaViDa on five of the evaluation benchmarks, in Appendix Table 6.

Implementation Details. We adopt LLaDA-V [52], a 7B-parameter multimodal dLLM built on $L = 3 2$ transformer layers with $H = 3 2$ attention heads and a frozen SigLIP2 [43] vision encoder, as our evaluation backbone. Inference is performed in bfloat16 on a single NVIDIA RTX PRO 6000 GPU (96 GB) with a batch size of 1. Generation uses a single block of $\bar { N _ { g } } = 1 2 8$ tokens with deterministic decoding (temperature 0.0) and low-confidence remasking [38]. For SGAS, the maximum denoising budget is $T = 6 4$ with $T _ { \mathrm { m i n } } = 1 ;$ checkpoint mode evaluates at steps $\mathcal { C } = \{ 1 6 , 3 2 , 4 8 , 6 4 \}$ , and fine-grained mode evaluates at every step beyond $T _ { \mathrm { m i n } }$ . The full study consumes 747.99 GPU-hours in total (including baselines, ablations, and preliminary runs), of which 82.18 GPU-hours are spent on the four reported DRAMA variants.

## 4.2 Main Results

Accuracy-Speed Trade-Off. We report accuracy, wall-clock latency, throughput, and speedup in Table 1 across all six benchmarks for three fixed-budget baselines, four full DRAMA variants, and five ablation configurations (detailed in Section 4.3). DRAMA CK Prog and CK Uni achieve ${ \sim } 9 . 5 \times$ average speedup at 64.5% accuracy (—1.9 pp vs. the 64-step reference), simultaneously surpassing Baseline 32 in both speed $( \mathord { \sim } 5 \times$ faster) and accuracy (+0.4 pp). FG Prog offers a middle ground at 10.00× and 64.5% (—1.9 pp), comparable to CK variants but with finer-grained stopping control. At the aggressive end, FG Uni pushes the speedup to $1 8 . 6 7 \times \mathrm { \ a t \ 6 3 . 0 \% \ ( - 3 . 4 p p ) }$ . All DRAMA variants substantially outperform Baseline 16 (51.8%, 3.80×) in both quality and speed, confirming that analysis-guided acceleration yields strictly better Pareto trade-offs than uniform step reduction. Notably, acceleration does not always come at an accuracy cost: on MME, every DRAMA variant surpasses the unaccelerated baseline (78.9% vs. 78.5%), and on AI2D, all variants match it, suggesting that pruning redundant visual tokens can alleviate attention dilution when the model's full visual budget exceeds the task's information requirement. The accuracy-latency Pareto frontier (Fig. 1) further illustrates this: DRAMA variants consistently dominate the front across all benchmarks, with no fixed-budget baseline achieving a comparable speed-accuracy combination.

Table 1: Results on six benchmarks. Metrics: Acc (%, ↑), Latency (s/sample, ↓), TPS (tokens/s, ↑), Speedup vs. Baseline 64 steps (↑). w/o SGAS uses 32 denoising steps; its Speedup is normalized by ÷2 to discount the inherent gain from halving the step count. Best 2nd best(Baselines and DRAMA);Best 2nd best (Ablation). †Evaluated on a subset of the full benchmark.
<table><tr><td rowspan=2 colspan=12>Baseline                        Ablation                          DRAMA (Ours)64    32    16   w/o  w/o SVTP w/o SVTP w/o SVTP w/o SVTPMetric                                                                    FG Prog  FG Uni  CK Prog  CK Unisteps   steps  steps  SGAS  FG Prog  FG Uni  CK Prog  CK Uni</td></tr><tr><td rowspan=1 colspan=1>AI2D</td></tr><tr><td rowspan=1 colspan=1>Acc (%)</td><td rowspan=1 colspan=1>73.0</td><td rowspan=1 colspan=2>72.7  66.3</td><td rowspan=1 colspan=1>72.6</td><td rowspan=1 colspan=1>72.7</td><td rowspan=1 colspan=1>72.4</td><td rowspan=1 colspan=2>73.3    72.6</td><td rowspan=1 colspan=1>73.0</td><td rowspan=1 colspan=1>73.0</td><td rowspan=1 colspan=1>73.0   73.0</td></tr><tr><td rowspan=1 colspan=1>Latency (s)</td><td rowspan=1 colspan=1>78.2</td><td rowspan=1 colspan=2>41.9  19.9</td><td rowspan=1 colspan=1>10.4</td><td rowspan=1 colspan=1>18.9</td><td rowspan=1 colspan=1>9.1</td><td rowspan=1 colspan=2>24.2    20.4</td><td rowspan=1 colspan=1>5.5</td><td rowspan=1 colspan=1>2.6</td><td rowspan=1 colspan=1>5.7    6.1</td></tr><tr><td rowspan=1 colspan=1>TPS (tok/s)</td><td rowspan=1 colspan=1>2.4</td><td rowspan=1 colspan=2>4.2   9.2</td><td rowspan=1 colspan=1>12.2</td><td rowspan=1 colspan=1>9.8</td><td rowspan=1 colspan=1>23.4</td><td rowspan=1 colspan=2>7.8    9.0</td><td rowspan=1 colspan=1>23.9</td><td rowspan=1 colspan=1>60.2</td><td rowspan=1 colspan=1>23.5   21.7</td></tr><tr><td rowspan=1 colspan=1>Speedup</td><td rowspan=1 colspan=1>1.00×</td><td rowspan=1 colspan=2>1.87× 3.93×</td><td rowspan=1 colspan=1>3.76×</td><td rowspan=1 colspan=1>4.14×</td><td rowspan=1 colspan=1>8.59×</td><td rowspan=1 colspan=2>3.23×   3.83×</td><td rowspan=1 colspan=1>14.22×</td><td rowspan=1 colspan=1>30.08×</td><td rowspan=1 colspan=1>13.72× 12.82×</td></tr><tr><td rowspan=1 colspan=1>MMBench</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Acc (%)</td><td rowspan=1 colspan=1>81.9</td><td rowspan=1 colspan=2>79.3  63.9</td><td rowspan=1 colspan=1>79.9</td><td rowspan=1 colspan=1>78.9†</td><td rowspan=1 colspan=1>70.3</td><td rowspan=1 colspan=2>78.5    79.3</td><td rowspan=1 colspan=1>77.8</td><td rowspan=1 colspan=1>69.0</td><td rowspan=1 colspan=1>78.1   78.1</td></tr><tr><td rowspan=1 colspan=1>Latency (s)</td><td rowspan=1 colspan=1>48.8</td><td rowspan=1 colspan=2>24.2  12.3</td><td rowspan=1 colspan=1>11.0</td><td rowspan=1 colspan=1>11.9</td><td rowspan=1 colspan=1>5.7</td><td rowspan=1 colspan=2>12.1    13.4</td><td rowspan=1 colspan=1>5.4</td><td rowspan=1 colspan=1>2.5</td><td rowspan=1 colspan=1>5.6    5.8</td></tr><tr><td rowspan=1 colspan=1>TPS (tok/s)</td><td rowspan=1 colspan=1>2.8</td><td rowspan=1 colspan=2>5.7   11.3</td><td rowspan=1 colspan=1>11.6</td><td rowspan=1 colspan=1>11.7</td><td rowspan=1 colspan=1>25.3</td><td rowspan=1 colspan=2>11.5    10.4</td><td rowspan=1 colspan=1>23.6</td><td rowspan=1 colspan=1>53.7</td><td rowspan=1 colspan=1>22.9   21.8</td></tr><tr><td rowspan=1 colspan=1>Speedup</td><td rowspan=1 colspan=1>1.00×</td><td rowspan=1 colspan=2>2.02× 3.97×</td><td rowspan=1 colspan=1>2.22×</td><td rowspan=1 colspan=1>4.10×</td><td rowspan=1 colspan=1>8.56×</td><td rowspan=1 colspan=2>4.03×   3.64×</td><td rowspan=1 colspan=1>9.04×</td><td rowspan=1 colspan=1>19.52×</td><td rowspan=1 colspan=1>8.71×  8.41×</td></tr><tr><td rowspan=1 colspan=1>MME</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Acc (%)</td><td rowspan=1 colspan=1>78.5</td><td rowspan=1 colspan=2>76.4  66.3</td><td rowspan=1 colspan=1>75.7</td><td rowspan=1 colspan=1>78.7</td><td rowspan=1 colspan=1>78.7</td><td rowspan=1 colspan=2>78.7    78.7</td><td rowspan=1 colspan=1>78.9</td><td rowspan=1 colspan=1>78.7</td><td rowspan=1 colspan=1>78.9   78.9</td></tr><tr><td rowspan=1 colspan=1>Latency (s)</td><td rowspan=1 colspan=1>50.0</td><td rowspan=1 colspan=2>31.7  15.1</td><td rowspan=1 colspan=1>9.9</td><td rowspan=1 colspan=1>11.6</td><td rowspan=1 colspan=1>7.7</td><td rowspan=1 colspan=2>13.9    13.8</td><td rowspan=1 colspan=1>5.5</td><td rowspan=1 colspan=1>3.3</td><td rowspan=1 colspan=1>5.9    5.8</td></tr><tr><td rowspan=1 colspan=1>TPS (tok/s)</td><td rowspan=1 colspan=1>2.8</td><td rowspan=1 colspan=1>4.3</td><td rowspan=1 colspan=1>9.3</td><td rowspan=1 colspan=1>12.8</td><td rowspan=1 colspan=1>12.1</td><td rowspan=1 colspan=1>21.4</td><td rowspan=1 colspan=1>10.6</td><td rowspan=1 colspan=1>10.7</td><td rowspan=1 colspan=1>23.9</td><td rowspan=1 colspan=1>46.0</td><td rowspan=1 colspan=1>22.9   23.2</td></tr><tr><td rowspan=1 colspan=1>Speedup</td><td rowspan=1 colspan=1>1.00×</td><td rowspan=1 colspan=1>1.58×</td><td rowspan=1 colspan=1>3.31×</td><td rowspan=1 colspan=1>2.53×</td><td rowspan=1 colspan=1>4.31×</td><td rowspan=1 colspan=1>6.49×</td><td rowspan=1 colspan=1>3.60×</td><td rowspan=1 colspan=1>3.62×</td><td rowspan=1 colspan=1>9.09×</td><td rowspan=1 colspan=1>15.15×</td><td rowspan=1 colspan=1>8.47×  8.62×</td></tr><tr><td rowspan=1 colspan=1>MMMU</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Acc (%)</td><td rowspan=1 colspan=1>42.9</td><td rowspan=1 colspan=1>42.4</td><td rowspan=1 colspan=1>37.2</td><td rowspan=1 colspan=1>42.5</td><td rowspan=1 colspan=1>41.6</td><td rowspan=1 colspan=1>41.0</td><td rowspan=1 colspan=2>43.0    41.8</td><td rowspan=1 colspan=1>41.6</td><td rowspan=1 colspan=1>41.6</td><td rowspan=1 colspan=1>41.5   41.5</td></tr><tr><td rowspan=1 colspan=1>Latency (s)</td><td rowspan=1 colspan=1>58.7</td><td rowspan=1 colspan=1>29.3</td><td rowspan=1 colspan=1>15.9</td><td rowspan=1 colspan=1>10.5</td><td rowspan=1 colspan=1>15.0</td><td rowspan=1 colspan=1>8.9</td><td rowspan=1 colspan=2>17.9    15.5</td><td rowspan=1 colspan=1>7.5</td><td rowspan=1 colspan=1>4.7</td><td rowspan=1 colspan=1>7.8    7.6</td></tr><tr><td rowspan=1 colspan=1>TPS (tok/s)</td><td rowspan=1 colspan=1>2.5</td><td rowspan=1 colspan=1>5.1</td><td rowspan=1 colspan=1>9.6</td><td rowspan=1 colspan=1>12.1</td><td rowspan=1 colspan=1>10.3</td><td rowspan=1 colspan=1>21.0</td><td rowspan=1 colspan=1>9.0</td><td rowspan=1 colspan=1>10.0</td><td rowspan=1 colspan=1>20.8</td><td rowspan=1 colspan=1>46.3</td><td rowspan=1 colspan=1>19.9   20.4</td></tr><tr><td rowspan=1 colspan=1>Speedup</td><td rowspan=1 colspan=1>1.00×</td><td rowspan=1 colspan=1>2.00×</td><td rowspan=1 colspan=1>3.69×</td><td rowspan=1 colspan=1>2.80×</td><td rowspan=1 colspan=1>3.91×</td><td rowspan=1 colspan=1>6.60×</td><td rowspan=1 colspan=1>3.28×</td><td rowspan=1 colspan=1>3.79×</td><td rowspan=1 colspan=1>7.83×</td><td rowspan=1 colspan=1>12.49×</td><td rowspan=1 colspan=1>7.53×  7.72×</td></tr><tr><td rowspan=1 colspan=1>MM-Star</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Acc (%)</td><td rowspan=1 colspan=1>51.8</td><td rowspan=1 colspan=1>50.3</td><td rowspan=1 colspan=1>37.5</td><td rowspan=1 colspan=1>48.7</td><td rowspan=1 colspan=1>51.8</td><td rowspan=1 colspan=1>51.7</td><td rowspan=1 colspan=2>51.9    51.9</td><td rowspan=1 colspan=1>47.6</td><td rowspan=1 colspan=1>47.6</td><td rowspan=1 colspan=1>47.5   47.5</td></tr><tr><td rowspan=1 colspan=1>Latency (s)</td><td rowspan=1 colspan=1>63.0</td><td rowspan=1 colspan=1>29.3</td><td rowspan=1 colspan=1>16.2</td><td rowspan=1 colspan=1>10.1</td><td rowspan=1 colspan=1>15.3</td><td rowspan=1 colspan=1>9.6</td><td rowspan=1 colspan=2>17.4    17.0</td><td rowspan=1 colspan=1>5.8</td><td rowspan=1 colspan=1>3.5</td><td rowspan=1 colspan=1>6.3    6.4</td></tr><tr><td rowspan=1 colspan=1>TPS (tok/s)</td><td rowspan=1 colspan=1>2.5</td><td rowspan=1 colspan=1>5.3</td><td rowspan=1 colspan=1>9.9</td><td rowspan=1 colspan=1>12.6</td><td rowspan=1 colspan=1>11.3</td><td rowspan=1 colspan=1>25.2</td><td rowspan=1 colspan=2>10.2    10.4</td><td rowspan=1 colspan=1>23.5</td><td rowspan=1 colspan=1>49.7</td><td rowspan=1 colspan=1>22.2   21.8</td></tr><tr><td rowspan=1 colspan=1>Speedup</td><td rowspan=1 colspan=1>1.00×</td><td rowspan=1 colspan=1>2.15×</td><td rowspan=1 colspan=1>3.89×</td><td rowspan=1 colspan=1>3.12×</td><td rowspan=1 colspan=1>4.12×</td><td rowspan=1 colspan=1>6.56×</td><td rowspan=1 colspan=2>3.62×  3.71×</td><td rowspan=1 colspan=1>10.86×</td><td rowspan=1 colspan=1>18.00×</td><td rowspan=1 colspan=1>10.00× 9.84×</td></tr><tr><td rowspan=1 colspan=1>SEED-Bench</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Acc (%)</td><td rowspan=1 colspan=1>70.1†</td><td rowspan=1 colspan=2>63.7† 39.3†</td><td rowspan=1 colspan=1>61.3†</td><td rowspan=1 colspan=1>70.4†</td><td rowspan=1 colspan=1>70.3†</td><td rowspan=1 colspan=2>70.4†   70.4†</td><td rowspan=1 colspan=1>68.2†</td><td rowspan=1 colspan=1>68.2†</td><td rowspan=1 colspan=1>68.2†  68.2†</td></tr><tr><td rowspan=1 colspan=1>Latency (s)</td><td rowspan=1 colspan=1>50.3</td><td rowspan=1 colspan=2>25.4  12.5</td><td rowspan=1 colspan=1>11.3</td><td rowspan=1 colspan=1>12.5</td><td rowspan=1 colspan=1>6.7</td><td rowspan=1 colspan=2>13.9    13.8</td><td rowspan=1 colspan=1>5.6</td><td rowspan=1 colspan=1>3.0</td><td rowspan=1 colspan=1>5.7    5.8</td></tr><tr><td rowspan=1 colspan=1>TPS (tok/s)</td><td rowspan=1 colspan=1>2.7</td><td rowspan=1 colspan=2>5.3   10.8</td><td rowspan=1 colspan=1>11.3</td><td rowspan=1 colspan=1>11.2</td><td rowspan=1 colspan=1>26.6</td><td rowspan=1 colspan=1>10.2</td><td rowspan=1 colspan=1>10.3</td><td rowspan=1 colspan=1>23.8</td><td rowspan=1 colspan=1>55.5</td><td rowspan=1 colspan=1>23.3   23.1</td></tr><tr><td rowspan=1 colspan=1>Speedup</td><td rowspan=1 colspan=1>1.00×</td><td rowspan=1 colspan=1>1.98×</td><td rowspan=1 colspan=1>4.02×</td><td rowspan=1 colspan=1>2.23×</td><td rowspan=1 colspan=1>4.02×</td><td rowspan=1 colspan=1>7.51×</td><td rowspan=1 colspan=1>3.62×</td><td rowspan=1 colspan=1>3.64×</td><td rowspan=1 colspan=1>8.98×</td><td rowspan=1 colspan=1>16.77×</td><td rowspan=1 colspan=1>8.82×  8.67×</td></tr><tr><td rowspan=1 colspan=1>Average</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Acc (%)</td><td rowspan=1 colspan=1>66.4</td><td rowspan=1 colspan=2>64.1  51.8</td><td rowspan=1 colspan=1>63.5</td><td rowspan=1 colspan=1>65.7</td><td rowspan=1 colspan=1>64.1</td><td rowspan=1 colspan=1>66.0</td><td rowspan=1 colspan=1>65.8</td><td rowspan=1 colspan=1>64.5</td><td rowspan=1 colspan=1>63.0</td><td rowspan=1 colspan=1>64.5   64.5</td></tr><tr><td rowspan=1 colspan=1>Latency (s)</td><td rowspan=1 colspan=1>58.2</td><td rowspan=1 colspan=2>30.3  15.3</td><td rowspan=1 colspan=1>10.5</td><td rowspan=1 colspan=1>14.2</td><td rowspan=1 colspan=1>8.0</td><td rowspan=1 colspan=1>16.6</td><td rowspan=1 colspan=1>15.7</td><td rowspan=1 colspan=1>5.9</td><td rowspan=1 colspan=1>3.3</td><td rowspan=1 colspan=1>6.2    6.3</td></tr><tr><td rowspan=1 colspan=1>TPS (tok/s)</td><td rowspan=1 colspan=1>2.6</td><td rowspan=1 colspan=2>5.0   10.0</td><td rowspan=1 colspan=1>12.1</td><td rowspan=1 colspan=1>11.1</td><td rowspan=1 colspan=1>23.8</td><td rowspan=1 colspan=1>9.9</td><td rowspan=1 colspan=1>10.1</td><td rowspan=1 colspan=1>23.3</td><td rowspan=1 colspan=1>51.9</td><td rowspan=1 colspan=1>22.5   22.0</td></tr><tr><td rowspan=1 colspan=1>Speedup</td><td rowspan=1 colspan=1>1.00×</td><td rowspan=1 colspan=2>1.93× 3.80×</td><td rowspan=1 colspan=1>2.78×</td><td rowspan=1 colspan=1>4.10×</td><td rowspan=1 colspan=1>7.39×</td><td rowspan=1 colspan=2>3.56×  3.71×</td><td rowspan=1 colspan=1>10.00×</td><td rowspan=1 colspan=1>18.67×</td><td rowspan=1 colspan=1>9.54×  9.35×</td></tr></table>

Throughput. In absolute throughput, the baseline's 2.6 tokens/s is impractical for interactive use. DRAMA CK variants raise this to 22.3 tokens/s, while FG Uni reaches 51.9 tokens/s on average and peaks at 60.2 tokens/s on AI2D, bringing multimodal dLLM inference into a regime comparable to autoregressive models of similar scale. Even the most conservative variant, CK Prog, delivers a \~9× throughput increase, confirming that the redundancy structures identified in Section 3.1 and Section 3.2 translate into substantial real-world speed gains.

Per-Benchmark Variation. Acceleration gains vary with task characteristics. On AI2D, FG Uni achieves 30.08× speedup while matching the 64-step baseline accuracy (73.0%), as diagram-centric queries concentrate attention on a few salient patches. On MME, CK variants simultaneously improve accuracy (78.9% vs. 78.5%) and deliver up to 8.47× speedup, reflecting rapid attention convergence for perceptual tasks. On MMMU, CK variants retain 41.5% (—1.4 pp) at \~7.5× speedup; on MM-Star, 47.5% (−4.3 pp) at \~10×; MMBench follows a similar pattern at 78.1% (—3.8 pp, \~8.5×). We further disentangle the contribution of each component in Section 4.3, where SGAS alone matches accuracy on most of the benchmarks, confirming that it serves as a stable backbone that absorbs the accuracy cost of aggressive visual token pruning

## 4.3 Ablation Study

We further ablate (i) the individual contributions of SVTP and SGAS as orthogonal cost-axis reductions; and (ii) the interaction between SGAS's evaluation-granularity and threshold-scheduling axes that defines the 2×2 design space introduced in Section 3.3.2. We defer (iii) generalisation to a second backbone (LaViDa); and (iv) head-to-head comparisons with two concurrent multimodal dLLM acceleration methods to Appendix C

![](images/d0fbe10e3c1863715e27fbb2e5a5c406fbb602892f5059d90b53fc036ec5b602.jpg)  
Figure 5: Distribution of actual denoising steps under SGAS across six benchmarks. Each subplot shows the histogram and KDE for one SGAS variant. FG Uni concentrates most samples at very early steps $( \mu { \approx } 8 – 1 2 )$ , while CK variants cluster around the first eligible checkpoint (step 16). Progressive scheduling shifts the distribution rightward, reflecting its stricter early-step requirements.

Effect of SVTP. At a fixed budget of T = 32 steps, SVTP alone yields 2.78× normalised speedup at 63.5% accuracy, reducing per-sample time from 30.3s to 10.5 s (see the ablation columns of Table 1). The gain comes entirely from shortening the key-value sequence, consistent with the spatial concentration in Section 3.1, and accuracy remains within —2.4 to +0.6 pp of the 32 steps baseline on all benchmarks. When combined with SGAS, SVTP contributes a consistent \~2.5× multiplicative factor regardless of the stopping variant. This confirms that spatial pruning and temporal stopping target orthogonal cost axes (N and T in $\mathcal { O } ( T \cdot L \cdot N ^ { 2 } ) )$ and compose independently.

Effect of SGAS. Without SVTP, all four SGAS configurations achieve 3.56–7.39× speedup while staying within 2.3 pp of the 64 steps baseline on average (see the ablation columns of Table 1). The most conservative variant, CK Prog, matches or exceeds the 64-step baseline's accuracy on five of six benchmarks at 3.56× speedup, indicating that the fixed budget over-denoises on most tasks. At the aggressive end, FG Uni reaches 7.39× while matching Baseline 32 accuracy. This confirms that the temporal redundancy identified in Section 3.2 translates directly into our SGAS component without dedicated retraining.

Stopping Strategy Interaction. We discover that evaluation granularity (FG vs. CK) and threshold schedule (Prog vs. Uni) interact strongly. Under uniform thresholds, FG nearly doubles CK's speedup, aligning with per-step evaluation's more exit opportunities. Under progressive thresholds, this gap largely vanishes: the stricter early-step requirements absorb FG's frequent evaluations, keeping accuracy close to the baseline. Symmetrically, switching from Prog to Uni barely affects CK $( 3 . { \bar { 5 } } 6 \times { \bar { v } } s . 3 . 7 1 { \bar { \times } } )$ but dramatically shifts FG $( 4 . 1 \times \overset { \cdot } { v } s . 7 . 4 \times )$ . We visualize DRAMA's stopping-step distributions (Fig. 5, per-benchmark breakdowns are shown in Supplementary Materials), which corroborate our JSD redundancy analysis: FG Uni concentrates exits at early steps, CK variants cluster around the first checkpoint, and progressive scheduling shifts the distribution rightward. This shows that progressive scheduling acts as a safeguard during the structure-formation phase (Section 3.2), while still allowing the system to exploit plateau-phase redundancy for substantial acceleration.

## 5 Conclusion

We present DRAMA, a training-free inference acceleration framework for multimodal dLLMs that closes the gap between merely observing attention redundancy and systematically exploiting it. Driven by a measured dissection of where redundancy lies, DRAMA delivers an average 18.67× inference speedup on LLaDA-V across six vision-language benchmarks with competitive accuracy, attained without any training, architectural modification, or attention caching. Beyond its empirical gains, DRAMA reframes acceleration in this emerging paradigm as a problem of identifying and exploiting structural redundancy, offering an analysis-driven perspective which informs broader efforts toward measurement-grounded acceleration of multimodal diffusion large language models.

## Acknowledgments and Disclosure of Funding

This work was supported by the National Science and Technology Council (NSTC) of Taiwan under Grant 115-2634-F-002-012 and by the NTU AI Center of Research Excellence (AI-CoRE), with additional computational support provided through H100 GPUs donated by Wistron Corporation. Shao-Yuan Lo is honored to be supported by the Yushan Young Fellow Program by the Ministry of Education (MOE) of Taiwan.

## References

[1] Sudhanshu Agrawal, Risheek Garrepalli, Raghavv Goel, Mingu Lee, Christopher Lott, and Fatih Porikli. Spiffy: Multiplying diffusion llm acceleration via lossless speculative decoding. arXiv preprint arXiv:2509.18085, 2025.

[2] Marianne Arriola, Aaron Gokaslan, Justin T Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block diffusion: Interpolating between autoregressive and diffusion 1anguage models. arXiv preprint arXiv:2503.09573, 2025.

[3] Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in neural information processing systems, 34:17981–17993,2021.

[4] Zechen Bai, Pichao Wang, Tianjun Xiao, Tong He, Zongbo Han, Zheng Zhang, and Mike Zheng Shou. Hallucination of multimodal large language models: A survey. arXiv preprint arXiv:2404.18930, 2024.

[5] Parikshit Bansal and Sujay Sanghavi. Enabling approximate joint sampling in diffusion lms. arXiv preprint arXiv:2509.22738, 2025.

[6] Wenrui Bao, Zhiben Chen, Dan Xu, and Yuzhang Shang. Learning to parallel: Accelerating diffusion large language models via adaptive parallel decoding. In The Fourteenth International Conference on Learning Representations, 2025.

[7] Shuochen Chang, Xiaofeng Zhang, Qingyang Liu, and Li Niu. D3tom: Decider-guided dynamic token merging for accelerating diffusion mllms. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 19961–19969, 2026.

[8] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? Advances in Neural Information Processing Systems, 37:27056–27087, 2024.

[9] Xinhua Chen, Sitao Huang, Cong Guo, Chiyue Wei, Yintao He, Jianyi Zhang, Hai Li, Yiran Chen, et al. Dpad: Efficient diffusion language models with suffix dropout. arXiv preprint arXiv:2508.14148, 2025.

[10] Zigeng Chen, Gongfan Fang, Xinyin Ma, Ruonan Yu, and Xinchao Wang. dparallel: Learnable parallel decoding for dllms. arXiv preprint arXiv:2509.26488, 2025.

[11] Google DeepMind. Gemini diffusion. https://blog.google/technology/google-deepmind/gemini-diffusion/. https://blog.google/technology/google-deepmind/gemini-diffusion/, 2025. Accessed: 2024-07-24.

[12] Yihong Dong, Zhaoyu Ma, Xue Jiang, Zhiyuan Fan, Jiaru Qian, Yongmin Li, Jianha Xiao, Zhi Jin, Rongyu Cao, Binhua Li, et al. Saber: An efficient sampling with adaptive acceleration and backtracking enhanced remasking for diffusion language model. arXiv preprint arXiv:2510.18165, 2025.

[13] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023.

[14] Feiyang Fu, Tongxian Guo, and Zhaoqiang Liu. Learnable sampler distillation for discrete diffusion models. arXiv preprint arXiv:2509.19962, 2025.

[15] Yifeng Gao, Ziang Ji, Yuxuan Wang, Biqing Qi, Hanlin Xu, and Linfeng Zhang. Self speculative decoding for diffusion large language models. arXiv preprint arXiv:2510.04147, 2025.

[16] Feng Hong, Geng Yu, Yushi Ye, Haicheng Huang, Huangjie Zheng, Ya Zhang, Yanfeng Wang, and Jiangchao Yao. Wide-in, narrow-out: Revokable decoding for efficient and effective dllms. arXiv preprint arXiv:2507.18578, 2025.

[17] Zhanqiu Hu, Jian Meng, Yash Akhauri, Mohamed S Abdelfattah, Jae-sun Seo, Zhiru Zhang, and Udit Gupta. Flashdlm: Accelerating diffusion language model inference via efficient kv caching and guided diffusion. arXiv preprint arXiv:2505.21467, 2025.

[18] Pengcheng Huang, Shuhao Liu, Zhenghao Liu, Yukun Yan, Shuo Wang, Zulong Chen, and Tong Xiao. Pc-sampler: Position-aware calibration of decoding bias in masked diffusion models. arXiv preprint arXiv:2508.13021, 2025.

[19] Daniel Israel, Tian Jin, Ellie Cheng, Guy Van den Broeck, Aditya Grover, Suvinay Subramanian, and Michael Carbin. Planned diffusion. arXiv preprint arXiv:2510.18087, 2025.

[20] Huiqiang Jiang, Yucheng Li, Chengruidong Zhang, Qianhui Wu, Xufang Luo, Surin Ahn, Zhenhua Han, Amir H Abdi, Dongsheng Li, Chin-Yew Lin, et al. Minference 1.0: Accelerating pre-filling for long-context 1lms via dynamic sparse attention. Advances in Neural Information Processing Systems, 37:52481–52515, 2024.

[21] Yuchu Jiang, Yue Cai, Xiangzhong Luo, Jiale Fu, Jiarui Wang, Chonghan Liu, and Xu Yang. d2 cache: Accelerating diffusion-based llms via dual adaptive caching. arXiv preprint arXiv:2509.23094, 2025.

[22] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In European conference on computer vision, pages 235–251. Springer, 2016.

[23] Fanheng Kong, Jingyuan Zhang, Yahui Liu, Zirui Wu, Yu Tian, Guorui Zhou, et al. Accelerating diffusion 1lm inference via local determinism propagation. arXiv preprint arXiv:2510.07081, 2025.

[24] Inception Labs, Samar Khanna, Siddhant Kharbanda, Shufan Li, Harshit Varma, Eric Wang, Sawyer Birnbaum, Ziyang Luo, Yanis Miraoui, Akash Palrecha, et al. Mercury: Ultra-fast language models based on diffusion. arXiv preprint arXiv:2506.17298, 2025.

[25] Xunhao Lai, Jianqiao Lu, Yao Luo, Yiyuan Ma, and Xun Zhou. Flexprefill: A context-aware sparse attention mechanism for efficient long-sequence inference. arXiv preprint arXiv:2502.20766, 2025.

[26] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023.

[27] Pengxiang Li, Yefan Zhou, Dilxat Muhtar, Lu Yin, Shilin Yan, Li Shen, Soroush Vosoughi, and Shiwei Liu, Diffusion language models know the answer before decoding. arXiv preprint arXiv:2508.19982, 2025.

[28] Shufan Li, Konstantinos Kallidromitis, Hritik Bansal, Akash Gokul, Yusuke Kato, Kazuki Kozuka, Jason Kuen, Zhe Lin, Kai-Wei Chang, and Aditya Grover. Lavida: A large diffusion language model for multimodal understanding. arXiv preprint arXiv:2505.16839, 2025.

[29] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 292–305, 2023.

[30] Yucheng Li, Huiqiang Jiang, Chengruidong Zhang, Qianhui Wu, Xufang Luo, Surin Ahn, Amir H Abdi, Dongsheng Li, Jianfeng Gao, Yuqing Yang, et al. Mminference: Accelerating pre-filling for long-context vlms via modality-aware permutation sparse attention. arXiv preprint arXiv:2504.16083, 2025.

[31] Hanchao Liu, Wenyuan Xue, Yifei Chen, Dapeng Chen, Xiutian Zhao, Ke Wang, Liping Hou, Rongjun Li, and Wei Peng. A survey on hallucination in large vision-language models. arXiv preprint arXiv:2402.00253, 2024.

[32] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024. URL https://11ava-v1. github. io/ blog/2024-01-30-11ava-next/.

[33] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024.

[34] Zhiyuan Liu, Yicun Yang, Yaojie Zhang, Junjie Chen, Chang Zou, Qingyuan Wei, Shaobo Wang, and Linfeng Zhang. dllm-cache: Accelerating diffusion large language models with adaptive caching. arXiv preprint arXiv:2506.06295, 2025.

[35] Xinyin Ma, Runpeng Yu, Gongfan Fang, and Xinchao Wang. dkv-cache: The cache for diffusion language models. arXiv preprint arXiv:2505.15781, 2025.

[36] Yuxin Ma, Lun Du, Lanning Wei, Kun Chen, Qian Xu, Kangyu Wang, Guofeng Feng, Guoshan Lu, Lin Liu, Xiaojing Qi, et al. dinfer: An efficient inference framework for diffusion language models. arXiv preprint arXiv:2510.08666, 2025.

[37] Quan Nguyen-Tri, Mukul Ranjan, and Zhiqiang Shen. Attention is all you need for kv cache in diffusion 1lms. arXiv preprint arXiv:2510.14973, 2025.

[38] Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. arXiv preprint arXiv:2502.09992, 2025.

[39] Yifan Pu, Zhuofan Xia, Jiayi Guo, Dongchen Han, Qixiu Li, Duo Li, Yuhui Yuan, Ji Li, Yizeng Han, Shiji Song, et al. Efficient diffusion transformer with step-wise dynamic attention mediators. In European Conference on Computer Vision, pages 424–441. Springer, 2024.

[40] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8309–8318. IEEE, 2019.

[41] Yuerong Song, Xiaoran Liu, Ruixiao Li, Zhigeng Liu, Zengfeng Huang, Qipeng Guo, Ziwei He, and Xipeng Qiu. Sparse-dllm: Accelerating diffusion llms with dynamic cache eviction. arXiv preprint arXiv:2508.02558, 2025.

[42] Yuxuan Song, Zheng Zhang, Cheng Luo, Pengyang Gao, Fan Xia, Hao Luo, Zheng Li, Yuehang Yang, Hongli Yu, Xingwei Qu, et al. Seed diffusion: A large-scale diffusion language model with high-speed inference. arXiv preprint arXiv:2508.02193, 2025.

[43] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.

[44] Xu Wang, Chenkai Xu, Yijie Jin, Jiachun Jin, Hao Zhang, and Zhijie Deng. Diffusion llms can do faster-than-ar inference via discrete diffusion forcing. arXiv preprint arXiv:2508.09192, 2025.

[45] Qingyan Wei, Yaojie Zhang, Zhiyuan Liu, Dongrui Liu, and Linfeng Zhang. Accelerating diffusion large language models with slowfast sampling: The three golden principles. arXiv preprint arXiv:2506.10848, 2025.

[46] Chengyue Wu, Hao Zhang, Shuchen Xue, Shizhe Diao, Yonggan Fu, Zhijian Liu, Pavlo Molchanov, Ping Luo, Song Han, and Enze Xie. Fast-dllm v2: Efficient block-diffusion llm. arXiv preprint arXiv:2509.26328, 2025.

[47] Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, and Enze Xie. Fast-dllm: Training-free acceleration of diffusion llm by enabling kv cache and parallel decoding. arXiv preprint arXiv:2505.22618, 2025.

[48] Chengyue Wu, Shiyi Lan, Yonggan Fu, Sensen Gao, Jin Wang, Jincheng Yu, Jose M Alvarez, Pavlo Molchanov, Ping Luo, Song Han, et al. Fast-dvlm: Efficient block-diffusion vlm via direct conversion from autoregressive vlm. arXiv preprint arXiv:2604.06832, 2026.

[49] Shutong Wu and Jiawei Zhang. Free draft-and-verification: Toward lossless parallel decoding for diffusion large language models. arXiv preprint arXiv:2510.00294, 2025.

[50] Jingqi Xu, Jingxi Lu, Chenghao Li, Sreetama Sarkar, Souvik Kundu, and Peter A Beerel. Redvtp: Trainingfree acceleration of diffusion vision-language models inference via masked token-guided visual token pruning. arXiv preprint arXiv:2511.12428, 2025.

[51] Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke Shen, Yunhai Tong, and Mengdi Wang. Mmada Multimodal large diffusion language models. arXiv preprint arXiv:2505.15809, 2025.

[52] Zebin You, Shen Nie, Xiaolu Zhang, Jun Hu, Jun Zhou, Zhiwu Lu, Ji-Rong Wen, and Chongxuan Li. Llada-v: Large language diffusion models with visual instruction tuning. arXiv preprint arXiv:2505.16933, 2025.

[53] Zhihang Yuan, Hanling Zhang, Lu Pu, Xuefei Ning, Linfeng Zhang, Tianchen Zhao, Shengen Yan, Guohao Dai, and Yu Wang. Ditfastattn: Attention compression for diffusion transformer models. Advances in Neural Information Processing Systems, 37:1196–1219, 2024.

[54] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9556–9567, 2024.

[55] Hanling Zhang, Rundong Su, Zhihang Yuan, Pengtao Chen, Mingzhu Shen, Yibo Fan, Shengen Yan, Guohao Dai, and Yu Wang. Ditfastattnv2: Head-wise attention compression for multi-modality diffusion transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16399–16409, 2025.

## A Limitations and Potential Negative Social Impact

## A.1 Limitations

Our work accelerates inference for multimodal dLLMs without modifying the underlying model weights or training data. As such, it is not intended to introduce new capabilities beyond those already present in the base model. Rather than improving the model's intrinsic reasoning or multimodal understanding ability, our method focuses on making existing capabilities more efficient at inference time. This limitation is shared by prior training-free acceleration methods [35, 34, 47, 37, 50, 7], whose primary goal is to reduce latency and computation while preserving model quality as much as possible, instead of advancing the underlying performance frontier.

## A.2 Potential Negative Social Impact

By substantially lowering the computational cost of deploying multimodal dLLMs, DRAMA may also reduce the barrier for malicious use, including the large-scale generation of misleading or deceptive visual question answering outputs [31, 4]. Such misuse could amplify the spread of incorrect visual interpretations or plausible-sounding but unreliable multimodal responses. This concern is not specific to our method, but is shared broadly by inference acceleration techniques that improve the accessibility and scalability of foundation models.

## B Additional Implementation Details

## B.1 Datasets and Evaluation Splits

We evaluate on six standard vision-language benchmarks. For AI2D [22], MMBench [33], MME [13], MMMU [54], and MM-Star [8], we use the full evaluation split released by each benchmark. For SEED-Bench [26], we evaluate on a fixed random subset of 2,000 samples (seed fixed for reproducibility) to manage computational cost; the same subset is used across all baselines, ablations, and DRAMA variants to ensure fair comparison. Table 2 summarizes the splits and sample counts.

Table 2: Evaluation splits used in our experiments.
<table><tr><td>Benchmark</td><td>Split</td><td>#Samples</td></tr><tr><td>AI2D [22]</td><td>test (full)</td><td>3,088</td></tr><tr><td>MMBench [33]</td><td>dev-en (full)</td><td>4,329</td></tr><tr><td>MME [13]</td><td>full</td><td>2,374</td></tr><tr><td>MMMU [54]</td><td>validation (full)</td><td>1,050</td></tr><tr><td>MM-Star [8]</td><td>full</td><td>1,500</td></tr><tr><td>SEED-Bench [26]</td><td>random subset (seed-fixed)</td><td>2,000</td></tr></table>

## B.2 Prompt Template

Following LLaDA-V [52], we use a multi-turn chat format with a system message (“You are a helpful language and vision assistant. .."), a user turn containing the image token, question, and a chain-ofthought thinking instruction, and an empty assistant turn for the model to complete. All experiments use a fixed generation length of $N _ { g } = 1 2 8$ tokens with deterministic decoding (temperature 0.0) and low-confidence remasking. Table 3 shows an example of a prompt template used for our evaluation benchmarks.

## B.3 SGAS Configuration Details

We summarize the four SGAS variants evaluated in the main paper in Table 4. Under progressive scheduling, the stopping criterion is strict at early evaluation steps, requiring both semantic sufficiency and high lexical coherence, and progressively relaxed at later steps, reflecting the diminishing marginal return of additional denoising iterations identified in the temporal analysis of the main paper. Under uniform scheduling, fixed thresholds are applied throughout all evaluation steps. Checkpoint mode only evaluates at the sparse set ${ \mathcal { C } } ,$ while fine-grained mode evaluates at every step beyond $T _ { \mathrm { m i n } }$ providing more exit opportunities at the cost of slightly more frequent criterion checks.

Table 3: An example of prompt template
<table><tr><td>Component</td><td>Content</td></tr><tr><td>Image token</td><td>&lt;image&gt;</td></tr><tr><td>Question</td><td>Question: {question}</td></tr><tr><td>Options</td><td>Options:</td></tr><tr><td rowspan="3"></td><td>A. {option_a}</td></tr><tr><td>B. {option_b}</td></tr><tr><td>You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within &lt;think&gt; &lt;/think&gt; tags, i.e. &lt;think&gt; reasoning process here &lt;/think&gt;</td></tr><tr><td>instruction</td><td>&lt;answer&gt; answer here: Please select the correct option by responding with only the letter (A, B, C, D, or E) &lt;/answer&gt;</td></tr></table>

Table 4: SGAS configuration variants. All variants share $T = 6 4 , T _ { \mathrm { m i n } } = 1$
<table><tr><td>Variant</td><td>Granularity</td><td>Schedule</td><td>Evaluation Steps</td></tr><tr><td>CK Prog</td><td>Checkpoint</td><td>Progressive</td><td> $\mathcal { C } = \{ 1 6 , 3 2 , 4 8 , 6 4 \}$ </td></tr><tr><td>CK Uni</td><td>Checkpoint</td><td>Uniform</td><td> $\mathcal { C } = \{ 1 6 , 3 2 , 4 8 , 6 4 \}$ </td></tr><tr><td>FG Prog</td><td>Fine-grained</td><td>Progressive</td><td> $\mathrm { E v e r y ~ s t e p ~ } t > T _ { \mathrm { m i n } }$ </td></tr><tr><td>FG Uni</td><td>Fine-grained</td><td>Uniform</td><td> $\mathrm { E v e r y ~ s t e p } \ t > T _ { \mathrm { m i n } }$ </td></tr></table>

## B.3.1 Stopping Decision Procedure

We formalize the full SGAS procedure in Algorithm 1. After each denoising step, if $t \geq T _ { \mathrm { m i n } }$ and t is admitted by the granularity schedule E, the decoded sequence is evaluated against two complementary checks. The semantic sufficiency check parses the decoded text under the task schema. The lexical coherence check verifies three statistics of $\mathbf { x } _ { t }$ against the threshold vector $\pmb { \theta } _ { t } = ( \theta _ { t } ^ { \mathrm { d i v } } , \theta _ { t } ^ { \mathrm { r e p } } , \theta _ { t } ^ { \mathrm { c o n } } )$

• Vocabulary diversity $r ^ { \mathrm { d i v } } \colon$ ratio of unique tokens to total non-mask tokens.

• Max single-token frequency $r ^ { \mathrm { r e p . } }$ fraction of the most repeated token.

• Consecutive-repetition ratio $r ^ { \mathrm { c o n } }$ : fraction of adjacent token pairs that are identical.

The coherence gate passes when $r ^ { \mathrm { d i v } } \geq \theta _ { t } ^ { \mathrm { d i v } } , r ^ { \mathrm { r e p } } \leq \theta _ { t } ^ { \mathrm { r e p } }$ , and $r ^ { \mathrm { c o n } } \leq \theta _ { t } ^ { \mathrm { c o n } }$ simultaneously. If both checks pass, denoising terminates and $\mathbf { x } _ { t }$ is returned; otherwise the procedure proceeds to step t+1. If no step in the trajectory triggers the predicate, $\mathbf { x } _ { T }$ is returned, recovering the unaccelerated baseline as a worst case.

## B.3.2 Threshold Parameters

Table 5 lists the threshold vectors used by each variant. Checkpoint variants use a set of per-checkpoint thresholds; fine-grained uniform uses a single fixed vector; and fine-grained progressive smoothly interpolates thresholds as a function of denoising progress $p = t / T$

## B.4 Compatibility with Block Diffusion

Block diffusion [2] is a semi-autoregressive generation paradigm for dLLMs in which the $N _ { g }$ generation tokens are partitioned into B blocks of size $\bar { b } = \bar { N } _ { q } / B$ , with each block denoised sequentially over its own trajectory of $T _ { b }$ steps; it is a generation strategy rather than an acceleration mechanism. Our experiments adopt the default single-block setting $( \stackrel { \triangledown } { \boldsymbol { B } } = 1 )$ of LLaDA-V [52], but DRAMA extends to the multi-block regime without modification. The visual prefix is shared across all blocks, so SVTP performs its cross-modal scoring once before the first block and reuses the selected patch set $\mathcal { P }$ for every subsequent block. SGAS evaluates its stop predicate inside each block independently, using the same threshold schedule $\pmb \theta ( t )$ , so each block terminates at its own first-hit step $t _ { b } ^ { \star }$ without coupling across blocks. The aggregate skipped-step count is then $\begin{array} { r } { \sum _ { b = 1 } ^ { B } \left( T _ { b } - t _ { b } ^ { \star } \right) } \end{array}$

Algorithm 1 Sufficiency-Guided Adaptive Stopping   
Require: Denoising budget T, minimum steps $T _ { \mathrm { m i n } } ,$ evaluation set E ▶ C or $\{ T _ { \operatorname* { m i n } } + 1 , \dots , T \}$   
Require: Threshold schedule $\theta ( \cdot ) = ( \theta ^ { \mathrm { d i v } } , \bar { \theta ^ { \mathrm { r e p } } } , \theta ^ { \mathrm { c o n } } )$ progressive or uniform   
1: $\mathbf { \dot { x } } _ { 0 } \gets [ \mathbf { M A S K } ] ^ { N _ { g } }$   
2: for $t = 1 , \dots , T$ do   
3: xt ← DENOISE $\left( \mathbf { x } _ { t - 1 } \right)$ Predict & unmask ${ \sim } N _ { g } / T$ tokens   
4: $\mathbf { i f } t < T _ { \operatorname* { m i n } }$ or $\sharp \mathcal { E } ^ { \varepsilon }$ then   
5: continue   
6: end if   
7: $( \theta _ { t } ^ { \mathrm { d i v } } , \theta _ { t } ^ { \mathrm { r e p } } , \theta _ { t } ^ { \mathrm { c o n } } ) \gets \theta ( t )$   
8: text ← DECODE(xt)   
9: sufficient ← SEMANTICSUFFICIENCYCHECK(text, task)   
10: $r ^ { \mathrm { d i v } } \gets \mathrm { V O C A B U L A R Y D I V E R S I T Y } ( \mathbf { x } _ { t } )$   
11: $r ^ { \mathrm { r e p } } \gets \mathbf { M a X T O K E N F R E Q U E N C Y } ( \mathbf { \dot { x } } _ { t } )$   
12: rcon ← CONSECUTIVEREPETITIONRATIO(xt)   
13: coherent $ ( r ^ { \mathrm { d i v } } \geq \theta _ { t } ^ { \mathrm { d i v } } ) \wedge ( r ^ { \mathrm { r e p } } \leq \theta _ { t } ^ { \mathrm { r e p } } ) \wedge ( r ^ { \mathrm { c o n } } \leq \theta _ { t } ^ { \mathrm { c o n } } )$   
14: if sufficient and coherent then   
15: return $\mathbf { x } _ { t }$ Early termination   
16: end if   
17: end for   
18: return $\mathbf { x } _ { T }$ Fallback: full budget

Table 5: Lexical coherence thresholds for all SGAS variants $( T = 6 4 )$
<table><tr><td>Variant</td><td>Step / Phase</td><td> $\theta ^ { \mathrm { d i v } } \uparrow$ </td><td> $\theta ^ { \mathrm { r e p } } \downarrow$ </td><td> $\theta ^ { \mathrm { c o n } } \downarrow$ </td></tr><tr><td rowspan="4">CK Uni</td><td>t = 16</td><td>0.50</td><td>0.22</td><td>0.09</td></tr><tr><td>t = 32</td><td>0.50</td><td>0.22</td><td>0.09</td></tr><tr><td>t = 48</td><td>0.50</td><td>0.22</td><td>0.09</td></tr><tr><td>t = 64</td><td>0.50</td><td>0.22</td><td>0.09</td></tr><tr><td rowspan="4">CK Prog</td><td>t = 16</td><td>0.50</td><td>0.23</td><td>0.09</td></tr><tr><td>t = 32</td><td>0.55</td><td>0.20</td><td>0.08</td></tr><tr><td>t = 48</td><td>0.50</td><td>0.23</td><td>0.09</td></tr><tr><td>t = 64</td><td>0.45</td><td>0.25</td><td>0.10</td></tr><tr><td>FG Uni</td><td>All steps</td><td>0.50</td><td>0.22</td><td>0.09</td></tr><tr><td rowspan="2">FG Prog</td><td> $p \leq 0 . 5 ( \mathrm { t i g h t e n } )$ </td><td>0.45→0.55</td><td>0.25→0.20</td><td>0.10→0.08</td></tr><tr><td> $p > 0 . 5 ( \mathrm { r e l a x } )$ </td><td>0.55→0.45</td><td>0.20→0.25</td><td>0.08→0.10</td></tr></table>

## B.5 Hyperparameter selection.

All settings reported in this paper are fixed once and shared across every benchmark and both backbones; we perform no per-benchmark tuning and no search. SVTP retains a single patch, $| \mathcal { P } | = 1$ , in all experiments. The SGAS threshold vectors of Table 5 were set from the output statistics of 100 LLaDA-V samples on MMMU, and the resulting values were then applied unchanged to the remaining five benchmarks and to LaViDa. Accuracy on the benchmarks we report was therefore never consulted when choosing these values.

## C Detailed Ablation Study

## C.1 Model Architecture Ablation

We further evaluate DRAMA on LaViDa [28], a multimodal dLLM that already employs $2 \times 2$ visualtoken pooling and therefore exposes a substantially smaller acceleration headroom than LLaDA-V (LaViDa baseline averages 9.76 s vs. LLaDA-V's 58.2 s under the same think-mode regime). All

Table 6: Benchmark results on LaViDa (128 denoising steps, think mode). Sample counts are given in the benchmark headers. Latency is wall-clock seconds per sample and TPS is computed from the fixed generation length of 128 tokens under the measured latency. Metrics: Acc (%, ↑), Latency (s/sample, ↓), TPS (tokens/s, ↑), Speedup vs. the 128-step baseline (↑), Acc Retention (↑).
<table><tr><td rowspan="2">Metric</td><td rowspan="2">Baseline 128</td><td colspan="5">Ablation</td><td rowspan="2"></td><td colspan="3">DRAMA (Ours)</td></tr><tr><td>w/o</td><td>w/o SVTP FG Prog</td><td>w/o SVTP</td><td>w/o SVTP</td><td>w/o SVTP CK Uni</td><td>FG Prog</td><td>FG Uni</td><td>CK Prog CK Uni</td></tr><tr><td>AI2D (n = 3088)</td><td>steps</td><td>SGAS</td><td></td><td>FG Uni</td><td>CK Prog</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9">Acc (%)</td><td>67.45</td></tr><tr><td>Latency (s)</td><td>69.88 9.98</td><td>69.04 7.40</td><td>68.36 7.04</td><td>68.72 7.07</td><td>68.49 7.49</td><td>68.75 7.50</td><td>67.58 5.30</td><td>67.81 5.23</td><td>67.81 5.50</td><td>5.54</td></tr><tr><td>TPS (tok/s)</td><td>12.82</td><td>17.29</td><td>18.17</td><td>18.11</td><td>17.08</td><td>17.06</td><td>24.13</td><td>24.49</td><td>23.27</td><td>23.11</td></tr><tr><td></td><td></td><td>1.35×</td><td>1.42×</td><td></td><td>1.33×</td><td>1.33×</td><td></td><td>1.91×</td><td></td><td></td></tr><tr><td>Speedup</td><td>1.00×</td><td></td><td>0.98</td><td>1.41×</td><td></td><td></td><td>1.88×</td><td></td><td>1.82×</td><td>1.80×</td></tr><tr><td>Acc Retention</td><td>1.00</td><td>0.99</td><td></td><td>0.98</td><td>0.98</td><td>0.98</td><td>0.97</td><td>0.97</td><td>0.97</td><td>0.97</td></tr><tr><td colspan="9">MME (n = 2374)</td><td></td></tr><tr><td>Acc (%)</td><td>68.91</td><td>69.46</td><td>68.32</td><td>68.62</td><td>66.51</td><td>68.62</td><td>68.53</td><td>69.17</td><td>62.22</td><td>69.17</td></tr><tr><td>Latency (s)</td><td>9.82</td><td>7.15</td><td>7.06</td><td>7.24</td><td>7.04</td><td>7.72</td><td>5.34</td><td>5.40</td><td>5.39</td><td>5.70</td></tr><tr><td>TPS (tok/s)</td><td>13.04</td><td>17.89</td><td>18.12</td><td>17.67</td><td>18.19</td><td>16.58</td><td>23.97</td><td>23.72</td><td>23.76</td><td>22.44</td></tr><tr><td>Speedup</td><td>1.00×</td><td>1.37×</td><td>1.39×</td><td>1.35×</td><td>1.40×</td><td>1.27×</td><td>1.84×</td><td>1.82×</td><td>1.82×</td><td>1.72×</td></tr><tr><td>Acc Retention</td><td>1.00</td><td>1.01</td><td>0.99</td><td>1.00</td><td>0.97</td><td>1.00</td><td>0.99</td><td>1.00</td><td>0.90</td><td>1.00</td></tr><tr><td colspan="9">MMMU (n = 900)</td><td></td></tr><tr><td>Acc (%)</td><td>42.44</td><td>41.78</td><td>41.67</td><td>42.11</td><td>42.00</td><td>42.44</td><td>41.11</td><td>41.22</td><td>41.44</td><td>42.00</td></tr><tr><td>Latency (s)</td><td>9.91</td><td>7.68</td><td>7.06</td><td>7.19</td><td>7.47</td><td>7.60</td><td>5.52</td><td>5.65</td><td>5.77</td><td>5.84</td></tr><tr><td>TPS (tok/s)</td><td>12.91</td><td>16.66</td><td>18.13</td><td>17.79</td><td>17.14</td><td>16.84</td><td>23.19</td><td>22.66</td><td>22.18</td><td>21.90</td></tr><tr><td>Speedup</td><td>1.00×</td><td>1.29×</td><td>1.40×</td><td>1.38×</td><td>1.33×</td><td>1.30×</td><td>1.80×</td><td>1.75×</td><td>1.72×</td><td>1.70×</td></tr><tr><td>Acc Retention</td><td>1.00</td><td>0.98</td><td>0.98</td><td>0.99</td><td>0.99</td><td>1.00</td><td>0.97</td><td>0.97</td><td>0.98</td><td>0.99</td></tr><tr><td colspan="9">MM-Star (n = 1500)</td><td></td><td></td></tr><tr><td>Acc (%) Latency (s)</td><td>43.00</td><td>40.73</td><td>41.33</td><td>42.47</td><td>41.33</td><td>42.20</td><td>39.13</td><td>40.00</td><td>39.00</td><td>40.00</td></tr><tr><td></td><td>9.19</td><td>7.31</td><td>6.50</td><td>6.58</td><td>6.87</td><td>6.99</td><td>5.23</td><td>5.36</td><td>5.63</td><td>5.57</td></tr><tr><td>TPS (tok/s)</td><td>13.93</td><td>17.52</td><td>19.68</td><td>19.45</td><td>18.64</td><td>18.32</td><td>24.46</td><td>23.89</td><td>22.74</td><td>22.97</td></tr><tr><td>Speedup</td><td>1.00×</td><td>1.26×</td><td>1.41×</td><td>1.40×</td><td>1.34×</td><td>1.32×</td><td>1.76×</td><td>1.72×</td><td>1.63×</td><td>1.65×</td></tr><tr><td>Acc Retention</td><td>1.00</td><td>0.95</td><td>0.96</td><td>0.99</td><td>0.96</td><td>0.98</td><td>0.91</td><td>0.93</td><td>0.91</td><td>0.93</td></tr><tr><td colspan="9">SEED-Bench (n = 1000)</td><td></td><td></td></tr><tr><td>Acc (%) Latency (s)</td><td>63.50</td><td>61.00</td><td>61.40</td><td>63.40</td><td>61.40</td><td>63.40</td><td>59.10</td><td>61.00</td><td>59.10</td><td>61.00</td></tr><tr><td></td><td>9.90</td><td>7.34</td><td>7.09</td><td>7.21</td><td>7.52</td><td>7.67</td><td>5.19</td><td>5.24</td><td>5.55</td><td>5.61</td></tr><tr><td>TPS (tok/s)</td><td>12.93</td><td>17.45</td><td>18.06</td><td>17.75</td><td>17.02</td><td>16.68</td><td>24.67</td><td>24.45</td><td>23.06</td><td>22.82</td></tr><tr><td>Speedup</td><td>1.00×</td><td>1.35×</td><td>1.40×</td><td>1.37×</td><td>1.32×</td><td>1.29×</td><td>1.91×</td><td>1.89×</td><td>1.78×</td><td>1.77×</td></tr><tr><td>Acc Retention</td><td>1.00</td><td>0.96</td><td>0.97</td><td>1.00</td><td>0.97</td><td>1.00</td><td>0.93</td><td>0.96</td><td>0.93</td><td>0.96</td></tr><tr><td colspan="9">Average (over the five benchmarks above)</td><td></td><td></td></tr><tr><td>Acc (%)</td><td>57.55</td><td>56.40</td><td>56.22</td><td>57.06</td><td>55.95</td><td>57.08</td><td>55.09</td><td>55.84</td><td>53.91</td><td>55.92</td></tr><tr><td>Latency (s)</td><td>9.76</td><td>7.38</td><td>6.95</td><td>7.06</td><td>7.28</td><td>7.50</td><td>5.32</td><td>5.38</td><td>5.57</td><td>5.65</td></tr><tr><td>TPS (tok/s)</td><td>13.11</td><td>17.35</td><td>18.42</td><td>18.14</td><td>17.59</td><td>17.08</td><td>24.08</td><td>23.81</td><td>22.99</td><td>22.65</td></tr><tr><td>Speedup</td><td>1.00×</td><td>1.32×</td><td>1.40×</td><td>1.38×</td><td>1.34×</td><td>1.30×</td><td>1.84×</td><td>1.82×</td><td>1.75×</td><td>1.73×</td></tr><tr><td>Acc Retention</td><td>1.00</td><td>0.98</td><td>0.98</td><td>0.99</td><td>0.97</td><td>0.99</td><td>0.96</td><td>0.97</td><td>0.94</td><td>0.97</td></tr></table>

LaViDa results are evaluated on the full (AI2D 3088; MME 2374; MM-Star 1500; MMMU 900) benchmark and a 1,000-sample SEED-Bench subset . Despite the compressed margin, DRAMA delivers consistent 1.73–1.84× wall-clock speedup on average across five benchmarks (Table 6), with accuracy preserved to 97.2% overall by CK Uni and 97.0% by FG Uni, and with MME improving over the LaViDa baseline of 68.91, reaching 69.46 under SVTP alone and 69.17 under DRAMA FG Uni and CK Uni, indicating that pruning redundant visual tokens can mitigate attention dilution even when the backbone has already partially compressed its visual context. The pattern of variant trade-offs largely mirrors LLaDA-V: FG Uni and FG Prog dominate the speedup axis, while CK Uni retains the most accuracy, so the original split between the two schedules still holds under full-set evaluation.

Table 7: Per-benchmark comparison with RedVTP on LLaDA-V, both reproduced by us. RedVTP runs at its most aggressive operating point (r=0.25) under the same 32-step think-mode regime as DRAMA.
<table><tr><td>Benchmark</td><td>Method</td><td>Setting</td><td>Performance</td><td>Latency (s)</td><td>Speedup</td><td>Acc. Retention</td></tr><tr><td rowspan="4">AI2D</td><td>LLaDA-V (32-step)</td><td>500 samples</td><td>73.00</td><td>49.194</td><td>1.00×</td><td>100.0%</td></tr><tr><td>+ RedVTP (r=0.25)</td><td>500 samples</td><td>71.60</td><td>38.319</td><td>1.28×</td><td>98.1%</td></tr><tr><td>LLaDA-V (32-step)</td><td>full set (3,088)</td><td>72.73</td><td>41.94</td><td>1.00×</td><td>100.0%</td></tr><tr><td>+ DRAMA (ours)</td><td>full set (3,088)</td><td>73.02</td><td>2.57</td><td>16.32×</td><td>100.4%</td></tr><tr><td rowspan="4">MM-Star</td><td>LLaDA-V (32-step)</td><td>500 samples</td><td>50.00</td><td>34.663</td><td>1.00×</td><td>100.0%</td></tr><tr><td>+ RedVTP (r=0.25)</td><td>500 samples</td><td>42.40</td><td>28.718</td><td>1.21×</td><td>84.8%</td></tr><tr><td>LLaDA-V (32-step)</td><td>full set (1,500)</td><td>50.27</td><td>29.29</td><td>1.00×</td><td>100.0%</td></tr><tr><td>+ DRAMA (ours)</td><td>full set (1,500)</td><td>47.60</td><td>3.52</td><td>8.32×</td><td>94.7%</td></tr><tr><td rowspan="2">Average</td><td>+ RedVTP (r=0.25)</td><td>500 samples</td><td></td><td></td><td>1.25×</td><td>91.5%</td></tr><tr><td>+ DRAMA (ours)</td><td>full set</td><td></td><td></td><td>12.32×</td><td>97.6%</td></tr></table>

Table 8: Same-backbone wall-clock speedup comparison with RedVTP on LaViDa. RedVTP reports only $r = 7 5 \%$ on LaViDa, since LaViDa's built-in $2 \times 2$ visual-token pooling already retains 25% of the original tokens. RedVTP LaViDa numbers are taken from Xu et al. [50]

; DRAMA runs under 128-step think mode. The AI2D and MME columns are measured on every sample (3,088 and 2,374); the MMBench column is measured on a 100-sample subset against a  
baseline on the same subset. Best, 2nd best
<table><tr><td>Method</td><td>AI2D</td><td>MME</td><td>MMBench</td></tr><tr><td>RedVTP regime (LaViDa)</td><td></td><td></td><td></td></tr><tr><td>LaViDa (baseline)</td><td>1.00×</td><td>1.00×</td><td>1.00×</td></tr><tr><td>+ RedVTP (r = 75%)</td><td>1.08×</td><td>1.06×</td><td>1.06×</td></tr><tr><td>DRAMA regime (LaViDa, 128-step think mode)</td><td></td><td></td><td></td></tr><tr><td>LaViDa (baseline)</td><td>1.00×</td><td>1.00×</td><td>1.00×</td></tr><tr><td>DRAMA CK Uni</td><td>1.80×</td><td>1.72×</td><td>1.56×</td></tr><tr><td>DRAMA CK Prog</td><td>1.82×</td><td>1.82×</td><td>1.58×</td></tr><tr><td>DRAMA FG Prog</td><td>1.88×</td><td>1.84×</td><td>1.69×</td></tr><tr><td>DRAMA FG Uni</td><td>1.91×</td><td>1.82×</td><td>1.68×</td></tr></table>

## C.2 Comparison with Concurrent Acceleration Methods

## C.2.1 Comparison with RedVTP [50]

RedVTP [50] accelerates multimodal dLLMs by pruning visual tokens, and is the closest prior method on our backbone. We re-run it ourselves at its most aggressive operating point (r=0.25) under our own 32-step think-mode setting, against a baseline with the same batch and the same 500 samples. Table 7 pairs each method with its own baseline on LLaDA-V. RedVTP buys 1.21–1.28× while retaining 84.8–98.1% of baseline accuracy, averaging 1.25× at 91.5%. DRAMA reaches 8.32– 16.32× at 94.7–100.4% retention, averaging 12.32× at 97.6%. Pruning the visual prefix removes work that is paid once per sample, so its gain is bounded by the share of the forward pass that the prefix occupies; the step axis is not bounded in the same way, and the two axes compose. The same ordering holds on LaViDa (Table 8), where RedVTP reports only $r { = } 7 5 \%$ because the 2×2 visual-token pooling already built into LaViDa absorbs most of its pruning headroom. Its wall-clock gain there is a marginal 1.06–1.08×, against 1.56–1.91× for DRAMA under a heavier reasoning regime. That the ordering survives a backbone whose visual prefix is already compressed supports the reading that the temporal redundancy identified in Section 3.2 is a separate, multiplicative source of acceleration rather than a restatement of spatial redundancy.

## C.2.2 Comparison with D3ToM [7]

D3ToM [7] accelerates LaViDa by merging visual tokens under decider-token guidance. It reports wall-clock time only on LLaVA-Bench in a short-generation setting $\scriptstyle ( T = 3 2 , | O | = 6 4 )$ , so its published speedups do not transfer to think mode. We therefore re-run D3ToM-t at its own default operating point (25% of visual tokens retained, merged at layer 3) under our 128-step think mode and against our own baseline (Table 9). It gains only 1.15-1.23× against DRAMA's 1.63–1.91×, and retains less accuracy on every benchmark, falling to 0.81 and 0.83 on MM-Star and SEED-Bench where DRAMA holds 0.91–0.96. On AI2D and MMMU, the two benchmarks D3ToM also evaluates, our reproduction reaches 0.96 and 0.94 against the 0.94–0.96 and 0.97–0.98 they report.

Table 9: Comparison with D3ToM on LaViDa. All rows are re-run by us under 128-step think mode, D3ToM included, so that accuracy and speedup are measured under one regime against one baseline; retention is accuracy divided by that baseline. Rows use the full set for AI2D, MME, MMMU, MMstar, and a 1000-sample subset for SEED-Bench. Best, 2nd best
<table><tr><td></td><td colspan="2">AI2D</td><td colspan="2">MMMU</td><td colspan="2">MM-Star</td><td colspan="2">SEED-Bench</td><td>Avg.</td></tr><tr><td>Method</td><td>Acc (Ret.)</td><td>Sp.</td><td>Acc (Ret.)</td><td>Sp.</td><td>Acc (Ret.)</td><td>Sp.</td><td>Acc (Ret.)</td><td>Sp.</td><td>Sp.</td></tr><tr><td>LaViDa (baseline)</td><td>69.88 (1.00)</td><td>1.00×</td><td>42.44 (1.00)</td><td>1.00×</td><td>43.00 (1.00)</td><td>1.00×</td><td>63.50 (1.00)</td><td>1.00×</td><td>1.00×</td></tr><tr><td>D3ToM-t (keep=0.25, L3)</td><td>66.80 (0.96)</td><td>1.20×</td><td>39.89 (0.94)</td><td>1.15×</td><td>38.80 (0.81)</td><td>1.21×</td><td>53.00 (0.83)</td><td>1.23×</td><td>1.20×</td></tr><tr><td>DRAMA CK Uni</td><td>67.45 (0.97)</td><td>1.80×</td><td>42.00 (0.99)</td><td>1.70×</td><td>40.00 (0.93)</td><td>1.65×</td><td>61.00 (0.96)</td><td>1.77×</td><td>1.73×</td></tr><tr><td>DRAMA CK Prog</td><td>67.81 (0.97)</td><td>1.82×</td><td>41.44 (0.98)</td><td>1.72×</td><td>39.00 (0.91)</td><td>1.63×</td><td>59.10 (0.93)</td><td>1.78×</td><td>1.74×</td></tr><tr><td>DRAMA FG Prog</td><td>67.58 (0.97)</td><td>1.88×</td><td>41.11 (0.97)</td><td>1.80×</td><td>39.13 (0.91)</td><td>1.76×</td><td>59.10 (0.93)</td><td>1.91×</td><td>1.84×</td></tr><tr><td>DRAMA FG Uni</td><td>67.81 (0.97)</td><td>1.91×</td><td>41.22 (0.97)</td><td>1.75×</td><td>40.00 (0.93)</td><td>1.72×</td><td>61.00 (0.96)</td><td>1.89×</td><td>1.82×</td></tr></table>

Table 10: Held-out evaluation on POPE and TextVQA, both on LLaDA-V. Neither benchmark was used when setting any threshold. Speedup is wall-clock against the 64-step baseline of the same benchmark. POPE reports F1 on 500 samples; TextVQA reports accuracy on 100 samples. “SGAS only" is DRAMA with SVTP disabled.
<table><tr><td rowspan="2">Setting</td><td colspan="2">POPE (F1)</td><td colspan="2">TextVQA (Acc)</td></tr><tr><td>Score</td><td>Speedup</td><td>Score</td><td>Speedup</td></tr><tr><td>Baseline (64 steps)</td><td>88.30</td><td>1.00×</td><td>64.60</td><td>1.00×</td></tr><tr><td>Baseline (32 steps)</td><td>86.37</td><td>1.90×</td><td>60.70</td><td>2.00×</td></tr><tr><td>Baseline (16 steps)</td><td>87.24</td><td>3.77×</td><td></td><td></td></tr><tr><td>SGAS only (FG Uni)</td><td>88.30</td><td>3.81×</td><td></td><td></td></tr><tr><td>DRAMA FG Prog</td><td>85.35</td><td>10.69×</td><td>51.60</td><td>7.91×</td></tr><tr><td>DRAMA FG Uni</td><td>85.35</td><td>11.23×</td><td>52.50</td><td>5.87×</td></tr></table>

## C.3 Generalization to Additional Benchmarks

Neither TextVQA [40] nor POPE [29] was used when setting any threshold, so both serve as held-out checks on whether the redundancy structures of Section 3.1 3.2 transfer beyond the benchmarks we report in the main paper. They probe opposite ends of the same question. POPE asks whether a named object is present, which depends on the visual prefix surviving pruning; TextVQA asks open-ended questions about dense text inside the image, which depends on both the prefix and a sufficient number of denoising steps. Table 10 reports both.

On POPE, SGAS alone is lossless: all four schedules reproduce the 64-step F1 of 88.30 while reaching up to 3.81 ×, confirming that the temporal stabilization measured in Section 3.2 holds on a task the thresholds were never tuned for. Adding SVTP costs 2.95 F1 and buys 11.23× speedup. That the loss appears only once the visual prefix is pruned is consistent with the nature of the task: object-existence judgements read directly off the prefix, so they are the first to suffer when it is compressed.

TextVQA is where DRAMA is weakest: accuracy falls from 64.60 at 64 steps to 52.50 for FG Uni at 5.87×, and halving the step budget alone already accounts for 3.90 of those points. As stated in Appendix A.1, DRAMA adds no mechanism to recover quality lost to a reduced budget, and on open-ended text-dense inputs that cost is simply larger than on the six main benchmarks.

## D Derivation of the Output-Stationarity Bound

This appendix supplies the derivation of the Pinsker-type output-stationarity bound invoked in Section 3.3.2, showing that the per-step layer-output drift vanishes as a direct corollary of the JSD collapse identified in Section 3.2.

## D.1 Setup and Decomposition

At denoising step t and transformer layer l, let $\mathbf { A } _ { l } ^ { ( t ) } \in \mathbb { R } ^ { H \times N \times N }$ be the attention tensor, $\mathbf { V } _ { l } ^ { ( t ) } \in$ $\mathbb { R } ^ { H \times N \times d _ { v } }$ the value tensor, and $\mathbf { O } _ { l } ^ { ( t ) } \mathbf { \Lambda } = \mathbf { A } _ { l } ^ { ( t ) } \mathbf { V } _ { l } ^ { ( t ) }$ the layer output. Adding and subtracting $\mathbf { A } _ { l } ^ { ( t ) } \mathbf { V } _ { l } ^ { ( t - 1 ) }$ yields the canonical decomposition

$$
\begin{array} { r } { \Delta \mathbf { O } _ { l } ^ { ( t ) } \equiv \mathbf { O } _ { l } ^ { ( t ) } - \mathbf { O } _ { l } ^ { ( t - 1 ) } = \underbrace { \Delta \mathbf { A } _ { l } ^ { ( t ) } \mathbf { V } _ { l } ^ { ( t - 1 ) } } _ { \mathrm { a t t e n t i o n } \mathrm { d r i f t } } + \underbrace { \mathbf { A } _ { l } ^ { ( t ) } \Delta \mathbf { V } _ { l } ^ { ( t ) } } _ { \mathrm { v a l u e ~ u p d a t e } } , } \end{array}\tag{6}
$$

with $\Delta \mathbf { A } _ { l } ^ { ( t ) } = \mathbf { A } _ { l } ^ { ( t ) } - \mathbf { A } _ { l } ^ { ( t - 1 ) }$ and $\Delta \mathbf { V } _ { l } ^ { ( t ) } = \mathbf { V } _ { l } ^ { ( t ) } - \mathbf { V } _ { l } ^ { ( t - 1 ) }$ . The triangle inequality, combined with submultiplicativity in the Frobenius/operator-norm pair, gives

$$
\big \| \Delta \mathbf { O } _ { l } ^ { ( t ) } \big \| _ { F } \leq \big \| \Delta \mathbf { A } _ { l } ^ { ( t ) } \big \| _ { F } \big \| \mathbf { V } _ { l } ^ { ( t - 1 ) } \big \| _ { \mathrm { o p } } + \big \| \mathbf { A } _ { l } ^ { ( t ) } \big \| _ { \mathrm { o p } } \big \| \Delta \mathbf { V } _ { l } ^ { ( t ) } \big \| _ { F } .\tag{7}
$$

The two summands are bounded separately in the next two subsections.

## D.2 Bounding the Attention-Drift Term via JSD

For each (head h, query i), the per-row attention vectors $\mathbf { a } _ { l , h , i } ^ { ( t ) } , \mathbf { a } _ { l , h , i } ^ { ( t - 1 ) } \in \Delta ^ { N - 1 }$ are probability distributions over key positions. They satisfy the Lin (1991) inequality, the JSD analogue of Pinsker's bound,

$$
\begin{array} { r } { \mathrm { J S D } \big ( { \mathbf a } _ { l , h , i } ^ { ( t ) } \ \| { \mathbf a } _ { l , h , i } ^ { ( t - 1 ) } \big ) \geq \frac { 1 } { 2 } \mathrm { T V } \big ( { \mathbf a } _ { l , h , i } ^ { ( t ) } , { \mathbf a } _ { l , h , i } ^ { ( t - 1 ) } \big ) ^ { 2 } . } \end{array}\tag{8}
$$

Combined with $\| \mathbf { a } _ { l , h , i } ^ { ( t ) } - \mathbf { a } _ { l , h , i } ^ { ( t - 1 ) } \| _ { 2 } \leq \| \cdot \| _ { 1 } = 2 \mathrm { T V } ( \cdot )$ and aggregation over all rows,

$$
\big \| \Delta \mathbf { A } _ { l } ^ { ( t ) } \big \| _ { F } ^ { 2 } = \sum _ { h , i } \big \| \mathbf { a } _ { l , h , i } ^ { ( t ) } - \mathbf { a } _ { l , h , i } ^ { ( t - 1 ) } \big \| _ { 2 } ^ { 2 } \leq H N \bar { J } _ { l } ( t ) ,\tag{9}
$$

where ${ \bar { J } } _ { l } ( t )$ is the mean per-layer divergence defined in Eq. (3). Substituting Eq. (9) into the first summand of Eq. (7) recovers the headline form used in the main text,

$$
\big \| \Delta \mathbf { A } _ { l } ^ { ( t ) } \mathbf { V } _ { l } ^ { ( t - 1 ) } \big \| _ { F } \leq C _ { l } \sqrt { 2 \bar { J } _ { l } ( t ) } , \qquad C _ { l } : = 2 \sqrt { H N } \cdot \big \| \mathbf { V } _ { l } ^ { ( t - 1 ) } \big \| _ { \mathrm { o p } } .\tag{10}
$$

Standard layer-norm and weight-decay regularisation keep $\| \mathbf { V } _ { l } ^ { ( t - 1 ) } \| _ { \mathrm { o p } }$ in a bounded operator-norm regime across steps, so $C _ { l }$ is a step-independent constant for the purposes of the argument.

## D.3 Bounding the Value-Update Term via Layer-wise Propagation

The value tensor at layer l is computed from the previous layer's output through a Lipschitz block $f _ { l }$ (linear projection composed with layer norm and, in pre-norm transformers, residual addition): $\mathbf { V } _ { l } ^ { ( t ) } = f _ { l } ( \mathbf { O } _ { l - 1 } ^ { ( t ) } )$ with Lipschitz constant $L _ { l } < \infty$ . Hence

$$
\big \| \Delta \mathbf { V } _ { l } ^ { ( t ) } \big \| _ { F } \leq L _ { l } \big \| \Delta \mathbf { O } _ { l - 1 } ^ { ( t ) } \big \| _ { F } .\tag{11}
$$

Inserting Eqs. (10) and (11) into Eq. (7) produces the layer-wise recurrence

$$
\big \| \boldsymbol { \Delta } \mathbf { O } _ { l } ^ { ( t ) } \big \| _ { F } \leq C _ { l } \sqrt { 2 \bar { J } _ { l } ( t ) } + \rho _ { l } \big \| \boldsymbol { \Delta } \mathbf { O } _ { l - 1 } ^ { ( t ) } \big \| _ { F } , \qquad \rho _ { l } : = \big \| \mathbf { A } _ { l } ^ { ( t ) } \big \| _ { \mathrm { o p } } L _ { l } .\tag{12}
$$

Unrolling along the layer stack with the boundary condition $\Delta \mathbf { O } _ { 0 } ^ { ( t ) } = \mathbf { 0 }$ at the input embedding (whose only step-to-step change comes from the \~ $\dot { N _ { g } } / T$ newly unmasked tokens, which the converged attention assigns negligible aggregate weight) gives

$$
\big \| \Delta \mathbf { O } _ { l } ^ { ( t ) } \big \| _ { F } \leq \sum _ { k = 1 } ^ { l } \Big ( \prod _ { j = k + 1 } ^ { l } \rho _ { j } \Big ) C _ { k } \sqrt { 2 \bar { J } _ { k } ( t ) } .\tag{13}
$$

![](images/d637e3d24a860965dd6cb4b5e95e3ce1ffaecbf13f01a28396937d14cc096529.jpg)

![](images/efe18e87214b4868ca7fdabf4f67d6c00aea6ca7f000c936ed896215ab2fac38.jpg)

![](images/44ef0fcc7a2a9002537f8b369831bc9bac964dfa12173122a75d0933653547d9.jpg)

![](images/b97337c32751a0be2a4e140700b6586cf460564e7f1793b4c7d416d59d0864a7.jpg)

Figure 6: Spatial redundancy analysis – AI2D. The diagram-centric nature of AI2D leads to highly concentrated attention on a single informative patch.  
![](images/dedeb9f7336d1fde67fd10e4f82d77ee0603708ce2e09e6cbbec5c6665215f1f.jpg)

![](images/f0ed824ac925a4ef28d994547ad663fc1115bb0b646c43ff8c7aa5b6652240e9.jpg)

![](images/3f1677b794e53588434623326fbe83d144348d31a0c17bf8ec0ad815a20bf4e7.jpg)  
Figure 7: Spatial redundancy analysis – MMBench. Despite diverse image content, the top 2–3 patches consistently capture the majority of output→image attention.

The temporal dissection in Section 3.2 establishes that $\bar { J } _ { k } ( t ) \to 0$ simultaneously across all $k \in [ L ]$ once past the structure-formation phase (the bright plateau region of Fig. 3(b)). Every term in the sum is therefore suppressed in lockstep, and $\lVert \Delta \mathbf { O } _ { l } ^ { ( t ) } \rVert _ { F } \to 0$ uniformly in l. The simultaneity is essential: it converts what would otherwise be a worst-case accumulating bound into one in which all contributing factors vanish concurrently.

## D.4 From Layer Output to Decoded Sequence

The unembedding head is a fixed linear map $\mathbf { W } _ { U }$ followed by softmax, so the per-token logit drift obeys $\| \boldsymbol { \Delta } \mathbf { z } ^ { ( t ) } \| _ { F } \leq \| \mathbf { W } _ { U } \| _ { \mathrm { o p } } \| \boldsymbol { \Delta } \mathbf { O } _ { L } ^ { ( t ) } \| _ { F } \to 0$ . Combined with the dLLM low-confidence remasking schedule, which unmasks onl $\mathbf { y } \sim N _ { g } / T$ positions per step, the argmax-decoded sequence inherits the same stationarity and tends to stabilize before the nominal budget T is exhausted. This justifies the promotion of the decoded text to a sufficiency surrogate in Section 3.3.2, replacing any runtime monitoring of ${ \bar { J } } _ { l } ( t )$ that would otherwise require the per-layer attention caching this work seeks to eliminate.

## E Additional Redundancy Analysis and Results

## E.1 Dissecting Spatial Redundancy: Analysis per Benchmark

We visualize spatial redundancy with MMMU [54] for illustration in the main paper. Below, we provide the same four-panel spatial analysis for all six benchmarks individually. Each figure reports: (a) per-rank crop attention fraction, (b) a representative spatial attention heatmap, (c) token-level concentration CDF stratified by modality pair, and (d) the per-query visual token budget κ(0.9) across denoising steps.

The key findings from the main paper generalize consistently: attention mass concentrates on 1– 2 dominant patches across all benchmarks, the t2v concentration curve is the steepest among all modality pairs, and $\kappa ( 0 . 9 )$ remains step-invariant. Benchmarks with simpler visual content $( \mathrm { e . g . }$ AI2D [22] diagrams, MME [13] perceptual images) exhibit slightly higher concentration, while those with richer scenes (e.g., SEEDBench [26], MMBench [33]) show moderately broader spatial distributions—yet even in the latter case, the most-attended patch captures $\ge 4 0 \%$ of output→image attention.

![](images/ade91b7435a27d7a0b5e166c586224ca6f163bedd0ceec09a9d85520236e9e7d.jpg)

![](images/be150d8670bfb07e9fc6f41030a4db696da4df3daa38b26a82f0eb0208c7abdc.jpg)

![](images/cfcea5b82315508eae7578315f8028896790f34baf8846ce62ac88a36ac300dc.jpg)

Figure 8: Spatial redundancy analysis – MME. Perceptual tasks in MME yield concentrated attention patterns, with κ(0.9) remaining stable across all denoising steps.  
![](images/8658a21bc2b38cfa910900160a45f837a24397636948907eac75642f3456e573.jpg)

![](images/10f47ce16816da89b8268c8fbb494f14ba8c01c16821c50f9af0887416c6ae56.jpg)

![](images/778260693dd6cf6cb8628a1f6fe8afd3706cf0623c7ba527cbba491159c3300a.jpg)

![](images/74ce1892a5c47d3afb8fc2965a431ec179144f7e6c1733196136401256310fee.jpg)  
Figure 9: Spatial redundancy analysis – MM-Star. Knowledge-intensive queries in MM-Star show similar spatial concentration to MMMU.

## E.2 Dissecting Temporal Redundancy: Analysis per Benchmark

We present the temporal analysis averaged over all benchmarks in the main paper. Here we provide perbenchmark breakdowns, each showing (a) the per-layer inter-step JSD Ji(t) across all 64 denoising steps, and (b) the layer× step redundancy heatmap $\dot { R _ { l } } ( t )$

The three-phase temporal structure identified in the main paper—structure formation, stable plateau, and terminal refinement—is clearly visible in every benchmark. Across all datasets, the JSD peaks within the first 2–5 steps, then drops by over two orders of magnitude into a prolonged plateau where $R _ { l } ( t ) > 0 . 9 9$ , before a secondary rise in the final 5–10 steps. The heatmaps further reveal that deeper layers (L25–L32) exhibit the lowest redundancy scores throughout the trajectory, consistent with these layers performing the fine-grained disambiguation required for final token selection.

Notably, the plateau onset and duration are remarkably consistent across benchmarks despite their diverse task characteristics (diagram understanding in AI2D [22], perceptual reasoning in MME [13] multi-discipline knowledge in MMMU [54]), confirming that the temporal redundancy is a structural property of the denoising dynamics rather than a task-specific artefact. This consistency is what enables a single set of SGAS hyperparameters to work effectively across all benchmarks without per-task tuning.

## E.3 Pareto Frontier

We present the accuracy-latency Pareto frontier averaged across all six benchmarks in Figure 13. The averaged view confirms that DRAMA variants consistently dominate the frontier: no fixed-budget baseline achieves a comparable combination of speed and accuracy. Among DRAMA variants, CK Prog and CK Uni occupy the high-accuracy region $( \sim 6 4 . 5 \% , \sim 6 \mathrm { s } )$ , while FG Uni pushes to the extreme-speed end $( \sim 6 3 . 0 \bar { \% } , \sim 3 . 3 \bar { \mathrm { s } } )$ , demonstrating the flexibility of the framework.

## E.4 Step Distribution Analysis per Benchmark

We provide the combined step distribution overview in the main paper. Below we show per-benchmark distributions with full statistics for each of the four SGAS configurations under both the w/o SVTP ablation (top row) and the full DRAMA setting (bottom row).

Each subplot displays a histogram of actual stopping steps overlaid with a kernel density estimate (KDE), together with summary statistics: mean (μ), median, standard deviation (σ), computational saving relative to the 64-step baseline, and sample count (N). Several patterns are consistent across benchmarks:

![](images/de1b73c1fc4829757a0f9846b50dbd0b537bd0ddf3e649dd9bdea9660c5ac876.jpg)

![](images/0eefa2ea5f2cb48834137027a0799d0e8bb31197e7b7ab77c8c1b53dbd47ffde.jpg)

![](images/27fdf96af9926996bbde37f2065e7dfad707e64bff56064e0d41e54f801f275d.jpg)

![](images/63c40ff9eb9858c8925b8dab47a3db1679f30da38e7c8db9d53ddcebce8f1063.jpg)

Figure 10: Spatial redundancy analysis – SEED-Bench. SEED-Bench exhibits slightly broader spatial distributions due to its diverse scene understanding tasks, yet the dominant patch still captures ≥40% of attention.  
![](images/656e19461ef9dc93545311b6ed7fb8596843aa797b9ad7e0b5cf3890c5d405ff.jpg)  
Figure 11: Temporal redundancy analysis for all benchmarks. (a) Per-layer inter-step JSD across denoising steps. (b) Layer×step redundancy heatmap.

• DRAMA (bottom rows) shows similar stopping patterns to the w/o SVTP ablation (top rows), confirming that SVTP and SGAS operate on orthogonal axes (spatial vs. temporal) and compose independently.

• w/o SVTP FG Uni concentrates exits at the earliest steps (typically around 7——11 steps, with a later mean of 15.6 on MM-Star), reflecting its willingness to terminate as soon as both criteria are met, resulting in the highest speedup but occasionally at the cost of accuracy on harder benchmarks.

• w/o SVTP CK variants cluster around the first eligible checkpoint (step 16), as the sparse evaluation schedule prevents earlier termination regardless of output convergence.

• w/o SVTP Prog Variants consistently shifts the distribution rightward compared to uniform scheduling, reflecting its stricter early-step requirements that act as a safeguard during the structure-formation phase.

The alignment between the stopping step distributions and the JSD plateau onset provides empirical validation that SGAS terminates denoising precisely when the temporal redundancy analysis predicts the output has stabilized.

## F Additional Related Work

## F.1 Attention-Based Acceleration in Vision and Language Models

Attention-based acceleration has been explored in autoregressive vision-language models (VLMs) and generative diffusion transformers (DiTs), typically following an analyze-then-accelerate paradigm:

![](images/279073bc650efea10b2d8628f09e6c65ec817d8631419caff5b42c2cbb22acc0.jpg)  
Figure 12: DRAMA achieves an average 18.67× inference speedup over fixed-budget baselines across six benchmarks (MMMU [54], AI2D [22], MME [13], MM-Star [8], SEED-Bench [26], and MMBench [33]) while maintaining competitive accuracy

first identifying structured attention patterns, then designing targeted strategies to exploit them. In autoregressive VLMs, methods such as FlexPrefill [25] and MInference [20, 30] observe that attention is sparse and concentrated on a few key tokens, motivating attention sparsification and token pruning. In generative DiTs, DiTFastAttn [53, 55] identifies temporal similarity in attention maps across denoising steps and caches redundant computations accordingly. Despite the success of this paradigm in both domains, no systematic redundancy analysis-grounded acceleration or attention-based acceleration framework exists for multimodal dLLMs. Our work addresses this gap by dissecting attention redundancy across two orthogonal axes, spatial and temporal, and deriving dimension-specific acceleration strategies from the empirical findings.

## G Preliminary

## G.1 Diffusion Large Language Models.

Diffusion large language models (dLLMs) [38, 3] define a forward noising process and learn a reverse denoising process for generation. Multimodal dLLMs [52, 28, 51] extend this framework by conditioning on visual input: a frozen vision encoder maps the input image to $N _ { v }$ visual tokens $\mathbf { v } ,$ which, together with $N _ { c }$ instruction tokens $\mathbf { c } ,$ form a fixed prefix exempt from masking. The forward process acts exclusively on the $N _ { g }$ generation tokens $\mathbf { x } _ { \mathrm { 0 } }$ , independently masking each with probability $t \in [ 0 , 1 ]$

$$
q _ { t | 0 } ( \mathbf { x } _ { t } \mid \mathbf { x } _ { 0 } ) = \prod _ { i = 1 } ^ { N _ { g } } \mathrm { C a t } \Big ( x _ { t } ^ { i } ; ( 1 - t ) \delta _ { x _ { 0 } ^ { i } } + t \delta _ { [ \mathrm { M A S K } ] } \Big ) .\tag{14}
$$

The model $p _ { \theta } ( x _ { 0 } ^ { i } \mid \mathbf { x } _ { t } , \mathbf { v } , \mathbf { c } )$ is trained to predict the clean token at each masked position. At inference, generation proceeds from a fully masked sequence through T discrete steps; at each step, the model predicts all masked positions and selects ${ \sim } \bar { N } _ { g } / T$ to unmask based on prediction confidence, with previously revealed tokens preserved. Each step processes the concatenated sequence $\left[ \mathbf { v } ; \mathbf { c } ; \mathbf { x } _ { t } \right]$ of length $N { \ ' } = N _ { v } + N _ { c } + N _ { g }$ through L layers of bidirectional self-attention with H heads, yielding attention tensors $\{ \mathbf { A } _ { l } ^ { ( t ) } \in \mathbb { R } ^ { H \times N \times N } \} _ { l = 1 } ^ { L }$ . The total inference cost thus scales as $\mathcal { O } ( T \cdot L \cdot N ^ { 2 } )$ where $N _ { v }$ dominates the sequence length. Since v is fixed across all $T$ steps, its quadratic attention cost is replicated at every step without diminishing return.

![](images/7a6bfa5e1411cfc31070bef8d0edc493a97b13adbce19dbc4482b135aac59406.jpg)  
Figure 13: Averaged accuracy-latency Pareto frontier across all benchmarks. Each point represents a method's mean accuracy and mean latency. DRAMA variants occupy the Pareto front, achieving superior accuracy at substantially lower latency than fixed-budget baselines.

## G.2 Multimodal Diffusion Language Models.

Multimodal dLLMs [52, 28, 51] extend this framework by conditioning generation on a visual input. A frozen vision encoder maps an input image to $N _ { v }$ visual tokens $\mathbf { v } = \overline { { ( v ^ { 1 } , \ldots , v ^ { N _ { v } } ) } }$ ; together with $N _ { c }$ instruction tokens $\mathbf { c } = ( \dot { c } ^ { 1 } , \dots , \dot { c } ^ { N _ { c } } )$ , these form a fixed conditioning prefix that is never subject to masking. The forward and reverse processes act exclusively on the $N _ { g }$ generation tokens $\mathbf { x } _ { t } .$ , with the model's predictive distribution conditioned on the full context: $p _ { \theta } ( x _ { 0 } ^ { i } \mid \mathbf { x } _ { t } , \mathbf { v } , \mathbf { c } )$ . At each denoising step, the model processes the concatenated sequence [v; c; xt] of length $\overset { \vartriangle } { \boldsymbol { N } } = \boldsymbol { N } _ { v } + \boldsymbol { N } _ { c } + \boldsymbol { N } _ { g }$ through L layers of bidirectional self-attention with H heads, yielding attention tensors $\{ \mathbf { A } _ { l } ^ { ( t ) } \in$ $\mathbb { R } ^ { H \times N \times N } \big \} _ { l = 1 } ^ { L } .$ The total inference cost scales as $\mathcal { O } ( T \cdot L \cdot N ^ { 2 } )$ , where $N _ { v }$ constitutes the dominant factor in sequence length. As v is fixed across all $\dot { T }$ steps, its quadratic attention cost is replicated at every step without diminishing return, leading to a structural inefficiency that does not arise in text-only dLLMs.

![](images/dcbb371aa8b8f81c22f0fd34e9ec515cd1d94d09ee90e0aaa8149ea161cd0b92.jpg)  
Figure 14: Step distribution – AI2D. FG Uni achieves 85%/79% step saving (w/o SVTP / DRAMA) owing to the concentrated spatial attention on diagram elements.

![](images/d43acb4c0d55842732e1bd453f27a760257c6c0a80ace1bedf9ae87316591f04.jpg)  
Figure 15: Step distribution - MMBench. The diverse visual content of MMBench leads to a slightly broader stopping distribution compared to simpler benchmarks.

![](images/498dff97fdcef9a8b87dd02d6e5cb0a3626d8b5fef2a9f70ad70e85696ffa104.jpg)  
Figure 16: Step distribution – MME. Perceptual yes/no tasks converge rapidly, with most samples stopping within the first 15–20 steps.

![](images/f7a3a5e9688ded139f73516c28314d86b9cf1c715012b2be2f2a4fc947553de9.jpg)  
Figure 17: Step distribution – MMMU. Multi-discipline knowledge questions exhibit moderate step savings, with progressive scheduling providing a useful accuracy safeguard.

![](images/99dd2983ca4a348b75b3f90204c706185722ab30e5f41d935577cdf5a28f0518.jpg)  
Figure 18: Step distribution – MM-Star. Knowledge-intensive queries show stopping patterns similar to MMMU, confirming that task difficulty correlates with stopping behaviour.

![](images/b1d44aae1277af0f6958769747f855e9958efba8d7a2357ca0616e22b078a691.jpg)  
Figure 19: Step distribution - SEED-Bench. The diverse scene understanding tasks in SEED-Bench produce a broad stopping distribution, yet SGAS still achieves substantial step savings.