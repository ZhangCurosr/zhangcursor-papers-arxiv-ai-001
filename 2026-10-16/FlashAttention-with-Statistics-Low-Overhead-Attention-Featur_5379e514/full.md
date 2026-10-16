# FlashAttention with Statistics: Low-Overhead Attention Features for Compact Hallucination Detectors

Andrei Volodichev<sup>1</sup>, Maksim Savichev<sup>2</sup>, Nikita Sviridov<sup>3</sup>, Artem Yuliy<sup>3</sup>, Dmitry Simakov<sup>3</sup>, Alexey Zaytsev<sup>1,</sup> <sup>3</sup> <sup>1</sup>Applied AI Institute, <sup>2</sup>Yandex, <sup>3</sup>Risk AI Research Lab Correspondence: a.volodichev@applied-ai.ru

## Abstract

Large language models can produce fluent answers that the supplied context does not support. A compact frozen proxy model can detect such answers by reading the context and another model’s response, even from a proprietary generator. Attention-based detectors use summary statistics of the proxy’s attention weights, but standard pipelines materialize the attention matrix, whose memory grows quadratically with sequence length. We compute these statistics without materializing attention, by accumulation in the FlashAttention pass, sparse lookups after it, or a second tiled pass over the scores, and classify them by this path into nearly free and costlier ones. On an H100, the statistics of five published detectors add 16% to the latency of a Qwen3-8B forward at 2K tokens, falling to 1.4% at 35K, and store 22.5 KiB per response token. On three benchmarks, probes trained on frozen Qwen3 proxies of 0.6B to 8B parameters prove strong detectors: a 4B proxy outperforms the strongest of nine 4–9B LLM judges in F1 on two of the three, the proxies match or outperform specialized grounding checkers of up to 7B parameters, probing each proxy beats prompting it as a judge, and adding attention features significantly improves F1 in half of the 12 benchmark–proxy pairs, never significantly lowering it.

## 1 Introduction

Contextual hallucination detection asks whether an answer is supported by the supplied information. The detector need not be the model that generated the answer: the generator may expose neither its weights nor its attention, while a separate detector can be much smaller and cheaper. A compact proxy instead reads the context and completed answer and uses its own internal signals to assess context reliance. Prior work has used surrogate probabilities to estimate black-box confidence (Shrivastava et al., 2023), auxiliary models to calibrate generators (Ulmer et al., 2024; Kapoor et al., 2024; Hayes et al.,

![](images/77e080154e0f619c382e3a4da5faf09d0059bff725593d95b821f102eff68552.jpg)  
Figure 1: Cost of extracting all implemented attention statistics for hallucination detection in every layer of Qwen3-8B and Qwen3-0.6B using our kernels. Top: model forward-pass latency. Bottom: relative overhead compared with the same forward using standard scaled dot-product attention (SDPA), without statistic extraction.

2026), and hidden states or attention from reader models to detect hallucinations in another model’s answers (CH-Wang et al., 2024; Han et al., 2025; Yeh et al., 2026; Singh et al., 2026).

Attention-based detectors need only summary statistics of attention weights, such as prompt mass (Chuang et al., 2024), diagonal or previous-token weights (Sriramanan et al., 2024; Vazhentsev et al., 2026), sink rates (Binkowski et al., 2026), or graph features (Bazarova et al., 2026). Yet standard pipelines materialize the full attention matrix. For a 32-head layer, it occupies about a gigabyte at 4K tokens and tens of gigabytes at 16K–32K, although the detector keeps only a linear number of statistics.

FlashAttention computes exact attention in tiles without storing this matrix (Dao et al., 2022; Dao, 2024). We compute detector statistics<sup>1</sup> during or right after that pass so they inherit its linear memory and most of its speed. We classify attention statistics, including those of published detectors, by how our implementation computes them exactly without the matrix: accumulation inside the pass, a fixed number of post-pass lookups, a second pass over the scores, or dense materialization (Figure 2). Our Triton implementation (Tillet et al., 2019) extends the FlashAttention-2 forward to compute the statistics of five published detectors: Lookback Lens (Chuang et al., 2024), LLM-Check (Sriramanan et al., 2024), RAUQ (Vazhentsev et al., 2026), IN-TRYGUE (Kuleshova et al., 2026), and SinkProbe (Binkowski et al., 2026). Extracting them from every head and layer adds 16% to a Qwen3-8B forward at 2K tokens and 1.4% at 35K, and stores 22.5 KiB per response token (Figures 1 and 6).

![](images/5e6e9635db9743a90aa068f49e10e9e07e1d697c48dfcdb8a64243e6a10c610b.jpg)  
Figure 2: Attention with statistics. Blue: standard FlashAttention (Dao et al., 2022; Dao, 2024); green: our additions. (a) We compute statistics for response rows only; prompt rows can go through any production kernel (§2.1). s : scores of query tile I against key tile J. (b) One tile of the pass: the exponentials that FlashAttention already computes also feed the T1 statistics, which are finalized after the last tile; the pass also stores the log-sum-exp (LSE) that T2 and T3 reuse (§2.2). (c) The four tiers, i.e. computation paths of a statistic, ordered by the extra work they need: none (T1, fused), a few positions per row (T2, lookup), a second pass over all scores (T3, rescore), or the whole stored map (T4, dense) (§3).

Our contributions are threefold. (i) We develop a taxonomy of attention statistics based on how they can be computed exactly without materializing attention maps, identifying which come almost for free with FlashAttention and which require a second pass. This taxonomy also guides the design of new inexpensive statistics. (ii) We implement these computation paths in Triton kernels and provide an attention backend, which we release publicly, that integrates them into existing model inference pipelines. Experiments evaluate their overhead and detection performance. (iii) We show that frozen Qwen3 proxies (0.6B–8B) can detect hallucinations in larger model’s answers across three benchmarks. A 4B proxy outperforms the strongest of nine LLM judges (4B–9B) on two benchmarks, and even the smallest, 0.6B proxy beats it on the third and trails it by at most 0.03 F1 on the other two. Adding attention features significantly improves F1 in 6 of the 12 dataset–proxy pairs and never significantly lowers it, emphasizing the usefulness of attention statistics.

## 2 Background

## 2.1 Setting and notation

We consider one head of one layer and drop both indices. Let Q, K, $V \in \mathbb { R } ^ { L \times d }$ be its query, key, and value projections after positional encoding; under grouped-query attention, the heads of a group share K and V . Positions are indexed from 0. For a query row i, let J<sub>i</sub> be the keys it may attend to after causal and padding masking, and define

$$
\begin{array} { r l r } & { s _ { i j } = \tau q _ { i } ^ { \top } k _ { j } , } & { \mathrm { L S E } _ { i } = \log \displaystyle \sum _ { j \in \mathcal { I } _ { i } } e ^ { s _ { i j } } , } \\ & { p _ { i j } = e ^ { s _ { i j } - \mathrm { L S E } _ { i } } , } & { o _ { i } = \displaystyle \sum _ { j \in \mathcal { I } _ { i } } p _ { i j } v _ { j } , } \end{array}\tag{1}
$$

where $s _ { i j }$ are the attention scores, $p _ { i j }$ the attention weights, and $\tau = 1 / \sqrt { d }$ unless the model sets another scale. A statistic is a per-head quantity that a kernel returns: a row reduction of one row $p _ { i } .$ such as its entropy, or a column reduction of one column $p . _ { j }$ over a set of rows. A kernel computes each statistic through one computation path, or tier (Section 3.1). Afeature is a classifier input that the host derives from statistics, for example by averaging over rows (Appendix B), and a detector recipe selects features, heads, and layers for a classifier.

When the response is known in advance, as in teacher-forced detection, the input is a prompt of $P$ tokens followed by $R = L - P$ response tokens at positions $\mathcal { R } = \{ P , . . . , L - 1 \}$ . The detectors we target read statistics on these response rows. Under the causal mask, a prompt row $i < P$ sees only prompt keys, so its output does not depend on the response and any production kernel can compute it. The response stage evaluates $Q _ { \mathcal { R } }$ against all of K and V with a causal mask aligned to the bottomright corner, as in key–value-cache append kernels (Ye et al., 2025) (Figure 2a). We call its $R \times L$ block of scores and weights the response block; without padding, it has $N _ { \mathscr { R } } = R P + R ( R + 1 ) / 2$ unmasked scores per head.

## 2.2 Online softmax in FlashAttention

FlashAttention computes $o _ { i }$ in tiles without storing the weights (Dao et al., 2022; Dao, 2024). The FlashAttention-2 forward pass is query-major: a tile I of query rows stays resident while the key tiles J it sees stream past (Figure 2b). Its backward pass is key-major: a key tile stays resident while query tiles stream past. We call one evaluation of every score of the response block, in tiles, a pass; the pass is the query-major response stage.

For one row, online softmax carries a running maximum m, a normalizer ℓ, and an unnormalized output O (Milakov and Gimelshein, 2018; Rabe and Staats, 2021). For a new key tile $J ,$ let $m ^ { \prime } = \operatorname* { m a x } ( m , \operatorname* { m a x } _ { j \in J } s _ { j } ) , a = e ^ { m - m ^ { \prime } }$ , and $\tilde { p } _ { j } = e ^ { s _ { j } - m ^ { \prime } }$ . The update is

<table><tr><td>Tier</td><td>Scores Stat. ops</td><td></td><td>Memory</td></tr><tr><td>T1 (fused) entropy, prompt mass, effective support, max weight</td><td>none</td><td> $O ( N _ { s } N _ { \mathcal { R } } )$ </td><td>NsR</td></tr><tr><td>T2 (lookup) log pii, pi,i−1, BOS weight</td><td>bR</td><td>O(bR)</td><td>bR</td></tr><tr><td>T3 (rescore) column sums, sink rates, threshold counts</td><td> $N _ { \mathcal { R } }$ </td><td>O(NR)</td><td> $O ( R + L )$ </td></tr><tr><td>T4 (dense) MTopDiv, stored maps</td><td> $N _ { \mathcal { R } }$ </td><td>Ω(NR)</td><td> $N _ { \mathcal { R } }$ </td></tr></table>

Table 1: Work per head beyond the pass over the response block, by tier. Scores: extra attention scores evaluated, each costing d multiply-adds and one exponential. Stat. ops: statistic-specific elementwise operations, plus the statistic’s own algorithm for T4. Memory: numbers written per head besides the attention output and LSE.

$$
\begin{array} { l } { { \ell \gets a \ell + \displaystyle \sum _ { j \in J } \tilde { p } _ { j } , } } \\ { { \ O \gets a O + \displaystyle \sum _ { j \in J } \tilde { p } _ { j } v _ { j } , \qquad m \gets m ^ { \prime } . } } \end{array}\tag{2}
$$

After the last tile, $o = O / \ell$ and $\mathrm { L S E } = m +$ log ℓ. As FlashAttention stores the LSE, a later kernel can recover any weight (Equation 1) by re-evaluating its score.

## 3 Computing Attention Statistics without Materialization

## 3.1 Computation paths

We compute statistics only in the response stage (Section 2.1, Figure 2a). A statistic either fits a bounded state that the pass can update, or it needs final weights, which a later kernel recovers from a re-evaluated score and the stored LSE (Section 2.2). We sort statistics into four tiers by the scores evaluated beyond the pass (Table 1, Figure 2c): T1 (fused) evaluates none, T2 (lookup) a fixed number b per row, T3 (rescore) every score a second time, and T4 (dense) materializes the response block. A tier is the path our implementation uses, with Triton kernels for T1–T3 and eager PyTorch for T4, so it bounds the cost from above: the pass could also compute the T2 lookups (Section 3.3), and T1 holds only the bounded-state reductions we implement. The LSE, which every FlashAttention forward writes, belongs to no tier. Appendix B defines every statistic we compute. Appendix D gives the pseudocode and kernel details.

## 3.2 T1 (fused): statistics in online softmax

T1 (fused) statistics are accumulated inside the pass and need no extra scores. Consider prompt mass $\sum _ { j < P } p _ { i j }$ , the share of attention that row i puts on the prompt, from which Lookback Lens builds its features (Chuang et al., 2024). During the pass, row i knows each weight only up to the factor $e ^ { m _ { i } - \mathrm { L S E } _ { i } }$ , which is fixed only after its last key tile. Prompt mass can nevertheless be accumulated like ℓ: keep $\begin{array} { r } { A = \sum _ { i < P } e ^ { s _ { j } - m } } \end{array}$ over the keys seen so far, update it as $\begin{array} { r } { \dot { A } \gets a A + \sum _ { j \in J , j < P } \tilde { p } _ { j } } \end{array}$ next to Equation 2, and output $A / \ell$ after the last tile.

The same works for integer powers of the weights times any function of the score. For an integer $\alpha _ { u } \geq 1$ and a fixed-dimensional function $g _ { u }$ that does not depend on m, ℓ, or the LSE, keep the state $A _ { u }$ and update it for each key tile:

$$
A _ { u } = \sum _ { j { \mathrm { ~ s e e n } } } e ^ { \alpha _ { u } ( s _ { j } - m ) } g _ { u } ( i , j , s _ { j } ) ,\tag{3}
$$

$$
A _ { u }  a ^ { \alpha _ { u } } A _ { u } + \sum _ { j \in J } \tilde { p } _ { j } ^ { \alpha _ { u } } g _ { u } ( i , j , s _ { j } ) .\tag{4}
$$

Since $\alpha _ { u }$ is an integer, $\tilde { p } _ { j } ^ { \alpha _ { u } }$ is a product of exponentials the pass already computes. This standard consequence of online softmax defines T1:

Lemma 1 (Separable row reductions) For $N _ { s }$ statistics, let $g _ { 1 } , \ldots , g _ { N _ { s } }$ be functions as above and $\alpha _ { 1 } , \ldots , \alpha _ { N _ { s } } \geq 1$ integers. The sums

$$
F _ { i } ^ { ( u ) } = \sum _ { j \in \mathcal { I } _ { i } } p _ { i j } ^ { \alpha _ { u } } g _ { u } ( i , j , s _ { i j } ) ,\tag{5}
$$

and any fixed function of $F _ { i } ^ { ( 1 ) } , \dots , F _ { i } ^ { ( N _ { s } ) }$ and $\operatorname { L S E } _ { i } ,$ can be computed in the pass, with no additional exponentials or scores and $\textstyle \sum { u }$ dim $g _ { u }$ numbers of state per row, independent of L.

Prompt mass is the case $\alpha = 1 , g = \mathbf { 1 } [ j < P ]$ entropy is $H = \mathrm { L S E } - F$ with α = 1 and g = s; effective support is $1 / F$ with α = 2 and $g =$ 1; and the maximum weight is $1 / \ell .$ . The kernel updates these accumulators per score tile next to the FlashAttention state, keeping nothing per key (Figure 2b). Appendix C gives the proof and the accumulator’s merge and stability properties.

## 3.3 T2–T4: post-pass statistics

T2 (lookup). Some statistics read a fixed number b of weights per response row at positions known in advance, optionally summed over rows: the logdiagonal log $p _ { i i } = s _ { i i } - \mathrm { L S E } _ { i }$ , the previous-token weight $p _ { i , i - 1 }$ , and the weight on a fixed sink column such as the beginning-of-sequence (BOS) token (Sriramanan et al., 2024; Vazhentsev et al., 2026; Xiao et al., 2024). The pass could keep these b scores per row and finalize them with the LSE, which would place them in T1. We instead compute them in a separate kernel that reads only $Q ,$ K, and the LSE, so it works after any forward kernel that stores the LSE and takes its positions at run time (Appendix D). Its bR extra scores grow linearly with the sequence length.

T3 (rescore). Column reductions, such as column entropy (not implemented) and the column sums $\begin{array} { r } { c _ { j } = \sum _ { i \in \mathcal { R } } p _ { i j } } \end{array}$ from which sink rates are read (Gu et al., 2025; Binkowski et al., 2026), cannot be finalized by the query-major pass from bounded state per query row. Row i contributes $e ^ { s _ { i j } - m _ { i } }$ to column j, and the factor that turns this into $p _ { i j }$ is known only after the row’s last key tile and differs between rows (Section 3.2), so a column sum needs each score again once its row’s LSE is final, or all scores stored. The same holds for row reductions that compare each weight with a function of $\mathrm { L S E } _ { i }$ , such as counts of $p _ { i j } ~ > ~ \theta$ (not implemented). A second kernel re-evaluates each score, converts it with the stored LSE, and reuses the key-major loop of the FlashAttention-2 backward pass (Appendix D). It evaluates all the exponentials of the response stage, which run on slow non-tensor-core units (Shah et al., 2024), but only half its multiply-adds (no product with V ), so it costs close to a second response stage (Figure 3a).

T4 (dense). Statistics that are not bounded-state reductions of the weights, such as the attentiongraph divergence MTopDiv (Bazarova et al., 2026) and stored attention maps, need the materialized response block, with $O ( N _ { \mathcal { R } } )$ memory; we have no T4 kernel.

## 4 Kernel Benchmark

We evaluate the kernels on their own and ask three questions. (Q1) Are the fused statistics numerically at least as accurate as those from materialized attention? (Q2) What does each tier add on top of attention itself, and how does that compare with production kernels? (Q3) How much does a wholemodel forward slow down when it computes the statistics underlying several published detectors with our kernels, and how does that compare with computing them from materialized attention?

(a) overhead of cumulative configurations at $L = 3 2 K$  
![](images/ad06b1b213acc3b8c95340dddb95bf993314aaa0356bf8f29b234d5a40a64771.jpg)

![](images/d50ca072bf9e3a2bc45f99dfc7294c268ef7fd1305dbb121c758ff67040ffc9c.jpg)

![](images/0da6473dba1cf1ff02c29b8bebe4fdf78d21e2199d633ca262ac91b926843f9a.jpg)  
≤ T1 (fused) vs. Base ≤ T2 (+ lookup) vs. Base ≤ T3 (+ rescore) vs. Base Base vs. full SDPA cuDNN Base vs. full SDPA FA2

(b) latency, d = 128, r= 0.3  
![](images/dfb7fb75c069e7a62ea38e2210511adbdec2a0525f5090d1a82397fda3ddcfaf.jpg)

(c) relative to full SDPA cuDNN, L ≥ 4K  
![](images/3f79785fd2e8af87193240a43190a7b5587d92290e43fe5f57ce7f479a280d99.jpg)  
Figure 3: What each benchmark configuration adds to one attention layer (Q2). Configuration ${ \le } \mathrm { T k }$ enables the paths up to $\mathrm { T } k ; \leq \mathrm { T } 3$ reads the BOS column sum from the column sums instead of the lookup. (a) Overhead of each configuration over our statistics-off Base layer at L=32K (whiskers: 16K–32K), and of Base relative to SDPA on the whole sequence (diamonds). (b) Latency from 1K to 100K tokens at d = 128, $r = 0 . 3$ . (c) Overhead relative to SDPA cuDNN on the whole sequence, at $r = 0 . 3$ (solid) and $r = 0 . 1$ (dashed).

Setup. All runs use a single NVIDIA H100 NVL in bf16. Each of our configurations is a whole attention layer: prompt rows go to PyTorch’s SDPA and response rows to our Triton kernels. Base is the same layer with statistics off. We compare both with SDPA on the whole sequence (“full SDPA”) with the FlashAttention-2 and cuDNN backends. See Appendix E for software versions and the timing protocol, and Appendix I for memory.

Benchmark configurations. We measure the tiers in cumulative configurations: ≤Tk enables the paths up to Tk on top of Base, which writes only the output and the LSE. ≤T1 adds the T1 row statistics of Table 1, ≤T2 the lookups of log p<sub>ii</sub>, $p _ { i , i - 1 }$ , and the BOS weight $p _ { i s } ,$ and ${ \le } \mathrm { T } 3$ the column sums, which replace the BOS lookup by the BOS column sum $c _ { s } , { \bf s o \le T } 3$ does not return the per-row $p _ { i s }$ . All configurations run on the $R = r L$ response rows, so their cost grows with r.

Numerical accuracy (Q1). We measure the absolute error on response rows against an fp64 reference, for materialized attention and for our kernels, on random inputs and on post-RoPE Q, K, V captured from Qwen3-8B, from 2K to 32K tokens. Finding. Our statistics are three to four orders of magnitude more accurate for every statistic (Table 4 in Appendix F), because our kernels reduce fp32 probabilities in registers, whereas materialized attention first rounds each to bf16.

Marginal cost per tier (Q2). We time one attention layer at batch 1 on synthetic inputs of $L \in \{ 4 \mathrm { K } , 1 6 \mathrm { K } , 3 2 \mathrm { K } \}$ tokens with response fraction $r ~ = ~ R / L ~ \in ~ \{ 0 . 1 , 0 . 3 , 0 . 5 \}$ , in Qwen3-8B shapes (32 query heads, 8 KV heads, $d = 1 2 8 )$ and at $d = 6 4$ and 256. ≤T2 adds at most 5 percentage points of compute time over ≤T1, and both add 1–30% over Base (Figure 3a; Table 5 in Appendix G). The overhead generally increases with r and decreases with $d ,$ both trends explained by T1: the former reflects the $R = r L$ response rows processed, while the latter reflects matrix computation scaling with d versus T1’s elementwise work. ≤T3 is the costliest configuration, at 16–87%, and shows the same scaling in r as ≤T1; in $d ,$ it drops from d = 64 to d = 128 but not further, because T3’s second pass adds its own matrix work.

Our kernels scale like the production ones all the way to 100K tokens (Figure 3b–c), with no wall expected further: at 100K tokens and r = 0.3, a layer with all row statistics takes 387 ms, against 351 ms for cuDNN and 523 ms for FA2 without statistics. Below about 4K tokens, our latency stops falling because a fixed CPU overhead per call dominates (Appendix G). In memory, the statistics take 7.6– 20 MiB of a layer at 32K (Appendix I). Finding. T1 and T2 are cheap enough to compute together by default; T3 is worthwhile only when a detector needs reductions over all final weights.

Ours: samples, median Dense: samples, median  
![](images/26a9e519abf3ed4dedaa2cef0f3741f1661054c14e7b3d24065d5a6854701b00.jpg)  
Figure 4: Added latency and peak memory over the forward without features, per sample, by prompt length (Q3). Lines: binned medians. The shortest prompts are RAGTruth QA, the others PHANTOM.

Comparison with prior methods (Q3). We compute the attention statistics underlying Lookback Lens (T1) (Chuang et al., 2024), LLM-Check (T2) (Sriramanan et al., 2024), RAUQ (T2) (Vazhentsev et al., 2026), INTRYGUE (T3) (Kuleshova et al., 2026), and SinkProbe (T3) (Binkowski et al., 2026), with the other T1 statistics, for all 32 heads of all 36 layers of Qwen3-8B, in one teacher-forced forward at batch 1. Where a method pools over all tokens, as LLM-Check and SinkProbe do, our responserow version is an adaptation (Appendix B). The floor is the same forward without statistics. We stop at feature extraction; the classifiers cost little and the same for every path. See Appendix H for the SDPA backends and the dense baseline.

Data. From RAGTruth (Niu et al., 2024), we take the subset of QA responses generated by Llama-2-70B-chat, and from PHANTOM (Ji et al., 2025), the 2,000-token split of the 10-K filings set (Appendix H). Inputs are short here because dense pipelines do not scale much further.

Results. Figure 4 shows the cost of every sample, grouped by prompt length. At these lengths, both paths add about the same latency for every prompt length, but dense latency starts to grow once a response exceeds about 100 tokens, and faster for longer prompts: on the sample with the most response weights, 785 response tokens in 7.2K, dense adds 501 ms to the forward and ours 53 ms. Finding. The clear difference is memory: dense memory grows with RL, by 2.1 GiB on that sample, whereas ours grows only with R, by 23 MiB, so the gap widens with sequence length.

## 5 Proxy Detection

Section 4 showed that attention statistics are cheap to extract; here we ask whether a compact proxy detector should extract them. Every proxy forward already yields hidden states and logits at no extra cost, and probes on hidden states are strong detectors on their own (Azaria and Mitchell, 2023; Chen et al., 2024), so attention statistics earn their place only if they add to these signals. We ask four questions. (Q4) Do attention features improve a detector built on hidden states and logits? (Q5) Which attention representation helps: a few light statistics pooled over selected heads, from T1 and T2, or heavier sink statistics of every head, which need T3? (Q6) How does detection scale with proxy size? (Q7) How do proxy detectors compare with LLM judges? Section 5.5 then gives the cost of the features in the detection pipeline, and Appendix M tests whether published attention detectors, computed from one forward of the 8B proxy, transfer across the six domains of HaluBench.

## 5.1 Task, data, and metrics

A detector reads a context, a question, and an existing answer and predicts whether the context supports the answer. A frozen Qwen3 proxy (Yang et al., 2025) of 0.6B, 1.7B, 4B, or 8B parameters reads each example in one teacher-forced forward, so it need not be the model that wrote the answer.

We use RAGTruth QA (Niu et al., 2024), HaluBench (Ravi et al., 2024), and TRIVIA+ (Chen et al., 2026b), split into training, validation, and test so that groups of related examples never cross splits. We filter and split HaluBench ourselves, because on its unfiltered answers a TF–IDF classifier given only the answer reaches 0.84 AU-ROC. Appendix K gives the splits, the filtering, and text-only baselines.

<table><tr><td></td><td></td><td colspan="4">RAGTruth QA</td><td colspan="4">HaluBench</td><td colspan="4">TRIVIA+</td></tr><tr><td>Features</td><td>N</td><td>0.6B</td><td>1.7B</td><td>4B</td><td>8B</td><td>0.6B</td><td>1.7B</td><td>4B</td><td>8B</td><td>0.6B</td><td>1.7B</td><td>4B</td><td>8B</td></tr><tr><td> ${ \bf A } _ { 7 }$ </td><td>7</td><td>.573</td><td>.567</td><td>.503</td><td>.550</td><td>.615</td><td>.639</td><td>.631</td><td>.637</td><td>.531</td><td>.524</td><td>.545</td><td>.527</td></tr><tr><td> $\mathrm { { L _ { 9 } } }$ </td><td>9</td><td>.440</td><td>.444</td><td>.467</td><td>.433</td><td>.633</td><td>.627</td><td>.642</td><td>.665</td><td>.545</td><td>.523</td><td>.518</td><td>.576</td></tr><tr><td> ${ \bf A } _ { 7 } { \bf L } _ { 9 }$ </td><td>16</td><td>.566</td><td>.532</td><td>.541</td><td>.546</td><td>.642</td><td>.643</td><td>.656</td><td>.675</td><td>.555</td><td>.560</td><td>.543</td><td>.578</td></tr><tr><td> $\mathrm { H _ { 6 4 } }$ </td><td>64</td><td>.557</td><td>.559</td><td>.566</td><td>.557</td><td>.697</td><td>.734</td><td>.742</td><td>.742</td><td>.627</td><td>.638</td><td>.634</td><td>.670</td></tr><tr><td> $\mathrm { \bf A } _ { 7 } \mathrm { H } _ { 6 4 }$ </td><td>71</td><td>.637</td><td>.581</td><td>.547</td><td>.600</td><td>.698</td><td>.729</td><td>.744</td><td>.746</td><td>.613</td><td>.638</td><td>.639</td><td>.645</td></tr><tr><td> $\mathrm { H _ { 6 4 } L _ { 9 } }$ </td><td>73</td><td>.614</td><td>.559</td><td>.543</td><td>.533</td><td>.697</td><td>.736</td><td>.749</td><td>.754</td><td>.599</td><td>.641</td><td>.641</td><td>.647</td></tr><tr><td> $\mathrm { A _ { 7 } H _ { 6 4 } L _ { 9 } }$ </td><td>80</td><td>.621</td><td>.563</td><td>.528</td><td>.563</td><td>.702</td><td>.726</td><td>.755</td><td>.756</td><td>.641</td><td>.643</td><td>.646</td><td>.667</td></tr><tr><td> $\mathbf { A } _ { 6 4 }$ </td><td>64</td><td>.593</td><td>.577</td><td>.583</td><td>.592</td><td>.727</td><td>.718</td><td>.769</td><td>.768</td><td>.631</td><td>.634</td><td>.616</td><td>.656</td></tr><tr><td> $\mathrm { { A _ { 6 4 } H _ { 6 4 } } }$ </td><td>128</td><td>.601</td><td>.529</td><td>.566</td><td>.594</td><td>.724</td><td>.746</td><td>.776</td><td>.791</td><td>.636</td><td>.640</td><td>.674</td><td>.671</td></tr><tr><td>Best LLM judge</td><td></td><td></td><td>.623</td><td></td><td></td><td></td><td>.755</td><td></td><td></td><td></td><td>.655</td><td></td><td></td></tr></table>

Table 2: Test F1 by feature set and Qwen3 proxy size. $\mathbf { A } _ { 7 } { \mathrm { : } }$ cheap attention features; $\mathbf { A } _ { 6 4 } \colon$ all-head sink values; $_ \mathrm { H } \mathrm { _ { 6 4 } } \mathrm { : }$ compact hidden states; $\operatorname { L } _ { 9 } { \mathrm { : } }$ logit features; combinations concatenate blocks (Section 5.2). N: number of classifier inputs. Bold: best in column; underlined: second best. Bottom: the best of nine LLM judges of 4–9B parameters, Qwen3.5-9B on every dataset (Table 13).

Table 2 reports test F1 with validation-selected threshold, because a deployed detector must label each answer; AUROC is in Table 10 of Appendix L.

## 5.2 Feature blocks

One proxy forward yields three feature blocks: hidden states and logits from the forward pass, and attention features added by our kernels in two variants, ${ \bf A } _ { 7 }$ and $\mathbf { A } _ { 6 4 }$ . Feature sets concatenate one or more blocks $\left( \mathbf { e . g . , A . 7 H _ { 6 4 } } \right)$ . We select the ${ \bf A } _ { 7 }$ and $\mathrm { L _ { 9 } }$ features and the $\mathrm { H } _ { 6 4 }$ pooling and projection recipe once, using the highest mean AUROC on the TRIVIA+ and reuse it across all experiments with the details of the protocol in Appendix L. Each per-token quantity, i.e., an attention-head statistic, a logit statistic, or a hidden-state coordinate, is pooled over answer tokens by its mean, standard deviation, minimum, and maximum, producing distinct ppoled features.

Cheap attention features $\mathbf { \left( A _ { 7 } \right) }$ use the K strongest induction heads, which attend to earlier input tokens (Olsson et al., 2022). Each head yields 22 candidate features, averaged across the $K$ heads: four pooled features for each of five per-token statistics, the T2 BOS sink rate, and the log-diagonal. Feature selection retains the seven in bold: maximum weight (mean, std, min, max), effective support (mean, std, min, max), entropy (mean, std, min, max), the lookback ratio built from prompt mass and the previous-token weight (mean, std, min, max), the BOS sink rate, and the log-diagonal.

All-head sink values $\mathbf { \left( A _ { 6 4 } \right) }$ are the richest attention features our kernels provide. Following SinkProbe (Binkowski et al., 2026), we keep, for every head of every layer, its k largest sink values from T3, the normalized column sums $\bar { c } _ { j } ( \mathsf { A p - }$ pendix B), concatenate them, and project them onto 64 principal components, matching the size of $\mathrm { H } _ { 6 4 }$ without selection.

Compact hidden states $\mathbf { ( H _ { 6 4 } ) }$ are the standard probe input. Following prior work (Azaria and Mitchell, 2023; Chen et al., 2024), we use one middle layer without selection. The selection keeps the stacked mean and standard deviation of its coordinates, which are standardized and projected onto 64 principal components (Appendix L.1).

Logit features $\mathbf { ( L _ { 9 } ) }$ are nine of 224 candidates from seven per-token statistics of the next-token distribution, e.g., the answer token’s minimum probability, kept by recursive feature elimination (Appendix L.2).

## 5.3 Selection and training per dataset and proxy

Each dataset–proxy pair refits the scalers and projections on its training split and selects hyperparameters without test labels. For $\mathbf { A } _ { 7 } ,$ , the number of heads $K \in \{ 1 , 2 , 4 , 8 , 1 6 , 3 2 \}$ is chosen by the validation AUROC of ${ \bf A } _ { 7 }$ alone and reused in every set that contains ${ \bf A } _ { 7 }$ . For $\mathbf { A } _ { 6 4 }$ , the number of sink values per head $k \in \{ 1 , \ldots , 1 0 \}$ is chosen for each pair by grouped five-fold cross-validation on its training split and reused in $\mathrm { { A _ { 6 4 } H _ { 6 4 } } }$ . Every feature set serves as an input to a class-balanced $L _ { 2 }$ regularized logistic regression with regularization strength chosen by validation AUROC.

## 5.4 Main results

Attention features (Q4). In every one of the 12 dataset–proxy pairs, the feature set with the best F1 contains $\mathbf { A } _ { 7 }$ or $\mathbf { A } _ { 6 4 }$ (Table 2), and so does the set with the best AUROC (Table 10). With only 150–546 test groups, uncertainty in F1 differences remains substantial. After Holm correction, adding $A _ { 7 }$ significantly improves F1 for all four proxies on RAGTruth QA when combined with logits, and for two proxies when combined with hidden states (Table 11). Adding ${ \cal A } _ { 6 4 }$ to hidden states yields significant improvements for the 4B and 8B proxies on HaluBench (Table 12). Thus, overall attention features provide improvements if compared to variants without them.

Which attention features (Q5). Alone, $\mathbf { A } _ { 6 4 }$ beats ${ \bf A } _ { 7 }$ in all 12 pairs in both F1 and AUROC, which supports the heavier T3 path when attention is the main signal, although $\mathbf { A } _ { 6 4 }$ also has more inputs and covers every head. With hidden states, $\mathrm { \bf A } _ { 6 4 } \mathrm { H } _ { 6 4 }$ wins on HaluBench and TRIVIA+ with every proxy, while on RAGTruth QA $\mathrm { \bf A } _ { 7 } \mathrm { H } _ { 6 4 }$ wins with three of four. The cheap features thus remain useful on some datasets alongside T3 sinks.

Proxy size (Q6). The benefit of larger proxies depends on the benchmark and feature representation. On HaluBench and TRIVIA+, larger models improve detectors using hidden states, logits, or all-head sink features. On RAGTruth QA, even the smallest proxy remains competitive. These results favour selecting proxy size and features jointly on validation data with small 0.6B proxy showing strong performance in multiple settings.

LLM judges (Q7). Table 13 compares our proxy detectors with nine instruction-tuned LLM judges of 4–9B parameters (Zheng et al., 2023); evaluation details are in Appendix L.5. Against Qwen3.5-9B, the strongest evaluated judge, a 4B proxy achieves higher observed F1 on HaluBench and TRIVIA+, and a 0.6B proxy does so on RAGTruth QA. Our probes are also competitive with the specialized grounding checkers MiniCheck-7B (Tang et al., 2024) and HHEM-2.1 (Li et al., 2024a). The most consistent advantage is within the same backbone: $\mathrm { { A _ { 6 4 } H _ { 6 4 } } }$ achieves higher F1 and AUROC than prompting the corresponding Qwen3 model in all 12 dataset–proxy pairs. When task-specific training data are available, supervised probing therefore offers an effective way to build compact detectors that outperform prompting the same models and compete with larger judges.

## 5.5 Cost of feature extraction

A proxy detector adds a linear classifier to one proxy forward, so it costs one forward with features. Figure 1 times forwards of the largest and smallest proxies, Qwen3-8B and Qwen3-0.6B, with all kernel statistics for every head and layer, against the same forwards without features (Appendix H). From 2K to 35K tokens, the overhead falls from 16% to 1.4% for Qwen3-8B and from about 170% to 6% for Qwen3-0.6B, whose whole forward at 2K still takes under 100 ms. On shorter inputs, Pythonside CPU overhead sets the floor. In peak memory above the model weights (Figure 6), the features add about 28 KiB per response token for Qwen3- 8B and 11 KiB for Qwen3-0.6B, mostly the stored statistics (22.5 and 8.75 KiB; Appendix I).

## 6 Conclusion

Attention statistics are an active basis for hallucination detection (Chuang et al., 2024; Binkowski et al., 2026; Kuleshova et al., 2026). We confirmed that many those statistics can be extracted without materializing attention matrices. Our paper classifies these statistics by how they are computed, clarifying which can reuse the attention pass and which require additional lookups or a second pass over the scores. Our FlashAttention-compatible backend implements these paths with memory linear in sequence length. In our H100 NVL benchmarks at batch size one, extracting the statistics underlying five detectors adds just 16% to Qwen3-8B forward latency at 2K tokens and 1.4% at 35K. This implementation supports attention features used by existing detectors and their use with small, frozen proxy models. Our experiments show that supervised detectors built on these proxies can compete with larger prompted judges. Reducing Python overhead and synchronization with the GPU would further improve extraction speed on short inputs.

## Limitations

We do not claim that our tiers list every attention statistic that can be computed efficiently. A tier is the path our implementation uses (Section 3.1): a statistic in T4 may admit an efficient exact kernel that we did not write, and we do not claim that our algorithms are optimal.

We write our kernels in Triton, which is portable across GPUs but can be slower than specialized CUDA kernels: our response-stage kernel is faster than FA2 on Hopper but slower than cuDNN. Nothing prevents computing the same statistics inside such a kernel. Our Python wrapper also adds a fixed host cost of 47–60 ms per forward. Little of it is GPU work, which grows by only 5 ms on short Qwen3-8B inputs (Appendix H), so a compiled host implementation with CUDA graphs could remove most of it.

Our implementation is not yet a production kernel. It computes the statistics of a complete response in one call. Computing them token by token during generation would need a split-KV decoding kernel, which is possible because the T1 accumulators merge associatively, but which we have not implemented. It supports neither sliding-window nor soft-capped attention, takes only a padding mask, and covers head dimensions 64, 128, and 256, so it does not yet serve an arbitrary model. Our wrapper also runs the prompt stage in the same layer call and copies both stage outputs into one tensor. This copy does not raise the peak of a whole forward (Appendix I) and is avoidable: since our statistics cover response rows only, the prompt can go through a separate prefill whose key–value cache the response stage reads, as serving systems do (Kwon et al., 2023).

The proxy requires an auxiliary forward pass and labeled probe-training examples. A small proxy may miss evidence available to a larger generator. We assess support relative to supplied context, not general factual truth. Scores require separate calibration before interpretation as error probabilities. Dataset artifacts, label disagreements, truncation, and domain shift can affect results.

## Ethical Considerations

Our work aims to make hallucination detection cheap enough to check every response of a retrievalaugmented system. Like any detector, it can miss unsupported answers or flag supported ones, so in high-stakes settings it should complement human review rather than replace it. All experiments use public benchmarks under their terms of use; we collect no new data and involve no human subjects.

## References

Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, et al. 2025. Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs. arXiv preprint arXiv:2503.01743.

Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Alexey Tumanov, and Ramachandran Ramjee. 2024. Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve. In 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24), pages 117–134.

Amos Azaria and Tom Mitchell. 2023. The Internal State of an LLM Knows When It’s Lying. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 967–976.

Alexandra Bazarova, Andrei Volodichev, Aleksandr Yugay, Andrey Shulga, Alina Ermilova, Konstantin Polev, Julia Belikova, Rauf Parchiev, Dmitry Simakov, Maxim Savchenko, Andrey Savchenko, Serguei Barannikov, and Alexey Zaytsev. 2026. Hallucination Detection in LLMs with Topological Divergence on Attention Graphs. In Proceedings ofthe 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 15449–15470.

Jakub Binkowski, Kamil Adamczewski, and Tomasz Jan Kajdanowicz. 2026. Attention Sinks as Internal Signals for Hallucination Detection in Large Language Models. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings ofMachine Learning Research, pages 8302–8319.

Sky CH-Wang, Benjamin Van Durme, Jason Eisner, and Chris Kedzie. 2024. Do Androids Know They’re Only Dreaming of Electric Sheep? In Findings of the Association for Computational Linguistics: ACL 2024, pages 4401–4420.

Chao Chen, Kai Liu, Ze Chen, Yi Gu, Yue Wu, Mingyuan Tao, Zhihang Fu, and Jieping Ye. 2024. INSIDE: LLMs’ Internal States Retain the Power of Hallucination Detection. In The Twelfth International Conference on Learning Representations, pages 3056–3076.

Jiayu Chen, Ruoyu Lin, Zihao Zheng, Jingxin Li, Maoliang Li, Guojie Luo, and Xiang Chen. 2026a. ToProVAR: Efficient Visual Autoregressive Modeling via Tri-Dimensional Entropy-Aware Semantic Analysis and Sparsity Optimization. In The Fourteenth International Conference on Learning Representations.

Wenbo Chen, Veena Padmanabhan, Tootiya Giyahchi, Elaine Wong, and Leman Akoglu. 2026b. Rethinking Evaluation for LLM Hallucination Detection: A Desiderata, A New RAG-based Benchmark, New Insights. In Proceedings of the 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 14912–14931.

Yung-Sung Chuang, Linlu Qiu, Cheng-Yu Hsieh, Ranjay Krishna, Yoon Kim, and James R. Glass. 2024. Lookback Lens: Detecting and Mitigating Contextual Hallucinations in Large Language Models Using Only Attention Maps. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 1419–1436.

Cohere Labs. 2024. Command R7B.

Tri Dao. 2024. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. In The Twelfth International Conference on Learning Representations, pages 35549–35562.

Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. In Advances in Neural Information Processing Systems, volume 35, pages 16344–16359.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, et al. 2024. The Llama 3 Herd of Models. arXiv preprint arXiv:2407.21783.

Leo Feng, Frederick Tung, Hossein Hajimirsadeghi, Mohamed Osama Ahmed, Yoshua Bengio, and Greg Mori. 2024. Attention as an RNN. arXiv preprint arXiv:2405.13956.

Gemma Team. 2026. Gemma 4 Technical Report. arXiv preprint arXiv:2607.02770.

Shaona Ghosh, Prasoon Varshney, Makesh Narsimhan Sreedhar, Aishwarya Padmakumar, Traian Rebedea, Jibin Rajan Varghese, and Christopher Parisien. 2025. AEGIS2.0: A diverse AI safety dataset and risks taxonomy for alignment of LLM guardrails. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5992–6026.

Xiangming Gu, Tianyu Pang, Chao Du, Qian Liu, Fengzhuo Zhang, Cunxiao Du, Ye Wang, and Min Lin. 2025. When Attention Sink Emerges in Language Models: An Empirical View. In The Thirteenth International Conference on Learning Representations.

Jiatong Han, Neil Band, Muhammed Razzak, Jannik Kossen, Tim G. J. Rudner, and Yarin Gal. 2025. Simple Factuality Probes Detect Hallucinations in Long-Form Natural Language Generation. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 16209–16226.

Seungju Han, Kavel Rao, Allyson Ettinger, Liwei Jiang, Bill Yuchen Lin, Nathan Lambert, Yejin Choi, and Nouha Dziri. 2024. WildGuard: Open one-stop moderation tools for safety risks, jailbreaks, and refusals of LLMs. In Advances in Neural Information Processing Systems, volume 37, pages 8093–8131.

Kevin David Hayes, Arka Pal, Haosong Zhang, Tom Goldstein, and Micah Goldblum. 2026. Pinocchio: Fast Uncertainty Estimates for Black-Box Language Models. In Third Conference on Language Modeling.

IBM Granite Team. 2026. Granite-4.2-8B.

Lanlan Ji, Dominic Seyler, Gunkirat Kaur, Manjunath Hegde, Koustuv Dasgupta, and Bing Xiang. 2025. PHANTOM: A Benchmark for Hallucination Detection in Financial Long-Context QA. In Advances in Neural Information Processing Systems, volume 38, pages 140939–140968.

Sanyam Kapoor, Nate Gruver, Manley Roberts, Katherine Collins, Arka Pal, Umang Bhatt, Adrian Weller, Samuel Dooley, Micah Goldblum, and Andrew Gordon Wilson. 2024. Large Language Models Must Be Taught to Know What They Don’t Know. In Advances in Neural Information Processing Systems, volume 37.

Alexandra Kuleshova, Andrei Volodichev, Daria Kotova, and Alexey Zaytsev. 2026. INTRYGUE: Induction-Aware Entropy Gating for Reliable RAG Uncertainty Estimation. arXiv preprint arXiv:2603.21607. Version 3, 31 August 2026. Accepted to EMNLP 2026 (Main).

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient Memory Management for Large Language Model Serving with PagedAttention. In Proceedings of the 29th Symposium on Operating Systems Principles, pages 611–626.

Miaoran Li, Rogger Luo, and Ofer Mendelevitch. 2024a. HHEM-2.1-Open.

Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. 2024b. SnapKV: LLM Knows What You are Looking for Before Generation. In Advances in Neural Information Processing Systems, volume 37, pages 22947–22970.

Cheng Liang, Haoxian Chen, Liang Hou, Qi Fan, Gangshan Wu, Xin Tao, and Limin Wang. 2026. VMonarch: Efficient Video Diffusion Transformers with Structured Attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4613–4623.

Alexander H. Liu, Kartik Khandelwal, Sandeep Subramanian, et al. 2026. Ministral 3. arXiv preprint arXiv:2601.08584.

Hao Liu, Matei Zaharia, and Pieter Abbeel. 2024. RingAttention with Blockwise Transformers for Near-Infinite Context. In The Twelfth International Conference on Learning Representations, pages 3992–4008.

Maxim Milakov and Natalia Gimelshein. 2018. Online Normalizer Calculation for Softmax. arXiv preprint arXiv:1805.02867.

Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu, KaShun Shum, Randy Zhong, Juntong Song, and Tong Zhang. 2024. RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 10862–10878.

Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, and 7 others. 2022. Incontext Learning and Induction Heads. Transformer Circuits Thread.

Qwen Team. 2026. Qwen3.5-9B.

Markus N. Rabe and Charles Staats. 2021. Selfattention Does Not Need O(n<sup>2</sup>) Memory. arXiv preprint arXiv:2112.05682.

Selvan Sunitha Ravi, Bartosz Mielczarek, Anand Kannappan, Douwe Kiela, and Rebecca Qian. 2024. Lynx: An Open Source Hallucination Evaluation Model. arXiv preprint arXiv:2407.08488.

Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. 2024. FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision. In Advances in Neural Information Processing Systems, volume 37, pages 68658–68685.

Vaishnavi Shrivastava, Percy Liang, and Ananya Kumar. 2023. Llamas Know What GPTs Don’t Show: Surrogate Models for Confidence Estimation. arXiv preprint arXiv:2311.08877.

Akshita Singh, Prabesh Paudel, and Siddhartha Roy. 2026. Hallucination Detection via Activations of Open-Weight Proxy Analyzers. arXiv preprint arXiv:2605.07209.

Gaurang Sriramanan, Siddhant Bharti, Vinu Sankar Sadasivan, Shoumik Saha, Priyatham Kattakinda, and Soheil Feizi. 2024. LLM-Check: Investigating Detection of Hallucinations in Large Language Models. In Advances in Neural Information Processing Systems, volume 37, pages 34188–34216.

Liyan Tang, Philippe Laban, and Greg Durrett. 2024. MiniCheck: Efficient fact-checking of LLMs on grounding documents. In Proceedings of the 2024

Conference on Empirical Methods in Natural Language Processing, pages 8818–8847.

Team GLM. 2024. ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools. arXiv preprint arXiv:2406.12793.

Philippe Tillet, H. T. Kung, and David Cox. 2019. Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations. In Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages, pages 10–19.

Dezhan Tu, Danylo Vashchilenko, Yuzhe Lu, and Panpan Xu. 2025. VL-Cache: Sparsity and Modality-Aware KV Cache Compression for Vision-Language Model Inference Acceleration. In The Thirteenth International Conference on Learning Representations.

Dennis Ulmer, Martin Gubri, Hwaran Lee, Sangdoo Yun, and Seong Joon Oh. 2024. Calibrating Large Language Models Using Their Generations Only. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 15440–15459.

Artem Vazhentsev, Lyudmila Rvanova, Gleb Kuzmin, Ekaterina Fadeeva, Ivan Lazichny, Alexander Panchenko, Maxim Panov, Mrinmaya Sachan, Preslav Nakov, Timothy Baldwin, and Artem Shelmanov. 2026. Efficient Hallucination Detection for LLMs Using Uncertainty-Aware Attention Heads. In Proceedings ofthe 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 123596–123627.

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. 2024. Efficient Streaming Language Models with Attention Sinks. In The Twelfth International Conference on Learning Representations.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, et al. 2025. Qwen3 Technical Report. arXiv preprint arXiv:2505.09388.

Zihao Ye, Lequn Chen, Ruihang Lai, Wuwei Lin, Yineng Zhang, Stephanie Wang, Tianqi Chen, Baris Kasikci, Vinod Grover, Arvind Krishnamurthy, and Luis Ceze. 2025. FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving. In Proceedings of Machine Learning and Systems, volume 7.

Samuel Yeh, Sharon Li, and Tanwi Mallick. 2026. LU-MINA: Detecting Hallucinations in RAG System with Context–Knowledge Signals. In The Fourteenth International Conference on Learning Representations.

Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, Zhangyang

Wang, and Beidi Chen. 2023. H O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models. In Advances in Neural Information Processing Systems, volume 36, pages 34661–34710.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems 36 (Datasets and Benchmarks Track), volume 36, pages 46595–46623.

Jingwei Zuo, Maksim Velikanov, Ilyas Chahed, et al. 2025. Falcon-H1: A Family of Hybrid-Head Language Models Redefining Efficiency and Performance. arXiv preprint arXiv:2507.22448.

## Guide to the appendices

The appendices follow the order of the main text.

Related work. Appendix A.

Method (Sections 2–3). Definitions, tiers, and users of every statistic the kernels compute (Appendix B); proof of Lemma 1 and the accumulator’s merge and stability (Appendix C); pseudocode and kernel implementation (Appendix D).

Kernel benchmark (Section 4). Hardware, software, and timing protocol (Appendix E); numerical accuracy, Q1 (Appendix F); cost per tier and length scaling, Q2 (Appendix G); whole-model latency, Q3 and Figure 1 (Appendix H); memory (Appendix I); a per-layer comparison with dense attention on short inputs (Appendix J).

Proxy detection (Section 5). Datasets,

HaluBench filtering, and text-only baselines (Appendix K); feature recipes, logit features, training and significance tests, full AUROC results, and LLM judges (Appendix L); extraction cost (Appendices H and I).

Additional experiment. Cross-domain transfer of attention detectors on HaluBench (Appendix M). Judge prompts. The four prompt templates of the LLM judges (Appendix N).

## A Related Work

Online softmax, its associative merge, and its fusion with other reductions such as top-k are established (Milakov and Gimelshein, 2018; Rabe and Staats, 2021; Liu et al., 2024; Feng et al., 2024), and the separable row statistics of Section 3.2 follow directly from it; recovering a weight from the stored LSE is how the FlashAttention backward pass recomputes attention (Dao, 2024); and attending from a block of new queries to a cached prefix with a shifted causal mask is the key–value-cache append that serving systems and their kernels already support (Kwon et al., 2023; Agrawal et al., 2024; Ye et al., 2025). Column sums of attention over past queries are the eviction scores of key– value-cache compression: $_ \mathrm { H _ { 2 } O }$ sums over all past queries and SnapKV over an observation window (Zhang et al., 2023; Li et al., 2024b). Similar fused kernels appear as implementation details elsewhere: an online entropy accumulator in FlashAttention for visual generation (Chen et al., 2026a; Liang et al., 2026), and a second key-major pass for column sums over a short query window in VL-Cache (Tu et al., 2025).

Several hallucination detectors read attention: Lookback Lens uses context-versus-answer attention ratios (Chuang et al., 2024), LLM-Check studies internal signals of an auxiliary model (Sriramanan et al., 2024), SinkProbe probes attentionsink scores (Binkowski et al., 2026), TOHA uses graph structure in attention (Bazarova et al., 2026), and INTRYGUE gates predictive entropy by the sink rates of induction heads (Kuleshova et al., 2026). These methods motivate our feature families.

Outside hallucination detection, guard models such as WildGuard and Aegis2.0 moderate the responses of other models with 7–8B decoders (Han et al., 2024; Ghosh et al., 2025). Our proxies follow the same setup for contextual hallucinations and read their own attention rather than generating a verdict.

Taken together, efficient-attention work fuses reductions into the pass to produce the attention output or to rank keys for eviction, while attentionbased detectors read their statistics from the materialized attention matrix. We connect the two: we map detector statistics to exact paths that avoid the matrix, measure their cost, and use them in small proxy detectors.

## B Statistics computed by the kernels

Table 3 defines every statistic the response-stage kernels compute and the detectors that use it. Notation follows Sections 2 and 3: a response row i ∈ R sees the keys $\mathcal { T } _ { i } .$ , which are the P prompt keys and the response keys $P \ \leq \ j \ \leq \ i$ minus padding; $t _ { i } = i - P + 1$ is the number of response keys it sees, including itself, and $n _ { i } = | \mathcal { I } _ { i } |$ . The kernels return per-row or per-column statistics for each head; the host turns them into the features of the last column in $O ( R )$ or $O ( L )$ time per head, and a detector recipe reduces these over response rows and selected heads (Section 5.2). For the fixed sink we write $p _ { i s }$ for the per-row weight, $c _ { s } = \textstyle \sum _ { i } p _ { i s }$ for its column sum, and $\bar { c } _ { s }$ for the normalized sink rate. Qwen3 defines no beginning-of-sequence token; we call its first non-padded token, which opens the chat template, BOS.

Statistics only on response rows. Every statistic our detector reads is indexed by a response query row, and its column statistics sum over response rows only (Section 5.2). We therefore run the prompt stage with any production kernel and compute statistics only in the response stage. For most published features this changes nothing: the lookback ratio of Lookback Lens and the previoustoken weight of RAUQ are defined on generated tokens (Chuang et al., 2024; Vazhentsev et al., 2026). Two features change. LLM-Check averages log $p _ { i i }$ over all tokens (Sriramanan et al., 2024), and SinkProbe averages a key’s weight over all later tokens (Binkowski et al., 2026); we restrict both to response rows (Table 3). On long contexts this keeps the prompt from dominating both averages, but it is a different feature, and we have not compared the two versions in detection.

Features built from the statistics. Normalized entropy divides $H _ { i }$ by log $n _ { i }$ , the entropy of uniform attention over the visible keys, so rows at different positions are comparable. The lookback ratio of Lookback Lens compares the mean weight per prompt key with the mean weight per response key (Chuang et al., 2024):

$$
\mathrm { L R } _ { i } = \frac { \mu _ { i } ^ { \mathrm { P } } / P } { \mu _ { i } ^ { \mathrm { P } } / P + \mu _ { i } ^ { \mathrm { R } } / t _ { i } } .
$$

The log-diagonal feature is the per-head mean $\begin{array} { r } { \frac { 1 } { R } \sum _ { i \in \mathcal { R } } \log p _ { i i } } \end{array}$ , with each term clamped at log ϵ. It is the attention score of LLM-Check (Sriramanan et al., 2024), the log-determinant of a head’s attention matrix, which for a causal, hence lowertriangular, matrix is the sum of log $p _ { i i }$ over its rows. The previous-token feature is $p _ { i , i }$ <sub>−1</sub> on each response row, set to zero on the first response row, which has no previous response token; RAUQ reads it on selected heads (Vazhentsev et al., 2026). Attention sinks are keys that receive a large share of attention regardless of content (Xiao et al., 2024);

sink rates measure how much attention response rows put on a single key (Gu et al., 2025). Each column sum is divided by the number of response rows that see its key, $\bar { c } _ { j } = c _ { j } / \operatorname* { m i n } ( R , L - j )$ , so $\bar { c } _ { j }$ is the mean weight on key $j$ . The BOS sink rate is $\bar { c } _ { s }$ for the first non-padded key $s ,$ which T2 obtains without the full column sums when every sequence in the batch has the same padding; otherwise, and for any other sink definition, the kernels fall back to T3. The top-k sink rate is the mean of the k largest $\bar { c } _ { j } ( k = 1$ by default). SinkProbe uses the same per-key sink score, the mean weight a key receives from later tokens, and feeds the k largest scores of every head to a probe (Binkowski et al., 2026). INTRYGUE gates the entropy of the next-token distribution, which comes from the logits rather than the attention, by the top-1 sink rates of induction heads, computed over response rows as here (Kuleshova et al., 2026).

Not computed by our kernels. Stored attention maps, the attention-graph divergence MTopDiv (Bazarova et al., 2026), and row top-k mass are computed from the materialized response block. Row top-k mass needs $O ( k )$ state per row, which we keep out of the pass, so T1 does not contain every bounded-state reduction. Threshold counts and column entropy fit T3 but are not implemented.

<table><tr><td>Statistic</td><td>Definition (per head)</td><td>Tier</td><td>Shape</td><td>Kernel computation</td><td>Derived feature and use</td></tr><tr><td>Log-sum-exp</td><td> $\begin{array} { r } { \mathrm { L S E } _ { i } = \log \sum _ { j \in \mathcal { I } _ { i } } e ^ { s _ { i j } } } \end{array}$ </td><td>Base</td><td> $N _ { h } \times R$ </td><td> $m _ { i } + \log \ell _ { i }$  after the last key tile, as in FlashAttention</td><td>none; normalizer reused by T2 and T3</td></tr><tr><td>Entropy</td><td> $\begin{array} { r } { H _ { i } = - \sum _ { j } p _ { i j } \log p _ { i j } } \end{array}$ </td><td>T1</td><td> $N _ { h } \times R$ </td><td> $\mathrm { L S E } _ { i } - A _ { i } / \ell _ { i }$  with  $\alpha = 1 , g = s$ </td><td> $H _ { i } / \log n _ { i } .$  Use: ours (A7)</td></tr><tr><td>Effective support</td><td> $\begin{array} { r } { N _ { \mathrm { e f f } , 2 , i } = \big ( \sum _ { j } p _ { i j } ^ { 2 } \big ) ^ { - 1 } } \end{array}$ </td><td>T1</td><td> $N _ { h } \times R$ </td><td> $\ell _ { i } ^ { 2 } / A _ { i } \mathrm { ~ w i t h ~ } \alpha = 2 ,$   $g = 1$ </td><td>Use: ours  $( \mathbf { A } _ { 7 } )$ </td></tr><tr><td>Maximum weight</td><td> $\operatorname* { m a x } _ { j } p _ { i j }$ </td><td>T1</td><td> $N _ { h } \times R$ </td><td> $1 / \ell _ { i }$  , no accumulator</td><td>Use: ours  $( \mathbf { A } _ { 7 } )$ </td></tr><tr><td>Prompt mass</td><td> $\begin{array} { r } { \mu _ { i } ^ { \mathrm { P } } = \sum _ { j < P } p _ { i j } } \end{array}$ </td><td>T1</td><td> $N _ { h } \times R$ </td><td> $A _ { i } / \ell _ { i } \mathrm { \ w i t h } \alpha = 1 ,$   $g = { \bf 1 } [ j < P ]$ </td><td>lookback ratio. Use: Lookback Lens (Chuang et al., 2024)</td></tr><tr><td>Response mass</td><td> $\begin{array} { r } { \mu _ { i } ^ { \mathrm { R } } = \sum _ { j \geq P } p { i j } } \end{array}$ </td><td>T1</td><td> $N _ { h } \times R$ </td><td> $1 - \mu _ { i } ^ { \mathrm { P } } , \mathrm { d e r i v e d } ; \mathrm { n o }$  extra state</td><td>lookback ratio. Use: Lookback Lens</td></tr><tr><td>Log-diagonal</td><td> $\log p _ { i i }$ </td><td>T2</td><td> $N _ { h } \times R$ </td><td> $s _ { i i } - \mathrm { L S E } _ { i }$ </td><td>mean over response rows. Use: LLM-Check (Sriramanan et al., 2024), adapted</td></tr><tr><td>Previous-token weight</td><td> $p _ { i , i - 1 }$ </td><td>T2</td><td> $N _ { h } \times R$ </td><td> $\mathrm { e x p } ( s _ { i , i - 1 } - \mathrm { L S E } _ { i } )$ </td><td>Use: RAUQ (Vazhentsev et al., 2026)</td></tr><tr><td>Fixed-sink weight</td><td> $p _ { i s }$  for a fixed key s, e.g. the first non-padded token (BOS)</td><td>T2</td><td> $N _ { h } \times S \times R$ </td><td> $\exp ( { s _ { i s } } - \mathrm { L S E } _ { i } ) ;$  the host sums it into cs</td><td>BOS sink rate  $\overline { { c } } _ { s } .$  Use: ours (A7)</td></tr><tr><td>Column sum</td><td> $\begin{array} { r } { c _ { j } = \sum _ { i \in \mathcal { R } , i \geq j } p _ { i j } } \end{array}$  for every key j</td><td>T3</td><td> $N _ { h } \times L$ </td><td>key-major rescoring with LSE</td><td>top-k sink rate,  $\mathbf { A } _ { 6 4 } .$  Use: SinkProbe (Binkowski et al., 2026), adapted; INTRYGUE (Kuleshova et al., 2026); ours (A64)</td></tr></table>

Table 3: Statistics computed by the response-stage kernels. Tier: computation path (Section 3.1); Base: written by the statistics-off forward. Shape: kernel output per layer for one sequence, with $N _ { h }$ query heads, R response rows, L keys, and S fixed sink keys. $A _ { i }$ is the T1 accumulator of Equation 3 for the given α and $g$ after the last key tile, and $\ell _ { i }$ the running sum of the online softmax. Derived features are defined in the text; “adapted”: restricted to response rows.

## C Separable row reductions

Proof of Lemma 1. By induction over key tiles, Equation 4 keeps $A _ { u }$ equal to Equation 3. After the last tile, $\ell = e ^ { \mathrm { L S E } - m }$ , so

$$
{ \frac { A _ { u } } { \ell ^ { \alpha _ { u } } } } = \sum _ { j } \left( { \frac { e ^ { s _ { j } - m } } { \ell } } \right) ^ { \alpha _ { u } } g _ { u } ( i , j , s _ { j } ) = F _ { i } ^ { ( u ) } .
$$

Thus each $F _ { i } ^ { ( u ) }$ , and any fixed function of these values and the LSE, is available after the pass with the stated state and work. □

Merge and stability. Two accumulator states over disjoint key sets merge as

$$
\begin{array} { r l } & { A _ { u } = e ^ { \alpha _ { u } \left( m _ { 1 } - m \right) } A _ { u , 1 } } \\ & { \phantom { A _ { u } = e ^ { \alpha _ { u } \left( m _ { 2 } - m \right) } A _ { u , 2 } , } } \\ & { m = \operatorname* { m a x } ( m _ { 1 } , m _ { 2 } ) , } \end{array}
$$

an associative operation that also permits keys to be split across kernel instances or devices (Liu et al., 2024; Feng et al., 2024; Ye et al., 2025). Since $m ^ { \prime } \geq m$ and $m ^ { \prime } \geq s _ { j }$ , the rescaling factors in Equation 4 lie in (0, 1], so no exponential overflows. The maximizing key contributes one, giving $1 \leq$ $\ell \leq | { \mathcal { I } } _ { i } |$ and $| A _ { u } | \leq | { \mathcal { I } } _ { i } |$ max<sub>j</sub> $| g _ { u } ( i , j , s _ { j } )$ |.

## D Implementation

Statistics in the pass. The response-stage kernel applies the update of Equation 4 inside its loop, next to the FlashAttention update (Figure 2b). A response query tile stays resident while key/value tiles stream across it, and each score tile updates both the FlashAttention state and the statistic accumulators before the next tile is loaded; nothing is retained per key. Algorithm 1 gives the pseudocode for all three kernels.

Entropy uses the unshifted score $g = s$ rather than $s - m$ , since $\begin{array} { r } { H _ { i } = \mathrm { L S E } _ { i } - \sum _ { j } p _ { i j } s _ { i j } } \end{array}$ . Nonlinearity alone is not the boundary of T1: $\textstyle \sum _ { j } p _ { i j } ^ { 2 }$ remains separable. The maximum weight $1 / \ell _ { i }$ requires m to be the exact running maximum; a kernel that defers rescaling when the maximum grows would have to track it separately. The state is $N _ { s }$ numbers per row for $N _ { s }$ scalar accumulators, independent of $L ,$ and writing them for $N _ { h }$ heads costs $O ( N _ { h } R N _ { s } )$ memory traffic.

Algorithm 1 Attention with statistics: pseudocode   
for Figure 2b–c. Black lines are the unchanged   
FlashAttention-2 forward (blue in Figure 2); green   
lines are our additions, tagged with their tier. Each   
kernel instance handles one query tile I (or, in T3,   
one key tile J) of one head; $m , \ell , A _ { u }$ hold one entry   
per query row. A key tile is visible to a query tile   
if some row of I may attend to some key of $J ;$ the   
mask removes padded keys and, on tiles that cross   
the diagonal, future keys. $G _ { u } = g _ { u } ( J , S )$ , and S   
is an optional set of fixed sink columns, whose   
weights $p _ { i s }$ the host sums over rows into $c _ { s }$ . The   
LSE is part of the unchanged forward and belongs   
to no tier.   
1: $O _ { < P } \gets \mathrm { C a u s a l A t t n } ( Q _ { < P } , K _ { < P } , V _ { < P } )$ ▷ any kernel   
2: for each response query tile $I \subseteq { \mathcal { R } }$ do   
3: $m \gets - \mathrm { \hat { \infty } } , \ell \gets 0 , O \gets 0$   
4: $A _ { u } \gets 0 \mathrm { f o r } u = 1 , \dots , N _ { s }$ T1   
5: for each key tile $\_$ visible to I do   
6: $S \gets \tau \dot { Q } _ { I } K _ { J } ^ { \top }$ ; mask   
7: $m ^ { \prime } \gets$ max(m, rowmax $( S ) )$   
8: $a  e ^ { m - m ^ { \prime } } ; \tilde { P }  e ^ { S - m ^ { \prime } }$   
9: $A _ { u }  a ^ { \alpha _ { u } } A _ { u } +$ rowsum $\left( \tilde { P } ^ { \alpha _ { u } } \odot G _ { u } \right)$   
10: ${ \cal O } \gets a { \cal O } + \tilde { P } V _ { J } ; \ell \gets a \ell +$ rowsum $( \tilde { P } )$   
11: $m \gets m ^ { \prime }$   
12: end for   
13: write $O / \ell$ and LSE ← m + log ℓ   
14: write $F ^ { \dot { ( u ) } }$ from $\mathrm { L S E } , \ell , A _ { u } / \ell ^ { \alpha _ { u } }$ T1   
15: end for   
16: for each response row $i \in \mathcal { R }$ do   
17: log $p _ { i i } \gets s _ { i i } - \mathrm { L S E } _ { i }$ T2   
18: $p _ { i , i - 1 } \gets \mathrm { e x p } ( s _ { i , i - 1 } - \mathrm { L S E } _ { i } )$   
19: $\mathbf { \hat { \phi } } _ { p _ { i s } } \gets \exp ( \hat { s _ { i s } } - \mathrm { L S E } _ { i } ) \mathrm { f o r } s \in \mathcal { S }$   
20: end for   
21: for each key tile $J$ do   
22: $c _ { J }  0$ T3   
23: for each response query tile I that sees J do   
24: $W  \mathrm { e x p } ( \tau Q _ { I } K _ { J } ^ { \top } - \mathrm { L S E } _ { I } ) ;$ mask   
25: $c _ { J }  c _ { J } +$ colsum(W)   
26: end for   
27: write $\scriptstyle { \mathcal { C } } J$   
28: end for

Kernels and attention backend. A Hugging Face attention backend, installed only on the scored layers, splits each layer call into the two stages. The prompt stage calls the model’s original attention backend (in our runs, PyTorch SDPA with the FlashAttention-2 or cuDNN backend). The response stage is a Triton kernel (Tillet et al., 2019) derived from the FlashAttention-2 causal forward, with R query rows, L keys, and the causal mask shifted by $P ;$ it accumulates the T1 statistics and writes the LSE. T2 is a separate kernel that reads $Q , K$ , and the LSE. It does not depend on which

T1 statistics are compiled into the response-stage kernel, it works after any forward kernel that writes the LSE, and its positions are runtime arguments, so any set of sink columns can be read without recompiling. T3 is a key-major kernel modeled on the FlashAttention-2 backward: column sums are its value-gradient step, $d V = \mathbf { W } ^ { \top } d O$ for the attention weights W and the output gradient dO, with dO replaced by a column of ones on response rows. Each kernel instance owns one key tile, visits the response query tiles that see it, and writes its column sums once, without atomic additions. All kernels support grouped-query attention, bf16/fp16/fp8- e5m2 inputs, head dimensions 64/128/256, and a key keep mask; statistic families are compile-time flags. The kernels do not support sliding-window or soft-capped attention, both elementwise changes to the mask and the scores; layers that use them run the model’s default attention without statistics.

## E Kernel benchmark setup

All timed runs use a single NVIDIA H100 NVL (Hopper, 94 GB, power-capped at 270 W; CUDA 12.9, PyTorch 2.11, Triton 3.6, cuDNN 9.17, Transformers 5.5) in bf16. The production baselines are PyTorch SDPA with the flash backend (FlashAttention-2) and the cuDNN backend; FlashAttention-3 is not installed, and cuDNN is the Hopper-specific baseline. Besides the Qwen3-8B shapes, $d = 6 4$ uses the same head counts and $d = 2 5 6$ uses 8/4 heads (Gemma-3 global layers). Kernel tiles are autotuned once per head dimension at $L = 3 2 \mathrm { K } , r = 0 . 3$ and reused for every shape. Timings are medians over 100 repetitions after 25 warm-up runs with an L2 flush (configuration sweep), 15–80 repetitions depending on length (length scaling), or 20 repetitions after 3 warm-ups (per-layer comparison, Appendix J). Appendix I describes how we measure memory. Sink positions are resolved outside the timed region. The host is shared, but no other process ran on the measured GPU.

Dense baseline. Every comparison with materialized attention uses the path of detectors that read output\_attentions: the attention weights of the response rows are computed as an fp32 softmax cast to bf16 and reduced in PyTorch.

## F Numerical accuracy (Q1)

Unit tests compare the kernels against a dense PyTorch reference across dtypes (including fp8- e5m2), head dimensions, GQA ratios, padding masks, prompt/response splits, and statistic sets. Against an fp64 reference computed from the materialized matrix (row-chunked at 32K), we measure the absolute error on response rows of (i) the dense bf16 path (Appendix E) and (ii) our kernels (T1 in the pass, T2 lookups, T3 column sums over the response rows), on random Gaussian Q, K, V (response fraction 0.3) and on real Q, K, V from four Qwen3-8B layers (2, 16, 24, 34) on the longest input of the PHANTOM 20K set (Ji et al., 2025), its prompt truncated so that the input has 2K, 8K, or 32K tokens, with a response of at most 156 tokens. Random inputs give near-uniform attention and hide sinks and peaked heads; the real inputs do not.

<table><tr><td>Statistic</td><td>dense</td><td>fused</td><td>ratio</td></tr><tr><td>Entropy H (T1)</td><td>1.6e-2</td><td>3.5e-6</td><td>4.5e3</td></tr><tr><td>Effective support Neff,2 (T1)</td><td></td><td>5.3e0 5.9e-3 9.0e2</td><td></td></tr><tr><td>Maximum weight (T1)</td><td>4.3e-3 1.5e-7 2.9e4</td><td></td><td></td></tr><tr><td>Prompt mass (T1)</td><td>2.2e-3 1.3e-7 1.7e4</td><td></td><td></td></tr><tr><td>LSE (Base)</td><td>1.1e-2 1.1e-6 1.0e4</td><td></td><td></td></tr><tr><td> $\log p _ { i i } \left( \mathrm { T } 2 \right)$ </td><td>2.0e-2 1.2e-6 1.7e4</td><td></td><td></td></tr><tr><td> $p _ { i , i - 1 } \left( \mathrm { T } 2 \right)$ </td><td>5.0e-4 3.3e-8 1.5e4</td><td></td><td></td></tr><tr><td>BÓS column sum  $c _ { s } ~ ( \mathrm { T 2 } )$ </td><td>5.3e-2 7.1e-5 7.4e2</td><td></td><td></td></tr><tr><td>Column sum (T3)</td><td> $2 . 2 \mathrm { e } { - 4 }$ </td><td>6.6e-8 3.3e3</td><td></td></tr><tr><td>Attention output</td><td> $3 . 5 \mathrm { e } { - 3 }$ </td><td> $4 . 8 \mathrm { e } { - 4 }$ </td><td>7.2</td></tr></table>

Table 4: Mean absolute error on response rows against an fp64 reference for materialized attention (the dense bf16 path) and our bf16 kernels, and their ratio. Each entry averages the per-head error over heads and layers and reports the worst input type (random or PHANTOM) and length (2K, 8K, 32K). Our statistics are about three to four orders of magnitude closer to fp64; they have the lower error for every statistic, input, and length, in both mean and maximum error.

The dense path rounds each probability to bf16 (8 bits of mantissa) before reducing, while the kernels reduce fp32 probabilities that never leave registers. The T2 lookups and T3 column sums reuse the stored LSE, so moving a statistic out of the pass costs time, not accuracy. Both paths rank heads by their per-head means over response rows as fp64 does, with Spearman correlation 1.000 for our kernels and at least 0.998 for the dense path. The entropy accumulator uses the unshifted score g = s (Appendix D); its per-head mean error stays below 4.7e−5 even for heads with $H < 0 . 2 5$ at 32K, where the dense error reaches 1.4e−2.

## G Cost per tier and length scaling (Q2)

Notes on Table 5. Base runs the response rows on our kernel, which is faster than FlashAttention-2 on Hopper and slower than cuDNN, so as r grows, Base gains on full FA2 and loses to full cuDNN. On the response stage alone, ≤T3 takes 1.7–1.8× the time of Base at $d \geq 1 2 8$ and $2 . 1 \mathrm { - } 2 . 2 \times$ at $d = 6 4 ;$ the key-major T3 kernel does not spill registers, so this cost is the arithmetic of a second $Q K ^ { \top }$ pass over the response block, not register pressure. Inputs of 4K tokens are excluded: a layer then takes 0.3–1.0 ms, and its ratios reflect the fixed host time per call (Appendix H).

Length scaling. Figure 3b–c times the same layer at 20 geometrically spaced lengths from 1K to 100K tokens, at $d \ = \ 1 2 8$ (32 query heads, 8 KV heads, batch 1) and $r ~ \in ~ \{ 0 . 1 , 0 . 3 \}$ : full FlashAttention-2 and cuDNN without statistics, and the split layers Base, ≤T2 (with a fixed BOS sink), and ≤T3, with the prompt on cuDNN. From 15K to 100K tokens, each overhead stays within a band that does not grow with L. At $r = 0 . 3 , \pm \mathrm { T } 2$ adds 3–14% over Base, ${ \le } \mathrm { T } 3$ adds 32–47%, and Base takes 4–14% longer than cuDNN; at $r = 0 . 1$ the same overheads are −2 to +13%, 9–22%, and −2 to +8%. Below about 4K tokens, the split layer stays at 0.3–0.5 ms down to 1K tokens, against 0.04–0.07 ms for SDPA; this floor is host time per call (Appendix H).

<table><tr><td></td><td colspan="3"> $d = 6 4$ </td><td colspan="3"> $d = 1 2 8$ </td><td colspan="3"> $d = 2 5 6$ </td></tr><tr><td>Overhead (%)</td><td> $r { = } . 1$ </td><td> $r { = } . 3$ </td><td> $r { = } . 5$ </td><td> $r { = } . 1$ </td><td> $r { = } . 3$ </td><td> $r { = } . 5$ </td><td> $r { = } . 1$ </td><td> $r { = } . 3$ </td><td> $r { = } . 5$ </td></tr><tr><td>≤T1 (in pass) vs. Base</td><td>9-12</td><td>22</td><td>27-28</td><td>3-7</td><td>12-15</td><td>9</td><td>1-4</td><td>6-9</td><td>2-6</td></tr><tr><td>≤T2 (+ lookups) vs. Base</td><td>9-12</td><td>21-22</td><td>27-30</td><td>3-6</td><td>12-20</td><td>9-10</td><td>1-4</td><td>6-10</td><td>4-7</td></tr><tr><td>≤T3 (+ col. sums) vs. Base</td><td>25-26</td><td>64</td><td>87</td><td>16-19</td><td>39-44</td><td>53-58</td><td>19-26</td><td>39-43</td><td>54-57</td></tr><tr><td>Base vs. full cuDNN</td><td>6-11</td><td>12-19</td><td>16-20</td><td>4-9</td><td>7-15</td><td>10-17</td><td>10-13</td><td>20-25</td><td>28-34</td></tr><tr><td>Base vs. full FA2</td><td>0-3</td><td></td><td>-4--1-9--8</td><td></td><td></td><td>-2-+1-8--5-13--12</td><td></td><td>2-5--2 -16--11 -21--20</td><td></td></tr><tr><td>≤T2 vs. full cuDNN</td><td>15-25</td><td>36-45</td><td>48-55</td><td>7-16</td><td>19-38</td><td>20-29</td><td>14</td><td>28-37</td><td>33-43</td></tr></table>

Table 5: The numbers behind Figure 3a (Q2): latency overhead (%) of one attention layer, as a range over ${ \cal L } \in \{ 1 6 \mathrm { K } , 3 2 \mathrm { K } \}$ . Upper rows: each configuration of Section 4 over the statistics-off split Base, with the prompt stage on SDPA cuDNN. Lower rows: Base over SDPA on the whole sequence, with the prompt stage on the same backend, and the cost of all row statistics (≤T2) over full SDPA cuDNN, the fastest production kernel here. H100 NVL, bf16.

## H Whole-model latency and extraction cost (Q3)

Setup of Figure 1. Figure 1 and the figures of this appendix time one forward of Qwen3-8B (all 32 heads of all 36 layers) and of Qwen3-0.6B (all 16 heads of all 28 layers) on PHANTOM and TRIVIA+ inputs. Our kernels compute the statistics underlying Lookback Lens, LLM-Check, RAUQ, INTRYGUE, and SinkProbe; the baseline is the same forward without features. Attention outside our kernels runs on SDPA with FlashAttention-2 for prompts up to 8K tokens and with its default cuDNN backend for longer ones, the same in both forwards; Q3 uses FlashAttention-2 throughout. Means are taken per length bin, and bins with fewer than 12 samples are omitted. Batch 1, H100 NVL, bf16.

Data for Q3. From RAGTruth QA, Q3 takes the 989 responses of Llama-2-70B-chat, median 0.5K and at most 1.0K tokens. From PHANTOM, it takes the split of the 10-K filings set that PHAN-TOM labels 2,000 tokens, with the answer-bearing passage in the middle: 980 samples with median 2.6K and at most 8.4K Qwen3 tokens. Token counts include the prompt template.

Dense baseline for Q3. Dense (Appendix E) takes each layer’s output from SDPA, then recomputes the attention weights of the response rows for all 32 heads at once, reduces them, and frees them before the next layer. Keeping every layer’s full weights from the forward, as output\_attentions does, would take 152 GiB at 8.4K tokens, more than the GPU holds; Dense instead trades one recomputation of $Q K ^ { \top }$ and the softmax per layer for the memory of one layer’s response block, 32RL weights. Taking the response output from the materialized weights instead would avoid scoring the response rows twice but add a product with V.

<table><tr><td>Per forward</td><td>Floor</td><td>Ours</td></tr><tr><td>Latency (ms)</td><td>33</td><td>108</td></tr><tr><td>GPU busy (ms)</td><td>13</td><td>18</td></tr><tr><td>GPU kernels</td><td>2,005</td><td>5,173</td></tr><tr><td>Host-device syncs</td><td>1</td><td>361</td></tr></table>

Table 6: Where the time of a short forward goes: Qwen3- 8B on five TRIVIA+ inputs of 90–253 tokens, all 36 layers scored, FlashAttention-2, H100 NVL. GPU busy is the summed device time of all kernels (PyTorch profiler); syncs are counted with PyTorch’s synchronization debug mode.

Fixed host cost on short inputs. Figure 5 repeats Figure 1 without the 2K cut, with one panel per model. Below about 600 tokens for Qwen3-8B and 3K for Qwen3-0.6B, a forward with features takes the same time at every length: the features add a fixed 60 and 47 ms, about 1.7 ms per scored layer for both models, although a Qwen3-0.6B layer does a fraction of the work of a Qwen3-8B layer.

This cost is host time, not GPU work (Table 6). For every scored layer, the wrapper runs Python code, launches 88 more kernels than the plain layer, most of them small element-wise operations that turn row statistics and column sums into features, and reads ten values back from the GPU: the prompt length, response length, and padding of the sample, twice; a check for padding in the prompt block; and three index lists for the selected heads. Each read stalls the host until the GPU queue drains. Under the Python profiler, launching our three kernels takes 0.8 ms per layer, assembling features another 0.8 ms, and copying the configuration of every scored layer to switch its attention backend 9 ms per forward. None of this grows with the input. The ten values could be computed once per forward, and capturing the scored layers in a CUDA graph would remove the launch cost. A split layer called on its own, as in Appendices G and J, pays only its setup and the 19 kernel launches for all row statistics, about 0.3–0.6 ms of host time per call.

![](images/cf81acdecf9a5b9db4bb230c4ba5a30aace2dce0ed04b782d029452d190cb1f8.jpg)

![](images/ff80d42910ab635a29f5b843d19d38027473a96902685a8c6a9d67cf3ba4531e.jpg)  
Figure 5: Figure 1 at every length, for Qwen3-8B (left) and Qwen3-0.6B (right, all 16 heads of all 28 layers), with the overhead on a log scale. Dotted line: the 2K cut of Figure 1. Same inputs, backends, and binning; batch 1, H100 NVL, bf16.

## I Memory

Measurement. All memory numbers count Py-Torch allocations: max\_memory\_allocated after reset\_peak\_memory\_stats, minus the memory allocated before the measured call. They exclude memory that the caching allocator has reserved but not handed out, so nvidia-smi shows more. A buffer counts only if it is allocated at the moment of the peak. For one layer (Q2 and Appendix J), Q, K, and V are allocated before the call, so the extra peak is what the layer allocates: its output, the statistics, and temporary buffers. For a whole forward (Q3 and Figure 6), the weights and inputs are allocated before the call, so the extra peak covers activations and features but not the weights.

One attention layer (Q2). At 32K and d = 128, the extra peak of a layer is 520–532 MiB in every configuration, against 256 MiB for SDPA on the whole sequence, and 1.6 against 0.8 GiB at 100K. Our kernel allocates the statistic outputs even with statistics off, so Base has the same peak; the statistics take 7.6–20 MiB of it, 1.5–4%. Most of the difference from SDPA is the concatenation of the two stage outputs: the layer must return all L rows, so it copies them into one output-sized tensor while both are still allocated. The attention matrix of the same layer would take 64 GiB at 32K.

Whole forward (Q3). The concatenated copy is freed within its layer and does not reach the peak of a whole forward. Over the floor, our extra peak grows by 26–32 KiB per response token and not with the prompt (least-squares fit on each dataset), whereas an output-sized copy at the peak would add 8 KiB per token of the sequence. The excess is mostly the extracted statistics, 22.5 KiB per response token for Qwen3-8B, which stay on the GPU until the forward returns. Dense holds the response block of one layer, 32RL weights, at a time (Appendix H). Ours adds at most 3.2% to the floor on PHANTOM, whose prompts all exceed 1.1K tokens, and 6% at the median and up to 17% on the short inputs of RAGTruth QA.

Longer inputs. Figure 6 extends the wholeforward measurement to 35K tokens, on the runs of Figure 1 (Appendix H). Without features, a forward peaks at 104.5 KiB per token above the weights for Qwen3-8B and 26.6 KiB for Qwen3-0.6B. Our features store 20 bytes per head, layer, and response token, the five per-token features in fp32 (22.5 and 8.75 KiB per response token), plus 1.0 and 0.5 MiB that the wrapper holds between forwards; the measured peak grows by about 28 and 11 KiB per response token (least-squares fit over these runs). On short inputs this is a larger share of a smaller forward, up to 14% for Qwen3-8B below 100 tokens.

![](images/bb08a0733378948996895c6302e5c9277146c8d48d5e67b709c72a052eea0bb6.jpg)  
Figure 6: Figure 1 for peak memory: the peak of each forward above the memory allocated before it, so the model weights are excluded, with all features (solid, points) and without (dashed). Same runs, inputs of at least 2K tokens, and binning; batch 1, H100 NVL, bf16.

Below about 1K tokens, the peaks of Qwen3-0.6B jump between levels at multiples of 128 tokens in both forwards. This matches the scratch buffer that FlashAttention-2 sizes by the number of query tokens when it splits a short input across more thread blocks: the plain forward attends with all L queries and ours runs the prompt with P, so near a step the two can land on different levels, by up to 31 MiB, independently of the features.

Reducing memory further. The concatenation is avoidable. If the prompt goes through a separate prefill, or its key–value cache is reused from generation, the response stage reads the cached keys and values and returns only its R rows, so nothing is concatenated; the price is that the prompt cache stays allocated across all layers. The per-token features stay on the GPU until the forward returns; reducing them to per-head aggregates within each layer would make their memory independent of the response length. Finally, we compare with dense end to end only up to 8.4K tokens, where a dense pipeline still fits; a long-context end-to-end comparison is future work.

## J Per-layer comparison on short inputs

Q3 compares whole forwards; here we time a single attention layer with statistics, ours against dense materialization, on the short real inputs of Figure 4, and check that both give the same features.

We use the 989 RAGTruth QA inputs of Q3 (Appendix H; response fraction ≈0.3), on which every method, including dense materialization, fits in memory. We capture Q, K, V of 48 samples (16 per length tercile) from 11 Qwen3-8B layers between layers 3 and 24 and time each method per layer call, for all 32 heads, at batch 1 and at batch 8 with padding and a keep mask. Dense materializes the attention matrix for all heads at once (Appendix E); Dense-loop does the same one head at a time, as the dense fallback of our pipeline does by default; Ours is our split layer (prompt rows on SDPA, response rows on our kernels). Every method obtains the attention output from the same PyTorch SDPA call and computes, on response rows only, the T1 and T2 statistics (entropy group, prompt mass, BOS sink, diagonal, previous token) and the full T3 column sums, from which the sink values of $\mathbf { A } _ { 6 4 }$ are read.

Latency and memory. At these lengths both methods are bound by host time, not GPU work: profiling a 1K-token layer with all row statistics shows 0.62 ms of host time to enqueue 19 kernels that together run for 0.10 ms on the GPU, against 0.06 ms for one cuDNN call. Dense is therefore slightly faster than ours at batch 1, with a median of 1.19 against 1.26 ms per layer call (ours/Dense

= 1.04–1.09 in every length tercile), and batching amortizes the host time (Appendix H), so ours is faster at batch 8, 4.71 against 7.60 ms. Dense-loop takes 35.9 ms at batch 1; most of this gap is its loop over heads, not materialization, so we use batched Dense as the dense baseline. The extra peak memory (Appendix I) of a layer call is 14 MiB for ours against 116 MiB for Dense at batch 1, and 136 MiB against 1.4 GiB at batch 8. Appendix G gives the dependence on L.

Feature agreement. Ours differs from Dense by a mean relative 0.04% (the log-diagonal feature of LLM-Check, Appendix B) to 1.6% (previous-token weight). The largest absolute difference, 6.8, is on effective support (typical values ≈ 12); against fp64, the mean error of effective support is 5.3 for Dense and 5.9e−3 for ours (Table 4).

## K Data

Table 7 gives the splits of the three benchmarks of Section 5.

RAGTruth QA (Niu et al., 2024): the questionanswering subset, with the answers of quality good from all six generators. We keep the official test split and divide the training answers 80/20 by query into training and validation (seed 42); groups are queries.

HaluBench (Ravi et al., 2024): 14,900 answers from six sources: HaluEval (10,000); DROP, CovidQA, PubMedQA, and FinanceBench (1,000 each); and RAGTruth (900). We filter and split it ourselves (below); the transfer experiment of Appendix M uses the same split.

TRIVIA+ (Chen et al., 2026b): the original training, validation, and test splits; groups are normalized source articles.

Within each dataset, every feature set and proxy uses the same training, validation, and test examples, and groups never cross splits. The datasets may overlap with each other, which does not matter here because no detector is applied across datasets.

<table><tr><td>Dataset</td><td>Train</td><td>Val</td><td>Test</td><td>Hal.</td><td>Grp.</td></tr><tr><td>RAGTruth QA</td><td>3921</td><td>971</td><td>875</td><td>160</td><td>150</td></tr><tr><td>HaluBench</td><td>4206</td><td>898</td><td>904</td><td>409</td><td>546</td></tr><tr><td>TRIVIA+</td><td>2263</td><td>316</td><td>645</td><td>224</td><td>358</td></tr></table>

Table 7: Splits used in Section 5. Hal.: hallucinated test answers; Grp.: test groups for the bootstrap.

HaluBench filtering and splitting. In HaluEval, faithful answers are mostly short entity names and hallucinated ones long sentences, so the answer length in words alone reaches an AUROC of 0.970 on its 10,000 answers and 0.720 on all 14,900. We therefore subsample HaluEval to equally many faithful and hallucinated answers at every answer length in words, which keeps 1,108 of its answers and 6,008 in total; after this, length gives an AU-ROC of 0.500 on HaluEval and 0.586 overall, where shorter answers are slightly more often hallucinated. We then group near-duplicate contexts within each source: two contexts are linked if at least 30% of the word 5-grams of the shorter one occur in the other, and two FinanceBench filings also if their word sets have a Jaccard similarity of at least 0.6 or they share at least six numbers of four or more digits other than years. Groups are the connected components of these links; whole groups are assigned to splits to approach 70/15/15% within every source and label (seed 42), and they are the units of the bootstrap.

Text-only baselines. To measure how much of each label the answer text reveals without the context, we score the test answers by their length and by a TF–IDF classifier on the answer alone (word unigrams and bigrams, the logistic regression of

<table><tr><td colspan="3">RAGTruth</td><td colspan="2">HaluBench</td></tr><tr><td>Baseline</td><td>QA</td><td>TRIVIA+ filtered unfiltered</td><td></td><td></td></tr><tr><td colspan="5">AUROC</td></tr><tr><td>Words</td><td>.702</td><td>.574</td><td>.580</td><td>.727</td></tr><tr><td>Characters</td><td>.720</td><td>.598</td><td>.568</td><td>.725</td></tr><tr><td>TF-IDF</td><td>.726</td><td>.658</td><td>.663</td><td>.842</td></tr><tr><td colspan="5">F1</td></tr><tr><td>Words</td><td>.408</td><td>.515</td><td>.631</td><td>.781</td></tr><tr><td>Characters</td><td>.426</td><td>.515</td><td>.628</td><td>.785</td></tr><tr><td>TF-IDF</td><td>.390</td><td>.529</td><td>.642</td><td>.755</td></tr></table>

Table 8: Text-only baselines on the test splits of Table 7: answer length in words or characters, and TF– IDF of the answer alone. The last column uses a split of HaluBench built the same way without the length filtering (2,231 test answers, 1,075 hallucinated). The 95% group-bootstrap intervals of AUROC are at most ±0.05.

Section 5.3, with regularization and threshold chosen on validation); Table 8 gives the results. On unfiltered HaluBench, TF–IDF on the answer alone reaches 0.842 AUROC: much of the label can be predicted without the context, so the unfiltered benchmark is too easy a test of contextual hallucination detection. Compared with Table 10, every proxy detector beats the text-only baselines on RAGTruth QA, where hallucinated answers are longer; on HaluBench and TRIVIA+, ${ \bf A } _ { 7 }$ alone stays within 0.04 of TF–IDF, whereas $\mathbf { A } _ { 6 4 }$ and every set with hidden states exceed it by more than 0.1.

## L Proxy detection: details

## L.1 Feature recipes

We format each input using the proxy model’s native chat template, retain the full context, question, and answer without truncation, and run inference with thinking mode disabled. We selected the feature subsets for ${ \bf A } _ { 7 }$ and $\mathrm { L _ { 9 } }$ and the pooling and dimensionality reduction settings for $\mathrm { H } _ { 6 4 }$ once, independently for each block, using grouped five-fold cross-validation on the TRIVIA+ training split with Qwen3-4B. Because the benchmarks can share source questions and contexts, we first excluded 61 TRIVIA+ training groups containing examples whose parts exactly matched the corresponding field of a test example in another benchmark. This left 2,079 training examples for the search and avoided using overlapping test content to select subsequently evaluated configurations. For each block, we chose the variant with the highest mean AUROC across the five folds, breaking ties in favor of fewer features, simpler representations, and smaller C. These feature definitions and preprocessing choices are then reused across all other datasets and proxy models without repeating the selection.

Cheap attention features $( \mathbf { A } _ { 7 } )$ . The heads of each proxy are ranked once by their induction score, without hallucination labels, and ${ \bf A } _ { 7 }$ reads the first K heads of this ranking.

The induction score of a head is its mean attention weight from each token of a repeated sequence to the token that followed its first occurrence, the weight an induction head puts on the token it copies (Olsson et al., 2022), averaged over 128 sequences (seed 42), each a beginning-of-sequence token followed by 64 random tokens twice. The ranking is a one-time preparation step and is not part of the measured extraction time.

The 22 candidate features of Section 5.2, each averaged over the selected heads, were searched with $K \in \{ 1 , 2 , 4 , 8 , 1 6 , 3 2 \}$ by recursive elimination of one feature at a time. The seven kept features defined in Appendix B are the maximum weight (mean, standard deviation, and maximum over answer tokens), effective support (minimum), entropy normalized by log $n _ { i }$ (mean and standard deviation), and the BOS sink rate. BOS is typically the top-1 sink of the selected heads.

Validation selects K = 32, 16, 32, and 16 on RAGTruth QA for the 0.6B, 1.7B, 4B, and 8B proxies correspondingly, and $K = 1$ on HaluBench with every proxy and on TRIVIA+ except with 8B $( K = 1 6 )$

All-head sink values $\mathbf { \left( A _ { 6 4 } \right) }$ . For each head, we divide each column sum by the number of answer rows that see its key, $\bar { c } _ { j } = c _ { j } / \operatorname* { m i n } ( R , L - j )$ , and keep its k largest values in descending order, without their positions. Concatenated over all heads and layers, standardized, and projected onto 64 principal components, they form $\mathbf { A } _ { 6 4 }$ . The number $k \in \{ 1 , \ldots , 1 0 \}$ is chosen jointly with a provisional regularization of $\mathbf { A } _ { 6 4 }$ alone by five-fold grouped cross-validation on the training split, with the scaler and projection refit in every fold; the chosen k is 1–4, except 9 for TRIVIA+ with 1.7B. $\mathrm { \bf A } _ { 6 4 } \mathrm { H } _ { 6 4 }$ keeps that k and reuses the projection of $_ \mathrm { H _ { 6 4 } } ;$ the final regularization and threshold of $\mathbf { A } _ { 6 4 }$ and $\mathrm { { A _ { 6 4 } H _ { 6 4 } } }$ are chosen on validation as for every other set.

Compact hidden states $\mathbf { ( H _ { 6 4 } ) }$ The block takes decoder layer 14 for 0.6B and 1.7B and layer 18 for 4B and 8B (counting from 1; hidden sizes 1024, 2048, 2560, and 4096), concatenates the percoordinate mean and population standard deviation over answer tokens, standardizes them, projects them onto the first 64 components of a randomized PCA, and standardizes again; the scalers and the PCA are fit on the training split of each dataset and proxy. The selection compared three poolings (mean; mean and standard deviation; mean, standard deviation, minimum, and maximum) with 16, 32, 64, 128, 256, or 512 principal components or no projection. The layer itself is not selected: following prior work on hidden-state hallucination signals (Azaria and Mitchell, 2023; Chen et al., 2024), we take a middle layer.

Logit features $\mathbf { ( L _ { 9 } ) }$ The 224 candidate features, the four pooled features of each statistic of Table 9, were searched by recursive feature elimination as for $\mathbf { A } _ { 7 }$ . The block keeps the standard deviation and minimum of the Zipf slope, the minimum Jensen–Shannon divergence, the mean and minimum answer-token probability, the minimum of the fourth-largest probability, and the maxima of the 7th, 10th, and 49th largest, 9 in total.

## L.2 Logit features

The logit block reads the proxy’s output logits in the same forward as the attention statistics. A hook on the final normalized hidden states applies the language-model head only at the positions that predict answer tokens, plus the one before them, so the full vocabulary distribution is never formed for the prompt. Let $q _ { t }$ be the next-token distribution at the position that predicts answer token y<sub>t</sub>, V the vocabulary, and $q _ { t } ^ { ( r ) }$ the r-th largest probability of $q _ { t }$ Table 9 defines the seven per-token statistics. Their mean, standard deviation, minimum, and maximum over the answer tokens, taken for the top-k probabilities separately at each rank $r \leq 5 0$ , are the 224 candidate features of Appendix L.1.

## L.3 Training and significance tests

The regression uses liblinear with balanced class weights. For each dataset, proxy, and feature set, $C \in \{ 0 . 0 0 1 , 0 . 0 1 , 0 . 1 , 1 , 1 0 \}$ is chosen by validation AUROC and the threshold by validation F1, with ties broken toward smaller K and C and, for the threshold, toward the higher one (Section 5.3). Models are not refit on training and validation data together. To test an addition of an attention block, we draw 100,000 paired bootstrap resamples of test groups (Table $7 ;$ seed 42), keeping models, regularization, and thresholds fixed. The two-sided p-value of an observed difference $\Delta$ is the share of centered bootstrap differences at least as large in absolute value, $p = ( 1 + \# \{ b : | \Delta _ { b } ^ { * } - \Delta | \geq$ $| \Delta | \} ) / ( B + 1 )$ with $B = 1 0 0 , 0 0 0 .$ , and we Holmcorrect these p-values over the 12 dataset–proxy pairs of each addition. With only 150–546 test groups, the intervals are wide, so only large gains stay significant. Table 11 tests each addition of ${ \bf A } _ { 7 }$ and Table 12 tests $\mathrm { H } _ { 6 4 } {  } \mathrm { A } _ { 6 4 } \mathrm { H } _ { 6 4 }$

<table><tr><td>Statistic</td><td>Definition (per answer token t)</td><td>Shape</td></tr><tr><td>Answer-token probability</td><td> $q _ { t } ( y _ { t } )$ </td><td>R</td></tr><tr><td>Answer-token rank</td><td> $1 + | \{ v \in V : q _ { t } ( v ) > q _ { t } ( y _ { t } ) \} |$ </td><td> $R$ </td></tr><tr><td>Top-k probabilities</td><td> $q _ { t } ^ { ( r ) } \mathrm { f o r e a c h r a n k } r = 1 , \ldots , 5 0$ </td><td> $R \times 5 0$ </td></tr><tr><td>Entropy</td><td> $\begin{array} { r } { - \sum _ { v \in V } q _ { t } ( v ) \log q _ { t } ( v ) } \end{array}$ </td><td>R</td></tr><tr><td>Effective vocabulary size</td><td> $\begin{array} { r } { \big ( \sum _ { v \in V } q _ { t } ( v ) ^ { 2 } \big ) ^ { - 1 } } \end{array}$ </td><td>R</td></tr><tr><td>Zipf slope</td><td>least-squares slope of  $\log q _ { t } ^ { ( r ) }$  against log  $r , r = 1 , \ldots , 5 0$ </td><td>R</td></tr><tr><td>Jensen-Shannon divergence</td><td> $\mathrm { J S } ( q _ { t } \parallel q _ { t - 1 } )$  between consecutive positions;  $q _ { t - 1 }$  of the first answer token is the last prompt position</td><td>R</td></tr></table>

Table 9: Logit statistics computed per answer token from the proxy’s next-token distribution. Shape: output for one sequence with R answer tokens.

## L.4 Full results

Table 10 gives the test AUROC of every detector of Table 2. Tables 11 and 12 test the additions of ${ \bf A } _ { 7 }$ and $\mathbf { A } _ { 6 4 }$ (Appendix L.3); bold marks the gains that stay significant after correction over the 12 pairs of an addition. The significant gains fall where each block helps most, and no tested addition significantly lowers F1.

## L.5 LLM judges

We follow a typical protocol for LLM judges. Each judge receives the context, question, and candidate answer using one of the four prompt templates in Appendix N. With thinking disabled, it produces one token, restricted to True (hallucinated) or False (supported). We use the probability assigned to True as the hallucination score. The templates contain either no demonstrations or eight labeled demonstrations. For each of the nine judges, we select the template using the three training splits combined and select the decision threshold by validation F1, as for the proxy detectors.

The four Qwen3 proxy models are prompted as judges with each template, and scored like the judges. For each model we keep the template with the best mean validation AUROC (Strict few-shot for 4B and 8B, Claims for 0.6B and 1.7B). Bespoke-MiniCheck-7B (Tang et al., 2024) and HHEM-2.1- Open (Li et al., 2024a) estimate the probability that a premise supports a claim. We use one minus this probability as the hallucination score. MiniCheck splits long premises into chunks and keeps the highest support, and we do the same for HHEM above 8,192 tokens. MiniCheck is trained on ANLI and synthetic data; the training data of HHEM is not disclosed.

The nine judges are Qwen3.5-9B (Qwen Team, 2026), Gemma-4-E4B (Gemma Team, 2026), Granite-4.2-8B (IBM Granite Team, 2026), Ministral-3-8B (Liu et al., 2026), Falcon-H1-7B (Zuo et al., 2025), Llama-3.1-8B (Dubey et al., 2024), GLM-4-9B (Team GLM, 2024), Phi-4-mini (Abouelenin et al., 2025), and Command-R7B (Cohere Labs, 2024). We report results for them in Table 13.

<table><tr><td></td><td></td><td colspan="4">RAGTruth QA</td><td colspan="4">HaluBench</td><td colspan="4">TRIVIA+</td></tr><tr><td>Features</td><td>N</td><td>0.6B</td><td>1.7B</td><td>4B</td><td>8B</td><td>0.6B</td><td>1.7B</td><td>4B</td><td>8B</td><td>0.6B</td><td>1.7B</td><td>4B</td><td>8B</td></tr><tr><td> ${ \bf A } _ { 7 }$ </td><td>7</td><td>.852</td><td>.847</td><td>.838</td><td>.856</td><td>.627</td><td>.636</td><td>.652</td><td>.660</td><td>.642</td><td>.653</td><td>.666</td><td>.645</td></tr><tr><td> $\mathrm { { L _ { 9 } } }$ </td><td>9</td><td>.764</td><td>.765</td><td>.793</td><td>.765</td><td>.654</td><td>.646</td><td>.661</td><td>.680</td><td>.684</td><td>.661</td><td>.696</td><td>.703</td></tr><tr><td> $\mathbf { A } \mathrm { { 7 L } } _ { 9 }$ </td><td>16</td><td>.852</td><td>.849</td><td>.859</td><td>.859</td><td>.663</td><td>.667</td><td>.680</td><td>.693</td><td>.686</td><td>.682</td><td>.706</td><td>.707</td></tr><tr><td> $\mathrm { H } _ { 6 4 }$ </td><td>64</td><td>.864</td><td>.855</td><td>.869</td><td>.862</td><td>.792</td><td>.824</td><td>.846</td><td>.846</td><td>.775</td><td>.786</td><td>.807</td><td>.804</td></tr><tr><td> $\mathrm { \bf A } _ { 7 } \mathrm { H } _ { 6 4 }$ </td><td>71</td><td>.871</td><td>.861</td><td>.866</td><td>.872</td><td>.795</td><td>.824</td><td>.846</td><td>.847</td><td>.777</td><td>.789</td><td>.806</td><td>.793</td></tr><tr><td> $\mathrm { H _ { 6 4 } L _ { 9 } }$ </td><td>73</td><td>.865</td><td>.856</td><td>.876</td><td>.867</td><td>.797</td><td>.823</td><td>.848</td><td>.851</td><td>.776</td><td>.791</td><td>.804</td><td>.804</td></tr><tr><td> $\mathrm { A _ { 7 } H _ { 6 4 } L _ { 9 } }$ </td><td>80</td><td>.873</td><td>.861</td><td>.877</td><td>.875</td><td>.800</td><td>.822</td><td>.848</td><td>.851</td><td>.781</td><td>.796</td><td>.799</td><td>.813</td></tr><tr><td> $\mathbf { A } _ { 6 4 }$ </td><td>64</td><td>.866</td><td>.873</td><td>.872</td><td>.880</td><td>.801</td><td>.821</td><td>.871</td><td>.877</td><td>.780</td><td>.774</td><td>.801</td><td>.803</td></tr><tr><td> $\mathrm { { A _ { 6 4 } H _ { 6 4 } } }$ </td><td>128</td><td>.866</td><td>.857</td><td>.875</td><td>.874</td><td>.814</td><td>.846</td><td>.877</td><td>.881</td><td>.796</td><td>.792</td><td>.820</td><td>.824</td></tr><tr><td>Best LLM judge</td><td></td><td></td><td>.909</td><td></td><td></td><td></td><td>.859</td><td></td><td></td><td></td><td>.786</td><td></td><td></td></tr></table>

Table 10: Test AUROC of the detectors of Table 2. Bold: best in column; underlined: second best (ties marked alike). Bottom: the best of nine LLM judges of 4–9B parameters, Qwen3.5-9B on every dataset (Table 13).

<table><tr><td></td><td></td><td colspan="4">RAGTruth QA</td><td colspan="4">HaluBench</td><td colspan="4">TRIVIA+</td></tr><tr><td></td><td>Addition</td><td>0.6B</td><td>1.7B</td><td>4B</td><td>8B</td><td>0.6B</td><td>1.7B</td><td>4B</td><td>8B</td><td>0.6B</td><td>1.7B</td><td>4B</td><td>8B</td></tr><tr><td rowspan="3"> $\Delta \mathsf { F } 1$ </td><td> $\mathrm { H } _ { 6 4 } {  } \mathrm { A } _ { 7 } \mathrm { H } _ { 6 4 }$ </td><td>+.080</td><td>+.022</td><td>-.020</td><td>+.043</td><td>+.001</td><td>-.005</td><td>+.002</td><td>+.004</td><td>-.014</td><td>.000</td><td>+.005</td><td>-.025</td></tr><tr><td> $\mathrm { L _ { 9 } {  } A _ { 7 } L _ { 9 } }$ </td><td>+.126</td><td>+.088</td><td>+.074</td><td>+.114</td><td>+.009</td><td> $+ . 0 1 5 + . 0 1 4 + . 0 0 9$ </td><td></td><td></td><td>+.011</td><td></td><td> $+ . 0 3 8 + . 0 2 5$ </td><td>+.002</td></tr><tr><td> $\mathrm { H _ { 6 4 } L _ { 9 } {  } A _ { 7 } H _ { 6 4 } L _ { 9 } }$ </td><td>+.007</td><td>+.005</td><td>-.016</td><td>+.030</td><td>+.006</td><td>-.010</td><td>+.006</td><td>+.002</td><td>+.042</td><td>+.002</td><td>+.005</td><td>+.020</td></tr><tr><td rowspan="3"> $p ,$  Holm-12</td><td> $\mathrm { H } _ { 6 4 } {  } \mathrm { A } _ { 7 } \mathrm { H } _ { 6 4 }$ </td><td>.005</td><td>.557</td><td>.605</td><td>.017</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td> $\mathrm { L _ { 9 } {  } A _ { 7 } L _ { 9 } }$ </td><td>&lt;.001</td><td>.004</td><td>.002</td><td>&lt;.001</td><td>1</td><td>1</td><td>.503</td><td>1</td><td>1</td><td>.317</td><td>1</td><td>1</td></tr><tr><td> $\mathrm { H _ { 6 4 } L _ { 9 } {  } A _ { 7 } H _ { 6 4 } L _ { 9 } }$ </td><td>1</td><td>1</td><td>.885</td><td>.065</td><td>1</td><td>.458</td><td>1</td><td>1</td><td>.174</td><td>1</td><td>1</td><td>.477</td></tr></table>

Table 11: Change in test F1 when $\mathbf { A } _ { 7 }$ is added, with two-sided p-values from 100,000 paired bootstrap resamples of test groups (seed $4 2 ;$ models, regularization, and thresholds fixed), Holm-corrected over the 12 dataset–proxy pairs of an addition (Holm-12). A gain is significant if $\Delta > 0$ and the corrected $p < 0 . 0 5 ;$ bold: significant gain. No addition gives a significant loss.

<table><tr><td></td><td></td><td colspan="4">RAGTruth QA</td><td colspan="4">HaluBench</td><td colspan="4">TRIVIA+</td></tr><tr><td></td><td>Addition</td><td>0.6B</td><td>1.7B</td><td>4B</td><td>8B</td><td>0.6B</td><td>1.7B</td><td>4B</td><td>8B</td><td>0.6B</td><td>1.7B</td><td>4B</td><td>8B</td></tr><tr><td>∆F1</td><td> $\mathrm { H } _ { 6 4 } {  } \mathrm { A } _ { 6 4 } \mathrm { H } _ { 6 4 }$ </td><td>+.044</td><td>-.030</td><td>-.001</td><td>+.037</td><td>+.027</td><td>+.012</td><td>+.035</td><td>+.050</td><td>+.009</td><td>+.002</td><td>+.040</td><td>+.001</td></tr><tr><td>p, Holm-12</td><td> $\mathrm { H } _ { 6 4 } {  } \mathrm { A } _ { 6 4 } \mathrm { H } _ { 6 4 }$ </td><td>.201</td><td>.324</td><td>1</td><td>.269</td><td>.198</td><td>1</td><td>.027</td><td>&lt;.001</td><td>1</td><td>1</td><td>.189</td><td>1</td></tr></table>

Table 12: Change in test F1 when $\mathbf { A } _ { 6 4 }$ is added to $\mathrm { H } _ { 6 4 }$ (Section 5.4), with two-sided p-values from 100,000 paired bootstrap resamples of test groups as in Table 11, Holm-corrected over the 12 dataset–proxy pairs (Holm-12). Bold: significant gain. No change is a significant loss.

<table><tr><td colspan="2"></td><td colspan="3">AUROC</td><td colspan="3">F1</td></tr><tr><td>Judge</td><td>Prompt</td><td>RAGTruth</td><td>HaluB.</td><td>TRIVIA+</td><td>RAGTruth</td><td>HaluB.</td><td>TRIVIA+</td></tr><tr><td>Qwen3.5-9B</td><td>Strict few-shot</td><td>.909</td><td>.859</td><td>.786</td><td>.623</td><td>.755</td><td>.655</td></tr><tr><td>Gemma-4-E4B</td><td>Evidence</td><td>.878</td><td>.855</td><td>.774</td><td>.617</td><td>.751</td><td>.621</td></tr><tr><td>Granite-4.2-8B</td><td>Strict</td><td>.843</td><td>.848</td><td>.755</td><td>.547</td><td>.753</td><td>.645</td></tr><tr><td>Ministral-3-8B</td><td>Strict</td><td>.823</td><td>.817</td><td>.735</td><td>.517</td><td>.721</td><td>.613</td></tr><tr><td>Falcon-H1-7B</td><td>Strict few-shot</td><td>.752</td><td>.834</td><td>.747</td><td>.434</td><td>.726</td><td>.624</td></tr><tr><td>Llama-3.1-8B</td><td>Strict</td><td>.688</td><td>.754</td><td>.688</td><td>.363</td><td>.684</td><td>.564</td></tr><tr><td>GLM-4-9B</td><td>Strict few-shot</td><td>.725</td><td>.753</td><td>.683</td><td>.399</td><td>.678</td><td>.587</td></tr><tr><td>Phi-4-mini</td><td>Claims</td><td>.550</td><td>.771</td><td>.664</td><td>.306</td><td>.693</td><td>.566</td></tr><tr><td>Command-R7B</td><td>Evidence</td><td>.501</td><td>.604</td><td>.624</td><td>.302</td><td>.634</td><td>.536</td></tr><tr><td>Qwen3-8B, zero/few-shot</td><td>Strict few-shot</td><td>.814</td><td>.835</td><td>.752</td><td>.521</td><td>.742</td><td>.638</td></tr><tr><td>Qwen3-4B, zero/few-shot</td><td>Strict few-shot</td><td>.769</td><td>.799</td><td>.753</td><td>.439</td><td>.724</td><td>.631</td></tr><tr><td>Qwen3-1.7B, zero/few-shot</td><td>Claims</td><td>.443</td><td>.510</td><td>.433</td><td>.309</td><td>.622</td><td>.492</td></tr><tr><td>Qwen3-0.6B, zero/few-shot</td><td>Claims</td><td>.469</td><td>.473</td><td>.495</td><td>.306</td><td>.622</td><td>.516</td></tr><tr><td>Bespoke-MiniCheck-7B</td><td></td><td>.881</td><td>.822</td><td>.751</td><td>.578</td><td>.717</td><td>.616</td></tr><tr><td>HHEM-2.1-Open</td><td></td><td>.852</td><td>.703</td><td>.726</td><td>.563</td><td>.663</td><td>.604</td></tr><tr><td> $\mathrm { Q w e n 3 - 0 . 6 B , A _ { 7 } H _ { 6 4 } L _ { 9 } }$ </td><td></td><td>.873</td><td>.800</td><td>.781</td><td>.621</td><td>.702</td><td>.641</td></tr><tr><td> $\mathrm { Q w e n 3 - 4 B , A _ { 7 } H _ { 6 4 } L _ { 9 } }$ </td><td></td><td>.877</td><td>.848</td><td>.799</td><td>.528</td><td>.755</td><td>.646</td></tr><tr><td> $\mathrm { Q w e n 3 - 0 . 6 B , A _ { 6 4 } H _ { 6 4 } }$ </td><td></td><td>.866</td><td>.814</td><td>.796</td><td>.601</td><td>.724</td><td>.636</td></tr><tr><td> $\mathrm { Q w e n 3 - } 4 \mathrm { B } , \mathrm { A } _ { 6 4 } \mathrm { H } _ { 6 4 }$ </td><td></td><td>.875</td><td>.877</td><td>.820</td><td>.566</td><td>.776</td><td>.674</td></tr></table>

Table 13: LLM judges on the test splits of Table 7 with F1 at the threshold selected on validation. Top: nine instruction-tuned LLMs prompted as judges. Middle: the four Qwen3 proxy backbones prompted as judges and two specialized grounding checkers. Bottom: our proxy detectors for reference.

## M Cross-domain transfer on HaluBench

Section 5 fits and tests each detector within one dataset. Attention detectors that use few or no labels, however, are meant to work on domains they were not tuned on. We test whether they do across the six sources of HaluBench, with five published detectors: Lookback Lens, LLM-Check, RAUQ, and INTRYGUE, whose statistics our kernels compute (Table 3), and TOHA, which needs the materialized response block, plus the sink-rate term of INTRYGUE alone.

Protocol. Qwen3-8B reads the context, question, and existing answer of every example in our filtered HaluBench split (Appendix K), teacher-forced with its chat template and thinking off. One forward extracts the features of all methods, with our kernels for the statistics of Table 3 and the materialized response block for TOHA (T4); we then fit and test them on CPU. Each of the six sources has 138–166 test examples. RAUQ needs no labels (Vazhentsev et al., 2026). INTRYGUE (Kuleshova et al., 2026), its sink-rate term alone, TOHA (Bazarova et al., 2026), LLM-Check (Sriramanan et al., 2024), and our variant of Lookback Lens, which averages the lookback ratios of at most 10 heads instead of training a probe on all heads (Chuang et al., 2024), choose one setting by AUROC on the validation split of the fitting sources, about 150 labels per source: the number of induction heads, the layer, or the heads.

Each method is fit on one source and tested on all six, fit on the five sources other than the test domain (leave-one-source-out), and fit on all six. Table 14 gives the test AUROC of every setting; 95% bootstrap intervals use 1,000 resamples of individual test examples rather than of the context groups used in Section 5, so they are somewhat narrower than group intervals would be.

Results. RAUQ and INTRYGUE transfer; detectors that select heads or a layer on one source do not. Averaged over the six fitting sources, RAUQ and INTRYGUE lose at most 0.01 AUROC from the fitting source to the other five, and the sink rate 0.04, whereas LLM-Check loses 0.05, TOHA 0.06, and Lookback Lens 0.15, which falls to chance (0.51). Fitting on more sources does not help much, since each detector still selects only one setting. Two domains defeat every method: a fixed-direction score cannot adapt where its direction reverses, so INTRYGUE stays at 0.38–0.41 on DROP from every source, and no method exceeds 0.53 on FinanceBench.

<table><tr><td rowspan="2">Fit on</td><td rowspan="2">Method</td><td colspan="6">Test domain</td><td rowspan="2">Mean</td></tr><tr><td>RAGTruth</td><td>CovidQA</td><td>PubMedQA</td><td>FinanceBench</td><td>DROP</td><td>HaluEval</td></tr><tr><td colspan="2">RAUQ (no fitting)</td><td>.58</td><td>.67</td><td>.66</td><td>.52</td><td>.66</td><td>.85</td><td>.66</td></tr><tr><td rowspan="5">RAGTruth</td><td>INTRYGUE</td><td>.89</td><td>.77</td><td>.55</td><td>.48</td><td>.41</td><td>.92</td><td>.62</td></tr><tr><td>Sink rate</td><td>.80</td><td>.72</td><td>.53</td><td>.49</td><td>.56</td><td>.91</td><td>.64</td></tr><tr><td>LLM-Check</td><td>.59</td><td>.58</td><td>.62</td><td>.53</td><td>.53</td><td>.75</td><td>.60</td></tr><tr><td>TOHA</td><td>.68</td><td>.59</td><td>.44</td><td>.46</td><td>.61</td><td>.80</td><td>.58</td></tr><tr><td>Lookback Lens</td><td>.88</td><td>.56</td><td>.47</td><td>.51</td><td>.49</td><td>.73</td><td>.55</td></tr><tr><td rowspan="5">CovidQA</td><td>INTRYGUE</td><td>.89</td><td>.77</td><td>.55</td><td>.48</td><td>.41</td><td>.92</td><td>.65</td></tr><tr><td>Sink rate</td><td>.51</td><td>.62</td><td>.52</td><td>.44</td><td>.55</td><td>.78</td><td>.56</td></tr><tr><td>LLM-Check</td><td>.44</td><td>.55</td><td>.61</td><td>.45</td><td>.56</td><td>.59</td><td>.53</td></tr><tr><td>TOHA</td><td>.68</td><td>.59</td><td>.44</td><td>.46</td><td>.61</td><td>.80</td><td>.60</td></tr><tr><td>Lookback Lens</td><td>.31</td><td>.65</td><td>.40</td><td>.47</td><td>.53</td><td>.57</td><td>.45</td></tr><tr><td rowspan="5">PubMedQA</td><td>INTRYGUE</td><td>.89</td><td>.75</td><td>.55</td><td>.47</td><td>.40</td><td>.92</td><td>.69</td></tr><tr><td>Sink rate</td><td>.51</td><td>.62</td><td>.52</td><td>.44</td><td>.54</td><td>.82</td><td>.59</td></tr><tr><td>LLM-Check</td><td>.65</td><td>.60</td><td>.68</td><td>.44</td><td>.59</td><td>.71</td><td>.60</td></tr><tr><td>TOHA</td><td>.39</td><td>.59</td><td>.50</td><td>.53</td><td>.51</td><td>.43</td><td>.49</td></tr><tr><td>Lookback Lens</td><td>.80</td><td>.40</td><td>.53</td><td>.52</td><td>.50</td><td>.48</td><td>.54</td></tr><tr><td rowspan="5">FinanceBench</td><td>INTRYGUE</td><td>.82</td><td>.70</td><td>.54</td><td>.47</td><td>.38</td><td>.86</td><td>.66</td></tr><tr><td>Sink rate</td><td>.51</td><td>.62</td><td>.52</td><td>.44</td><td>.54</td><td>.82</td><td>.60</td></tr><tr><td>LLM-Check</td><td>.42</td><td>.61</td><td>.49</td><td>.45</td><td>.56</td><td>.78</td><td>.57</td></tr><tr><td>TOHA</td><td>.68</td><td>.54</td><td>.44</td><td>.48</td><td>.57</td><td>.55</td><td>.55</td></tr><tr><td>Lookback Lens</td><td>.68</td><td>.45</td><td>.41</td><td>.51</td><td>.54</td><td>.37</td><td>.49</td></tr><tr><td rowspan="5">DROP</td><td>INTRYGUE</td><td>.82</td><td>.70</td><td>.54</td><td>.47</td><td>.39</td><td>.87</td><td>.68</td></tr><tr><td>Sink rate</td><td>.72</td><td>.68</td><td>.54</td><td>.44</td><td>.56</td><td>.86</td><td>.65</td></tr><tr><td>LLM-Check</td><td>.44</td><td>.63</td><td>.45</td><td>.47</td><td>.58</td><td>.81</td><td>.56</td></tr><tr><td>TOHA</td><td>.66</td><td>.52</td><td>.41</td><td>.49</td><td>.61</td><td>.59</td><td>.54</td></tr><tr><td>Lookback Lens</td><td>.66</td><td>.40</td><td>.37</td><td>.53</td><td>.58</td><td>.41</td><td>.47</td></tr><tr><td rowspan="5">HaluEval</td><td>INTRYGUE</td><td>.89</td><td>.75</td><td>.55</td><td>.47</td><td>.40</td><td>.92</td><td>.61</td></tr><tr><td>Sink rate</td><td>.73</td><td>.74</td><td>.51</td><td>.49</td><td>.56</td><td>.93</td><td>.61</td></tr><tr><td>LLM-Check</td><td>.59</td><td>.63</td><td>.50</td><td>.46</td><td>.57</td><td>.88</td><td>.55</td></tr><tr><td>TOHA</td><td>.68</td><td>.59</td><td>.44</td><td>.46</td><td>.60</td><td>.81</td><td>.56</td></tr><tr><td>Lookback Lens</td><td>.79</td><td>.52</td><td>.39</td><td>.52</td><td>.42</td><td>.82</td><td>.53</td></tr><tr><td rowspan="5">Other five</td><td>INTRYGUE</td><td>.89</td><td>.77</td><td>.55</td><td>.48</td><td>.41</td><td>.92</td><td>.67</td></tr><tr><td>Sink rate</td><td>.80</td><td>.72</td><td>.53</td><td>.49</td><td>.56</td><td>.91</td><td>.67</td></tr><tr><td>LLM-Check</td><td>.59</td><td>.63</td><td>.45</td><td>.46</td><td>.56</td><td>.71</td><td>.57</td></tr><tr><td>TOHA</td><td>.68</td><td>.59</td><td>.44</td><td>.46</td><td>.60</td><td>.55</td><td>.55</td></tr><tr><td>Lookback Lens</td><td>.71</td><td>.54</td><td>.40</td><td>.47</td><td>.53</td><td>.52</td><td>.53</td></tr><tr><td rowspan="5">All six</td><td>INTRYGUE</td><td>.89</td><td>.77</td><td>.55</td><td>.48</td><td>.41</td><td>.92</td><td>.67</td></tr><tr><td>Sink rate</td><td>.80</td><td>.72</td><td>.53</td><td>.49</td><td>.56</td><td>.91</td><td>.67</td></tr><tr><td>LLM-Check</td><td>.59</td><td>.63</td><td>.50</td><td>.46</td><td>.57</td><td>.88</td><td>.60</td></tr><tr><td>TOHA</td><td>.68</td><td>.59</td><td>.44</td><td>.46</td><td>.60</td><td>.81</td><td>.60</td></tr><tr><td>Lookback Lens</td><td>.36</td><td>.54</td><td>.40</td><td>.52</td><td>.60</td><td>.48</td><td>.48</td></tr></table>

Table 14: Cross-domain transfer on HaluBench (Qwen3-8B reader): test AUROC of each method fit on the source of its row group and tested on each source. Shaded: same source. Otherfive: fit on the five sources other than the test domain (leave-one-source-out); All six: fit on all six sources (pooled). Mean: over the five other test domains for a single fitting source, over all six for the last two groups. RAUQ fits nothing and scores the same for every fitting source. Bold: best method of the group in each column (ties marked alike). Each test domain has 138–166 examples; the 95% bootstrap intervals span ±0.09 at the median (±0.03 to ±0.12).

## N LLM judge prompts

Each judge of Table 13 reads one of the four templates below, given verbatim. ### system and ### user open the system and user messages of the judge’s chat template; {context}, {question}, and {response} are filled with the example. Every template asks for the verdict True (hallucinated) or False (supported).

## Evidence (Gemma-4-E4B, Command-R7B).

### user   
You are assessing a candidate answer to a question.   
You are given the question, reference evidence,,→   
and one candidate answer.,→   
# Question   
{question}   
# Reference evidence   
{context}   
# Candidate answer   
{response}   
Definition. A hallucinated answer contains   
information that is incorrect, fabricated, or,→   
not supported by the evidence: for example, it,→   
,→ misreads the question, reverses a finding,   
,→ invents a mechanism, or adds specifics absent   
,→ from the source. A supported answer is fully   
,→ consistent with the evidence.   
Is the candidate answer hallucinated? Answer with   
exactly one word: True if it is hallucinated or,→   
unsupported, False if it is fully supported.,→

## Claims (Phi-4-mini).

```markdown
### system
You check model responses against a source
,→ document. Use only the document, never outside
,→ knowledge. A response passes only if every
,→ factual claim in it is stated in, or directly
,→ entailed by, the document. Any invented,
,→ altered, or unverifiable detail makes it fail,
,→ even if the rest is correct. Leaving
,→ information out does not make it fail.
### user
Document:

{context}
n n n
Request: {question}
Response:
" n n
{response}
n n n
Does the response contain any claim that the
,→ document does not support? Answer True (it
,→ does, the response is hallucinated) or False
,→ (every claim is supported).
```  
Strict (Granite-4.2-8B, Ministral-3-8B, Llama-3.1-8B) is Strict few-shot without its block of examples, from Examples: up to Output format:.

Strict few-shot (Qwen3.5-9B, Falcon-H1-7B, GLM-4-9B). The prompt contains eight authorwritten illustrative examples: four hallucinated and four faithful.

### system   
You are an expert evaluator specialized in   
,→ detecting hallucinations. Use only the CONTEXT,   
,→ never outside knowledge.   
Your task is to determine whether the RESPONSE   
,→ contains any information that the CONTEXT does   
,→ not support, or answers the QUESTION   
,→ incorrectly according to the CONTEXT.   
A response is faithful only if every factual claim   
,→ in it is stated in, or directly entailed by,   
→ the context. Any invented, altered, or   
,→ unverifiable detail makes it hallucinated, even   
→   
True means HALLUCINATED:   
At least one claim is not stated in or entailed   
→ by the context, including plausible but   
,→ unstated details and claims more specific than   
→ the context allows.   
The response invents or alters entities, numbers,   
dates, policies, events, or relationships.   
The response gives a number, name, or date that   
,→ differs from what the context implies for this   
,→ question, including wrong counting or   
,→ arithmetic over values in the context.   
The response answers a different question than   
→ the one asked (wrong entity, team, period, or   
unit).   
The response refuses to answer although the   
,→ context clearly contains the answer.   
False means FAITHFUL:   
Every factual claim is supported by the context   
,→ and the response correctly answers the   
,→ question.   
The response declines to answer because the   
,→ context does not contain the required   
,→ information; such a refusal is faithful.   
Important rules:   
- One unsupported or incorrect claim is enough to   
,→ mark the response as hallucinated.   
Leaving information out is not a hallucination.   
A plausible or well-known fact is still   
hallucinated if the context does not support   
it.   
For short answers, recompute or re-locate the   
,→ answer in the context and compare exactly.   
Examples:   
<context>The museum opened in 1998 and has three   
,→ floors of galleries.</context>   
<question>When did the museum open?</question>   
<response>The museum opened in 1998 and attracts   
,→ over a million visitors a year.</response>   
Answer: True   
(The visitor number is not in the context: an added   
,→ unsupported detail.)   
<context>In the first half the Hawks scored 14   
,→ points and the Owls scored 10. In the second   
,→   
,→ 21.</context>   
<question>How many points did the Owls score in   
,→ total?</question>   
<response>24</response>   
Answer: True   
(10 + 21 = 31: wrong arithmetic over the context.)   
<context>Maria Lopez founded the company in 2004.   
,→ Her brother Diego joined as CFO in   
2010.</context>,→   
<question>Who founded the company?</question>   
<response>Diego Lopez founded the   
,→ company.</response>   
Answer: True   
(The response names the wrong person, contradicting   
,→ the context.)   
<context>The trial found that the drug lowered   
,→ blood pressure in 62% of patients but had no   
effect on heart rate.</context>,→   
<question>What effect did the drug have on heart   
rate?</question>

<response>The drug noticeably slowed patients'   
,→ heart rate.</response>   
Answer: True   
(The context says there was no effect: the response   
,→ reverses the finding.)   
<context>The bridge, completed in 1932, spans 503   
meters and carries both trains and,→   
,→ cars.</context>   
<question>How long is the bridge?</question>   
<response>It is about half a kilometer long, 503   
,→ meters.</response>   
Answer: False   
(A faithful paraphrase with an equivalent unit.)   
<context>The library offers free Wi-Fi, study   
,→ rooms, a children's section, and a   
cafe.</context>,→   
<question>What services does the library   
,→ offer?</question>   
<response>The library has free Wi-Fi and study   
,→ rooms.</response>   
Answer: False   
(Incomplete but every stated claim is supported;   
,→ omissions are not hallucinations.)   
<context>The report covers the company's sales in   
,→ Europe for 2021.</context>   
<question>What were the company's sales in Asia in   
,→ 2021?</question>   
<response>The context does not contain information   
,→ about sales in Asia.</response>   
Answer: False   
(A correct refusal: the answer is not in the   
,→ context.)   
<context>Team A won 3 games in March and 5 games in   
,→ April.</context>   
<question>How many games did Team A win in March   
,→ and April combined?</question>   
<response>8</response>   
Answer: False   
(3 + 5 = 8: correctly derived from the context.)   
Output format:   
Reply with exactly one word: True or False.   
- True = the response is hallucinated; False = the   
,→ response is faithful.   
- Do not write any explanation, punctuation,   
quotes, tags, or other text before or after the,→   
word.,→   
### user   
<context>   
{context}   
</context>   
<question>   
{question}   
</question>   
<response>   
{response}   
</response>   
Answer:

## O Use of scientific artifacts

Table 15 lists the licences of the datasets and models we use. We use all of them for non-commercial research, consistent with their terms, and redistribute none of them; this matters for the noncommercial licences and for TRIVIA+, whose licence forbids derivatives. Our anonymized code is released under the MIT licence.

AI assistants were used to proofread and refine the text for clarity.

<table><tr><td>Artifact</td><td>Use</td><td>Licence</td></tr><tr><td>RAGTruth</td><td>detection</td><td>MIT</td></tr><tr><td>HaluBench</td><td>detection</td><td>CC BY-NC 2.0</td></tr><tr><td>TRIVIA+</td><td>detection</td><td>CC BY-NC-ND 4.0</td></tr><tr><td>PHANTOM</td><td>kernels</td><td>Apache 2.0</td></tr><tr><td>Qwen3 0.6B-8B</td><td>proxy, judge</td><td>Apache 2.0</td></tr><tr><td>Qwen3.5-9B</td><td>judge</td><td>Apache 2.0</td></tr><tr><td>Gemma-4-E4B</td><td>judge</td><td>Apache 2.0</td></tr><tr><td>Granite-4.2-8B</td><td>judge</td><td>Apache 2.0</td></tr><tr><td>Ministral-3-8B</td><td>judge</td><td>Apache 2.0</td></tr><tr><td>Phi-4-mini</td><td>judge</td><td>MIT</td></tr><tr><td>Llama-3.1-8B</td><td>judge</td><td>Llama 3.1 License</td></tr><tr><td>Falcon-H1-7B</td><td>judge</td><td>Falcon LLM</td></tr><tr><td>GLM-4-9B</td><td>judge</td><td>GLM-4</td></tr><tr><td>Command-R7B</td><td>judge</td><td>CC BY-NC 4.0</td></tr><tr><td>Bespoke-MiniCheck-7B</td><td>checker</td><td>CC BY-NC 4.0</td></tr><tr><td>HHEM-2.1-Open</td><td>checker</td><td>Apache 2.0</td></tr></table>

Table 15: Licences of the artifacts we use, as stated by their releases; Falcon LLM and GLM-4 are the custom licences of these models. Detection: Section 5; kernels: Section 4; judges and grounding checkers (checker): Appendix L.5, where the models are cited.