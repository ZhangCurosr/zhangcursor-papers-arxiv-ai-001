# EFFICIENT SPECULATIVE DECODING WITH A FIXED-COST PARALLEL DRAFTER

Hao-Yuan He<sup>∗</sup> <sup>†</sup>, Peng-Fei Liu<sup>∗</sup>, Si Shen, Ming Li

Page Code Models

## ABSTRACT

Speculative decoding accelerates autoregressive inference by verifying multiple draft tokens in a single target forward pass. However, as the context grows, existing state-of-the-art drafters become increasingly expensive, eroding the very efficiency advantage they are designed to provide. We argue that this scaling is unnecessary. A standalone language model must grow with its prefix because it is solely responsible for every token it produces. A drafter, by contrast, only proposes candidates; the target catches and corrects every error before any token is committed. The drafter’s decoding cost can therefore be made entirely independent of the prefix length. We introduce LONGSPARK, a block-diffusion drafter that achieves this by extracting fixed-size, multiscale views from the target’s verification pass, thereby eliminating the need for a growing persistent state. Extensive evaluations demonstrate that LONGSPARK achieves state-of-the-art end-to-end efficiency across multiple model scales and realistic serving conditions. Notably, it delivers the lowest time-per-output-token on long-context tasks while reducing the drafter’s context state by several orders of magnitude.

![](images/f81f0152537654b6146ab41da41bca9e7dcd0d5583169a52af5530e70f19f2e2.jpg)

![](images/9fcae981e288e3957750a4ef5eaf35420f7a0efd4f7d34629a22c33af62f50b7.jpg)  
Figure 1: LONGSPARK: Scalable speculative decoding via fixed-cost drafting. Its (1) drafting cost keeps drafting time nearly constant as the prefix grows, with 406× smaller drafter context state than DSpark at 128K tokens (left). Drafting time includes proposal and sampling. Lower drafting overhead improves end-to-end throughput on LongSpec (right). Both panels use Qwen3-8B with concurrency 16.

## 1 INTRODUCTION

Speculative decoding (Leviathan et al., 2023) accelerates autoregressive inference by delegating token proposal to a lightweight drafter, which allows a target model to verify multiple candidates in a single forward pass. The core premise of this acceleration is that the drafter must be significantly cheaper than the target for the speedup to be meaningful. However, the cost of producing draft candidates typically scales with the context length, eroding the efficiency advantage exactly when it is most needed.

This scaling bottleneck is systemic across current drafting architectures. Autoregressive drafters maintain KV caches that grow linearly with the prefix; while reusing target representations (Li et al., 2024; 2025) can strengthen proposals, it does not eliminate this growing state. Windowed alternatives (Yang et al., 2026) bound their own state but still require full-prefix traversals of the target’s context. Even advanced block-diffusion drafters, which predict multiple positions jointly (Chen et al., 2026; Nguyen et al., 2026), typically inject target features from the entire prefix into every draft layer (Chen et al., 2026; Cheng et al., 2026; Huang et al., 2026; Zhang et al., 2026). In each case, the drafter inherits the (�) scaling behavior of the target model, transforming the drafter from a lightweight accelerator into a scaling bottleneck in its own right.

But a drafter need not scale this way. A standalone language model must represent the full prefix faithfully because it bears sole responsibility for every token it produces; any information lost from the prefix becomes an error. A drafter, by contrast, operates under a different contract: the target verifies every proposal before any token is committed, catching and correcting errors via rejection sampling. A wrong proposal merely adds a decoding round, never an incorrect output. The drafter can therefore work from a compressed, lossy view of the prefix.

This observation shifts the fundamental design objective for drafting: the true measure of efficiency is the number of accepted tokens produced relative to the drafting overhead. When a drafter maintains sufficient context to generate useful proposals at a constant cost, its per-round overhead decouples from the prefix length—a principle we termfixed-cost drafting.

We achieve this with LONGSPARK, a block-diffusion drafter. It extracts all context from the target’s verification pass as fixed-size, multiscale views: a single-position anchor at the decoding boundary, a short window of recent token-level detail, and a global context summary that compresses the entire prefix into a fixed number of entries. By integrating these views into a parallel block-prediction framework, LONGSPARK ensures that target verification is the only full-prefix traversal in each round. Every other operation—from context extraction to iterative refinement—remains strictly �(1) relative to the confirmed prefix.

We evaluate LONGSPARK across reasoning, coding, and dialogue benchmarks at multiple model scales, confirming that it achieves the highest end-to-end throughput, delivering 1.88× to 2.13× speedups that grow with target size. Indeed, LONGSPARK outperforms state-of-the-art drafters in throughput even when they achieve longer accepted lengths, validating that strictly bounded overhead is more critical for overall latency than marginal gains in proposal quality. Across three longcontext benchmarks up to 128K tokens and all three target scales, LONGSPARK delivers the lowest mean time per output token while reducing drafter state by several orders of magnitude.

The contributions of this work are:

• We identify an asymmetry between drafter and target: because verification guarantees output correctness, the drafter’s decoding cost need not scale with the prefix length. We formalize this as fixed-cost drafting.

• We introduce LONGSPARK, a block-diffusion drafter that extracts all of its context as fixed-size, multiscale views of the target’s verification pass, making every additional per-round operation �(1) with respect to the prefix length.

• We empirically validate fixed-cost drafting across diverse benchmarks and model scales, demonstrating that LONGSPARK improves end-to-end throughput while reducing drafter memory overhead by several orders of magnitude.

![](images/8a2d562d1b459fa2d43a1d0ff3ce455aafd1803c9c075444db3c452d3c5d89ea.jpg)  
Figure 2: One decoding round of LONGSPARK. (a) The target verifies the proposed tokens. (b) Three fixed-size context views are extracted from the target’s state: a boundary state, a recent KV window, and a global context summary. (c) The drafter uses these views to propose the next token block at a cost independent of prefix length.

## 2 LONGSPARK: FIXED-COST SPECULATIVE DRAFTING

Fixed-cost drafting requires two ingredients: a context interface that extracts bounded-size views from an unbounded prefix, and a proposal model that produces competitive drafts from these views alone. We formalize both below, after establishing the necessary background (§ 2.1). The context interface (§ 2.2) and proposal model (§ 2.3) are followed by training, inference, and complexity analysis (§ 2.4).

## 2.1 PRELIMINARIES

Speculative decoding. Speculative decoding (Leviathan et al., 2023) pairs a target model with a lightweight drafter. At the start of a round, the target has cached $x _ { 1 : t }$ but has yet to process its latest committed token $\mathsf { a } = x _ { t + 1 }$ . Given the committed prefix $s = ( x _ { 1 : t } , \mathsf { a } )$ , the drafter proposes � candidates $y _ { 1 : B }$ by sampling from distributions $q _ { i } ( \cdot ) = q ( \cdot \mid s , y _ { < i } )$ . The target verifies these by processing $\left[ \mathsf { a } , y _ { 1 : B } \right]$ in a single forward pass to compute $p _ { i } ( \cdot ) = p ( \cdot \mid s , y _ { < i } )$ . Candidates are tested in order, accepting �<sub>�</sub> with probability

$$
\alpha _ { i } = \operatorname* { m i n } \left( 1 , { \frac { p _ { i } ( y _ { i } ) } { q _ { i } ( y _ { i } ) } } \right) .\tag{1}
$$

At the first rejected position �, we discard $y _ { i : B }$ and sample a correction token from

$$
\mathcal { P } _ { i } ^ { \mathrm { c o r r } } ( v ) = \frac { \left[ \phi _ { i } ( v ) - q _ { i } ( v ) \right] _ { + } } { \sum _ { u } [ \phi _ { i } ( u ) - q _ { i } ( u ) ] _ { + } } , \qquad [ z ] _ { + } = \operatorname* { m a x } ( z , 0 ) .\tag{2}
$$

Each round adds one token beyond the accepted candidates: a correction token upon rejection, or a sample from $p ( \cdot \mid s , y _ { 1 : B } )$ when all � candidates are accepted. Accepting � candidates therefore advances the prefix by $r + 1$ tokens while preserving the target distribution. With mean drafting and verification latencies $T _ { \mathrm { d r a f t } }$ and $T _ { \mathrm { v e r i f y } }$ and mean advancement $\tau = \mathbb { E } [ r + 1 ]$ , the average decoding time per token is modeled as

$$
L = \frac { T _ { \mathrm { d r a f t } } + T _ { \mathrm { v e r i f y } } } { \tau } .\tag{3}
$$

Block-diffusion drafting. Single-step block-diffusion drafters such as DFlash (Chen et al., 2026) reduce $T _ { \mathrm { d r a f t } }$ by predicting a block of proposal distributions in one denoising pass. Let ${ \mathit { C } } ( s )$ denote context features extracted from the target and $\pmb { u } _ { 1 : B }$ denote input representations for the proposal slots, such as mask embeddings or available committed-token embeddings. A bidirectional draft network $F _ { \theta }$ produces base logits jointly:

$$
z _ { 1 : B } ^ { ( 0 ) } = F _ { \theta } ( { u _ { 1 : B } } ; C ( s ) ) , \qquad q _ { i } ^ { ( 0 ) } = \mathrm { s o f t m a x } ( z _ { i } ^ { ( 0 ) } ) .\tag{4}
$$

![](images/1c1719bce031c48b1abfe423e9f7af19948432c7ce62eff5bd7fd721394d45b1.jpg)  
Figure 3: Incremental global context summary during inference. The summary is initialized over the full prefix during target prefill, then updated using only newly retained KV rows � $( \Delta = | N | \le B + 1 )$ .

Sampling each position from these distributions yields the factorized proposal $q ^ { ( 0 ) } ( y _ { 1 : B } \mid s ) =$ $\textstyle \prod _ { i = 1 } ^ { B } q _ { i } ^ { ( 0 ) } ( y _ { i } \mid s )$ . The backbone mixes information across positions through bidirectional attention, while tokens are sampled independently from its outputs. Semi-autoregressive variants (Cheng et al., 2026) add lightweight token dependencies after this parallel pass. The cost of this pass still grows with prefix length when attention spans the full prefix. The remaining question is whether a fixed-size context view ${ \mathit { C } } ( s )$ can retain competitive proposal quality.

## 2.2 FIXED-COST CONTEXT INTERFACE

We instantiate ${ \mathit { C } } ( s )$ with three fixed-size views that capture the target’s cached state at complementary temporal scales. A boundary state anchors the current generation point, fusing target hidden representations $\{ h _ { t } ^ { \ell } \} _ { \ell \in \mathcal { L } }$ at the decoding boundary into a single conditioning vector $\mathbf { c } _ { t }$ via learned projection. A recent KV window of � entries per selected layer preserves token-level detail in the local neighborhood. A global context summary compresses the entire prefix into � entries per layer and head. All three are extracted from a subset  of evenly spaced target layers (Appendix B.1) and involve only constant-size reads and transformations.

Global context summary. The summary must compress the entire prefix into a fixed number of entries while remaining incrementally updatable as new tokens are committed. To achieve this, each selected layer and attention head maintains a bank of � learned summary queries $G \in \mathbb { R } ^ { R \times d }$ which are independent of the prefix and held fixed during inference (Appendix B.2). For a head of dimension � and corresponding target keys and values $K _ { 1 : t }$ and $V _ { 1 : t }$ , the summary values are computed as

$$
\begin{array} { r } { { \cal M } _ { t } = \mathrm { A t t n } ( G , K _ { 1 : t } , V _ { 1 : t } ) . } \end{array}\tag{5}
$$

Because � remains fixed during inference, historical attention contributions remain valid as the context grows, and only the newly retained target KV rows need to be incorporated. After verification, the summary absorbs at most $B + 1$ new rows (Figure 3), keeping update cost and stored state independent of �. The drafter reads (�, �<sub>�</sub>) as summary keys and values, respectively: the same queries that produced the summary now serve as retrieval keys for the draft layers. Appendix A.2 gives the incremental update algorithm.

## 2.3 PROPOSAL MODEL

We instantiate the parallel backbone in (4) with the fixed-cost interface from § 2.2, then introduce token-level dependencies through a lightweight sequential correction. The result is a �-layer Transformer that processes all � draft positions in parallel, followed by a per-position Markov correction that conditions each token on its predecessor.

The backbone operates on � input slots, where the first encodes the committed token a and the remaining � − 1 employ mask-token embeddings. Their embeddings are combined with withinblock positional encodings and the boundary state $\mathbf { c } _ { t }$ to initialize the draft representations. The output of slot � supplies the base logits for proposal $y _ { i } ,$ for $i = 1 , \dots , B$ (Appendix A.1).

These representations are processed by $D = | { \boldsymbol { L } } |$ Transformer layers, where each draft layer � is paired with a corresponding selected target layer $\ell ( j )$ , from which it reads the recent window and global summary. At each layer, attention is computed over a concatenated context:

$$
\begin{array} { r } { K = [ K ^ { \mathrm { b } } ; K ^ { \mathrm { w } } ; G ] , } \\ { V = [ V ^ { \mathrm { b } } ; V ^ { \mathrm { w } } ; M _ { t } ] , } \end{array}\tag{6}
$$

where superscripts b and w denote the proposal block and recent window, respectively; layer and head indices are omitted for readability. Consequently, all block positions attend to one another through a fixed-size context of at most $\dot { B } + W + R$ entries per head, ensuring the computation remains independent of the prefix length.

Following the � draft layers, the frozen Target normalization and vocabulary head generate base logits $z _ { i } ^ { ( 0 ) }$ for all � positions in parallel. These are then refined by a low-rank sequential correction (Cheng et al., 2026) that conditions each position on the token sampled at its predecessor. With $y _ { 0 } = \mathtt { a }$ , a correction embedding $E _ { \mathrm { c } }$ , and an output mapping $W _ { \mathrm { c } } ,$ the final logits are given by:

$$
z _ { i } = z _ { i } ^ { ( 0 ) } + W _ { \mathrm { c } } E _ { \mathrm { c } } ( y _ { i - 1 } ) , \qquad i = 1 , \ldots , B .\tag{7}
$$

The resulting proposal factorizes as

$$
q ( y _ { 1 : B } \mid s ) = \prod _ { i = 1 } ^ { B } q _ { i } ( y _ { i } \mid s , y _ { i - 1 } ) , q _ { i } = \operatorname { s o f t m a x } ( z _ { i } ) ,\tag{8}
$$

with $y _ { 0 } = \mathtt { a } .$ . These corrected distributions are used for both sampling and verification in (1)–(2). Only the correction and sampling occur sequentially; the Transformer backbone and base vocabulary projection remain fully parallel.

## 2.4 TRAINING AND INFERENCE

Training. All Target parameters, including the token embedding, normalization, and vocabulary head, remain frozen. We jointly train the draft network, boundary fusion and input mappings, summary queries, and the DSpark-style sequential correction (Cheng et al., 2026). The full-vocabulary distributions $\mathbf { \nabla } \mathcal { P } \boldsymbol { i }$ and $q _ { i }$ are evaluated under teacher forcing: the target sees the ground-truth prefix $x _ { 1 : t + i }$ , and the correction in (7) uses the predecessor $x _ { t + i }$ . For ground-truth token $x _ { t + 1 + i } ,$ the weighted loss at position � is

$$
\mathcal { I } _ { i } = w _ { i } \Big [ \lambda \big [ - \log q _ { i } ( x _ { t + 1 + i } ) \big ] + ( 1 - \lambda ) \| q _ { i } - p _ { i } \| _ { 1 } \Big ] ,\tag{9}
$$

where $\lambda = 0 . 1$ and $w _ { i } = \exp ( - ( i - 1 ) / 4 )$ . We sum these losses over valid training positions and normalize by the sum of their weights. At inference, the correction instead uses the previously sampled token.

Inference. Verification follows § 2.1, using the corrected proposal distributions from (8). When � candidates are accepted, the prefix advances by � + 1 positions, and the committed token a along with $y _ { 1 : r }$ enter the Target KV cache. This update naturally drives the incremental maintenance of the interface: their KV rows update the global context summary and advance the recent window, while the hidden states at the new boundary yield the updated $\mathbf { } c _ { t } .$ . The process concludes by setting the correction or bonus token as the next committed token a and discarding all transient draft KV, leaving the updated interface ready for the next round.

Fixed-cost complexity. For fixed model dimensions, each draft layer attends to � block positions, � recent tokens, and � summary entries, costing $O ( B ^ { 2 } + B W + \dot { B } R )$ per round. Full-prefix draft attention instead costs $O ( B ^ { 2 } { + } B t )$ . Interface maintenance reads a constant-size boundary and window and incorporates at most $B + 1$ newly retained KV rows into the summary. With � attention heads of dimension $d ,$ the summary occupies $O ( | { \mathcal { L } } | H R d )$ persistent state; the recent window is read directly from the target cache, and block KV is transient. Both the drafter’s additional state and its per-round computation are therefore independent of �. Full-prefix processing is confined to target verification, with the global summary initialized once during prefill.

Table 1: Speculative decoding across model scales and tasks. Entries report accepted length �, with throughput speedup over autoregressive decoding in parentheses. Avg. denotes the mean across eight benchmarks; bold marks the highest speedup.
<table><tr><td rowspan="2">Target Drafter</td><td rowspan="2"></td><td colspan="3">Math</td><td colspan="3">Code</td><td colspan="2">Chat</td><td>Overall</td></tr><tr><td>GSM8K</td><td>MATH-500</td><td>AIME25</td><td>MBPP</td><td>HumanEval</td><td>LCB</td><td>MT-Bench</td><td>Alpaca</td><td>Avg.</td></tr><tr><td rowspan="4">Q3-4B</td><td>EAGLE-3</td><td>5.15 (1.36)</td><td>4.60 (1.35)</td><td>3.84 (1.24)</td><td>3.69 (1.03)</td><td>4.15 (1.17)</td><td>3.77 (1.14)</td><td>2.42 (0.72)</td><td>2.26 (0.69)</td><td>3.74 (1.09)</td></tr><tr><td>DFlash</td><td>5.37 (1.81)</td><td>4.89 (1.89)</td><td>4.01 (1.72)</td><td>4.40 (1.55)</td><td>4.74 (1.72)</td><td>4.23 (1.61)</td><td>3.05 (1.18)</td><td>2.94 (1.14)</td><td>4.20 (1.58)</td></tr><tr><td>DSpark</td><td>6.10 (1.99)</td><td>5.72 (2.14)</td><td>4.92 (2.05)</td><td>5.14 (1.73)</td><td>5.44 (1.91)</td><td>4.92 (1.79)</td><td>3.65 (1.36)</td><td>3.55 (1.32)</td><td>4.93 (1.79)</td></tr><tr><td>LONGSPARK</td><td>5.90 (2.08)</td><td>5.52 (2.28)</td><td>4.67 (2.17)</td><td>5.01 (1.84)</td><td>5.12 (1.97)</td><td>4.63 (1.86)</td><td>3.52 (1.46)</td><td>3.46 (1.42)</td><td>4.73 (1.88)</td></tr><tr><td rowspan="4">Q3-8B</td><td>EAGLE-3</td><td>5.26 (1.49)</td><td>4.75 (1.50)</td><td>4.02 (1.37)</td><td>3.92 (1.16)</td><td>4.32 (1.30)</td><td>4.17 (1.28)</td><td>2.67 (0.85)</td><td>2.54 (0.82)</td><td>3.96 (1.22)</td></tr><tr><td>DFlash</td><td>5.36 (1.96)</td><td>4.91 (2.02)</td><td>4.07 (1.83)</td><td>4.36 (1.66)</td><td>4.67 (1.82)</td><td>4.44 (1.68)</td><td>3.09 (1.28)</td><td>3.00 (1.24)</td><td>4.24 (1.69)</td></tr><tr><td>DSpark</td><td>6.17 (2.17)</td><td>5.80 (2.32)</td><td>4.99 (2.19)</td><td>5.18 (1.89)</td><td>5.48 (2.06)</td><td>5.09 (1.84)</td><td>3.69 (1.47)</td><td>3.59 (1.42)</td><td>5.00 (1.92)</td></tr><tr><td>LONGSPARK</td><td>5.93 (2.24)</td><td>5.57 (2.42)</td><td>4.72 (2.27)</td><td>5.03 (1.98)</td><td>5.16 (2.11)</td><td>4.83 (1.88)</td><td>3.53 (1.54)</td><td>3.46 (1.49)</td><td>4.78 (1.99)</td></tr><tr><td rowspan="4">Q3-14B</td><td>EAGLE-3</td><td>5.22 (1.69)</td><td>4.62 (1.63)</td><td>3.82 (1.44)</td><td>3.81 (1.29)</td><td>4.11 (1.41)</td><td>4.01 (1.35)</td><td>2.61 (0.93)</td><td>2.49 (0.90)</td><td>3.84 (1.33)</td></tr><tr><td>DFlash</td><td>5.37 (2.16)</td><td>4.86 (2.17)</td><td>4.01 (1.91)</td><td>4.45 (1.85)</td><td>4.57 (1.94)</td><td>4.35 (1.72)</td><td>3.09 (1.38)</td><td>2.96 (1.33)</td><td>4.21 (1.81)</td></tr><tr><td>DSpark</td><td>6.19 (2.43)</td><td>5.77 (2.53)</td><td>4.93 (2.32)</td><td>5.25 (2.12)</td><td>5.41 (2.24)</td><td>4.97 (1.88)</td><td>3.67 (1.61)</td><td>3.56 (1.56)</td><td>4.97 (2.09)</td></tr><tr><td>LONGSPARK</td><td>6.03 (2.47)</td><td>5.63 (2.61)</td><td>4.76 (2.37)</td><td>5.14 (2.18)</td><td>5.15 (2.25)</td><td>4.78 (1.90)</td><td>3.57 (1.65)</td><td>3.48 (1.60)</td><td>4.82 (2.13)</td></tr></table>

## 3 EXPERIMENTS

We organize the evaluation around three questions. Does a fixed-cost context interface outperform drafters that read the entire prefix across model scales, tasks, and serving loads (§ 3.2)? Does its advantage persist as contexts lengthen, with drafting cost independent of the prefix length (§ 3.3)? And how much does each context component contribute (§ 3.4)?

## 3.1 EXPERIMENTAL SETUP

Models and baselines. To demonstrate the generalizability and scalability of LONGSPARK, we evaluate it across three Target scales: Qwen3-4B, 8B, and 14B. We compare our approach against a suite of representative baselines, including the autoregressive EAGLE-3 (Li et al., 2025) and stateof-the-art parallel drafters DFlash (Chen et al., 2026) and DSpark (Cheng et al., 2026), alongside vanilla autoregressive decoding.

Training data. We train LONGSPARK on Open-PerfectBlend (Labonne, 2024), a high-quality instruction-tuning dataset of approximately 1.42 million samples, for 10 epochs. We use only the prompts and regenerate all responses with the corresponding Target model, ensuring the drafter learns the exact distribution it is intended to accelerate. Training details are provided in Appendix B.2.

Evaluation benchmarks. We subject LONGSPARK to a comprehensive set of stress tests across eight benchmarks spanning three primary domains: mathematical reasoning, i.e., GSM8K (Cobbe et al., 2021), MATH-500 (Lightman et al., 2024), and AIME25 (MAA, 2025), code generation, i.e., MBPP (Austin et al., 2021), HumanEval (Chen et al., 2021), and LiveCodeBench (LCB) (Jain et al., 2025), and open-ended dialogue, i.e., MT-Bench (Zheng et al., 2023) and Alpaca (Taori et al., 2023). The main long-context evaluation covers continuation tasks from LongSpec’s training corpus (Yang et al., 2026) at 32K, our curated CodeSpan at 64K, and LongSWE-Bench (LSWE) (Rando et al., 2025) at 128K. Details of the long-context datasets are provided in Appendix C.1.

Evaluation settings and metrics. We use a target–drafter (TD) disaggregated framework: speculative methods add one dedicated drafter GPU to the Target GPUs used by autoregressive decoding. Main-text results use � = 1, thinking mode disabled, and seven draft tokens per verification round for speculative methods. We report throughput speedup over autoregressive decoding and average accepted length �, with output throughput (TPS) and time per output token (TPOT) for long-context workloads. Detailed metric definitions are provided in Appendix C.2.

![](images/6eb21c2a102a84072686aafc3d1cdcd59541a7472e6f6afb8f232b22317d11e6.jpg)  
Figure 4: Higher concurrency increases serving load. As this load rises, LONGSPARK’s throughput advantage over DSpark grows.

Table 2: Long-context decoding on Qwen3-8B. We report TPS (tokens/s) and TPOT (ms/token), averaged over ten seeds. Bold marks the best mean.
<table><tr><td rowspan="2">Method</td><td colspan="2">LongSpec 32K</td><td colspan="2">CodeSpan 64K</td><td colspan="2">LSWE 128K</td></tr><tr><td>TPS ↑</td><td>TPOT ↓</td><td>TPS ↑</td><td>TPOT ↓</td><td>TPS ↑</td><td>TPOT ↓</td></tr><tr><td>Vanilla</td><td>1128.9</td><td>14.0</td><td>884.4</td><td>17.8</td><td>160.5</td><td>97.2</td></tr><tr><td>EAGLE-3</td><td>1258.5</td><td>12.5</td><td>1071.8</td><td>14.7</td><td>158.7</td><td>97.4</td></tr><tr><td>DFlash</td><td>1313.7</td><td>12.0</td><td>919.1</td><td>17.2</td><td>163.6</td><td>94.6</td></tr><tr><td>DSpark</td><td>1736.6</td><td>9.0</td><td>1217.3</td><td>13.0</td><td>190.9</td><td>81.4</td></tr><tr><td>LONGSPARK</td><td>1917.6</td><td>8.2</td><td>1571.8</td><td>10.0</td><td>212.7</td><td>74.3</td></tr></table>

## 3.2 OVERALL PERFORMANCE

We first evaluate end-to-end efficiency across model scales and task domains, using target TP1, concurrency 32, and up to 2,048 generated tokens per request. As shown in Table 1, LONGSPARK achieves the highest average throughput at every scale, delivering 1.88×, 1.99×, and 2.13× speedups from 4B to 14B—with the speedup growing as the target scales. Crucially, it keeps this lead even when DSpark attains longer accepted lengths. This separation exposes the central tension of spec ulative decoding: end-to-end efficiency depends not on proposal quality alone, but on the balance between accepted tokens and the overhead of producing them. The same ranking holds under greedy decoding, confirming that the advantage of a fixed-cost drafter is structural rather than samplingdependent (Appendix D.1).

High-concurrency performance. To examine how the advantage scales with serving load, we vary concurrency from 8 to 128 on Alpaca, MBPP, and MATH-500. As Figure 4 shows, all three target scales exhibit the same pattern: LONGSPARK is marginally behind DSpark at low concurrency, then overtakes it and pulls away steadily as load rises, leading by a clear advantage at concurrency 128. The gap widens on every benchmark despite shorter accepted lengths, revealing that the value of low-overhead drafting compounds under high serving load. Appendix D.4 provides additional concurrency results.

## 3.3 SCALING WITH CONTEXT LENGTH

Long-context performance. We evaluate contexts from 32K to 128K using Qwen3-8B with target TP4, concurrency 16, and up to 8,192 generated tokens per request. Long-context evaluation uses fixed NTK scaling (bloc97, 2023) with � = 4. As shown in Table 2, LONGSPARK achieves the highest mean throughput and lowest mean TPOT across all three workloads. On CodeSpan at 64K, its mean TPS exceeds that of DSpark, the strongest competing method, by 29.1%, while reducing mean TPOT by 23.1%. Results for Qwen3-4B and 14B are provided in Appendix D.2.

(a) Accepted length �.
<table><tr><td>Variant</td><td>Math Code Chat Avg.</td></tr><tr><td>LONGSPARK</td><td>5.33 4.91 3.40 4.69</td></tr><tr><td>w/o global summary</td><td>4.99 4.50 3.17</td></tr><tr><td>w/o boundary state</td><td>4.35 4.90 4.42 3.15 4.28</td></tr><tr><td>w/o recent KV window</td><td>4.17 3.82 2.86 3.71</td></tr></table>

(b) Training loss.  
![](images/36bbdc330e3f30dfdda92bffa03853d79781ce5acf949111a383d624235d3e97.jpg)  
Figure 5: Ablation on Qwen3-8B. (a) Mean accepted length by domain; (b) Training loss curve.

Drafting cost and memory. The throughput gains of LONGSPARK stem from its lean drafting overhead. Figure 1 (right) decomposes a decoding round on LongSpec with Qwen3-8B into proposal and verification. Target verification dominates the round, and LONGSPARK nearly halves the proposal component relative to DSpark. Because the round is bounded by verification, this reduction lowers total round latency only modestly; its value lies in repetition, as the saving accumulates over the thousands of rounds needed to generate a long response.

The context-length sweep in Figure 1 (left) confirms that the fixed-size interface decouples the drafter’s decoding cost from the prefix length, allowing the advantage to grow with context rather than fade. As the context grows from 32K to 128K, DSpark’s drafting time rises from 6.3 to 13.8 ms and its effective context state grows from 640 to 2560 MiB, since both quantities are tied to the number of retained tokens. LONGSPARK holds both nearly constant: its drafting time stays near 3.5 ms and its context state remains fixed at 6.3 MiB across the same range, amounting to a 406× reduction in drafter state at 128K.

These measurements isolate where the cost of drafting actually resides. Since target verification remains the only full-prefix operation in each round, the drafter’s contribution can be made both small and constant. This is the structural distinction that separates LONGSPARK from prefix-scaling drafters: for them, a longer context translates directly into more work per proposal, whereas here the proposal cost is a fixed tax that never grows.

## 3.4 ABLATION STUDY

Figure 5(a) isolates each context component by removing it in turn, with up to 2,048 generated tokens per request. The four variants are trained for 5 epochs for a fair comparison. The recent KV window proves the most critical: without it, average accepted length falls from 4.69 to 3.71, confirming that token-level local detail is the primary driver of proposal quality. Removing the global summary or the boundary state causes smaller degradations, to 4.35 and 4.28 respectively, so both are secondary yet not redundant. The training dynamics in Figure 5(b) add a complementary view: removing the global summary produces pronounced transient loss spikes, whereas the full model descends smoothly. The summary thus contributes to training stability as well as proposal quality, since without it a bounded-context drafter has no reliable signal about the global prefix and must reconstruct it from local evidence alone.

Together, these results reveal a fundamental asymmetry between drafter and target. The target requires a full, precise state to verify; the drafter needs only enough orientation to propose. The recent window supplies immediate local grounding, while the global summary supplies a stable, lowresolution view of the entire prefix. Detailed long-context ablations are provided in Appendix D.3.

## 4 RELATED WORK

We first review the lossless verification framework that makes drafter quality a pure efficiency concern, then survey how autoregressive, recurrent, and block-diffusion drafters condition on the confirmed prefix.

Lossless Speculative Decoding Stern et al. (2018) first showed that a drafter can propose multiple future tokens at once and that a Target model can verify them, accepting the longest prefix consistent with exact greedy decoding. Speculative sampling extends this to stochastic generation: Leviathan et al. (2023) use rejection and residual correction to preserve the Target distribution exactly. Because the Target’s verification preserves the output distribution regardless of proposal quality, the drafter determines only efficiency. How it conditions on the confirmed prefix is therefore a design choice that determines how its cost scales with the prefix length.

Autoregressive Drafters The standard construction uses a smaller autoregressive language model whose own KV cache grows with the prefix. Later methods strengthen proposals by incorporating Target representations: EAGLE and EAGLE-3 reuse Target hidden features (Li et al., 2024; 2025; Hui et al., 2026), and ReDrafter conditions a recurrent proposer on the Target state (Cheng et al., 2024). LongSpec bounds its own draft cache using windowed self-attention, but it still cross-attends to the Target’s full KV history (Yang et al., 2026). Bounding the drafter’s own cache does not remove this growing read, and these drafters still advance one position at a time.

Block-Parallel and Block-Diffusion Drafting Block-parallel methods take a different approach: instead of drafting tokens one by one, Xiao et al. (2024) and An et al. (2026) predict several future positions at once. Medusa attaches independent prediction heads to the Target for fixed-horizon parallel proposals (Cai et al., 2024; Wertheimer et al., 2024). Each head conditions only on the Target’s last hidden state, making its cost prefix-independent but limiting it to single-position predictions that do not attend to the prefix. DFlash replaces these independent heads with a block-diffusion back bone that predicts all masked positions jointly and injects Target features from the entire confirmed prefix into every draft layer (Chen et al., 2026; Li et al., 2026). This yields more expressive proposals, but makes every draft layer process a context sequence that grows with the prefix. Orthrus shares the Target KV cache to eliminate duplicate storage, but its diffusion view still attends to the complete cache, leaving prefix-length context computation intact (Nguyen et al., 2026).

Subsequent work improves along other dimensions while retaining the same growing context interface. DFlare strengthens Target-to-drafter feature fusion (Zhang et al., 2026); DSpark, Domino, Jet-Spec and xPress restore causal dependence within the proposed block (Cheng et al., 2026; Hu et al., 2026; Huang et al., 2026; Wang et al., 2026a); DDTree organizes the position-wise distributions into a draft tree (Agrawal et al., 2025; Gao et al., 2026; Ringel & Romano, 2026; Wang et al., 2026b); and AdaFlash, like DSpark, schedules the proposal horizon with a dynamic confidence head (Cheng et al., 2026; Qian et al., 2026). Each of these improves feature conditioning, within-block coherence, candidate coverage, or the proposal horizon, yet each keeps the context that proposals read tied to the prefix. LONGSPARK changes this remaining axis: it is, to our knowledge, the first blockdiffusion drafter to replace the growing context with a fixed-size, multiscale view of the Target state. Target verification is consequently the only full-prefix traversal in each round; all additional context processing, proposal computation, and drafter state remain fixed as the context grows.

## 5 CONCLUSION AND FUTURE DIRECTIONS

This work establishes that, during decoding, a speculative drafter’s cost can be made entirely independent of the prefix length while maintaining competitive accepted lengths. Because verification guarantees output correctness, the drafter needs only enough context to produce useful proposals, and we show that this context can be extracted from the target’s state at fixed cost.

The end-to-end results bear out the value of this fixed cost. LONGSPARK attains the highest through put across model scales even though the strongest prefix-scaling drafter reaches slightly longer accepted lengths, showing that low proposal overhead outweighs marginal acceptance gains. This advantage grows with context length and serving concurrency because the drafter’s cost remains constant while the baselines’ costs grow. Ablations further show that a compact, lossy view of the prefix provides sufficient orientation for competitive proposals. Together, these results identify accepted tokens per unit of proposal overhead as the quantity drafting design should optimize.

Future directions include evaluating the fixed-cost interface with larger target models and longer context windows to assess its effectiveness as model scale and context demands increase. The interface developed here offers one viable realization of fixed-cost drafting; alternative interface designs and drafter architectures within this framework may further improve proposal quality while retaining prefix-independent cost. Beyond proposal design, exploring tree-structured and multi-drafter verification schemes could help increase the number of accepted tokens per round while preserving the �(1) drafting cost. On the application side, reinforcement-learning rollouts offer a promising setting for these extensions, as many long trajectories run under tight latency budgets and a constant proposal overhead across trajectory length could compound throughput gains.

## AI USE OF GENERATIVE MODELS

We used generative AI tools to assist with the research workflow: developing and debugging code, analysing experimental results, and preparing the manuscript, tables, and figures. The authors reviewed and verified all AI-assisted work and take full responsibility for the final content of this work, including all text, claims, and artefacts.

## REFERENCES

Sudhanshu Agrawal, Risheek Garrepalli, Raghavv Goel, Christopher Lott, Fatih Porikli, and Mingu Lee. Structuring the future: Diffusion LLM speculative decoding via calibrated draft graphs. arXiv preprint arXiv:2509.18085, 2025.

Zihao An, Huajun Bai, Ziqiong Liu, Dong Li, and Emad Barsoum. PARD: Accelerating LLM inference with low-cost parallel draft model adaptation. In The 14th International Conference on Learning Representations, Rio de Janeiro, Brazil, 2026.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and Charles Sutton. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

bloc97. NTK-Aware Scaled RoPE allows LLaMA models to have extended context length, 2023. URL https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkawar e\_scaled\_rope\_allows\_llama\_models\_to\_have/.

Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, and Tri Dao. Medusa: Simple LLM inference acceleration framework with multiple decoding heads. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pp. 5209–5235, Vienna, Austria, 2024. PMLR.

Jian Chen, Yesheng Liang, and Zhijian Liu. DFlash: Block diffusion for flash speculative decoding. In Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea, 2026.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob Mc-Grew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Xin Cheng, Xingkai Yu, Chenze Shao, Jiashi Li, Yunfan Xiong, Yi Qian, Jiaqi Zhu, Shirong Ma, Xiaokang Zhang, Jiasheng Ye, Qinyu Chen, Chengqi Deng, Jiping Yu, Damai Dai, Zhengyan Zhang, Yixuan Wei, Yixuan Tan, Wenkai Yang, Runxin Xu, Yu Wu, Zhean Xu, Xuanyu Wang, Muyang Chen, Rui Tian, Xiao Bi, Zhewen Hao, Shaoyuan Chen, Huanqi Cao, Wentao Zhang, Anyi Xu, Huishuai Zhang, Dongyan Zhao, and Wenfeng Liang. DSpark: Confidence-scheduled speculative decoding with semi-autoregressive generation. arXiv preprint arXiv:2607.05147, 2026.

Yunfei Cheng, Aonan Zhang, Xuanyu Zhang, Chong Wang, and Yi Wang. Recurrent drafter for fast speculative decoding in large language models. arXiv preprint arXiv:2403.09919, 2024.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Zipeng Gao, Zhi Zheng, Qingrong Xia, Junda Lin, Ziwei Zhao, Tong Xu, Zhefeng Wang, and Enhong Chen. Unlocking parallelism in autoregressive language models via speculative decoding with progressive tree drafting. In The 3rd Conference on Language Modeling, San Francisco, California, United States, 2026.

Lanxiang Hu, Zhaoxiang Feng, Yulun Wu, Haoran Yuan, Yujie Zhao, Yu-Yang Qian, Bojun Wang, Peng Zhao, Daxin Jiang, Yibo Zhu, Tajana Rosing, and Hao Zhang. JetSpec: Breaking the scaling ceiling of speculative decoding with parallel tree drafting. arXiv preprint arXiv:2606.18394, 2026.

Jianuo Huang, Yaojie Zhang, Qituan Zhang, Hao Lin, Hanlin Xu, and Linfeng Zhang. Domino: Decoupling causal modeling from autoregressive drafting in speculative decoding. arXiv preprint arXiv:2605.29707, 2026.

Mude Hui, Xin Huang, Jaime Campos Salas, Yue Sun, Nathan Pemberton, Xiang Song, Ashish Khetan, and George Karypis. P-EAGLE: Parallel-drafting EAGLE with scalable training. arXiv preprint arXiv:2602.01469, 2026.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. LiveCodeBench: Holistic and contamination free evaluation of large language models for code. In The 13th International Conference on Learning Representations, Singapore, 2025.

Maxime Labonne. Open-PerfectBlend, 2024. URL https://huggingface.co/dataset s/mlabonne/open-perfectblend.

Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pp. 19274–19286, Honolulu, Hawaii, United States, 2023. PMLR.

Guanghao Li, Zhihui Fu, Min Fang, Qibin Zhao, Ming Tang, Chun Yuan, and Jun Wang. DiffuSpec: Unlocking diffusion language models for speculative decoding. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pp. 20896–20910, San Diego, California, United States, 2026. Association for Computational Linguistics.

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. EAGLE: Speculative sampling requires rethinking feature uncertainty. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pp. 28935–28948, Vienna, Austria, 2024. PMLR.

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. EAGLE-3: Scaling up inference acceleration of large language models via training-time test. In Advances in Neural Information Processing Systems, volume 38, pp. 136737–136756, San Diego, California, United States, 2025.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The 12th International Conference on Learning Representations, Vienna, Austria, 2024.

MAA. American Invitational Mathematics Examination - AIME, 2025. URL https://maa.or g/math-competitions/american-invitational-mathematics-examinati on-aime.

Chien Van Nguyen, Chaitra Hegde, Van Cuong Pham, Ryan A. Rossi, Franck Dernoncourt, and Thien Huu Nguyen. Orthrus: Memory-efficient parallel token generation via dual-view diffusion. arXiv preprint arXiv:2605.12825, 2026.

Yu-Yang Qian, Hao-Cong Wu, Chen Chen, Jiacheng Sun, Zhenhua Dong, Peng Zhao, and Zhi-Hua Zhou. AdaFlash: Adaptive speculative decoding via on-policy distilled diffusion drafters. arXiv preprint arXiv:2607.19223, 2026.

Stefano Rando, Luca Romani, Alessio Sampieri, Luca Franco, John Yang, Yuta Kyuragi, Fabio Galasso, and Tatsunori Hashimoto. LongCodeBench: Evaluating coding LLMs at 1M context windows. In The 2nd Conference on Language Modeling, Montreal, Canada, 2025.

Liran Ringel and Yaniv Romano. Accelerating speculative decoding with block diffusion draft trees. In The 3rd Conference on Language Modeling, San Francisco, California, United States, 2026.

Mitchell Stern, Noam Shazeer, and Jakob Uszkoreit. Blockwise parallel decoding for deep autoregressive models. In Advances in Neural Information Processing Systems, volume 31, pp. 10107–10116, Montreal, Canada, 2018.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford Alpaca: An instruction-following LLaMA model, 2023. URL https://github.com/tatsu-lab/stanford\_alpaca.

Zheng Wang, Davis Wertheimer, Yu Chin Fabian Lim, Mudhakar Srivatsa, Raghu K. Ganti, Minjia Zhang, and Naigang Wang. xPress: Parallel refinement for diffusion drafters in speculative decoding. arXiv preprint arXiv:2608.02438, 2026a.

Zheng Wang, Zhifan Ye, Qi Cheng, Yonggan Fu, Ziyan Wang, Feng Zhu, Haozhe Zhao, Jan Kautz, Pavlo Molchanov, Humphrey Shi, and Minjia Zhang. PRESTO: Prefix-aligned tree drafting for diffusion speculative decoding. arXiv preprint arXiv:2607.22634, 2026b.

Davis Wertheimer, Joshua Rosenkranz, Thomas Parnell, Sahil Suneja, Pavithra Ranganathan, Raghu Ganti, and Mudhakar Srivatsa. Accelerating production LLMs with combined token/embedding speculators. arXiv preprint arXiv:2404.19124, 2024.

Zilin Xiao, Hongming Zhang, Tao Ge, Siru Ouyang, Vicente Ordonez, and Dong Yu. ParallelSpec: Parallel drafter for efficient speculative decoding. arXiv preprint arXiv:2410.05589, 2024.

Penghui Yang, Cunxiao Du, Fengzhuo Zhang, Haonan Wang, Tianyu Pang, Chao Du, and Bo An. LongSpec: Long-context lossless speculative decoding with efficient drafting and verification. In Proceedings ofthe 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 1826–1844, San Diego, California, United States, 2026. Association for Computational Linguistics.

Jiebin Zhang, Zhenghan Yu, Song Liu, Eugene J. Yu, Zheng Li, Dawei Zhu, Jiangshan Duo, Weimin Xiong, Yifan Song, Guanghua Yu, Jianchen Zhu, and Sujian Li. DFlare: Scaling up draft capacity for block diffusion speculative decoding. arXiv preprint arXiv:2606.02091, 2026.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-judge with MT-Bench and chatbot arena. In Advances in Neural Information Processing Systems, volume 36, pp. 46595–46623, New Orleans, Louisiana, United States, 2023.

## APPENDIX

## A METHOD AND ALGORITHMIC DETAILS

This section details the proposal computation and the incremental maintenance of the global summary.

## A.1 DRAFTER FORWARD PASS

We first describe how the drafter turns the three fixed-size context views into one block proposal.

Figure 6 summarizes one proposal step after the Target has cached $x _ { 1 : t } ,$ with $\mathsf { a } = x _ { t + 1 }$ as the next committed token. The drafter takes the selected-layer boundary states $\{ h _ { t } ^ { \ell } \} _ { \ell \in \mathcal { L } }$ , recent Target KV windows, and global summary pairs $( G , M _ { t } )$ , all ordered by selected target layer.

![](images/efcef8c44ee13466d00ad814c5097bf6be75db83fa05b661e5d4bfbb8395d2a4.jpg)  
Figure 6: One block proposal by the LONGSPARK drafter.

Input slot 0 supplies the base logits for �<sub>1</sub>. All slots pass through the draft layers in parallel, with one attention normalization across the three sources and bidirectional attention among local slots. The Target embedding, final normalization, and vocabulary head remain frozen. Local block KV is discarded after drafting; only the Markov correction and sampling proceed sequentially. The returned distributions are used in verification, after which retained Target KV rows update the summary as described next.

## A.2 INCREMENTAL GLOBAL CONTEXT SUMMARY

We then derive the update that folds newly retained rows into the global summary at constant cost.

For a fixed summary query �, let $s _ { j } = \mathbf { g } ^ { \top } k _ { j } / \sqrt { d }$ be its attention score for Target key $k _ { j }$ . For any nonempty set of KV rows �, define its summary value and log-normalizer as

$$
\eta _ { A } = \log \sum _ { j \in A } \exp ( s _ { j } ) , \qquad \pmb { m } _ { A } = \sum _ { j \in A } \exp ( s _ { j } - \eta _ { A } ) \pmb { v } _ { j } .\tag{10}
$$

Let � contain the previously processed prefix and � the newly retained rows. Their attention statistics merge as

$$
\eta = \log \bigl ( \exp ( \eta _ { A } ) + \exp ( \eta _ { N } ) \bigr ) ,
$$

$$
\pmb { m } = \exp ( \eta _ { A } - \eta ) \pmb { m } _ { A } + \exp ( \eta _ { N } - \eta ) \pmb { m } _ { N } .\tag{11}
$$

The merged value � equals attention over � ∪ �; an empty new set leaves the state unchanged. The log-sum-exp operations are evaluated with max subtraction for numerical stability. Applying this merge independently to all summary queries recovers (5) while storing only one value vector and one log-normalizer per query. Since the queries and historical Target KV remain fixed, only the statistics over � need to be computed in each round.

## B MODEL ARCHITECTURE AND TRAINING

This section records the architecture and training settings shared by all reported drafters.

## B.1 DRAFTER ARCHITECTURE

We specify the drafter depth, its context-view sizes, and the Target layers it reads.

LONGSPARK employs a lightweight decoder comprising 5 Qwen3-style blocks that operate in parallel to propose 7 tokens. To maintain architectural consistency, all drafters adhere to the standard Qwen3 hyperparameters for a given hidden dimension � (e.g., head dimension and intermediate width). The architecture is designed as an efficient distilled version of the Target, integrating a global context summary, a recent KV window, and a small local workspace. We extract context views from Target layers {0, 9, 18, 26, 35} for Qwen3-4B and 8B, and {0, 10, 20, 29, 39} for Qwen3- 14B. Across all scales, the global context summary contains $R = 1 6$ entries per selected layer and attention head, and the recent KV window contains � = 256 entries per selected layer. We reuse the Target’s embedding and final LM head to ensure vocabulary consistency.

## B.2 TRAINING PROCEDURE

We list the data, objective, and optimization settings used to train each drafter.

We train the LONGSPARK drafters with the objective in (9) on the Open-PerfectBlend dataset (Labonne, 2024), which contains approximately 1.42M sessions. For each target scale, we regenerate the assistant responses in a non-thinking mode using the frozen Target model (temperature = 0.7, top-p = 0.8, top-k = 20). During training, we limit the total sequence length to 4096 tokens and randomly sample up to 512 anchors per sequence to optimize the distillation loss across multiple positions in each generated response.

Summary queries. Each selected layer and attention head has an independent bank of � summary queries, initialized from $\mathcal { N } ( 0 , d ^ { - 1 } I )$ and jointly optimized with the drafter under the same training objective, where � is the head dimension. The queries are shared across examples and remain fixed during inference, enabling the incremental summary updates in Appendix A.2.

The main drafters are trained for 10 epochs and the ablation variants for 5 epochs, all with a global batch size of 512. We use the AdamW optimizer with BF16 mixed-precision and FP32 master weights, a peak learning rate of $6 \times 1 0 ^ { - 4 }$ , and a cosine decay scheduler with a 4% linear warmup. For all model scales, we employ a data-parallel (DP) degree of 8 with gradient accumulation steps of 64.

## C EVALUATION PROTOCOL AND DATASET DETAILS

This section documents the long-context workloads and the measurement protocol behind the reported numbers.

## C.1 LONG-CONTEXT EVALUATION DATASETS

We describe the composition of the three long-context workloads.

LongSpec. We construct continuation tasks from the code, arXiv, and book subsets of LongSpec’s released training corpus (Yang et al., 2026), using 32,768-token inputs. These subsets form the LongSpec 32K evaluation workload. Our drafting-cost analysis uses a prefix from the code subset.

CodeSpan. CodeSpan comprises 32 source-code continuation examples drawn from 32 distinct files across 17 open-source projects, including LLVM, GCC, Linux, and PostgreSQL. We select sufficiently long files, with at most four files per project. The collection is predominantly C/C++ (28 files), with additional Python, TypeScript, and Yacc grammar files. Each example uses a contiguous prefix of a single file, without concatenation or repetition. Using the Qwen3-8B tokenizer, we choose the prefix length so that the complete input, including the continuation instruction and chat template, contains 65,536 tokens.

LongSWE-Bench. LongSWE-Bench (Rando et al., 2025) evaluates repository-level bug fixing with long code contexts. We select inputs that fit within a 128K total context budget while reserving space for up to 8,192 generated tokens. The same evaluation inputs and ordering are shared across methods and random seeds.

## C.2 MEASUREMENT PROTOCOL

We define how throughput, accepted length, and time per output token are measured.

Throughput and accepted length. We measure steady-state serving performance over a 120- second interval after a 30-second warm-up, continuously admitting new requests as others finish. Each method samples its own responses independently. Throughput counts all output tokens emitted during this interval. For each seed, � is computed by dividing the number of emitted tokens after the first token by the number of verification request-rounds, including bonus or residual tokens and accounting for sequence termination. We take arithmetic means over seeds, then compute speedup as the ratio of a method’s mean throughput to the autoregressive baseline’s mean throughput. For the long-context component ablations in Table 5, drafters are trained for five epochs. Entries report the mean and standard deviation over ten seeds, with variants evaluated on the same host within each seed.

Time per output token. TPOT is the total post-first-token request time overlapping the measurement interval divided by the number of subsequent output tokens emitted within that interval. The numerator includes the observed time of requests still active at the end of measurement. This definition excludes time to first token while retaining scheduling and concurrent prefill effects.

## D ADDITIONAL DECODING RESULTS

This section collects results that complement the main evaluation.

## D.1 GREEDY-DECODING RESULTS

We first check whether the efficiency advantage depends on sampling.

Table 3 presents greedy-decoding results $( T = 0 )$ with target TP1, concurrency 32, and up to 2,048 generated tokens per request, complementing the sampling-based evaluation in Table 1. The performance trends here closely mirror those observed at � = 1, with LONGSPARK consistently achieving the highest speedups across all model scales and benchmarks. This consistency demonstrates that the efficiency gains of LONGSPARK are a structural advantage independent of the decoding strategy, further validating that the �(1) proposal cost is the primary driver of end-to-end performance regardless of the temperature setting.

Table 3: Greedy decoding across model scales (� = 0). Entries report accepted length and throughput speedup using the conventions of Table 1.
<table><tr><td rowspan="2">Target</td><td rowspan="2">Drafter</td><td colspan="3">Math</td><td colspan="3">Code</td><td colspan="2">Chat</td><td rowspan="2">Overall</td></tr><tr><td>GSM8K</td><td>MATH-500</td><td>AIME25</td><td>MBPP</td><td>HumanEval</td><td>LCB</td><td>MT-Bench</td><td>Alpaca</td></tr><tr><td rowspan="4">Q3-4B</td><td>EAGLE-3</td><td>5.46 (1.55)</td><td>5.23 (1.68)</td><td>4.72 (1.68)</td><td>4.12 (1.24)</td><td>4.56 (1.40)</td><td>4.92 (1.60)</td><td>2.70 (0.89)</td><td>2.48 (0.82)</td><td>4.27 (1.36)</td></tr><tr><td>DFlash</td><td>5.93 (2.21)</td><td>5.87 (2.54)</td><td>5.28 (2.57)</td><td>4.88 (1.90)</td><td>5.21 (2.12)</td><td>5.34 (2.25)</td><td>3.44 (1.51)</td><td>3.30 (1.43)</td><td>4.90 (2.07)</td></tr><tr><td>DSpark</td><td>6.33 (2.26)</td><td>6.30 (2.64)</td><td>5.77 (2.73)</td><td>5.38 (2.01)</td><td>5.68 (2.24)</td><td>5.71 (2.33)</td><td>3.84 (1.62)</td><td>3.67 (1.53)</td><td>5.34 (2.17)</td></tr><tr><td>LONGSPARK</td><td>6.14 (2.41)</td><td>6.07 (2.85)</td><td>5.44 (2.94)</td><td>5.27 (2.17)</td><td>5.37 (2.37)</td><td>5.44 (2.47)</td><td>3.70 (1.76)</td><td>3.60 (1.68)</td><td>5.13 (2.33)</td></tr><tr><td rowspan="4">Q3-8B</td><td>EAGLE-3</td><td>5.58 (1.68)</td><td>5.34 (1.79)</td><td>4.88 (1.78)</td><td>4.30 (1.35)</td><td>4.68 (1.51)</td><td>5.11 (1.66)</td><td>2.92 (1.00)</td><td>2.75 (0.94)</td><td>4.44 (1.46)</td></tr><tr><td>DFlash</td><td>6.00 (2.37)</td><td>5.91 (2.66)</td><td>5.41 (2.69)</td><td>4.89 (2.00)</td><td>5.27 (2.25)</td><td>5.34 (2.19)</td><td>3.41 (1.56)</td><td>3.32 (1.51)</td><td>4.94 (2.15)</td></tr><tr><td>DSpark</td><td>6.46 (2.50)</td><td>6.38 (2.82)</td><td>5.89 (2.88)</td><td>5.47 (2.17)</td><td>5.81 (2.41)</td><td>5.77 (2.30)</td><td>3.82 (1.70)</td><td>3.73 (1.65)</td><td>5.42 (2.30)</td></tr><tr><td>LONGSPARK</td><td>6.24 (2.59)</td><td>6.14 (2.96)</td><td>5.57 (3.01)</td><td>5.32 (2.28)</td><td>5.50 (2.49)</td><td>5.53 (2.37)</td><td>3.65 (1.78)</td><td>3.60 (1.74)</td><td>5.19 (2.40)</td></tr><tr><td rowspan="4">Q3-14B</td><td>EAGLE-3</td><td>5.55 (1.90)</td><td>5.23 (1.97)</td><td>4.69 (1.89)</td><td>4.09 (1.47)</td><td>4.51 (1.63)</td><td>4.90 (1.71)</td><td>2.82 (1.08)</td><td>2.68 (1.03)</td><td>4.31 (1.59)</td></tr><tr><td>DFlash</td><td>6.01 (2.59)</td><td>5.90 (2.85)</td><td>5.33 (2.77)</td><td>4.88 (2.19)</td><td>5.22 (2.39)</td><td>5.21 (2.16)</td><td>3.42 (1.67)</td><td>3.32 (1.62)</td><td>4.91 (2.28)</td></tr><tr><td>DSpark</td><td>6.47 (2.70)</td><td>6.37 (2.99)</td><td>5.81 (2.94)</td><td>5.47 (2.36)</td><td>5.77 (2.54)</td><td>5.66 (2.26)</td><td>3.89 (1.83)</td><td>3.76 (1.77)</td><td>5.40 (2.43)</td></tr><tr><td>LONGSPARK</td><td>6.32 (2.74)</td><td>6.21 (3.10)</td><td>5.54 (3.00)</td><td>5.39 (2.46)</td><td>5.52 (2.57)</td><td>5.46 (2.29)</td><td>3.79 (1.89)</td><td>3.68 (1.83)</td><td>5.24 (2.49)</td></tr></table>

## D.2 LONG-CONTEXT PERFORMANCE ACROSS MODEL SCALES

We then check whether the long-context advantage is specific to the 8B target.

Table 4 extends the Qwen3-8B evaluation to 4B and 14B. Across all three model scales, LONGSPARK achieves the highest average TPS and lowest average TPOT on each benchmark.

Table 4: Long-context decoding on Qwen3-4B and 14B, using the evaluation protocol and reporting conventions of Table 2.
<table><tr><td rowspan="2">Target</td><td rowspan="2">Method</td><td colspan="2">LongSpec 32K</td><td colspan="2">CodeSpan 64K</td><td colspan="2">LSWE 128K</td></tr><tr><td>TPS ↑</td><td>TPOT ↓</td><td>TPS ↑</td><td>TPOT ↓</td><td>TPS ↑</td><td>TPOT ↓</td></tr><tr><td rowspan="5">Q3-4B</td><td>Vanilla</td><td>1250.1</td><td>12.7</td><td>864.4</td><td>18.2</td><td>185.1</td><td>83.0</td></tr><tr><td>EAGLE-3</td><td>1810.5</td><td>8.7</td><td>1197.2</td><td>13.1</td><td>202.0</td><td>77.4</td></tr><tr><td>DFlash</td><td>1453.1</td><td>10.9</td><td>748.9</td><td>21.1</td><td>175.6</td><td>88.7</td></tr><tr><td>DSpark</td><td>1761.8</td><td>8.9</td><td>884.0</td><td>17.9</td><td>184.0</td><td>83.8</td></tr><tr><td>LONGSPARK</td><td>2274.6</td><td>6.9</td><td>1497.0</td><td>10.4</td><td>233.2</td><td>66.2</td></tr><tr><td rowspan="5">Q3-14B</td><td>Vanilla</td><td>951.8</td><td>16.6</td><td>615.7</td><td>25.5</td><td>115.1</td><td>133.6</td></tr><tr><td>EAGLE-3</td><td>964.4</td><td>16.4</td><td>628.1</td><td>25.0</td><td>112.0</td><td>137.0</td></tr><tr><td>DFlash</td><td>1248.8</td><td>12.5</td><td>832.9</td><td>18.8</td><td>112.3</td><td>136.4</td></tr><tr><td>DSpark</td><td>1416.2</td><td>11.0</td><td>945.1</td><td>16.5</td><td>121.9</td><td>127.8</td></tr><tr><td>LONGSPARK</td><td>1464.2</td><td>10.7</td><td>1002.8</td><td>15.5</td><td>133.5</td><td>115.3</td></tr></table>

## D.3 LONG-CONTEXT COMPONENT ABLATIONS

We evaluate the contribution of each context component on workloads spanning 32K–128K tokens. Table 5 reports accepted length and serving performance for the full drafter and variants that remove one component at a time.

Table 5: Long-context component ablations. All other settings follow Table 2.  
(a) Accepted length � ↑
<table><tr><td>Variant</td><td>LongSpec 32K</td><td>CodeSpan 64K</td><td>LSWE 128K</td></tr><tr><td>LONGSPARK</td><td> ${ \bf 2 . 8 8 \pm 0 . 1 1 }$ </td><td> ${ \bf 3 . 4 0 \pm 0 . 1 1 }$ </td><td> $2 . 5 3 \pm 0 . 2 2$ </td></tr><tr><td>w/o global summary</td><td> $2 . 7 0 \pm 0 . 0 9$ </td><td> $3 . 1 8 \pm 0 . 1 1$ </td><td> $2 . 2 5 \pm 0 . 0 6$ </td></tr><tr><td>w/o boundary state</td><td> $2 . 4 4 \pm 0 . 0 6$ </td><td> $2 . 8 2 \pm 0 . 1 0$ </td><td> $2 . 0 8 \pm 0 . 1 0$ </td></tr><tr><td>w/o recent KV window</td><td> $1 . 9 0 \pm 0 . 0 7$ </td><td> $1 . 7 7 \pm 0 . 0 4$ </td><td> $1 . 7 6 \pm 0 . 0 8$ </td></tr></table>

(b) Serving performance
<table><tr><td>Variant</td><td colspan="2">LongSpec 32K</td><td colspan="2">CodeSpan 64K</td><td colspan="2">LSWE 128K</td></tr><tr><td></td><td>TPS ↑</td><td>TPOT ↓</td><td>TPS ↑</td><td>TPOT ↓</td><td>TPS ↑</td><td>TPOT ↓</td></tr><tr><td>LONGSPARK</td><td> $1 5 9 7 . 3 \pm 7 7 . 0$  _</td><td> ${ \bf 9 . 7 9 \pm 0 . 4 6 }$  </td><td> $1 5 2 7 . 2 \pm 1 0 3 . 8$  </td><td> ${ \bf 1 0 . 2 6 \pm 0 . 6 7 }$  </td><td> ${ \bf 2 0 8 . 4 } \pm { \bf 4 5 . 3 }$  </td><td> $7 5 . 3 9 \pm 1 3 . 9 8$ </td></tr><tr><td>w/o global summary</td><td> $1 5 1 4 . 4 \pm 7 7 . 8$ </td><td> $1 0 . 3 6 \pm 0 . 5 2$ </td><td> $1 4 6 2 . 3 \pm 9 0 . 5$ </td><td> $1 0 . 7 4 \pm 0 . 6 5$ </td><td> $1 8 3 . 5 \pm 1 0 . 6$ </td><td> $8 2 . 9 0 \pm 4 . 8 9$ </td></tr><tr><td>w/o boundary state</td><td> $1 4 6 0 . 5 \pm 7 6 . 9$ </td><td> $1 0 . 7 5 \pm 0 . 5 5$ </td><td> $1 3 9 1 . 6 \pm 7 3 . 8$ </td><td> $1 1 . 2 9 \pm 0 . 5 8$ </td><td> $1 7 7 . 4 \pm 1 9 . 2$ </td><td> $8 6 . 4 4 \pm 8 . 8 5$ </td></tr><tr><td>w/o recent KV window</td><td> $1 1 4 6 . 3 \pm 5 8 . 9$ </td><td> $1 3 . 7 4 \pm 0 . 7 1$ </td><td> $9 7 6 . 3 \pm 4 8 . 5 $ </td><td> $1 6 . 2 0 \pm 0 . 7 9$ </td><td> $1 7 8 . 1 \pm 2 3 . 9$ </td><td> $8 6 . 8 8 \pm 1 1 . 9 6$ </td></tr></table>

## D.4 PERFORMANCE ACROSS CONCURRENCY LEVELS

We next check how the advantage changes with serving load.

Table 6 extends the Qwen3-8B evaluation to concurrency levels 8, 32, and 128, using target TP1, $T = 1$ , and up to 2,048 generated tokens per request. The concurrency-32 results reproduce the corresponding entries in Table 1. We observe a clear trend: while DSpark is competitive at low concurrency $( \mathrm { e } . \mathrm { g } . , \ : C = 8 )$ , LONGSPARK becomes increasingly dominant as the number of concurrent requests grows. ${ \mathrm { A t ~ } } C = 1 2 8 { \mathrm { . } }$ LONGSPARK maintains a significant throughput lead over all baselines. This scaling behavior stems from the minimal state overhead of LONGSPARK. Un like existing drafters that must manage and access large KV caches for each concurrent request, LONGSPARK’s fixed-cost drafting mechanism avoids the memory-bandwidth bottleneck associated with scaling the drafter’s state. Consequently, LONGSPARK is exceptionally well-suited for highconcurrency serving environments where maximizing total throughput is critical.

Table 6: Concurrency scaling on Qwen3-8B (�: concurrency). Entries follow Table 1, with speedups relative to autoregressive decoding at the same concurrency.
<table><tr><td rowspan="2">C</td><td rowspan="2">Drafter</td><td colspan="3">Math</td><td colspan="3">Code</td><td colspan="2">Chat</td><td>Overall</td></tr><tr><td>GSM8K</td><td>MATH-500</td><td>AIME25</td><td>MBPP</td><td>HumanEval</td><td>LCB</td><td>MT-Bench</td><td>Alpaca</td><td>Avg.</td></tr><tr><td rowspan="4">8</td><td>EAGLE-3</td><td>5.23 (1.92)</td><td>4.70 (1.82)</td><td>4.01 (1.61)</td><td>3.91 (1.48)</td><td>4.34 (1.65)</td><td>4.18 (1.58)</td><td>2.72 (1.06)</td><td>2.54 (0.99)</td><td>3.95 (1.51)</td></tr><tr><td>DFlash</td><td>5.38 (2.68)</td><td>4.92 (2.64)</td><td>4.07 (2.27)</td><td>4.37 (2.25)</td><td>4.68 (2.44)</td><td>4.50 (2.27)</td><td>3.09 (1.66)</td><td>2.99 (1.60)</td><td>4.25 (2.23)</td></tr><tr><td>DSpark</td><td>6.16 (2.98)</td><td>5.78 (3.03)</td><td>5.00 (2.72)</td><td>5.19 (2.59)</td><td>5.47 (2.78)</td><td>5.12 (2.49)</td><td>3.67 (1.92)</td><td>3.60 (1.87)</td><td>5.00 (2.55)</td></tr><tr><td>LONGSPARK</td><td>5.96 (2.96)</td><td>5.57 (2.98)</td><td>4.69 (2.63)</td><td>5.03 (2.54)</td><td>5.14 (2.69)</td><td>4.86 (2.44)</td><td>3.51 (1.89)</td><td>3.46 (1.82)</td><td>4.78 (2.49)</td></tr><tr><td rowspan="4">32</td><td>EAGLE-3</td><td>5.26 (1.49)</td><td>4.75 (1.50)</td><td>4.02 (1.37)</td><td>3.92 (1.16)</td><td>4.32 (1.30)</td><td>4.17 (1.28)</td><td>2.67 (0.85)</td><td>2.54 (0.82)</td><td>3.96 (1.22)</td></tr><tr><td>DFlash</td><td>5.36 (1.96)</td><td>4.91 (2.02)</td><td>4.07 (1.83)</td><td>4.36 (1.66)</td><td>4.67 (1.82)</td><td>4.44 (1.68)</td><td>3.09 (1.28)</td><td>3.00 (1.24)</td><td>4.24 (1.69)</td></tr><tr><td>DSpark</td><td>6.17 (2.17)</td><td>5.80 (2.32)</td><td>4.99 (2.19)</td><td>5.18 (1.89)</td><td>5.48 (2.06)</td><td>5.09 (1.84)</td><td>3.69 (1.47)</td><td>3.59 (1.42)</td><td>5.00 (1.92)</td></tr><tr><td>LONGSPARK</td><td>5.93 (2.24)</td><td>5.57 (2.42)</td><td>4.72 (2.27)</td><td>5.03 (1.98)</td><td>5.16 (2.11)</td><td>4.83 (1.88)</td><td>3.53 (1.54)</td><td>3.46 (1.49)</td><td>4.78 (1.99)</td></tr><tr><td rowspan="4">128</td><td>EAGLE-3</td><td>5.24 (1.10)</td><td>4.73 (1.06)</td><td>4.02 (1.02)</td><td>3.91 (0.82)</td><td>4.33 (0.92)</td><td>4.14 (0.99)</td><td>2.69 (0.60)</td><td>2.54 (0.58)</td><td>3.95 (0.88)</td></tr><tr><td>DFlash</td><td>5.37 (1.33)</td><td>4.92 (1.30)</td><td>4.09 (1.22)</td><td>4.37 (1.07)</td><td>4.67 (1.16)</td><td>4.46 (1.19)</td><td>3.09 (0.80)</td><td>3.00 (0.79)</td><td>4.25 (1.11)</td></tr><tr><td>DSpark</td><td>6.17 (1.49)</td><td>5.80 (1.50)</td><td>4.98 (1.45)</td><td>5.19 (1.25)</td><td>5.48 (1.32)</td><td>5.11 (1.29)</td><td>3.68 (0.93)</td><td>3.59 (0.93)</td><td>5.00 (1.27)</td></tr><tr><td>LONGSPARK</td><td>5.95 (1.63)</td><td>5.57 (1.62)</td><td>4.72 (1.58)</td><td>5.03 (1.35)</td><td>5.16 (1.41)</td><td>4.84 (1.36)</td><td>3.52 (1.01)</td><td>3.46 (1.00)</td><td>4.78 (1.37)</td></tr></table>

## D.5 FIXED-REQUEST END-TO-END PERFORMANCE

We finally report end-to-end completion time for a fixed set of requests.  
Table 7: Mean end-to-end completion time for 32 requests at concurrency 16, averaged over three seeds (seconds; lower is better).
<table><tr><td>Method</td><td>LongSpec 32K</td><td>CodeSpan 64K</td><td>LSWE 128K</td></tr><tr><td>EAGLE-3</td><td>82.61</td><td>175.67</td><td>127.39</td></tr><tr><td>DFlash</td><td>71.11</td><td>202.01</td><td>138.75</td></tr><tr><td>DSpark</td><td>63.66</td><td>153.60</td><td>116.23</td></tr><tr><td>LONGSPARK</td><td>56.14</td><td>106.77</td><td>99.71</td></tr></table>

We complement the steady-state throughput measurements with the time required to complete a fixed set of 32 requests on each long-context benchmark. All requests arrive together and are served with concurrency 16 using Qwen3-8B, Target TP4, � = 1, and a maximum of 8,192 generated tokens. Elapsed time includes prefill, queuing, decoding, and completion of the final requests, excluding model loading and warm-up. Table 7 reports mean completion times. Responses are sampled independently, so completion times reflect both serving efficiency and variation in generated response length. LONGSPARK achieves the lowest mean completion time on all three workloads.