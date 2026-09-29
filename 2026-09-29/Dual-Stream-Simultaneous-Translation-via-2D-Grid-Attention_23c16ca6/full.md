# Dual-Stream Simultaneous Translation via 2D Grid Attention

Yu Pu and Wei-Qiang Zhang , Senior Member, IEEE

Abstract—Simultaneous machine translation must generate target tokens before the source input is complete. Existing approaches address this through post-hoc read-write policies, leaving the attention mechanism unaware of bidirectional stream dependencies. We propose a dual-stream attention framework that represents source and target streams as a twodimensional grid of hidden states and models their interaction through four structurally distinct attention types merged via joint QK Softmax normalization. Two approximations— broadcast and Hadamard—reduce the per-layer complexity from $O ( X ^ { 2 } Y + X Y ^ { 2 } )$ to $O ( X ^ { 2 } + Y ^ { 2 } + X Y )$ ) with provably decaying error. Training uses a self-guided loop: a per-cell loss heatmap drives dynamic-programming path recovery, which generates read/write decision supervision labels without external alignment. An incremental KV cache with anchored rotary position embeddings enables efficient streaming inference. On Chineseto-English simultaneous translation, the proposed model outperforms the Wait-k baseline by +5.66 BLEURT and +10.36 COMET at comparable latency, and surpasses the non-streaming reference on COMET at a fraction of the response delay.

Index Terms—Simultaneous machine translation, dual-stream attention, 2D grid representation, incremental decoding, readwrite policy.

## I. INTRODUCTION

IMULTANEOUS machine translation (SimulMT) targets a qualitatively different operating regime from offline translation: a target-language token must be emitted before the full source sentence is [1]. This constraint is not merely a latency budget; it demands that the model continuously reason about an incomplete, still-arriving source stream while producing a coherent, causally valid output stream [2]. The resulting latency-quality trade-off—emit too early and translation degrades; wait too long and real-time utility is lost—is the central challenge of the field [3].

The dominant paradigm addresses this trade-off through post-hoc waiting policies layered on top of standard offline architectures [4]. Fixed policies such as Wait-k [5] prescribe a deterministic schedule: read k source tokens, then alternate one read with one write. Adaptive policies [6], [7] learn contentdependent schedules via reinforcement learning or multi-path training. These approaches are practical and effective, yet they share a structural limitation: the underlying attention mechanism remains unaware of the bidirectional dependency between streams [8]. The decoder attends to a frozen source prefix and generates autoregressively, with no mechanism for the source representations to respond to what has already been generated, or for the output representations to dynamically integrate newly arriving source tokens. The read-write schedule is thus imposed on top of the computation, rather than being woven into it.

We address this gap by modeling stream dependencies directly at the attention level, treating simultaneous translation as a bidirectional interaction between two co-evolving streams. Instead of a one-dimensional input sequence and a separately unrolled decoder, we represent the joint state at any point (x, y)—having consumed x source tokens and emitted y target tokens—as a node in a 2D grid of hidden states, indexed by both source position and output time-step. Within this grid, we identify four structurally distinct attention patterns and instantiate each with a dedicated computation:

• I→I (source self-attention, causal along x): captures leftto-right dependencies within the source stream.

• O→O (target self-attention, causal along y): maintains autoregressive coherence in the output stream.

• I←O (output-to-input cross-attention, prefix along y): allows source representations to be updated by the partial output generated so far.

• O←I (input-to-output cross-attention, prefix along x): allows output representations to attend to the source prefix available at each state.

Because each output node (x, y) receives contributions from both its own-stream and cross-stream attention, we derive a joint QK Softmax formulation that merges the unnormalized statistics from O→O and O←I in a single numerically stable pass.

Full grid computation is quadratic in both X and Y, which is prohibitive at scale. We therefore develop two lowcomplexity approximations: (1) a broadcast approximation that computes I←O attention only at $y = 0$ and replicates the result along the y axis, reducing its cost from $O ( X ^ { 2 } Y )$ to $O ( X ^ { 2 } )$ ; and (2) a Hadamard approximation that replaces the cross-position dot product in I←O with an element-wise product at the same position, cutting the remaining crossstream cost to O(XY). We derive a formal error bound showing that the resulting approximation error decays as $O ( 1 / { \sqrt { y } } )$ as the output prefix grows, providing theoretical justification for the approximation quality.

Training exploits the 2D grid structure to derive selfguided supervision without external alignment labels. We compute a loss heatmap $\mathcal { L } ( x , y )$ from grid-level languagemodel log-likelihoods, then recover the optimal monotone path $\pi ^ { * }$ through this heatmap via dynamic programming. The path determines an emission mask that labels each grid state as “should emit” or “should wait”, providing direct binary supervision for a lightweight decision head. Crucially, as translation quality improves during training, the heatmap reshapes, yielding better paths and therefore better decision labels—a self-reinforcing loop that tightens both objectives simultaneously.

At inference time, incremental decoding requires maintaining eight categories of KV-cache tensors per layer to avoid redundant recomputation. We design an incremental KV cache that exploits the causal and prefix masking structure of each attention type, and pair it with an anchored RoPE strategy that assigns output-stream positions starting from $X _ { \mathrm { t o t a l } }$ , eliminating position-embedding conflicts between the two streams.

Experiments on Chinese-to-English simultaneous translation demonstrate that at a comparable output speed (FRL ≈ 3.3 vs. 3.0), our model achieves +5.66 BLEURT and +10.36 COMET over the Wait-k (k=3) baseline, while also outperforming the non-streaming reference on COMET at the most permissive latency setting (θ=0.9, COMET 64.73 vs. 63.85).

The contributions of this paper are as follows:

1) A 2D grid attention framework with four structurally distinct stream interaction types, together with a joint QK Softmax for numerically stable merging of crossstream attention statistics.

2) Two efficient approximations—broadcast and Hadamard—with a formal error analysis showing $O ( 1 / { \sqrt { y } } )$ decay.

3) A self-guided co-training procedure based on loss heatmap, DP path recovery, and an emission mask, requiring no external alignment supervision.

4) An incremental inference design with a structured KV cache and anchored RoPE that supports low-latency streaming decoding.

The remainder of the paper is organized as follows. Section II reviews related work. Section III presents the duplex attention model and joint QK Softmax. Section IV develops the low-complexity approximations and error analysis. Section V describes the training and inference algorithms. Section VI reports experimental results, and Section VII concludes.

## II. RELATED WORK

## A. Simultaneous Translation Strategies and Policies

Simultaneous translation (SimulMT) requires generating target-language output before the source input is complete, departing fundamentally from the offline “read-then-translate” paradigm [9]. The central challenge is the latency-quality trade-off: early emission risks quality degradation from insufficient context, while excessive waiting violates real-time constraints [10]. Task variants span text-to-text (SimulMT), speech-to-text (SimulST), and speech-to-speech (SimulS2S) translation, with SimulMT serving as the methodological foundation for the others [11].

Prior work has pursued this challenge through two broad families of read-write policies. Fixed policies prescribe a deterministic schedule regardless of content. The Wait-k strategy [5], [12], [13], [14] is the most widely adopted: the decoder first reads k source tokens and then alternates between consuming one source token and emitting one target token. The prefix-to-prefix framework [9], [15], [16] takes a complementary approach, training models to translate from source prefixes alone; within this framework, Ma et al. [17] proposed STACL, which incorporates an anticipation mechanism to improve prefix-conditioned generation under limited context.

Adaptive policies, by contrast, learn content-dependent schedules. Arivazhagan et al. [6] employ deep reinforcement learning to directly optimize a read-write scheduling policy, rewarding favorable latency-quality outcomes. Xia et al. [9] introduce speculative decoding, where a lightweight predictor proposes candidate tokens verified by the main model, reducing effective latency without quality degradation. Dalvi et al. [7] propose incremental training with multi-path optimization, exposing the model to diverse read-write trajectories to improve robustness across latency budgets. Curriculum-based approaches [18] further ease training by gradually decreasing the latency tolerance across epochs.

## B. Simultaneous Translation Model Training and Evaluation

Architecturally, most simultaneous systems extend the standard autoregressive Transformer [19], [20], [21]. Policyintegrated variants introduce dedicated read/write decision heads alongside token logits, decoupling translation quality from scheduling [22], [23], [24]. Monotonic attention mechanisms constrain the source-target alignment to be strictly leftto-right, enabling online decoding without look-ahead [25], [26]. Dual-stream designs separate input reading from output generation into distinct submodules, making bidirectional dependencies explicit within the computation graph [27]. Compared to offline NMT, simultaneous systems commonly adopt multi-path training [7], [28], randomly sampling readwrite trajectories at each training step so that the model learns to handle a range of prefix lengths simultaneously. Knowledge distillation from an offline teacher further stabilizes this process: the offline model provides soft targets under full source context, guiding the simultaneous model toward more robust prefix-conditioned predictions [29], [30], [31]. Prefixalignment augmentation and curriculum scheduling [18] are complementary strategies that similarly mitigate the information scarcity inherent to simultaneous decoding.

For evaluation, the field employs latency-specific metrics alongside standard quality measures such as BLEU [32]. Average Lagging (AL) [5] quantifies the mean temporal offset between source and target token generation; Differentiable AL [7] makes this quantity amenable to end-to-end gradientbased optimization. First Token Latency is particularly salient for user perception in real-time settings. Most evaluations are conducted on WMT [33], [34], [35] and IWSLT [36], [37], [38] benchmarks, reporting results along the quality-latency Pareto frontier under varying latency budgets.

## III. DUPLEX ATTENTION MODELING

## A. Core Challenge of Simultaneous Translation

Traditional interaction systems follow a “non-streaming” paradigm: speakers take turns occupying the channel, and the system begins output only after receiving the complete input [39]. This paradigm is fundamentally inadequate for simultaneous interpretation, where the interpreter must continuously produce the target-language output while simultaneously listening to the source [40]. Under non-streaming modeling, the system either introduces unacceptable translation latency by waiting for the input to end, or sacrifices translation quality by outputting prematurely.

The core challenge of simultaneous translation modeling is: how to explicitly represent the bidirectional dependency between the input stream and the output stream within the same neural network [41]. Let the input sequence be $\begin{array} { l c l } { \mathbf { i } } & { = } & { \left( i _ { 1 } , i _ { 2 } , \ldots , i _ { X } \right) } \end{array}$ and the output sequence be $\begin{array} { r l } { \mathbf { O } } & { { } = } \end{array}$ $\left( o _ { 1 } , o _ { 2 } , \ldots , o _ { Y } \right)$ . At any state $( x , y )$ (having seen x inputs and generated y outputs), the model must simultaneously satisfy:

1) Output causality: $o _ { y + 1 }$ can only depend on $i _ { 1 } , \dots , i _ { x }$ and $o _ { 1 } , \ldots , o _ { y } ,$ , and cannot “see” future inputs;

2) Input awareness: output generation should dynamically adapt as new inputs arrive, rather than being deferred until the input ends.

Existing work typically inserts waiting strategies (e.g., CIF [42], Wait-k) as post-processing, without directly modeling these dependencies at the attention level of the language model. Our dual-stream model explicitly introduces crossstream dependencies inside the attention mechanism of a large language model, simultaneously satisfying both constraints within a unified framework.

## B. Inference State Machine

We formalize simultaneous inference as a deterministic state machine. Define state $\boldsymbol { s } = ( x , y )$ as “x input tokens have been exposed to the model and y output tokens have been emitted externally.” The state machine supports two actions:

• EMIT: generate the (y + 1)-th output token; state transitions $( x , y ) \to ( x , y + 1 ) \quad$ ;

• WAIT: expose the (x+1)-th input token; state transitions $( x , y )  ( x + 1 , y )$

At state $( x , y )$ , the model must output: (1) the probability distribution $p ( o _ { y + 1 } \mid i _ { 1 : x } , o _ { 1 : y } )$ over the next output token; (2) the EMIT/WAIT decision probability $p ( \mathrm { E M I T } \mid i _ { 1 : x } , o _ { 1 : y } )$

Table I lists the principal symbols used throughout this and the following sections.

## C. 2D Grid Hidden State Representation

Standard autoregressive language models represent each token in a sequence as a one-dimensional vector. We extend dual-stream modeling to a two-dimensional grid. For layer l, define:

$$
\mathbf { I } ^ { ( l ) } \in \mathbb { R } ^ { B \times X \times Y \times D } , \quad \mathbf { O } ^ { ( l ) } \in \mathbb { R } ^ { B \times X \times Y \times D } ,\tag{1}
$$

where $\mathbf { I } _ { x , y } ^ { ( l ) }$ denotes the hidden state of the x-th input token at layer l under the context of having seen x inputs and y outputs, and $\mathbf { O } _ { x , y } ^ { ( l ) }$ denotes the corresponding hidden state for the y-th output token. The same token thus has different hidden state representations at different states $( x , y )$ , reflecting the variation of its semantics under different contexts.

At layer 0 (the embedding layer), the grid is initialized by broadcasting token embeddings:

$$
\mathbf { I } _ { x , y } ^ { ( 0 ) } = \operatorname { E m b e d } ( i _ { x } ) , \quad \mathbf { O } _ { x , y } ^ { ( 0 ) } = \operatorname { E m b e d } ( o _ { y } ) .\tag{2}
$$

Note that $\mathbf { I } _ { x , y } ^ { ( 0 ) }$ is independent of $y ,$ and $\mathbf { O } _ { x , y } ^ { ( 0 ) }$ is independent of x. In subsequent Transformer layers, cross-stream attention introduces mutual dependencies along both dimensions.

![](images/0bd179fd111be47bd295082a728825074f505b7e3e997a5ec5f9dc1838afea01.jpg)

![](images/99e6a6a42e397cb05e7c77ab7142afebbf63b4a2f27bc45f0bd3cb96390aa4f4.jpg)  
Fig. 1. Illustration of the 2D grid hidden state representation. Each grid cell $( x , y )$ maintains hidden states for both the input stream $\mathbf { I } _ { x , y } ^ { ( l ) }$ and the output stream $\mathbf { O } _ { x , y } ^ { ( l ) } .$ The horizontal axis corresponds to input positions and the vertical axis to output positions.

For positional encoding, we assign input tokens position $0 , 1 , \ldots , X - 1$ and output tokens positions $X , X + 1 , \ldots , X +$ $Y - 1$ within the same Rotary Position Embedding (RoPE) sequence:

$$
{ \bf Q } _ { I } [ x , y , h ] , { \bf K } _ { I } [ x , y , h ] \stackrel { \mathrm { R o P E } ( x ) } { \overbrace { \mathrm { \ p o s i t i o n } } } x ,\tag{3}
$$

$$
\mathbf { Q } _ { O } [ x , y , h ] , \mathbf { K } _ { O } [ x , y , h ] \xleftarrow { \mathrm { R o P E } ( X + y ) } \mathrm { p o s i t i o n } X + y .\tag{4}
$$

This design ensures that intra-stream relative positions are consistent with standard autoregressive models, enabling full reuse of pre-trained positional priors.

## D. Four Types of Attention

In each Transformer layer [43], attention contexts are computed separately for the input and output streams, followed by residual connections and MLP to obtain the next-layer hidden states. The dual-stream attention comprises four types, as summarized in Table II.

For layer l, both streams first apply LayerNorm, and their query, key, and value tensors are projected via shared weight matrices $\mathbf { \bar { \boldsymbol { W } } } _ { Q } \in \mathbb { R } ^ { D \times H H _ { d } }$ and $\dot { W _ { K } } , \mathbf { \bar { \it W _ { V } } } \in \mathbb { R } ^ { D \times H _ { k } H _ { d } }$ (with GQA heads expanded to H):

$$
\mathbf { Q } _ { I } = \tilde { \mathbf { I } } ^ { ( l ) } W _ { Q } , \quad \mathbf { K } _ { I } = \tilde { \mathbf { I } } ^ { ( l ) } W _ { K } , \quad \mathbf { V } _ { I } = \tilde { \mathbf { I } } ^ { ( l ) } W _ { V } ,\tag{5}
$$

$$
\mathbf { Q } _ { O } = \tilde { \mathbf { O } } ^ { ( l ) } W _ { Q } , \quad \mathbf { K } _ { O } = \tilde { \mathbf { O } } ^ { ( l ) } W _ { K } , \quad \mathbf { V } _ { O } = \tilde { \mathbf { O } } ^ { ( l ) } W _ { V } ,\tag{6}
$$

where $\tilde { \mathbf { I } } ^ { ( l ) } , \tilde { \mathbf { O } } ^ { ( l ) }$ are the LayerNorm-normalized hidden states and $s = H _ { d } ^ { - 1 / 2 }$ is the attention scaling factor. Sharing $W _ { Q } , W _ { K } , W _ { V }$ across both streams fully reuses pre-trained weights.

TABLE I PRINCIPAL NOTATION
<table><tr><td>Symbol</td><td>Meaning</td><td>Typical Value</td></tr><tr><td>B</td><td>Batch size</td><td></td></tr><tr><td>X</td><td>Input sequence length</td><td></td></tr><tr><td>Y</td><td>Output sequence length</td><td></td></tr><tr><td> $D$ </td><td>Model hidden dimension</td><td>896</td></tr><tr><td> $H$ </td><td>Number of attention heads</td><td>14</td></tr><tr><td> $H _ { k }$ </td><td>Number of Key/Value heads (GQA)</td><td>2</td></tr><tr><td> $H _ { d }$ </td><td>Per-head dimension;  $H _ { d } = D / H$ </td><td>64</td></tr><tr><td> $L$ </td><td>Number of Transformer layers</td><td>24</td></tr><tr><td> $s$ </td><td>Attention scaling factor;  $s = H _ { d } ^ { - 1 / 2 }$ </td><td></td></tr><tr><td> $\odot$ </td><td>Element-wise (Hadamard) product</td><td></td></tr><tr><td> $\mathbf { I } ^ { ( l ) }$ </td><td>Input-stream hidden states at layer  $l ; { \bf { I } } ^ { ( l ) } \in \mathbb { R } ^ { B \times X \times Y \times D }$ </td><td></td></tr><tr><td> $\mathbf { O } ^ { ( l ) }$ </td><td>Output-stream hidden states at layer  $\mathbf { \Phi } _ { l ; \mathbf { O } ^ { ( l ) } } \in \mathbb { R } ^ { B \times X \times Y \times D }$ </td><td></td></tr><tr><td> $\tilde { \mathbf { I } } ^ { ( l ) } , \tilde { \mathbf { O } } ^ { ( l ) }$ </td><td>LayerNorm-normalized inputs to attention at layer l</td><td></td></tr><tr><td> $W _ { Q } , W _ { K } , W _ { V }$ </td><td>Shared QKV projection matrices;  $W _ { Q } \in \mathbb { R } ^ { D \times \check { H } H _ { d } } , W _ { K } , W _ { V } \in \mathbb { R } ^ { D \times H _ { k } H _ { d } }$ </td><td></td></tr><tr><td> $W _ { O }$ </td><td>Output projection matrix;  $W _ { O } \in \mathbb { R } ^ { \tilde { H } _ { H _ { d } \times D } }$ </td><td></td></tr><tr><td> $ { \mathcal Ḋ L Ḍ } ( x , y )$ </td><td>Grid loss at state  $( x , y )$ </td><td></td></tr><tr><td> $\pi ^ { * }$ </td><td>Optimal monotone path from  $( 0 , 0 ) \ { \mathrm { t o } } \ ( X , Y )$ </td><td></td></tr><tr><td> $\lambda$ </td><td>Latency tolerance hyperparameter in path score</td><td>0.1</td></tr><tr><td> $\theta$ </td><td>EMIT decision threshold at inference time</td><td></td></tr></table>

TABLE II  
FOUR TYPES OF DUPLEX ATTENTION AND THEIR PROPERTIES
<table><tr><td>Type</td><td>Query</td><td>Key/Value</td><td>Causal Constraint</td></tr><tr><td>I→I</td><td> $\mathbf { Q } _ { I }$ </td><td> $\mathbf { K } _ { I } , \mathbf { V } _ { I }$ </td><td>Causal in x  $( x ^ { \prime } \leq x )$ </td></tr><tr><td>0→0</td><td> $\mathbf { Q } _ { O }$ </td><td> $\mathbf { K } _ { O } , \mathbf { V } _ { O }$ </td><td>Causal in y  $( y ^ { \prime } \leq y )$ </td></tr><tr><td>I←0</td><td> $\mathbf { Q } _ { I }$ </td><td> $\mathbf { K } _ { O } , \mathbf { V } _ { O }$ </td><td>Prefix in y  $( y ^ { \prime } \leq y )$ </td></tr><tr><td>0←I</td><td> $\mathbf { Q } _ { O }$ </td><td> $\mathbf { K } _ { I } , \mathbf { V } _ { I }$ </td><td>Prefix in  $\check { x } \ ( \check { x } ^ { \prime } \overset { = } { \leq } x )$ </td></tr></table>

![](images/69036a306c0a98cdec38bc4dd5be92c9ba3c6886127cd31b4a57c91132622b57.jpg)  
Q position (query source for I→I or I←O)K, V source for I→I (input self-attention)K, V source for I—O (input attends to output)

Fig. 2. Grid illustration of input stream attention. Left: I→I self-attention, where each cell attends causally along x and results are broadcast along y. Right: I←O cross-attention, where each input cell aggregates from the output prefix $y ^ { \prime } \leq y$  
![](images/435327696edb5c8f85cfe5fa7180fce7ff15f78a787c244c480616b21d626608.jpg)  
Q position (query source for O→O or O←I)K, V source for O→O (output self-attention)K, V source for O←I (output attends to input)

Fig. 3. Grid illustration of output stream attention. Left: O→O self-attention, broadcast along x. Right: O←I cross-attention, where each output cell aggregates from the input prefix $x ^ { \prime } \leq x$

1) I→I Self-Attention (Causal along x): The input stream attends causally to its own tokens:

$$
a _ { x , x ^ { \prime } } ^ { \mathrm { i i } } = s \cdot { \bf Q } _ { I } [ x , h ] \cdot { \bf K } _ { I } [ x ^ { \prime } , h ] ^ { \top } , \quad x ^ { \prime } \leq x ,\tag{7}
$$

$$
\operatorname { c t x } _ { x } ^ { \mathrm { { i i } } } = \sum _ { x ^ { \prime } = 0 } ^ { x } { \frac { e ^ { a _ { x , x ^ { \prime } } ^ { \mathrm { { i i } } } } } { \sum _ { k = 0 } ^ { x } { e ^ { a _ { x , k } ^ { \mathrm { { i i } } } } } } } { \mathbf { V } _ { I } [ x ^ { \prime } , h ] } .\tag{8}
$$

Since this result depends only on $x ,$ it is broadcast across all output positions: $\mathrm { c t x } ^ { \mathrm { i i } } [ x , y ] = \mathrm { c t x } _ { x } ^ { \mathrm { i i } } , \forall y .$

2) O→O Self-Attention (Causal along y): Symmetric to I→I, the output stream attends causally to its own tokens:

$$
a _ { y , y ^ { \prime } } ^ { \mathrm { o o } } = s \cdot { \bf Q } _ { O } [ y , h ] \cdot { \bf K } _ { O } [ y ^ { \prime } , h ] ^ { \top } , \quad y ^ { \prime } \leq y ,\tag{9}
$$

$$
\mathrm { c t x } _ { y } ^ { \mathrm { o o } } = \sum _ { y ^ { \prime } = 0 } ^ { y } \frac { e ^ { a _ { y , y ^ { \prime } } ^ { \mathrm { o o } } } } { \sum _ { k = 0 } ^ { y } e ^ { a _ { y , k } ^ { \mathrm { o o } } } } { \bf V } _ { O } [ y ^ { \prime } , h ] ,\tag{10}
$$

broadcast as c $\begin{array} { r } { \mathrm { t x } ^ { \mathrm { o o } } [ x , y ] = \mathrm { c t x } _ { y } ^ { \mathrm { o o } } , \forall x . } \end{array}$

3) I←O Cross-Attention (Prefix along y): Each input token aggregates information from the output prefix generated so far. Using the Hadamard approximation (Section IV), the attention score at grid cell (x, y) is:

$$
s _ { x , y , h } ^ { \mathrm { i o } } = s \cdot \left( \mathbf { Q } _ { I } [ x , y , h ] \odot \mathbf { K } _ { O } [ x , y , h ] \right) \cdot \mathbf { 1 } ,\tag{11}
$$

and the context is a prefix Softmax weighted sum over $y ^ { \prime } \leq y \colon$

$$
\mathrm { c t x ^ { i o } } [ x , y , h ] = \frac { \displaystyle \sum _ { y ^ { \prime } = 0 } ^ { y } e ^ { s _ { x , y ^ { \prime } , h } ^ { \mathrm { i o } } - m _ { x , y , h } ^ { \mathrm { i o } } } \mathbf { V } _ { O } [ x , y ^ { \prime } , h ] } { Z _ { x , y , h } ^ { \mathrm { i o } } } ,\tag{12}
$$

where $m _ { x , y , h } ^ { \mathrm { i o } } = \mathrm { m a x } _ { y ^ { \prime } \leq y } s _ { x , y ^ { \prime } , h } ^ { \mathrm { i o } }$ is the running maximum and $Z _ { x , y , h } ^ { \mathrm { i o } }$ is the normalization constant.

4) O←I Cross-Attention (Prefix along x): Symmetric to I←O, each output token aggregates the input prefix:

$$
s _ { x , y , h } ^ { \mathrm { o i } } = s \cdot \left( \mathbf { Q } _ { O } [ x , y , h ] \odot \mathbf { K } _ { I } [ x , y , h ] \right) \cdot \mathbf { 1 } ,\tag{13}
$$

$$
\begin{array} { r } { \operatorname { c t x } ^ { \mathrm { o i } } [ x , y , h ] = \frac { \displaystyle \sum _ { x ^ { \prime } = 0 } ^ { x } e ^ { s _ { x ^ { \prime } , y , h } ^ { \mathrm { { o i } } } - m _ { x , y , h } ^ { \mathrm { o i } } } \mathbf { V } _ { I } [ x ^ { \prime } , y , h ] } { { Z _ { x , y , h } ^ { \mathrm { { o i } } } } } . } \end{array}\tag{14}
$$

A naive approach would combine the two attention outputs by simple addition after independent normalization: ctx<sub>I</sub> = $W _ { O } ( \mathrm { c t x ^ { i i } + c t x ^ { i o } } )$ . However, because each of $\mathrm { c t x } ^ { \mathrm { i i } }$ and $\operatorname { c t x } ^ { \mathrm { { i o } } }$ is already individually Softmax-normalized, their sum always weights them equally regardless of the actual attention score magnitudes, preventing the model from learning which type of attention is more informative for a given state.

We instead merge via a joint QK Softmax normalization, combining the unnormalized running statistics $( m , Z , S )$ of both attention types under a shared normalization constant before applying $W _ { O }$ . For the I-stream:

$$
m _ { j } = \operatorname* { m a x } ( m _ { x } ^ { \mathrm { i i } } , \ m _ { x , y } ^ { \mathrm { i o } } ) ,\tag{15}
$$

$$
Z _ { j } = Z _ { x } ^ { \mathrm { i i } } \cdot e ^ { m _ { x } ^ { \mathrm { i i } } - m _ { j } } + Z _ { x , y } ^ { \mathrm { i o } } \cdot e ^ { m _ { x , y } ^ { \mathrm { i o } } - m _ { j } } ,\tag{16}
$$

$$
S _ { j } = S _ { x } ^ { \mathrm { i i } } \cdot e ^ { m _ { x } ^ { \mathrm { i i } } - m _ { j } } + S _ { x , y } ^ { \mathrm { i o } } \cdot e ^ { m _ { x , y } ^ { \mathrm { i o } } - m _ { j } } ,\tag{17}
$$

$$
\mathrm { c t x } _ { I } [ x , y ] = { \cal W } _ { \cal O } ( S _ { j } / Z _ { j } ) .\tag{18}
$$

This is equivalent to placing all I→I and I←O key-value pairs into the same Softmax denominator, allowing the model to allocate attention weight between self-attention and crossattention in a data-driven manner. The O-stream is merged symmetrically using $m _ { y } ^ { \mathrm { o o } }$ and $m _ { x , y } ^ { \mathrm { o i } }$

$$
m _ { j } ^ { O } = \operatorname* { m a x } ( m _ { y } ^ { \mathrm { o o } } , \ m _ { x , y } ^ { \mathrm { o i } } ) ,\tag{19}
$$

$$
Z _ { j } ^ { \cal O } = Z _ { y } ^ { \mathrm { o o } } \cdot e ^ { m _ { y } ^ { \mathrm { o o } } - m _ { j } ^ { \cal O } } + Z _ { x , y } ^ { \mathrm { o i } } \cdot e ^ { m _ { x , y } ^ { \mathrm { o i } } - m _ { j } ^ { \cal O } } ,\tag{20}
$$

$$
\mathrm { c t x } _ { O } [ x , y ] = W _ { O } \Big ( \big ( S _ { y } ^ { \mathrm { o o } } \cdot e ^ { m _ { y } ^ { \mathrm { o o } } - m _ { j } ^ { O } } + S _ { x , y } ^ { \mathrm { o i } } \cdot e ^ { m _ { x , y } ^ { \mathrm { o i } } - m _ { j } ^ { O } } \big ) / Z _ { j } ^ { O } \Big )\tag{21}
$$

The attention outputs are then applied via residual connections to yield the next-layer hidden states:

$$
\mathbf { I } _ { x , y } ^ { ( l ) } = \mathrm { M L P } \left( \mathbf { I } _ { x , y } ^ { ( l - 1 ) } + \mathrm { c t x } _ { I } [ x , y ] \right) ,\tag{22}
$$

$$
\begin{array} { r } { \mathbf { O } _ { x , y } ^ { ( l ) } = \mathrm { M L P } \left( \mathbf { O } _ { x , y } ^ { ( l - 1 ) } + \mathrm { c t x } _ { O } [ x , y ] \right) , } \end{array}\tag{23}
$$

where MLP denotes the standard two-layer feed-forward subblock with LayerNorm (identical in structure to a vanilla Transformer layer). Both streams share the same MLP weights, again fully reusing pre-trained parameters.

## E. Online Prefix Softmax Updates

The core computation of cross-attention is a prefix Softmax weighted sum, and maintaining this sum incrementally is essential for efficient inference. For the I←O cross-attention, we maintain three prefix statistics for each $( x , y , h )$

$$
m _ { x , y } ^ { \mathrm { i o } } \in \mathbb { R } ^ { B \times X \times H } ,\tag{24}
$$

$$
Z _ { x , y } ^ { \mathrm { i o } } \in \mathbb { R } ^ { B \times X \times H } ,\tag{25}
$$

$$
S _ { x , y } ^ { \mathrm { i o } } \in \mathbb { R } ^ { B \times X \times H \times H _ { d } } ,\tag{26}
$$

initialized as $m ^ { ( 0 ) } = - \infty , Z ^ { ( 0 ) } = 0 , S ^ { ( 0 ) } = { \bf 0 }$ . When a new output position y arrives, the update rules are:

$$
m ^ { \mathrm { n e w } } = \operatorname* { m a x } \left( m ^ { \mathrm { o l d } } , \ s _ { x , y } ^ { \mathrm { i o } } \right) ,\tag{27}
$$

$$
\begin{array} { r } { Z ^ { \mathrm { n e w } } = Z ^ { \mathrm { o l d } } \cdot e ^ { m ^ { \mathrm { o l d } } - m ^ { \mathrm { n e w } } } + e ^ { s _ { x , y } ^ { \mathrm { i o } } - m ^ { \mathrm { n e w } } } , } \end{array}\tag{28}
$$

$$
S ^ { \mathrm { n e w } } = S ^ { \mathrm { o l d } } \cdot e ^ { m ^ { \mathrm { o l d } } - m ^ { \mathrm { n e w } } } + e ^ { s _ { x , y } ^ { \mathrm { i o } } - m ^ { \mathrm { n e w } } } \cdot { \bf V } _ { O } [ x , y ] ,\tag{29}
$$

$$
\mathrm { c t x } ^ { \mathrm { i o } } [ x , y ] = S ^ { \mathrm { n e w } } / Z ^ { \mathrm { n e w } } .\tag{30}
$$

This online log-sum-exp algorithm is numerically stable and analogous to the FlashAttention prefix Softmax routine [44]. The $\mathrm { O } {  } \mathrm { I }$ prefix statistics are maintained symmetrically along the x dimension.

## IV. LOW-COMPLEXITY APPROXIMATIONS

Without approximation, the exact computation of duplex attention has total complexity $O ( X ^ { 2 } Y + X Y ^ { 2 } )$ per layer, which is prohibitive for long sequences. We propose two approximations that reduce the complexity to $O ( X ^ { 2 } + Y ^ { 2 } + X Y )$ .

## A. Broadcast Approximation

We compute I→I self-attention only at column $y = 0$ and broadcast the result across the entire grid:

$$
\mathrm { c t x } ^ { \mathrm { i i } } [ x , y ] \approx \mathrm { c t x } ^ { \mathrm { i i } } [ x , 0 ] , \quad \forall y .\tag{31}
$$

The symmetric approximation applies to $\mathrm { O } {  } \mathrm { O }$ . This reduces the self-attention complexity from $O ( X ^ { 2 } Y + X Y ^ { 2 } )$ to $O ( X ^ { 2 } +$ $Y ^ { 2 } )$

The approximation error stems from the y-dependence of the I-stream hidden state $\mathbf { I } ^ { ( l ) } [ x , y ]$ . At initialization (layer 0), ${ \bf I } ^ { ( 0 ) } [ x , y ] = \mathrm { E m b e d } ( i _ { x } )$ is exactly y-independent, so the error is exactly zero at the start of training and grows gradually as cross-stream attention builds up. Semantically, the intrastream attention pattern (which input attends to which) should primarily depend on input content and relative positions, not on how many output tokens have been generated—making this approximation linguistically well-motivated.

## B. Hadamard Approximation

For I←O cross-attention, the exact score at grid position $( x , y )$ for attending to key at position $y ^ { \prime }$ requires a dot product ${ \mathbf { Q } } _ { I } [ x , y , h ] \cdot { \mathbf { K } } _ { O } [ x , y ^ { \prime } , h ] ^ { \top }$ . We approximate this by replacing the query at $y$ with the query at $y ^ { \prime }$ , turning the cross-position dot product into an element-wise product at the same position:

$$
a _ { x , y , y ^ { \prime } } ^ { \mathrm { i o } } \approx s \cdot \left( \mathbf { Q } _ { I } [ x , y ^ { \prime } , h ] \odot \mathbf { K } _ { O } [ x , y ^ { \prime } , h ] \right) \cdot \mathbf { 1 } .\tag{32}
$$

This allows each grid cell $( x , y ^ { \prime } )$ to pre-compute a scalar score, enabling prefix Softmax accumulation with $O ( H _ { d } )$ cost per step, reducing cross-attention complexity to $O ( X Y )$

1) Error Analysis: The Hadamard approximation assumes the query $\mathbf { Q } _ { I } [ x , y , h ]$ is approximately constant in y. The perpair score error is:

$$
\Delta ^ { \mathrm { i o } } ( x , y , y ^ { \prime } ) = s \cdot \left( \mathbf { Q } _ { I } [ x , y , h ] - \mathbf { Q } _ { I } [ x , y ^ { \prime } , h ] \right) \cdot \mathbf { K } _ { O } [ x , y ^ { \prime } , h ] ^ { \top } .\tag{33}
$$

Since ${ \bf Q } _ { I } [ x , y , h ] = { \cal W } _ { Q } \tilde { \bf I } ^ { ( l ) } [ x , y ]$ , the query variation is linearly controlled by the I-stream hidden-state perturbation $\delta { \bf { I } } ^ { ( l ) } [ \dot { x _ { \mathrm { } } } , y ] = { \bf { I } } ^ { ( l ) } [ x , y ] - { \bf { I } } ^ { ( l ) } [ x , 0 ]$ , giving the error bound:

$$
\begin{array} { r } { \left| \Delta ^ { \mathrm { i o } } ( x , y , y ^ { \prime } ) \right| \leq s \cdot \| W _ { Q } \| _ { 2 } \cdot \| \delta \mathbf { I } ^ { ( l ) } [ x , y - y ^ { \prime } ] \| _ { 2 } \cdot \| \mathbf { K } _ { O } [ x , y ^ { \prime } , h ] \| _ { 2 } . } \end{array}\tag{34}
$$

Both the broadcast and Hadamard errors are therefore governed by the same root cause: the amplitude of I←O crossattention. At layer ${ \bf 0 } , \delta { \bf I } ^ { ( 0 ) } = { \bf 0 }$ , so both approximations are exact at initialization and diverge only gradually as crossstream attention builds up during training.

Furthermore, the Hadamard approximation introduces perpair score errors rather than direct context-vector errors. Within the prefix Softmax, individual score deviations are smoothed by the shared normalization: if the errors $\Delta ^ { \mathrm { i o } }$ have mixed signs across positions $y ^ { \prime } .$ , their net effect on the normalized context vector decays as $O ( 1 / \sqrt { y } )$ with prefix length. In practice, attention weights tend to be concentrated on a small number of positions, so errors at low-weight positions contribute negligibly to the final context.

Table III summarizes the complexity before and after approximation.

TABLE III  
COMPUTATIONAL COMPLEXITY PER LAYER BEFORE AND AFTER APPROXIMATION
<table><tr><td>Attention Type</td><td>Exact</td><td>Approximated</td></tr><tr><td>I→I self-attention</td><td> $O ( X ^ { 2 } Y )$ </td><td> $O ( X ^ { 2 } )$ </td></tr><tr><td>O→O self-attention</td><td> $O \dot { ( } X Y ^ { 2 } )$ </td><td>O(Y2)</td></tr><tr><td>I←O cross-attention</td><td> $O \dot { ( } X Y ^ { 2 } \dot { ) }$ </td><td>O(XY)</td></tr><tr><td> $\mathrm { O } {  } \mathrm { I }$  cross-attention</td><td> $O ( X ^ { 2 } Y )$ </td><td>O(XY)</td></tr><tr><td>Total</td><td> $O ( X ^ { 2 } Y + X Y ^ { 2 } )$ </td><td> $O ( X ^ { 2 } + Y ^ { 2 } + X Y )$ </td></tr></table>

For typical sentence lengths $X = Y = 3 0 .$ , this reduces the per-layer operation count by approximately $2 0 \times .$ , bringing duplex attention to the same order as standard single-stream autoregressive attention.

## V. TRAINING AND INFERENCE

The training objective comprises two interrelated tasks: (1) generating high-quality translations at the optimal EMIT moments; (2) learning an online EMIT/WAIT decision policy.

## A. Loss Heatmap Construction

Given an input sequence of length X and a target sequence of length Y, the model performs a full grid forward pass and computes the log-probability of each target token at every grid state:

$$
\mathcal { L } ( x , y ) = - \log p [ x , y ] _ { o _ { y + 1 } } = \mathrm { C E } ( p [ x , y ] , o _ { y + 1 } ) ,\tag{35}
$$

yielding a loss heatmap $\mathcal { L } \in \mathbb { R } ^ { X \times Y }$ , where a smaller value at $( x , y )$ indicates that the model can predict $o _ { y + 1 }$ more accurately given x input tokens and y output tokens.

The heatmap encodes all the information required to make optimal EMIT/WAIT decisions. For a fixed output position y, observing $ { \mathcal Ḋ L Ḍ } ( x , y )$ as x increases reveals how much each new input token helps:

• If $\mathcal { L } ( x , y ) ~ < ~ \mathcal { L } ( x ~ - ~ 1 , y )$ , the new input token $i _ { x }$ provides information useful for predicting $o _ { y + 1 }$ , so WAIT is beneficial at this state.

• If $\mathcal { L } ( x , y ) \approx \mathcal { L } ( x - 1 , y )$ , the new input contributes little to the current output prediction; WAIT only adds latency, making EMIT the better choice.

The optimal simultaneous translation strategy is therefore equivalent to finding a monotone path on the heatmap that minimizes the total EMIT-step loss while maximizing the area below the path (i.e., the latency reward).

## B. Optimal Monotone Path via Dynamic Programming

A monotone path π from (0, 0) to $( X , Y )$ consists of exactly X WAIT steps and $Y$ EMIT steps. We define the path score as a trade-off between translation quality and latency:

$$
R ( \pi ) = \lambda \cdot { \mathrm { A r e a } } ( \pi ) - \sum _ { ( x , y ) \underset { \in \pi } { \sum } } \mathcal { L } ( x , y ) ,\tag{36}
$$

where $\operatorname { A r e a } ( \pi )$ counts the grid cells below the path (rewarding early WAIT), and $\lambda > 0$ is the latency tolerance hyperparameter. Intuitively, λ sets the exchange rate between one unit of path area (one extra WAIT step at height y) and one unit of translation loss. A larger λ makes area more valuable relative to loss, pushing the optimal path to WAIT longer before emitting and accumulating more input context; translation quality may improve but latency increases. A smaller λ makes the path prefer early emission, reducing latency at the potential cost of translating with less context. In our experiments we fix $\lambda = 0 . 1$ , which we found to yield a good balance between quality and latency on the development set.

We find the optimal path via dynamic programming:

$$
\mathrm { d } \mathrm { p } [ x ] [ y ] = \operatorname* { m a x } \left\{ \mathrm { d } \mathrm { p } [ x - 1 ] [ y ] + \lambda y , \quad \begin{array} { l l } { x > 0 } \\ { \mathrm { d } \mathrm { p } [ x ] [ y - 1 ] - \mathcal { L } ( x , y - 1 ) , } & { y > 0 , \ x < X } \end{array} \right.\tag{37}
$$

with $\mathrm { d } \mathrm { p } [ 0 ] [ 0 ] = 0$ . The time and space complexity of the DP is $O ( X Y )$ , matching the grid forward pass.

After the DP table is filled, the optimal path is recovered by backtracking from $( X , Y )$ along parent pointers: if the optimal transition into $( x , y )$ was a WAIT step (from $( x { - } 1 , y ) )$ , we record the row-exit value $y _ { \mathrm { a t } } [ x { - } 1 ] = y$ and move to $( x { - } 1 , y ) ;$ ; if it was an EMIT step (from $( x , y - 1 ) )$ , we move to $( x , y - 1 )$ without updating $y _ { \mathrm { a t } }$ . This yields the row-exit sequence $y _ { \mathrm { a t } } [ 0 ] , \dots , y _ { \mathrm { a t } } [ X - 1 ]$ , where $y _ { \mathrm { { a t } } } [ x ]$ is the y-coordinate at which the path leaves row x (i.e., the number of EMIT steps performed before the (x+1)-th WAIT).

![](images/c556a04e5b7e3fe6775d8cb5a6f1bac03fe83f42f3b48f211495fbcb8646dd39.jpg)  
Fig. 4. Illustration of the loss heatmap and the optimal DP path. Color intensity indicates the grid loss $\mathcal { L } ( x , y ) ^ { \mathbf { \lambda } }$ (darker = higher loss). The black staircase line is the optimal monotone path: horizontal segments are WAIT steps and vertical segments are EMIT steps.

## C. EMIT Mask and Joint Training Loss

From the optimal path’s row-exit sequence $\{ y _ { \mathrm { a t } } [ x ] \}$ , we derive an EMIT mask $M \in \{ 0 , 1 \} ^ { X \times Y }$

$$
M [ x ] [ y ] = \mathbf { 1 } [ y < y _ { \mathrm { a t } } [ x ] ] .\tag{38}
$$

Geometrically, $M [ x ] [ y ] = 1$ means that at state $( x , y )$ the optimal path has already performed at least y EMIT steps while on row x, i.e., the path lies above position y on that row. The set of cells with $M [ x ] [ y ] = 1$ therefore forms exactly the region below the optimal path, so $\begin{array} { l } { { \sum _ { x , y } M [ x ] [ y ] } } \end{array} = \mathrm { A r e a } ( \pi ^ { * } ) - \mathrm { \Gamma }$ directly connecting the mask to the area reward in the path score.

The language model loss is computed only on EMIT-masked cells:

$$
\mathcal { L } _ { \mathrm { L M } } = \frac { 1 } { \vert M _ { + } \vert } \sum _ { ( x , y ) : M [ x ] [ y ] = 1 } \mathcal { L } ( x , y ) .\tag{39}
$$

A binary EMIT decision head is trained with cross-entropy against the mask:

$$
\mathcal { L } _ { \mathrm { e m i t } } = \frac { 1 } { | \mathcal { V } | } \sum _ { ( x , y ) \in \mathcal { V } } \mathrm { C E } ( e [ x ] [ y ] , \ M [ x ] [ y ] ) .\tag{40}
$$

The total training loss is:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { L M } } + \mathcal { L } _ { \mathrm { e m i t } } .\tag{41}
$$

The two losses have complementary roles: $\mathcal { L } _ { \mathrm { L M } }$ trains translation ability, ensuring that tokens emitted at the optimal moment are semantically accurate; $\mathcal { L } _ { \mathrm { e m i t } }$ trains the decision policy, teaching the EMIT head to reproduce the optimal path during inference. Together they form a self-guided online path optimization loop that requires no external alignment annotation:

1) Improved translation ability lowers grid losses $\mathcal { L } ( x , y )$ reshaping the loss heatmap.

2) A reshaped heatmap yields a better optimal path via DP, providing higher-quality supervision labels for the EMIT head.

3) A better-trained EMIT head makes inference-time decisions closer to the optimal path, supplying the language model with more appropriate input context.

4) More appropriate context in turn further improves translation quality, closing the loop.

Through this cycle, the model self-consistently learns the optimal latency-quality trade-off as translation competence and scheduling competence co-evolve throughout training.

## D. Incremental KV Cache Design

Naively re-running the full $X \times Y$ grid forward pass at each step would cost $O ( X Y L )$ per step. We design an incremental KV cache that reduces EMIT steps to $O ( X _ { \mathrm { v i s } } \cdot L )$ and WAIT steps to $O ( Y _ { \mathrm { g e n } } \cdot L )$

The cache stores, per layer, eight types of tensors: the keys and values for I→I and O→O (under broadcast approximation), their log-sum-exp statistics for joint Softmax, and the running prefix statistics $( m , Z , S )$ for the two cross-attention types (I←O and O←I). Total cache memory is approximately:

$$
\begin{array} { r l } & { \mathcal { M } _ { \mathrm { c a c h e } } = L \cdot [ 2 ( X + Y ) H _ { k } H _ { d } } \\ & { \quad + 6 ( X + Y ) H + 2 ( X + Y ) H H _ { d } ] \cdot 2  { \mathrm { b y t e s ~ } } ( \mathrm { f p } 1 6 ) , } \end{array}\tag{42}
$$

which is around 27 MB for L = 24, $X = Y = 1 2 8$ , H = 14, $H _ { k } = 2 , H _ { d } = 6 4 -$ far smaller than the model parameters.

An EMIT step computes only the new O-stream column y for all $x \in \{ 0 , \ldots , x _ { \mathrm { v i s } } - 1 \}$ : it reads $\mathbf { K } _ { O } ^ { x = 0 } , \mathbf { V } _ { O } ^ { x = 0 }$ from cache for $\mathrm { O } {  } \mathrm { O }$ , uses cached I-stream keys and prefix statistics for O←I, and updates the I←O prefix statistics with the new O-stream hidden state. A WAIT step is symmetric: it computes the new I-stream row $x _ { \mathrm { n e w } }$ and updates O←I prefix statistics.

## E. KV Cache Structure

The duplex inference cache maintains eight categories of tensors per layer. Table IV summarizes the cached items and their update rules.

TABLE IV  
PER-LAYER KV CACHE ENTRIES AND UPDATE RULES
<table><tr><td>Cache Entry</td><td>Purpose</td><td>Update Step</td></tr><tr><td> $\scriptstyle \mathbf { K } _ { r } ^ { y = 0 }$ </td><td>I→I broadcast keys</td><td>WAIT</td></tr><tr><td> $\mathbf { V } _ { \tau } ^ { y = 0 }$ </td><td>I→I broadcast values</td><td>WAIT</td></tr><tr><td> ${ \bf K } _ { \mathrm { O } } ^ { x = 0 }$ </td><td>O→O broadcast keys</td><td>EMIT</td></tr><tr><td> $\stackrel { - } { \mathbf { V } } _ { c } ^ { x } = 0$ </td><td>O→O broadcast values</td><td>EMIT</td></tr><tr><td> $m ^ { \mathrm { i i } } , Z ^ { \mathrm { i i } } , S ^ { \mathrm { i i } }$ </td><td>I→I joint Softmax statistics</td><td>WAIT</td></tr><tr><td> $m ^ { \mathrm { o o } } , Z ^ { \mathrm { o o } } , S ^ { \mathrm { o o } }$ </td><td>O→O joint Softmax statistics</td><td>EMIT</td></tr><tr><td> $m ^ { \mathrm { i o } } , Z ^ { \mathrm { i o } } , S ^ { \mathrm { i o } }$ </td><td>I←O prefix statistics</td><td>EMIT</td></tr><tr><td> $m ^ { \mathrm { o i } } , Z ^ { \mathrm { o i } } , S ^ { \mathrm { o i } }$ </td><td>O←I prefix statistics</td><td>WAIT</td></tr></table>

Overall cache memory is approximately:

$$
\begin{array} { r l } & { \mathcal { M } _ { \mathrm { c a c h e } } = L \cdot \left[ 2 ( X + Y ) H _ { k } H _ { d } \right. } \\ & { \left. ~ + 6 ( X + Y ) H + 2 ( X + Y ) H H _ { d } \right] \cdot 2  { \mathbf { b y t e s } } ~ ( \mathrm { f p } 1 6 ) . } \end{array}\tag{43}
$$

With $L = 2 4 , X = Y = 1 2 8 , H = 1 4 , H _ { k } = 2 , H _ { d } = 6 4 .$ the cache requires roughly 27 MB, which is modest relative to the full model footprint.

## F. Incremental Inference Algorithm

The full duplex inference loop alternates EMIT and WAIT steps, reusing cached keys, values, and prefix statistics to avoid repeated full-grid recomputation. Table V summarizes the main loop.

TABLE V  
PSEUDOCODE FOR THE DUPLEX INFERENCE MAIN LOOP  
```latex
Initialize cache with $i _ { 1 : 1 }$ and optional seed output $\scriptstyle O _ { 1 : } Y _ { \mathrm { s e e d } } .$ . Set $x _ { \mathrm { v i s } } = 1 ,$
$y _ { \mathrm { g e n } } = Y _ { \mathrm { s e e d } } .$
While not terminated:
Compute new O-stream column for current $y _ { \mathrm { g e n } }$ using cached I←O and
$\mathrm { O } {  } \mathrm { O }$ statistics.
Sample next token $o _ { y _ { \mathrm { g e n } } + 1 } .$
If EMIT head indicates WAIT and $x _ { \mathrm { v i s } } ~ < ~ X _ { \mathrm { t o t a l } }$ , read next input token
and update I-stream row cache.
Otherwise continue EMIT.
Return generated output sequence.
```

## G. RoPE Anchoring Strategy

A naive approach assigns output token at position y the RoPE index $x _ { \mathrm { v i s } } + y .$ . However, each WAIT step increments $x _ { \mathrm { v i s } } .$ invalidating all cached O-stream keys and requiring $O ( Y _ { \mathrm { g e n } } \cdot L )$ recomputation—defeating the purpose of caching.

We propose an anchored RoPE [45] strategy: fix the output position encoding to the total input length $X _ { \mathrm { t o t a l } }$ (known or estimated at inference start):

$$
\mathrm { p o s } _ { O } ( y ) = X _ { \mathrm { t o t a l } } + y .\tag{44}
$$

Since $X _ { \mathrm { t o t a l } }$ is constant throughout inference, all cached Ostream key vectors remain valid across WAIT steps. The resulting full-sequence inference complexity is $O ( ( X ^ { 2 } { + } X Y +$ $Y ^ { 2 } ) \cdot L )$ ), a reduction of min $( X , Y ) \times$ over the naive approach.

## VI. EXPERIMENTS

## A. Experimental Setup

1) Datasets: The experiments employ the Chinese-to-English (Zh-En) subset of the WMT 2021 [33] news translation task as training data. The dataset combines multiple sources including ParaCrawl [46], News-Commentary, Wiki-Titles, UN Parallel Corpus [47], WikiMatrix [48], and CCMT [49], totaling approximately 25 million bilingual sentence pairs.

Data cleaning operations include:

• Filtering numeric sequences: Samples containing meaningless Arabic numerals are removed.

![](images/a1ac306a3a8bf86eae1f9b3843d6320bc4e37dfd4b87866fac20ff2ce899b8dd.jpg)

EMIT step (add output column ynew)  
![](images/4357ccbf3e340dc8e4a078c57943f467d093688a58fd7b37226570b74b7ec375.jpg)

![](images/1a959666ad95b0eac1e32378046c3a9160ec50e78d8b0711bda6ab4dbbc761ac.jpg)  
Fig. 5. Incremental KV cache update. Left: EMIT step appends a new Ostream column and updates cross-stream statistics. Right: WAIT step appends a new I-stream row. Gray regions are cached; colored regions are newly computed.

• Normalizing special punctuation: MyBatis-style escaped punctuation (e.g., &apos;, &quot;) are mapped to standard punctuation.

• Space normalization: All spaces are removed from Chinese text; redundant spaces before punctuation are removed from English text.

After cleaning, approximately 16 million valid sentence pairs remain. To control GPU memory during training, data are uniformly sampled such that both source and target sequences have length $\leq 3 2$ . This yields 100K training examples and 1K validation examples.

2) Evaluation Metrics and Baselines:

• Translation Quality. We evaluate translation quality using BLEURT [32] and COMET [50]. BLEURT is a learned reference-based metric that fine-tunes a BERTlike encoder on human judgement data; it captures semantic equivalence and fluency beyond surface n-gram overlap. COMET (Crosslingual Optimized Metric for Evaluation of Translation) is a neural metric trained on direct assessment scores from professional translators, and has shown strong correlation with human judgements across language pairs in recent WMT shared tasks.

• Latency. We adopt three complementary latency metrics. Average Lagging (AL) [5] measures the average number of extra source tokens the system has consumed relative to an ideal simultaneous policy that reads and writes at the same rate; lower AL indicates tighter source–target synchronization. Average Proportion (AP) measures the average fraction of the source sentence consumed at the time each output token is emitted; $\mathrm { A P } { = } 1 . 0 0$ means the full input was read before any output was produced. First Response Latency (FRL) is the number of input tokens consumed before the first output token is generated, reflecting how quickly the system begins responding; for the grid model, this corresponds to the x-coordinate of the first EMIT state.

• Baselines. We compare against two systems. The nonstreaming baseline is the unmodified Qwen2.5-0.5B model prompted to translate after receiving the complete source input $( \mathrm { A P } = 1 . 0 0 ) $ ; it represents the quality upper bound with no latency constraint. The Wait-k baseline couples the same Qwen2.5-0.5B [51] backbone with the Wait-k policy, evaluated at $k \in \{ 3 , 5 \}$ ; it represents the standard fixed-policy simultaneous translation approach.

3) Training Details: Experiments are conducted on 4 NVIDIA RTX 3090 GPUs. The Qwen2.5-0.5B [51] backbone uses LoRA [52] fine-tuning; the text decoder head and EMIT decision head use full fine-tuning. Hyperparameters are shown in Table VI.

TABLE VI  
DUAL-STREAM MODEL TRAINING HYPERPARAMETERS
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Base model</td><td>Qwen2.5-0.5B</td></tr><tr><td>LoRA rank / scale</td><td>64  /  128</td></tr><tr><td>LoRA modules</td><td>q_proj, k_proj, v_proj, o_proj,</td></tr><tr><td>Text decoder</td><td>gate_proj, up_proj, down_proj Full fine-tuning</td></tr><tr><td>EMIT decision head</td><td>Full fine-tuning</td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Batch size per GPU</td><td>4</td></tr><tr><td>Precision</td><td>fp16</td></tr><tr><td>Optimizer</td><td>Adam</td></tr><tr><td>Warmup steps</td><td>3000</td></tr><tr><td>Training epochs</td><td>1</td></tr><tr><td>Path hyperparameter λ</td><td> $0 . 1$ </td></tr><tr><td>Hardware</td><td>4×NVIDIA RTX 3090</td></tr></table>

Training uses fp16 mixed precision with Adam [53] optimizer, initial learning rate $1 \times 1 0 ^ { - 4 }$ , and 3000 warmup steps. All linear projection layers in the backbone are adapted with LoRA (rank 64, scale 128); decoder and EMIT heads use full fine-tuning. The dynamic programming optimal path is recomputed online per minibatch; latency tolerance hyperparameter λ is fixed at 0.1.

## B. Experimental Results

1) Case Study: Clause Reordering: To qualitatively illustrate the proposed model’s translation and decision behavior, we analyze a Chinese-to-English translation pair containing a relative clause:

TABLE VII  
CHINESE-TO-ENGLISH TRANSLATION CASE STUDY
<table><tr><td>Source</td><td>站在那边的男人曾是棒球手。</td></tr><tr><td>Reference</td><td>The man standing there was a baseball player.</td></tr></table>

The core difficulty is word order inversion between Chinese and English relative clauses: the Chinese modifier (“standing there”) precedes the head noun (“man”), while English requires the head noun “The man” to appear first, followed by the modifier “standing there”. The model must wait for the head noun to appear before outputting in the correct English word order.

a) Grid Forward Propagation Visualization: Figure 6 visualizes the grid forward pass for the above source and reference as I-stream and O-stream inputs. Each cell contains the highest-probability predicted token; background color intensity indicates EMIT decision confidence (darker indicates stronger EMIT signal).

![](images/1700a347a577c7986e0c1f6517eb9b09adfa158cd3951403c4f83d0098311eda.jpg)  
Fig. 6. Grid forward propagation visualization. Horizontal axis: input sequence (left to right). Vertical axis: output sequence (bottom to top). Cell text: highest-probability token. Cell color: EMIT probability intensity.

Key observations:

1) Overall trend: EMIT probability exhibits a left-to-right, bottom-to-top gradation (lighter → darker). Longer input context and shorter pending output correlate with stronger EMIT confidence.

2) Clause region response: When reading input token 1 (“standing”), the predicted output is “standing” but EMIT probability is low. The model recognizes the semantic correspondence but refrains from output because English word order requires the head noun first.

TABLE VIII  
SIMULTANEOUS INFERENCE UNDER DIFFERENT EMIT THRESHOLDS
<table><tr><td>θ</td><td>站在</td><td>那边</td><td>的男人</td><td>曾</td><td>是</td><td>棒</td><td>球</td><td>手</td><td>0</td><td>&lt;eos&gt;</td></tr><tr><td>0.3</td><td>standing</td><td></td><td>there, the man</td><td>had</td><td>been</td><td>a</td><td>pitcher</td><td></td><td></td><td></td></tr><tr><td>0.4</td><td></td><td>the</td><td></td><td>man standing there had</td><td>been</td><td>a</td><td>pitcher</td><td></td><td></td><td></td></tr><tr><td>0.5</td><td></td><td></td><td>the</td><td>man standing there</td><td>was</td><td>a</td><td></td><td>baseball</td><td>player</td><td></td></tr><tr><td>0.6</td><td></td><td></td><td></td><td>the man standing there</td><td>was</td><td>a</td><td></td><td>baseball</td><td>player</td><td></td></tr><tr><td>0.7</td><td></td><td></td><td></td><td>the man standing there</td><td>was</td><td></td><td>a</td><td></td><td>basebali player</td><td></td></tr><tr><td>0.8</td><td></td><td></td><td></td><td>the</td><td>man standing there</td><td>was</td><td></td><td>a</td><td>baseball</td><td>player .</td></tr><tr><td>0.9</td><td></td><td></td><td></td><td></td><td>the</td><td>man standing there</td><td>was</td><td></td><td>a</td><td>baseball player .</td></tr></table>

3) Post-head-noun transition: Upon reading input token 3 (“the man”), the prediction shifts to “the” and EMIT probability increases significantly. The model correctly identifies that the head noun has appeared and begins output in proper English order.

This demonstrates that the model has internalized Chinese-English clause reordering at the attention level: “know the translation but hold back” during modifiers, then “switch target and commit” once the head noun appears.

b) Inference Process Under Different EMIT Thresholds: Table VIII shows simultaneous inference with varying EMIT thresholds θ (emission only when predicted EMIT probability exceeds θ).

Low threshold $( \theta ~ = ~ 0 . 3 ~ \sim ~ 0 . 4 ) \colon$ Premature emission causes word order and semantic errors. $\mathrm { A t } \ : \theta = 0 . 3 .$ , “standing” is output upon reading token 1, then “there, the man” at token 3. This reflects Chinese word order (modifier before head), deviating from English. Additionally, “baseball player” (tokens 6–8) is incorrectly translated to “pitcher” and “was” (tokens 4–5) to “had been”, showing that insufficient context harms both syntax and semantics.

Medium threshold $( \theta ~ = ~ 0 . 5 ~ \sim ~ 0 . 7 ) { : }$ Correct waiting followed by quality output. $\mathrm { A t   ~ } \theta \ : = \ : 0 . 5 ,$ , the model WAITs at tokens 1 and 2, then outputs “the” upon token 3, followed by “man standing there” at token 4. This achieves perfect clause reordering. $\theta = 0 . 6 \sim 0 . 7$ behave similarly, with output aligned to the reference: “was a baseball player”. Sufficient context ensures both grammatical and semantic correctness.

High threshold $( \theta = 0 . 8 \sim 0 . 9 ) \colon$ Correct translation but unnecessarily high latency. $\mathbf { A } \mathbf { t } \ { \boldsymbol { \theta } } = 0 . 8 .$ , output starts at token 4 $( { \mathrm { F R L } } = 4 ) ;$ ; at $\theta ~ = ~ 0 . 9 .$ , at token 5 (FRL = 5). Translation quality matches medium thresholds but First Response Latency increases from 3 to 5, introducing avoidable delay.

2) Quantitative Results: Table IX reports BLEURT, COMET, Average Lagging (AL), Average Proportion (AP), and First Response Latency (FRL) for the proposed model across five EMIT thresholds, the Wait-k baseline at $k \in \{ 3 , 5 \}$ and the baseline (Qwen2.5-0.5B translating after the full input is received), which serves as a quality reference upper bound.

Quality–latency trade-off. Within the proposed model, raising θ from 0.5 to 0.9 monotonically improves translation quality (BLEURT: 47.01 → 50.74; COMET: 59.59 → 64.73) at the cost of increased latency (AL: 0.23 → 4.39; FRL: $2 . 7 6  6 . 0 9 )$ . The emission threshold therefore provides a single, continuous knob for balancing quality against latency without retraining. Notably, at $\theta = 0 . 9$ the proposed model achieves $\mathrm { C O M E T } = 6 4 . 7 3 $ , surpassing the base model (63.85)

TABLE IX  
TRANSLATION QUALITY AND LATENCY ON THE ZH→EN TEST SET. THE BASELINE READS THE COMPLETE INPUT BEFORE TRANSLATING $( \mathrm { A P } = 1 . 0 0 ) $ . BOLD INDICATES THE BEST VALUE IN EACH COLUMN.
<table><tr><td>System</td><td>Param</td><td>BLEURT↑</td><td>COMET↑</td><td>AL↓</td><td>AP↓</td><td>FRL↓</td></tr><tr><td>Base model</td><td>一</td><td>54.03</td><td>63.85</td><td>9.39</td><td>1.00</td><td>18.06</td></tr><tr><td rowspan="2">Wait-k</td><td> $k = 3$ </td><td>43.20</td><td>52.06</td><td>4.98</td><td>0.79</td><td>3.00</td></tr><tr><td> $k = 5$ </td><td>45.20</td><td>54.93</td><td>5.99</td><td>0.84</td><td>4.96</td></tr><tr><td rowspan="5">Ours</td><td> $\theta = 0 . 5$ </td><td>47.01</td><td>59.59</td><td>0.23</td><td>0.49</td><td>2.76</td></tr><tr><td> $\theta = 0 . 6$ </td><td>48.86</td><td>62.42</td><td>1.14</td><td>0.53</td><td>3.31</td></tr><tr><td> $\theta = 0 . 7$ </td><td>49.10</td><td>62.79</td><td>1.61</td><td>0.55</td><td>3.79</td></tr><tr><td> $\theta = 0 . 8$ </td><td>49.75</td><td>63.54</td><td>2.87</td><td>0.59</td><td>4.65</td></tr><tr><td> $\theta = 0 . 9$ </td><td>50.74</td><td>64.73</td><td>4.39</td><td>0.65</td><td>6.09</td></tr></table>

while reducing FRL from 18.06 to 6.09—demonstrating that the dual-stream attention mechanism not only preserves but can exceed offline translation quality at a fraction of the latency. Figure 7 visualizes the AL–BLEURT Pareto frontier for all systems, showing that the proposed model dominates the Wait-k baseline across the full latency range.

![](images/60ef84e03e908f910aa580afc4a0d5c85f8dec5636abfd50a05142b1557c8856.jpg)  
Fig. 7. BLEURT–AL quality–latency trade-off on the zh→en test set.

Comparison with Wait-k. At comparable first-response latency, the proposed model consistently outperforms Waitk on both quality and latency efficiency. At FRL ≈ 3, our model at $\theta ~ = ~ 0 . 6 ~ \ : \ : ( \mathrm { F R L } = 3 . 3 1 )$ achieves BLEURT = 48.86 and COMET = 62.42, compared with BLEURT = 43.20 and COMET = 52.06 for Wait-k with $k = 3 \ \mathrm { ( F R L } = 3 . 0 0 ) \mathrm { - a }$ gain of +5.66 BLEURT and +10.36 COMET points. At FRL ≈ 5, our model at $\theta = 0 . 9 \ ( \mathrm { F R L } = 6 . 0 9 )$ similarly surpasses Waitk with k = 5 (FRL = 4.96) by +5.54 BLEURT and +9.80

COMET.

Furthermore, the proposed model exhibits substantially lower AL and AP across all operating points. Even at $\theta = 0 . 9 $ our AL (4.39) remains below that of Wait-k with $k \ = \ 3$ (4.98), and our AP (0.49–0.65) is markedly lower than the Wait-k range (0.79–0.84). This confirms that the learned EMIT/WAIT policy avoids the rigid read-ahead imposed by a fixed-k schedule, achieving tighter source–target synchronization while maintaining higher translation quality.

## VII. CONCLUSION

We presented a dual-stream simultaneous translation model that directly encodes the bidirectional dependency between the input and output streams inside the Transformer attention mechanism, rather than imposing a read-write schedule as post-processing. The key architectural contribution is a twodimensional grid hidden-state representation in which each cell $( x , y )$ maintains separate input-stream and output-stream hidden states conditioned on x visible source tokens and y generated target tokens. Within this grid, four types of attention—I→I, O→O, I←O, and O←I—capture all relevant intra- and cross-stream dependencies under the appropriate causal constraints, and are merged via a joint QK Softmax normalization that lets the model learn the relative importance of self-attention and cross-attention in a data-driven manner.

To make the grid computation tractable, we proposed two complementary approximations: a broadcast approximation that reduces intra-stream self-attention from $O ( X ^ { 2 } Y + X Y ^ { 2 } )$ to $O ( X ^ { 2 } + Y ^ { 2 } )$ , and a Hadamard approximation that reduces cross-attention from $O ( X Y ^ { 2 } + X ^ { 2 } Y )$ to O(XY). Both approximations are exact at initialization and are linguistically well-motivated—intra-stream attention patterns and cross-stream attention scores depend primarily on token content and relative position rather than on the precise count of tokens in the other stream.

For training, we constructed a per-grid-cell loss heatmap and found the optimal EMIT/WAIT path via dynamic programming, balancing translation quality against latency through a scalar hyperparameter λ. The resulting self-guided optimization loop requires no external alignment annotation: improved translation quality reshapes the heatmap, yielding better optimal paths that in turn provide higher-quality supervision for the EMIT decision head, whose improved inference-time decisions further benefit the language model. For inference, an incremental KV cache together with an anchored RoPE strategy reduces the per-step cost from O(XY L) to $O ( X _ { \mathrm { v i s } } L )$ for EMIT steps and ${ \cal O } ( Y _ { \mathrm { g e n } } L )$ for WAIT steps.

Experiments on Chinese-to-English simultaneous translation show that the proposed model substantially outperforms the Wait-k baseline at comparable first-response latency. At $\mathrm { F R L } \approx 3$ , our model gains +5.66 BLEURT and +10.36 COMET over Wait-k $( k = 3 )$ , while also achieving markedly lower Average Lagging and Average Proportion. At the highquality end $( \theta ~ = ~ 0 . 9 )$ , the model attains COMET = 64.73, exceeding even the non-streaming baseline (63.85) while reducing first-response latency from 18.06 to 6.09 tokens. These results demonstrate that explicit dual-stream attention is a principled and effective alternative to post-hoc read-write policies for simultaneous translation.

Future work includes scaling to larger backbone models and longer sequences, extending the grid framework to speech inputs and outputs, and exploring differentiable path optimization to allow end-to-end gradient flow through the DP step.

## REFERENCES

[1] M. Wang, T. Vu, J. Zhao, F. Shiri, E. Shareghi, and G. Haffari, “Simultaneous machine translation with large language models,” in Proceedings ofthe 22nd Annual Workshop ofthe Australasian Language Technology Association, 2024, pp. 89–103.

[2] B. Fu, M. Liao, K. Fan, C. Li, L. Zhang, Y. Chen, and X. Shi, “Llms can achieve high-quality simultaneous machine translation as efficiently as offline,” in Findings of the Association for Computational Linguistics: ACL 2025, 2025, pp. 20 372–20 395.

[3] M. Wang, T. Vu, Y. Wang, E. Shareghi, and G. Haffari, “Conversational simulmt: Efficient simultaneous translation with large language models,” in Proceedings of the 22nd International Conference on Spoken Language Translation (IWSLT 2025), 2025, pp. 93–105.

[4] S. Papi, M. Gaido, M. Negri, and M. Turchi, “Does simultaneous speech translation need simultaneous models?” in Findings of the Association for Computational Linguistics: EMNLP 2022, 2022, pp. 141–153.

[5] M. Elbayad, L. Besacier, and J. Verbeek, “Efficient wait-k models for simultaneous machine translation,” in Proc. Interspeech 2020, 2020, pp. 1461–1465.

[6] N. Arivazhagan, A. Bapna, O. Firat, D. Lepikhin, M. Johnson, M. Krikun, M. X. Chen, Y. Cao, G. Foster, C. Cherry et al., “Massively multilingual neural machine translation in the wild: Findings and challenges,” arXiv preprint arXiv:1907.05019, 2019.

[7] F. Dalvi, N. Durrani, H. Sajjad, and S. Vogel, “Incremental decoding and training methods for simultaneous translation in neural machine translation,” in Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), 2018, pp. 493–499.

[8] L. Zhao, K. Fan, W. Luo, W. Jing, S. Wang, Z. Zeng, and Z. Huang, “Adaptive policy with wait-k model for simultaneous translation,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 2023, pp. 4816–4832.

[9] H. Xia, T. Ge, P. Wang, S.-Q. Chen, F. Wei, and Z. Sui, “Speculative decoding: Exploiting speculative execution for accelerating seq2seq generation,” in Findings of the Association for Computational Linguistics: EMNLP 2023, 2023, pp. 3909–3925.

[10] S. Zhang and Y. Feng, “Universal simultaneous machine translation with mixture-of-experts wait-k policy,” in Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, 2021, pp. 7306–7317.

[11] Y. Ren, J. Liu, X. Tan, C. Zhang, T. Qin, Z. Zhao, and T.-Y. Liu, “SimulSpeech: End-to-end simultaneous speech to text translation,” in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 2020, pp. 3787–3796.

[12] N. Arivazhagan, C. Cherry, W. Macherey, and G. Foster, “Re-translation versus streaming for simultaneous translation,” in Proceedings of the 17th International Conference on Spoken Language Translation, 2020, pp. 220–227.

[13] H. Han, S. Ahn, Y. Choi, I. Chung, S. Kim, and K. Cho, “Monotonic simultaneous translation with chunk-wise reordering and refinement,” in Proceedings of the Sixth Conference on Machine Translation, 2021, pp. 1110–1123.

[14] J. Chen, R. Zheng, A. Kita, M. Ma, and L. Huang, “Improving simultaneous translation by incorporating pseudo-references with fewer reorderings,” in Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, 2021, pp. 5857–5864.

[15] L. Lin, S. Li, and X. Shi, “Leapt: Learning adaptive prefix-to-prefix translation for simultaneous machine translation,” in Proceedings of the 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2023), 2023.

[16] Y. Kano, K. Sudoh, and S. Nakamura, “Simultaneous neural machine translation with prefix alignment,” in Proceedings of the 19th International Conference on Spoken Language Translation (IWSLT 2022), 2022, pp. 22–31.

[17] M. Ma, L. Huang, H. Xiong, R. Zheng, K. Liu, B. Zheng, C. Zhang, Z. He, H. Liu, X. Li et al., “STACL: Simultaneous translation with implicit anticipation and controllable latency using prefix-to-prefix framework,” in Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 2019, pp. 3025–3036.

[18] X. Zhang, P. Shapiro, G. Kumar, P. McNamee, M. Carpuat, and K. Duh, “Curriculum learning for domain adaptation in neural machine translation,” in Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), 2019, pp. 1903–1915.

[19] S. Guo, S. Zhang, and Y. Feng, “Decoder-only streaming transformer for simultaneous translation,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024, pp. 8851–8864.

[20] S. Zhang, Y. Feng, and L. Li, “Future-guided incremental transformer for simultaneous translation,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, no. 16, 2021, pp. 14 428–14 436.

[21] Z. Ma, S. Zhang, S. Guo, C. Shao, M. Zhang, and Y. Feng, “Nonautoregressive streaming transformer for simultaneous translation,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 2023, pp. 5177–5190.

[22] B. Zheng, K. Liu, R. Zheng, M. Ma, H. Liu, and L. Huang, “Simultaneous translation policies: From fixed to adaptive,” in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 2020, pp. 2847–2853.

[23] B. Zheng, R. Zheng, M. Ma, and L. Huang, “Simpler and faster learning of adaptive policies for simultaneous translation,” in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 2019, pp. 1349–1354.

[24] R. Zhang, C. Zhang, Z. He, H. Wu, and H. Wang, “Learning adaptive segmentation policy for simultaneous translation,” in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020, pp. 2280–2289.

[25] D. Liu, M. Du, X. Li, Y. Li, and E. Chen, “Cross attention augmented transducer networks for simultaneous translation,” in Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, 2021, pp. 39–55.

[26] Y. Lee, J. Shin, and Y. Kim, “Simultaneous neural machine translation with a reinforced attention mechanism,” Etri Journal, vol. 43, no. 5, pp. 775–786, 2021.

[27] A. Xia, J. Lu, Y. Chen, Y. Zhou, and J. Zhang, “Dual-stream iterative semantic-aware mechanism for sign language translation,” in Proceedings of the International Conference on Neural Information Processing, 2024, pp. 319–333.

[28] Z. Bao, J. Wang, and Z. Yang, “Multi-path based self-adaptive crosslingual summarization,” in Proceedings of the International Conference on Knowledge Science, Engineering and Management, 2023, pp. 282– 294.

[29] S. Wang, J. Wu, K. Fan, W. Luo, J. Xiao, and Z. Huang, “Better simultaneous translation with monotonic knowledge distillation,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2023, pp. 2334–2349.

[30] E. Yang, D. Lawrie, J. Mayfield, D. W. Oard, and S. Miller, “Translatedistill: Learning cross-language dense retrieval by translation and distillation,” in Proceedings of the European Conference on Information Retrieval, 2024, pp. 50–65.

[31] Y. Wan, W. Zhang, Z. Li, H. Zhang, and Y. Li, “Dual knowledge distillation for neural machine translation,” Computer Speech & Language, vol. 84, p. 101583, 2024.

[32] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, “BLEU: a method for automatic evaluation of machine translation,” in Proceedings of the 40th annual meeting of the Association for Computational Linguistics, 2002, pp. 311–318.

[33] L. Specia, F. Blain, M. Fomicheva, C. Zerva, Z. Li, V. Chaudhary, and A. F. Martins, “Findings of the wmt 2021 shared task on quality estimation,” in Proceedings of the Sixth Conference on Machine Translation, 2021, pp. 684–725.

[34] C. Zerva, F. Blain, R. Rei, P. Lertvittayakumjorn, J. G. de Souza, S. Eger, D. Kanojia, D. Alves, C. Orasan, M. Fomicheva et al., “Findings of the wmt 2022 shared task on quality estimation,” in Proceedings of the Seventh Conference on Machine Translation (WMT), 2022, pp. 69–99.

[35] F. Blain, C. Zerva, R. Rei, N. M. Guerreiro, D. Kanojia, J. G. de Souza, B. Silva, T. Vaz, Y. Jingxuan, F. Azadi et al., “Findings of the wmt 2023 shared task on quality estimation,” in Proceedings of the Eighth Conference on Machine Translation, 2023, pp. 629–653.

[36] E. Ansari, A. Axelrod, N. Bach, O. Bojar, R. Cattoni, F. Dalvi, N. Durrani, M. Federico, C. Federmann, J. Gu et al., “Findings of the iwslt 2020 evaluation campaign,” in Proceedings of the 17th International Conference on Spoken Language Translation, 2020, pp. 1–34.

[37] M. Agarwal, S. Agrawal, A. Anastasopoulos, L. Bentivogli, O. Bojar, C. Borg, M. Carpuat, R. Cattoni, M. Cettolo, M. Chen et al., “Findings of the iwslt 2023 evaluation campaign,” in Proceedings of the 20th International Conference on Spoken Language Translation (IWSLT 2023), 2023, pp. 1–61.

[38] V. Agostinelli, T. Alumae, A. Anastasopoulos, L. Bentivogli, O. Bojar,¨ C. Borg, F. Bougares, R. Cattoni, M. Cettolo, L. Chen et al., “Findings of the iwslt 2025 evaluation campaign,” in Proceedings of the 22nd International Conference on Spoken Language Translation (IWSLT 2025), 2025, pp. 412–481.

[39] H. Wang, H. Wu, Z. He, L. Huang, and K. W. Church, “Progress in machine translation,” Engineering, vol. 18, pp. 143–153, 2022.

[40] S. Guo, X. Li, M. Liu, W. Chen, and Y. Feng, “StreamUni: Achieving streaming speech translation with a unified large speech-language model,” arXiv preprint arXiv:2507.07803, 2025.

[41] Q. Dong, Y. Zhu, M. Wang, and L. Li, “Learning when to translate for streaming speech,” in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2022, pp. 680–694.

[42] L. Dong and B. Xu, “CIF: Continuous integrate-and-fire for end-toend speech recognition,” in Proceedings of the 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2020), 2020, pp. 6079–6083.

[43] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,” in Proceedings ofthe Advances in Neural Information Processing Systems 30 (NeurIPS), Long Beach, CA, USA, 2017, pp. 5998–6008.

[44] T. Dao, D. Fu, S. Ermon, A. Rudra, and C. Re, “FlashAttention: Fast´ and memory-efficient exact attention with io-awareness,” Proceedings of the Advances in Neural Information Processing Systems, vol. 35, pp. 16 344–16 359, 2022.

[45] J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu, “Roformer: Enhanced transformer with rotary position embedding,” Neurocomputing, vol. 568, p. 127063, 2024.

[46] M. Ban˜on, P. Chen, B. Haddow, K. Heafield, H. Hoang, M. Espl´ a-Gomis,\` M. L. Forcada, A. Kamran, F. Kirefu, P. Koehn et al., “ParaCrawl: Webscale acquisition of parallel corpora,” in Proceedings of the 58th annual meeting ofthe associationfor computational linguistics, 2020, pp. 4555– 4567.

[47] M. Ziemski, M. Junczys-Dowmunt, and B. Pouliquen, “The United Nations parallel corpus v1.0,” in Proceedings of the tenth international conference on language resources and evaluation (LREC’16), 2016, pp. 3530–3534.

[48] H. Schwenk, V. Chaudhary, S. Sun, H. Gong, and F. Guzman, “Wiki-´ Matrix: Mining 135M parallel sentences in 1620 language pairs from Wikipedia,” in Proceedings of the 16th conference of the European Chapter ofthe Associationfor Computational Linguistics: Main volume, 2021, pp. 1351–1361.

[49] M. Yang, X. Hu, H. Xiong, J. Wang, Y. Jiaermuhamaiti, Z. He, W. Luo, and S. Huang, “CCMT 2019 machine translation evaluation report,” in Proceedings of the China Conference on Machine Translation, 2019, pp. 105–128.

[50] R. Rei, C. Stewart, A. C. Farinha, and A. Lavie, “COMET: A neural framework for mt evaluation,” in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020, pp. 2685–2702.

[51] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv et al., “Qwen3 technical report,” arXiv preprint arXiv:2505.09388, 2025.

[52] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-rank adaptation of large language models,” in Proceedings of the 10th International Conference on Learning Representations (ICLR), 2022.

[53] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in Proceedings of the International Conference on Learning Representations (ICLR), 2015.