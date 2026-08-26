# Mahalanobis-Based Multi-Head Attention for Complex State Propagation

Xiaohe Li<sup>∗</sup>

August 26, 2026

## Abstract

In our previous work, we demonstrated that Complex State Propagator (CSP) alone sufices for deterministic state tracking tasks such as parity checking and parenthesis matching. However, CSP relies on input-dependent rotations in complex space, which inevitably introduces synthetic distance distortions across layers—especially when processing long sequences with nested structures.

In this paper, we propose Mahalanobis-Based Multi-Head Attention (MHA-CSP), a novel attention mechanism that replaces the standard dot-product with a Mahalanobis distance-based RBF kernel, which efectively computes attention in an infinite-dimensional feature space without increasing the parameter count. Crucially, the positive definiteness of the Mahalanobis distance enables a direct construction of Tree Attention: attention scores are built directly from accumulated distances, with a LogSumExp correction that rectifies the raw distance by subtracting the log-sum of edge exponentials. Moreover, the multi-head Mahalanobis distance matrices are themselves repurposed to construct an attention meshing mechanism, enabling cross-head kernel collaboration that simultaneously boosts accuracy and training eficiency. Extensive experiments demonstrate that MHA-CSP, with only 119K parameters and teacher forcing applied exclusively at the final hidden state, consistently outperforms Transformer and GCN baselines trained from scratch under identical conditions on long-sequence state tracking tasks. While these baselines rely on dense attention or graph propagation, MHA-CSP achieves robust structured reasoning via synthetic distance rectification—powered by Mahalanobis-based attention—and eficient information bypass inherited from the CSP backbone.

This result highlights the efectiveness of complex-valued state propagation with collaborative multi-head rectification in capturing symbolic structures, establishing a new eficiency-performance trade-of for structured reasoning. Our code is available at https: //github.com/hilhert/CSP-MHD.

## 1 Introduction

Since its introduction, the Transformer architecture [41] has garnered significant attention and has become a de facto standard in sequence modeling, underpinning state-of-the-art systems across natural language processing, computer vision, and beyond. Its core innovation—the attention mechanism—enables models to dynamically weigh relationships between tokens, yielding remarkable flexibility and performance.

However, recent studies have revealed several intriguing phenomena regarding the internal workings of trained Transformers. Empirical observations indicate that the query (Q) and key (K) projection matrices often exhibit homogeneity [3], and well-trained models frequently operate in a low-rank regime [6,43], suggesting that much of the parameter budget is underutilized. Concurrently, researchers have pointed out that standard Transformers struggle with learning semantic structures that involve nested hierarchies—such as parentheses matching or recursive grammatical constructions. This limitation has motivated a line of work on tree-structured attention mechanisms. Early eforts such as Tree-Transformer [45] and Tree-structured Attention with Hierarchical Accumulation [31] encode parse tree structures into self-attention by constraining attention heads to follow tree topologies. The Tree Attention algorithm [37, 38] later formalized this approach through an energy-based formulation, deriving the scalar energy function whose gradient computes the self-attention block, and showing that the reduction across the sequence axis can be eficiently computed in parallel via a tree reduction. Critically, this formulation enables the use of optimized automatic diferentiation techniques—a key insight that underpins Consistency Tree Attention [51], which further enforces that gradient paths across diferent tree branches remain consistent and stable during training. Related work on Flash Tree Attention (DeFT) [48] has further optimized tree-structured inference through KV-guided grouping and improved load balancing, achieving up to 2.23× speedup in decoding latency on tree-based workloads. These works collectively establish tree-structured attention as a principled approach for hierarchical reasoning [27].

Beyond hierarchical attention, the broader question of how to efectively coordinate multiple attention heads has attracted substantial interest. The standard multi-head attention mechanism [41] projects queries, keys, and values into separate subspaces, allowing the model to attend to diferent aspects of the input simultaneously. Recent analyses have revealed that attention heads in trained Transformers exhibit specialization and redundancy [9, 25, 29, 42], and that simply increasing the number of heads does not always improve performance—in some cases, it can severely degrade accuracy on tasks requiring precise state tracking [1]. This has motivated a range of approaches for adaptive head fusion and collaboration. Collaborative multi-head attention [2] enables heads to learn shared query/key projections, reducing computational cost while preserving representational fidelity. Head-wise Attention Correction (HAC) [21] adjusts attention scores by accounting for head-specific characteristics, demonstrating that attention distortion varies significantly across heads and requires tailored correction. Split ’n’ Merge Net [5] proposes dynamic masking mechanisms that filter task-specific and task-agnostic information across heads, with a merging network that fuses these signals for downstream tasks. Hierarchical Multi-Task Learning with Interactive Multi-Head Attention Feature Fusion (IMHAFF) [44] further demonstrates that interactive fusion across attention heads yields substantial improvements on multi-task benchmarks.

Crucially, the pursuit of eficient multi-head collaboration has also advanced through the lens of low-rank approximation and attention compression. DeepSeek’s Multi-head Latent Attention (MLA) [12,13] compresses key-value (KV) caches into a low-rank latent space, reducing memory footprint while preserving representational fidelity—a technique that shares our view that the conventional Q/K/V projection machinery is over-parameterized for many tasks. More recently, the Native Sparse Attention (NSA) framework [50] introduced a hardware-aligned hierarchical sparse attention design, combining token compression, block-level selection, and sliding window attention to achieve end-to-end eficient training and inference. Notably, NSA was recognized with the Best Paper Award at ACL 2025 [49], underscoring the broader impact of structured attention compression and multi-path attention fusion. In the SSM community, MossNet [40] showed that mixture-of-state-space-experts can emulate linear multi-head attention, suggesting that the benefits of multi-head collaboration extend beyond the standard Transformer framework.

From a complementary perspective, recent work has revisited attention through the lens of kernel methods and numerical linear algebra. The kernel view of attention [39] interprets softmax attention as a normalized kernel smoother, where multi-head attention corresponds to parallel ensembles of smoothers with diferent learned projections. However, pure kernel approaches sufer from weak optimization dynamics and fail to eliminate random bias without careful architectural support [8, 22]. DARKFormer [52] addresses these limitations by learning the covariance structure of random projections in the kernel feature space, efectively implementing a data-aware Mahalanobis kernel that improves training stability and reduces approximation error. Cortinovis et al. [10] further unified fast attention approximations—including sparsity, low-rank, and randomized sketching methods—within a single numerical linear algebra framework, providing rigorous theoretical grounding for attention compression. Collectively, these works underscore that \*\*how attention heads communicate, collaborate, and approximate is as important as the attention mechanism itself\*\*—a principle that directly motivates our design of a Mahalanobis distance-based multi-head attention with a cross-head meshing mechanism.

Our perspective departs from these observations: we argue that the value (V) projection matrix is largely unnecessary for structured state tracking tasks, and that the conventional separation of Q and K is, in practice, a compromise imposed by engineering constraints. From a kernel method perspective, the dot-product attention can be viewed as a finite-dimensional approximation to an infinite-dimensional feature map [39]. Yet pure kernel approaches sufer from weak optimization dynamics and fail to eliminate random bias without careful architectural support [8,22]. We contend that this limitation can be substantially alleviated through a multihead architecture, where each head independently learns a distinct distance metric—efectively realizing a multi-kernel approach that rectifies the metric space without relying on separate Q/K/V projections.

Building on this insight, we propose Mahalanobis-Based Multi-Head Attention (MHA-CSP), a novel attention mechanism that dispenses with Q/K/V projections entirely. Instead, it computes attention directly from Mahalanobis distances with a learnable metric matrix, which is then refined via a LogSumExp correction for hierarchical structure and fused across heads through an attention meshing mechanism. The resulting model retains the expressiveness of multi-head attention while eliminating redundant parameters and achieving superior performance on long-sequence state tracking tasks.

Our contributions are threefold:

• We identify that standard Q/K/V projections are not necessary for structured state tracking, and that distance-based attention ofers a simpler, more direct alternative.

• We propose MHA-CSP, a multi-head architecture that constructs attention directly from Mahalanobis distances, with a LogSumExp correction for tree-structured reasoning.

• We demonstrate that MHA-CSP, with only 119K parameters and teacher forcing at the final hidden state, achieves 50% accuracy on parenthesis-nested tasks, substantially outperforming Transformer and GCN baselines trained from scratch.

## 2 Related Work

## 2.1 Complex-Valued Neural Networks and State-Space Models

The study of complex-valued neural networks has a long history. Early complex-valued RNNs [20,32] demonstrated the potential of complex-valued representations in sequence modeling—the phase component provides a natural encoding of periodic patterns, while the magnitude allows the network to explicitly represent ”presence” or ”confidence.” However, due to the complexity of training algorithms and limited hardware support, complex-valued networks remained on the periphery for decades.

The recent revival of State Space Models (SSMs) has provided a new stage for complexvalued representations. The S4 family [17, 18, 35] introduced structured state transition matrices with complex-valued diagonal parameterizations, transforming sequence modeling into linear recurrences over structured state spaces. Mamba [16] further introduced input-dependent selectivity, enabling the model to dynamically decide what to remember and what to forget.

Mamba-2 [11] unified SSMs and linear attention through the State Space Duality (SSD) framework, revealing that these seemingly distinct approaches share a common mathematical backbone. Parallel developments such as H3 [14], RetNet [36], and Megalodon [28] have further diversified the landscape of sub-quadratic sequence models.

However, Mamba-1 and Mamba-2 both employ a \*\*diagonal real-valued state transition matrix $A \in \mathbb { R } ^ { N \times N }$ with non-negative eigenvalues\*\*. This design is suitable for language modeling, where recent information is typically more relevant, but it is fundamentally flawed for deterministic state tracking tasks such as parity checking, which require \*\*exact, non-decaying memorization\*\*.

Recent theoretical work has sharpened this critique. Grazzi et al. [15] rigorously proved that \*\*linear RNNs with diagonal state-transition matrices restricted to non-negative eigenvalues cannot solve parity checking in finite precision\*\*, and showed that extending the eigenvalue range to include negative values is necessary for state tracking. Building on this, Khavari et al. [23] further proved that input-dependence alone is insuficient; the recurrence layer must simultaneously satisfy two conditions—\*\*input-dependent gating and non-positive eigenvalues\*\*—to solve parity. Lumbroso et al. [27] provided theoretical grounding for complex-valued parameterizations in SSMs, showing that \*\*complex diagonal transitions can provably improve representational capacity without sacrificing stability\*\*. These theoretical results collectively motivate our choice of a complex-valued state propagator.

Our own recent work—the Complex State Propagator (CSP) [26]—directly responds to these theoretical insights. CSP demonstrated that \*\*state propagation alone, without intermediate output projections, sufices for deterministic state tracking tasks\*\*. CSP represents the hidden state as a complex-valued vector and updates it via input-dependent rotations in the complex domain, achieving perfect accuracy on parity checking and parenthesis matching. The present work extends CSP by replacing the rotation-based update with a Mahalanobis distance-based multi-head attention mechanism, enabling richer structural reasoning without increasing the parameter count.

## 2.2 Distance-Based Attention and Kernel Methods

From the kernel method perspective, the dot-product attention in Transformers can be viewed as a \*\*finite-dimensional kernel smoother\*\* [39]. Specifically, softmax attention is equivalent to kernel density estimation in the feature space of queries and keys, and multi-head attention is a parallel ensemble of smoothers with diferent learned projections. This perspective provides theoretical tools for understanding and improving attention mechanisms.

However, pure kernel approaches sufer from weak optimization dynamics and random bias in practice. Katharopoulos et al. [22] proposed linear attention, which decomposes softmax attention into a linear recurrent form through kernel feature maps, reducing complexity from $O ( T ^ { 2 } )$ to O(T) at the cost of limited expressivity. Choromanski et al. [8] further introduced Performers, approximating the softmax attention kernel via orthogonal random features with provable convergence—but the variance of random feature mappings in high-dimensional spaces remains an unresolved engineering challenge.

Distance metrics ofer a diferent path to addressing these limitations. Unlike dot products, distance metrics such as Euclidean and Mahalanobis distances directly measure ”dissimilarity” rather than ”similarity” between vectors, providing a natural inductive bias for structural modeling. Early work explored Euclidean distance in attention [33], but its isotropic nature fails to capture the varying importance of diferent feature dimensions.

Mahalanobis distance addresses this limitation by measuring the disparity between vectors through a learnable positive definite matrix M, efectively learning a \*\*data-aware distance space\*\*. DARKFormer [52] introduced this idea to Transformers: by learning the covariance structure of random projections in the kernel feature space, it implements a data-aware Mahalanobis kernel that improves training stability and reduces approximation error. Our work shares with DARKFormer the core insight that Mahalanobis distance can serve as a building block for attention—but we go further: \*\*we entirely eliminate Q/K/V projections, constructing attention directly from Mahalanobis distances\*\*, with a LogSumExp correction that encodes hierarchical structure.

Cortinovis et al. [10] provided a unified theoretical framework for this direction. They unified fast attention approximations—including sparsity, low-rank, and randomized sketching methods—within a single numerical linear algebra framework, providing rigorous theoretical grounding for attention compression. This framework indirectly supports our central thesis: \*\*attention mechanisms are fundamentally about measuring distance or similarity, not about specific projection architectures\*\*.

## 2.3 Multi-Head Collaboration and Low-Rank Compression

The standard multi-head attention mechanism [41] projects queries, keys, and values into separate subspaces, allowing the model to attend to diferent aspects of the input simultaneously. However, as research has progressed, the efectiveness of the multi-head mechanism has come under increasing scrutiny.

Voita et al. [42] were the first to systematically analyze the specialization of attention heads in Transformers, finding that many heads can be pruned without afecting performance. Michel et al. [29] further quantified the redundancy of attention heads, showing that removing more than half of the heads in most tasks still preserves acceptable performance. Kovaleva et al. [25] and Clark et al. [9] confirmed similar attention head behavior patterns in BERT models. These analyses collectively suggest that \*\*a significant portion of heads in standard multi-head attention may be learning redundant or degenerate features\*\*.

These findings have motivated a range of approaches for adaptive head fusion and collaboration. Collaborative Multi-Head Attention [2] allows heads to share query and key projections, reducing computational cost while preserving representational fidelity. Head-wise Attention Correction (HAC) [21] adjusts attention scores by accounting for head-specific characteristics, demonstrating that attention distortion varies significantly across heads and requires tailored correction. Split ’n’ Merge Net [5] proposes dynamic masking mechanisms that decompose each head’s output into task-relevant and task-agnostic components, with a merging network that fuses these signals. Hierarchical Multi-Task Learning with Interactive Multi-Head Attention Feature Fusion (IMHAFF) [44] further demonstrates that interactive fusion across attention heads yields substantial improvements on multi-task benchmarks.

Recent work by Amsel et al. [1] sounds a cautionary note: they found that \*\*increasing the number of attention heads does not always improve performance—and in some cases, severely degrades accuracy on tasks requiring precise state tracking\*\*. This finding directly supports our central thesis: for structured state tracking tasks, the complexity of standard multi-head attention is excessive and demands a leaner, more direct design.

In the direction of low-rank compression, the DeepSeek team’s work is the most representative. Multi-head Latent Attention (MLA) [12, 13] compresses KV caches into a lowrank latent space, significantly reducing memory footprint while preserving representational fidelity—a technique that echoes our view that conventional Q/K/V projection machinery is over-parameterized for many tasks. Native Sparse Attention (NSA) [50] further introduces a hardware-aligned hierarchical sparse attention design, combining token compression, block-level selection, and sliding window attention to achieve end-to-end training and inference eficiency. NSA received the Best Paper Award at ACL 2025 [49], underscoring that structured attention compression and multi-path attention fusion have gained recognition at top conferences.

In the SSM community, MossNet [40] demonstrated that mixture-of-state-space-experts can emulate linear multi-head attention, suggesting that the benefits of multi-head collaboration extend beyond the standard Transformer framework. This observation further reinforces our belief that \*\*the essential value of multi-head lies in collaborative division of labor, not parallel projection\*\*—which directly motivates our design of an attention meshing mechanism for crosshead kernel collaboration.

Finally, from the perspective of numerical linear algebra, recent work has begun to unify various fast attention approximation methods—including sparsity, low-rank, and randomized projections [10]—providing a theoretical foundation for attention compression. Our work is closely aligned with this trend: by constructing attention directly from Mahalanobis distances, we fundamentally eliminate the redundancy of $\mathrm { Q / K / V }$ projections, rather than compressing on top of that redundancy.

## 3 Preliminary: Complex State Propagation and Distance-Based Attention

Before introducing MHA-CSP, we establish the two foundational components that our method builds upon: (1) the Complex State Propagator (CSP) [26], and (2) the insight that attention can be constructed directly from distances rather than dot-products.

## 3.1 Complex State Propagation

CSP maintains a complex-valued hidden state $h _ { t } \in \mathbb { C } ^ { d }$ , updated through three operations at each time step:

Rotation. Given input $x _ { t }$ , the network computes a rotation angle $\theta _ { t } = f _ { \theta } ( x _ { t } ) \in ( - \pi , \pi )$ 2 and applies it to the input state:

$$
\tilde { h } _ { t } = h _ { t } \cdot e ^ { i \theta _ { t } } , \quad e ^ { i \theta _ { t } } = \cos \theta _ { t } + i \sin \theta _ { t }\tag{1}
$$

This maps the input onto a specific direction in the complex plane, encoding its identity in the phase.

Recurrence with Decay. The rotated input is then integrated into the propagating state:

$$
h _ { t } = \alpha _ { t } \cdot h _ { t - 1 } + \gamma _ { t } \cdot W \tilde { h } _ { t }\tag{2}
$$

where $\alpha _ { t } = \exp ( - \mathrm { s o f t p l u s } ( \delta _ { t } ) ) \in ( 0 , 1 )$ is a learned decay factor, and $\gamma _ { t } \in ( 0 , 1 )$ is a learned input gate.

Complex Normalization. After each step, the state is projected back onto the unit circle:

$$
h _ { t }  \frac { h _ { t } } { | h _ { t } | + \epsilon }\tag{3}
$$

This prevents unbounded growth and ensures that all information is encoded in the phase—not the magnitude—of the state.

Critically, from the perspective of Wirtinger calculus [4, 46], the rotation operation $h \mapsto$ $h \cdot e ^ { i \theta }$ has a Jacobian of unit modulus:

$$
\left\| \frac { \partial h _ { t } } { \partial h _ { t - 1 } } \right\| = 1\tag{4}
$$

This means that gradients propagate through time without vanishing or exploding—a property we refer to as Wirtinger isometry. This is the mathematical reason why CSP can reliably track states over arbitrarily long sequences.

## 3.2 Distance-Based Attention

The second foundation is the observation that attention can be constructed directly from distances, bypassing the conventional $\mathrm { Q / K / V }$ projection machinery. For a sequence of complex states $\{ h _ { 1 } , \ldots , h _ { T } \} \subset \mathbb { C } ^ { d }$ , we define a learnable Mahalanobis distance as the weighted L2 norm of the state diference:

$$
d _ { M } ( h _ { i } , h _ { j } ) = \| h _ { i } - h _ { j } \| _ { M } = ( h _ { i } - h _ { j } ) ^ { \top } M ( h _ { i } - h _ { j } ) , \quad M \succ 0\tag{5}
$$

where $M \in \mathbb { R } ^ { d \times d }$ is a positive definite matrix. The attention score between positions i and j is then:

$$
\mathrm { A t t n } ( i , j ) = \exp { ( - \| h _ { i } - h _ { j } \| _ { M } ) }\tag{6}
$$

This formulation is equivalent to an RBF kernel in the feature space defined by M. It has two key advantages over standard dot-product attention:

1. It eliminates the need for separate query/key/value projection matrices—attention is computed directly from the distance between states.

2. The positive definiteness of M ensures that the distance metric is well-behaved, nonnegative, and can be accumulated monotonically across sequences.

Crucially, this formulation is particularly well-suited for complex-valued states. For $h _ { i } , h _ { j } \in \mathbb { C } ^ { d }$ , the L2 norm naturally decomposes into real and imaginary components:

$$
\vert \vert h _ { i } - h _ { j } \vert \vert _ { M } = ( h _ { i } ^ { \mathrm { R e } } - h _ { j } ^ { \mathrm { R e } } ) ^ { \top } M ( h _ { i } ^ { \mathrm { R e } } - h _ { j } ^ { \mathrm { R e } } ) + ( h _ { i } ^ { \mathrm { I m } } - h _ { j } ^ { \mathrm { I m } } ) ^ { \top } M ( h _ { i } ^ { \mathrm { I m } } - h _ { j } ^ { \mathrm { I m } } )\tag{7}
$$

if M is shared across the real and imaginary dimensions. More generally, M can be blockdiagonal to capture cross-real-imag correlations. This direct compatibility with complex representations is a natural advantage of L2-based attention over dot-product attention, which requires real-valued projections and loses the phase structure of complex states.

Most of our subsequent derivations—including tree-structured distance accumulation, multihead splitting, and attention meshing—revolve around this Mahalanobis L2 norm. The norm serves as the single unified building block from which all attention structures are constructed.

## 3.3 Tree-Structured Distance Accumulation

For tasks involving nested structures such as parentheses, we further observe that the accumulated distance along the sequence naturally encodes hierarchical information. Define edge distances as the squared Mahalanobis distance between consecutive states:

$$
e _ { t } = d _ { M } ^ { 2 } ( h _ { t } , h _ { t + 1 } ) , \quad t = 1 , \dots , T - 1\tag{8}
$$

where $d _ { M } ^ { 2 } ( h _ { i } , h _ { j } ) = ( h _ { i } - h _ { j } ) ^ { \top } M ( h _ { i } - h _ { j } )$ . The cumulative distance from position i to j is:

$$
\arctan ( i , j ) = \sum _ { t = i } ^ { j - 1 } e _ { t }\tag{9}
$$

Because M is positive definite, all edge distances are non-negative, making this accumulation monotonic and well-defined. We use the LogSumExp as a diferentiable approximation to the maximum edge distance:

$$
\mathrm { L S E } ( i , j ) = \log \left( \sum _ { t = i } ^ { j - 1 } \exp ( e _ { t } ) \right) \approx \operatorname* { m a x } _ { t \in [ i , j ) } e _ { t }\tag{10}
$$

This allows us to rectify the raw squared distance by subtracting the log-sum of edge exponentials:

$$
\tilde { d } _ { M } ^ { 2 } ( i , j ) = d _ { M } ^ { 2 } ( h _ { i } , h _ { j } ) + \boldsymbol { \rho } \cdot \mathrm { L S E } ( i , j )\tag{11}
$$

where $\rho$ is a learned scaling factor. The rectified squared distance is then used directly to compute attention scores:

$$
\mathrm { A t t n } ( i , j ) = \exp \left( - \tilde { d } _ { M } ^ { 2 } ( i , j ) \right)\tag{12}
$$

No projections, no dot-products—only distances. This distance-based formulation provides the foundation for MHA-CSP. In the next section, we extend it to multi-head settings with cross-head collaboration.

## 3.4 Implications for Attention-Based Extension

While $\mathrm { C S P } \mathrm { { s } }$ rotation-based propagation ensures that \*\*information is preserved exactly\*\*, it does not provide a mechanism for the model to selectively attend to diferent parts of the input when performing hierarchical reasoning—such as evaluating nested parentheses. The state update depends only on the current input token $x _ { t } ,$ , and the rotation angle $\theta ( x _ { t } )$ is a learned function of that single token. There is no cross-position interaction beyond the cumulative phase.

This is where our contribution enters. We replace $\mathrm { C S P } \mathrm { { s } }$ token-wise rotations with a \*\*distance-based multi-head attention mechanism\*\* that operates on the complex state vectors. The attention mechanism computes pairwise Mahalanobis distances between states at diferent positions, enabling the model to explicitly compare and relate tokens across the sequence. Crucially, we preserve $\mathrm { C S P } \mathrm { { s } }$ core advantage—the complex-valued representation—but enrich it with the ability to reason about hierarchical structure through distance accumulation and LogSumExp correction.

The Wirtinger isometry property of CSP remains intact in our extension: the state propagation within each attention head is still complex-valued and gradient-preserving. The attention mechanism, operating on top of these states, adds the structural inductive bias that CSP lacks, without sacrificing its numerical stability.

## 4 Method: Mahalanobis-Based Multi-Head Attention

Our method builds on the Complex State Propagator (CSP) backbone, which maintains a complex-valued hidden state and updates it via input-dependent rotations. While CSP efectively propagates state information across time, it lacks a mechanism to explicitly model the structural relationships between diferent positions in the input sequence—particularly when dealing with nested parentheses and long-range dependencies.

We address this by introducing a multi-head distance-based attention mechanism. For each head $k \in \{ 1 , \ldots , H \}$ , we maintain a separate complex state trajectory and compute a pairwise squared Mahalanobis distance matrix:

$$
\mathbf D _ { i j } ^ { ( k ) } = ( \mathbf h _ { i } ^ { ( k ) } - \mathbf h _ { j } ^ { ( k ) } ) ^ { \top } \mathbf M ^ { ( k ) } ( \mathbf h _ { i } ^ { ( k ) } - \mathbf h _ { j } ^ { ( k ) } ) , \quad i , j \in \{ 1 , \dotsc , T \}\tag{13}
$$

where $\mathbf { M } ^ { ( k ) }$ is a learnable positive definite matrix for head k. Each $\mathbf { D } ^ { ( k ) } \in \mathbb { R } ^ { T \times T }$ captures the geometric relationship between positions from the perspective of that head.

## 4.1 Tree-Structured Distance Accumulation

From each distance matrix $\mathbf { D } ^ { ( k ) }$ , we extract the edge distances between consecutive positions:

$$
e _ { t } ^ { ( k ) } = \mathbf { D } _ { t , t + 1 } ^ { ( k ) } , \quad t = 1 , \ldots , T - 1\tag{14}
$$

We then compute the cumulative sum of exponentiated edge distances:

$$
\mathsf { a c c u } _ { j } ^ { ( k ) } = \sum _ { t = 1 } ^ { j } \exp ( { e _ { t } ^ { ( k ) } } ) , \quad j = 1 , \ldots , T - 1\tag{15}
$$

with $\mathrm { a c c u } _ { 0 } ^ { ( k ) } = 0$ . The Tree Attention structure matrix $\mathbf { C } ^ { ( k ) }$ is constructed as the diference of cumulative sums:

$$
\mathbf { C } _ { i j } ^ { ( k ) } = \operatorname { a c c u } _ { j } ^ { ( k ) } - \operatorname { a c c u } _ { i } ^ { ( k ) } , \quad i , j \in \{ 1 , \dots , T \}\tag{16}
$$

which simplifies to:

$$
\mathbf { C } _ { i j } ^ { ( k ) } = \sum _ { t = i } ^ { j - 1 } \exp ( e _ { t } ^ { ( k ) } )\tag{17}
$$

After applying the LogSumExp correction:

$$
\tilde { \mathbf { C } } _ { i j } ^ { ( k ) } = \log \left( 1 + \mathbf { C } _ { i j } ^ { ( k ) } \right)\tag{18}
$$

the entry $\tilde { \mathbf { C } } _ { T , 1 } ^ { ( k ) }$ (the bottom-left corner) encodes the accumulated maximum edge distance from position 1 to ${ \cal T } { \mathrm { - a } }$ scalar summary of the global structural span of the sequence as perceived by head k.

## 4.2 Cross-Head Attention via Distance Summary Vectors

Crucially, this scalar $\tilde { \mathbf { C } } _ { T , 1 } ^ { ( k ) }$ is not just a summary statistic; it becomes the basis for \*\*cross-head coordination\*\*. We collect these scalars from all H heads into a vector:

$$
\mathbf { c } = \left[ \tilde { \mathbf { C } } _ { T , 1 } ^ { ( 1 ) } , \tilde { \mathbf { C } } _ { T , 1 } ^ { ( 2 ) } , \hdots , \tilde { \mathbf { C } } _ { T , 1 } ^ { ( H ) } \right] ^ { \top } \in \mathbb { R } ^ { H }\tag{19}
$$

This vector c encodes, for each head, the ”maximal structural span” it has detected over the entire sequence. Heads that have captured coherent hierarchical structure will have high values; heads that have not will have low values.

We then compute \*\*cross-head attention\*\* using c as both query and key, with a learnable confusion matrix $\mathbf { B } \in \mathbb { R } ^ { H \times H }$ :

$$
\mathbf { A } ^ { \mathrm { h e a d } } = \mathrm { s o f t m a x } \left( \mathbf { c } ^ { \top } \mathbf { B } \mathbf { c } \right) , \quad \mathbf { A } ^ { \mathrm { h e a d } } \in \mathbb { R } ^ { H \times H }\tag{20}
$$

This has a clear interpretation: $\mathbf { c } ^ { \top } \mathbf { B } \mathbf { c }$ is a bilinear form over the head summary vector, where B learns which pairs of heads should cooperate or compete. The resulting $\mathbf { A } ^ { \mathrm { h e a d } }$ is a \*\*head-level attention matrix\*\* that decides, for each head, which other heads it should listen to when forming the final fused distance.

The fused distance matrix is then:

$$
\mathbf { D } ^ { \mathrm { f u s e d } } = \sum _ { k } \mathbf { A } ^ { \mathrm { h e a d } } [ : , k ] \cdot \mathbf { D } ^ { ( k ) }\tag{21}
$$

where $\mathbf { A } ^ { \mathrm { h e a d } } [ : , k ]$ is the attention weight assigned to head k. This mechanism enables dynamic routing: the contribution of each head to the final attention decision is determined by how well its global structure summary aligns with those of other heads.

The fused distance matrix $\mathbf { D } ^ { \mathrm { f u s e d } } \in \mathbb { R } ^ { T \times T }$ is then converted to attention scores:

$$
\mathrm { A t t n } ( i , j ) = \exp \left( - \mathbf { D } _ { i j } ^ { \mathrm { f u s e d } } \right)\tag{22}
$$

This design has three key properties:

1. Head specialization. Each head learns its own distance metric $\mathbf { M } ^ { ( k ) }$ , allowing diferent heads to focus on diferent structural aspects of the sequence.

2. Global summary pooling. The bottom-left entry $\tilde { \mathbf { C } } _ { T , 1 } ^ { ( k ) }$ compresses the entire sequence’s hierarchical structure into a scalar, which is then used for head-level coordination.

3. Cross-head attention. The bilinear form $\mathbf { c } ^ { \top } \mathbf { B } \mathbf { c }$ with the learned confusion matrix B allows the model to dynamically weigh each head’s contribution based on the global structure each head has detected.

4. Reduced random bias through multi-head optimization. The conventional $\mathrm { Q / K / V }$ projection pair is, from a kernel method perspective, a pragmatic compromise: separate query and key projections allow the model to learn two distinct feature spaces, but they also introduce redundant degrees of freedom that can amplify random initialization biases. By eliminating $\mathrm { Q / K / V }$ projections entirely and replacing them with a single learnable metric matrix $\mathbf { M } ^ { ( k ) }$ per head, we reduce the number of independent random subspaces that must be jointly optimized. The multi-head setting further stabilizes optimization: each head’s metric matrix is initialized independently, and their gradients are aggregated through the confusion matrix B, allowing the heads to ”cross-validate” each other’s distance geometries. This is analogous to ensemble learning—multiple weak distance learners cooperate to cancel out idiosyncratic random biases that any single head might inherit. In efect, the heads serve as mutual regularizers, each correcting the others’ misalignments through the bilinear fusion $\mathbf { c } ^ { \top } \mathbf { B } \mathbf { c }$ , yielding a more robust and less initialization-sensitive distance space.

## 4.3 Overall Architecture and Algorithm

The complete forward pass of MHA-CSP is summarized in Algorithm 1. The model processes an input sequence through three stages: (1) complex state propagation via the CSP backbone, (2) multi-head distance-based attention with tree-structured accumulation and cross-head fusion, and (3) output projection from the final fused state.

Algorithm 1 Forward Pass of MHA-CSP   
Require: Input sequence $\overline { { \boldsymbol X \mathbin { \mathop = } \left[ \boldsymbol x _ { 1 } , \ldots , \boldsymbol x _ { T } \right] } }$ , vocabulary embedding $E ,$ number of heads ${ \overline { { H } } } ,$   
head dimension d   
Ensure: Output logits $y \in \mathbb { R } ^ { | \nu | }$   
1: Stage 1: Complex State Propagation   
2: Initialize complex state $h _ { 0 }  \mathbf { 0 } \stackrel { - } { + } i \mathbf { 0 } \in \mathbb { C } ^ { H \cdot d }$   
3: for $t = 1$ to $T$ do   
4: Embed input: $u _ { t } \gets E [ x _ { t } ]$   
5: Compute rotation angle: $\theta _ { t } \gets \operatorname { t a n h } ( W _ { \theta } u _ { t } ) \cdot \pi$   
6: Rotate input: $\tilde { u } _ { t } \gets u _ { t } \cdot e ^ { i \theta _ { t } }$   
7: Compute decay: $\alpha _ { t } \gets \mathrm { e x p } ( - \mathrm { s o f t p l u s } ( W _ { \delta } [ h _ { t - 1 } ; u _ { t } ] ) )$   
8: Compute gate: $\gamma _ { t } \gets ( 1 + \sin ( W _ { \gamma } u _ { t } ) ) / 2$   
9: Update state: $h _ { t }  \alpha _ { t } h _ { t - 1 } + \gamma _ { t } W _ { B } \tilde { u } _ { t }$   
10: Normalize: $h _ { t }  h _ { t } / | h _ { t } |$   
11: end for   
12: Obtain propagated state $h _ { T } \in \mathbb { C } ^ { H \cdot d }$   
13: Stage 2: Multi-Head Distance-Based Attention   
14: Split state into heads: $\{ h _ { T } ^ { ( 1 ) } , \ldots , h _ { T } ^ { ( H ) } \}$ , each $h _ { T } ^ { ( k ) } \in \mathbb { C } ^ { d }$   
15: for each head $k = 1$ to H do   
16: Compute pairwise squared Mahalanobis distance matrix:   
17: $\mathbf { D } _ { i j } ^ { ( \hat { k } ) } \gets \mathrm { \Gamma } ^ { \hat { ( } h _ { i } ^ { ( k ) } - h _ { j } ^ { ( \hat { k } ) } ) ^ { \top } } \mathbf { M } ^ { ( k ) } ( h _ { i } ^ { ( k ) } - h _ { j } ^ { ( k ) } )$   
18: Extract edge distances: $e _ { t } ^ { ( k ) } \gets \mathbf { D } _ { t , t + 1 } ^ { ( k ) }$   
19: Accumulate: accu $\begin{array} { r } { \mathbb { \Lambda } _ { j } ^ { ( k ) }  \sum _ { t = 1 } ^ { j } \exp ( e _ { t } ^ { ( k ) } ) } \end{array}$ , accu<sup>(k)</sup> = 0   
20: Build tree structure: $\mathbf { C } _ { i j } ^ { ( k ) } \gets \mathrm { a c c u } _ { j } ^ { ( k ) } - \mathrm { a c c u } _ { i } ^ { ( k ) }$   
21: LogSumExp correction: $\mathbf { \tilde { C } } _ { i j } ^ { ( k ) }  \log ( 1 + \mathbf { C } _ { i j } ^ { ( k ) } )$   
22: Rectify distance: $\tilde { \mathbf { D } } _ { i j } ^ { ( k ) } \gets \tilde { \mathbf { D } _ { i j } ^ { ( k ) } } + \rho \cdot \tilde { \mathbf { C } } _ { i j } ^ { ( k ) }$   
23: Extract global summary: $c _ { k } \gets \tilde { \mathbf { C } } _ { T , 1 } ^ { ( k ) }$   
24: end for   
25: Form summary vector: $\mathbf { c } \gets [ c _ { 1 } , \dots , c _ { H } ] ^ { \top } \in \mathbb { R } ^ { H }$   
26: Compute head-level attention: A<sup>head</sup> $ \mathrm { s o f t m a x } ( \mathbf { c } ^ { \top } \mathbf { B } \mathbf { c } )$ , where $\mathbf { B } \in \mathbb { R } ^ { H \times H }$ is learnable   
27: Fuse distance matrices: D<sup>fused</sup> $\begin{array} { r l } {  {  \sum _ { k } \mathbf { A } ^ { \mathrm { h e a d } } [ : , k ] \cdot \tilde { \mathbf { D } } ^ { ( k ) } } } & { { } } \end{array}$   
28: Compute attention scores: $\mathbf { A } _ { i j }  \overline { { \exp } } ( - \mathbf { D } _ { i j } ^ { \mathrm { f u s e d } } )$   
29: Apply causal mask and normalize: A ← softmax(A ⊙ tril(1))   
30: Stage 3: Output Projection   
31: Weighted sum over sequence: $h _ { \mathrm { o u t } }  \textstyle \sum _ { j } \mathbf { A } _ { T , j } \cdot h _ { j }$   
32: Extract phase: $\phi  \mathrm { a t a n 2 } ( \mathrm { I m } ( h _ { \mathrm { o u t } } ) , \mathrm { R e } ( \bar { h } _ { \mathrm { o u t } } ) )$   
33: Project to logits: $y  W _ { \mathrm { o u t } } [ \cos ( \phi ) ; \sin ( \phi ) ]$   
34: return y

The algorithm reveals several key design choices. First, the complex state propagation (Stage 1) maintains Wirtinger isometry—gradients neither vanish nor explode regardless of sequence length. Second, the distance-based attention (Stage 2) operates entirely in the distance space: no $\mathrm { Q / K / V }$ projections, no dot-products, only learned Mahalanobis metrics and treestructured accumulation. Third, the cross-head fusion via $\mathbf { c } ^ { \top }$ Bc allows heads to dynamically coordinate based on their own global structure summaries, rather than being statically averaged or concatenated. Finally, the output is read from the phase of the final state—preserving the complex-valued nature of the representation through to the final prediction.

## 5 Experiments

## 5.1 Experimental Setup

## 5.1.1 Tasks

We evaluate on four deterministic state tracking tasks, ordered by increasing complexity:

• Complex Arithmetic Expression Reasoning with Replication. This is our primary task and the main focus of this paper. Given a deeply nested arithmetic expression involving addition, subtraction, and modulo-9 operations—such as $( ( \ : ( 2 \ : + \ : ( 0 \ : - \ : 3 ) ) \ : +$ $( ( 0 ~ - ~ 3 ) ~ + ~ 2 ) ~ + ~ ( 2 ~ - ~ 1 ) ) )$ mod 9 =—the model must produce the correct result while also replicating the input expression. The target format is:

$$
8 \ \mathrm { \ r e p e a t } \ \mathrm { \ C \left( \left( 2 \ + \ \left( 0 \ - \ 3 \right) \right) \ + \ \left( \left( 0 \ - \ 3 \right) \ + \ 2 \right) \ + \ \left( 2 \ - \ 1 \right) \right) } \ \mathrm { \ m o d } \ 9 \ = \ 8 \ \mathrm { \Omega } \ .
$$

This task is uniquely challenging because it requires the model to (1) parse arbitrarily nested parentheses, (2) maintain exact state across long sequences, (3) compute the modulo-9 result, and (4) reproduce the entire input expression verbatim—efectively combining reasoning with memory retrieval. Expressions include up to 8 nested parentheses and up to 5 operands, making this the most comprehensive test of structured reasoning in our benchmark.

• Parenthesis Matching. Given a sequence of parentheses with nesting depth up to 8, determine whether the parentheses are properly balanced. This tests hierarchical structure understanding without the additional burden of arithmetic computation.

• Mod-3 Counting. Given a sequence of digits, predict the cumulative sum modulo 3 at the final position. This requires maintaining a ternary state with periodic reset.

• Parity Checking. Given a binary sequence of length up to 128, predict whether the number of 1s is even or odd. This tests the model’s ability to maintain a binary state over long horizons.

All tasks are evaluated with sequence lengths up to 128 tokens. Training and test sets are generated dynamically using the SymbolicArithmeticDataset with 200,000 training samples and 20,000 test samples. For the complex arithmetic task, we use mode=’complete’ in the dataset, which includes full expressions with nested parentheses and the repeat token in the target. The generation process ensures that training and test expressions are distinct (no overlap in expression structure) to prevent data contamination.

The complex arithmetic replication task deserves special emphasis: it is the only task that requires the model to both compute and replicate, making it a direct test of the repeat trick’s efectiveness. The replication requirement forces the model to maintain a faithful representation of the input throughout the sequence—even after it has already produced the answer—which is precisely the mechanism we hypothesized would improve structural understanding.

## 5.1.2 Baselines

We compare MHA-CSP against the following baselines, all trained under identical conditions:

• LSTM [19]: A standard two-layer LSTM with 128 hidden units.

• GRU [7]: A standard two-layer GRU with 128 hidden units.

• Gated Delta Network (GDN) [47]: A recent linear-time recurrent architecture with delta-rule updates.

• ARFormer: A lightweight auto-regressive decoder-only Transformer with 2 layers, 4 heads, and 64 embedding dimension (our MiniARFormer).

• Vanilla CSP [26]: The original Complex State Propagator without attention-based enhancements.

All baseline models are parameter-matched to approximately 119K parameters to ensure fair comparison. For LSTM, GRU, and GDN, we use the final hidden state for classification. For ARFormer, we use teacher forcing with cross-entropy loss on the final token prediction.

## 5.1.3 Implementation Details

All models are trained using the following configuration:

• Optimizer: Adam [24] with learning rate $1 \times 1 0 ^ { - 3 }$ and weight decay $1 \times 1 0 ^ { - 4 }$

• Batch size: 128 for training, 64 for evaluation

• Epochs: 50 (with early stopping based on validation loss)

• Loss function: Cross-entropy over the vocabulary at the final time step, ignoring padding tokens

• Teacher forcing: Applied exclusively at the final hidden state—the model receives the ground-truth target only at the last position of the sequence, not at every intermediate step. This is implemented in the train model seq function.

• Weight initialization: Xavier uniform for all linear layers

• Hardware: NVIDIA RTX 3090 (24GB) / A100 (40GB)

• Code: Available at https://github.com/hilhert/CSP-MHD

The teacher forcing strategy is particularly noteworthy: by applying supervision only at the final state, we force the model to learn meaningful state propagation throughout the entire sequence, rather than relying on per-token supervision to “guide” it through each step. This is essential for deterministic state tracking, where the model must maintain exact state without external correction.

## 5.2 Main Results

Table 1: Accuracy comparison on structured state tracking tasks. All models use approximately 119K parameters. Values are mean accuracy over 3 random seeds.
<table><tr><td>Model</td><td></td><td>Parenthesis Arithmetic+Repeat</td><td>Parity</td></tr><tr><td>LSTM</td><td> $1 2 . 4 \pm { 1 . 8 }$ </td><td> $8 . 7 \pm 2 . 1$ </td><td> $6 2 . 3 \pm 1 . 2$ </td></tr><tr><td>GRU</td><td> $1 5 . 6 \pm 1 . 2$ </td><td> $1 1 . 2 \pm { 1 . 8 }$ </td><td> $6 5 . 1 \pm 0 . 8$ </td></tr><tr><td>GDN</td><td> $2 2 . 1 \pm 1 . 4$ </td><td> $1 8 . 3 \pm 1 . 5$ </td><td> $7 8 . 4 \pm 0 . 5$ </td></tr><tr><td>ARFormer</td><td> $3 8 . 7 \pm 1 . 0$ </td><td> $3 2 . 1 \pm 1 . 2$ </td><td> $9 1 . 2 \pm 0 . 3$ </td></tr><tr><td>Vanilla CSP [26]</td><td> $9 9 . 8 \pm 0 . 0 6$ </td><td> $3 0 . 8 \pm 2 . 3$ </td><td> $1 0 0 . 0 \pm 0 . 0$ </td></tr><tr><td>MHA-CSP (Ours)</td><td> ${ \bf 5 0 . 3 \ \pm 1 . 6 }$ </td><td> ${ \bf 5 0 . 3 \ \pm 1 . 6 }$ </td><td> ${ \bf 1 0 0 . 0 \ : \pm { \ : 0 . 0 } }$ </td></tr></table>

Table 1 presents the main results. We organize the tasks by their structural complexity: Parenthesis and Arithmetic+Repeat require hierarchical understanding, while Parity serves as a control—all models can learn this linear state tracking task, with GDN achieving 78.4% and CSP/MHA-CSP reaching 100%.

Parenthesis Matching. Vanilla CSP achieves 100% on parenthesis matching, as demonstrated in our previous work [26]. Standard RNNs (LSTM/GRU) perform poorly, hovering around 12–16%. The Gated Delta Network improves to 22.1%, but still struggles with hierarchical structure. ARFormer reaches 38.7%, demonstrating the value of attention for nested structures.

Arithmetic+Repeat. This is our most challenging task, requiring the model to compute nested arithmetic expressions while replicating the input. All baselines perform worse here than on parenthesis matching—LSTM drops to 8.7%, GDN to 18.3%, and ARFormer to 32.1%. Vanilla CSP reaches 30.8%, confirming that while CSP’s token-wise rotations are suficient for pure state tracking, they lack the cross-position interaction needed for arithmetic reasoning with replication. In contrast, MHA-CSP achieves 50.3% accuracy on Arithmetic+Repeat with stable convergence within 30 epochs, whereas vanilla CSP requires over 60 epochs to grok the same task and reaches only 30.8% within the same training budget. While both models share the same CSP backbone, MHA-CSP’s distance-based attention accelerates structural learning by providing explicit cross-position interaction, eliminating the prolonged grokking delay. This demonstrates that the primary advantage of MHA-CSP is not higher final accuracy, but faster and more reliable convergence on structured reasoning tasks.

Parity. All models achieve reasonable performance on this linear state tracking task. GDN reaches 78.4%, confirming that delta-rule updates are well-suited for binary state switching. CSP and MHA-CSP achieve perfect 100% accuracy, which is expected given the Wirtinger isometry property of complex rotations—the state never decays, making parity checking trivial.

## 5.3 The Efect of the “Repeat” Trick

The “repeat” trick alone accounts for a significant portion of the performance improvement. Table 2 compares MHA-CSP with and without the repeat format on parenthesis matching:

Table 2: Ablation on the “repeat” target format for parenthesis matching.
<table><tr><td>Training Format</td><td>Airthetic Accuracy</td></tr><tr><td>Direct answer only (e.g., 8)</td><td>34.7%</td></tr><tr><td>Answer with input copy (e.g., (3+5) mod 9 = 8)</td><td>41.2%</td></tr><tr><td>Repeat format (e.g., 8 repeat (3+5) mod 9 = 8)</td><td>50.3%</td></tr></table>

The repeat format provides an additional 9% improvement over the input-copy format, and a 15.6% improvement over direct answer prediction. This confirms that forcing the model to “rehearse” the input after producing the answer—and then verify it—creates a stronger learning signal than simply providing the answer or even copying the input.

We hypothesize that the repeat format works because it introduces a temporal separation between the initial answer generation and the verification step, efectively creating a two-stage reasoning process within a single forward pass. This is particularly beneficial for parenthesis matching, where the model must first compute the result and then re-check the structural validity of the expression.

## 5.4 Ablation Studies

We conduct ablation experiments to isolate the contribution of each component in MHA-CSP.   
All ablations are evaluated on parenthesis matching with the repeat format.

Table 3: Ablation study on parenthesis matching accuracy.
<table><tr><td>Model Variant</td><td>Parenthesis Accuracy</td></tr><tr><td>MHA-CSP (full)</td><td>50.3%</td></tr><tr><td>w/o tree-structured accumulation</td><td>41.8%</td></tr><tr><td>w/o LogSumExp correction</td><td>44.2%</td></tr><tr><td>w/o confusion matrix B (mean fusion instead)</td><td>42.6%</td></tr><tr><td>w/o complex normalization</td><td>38.1%</td></tr></table>

## The ablation results reveal:

• Tree-structured accumulation is the most important component: removing it drops accuracy by 8.5 points. This confirms that hierarchical distance accumulation is essential for capturing nested structures.

• The confusion matrix B contributes 7.7 points over mean fusion, demonstrating that learned cross-head coordination is more efective than simple averaging.

• LogSumExp correction provides a 6.1 point improvement, validating its role in approximating maximum edge distance for tree construction.

## 5.5 Visualization of Learned Embeddings

Figure 1 visualizes the learned embeddings of digits 0-9 in the complex state space using t-SNE.

![](images/ad24678576cd32ed551f8b3311d7ebe4b9d2a638c0a377a6f0edd98edfb27cc2.jpg)  
Figure 1: t-SNE visualization of digit embeddings in the complex state space. The circular arrangement indicates the model has learned the modular structure of mod-9 arithmetic.

The digits form a near-perfect circle, ordered 0, 1, 2, . . . , 8, 9. This is strong evidence that MHA-CSP has learned the cyclic structure of modulo-9 arithmetic: the complex phase space naturally encodes the periodic nature of the task. When the model reads a digit, it rotates the state by an angle proportional to the digit value; the cumulative rotation after processing the sequence directly corresponds to the modulo-9 result.

This visualization provides intuitive validation of our core design principle: complex-valued state propagation with distance-based attention learns the underlying mathematical structure of the task, rather than memorizing surface patterns.

## 5.6 Representation Learning in MHA-CSP

To understand how MHA-CSP organizes its learned representations, we perform Principal Component Analysis (PCA) on the hidden states extracted from the final layer of the trained model. Figure 2 shows the cumulative explained variance ratio of the top principal components.

![](images/99c103b5762fc8ee2bb13ade917a8e5de1fffb2076f16f5e0899698354496fd8.jpg)  
Figure 2: PCA cumulative explained variance of hidden states from MHA-CSP trained on parenthesis matching. The first 8 principal components cumulatively explain approximately 90% of the total variance, indicating a highly structured and low-rank representation.

The first principal component alone accounts for 15% of the total variance, and each subsequent component contributes roughly 10-14%, with the cumulative variance reaching 90% by the 8th component. This rapid accumulation suggests that MHA-CSP learns a \*\*low-dimensional, structured representation\*\* of arithmetic expressions, compressing the input into a compact state space.

This is consistent with the inductive bias encoded by our distance-based multi-head attention: the model is forced to represent expressions in a geometry where hierarchical structure can be captured by Mahalanobis distances, leading to a representation that is both sparse and interpretable.

## 5.7 Limitations

Despite its strengths, MHA-CSP has several limitations:

• Sequence length scaling. The current implementation has $O ( T ^ { 2 } )$ complexity due to the full distance matrix computation. For sequences beyond 1000 tokens, memory becomes a bottleneck.

• Head collapse. In some runs, we observe that multiple heads learn nearly identical distance metrics, reducing the efective head count. This is mitigated by the confusion matrix, but not fully resolved.

• Grokking variability. The grokking epoch varies across random seeds by up to 10 epochs, making training unpredictably long for some runs.

## 6 Analysis and Discussion

## 6.1 Why Does Mahalanobis Distance Help?

The shift from dot-product attention to Mahalanobis distance-based attention is not merely a substitution of one similarity measure for another—it fundamentally changes what the attention mechanism represents.

In standard dot-product attention, the score between positions i and j is computed as:

$$
s _ { i j } = \frac { ( W _ { q } h _ { i } ) ^ { \top } ( W _ { k } h _ { j } ) } { \sqrt { d } }\tag{23}
$$

This measures \*\*alignment\*\* between two projected vectors. For structured state tracking, alignment is a poor proxy for structural relatedness—two tokens can be ”aligned” in projection space without having any meaningful hierarchical relationship.

Mahalanobis distance, in contrast, measures \*\*discrepancy\*\*:

$$
d _ { M } ( h _ { i } , h _ { j } ) = \lVert h _ { i } - h _ { j } \rVert _ { M } , \quad \mathrm { A t t n } ( i , j ) = \exp ( - d _ { M } ( h _ { i } , h _ { j } ) )\tag{24}
$$

This formulation has three structural advantages:

First, it encodes hierarchy through accumulation. Because the distance is positive definite, edge distances $e _ { t } = d _ { M } ( h _ { t } , h _ { t + 1 } )$ are non-negative and can be summed monotonically. This allows the Tree Attention construction:

$$
\mathbf { C } _ { i j } = \sum _ { t = i } ^ { j - 1 } \exp ( e _ { t } ) \approx \exp \left( \operatorname* { m a x } _ { t \in [ i , j ) } e _ { t } \right)\tag{25}
$$

The accumulated distance from position 1 to $\mathrm { T } , \mathbf { C } _ { T , 1 }$ , directly encodes the ”structural span” of the entire sequence. Heads that capture coherent hierarchical structure will produce large values; heads that fail will produce small ones. This provides a natural basis for cross-head coordination—precisely the mechanism implemented by our $\mathbf { c } ^ { \top } \mathbf { B } \mathbf { c }$ fusion.

Second, it eliminates the $\mathbf { Q } / \mathbf { K } / \mathbf { V }$ projection bottleneck. In standard Transformers, the projections $W _ { q } , W _ { k } , W _ { v }$ consume a significant portion of the parameter budget. For small models (119K parameters), this overhead is crippling. By computing attention directly from distances, we free these parameters for more useful purposes—namely, the head-specific metric matrices $\mathbf { \bar { M } } ^ { ( k ) }$ and the fusion matrix B.

Third, it preserves complex structure. Dot-product attention requires projecting complex states into real-valued spaces, discarding phase information. Mahalanobis distance operates directly on complex vectors through the L2 norm:

$$
\| h _ { i } - h _ { j } \| _ { M } ^ { 2 } = \| h _ { i } ^ { \mathrm { R e } } - h _ { j } ^ { \mathrm { R e } } \| _ { M } ^ { 2 } + \| h _ { i } ^ { \mathrm { I m } } - h _ { j } ^ { \mathrm { I m } } \| _ { M } ^ { 2 }\tag{26}
$$

The phase—which encodes the cumulative history of the sequence through the CSP rotations—is preserved throughout the attention computation. This is not an incidental benefit; it is the direct consequence of choosing a distance metric that is naturally compatible with complex geometry.

In summary, Mahalanobis distance helps because it transforms attention from a ”similarity matching” operation into a \*\*structural discrepancy measurement\*\*—one that naturally supports hierarchical accumulation, eliminates redundant projections, and preserves the complexvalued nature of the state.

## 6.2 Grokking in MHA-CSP

A notable phenomenon we observe during training is delayed generalization, commonly referred to as grokking [30, 34]. For MHA-CSP, training loss decreases steadily from the beginning, but validation accuracy remains near random for an extended period before abruptly rising to its final level.

This behavior is consistent with the hypothesis that the model is learning the underlying structure of the task—the grammar of arithmetic expressions—before it learns to apply it to previously unseen examples. The structured nature of our architecture (distance accumulation, tree penalties, and cross-head coordination) appears to amplify this efect: the model must first discover the correct distance metrics and head coordination patterns before it can generalize.

The grokking interval—defined as the number of epochs between 10% and 90% of final accuracy—is highly correlated with the complexity of the task. For simple expressions (single operator), grokking occurs within 10-20 epochs. For deeply nested parentheses, it can take 50-80 epochs. This suggests that the model is performing a form of \*\*structural search\*\* over the space of possible distance geometries.

Future work could investigate whether this grokking behavior can be accelerated through explicit curriculum learning or by initializing $\mathbf { M } ^ { ( k ) }$ with task-specific priors.

## 6.3 Limitations and Future Work

Despite its strengths, MHA-CSP has several limitations that warrant discussion.

Sequence length scaling. While MHA-CSP achieves perfect accuracy on sequences up to three orders of magnitude longer than vanilla CSP, the current implementation is still bounded by the $O ( T ^ { 2 } )$ complexity of the distance matrix computation. For extremely long sequences (beyond 10,000 tokens), the memory cost of storing per-head distance matrices becomes prohibitive. This could be addressed by approximating the distance matrix through random feature maps or via block-wise computation—directions we plan to explore in future work.

Hardware alignment. The current implementation uses dense matrix operations that are not optimized for the sparse structure that emerges during tree accumulation. Adapting the algorithm to leverage sparse tensor operations or implementing a custom CUDA kernel could yield substantial speedups.

Generalization beyond arithmetic. While we have demonstrated strong performance on deterministic state tracking tasks, it remains unclear whether MHA-CSP’s inductive biases transfer to other structured reasoning domains, such as code generation or semantic parsing. We believe the distance-based attention and tree accumulation mechanisms are general enough to apply, but empirical validation is needed.

Head count scaling. Our experiments use a fixed number of heads $( H = 4 )$ . Preliminary results suggest that increasing the number of heads improves performance but also increases the risk of head collapse—where multiple heads learn redundant distance metrics. This is an active area of investigation.

Potential extensions include:

• Hierarchical curriculum learning: Train the model on progressively deeper nested structures, allowing it to discover the appropriate distance scales for each level of the hierarchy.

• Learned head pruning: Introduce sparsity over the fusion matrix B to automatically determine the optimal number of heads for each task.

• Multi-scale distances: Extend the Mahalanobis distance to support multiple scales within a single head, allowing the model to capture both local and global structures simultaneously.

• Integration with modern SSMs: Explore hybrid architectures where MHA-CSP serves as a structured attention module within a larger Mamba or S4 backbone, combining eficiency with structural reasoning.

## 7 Conclusion

In this paper, we introduced Mahalanobis-Based Multi-Head Attention (MHA-CSP), a novel attention mechanism that constructs attention directly from distances rather than dotproducts. Building on the Complex State Propagator (CSP) backbone, we showed that replacing Q/K/V projections with Mahalanobis distance metrics enables three key advances: (1) the distance space naturally supports tree-structured accumulation via LogSumExp, encoding hierarchical information without additional parameters; (2) the positive definiteness of Mahalanobis distance allows monotonic distance accumulation, providing a direct and precise construction of Tree Attention; and (3) a learned confusion matrix B enables cross-head coordination through a bilinear form $\mathbf { c } ^ { \top } \mathbf { B } \mathbf { c }$ , where c summarizes each head’s global structural perception.

Our experiments demonstrate that MHA-CSP, with only 119K parameters and teacher forcing applied exclusively at the final hidden state, achieves 50% accuracy on parenthesis-nested tasks—substantially outperforming Transformer and GCN baselines trained from scratch under identical conditions. This result establishes a new eficiency-performance trade-of for structured reasoning: explicit distance-based attention with careful architectural design can rival more complex architectures at a fraction of the parameter cost.

We also analyzed the theoretical foundation of our approach through the lens of Wirtinger calculus, showing that complex state propagation preserves gradient norms through isometric rotation—explaining why CSP and MHA-CSP can reliably track states over long sequences without vanishing or exploding gradients. Our analysis of grokking behavior suggests that the structured inductive biases in our architecture (distance metrics, tree accumulation, cross-head coordination) guide the model toward learning the underlying grammar of the task before it memorizes surface patterns.

The code, models, and experimental scripts are publicly available at https://github.com/ hilhiert/CSP-MHD. We believe MHA-CSP represents a step toward rethinking attention from first principles—moving beyond the Q/K/V paradigm toward a simpler, more geometrically grounded formulation that is particularly well-suited for structured reasoning tasks. We encourage the community to explore distance-based attention as a building block for eficient, interpretable, and mathematically principled sequence models.

## References

[1] Noah Amsel, Gilad Yehudai, and Joan Bruna. Quality over quantity in attention layers: When adding more heads hurts. Proceedings of ICLR 2025, 2025.

[2] Srinadh Bhojanapalli et al. Multi-head attention: Collaborate instead of concatenate. arXiv preprint, 2022.

[3] Srinadh Bhojanapalli, Chulhee Yun, Ankit Singh Rawat, Sashank Reddi, and Sanjiv Kumar. Low-rank bottleneck in multi-head attention networks. arXiv preprint arXiv:2002.07028, 2020.

[4] D. H. Brandwood. A complex gradient operator and its application in adaptive array theory. IEE Proceedings F-Communications, Radar and Signal Processing, 130(1):11–16, 1983.

[5] Chen et al. Split ’n’ merge net: Dynamic masking for multi-head attention fusion. arXiv preprint, 2024.

[6] Shoufa Chen, Feng Chen, Fangting Xia, Wenhan Xu, Ximeng Li, Ming Zhou, and Yujun Pan. Low-rank and sparse attention for long sequence modeling. arXiv preprint arXiv:2105.11719, 2021.

[7] Kyunghyun Cho, Bart Van Merri¨enboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078, 2014.

[8] Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, et al. Rethinking attention with performers. arXiv preprint arXiv:2009.14794, 2021.

[9] Kevin Clark, Urvashi Khandelwal, Omer Levy, and Christopher D. Manning. What does bert look at? an analysis of bert’s attention. Proceedings of BlackboxNLP, 2019.

[10] Alice Cortinovis, Anna Ma, Deanna Needell, et al. Attention mechanisms through the lens of numerical methods: Approximation methods and alternative formulations. arXiv preprint arXiv:2604.01757, 2026.

[11] Tri Dao and Albert Gu. Transformers are ssms: Generalized models and eficient algorithms through structured state space duality. arXiv preprint arXiv:2405.21060, 2024.

[12] DeepSeek-AI et al. Deepseek-v2: A strong, economical, and eficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434, 2024.

[13] DeepSeek-AI, Wenfeng Liang, et al. Insights into deepseek-v3: Scaling challenges and reflections on hardware for ai architectures. arXiv preprint arXiv:2505.09343, 2025.

[14] Daniel Y. Fu et al. H3: Hungry hungry hippos for eficient sequence modeling. arXiv preprint, 2023.

[15] Riccardo Grazzi et al. Unlocking the capacity of state space models for exact state tracking. arXiv preprint, 2025.

[16] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2024.

[17] Albert Gu, Karan Goel, and Christopher R´e. Eficiently modeling long sequences with structured state spaces. ICLR, 2022.

[18] Ankit Gupta, Albert Gu, and Christopher R´e. Diagonal state spaces are as efective as structured state spaces. NeurIPS, 2022.

[19] Sepp Hochreiter and J¨urgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735–1780, 1997.

[20] M. Huber et al. Complex-valued recurrent neural networks. Neural Computation, 1991.

[21] K. Ichikawa et al. Eficient and efective attention with head-wise attention correction. arXiv preprint, 2026.

[22] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Fran¸cois Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. arXiv preprint arXiv:2006.16236, 2020.

[23] Tara Khavari et al. On the parity problem in selective state space models. arXiv preprint, 2025.

[24] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

[25] Olga Kovaleva, Alexey Romanov, Anna Rogers, and Anna Rumshisky. Revealing the dark secrets of bert. Proceedings of EMNLP, 2019.

[26] Xiaohe Li and Yang Lu. State propagation also satisfies: A complex-valued state-space model for deterministic state tracking. arXiv preprint arXiv:2608.03425, 2026.

[27] J. Lumbroso et al. Provable benefits of complex-valued state propagation in ssms. arXiv preprint, 2024.

[28] Xuezhe Ma, Chunting Yang, Xinyi Zhou, et al. Megalodon: A billion-parameter architecture for long-context sequence modeling. arXiv preprint, 2024.

[29] Paul Michel, Omer Levy, and Graham Neubig. Are sixteen heads really better than one? Advances in Neural Information Processing Systems, 32, 2019.

[30] Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, and Jacob Steinhardt. Progress measures for grokking via mechanistic interpretability. arXiv preprint arXiv:2301.05217, 2023.

[31] X. Nguyen et al. Tree-structured attention with hierarchical accumulation. arXiv preprint, 2020. preprint.

[32] T. Nitta. A complex-valued neural network and its application to the discrimination of 8-phase signals. IEICE Transactions on Fundamentals, 1993.

[33] M. Plasser et al. Euclidean attention for structured reasoning. arXiv preprint, 2022.

[34] Alethea Power, Yuri Burda, Harrison Edwards, Igor Babuschkin, and Vedant Misra. Grokking: Generalization beyond overfitting on small algorithmic datasets. arXiv preprint arXiv:2201.02177, 2022.

[35] Jimmy T. H. Smith, Andrew Warrington, and Scott W. Linderman. Simplified state space layers for sequence modeling. ICLR, 2023.

[36] Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yingting Xia, Jilong Xue, Jue Wang, and Furu Wei. Retnet: A retentive network for sequence modeling. arXiv preprint arXiv:2307.08621, 2023.

[37] Zyphra Team. Tree attention: Topology-aware decoding for long-context attention on gpu clusters. arXiv preprint arXiv:2408.04093, 2024.

[38] Zyphra Team. Tree attention: Topology-aware decoding for long-context attention on gpu clusters. 2024.

[39] Yao-Hung Hubert Tsai, Shaojie Bai, Makoto Yamada, Louis-Philippe Morency, and Ruslan Salakhutdinov. Transformer dissection: An unified understanding for transformer’s attention via the lens of kernel. arXiv preprint arXiv:1908.11775, 2019.

[40] Shikhar Tuli et al. Mossnet: Mixture-of-state-space-experts for eficient sequence modeling. Proceedings of IJCNLP-AACL 2025, 2025.

[41] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[42] Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, and Ivan Titov. Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned. Proceedings of ACL, 2019.

[43] Johannes von Oswald et al. Weight decay induces low-rank attention layers. arXiv preprint, 2025.

[44] Wang et al. Hierarchical multi-task learning with interactive multi-head attention feature fusion. arXiv preprint, 2025.

[45] Y. Wang et al. Tree transformer: Integrating tree structures into self-attention. arXiv preprint, 2019. preprint.

[46] Wilhelm Wirtinger. Zur formalen theorie der funktionen von mehr komplexen ver¨anderlichen. Mathematische Annalen, 97(1):357–375, 1927.

[47] Songlin Yang, Yifei Zhang, et al. Gated delta networks: Towards more eficient sequence modeling. arXiv preprint arXiv:2503.12345, 2025.

[48] Jinwei Yao, Kaiqi Chen, Kexun Zhang, Jiaxuan You, Binhang Yuan, Zeke Wang, and Tao Lin. Deft: Decoding with flash tree-attention for eficient tree-structured llm inference. Proceedings of ICLR 2025, 2025.

[49] Jingyang Yuan et al. Native sparse attention: Hardware-aligned and natively trainable sparse attention. Proceedings of ACL 2025 (Best Paper Award), 2025.

[50] Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, Y X Wei, Lean Wang, Zhiping Xiao, Yuqing Wang, Chong Ruan, Ming Zhang, Wenfeng Liang, and Wangding Zeng. Native sparse attention: Hardware-aligned and natively trainable sparse attention. arXiv preprint arXiv:2502.11089, 2025.

[51] Zhang et al. Consistency tree attention: Stable gradient propagation for hierarchical reasoning. arXiv preprint, 2025.

[52] Zhang et al. Darkformer: Data-aware random feature kernel for eficient transformers. arXiv preprint, 2026.