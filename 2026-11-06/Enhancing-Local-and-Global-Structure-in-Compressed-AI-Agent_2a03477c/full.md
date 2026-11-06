# Enhancing Local and Global Structure in Compressed AI Agent Memory

Henry Han

Abstract—AI agents store text as embedding vectors and compress them at scale. A compression is accepted when each item’s nearest neighbors survive, but agents also consolidate, delete, and compare memories, tasks that depend on which items lie far apart. We evaluate both ends with two label-free fidelities from our prior work: the local fidelity L at 10 neighbors and the global fidelity G at 30 to 50 percent of the memory. At 96 bytes per passage, rescored binary codes reach an L of 0.97 yet move 10.1 percent of passages to another theme; PCA with 8-bit codes, at an L of 0.77, moves 1.7 percent. Rescoring leaves the stored codes, and thus G, unchanged. We prove that across the G band, $1 - \nu _ { k }$ is at most 2 to 2.18 times the square root of the near–far rank-violation rate, and that this rate follows the distance noise of the compression (correlation 0.979). Our locality-aware quantization (LAQ) turns the bound into a hinge objective on the codebooks of an OPQ index, keeping its bytes, rotation, and search. To our knowledge, LAQ is the first quantizer whose objective comes from a proven bound on the far end. Reconstruction alone raises G in 0 of 52 held-out runs; the far hinge raises it in all 52. LAQ raises G in all 52 $( p = 8 . 5 \times 1 0 ^ { - 4 }$ over 13 memory–encoder groups), including the external FinDER memory, has the highest G of seven equal-size quantizers in all 12 Multi-Doc-2025 runs, scales to a million passages, and keeps evidence recall.

Index Terms—Vector quantization, embedding compression, agent memory, neighborhood preservation, rank correlation, product quantization, label-free evaluation.

✦

## 1 INTRODUCTION

L <sup>ANGUAGE-MODEL</sup> <sup>agents</sup> <sup>remember</sup> <sup>by</sup> <sup>embedding</sup> <sup>what</sup>they read, say, and do. When they need an item, they recall the nearest embeddings [1]. Managed agent memories extract records from conversations, retrieve them by semantic search, and consolidate them over time [2]. At millions of vectors, every byte per vector costs memory, latency, and money. The vectors are therefore compressed: by truncating Matryoshka embeddings [3], by projection, by scalar or binary codes, or by product quantization [4]. Cloud services offer each of these as a switch [5, 6].

How should an agent memory be compressed so that it keeps both what lies near and what lies far? Today, a compression is accepted by its nearest neighbors. Vendors report that binary embeddings retain 97 to 98.5 percent of retrieval accuracy [5] and evaluate quantized indexes by recall@100 [6]. At 96 bytes per passage on FinQA, rescored binary codes keep L at 0.97 and top-10 agreement at 0.87, yet move 10.1 percent of passages to another theme (Section 6). Embedding compression studies report nDCG@10 [7] or benchmark task scores [8]. Yet an agent also acts on what lies far apart: it consolidates a year of filings into themes [9, 10], deletes every item of a client or a session on request, and maps how companies and sectors relate. Because these operations compare each item with the whole memory, they decide what a sector brief says and what a deletion removes. Keeping this far end is difficult for two reasons: a private agent memory has no labels to test it with, and quantizers are trained for reconstruction, nearest neighbors, or query relevance. Our idea is to measure the far end with the same statistic that measures the near end, prove which pairwise order controls it, and train the codebooks on that order.

At the same bytes and search, our proposed localityaware quantization (LAQ) keeps the nearest neighbors of a held-out financial memory while moving far fewer passages to another theme (Figure 1a–c). In such a memory, passages from a single company form islands: in the outer fifth of the map, all 10 nearest neighbors of each passage belong to its own company. Passages from different banks meet in shared themes. At 128 bytes per passage, an optimized product quantization (OPQ) index [16] keeps local fidelity L at 0.85 but moves 224 of the 2,688 passages to another theme. LAQ keeps the bytes, the search, and L at 0.85, while moving only 94 passages.

Measurement. One label-free profile reads both ends of a compressed memory. Following our earlier work [17], we evaluate the two ends of a chance-corrected locality profile [18]: the near end (local structure) and the far end (global structure). Local fidelity L is chance-corrected recall@10. Global fidelity G averages, over 30 to 50 percent of the memory, the agreement on which items lie in each item’s nearer part. At its upper edge, the profile measures agreement on the near and far halves, which under a Gaussian copula equals Kendall’s τ of the distances. Both fidelities require only the vectors before and after compression, and G is estimated from a subsample of 2,000 items, however large the memory.

Learning. We prove that rank violations among cross pairs control the profile at every scale. A cross pair consists of one near item and one far item relative to an anchor. Let $\nu _ { k }$ be the profile at scale k and $N = n - 1$ for a memory of n items. Then $1 - \nu _ { k }$ is at most $N / \sqrt { k ( N - k ) }$ times the square root of the cross-pair violation rate. The leading factor

A financial memory held out from training, and what 128 bytes per passage keep of it

h LAQ minus OPQ on every held-out run, same bytes  
g LAQ in four steps, with the cost of each  
![](images/38475c8e8a63696b83e4dde41d230b4dbded8bed191793f0ba1a5302dca5cea7.jpg)

The locality profile: compressions of equal size agree at the near end and part toward the far end  
![](images/56ce1d2861563afdfe53327304590baa84ac823cbb7354950090d4a297eef255.jpg)  
LAQ learns the codebooks of an OPQ index and raises G on all 52 held-out memories

![](images/ef1d9ac80941d6c0eab67b2f23ff13b09c0d7a08dcc9d7dfad1720d33e90be42.jpg)

![](images/dda1364060eccc03b1309ef9c8e0f14c44bab43b1aa8bab9e69a67d29f30ac0c.jpg)

Fig. 1. At the same bytes and search, locality-aware quantization (LAQ) keeps the near end of an optimized product quantizer (OPQ) and moves fewer passages to another theme. L: local fidelity (chance-corrected recall@10); G: global fidelity (the profile averaged over 30 to 50 percent of the memory). (a) Multi-Doc-2025 [11], BGE-large [12]: 2,688 held-out passages of 22 companies, each joined to its 10 nearest neighbors and colored by theme (t-SNE layout [13]); companies form islands, and banks share themes. (b, c) At 128 bytes, both keep L at 0.85; OPQ moves 8.3 percent of the passages to another theme (red) and LAQ 3.5 percent, on the split where OPQ has its lowest theme agreement; over 16 runs the mean falls from 2.8 to 2.1 percent. (d, e) Locality profiles (Definition 1) of five compressions of FinQA [14] and LoCoMo [15]: compressions of equal size agree near k = 10 and part by up to 0.30 in the G band (gray). (f) Every lost–gained neighbor pair is a violated cross pair, which bounds the profile (Theorem 1). (g) LAQ in four steps; gray: cost of each step (Section 4). (h) LAQ minus OPQ on all 52 held-out runs: G rises in all 52 and L in 46.

stays between 2 and 2.18 across the G band. A hinge loss on distance gaps bounds this violation rate. LAQ lowers this hinge at middle and far scales by refining the codebooks of an OPQ index without labels or queries. The principal component analysis (PCA) basis, rotation, bytes per item, and search remain identical to those of the OPQ index.

What is new. To our knowledge, LAQ is the first quantizer trained on the far end of a memory, and its objective comes from a bound we prove. The closest learned quantizers train codebooks with pairwise or triplet losses. PQN uses triplets from class labels [19], and spreading vectors pair a nearest-neighbor triplet loss with an entropy term [20]. DPQ trains product codes end to end via straight-through assignments [21], and JPQ and Distill-VQ train with queries and relevance labels [22, 23]. LAQ differs in target, input, and output. Its target is the rank order at 30 to 50 percent of the memory, selected via our theoretical bound; its input is the unlabeled memory alone; and its output modifies only the codebook values of an OPQ index. We arrived at ν<sub>k</sub> from our DLP curve [18]. Rank-based quality criteria [24, 25] also count kept neighbors across scales and summarize them by an area that weights the near end. To our knowledge, the far end had no proven meaning until our earlier work showed that its midpoint is a rank correlation of distances [17] (Propositions 2–3). This paper takes the far end from a measure to a target. We prove that rank violations bound the profile at every scale (Theorem 1), that under noisy order the profile is fixed by one ratio, the distance noise of a compression against the spread of distances (Proposition 4, Corollary 2), and that at an OPQ start the far hinge falls at no first-order cost in reconstruction (Proposition 5). To our knowledge, no earlier quantizer, learned or not, trains its codes on the far end of this curve.

Contributions. Our contributions, in order of importance, are:

1) A quantizer derived from a bound. A new rank-violation bound ties the profile at every scale to the order of cross pairs, and a hinge surrogate makes it trainable. A noisyorder proposition and its corollary predict the violation rate and the profile in the G band from one ratio, the distance noise of the compression against the spread of distances. A further proposition shows that, at the OPQ start, the far hinge falls at no first-order cost in reconstruction, and the straight-through steps of training meet its condition at every start. Each result comes with a proof. LAQ turns the bound into a label-free objective on the codebooks of an OPQ index and replaces that index at the same bytes and search. The ablation confirms that the gain comes from the far hinge: reconstruction alone raises G in 0 of 52 held-out runs, the near hinge raises it by 0.0058 on average, and the far hinge raises it in all 52, by 0.0127.

2) Held-out evidence that it works. On held-out companies and conversations across three memories, LAQ raises G in all 40 runs and L in 34. The gain is largest with Amazon Titan V2 [5], where G rises by 0.032 to 0.038 on every split, two to five times the gain with the three retrievaltuned encoders. On FinDER [26], an external financial memory run once after the recipe was fixed, LAQ raises both fidelities in all 12 runs. On Multi-Doc-2025 it lifts the lowest theme agreement from 0.80 to 0.89 and has the highest G among seven quantizers of equal size. Paired tests confirm the gain: G rises in all 13 memory–encoder groups $( p = 8 . 5 \times 1 0 ^ { - 4 }$ , sign test), LAQ has a higher G than each of six other quantizers in all 12 Multi-Doc-2025 runs, and evidence recall@10 equals that of OPQ $( p = 0 . 8 4 )$ . On a million passages, one training run takes 175 seconds and a second run reproduces the codebooks bit for bit.

3) Why the gain matters. Each end governs its own operations. Across two agent memories and six encoders, retrieval tracks L (partial rank correlation 0.83 to 0.99 given G). With the retrieval-tuned encoders, consolidation and deletion track G (0.55 to 0.93 given L). Rescoring repairs the near end at query time but leaves the far end of the stored codes unchanged.

LAQ applies wherever a vector index also serves clustering, deduplication, or mapping.

Section 2 reviews related work. Section 3 presents the measurement theory and the bound, and Section 4 introduces the LAQ quantizer. Section 5 describes the memories, encoders, and operations. Section 6 shows what each end measures, and Section 7 evaluates LAQ on held-out memories.

## 2 RELATED WORK

We draw on three lines of research and apply them to agent memory: vector quantization, the evaluation of compressed vectors, and local and global structure.

Vector quantization. Quantizers compress vectors through subspace decomposition, additive codewords, or binary projections. Product quantization (PQ) splits a vector into subvectors and codes each separately [4], while OPQ learns a rotation to balance variance across subvectors [16]. Residual quantization (RQ) [27] and local search quantization (LSQ++) [28] represent a vector as a sum of codewords, whereas ScaNN weights quantization error by its impact on query inner products [29]. For binary representations, iterative quantization (ITQ) rotates the data before taking signs [30], and RaBitQ generates randomized binary codes with a theoretical error bound on distance estimates [31]. Learned quantizers train codebooks end to end. PQN applies an asymmetric triplet loss on labeled images [19]. UNQ trains a nonlinear encoder for nearest-neighbor search [32]. Spreading vectors combine a nearest-neighbor triplet loss with an entropy regularizer [20]. DPQ trains product quantization end to end via soft assignments and straight-through estimators [21]. In dense retrieval, JPQ, RepCONC, and Distill-VQ optimize quantizers using queries and relevance labels or teacher rankings [22, 23, 33]. Unlike these methods, our proposed LAQ trains codebooks for far-end rank order using the unlabeled memory alone.

Evaluation of compressed vectors. Quantized indexes are typically evaluated by nearest-neighbor recall [6, 34], retrieval quality [5, 7], or downstream task scores [8]. Kisako et al. [8] conjecture that both clustering and retrieval depend on neighborhood geometry, and Kuffo and Boncz [35] find that one-bit codes support the internal clustering an index uses to partition its vectors. Earlier, Turaga et al. [36] showed that fitting a moment-preserving quantizer per cluster preserves a known k-means partition. In contrast, our measure asks whether the compressed memory keeps both the partition and the global arrangement of the uncompressed memory, with no partition given in advance.

Local and global structure. Rank-based quality criteria evaluate neighborhood preservation across scales [24, 25]. Lueks et al. [37] split the co-ranking curve into local and global regimes, and recent visualization methods such as DREAMS and CMET balance or score both ends [38, 39]. Global embedding structure is also commonly scored by pairwise distance correlation and triplet accuracy [40, 41]. In our prior work, we introduced the degree of locality preservation [18]. In a subsequent study [17], we defined the local fidelity $L ,$ global fidelity $G ,$ and entity fidelity on its chance-corrected profile and proved that the profile’s midpoint converges to a rank correlation of distances. We also derived the subsample estimator of $G$ and applied these fidelities to two-dimensional maps and compressed agent memory. Building on that measurement framework, the present paper proves which pairwise order controls the profile, turns the resulting bound into a trainable quantizer, and evaluates both ends on held-out memories.

Agent memory. Benchmarks such as LoCoMo [15] and LongMemEval [42] evaluate whether agents can recall facts from long interactions. Agent memory systems consolidate new records against the most similar stored entries [2], while corpus-level frameworks cluster embeddings into themes and summaries [9, 10, 43]. Whereas these systems compress memory at the text level through summarization and consolidation, we study the compression of the underlying embedding vectors and the memory-wide operations that depend on them.

## 3 TWO-END FIDELITY OF A COMPRESSED MEMORY

One chance-corrected profile measures both ends of a compressed memory, while the relative order of near and far items controls this profile at every scale. This section defines the profile, proves what its two ends measure, defines the local and global fidelities $L$ and $G ,$ , and proves the bound that LAQ optimizes in Section 4.

## 3.1 Setting

A memory holds n items with vectors $x _ { 1 } , \ldots , x _ { n } \in \mathbb { R } ^ { d }$ where d is the embedding dimension. A compression maps each $x _ { i }$ to a stored code. We write $z _ { i }$ for the vector with which the memory computes distances: the decoded vector for scalar, PCA, and product codes, or the ±1 vector for binary codes compared by Hamming distance. All other distances are Euclidean. Let $N = n - 1$ . For item $i ,$ let $r _ { i } ^ { x } ( j ) \in \{ 1 , \ldots , N \}$ be the rank of item $j \neq i$ among the other items by increasing distance to $x _ { i } ,$ with ties broken by index, and let $r _ { i } ^ { z } ( j )$ be the corresponding rank among the compressed vectors. For $1 \leq k \leq \hat { N } - 1$ , the k-neighborhoods are $\begin{array} { r } { \dot { \mathcal { N } } _ { k } ^ { x } ( i ) = \{ j : r _ { i } ^ { x } ( j ) \le k \} } \end{array}$ and $\mathcal { N } _ { k } ^ { z } ( i ) = \{ j : r _ { i } ^ { z } ( j ) \leq k \}$

## 3.2 The locality profile

At every scale, the profile counts how many neighbors a compression keeps.

Definition 1 (Locality profile). The degree of locality preservation at scale k is

$$
\mathrm { D L P } _ { k } = \frac { 1 } { n k } \sum _ { i = 1 } ^ { n } \lvert \mathcal { N } _ { k } ^ { x } ( i ) \cap \mathcal { N } _ { k } ^ { z } ( i ) \rvert ,\tag{1}
$$

and its chance-corrected value is

$$
\nu _ { k } = \frac { N \mathrm { D L P } _ { k } - k } { N - k } .\tag{2}
$$

The locality profile is the curve $k \mapsto \nu _ { k }$ , read over the relative scale $k / n$

We introduced $\mathrm { D L P } _ { k }$ in our earlier work on the explainability of manifold learning [18]. Rank-based quality criteria for dimensionality reduction [24, 25] also count kept neighbors across scales. At a single scale k with the uncompressed memory as reference, $\tilde { \mathrm { D L P } } _ { k }$ is the recall@k used to evaluate approximate nearest-neighbor indexes [34]. Reading this quantity across every scale yields the profile, and the results below say what its two ends measure. Propositions 1–3 and Definition 2 are from our earlier work on nonlinear embeddings [17]; we give the proofs so that the paper is self-contained.

The next result gives the chance level and invariance of the profile: a compression that keeps neighbors at random scores $0 ,$ and codes whose distances differ in scale are compared on one statistic.

Proposition 1 (Chance level and invariance). (i) If each $\mathcal { N } _ { k } ^ { z } ( i )$ is a uniformly random k-subset of the other N items, independent of the uncompressed vectors, then $\mathbb { E } \operatorname { D L P } _ { k } = k / N$ and $\mathbb { E } \nu _ { k } = 0$ A compression that keeps every ranking has $\nu _ { k } = 1 . ( i i ) \nu _ { k }$ is invariant under any rotation, reflection, translation, or uniform scaling of the $z _ { i } ,$ and under any strictly increasing transformation of either distance.

All proofs except that of Theorem 1 are in Appendix A. Invariance (ii) allows one statistic to compare Euclidean distances between decoded vectors, Hamming distances between bits, and distances after a learned rotation. All 486 compressions evaluated in Section 6 lie above the chance level; the lowest $G$ among them, 0.04, belongs to UMAP [44] at 16 dimensions (MobileCLIP2 [45], FinQA).

The next result shows what the profile reads at its midpoint: for every item, which half of the memory lies near.

Proposition 2 (Near–far duality). Let $\mathcal { F } _ { k } ^ { x } ( i )$ and $\mathcal { F } _ { k } ^ { z } ( i )$ be the complements $o f \mathcal { N } _ { k } ^ { x } ( i )$ and $\mathcal { N } _ { k } ^ { z } ( i )$ among the other N items. Then $| \mathcal { F } _ { k } ^ { x } ( i ) \cap \mathcal { F } _ { k } ^ { z } ( i ) | = N - 2 k + | \mathcal { N } _ { k } ^ { x } ( i ) \bar { \cap } \mathcal { N } _ { k } ^ { z } ( i ) |$ . For $k ^ { \star } = N / 2$ with N even,

$$
\begin{array} { r l r } {  { \nu _ { k ^ { \star } } = 2 A - 1 , } } \\ & { } & { A = \frac { 1 } { n N } \sum _ { i = 1 } ^ { n } \Big | \big \{ j \neq i : \mathbb { k } [ r _ { i } ^ { x } ( j ) \leq k ^ { \star } ] = \mathbb { k } [ r _ { i } ^ { z } ( j ) \leq k ^ { \star } ] \big \} \Big | , } \end{array}\tag{3}
$$

where $A$ is the probability that the uncompressed and the compressed memory agree on whether a random other item lies in the nearer or the farther half.

While at small k the profile asks whether an item keeps its nearest neighbors, at $\grave { k } ^ { \star }$ it asks which half of the memory is near and which is far for every item. That question is decided by large distances, and thus by the relative positions of themes, sessions, and companies. The median split also acts as a rank correlation of all distances to an item, and under a Gaussian copula the two coincide.

Proposition 3 (Median split and rank correlation). Fix item i and let u be a uniformly random other item. Suppose that, after strictly increasing transformations, the pair of distances $\ r \tilde { ( } \| x _ { i } - x _ { u } \| , \| z _ { i } - z _ { u } \| )$ is bivariate normal with correlation $\gamma _ { i } ,$ that is, the pair has a Gaussian copula. Then for large n the contribution $2 A _ { i } - 1$ of item i to $\nu _ { k ^ { \star } }$ , with $A _ { i }$ its term in (3), has expectation

$$
\mathbb { E } \left( 2 A _ { i } - 1 \right) = \frac { 2 } { \pi } \arcsin \gamma _ { i } ,\tag{4}
$$

which is Kendall’s τ of the pair and a strictly increasing function of its Spearman correlation $\begin{array} { r } { \rho _ { i } ^ { S } = \frac { 6 } { \pi } } \end{array}$ arcsin $\begin{array} { l } { { \frac { \gamma _ { i } } { 2 } } } \end{array}$

Compressed memories follow this model closely. Across all 486 compressions, converting the Spearman correlation of all distances into $\textstyle { \frac { 2 } { \pi } }$ arcsin γ via Proposition 3 yields a rank correlation of 0.98 with the measured median split, and the two differ by only 0.03 on average.

## 3.3 Local and global fidelity

Two numbers summarize the profile, one read at each end.

Definition 2 (Local and global fidelity). The local fidelity and the global fidelity of a compressed memory are

$$
L = \nu _ { 1 0 } , ~ G = \frac { 1 } { | { \cal K } | } \sum _ { k \in { \cal K } } \nu _ { k } , ~ { \cal K } = \{ k : 0 . 3 \leq k / n \leq 0 . 5 \} .\tag{5}
$$

Here, L is chance-corrected recall@10, whereas $G$ averages the profile over the band whose upper edge is the median split of Proposition 2. Averaging over this band reduces the variance of a single $\nu _ { k }$ ⋆ while preserving the meaning of the median split. Within each of the 18 runs in Section ${ \bar { 6 } } , G$ and the median split have a rank correlation of 0.99 to 1.00 across the $2 7$ compressions. Because both fidelities evaluate the same chance-corrected statistic at two scales, they can be compared directly. Every compression thus lies on the $( L , G )$ plane, where its vertical distance below the diagonal $G = \bar { L }$ measures how much more of the far end it loses relative to the near end.

Because scale is defined relative to n, G can be estimated from a subsample: if $n _ { 0 }$ items are drawn uniformly, then for a relative scale $\alpha = k / n$ the $\lfloor \alpha n _ { 0 } \rfloor$ nearest items within the subsample approximate the α-quantile neighborhood of the whole memory. We compute G on five uniform subsamples of $2 { , } 0 0 0$ items and report the mean. Across twelve datasets of up to 1.38 million items, subsamples of 2,000 items estimate $\hat { G }$ within 0.010 of its value at 8,000 items [17]. By contrast, L refers to the fixed scale $k = 1 0$ and is computed on the whole memory. For each pair of items we record the neighbor rank in the uncompressed and in the compressed memory. $\mathrm { D L P } _ { k }$ counts the pairs ranked within k in both, so one running count over k gives $\mathrm { D L P } _ { k }$ and $\nu _ { k }$ at every scale in $O ( n _ { 0 } ^ { 2 } \log { n _ { 0 } } )$ time.

Which vectors each fidelity reads. Each fidelity is read from the vectors that its corresponding operations use. When a retrieval pipeline uses rescoring, it fetches the 40 nearest codes and rescores them using full-precision vectors read from disk. Rescoring modifies the lists returned to a query while leaving the stored codes unchanged. We therefore read L and retrieval from the rescored lists seen by the query. We read G and every other operation from the stored codes on which those operations run.

Figure 1d, e shows the profiles of real compressions. On the two agent memories, scalar quantization to 8 bits keeps the profile near 1 at every scale. Four compressions to 96 bytes start at similar heights near $k = 1 0$ and separate as k grows. Rotated binary codes fall fastest toward the G band.

## 3.4 Rank violations bound the profile

The order of near and far items controls the profile at every scale. Whereas the profile counts kept neighbors, a learning objective requires a quantity that varies smoothly with the codes. Pairwise rank order provides such a quantity. For item i, the anchor, and scale $k ,$ we call $( p , q )$ a cross pair if $p \in \mathcal { N } _ { k } ^ { x } ( i )$ and $q \notin \mathcal { N } _ { k } ^ { x } ( i )$ ; that is, $p$ is near and $q$ is far in the uncompressed memory. Item i has $k ( N - k )$ cross pairs, and a pair is violated if the compression reverses its order so that $r _ { i } ^ { z } ( q ) < r _ { i } ^ { z } ( p )$ . Let $p _ { k } ( i )$ be the fraction of violated cross pairs of i, and let $\begin{array} { r } { \bar { p } _ { k } = \frac { 1 } { n } \sum _ { i } p _ { k } ( i ) } \end{array}$

The theorem below states that a memory cannot lose many neighbors at scale k unless it reverses many cross pairs.

Theorem 1 (Rank-violation bound). For every compression and every $1 \leq k \leq N - 1$ ,

$$
1 - \nu _ { k } \ \leq \ \frac { N } { \sqrt { k ( N - k ) } } \cdot \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sqrt { p _ { k } ( i ) } \ \leq \ \frac { N } { \sqrt { k ( N - k ) } } \sqrt { \bar { p } _ { k } } .\tag{6}
$$

Proof. Fix i and let $m _ { i } = k - | \mathcal { N } _ { k } ^ { x } ( i ) \cap \mathcal { N } _ { k } ^ { z } ( i ) |$ . Exactly $m _ { i }$ items are lost, in $\mathcal { N } _ { k } ^ { x } ( i )$ and outside $\mathcal { N } _ { k } ^ { z } ( i )$ , and exactly $m _ { i }$ are gained, in $\mathcal { N } _ { k } ^ { z } ( i )$ and outside $\mathcal { N } _ { k } ^ { x } ( i )$ , because both sets have size $k .$ . Take a lost p and a gained $q .$ Then $p$ is near and $q$ is far in the uncompressed memory, so $( p , q )$ is a cross pair, and $r _ { i } ^ { z } ( q ) \leq k < \hat { r } _ { i } ^ { z } ( p )$ , so it is violated. These $m _ { i } ^ { 2 }$ pairs are distinct, hence $m _ { i } ^ { 2 } \le p _ { k } ( i ) k ( N - k )$ and $m _ { i } \ \leq$ $\overline { { \sqrt { p _ { k } ( i ) k ( N - k ) } } }$ . By (1), $\begin{array} { r } { 1 \dot { - } \operatorname { D L P } _ { k } = \frac { 1 } { n k } \sum _ { i } m _ { i } } \end{array}$ , and by (2), $\begin{array} { r } { 1 - \nu _ { k } = \frac { N } { N - k } ( 1 - \mathrm { D L P } _ { k } ) } \end{array}$ . Combining,

$$
1 - \nu _ { k } = \frac { N } { n k ( N - k ) } \sum _ { i = 1 } ^ { n } m _ { i } \leq \frac { N } { \sqrt { k ( N - k ) } } \cdot \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sqrt { p _ { k } ( i ) } .
$$

The second inequality is Jensen’s inequality for the concave square root. □

The middle term of (6) is $\begin{array} { r } { \frac { N } { \sqrt { k ( N - k ) } } \overline { { \sqrt { p _ { k } } } } , } \end{array}$ , with $\overline { { \sqrt { p _ { k } } } } =$ $\begin{array} { r } { \frac { 1 } { n } \sum _ { i } \sqrt { p _ { k } ( i ) } } \end{array}$ . This bound is strongest where G is read. Because the factor $N / \sqrt { k ( N - k ) }$ equals $1 / \sqrt { ( k / N ) ( 1 - k / N ) } .$ it lies between 2 and 2.18 across the G band $( 0 . 3 \leq k / n \leq$ 0.5). Consequently, a small violation rate at these scales forces $\nu _ { k }$ close to 1, transferring the bound directly to G. $\mathtt { B y }$ contrast, at $k = 1 0$ the factor is approximately ${ \dot { \sqrt { N / 1 0 } } } ,$ which reaches 14 for a memory of 2,000 items. Theorem 1 thus singles out the middle and far scales as the scales where ordering cross pairs controls the profile, motivating LAQ to train its codes at those scales. On the held-out memories of

Section $^ { 7 , }$ the bound holds across all 160 combinations of memory, quantizer, and scale (Figure 3e).

Because rank violations are counted by a step function, we bound them with a hinge on distance gaps. Let $\Delta ^ { x } =$ $\| x _ { i } - x _ { q } \| ^ { 2 } - \| x _ { i } - x _ { p } \| ^ { 2 } \geq 0$ be the uncompressed gap of a cross pair, $\Delta ^ { z ^ { . . } } = | | z _ { i } - z _ { q } | | ^ { 2 } - | | z _ { i } - z _ { p } | | ^ { 2 }$ be its compressed gap, and $h = [ \eta \Delta ^ { x } - \bar { \Delta } ^ { z } ] _ { + }$ <sub>+</sub> with margin $\eta > 0$ . Pushing compressed gaps above a fraction of the uncompressed gaps drives the violation rate down, as the following corollary shows.

Corollary 1 (Hinge surrogate). Let $( p , q )$ be a uniformly random cross pair of a uniformly random item at scale $k ,$ and $\check { H } _ { k } = \mathbb { E } h$ For every $\delta > 0$

$$
\bar { p } _ { k } \ \leq \ \mathrm { P r } ( \Delta ^ { x } < \delta ) + \frac { \bar { H } _ { k } } { \eta \delta } .\tag{7}
$$

The first term depends on the uncompressed memory alone, measuring the share of cross pairs whose two squared distances to i lie within δ of each other. The second term is the expected hinge, for which the average hinge over uniformly drawn cross pairs provides an unbiased estimate. Together, Theorem 1 and Corollary 1 turn the profile at a chosen scale into an objective that gradient descent can lower.

## 3.5 Predicting the far-end loss

Theorem 1 says that the far end is kept when few cross pairs are reversed. The next result says how many are reversed. For anchor i, let $D _ { j } = \Vert x _ { i } - x _ { j } \Vert ^ { 2 }$ and $\tilde { D } _ { j } = \Vert \dot { z } _ { i } - z _ { j } \Vert ^ { 2 }$ for $j \neq i ,$ and let $m _ { i }$ and $s _ { i }$ be the mean and standard deviation of the $D _ { i }$ . Fit the compressed distances by the least-squares line $\tilde { D } _ { j } \approx a _ { i } D _ { j } + b _ { i }$ with $a _ { i } > 0 .$ . A common map of this form leaves every rank unchanged, so a pair can be reversed by the residual $e _ { j } = ( \tilde { D } _ { j } - b _ { i } ) / \bar { a } _ { i } - D _ { j }$ alone. We call $\omega _ { i } = \mathrm { s d } ( \boldsymbol { \acute { e } } _ { j } ) / s _ { i }$ the relative distance noise of anchor i: the distortion of its distances measured against their spread. Let $\phi$ and Φ be the density and distribution function of $\mathcal { N } ( 0 , 1 )$ and $z _ { \alpha } = \Phi ^ { - 1 } ( \alpha )$

Proposition 4 (Noisy order). Fix $\alpha = k / n \in ( 0 , 1 )$ . Suppose the standardized distances $u _ { j } ~ = ~ ( D _ { j } - m _ { i } ) / s _ { i }$ of anchor i are independent $\mathcal { N } ( 0 , 1 )$ , and the scaled residuals $e _ { j } / s _ { i }$ are independent normal with mean 0 and variance $\omega _ { i } ^ { 2 }$ , independent of the $u _ { j }$ . Then, as $n  \infty ,$ , the expected violation rate of anchor i at scale k converges to

$$
\pi ( \alpha , \omega _ { i } ) = \frac { 1 } { \alpha ( 1 - \alpha ) } \int _ { - \infty } ^ { z _ { \alpha } } \int _ { z _ { \alpha } } ^ { \infty } \phi ( u ) \phi ( v ) \Phi \Big ( - \frac { u - v } { \sqrt { 2 } \omega _ { i } } \Big ) d u d v ,\tag{8}
$$

$$
\pi ( \alpha , \omega ) = \phi ( z _ { \alpha } ) ^ { 2 } \omega ^ { 2 } / \big ( 2 \alpha ( 1 - \alpha ) \big ) + O ( \omega ^ { 3 } ) a s \omega  0 .
$$

Proposition 4 counts reversed pairs. The same model gives the profile itself.

Corollary 2 (Profile under noisy order). Under the assumptions of Proposition $^ { 4 , }$ as $n  \infty$ the contribution of anchor i to $\nu _ { k }$ converges to

$$
\nu ( \alpha , \omega _ { i } ) = \frac { \Phi _ { 2 } ( z _ { \alpha } , z _ { \alpha } ; \varrho _ { i } ) - \alpha ^ { 2 } } { \alpha ( 1 - \alpha ) } , \qquad \varrho _ { i } = \big ( 1 + \omega _ { i } ^ { 2 } \big ) ^ { - 1 / 2 } ,\tag{9}
$$

where $\Phi _ { 2 } ( \cdot , \cdot ; \varrho )$ is the bivariate normal distribution function with unit variances and correlation $\varrho .$ It decreases strictly in $\omega _  i , $ , and at $\alpha = 1 / 2$ it equals $\frac { 2 } { \pi }$ arcsin $\varrho _ { i } ,$ , as in Proposition 3.

The far end thus depends on one ratio: how much the compression perturbs the distances from an anchor, against how much those distances spread. At the same distortion, a memory whose distances spread less loses more of its far end, and any quantizer that lowers $\omega _ { i }$ in the G band raises $\nu _ { k }$ there. The model rests on a normal approximation of each anchor’s distances. In the G band, these distances are sums over many directions and lie in the bulk of their distribution. $\mathrm { A t } k = 1 0 ,$ , they lie in the lower tail, where the local structure of the memory decides. The independence of the residuals also holds approximately: items that share a codeword have correlated errors, and the noise varies across items. Figure 12 tests the consequences of both approximations.

The held-out memories bear out both results (Figure 12 in Appendix C). Standardized by their normal mean and variance, the distances of each anchor match the normal quantile to within a median of 0.10 standard deviations at $\bar { k } = 0 . 4 n$ and 0.06 at $k = 0 . 5 n ,$ against 0.42 at $k \ = \ 1 0 .$ Across all 52 held-out runs, for the OPQ start and LAQ, $\pi ( 0 . 4 , \omega _ { i } )$ predicts the measured violation rate at $k = 0 . 4 n$ with correlation 0.979. The median error is 5.3 percent with the retrieval-tuned encoders, and the prediction is 25 to 31 percent low with Titan V2. These predictions use the full integral (8). Its leading term is larger, by a median of 25 percent with Titan V2, so the gap with Titan V2 lies in the assumptions of the model and not in the expansion. At $k = 1 0 \bar { . }$ , the normal model predicts 2.8 times the measured rate (median), so the near end keeps more order than the model allows. Evaluated at the mean relative distance noise of each run, Corollary 2 gives $\nu _ { k }$ at $k ~ = ~ 0 . 4 n$ within a median of 0.005 of the measured value (correlation 0.982). The relative distance noise ranges from 0.16 to 0.37 with the retrieval-tuned encoders and from 0.47 to 0.52 with Titan V2, whose OPQ start keeps the least of the far end. The data entropy of a memory [46], $\begin{array} { r } { h = - \sum _ { \ell } u _ { \ell } \log _ { 2 } u _ { \ell } } \end{array}$ with $u _ { \ell }$ the singular values of the standardized memory divided by their sum, tells how much LAQ lowers this noise. Within each retrieval-tuned encoder, across its 16 held-out runs, h has a rank correlation of 0.77 to 0.81 with the drop in mean relative distance noise from OPQ to LAQ and of 0.82 to 0.84 with the gain in $G \left( p \leq 4 \times 1 0 ^ { - 4 } \right.$ in each case). Memories of higher data entropy spread their information over more directions, and LAQ removes more of their distance noise.

## 4 LOCALITY-AWARE QUANTIZATION

Locality-aware quantization (LAQ) trains the codebooks of an OPQ index to keep the order of cross pairs at the middle and far scales, the target identified by Theorem 1. Whereas product quantizers learn codebooks to reconstruct vectors or keep nearest neighbors, LAQ adds this far-end target, needs no labels, and keeps the rest of the OPQ index intact. Figure 1g shows the procedure.

Starting point. LAQ first fits PCA on the training memory in the full basis, followed by OPQ with $M = d / 8$ subspaces of eight dimensions and $\dot { K } = 2 5 6$ codewords each, storing every item in $d / 8$ bytes. Let $\mu$ be the mean of the training vectors and R be the orthogonal map combining the PCA basis with the OPQ rotation. Let $y _ { i } = R ( x _ { i } - \mu ) $ denote the rotated item, $y _ { i , \ell }$ its ℓ-th subvector, and $C \ = \ \{ c _ { \ell , j } \}$ $( \ell = 1 , \dots , M , j = 1 , \dots , K )$ the codebooks initialized at the

OPQ values $C _ { 0 }$ . Each item is encoded by its nearest codeword in each subspace and decoded as the concatenation $z _ { i } ( C )$ Because R is orthogonal, distances in the rotated basis equal those in the original space.

Objective. LAQ minimizes

$$
J ( C ) = \frac { \mathcal { L } _ { \mathrm { r e c } } ( C ) } { \mathcal { L } _ { \mathrm { r e c } } ( C _ { 0 } ) } + \lambda _ { \mathrm { m i d } } \mathcal { L } _ { \mathrm { m i d } } ( C ) + \lambda _ { \mathrm { f a r } } \mathcal { L } _ { \mathrm { f a r } } ( C ) ,\tag{10}
$$

where $\begin{array} { r } { \mathcal { L } _ { \mathrm { r e c } } = \frac { 1 } { n } \sum _ { i } \| y _ { i } - z _ { i } \| ^ { 2 } } \end{array}$ is the reconstruction error, and ${ \mathcal { L } } _ { \mathrm { m i d } }$ and ${ \mathcal L } _ { \mathrm { f a r } }$ are hinge losses evaluated at two scale groups with weights $\lambda _ { \mathrm { m i d } }$ and $\lambda _ { \mathrm { f a r } }$ . For a scale group ${ \boldsymbol { s } } \in$ {mid, far} and a batch of sampled cross pairs $( p , q )$ with anchors $i ,$

$$
\mathcal { L } _ { S } = \frac { 1 } { s _ { S } } \operatorname* { m e a n } \bigl [ \eta \Delta ^ { x } - \Delta ^ { z } \bigr ] _ { + } ,\tag{11}
$$

where $\Delta ^ { x }$ and $\Delta ^ { z }$ are the distance gaps defined in Corollary 1. The normalizer $s _ { S }$ is the median uncompressed gap of the group, computed once at initialization so that both groups contribute on a common scale. The middle group draws $k / n \in \{ 1 \% , 5 \% \}$ , while the far group draws k uniformly from $\kappa ,$ the G band. The normalized reconstruction term keeps the near end, which product quantization already serves well.

Sampling cross pairs. Each step draws 1,024 anchors per group, and for each anchor samples a scale k and a cross pair. Half of the pairs are drawn uniformly: $p$ from the k nearest items of $i ,$ and $q$ from the remaining $N - k$ items. By Corollary 1, these pairs yield an unbiased estimate of $\bar { H } _ { k }$ which bounds the violation rate and, through Theorem $^ { 1 , }$ bounds $1 - \nu _ { k }$ . The other half straddle rank k: $p$ is drawn from ranks $k - w + 1$ to k and $q$ from ranks $k + 1$ to $k + w ,$ with w = max(3, ⌊0.1k⌋). Because lost and gained neighbors arise at this boundary, as shown in the proof of Theorem 1, these boundary pairs focus half of the gradient where the neighborhood changes.

Straight-through codes. Because encoding is a hard assignment, LAQ uses hard assignments in the forward pass and the soft assignment softmax $( - \| y _ { i , \ell } - c _ { \ell , j } \| ^ { 2 } / T )$ in the backward pass [21]. It re-encodes all training items with the current codebooks at every step. The temperature at step t decays as $T _ { t } = T _ { 0 } \cdot 0 . 0 5 ^ { t / t _ { \operatorname* { m a x } } }$ over $t _ { \mathrm { m a x } }$ steps, where $T _ { 0 }$ is the initial mean squared distance from a subvector to its nearest codeword. The codebooks are updated using Adam [47] with a cosine-decayed learning rate equal to 0.002 times the standard deviation of the entries of $\bar { C _ { 0 } }$

Checkpoint on a monitor set. LAQ draws 2,000 training items as a monitor set. Every 250 steps, it evaluates on this monitor set both $\nu _ { k }$ at $k / n ~ = ~ 0 . 4$ on the set (the midpoint of the G band) and the relative reconstruction error $\mathcal { L } _ { \mathrm { r e c } } \mathrm { \hat { ( } } C ) / \mathcal { L } _ { \mathrm { r e c } } ( C _ { 0 } )$ . It keeps the codebooks with the highest $\nu _ { k }$ among checkpoints whose relative reconstruction error is at most 1.15, falling back to $C _ { 0 }$ if no checkpoint improves upon the start.

The far hinge at the OPQ start. At the OPQ start, the far hinge can fall while the reconstruction error holds to first order. Let $n _ { \ell j }$ be the number of training items whose ℓ-th subvector is assigned to $c _ { \ell , j } ,$ , and let $\langle \cdot , \cdot \rangle$ be the inner product of codebooks.

Proposition 5 (Far hinge at no first-order cost). Suppose every codeword $o f C _ { 0 }$ is the mean of the training subvectors assigned to $i t ,$ no subvector is equidistant from its two nearest codewords, and ${ \mathcal { L } } _ { \mathrm { f a r } } ,$ , computed with the assignments $o f C _ { 0 } ,$ has a gradient g at $C _ { 0 }$ . Then, for every direction h with $\langle g , h \rangle > 0$ and all small enough $\epsilon > 0$ , the step $C _ { \epsilon } = C _ { 0 } - \epsilon I$ h satisfies

$$
\begin{array} { l } { { \displaystyle \mathcal { L } _ { \mathrm { r e c } } ( C _ { \epsilon } ) = \mathcal { L } _ { \mathrm { r e c } } ( C _ { 0 } ) + \epsilon ^ { 2 } \sum _ { \ell , j } \frac { n _ { \ell j } } { n } \| h _ { \ell j } \| ^ { 2 } , } } \\ { { \displaystyle \mathcal { L } _ { \mathrm { f a r } } ( C _ { \epsilon } ) = \mathcal { L } _ { \mathrm { f a r } } ( C _ { 0 } ) - \epsilon \langle g , h \rangle + o ( \epsilon ) . } } \end{array}\tag{12}
$$

Proposition 5 is a statement about the two training terms: along any direction that lowers the far hinge, the reconstruction error rises at second order in the step. This holds for $h = g$ and for the first step of Adam, which is $h = \mathrm { s i g n } ( g )$ up to scale, with $\langle g , \mathrm { s i g n } ( \bar { g } ) \rangle = \lVert g \rVert _ { 1 } > 0 .$

Straight-through steps. Training computes a different gradient. The straight-through codes use the hard assignments of Proposition 5 in the forward pass and the soft assignments at temperature $T$ in the backward pass, so the computed gradient $g _ { \mathrm { s t } }$ differs from g. As $T  0$ the soft assignments become hard and $g _ { \mathrm { s t } }  g _ { \cdot }$ . At the starting temperature $T _ { 0 } ,$ we measured both gradients on the same $^ { 8 , 1 9 2 }$ far cross pairs at all 52 OPQ starts of Section 7. The cosine between $g _ { \mathrm { s t } }$ and $g$ is 0.56 to 0.69 (median 0.62), so $h = g _ { \mathrm { s t } }$ meets the condition of Proposition 5 at every start. One first Adam step along $g _ { \mathrm { s t . } }$ , re-encoding all items, lowers the far hinge by 1.1 to 2.5 percent (median 1.6 percent) and raises the reconstruction error by at most 0.008 percent, while 0.05 to 0.11 percent of the subvectors change codeword.

From the hinge to G. A lower far hinge lowers the bound of Corollary 1 on the violation rate in the G band, and through Theorem 1 the bound on $1 - \nu _ { k }$ . The checkpoint rule keeps the gain on the monitor set: the start is itself a candidate, so the kept codebooks have a monitor $\nu _ { k }$ at least that of $C _ { 0 . }$ , at a reconstruction error at most 1.15 times that of $C _ { 0 } .$ . Section 7 tests the gain on held-out memories. The OPQ starts are close to the fixed point that Proposition 5 assumes. On all 52 training memories, one more centroid step lowers the reconstruction error by at most 0.032 percent (median 0.016 percent) and moves the codebooks by at most 0.6 percent of their norm. Corollary 2 also predicts the size of the gain. Evaluated at the mean relative distance noise of OPQ and of LAQ, it predicts a mean gain in $\nu _ { k }$ at $k = 0$ .4n of 0.0126, against 0.0124 measured on the held-out memories, with a rank correlation of 0.92 across runs and a predicted gain above zero in all 52.

What LAQ changes. LAQ modifies the codebook values and nothing else: the PCA basis, rotation, bytes per item, and asymmetric distance computation (ADC) search all remain identical to those of OPQ. An LAQ index therefore replaces an OPQ index directly and adds no operation at query time. A single recipe serves every memory and encoder: $t _ { \mathrm { m a x } } =$ 1,000 steps, $\bar { \lambda } _ { \mathrm { m i d } } = \lambda _ { \mathrm { f a r } } = \bar { 1 }$ , and $\eta = 1$ (Appendix B).

Complexity. Let n be the number of training items, d the dimension, $K = 2 5 6$ the codebook size, $B = 2 { , } 0 4 8$ the number of cross pairs per step (1,024 anchors in each of the two groups), and $t _ { \mathrm { m a x } }$ the number of steps. Ranking the training memory once costs $O ( n ^ { 2 } ( d + \log { \dot { n } } ) )$ time and $O ( n ^ { 2 } )$ memory. Each step encodes all items with the current codebooks in $O ( n K d )$ time and evaluates the hinge on B cross pairs in $O ( B d )$ time. Training therefore costs ${ \bar { O } } ( n ^ { 2 } ( d +$ $\log n ) ^ { - } + t _ { \operatorname* { m a x } } ( n K d + B d ) \gamma$ ) time in total, while the query cost and the encoding of a new item, $O ( K d )$ , remain those of OPQ. The $n \times n$ matrix of $^ { 7 , 3 3 6 }$ to $^ { 7 , 6 7 2 }$ items fits on one 48 GB GPU, and the 1,000 steps take 35 to 112 seconds on Multi-Doc-2025. For a memory of a million items, a scalable variant ranks against $m = \mathrm { { 8 , 1 9 2 } }$ reference items at each step and decodes only the items used in that step, with a reconstruction batch of $b = 1 6 { , } 3 8 4$ . This reduces the perstep cost to $O ( B m ( d + \log m ) + ( B + b ) K d )$ and memory to $O ( \bar { n } d + B m )$ (Section 7.5).

## 5 EXPERIMENTAL SETUP

We evaluate the two fidelities across five memories and six encoders, and test LAQ on held-out splits of four memories as well as on a million-passage memory.

Memories. Table 1 lists five memories: three agent memories and two topic corpora. FinQA, LoCoMo, and the topic corpora serve the measurement, and Multi-Doc-2025 and two additional memories test LAQ. FinQA [14] contains pages from the annual reports of S&P 500 companies paired with numerical questions annotated with supporting facts. We retain the 106 companies in the current index and up to three report pages per company. Every sentence and table row is one item, prefixed by the company name and fiscal year. This yields 8,356 items across 11 Global Industry Classification Standard (GICS) sectors and 863 ques tions. Multi-Doc-2025 [11] is a financial question-answering benchmark spanning 179 10-K filings from 89 S&P 500 companies for fiscal years 2022 to 2024. Each paragraph and table row serves as a candidate item with the same prefix. Every filing retains the passages containing its questions’ evidence plus a random sample, up to 56 passages, yielding 10,024 items and 898 questions with located evidence. Multi-Doc-2025 is used to test LAQ. LoCoMo [15] is a long-term conversational memory benchmark for agents, comprising ten conversations of 19 to 32 sessions, 5,882 turns, and 1,536 answerable questions annotated with their evidence turns. Each turn, prefixed by its speaker, forms one item. 20 Newsgroups [48] and AG News [49] contribute 10,000 random documents each, spanning 20 and 4 topics, respectively. A sixth memory, FinDER [26], is reserved for testing LAQ: it pairs $5 { , } 7 0 3$ queries written by financial professionals with expert-selected evidence from the 10-K filings of 497 S&P 500 companies. Following our Multi-Doc-2025 construction, we keep up to 30 passages per filing. Because FinDER provides the evidence text without specifying its company, we locate each excerpt in the filing that contains the most of its rarer words. When multiple filings match, the company named in the query breaks the tie, and any excerpt located in a company other than the one named is dropped. Taking the one or two passages in that filing whose words appear in the excerpt as the evidence passages yields 14,949 items and 5,129 questions. Finally, a seventh memory comprising 1,000,000 passages from MS MARCO [50] tests LAQ at scale (Section 7.5).

Encoders. Six encoders serve the fidelity measurement runs. Three are retrieval-tuned sentence encoders used with their query instructions: BGE-base-en-v1.5 (BGE-base)

TABLE 1  
Five memories with company, session, or topic structure, three with annotated questions.
<table><tr><td>Memory</td><td>Items</td><td>Questions Structure used</td><td></td></tr><tr><td>FinQA</td><td>8,356</td><td>863</td><td>company, sector, fiscal year</td></tr><tr><td>Multi-Doc-2025</td><td>10,024</td><td></td><td>898 company, sector, fiscal year</td></tr><tr><td>LoCoMo</td><td>5,882</td><td></td><td>1,536 conversation, session</td></tr><tr><td>20 Newsgroups</td><td>10,000</td><td></td><td>– topic, topic family</td></tr><tr><td>AG News</td><td>10,000</td><td></td><td>- topic</td></tr></table>

and BGE-large-en-v1.5 (BGE-large) [12] at 768 and 1,024 dimensions, and E5-base-v2 (E5-base) [51] at 768 dimensions. Three are encoders of other kinds: the text encoder of MobileCLIP2-L/14 [45] (768 dimensions), RoBERTa-base [52] with mean pooling (768), and averaged GloVe vectors [53] (300). A seventh encoder, the commercial Amazon Titan Text Embeddings V2 (Titan V2) [5] at 1,024 dimensions, is used on Multi-Doc-2025 to test LAQ. All vectors have unit length.

Compressions. There are 18 measurement runs: the six encoders on the two agent memories FinQA and LoCoMo, and the three encoders of other kinds on the two topic corpora. Each run applies 27 compressions fitted on the memory itself. Scalar codes comprise half precision along with 8- and 4-bit codes. Binary codes comprise 1-bit sign codes, the same codes with full-precision rescoring of the 40 nearest candidates, and 2- and 4-bit thermometer codes that set one bit per threshold passed on each coordinate. Truncation, PCA, and Gaussian random projection reduce vectors to $d / 2 , d / 4 , d / 8 ,$ and $d / 1 6$ dimensions. In addition, PCA to $d / 4$ and $d / 8$ is stored with 8-bit codes, and PCA to d and $d / 2$ is followed by a random rotation and binarization. Product quantization uses $d / 8 ,$ $d / 1 6 ,$ and $d / 3 2$ bytes, while UMAP [44] maps vectors to 16 dimensions. Finally, OPQ with and without rescoring is included at $d / 8$ bytes. For comparison, t-SNE [13], SRtSNE [54], PHATE [55], Isomap [56], and nonnegative PCA (NPCA) [57] are fitted on FinQA and LoCoMo with BGEbase (Table 4). Binary codes are compared by Hamming distance and all other codes by Euclidean distance on the decoded vectors. In our figures, PCA with 8-bit codes is labeled PCA + int8 and rescored binary codes binary + rescoring. Binarization after PCA and a random rotation is labeled PCA + rotation + binary (PCA + rot. + binary where space is short).

Operations. Each operation is executed on both the compressed and uncompressed memories, and we report their agreement. We model an agent’s memory operations directly on the vectors, which scores every operation on every compression without a language model in the loop.

• Retrieval. Each question retrieves its 10 nearest items (within its conversation for LoCoMo). Agreement is the overlap of the two top-10 lists, while quality is evidence recall@10, the share of a question’s evidence among those 10 items.

• Consolidation into themes. We run k-means on the compressed memory, initialized from the centroids of the uncompressed partition, using 11 themes for the financial memories and 20 and 4 themes for the topic corpora. Agreement is measured by the adjusted Rand index (ARI) [58], which we call theme agreement.

• Deletion. For each session or topic, the memory selects as many items as the session or topic contains, choosing those nearest its centroid. This models how a memory identifies deletion candidates by meaning when metadata are incomplete. Agreement is the Jaccard overlap of the deleted sets.

• Company operations. The information barrier walls off a company by applying deletion to it, while peer groups consist of each company’s five nearest companies by centroid. The company map is the rank correlation between distances among compressed company centroids and the uncompressed ones; the sector, session, and topic maps are defined in the same way.

• Global references. These use the subsamples of G. All distances is the Spearman correlation across 200,000 pairwise distances [40]. Triplets is the accuracy on random triplets [41]. The median split is $\nu _ { k ^ { \star } }$ ⋆ from Proposition 2.

Because consolidation, deletion, the session, sector, and topic maps, and the global references compare each item against the whole memory, we call them the far-end operations. Across the 27 compressions of each run, we relate each operation’s agreement to $L$ and $G$ using Spearman correlation as well as the partial rank correlation with one fidelity given the other.

Held-out protocol for LAQ. LAQ is trained and tested on disjoint companies or conversations. For FinQA and Multi-Doc-2025, each split holds out a quarter of the companies (26 of 106 and 22 of 89). For LoCoMo it holds out two of the ten conversations. Three splits (seeds 2026 to 2028) were fixed in advance. A fourth split (seed 2029) was drawn after the LAQ recipe was fixed and run once with the recipe unchanged. PCA, OPQ, and LAQ are fitted on the training part only, and every reported number is computed on the held-out part using its own questions. Evaluating the three retrievaltuned encoders across all three memories and Titan V2 on Multi-Doc-2025 yields 40 held-out runs, each comparing LAQ against its own OPQ start at $d / 8$ bytes. The other quantizers were evaluated on the first three splits. On Multi-Doc-2025, queries remain in full precision and are searched via ADC, just as an OPQ index serves them. $L , G ,$ and the far-end operations are evaluated on the stored codes. FinDER was run after all of the above, with the recipe unchanged. Each of its four splits (seeds 2026 to 2029) holds out 124 of its 497 companies, with queries searched by ADC. The three retrieval-tuned encoders thus provide 12 further runs, which we report separately from the 40.

Quantizers compared. On the held-out Multi-Doc-2025 memories, we compare LAQ at $d / 8$ bytes against OPQ and eight other quantizers. Five store codes of approximately the same size: PQ [4], RQ [27], LSQ++ [28], PCA to $d / \dot { 8 }$ dimensions with 8-bit codes, and RaBitQ [31] with d bits plus its per-vector factors (104 and 136 bytes). Sign and ITQ [30] binary codes are rescored on the 40 nearest candidates, and ScaNN uses anisotropic quantization [29]. All but ScaNN are fitted on the training companies.

## 6 WHAT THE TWO ENDS MEASURE

Each end of the profile predicts its own operations: retrieval follows $L ,$ consolidation and deletion follow $G ,$

and compressions of equal size can keep very different far ends. Section 7 tests whether LAQ raises the far end at the same bytes.

## 6.1 Compressions separate on the $( L , G )$ plane

Compressions of equal size keep different ends of a memory. Figure 2a places every compression of FinQA with BGE-base on the $( L , G )$ plane; in Appendix C, Figure 5 shows both agent memories with the retrieval-tuned encoders, and Figure 6 shows the other three encoders. Within a given memory, the two ends correlate across compressions (Spearman 0.81 to 0.95 over the 18 runs) because fewer bytes lower both. Yet compressions with the same L can differ widely in G. With the retrieval-tuned encoders, half precision and 8-bit scalar codes keep both ends at 0.97 or above. PCA, PCA with 8-bit codes, and product quantization keep $G$ within 0.12 of $L ,$ whereas binarization after PCA and a random rotation places G 0.06 to 0.22 below L. Among the linear compressions, rescored binary codes fall furthest below the diagonal, by 0.21 to 0.41. At 96 bytes per passage on FinQA with BGE-base, they keep L at 0.97 but move 10.1 percent of the passages to another theme. PCA with 8-bit codes keeps L at 0.77 and moves only 1.7 percent (Figure 2b, c). Among nonlinear maps with BGE-base, UMAP to 16 dimensions keeps L at 0.36 to 0.54 and G at 0.10 to 0.20, while t-SNE and PHATE keep G at 0.22 or less. Isomap keeps G at 0.25 and 0.43. NPCA lies above the diagonal, with G at 0.43 and 0.44 and L at 0.22 and 0.24, as does PCA to the same 16 dimensions, with G at 0.54 and L at 0.35 (Figure 2a, Table 4). With the encoders of other kinds, PCA and product quantization lie on or above the diagonal on both the agent memories and the topic corpora (Figures 6 and 10a–f). Onebit codes of GloVe and RoBERTa vectors reach G of 0.58 to 0.81 at L of 0.26 to 0.51. Because these encoders have coordinates of highly unequal variance, taking the sign of a low-variance coordinate perturbs the nearest neighbors far more than the nearer half.

## 6.2 Each end predicts its own operations

Retrieval follows $L ,$ whereas the far-end operations follow G (Figure 2d, e); Figure 7 in Appendix C plots each operation against both fidelities. Top-10 agreement lies close to a single increasing curve in $L ,$ with a median within-run Spearman correlation of 0.99 versus 0.89 for G. Conversely, the far-end operations lie closer to a single curve in $G ,$ with a median correlation of 0.95 versus 0.87 for L. Because L and G fall together under more severe compression, partial correlations separate the two ends even more sharply.

Partial correlations. This separation holds when conditioning on the other end: retrieval follows L given $G ,$ and with the retrieval-tuned encoders, consolidation and deletion follow G given L. Figure 8 in Appendix C gathers every operation across every run. Retrieval sits at the lower right, with a partial correlation with L given G of 0.83 to 0.99 (Tables 5 and 6 in Appendix C). Consolidation, deletion, and the session and topic maps sit at the upper left. With the retrieval-tuned encoders, consolidating the financial memory into themes follows G. Its partial correlations with G given L are 0.68, 0.76, and 0.66 for BGE-base, E5-base, and BGElarge, whereas with L given G they are 0.04, 0.01, and −0.02.

FinQA LoCoMo 20 Newsgroups AG News BGE-base E5-base BGE-large MobileCLIP2 RoBERTa GloVe

At 96 bytes, top-10 agreement of 0.87 against 0.60, yet six times as many passages change theme  
![](images/9723cf494e411c92b40a8dbed1f5c736374ecdd619f11364ab0a183c8c3855aa.jpg)  
Each operation follows the end at which it compares items: retrieval follows L, themes and deletion follow G

d Retrieval against L  
![](images/0310e0ee41f6a311c25ac0042c5d91a6d835cff64cc434ed4a23816267ec9a89.jpg)

e Themes and deletion against G  
![](images/84adeaf97d900c09803243bdb7d1251e15e003d06bb977b2859d10aaa58f26f1.jpg)

Rescoring repairs the near end at query time; the far end stays with the stored codes  
f FinQA: binary codes → rescored, six encoders  
![](images/92beb0b6ddf21dba73486309f567e8263e44d03e710babd44ceeb4fa17de1ad7.jpg)

g LoCoMo: binary codes → rescored, six encoders  
![](images/eb7e7f6b01e114e6ca48c68e83486e6c4fb578360593d290b78deeebb8c8388b.jpg)

Fig. 2. At the same bytes, a compression that keeps more nearest neighbors can displace more of the overall memory, because retrieval follows L whereas themes and deletion follow G (local and global fidelity). (a) $2 { \breve { 7 } }$ compressions of FinQA (BGE-base) alongside four nonlinear maps (rings: Isomap [56], NPCA [57], PHATE [55], t-SNE); triangle: PCA to 16 dimensions. Rescored binary codes lie 0.32 below the diagonal G = L, while t-SNE, PHATE, and UMAP keep G at 0.22 or less. (b, c) Two 96-byte compressions plotted on the uncompressed memory layout; colored points mark passages moved to another theme: 1.7 percent for PCA with 8-bit codes versus 10.1 percent for rescored binary codes. (d, e) Across every compression and run, retrieval follows L (median Spearman ρ 0.99; partial 0.97 given G), whereas themes and deletion follow G (ρ 0.95; partial 0.79 given L). (f, g) Rescoring the 40 nearest binary codes raises L and top-10 agreement while leaving G, themes, and deletion unchanged at the binary values (black ticks). All runs: Appendix C.

Deleting a LoCoMo session has partial correlations of 0.86, 0.55, and 0.93 with G given $L ,$ versus −0.11, 0.36, and −0.14 with L given G. The session and topic maps follow G at 0.56 to 0.93 given $L ,$ and the two global references, all distances and triplets, follow G at 0.78 to 1.00. The operations defined on companies sit between the two ends. Peer groups, the barrier, and the company and sector maps compare 106 company centroids, making each a nearest-neighbor question in the space of companies. With the retrieval-tuned encoders, they follow L (0.39 to 0.65 given G) at least as closely as $G$ (0.21 to 0.51 given $L )$ . In short, an operation follows the end of the profile at which it compares items.

## 6.3 Rescoring repairs the near end and keeps the stored codes

Rescoring repairs the near end while leaving the far end tied to the stored codes. Disk-based vector search keeps binary codes in memory and rescores candidates using fullprecision vectors read from disk; Figure 2f, g reproduces this pipeline using the 40 nearest codes. On FinQA, rescoring raises L from 0.74 to 0.97 with BGE-base, from 0.61 to 0.87 with E5-base, and from 0.77 to 0.98 with BGE-large. It lifts top-10 agreement from 0.57 to 0.87, 0.30 to 0.53, and 0.62 to 0.90. Yet G remains fixed at the value of the binary codes, as does every operation computed on them. Theme agreement on the financial memory stays at 0.79, 0.63, and 0.82, and LoCoMo session deletion stays at 0.56, 0.35, and 0.57. Judged by a recall check, rescored binary codes appear close to the uncompressed memory; judged by G, they keep only the far end of plain binary codes. The same pattern holds for every code family that preserves Euclidean distances (Table 7).

Evidence and fiscal year. With the retrieval-tuned encoders, evidence recall follows $L ,$ with partial correlations of 0.71 to 0.83 given G on FinQA and 0.61 to 0.87 on LoCoMo (Figure 9c in Appendix C). Because financial text repeats across years, a retrieved passage can come from the right company but the wrong fiscal year. In the uncompressed memories with the retrieval-tuned encoders, 0.18 of the right-company passages come from another fiscal year, and every linear compression keeps this share between 0.15 and 0.24 (Figure 9d).

## 6.4 Same bytes, different ends

Memories of the same size differ most at the far end. At $d / 8$ bytes per item, 32 times smaller than full precision, we compare seven compressions. Bytes per item count only the per-item codes. The PCA basis, the rotation, and the codebooks are shared across all items (5.5 MB with BGEbase and 9.4 MB with BGE-large, identical for OPQ and LAQ), while rescoring additionally retains the full vectors on disk. At these bytes, a memory can store one-bit codes with or without rescoring, rotated one-bit codes, productquantization codes, $d / 8$ principal components in 8 bits, or OPQ codes with or without rescoring. Figure 3a, b and Figure 11 in Appendix C compare these seven options, while Table 8 lists all 27 compressions on FinQA with BGE-base. On FinQA with BGE-base, rescored binary codes have the highest L among the first five methods (0.97) but yield a theme agreement of 0.79. PCA with 8-bit codes keeps theme agreement at 0.96. The same ordering holds with E5-base (0.96 against 0.63) and with BGE-large at 128 bytes (0.97 against 0.82). The gap also shows by company. Even when every company keeps at least 91 percent of its neighbors, rescored binary BGE-base codes move a fifth or more of the passages to another theme for 5 of the 106 FinQA companies (Figure 10g). With PCA and 8-bit codes, every company keeps at least 86 percent of its passages in their original theme. On LoCoMo, product quantization and PCA with 8-bit codes delete a session with an agreement of 0.43 to 0.68, compared with 0.35 to 0.57 for binary codes. Across every encoder on both memories, OPQ has the highest G of the seven compressions, 0.82 to 0.86 on FinQA and 0.85 to 0.91 on LoCoMo. It deletes a LoCoMo session with an agreement of 0.70 to 0.82. When rescored, OPQ reaches a top-10 agreement of 0.99 to 1.00. OPQ thus keeps both ends best among the seven compressions at these bytes, making it the natural starting point for LAQ.

TABLE 2  
LAQ raises G over its OPQ start in every held-out run. $d / 8$ bytes; runs in which LAQ is higher, means over runs, and the range of the change. Retrieval-tuned encoders (BGE-base, E5-base, BGE-large), four splits each.
<table><tr><td>Memory</td><td>higher</td><td>OPQ</td><td>LAQ</td><td>LAQ – OPQ</td></tr><tr><td>FinQA</td><td>G</td><td>12/12 .800</td><td>.808</td><td> $+ . 0 0 5 \mathrm { t o } + . 0 1 0$ </td></tr><tr><td></td><td>L</td><td>10/12 .861</td><td>.864</td><td>−.002 to +.008</td></tr><tr><td>LoCoMo</td><td>G</td><td>12/12 .845</td><td>.853</td><td>+.005 to +.012</td></tr><tr><td></td><td>L</td><td>8/12 .820</td><td>.822</td><td>−.002 to +.009</td></tr><tr><td>Multi-Doc-2025</td><td>G</td><td>12/12 .830</td><td>.842</td><td>+.007 to +.017</td></tr><tr><td></td><td>L</td><td>12/12 .823</td><td>.828</td><td>+.0003 to +.010</td></tr><tr><td>Multi-Doc-2025, Titan V2</td><td>G</td><td>4/4 .656</td><td>.692</td><td>+.032 to +.038</td></tr><tr><td></td><td>L</td><td>4/4 .790</td><td>.798</td><td>+.006 to +.011</td></tr><tr><td>FinDER</td><td>G</td><td>12/12 .821</td><td>.835</td><td> $+ . 0 1 0 \mathrm { t o } + . 0 1 7$ </td></tr><tr><td></td><td>L</td><td>12/12 .809</td><td>.814</td><td> $+ . 0 0 3 \mathrm { t o } + . 0 0 8$ </td></tr></table>

## 7 LAQ ON HELD-OUT MEMORIES

LAQ raises the far end of every held-out memory, keeps the near end, and improves the far-end operations. It also has the highest G among quantizers of the same size. These gains extend to an external financial memory and to a memory of one million passages.

## 7.1 LAQ raises the far end of every held-out memory

LAQ increases G in all 40 held-out runs (Table 2, Figure 1h). With the retrieval-tuned encoders, the gain ranges from 0.005 to 0.017. Every gain exceeds the standard deviation of G over subsamples, which is at most 0.0023. With Titan V2, whose OPQ start keeps the least of the far end, the gain reaches 0.032 to 0.038. L increases in 34 of the 40 runs and stays within 0.0023 of OPQ in the rest, so the far end is gained with the near end kept.

Per-company intervals. These gains hold across the companies of each memory. L, G, and evidence recall@10 are means over items or queries. We construct paired bootstrap intervals (95% confidence, 10,000 resamples) by resampling these terms by held-out company. On LoCoMo we resample by session, since the held-out part of each split holds two conversations. The interval for G lies above zero in all 52 held-out runs, including FinDER. The interval for $L$ lies

At d/8 bytes, OPQ keeps the highest G of seven compressions (0.82 to 0.91); rescoring moves $L ,$ never G

c Titan V2 profile  
a L and G, FinQA and LoCoMo, three encoders  
![](images/fd49c3af1b7bb3483e9b2fad14a0efbfed0bf6b658ac1dae3a8ac4a67a207db1.jpg)

b Far-end operations, the same compressions  
![](images/9d85a49dd8876e5c35ee7a4365101c92b6fc9fe5aea5b0efbbd6fce9fa9988b2.jpg)  
LAQ raises G over OPQ in all 40 held-out runs by lowering rank violations, 12 to 18 percent at $k \approx 0 . 4 n$

![](images/71f408ea00c0f0c58fda9303c8c66211ee6e05d31d9d96f44f1ff861f6068c69.jpg)

![](images/b45be7c3d02c49162e2395ee1232624af3e8089e0aa83699375790f24885ab8c.jpg)

![](images/91b717bde61afbfed0bbb61a56f11dd8abc4c270c98b1426646f41255f530cc8.jpg)

f Violated pairs  
![](images/d12a3368a73a3e82b7547d5e8cc7229fc09a8593adb2c59f4ddb4c2786457205.jpg)

The gain reaches the far-end operations: the lowest theme agreement rises from 0.80 to 0.89  
g Operations, every held-out run  
![](images/1ca2414d1eb9f4295c6fc074fd85408b215051f757ee45ef2a7c3066271a4b4f.jpg)

h Multi-Doc quantizers, d/8 bytes  
![](images/d7fd5240ab2141c9c793d5747b82dde90a94d34eea126e24126616500815b342.jpg)  
i Themes, Multi-Doc

![](images/27e8b738b22de38628692580deaf618574816a8049bb7c115f0cdfd92e595bff.jpg)  
Fig. 3. At the same bytes, OPQ keeps the far end best, and LAQ raises it further on every held-out memory by lowering rank violations $( L , G \colon$ local and global fidelity). (a, b) Seven compressions at $d / 8$ bytes on FinQA and LoCoMo with the retrieval-tuned encoders: OPQ has the highest G (0.82 to 0.91), whereas rescoring (arrows to open markers) raises L and top-10 agreement but leaves G and the far-end operations unchanged. (c–f) Held-out Multi-Doc-2025 across four splits: LAQ lies above OPQ at every scale, most notably in the G band (gray) and with Titan V2; the bound of Theorem 1 holds at all 160 points, with $1 - \nu _ { k }$ at 0.49 to 0.60 of it; and LAQ lowers the violation rate at k ≈ 0.4n by 12 to 18 percent on average (10 to 20 percent per run). (g) Change in each operation across all 40 held-out runs; right: runs in which LAQ is higher. (h) LAQ (black) has the highest G among seven quantizers in each of nine runs with the retrieval-tuned encoders. (i) The lowest theme agreement rises from 0.80 to 0.89.

## Reconstruction alone leaves G at the OPQ start; the far hinge lifts it in all 52 held-out runs

a Change in G for each training objective, every held-out run  
![](images/1bcf716aa04f9c415d84e8ec586ecbd2c35fb22c38c2025298f42802f889c3b1.jpg)

b LAQ gains what ${ \mathcal { L } } _ { \mathrm { f a r } }$ alone gains  
![](images/e2d24ea442425940ca4657aa1c18d7b7968cd8d0f3cbade3b2ec87767d0b8858.jpg)  
The far hinge acts in the G band, where Theorem 1 ties $\nu _ { k }$ to rank violations  
e Two ends, two hinges

c Gain by scale, retrieval-tuned encoders  
d Violation rate cut at k ≈ 0.4n  
![](images/608ebe4e20b8021c76eef459ee9f847222dded35d60fcee8cc4e4c6c3c003eae.jpg)

![](images/3aa61a373e3a7965719af693ecc7ee2720a3f3b9cc04273a70e7f12d05132f39.jpg)

![](images/b772c6f396cb18567f5ab9be99550ff9c78a7cc0914d93a549f9182afdc40f31.jpg)  
Paired tests: LAQ has higher G than OPQ in 52 of 52 runs and than each other quantizer in 12 of 12

f LAQ against OPQ, every held-out run  
![](images/5230b478b7fb222c2386d51377be55bf3b4b65414b5faff2a5820ec1f7eeccf3.jpg)

g LAQ against other quantizers, G  
![](images/62a538b3ccbc888c1409aa4b0bdc5f54fc17f585284968933087f16b77da0cf7.jpg)

h Company bootstrap, G  
![](images/6623b673a595c4c6787b0e86689d500a01633d44555f57388335deb7ddb9e14c.jpg)  
Fig. 4. The far hinge carries the gain in $G ,$ and LAQ is better than OPQ and the other quantizers in paired tests. (a) Change in $G$ from the OPQ start for each training objective on all 52 held-out runs; bars are means. (b) LAQ against reconstruction with ${ \mathcal { L } } _ { \mathrm { f a r } }$ alone, run by run. (c) Mean change in $\nu _ { k }$ by scale with the retrieval-tuned encoders; shading, interquartile range; gray, the G band. (d) Reduction of the violation rate ${ \bar { p } } _ { k }$ at k ≈ 0.4n relative to OPQ. (e) Change in L against change in G; large markers are means. (f) One-sided Wilcoxon tests of LAQ against OPQ over runs, Holm-corrected, with the runs in which LAQ is higher; evidence recall@10 is tested two-sided. (g) G of LAQ minus that of each quantizer on Multi-Doc-2025, 12 runs (h) Paired company-bootstrap 95 percent intervals for G, LAQ minus OPQ.

above zero in 25 runs and below zero in none. LAQ keeps evidence recall@10: its interval contains zero in 47 of the 52 runs.

Mechanism. The gain spans the whole profile (Figure 3c–f). Across the 16 held-out Multi-Doc-2025 memories, LAQ lifts $\nu _ { k }$ above OPQ at all scales from $k = 1 0 \mathrm { t o } k ^ { \star }$ . The gain grows toward the G band, where LAQ trains. Theorem 1 holds at all 160 combinations of memory, quantizer, and scale, with $1 - \nu _ { k }$ at 0.49 to 0.60 of the bound. The theorem also explains the mechanism. $\mathrm { A t } k \approx 0 . 4 n , \mathrm { L A Q }$ reduces the violation rate $\bar { p } _ { k }$ by 10 to 16 percent with the retrieval-tuned encoders and by 16 to 20 percent with Titan V2. This lowers the bound on $1 - \nu _ { k }$ in the G band, and the measured $\nu _ { k }$ rises there as well.

Ablation. The far hinge ${ \mathcal L } _ { \mathrm { f a r } }$ carries the gain in G (Figure 4a, b; Table 10 in Appendix D). We retrained every held-out run with the same recipe and checkpoint rule and changed only the terms of Eq. (10). With reconstruction alone, training keeps the OPQ start $C _ { 0 }$ in all 52 runs, since OPQ already fits its codebooks to reconstruction. A near hinge, Eq. (11) at $k = 1 0$ or 50 on stored and ADC distances, raises G by 0.0058 on average. Reconstruction with ${ \mathcal L } _ { \mathrm { f a r } }$ raises G in all 52 runs, by 0.0127 on average, and above the near hinge in 48 of them (paired Wilcoxon test, $p = 6 \times 1 0 ^ { - 9 } )$ The full objective raises G by 0.0124. With the retrieval-tuned encoders, its G equals that of ${ \mathcal L } _ { \mathrm { f a r } }$ alone (mean difference 0.0001), and the middle hinge adds to L (0.0036 against 0.0028). With Titan V2, ${ \mathcal L } _ { \mathrm { f a r } }$ alone raises G by 0.041 and the full objective by 0.036, with the larger gain in L (0.008 against 0.005). The far end is thus learned from the order of cross pairs in the G band, the pairs that Theorem 1 links to $\nu _ { k }$

Where the far hinge acts. The far hinge acts in the G band (Figure 4c–e). The near hinge lifts $\nu _ { k }$ most near $k = 0 . 0 5 n ,$ whereas ${ \mathcal L } _ { \mathrm { f a r } }$ and LAQ lift it most in the G band. $\mathbf { A } \mathbf { t } k \approx 0 . 4 n _ { . }$ the near hinge reduces the violation rate by 6 percent with the retrieval-tuned encoders, and ${ \mathcal L } _ { \mathrm { f a r } }$ and LAQ by 11 percent.

Paired tests. We test whether LAQ is better than OPQ with a one-sided Wilcoxon signed-rank test over the 52 held-out runs and correct each family of tests by the Holm method (Figure 4f). Runs that share a corpus are dependent, so we repeat each test as a sign test over the 13 combinations of memory and encoder, each the mean over its splits. LAQ has higher G, L, all-distance correlation, triplet agreement, and median split at $p \leq 6 . 4 \times 1 0 ^ { - 9 }$ over runs and $\overset { \smile } { p } = 8 . 5 \times 1 0 ^ { - 4 }$ over the 13 combinations. Theme agreement and session deletion are higher at $p = 0 . 0 2 8$ and $p = 0 . 0 4 6$ over runs. Evidence recall@10 is equal to that of OPQ (p = 0.84, twosided). The company-bootstrap intervals for G lie above zero in all 52 runs (Figure 4h).

## 7.2 The gain reaches the far-end operations

Global references and maps. The far-end operations of Section 6 move with G (Figure 3g). Both global references and the median split improve in all 40 runs. The Spearman correlation of all distances increases by 0.002 to 0.016 with the retrieval-tuned encoders and by 0.046 to 0.063 with Titan V2. The median split increases by 0.004 to 0.019 and by 0.036 to 0.043, respectively. The company map improves in 27 of

TABLE 3  
LAQ has the highest G with every encoder. G / L on held-out Multi-Doc-2025 at d/8 bytes, means over the first three splits. RaBitQ uses 104 and 136 bytes. ScaNN is fitted on the held-out memory and searched with full-precision queries; we report its L.
<table><tr><td></td><td>BGE-base</td><td>E5-base BGE-large</td><td></td><td>Titan V2</td></tr><tr><td>LAQ</td><td></td><td></td><td>.832 / .817 .830 / .807 .866 / .857 .696 / .800</td><td></td></tr><tr><td>OPQ</td><td>.820 / .814 .816 / .801 .857 / .852 .659 / .792</td><td></td><td></td><td></td></tr><tr><td>PQ</td><td>.754 / .743 .759 / .749 .769 / .771 .661 / .787</td><td></td><td></td><td></td></tr><tr><td>LSQ++</td><td>.759 / .725 .730 / .689 .764 / .744 .543 / .679</td><td></td><td></td><td></td></tr><tr><td>RQ</td><td>.693 / .582 .693 / .552 .697 / .598.429 / .511</td><td></td><td></td><td></td></tr><tr><td></td><td>PCA + int8 .729 / .631 .710 / .584 .768 / .695 .521 / .578</td><td></td><td></td><td></td></tr><tr><td>RaBitQ</td><td>.677 / .666.561 / .545.712 / .704 .656 / .776</td><td></td><td></td><td></td></tr><tr><td>ScaNN</td><td>- / .615</td><td></td><td>5 - / .507 - / .578 - / .837</td><td></td></tr></table>

28 runs. The session map improves in 10 of 12 runs, and the remaining two stay within 0.001 of OPQ.

Themes and deletion. Consolidation into themes gains the most where the OPQ start keeps it least. On Multi-Doc-2025 with the retrieval-tuned encoders, the mean theme agreement rises from 0.908 to 0.931. The lowest theme agreement across the 12 runs rises from 0.800 to 0.886 (Figure 3i). With Titan V2, both keep theme agreement at 0.98 or above. Across the 16 Multi-Doc-2025 runs, OPQ moves 1,203 passages to another theme, whereas LAQ moves 902. On FinQA, the mean theme agreement rises from 0.911 to 0.914. For conversational memory, deleting a LoCoMo session improves in 9 of 12 runs, with mean agreement rising from 0.686 to 0.695.

Retrieval. Retrieval is served by the near end, which LAQ keeps. Top-10 agreement increases in 27 of 40 runs. Mean evidence recall@10 stays within 0.005 of OPQ in every memory: 0.545 versus 0.546 on FinQA, 0.649 versus 0.649 on LoCoMo, 0.795 versus 0.796 on Multi-Doc-2025, and 0.749 versus 0.754 with Titan V2. Peer groups and the information barrier are nearest-neighbor questions among company centroids. As Section 6 found, they follow L at least as closely as G. LAQ raises peer agreement in 17 of 28 runs and keeps barrier agreement within 0.011 of OPQ in every run. All runs are listed in Appendix D (Table 9).

## 7.3 LAQ against other quantizers at the same size

LAQ has the highest G among the quantizers of its size (Table 3, Figure 3h). In each of the nine runs with the retrieval-tuned encoders, LAQ has both the highest G and the highest L of all codes without rescoring. LSQ++, the strongest additive quantizer tested, reaches a G of 0.73 to 0.77, against 0.83 to 0.87 for LAQ. With Titan V2, LAQ has the highest G in all three runs, and ScaNN, fitted on the held-out memory itself, has the highest L. With rescoring of the 40 nearest codes, LAQ reaches an L of 0.995 to 0.999 in all twelve runs. This is above rescored sign codes (0.861 to 0.996) and rescored ITQ codes (0.938 to 0.994) in each run, and the stored LAQ codes keep the highest G. Against each of OPQ, PQ, LSQ++, RQ, PCA + int8, and RaBitQ, LAQ has higher G and L in all 12 runs (paired Wilcoxon test, $p = 2 . 9 \times \mathrm { \tilde { 1 0 ^ { - 3 } } }$ after Holm correction; Figure 4g).

## 7.4 An external financial memory

These gains hold on FinDER, a memory built by others from the queries of financial professionals. It was run once, with the recipe unchanged, on 124 held-out companies in each of four splits (Table 2). LAQ raises G in all 12 runs by 0.010 to 0.017, lifting the mean from 0.821 to 0.835. Every gain exceeds the standard deviation of G over subsamples, which is at most 0.0025 here. L increases in all 12 runs by 0.003 to 0.008, and top-10 agreement rises in all 12. Mean evidence recall@10 is 0.458 for both, within 0.007 of OPQ in every run.

## 7.5 A memory of a million passages

LAQ also raises G on a memory of one million passages. The memory holds 1,000,000 passages of MS MARCO [50], encoded with BGE-base and stored at 96 bytes each. OPQ and LAQ were fitted on 100,000 other passages, and the 6,980 development queries search the memory by ADC. At this size, training draws a new subsample of 8,192 reference items at each step and chooses the cross pairs by their ranks within it, the argument behind the estimator of G. It decodes only the items a step uses and averages the reconstruction term over 16,384 items. All other settings are those of the recipe. Over three seeds, LAQ raises G from 0.774 to 0.784 to 0.785. The gain of 0.010 to 0.011 is about three times the standard deviation of G over subsamples (at most 0.0039). L, computed on the 10 nearest neighbors of 20,000 random passages, stays within 0.002 of OPQ. Top-10 agreement is 0.818 for OPQ and 0.818 to 0.820 for LAQ. Evidence recall@10, over the 820 queries whose relevant passage is in the memory, is 0.860 for OPQ and within 0.008 of it for LAQ. One training run takes 175 seconds and at most 10.0 GiB of one 48 GB GPU, and a second run reproduces the codebooks bit for bit.

## 8 DISCUSSION

Why the far end can be learned. Ordering cross pairs controls the far end, and this is what makes LAQ work. The bound has a leading factor near 2 across the G band and near $\sqrt { N / 1 0 }$ at k = 10. Reconstruction and nearestneighbor objectives serve the near end. LAQ combines the two through a hinge loss at middle and far scales, a normalized reconstruction term for the near end, and a checkpoint selected on the far end of a monitor set. The ablation of Section 7 isolates the far hinge: reconstruction alone keeps the OPQ start, and ${ \mathcal L } _ { \mathrm { f a r } }$ alone gives the full gain in G in all 52 runs (Figure 4). The gain is largest where the OPQ codes keep the least of the far end, as with Titan V2, and it reaches the operations that follow G.

Cost against other quantizers. LAQ trains in minutes, between PQ and the additive quantizers, and keeps the query cost of OPQ. Each LAQ step costs about one k-means iteration of PQ and OPQ [4, 16], plus a one-time ranking of the training memory. In contrast, additive quantizers spend their cost on encoding. RQ keeps several partial codes per item in a beam search, and LSQ++ encodes vectors by joint local search over all codebooks [27, 28]. ITQ alternates a d × d rotation with sign codes [30], and RaBitQ applies a random rotation without training [31]. On the Multi-Doc-2025 training memories, the faiss implementations [59] on

CPU took 6 to 12 seconds to fit PQ, 146 to 517 seconds for ITQ, 511 to 1,193 seconds for RQ, and 8.2 to 17.1 hours for LSQ++. By comparison, LAQ took 35 to 112 seconds on one GPU after its OPQ start. LAQ and OPQ encode a new item in O(Kd) time, while RQ and LSQ++ repeat their search for every item added to the memory. LAQ thus has the highest G among these quantizers (Table 3) at a training cost between PQ and the additive quantizers.

What to report. A compressed memory can be reported with (L, G) against the memory it replaces, computed on the operator’s own vectors. L says whether queries return the same items. G says whether the memory keeps its far end: which items are consolidated or deleted together, and how sessions, companies, and themes relate. Neither needs labels, and G is estimated from a subsample of 2,000 items. Where a memory serves consolidation or deletion, G belongs in its acceptance test. The compression is then chosen as the one with the highest G among those with the required L at the bytes available. Theorem 1 adds a diagnostic: the violation rate of cross pairs at a scale shows which part of the profile a compression loses.

Financial memory. The consequences are clearest in financial memories. A consolidation that groups filings differently changes the sector briefs an analyst agent writes. A deletion that selects different passages changes what a client request removes. Peer groups, the information barrier, and the company map compare company centroids. A check of these operations combines L with agreement measured on the firm’s own companies.

Extensions. Residual and additive codebooks are natural next targets for this objective, and sampling reference items already extends training to one million passages (Section 7.5).

## 9 CONCLUSION

Agent memories are compressed and accepted by their nearest neighbors, even though agents also consolidate, delete, and map what they hold. One label-free profile measures both ends. Its near end predicts retrieval, and with retrieval-tuned encoders its far end predicts consolidation and deletion, each beyond the other. Our main result is that the order of near and far items bounds the profile at every scale, and that this bound can be trained. Locality-aware quantization (LAQ) learns this order and raises the global fidelity of every held-out memory at the same bytes and search as the OPQ index it replaces. To our knowledge, LAQ is the first quantizer whose objective comes from a proven bound on the far end of a memory. It needs no labels and no change at query time, and it applies wherever embeddings are compressed and then clustered, deduplicated, or mapped. For agent memory, (L, G) gives an acceptance test for both ends, and LAQ gives a way to pass it.

Data and code: https://github.com/hank08819/LAQ

## REFERENCES

[1] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel, S. Riedel, and D. Kiela, “Retrieval-augmented generation for knowledge-intensive NLP tasks,” in Advances in

Neural Information Processing Systems, vol. 33, 2020, pp. 9459–9474.

[2] Amazon Web Services, “Building smarter AI agents: AgentCore long-term memory deep dive,” AWS Machine Learning Blog, Oct. 2025.

[3] A. Kusupati, G. Bhatt, A. Rege, M. Wallingford, A. Sinha, V. Ramanujan, W. Howard-Snyder, K. Chen, S. Kakade, P. Jain, and A. Farhadi, “Matryoshka representation learning,” in Advances in Neural Information Processing Systems, vol. 35, 2022, pp. 30 233–30 249.

[4] H. Jégou, M. Douze, and C. Schmid, “Product quantization for nearest neighbor search,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 33, no. 1, pp. 117–128, 2011.

[5] Amazon Web Services, “Build cost-effective RAG applications with binary embeddings in Amazon Titan Text Embeddings V2, Amazon OpenSearch Serverless, and Amazon Bedrock Knowledge Bases,” AWS Machine Learning Blog, Nov. 2024.

[6] A. Govindaraju and V. V. Nakkirtha, “Cost optimized vector database: Introduction to Amazon OpenSearch Service quantization techniques,” AWS Big Data Blog, Jan. 2025.

[7] N. Huerga-Pérez, R. Álvarez, R. Ferrero-Guillén, A. Martínez-Gutiérrez, and J. Díez-González, “Optimization of embeddings storage for RAG systems using quantization and dimensionality reduction techniques,” arXiv preprint arXiv:2505.00105, 2025.

[8] R. Kisako, H. Tsukagoshi, and R. Sasano, “When is 0.1% enough? Analyzing the combined effects of dimensionality reduction and quantization on text embedding compression,” arXiv preprint arXiv:2606.01074, 2026.

[9] P. Sarthi, S. Abdullah, A. Tuli, S. Khanna, A. Goldie, and C. D. Manning, “RAPTOR: Recursive abstractive processing for tree-organized retrieval,” in International Conference on Learning Representations (ICLR), 2024.

[10] M. Grootendorst, “BERTopic: Neural topic modeling with a class-based TF-IDF procedure,” arXiv preprint arXiv:2203.05794, 2022.

[11] “Multi-Doc-2025: A financial question-answering benchmark over SEC 10-K filings,” Hugging Face dataset Anonymous-Team-HC-RAG/Multi-doc-2025, 2026.

[12] S. Xiao, Z. Liu, P. Zhang, N. Muennighoff, D. Lian, and J.-Y. Nie, “C-Pack: Packed resources for general Chinese embeddings,” in Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2024, pp. 641–649.

[13] L. van der Maaten and G. Hinton, “Visualizing data using t-SNE,” Journal of Machine Learning Research, vol. 9, pp. 2579–2605, 2008.

[14] Z. Chen, W. Chen, C. Smiley, S. Shah, I. Borova, D. Langdon, R. Moussa, M. Beane, T.-H. Huang, B. Routledge, and W. Y. Wang, “FinQA: A dataset of numerical reasoning over financial data,” in Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, 2021, pp. 3697–3711.

[15] A. Maharana, D.-H. Lee, S. Tulyakov, M. Bansal, F. Barbieri, and Y. Fang, “Evaluating very long-term conversational memory of LLM agents,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics, 2024.

[16] T. Ge, K. He, Q. Ke, and J. Sun, “Optimized product quantization,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 36, no. 4, pp. 744–755, 2014.

[17] H. Han, “Quantifying local and global fidelity of nonlinear embeddings,” arXiv preprint arXiv:2611.00290v2 [cs.AI], 2026, doi: 10.48550/arXiv.2611.00290.

[18] H. Han, W. Li, J. Wang, G. Qin, and X. Qin, “Enhance explainability of manifold learning,” Neurocomputing, vol. 500, pp. 877–895, 2022.

[19] T. Yu, J. Yuan, C. Fang, and H. Jin, “Product quantization network for fast image retrieval,” in European Conference on Computer Vision (ECCV), 2018, pp. 186–201.

[20] A. Sablayrolles, M. Douze, C. Schmid, and H. Jégou, “Spreading vectors for similarity search,” in International Conference on Learning Representations (ICLR), 2019.

[21] T. Chen, L. Li, and Y. Sun, “Differentiable product quantization for end-to-end embedding compression,” in International Conference on Machine Learning (ICML), ser. PMLR, vol. 119, 2020, pp. 1617–1626.

[22] J. Zhan, J. Mao, Y. Liu, J. Guo, M. Zhang, and S. Ma, “Jointly optimizing query encoder and product quantization to improve retrieval performance,” in ACM International Conference on Information and Knowledge Management (CIKM), 2021, pp. 2487–2496.

[23] S. Xiao et al., “Distill-VQ: Learning retrieval oriented vector quantization by distilling knowledge from dense embeddings,” in ACM SIGIR Conference on Research and Development in Information Retrieval, 2022.

[24] J. A. Lee and M. Verleysen, “Quality assessment of dimensionality reduction: Rank-based criteria,” Neurocomputing, vol. 72, pp. 1431–1443, 2009.

[25] J. A. Lee, D. H. Peluffo-Ordóñez, and M. Verleysen, “Multi-scale similarities in stochastic neighbour embedding: Reducing dimensionality while preserving both local and global structure,” Neurocomputing, vol. 169, pp. 246–261, 2015.

[26] C. Choi, J. Kwon, J. Ha, H. Choi, C. Kim, Y. Lee, J.-y. Sohn, and A. Lopez-Lira, “FinDER: Financial dataset for question answering and evaluating retrieval-augmented generation,” arXiv preprint arXiv:2504.15800, 2025.

[27] Y. Chen, T. Guan, and C. Wang, “Approximate nearest neighbor search by residual vector quantization,” Sensors, vol. 10, no. 12, pp. 11 259–11 273, 2010.

[28] J. Martinez, S. Zakhmi, H. H. Hoos, and J. J. Little, “LSQ++: Lower running time and higher recall in multi-codebook quantization,” in European Conference on Computer Vision (ECCV), 2018.

[29] R. Guo, P. Sun, E. Lindgren, Q. Geng, D. Simcha, F. Chern, and S. Kumar, “Accelerating large-scale inference with anisotropic vector quantization,” in International Conference on Machine Learning (ICML), 2020.

[30] Y. Gong, S. Lazebnik, A. Gordo, and F. Perronnin, “Iterative quantization: A Procrustean approach to learning binary codes for large-scale image retrieval,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 35, no. 12, pp. 2916–2929, 2013.

[31] J. Gao and C. Long, “RaBitQ: Quantizing highdimensional vectors with a theoretical error bound for approximate nearest neighbor search,” in ACM SIGMOD International Conference on Management of Data, 2024.

[32] S. Morozov and A. Babenko, “Unsupervised neural quantization for compressed-domain similarity search,” in IEEE/CVF International Conference on Computer Vision (ICCV), 2019.

[33] J. Zhan et al., “Learning discrete representations via constrained clustering for effective and efficient dense retrieval,” in ACM International Conference on Web Search and Data Mining (WSDM), 2022.

[34] M. Aumüller, E. Bernhardsson, and A. Faithfull, “ANN Benchmarks: A benchmarking tool for approximate nearest neighbor algorithms,” Information Systems, vol. 87, p. 101374, 2020.

[35] L. Kuffo and P. Boncz, “Stop indexing at full precision: Revisiting clustering for vector embeddings,” in VLDB Workshop on Vector Databases, 2026, arXiv:2608.14648.

[36] D. S. Turaga, M. Vlachos, and O. Verscheure, “On kmeans cluster preservation using quantization schemes,” in IEEE International Conference on Data Mining, 2009.

[37] W. Lueks, B. Mokbel, M. Biehl, and B. Hammer, “How to evaluate dimensionality reduction? Improving the co-ranking matrix,” arXiv preprint arXiv:1110.3917, 2011.

[38] N. Kury, D. Kobak, and S. Damrich, “DREAMS: Preserving both local and global structure in dimensionality reduction,” Transactions on Machine Learning Research, 2026.

[39] S. Ghosh, C. Maitra, and R. K. De, “CMET: Clustering guided metric for quantifying embedding quality,” arXiv preprint arXiv:2507.04840, 2025.

[40] D. Kobak and P. Berens, “The art of using t-SNE for single-cell transcriptomics,” Nature Communications, vol. 10, p. 5416, 2019.

[41] Y. Wang, H. Huang, C. Rudin, and Y. Shaposhnik, “Understanding how dimension reduction tools work: An empirical approach to deciphering t-SNE, UMAP, TriMAP, and PaCMAP for data visualization,” Journal of Machine Learning Research, vol. 22, no. 201, pp. 1–73, 2021.

[42] D. Wu, H. Wang, W. Yu, Y. Zhang, K.-W. Chang, and D. Yu, “LongMemEval: Benchmarking chat assistants on long-term interactive memory,” in International Conference on Learning Representations, 2025.

[43] D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody, S. Truitt, and J. Larson, “From local to global: A graph RAG approach to query-focused summarization,” arXiv preprint arXiv:2404.16130, 2024.

[44] L. McInnes, J. Healy, and J. Melville, “UMAP: Uniform manifold approximation and projection for dimension reduction,” arXiv:1802.03426, 2018.

[45] F. Faghri, P. K. A. Vasu, C. Koc, V. Shankar, A. Toshev, O. Tuzel, and H. Pouransari, “MobileCLIP2: Improving multi-modal reinforced training,” Transactions on Machine Learning Research, 2025.

[46] H. Han, J. Y.-L. Forrest, J. Wang, S. Yuan, F. Han, and D. Li, “Explainable machine learning for high frequency trading dynamics discovery,” Information Sciences, vol. 684, p. 121286, 2024.

[47] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in International Conference on Learning Representations (ICLR), 2015.

[48] K. Lang, “NewsWeeder: Learning to filter netnews,” in Proceedings of the Twelfth International Conference on

Machine Learning, 1995, pp. 331–339.

[49] X. Zhang, J. Zhao, and Y. LeCun, “Character-level convolutional networks for text classification,” in Advances in Neural Information Processing Systems, vol. 28, 2015, pp. 649–657.

[50] P. Bajaj, D. Campos, N. Craswell, L. Deng, J. Gao, X. Liu, R. Majumder, A. McNamara, B. Mitra, T. Nguyen, M. Rosenberg, X. Song, A. Stoica, S. Tiwary, and T. Wang, “MS MARCO: A human generated MAchine reading COmprehension dataset,” arXiv preprint arXiv:1611.09268, 2016.

[51] L. Wang, N. Yang, X. Huang, B. Jiao, L. Yang, D. Jiang, R. Majumder, and F. Wei, “Text embeddings by weaklysupervised contrastive pre-training,” arXiv preprint arXiv:2212.03533, 2022.

[52] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov, “RoBERTa: A robustly optimized BERT pretraining approach,” arXiv preprint arXiv:1907.11692, 2019.

[53] J. Pennington, R. Socher, and C. D. Manning, “GloVe: Global vectors for word representation,” in Conference on Empirical Methods in Natural Language Processing, 2014, pp. 1532–1543.

[54] H. Han, L. Niu, D. Li, and J. Wang, “Differential rank preservation manifold learning,” 2026, under review at Pattern Recognition. [Online]. Available: http://arxiv.org/abs/2610.41107

[55] K. R. Moon et al., “Visualizing structure and transitions in high-dimensional biological data,” Nature Biotechnology, vol. 37, pp. 1482–1492, 2019.

[56] J. B. Tenenbaum, V. de Silva, and J. C. Langford, “A global geometric framework for nonlinear dimensionality reduction,” Science, vol. 290, pp. 2319–2323, 2000.

[57] X. Han, “Nonnegative principal component analysis for cancer molecular pattern discovery,” IEEE/ACM Transactions on Computational Biology and Bioinformatics, vol. 7, no. 3, pp. 537–549, 2010.

[58] L. Hubert and P. Arabie, “Comparing partitions,” Journal of Classification, vol. 2, no. 1, pp. 193–218, 1985.

[59] M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy, P.-E. Mazaré, M. Lomeli, L. Hosseini, and H. Jégou, “The Faiss library,” arXiv:2401.08281, 2024.

[60] W. F. Sheppard, “On the application of the theory of error to cases of normal distribution and normal correlation,” Philosophical Transactions of the Royal Society A, vol. 192, pp. 101–167, 1899.

[61] W. H. Kruskal, “Ordinal measures of association,” Journal of the American Statistical Association, vol. 53, no. 284, pp. 814–861, 1958.

## APPENDIX A PROOFS

This appendix proves the results of Sections 3 and 4, except Theorem 1, whose proof is in Section 3.4.

Proof of Proposition 1. (i) Each of the k members of $\mathcal { N } _ { k } ^ { x } ( i )$ lies in a uniform k-subset of N items with probability $k / N _ { \parallel }$ , so $\mathbb { E } | \dot { \mathcal { N } } _ { k } ^ { x } \dot { ( i ) } \cap \dot { \mathcal { N } } _ { k } ^ { z } ( i ) | = k ^ { 2 } / N$ . Substitution in (2) gives 0. (ii) A rotation, reflection, or translation keeps every distance, and a uniform scaling multiplies every distance by one positive constant. A strictly increasing transformation of a distance keeps its order. In each case every ranking $r _ { i } ^ { z }$ , and hence every $\mathcal { N } _ { k } ^ { z } ( i )$ , is unchanged. □

Proof of Proposition 2. Inclusion–exclusion gives the identity. $\mathrm { A t } 2 k ^ { \star } = N$ it reads $| \mathcal { F } ^ { x } \cap \mathcal { F } ^ { z } | = | \mathcal { N } ^ { x } \cap \mathcal { N } ^ { z } |$ , so item i has $2 | \mathcal { N } _ { k ^ { \star } } ^ { x } ( i ) \cap ^ { ^ { \cdot } } \mathcal { N } _ { k ^ { \star } } ^ { z } ( i )$ | agreements out of N and $A = \mathrm { D L P } _ { k ^ { \star } }$ . Equation (2) with $N - k ^ { \star } = k ^ { \star }$ gives $\nu _ { k ^ { \star } } = 2 \mathrm { D L P } _ { k ^ { \star } } - 1$ □

Proof of Proposition 3. The transformations map both medians to 0. For a bivariate normal pair with unit variances and correlation $\gamma ,$ Sheppard’s formula [60] gives Pr(both coordinates on the same side of $\begin{array} { r } { 0 ) = \frac { 1 } { 2 } \dot { + } \frac { 1 } { \pi } } \end{array}$ arcsin γ, so $\mathbb { E } ( 2 A _ { i } - 1 ) =$ $\frac { 2 } { \pi }$ arcsin $\gamma _ { i }$ . Kendall’s τ and Spearman’s $\rho$ of the bivariate normal take the forms above [61], and both increase in γ. □

Proof of Corollary 1. A violated pair has $\Delta ^ { z } \leq 0 , \mathsf { s o } h \geq \eta \Delta ^ { x }$ . If also $\Delta ^ { x } \geq \delta ,$ , then $h \geq \eta \delta$ . Hence ⊮[violated] $\leq | \mathcal { H } | \Delta ^ { x } <$ $\delta ] + \dot { h _ { ☉ } } / ( \eta \delta )$ , and taking expectations gives (7). Rankings by squared distance equal rankings by distance, so $\Delta ^ { z } \leq 0$ for every violated pair, with equality when two compressed distances tie. □

Proof of Proposition 4. ${ \bf A s } n  \infty$ , the k nearest items of i are those with $u _ { j } < z _ { \alpha . }$ , so a cross pair $( p , q )$ has $v = u _ { p } < z _ { \alpha } < u =$ $u _ { q } ,$ with density $\phi ( u ) \phi ( v ) / ( \alpha ( 1 - \alpha ) )$ on this region. The pair is violated when $\tilde { D } _ { q } < \tilde { D } _ { p } .$ , that is, when u $\iota + e _ { q } / s _ { i } < v + e _ { p } / s _ { i }$ The difference $( e _ { p } - e _ { q } ) / s _ { i }$ is normal with variance $2 \omega _ { i } ^ { 2 }$ , so this happens with probability $\Phi ( - ( u - v ) / ( \sqrt { 2 } \bar { \omega } _ { i } ) )$ ). Integrating over the region gives (8). For small $\omega ,$ write $u = z _ { \alpha } + x$ and $v = z _ { \alpha } - y$ with $x , y \geq 0$ . The factor $\Phi ( - ( x + y ) / ( { \sqrt { 2 } } \omega ) )$ confines the integral to $x + y = O ( \omega )$ , where $\phi ( u ) \phi ( v ) = \phi ( z _ { \alpha } ) ^ { 2 } ( 1 + O ( x + y ) )$ . With $t = x + y .$ , the area element of $\{ x , y \ge 0 , x + y \in d t \}$ is $t d t ,$ , and

$$
\int _ { 0 } ^ { \infty } t \Phi \Bigl ( - \frac { t } { \sqrt { 2 } \omega } \Bigr ) d t = 2 \omega ^ { 2 } \int _ { 0 } ^ { \infty } s \Phi ( - s ) d s = \frac { \omega ^ { 2 } } { 2 } ,
$$

since $\begin{array} { r } { \int _ { 0 } ^ { \infty } s \Phi ( - s ) d s = \frac { 1 } { 2 } \int _ { 0 } ^ { \infty } s ^ { 2 } \phi ( s ) d s = \frac { 1 } { 4 } } \end{array}$ by parts. The $O ( x + y )$ term contributes $O ( \omega ^ { 3 } )$

Proof of Corollary 2. Write $\epsilon _ { j } = e _ { j } / s _ { i }$ . The compressed distances rank the items as $u _ { j } + \epsilon _ { j }$ does, and $y _ { j } = ( u _ { j } + \epsilon _ { j } ) / \sqrt { 1 + \omega _ { i } ^ { 2 } }$ is $\mathcal { N } ( 0 , 1 )$ with correlation $\varrho _ { i }$ to $u _ { j } . \operatorname { A s } n \to \infty ,$ the k nearest items of i are those with $u _ { j } < z _ { \alpha }$ before compression and those with $y _ { j } < z _ { \alpha }$ after it, so the share of the k nearest items that both keep converges to $\Phi _ { 2 } \big ( z _ { \alpha } , z _ { \alpha } ; \varrho _ { i } \big ) / \alpha$ . Substituting this share for $\mathrm { D L P } _ { k } \mathrm { i n } ( 2 ) .$ , with $k / N \to \alpha , \mathrm { g i v e s } \left( 9 \right)$ . Plackett’s identity $\partial \Phi _ { 2 } ( z , \bar { z } ; \varrho ) / \partial \varrho = \phi _ { 2 } ( z , z ; \varrho ) > 0 ,$ with $\phi _ { 2 }$ the bivariate normal density, shows that ν increases strictly in $\varrho _ { i }$ , which decreases strictly in $\omega _ { i } . \mathrm { A t } \alpha = 1 / 2 , z _ { \alpha } = 0$ and Sheppard’s formula gives $\begin{array} { r } { \bar { \Phi } _ { 2 } ( 0 , 0 ; \varrho ) = \frac { 1 } { 4 } + \frac { 1 } { 2 \pi } } \end{array}$ arcsin $\varrho ,$ so $\textstyle \nu = { \frac { 2 } { \pi } }$ arcsin $\varrho _ { i }$ . Finally, $1 - \nu = ( \alpha - \Phi _ { 2 } ) / ( \alpha ( 1 - \alpha ) )$ , and $\alpha - \Phi _ { 2 }$ is the probability that $u _ { j } < z _ { \alpha } < u _ { j } + \epsilon _ { j }$ after rescaling. $\mathbf { A } \mathbf { s } \ \omega _ { i }  0$ , this probability is $\phi ( z _ { \alpha } ) \mathbb { E } [ \epsilon _ { j } ] _ { + } + o ( \omega _ { i } ) = \phi ( z _ { \alpha } ) \omega _ { i } / \sqrt { 2 \pi } + o ( \omega _ { i } )$ , so $1 - \nu = \phi ( z _ { \alpha } ) \omega _ { i } / \big ( \sqrt { 2 \pi } \alpha ( 1 - \alpha ) \big ) + o ( \omega _ { i } )$ , which is $0 . 6 4 \omega _ { i }$ at $\alpha = 0 . 4$ □

Proof of Proposition 5. Because no subvector is equidistant from its two nearest codewords, the assignments of $C _ { 0 }$ stay unchanged for small $\epsilon ,$ and every decoded vector is then a linear function of the codebooks. The reconstruction error is $\begin{array} { r } { \mathcal { L } _ { \mathrm { r e c } } ( C ) { } = \frac { 1 } { n } \sum _ { \ell , j } \sum _ { i : a _ { \ell } ( i ) = j } \| y _ { i , \ell } - c _ { \ell , j } \| ^ { 2 } } \end{array}$ , with $a _ { \ell } ( i )$ the assignment, a quadratic in C. Its gradient with respect to $c _ { \ell , j }$ is $\begin{array} { r } { \frac { 2 } { n } \sum _ { i : a _ { \ell } ( i ) = j } ( c _ { \ell , j } - y _ { i , \ell } ) . } \end{array}$ , which is zero when $c _ { \ell , j }$ is the mean of its subvectors, and its Hessian is $\frac { 2 n _ { \ell j } } { n }$ times the identity on $c \ell , j \cdot \mathrm { A }$ second-order expansion along −h, which has no remainder for a quadratic, gives the first line of (12). The second line is the first-order expansion of ${ \mathcal L } _ { \mathrm { f a r } }$ along −h, whose directional derivative is $- \langle g , h \rangle$ □

## APPENDIX B

## LAQ RECIPE

Every LAQ run uses one recipe: $M = d / 8$ subspaces of 8 dimensions with 256 codewords each, the full PCA basis, 1,000 steps, 1,024 anchors per scale group and step, $\lambda _ { \mathrm { m i d } } = \lambda _ { \mathrm { f a r } } = 1$ , margin $\eta = 1$ , Adam with a learning rate of 0.002 times the standard deviation of $C _ { 0 }$ and cosine decay, temperature from $T _ { 0 } \mathrm { t o } 0 . 0 5 T _ { 0 } .$ , a monitor set of 2,000 training items checked every 250 steps, and a reconstruction limit of 1.15 for a checkpoint. The scale normalizers $s _ { \mathrm { m i d } }$ and $s _ { \mathrm { f a r } }$ are medians of $^ { 1 6 , 3 8 4 }$ uncompressed gaps sampled at the start.

## APPENDIX C

## ADDITIONAL MEASUREMENT RESULTS

The figures and tables of this appendix give every run behind the main figures. Figure 5 places the 27 compressions of the two agent memories on the $( \grave { L } , \grave { G } )$ plane for the retrieval-tuned encoders, and Figure 6 for the three encoders of other kinds. Figures 7 and 8 relate each operation to L and G, and Figure 9 gives rescoring. The topic corpora repeat the pattern of the agent memories: Figure 10 places their compressions on the plane and breaks the 96-byte comparison of FinQA down by company. Figure 11 gives the seven compressions at d/8 bytes. Tables 4–8 give the nonlinear maps, the correlations behind Figures $\mathsf { \check { 7 } }$ and 8, rescoring for every code family that keeps Euclidean distances, and every compression of the financial memory.

![](images/ea44914f375f0705523c2316e569f7f712c81317b9b50a19bb33e8c79388a094.jpg)

![](images/8ecaa803e852f3d0ae3a3ecc0a7422ae46e62b41a8573b1869bc1060fb6d1d16.jpg)

![](images/d6af0daf5503c4047a4409d6cd5dc004d8fc756197f83058842c33bec1c47339.jpg)

![](images/a667a404dd2b26742b5d1033ca0a5b529934c74514ec384bf6af8d6b4b1dcd55.jpg)

![](images/2820fd0c9f89f82a7f38ed2167f5d04d9cb74ddcbe172e5e1228949375a11288.jpg)

![](images/4e9264222d5e1406728cfa62e3b23e7d7c9eeff334fd4d0c76343ef5a9dc3516.jpg)  
compression half precision scalar quantization binary (1, 2, 4 bit) binary + rescoring PCA + rotation + binary truncation PCA PCA + int8 random projection product quantization UMAP  
Fig. 5. Compressions of one memory spread off a single curve, and rescored binary codes lie furthest below the diagonal, by 0.21 to 0.41. L and G of 27 compressions of the two agent memories, three retrieval-tuned encoders; Figure 6 gives the other three. Dashed line: G = L; shaded: less of the far end kept than of the near end.

PCA + int8

(a) MobileCLIP2  
![](images/d71f8d40eaf1dbd899222d970b92edcef03f081235c657f90b19eeee52098e4a.jpg)

(b) RoBERTa  
![](images/53dd91e855a58b056f4b7175655d5a344ce1398c7e291dbdabbb0159afc75590.jpg)

(c) GloVe  
![](images/95df3d4469e647f70f6fba8f413a2dc07d2a2994dbba9e9a114af02faede698d.jpg)  
compression  
◆ PCA + rotation + binary

(d) MobileCLIP2  
![](images/12468ee6965043ed4cda742d108e92696972c4e0df28813d0266f33d855d8170.jpg)  
local fidelity L

(e) RoBERTa  
![](images/ccff1c4de0f90d164a88bcc85d8dcdc830c96f0bb1e9fff38cf4e066c6e6c7a6.jpg)  
local fidelity L

(f) GloVe  
![](images/70fbdbec0f11347415fef5b2a7ed86c9991993efa22765dd0e9a922a25aff0c8.jpg)  
truncation

local fidelity L  
Fig. 6. With encoders of other kinds, PCA and product quantization lie on or above the diagonal, and one-bit codes of GloVe and RoBERTa keep more of the far end than of the near end. L and G of 27 compressions of the two agent memories; dashed line: G = L.  
(a) retrieval vs L  
![](images/b446446a1fb689c8d0ecbd22c45f47078d3f3f645ffd4b960b71aa7cf0f733d2.jpg)

(b) retrieval vs G  
![](images/1b25b2ea538a3d9ddfc7368a9bbb6a161cc6fd11455009acf1bb067ea044e062.jpg)

(c) far-end ops vs L  
![](images/f0311dd8b98c7439bc2bf665d7f341f0324198ff2c87cfaca293bf0e1977bacd.jpg)

(d) far-end ops vs G  
![](images/2bdb4c589d2a3980a5fe4240f3d1b8bafb21424fd44873202a8b06e5e873c3d0.jpg)  
FinQA LoCoMo 20 Newsgroups AG News BGE-base E5-base BGE-large MobileCLIP2 RoBERTa GloVe

Fig. 7. Retrieval follows L (ρ 0.99, against 0.89 with G), and the far-end operations follow G (ρ 0.95, against 0.87 with L). Agreement with the uncompressed memory, every compression and run. (a, b) Retrieval on the two agent memories. (c, d) Consolidation into themes for FinQA, deletion for LoCoMo and the topic corpora. ρ: median over runs of the Spearman correlation across the 27 compressions of a run.

(d) wrong fiscal year

![](images/fa52fb6d23747d63e80fd0eaa3ba9951dbddb84918e1f246ed1c58bd54bb5b62.jpg)

![](images/bbf8a9f993ceb153ee9ebf6d9981eec20e2b819240f99a22717a7ff745b49e76.jpg)

(c) retrieval-tuned encoders  
![](images/f782e788e15845c0e0a8609e3a76cba5b3269ae0756e6ecb9b6be232b02bdbf4.jpg)  
retrieval (near end) maps of sessions, companies, sectors, topics (c) L given G BGE-base BGE-large RoBERTa consolidation and deletion peer groups and barrier (c) G given L E5-base MobileCLIP2 GloVe

Fig. 8. Each operation follows the end at which it compares items: retrieval lies in the L region, consolidation and deletion in the G region. Partia rank correlation with G given L against that with L given G, across the 27 compressions of each run. (a) Agent memories, (b) topic corpora; color: operation; shape: encoder. (c) Medians and ranges over the six runs of the retrieval-tuned encoders.  
![](images/79bec0346fda1cc43c8b124ed242274bd483cd6438496648e532a9b78d15a635.jpg)

![](images/92ce8413c902cfbc29e014fcfcfe0f475dd92266383e1b86aec82c39186ce9e0.jpg)

(c) evidence recall  
![](images/1e6c2a8c7be595cad0f922750552d25919a4835782d28697be466d63562a8118.jpg)

![](images/b7dac0f83ba3dbd8e708fb0ec2fac33cf528b44fc498cb48e4a40576eca8c532.jpg)  
Fig. 9. Rescoring raises L and retrieval and leaves G and the far-end operations at the values of the binary codes. (a, b) Binary codes before and after full-precision rescoring of the 40 nearest codes; arrows run from binary to rescored, and a marker without an arrow is unchanged. (c) Evidence recall@10 on FinQA rises with L; dotted lines: uncompressed memories. (d) The share of right-company passages from another fiscal year stays near its uncompressed level.

(a) MobileCLIP2  
![](images/d67c8213b3ac0c7a4324a560453dab5f318d26bbe462210377581fc1520ff4f3.jpg)

(b) RoBERTa  
![](images/aaa0a6bca9a0444f30544a2e2677f28a33977b48d0259f267ee20c6a06d2ac2e.jpg)  
(c) GloVe

(d) MobileCLIP2  
(e) RoBERTa  
![](images/50da4e3eef36e89a35834802c869a280a1993c33e48bd4c9c076b24302af8e0b.jpg)

![](images/5280374a4726f246db228a9d142507c7a7ee1b33d699f015ac1d04839f917f2d.jpg)

![](images/84253bbc7fd04052e025751aff3eab73be9048be594ec474a73bdadf52783b89.jpg)

(f) GloVe  
![](images/12324fc1b8475604729610f7e22b2fcf24c47e4bb78c9d2198c6cd2b44a4346f.jpg)

(g) FinQA, BGE-base, 96 B, by company  
![](images/bf78de6133d0289ff44e7fcf19d000bf6a614dd0474a992ec044fab8ce67f5aa.jpg)

Fig. 10. On the topic corpora, PCA and product quantization lie on or above the diagonal (a–f). On FinQA, rescored binary codes keep neighbors yet move up to 39 percent of a company’s passages to another theme (g). (a–f) L and G of 27 compressions, three encoders of other kinds; dotted line: G = L. (g) The two 96-byte compressions of FinQA (BGE-base) by company; diamonds: means; dotted line: 20 percent moved.

(a) FinQA, BGE-base, 96 B  
![](images/862128b4f9b8a23210b4db65cd2b97e539d4f89f869a46922735943c1299df26.jpg)

(b) FinQA, E5-base, 96 B  
![](images/08ec8ca077ddb2d26773c5f6df71826fabd7891ada29d121c08421e674a45bfe.jpg)

(c) FinQA, BGE-large, 128 B  
![](images/806bd71709f151c037aef66c055b5e606db5e09d41431b6251edc40e74e640c3.jpg)

(d) LoCoMo, BGE-base, 96 B  
![](images/e1bc6d4acb620c646afea110310ebacb7527874d1ac75cb940749f1abf7604b0.jpg)

(e) LoCoMo, E5-base, 96 B  
![](images/e54f5afafb8cac420ef46748a31ebc1d5c9cffb177e64f648d3b37242f00da19.jpg)

(f) LoCoMo, BGE-large, 128 B  
![](images/3d865f8ddd28fff2342f152300025e66cbd09a0b1ccc5cc85676ed59a71062be.jpg)  
Fig. 11. At d/8 bytes, OPQ keeps the highest G of seven compressions, and rescored binary codes keep a high L with a lower G. Fidelity and agreement with the uncompressed memory, retrieval-tuned encoders; (a–c) FinQA, (d–f) LoCoMo.

![](images/27257ec32fb79cb28d02dca15fe182301f5f5ca42ada025baaa52c5838272668.jpg)  
Fig. 12. The normal model of Proposition 4 holds in the G band and predicts its violations and its loss of profile. All 52 held-out runs; open markers, OPQ start; filled, LAQ. (a) Median error of each anchor’s standardized distance quantile against the normal quantile, in standard deviations, at the near end and in the G band. (b) The full integral (8) against the measured violation rate at $k = 0 . 4 n .$ . (c) The leading term at $k = 1 0$ against the measured rate. (d) Measured 1 − ν at k = 0.4n against the mean relative distance noise ω; solid line, Corollary 2; dashed line, its leading term 0.64 ω (Appendix A).

TABLE 4  
Neighborhood-embedding maps keep G at 0.22 or less, and nonnegative PCA keeps more of the far end than of the near end. Maps fitted to the memory (BGE-base), kept in full precision; PCA to 16 dimensions is the linear reference for NPCA. Queries are mapped by the fitted transform (UMAP [44], PHATE [55], Isomap [56], NPCA [57], PCA) or placed at the inverse-distance-weighted mean of the positions of their 10 nearest stored items (t-SNE [13], SR-tSNE [54]). Top-10 agreement: overlap of the top-10 lists. Last column: theme agreement (FinQA) or agreement of session deletion (LoCoMo).
<table><tr><td>Memory FinQA</td><td>Map</td><td>Bytes</td><td>L</td><td>G</td><td>Retrieval</td><td>Themes / deletion</td></tr><tr><td></td><td>UMAP, 16 dims</td><td>64</td><td>0.523</td><td>0.10</td><td>0.24</td><td>0.08</td></tr><tr><td></td><td>PCA to 50, UMAP to 16</td><td>64</td><td>0.474</td><td>0.18</td><td>0.16</td><td>0.21</td></tr><tr><td></td><td>PHATE, 16 dims</td><td>64</td><td>0.537</td><td>0.15</td><td>0.19</td><td>0.08</td></tr><tr><td></td><td>UMAP, 2 dims</td><td>8</td><td>0.453</td><td>0.07</td><td>0.18</td><td>0.07</td></tr><tr><td></td><td>PHATE, 2 dims</td><td>8</td><td>0.231</td><td>0.12</td><td>0.04</td><td>0.07</td></tr><tr><td></td><td>t-SNE, 2 dims</td><td>8</td><td>0.585</td><td>0.13</td><td>0.07</td><td>0.09</td></tr><tr><td></td><td>SR-tSNE, 2 dims</td><td>8</td><td>0.590</td><td>0.13</td><td>0.07</td><td>0.09</td></tr><tr><td></td><td>Isomap, 16 dims</td><td>64</td><td>0.560</td><td>0.25</td><td>0.28</td><td>0.15</td></tr><tr><td></td><td>NPCA, 16 dims</td><td>64</td><td>0.237</td><td>0.43</td><td>0.10</td><td>0.41</td></tr><tr><td>LoCoMo</td><td>PCA, 16 dims (linear)</td><td>64</td><td>0.349</td><td>0.54</td><td>0.17</td><td>0.71</td></tr><tr><td></td><td>UMAP, 16 dims</td><td>64</td><td>0.362</td><td>0.19</td><td>0.30</td><td>0.08</td></tr><tr><td></td><td>PCA to 50, UMAP to 16</td><td>64</td><td>0.329</td><td>0.19</td><td>0.25</td><td>0.07</td></tr><tr><td></td><td>PHATE, 16 dims</td><td>64</td><td>0.292</td><td>0.22</td><td>0.24</td><td>0.08</td></tr><tr><td></td><td>UMAP, 2 dims</td><td>8</td><td>0.255</td><td>0.16</td><td>0.21</td><td>0.04</td></tr><tr><td></td><td>PHATE, 2 dims</td><td>8</td><td>0.141</td><td>0.19</td><td>0.14</td><td>0.05</td></tr><tr><td></td><td>t-SNE, 2 dims</td><td>8</td><td>0.388</td><td>0.20</td><td>0.15</td><td>0.05</td></tr><tr><td></td><td>SR-tSNE, 2 dims</td><td>8</td><td>0.392</td><td>0.20</td><td>0.15</td><td>0.05</td></tr><tr><td></td><td>Isomap, 16 dims</td><td>64</td><td>0.388</td><td>0.43</td><td>0.33</td><td>0.23</td></tr><tr><td></td><td>NPCA, 16 dims</td><td>64</td><td>0.220</td><td>0.44</td><td>0.18</td><td>0.11</td></tr><tr><td></td><td>PCA, 16 dims (linear)</td><td>64</td><td>0.349</td><td>0.54</td><td>0.26</td><td>0.14</td></tr></table>

TABLE 5  
Retrieval follows L, and consolidation and deletion follow G. Spearman correlation across 27 compressions of each operation’s agreement with the uncompressed memory (label precision for the topic corpora) with L and with G, and the partial rank correlation with each end given the other (L | G: with L given G; G | L: with G given L). Agent memories, FinQA.
<table><tr><td>Memory, encoder</td><td>Operation</td><td>End</td><td>L</td><td>G</td><td>L|G</td><td>G |L</td></tr><tr><td>FinQA, BGE-base</td><td>same top-10 retrieved</td><td>near</td><td>1.00</td><td>0.89</td><td>0.99</td><td>0.30</td></tr><tr><td></td><td>same themes (ARI)</td><td>far</td><td>0.80</td><td>0.90</td><td>0.04</td><td>0.68</td></tr><tr><td></td><td>same peer groups</td><td>far</td><td>0.87</td><td>0.83</td><td>0.53</td><td>0.26</td></tr><tr><td></td><td>information barrier</td><td>far</td><td>0.91</td><td>0.84</td><td>0.65</td><td>0.21</td></tr><tr><td></td><td>company map</td><td>far</td><td>0.91</td><td>0.86</td><td>0.64</td><td>0.26</td></tr><tr><td></td><td>sector map</td><td>far</td><td>0.86</td><td>0.84</td><td>0.45</td><td>0.35</td></tr><tr><td>FinQA, E5-base</td><td>same top-10 retrieved</td><td>near</td><td>0.98</td><td>0.94</td><td>0.94</td><td>0.78</td></tr><tr><td></td><td>same themes (ARI)</td><td>far</td><td>0.83</td><td>0.93</td><td>0.01</td><td>0.76</td></tr><tr><td></td><td>same peer groups</td><td>far</td><td>0.89</td><td>0.90</td><td>0.45</td><td>0.51</td></tr><tr><td></td><td>information barrier</td><td>far</td><td>0.88</td><td>0.88</td><td>0.44</td><td>0.44</td></tr><tr><td></td><td>company map</td><td>far</td><td>0.90</td><td>0.89</td><td>0.50</td><td>0.45</td></tr><tr><td>FinQA, BGE-large</td><td>sector map</td><td>far</td><td>0.89</td><td>0.87</td><td>0.52</td><td>0.37</td></tr><tr><td></td><td>same top-10 retrieved</td><td>near</td><td>1.00</td><td>0.92</td><td>0.97</td><td>0.31</td></tr><tr><td></td><td>same themes (ARI)</td><td>far</td><td>0.83</td><td>0.91</td><td>-0.02</td><td>0.66</td></tr><tr><td></td><td>same peer groups</td><td>far</td><td>0.88</td><td>0.86</td><td>0.43</td><td>0.30</td></tr><tr><td></td><td>information barrier</td><td>far</td><td>0.90</td><td>0.88</td><td>0.48</td><td>0.32</td></tr><tr><td></td><td>company map</td><td>far</td><td>0.94</td><td>0.91</td><td>0.63</td><td>0.34</td></tr><tr><td>FinQA, MobileCLIP2</td><td>sector map</td><td>far</td><td>0.87</td><td>0.86</td><td>0.39</td><td>0.32</td></tr><tr><td></td><td>same top-10 retrieved</td><td>near</td><td>0.95</td><td>0.92</td><td>0.83</td><td>0.70</td></tr><tr><td></td><td>same themes (ARI)</td><td>far</td><td>0.77</td><td>0.90</td><td>0.02</td><td>0.74</td></tr><tr><td></td><td>same peer groups</td><td>far</td><td>0.88</td><td>0.90</td><td>0.49</td><td>0.63</td></tr><tr><td></td><td>information barrier</td><td>far</td><td>0.88</td><td>0.88</td><td>0.55</td><td>0.54</td></tr><tr><td></td><td>company map</td><td>far</td><td>0.88</td><td>0.90</td><td>0.51</td><td>0.61</td></tr><tr><td>FinQA, RoBERTa</td><td>sector map same top-10 retrieved</td><td>far</td><td>0.73</td><td>0.83</td><td>0.11</td><td>0.57</td></tr><tr><td></td><td>same themes (ARI)</td><td>near far</td><td>0.98 0.84</td><td>0.88</td><td>0.93 0.25</td><td>-0.24 0.43</td></tr><tr><td></td><td></td><td>far</td><td>0.91</td><td>0.86 0.85</td><td>0.63</td><td>0.10</td></tr><tr><td></td><td>same peer groups information barrier</td><td>far</td><td>0.91</td><td>0.99</td><td>0.14</td><td>0.92</td></tr><tr><td></td><td>company map</td><td>far</td><td>0.95</td><td>0.93</td><td>0.67</td><td>0.46</td></tr><tr><td></td><td>sector map</td><td>far</td><td>0.92</td><td>0.87</td><td>0.64</td><td>0.15</td></tr><tr><td>FinQA, GloVe</td><td></td><td></td><td></td><td></td><td></td><td>0.35</td></tr><tr><td></td><td>same top-10 retrieved same themes (ARI)</td><td>near</td><td>0.99</td><td>0.83</td><td>0.97</td><td>0.63</td></tr><tr><td></td><td>same peer groups</td><td>far far</td><td>0.77 0.90</td><td>0.86 0.72</td><td>0.26 0.77</td><td>-0.01</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>0.81</td></tr><tr><td></td><td>information barrier</td><td>far</td><td>0.92</td><td>0.93</td><td>0.78</td><td></td></tr><tr><td></td><td>company map</td><td>far</td><td>0.89</td><td>0.94</td><td>0.66</td><td>0.84</td></tr><tr><td></td><td>sector map</td><td>far</td><td>0.90</td><td>0.76</td><td>0.74</td><td>0.12</td></tr></table>

TABLE 6  
The same split holds on LoCoMo and the topic corpora. Continuation of Table 5.
<table><tr><td>Memory, encoder</td><td>Operation</td><td>End</td><td>L</td><td>G</td><td>L|G</td><td>G|L</td></tr><tr><td rowspan="3">LoCoMo, BGE-base</td><td>same top-10 retrieved</td><td>near</td><td>0.99</td><td>0.89</td><td>0.95</td><td>-0.16</td></tr><tr><td>same memories deleted</td><td>far</td><td>0.87</td><td>0.97</td><td>-0.11</td><td>0.86</td></tr><tr><td>session map</td><td>far</td><td>0.89</td><td>0.97</td><td>0.10</td><td>0.82</td></tr><tr><td rowspan="3">LoCoMo, E5-base</td><td>same top-10 retrieved</td><td>near</td><td>0.99</td><td>0.89</td><td>0.96</td><td>0.18</td></tr><tr><td>same memories deleted</td><td>far</td><td>0.87</td><td>0.90</td><td>0.36</td><td>0.55</td></tr><tr><td>session map</td><td>far</td><td>0.90</td><td>0.98</td><td>0.36</td><td>0.92</td></tr><tr><td rowspan="3">LoCoMo, BGE-large</td><td>same top-10 retrieved</td><td>near</td><td>0.99</td><td>0.90</td><td>0.97</td><td>-0.35</td></tr><tr><td>same memories deleted</td><td>far</td><td>0.90</td><td>0.99</td><td>-0.14</td><td>0.93</td></tr><tr><td>session map</td><td>far</td><td>0.91</td><td>0.94</td><td>0.30</td><td>0.66</td></tr><tr><td rowspan="3">LoCoMo, MobileCLIP2</td><td>same top-10 retrieved</td><td>near</td><td>1.00</td><td>0.88</td><td>0.99</td><td>-0.25</td></tr><tr><td>same memories deleted</td><td>far</td><td>0.88</td><td>0.96</td><td>0.19</td><td>0.84</td></tr><tr><td>session map</td><td>far</td><td>0.86</td><td>0.92</td><td>0.22</td><td>0.67</td></tr><tr><td rowspan="3">LoCoMo, RoBERTa</td><td>same top-10 retrieved</td><td>near</td><td>0.97</td><td>0.89</td><td>0.87</td><td>-0.36</td></tr><tr><td>same memories deleted</td><td>far</td><td>0.92</td><td>0.98</td><td>-0.03</td><td>0.83</td></tr><tr><td>session map</td><td>far</td><td>0.93</td><td>0.97</td><td>0.26</td><td>0.71</td></tr><tr><td rowspan="3">LoCoMo, GloVe</td><td>same top-10 retrieved</td><td>near</td><td>0.99</td><td>0.85</td><td>0.97</td><td>-0.57</td></tr><tr><td>same memories deleted</td><td>far</td><td>0.92</td><td>0.96</td><td>0.43</td><td>0.78</td></tr><tr><td>session map</td><td>far</td><td>0.90</td><td>0.99</td><td>0.20</td><td>0.93</td></tr><tr><td rowspan="4">20 Newsgroups, MobileCLIP2</td><td>label precision of 10 neighbors</td><td>near</td><td>0.59</td><td>0.29</td><td>0.68</td><td>-0.48</td></tr><tr><td>same memories deleted</td><td>far</td><td>0.90</td><td>0.95</td><td>0.60</td><td>0.80</td></tr><tr><td>same themes (ARI)</td><td>far</td><td>0.77</td><td>0.81</td><td>0.25</td><td>0.48</td></tr><tr><td>topic map</td><td>far</td><td>0.68</td><td>0.79</td><td>0.03</td><td>0.56</td></tr><tr><td rowspan="4">20 Newsgroups, RoBERTa</td><td>label precision of 10 neighbors</td><td>near</td><td>0.62</td><td>0.41</td><td>0.76</td><td>-0.65</td></tr><tr><td>same memories deleted</td><td>far</td><td>0.92</td><td>0.98</td><td>-0.06</td><td>0.85</td></tr><tr><td>same themes (ARI)</td><td>far</td><td>0.91</td><td>0.88</td><td>0.53</td><td>0.12</td></tr><tr><td>topic map</td><td>far</td><td>0.93</td><td>0.95</td><td>0.38</td><td>0.56</td></tr><tr><td rowspan="4">20 Newsgroups, GloVe</td><td>label precision of 10 neighbors</td><td>near</td><td>0.67</td><td>0.34</td><td>0.83</td><td>-0.71</td></tr><tr><td>same memories deleted</td><td>far</td><td>0.93</td><td>0.97</td><td>0.59</td><td>0.86</td></tr><tr><td>same themes (ARI)</td><td>far</td><td>0.87</td><td>0.89</td><td>0.40</td><td>0.52</td></tr><tr><td>topic map</td><td>far</td><td>0.80</td><td>0.88</td><td>0.09</td><td>0.61</td></tr><tr><td rowspan="3">AG News, MobileCLIP2</td><td>label precision of 10 neighbors</td><td>near</td><td>0.31</td><td>0.49</td><td>-0.26</td><td>0.47</td></tr><tr><td>same memories deleted</td><td>far</td><td>0.86</td><td>0.93</td><td>0.32</td><td>0.74</td></tr><tr><td>same themes (ARI)</td><td>far</td><td>0.73</td><td>0.83</td><td>0.03</td><td>0.59</td></tr><tr><td rowspan="3">AG News, RoBERTa</td><td>label precision of 10 neighbors</td><td>near</td><td>0.62</td><td>0.46</td><td>0.65</td><td>-0.51</td></tr><tr><td>same memories deleted</td><td>far</td><td>0.93</td><td>0.98</td><td>0.03</td><td>0.83</td></tr><tr><td>same themes (ARI)</td><td>far</td><td>0.86</td><td>0.85</td><td>0.32</td><td>0.21</td></tr><tr><td rowspan="3">AG News, GloVe</td><td>label precision of 10 neighbors</td><td>near</td><td>0.24</td><td>0.22</td><td>0.10</td><td>0.04</td></tr><tr><td>same memories deleted</td><td>far</td><td>0.86</td><td>0.95</td><td>0.42</td><td>0.85</td></tr><tr><td>same themes (ARI)</td><td>far</td><td>0.62</td><td>0.80</td><td>-0.14</td><td>0.65</td></tr></table>

TABLE 7  
Rescoring raises L and retrieval for every code family, and G and the far-end operations keep the values of the stored codes. Code families that keep Euclidean distances, d/8 bytes per item. The 40 nearest codes are rescored with the full-precision vectors; L and retrieval come from the rescored lists, G and consolidation or deletion from the stored codes. Top-10 agreement: overlap of the top-10 lists. Last column: theme agreement (FinQA) or agreement of session deletion (LoCoMo).
<table><tr><td>Memory</td><td>Encoder</td><td>Compression</td><td>L</td><td>G</td><td>Retrieval</td><td>Themes / deletion</td></tr><tr><td>FinQA</td><td>BGE-base (96 B)</td><td>PCA + int8</td><td>0.77</td><td>0.74</td><td>0.60</td><td>0.96</td></tr><tr><td></td><td></td><td>PCA + int8 + rescoring</td><td>0.99</td><td>0.74</td><td>0.91</td><td>0.96</td></tr><tr><td></td><td></td><td>PQ</td><td>0.82</td><td>0.73</td><td>0.68</td><td>0.88</td></tr><tr><td></td><td></td><td>PQ + rescoring</td><td>0.99</td><td>0.73</td><td>0.95</td><td>0.88</td></tr><tr><td></td><td></td><td>OPQ</td><td>0.88</td><td>0.83</td><td>0.78</td><td>0.93</td></tr><tr><td>FinQA</td><td></td><td>OPQ + rescoring</td><td>1.00</td><td>0.83</td><td>0.99</td><td>0.93</td></tr><tr><td></td><td>E5-base (96 B)</td><td>PCA + int8</td><td>0.77</td><td>0.73</td><td>0.57</td><td>0.96</td></tr><tr><td></td><td></td><td>PCA + int8 + rescoring</td><td>0.99</td><td>0.73</td><td>0.89</td><td>0.96</td></tr><tr><td></td><td></td><td>PQ</td><td>0.82</td><td>0.73</td><td>0.64</td><td>0.89</td></tr><tr><td></td><td></td><td>PQ + rescoring</td><td>1.00</td><td>0.73</td><td>0.94</td><td>0.89</td></tr><tr><td></td><td></td><td>OPQ</td><td>0.88</td><td>0.82</td><td>0.76</td><td>0.93</td></tr><tr><td>FinQA</td><td>BGE-large (128 B)</td><td>OPQ + rescoring</td><td>1.00</td><td>0.82</td><td>0.99</td><td>0.93</td></tr><tr><td></td><td></td><td>PCA + int8</td><td>0.82</td><td>0.77</td><td>0.68</td><td>0.97</td></tr><tr><td></td><td></td><td>PCA + int8 + rescoring</td><td>1.00</td><td>0.77</td><td>0.96</td><td>0.97</td></tr><tr><td></td><td></td><td>PQ</td><td>0.84</td><td>0.74</td><td>0.71</td><td>0.89</td></tr><tr><td></td><td></td><td>PQ + rescoring</td><td>1.00</td><td>0.74</td><td>0.97</td><td>0.89</td></tr><tr><td></td><td></td><td>OPQ OPQ + rescoring</td><td>0.91</td><td>0.86</td><td>0.83</td><td>0.94</td></tr><tr><td>LoCoMo</td><td>BGE-base (96 B)</td><td></td><td>1.00</td><td>0.86</td><td>1.00</td><td>0.94</td></tr><tr><td></td><td></td><td>PCA + int8</td><td>0.76</td><td>0.81</td><td>0.69</td><td>0.59</td></tr><tr><td></td><td></td><td>PCA + int8 + rescoring</td><td>1.00</td><td>0.81</td><td>0.99</td><td>0.59</td></tr><tr><td></td><td></td><td>PQ</td><td>0.78</td><td>0.79</td><td>0.70</td><td>0.65</td></tr><tr><td></td><td></td><td>PQ + rescoring</td><td>1.00</td><td>0.79</td><td>0.98</td><td>0.65</td></tr><tr><td></td><td></td><td>OPQ</td><td>0.87</td><td>0.89</td><td>0.81</td><td>0.79</td></tr><tr><td>LoCoMo</td><td>E5-base (96 B)</td><td>OPQ + rescoring</td><td>1.00</td><td>0.89</td><td>1.00</td><td>0.79</td></tr><tr><td></td><td></td><td>PCA + int8</td><td>0.72</td><td>0.77</td><td>0.63</td><td>0.43</td></tr><tr><td></td><td></td><td>PCA + int8 + rescoring</td><td>0.99</td><td>0.77</td><td>0.97</td><td>0.43</td></tr><tr><td></td><td></td><td>PQ PQ + rescoring</td><td>0.77 0.99</td><td>0.75 0.75</td><td>0.67 0.96</td><td>0.59</td></tr><tr><td></td><td></td><td>OPQ</td><td>0.85</td><td>0.85</td><td>0.78</td><td>0.59</td></tr><tr><td></td><td></td><td>OPQ + rescoring</td><td>1.00</td><td>0.85</td><td>0.99</td><td>0.70 0.70</td></tr><tr><td>LoCoMo</td><td>BGE-large (128 B)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>PCA + int8 PCA + int8 + rescoring</td><td>0.82 1.00</td><td>0.85 0.85</td><td>0.75 1.00</td><td>0.68 0.68</td></tr><tr><td></td><td></td><td>PQ</td><td>0.80</td><td>0.80</td><td>0.74</td><td>0.66</td></tr><tr><td></td><td></td><td>PQ + rescoring</td><td>1.00</td><td>0.80</td><td>0.98</td><td>0.66</td></tr><tr><td></td><td></td><td>OPQ</td><td>0.90</td><td>0.91</td><td>0.86</td><td>0.82</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>OPQ + rescoring</td><td>1.00</td><td>0.91</td><td>1.00</td><td>0.82</td></tr></table>

TABLE 8  
At equal bytes, compressions of the financial memory differ most in G and theme agreement. FinQA, BGE-base, 27 compressions: bytes per item, fidelity, and agreement with the uncompressed memory. Top-10 agreement: overlap of the top-10 lists for 863 questions. Evidence: evidence recall@10 (0.50 uncompressed). Themes: theme agreement over 11 themes. Peers: overlap of five-company peer groups. Barrier: Jaccard overlap of the memories removed to wall off a company. Map: rank correlation of distances among company centroids.
<table><tr><td>Compression</td><td>Bytes</td><td>L</td><td>G</td><td>Retrieval</td><td>Evidence</td><td>Themes</td><td>Peers</td><td>Barrier</td><td>Map</td></tr><tr><td>float16</td><td>1536</td><td>1.00</td><td>1.00</td><td>1.00</td><td>0.50</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td>int8</td><td>768</td><td>0.98</td><td>0.97</td><td>0.97</td><td>0.50</td><td>0.99</td><td>0.99</td><td>0.98</td><td>1.00</td></tr><tr><td>int4</td><td>384</td><td>0.96</td><td>0.95</td><td>0.94</td><td>0.50</td><td>0.99</td><td>0.99</td><td>0.97</td><td>1.00</td></tr><tr><td>binary, 1 bit</td><td>96</td><td>0.74</td><td>0.65</td><td>0.57</td><td>0.44</td><td>0.79</td><td>0.80</td><td>0.87</td><td>0.97</td></tr><tr><td>binary, 2 bit</td><td>192</td><td>0.83</td><td>0.65</td><td>0.70</td><td>0.48</td><td>0.90</td><td>0.83</td><td>0.89</td><td>0.97</td></tr><tr><td>binary, 4 bit</td><td>384</td><td>0.86</td><td>0.72</td><td>0.76</td><td>0.49</td><td>0.94</td><td>0.85</td><td>0.90</td><td>0.97</td></tr><tr><td>binary + rescoring</td><td>96</td><td>0.97</td><td>0.65</td><td>0.87</td><td>0.49</td><td>0.79</td><td>0.80</td><td>0.87</td><td>0.97</td></tr><tr><td>truncation 384</td><td>1536</td><td>0.86</td><td>0.76</td><td>0.74</td><td>0.49</td><td>0.77</td><td>0.75</td><td>0.89</td><td>0.96</td></tr><tr><td>truncation 192</td><td>768</td><td>0.76</td><td>0.62</td><td>0.58</td><td>0.46</td><td>0.62</td><td>0.59</td><td>0.81</td><td>0.90</td></tr><tr><td>truncation 96</td><td>384</td><td>0.64</td><td>0.47</td><td>0.40</td><td>0.33</td><td>0.44</td><td>0.46</td><td>0.66</td><td>0.80</td></tr><tr><td>truncation 48</td><td>192</td><td>0.47</td><td>0.33</td><td>0.22</td><td>0.21</td><td>0.22</td><td>0.31</td><td>0.49</td><td>0.64</td></tr><tr><td>PCA 384</td><td>1536</td><td>0.97</td><td>0.97</td><td>0.96</td><td>0.49</td><td>1.00</td><td>1.00</td><td>0.98</td><td>1.00</td></tr><tr><td>PCA 192</td><td>768</td><td>0.89</td><td>0.86</td><td>0.81</td><td>0.44</td><td>0.99</td><td>0.92</td><td>0.92</td><td>0.99</td></tr><tr><td>PCA 96</td><td>384</td><td>0.77</td><td>0.74</td><td>0.60</td><td>0.35</td><td>0.96</td><td>0.78</td><td>0.84</td><td>0.95</td></tr><tr><td>PCA 48</td><td>192</td><td>0.63</td><td>0.65</td><td>0.39</td><td>0.24</td><td>0.92</td><td>0.61</td><td>0.60</td><td>0.88</td></tr><tr><td>random projection 384</td><td>1536</td><td>0.80</td><td>0.66</td><td>0.64</td><td>0.47</td><td>0.67</td><td>0.67</td><td>0.82</td><td>0.91</td></tr><tr><td>random projection 192</td><td>768</td><td>0.73</td><td>0.57</td><td>0.51</td><td>0.39</td><td>0.59</td><td>0.59</td><td>0.72</td><td>0.87</td></tr><tr><td>random projection 96</td><td>384</td><td>0.62</td><td>0.45</td><td>0.37</td><td>0.37</td><td>0.34</td><td>0.40</td><td>0.57</td><td>0.73</td></tr><tr><td>random projection 48</td><td>192</td><td>0.45</td><td>0.35</td><td>0.20</td><td>0.18</td><td>0.21</td><td>0.33</td><td>0.37</td><td>0.67</td></tr><tr><td>PCA 192 + int8</td><td>192</td><td>0.88</td><td>0.84</td><td>0.80</td><td>0.43</td><td>0.99</td><td>0.93</td><td>0.92</td><td>0.99</td></tr><tr><td>PCA 96 + int8</td><td>96</td><td>0.77</td><td>0.74</td><td>0.60</td><td>0.35</td><td>0.96</td><td>0.78</td><td>0.84</td><td>0.95</td></tr><tr><td>PCA 768 + rotation + binary</td><td>96</td><td>0.77</td><td>0.57</td><td>0.62</td><td>0.45</td><td>0.85</td><td>0.82</td><td>0.88</td><td>0.96</td></tr><tr><td>PCA 384 + rotation + binary</td><td>48</td><td>0.72</td><td>0.51</td><td>0.52</td><td>0.41</td><td>0.79</td><td>0.82</td><td>0.88</td><td>0.96</td></tr><tr><td>PQ 96</td><td>96</td><td>0.82</td><td>0.73</td><td>0.68</td><td>0.44</td><td>0.88</td><td>0.87</td><td>0.90</td><td>0.98</td></tr><tr><td>PQ 48</td><td>48</td><td>0.71</td><td>0.62</td><td>0.49</td><td>0.33</td><td>0.79</td><td>0.78</td><td>0.85</td><td>0.95</td></tr><tr><td>PQ 24</td><td>24</td><td>0.60</td><td>0.51</td><td>0.32</td><td>0.20</td><td>0.52</td><td>0.65</td><td>0.78</td><td>0.89</td></tr><tr><td>UMAP 16</td><td>64</td><td>0.52</td><td>0.10</td><td>0.24</td><td>0.19</td><td>0.08</td><td>0.20</td><td>0.69</td><td>0.59</td></tr></table>

TABLE 9  
LAQ raises G over OPQ on every held-out memory. G and L of OPQ and LAQ, four splits.
<table><tr><td>Memory</td><td>Encoder</td><td>Split</td><td>G OPQ</td><td>G LAQ</td><td>L OPQ</td><td>L LAQ</td></tr><tr><td>FinQA</td><td>BGE-base</td><td>2026</td><td>0.7927</td><td>0.8021</td><td>0.8537</td><td>0.8566</td></tr><tr><td></td><td></td><td>2027</td><td>0.7930</td><td>0.8020</td><td>0.8579</td><td>0.8570</td></tr><tr><td></td><td></td><td>2028</td><td>0.7885</td><td>0.7951</td><td>0.8534</td><td>0.8557</td></tr><tr><td></td><td>E5-base</td><td>2029</td><td>0.7958</td><td>0.8035</td><td>0.8489</td><td>0.8514</td></tr><tr><td></td><td></td><td>2026</td><td>0.7872</td><td>0.7959</td><td>0.8413</td><td>0.8458</td></tr><tr><td></td><td></td><td>2027</td><td>0.7858</td><td>0.7952</td><td>0.8463</td><td>0.8544</td></tr><tr><td></td><td></td><td>2028</td><td>0.7781</td><td>0.7884</td><td>0.8440</td><td>0.8486</td></tr><tr><td></td><td>BGE-large</td><td>2029</td><td>0.7867</td><td>0.7944</td><td>0.8469</td><td>0.8450</td></tr><tr><td></td><td></td><td>2026</td><td>0.8227</td><td>0.8280</td><td>0.8819</td><td>0.8840</td></tr><tr><td></td><td></td><td>2027</td><td>0.8209</td><td>0.8261</td><td>0.8844</td><td>0.8869</td></tr><tr><td></td><td></td><td>2028</td><td>0.8185</td><td>0.8260</td><td>0.8884</td><td>0.8910</td></tr><tr><td></td><td></td><td>2029</td><td>0.8266</td><td>0.8342</td><td>0.8879</td><td>0.8880</td></tr><tr><td>LoCoMo</td><td>BGE-base</td><td>2026</td><td>0.8667</td><td>0.8744</td><td>0.8186</td><td>0.8224</td></tr><tr><td></td><td></td><td>2027</td><td>0.8572</td><td>0.8641</td><td>0.8349</td><td>0.8358</td></tr><tr><td></td><td></td><td>2028</td><td>0.8319</td><td>0.8440</td><td>0.8148</td><td>0.8182</td></tr><tr><td></td><td></td><td>2029</td><td>0.8722</td><td>0.8777</td><td>0.8319</td><td>0.8302</td></tr><tr><td></td><td>E5-base</td><td>2026</td><td>0.7979</td><td>0.8043</td><td>0.7651</td><td>0.7671</td></tr><tr><td></td><td></td><td>2027</td><td>0.7681</td><td>0.7785</td><td>0.7859</td><td>0.7946</td></tr><tr><td></td><td></td><td>2028</td><td>0.7703</td><td>0.7807</td><td>0.7612</td><td>0.7697</td></tr><tr><td></td><td></td><td>2029</td><td>0.8072</td><td>0.8156</td><td>0.7825</td><td>0.7806</td></tr><tr><td></td><td>BGE-large</td><td>2026</td><td>0.8921</td><td>0.8994</td><td>0.8496</td><td>0.8574</td></tr><tr><td></td><td></td><td>2027</td><td>0.8928</td><td>0.8989</td><td>0.8813</td><td>0.8790</td></tr><tr><td></td><td></td><td>2028</td><td>0.8678</td><td>0.8759</td><td>0.8522</td><td>0.8518</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>2029</td><td>0.9128</td><td>0.9175</td><td>0.8587</td><td>0.8602</td></tr></table>

<table><tr><td>Memory</td><td>Encoder</td><td>Split</td><td>G OPQ</td><td>G LAQ</td><td>L OPQ</td><td>L LAQ</td></tr><tr><td>Multi-Doc</td><td>BGE-base</td><td>2026</td><td>0.8196</td><td>0.8306</td><td>0.8132</td><td>0.8183</td></tr><tr><td></td><td></td><td>2027</td><td>0.8140</td><td>0.8266</td><td>0.8181</td><td>0.8208</td></tr><tr><td></td><td></td><td>2028</td><td>0.8274</td><td>0.8381</td><td>0.8102</td><td>0.8122</td></tr><tr><td></td><td>E5-base</td><td>2029</td><td>0.8165</td><td>0.8271</td><td>0.8183</td><td>0.8207</td></tr><tr><td></td><td></td><td>2026</td><td>0.8154</td><td>0.8320</td><td>0.8018</td><td>0.8092</td></tr><tr><td></td><td></td><td>2027</td><td>0.8154</td><td>0.8290</td><td>0.8068</td><td>0.8122</td></tr><tr><td></td><td></td><td>2028</td><td>0.8167</td><td>0.8302</td><td>0.7944</td><td>0.7999</td></tr><tr><td></td><td>BGE-large</td><td>2029</td><td>0.8140</td><td>0.8282</td><td>0.8037</td><td>0.8132</td></tr><tr><td></td><td></td><td>2026</td><td>0.8552</td><td>0.8647</td><td>0.8526</td><td>0.8572</td></tr><tr><td></td><td></td><td>2027</td><td>0.8540</td><td>0.8635</td><td>0.8564</td><td>0.8601</td></tr><tr><td></td><td></td><td>2028</td><td>0.8608</td><td>0.8686</td><td>0.8462</td><td>0.8524</td></tr><tr><td></td><td>Titan V2</td><td>2029</td><td>0.8545</td><td>0.8617</td><td>0.8557</td><td>0.8559</td></tr><tr><td></td><td></td><td>2026</td><td>0.6680</td><td>0.7025</td><td>0.7942</td><td>0.8010</td></tr><tr><td></td><td></td><td>2027</td><td>0.6434</td><td>0.6811</td><td>0.7962</td><td>0.8043</td></tr><tr><td></td><td></td><td>2028</td><td>0.6651</td><td>0.7030</td><td>0.7848</td><td>0.7958</td></tr><tr><td></td><td></td><td>2029</td><td>0.6480</td><td>0.6803</td><td>0.7852</td><td>0.7909</td></tr><tr><td>FinDER</td><td>BGE-base</td><td>2026</td><td>0.8139</td><td>0.8303</td><td>0.8046</td><td>0.8072</td></tr><tr><td></td><td></td><td>2027</td><td>0.8142</td><td>0.8312</td><td>0.8024</td><td>0.8055</td></tr><tr><td></td><td></td><td>2028</td><td>0.8142</td><td>0.8305</td><td>0.8029</td><td>0.8069</td></tr><tr><td></td><td></td><td>2029</td><td>0.8137</td><td>0.8289</td><td>0.8049</td><td>0.8076</td></tr><tr><td></td><td>E5-base</td><td>2026</td><td>0.7982</td><td>0.8157</td><td>0.7809</td><td>0.7893</td></tr><tr><td></td><td></td><td>2027</td><td>0.8028</td><td>0.8202</td><td>0.7793</td><td>0.7854</td></tr><tr><td></td><td></td><td>2028</td><td>0.8075</td><td>0.8238</td><td>0.7812</td><td>0.7883</td></tr><tr><td></td><td></td><td>2029</td><td>0.8032</td><td>0.8200</td><td>0.7842</td><td>0.7905</td></tr><tr><td></td><td>BGE-large</td><td>2026</td><td>0.8456</td><td>0.8559</td><td>0.8450</td><td>0.8480</td></tr><tr><td></td><td></td><td>2027</td><td>0.8458</td><td>0.8558</td><td>0.8445</td><td>0.8494</td></tr><tr><td></td><td></td><td>2028</td><td>0.8461</td><td>0.8563</td><td>0.8433</td><td>0.8478</td></tr><tr><td></td><td></td><td>2029</td><td>0.8439</td><td>0.8567</td><td>0.8393</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>0.8454</td></tr></table>

TABLE 10

The far hinge ${ \mathcal L } _ { \mathrm { f a r } }$ carries the gain in G. Change from the OPQ start over all 52 held-out runs, each objective trained with the same recipe and checkpoint rule; means, range of the change in G, and runs in which G is higher.
<table><tr><td>Objective</td><td>G change</td><td>range</td><td>higher</td><td>L change</td></tr><tr><td> $\mathcal { L } _ { \mathrm { r e c } }$ </td><td>.0000</td><td>.0000 to .0000</td><td>0/52</td><td>.0000</td></tr><tr><td> $\mathcal { L } _ { \mathrm { r e c } } +$  near hinge</td><td>+.0058</td><td> $. 0 0 0 0 \ \mathrm { t o \ t . 0 1 2 4 }$ </td><td>50/52</td><td>+.0043</td></tr><tr><td> $\mathcal { L } _ { \mathrm { r e c } } + \mathcal { L } _ { \mathrm { f a r } }$ </td><td>+.0127</td><td> $+ . 0 0 2 2 \ \mathrm { t o } + . 0 4 5 3$ </td><td>52/52</td><td>+.0030</td></tr><tr><td> $\mathrm { L A Q , E q . }$  (10)</td><td>+.0124</td><td> $+ . 0 0 4 7 \ \mathrm { t o \ t . 0 3 7 9 }$ </td><td>52/52</td><td>+.0039</td></tr></table>