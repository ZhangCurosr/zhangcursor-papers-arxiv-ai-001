# A Power Law in Logarithm's Clothing: On the Scalability of Graph-Based Vector Search

Sajad Faghfoor Maghrebi University of Toronto Canada smaghrebi@cs.toronto.edu

Navid Eslami University of Toronto Canada navideslami@cs.toronto.edu

Niv Dayan   
University of Toronto   
Canada   
nivdayan@cs.toronto.edu

## Abstract

Most vector databases rely on graph-based indexes, notably HNSW and Vamana, for approximate nearest neighbor search. With the recent widespread adoption of embedding models, the datasets these databases store have grown rapidly. This growth raises a natural question: at a fixed accuracy, how does search cost scale with dataset size? The prevailing answer is that search cost grows polylogarithmically. Yet this claim is formally proven only under special conditions; for the indexes used in practice, it is asserted without proof. The claim is also largely untested: standard benchmarks measure search cost at a single dataset size, not across sizes.

We put this claim to the test and measure how search cost scales across dataset sizes. The answer turns out to depend on the scale itself. (1) As long as the dataset size N is small relative to the data's intrinsic dimensionality, search cost grows in step with Nc for a constant 0 < c < 1. We call this scaling a Sublinear Power Law. (2) Once N is large enough, the growth slows to subpolynomial, consistent with the prevailing poly-logarithmic claim. Empirically, the Sublinear Power Law appears on every dataset, mostly up to its full size, and at every recall target, query hardness level, and index configuration we test. Still, the transition to subpolynomial growth does appear on the two datasets that grow large enough relative to their intrinsic dimensionality. Underlying both behaviors is a single mechanism: the intrinsic dimensionality a dataset exhibits grows with its size, until the data resolves its underlying distribution. Higher intrinsic dimensionality packs more vectors into the small neighborhood around the query that the search must examine.

We present a unifying theory that formalizes beam-search cost and explains all of our observations. For both exact and boundeddegree constructions, we prove the Sublinear Power Law and the eventual transition to poly-logarithmic scaling, and derive the scale at which the transition occurs. We also develop models that predict the power-law exponents for any recall target and index configuration. These models provide a principled way to navigate trade-offs among search cost, insertion cost, and recall as data grows.

## 1 Introduction

Unstructured data, spanning documents, images, videos, and audio, has grown rapidly [111]. Querying such datasets using traditional approaches based on attribute values and keywords fails to capture semantic similarity [105]. Meanwhile, advances in learning-based methods, especially deep neural networks, have enabled models that transform unstructured data into dense vector representations suitable for semantic search [91]. These embedding models are trained so that semantic similarity corresponds to geometric proximity: the vectors closest to a query usually represent the most relevant data items.

Vector Search. Together, these trends have driven the rise of vector databases, which index the embeddings produced by these models [7, 105]. Their core retrieval primitive is nearest neighbor search. However, computing the exact nearest neighbors is prohibitively expensive at scale [12, 14, 41]. Vector databases instead accept a small loss in accuracy and perform approximate nearest neighbor (ANN) search. To do so, they use an index that narrows the search to a small set of promising candidates [32, 64, 97].

Graph-based Indexes. The ANN indexes that perform best in practice are graph-based [61, 107]. A graph-based index represents each vector as a node, connected to a subset of its nearby vectors, with a few longer-range edges that keep distant regions reachable in a few hops. To answer a query, the index runs beam search, a generalization of greedy search that walks the graph toward the query and returns the closest vectors encountered [64, 97]. Some graph constructions provide guarantees on accuracy and on reaching a query's neighborhood, but are often too expensive to build exactly [2, 97]. Practical graph-based indexes draw on these exact constructions but build them with cheaper heuristics [64, 97]. The two most widely used such indexes, HNSW [64] and Vamana [97], power vector search across databases [51, 69, 87, 99, 102, 109], search engines [77, 103], and libraries [26, 68, 72].

Sustaining Accuracy As Data Grows. The data stored in vector databases grows continuously [9, 94, 112]. As the data grows, however, search accuracy—typically measured by recall, the fraction of true nearest neighbors returned by the search—deteriorates [19, 27, 101]. Maintaining stable recall therefore requires adjusting the index parameters in response [27, 65, 115]. A common practice is to periodically measure recall using exact search [28, 86, 104] and adjust parameters accordingly [19, 65, 101]. If the measured recall falls short of the user's recall target, the user raises the search effort, remeasures, and repeats until the target is reached. Recent work in academia as well as in industry [101, 116] improves upon this practice by using statistical or learned models of the search effort required to achieve a recall target [19, 56, 114]. Underlying all these approaches is a natural research question: At a fixed level of accuracy, how does search cost scale as the dataset grows?

The Logarithmic Folklore. The literature answers this question with poly-logarithmic scaling. For exact graph constructions, the claim is sometimes backed by proof [2, 42]. For the heuristic indexes used in practice, it is asserted and left unproven. In particular, the paper introducing HNSW claims logarithmic search cost, but the scaling experiments supporting this claim use only low-dimensional synthetic vectors (d ≤ 8), and its largest experiment deviates from logarithmic scaling [64]. The paper introducing Vamana argues that greedy search on its exact construction takes logarithmically many steps, but reports no measurement of search cost across dataset sizes [97]. The claim has nonetheless propagated into industry documentation [26, 69, 77, 83, 90, 109] and surveys [78, 79, 107] which restate it as a settled property of these indexes. It also remains largely untested: benchmarks commonly measure the recall-cost trade-off at a single dataset size, not the scaling of cost across sizes at fixed recall [4, 43, 93, 107].

Observations. We put the folklore to the test and make two observations: (1) When the dataset size N is small relative to the data's intrinsic dimensionality, the average search cost of a graphbased index at fixed recall grows in step with $N ^ { c }$ for a constant $0 \textless c \textless 1$ . We call this scaling a Sublinear Power Law. (2) Once N is large enough, the growth slows to subpolynomial, consistent with the prevailing poly-logarithmic claim. We verify these observations through extensive experiments: the Sublinear Power Law appears on all eight of our datasets, at recall targets between 90% and 99%, under many index configurations, and for hard and easy queries alike (Section 3.2). On two datasets, the data grows past the transition point and search cost begins to slow below the power law (Sections 3.2 and 3.3).

Core Contribution: Unifying Theory. By analyzing the anatomy of query cost, we find the mechanism behind these scaling behaviors: search cost is governed by the number of vectors in a small neighborhood around the query. The size of that neighborhood grows exponentially with the intrinsic dimensionality the dataset exhibits at its current size. As long as the dataset sparsely samples its underlying distribution, that dimensionality grows in step with log N. An exponential in log N is a power of N, so the neighborhood, and hence the cost, grows as a power law. Once the dataset samples the distribution densely, intrinsic dimensionality growth slows and cost growth approaches poly-logarithmic (Section 4.1). Our unifying theory formalizes this claim. For both exact and bounded-degree constructions¹ of HNSW and Vamana, we prove the Sublinear Power Law and the eventual transition to polylogarithmic scaling, derive the scale at which that transition occurs, and extend the results to other graph-based indexes (Section 4.2).

Additional Contribution: Cost Models. Under the Sublinear Power Law, we model the query cost as a function of the recall target and the construction parameters, across datasets, for both HNSW and Vamana (Section 5.1). For insertions, we model the cost analogously (Section 5.2). Combined, the models help users navigate the trade-offs between query cost, insertion cost, and recall.

## 2 Background

This section provides background on vector databases, graph indexes, and how they are thought to scale with dataset size. A vector database stores complex objects as points in a high-dimensional space, produced by an embedding model that maps similar objects to nearby points [34, 48, 50, 106]. A query is embedded into that same space by the same embedding model, and the search process returns the point(s) closest to it. Formally, given a set of points $S \subset \mathbb { R } ^ { d }$ and a query point $q ,$ the goal is to identify the point in S closest to q. The k-nearest neighbor variant generalizes this, returning the k points in S with the smallest distance to q. The embedding model dictates the dimensionality d, which ranges from hundreds to thousands (e.g., 1536 for OpenAI's text-embedding-3-small and 3072 for text-embedding-3-large [75]). The dimensionality must be high enough to represent the many attributes needed to distinguish objects across a large and diverse corpus.

Curse of Dimensionality & Exact Search. In such high-dimensional spaces, search is hard due to the curse of dimensionality [11, 41], a phenomenon whereby most points lie at similar distances from the query, and the nearest neighbor is only slightly closer than the rest [38, 96]. Because distances concentrate, points do not separate cleanly, so traditional multidimensional indexes (e.g., Rtrees [37] or KD-trees [10]) lose their pruning power [13, 57, 110]. To distinguish the nearest neighbors from the many nearly equidistant points, the search must examine a large fraction of the dataset. This is expensive [70] and degrades to a linear scan in the worst case [15].

Manifolds & Intrinsic Dimensionality. Real datasets, despite their high embedding dimension, do not span the entire embedding space. Instead, the data resides near lower-dimensional structures embedded within this space [30]. These structures are called manifolds [16], each of which behaves locally like a flat, Euclidean space. Each manifold's dimensionality is considered the intrinsic dimensionality of the data residing near the manifold [54]. What makes search expensive is this intrinsic dimensionality, not the number of dimensions in the embedding. It is smaller than the embedding dimension, but still big enough that exact search stays slow [82].

Approximate Nearest Neighbor Search. Approximate nearest neighbor (ANN) search sidesteps this expensive query cost by giving up the guarantee of returning the exact nearest neighbors. This allows the search procedure to apply a rough pruning rule that discards a sizable portion of the search space while ensuring that a nearest neighbor is retained with high probability. Query accuracy for ANN is commonly formalized in two ways. The first is the classical (1 + ε)-approximate distance guarantee: the returned point's distance to query q is within a factor of (1 + ε) of the nearest neighbor distance. The same relaxation extends to the k-nearest neighbor case. This guarantee constrains only the distance, not whether the point is a true nearest neighbor. The second, recall, avoids this issue, and is defined as the fraction of the true nearest neighbors that the search returns. Formally, with R the set of true k nearest neighbors and R' the returned set, recall is |R ∩ R'|/k. A higher recall value indicates a closer match to the exact search result. This tolerance for inexact answers motivates a class of data structures, called vector indexes, built to answer ANN queries [21, 107].

Graph-based Indexes. Graph-based indexes are widely regarded as the state of the art among vector indexes [64, 97]. They answer ANN queries by traversing a graph whose nodes are the data points. Edges connect each point to a small set of other points, arranged so that search toward a query approaches its nearest neighbors. All graph-based indexes draw these edges following two core intuitions: (1) each point's edges should lead to points near it, so that once the search closes in on the query, it has easy access to the true nearest neighbors, and (2) the edges should spread across distinct directions, so that from any position some edge makes progress toward the query. The two intuitions act in tandem: every point has many nearby points it could connect to, but diversity decides which few of them become edges, sparsifying the graph while preserving navigability. A query that comes from the same distribution as the data lands amid points whose edges already cover its general direction, and is reachable for the same reason the data points are. Next, we present the popular graph constructions, beginning with the theoretically grounded designs and moving to the practical relaxations they inspired.

![](images/e28a8ca6b183d28600e495aa5978ec42819d1667a82bf4988d6dc3cb6e6daa3a.jpg)

![](images/fd2956bfec3da41d7e10dea5851a2b8e342d229c4ce53e8de6d6348c86107586.jpg)  
(b) SNG

(a) Randomized NG  
![](images/daa1674f7025738009e2f22b0babb8c4b0ceb117365cd92c27de9ea3680b94ca.jpg)

![](images/fe898d995a0aacf502d1b610f261348267c12bf06250e83ba5ae481c2841fa69.jpg)  
(c) Vamana  
(d) τ-MG  
Figure 1: Edge selection across graph constructions. In each subfigure, p decides whether to add an edge to q. The node r already connected to $\boldsymbol { p }$ (red directed edge) lies in the shaded region, where it occludes q and suppresses the edge $\scriptstyle { p - q } .$

Randomized Neighborhood Graph. The Randomized Neighborhood Graph (Randomized NG) [2] is one of the earliest graph-based indexes with a theoretical guarantee for ANN search (Figure 1-a). It builds a directed graph over the data points. Specifically, it guarantees that for any query $q ,$ the search finds a point whose distance to q is within a factor of 1 + ε of the distance to q's nearest neighbor. To build this graph, the construction process partitions the space around each point p into cones with apex p. Within each cone, it scans the enclosed points in random order and links p to every point closer to $\mathcal { P }$ than all previously scanned points, i.e., every new running minimum. This scan results in an average of O(log N) neighbors in every cone, at approximately geometrically increasing distances from p. Drawing the edges of a single point requires comparing it against all other points to track the running minimum per cone, so drawing the edges of all N points costs $O ( N ^ { 2 } )$

The query cost of a graph-based index factors into the number of search steps and the out-degree at each step. For Randomized $\mathrm { N G } ,$ both are poly-logarithmic in N. The search takes $O ( \log ^ { 2 } N )$ steps: an ideal straight path to q takes O(log N) steps, each halving the remaining candidates as in a skip list [85], and each ideal step expands into O(log N) actual steps as the path bends in high dimensions. The out-degree is $O ( ( 1 \bar { / \epsilon } ) ^ { d - 1 }$ log N): the $( 1 / \epsilon ) ^ { d - 1 }$ factor accounts for the number of cones around each point, and each cone contributes O(log N) neighbors [2]. Although both factors scale poly-logarithmically with $N ,$ the $( 1 / \epsilon ) ^ { d - 1 }$ factor grows large in high dimension: smaller € needs narrower cones, and higher dimension needs more cones to cover all directions. The graph therefore grows dense on both counts, which slows search [2].

Table 1: Notations used in the paper.
<table><tr><td>Symbol</td><td>Description</td></tr><tr><td>Dataset and query</td><td></td></tr><tr><td>N</td><td>Number of vectors in the index</td></tr><tr><td> $d _ { \mathrm { a m b } }$ </td><td>Ambient dimensionality of the embedding space</td></tr><tr><td> $d _ { \mathrm { i n t } }$ </td><td>Intrinsic dimensionality of the dataset</td></tr><tr><td> $k$ </td><td>Number of nearest neighbors returned per query</td></tr><tr><td> $q$ </td><td>A query vector</td></tr><tr><td>Index parameters</td><td></td></tr><tr><td> $M$ </td><td>Maximum out-degree of each node in the  $\mathrm { g r a p h }$ </td></tr><tr><td>efconstruction,  $e f _ { \mathrm { s e a r c h } }$ </td><td>Candidate list size at build and search time</td></tr><tr><td>Cost metrics</td><td></td></tr><tr><td> $\operatorname { d c }$ </td><td>Distance computations per query</td></tr><tr><td> $\mathrm { d c } _ { \mathrm { b u i l d } }$ </td><td>Total distance computations to build an index of size N</td></tr><tr><td>Query-cost model (Eq. 1, 2, 3)</td><td></td></tr><tr><td>c</td><td>Query-cost scaling exponent, dc α  $N ^ { c }$ </td></tr><tr><td>c0</td><td>Dataset-specific baseline exponent</td></tr><tr><td> $\alpha$ </td><td>Sensitivity of c to recall demand ln(  $1 / \delta )$ </td></tr><tr><td> $\gamma _ { 1 } , \gamma _ { 2 }$ </td><td>Reduction in c per ln(efconstruction) and per ln(M)</td></tr><tr><td> $\beta$ </td><td>Search-side exponent,  $\operatorname { d c } / q \propto e f _ { s \mathrm { e a r c h } } ^ { \beta }$ </td></tr><tr><td>Build-cost model (Eq. 4)</td><td></td></tr><tr><td> $\kappa _ { . }$ </td><td>Dataset-specific intercept for build cost</td></tr><tr><td> $\beta _ { . } ^ { \prime }$ </td><td>Exponent of efconstruction in build cost</td></tr><tr><td></td><td></td></tr><tr><td> $c _ { 0 } ^ { \prime }$   $\gamma _ { 2 } ^ { \prime }$ </td><td>Baseline size-dependence of build cost Interaction coefficient between ln M and ln N</td></tr></table>

Sparse Neighborhood Graph. The same paper [2] that introduced Randomized NG also proposed a simpler index, the Sparse Neighborhood Graph (SNG), which underlies many practical graph-based indexes today. SNG connects two points p and q only when no third point r to which $\mathcal { P }$ is already connected lies in the intersection of the two balls of radius $d ( p , q )$ centered at $\mathcal { P }$ and $q .$ This intersection is called the lune of $\boldsymbol { p }$ and q (Figure 1-b). Intuitively, such a point r provides a closer intermediate step from p toward $q ,$ so the direct edge from p to q can be removed while preserving navigability. SNG is built by considering, for each point ${ \boldsymbol { \mathit { p } } } ,$ all other points in increasing order of distance from $\mathcal { P }$ and pruning candidates that violate this rule. A naive construction takes $O ( N ^ { \bar { 3 } } )$ time, while the paper achieves O(N²) [2] by drawing on algorithms for the relative neighborhood graph2 from the 1980s [45, 100].

For ${ \mathrm { S N G } } , ^ { 3 }$ a later work [32] proves that the search takes in expectation $O ( N ^ { 1 / d }$ log N/∆r) steps for uniformly distributed data and queries that are dataset points, where ∆r is the minimum gap between pairwise distances in the dataset. A later paper [80] proves that, under the same assumptions, ∆r shrinks as $O ( N ^ { - 1 / d } )$ , bounding the steps by $O ( N ^ { 2 / d }$ log N). The out-degree, in turn, is independent of N and hence enters the query time only as a constant, but it is exponential in d: the SNG paper bounds it by the number of caps of angular diameter $\pi / 3$ fitting on the d-dimensional sphere, $\mathfrak { i . e . , 2 ^ { O ( d ) } }$ , and the empirical out-degree grows accordingly [2]. This makes each step, and hence the search, expensive in high dimensions even when the number of steps remains logarithmic.

Vamana. Inspired by SNG, Vamana relaxes the pruning rule with a slack parameter α ≥ 1 [97] (Figure 1-c). Vamana connects two points p and q only when no point r already connected to $\boldsymbol { p }$ comes closer to q than p by a factor of at least α, i.e., $\alpha \cdot d ( r , q ) < d ( p , q )$ $\operatorname { A t } \alpha = 1$ , this is exactly the pruning rule in SNG. The intuition behind the slack parameter is that every node has an edge either to the query's nearest neighbor or to a point at least α times closer to the query than the node itself ${ \mathrm { i } } s ,$ so each greedy step cuts the remaining distance by the constant factor α. Guaranteeing this progress costs extra edges: a larger α makes the pruning condition harder to satisfy, so fewer edges are removed, increasing the graph's out-degree.

Table 2: Datasets used in the experiments.
<table><tr><td>Dataset</td><td>Dim.</td><td>Int. Dim. @ 1M</td><td>Type</td><td>Metric</td><td>Model</td><td>Sampled From</td></tr><tr><td>SIFT</td><td>128</td><td>19.2</td><td>Image</td><td>L2</td><td> $\mathrm { S I F T \ d e s c r i p t o r s \ [ 5 9 ] }$ </td><td>SIFT1B [47]</td></tr><tr><td>DEEP</td><td>96</td><td>20.7</td><td>Image</td><td>L2</td><td>GoogLeNet [98]</td><td>DEEP1B [6]</td></tr><tr><td>SpaceV</td><td>100</td><td>29.2</td><td>Text</td><td>L2</td><td> $\mathrm { S p a c e V } \mathrm { S u p e r i o r } [ 9 2 ]$ </td><td>SpaceV1B [67]</td></tr><tr><td>Wiki</td><td>768</td><td>30.1</td><td>Text</td><td>L2</td><td>MPNet [95]</td><td> $\stackrel { \bullet } { \mathrm { W i k i - a l l } } \left[ \tilde { 8 } 9 \right]$ </td></tr><tr><td>GloVe</td><td>100</td><td>32.4</td><td>Text</td><td>Angular</td><td>GloVe [81]</td><td>GloVe-100 [81]</td></tr><tr><td>Rand64</td><td>64</td><td>40.2</td><td>Synthetic</td><td>Angular</td><td></td><td> $\operatorname { U n i f } ( S ^ { 6 3 } ) ^ { 5 }$ </td></tr><tr><td>GIST</td><td>960</td><td>42.5</td><td>Image</td><td>L2</td><td>GIST descriptors [73]</td><td>GIST1M [46]</td></tr><tr><td>OpenAI</td><td>1536</td><td>46.5</td><td>Text</td><td>L2</td><td>ada-002 [74]</td><td>OpenAI-ArXiv [76]</td></tr></table>

A later work [42] proves that greedy search reaches an $\textstyle \left( { \frac { \alpha + 1 } { \alpha - 1 } } + \epsilon \right) \cdot$ approximate nearest neighbor in $\begin{array} { r } { O ( \log _ { \alpha } { \frac { \Delta } { ( \alpha - 1 ) \epsilon } } ) } \end{array}$ steps, where ∆ is the ratio of the largest to smallest pairwise distance in the dataset $( { \mathrm { i . e . } }$ , the spread). This ratio stays small in practice even at large N [42, 96]. Each step pays the out-degree of the visited point, bounded by $O ( ( 4 \alpha ) ^ { d } \log \Delta )$ , again exponential in the dimensionality.4 When ∆ grows at most polynomially with N, as under a uniform distribution, both log ∆ factors become O(log N), and multiplying steps by out-degree gives a query time poly-logarithmic in N. Also, the construction process, as in SNG, prunes each point against the entire dataset, incurring a total cost at least quadratic in N.

τ-Monotonic Graph. Also inspired by SNG, the τ-Monotonic Graph (τ-MG) [80] extends SNG's guarantees to query points within distance τ of some dataset point, for a fixed τ. The construction process connects any two points within distance 3τ, and for farther pairs applies an SNG-like rule on the lune shrunk by 3τ (Figure 1-d). These extra edges densify SNG and guarantee a path that moves at least τ closer to the query at every step. Assuming uniformly distributed data and a query within τ of some dataset point, the paper proves the search reaches the nearest neighbor in $\mathsf { \bar { O } } ( N ^ { 1 / d } \log ^ { 2 } N )$ steps. The paper does not analyze the out-degree, treating it as a constant. In fact, because all points within 3τ are connected, the out-degree is $\Omega ( ( 3 \tau ) ^ { d } )$ , so the constant hidden in its query cost is exponential in the dimensionality. As in SNG, construction cost is at least quadratic in N.

Practical Graph-based Indexes. The quadratic construction cost of the above graphs is the core obstacle to their practical use at scale [2, 32, 80]. Modern practical indexes lower this cost by restricting the set of points to which each point can connect. Rather than treating every point as a candidate for a point's edges, they apply a greedy beam search to find a much smaller set of nearby points, then apply the pruning rule within that smaller set. More precisely, when inserting a new point, the index runs a beam search on the current graph for the new point. Beam search maintains a candidate list of fixed size, here $e f _ { \mathrm { c o n s t r u c t i o n } } .$ and starts from one or more entry points. It repeatedly expands the closest unexpanded candidate, keeping only the closest points found so far in the list. Once no unexpanded candidate remains, the list is pruned locally and the new point is connected to at most M of the closest survivors. The same beam search procedure is used at query time, with a candidate list of size $e f _ { \mathrm { s e a r c h } } \geq k ,$ and the top-k results are selected from it.

![](images/904533c6b18b6a4e050df2af283fb83483e344fad960acee00958bac8ee2f3b5.jpg)  
Figure 2: As the data grows, distance computations per query rise (top) and recall drops (bottom). Here, we use the HNSW index with $e f _ { \mathbf { c o n s t r u c t i o n } } = 2 0 0 , M = 3 2$ , and $k = 1 0$ , with one curve per efsearch value.

The practical indexes differ mainly in the pruning rule each applies to this candidate list rather than to the full dataset. HNSW [64] organizes nodes into hierarchical layers, assigning each node to higher layers with exponentially decaying probability, and applies the SNG rule within each layer. The practical version of Vamana [97] applies the α-based pruning rule; henceforth, we refer to this version of Vamana simply as Vamana. The HNSW paper claims that its query cost scales logarithmically with dataset size while its construction cost stays below quadratic, at O(N log N) [64]. The paper that introduced Vamana [97] argues that greedy search on its exact construction takes logarithmically many steps but leaves the query cost scaling of its practical version unstated; benchmarks nonetheless place it alongside HNSW as state-of-the-art [4, 97]. These logarithmic scalability claims and similar claims from the same line of work [31, 32, 55, 63, 80] have since propagated into vendor documentation and industry blogs, which restate them as a settled property of these indexes [29, 69, 77, 83, 90, 109]. In the next section, we evaluate how these indexes scale and bridge this gap between claimed and measured scaling both theoretically and empirically.

In this section, we study empirically how query cost scales as data grows, focusing on HNSW and Vamana6, the most widely used graph-based indexes [69, 78]. Later in Section 4 we show that our findings generalize to other graph-based indexes.

## 3 How Graph-Based Indexes Scale

## 3.1 The Problem: Recall Deterioration

Our first experiment verifies a known phenomenon: when the index parameters are held fixed, recall degrades as the dataset grows [27, 65, 115]. To simulate data growth, we draw uniformly random subsets of increasing sizes from each of the datasets in Table 2 up to their full sizes. Each subset is shuffled before index construction. When a dataset provides a canonical query set, we use it as our query workload. Otherwise, we reserve 10,000 uniformly sampled points as queries, excluding them from the set of points over which the index is built. To compute the true nearest neighbors for each query (i.e., its ground truth), we perform an exact search by linearly scanning the entire dataset. We then execute the query using the index and compare the returned results against the ground truth to compute recall. We measure query cost in terms of the number of distance computations (dc) per query as this is the dominant cost in graph traversal [32, 33, 113]. By so doing, we avoid conflating algorithmic performance with implementation idiosyncrasies (cache evictions, prefetching, hardware-optimized distance computations, etc.). All experiments in the paper follow this setup.

![](images/16263485ac0a1d3432801f498a07644a2761a2259631a8eec5ed93e68c4061a7.jpg)  
Figure 3: When recall is fixed, the number of distance computations (y-axis) increases with dataset size (x-axis) across the datasets from Table 2, forming a linear pattern in the log-log diagram. This is consistent with a sublinear power-law.

Figure 2 shows the results on HNSW using three representative datasets (the remaining ones behave similarly). Each column represents a different dataset (SIFT, DEEP, and GloVe). The top row reports distance computations and the bottom recall. Each subfigure shows one curve per setting of the $e f _ { \mathrm { s e a r c h } }$ parameter. For each of these curves, we observe that distance computations per query slightly increase with dataset size N (top row), while recall drops significantly (bottom row). The reason for this behavior is that as the data grows, more points lie close to the query. Beam search therefore converges more slowly, and the true nearest neighbors face more competition for a place in the candidate list, as suggested in prior work [65]. This recall degradation is problematic for applications such as retrieval-augmented generation (RAG) [52] and recommendation systems [22], which require stable recall even as datasets grow.

## 3.2 The Impact of Fixing Recall

The experiment in the previous section raises a natural question: can recall be kept stable as the dataset grows? Although prior work has not addressed this question directly, it has considered the related problem of meeting a specified recall target. The standard approach is to periodically measure recall using exact search [28, 86, 104] and adjust search parameters accordingly. If the measured recall falls below the target, the user increases the $e f _ { \mathrm { s e a r c h } }$ parameter, measures recall again, and repeats this process until the target is reached [19, 65, 101]. Recent work in academia as well as in industry (e.g., Zilliz Cloud [116] and Turbopuffer [101]) improves upon this approach by using statistical or learned models of the search effort required to achieve a recall target [19, 56, 114]. Once recall is held fixed, another core question emerges: how does query cost scale as the dataset grows?

![](images/2562bc4d487ecb58f976993e8d16ee2db78b0702a38099bfb6bd981b03208c54.jpg)  
Figure 4: Both hard and easy queries follow a power law scaling. We use the same HNSW setup as Figure 2.

Experimental Setup. To hold recall fixed, we set $e f _ { \mathrm { s e a r c h } }$ to the smallest value that attains a given fixed recall target. Figure 3 reports the average number of distance computations per query across a grid of datasets, recall targets, and build settings. Rows correspond to different datasets. We progressively grow each dataset by increasing its number of points up to its original full size. For the substantially larger SIFT, DEEP, and SpaceV7 datasets, we cap the size at 100M points to keep the total experimental runtime to approximately one week. Recall targets of 90%, 95%, 97.5%, and 99% appear as separate curves in each subfigure. Each column corresponds to a different index configuration. HNSW occupies the left four columns and Vamana the right four, with the maximum out-degree $M \in \{ 1 6 , 3 2 , 6 4 , 1 2 8 \}$ and construction list size efconstruction ∈ {100, 200, 400, 800} varying across the columns.8 Due to limited space, we show four of the sixteen benchmarked $( M , e f _ { \mathrm { c o n s t r u c t i o n } } )$ configurations per index; the remaining settings behave similarly.

Results. On the log-log axes in Figure 3, most curves are approximately linear as the dataset grows, indicating that query cost follows a Sublinear Power Law:

$$
\mathrm { d c } \propto N ^ { c } .\tag{1}
$$

Here, $0 < c < 1$ , and the value at the right end of each curve reports the fitted exponent c.

Figure 3 also indicates the factors that impact the exponent c in Equation 1. The exponent increases with higher recall targets, since they leave less margin for missing true nearest neighbors and force the search process to traverse more of the graph. Higher values of $e f _ { \mathrm { c o n s t r u c t i o n } }$ and M reduce c by yielding graphs in which the beam search reaches the query's nearest neighbors in fewer hops. We also measured $k \in \{ 1 , 5 , 1 0 , 2 0 \}$ and found that varying it does not materially change c, so we report only $k = 1 0 .$ The nearest neighbors lie at nearly the same distance from the query, so retrieving all k is about as hard as retrieving one. We give more intuition for this in Section 4.1.

Beyond these parameters, the exponent tends to increase with dataset hardness. The rows in Figure 3 are ordered by increasing intrinsic dimensionality, a proxy for hardness, with estimates listed in Table 2. For example, GloVe has higher intrinsic dimensionality than SIFT (32.4 vs. 19.2) and a higher exponent at the same recall level. This suggests that beam-search cost scales more steeply with N on harder datasets.

![](images/775c242b5683fe0787249c1e4aa5d48025f7befbb9e49ed1151ed70fbb7b5c02.jpg)  
Figure 5: The power law persists to billion scale on SIFT and DEEP, while SpaceV's slope decreases as the dataset grows.

Varying Query Hardness. As discussed in Section 2, queries within a workload can vary substantially in hardness [19, 56, 108], with queries in regions of higher intrinsic dimensionality generally being more expensive to answer [5]. To test whether the Sublinear Power Law holds across query hardness levels within the same workload, we rank queries by intrinsic dimensionality9 and rerun the experiment separately on the easiest and hardest 20%. On SIFT, DEEP, and GloVe, the hard group's intrinsic dimensionality exceeds that of the easy group by at least 7.9, 9.6, and 11.0 points, respectively. Figure 4 shows that while the two groups have different exponents, both still follow the Sublinear Power Law.

Summary. Our experimental results so far challenge the conventional logarithmic characterization of graph-based indexes from Section 2. Under logarithmic scaling, each doubling of N would add only a constant number of distance computations, causing the curves in Figures 3 and 4 to bend downward rather than remain straight. Instead, the Sublinear Power Law describes most of the datasets, index configurations, and query hardness levels we measure.

That said, we also observe exceptions to the Sublinear Power Law. For SpaceV and OpenAI, some of the curves in Figure 3 seem to drift very slightly below their fitted power laws, as if the scaling were slowing down. Only larger dataset sizes can tell whether the drift is real. OpenAI already uses its full base dataset, whereas SpaceV can grow by another order of magnitude; we explore this next.

## 3.3 Scaling with Larger Data

The SIFT, DEEP, and SpaceV datasets each have a base collection size of one billion points, but at that scale the index for a single configuration occupies 300 GB-1 TB and takes several days to build [94]. We therefore choose one representative configuration $( e f _ { \mathrm { c o n s t r u c t i o n } } = 1 0 0 , M = 3 2 )$ and repeat the experiment with Vamana10. We grow each of the three datasets from 100K points to their full base size of one billion points.

Figure 5 shows that on SIFT and DEEP, the same Sublinear Power Law continues to hold as the data grows. The query cost of SpaceV, however, behaves differently: it slopes downwards as the dataset grows. Thus, while the scalability of graph-based indexes exhibits clear empirical patterns, two questions remain: what explains the initial sublinear power-law scaling across all workloads, and why does this scaling eventually slow for some workloads?

## 4 Unifying Theory

We now develop a unifying theory that explains the observations so far: when recall is held fixed, query cost initially scales as a Sublinear Power Law and then slows as the dataset grows. The theory suggests that the slowdown observed for SpaceV and OpenAI in the previous section is not specific to these datasets; we would observe the same transition on the other datasets if they could be scaled to sufficiently large sizes.

## 4.1 Anatomy of Query Cost

Search Phases. Conceptually, beam search on graph-based indexes such as HNSW or Vamana involves two phases. The first, which we call the shortcut phase, navigates from the entry point toward the query's neighborhood: at each step, the search picks the candidate closest to the query and checks its neighbors, moving steadily inward [64, 107]. In this phase, beam search behaves essentially like greedy search, since the best candidate improves at every step. The second, which we call the exploration phase, searches the neighborhood of the query for its true k-nearest neighbors: the search now moves outward, expanding candidates it had passed over and branching into paths it initially skipped, until it can no longer make useful headway. We now study the costs of these two phases.

Exploration Phase. Across all benchmarks in Figures 3-5, we measure the proportion of search cost spent in the shortcut phase versus the exploration phase. In every case, the exploration phase accounts for at least 80% of the cost, consistent with prior observations [19, 20, 58, 60]. This phase is expensive because reaching the query vicinity does not imply that all true nearest neighbors are easily reachable. In high dimensions, distances concentrate [96], so points close to the same query need not be close to one another. At the same time, bounded node degree limits local connectivity to control memory and per-hop search cost. Together, these effects prevent the query neighborhood from forming a tightly connected region [17, 25], forcing nearby points to be reached through roundabout rather than direct paths and requiring exploration beyond the true nearest neighbors to achieve a recall target.

Modeling Exploration Cost. We now construct a mathematical model of the cost of the exploration phase, aiming to understand why the Sublinear Power Law arises. The model represents the query's neighborhood as two nested hyper-balls (multi-dimensional balls) centered at the query point q. The inner hyper-ball extends from the query to its k-th nearest neighbor and thus has radius rk; the outer hyper-ball extends to distance $( 1 + \varepsilon ) r _ { k } .$ Here, 1 + ε is the smallest factor for which extensively searching the outer hyper-ball visits the true k-nearest neighbors with probability equal to the recall target. As the dataset grows, the distances from the query to the points in its neighborhood shrink by a common factor. The inner and outer hyper-balls therefore contract by the same factor, leaving their radii ratio (i.e., 1 + ε) unchanged (we return to this point later in Discussion 2). Henceforth, we define the exploration phase precisely as the distance computations the search performs within the outer hyper-ball, and the shortcut phase as the rest. The cost of the exploration phase is governed by the number of points inside the outer hyper-ball. Hence, the key question is: how does the number of points in the outer hyper-ball grow with the dataset size N?

![](images/bba28175c9d470feb46c51e6042e89f66c5392a077701ecc4ad2344187c7e654.jpg)  
Figure 6: (a) The data is not globally uniform, but the query q's neighborhood is roughly uniform on the manifold. (b) The k-NNs of q lie within radius rk; expanding this radius by a factor (1 + ε) captures about $k ( 1 + \bar { \varepsilon } ) ^ { d _ { \mathrm { i n t } } }$ points.

Local Uniformity on a Manifold. To answer the question above, we adopt a standard assumption from the literature: real embedding datasets exhibit complex structure at the global scale, yet if we zoom closely enough into any subspace, the distribution within it is approximately uniform [5, 84]. More precisely, real-world datasets tend to concentrate across lower-dimensional manifolds. Within a sufficiently small neighborhood around the query, the data behaves as if its density were approximately uniform [40, 62, 66]. The intuition is that real-world data items vary along continuous attributes, such as the visual style of an image or the topical blend of a document. Embedding models are trained to map similar data items to nearby vectors, so the variation carries over to the vector space, where density changes gradually rather than abruptly. Within a small enough neighborhood, points can therefore be treated as uniform on their manifold. For example, in Figure 6, the dataset forms a complex manifold in three dimensions, yet the subspace we zoom into is two-dimensional with approximately uniform points. We thus model the points in both the inner hyper-ball, which contains the nearest neighbors, and the outer hyper-ball as uniformly distributed around the query; we call this the local uniformity model. Figure 9 later confirms that the outer hyper-ball is indeed local: the search rarely goes far beyond the nearest neighbors.

Assumptions. Beyond the local uniformity model, our analysis in this section adopts three assumptions. (1) The query is drawn from the same distribution as the data. (2) The dataset is dense enough that the hyper-balls' projections onto the manifold are themselves approximately lower-dimensional hyper-balls. (3) The distance metric is Euclidean. Discussion 3 revisits what happens when each of these assumptions is dropped.

Dimensionality and Ball Volume. Under the above assumptions, the cost of the exploration phase is proportional to the number of points inside the outer hyper-ball (we revisit this point in Section $4 . 2 ) . ^ { 1 1 }$ The number of such points is proportional to the outer hyper-ball's volume on the manifold. The volume, in turn, is governed by the manifold's dimensionality, i.e., the intrinsic dimensionality near the query point $( d _ { \mathrm { i n t } } ) ,$ rather than by the ambient dimensionality of the embedding space $( d _ { \mathrm { a m b } } )$ The following theorem counts the points inside the outer hyper-ball in terms of its volume on the manifold.

Table 3: The order of the query cost falls as N grows.
<table><tr><td></td><td colspan="2">SPARSE:  $d _ { \mathrm { i n t } } = \Theta ( \log N )$ </td><td colspan="2">DENSE: dint = o(log N)</td></tr><tr><td>N:</td><td> $O ( M ) ^ { * }$ </td><td> $2 ^ { \Theta ( d _ { \mathrm { i n t } } ) }$ </td><td> $2 ^ { 2 ^ { \Theta ( d _ { \mathrm { i n t } } ) } }$ </td><td>→</td></tr><tr><td>Shortcut</td><td>O(1)</td><td> $N ^ { \Theta ( 1 ) }$ </td><td> $N ^ { o ( 1 ) }$ </td><td>Θ(poly log  $N ) ^ { \dagger }$ </td></tr><tr><td>Exploration Θ(N)</td><td></td><td> $N ^ { \Theta ( 1 ) }$ </td><td> $N ^ { o ( 1 ) }$ </td><td>O(poly log N)</td></tr><tr><td>Total</td><td>Θ(N)</td><td> $N ^ { \Theta ( 1 ) }$ </td><td> $N ^ { o ( 1 ) }$ </td><td>Θ(poly log N)</td></tr></table>

$^ { * } \operatorname { F o r } M = 2 ^ { o ( d _ { \mathrm { i n t } } ) }$ † For some graph-based indexes, these expressions are lower bounds on the cost that may not be $\mathrm { \ t i g h t } .$ For example, as argued in Section 4.2.2, SNG's cost is at least $N ^ { ( 1 - o ( 1 ) ) / \dot { d } _ { \mathrm { i n t } } }$ , which is larger than the poly-logarithmic term in the right column. Nevertheless, the table's expressions are tight for the exactly constructed version of Vamana, intuitively because it behaves like a skip list.

Theorem 1 (SCALING WITH INTRINSIC DIMENSIONALITY). Let $B ( q , r )$ be the hyper-ball of radius r around query q on the manifold, and let $C ( r )$ be the number of data points inside B(q, r). For any radius r and any ε such that the local uniformity model holds within $B ( q , ( 1 + \varepsilon ) r )$

$$
C ( ( 1 + \varepsilon ) r ) \approx ( 1 + \varepsilon ) ^ { d _ { \mathrm { i n t } } } C ( r ) .
$$

In particular, $i f r _ { k }$ is the query's distance to its k-th nearest neighbor, then the expected number of points inside the outer hyper-ball is

$$
C ( ( 1 + \varepsilon ) r _ { k } ) \approx k ( 1 + \varepsilon ) ^ { d _ { \mathrm { i n t } } } .
$$

PROOF. A $d _ { \mathrm { i n t } }$ -dimensional hyper-ball has volume proportional to rdint [8]. By assumption, the local uniformity model holds within $B ( q , ( 1 + \varepsilon ) r )$ , and $B ( q , r )$ is contained in it, so the expected number of points in each of the two hyper-balls scales with its volume on the manifold (Figure 6-b). Specifically, we have

$$
\frac { C ( ( 1 + \varepsilon ) r ) } { C ( r ) } \approx \frac { ( ( 1 + \varepsilon ) r ) ^ { d _ { \mathrm { i n t } } } } { r ^ { d _ { \mathrm { i n t } } } } = ( 1 + \varepsilon ) ^ { d _ { \mathrm { i n t } } } .
$$

Setting $r = r _ { k }$ gives C(rk) ≈ k, so C((1 + ε)rk) ≈ k(1 + ε)dint .

□

The base $1 + \varepsilon$ can be rewritten as $N ^ { \log ( 1 + \varepsilon ) / \log N }$ .Substituting this into the factor $( 1 + \varepsilon ) ^ { d _ { \mathrm { i n } } }$ t and simplifying the exponent yields the following:

Corollary 2 (SCALING WITH DATASET SIZE). Under the assumptions of Theorem 1, the expected number of data points inside the outer hyper-ball around the query can be written as

$$
C ( ( 1 + \varepsilon ) r _ { k } ) \approx k N ^ { \frac { d _ { \mathrm { i n t } } } { \log N } \log ( 1 + \varepsilon ) } .
$$

The exponent of N in Corollary 2 depends on $d _ { \mathrm { i n t } }$ through the ratio $d _ { \mathrm { i n t } } / \mathrm { l o g } N _ { \mathrm { : } }$ so applying it requires understanding how $d _ { \mathrm { i n t } }$ behaves as N grows.

LID. We use the local intrinsic dimensionality (LID) as our empirical proxy for $d _ { \mathrm { i n t } } ,$ as it is computationally feasible and standard in practice [1, 5, 40]. LID measures intrinsic dimensionality by how tightly the query's distances to its nearest neighbors concentrate. The more these distances concentrate around a common value, the higher the intrinsic dimensionality. We estimate LID using the maximum-likelihood estimator [5, 54]

$$
\widehat { \mathrm { L I D } } _ { k } ( q ) = - \left( \frac { 1 } { k } \sum _ { i = 1 } ^ { k } \log \frac { r _ { i } } { r _ { k } } \right) ^ { - 1 } .
$$

![](images/24e237297b745a90faaaebd71f62d758fcebc588383ed57928f8510e03770e8f.jpg)  
Figure 7: LID grows roughly with log $N ,$ as the sparse regime predicts. Two datasets deviate and move toward the dense regime. LID is measured over 500 random queries and six values of $k ;$ shaded bands mark the 25th-75th percentiles.

Here, $r _ { i }$ is the query's distance to its i-th nearest neighbor.

Figure 7 reports how LID changes with N across datasets and values of k. For most datasets, LID follows an approximately straight line when plotted against a logarithmic dataset size, meaning LID grows in step with log N. Meanwhile, the LID of OpenAI still increases, but with a gradually decreasing slope. The decreasing slope matches Figure 3, where some of OpenAI's curves deviate slightly from the Sublinear Power Law. SpaceV departs from the linear trend even more significantly than OpenAI: its LID rises at first but then almost flattens. This behavior of SpaceV is consistent with Figure 5, where it deviates from the Sublinear Power Law.

Sparse and Dense Regimes. In principle, $d _ { \mathrm { i n t } }$ can take any value from 1 to O(log N). The upper bound is motivated by the Johnson-Lindenstrauss (JL) lemma [24, 49]. The lemma states that any N points in Euclidean space can be embedded into a space with an ambient dimensionality of O(log N) while changing every pairwise distance by at most a constant factor. Since $d _ { \mathrm { i n t } }$ never exceeds $d _ { \mathrm { a m b } }$ [23] and a constant-factor change in pairwise distances changes $d _ { \mathrm { i n t } }$ by at most a constant factor [3, 39], the same bound holds for the original dataset.12 Motivated by this upper bound on $d _ { \mathrm { i n t } } .$ and guided by the LID curves in Figure 7, we define two regimes of interest. In the sparse regime, $d _ { \mathrm { i n t } }$ grows proportionally to log N, i.e., $d _ { \mathrm { i n t } } = \Theta ( \log N )$ . In the dense regime, dint grows more slowly than log N, i.e., $d _ { \mathrm { i n t } } = o ( \log N )$ . Later, in Discussion 1, we further substantiate these two regimes and analyze the transition between the two. We now apply Corollary 2 to analyze the search cost in each regime as N grows.

Sparse Regime: Power-Law. In the sparse regime, $d _ { \mathrm { i n t } } = \Theta ( \log N )$ SO $d _ { \mathrm { i n t } } = ( \log N ) / \rho$ for a constant $\rho = \Theta ( 1 )$ . Plugging this into Corollary 2, the expected number of points in the outer hyperball is

$$
C ( ( 1 + \varepsilon ) r _ { k } ) \approx k N ^ { \frac { d _ { \mathrm { i n t } } } { \log N } \log ( 1 + \varepsilon ) } = k N ^ { \log ( 1 + \varepsilon ) / \rho } \propto N ^ { c } .
$$

Notably, $c = \log ( 1 + \varepsilon ) / \rho$ is a constant independent of N, which explains the power-law scaling of search cost in the sparse regime. Dense Regime: Subpolynomial. In the dense regime, $d _ { \mathrm { i n t } } ~ =$ o(log N). Plugging this into Corollary 2, the expected number of

![](images/af59c287c7417603d38694a70c9ce5c7bbf1d449526dd5203d1eb73785881ed8.jpg)  
Figure 8: The total query cost falls from Θ(N) to Θ(poly log N) as N crosses the regimes of Table 3.

points in the outer hyper-ball is

$$
C ( ( 1 + \varepsilon ) r _ { k } ) \approx k N ^ { \frac { d _ { \mathrm { i n t } } } { \log N } \log ( 1 + \varepsilon ) } = k N ^ { o ( 1 ) } .
$$

The exponent $\frac { d _ { \mathrm { i n t } } } { \log N } \log ( 1 + \varepsilon )$ converges to 0 as $N  \infty ,$ so the number of points in the outer hyper-ball grows slower than any power of N, resulting in subpolynomial scaling.

Shortcut Phase. Although the exploration phase empirically accounts for most of the search cost, we analyze the shortcut phase to complete the unifying theory. Unlike the exploration phase, the shortcut phase stretches beyond the neighborhood of the query. Its cost therefore depends also on the global geometry of the dataset. Because of that, the shortcut phase's cost analysis assumes the data is uniformly distributed on a sphere. Our analysis is presented in Table 3's first row and the formal proof of our arguments resides in Section 4.2. When the dataset has fewer points than the graph out-degree $( N = O ( M ) )$ , the outer hyper-ball contains the entire dataset, so the search skips straight to the exploration phase and the shortcut phase costs $O ( 1 )$ . When the dataset is larger than the graph out-degree but has not yet saturated the underlying manifold (we later establish the saturation point is $2 ^ { \Theta ( d _ { \mathrm { i n t } } ) } )$ , the cost of the shortcut phase grows no faster than that of the exploration phase, as seen in the table. Once the dataset saturates the manifold, the shortcut phase costs Θ(poly log N), matching the guarantees established for fixed dimensionality [58, 84, 97].

Unifying Theory. We now unify the two search phases into a single account of total search cost for data uniformly distributed on a sphere, summarized in Table 3 and visualized in Figure 8. While $N = O ( M )$ , the outer hyper-ball contains the entire dataset and the search cost is Θ(N) (Corollary 2). As the dataset grows beyond O(M) points but stays at most exponential in the intrinsic dimensionality $( N = 2 ^ { O ( d _ { \mathrm { i n t } } ) } )$ , the cost follows a power law $N ^ { c }$ for $0 < c < 1$ . Once the dataset exceeds $2 ^ { O ( d _ { \mathrm { i n t } } ) }$ points, the cost becomes $N ^ { o ( 1 ) }$ . Beyond $2 ^ { 2 ^ { \Theta ( d _ { \mathrm { i n t } } ) } }$ points, the outer hyper-ball contains only O(poly log N) points, the shortcut phase therefore governs the scaling, and the cost is Θ(poly log N).

Discussion 1: Doubling Dimension. The literature offers several estimators for $d _ { \mathrm { i n t } } ,$ including the doubling dimension, the expansion dimension, and the local intrinsic dimensionality $[ 3 , 5 , 4 0 ] . ^ { 1 3 }$ We have already used the local intrinsic dimensionality to identify which regime the datasets of Table 2 fall into (Figure 7). We now turn to the doubling dimension, prevalent in theoretical work [36, 42]. Unlike LID, which models the dataset as i.i.d. draws from an underlying smooth distribution, the doubling dimension is defined for any dataset as given, so guarantees proven in terms of it hold for a broader class of inputs

![](images/b5024f82e8bab217aab98f9726b603fe81a8691c20b7bd82e4a070d574c6029f.jpg)  
Figure 9: Visited-node distance distributions for HNSW across three datasets and three dataset sizes at recall = 0.95 $( M = 3 2 , e f c = 2 0 0 , k = 1 0 )$ . Distances are normalized by the 10th-NN distance. Dashed lines indicate the normalized distance containing 80% of distance computations.

Consider a hyper-ball centered at any dataset point with any radius R. Find the smallest number of non-empty hyper-balls of radius $R / 2$ that can cover all the data points within this hyperball. The largest such value over all dataset points and radii R is the doubling constant λ, and the doubling dimension is defined as $\log _ { 2 } \lambda$ . Computing the value of λ exactly is NP-hard [35], so we use the doubling dimension only in our theoretical analysis. Theorem 3 places the data in the sparse regime whenever N is at most exponential in the dimensionality of the underlying manifold.

Theorem 3 (LOGARITHMIC SCALING OF THE DOUBLING DIMENSION). Let P consist $o f N = 2 ^ { O ( d ) }$ points drawn i.i.d. uniformly from $\mathbb { S } ^ { d - 1 }$ With high probability, the doubling dimension of P is Θ(log N).

ProoF. Fix a constant c with $N \leq 2 ^ { c d }$ and set $\begin{array} { r } { \varepsilon = 2 ^ { - ( c + 1 ) } \leq \frac { 1 } { 2 } } \end{array}$ Call a pair of points bad if its distance is at most $\varepsilon . \ A$ fixed pair is bad only if the second point lands in a spherical cap of radius ε around the first, which occupies at most an $\varepsilon ^ { d - 2 }$ fraction of the sphere. The expected number of bad pairs is thus at most $N ^ { 2 } \varepsilon ^ { d - 2 } \leq$ $\mathring { N } \cdot 2 ^ { c d } \cdot 2 ^ { - ( c + \hat { 1 } ) ( d - 2 ) } = 2 ^ { 2 c + 2 - d } N .$ , so by Markov's inequality, with probability at least $1 - 2 ^ { 2 c + 3 - d } = 1 - \overset { \cdot } { 2 } ^ { - \Omega ( d ) }$ there are fewer than $N / 2$ bad pairs. In that case, remove one point of each bad pair and let S be the set of remaining points; then $| S | > N / 2 ,$ and any two points of S are at distance greater than ε.

Every ball can be covered by λ balls of half its radius. Start with a single ball of radius 2, which covers all of P, and halve k times; this covers P by at most $\lambda ^ { k }$ balls of radius $2 ^ { 1 - k }$ . After $k = \lceil \log _ { 2 } ( 4 / \varepsilon ) \rceil \leq c + 4$ halvings, each ball has diameter at most ε, so it contains at most one point of S. Since all of S must be covered, $\lambda ^ { k } \ge | S | > N / 2$ , hence dim $( P ) \ge ( \log _ { 2 } N - 1 ) / ( c + 4 ) = \Omega ( \log N )$ Also, any ball is covered by the N balls of half its radius centered at the points, so $\lambda \leq N$ and dim ${ \mathit { \Omega } } ^ { \prime } P ) = O ( \log N )$ □

Regime Change. As long as the manifold close to a query point is undersampled, each new point can locally reveal a new geometric direction of the manifold. The growing intrinsic dimensionality of Theorem 3 therefore reflects newly added points progressively resolving the manifold. Once N is large enough to sample every neighborhood densely, new points reveal no new direction, and the intrinsic dimensionality stabilizes. Theorem 3 also suggests where the transition from the sparse to the dense regime occurs: the logarithmic growth of $d _ { \mathrm { i n t } }$ only holds while N is at most exponential in the manifold's dimensionality, and beyond that scale the data enters the dense regime. This aligns with the results of prior work [71], which shows that resolving the structure of a manifold requires dataset size exponential in the manifold's dimensionality.

Discussion 2: ε Interpretation. As noted earlier in this section, growing the dataset rescales the inner and outer hyper-balls by the same factor. The value of ε should therefore stay stable as the dataset grows. We report whether this holds on three datasets from Table 2; the others exhibit a similar pattern. For each dataset we record every node v that the beam search visits, i.e., every node for which it computes the distance to the query. For each visited node we then compute the normalized distance to the query point: $d ( q , v ) / r _ { k }$ with $k ~ = ~ 1 0$ . Figure 9 shows the histogram of these normalized distances and reveals how far beyond the true nearest neighbors the search explores to find them.

In the top row, in each bucket of the normalized distance, we report the fraction of dataset points that the search process visits. For each dataset, this profile nearly coincides across dataset sizes. The fraction of points the search visits at each normalized distance is therefore scale-invariant. As a result, the outer hyperball rescales similarly with the inner hyper-ball, and ε stays stable as the dataset grows.

The bottom row shows where the search process spends most of its effort, reporting the absolute number of visited nodes, i.e. distance computations, per bucket. At every dataset size, the cost concentrates in a band of near-ties just beyond the inner hyperball. The vertical dashed line marks the normalized distance below which 80% of the distance computations occur, and this threshold stays small at every scale.

Discussion 3: Relaxing the Assumptions. We now revisit the three assumptions behind our analysis and examine what changes when they are relaxed. The first assumption is that queries are drawn from the same distribution as the data. When this does not hold, as in out-of-distribution workloads, the outer hyper-ball may intersect multiple manifolds [16]. In this case, the growth of the exploration phase's cost is governed by the largest intrinsic dimensionality among the intersected manifolds.

The second assumption is that the dataset is sufficiently dense. This ensures that the outer hyper-ball remains small relative to the scale at which the manifold bends. When the dataset is sparse, however, the projection of the hyper-ball onto the manifold may no longer resemble a hyper-ball and can instead take more irregular shapes. This changes the number of points contained in the hyper-ball and, consequently, affects Corollary 2. Nevertheless, the corollary continues to hold as long as the intrinsic dimensionality realized at each scale follows the same trend across the sparse and dense regimes. The same argument applies when the local uniformity model is violated: our analysis remains valid as long as this trend in intrinsic dimensionality is preserved.14

Finally, we assume that the distance metric is Euclidean. If an angular metric is used instead, Euclidean distance between l2- normalized vectors is a strictly increasing function of the angle, so the two metrics induce the same ordering of distances from any query point and, in particular, the same nearest neighbors [44, 88]. Even more strongly, the two metrics are locally essentially identical; the outer hyper-ball around the query is therefore unaffected by the choice of metric, and the analysis carries over without modification.

## 4.2 Theoretical Analysis

In Section 4.1, we estimated the search cost of graph-based indexes by empirically tying the search cost to the number of points in the query's neighborhood. In this section, we rigorously establish the same cost bounds by also accounting for the effect of these graphs edges on the search. We analyze the costs of the shortcut and exploration phases in the sparse regime (Section 4.2.1), followed by the dense regime (Section 4.2.2). Together, these analyses establish the bounds of Table 3 and the relevant transition thresholds.

Setting. To streamline our exposition, we focus on SNG [2] in our proofs, and later argue how the same proofs can be adapted to other graph-based indexes such as HNSW [64] and Vamana [97]. To provide the strongest result possible, we study the query cost of the exact version of SNG and allow the construction to run free of charge. We also discuss how the same results hold for the practical versions of the graph-based indexes in the sparse regime, as long as they are allowed to compute the highest quality neighbors for each node using a candidate list of unbounded size $( \mathrm { i . e . , } e f _ { \mathrm { c o n s t r u c t i o n } } = \infty )$

We analyze the exploration phase under the local uniformity model, where the points within a local neighborhood of a query are approximately uniformly distributed. Since the shortcut phase depends on the global geometry of the data, one cannot prove a non-trivial bound for it in the local uniformity model without additional assumptions. Thus, we restrict the shortcut phase's analysis to datasets whose points are uniformly distributed on a sphere. Henceforth, we refer to the costs of these two phases in their respective settings as the locally uniform exploration and uniform shortcut costs.

4.2.1 Sparse Regime. In the sparse regime, $d _ { \mathrm { i n t } } = \Theta ( \log N )$ . As discussed in Section 4.1, such a value of $d _ { \mathrm { i n t } }$ causes the number of points within the approximately uniform neighborhood of a query point to be $N ^ { c } ,$ where N is the dataset size and c ∈ (0, 1] is a constant. We now prove that SNG's locally uniform exploration cost grows as a power law in the size of such a neighborhood, yielding a Sublinear Power Law in N. We then show that the same argument implies that SNG's uniform shortcut cost grows similarly.

Reduction to M-NN Graphs or High-Degree Graphs. The central part of our argument shows that, when using SNG to index a set of uniformly distributed points, one of two scenarios occurs: (1) if the maximum degree M is small, SNG connects each node to its M nearest neighbors with high probability, and (2) if M is large, the average node degree follows a power law in N. In the former case, SNG behaves similarly to an M-NN graph, which is known to have a power-law search cost [53, 84]. In the latter case, navigating a single node entails computing the distances of a power-law number of neighboring nodes to the query. Together, these cases imply that the uniform shortcut cost and the locally uniform exploration cost follow Sublinear Power Laws in N, giving rise to the leftmost two columns of Table 3.

Connection Probability. To prove the above results, we begin by bounding the probability of each node connecting to its k-th nearest neighbor:

LEMMA 4 (CONNECTION PROBABILITY). Consider a node u within a region of the space where points are (approximately) uniformly distributed. Denoting u's k-th nearest neighbor as ${ { v } _ { k } }$ and letting $p ^ { * } = d _ { \mathrm { i n t } } ^ { \Theta ( 1 ) } \cdot 2 ^ { - { d _ { \mathrm { i n t } } } / 2 } = \log ^ { \Theta ( 1 ) }$ N·N−c/2, u is connected to $v _ { k }$ with a probability of at least $1 - 2 \cdot ( k - 1 ) \cdot p ^ { * }$ when $k \leq M .$

ProoF. We observe that the exact version of SNG connects u to its M nearest neighbors that are not pruned. Recall that a node vk is pruned if u is already connected to some other node $v _ { i }$ with $i < k$ that satisfies $d ( u , v _ { k } ) \geq d ( v _ { i } , v _ { k } )$ . This corresponds to the neighbor vi being included in the "lune" between u and ${ \boldsymbol { v } } _ { k }$ as shown in Figure 1-b.

Thus, denoting the lune between u and ${ \boldsymbol { v } } _ { k }$ by $L _ { k } , v _ { k }$ is pruned with a probability of at most

$$
\begin{array} { r l } { \operatorname* { P r } [ v _ { k } \mathrm { ~ p r u n e d } ] = \operatorname* { P r } [ \exists v _ { i } , i < k : v _ { i } \in L _ { k } ] } & { } \\ & { \qquad \le \displaystyle \sum _ { i = 1 } ^ { k - 1 } \operatorname* { P r } [ u \mathrm { ~ c o n n e c t e d ~ t o ~ } v _ { i } \wedge v _ { i } \in L _ { k } ] } \\ & { \qquad \le \displaystyle \sum _ { i = 1 } ^ { k - 1 } \operatorname* { P r } [ v _ { i } \in L _ { k } ] . } \end{array}
$$

Now, we observe that the lune $L _ { k }$ is comprised of two spherical caps of height $d ( u , v _ { k } ) / 2$ The probability of a uniformly distributed point $v _ { i }$ with $i < k$ being within such a spherical cap is at most the ratio of the volume of this cap to the volume of the hyper-ball of radius $d ( u , v _ { k } )$ around u. This quantity is known to be $p ^ { * } =$ $d _ { \mathrm { i n t } } ^ { \Theta ( 1 ) } \cdot 2 ^ { - d _ { \mathrm { i n t } } / 2 }$ [53, Lemma 2]. Thus, the probability that $v _ { i }$ is within the lune $L _ { k }$ is at most $2 \cdot p ^ { * }$ . Plugging this expression into the sum above yields $\mathrm { P r } [ v _ { k }$ pruned] $\begin{array} { r } { \leq \sum _ { i = 1 } ^ { k - 1 } 2 \cdot p ^ { * } = 2 \cdot ( k - 1 ) \cdot p ^ { * } } \end{array}$ . Finally, noting that u must be connected to $v _ { k } { \mathrm { ~ i f ~ } } v _ { k }$ is not pruned yields

Pr[u connected to $v _ { k } ] = 1 - \operatorname* { P r } [ v _ { k }$ pruned] $\geq 1 - 2 \cdot ( k - 1 ) \cdot p ^ { * }$ establishing the lemma. □

Lemma 4 has two key consequences in the following two cases of the maximum degree M:

Case (1) of Small M. This case leads to a large subgraph of SNG being an M-NN graph with high probability within a region of the space where the dataset is approximately uniformly distributed:

Theorem 5 (M-NN SuBGRAPH). Consider a point x around which the dataset is approximately uniformly distributed. The $n = 1 / ( 2$ $M ^ { 2 } \cdot p ^ { * } ) = { N ^ { c / 2 } } / ( 2 \cdot M ^ { 2 } \cdot \log ^ { \Theta ( 1 ) } \stackrel { . . } { N } )$ nearest points to x are connected to their M nearest neighbors with a probability of at least $1 / 2$

ProoF. By applying a union bound to the connection probability of Lemma 4, it follows that SNG connects a node u to all of its M nearest neighbors with a probability of at least $1 - M ^ { 2 } \cdot p ^ { * } = 1 -$ $M ^ { 2 } \cdot \log ^ { \Theta ( 1 ) } N \cdot N ^ { - c / 2 }$ . Through another union bound, this implies that all of the $n = N ^ { c / 2 } / ( 2 \cdot M ^ { 2 } \cdot \log ^ { \Theta ( 1 ) } N )$ nearest points to x must be connected to their M nearest neighbors with a probability of at least $1 / 2 .$ □

When $M = N ^ { o ( 1 ) }$ , Theorem 5 implies that in the subgraph of SNG restricted to the $N ^ { c / 2 }$ -o(1) nearest points of x, denoted by ${ \overline { { G } } } ,$ each node is connected to at most M of its nearest neighbors. Since the points in $\overline { G }$ are uniformly distributed, this intuitively implies that each of its nodes is connected to at most M other nodes that are “random," meaning that they may or may not aid in navigating closer to a query. In such a scenario, the search must traverse many nodes and compare the distances of many points to the query to find its nearest neighbor with a non-trivial probability.

This intuition is formalized in [53], where the author shows that using greedy search on an M-NN graph¹⁵ to reach a node that is at most a constant factor farther than a query's true nearest neighbor entails a number of distance computations that follows a Sublinear Power Law in the graph's size.16A slight adaptation of their proof would extend their result to any graph where each node is connected to some number of its nearest neighbors that is less than or equal to M

To analyze SNG's uniform shortcut cost, we consider the subgraph $\overline { { G } } _ { \mathrm { e n t r y } }$ around SNG's entry point and apply Theorem 5. Then, by invoking the result of [53] for $\overline { { G } } _ { \mathrm { e n t r y } } ,$ we derive the cost as a power law in the size of the entry point's neighborhood, i.e., $N ^ { c }$ for some constant $c \in ( 0 , 1 ]$ . As discussed at the beginning of this section, this corresponds to a Sublinear Power Law in N. Similarly, to analyze SNG's locally uniform exploration cost, we consider the subgraph $\overline { { G } } _ { q }$ around the query q and leverage the approximately uniform distribution of its points to apply Theorem 5. This allows us to once again invoke the result of [53] to show that the cost follows a power law in N.

Case (2) of Large M. In this case, each node within the approximately uniform neighborhood of the query q has a high degree with high probability:

Theorem 6 (HIGH-DEGREE GRAPH). When the dataset is uniformly distributed and $M = N ^ { \Theta ( 1 ) }$ , each node within the uniform neighborhood of a query q has a degree of at least min(M, $N ^ { c / 4 - o ( 1 ) } ) = \overline { { N } } ^ { \Theta ( 1 ) }$ with a probability of at least $1 / 2$

PROOF. If $M = N ^ { \Theta ( 1 ) } < N ^ { c / 4 } / ( 2 \cdot \log ^ { \Theta ( 1 ) } N )$ , a similar union bound to that of Theorem 5 applied to the connection probability of Lemma 4 shows that any node u will be connected to its $M = N ^ { \dot { \Theta } ( 1 ) }$ nearest neighbors with a probability of at least $1 / 2 . \mathrm { I f } M \ge N ^ { c / 4 } / ( 2$ $\log ^ { \Theta ( 1 ) } N )$ , by the same union bound, any node u will be connected to its $N ^ { c / 4 } / ( 2 \cdot \log ^ { \Theta ( 1 ) } N )$ nearest neighbors with a probability of at least $1 / 2 .$ □

Theorem 6 implies that, for both the uniform shortcut cost and the locally uniform exploration cost, processing even a single node entails a number of distance computations that follows a power law in N.

As such, the results in the two columns on the left of Table 3 hold in both cases of M. Note that the uniform shortcut cost in the leftmost column is constant since, in that setting, Theorem 5 implies that all nodes are connected to each other, so the search begins in the exploration phase and skips the shortcut phase.

Extension to Other Graph-based Indexes. The exact versions of the major graph-based indexes, such as HNSW, Vamana, and $\tau { \mathrm { - } } \mathrm { M G } $ employ pruning rules at least as strict as that of SNG. As such, an analogue of Lemma 4 holds for them with a probability bound that is at least as high. This implies that the same conclusions as those of Theorems 5 and 6 hold for these graphs, as these results only depend on the bounds of Lemma 4.

For the practical versions of these graphs, when allowed to find the highest quality neighbors of each node using an unbounded candidate list $( \mathrm { i . e . , } e f _ { \mathrm { c o n s t r u c t i o n } } = \infty )$ , the same Sublinear Power Law results hold.17 The reason is two-fold: (1) When the maximum degree M is small, the same analysis as Lemma 4 shows that, with high probability, each node after the first M nodes is connected to the M nearest nodes from those that were inserted earlier. This does not affect our use of the results in [53], as it only slightly alters the probability of the search encountering a local minimum. (2) When M is large, Lemma 4 applied to the N/2 points inserted later during construction implies that their degrees must be asymptotically as large as that of Theorem 6. Thus, processing even one of these nodes, which happens with a constant probability, incurs a power-law cost.

4.2.2 Dense Regime. We now turn to the dense regime and analyze the uniform shortcut and locally uniform exploration costs. We begin by showing that the node degree is negligible in this regime, which leaves the bulk of the search costs to the number of steps. We lower bound the number of shortcut steps by analyzing the average progress each step makes towards the query. For the locally uniform exploration cost, we simply bound the number of points within a query's neighborhood.

Node Degree. As proven in [42, Lemma 3.3], the degree of a node in Vamana is at most $O ( 2 ^ { \Theta ( d _ { \mathrm { i n t } } ) }$ · poly log $N ) . ^ { 1 8 }$ We further observe that applying the same lemma with a value of $\alpha = 1$ shows that the degree of each node in SNG or HNSW is at most $O ( 2 ^ { \Theta ( d _ { \mathrm { i n t } } ) } )$ . Since, as described in Section $4 . 1 , d _ { \mathrm { i n t } } = o ( \log N )$ in the dense regime, we get the following lemma:

LEMMA 7 (NEGLIGIBLE DEGREE). In the dense regime, the degree of any node in SNG, HNSW, or Vamana is at most $O ( 2 ^ { \Theta ( d _ { i n t } ) }$ $\bar { p } o l y \bar { \log { N } } ) = N ^ { o ( \log { N } ) / \log { N } }$ · polylog $N = N ^ { o ( 1 ) }$

Lemma 7 posits that processing a single node, be it in the shortcut phase or the exploration phase, takes $N ^ { o ( 1 ) }$ time. In particular, when $N \geq 2 ^ { 2 ^ { d _ { \mathrm { i n t } } } }$ , we have $d _ { \mathrm { i n t } } = O ( \log \log N )$ , which implies that the degree of each node is at most $O ( 2 ^ { O ( \log \log N ) } ) = O ( \mathrm { p o l y l o g } N )$ This establishes that the uniform shortcut and locally uniform exploration costs are at least as large as the expressions in the right two columns of Table 3. We now discuss upper bounds on these costs by analyzing the number of nodes processed in each phase.

Uniform Shortcut Cost. For Vamana, it is known that each step of the shortcut phase takes the search closer to any query q by a constant factor [42]. From [53, Lemma 2], we observe that the nearest neighbor of any query q is within an expected distance of $N ^ { - \Theta ( 1 / d _ { \mathrm { i n t } } ^ { - } ) }$ of it, implying that the outer hyper-ball around q must have a radius of $\hat { O ( N ^ { - \Theta ( 1 / d _ { \mathrm { i n t } } ) } } )$ . Thus, since the distance from the graph's entry point to q is at most a constant when the points are distributed on a sphere, the number of shortcut steps is at most $\begin{array} { r } { O ( \log \frac { O ( 1 ) } { O ( N ^ { - \Theta ( 1 / d _ { \operatorname* { i n t } } ) } ) } ) = \Theta ( ( 1 / d _ { \operatorname* { i n t } } ) } \end{array}$ · poly log N). Multiplying by each node's degree, one derives upper bounds matching the expressions in the two columns on the right of Table 3.

Compared to Vamana, upper bounding SNG and HNSW's uniform shortcut costs is more complex. In fact, we show the number of steps in SNG's shortcut phase is at least $\Omega ( N ^ { - 1 / d _ { \mathrm { i n t } } } )$ , i.e., significantly higher than Vamana's poly-logarithmic number of steps. Our argument quantifies the average progress each search step makes:

LEMMA 8 (SNG MAXIMUM PROGRESs). In SNG's shortcut phase on a dataset of N points uniformly distributed on a sphere, with high probability, all steps that take the search closer to a query q reduce the distance to q by at most poly( $\begin{array} { r } { d _ { \mathrm { i n t } } ) \cdot \left( \frac { \log N } { N } \right) ^ { 1 / d _ { \mathrm { i n t } } } } \end{array}$

ProoF. Our argument analyzes the maximum length of an edge in SNG. Recall that if two nodes u and v are connected, then there must be no other node in their lune. By [53, Lemma 2], the probability that a node w is within u and v's lune is $\mathrm { p o l y } ( d _ { \mathrm { i n t } } ) \cdot d ( u , v ) ^ { d _ { \mathrm { i n t } } }$ When $\begin{array} { r } { d ( u , v ) = \left( \frac { \log N } { N } \right) ^ { 1 / { d _ { \mathrm { i n t } } } } } \end{array}$ , this probability becomes poly $( d _ { \mathrm { i n t } } )$ ${ \frac { \log N } { N } } , { \mathrm { i . e . } }$ , the expected number of nodes within the lune is $\Omega ( \log N )$ By applying a standard Chernoff bound, it follows that such a lune is empty with a probability of at most $1 / N ^ { 3 }$ . Thus, by a union bound over all pairs of nodes, we can conclude that all edges within SNG have a length of at most poly $( d _ { \mathrm { i n t } } ) \cdot \left( { \frac { \log N } { N } } \right) ^ { 1 / d _ { \mathrm { i n t } } }$ with high probability. Therefore, any edge that brings the search closer to a query must also be at most this long. □

As discussed above, the radius of a query's outer hyper-ball is $O ( N ^ { - 1 / d _ { \mathrm { i n t } } } )$ in expectation. Thus, since the distance from the entry point to a query can be a constant number, Lemma 8 implies that the shortcut phase requires at least $\begin{array} { r } { \frac { U ( 1 ) } { \mathrm { p o l y } ( d _ { \mathrm { i n t } } ) \cdot ( \log N / N ) ^ { 1 / d _ { \mathrm { i n t } } } } = } \end{array}$ $\Omega ( N ^ { ( 1 - o ( 1 ) ) / d _ { \mathrm { i n t } } } )$ many steps to reach the outer hyper-ball. The actual number of steps required to reach this hyper-ball may be larger than this quantity, since beam search may run into local minima which prevent it from making progress towards a query.

Since each layer of HNSW behaves similarly to SNG, similar concerns around an upper bound on its search cost persist. We leave deriving a tight upper bound for SNG and HNSW as an interesting open question for future work.

Locally Uniform Exploration Cost. The exploration phase in any graph-based index searches at most all nodes within the locally uniform neighborhood of a query. Thus, the number of steps in the exploration phase is upper bounded by the number of nodes in this neighborhood. Since $d _ { \mathrm { i n t } } = o ( \log N )$ in the dense regime, Theorem 1 implies that the number of nodes in the query's uniform neighborhood is at most $2 ^ { d _ { \mathrm { i n t } } } = N ^ { d _ { \mathrm { i n t } } / \log N } = N ^ { o ( 1 ) }$ . Multiplied by the degree of a node (quantified in Lemma 7), this yields a locally uniform exploration cost of $N ^ { o ( 1 ) }$ , matching the third column of Table 3. Moreover, when $N \geq 2 ^ { \Theta ( d _ { \mathrm { i n t } } ) }$ , we have $d _ { \mathrm { i n t } } = O ( \log \log N )$ making the number of nodes in the query's neighborhood polylogarithmic in N. Multiplied by the node degree (Lemma 7), which is poly-logarithmic in this case, we derive a poly-logarithmic cost for the exploration phase. Thus, the bound in the fourth column of Table 3 also holds.

## 5 Cost Modeling

The previous sections established that, across many datasets and scales, query cost follows a Sublinear Power Law in the dataset size N, with exponent c, and explained why. A natural next step is to predict the exponent, so that its scaling is known before the data grows. This section develops such models empirically for query cost (Section 5.1) and insertion cost (Section 5.2). Together, they cover the parameter ranges of practical interest and expose the trade-offs between query cost, insertion cost, and recall.

## 5.1 Modeling Query Cost

We begin with query cost, asking what determines the exponent c for a given dataset. Our approach is to model how c varies with the recall target and the construction parameters of the index, and then to connect the predicted cost back to the $e f _ { \mathrm { s e a r c h } }$ value that realizes it.

Parameters. We focus on three parameters that govern how c varies for a given dataset. The first is the recall target. A higher target leaves less room to miss a true nearest neighbor, so the search process must cover more of the local neighborhood and widen its candidate list. The other two are the construction list size $e f _ { \mathrm { c o n s t r u c t i o n } }$ and the maximum out-degree M, which together set the quality of the graph and the number of edges it carries. Those edges act as shortcuts that bring every query within fewer hops of its neighborhood, at the cost of more work during construction.

Expected Effects. Increasing recall should raise c, but the rise is not linear in recall. Raising recall from 0.90 to 0.92 barely makes ANN search harder, but raising it from 0.97 to 0.99 does. To capture the nonlinearity, we quantify the demand for recall as $1 / \delta ,$ where $\delta = 1 - \mathrm { r e c a l l }$ . In the previous example, the first increment raises $1 / \delta$ from 10 to 12.5, while the second raises it from 33.3 to 100. On the other hand, increasing the construction parameters should lower c, since a denser graph offers more reliable routes into and within the query's neighborhood. This benefit should diminish and eventually turn harmful: once the graph is dense enough, additional edges add little routing power while inflating the search cost per step. Across the parameter ranges we study (recall up to 0.99, efconstruction ∈ [100,800], $M \ \in \ [ 1 6 , 1 2 8 ] )$ , this reversal never appears, and we observe c decreasing throughout.

Modeling the exponent c. The simplest model consistent with these effects treats them as independent: each parameter contributes its own term, and the exponent c is their sum. Figure 10-a supports this decomposition empirically. Each column isolates the effect of one factor on the exponent $( 1 / \delta , e f _ { \mathrm { c o n s t r u c t i o n } } , M )$ . We subtract the fitted contributions of the other two, and plot what remains, the partial residual, against the factor in question. In each case the partial residual is approximately linear when the factor is shown on a logarithmic scale, so each factor contributes an additive logarithmic term to the exponent. We therefore model the scaling exponent as

Table 4: Fitted parameters and goodness of fit per dataset and algorithm. (a) Query cost model: coefficients of $\mathbf { E q } .$ 2 and the exponent $\beta$ of Eq. 3, with $R _ { c } ^ { 2 }$ and $R _ { \beta } ^ { 2 }$ the respective fits. (b) Build cost model: coefficients of Eq. 4.
<table><tr><td colspan="2"></td><td colspan="7">(a) Query cost model</td><td colspan="5">(b) Build cost model</td></tr><tr><td colspan="2">Dataset</td><td> $c _ { 0 }$ </td><td>α</td><td>Y1</td><td>Y2</td><td>β</td><td>R2</td><td> $R _ { \beta } ^ { 2 }$ </td><td></td><td>K β′</td><td> $c _ { 0 } ^ { \prime }$ </td><td> $\gamma _ { 2 } ^ { \prime }$ </td><td> $R ^ { 2 }$ </td></tr><tr><td colspan="2">sift</td><td>.244</td><td>.035</td><td>.012</td><td>.022</td><td>.762</td><td>.919</td><td>.992</td><td>2.497</td><td>.864</td><td>.031</td><td>.020</td><td>.979</td></tr><tr><td colspan="2">deep</td><td>.281</td><td>.054</td><td>.020</td><td>.022</td><td>.742</td><td>.934</td><td>.992</td><td>2.163</td><td>.880</td><td>.048</td><td>.018</td><td>.983</td></tr><tr><td colspan="2">spacev</td><td>.181</td><td>.067</td><td>.020</td><td>.017</td><td>.747</td><td>.818</td><td>.992</td><td>2.653</td><td>.900</td><td>.002</td><td>.024</td><td>.963</td></tr><tr><td colspan="2">MSNH wiki</td><td>.310</td><td>.079</td><td>.037</td><td>.022</td><td>.737</td><td>.886</td><td>.991</td><td>2.424</td><td>.869</td><td>.032</td><td>.018</td><td>.991</td></tr><tr><td colspan="2">glove</td><td>.336</td><td>.090</td><td>.036</td><td>.012</td><td>.759</td><td>.916</td><td>.993</td><td>2.147</td><td>.917</td><td>.052</td><td>.023</td><td>.977</td></tr><tr><td colspan="2">rand64</td><td>1.068</td><td>.022</td><td>.055</td><td>.082</td><td>.812</td><td>.912</td><td>.992</td><td>3.569</td><td>.681</td><td>.028</td><td>.044</td><td>.922</td></tr><tr><td colspan="2">gist</td><td>.513</td><td>.096</td><td>.045</td><td>.017</td><td>.705</td><td>.859</td><td>.992</td><td>1.268</td><td>.836</td><td>.155</td><td>.015</td><td>.991</td></tr><tr><td colspan="2">openai</td><td>.191</td><td>.079</td><td>.029</td><td>.001</td><td>.741</td><td>.910</td><td>.992</td><td>2.197</td><td>.858</td><td>.063</td><td>.019</td><td>.981</td></tr><tr><td colspan="2">sift</td><td></td><td>.046</td><td></td><td>.069</td><td>.752</td><td>.841</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2" rowspan="8">deep Vaana spacev</td><td>.415</td><td></td><td>.014</td><td></td><td></td><td></td><td>.992</td><td>3.622</td><td>.826</td><td>-.076</td><td>.041</td><td>.977</td></tr><tr><td></td><td>.479</td><td>.064</td><td>.021</td><td>.080</td><td>.730</td><td>.869 .825</td><td>.992</td><td>3.388</td><td>.843</td><td>-.064</td><td>.038</td><td>.973</td></tr><tr><td colspan="2">wiki</td><td>.351</td><td>.074</td><td>.021</td><td>.066</td><td>.739</td><td>.991 .992</td><td></td><td>4.253 .857</td><td>-.168</td><td>.055</td><td>.976</td></tr><tr><td colspan="2">glove</td><td>.447</td><td>.086</td><td>.036</td><td>.068</td><td>.742</td><td>.845</td><td></td><td>3.137</td><td>.859 一.</td><td>-.059 .037</td><td>.986</td></tr><tr><td colspan="2"></td><td>.480</td><td>.086</td><td>.027</td><td>.059</td><td>.760</td><td>.835 .993</td><td></td><td>3.977 .857</td><td>-.159</td><td>.063</td><td>.990</td></tr><tr><td colspan="2">rand64</td><td>1.252</td><td>.020</td><td>.059</td><td>.111</td><td>.833</td><td>.929 .993</td><td></td><td>5.239</td><td> $. 7 6 4 \mathrm { ~  ~ { ~ - . 2 3 3 } ~ }$ </td><td>.076</td><td>.996</td></tr><tr><td colspan="2">gist</td><td>.670</td><td>.108</td><td>.026</td><td>.098</td><td>.706</td><td>.782 .991</td><td></td><td>2.654.797</td><td>-.018</td><td>.046</td><td>.985</td></tr><tr><td colspan="2">openai</td><td>.334</td><td>.081</td><td>.025</td><td>.046</td><td>.714</td><td>.947 .991</td><td></td><td>4.492</td><td> $. 8 1 8 \mathrm { ~  ~ { ~ - . 1 7 7 } ~ }$ </td><td>.057</td><td>.983</td></tr></table>

$$
\begin{array} { r } { c = c _ { 0 } \ + \ \alpha \ln \bigl ( \frac { 1 } { \delta } \bigr ) \ - \ \gamma _ { 1 } \ \ln ( e f _ { \mathrm { c o n s t r u c t i o n } } ) \ - \ \gamma _ { 2 } \ \ln ( M ) . } \end{array}\tag{2}
$$

Here $c _ { 0 }$ is the baseline set by the dataset's geometry, α the cost of higher demand for recall, and $\gamma _ { 1 }$ and $\gamma _ { 2 }$ the benefits of $e f _ { \mathrm { c o n s t r u c t i o n } }$ and M, respectively.

Goodness of fit. We fit the model separately for each dataset and index construction algorithm. Table 4-a reports the fitted coefficients and goodness of fit. The model achieves an average $R ^ { 2 }$ of approximately 0.88, meaning it explains about 88% of the variation in the measured exponent across datasets and algorithms. The fitted coefficients are consistent with the slopes in Figure $^ { 1 0 - \mathbf { a } , }$ showing the additive log-linear structure directly on the data.

Effect of $e f _ { s e a r c \mathbf { h } } .$ The model above explains how fast the number of distance computations scales as the dataset grows, i.e., how large the exponent c is for a given recall level and construction parameters. It leaves open a practical question: for a dataset of a given size, how should the candidate list size during search, $e f _ { \mathrm { s e a r c h } } ,$ be tuned so that the search performs the distance computations predicted by Equation 2 and thus reaches the recall target? We answer this by observing that, for a given graph-based index, distance computations per query scale as a Sublinear Power Law in $e f _ { \mathrm { s e a r c h } } { \mathrm { . } }$

$$
\mathrm { d } \mathbf { c } / q \propto e f _ { \mathrm { s e a r c h } } ^ { \beta } .\tag{3}
$$

Figure 10-c shows this relationship on log-log axes, where the linear pattern indicates a power law, this time between $e f _ { \mathrm { s e a r c h } }$ and the number of distance computations. The fit is tight across all datasets, with $R ^ { 2 } > 0 . 9 8$ as reported in Table 4-a.

## 5.2 Modeling Insertion Cost

As discussed in Section 2, the insertion process runs a beam search to locate the new point's neighbors. This beam search performs the same distance computations that govern query cost, and it dominates the cost of insertion [64, 97]. We therefore measure the total number of distance computations during index construction, denoted $\mathrm { d c _ { b u i l d } }$ and study how the average insertion cost $\mathrm { d c } _ { \mathrm { b u i l d } } / N$ changes with the dataset size and the construction parameters.

![](images/8186139c6c7589792ea6e86bc46527a1153f4bd9c19a0be7a83043326705e32b.jpg)

![](images/11140cd2e20a6e6969a87c17e2d201da5bfdee3bf9b95b43e151429ada0de9c7.jpg)  
Figure 10: Partial regression plots for the query-, insertion-, and beam-width-cost models; each subfigure shows the partial residual after removing all other fitted terms, so the fitted slope equals the associated model coefficient. (a) Query-side model of Eq. 2: the effects of ln $( 1 / \delta ) , \ln ( e f _ { \mathrm { c o n s t r u c t i o n } } ) ,$ and ln M on the scaling exponent c. (b) Insertion-cost model of Eq. 4: the effects of ln( $\scriptstyle : f _ { \mathrm { c o n s t r u c t i o n } } ) ,$ ln $N ,$ and the interaction ln(M) ln(N) on average insertion cost $\ln ( \mathrm { d c } _ { \mathbf { b u i l d } } / N )$ . (c) Beam-width model of Eq. 3: $\ln ( e f _ { \mathrm { s e a r c h } } ) ~ { \bf v s . \ln ( \mathrm { d c } } / q )$ , centered within each configuration; the linear pattern confirms the power law with slope $\beta .$

Parameters. Since insertion runs the same beam search as a query, the insertion cost model shares the structure of the query cost models. The key difference is that insertion has no recall target, so recall demand drops out of the model. The parameters are the dataset size N and the two construction parameters, the graph out-degree M and the candidate list size efconstruction.

Expected Effects. A larger N means each new point searches a larger graph, so insertion cost should rise with the dataset, and since query cost rises with N as a Sublinear Power Law, we expect insertion cost to do the same; we empirically verify this power law. A larger $e f _ { \mathrm { c o n s t r u c t i o n } }$ makes each insertion explore more of the graph before identifying the new node's neighbors, so, just as query cost scales as $e f _ { \mathrm { s e a r c h } } ^ { \beta }$ (Equation 3), insertion cost should grow as a Sublinear Power Law in $e f _ { \mathrm { c o n s t r u c t i o n } } . \ A$ higher M forces the beam search to scan more candidates per step, but may also help it converge in fewer steps; the net effect is settled empirically.

Modeling the Scaling. Figure 10-b isolates each effect through partial regressions, as in Section 5.1. At fixed M, the average insertion cost is log-linear in both $e f _ { \mathrm { c o n s t r u c t i o n } }$ and $N ,$ as anticipated, verifying the power-law scaling on both counts. Raising M increases the power-law exponent of $N ,$ so a denser graph makes build cost grow faster with dataset size; M therefore belongs inside the exponent of N. We thus model

$$
\mathrm { d c } _ { \mathrm { b u i l d } } = e ^ { \kappa } ( e f _ { \mathrm { c o n s t r u c t i o n } } ) ^ { \beta ^ { \prime } } N ^ { c ^ { \prime } } , \qquad c ^ { \prime } = 1 + c _ { 0 } ^ { \prime } + \gamma _ { 2 } ^ { \prime } \ln M .\tag{4}
$$

Equivalently,

$$
\ln \left( \frac { \mathrm { d c } _ { \mathrm { b u i l d } } } { N } \right) = \kappa + \beta ^ { \prime } \ln ( e f _ { \mathrm { c o n s t r u c t i o n } } ) + c _ { 0 } ^ { \prime } \ln N + \gamma _ { 2 } ^ { \prime } \ln ( M ) \ln ( N ) .
$$

The primed symbols mark build-phase analogues of the query model's coefficients of Section 5.1. The build exponent $c ^ { \prime }$ explains how construction cost scales with the dataset size $N ,$ the buildphase analogue of the query exponent c. Within $c ^ { \prime } ,$ the leading 1 is the trivial cost of performing N insertions, $c _ { 0 } ^ { \prime }$ captures how perinsertion search cost grows with index size, and $\gamma _ { 2 } ^ { \prime }$ captures how a larger M steepens that growth. The intercept κ depends on the data geometry and the graph construction algorithm, and $\beta ^ { \prime }$ is the analogue of $\dot { \boldsymbol { { \beta } } }$ in Equation $^ { 3 , }$ the exponent of the candidate list size. Goodness of Fit. Table 4-b lists the fitted coefficients for HNSW and Vamana. The fits are tight $( R ^ { 2 } \geq 9 2 \% )$ , and the slopes we recover line up with the partial regressions in Figure 10-b. Except for the synthetic Rand64 dataset, $\beta ^ { \prime }$ sits in a fairly narrow band, 0.80–0.92, running 0.07–0.16 above $\beta .$ In other words, doubling $e f _ { \mathrm { c } }$ onstruction raises the average insertion cost by about 1.8×. Insertion is thus more sensitive to its candidate list size than query cost is to $e f _ { \mathrm { s e a r c h } }$ (Equation 3). This gap comes from the extra distance computations insertion has to do beyond plain beam search, mostly reverse-link rewiring. This rewiring is triggered whenever a candidate neighbor has already hit its degree budget. Rand64 breaks this pattern: insertion is less sensitive to candidate list size than search, with $\beta ^ { \prime }$ (0.68–0.76) staying below β (0.81–-0.83).

## 6 Conclusion

The literature has long held that, at fixed recall, the query cost of graph-based indexes scales poly-logarithmically. Our experiments and theory confine this claim to scales beyond a transition point. Below that point, cost follows a Sublinear Power Law, growing in step with $N ^ { c }$ , and most datasets in practice stay in this regime up to their full size. The power law appears across eight datasets, recall targets from 90% to 99%, hard and easy queries, and many configurations up to billion scale. Cost bends toward poly-logarithmic only once the dataset densely samples its underlying distribution. Our unifying theory explains both regimes through the intrinsic dimensionality the dataset exhibits at its current size, reconciling our measurements with the logarithmic claims as two regimes of the same scaling law. The resulting cost models let practitioners set index parameters in a principled way and sustain a recall target as the dataset grows.

## References

[1] Laurent Amsaleg, Oussama Chelly, Teddy Furon, Stéphane Girard, Michael E. Houle, Ken-ichi Kawarabayashi, and Michael Nett. 2015. Estimating Local Intrinsic Dimensionality. In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (Sydney, NSW, Australia) (KDD '15). Association for Computing Machinery, New York, NY, USA, 29–38. https://doi.org/10.1145/2783258.2783405

[2] Sunil Arya and David M. Mount. 1993. Approximate nearest neighbor queries in fixed dimensions. In Proceedings of the Fourth Annual ACM-SIAM Symposium on Discrete Algorithms (Austin, Texas, USA) (SODA '93). Society for Industrial and Applied Mathematics, USA, 271–280.

[3] Patrice Assouad. 1983. Plongements lipschitziens dans Rⁿ. Bulletin de la Société Mathématique de France 111 (1983), 429–448.

[4] Martin Aumüller, Erik Bernhardsson, and Alexander Faithfull. 2020. ANN-Benchmarks: A benchmarking tool for approximate nearest neighbor algorithms. Information Systems 87 (2020), 101374.

[5] Martin Aumüller and Matteo Ceccarello. 2021. The Role of Local Dimensionality Measures in Benchmarking Nearest Neighbor Search. Information Systems 101 (2021), 101807. https://doi.org/10.1016/j.is.2021.101807

[6] Artem Babenko and Victor Lempitsky. 2016. Efficient indexing of billion-scale datasets of deep descriptors. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2055-2063.

[7] Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, and Michael Auli. 2020. wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. arXiv:2006.11477 [cs.CL] https://arxiv.org/abs/2006.11477

[8] Keith Ball et al. 1997. An elementary introduction to modern convex geometry. Flavors of geometry 31, 1-58 (1997), 26.

[9] Dmitry Baranchuk, Matthijs Douze, Yash Upadhyay, and I Zeki Yalniz. 2023. Dedrift: Robust similarity search under content drift. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 11026-11035.

[10] Jon Louis Bentley. 1975. Multidimensional binary search trees used for associative searching. Commun. ACM 18, 9 (1975), 509–517.

[11] Kevin Beyer, Jonathan Goldstein, Raghu Ramakrishnan, and Uri Shaft. 1999. When is "nearest neighbor" meaningful?. In International conference on database theory. Springer, 217–235.

[12] Kevin S. Beyer, Jonathan Goldstein, Raghu Ramakrishnan, and Uri Shaft. 1999. When Is "Nearest Neighbor" Meaningful?. In Proceedings of the 7th International Conference on Database Theory (ICDT '99). Springer-Verlag, Berlin, Heidelberg 217-235.

[13] Christian Böhm. 2000. A cost model for query processing in high dimensional data spaces. ACM Transactions on Database Systems (TODS) 25, 2 (2000), 129– 178.

[14] Allan Borodin, Rafail Ostrovsky, and Yuval Rabani. 1999. Lower bounds for high dimensional nearest neighbor search and related problems. In Proceedings of the thirty-first annual ACM symposium on Theory of computing. 312-321.

[15] Karl Bringmann. 2021. Fine-grained complexity theory: Conditional lower bounds for computational geometry. In Conference on Computability in Europe. Springer, 60–70.

[16] Bradley CA Brown, Anthony L Caterini, Brendan Leigh Ross, Jesse C Cresswell, and Gabriel Loaiza-Ganem. 2022. Verifying the union of manifolds hypothesis for image data. arXiv preprint arXiv:2207.02862 (2022).

[17] T Tony Cai, Jianqing Fan, and Tiefeng Jiang. 2013. Distributions of angles in random packing on spheres. Journal of Machine Learning Research 14, 136 (2013), 1837–1864.

[18] Matteo Ceccarello, Alexandra Levchenko, Ioana Ileana, and Themis Palpanas. 2025. Evaluating and generating query workloads for high dimensional vector similarity search. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2. 5299–5310.

[19] Manos Chatzakis, Yannis Papakonstantinou, and Themis Palpanas. 2025. Darth: Declarative recall through early termination for approximate nearest neighbor search. Proceedings of the ACM on Management of Data 3, 4 (2025), 1–26.

[20] Patrick Chen, Wei-Cheng Chang, Jyun-Yu Jiang, Hsiang-Fu Yu, Inderjit Dhillon, and Cho-Jui Hsieh. 2023. Finger: Fast inference for graph-based approximate

nearest neighbor search. In Proceedings of the ACM Web Conference 2023. 3225– 3235.

[21] Yannis Chronis, Helena Caminal, Yannis Papakonstantinou, Fatma Özcan, and Anastasia Ailamaki. 2025. Filtered vector search: State-of-the-art and research opportunities. Proceedings of the VLDB Endowment 18, 12 (2025), 5488–5492.

[22] Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep neural networks for youtube recommendations. In Proceedings of the 10th ACM conference on recommender systems. 191–198.

[23] Sanjoy Dasgupta and Yoav Freund. 2008. Random projection trees and low dimensional manifolds. In Proceedings of the fortieth annual ACM symposium on Theory of computing. 537–546.

[24] Sanjoy Dasgupta and Anupam Gupta. 2003. An elementary proof of a theorem of Johnson and Lindenstrauss. Random Structures & Algorithms 22, 1 (2003), 60-65.

[25] Haya Diwan, Jinrui Gou, Cameron Musco, Christopher Musco, and Torsten Suel. 2024. Navigable graphs for high-dimensional nearest neighbor search: Constructions and limits. Advances in Neural Information Processing Systems 37 (2024), 59513–59531.

[26] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé Jégou. 2025 The Faiss library. arXiv:2401.08281 [cs.LG] https://arxiv.org/abs/2401.08281

[27] Hao Duan, Yitong Song, Bin Yao, and Anqi Liang. 2025. PGTuner: An Efficient Framework for Automatic and Transferable Configuration Tuning of Proximity Graphs. Proceedings of the ACM on Management of Data 3, 4 (2025), 1–27.

[28] Elastic. [n.d.]. Measuring the Recall of Quantized Vector Search. https://www. elastic.co/search-labs/blog/recall-vector-search-quantization. Elasticsearch Labs blog. Accessed: 2026-07-21.

[29] Elastic. 2024. Vector Search and kNN Implementation Guide. https://www.elastic.co/search-labs/blog/vector-search-implementationguide-api-edition Accessed: 2026-04-22.

[30] Charles Fefferman, Sanjoy Mitter, and Hariharan Narayanan. 2016. Testing the manifold hypothesis. Journal of the American Mathematical Society 29, 4 (2016), 983-1049.

[31] Cong Fu, Changxu Wang, and Deng Cai. 2021. High dimensional similarity search with satellite system graph: Efficiency, scalability, and unindexed query compatibility. IEEE Transactions on Pattern Analysis and Machine Intelligence 44, 8 (2021), 4139–4150.

[32] Cong Fu, Chao Xiang, Changxu Wang, and Deng Cai. 2025. Fast Approximate Nearest Neighbor Search With The Navigating Spreading-out Graph. arXiv:1707.00143 [cs.LG] https://arxiv.org/abs/1707.00143

[33] Jianyang Gao and Cheng Long. 2023. High-dimensional approximate nearest neighbor search: with reliable and efficient distance comparison operations. Proceedings of the ACM on Management of Data 1, 2 (2023), 1–27.

[34] Yunfan Gao, Yun Xiong, Xinyu Ġao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen Wang, and Haofen Wang. 2023. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997 2, 1 (2023).

[35] Lee-Ad Gottlieb and Robert Krauthgamer. 2013. Proximity algorithms for nearly doubling spaces. SIAM Journal on Discrete Mathematics 27, 4 (2013), 1759–1769.

[36] Anupam Gupta, Robert Krauthgamer, and James R Lee. 2003. Bounded geometries, fractals, and low-distortion embeddings. In 44th Annual IEEE Symposium on Foundations of Computer Science, 2003. Proceedings. IEEE, 534–543.

[37] Antonin Guttman. 1984. R-trees: a dynamic index structure for spatial searching. In Proceedings of the 1984 ACM SIGMOD International Conference on Management of Data (Boston, Massachusetts) (SIGMOD '84). Association for Computing Machinery, New York, NY, USA, 47–57. https://doi.org/10.1145/602259.602266

[38] Junfeng He, Sanjiv Kumar, and Shih-Fu Chang. 2012. On the Difficulty of Nearest Neighbor Search. arXiv:1206.6411 [cs.LG] https://arxiv.org/abs/1206.6411

[39] Juha Heinonen. 2001. Lectures on Analysis on Metric Spaces. Springer-Verlag, New York. https://doi.org/10.1007/978-1-4613-0131-8

[40] Michael E Houle. 2017. Local intrinsic dimensionality I: an extreme-valuetheoretic foundation for similarity applications. In International Conference on Similarity Search and Applications. Springer, 64–79.

[41] Piotr Indyk and Rajeev Motwani. 1998. Approximate nearest neighbors: towards removing the curse of dimensionality. In Proceedings of the thirtieth annual ACM symposium on Theory of computing. 604–613.

[42] Piotr Indyk and Haike Xu. 2023. Worst-case performance of popular approximate nearest neighbor search implementations: Guarantees and limitations. Advances in Neural Information Processing Systems 36 (2023), 66239–66256.

[43] Elias Jääsaari, Ville Hyvönen, Matteo Ceccarello, Teemu Roos, and Martin Aumüller. 2025. VIBE: vector index benchmark for embeddings. arXiv preprint arXiv:2505.17810 (2025).

[44] Himalaya Jain, Patrick Pérez, Rémi Gribonval, Joaquin Zepeda, and Hervé Jégou. 2016. Approximate Search with Quantized Sparse Representations. In Computer Vision – ECCV 2016 (Lecture Notes in Computer Science), Vol. 9911. Springer, 681-696. https://doi.org/10.1007/978-3-319-46478-7\_42

[45] Jerzy W Jaromczyk and Miroslaw Kowaluk. 1987. A note on relative neighborhood graphs. In Proceedings of the third annual symposium on Computational

geometry. 233–241.

[46] Herve Jegou, Matthijs Douze, and Cordelia Schmid. 2011. Product quantization for nearest neighbor search. IEEE transactions on pattern analysis and machine intelligence 33, 1 (2011), 117–128.

[47] Hervé Jégou, Romain Tavenard, Matthijs Douze, and Laurent Amsaleg. 2011. Searching in one billion vectors: re-rank with source coding. In 2011 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 861-864.

[48] Zhi Jing, Yongye Su, and Yikun Han. 2025. When large language models meet vector databases: A survey. In 2025 Conference on Artificial Intelligence x Multimedia (AIxMM). IEEE, 7-13.

[49] William B Johnson, Joram Lindenstrauss, et al. 1984. Extensions of Lipschitz mappings into a Hilbert space. Contemporary mathematics 26, 189-206 (1984), 1.

[50] Satyadhar Joshi. [n.d.]. Introduction to Vector Databases for Generative AI: Applications, Performance, Future Projections, and Cost Considerations. International Advanced Research Journal in Science, Engineering and Technology ISSN (O) ([n. d.]), 2393–8021.

[51] Andrew Kane. 2023. pgvector: Vector Similarity Search for PostgreSQL. https: //github.com/pgvector/pgvector. Open-source PostgreSQL extension for vector similarity search.

[52] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for opendomain question answering. In Proceedings of the 2020 conference on empirical methods in natural language processing (EMNLP). 6769–6781.

[53] Thijs Laarhoven. 2017. Graph-based time-space trade-offs for approximate near neighbors. arXiv preprint arXiv:1712.03158 (2017).

[54] Elizaveta Levina and Peter Bickel. 2004. Maximum likelihood estimation of intrinsic dimension. Advances in neural information processing systems 17 (2004).

[55] Binhong Li, Xiao Yan, and Shangqi Lu. 2026. Fast-Convergent Proximity Graphs for Approximate Nearest Neighbor Search. Proceedings of the ACM on Management of Data 4, 1 (SIGMOD (2026), 1–24.

[56] Conglong Li, Minjia Zhang, David G. Andersen, and Yuxiong He. 2020. Improving Approximate Nearest Neighbor Search through Learned Adaptive Early Termination. In Proceedings of the 2020 ACM SIGMOD International Conference on Management of Data (Portland, OR, USA) (SIGMOD '20). Association for Computing Machinery, New York, NY, USA, 2539–2554. https: //doi.org/10.1145/3318464.3380600

[57] Ke Li and Jitendra Malik. 2017. Fast k-nearest neighbour search via prioritized DCI. In International conference on machine learning. PMLR, 2081-2090.

[58] Peng-Cheng Lin and Wan-Lei Zhao. 2019. Graph based nearest neighbor search: Promises and failures. arXiv preprint arXiv:1904.02077 (2019).

[59] David G. Lowe. 2004. Distinctive Image Features from Scale-Invariant Keypoints. International Fournal of Computer Vision 60, 2 (2004), 91–110.

[60] Jianan Lu, Asaf Cidon, and Michael J. Freedman. 2026. When Enough is Enough: Rank-Aware Early Termination for Vector Search. In Proceedings of the 9th MLSys Conference. Bellevue, WA, USA. https://www.cs.princeton.edu/\~mfreed/ docs/terminus-mlsys26.pdf

[61] Kejing Lu, Chuan Xiao, and Yoshiharu Ishikawa. 2024. Probabilistic Routing for Graph-Based Approximate Nearest Neighbor Search. In Proceedings of the 41st International Conference on Machine Learning (Proceedings of Machine Learning Research), Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (Eds.), Vol. 235. PMLR, 33177-33195. https://proceedings.mlr.press/v235/lu24l.html

[62] Xinran Ma, Zhaoqi Zhou, Chuan Zhou, Qi Meng, Zaijiu Shang, Guoliang Li, and Zhiming Ma. 2025. Graph-Based Approximate Nearest Neighbor Search Revisited: Theoretical Analysis and Optimization. arXiv preprint arXiv:2509.15531 (2025).

[63] Yury Malkov, Alexander Ponomarenko, Andrey Logvinov, and Vladimir Krylov. 2014. Approximate nearest neighbor algorithm based on navigable small world graphs. Information Systems 45 (2014), 61–68.

[64] Yu. A. Malkov and D. A. Yashunin. 2018. Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs. arXiv:1603.09320 [cs.DS] https://arxiv.org/abs/1603.09320

[65] Magdalen Dobson Manohar, Zheqi Shen, Guy Blelloch, Laxman Dhulipala, Yan Gu, Harsha Vardhan Simhadri, and Yihan Sun. 2024. Parlayann: Scalable and deterministic parallel graph-based approximate nearest neighbor search algorithms. In Proceedings of the 29th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming. 270–285.

[66] Leland McInnes, John Healy, and James Melville. 2018. Umap: Uniform manifold approximation and projection for dimension reduction. arXiv preprint arXiv:1802.03426 (2018).

[67] Microsoft. 2021. SPACEV1B: A Billion-Scale Vector Dataset for Text Descriptors. https://github.com/microsoft/SPTAG/tree/main/datasets/SPACEV1B. Accessed July 2026.

[68] Microsoft. 2026. DiskANN: Graph-Structured Indices for Scalable, Fast, Fresh and Filtered Approximate Nearest Neighbor Search. https://github.com/

microsoft/DiskANN. Accessed: 2026-07-15.

[69] Milvus. 2024. Index Explained. https://milvus.io/docs/index-explained.md Accessed: 2026-04-22.

[70] Gonzalo Navarro. 2002. Searching in metric spaces by spatial approximation. The VLDB Journal 11, 1 (2002), 28–46.

[71] Partha Niyogi, Stephen Smale, and Shmuel Weinberger. 2008. Finding the Homology of Submanifolds with High Confidence from Random Samples. Discrete Comput. Geom. 39, 1–3 (March 2008), 419–441.

[72] NMSLIB. 2026. NMSLIB: Non-Metric Space Library, an Efficient Similarity Search Library and a Toolkit for Evaluation of k-NN Methods for Generic Non-Metric Spaces. https://github.com/nmslib/nmslib. Accessed: 2026-07-15.

[73] Aude Oliva and Antonio Torralba. 2001. Modeling the Shape of the Scene: A Holistic Representation of the Spatial Envelope. International Journal of Computer Vision 42, 3 (2001), 145–175.

[74] OpenAI. 2022. New and Improved Embedding Model (text-embedding-ada-002). OpenAI Blog. https://openai.com/index/new-and-improved-embeddingmodel/ Published December 15, 2022. Accessed July 2026.

[75] OpenAI. 2024. New Embedding Models and API Updates. https://openai.com/ index/new-embedding-models-and-api-updates/. Accessed: 2026-06-22.

[76] OpenAI-ArXiv 2023. OpenAI-ArXiv: 2.3M ada-002 Embeddings of arXiv Abstracts. Dataset distributed with big-ann-benchmarks. 2,321,096 base vectors, 1536 dimensions, 20,000 queries. https://comp21storage.z5.web.core.windows. net/arxiv-openaiv2-2M/.

[77] OpenSearch. 2024. k-NN Performance Tuning. https://docs.opensearch.org/1. 0/search-plugins/knn/performance-tuning/ Accessed: 2026-04-22.

[78] James Jie Pan, Jianguo Wang, and Guoliang Li. 2024. Survey of vector database management systems. The VLDB Journal 33, 5 (2024), 1591–1615.

[79] James Jie Pan, Jianguo Wang, and Guoliang Li. 2024. Vector database management techniques and systems. In Companion of the 2024 International Conference on Management of Data. 597–604.

[80] Yun Peng, Byron Choi, Tsz Nam Chan, Jianye Yang, and Jianliang Xu. 2023. Efficient Approximate Nearest Neighbor Search in Multi-dimensional Databases. Proc. ACM Manag. Data 1, 1, Article 54 (May 2023), 27 pages. https://doi.org/ 10.1145/3588908

[81] Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation. In Empirical Methods in Natural Language Processing (EMNLP). 1532-1543. http://www.aclweb.org/anthology/D14- 1162

[82] Vladimir Pestov. 2008. An axiomatic approach to intrinsic dimension of a dataset. Neural Networks 21, 2-3 (2008), 204–213.

[83] Pinecone. [n.d.]. Hierarchical Navigable Small Worlds (HNSW). https://www. pinecone.io/learn/series/faiss/hnsw/. Accessed: 2026-06-24.

[84] Liudmila Prokhorenkova and Aleksandr Shekhovtsov. 2020. Graph-based nearest neighbor search: From practice to theory. In International Conference on Machine Learning. PMLR, 7803–7813.

[85] William Pugh. 1990. Skip lists: a probabilistic alternative to balanced trees. Commun. ACM 33, 6 (June 1990), 668–676. https://doi.org/10.1145/78973.78977

[86] Qdrant. [n.d.]. Measuring ANN Recall. https://qdrant.tech/documentation/ tutorials-search-engineering/ann-recall/. Qdrant documentation. Accessed: 2026-07-21.

[87] Qdrant. 2025. Qdrant: Vector Search Engine. https://qdrant.tech. Accessed: 2026-01-29.

[88] Gang Qian, Shamik Sural, Yuelong Gu, and Sakti Pramanik. 2004. Similarity between Euclidean and Cosine Angle Distance for Nearest Neighbor Queries. In Proceedings of the 2004 ACM Symposium on Applied Computing (SAC). ACM, Nicosia, Cyprus, 1232–1237.

[89] RAPIDS Team. 2024. The Wiki-all Dataset. https://docs.rapids.ai/api/cuvs/ stable/cuvs\_bench/wiki\_all\_dataset/. 88M 768-dimensional embeddings of English and multilingual Wikipedia text, generated with the paraphrasemultilingual-mpnet-base-v2 model. Accessed July 2026.

[90] Redis. 2023. How Hierarchical Navigable Small World (HNSW) Algorithms Can Improve Search. https://redis.io/blog/how-hnsw-algorithms-can-improvesearch/ Accessed: 2026-04-22.

[91] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. arXiv:1908.10084 [cs.CL] https://arxiv.org/abs/ 1908.10084

[92] Xuan Shan, Chuanjie Liu, Yiqian Xia, Qi Chen, Yusi Zhang, Kaize Ding, Yaobo Liang, Angen Luo, and Yuxiang Luo. 2021. GLOW: Global Weighted Self-Attention Network for Web Search. In 2021 IEEE International Conference on Big Data (Big Data). IEEE, 519–528.

[93] Harsha Vardhan Simhadri, Martin Aumüller, Amir Ingber, Matthijs Douze, George Williams, Magdalen Dobson Manohar, Dmitry Baranchuk, Edo Liberty Frank Liu, Ben Landrum, et al. 2024. Results of the Big ANN: NeurIPS'23 competition. arXiv preprint arXiv:2409.17424 (2024).

[94] Aditi Singh, Suhas Jayaram Subramanya, Ravishankar Krishnaswamy, and Harsha Vardhan Simhadri. 2021. Freshdiskann: A fast and accurate graphbased ann index for streaming similarity search. arXiv preprint arXiv:2105.09613 (2021).

[95] Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, and Tie-Yan Liu. 2020. MPNet: Masked and Permuted Pre-training for Language Understanding. In Advances in Neural Information Processing Systems (NeurIPS).

[96] Yitong Song, Pengcheng Zhang, Chao Gao, Bin Yao, Kai Wang, Zongyuan Wu, and Lin Qu. 2025. TRIM: Accelerating High-Dimensional Vector Similarity Search with Enhanced Triangle-Inequality-Based Pruning. Proceedings of the ACM on Management of Data 3, 6 (2025), 1–26.

[97] Suhas Jayaram Subramanya, Devvrit, Rohan Kadekodi, Ravishankar Krishaswamy, and Harsha Vardhan Simhadri. 2019. DiskANN: fast accurate billionpoint nearest neighbor search on a single node. Curran Associates Inc., Red Hook, NY, USA.

[98] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. 2015. Going Deeper with Convolutions. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 1–9.

[99] Timescale. 2026. pgvectorscale: Postgres Extension for Vector Search with StreamingDiskANN. https://github.com/timescale/pgvectorscale. Accessed: 2026-07-15.

[100] Godfried T Toussaint. 1980. The relative neighbourhood graph of a finite planar set. Pattern recognition 12, 4 (1980), 261–268.

[101] turbopuffer. 2024. Continuous Recall Measurement. https://turbopuffer.com/ blog/continuous-recall. Accessed: 2026-06-25.

[102] Nitish Upreti. 2024. Azure Cosmos DB Vector Search with DiskANN Part 1: Full Space Search. Azure Cosmos DB Blog https://devblogs.microsoft.com/cosmosdb/azure-cosmos-db-vector-searchwith-diskann-part-1-full-space-search/. Accessed: 2026-07-15.

[103] Thomas Veasey and Mayya Sharipova. 2025. Speeding up merging of HNSW graphs.https://www.elastic.co/search-labs/blog/hnsw-graphs-speed-upmerging

[104] Vespa. [n.d.]. Nearest Neighbor Search Guide. https://docs.vespa.ai/en/nearestneighbor-search-guide.html. Vespa documentation. Accessed: 2026-07-21.

[105] Jianguo Wang, Xiaomeng Yi, Rentong Guo, Hai Jin, Peng Xu, Shengjun Li, Xiangyu Wang, Xiangzhou Guo, Chengming Li, Xiaohai Xu, et al. 2021. Milvus: A purpose-built vector data management system. In Proceedings of the 2021 international conference on management of data. 2614–2627.

[106] Mengzhao Wang, Xiangyu Ke, Xiaoliang Xu, Lu Chen, Yunjun Gao, Pinpin Huang, and Runkai Zhu. 2024. Must: An effective and scalable framework for multimodal search of target modality. In 2024 IEEE 40th International Conference on Data Engineering (ICDE). IEEE, 4747–4759.

[107] Mengzhao Wang, Xiaoliang Xu, Qiang Yue, and Yuxiang Wang. 2021. A comprehensive survey and experimental comparison of graph-based approximate nearest neighbor search. arXiv preprint arXiv:2101.12631 (2021).

[108] Zeyu Wang, Qitong Wang, Xiaoxing Cheng, Peng Wang, Themis Palpanas, and Wei Wang. 2024. Steiner-hardness: A query hardness measure for graph-based ann indexes. Proceedings of the VLDB Endowment 17, 13 (2024), 4668–4682.

[109] Weaviate. 2024. Vector Indexing. https://weaviate.io/developers/weaviate/ concepts/vector-index Accessed: 2026-04-22.

[110] Roger Weber, Hans-J Schek, and Stephen Blott. 1998. A quantitative analysis and performance study for similarity-search methods in high-dimensional spaces. (1998).

[111] Chuangxian Wei, Bin Wu, Sheng Wang, Renjie Lou, Chaoqun Zhan, Feifei Li, and Yuanzhe Cai. 2020. AnalyticDB-V: a hybrid analytical engine towards query fusion for structured and unstructured data. Proc. VLDB Endow. 13, 12 (Aug. 2020), 3152-3165. https://doi.org/10.14778/3415478.3415541

[112] Yuming Xu, Hengyu Liang, Jin Li, Shuotao Xu, Qi Chen, Qianxi Zhang, Cheng Li, Ziyue Yang, Fan Yang, Yuqing Yang, et al. 2023. Spfresh: Incremental in-place update for billion-scale vector search. In Proceedings of the 29th Symposium on Operating Systems Principles. 545-561.

[113] Mingyu Yang, Wentao Li, Jiabao Jin, Xiaoyao Zhong, Xiangyu Wang, Zhitao Shen, Wei Jia, and Wei Wang. 2025. Effective and general distance computation for approximate nearest neighbor search. In 2025 IEEE 41st International Conference on Data Engineering (ICDE). IEEE, 1098–1110.

[114] Chao Zhang and Renée J. Miller. 2026. Distribution-aware exploration for adaptive hnsw search. Proceedings of the ACM on Management of Data 4, 1 (SIGMOD (2026), 1–27.

[115] Xinkui Zhao, Hengxuan Lou, Yifan Zhang, Junjie Dai, Shuiguang Deng, and Jianwei Yin. 2026. GRAB-ANNS: High-Throughput Indexing and Hybrid Search via GPU-Native Bucketing. arXiv preprint arXiv:2604.16402 (2026).

[116] Zilliz. [n.d.]. AUTOINDEX Explained. https://docs.zilliz.com/docs/autoindexexplained. Accessed: 2026-06-25.

[117] Kostas Zoumpatianos, Yin Lou, Ioana Ileana, Themis Palpanas, and Johannes Gehrke. 2018. Generating data series query workloads: K. Zoumpatianos et al. The VLDB Journal 27, 6 (2018), 823–846.