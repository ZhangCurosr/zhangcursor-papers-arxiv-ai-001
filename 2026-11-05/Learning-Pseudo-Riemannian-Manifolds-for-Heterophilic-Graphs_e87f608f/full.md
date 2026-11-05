# Learning Pseudo-Riemannian Manifolds for Heterophilic Graphs via Graph Signature

Yun Young Choi<sup>∗†</sup>   
SolverX   
Seoul, Republic of Korea   
young@solverx.ai   
Asung Kil<sup>∗</sup>   
SolverX   
Seoul, Republic of Korea   
askil@solverx.ai   
Minho Lee   
SolverX   
Seoul, Republic of Korea   
mhlee@solverx.ai

Sun Woo Park<sup>∗</sup> Max Planck Institute for Mathematics Bonn, Germany s.park@mpim-bonn.mpg.de

Seokhwan Kim   
SolverX   
Seoul, Republic of Korea   
kshsh0405@solverx.ai

## Abstract

Several widely used families of Graph Neural Networks carry a positive-definite geometric prior. Graph convolution read as heat diffusion needs a positive semidefinite operator, hyperbolic models place features on positive-definite Riemannian manifolds, and dot-product and GAT-style attention scores use positive-definite inner products. On heterophilic graphs, where edges often signify dissimilarity or structural repulsion, that prior is a mismatch: no positive-definite form can assign a negative squared distance to a direction the labels separate. To measure the mismatch, we introduce the Graph Signature Index $( s _ { \mathcal { G } } )$ , the inertia of a classcompatibility matrix built from label topology alone, and we prove that no value of the edge or node homophily index in (0, 1) rules out a negative eigenvalue. We then propose the Pseudo-Riemannian Attention Network (PRAT), whose attention logit is a signed squared pseudo-distance under a learnable diagonal metric initialized from $s _ { \mathcal { G } }$ . The design is diagnose-then-learn: the signature supplies a sign structure that unconstrained training keeps outside near-zero directions, and the projection learns the rest of the metric around it. Across fifteen node-classification benchmarks PRAT outperforms or matches the strongest heterophily-adapted models. A controlled ablation ties that performance to the sign structure itself: with architecture and protocol fixed, forbidding the metric to be indefinite costs accuracy on the graphs $s _ { \mathcal { G } }$ marks as indefinite and leaves the others unchanged.

## 1 Introduction

Graph Neural Networks (GNNs) are the standard tool for relational learning [4]. Several of the most widely used families rest on a geometric prior rooted in Riemannian manifolds, typically Euclidean space, in which the metric tensor is strictly positive definite and connected nodes are therefore “close” and expected to share representations. The prior is explicit in three places. Graph convolution read as heat diffusion requires a positive semi-definite operator [5]; hyperbolic architectures such as HGCN [6] and HAT [47] embed features on positive-definite Riemannian manifolds; and dotproduct [38] and GAT-style [39] attention use Euclidean inner products of projected features, which admit no negative squared distance. The assumption is not universal. Signed attention [1] and sheaf models [2] are natural counterexamples, and precisely because they work, the geometry deserves to be treated as a modeling choice rather than a default.

While this Riemannian inductive bias serves well for homophilic graphs, it creates a fundamental geometric mismatch when applied to heterophilic graphs, where edges often denote dissimilarity or structural repulsion [49, 27]. Heterophily-adapted GNNs address this architecturally with highfrequency spectral filters [11, 7, 14] or signed aggregation weights [1, 44, 41]. These are empirically effective, but the space they act in stays positive definite, so repulsion is expressed by distorting an embedding built for attraction [36].

We read heterophily geometrically, whereas the prevailing view is statistical. Non-Euclidean GNNs [6, 42] adapt curvature to hierarchical structure. We argue that some heterophilic graphs call instead for an indefinite metric, which admits negative squared distances [17] and so can sharpen across one class pair while smoothing across another. We diagnose the geometry the labels induce and then learn the metric inside it. The diagnostic is the Graph Signature Index $( s _ { \mathcal { G } } ) _ { : }$ , a spectral invariant of the label topology. The model is the Pseudo-Riemannian Attention Network (PRAT), whose attention logit is a signed squared pseudo-distance under a diagonal metric initialized from $s _ { \mathcal { G } }$ and learned thereafter.

## 2 Theoretical Framework: The Geometry of Heterophily

## 2.1 Graph Signature Index

Previous studies proposed measures of graph homophily, including the node homophily index $h _ { V }$ [27] and edge homophily index $h _ { E } \ [ 5 0 ]$ described in Appendix ${ \mathrm { A . 5 . } }$ . Their values, ranging between 0 and 1, quantify the likelihood that nodes with identical labels are connected by edges (see Appendix $\mathbf { A . 7 } )$ . The graph signature index refines these indices by recording the signature of the eigenvalues of a real symmetric matrix constructed from a graph with node labels.

Definition 2.1. Let $\mathcal { G } : = ( \nu , \mathcal { E } )$ be an undirected graph without self-loops. Let $l : \mathcal { V }  L =$ $\{ 1 , 2 , \cdots , | L | \}$ be the ground-truth node label function assigning each node to one of $| L |$ classes. The graph signature matrix $S _ { G }$ is the $| L | \times | L |$ real symmetric matrix whose $( i , j )$ -th entry is given as

$$
s _ { i , j } : = \frac { | \{ ( v , w ) \in \mathcal { E } \mid l ( v ) = i , l ( w ) = j \} | } { | \{ v \in \mathcal { V } \mid l ( v ) = i \} | \cdot | \{ v \in \mathcal { V } \mid l ( v ) = j \} | }\tag{1}
$$

The graph signature index of ${ \mathcal { G } } .$ , denoted as $s _ { \mathcal { G } } .$ is the 3-tuple consisting of the number of positive, zero, and negative eigenvalues of $S _ { \mathcal { G } }$

Because $S _ { \mathcal { G } }$ is a real symmetric matrix, every eigenvalue of $S _ { \mathcal { G } }$ is real. Sylvester’s law of inertia shows that the graph signature index is invariant under any invertible linear change of coordinates on $\mathbb { R } ^ { | L | }$

Theorem 2.2 ([34]). Let $S \in \mathbb { R } ^ { n \times n }$ be a real symmetric matrix. Then there exists a diagonal matrix $D \in \mathbb { R } ^ { n \times n }$ and an invertible matrix $W \in \mathbb { R } ^ { n \times n }$ such that $S = W ^ { T } D W$ , where $( \cdot ) ^ { T }$ denotes the transpose of a square matrix. The numbers ofpositive, zero, and negative eigenvalues of $S$ are invariant under the change ofcoordinates.

Each $( i , j )$ -th entry of $S _ { \mathcal { G } }$ is the expected probability that a node with a preassigned label $j$ lies in the depth 1 unfolding tree rooted at a node with label $i ;$ see Appendix A.2. The signature of the eigenvalues of $S _ { \mathcal { G } }$ can be used to determine whether graphs with discrete labels are heterophilic or homophilic. For example, graphs with two discrete node labels and $\mathcal { E } \neq \emptyset$ have three types of $s _ { \mathcal G } ;$ see Appendix A.3. Appendix $\mathrm { A . 4 }$ provides heuristics on the density of these three graph types.

The following theorem relates the edge homophily level $h _ { E }$ to the graph signature index $s _ { \mathcal { G } }$

Theorem 2.3 (Critical Homophily Threshold). Assume class i contains $n _ { i }$ nodes and $\begin{array} { r } { N = \sum _ { i = 1 } ^ { | L | } n _ { i } . } \end{array}$ Under a proportional allocation model, the expected graph signature matrix $S _ { \mathcal { G } }$ has eigenvalues:

$$
\lambda _ { 1 } > 0 , \lambda _ { 2 } = \cdots = \lambda _ { | L | } = \frac { 2 | \mathcal { E } | \left( h _ { E } N ^ { 2 } - \sum _ { k = 1 } ^ { | L | } n _ { k } ^ { 2 } \right) } { \sum _ { k = 1 } ^ { | L | } n _ { k } ^ { 2 } \cdot \left( N ^ { 2 } - \sum _ { k = 1 } ^ { | L | } n _ { k } ^ { 2 } \right) } .\tag{2}
$$

The critical homophily threshold is given by: $\begin{array} { r } { h _ { E } ^ { * } = \frac { \sum _ { k = 1 } ^ { | L | } n _ { k } ^ { 2 } } { N ^ { 2 } } } \end{array}$ . For balanced classes where $n _ { i } = n f o r$ all $i ,$ this reduces to ${ h } _ { E } ^ { * } = 1 / | L |$

While Theorem 2.3 reveals a correlation between edge homophily and graph signature under idealized conditions, homophily indices cannot fully characterize the spectral properties of $S _ { \mathcal { G } }$ . We formalize this limitation by showing that no value of $h _ { E }$ or $h _ { V }$ in (0, 1) rules out a negative eigenvalue of $S _ { \mathcal { G } }$

Theorem 2.4 (Limitation of Homophily Indices). For every rational $h \in ( 0 , 1 )$ , there exists a graph $\mathcal { G } : = ( \nu , \mathcal { E } )$ with discrete node labels such that $h _ { E } = h _ { V } = h$ and $S _ { \mathcal { G } }$ has a negative eigenvalue.

## 2.2 Limitations of Riemannian GNNs

We reformulate the problem of analyzing heterophilic graphs using pseudo-Riemannian manifolds. A pseudo-Riemannian manifold $( \mathcal { M } , g )$ is a smooth manifold endowed with a metric tensor $g _ { p }$ $T _ { p } \mathbf { \bar { \mathcal { M } } } \times T _ { p } \mathbf { \mathcal { M } } \to \mathbb { R }$ which defines a non-degenerate symmetric bilinear form over the tangent space $\dot { T _ { p } } \mathcal { M }$ at each point $p \in { \mathcal { M } }$ A pseudo-Riemannian manifold is Riemannian if the tensor $g _ { p }$ is positive-definite for every point $p \in \mathcal { M }$ . We keep this terminology for a symmetric bilinear form that is allowed to be degenerate. This is deliberate: the form on the hidden space is a pullback of rank at most $| L |$ , hence degenerate whenever the hidden width exceeds the number of classes, and it depends on a hidden vector only through its image in L. Non-degeneracy returns on $\mathcal { L }$ itself whenever $s _ { \mathcal { G } }$ has no zero eigenvalue, and Theorem 2.5 counts only negative eigenvalues, so degeneracy does not affect it. We focus on the h-dimensional manifold $\mathcal { M }$ endowed with a metric tensor $g _ { p } : \dot { = } S \in \mathbb { R } ^ { h \times h }$ for some fixed real symmetric matrix S at every point $p \in \mathcal { M }$ , which defines the following quadratic form on tangent vectors $w _ { 1 } , w _ { 2 } \in T _ { p } { \mathcal { M } } \colon g _ { p } \bar { ( w _ { 1 } , w _ { 2 } ) } = w _ { 1 } ^ { T } S w _ { 2 } \left[ 2 4 , 3 7 \right]$

Let $\mathcal { G } : = ( \nu , \mathcal { E } )$ be a graph with continuous node attributes $X : \mathcal { V } \to \mathbb { R } ^ { h }$ and embedded node labels $l : \mathcal { V } \to \mathbb { R } ^ { | L | }$ . We set the metric tensor over $\mathbb { R } ^ { | L | }$ to be the graph signature matrix $S _ { \mathcal { G } }$ of the given graph G. By doing so, we regard the graph $\mathcal { G }$ as a subspace of some smooth manifold M admitting a smooth function $f : \mathcal { M } \xrightarrow { } \mathcal { L } \cong \mathbb { R } ^ { | \bar { L } | }$ , whose metric tensor is the pullback of metric tensor $f ^ { * } S _ { \mathcal { G } }$ Given a metric tensor g on $\mathcal { L } ,$ , its pullback $f ^ { * } g$ on M is defined as

$$
( f ^ { * } g ) _ { p } ( v , w ) : = g _ { f ( p ) } ( T _ { p } f ( v ) , T _ { p } f ( w ) ) .
$$

Because $S _ { \mathcal { G } }$ can have negative eigenvalues, M is a pseudo-Riemannian manifold, thereby covering a class of manifolds which are not Riemannian. The theorem below, proved in Appendix A.8, makes this observation explicit.

Theorem 2.5 (Geometric Mismatch). Let M and L be differentiable manifolds and $f : \mathcal { M } \to \mathcal { L }$ be a smooth map. $I f S _ { \mathcal { G } }$ defined on L is indefinite and the pushforward $T _ { p } \dot { f } : T _ { p } { \cal M } \to T _ { f ( p ) } { \cal  L }$ is surjective for every $p \in \mathcal { M }$ , then there exists no diffeomorphism ϕ : ${ \mathcal { L } } \to { \mathcal { L } }$ such that the pullback of the Euclidean metric $I ,$ the identityform, satisfies $( \phi \circ f ) ^ { \ast } I = f ^ { \ast } S g$

Theorem 2.5 establishes a geometric mismatch between the Riemannian prior and graphs whose $S _ { \mathcal { G } }$ is indefinite. It proves that this geometric mismatch is intrinsic and cannot be bridged by any smooth transformation of the Riemannian space. Surjectivity of the pushforward is what carries the indefiniteness through to the pullback: the image of $\bar { T _ { p } } f$ is then all of $T _ { f ( p ) } { \mathcal { L } } $ , so a negative direction of $S _ { \mathcal { G } }$ has a preimage. Injectivity alone would not suffice, since an injective map can pull an indefinite form back to a degenerate one. In our architecture $T _ { p } f$ is the projection W of Equation (4), surjective exactly when rank $\mathbf { W } = \left| L \right|$ |. Every configuration we report has $h ^ { \prime } > | L |$ , so the hypothesis holds for almost every W. Appendix A.8 discusses further implications for heterophilic graphs.

## 3 Method: Pseudo-Riemannian Attention Network (PRAT)

## 3.1 Pseudo-Riemannian Edge Attention using continuous features

We define the tangent attention scores of a finite graph ${ \mathcal { G } } .$ which generalize the notion of attention scores used in graph attention networks (GAT) and attention-transformers. The geometric structure endowed by the pseudo-metric tensor g allows us to compute the numerical relation between two edges of a finite directed graph $\mathcal { G }$ by comparing their derivatives. Given a curve $\gamma : [ 0 , 1 ] \to \mathcal { M }$ the derivative of the curve at time $t = t _ { 0 } .$ , denoted as $\gamma ^ { \prime } ( t _ { 0 } )$ , is an element of the tangent space $T _ { \gamma ( t _ { 0 } ) } \mathcal { M }$ . Given two curves $\gamma _ { 1 } , \gamma _ { 2 } : [ 0 , 1 ] \to \mathcal { M }$ , the two derivatives $\gamma _ { 1 } ^ { \prime } ( t _ { 0 } )$ and $\gamma _ { 2 } ^ { \prime } ( t _ { 0 } )$ may not lie in the same tangent space. To overcome this, we define a function $\gamma _ { 1 2 } : [ 0 , 1 ] \times [ 0 , 1 ]  \mathcal { M }$ which satisfies $\gamma _ { 1 2 } ( t , 0 ) = \gamma _ { 2 } ( t )$ and $\gamma _ { 1 2 } ( t , 1 ) = \gamma _ { 1 } ( t )$ . For each time $t = t _ { 0 } ,$ , the parallel transport operator $\Gamma _ { \gamma _ { 1 2 } ( t _ { 0 } , 1 ) } : T _ { \gamma _ { 2 } ( t _ { 0 } ) } \mathcal { M } \to T _ { \gamma _ { 1 } ( t _ { 0 } ) } \mathcal { M }$ moves any tangent vector w $\in T _ { \gamma _ { 2 } ( t _ { 0 } ) } \bar { \mathcal { M } }$ to the tangent vector $\Gamma _ { \gamma _ { 1 2 } ( t _ { 0 } , 1 ) } ( w _ { 2 } ) \in T _ { \gamma _ { 1 } ( t _ { 0 } ) } \mathcal { M }$ along the curve $\gamma _ { 1 2 } ( t _ { 0 } , s ) : \{ t _ { 0 } \} \times [ 0 , 1 ]  \mathcal { M }$ . After performing parallel transport, we define the tangent attention score between $\gamma _ { 1 }$ and $\gamma _ { 2 }$ . Figure 1b illustrates how to compute a tangent attention score.

![](images/4b98a82f71c34df6d64ca791544902ab1feabdc54ffe8e39ad68ab6162644143.jpg)  
Figure 1: Figure 1(a) illustrates how one can map a directed graph $\mathcal { G }$ to a subspace of a pseudo-Riemannian manifold $( \mathcal { M } , g )$ . The nodes of $\mathcal { G }$ correspond to points on $\mathcal { M } ,$ whereas the edges of $\mathcal { G }$ correspond to curves on M. Figure 1(b) demonstrates how one may compute tangent attention scores between two edges of $\mathcal { G }$ using the pseudo-Riemannian metric g defined over the tangent space $T _ { \gamma _ { 1 } ( t ) } \mathcal { M }$ at each point of the curve representing an edge.

Definition 3.1. The tangent attention score between two curves $\gamma _ { 1 } , \gamma _ { 2 } : [ 0 , 1 ] \to \mathcal { M }$ with a choice of a function $\gamma _ { 1 2 } : [ 0 , 1 ] \times \mathbf { \bar { [ 0 , 1 ] } }  \mathcal { M }$ is

$$
e _ { \gamma _ { 1 2 } } ( \gamma _ { 1 } , \gamma _ { 2 } ) : = \int _ { 0 } ^ { 1 } g _ { \gamma _ { 1 } ( t ) } \left( \gamma _ { 1 } ^ { \prime } ( t ) , \Gamma _ { \gamma _ { 1 2 } ( t , 1 ) } ( \gamma _ { 2 } ^ { \prime } ( t ) ) \right) d t .
$$

By imposing suitable topological structures on finite directed graphs, we can compute a tangent attention score between two edges of a directed graph $\mathcal { G } : = ( \nu , \mathcal { E } )$ (including self-loops) as a function of node features $X : \mathcal { V } \to \mathbb { R } ^ { h }$ . See Appendix A.9 for details. We also choose a continuous function $X _ { \gamma } : \mathcal { G }  \mathcal { M }$ which extends the node feature function X. Given two edges $e _ { 1 } , e _ { 2 } \in \mathcal { E }$ with the corresponding paths $\gamma _ { e _ { 1 } } , \gamma _ { e _ { 2 } } : [ 0 , 1 ] \to \mathbb { R } ^ { h }$ , we define $\gamma _ { e _ { 1 } , e _ { 2 } } ( t , s ) : = s \cdot \gamma _ { e _ { 1 } } ( t ) + ( 1 - s ) \cdot \gamma _ { e _ { 2 } } ( t )$ Tangent attention scores are symmetric in their two edges. Section $\mathrm { A } . 9$ gives the argument. We can quantify the differences among node features of G using these scores.

Definition 3.2. The tangent attention score matrix of G, denoted as $e _ { \mathcal { G } , \gamma } ,$ is the $| \mathcal { E } | \times | \mathcal { E } |$ real symmetric matrix whose $( e _ { 1 } , e _ { 2 } ) \cdot \mathrm { t h }$ entry for $e _ { 1 } , e _ { 2 } \in \mathcal { E }$ is the tangent attention score $e _ { \gamma _ { e _ { 1 } , e _ { 2 } } } ( \gamma _ { e _ { 1 } } , \gamma _ { e _ { 2 } } )$

Using $e _ { \mathcal { G } , \gamma } ,$ , we recover several classes of previously studied attention functions. The detailed version of the theorem stated below can be found in Appendix A.10.

Theorem 3.3. There exist continuous extensions $X _ { \gamma } : \mathcal { G }  \mathcal { M }$ such that $e _ { \mathcal { G } , \gamma }$ respectively recovers dot-product attention, single-layerfeedforward linear layer, and dynamic graph attention [38, 39, 3].

## 3.2 Learnable Indefinite Metric

Because tangent attention scores encompass various types of previously studied attention scores, a wide class of novel graph attention networks can be constructed by carefully choosing the curves representing the edges of ${ \mathcal { G } } .$ , choosing the symmetric matrix S, and choosing which subsets of entries of the tangent attention score matrix to use.

Building on this, our proposed Pseudo-Riemannian Attention Network (PRAT) architecture reformulates classical attention networks with tangent attention scores defined from the curve $\gamma _ { e } ^ { [ 1 ] } : [ 0 , 1 ] \to \mathcal { M }$ as shown in Theorem 3.3 and Appendix A.10. The following equation governs the PRAT algorithm.

$$
h _ { v } ^ { ( i + 1 ) } : = \bigoplus _ { k = 1 } ^ { K } \sigma \left( \sum _ { w \in \mathcal { N } _ { v } \cup \{ v \} } \alpha _ { v , w } ^ { [ i , k ] } \mathbf { W _ { v } } ^ { [ i , k ] } h _ { w } ^ { ( i ) } \right) , \quad \quad h _ { v } ^ { ( 0 ) } : = \psi _ { e n c } ( X ( v ) )\tag{3}
$$

For each $1 \leq k \leq K$

$$
\alpha _ { v , w } ^ { [ i , k ] } : = \frac { \exp ( e ( h _ { v } ^ { ( i ) } , h _ { w } ^ { ( i ) } ) ^ { [ k ] } ) } { \sum _ { u \in \mathcal { N } _ { v } \cup \{ v \} } \exp ( e ( h _ { v } ^ { ( i ) } , h _ { u } ^ { ( i ) } ) ^ { [ k ] } ) } , e ( h _ { v } ^ { ( i ) } , h _ { u } ^ { ( i ) } ) ^ { [ k ] } : = \left\{ \begin{array} { l l } { \Delta ( h _ { v , u } ^ { ( i ) } ) ^ { \top } \mathbf { D } ^ { [ i , k ] } \Delta ( h _ { v , u } ^ { ( i ) } ) } & { \mathrm { i f } u \in \mathcal { N } _ { v } } \\ { a ^ { [ i , k ] } ( h _ { v } ^ { ( i ) } ) ^ { \top } \mathbf { D } ^ { [ i , k ] } ( h _ { v } ^ { ( i ) } ) } & { \mathrm { i f } v = u } \end{array} \right.
$$

where $\mathcal { N } _ { v }$ is the set of nodes adjacent to the node $v , \Delta ( h _ { v , u } ^ { ( i ) } )$ is equal to $h _ { v } ^ { ( i ) } - h _ { u } ^ { ( i ) }$ given $u \in \mathcal { N } _ { v }$ and σ is the nonlinear activation function. $\psi _ { e n c } : \mathbb { R } ^ { h }  \mathbb { R } ^ { h ^ { \prime } }$ is a learnable function (e.g., multilayer perceptron) that maps node features to hidden states of dimension $h ^ { \prime }$ . The $a ^ { [ i , k ] }$ are learnable self-loop weights playing the role of $a _ { v } ^ { 2 }$ in Equation (40), calibrated in Appendix F.3, and $\mathbf { D } ^ { [ i , k ] } \in \mathbb { R } ^ { h ^ { \prime } \times h ^ { \prime } }$ are symmetric matrices that serve as the metric tensor.

In Appendix B, we conceptualize the learning process of PRAT as a sequence of geometric transformations, from which we derive $\mathbf { D } ^ { [ l , k ] }$ for the k-th head at layer l as follows, where $\mathbf { \bar { W } } ^ { [ l , k ] } \in \mathbb { R } ^ { | L | \times h ^ { \prime } }$ is a learnable weight matrix, and $S ^ { | L | } \in \mathbb { R } ^ { | L | \times | L | }$ is a pseudo-metric on the latent manifold.

$$
\mathbf { D } ^ { [ l , k ] } = ( \mathbf { W } ^ { [ l , k ] } ) ^ { \top } S ^ { | L | } \mathbf { W } ^ { [ l , k ] } .\tag{4}
$$

Following the theoretical motivation in Section 2.2, we instantiate the pseudo-metric $S ^ { | L | }$ as the graph signature matrix $S _ { \mathcal { G } }$ . The pullback then carries the label topology into the attention score. Substituting $S ^ { | L | } = S _ { \mathcal { G } }$ into Equation (4) and performing the eigendecomposition $S _ { \mathcal { G } } = Q ^ { \top } \Lambda _ { \mathcal { G } } Q$ we can reformulate the induced metric tensor. By absorbing the orthogonal eigenvector matrix $Q$ into the learnable weight matrix, i.e., defining $\mathbf { W } ^ { \prime [ l , k ] } : = Q \mathbf { W } ^ { [ l , k ] }$ , we arrive at:

$$
\mathbf D ^ { [ l , k ] } = ( \mathbf W ^ { [ l , k ] } ) ^ { \top } ( Q ^ { \top } \Lambda _ { \mathcal { G } } Q ) \mathbf W ^ { [ l , k ] } = ( Q \mathbf W ^ { [ l , k ] } ) ^ { \top } \Lambda _ { \mathcal { G } } ( Q \mathbf W ^ { [ l , k ] } ) = ( \mathbf W ^ { \prime [ l , k ] } ) ^ { \top } \Lambda _ { \mathcal { G } } \mathbf W ^ { \prime [ l , k ] } .\tag{5}
$$

In Equation $( 5 ) , \Lambda _ { \mathcal { G } }$ is a diagonalized graph signature matrix which directly encodes the homophilic or heterophilic character of the graph. In practical semi-supervised settings, however, the ground-truth labels for the entire graph are inaccessible, so we compute the empirical graph signature $S _ { \mathcal { G } , \mathrm { t r a i n } }$ and its eigenvalues $\Lambda _ { \mathcal { G } , \mathrm { t r a i n } }$ on the subgraph induced by the training nodes. We initialize the diagonal matrix Λ using a normalized version of $\Lambda _ { { \mathcal { G } } , \mathrm { t r a i n } } ,$ i.e. $\begin{array} { r } { \Lambda _ { \mathrm { i n i t } } = \frac { \Lambda _ { \mathcal { G } , \mathrm { t r a i n } } } { \mathrm { s t d } ( \Lambda _ { \mathcal { G } , \mathrm { t r a i n } } ) } \cdot \frac { 1 } { \sqrt { | L | } } } \end{array}$ , where std(·) is the standard deviation, set to 1 if zero. We leave these values free and treat the diagonal matrix Λ as a learnable parameter. This design allows the model to fine-tune the geometric prior derived from the training set and generalize it to the unseen test nodes. Labels enter only here, and the forward pass never reads one. What the initialization has to supply is an informed indefinite prior, not ground truth: Section J.2 shows that a signature estimated from k-means pseudo-labels on node features recovers the label-based accuracy on both indefinite benchmarks, and Section K shows how far the estimate and the accuracy survive when the training labels are few or wrong.

## 4 Experiments

## 4.1 Datasets and Baselines

Standard Benchmark Datasets. We first evaluate PRAT on node classification across eight standard benchmarks covering a broad spectrum of homophily levels. We use Texas and Cornell from WebKB <sup>3</sup>, Chameleon and Squirrel [30], and Actor [35] as heterophilic datasets, and Cora, Citeseer, and PubMed [32, 45] as homophilic counterparts. Detailed statistics, including edge homophily levels $( h _ { E } )$ and graph signature indices $( s _ { \mathcal { G } } )$ , are summarized in Table 1. The homophilic graphs here carry strictly positive signatures and the heterophilic ones other than Actor carry negative components. That association only motivates the design; the ablation in Table 5, not this correlation, shows that the sign structure produces the gain.

Platonov Benchmark Datasets. We additionally evaluate PRAT on seven datasets curated by Platonov et al. [28]: (i) the filtered versions of Chameleon and Squirrel, in which duplicate nodes causing severe train–test leakage in the original benchmarks are removed; and (ii) five largescale heterophilic datasets, roman-empire, amazon-ratings, minesweeper, tolokers and questions, spanning 10K to 49K nodes They test PRAT at scale without the data-quality artifacts of the original Wikipedia networks. Statistics are in Table 2 and descriptions in Appendix C.

Table 1: Standard benchmark statistics $( h _ { E } , s _ { \mathcal { G } } )$
<table><tr><td>Dataset</td><td>Cora</td><td>PubMed</td><td>Citeseer</td><td>Chameleon Squirrel</td><td></td><td>Actor</td><td>Cornell</td><td>Texas</td></tr><tr><td>Nodes</td><td>2,708</td><td>19,717</td><td>3,327</td><td>2,277</td><td>5,201</td><td>7,600</td><td>183</td><td>183</td></tr><tr><td>Edges Classes</td><td>5,278</td><td>44,324</td><td>4,552</td><td>36,101</td><td>217,073 30,019</td><td></td><td>298</td><td>325</td></tr><tr><td>hE</td><td>7 0.810</td><td>3 0.802</td><td>6 0.736</td><td>5 0.230</td><td>5 0.222</td><td>5 0.217</td><td>5 0.123</td><td>5</td></tr></table>

Table 2: Platonov benchmark statistics [28].
<table><tr><td></td><td></td><td></td><td></td><td>Dataset Cham.-filt. Sq.-filt. roman-empire amazon-ratings minesweeper tolokers</td><td></td><td></td><td>questions</td></tr><tr><td>Nodes</td><td>890</td><td>2,223</td><td>22,662</td><td>24,492</td><td>10,000</td><td>11,758</td><td>48,921</td></tr><tr><td>Edges</td><td>8,854</td><td>57,134</td><td>32,927</td><td>93,050</td><td>39,402</td><td>519,000</td><td>153,540</td></tr><tr><td>Classes</td><td>5</td><td>5</td><td>18</td><td>5</td><td>2</td><td>2</td><td>2</td></tr><tr><td>hE</td><td>0.235</td><td>0.220</td><td>0.046</td><td>0.380</td><td>0.683</td><td>0.595</td><td>0.840</td></tr><tr><td>sg</td><td>(3,0,2)</td><td>(2,1,2)</td><td>(6,7,5)</td><td>(3,2,0)</td><td>(1,1,0)</td><td>(2,0,0)</td><td>(1,1,0)</td></tr></table>

Baselines on Standard Benchmarks. We compare PRAT against a diverse set of architectures spanning multiple paradigms: (1) Standard GNNs: GCN [16], GIN [43], and GraphSAGE [12]; (2) Attention-based models: GAT [39], GATv2 [3], Scaled Dot-Product (SDP) Attention [38, 33], and Graph Transformer with and without ego-neighbor separation (GT, GT-sep) [33, 49]; (3) Heterophilyadapted GNNs: CPGNN [50], FAGCN [1], Geom-GCN [27], GPR-GNN [7], BernNet [14], and H<sub>2</sub>GCN [49]; (4) Geometric models: Q-GCN [42] on a pseudo-hyperboloid and HAT [47] on the Poincaré ball.

Baselines on Platonov Benchmarks. On the Platonov benchmarks, we compare PRAT against the best of three ResNet variants and the 14 GNN baselines reported in Platonov et al. [28]. (1) Standard GNNs: GCN [16] and GraphSAGE [12]; (2) Attention-based models: GAT [39], GAT-sep [49], Graph Transformer (GT) [33], and GT-sep [49], where the “-sep” suffix denotes the ego- and neighborembedding separation trick of Zhu et al. [49]; (3) Heterophily-adapted GNNs: H GCN [49], CPGNN [50], GPR-GNN [7], FSGNN [23], GloGNN [19], FAGCN [1], GBK-GNN [10], and JacobiConv [40].

## 4.2 Experimental Setup

Standard Benchmarks. We evaluate performance using 10 splits. For heterophilic datasets, we adopt the fixed 48%/32%/20% train/validation/test splits provided by Pei et al. [27]. For homophilic datasets, we generate random 60%/20%/20% splits, avoiding the selection bias of a fixed set. All graphs are undirected with self-loops added, and the seed is fixed across splits. Baselines and PRAT share one sweep over a predefined range (hidden dimension, learning rate, dropout), and we report mean test accuracy with standard deviation at the best-validation step.<sup>4</sup> Detailed configurations are provided in Appendix F.1.

Platonov Benchmarks. For the seven Platonov benchmarks, we use the official training pipeline of Platonov et al. [28]<sup>5</sup> as our experimental harness, where PRAT is implemented as a drop-in attention replacement inside the Platonov backbone. Data loading, the ten predefined splits, validation-based model selection and 10-split mean ± std reporting all follow Platonov et al. [28] exactly, so every deviation we report is across splits at a fixed seed. Section G.2 adds three seeds and finds seed variance the smaller of the two. PRAT is hyperparameter-tuned inside the same harness published by Platonov et al. [28]. Detailed configurations are provided in Appendix F.1.

Cost. A PRAT layer has GAT’s asymptotic cost, since the metric it adds is diagonal and acts on a latent space whose dimension is the number of classes (Section E gives the expressions). Measured on the two largest graphs at equal backbone, width and depth, a training step takes 88.7 ms on questions against 91.2 for GAT and 175.2 for GT, and 26.7 ms on tolokers against 22.8 and 32.9 (Section E.3). At this scale, PRAT costs as much as GAT.

## 4.3 Main Results

Table 3 reports node classification accuracy on the eight standard benchmarks. PRAT is best on Actor and Texas, second on Squirrel, third on Chameleon and fourth on Cornell, and among the homophilic datasets it is best on Citeseer and third on PubMed; Section G.1 reports the pairwise tests, four of which are ties. PRAT attends over 1-hop neighborhoods only, yet it exceeds the multi-hop spectral filters BernNet and GPR-GNN on Texas, Squirrel and Actor, so the margin there does not come from a larger receptive field. Section H reads the GT and geometric rows.

Table 3: Performance of PRAT and baselines on benchmark datasets. The top 3 performances are highlighted in First, Second, and Third.
<table><tr><td></td><td colspan="3">Homophilic</td><td colspan="5">Heterophilic</td></tr><tr><td>hE</td><td>Cora 0.810</td><td>PubMed 0.802</td><td>Citeseer 0.736</td><td>Chameleon 0.230</td><td>Squirrel 0.222</td><td>Actor 0.217</td><td>Cornell 0.123</td><td>Texas 0.061</td></tr><tr><td>GCN</td><td>88.32±1.02</td><td>88.75±0.43</td><td>77.69±1.06</td><td>67.43±2.45</td><td>54.95±1.19</td><td>30.56±0.70</td><td>44.32±6.75</td><td>58.38±4.86</td></tr><tr><td>GIN</td><td>87.38±1.56</td><td>89.20±0.51</td><td>76.82±1.29</td><td>62.52±2.61</td><td>53.56±1.66</td><td>30.97±1.56</td><td>41.62±6.86</td><td>59.73±3.51</td></tr><tr><td>SAGE</td><td>88.80±1.04</td><td>89.77±0.45</td><td>77.66±0.81</td><td>64.91±1.81</td><td>43.92±1.87</td><td>36.17±1.26</td><td>69.46±8.64</td><td>81.35±3.72</td></tr><tr><td>GAT</td><td>88.32±1.09</td><td>88.87±0.41</td><td>77.10±1.27</td><td>69.67±1.67</td><td>55.18±2.13</td><td>31.61±1.99</td><td>50.54±7.55</td><td>64.59±6.67</td></tr><tr><td>GATv2</td><td>87.99±0.93</td><td>89.63±0.47</td><td>76.97±0.96</td><td>67.94±2.34</td><td>59.79±1.83</td><td>32.97±1.37</td><td>54.32±9.70</td><td>66.76±6.63</td></tr><tr><td>SDP</td><td>87.99±1.30</td><td>89.45±0.55</td><td>76.85±0.97</td><td>67.06±1.53</td><td>56.42±3.30</td><td>34.19±0.82</td><td>52.97±8.04</td><td>65.68±5.28</td></tr><tr><td>GT</td><td>87.64±0.94</td><td>89.10±0.50</td><td>76.76±1.31</td><td>68.71±2.35</td><td>56.83±1.40</td><td>35.70±0.97</td><td>66.22±4.81</td><td>76.76±6.14</td></tr><tr><td>GT-sep</td><td>87.92±1.31</td><td>89.13±0.45</td><td>77.69±1.27</td><td>60.59±1.78</td><td>44.07±1.29</td><td>35.86±0.67</td><td>70.00±2.97</td><td>77.03±5.59</td></tr><tr><td>CPGNN</td><td>87.24±1.07</td><td>89.78±0.48</td><td>76.83±1.00</td><td>56.18±2.93</td><td>42.34±1.07</td><td>33.95±1.08</td><td>64.05±6.63</td><td>77.57±7.55</td></tr><tr><td>FAGCN</td><td>88.75±1.38</td><td>89.03±0.55</td><td>77.46±1.11</td><td>65.07±1.56</td><td>49.55±1.92</td><td>36.51±1.02</td><td>60.27±6.40</td><td>71.89±7.17</td></tr><tr><td>Geom-GCN</td><td>85.35±1.57</td><td>89.95±0.47</td><td>78.02±1.15</td><td>60.00±2.81</td><td>38.15±0.92</td><td>31.59±1.15</td><td>60.54±3.67</td><td>66.76±2.72</td></tr><tr><td>GPR-GNN</td><td>89.43±1.06</td><td>90.01±0.60</td><td>77.76±0.83</td><td>68.18±1.92</td><td>51.46±1.34</td><td>35.38±0.98</td><td>74.59±4.12</td><td>83.51±5.78</td></tr><tr><td>BernNet</td><td>89.67±1.04</td><td>90.13±0.51</td><td>77.79±1.38</td><td>70.77±1.63</td><td>55.38±1.46</td><td>35.20±1.14</td><td>76.76±3.68</td><td>83.78±4.27</td></tr><tr><td>H2GCN</td><td>89.61±1.15</td><td>90.14±0.51</td><td>76.86±1.35</td><td>59.30±2.12</td><td>35.55±1.58</td><td>34.72±1.18</td><td>74.86±4.68</td><td>82.43±5.28</td></tr><tr><td>Q-GCN</td><td>87.70±1.19</td><td>88.22±0.40</td><td>77.07±1.50</td><td>59.67±2.09</td><td>46.27±1.22</td><td>29.65±0.70</td><td>45.68±4.90</td><td>61.62±5.10</td></tr><tr><td>HAT</td><td>88.36±1.09</td><td>89.89±0.53</td><td>77.90±1.35</td><td>57.48±2.03</td><td>38.87±1.40</td><td>34.34±1.03</td><td>66.49±4.39</td><td>78.65±5.05</td></tr><tr><td>PRAT (ours)</td><td>88.42±1.17</td><td>90.03±0.47</td><td>78.06±1.37</td><td>68.73±1.74</td><td>56.96±1.51</td><td>37.26±1.44</td><td>71.89±4.71</td><td>84.32±5.10</td></tr></table>

Table 4: Performance on Platonov et al. [28] benchmarks. Accuracy (↑) is reported for roman-empire, amazon-ratings, Cham.-filt., Sq.-filt.; ROC AUC (↑) for minesweeper, tolokers, questions. <sup>†</sup>Best per dataset of the graph-agnostic ResNet of Platonov et al. [28] and its graph-aware variants ResNet+SGC and ResNet+adj.
<table><tr><td rowspan="2"></td><td colspan="2">Filtered Wikipedia</td><td colspan="5">Large-scale heterophilic</td></tr><tr><td>Cham.-filt. 0.235</td><td>Sq.-filt. 0.220</td><td>roman-empire amazon-ratings minesweeper 0.046</td><td>0.380</td><td>0.683</td><td>tolokers 0.595</td><td>questions 0.840</td></tr><tr><td>hE ResNet†</td><td>41.01±4.54</td><td>38.37±1.99</td><td>73.90±0.51</td><td>51.83±0.57</td><td>70.88±0.90</td><td>80.70±0.97</td><td>75.81±0.96</td></tr><tr><td>GCN</td><td>40.89±4.12</td><td>39.47±1.47</td><td>73.69±0.74</td><td>48.70±0.63</td><td>89.75±0.52</td><td>83.64±0.67</td><td>76.09±1.27</td></tr><tr><td>SAGE</td><td>37.77±4.14</td><td>36.09±1.99</td><td>85.74±0.67</td><td>53.63±0.39</td><td>93.51±0.57</td><td>82.43±0.44</td><td>76.44±0.62</td></tr><tr><td>GAT</td><td>39.21±3.08</td><td>35.62±2.06</td><td>80.87±0.30</td><td>49.09±0.63</td><td>92.01±0.68</td><td>83.70±0.47</td><td>77.43±1.20</td></tr><tr><td>GAT-sep</td><td>39.26±2.50</td><td>35.46±3.10</td><td>88.75±0.41</td><td>52.70±0.62</td><td>93.91±0.35</td><td>83.78±0.43</td><td>76.79±0.71</td></tr><tr><td>GT</td><td>38.87±3.66</td><td>36.30±1.98</td><td>86.51±0.73</td><td>51.17±0.66</td><td>91.85±0.76</td><td>83.23±0.64</td><td>77.95±0.68</td></tr><tr><td>GT-sep</td><td>40.31±3.01</td><td>36.66±1.63</td><td>87.32±0.39</td><td>52.18±0.80</td><td>92.29±0.47</td><td>82.52±0.92</td><td>78.05±0.93</td></tr><tr><td>H2GCN</td><td>26.75±3.64</td><td>35.10±1.15</td><td>60.11±0.52</td><td>36.47±0.23</td><td>89.71±0.31</td><td>73.35±1.01</td><td>63.59±1.46</td></tr><tr><td>CPGNN</td><td>33.00±3.15</td><td>30.04±2.03</td><td>63.96±0.62</td><td>39.79±0.77</td><td>52.03±5.46</td><td>73.36±1.01</td><td>65.96±1.95</td></tr><tr><td>GPR-GNN</td><td>39.93±3.30</td><td>38.95±1.99</td><td>64.85±0.27</td><td>44.88±0.34</td><td>86.24±0.61</td><td>72.94±0.97</td><td>55.48±0.91</td></tr><tr><td>FSGNN</td><td>40.61±2.97</td><td>35.92±1.32</td><td>79.92±0.56</td><td>52.74±0.83</td><td>90.08±0.70</td><td>82.76±0.61</td><td>78.86±0.92</td></tr><tr><td>GloGNN</td><td>25.90±3.58</td><td>35.11±1.24</td><td>59.63±0.69</td><td>36.89±0.14</td><td>51.08±1.23</td><td>73.39±1.17</td><td>65.74±1.19</td></tr><tr><td>FAGCN</td><td>41.90±2.72</td><td>41.08±2.27</td><td>65.22±0.56</td><td>44.12±0.30</td><td>88.17±0.73</td><td>77.75±1.05</td><td>77.24±1.26</td></tr><tr><td>GBK-GNN</td><td>39.61±2.60</td><td>35.51±1.65</td><td>74.57±0.47</td><td>45.98±0.71</td><td>90.85±0.58</td><td>81.01±0.67</td><td>74.47±0.86</td></tr><tr><td>JacobiConv</td><td>39.00±4.20</td><td>29.71±1.66</td><td>71.14±0.42</td><td>43.55±0.48</td><td>89.66±0.40</td><td>68.66±0.65</td><td>73.88±1.16</td></tr><tr><td>PRAT (ours)</td><td>42.69±2.66</td><td>39.60±2.25</td><td>90.08±0.62</td><td>53.78±0.66</td><td>92.24±0.68</td><td>85.04±0.48</td><td>78.04±1.19</td></tr></table>

Table 4 extends the comparison to Platonov et al. [28] benchmarks. PRAT is best or statistically tied on four of the seven datasets (Cham.-filt., roman-empire, amazon-ratings, tolokers), ranks 2nd on Sq.-filt., and sits 0.05pp below the third-placed model on minesweeper and 0.01pp below GT-sep at rank 3 on questions. Only two of those four are wins; the other two are ties: against SAGE on amazon-ratings a Welch test gives $p = 0 . 5 5$ , and against FAGCN on filtered Chameleon p = 0.52 (Section G.1). Across the seven benchmarks PRAT therefore outperforms or remains competitive with all standard and heterophily-adapted GNNs reported by Platonov et al. [28]. The ResNet row is graph-aware on every dataset (ResNet+SGC or ResNet+adj) and outscores GCN on filtered Chameleon, yet PRAT leads it on all seven, by 1.23 to 21.36 points. On filtered Chameleon, where the duplicate nodes responsible for train–test leakage in the original benchmark are removed, the gain survives; removing that leakage shows the gain is not an artifact of it, though it does not by itself rule out memorization of other kinds. These results support our central thesis: the geometry of the latent space, not the depth of aggregation, is the dominant factor for learning under heterophily.

![](images/87e4b15334b2358738e40ded76d6bda65462e777bb0abc83b9f2670baded89da.jpg)

![](images/58e3e11cf077c5ae4849be4a1a40ab138c7045fe5354d5d5a257c2847ecec3e9.jpg)  
Figure 2: Accuracy across edge homophily levels, within the attention family.  
Figure 3: Ratio of positive eigenvalues of the learned metric in PRAT across varying homophily levels.

![](images/e8b35a6f557683ad2c82fabc6ea1aa0260739f0a009694a1cad40bfad0ceff14.jpg)

![](images/53c009f7a41ae6aa0ef8c47ca2cc58dc51db7f4a2b9ffcbf66a8fe327742ec50.jpg)  
(a) $h _ { E } = 0 . 1$ (heterophilic)

![](images/190e1af5b38ff93ab968d00afaf19fd64304ef1084d23d7bd042164a4822a5b8.jpg)

![](images/ae8a38ffb3cea8081a019e2fedc05ce4679c93b11d323da8323e7b4513622c1a.jpg)  
(b) $h _ { E } = 0 . 9$ (homophilic)  
Figure 4: Attention score distributions of PRAT and GAT. Each heatmap gives mean attention from source class (row) to target class (column).

## 4.4 Validation of Geometric Hypothesis Based on Synthetic Data

Generated graphs let us vary one structural property at a time. Nodes, edges, features and classes are held fixed across all of them, and every cell is five random $6 0 / 2 0 / 2 0$ splits. Section D gives the generator.

Varying Homophily level We generated synthetic graphs with edge homophily $\begin{array} { r l } { h _ { E } } & { { } \in } \end{array}$ $\{ 0 , 0 . 0 5 , \ldots , 1 . 0 \}$ and evaluated classification accuracy. The comparison stays inside the attention family, so the receptive field is fixed and only the metric differs: GAT and GATv2 keep it positive definite, FAGCN signs the aggregation weights without touching the geometry, and PRAT makes the metric itself indefinite. In Figure 2, PRAT holds its accuracy across the sweep while the other three fall away together below $h _ { E } = 0 . 5$ . The gap to FAGCN averages 40.6 points over the nine levels at $h _ { E } \leq 0 . 4$ . The curves converge above $h _ { E } = 0 . 8 $ , where a positive-definite metric is no longer a mismatch. Signed weights alone do not substitute for an indefinite metric.

Figure 3 tracks the ratio of positive eigenvalues in the learned metric: as homophily falls the negative share rises, so the signature the model settles on is not imposed but follows the data. On real graphs the learned Λ instead keeps the initial sign of every entry with $| \lambda | > 1 0 ^ { - 4 }$ in $\Lambda _ { \mathcal { G } , \mathrm { t r a i n } }$ , which Section J.1 reports.

Figure 4 visualizes the attention scores of PRAT and GAT, aggregated by source-target label pairs across all edges. Unlike GAT, PRAT concentrates attention on same-class connections (the diagonal) at $h _ { E } = 0 . 1$ , suppressing dissimilar neighbors while keeping the node’s own signal, but does not do so at $h _ { E } = 0 . 9$ . The metric changes how attention uses the same neighborhood.

Varying Graph Signature Index To investigate the influence of higher-order structural patterns beyond simple homophily, we generated synthetic graphs with a fixed homophily level $( h _ { E } = 0 . 5 )$ but varying graph signature indices $s _ { \mathcal { G } } \in \bar { \{ ( 5 , 0 , 0 ) , \bar { ( 4 , 0 , 1 ) } , ( 3 , 0 , 2 ) \} }$ .

In Figure 5, baseline performance fluctuates significantly with the graph signature index. Specifically, the standard deviation of accuracy across different signatures is 3.99 for GAT and 3.42 for GATv2. PRAT varies by 0.68. Two readings follow: PRAT is the less sensitive of the three to latent structure, and $s g$ separates graphs that a scalar $h _ { E }$ cannot, since these three share an $h _ { E }$ and behave differently.

![](images/7d146795aa4286f43a90147e26dcfc0aea57e02a234f6d4038a2bc5040e967e8.jpg)  
Figure 5: Performance comparison under fixed $h _ { E } = 0 . 5$ across different graph signature indices.

## 4.5 Ablation Study

To evaluate the effectiveness of graph-aware metric initialization in PRAT, we ran an ablation with two variants: (1) $\mathrm { P R A T _ { p o s } }$ , whose learnable Λ is constrained to be positive definite, and (2) PRAT<sub>fixed</sub>, whose Λ is frozen at its signature initialization.

In Table 5 the gap between PRAT and $\mathrm { P R A T _ { p o s } }$ falls on the graphs whose signature has negative eigenvalues: Chameleon, Squirrel, Cornell and Texas all lose under the constraint, while Cora, PubMed and Citeseer, whose signatures are strictly positive, are unchanged. Section I.1 repeats the comparison on the Platonov suite with an exact paired test: both significant losses are on indefinite graphs and no graph

Table 5: Ablation study results.
<table><tr><td>Dataset</td><td>PRAT</td><td> $\mathrm { P R A T _ { f i x e d } }$ </td><td> $\mathrm { P R A T _ { p o s } }$ </td></tr><tr><td>Cora</td><td> $8 8 . 4 2 \pm 1 . 1 7$ </td><td> ${ \bf 8 8 . 4 9 \pm 1 . 2 7 }$ </td><td> $8 8 . 1 8 { \pm } 1 . 1 0 $ </td></tr><tr><td>PubMed</td><td> $\mathbf { 9 0 . 0 3 \pm 0 . 4 7 }$ </td><td>89.94±0.46</td><td> $8 9 . 8 7 { \pm } 0 . 4 5 $ </td></tr><tr><td>Citeseer</td><td> $7 8 . 0 6 \pm 1 . 3 7$ </td><td> $7 7 . 9 3 \pm 1 . 3 8 $ </td><td> ${ \bf 7 8 . 0 9 \pm 1 . 4 5 }$ </td></tr><tr><td>Chameleon</td><td> ${ \bf 6 8 . 7 3 \pm 1 . 7 4 }$ </td><td>68.40±2.42 65.64±1.60</td><td></td></tr><tr><td>Squirrel</td><td> ${ \pm } 6 . 9 6 { \pm } 1 . 5 1 $ </td><td>54.01±2.08 53.47±1.43</td><td></td></tr><tr><td>Actor</td><td> $\pm 7 . 2 6 \pm 1 . 4 4$ </td><td>36.93±1.21 36.69±0.99</td><td></td></tr><tr><td>Cornell</td><td> ${ \bf 7 1 . 8 9 \pm 4 . 7 1 }$ </td><td>55.15±8.97</td><td>57.58±8.87</td></tr><tr><td>Texas</td><td> $\mathbf { 8 4 . 3 2 \pm 5 . 1 0 }$ </td><td>74.89±5.70</td><td>76.78±2.90</td></tr><tr><td>Average</td><td>71.96</td><td>68.22</td><td>68.29</td></tr></table>

without a negative eigenvalue changes significantly, while filtered Squirrel, whose negative eigenvalues sit near zero, ties. A negative signature is therefore necessary for a significant loss, and Section I.2 finds no significant rank correlation between n<sub>−</sub> and what signed or high-pass baselines gain, so we offer $s _ { \mathcal { G } }$ as a diagnostic and not as a predictor.

The two variants isolate different things. $\mathrm { P R A T _ { p o s } }$ removes the freedom to be indefinite and starts Λ at a scaled identity, so it measures the sign structure. $\mathrm { P R A T _ { f i x e d } }$ freezes Λ but not the metric, since $\mathbf { D } = ( \mathbf { W } ^ { \prime } ) ^ { \top } \boldsymbol { \Lambda } \mathbf { W } ^ { \prime }$ keeps adapting through the projection, so it measures whether the inertia has to be supplied at initialization. Its drop sits on the small graphs, 16.7 points on Cornell and 9.4 on Texas, where the training-split signature varies most; on the large Platonov graphs the two are within noise. Learning Λ earns its place where the signature is noisy, not where it is clean.

Which part of the signature carries the effect differs by graph. In Section J.1 the only control that loses decisively on roman-empire deletes the negative directions and keeps the magnitudes, 2.26 points at $p = 0 . 0 0 2$ , while sign-only, sign-flipped and three random draws stay within 0.7. On filtered Chameleon the magnitudes matter too, since every random draw loses 1.63 to 2.62 points.

## 5 Related Works

GNNs for Heterophily. Standard GNNs assume homophily, acting as low-pass filters that smooth features across edges. Heterophily-adapted models answer this architecturally: Geom-GCN [27] and $H _ { 2 } \mathrm { G C N }$ [49] separate aggregation from the immediate neighborhood, spectral methods such as GPR-GNN [7] and BernNet [14] learn polynomial filters that mix high- and low-frequency signals, and attention-based models use signed edge weights [1, 44] to represent repulsion. All of these operate within a Euclidean space.

Pseudo-Riemannian graph models. We are not the first to bring an indefinite geometry to graphs. Q-GCN [42] embeds node features on a pseudo-hyperboloid $\bar { \mathcal { Q } } _ { \beta } ^ { s , t }$ and convolves there through diffeomorphic exponential and logarithmic maps; Q-GT [18] carries the same ambient space into a transformer with linear attention. Three differences separate them from PRAT. First, where the indefinite form lives: in both it is the canonical scalar product of the manifold the nodes are embedded on, whereas in PRAT it is the attention logit itself, a signed squared pseudo-distance $\Delta ^ { \top } \mathbf { D } \Delta . \mathrm { Q } – \mathrm { G T } ^ { \ast } \mathrm { s }$ attention weights come from a positive-definite kernel, so its indefinite product never reaches a score. Second, how the signature is chosen: Q-GCN picks the number of time dimensions by a curvature heuristic and a sweep, which its own conclusion names as an open limitation, and Q-GT selects its own signature $( p , q )$ , the counts of positive and negative dimensions, offline by matching a sectionalcurvature distribution. Neither criterion reads the label topology, so neither can see heterophily, while $s _ { \mathcal { G } }$ is built from that topology and Λ is learned from the spectrum of $S _ { \mathcal { G } , \mathrm { t r a i n } } .$ . Third, coverage: ${ \bf Q - G C N \vec { s } }$ labeled evaluation is citation networks and Airport, Q-GT adds two non-homophilous social graphs, and neither reports the WebKB, Wikipedia or Platonov suites used here. We claim a learnable label-initialized indefinite metric inside attention together with a diagnostic for when it should help, not priority for pseudo-Riemannian geometry on graphs. ACM-GCN [21] is closest on the channel side, mixing aggregation, diversification and identity channels node-wise, which puts the sign in the channel domain rather than in a metric. Its published numbers use different splits, so we cite it as a counterpart, not a comparable row.

Geometric Graph Learning. Hyperbolic GNNs [6, 47] use constant negative curvature for hierarchical structure, and ARGNN [41] makes the metric node-wise and learnable. Both stay Riemannian $( g > 0 )$ ; only an indefinite metric admits a negative squared distance [17].

Graph Attention. Standard $L _ { 2 }$ attention can be reformulated as an SDP and is biased accordingly [22]. DGAT [20] uses spectral distances for rewiring in a separate preprocessing step, and global Graph Transformers [46, 29, 9] reach long range at quadratic cost while remaining Euclidean. PRAT instead reads attention as a tangent-space interaction on a learnable manifold.

Statistical Foundations. The Generalised Random Dot Product Graph (GRDPG) [31] shows statistically that indefinite inner products are needed to model disassortative networks. Theorem 2.5 reaches a complementary conclusion from the geometric side. GRDPG gives a fixed spectral embedding. PRAT learns the indefinite metric end-to-end, so attention moves with it at every layer.

## Conclusion

We examined the positive-definite geometric prior carried by diffusion, hyperbolic and dot-product attention models, and asked what it costs when the labels induce an indefinite form. Theorem 2.5 answers in one direction: where $S _ { \mathcal { G } }$ is indefinite and the pushforward is surjective, no diffeomorphism of the target turns the Riemannian pullback into $f ^ { \ast } S _ { \mathcal { G } }$ . That is a statement about those model families and about graphs with a negative signature, not about heterophily as a whole.

The empirical picture matches that scope. The Graph Signature Index says which graphs carry a negative signature. In the paired Platonov ablation every significant loss from constraining the metric falls on such a graph, and no other graph changes significantly. The condition is necessary for a detectable loss. Several of our heterophilic benchmarks do not meet it, so the diagnosis, not heterophily as such, is a precondition of the indefinite metric’s advantage.

Limitations and future work. The signature is an initialization prior rather than an architectural requirement, since the forward pass never reads labels and Λ is an ordinary learnable parameter. Even so, four limits are worth stating. It is computed from training-node labels, which ties the scheme as presented to (semi-)supervised node classification. Section J.2 weakens that, but a full self-supervised treatment, objective as well as initialization, is outside our scope. Inductive use needs no recomputation in the forward pass, yet signature shift between training and test graphs is open. Graph-level tasks are out of reach as the method stands, since $S _ { \mathcal { G } }$ is undefined without node labels. Finally, rank $\mathbf { \eta } ( \mathbf { D } ) \leq \vert L \vert$ bounds the structure the metric can carry when classes are few, and on the binary benchmarks the inertia reduces to a single bit. Dynamic curvature and directed graphs are natural next steps.

## Acknowledgments and Disclosure of Funding

This research was supported by the “Advanced GPU Utilization Support Program” funded by the Government of the Republic of Korea (Ministry of Science and ICT). Sun Woo Park would like to sincerely thank Max Planck Institute for Mathematics for its kind and generous support. The authors declare no competing interests.

## References

[1] Deyu Bo, Xiao Wang, Chuan Shi, and Huawei Shen. Beyond low-frequency information in graph convolutional networks. In AAAI. AAAI Press, 2021.

[2] Cristian Bodnar, Francesco Di Giovanni, Benjamin Paul Chamberlain, Pietro Liò, and Michael M. Bronstein. Neural sheaf diffusion: A topological perspective on heterophily and oversmoothing in GNNs. In Advances in Neural Information Processing Systems, volume 35, pages 18527–18541, 2022.

[3] Shaked Brody, Uri Alon, and Eran Yahav. How attentive are graph attention networks? In International Conference on Learning Representations, 2022. URL https://openreview. net/forum?id=F72ximsx7C1.

[4] Michael M. Bronstein, Joan Bruna, Yann LeCun, Arthur Szlam, and Pierre Vandergheynst. Geometric deep learning: Going beyond euclidean data. IEEE Signal Processing Magazine, 34 (4):18–42, 2017. doi: 10.1109/MSP.2017.2693418.

[5] Benjamin Paul Chamberlain, James Rowbottom, Maria Goronova, Stefan Webb, Emanuele Rossi, and Michael M Bronstein. Grand: Graph neural diffusion. Proceedings of the 38th International Conference on Machine Learning, (ICML) 2021, 18-24 July 2021, Virtual Event, 2021.

[6] Ines Chami, Zhitao Ying, Christopher Ré, and Jure Leskovec. Hyperbolic graph convolutional neural networks. In Advances in Neural Information Processing Systems, pages 4869–4880, 2019.

[7] Eli Chien, Jianhao Peng, Pan Li, and Olgica Milenkovic. Adaptive universal generalized pagerank graph neural network. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=n6jl7fLxrP.

[8] Yun Young Choi, Sun Woo Park, Youngho Woo, and U Jin Choi. Cycle to clique (cy2c) graph neural network: A sight to see beyond neighborhood aggregation. In The eleventh international conference on learning representations, 2022.

[9] Yun Young Choi, Sun Woo Park, Minho Lee, and Youngho Woo. Topology-informed graph transformer. In Proceedings ofthe Geometry-grounded Representation Learning and Generative Modeling Workshop (GRaM), volume 251 of Proceedings ofMachine Learning Research, pages 20–34. PMLR, 2024.

[10] Lun Du, Xiaozhou Shi, Qiang Fu, Xiaojun Ma, Hengyu Liu, Shi Han, and Dongmei Zhang. GBK-GNN: Gated bi-kernel graph neural networks for modeling both homophily and heterophily. In Proceedings of the ACM Web Conference 2022 (WWW), 2022.

[11] Johannes Gasteiger, Aleksandar Bojchevski, and Stephan Günnemann. Predict then propagate: Graph neural networks meet personalized pagerank. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=H1gL-2A9Ym.

[12] William L. Hamilton, Rex Ying, and Jure Leskovec. Inductive representation learning on large graphs. In NIPS, 2017.

[13] Allen Hatcher. Algebraic Topology. Cambridge University Press, 2002.

[14] Mingguo He, Zhewei Wei, Zengfeng Huang, and Hongteng Xu. Bernnet: Learning arbitrary graph spectral filters via bernstein approximation. In NeurIPS, 2021.

[15] Feng Ji, Yanan Zhao, Kai Zhao, Hanyang Meng, Jielong Yang, and Wee Peng Tay. Rethinking graph neural networks from a geometric perspective of node features. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview. net/forum?id=lBMRmw59Lk.

[16] Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations, 2017. URL https: //openreview.net/forum?id=SJU4ayYgl.

[17] Marc Law and Jos Stam. Ultrahyperbolic representation learning. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 1668–1678. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper\_files/paper/2020/file/ 123b7f02433572a0a560e620311a469c-Paper.pdf.

[18] Viet Quan Le and Cuong Van Ta. Pseudo-riemannian graph transformer. In Advances in Neural Information Processing Systems, volume 38, pages 15951–15983, 2025.

[19] Xiang Li, Renyu Zhu, Yao Cheng, Caihua Shan, Siqiang Luo, Dongsheng Li, and Weining Qian. Finding global homophily in graph neural networks when meeting heterophily. In International Conference on Machine Learning (ICML), 2022.

[20] Qincheng Lu, Jiaqi Zhu, Sitao Luan, and Xiao-Wen Chang. Representation learning on heterophilic graph with directional neighborhood attention, 2024. URL https://arxiv.org/ abs/2403.01475.

[21] Sitao Luan, Chenqing Hua, Qincheng Lu, Jiaqi Zhu, Mingde Zhao, Shuyuan Zhang, Xiao-Wen Chang, and Doina Precup. Revisiting heterophily for graph neural networks. In Advances in Neural Information Processing Systems, 2022.

[22] Liheng Ma, Soumyasundar Pal, Yingxue Zhang, Philip H. S. Torr, and Mark Coates. Plain transformers can be powerful graph learners. Transactions on Machine Learning Research, 2026. URL https://openreview.net/forum?id=bEmDvP0fdv.

[23] Sunil Kumar Maurya, Xin Liu, and Tsuyoshi Murata. Simplifying approach to node classification in Graph Neural Networks. Journal ofComputational Science, 62:101695, 2022.

[24] John Douglas Moore. Lectures on differential geometry. Lecture notes, University of California, Santa Barbara. https://web.math.ucsb.edu/\~moore/riemanniangeometry.pdf, 2009.

[25] Sun Woo Park, Yun Young Choi, Dosang Joe, U Jin Choi, and Youngho Woo. The pwlr graph representation: A persistent weisfeiler-lehman scheme with random walks for graph classification. In Topological, Algebraic and Geometric Learning Workshops 2022, pages 287–297. PMLR, 2022.

[26] Sun Woo Park, Yun Young Choi, U Jin Choi, and Youngho Woo. A functorial formulation of neighborhood aggregating deep learning. arXiv preprint arXiv:2604.24672, 2026.

[27] Hongbin Pei, Bingzhe Wei, Kevin Chen-Chuan Chang, Yu Lei, and Bo Yang. Geom-gcn: Geometric graph convolutional networks. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=S1e2agrFvS.

[28] Oleg Platonov, Denis Kuznedelev, Michael Diskin, Artem Babenko, and Liudmila Prokhorenkova. A critical look at the evaluation of GNNs under heterophily: Are we really making progress? In International Conference on Learning Representations, 2023.

[29] Ladislav Rampášek, Michael Galkin, Vijay Prakash Dwivedi, Anh Tuan Luu, Guy Wolf, and Dominique Beaini. Recipe for a general, powerful, scalable graph transformer. Advances in Neural Information Processing Systems, 35:14501–14515, 2022.

[30] Benedek Rozemberczki, Carl Allen, and Rik Sarkar. Multi-scale attributed node embedding. Journal ofComplex Networks, 9(2):cnab014, 2021.

[31] Patrick Rubin-Delanchy, Joshua Cape, Minh Tang, and Carey E. Priebe. A statistical interpretation of spectral embedding: the generalised random dot product graph. Journal of the Royal Statistical Society Series B, 84(4):1446–1473, 2022.

[32] Prithviraj Sen, Galileo Namata, Mustafa Bilgic, Lise Getoor, Brian Galligher, and Tina Eliassi-Rad. Collective classification in network data. AI magazine, 29(3):93–93, 2008.

[33] Yunsheng Shi, Zhengjie Huang, Shikun Feng, Hui Zhong, Wenjing Wang, and Yu Sun. Masked label prediction: Unified message passing model for semi-supervised classification. In Proceed ings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21, pages 1548–1554. International Joint Conferences on Artificial Intelligence Organization, 2021. doi: 10.24963/ijcai.2021/214.

[34] J. J. Sylvester. Xix. a demonstration of the theorem that every homogeneous quadratic polynomial is reducible by real orthogonal substitutions to the form of a sum of positive and negative squares. The London, Edinburgh, and Dublin Philosophical Magazine and Journal ofScience, 4(23):138–142, 1852.

[35] Jie Tang, Jimeng Sun, Chi Wang, and Zi Yang. Social influence analysis in large-scale networks. In Proceedings ofthe 15th ACM SIGKDD international conference on Knowledge discovery and data mining, pages 807–816, 2009.

[36] Jake Topping, Francesco Di Giovanni, Benjamin Paul Chamberlain, Xiaowen Dong, and Michael M. Bronstein. Understanding over-squashing and bottlenecks on graphs via curvature. In International Conference on Learning Representations, 2022. URL https://openreview. net/forum?id=7UmjRGzp-A.

[37] Loring Tu. Differential Geometry: Connections, Curvature, and Characteristic Classes. Springer, 2017.

[38] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[39] Petar Velickoviˇ c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua´ Bengio. Graph attention networks. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=rJXMpikCZ.

[40] Xiyuan Wang and Muhan Zhang. How powerful are spectral graph neural networks. In International Conference on Machine Learning (ICML), 2022.

[41] Xudong Wang, Tongxin Li, Chris Ding, and Jicong Fan. Adaptive riemannian graph neural networks. In Proceedings of the AAAI Conference on Artificial Intelligence, 2026.

[42] Bo Xiong, Shichao Zhu, Nico Potyka, Shirui Pan, Chuan Zhou, and Steffen Staab. Pseudoriemannian graph convolutional networks. In Advances in Neural Information Processing Systems, 2022.

[43] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? In International Conference on Learning Representations, 2019. URL https: //openreview.net/forum?id=ryGs6iA5Km.

[44] Yujun Yan, Milad Hashemi, Kevin Swersky, Yaoqing Yang, and Danai Koutra. Two sides of the same coin: Heterophily and oversmoothing in graph convolutional neural networks. In 2022 IEEE International Conference on Data Mining (ICDM), pages 1287–1292. IEEE, 2022.

[45] Zhilin Yang, William Cohen, and Ruslan Salakhudinov. Revisiting semi-supervised learning with graph embeddings. In International conference on machine learning, pages 40–48. PMLR, 2016.

[46] Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin Ke, Di He, Yanming Shen, and Tie-Yan Liu. Do transformers really perform badly for graph representation? Advances in neural information processing systems, 34:28877–28888, 2021.

[47] Yiding Zhang, Xiao Wang, Chuan Shi, Xunqiang Jiang, and Yanfang Fanny Ye. Hyperbolic graph attention network. IEEE Transactions on Big Data, 2021.

[48] Yilun Zheng, Sitao Luan, and Lihui Chen. What is missing for graph homophily? disentangling graph homophily for graph neural networks. In Advances in Neural Information Processing Systems, volume 37, pages 68406–68452, 2024.

[49] Jiong Zhu, Yujun Yan, Lingxiao Zhao, Mark Heimann, Leman Akoglu, and Danai Koutra. Beyond homophily in graph neural networks: Current limitations and effective designs. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 7793–7804. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper\_files/paper/2020/file/ 58ae23d878a47004366189884c2f8440-Paper.pdf.

[50] Jiong Zhu, Ryan A. Rossi, Anup Rao, Tung Mai, Nedim Lipka, Nesreen K. Ahmed, and Danai Koutra. Graph neural networks with heterophily. Proceedings ofthe AAAI Conference on Artificial Intelligence, 35(12):11168–11176, May 2021. URL https://ojs.aaai.org/ index.php/AAAI/article/view/17332.

## A Theoretical analysis and proofs

## A.1 Normalization of the Graph Signature Matrix

The $( i , j )$ -th entry of $S _ { \mathcal { G } }$ is defined as $s _ { i j } = | E _ { i j } | / ( n _ { i } \cdot n _ { j } )$ , where $| E _ { i j } |$ is the number of ordered pairs $( v , w )$ of adjacent nodes with $l ( v ) = i$ and $l ( w ) = j$ (so an edge inside class i is counted twice, which is the factor 2 of the diagonal entries in Appendix $_ { \mathrm { A } . 6 ) }$ , and $n _ { i } , n _ { j }$ are the class sizes. This quantity is the edge density between classes i and $j \colon$ the probability that a randomly chosen pair $( v , w )$ with $l ( v ) = i , l ( w ) = j$ is connected. This normalization is standard in stochastic block models (SBMs) and carries a clear physical interpretation: positive eigenvalues correspond to over-connected class pairs (homophilic tendency), while negative eigenvalues indicate under-connected pairs (heterophilic tendency).

The signature does not depend on the normalization by $n _ { i } \cdot n _ { j }$ Write C for the matrix of raw counts $| E _ { i j } |$ and $N _ { L } : = \mathrm { d i a g } ( n _ { 1 } , . . . , n _ { | L | } )$ . Since every class is nonempty, $N _ { L }$ is invertible, so $S _ { \mathcal { G } } = N _ { L } ^ { - 1 } C N _ { L } ^ { - 1 }$ is congruent to $C$ and has the same inertia by Theorem 2.2. The normalization acts on the eigenvalues instead. Those of $C$ scale with class sizes, so class imbalance enters their magnitudes, while those of $S _ { \mathcal { G } }$ are the spectrum of the connection-probability matrix above. Rescaled, they initialize Λ (Equation (5)), and Section J.1 shows that on filtered Chameleon their magnitudes matter beyond their signs. The critical homophily threshold in Theorem 2.3 is derived under this normalization, where the expected matrix takes the form $a I + b ( J - I )$ with eigenvalues in closed form, and the resulting $h _ { E } ^ { * } \stackrel { \star } { = } \sum n _ { k } ^ { 2 } / N ^ { 2 }$ explicitly accounts for class imbalance.

## A.2 A different characterization of $S _ { \mathcal { G } }$

Using the unfolding-tree perspective on local graph structure [25], we interpret entries of the graph signature matrix through depth 1 unfolding trees rooted at each node of $\mathcal { G }$

Definition A.1. For any node $v \in \mathcal V$ , we define the probability that a node with a given node label j is connected to a directed edge starting from node v as

$$
\begin{array} { r l } & { p _ { v , j } : = \mathbb { P } \bigg [ ( v , w ) \in { \mathcal E } \biggm | \imath ( w ) = j \bigg ] } \\ & { \qquad = \frac { \left| \left\{ w \in { \mathcal V } \mid l ( w ) = j , ( v , w ) \in { \mathcal E } \right\} \right| } { \left| \left\{ w \in { \mathcal V } \mid l ( w ) = j \right\} \right| } } \end{array}\tag{6}
$$

Lemma A.2. Given a graph G, each matrix entry of $S _ { \mathcal { G } }$ satisfies

$$
p _ { i , j } = { \frac { 1 } { | \{ v \in \mathcal { V } \mid l ( v ) = i \} | } } \sum _ { \stackrel { v \in \mathcal { V } } { l ( v ) = i } } p _ { v , j }\tag{7}
$$

Proof. We note that

$$
\begin{array} { l } { \displaystyle p _ { i , j } : = \frac { 1 } { \left| \left\{ v \in \mathcal { V } \mid l ( v ) = i \right\} \right| } \frac { \left| \left\{ \left( v , w \right) \in \mathcal { E } \mid l ( v ) = i , l ( w ) = j \right\} \right| } { \left| \left\{ v \in \mathcal { V } \mid l ( v ) = j \right\} \right| } } \\ { \displaystyle \quad = \frac { 1 } { \left| \left\{ v \in \mathcal { V } \mid l ( v ) = i \right\} \right| } \frac { \sum _ { \stackrel { v \in \mathcal { V } } { l ( v ) = i } } \left| \left\{ \left( v , w \right) \in \mathcal { E } \mid l ( w ) = j \right\} \right| } { \left| \left\{ w \in \mathcal { V } \mid l ( w ) = j \right\} \right| } } \\ { \displaystyle \quad = \frac { 1 } { \left| \left\{ v \in \mathcal { V } \mid l ( v ) = i \right\} \right| } \displaystyle \sum _ { \stackrel { l ( v ) = i } { l ( v ) = i } } p _ { v , j } } \end{array}\tag{8}
$$

In other words, each $( i , j )$ -th entry of $S _ { \mathcal { G } }$ is the expected value of the probability that a node with a preassigned label j lies in the depth 1 unfolding tree rooted at a node with label i.

## A.3 An example of graph signature index for graphs with two discrete node labels

Theorem A.3 (2 Discrete Node Labels). Let $\mathcal { G } : = ( \nu , \mathcal { E } )$ be an undirected graph without self-loops and with ${ \mathcal { E } } \neq \emptyset .$ . Let $l : \mathcal { V }  \{ 1 , 2 \}$ be the discrete node label function over the set of nodes $o f { \mathcal { G } } .$ Then thefollowing correspondence holds between $s g$ and entries of $S _ { \mathcal { G } } : = ( p _ { i , j } ) _ { 1 \leq i , j \leq 2 } :$

$$
\begin{array} { r } { s _ { \mathcal { G } } = ( 2 , 0 , 0 ) \iff p _ { 1 , 2 } < \sqrt { p _ { 1 , 1 } p _ { 2 , 2 } } } \\ { s _ { \mathcal { G } } = ( 1 , 1 , 0 ) \iff p _ { 1 , 2 } = \sqrt { p _ { 1 , 1 } p _ { 2 , 2 } } } \\ { s _ { \mathcal { G } } = ( 1 , 0 , 1 ) \iff p _ { 1 , 2 } > \sqrt { p _ { 1 , 1 } p _ { 2 , 2 } } } \end{array}\tag{9}
$$

Proof. Let $S _ { \mathcal { G } }$ be the graph signature matrix of $\mathcal { G } .$ . Its explicit form is given as

$$
S _ { \mathcal { G } } : = \binom { p _ { 1 , 1 } } { p _ { 2 , 1 } } \begin{array} { l } { p _ { 1 , 2 } } \\ { p _ { 2 , 2 } } \end{array}\tag{10}
$$

The characteristic polynomial of $S _ { \mathcal { G } }$ is given by

$$
( \lambda - p _ { 1 , 1 } ) ( \lambda - p _ { 2 , 2 } ) - p _ { 1 , 2 } ^ { 2 } = 0\tag{11}
$$

The eigenvalues of $S _ { \mathcal { G } }$ are given by

$$
\lambda = \frac { ( p _ { 1 , 1 } + p _ { 2 , 2 } ) \pm \sqrt { ( p _ { 1 , 1 } + p _ { 2 , 2 } ) ^ { 2 } - 4 ( p _ { 1 , 1 } p _ { 2 , 2 } - p _ { 1 , 2 } ^ { 2 } ) } } { 2 }\tag{12}
$$

Since $\mathcal { E } \neq \emptyset .$ , the trace $p _ { 1 , 1 } + p _ { 2 , 2 }$ is positive whenever $p _ { 1 , 1 } p _ { 2 , 2 } \geq p _ { 1 , 2 } ^ { 2 }$ , as otherwise all three entries vanish. Hence, the graph signature $s _ { \mathcal { G } }$ is determined by the sign of the term $p _ { 1 , 1 } p _ { 2 , 2 } - p _ { 1 , 2 } ^ { 2 }$ □

## A.4 Heuristics on heterophilic and homophilic graphs with two discrete node labels

The phase diagram for the graph $F : p _ { 1 , 2 } - \sqrt { p _ { 1 , 1 } p _ { 2 , 2 } } = 0$ is shown in Figure 6, where the region below $F$ characterizes graphs whose nodes with identical labels are more likely to be connected than those with distinct labels (i.e. homophilic graphs), whereas the region above $\dot { F }$ contains heterophilic graphs. A third region, defined by $\dot { F }$ , contains graphs whose nodes with identical labels are as likely to be connected as those with distinct labels. Note that the volume of the region below $F$ over $0 \leq p _ { 1 , 1 } \leq 1$ and $0 \leq p _ { 2 , 2 } \leq 1$ is given by

$$
\int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \sqrt { p _ { 1 , 1 } p _ { 2 , 2 } } d p _ { 1 , 1 } d p _ { 2 , 2 } = \frac { 4 } { 9 } .\tag{13}
$$

Consider the idealized model with 2 discrete node labels whose connectivity parameters $( p _ { 1 , 1 } , p _ { 1 , 2 } , p _ { 2 , 2 } )$ are uniformly distributed on $[ 0 , 1 ] ^ { 3 }$ , and let $P = \left( p _ { i , j } \right)$ be the symmetric $2 \times 2$ connectivity matrix, with inertia $\mathrm { I n } ( P )$ . The integral computed above shows that

$$
\mathbb { P } \left[ \mathrm { I n } ( P ) = ( i , j , k ) \right] = \left\{ { \begin{array} { l l } { \frac { 5 } { 9 } } & { \mathrm { i f } \left( i , j , k \right) = ( 1 , 0 , 1 ) } \\ { 0 } & { \mathrm { i f } \left( i , j , k \right) = ( 1 , 1 , 0 ) } \\ { \frac { 4 } { 9 } } & { \mathrm { i f } \left( i , j , k \right) = ( 2 , 0 , 0 ) } \end{array} } \right.\tag{14}
$$

![](images/0409c24da4561d476ffcbb3f11e72f33bdccbec788fc949cbb2eeae012c88c10.jpg)  
Figure 6: A phase diagram of graph signature index with two types of discrete node labels. The blue surface denotes the region where $F : p _ { 1 , 2 } = { \sqrt { p _ { 1 , 1 } p _ { 2 , 2 } } } .$

## A.5 Notion of homophily indices

Remark A.4. Earlier work measured homophily in undirected graphs G without self-loops by using the node homophily index defined as in [27]

$$
h _ { V } : = \frac { 1 } { | \mathcal { V } | } \sum _ { v \in \mathcal { V } } \frac { | \{ w \in \mathcal { V } \mid { _ { l ( w ) = l ( v ) } ^ { ( v , w ) \in \mathcal { E } } } \} | } { \deg ( v ) } ,\tag{15}
$$

or by using the edge homophily index defined as in [50]:

$$
h _ { E } : = \frac { | \{ ( u , v ) \in \mathcal { E } \mid l ( u ) = l ( v ) \} | } { | \mathcal { E } | } .\tag{16}
$$

A common way to distinguish homophilic from heterophilic graphs is to compute the homophily indices $h _ { V }$ and $h _ { E }$ and check whether their values are closer to 0 or 1. We demonstrate that the graph signature matrix allows one to closely analyze the relations among different node labels, a trait that numerical homophily indices do not necessarily share.

## A.6 Proof of Theorem 2.3

Setup and Notation. Consider an undirected graph $\mathcal { G } : = ( \nu , \mathcal { E } )$ with $| L | \geq 2$ class labels, where class i contains $n _ { i } \geq 1$ nodes. Let $N = \textstyle \sum _ { i = 1 } ^ { | L | }$ n<sub>i</sub> denote the total number of nodes. The graph has $| \mathcal { E } | > 0$ edges with edge homophily level $\bar { h _ { E } } \doteq [ 0 , 1 ]$ , partitioned into $h _ { E } | \mathcal { E } |$ intra-class edges and $( 1 - h _ { E } ) | \bar { \mathcal { E } } |$ inter-class edges.

We adopt a proportional allocation model:

• Intra-class edges are distributed among classes proportionally to $n _ { i } ^ { 2 }$ (the denominator of $s _ { i , i }$ in Definition 2.1).

• Inter-class edges are distributed among class pairs $( i , j )$ with $i \neq j$ proportionally to $n _ { i } n _ { j }$

Define the following quantities for notational convenience:

$$
\Sigma _ { 2 } : = \sum _ { k = 1 } ^ { | L | } n _ { k } ^ { 2 } , \qquad \Sigma _ { \times } : = \sum _ { i < j } n _ { i } n _ { j } = \frac { N ^ { 2 } - \Sigma _ { 2 } } { 2 } .\tag{17}
$$

Expected Edge Counts. Under the proportional allocation model, the expected number of intraclass edges in class i is

$$
\mathbb { E } [ X _ { i } ] = h _ { E } | \mathcal { E } | \cdot \frac { n _ { i } ^ { 2 } } { \sum _ { 2 } } ,\tag{18}
$$

and the expected number of inter-class edges between classes i and $j$ (where $i \neq j )$ is

$$
\mathbb { E } [ Y _ { i j } ] = ( 1 - h _ { E } ) | \mathcal { E } | \cdot \frac { n _ { i } n _ { j } } { \Sigma _ { \times } } = \frac { 2 ( 1 - h _ { E } ) | \mathcal { E } | \cdot n _ { i } n _ { j } } { N ^ { 2 } - \Sigma _ { 2 } } .\tag{19}
$$

Expected Signature Matrix Entries. The expected signature matrix $S _ { \mathcal { G } } = \mathbb { E } [ \widehat { S } _ { G } ]$ has entries given by Definition 2.1. For diagonal entries:

$$
( S _ { \mathcal { G } } ) _ { i i } = \frac { 2 \mathbb { E } [ X _ { i } ] } { n _ { i } ^ { 2 } } = \frac { 2 h _ { E } | \boldsymbol { \mathcal { E } } | \cdot n _ { i } ^ { 2 } / \Sigma _ { 2 } } { n _ { i } ^ { 2 } } = \frac { 2 h _ { E } | \boldsymbol { \mathcal { E } } | } { \Sigma _ { 2 } } = : a .\tag{20}
$$

For off-diagonal entries:

$$
( S _ { \mathcal { G } } ) _ { i j } = \frac { \mathbb { E } [ Y _ { i j } ] } { n _ { i } n _ { j } } = \frac { 2 ( 1 - h _ { E } ) | \mathcal { E } | \cdot n _ { i } n _ { j } / ( N ^ { 2 } - \Sigma _ { 2 } ) } { n _ { i } n _ { j } } = \frac { 2 ( 1 - h _ { E } ) | \mathcal { E } | } { N ^ { 2 } - \Sigma _ { 2 } } = : b .\tag{21}
$$

Both diagonal and off-diagonal entries are constant across all indices, yielding the matrix structure:

$$
S _ { \mathcal { G } } = a \cdot I _ { | L | } + b \cdot ( J _ { | L | } - I _ { | L | } ) = ( a - b ) I _ { | L | } + b J _ { | L | } ,\tag{22}
$$

where $I _ { | L | }$ is the $| L | \times | L |$ identity matrix and ${ \cal J } _ { | L | }$ is the $| L | \times | L |$ all-ones matrix.

Eigenvalue Computation. The all-ones matrix ${ \cal J } _ { | L | }$ has eigenvalue |L| with eigenvector $\mathbf { 1 } _ { | L | } =$ $( 1 , \ldots , 1 ) ^ { \top }$ , and eigenvalue 0 with multiplicity $| L | - 1$ for eigenvectors orthogonal to $\mathbf { 1 } _ { | L | }$ . Therefore, the first eigenvalue (with eigenvector $\mathbf { 1 } _ { | L | } )$ is

$$
\lambda _ { 1 } = ( a - b ) + | L | b = a + ( | L | - 1 ) b ,\tag{23}
$$

and the remaining eigenvalues (with eigenvectors in $\mathbf { 1 } _ { | L | } ^ { \perp }$ , multiplicity $| L | - 1 )$ are

$$
\lambda _ { 2 } = \cdot \cdot \cdot = \lambda _ { | L | } = a - b .\tag{24}
$$

Substituting the definitions of a and b:

$$
\begin{array} { l } { \displaystyle { a - b = \frac { 2 h _ { E } | \mathcal { E } | } { \Sigma _ { 2 } } - \frac { 2 ( 1 - h _ { E } ) | \mathcal { E } | } { N ^ { 2 } - \Sigma _ { 2 } } = 2 | \mathcal { E } | \left( \frac { h _ { E } } { \Sigma _ { 2 } } - \frac { 1 - h _ { E } } { N ^ { 2 } - \Sigma _ { 2 } } \right) } } \\ { \displaystyle { ~ = 2 | \mathcal { E } | \cdot \frac { h _ { E } ( N ^ { 2 } - \Sigma _ { 2 } ) - ( 1 - h _ { E } ) \Sigma _ { 2 } } { \Sigma _ { 2 } ( N ^ { 2 } - \Sigma _ { 2 } ) } . } } \end{array}\tag{25}
$$

Expanding the numerator:

$$
h _ { E } ( N ^ { 2 } - \Sigma _ { 2 } ) - ( 1 - h _ { E } ) \Sigma _ { 2 } = h _ { E } N ^ { 2 } - h _ { E } \Sigma _ { 2 } - \Sigma _ { 2 } + h _ { E } \Sigma _ { 2 } = h _ { E } N ^ { 2 } - \Sigma _ { 2 } .\tag{26}
$$

Therefore:

$$
\lambda _ { 2 } = \cdots = \lambda _ { | L | } = \frac { 2 | \mathcal { E } | ( h _ { E } N ^ { 2 } - \Sigma _ { 2 } ) } { \Sigma _ { 2 } ( N ^ { 2 } - \Sigma _ { 2 } ) } .\tag{27}
$$

Critical Threshold. The sign of $\lambda _ { 2 } , \ldots , \lambda _ { | L | }$ is determined by the numerator $\left( h _ { E } N ^ { 2 } - \Sigma _ { 2 } \right)$

$$
h _ { E } N ^ { 2 } - \Sigma _ { 2 } \stackrel { \geq } { \underline { { { \asymp } } } } 0 \Longleftrightarrow \quad h _ { E } \stackrel { \geq } { \underline { { { \asymp } } } } \frac { \Sigma _ { 2 } } { N ^ { 2 } } = \frac { \sum _ { k = 1 } ^ { | L | } n _ { k } ^ { 2 } } { N ^ { 2 } } .\tag{28}
$$

Thus, the critical threshold is:

$$
h _ { E } ^ { * } = \frac { \sum _ { k = 1 } ^ { | L | } n _ { k } ^ { 2 } } { N ^ { 2 } } = \sum _ { k = 1 } ^ { | L | } \left( \frac { n _ { k } } { N } \right) ^ { 2 } .\tag{29}
$$

Positivity of $\lambda _ { 1 }$ . We verify that $\lambda _ { 1 } > 0$ for all valid parameter ranges:

$$
\lambda _ { 1 } = a + ( | L | - 1 ) b = \frac { 2 h _ { E } | \mathcal { E } | } { \Sigma _ { 2 } } + \frac { 2 ( | L | - 1 ) ( 1 - h _ { E } ) | \mathcal { E } | } { N ^ { 2 } - \Sigma _ { 2 } } .\tag{30}
$$

Since $h _ { E } \in [ 0 , 1 ] , | \mathcal { E } | > 0 , \Sigma _ { 2 } > 0$ , and $\begin{array} { r } { N ^ { 2 } - \Sigma _ { 2 } = 2 \sum _ { i < j } n _ { i } n _ { j } > 0 } \end{array}$ for $| L | \geq 2$ , we have $\lambda _ { 1 } > 0$ always.

Signature Classification. Combining the above results:

$$
\bullet \operatorname { I f } h _ { E } > h _ { E } ^ { * } ; \lambda _ { 1 } > 0 { \mathrm { ~ a n d ~ } } \lambda _ { 2 } = \cdot \cdot \cdot = \lambda _ { | L | } > 0 , \operatorname { s o } s _ { \mathcal { G } } = ( | L | , 0 , 0 ) .
$$

$$
\bullet \operatorname { I f } h _ { E } = h _ { E } ^ { \ast } \colon \lambda _ { 1 } > 0 \mathrm { ~ a n d ~ } \lambda _ { 2 } = \cdot \cdot \cdot = \lambda _ { | L | } = 0 , \mathsf { s o } s _ { \mathcal { G } } = ( 1 , | L | - 1 , 0 ) .
$$

• I ${ } ^ { ? } h _ { E } < h _ { E } ^ { * } ; \lambda _ { 1 } > 0 \mathrm { ~ a n d ~ } \lambda _ { 2 } = \cdot \cdot \cdot = \lambda _ { | L | } < 0 , \mathsf { s o } s _ { \mathcal { G } } = ( 1 , 0 , | L | - 1 ) .$

Reduction to Balanced Case. For balanced classes where $n _ { i } = n$ for all $i \in \{ 1 , \ldots , | L | \}$

$$
h _ { E } ^ { \ast } = \frac { \sum _ { k = 1 } ^ { | L | } n ^ { 2 } } { ( | L | n ) ^ { 2 } } = \frac { | L | n ^ { 2 } } { | L | ^ { 2 } n ^ { 2 } } = \frac { 1 } { | L | } .\tag{31}
$$

Probabilistic Interpretation. The critical threshold $\begin{array} { r } { h _ { E } ^ { * } = \sum _ { k = 1 } ^ { | L | } ( n _ { k } / N ) ^ { 2 } } \end{array}$ equals the probability that two nodes chosen uniformly at random (with replacement) belong to the same class. This quantity provides a natural baseline:

$h _ { E } > h _ { E } ^ { * } ;$ : Edges preferentially connect intra-class nodes relative to random chance (homophilic).

$h _ { E } < h _ { E } ^ { * }$ : Edges preferentially connect inter-class nodes relative to random chance (heterophilic).

## A.7 Proof of Theorem 2.4

Both homophily indices of a finite graph are rational: $h _ { E }$ is a ratio of edge counts and $h _ { V }$ is an average of ratios of integers. The construction below attains every rational number in $( 0 , 1 )$ , so Theorem 2.4 covers every value that $h _ { E }$ or $h _ { V }$ takes in $( 0 , 1 )$ . We prove it in the following stronger form, which also prescribes the number of classes.

Theorem A.5. For every rational $h \in ( 0 , 1 )$ and every number of classes $| L | \geq 2 ,$ , there exists a graph $\mathcal { G } : = ( \nu , \mathcal { E } )$ ) with node labels $l : \mathcal { V } \overset { } { \to } L$ such that $h _ { E } = h _ { V } = h$ and $S _ { \mathcal { G } }$ has a negative eigenvalue.

Proof. We first record a sufficient condition that involves neither h nor h . Suppose that some class $j$ has no intra-class edge and at least one edge to a class i $\neq j ,$ , so that $s _ { j , j } = 0$ and $s _ { i , j } = s _ { j , i } > 0$ For the standard basis vectors $e _ { i } , e _ { j } \in \mathbb { R } ^ { | L | }$ and $\boldsymbol { x } = t \boldsymbol { e } _ { i } - \boldsymbol { e } _ { j }$ with $t > 0$

$$
\boldsymbol { x } ^ { T } \boldsymbol { S } _ { \mathcal { G } } \boldsymbol { x } = t ^ { 2 } \boldsymbol { s } _ { i , i } - 2 t \boldsymbol { s } _ { i , j } + \boldsymbol { s } _ { j , j } = t \left( t \boldsymbol { s } _ { i , i } - 2 \boldsymbol { s } _ { i , j } \right) ,\tag{32}
$$

which is negative for every sufficiently small $t > 0$ . The smallest eigenvalue of $S _ { \mathcal { G } }$ is at most the Rayleigh quotient $x ^ { T } S g x \ ' / x ^ { T } x$ , hence negative.

Write $h = p / q$ with integers $1 \leq p < q$ , and let $k = | L |$ | and $m = k - 1$ . Let G be the disjoint union of mq edges: mp of them join two nodes of class 1, and each of the other m $( q - p ) \geq k - 1$ edges joins a node of class 1 to a node of a class $j \in \{ 2 , \ldots , k \}$ , with every such $j$ used at least once. Every class is nonempty, so $S _ { \mathcal { G } }$ is well defined, and $h _ { E } = m p / ( m q ) = h$ . Every node has degree 1: each of the 2mp endpoints of an intra-class edge has its only neighbor in its own class, and each of the other $2 m ( q - p )$ nodes has its only neighbor in another class, so $h _ { V } = 2 m p / ( 2 m q ) = h$ . Class 2 has no intra-class edge and an edge to class 1, so $S _ { \mathcal { G } }$ has a negative eigenvalue by the sufficient condition.

The signature is explicit. Class 1 has $n _ { 1 } = m ( q + p )$ nodes and mp $\geq 1$ intra-class edges, so $s _ { 1 , 1 } > 0$ . Each node of a class $j \geq 2$ has its single edge to class 1, so $s _ { j , 1 } = 1 / n _ { 1 }$ and $s _ { j , j ^ { \prime } } = 0$ for all $j , j ^ { \prime } \geq 2$ . Rows $2 , \ldots , k$ of $S _ { \mathcal { G } }$ therefore all equal $( 1 / n _ { 1 } , 0 , \ldots , 0 )$ , while row 1 has the entry $1 / n _ { 1 }$ in columns $2 , \ldots , k .$ , so $S _ { \mathcal { G } }$ has rank 2. One of its two nonzero eigenvalues is negative and their sum is tr $S _ { \mathcal { G } } = s _ { 1 , 1 } > 0 .$ , so $s _ { \mathcal { G } } = ( 1 , | L | - 2 , 1 )$ . For $| L | = 2$

$$
S _ { \mathcal { G } } = \left( \begin{array} { c c } { { s _ { 1 , 1 } } } & { { \frac { 1 } { q + p } } } \\ { { \frac { 1 } { q + p } } } & { { 0 } } \end{array} \right) , \qquad s _ { 1 , 1 } > 0 , \qquad \operatorname* { d e t } S _ { \mathcal { G } } = - \frac { 1 } { ( q + p ) ^ { 2 } } < 0 .\tag{33}
$$

## A.8 Proof and interpretations of Theorem 2.5

For a smooth map $f : \mathcal { M } \to \mathcal { L }$ between differentiable manifolds M and ${ \mathcal { L } } ,$ the signature of the metric $g$ over $\mathcal { L }$ and that of the pullback $f ^ { \ast } g$ over M satisfy the following conditions:

Number of positive eigenvalues of $( f ^ { * } g ) _ { p } \leq$ Number of positive eigenvalues of $g _ { f ( p ) }$

Number of negative eigenvalues of $( f ^ { * } g ) _ { p } \leq$ Number of negative eigenvalues of $g _ { f ( p ) }$

$$
\mathrm { N u m b e r ~ o f ~ z e r o ~ e i g e n v a l u e s ~ o f } ( f ^ { * } g ) _ { p } = \mathrm { d i m _ { \mathbb { R } } } \mathrm { K e r } ( T _ { p } f ) + \mathrm { d i m _ { \mathbb { R } } } \left( \mathrm { I m } ( T _ { p } f ) \cap \mathrm { I m } ( T _ { p } f ) ^ { \perp _ { g } } \right)\tag{34}
$$

The second term in the last line is needed whenever g is indefinite. A linear map can be injective and still pull an indefinite form back to a degenerate one: for $A = ( 1 , 1 ) ^ { \top } : \mathbb { R }  \mathbb { R } ^ { 1 , 1 }$ with $g = \mathrm { d i a g } ( 1 , - 1 )$ we have ker $A = 0$ yet $A ^ { \top } g A = 0$ , so the nullity is 1. The image of A is a null line, contained in its own g-orthogonal complement, which is exactly what the second term counts. When g is positive definite that intersection is trivial and the identity reduces to the familiar dim ker. The two inequalities on the positive and negative counts hold regardless.

By Sylvester’s law of inertia (Theorem $2 . 2 ) , \phi ^ { * } I$ is positive-definite. Hence, for any pushforward $\dot { T _ { p } } f : { T _ { p } } { \cal M } \to { T _ { f ( p ) } } { \cal { L } } .$ the eigenvalues of the pullback $f ^ { * } ( \phi ^ { * } I )$ satisfy:

$$
{ \mathrm { N u m b e r ~ o f ~ n e g a t i v e ~ e i g e n v a l u e s ~ o f ~ } } ( f ^ { * } ( \phi ^ { * } I ) ) _ { p } = 0\tag{35}
$$

On the other hand, if $T _ { p } f$ is surjective then Im $\begin{array} { r } { \ i ( T _ { p } f ) = T _ { f ( p ) } \mathcal { L } . } \end{array}$ , so a negative direction of $S _ { \mathcal { G } }$ has a preimage and the eigenvalues of the pullback $f ^ { * } S _ { \mathcal { G } }$ satisfy:

$$
\mathrm { N u m b e r ~ o f ~ n e g a t i v e ~ e i g e n v a l u e s ~ o f ~ } ( f ^ { \ast } S _ { \mathcal { G } } ) _ { p } > 0\tag{36}
$$

Surjectivity is what this step needs, and it is strictly stronger than full rank when dim $\mathcal { M } <$ dim $\mathcal { L } \mathrm { : }$ an injective map into a higher-dimensional indefinite space can miss every negative direction, as the example above shows. A linear map is surjective exactly when its rank equals dim $\mathcal { L } ,$ which requires dim ${ \mathcal { M } } \geq \dim { \mathcal { L } }$ and is not implied by it. In our setting $T _ { p } f$ is the projection $\mathbf { W } \in \mathbb { R } ^ { | L | \times h ^ { \prime } }$ of Equation (4), and every configuration we report has $h ^ { \prime } > | L |$ . The matrices of rank below $| L |$ are the common zero set of the $| L | \times | L |$ minors, a proper algebraic subset of $\mathbb { R } ^ { | L | \times h ^ { \prime } }$ of measure zero, so the hypothesis holds for almost every W, and almost surely at the random initialization. Therefore, there does not exist a diffeomorphism $\phi : { \mathcal { L } } \to { \mathcal { L } }$ which induces $( \phi \circ f ) ^ { * } I = f ^ { * } S g$ . This concludes the proof of Theorem 2.5.

We can also deduce the following three observations from Theorem 2.5.

• Discretization of graph diffusion processes [5], which inherently assumes that the underlying topological space with graphs as subspaces is a Riemannian manifold, may have limited capabilities in analyzing relations among nodes of graphs whose $S _ { \mathcal { G } }$ has a negative eigenvalue.

• Continuous node attributes of graphs may not be beneficial in classifying node labels of heterophilic graph data sets, implying that utilizing only the connectivity structure between node labels could be advantageous. Empirical evidence also supports this observation, as shown in [50].

• Using adjacency matrices other than positive-definite symmetric matrices, such as graph laplacians, may boost results in classifying node labels of heterophilic graph data sets. Examples include using adjacency matrices multiplied with cosine similarities of node labels [44] or using sheaf graph laplacian operator, a type of block matrix which is not necessarily positive-definite (but positive-definite as a block matrix) [2].

## A.9 Topological structure on directed finite graphs in preparation for Definition 3.2

The parallel transport $\Gamma _ { \gamma _ { e _ { 1 } , e _ { 2 } } ( t _ { 0 } , s ) }$ for each $t = t _ { 0 }$ is taken with respect to the flat connection of $\mathbb { R } ^ { h }$ so it translates vectors, preserving both their directions and lengths. Because the metric $g = S$ is constant, this connection is compatible with g even when S is degenerate, and it is the Levi-Civita connection of $g$ when S is non-degenerate. Since Γ acts as the identity on $\mathbb { R } ^ { h }$ and S is symmetric, $e _ { \gamma _ { e _ { 1 } , e _ { 2 } } } ( \gamma _ { e _ { 1 } } , \gamma _ { e _ { 2 } } ) = e _ { \gamma _ { e _ { 2 } , e _ { 1 } } } ( \gamma _ { e _ { 2 } } , \gamma _ { e _ { 1 } } )$ , which proves the commutative property of tangent attention scores.

This subsection describes suitable topological structures to impose on directed finite graphs so that we can rigorously define the tangent attention score matrix of a graph. We treat graphs with $h -$ dimensional node features as subspaces of an h-dimensional connected pseudo-Riemannian manifold. Without loss of generality, let $\bar { \mathcal { G } } : = ( \nu , \mathcal { E } )$ be a finite directed graph with a node feature function $X : \mathcal { V } \to \mathbb { R } ^ { h }$ . We may view undirected graphs as directed graphs by regarding each undirected edge between the node v and w as two reciprocal directed edges between them. Any graph G can be viewed as a 1-dimensional topological space with nodes as discrete points and edges as unit intervals [0, 1] [26, 8]; see Chapter 0 of [13] for further discussion. This presupposition allows us to extend a function of node features $X : \mathcal { V } \to \mathbb { R } ^ { h }$ to a continuous function $X _ { \gamma } : { \mathcal { G } }  { \mathcal { M } }$ for some h-dimensional pseudo-Riemannian manifold $( \mathcal { M } , g )$ , where each directed edge e from the source node v to the target node w, denoted as $\boldsymbol { e } : = ( v , w )$ , is represented by a smooth curve of our choice $\gamma _ { e } : [ 0 , 1 ] \to { \mathcal { M } }$ satisfying the constraint that $\gamma _ { e } ( 0 ) = X ( v )$ and $\gamma _ { e } ( 1 ) = X ( w )$ . Figure 1a illustrates how one may regard a directed graph G as a subspace of $\mathcal { M }$

## A.10 Detailed description and proof of Theorem 3.3

This subsection shows, with proof, which choice of extension $X _ { \gamma }$ allows us to reconstruct a wide class of attention functions, including the dot-product attention from [38], the single-layer feedforward linear layer from GAT [39], and the dynamic graph attention from GATv2 [3]. Provided below is a detailed description of Theorem 3.3.

Theorem A.6. Let $\mathcal { G } : = ( \nu , \mathcal { E } )$ be a finite directed complete graph with self-loops, along with an h-dimensional nodefeaturefunction $X : \mathcal { V } \to \mathbb { R } ^ { h }$ . Fix a symmetric matrix $S \in \mathbb { R } ^ { h \times h }$ which endows $\mathbb { R } ^ { h }$ a pseudo-metric structure.

Suppose that for each edge $e : = ( v , w ) \in \mathcal { E } ,$ , the curves $\gamma _ { e } ^ { [ i ] } : [ 0 , 1 ]  \mathcal { M } \cong \mathbb { R } ^ { h }$ are given by:

$$
\begin{array} { r } { \gamma _ { e } ^ { [ 1 ] } ( t ) : = \left\{ \begin{array} { l l } { ( 1 - t ) \cdot X ( v ) + t \cdot X ( w ) } & { i f v \neq w } \\ { \sqrt { 3 } \left( a _ { v } t ( t - 1 ) + \frac { 1 } { \sqrt { 3 } } \right) \cdot X ( v ) } & { i f v = w } \end{array} \right. } \end{array}\tag{37}
$$

$$
\gamma _ { e } ^ { [ 2 ] } ( t ) : = { \left\{ \begin{array} { l l } { ( 1 - t ^ { 2 } ) \cdot X ( v ) + t ^ { 3 } \cdot X ( w ) } & { i f v \neq w } \\ { X ( v ) + 6 t ( t - 1 ) ( 5 t - 3 ) \cdot a _ { v , 1 } + 1 0 t ( t - 1 ) ( 2 t - 1 ) \cdot a _ { v , 2 } } & { i f v = w } \end{array} \right. }\tag{38}
$$

$$
\gamma _ { e } ^ { [ 3 ] } ( t ) : = \left\{ { \begin{array} { l l } { ( 1 - t ) \cdot X ( v ) + t \cdot X ( w ) + t ( t - 1 ) \cdot a _ { e } } & { { i f v \neq w } } \\ { X ( v ) + 3 t ( t - 1 ) \cdot a _ { v , 3 } } & { { i f v = w } } \end{array} } \right.\tag{39}
$$

for some fixed choices of set of scalars $\{ a _ { v } \} _ { v \in \mathcal { V } }$ and h-dimensional real vectors $\{ a _ { e } \} _ { e \in \mathcal { E } } , \{ a _ { v , i } \} _ { v \in \mathcal { V } } \subset \mathbb { R } ^ { h }$ . Then given a pair ofedges $\boldsymbol { e } _ { 1 } : = ( v _ { 1 } , w _ { 1 } ) , \boldsymbol { e } _ { 2 } : = ( v _ { 2 } , w _ { 2 } )$ , the $( e _ { 1 } , e _ { 2 } )  – t h$ entries ofthe tangent attention score matrices $e _ { \mathcal { G } , \gamma ^ { [ i ] } }$ are equal to:

$$
\left( e _ { \mathcal { G } , \gamma ^ { [ 1 ] } } \right) _ { e _ { 1 } , e _ { 2 } } : = \left\{ \begin{array} { l l } { ( X ( w _ { 1 } ) - X ( v _ { 1 } ) ) ^ { T } \cdot S \cdot ( X ( w _ { 2 } ) - X ( v _ { 2 } ) ) } & { \mathrm { ~ } i f v _ { 1 } \neq w _ { 1 } , \ v _ { 2 } \neq w _ { 2 } } \\ { 0 } & { \mathrm { ~ } i f v _ { 1 } \neq w _ { 1 } , \ v _ { 2 } = w _ { 2 } } \\ { 0 } & { \mathrm { ~ } i f v _ { 1 } = w _ { 1 } , \ v _ { 2 } \neq w _ { 2 } } \\ { a _ { v _ { 1 } } a _ { v _ { 2 } } X ( v _ { 1 } ) ^ { T } \cdot S \cdot X ( v _ { 2 } ) } & { \mathrm { ~ } i f v _ { 1 } = w _ { 1 } , \ v _ { 2 } = w _ { 2 } } \end{array} \right.\tag{40}
$$

$$
\left( e _ { \mathcal G , \gamma ^ { [ 2 ] } } \right) _ { e _ { 1 } , e _ { 2 } } : = \left\{ \begin{array} { l l } { \left( \frac { 4 } { 3 } X ( v _ { 1 } ) - \frac { 3 } { 2 } X ( w _ { 1 } ) \right) ^ { T } \cdot S \cdot X ( v _ { 2 } ) } \\ { \qquad - \left( \frac { 3 } { 2 } X ( v _ { 1 } ) - \frac { 9 } { 5 } X ( w _ { 1 } ) \right) ^ { T } \cdot S \cdot X ( w _ { 2 } ) } \\ { X ( v _ { 1 } ) ^ { T } \cdot S \cdot a _ { v _ { 2 } , 1 } + X ( w _ { 1 } ) ^ { T } \cdot S \cdot a _ { v _ { 2 } , 2 } } \\ { a _ { v _ { 1 } , 1 } ^ { T } \cdot S \cdot X ( v _ { 2 } ) + a _ { v _ { 1 } , 2 } ^ { T } \cdot S \cdot X ( w _ { 2 } ) } \\ { 4 8 a _ { v _ { 1 } , 1 } ^ { T } \cdot S \cdot a _ { v _ { 2 } , 1 } + 3 0 a _ { v _ { 1 } , 1 } ^ { T } \cdot S \cdot a _ { v _ { 2 } , 2 } } \\ { \qquad + 3 0 a _ { v _ { 1 } , 2 } ^ { T } \cdot S \cdot a _ { v _ { 2 } , 1 } + 2 0 a _ { v _ { 1 } , 2 } ^ { T } \cdot S \cdot a _ { v _ { 2 } , 2 } } \end{array} \right.
$$

$$
\begin{array} { l } { i f v _ { 1 } \neq w _ { 1 } , v _ { 2 } \neq w _ { 2 } } \\ { i f v _ { 1 } \neq w _ { 1 } , v _ { 2 } = w _ { 2 } } \\ { i f v _ { 1 } = w _ { 1 } , v _ { 2 } \neq w _ { 2 } } \end{array}\tag{41}
$$

$$
\left( e _ { \boldsymbol { \mathcal { G } } , \gamma ^ { [ 3 ] } } \right) _ { e _ { 1 } , e _ { 2 } } : = \left\{ \begin{array} { l l } { ( X ( \boldsymbol { w _ { 1 } } ) - X ( \boldsymbol { v _ { 1 } } ) ) ^ { T } \cdot S \cdot ( X ( \boldsymbol { w _ { 2 } } ) - X ( \boldsymbol { v _ { 2 } } ) ) } \\ { \qquad + \frac { 1 } { 3 } a _ { e _ { 1 } } ^ { T } \cdot S \cdot a _ { e _ { 2 } } } \\ { a _ { e _ { 1 } } ^ { T } \cdot S \cdot a _ { \boldsymbol { v _ { 2 } } , 3 } } \\ { a _ { \boldsymbol { v _ { 1 } } , 3 } ^ { T } \cdot S \cdot a _ { e _ { 2 } } } \\ { 3 a _ { \boldsymbol { v _ { 1 } } , 3 } ^ { T } \cdot S \cdot a _ { \boldsymbol { v _ { 2 } } , 3 } } \end{array} \right.
$$

$$
i f v _ { 1 } = w _ { 1 } , v _ { 2 } = w _ { 2 }
$$

$$
\begin{array} { l } { i f v _ { 1 } \neq w _ { 1 } , v _ { 2 } \neq w _ { 2 } } \\ { i f v _ { 1 } \neq w _ { 1 } , v _ { 2 } = w _ { 2 } } \\ { i f v _ { 1 } = w _ { 1 } , v _ { 2 } \neq w _ { 2 } } \\ { i f v _ { 1 } = w _ { 1 } , v _ { 2 } = w _ { 2 } } \end{array}\tag{42}
$$

Proof. We prove the statement of the theorem for the curve $\gamma _ { e } ^ { [ 1 ] } : [ 0 , 1 ]  \mathcal { M }$ . The remaining cases follow analogously. Let $e _ { 1 } = ( v _ { 1 } , w _ { 1 } ) , e _ { 2 } = ( v _ { 2 } , w _ { 2 } )$ be a pair of directed edges of a finite directed complete graph $\mathcal { G } .$

Suppose $v _ { 1 } \neq w _ { 1 }$ and $v _ { 2 } \ne w _ { 2 }$ . Then we obtain that

$$
\begin{array} { r } { \gamma _ { e _ { 1 } } ^ { ' [ 1 ] } ( t ) = X ( w _ { 1 } ) - X ( v _ { 1 } ) } \\ { \gamma _ { e _ { 2 } } ^ { ' [ 1 ] } ( t ) = X ( w _ { 2 } ) - X ( v _ { 2 } ) } \end{array}\tag{43}
$$

Because both derivatives are constant functions with respect to the variable t,

$$
e _ { \gamma _ { e _ { 1 } , e _ { 2 } } } ( \gamma _ { e _ { 1 } } ^ { [ 1 ] } , \gamma _ { e _ { 2 } } ^ { [ 1 ] } ) = \left( X ( w _ { 1 } ) - X ( v _ { 1 } ) \right) ^ { T } \cdot S \cdot \left( X ( w _ { 2 } ) - X ( v _ { 2 } ) \right) .\tag{44}
$$

If $v _ { 1 } = w _ { 1 }$ whereas $v _ { 2 } \ne w _ { 2 }$ , then

$$
\begin{array} { r l } & { \gamma _ { e _ { 1 } } ^ { ' [ 1 ] } ( t ) = \sqrt { 3 } a _ { v _ { 1 } } ( 2 t - 1 ) \cdot X ( v _ { 1 } ) } \\ & { \gamma _ { e _ { 2 } } ^ { ' [ 1 ] } ( t ) = X ( w _ { 2 } ) - X ( v _ { 2 } ) } \end{array}\tag{45}
$$

which leads to

$$
e _ { \gamma _ { e _ { 1 } , e _ { 2 } } } ( \gamma _ { e _ { 1 } } ^ { [ 1 ] } , \gamma _ { e _ { 2 } } ^ { [ 1 ] } ) = \left( \int _ { 0 } ^ { 1 } \sqrt { 3 } a _ { v _ { 1 } } ( 2 t - 1 ) d t \right) \cdot \left( X ( v _ { 1 } ) ^ { T } \cdot S \cdot ( X ( w _ { 2 } ) - X ( v _ { 2 } ) ) \right) = 0 .\tag{46}
$$

Lastly, if $v _ { 1 } = w _ { 1 }$ and $v _ { 2 } = w _ { 2 }$ , then

$$
\begin{array} { r } { \gamma _ { e _ { 1 } } ^ { ' [ 1 ] } ( t ) = \sqrt { 3 } a _ { v _ { 1 } } ( 2 t - 1 ) \cdot X ( v _ { 1 } ) } \\ { \gamma _ { e _ { 2 } } ^ { ' [ 1 ] } ( t ) = \sqrt { 3 } a _ { v _ { 2 } } ( 2 t - 1 ) \cdot X ( v _ { 2 } ) } \end{array}\tag{47}
$$

which gives

$$
\begin{array} { r l } & { e _ { \gamma _ { e _ { 1 } , e _ { 2 } } } ( \gamma _ { e _ { 1 } } ^ { [ 1 ] } , \gamma _ { e _ { 2 } } ^ { [ 1 ] } ) = \displaystyle \left( \int _ { 0 } ^ { 1 } 3 a _ { v _ { 1 } } a _ { v _ { 2 } } ( 2 t - 1 ) ^ { 2 } d t \right) \cdot \big ( X ( v _ { 1 } ) ^ { T } \cdot S \cdot X ( v _ { 2 } ) \big ) } \\ & { \qquad = a _ { v _ { 1 } } a _ { v _ { 2 } } X ( v _ { 1 } ) ^ { T } \cdot S \cdot X ( v _ { 2 } ) . } \end{array}\tag{48}
$$

Theorem 3.3 demonstrates that previously researched attention scores correspond to computing the tangent attention score between two directed edges where at least one of them is a self-loop at each node of ${ \mathcal { G } } .$ . The dot product attention score from the encoder layer of attention-transformers [38] with identical query and key weight matrices is equal to the tangent attention score between two self-loops represented by $\gamma _ { e } ^ { [ 1 ] }$ , with $S = W ^ { T } W$ a symmetric matrix obtained from a product of invertible square weight matrix $\dot { W } \in \mathbb R ^ { h \times h }$ . With the same symmetric matrix $S ,$ the attention score of the single-layer feedforward linear layer in [39] can be obtained from the tangent attention score between a self-loop and a directed edge represented by $\gamma _ { e } ^ { [ 2 ] }$ . Lastly, by setting S to be the identity matrix I, and by choosing the fixed vector $a _ { e }$ as $a _ { e } = \mathrm { L e a k y }$ Relu $( \Breve { W } [ X ( v ) \ ] | X ( w ) ] )$ ) for each edge $e = ( v , w ) \in \dot { \mathcal { E } }$ and every ${ { a } _ { v , 3 } }$ as the shared attention vector $\mathbf { a } \in \mathbb { R } ^ { h }$ of GATv2, where $[ X ( v ) | | \bar { X } ( w ) ] \stackrel { . } { \in } \mathbb { R } ^ { 2 h }$ is a concatenation of two node feature vectors, and $W \in \mathbb { R } ^ { h \times 2 h }$ is a weight matrix, the tangent attention score between a self-loop and a directed edge represented by $\gamma _ { e } ^ { [ 3 ] }$ is equal to the dynamic graph attention score constructed from [3].

## B Geometric Framework of PRAT

In Figure $7 , \mathcal { H } ^ { ( l ) } \cong \mathbb { R } ^ { h ^ { \prime } }$ denotes the hidden manifold at layer l, and $\mathcal { L } \cong \mathbb { R } ^ { | L | }$ is the common latent manifold corresponding to the node labels. The encoder $\psi _ { \mathrm { e n c } } : \mathcal { M } \to \mathcal { H } ^ { ( 0 ) }$ calibrates node features to the initial hidden feature.

For each layer l, we posit the existence of a linear map $\varphi ^ { ( l ) } : \mathcal { H } ^ { ( l ) } \to \mathcal { L }$ , which projects the hidden representation to the label space. While the final layer’s map $\varphi ^ { ( l _ { \mathrm { m a x } } ) }$ acts as the explicit readout function for classification, the intermediate maps $\varphi ^ { ( l ) }$ (for $l < l _ { \mathrm { m a x } } )$ provide a critical geometric link:

![](images/b940de9a7328647e6711544534512855a33a0742490141f2ddf05627db125afe.jpg)  
Figure 7: Geometric interpretation of PRAT. Each hidden manifold $\mathcal { H } ^ { ( l ) }$ is mapped to the common latent manifold $\mathcal { L }$ via $\varphi ^ { ( l ) }$

they allow us to transport the intrinsic geometric structure of the latent manifold back to the hidden manifolds.

Let $g _ { \mathcal { L } }$ be the pseudo-metric tensor defined on ${ \mathcal { L } } ,$ represented by a symmetric matrix $S ^ { | L | }$ . To ensure that the hidden representations at layer l respect the global structural prior of the latent space, we set the metric on $\mathbf { \dot { \mathcal { H } } } ^ { ( l ) }$ to the pullback metric induced by $\varphi ^ { ( l ) }$ . Hence, the pseudo-metric on $\mathcal { H } ^ { ( l ) }$ is determined by pulling back $g _ { \mathcal { L } }$ via $\varphi ^ { ( l ) }$

$$
g _ { \mathcal { H } ^ { ( l ) } } = ( \varphi ^ { ( l ) } ) ^ { * } g _ { \mathcal { L } } .\tag{49}
$$

Considering the multi-head attention mechanism where k denotes the head index, the projection $\varphi ^ { ( l ) }$ is parameterized by a weight matrix $\mathbf { W } _ { \varphi } ^ { [ l , k ] }$ . The induced metric tensor $\mathbf { D } ^ { [ l , k ] }$ for the k-th head at layer l is then derived as:

$$
\mathbf { D } ^ { [ l , k ] } = ( \mathbf { W } _ { \varphi } ^ { [ l , k ] } ) ^ { \top } S ^ { | L | } \mathbf { W } _ { \varphi } ^ { [ l , k ] } ,\tag{50}
$$

which is identical to the expression for $\mathbf { D } ^ { [ l , k ] }$ in Equation (4).

## C Benchmark Datasets

We use fifteen real-world benchmarks in two groups: (i) eight standard benchmarks spanning homophilic and heterophilic regimes, and (ii) seven Platonov benchmarks [28] consisting of filtered Wikipedia networks and five large-scale heterophilic datasets.

Standard benchmarks.

Citation Networks (Cora, Citeseer, PubMed) Standard benchmark datasets representing citation relationships between scientific publications [32, 45]. Nodes correspond to documents, and edges denote citation links. For each node, the features are typically sparse bag-of-words vectors derived from the document text, and the label indicates the subject category of the paper.

Wikipedia Networks (Chameleon, Squirrel) These are page-page networks derived from Wikipedia [30]. Nodes represent Wikipedia articles, and edges indicate mutual hyperlinks between them. Node features are informative nouns extracted from the corresponding article, and the node label represents the average monthly traffic of the webpage.

Actor Co-occurrence Network (Actor) The Actor dataset (often referred to as Film) [35] is a subgraph of a film-director-actor-writer network. Nodes represent actors, and edges signify co-occurrence on the same Wikipedia page. Node features are derived from keywords on the actor’s Wikipedia page, and node labels are determined by the words found on the webpage.

Webpage Networks (Texas, Cornell) Subsets of the WebKB dataset<sup>6</sup>. Nodes represent university webpages, and edges represent hyperlinks between them. For each node, the features are bag-of-words representations of the webpage content, and the node label denotes the category of the webpage.

![](images/f1658c0510503bb4ac0ac3dd0bc79719f219b025d0bc754e76ec4e1146632c4b.jpg)  
Figure 8: Eigenvalue distributions of the graph signature matrix of the full labeled graph (Definition 2.1) for all fifteen benchmark datasets (eight standard + seven Platonov), which give the signatures of Table 1 and Table 2. Eigenvalues with $| \lambda | < 1 . 0 \times 1 0 ^ { - 4 }$ are rounded to zero for reporting.

## Platonov benchmarks [28].

Filtered Wikipedia networks (Chameleon-filtered, Squirrel-filtered) Platonov et al. [28] identified that 57–61% of nodes in the original Chameleon and Squirrel datasets are exact duplicates with identical neighborhoods and labels, causing severe train–test leakage. The filtered versions remove these duplicate nodes, yielding 890 and 2,223 nodes respectively. Node features and class definitions follow the original Rozemberczki et al. [30] construction. These filtered benchmarks are widely adopted in subsequent heterophily research as a more reliable proxy for the underlying task.

roman-empire Each node corresponds to a non-unique word in the English Wikipedia article “Roman Empire,” and edges connect words that are syntactically dependent or sequential in the article. Node features are GloVe-300 embeddings of the word, and the 18-class label is the syntactic role assigned by spaCy. With $h _ { E } = 0 . 0 5$ , this is the most extreme heterophilic dataset in the suite.

amazon-ratings A product co-purchase network from Amazon, where nodes are products (books, music CDs, DVDs, VHS tapes) and edges connect frequently co-purchased products. Node features are the mean of fastText embeddings of the product description, and the 5-class label is the average rating, binned into five ranges.

minesweeper A synthetic 100 × 100 grid graph from the Minesweeper game: nodes are cells, edges connect each cell to its eight surrounding neighbors, and 20% of cells are randomly designated as mines (the binary class). Node features are 7-dimensional one-hot encodings of the number of neighboring mines for non-mine cells (with mine cells masked); the resulting graph has very low-dimensional node features and a regular topological structure.

tolokers A worker-collaboration network from the Toloka crowdsourcing platform, where nodes are workers and edges connect workers who co-completed at least one task. Node features include task performance and profile statistics, and the binary label indicates whether a worker was banned. With over 519K edges over 11.8K nodes, tolokers is the densest graph in the suite.

questions A user-interaction network on the Yandex.Q question-answering site, where nodes are users and edges connect users who answered each other’s questions during a one-year window. Node features are FastText embeddings of the user description, and the binary label indicates account activity at the end of the window. With 48,921 nodes, this is the largest graph in the suite.

Analysis on Graph Signature Index. Figure 8 illustrates the eigenvalue distributions of the graph signature matrices for all benchmark datasets, revealing distinct structural characteristics. Among the standard benchmarks, the Actor dataset stands out for its many zero eigenvalues. Its signature $s _ { \mathcal { G } } = ( 1 , 4 , 0 )$ indicates connectivity of numerical rank one at the $1 \dot { 0 } ^ { - 4 }$ threshold, where all 5 classes share nearly identical edge density patterns, with 4 out of 5 label-space dimensions carrying negligible discriminative signal. PRAT leverages this by learning Λ values near zero for the corresponding dimensions during training, effectively projecting out uninformative directions and concentrating the metric on the single informative one. This “feature selection via metric learning” explains why PRAT achieves the best accuracy on Actor despite the absence of negative eigenvalues: the pseudo-Riemannian framework naturally accommodates zero eigenvalues, providing a mechanism to suppress noise that other methods aggregate indiscriminately.

## D Synthetic Data Generation

Synthetic datasets are generated based on Model 1 proposed in Appendix B of [15]. According to this model, each node feature is initially sampled from a Bernoulli distribution, yielding binary values of 0 or 1. However, we observed that this formulation results in class labels being trivially predictable by GNNs, leading to near-perfect accuracy even in highly heterophilic regimes.

To address this limitation and make classification harder, we modify the original model’s features. We add class-independent Gaussian noise of standard deviation 0.4 to each node feature, thereby transforming the discrete binary features into continuous values and increasing the difficulty of class discrimination. The added noise prevents perfect predictions while preserving the synthetic dataset’s underlying structure.

For graph construction, given a target number of edges, a desired homophily level $h _ { E } \in [ 0 , 1 ]$ , and a target graph signature index, we first compute the number of homophilic edges as $| \mathcal { E } _ { h o m o } | = h _ { E } \cdot | \mathcal { E } |$ and heterophilic edges as $| { \mathcal { E } } _ { h e t e r o } | = | { \dot { \mathcal { E } } } | - | { \mathcal { E } } _ { h o m o } |$ . To ensure the resulting graph matches the specified signature index, these edge counts are strategically distributed across valid intra-class and inter-class pairs by a random search over at most 2,000 symmetric candidates for a class-connectivity matrix that satisfies both the homophily and signature constraints. Homophilic edges are then generated by randomly sampling node pairs within the same class, while heterophilic edges are created by connecting nodes from different classes according to the determined connectivity matrix. Certain combinations of homophily levels and signature indices are mathematically incompatible; in such cases, synthetic data generation is restricted. This procedure precisely controls both the graph’s homophily ratio and signature index, enabling systematic evaluation of model performance across diverse structural scenarios. Every graph is generated with seed 42, the five random splits use seeds 42 to 46, and every model is trained with seed 42.

Eigenvalue Evolution. Figure 9 illustrates the evolution of eigenvalues in the graph signature matrix across synthetic datasets generated with varying homophily levels. At the heterophilic extreme $( h _ { E } = 0 )$ , the graph signature index is (1, 0, 4), indicating a predominance of negative eigenvalues. $\mathbf { A } \mathbf { s } h _ { E }$ increases, these eigenvalues exhibit a continuous upward trend, eventually transitioning to a strictly positive regime. This trajectory closely mirrors the sign evolution of the learned metric observed in Figure 3. These findings provide empirical evidence that PRAT successfully leverages the underlying topological information regarding graph homophily.

## E Computational Complexity Analysis

We analyze the computational complexity of PRAT compared to other attention-based models. Let N denote nodes, E edges, H hidden dimension, C classes, K attention heads, and L layers. $H / K$ is the per-head dimension.

## E.1 Time Complexity

Theorem E.1 (PRAT Layer Complexity). The time complexity of a single PRAT layer is $\mathcal { O } ( N H ( H +$ $K C ) + E ( H + K C ) )$ .

Proof. PRAT has operations:

![](images/2f4a9245cd47498708dd1c66c6cdcfec3168e1ee3cb647108160b8ad58ee254c.jpg)  
Figure 9: Evolution of eigenvalues of graph signature matrix across varying homophily levels. As homophily increases, eigenvalues change from mixed signs to positive values, indicating a shift from pseudo-Riemannian to Riemannian geometry.

1. Latent space projection (XW, $\mathbf { W } \in \mathbb { R } ^ { H \times K C } ) \colon { \mathcal { O } } ( N H K C )$

2. Value projection $( \mathbf { X } \mathbf { W } _ { V } , \mathbf { W } _ { V } \in \mathbb { R } ^ { H \times H } ) \colon \mathcal { O } ( N H ^ { 2 } )$

3. Attention scores (difference, square, weighted sum per edge): $\mathcal { O } ( E K C )$

4. Softmax and aggregation: $\mathcal { O } ( E K + E H )$

Theorem E.2 (GAT/GATv2 Layer Complexity). The time complexity of GAT and GATv2 layers is $\mathcal { O } ( N H ^ { 2 } + E H )$ .

Proof. GAT computes $e _ { i j } = \mathrm { L e a k y R e L U } ( \mathbf { a } ^ { \top } [ \mathbf { W } \mathbf { h } _ { i } \| \mathbf { W } \mathbf { h } _ { j } ] )$

1. Feature transform $( \mathbf { X } \mathbf { W } ) \colon { \mathcal { O } } ( N H ^ { 2 } )$

2. Attention computation (concatenation + linear per edge): $\mathcal { O } ( E K ( H / K ) ) = \mathcal { O } ( E H )$

3. Softmax and aggregation: $\mathcal { O } ( E K + E H )$

GATv2 applies LeakyReLU before the attention vector but has identical complexity.

Theorem E.3 (SDP Layer Complexity). The time complexity of Scaled Dot-Product attention is $\mathcal { O } ( 3 N H ^ { 2 } + E H )$

Proof. SDP computes $e _ { i j } = ( { \bf Q } _ { i } \cdot { \bf K } _ { j } ) / \sqrt { d _ { k } }$ with three projections:

1. Query, key, and value projections: $\mathcal { O } ( 3 N H ^ { 2 } )$

2. Dot product per edge: $\mathcal { O } ( E K ( H / K ) ) = \mathcal { O } ( E H )$

3. Softmax and aggregation: $\mathcal { O } ( E K + E H )$

Table 6 summarizes the time complexity of PRAT and attention-based models. PRAT shares asymptotic complexity $\mathcal { O } ( N H ^ { 2 } + E H )$ with GAT and GATv2 since KC is typically small relative to H $( \mathrm { e . g . , } K \mathrm { = } 4 , C \mathrm { = } 5 , \dot { H } \mathrm { = } 6 4$ gives $\dot { K } C = 2 0 < 6 4 )$

Table 6: Time complexity comparison per layer.
<table><tr><td>Model Complexity</td></tr><tr><td>PRAT  $\mathcal { O } ( N H ( H + K C ) + E ( H + K C ) )$  GAT  $\mathcal { O } ( N H ^ { 2 } + E H )$  GATv2  $\mathcal { O } ( N H ^ { 2 } + E H )$ </td></tr></table>

## E.2 Parameter Count

For a model with encoder, L attention layers, and decoder:

Theorem E.4 (Parameter Counts). The total parameters for each model are:

$$
P _ { P R A T } = ( F + 1 ) H + L ( H ^ { 2 } + H K C + H + K C + K ) + ( H + 1 ) C\tag{51}
$$

$$
P _ { G A T } = ( F + 1 ) H + L ( H ^ { 2 } + 3 H ) + ( H + 1 ) C\tag{52}
$$

$$
P _ { G A T \nu 2 } = ( F + 1 ) H + L ( H ^ { 2 } + 3 H ) + ( H + 1 ) C\tag{53}
$$

$$
P _ { S D P } = ( F + 1 ) H + L ( 3 H ^ { 2 } + 3 H ) + ( H + 1 ) C\tag{54}
$$

Proof. Per-layer parameters:

• PRAT: $\mathbf { W } \in \mathbb { R } ^ { H \times K C }$ without bias, $\mathbf { W } _ { V } \in \mathbb { R } ^ { H \times H }$ with bias, the C diagonal entries of the learnable metric held for each of the K heads, $\Lambda \in \mathbb { R } ^ { K \times C }$ , self-loop scaling parameters $\{ a \} \in \mathbb { R } ^ { K }$ . Total: $H K C + H ^ { 2 } + H + K C + K$

• GAT/GATv2: $\mathbf { W } \in \mathbb { R } ^ { H \times H }$ and an output bias, with $\mathbf { a } \in \mathbb { R } ^ { 2 H }$ for GAT; GATv2 shares W between source and target and adds a bias to it, with $\mathbf { a } \in \mathbb { R } ^ { H }$ . Total: $H ^ { 2 } + 3 H$

• SDP: $\mathbf { W } _ { Q } , \mathbf { W } _ { K } , \mathbf { W } _ { V } \in \mathbb { R } ^ { H \times H }$ each with bias. Total: $3 H ^ { 2 } + 3 H$

Single layer MLP encoder and decoder contribute $( F { + } 1 ) H$ and $( H { + } 1 ) C$ , respectively, for all models.

Table 7: Parameter count per layer (H=64, C=5, K=4).
<table><tr><td>Model</td><td>Formula</td><td>Count</td></tr><tr><td>PRAT</td><td> $H ^ { 2 } + H K C + H + K C + K$ </td><td>5,464</td></tr><tr><td>GAT</td><td> $H ^ { 2 } + 3 H$ </td><td>4,288</td></tr><tr><td>GATv2</td><td> $H ^ { 2 } + 3 H$ </td><td>4,288</td></tr><tr><td>SDP</td><td> $3 H ^ { 2 } + 3 H$ </td><td>12,480</td></tr></table>

Table 7 shows that PRAT has modestly more parameters than GAT and GATv2 because of its learnable metric components. Since $K C < H$ in practice, this overhead remains negligible. Moreover, PRAT is substantially more parameter-efficient than SDP attention, which requires separate query, key, and value projections.

## E.3 Measured cost

The asymptotics above say that PRAT and GAT are in the same class. Table 8 shows what that means in practice, measured on the two largest graphs with the same backbone, width, depth and number of steps, so that the only difference is the attention module.

PRAT adds 80 parameters over GAT on questions, out of 2.83 million, and 32 on tolokers. The step time is 0.97 times $\mathrm { G A T } \ '$ on questions and 1.17 on tolokers, and peak memory is 1.03 and 1.20 times GAT’s. On questions, PRAT and GAT are indistinguishable at that scale: single-step timing has

Table 8: Training cost per step and peak memory, measured on one A100 80GB with identical backbone, width and depth.
<table><tr><td>Dataset</td><td>Model</td><td>ms / step</td><td>peak memory (MiB)</td><td>parameters</td></tr><tr><td rowspan="3">tolokers (11,758 nodes)</td><td>PRAT</td><td>26.72</td><td>1084.4</td><td>336,689</td></tr><tr><td>GAT</td><td>22.80</td><td>903.0</td><td>336,657</td></tr><tr><td>GT</td><td>32.85</td><td>466.1</td><td>531,713</td></tr><tr><td rowspan="3">questions (48,921 nodes)</td><td>PRAT</td><td>88.73</td><td>3791.2</td><td>2,830,201</td></tr><tr><td>GAT</td><td>91.23</td><td>3696.9</td><td>2,830,121</td></tr><tr><td>GT</td><td>175.17</td><td>5285.5</td><td>6,734,081</td></tr></table>

run-to-run spread of the same order, so their timings do not establish that PRAT is faster. GT, whose attention is also restricted to graph neighbors, costs 1.92 times GAT on questions and needs 2.4 times the parameters.

## F Experimental Details

Signature estimation. $S _ { \mathcal { G } , \mathrm { t r a i n } }$ is Definition 2.1 applied to the subgraph induced by the training nodes, with self-loops removed: an edge counts only when both endpoints are training nodes, the class sizes in the denominator are training-node counts, and a class with no training node contributes a zero row and column. $\Lambda _ { \mathrm { i n i t } }$ is computed from the unrounded eigenvalues, with std(·) the population standard deviation over the |L| eigenvalues. When all eigenvalues are equal, so that this deviation is zero, the division is skipped and $\Lambda _ { \mathrm { i n i t } } = \Lambda _ { \mathcal { G } , \mathrm { t r a i n } } / \sqrt { | L | }$ ; the identity initialization of Table 15 takes this path. The $1 0 ^ { - 4 }$ rounding of Figure 8 is applied to the unnormalized eigenvalues and only for reported inertias.

## F.1 Hyperparameters

Standard Benchmark Experiments. For experiments on the eight standard benchmarks, the hyperparameters of every model (PRAT and baselines) were selected via a hyperparameter sweep over the search ranges and distributions listed in Table 9, with the best configuration chosen by validation accuracy.

Table 9: Hyperparameter search ranges used for the standard benchmark experiments.
<table><tr><td>Hyperparameter</td><td>Search Range</td><td>Distribution</td></tr><tr><td>Model Architecture</td><td></td><td></td></tr><tr><td>Hidden channels Number of layers</td><td>64</td><td>Fixed</td></tr><tr><td>Attention heads</td><td>{1, 2, 3, 4}</td><td>Discrete</td></tr><tr><td></td><td>{1, 2, 4, 8, 16}</td><td>Discrete</td></tr><tr><td>Input dropout</td><td>[0.0, 0.9]</td><td>Uniform</td></tr><tr><td>Output dropout</td><td>[0.0, 0.9]</td><td>Uniform</td></tr><tr><td>Optimization</td><td></td><td></td></tr><tr><td>Learning rate</td><td> $[ 1 0 ^ { - 4 } , 1 0 ^ { - 1 } ]$ </td><td>Log-uniform</td></tr><tr><td>Weight decay</td><td> $[ 1 0 ^ { - 5 } , 1 0 ^ { - 2 } ]$ </td><td>Log-uniform</td></tr><tr><td>Epochs</td><td>1,000</td><td>Fixed</td></tr><tr><td>Patience</td><td>100</td><td>Fixed</td></tr></table>

Platonov Benchmark Experiments. For experiments on the seven Platonov et al. [28] benchmarks (filtered Chameleon/Squirrel and the five large-scale heterophilic datasets), PRAT’s hyperparameters were selected via a sweep starting from the official PRAT-compatible defaults of Platonov et al. [28] (hidden channels 512, layers $\in \{ \bar { 1 } , 2 , 3 , 4 , 5 \}$ , lr $3 \times 1 0 ^ { - 5 }$ , dropout 0.2, 1,000 training steps), and we report 10-split mean ± std on the official predefined splits. Table 10 lists the selected configurations.

Synthetic Data Experiments. When training models on synthetic data, we employed a fixed set of hyperparameters shown in Table 11 across all synthetic datasets and models. This controlled setup ensures that any observed performance fluctuations are solely attributable to changes in the homophily level or graph signature index, thus facilitating a clear analysis of the behavioral trends and stability of each method.

Table 10: Selected PRAT configurations on the Platonov benchmarks, shared by Table 4 and the matched run of Section I.1. Every configuration uses LayerNorm, AdamW and training seed 42; warmup is a fraction of the training steps.
<table><tr><td>Dataset</td><td>Hidden</td><td> $\mathrm { L a y e r s }$ </td><td>Heads</td><td>FFN mult.</td><td>Dropout</td><td>Learning rate</td><td>Weight decay</td><td> ${ \tt S t e p s }$ </td><td>Warmup</td></tr><tr><td>Cham.-filt.</td><td>128</td><td>1</td><td>8</td><td>1</td><td>0.217</td><td> $1 . 3 7 \times 1 0 ^ { - 4 }$ </td><td> $3 . 2 8 \times 1 0 ^ { - 2 }$ </td><td>2,000</td><td>0.031</td></tr><tr><td> $\operatorname { S q . - f i l t . }$ </td><td>512</td><td>1</td><td>8</td><td>0.5</td><td>0.627</td><td> $1 . 5 9 \times 1 0 ^ { - 4 }$ </td><td> $2 . 3 4 \times 1 0 ^ { - }$  -2</td><td>500</td><td>0.042</td></tr><tr><td>roman-empire</td><td>512</td><td>5</td><td>8</td><td>1</td><td>0.2</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td>0</td><td>1,000</td><td>0</td></tr><tr><td>amazon-ratings</td><td>512</td><td>2</td><td>8</td><td>1</td><td>0.2</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 3 }$ </td><td>2,000</td><td>0.1</td></tr><tr><td>minesweeper</td><td>128</td><td>4</td><td>8</td><td>0.5</td><td>0.1</td><td> $7 \times 1 0 ^ { - 5 }$ </td><td> $1 0 ^ { - 3 }$ </td><td>2,000</td><td>0</td></tr><tr><td>tolokers</td><td>128</td><td>4</td><td>4</td><td>2</td><td>0.1</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td>0</td><td>2,000</td><td>0</td></tr><tr><td>questions</td><td>512</td><td>5</td><td>8</td><td>0.5</td><td>0.2</td><td> $1 0 ^ { - 4 }$ </td><td> $5 \times 1 0 ^ { - 4 }$ </td><td>1,000</td><td>0</td></tr></table>

Table 11: Configuration for synthetic data experiments.
<table><tr><td colspan="2">Dataset Configuration</td><td colspan="2">Hyperparameters</td></tr><tr><td>Nodes</td><td>1,000</td><td>Hidden channels</td><td>64</td></tr><tr><td>Edges</td><td>2,000</td><td>Number of layers</td><td>2</td></tr><tr><td>Features Classes</td><td>1,000 5</td><td>Attention heads Input dropout</td><td>8 0.6</td></tr><tr><td></td><td></td><td>Output dropout Learning rate Weight decay Epochs</td><td>0.6 0.002 0.0005 1,000</td></tr></table>

## F.2 Computing Resources

All experiments we conducted for this paper were run on a single NVIDIA A100 80GB GPU. The baseline rows of Table 4 are quoted from Platonov et al. [28], and the Geom-GCN results from Yan et al. [44]. No multi-GPU or multi-node training was used. One full hyperparameter sweep per dataset takes between 0.5 and 8 GPU-hours depending on dataset size, after which the selected configuration is rerun across multiple splits for the reported test mean / standard deviation. Final 10-split evaluations on the Platonov benchmarks (Table 4) require an additional ∼ 0.5–2 GPU-hours per dataset. The total compute budget across the project, including preliminary and discarded configurations not reported in the paper, is conservatively estimated at ∼200 A100-hours.

## F.3 Calibration of the Self-Loop Scaling Parameter

From the tangent attention framework in Appendix A.10, self-loop scores involve a scaling coefficient applied to the quadratic form $h _ { v } ^ { \top } \mathbf Ḋ h Ḍ _ { v }$ . We derive a principled initialization for the self-loop scaling parameter a by matching expected attention scores.

Pseudo-Label Representation. Assume hidden features encode noisy pseudo-labels in the latent space $\mathcal { L } \mathrm { : ~ }$

$$
\mathbf { W } h _ { v } = e _ { \ell ( v ) } + \epsilon _ { v } , \quad \epsilon _ { v } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I _ { | L | } ) ,\tag{55}
$$

where W is the weight matrix from Equation (4). $e _ { i } \in \mathbb { R } ^ { | L | }$ is the one-hot encoding of label i and $\sigma ^ { 2 }$ quantifies the noise level in the representations.

Matching Condition. Using the graph signature matrix $S _ { \mathcal { G } }$ as the metric tensor and the proportional allocation model introduced in Appendix $\mathrm { A } . 6 ,$ , matching the expected self-loop and edge scores yields, for $h _ { E } > 0 ;$

$$
a ( 1 + \sigma ^ { 2 } | L | ) = \frac { 2 ( 1 - h _ { E } ) ( h _ { E } - h _ { E } ^ { * } ) } { h _ { E } ( 1 - h _ { E } ^ { * } ) } + 2 \sigma ^ { 2 } | L | ,\tag{56}
$$

where $\begin{array} { r } { h _ { E } ^ { * } = \sum _ { k = 1 } ^ { | L | } ( n _ { k } / N ) ^ { 2 } } \end{array}$ is the critical homophily threshold from Theorem 2.3, with $n _ { k }$ denoting the number of nodes in class k. Solving for a:

$$
a = \frac { \displaystyle \frac { 2 ( 1 - h _ { E } ) ( h _ { E } - h _ { E } ^ { * } ) } { h _ { E } ( 1 - h _ { E } ^ { * } ) } + 2 \sigma ^ { 2 } | L | } { \displaystyle 1 + \sigma ^ { 2 } | L | } .\tag{57}
$$

Noise-Dominated Initialization. At the onset of training, before the network has learned labeldiscriminative representations, the hidden features are approximately uninformative about class labels, corresponding to the high-noise regime. In this regime, the structure-dependent term $\frac { 2 ( 1 - h _ { E } ) ( h _ { E } - h _ { E } ^ { * } ) } { h _ { E } ( 1 - h _ { E } ^ { * } ) }$ contributes negligibly compared to the noise-driven term $2 \sigma ^ { 2 } | L |$ . Consequently, Equation (57) simplifies to:

$$
a _ { \mathrm { i n i t } } \simeq \frac { 2 \sigma ^ { 2 } | L | } { \sigma ^ { 2 } | L | } = 2 .\tag{58}
$$

This initial value is independent of both the homophily level $h _ { E }$ and the class distribution $\{ n _ { k } \}$ providing a universal, structure-agnostic initialization.

Separation of Concerns. This derivation reveals a principled division of roles between PRAT’s two learnable components:

• Metric tensor Λ: Encodes the geometric structure of the graph. Initialized from the graph signature eigenvalues $\Lambda _ { \mathcal { G } , \mathrm { t r a i n } } ,$ , it captures whether the underlying manifold is Riemannian or pseudo-Riemannian, directly reflecting the homophilic or heterophilic nature of the graph.

• Self-loop scaling a: Controls the signal-to-noise balance in attention computation. Initialized at 2 corresponding to uninformative representations, it adapts during training as the network learns discriminative features.

## G Statistical reliability

How the reported margins are tested, and how the spread divides between seeds and splits.

## G.1 Statistical significance

Several margins in Table 3 and Table 4 are small relative to the split deviation, so we test them at two levels.

A Friedman test over all models of each table rejects equality of mean ranks on both suites, and PRAT holds the best mean rank on each: 2.63 against 2.88 for BernNet on the standard suite, with $\chi ^ { 2 } = 5 4 . 4 6 , p = 4 . 4 \times 1 0 ^ { - 6 }$ , and 1.86 against 4.57 for GT-sep on the Platonov suite, with $\chi ^ { 2 } = \mathrm { 6 7 . 5 9 , } p = \mathrm { \bar { 1 } . 2 } \times 1 0 ^ { - 8 }$ . With eight and seven datasets the Nemenyi critical difference is 8.73 and 8.72 rank units at $\alpha = 0 . 0 5$ , so this post-hoc test separates PRAT from the four and five lowest-ranked models, and the per-dataset tests carry the comparison with the strongest baselines.

Per dataset we use Welch’s t where only a mean and a standard deviation are available, and an exact paired Wilcoxon signed-rank test where per-split values exist on both sides. PRAT beats GAT-sep on roman-empire at $p = 3 . 9 \times 1 0 ^ { - 5 }$ and on tolokers at $p = 8 . 2 \times 1 0 ^ { - 6 }$ . We re-ran GAT-sep, the strongest baseline on roman-empire, in the identical pipeline, reproducing its 88.75 there, which allows a paired test on the shared splits that PRAT wins on all ten at $p = 0 . 0 0 2$

In four comparisons PRAT has the higher mean but the difference is not significant at $\alpha = 0 . 0 5$ , and we count these as ties: amazon-ratings against SAGE at $p = 0 . 5 5$ , Citeseer against Geom-GCN at $p = 0 . 9 4$ , filtered Chameleon against FAGCN at $p = 0 . 5 2$ , and Actor against FAGCN at $p = 0 . 2 0$ The paired tests behind the ablation are given with Table 13 and Table 14.

## G.2 Variation across training seeds

The tables vary data splits with the training seed fixed, so that a split is reproducible. To check that fixing the seed hides no variance, we reran the ten-split evaluation on the five large Platonov datasets with three training seeds, in one matched run that reproduces Table 4 to within 0.11 points.

Table 12: Ten-split mean under three training seeds, against the split-to-split deviation of the reported run.
<table><tr><td>Dataset</td><td>seed 42</td><td>seed 7</td><td>seed 123</td><td>seed-to-seed sd</td><td>split-to-split sd</td></tr><tr><td>roman-empire</td><td>90.16</td><td>90.08</td><td>89.98</td><td>0.09</td><td>0.56</td></tr><tr><td>amazon-ratings</td><td>53.79</td><td>53.56</td><td>53.39</td><td>0.20</td><td>0.59</td></tr><tr><td>minesweeper</td><td>92.25</td><td>92.02</td><td>91.34</td><td>0.47</td><td>0.68</td></tr><tr><td>tolokers</td><td>84.93</td><td>85.09</td><td>85.14</td><td>0.11</td><td>0.65</td></tr><tr><td>questions</td><td>78.06</td><td>78.08</td><td>77.96</td><td>0.07</td><td>1.00</td></tr></table>

In Table 12 each ten-split mean moves by under one point across the three seeds. On a fixed split, the deviation across the three seeds, pooled over the ten splits, is 0.25, 0.40, 0.50, 0.37 and 0.47 points in the order of the table, below the split-to-split deviation of a single run on every dataset, so the error bars we report capture the dominant source of variability.

## H Graph Transformer and geometric baselines

Four rows of Table 3 answer two further questions: how the neighbor-only Graph Transformer used by Platonov et al. [28] behaves on these splits (GT and GT-sep), and how PRAT compares against models that already carry an indefinite or curved geometry (Q-GCN on a pseudo-hyperboloid, HAT on the Poincaré ball).

Two readings follow. First, GT is close on the two Wikipedia graphs, at 68.71 against 68.73 on Chameleon and 56.83 against 56.96 on Squirrel, and an exact paired test on the shared splits finds no significant difference $( p = 0 . 1 6$ and $p = 0 . 7 0 )$ . GT differs from PRAT in its projections and layer architecture as well as in the metric, so it is not a geometry-matched control; the causal claim in this paper rests on the paired ablation of Section I.1, not on this table.

Second, among the geometric models the pattern is cleaner. On the three homophilic graphs, where no negative direction can be exploited, everything sits within two points. On the heterophilic half Q-GCN trails by 7.61 to 26.21 points and HAT, curved but positive definite, trails on every dataset here. Q-GCN also trails PRAT by 3.87 and 4.48 points on filtered Chameleon and filtered Squirrel, at $3 8 . 8 2 \pm 2 . 2 8$ and $3 5 . 1 2 \pm 1 . 3 6$

## I The sign-conditional ablation

What happens when the metric is forbidden to be indefinite, and what that does and does not license.

## I.1 Three-way ablation on the Platonov suite

Table 5 runs the three-way ablation on the standard benchmarks. Table 13 repeats it on the Platonov suite and the filtered pair, inside one matched run whose PRAT rows reproduce Table 4 to within 0.24 points. Because the three variants share every split, an exact paired Wilcoxon signed-rank test over the ten splits is available, which a comparison of means alone is not.

Table 13: Three-way ablation on the Platonov benchmarks, grouped as in Table 4, with the graph signature index of each dataset. p is an exact paired Wilcoxon signed-rank test of PRAT against $\mathrm { P R A T _ { p o s } }$ over the ten shared splits; bold $p$ marks significance at $\alpha = 0 . 0 5$ . Among the three variants the best mean is in bold, and where means tie the smaller deviation takes it.
<table><tr><td>Dataset</td><td>sg</td><td>PRAT</td><td> $\mathrm { P R A T _ { p o s } }$ </td><td> $\mathrm { P R A T _ { f i x e d } }$ </td><td>p</td></tr><tr><td>Cham.-filt. Sq.-filt.</td><td>(3,0,2)</td><td> $\pm 2 . 4 5 \pm 2 . 2 0$ </td><td> $4 0 . 3 9 { \pm } 2 . 7 3$   $3 9 . 3 9 { \pm } 2 . 1 6$ </td><td> $4 2 . 0 0 { \pm } 2 . 6 6$ </td><td>0.039</td></tr><tr><td></td><td>(2,1,2)</td><td> $3 9 . 5 8 { \pm } 1 . 7 1 $ </td><td></td><td> $\mathbf { 4 0 . 2 2 \pm 1 . 5 9 }$ </td><td>0.770</td></tr><tr><td>roman-empire</td><td>(6,7,5)</td><td> $\mathbf { 9 0 . 1 6 { \pm 0 . 5 6 } }$ </td><td> $8 7 . 5 6 { \pm } 0 . 5 7 $ </td><td> $9 0 . 0 2 { \pm } 0 . 5 1 $ </td><td>0.002</td></tr><tr><td>amazon-ratings</td><td>(3,2,0)</td><td> ${ \pm } \mathbf { 0 . } 7 \mathbf { 9 } { \pm } \mathbf { 0 . } 5 \mathbf { 9 }$ </td><td> $5 3 . 7 2 { \scriptstyle \pm 0 . 4 8 }$ </td><td> $5 3 . 7 0 { \scriptstyle \pm 0 . 6 3 }$ </td><td>0.492</td></tr><tr><td>minesweeper</td><td>(1,1,0)</td><td> $9 2 . 2 5 { \pm } 0 . 6 8$ </td><td> $\mathbf { 9 2 . 2 8 { \pm } 0 . 5 6 }$ </td><td> $9 2 . 0 9 { \pm } 0 . 7 0 $ </td><td>0.846</td></tr><tr><td>tolokers</td><td>(2,0,0)</td><td> $8 4 . 9 3 { \pm } 0 . 6 5$ </td><td> $8 4 . 8 3 { \pm } 0 . 7 2 $ </td><td> $\mathbf { 8 4 . 9 3 \pm 0 . 5 7 }$ </td><td>0.131</td></tr><tr><td>questions</td><td>(1,1,0)</td><td> $\mathbf { 7 8 . 0 6 { \pm } 1 . 0 0 }$ </td><td> $7 7 . 9 2 \pm 1 . 2 1 $ </td><td> $7 7 . 9 1 \pm 1 . 0 7$ </td><td>0.625</td></tr></table>

The gap between PRAT and $\mathrm { P R A T _ { p o s } }$ appears only where the signature has negative eigenvalues, and there it is significant: +2.60 points on roman-empire on all ten splits, and +2.06 on filtered Chameleon on eight of ten. On the four datasets with $n _ { - } = 0$ the two variants differ by at most 0.14 points at $p \geq 0 . 1 3 .$ and the single nominal win for the constraint is minesweeper by 0.03 points at $p = 0 . 8 5$ . Filtered Squirrel is the boundary case. Its training-split spectrum is nearly flat beyond $\lambda _ { 1 } = 0 . 1 3 7 5$ , the next three eigenvalues carrying 3.2%, 0.2% and −0.7% of it, so what the initialization supplies there is near-zero suppression rather than sign, and the positive-definite constraint costs 0.19 points at $p = 0 . 7 7$ . A constraint that mattered for generic reasons would not localize this way.

$\mathrm { P R A T _ { f i x e d } }$ matches PRAT on six of the seven, the exception being minesweeper by 0.16 points at $p = 0 . 0 0 2$ . Freezing Λ freezes the inertia and not the metric, since $\mathbf { \breve { D } } = ( \mathbf { W ^ { \prime } } ) ^ { \top } \boldsymbol { \Lambda } \mathbf { W ^ { \prime } }$ keeps adapting through $\mathbf { W } ^ { \prime } ,$ , so these rows say that on large graphs the initialization is what has to supply the spectrum while the pullback is still learned. Where the training-split signature is noisy, on the small graphs of Table 5, learning Λ as well is what recovers 16.7 points on Cornell.

## I.2 The signature as a diagnostic rather than a predictor

It is natural to ask whether n<sub>−</sub> predicts how much a signed or high-pass model gains. Rank-correlating n<sub>−</sub> from Table 1 against what each such baseline of Table 3 gains over GCN is positive throughout, at $\rho = + 0 . 2 3$ for FAGCN, +0.33 for GPR-GNN, +0.60 for BernNet and +0.33 for their mean gain; none is significant at eight datasets, and Squirrel is the exception, where the three models lose 2.82 points on average despite $n _ { - } = 1$

We therefore present $s g$ as a diagnostic of the condition under which an indefinite mechanism should pay, not as a predictor of accuracy. The direct evidence for the diagnostic reading is the signconditional ablation of Section I.1, which manipulates the admissible sign structure with everything else held fixed and finds the effect exactly where the condition holds. Predicting the gain is hard for measures of this kind in general: Zheng et al. [48] report their own composite homophily score correlating with the graph-aware advantage at $p \geq 0 . 1 1$ over thirty-one graphs.

## J What the initialization has to supply

Which part of the signature carries the effect, and whether ground-truth labels are needed to obtain it.

## J.1 What the initialization has to supply

The ablation above shows that the freedom to be indefinite is worth something. It does not say which part of the signature carries it, the sign pattern or the magnitudes. Table 14 separates the two with four controls that keep the architecture fixed and change the initial diagonal; abs\_pos also holds Λ positive definite during training, as $\mathrm { P R A T _ { p o s } }$ does.

Table 14: Initialization controls on the two indefinite benchmarks. The signature row is the reference: $\Delta$ is the loss against it and p an exact paired Wilcoxon over ten shared splits, with bold $p$ marking significance at $\alpha = 0 . 0 5$ . abs\_pos keeps the magnitudes and deletes the negative directions; sign keeps the signs at a uniform magnitude; flip keeps the magnitudes and inverts the signs; random draws an indefinite diagonal of the same inertia at the signature’s scale.
<table><tr><td></td><td colspan="3">roman-empire (6,7,5)</td><td colspan="3">Cham.-filt. (3,0,2)</td></tr><tr><td>Initialization</td><td>Accuracy</td><td>∆</td><td>p</td><td>Accuracy</td><td>∆</td><td>p</td></tr><tr><td>signature (PRAT)</td><td>90.16±0.56</td><td></td><td></td><td>42.45±2.20</td><td></td><td></td></tr><tr><td>abs_pos</td><td>87.91±0.61</td><td>2.26</td><td>0.002</td><td>41.64±2.50</td><td>0.81</td><td>0.275</td></tr><tr><td>sign</td><td>89.86±0.62</td><td>0.30</td><td>0.065</td><td>39.90±2.22</td><td>2.55</td><td>0.008</td></tr><tr><td>flip</td><td>89.96±0.64</td><td>0.21</td><td>0.084</td><td>39.44±3.20</td><td>3.01</td><td>0.027</td></tr><tr><td>random 1</td><td>89.53±0.31</td><td>0.63</td><td>0.010</td><td>40.81±2.77</td><td>1.63</td><td>0.037</td></tr><tr><td>random 2</td><td>90.16±0.47</td><td>0.00</td><td>1.000</td><td>39.83±2.40</td><td>2.62</td><td>0.037</td></tr><tr><td>random 3</td><td>89.58±0.42</td><td>0.58</td><td>0.002</td><td>40.11±3.52</td><td>2.34</td><td>0.055</td></tr><tr><td>identity</td><td> $8 7 . 6 1 { \pm } 0 . 6 9$ </td><td>2.55</td><td></td><td>40.18±2.39</td><td>2.27</td><td></td></tr><tr><td>positive definite</td><td> $8 7 . 5 6 { \pm } 0 . 5 7 $ </td><td>2.60</td><td>0.002</td><td>40.39±2.73</td><td>2.06</td><td>0.039</td></tr></table>

The controls give different results on the two graphs, and the difference is informative. On romanempire the only variant that loses decisively is abs\_pos, which keeps the magnitude profile and removes the negative directions: it costs 2.26 points on all ten splits. Everything that retains negative directions, including sign-only, sign-flipped and all three random draws of matching inertia, stays within 0.7 points. The reason is visible in the spectrum: eleven of roman-empire’s eighteen trainingsplit eigenvalues are below a sixth of $\lambda _ { 1 }$ in magnitude, so the profile is close to flat and the only information the signature carries there is that negative directions exist. On filtered Chameleon the spectrum is not flat, its second eigenvalue carrying 10.8% of the first against 3.2% on filtered Squirrel, and there the specific values matter: every random draw of matching inertia loses 1.63 to 2.62 points and sign-only loses 2.55.

An uninformed initialization is the other end of the scale. An identity initialization that is free to become indefinite but carries no signature information reaches 87.61 on roman-empire, level with the positive-definite variant at 87.56. Freedom alone therefore buys nothing; what has to be supplied is the negative directions and, where the spectrum is informative, their relative sizes as well.

The initialization also survives training. Figure 3 reports the converged positive-eigenvalue ratio on synthetic graphs. On real graphs we logged Λ throughout training. Between 97.2% and 100% of its entries keep their initial sign at convergence, the cosine to the initialization stays above 0.985, and roman-empire’s negative-entry fraction holds between 0.38 and 0.42 while the magnitudes adapt. Every sign change falls on an entry initialized from a training-split eigenvalue within $1 0 ^ { - 4 }$ of zero, which the convention of Figure 8 rounds to zero, so every entry whose training-split eigenvalue counts as positive or negative keeps its sign. On tolokers, whose signature is strictly positive, no entry becomes negative. The model does not drift back to a Riemannian solution when it is free to, which is what the diagnose-then-learn reading predicts.

## J.2 Estimating the signature without labels

The signature is computed from training-node labels, but nothing in the forward pass reads a label. To test whether the labels are the essential ingredient or merely a convenient one, we re-estimate the signature from k-means pseudo-labels on node features, with the number of clusters set to the number of classes and no ground-truth label used in the estimate, and train as usual.

Table 15: Label-free initialization. The inertia is that of the eigenvalue vector each initialization is built from, before normalization, at the $1 0 ^ { - 4 }$ rounding convention of Figure 8; the two signature estimates use the training nodes.
<table><tr><td></td><td colspan="2">roman-empire</td><td colspan="2">Cham.-filt.</td></tr><tr><td>Initialization</td><td>Accuracy</td><td>inertia</td><td>Accuracy</td><td>inertia</td></tr><tr><td>label-based signature</td><td> $\mathbf { 9 0 . 1 6 { \pm } 0 . 5 6 }$ </td><td>(8,4,6)</td><td>42.45±2.20</td><td>(3,0,2)</td></tr><tr><td>k-means pseudo-labels</td><td> $9 0 . 1 0 { \pm } 0 . 4 8$ </td><td>(7,4,7)</td><td>42.70±2.00</td><td>(1,3,1)</td></tr><tr><td>uninformed identity</td><td> $8 7 . 6 1 { \pm } 0 . 6 9$ </td><td>(18,0,0)</td><td> $4 0 . 1 8 \pm 2 . 3 9$ </td><td>(5,0,0)</td></tr></table>

In Table 15 recovery is complete on both indefinite benchmarks, while the uninformed initialization loses 2.55 and 2.27 points. The feature-only estimate is not a reconstruction of the label signature: its inertia is $( 7 , 4 , 7 )$ against (8, 4, 6) on roman-empire and (1, 3, 1) against $( 3 , 0 , 2 )$ on filtered Chameleon, and it still recovers the accuracy. Taken with Section J.1, the picture is that what has to be supplied is the presence of negative directions together with near-zero suppression, not a faithful copy of the label spectrum, and that node features alone can supply it. A complete self-supervised treatment, the training objective as well as the initialization, is outside the scope of this paper.

## K Behavior under sparse and noisy labels

How far the signature and the accuracy survive when training labels are few or wrong.

## K.1 Stability of the signature under label sparsity and noise

Table 17 varies how many labels the model trains on. A separate question is how quickly the signature estimate itself degrades. We keep a random fraction p of the training nodes, or flip a fraction η of the training labels to a random other class, recompute the signature and report the mean sign agreement of its eigenvalues against the full-training-label estimate over 100 repeats.

Table 16: Mean eigenvalue sign agreement of the estimated signature against the full-training-label signature, over 100 repeats.
<table><tr><td rowspan="2">Dataset</td><td colspan="3">fraction of training nodes kept</td><td colspan="3">fraction of training labels flipped</td></tr><tr><td> $p { = } 1 0 \%$ </td><td> $p { = } 2 5 \%$ </td><td> $p { = } 5 0 \%$ </td><td> $\eta { = } 1 0 \%$ </td><td> $\eta = 2 0 \%$ </td><td> $\eta = 3 0 \%$ </td></tr><tr><td>roman-empire</td><td>0.88</td><td>0.88</td><td>0.93</td><td>0.97</td><td>0.93</td><td>0.87</td></tr><tr><td>amazon-ratings</td><td>0.74</td><td>0.85</td><td>0.91</td><td>1.00</td><td>0.87</td><td>0.80</td></tr><tr><td>minesweeper</td><td>0.69</td><td>0.82</td><td>0.98</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td>tolokers</td><td>0.99</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>0.96</td></tr><tr><td>questions</td><td>0.93</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td>Chameleon</td><td>0.92</td><td>0.93</td><td>0.91</td><td>0.94</td><td>0.93</td><td>0.92</td></tr><tr><td>Squirrel</td><td>0.93</td><td>0.92</td><td>0.93</td><td>0.95</td><td>0.93</td><td>0.90</td></tr><tr><td>Actor</td><td>0.45</td><td>0.53</td><td>0.67</td><td>0.86</td><td>0.83</td><td>0.84</td></tr><tr><td>Cornell</td><td>0.11</td><td>0.53</td><td>0.85</td><td>0.93</td><td>0.90</td><td>0.87</td></tr><tr><td>Texas</td><td>0.34</td><td>0.67</td><td>0.94</td><td>0.86</td><td>0.81</td><td>0.81</td></tr></table>

Table 16 reports the agreement. Large graphs are stable at modest label fractions. Actor’s low agreement sits in near-zero eigenvalues at the rounding threshold, where the dominant directions keep their sign, and the 183-node WebKB graphs lose the signature below a quarter of an already small training set. That is exactly where Table 17 shows Cornell losing its lead, with all three models inside 0.6 points at 10%, and it is why the learnable Λ matters most on small graphs.

What matters in the end is accuracy. Training PRAT from the corrupted initialization, with training labels left clean and only the initialization degraded, costs at most 0.11 points on roman-empire $( 9 0 . 1 6 \to 9 0 . 0 5 \mathrm { ~ a t ~ } \eta = 1 0 \% )$ and 1.46 on filtered Chameleon $( 4 2 . 4 5 \to 4 0 . 9 9 \mathrm { a t } p = 2 5 \% )$ , both inside the split deviation, and every corrupted variant stays at or above the positive-definite ablation.

## K.2 Training Fraction Sensitivity

We evaluate how PRAT’s advantage depends on label availability by training on 10%, 20%, 40%, 60%, and 100% of the original training split on Cornell, Texas, Actor, and Cora, using each model’s best hyperparameters and 10 data splits.

Table 17: Accuracy (%) at varying training fractions. Bold indicates the best model per dataset and fraction.
<table><tr><td>Dataset</td><td>Model</td><td>10%</td><td>20%</td><td>40%</td><td>60%</td><td>100%</td></tr><tr><td rowspan="3">Cornell</td><td>PRAT</td><td> $4 4 . 0 5 { \pm } 4 . 5 3 $ </td><td> $\pm 9 . 4 6 \pm 9 . 3 7$ </td><td> ${ \pm } \mathbf { 7 . 0 3 } { \pm } \mathbf { 8 . 5 9 }$ </td><td> ${ \bf 6 8 . 3 8 { \pm 6 . 4 0 } }$ </td><td> ${ \bf 7 1 . 8 9 \pm 4 . 7 1 }$ </td></tr><tr><td>GAT</td><td> $4 4 . 3 2 \pm 5 . 1 6$ </td><td>41.62±8.04</td><td> $4 5 . 6 8 \pm 9 . 0 0$ </td><td> $4 6 . 7 6 { \pm } 9 . 3 7$ </td><td> $5 0 . 5 4 \pm 7 . 5 5$ </td></tr><tr><td>GATv2</td><td>44.59±5.44</td><td>41.35±4.02</td><td> $4 3 . 5 1 \pm 3 . 9 1$ </td><td> $4 6 . 4 9 { \pm } 1 0 . 3 8 $ </td><td>54.32±9.70</td></tr><tr><td rowspan="3">Texas</td><td>PRAT</td><td>63.24±6.42</td><td> ${ \bf 6 7 . 5 7 } \pm { \bf9 . 8 2 }$ </td><td> $\mathbf { 7 4 . 0 5 \pm 6 . 4 2 }$ </td><td> ${ \bf 7 8 . 6 5 \pm 5 . 3 3 }$ </td><td>84.32±5.10</td></tr><tr><td>GAT</td><td>60.54±6.07</td><td>59.73±9.16</td><td> $6 1 . 6 2 \pm 5 . 5 1$ </td><td> $6 1 . 0 8 { \pm } 7 . 2 7$ </td><td> $6 4 . 5 9 { \pm } 6 . 6 7$ </td></tr><tr><td>GATv2</td><td>57.57±6.95</td><td>60.54±8.65</td><td> $6 1 . 6 2 { \pm } 4 . 8 0 $ </td><td>62.16±6.16</td><td>66.76±6.63</td></tr><tr><td rowspan="3">Actor</td><td>PRAT</td><td> $\pm 7 . 4 0 \pm 1 . 6 1$ </td><td> $\pm \mathbf { 8 . 9 3 } \pm \mathbf { 1 . 2 0 }$ </td><td> ${ \bf 3 3 . 7 2 \pm 1 . 1 4 }$ </td><td> ${ \bf 3 5 . 6 1 \pm 1 . 5 7 }$ </td><td>37.26±1.44</td></tr><tr><td>GAT</td><td> $2 6 . 3 2 { \pm } 1 . 1 1$ </td><td> $2 7 . 5 3 { \pm } 1 . 7 5 $ </td><td> $2 8 . 7 9 { \pm } 1 . 6 4 $ </td><td> $2 9 . 5 6 \pm 1 . 3 1 $ </td><td>31.42±0.69</td></tr><tr><td>GATv2</td><td> $2 5 . 6 0 { \pm } 1 . 3 3 $ </td><td> $2 7 . 4 6 \pm 1 . 1 0$ </td><td> $3 0 . 3 6 \pm 1 . 3 9$ </td><td> $3 2 . 2 8 \pm 1 . 3 3$ </td><td>34.24±1.10</td></tr><tr><td rowspan="3">Cora</td><td>PRAT</td><td> $7 4 . 2 4 \pm 1 4 . 9 7$ </td><td> $8 3 . 0 9 \pm 2 . 1 2$ </td><td> $8 5 . 2 7 { \pm } 1 . 1 6 $ </td><td> $8 6 . 6 1 \pm 1 . 1 4$ </td><td>88.42±1.17</td></tr><tr><td>GAT</td><td> $\mathbf { 8 0 . 1 1 \pm 1 . 9 9 }$ </td><td> $8 2 . 9 8 { \pm } 1 . 7 6 $ </td><td> $8 4 . 8 8 \pm 1 . 1 7$ </td><td> $8 6 . 5 4 \pm 1 . 3 2$ </td><td>88.32±1.09</td></tr><tr><td>GATv2</td><td> $7 8 . 8 8 \pm 1 . 7 5$ </td><td> $\mathbf { 8 3 . 2 6 { \pm 1 . 9 8 } }$ </td><td> $\mathbf { 8 5 . 6 2 \pm 1 . 2 6 }$ </td><td> $\mathbf { 8 6 . 7 4 \bot 1 . 0 6 }$ </td><td>87.99±0.93</td></tr></table>

On heterophilic datasets, PRAT consistently outperforms GAT and GATv2, but the gap narrows with fewer labels (Texas: $+ 1 9 . 7 \mathrm { p p }$ at $1 0 0 \%  + 2 . 7 \mathrm { p p }$ at 10%). This reveals the mechanism behind PRAT’s advantage: as $S _ { \mathcal { G } }$ becomes noisier with fewer labels, initialization quality degrades and performance approaches that of standard attention.

On homophilic Cora, all models perform comparably at fractions $\geq 2 0 \% ,$ as expected, since the positive-definite structure of Cora’s $S _ { \mathcal { G } }$ initializes PRAT’s metric as positive semi-definite, offering no additional geometric advantage over GAT/GATv2. At 10%, PRAT shows higher variance (74.24±14.97%) due to one split where $S _ { \mathcal { G } }$ computed from ∼163 training nodes across 7 classes produces a severely noisy initialization. This is consistent with our analysis: the $S _ { \mathcal { G } }$ quality degrades when training labels are extremely sparse relative to the number of classes, but performance recovers rapidly at 20% (83.09±2.12%).

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: The main claims in the abstract and introduction are supported by the theoretical analysis in Section 2, the proposed PRAT method in Section 3, and empirical results in Section 4. The scope is limited to node classification on homophilic and heterophilic graph benchmarks.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The Limitations paragraph after the conclusion states four: the signature is initialized from training-node labels, which ties the method as presented to (semi-)supervised node classification; signature shift between training and test graphs is not studied; graphlevel tasks are out of reach because S<sub>G</sub> is undefined without node labels; and rank(D) ≤ |L| bounds the structure the metric can carry when classes are few. Appendix J.2 reports the label-free variant that weakens the first of these.

Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate “Limitations” section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: The paper states the assumptions and formal results in Section 2, including Theorems 2.2–2.5, and provides detailed proofs in Appendix A.

## Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: Section 4.2 specifies the train/validation/test split, use of fixed and random splits, self-loop preprocessing, random seed control, and hyperparameter tuning protocol. Detailed configurations are provided in Appendix F.1.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: We provide a repository at https://github.com/voltwin-dev/PRAT\_ NeurIPS\_2026, containing the implementation of PRAT, scripts for reproducing the experiments, configuration files, license information, and instructions for the reported benchmarks. The released code is provided for research and non-commercial use only due to patentrelated restrictions, while the datasets used in this work are publicly available benchmark datasets and cited in Section 4.1.

## Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: The paper specifies dataset splits, preprocessing, random seeds, hyperparameter tuning, and evaluation metrics in Section 4.2, with additional configuration details in Appendix F.1.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: All benchmark, ablation and appendix tables report the mean over the ten fixed evaluation splits with the standard deviation across those splits, so the error bars capture split variability at a fixed seed. Appendix G.1 adds a Friedman test across models on each suite and, per dataset, Welch’s t-test and the exact paired Wilcoxon signed-rank test over the ten splits, and reports which comparisons are ties. Appendix G.2 separates seed variability from split variability by repeating the ten-split evaluation under three training seeds.

## Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

• If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: All experiments we conducted were run on a single NVIDIA A100 80GB GPU; the used computing resources are reported in Appendix F.2.

## Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The research appears to conform to the NeurIPS Code of Ethics and does not involve human subjects, private data, or high-risk deployment.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [No]

Justification: The paper does not include a separate discussion of societal impacts. PRAT is a foundational method for node classification on graphs and is not tied to a particular application or deployment, and we do not see a direct path from it to negative applications.

Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: The paper does not release a high-risk pretrained model, scraped dataset, or system with a clear high-risk misuse scenario requiring special safeguards.

Guidelines:

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: The paper cites the original sources of the benchmark datasets and baseline methods used in the experiments. The repository documents the licenses and terms of use for the datasets and baseline code assets, and states that the released PRAT implementation is provided under a research-only, non-commercial license due to patent-related restrictions.

Guidelines:

• The answer [N/A] means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: We release an implementation of PRAT with documentation, experiment scripts, configuration files, license information, and instructions for reproducing the reported results. The released PRAT code is made available for research and non-commercial use only due to patent-related restrictions.

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: The paper does not involve crowdsourcing experiments or research with human subjects.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: The paper does not involve crowdsourcing experiments or research with human subjects, so IRB approval or equivalent review is not applicable.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [N/A]

Justification: LLMs are not used as an important, original, or non-standard component of the core method, experiments, or scientific contributions of this work.

Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.