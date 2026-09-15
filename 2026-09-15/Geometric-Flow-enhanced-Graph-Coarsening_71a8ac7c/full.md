# Geometric Flow enhanced Graph Coarsening

Chaoqun Fei, Guoxuan Li, Tinglve Zhou, Chuanqing Wang, Yangyang Li

Abstract—Recently, researchers have proposed a graph pooling operation, akin to the pooling process in conventional convolutional neural networks (CNN), aimed at reducing the computation cost of Graph convolutional neural networks (GCNNs). While most GCNN-based methods treat graph pooling as a node clustering problem and propose learning a cluster assignment matrix, existing clustering-based pooling methods tend to focus solely on the rough topology information of graphs, neglecting the exploitation of higher-order mutual connections among neighbors. In terms of message passing on graph, the ease of information passing on edges reflects the closeness between neighboring nodes, which significantly relies on the interconnectivity among neighbors. In this study, we address this gap by considering such local connection information and introducing a novel graph pooling method named RicciPool. We introduce discrete graph curvature, particularly Ollivier-Ricci curvature, as a measure of higher-order connectivity around an edge. Subsequently, we construct an Ollivier-Ricci flow formula to reweigh edge weights, leveraging the crucial information provided by Ricci curvature, particularly vital for extracting clusters in graphs. Building upon this foundation, we utilize the spectral clustering technique to learn a new cluster assignment matrix. Experimental results on multiple bioinformatics protein datasets and social networks underscore the effectiveness of our proposed method.

Index Terms—Ollivier-Ricci curvature, Graph Ricci flow, Graph pooling, Spectral clustering, Graph convolutional neural network

## I. Introduction

Over the past decade, Convolutional Neural Networks (CNNs) have revolutionized fields such as computer vision and other domains characterized by Euclidean data structures with uniform, grid-like arrangements. Despite their significant success, fully exploiting the geometric or topological structures of non-Euclidean domains remains challenging. Non-Euclidean domains include data structures represented by sub-manifolds or graphs [24], encompassing areas like 3D graphics, social networks, biological systems, and web traffic. Traditional convolutional operations in CNNs rely on the Euclidean metric to compute linear weighted sums, but this metric is limited in capturing the nonlinear feature distribution of non-Euclidean structured data [4].

Graph convolutional neural networks (GCNNs) [5] [9] [41] [16] [7] are a novel deep learning method that extends Convolutional Neural Networks (CNNs) from traditional Euclidean domains to graph-structured data. GCNNs leverage the topology of graphs to learn node and graph embeddings by propagating information between nodes and their neighbors. Unlike traditional CNNs, GCNNs can handle non-uniform and irregular data structures such as social networks, molecular structures, and knowledge graphs. In recent years, GCNNs have demonstrated strong performance in tasks such as node classification, link prediction, and graph classification, becoming a key tool for handling graph data [6]. Node classification [12] [35] and link prediction [8] [47] focus on learning representations at the node level using features and topology, while graph classification [48] targets graph-level representation learning to predict labels for entire graphs. These methods have demonstrated promising results in various graphrelated tasks. Despite their promising results in various tasks, traditional GCNNs face challenges in efficiently computing large-scale graphs, characterized by many nodes or links [14]. In conventional CNNs, pooling operations effectively reduce feature dimensionality and computational costs for large-scale grid-structured data, allowing for deeper network designs and mitigating overfitting. However, these pooling techniques cannot be directly applied to graph-structured data due to its non-Euclidean nature. Consequently, several graph-based pooling methods have been developed in conjunction with GCNNs [17] [45] [46] [18] [19] [20].

Mainstream methods treat graph pooling as a node clustering task, where each cluster corresponds to a supernode in the pooled graph, as demonstrated in DiffPool [45] MinCutPool [1], and StructPool [46]. These methods focus on learning a new cluster assignment matrix. Topologically, nodes with shorter path distances in a graph should have a higher probability of being assigned to the same cluster. Accurately capturing pairwise node similarities is crucial for effective graph pooling. However, current methods primarily focus on node-specific attributes or coarse topological information, such as node degree, which fails to accurately reflect spatial similarities in node distribution. For instance, different graph topologies can have same node degree distribution, as shown in Fig. 1.

In terms of message passing on a graph, the ease of information transfer on edges reflects the closeness between neighboring nodes. Adjacency nodes in a well-connected community have highly overlapped neighborhood sets and numerous shortcuts, facilitating easier message passing compared to a hierarchically structured community. However, existing graph pooling methods do not distinguish between these node connection structures, often setting edge weights equally and overlooking local similarities. To address these differences, we draw inspiration from recent studies on graph curvature [22] to explore a new metric for reweighting graph edges. This metric leverages curvature information to ensure that easier message passing on an edge corresponds to shorter edge distances.

![](images/862176dfd031b011bdd2a3606d8faf477bb3e3713ce4cccdc27a58bcadebe62d.jpg)  
Fig. 1: Example of Ricci curvature on graphs. From left to right, the Ricci curvature of the red edges are positive, the orange edges are zero, and negative for the green edges in tree-structure graph, respectively. However, all edges have same weight 1.

Discrete graph curvature measures the deviation of a graph's neighborhood domains from a "flat" domain, such as a grid graph. Two prominent discrete curvature measures, Ollivier-Ricci curvature [27] [28] and Forman curvature [11], have recently garnered attention. Ollivier-Ricci curvature, based on optimal transportation theory, has been applied to various structural tasks [32] [25], whereas Forman curvature, derived from the graph Laplacian, is computationally simpler and faster but lacks geometric interpretability. In this work, we utilize Ollivier-Ricci curvature to investigate the local geometric structure of node distribution, measuring the difficulty of message passing on edges. Essentially, Ollivier-Ricci curvature assesses the connection density of subgraphs, which can be categorized into positive, zero, and negative values. Positive curvature signifies a dense, clique-like graph, zero curvature represents a grid-like graph, and negative curvature indicates a sparse, tree-like graph [21], as shown in Fig. 1.

In this paper, we introduce a novel method called RicciPool, a Graph Pool via Ricci Flow. We use the edge weight matrix as a distance metric to quantify message passing distances between nodes. Initially, like other GCNN-based methods, the edge weight matrix is an unweighted adjacency matrix with all edge weights set to one. We develop an Ollivier-Ricci flow formula on the graph, leveraging the corresponding Ollivier-Ricci curvature [26] to iteratively adjust the edge weight matrix for better discrimination. This flow guides graph curvature evolution through a nonlinear partial differential equation, expanding negative curvature edges and contracting positive curvature edges. This iterative process condenses nodes connected by overlapping edges and stretches treestructured edges. Following the graph Ricci flow evolution, we adopt the spectral clustering technique [31], similar to MinCutPool [1], combined with GCNN to derive a new cluster assignment matrix. Consequently, nodes in cliquelike subgraphs are more likely to be clustered together. During pooling, nodes within each cluster are aggregated into a new supernode, and the edge weight matrix and associated node features are updated using the learned assignment matrix. Theoretically, this preserves the original graph's topology. After pooling, the attribute features of supernodes incorporate local topology structures and the original node set's attributes.

The contributions of this paper are summarized as follows:

1) Enhance Ollivier-Ricci flow for graph pooling via a curvature-driven coarsening strategy that dynamically adjusts edge weights and preserves structural bottlenecks by focusing on high positive curvature connections.

2) Propose a novel spectral clustering-based topologypreserving graph pooling method. By learning a structure-aware node assignment matrix, this approach systematically preserves the global topological features of the original graph during coarsening.

3) Extensive experiments on benchmark graph classification datasets demonstrate the effectiveness and competitiveness of RicciPool.

The remainder of this paper is organized as follows. In Section II, we provide an overview of related works on graph pooling in Graph Neural Networks and discrete graph curvature. Section III presents the preliminaries, including basic definitions and theories. The proposed graph pooling method based on Ricci flow is detailed in Section IV. Section V presents the experiments and evaluation results. Finally, we conclude our work in Section VI.

## II. Related Works

## A. Graph Pooling in GNNs

In recent years, there has been significant development in GNN-based graph pooling methods, which can be primarily categorized into two types based on distinct pooling principles. The first type involves methods based on node selection principles. These methods establish a scoring metric to quantify node importance and subsequently rank nodes according to these scores. Examples include SORTPool [48], TOPKPool [13], and SAGPool [17]. However, their exclusive consideration of the global topological structure of a graph neglects potentially significant local connections contained within ignored nodes, which are lost during pooling.

The second type, termed graph coarsening, treats graph pooling as a node clustering problem based on graph topology. DIFFPool [45] constructs a multi-level GNN network with pooling layers in each hidden layer. Struct-Pool [46] utilizes conditional random fields to learn a cluster assignment matrix, while MinCutPool [1] frames the node clustering task as a minCUT problem. Additional approaches include EigenPool [23] and HaarPool [36], which consider both topology information between nodes and the embedding of the entire graph. However, few of them explicitly differentiate between different pairwise connected structures of nodes.

## B. Discrete Graph Curvature

Recent attention has been drawn to various proposals of discrete curvature on weighted or unweighted graphs, notably Ollivier-Ricci curvature and Forman curvature. Ollivier-Ricci curvature, rooted in optimal transportation theory, exhibits a more geometric nature. It has been integrated into graph neural networks to recalibrate different message channels [21] [44] and applied in network alignment [25] and community detection [32] [26]. In contrast, Forman curvature, grounded in the graph Laplacian, offers faster computation and has been used to analyze large-scale graphs such as complex networks [37] [34] [29] and bioinformatics-related graphs [30]. Forman curvature flow has also been proposed for community detection [38] [39]. However, comparative studies have favored Ollivier-Ricci curvature for graph structure detection [26]. Thus, in this study, we choose to utilize Ollivier-Ricci curvature for reweighing edge weights.

## III. Preliminaries

In this section, we aim to elucidate the fundamental concepts pertinent to our approach. Firstly, we delineate the foundational framework of graph convolutional neural networks. Subsequently, we expound upon the pooling operation within the realm of graph-structured domains. Lastly, we provide the formal definition of Ricci flow on a continuous manifold. Formally, let's consider a graph $G = \{ V , E \}$ with an unweighted adjacent matrix $\bar { \boldsymbol { A } } \in \left( 0 , 1 \right) ^ { N \times N }$ and a feature matrix $\bar { \boldsymbol { X } } \in \mathbb { R } ^ { N \times F }$ pertaining to the node set V. Here E denotes the edge set, F signifies the feature dimensions, and N represents the total number of nodes within the graph.

## A. Graph Convolutional Neural Networks

A graph can be characterized by its adjacency matrix and node feature matrix. Graph Convolutional Neural Networks (GCNNs) leverage these matrices to learn feature embeddings at either the node or graph level, employing both spectral and spatial methods. Spectral methods [9] [16] utilize the graph Laplacian operator to devise convolution operations. However, the convolution filters learned through spectral methods are tailored to specific graphs and thus lack universality to address diverse graph structures. In contrast, spatial approaches [41] [35] in GCNNs typically aggregate neighbor representations to formulate convolution operations for updating the features of individual nodes [48], formally defined as follows:

$$
X ^ { i + 1 } = f \left( D ^ { - 1 } { \hat { A } } X ^ { i } T ^ { i } \right) ,\tag{1}
$$

where $\begin{array} { r } { \hat { A } = A + I , D _ { j , j } = \sum _ { k } \hat { A } _ { j k } , X ^ { i } } \end{array}$ is the corresponding node feature matrix after i-th convolution operation, $T ^ { \bar { i } }$ is a trainable weight matrix to perform feature transformation, and $f \left( \cdot \right)$ represents a non-linear activation function. In our model, we use this version of GCNNs in Equation (1) for the basic convolution operation.

## B. Graph Pooling

Similar to the pooling operation in CNNs, graph pooling aims to reduce the size of nodes and edges within a graph. Formally, a graph pooling operator should produce a new feature matrix Xpool $\in \mathbb { R } ^ { \tilde { K } \times F ^ { \prime } }$ and an adjacency matrix $A ^ { p o o l } \in \mathbb { R } ^ { K \times K }$ in the pooling layer, typically with $N \geq K \ [ 1 ]$ . To achieve this objective, the graph pooling operation endeavors to learn a bilinear transformation matrix $P \in \mathbb { R } ^ { N \times K }$ between two adjacent convolutional layers, satisfying:

$$
A ^ { p o o l } = P ^ { T } A P ; \quad X ^ { p o o l } = P ^ { T } X .\tag{2}
$$

The new adjacent matrix $A ^ { p o o l }$ should inherit and preserve the global topological structure of the initial A. In clustering-based graph pooling methods, the transformation matrix P is referred to as the cluster assignment matrix.

## C. Ricci Flow

In 1982, Richard Hamilton first proposed Ricci flow equation [15], a parabolic partial differential equation deforming the Riemannian metric by its Ricci curvature:

$$
\frac { \partial } { \partial t } g _ { i j } = - 2 R _ { i j } .\tag{3}
$$

Here, $g _ { i j }$ denotes the Riemannian metric defined on a continuous manifold, and $R _ { i j }$ represents the corresponding Riemannian curvature. The Ricci flow can be conceptualized as a heat equation for the Riemannian metric. Through Ricci flow, the volumes of neighborhoods with positive sectional curvature generally contract, whereas those with negative sectional curvature expand. In Riemannian geometry, regions with significant positive curvature are generally more densely packed than those with negative curvature [26].

Inspired by Hamilton's Ricci flow, we propose a geometric evolution equation on graphs, termed the Ollivier-Ricci flow, for detecting clique-like subgraphs and treelike subgraphs. The Ollivier-Ricci flow interprets the edge weight matrix w as a discrete distance metric defined on a graph, analogous to the Riemannian metric $g _ { i j }$ in continuous manifold and evolves the edge weights over time: the edge weights of densely connected structures (positive Ricci curvature) are diminished, while those of sparsely connected structures (negative Ricci curvature) are enlarged [26].

## IV. Proposed Method

Firstly, we propose leveraging the curvature information of a graph to construct an Ollivier-Ricci flow on the graph, facilitating the iterative update of the edge weight matrix. Subsequently, based on the updated weight matrix, we partition the node set into K disjoint clusters. Finally, we utilize both GCNN and a learned cluster assignment matrix to condense the graph into a more compact version within the graph pooling layer.

## A. Ollivier-Ricci Curvature

Similar to the Ricci curvature in differential manifold Ollivier-Ricci curvature as its discrete counterpart on graphs was initially introduced by Y. Ollivier utilizing the optimal transportation theory. Ollivier defines the curvature of an edge by assigning a probability measure to each node at both ends of the edge, with the Ollivier-Ricci curvature of the edge determined by the optimal transportation cost between these two probability measures.

Given a graph $G = ( V , E , A )$ , we define an edge distance metric on $G ,$ termed the edge weight matrix w. Here, we consider the unweighted adjacency matrix $A \in ( 0 , 1 ) ^ { \dot { N } \times N }$ as the initial edge weight matrix. For each node $x \in V .$ we denotes a probability distribution $m _ { x }$ on the neighbor node set of x [2] (i.e., the nodes connected to x). The discrete Ricci curvature $R _ { x y }$ on edge $x y \in E$ is formally represented as [27]:

$$
R _ { x y } = 1 - \frac { W \left( m _ { x } , m _ { y } \right) } { d \left( x , y \right) } .\tag{4}
$$

The Wasserstein distance $W \left( m _ { x } , m _ { y } \right)$ here is defined as the optimal transportation cost required to move from $m _ { x }$ to $m _ { y }$ . The term $d \left( x , y \right)$ signifies the shortest path between nodes x and $y ,$ as outlined in Section IV-B. The measure $m _ { x }$ is represented as [26]:

$$
m _ { x } ^ { \alpha } ( x _ { i } ) = \left\{ \begin{array} { l l } { \alpha } & { \mathrm { i f } \quad x _ { i } = x } \\ { ( 1 - \alpha ) / | \pi ( x ) | } & { \mathrm { i f } \quad x _ { i } \sim x } \\ { 0 } & { \mathrm { o t h e r w i s e , } } \end{array} \right.\tag{5}
$$

where α is a parameter within $[ 0 , 1 ] , \pi ( x )$ denotes the neighbor set of x and $| \pi ( x ) |$ denotes nodes size in $\pi ( x )$ In this context, we set $\alpha = 0 . 5$ following to [26].

## B. Ollivier-Ricci Flow

Ollivier-Ricci flow defined on a graph can be seen as a discrete version of the Ricci flow on a continuous manifold. It regularize the edge weight matrix w on a graph by evolving the Ollivier-Ricci curvature R. The equation of Ollivier-Ricci flow is shown below:

$$
\frac { d } { d t } d ^ { t } \left( x , y \right) = - R _ { x y } ^ { t } d ^ { t } \left( x , y \right) .\tag{6}
$$

where, $d ( x , y )$ denotes the shortest path distance between nodes x and y.

Here, we define the concept of the shortest path d in a graph. Let $G = ( V , E , w )$ be a weighted graph where $w _ { x y } \mathrm { ~ > ~ } 0$ for every edge. The shortest path d between nodes in the set V is determined as follows:

$$
d \left( v , v ^ { \prime } \right) = \operatorname* { m i n } _ { w } \left\{ \sum _ { i = 1 } ^ { n } w _ { s _ { i } , s _ { i + 1 } } : s _ { i } \in V , s _ { 0 } = v , s _ { n + 1 } = v ^ { \prime } \right\} .\tag{7}
$$

The shortest path is calculated over all edge paths connecting v to $v ^ { \prime } .$ We refer to d as the distance metric induced by the edge weight w. By definition, for any three vertices $v , v ^ { \prime } ,$ and $v ^ { \prime \prime } \in \textit { V }$ , it holds that $d ( v , v ^ { \prime } ) + d ( v ^ { \prime } , v ^ { \prime \prime } ) \ \geq$ $d ( v , v ^ { \prime \prime } )$ , adhering to the metric rule.

Algorithm 1 Ollivier-Ricci Flow   
Input: the initial adjacency matrix $A \in ( 0 , 1 ) ^ { N \times N }$   
a parameter η and a threshold δ.   
1. Initialize that $w ^ { 0 } = A , d ^ { 0 } ( x , y ) = w _ { x y } ^ { 0 } .$   
2. Calculate the Ollivier-Ricci curvature of each edge   
using Eq. (4).   
3. while $\lvert \tilde { R } _ { x y } ^ { t + 1 } - \tilde { R } _ { x y } ^ { t } \rvert > \delta$ do   
4. Update the edge weight by Eq. (8);   
5. Update the Ollivier-Ricci curvature by Eq. (9).   
6. end while   
Output: the updated edge weight matrix $w ^ { R } .$

For the initial unweighted edge weight matrix, denoted as $w ^ { 0 } = A \in ( 0 , 1 ) ^ { N \times N }$ , the shortest path $d ( x , y )$ between any two neighboring nodes is equal to the edge weight $w ^ { 0 } ( x , y )$ and set to one.

According to the Ollivier-Ricci flow described in Eq. (6), we restrict our attention to the edge curvature between neighboring nodes. For such nodes, the shortest path distance $d ( x , y )$ coincides with the edge weight $w _ { x y } = d ( x , y )$ In each iteration, both the edge weight matrix w and the Ollivier-Ricci curvature are updated concurrently via the following flow process:

$$
w _ { x y } ^ { t + 1 } = d ^ { t } \left( x , y \right) \left( 1 - \eta \cdot R _ { x y } ^ { t } \right) ;\tag{8}
$$

$$
R _ { x y } ^ { t + 1 } = 1 - \frac { W ^ { t + 1 } \left( m _ { x } , m _ { y } \right) } { d ^ { t + 1 } \left( x , y \right) } .\tag{9}
$$

At the $( t + 1 )$ -th iteration, $w _ { x y } ^ { t + 1 }$ represents the weight of edge xy, $R _ { x y } ^ { t + 1 }$ denotes the Ollivier-Ricci curvature of edge xy, and $d ^ { \check { t } } \left( x , y \right)$ is the corresponding shortest path distance between x and y induced by the edge weight matrix $w ^ { t }$ . Initially, $w _ { x y } ^ { 0 } = A _ { x y }$ and $d ^ { 0 } \left( x , y \right) = w _ { x y } ^ { 0 } .$

Note that according to Eq. (8), this discrete Ricci flow process enlarges the edge weights with negative edge curvature while shrinking the weights of the positive curvature edges, if we consider the weight of an edge as the edge distance. Eventually, nodes connected by the clique-like structural edges are condensed and the tree-like structural edges are stretched. This effect allows for the easy separation of closely connected subgraphs through a simple strategy, such as removing edges with weights greater than a specified threshold. The detailed algorithm framework of the Ollivier-Ricci flow process is illustrated in Algorithm 1. In Step 3 of the algorithm, we define the stopping criterion for the Ricci flow iteration: the process terminates when the Ollivier-Ricci curvature of all graph edges stabilizes (i.e., ceases to change or changes by less than a predefined threshold δ) between consecutive iterations. The core function of this step is to verify satisfaction of this criterion. Upon meeting the condition, the Ricci flow iteration halts immediately.

## C. Nodes Clustering

Building upon the minCUT problem in spectral clustering [31], we employ spectral techniques to learn a node clustering assignment matrix for graph partitioning.

![](images/8d9dcc33c97439147a313d6bdf17e5d279c843837dffe14357fbfdede4bdd0b4.jpg)  
Fig. 2: Ollivier Ricci flow based graph pooling model.

Specifically, given a graph $G = ( V , E , A )$ , with N nodes and a predefined number of clusters K $( K < N )$ . V is the node set of G. The assignment principle in our approach is to minimize the total edge weights within clusters while maximizing the number of intra-cluster edges, and simultaneously minimize the number of inter-cluster edges while maximizing inter-cluster edge weights. This leads to the following loss function $\left( \mathrm { E q . ( 1 0 ) } \right)$ :

$$
\mathcal { L } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \left\{ \frac { w e i g h t s \left( V _ { k } \right) } { w e i g h t s \left( V / V _ { k } \right) } + \frac { d e g r e e \left( V / V _ { k } \right) } { l i n k s \left( V _ { k } \right) } \right\} .\tag{10}
$$

For the k-th cluster $V _ { k }$ , we define the following metrics:

$V / V _ { k }$ : The set of all nodes not in cluster $V _ { k }$

$w e i g h t s ( V _ { k } )$ : The sum of all edge weights within $V _ { k }$

$l i n k s ( V _ { k } ) ;$ The number of edges within $V _ { k }$

$d e g r e e ( V / V _ { k } )$ : The number of edges connecting $V _ { k }$ to the rest of the graph $\left( V / V _ { k } \right)$

$w e i g h t s ( V / V _ { k } )$ : The total weight of all edges between $V _ { k }$ and $V / V _ { k }$

To incorporate geometric information from graph curvature flow, we introduce the Ollivier-Ricci flow evolved weight matrix $w ^ { R }$ , obtained by iteratively updating the initial weight matrix $A .$ Let $P \in \{ 0 , 1 \} ^ { N \times \check { K } }$ be the cluster assignment matrix, where $P _ { i j } = 1$ if node i belongs to cluster $j ,$ and 0 otherwise, with $P _ { k }$ denoting its k-th column. The degree matrices are defined as $D = d i a g \left( A 1 _ { N } \right)$ and $D ^ { R } = { \dot { d } } i a g ( w ^ { R } 1 _ { N } ) $ for the original and evolved graphs, respectively. The cluster-wise objective is formulated as Eq.(11):

$$
\underset { P } { \arg \operatorname* { m i n } } \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \left\{ \frac { P _ { k } ^ { T } w ^ { R } P _ { k } } { P _ { k } ^ { T } D ^ { R } P _ { k } } + \frac { P _ { k } ^ { T } D P _ { k } } { P _ { k } ^ { T } A P _ { k } } \right\} .\tag{11}
$$

For the k-th cluster $V _ { k }$ , we define the following equivalent matrix forms:

• First Term $( P _ { k } ^ { T } w ^ { R } P _ { k } )$ : The equivalence between $P _ { k } ^ { T } w ^ { R } P _ { k }$ and $w e i g h t s ( V _ { k } )$ is exact. The quadratic form $P _ { k } ^ { T } w ^ { R } P _ { k }$ computes the sum of edge weights for all pairs of nodes in cluster k. In an undirected graph without self-loops, this equals $2 \times w e i g h t s ( V _ { k } )$ , as each internal edge is counted twice. Minimizing $\hat { P } _ { k } ^ { T } w ^ { R } P _ { k }$ is thus functionally identical to minimizing weights $( V _ { k } )$ for the purpose of optimization.

• Second Term $( P _ { k } D ^ { R } P _ { k } )$ : The denominator term $P _ { k } D ^ { R } P _ { k }$ is a surrogate for weights(V/Vk).

$$
\begin{array} { r l } & { P _ { k } ^ { T } D ^ { R } P _ { k } = \displaystyle \sum _ { i \in V _ { k } } ( \sum _ { j \in V _ { k } } w _ { i j } ^ { R } + \sum _ { j \in V / V _ { k } } w _ { i j } ^ { R } ) } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \quad \quad \quad  \\ &  \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \ \end{array}
$$

Consequently, maximizing $P _ { k } D ^ { R } P _ { k }$ is functionally equivalent to maximizing $w e i g h t s ( V / V _ { k } )$

• The metrics $l i n k ( V _ { k } )$ and $d e g r e e ( V / V _ { k } )$ represent the unweighted counterparts of $w e i g h t s ( V _ { k } )$ and u $r i g h t s ( V / V _ { k } )$ , respectively, focusing on edge counts rather than the sum of their weights.

Using the matrix identity for any matrix $X { : }$

$$
\sum _ { k = 1 } ^ { K } P _ { k } ^ { T } X P _ { k } = t r ( P ^ { T } X P )
$$

where the equality follows from the observation that $\begin{array} { l l l } { ( P ^ { T } X P ) _ { k k } } & { \stackrel { \circ } { = } } & { P _ { k } ^ { \check { T } } X P _ { k } } \end{array}$ and the trace sums the diagonal entries, i.e., $\begin{array} { r l r } { t r ( P ^ { T } X P ) } & { { } = } & { \sum _ { k } ^ { K } ( P ^ { T } X P ) _ { k k } } \end{array}$ This allows us to transform the component-wise sums into global trace terms. Denote these common values as $\Delta _ { m i n } ~ =$ min $\{ P _ { k } ^ { T } D ^ { R } P _ { k } , P _ { k } ^ { T } A P _ { k } \}$ and $\Delta _ { m a x } =$ max $\{ P _ { k } ^ { T } D ^ { R } P _ { k } , P _ { k } ^ { T } A P _ { k } \}$ for all k. Then, we derive the following inequalities:

$$
\begin{array} { r l } & { K \cdot \Delta _ { m i n } \leq t r ( P ^ { T } D ^ { R } P ) \leq K \cdot \Delta _ { m a x } , } \\ & { K \cdot \Delta _ { m i n } \leq t r ( P ^ { T } A P ) \leq K \cdot \Delta _ { m a x } . } \end{array}
$$

Under these conditions, we derive that:

$$
\begin{array} { r l } & { \frac { 1 } { K } \displaystyle \sum _ { k = 1 } ^ { K } \left\{ \frac { P _ { k } ^ { T } w ^ { R } P _ { k } } { P _ { k } ^ { T } D ^ { R } P _ { k } } + \frac { P _ { k } ^ { T } D P _ { k } } { P _ { k } ^ { T } A P _ { k } } \right\} } \\ & { \leq \frac { \sum _ { k = 1 } ^ { K } P _ { k } ^ { T } w ^ { R } P _ { k } } { K \cdot \Delta _ { m i n } } + \frac { \sum _ { k = 1 } ^ { K } P _ { k } ^ { T } A P _ { k } } { K \cdot \Delta _ { m i n } } . } \end{array}
$$

Then, we obtain the following inequation:

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \left\{ \frac { P _ { k } ^ { T } w ^ { R } P _ { k } } { P _ { k } ^ { T } D ^ { R } P _ { k } } + \frac { P _ { k } ^ { T } D P _ { k } } { P _ { k } ^ { T } A P _ { k } } \right\} } \\ & { \le \displaystyle \frac { \Delta _ { m a x } } { \Delta _ { m i n } } \left\{ \frac { t r \left( P ^ { T } w ^ { R } P \right) } { t r \left( P ^ { T } D ^ { R } P \right) } + \frac { t r \left( P ^ { T } D P \right) } { t r \left( P ^ { T } A P \right) } \right\} . } \end{array}
$$

Therefore, under the condition that $\Delta _ { m i n } \neq 0$ , minimizing Eq. (11) is equivalent to minimizing the following Eq. (12). This transformation is in line with MinCutPool [1]:

$$
\operatorname { a r g m i n } _ { P } \left\{ { \frac { t r \left( P ^ { T } w ^ { R } P \right) } { t r \left( P ^ { T } D ^ { R } P \right) } } + { \frac { t r \left( P ^ { T } D P \right) } { t r \left( P ^ { T } A P \right) } } \right\} ,\tag{12}
$$

The constraint $P 1 _ { K } = 1 _ { N }$ ensures each node is assigned to exactly one cluster, thereby maintaining the validity of the partitioning. However, in node clustering, it is possible that a cluster contains only a single node, in which case the minimum value of $\Delta _ { m i n } = 0$ under extreme conditions. Such a degenerate scenario would render the optimization of Eq. (11) ill-posed. To circumvent this issue, we reformulate Eq. (11) into Eq. (12). Moreover, in the design of Eq. (10), we impose a reciprocal constraint between the intra-cluster and inter-cluster degree distributions and their corresponding edge weight distributions. This reciprocal constraint acts as a mutual regularizer that effectively prevents extreme cases from arising.

Notice that problem (12) involves graph node clustering based on the principle of dense intra-group connections and sparse inter-group connections. This is a normalized cut problem and is NP-hard. Following the approach of MinCutPool, we employ a deep neural network to approximate the solution by learning a continuous mapping P. Technically, we integrate a Graph Convolutional Neural Network (GCNN) and optimize a relaxed version of the loss function, applying a softmax activation to the output layer to obtain a probabilistic assignment. Specifically, we learn a near-optimal soft cluster assignment matrix P by minimizing the following unsupervised loss function (Eq.(13)):

$$
\mathcal { L } _ { u } = \frac { t r \left( P ^ { T } w ^ { R } P \right) } { t r \left( P ^ { T } D ^ { R } P \right) } + \frac { t r \left( P ^ { T } D P \right) } { t r \left( P ^ { T } A P \right) } + \left. \frac { P ^ { T } P } { \Vert P ^ { T } P \Vert _ { F } } - \frac { I _ { K } } { \sqrt { K } } \right. _ { F } ^ { 2 }\tag{13}
$$

By minimizing the loss function $\mathcal { L } _ { u } ,$ each clique-like structural subgraph is more likely to be grouped into a same cluster depending on the updated edge weight matrix. The learnt assignment matrix P is a real matrix, with $P _ { j k }$ close to 1 if node j belongs to cluster k. This real matrix gives the opportunity of designing deeper network on graphs and then applying it to enhance the downstream tasks.

Compared with the MinCutPool, the advantage of our method is that we argue the geometric connection structure between neighbors and leverage the Ollivier-Ricci curvature to measure the connectivity around an edge. The Ricci curvature is very important information especially for extracting node clusters in the graph.

## D. Graph Pooling via Node Clustering

We conduct the pooling operation on graph nodes via the clustering assignment matrix P to aggregate the nodes within each cluster into a new supernode. In this way, since the node set is partitioned into K clusters, the new pooled graph $G ^ { p o o l }$ contains K supernodes. We need to recalculate two matrices around the pooled graph $G ^ { p o o l }$ One is the coarsened adjacency matrix $A ^ { p o o l }$ , and the other is the feature matrix $X ^ { p o o l }$ of the new supernodes. Suppose the graph pooling layer is added behind the i-th convolutional layer, then the pooled results are shown as follows:

$$
A ^ { p o o l } = P ^ { T } A P ; \quad X ^ { p o o l } = P ^ { T } X ^ { i } ,\tag{14}
$$

where $A ^ { p o o l } \in \mathbb { R } ^ { K \times K }$ denotes the pooled adjacent matrix as well as a symmetric matrix. $A _ { j , j } ^ { p o o l }$ indicates the weights sum of all the edges between the nodes in j-th cluster, while $A _ { j , k } ^ { p o o l }$ is the weights sum of the edges between cluster j and k. Xpool is the corresponding feature matrix, whose entries $x _ { j , k } ^ { p o o l }$ in $X ^ { p o o l } \in \mathbb { R } ^ { \hat { K } \times F }$ are the sum of feature k among the elements in cluster j, weighted by the cluster assignment scores.

To avoid the obstacle of self-loops to the propagation in the convolutional layer of GCNN, we compute a normalized adjacency matrix $\tilde { A } ^ { p o o l }$

$$
\hat { A } ^ { p o o l } = A ^ { p o o l } + I _ { K } ; \quad \tilde { A } ^ { p o o l } = \hat { D } ^ { - 1 } \hat { A } ^ { p o o l } .\tag{15}
$$

Âpool is used to add self-loops to the adjacency atrix, and $\hat { D }$ is a diagonal matrix with $\begin{array} { r } { \hat { D } _ { j , j } = \sum _ { k } \tilde { A } _ { j , k } ^ { p o o l } } \end{array}$ . The pooled node features are transformed into the $( i + 1 )$ -th convolutional layer by updating the following operation:

$$
X ^ { i + 1 } = f ( \tilde { A } ^ { p o o l } X ^ { p o o l } T ^ { p o o l } ) .\tag{16}
$$

Tpool denotes the corresponding trainable weight matrix to perform the feature transformation followed by Eq. (1).

Overall, in our proposed method, we assume the initial adjacent matrix is $\bar { A ( \mathbf { \xi } ) } \in ( 0 , 1 ) ^ { N \times N }$ , with equal weight for each edge. We employ the Ollivier-Ricci flow to iteratively update the edge weight matrix according to the geometric structure of the node connections, facilitating the natural distinction between node clusters. Subsequently, motivated by the spectral clustering method, we learn a new clustering assignment matrix. The coarsened adjacency matrix $A ^ { p o o l }$ preserves the global topology structure of the graph, while the cluster embeddings $X ^ { p o o l }$ retain the geometric structure of the attribute features from the previous layer Xi. The overall process is illustrated in Fig. 2 and Algorithm 2.

Our proposed Ricci flow-based clustering method serves as a general technique applicable to solving clustering tasks on any graph-structured data. In this study, we specifically concentrate on graph pooling techniques combined with GCNN to learn high-level representations for graphs. Ricci flow describes the evolution process of non-Euclidean domains and offers strong geometric interpretability. By leveraging the Ollivier-Ricci flow, we can differentiate between clique-like structure subgraphs and tree-like structure subgraphs. These two structures contain embedded semantic information, such as positive or negative node pairs, which can provide additional unsupervised information for downstream tasks and enhance performance.

```latex
Algorithm 2 RicciPool
Input: the initial adjacency matrix $A \in ( 0 , 1 ) ^ { N \times N }$ ，
the feature matrix $X ^ { i }$ of i-th convolutional layer,
the number of clusters K and the iteration value
l.
1. Initialize that $w ^ { 0 } = A , d ^ { 0 } ( x , y ) = w _ { x y } ^ { 0 } .$
2. Iterate the adjacency matrix $w ^ { 0 }$ by Oilivier-Ricci flow
l times following Algorithm 1.
3. Train the loss function Eq. (13) using GCNN and
obtain an assignment matrix $\mathbf { \bar { \Psi } } P \in \mathbb { R } ^ { N \times K }$
4. Calculate $\bar { A } ^ { p o o l }$ and $X ^ { p o o l }$ after pooling using
Eq. (14).
Output: the input adjacency matrix $A ^ { p o o l }$ and feature
matrix $X ^ { p o o l }$ of the $( i + 1 )$ -th convolutional layer.
```

## E. Computational Complexity Analysis

In this section, we delve into the computational complexity of our proposed RicciPool theoretically. As mentioned earlier, assuming the graph has N nodes, K cluster assignments, and an initial feature dimension is F, with $N > K$ The total time cost of our GCNN-based pooling method primarily consists of two parts. The first part involves the computational complexity of the Ollivier-Ricci flow. Let's assume there are l iterations in our Ollivier-Ricci flow operation. In each iteration, computing the Ollivier-Ricci curvature takes a time cost of $\mathcal { O } ( N ^ { \bar { 4 } } \log ^ { 2 } N )$ [33], hence the total computational complexity of the Ollivier-Ricci flow is $\mathcal { O } ( l N ^ { 4 } \log ^ { 2 } N )$ . The second part deals with the computational complexity of GCNN. The time cost of one GCNN layer is quantified by $\mathcal { O } ( N ^ { 3 } + N ^ { 2 } F + N F ^ { 2 } )$ which is approximately $\mathcal { O } ( N ^ { 3 } )$ . Similar to [46],let's suppose the number of iterations in our training network is m, then the total computational cost of this part is $\mathcal { O } ( m N ^ { 3 } )$ . After the pooling operation, the final step involves computing the pooled adjacent matrix $A ^ { p o o l }$ and the associated feature matrix $X ^ { p o o l }$ , which takes about $\mathcal { O } ( N K F + N ^ { 2 } K + N K ^ { 2 } )$ approximately $\mathcal { O } ( N ^ { 3 } )$ Altogether, the total computational complexity of our GCNN-based RicciPool is $\mathcal { O } ( ( l N \log ^ { 2 } N + \bar { m } ) N ^ { 3 } )$

Compared with other GCNN based graph pooling methods, RicciPool uses the Ollivier-Ricci flow to update the edge weight matrix before network training, without changing the architecture, training process, or inference process of the GCNN itself. Therefore, the curvature computation should be regarded as an offline preprocessing cost in the full pipeline, rather than an additional cost inside GCNN training or testing. As discussed in the scalability analysis (detailed in Section V-E and Section V-F), this cost is more manageable on sparse or moderately connected graphs, while it may become expensive on highly dense graphs.

## V. Experiments

## A. Datasets and Experimental Settings

We evaluate proposed RicciPool on seven benchmark datasets, including three bioinformatics datasets: EN-

ZYMES, PROTEINS [3], D&D [10]; four social networks: IMDB-B [42], IMDB-M [43], COLLAB [50], and REDDIT-M-5K [51]. The statistics and properties of these datasets are reported in Table I, following the introduction in [46]. Almost all existing graph pooling methods run on these datasets, as they are relatively large-scale with a significant number of graphs.

We compare our method with three categories of state-of-the-art approaches: 1) Node selection-based methods, including SUMPool [17], SORTPool [48], TOP-KPool [13], SAGPool [17], MID [53], and CCP-GNN [52]; 2) Node clustering-based methods, including DIFF-Pool [45], StructPool [46], MinCutPool [1], and SEP [40]; 3) Geometry-aware methods. Since our method also falls into the geometry-aware category, we further include comparisons with ORC [49], SPREADPool [54], and MAGPool [54] from this category.

Our method was implemented in Python 3.8 and experiments were conducted on a server configured with an Inter(R) Xeon(R) Gold 6246R CPU, 256GB RAM and four NVIDIA GeForce RTX 3090 GPUs. Within this setup, the curvature precomputation was handled by the CPU, while the graph pooling model was trained on the GPUs. Ollivier-Ricci flow computed using open-source tool GraphRicciCurvature¹. The code and experiments are available at https://github.com/cqfei/RicciPool/tree/ master.

## B. Classification Results

To demonstrate the effectiveness of our proposed RicciPool, we compare our method with 13 graph pooling methods on the seven standard datasets shown in Table I within the same network framework, following the protocol outlined in [46]. All pooling methods are applied in the spatial GCNN framework introduced in Section III-A. The comparison results are presented in Table II, with the best results highlighted in bold. Furthermore, for all methods, we conducted 10-fold cross-validations and computed the average accuracy and standard deviation of each dataset. In our RicciPool, the Ollivier-Ricci flow iteration steps l are set to 5. The hyperparameters $\alpha = 0 . 5$ and $\eta \ : = \ : 1 . 0$ are fixed adhere to the recommendations of [26], consistent with all other experiments in this paper. An ablation analysis of these hyperparameters is detailed in Section V-C. To ensure a fair comparison, we reimplemented all baseline methods using the same data split and uniform hyperparameters (learning rate: $1 e - 3 ;$ hidden dimension: 64; maximum training epochs: 150). It should be noted that, due to the high time complexity of the SPREADPOOL and MAGPOOL methods, we were unable to complete their computations on the REDDIT-M-5K dataset within a feasible timeframe using the same computing platform as the other baseline methods, therefore, the corresponding results are not available (marked as ‘-').

TABLE I: Dataset statistics.
<table><tr><td>name</td><td>Category</td><td>Nodes (avg)</td><td>Edges (avg)</td><td>Graphs</td><td>Classes</td></tr><tr><td>ENZYMES</td><td>Bioinformatics</td><td>32.63</td><td>124.20</td><td>600</td><td>6</td></tr><tr><td>PROTEINS</td><td>Bioinformatics</td><td>39.06</td><td>72.82</td><td>1113</td><td>2</td></tr><tr><td>D&amp;D</td><td>Bioinformatics</td><td>284.32</td><td>1431.3</td><td>1178</td><td>2</td></tr><tr><td>IMDB-B</td><td>Social Network</td><td>19.77</td><td>96.53</td><td>1000</td><td>2</td></tr><tr><td>IMDB-M</td><td>Social Network</td><td>13.00</td><td>65.94</td><td>1500</td><td>3</td></tr><tr><td>COLLAB</td><td>Social Network</td><td>74.00</td><td>2457</td><td>5000</td><td>3</td></tr><tr><td>REDDIT-M-5K</td><td>Social Network</td><td>508.52</td><td>594.87</td><td>5000</td><td>5</td></tr></table>

TABLE II: Comparison results between different graph pooling methods under the same framework with the same number of clusters K = 32 and the same Ollivier-Ricci flow iteration value l = 5 for RicciPool. All the comparison results represent ten-run averages.
<table><tr><td rowspan="2">Category</td><td rowspan="2">Method</td><td colspan="7">Dataset</td></tr><tr><td>Enzymes</td><td>D&amp;D</td><td>Proteins</td><td>IMDB-B</td><td>IMDB-M</td><td>Collab</td><td>REDDIT-M-5K</td></tr><tr><td rowspan="6">Node Selection Pooling</td><td>SumPool [17]</td><td>41.33±5.31</td><td>77.26±3.51</td><td>77.21±4.34</td><td>56.90±3.01</td><td>39.80±3.20</td><td>73.46±1.82</td><td>49.15±2.58</td></tr><tr><td>SortPool [48]</td><td>34.67±3.71</td><td>76.75±3.10</td><td>73.96±4.46</td><td>60.70±3.55</td><td>39.87±3.51</td><td>71.04±2.03</td><td>49.63±1.55</td></tr><tr><td>TopkPool [13]</td><td>31.67±3.94</td><td>72.99±3.83</td><td>73.96±4.33</td><td>61.30±2.53</td><td>41.53±3.13</td><td>71.22±1.62</td><td>48.23±2.08</td></tr><tr><td>SagPool [17]</td><td>32.00±3.56</td><td>74.62±2.84</td><td>73.87±4.50</td><td>62.80±4.33</td><td>42.13±3.56</td><td>73.58±1.82</td><td>49.61±1.67</td></tr><tr><td>CCP-GNN [52]</td><td>52.83±3.66</td><td>71.62±4.47</td><td>71.89±5.33</td><td>73.80±2.86</td><td>50.80±2.81</td><td>78.98±2.13</td><td>53.39±2.98</td></tr><tr><td>MID [53]</td><td>41.05±1.88</td><td>77.56±4.46</td><td>75.05±3.18</td><td>73.64±5.53</td><td>51.47±3.22</td><td>80.19±2.39</td><td>56.04±2.87</td></tr><tr><td></td><td>StructPool [46]</td><td>34.67±5.52</td><td>76.32±2.57</td><td>73.24±5.51</td><td>67.40±3.44</td><td>46.47±2.75</td><td>70.44±6.65</td><td>48.07±1.83</td></tr><tr><td rowspan="3">Node Clustering</td><td></td><td>38.67±6.23</td><td>73.24±5.51</td><td>76.94±4.36</td><td>55.80±2.96</td><td>39.20±2.92</td><td>73.26±2.23</td><td>50.15±2.02</td></tr><tr><td>DiffPool [45] MinCutPool [1]</td><td>40.67±5.33</td><td>78.89±2.89</td><td>76.85±4.34</td><td>58.00±5.25</td><td>40.60±4.35</td><td>73.32±1.98</td><td>50.07±1.75</td></tr><tr><td>SEP [40]</td><td>37.00±9.36</td><td>77.35±4.94</td><td>75.50±5.42</td><td>73.60±4.25</td><td>51.53±3.51</td><td>81.40±1.33</td><td>49.99±1.71</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">Geometry Aware</td><td>ORC [49]</td><td>38.50±8.45</td><td>81.62±2.14</td><td>79.01±3.37</td><td>64.50±5.33 75.80±4.07</td><td>43.07±3.59</td><td>73.80±1.98</td><td>51.53±1.82</td></tr><tr><td>SpreadPool [54]</td><td>66.30±7.60 67.50±4.55</td><td>80.60±2.21 80.17±2.81</td><td>76.40±3.03 76.04±2.83</td><td>75.10±3.45</td><td>50.40±2.85 50.60±2.84</td><td>67.72±2.94 67.84±2.87</td><td></td></tr><tr><td>MagPool [54] RicciPool</td><td>62.42±6.46</td><td>82.99±3.23</td><td>79.46±3.82</td><td>76.00±4.53</td><td>51.62±3.42</td><td>75.30±1.73</td><td>55.20±2.02</td></tr></table>

As shown in Table II, RicciPool achieves the best performance among the compared methods on D&D, PRO-TEINS, IMDB-B, and IMDB-M, which demonstrates the effectiveness of our proposed curvature flow pooling strategy. To illustrate the role of the curvature flow, we focus on comparing MinCutPool and StructPool. While these two methods primarily rely on pairwise connections between neighboring nodes, RicciPool can distinguish different substructures based on the Ollivier-Ricci curvature flow, thereby offering stronger geometric interpretability. The consistent improvements over these two pooling methods on all seven datasets suggest that curvature flow provides useful structural information for graph pooling.

Furthermore, in comparisons with existing geometryaware methods, RicciPool achieves better results than existing geometry-aware methods on most datasets, except on ENZYMES. This suggests that refining the node assignment matrix via curvature flow can be beneficial for graph pooling. On the ENZYMES dataset, RicciPool does not achieve the best result among all compared methods, but it still outperforms representative clustering based pooling methods such as MinCutPool and StructPool. This shows that curvature flow remains useful, although its additional gain is not sufficient to surpass the strongest geometry based methods.

On the COLLAB dataset, RicciPool also fails to show a significant improvement. The reason is that this is a highly dense graph (average density ≈ 0.91) where the

Ollivier-Ricci curvature of most edges is positive. This causes the geometric flow to adjust edge weights in a consistent direction, weakening its ability to discriminate between graph structures and limiting the refinement effect of the curvature flow on the node assignment matrix. Unlike COLLAB, REDDIT-M-5K is extremely sparse and close to a tree like structure (average density ≈ 0.0046). In this case, many edges tend to have negative curvature, and the Ricci flow also adjusts edge weights in a consistent direction as opposed to positive curvature, which limits its discriminative effect. These results indicate that both overly dense and overly sparse graphs may weaken the effectiveness of RicciPool. Overall, RicciPool is more suitable for graphs with moderate connectivity and distinct cluster structures of nodes.

## C. Ablation Study and Analysis

In the following, we evaluate RicciPool under various parameter settings and provide a detailed discussion based on a series of comparison experiments.

Effects of graph pooling layer. We conduct an ablation analysis on the graph pooling layer to demonstrate its impact on the final graph classification results. Through a set of experiments, we compare our proposed RicciPool with a baseline method lacking the graph pooling layer, and present the results in Table III. From Table III, it's evident that without the graph pooling layer, the baseline method behaves similarly to traditional GCNNs. Like

TABLE III: Comparison results with the baseline which excludes the graph pooling layer. For RicciPool, the number of clusters is set to $K = 3 2$ and the Ollivier-Ricci flow iteration value is set to $l = 5 ,$ All the comparison results represent ten-run averages.
<table><tr><td>Dataset</td><td>ENZYMES</td><td>D&amp;D</td><td>Proteins</td><td>IMDB-B</td><td>IMDB-M</td><td>COLLAB</td><td>REDDIT-M-5K</td></tr><tr><td>No Pooling</td><td>57.33</td><td>81.42</td><td>78.27</td><td>73.40</td><td>50.53</td><td>75.26</td><td>54.6</td></tr><tr><td>RicciPool</td><td>62.42</td><td>82.99</td><td>79.46</td><td>76.00</td><td>51.62</td><td>75.30</td><td>55.2</td></tr></table>

TABLE IV: The prediction accuracy of RicciPool under different Ollivier-Ricci flow iterations l with the same number of clusters $K = 3 2$
<table><tr><td rowspan="2">Dataset</td><td colspan="6">Iteration</td></tr><tr><td> $l = 0$ </td><td> $l = 1$ </td><td> $l = 2$ </td><td> $l = 3$ </td><td> $l = 4$ </td><td> $l = 5$ </td></tr><tr><td>D&amp;D</td><td>81.47</td><td>82.39</td><td>81.47</td><td> $_ { 8 2 . 3 1 }$ </td><td>82.39</td><td>82.99</td></tr><tr><td>Proteins</td><td>76.83</td><td>79.01</td><td>79.46</td><td>79.82</td><td>79.28</td><td>79.46</td></tr><tr><td>IMDB-B</td><td>72.74</td><td>75.60</td><td>76.40</td><td>76.20</td><td>75.80</td><td>76.00</td></tr></table>

TABLE V: Curvature convergence statistic of the Ollivier-Ricci flow.
<table><tr><td>Dataset</td><td>ENZYMES</td><td>Proteins</td><td>IMDB-B</td><td>IMDB-M</td></tr><tr><td>Expectation</td><td>11</td><td>10</td><td>6</td><td>6</td></tr><tr><td>Proportion</td><td>60.67</td><td>59.03</td><td>72.6</td><td>43.53</td></tr></table>

RicciPool, the baseline method adheres to the network framework described in Section III-A and undergoes 10- fold cross-validation to compute the average accuracy for each graph dataset. Notably, our proposed method outperforms the baseline across all 7 datasets, indicating that the graph pooling operation contributes to extracting useful features among nodes and facilitating effective graph-level embedding representation.

Effects of Ricci flow iterations l. We analyze how the number of Ricci flow iterations l affects prediction performance on D&D, PROTEINS, and IMDB-B, with the results shown in Table IV and Fig. 3. Compared with $l = 0 .$ where no Ricci flow based edge reweighting is applied, using $l > 0$ generally improves classification performance, indicating that Ollivier Ricci flow provides useful local structural information for graph pooling. As l increases, positive curvature edges are gradually shortened and negative curvature edges are stretched, making dense substructures more compact and sparse or bridge like connections easier to separate. However, the optimal iteration number varies across datasets, with $l = 5$ for D&D, l = 3 for PROTEINS, and l = 2 for IMDB-B. This suggests that different datasets require different degrees of curvature flow refinement. However, excessive iterations may over adjust edge weights and weaken useful topology for node assignment learning.

To further explain this phenomenon, we further investigate the correlation between curvature convergence and optimal performance in RicciPool. We analyzed the convergence properties of Ollivier-Ricci flow on four benchmark datasets, with convergence defined as a difference between maximum and minimum edge curvature below $1 0 ^ { - 3 }$ . Statistical results for curvature convergence iterations are summarized in Table V. Two trends can be observed.

![](images/87d3a202d5f5294276510d582b97e7afd07454cba954aed5b31455fa0ccd6e2f.jpg)  
Fig. 3: The prediction accuracy of RicciPool under different Ollivier-Ricci flow iterations l with the same number of clusters $K = 3 2$

First, larger graphs usually require more iterations to reach curvature convergence. For example, PROTEINS needs about 10 iterations, compared with about 6 for IMDB-B. This is because larger graphs often contain more diverse local connection patterns and more types of edges such as dense intra community edges, sparse branch edges, and bridge edges. These different edge types evolve at different speeds under Ricci flow, so more iterations are needed for the curvature values and the induced edge weights to stabilize.

Second, datasets with slower curvature convergence tend to require more RicciPool iterations to reach the best classification performance. For example, PROTEINS converges more slowly and peaks at $l = 3$ , whereas IMDB-B converges faster and peaks at $l \ = \ 2$ . This suggests that curvature convergence speed reflects the complexity of geometric regularization.

Third, the optimal iteration number for classification is usually smaller than the full convergence iteration number. For example, after PROTEINS peaks at l = 3 and IMDB-B peaks at $l = 2 .$ further increasing l does not bring additional gains. RicciPool only needs moderate curvature evolution to provide sufficient structural separation for node clustering. Excessive iterations may over contract positive curvature edges and over stretch negative curvature edges, which can distort useful graph connectivity and accelerate over squashing and over smoothing during subsequent GNN propagation. Therefore, the iteration number should balance structural discrimination and graph connectivity preservation.

Effects of graph pooling rate r. In this segment, we examine how the number of clusters K influences RicciPool by setting a pooling rate $r \in ( 0 , 1 )$ to control K $( r = 1$ means $K = 6 4 )$ . We adjust r to $\{ 0 . 2 5 , 0 . 5 0 , 0 . 7 5 \}$ to evaluate the sensitivity of graph pooling methods to the pooling ratio. Figure 4 compares the performance of RicciPool, ORC (also a Ricci Flow-based method), and MinCutPool under different pooling ratios. The results show that RicciPool maintains the highest average classification accuracy across all ratios, validating the effectiveness of its core mechanism. It is worth noting that RicciPool exhibits a relatively wide performance distribution, which reflects the sensitivity of its curvature-based global reweighting mechanism to graph structural features. Further comparison reveals that ORC, another geometry-aware method, outperforms MinCut on average. However, ORC shows a larger performance decrease at the high pooling ratio (0.75) and exhibits higher variability than RicciPool in our experiments, which may suggest that it is more sensitive to structural compression under these settings. In contrast, MinCut shows narrower confidence intervals and more stable results, but its average performance is lower than RicciPool. This suggests that MinCut is stable but may be less effective in extracting discriminative structural information under different pooling ratios. In summary, despite some performance variability, RicciPool achieves competitive and generally better performance across different pooling ratios, supporting the advantage of its curvature-driven mechanism in extracting key structural information. Compared to ORC, RicciPool shows less performance degradation at higher compression ratios in our experiments.

![](images/0e57d13e2850bf4e722f41a09a0c1ed2c448ebef7b7e910cb6beec48f607677e.jpg)

![](images/751f0c64999e6f1d504ff71d0b5261a40e689e929a4d21f100eeb3c4ada3f6aa.jpg)  
Fig. 4: Comparison under different pooling rates with the same Ollivier-Ricci flow iteration $l = 5$ (left: IMDB-B, right: PROTEINS).

![](images/d0814304e27ec25f29ca9b1297f9a1af8b1db9ac1b124a60c131a42403539b6b.jpg)  
(a)

![](images/ba0ca91f204d26101ea6b9cd1e1e4a47c0f85f8d72580b910dd50c05a55d2a52.jpg)  
(b)  
Fig. 5: Performance on different η (left) and α (right) values for dataset IMDB-B and PROTEINS.

Effects of curvature intervention and neighbor information. We introduced the hyperparameter η (ranging from 0 to 1) to observe the impact of curvature information on our algorithm. The experimental results are shown in Fig. 5a. It can be seen that for the IMDB-B dataset, our method achieved optimal performance at $\eta = 0 . 7$ and $\eta = 0 . 6$ when $\alpha = 0 . 1$ and $\alpha = 0 . 5$ , respectively. For the PROTEINS dataset, the best performance was achieved at $\eta = 0 . 9$ and $\eta \ : = \ : 1 . 0$ Based on this, we draw two conclusions: 1) Incorporating curvature information is indeed effective $( \eta = 0$ represents results without curvature information); 2) The optimal level of curvature intervention (η) differs between the IMDB-B and PROTEINS datasets. This suggests that the impact of curvature information may be dataset-dependent, and tuning this parameter could be important for optimal performance on a given dataset. One influencing factor, according to our analysis, is the connectivity of the dataset itself. Structural analysis shows that the IMDB-B dataset is denser than the PROTEINS dataset (the average degree of IMDB-B is higher than that of PROTEINS). Consequently, optimal results for IMDB-B were obtained with less curvature information $( \eta = 0 . 7$ and $\eta = 0 . 6 )$ , whereas the PROTEINS dataset required more curvature intervention $( \eta \ : = \ : 0 . 9$ and $\eta \ : = \ : 1 . 0 )$ to achieve the best results

Additionally, when computing graph curvature, the hyperparameter α controls the weight distribution between the source and target nodes' probability distributions in the calculation of Wasserstein distance. A smaller α means that the source node's neighborhood has a greater influence on the calculation, while the target node's neighborhood has a lesser influence. The experiments regarding this hyperparameter are shown in Fig. 5b. From Fig. 5b, it can be observed that for the IMDB-B dataset, the best performance was achieved at $\alpha = 0 . 2$ for both $\eta = 0 . 6$ and $\eta = 1 . 0$ . For the PROTEINS dataset, the optimal results were obtained at $\alpha = 0 . 3$ and $\alpha = 0 . 4$ Based on this, we draw another conclusion: the influence of neighborhood information varies across different datasets. Both the source and target nodes' neighborhood information affect edge curvature and indirectly influence the final performance of our algorithm.

A trick for hyperparameter tuning. In the RicciPool method, we introduced three novel Ricci flow-related hyperparameters. Based on extensive experiments, we concluded that optimal performance is achieved when α ranges from 0.1 to 0.5, η ranges from 0.5 to 1.0, and iterations their optimal ranges: $\alpha \in [ 0 . 1 , 0 . 5 ] , \eta \in [ 0 . 5 , 1 . 0 ]$ and iteration times generally not exceeding 5.

## D. Analysis of Cluster Assignment in RicciPool

Visualization of graphs in RicciPool. We examined the extent to which RicciPool learns meaningful node clusters by visualizing the node distribution after the Ollivier-Ricci flow, similar to [45]. Fig. 6 illustrates this visualization of node assignments in the RicciPool layer on three graphs from the ENZYMES and PROTEINS datasets. Notably, we observe distinct node cluster memberships across different datasets. RicciPool effectively exploits graph curvature to uncover embedded cluster structures, shrinking the edge weight of fully connected subgraphs and extending the edge distance of tree-structured subgraphs. Graphs A, B, and C represent three types of graphs with varying connection strengths: Graph A has a tree-like structure, Graph C has a nearly fully connected structure, and the connectivity of Graph B is intermediate between A and C. We can observe the different changes in the subgraph structures of these graphs after the Ollivier-Ricci flow.

![](images/7069a59d95de4d0a630d44dcbba6d52bda8f2b5c201f69a5f6f36dc91555fc69.jpg)  
(g) l = 0  
(h) l = 1  
(i) l = 2  
Fig. 6: Visualization of the Ollivier-Ricci flow in RicciPool, using three real example graphs from the ENZYMES (first row, ID 351, referred to as graph A below) and PROTEINS datasets (IDs 1084 and 412, referred to as graphs B and C below). Figures a-c, d-f, and g-i show the states of graphs A, B, and C after 0, 1, and 2 iterations of the Ollivier-Ricci curvature flow, with the numbers on the edges indicating the distances between nodes. The subgraphs with different background colors represent different communities. The lengths of the edges represent the edge weights. The black lines on the edges indicate positive curvature, while the green lines indicate negative curvature.

Subgraph structures. We illustrate the edge lengths corresponding to the edge weights. Fig. 6 depicts the evolving node distributions of the graphs under the Ollivier-Ricci flow. The visualization demonstrates that Ollivier-Ricci curvature effectively measures pairwise connections between neighboring nodes and the density of subgraphs. The flow transforms dense subgraphs into even denser ones (see graphs A and B), while extending the edge lengths of sparse subgraphs. Moreover, Ollivier-Ricci curvature distinguishes various subgraph structures, including fully connected (see graph C) and tree-like formations (see graph A). For instance, in Fig. 6 (g-i), graph C exhibits quasi-full connectivity, with node distances decreasing post Ollivier-Ricci flow evolution, thereby tightening connections. These findings underscore the strong geometric nature and interpretability of Ollivier-Ricci curvature.

Sensitivity of the Number of Clusters. We observed that cluster assignment varies across different graph datasets and with K, the number of clusters, indicating sensitivity to the pooling result. For the Proteins dataset, the optimal K varies depending on the specific graphs. Setting K too large may fragment densely connected subgraphs into separate clusters, leading to a loss of closely connected edges between neighboring nodes. Conversely, if K is too small, the resulting pooled graph may have excessively small node sizes, failing to preserve the global topology of the original graph. In the case of the IMDB-M dataset, where the average number of nodes per graph is 13 and the average number of edges is 66, nearly all graphs are fully connected. Setting K too large can disrupt the fully connected structure. This explains why the IMDB-M dataset shows poorer classification results under graph pooling when K > 1, compared to the other four datasets presented in Table II.

## E. Scalability and Efficiency of RicciPool on Large-Scale Graph Datasets

This subsection evaluates the scalability and computational efficiency of RicciPool on COLLAB and REDDIT-M-5K. These two datasets have different graph sizes and densities. COLLAB is highly dense, with an average density of approximately 0.91, while REDDIT-M-5K contains larger but much sparser graphs, with an average density of approximately 0.0046. This contrast allows us to examine how graph size and graph density affect the computational overhead of RicciPool, especially the offline curvature precomputation cost.

![](images/f3ca04210cec71a42285e2a16f2929a18066cab3c79d8a7ba653c7aee8f62a14.jpg)  
Fig. 7: Wall-clock time allocation on Random Regular Graph. P: curvature precomputation phase, T: GCNN training phase.

![](images/970f69a7db93f4b09167eda79e7cdb4f9ace52e887554e7f92078e0fe41816b6.jpg)  
Fig. 8: Memory allocation on Random Regular Graph. P: curvature precomputation phase, T: GCNN training phase.

Table VI further presents the memory usage and wallclock time for the curvature precomputation and model training phases on these two large graph datasets. It is noteworthy that on the REDDIT-M-5K dataset, which has a larger number of nodes, the curvature precomputation time is much shorter than the model training time. In contrast, on the relatively denser COLLAB dataset, the curvature precomputation time exceeds the training time. This indicates that curvature precomputation and model training respond differently to graph scale (including the number of nodes and density). To investigate this relationship further, we conducted controlled experiments on random regular graphs.

We systematically varied the graph scale by adjusting the number of nodes {120, 180, 240, 300} and the node degree {5, 10, 15, 20, 25, 30, 45, 50, 60}. For each combination of number of node and node degree, we generated 500 graphs, and recorded the resource consumption for each during both the curvature precomputation and model training phases.

The experimental results (Fig. 7 and Fig. 8) show that as the number of nodes increases, the time required for both curvature precomputation and model training increases. However, for a fixed number of nodes, as the graph density (node degree) increases, the curvature computation time exhibits an approximate polynomial growth, while the training time increases only linearly. For instance, in a graph with 120 nodes, when the node degree exceeds 60 (density ≈ 0.5), the curvature precomputation time surpasses the model training time. This indicates that in large-scale, high density graph scenarios, the complexity of curvature precomputation may limit the scalability of algorithm.

![](images/7631f2e1a243ff375eadbfaa337863901d0aa48fb685621eb3e4683bbcb2aba1.jpg)  
Fig. 9: Costs on curvature precomputation and GCNN training.

This scalability limitation is mainly related to graph density. Curvature precomputation becomes more expensive on dense graphs, because more local neighborhood interactions need to be considered during Ollivier Ricci flow. In contrast, for graphs with moderate connectivity, the offline curvature precomputation cost is usually more manageable. In our experiments, the precomputation time is generally lower than the model training time within this range. Therefore, the extra computational cost of RicciPool mainly comes from the offline curvature precomputation stage and is sensitive to graph density.

## F. Computational Efficiency Analysis

The computational efficiency challenges inherent in Ollivier-Ricci curvature motivated our design of an optimization mechanism: the offline curvature computation and online pooling decoupling framework. RicciPool adopts a two phase computational paradigm. Precomputation Phase: Discrete Ricci curvature flow iterations are executed to extract curvature features, which is a process entirely independent of the neural network training pipeline. Pooling Phase: Hierarchical pooling is performed based on precomputed curvature features, exhibiting computational cost comparable to standard graph convolutional layers significantly lower than online curvature optimization approaches.

We conducted a comparative analysis between curvature precomputation cost and GCNN training duration. As demonstrated in Fig. 9, for the four datasets considered in this comparison, the curvature precomputation time is lower than the GCNN training time. Taking the PROTEINS dataset as an example, the curvature precomputation time constitutes only 6.25% of the total training time (91.0 seconds vs. 1455.8 seconds).

Table VI compares the wall-clock time and peak memory usage between the curvature precomputation phase and the model training phase across the experimental datasets. In terms of memory consumption, the curvature precomputation stage requires less memory than the training stage on all datasets, and this memory usage is independent of the number of Ricci flow iterations. Regarding time cost, the training time significantly exceeds the curvature precomputation time on most datasets. COLLAB is an exception, mainly because its high graph density makes curvature computation over local neighborhoods much more expensive. This result is consistent with the scalability analysis, showing that graph density has a stronger impact on curvature precomputation than on GCNN training.

Overall, the extra computational cost of RicciPool mainly comes from the offline curvature precomputation stage, which is sensitive to graph density. The GCNN training and inference processes remain unchanged, since curvature information is computed before model training and then used to guide graph pooling.

## G. Geometric Interpretability of RicciPool

We systematically elucidate the mechanistic interpretability of RicciPool through two dimensions: simulation validation and case studies. Simulation experiments explain the relationship between graph curvature and clustering, while real-world case studies demonstrate curvature's impact on clustering outcomes.

![](images/f892be9b37ed13c75ffd87df2f19d448e244fb0de04c46246a9b6cdd68704195.jpg)  
Fig. 10: Ollivier-Ricci curvature distribution in different clustering phases. intra-positive: proportion of intracluster edges with positive curvature, intra-negative: proportion of intra-cluster edges with negative curvature, inter-positive: proportion of inter-cluster edges with positive curvature, inter-negative: proportion of inter-cluster edges with negative curvature.

Šimulation Experiments. We simulate the clustering process on a Stochastic Block Model (SBM) to observe changes in Ollivier-Ricci curvature distribution under different clustering states. SBM model has two key parameters: p is the intra-cluster connection probability to control cluster tightness and q is inter-cluster connection probability (controls separation between clusters). Therefore, we can simulate this process by adjusting the values of p and q.

Clustering Simulation: The clustering process is simulated by gradually increasing p (from 0.05 loose to 0.40 tight) while decreasing q (from 0.05 to 0.0). Initial $q = p$ implies indistinguishable intra/inter-cluster connections at the initial clustering stage.

TABLE VI: Wall-clock Time (s) and Memory (GB) Comparison for Training and Precomputation Phase
<table><tr><td></td><td colspan="3">Training</td><td colspan="2">Precomputation (1=1)</td><td colspan="2">Precomputation (1=2)</td><td colspan="2">Precomputation (1=3)</td></tr><tr><td>Dataset</td><td>Wall-clock Time</td><td>Memory</td><td>GPU Memory</td><td>Wall-clock Time Memory</td><td></td><td>Wall-clock Time</td><td>Memory</td><td>Wall-clock Time</td><td>Memory</td></tr><tr><td>ENZYMES</td><td>928.95</td><td>5.1146</td><td>2.044</td><td>33.67</td><td>0.4442</td><td>69.10</td><td>0.4387</td><td>105.68</td><td>0.4473</td></tr><tr><td>D&amp;D</td><td>30848.21</td><td>5.4996</td><td>7.528</td><td>478.58</td><td>2.0508</td><td>944.91</td><td>2.0306</td><td>1435.05</td><td>2.0090</td></tr><tr><td>PROTEINS</td><td>1964.74</td><td>5.1793</td><td>2.090</td><td>73.84</td><td>0.4865</td><td>145.87</td><td>0.4973</td><td>226.66</td><td>0.4866</td></tr><tr><td>IMDB-B</td><td>1416.51</td><td>5.1338</td><td>2.012</td><td>81.03</td><td>0.4967</td><td>146.96</td><td>0.5027</td><td>217.90</td><td>0.5023</td></tr><tr><td>IMDB-M</td><td>1963.49</td><td>5.1116</td><td>2.010</td><td>107.01</td><td>0.5074</td><td>156.90</td><td>0.5057</td><td>220.96</td><td>0.5051</td></tr><tr><td>COLLAB</td><td>16988.21</td><td>9.1408</td><td>2.554</td><td>24803.02</td><td>3.6249</td><td>25370.60</td><td>3.6138</td><td>37287.22</td><td>3.6086</td></tr><tr><td>REDDIT-M-5K</td><td>505102.92</td><td>7.5728</td><td>14.086</td><td>19493.68</td><td>13.8133</td><td>39442.66</td><td>13.9213</td><td>59093.87</td><td>13.9824</td></tr></table>

Curvature Distribution Findings (shown in Fig. 10): for intra-cluster edges, the proportion of positive-curvature edges increases as clustering progresses, the proportion of negative-curvature edges decreases, and finally converges to 100% positive curvature. For inter-cluster edges, the proportion of positive-curvature edges increases initially then decreases; the proportion of negative-curvature edges decreases initially, then increases, and finally converges to 100% negative curvature. Therefore, our simulation suggests a clear relationship between the sign of curvature and cluster cohesion: positive curvature becomes predominant within clusters, while negative curvature becomes predominant between clusters as separation increases. The prevalence of negative-curvature edges between clusters in well-separated SBM graphs is consistent with the role of Ricci flow in refining cluster boundaries, as observed in our graph pooling experiments.

Case Study. We analyze real graphs to observe curvature-driven changes in edge weights, shown in Fig. 6. Edge weights (red) and curvature values (blue) are annotated. We can clearly see that positive curvature can promote tighter connections for intra-cluster edges (taking edge (6, 7) in the ENZYMES graph 351 as an example, the distances after curvature flow iterations 1 and 2 are 0.5 and 0.21, respectively). Negative curvature can promote looser connections for inter-cluster edges (taking edge 〈1, 2〉 in the PROTEINS graph 1084 as an example, the distances after curvature flow iterations 1 and 2 are 1.33 and 1.85, respectively).

## H. Algorithm Stability and Sensitivity Analysis

To evaluate the stability and sensitivity of the curvature flow-based graph pooling method, experiments were conducted on the ENZYMES and IMDB-B datasets. The maximum number of training epochs was set to 300 for ENZYMES and 500 for IMDB-B (show in Appendix A), while the number of curvature flow iterations was varied as {1, 2, 3, 4}. Fig. 11 illustrates the evolution of validation loss and accuracy during training. As observed, the loss curves decrease steadily under different iteration numbers indicating that RicciPool maintains stable optimization when the iteration number changes within a moderate range.

We further examine the effect of the stopping threshold. The threshold is defined by the mean curvature change and is varied in {0.001,0.005, 0.01,0.05}. The results, presented in Fig. 12, indicate that accuracy remains largely stable, exhibiting only minor fluctuations and forming a clear plateau across a broad intermediate range of threshold values. One possible reason is that RicciPool does not require exact curvature convergence. Once the curvature flow has provided sufficient structural separation for node clustering, small changes in the stopping threshold have limited influence on the relative edge weight patterns and the learned cluster assignment matrix. Therefore, a threshold selected from this stable range can provide reliable performance while avoiding unnecessary curvature iterations.

![](images/067b123aeee8bcb5ddbb013f1e17c8ab91c7384425c57506d3240c28e2c0aa75.jpg)

Fig. 11: Trends in validation loss and accuracy in different ricci flow iteration l on ENZYMES dataset.  
![](images/458a85fd456f5c2d712ea01884c424e7cf68088774b1eb76c35f64d29d8c96b9.jpg)  
Fig. 12: Trends in validation loss and accuracy in different stop threshold δ on ENZYMES dataset.

## VI. Conclusion

Graph pooling operations are essential for managing large-scale graph-structured datasets. Recently, several advanced methods have been proposed, yet they often overlook higher-order spatial connections among neighboring nodes, focusing primarily on rough graph topology.

Addressing these limitations, we introduce a novel graph pooling approach using discrete graph Ricci flow. Our method treats graph pooling as a node clustering task, leveraging Ollivier-Ricci curvature to distinguish wellconnected subgraphs from tree-like structures. We employ spectral clustering theory to learn a cluster assignment matrix, ensuring that the coarse graph after pooling preserves the original global topology. Experiments on benchmark datasets demonstrate the effectiveness and competitiveness of RicciPool.

Furthermore, our method sheds light on specific phenomena. For instance, in fully connected graphs where all edge weights and Ollivier-Ricci curvatures are uniform, RicciPool cannot uncover a more compact latent structure, limiting its ability to enhance graph representations for high-level tasks. While clustering nodes in a fully connected graph may seem impractical due to uniform importance scores, our method effectively uses Ollivier-Ricci curvature to identify these structures. Thus, our approach provides robust geometric interpretability, offering theoretical insights into these observations.

In the future, we will continue to explore the application potential of geometric flow within GNNs. For instance, Ricci flow can automatically identify the clustering structure of graph node sets. A key question then arises: how can geometric flow be leveraged to autonomously determine the optimal number of clusters? Furthermore the evolution of geometric flows on graph data exhibits parallels with GNN dynamics: node clustering resembles the GNN over-smoothing phenomenon, while the interclass separation driven by geometric flow evolution mirrors the GNN over-squashing process. Consequently, our subsequent work will focus on an in-depth investigation of the relationship between the Ricci flow evolution process and these two specific GNN phenomena, aiming to establish a more robust theoretical explanation.

## Appendix

## A. Stability and Sensitivity for IMDB-B

Fig. 13 and Fig. 14 present the experimental results on the stability and sensitivity of RicciPool on the IMDB-B dataset. The results indicate that, relative to the ENZYMES dataset, IMDB-B exhibits greater stability in its loss and accuracy convergence curves across different numbers of curvature flow iterations. Furthermore, its training trajectory shows marked insensitivity to variations in the stopping threshold.

## References

[1] Bianchi, F. M., Grattarola, D., and Alippi, C.. Spectral clustering with graph neural networks for graph pooling. In International Conference on Machine Learning, 2020.

[2] Bonciocat, A. I., and Sturm, K. T.. Mass transportation and rough curvature bounds for discrete spaces. Journal of Functional Analysis, 256(9):2944–2966, 2009.

[3] Borgwardt, K.M., Ong, C. S., Schonauer, S., Vishwanathan, S. V. N., Smola, A. J., and Kriegel, H. P.. Protein function prediction via graph kernels. Bioinformatics, 21(suppl-1): i47- i56, 2005.

![](images/9e86dc5d6cce0bc6827c1abc720d18621ecc6de7ab303b0bdab0b0d63942d662.jpg)  
Fig. 13: Trends in validation loss and accuracy in different ricci flow iteration l on IMDB-B dataset.

![](images/cb9bb6f11aa5a27cde41413f38b71332063c938851eec4eacabf4329c7989c8a.jpg)  
Fig. 14: Trends in validation loss and accuracy in different stop threshold δ on ENZYMES dataset.

[4] Bronstein, M. M., Bruna, J., LeCun, Y., Szlam, A., and Vandergheynst, P.. Geometric deep learning: going beyond euclidean data. IEEE Signal Processing Magazine, 34(4):18–42, 2017.

[5] Bruna, J., Zaremba, W., Szlam, A., and LeCun, Y.. Spectral networks and locally connected networks on graphs. arXiv preprint arXiv:1312.6203, 2013.

[6] Cai, H., Zheng, V. W., and Chang, K. C. C.. A comprehensive survey of graph embedding: problems, techniques, and applications. IEEE Transactions on Knowledge and Data Engineering, 30, 9, pp. 1616-1637, 2018.

[7] Z.Wu, S. Pan, F. Chen, G. Long, C. Zhang and P. S. Yu. A Comprehensive Survey on Graph Neural Networks. In IEEE Transactions on Neural Networks and Learning Systems, 32(1):4-24, 2021.

[8] Cai, L., and Ji, S.. A multi-scale approach for graph link prediction. In Thirty-Fourth AAAI Conference on Artificial Intelligence, 2020.

[9] Defferrard, M., Bresson, X., and Vandergheynst, P.. Convolutional neural networks on graphs with fast localized spectral filtering. In Advances in Neural Information Processing Systems, pp. 3844–3852, 2016.

[10] Dobson, P. D., and Doig, A. J.. Distinguishing enzyme structures from non-enzymes without alignments Journal of molecular biology, 330(4):771-783, 2003.

[11] Forman, R.. Bochner's method for cell complexes and combinatorial ricci curvature. Discrete and Computational Geometry, 29(3):323–374, 2003.

[12] Gao, H., and Ji, S.. Graph representation learning via hard and channel-wise attention networks. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 741–749, 2019.

[13] Gao, H., and Ji, S.. Graph u-nets. In International Conference on Machine Learning, pp. 2083–2092, 2019.

[14] Hamilton, W., Ying, Z., and Leskovec, J.. Inductive representation learning on large graphs. In Advances in Neural Information Processing Systems, 2017.

[15] Hamilton, R. S. Three-manifolds with positive ricci curvature. Journal of Differential Geometry, 17(2):255–306, 1982.

[16] Henaff, M., Bruna, J., and LeCun, Y.. Deep convolutional networks on graph-structured data. arXiv preprint arXiv:1506.05163, 2015.

[17] Lee, J., Lee, I., and Kang, J.. Self-attention graph pooling. In Proceedings of the 36th International Conference on Machine learning (ICML-19), 2019.

[18] D. Grattarola, D. Zambon, F. M. Bianchi and C. Alippi. Understanding Pooling in Graph Neural Networks. In IEEE Transactions on Neural Networks and Learning Systems, 35(2):2708- 2718, 2024.

[19] Y.Wang, D. Chang, Z. Fu and Y. Zhao. Seeing All From a Few: Nodes Selection Using Graph Pooling for Graph Clustering. In IEEE Transactions on Neural Networks and Learning Systems, 35(5):7231-7237, 2024.

[20] F. M. Bianchi, D. Grattarola, L. Livi and C. Alippi. Hierarchical Representation Learning in Graph Neural Networks With Node Decimation Pooling. In IEEE Transactions on Neural Networks and Learning Systems, 33(5):2195-2207, 2022.

[21] Li, H., Cao, J., Zhu, J., Liu, Y., Zhu, Q., and Wu, G.. Curvature graph neural network. Information Sciences, 592:50–66, 2022.

[22] Lin, Y., Lu, L., and Yau, S. T.. Ricci curvature of graphs. Tohoku Mathematical Journal, Second Series, 63(4):605–627, 2011.

[23] Ma, Y., Wang, S., Aggarwal, C. C., and Tang, J.. Graph convolutional networks with eigenpooling. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp:723-731, 2019.

[24] Monti, F., Boscaini, D., Masci, J., Rodola, E., Svoboda, J., and Bronstein, M. M.. Geometric deep learning on graphs and manifolds using mixture model cnns. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 5115–5124, 2017.

[25] Ni, C.-C., Lin, Y.-Y., Gao, J., and Gu, X.. Network alignment by discrete ollivier-ricci flow. In International Symposium on Graph Drawing and Network Visualization, pp. 447-462. Springer, 2018.

[26] Ni, C.-C., Lin, Y.-Y., Luo, F., and Gao, J.. Community detection on networks with ricci flow. Scientific Reports, 9(1):1–12, 2019.

[27] Ollivier, Y.. Ricci curvature of markov chains on metric spaces. Journal of Functional Analysis, 256(3):810–864, 2009.

[28] Ollivier, Y.. A survey of ricci curvature for metric spaces and markov chains. In Probabilistic Approach to Geometry, 343-381, 2010.

[29] Samal, A., Sreejith, R. P., Gu, J., Liu, S., Saucan, E., and Jost, J.. Comparative analysis of two discretizations of Ricci curvature for complex networks. Sci. Rep. 8, 8650, 2018.

[30] Sandhu, R., Georgiou, T., Reznik, E., Zhu, L., Kolesov, I., Senbabaoglu, Y., and Tannenbaum, A.. Graph curvature for differentiating cancer networks. Sci. Rep. 5, 12323, 2015.

[31] Shi. Multiclass spectral clustering. In Proceedings Ninth IEEE International Conference on Computer Vision, pp.313-319 vol.1, Oct 2003.

[32] Sia, J., Jonckheere, E., and Bogdan, P.. Ollivier-ricci curvaturebased method to community detection in complex networks. Scientific Reports, 9(1):1–12, 2019.

[33] Siddharth P., Feng Y., Terrence J., Ram R., Amotz B. and Ananthram S.. An efficient alternative to Ollivier-Ricci curvature based on the Jaccard metric. arXiv:1710.01724v1, 2021.

[34] Sreejith, R. P., Mohanraj, K., Jost, J., Saucan, E. and Samal, A.. Forman curvature for complex networks. J. Stat. Mech: Theory Exp. 2016, 063206, 2016.

[35] Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., and Bengio, Y.. Graph attention networks. In International Conference on Learning Representations, 2018.

[36] Wang, Y. G., Li, M., Ma, Z., Montufar, G., Zhuang, X., and Fan, Y.. Haar graph pooling. In International Conference on Machine Learning, 2020.

[37] Weber, M., Jost, J., and Saucan, E.. Detecting the coarse geometry of networks. In NeurIPS 2018 Workshop, 2018.

[38] Weber, M., Jost, J., and Saucan, E.. Forman-ricci flow for change detection in large dynamic data sets. Axioms, 5(4):26, 2016.

[39] Weber, M., Saucan, E., and Jost, J.. Characterizing complex networks with forman-ricci curvature and associated geometric flows. Journal of Complex Networks, 5(4):527–550, 2017.

[40] Wu, J., Chen, X., Xu, K., and Li, S.. Structural entropy guided graph hierarchical pooling.In International conference on machine learning. ICML, 2022: 24017-24030.

[41] Xu, K., Hu, W., Leskovec, J., and Jegelka, S.. How powerful are graph neural networks? In International Conference on Learning Representations, 2019.

[42] Yanardag, P., and Vishwanathan, S. V. N.. A structural smoothing framework for robust graph comparison. In Advances in Neural Information Processing Systems, pp.2134-2142, 2015a.

[43] Yanardag, P., and Vishwanathan, S. V. N.. Deep graph kernels. In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp.1365-1374, ACM, 2015b.

[44] Ye, Z., Liu, K. S., Ma, T., Gao, J., and Chen, C. Curvature graph network. In International Conference on Learning Representations, 2019.

[45] Ying, Z., You, J., Morris, C., Ren, X., Hamilton, W., and Leskovec, J.. Hierarchical graph representation learning with differentiable pooling. In Advances in Neural Information Processing Systems, pp. 4800–4810, 2018.

[46] Yuan, H., and Ji, S.. Structpool: Structured graph pooling via conditional random fields. In Proceedings of the 8th International Conference on Learning Representations, 2020.

[47] Zhang, M., and Chen, Y.. Link prediction based on graph neural networks. In Advances in Neural Information Processing Systems, pp. 5165–5175, 2018.

[48] Zhang, M., Cui, Z., Neumann, M., and Chen, Y.. An end-to-end deep learning architecture for graph classification. In AAAI, pp. 4438–4445, 2018.

[49] Amy F and Melanie W. Graph Pooling via Ricci Flow. Transactions on Machine Learning Research, 2024.

[50] P. Yanardag and S. V. N. Vishwanathan. A structural smoothing framework for robust graph comparison. In Advances in Neural Information Processing Systems, 28:2134–2142, 2015.

[51] Yanardag, Pinar and Vishwanathan, S.V.N. Deep Graph Kernels. In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 10:1365– 1374, 2015.

[52] P. Zhu, J. Li, Z. Dong, Q. Hu, X. Wang and Q. Wang. CCP-GNN: Competitive Covariance Pooling for Improving Graph Neural Networks. In IEEE Transactions on Neural Networks and Learning Systems, 36(4):6395-6406, 2025.

[53] C. Liu, Y. Zhan, B. Yu, L. Liu, B. Du, W. Hu, T. Liu. On exploring node-feature and graph-structure diversities for node drop graph pooling. In Neural Networks, 167:559-571, 2023.

[54] K. Limbeck, L. Mezrag, G. Wolf, B. Rieck. Geometry-Aware Edge Pooling for Graph Neural Networks. Advances in Neural Information Processing Systems, 2025.