# HyBridge: Bridge-Guided Self-Supervised Hyperbolic Hypergraph Representation Learning

So-Bin Jung

David Yoon Suk Kang

Sang-Wook Kim

Hanyang University

Chungbuk National University

Hanyang University

Seoul, South Korea

sobin98@hanyang.ac.kr

Cheongju, South Korea

Seoul, South Korea

dyskang@cbnu.ac.kr

wook@hanyang.ac.kr

Abstract—Hypergraphs capture groupwise relationships among multiple nodes and often exhibit tree-like structures. Although hyperbolic space is well suited for modeling such structures, simply extending existing hypergraph representation learning (HRL) methods to hyperbolic space does not consistently improve performance. In this paper, we investigate the relationship between node structural properties and hyperbolic embedding radii and find that bridge nodes tend to lie closer to the center in graphs, whereas this relationship is not consistently observed in hypergraphs. We further find that conventional bridge score estimation may fail to capture the groupwise connectivity of hypergraphs. Motivated by these findings, we propose HyBridge, a self-supervised hyperbolic HRL framework that (1) estimates node bridge scores by considering the external connectivity of individual hyperedges, (2) learns node and hyperedge embeddings through hyperbolic hypergraph encoding, (3) guides node radial positions according to their bridge scores, and (4) employs contrastive learning to preserve node discriminability. Extensive experiments on 10 realworld hypergraphs demonstrate that HyBridge outperforms 22 competing methods, achieving performance improvements of up to 9.82% in node classification. For reproducibility, we provide the code at https://anonymous.4open.science/r/HyBridge/.

Index Terms—Hypergraph representation learning, hyperbolic representation learning, bridge-guided learning, contrastive learning, self-supervised learning

## I. INTRODUCTION

Graph Representation Learning (GRL) aims to learn node embeddings by capturing graph structures and node attributes for downstream tasks, such as node classification and link prediction [1]–[5]. However, many real-world systems involve groupwise relations, where multiple entities jointly participate in a single interaction, as commonly observed in co-authorship networks and biological systems. Conventional graphs, which represent interactions as pairwise relations between nodes, are inherently limited in capturing such higher-order interactions. In contrast, hypergraphs naturally model groupwise relations by allowing each hyperedge to connect more than two nodes. Accordingly, Hypergraph Representation Learning (HRL) aims to learn node embeddings that jointly capture higher-order structural information encoded by hyperedges and node attributes [6]–[9].

Most existing GRL and HRL methods learn node representations in Euclidean space [1]–[9]. However, many realworld graphs and hypergraphs exhibit tree-like structures [10], [11], which are fundamentally difficult to represent faithfully in Euclidean space. In a regular tree where each node has b children, the number of nodes at distance r from the root grows exponentially as b<sup>r</sup>, whereas the volume of a d-dimensional Euclidean ball of radius r grows only polynomially as r<sup>d</sup>. This mismatch in growth rates can introduce substantial structural distortion when embedding tree-like structures into Euclidean space [10], [12].

To alleviate these limitations, Hyperbolic Graph Representation Learning (HB-GRL) has been studied [10], [13]–[18]. Hyperbolic space has negative curvature and expands exponentially with radius, making it well suited for representing tree-like structures [10], [12], [16]. Indeed, previous HB-GRL studies have shown that hyperbolic methods tend to perform particularly well on graph datasets with stronger tree-like structures [10], [16]. Prior HB-GRL and HB-HRL studies have employed hyperbolicity [19] as a measure of tree-likeness, where lower hyperbolicity generally indicates a stronger treelike structure [11], [16]. Motivated by this geometric property, we first examine the hyperbolicity of real-world graph and hypergraph datasets and find that hypergraphs consistently exhibit relatively low hyperbolicity (see Section III-A), suggesting that hyperbolic space may be well suited for HRL.

Despite this geometric compatibility, HB-HRL has received relatively limited attention compared with HB-GRL [11]. We therefore extend representative Euclidean HRL models to hyperbolic space and compare their node classification performance with that of their Euclidean counterparts (see Section III-B). Surprisingly, the results show that hyperbolic models do not consistently outperform their Euclidean counterparts. This finding suggests that merely adopting hyperbolic geometry is insufficient and that effectively leveraging its geometric properties may be crucial for HB-HRL. We therefore draw insights from hyperbolic graph embeddings, where HB-GRL has been extensively studied, to identify structural guidance that can be leveraged for HB-HRL. To this end, we first examine how nodes are positioned in hyperbolic space for graphs.

In hyperbolic graph embeddings, nodes have been shown to form a hierarchical center–periphery structure, with higherlevel nodes in a tree-like structure positioned closer to the center and lower-level nodes positioned farther away [16]. In such a structure, the distance of a node embedding from the origin (i.e., center), referred to as its radial position, reflects its position in the underlying hierarchy. This motivates us to investigate which node structural properties are associated with radial positions in HB-GRL. Using HB-GCN [16], we analyze the relationship between node radial positions and various structural properties (see Section III-C). Our analysis reveals that nodes closer to the center exhibit more diverse neighborhood labels, a characteristic of bridge nodes that connect different node groups [20]–[24]. This finding reveals a connection between a node’s bridging role and its radial position: nodes with stronger bridging roles tend to be positioned closer to the center. Moreover, guiding nodes with higher bridge scores toward the center improves node classification performance, demonstrating the effectiveness of bridge scores as structural guidance for hyperbolic node representations.

We next examine whether the same relationship holds in hypergraphs (see Section III-D). In contrast to graphs, conventional bridge scores do not exhibit a consistent relationship with radial positions, and bridge-guided learning does not consistently improve node classification performance. Conventional bridge score estimation relies on the label distribution of each node’s direct neighbors [20], [24]. In hypergraphs, however, such node-level aggregation can obscure the distinct bridging roles of individual hyperedges, as nodes within the same hyperedge tend to exhibit highly similar neighborhood label distributions despite participating in different hyperedges (see Section III-E). These observations suggest that conventional bridge scores may not adequately capture the groupwise connectivity inherent in hypergraphs, motivating a hypergraphspecific bridge score that explicitly accounts for the distinct bridging roles of individual hyperedges.

To this end, we propose HyBridge, a bridge-guided HB-HRL framework that estimates node-level bridging roles by explicitly accounting for groupwise connectivity and encourages nodes with stronger bridging roles to be positioned closer to the center of hyperbolic space. HyBridge consists of four main steps. First, without relying on ground-truth node labels, HyBridge generates pseudo-cluster labels from the hypergraph connectivity structure using a clustering approach adopted in prior HRL studies [25]. It then estimates the bridge score of each hyperedge based on the pseudo-cluster label distribution of nodes in neighboring hyperedges but outside the target hyperedge and obtains each node’s bridge score by aggregating the scores of its incident hyperedges. Next, HyBridge learns node and hyperedge representations in hyperbolic space using a hyperbolic hypergraph encoder. Then, HyBridge incorporates a bridge-guided loss that encourages nodes with higher bridge scores to be positioned closer to the center. Finally, HyBridge employs contrastive learning to preserve representation discriminability among nodes with similar bridge scores.

Extensive experiments on 10 real-world hypergraph datasets against 22 state-of-the-art competing methods demonstrate the effectiveness of HyBridge across node classification and hyperedge prediction, with improvements of up to 9.82% in node classification. Furthermore, our ablation studies demonstrate the complementary roles of hyperbolic geometry and bridgeguided learning: hyperbolic representations alone do not consistently outperform their Euclidean counterparts, whereas incorporating bridge guidance consistently improves the effectiveness of hyperbolic representations.

The contributions of this paper are summarized as follows:

• Important observations. We investigate the relationship between node structural properties and radial positions in hyperbolic space and identify two key observations.

– (O1) In graphs, nodes with higher bridge scores tend to be positioned closer to the center of hyperbolic space.

– (O2) In hypergraphs, this relationship is not consistently observed, and conventional bridge score estimation may not adequately capture hyperedge-level groupwise connectivity.

• Novel framework. Based on these observations, we propose HyBridge, a self-supervised HB-HRL framework.

– (C1) Hypergraph-specific bridge score estimation that captures groupwise connectivity by explicitly accounting for the distinct bridging roles of individual hyperedges.

– (C2) Bridge-guided learning that uses the estimated bridge scores as structural guidance for node radial positions, together with contrastive learning to preserve representation discriminability.

• Extensive evaluation. We evaluate HyBridge against 22 state-of-the-art competing methods on 10 real-world hypergraph datasets across node classification and hyperedge prediction tasks.

## II. RELATED WORK

Representation Learning in Euclidean Space. Existing GRL methods learn node representations in Euclidean space by leveraging graph structures and node features [1], [2], [26]. More recently, self-supervised GRL methods have been developed to learn node representations without node labels through contrastive or generative objectives [3], [4], [27], [28].

To capture groupwise relationships beyond pairwise graph structures, various HRL methods have been proposed [7], [8]. While early methods transform hypergraph structures into graph-based representations [6], [8], subsequent approaches directly model groupwise relationships through hyperedgebased message passing [7], [29]–[32]. Recent self-supervised HRL methods learn node representations without node labels through contrastive learning [33]–[35], generative learning [25], [36], or virtual label distributions derived from hypergraph connectivity [37].

Representation Learning in Hyperbolic Space. Hyperbolic space is a non-Euclidean manifold with negative curvature whose volume grows exponentially with radius, making it well suited for representing hierarchical structures [10], [15]. Since standard Euclidean operations cannot be directly applied to hyperbolic space, existing hyperbolic GRL methods commonly use logarithmic maps to project node representations onto Euclidean tangent spaces, where linear transformations and neighborhood aggregation are performed, and then map the resulting representations back to hyperbolic space using exponential maps [10], [38], [39]. Representative methods include HB-GCN [16], HB-GAT [15], and HB-GCL [38], which extend graph convolution, attention, and contrastive learning to hyperbolic space, respectively.

More recently, hyperbolic representation learning has been extended from graphs to hypergraphs. Specifically, THINK [11] learns hyperbolic representations of temporal hypergraphs by performing message passing based on hyperbolic distances between node and hyperedge embeddings. However, the structural properties of hyperbolic space have received relatively limited attention in HB-HRL, particularly regarding which structural properties can provide effective guidance for hypergraph representation learning.

## III. PRELIMINARY ANALYSIS

Hyperbolic space has geometric properties well suited for representing tree-like structures, and hyperbolic representation learning has been primarily studied on graphs. However, it remains unclear whether hypergraphs exhibit tree-like structures suitable for hyperbolic representation learning and whether simply extending existing hypergraph encoders to hyperbolic space can improve performance over their Euclidean counterparts. Moreover, unlike graphs with pairwise connectivity, hypergraphs exhibit groupwise connectivity, where multiple nodes interact through a single hyperedge. This structural difference raises the question of whether structural properties identified in graphs remain effective for guiding hyperbolic representation learning on hypergraphs. To systematically investigate these issues and identify structural properties suitable for effective HB-HRL, we conduct preliminary experiments guided by the following five preliminary research questions (PQs).

• PQ1: Do hypergraphs exhibit strong tree-like structures suitable for hyperbolic representation learning?

• PQ2: Is simply extending existing hypergraph encoders to hyperbolic space sufficient to improve downstream performance?

• PQ3: Which node structural properties can provide effective guidance for HB-GRL?

• PQ4: Does the same structural guidance remain effective when applied to hypergraphs?

• PQ5: Why does the effectiveness of structural guidance differ between graphs and hypergraphs, and what hypergraphspecific structural properties should be considered for HB-HRL?

## A. PQ1. Suitability of Hyperbolic Space for HRL

To assess the suitability of hyperbolic space for HRL, we measure hyperbolicity $( \delta _ { h g } )$ on four graph datasets used in prior HB-GRL studies [16], [38] and ten hypergraph datasets widely used in HRL [33], [36], using established graphspecific [16] and hypergraph-specific [11] measures. Lower hyperbolicity indicates a stronger tree-like structure. Fig. 1 presents the distribution of hyperbolicity across graph and hypergraph datasets, where the x-axis represents the graph and hypergraph categories and the y-axis represents hyperbolicity $( \delta _ { h g } )$ . The results show that graph hyperbolicity varies widely from approximately 0 to 11, whereas hypergraphs exhibit relatively low values ranging from 1 to 3.5, despite the two categories having similar median values. These results indicate that hypergraphs consistently exhibit strong tree-like structures, supporting their geometric suitability for hyperbolic representation learning. However, whether this geometric suitability translates into improved downstream performance remains unclear.

![](images/10b041f12342499b007ed7377fe68ed6f10b8a2b80cd78974533e8231a4cc1c6.jpg)  
Fig. 1: Hyperbolicity $( \delta _ { h g } )$ of graphs and hypergraphs.

## B. PQ2. Effectiveness of Hyperbolic Extension

To examine whether hyperbolic space improves HRL performance, we extend four existing HRL models to hyperbolic space using mapping-based operations adopted in HB-GRL, which perform message passing in Euclidean tangent spaces [16]. Specifically, we select HGNN [8] and HNHN [7] as semi-supervised methods based on clique and star expansion, respectively, and TriCL [33] and HypeBoy [36] as self-supervised methods based on contrastive and generative learning, respectively. Table I reports the node classification accuracy of the original Euclidean HRL models and their hyperbolic counterparts across ten hypergraph datasets, together with the relative performance changes achieved by the hyperbolic variants. The results show that the effectiveness of hyperbolic extensions varies substantially across models and datasets. Specifically, HB-HGNN improves HGNN by 18.96% on Cora-CA and 11.18% on 20News, while resulting in an 8.20% decrease on Cora. Similarly, HB-HNHN achieves an 8.25% improvement on Pubmed but a 7.24% degradation on Cora. Overall, these results demonstrate that simply extending existing HRL models to hyperbolic space does not consistently improve downstream performance, even on hypergraphs with strong tree-like structures.

“‘latex “‘

## C. PQ3. Structural Guidance for HB-GRL

Previous HB-GRL studies have reported consistent performance improvements over Euclidean methods [13], [15], [16], [38], [39]. To investigate which node structural properties can provide effective guidance for HB-GRL, we analyze node embeddings learned by HB-GCN [16] with respect to nine structural properties: degree, clustering coefficient, and coreness as local-level properties; closeness, betweenness, PageRank, and eigenvector centrality as global-level properties; and label entropy and homophily as community-level properties [21], [40]–[43]. We sort nodes by their embedding radii from the center to the periphery, divide them into ten equal-sized groups, and compute the average value of each structural property within each group.

TABLE I: Node classification accuracy of Euclidean and hyperbolic HRL methods
<table><tr><td>Method</td><td>Citeseer</td><td>Cora</td><td>DBLP-A</td><td> $\mathbf { C o r a - C A }$ </td><td>IMDB</td><td>Pubmed</td><td>DBLP-P</td><td>Aminer</td><td>MN-40</td><td>20News</td></tr><tr><td>HGNN</td><td> $3 8 . 1 \pm 1 0 . 7$ </td><td> $4 5 . 1 \pm 9 . 1$ </td><td> $6 5 . 0 \pm 5 . 7$ </td><td> $4 4 . 3 \pm 8 . 0$ </td><td> $4 1 . 5 \pm 2 . 8$ </td><td> $7 0 . 8 \pm 3 . 8$ </td><td> $8 4 . 0 \pm 0 . 9$ </td><td> $2 9 . 7 \pm 2 . 0$ </td><td> $8 9 . 5 \pm 1 . 3$ </td><td> $6 7 . 1 \pm 2 . 0$ </td></tr><tr><td>HB-HGNN</td><td> $3 6 . 8 \pm 7 . 1$ </td><td> $4 1 . 4 \pm 5 . 7$ </td><td> $6 4 . 3 \pm 3 . 5$ </td><td> $5 2 . 7 \pm 6 . 7$ </td><td> $4 0 . 1 \pm 1 . 8$ </td><td> $7 1 . 9 \pm 4 . 0$ </td><td> $8 5 . 7 \pm 0 . 8$ </td><td> $3 0 . 3 \pm 1 . 1$ </td><td> $8 6 . 0 \pm 2 . 1$ </td><td> $7 4 . 6 \pm 0 . 8$ </td></tr><tr><td>Improvement</td><td>-3.41%</td><td>-8.20%</td><td>-1.08%</td><td>+18.96%</td><td>-3.37%</td><td>+1.55%</td><td>+2.02%</td><td>+2.02%</td><td>-3.91%</td><td>+11.18%</td></tr><tr><td>HNHN</td><td> $4 4 . 2 \pm 7 . 9$ </td><td> $5 2 . 5 \pm 7 . 3$ </td><td> $6 9 . 0 \pm 4 . 5$ </td><td> $5 3 . 1 \pm 7 . 4$ </td><td> $4 2 . 5 \pm 3 . 6$ </td><td> $6 9 . 1 \pm 3 . 4$ </td><td> $8 6 . 4 \pm 0 . 8$ </td><td> $3 2 . 1 \pm 2 . 7$ </td><td> $8 9 . 2 \pm 1 . 4$ </td><td> $7 4 . 4 \pm 1 . 1$ </td></tr><tr><td>HB-HNHN</td><td> $4 2 . 1 \pm 7 . 7$ </td><td> $4 8 . 7 \pm 1 0 . 8$ </td><td> $6 5 . 2 \pm 5 . 2$ </td><td> $5 3 . 6 \pm 6 . 2$ </td><td> $4 1 . 6 \pm 3 . 9$ </td><td> $7 4 . 8 \pm 4 . 1$ </td><td> $8 6 . 3 \pm 0 . 7$ </td><td> $3 0 . 9 \pm 1 . 5$ </td><td> $9 0 . 1 \pm 0 . 7$ </td><td> $7 6 . 7 \pm 0 . 8$ </td></tr><tr><td>Improvement</td><td>-4.75%</td><td>-7.24%</td><td>-5.51%</td><td>+0.94%</td><td>-2.12%</td><td>+8.25%</td><td>-0.12%</td><td>-3.74%</td><td>+1.01%</td><td>+3.09%</td></tr><tr><td>TriCL</td><td> $5 3 . 0 \pm 9 . 9$ </td><td> $6 1 . 5 \pm 5 . 3$ </td><td> $8 0 . 3 \pm 3 . 5$ </td><td>63.4 ± 3.9</td><td> $4 7 . 5 \pm 4 . 6$ </td><td> $7 4 . 0 \pm 5 . 2$ </td><td> $8 7 . 6 \pm 0 . 5$ </td><td> $3 4 . 6 \pm 2 . 7$ </td><td> $9 3 . 0 \pm 0 . 8$ </td><td> $7 3 . 8 \pm 1 . 8$ </td></tr><tr><td>HB-TriCL</td><td> $5 5 . 8 \pm 8 . 7$ </td><td> $5 9 . 7 \pm 6 . 1$ </td><td> $7 8 . 9 \pm 3 . 9$ </td><td> $6 4 . 2 \pm 4 . 3$ </td><td> $4 6 . 8 \pm 4 . 2$ </td><td> $7 6 . 1 \pm 4 . 7$ </td><td>0.0.M</td><td>0.0.M</td><td> $9 2 . 5 \pm 1 . 0$ </td><td> $7 5 . 4 \pm 1 . 5$ </td></tr><tr><td>Improvement</td><td>+5.28%</td><td>-2.93%</td><td>-1.74%</td><td>+1.26%</td><td>-1.47%</td><td>+2.84%</td><td>一</td><td></td><td>-0.54%</td><td>+2.17%</td></tr><tr><td>Hypeboy</td><td> $5 7 . 7 \pm 1 0 . 8$ </td><td> $6 1 . 9 \pm 8 . 5$ </td><td> $8 1 . 2 \pm 2 . 7$ </td><td> $6 7 . 0 \pm 3 . 7$ </td><td> $4 8 . 3 \pm 5 . 0$ </td><td> $7 3 . 7 \pm 4 . 1$ </td><td> $8 7 . 9 \pm 0 . 5$ </td><td> $3 4 . 6 \pm 2 . 8$ </td><td> $8 9 . 2 \pm 2 . 2$ </td><td> $7 5 . 7 \pm 2 . 1$ </td></tr><tr><td>HB-Hypeboy</td><td> $5 8 . 0 \pm 8 . 1$ </td><td> $6 3 . 1 \pm 1 . 5$ </td><td> $7 9 . 0 \pm 3 . 5$ </td><td> $6 4 . 0 \pm 1 . 0$ </td><td> $5 0 . 0 \pm 1 . 7$ </td><td> $7 1 . 6 \pm 2 . 0$ </td><td>0.0.M</td><td> $3 4 . 9 \pm 1 . 8$ </td><td> $9 3 . 3 \pm 1 . 2$ </td><td> $7 3 . 8 \pm 2 . 6$ </td></tr><tr><td>Improvement</td><td>+0.52%</td><td>+1.94%</td><td>-2.71%</td><td>-4.48%</td><td>+3.52%</td><td>-2.85%</td><td></td><td>+0.87%</td><td>+4.60%</td><td>-2.51%</td></tr></table>

(a) Degree  
![](images/2fd8c52dfcf88a06de06858f1f42a15700f8937c55db54221b8a68967360b8c9.jpg)  
(d) Closeness

(b) Clustering  
![](images/086bb26c228d774afcb0968939e120969eae7bf2081c2fa7cf3a097e7ced5da0.jpg)  
(e) Betweenness (×10<sup>−2</sup>)

(c) Coreness  
![](images/09ac5023f38e7265531c5dc8d4759d9a593d7bb36e7e790eb5a12b7dbb531d2d.jpg)  
(f) PageRank (×10<sup>−4</sup>)

![](images/678f91e0b6382839411de153cf4765a879afd8a0bdbe4772ad233f128ee09151.jpg)

![](images/2f36dc27dd4c88d2f8c112ec0980f8c7867e097c5f3956d5fe499a825ee25234.jpg)

![](images/25f7f6e3679f07fdac334485d4d1046d9dee16c116e359e36586a6ad35cfe1be.jpg)

![](images/d31f25f83969304fd6cddc48d6c3d4364e369c588a3723f2d8e72a813950614d.jpg)

![](images/40cc9dab0241e43fc202d074b94e53d2ae2100dcd80e77eb00357dfcd41aa9c7.jpg)

![](images/0981c8676442ef456cc0a003b81fc47c4a9f9eba0bf7a602c1cc9a72127c2460.jpg)  
Fig. 2: Graph structural properties across hyperbolic radius bins.

TABLE II: Node classification accuracy of HB-GCN with and without bridge-guided learning
<table><tr><td>Method</td><td>Cora</td><td>Airport</td><td>Pubmed</td><td>Disease</td></tr><tr><td>HB-GCN</td><td>0.787</td><td>0.826</td><td>0.738</td><td>0.763</td></tr><tr><td> $\mathrm { H B - G C N + B r i d g e }$ </td><td>0.802</td><td>0.870</td><td>0.762</td><td>0.799</td></tr><tr><td>Improvement</td><td>+1.91%</td><td>+5.33%</td><td>+3.25%</td><td>+4.72%</td></tr></table>

![](images/35fb1e814d1c52b1660ab7089f2601e915ea26c597d61b6fb21ea2c571fc605e.jpg)  
Fig. 3: Hypergraph community properties across hyperbolic radius bins.

Fig. 2 presents how the nine structural properties vary with node radial positions, where the x-axis represents the ten node groups ordered from the center to the periphery and the y-axis represents the average value of each structural property within each group. The results show that only the community-level properties exhibit consistent trends across graph datasets: label entropy decreases while homophily increases from the center

TABLE III: Node classification accuracy of HB-HNHN with and without bridge-guided learning
<table><tr><td>Method</td><td>Citeseer</td><td>Cora</td><td>DBLP-P</td><td> $\mathbf { C o r a – C A }$ </td><td>IMDB</td><td>Pubmed</td><td>DBLP-A</td><td>Aminer</td><td>MN-40</td><td>20News</td></tr><tr><td>HB-HNHN</td><td> ${ \bf 4 2 . 1 \pm 7 . 7 }$ </td><td> $4 8 . 7 \pm 1 0 . 8$ </td><td> $6 5 . 2 \pm 5 . 2$ </td><td> $5 3 . 6 \pm 6 . 2$ </td><td> ${ \bf 4 1 . 6 \pm } \ : 3 . 9$ </td><td> $7 4 . 8 \pm 4 . 1$ </td><td> ${ \bf 8 6 . 3 \pm 0 . 7 }$ </td><td> $3 0 . 9 \pm 1 . 5$ </td><td> ${ \bf 9 0 . 1 } \pm 0 . 7$ </td><td> $7 6 . 7 \pm 0 . 8$ </td></tr><tr><td>HB-HNHN + Bridge</td><td> $4 1 . 5 \pm 8 . 1$ </td><td> ${ \bf 4 9 . 3 \pm 8 . 8 }$ </td><td> ${ \bf 6 7 . 0 \pm 4 . 1 }$ </td><td> ${ \bar { 5 } } 3 . 7 \pm 8 . 6$ </td><td> $4 1 . 5 \pm 4 . 0$ </td><td> $7 4 . 5 \pm 3 . 9$ </td><td> $8 6 . 2 \pm 1 . 0$ </td><td> $3 1 . 3 \pm 1 . 6$ </td><td> $9 0 . 0 \pm 0 . 7$ </td><td> ${ \bf 7 6 . 8 \pm 1 . 3 }$ </td></tr><tr><td>Improvement</td><td>-1.43%</td><td>+1.23%</td><td>+2.76%</td><td>+0.19%</td><td>-0.24%</td><td>-0.40%</td><td>-0.12%</td><td>+1.29%</td><td>-0.11%</td><td>+0.13%</td></tr></table>

![](images/9ba8116983cd4366e9897db5d8fc80108c309babd97191dc8c49ca5925090724.jpg)

Fig. 4: Jensen-Shannon divergence (JSD) of bridge and toward the periphery. These trends indicate that nodes closer to the center tend to have more diverse neighborhood labels and lower homophily, characteristics associated with bridge nodes that connect different label groups. Thus, the results reveal a consistent relationship between node bridging roles and radial positions in hyperbolic space: nodes with stronger bridging roles tend to be positioned closer to the center.

Based on this observation, we quantify such bridging roles using bridge scores and examine whether they can serve as effective structural guidance for HB-GRL. We incorporate a bridge-guided loss into HB-GCN [16], which encourages nodes with higher bridge scores to have smaller embedding radii while retaining the original node classification objective. Table II compares the node classification accuracy of HB-GCN with and without the bridge-guided loss. The results show that incorporating bridge guidance consistently improves node classification performance across graph datasets. These results demonstrate that bridge scores can serve as effective structural guidance for organizing node representations along the radial dimension in HB-GRL.

## D. PQ4. Generalization to Hypergraphs

To examine whether the bridge–radius relationship observed in HB-GRL also holds for HB-HRL, we learn node embeddings using HB-HNHN and repeat the same analysis on hypergraph datasets. Fig. 3 presents the relationship between node radial positions and the two community-level structural properties, label entropy and homophily. Unlike HB-GRL, HB-HRL exhibits no consistent relationship between radial positions and either label entropy or homophily across hypergraph datasets.

We further examine whether bridge scores can provide effective structural guidance for HB-HRL by applying the same bridge-guided loss used in HB-GRL to HB-HNHN. Table III compares the node classification accuracy of HB-HNHN with and without the bridge-guided loss across hypergraph datasets. The results show that bridge-guided learning does not consistently improve node classification performance, non-bridge nodes across graph and hypergraph datasets. yielding improvements on some datasets but degradation on others. These results indicate that neither the bridge–radius relationship nor the effectiveness of bridge guidance observed in HB-GRL directly generalizes to HB-HRL.

## E. PQ5. Hypergraph-Specific Structural Properties

To investigate why conventional bridge score estimation may be less effective in hypergraphs, we examine if bridge nodes have neighborhood label distributions different from those of their surrounding nodes. Specifically, for each node $v _ { i } ,$ we quantify how different its neighborhood label distribution is from those of its connected nodes $v _ { j }$ (i.e., neighbors in graphs or hyperedge co-members in hypergraphs) by measuring the difference for each pair $( v _ { i } , v _ { j } )$ and averaging these differences for each $v _ { i } .$ We use Jensen–Shannon Divergence (JSD) to measure these differences, where lower values indicate greater similarity. To examine whether these differences distinguish bridge from non-bridge nodes, we classify a node as a bridge node if it connects to at least one node with a different label, and as a non-bridge node otherwise.

Fig. 4 presents the JSD distributions of bridge and nonbridge nodes across graph and hypergraph datasets, where the x-axis represents the datasets and the y-axis represents JSD. In graph datasets, bridge nodes consistently exhibit higher JSD values than non-bridge nodes, showing a clear distinction between the two groups. In hypergraph datasets, bridge nodes generally exhibit higher JSD values, but their distributions vary substantially and frequently extend toward near-zero JSD values. This indicates that many bridge nodes can still have neighborhood label distributions highly similar to those of their hyperedge co-members, making their bridging roles difficult to distinguish using conventional node-level bridge score estimation. Moreover, conventional bridge scores aggregate all neighboring nodes without accounting for the distinct bridging roles of individual hyperedges. These findings suggest that effective bridge score estimation for hypergraphs should explicitly account for how individual hyperedges connect different external node groups.

![](images/23bbb30420f0d8752160548bc495987fd74bfea02d01f96c0ce4b945e88f756f.jpg)  
Fig. 5: Overview of representation learning in HyBridge.

## IV. HYBRIDGE: THE PROPOSED FRAMEWORK

## A. Notations and Overview

A hypergraph is defined as $\mathcal { H } = ( V , E )$ , where V denotes the set of nodes and E denotes the set of hyperedges. Its structure is represented by an incidence matrix $\mathbf { \bar { H } } \in \bar { \left\{ 0 , 1 \right\} } ^ { | V | \times | E | }$ where $\mathbf { H } ( v , e ) \ = \ 1$ indicates that node v is incident to hyperedge e. Each hyperedge is associated with a positive weight, represented by a diagonal matrix $\textbf { W } \in \ \mathbb { R } ^ { \bar { | } E | \times | E | }$ Node features are given by $\mathbf { X } \in \mathbb { R } ^ { | V | \times F }$ , and we equivalently represent the hypergraph as $\mathcal { H } ~ = ~ ( \mathbf { X } , \mathbf { H } )$ . The node and hyperedge degree matrices are denoted by $\mathbf { D } _ { V }$ and $\mathbf { D } _ { E } .$ respectively. In this paper, we address HRL under a $\it { f u l l y }$ unsupervised setting, where the goal is to learn a hypergraph encoder $f _ { \theta }$ that maps each node $v _ { i } \in V$ to an $F ^ { \prime }$ -dimensional representation $\mathbf { p } _ { i } \in \mathbb { R } ^ { F ^ { \prime } }$ . The learned representations aim to preserve both the hypergraph structure and node attribute information in the latent space.

Fig. 5 illustrates the overall HB-HRL process of HyBridge, which consists of four main steps: (a) bridge score estimation, (b) hyperbolic hypergraph encoding, (c) bridge-guided learning, and (d) contrastive learning. Given a hypergraph H as input, bridge score estimation first estimates the bridge score of each node (Fig. 5(a)). Then, hyperbolic hypergraph encoding maps nodes and hyperedges to representations in an $F ^ { \prime }$ -dimensional hyperbolic latent space (Fig. 5(b)). Next, bridge-guided learning aligns the relative ordering of node embedding radii with their bridge scores (Fig. 5(c)). Finally, contrastive learning preserves representation discriminability by treating the same node or hyperedge across two augmented views as a positive pair and different nodes or hyperedges as negative pairs (Fig. 5(d)).

## B. Bridge Score Estimation

This step aims to quantify how strongly a node serves as a bridge across different communities in a hypergraph. To this end, we estimate a bridge score for each hyperedge $e _ { j }$ based on the label diversity of its external nodes $\nu _ { \mathrm { e x t } } ( e _ { j } )$ ), which belong to neighboring hyperedges but not to the target hyperedge $e _ { j }$ itself. We then obtain each node’s bridge score by averaging the bridge scores of its incident hyperedges.

To estimate bridge scores without relying on ground-truth node labels, we first generate pseudo-cluster labels based solely on the hypergraph structure. We apply METIS [44], which has been used in previous hypergraph studies [25], to partition nodes into K clusters based on structural connectivity.<sup>1</sup> Following [25], we set K to the number of groundtruth classes in each hypergraph, while using no groundtruth label assignments. Using the resulting pseudo-cluster labels, we then characterize the external nodes $\nu _ { \mathrm { e x t } } ( e _ { j } )$ for each hyperedge $e _ { j }$ using two hyperedge-level metrics adopted in previous hypergraph studies: label entropy [47] and label homophily [48].

Unlike these studies, which compute these metrics using the labels of nodes within the target hyperedge, we compute both metrics using the pseudo-cluster labels of external nodes.<sup>2</sup> We then combine the normalized label entropy $( \mathrm { E n t _ { e x t } } ( e _ { j } ) )$ and the complement of label homophily $( 1 { - } \mathrm { H o m } _ { \mathrm { e x t } } ( e _ { j } ) )$ with equal weights to obtain the bridge score of each hyperedge. Finally, we compute each node’s bridge score by averaging the bridge scores of its incident hyperedges.

## C. Hyperbolic Hypergraph Encoding

This step aims to learn node representations that capture groupwise relationships in hyperbolic space. To this end,

HyBridge employs HB-HNHN, a hyperbolic encoder that extends the node-to-hyperedge and hyperedge-to-node aggregation operations of the Euclidean HNHN [7] to hyperbolic space. We choose HNHN as the base encoder because its two-stage message passing directly operates on hyperedges, thereby preserving groupwise structural information that may be lost when hyperedges are reduced to pairwise connections.

Existing HB-GRL methods typically perform linear transformations and aggregation operations in tangent spaces. Specifically, hyperbolic representations are projected onto tangent spaces via logarithmic maps $\mathrm { l o g } _ { o } ( \cdot )$ , where standard Euclidean operations can be performed, and then mapped back to hyperbolic space via exponential maps $\exp _ { o } ( \cdot )$ . Following this approach, HB-HNHN first maps the initial Euclidean node features to hyperbolic space via an exponential map. It then performs node-to-hyperedge and hyperedge-to-node aggregation in the tangent space and maps the resulting representations back to hyperbolic space, as formulated in Eq. 1.

$$
\begin{array} { r } { \mathbf { Q } ^ { k } = \exp _ { o } ^ { c ^ { k } } \left( \sigma \left( \mathbf { D } _ { E } ^ { - 1 } \mathbf { H } ^ { \top } \log _ { o } ^ { c ^ { k } } ( \mathbf { P } ^ { k } ) \boldsymbol { \Theta } _ { N } ^ { k } + b _ { N } ^ { k } \right) \right) , } \\ { \mathbf { P } ^ { k + 1 } = \exp _ { o } ^ { c ^ { k + 1 } } \left( \sigma \left( \mathbf { D } _ { V } ^ { - 1 } \mathbf { H } \log _ { o } ^ { c ^ { k } } ( \mathbf { Q } ^ { k } ) \boldsymbol { \Theta } _ { E } ^ { k } + b _ { E } ^ { k } \right) \right) , } \end{array}\tag{1}
$$

where $\mathbf { P } ^ { k }$ and $\mathbf { Q } ^ { k }$ denote the node and hyperedge embeddings at the k-th layer, respectively; H is the incidence matrix; $\mathbf { D } _ { V }$ and $\mathbf { D } _ { E }$ are the node and hyperedge degree matrices, respectively; $\Theta _ { N } ^ { k }$ and $\Theta _ { E } ^ { k }$ denote learnable transformation matrices; $b _ { N } ^ { k }$ and $b _ { E } ^ { k }$ denote learnable bias terms; and $\sigma ( \cdot )$ denotes a nonlinear activation function.

The node embeddings $\mathbf { P } ^ { k }$ are first mapped to the tangent space, where the representations of incident nodes are aggregated for each hyperedge via $\mathbf { H } ^ { \top }$ . The aggregated representations are then transformed using the learnable parameters $\Theta _ { N } ^ { k }$ and $b _ { N } ^ { k }$ and mapped back to hyperbolic space, yielding the hyperedge embeddings $\mathbf { Q } ^ { k }$ . Similarly, the hyperedge embeddings $\mathbf { Q } ^ { k }$ are mapped to the tangent space and aggregated for each incident node via H. The aggregated representations are then transformed using $\Theta _ { E } ^ { k }$ and $\bar { b } _ { E } ^ { k }$ and mapped back to hyperbolic space, yielding the next-layer node embeddings $\mathbf { P } ^ { k + \mathrm { i } }$

## D. Bridge-guided Learning

This step aims to determine the appropriate embedding radii of nodes within the hierarchical center–periphery structure using their bridge scores as structural guidance. Accordingly, we encourage nodes with higher bridge scores to be positioned closer to the hyperbolic center.

We first obtain the node embeddings through the hyperbolic hypergraph encoder $f _ { \theta }$ . Given the embedding $\mathbf { p } _ { i } = f _ { \theta } ( v _ { i } )$ of node $v _ { i }$ , its hyperbolic radius is defined as $r _ { i } = d _ { \mathbb { H } } ( { \bf p } _ { i } , { \bf o } )$ where $d _ { \mathbb { H } } ( \cdot , \cdot )$ denotes the hyperbolic distance function and o denotes the origin of hyperbolic space. We then randomly sample a set of node pairs $\mathcal { P }$ and construct ${ \mathcal { P } } ^ { + }$ based on their bridge-score ordering. Specifically, if node $v _ { i }$ has a higher bridge score than node $v _ { j } \ ( B _ { i } > B _ { j } )$ but is positioned farther from the hyperbolic center $( r _ { i } > r _ { j } )$ , a penalty is imposed to encourage $r _ { i } < r _ { j }$ . Based on this ranking principle, we define the bridge-guided loss as follows:

$$
\mathcal { L } _ { \mathrm { b r i d g e } } = \frac { 1 } { | \mathcal { P } ^ { + } | } \sum _ { ( i , j ) \in \mathcal { P } ^ { + } } \operatorname* { m a x } ( 0 , r _ { i } - r _ { j } ) ,\tag{2}
$$

where $\mathcal { P }$ denotes the set of randomly sampled node pairs, and $\mathcal { P } ^ { + } = \{ ( i , j ) \in \mathcal { P } \mid B _ { i } > B _ { j } \}$ denotes the subset of pairs in which node $v _ { i }$ has a higher bridge score than node $v _ { j }$

## E. Contrastive Learning

This step aims to preserve representation discriminability among nodes while maintaining the radial structure guided by their bridge scores. Since nodes with similar bridge scores may be positioned at similar radii, we employ contrastive learning to distinguish their representations. Specifically, contrastive learning encourages representations of the same node across two augmented views to be closer while pushing representations of different nodes farther apart.

We first generate two views of the original hypergraph by randomly masking node features and node–hyperedge memberships according to their respective augmentation ratios, where higher ratios indicate a larger proportion of masked elements. We then obtain node embeddings P and hyperedge embeddings Q for each view using the same hyperbolic hypergraph encoder $f _ { \theta } .$ . Next, we perform contrastive learning at both the node level, which aligns representations of the same node across the two views, and the hyperedge level, which aligns representations of the same hyperedge across the two views.

More specifically, for node-level contrastive learning, Hy-Bridge uses the representation of node $v _ { i }$ from the first view, ${ \bf z } _ { 1 , i } ,$ as the anchor, its counterpart from the second view, $\mathbf { z } _ { 2 , i }$ as the positive sample, and all other node representations ${ \bf z } _ { 2 , k } ,$ where $k \neq i ,$ , as negative samples. The contrastive loss for each node $v _ { i }$ is defined as follows:

$$
\ell _ { n } ( \mathbf { z } _ { 1 , i } , \mathbf { z } _ { 2 , i } ) = - \log \frac { \exp \left( s ( \mathbf { z } _ { 1 , i } , \mathbf { z } _ { 2 , i } ) / \tau _ { n } \right) } { \sum _ { k = 1 } ^ { | V | } \exp \left( s ( \mathbf { z } _ { 1 , i } , \mathbf { z } _ { 2 , k } ) / \tau _ { n } \right) } ,\tag{3}
$$

where $\tau _ { n }$ is a temperature parameter [33], and $s ( \cdot , \cdot )$ is a scoring function based on hyperbolic distance that assigns scores between 0 and 1, with higher scores indicating closer representations in hyperbolic space [38]. The overall nodelevel contrastive loss is then computed by averaging the losses over all nodes in both directions between the two views:

$$
\mathcal { L } _ { n } = \frac { 1 } { 2 | V | } \sum _ { i = 1 } ^ { | V | } \left\{ \ell _ { n } ( \mathbf { z } _ { 1 , i } , \mathbf { z } _ { 2 , i } ) + \ell _ { n } ( \mathbf { z } _ { 2 , i } , \mathbf { z } _ { 1 , i } ) \right\} .\tag{4}
$$

For hyperedge-level contrastive learning, HyBridge uses the representation of hyperedge $e _ { j }$ from the first view, $\mathbf { y } _ { 1 , j }$ , as the anchor, its counterpart from the second view, $\mathbf { y } _ { 2 , j }$ , as the positive sample, and all other hyperedge representations $\mathbf { y } _ { 2 , k } .$ where $k \neq j$ , as negative samples. The contrastive loss for each hyperedge $e _ { j }$ is defined as follows:

$$
\ell _ { h } ( \mathbf { y } _ { 1 , j } , \mathbf { y } _ { 2 , j } ) = - \log \frac { \exp { ( s ( \mathbf { y } _ { 1 , j } , \mathbf { y } _ { 2 , j } ) / \tau _ { h } ) } } { \sum _ { k = 1 } ^ { | E | } \exp { ( s ( \mathbf { y } _ { 1 , j } , \mathbf { y } _ { 2 , k } ) / \tau _ { h } ) } } ,\tag{5}
$$

where $\tau _ { h }$ is a temperature parameter [33]. The overall hyperedge-level contrastive loss is then computed by averaging

the losses over all hyperedges in both directions between the two views:

$$
\mathcal { L } _ { h } = \frac { 1 } { 2 | E | } \sum _ { j = 1 } ^ { | E | } \left\{ \ell _ { h } ( \mathbf { y } _ { 1 , j } , \mathbf { y } _ { 2 , j } ) + \ell _ { h } ( \mathbf { y } _ { 2 , j } , \mathbf { y } _ { 1 , j } ) \right\} .\tag{6}
$$

By combining the node-level and hyperedge-level contrastive losses in Eqs. 4 and $^ { 6 , }$ respectively, the overall contrastive loss is defined as follows:

$$
\mathcal { L } _ { c l } = \omega _ { n } \mathcal { L } _ { n } + \omega _ { h } \mathcal { L } _ { h } .\tag{7}
$$

Finally, the overall objective of HyBridge jointly optimizes the bridge-guided radial organization and representation discriminability:

$$
\mathcal { L } = \omega _ { b r i d g e } \mathcal { L } _ { b r i d g e } + \mathcal { L } _ { c l } .\tag{8}
$$

Through this joint optimization, HyBridge learns discriminative node representations while preserving the bridge-guided hierarchical center–periphery structure in hyperbolic space.

## F. Time Complexity Analysis

We summarize the overall learning process of HyBridge in Algorithm 1. HyBridge first initializes the model parameters θ (line 1) and estimates node-level bridge scores by computing the bridge score of each hyperedge and averaging the scores of its incident hyperedges for each node (lines 2–6). During training, HyBridge generates two augmented hypergraph views and obtains their node and hyperedge embeddings using the hyperbolic hypergraph encoder (lines 7–11). It then randomly samples node pairs, constructs ${ \mathcal { P } } ^ { + }$ according to their bridgescore ordering, and computes the bridge-guided loss (lines 12–14), followed by the contrastive loss (line 15). Finally, HyBridge jointly optimizes the two losses to update the model parameters (lines 16–17) and generates the final node representations (lines 19–20).

We further analyze the time complexity of HyBridge. Step 1 is a one-time preprocessing step that generates pseudocluster labels, identifies external node sets, and computes node bridge scores. Step 2 requires $O ( L ( | V | + | E | + S ) )$ per epoch, where L is the number of encoder layers and S is the number of node–hyperedge incidences. Step 3 requires $O ( | \mathcal { P } | )$ for sampled node pairs P. Step 4 requires $O ( | V | ^ { 2 } + | E | ^ { 2 } )$ for node- and hyperedge-level contrastive objectives. Therefore, the overall per-epoch time complexity of HyBridge is $O ( L ( | V | + | E | + S ) + | \mathcal { P } | + | V | ^ { 2 } + | E | ^ { 2 } )$ , excluding the one-time preprocessing cost. We further evaluate its practical training efficiency and preprocessing overhead in Section V-B-EQ6.

## V. EVALUATION

In this section, we validate the effectiveness of HyBridge through extensive experiments designed to answer the following evaluation questions (EQs):

• EQ1: What are the contributions of hyperbolic representation learning and bridge-guided learning?

• EQ2: Does HyBridge outperform existing state-of-the-art methods across different downstream tasks?

## Algorithm 1 HyBridge framework

Input: Hypergraph H = (X, H) (node vector matrix X, incidence matrix H)   
Output: Learned node representations P   
1: Initialize model parameters θ   
2: Assign pseudo cluster label using METIS ▷ Step 1: Bridge score estimation   
3: Identify external node sets $\{ \mathcal { V } _ { \mathrm { e x t } } ( e _ { j } ) \} _ { e _ { j } \in E }$   
4: Compute Ent<sub>ext</sub>(e<sub>j</sub> ) and Hom<sub>ext</sub>(e<sub>j</sub> ) using pseudo cluster label   
5: Hyperedge bridge score ← <sup>1</sup> (Ent<sub>ext</sub> + 1 − Hom<sub>ext</sub>)   
6: Compute node-level bridge scores B by averaging incident hyperedge bridge scores   
7: for epoch ← 1 to epochs do ▷ Step 2: Hyperbolic hypergraph encoding   
8: Generate two augmented views $\mathcal { H } ^ { ( 1 ) } , \mathcal { H } ^ { ( 2 ) }$   
9: for $a \in \{ 1 , 2 \} \ { \overset { \smile } { \mathbf { d o } } }$   
10: $( \mathbf { P } ^ { ( a ) } , \mathbf { Q } ^ { \mathsf { \bar { ( } } a ) } ) \gets f _ { \theta } ( \mathcal { H } ^ { ( a ) } )$ via Eq. 1   
11: end for   
12: Sample ordered node pairs P ▷ Step 3: Bridge-guided learning   
13: Construct ${ \mathcal { P } } ^ { + }$ based on B   
14: Compute bridge-guided loss $\mathcal { L } _ { \mathrm { b r i d g e } }$ via Eq. 2   
15: Compute contrastive loss $\mathcal { L } _ { c l }$ via Eq. 7 ▷ Step 4: Contrastive learning   
16: $\mathcal { L } \gets \omega _ { b r i d g e } \mathcal { L } _ { \mathrm { b r i d g e } } + \mathcal { L } _ { c l }$   
17: Update θ by minimizing L   
18: end for   
$1 9 \colon ( \mathbf { P } , \mathbf { Q } )  f _ { \theta } ( \mathcal { H } )$   
20: return P

• EQ3: How does using within-hyperedge versus externalnode label distributions for bridge score estimation affect the performance of HyBridge?

• EQ4: Does bridge-guided learning induce consistent structural patterns across hyperbolic radii?

• EQ5: How sensitive is HyBridge to its key hyperparameters?

• EQ6: How computationally efficient and scalable is Hy-Bridge?

## A. Experimental Settings

Datasets. We use 10 real-world hypergraph datasets from diverse domains, including co-citation (i.e., Cora [6], Citeseer [6], and PubMed [6]), co-authorship (i.e., Cora-CA [6], DBLP-P [6], DBLP-A [49], and AMiner [50]), computer graphics (i.e., MN-40 [51]), movie–actor relationships (i.e., IMDB [49]), and news content $( i . e .$ , 20News [52]). These datasets constitute the union of those adopted in prior work [33], [36]. Table IV summarizes their basic statistics.

Competing Methods. We compare HyBridge with 22 competing methods. The Euclidean methods include (1) 15 HRL methods: HGNN [8], HyperGCN [6], HNHN [7], UniGNN variants (i.e., UniGCN, UniGIN, and UniGC-NII) [29], AllSet [30], ED-HNN [31], PhenomNN [32], TriCL [33], HyperGCL [34], HyperGRL [25], HypeBoy [36], SE-HSSL [35], and VilLain [37]; (2) three GRL methods: GGD [28], GraphMAE2 [3], and MaskGAE [27]; and (3) one baseline method: MLP [53]. The hyperbolic methods include (1) two HB-GRL methods: HB-GCN [16] and HB-GCL [38]; and (2) one HB-HRL method: THINK [11].

Since THINK is designed for temporal hypergraphs, we use its static variant without temporal information. For methods originally designed for graphs, we follow the experimental settings of prior work [36]. Specifically, for GGD, we replace its original feature and topology augmentations with the augmentation strategy used in HypeBoy [36]. For GraphMAE2, MaskGAE, HB-GCN, and HB-GCL, we first convert each hypergraph into a graph via clique expansion and then apply the respective methods.

TABLE IV: Statistics of ten real-world hypergraphs used in our experiments
<table><tr><td></td><td>Citeseer</td><td>Cora</td><td>DBLP-A</td><td>Cora-CA</td><td>IMDB</td><td>Pubmed</td><td>DBLP-P</td><td>Aminer</td><td>MN-40</td><td>20News</td></tr><tr><td># of Nodes</td><td>1,458</td><td>1,434</td><td>2,591</td><td>2,388</td><td>3,939</td><td>3,840</td><td>41,302</td><td>20,201</td><td>12,311</td><td>16,242</td></tr><tr><td># of Hyperedges</td><td>1,079</td><td>1,579</td><td>2,690</td><td>1,072</td><td>2,015</td><td>7,963</td><td>22,263</td><td>8,052</td><td>12,311</td><td>100</td></tr><tr><td># of Classes</td><td>6</td><td>7</td><td>4</td><td>7</td><td>3</td><td>3</td><td>6</td><td>12</td><td>40</td><td>4</td></tr><tr><td># of Features</td><td>3,703</td><td>1,433</td><td>334</td><td>1,433</td><td>3,066</td><td>500</td><td>1,425</td><td>500</td><td>100</td><td>100</td></tr></table>

Target downstream tasks and evaluation protocols. To evaluate HyBridge across different downstream tasks, we consider node classification [34], [36] and hyperedge prediction [33], [36]. For node classification, following [34], [36], we randomly split nodes into training, validation, and test sets with ratios of 1%, 1%, and 98%, respectively, and report the mean and standard deviation of accuracy over 20 random splits. Each self-supervised method first learns node representations, after which an MLP classifier is trained on the learned representations [36].

For hyperedge prediction, following [34], [36], we randomly split hyperedges into training, validation, and test sets with ratios of 60%, 20%, and 20%, respectively, and report the average AUROC over 20 random splits. Negative hyperedges are generated using Size-Negative-Sampling [54]. For Euclideanbased methods, hyperedge representations are obtained by applying max-min pooling to the embeddings of nodes contained in each hyperedge [6], [36]. For hyperbolic-based methods, standard Euclidean pooling may not adequately characterize the spatial distribution of node embeddings within each hyperedge. Therefore, we incorporate the mean, maximum, and standard deviation of pairwise hyperbolic distances among nodes within each hyperedge, along with the logarithm of the hyperedge size, as additional hyperedge-level features.<sup>3</sup>

Reproducibility. We conduct all experiments on a Linux server equipped with an AMD 7960X processor, a 2TB NVMe SSD, 128GB of DDR5 memory, and an NVIDIA RTX A6000 GPU with 48GB of memory. For the competing methods, we use the source code provided by the authors of the original papers [3], [6]–[8], [10], [11], [25], [27]–[38], [53]. For each method, we use the best hyperparameter setting identified through grid search over the ranges suggested in the corresponding paper.

We tune the hyperparameters of HyBridge via grid search over the following ranges. The loss weights $\omega _ { n }$ and $\omega _ { h }$ are selected from $\{ 2 ^ { - 4 } , 2 ^ { - 3 } , \dots , 2 ^ { 4 } \}$ , while $\omega _ { \mathrm { b r i d g e } }$ is selected from $\{ 1 0 ^ { - 4 } , 5 \times 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 5 \times 1 0 ^ { - 3 } , \dots , 1 \}$ . The temperature parameters for the contrastive losses, $\tau _ { n }$ and $\tau _ { h } ,$ are selected from {0.1, 0.2, . . . , 1.0}. For hypergraph augmentation, the feature, incidence, and hyperedge masking rates are varied within {0.1, 0.2, . . . , 0.9}. The hidden dimension is selected from $\{ \bar { 2 } ^ { 6 } , 2 ^ { 7 } , \dots , 2 ^ { 1 0 } \}$ , and the learning rate is selected from $\{ 1 0 ^ { - 5 } , 5 \times 1 0 ^ { - 5 } , \hat { 1 0 } ^ { - 4 } , 5 \times 1 0 ^ { - 4 } , 1 0 ^ { - 3 } \}$ . The weight decay is tuned over $\{ 1 0 ^ { - 5 } , 5 \times 1 0 ^ { - 5 } , 1 0 ^ { - 4 } , 5 \times \mathrm { { i } } 0 ^ { - 4 } \}$

The number of sampled node pairs $| \mathcal { P } |$ is selected from $\{ 2 ^ { 1 2 } , 2 ^ { 1 3 } , \dots , 2 ^ { 1 7 } \}$ . For reproducibility, we provide the code at https://anonymous.4open.science/r/HyBridge/.

## B. Experimental Results

EQ1: Ablation study. We conduct an ablation study with three variants: V1 performs contrastive learning in Euclidean space; V2 performs contrastive learning in hyperbolic space; and V3 additionally incorporates bridge-guided learning in hyperbolic space. Thus, comparing V1 and V2 evaluates the effect of hyperbolic geometry, while comparing V2 and V3 assesses the additional contribution of bridge-guided learning.

Table V presents the results. For each dataset, the best and second-best performances are marked in bold and underline, respectively. AR denotes the average rank. We make two key observations. First, V2 outperforms V1 on some datasets but shows comparable or lower performance on others, indicating that hyperbolic geometry alone does not consistently improve hypergraph representation learning. Second, V3 achieves the best average rank, demonstrating that incorporating bridgeguided learning consistently improves the effectiveness of hyperbolic representations. These results show that hyperbolic geometry alone provides mixed benefits, whereas bridge guidance provides effective structural guidance for learning hyperbolic hypergraph representations.

EQ2: Performance comparison. We evaluate the performance of HyBridge against the 22 competing methods introduced in Section V-A. Table VI presents the node classification results. OOT (out of time) indicates that a method did not complete within 24 hours, while OOM (out of memory) indicates that a method exceeded the memory capacity of the 48GB GPU. We make three key observations. First, HyBridge consistently outperforms all competing methods, achieving the highest accuracy across all datasets and the lowest average rank (AR) of 1.0. Second, existing hyperbolic methods, including HB-GCN (AR: 13.2) and HB-GCL (AR: 13.6), generally perform worse than strong Euclidean HRL methods, such as HypeBoy (AR: 4.2) and TriCL (AR: 5.2). Third, THINK, which is originally designed for temporal hypergraphs, achieves an AR of 12.2 when evaluated using its static variant. Similar trends are observed for hyperedge prediction, as reported in Table VII.

EQ3: Bridge score design choice validation. We investigate the impact of using external-node label distributions, rather than within-hyperedge label distributions, for bridge score estimation in HyBridge. Specifically, we compare bridge scores estimated using the label distribution of nodes within the target hyperedge with those estimated using the label distribution of external nodes, as proposed in HyBridge. We vary only the node set used for bridge score estimation while keeping all other components and training settings unchanged. We evaluate the learned node representations on node classification. Table VIII presents the node classification results. The proposed approach using external-node label distributions achieves higher node classification accuracy across all datasets. These results suggest that using external-node label distributions enables HyBridge to better capture the bridging roles of individual hyperedges, thereby providing more effective structural guidance for hyperbolic representation learning.

TABLE V: Ablation study: Accuracy comparison of 3 variants of HyBridge with node classification on 10 real-world hypergraphs
<table><tr><td></td><td>Hyperbolic Bridge</td><td></td><td>Citeseer</td><td> $\mathbf { { C o r a } }$ </td><td>DBLP-A</td><td>Cora-CA</td><td>IMDB</td><td>Pubmed</td><td>DBLP-P</td><td>AMiner</td><td>MN-40</td><td>20News</td><td>A.R.</td></tr><tr><td>V1</td><td>X</td><td>X</td><td> $4 9 . 2 \pm 8 . 4$ </td><td> $6 5 . 8 \pm 6 . 2$ </td><td> ${ \underline { { 7 9 . 4 } } } \pm 2 . 7$ </td><td> $6 6 . 7 \pm 5 . 5$ </td><td> $5 1 . 5 \pm 3 . 2$ </td><td> $6 8 . 3 \pm 5 . 2$ </td><td> $8 7 . 2 \pm 0 . 6$ </td><td> $3 4 . 8 \pm 2 . 5$ </td><td> $\underline { { 9 4 . 8 \pm 1 . 3 } }$ </td><td> $7 7 . 3 \pm 1 . 0$ </td><td>2.7</td></tr><tr><td>V2</td><td>V</td><td>X</td><td> $5 9 . 5 \pm 8 . 5$ </td><td> $6 4 . 6 \pm 9 . 1$ </td><td> $7 8 . 8 \pm 4 . 4$ </td><td> $\underline { { 7 0 . 4 \pm 5 . 4 } }$ </td><td> ${ \underline { { 5 2 . 5 \pm 3 . 2 } } }$ </td><td> $\underline { { 7 4 . 0 \pm 5 . 1 } }$ </td><td> $\underline { { 8 9 . 0 \pm 0 . 6 } }$ </td><td> $3 5 . 3 \pm 5 . 2$ </td><td> $9 4 . 5 \pm 1 . 3$ </td><td> $\underline { { 7 8 . 9 \pm 0 . 5 } }$ </td><td>2.3</td></tr><tr><td>V3</td><td>V</td><td>V</td><td> ${ \bf 6 0 . 7 \pm 8 . 0 }$ </td><td> ${ \bf 6 7 . 2 \pm 6 . 7 }$ </td><td> ${ \bf 8 1 . 3 \pm 1 . 2 }$ </td><td> ${ \bf 7 1 . 6 \pm 4 . 3 }$ </td><td> ${ \pm } 3 . 7 \pm 2 . 9$ </td><td> $7 7 . 4 \pm 3 . 3$ </td><td> ${ \bf 8 9 . 1 \pm 0 . 4 }$ </td><td> ${ \bf 3 6 . 6 \pm 5 . 5 }$ </td><td> ${ \bf 9 5 . 1 \pm 1 . 1 }$ </td><td> ${ \bf 7 9 . 0 \pm 0 . 5 }$ </td><td>1.0</td></tr></table>

TABLE VI: Accuracy comparison of HyBridge and 22 competing methods with node classification on 10 real-world hypergraphs
<table><tr><td></td><td>二</td><td>Citeseer</td><td>Cora</td><td>DBLP-A</td><td>Cora-CA</td><td>IMDB</td><td>Pubmed</td><td>DBLP-P</td><td>Aminer</td><td>MN-40</td><td>20News</td><td>A.R.</td></tr><tr><td></td><td>MLP</td><td> $3 2 . 4 \pm 8 . 1$ </td><td> $2 9 . 4 \pm 5 . 3$ </td><td> $5 6 . 6 \pm 5 . 4$ </td><td> $3 6 . 0 \pm 4 . 7$ </td><td> $3 7 . 6 \pm 3 . 3$ </td><td> $6 2 . 8 \pm { 3 . 0 }$ </td><td> $7 4 . 6 \pm 0 . 8$ </td><td> $2 2 . 7 \pm 1 . 8$ </td><td> $8 8 . 5 \pm 1 . 5$ </td><td> $7 3 . 3 \pm 1 . 3$ </td><td>18.9</td></tr><tr><td></td><td>HGNN</td><td>38.1 ± 10.7</td><td> $4 5 . 1 \pm 9 . 1$ </td><td> $6 5 . 0 \pm 5 . 7$ </td><td> $4 4 . 3 \pm 8 . 0$ </td><td> $4 1 . 5 \pm 2 . 8$ </td><td> $7 0 . 8 \pm 3 . 8$ </td><td> $8 4 . 0 \pm 0 . 9$ </td><td> $2 9 . 7 \pm 2 . 0$ </td><td> $8 9 . 5 \pm 1 . 3$ </td><td> $6 7 . 1 \pm 2 . 0$ </td><td>13.8</td></tr><tr><td></td><td>HyperGCN</td><td> $3 3 . 8 \pm 7 . 3$ </td><td> $3 9 . 1 \pm 8 . 2$ </td><td> $6 4 . 3 \pm 6 . 3$ </td><td> $4 5 . 0 \pm 9 . 5$ </td><td> $4 0 . 6 \pm 3 . 1$ </td><td> $5 9 . 3 \pm 1 5 . 2$ </td><td> $8 3 . 6 \pm 1 . 6$ </td><td> $2 6 . 8 \pm 2 . 2$ </td><td> $5 7 . 8 \pm 1 1 . 6$ </td><td> $7 1 . 7 \pm 2 . 1$ </td><td>17.7</td></tr><tr><td></td><td>HNHN</td><td> $4 4 . 2 \pm 7 . 9$ </td><td> $5 2 . 5 \pm 7 . 3$ </td><td> $6 9 . 0 \pm 4 . 5$ </td><td> $5 3 . 1 \pm 7 . 4$ </td><td> $4 2 . 5 \pm 3 . 6$ </td><td> $6 9 . 1 \pm 3 . 4$ </td><td> $8 6 . 4 \pm 0 . 8$ </td><td> $3 2 . 1 \pm 2 . 7$ </td><td> $8 9 . 2 \pm 1 . 4$ </td><td> $7 4 . 4 \pm 1 . 1$ </td><td>9.7</td></tr><tr><td></td><td>UniGCN</td><td> $3 9 . 8 \pm 9 . 2$ </td><td> $4 3 . 1 \pm 6 . 6$ </td><td> $5 9 . 4 \pm 5 . 1$ </td><td> $4 6 . 3 \pm 6 . 6$ </td><td> $4 1 . 0 \pm 3 . 0$ </td><td> $6 7 . 6 \pm 5 . 6$ </td><td> $8 2 . 2 \pm 0 . 9$ </td><td> $2 9 . 8 \pm 1 . 8$ </td><td> $7 5 . 7 \pm 1 1 . 4$ </td><td> $6 7 . 4 \pm 2 . 5$ </td><td>16.1</td></tr><tr><td></td><td>UniGIN</td><td> $4 1 . 4 \pm 1 0 . 3$ </td><td> $4 5 . 5 \pm 7 . 9$ </td><td> $6 3 . 1 \pm 5 . 5$ </td><td> $4 9 . 2 \pm 6 . 9$ </td><td> $4 1 . 7 \pm 2 . 7$ </td><td> $7 0 . 4 \pm 4 . 9$ </td><td> $8 3 . 5 \pm 0 . 8$ </td><td> $3 0 . 8 \pm 1 . 2$ </td><td> $8 7 . 1 \pm 2 . 3$ </td><td> $6 8 . 1 \pm 2 . 4$ </td><td>13.2</td></tr><tr><td></td><td>UniGCN2</td><td> $3 9 . 6 \pm 1 0 . 7$ </td><td> $4 3 . 9 \pm 9 . 5$ </td><td> $5 8 . 7 \pm 7 . 9$ </td><td> $5 5 . 3 \pm 5 . 3$ </td><td> $4 1 . 6 \pm 3 . 9$ </td><td> $7 2 . 6 \pm 3 . 8$ </td><td> $8 6 . 1 \pm 0 . 6$ </td><td> $3 2 . 3 \pm 1 . 5$ </td><td> $7 9 . 7 \pm 6 . 0$ </td><td> $7 6 . 8 \pm 1 . 4$ </td><td>11.4</td></tr><tr><td>Euan</td><td>AllSet</td><td> $4 1 . 1 \pm 9 . 8$ </td><td> $4 9 . 0 \pm 8 . 1$ </td><td> $6 4 . 9 \pm 4 . 6$ </td><td> $5 3 . 6 \pm 8 . 2$ </td><td> $4 1 . 7 \pm 2 . 9$ </td><td> $7 4 . 1 \pm 5 . 0$ </td><td> $8 5 . 6 \pm 0 . 7$ </td><td> $2 9 . 8 \pm 3 . 2$ </td><td> $8 9 . 4 \pm 1 . 3$ </td><td> $7 7 . 2 \pm 0 . 9$ </td><td>9.4</td></tr><tr><td></td><td>ED-HNN</td><td>32.9 ± 11.6</td><td> $2 9 . 3 \pm 3 . 9$ </td><td> $5 8 . 3 \pm 5 . 2 $ </td><td> $3 6 . 3 \pm 8 . 7$ </td><td> $3 7 . 2 \pm 2 . 6$ </td><td>61.9 ± 3.3</td><td>85.1 ± 1.1</td><td> $2 7 . 1 \pm 2 . 8$ </td><td> $7 3 . 0 \pm 3 . 8$ </td><td> $7 5 . 7 \pm 1 . 2$ </td><td>18.0</td></tr><tr><td></td><td>PhenomNN</td><td> $4 2 . 2 \pm 9 . 0$ </td><td> $5 5 . 8 \pm 8 . 0$ </td><td> $7 0 . 3 \pm 5 . 3$ </td><td> $5 6 . 2 \pm 6 . 6$ </td><td> $4 2 . 1 \pm 2 . 6$ </td><td> ${ 7 6 . 8 \pm 3 . 4 }$ </td><td>0.0.M</td><td>0.0.M</td><td> $9 4 . 0 \pm 0 . 5$ </td><td>0.0.T</td><td>7.0</td></tr><tr><td></td><td>GraphMAE2</td><td> $5 1 . 7 \pm 1 2 . 2$ </td><td> $6 1 . 0 \pm 9 . 1$ </td><td> $7 7 . 2 \pm 2 . 4$ </td><td> $6 4 . 3 \pm 6 . 3$ </td><td> $4 5 . 6 \pm 3 . 1$ </td><td> $7 2 . 6 \pm 4 . 4$ </td><td> $8 7 . 4 \pm 0 . 4$ </td><td> $3 4 . 7 \pm 2 . 3$ </td><td> $9 0 . 6 \pm 1 . 2$ </td><td> $7 1 . 8 \pm 0 . 4$ </td><td>6.2</td></tr><tr><td></td><td>MaskGAE</td><td> $5 3 . 2 \pm 1 0 . 8$ </td><td> $5 5 . 8 \pm 9 . 0$ </td><td> $7 8 . 2 \pm 2 . 7$ </td><td> $5 9 . 8 \pm 4 . 1$ </td><td> $4 5 . 0 \pm 3 . 0$ </td><td> $7 5 . 4 \pm 3 . 2$ </td><td> $8 6 . 9 \pm 0 . 5$ </td><td> $3 3 . 8 \pm 1 . 7$ </td><td> $9 1 . 0 \pm 0 . 8$ </td><td>0.0.T</td><td>5.6</td></tr><tr><td></td><td>GGD</td><td> $3 4 . 0 \pm 5 . 8$ </td><td> $3 7 . 2 \pm 6 . 7$ </td><td> $5 8 . 8 \pm 6 . 5$ </td><td> $3 2 . 2 \pm 6 . 3$ </td><td> $3 7 . 6 \pm 3 . 2$ </td><td> $6 4 . 9 \pm 4 . 4$ </td><td> $5 0 . 5 \pm 6 . 4$ </td><td> $3 1 . 5 \pm 1 . 7$ </td><td> $7 6 . 8 \pm 2 . 0$ </td><td> $4 7 . 5 \pm 7 . 9$ </td><td>18.3</td></tr><tr><td></td><td>TriCL</td><td> $5 3 . 0 \pm 9 . 9$ </td><td> $6 1 . 5 \pm 5 . 3$ </td><td> $8 0 . 3 \pm 3 . 5$ </td><td> $6 3 . 4 \pm 3 . 9$ </td><td> $4 7 . 5 \pm 4 . 6$ </td><td> $7 4 . 0 \pm 5 . 2$ </td><td> $8 7 . 6 \pm 0 . 5$ </td><td> $3 4 . 6 \pm 2 . 7$ </td><td> $9 3 . 0 \pm 0 . 8$ </td><td> $7 3 . 8 \pm 1 . 8$ </td><td>5.2</td></tr><tr><td></td><td>HyperGCL</td><td> $4 3 . 4 \pm 8 . 4$ </td><td> $5 4 . 8 \pm 5 . 6$ </td><td> $6 9 . 5 \pm 3 . 7$ </td><td> $6 1 . 8 \pm 3 . 0$ </td><td> $4 8 . 4 \pm 2 . 5$ </td><td> $7 1 . 0 \pm 2 . 2$ </td><td>0.0.M</td><td> $3 0 . 4 \pm 1 . 8$ </td><td> $9 4 . 6 \pm 1 . 2$ </td><td> $7 4 . 0 \pm 1 . 9$ </td><td>8.0</td></tr><tr><td></td><td>HyperGRL</td><td> $3 5 . 1 \pm 7 . 6$ </td><td> $3 7 . 0 \pm 5 . 3$ </td><td> $4 1 . 1 \pm 5 . 1$ </td><td> $4 1 . 8 \pm 5 . 8$ </td><td> $3 5 . 7 \pm 4 . 2$ </td><td> $5 0 . 2 \pm 8 . 7$ </td><td> $7 8 . 7 \pm 1 . 2$ </td><td> $2 8 . 0 \pm 2 . 8$ </td><td> $8 9 . 4 \pm 1 . 5$ </td><td>0.0.T</td><td>18.7</td></tr><tr><td></td><td>HypeBoy</td><td> $5 7 . 7 \pm 1 0 . 8$ </td><td> $6 1 . 9 \pm 8 . 5$ </td><td> $8 1 . 2 \pm 2 . 7$ </td><td> $6 7 . 0 \pm 3 . 7$ </td><td> $4 8 . 3 \pm 5 . 0$ </td><td> $7 3 . 7 \pm 4 . 1$ </td><td> $\underline { { 8 7 . 9 \pm 0 . 5 } }$ </td><td> $3 4 . 6 \pm 2 . 8$ </td><td> $8 9 . 2 \pm 2 . 2$ </td><td> $7 5 . 7 \pm 2 . 1$ </td><td>4.2</td></tr><tr><td></td><td>VilLain</td><td> $3 3 . 6 \pm 9 . 4$ </td><td> $4 7 . 1 \pm 8 . 4$ </td><td> $4 0 . 4 \pm 3 . 9$ </td><td> $3 1 . 4 \pm 4 . 1$ </td><td> $3 9 . 7 \pm 4 . 4$ </td><td> $7 3 . 7 \pm 3 . 3$ </td><td> $6 9 . 0 \pm 0 . 9$ </td><td> $1 9 . 9 \pm 1 . 2$ </td><td> $6 9 . 7 \pm 2 . 3$ </td><td> $7 4 . 3 \pm 3 . 0$ </td><td>17.7</td></tr><tr><td></td><td>SE-HSSL</td><td> $5 7 . 0 \pm 8 . 2 $ </td><td> $6 2 . 5 \pm 6 . 2$ </td><td> $7 7 . 1 \pm 2 . 7$ </td><td> $6 3 . 9 \pm 5 . 0$ </td><td> $4 8 . 9 \pm 4 . 1$ </td><td> $6 9 . 8 \pm 2 . 8$ </td><td> $8 3 . 6 \pm 0 . 9$ </td><td> $3 4 . 8 \pm 4 . 2$ </td><td> $8 5 . 8 \pm 5 . 2$ </td><td> $7 5 . 8 \pm 1 . 0$ </td><td>6.9</td></tr><tr><td></td><td>HB-GCN</td><td> $3 6 . 8 \pm 7 . 1$ </td><td> $4 1 . 4 \pm 5 . 7$ </td><td> $6 4 . 3 \pm 3 . 5$ </td><td> $5 2 . 7 \pm 6 . 7$ </td><td> $4 0 . 1 \pm 1 . 8$ </td><td> $7 1 . 9 \pm 4 . 0$ </td><td> $8 5 . 7 \pm 0 . 8$ </td><td> $3 0 . 3 \pm 1 . 1$ </td><td> $8 6 . 0 \pm 2 . 1$ </td><td> $7 4 . 6 \pm 0 . 8$ </td><td>13.2</td></tr><tr><td>Hypeolie</td><td>HB-GCL</td><td> $5 7 . 3 \pm 2 . 6$ </td><td> $5 7 . 2 \pm 3 . 7$ </td><td> $4 8 . 7 \pm 5 . 1 $ </td><td> $6 3 . 6 \pm 1 . 9$ </td><td> $4 4 . 5 \pm 2 . 4$ </td><td> $6 5 . 7 \pm 3 . 7$ </td><td>0.0.M</td><td> $2 5 . 7 \pm 1 . 7$ </td><td> $6 1 . 4 \pm 4 . 7$ </td><td> $6 1 . 0 \pm 3 . 9$ </td><td>13.6</td></tr><tr><td></td><td>THINK</td><td> $4 2 . 5 \pm 8 . 7$ </td><td> $4 5 . 0 \pm 8 . 5$ </td><td> $5 5 . 6 \pm 4 . 6$ </td><td> $4 6 . 9 \pm 5 . 2$ </td><td> $4 1 . 2 \pm 4 . 2$ </td><td> $7 4 . 0 \pm 4 . 4$ </td><td> $8 5 . 0 \pm 1 . 0$ </td><td> $2 9 . 9 \pm 2 . 6$ </td><td> $8 6 . 8 \pm 1 . 8$ </td><td> $7 6 . 7 \pm 1 . 3$ </td><td>12.2</td></tr><tr><td></td><td> $\mathbf { H y B r i d g e }$ </td><td> ${ \bf 6 0 . 7 \pm 8 . 0 }$ </td><td> ${ \bf 6 7 . 2 \pm 6 . 7 }$ </td><td> ${ \bf 8 1 . 3 \pm 1 . 2 }$ </td><td> ${ \bf 7 1 . 6 \pm 4 . 3 }$ </td><td> ${ \pm 3 . 7 \pm 2 . 9 }$ </td><td> $7 7 . 4 \pm 3 . 3$ </td><td> ${ \bf 8 9 . 1 \pm 0 . 4 }$ </td><td> ${ \bf 3 6 . 6 \pm 5 . 5 }$ </td><td> ${ \bf 9 5 . 1 \pm 1 . 1 }$ </td><td> ${ \bf 7 9 . 0 \pm 0 . 5 }$ </td><td>1.0</td></tr><tr><td></td><td>Improvement</td><td> $+ 5 . 2 0 \%$ </td><td> $+ 7 . 5 2 \%$ </td><td> $+ 0 . 1 2 \%$ </td><td>+6.87%</td><td> $+ 9 . 8 2 \%$ </td><td> $+ 0 . 7 8 \%$ </td><td>+1.37%</td><td>+5.17%</td><td>+0.53%</td><td>+2.33%</td><td></td></tr></table>

![](images/7799ea85d272389f85d5b01d7b92d200ef5fc39a5574d6a95096bfc471e08a6d.jpg)  
Fig. 6: Label entropy and homophily across hyperbolic radius bins after applying bridge loss.

EQ4: Bridge-guided embedding radius analysis. As discussed in Section III-D, HB-HRL does not exhibit consistent relationships between node radial positions and neighborhood label diversity. To investigate whether bridge-guided learning addresses this limitation, we examine the radial patterns of the node representations learned by HyBridge in terms of label entropy and homophily. Specifically, following the same procedure as in Section III-D, we divide nodes into ten groups according to their radial positions and compute the average neighborhood label entropy and homophily within each group.

homophily, where the x-axis represents the ten node groups ordered from the center to the periphery and the y-axis represents the average label entropy or homophily within each group. Overall, label entropy tends to decrease while homophily tends to increase from the center toward the periphery. Despite some fluctuations across datasets, these trends indicate that bridge-guided learning encourages a more consistent center– periphery organization, positioning nodes with more diverse neighborhood label distributions closer to the center.

Fig. 6 presents the radial patterns of label entropy and

EQ5: Parameter sensitivity. To investigate the learning stability of HyBridge, we examine its sensitivity to key hyperparameters associated with (1) bridge-guided learning, including the bridge loss weight $\omega _ { \mathrm { b r i d g e } }$ and the number of sampled node pairs $| { \mathcal { P } } | ,$ and (2) contrastive learning, including the augmentation ratios for feature masking and membership masking. For bridge-guided learning, we select $\omega _ { \mathrm { b r i d g e } }$ and |P| from the sets $\{ 1 0 ^ { - 4 } , 5 \times 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 5 \times 1 0 ^ { - 3 } , \hat { . . . } , 1 \}$ and $\{ 2 ^ { 1 2 } , 2 ^ { 1 3 } , \ldots , 2 ^ { \tilde { 1 } 7 } \}$ , respectively.

TABLE VII: AUROC comparison of HyBridge and 22 competing methods with hyperedge prediction on 10 real-world hypergraphs
<table><tr><td></td><td></td><td>Citeseer</td><td>Cora</td><td>DBLP-A</td><td>Cora-CA</td><td>IMDB</td><td>Pubmed</td><td>DBLP-P</td><td>Aminer</td><td>MN-40</td><td>20News</td><td>A.R.</td></tr><tr><td></td><td>MLP</td><td> $6 5 . 6 \pm 2 . 0$ </td><td> $7 7 . 5 \pm 1 . 5$ </td><td> $7 4 . 2 \pm 1 . 5$ </td><td> $5 8 . 6 \pm 2 . 9$ </td><td> $4 2 . 0 \pm 2 . 2$ </td><td> $8 8 . 3 \pm 0 . 6 $ </td><td> $9 0 . 2 \pm 0 . 5$ </td><td> $8 2 . 3 \pm 1 . 0$ </td><td> $9 4 . 9 \pm 8 . 4$ </td><td> $9 5 . 1 \pm 3 . 9$ </td><td>12.0</td></tr><tr><td></td><td>HGNN</td><td> $5 4 . 8 \pm 4 . 1$ </td><td> $5 7 . 6 \pm 6 . 0$ </td><td> $6 4 . 7 \pm 1 . 3$ </td><td> $6 5 . 0 \pm 4 . 6$ </td><td> ${ \bf 6 0 . 5 \pm 1 . 9 }$ </td><td> $6 5 . 9 \pm 2 . 1$ </td><td> $6 1 . 0 \pm 8 . 4$ </td><td> $6 9 . 6 \pm 1 . 2$ </td><td> $5 0 . 2 \pm 0 . 6$ </td><td> $4 9 . 4 \pm 8 . 2$ </td><td>15.5</td></tr><tr><td></td><td>HyperGCN</td><td> $5 4 . 9 \pm 2 . 1$ </td><td> $6 4 . 9 \pm 2 . 0$ </td><td> $6 6 . 5 \pm 1 . 2$ </td><td> $4 9 . 3 \pm 1 . 5$ </td><td> $4 8 . 4 \pm 1 . 9$ </td><td> $7 4 . 1 \pm 0 . 6$ </td><td> $5 3 . 9 \pm 2 . 1$ </td><td> $5 0 . 8 \pm 2 . 1$ </td><td> $6 0 . 9 \pm 3 . 0$ </td><td> $5 5 . 6 \pm 1 5 . 6$ </td><td>17.3</td></tr><tr><td></td><td>HNHN</td><td> $5 3 . 7 \pm 2 . 5$ </td><td> $5 9 . 0 \pm 1 . 8$ </td><td> $6 2 . 4 \pm 1 . 9$ </td><td> $5 1 . 6 \pm 1 . 4$ </td><td> $4 8 . 2 \pm 0 . 7$ </td><td>65.7 ± 2.4</td><td> $5 1 . 2 \pm 1 . 6$ </td><td> $4 8 . 1 \pm 1 . 3$ </td><td> $6 1 . 0 \pm 4 . 3$ </td><td> $5 6 . 0 \pm 9 . 7$ </td><td>18.6</td></tr><tr><td></td><td>UniGCN</td><td> $5 4 . 7 \pm 3 . 2$ </td><td> $5 1 . 6 \pm 2 . 0$ </td><td> $5 1 . 8 \pm 2 . 1$ </td><td> $6 0 . 0 \pm 5 . 9$ </td><td> $5 3 . 7 \pm 4 . 1$ </td><td> $5 2 . 1 \pm 3 . 2$ </td><td> $6 7 . 0 \pm 1 . 5$ </td><td> $6 9 . 0 \pm 1 . 3$ </td><td> $5 1 . 2 \pm 2 . 4$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td>18.2</td></tr><tr><td></td><td>UniGIN</td><td> $5 2 . 9 \pm 3 . 0$ </td><td> $6 3 . 1 \pm 3 . 7$ </td><td> $5 4 . 0 \pm 3 . 0$ </td><td> $5 8 . 7 \pm 8 . 2 $ </td><td> $5 4 . 6 \pm 4 . 2$ </td><td>51.5 ± 3.6</td><td> $6 9 . 2 \pm 0 . 6$ </td><td> $6 9 . 1 \pm 1 . 4$ </td><td> $5 0 . 4 \pm 1 . 7$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td>17.7</td></tr><tr><td></td><td>UniGCN2</td><td> $6 0 . 2 \pm 2 . 1$ </td><td> $6 3 . 5 \pm 3 . 5$ </td><td> $6 1 . 6 \pm 1 . 7$ </td><td> $5 1 . 9 \pm 4 . 3$ </td><td> $5 7 . 5 \pm 7 . 0$ </td><td> $6 9 . 1 \pm 1 . 2$ </td><td> $6 0 . 2 \pm 2 . 5$ </td><td> $4 5 . 9 \pm 4 . 9$ </td><td> $5 1 . 7 \pm 1 . 9$ </td><td> $5 0 . 0 \pm 0 . 0$ </td><td>16.6</td></tr><tr><td>Euean</td><td>AllSet</td><td> $5 1 . 8 \pm 3 . 2$ </td><td> $5 4 . 5 \pm 3 . 4$ </td><td> $6 0 . 5 \pm 3 . 0$ </td><td> $5 1 . 2 \pm 3 . 2$ </td><td> $5 0 . 3 \pm 1 . 0$ </td><td>52.3 ± 1.6</td><td> $5 3 . 2 \pm 1 . 5$ </td><td> $5 1 . 0 \pm 1 . 8$ </td><td> $5 6 . 4 \pm 1 . 6$ </td><td> $5 0 . 1 \pm 0 . 4$ </td><td>19.5</td></tr><tr><td></td><td>ED-HNN</td><td> $5 8 . 0 \pm 3 . 8 $ </td><td> $7 2 . 2 \pm 2 . 1$ </td><td> $6 1 . 9 \pm 6 . 4$ </td><td> $5 8 . 9 \pm 2 . 1 $ </td><td> $4 9 . 4 \pm 1 . 2$ </td><td> $5 3 . 2 \pm 2 . 7$ </td><td> $5 1 . 2 \pm 1 . 6$ </td><td> $5 6 . 0 \pm 3 . 7$ </td><td> $5 2 . 7 \pm 1 . 6$ </td><td> $5 1 . 4 \pm 3 . 5$ </td><td>17.2</td></tr><tr><td></td><td>PhenomNN</td><td> $5 4 . 5 \pm 2 . 7$ </td><td> $5 6 . 3 \pm 1 . 4$ </td><td> $6 4 . 6 \pm 2 . 0$ </td><td> $5 0 . 3 \pm 1 . 9$ </td><td> $4 9 . 7 \pm 0 . 9$ </td><td> $6 4 . 0 \pm 4 . 6$ </td><td>0.0.M</td><td> $_ { 0 . 0 . { \bf M } }$ </td><td> $6 3 . 5 \pm 8 . 6$ </td><td>0.0.T</td><td>18.4</td></tr><tr><td></td><td>GraphMAE2</td><td> $7 8 . 2 \pm 2 . 3$ </td><td> $8 2 . 0 \pm 1 . 4$ </td><td> $8 3 . 3 \pm 1 . 4$ </td><td> $7 4 . 7 \pm 2 . 7$ </td><td> $4 7 . 2 \pm 1 . 0$ </td><td> $9 3 . 6 \pm 2 . 6$ </td><td> $9 2 . 7 \pm 0 . 3$ </td><td> $7 8 . 1 \pm 0 . 8$ </td><td> $\underline { { 9 9 . 8 \pm 0 . 0 } }$ </td><td> $8 7 . 0 \pm 3 . 4$ </td><td>8.7</td></tr><tr><td></td><td>MaskGAE</td><td> $8 6 . 7 \pm 1 . 8$ </td><td> $8 6 . 7 \pm 1 . 6$ </td><td> $8 7 . 3 \pm 1 . 3$ </td><td> $7 6 . 5 \pm 2 . 1$ </td><td> $5 4 . 4 \pm 1 . 7$ </td><td> $9 5 . 4 \pm 0 . 3$ </td><td> $9 4 . 8 \pm 0 . 4$ </td><td> $8 7 . 6 \pm 0 . 9$ </td><td> ${ \bf 9 9 . 9 \pm 0 . 0 }$ </td><td>0.0.T</td><td>4.6</td></tr><tr><td></td><td>GGD</td><td> $7 2 . 2 \pm 5 . 0$ </td><td> $7 1 . 9 \pm 3 . 1$ </td><td> $8 1 . 4 \pm 1 . 9$ </td><td> $7 3 . 2 \pm 4 . 0$ </td><td> $5 3 . 1 \pm 1 . 8$ </td><td> $8 7 . 2 \pm 0 . 7$ </td><td> $9 1 . 6 \pm 1 . 0$ </td><td> $8 4 . 9 \pm 2 . 1$ </td><td> $9 9 . 5 \pm 0 . 1$ </td><td> $8 7 . 9 \pm 3 . 1$ </td><td>9.7</td></tr><tr><td></td><td>TriCL</td><td> $9 0 . 5 \pm 1 . 2$ </td><td> $9 0 . 7 \pm 1 . 3 $ </td><td> $8 7 . 9 \pm 1 . 4$ </td><td> $8 7 . 8 \pm 1 . 5$ </td><td> $5 8 . 9 \pm 2 . 1 $ </td><td> $9 1 . 9 \pm 0 . 5$ </td><td> $9 4 . 8 \pm 0 . 2$ </td><td> $9 0 . 4 \pm 0 . 6 $ </td><td> $9 9 . 6 \pm 0 . 1$ </td><td> $9 8 . 2 \pm 3 . 0 $ </td><td>3.3</td></tr><tr><td></td><td>HyperGCL</td><td> $7 3 . 9 \pm 2 . 6$ </td><td> $8 5 . 4 \pm 1 . 5$ </td><td> $8 3 . 5 \pm 1 . 0$ </td><td> $8 1 . 1 \pm 1 . 9$ </td><td> $5 3 . 8 \pm 2 . 4$ </td><td> $8 9 . 6 \pm 0 . 5$ </td><td> $8 3 . 6 \pm 0 . 6$ </td><td> $8 2 . 1 \pm 7 . 6$ </td><td> $9 9 . 4 \pm 0 . 1$ </td><td> $7 6 . 3 \pm 6 . 3$ </td><td>8.7</td></tr><tr><td></td><td>HyperGRL</td><td> $8 3 . 2 \pm 8 . 0$ </td><td> $8 4 . 0 \pm 1 . 7$ </td><td> $8 0 . 8 \pm 1 . 4$ </td><td> $8 0 . 0 \pm 3 . 3$ </td><td> $5 4 . 7 \pm 2 . 8$ </td><td> $7 8 . 3 \pm 1 . 8$ </td><td> $8 8 . 2 \pm 0 . 3 $ </td><td> $8 1 . 5 \pm 0 . 7$ </td><td> $9 8 . 8 \pm 0 . 4$ </td><td>0.0.T</td><td>9.1</td></tr><tr><td></td><td>VilLain</td><td> $5 8 . 3 \pm 2 . 7$ </td><td> $6 8 . 5 \pm 1 . 9$ </td><td> $5 7 . 3 \pm 2 . 0$ </td><td> $6 4 . 4 \pm 3 . 2$ </td><td> $4 3 . 9 \pm 2 . 4$ </td><td> $8 3 . 7 \pm 0 . 7$ </td><td> $5 6 . 3 \pm 0 . 7$ </td><td> $6 2 . 7 \pm 1 . 3$ </td><td> $7 9 . 8 \pm 0 . 7$ </td><td> $5 9 . 2 \pm 7 . 8$ </td><td>15.8</td></tr><tr><td></td><td>SE-HSSL</td><td> $9 0 . 5 \pm 1 . 1 $ </td><td> $9 1 . 9 \pm 1 . 3 $ </td><td> $8 7 . 4 \pm 1 . 2$ </td><td> $8 5 . 6 \pm 2 . 1$ </td><td> $5 5 . 6 \pm 2 . 0$ </td><td> $9 4 . 5 \pm 0 . 7$ </td><td> $9 4 . 0 \pm 0 . 4$ </td><td> $8 7 . 5 \pm 0 . 6$ </td><td> $9 9 . 8 \pm 0 . 1$ </td><td> $9 5 . 2 \pm 5 . 8$ </td><td>4.1</td></tr><tr><td></td><td>HypeBoy</td><td> $8 6 . 4 \pm 2 . 1$ </td><td> $9 1 . 1 \pm 1 . 2 $ </td><td> $8 7 . 0 \pm 1 . 2$ </td><td> $8 7 . 3 \pm 1 . 6$ </td><td> $5 9 . 4 \pm 1 . 9$ </td><td> $9 2 . 1 \pm 0 . 5$ </td><td> $9 5 . 2 \pm 0 . 2 $ </td><td> $8 9 . 7 \pm 0 . 6$ </td><td> $9 9 . 7 \pm 0 . 0$ </td><td> $9 7 . 2 \pm 3 . 2 $ </td><td>4.1</td></tr><tr><td>Hypperodoic</td><td>HB-GCN</td><td> $6 6 . 9 \pm 3 . 3$ </td><td> $7 6 . 0 \pm 4 . 1$ </td><td> $6 6 . 9 \pm 1 . 8$ </td><td> $7 4 . 7 \pm 3 . 9$ </td><td> $5 4 . 0 \pm 3 . 1$ </td><td> $9 1 . 8 \pm 0 . 5$ </td><td> $9 0 . 5 \pm 0 . 8$ </td><td> $6 9 . 7 \pm 3 . 8$ </td><td> $9 7 . 3 \pm 1 . 0$ </td><td> $8 4 . 0 \pm 7 . 1$ </td><td>10.2</td></tr><tr><td></td><td>HB-GCL</td><td> $8 9 . 0 \pm 1 . 8$ </td><td> $8 5 . 4 \pm 1 . 2$ </td><td> $8 1 . 8 \pm 1 . 1$ </td><td> $8 7 . 8 \pm 1 . 4$ </td><td> $5 3 . 6 \pm 3 . 2$ </td><td> $9 5 . 3 \pm 0 . 5$ </td><td> $8 1 . 8 \pm 1 . 1$ </td><td> $8 8 . 0 \pm 0 . 9$ </td><td> $9 9 . 6 \pm 0 . 1$ </td><td> $9 7 . 5 \pm 2 . 9$ </td><td>6.2</td></tr><tr><td></td><td>THINK</td><td> $6 2 . 5 \pm 6 . 7$ </td><td> $6 8 . 4 \pm 4 . 0$ </td><td> $6 5 . 7 \pm 3 . 1 $ </td><td> $6 3 . 8 \pm 6 . 9$ </td><td> $4 9 . 4 \pm 3 . 0$ </td><td> $7 4 . 3 \pm 3 . 7$ </td><td> $8 4 . 4 \pm 2 . 3$ </td><td> $6 4 . 4 \pm 8 . 2$ </td><td> $9 8 . 7 \pm 2 . 3 $ </td><td> $6 5 . 2 \pm 1 0 . 6$ </td><td>13.6</td></tr><tr><td></td><td>HyBridge</td><td> ${ \bf 9 1 . 7 \pm 1 . 1 }$ </td><td> ${ \bf 9 3 . 0 \pm 1 . 2 }$ </td><td> ${ \bf 8 9 . 0 \pm 0 . 9 }$ </td><td> ${ \bf 8 9 . 9 \pm 1 . 4 }$ </td><td> $6 0 . 5 \pm 2 . 7$ </td><td> ${ \bf 9 5 . 4 } \pm { \bf 0 . 2 }$ </td><td> ${ \bf 9 5 . 6 \pm 0 . 3 }$ </td><td> ${ \bf 9 0 . 4 \pm 0 . 5 }$ </td><td> ${ \bf 9 9 . 9 \pm 0 . 0 }$ </td><td> ${ \bf 9 8 . 8 \pm 1 . 6 }$ </td><td>1.1</td></tr></table>

<table><tr><td colspan="10">TABLE VIII: Node classification accuracy of HyBridge according to the label distributions used for bridge score estimation</td></tr><tr><td>Label distributionCiteseer</td><td></td><td>Cora</td><td>DBLP-A</td><td> $\mathbf { C o r a – C A }$ </td><td>IMDB</td><td>Pubmed</td><td>DBLP-P</td><td>Aminer</td><td>MN-40</td><td>20News</td><td>A.R.</td></tr><tr><td>neighborhood</td><td> $5 9 . 3 \pm 7 . 4$ </td><td> $6 5 . 0 \pm 8 . 0$ </td><td> $7 7 . 0 \pm 2 . 3 $ </td><td> $6 9 . 5 \pm 3 . 9$ </td><td> $5 3 . 3 \pm 3 . 2 $ </td><td> $7 6 . 6 \pm 3 . 8$ </td><td> $8 8 . 1 \pm 0 . 5$ </td><td> $3 6 . 2 \pm 5 . 8 $ </td><td> $9 5 . 1 \pm 1 . 2 $ </td><td> $7 8 . 1 \pm 0 . 7$ </td><td>2.0</td></tr><tr><td>external</td><td> ${ \bf 6 0 . 7 \pm 8 . 0 }$ </td><td> ${ \bf 6 7 . 2 \pm 6 . 7 }$ </td><td> ${ \bf 8 1 . 3 \pm 1 . 2 }$ </td><td> ${ \bf 7 1 . 6 \pm 4 . 3 }$ </td><td> ${ \pm 3 . 7 \pm 2 . 9 }$ </td><td> $7 7 . 4 \pm 3 . 3$ </td><td> ${ \bf 8 9 . 1 \pm 0 . 4 }$ </td><td> ${ \bf 3 6 . 6 \pm 5 . 5 }$ </td><td> ${ \bf 9 5 . 1 \pm 1 . 1 }$ </td><td> ${ \bf 7 9 . 0 \pm 0 . 5 }$ </td><td>1.0</td></tr></table>

![](images/c25523879905b3fe680a34f6396313a78fc3cc62814479d39102940e4820ca32.jpg)

![](images/ed19e0a46b8c6df57949606dfc4d44cf6b01ac55aa3915ab04cbfd6a31956b65.jpg)  
(a) Citeseer

![](images/b799218667f8bd1924c59a1da1da4b6f5ce7c0edf374a0f4ce3b0702dcd3c00e.jpg)  
(b) Cora

![](images/54c4847d32f84ec22975753e83e5e7820d46106f3f175bcbb05fb2c2f5648797.jpg)  
Fig. 7: Node classification accuracy of HyBridge under different bridge loss weights.  
(a) Citeseer  
(b) Cora  
Fig. 8: Node classification accuracy of HyBridge under different numbers of sampled node pairs.  
on bridge-guided learning can adversely affect performance.

Fig. 8 presents the node classification accuracy under different numbers of sampled node pairs on (a) Citeseer and (b) Cora, where the x-axis represents $| \mathcal { P } |$ and the y-axis represents node classification accuracy. Performance remains relatively stable once a sufficient number of node pairs is sampled. This suggests that HyBridge is relatively insensitive to $| \mathcal { P } |$ beyond this point.

Fig. 7 presents the node classification accuracy under different bridge loss weights on (a) Citeseer and (b) Cora, where the x-axis represents $\omega _ { \mathrm { b r i d g e } }$ and the y-axis represents node classification accuracy. $\mathbf { A s } \ \omega _ { \mathrm { b r i d g e } }$ increases, performance initially improves but begins to decline beyond a certain point. This indicates that an appropriate bridge loss weight is important for effective representation learning, while excessive emphasis

For contrastive learning, following [33], we select the feature masking and membership masking ratios from the set $\{ 0 . 1 , 0 . 2 , \ldots , 0 . 9 \}$ . Fig. 9 presents the differences in node classification accuracy between HyBridge and the second-best competing model Hypeboy under different combinations of augmentation ratios on (a) Citeseer and (b) Cora, where the xaxis and y-axis represent the feature masking and membership masking ratios, respectively. We make two key observations. First, excessively high feature masking ratios (0.8–0.9) lead to substantial performance degradation on both datasets, particularly on Cora, indicating that excessive feature removal can negatively affect representation learning. Second, the favorable feature masking ratios differ across datasets: Citeseer achieves better performance with ratios of 0.5–0.7, whereas Cora performs better with ratios of 0.2–0.3. These results highlight the importance of selecting appropriate augmentation ratios according to dataset characteristics.

![](images/b8e08255b04c5627025f320077dcfedac46b86f8a94ac1abbbb77220b028aca4.jpg)  
(a) Citeseer

![](images/2632f7d8b5883763c43421015cb762ac8cb6bc5688d680ec98c37bcbbd2a5ef3.jpg)  
(b) Cora

Fig. 9: Node classification accuracy of HyBridge under different masking ratio.  
![](images/a4811220d3babcb12d8215b0a67af0d1c18e249dfbe40cf96646fd0bac296082.jpg)  
Fig. 10: Preprocessing time of METIS and bridge score estimation.

EQ6: Computational efficiency and scalability. To assess the computational efficiency of HyBridge, we evaluate both the preprocessing overhead of Step 1 (bridge score estimation) and the training efficiency of the remaining steps. First, we measure the total preprocessing time, including pseudo-cluster label generation using METIS and node-level bridge score estimation. We evaluate the preprocessing time on Citeseer and Cora, as well as the two largest datasets, DBLP-P and AMiner. Fig. 10 presents the total preprocessing time, where the x-axis represents the datasets and the y-axis represents the preprocessing time. The total preprocessing time is less than one second on three datasets and reaches only about two seconds on DBLP-P, demonstrating the low computational overhead of the preprocessing steps.

Second, we compare the training efficiency of HyBridge with that of competing methods by measuring the average training time per epoch. Fig. 11 presents the relationship between training time and node classification accuracy on (a) Citeseer and (b) Cora, where the x-axis represents the average training time per epoch and the y-axis represents node classification accuracy. Although HyBridge is not the fastest method, it achieves a favorable trade-off between training time and classification accuracy. In particular, HyBridge requires less training time than existing contrastive HRL methods, such as TriCL and HyperGCL, while achieving higher accuracy on both datasets.

Finally, we empirically evaluate the scalability of HyBridge

![](images/7b86843a944b5f8828dae75fa6fda90a2346b84f9178f0c967570054948b3375.jpg)  
(a) Citeseer

![](images/c92eb5369e7c6f7c0b67a941a494fd6ecd0394b104ce2f7f916efb0c8934053f.jpg)  
(b) Cora

Fig. 11: The accuracy of HRL methods over the average training time per epoch (sec).  
![](images/8bfd51dd309592bc650a95614d6f3a15ce69e96a9eeb82af27e17eb71b1ea19c.jpg)  
(a) DBLP-P

![](images/123a35b522605431e2e514ad694d64138212961ec7013650f86b387dab112aac.jpg)  
(b) AMiner

Fig. 12: Training time per epoch of HyBridge as the number of hyperedges increases.

as the size of the input hypergraph increases. To this end, we train HyBridge for 20 epochs on each dataset while varying the proportion of hyperedges used for training from 10% to 100% in increments of 10%, and measure the average training time per epoch. Fig. 12 presents the results on the two largest datasets, (a) DBLP-P and (b) AMiner, where the x-axis represents the proportion of hyperedges used for training and the y-axis represents the average training time per epoch. The training time increases approximately linearly with the proportion of hyperedges on both datasets, indicating approximately linear empirical scaling over the evaluated range.

## VI. CONCLUSIONS

In this paper, we investigated the effectiveness of hyperbolic representation learning for hypergraphs and the structural guidance needed to effectively leverage hyperbolic space. Our preliminary studies revealed that, although hypergraphs exhibit strong tree-like characteristics, simply extending existing HRL methods to hyperbolic space does not consistently improve downstream performance. We further found that bridge scores provide effective structural guidance for organizing node representations along the radial dimension in graphs but do not directly generalize to hypergraphs due to their distinct groupwise connectivity, motivating HyBridge, a selfsupervised HB-HRL framework that estimates node bridge scores by explicitly accounting for the external connectivity of individual hyperedges and uses these scores to guide node radial positions together with contrastive learning. Experiments on ten real-world hypergraphs against 22 competing methods demonstrated the effectiveness of HyBridge across two downstream tasks (i.e., node classification and hyperedge prediction). Further analyses showed that HyBridge induces a more consistent center–periphery organization, incurs low preprocessing overhead, and exhibits favorable empirical scalability over the evaluated range.

[1] H. Wang, J. Wang, J. Wang, M. Zhao, W. Zhang, F. Zhang, X. Xie, and M. Guo, “Graphgan: Graph representation learning with generative adversarial nets,” in Proc. of AAAI Conf. on Artificial Intelligence (AAAI), 2018.

[2] T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” in Proc. of Int’l Conf. on Learning Representations (ICLR), 2017.

[3] Z. Hou, Y. He, Y. Cen, X. Liu, Y. Dong, E. Kharlamov, and J. Tang, “Graphmae2: A decoding-enhanced masked self-supervised graph learner,” in Proc. of ACM Web Conf. (WWW), 2023.

[4] Y. You, T. Chen, Y. Sui, T. Chen, Z. Wang, and Y. Shen, “Graph contrastive learning with augmentations,” in Proc. of the Conf. on Advances in Neural Information Processing Systems (NeurIPS), 2020.

[5] Y. Lee, N. Seo, K. Han, and S. Kim, “Asine: Adversarial signed network embedding,” in Proc. of ACM Int’l Conf. on Research and Development in Information Retrieval (SIGIR), 2020.

[6] N. Yadti, M. Nimishakavi, P. Yadav, V. Nitin, A. Louis, and P. Talukdar, “Hypergcn: A new method of training graph convolution networks on hypergraphs,” in Proc. of Conf. on Neural Information Processing Systems (NeurIPS), 2019.

[7] Y. Dong, W. Sawin, and Y. Bengio, “Hnhn: Hypergraph networks with hyperedge neurons,” in ICML Graph Representation Learning and Beyond Workshop, 2020.

[8] Y. Feng, H. You, Z. Zhang, R. Ji, and Y. Gao, “Hypergraph neural networks,” in Proc. of AAAI Conf. on Artificial Intelligence (AAAI), 2019.

[9] R. Zhang, Y. Zou, and J. Ma, “Hyper-sagnn: A self-attention based graph neural network for hypergraphs,” in Proc. of Int’l Conf. on Learning Representations (ICLR), 2020.

[10] Q. Liu, M. Nickel, and D. Kiela, “Hyperbolic graph neural networks,” in Proc. of the Conf. on Advances in Neural Information Processing Systems (NeurIPS), 2019.

[11] S. Agarwal, R. Sawhney, M. Thakkar, P. Nakov, J. Han, and T. Derr, “Think: Temporal hypergraph hyperbolic network,” in Proc. of IEEE Int. Conf. on Data Mining (ICDM), 2022.

[12] M. Nickel and D. Kiela, “Poincare embeddings for learning hierarchical´ representations,” in Proc. of the Conf. on Advances in Neural Information Processing Systems (NeurIPS), 2017.

[13] J. Dai, Y. Wu, Z. Gao, and Y. Jia, “A hyperbolic-to-hyperbolic graph convolutional network,” in Proc. of IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2021.

[14] R. Shimizu, Y. Mukuta, and T. Harada, “Hyperbolic neural networks++,” in Proc. of Int’l Conf. on Learning Representations (ICLR), 2021.

[15] C. Gulcehre, M. Denil, M. Malinowski, A. Razavi, R. Pascanu, K. M. Hermann, P. Battaglia, V. Bapst, D. Raposo, A. Santoro, and N. de Freitas, “Hyperbolic attention networks,” in Proc. ofInt’l Conf. on Learning Representations (ICLR), 2019.

[16] I. Chami, Z. Ying, C.Re, and J. Leskovec, “Hyperbolic graph convolu-´ tional neural networks,” in Proc. of the Conf. on Advances in Neural Information Processing Systems (NeurIPS), 2019.

[17] P. Jiao, Y. Liu, Y. Wang, and G. Zhang, “Cina: Curvature-based integrated network alignment with hypergraph,” in Proc. of IEEE Int. Conf. on Data Engineering (ICDE), 2024.

[18] S. Feng, L. Chen, K. Zhao, W. Wei, F. Li, and S. Shang, “Node2lv: Squared lorentzian representations for node proximity,” in Proc. of IEEE Int. Conf. on Data Engineering (ICDE), 2021.

[19] M. Gromov, “Hyperbolic groups,” in Essays in Group Theory, ser. Mathematical Sciences Research Institute Publications, S. M. Gersten, Ed. Springer, 1987, vol. 8, pp. 75–263.

[20] L. He, C.-T. Lu, J. Ma, J. Cao, L. Shen, and P. S. Yu, “Joint community and structural hole spanner detection via harmonic modularity,” in Proc. of ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining (KDD), 2016.

[21] O. Platonov, D. Kuznedelev, A. Babenko, and L. Prokhorenkova, “Characterizing graph datasets for node classification: Homophily-heterophily dichotomy and beyond,” in Proc. of the Conf. on Advances in Neural Information Processing Systems (NeurIPS), 2023.

[22] J. Zhu, R. A. Rossi, A. Rao, T. Mai, N. Lipka, N. K. Ahmed, and D. Koutra, “Graph neural networks with heterophily,” in Proc. of AAAI Conf. on Artificial Intelligence (AAAI), 2021.

[23] J. Wang, Y. Guo, L. Yang, and Y. Wang, “Understanding heterophily for graph neural networks,” in Proc. of Int. Conf. on Machine Learning (ICML), 2024.

[24] Z. Hua, S. Xu, J. Wang, J. Liu, and L. Mart´ınez-Lopez, “Bilevel´ consensus in large-scale group decision making: Integrating structural holes and community dynamics,” IEEE Trans. on Fuzzy Systems, vol. 34, no. 4, pp. 1282–1294, 2026.

[25] B. Du, C. Yuan, R. Barton, T. Neiman, and H. Tong, “Self-supervised hypergraph representation learning,” in Proc. of IEEE Int’l Conf. on Big Data (Big Data), 2022.

[26] P. Velickoviˇ c, G. Cucurull, A. Casanova, A. Romero, P. Li´ o, and\` Y. Bengio, “Graph attention networks,” in Proc. of Int’l Conf. on Learning Representations (ICLR), 2018.

[27] J. Li, R. Wu, W. Sun, L. Chen, S. Tian, L. Zhu, C. Meng, Z. Zheng, and W. Wang, “What’s behind the mask: Understanding masked graph modeling for graph autoencoders,” in Proc. of ACM Int’l Conf. on Knowledge Discovery and Data Mining (KDD), 2023.

[28] Y. Zheng, S. Pan, V. Lee, Y. Zheng, and P. Yu, “Rethinking and scaling up graph contrastive learning: An extremely efficient approach with group discrimination,” in Proc. of the Conf. on Advances in Neural Information Processing Systems Workshop on Self-Supervised Learning (NeurIPS 2nd SSL workshop), 2022.

[29] J. Huang and J. Yang, “Unignn: a unified framework for graph and hypergraph neural networks,” in Proc. of Int’l Joint Conf. on Artificial Intelligence (IJCAI), 2021.

[30] E. Chien, C. Pan, J. Peng, and O. Milenkovic, “You are allset: A multiset function framework for hypergraph neural networks,” in Proc. of Int’l Conf. on Learning Representations (ICLR), 2022.

[31] P. Wang, S. Yang, Y. Liu, Z. Wang, and P. Li, “Equivariant hypergraph diffusion neural operators,” in Proc. of Int’l Conf. on Learning Representations (ICLR), 2023.

[32] Y. Wang, Q. Gan, X. Qiu, X. Huang, and D. Wipf, “From hypergraph energy functions to hypergraph neural networks,” in Proc. of Int’l Conf. on Machine Learning (ICML), 2023.

[33] D. Lee and K. Shin, “I’m me, we’re us, and i’m us: Tri-directional contrastive learning on hypergraphs,” in Proc. ofAAAI Conf. on Artificial Intelligence (AAAI), 2023.

[34] T. Wei, Y. You, T. Chen, Y. Shen, J. He, and Z. Wang, “Augmentations in hypergraph contrastive learning: Fabricated and generative,” in Proc. of the Conf. on Advances in Neural Information Processing Systems (NeurIPS), 2022.

[35] F. Li, X. Wang, D. Cheng, W. Zhang, Y.Zhang, and X. Lin, “Hypergraph self-supervised learning with sampling-efficient signals,” in Proc. ofInt’l Joint Conf. on Artificial Intelligence (IJCAI), 2024.

[36] S. Kim, S. Kang, F. Bu, S. Lee, J. Yoo, and K. Shin, “Hypeboy: Generative self-supervised representation learning on hypergraphs,” in Proc. of Int’l Conf. on Learning Representations (ICLR), 2024.

[37] G. Lee, S. Lee, and K. Shin, “Villain: Self-supervised learning on homogeneous hypergraphs without features via virtual label propagation,” in Proc. of ACM Web Conf. (WWW), 2024.

[38] Y. Zhang, H. Zhu, M. Yang, J. Liu, R. Ying, I. King, and P. Koniusz, “Understanding and mitigating hyperbolic dimensional collapse in graph contrastive learning,” in Proc. of ACM Int’l Conf. on Knowledge Discovery and Data Mining (KDD), 2025.

[39] J. Liu, M. Yang, M. Zhou, S. Feng, and P. Fournier-Viger, “Enhancing hyperbolic graph embeddings via contrastive learning,” in Proc. of the Conf. on Advances in Neural Information Processing Systems (NeurIPS), 2021.

[40] K. Henderson, B. Gallagher, L. Li, L. Akoglu, T. Eliassi-Rad, H. Tong, and C. Faloutsos, “It’s who you know: Graph mining using recursive structural features,” in Proc. ofACM Int’l Conf. on Knowledge Discovery and Data Mining (KDD), 2011.

[41] K. Henderson, B. Gallagher, T. Eliassi-Rad, H. Tong, S. Basu, L. Akoglu, D. Koutra, C. Faloutsos, and L. Li, “Rolx: Structural role extraction and mining in large graphs,” in Proc. of ACM Int’l Conf. on Knowledge Discovery and Data Mining (KDD), 2012.

[42] D. J. Watts and S. H. Strogatz, “Collective dynamics of ‘small-world networks,” Nature, 1998.

[43] M. Kitsak, L. K. Gallos, S. Havlin, F. Liljeros, L. Muchnik, H. E. Stanley, and H. A. Makse, “Identification of influential spreaders in complex networks,” Nature Physics, 2010.

[44] G. Karypis and V. Kumar, “A fast and high quality multilevel scheme for partitioning irregular graphs,” SIAM J. on Scientific Computing, vol. 20, no. 1, pp. 359–392, 1998.

[45] V. D. Blondel, J.-L. Guillaume, R. Lambiotte, and E. Lefebvre, “Fast unfolding of communities in large networks,” Journal of Statistical Mechanics: Theory and Experiment, 2008.

[46] M. Rosvall and C. T. Bergstrom, “Maps of random walks on complex networks reveal community structure,” Proceedings of the National Academy of Sciences, 2008.

[47] J. Chun, S. Yoon, M. Choe, G. Lee, and K. Shin, “Attributed hypergraph generation with realistic interplay between structure and attributes,” in Proc. of IEEE Int. Conf. on Data Mining (ICDM), 2025.

[48] M. Li, Y. Gu, Y. Wang, Y. Fang, L. Bai, X. Zhuang, and P. Lio, “When hypergraph meets heterophily: New benchmark datasets and baseline,” in Proc. of AAAI Conf. on Artificial Intelligence (AAAI), 2025.

[49] X. Wang, H. Ji, C. Shi, B. Wang, Y. Ye, P. Cui, and P. Yu, “Heterogeneous graph attention network,” in Proc. of ACM Web Conf. (WWW), 2019.

[50] C. Zhang, D. Song, C. Huang, A. Swami, and N. Chawla, “Heterogeneous graph neural network,” in Proc. ofACM Int’l Conf. on Knowledge Discovery and Data Mining (KDD), 2019.

[51] Z. Wu, S. Song, A. Khosla, F. Yu, L. Zhang, X. Tang, and J. Xiao, “3d shapenets: A deep representation for volumetric shapes,” in Proc. of IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2015.

[52] M. Kelly, R. Longjohn, and K. Nottingham, “Uci machine learning repository,” 2017. [Online]. Available: https://archive.ics.uci.edu

[53] M. Gardner and S. Dorling, “Artificial neural networks (the multilayer perceptron)—a review of applications in the atmospheric sciences,” Atmospheric environment, 1998.

[54] P. Patil, G. Sharma, and M. Murty, “Negative sampling for hyperlink prediction in networks,” in Proc. of Pacific-Asia Conf. on Knowledge Discovery and Data Mining (PAKDD), 2020.