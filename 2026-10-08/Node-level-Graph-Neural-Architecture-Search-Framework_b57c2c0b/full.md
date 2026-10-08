# Node-level Graph Neural Architecture Search Framework

Lintao Yang<sup>a,1</sup>, Sirui Li<sup>a,1</sup>, Yaqing Wang<sup>a</sup>, Pietro Liò<sup>b</sup>, Xu Shen<sup>c</sup>, Baisong Liu<sup>a</sup> and Chengbin Peng<sup>a,∗</sup>

<sup>a</sup>College of Information Science and Engineering, Ningbo University, Ningbo, 315211, China

<sup>b</sup>Department of Computer Science and Technology, University of Cambridge, Cambridge, CB3 0FD, UK

<sup>c</sup>School of Artificial Intelligence, Jilin University, Changchun, 130015, China

## A R T I C L E I N F O

Keywords: Graph Neural Networks Graph Neural Architecture Search Graph Representation Learning

## A BS T RA C T

In recent years, Graph Neural Networks (GNNs) and architecture search frameworks have gained extensive application in non-Euclidean data processing, attributable to their superior capacity in managing unstructured data. Nevertheless, traditional approaches typically apply uniform convolution operations to all nodes, regardless of their varying structural and feature characteristics, which can undermine model performance and result in over-smoothing issues as the number of layers increases. To overcome this limitation, in this work, we propose a Node-Level Graph Neural Architecture Search (N-GNAS) algorithm. It can automatically choose an appropriate network architecture for each subset of nodes when updating node features. N-GNAS also introduces a contrastive learning loss to separate sample features from diferent categories and vice versa. In experiments conducted on eight datasets for node and graph classification, our methodology outperforms current leading GNAS techniques and traditional human-designed GNNs. For example, it achieves an accuracy rate of 78.26% on the CiteSeer dataset.

## 1. Introduction

Graphs are an essential representation of data structure that can describe various relationships and structures, which has been a pivotal area of investigation for many years. Graph neural networks (GNNs) [1, 2] are a kind of neural network model specially designed to deal with graph structure data. They extract features of graph data by learning relationships from nodes and edges. They can perform many graph learning tasks, such as node classification [3], link prediction [4, 5, 6], and graph classification [7].

Despite the success of GNNs [8, 9, 10, 11, 12, 13], huge challenges remain. For example, a specific graph neural architecture can usually only fit some graph learning tasks on certain datasets, and even for these datasets, lots of human expertise and trials are typically needed to optimize the graph neural architecture. Thus, it undoubtedly requires significant computational investments and professional skills.

Inspired by Neural Architecture Search (NAS) on CNNs [14], Graph Neural Architecture Search (GNAS) [15] techniques have been further proposed to search for the best GNN automatically. For example, in the beginning, Graph-NAS [16] pioneered this research field by using reinforcement learning to design graph neural architectures and proposing a novel weight-sharing strategy to reduce the number of parameters to efectively select graph neural network components such as attention functions and attention heads. For heterogeneous graphs, HGNAS [17] designs a new search space by utilizing popular heterogeneous graph neural networks such as RGCN [18] and MAGNN [19], and proposes diferent kinds of message encoding and aggregation methods. SANE [20] introduces a diferentiable gradient graph neural architecture search method [21] to explore feature aggregation strategies. ARGNP [22] introduces a relation-aware message passing strategy into the search space, to optimize the learning of node and relational information.

However, traditional GNAS methods usually perform a uniform operation over diferent nodes, which is inappropriate as nodes in a real-world graph can usually encompass various types with diverse characteristics. Meanwhile, performing the same number of graph neural network operations on each node may result in excessive averaging of node features across the entire graph, reducing diferentiation between diferent nodes and causing the problem of oversmoothing [23]. To address this challenge, in this work, we propose a node level Graph Neural Architecture Search (N-GNAS) framework, which can adaptively apply diferent GNN operations for diferent nodes and alleviate oversmoothing. Figure 1 illustrates the graph neural network model that has been generated using the N-GNAS algorithm.

This study represents the first efort to apply GNAS to optimize node-level operations within graph neural networks. The primary contributions of this research are outlined as follows:

1. We propose a node-level search framework that can automatically select the optimal graph neural network (GNN) architecture for diferent node subsets based on their unique characteristics. Unlike traditional GNN or GNAS approaches, which apply uniform convolution operations to all nodes, this is the first to achieve node-level optimization for more precise and efective learning.

![](images/7030705be74555457bac2bf5a15a96370c21a8c85edbec1daee385ef282339e7.jpg)  
Figure 1: An example of a graph neural network discovered by N-GNAS. Part I illustrates an �-layer architecture. Part II demonstrates the search space across three stages, and Part III provides an example of search outcomes. Each layer comprises two stages: node selection operation search and graph neural network operation search, denoted by the symbols � and � with subscripts reflecting their diversity. Every layer comprises an input cell, an output cell, and operation cells. In this example, seven cells and five operation are used in the first and the second stage respectively. The variable � denotes the respective features obtained from these cells, with subscripts distinguishing stages and superscripts defining layers and cells. Links indicate possible operations.

2. We propose a contrastive learning loss into GNAS to separate sample features from diferent categories and vice versa, which can also efectively alleviate the over-smoothing issue commonly observed in traditional deep GNNs.

3. Extensive evaluations across diverse graph learning tasks have demonstrated that our approach surpasses conventional GNAS and manually crafted GNN methods in performance.

## 2. Related Work

## 2.1. Recent Advances In Graph Neural Networks

Graph Neural Networks (GNNs) is a type of deep learning model used for processing graph-structured data [2, 24, 25, 7] that can learn representations for nodes and graphs and can perform various tasks such as node classification, graph classification, link prediction, and so on. GNN operations can aggregate node information, and GNNs can be divided into two categories: spectral-based GNNs [2] and spatialbased GNNs [25].

Spectral-based GNNs learn node representations by spectral decomposition of graphs. Specifically, for a graph, the eigen-decomposition of its Laplacian matrix is computed, and these eigenvectors are then used as node representations. Deferrard et al. [26] approximate spectral filters using Chebyshev polynomials to reduce the complexity of matrix decomposition. GCN [2] optimized upon this by simplifying the previous method through local first-order approximations. JK-Net [27] improves node representations by adaptively aggregating nodes from diferent locations. SGC [28] simplifies GCN by removing non-linear activation functions and feature extraction as a feature preprocessing operation.

Spatial-based GNNs directly utilize the neighborhood information to learn node representations, which update node features simply by aggregating information within the local neighborhood of each node and do not require expensive graph spectral decomposition. For example, Graph-SAGE [24] is a framework that leverages both node feature information and node structural information. The feature information is sampled and aggregated from the local neighborhood of nodes to obtain node embeddings. The structural information is learned by employing trainable aggregation functions. GAT [25] introduces a masked selfattention mechanism to assign diferent weights for diferent node representations. GatedGCN [29] replaces the recursive definition of GNNs with gated edges and residuality, employing gate mechanisms to determine whether information should propagate from one node to another and utilizing residual connections to enable the network to learn features of graph data at deeper levels.

## 2.2. Graph Neural Architecture Search

The objective of NAS is to autonomously identify the most efective neural architecture tailored for specific datasets through a systematic optimization process [14]. Traditionally, the design of neural network architectures is conducted manually, a process that is not only laborious but also timeconsuming [21]. To overcome this limitation, NAS can automatically discover suitable network architectures for predefined tasks. Similar to NAS on CNNs and RNNs, NAS on graph has been proposed [20, 22]. For example, there are GNAS based on reinforcement learning (RL) [14, 16, 30], GNAS based on evolutionary algorithms (EA) [31, 32, 33, 34], GNAS based on gradient descent (GD) [21, 20, 35, 36], etc. These algorithms can not only automatically search the network structure but also avoid the "subjective" bias of the designer for the network structure.

In search space design, prevailing approaches are generally categorized into two main paradigms: componentbased and architecture-based methods [18]. The componentbased search method targets a predefined set of components, such as aggregation operations, their respective weights, and combination mechanisms. It aims to optimize each module by fine-tuning its attributes and configurations. In contrast, architecture-based search involves a broader exploration scope, including elements like network depth and integrating diferent modules, to identify the optimal configuration for the entire network’s structure [37]. Futhermore, DSS [38] further improves search eficiency by adaptively pruning the search space during the search process.

Nevertheless, these approaches overlook the diversity of node attributes and solely rely on cross-entropy as the loss function. Unlike heterogeneous GNN methods [18, 19, 17] that rely on predefined type information and manual metapaths, our approach automatically determines nodelevel operations from learned features and structures without requiring any type annotations, making it applicable to both homogeneous and heterogeneous graphs in a fully datadriven manner. Based on these insights, we propose an approach for the node-level graph neural architecture search.

## 3. METHODOLOGY

## 3.1. Notations

In this work, we define � as a graph of � nodes, with � representing the set of nodes within �. We denote the node features by $\boldsymbol { \mathcal { X } } \in \mathbb { R } ^ { n \times d _ { V } }$ , where $d _ { V }$ indicates the dimension of node features. Specifically, the �-th row of , corresponding to node � is denoted as $x _ { u }$ . We use $N ( v )$ to denote the onehop neighbors of a node � in �. We use $A \in \mathbb { R } ^ { n \times n }$ and $D \in$ $\mathbb { R } ^ { n \times n }$ to represent the adjacency matrix and the degree matrix of this graph. Thus, the structural relationships and the node features in graph � can be represented by an adjacency matrix � and a feature matrix .

## 3.2. Search Space Architecture Design

Traditional GNAS applies the same number of graph neural networks to all nodes in the graph, which cannot adapt well to the diversified characteristics of diferent nodes, thus limiting the expressive power of the graph neural network model. In this work, we propose applying varying numbers of graph neural networks to diferent nodes based on their characteristics and needs. For example, by considering node features, positions, and local structures, we tailor the number of graph neural networks to enhance the model’s expressive capacity.

Without loss of generality, we assume that the network architecture consists of � layers. Each layer consists of node selection operations and graph neural network operations, except that Layer 1 and Layer � are usually linear and classification layers, respectively. We use $\chi _ { i n } ^ { ( l ) }$ and $\mathcal { X } _ { o u t } ^ { ( l ) }$ to represent the input and the output of the �-th layer respectively. Thus, $\mathcal { X } _ { i n } ^ { ( \mathrm { i } ) }$ is the input graph and ${ \boldsymbol { \chi } } _ { o u t } ^ { ( L ) }$ is the network output, and $\mathscr { X } _ { i n } ^ { ( \ddot { l } ) } = \mathscr { X } _ { o u t } ^ { ( l - 1 ) }$ , for $l \in \{ 2 , 3 , \ldots , L \}$

Each layer encompasses two stages. The first is to select nodes for further processing, and the second is to search for optimal graph neural network operations for selected nodes. Hence, the search space can be represented by a directed acyclic graph (DAG). Cells are defined as the fundamental elements within the search space, and links represent the connections between these cells. For the �-th layer, a vector $\vec { X } _ { S } ^ { ( l , i ) } \in \mathbb { R } ^ { 1 \times n }$ to represents the node information score in the �-th cells corresponding to the �-th scoring strategy in the first stage, and a matrix $\mathcal { X } _ { G } ^ { \bar { ( l , i ) } } \in \mathbb { R } ^ { n \times n }$ represents the feature embedding corresponding to the �-th graph neural network operation in the second stage. There is a total of $n _ { 1 }$ and $n _ { 2 }$ cells for corresponding stages. In each stage, an operation depicted by a directed link $S ^ { ( l , i ) }$ or $G ^ { ( l , i ) }$ extracts features from its input cell and stores the results in its output cell. The framework is as shown in Figure 1.

## 3.2.1. Stage I: Node Selection Operations.

Before applying a graph neural network operation in each layer, we propose diferent scoring approaches and ensemble them to compute a soft gating vector, where each entry indicates the extent to which a node should undergo further GNN processing. Without loss of generality, for the �-th layer and the �-th scoring approach, node information score $\mathbf { \bar { \chi } } _ { S } ^ { ( l , i ) }$ are obtained after applying an operation $S ^ { ( l , i ) }$ on the input cell $\mathscr { X } _ { i n } ^ { ( l ) }$ . After assembling these scoring approaches, nodes with higher scores are assigned larger gate values, allowing them to contribute more to the subsequent GNN stage, while nodes with lower scores are mostly routed through residual connections.

Node scoring approaches consider node features, positions, and structures, and we discuss each in detail as follows.

(1) Feature Projection $f _ { S _ { 1 } }$ : From the perspective of node features, we calculate the score for each node by employing a learnable feature projection vector $W ^ { ( l ) }$ as follows:

$$
f _ { S _ { 1 } } ( \mathcal { X } _ { i n } ^ { ( l ) } ) = W ^ { ( l ) } \cdot \mathcal { X } _ { i n } ^ { ( l ) } .\tag{1}
$$

(2) Multi-layer Perceptron $f _ { S _ { 7 } }$ : We also employ a Multi-Layer Perceptron (���) to obtain a score vector for

the entire feature matrix for the score calculation as follows:

$$
f _ { S _ { 2 } } ( \mathcal { X } _ { i n } ^ { ( l ) } ) = M L P ( \mathcal { X } _ { i n } ^ { ( l ) } ) .\tag{2}
$$

(3) Local Position Score $f _ { S _ { 3 } }$ : Regarding node positions, we can consider local and global aspects. The local position score is computed by generating a random walk matrix that quantifies the likelihood of each node reaching other nodes within the graph. The score of each node is obtained by summing each column of the random walk matrix:

$$
M = D \cdot A ,\tag{3}
$$

$$
f _ { S _ { 3 } } ( \mathcal { X } _ { i n } ^ { ( l ) } ) = \left[ \sum _ { j } ^ { n } M _ { j , 1 } , \sum _ { j } ^ { n } M _ { j , 2 } , \cdots , \sum _ { j } ^ { n } M _ { j , n } \right] .\tag{4}
$$

(4) Global Position Score $f _ { S _ { 4 } }$ : The global position score is derived using the distance matrix $P ~ \in ~ \mathbb { R } ^ { n \times n }$ between nodes, in which each column indicates the distance from each node to every other node and can be computed by the Floyd–Warshall [39] algorithm or approximated by more eficient methods for large graphs. The score for each node is obtained by summing elements of each column in the matrix as follows:

$$
f _ { S _ { 4 } } ( \mathcal { X } _ { i n } ^ { ( l ) } ) = \left[ \sum _ { j } ^ { n } P _ { j , 1 } , \sum _ { j } ^ { n } P _ { j , 2 } , \cdots , \sum _ { j } ^ { n } P _ { j , n } \right] .\tag{5}
$$

(5) Local Structural Feature Score $f _ { S _ { 5 } }$ : We also use node structural features for scoring. Local structure information is directly obtained from the degree matrix as follows:

$$
f _ { S _ { 5 } } ( \mathcal { X } _ { i n } ^ { ( l ) } ) = \left[ D _ { 1 , 1 } , D _ { 2 , 2 } , \cdots , D _ { n , n } \right] .\tag{6}
$$

(6) Relative Structural Feature Score $f _ { S _ { 6 } }$ : Relative structural feature scores leverage node features from neighborhoods and are obtained by aggregating their feature projection scores $f _ { S _ { 1 } } ( \mathcal { X } _ { i n } ^ { ( l ) } )$ , rather than simply the number of neighbors as follows:

$$
\psi = f _ { S _ { 1 } } ( \mathcal { X } _ { i n } ^ { ( l ) } ) ,\tag{7}
$$

$$
f _ { S _ { 6 } } ( \mathcal { X } _ { i n } ^ { ( l ) } ) = \left\lfloor \sum _ { j \in N ( v _ { 1 } ) } \psi _ { j } , \sum _ { j \in N ( v _ { 2 } ) } \psi _ { j } , \cdots , \sum _ { j \in N ( v _ { n } ) } \psi _ { j } \right\rfloor\tag{8}
$$

(7) Attention-based Structural Feature Score $f _ { S _ { 7 } } \mathrm { : }$ Adaptive Structure Aware Pooling (����) [40] can compute node attentions to divide nodes in a graph into diferent local clusters. Thus, we propose to use this attention information as node scores as follows:

$$
f _ { S _ { 7 } } ( \mathcal { X } _ { i n } ^ { ( l ) } ) = A S A P ( \mathcal { X } _ { i n } ^ { ( l ) } ) .\tag{9}
$$

We use $\vec { X } _ { S } ^ { ( l , i ) }$ to represent the scores obtained from these scoring approaches:

$$
\vec { X } _ { S } ^ { ( l , i ) } = N o r m ( f _ { S _ { i } } ( \mathcal { X } _ { i n } ^ { ( l ) } ) ) , i \in \{ 1 , 2 , \ldots , 7 \} .\tag{10}
$$

An activation function � is applied on the ensemble of these scores to compute the node mask vector $\hat { X } _ { S } ^ { ( l ) } \in \mathbb { R } ^ { 1 \times n }$ of the �-th layer based on that from the last layer, namely, $\hat { X } _ { S } ^ { ( l - 1 ) }$

$$
\hat { X } _ { S } ^ { ( l ) } = \sigma ( \sum _ { i } \alpha _ { S } ^ { ( l , i ) } \vec { X } _ { S } ^ { ( l , i ) } ) \odot \hat { X } _ { S } ^ { ( l - 1 ) } ,\tag{11}
$$

where $\alpha _ { S } ^ { ( l , i ) }$ is a scalar representing the weight of the �-th operation, ⊙ is an element-wise multiplication operation, and $\sigma ( \cdot )$ is the sigmoid function ensuring each entry lies in (0, 1). Thus, $\hat { X } _ { S } ^ { ( l ) } \in ( 0 , 1 ) ^ { n }$ serves as a continuous soft gating vector of the �-th layer, where the �-th entry indicates the degree to which node � should be processed by subsequent GNN operations. A value close to 1 means the node is strongly selected, while a value close to 0 means its features mostly pass through via the residual connection in Eq. (13). This continuous relaxation keeps the entire framework fully diferentiable without requiring any hard discretization during training.

After obtaining $\hat { X } _ { S } ^ { ( l ) }$ , we zero out the corresponding columns in the feature matrix $\mathscr { X } _ { i n } ^ { ( l ) }$ row by row to obtain $\it \Delta \phi _ { S } ^ { ( l ) }$ for further processing:

$$
\mathcal X _ { S } ^ { ( l ) } = \hat { X } _ { S } ^ { ( l ) } \odot \mathcal X _ { i n } ^ { ( l ) } ,\tag{12}
$$

where broadcasting is used in ⊙ to extend a vector to match the dimensions required for matrix multiplication.

## 3.2.2. Stage II: Graph Neural Network Operations.

In the second stage, each cell also connects with the cell $\it \mathcal { X } _ { S } ^ { ( l ) }$ and stores the results obtained from diferent graph neural network operations. We can select the optimal graph neural network operation for the current layer by analyzing the final results, thus gaining advantages in graph neural architecture search.

Diferent graph neural network operations have distinct advantages over diferent tasks and datasets. In this work, several widely used GNNs are included in the search space to enhance the model performance, as shown in Table 1, where each operation $f _ { G _ { i } }$ is fed by a graph and the corresponding feature matrix, and returns the resulted feature matrix.

Therefore, the objective function can be expressed as follows:

$$
X _ { G } ^ { ( l , i ) } = f _ { G _ { i } } ( \mathcal { X } _ { S } ^ { ( l ) } ) + ( 1 - \hat { X } _ { S } ^ { ( l ) } ) \cdot \mathcal { X } _ { i n } ^ { ( l ) } ,\tag{13}
$$

where the term $( 1 - \hat { X } _ { S } ^ { ( l ) } ) \cdot \mathcal { X } _ { i n } ^ { ( l ) }$ acts as a residual connection: nodes with low gate values are directly carried over without further GNN processing, which helps alleviate

![](images/19cd505363e533b8fb1583cd569d39f3249346c0e89f314fcfd2bfd09db6d605.jpg)  
Figure 2: The specific process of N-GNAS in graph classification task. Select Node refers to the process of searching for node selection operations. In contrast, Search GNNs indicates the process of searching for graph neural network operations. A Readout operation is required for the result of each layer, and the Readout result is used as the output of the current layer.  
Table 1

Some popular graph neural network operation within our search space. Def. denotes diferent graph neural network operations. Operation denotes the corresponding concrete graph neural network operation respectively.

<table><tr><td>Def.</td><td>Operation</td></tr><tr><td> $f _ { G _ { 1 } }$  1</td><td>GCN [2]</td></tr><tr><td> $f _ { G _ { 2 } }$  1</td><td>GAT [25]</td></tr><tr><td> $f _ { G _ { 3 } }$ </td><td>GraphSAGE [24]</td></tr><tr><td> $f _ { G _ { 4 } }$  |</td><td>GIN [7]</td></tr><tr><td> $f _ { G _ { 5 } }$ </td><td>GatedGCN [29]</td></tr></table>

over-smoothing. Through node-level graph neural network operation selection, we can obtain the final node features $\boldsymbol { \chi } _ { G } ^ { ( l ) }$ for each layer:

$$
\mathcal { X } _ { G } ^ { ( l ) } = \sum _ { 1 \leq i < 5 } \alpha _ { G } ^ { ( l , i ) } X _ { G } ^ { ( l , i ) } .\tag{14}
$$

where $\alpha _ { G } ^ { ( l , i ) }$ denotes the scalar weight for the �-th operation.

## 3.2.3. Stage III: Readout Operations For Graph Classification Tasks.

Although for node classification tasks, node feature matrix $\boldsymbol { \chi } _ { G } ^ { ( l ) }$ can be directly used for subsequent classification tasks, for graph classification tasks, it is necessary to perform a Readout operation [41] to summarize the obtained feature matrix. In this stage, we discuss the latter case.

We include common Readout operations such as GLOBAL\_MEAN, GLOBAL\_MAX, GLOBAL\_SUM, GLOBAL\_SORT, GLOBAL\_ATT, and ZERO in the search space to explore the best Readout method for each layer, aiming to obtain better global graph features. An example of this procedure is depicted in Figure 2.

## 3.3. Search Strategy Design

In this part, we discuss the loss functions and the search strategies.

## 3.3.1. Loss Function

The loss functions are composed of two parts: a traditional cross-entropy (CE) fit of the training data and a proposed contrastive learning (CL) Loss to balance model performance and alleviate the problem of over-smoothing.

Cross-entropy (CE) Loss. The metric of cross-entropy [42] is frequently utilized in classification problems. When a model’s objective is to minimize this loss, it aligns the predictions more closely with the actual labels, thereby improving its predictive performance. They commonly gauge the disparity between model predictions and actual labels. By minimizing the cross-entropy loss, the model can better fit the training data and enhance accuracy in classification tasks. The cross-entropy loss in our context is defined as follows:

$$
L _ { c e } = - \frac { \sum _ { i = 1 } ^ { N } y _ { i } \log \widehat { y } _ { i } + ( 1 - y _ { i } ) \log ( 1 - \widehat { y } _ { i } ) } { N } ,\tag{15}
$$

where $y _ { i }$ and $\widehat { y } _ { i }$ represent the ground-truth label and the predicted label of sample $i ,$ respectively, � represents the number of nodes in node classification tasks or the number of graphs in graph classification tasks, respectively.

Contrastive Learning (CL) Loss. Recent studies highlight the importance of over-smoothing issues in model performance. Thus, besides the CE loss, we propose a contrastive learning loss. While such losses have been adapted in computer vision or GNN, they have seldom been used in GNAS. We propose contrasting the positive and negative samples with a sampling strategy to enhance node-level feature learning.

Specifically, we randomly select � samples from the training set and use $m _ { i }$ to represent the �th sample, which represents the node feature of the �-th node in node classification tasks or the global graph feature of the �-th graph in graph classification tasks. We use cosine similarity to measure the distance between sample � and sample � as follows:

$$
c o s ( m _ { i } , m _ { j } ) = \frac { m _ { i } \cdot m _ { j } } { | | m _ { i } | | \cdot | | m _ { j } | | } ,\tag{16}
$$

$$
s i m ( m _ { i } , m _ { j } ) = e x p ( \frac { c o s ( m _ { i } , m _ { j } ) } { \tau } ) ,\tag{17}
$$

where ���() is the exponential function, � is a temperature parameter. A pair of samples with the same ground-truth

label is considered a positive sample pair, and vice versa, so we can obtain the positive and the negative distances.

$$
p o s = \sum _ { i = 1 } ^ { M } \sum _ { j = 1 } ^ { M } \delta _ { y _ { i } = y _ { j } } s i m ( m _ { i } , m _ { j } ) ,\tag{18}
$$

$$
n e g = \sum _ { i = 1 } ^ { M } \sum _ { j = 1 } ^ { M } \delta _ { y _ { i } \neq y _ { j } } s i m ( m _ { i } , m _ { j } ) .\tag{19}
$$

where $\delta _ { y _ { i } = y _ { j } }$ equals 1 if $y _ { i } ~ = ~ y _ { j }$ , and $\delta _ { y _ { i } = y _ { j } }$ equals 0 otherwise. ��� denotes the contribution of positive samples, and ��� represents the contribution of negative samples.

Finally, the loss function is

$$
L _ { c l } = - \frac { p o s } { p o s + n e g } .\tag{20}
$$

It enhances the similarity between positive samples and reduces the similarity between negative samples.

Total Loss. Therefore, the final loss function � can be written as

$$
L = L _ { c e } + L _ { c l } .\tag{21}
$$

## 3.3.2. Search Strategy

Similar to numerous other GNAS methodologies, we utilize the Diferentiable Architecture Search (DARTS) [21, 20] to investigate the designed search space. By leveraging a continuous relaxation strategy, DARTS facilitates the conversion of the discrete architecture into a format amenable to gradient-based optimization. In our DAG-based search space, edges connect pairs of cells, and each edge carries a set of candidate operations. For an edge connecting cell � to cell $q$ in layer �, the output is computed as follows:

$$
\bar { o } ^ { ( p , q ) } ( X ) = \sum _ { o \in \mathcal { O } ^ { ( l , p , q ) } } \frac { \exp ( \alpha _ { o } ^ { ( l , p , q ) } ) } { \sum _ { o ^ { \prime } \in \mathcal { O } ^ { ( l , p , q ) } } \exp ( \alpha _ { o ^ { \prime } } ^ { ( l , p , q ) } ) } o ( X ) ,\tag{22}
$$

where $\alpha _ { o } ^ { ( l , p , q ) } \in \mathbb { R }$ is a learnable scalar quantifying the contribution of candidate operation � on edge $( p , q )$ . Crucially, � and � index \*\*cells in the DAG supernetwork\*\* (e.g., the input cell, operation cells, and output cell), not nodes in the original input graph. In the first stage, each operation cell � is directly connected from the input cell to the output cell, forming a star topology. Consequently, the general DARTS weight $\mathbf { \bar { \alpha } } \alpha _ { o } ^ { ( l , p , q ) }$ reduces to the simpler layer-operation notation $\alpha _ { S } ^ { ( l , i ) }$ used in Eq. (11). Similarly, in the second stage, it reduces to $\alpha _ { G } ^ { ( l , i ) }$ in Eq. (14). The task of architecture search is thus simplified to learning the continuous variables $\alpha _ { S } ^ { ( l , i ) }$ and $\alpha _ { G } ^ { ( l , i ) }$ , a problem that can be efectively tackled using DARTS.

## 3.3.3. Discretization and Final Architecture

Once the search phase is complete, the continuous architecture weights are converted into a discrete architecture before retraining. Following the standard DARTS protocol

[21], for each layer �, we select the operation with the largest weight and discard all others:

$$
i _ { S } ^ { ( l ) } = \arg \operatorname* { m a x } _ { i } \alpha _ { S } ^ { ( l , i ) } , \quad i _ { G } ^ { ( l ) } = \arg \operatorname* { m a x } _ { i } \alpha _ { G } ^ { ( l , i ) } ,\tag{23}
$$

where $i _ { S } ^ { ( l ) }$ denotes the selected node scoring strategy and $i _ { G } ^ { ( l ) }$ denotes the selected graph neural network operation for the �-th layer. This yields a fixed, layer-specific architecture that is then retrained from scratch using the standard training procedure described in Section 4. By deriving a discrete architecture in this manner, we retain the flexibility of the continuous search while producing a compact and eficient model for deployment.

## 3.4. Interpretability and Explainability

In graph learning, interpretable models are those that can provide human-understandable mechanisms to make predictions [43], while results by explainable models are understandable by post hoc explanation techniques [44]. Our proposed approach enhances interpretability by learning a soft gating vector for each layer from seven diferent scoring perspectives, where each entry indicates how much a node contributes to subsequent GNN processing. By assigning low gate values to nodes that require less further processing, the proposed approach can efectively reduce oversmoothing and enhance discrimination power. For visualization purposes, we binarize the continuous gate values with a threshold of 0.5 to highlight which nodes are strongly selected $\mathrm { ( g a t e > 0 . 5 , }$ , shown in red) versus weakly selected $( \mathrm { g a t e } \le 0 . 5$ , shown in green), as detailed in the experiments.

In short, the explicit soft gating mechanism provides valuable insights into improving neural network architectures, ofering a controllable and efective information filtering mechanism for GNNs, thereby optimizing the learning and information propagation process across deep layers.

## 4. Experiments

## 4.1. Experimental setting

Datasets and Tasks. We evaluated our approach on two task categories—graph classification and node classification—utilizing eight diferent datasets: Cora, Cite-Seer, PubMed, D&D, PROTEINS, IMDB-MULTI, COX2, and MR. The Cora, CiteSeer, and PubMed datasets are publication citation networks where nodes represent documents and edges denote citations, adhering to the standard train/validation/test split proposed by Yang et al [45]. The D&D and PROTEINS datasets encode protein structures, with nodes representing amino acids and edges formed if the distance between them is less than 6 angstroms. The IMDB-MULTI dataset focuses on movie collaborations, detailing actor/actress and genre information sourced from IMDB, where nodes correspond to actors/actresses and edges indicate their co-appearance in movies. The COX2 dataset includes 467 cyclooxygenase-2 (COX-2) inhibitors, where these compounds are classified based on their activity levels as either active or inactive against human recombinant enzymes in vitro. The MR dataset consists of movie reviews for binary sentiment classification, featuring 5331 positive and 5331 negative reviews, each represented as a single sentence. In these graphs, nodes denote unique words, and edges represent co-occurrences of word pairs within a fixedsize sliding window. Detailed information about the eight datasets can be found in Table 2.

Statistics of the datasets used in the experiments. For node classification tasks, the terms "Nodes" and "Edges" correspond to the cumulative count of nodes and all edges within the datasets. Conversely, "Nodes" and "Edges" signify the mean number of nodes and edges across the graphs for graph classification tasks.
<table><tr><td>Task</td><td>Dataset</td><td>Graphs</td><td>Nodes</td><td>Edges</td><td>Features</td><td>Classes</td></tr><tr><td rowspan="3">Node Classification</td><td>Cora</td><td>-</td><td>2708</td><td>10556</td><td>1433</td><td>7</td></tr><tr><td>Citeseer</td><td></td><td>3327</td><td>9104</td><td>3703</td><td>6</td></tr><tr><td>PubMed</td><td></td><td>19717</td><td>88648</td><td>500</td><td>3</td></tr><tr><td rowspan="5">Graph Classification</td><td>D&amp;D</td><td>1178</td><td>384.3</td><td>715.7</td><td>89</td><td>2</td></tr><tr><td>PROTEINS</td><td>1113</td><td>39.1</td><td>72.8</td><td>3</td><td>2</td></tr><tr><td>IMDB-MULTI</td><td>1500</td><td>13</td><td>65.9</td><td>0</td><td>3</td></tr><tr><td>COX2</td><td>467</td><td>41.2</td><td>43.5</td><td>3</td><td>2</td></tr><tr><td>MR</td><td>10654</td><td>19.5</td><td>73.5</td><td>300</td><td>2</td></tr></table>

Compared Methods. We conduct a comparative analysis of N-GNAS against manually crafted GNN architectures and existing NAS methods. The manually crafted baselines encompass: GCN [2], GIN [7], GraphSAGE [24], GAT [25], SGC [28], alongside more powerful baselines such as PNA [46], AP-GCN [47], and DGCNN [48]. Our experiments utilize the widely recognized, open-source library PyTorch Geometric (PyG) [49], which facilitates the implementation of diverse GNN models. We conduct training for each baseline five times across various datasets, using the optimal hyperparameters specified in the paper. We present the mean accuracy and variability measure to assess the models’ evaluative performance. Furthermore, we compare with a suite of GNAS baselines, including (I) GraphNAS [16], an RL-based method incorporating a parameter-sharing scheme. (II) SNAG [30], an RL-based GNAS method emphasizing the combination of nodes and layers. (III) SANE [20], a GD-based method that operates on a search space comprising the combination of nodes and layers along with skip-connections (IV) PAS [36], a GD-based GNAS method approach that integrates Aggregation, Pooling, Readout, and Merge components. (V) Genetic-GNN [32], an EAbased GNAS method that dynamically adapts the interplay between the GNN model structure and hyperparameters through an alternating evolutionary process, aiming to find the optimal combination of GNN model structure and hyperparameters. (VI) KEGNAS [33], a knowledge-aware EAbased GNAS method that transfers prior knowledge from NAS-Bench-Graph[50] via a knowledge model and a deep multi-output Gaussian process (DMOGP) to warm-start an evolutionary algorithm for optimizing accuracy and model size. (VII) DFG-NAS [34], an EA-based GNAS method concentrating on macro-architecture design, involves integrating and organizing propagation and transformation operations into GNNs to optimize GNN architectures. (VIII)

ExGNAS[51], an explainable Graph NAS method based on Monte-Carlo tree search that improves interpretability via a simple search space and decision-traceable architecture selection.

Searching setting. In our implementation of N-GNAS, we delineate a two-stage search space within each layer, comprising seven and five operational cells, respectively, denoted by $n _ { c } = 7 , 5 .$ . To ensure experimental consistency, the depth of our model is consistently set to two layers. We allocate half of the available training data for the search process for validation purposes. We adopt DARTS as our search strategy, where the operation weights, architecture parameters, and the corresponding optimizer are configured similarly to previous works [20, 30]. Furthermore, we initiate the architecture search with distinct random seeds on eight diverse datasets, each employing a 2-layer GNN model throughout 100 epochs. The batch size is standardized to 64, and the node feature dimensionality is upscaled to 64. Through this sequence of eight architectural searches, we successfully identify the premier architectural configuration on most datasets, with the top-1 architecture prevailing on seven datasets and the top-2 on one remaining dataset.

Training setting. In the training phase, we follow the configurations employed in earlier studies [52], which encompass data splitting, optimizer selection, feature dimension adjustment, batch normalization, and the inclusion of residual connections. In particular, we employed the Adam optimizer with an initial learning rate 1e-5 for diferent datasets. To alleviate overfitting, a dropout rate of 0.2 was used. We conducted five repetitions of the top-1 architecture identified on the validation dataset, each for 400 epochs, using a batch size of 64. Ultimately, We reported the average performance and standard deviation of the top-1 architecture on the validation dataset. All experiments used a single NVIDIA GeForce RTX 3090 GPU (with 24GB memory, CUDA version 11.2).

## 4.2. Results and Analysis

Results on node classification task. Cora originates from machine learning papers across various disciplines,

Node classification task accuracy on the Cora, CiteSeer, and PubMed datasets. The top three are emphasized by first, second, and third.
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Methods</td><td rowspan=1 colspan=1>Cora        CiteSeer      PubMed</td></tr><tr><td rowspan=6 colspan=1>manually crafted</td><td rowspan=2 colspan=1>GCN [2]GIN [7]</td><td rowspan=1 colspan=1>86.09±0.50    $\overline { 7 4 . 6 4 { \pm } 0 . 2 0 }$     $\overline { { 8 8 . 9 6 { \pm 0 . 2 9 } } }$ </td></tr><tr><td rowspan=1 colspan=1>85.68±0.61    $7 3 . 4 0 { \pm } 0 . 1 4$    $8 8 . 2 3 { \pm } 0 . 2 8 $ </td></tr><tr><td rowspan=4 colspan=1>GraphSAGE [24]GAT [25]SGC [28]PNA [46]AP-GCN [47]</td><td rowspan=1 colspan=1>85.66±0.52   $7 4 . 5 9 { \pm } 0 . 6 3$    $8 9 . 2 1 { \pm } 0 . 2 9$ 85.92±0.72   $7 4 . 2 6 { \pm } 0 . 1 3$    $8 8 . 6 7 { \scriptstyle \pm 0 . 1 9 }$ </td></tr><tr><td rowspan=1 colspan=1> $8 5 . 3 1 { \pm } 0 . 8 6$    $7 2 . 9 4 { \pm } 0 . 9 8$     $8 8 . 4 0 { \pm } 0 . 2 5 $ </td></tr><tr><td rowspan=1 colspan=1> $8 5 . 0 6 { \scriptstyle \pm 0 . 7 2 }$    $7 5 . 0 6 { \scriptstyle \pm 0 . 6 1 }$    $8 7 . 1 8 { \pm } 0 . 3 0 $ </td></tr><tr><td rowspan=1 colspan=1> $8 6 . 7 5 { \scriptstyle \pm 0 . 2 9 }$    $7 5 . 6 4 \pm 0 . 7 5$    $8 9 . 7 9 { \scriptstyle \pm 0 . 2 6 }$ </td></tr><tr><td rowspan=6 colspan=1>NAS</td><td rowspan=4 colspan=1>GraphNAS [16]SNAG [30]SANE [20]Genetic-GNN [32]KEGNAS [33]DFG-NAS [34]</td><td rowspan=1 colspan=1> $\overline { { 8 6 . 6 9 \pm 0 . 7 1 } }$     $\overline { { 7 6 . 3 2 { \pm 0 . 6 1 } } }$    $\overline { { 8 7 . 9 3 \pm 0 . 1 6 } }$ </td></tr><tr><td rowspan=1 colspan=1> $8 4 . 9 9 { \pm } 1 . 0 4 $    $7 5 . 9 4 \pm 1 . 2 7$    $8 8 . 4 4 { \pm } 0 . 2 5 $ </td></tr><tr><td rowspan=1 colspan=1> $8 6 . 4 0 { \scriptstyle \pm 0 . 3 8 }$    $7 6 . 7 2 { \scriptstyle \pm 0 . 9 8 }$    $8 9 . 3 4 { \pm } 0 . 3 1 $ </td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>ExGNAS [51]</td><td rowspan=1 colspan=1> $8 8 . 5 0 { \pm } 0 . 0 1$     $7 4 . 7 { \pm } 0 . 0 2 $     $8 9 . 3 { \pm } 0 . 1 0 $ </td></tr><tr><td rowspan=1 colspan=1>N-GNAS</td><td rowspan=1 colspan=1> $\overline { { 8 8 . 5 9 2 0 . 6 3 } }$    $7 8 . 2 6 { \pm } 0 . 4 2 $    $9 0 . 2 7 { \scriptstyle \pm 0 . 4 8 }$ </td></tr></table>

Graph classification task accuracy on the D&D, PROTEINS, IMDB-MULTI, COX2, and MR datasets. The top three are emphasized by first, second, and third.
<table><tr><td></td><td>Methods</td><td>D&amp;D</td><td>PROTEINS</td><td>IMDB-MULTI</td><td>COX2</td><td>MR</td></tr><tr><td rowspan="5">manually crafted</td><td>GCN [2]</td><td>76.98±4.43</td><td> $\overline { { 7 2 . 9 4 \pm 1 . 8 2 } }$ </td><td> $\overline { { 5 0 . 2 5 { \pm } 3 . 4 2 } }$ </td><td> $\overline { { 7 8 . 6 8 { \pm } 1 . 9 2 } }$ </td><td> $\overline { { 7 5 . 6 2 \pm 0 . 8 5 } }$ </td></tr><tr><td>GIN [7]</td><td> $7 3 . 9 5 { \pm } 2 . 9 8 $ </td><td> $7 3 . 6 8 { \pm } 2 . 7 8$ </td><td> $5 0 . 0 4 { \scriptstyle \pm 2 . 7 5 }$ </td><td> $8 0 . 5 2 { \scriptstyle \pm 3 . 4 1 }$ </td><td> $7 6 . 0 5 { \scriptstyle \pm 0 . 7 4 }$ </td></tr><tr><td>GraphSAGE [24]</td><td> $7 6 . 7 8 { \scriptstyle \pm 4 . 0 6 }$ </td><td> $7 2 . 5 8 { \pm } 2 . 4 3$ </td><td> $4 9 . 6 2 \pm 4 . 5 8$ </td><td> $7 9 . 6 3 { \pm } 2 . 6 3$ </td><td> $7 6 . 8 5 { \pm } 0 . 6 3$ </td></tr><tr><td>GAT [25]</td><td> $7 5 . 1 4 { \scriptstyle \pm 2 . 8 4 }$ </td><td> $\pm 1 . 2 9 { \pm } 1 . 6 9$ </td><td> $4 9 . 8 5 { \pm } 3 . 6 5$ </td><td> ${ \pm 1 . 1 6 \pm 3 . 6 4 }$ </td><td> $7 6 . 9 2 { \scriptstyle \pm 1 . 0 2 }$ </td></tr><tr><td>DGCNN [48]</td><td> $7 6 . 6 6 { \scriptstyle \pm 4 . 0 3 }$ </td><td> $7 3 . 2 8 { \pm } 3 . 1 6$ </td><td> $4 9 . 6 3 { \pm } 3 . 7 3 $ </td><td> $8 0 . 1 6 { \pm } 3 . 2 3 $ </td><td> $7 7 . 0 6 { \pm } 0 . 6 8 $ </td></tr><tr><td rowspan="5">NAS</td><td>GraphNAS [16]</td><td> $\overline { { 7 3 . 5 6 { \pm 2 . 4 7 } } }$ </td><td> $\overline { { 7 3 . 1 2 \pm 4 . 2 7 } }$ </td><td> $\overline { { 4 7 . 2 3 { \pm 4 . 5 9 } } }$ </td><td> $\overline { { 7 8 . 9 1 \pm 2 . 3 7 } }$ </td><td> $\overline { { 7 6 . 3 7 { \pm } 2 . 1 7 } }$ </td></tr><tr><td>SNAG [30]</td><td> $7 4 . 4 6 { \pm } 2 . 6 9$ </td><td> $7 1 . 9 8 { \pm } 3 . 4 2 $ </td><td> $4 9 . 3 8 { \pm } 1 . 7 9 $ </td><td> $7 9 . 3 2 { \pm } 2 . 7 4 $ </td><td> $7 6 . 2 3 { \scriptstyle \pm 2 . 6 1 }$ </td></tr><tr><td>SANE [20]</td><td> $7 3 . 6 2 { \pm } 3 . 8 2$ </td><td> $7 2 . 2 3 { \scriptstyle \pm 3 . 8 6 }$ </td><td> $4 9 . 9 2 { \scriptstyle \pm 3 . 7 1 }$ </td><td> $8 0 . 0 4 { \pm } 2 . 6 2 $ </td><td> $7 7 . 2 7 { \pm 2 . 1 2 }$ </td></tr><tr><td>PAS [36]</td><td> $7 7 . 4 6 { \pm } 3 . 6 8 $ </td><td> $7 4 . 8 5 { \pm } 2 . 9 3 $ </td><td> ${ \pm } \mathbf { 0 . 0 7 \pm } \mathbf { 2 . 9 4 }$ </td><td> $8 2 . 1 6 { \pm } 3 . 9 5 $ </td><td> $7 6 . 7 2 { \scriptstyle \pm 2 . 2 9 }$ </td></tr><tr><td>N-GNAS</td><td> $7 8 . 0 3 { \scriptstyle \pm 3 . 6 4 }$ </td><td> $\overline { 7 5 . 3 0 { \pm } 3 . 2 7 }$ </td><td> $\overline { { 5 1 . 6 2 \pm 3 . 5 1 } }$ </td><td> $8 1 . 2 9 { \pm } 4 . 1 7$ </td><td> $\overline { { 7 8 . 2 7 { \pm } 1 . 9 5 } }$ </td></tr></table>

CiteSeer is sourced from academic papers in computer science, and PubMed encompasses research content from multiple biomedical domains. These datasets are commonly employed for node classification tasks. The main reason is that current manually crafted and NAS methods in architecture design consider using the same number of graph neural networks for each node, leading to over-smoothing. Table 3 and Figure 3 present the results of the node classification task on the Cora, CiteSeer, and PubMed datasets. Our node classification performance surpasses current models, showing a 1.3% increase in test accuracy, highlighting our approach’s eficacy.

Results on graph classification task. Our model’s performance in graph classification tasks was assessed through experiments conducted on the D&D, PROTEINS, IMDB-MULTI, COX2, and MR datasets. The results in Table 4 show that our model achieves excellent results in this domain, similar to the node classification tasks. Compared to manually crafted and NAS methods, our model outperforms the baselines by 0.7% on the D&D dataset and by 0.6% on the PROTEINS dataset.

Discussion of diferent layers. As illustrated in Figure 4, we investigate the performance of our method on the Cora and PubMed datasets across various layers. We compared N-GNAS with several baselines, including GCN, GAT, SANE, and DFG-NAS, in terms of accuracy for diferent numbers of layers. As depicted in Figure 4, we conclude that GNAS’s performance remains stable as the layers’ depth increases, outperforming other baselines. However, the performance of the baselines relative to N-GNAS is lower, and it decreases as the depth increases, indicating the occurrence of oversmoothing issues in these baselines. Therefore, N-GNAS can simultaneously consider model performance and alleviate over-smoothing problems.

Explainability of the search results. We conducted comprehensive experiments to explain the results on the Cora dataset, as illustrated in Figure 5. The first layer of the obtained network determined by N-GNAS uses the fifth metric, the local structural feature scoring, as the node selection metric, and thus, the selected nodes in red are typical with larger degrees, as shown in Figure 5a. In the second layer, the fourth metric, the global position scoring, is used as the node selection metric, indicating that more central nodes with shorter overall distances to others are more likely to be selected, as shown in Figure 5b. These results demonstrated the explainability. Moreover, the prediction quality benefits directly from this mechanism: as shown in Table 5, the accuracy gap between N-GNAS and its variant without node selection widens from 1.58% (1 layer) to 4.19% (5 layers), confirming that the learned gating structure becomes increasingly critical for maintaining discriminative predictions in deeper models.

Table 5  
CiteSeer  
![](images/ff7842f287c20155f18a9002d7726dd396cb1280102ac05167a49dc399020130.jpg)  
Figure 3: Results of various methods on the Cora, CiteSeer, and PubMed datasets. Diferent colors represent diferent methods. It can be found that our work achieves the best results on both datasets

![](images/ef8f3812e54f8a211d7786f51b662993e781909e783fac343e953df2fb6a0151.jpg)

![](images/709f5269751350f1fd1accaaae6105f846adf7bcb968e87a0d13cadd391583fd.jpg)  
Figure 4: Performance of the model with diferent numbers of layers. Diferent colors represent diferent methods. It can be found that our work achieves the best results on both datasets

![](images/291aa3564796ab6024cd45a7c20d99b84faed3bc294d4016b5a4db8eac0dbd56.jpg)

![](images/81901cfb3f012d5b7169e3ecaca97b40eedaee2b24c49e50a603a27f50d723df.jpg)  
(b) the second layer  
Figure 5: Node selection results obtained by the first and the second layer of graph neural networks optimized with N-GNAS. Red nodes (gate value > 0.5) are strongly selected for GNN processing, while green nodes (gate value ≤ 0.5) mostly bypass GNN operations via residual connections.

Performance of N-GNAS with varying layers on the Cora dataset. We present the test accuracy results for N-GNAS without Node Selection Operations and for N-GNAS with these operations.
<table><tr><td colspan="3">Cora</td></tr><tr><td>Layer</td><td>Stage II</td><td>N-GNAS</td></tr><tr><td>1</td><td> $\overline { { 7 6 . 9 3 \pm 0 . 4 9 } }$ </td><td> $\overline { { 7 8 . 5 1 \pm 1 . 1 3 } }$ </td></tr><tr><td>2</td><td> $8 6 . 3 6 { \pm } 1 . 1 2 $ </td><td> $8 8 . 5 9 { \pm } 0 . 6 3 $ </td></tr><tr><td>3</td><td> $8 4 . 4 3 { \scriptstyle \pm 0 . 7 1 }$ </td><td> $8 7 . 9 2 { \pm } 0 . 6 8 $ </td></tr><tr><td>4</td><td> $8 2 . 7 1 { \scriptstyle \pm 0 . 8 3 }$ </td><td> $8 5 . 4 3 { \pm } 0 . 6 6$ </td></tr><tr><td>5</td><td> $7 9 . 5 7 { \pm } 0 . 5 5$ </td><td> $8 3 . 7 6 { \scriptstyle \pm 0 . 4 7 }$ </td></tr></table>

The explainability can also be perceived from the ablation study, as shown in Table 5. Over-smoothing can significantly deteriorate model efectiveness in traditional GNNs as the number of layers increases. Nevertheless, the N-GNAS framework mitigates the negative impact of increased layer depth on performance, which indicates that it reduces the over-smoothing by only using selected nodes for further processing.

Ablation study. An ablation study was carried out to showcase the eficacy of our custom-designed search space and loss function. Table 6 compares N-GNAS against two baseline models: (1) Random: randomly selecting one architecture without neural architecture search, and (2) $L _ { c e } \mathrm { : }$ using only the cross-entropy loss $L _ { c e }$ in the loss function while turning of $L _ { c l } .$ . For models that did not utilize our search space and instead employed a randomly chosen fixed architecture, the accuracy decreased by 1.8-4.6%. This decrease is attributed to the inability of the model to assign appropriate graph neural network counts to diferent nodes without using the proper Node-wise optimal convolution selection operation and Graph convolution, leading to performance degradation. For models that did not employ $L _ { c l }$ and only used $L _ { c e } ,$ , the accuracy decreased by 1.8-2.5%. This decrease is because not using $L _ { c l }$ only considers the model’s performance while ignoring the over-smoothing issue, posing a risk of over-smoothing. These ablation studies indicate the efectiveness of our designed search space and loss function.

Ablation study on search space and loss function.
<table><tr><td>Methods</td><td>Cora</td><td>CiteSeer</td><td>PubMed</td></tr><tr><td>Random</td><td>85.74</td><td>74.63</td><td>88.62</td></tr><tr><td> $L _ { c e }$ </td><td>86.94</td><td>76.52</td><td>87.98</td></tr><tr><td>N-GNAS</td><td>88.59</td><td>78.26</td><td>90.27</td></tr></table>

## 5. Conclusion

This work proposes an innovative N-GNAS model for graph neural architecture search. This framework is distinguished by its adaptability and generality, featuring a dynamic search space for the optimal count of graph neural networks. It facilitates the discovery of superior GNN configurations by integrating node-wise optimal convolution selection and graph neural network operation. A novel loss function has been engineered to balance model performance challenges and the over-smoothing phenomenon. It demonstrates strong competitiveness against manually crafted GNNs across various tasks. Furthermore, a twophase search approach is presented to enhance search eficiency. Broad experimental results spanning multiple graph learning tasks demonstrate the efectiveness of our method, with graphical illustrations of the outcomes reinforcing widely accepted insights.

## CRediT authorship contribution statement

Lintao Yang: Conceptualization; Formal analysis; Writing – original draft; Methodology; Writing – review & editing. Sirui Li: Conceptualization; Formal analysis; Writing – original draft; Methodology; Writing – review & editing. Yaqing Wang: Investigation; Validation; Writing – review & editing. Pietro Liò: Investigation; Software; Resources. Xu Shen: Methodology; Writing – original draft; Investigation. Baisong Liu: Writing – review & editing; Supervision; Resources. Chengbin Peng: Formal analysis; Methodology; Resources; Supervision; Project administration; Writing – review & editing.

## References

[1] P. W. Battaglia, J. B. Hamrick, V. Bapst, A. Sanchez-Gonzalez, V. Zambaldi, M. Malinowski, A. Tacchetti, D. Raposo, A. Santoro, R. Faulkner, et al., Relational inductive biases, deep learning, and graph networks, arXiv preprint arXiv:1806.01261 (2018).

[2] T. N. Kipf, M. Welling, Semi-supervised classification with graph convolutional networks, arXiv preprint arXiv:1609.02907 (2016).

[3] X. Wang, H. Ji, C. Shi, B. Wang, Y. Ye, P. Cui, P. S. Yu, Heterogeneous graph attention network, in: The world wide web conference, pp. 2022–2032.

[4] A. Kumar, S. S. Singh, K. Singh, B. Biswas, Link prediction techniques, applications, and performance: A survey, Physica A: Statistical Mechanics and its Applications 553 (2020) 124289.

[5] D. Liben-Nowell, J. Kleinberg, The link prediction problem for social networks, in: Proceedings of the twelfth international conference on Information and knowledge management, pp. 556–559.

[6] L. Lü, T. Zhou, Link prediction in complex networks: A survey, Physica A: statistical mechanics and its applications 390 (2011) 1150– 1170.

[7] K. Xu, W. Hu, J. Leskovec, S. Jegelka, How powerful are graph neural networks?, arXiv preprint arXiv:1810.00826 (2018).

[8] X. Li, Q. Xu, Z. Xu, H. Zhang, L. Xu, Graph reconfigurable pooling for graph representation learning, IEEE Transactions on Emerging Topics in Computing 12 (2023) 139–149.

[9] Y.-P. Wang, W.-C. Wang, Y.-H. Chang, C.-L. Tsai, T.-W. Kuo, C.-F. Wu, C.-C. Ho, H.-W. Hu, Tcam-gnn: A tcam-based data processing strategy for gnn over sparse graphs, IEEE Transactions on Emerging Topics in Computing (2023).

[10] X. Shen, Y. Wang, K. Zhou, S. Pan, X. Wang, Optimizing ood detection in molecular graphs: A novel approach with difusion models, arXiv preprint arXiv:2404.15625 (2024).

[11] R. Miao, K. Zhou, Y. Wang, N. Liu, Y. Wang, X. Wang, Rethinking independent cross-entropy loss for graph-structured data, arXiv preprint arXiv:2405.15564 (2024).

[12] Y. Chen, J. You, J. He, Y. Lin, Y. Peng, C. Wu, Y. Zhu, Sp-gnn: Learning structure and position information from graphs, Neural Networks 161 (2023) 505–514.

[13] G. Xue, M. Zhong, T. Qian, J. Li, Psa-gnn: An augmented gnn framework with priori subgraph knowledge, Neural Networks 173 (2024) 106155.

[14] B. Zoph, Q. V. Le, Neural architecture search with reinforcement learning, arXiv preprint arXiv:1611.01578 (2016).

[15] J. Gao, J. Chen, B. M. Oloulade, R. Al-Sabri, T. Lyu, J. Zhang, Z. Li, Commgnas: unsupervised graph neural architecture search for community detection, IEEE Transactions on Emerging Topics in Computing (2023).

[16] Y. Gao, H. Yang, P. Zhang, C. Zhou, Y. Hu, Graph neural architecture search, in: International joint conference on artificial intelligence, International Joint Conference on Artificial Intelligence.

[17] Y. Gao, P. Zhang, Z. Li, C. Zhou, Y. Liu, Y. Hu, Heterogeneous graph neural architecture search, in: 2021 IEEE International Conference on Data Mining (ICDM), IEEE, pp. 1066–1071.

[18] M. Schlichtkrull, T. N. Kipf, P. Bloem, R. Van Den Berg, I. Titov, M. Welling, Modeling relational data with graph convolutional networks, in: The Semantic Web: 15th International Conference, ESWC 2018, Heraklion, Crete, Greece, June 3–7, 2018, Proceedings 15, Springer, pp. 593–607.

[19] X. Fu, J. Zhang, Z. Meng, I. King, Magnn: Metapath aggregated graph neural network for heterogeneous graph embedding, in: Proceedings of The Web Conference 2020, pp. 2331–2341.

[20] Z. Huan, Y. Quanming, T. Weiwei, Search to aggregate neighborhood for graph neural network, in: 2021 IEEE 37th International Conference on Data Engineering (ICDE), IEEE, pp. 552–563.

[21] H. Liu, K. Simonyan, Y. Yang, Darts: Diferentiable architecture search, arXiv preprint arXiv:1806.09055 (2018).

[22] S. Cai, L. Li, X. Han, J. Luo, Z.-J. Zha, Q. Huang, Automatic relationaware graph network proliferation, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10863– 10873.

[23] X. Shen, P. Lio, L. Yang, R. Yuan, Y. Zhang, C. Peng, Graph rewiring and preprocessing for graph neural networks based on efective resistance, IEEE Transactions on Knowledge and Data Engineering (2024).

[24] W. Hamilton, Z. Ying, J. Leskovec, Inductive representation learning on large graphs, Advances in neural information processing systems 30 (2017).

[25] P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Lio, Y. Bengio, Graph attention networks, arXiv preprint arXiv:1710.10903 (2017).

[26] M. Deferrard, X. Bresson, P. Vandergheynst, Convolutional neural networks on graphs with fast localized spectral filtering, Advances in neural information processing systems 29 (2016).

[27] K. Xu, C. Li, Y. Tian, T. Sonobe, K.-i. Kawarabayashi, S. Jegelka, Representation learning on graphs with jumping knowledge networks,

in: International conference on machine learning, PMLR, pp. 5453– 5462.

[28] F. Wu, A. Souza, T. Zhang, C. Fifty, T. Yu, K. Weinberger, Simplifying graph convolutional networks, in: International conference on machine learning, PMLR, pp. 6861–6871.

[29] X. Bresson, T. Laurent, Residual gated graph convnets, arXiv preprint arXiv:1711.07553 (2017).

[30] H. Zhao, L. Wei, Q. Yao, Simplifying architecture search for graph neural network, arXiv preprint arXiv:2008.11652 (2020).

[31] H. Liu, K. Simonyan, O. Vinyals, C. Fernando, K. Kavukcuoglu, Hierarchical representations for eficient architecture search, arXiv preprint arXiv:1711.00436 (2017).

[32] M. Shi, Y. Tang, X. Zhu, Y. Huang, D. Wilson, Y. Zhuang, J. Liu, Genetic-gnn: Evolutionary architecture search for graph neural networks, Knowledge-Based Systems 247 (2022) 108752.

[33] C. Wang, J. Zhao, L. Li, L. Jiao, F. Liu, X. Liu, S. Yang, Knowledgeaware evolutionary graph neural architecture search, Knowledge-Based Systems 309 (2025) 112810.

[34] W. Zhang, Z. Lin, Y. Shen, Y. Li, Z. Yang, B. Cui, Deep and flexible graph neural architecture search, in: International Conference on Machine Learning, PMLR, pp. 26362–26374.

[35] J. Chen, J. Gao, Z. Wu, R. Al-Sabri, B. M. Oloulade, Decoupled diferentiable graph neural architecture search, Information Sciences 673 (2024) 120700.

[36] L. Wei, H. Zhao, Z. He, Q. Yao, Neural architecture search for gnn-based graph classification, ACM Transactions on Information Systems (2023).

[37] Z. Zhang, X. Wang, W. Zhu, Automated machine learning on graphs: A survey, arXiv preprint arXiv:2103.00742 (2021).

[38] Y. Li, Z. Wen, Y. Wang, C. Xu, One-shot graph neural architecture search with dynamic search space, in: Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, AAAI Press, 2021, pp. 8510–8517.

[39] R. W. Floyd, Algorithm 97: shortest path, Communications of the ACM 5 (1962) 345–345.

[40] E. Ranjan, S. Sanyal, P. Talukdar, Asap: Adaptive structure aware pooling for learning hierarchical graph representations, in: Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pp. 5470–5477.

[41] Z. Ying, J. You, C. Morris, X. Ren, W. Hamilton, J. Leskovec, Hierarchical graph representation learning with diferentiable pooling, Advances in neural information processing systems 31 (2018).

[42] P.-T. De Boer, D. P. Kroese, S. Mannor, R. Y. Rubinstein, A tutorial on the cross-entropy method, Annals of operations research 134 (2005) 19–67.

[43] H. Yuan, H. Yu, S. Gui, S. Ji, Explainability in graph neural networks: A taxonomic survey, IEEE transactions on pattern analysis and machine intelligence 45 (2022) 5782–5799.

[44] H. Xuanyuan, P. Barbiero, D. Georgiev, L. C. Magister, P. Liò, Global concept-based interpretability for graph neural networks via neuron analysis, in: Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pp. 10675–10683.

[45] Z. Yang, W. Cohen, R. Salakhudinov, Revisiting semi-supervised learning with graph embeddings, in: International conference on machine learning, PMLR, pp. 40–48.

[46] G. Corso, L. Cavalleri, D. Beaini, P. Liò, P. Veličković, Principal neighbourhood aggregation for graph nets, Advances in Neural Information Processing Systems 33 (2020) 13260–13271.

[47] I. Spinelli, S. Scardapane, A. Uncini, Adaptive propagation graph convolutional network, IEEE transactions on neural networks and learning systems 32 (2021) 4755–4760.

[48] M. Zhang, Z. Cui, M. Neumann, Y. Chen, An end-to-end deep learning architecture for graph classification, in: Proceedings of the AAAI conference on artificial intelligence, volume 32.

[49] M. Fey, J. E. Lenssen, Fast graph representation learning with pytorch geometric. arxiv 2019, arXiv preprint arXiv:1903.02428 (1903).

[50] Y. Qin, Z. Zhang, X. Wang, Z. Zhang, W. Zhu, Nas-bench-graph: Benchmarking graph neural architecture search, Advances in Neural

Information Processing Systems 35 (2022) 54–69.

[51] Y. Sasaki, Explainable graph neural architecture search via montecarlo tree search, in: 2025 IEEE International Conference on Big Data (BigData), pp. 752–757.

[52] S. Cai, L. Li, J. Deng, B. Zhang, Z.-J. Zha, L. Su, Q. Huang, Rethinking graph neural architecture search from message-passing, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6657–6666.