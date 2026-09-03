# PEARL: Path-Entity Aligned Relational Learning with Contextual Subgraphs for Inductive Knowledge Graph Completion

Yunchi Yang School of Mathematics, Shandong University Jinan 250100, China ycyang@mail.sdu.edu.cn

Longlong Li<sup>∗</sup> School of Physical and Mathematical Sciences, Nanyang Technological University Singapore 637371, Singapore longlong.li@ntu.edu.sg

Cunquan Qu<sup>∗</sup> Data Science Institute, Shandong University Jinan 250100, China cqqu@sdu.edu.cn

September 3, 2026

## Abstract

Inductive knowledge graph completion (IKGC) aims to predict missing links involving entities unseen during training, requiring models to learn transferable relational and structural patterns. Existing subgraph- and path-based approaches often encode relational paths independently of their surrounding query subgraphs, although the predictive relevance of a path may vary across structural contexts. We propose PEARL, a Path–Entity Aligned Relational Learning framework that models paths as contextconditioned reasoning signals. PEARL constructs a query-specific contextual subgraph from the union of the query entities’ neighborhoods and uses a large language model (LLM)-guided retriever to distill semantically relevant paths. It then builds a bipartite interaction graph over paths, contextual entities, and a global subgraph representation, allowing path embeddings to adapt to local and global structural evidence. To suppress noise introduced by the enlarged context, PEARL employs a dual-view contrastive objective that promotes representation consistency under stochastic contextual perturbations. Experiments on WN18RR, FB15k-237, and NELL-995 show that PEARL obtains the best average Hits@10 among the compared IKGC methods on all three benchmarks. Ablation studies, eficiency analyses, and case studies further validate the contributions of contextual subgraph modeling, semantic path retrieval, path–entity interaction, and contrastive regularization.

Keywords: Inductive Knowledge Graph Completion; Large Language Models; Path–Entity Alignment;   
Context-aware Path Reasoning.

## 1 Introduction

Knowledge Graphs (KGs) have become a fundamental infrastructure for numerous AI applications, including recommendation [1, 2], question answering [3, 4], and semantic search [5, 6], by organizing factual knowledge as relational triples. Despite their large scale, real-world KGs are inherently incomplete, motivating extensive research on knowledge graph completion (KGC) to predict missing facts [7, 8]. Most existing KGC models are developed under the transductive setting, where all entities are assumed to be observed during training. However, real-world KGs evolve continuously and frequently introduce new entities. For example, e-commerce knowledge graphs are constantly expanded with newly released products and brands. Retraining the entire model whenever new entities arrive is therefore impractical [9]. Consequently, inductive knowledge graph completion (IKGC) [10–12] has attracted increasing attention, requiring models to learn transferable relationa and structural patterns that generalize to unseen entities rather than relying on entity-specific embeddings.

Existing IKGC methods mainly exploit three types of reasoning evidence. The first is rule-based reasoning [13, 14], which learns transferable logical rules and naturally supports inductive inference through relation patterns. The second is subgraph evidence [15–18], which captures the local structural context surrounding the query entities and enables reasoning through neighborhood interactions. The third is path evidence [19–21], which provides explicit multi-hop relational chains connecting the query entities and ofers complementary logical clues beyond local structural information. These three paradigms have significantly advanced inductive knowledge graph completion from diferent perspectives.

Efective inductive reasoning often benefits from exploiting a broad range of relational evidence surrounding the query entities, as additional triples may provide complementary structural and semantic clues for relation prediction. However, real-world knowledge graphs inevitably contain noisy or irrelevant information [22, 23], making not all available evidence equally useful for reasoning. Moreover, the predictive utility of a relational path may vary considerably across diferent structural contexts [24]. For example, the same reasoning path $( { \mathrm { C o m p a n y } } _ { A } ;$ , invests\_in, $\mathrm { S t a r t u p } _ { X }$ , produces, ${ \mathrm { P r o d u c t } } _ { Y } )$ may strongly support predicting an acquisition in one query, particularly when the surrounding structure reveals that $\operatorname { S t a r t u p } _ { X }$ heavily relies on $\mathrm { C o m p a n y } _ { A } \mathrm { \dot { s } }$ supply chain and $\mathrm { C o m p a n y } _ { A }$ holds a majority stake. In another query, however, the same path may provide much weaker evidence if $\mathrm { C o m p a n y } _ { A }$ is merely a minor investor and Product is primarily sold to riva companies, suggesting a purely financial relationship rather than strategic alignment.

These characteristics give rise to two key challenges. First, how can richer relational and structural information be efectively incorporated while preserving the core semantics of the target triple and limiting the influence of noisy evidence? Second, local subgraphs often contain multiple relational paths connecting the query entities, some of which may be redundant or only weakly related to the target relation [25]. Since the semantic importance of a path may further vary across diferent local structural environments, it remains challenging to identify informative reasoning paths and appropriately associate them with their surrounding context for query-specific reasoning.

To address these challenges, we propose PEARL, an IKGC framework based on Path-Entity Aligned Relational Learning. PEARL first constructs union-based contextual subgraphs to capture richer structural evidence. To alleviate the accompanying noise, a dual-view subgraph contrastive objective is introduced to preserve invariant structural semantics. Meanwhile, an LLM-guided path retrieval module ranks candidate paths and retains the most informative logical evidence. Finally, PEARL explicitly aligns paths with contextual entity nodes and the global subgraph representation through a Bipartite Graph Neural Network (BGNN), enabling context-conditioned path reasoning. Consequently, identical relational paths can be interpreted diferently under diferent structural environments, leading to more robust inductive reasoning.

Our main contributions are summarized as follows:

• We propose PEARL, a unified framework for inductive knowledge graph completion that jointly models contextual subgraphs and relational paths to improve reasoning over unseen entities.

• We develop an LLM-guided path retrieval and path–entity alignment strategy, together with contextual subgraph contrastive learning, to identify informative relational evidence and learn robust context-aware representations.

• Extensive experiments on WN18RR, FB15k-237, and NELL-995 demonstrate that PEARL achieves strong overall performance and the best average Hits@10 across all three benchmarks.

## 2 Related Work

Inductive knowledge graph completion aims to predict missing triples involving entities that are unseen during training. Unlike transductive link prediction, inductive reasoning requires models to learn transferable

relational patterns rather than memorizing entity-specific embeddings [26]. Existing methods can be broadly categorized into rule-based methods, subgraph-based methods, and subgraph & path-based methods.

## 2.1 Rule-based Methods

Rule-based methods are naturally suitable for inductive reasoning because they focus on learning logica patterns that are independent of specific entities. NeuralLP [14] introduces an end-to-end diferentiable framework that learns logical rules through diferentiable reasoning operators. DRUM [13] further improves expressiveness by introducing virtual relations and supporting rules of varying lengths through parameter sharing. Compared with traditional rule mining approaches such as AMIE [27] and RuleN [28], diferentiable rule learning methods provide better scalability and end-to-end optimization. Although rule-based methods ofer strong interpretability and generalization to unseen entities, they often struggle to capture complex structural semantics and higher-order interactions within local graph neighborhoods.

## 2.2 Subgraph-based Methods

Subgraph-based methods perform reasoning by extracting and encoding local graph structures surrounding the query entities. GraIL [26] first introduced enclosing subgraph extraction for inductive link prediction, demonstrating the efectiveness of local structural patterns for reasoning over unseen entities. CoMPILE [17] further enhances subgraph representations through relation-aware message passing, while TACT [16] incorporates topological relation patterns to capture richer structural semantics.

To improve subgraph quality, LCILP [29] employs a Personalized PageRank (PPR)-based clustering strategy to mitigate hub-node noise and extract more informative local neighborhoods. MINES [30] introduces message intercommunication mechanisms over neighbor-enhanced subgraphs to strengthen information propagation. More recent methods such as RMPI [18] and S<sup>2</sup>DN [15] further improve inductive reasoning through relational message passing and multi-view subgraph modeling, enabling more efective exploitation of structural semantics.

## 2.3 Subgraph & Path-based Methods

Beyond local subgraph structures, explicit relational paths provide complementary multi-hop evidence by capturing the relation sequences that connect query entities [31, 32]. NBFNet [12] formulates relational reasoning as a generalized Bellman–Ford process and learns to aggregate path evidence through neural message passing, achieving strong performance on both transductive and inductive link prediction tasks. SNRI [19] incorporates neighboring relational information into subgraph representations to better capture the semantics of unseen entities. RPC-IR [20] further enhances relational path representations through contrastive learning and first-order rule induction, improving the quality of logical reasoning evidence. More recently, CATS [21] introduces a context-aware framework that leverages large language models (LLMs) to assess query triples using latent type constraints, relational paths, and neighboring facts extracted from local subgraphs.

![](images/ca40aac05bca5c3232e21274fdc62302aaef406ec057645f92770ade208a17ca.jpg)  
<sup>e</sup>6e4 <sup>e1</sup>Figure 1: Overall framework of PEARL. (A) Given a query triple, PEARL first extracts a contextual <sup>Path</sup> <sup>2</sup> e<sub>7</sub> <sup>e</sup>6e4 <sup>e1</sup>subgraph from the knowledge graph. The proposed LLM-based Path Retrieval and Alignment (LPRA) e<sub>7</sub> <sup>e</sup>4 tmodule then retrieves informative relational paths and constructs the path–entity bipartite graph. The e<sub>2</sub> <sup>e</sup>7 t <sub>e</sub>contextual subgraph is subsequently encoded by the Contextual Subgraph Neural Network (CSNN) to obtain <sup>e</sup>2 <sup>e</sup>5contextual entity representations and a global subgraph representation. Subgraph Contrastive Learning (SCL) further regularizes the CSNN representations by enforcing consistency between two augmented contextual subgraph views. The resulting contextual representations are incorporated into the path–entity bipartite graph, where the Bipartite Graph Neural Network (BGNN) performs context-aware interactions between relational paths and structural information. Finally, the Relation-Aware Path Attention Fusion (RAPF) module aggregates the refined path representations for prediction. (B) Detailed illustration of the proposed LPRA module. (C) Detailed illustration of the proposed RAPF module.

## 3 Preliminary

## 3.0.1 Knowledge graph

A knowledge graph is a tuple $\mathcal { G } = ( \mathcal { E } , \mathcal { R } , \mathcal { T } )$ , where E and R are finite sets of entities and relation types, respectively, and ${ \mathcal { T } } \subseteq { \mathcal { E } } \times { \mathcal { R } } \times { \mathcal { E } }$ is the set of observed triples. Each triple $( h , r , t ) \in \mathcal { T }$ represents a directed edge of type r from the head entity h to the tail entity t.

## 3.0.2 Contextual Subgraph

Subgraph-based IKGC [26] reasons over a local graph associated with each query triple $q = ( h , r _ { q } , t )$ . We instantiate this locality using undirected k-hop neighborhoods. Edge directions are ignored only when computing graph distances. Let $\operatorname { d i s t } _ { \mathcal { G } } ( u , v )$ denote the shortest-path distance between u and v in the undirected view of G. The k-hop neighborhood of an entity v is

$$
\mathcal { N } ^ { k } ( v ) = \left\{ u \in \mathcal { E } \mid \mathrm { d i s t } _ { \mathcal { G } } ( u , v ) \leq k \right\} .\tag{1}
$$

Prior work often employs the enclosing subgraph $\mathcal { G } _ { e n c } = ( \nu _ { e n c } , \mathcal { E } _ { e n c } )$ ，

$$
\mathcal { V } _ { e n c } = \mathcal { N } ^ { k } ( h ) \cap \mathcal { N } ^ { k } ( t ) ,\tag{2}
$$

$$
\mathcal { E } _ { e n c } = \left\{ ( u , r , v ) \in \mathcal { T } | u , v \in \mathcal { V } _ { e n c } \right\} .\tag{3}
$$

We employ the contextual subgraph $\mathcal { G } _ { c s } = ( \nu _ { c s } , \mathcal { E } _ { c s } )$ obtained from the union of the query endpoints neighborhoods:

$$
\mathcal { V } _ { c s } = \mathcal { N } ^ { k } ( h ) \cup \mathcal { N } ^ { k } ( t ) ,\tag{4}
$$

$$
{ \mathcal { E } } _ { c s } = \left\{ \left( u , r , v \right) \in { \mathcal { T } } | u , v \in \mathcal { V } _ { c s } \right\} .\tag{5}
$$

Thus $\mathcal { V } _ { e n c } \subseteq \mathcal { V } _ { c s }$ , and $\mathcal { G } _ { e n c }$ is the subgraph of $\mathcal { G } _ { c s }$ induced by $\nu _ { e n c }$

## 3.0.3 Link Prediction and Inductive Setting

KGC ranks candidate entities for a query of the form $( h , r _ { q } , ? ) \mathrm { ~ o r ~ } ( ? , r _ { q } , t )$ . We learn a scoring function $f : \mathcal { E } \times \mathcal { R } \times \mathcal { E } $ R such that positive triples receive higher scores than corrupted triples. In the inductive setting considered here, the training and test entity sets are disjoint. The model must therefore rely on transferable structural and relational patterns rather than entity-specific identifiers [26].

## 4 Methodology

In this section, we present PEARL, as illustrated in Fig. 1, a unified framework for inductive link prediction that jointly models contextual structural evidence and multi-hop relational paths. PEARL constructs a richer query-specific context while preserving its core semantics. It then leverages the semantic reasoning capability of a large language model to identify relational paths that are semantically relevant to the target relation, thereby complementing purely structural signals.

PEARL consists of five main components. (i) The Contextual Subgraph Neural Network (CSNN) encodes the union-based contextual subgraph and produces both contextual entity representations and a global subgraph representation. (ii) The LLM-based Path Retrieval and Alignment (LPRA) module retrieves candidate paths from the contextual subgraph, ranks them according to their semantic relevance to the query relation, and constructs a path–entity bipartite graph by aligning the retained paths with their associated contextual entities and the global subgraph node. (iii) The Bipartite Graph Neural Network (BGNN) performs message passing over the path–entity bipartite graph to obtain refined path representations. (iv) The Relation-Aware Path Attention Fusion (RAPF) module adaptively aggregates the refined path representations according to the query relation. (v) The Subgraph Contrastive Learning (SCL) module promotes consistency between two augmented subgraphs to preserve query-relevant structural semantics and suppress noisy contextual perturbations.

## 4.1 Contextual Subgraph Neural Network (CSNN)

## 4.1.1 Entity Node Representation Initialization

For inductive reasoning, each entity node $v \in \mathcal { V } _ { c s }$ is represented by structural and relational features rather than a fixed entity identifier. Let $\mathbf { z } _ { v } \in \mathbb { R } ^ { d _ { z } }$ be the double-radius node label:

$$
\mathbf { z } _ { v } = \left[ \mathrm { o n e h o t } ( \mathrm { d i s t } _ { \mathcal { G } } ( v , h ) ) \ | | \ \mathrm { o n e h o t } ( \mathrm { d i s t } _ { \mathcal { G } } ( v , t ) ) \right] ,
$$

where $d _ { z }$ is the dimension of the concatenated distance encoding. The initial entity node representation is

$$
\mathbf { m } _ { v } ^ { ( 0 ) } = \sigma ( \mathbf { W } _ { 0 } \left[ \mathbf { z } _ { v } \mid \mid \mathbf { s } _ { v } \right] + \mathbf { b } _ { 0 } ) , \qquad \mathbf { W } _ { 0 } \in \mathbb { R } ^ { d \times ( d _ { z } + d ) } .\tag{6}
$$

Here, $\mathbf { s } _ { v } \in \mathbb { R } ^ { d }$ is a query-aware aggregation of the relation embeddings incident to v:

$$
\begin{array} { r l } & { \mathbf { s } _ { v } = \displaystyle \sum _ { r \in \mathcal { R } ( v ) } \alpha _ { v , r } \mathbf { e } _ { r } ^ { ( 0 ) } , } \\ & { \alpha _ { v , r } = \frac { \displaystyle \exp \Big ( ( \mathbf { e } _ { r } ^ { ( 0 ) } ) ^ { \top } \mathbf { e } _ { r _ { q } } ^ { ( 0 ) } / \sqrt { d } \Big ) } { \displaystyle \sum _ { r ^ { \prime } \in \mathcal { R } ( v ) } \exp \Big ( ( \mathbf { e } _ { r ^ { \prime } } ^ { ( 0 ) } ) ^ { \top } \mathbf { e } _ { r _ { q } } ^ { ( 0 ) } / \sqrt { d } \Big ) } . } \end{array}\tag{7}
$$

Here, $\mathcal { R } ( v )$ denotes the set of relation types incident to $v ,$ and ${ \bf e } _ { r _ { q } } ^ { ( 0 ) }$ is the initial trainable embedding of the query relation $r _ { q }$

## 4.1.2 Relation-aware Message Passing

To capture higher-order structural patterns and relational semantics, the node and relation representations are updated iteratively. At layer $\ell \in \{ 1 , \ldots , L \}$ , the update for node i is

$$
\mathbf { c } _ { i j r } ^ { ( \ell ) } = \sigma _ { 1 } \left( \mathbf { W } _ { 1 } ^ { ( \ell ) } \left[ \mathbf { m } _ { i } ^ { ( \ell - 1 ) } \parallel \mathbf { m } _ { j } ^ { ( \ell - 1 ) } \parallel \mathbf { e } _ { r } ^ { ( \ell - 1 ) } \right] + \mathbf { b } _ { 1 } ^ { ( \ell ) } \right) ,\tag{8}
$$

$$
\boldsymbol { \beta } _ { i j r } ^ { ( \ell ) } = \sigma _ { 2 } \Big ( ( \mathbf { w } _ { 2 } ^ { ( \ell ) } ) ^ { \top } \mathbf { c } _ { i j r } ^ { ( \ell ) } + b _ { 2 } ^ { ( \ell ) } \Big ) ,\tag{9}
$$

$$
\mathbf { m } _ { i } ^ { ( \ell ) } = \sum _ { r \in { \mathcal { R } } } \sum _ { j \in { \mathcal { N } } _ { r } ( i ) } \beta _ { i j r } ^ { ( \ell ) } \mathbf { W } _ { r } ^ { ( \ell ) } \phi \Bigl ( \mathbf { m } _ { j } ^ { ( \ell - 1 ) } , \mathbf { e } _ { r } ^ { ( \ell - 1 ) } \Bigr ) ,\tag{10}
$$

$$
\mathbf { e } _ { r } ^ { ( \ell ) } = \mathbf { U } ^ { ( \ell ) } \mathbf { e } _ { r } ^ { ( \ell - 1 ) } .\tag{11}
$$

Here, $\mathcal { N } _ { r } ( i ) = \{ j \ | \ ( j , r , i ) \in \mathcal { E } _ { c s }$ or $( i , r , j ) \in \mathcal { E } _ { c s } \}$ is the relation-specific neighborhood of $i , \phi ( \cdot , \cdot )$ composes node and relation representations, and $\beta _ { i j r } ^ { ( \ell ) }$ controls the contribution of the message from $j$ to i. The matrix $\mathbf { W } _ { r } ^ { ( \ell ) }$ is relation specific, whereas $\mathbf { U } ^ { ( \ell ) }$ updates relation embeddings shared across relation types.

## 4.1.3 Global Subgraph Embedding

Let $\mathbf { M } ^ { ( L ) } = [ \mathbf { m } _ { v } ^ { ( L ) } ] _ { v \in \mathcal { V } _ { c s } } \in \mathbb { R } ^ { | \mathcal { V } _ { c s } | \times d }$ denote the entity node embeddings obtained after L layers of message passing, where $\mathbf { m } _ { v } ^ { ( L ) } \in \mathbb { R } ^ { d }$ is the embedding of node v. Following CoMPILE [17], we further employ a Gated Recurrent Unit (GRU) [33] to refine the propagated node embeddings. The gating mechanism adaptively preserves informative structural features while suppressing redundant information accumulated during message passing. Specifically,

$$
\widetilde { \mathbf { M } } ^ { ( L ) } = \mathbf { G R U } \Big ( \mathbf { M } ^ { ( L ) } \Big ) = [ \widetilde { \mathbf { m } } _ { v } ^ { ( L ) } ] _ { v \in \mathcal { V } _ { c s } } ,\tag{12}
$$

where $\widetilde { \mathbf { M } } ^ { ( L ) } \in \mathbb { R } ^ { | \mathcal { V } _ { c s } | \times d }$ denotes the final node embedding matrix and $\widetilde { \mathbf { m } } _ { v } ^ { ( L ) } \in \mathbb { R } ^ { d }$ is the final embedding of node v. The global embedding of the contextual subgraph is then obtained using an average readout function:

$$
\mathbf { m } _ { \mathcal { G } _ { c s } } = \frac { 1 } { | \mathcal { V } _ { c s } | } \sum _ { v \in \mathcal { V } _ { c s } } \widetilde { \mathbf { m } } _ { v } ^ { ( L ) } \in \mathbb { R } ^ { d } .\tag{13}
$$

Table 1: Path Count Statistics on Various FB15k-237 Subsets.
<table><tr><td>Version</td><td>Avg. Paths</td><td>&gt;3 Paths (%)</td><td>&gt;10 Paths (%)</td></tr><tr><td>V1</td><td>2.7</td><td>16.02</td><td>4.05</td></tr><tr><td>V2</td><td>6.6</td><td>31.71</td><td>14.65</td></tr><tr><td>V3</td><td>10.5</td><td>41.84</td><td>22.43</td></tr><tr><td>V4</td><td>15.1</td><td>47.64</td><td>29.19</td></tr></table>

## 4.2 LLM-based Path Retrieval and Alignment (LPRA)

While the CSNN captures rich contextual structural information, simply extracting more paths from the contextual subgraph does not necessarily improve reasoning. Local subgraphs may contain multiple candidate paths with varying relevance to the query triple, and treating them equally can dilute informative evidence and introduce noisy reasoning signals.

To address this limitation, PEARL introduces an LLM-based Path Retrieval and Alignment (LPRA) module. LPRA leverages the semantic reasoning capability and relational knowledge of large language models to evaluate the semantic compatibility between each candidate path and the query triple. This provides semantic guidance beyond purely structural evidence. The most informative paths are then retained as reliable relational evidence for subsequent path modeling.

We first perform a breadth-first search on the contextual subgraph $\mathcal { G } _ { c s }$ to extract a candidate path set $\mathcal { P } _ { \mathrm { c a n d } } = \{ p _ { 1 } , . . . , p _ { N } \}$ . Each path $p _ { m }$ is defined as a sequence of entities and relations connecting the head and tail entities:

$$
p _ { m } = ( v _ { m , 0 } = h , r _ { m , 1 } , v _ { m , 1 } , \ldots , r _ { m , L _ { m } } , v _ { m , L _ { m } } = t ) ,\tag{14}
$$

where $L _ { m }$ denotes the length of path $p _ { m }$

Although these candidate paths provide diverse multi-hop relational evidence, many of them are redundant or only weakly related to the target relation [24, 25]. As shown in Table 1, the number of candidate paths grows substantially across diferent inductive settings of the FB15k-237 dataset [34]. While the average number of paths is only 2.7 in V1, it increases to 15.1 in V4, making it increasingly dificult to identify truly informative reasoning chains. Directly utilizing all candidate paths would not only introduce noisy reasoning signals but also increase the computational burden of downstream path modeling.

To improve the quality of path reasoning, we employ Qwen3-4B-Instruct [35] as an LLM-based path retriever [36, 37]. The detailed prompt design is provided in supplemental material A. Given the query triple q and the candidate path set $\mathcal { P } _ { \mathrm { c a n d } } = \{ p _ { 1 } , . . . , p _ { N } \}$ , the retriever evaluates the semantic support of each candidate path for the triple and assigns it an importance score:

$$
s _ { q , m } = \mathbf { L } \mathbf { L } \mathbf { M } \mathbf { S } \mathbf { c o r e } \left( q , p _ { m } \right) , \qquad p _ { m } \in \mathcal { P } _ { \mathrm { c a n d } } ,\tag{15}
$$

where $s _ { q , m }$ denotes the semantic importance score of path $p _ { m }$ for query $q .$ We rank all candidate paths in descending order of their importance scores and denote the resulting index permutation as $\pi _ { q } =$ $( \pi _ { q } ( 1 ) , \ldots , \pi _ { q } ( N ) )$ .

According to the ranking results, we retain the top-M candidate paths $\mathcal { P } _ { q } = \{ p _ { \pi _ { q } ( 1 ) } , . . . , p _ { \pi _ { q } ( M ) } \}$ . Each retained path is explicitly aligned with the contextual entity nodes appearing along it, thereby establishing path–entity correspondences.

Based on these paths, we construct a path–entity bipartite graph $\boldsymbol { B } _ { q } = ( \boldsymbol { \mathcal { U } } _ { q } , \boldsymbol { \mathcal { E } } _ { B } )$ . Its node set is defined as $\mathcal { U } _ { q } = \mathcal { V } _ { c s } \cup \{ g \} \cup \mathcal { P } _ { q }$ , where $\nu _ { c s }$ denotes the contextual entity nodes, $\mathcal { P } _ { q }$ is the retained path set, and $g$ is a virtual node representing the global contextual subgraph.

The edge set is defined as

$$
{ \mathcal E } _ { \mathcal B } = \left\{ \left\{ v , p _ { m } \right\} | v \in { \mathcal V } _ { c s } , v \in p _ { m } \right\} \cup \left\{ \left\{ g , p _ { m } \right\} | p _ { m } \in { \mathcal P } _ { q } \right\} ,
$$

which contains two types of connections, namely path–entity (PE) edges and path–global (PG) edges. The path–entity edges connect each relational path with the contextual entities appearing on that path, while the path–global edges link every path to the global contextual node.

## 4.3 Bipartite Graph Neural Network (BGNN)

BGNN performs message passing over the path–entity bipartite graph to model interactions among relational paths, contextual entity nodes, and the global contextual node. Through iterative information propagation, each path representation is adaptively refined according to its surrounding structural context.

## 4.3.1 Path Embedding Initialization

For each retained path

$$
p _ { m } = ( v _ { m , 0 } \mathrm { = } h , r _ { m , 1 } , v _ { m , 1 } , \ldots , r _ { m , L _ { m } } , v _ { m , L _ { m } } \mathrm { = } t ) \in \mathcal { P } _ { q } ,
$$

we encode its relation sequence using a GRU encoder to capture sequential dependencies and directional semantics:

$$
\mathbf { p } _ { m } ^ { ( 0 ) } = \mathbf { G R U } \Bigl ( \mathbf { e } _ { r _ { m , 1 } } ^ { ( L ) } , \ldots , \mathbf { e } _ { r _ { m , L _ { m } } } ^ { ( L ) } \Bigr ) \in \mathbb { R } ^ { d } .\tag{16}
$$

## 4.3.2 Node Initialization

The initial embeddings of the nodes in $B _ { q }$ are defined as

$$
\mathbf { z } _ { i } ^ { ( 0 ) } = \left\{ \begin{array} { l l } { \widetilde { \mathbf { m } } _ { v } ^ { ( L ) } , } & { i = v \in \mathcal { V } _ { c s } , } \\ { \mathbf { m } _ { \mathcal { G } _ { c s } } , } & { i = g , } \\ { \mathbf { p } _ { m } ^ { ( 0 ) } , } & { i = p _ { m } \in \mathcal { P } _ { q } , } \end{array} \right.\tag{17}
$$

where $\widetilde { \mathbf { m } } _ { v } ^ { ( L ) }$ denotes the contextual entity embedding produced by CSNN, $\mathbf { m } _ { \mathcal { G } _ { c s } }$ is the global contextual subgraph embedding, and $\mathbf { p } _ { m } ^ { ( 0 ) }$ is the initial embedding of path $p _ { m }$

## 4.3.3 Message Passing with R-GAT

To preserve the distinct semantic roles of diferent interactions during information propagation, BGNN employs a multi-head Relational Graph Attention Network (R-GAT) [38].

Let $\mathcal { T } _ { B } = \{ \mathrm { P E } , \mathrm { P G } \}$ denote the interaction-type set of the bipartite graph. Accordingly, the type-specific neighborhood of node i is defined as

$$
\mathcal { N } _ { \tau } ^ { \mathcal { B } } ( i ) = \{ j \in \mathcal { U } _ { q } \ | \ \{ i , j \} \in \mathcal { E } _ { B } , \ \tau ( i , j ) = \tau \} .\tag{18}
$$

The interaction type of each edge is determined by the node categories:

$$
\tau ( i , j ) = \left\{ \mathrm { P E } , \quad ( i \in \mathcal { V } _ { c s } , j \in \mathcal { P } _ { q } ) ~ \mathrm { o r } ~ ( i \in \mathcal { P } _ { q } , j \in \mathcal { V } _ { c s } ) , \right.\tag{19}
$$

For each edge $( i , j )$ with interaction type τ , we first compute the attention score to measure the importance of node $j$ to node i under the specific interaction type:

$$
e _ { i j , \tau } ^ { ( \ell , a ) } = \mathrm { L e a k y R e L U } \left( ( \mathbf { a } _ { \tau } ^ { ( \ell , a ) } ) ^ { \top } [ \mathbf { h } _ { i , \tau } ^ { ( \ell , a ) } \parallel \mathbf { h } _ { j , \tau } ^ { ( \ell , a ) } ] \right) ,\tag{20}
$$

where $\mathbf { h } _ { i , \tau } ^ { ( \ell , a ) } = \mathbf { W } _ { \tau } ^ { ( \ell , a ) } \mathbf { z } _ { i } ^ { ( \ell - 1 ) }$ denotes the type-specific transformed representation, and $\mathbf { a } _ { \tau } ^ { ( \ell , a ) }$ is a learnable attention vector.

The attention scores are normalized over all neighbors of node i to obtain the attention coeficients:

$$
\alpha _ { i j , \tau } ^ { ( \ell , a ) } = \frac { \exp ( e _ { i j , \tau } ^ { ( \ell , a ) } ) } { \displaystyle \sum _ { \tau ^ { \prime } \in \mathcal { T } _ { B } } \sum _ { j ^ { \prime } \in \mathcal { N } _ { \tau ^ { \prime } } ^ { B } ( i ) } \exp ( e _ { i j ^ { \prime } , \tau ^ { \prime } } ^ { ( \ell , a ) } ) } .\tag{21}
$$

Using the obtained attention coeficients, the embedding of node i is updated through multi-type message aggregation:

$$
\mathbf { z } _ { i } ^ { ( \ell ) } = \left. \begin{array} { l } { H } \\ { \sigma \left( \displaystyle \sum _ { \tau \in \mathcal { T } _ { \mathcal { B } } } \displaystyle \sum _ { j \in \mathcal { N } _ { \tau } ^ { \mathcal { B } } ( i ) } \alpha _ { i j , \tau } ^ { ( \ell , a ) } \mathbf { W } _ { \tau } ^ { ( \ell , a ) } \mathbf { z } _ { j } ^ { ( \ell - 1 ) } \right) , } \end{array} \right.\tag{22}
$$

where H denotes the number of attention heads, and $\mathbf { W } _ { \tau } ^ { ( \ell , a ) } \in \mathbb { R } ^ { \frac { d } { H } \times d }$ is the transformation matrix associated with interaction type τ. After $L _ { B }$ R-GAT layers, the refined embedding of each retained relational path is obtained as

$$
\widetilde { \mathbf { p } } _ { m } = \mathbf { z } _ { p _ { m } } ^ { ( L _ { B } ) } .\tag{23}
$$

Through type-aware attention propagation over the path–entity bipartite graph, each path embedding selectively aggregates local contextual entity information and global subgraph semantics, enabling identica relational paths to be interpreted diferently under diferent structural environments.

## 4.4 Relation-Aware Path Attention Fusion (RAPF)

The refined path embeddings $\{ \widetilde { \bf p } _ { m } \} _ { m = 1 } ^ { M }$ produced by BGNN encode diverse multi-hop relational semantics between the query entities. To identify the most informative logical evidence for the query relation $r _ { q } ,$ we employ a Scaled Dot-Product Attention mechanism to aggregate these paths into a relation-aware path representation $\mathbf { p } _ { \mathcal { G } _ { c s } } \in \mathbb { R } ^ { d }$

Specifically, the final query-relation embedding ${ \bf e } _ { r _ { q } } ^ { ( L ) } \in \mathbb { R } ^ { d }$ is used as the query, while the refined path embeddings serve as keys and values. The projected query, key, and value vectors are

$$
\mathbf { q } _ { r _ { q } } = \mathbf { W } ^ { Q } \mathbf { e } _ { r _ { q } } ^ { ( L ) } , \qquad \mathbf { k } _ { m } = \mathbf { W } ^ { K } \widetilde { \mathbf { p } } _ { m } , \qquad \mathbf { v } _ { m } = \mathbf { W } ^ { V } \widetilde { \mathbf { p } } _ { m } ,\tag{24}
$$

where $\mathbf { q } _ { r _ { q } } , \mathbf { k } _ { m } , \mathbf { v } _ { m } \in \mathbb { R } ^ { d }$ . The matrices $\mathbf { W } ^ { Q } , \mathbf { W } ^ { K }$ , and $\mathbf { W } ^ { V }$ are learnable projections.

The attention weight assigned to each path is computed as follows:

$$
\alpha _ { m } = \frac { \exp \left( \frac { \mathbf { q } _ { r _ { q } } ^ { \top } \mathbf { k } _ { m } } { \sqrt { d } } \right) } { \sum _ { i = 1 } ^ { M } \exp \left( \frac { \mathbf { q } _ { r _ { q } } ^ { \top } \mathbf { k } _ { i } } { \sqrt { d } } \right) } .\tag{25}
$$

Finally, the aggregated path embedding is obtained as:

$$
{ \bf p } { \mathcal { G } } _ { c s } = \sum _ { m = 1 } ^ { M } \alpha _ { m } { \bf v } _ { m } .\tag{26}
$$

## 4.5 Subgraph Contrastive Learning (SCL)

Existing inductive reasoning methods typically construct enclosing subgraphs based on the intersection of the k-hop entity neighborhoods of the query entities, which preserves reliable shared structural information but may overlook complementary contextual semantics. In contrast, the proposed contextual subgraph $\mathcal { G } _ { c s }$ incorporates nodes from the union of the two entity neighborhoods, enabling richer structural representation and broader contextual coverage. However, directly introducing all contextual entity nodes may also bring irrelevant or noisy structural signals, potentially weakening the core semantics associated with the query triple.

To balance structural enrichment and semantic consistency, PEARL adopts a dual-view subgraph augmentation contrastive learning strategy [39]. Specifically, we partition the entity node set $\nu _ { c s }$ into core nodes and contextual nodes:

$$
\mathcal V _ { e n c } = \mathcal N ^ { k } ( h ) \cap \mathcal N ^ { k } ( t ) , \qquad \mathcal V _ { \Delta } = \mathcal V _ { c s } \setminus \mathcal V _ { e n c } ,\tag{27}
$$

$\nu _ { e n c }$ preserves the shared structural semantics of the query entities, while $\nu _ { \Delta }$ contains complementary contextual information.

Two stochastic subgraph views are then constructed by independently sampling from $\nu _ { \Delta }$ without replacement. Specifically, for each view $a \in \{ 1 , 2 \}$ , we randomly draw exactly $\lfloor | \nu _ { \Delta } | / 2 \rfloor$ nodes from $\nu _ { \Delta }$ to form $\mathcal { V } _ { \Delta } ^ { ( a ) }$ with each subset being sampled uniformly among all possible subsets of that size. The entity node sets of the two subgraph views are then defined as

$$
\mathcal { V } _ { c s } ^ { ( a ) } = \mathcal { V } _ { e n c } \cup \mathcal { V } _ { \Delta } ^ { ( a ) } , \qquad a \in \{ 1 , 2 \} ,\tag{28}
$$

and each view ${ \mathcal G } _ { c s } ^ { ( a ) }$ is the subgraph of $\mathcal { G } _ { c s }$ induced by $\mathcal { V } _ { c s } ^ { ( a ) }$

Since the two augmented subgraphs share the same core structure but difer in the sampled contextual nodes, the model is encouraged to learn embeddings that are invariant to contextual perturbations. For a mini-batch of $B$ query triples, the two augmented subgraphs are encoded by the same CSNN with shared parameters, producing $\mathbf { m } _ { c s , 1 } ^ { ( \bar { b } ) }$ and $\mathbf { m } _ { c s , 2 } ^ { ( b ) }$ , for the b-th query. We optimize the subgraph contrastive loss using the following one-directional InfoNCE objective [40]:

$$
\mathcal { L } _ { \mathrm { S C L } } = - \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \log \frac { \exp \left( \cos \left( \mathbf { m } _ { c s , 1 } ^ { ( b ) } , \mathbf { m } _ { c s , 2 } ^ { ( b ) } \right) / \tau \right) } { \sum _ { c = 1 } ^ { B } \exp \left( \cos \left( \mathbf { m } _ { c s , 1 } ^ { ( b ) } , \mathbf { m } _ { c s , 2 } ^ { ( c ) } \right) / \tau \right) } ,\tag{29}
$$

where $\cos ( \cdot , \cdot )$ denotes cosine similarity and $\tau > 0$ is the temperature parameter. For the anchor embedding $\mathbf { m } _ { c s , 1 } ^ { ( b ) }$ , its counterpart $\mathbf { m } _ { c s , 2 } ^ { ( b ) }$ forms the positive sample, while $\{ \mathbf { m } _ { c s , 2 } ^ { ( c ) } \} _ { c \neq b }$ are treated as in-batch negative samples.

By maximizing consistency across two augmented subgraphs, the proposed contrastive objective encourages PEARL to focus on stable and query-relevant structural semantics while suppressing the influence of unstable contextual noise.

## 4.6 Training and Optimization

## 4.6.1 Joint Scoring Function

To predict the plausibility of a query triple $q = ( h , r _ { q } , t )$ , we first integrate the contextual subgraph embedding and the aggregated path embedding into a unified structural-semantic vector:

$$
\widetilde { \mathbf { s } _ { q } } = [ \mathbf { m } _ { \mathcal { G } _ { c s } } \ | | \ \mathbf { p } _ { \mathcal { G } _ { c s } } ] .\tag{30}
$$

The final prediction score is computed based on the query entities, the query relation, and the fused structural-path representation:

$$
\begin{array} { r } { f ( h , r _ { q } , t ) = \mathbf { w } _ { s } ^ { \top } \left[ \widetilde { \mathbf { m } } _ { h } ^ { ( L ) } \parallel \widetilde { \mathbf { m } } _ { t } ^ { ( L ) } \parallel \mathbf { e } _ { r _ { q } } ^ { ( L ) } \parallel \widetilde { \mathbf { s } _ { q } } \right] + b _ { s } , } \end{array}\tag{31}
$$

where ∥ denotes vector concatenation, $\widetilde { \mathbf { m } } _ { h } ^ { ( L ) }$ and $\widetilde { \mathbf { m } } _ { t } ^ { ( L ) }$ are the embeddings from CSNN, and ${ \bf w } _ { s }$ and $b _ { s }$ are learnable scoring parameters.

## 4.6.2 Objective Function

To train PEARL, for each positive triple $q _ { n } ^ { + } = ( h _ { n } , r _ { n } , t _ { n } ) \in \mathcal { T } _ { \mathrm { t r a i n } }$ , we construct a negative triple by corrupting either the head or the tail:

$$
q _ { n } ^ { - } \in \{ ( h _ { n } ^ { \prime } , r _ { n } , t _ { n } ) , ( h _ { n } , r _ { n } , t _ { n } ^ { \prime } ) \} .\tag{32}
$$

The prediction objective is optimized using a margin-based ranking loss:

$$
\mathcal { L } _ { \mathrm { t a s k } } = \frac { 1 } { \left| \mathcal { T } _ { \mathrm { t r a i n } } \right| } \sum _ { n = 1 } ^ { \left| \mathcal { T } _ { \mathrm { t r a i n } } \right| } \left[ \gamma + f ( q _ { n } ^ { - } ) - f ( q _ { n } ^ { + } ) \right] _ { + } ,\tag{33}
$$

where $[ x ] _ { + } = \operatorname* { m a x } ( 0 , x )$ and $\gamma > 0$ denotes the margin.

To further improve the robustness and semantic consistency of contextual subgraph representations, we additionally incorporate the proposed subgraph contrastive loss $\mathcal { L } _ { \mathrm { S C L } }$ . The overall training objective is defined as:

$$
\begin{array} { r } { \mathcal { L } = \lambda _ { 1 } \mathcal { L } _ { \mathrm { t a s k } } + \lambda _ { 2 } \mathcal { L } _ { \mathrm { S C L } } , } \end{array}\tag{34}
$$

where $\lambda _ { 1 } , \lambda _ { 2 } \geq 0$ control the relative contributions of the two objectives.

Table 2: Performance comparison on WN18RR, FB15k-237, and NELL-995 under the V1 and V3 inductive settings (%). results are reported as mean ± standard deviation. Best results are in bold; second-best are underlined.
<table><tr><td rowspan="3">Methods</td><td colspan="4">WN18RR</td><td colspan="4">FB15k-237</td><td colspan="4">NELL-995</td></tr><tr><td colspan="2">V1</td><td colspan="2">V3</td><td colspan="2">V1</td><td colspan="2">V3</td><td colspan="2">V1</td><td colspan="2">V3</td></tr><tr><td>Hits@10</td><td>MRR</td><td>Hits@10</td><td>MRR</td><td>Hits@10</td><td>MRR</td><td>Hits@10</td><td>MRR</td><td>Hits@10</td><td>MRR</td><td>Hits@10</td><td>MRR</td></tr><tr><td colspan="9">Rule-based Methods:</td><td></td><td></td><td></td><td></td></tr><tr><td>DRUM [13]</td><td>75.36 ± 0.0013</td><td>48.67 ± 0.0006</td><td>47.19 ± 0.0007</td><td>30.73 ± 0.0014</td><td>52.05 ± 0.0025</td><td>33.29 ± 0.0022</td><td>48.56 ± 0.0020</td><td>31.63 ± 0.0014</td><td>40.80 ± 0.0024</td><td>40.76 ± 0.0003</td><td>50.53 ± 0.0018</td><td>31.41 ± 0.0017</td></tr><tr><td>NeuralLP [14]</td><td>75.76 ± 0.0000</td><td>45.83 ± 0.0019</td><td>46.90 ± 0.0011</td><td>30.28 ± 0.0007</td><td>49.56 ± 0.0024</td><td>25.31 ± 0.0028</td><td>48.05 ± 0.0017</td><td>29.60 ± 0.0014</td><td>40.50 ± 0.0000</td><td>40.60 ± 0.0001</td><td>49.94 ± 0.0040</td><td>29.45 ± 0.0024</td></tr><tr><td colspan="9"></td><td></td><td></td><td></td><td></td></tr><tr><td>Subgraph-based Methods:</td><td></td><td></td><td></td><td></td><td>62.92 ± 0.0086</td><td></td><td></td><td>59.62 ± 0.0028</td><td></td><td>43.86 ± 0.0113</td><td></td><td></td></tr><tr><td>GraIL [26]</td><td>82.04 ± 0.0035</td><td>74.69 ± 0.0116</td><td>59.72 ± 0.0027</td><td>54.24 ± 0.0042</td><td></td><td>45.69 ± 0.0053</td><td>80.66 ± 0.0025</td><td></td><td>54.80 ± 0.0057 55.50 ± 0.0054</td><td>48.85 ± 0.0056</td><td>87.00 ± 0.0051</td><td>74.24 ± 0.0057</td></tr><tr><td>CoMPILE [17]</td><td>83.24 ± 0.0033</td><td>76.95 ± 0.0118</td><td>59.75 ± 0.0037</td><td>54.44 ± 0.0036</td><td>64.73 ± 0.0085 63.17 ± 0.0045</td><td>48.01 ± 0.0050 48.55 ± 0.0065</td><td>83.91 ± 0.0036 82.29 ± 0.0031</td><td>63.81 ± 0.0041 62.81 ± 0.0040</td><td>53.50 ± 0.0040</td><td>45.27 ± 0.0020</td><td>94.43 ± 0.0043 93.82 ± 0.0028</td><td>79.44 ± 0.0053 69.74 ± 0.0043</td></tr><tr><td>TACT [16]</td><td>82.45 ± 0.0041 82.45 ± 0.0031</td><td>75.45 ± 0.0115 65.66 ± 0.0111</td><td>58.60 ± 0.0005 58.84 ± 0.0058</td><td>54.29 ± 0.0026 53.84 ± 0.0052</td><td>66.10 ± 0.0029</td><td>47.55 ± 0.0057</td><td>83.06 ± 0.0040</td><td>63.37 ± 0.0029</td><td>59.00 ± 0.0080</td><td>48.87 ± 0.0059</td><td>92.89 ± 0.0008</td><td>78.89 ± 0.0042</td></tr><tr><td>RMPI [18] LCILP [29]</td><td>88.29 ± 0.0024</td><td>77.69 ± 0.0052</td><td>67.27 ± 0.0023</td><td>61.11 ± 0.0039</td><td>59.26 ± 0.0021</td><td>45.84 ± 0.0055</td><td>82.77 ± 0.0014</td><td>56.65 ± 0.0013</td><td>52.00 ± 0.0090</td><td>46.57 ± 0.0024</td><td>87.20 ± 0.0033</td><td>69.38 ± 0.0060</td></tr><tr><td>S2DN [15]</td><td>86.43 ± 0.0089</td><td>78.62 ± 0.0119</td><td>69.01 ± 0.0089</td><td>54.82 ± 0.0050</td><td>64.39 ± 0.0118</td><td>48.59 ± 0.0099</td><td>82.94 ± 0.0033</td><td>59.21 ± 0.0052</td><td>61.00 ± 0.0082</td><td>49.69 ± 0.0135</td><td>91.71 ± 0.0040</td><td>71.83 ± 0.0034</td></tr><tr><td>MINES [30]</td><td>88.45 ± 0.0085</td><td>75.51 ± 0.0095</td><td>62.46 ± 0.0043</td><td>55.44 ± 0.0030</td><td>61.51 ± 0.0065</td><td>46.60 ± 0.0086</td><td>80.31 ± 0.0072</td><td>58.51 ± 0.0077</td><td>59.40 ± 0.0027</td><td>49.77 ± 0.0076</td><td>92.37 ± 0.0036</td><td>64.20 ± 0.0040</td></tr><tr><td colspan="9">Subgraph &amp; Path-based Methods:</td><td></td><td></td><td></td><td></td></tr><tr><td>NBFNet [12]</td><td>83.24 ± 0.0034</td><td>69.50 ± 0.0032</td><td>54.46 ± 0.0132</td><td>43.40 ± 0.0084</td><td>50.63 ± 0.0158</td><td>30.49 ± 0.0100</td><td>61.44 ± 0.0016</td><td>35.27 ± 0.0037</td><td>57.70 ± 0.0075</td><td>50.25 ± 0.0163</td><td></td><td></td></tr><tr><td>CATS [21]</td><td>78.45 ± 0.0054</td><td>19.38 ± 0.0115</td><td>56.77 ± 0.0059</td><td>35.24 ± 0.0053</td><td>54.14 ± 0.0109</td><td>32.96 ± 0.0074</td><td>59.53 ± 0.0053</td><td>34.03 ± 0.0040</td><td>66.10 ± 0.0065</td><td>50.27 ± 0.0031</td><td>49.15 ± 0.0040 94.19 ± 0.0052</td><td>28.28 ± 0.0186</td></tr><tr><td>SNRI [19]</td><td>85.84 ± 0.0060</td><td>75.41 ± 0.0043</td><td>57.68 ± 0.0027</td><td>46.92 ± 0.0028</td><td>63.90 ± 0.0124</td><td>44.76 ± 0.0082</td><td>82.65 ± 0.0034</td><td>59.99 ± 0.0045</td><td>55.50 ± 0.0108</td><td>42.72 ± 0.0045</td><td>93.56 ± 0.0028</td><td>73.43 ± 0.0039 72.23 ± 0.0049</td></tr><tr><td>RPC-IR [20]</td><td>78.73 ± 0.0069</td><td>69.18 ± 0.0060</td><td>64.64 ± 0.0048</td><td>55.08 ± 0.0044</td><td>62.53 ± 0.0085</td><td>41.67 ± 0.0050</td><td>80.80 ± 0.0067</td><td>56.60 ± 0.0036</td><td>60.80 ± 0.0120</td><td>46.33 ± 0.0142</td><td>84.61 ± 0.0075</td><td>71.94 ± 0.0019</td></tr><tr><td>PEARL</td><td>93.48 ± 0.0048</td><td>80.36 ± 0.0029</td><td>81.25 ± 0.0022</td><td>66.82 ± 0.0026</td><td>72.63 ± 0.0060</td><td>50.55 ± 0.0115</td><td>86.31 ± 0.0042</td><td>66.37 ± 0.0030</td><td>63.00 ± 0.0073</td><td>50.77 ± 0.0188</td><td>96.89 ± 0.0095</td><td>75.83 ± 0.0077</td></tr></table>

## 5 Experiments

## 5.1 Datasets and Evaluation

We evaluate PEARL on three widely used IKGC datasets: WN18RR [41], FB15k-237 [34], and NELL-995 [42]. Following GraIL [26], each dataset is divided into four inductive subsets (V1–V4) with increasing scales, where the training and test graphs contain disjoint entity sets and are constructed under a shared global relation vocabulary.

For evaluation, we adopt the standard filtered ranking protocol and report Hits@1, Hits@10, and Mean Reciprocal Rank (MRR), where higher values indicate better performance. All results are averaged over five independent runs. Detailed dataset statistics and evaluation settings are provided in supplemental material B.

## 5.2 Baselines

To evaluate the efectiveness of PEARL, we compare it with representative IKGC methods from three categories:

• Rule-based Methods: NeuralLP [14], DRUM [13].

• Subgraph-based Methods: GraIL [26], CoMPILE [17], TACT [16], RMPI [18], LCILP [29], S<sup>2</sup>DN [15], and MINES [30].

• Subgraph & Path-based Methods: NBFNet [12], SNRI [19], RPC-IR [20], and CATS [21].

Detailed descriptions of all baseline models are provided in supplemental material C.

## 5.3 Implementation Details

The number of message-passing layers is set to L = 3 for CSNN and L<sub>B</sub> = 3 for BGNN. The maximum relational path length is set to $L _ { m } = 2$ , and the R-GAT employs H = 2 attention heads. The hidden dimension d is fixed to 32 for all node, relation, and path embeddings. All experiments are implemented in PyTorch and conducted on a single NVIDIA RTX 4090 GPU. The remaining hyperparameters, including the number of retained paths M, learning rate, batch size, loss coeficients, and the subgraph size k, are selected according to validation performance. Detailed configurations and search ranges are provided in supplemental material E. The source code and implementation details are publicly available at https: //github.com/yychi-code/PEARL.

## 5.4 Results and Analysis

The experimental results on WN18RR, FB15k-237, and NELL-995 are summarized in Table 2. Overall, PEARL achieves the strongest average performance across the evaluated settings. Compared with the strongest competing methods, PEARL achieves relative improvements of 8.67%, 6.01%, and 3.20% in average Hits@10 over the strongest competing methods on WN18RR, FB15k-237, and NELL-995, respectively, demonstrating its strong generalization ability for inductive knowledge graph completion.

Several observations can be drawn from the experimental results. Rule-based methods, including NeuralLP and DRUM, generally achieve relatively lower performance, suggesting that logical rule induction alone may not fully capture the rich structural dependencies surrounding unseen entities. Subgraph-based approaches consistently achieve stronger performance by explicitly exploiting neighborhood structures. GraIL establishes an efective inductive reasoning framework based on enclosing subgraphs. Subsequent methods further enhance subgraph representations from diferent perspectives. Specifically, CoMPILE improves message passing, TACT models topological relation patterns, SNRI incorporates semantic neighboring relations, RMPI develops relational message passing, LCILP adopts PPR-based local clustering, MINES introduces message intercommunication, and S<sup>2</sup>DN employs dual-view structural learning. NBFNet formulates multi-hop reasoning as diferentiable path aggregation and achieves competitive performance across several datasets. More recently, CATS incorporates LLMs to leverage latent type constraints, neighboring facts, and relational paths, demonstrating the potential of LLM-assisted inductive reasoning. Nevertheless, these methods either primarily focus on structural representations or utilize relational paths without explicitly modeling their interactions with the surrounding contextual subgraph. Although RPC-IR further improves path representations through contrastive learning, it primarily focuses on relational paths and does not explicitly model their interactions with the surrounding structural context.

In contrast, PEARL jointly models contextual subgraphs and relational paths within a unified framework. The LLM-guided path retrieval module identifies informative reasoning paths, while BGNN enables contextaware interactions between paths and structural embeddings. Combined with contextual subgraph modeling and subgraph contrastive learning, PEARL efectively exploits both structural and logical evidence, resulting in strong and generally consistent performance across the benchmark datasets.

## 5.5 Eficiency Analysis

We further evaluate the computational eficiency of PEARL from both the training and inference perspectives. We select several competitive methods with strong performance on the FB15k-237 V1 and V3 datasets for eficiency comparison. To ensure a fair comparison, all models are evaluated under the same hardware environment and evaluation protocol. Training is performed on a single NVIDIA RTX 4090 GPU. For inference evaluation, preprocessing and downstream model inference are measured separately, with the latter conducted on a CPU.

For training eficiency, we track the validation performance throughout the training process. Specifically, we record the validation area under the precision–recall curve (AUC-PR) with respect to the accumulated training time. Fig. 2 presents the AUC-PR curves on the FB15k-237 V1 and V3 datasets. On both datasets, PEARL exhibits rapid performance improvement during the early stages of training and consistently achieves higher validation AUC-PR than competing methods under comparable training time budgets. Moreover, PEARL converges faster than most baselines and ultimately attains the highest validation performance. These results indicate that the proposed contextual subgraph modeling and path-aware reasoning framework supports both accurate prediction and efective optimization. In addition, Table 3 reports the peak GPU memory consumption during training. PEARL requires the lowest GPU memory on FB15k-237 V1 and remains comparable to the most memory-eficient baseline on V3, demonstrating favorable memory eficiency. This eficiency is partly attributed to the LLM-guided path retrieval strategy, which reduces unnecessary path-level computations in subsequent reasoning.

For inference eficiency, Table 3 separately reports the preprocessing time and downstream inference time per query triple, together with the corresponding Hits@10 performance on FB15k-237 V1 and V3. The preprocessing time, reported as subgraph extraction time in the table, includes both contextual subgraph construction and LLM-guided path retrieval. The resulting contextual subgraph and retained relational paths are then directly used for subsequent neural inference. Although LLM-guided retrieval introduces additional preprocessing overhead, it removes redundant candidate paths and retains only informative reasoning chains, thereby reducing the complexity of downstream path modeling. As a result, PEARL maintains competitive downstream inference eficiency while achieving the best Hits@10 performance on both inductive settings.

![](images/565fdf84de79a79b000b6ea13698d113e27d1fb5b4eedaf62e655504932e865a.jpg)

![](images/14d4e088acad3cee30c9eea58264056a4b8625fe09ca8222a227e3032b698a42.jpg)  
Figure 2: Validation AUC-PR curves of PEARL and several competitive methods on FB15k-237 datasets. (A) Training convergence curves on FB15k-237 V1; (B) training convergence curves on FB15k-237 V3.

Table 3: The computational costs and prediction performance of several models on the FB15k-237 V1 and V3 datasets.
<table><tr><td rowspan="2">Model</td><td colspan="2">Peak GPU Memory (GB)</td><td colspan="2">Inference Time (ms)</td><td colspan="2">Subgraph Extraction Time (ms)</td><td colspan="2">Hits@10</td></tr><tr><td>V1</td><td>V3</td><td>V1</td><td>V3</td><td>V1</td><td>V3</td><td>V1</td><td>V3</td></tr><tr><td>CoMPILE</td><td>1.33</td><td>5.57</td><td>23</td><td>104</td><td>90</td><td>120</td><td>64.73</td><td>83.91</td></tr><tr><td>SNRI</td><td>2.50</td><td>13.47</td><td>80</td><td>192</td><td>52</td><td>53</td><td>63.90</td><td>82.65</td></tr><tr><td>RMPI</td><td>3.83</td><td>15.33</td><td>64</td><td>70</td><td>111</td><td>144</td><td>66.10</td><td>83.06</td></tr><tr><td>TACT</td><td>2.86</td><td>23.03</td><td>65</td><td>72</td><td>50</td><td>61</td><td>63.17</td><td>82.29</td></tr><tr><td>S2DN</td><td>2.03</td><td>9.49</td><td>250</td><td>570</td><td>59</td><td>84</td><td>64.39</td><td>82.94</td></tr><tr><td>PEARL</td><td>1.14</td><td>5.59</td><td>46</td><td>73</td><td>105</td><td>140</td><td>72.63</td><td>86.31</td></tr></table>

![](images/9be4a22b1f5e06cf0dca2b1aac9eadcff7bb0274200d0dce5b621af532afb72d.jpg)

![](images/b40ae8076bff365dbaeb642d57dbf5174affdd47be06f3995c4a27d0050c506c.jpg)

Figure 3: Comparison between enclosing subgraphs and contextual subgraphs on WN18RR, FB15k-237, and NELL-995, together with their corresponding numbers of nodes. The red and green nodes denote the head and tail entities of the query triple, respectively. Blue nodes represent the entities belonging to the intersection of the k-hop neighborhoods of the query entities, while purple nodes denote the additional contextual entities introduced from the neighborhood union.  
![](images/f2190fa1ea627d2c1970848247f8f93f036ac5fbc2472e4c80261a638e3fbccc.jpg)

![](images/5433a3f6039a9abca935c8a9c9822bd360efedf2b1ab187f75342b9651da3c2d.jpg)  
Figure 4: Ablation study of PEARL on WN18RR, FB15k-237, and NELL-995. Error bars represent scaled standard deviations. (A) Ablation on subgraph modeling, evaluating the efectiveness of contextual subgraph construction and subgraph contrastive learning. (B) Ablation on path-aware reasoning, including variants without relational path information, LLM-guided path retrieval, or the BGNN-based path–entity–global interaction mechanism.

## 5.6 Ablation Studies

To comprehensively evaluate the contribution of each major design in PEARL, we conduct ablation studies from two complementary perspectives: (1) subgraph modeling and (2) path-aware reasoning. The corresponding results are summarized in Fig. 4.

## 5.6.1 Ablation on Subgraph Modeling

To evaluate the efectiveness of contextual subgraph modeling and subgraph contrastive learning, we compare the following variants:

A. (The Artist, award\_won, Academy Award for Best Picture):

Table 4: Case study of LLM-guided relational path ranking.
<table><tr><td>Query Triple</td><td>Candidate Relation Path</td><td>Importance Score</td><td>Rank</td></tr><tr><td rowspan="5">(Conan O&#x27;Brien, gender, Male)</td><td>Profession link → gender</td><td>0.95</td><td>1</td></tr><tr><td>Marriage link → gender</td><td>0.90</td><td>2</td></tr><tr><td>Friendship link → gender</td><td>0.85</td><td>3</td></tr><tr><td>Religion link → film → gender</td><td>0.80</td><td>4</td></tr><tr><td>Influenced by → gender</td><td>0.75</td><td>5</td></tr><tr><td rowspan="4">(Beasts of the Southern Wild, film release region, Japan)</td><td>film release region → location contains → film release region</td><td>0.95</td><td>1</td></tr><tr><td>Netflix genre titles → film release region</td><td>0.60</td><td>2</td></tr><tr><td>Film release region (Continent) → location adjoins → film</td><td>0.55</td><td>3</td></tr><tr><td>release region</td><td></td><td></td></tr></table>

B. (Deadwood, award\_won, Primetime Emmy Award):

![](images/0c6f51a47e16c6b63db9a1c3d3c9d66c51329f5ab124f0fdae4ee1ed930af6e0.jpg)  
Figure 5: Visualization of path attention weights for two query triples sharing the same target relation (award\_won) on FB15k-237.

• PEARL w/o CS: This variant replaces the contextual subgraph with the conventional enclosing subgraph and removes the subgraph contrastive learning objective.

• PEARL w/o SCL: This variant retains the contextual subgraph but removes the subgraph contrastive learning objective.

As shown in Fig. 4(A), PEARL consistently outperforms both variants across all datasets. The performance gap between PEARL w/o CS and PEARL w/o SCL shows that introducing the contextual subgraph provides additional structural evidence beyond the conventional enclosing subgraph. Fig. 3 further illustrates their structural diferences on WN18RR, FB15k-237, and NELL-995. Specifically, the contextual subgraph preserves the shared neighborhoods of the query entities while incorporating additional entities from the union of their k-hop neighborhoods, thereby providing broader structural context. Meanwhile, the further improvement of PEARL over PEARL w/o SCL demonstrates the benefit of subgraph contrastive learning in mitigating the noisy contextual information introduced by subgraph expansion. Additional visualization and quantitative analyses of the learned representations are provided in supplemental material D.

## 5.6.2 Ablation on Path-aware Reasoning

To further investigate the contribution of relational path reasoning, we compare the following variants:

• PEARL w/o Path: This variant removes relational path information and performs prediction using only the contextual subgraph representation.

• PEARL w/o LLM: This variant removes the LLM-guided path retrieval module and randomly selects M paths from the extracted candidate paths for subsequent reasoning.

• PEARL w/o BGNN: This variant removes the BGNN module and directly uses the initial embeddings of the retrieved paths for prediction.

As shown in Fig. 4(B), removing relational path information consistently degrades performance, confirming that multi-hop paths provide complementary reasoning evidence beyond contextual subgraphs. Replacing LLM-guided path retrieval with random selection of M candidate paths also leads to lower performance and larger standard deviations, indicating that semantic path retrieval helps identify more informative reasoning evidence and improves robustness compared with unguided path selection. Among the three variants, removing BGNN results in the largest performance degradation, highlighting the importance of explicitly modeling interactions between relational paths and contextual structural information. Overall, these results demonstrate that path information, LLM-guided retrieval, and path–entity interaction play complementary roles in the proposed path-aware reasoning framework.

## 5.7 Case Study

## 5.7.1 LLM-guided Path Retrieval

To assess the efectiveness of the proposed LLM-guided path retrieval module, we analyze representative query triples together with their candidate relational paths. The examples presented in Table 4 illustrate how the LLM estimates the semantic relevance of candidate paths by assigning importance scores with respect to the query triple.

For the query triple (Conan O’Brien, gender, Male), the highest importance scores are assigned to paths involving Profession link, Marriage link, and Friendship link, since these relations provide direct semantic evidence for inferring the entity’s gender. In contrast, paths related to Religion and Influenced by receive considerably lower scores due to their weaker semantic relevance.

A similar trend can be observed for the query triple (Beasts of the Southern Wild, film release region, Japan). The path Film release region → location contains → film release region obtains the highest importance score because it directly captures the geographical containment relation required for predicting the release region. By comparison, the remaining candidate paths provide only indirect or less relevant evidence, resulting in substantially lower importance scores.

These examples indicate that the LLM efectively distinguishes informative reasoning paths from semantically weak candidates. Consequently, the subsequent BGNN operates on a more reliable set of relational paths, reducing the influence of noisy reasoning signals while preserving the most discriminative logical evidence.

## 5.7.2 Relation-Aware Path Attention Analysis

To further investigate how structural context influences the interpretation of relational paths, we analyze two query triples on FB15k-237. They share the same target relation but involve diferent entity pairs: (The Artist, award\_won, Academy Award for Best Picture) and (Deadwood, award\_won, Primetime Emmy Award). Since both queries share the same relation, the RAPF query vector remains identical, allowing us to isolate the influence of contextual subgraph structures on path importance.

As illustrated in Fig. 5, the two queries exhibit noticeably diferent attention distributions over candidate relational paths. For The Artist, the path award\_won → netflix\_genre → award\_won receives the highest attention weight, suggesting that genre-related evidence plays the dominant role in this structural context. In contrast, the two paths involving award\_nomination receive relatively lower attention.

For Deadwood, however, the attention distribution changes substantially. The paths containing award\_nomination become the dominant reasoning evidence, indicating that nomination-related information is more informative under this local structural context than genre similarity.

These observations demonstrate that the semantic contribution of a relational path is jointly determined by its relation sequence and the surrounding contextual subgraph. Even for the same triple relation, PEARL adaptively emphasizes diferent reasoning paths according to the local structural environment, thereby enabling context-conditioned path interpretation rather than context-invariant path reasoning. Moreover, the consistent efectiveness of PEARL across relations with diferent occurrence frequencies further verifies its robustness in exploiting relational evidence under diverse knowledge graph structures. Additional results are provided in supplemental material F.

## 6 Conclusion

This paper investigates inductive knowledge graph completion from the perspective of context-aware relational reasoning and proposes PEARL, a framework that jointly models contextual subgraphs and relational paths for inductive reasoning over unseen entities. Extensive experiments on WN18RR, FB15k-237, and NELL-995 demonstrate that PEARL achieves strong overall performance against competitive baselines. This confirms that jointly exploiting structural context and relational paths provides an efective paradigm for inductive knowledge graph completion. In particular, the LLM-guided retrieval mechanism enables PEARL to identify relation-relevant paths from noisy local structures, while the subsequent path–entity interaction further adapts their semantics to the query-specific context. This design provides a unified framework for integrating LLM-derived semantic guidance with context-aware graph reasoning. Nevertheless, the current framework relies on ofline LLM-based path retrieval and has not yet been evaluated on dynamic or temporal knowledge graphs. Future work will focus on more eficient adaptive path retrieval and extending PEARL to continual and temporal inductive reasoning scenarios.

## References

[1] H. Wang, M. Zhao, X. Xie, W. Li, and M. Guo, “Knowledge graph convolutional networks for recommender systems,” in The world wide web conference, 2019, pp. 3307–3313.

[2] X. Wang, T. Huang, D. Wang, Y. Yuan, Z. Liu, X. He, and T.-S. Chua, “Learning intents behind interactions with knowledge graph for recommendation,” in Proceedings of the web conference 2021, 2021, pp. 878–887.

[3] S. Vashishth, S. Sanyal, V. Nitin, and P. Talukdar, “Composition-based multi-relational graph convolutional networks,” arXiv preprint arXiv:1911.03082, 2019.

[4] Y. Zhang, H. Dai, Z. Kozareva, A. Smola, and L. Song, “Variational reasoning for question answering with knowledge graph,” in Proceedings of the AAAI conference on artificial intelligence, vol. 32, no. 1, 2018.

[5] Y. Li, “Research and analysis of semantic search technology based on knowledge graph,” in 2017 IEEE International Conference on Computational Science and Engineering (CSE) and IEEE Internationa Conference on Embedded and Ubiquitous Computing (EUC), vol. 1. IEEE, 2017, pp. 887–890.

[6] L. Thingbaijam, K. Palle, P. V. Prasad, B. Mallala, S. Patil et al., “Incorporating knowledge graphs in semantic search,” in 2024 15th International Conference on Computing Communication and Networking Technologies (ICCCNT). IEEE, 2024, pp. 1–6.

[7] B. Shi and T. Weninger, “Open-world knowledge graph completion,” in Proceedings of the AAAI conference on artificial intelligence, vol. 32, no. 1, 2018.

[8] J. Baek, D. B. Lee, and S. J. Hwang, “Learning to extrapolate knowledge: Transductive few-shot out-of-graph link prediction,” Advances in neural information processing systems, vol. 33, pp. 546–560, 2020.

[9] D. Xu, C. Ruan, E. Korpeoglu, S. Kumar, and K. Achan, “Product knowledge graph embedding for e-commerce,” in Proceedings of the 13th international conference on web search and data mining, 2020, pp. 672–680.

[10] Y. Zhang, Z. Zhou, Q. Yao, X. Chu, and B. Han, “Adaprop: Learning adaptive propagation for graph neural network based knowledge graph reasoning,” in Proceedings of the 29th ACM SIGKDD conference on knowledge discovery and data mining, 2023, pp. 3446–3457.

[11] J. Lee, C. Chung, and J. J. Whang, “Ingram: Inductive knowledge graph embedding via relation graphs,” in International conference on machine learning. PMLR, 2023, pp. 18 796–18 809.

[12] Z. Zhu, Z. Zhang, L.-P. Xhonneux, and J. Tang, “Neural bellman-ford networks: A general graph neural network framework for link prediction,” Advances in neural information processing systems, vol. 34, pp. 29 476–29 490, 2021.

[13] A. Sadeghian, M. Armandpour, P. Ding, and D. Z. Wang, “Drum: End-to-end diferentiable rule mining on knowledge graphs,” Advances in neural information processing systems, vol. 32, 2019.

[14] F. Yang, Z. Yang, and W. W. Cohen, “Diferentiable learning of logical rules for knowledge base reasoning,” Advances in neural information processing systems, vol. 30, 2017.

[15] T. Ma, Y. Chen, L. Wang, X. Lin, B. Song, and X. Zeng, “S<sup>2</sup>dn: Learning to denoise unconvincing knowledge for inductive knowledge graph completion,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 12, 2025, pp. 12 355–12 363.

[16] J. Chen, H. He, F. Wu, and J. Wang, “Topology-aware correlations between relations for inductive link prediction in knowledge graphs,” in Proceedings of the AAAI conference on artificial intelligence, vol. 35, no. 7, 2021, pp. 6271–6278.

[17] S. Mai, S. Zheng, Y. Yang, and H. Hu, “Communicative message passing for inductive relation reasoning,” in Proceedings of the AAAI conference on artificial intelligence, vol. 35, no. 5, 2021, pp. 4294–4302.

[18] Y. Geng, J. Chen, J. Z. Pan, M. Chen, S. Jiang, W. Zhang, and H. Chen, “Relational message passing for fully inductive knowledge graph completion,” in 2023 IEEE 39th international conference on data engineering (ICDE). IEEE, 2023, pp. 1221–1233.

[19] X. Xu, P. Zhang, Y. He, C. Chao, and C. Yan, “Subgraph neighboring relations infomax for inductive link prediction on knowledge graphs,” in Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI-22, L. D. Raedt, Ed. International Joint Conferences on Artificial Intelligence Organization, 7 2022, pp. 2341–2347, main Track. [Online]. Available: https://doi.org/10.24963/ijcai.2022/325

[20] Y. Pan, J. Liu, L. Zhang, T. Zhao, Q. Lin, X. Hu, and Q. Wang, “Inductive relation prediction with logical reasoning using contrastive representations,” in Proceedings of the 2022 conference on empirical methods in natural language processing, 2022, pp. 4261–4274.

[21] M. Li, C. Yang, C. Xu, Z. Song, X. Jiang, J. Guo, H.-f. Leung, and I. King, “Context-aware inductive knowledge graph completion with latent type constraints and subgraph reasoning,” in Proceedings of the Thirty-Ninth AAAI Conference on Artificial Intelligence and Thirty-Seventh Conference on Innovative Applications of Artificial Intelligence and Fifteenth Symposium on Educational Advances in Artificial Intelligence. AAAI Press, 2025. [Online]. Available: https://doi.org/10.1609/aaai.v39i11.33318

[22] J. Pujara, E. Augustine, and L. Getoor, “Sparsity and noise: Where knowledge graph embeddings fall short,” in Proceedings of the 2017 conference on empirical methods in natural language processing, 2017, pp. 1751–1756.

[23] B. Xue and L. Zou, “Knowledge graph quality management: a comprehensive survey,” IEEE Transactions on Knowledge and Data Engineering, vol. 35, no. 5, pp. 4969–4988, 2022.

[24] Y. Liu, X. Lin, Y. Shang, Y. Li, S. Wang, and Y. Cao, “Pathmind: A retrieve-prioritize-reason framework for knowledge graph reasoning with large language models,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 18, p. 15386–15393, Mar. 2026. [Online]. Available: https://ojs.aaai.org/index.php/AAAI/article/view/38565

[25] X. Tan, X. Wang, Q. Liu, X. Xu, X. Yuan, and W. Zhang, “Paths-over-graph: Knowledge graph empowered large language model reasoning,” in Proceedings of the ACM on Web Conference 2025, 2025, pp. 3505–3522.

[26] K. Teru, E. Denis, and W. Hamilton, “Inductive relation prediction by subgraph reasoning,” in International conference on machine learning. PMLR, 2020, pp. 9448–9457.

[27] L. A. Galárraga, C. Teflioudi, K. Hose, and F. Suchanek, “Amie: association rule mining under incomplete evidence in ontological knowledge bases,” in Proceedings of the 22nd international conference on World Wide Web, 2013, pp. 413–422.

[28] C. Meilicke, M. Fink, Y. Wang, D. Rufinelli, R. Gemulla, and H. Stuckenschmidt, “Fine-grained evaluation of rule-and embedding-based systems for knowledge graph completion,” in International semantic web conference. Springer, 2018, pp. 3–20.

[29] H. A. Mohamed, D. Pilutti, S. James, A. Del Bue, M. Pelillo, and S. Vascon, “Locality-aware subgraphs for inductive link prediction in knowledge graphs,” Pattern Recognition Letters, vol. 167, pp. 90–97, 2023.

[30] K. Liang, L. Meng, S. Zhou, W. Tu, S. Wang, Y. Liu, M. Liu, L. Zhao, X. Dong, and X. Liu, “Mines: Message intercommunication for inductive relation reasoning over neighbor-enhanced subgraphs,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 9, pp. 10 645–10 653, Mar. 2024. [Online]. Available: https://ojs.aaai.org/index.php/AAAI/article/view/28935

[31] G. Wang, R. Ying, J. Huang, and J. Leskovec, “Multi-hop attention graph neural networks,” in Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21, Z.-H. Zhou, Ed. International Joint Conferences on Artificial Intelligence Organization, 8 2021, pp. 3089–3096, main Track. [Online]. Available: https://doi.org/10.24963/ijcai.2021/425

[32] Y. Yang, L. Li, J. Wu, and C. Qu, “Misapp: Multi-hop intent-aware session graph learning for next app prediction,” arXiv preprint arXiv:2603.21653, 2026.

[33] K. Cho, B. Van Merriënboer, Ç. Gulçehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio, “Learning phrase representations using rnn encoder–decoder for statistical machine translation,” in Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), 2014, pp. 1724–1734.

[34] K. Toutanova, D. Chen, P. Pantel, H. Poon, P. Choudhury, and M. Gamon, “Representing text for joint embedding of text and knowledge bases,” in Proceedings of the 2015 conference on empirical methods in natural language processing, 2015, pp. 1499–1509.

[35] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, C. Zheng, D. Liu, F. Zhou, F. Huang, F. Hu, H. Ge, H. Wei, H. Lin, J. Tang, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Zhou, J. Lin, K. Dang, K. Bao, K. Yang, L. Yu, L. Deng, M. Li, M. Xue, M. Li, P. Zhang, P. Wang, Q. Zhu, R. Men, R. Gao, S. Liu, S. Luo, T. Li, T. Tang, W. Yin, X. Ren, X. Wang, X. Zhang, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Zhang, Y. Wan, Y. Liu, Z. Wang, Z. Cui, Z. Zhang, Z. Zhou, and Z. Qiu, “Qwen3 technical report,” 2025. [Online]. Available: https://arxiv.org/abs/2505.09388

[36] W. Sun, L. Yan, X. Ma, S. Wang, P. Ren, Z. Chen, D. Yin, and Z. Ren, “Is chatgpt good at search? investigating large language models as re-ranking agents,” in Proceedings of the 2023 conference on empirical methods in natural language processing, 2023, pp. 14 918–14 937.

[37] H. Liu, S. Wang, Y. Zhu, Y. Dong, and J. Li, “Knowledge graph-enhanced large language models via path selection,” in Findings of the Association for Computational Linguistics: ACL 2024, 2024, pp. 6311–6321.

[38] P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio, “Graph Attention Networks,” International Conference on Learning Representations, 2018, accepted as poster. [Online]. Available: https://openreview.net/forum?id=rJXMpikCZ

[39] Y. You, T. Chen, Y. Sui, T. Chen, Z. Wang, and Y. Shen, “Graph contrastive learning with augmentations,” in Advances in Neural Information Processing Systems, H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin, Eds., vol. 33. Curran Associates, Inc., 2020, pp. 5812–5823. [Online]. Available: https://proceedings.neurips.cc/paper/2020/file/3fe230348e9a12c13120749e3f9fa4cd-Paper.pdf

[40] Z. Wang, B. Xu, Y. Yuan, H. Shen, and X. Cheng, “Infonce is a free lunch for semantically guided graph contrastive learning,” in Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval, ser. SIGIR ’25. New York, NY, USA: Association for Computing Machinery, 2025, p. 719–728. [Online]. Available: https://doi.org/10.1145/3726302.3730007

[41] T. Dettmers, P. Minervini, P. Stenetorp, and S. Riedel, “Convolutional 2d knowledge graph embeddings,” in Proceedings of the AAAI conference on artificial intelligence, vol. 32, no. 1, 2018.

[42] W. Xiong, T. Hoang, and W. Y. Wang, “Deeppath: A reinforcement learning method for knowledge graph reasoning,” in Proceedings of the 2017 conference on empirical methods in natural language processing, 2017, pp. 564–573.

## Supplemental Materials

## A LLM-guided Path Retrieval Prompt

To improve the quality of relational path reasoning, PEARL employs Qwen3-4B-Instruct as a semantic path retriever. Given a query triple and a set of candidate relational paths extracted from the contextual subgraph, the LLM is instructed to evaluate the semantic relevance of each candidate path with respect to the query triple by assigning an importance score. The candidate paths are then ranked according to these scores, and the resulting ranking is used to prioritize informative reasoning evidence. Table 5 presents the prompt template used for path scoring and ranking.

Table 5: Prompt Template for LLM-guided Path Retrieval.
<table><tr><td>Role System</td><td>Content</td></tr><tr><td></td><td>You are an expert in knowledge graphs, specializing in relation reasoning and path ranking. Given a query triple and multiple candidate relational paths, assign an importance score in the range [0,1] to each candidate path according to how strongly it supports the query triple. A higher score indicates stronger semantic support. Return a valid JSON object containing the candidate path indices together with their corresponding importance scores, sorted in descending order of score. For example: { &quot;path_ranking&quot;:[  $\{ " \mathrm { i n d e x " } : 0 , " \mathrm { s c o r e " } : 0 . 9 7 \}$   $\{ " \mathrm { i n d e x } " : 3 , " \mathrm { s c o r e } " : 0 . 8 8 \}$   $\{ " \mathrm { i n d e x } " : 1 , " \mathrm { s c o r e } " : 0 . 7 3 \}$ </td></tr><tr><td>User</td><td>} The returned indices must correspond exactly to the provided candidate paths. Do not generate non-existent indices or additional paths. The input contains: (1) the query head entity h, (2) the query relation r, (3) the query tail entity t, and (4) a set of candidate relational paths represented as relation sequences from h to t. Evaluate the semantic relevance of each candidate path to the query triple by assigning an importance score, and rank all candidate paths from the strongest to the weakest supporting evidence. Only use the</td></tr></table>

## B Dataset and Evaluation Protocol

We evaluate PEARL on three widely used IKGC benchmarks: WN18RR, FB15k-237, and NELL-995. Following GraIL, each dataset is divided into four inductive subsets (V1–V4) of increasing scale. For each subset, the training and test graphs are constructed under a shared global relation vocabulary while containing disjoint entity sets. Table 6 reports the detailed statistics of all datasets, including the numbers of relations, entities, and triples in the training and test graphs.

We follow the standard inductive evaluation setting introduced by GraIL. During training, validation, and testing, reverse relations are added following prior work. To ensure eficient model selection, we evaluate checkpoints on the validation set using the Area Under the Precision–Recall Curve (AUC-PR). Specifically, positive triples and sampled negative triples are used to construct a binary classification task, and the checkpoint with the highest validation AUC-PR is selected for final testing. For the final evaluation, we adopt the filtered ranking protocol and report Hits@1, Hits@10, and Mean Reciprocal Rank (MRR). All reported results are averaged over five independent runs, and the corresponding standard deviations are reported.

Table 6: Inductive Benchmark Statistics. (REL: Relations, ENT: Entities, TR: Triples).
<table><tr><td rowspan="2">Version</td><td rowspan="2">Split</td><td colspan="3">WN18RR</td><td colspan="3">FB15k-237</td><td colspan="3">NELL-995</td></tr><tr><td>REL</td><td>ENT</td><td>TR</td><td>REL</td><td>ENT</td><td>TR</td><td>REL</td><td>ENT</td><td>TR</td></tr><tr><td>V1</td><td>train</td><td>9</td><td>2,746</td><td>6,678</td><td>183</td><td>2,000</td><td>5,226</td><td>14</td><td>10,915</td><td>5,540</td></tr><tr><td></td><td>test</td><td>9</td><td>922</td><td>1,991</td><td>146</td><td>1,500</td><td>2,404</td><td>14</td><td>225</td><td>1,034</td></tr><tr><td>V2</td><td>train</td><td>10</td><td>6,954</td><td>18,968</td><td>203</td><td>3,000</td><td>12,085</td><td>88</td><td>2,564</td><td>10,109</td></tr><tr><td></td><td>test</td><td>10</td><td>2,923</td><td>4,863</td><td>176</td><td>2,000</td><td>5,092</td><td>79</td><td>4,937</td><td>5,521</td></tr><tr><td>V3</td><td>train</td><td>11</td><td>12,078</td><td>32,150</td><td>218</td><td>4,000</td><td>22,394</td><td>142</td><td>4,647</td><td>20,117</td></tr><tr><td></td><td>test</td><td>11</td><td>5,084</td><td>7,470</td><td>187</td><td>3,000</td><td>9,137</td><td>122</td><td>4,921</td><td>9,668</td></tr><tr><td>V4</td><td>train</td><td>9</td><td>3,861</td><td>9,842</td><td>222</td><td>5,000</td><td>33,916</td><td>77</td><td>2,092</td><td>9,289</td></tr><tr><td></td><td>test</td><td>9</td><td>7,208</td><td>15,157</td><td>204</td><td>3,500</td><td>14,554</td><td>61</td><td>3,294</td><td>8,520</td></tr></table>

## C Details of Baseline Methods

## C.0.1 Rule-based Methods

• NeuralLP: An end-to-end diferentiable rule learning framework based on TensorLog that learns logical rules for multi-hop reasoning.

• DRUM: An extension of diferentiable rule learning that employs tensor products to learn logical rules of varying lengths while sharing parameters across path lengths.

## C.0.2 Subgraph-based Methods

• GraIL: A pioneering IKGC model that performs reasoning over enclosing subgraphs.

• CoMPILE: A subgraph-based model that strengthens the message-passing mechanism by explicitly modeling the directional information of relations within the local structure.

• TACT: A topology-aware inductive link prediction method that models correlations between relations within local subgraphs.

• RMPI: A relational message passing framework for inductive link prediction that reasons over subgraphs surrounding target entity pairs.

• LCILP: A locality-aware subgraph method that employs Personalized PageRank (PPR)-based local clustering to extract informative neighborhoods around target links.

• S<sup>2</sup>DN: A dual-view subgraph neural network that balances core and auxiliary structural information.

• MINES: A neighbor-enhanced subgraph reasoning framework with message intercommunication mechanisms.

## C.0.3 Subgraph & Path-based Methods

• NBFNet: A neural Bellman-Ford framework that generalizes path-based reasoning to subgraph-level inference via diferentiable path aggregation over the enclosing subgraph.

• SNRI: A subgraph neighbor representation learning method that captures semantic information of unseen entities through neighboring relations.

• RPC-IR: A relational path contrastive learning framework that enhances path representations via contrastive objectives.

• CATS<sup>1</sup>: An LLM-based IKGC framework that performs reasoning based on latent type constraints, relational paths, and neighboring facts.

## D Further Analysis of Contextual Subgraph Modeling

To further examine the impact of SCL, Fig. 6 visualizes the learned subgraph representations on FB15k-237. Without SCL, contextual subgraph embeddings show larger deviations from the core structural semantics captured by enclosing subgraphs. In contrast, PEARL with SCL achieves better alignment while maintaining richer contextual information, indicating that contrastive learning efectively reduces the influence of noisy contextual perturbations.

We additionally employ Positive Consistency $\mathcal { C } _ { \mathrm { p o s } }$ and Separation S to quantitatively evaluate representation quality. For a mini-batch of B query triples, let $\mathbf { m } _ { c s } ^ { ( b ) }$ and $\mathbf { m } _ { e n c } ^ { ( b ) }$ denote the contextual and enclosing subgraph embeddings of the b-th query, respectively. Positive Consistency is defined as:

$$
\mathcal { C } _ { \mathrm { p o s } } = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \cos \left( \mathbf { m } _ { c s } ^ { ( b ) } , \mathbf { m } _ { e n c } ^ { ( b ) } \right) ,\tag{35}
$$

whereas Separation S is computed as:

$$
\mathcal { S } = \mathcal { C } _ { \mathrm { p o s } } - \frac { 1 } { B ( B - 1 ) } \sum _ { b \neq c } \cos \left( \mathbf { m } _ { c s } ^ { ( b ) } , \mathbf { m } _ { c s } ^ { ( c ) } \right) .\tag{36}
$$

As shown in Fig. 6, PEARL achieves higher $\mathcal { C } _ { \mathrm { p o s } }$ and S than PEARL w/o SCL, demonstrating that the proposed contrastive objective improves semantic consistency with core structural patterns while preserving discriminative contextual representations.

A. PEARL w/o CS  
![](images/7c6a125aba1c94f085c55c3c7337a04ae6efe138ff94783451761945792c1458.jpg)

B. PEARL w/o SCL  
![](images/c76d3f873eea558cf3a977667e5d25dcd615b01eb74da8ba69b0386422ccdf8e.jpg)

C. PEARL  
![](images/322067e31ada34fff31aafb29f93ee5a0f707647536716ae0c06a19a46930501.jpg)

<table><tr><td>Variant</td><td> $\mathcal { C } _ { \mathrm { p o s } } \uparrow$ </td><td>S↑</td></tr><tr><td>PEARL  $\mathrm { w / o }$  SCL</td><td>0.3950</td><td>-0.3965</td></tr><tr><td>PEARL</td><td>0.5424</td><td>-0.2507</td></tr></table>

Figure 6: Visualization and quantitative analysis of learned global subgraph embeddings on FB15k-237. The upper part presents heatmaps of the subgraph embeddings learned by diferent variants: (A) PEARL $\mathrm { w / o }$ Contextual Subgraph (PEARL $\mathrm { w / o }$ CS), which uses conventional enclosing subgraphs; (B) PEARL $\mathrm { w / o }$ Subgraph Contrastive Learning (PEARL $\mathrm { w / o }$ SCL), which employs contextual subgraphs without contrastive regularization; and (C) PEARL, which incorporates both contextual subgraph modeling and subgraph contrastive learning. For a consistent comparison, all variants are evaluated on the same set of query triples. The lower part reports the positive consistency $( C _ { \mathrm { p o s } } )$ and separation (S) metrics.

## E Hyperparameter Sensitivity

In this subsection, we analyze the sensitivity of PEARL to diferent hyperparameter settings on the V1 subsets of WN18RR, FB15k-237, and NELL-995. The final hyperparameter configurations adopted for all datasets are summarized in Table 7.

Impact of batch size. We first investigate the influence of batch size by varying it from 8 to 64. As shown in Fig. 7 (A), PEARL exhibits relatively stable performance across diferent batch sizes on all datasets. The best performance is achieved with a batch size of 8 on WN18RR and NELL-995, while FB15k-237 obtains the optimal result with a batch size of 32.

Impact of learning rate. We further evaluate the efect of the learning rate within the range from 0.0005 to 0.01. Fig. 7 (B) shows that excessively large learning rates slightly degrade the model performance on all datasets. PEARL achieves the best performance with a learning rate of 0.001 on WN18RR and NELL-995, while FB15k-237 performs best with a smaller learning rate of 0.0005.

Impact of subgraph size. To investigate the influence of structural receptive fields, we vary the subgraph hop size k from 1 to 4. As illustrated in Fig. 7 (C), PEARL achieves the best performance with $k = 4$ on WN18RR, k = 3 on FB15k-237, and $k = 2$ on NELL-995. These results indicate that the optimal structural receptive field depends on the structural characteristics and density of diferent datasets.

Impact of path number. We study the influence of the number of relational paths by varying M from 2 to 5. As shown in Fig. 7(D), WN18RR and FB15k-237 achieve their best performance with $M = 3$ , while NELL-995 benefits from retaining more paths and performs best with $M = 5$ . Accordingly, we set $M = 3$ for WN18RR and FB15k-237, and $M = 5$ for NELL-995 in our experiments.

Impact of loss balancing coeficients. We investigate the influence of the loss balancing coeficients $\lambda _ { 1 }$ and $\lambda _ { 2 } .$ , which control the contributions of the primary link prediction loss $\mathcal { L } _ { t a s k }$ and the subgraph contrastive loss $\mathcal { L } _ { S C L }$ , respectively. Fig. 8 reports the Hits@10 performance under diferent coeficient combinations on WN18RR V1, FB15k-237 V1, and NELL-995 V1. For WN18RR, the best performance is achieved with a relatively small contrastive weight. In contrast, FB15k-237 and NELL-995 favor larger values of $\lambda _ { 2 } ,$ indicating a stronger benefit from contrastive regularization on these datasets.

![](images/8df1b264749f7af86e2890309927b8e92f37a7bd26f4914e98a9550de090705b.jpg)  
Figure 7: Hyperparameter sensitivity analysis of PEARL on WN18RR V1, FB15k-237 V1, and NELL-995 V1. (A) Efect of batch size; (B) efect of learning rate; (C) efect of subgraph size k; and (D) efect of the number of retrieved relational paths.

## F Impact of Relation Frequency

To further investigate the efectiveness of PEARL under varying structural conditions, we conduct a fine-grained analysis based on the frequency of query relations in the training set. We hypothesize that high-frequency relations tend to exhibit more diverse and complex local subgraph patterns, where the benefit of contextua modeling becomes more pronounced.

Specifically, we partition test triples into three groups according to the training-set frequency of their query relations: bottom 25%, middle 50%, and top 25%. We then evaluate PEARL and several competitive baselines on the V1 subsets of WN18RR, FB15k-237, and NELL-995. As reported in Table 8, PEARL achieves the best or competitive performance across most relation-frequency groups.

The gains are particularly evident in the high-frequency group of WN18RR and the medium- and highfrequency groups of FB15k-237. On NELL-995, PEARL achieves the best performance in the low- and high-frequency groups while remaining competitive in the middle-frequency group, indicating its efectiveness across diferent relation-frequency regimes.

Overall, these results demonstrate the robustness of PEARL across diferent relation-frequency regimes and show that the proposed framework can efectively exploit relational and structural evidence under diverse frequency distributions. By anchoring relational paths within the contextual subgraph, PEARL adaptively leverages informative structural signals even when the distribution of relational patterns varies, leading to more reliable inductive reasoning.

Table 7: DATASET-SPECIFIC HYPERPARAMETER SETTINGS OF PEARL.
<table><tr><td>Hyperparameters</td><td>WN18RR</td><td>FB15k-237</td><td>NELL-995</td></tr><tr><td>Subgraph hop size k</td><td>4</td><td>3</td><td>2</td></tr><tr><td>learning rate</td><td>0.001</td><td>0.0005</td><td>0.001</td></tr><tr><td>batch size</td><td>8</td><td>32</td><td>8</td></tr><tr><td>Loss weights  $( \lambda _ { 1 } , \lambda _ { 2 } )$ </td><td>1.0, 0.2</td><td>0.6, 0.2</td><td>0.8, 0.6</td></tr><tr><td>Number of retained paths M</td><td>3</td><td>3</td><td>5</td></tr></table>

![](images/bde91f0f59f3bc5f7012652e84e4648534076adb3c2b3e921b3f8d26c4fc7e5c.jpg)

![](images/c161c0f481c06b0a57df94f0fa4347e64a3f397ef9fb1500380ca71a2365f76b.jpg)

![](images/cb8b25df1d28bf5bb269923f7543eb59afde234e36e45dee7a9d6ddf985f4773.jpg)  
Figure 8: Hyperparameter sensitivity analysis with respect to the loss weights $\lambda _ { 1 }$ and $\lambda _ { 2 } .$ . (A) Hits@10 on WN18RR V1. (B) Hits@10 on FB15k-237 V1. (C) Hits@10 on NELL-995 V1.

Table 8: Impact of relation frequency on inductive link prediction (Hits@10, %). Relations are grouped into the bottom 25%, middle 50%, and top 25% according to their occurrence frequency in the training set. The best results are highlighted in bold.
<table><tr><td rowspan="2">Methods</td><td colspan="3">WN18RR V1</td><td colspan="3">FB15k-237 V1</td><td colspan="3">NELL-995 V1</td></tr><tr><td>Bottom 25%</td><td>Middle 50%</td><td>Top 25%</td><td>Bottom 25%</td><td>Middle 50%</td><td>Top 25%</td><td>Bottom 25%</td><td>Middle 50%</td><td>Top 25%</td></tr><tr><td>CoMPILE</td><td>100.00</td><td>92.30</td><td>82.75</td><td>52.61</td><td>53.84</td><td>72.01</td><td>58.47</td><td>58.33</td><td>56.89</td></tr><tr><td>SNRI</td><td>100.00</td><td>96.15</td><td>84.77</td><td>39.47</td><td>46.15</td><td>71.64</td><td>50.00</td><td>56.89</td><td>52.54</td></tr><tr><td>RMPI</td><td>100.00</td><td>92.31</td><td>83.33</td><td>39.47</td><td>39.42</td><td>60.45</td><td>79.17</td><td>55.17</td><td>58.86</td></tr><tr><td>TACT</td><td>100.00</td><td>92.31</td><td>83.33</td><td>39.47</td><td>55.77</td><td>72.76</td><td>54.17</td><td>56.90</td><td>55.08</td></tr><tr><td>S²DN</td><td>100.00</td><td>94.61</td><td>85.63</td><td>49.47</td><td>55.26</td><td>71.26</td><td>75.00</td><td>60.34</td><td>61.01</td></tr><tr><td>PEARL</td><td>100.00</td><td>96.15</td><td>89.65</td><td>57.89</td><td>59.61</td><td>77.31</td><td>83.33</td><td>58.60</td><td>62.71</td></tr></table>