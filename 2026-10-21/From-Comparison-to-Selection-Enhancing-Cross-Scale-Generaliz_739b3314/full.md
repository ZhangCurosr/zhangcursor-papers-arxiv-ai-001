# From Comparison to Selection: Enhancing Cross-Scale Generalization of Pointer Networks Trained with Reinforcement Learning for Capacitated Drone Delivery Routing

Sofiene Lassoued \*<sup>a</sup>, Devara Tertia<sup>a</sup>, Stefan Lier <sup>b</sup>, Andreas Schwung <sup>a</sup>

<sup>a</sup>South Westphalia University of Applied Sciences, Automation Technology and Learning Systems Luebecker Ring 2, Soest, 59494, North Rhine-Westphalia, Germany

<sup>b</sup>South Westphalia University of Applied Sciences, Logistik und Supply Chain Management Lindenstr.53, Meschede, 59872, North Rhine-Westphalia, Germany

## Abstract

Unmanned aerial vehicles (UAVs) enable fast, flexible last-mile and emergency delivery, but their routing remains challenging as<sup>6</sup> the number of clients varies. Classical approaches based on mixed-integer programming and heuristics can scale poorly or require problem-specific design, motivating learning-based alternatives. Reinforcement learning is particularly promising, provided<sub>2</sub> a single policy can generalise across problem sizes without retraining. Although attention-based Pointer Networks are inherently size-agnostic, their solution quality degrades under both upscaling and downscaling. Using the Capacitated Vehicle Routing Prob-<sup>c</sup> lem (CVRP) as a controlled proxy, with fixed capacity reflecting drone payload constraints, we investigate this degradation and identify three contributing factors: shifts in node density, attention dispersion, and dilution of the mean-pooled graph context. We systematically compare architecture-wide, encoder-side, and decoder-side modifications to address these limitations and propose2 a learned, permutation-invariant graph-context aggregation mechanism, implemented as a weighted average and as Set2Set. Results reveal distinct requirements for transfer in each direction: entropy-based attention scaling reduces the CVRP20-to-CVRP100 performance gap by 1.41 percentage points, while the proposed aggregation methods achieve the largest improvement on the moreA challenging CVRP100-to-CVRP20 transfer, approximately 0.9 percentage points. Exploiting this complementarity, we introduce. an approach selector that chooses the best model for each training–target size pair. The selector reduces cost relative to the refer ence model across all nine seen train–test pairs and eight of nine pairs involving unseen sizes (CVRP35, CVRP75, and CVRP200),<sup>[</sup> achieving improvements of up to 3.2%.

Keywords: Drone delivery, Reinforcement learning, Pointer network, Graph-context aggregation, Capacitated vehicle routing,<sup>v</sup> Size generalization, Approach selection

## 1. Introduction

The growth of e-commerce, on-demand services, and urban logistics pushes delivery operators to plan routes under tight constraints on capacity, distance, energy, and cost. Unmanned aerial vehicles complement ground vehicles because they bypass road congestion and fly directly between locations, which is valuable for medical, emergency, rural, and last-mile delivery, but they bring limited payload and battery range and sen-<sup>r</sup> sitivity to weather, airspace regulation, and multi-drone coordination [1, 2]. This work investigates AI-based route planning for drone networks under such constraints. From an optimization standpoint, the task is closely related to the Capacitated Vehicle Routing Problem (CVRP): a fleet must serve customers with known demands within a capacity limit at minimal total route length [3, 4]. CVRP omits battery dynamics, charging infrastructure, and air corridors, but it isolates the capacityconstrained routing core and is a standard controlled benchmark for routing algorithms, including learning-based ones [5, 6].

Exact methods scale poorly on this NP-hard problem [7, 8], and heuristics need problem-specific, hand-tuned design [9,

10]. Neural Combinatorial Optimization (NCO) instead trains a model to construct solutions directly, using reinforcement learning (RL) so that no optimal-solution labels, which are expensive for large CVRP instances, are needed [11, 6, 12].

The central dificulty for a drone-routing policy is that the number of clients is not fixed: it varies from day to day and region to region. Training one policy per client count is not viable, so the policy must construct solutions for a variable, a priori unknown number of nodes, which is a known weakness of neural routing solvers [13]. A prominent size-agnostic approach is the Pointer Network with attention, which points to one of the input nodes at each decoding step instead of choosing from a fixed vocabulary, so the same trained policy can be applied to graphs of any size [14, 6]. Trained with RL, it has been benchmarked at scale in recent studies of neural routing solvers [15, 16], which show that it solves instances of the training size well but that generalization across problem sizes remains a clear bottleneck in both directions: performance degrades when the model is upscaled to larger graphs and when it is downscaled to smaller ones.

This paper therefore investigates which modifications of the attention-based Pointer Network, applied architecture-wide, in the encoder or in the decoder, reduce the upscaling and the downscaling generalization gap for CVRP, and how their complementary strengths can be combined in one size-agnostic routing solver.

The study focuses on the encoder–decoder architecture of the attention-based Pointer Network: the encoder embeds the nodes of an instance, and the decoder uses these embeddings to point to the next node at each step. This architecture is kept as the common frame, and every approach modifies it at one of three levels: architecture-wide, acting on the attention of both encoder and decoder; encoder-side, replacing how node embeddings are computed; or decoder-side, changing how the decoder summarises the instance and builds its query. To this end, we first identify the sources of the cross-size degradation of the reference policy, then assign to each source the levers that act on it, and finally compare all levers head-to-head under an identical training and evaluation protocol, scored by how much they reduce the cross-size optimality gap of the reference in each direction.

Contributions. The reference policy, the attention-scaling factors and the two alternative encoders are taken from the literature [6, 17, 18, 19] and are re-implemented in one common framework. The contributions of this paper are:

(i) A learned, permutation-invariant graph-context aggregation for the attention-model decoder, realised as a weighted average with a scoring network and as a Set2Set summary. It replaces the mean-pooled context and gives the largest gain of all evaluated approaches on the hardest downscaling transfer while keeping same-size quality.

(ii) A diagnosis-driven, head-to-head comparison of seven levers at the architecture-wide, encoder and decoder levels of the Pointer Network, with fixed vehicle capacity so that only the number of clients changes. It shows that upscaling and downscaling respond to diferent levers.

(iii) An approach selector that exploits this complementarity by choosing, for a given training size and target size, the approach with the lowest estimated cost. It lowers the cost of the reference on seen and unseen problem sizes.

This paper is structured as follows: Section 2 reviews related work, and Section 3 introduces the necessary background on graph neural networks, attention and pointer decoding. Section 4 formulates the CVRP as a sequential decision problem, gives the training objective and describes the reference policy. Section 5 derives the sources of cross-size degradation and presents the evaluated approaches together with the proposed graph-context aggregation and approach selector. Section 6 details the experimental setup, Section 7 reports and discusses the results, and Section 8 concludes the paper and outlines future work.

## 2. Related Work

## 2.1. Combinatorial optimization and routing

Combinatorial optimization (CO) covers problems with a combinatorial explosion of feasible solutions [20]. Routing problems, notably the Traveling Salesman Problem and the Vehicle Routing Problem, are among the most studied because of their relevance to logistics [4, 21]. The Capacitated VRP (CVRP) adds a capacity constraint that partitions customers across routes [3, 22]. CVRP has been addressed with exact methods [23, 7], heuristics such as the savings algorithm and 2- opt/3-opt [22, 24], and metaheuristics such as tabu search and hybrid genetic search [25, 10]. Nevertheless, exact methods scale poorly and heuristics need problem-specific tuning [4, 9].

## 2.2. Machine learningfor combinatorial optimization

Machine learning has assisted classical solvers by learning branching decisions [26, 27], cutting-plane selection [28, 29], solver configuration and algorithm selection [30, 31], and warm starts [32], but the solver itself remains classical. NCO instead uses a neural network as the solver, learning to construct a solution end to end [12, 33].

Early NCO approaches for routing used size-agnostic Pointer Networks [14] (Section 3). Bello et al. trained a Pointer-Network policy with actor-critic RL for TSP [11], and Nazari et al. extended it to CVRP with capacity-aware masking [5]. Kool et al. replaced the recurrent encoder with a Transformer-style multi-head-attention (MHA) encoder and trained the resulting attention-model policy with REINFORCE and a rollout baseline, with strong results on TSP, CVRP, the Orienteering Problem, and the Prize-Collecting TSP [6, 34]. This architecture underlies RL4CO [16] and is the reference architecture used throughout this paper.

## 2.3. The size-generalization bottleneck in attention-based NCO

A systematic evaluation of neural routing solvers shows that the attention-based Pointer-Network policy performs well on instances matching its training size, but not when the graph size changes at test time [15], as also reported by RL4CO [16]. The degradation is bidirectional: models trained on small graphs lose quality when upscaled, and models trained on large graphs do not transfer cleanly when downscaled. NCO models more broadly fail on unseen sizes [13]. Meta-learning [13] and test-time adaptation [35] address varying task distributions, in cluding size variation. At the level of the attention computation, Wang et al. attribute part of the degradation to attention dispersion as the number of nodes grows [36], and Xiao et al. counteract it with an entropy-based scaling factor applied to the attention scores at inference time [17]. Architectural work asks whether the encoder or decoder composition can be changed: Berto et al. show that the encoder choice afects size generalization [16], and Luo et al. propose a Light-Encoder-Heavy-Decoder (LEHD) composition, in contrast to the Heavy-Encoder-Light-Decoder design of Kool et al., with substantially improved generalization under supervised learning [37]. The Set Transformer [19] ofers a further route to size-independent encoding through a fixed number of inducing points.

## 2.4. Research gap

Most studies target a single component in isolation and evaluate same-size performance or one transfer direction. In addition, attention-scaling methods have been evaluated in settings where vehicle capacity grows with the problem size, which confounds the efect of the number of nodes with that of the demand regime. The graph-level context that the decoder reuses at every step, a mean over node embeddings in the attention model, has received less attention as a lever for size generalization. Finally, algorithm selection is established for classical solvers [31], but, to our knowledge, it has not been used to combine NCO variants that generalize in diferent directions. This paper closes these gaps with a learned graph-context aggregation, a controlled comparison of levers in both directions, and an approach selector built on that comparison.

## 3. Theoretical Background

## 3.1. Node embeddings and graph neural networks

Neural routing policies do not operate on raw node features directly. An encoder first maps the features $x _ { \nu }$ of every node v of a graph $G = ( V , E )$ to an embedding $h _ { \nu } \in \mathbb { R } ^ { D }$ that also reflects the node’s relation to the rest of the graph. Because the embedding dimension D is fixed, the same downstream layers can process graphs with any number of nodes, which is the basis of size-agnostic policies. A graph-level representation is then obtained with a permutation-invariant readout, most simply the mean of all node embeddings.

Graph neural networks (GNNs) compute node embeddings by message passing [38, 39]. In each layer k, every node aggregates the embeddings of its neighbours $N ( \nu )$ and combines them with its own,

$$
h _ { \nu } ^ { ( k ) } = \phi ^ { ( k ) } \bigl ( h _ { \nu } ^ { ( k - 1 ) } , \mathrm { \ A G G } ^ { ( k ) } ( \{ h _ { u } ^ { ( k - 1 ) } : u \in N ( \nu ) \} ) \bigr ) ,\tag{1}
$$

so that after K layers the embedding summarises the K-hop neighbourhood of the node. Graph convolutional networks (GCNs) [40] realise Eq. 1 with a fixed aggregation weighted by the normalised adjacency matrix,

$$
H ^ { ( k ) } = \sigma ( \tilde { D } ^ { - 1 / 2 } \tilde { A } \tilde { D } ^ { - 1 / 2 } H ^ { ( k - 1 ) } W ^ { ( k ) } ) ,\tag{2}
$$

where ${ \tilde { A } } = A + I$ is the adjacency matrix with self-loops, $\tilde { D }$ its degree matrix and $W ^ { ( k ) }$ a learned weight matrix. Since the propagation matrix is built from one specific graph, GCNs were introduced for learning on a fixed graph.

GraphSAGE [18] instead learns how to combine the embeddings of a node’s neighbours, so that the learned rule can be applied to nodes and graphs that were not seen during training. With the mean aggregator, one layer reads

$$
\begin{array} { l } { { \displaystyle h _ { N ( \nu ) } ^ { ( k ) } = \frac { 1 } { | N ( \nu ) | } \sum _ { u \in N ( \nu ) } h _ { u } ^ { ( k - 1 ) } } , } \\ { { \displaystyle h _ { \nu } ^ { ( k ) } = \sigma \Big ( W ^ { ( k ) } [ h _ { \nu } ^ { ( k - 1 ) } | | h _ { N ( \nu ) } ^ { ( k ) } ] \Big ) } , } \end{array}\tag{3}
$$

where ∥ denotes concatenation. To bound the computation, the original method uses only a random sample of fixed size from each neighbourhood. Each layer mixes a node’s embedding with those of its neighbours, so when many layers are stacked, the embeddings of all nodes become nearly identical and the nodes can no longer be distinguished. This efect, known as over-smoothing [41], is why GNN encoders use only a few layers.

## 3.2. Multi-head attention and attention scaling

Attention was introduced for sequence alignment in neural machine translation [42] and generalized to Multi-Head Attention (MHA) in the Transformer [34]. From node embeddings X, MHA computes queries $Q \ = \ X W ^ { Q }$ , keys ${ \cal K } = { \cal X } W ^ { \cal K }$ and values $V = X W ^ { V }$ and applies scaled dot-product attention,

$$
\mathrm { A t t e n t i o n } ( Q , K , V ) = \mathrm { s o f t m a x } \left( { \frac { Q K ^ { \top } } { \sqrt { d _ { k } } } } \right) V ,\tag{4}
$$

in h parallel heads whose outputs are concatenated,

$$
\mathbf { M H A } ( X ) = \mathbf { C o n c a t ( h e a d _ { 1 } , \dots , h e a d _ { { h } } ) } \mathbf { \Lambda } _ { W } \mathbf { ^ { O } } .\tag{5}
$$

In self-attention, every node is updated with information from every other node at $O ( N ^ { 2 } )$ cost. Because the softmax in Eq. 4 normalises over all N keys, the sharpness of the attention distribution depends on N. Xiao et al. [17] therefore multiply the attention scores by a size-dependent entropy-based scaling factor (ESF),

$$
\begin{array} { r } { \mathrm { A t t e n t i o n } _ { \lambda } ( Q , K , V ) = \mathrm { s o f t m a x } \left( \lambda _ { n } \frac { Q K ^ { \top } } { \sqrt { d _ { k } } } \right) V , } \\ { \lambda _ { n } = \log _ { n _ { \mathrm { t r a i n } } } n = \frac { \log n } { \log n _ { \mathrm { t r a i n } } } , } \end{array}\tag{6}
$$

where $n$ is the size of the current instance and $n _ { \mathrm { { t r a i n } } }$ the training size. Since $\lambda _ { n } = 1$ for $n = n _ { \mathrm { t r a i n } } .$ the factor leaves the trained model unchanged on its training size and can be applied at inference time.

## 3.3. Attention on sets: Set Transformer and Set2Set

A set of nodes has no natural order, so models that process it should not depend on the order in which the nodes are presented. The Set Transformer [19] builds such a model from multi-head attention blocks (MAB). Its Induced Set Attention Block (ISAB) routes all interaction between the n elements through m learnable inducing points $\ b { I } \in \mathbb { R } ^ { m \times d }$

$$
\begin{array} { r l } { \mathrm { M A B } ( X , Y ) = \mathrm { L N } ( H + \mathrm { r F F } ( H ) ) , } & { } \\ { H = \mathrm { L N } ( X + \mathrm { M u l t i h e a d } ( X , Y , Y ) ) , } & { } \\ { \mathrm { I S A B } _ { m } ( X ) = \mathrm { M A B } ( X , \mathrm { M A B } ( I , X ) ) , } & { } \end{array}\tag{7}
$$

where LN is layer normalization, rFF a row-wise feed-forward network, and X and Y provide the queries and the keys/values, respectively. The inner block summarises the n elements into m vectors and the outer block lets every element attend to that summary, an $n  m  n$ bottleneck whose width does not depend on n and whose cost is O(nm) instead of $O ( n ^ { 2 } )$ .

Set2Set [43] is an order-invariant readout that summarises a set of embeddings $\{ h _ { i } \}$ into one vector. Over T processing steps, an LSTM state attends over the set and reads out a weighted sum,

$$
\begin{array} { r l r } & { q _ { t } = \mathrm { L S T M } ( q _ { t - 1 } ^ { * } ) , } & { \alpha _ { i , t } = \mathrm { s o f t m a x } _ { i } ( h _ { i } ^ { \top } q _ { t } ) , } \\ & { r _ { t } = \sum _ { i } \alpha _ { i , t } h _ { i } , } & { q _ { t } ^ { * } = [ q _ { t } \parallel r _ { t } ] , } \end{array}\tag{8}
$$

and $q _ { T } ^ { * }$ is the summary. Because each step attends again with an updated query, the readout can capture more than a single average.

## 3.4. Route decoding: pointer attention

Routing outputs are permutations of a variable-size input; the Pointer Network [14] therefore uses attention as a pointer over input positions,

$$
p ( \pi _ { t } = i \mid \pi _ { < t } , X ) = \frac { \exp ( \tilde { u } _ { i } ^ { t } ) } { \sum _ { j \in \mathcal { A } _ { t } } \exp ( \tilde { u } _ { j } ^ { t } ) } , \tilde { u } _ { i } ^ { t } = \left\{ \begin{array} { l l } { u _ { i } ^ { t } , } & { i \in \mathcal { A } _ { t } } \\ { - \infty , } & { \mathrm { e l s e } , } \end{array} \right.\tag{9}
$$

where $u _ { i } ^ { t }$ is the attention score of input i at step t and $\mathcal { A } _ { t }$ the set of inputs that may still be selected; all other inputs are masked. Because the output distribution is defined over whatever inputs are present, the same model applies to inputs of any size. The attention-model policy of Kool et al. [6] combines an MHA encoder with such a pointer decoder (Section 4.3).

## 4. Problem Formulation and Reference Policy

## 4.1. CVRP as a sequential decision problem

A CVRP instance is a weighted graph $G ~ = ~ ( V , E )$ with $V = \{ 0 , 1 , \ldots , n \}$ , where node 0 is the depot and nodes $1 , \ldots , n$ are customers with demands $d _ { i }$ . The cost between two nodes is their Euclidean distance $c _ { i j } = \sqrt { ( x _ { i } - x _ { j } ) ^ { 2 } + ( y _ { i } - y _ { j } ) ^ { 2 } } ,$ . Each vehicle has capacity Q, and every route $r _ { k }$ must satisfy $\textstyle \sum _ { i \in r _ { k } } d _ { i } \leq$ Q. A feasible solution $\textit { S } = \{ r _ { 1 } , \ldots , r _ { m } \}$ visits every customer exactly once, with every route starting and ending at the depot, and minimizes the total cost $\begin{array} { r } { C ( S ) = \sum _ { k = 1 } ^ { m } C ( r _ { k } ) } \end{array}$ , where a route $r _ { k } = ( i _ { 1 } , \dots , i _ { L _ { k } } )$ costs $\begin{array} { r } { C ( r _ { k } ) = c _ { 0 i _ { 1 } } + \sum _ { t = 1 } ^ { L _ { k } - 1 } c _ { i _ { t } i _ { t + 1 } } + c _ { i _ { L _ { k } } 0 } . } \end{array}$ The solution can equivalently be written as a depot-separated sequence, $\mathrm { e . g . } \ ( 0 , 3 , 5 , 2 , 0 , 4 , 1 , 6 , 0 )$ , so that CVRP becomes a sequential decision process: a policy selects one node at a time, the remaining capacity decreases with each customer and resets at each depot visit. This capacity-masked formulation is the one used by the attention-model policy and implemented in the RL4CO environment (Section 6).

## 4.2. Training objective

Following standard practice for attention-model routing policies, the policy is trained with REINFORCE and a rollout baseline [44, 6]. For a problem instance x sampled from a distribution $P ( x )$ , and a policy $\pi _ { \theta }$ that samples a solution a with cost $R ( a , x )$ , the gradient of the training objective is

$$
\nabla _ { \theta } \mathcal { L } ( \theta \mid x ) = \mathbb { E } _ { \pi _ { \theta } ( a \mid x ) } [ \left( R ( a , x ) - b ( x ) \right) \nabla _ { \theta } \log \pi _ { \theta } ( a \mid x ) ] ,\tag{10}
$$

where $b ( x )$ is a baseline (the cost obtained by a frozen, periodically-updated copy of the best-performing policy, decoded greedily) used to reduce gradient variance. During training, actions are sampled from the categorical distribution over feasible nodes to encourage exploration; during validation and testing, actions are selected greedily.

## 4.3. Reference policy: the attention model

As the policy for this decision process and the reference point for the head-to-head comparison, we adopt the attention-model policy of Kool et al. [6], trained with Eq. 10. The encoder first embeds depot and customer nodes with two separate linear layers of equal output dimension D (customers carry three raw features: coordinates and demand; the depot carries two), then processes the concatenated embeddings through L MHA layers (Eqs. 4 and 5) with residual connections [45], batch normalization, and a feed-forward sub-layer, producing final node embeddings $H = [ h _ { 0 } , h _ { 1 } , \ldots , h _ { n } ]$

The decoder pre-computes, once per instance, a graph-level context g and glimpse key/value/logit projections from H. The reference graph context uses mean pooling over node embeddings followed by a linear projection, $g ~ = ~ W _ { g } \bar { h } + b _ { g }$ with $\begin{array} { r } { \bar { h } \ = \ \frac { 1 } { N } \sum _ { i } { h _ { i } } } \end{array}$ . At each decoding step t, a dynamic step context concatenates the embedding of the previously visited node with the vehicle’s remaining capacity $c _ { t } , s _ { t } = [ h _ { a _ { t } }$ ∥c<sub>t</sub>], projects it linearly, and adds it to the static graph context to form the glimpse query $q _ { t } ^ { g l i m p s e } \ = \ g + \ W _ { s } s _ { t } \overline { { + \not p } } _ { s }$ . This query attends (via Eq. 4, with a mask over infeasible nodes) over the precomputed glimpse keys/values to produce logits $u _ { i } ^ { t }$ over the candidate nodes (Eq. 9), from which the next node is sampled (training) or chosen greedily (evaluation).

## 5. Sources of Degradation and Approaches

Fig. 1 summarises the policy and the locations of the evaluated approaches. A CVRP instance is given as a graph of the depot and the clients with their coordinates and demands. The encoder maps every node of this graph to a latent representation, the node embedding. From these embeddings the decoder forms a context that combines a summary of the whole graph with the current state of the vehicle. At each step, this context is compared with the embeddings of all feasible nodes to obtain one score per node, and a softmax over these scores gives the probability of each node to be visited next, from which the next client or a return to the depot is selected. The approaches studied in this paper modify this pipeline at three levels: architecture-wide in the attention of encoder and decoder (Section 5.2), in the encoder that produces the embeddings (Section 5.3), and in the decoder aggregation that forms the graph context (Section 5.4).

## 5.1. Sources of cross-size degradation

The reference policy (Section 4.3) contains several compo nents whose behaviour depends on the number of nodes. From its structure and from the literature we derive three candidate sources of degradation, which define the levers studied in this paper; Section 7.2 confirms them empirically.

The first source, S1 (node density), describes what changes between sizes. With fixed capacity and demand range, a larger instance places more and closer nodes in the same area and needs more tours, so a policy trained on one size meets a diferent density at test time. The other two sources are mechanisms inside the policy that react to this change, and the approaches in this paper act on these two mechanisms. S2 (attention dispersion): the softmax in Eqs. 4 and 9 normalises over all nodes, so attention becomes flatter as N grows [36]. S3 (graph-context dilution): the mean-pooled context weights every node equally, so with more nodes it moves towards the same average vector and carries less instance-specific information; since the decoder reuses it at every step, this loss afects every decision.

![](images/bc4ade4d0186e3f133495aef496aab4c916ff10fe7b37aec5fffd2a7af79df72.jpg)  
Figure 1: Overview of the encoder-decoder attention-model policy: the reference architecture together with the architecture-wide, encoder-side and decoder-side approaches evaluated in this work.

Table 1: Approaches, the degradation source they address and their origin. A: architecture-wide, E: encoder-side, D: decoder-side. Origin: citation, taken from the literature; proposed, contribution of this paper.
<table><tr><td>Approach</td><td></td><td>Source Origin</td></tr><tr><td></td><td>A Entropy-based scaling (ESF) S2</td><td>[17]</td></tr><tr><td></td><td>A Learnable scaling (LSF) S2</td><td>[17] (suggested)</td></tr><tr><td></td><td>E GraphSAGE encoder S2</td><td>[18]</td></tr><tr><td></td><td>E Set Transformer encoder</td><td>S2, S3 [19]</td></tr><tr><td></td><td>D Weighted-average context S3</td><td>proposed</td></tr><tr><td>D Set2Set context</td><td>S3</td><td>proposed</td></tr><tr><td>D GRU context</td><td>S3</td><td>proposed</td></tr><tr><td>Approach selector</td><td>all</td><td>proposed</td></tr></table>

Table 1 assigns each approach to the source it addresses and to its origin: approaches marked as proposed are contributions of this paper, whereas the cited approaches are taken from the literature and re-implemented in one common framework. Each changes only the component under test; everything else is identical to the reference. The selector (Section 5.5) then combines the trained approaches.

## 5.2. Architecture-wide approaches: size-aware attention

Attention dispersion (S2) is the most direct consequence of a change in $N ,$ since it acts on the attention scores themselves. Rescaling these scores with the graph size is therefore the most targeted lever, and it changes no learned weights.

Entropy-based scaling factor (ESF). We adopt ESF (Eq. 6) [17] because it is established for this problem and needs no retraining. We apply the factor in the scaled dotproduct attention of both encoder and decoder of the trained reference models. For $n \ > \ n _ { \mathrm { t r a i n } }$ the factor exceeds one and sharpens the softmax, which counteracts the flattening caused by the larger number of keys (S2); for $n < { n } _ { \mathrm { t r a i n } }$ it softens the softmax accordingly.

Learnable scaling factor (LSF). ESF corrects dispersion (S2) with a fixed logarithmic rule, which need not be the best sharp ening, and it ties the model to one training size. A learnable factor, suggested as an extension of ESF [17], lets the policy learn this correction from data and allows training on several sizes. The factor in Eq. 6 is replaced by a learned function of the instance size, $\lambda _ { n } = \exp ( f _ { \phi } ( \log n ) )$ , where $f _ { \phi }$ is a small feed-forward network (hidden dimension 16) with the scalar input log n; the exponential keeps the factor positive. The unit is trained jointly with the policy under the same REINFORCE objective, with no additional loss term, and is applied at the same locations as ESF. Because single-size training shows the unit only one value of log n, LSF is additionally trained sequentially on two sizes (Section 7.3).

## 5.3. Encoder-side approaches

The encoder produces the node embeddings from which both the graph context and the pointer keys are computed, so a sizerobust encoder addresses dispersion (S2), and possibly context dilution (S3), at their source. In full self-attention, each node mixes information from all N nodes through a softmax whose sharpness depends on N. We therefore test two established encoders that remove this dependence in diferent ways: by replacing the softmax with a uniform mean over the neighbourhood, and by routing all interaction through a fixed-width bottleneck. Both replace the MHA encoder while the decoder, the node-embedding layers and the training procedure stay identi cal.

GraphSAGE encoder. GraphSAGE is designed to generalize to graphs unseen during training [18], which matches the sizetransfer setting. We replace the MHA encoder by three Graph-SAGE layers with the mean aggregator $( \mathrm { E q } . 3 )$ . Unlike the original method, no neighbourhood is sampled: N(v) contains all other nodes of the instance. The message to node v is then $( \sum _ { u } h _ { u } - h _ { \nu } ) / ( | V | - 1 )$ , so, in contrast to self-attention, all other nodes are weighted equally and the aggregate is normalised by the neighbourhood size, which keeps its scale independent of N. The encoder thus avoids the size-dependent softmax over all nodes (S2). Its message is, however, itself a mean over all nodes and therefore subject to the same dilution as the graph context (S3).

Set Transformer encoder. The Set Transformer [19] is designed for unordered sets and thus matches the permutation-invariant node set of a CVRP instance. We replace the MHA encoder by the Set Transformer encoder with three ISAB layers (Eq. 7), $m = 1 6$ inducing points shared across instances and 8 heads in both MABs, and keep the reference decoder. Each node then receives information through the same m summary vectors for any N, which limits how strongly its embedding, and the context and keys derived from it, depend on the graph size (S2, S3).

## 5.4. Proposed: learned graph-context aggregation

The graph context is the only summary of the whole instance that the decoder receives, and it enters the glimpse query at every decoding step, so its dilution (S3) afects every decision. We therefore propose to replace the uniform mean by a learned aggregation that can decide which nodes shape the summary. Two design requirements follow from the problem. First, the context is computed before any route exists, so the node set has no meaningful order and the aggregation must be permutationinvariant. Second, it must accept any number of nodes and return a fixed-size vector. The weighted average and Set2Set satisfy both requirements, whereas the GRU deliberately drops the first one to test whether permutation invariance matters. All three add few parameters, and leave the encoder, the step context and the pointer unchanged; in RL4CO the graph context is computed once per instance in the decoder module.

Weighted-average graph context. The simplest learned extension of the mean gives each node a learned weight, $\begin{array} { r } { \bar { x } = \sum _ { i } w _ { i } h _ { i } , } \end{array}$ with $w _ { i } = \mathrm { s o f t m a x } _ { i } ( s _ { i } )$ . The scores $s _ { i }$ come from a two-layer scoring network $( 1 2 8 \to 1 2 8 \to 1 )$ with a ReLU after the first layer. Because the softmax normalises over the nodes, the context stays a convex combination of embeddings for any N, but it is no longer uniform. To test whether the gain depends on what the scoring network can express, a reduced variant with a single linear scoring layer, $s _ { i } = W h _ { i } + b . $ , is evaluated as an ablation.

Set2Set graph context. A single weighting pass can capture only one view of the instance. We therefore use the Set2Set readout (Eq. 8) [43] over the node embeddings as the graph context of the attention-model decoder, with $T \ = \ 3$ processing steps. Its repeated attention lets the summary capture interactions between nodes beyond a single mean, and it is orderinvariant by construction.

GRU graph context. To test whether permutation invariance matters, we also propose an order-dependent aggregation: the node embeddings are passed in sequence through a GRU and its last hidden state is the graph context. It also gates how much each node contributes, but its result depends on the node order.

## 5.5. Proposed: approach selector

The levers above act on diferent sources of degradation, so there is no reason to expect a single one to be best for every transfer; Section 7 confirms that upscaling and downscaling respond to diferent levers. Rather than forcing one architecture, the selector keeps all trained models and chooses one per situation. This matches the operational setting: on a given day the number of clients n is known when the instance is formed, and the training size a of the available models is known. For each approach k in the set K of trained approaches and each training size a, a quadratic function of the problem size is fitted through its average costs measured on the CVRP20, CVRP50 and CVRP100 test sets,

$$
\hat { z } _ { k , a } ( n ) = { \alpha _ { k , a } n ^ { 2 } + \beta _ { k , a } n + \gamma _ { k , a } } ,\tag{11}
$$

and the selector solves the instance with the model of the approach with the lowest estimated cost,

$$
k ^ { * } ( a , n ) = \arg \operatorname* { m i n } _ { k \in \mathcal K } \hat { z } _ { k , a } ( n ) .\tag{12}
$$

On the three measured sizes, Eq. 11 reproduces the measured costs; between them it interpolates, and beyond CVRP100 it extrapolates. The selector needs no additional training and is evaluated on a fresh instance set (Section 6).

## 6. Experimental Setup

## 6.1. Framework, hardware and environment

All experiments use the RL4CO framework [16], which provides the CVRP environment, the attention-model policy, the REINFORCE-with-rollout training loop and evaluation utilities shared by the reference and all variants. All models were trained on one NVIDIA RTX 3070 Ti GPU with PyTorch 2.11 and PyTorch Geometric 2.7.

## 6.2. Dataset generation

CVRP instances are generated with the framework’s CVRP generator. Validation and test sets contain 10,000 instances each, generated once with fixed seeds (4321 and 1234), so every model is scored on identical instances. Training instances are resampled on the fly (100,000 per epoch) from the same distribution to avoid a static-dataset bias. Node coordinates are uniform in the unit square, the capacity is fixed at $\ Q \ = \ 3 0$ and integer demands are drawn from $\{ 1 , . . . , 9 \}$ for all sizes, so CVRP20, CVRP50 and CVRP100 difer only in the number of clients. Unlike the setting of Kool et al. [6], where capacity grows with the problem size, this isolates the efect of the number of nodes. Because the selector is built from the costs on the test set, it is evaluated on a separate instance set generated with seed 6249, for CVRP20, CVRP50 and CVRP100 and for the unseen sizes CVRP35, CVRP75 and CVRP200.

## 6.3. Models and training procedure

Every trained approach is trained independently and from scratch on each of the three sizes (the sequential training in Section 7.3 is the only exception). ESF needs no training and is applied at inference time to the three trained reference models. Training follows the REINFORCE-with-rollout procedure of Section 4: actions are sampled during training and decoded greedily during validation and testing, and the rollout baseline is replaced by the current policy when it outperforms the frozen baseline on a held-out set. All approaches share the same hyperparameters: embedding dimension 128, feed-forward width 512, 8 attention heads, 3 encoder layers, temperature 1.0, tanh clipping 10.0, Adam with learning rate $1 0 ^ { - 4 }$ , batch size 512 and 150 training epochs. Each change is confined to the component under test and no variant receives a larger training budget.

## 6.4. Evaluation methodology

For every (training size, test size) pair, the average solution cost (total route length, the negative of the reward) is computed over the full test set. To measure generalization independently of absolute solution quality, the cost of a model trained on size a and evaluated on size b is compared with the cost of the model of the same approach trained natively on size b:

$$
\mathrm { G a p } _ { a  b } ( \% ) = \frac { \mathcal { Z } _ { a  b } - \mathcal { Z } _ { b  b } } { \mathcal { Z } _ { b  b } } \times 1 0 0 ,\tag{13}
$$

where $z _ { a  b }$ is the average cost of the model trained on a and tested on $^ { b , }$ and $z _ { b  b }$ that of the model trained and tested on b. A smaller gap indicates better generalization; it is evaluated in both directions, upscaling $( a < b )$ and downscaling $( a > b )$

Each approach is then compared head-to-head with the reference by its gap gain,

$$
\Delta _ { a  b } = { \mathrm { G a p } } _ { a  b } ^ { \mathrm { r e f } } - { \mathrm { G a p } } _ { a  b } \quad ( \mathrm { p e r c e n t a g e ~ p o i n t s } ) ,\tag{14}
$$

so that a positive ∆ means the approach generalizes better than the reference on that transfer. Because the denominator in Eq. 13 is the approach’s own native model, the gap gain is read together with the same-size costs (Table 5), and absolute costs are reported where the distinction matters.

## 7. Results and Discussion

The reference is reported with costs and gaps (Eq. 13, Table 2), the approaches by their gap gain (Eq. 14, Table 3) in percentage points (pp) on the three upscaling and three downscaling transfers. After the reference and its diagnosis, the approaches follow the order of Section 5.

## 7.1. Generalization loss of the reference policy

Table 2 reports the reference attention-model policy of Kool et al. [6] as implemented in RL4CO [16], an established, already-tuned construction policy, so every improvement below is a gain over a strong reference. Its cross-size behaviour reproduces the bidirectional bottleneck. First, the gap grows with the size mismatch: a one-step transfer costs 0 55–2 61%, whereas the two-step transfers cost 5 23% (CVRP20→CVRP100) and 7 77% (CVRP100→CVRP20). Second, the gap is asymmetric: for every size pair, downscaling is worse than upscaling (e.g. CVRP100→CVRP20 7 77% vs. CVRP20→CVRP100 5 23%). A policy trained on sparse graphs therefore transfers to dense graphs more readily than the reverse.

Table 2: Reference attention-model policy: average solution cost and, in brackets, cross-size optimality gap (%). Rows are training size, columns are test size.
<table><tr><td>Train</td><td>CVRP20</td><td>CVRP50</td><td>CVRP100</td></tr><tr><td>CVRP20</td><td>6.480 (0.00)</td><td>13.692 (2.18)</td><td>25.408 (5.23)</td></tr><tr><td>CVRP50</td><td>6.649 (2.61)</td><td>13.400 (0.00)</td><td>24.277 (0.55)</td></tr><tr><td>CVRP100</td><td>6.984 (7.77)</td><td>13.611 (1.58)</td><td>24.145 (0.00)</td></tr></table>

## 7.2. Sources of cross-size degradation

The measurements below confirm the three sources of Section 5.1 on the reference policy.

S1: the distribution shift is a shift in node density. On the test sets, the mean normalised demand per client (0 167), the mean depot-to-client distance (0 52) and the mean pairwise distance (0 52) are practically constant across sizes, while the mean nearest-neighbour distance shrinks from 0 124 (CVRP20) through 0 075 (CVRP50) to 0 052 (CVRP100). The mean total demand per instance, relative to the capacity and thus a lower bound on the average number of routes, grows from 3 34 through 8 33 to 16 68. The generalization gap studied here is therefore a response to node density and to the number of required tours, not to a change of operating region or demand regime.

![](images/68247d436a4073744569f7be2aa2764b8f0a8fbdca3ac492f928399a01336753.jpg)  
Figure 2: Entropy of the decoder action distribution of the reference policy at each decoding step, averaged over the test instances. Panels from left to right: test on CVRP20, CVRP50 and CVRP100 (note the diferent vertical scales); lines: models trained on CVRP20, CVRP50 and CVRP100.

S2: attention dispersion. Attention dispersion is measured by the entropy of the decoder’s action distribution, $H ( p _ { t } )$ , which is high when the probability mass is spread over many candidate nodes and low when the policy points clearly to one node. Fig. 2 shows this entropy at every decoding step for the three reference models on each test size. Two patterns are visible. First, within each panel the entropy grows with the mismatch between training and test size: on CVRP100 the CVRP20-trained model stays roughly twice as uncertain as the CVRP100-trained model over the whole route, and on CVRP20 the order is re versed, with the CVRP100-trained model highest. Second, every curve peaks at the first decoding step, when all customers are still candidates, and falls to zero at the end of the route, when few feasible nodes remain. A policy applied to an unfamiliar size therefore separates good from poor candidates less sharply, in line with Wang et al. [36]. On both extremes the higher entropy comes with the higher cost (CVRP20: 6 984 for the CVRP100-trained vs. 6 480 for the CVRP20-trained model; CVRP100: 25 408 vs. 24 145). The relation is not strictly monotone (on CVRP50, the CVRP100-trained model has slightly lower entropy than the CVRP50-trained model but produces longer routes), which indicates that dispersion is one source among several.

![](images/5e6db770ba9438e18fcaf98c31b013397b8885cfc6608c4c067f649a89979065.jpg)  
Figure 3: Mean-pooled graph context of the CVRP100-trained reference policy for the individual test instances, projected on its first two principal components. Panels from left to right: test on CVRP20, CVRP50 and CVRP100; darker cells contain more instances.

S3: graph-context dilution. Fig. 3 visualises the graph context that the CVRP100-trained model computes for each test instance. On CVRP20 the contexts of diferent instances spread over a wide region, on CVRP50 they shrink to a small cluster, and on CVRP100 they collapse almost to a single point: averaging over more nodes removes the diferences between instances, so the decoder receives nearly the same summary for every large instance. Quantitatively, the variance of the context across instances falls for all three reference models from about $1 . 4 \times 1 0 ^ { - 3 }$ on CVRP20 to $3 { \ - } 4 \times 1 0 ^ { - 4 }$ on CVRP100, and the learned projection compresses it further (to $8 { - } 9 \times 1 0 ^ { - 5 }$ on CVRP100). The figure also links dilution to the downscaling asymmetry of Table 2: a model trained on CVRP100 has only seen this narrow cluster of contexts, whereas on CVRP20 it receives contexts far outside that range. Because the decoder reuses the context at every step, this mismatch afects every decision, which motivates a richer, learned aggregation.

## 7.3. Architecture-wide approaches: size-aware attention

Entropy-based scalingfactor (ESF). Table 3 lists the gap gains of all approaches. Because $\begin{array} { l l l } { \lambda _ { n } } & { = } & { 1 } \end{array}$ on the training size, ESF leaves same-size behaviour unchanged, so its gains are purely cross-size. It is positive on five of the six transfers and gives a strong improvement on the hardest upscaling transfer: CVRP20→CVRP100 gains +1 41 pp (from 5 23% to 3 83%) and CVRP20→CVRP50 gains +0 48 pp. It also gives the best gain on CVRP50→CVRP20 (+0 37 pp) and +0 24 pp on CVRP100→CVRP20, with only CVRP100→CVRP50 marginally negative (−0 04 pp). ESF was originally evaluated with capacity growing with the problem size; these results show that it remains efective when only the number of nodes changes, without any retraining.

Table 3: Gap gain ∆ of every approach with respect to the reference (pp, Eq. 14). ↑: smaller gap than the reference (improvement), ↓: larger gap (degradation). A: architecture-wide, E: encoder-side, D: decoder-side; <sup>†</sup> proposed. Best gain per transfer in bold. The gains of GraphSAGE are read together with its weaker same-size models (Table 5).
<table><tr><td rowspan="2"></td><td rowspan="2">Approach</td><td colspan="3">Upscaling</td><td colspan="3">Downscaling</td></tr><tr><td>20→50</td><td>20→100</td><td> $5 0 \to 1 0 0$ </td><td>50→20</td><td>100→20</td><td>100→50</td></tr><tr><td>A</td><td>ESF</td><td>↑0.48</td><td>↑1.41</td><td>↑0.03</td><td>↑0.37</td><td>↑0.24</td><td>↓0.04</td></tr><tr><td>A</td><td>LSF</td><td>↑0.23</td><td>↑0.34</td><td>↑0.06</td><td>↓0.16</td><td>0.00</td><td>↑0.43</td></tr><tr><td>E</td><td>GraphSAGE</td><td>↑1.43</td><td>↑3.07</td><td>↑1.84</td><td>↓2.07</td><td>↓5.42</td><td>↓1.78</td></tr><tr><td>E</td><td>Set Transformer</td><td>↑0.67</td><td>↑0.49</td><td>↓0.15</td><td>↓0.31</td><td>↓1.08</td><td>↑0.54</td></tr><tr><td>D</td><td>Weighted avg.†</td><td>↑0.16</td><td>↓0.41</td><td>↓0.09</td><td>↓0.09</td><td>↑0.91</td><td>↑0.33</td></tr><tr><td>D</td><td>Set2Set†</td><td>↓0.36</td><td>↓0.60</td><td>↑0.03</td><td>↓0.06</td><td>↑0.92</td><td>↑0.25</td></tr><tr><td>D</td><td>GRU†</td><td>↓0.28</td><td>↓0.93</td><td>↓0.19</td><td>↓1.16</td><td>↓0.08</td><td>↑0.27</td></tr></table>

Learnable scaling factor (LSF). LSF is positive on four transfers and gives +0 43 pp on CVRP100→CVRP50. In addition, it lowers the same-size cost on all three sizes (e.g. 24 145 → 24 116 on CVRP100) and the cost in eight of the nine train– test cells, so its gap gains, measured against its own better native models, understate its absolute improvement. Its cross-size gains are smaller than those of ESF (e.g. +0 34 vs. +1 41 pp on CVRP20→CVRP100) because, trained on a single size, the network observes only one value of log n, whereas ESF encodes the size dependence analytically. The sequential training below shows that a second training size releases this potential.

LSF with sequential two-size training. Because single-size training exposes LSF to a single value of log n, we also trained it sequentially (75 epochs per size): an upscaling model on CVRP20 then CVRP50, tested on the unseen CVRP100, and a downscaling model on CVRP100 then CVRP50, tested on the unseen CVRP20. The reference trained in the same way has gaps of 3<sub>.</sub>73% (upscaling) and 5<sub>.</sub>65% (downscaling). LSF reaches 2<sub>.</sub>18% (+1<sub>.</sub>55 pp), the largest upscaling gain of the sequential setting, and 6 17% (−0 52 pp) in downscaling. A second training size thus gives LSF a large upscaling gain, which makes it a promising candidate for true mixed-size training.

## 7.4. Encoder-side approaches

GraphSAGE encoder. GraphSAGE gives the largest gap gains in Table 3 on all three upscaling transfers (+1 43, +3 07 and +1 84 pp). The CVRP20→CVRP100 gap falls from 5 23% to 2 17%, and on CVRP50→CVRP100 the gap becomes negative (−1 29%): the CVRP50-trained model beats the CVRP100-trained one on CVRP100. This indicates that the normalised, content-independent message lets embeddings learned on smaller graphs carry over to larger ones. Its same size costs are 2 4 to 4 2% above the reference (Table 5), so these gains reflect a flatter cost profile across sizes rather than cheaper transfers in absolute terms (e.g. 25 705 vs. 25 408 on CVRP20→CVRP100). In the downscaling direction, all three transfers get worse (−2 07, −5 42 and −1 78 pp), since relationships learned by aggregating over many nodes need more nodes than a small graph provides.

Set Transformer encoder. The Set Transformer improves both transfers from CVRP20 (+0 67 and +0 49 pp) and gives the best gain on CVRP100→CVRP50 (+0 54 pp), while keeping its same-size costs within 0 2% of the reference (Table 5). Its upscaling gains therefore translate into cheaper solutions: CVRP20→CVRP50 falls from 13 692 to 13 627 and CVRP20→CVRP100 from 25 408 to 25 258. This supports the fixed-width bottleneck as an upscaling lever that preserves same-size quality. The remaining transfers are negative, most strongly CVRP100→CVRP20 (−1 08 pp), so, as for Graph-SAGE, the benefit is directional.

## 7.5. Proposed: learned graph-context aggregation

Weighted-average graph context. In Table 3, the weighted average gives a large gain on the hardest downscaling transfer (CVRP100→CVRP20, +0 91 pp) together with +0 33 pp on CVRP100→CVRP50 and +0 16 pp on CVRP20→CVRP50. Unlike GraphSAGE, it achieves this while keeping same-size quality (e.g. 24 069 vs. 24 145 on CVRP100), so the gain is a real cost reduction: on CVRP100→CVRP20 the average cost falls from 6 984 to about 6 91. It has a lower cost than the reference in seven of the nine train–test cells, and its losses on the remaining transfers are moderate (−0 09 to −0 41 pp). The ablation with a single linear scoring layer loses the downscaling benefit (CVRP100→CVRP20: −1 30 pp), which indicates that the improvement comes from the expressiveness of the scoring network and not merely from abandoning the uniform mean.

Set2Set graph context. Set2Set produces the largest single downscaling gain in the study, +0 92 pp on CVRP100→CVRP20 (from 7 77% to 6 86%, an average cost of about 6 93), together with +0 25 pp on CVRP100→CVRP50, and it has the lowest same-size cost on CVRP50 (13 374) and the second-lowest on CVRP100 (24 050). Its gains are concentrated on transfers from the dense regime, where the dilution of the mean was measured (Section 7.2): a richer summary learned on dense graphs remains informative on sparse ones. Transfers from CVRP20 do not benefit (−0 36 and −0 60 pp), which is consistent with the measurements: on sparse graphs the mean is less diluted, so a learned summary has less to correct.

GRU graph context. The order-dependent GRU improves only CVRP100→CVRP50 (+0 27 pp) and worsens the other five transfers (−0 08 to −1 16 pp), although it has the lowest samesize cost on CVRP100 (24 030). A good same-size summary is therefore not suficient: the GRU can learn order artefacts that do not transfer across sizes, which supports permutation invariance as a design requirement of the proposed aggregations.

## 7.6. Proposed: approach selector

Table 4 shows that the selector turns the complementarity of the levers into a consistent improvement. On the fresh instance set it lowers the cost of the reference in all nine seen train– test pairs, by up to 1 07% on CVRP100→CVRP20, the hardest downscaling transfer. It also lowers the cross-size gap in four of the six transfers, by +0 14 to +0 95 pp (CVRP50→CVRP20 and CVRP50→CVRP100 change by −0 14 and −0 03 pp, since a lower native cost raises the gap of an otherwise cheaper transfer). On the unseen sizes it is cheaper in eight of nine pairs, and the largest gain occurs for the largest extrapolation: the CVRP20-trained models reach 48 43 instead of 50 04 on CVRP200 (3 2%). The only exception, CVRP50→CVRP35, difers by 0 03%.

Table 4: Approach selector on the fresh instance set (seed 6249): cost reduction relative to the reference (%), $( z ^ { \mathrm { r e f } } - z ^ { \mathrm { s e l } } ) / z ^ { \mathrm { r e f } } ;$ ×100; ↑: cheaper than the reference, ↓: more expensive. CVRP35, CVRP75 and CVRP200 are unseen sizes.
<table><tr><td rowspan="2">Train</td><td colspan="3">Seen sizes</td><td colspan="3">Unseen sizes</td></tr><tr><td>20</td><td>50</td><td>100</td><td>35</td><td>75</td><td>200</td></tr><tr><td>CVRP20</td><td>↑0.20</td><td>↑0.47</td><td>↑0.56</td><td>↑0.26</td><td>↑0.49</td><td>↑3.20</td></tr><tr><td>CVRP50</td><td>↑0.08</td><td>↑0.25</td><td>↑0.40</td><td>↓0.03</td><td>↑0.35</td><td>↑0.46</td></tr><tr><td>CVRP100</td><td>↑1.07</td><td>↑0.41</td><td>↑0.43</td><td>↑0.49</td><td>↑0.41</td><td>↑0.35</td></tr></table>

## 7.7. Synthesis: diferent leversfor upscaling and downscaling

The comparison shows that the two transfer directions respond to diferent levers (Table 3). Upscaling responds to the encoder approaches and to attention rescaling. Among the approaches that keep same-size quality, ESF is the strongest on the hardest upscaling transfer: it reaches +1 41 pp on CVRP20→CVRP100 without retraining and gives the lowest cost of all approaches there (25 068 vs. 25 408 for the reference), while the Set Transformer is strongest on CVRP20→CVRP50 (+0 67 pp). Downscaling, in particular the hardest CVRP100→CVRP20 transfer, responds most to the proposed graph-context aggregation: Set2Set and the weighted average each recover about 0 9 pp, whereas the encoder approaches make it worse. The efective downscaling levers share one mechanism, a more expressive summary of a dense graph that remains informative on a sparse one. This matches the asymmetry of the reference: the encoder and attention levers help in the direction in which density increases, and the context levers in the direction in which the training-size context was diluted.

Table 5: Same-size average cost (train = test) of the reference and of every approach; lower is better. ESF equals the reference by construction.
<table><tr><td>Approach</td><td>CVRP20</td><td>CVRP50</td><td>CVRP100</td></tr><tr><td>Reference / ESF</td><td>6.480</td><td>13.400</td><td>24.145</td></tr><tr><td>LSF</td><td>6.475</td><td>13.393</td><td>24.116</td></tr><tr><td>GraphSAGE</td><td>6.634</td><td>13.793</td><td>25.160</td></tr><tr><td>Set Transformer</td><td>6.474</td><td>13.423</td><td>24.115</td></tr><tr><td>Weighted avg.†</td><td>6.466</td><td>13.408</td><td>24.069</td></tr><tr><td>Set2Set†</td><td>6.487</td><td>13.374</td><td>24.050</td></tr><tr><td>GRU†</td><td>6.476</td><td>13.406</td><td>24.030</td></tr></table>

Most approaches keep the same-size quality (Table 5): they stay within 0 5% of the reference cost on every size, except GraphSAGE (+2 4, +2 9 and +4 2%). Same-size quality and cross-size gap are therefore separate objectives, and the selector chooses on estimated cost.

## 8. Conclusion and Future Work

This paper started from a practical requirement for dronedelivery route planning: a routing policy must handle a variable, a priori unknown number of clients. Using CVRP with fixed vehicle capacity as a controlled proxy, in which only the number of clients changes, we identified three sources of the cross-size degradation of the attention-based Pointer Network (node density, attention dispersion and graph-context dilution) and compared levers that act on the two policy-internal sources at three levels of its encoder–decoder architecture: architecturewide, encoder-side and decoder-side.

The reference gap grows with the size mismatch and is asymmetric: downscaling (CVRP100→CVRP20, 7 77%) is harder than upscaling (CVRP20→CVRP100, 5 23%). The comparison shows that the two directions respond to diferent levers. For upscaling, ESF lowers the CVRP20→CVRP100 gap by 1 41 percentage points without retraining and gives the lowest cost there, and the Set Transformer encoder improves the transfers from CVRP20 with same-size quality preserved. For downscaling, the proposed learned graph-context aggregations give the largest gains on the hardest transfer: Set2Set and the weighted average lower CVRP100→CVRP20 by 0 92 and 0 91 percentage points while keeping same-size quality, and the order-dependent GRU context supports permutation invariance as a requirement. With a second training size, the learnable scaling factor gains 1 55 points in upscaling. The proposed approach selector exploits this complementarity with a datadriven decision on estimated cost rather than a hand-crafted rule: it lowers the cost of the reference in all nine seen train– test pairs and in eight of nine pairs on the unseen sizes CVRP35, CVRP75 and CVRP200, by up to 3 2%.

Three future directions follow from these findings. First, the strongest upscaling and downscaling levers act on diferent components, so attention rescaling and the proposed graphcontext aggregation can be combined in one model and trained on several sizes at once, building on the sequential results of the learnable scaling factor. Second, the selector currently decides on problem size only; an ensemble that keeps the cheapest of several solutions, or a small learned selection network that also uses instance features such as demand relative to capacity or client density, can choose per instance. Third, the CVRP abstraction will be extended toward full drone routing with battery-aware costs, air-corridor constraints and multidrone coordination, evaluated on larger instances and realistic customer layouts.

## Funding

The research project was funded by The Ministry of the Environment, Nature Conservation, and Transport of the State of North Rhine-Westphalia in Germany, and co-financed by the European Union for the research project “AI4DRONE”, with grant number IN-NX-1-033b.

## References

[1] A. Otto, N. Agatz, J. Campbell, B. Golden, E. Pesch, Optimization approaches for civil applications of unmanned aerial vehicles (UAVs) or aerial drones: A survey, Networks 72 (4) (2018) 411–458. doi:10.1002/ net.21818.

[2] G. Macrina, L. Di Puglia Pugliese, F. Guerriero, G. Laporte, Drone-aided routing: A literature review, Transportation Research Part C: Emerging Technologies 120 (2020) 102762. doi:10.1016/j.trc.2020.102762.

[3] G. B. Dantzig, J. H. Ramser, The truck dispatching problem, Management Science 6 (1) (1959) 80–91. doi: 10.1287/mnsc.6.1.80.

[4] P. Toth, D. Vigo (Eds.), Vehicle Routing: Problems, Methods, and Applications, 2nd Edition, Society for Industrial and Applied Mathematics, Philadelphia, PA, 2014. doi:10.1137/1.9781611973594.

[5] M. Nazari, A. Oroojlooy, L. V. Snyder, M. Takác, Rein-ˇ forcement learning for solving the vehicle routing problem, in: Advances in Neural Information Processing Systems, Vol. 31, 2018, pp. 9839–9849.

[6] W. Kool, H. van Hoof, M. Welling, Attention, learn to solve routing problems!, International Conference on Learning Representations (ICLR) (2019). URL https://arxiv.org/abs/1803.08475

[7] R. Fukasawa, H. Longo, J. Lysgaard, M. Poggi de Aragão, M. Reis, E. Uchoa, R. F. Werneck, Robust branch-andcut-and-price for the capacitated vehicle routing problem, Mathematical Programming 106 (3) (2006) 491–511. doi:10.1007/s10107-005-0644-x.

[8] A. Pessoa, R. Sadykov, E. Uchoa, F. Vanderbeck, A generic exact solver for vehicle routing and related problems, Mathematical Programming 183 (2020) 483–523. doi:10.1007/s10107-020-01523-z.

[9] G. Laporte, F. Semet, Classical heuristics for the capacitated vrp, in: P. Toth, D. Vigo (Eds.), The Vehicle Routing Problem, Society for Industrial and Applied Mathematics, Philadelphia, PA, 2002, pp. 109–128. doi: 10.1137/1.9780898718515.ch5.

[10] T. Vidal, T. G. Crainic, M. Gendreau, C. Prins, A hybrid genetic algorithm with adaptive diversity management for a large class of vehicle routing problems with time-windows, Computers & Operations Research 40 (1) (2013) 475–489. doi:10.1016/j.cor.2012.07.018.

[11] I. Bello, H. Pham, Q. V. Le, M. Norouzi, S. Bengio, Neural combinatorial optimization with reinforcement learning, arXiv preprint arXiv:1611.09940, under review as a conference paper at ICLR 2017 (2016). doi:10.48550/ arXiv.1611.09940.

URL https://arxiv.org/abs/1611.09940

[12] Y. Bengio, A. Lodi, A. Prouvost, Machine learning for combinatorial optimization: A methodological tour d’horizon, European Journal of Operational Research 290 (2) (2021) 405–421. doi:10.1016/j.ejor.2020. 07.063.

[13] S. Manchanda, S. Michel, D. Drakulic, J.-M. Andreoli, On the generalization of neural combinatorial optimization heuristics, arXiv preprint arXiv:2206.00787 (2022). doi:10.48550/arXiv.2206.00787. URL https://arxiv.org/abs/2206.00787

[14] O. Vinyals, M. Fortunato, N. Jaitly, Pointer networks, in: Advances in Neural Information Processing Systems, Vol. 28, 2015, pp. 2692–2700.

[15] S. Liu, Y. Zhang, K. Tang, X. Yao, How good is neural combinatorial optimization? a systematic evaluation on the traveling salesman problem, IEEE Computational Intelligence Magazine 18 (3) (2023) 14–28. doi:10.1109/ MCI.2023.3277768.

[16] F. Berto, C. Hua, J. Park, L. Luttmann, Y. Ma, F. Bu, J. Wang, H. Ye, M. Kim, S. Choi, N. G. Zepeda, A. Hottung, J. Zhou, J. Bi, Y. Hu, F. Liu, H. Kim, J. Son, H. Kim, D. Angioni, W. Kool, Z. Cao, Q. Zhang, J. Kim, J. Zhang, K. Shin, C. Wu, S. Ahn, G. Song, C. Kwon, K. Tierney, L. Xie, J. Park, RL4CO: An extensive reinforcement learning for combinatorial optimization benchmark, arXiv preprint arXiv:2306.17100 (2023). arXiv:2306.17100, doi:10.48550/arXiv.2306.17100.

[17] Y. Xiao, D. Wang, X. Wu, Y. Wu, B. Li, W. Du, L. Wang, Y. Zhou, Improving generalization of neural vehicle routing problem solvers through the lens of model architecture, Neural Networks 187 (2025) 107380. doi:10. 1016/j.neunet.2025.107380.

[18] W. L. Hamilton, R. Ying, J. Leskovec, Inductive representation learning on large graphs, in: Advances in Neural Information Processing Systems, Vol. 30, 2017, pp. 1024– 1034.

[19] J. Lee, Y. Lee, J. Kim, A. Kosiorek, S. Choi, Y. W. Teh, Set transformer: A framework for attention-based permutation-invariant neural networks, in: Proceedings of the 36th International Conference on Machine Learning, Vol. 97 of Proceedings of Machine Learning Research, 2019, pp. 3744–3753.

[20] B. Korte, J. Vygen, Combinatorial Optimization: Theory and Algorithms, 6th Edition, Springer, Berlin, Germany, 2018. doi:10.1007/978-3-662-56039-6.

[21] G. Laporte, Fifty years of vehicle routing, Transportation Science 43 (4) (2009) 408–416. doi:10.1287/trsc. 1090.0301.

[22] G. Clarke, J. W. Wright, Scheduling of vehicles from a central depot to a number of delivery points, Operations

Research 12 (4) (1964) 568–581. doi:10.1287/opre. 12.4.568.

[23] P. Augerat, J. M. Belenguer, E. Benavent, A. Corberán, D. Naddef, G. Rinaldi, Computational results with a branch and cut code for the capacitated vehicle routing problem, Tech. Rep. RR 949-M, Université Joseph Fourier, Grenoble, France (1995).

[24] S. Lin, Computer solutions of the traveling salesman problem, Bell System Technical Journal 44 (10) (1965) 2245– 2269. doi:10.1002/j.1538-7305.1965.tb04146.x.

[25] M. Gendreau, A. Hertz, G. Laporte, A tabu search heuristic for the vehicle routing problem, Management Science 40 (10) (1994) 1276–1290. doi:10.1287/mnsc.40.10. 1276.

[26] E. B. Khalil, P. Le Bodic, L. Song, G. Nemhauser, B. Dilkina, Learning to branch in mixed integer programming, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 30, 2016, pp. 724–731. doi:10.1609/aaai. v30i1.10080.

[27] M. Gasse, D. Chételat, N. Ferroni, L. Charlin, A. Lodi, Exact combinatorial optimization with graph convolutional neural networks, in: Advances in Neural Information Processing Systems, Vol. 32, 2019, pp. 15580–15592.

[28] Y. Tang, S. Agrawal, Y. Faenza, Reinforcement learning for integer programming: Learning to cut, in: Proceedings of the 37th International Conference on Machine Learning, Vol. 119 of Proceedings of Machine Learning Research, 2020, pp. 9367–9376.

[29] Z. Huang, K. Wang, F. Liu, H.-L. Zhen, W. Zhang, M. Yuan, J. Hao, Y. Yu, J. Wang, Learning to select cuts for eficient mixed-integer programming, Pattern Recognition 123 (2022) 108353. doi:10.1016/j.patcog. 2021.108353.

[30] F. Hutter, H. H. Hoos, K. Leyton-Brown, Sequential model-based optimization for general algorithm configuration, in: Learning and Intelligent Optimiza tion, Springer, 2011, pp. 507–523. doi:10.1007/ 978-3-642-25566-3\_40.

[31] P. Kerschke, H. H. Hoos, F. Neumann, H. Trautmann, Automated algorithm selection: Survey and perspectives, Evolutionary Computation 27 (1) (2019) 3–45. doi: 10.1162/evco\_a\_00242.

[32] V. Nair, S. Bartunov, F. Gimeno, I. von Glehn, P. Lichocki, I. Lobov, B. O’Donoghue, N. Sonnerat, C. Tjandraatmadja, P. Wang, R. Addanki, T. Hapuarachchi, T. Keck, J. Keeling, P. Kohli, I. Ktena, Y. Li, O. Vinyals, Y. Zwols, Solving mixed integer programs using neural networks, arXiv preprint arXiv:2012.13349 (2020).

[33] N. Mazyavkina, S. Sviridov, S. Ivanov, E. Burnaev, Reinforcement learning for combinatorial optimization: A survey, Computers & Operations Research 134 (2021) 105400. doi:10.1016/j.cor.2021.105400.

[34] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, I. Polosukhin, Attention is all you need, in: Advances in Neural Information Processing Systems, Vol. 30, 2017, pp. 5998–6008.

[35] Y. Chen, R. Chen, F. Luo, Z. Wang, Improving generalization of neural combinatorial optimization for vehicle routing problems via test-time projection learning, arXiv preprint arXiv:2506.02392 (2025). doi:10. 48550/arXiv.2506.02392. URL https://arxiv.org/abs/2506.02392

[36] Y. Wang, Y.-H. Jia, W.-N. Chen, Y. Mei, Distance-aware attention reshaping for enhancing generalization of neural solvers, IEEE Transactions on Neural Networks and Learning Systems 36 (10) (2025) 18900–18914. doi: 10.1109/TNNLS.2025.3588209.

[37] F. Luo, X. Lin, F. Liu, Q. Zhang, Z. Wang, Neural combinatorial optimization with heavy decoder: Toward large scale generalization, arXiv preprint arXiv:2310.07985 (2023). arXiv:2310.07985, doi:10.48550/arXiv. 2310.07985.

[38] F. Scarselli, M. Gori, A. C. Tsoi, M. Hagenbuchner, G. Monfardini, The graph neural network model, IEEE Transactions on Neural Networks 20 (1) (2009) 61–80. doi:10.1109/TNN.2008.2005605.

[39] J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, G. E. Dahl, Neural message passing for quantum chemistry, in: Proceedings of the 34th International Conference on Machine Learning, Vol. 70 of Proceedings of Machine Learning Research, 2017, pp. 1263–1272.

[40] T. N. Kipf, M. Welling, Semi-supervised classification with graph convolutional networks, International Conference on Learning Representations (ICLR) (2017). URL https://arxiv.org/abs/1609.02907

[41] Q. Li, Z. Han, X.-M. Wu, Deeper insights into graph convolutional networks for semi-supervised learning, in: Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence, AAAI Press, 2018, pp. 3538–3545.

[42] D. Bahdanau, K. Cho, Y. Bengio, Neural Machine Translation by Jointly Learning to Align and Translate (2014). URL https://arxiv.org/pdf/1409.0473

[43] O. Vinyals, S. Bengio, M. Kudlur, Order matters: Sequence to sequence for sets, International Conference on Learning Representations (ICLR) (2016). URL https://arxiv.org/abs/1511.06391

[44] R. S. Sutton, D. A. McAllester, S. P. Singh, Y. Mansour, Policy gradient methods for reinforcement learning with function approximation, in: S. A. Solla, T. K. Leen, K.-R. Müller (Eds.), Advances in Neural Information Processing Systems 12 (NeurIPS 1999), MIT Press, 1999, pp. 1057–1063.

[45] K. He, X. Zhang, S. Ren, J. Sun, Deep residual learning for image recognition, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770–778. doi:10.1109/CVPR.2016.90. URL https://ieeexplore.ieee.org/document/ 7780459