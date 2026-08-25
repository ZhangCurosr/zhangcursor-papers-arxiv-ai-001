![](images/cf9488ab86065a3abb8e38a03745b33a39f1d8d553e709f3186486abe3718022.jpg)  
Mycelial Search

## Graphical Abstract

Mycelial Search: A Graph-Structured Metaheuristic for Continuous Optimisation Mohammad Mahdi Dehshibi

A Graph-Structured Metaheuristic for Continuous Optimisation

O(N log N + ND)

![](images/f6321822ed82fff9216a539e6af082f325f7521fb38a0bf49edb8800ce28788f.jpg)  
Photo by: Prof. Andrew Adamatzky Copyright © Unconventional Computing Lab, UWE Bristol, U.K  
1. Population as an Evolving Graph

![](images/648278e0d633ebe9219f8d111cec1c2e4407131cbb5178463e518493246a5b28.jpg)  
2. Community Detection (Louvain)  
Tips explore; anchors preserve historically good locations.

![](images/d9e20829c2f327de4a13f1c4261d12831dd11b60cb76f0ba03baef6742e2a67c.jpg)

Communities control the range of information exchange.

4. Anchor-Based Injection (Ridge-Oriented) Injected candidate

![](images/d536452204cacc190af8914ea5c7a71606ee7f39f4ae80f81d0afefbfe639a60.jpg)

![](images/8e2a1f6ee648459284f11f7b71255eb82ddce62e9f0a7c4030891332ccc440eb.jpg)  
Anchor-pair geometry & local gradients → diverse candidates.  
3. Adaptive Cord Plasticity (Conductance Update)

![](images/d08f48f8e37f0fd8f979391c06bc4a9bdc56f2a0a370ebdc9ed7710c97a28dd7.jpg)

![](images/cae470350b122a70e111d071f4f7ffde6e69150a2c27daf72e6d591504522f3b.jpg)  
— Reinforced (aligned with flow) — Unchanged Attenuated (against flow)

## Highlights

## Mycelial Search: A Graph-Structured Metaheuristic for Continuous Optimisation Mohammad Mahdi Dehshibi

• Mycelial Search introduces an evolving graph structure for continuous optimisation.

• Community detection regulates information exchange across local search regions.

• Adaptive cord plasticity reinforces flow-aligned links and attenuates others.

• Per-iteration complexity scales as �(� log � + ��) under sparse graph connectivity.

• Myco achieves competitive results on CEC 2022 functions across two dimensions.

# Mycelial Search: A Graph-Structured Metaheuristic for Continuous Optimisation

Mohammad Mahdi Dehshibi

Unconventional Computing Laboratory UWE Bristol UK

A R T I C L E I N F O

Keywords: Continuous optimisation Metaheuristic optimisation Graph-structured search Community detection Adaptive conductance Mycelia-inspired computing

## A BS T R AC T

Continuous optimisation methods need to balance sharing information and maintaining alternative search directions. In this paper, we introduce Mycelial Search (Myco), a graphstructured metaheuristic designed around active tips, community-weighted flow, adaptive cord plasticity, and anchor-based injection. Candidate solutions form an evolving spatial graph in which a Louvain partition distinguishes within-community from cross-community information exchange. Adaptive cord plasticity subsequently modifies active tip-to-tip edges according to their alignment with the local flow. An anchor-based injection mechanism supplements the graph-driven tip dynamics. We evaluated Myco on the CEC 2022 single-objective boundconstrained benchmark suite at dimensions � = 10 and � = 20, using 30 independent runs per algorithm-function pair. The comparison includes eleven established optimisers from several search families. Myco reaches competitive results on selected functions across both dimensions. The ablation analysis further shows that community structure regulates the range of graphbased information exchange, whereas cord plasticity controls the persistence of local directional influence. These findings indicate that graph-structured local interaction can support continuous optimisation, while its effectiveness depends on landscape structure and information transfer across local search regions.

## 1. Introduction

Continuous optimisation problems with interacting variables, multiple local optima, and regions with sharply different search behaviour appear in constrained engineering design and optimisation settings [10, 19, 24, 25, 30]. Population-based metaheuristics address this setting by maintaining a set of candidate solutions and using their relative quality to guide the search. Genetic algorithms use selection and variation, particle swarm optimisation uses learned exemplar positions, and differential evolution constructs trial solutions from population differences [13, 26, 31]. Later methods refined these principles through adaptive control parameters, external archives, and changing population sizes [5, 28, 33].

Many population methods still represent the relations among candidate solutions only weakly. Interaction is often mediated by a global best solution, sampled exemplars, or temporary population differences. These mechanisms can be effective, but they rarely retain a spatial interaction structure that changes with the search state. This matters when useful information is local. A candidate may benefit from nearby high-quality positions without requiring direct attraction to the same population-wide reference.

Graph-oriented optimisation offers a different perspective. Adaptive connectivity, flow, and edge reinforcement have been used to represent how information or resources move through a network [3, 11]. Community detection identifies groups that are more densely connected internally than externally [4]. These approaches commonly address problems in which a graph, route, or transport network is itself the object of optimisation. Candidate solutions in continuous search can be organised the same way as nodes in an evolving graph, where local connectivity determines which search information is exchanged and how strongly it contributes to subsequent motion.

Fungal mycelia offer a suitable biological motivation for this perspective. A mycelial network coordinates distributed exploration with transport across the colony. Established pathways can remain available while local connectivity adapts to usage patterns [1, 6, 7, 8, 9]. We use this principle without attempting to reproduce fungal physiology. We introduce Mycelial Search (Myco), a graph-structured metaheuristic for continuous optimisation. Myco represents candidate solutions as active tips and retains a bounded set of anchors at historically favourable positions within a spatial graph that is reconstructed at every iteration. Tips receive locally weighted flow information from neighbouring nodes. Anchors preserve high-quality locations that remain accessible to the evolving graph.

Myco combines two forms of graph organisation. A Louvain partition imposes a mesoscale organisation on the current graph and distinguishes within-community from cross-community interactions [4]. A conductance update then modifies the current tip-to-tip edges based on their directional agreement with the induced flow, reinforcing flow-aligned edges and attenuating unsupported ones. The community partition regulates the scope of information exchange across the graph. The conductance update retains short-term directional information within the local graph.

We evaluate Myco on the CEC 2022 [14] single-objective bound-constrained benchmark suite across two problem dimensions and compare it with established optimisation methods from several search families. We also examine the separate contributions of adaptive conductance and community partitioning. This analysis tests whether an evolving local interaction graph and adaptive edge strength provide useful search information in continuous optimisation.

The rest of this paper is organised as follows: Section 2 reviews population-based and graph-oriented optimisation. Section 3 presents Myco and its graph construction, community-weighted flow, adaptive conductance, and auxiliary search components. Section 4 describes the experimental protocol, comparative results, and ablation analyses. The final section concludes the paper and identifies limitations of the current design.

## 2. Related Work

Research on metaheuristic optimisation has produced a wide range of algorithms that may appear distinct at the metaphorical level, yet often share similar underlying computational structures [18, 23]. In many approaches, candidate solutions are updated using individual-level rules and exchange information through limited elite references or population-wide guidance, even as operator design and search configuration become more sophisticated [32]. Although such designs have led to many effective optimisers, they often provide only limited support for local cooperation, spatial grouping, and short-term structural memory. To address these limitations, research studies have increasingly explored methods that explicitly incorporate agent connectivity and population organisation into the search process, rather than relying solely on flat population-level updates [17].

## 2.1. Population-Based Search

Population-based optimisation is based on the idea that a set of concurrently updated candidate solutions can balance exploration and exploitation more effectively than a purely local, single-solution procedure. Genetic algorithm (GA) relies on population evolution through selection and variation [31]. Particle swarm optimisation (PSO) updates each particle by combining its own search history with information obtained from exemplar positions in the swarm [13]. Differential evolution (DE) generates trial solutions by taking vector differences among population members and has become a foundation of continuous global optimisation [26].

Subsequent work focused on increasing search quality by refining how individuals learn, mutate, or adapt their control parameters. CLPSO widened the learning source of each particle and improved diversity maintenance in multimodal search through comprehensive exemplar selection [16]. In the DE family, SAP-DE explored self-adaptive population sizing within differential evolution [29], JADE introduced adaptive differential evolution with an optional archive [33], SHADE formalised success-history-based parameter adaptation [27], L-SHADE added linear population size reduction [28], and jSO continued this progression toward high-performance real-parameter optimisation [5]. While these methods show that substantial gains can be obtained by improving parameter control, mutation design, and learning strategy within a shared population-update framework, their search logic remains centred on updating individuals within a flat population

Bio-inspired optimisers reformulated the population search logic by emulating various natural processes. Artificial bee colony (ABC) modelled foraging behaviour through employed, onlooker, and scout bees [12]. The grey wolf optimiser (GWO) and the whale optimisation algorithm (WOA) describe the search process through leadership hierarchy and hunting behaviour [21, 22]. The slime mould algorithm (SMA) translates oscillatory adaptation in slime mould into a stochastic optimiser [15]. The moss growth optimisation (MGO) is a recent example that organises the search through wind-direction estimation, spore dispersal, dual propagation, and cryptobiosis [34]. Despite their differences in inspiration and update mechanisms, these algorithms still update a population whose relational structure is only weakly modelled. This observation is important for the present paper because it motivates the shift from a flat population of individually updated agents to graph-structured search, in which candidate solutions interact through an explicitly defined and evolving interaction graph.

## 2.2. Network- and Graph-Oriented Optimisation

A more structured alternative to flat population search appears in optimisation methods that operate through adaptive graphs. Instead of relying solely on isolated update rules, these approaches allow connectivity, flow balance, and path reinforcement to influence the evolution of the search over time [3]. Physarum-inspired optimisation is a representative case, since it converts adaptive transport behaviour into graph-based computational dynamics [11].

Such methods have mainly been developed for network-centred problems. The accelerated Physarum solver demonstrated that the number of inactive nodes can be reduced and computation can be terminated earlier without sacrificing effectiveness in network optimisation tasks [11]. Physarum-inspired routing methods have similarly used adaptive graph dynamics to reconstruct favourable transmission paths in changing communication environments [20].

While these studies show that evolving network structure can guide optimisation, they do not fully address the setting considered in this study. Graph-oriented methods were designed for problems where the solution is already a path, route, or transport network [3, 20]. Mycelial Search (Myco) is motivated by this gap. It adopts a graph-based perspective on adaptive connectivity and flow-based reinforcement to organise interactions among candidate solutions in continuous optimisation. Therefore, the evolving network becomes a search mechanism rather than only a problemspecific representation.

## 3. Proposed Method

This section presents the Mycelial Search (Myco) at three connected levels. We first state the biological motivation that guided the design. We then define the network construction, the community-driven flow mechanism, the cord plasticity rule, and the anchor and Ridge-Oriented Injection (ROI) updates. Finally, we summarise the full algorithm and its computational cost.

## 3.1. Biological Motivation

The proposed method is motivated by the way fungal mycelia coordinate exploration and transport through a distributed network [1]. Fungal mycelia solve two linked problems at once: expanding into new regions while maintaining transport across the already-formed network. This dual behaviour motivates our proposed Mycelial Search, which formulates search as a spatial network in which candidate solutions interact through local connections, directional flow, and adaptive edge strength.

Myco does not model chemotropic growth or substrate-level physiology. Instead, it borrows the idea of many local explorers moving concurrently through a shared networked search space. Tips represent active exploratory units. Anchors preserve historically strong locations and stabilise the evolving network. Together, these components reflect three recurring features of mycelial organisation: exploration remains distributed, promising locations remain accessible to the rest of the network, and information is exchanged through local neighbourhoods rather than a single global communication pattern.

The strongest biological link in the method is the conductance update on tip-to-tip connections. In fungal transport networks, frequently used pathways tend to persist, whereas weakly used pathways regress [2, 6, 8]. The proposed method implements the same directional idea through adaptive cord conductance. Edges aligned with flow are reinforced, while edges not supported by flow decay over time. This mechanism is central to the method, where the community structure enters at a different level. The Louvain partition [4] introduces an intermediate scale of organisation. It is biologically motivated only in the limited sense that real mycelial networks exhibit spatially organised domains. However, the partition itself is algorithmic rather than biological. ROI remains outside the main biological claim and is included only as an auxiliary search mechanism.

## 3.2. Network Construction

We represent the current search state as an undirected, weighted spatial graph  whose nodes consist of active tips and retained anchors. Let � denote the problem dimension, � the number of tips, and � the number of retained anchors, where $K \leq K _ { \operatorname* { m a x } }$ . The position of tip � is denoted by $\boldsymbol { x } _ { i } \in \mathbb { R } ^ { D }$ for $i = 1 , \ldots , N$ , and the position of anchor � is denoted by $a _ { k } \in \mathbb { R } ^ { D }$ for $k = 1 , \dots , K$ . The corresponding node set is $\mathcal { V } = \{ x _ { 1 } , \ldots , x _ { N } , a _ { 1 } , \ldots , a _ { K } \}$

To keep the retained-anchor set compact, we cap the set at $K _ { \mathrm { m a x } }$ . This way, if more than $K _ { \mathrm { m a x } }$ anchors are available, only the $K _ { \mathrm { m a x } }$ anchors with the lowest objective values are retained.

Connectivity is induced locally through a fixed fusion radius. Let $\mathbf { l } \in \mathbb { R } ^ { D }$ and $\mathbf { u } \in \mathbb { R } ^ { D }$ denote the lower and upper bounds of the search space. We set the radius to $r _ { \mathrm { f u s e } } = 0 . 1 \| \mathbf { u } - \mathbf { l } \| _ { 2 }$ . The radius criterion determines only whether an edge is present. Since it does not specify the interaction strength between connected nodes, we quantify the effective edge weight using Eq. (1).

$$
e _ { p q } = \frac { g _ { p q } } { \lVert z _ { p } - z _ { q } \rVert _ { 2 } + \varepsilon } , \quad \varepsilon = 1 0 ^ { - 6 }\tag{1}
$$

where $g _ { p q }$ denotes the stored conductance associated with the undirected pair $( p , q )$ , and $z _ { p }$ and $z _ { q }$ are the position of nodes � and � in 

The conductance is shared symmetrically by $( p , q )$ and $( q , p )$ . Previously unseen tip-to-tip pairs are initialised with $g _ { p q } \ = \ 1$ , whereas anchor-incident edges retain fixed conductance values and are not modified by plasticity. Pairs that fail the radius condition are simply absent from the graph at that iteration. In this construction, anchors serve as elite landmarks that shape the local neighbourhood structure and influence the subsequent flow field without themselves becoming adaptive cords. We restrict conductance adaptation to tip-to-tip edges so that anchors provide stable reference points while the search network remains free to reorganise around them.

## 3.3. Cord Plasticity

We use cord plasticity to provide a short-term memory mechanism on the weighted graph by reinforcing directions that are repeatedly supported by the induced flow. The rule applies only to present tip-to-tip edges in the current graph. Let $\mathcal { N } _ { i }$ denote the set of all neighbours of tip � in the current graph, including anchors when present, and let $\mathcal { N } _ { i } ^ { \mathrm { t i p } } \subseteq \mathcal { N } _ { i }$ denote the subset containing only tip neighbours. For each undirected tip pair $( i , j )$ with $j \in \mathcal { N } _ { i } ^ { \mathrm { t i p } }$ , the conductance is stored as a single shared scalar $g _ { i j } = g _ { j i }$ . Edges incident to anchors remain part of the graph and contribute to the flow field, but they are excluded from conductance adaptation. Let $\phi _ { i } \in \mathbb { R } ^ { D }$ be the flow vector associated with tip �, and $\tau _ { 0 }$ denote a numerical tolerance. For $\| \phi _ { i } \| _ { 2 } \geq \tau _ { 0 }$ , we normalize the local flow according to Eq. (2).

$$
\hat { \phi } _ { i } = \frac { \phi _ { i } } { \lVert \phi _ { i } \rVert _ { 2 } } .\tag{2}
$$

To define the edge direction vector, we use $c _ { i j } = x _ { j } - x _ { i }$ for each $j \in \mathcal { N } _ { i } ^ { \mathrm { t i p } } . \mathrm { I f } \| c _ { i j } \| _ { 2 } < \tau _ { 0 } .$ , the edge is left unchanged during the update of tip �. Otherwise, directional agreement between the edge and the local flow is measured by a cosine alignment score defined in Eq. (3).

$$
s _ { i j } = \frac { c _ { i j } ^ { \top } \hat { \phi } _ { i } } { \Vert c _ { i j } \Vert _ { 2 } } .\tag{3}
$$

Given the alignment score in Eq. (3), reinforcement is applied only when $s _ { i j } > \tau _ { \mathrm { a l i g n } } .$ , where $\tau _ { \mathrm { a l i g n } }$ is the alignment threshold. In that case, the reinforced conductance is obtained from Eq. (4).

$$
\begin{array} { r } { \tilde { g } _ { i j } = \operatorname* { m i n } \mathopen { } \mathclose \bgroup \left( g _ { \mathrm { m a x } } , g _ { i j } ( 1 + \eta s _ { i j } ) \aftergroup \egroup \right) , } \end{array}\tag{4}
$$

where $\eta$ is the reinforcement rate and $g _ { \mathrm { m a x } }$ is the upper conductance bound. For $s _ { i j } \leq \tau _ { \mathrm { a l i g n } } .$ , we set $\tilde { g } _ { i j } = g _ { i j }$ . Decay with lower clipping is then applied through Eq. (5).

$$
g _ { i j } ^ { + } = \operatorname* { m a x } \left( g _ { \operatorname* { m i n } } , ( 1 - \delta ) \tilde { g } _ { i j } \right) .\tag{5}
$$

where � is the decay rate and $g _ { \mathrm { m i n } }$ is the lower conductance bound.

Because plasticity is evaluated once from the perspective of each tip, the same undirected conductance may be revised twice within a single iteration, once when processing endpoint � and once when processing endpoint �. Therefore, the final end-of-iteration value may include a second update from the opposite endpoint. This update structure preserves conductance symmetry because both endpoints modify the same shared scalar. As a result, edges repeatedly aligned with the induced flow are first reinforced using Eq. (4) and then decayed and clipped using Eq. (5), whereas in the low-flow regime only Eq. (5) is applied.

## 3.4. Community Structure and Mesoscale Flow

We use the Louvain method [4] to partition  into communities at each iteration. For all reported experiments, we fix the community-detection random seed to 42. The raw partition is refined by a two-criterion filter. A community is retained if it contains at least $n _ { \mathrm { m i n } }$ nodes and its induced subgraph density satisfies $\rho \geq \rho _ { \mathrm { m i n } } ,$ where $\rho = 2 m / ( n ( n - 1 ) )$ for a subgraph with � nodes and � internal edges. Nodes belonging to rejected communities are each reassigned a unique singleton label. They remain in the graph with all edges preserved and continue to contribute to the flow computation on equal terms with all other nodes. After reassignment, interactions involving singleton-labelled nodes use $\lambda _ { \mathrm { { i n t e r } } }$ in Eq. (7).

We define a potential for every node $v \in \mathcal { V }$ , over both tips and anchors. Let �(�) denote the objective value associated with node �. With $f _ { \mathrm { m i n } }$ and $f _ { \mathrm { m a x } }$ computed over all nodes in  at the current iteration, the potential is defined in Eq. (6).

$$
s ( v ) = \left\{ \begin{array} { c c } { \frac { f _ { \mathrm { m a x } } - f ( v ) } { f _ { \mathrm { m a x } } - f _ { \mathrm { m i n } } + \varepsilon } , } & { f _ { \mathrm { m a x } } \neq f _ { \mathrm { m i n } } } \\ { 1 , } & { \mathrm { o t h e r w i s e } } \end{array} \right. , \qquad \varepsilon = 1 0 ^ { - 6 }\tag{6}
$$

For tip �, the flow vector $\phi _ { i }$ is defined as a weighted mean of spatial displacements toward all neighbours $j \in \mathcal N _ { i }$ in the current graph, including anchors. We define the message weight from node � to tip � as in Eq. (7).

$$
m _ { i j } = \lambda ( i , j ) \cdot e _ { i j } \cdot \sigma _ { i j } ,\tag{7}
$$

where $e _ { i j }$ is the edge weight. The community factor $\lambda ( i , j )$ takes the value $\lambda _ { \mathrm { i n t r a } }$ for node pairs in the same filtered community and $\lambda _ { \mathrm { { i n t e r } } }$ for node pairs in different filtered communities, where $\lambda _ { \mathrm { i n t r a } }$ and $\lambda _ { \mathrm { { i n t e r } } }$ are constant weights for within-community and cross-community interactions, respectively. The sigmoid gate is defined in Eq. (8).

$$
\sigma _ { i j } = \frac { 1 } { 1 + \exp \left( - \kappa \left( s _ { j } - s _ { i } \right) \right) } ,\tag{8}
$$

where $\sigma _ { i j }$ modulates the contribution according to the potential difference between neighbouring node � and receiving tip �. We compute the flow vector $\phi _ { i }$ as the weighted mean of spatial displacements toward the neighbours of tip � using Eq. (9).

$$
\phi _ { i } = \frac { \sum _ { j \in \mathcal { N } _ { i } } m _ { i j } \left( z _ { j } - z _ { i } \right) } { \sum _ { j \in \mathcal { N } _ { i } } m _ { i j } } ,\tag{9}
$$

where $z _ { i }$ and $z _ { j }$ are the position of node � and � in , respectively. When the total message weight is zero, $\phi _ { i } = \mathbf { 0 }$ . We anneal the sigmoid steepness � linearly over the run. Let FE(�) denote the cumulative number of function evaluations consumed up to the start of iteration �, and let $\mathrm { F E } _ { \mathrm { m a x } }$ be the total evaluation budget. Then Eq. (10) defines the linear annealing schedule for �.

$$
\kappa ( t ) = \kappa _ { \mathrm { m a x } } - ( \kappa _ { \mathrm { m a x } } - \kappa _ { \mathrm { m i n } } ) \frac { \mathrm { F E } ( t ) } { \mathrm { F E } _ { \mathrm { m a x } } } .\tag{10}
$$

A larger � makes the sigmoid more selective with respect to potential differences, suppressing contributions from neighbours with lower potential. As � decreases toward $\kappa _ { \mathrm { m i n } }$ , the gate becomes less selective and the flow aggregates more uniformly across the neighbourhood.

## 3.5. Tip Update and Anchor Management

We update the tip states and the retained-anchor set within the same iteration. We first evaluate the tips and update the incumbent best position $x ^ { \star } ( t )$ , which is the only position eligible for anchor insertion. We insert this position only when its Euclidean distance from every retained anchor exceeds $\varepsilon = 1 0 ^ { - 6 }$ . If the number of retained anchors exceeds

$K _ { \mathrm { m a x } }$ , we keep the anchors with the best stored objective values and reevaluate the retained anchors before computing the node-wise potentials. These updated anchor values are then used in the current potential field and the resulting flow computation. When ROI is active, we also use values to evaluate the relative fitness gap that triggers the mechanism.

Let $\boldsymbol { v } _ { i } ^ { t } \in \mathbb { R } ^ { D }$ denote the velocity of tip � at iteration � and $r _ { i } ^ { t } \in [ 0 , 1 ] ^ { D }$ be a random vector sampled independently for tip �. We perturb the current velocity of tips with at most one graph neighbour before applying the main velocity update. This perturbation is defined in Eq. (11).

$$
\tilde { v } _ { i } ^ { t } = \{ \begin{array} { l l } { v _ { i } ^ { t } + \xi _ { i } ^ { t } , } & { | \mathcal { N } _ { i } | \leq 1 } \\ { v _ { i } ^ { t } , } & { | \mathcal { N } _ { i } | > 1 } \end{array} , \quad \xi _ { i } ^ { t } \sim \mathcal { V } ( [ - \delta , \delta ] ^ { D } ) , \quad \delta = 0 . 0 5 r _ { \mathrm { f u s e } } ,\tag{11}
$$

where $\mathcal { V } ( [ - \delta , \delta ] ^ { D } )$ denotes the uniform distribution over the hypercube $[ - \delta , \delta ] ^ { D }$ . We then calculate the velocity of tip � at iteration $t + 1$ using Eq. (12).

$$
v _ { i } ^ { t + 1 } = w \tilde { v } _ { i } ^ { t } + c _ { \mathrm { e x p } } r _ { i } ^ { t } \odot ( x ^ { \star } ( t ) - x _ { i } ^ { t } ) + c _ { \mathrm { f l o w } } \phi _ { i } ,\tag{12}
$$

where � is the inertia coefficient, $c _ { \mathrm { e x p } }$ is the coefficient of the best-position term, $c _ { \mathrm { f l o w } }$ is the coefficient of the flow term, and ⊙ denotes element-wise multiplication. We update the tip position through Eq. (13).

$$
\begin{array} { r } { x _ { i } ^ { t + 1 } = x _ { i } ^ { t } + v _ { i } ^ { t + 1 } . } \end{array}\tag{13}
$$

We clip each component of $x _ { i } ^ { t + 1 }$ to the corresponding interval defined by � and � after the position update.

## 3.6. Ridge-Oriented Injection Spawning

Ridge-Oriented Injection (ROI) is an auxiliary mechanism that injects a candidate position derived from the retained-anchor set. It is considered only when at least two anchors are present. We sample two distinct anchors $\boldsymbol { a } _ { k _ { 1 } }$ and $\boldsymbol { a } _ { k _ { 2 } }$ uniformly, without replacement, and evaluate their relative fitness gap using Eq. (14).

$$
\gamma = { \frac { \vert f ( a _ { k _ { 1 } } ) - f ( a _ { k _ { 2 } } ) \vert } { \operatorname* { m a x } _ { 1 \leq k \leq K } f ( a _ { k } ) - \operatorname* { m i n } _ { 1 \leq k \leq K } f ( a _ { k } ) + \varepsilon } } .\tag{14}
$$

The ROI branch proceeds when $\gamma > \tau _ { \mathrm { r o i } }$ , where $\tau _ { \mathrm { r o i } }$ is the relative-gap threshold. The midpoint $x _ { \mathrm { m i d } } = ( a _ { k _ { 1 } } + a _ { k _ { 2 } } ) / 2$ serves as the base for the spawn construction. We estimate $\nabla f ( x _ { \mathrm { m i d } } )$ by central finite differences, as specified in Eq. (15).

$$
\left[ \nabla f ( x _ { \mathrm { m i d } } ) \right] _ { d } = \frac { f ( x _ { \mathrm { m i d } } + h \mathbf { e } _ { d } ) - f ( x _ { \mathrm { m i d } } - h \mathbf { e } _ { d } ) } { 2 h } , \quad d = 1 , \ldots , D ,\tag{15}
$$

where $\mathbf { e } _ { d }$ denotes the �-th standard basis vector and $h = 1 0 ^ { - 5 } \operatorname* { m a x } \{ | f ( x _ { \mathrm { m i d } } ) | , 1 \}$ is the step size. In addition to the evaluation of $f ( x _ { \mathrm { m i d } } )$ needed to set ℎ, this sub-step requires $2 D + 1$ objective evaluations.

The geometric adjustment preserves only the gradient component orthogonal to the anchor axis $u \ = \ ( a _ { k _ { 2 } } \ -$ $a _ { k _ { 1 } } ) / \parallel a _ { k _ { 2 } } - a _ { k _ { 1 } } \parallel _ { 2 } .$ and the transverse component is derived using Eq. (16).

$$
\begin{array} { r } { p _ { \perp } = \hat { p } - ( \hat { p } ^ { \top } u ) u , \quad } \\ { \hat { p } = \cfrac { \nabla f ( x _ { \mathrm { m i d } } ) } { \| \nabla f ( x _ { \mathrm { m i d } } ) \| _ { 2 } } , \quad } \end{array}\tag{16}
$$

When either $\| a _ { k _ { 2 } } - a _ { k _ { 1 } } \| _ { 2 }$ or $\| \nabla f ( x _ { \mathrm { m i d } } ) \| _ { 2 }$ is below $\tau _ { \mathrm { d e g } } = 1 0 ^ { - 9 }$ , the correction degenerates and $x _ { \mathrm { m i d } }$ is passed through unchanged. The spawn candidate is then calculated using Eq. (17).

$$
x _ { \mathrm { r o i } } = \Pi _ { [ \mathbf { l } , \mathbf { u } ] } \big ( x _ { \mathrm { m i d } } - \alpha _ { \mathrm { r o i } } p _ { \perp } \big ) ,\tag{17}
$$

where $\Pi _ { [ \mathbf { l } , \mathbf { u } ] }$ denotes component-wise clipping to the feasible domain and $\alpha _ { \mathrm { r o i } }$ is the ROI correction coefficient. We evaluate $x _ { \mathrm { r o i } }$ and replace the current worst tip only when the spawned value is strictly lower, adding one objective evaluation to the cost of the triggered ROI branch.

## 3.7. Algorithm Summary and Complexity

Algorithm 1 summarises the complete procedure. We use it to consolidate the interaction between the retainedanchor mechanism, the community-weighted flow field, the conductance update, and the tip dynamics defined in Section 3.2 through Section 3.6. The resulting iteration uses the current tip and anchor evaluations to form the potential field, applies the filtered Louvain partition to the flow computation, updates tip-tip conductances based on the induced flow, and then advances the tips. When ROI is enabled, the same iteration may also inject one additional candidate derived from a selected anchor pair.

## Algorithm 1. Mycelial Search

```latex
Require: Objective function $f ,$ dimension �, bounds �, �, number of tips �, anchor cap $K _ { \mathrm { m a x } }$ , evaluation budge
$\mathrm { F E } _ { \mathrm { m a x } } .$ , model hyperparameters
Ensure: Incumbent best position $x ^ { \star } ( t )$ and, when needed, its objective value $f ^ { \star } ( t )$
% Initialization $\%$
1: $r _ { \mathrm { f u s e } }  0 . 1 \| \mathbf { u } - \mathbf { l } \| _ { 2 }$
2: Initialize $x _ { i } \sim \mathcal { V } ( [ \mathbf { I } ,$ �]) and $v _ { i } ^ { 0 }  \mathbf { 0 }$ for $i = 1 , \ldots , N$
3: Initialize retained anchors $\mathcal { A }  \emptyset$ and the conductance map for tip-tip pairs
4: Set FE(�) ← 0, �<sup>⋆</sup>(�) ← +∞, and $x ^ { \star } ( t ) \gets$ null
5: while FE(�) $< \mathrm { F E } _ { \mathrm { m a x } }$ do
% Annealed Sigmoid Steepness %
6: Update �(�) using Eq. (10) with the cumulative evaluation count at the start of the current iteration
% Tip Evaluation and Incumbent Update %
7: Evaluate all current tips and increment FE(�) by �
8: Update $f ^ { \star } ( t )$ and $x ^ { \star } ( t )$ if a better tip is found
% Anchor Management %
9: Insert $x ^ { \star } ( t )$ into  only if its Euclidean distance from every retained anchor exceeds $1 0 ^ { - 6 }$
10: $\mathbf { i f } \mid \mathcal { A } \mid > K _ { \operatorname* { m a x } }$ then
11: Retain the $K _ { \mathrm { m a x } }$ anchors with the best stored objective values
12: end if
13: Reevaluate all retained anchors and increment FE(�) by $| { \mathcal { A } } |$
% Node-wise Potentials from Already Available Objective Values %
14: Form the node set ${ \mathfrak { P } } = \{ x _ { 1 } , \dots , x _ { N } \} \cup A$
15: Collect the already evaluated objective values $\{ f ( v ) : v \in \mathcal { V } \}$ from the current tips and retained anchors
16: Compute �(�) for all $v \in \mathcal V$ using Eq. (6)
% Graph Construction and Community Structure $\%$
17: Construct the current graph  using $r _ { \mathrm { f u s e } }$
18: Compute edge weights $e _ { p q }$ using Eq. (1)
19: Compute the Louvain partition of  with seed 42
20: Filter the raw communities using $n _ { \mathrm { m i n } }$ and $\rho _ { \mathrm { m i n } }$ , then relabel rejected communities as singletons
% Mesoscale Flow %
21: for � ← 1 to � do
22: Compute $\phi _ { i }$ from Eqs. (7)–(9)
over all $j \in \mathcal { N } _ { i } ,$ using the previously computed
potentials and filtered community labels
23: end for
% Cord Plasticity %
24: for � ← 1 to � do
```

Mycelial Search for Continuous Optimisation   
25: Update the conductances of present tip-tip edges (�, �) with   
$j \in \mathcal { N } _ { i } ^ { \mathrm { t i p } }$ using the previously computed   
flow vector $\phi _ { i }$ and   
Eqs. (2)–(5)   
26: end for   
% Tip Motion %   
27: for $i \gets 1$ to � do   
28: if $| \mathcal { N } _ { i } | \leq 1$ then   
29: Sample $\xi _ { i } ^ { t } \sim \mathcal { V } ( [ - \delta , \delta ] ^ { D } )$ with $\delta = 0 . 0 5 r _ { \mathrm { f u s e } }$   
30: Set $\tilde { v } _ { i } ^ { t } \gets \dot { v } _ { i } ^ { t } + \xi _ { i } ^ { t }$   
31: else   
32: Set $\tilde { v } _ { i } ^ { t } \gets v _ { i } ^ { t }$   
33: end if   
34: Sample $r _ { i } ^ { t } \sim \mathcal { V } ( [ 0 , 1 ] ^ { D } )$   
35: Update $\boldsymbol { v } _ { i } ^ { t + 1 }$ using Eq. (12)   
36: Update $x _ { i } ^ { i + 1 }$ using Eq. (13)   
37: Clip $x _ { i } ^ { t + 1 }$ component-wise to [�, �]   
38: end for   
% Auxiliary ROI Branch %   
39: if ROI is enabled and $| { \mathcal { A } } | \geq 2$ then   
40: Sample two distinct anchors $a _ { k _ { 1 } } , a _ { k _ { 2 } } \in { \mathcal { A } }$ uniformly without replacement   
41: Compute the relative gap � using Eq. (14)   
42: if $\gamma > \tau _ { \mathrm { r o i } }$ then   
43: Set $x _ { \mathrm { m i d } }  ( a _ { k _ { 1 } } + a _ { k _ { 2 } } ) / 2$   
44: Estimate $\nabla f ( x _ { \mathrm { m i d } } ^ { \mathrm { ' } } )$ using Eq. (15)   
45: Update $\mathrm { F E } ( t )  \mathrm { F E } ( t ) + 2 D + 1$   
46: Compute $p _ { \perp }$ using Eq. (16)   
47: Construct $x _ { \mathrm { r o i } }$ using Eq. (17)   
48: Evaluate $x _ { \mathrm { r o i } }$ and update $\mathrm { F E } ( t ) \gets \mathrm { F E } ( t ) + 1$   
49: Replace the current worst tip only if $x _ { \mathrm { r o i } }$ yields a strictly lower objective value   
50: end if   
51: end if   
52: end while   
53: return $x ^ { \star } ( t )$ and $f ^ { \star } ( t )$

Let $K \ \leq \ K _ { \operatorname* { m a x } }$ denote the number of retained anchors at the current iteration, let $M = N + K$ denote the total number of graph nodes, and let $E = | { \boldsymbol { \mathcal { E } } } |$ denote the number of graph edges. Excluding the cost of objectivefunction evaluation, graph construction with spatial neighbourhood queries requires �(� log $M + E )$ operations. The mesoscale flow computation and the cord-plasticity update each scale as $O ( E D )$ . The velocity and position updates scale as �(��). Therefore, the per-iteration arithmetic cost is given by Eq. (18).

�(� log $M + E D + N D ) .$

(18)

We cap the anchor set by $K _ { \mathrm { m a x } }$ , and the fusion radius induces local connectivity. In the sparse regime considered here, � therefore grows approximately linearly with �. Under this condition, Eq. (18) reduces in practice to �(� log $N + N D )$ . Each iteration requires $N + K$ objective evaluations from tip evaluation and anchor reevaluation. When the ROI branch is triggered, it adds $2 D + 2$ additional evaluations: $2 D + 1$ for the midpoint-based numerical gradient and 1 for the spawned candidate. Since $K \le K _ { \operatorname* { m a x } }$ , the baseline evaluation cost per iteration remains linear in �, and the total number of evaluations is bounded by $\mathrm { F E } _ { \mathrm { m a x } }$

Table 1  
Experimental protocol for the CEC 2022 benchmark evaluation.
<table><tr><td>Protocol component</td><td>D = 10</td><td> $\overline { { \mathbf { D } = 2 \mathbf { 0 } } }$ </td></tr><tr><td>Test functions</td><td>F1-F11</td><td>F1-F11</td></tr><tr><td>Independent runs</td><td>30</td><td>30</td></tr><tr><td>Maximum function evaluations</td><td>200,000</td><td>1,000,000</td></tr><tr><td>Evaluation budget</td><td>20,000D</td><td>50,000D</td></tr><tr><td>Search bounds</td><td>[-100,100]</td><td>[-100,100]</td></tr><tr><td>Myco population</td><td>30 tips</td><td>30 tips</td></tr><tr><td>Comparator population</td><td>50 individuals</td><td>50 individuals</td></tr></table>

Table 2  
External comparator set organised by algorithm family.
<table><tr><td>Algorithm family</td><td>Optimisers</td></tr><tr><td>Differential evolution</td><td>jSO [5], SAP-DE [29], L-SHADE [28], JADE [33]</td></tr><tr><td>Swarm intelligence</td><td>CLPSO [16], ABC [12]</td></tr><tr><td>Genetic</td><td>GA [31]</td></tr><tr><td>Mammal-inspired</td><td>GWO [22], WOA [21]</td></tr><tr><td>Decentralised growth</td><td>SMA [15], MGO [34]</td></tr></table>

This gives $O ( \mathrm { F E _ { \mathrm { m a x } } } / N )$ iterations and yields the total arithmetic cost $O \big ( \mathrm { F E } _ { \operatorname* { m a x } } ( \log N + D ) \big )$ , excluding the cost of the objective function itself. When � dominates log �, the total arithmetic cost reduces to $O ( \mathrm { F E } _ { \operatorname* { m a x } } D )$

## 4. Experiments

We evaluate Myco on the CEC 2022 single-objective bound-constrained benchmark suite using functions F1 to F11 at dimensions � = 10 and � = 20. The experiments compare final-error performance, convergence behaviour, dimensional changes, and the contributions of cord plasticity and the community-detection backend. We first define the experimental protocol and comparator set. We then report the main comparative results and analyse the internal ablations.

## 4.1. Experimental Setup

Table 1 summarises the experimental protocol used for the CEC 2022 benchmark evaluation [14]. The experiments comprise 30 independent runs per algorithm and function. Myco achieved its reported results with 30 tips under an equal FE budget, while the comparator implementations used their configured population size of 50. We grouped the comparator methods by algorithmic family in Table 2. Final errors are reported by their mean and standard deviation. To facilitate reproducibility, the Python implementation of Myco is publicly available at https://github.com/ dehshibi/Mycelia-Search.

Following the benchmark rule, random seeds were generated deterministically for the function � and run � by $I _ { f , r } = f R + r - R$ , followed by $\sigma _ { f , r } = ( I _ { f , r }$ mod $1 0 0 0 ) + 1$ , where $R = 3 0$ is the number of independent runs. Convergence was recorded at 16 predefined checkpoints given by $\lfloor D ^ { k / 5 - 3 } \mathrm { F E } _ { \mathrm { m a x } } \rfloor$ for $k = 0 , \ldots , 1 5$ , where $\mathrm { F E } _ { \mathrm { m a x } }$ denotes the maximum number of function evaluations.

## 4.2. Results

Tables 3 and 4 report the final-error distributions at � = 10 and � = 20. The family ordering separates comparisons among methods that use different search principles. Boldface identifies the lowest mean final error for each function. Shading identifies the lowest mean within each non-differential-evolution family. The latter distinction is useful because a global row-wise winner does not, by itself, show whether the proposed network mechanism is competitive with search methods built on different information-sharing structures.

The benchmark classes impose distinct search demands. F1 is a shifted and fully rotated Zakharov function with a single basin. As a unimodal function, it tests directed progress in a rotated domain. F2 contains the curved and non-separable Rosenbrock valley. F3 to F5 present increasingly difficult multimodal conditions through expanded

Table3:Finalerrorsat�=10over30independentruns,reportedasmean±standarddeviation.Thereportederroristhedifferencebetweentheobtained objectivevalueandtheknownoptimumofthecorrespondingCEC2022function.Algorithmsaregroupedbysearchfamily.Boldfaceidentifiesthelowest meanfinalerrorforeachfunction.LightshadingidentifiesthelowestmeanwithintheGrowthNetwork,Genetic,SwarmIntelligence,Mammal-inspired,and <sub>ntralised</sub> <sub>Growth</sub> <sub>fa</sub>m
<table><tr><td>Function</td><td>Growth Network Myco</td><td>Genetic/Evolutionary GA</td><td colspan="2">Swarm Intelligence CLPSO</td><td colspan="2">Mammal-inspired</td><td colspan="2">Decentralised Growth</td><td colspan="4">Differential Evolution SAP-DE</td></tr><tr><td>F1</td><td>0± 0</td><td>3494 ±3337</td><td>1429 ± 486</td><td>ABC 4594 ± 1206</td><td>WOA 498 ± 1136</td><td>GWO 14.5 ± 19.7</td><td>SMA 6.89 × 10−4 ± 3.45 × 10−4</td><td>MGO 4900 ± 1797</td><td>jSO 0± 0</td><td>5635±2189</td><td>L-SHADE 0± 0</td><td>JADE 0 ± 0</td></tr><tr><td>F2</td><td>6.14±2.43</td><td>7.34± 11.6</td><td>21 ± 13.8</td><td>8.74 ± 0.739</td><td>11.7 ± 9.38</td><td>7.37 ± 3.14</td><td>7.25 ± 1.87</td><td>60 ± 30.1</td><td>4.78±3.53</td><td>3.19±2.98</td><td>7.14 ±2.6</td><td>6.45 ±2.46</td></tr><tr><td>F3</td><td>3.61 ± 0.418</td><td>3.2 ± 0.374</td><td>3.12 ± 0.16</td><td>3.26 ± 0.148</td><td>3.54 ± 0.294</td><td>2.21 ± 0.493</td><td>2.43 ± 0.459</td><td>3.02 ± 0.289</td><td>2.71 ± 0.475</td><td>2.62± 0.315</td><td>2.34±0.278</td><td>1.8 ± 0.373</td></tr><tr><td>F4</td><td>178 ± 737</td><td>70.4 ± 22.9</td><td></td><td></td><td></td><td></td><td></td><td>903 ± 390</td><td></td><td></td><td></td><td>7.28±3.38</td></tr><tr><td></td><td></td><td></td><td>69.9 ± 9.05</td><td>40.1 ± 4.9</td><td>1565 ± 1073</td><td>26.8±11.3</td><td>29.5 ± 8.8</td><td></td><td>22.9±7.76</td><td>22.6±4.55</td><td>9.16±4.55</td><td></td></tr><tr><td>F5</td><td>524 ± 617</td><td>97.6 ± 120</td><td>26 ± 11.3</td><td>8.93 ± 3.66</td><td>566 ± 343</td><td>1.54 ± 4.44</td><td>0.0727 ± 0.154</td><td>158 ± 63.6</td><td>0.0966 ± 0.172</td><td>7.83 ± 6.29</td><td>0 ± 0</td><td>0 ± 0</td></tr><tr><td>F6</td><td>92 ±143</td><td>7034 ± 5061</td><td>3.08 × 104 ± 8980</td><td>428 ± 268</td><td>4066 ± 4026</td><td>3295 ± 2482</td><td>3184 ± 2124</td><td>1.51 × 104 ± 4851</td><td>3.55 ± 1.7</td><td>205±380</td><td>0.309 ± 0.218</td><td>0.369 ± 0.111</td></tr><tr><td>F7</td><td>1691 ± 4411</td><td>2837±3084</td><td>4.00 × 104 ± 2.55 × 104</td><td>2472 ± 605</td><td>3.50 ×104 ± 1.86 × 105</td><td>3431 ± 2826</td><td>1614±1054</td><td>9.39 × 104 ± 8.30×104</td><td>20.2± 84.6</td><td>240±313</td><td>2.57 ± 13</td><td>41.9 ± 125</td></tr><tr><td>F8</td><td>20.1 ± 0.233</td><td>6.34±8.94</td><td>20.4 ± 0.542</td><td>20.2 ± 0.525</td><td>20 ± 0.156</td><td>18.3 ± 6.1</td><td>19.3 ± 3.59</td><td>19.8 ± 0.866</td><td>12.5 ± 9.4</td><td>0.54±1.87</td><td>0.057 ± 0.0661</td><td>11.9 ± 7.03</td></tr><tr><td>F9</td><td>3.97 ± 2.9</td><td>346 ± 639</td><td>8.46 × 104 ± 2.77 × 104</td><td>0.9 ± 0</td><td>4.86 ± 2.84</td><td>2.68× 104±2.80×104</td><td>0.907 ± 0.012</td><td>1.62 × 107 ± 1.26 × 107</td><td>1.7 ± 2.06</td><td>6.72 ± 16.1</td><td>0.9 ± 0</td><td>0.9 ± 0</td></tr><tr><td>F10</td><td>19.8 ± 19.3</td><td>298 ± 340</td><td>1466 ± 766</td><td>25.6 ± 11.3</td><td>54.7 ± 55</td><td>670 ± 2306</td><td>26.7 ± 19.6</td><td>3.23×107± 2.73×107</td><td>7.64±11.7</td><td>22.9±48.9</td><td>1.8 ± 0</td><td>1.8 ± 0</td></tr><tr><td>F11</td><td>4.69 ± 1.78</td><td>1.93 ± 0.641</td><td>5805 ± 9212</td><td>26.8 ± 7.06</td><td>48.8 ± 27.8</td><td>10.7 ± 4.94</td><td>0.804 ± 0.294</td><td>5.63×1011±5.11×1011</td><td>2.03 ± 0.845</td><td>3.3 ± 3.9</td><td>0.779 ± 0.165</td><td>0.775 ± 0.147</td></tr></table>

Table4:Finalerrorsat�=20over30independentruns,reportedasmean±standarddeviation.Thereportederroristhedifferencebetweentheobtained objectivevalueandtheknownoptimumofthecorrespondingCEC2022function.Algorithmsaregroupedbysearchfamily.Boldfaceidentifiesthelowest meanfinalerrorforeachfunction.LightshadingidentifiesthelowestmeanwithintheGrowthNetwork,Genetic,SwarmIntelligence,Mammal-inspired,and <sub>ntralised</sub> <sub>Growth</sub> <sub>fa</sub>m
<table><tr><td>Function</td><td>Growth Network Myco</td><td>Genetic/Evolutionary GA</td><td>Swarm Intelligence</td><td>ABC</td><td>Mammal-inspired</td><td></td><td>Decentralised Growth</td><td></td><td></td><td>Differential Evolution</td><td></td><td></td></tr><tr><td>F1</td><td>0 ± 0</td><td>2147 ± 972</td><td>CLPSO 3273 ± 895</td><td>2.68 × 104 ± 3882</td><td>WOA 7.89 ± 42.5</td><td>GWO 167 ± 230</td><td>SMA 1.66 × 10−3 ± 8.47 × 10−4</td><td>MGO 2.06 × 104 ± 2811</td><td>jSO 0 ± 0</td><td>SAP-DE 2.31× 104 ± 7685</td><td>L-SHADE 0 ± 0</td><td>JADE 0 ± 0</td></tr><tr><td>F2</td><td>29.9 ± 22.7</td><td>53.6 ± 17.4</td><td>74.5 ± 9.44</td><td>79.3 ± 4.42</td><td>48.8 ± 21.6</td><td>56.7 ± 20.9</td><td>48.4 ± 4.6</td><td>227 ± 32.1</td><td>31.1 ± 22.2</td><td>32.5 ± 13.4</td><td>42.4 ± 16.6</td><td>48.8 ± 1.04</td></tr><tr><td>F3</td><td>8.11 ± 0.421</td><td>7.57 ± 0.477</td><td>7.5 ± 0.206</td><td>7.87 ± 0.133</td><td>8.07 ± 0.369</td><td>6.02 ± 0.571</td><td>6.47 ± 0.538</td><td>7.74 ± 0.29</td><td>7.26 ± 0.57</td><td>6.94 ± 0.36</td><td>6.75 ± 0.308</td><td>5.81 ± 0.525</td></tr><tr><td>F4</td><td>3775 ± 3935</td><td>225 ± 60.8</td><td>145 ± 14.3</td><td>233 ± 18.6</td><td></td><td>128 ± 54.2</td><td>87.2 ± 17.6</td><td>6356 ± 1595</td><td>91.7 ± 18</td><td>101 ± 15.4</td><td>35.3 ± 5.48</td><td>29.9 ± 9.93</td></tr><tr><td>F5</td><td>2654 ± 1429</td><td>477 ± 239</td><td>138 ± 54.1</td><td>1005 ± 150</td><td>1.07 × 104 ± 3544 2392 ± 1107</td><td>27.4 ± 57.6</td><td>20.5 ± 34.8</td><td>838 ± 225</td><td>61.1 ± 70.7</td><td>467 ± 140</td><td>0 ± 0</td><td>2.98× 10-3 ± 0.0161</td></tr><tr><td>F6</td><td>9.53 × 104 ± 4.66× 104</td><td>4.94 × 104 ± 2.25 × 104</td><td>3.90 × 105 ± 1.31 ×105</td><td>1.01 × 107 ±4.19 × 106</td><td>1.84× 104 ± 1.16 × 104</td><td>7.20×104 ±2.31×104</td><td>1.47 × 105 ± 2.71 × 104</td><td>3.35 × 104 ± 6176</td><td>1962±1062</td><td>4.04× 104±8783</td><td>1118±534</td><td>1.23 × 104 ± 1.21 × 104</td></tr><tr><td>F7</td><td>2144±768</td><td>5.66 × 104 ± 6.28 × 104</td><td>1.59 × 105 ± 6.25 × 104</td><td>1.52 × 106 ± 4.50 × 105</td><td>1.91 × 105 ± 8.20 × 105</td><td>1.79 × 105 ± 1.91 × 105</td><td>9700 ± 4999</td><td>1.68 × 106 ± 3.18× 106</td><td>557±393</td><td>8.12×104±1.21×105</td><td>225±316</td><td>375±428</td></tr><tr><td>F8</td><td>20.2 ± 0.412</td><td>19.4 ± 3.54</td><td>21.3 ± 0.14</td><td>21.5 ± 0.217</td><td>20.3 ± 0.397</td><td>21 ± 0.572</td><td>20.1 ± 0.0632</td><td>20.1 ± 0.0258</td><td>20 ± 0.0263</td><td>20 ± 9.32× 10−3</td><td>20 ± 0.0156</td><td>20.6 ± 0.161</td></tr><tr><td>F9</td><td>6.98 ±3.85</td><td>37.2 ± 42.3</td><td>6.93 × 104 ± 2.17× 104</td><td>3.28 × 106 ±9.40× 105</td><td>10.4 ± 5.36</td><td>3.17×105± 8.96× 105</td><td>4.23 ± 3.32</td><td>3.90×107 ± 2.54× 107</td><td>6.02 ±4.05</td><td>9.76 ± 11.4</td><td>2.77 ±2.23</td><td>2.11 ± 1.16</td></tr><tr><td>F10</td><td>57.6±41.9</td><td>356 ± 398</td><td>3253 ± 1363</td><td>1.33 × 106 ± 4.37 × 105</td><td>122 ± 69.9</td><td>2.08× 104 ± 5.19 × 104</td><td>34.4 ± 24.3</td><td>5.33 × 107 ± 5.12× 107</td><td>75.5±46.8</td><td></td><td>10.1 ± 10.3</td><td>7.44±7.3</td></tr><tr><td>F11</td><td>17.8 ± 9.52</td><td>1.54 ± 0.0587</td><td>6080 ± 1.03 × 104</td><td>6.22 × 1010 ± 3.39 × 1010</td><td>132 ± 48.8</td><td>5.02 × 106 ± 2.47 × 107</td><td>1.62 ± 0.533</td><td>7.09 × 1012 ± 7.76× 1012</td><td>16.5 ±9.56</td><td>152±337 10.8 ± 9.46</td><td>1.92 ± 0.722</td><td>1.61 ± 0.565</td></tr></table>

Schaffer, non-continuous Rastrigin, and Levy structures. F6 to F8 are hybrid functions composed of transformed subcomponents, while F9-F11 are composition functions with heterogeneous local regions [14]. Thus, the transition from F1 and F2 to the remaining classes shifts the principal challenge from locating a single favourable direction to maintaining search diversity while discriminating among multiple basins.

The exact convergence of Myco on F1 at both dimensions suggests that its spatial graph construction, anchor retention, and flow-driven tip motion are sufficient to reach the single-basin optimum within the present evaluation budgets. Its performance on F2 at � = 20 further indicates that the interaction network provides each tip with a locally weighted displacement field, while anchors preserve previously favourable locations. Together, these mechanisms form a local structural memory that can maintain multiple directional signals rather than forcing the population to collapse prematurely toward a single global reference.

The basic multimodal functions place greater pressure on cross-basin exploration and late-stage coordinate refinement. Their local minima, discontinuities, and mixed components require sustained movement among competing basins. The DE variants often perform well on these functions because their mutation operators use pairwise population differences and adapt control parameters based on successful runs. This mechanism can generate broad exploratory displacements before concentrating the search around promising regions. Among these variants, L-SHADE further reduces the population size, which can shift the search toward exploitation as the run progresses [28]. In contrast, Myco retains a local interaction graph. When this graph separates into spatially distinct groups, lower inter-community interaction weights reduce the influence of distant regions. This can preserve local organisation, but it may also delay the transfer of information needed to escape deceptive basins or to cross discontinuous boundaries.

The hybrid and composition functions test whether search information remains useful when several component landscapes compete across the domain. Myco can maintain competitive errors as long as the local flow field continues to identify a productive search direction. However, its performance weakens when relevant information is distributed across disconnected or competing regions of the landscape. The results suggest a trade-off in the current implementation: community-weighted local flow supports organised exploration, but it may also restrict rapid redistribution of search effort when the landscape begins to favour a distant component.

Figure 1 illustrates the temporal pattern of this distinction. While Figure 2 provides a median-based summary that is less sensitive to extreme runs, Figure 3 isolates the change in mean final error, showing that increasing the dimension alters the final-error profile of Myco in a consistent direction.

## 4.3. Ablation Study

We examine the respective contributions of adaptive cord conductance and the community-detection backend (i.e., Louvain and Greedy Modularity) within two ablation variants. Myco (Louvain) preserves the Louvain partition and all remaining components of the Plasticity, but disables the conductance update on tip-to-tip edges. Myco (Greedy) instead replaces the Louvain backend with Greedy Modularity optimisation. These comparisons isolate distinct aspects of the search graph: the former evaluates whether edge strengths should adapt to the induced flow, whereas the latter examines whether the resulting flow field benefits from a non-trivial mesoscale partition.

## 4.3.1. Cord Plasticity

Table 5 compares the Louvain and Plasticity variants using paired mean and standard-deviation values. The logarithmic ratio quantifies both the direction and magnitude of each paired difference, with negative values indicating improved performance under Plasticity.

At � = 10, Plasticity achieves its largest reductions on F4 and F6, with corresponding decreases in standard deviation. In the Louvain backend, existing edges are weighted according to distance, conductance, potential difference, and community membership, whereas tip-to-tip conductances remain fixed throughout the search. By contrast, Plasticity changes this local graph after the flow field has been computed. Such changes reinforce tip-to-tip edges that are aligned with the current flow while attenuating unsupported edges. The observed reduction in both location and dispersion suggests that adaptive conductance suppresses transient local directions that would otherwise retain excessive influence.

The effect becomes pronounced at � = 20. Plasticity reduces the mean error on F1, F3, F4, F5, F7, F8, F9, and F11. The paired reductions on F4 and F7 are associated with substantially lower standard deviations. This pattern suggests that the conductance update helps preserve locally persistent search directions when distance-based neighbourhoods become less informative in higher-dimensional landscapes. However, the benefit is not universal.

![](images/db9734625ac61e2c18cd3d26d2d59a710d0155f75f3a72d4a3ffe525b4e4b486.jpg)  
Figure 1: Mean best-so-far error trajectories over 30 runs for representative functions from the unimodal, basic multimodal, hybrid, and composition classes at � = 10 and � = 20. Shaded regions denote one standard deviation.

![](images/74a4cd42dc228bcdfc5e29c1709f23640e32e72ba23a0d3258fcbcfeb318722a.jpg)  
Figure 2: Average rank of each algorithm over F1 to F11 at � = 10 and � = 20, computed from median final error. Lower ranks indicate lower median error.

Louvain remains preferable on F2, F6, and F10, indicating that conductance decay can, in some landscapes, weaken alternative local directions before their search value becomes evident.

Figure 4 shows the paired final-error outcomes, whereas Figure 5 demonstrates the corresponding run-to-run variability. Table 5 complements these visual summaries by providing the exact relative changes that distinguish modest paired differences from order-of-magnitude reductions.

Table 5  
![](images/24cae85cb7b9e198c85b3bb7db6c6e1c48609001914498a3d767be94620e9879.jpg)  
Figure 3: Change in the mean final error of Myco from � = 10 to $D = 2 0$ over F1 to F11, calculated as log $_ { 1 0 } ( \mu _ { D 2 0 } / \mu _ { D 1 0 } )$ Positive values indicate an increase in mean error at $D = 2 0$ . Functions with zero mean error in both dimensions are assigned a value of zero.

Effect of cord plasticity relative to the Louvain baseline across 30 independent runs. Values are reported as mean ± standard deviation. The final columns provide the relative effect size, computed as lo $\mathrm { \Omega _ { \mathrm { 1 0 } } } ( \mu _ { \mathrm { { P l a s t i c i t y } } } / \mu _ { \mathrm { { L o u v a i n } } } )$ , where negative values indicate lower mean error under Plasticity and positive values indicate lower mean error under Louvain. Empty entries report cases where one or both means are zero.
<table><tr><td rowspan="2">Function</td><td colspan="4"> $\overline { { { \cal D } = 1 0 } }$ </td><td colspan="4"> $\overline { { \mathbf { D } = 2 \mathbf { 0 } } }$ </td></tr><tr><td>Louvain</td><td>Plasticity</td><td></td><td> $\overline { { \mathrm { l o g } _ { 1 0 } \left( \frac { \mu _ { \mathrm { P l a s t i c i t y } } } { \mu _ { \mathrm { L o u v a i n } } } \right) } }$ </td><td>Louvain</td><td>Plasticity</td><td></td><td> $\underline { { \mathrm { l o g } _ { 1 0 } } } \left( \frac { \mu _ { \mathrm { P l a s t i c i t y } } } { \mu _ { \mathrm { L o u v a i n } } } \right)$ </td></tr><tr><td>F1</td><td> $0 \pm 0$ </td><td> $0 \pm 0$ </td><td></td><td></td><td> $\overline { { 4 . 9 5 \times 1 0 ^ { - 6 } \pm 1 . 4 7 \times 1 0 ^ { - 5 } } }$ </td><td> $0 \pm 0$ </td><td></td><td></td></tr><tr><td>F2</td><td> ${ 5 . 6 7 \pm 2 . 5 7 }$ </td><td> $6 . 1 4 \pm 2 . 4 3$ </td><td></td><td>0.034</td><td> $2 5 . 6 \pm 2 3 . 7$ </td><td> $2 9 . 9 \pm 2 2 . 7$ </td><td></td><td>0.066</td></tr><tr><td>F3</td><td> $3 . 5 2 \pm 0 . 3 8 2$ </td><td> $3 . 6 1 \pm 0 . 4 1 8$ </td><td></td><td>0.011</td><td> $8 . 2 3 \pm 0 . 5 0 0$ </td><td> $8 . 1 1 \pm 0 . 4 2 1$ </td><td></td><td>-0.007</td></tr><tr><td>F4</td><td> $7 2 7 \pm 1 3 4 1$ </td><td> $1 7 8 \pm 7 3 7$ </td><td></td><td>-0.610</td><td> $6 8 4 8 \pm 5 1 2 7$ </td><td> $3 7 7 5 \pm 3 9 3 5$ </td><td></td><td>-0.259</td></tr><tr><td>F5</td><td> $5 5 1 \pm 7 0 9$ </td><td> $5 2 4 \pm 6 1 7$ </td><td></td><td>-0.021</td><td> $3 2 9 1 \pm 1 6 9 8$ </td><td> $2 6 5 4 \pm 1 4 2 9$ </td><td></td><td>-0.093</td></tr><tr><td>F6</td><td> $8 6 5 \pm 2 3 0 5$ </td><td> $9 2 . 0 \pm 1 4 3 $ </td><td></td><td>-0.973</td><td> $9 . 1 2 \times 1 0 ^ { 4 } \pm 3 . 9 4 \times 1 0 ^ { 4 }$ </td><td> $9 . 5 3 \times 1 0 ^ { 4 } \pm 4 . 6 6 \times 1 0 ^ { 4 }$ </td><td></td><td>0.019</td></tr><tr><td>F7</td><td> $3 1 1 0 \pm 5 5 0 7$ </td><td> $1 6 9 1 \pm 4 4 1 1$ </td><td></td><td>-0.265</td><td> $3 6 6 3 \pm 4 3 3 5$ </td><td> $2 1 4 4 \pm 7 6 8$ </td><td></td><td>-0.233</td></tr><tr><td>F8</td><td> $2 0 . 1 0 \pm 0 . 1 6 6$ </td><td> $2 0 . 0 9 \pm 0 . 2 3 3$ </td><td></td><td>-0.000</td><td> $2 0 . 2 1 \pm 0 . 3 9 8$ </td><td> $2 0 . 1 8 \pm 0 . 4 1 2$ </td><td></td><td>-0.001</td></tr><tr><td>F9</td><td> $3 . 1 7 \pm 2 . 9 9$ </td><td> $3 . 9 7 \pm 2 . 9 0$ </td><td></td><td>0.097</td><td> $7 . 7 9 \pm 4 . 4 1$ </td><td> $6 . 9 8 \pm 3 . 8 5$ </td><td></td><td>-0.048</td></tr><tr><td>F10</td><td> $1 6 . 3 \pm 2 1 . 7$ </td><td> $1 9 . 8 \pm 1 9 . 3$ </td><td></td><td>0.083</td><td> $4 5 . 4 \pm 2 3 . 7$ </td><td> $5 7 . 6 \pm 4 1 . 9$ </td><td></td><td>0.103</td></tr><tr><td>F11</td><td> $4 . 6 8 \pm 3 . 3 7$ </td><td> $4 . 6 9 \pm 1 . 7 8$ </td><td></td><td>0.001</td><td> $3 0 . 4 \pm 1 4 . 5$ </td><td> $1 7 . 8 \pm 9 . 5 2$ </td><td></td><td>-0.233</td></tr></table>

## 4.3.2. Community Backend Sensitivity

Table 6 and Figure 6 report the sensitivity of the flow field to the choice of community detection backend. Across all reported functions and dimensions, the Greedy Modularity identifies a single raw community and a single retained community. Consequently, the distinction between intra-community and inter-community interactions is lost, and all present edges receive the same community factor.

The Greedy results contain extreme mean-to-median divergence for selected functions. At $D = 1 0 .$ , this behaviour is most evident on F1, F9, F10, and F11, whereas at $D = 2 0$ it is most pronounced on F6, F7, F10, and F11. The low medians observed in several cases indicate that the one-community configuration can occasionally reach favourable regions of the search space. However, the substantially larger means suggest that a subset of runs enters poor regions from which recovery is not achieved within the available evaluation budget.

Louvain restores a partitioned flow field in which cross-community messages are weighted less strongly than within-community messages. Plasticity subsequently adjusts the relative strength of the remaining tip-to-tip interactions. These mechanisms therefore operate at distinct graph scales: Louvain regulates communication among local groups, whereas Plasticity controls the persistence of search directions within the current local graph.

![](images/a594dbb8e6d2eb1d92e679fb7183d30e03749f1bc0541c4fae526aa46ecc89be.jpg)  
Figure 4: Mean final errors of Myco (Louvain) and Myco (Plasticity) over F1 to F11 at � = 10 and � = 20. Lower values are better. The horizontal axis is logarithmic.

![](images/7a614c16c799e8956aec6a664a257b53fbabaf894f6488168a05cb587fc91ed1.jpg)  
Figure 5: Standard deviations of final error for Myco (Louvain) and Myco (Plasticity) over 30 independent runs. Lower values indicate lower run-to-run variability. The horizontal axis is logarithmic.

## 5. Conclusion

In this study, we introduced Mycelial Search (Myco) to frame continuous optimisation as a search over an evolving local interaction graph. Candidate solutions are represented as active tips, while a bounded set of anchors preserves historically favourable positions. The graph itself is reconstructed at each iteration. We used the Louvain community detection algorithm to control the strength of information exchange within and across communities. We introduced cord plasticity, a rule that reinforces tip-to-tip connections aligned with the local flow and attenuates unsupported connections.

We evaluated Myco on the CEC 2022 single-objective bound-constrained benchmark suite, compared it with the established metaheuristic methods, and examined the roles of cord plasticity and the community detection backend in an ablation study. The ablation study results demonstrated that these mechanisms control different aspects of the search. Cord plasticity operates at the edge level, reinforcing or decaying individual tip-to-tip connections based on their alignment with the local flow. The community detection backend operates at the partition level, determining the range over which information is exchanged.

Table 6  
Greedy Modularity diagnostic for functions with large divergence between the mean and median final errors. Each entry reports 30 independent runs. The ratio $\mu / \tilde { x }$ indicates the extent to which rare high-error runs influence the mean.
<table><tr><td>Dimension</td><td>Function</td><td>Mean (µ)</td><td>Median (x)</td><td> $\mu / \tilde { x }$ </td></tr><tr><td> $\overline { { D = 1 0 } }$ </td><td>F1</td><td>2217</td><td> $\overline { { 3 . 1 5 \times 1 0 ^ { - 2 } } }$ </td><td> $\overline { { 7 . 0 5 \times 1 0 ^ { 4 } } }$ </td></tr><tr><td> $D = 1 0$ </td><td>F7</td><td> $2 . 9 9 \times 1 0 ^ { 5 }$ </td><td> $6 2 3 9 . 5$ </td><td> $4 8 . 0$ </td></tr><tr><td> $D = 1 0$ </td><td>F9</td><td> $5 . 3 1 \times 1 0 ^ { 6 }$ </td><td> $6 . 0 7$ </td><td> $8 . 7 6 \times 1 0 ^ { 5 }$ </td></tr><tr><td> $D = 1 0$ </td><td>F10</td><td> $6 . 4 4 \times 1 0 ^ { 6 }$ </td><td> $1 1 4 6 . 8$ </td><td> $5 . 6 2 \times 1 0 ^ { 3 }$ </td></tr><tr><td> $D = 1 0$ </td><td>F11</td><td> $2 . 0 9 \times 1 0 ^ { 1 0 }$ </td><td> $4 . 1 6$ </td><td> $5 . 0 1 \times 1 0 ^ { 9 }$ </td></tr><tr><td> $D = 2 0$ </td><td>F6</td><td> $7 . 8 1 \times 1 0 ^ { 6 }$ </td><td> $1 . 0 3 \times 1 0 ^ { 5 }$ </td><td>75.9</td></tr><tr><td> $D = 2 0$ </td><td>F7</td><td> $6 . 8 3 \times 1 0 ^ { 6 }$ </td><td> $1 . 8 3 \times 1 0 ^ { 5 }$ </td><td>37.4</td></tr><tr><td> $D = 2 0$ </td><td>F10</td><td> $3 . 6 5 \times 1 0 ^ { 7 }$ </td><td> $3 . 7 0 \times 1 0 ^ { 6 }$ </td><td>9.86</td></tr><tr><td> $D = 2 0$ </td><td>F11</td><td> $7 . 7 2 \times 1 0 ^ { 1 3 }$ </td><td> $4 8 . 5$ </td><td> $1 . 5 9 \times 1 0 ^ { 1 2 }$ </td></tr></table>

![](images/b26a993e169cce919c053401e893a7d4d16f5ab3f92ab5ddde314058369056ce.jpg)  
Figure 6: Mean final errors of the Louvain, Greedy Modularity, and Plasticity implementations over F1 to F11 at $D = 1 0$ and $D = 2 0$ . Lower values are better. The horizontal axis is logarithmic.

Myco makes the interaction structure among candidate solutions explicit and subject to analysis during the search. The local graph is not only a mechanism for selecting neighbours. Its community partition and tip-to-tip edge conductances form part of the search state and affect how information is passed between candidate solutions. This formulation makes local organisation a component of continuous optimisation that can be defined, examined, and modified.

## Acknowledgments

The author declares no competing interests.

## References

[1] Adamatzky, A., Ayres, P., Beasley, A.E., Chiolerio, A., Dehshibi, M.M., Gandia, A., Albergati, E., Mayne, R., Nikolaidou, A., Roberts, N., Tegelaar, M., Tsompanas, M.A., Phillips, N., Wösten, H.A., 2022. Fungal electronics. Biosystems 212, 104588. URL: https: //doi.org/10.1016/j.biosystems.2021.104588, doi:10.1016/j.biosystems.2021.104588.

[2] Adamatzky, A., Nikolaidou, A., Gandia, A., Chiolerio, A., Dehshibi, M.M., 2023. Reactive Fungal Wearable. Springer Nature Switzerland. chapter 8. pp. 93–104. URL: https://doi.org/10.1007/978-3-031-38336-6\_8, doi:10.1007/978-3-031-38336-6\_8.

[3] Awad, A., Coghill, G.M., Pang, W., 2023. A novel physarum-inspired competition algorithm for discrete multi-objective optimisation problems. Soft Computing 27, 14699–14719. URL: https://doi.org/10.1007/s00500-023-08505-1, doi:10.1007/ s00500-023-08505-1.

[4] Blondel, V.D., Guillaume, J.L., Lambiotte, R., Lefebvre, E., 2008. Fast unfolding of communities in large networks. Journal of Statistical Mechanics: Theory and Experiment 2008, P10008. URL: https://doi.org/10.1088/1742-5468/2008/10/P10008, doi:10.1088/1742-5468/2008/10/P10008.

[5] Brest, J., Maucec, M.S., Boškoviˇ c, B., 2017. Single objective real-parameter optimization: Algorithm jso, in: 2017 IEEE Congress on´ Evolutionary Computation (CEC), IEEE. pp. 1311–1318. URL: https://doi.org/10.1109/CEC.2017.7969456, doi:10.1109/ CEC.2017.7969456.

[6] Dehshibi, M.M., Adamatzky, A., 2021. Electrical activity of fungi: Spikes detection and complexity analysis. Biosystems 203, 104373. URL: https://doi.org/10.1016/j.biosystems.2021.104373, doi:10.1016/j.biosystems.2021.104373.

[7] Dehshibi, M.M., Adamatzky, A., 2023. Complexity of Electrical Spiking of Fungi. Springer Nature Switzerland. chapter 4. pp. 33–60. URL: https://doi.org/10.1007/978-3-031-38336-6\_4, doi:10.1007/978-3-031-38336-6\_4.

[8] Dehshibi, M.M., Chiolerio, A., Nikolaidou, A., Mayne, R., Gandia, A., Ashtari-Majlan, M., Adamatzky, A., 2021. Stimulating Fungi Pleurotus ostreatus with Hydrocortisone. ACS Biomaterials Science & Engineering 7, 3718–3726. URL: https://doi.org/10.1021 acsbiomaterials.1c00752, doi:10.1021/acsbiomaterials.1c00752.

[9] Dehshibi, M.M., Chiolerio, A., Nikolaidou, A., Mayne, R., Gandia, A., Ashtari-Majlan, M., Adamatzky, A., 2023. On Stimulating Fungi Pleurotus Ostreatus with Hydrocortisone. Springer Nature Switzerland. chapter 9. pp. 105–121. URL: https://doi.org/10.1007/ 978-3-031-38336-6\_9, doi:10.1007/978-3-031-38336-6\_9.

[10] Dehshibi, M.M., Sourizaei, M., Fazlali, M., Talaee, O., Samadyar, H., Shanbehzadeh, J., 2017. A hybrid bio-inspired learning algorithm for image segmentation using multilevel thresholding. Multimedia Tools and Applications 76, 15951–15986. URL: https://doi.org/10. 1007/s11042-016-3891-3, doi:10.1007/s11042-016-3891-3.

[11] Gao, C., Zhang, X., Yue, Z., Wei, D., 2020. An Accelerated Physarum Solver for Network Optimization. IEEE Transactions on Cybernetics 50, 765–776. URL: https://doi.org/10.1109/TCYB.2018.2872808, doi:10.1109/TCYB.2018.2872808.

[12] Karaboga, D., Basturk, B., 2007. A powerful and efficient algorithm for numerical function optimization: artificial bee colony (ABC) algorithm. Journal of Global Optimization 39, 459–471. URL: https://doi.org/10.1007/s10898-007-9149-x, doi:10. 1007/s10898-007-9149-x.

[13] Kennedy, J., Eberhart, R., 1995. Particle swarm optimization, in: Proceedings of ICNN’95 - International Conference on Neural Networks, IEEE. pp. 1942–1948. URL: https://doi.org/10.1109/ICNN.1995.488968, doi:10.1109/ICNN.1995.488968.

[14] Kumar, A., Price, K.V., Mohamed, A.W., Hadi, A.A., Suganthan, P.N., 2022. Problem Definitions and Evaluation Criteria for the CEC 2022 Special Session and Competition on Single Objective Bound Constrained Numerical Optimization. Technical Report. Nanyang Technological University. URL: https://github.com/P-N-Suganthan/2022-SO-BO/blob/main/CEC2022%20TR.pdf.

[15] Li, S., Chen, H., Wang, M., Heidari, A.A., Mirjalili, S., 2020. Slime mould algorithm: A new method for stochastic optimization. Future Generation Computer Systems 111, 300–323. URL: https://doi.org/10.1016/j.future.2020.03.055, doi:10.1016/j. future.2020.03.055.

[16] Liang, J., Qin, A., Suganthan, P., Baskar, S., 2006. Comprehensive learning particle swarm optimizer for solving multiobjective optimization problems. International Journal of Intelligent Systems 21, 209–226. URL: https://doi.org/10.1002/int.20128, doi:10.1002/int.20128.

[17] Ma, H., Shen, S., Yu, M., Yang, Z., Fei, M., Zhou, H., 2019. Multi-population techniques in nature inspired optimization algorithms: A comprehensive survey. Swarm and Evolutionary Computation 44, 365–387. URL: https://doi.org/10.1016/j.swevo.2018. 04.011, doi:10.1016/j.swevo.2018.04.011.

[18] Ma, Z., Wu, G., Suganthan, P.N., Song, A., Luo, Q., 2023. Performance assessment and exhaustive listing of 500+ nature-inspired metaheuristic algorithms. Swarm and Evolutionary Computation 77, 101248. URL: https://doi.org/10.1016/j.swevo.2023. 101248, doi:10.1016/ .swevo.2023.101248.

[19] Machado, M.C., Bellemare, M.G., Talvitie, E., Veness, J., Hausknecht, M., Bowling, M., 2018. Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents. Journal of Artificial Intelligence Research 61, 523–562. URL: https: //doi.org/10.1613/jair.5699, doi:10.1613/jair.5699.

[20] Martinelli, D., de Oliveira, A.S., Kalempa, V.C., 2025. Bioinspired algorithm based on physarum polycephalum for the formation of decentralized mesh networks in multi-robot systems. Scientific Reports 16, 3457. URL: https://doi.org/10.1038/ s41598-025-33456-y, doi:10.1038/s41598-025-33456-y.

[21] Mirjalili, S., Lewis, A., 2016. The Whale Optimization Algorithm. Advances in Engineering Software 95, 51–67. URL: https: //doi.org/10.1016/j.advengsoft.2016.01.008, doi:10.1016/j.advengsoft.2016.01.008.

[22] Mirjalili, S., Mirjalili, S.M., Lewis, A., 2014. Grey Wolf Optimizer. Advances in Engineering Software 69, 46–61. URL: https: //doi.org/10.1016/j.advengsoft.2013.12.007, doi:j.advengsoft.2013.12.007.

[23] Nanda, S.J., Panda, G., 2014. A survey on nature inspired metaheuristic algorithms for partitional clustering. Swarm and Evolutionary Computation 16, 1–18. URL: https://doi.org/10.1016/j.swevo.2013.11.003, doi:10.1016/j.swevo.2013.11.003.

[24] Sepas-Moghaddam, A., Arabshahi, A., Yazdani, D., Dehshibi, M.M., 2012. A novel hybrid algorithm for optimization in multimodal dynamic environments, in: 2012 12th International Conference on Hybrid Intelligent Systems (HIS), IEEE. pp. 143–148. URL: https: //doi.org/10.1109/HIS.2012.6421324, doi:10.1109/HIS.2012.6421324.

[25] Shabani, A., Asgarian, B., Salido, M., Asil Gharebaghi, S., 2020. Search and rescue optimization algorithm: A new optimization method for solving constrained engineering optimization problems. Expert Systems with Applications 161, 113698. URL: https://doi.org/10. 1016/j.eswa.2020.113698, doi:10.1016/j.eswa.2020.113698.

[26] Storn, R., Price, K., 1997. Differential evolution – A simple and efficient heuristic for global optimization over continuous spaces. Journal of Global Optimization 11, 341–359. URL: https://doi.org/10.1023/A:1008202821328, doi:10.1023/A:1008202821328.

[27] Tanabe, R., Fukunaga, A., 2013. Success-history based parameter adaptation for Differential Evolution, in: 2013 IEEE Congress on Evolutionary Computation, IEEE. pp. 71–78. URL: https://doi.org/10.1109/CEC.2013.6557555, doi:10.1109/CEC.

2013.6557555.

[28] Tanabe, R., Fukunaga, A.S., 2014. Improving the search performance of SHADE using linear population size reduction, in: 2014 IEEE Congress on Evolutionary Computation (CEC), IEEE. pp. 1658–1665. URL: https://doi.org/10.1109/CEC.2014.6900380, doi:10.1109/CEC.2014.6900380.

[29] Teo, J., 2005. Differential evolution with self-adaptive populations, in: Khosla, R., Howlett, R.J., Jain, L.C. (Eds.), Knowledge-Based Intelligent Information and Engineering Systems, Springer Berlin Heidelberg, Berlin, Heidelberg. pp. 1284–1290. URL: https://doi. org/10.1007/11552413\_183, doi:10.1007/11552413\_183.

[30] Turgut, O.E., Turgut, M.S., Kırtepe, E., 2023. A systematic review of the emerging metaheuristic algorithms on solving complex optimization problems. Neural Computing and Applications 35, 14275–14378. URL: https://doi.org/10.1007/s00521-023-08481-5, doi:10.1007/s00521-023-08481-5.

[31] Whitley, D., 1994. A genetic algorithm tutorial. Statistics and Computing 4, 65–85. URL: https://doi.org/10.1007/BF00175354, doi:10.1007/BF00175354.

[32] Wu, G., Mallipeddi, R., Suganthan, P.N., 2019. Ensemble strategies for population-based optimization algorithms – A survey. Swarm and Evolutionary Computation 44, 695–711. URL: https://doi.org/10.1016/j.swevo.2018.08.015, doi:10.1016/j.swevo. 2018.08.015.

[33] Zhang, J., Sanderson, A.C., 2009. JADE: Adaptive Differential Evolution With Optional External Archive. IEEE Transactions on Evolutionary Computation 13, 945–958. URL: https://doi.org/10.1109/TEVC.2009.2014613, doi:10.1109/TEVC.2009.2014613.

[34] Zheng, B., Chen, Y., Wang, C., Heidari, A.A., Liu, L., Chen, H., 2024. The moss growth optimization (mgo): concepts and performance. Journal of Computational Design and Engineering 11, 184–221. URL: https://doi.org/10.1093/jcde/qwae080, doi:10. 1093/jcde/qwae080.