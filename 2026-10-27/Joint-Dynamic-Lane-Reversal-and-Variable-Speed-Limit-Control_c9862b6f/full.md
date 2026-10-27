# Joint Dynamic Lane Reversal and Variable Speed Limit Control: A Graphormer-Based Approach

Hamed Araab

Abbas Ahmadi

Mohammad Hossein Basiri

Department of Industrial Engineering

Department of Industrial Engineering

Department of Electrical Engineering

and Management Systems

and Management Systems

Amirkabir University of Technology

Amirkabir University of Technology

Amirkabir University of Technology

(Tehran Polytechnic)

(Tehran Polytechnic)

Tehran, Iran

(Tehran Polytechnic)

Tehran, Iran

hamedaarab@aut.ac.ir

Tehran, Iran

abbas.ahmadi@aut.ac.ir

basiri@aut.ac.ir

Abstract—Connected and automated vehicles could allow reassigning lane directions and speed limits on every road frequently, but the benefit hinges on how traffic reroutes, and one-way roads can isolate junctions. We formulate network-wide dynamic lane allocation and directional speed limit control with one-way roads, lane clearing before each reversal, and a navigable graph kept strongly connected at all times. An action-resolution algorithm grounded in graph orientation theory admits a proposal closing a direction only if the network stays strongly connected and otherwise relaxes or reverts it, keeping all junctions mutually reachable within the model, even during clearing. All road controllers share one transformer whose attention, Global Endpoint Relation-Aware Spatio-Temporal Self-Attention (GERASTSA), spans all road–time tokens, biased by the endpoint relations between roads. It is pretrained on action-conditioned traffic forecasting and finetuned with per-road proximal policy optimization under a mixed reward that also credits neighboring roads and the network. On a 6 × 6 grid with commuter rush hours, the learned policy makes about two thirds of roads one-way in the peak direction and cuts daily delay by 90% against the initial configuration held fixed and by 85% against a tuned two-way load balancer. Restricted to two-way roads at test time, it gains only about as much as this balancer, whereas its learned speed limits add delay. Pretraining roughly halves the updates needed to surpass the fixed configuration. Of the ablations, only removing the relational bias or the spatial embedding clearly matters, multiplying delay about fivefold.

Index Terms—Dynamic lane reversal, Variable speed limits, Reinforcement learning, Graphormer, Connected and automated vehicles, Network-wide traffic control

## I. INTRODUCTION

Urban traffic demand is directionally unbalanced. In the morning, commuters travel from residential areas toward commercial ones, and in the evening they return, so that the lanes leading in one direction are congested while those leading in the other are lightly used. The lane directions of most roads, however, are fixed. Reversible lanes and contraflow operations relax this restriction by assigning lanes to the direction that needs them [1], [2], and variable speed limits (VSLs) regulate how traffic uses the capacity that a road offers [3]. In practice, both are usually deployed on individual corridors, where reversible lanes typically follow preset schedules and VSLs react to local conditions. Connected and automated vehicles (CAVs) change this picture: they can be informed of a new lane direction or speed limit immediately and, if they comply with it reliably, make it conceivable to reconfigure every road of a network at short intervals in response to current traffic [4], [5].

Reconfiguring a network, rather than a corridor, raises two difficulties. The first is that the effect of a change in capacity depends on how traffic redistributes across the network. Braess’s paradox shows that even adding capacity can lengthen travel times [6], and assigning more lanes to one direction of a road necessarily takes them from the other, so the traffic it sheds can burden the surrounding roads. This motivates deciding lane allocations with the whole network in view and, because speed limits shape how traffic uses the capacity that an allocation provides, deciding them together with the speed limits. The second difficulty is reachability. Assigning all lanes of a road to a single direction turns it into a oneway road, which can concentrate capacity where it is needed most but can also cut junctions off from each other. Robbins theorem [7] settles when every road of a network can be made one-way with all junctions still mutually reachable, but only for an orientation that never changes. We require a dynamic controller to preserve reachability at every time step, including while a reversal is in progress: before traffic may enter a lane in its new direction, the vehicles still traveling in its old direction must leave, and until they have, the lane serves neither direction [1].

Prior work addresses parts of this problem. Dynamic lane reversal has been formulated as an optimization problem over a given or forecast demand, solved with integer and bi-level programs or within dynamic traffic assignment [4], [5], [8], [9]. Reinforcement learning (RL) has been used to allocate lanes without a model of the demand, in experiments that keep at least one lane in each direction of every road [10]. Lane or road-width reallocation has been integrated with flow and speed control, but through model-based optimization on freeways and motorways [11], [12]. To our knowledge, no prior work learns a policy that jointly sets lane allocations and speed limits on every road of a general network while permitting one-way roads and guaranteeing that every junction remains reachable from every other.

We address this gap with a formulation that separates what a controller may propose from what takes effect. Every road has a road controller that proposes, at every decision step, how many of its lanes serve each direction and a speed limit for each direction, bounded above by the nominal speed limit. All proposals pass through action resolution, which accepts those that would close a direction one at a time, in random order, only as long as the network stays strongly connected, and otherwise relaxes them to keep one lane open or, on single-lane roads, reverts them. Together with a per-road restriction that keeps both directions of every bridge road open and an explicit model of lane clearing, this guarantees strong connectivity at every time step, including during the clearing that precedes a reversal. Like a shield in safe RL [13], action resolution thus lets the learned policy explore any admissible action without ever cutting a junction off from the rest of the network.

Because the traffic regulated by the controllers couples the whole network, we realize all road controllers with a single transformer [14] whose tokens are road–time pairs. Its attention mechanism, Global Endpoint Relation-Aware Spatio-Temporal Self-Attention (GERASTSA), attends jointly over space and time and, in the spirit of graph transformers [15], biases attention by the endpoint relations between roads. We pretrain the model on a self-supervised task native to the domain, forecasting the traffic of every road conditioned on the lane allocations and speed limits that take effect in between, and fine-tune it with proximal policy optimization (PPO) [16] applied per road. Each road optimizes a mixed reward that combines its own reward with those of its neighbors and of the whole network, so that the objective of a road also reflects the cost that its becoming one-way imposes on the roads that absorb its traffic.

Our contributions are as follows.

• We formulate network-wide dynamic lane allocation and directional speed limit control with one-way roads, explicit lane clearing, and a strong-connectivity requirement on the navigable graph (Section III).

• We give an action-resolution algorithm, grounded in graph orientation theory, that enforces this requirement at every time step while leaving per-road admissibility to the action space, and show that it does so (Section III-E, Proposition 1).

• We propose the GERASTSA transformer, a single model for all road controllers whose attention is biased by the endpoint relations between roads, including the directions in which adjacent roads meet, and whose embeddings encode the junction degrees of every road and the position of every time step within the window (Section IV).

• We train it by pretraining on action-conditioned traffic forecasting and fine-tuning with per-road PPO under a mixed reward (Section V), and evaluate the learned policy on a grid network with commuter rush hours against a baseline without a policy and a heuristic that keeps every road two-way, against fine-tuning without pretraining, against restrictions of its actions at test time, and against ablations of the components of the GERASTSA transformer (Section VI).

The remainder of the paper is organized as follows. Section II reviews related work on lane reversal, VSLs, integrated and learning-based traffic control, and spatio-temporal and graph attention. Section III formulates the control problem and action resolution, Section IV presents the GERASTSA transformer, and Section V describes the environment, pretraining, and fine-tuning. Section VI reports the results, Section VII discusses them, and Section VIII concludes.

## II. PRIOR WORK

We review work on lane reversal and VSLs, on their integration with other controls, on learning-based coordination of traffic controllers, and on attention models for spatio-temporal and graph-structured data, and then position our contribution.

## A. Lane Reversal and Contraflow

Reversible lanes and contraflow operations match road capacity to directionally unbalanced demand, both for recurring commuter peaks and for emergency evacuation [1], [2]. In practice, they follow fixed schedules or are activated for planned events [1]. At the network level, the choice of which roads to reverse has been formulated as an optimization problem, chiefly for evacuation, where the configuration is held for the duration of the event [17].

CAVs have motivated dynamic lane reversal, in which lane directions follow the traffic as it evolves. [4] compute lane configurations that maximize network flow with integer linear and bi-level programs, and examine how directional imbalance and the granularity of reallocation affect the benefit. [8] extend a system-optimal dynamic traffic assignment model based on the cell transmission model (CTM) so that it also determines lane directions at fine space–time resolution. [5] develop a CTM for dynamic lane reversal with autonomous vehicles and derive a heuristic from an analysis of a single link, which they apply to city networks. [9] optimize lane reversal together with the routes and schedules of individual vehicles as an integer program, solved with a distributed algorithm based on the alternating direction method of multipliers. For freeways, [18] model reversible lanes macroscopically and operate them with a logic-based controller and with model predictive control (MPC). [19] choose the locations, operating times, and installation modes of reversible lanes with a minimumcost maximum-flow formulation that accounts for risk and economic costs.

These approaches optimize against a demand that is given or forecast in advance. [10] instead learn lane allocations with RL: local agents propose lane-direction changes, which are coordinated through a graph of dependencies between roads derived from vehicle paths, and connected vehicles are rerouted in response to the new configuration. In their experiments, every road keeps at least one lane in each direction. We likewise learn a policy that maps the observed state of the network to lane allocations, but we allow roads to become one-way whenever the network remains strongly connected, and we model the clearing of lanes explicitly, so that a reversal takes effect only once traffic allows.

Allowing roads to become one-way raises the question of reachability between junctions. Robbins’ theorem [7] characterizes the graphs that admit a strongly connected orientation; it extends to mixed multigraphs [20], and such orientations can be computed in linear time [21]. These results concern a single, static orientation. We use them to guarantee that an admissible initial configuration exists and combine them with action resolution so that strong connectivity is preserved at every time step while lane allocations change.

## B. Variable Speed Limits

VSLs lower the posted speed in response to traffic conditions to delay the onset of congestion, dampen shock waves, and improve safety [3]. [22] found from field data that advisory variable speed signs barely changed capacity but stabilized traffic, and derived a control policy that postpones congestion on a freeway section. MPC coordinates speed limits with ramp metering through a macroscopic traffic model [23], whereas the SPECIALIST algorithm derives speed-limit interventions from shock-wave theory to resolve traffic waves [24]; its field test on a Dutch freeway confirmed its effect on shock waves but also showed that it was less effective for other types of congestion [25].

RL avoids the need for an explicit traffic model in the controller, although the policy is usually trained in simulation. [26] formulate dynamic speed limit control on a network with stochastic demand and supply as a Markov decision process, and evaluate it on the Sioux Falls network in terms of throughput, delay, and emissions. On freeways, [27] learn speed limits with Q-learning to relieve congestion at a recurrent bottleneck, [28] learn lane-specific speed limits with a deep actor-critic algorithm in a setting with CAVs, and [29] represent the freeway as a graph of road topology, occupancy, and speed to assign speed limits per lane and location. We set speed limits on every road and direction of an urban network and decide them jointly with lane allocations.

## C. Integrated Control

Lane reversal has also been integrated with other traffic controls. [11] combine dynamic lane reversal with trafficflow control on a motorway network: a multilane CTM and a mixed-integer linear program, solved within MPC, determine the lane directions and the control actions communicated to autonomous vehicles. For lane-free traffic of CAVs on bidirectional freeways, [12] integrate internal boundary control, which shifts the road width assigned to each direction, with ramp metering and VSLs, and minimize the total time spent with a quadratic program over an extended CTM. These studies indicate that, on freeways and motorways, reallocating directional capacity and regulating flow can complement each other. They rely on optimization over a macroscopic traffic model and consider freeway or motorway settings with continuous width or bidirectional roads, whereas we consider a general road network with integer lane counts, one-way roads, and an explicit requirement on directed reachability.

## D. Learning-Based Network Traffic Control

Network-wide traffic control with RL has been studied most extensively for traffic signals. Every intersection typically acts as an agent, and the agents must cooperate because their decisions affect each other’s traffic. [30] let agents share information with their neighbors and discount the rewards of distant agents, [31] let each intersection aggregate the states of its neighbors with graph attention, and [32] share a single graphconvolutional policy across all intersections, which transfers without retraining to road networks and traffic conditions not seen in training, including the 3,971 signals of Manhattan. Like the last, our model shares its parameters across all roads and is not tied to the size of the network (Section IV), and our mixed reward adapts the reward discounting of [30] (Section V-C1).

More broadly, independent and multi-agent variants of PPO, in which agents with shared parameters optimize their own clipped objectives, perform strongly on cooperative benchmarks [33], [34]. We adopt this per-agent treatment, but whereas the policy of each agent acts on a local observation in these methods, ours observes the whole network through a single transformer, as does our critic. Invalid action masking removes inadmissible discrete actions from the policy [35], and shielding replaces unsafe actions with safe ones before they reach the environment [13]; we use the former for per-road admissibility and the latter, in the form of action resolution, for the network-wide connectivity requirement.

Finally, self-supervised objectives can make RL more dataefficient, either as auxiliary tasks trained alongside the policy [36] or as a pretraining stage that learns representations, including models of the environment dynamics, before finetuning [37]. Our pretraining follows the latter approach with action-conditioned traffic forecasting, a task native to the traffic domain (Section V-B).

## E. Spatio-Temporal and Graph Attention

Traffic forecasting models capture spatial dependencies on the road network together with temporal dynamics, originally by combining graph convolutions with recurrent or convolutional sequence models [38], [39]. Attention-based models have since become prominent. GMAN stacks blocks of spatial and temporal attention in an encoder–decoder architecture [40], and PDFormer applies spatial self-attention with graph masks for short- and long-range dependencies and accounts for the delay with which traffic propagates between locations [41]. Unlike these models, which attend over space and time in separate steps, our model attends jointly over all road–time pairs of a window (Section IV).

The transformer [14] is permutation-equivariant and therefore needs structural information to be injected. Relative position representations add learned terms for clipped relative distances to the attention computation [42], and Graphormer adds a learnable bias indexed by the shortest-path distance between two nodes to the attention scores and a degree-based centrality encoding to the node features [15]. Our attention follows this design on the controller graph but refines the relations between adjacent roads by the directions in which both flow into their shared junction, and our spatial embedding refines the degree encoding by direction, since both distinctions bear on how traffic passes from one road to the next. The choice of graph matters: intersection adjacency [31], shared vehicle paths [10], and adjacency between controlled roads encode different relationships. We use the last, which matches our decision variables, and keep it separate from the navigable graph, which describes the movements that remain physically available.

## F. Positioning

In summary, prior work has controlled lane directions dynamically, learned lane allocations with RL, and integrated lane or road-width reallocation with flow and speed control through model-based optimization. As noted in Section I, however, to our knowledge none of it combines these elements in a learned policy for general networks with one-way roads and guaranteed strong connectivity. Our approach does so through action resolution grounded in graph orientation theory and a single Graphormer-style transformer for all road controllers.

## III. PROBLEM FORMULATION

## A. Road Graphs

Robbins’ theorem [7] states that a connected undirected graph admits a strongly connected orientation, that is, a direction for every edge such that every vertex can reach every other vertex, if and only if the graph has no bridge, an edge whose removal disconnects it. Operating every road as a oneway road is such an orientation, so, with reachability judged between junctions, the theorem characterizes exactly when a road network can run entirely on one-way roads with every junction still reachable from every other. Roads, however, consist of lanes, and a road with two or more lanes can serve both directions at once. We build on this observation: the lanes of each road may be split between its two directions, and roads that would otherwise violate the conditions of the theorem are required to have enough lanes for such a split.

We model the road infrastructure as the infrastructure graph G, a simple undirected graph whose vertices are a finite set of junctions J and whose edges are roads R, each road being an unordered pair of distinct junctions. The network has at least one road, and each road $r \in \mathcal { R }$ has $L _ { r } \in \mathbb { Z } _ { \geq 1 }$ total lanes. Fig. 1 (a) shows an example.

$$
G = ( { \mathcal { I } } , { \mathcal { R } } ) ; \quad { \mathcal { R } } \subseteq { \binom { { \mathcal { I } } } { 2 } } ; \quad | { \mathcal { R } } | \geq 1 ;\tag{1}
$$

A cut is a nonempty proper subset of junctions, and ${ \mathcal { C } } =$ $2 ^ { \mathcal { I } } \setminus \{ \emptyset , \mathcal { I } \}$ is the set of all cuts. The cut-set $B ( \mathcal { X } )$ of a cut X consists of the roads with exactly one endpoint in X.

$$
\begin{array} { r } { \mathcal { B } : 2 ^ { \mathcal { I } }  2 ^ { \mathcal { R } } ; \quad \mathcal { B } ( \mathcal { X } ) = \{ r \in \mathcal { R } \mid | r \cap \mathcal { X } | = 1 \} ; } \end{array}\tag{2}
$$

Requiring every cut-set to be nonempty is equivalent to requiring G to be connected.

$$
B ( \mathcal { X } ) \neq \emptyset \ \forall \mathcal { X } \in \mathcal { C } ;\tag{3}
$$

A bridge road is a road that forms the entire cut-set of some cut, and $\mathcal { R } _ {  }$ is the set of bridge roads.

$$
\mathcal { R } _ {  } = \{ r \in \mathcal { R } \mid \exists \mathcal { X } \in \mathcal { C } : B ( \mathcal { X } ) = \{ r \} \} ;\tag{4}
$$

In line with Robbins’ theorem, we assume that every bridge road has at least two lanes so that it can carry traffic in both directions at once, which a single-lane road cannot do.

$$
L _ { r } \geq 2 \ \forall r \in \mathcal { R } _ {  } ;\tag{5}
$$

Traffic on a road flows in one of two directions, $\mathcal { D } =$ {fwd, bwd}. We fix a reference ordering of the endpoints of every road, marked by the superscripts − and +: traveling from its first endpoint $j _ { r } ^ { - }$ to its second endpoint ${ j } _ { r } ^ { + }$ is forward, and the opposite is backward. The orientation map ω sends a road and a direction to the corresponding arc, an ordered pair of junctions. Throughout, an arc is an oriented road, and ω applied to a set denotes the set of images.

$$
r = \{ j _ { r } ^ { - } , j _ { r } ^ { + } \} \ \mathrm { w i t h \ f i x e d \ o r d e r i n g \ } ( j _ { r } ^ { - } , j _ { r } ^ { + } ) \ \forall r \in \mathcal { R } ;
$$

$$
\omega : \mathcal { R } \times \mathcal { D } \to \mathcal { I } ^ { 2 } ; \quad \omega ( r , d ) = \left\{ ( j _ { r } ^ { - } , j _ { r } ^ { + } ) \quad \mathrm { i f ~ } d = \mathrm { f w d } \atop ( j _ { r } ^ { + } , j _ { r } ^ { - } ) \quad \mathrm { i f ~ } d = \mathrm { b w d }  \right. ;\tag{6}
$$

We consider a horizon of $T \in \mathbb { Z } _ { > 1 }$ steps, with time steps $\breve { \tau } = \{ 0 , \ldots , T \}$ and decision steps ${ \bar { \mathcal { T } } } = { \check { \mathcal { T } } } \setminus \{ T \}$ , the time steps at which the controllers act. A superscript (t) marks the value of a quantity at time step t. At time step $\dot { t } \in \breve { T }$ , road $r \in \mathcal { R }$ has ${ \bar { \ell } } _ { r , d } ^ { ( t ) } \in \mathbf { \dot { \{ 0 , \dots , L _ { r } \} } }$ open lanes usable in direction $d \in \mathcal { D }$ , defined in (10). The arcs with at least one open lane form the navigable arc set $\vec { \mathcal { R } } ^ { ( t ) }$ , which defines the navigable graph $\vec { G } ^ { ( t ) }$ on the junctions (Fig. 1 (b)).

$$
\begin{array} { r l } & { \vec { \mathcal { R } } ^ { ( t ) } = \omega \Big ( \Big \{ ( r , d ) \in \mathcal { R } \times \mathcal { D } \Big | \bar { \ell } _ { r , d } ^ { ( t ) } > 0 \Big \} \Big ) , } \\ & { \vec { G } ^ { ( t ) } = \Big ( \mathcal { I } , \vec { \mathcal { R } } ^ { ( t ) } \Big ) \ \forall t \in \check { \mathcal { T } } ; } \end{array}\tag{7}
$$

We require that drivers be able to travel between any two junctions at every time step, that is, that the navigable graph be strongly connected. An arc set has this property exactly when, for every cut X, it contains an arc from X to its complement $\mathcal { X } ^ { c } = \mathcal { I } \setminus \mathcal { X }$ . Because the complement of a cut is again a cut, the same condition also guarantees an arc into every cut. We collect all strongly connected arc sets in S and require the navigable arc set to belong to $s$ at every time step.

$$
\begin{array} { r l } & { \mathcal { S } = } \\ & { \quad \{ \mathcal { V } \subseteq \omega ( \mathcal { R } \times \mathcal { D } ) | \forall \mathcal { X } \in \mathcal { C } : \mathcal { V } \cap ( \mathcal { X } \times \mathcal { X } ^ { c } ) \neq \emptyset \} ; } \\ & { \vec { \mathcal { R } } ^ { ( t ) } \in \mathcal { S } \forall t \in \check { \mathcal { T } } ; } \end{array}\tag{8}
$$

At t = 0, this strong-connectivity requirement is an assumption on the initial configuration; at later time steps, it must be maintained by action resolution (Section III-E). The assumptions on G in (3) and (5) ensure that the requirement can be met. By Robbins’ theorem, the roads within each connected component of G without its bridge roads, which has no bridges, admit a strongly connected orientation, and opening both directions of every bridge road, which is possible because it has at least two lanes, joins these components into a strongly connected whole. Equivalently, treating every bridge road as a pair of opposite arcs and every other road as an undirected edge yields a mixed graph in which no undirected edge is a bridge, which admits a strongly connected orientation by the extension of Robbins’ theorem to mixed multigraphs [20]; such an orientation, and hence an admissible initial configuration, can be found in linear time [21]. Moreover, S is closed under supersets: opening additional arcs never breaks strong connectivity, a fact we rely on during action resolution.

![](images/b2c3bc0f7e58837e5dc511086994f42f9fc0ebca057fa4578fbd564bd695e996.jpg)  
Fig. 1. An example network with |J| = 7 junctions and $| \mathcal { R } | = 8$ roads. (a) Each stroke is one lane. Road $r _ { 5 }$ (heavy) is the only bridge road, $\mathcal { R } _ {  } = \{ r _ { 5 } \}$ so it has at least two lanes, and strong connectivity requires it to keep at least one open lane in each direction. (b) A strongly connected navigable arc set $\vec { \mathcal { R } } ^ { ( t ) } \in { \mathcal { S } } :$ arrows show the navigable arcs (solid: forward; dashed: backward), with one stroke per open lane $\dot { \bar { \ell } } _ { r , d } ^ { ( t ) } ;$ most roads are one-way, yet $\vec { G } ^ { ( t ) }$ is strongly connected. (c) One road controller per road; controllers of roads that share a junction are neighbors.

## B. Controller Graph

Every road is governed by a road controller that decides the road’s lane allocation and speed limits. Two roads that share a junction exchange traffic directly, so their controllers are neighbors. The resulting coupling structure is the controller graph, the line graph $L ( G )$ of the infrastructure graph: its vertices are the roads, and its edges $\mathcal { E }$ join pairs of roads with a common endpoint (Fig. 1 (c)).

$$
\begin{array} { r l } & { \mathcal { E } = \bigg \{ \{ r _ { 1 } , r _ { 2 } \} \in { \binom { \mathcal { R } } { 2 } } \bigg | \ r _ { 1 } \cap r _ { 2 } \neq \emptyset \bigg \} ; } \\ & { L ( G ) = ( \mathcal { R } , \mathcal { E } ) ; } \end{array}\tag{9}
$$

The controller graph describes how roads are coupled directly through traffic, not which controllers may share information; in our model (Section IV), every controller draws on the state of the whole network. The strong-connectivity requirement in (8), by contrast, couples all roads at once and cannot be checked within any single neighborhood, so it is enforced during action resolution (Section III-E).

## C. Lane Allocations and Speed Limits

The lanes of a road are allocated between its two directions, and we describe an allocation by the number of lanes assigned to the forward direction; the remaining lanes are assigned to the backward direction, as in reversible-lane and contraflow operations [1] and dynamic lane reversal [5]. At time step t, road r has $\ell _ { r } ^ { ( t ) }$ active lanes and $\hat { \ell } _ { r } ^ { ( t ) }$ target lanes, both counted in the forward direction, and every road starts at its target, $\hat { \ell } _ { r } ^ { ( 0 ) } = \ell _ { r } ^ { ( 0 ) }$ . Whenever the two differ, the lanes in between must be cleared before they can change direction: they are closed in both directions until the vehicles traveling in their old direction have left, so that vehicles in the new direction can safely enter, as in the operation of reversible lanes in practice [1]. The open lanes in a direction are therefore those assigned to it by both the active and the target lanes, and the forward and backward open lanes need not sum to $L _ { r }$ . The roads with lanes being cleared are the clearing roads $\mathcal { R } _ { \approx } ^ { ( t ) }$ . Clearing is atomic: once it completes, all clearing lanes of the road open at once in their new direction, and the active lanes become equal to the target lanes. Fig. 2 illustrates this process.

![](images/3b15427cc8a86c2f76ba256b8de17de6c9f3246582761ea0575fffa0a6f9388c.jpg)  
Fig. 2. Clearing two lanes of a road with $L _ { r } = 5$ (lane 1, the first forward lane, at the bottom). When the target lanes $\boldsymbol { \hat { \ell } } _ { r }$ drop below the active lanes $\ell _ { r } ,$ lanes $\hat { \ell } _ { r } + 1 , \ldots , \ell _ { r }$ are closed to both directions (dark grey) while vehicles traveling in their old direction leave them. Clearing is atomic: once it completes, all of these lanes open at once to the backward direction (hatched; forward lanes are tinted) and $\bar { \ell } _ { r } = \hat { \ell } _ { r }$ . The open lanes $\bar { \ell } _ { r , d }$ below each state need not sum to $L _ { r }$ while the road is clearing.

$$
\begin{array} { r l } & { \ell _ { r } ^ { ( t ) } , \hat { \ell } _ { r } ^ { ( t ) } \in \{ 0 , . . . , L _ { r } \} \ \forall r \in \mathcal { R } \ \forall t \in \mathcal { T } ; } \\ & { \bar { \ell } _ { r , d } ^ { ( t ) } = \left\{ \begin{array} { l l } { \operatorname* { m i n } \left( \ell _ { r } ^ { ( t ) } , \hat { \ell } _ { r } ^ { ( t ) } \right) } & { \mathrm { i f ~ } d = \mathrm { f w d } } \\ { L _ { r } - \operatorname* { m a x } \left( \ell _ { r } ^ { ( t ) } , \hat { \ell } _ { r } ^ { ( t ) } \right) } & { \mathrm { i f ~ } d = \mathrm { b w d } } \end{array} \right. } \\ & { \forall ( r , d ) \in \mathcal { R } \times \mathcal { D } \ \forall t \in \check { \mathcal { T } } ; } \\ & { \mathcal { R } _ { \approx } ^ { ( t ) } = \left\{ r \in \mathcal { R } \ \Big | \ \ell _ { r } ^ { ( t ) } \neq \hat { \ell } _ { r } ^ { ( t ) } \right\} \ \forall t \in \check { \mathcal { T } } ; } \end{array}\tag{10}
$$

Each road controller also sets a speed limit for each direction of its road. The nominal speed limit $s _ { r , d } ^ { ( 0 ) }$ is the speed limit set by traffic engineers based on safety, road geometry, zoning, and similar considerations. It is an upper bound: the controller may lower the speed limit to regulate traffic, as in VSL control [3], [23], but never raise it beyond the nominal value, so the posted speed never exceeds the value that traffic engineers set for the road. From below, speed limits are bounded by a common crawl speed $s _ { \operatorname* { m i n } } \in \mathbb { R } _ { > 0 }$ , which every nominal speed limit exceeds. We parametrize the active speed limit $s _ { r , d } ^ { ( t ) }$ by the active normalized speed limit $u _ { r , d } ^ { ( t ) } \in [ 0 , 1 ]$ , which interpolates linearly between the crawl speed at 0 and the nominal speed limit at 1; initially, every active speed limit is nominal.

$$
\begin{array} { r l } & { u _ { r , d } ^ { ( 0 ) } = 1 , \ s _ { r , d } ^ { ( 0 ) } \in \mathbb { R } _ { > s _ { \operatorname* { m i n } } } \ \forall ( r , d ) \in \mathcal { R } \times \mathcal { D } ; } \\ & { u _ { r , d } ^ { ( t ) } \in [ 0 , 1 ] , \ s _ { r , d } ^ { ( t ) } = s _ { \operatorname* { m i n } } + u _ { r , d } ^ { ( t ) } \Big ( s _ { r , d } ^ { ( 0 ) } - s _ { \operatorname* { m i n } } \Big ) } \\ & { \ \forall ( r , d ) \in \mathcal { R } \times \mathcal { D } \ \forall t \in \check { \mathcal { T } } ; } \end{array}\tag{11}
$$

Since the crawl speed is strictly positive, lowering a speed limit slows the traffic in a direction but never stops it; traffic in a direction can only be stopped by closing all of its lanes. Vehicles on clearing lanes still travel in their old direction and remain subject to its active speed limit until they leave.

## D. Joint Action

At every decision step, the road controllers jointly select an action consisting of proposed lanes $\tilde { \ell } ^ { ( t ) }$ , one forward lane count per road, and proposed normalized speed limits $\tilde { u } ^ { ( t ) }$ , one per road and direction. The action space is the product of the admissible proposals for lanes and speed limits.

$$
\begin{array} { r l } & { \mathcal { A } ^ { ( t ) } = \tilde { \mathcal { L } } ^ { ( t ) } \times \tilde { \mathcal { U } } ^ { ( t ) } , } \\ & { a ^ { ( t ) } = \left( \tilde { \ell } ^ { ( t ) } , \tilde { u } ^ { ( t ) } \right) \in \mathcal { A } ^ { ( t ) } \forall t \in \mathcal { T } ; } \end{array}\tag{12}
$$

Admissible proposed lanes are defined per road. A clearing road is locked to its target: its only admissible proposal is its target lanes, so a road cannot be retargeted until its clearing completes. A bridge road that is not clearing must keep at least one lane in each direction, which excludes the extreme proposals 0 and $L _ { r }$ . Every other road may propose any allocation, including one that assigns all of its lanes to a single direction.

$$
\begin{array} { r l } & { \tilde { \mathcal { L } } _ { r } ^ { ( t ) } = \left\{ \begin{array} { l l } { \left\{ \hat { \ell } _ { r } ^ { ( t ) } \right\} } & { \mathrm { i f ~ } r \in \mathcal { R } _ { \approx } ^ { ( t ) } } \\ { \left\{ 1 , \ldots , L _ { r } - 1 \right\} } & { \mathrm { i f ~ } r \in \mathcal { R } _ { \Leftrightarrow } \setminus \mathcal { R } _ { \approx } ^ { ( t ) } } \\ { \left\{ 0 , \ldots , L _ { r } \right\} } & { \mathrm { o t h e r w i s e } } \end{array} \right. } \\ & { \forall r \in \mathcal { R } \ \forall t \in \mathcal { T } ; } \\ & { \tilde { \mathcal { L } } ^ { ( t ) } = \displaystyle \times \ \times \tilde { \mathcal { L } } _ { r } ^ { ( t ) } \ \forall t \in \mathcal { T } ; } \end{array}\tag{13}
$$

Admissible proposed normalized speed limits are defined per road and direction. A direction with open lanes may receive any proposed normalized speed limit in [0, 1], whereas the proposed normalized speed limit of a direction without open lanes is frozen at its active normalized speed limit, since no vehicle can enter that direction.

$$
\begin{array} { r l } & { \tilde { \mathcal { U } } _ { r , d } ^ { ( t ) } = \left\{ \left\{ \begin{array} { l l } { \boldsymbol { u } _ { r , d } ^ { ( t ) } } } & { \mathrm { i f ~ } \bar { \ell } _ { r , d } ^ { ( t ) } = 0 } \\ { \left[ 0 , 1 \right] } & { \mathrm { o t h e r w i s e } } \end{array} \right. } \\ & \right\{ \quad \forall ( r , d ) \in \mathcal { R } \times \mathcal { D } \ \forall t \in \mathcal { T } ; } \\ & { \tilde { \mathcal { U } } ^ { ( t ) } = \quad \quad \times \quad \tilde { \mathcal { U } } _ { r , d } ^ { ( t ) } \ \forall t \in \mathcal { T } ; } \\ & { \quad \quad ( r , d ) \in \mathcal { R } \times \mathcal { D } } \end{array}\tag{14}
$$

Every admissible set in (13) and (14) is nonempty; for bridge roads that are not clearing, this relies on $L _ { r } \ge 2$ from (5). Hence $\boldsymbol { \mathcal { A } ^ { ( t ) } } \neq \boldsymbol { \emptyset }$ for all $t \in \tau$

Admissibility is a per-road property and does not depend on the proposals of other roads. In particular, the action space encodes the strong-connectivity requirement in (8), which couples all roads, only through the bridge-road restriction. A closing proposal of a bridge road would always be rejected, so excluding it from the action space leaves less to action resolution and makes the effect of an action more predictable. All other actions that would violate the requirement are filtered during action resolution.

## E. Action Resolution

Proposed lanes do not take effect immediately: an accepted proposal becomes the road’s next target lanes, and the road then clears toward them. A new target closes the lanes on which it disagrees with the active lanes at once, but these lanes reopen in their new direction only once clearing completes, at a time determined by traffic. We therefore assess proposals in the state before clearing completes, which has the fewest open lanes; completing clearing can only add to them. To this end, we consider a profile n of target lanes, with $n _ { r }$ for road r. The prospective open lanes $\mu _ { r , d } ^ { ( t ) } ( \bar { n } _ { r } )$ are the open lanes of road r in direction d if its target lanes were $n _ { r }$ while its current active lanes $\ell _ { r } ^ { ( t ) }$ remained;

$$
\begin{array} { r l } & { \mu _ { r , d } ^ { ( t ) } : \{ 0 , \dots , L _ { r } \} \to \{ 0 , \dots , L _ { r } \} , } \\ & { \mu _ { r , d } ^ { ( t ) } ( n _ { r } ) = \left\{ \begin{array} { l l } { \operatorname* { m i n } \Bigl ( \ell _ { r } ^ { ( t ) } , n _ { r } \Bigr ) } & { \mathrm { i f ~ } d = \mathrm { f w d } } \\ { L _ { r } - \operatorname* { m a x } \Bigl ( \ell _ { r } ^ { ( t ) } , n _ { r } \Bigr ) } & { \mathrm { i f ~ } d = \mathrm { b w d } } \end{array} \right. } \\ & { \quad \forall ( r , d ) \in \mathcal { R } \times \mathcal { D } \ \forall t \in \breve { \mathcal { T } } ; } \end{array}\tag{15}
$$

in particular, by (10), $\mu _ { r , d } ^ { ( t ) } \Big ( \hat { \ell } _ { r } ^ { ( t ) } \Big ) ~ = ~ \bar { \ell } _ { r , d } ^ { ( t ) }$ . The prospective navigable arc set $\mathcal { P } ^ { ( t ) } ( n )$ contains the arcs that would have at least one open lane.

$$
\begin{array} { r l } & { \mathcal { P } ^ { ( t ) } : \bigcup _ { r \in \mathcal { R } } \{ 0 , \dotsc , L _ { r } \}  2 ^ { \omega ( \mathcal { R } \times \mathcal { D } ) } , } \\ & { \mathcal { P } ^ { ( t ) } ( n ) = \omega \Big ( \Big \{ ( r , d ) \in \mathcal { R } \times \mathcal { D } \Big | \mu _ { r , d } ^ { ( t ) } ( n _ { r } ) > 0 \Big \} \Big ) } \\ & { \forall t \in \check { \mathcal { T } } ; } \end{array}\tag{16}
$$

Finally, for a road with at least two lanes, the relaxation function $\rho _ { r }$ maps each extreme allocation, 0 or $L _ { r } ,$ , to the nearest allocation that leaves one lane in the opposite direction.

$$
\begin{array} { r l } & { \rho _ { r } : \{ 0 , L _ { r } \} \to \{ 1 , L _ { r } - 1 \} , } \\ & { \rho _ { r } ( n _ { r } ) = \left\{ \begin{array} { l l } { 1 } & { \mathrm { i f ~ } n _ { r } = 0 } \\ { L _ { r } - 1 } & { \mathrm { i f ~ } n _ { r } = L _ { r } } \end{array} \right. } \\ & { \quad \forall r \in \mathcal { R } \mathrm { ~ w i t h ~ } L _ { r } \geq 2 ; } \end{array}\tag{17}
$$

We call a road’s proposal closing if it would close a direction that is currently open, that is, if $\bar { \ell } _ { r , d } ^ { ( t ) } > 0$ but $\mu _ { r , d } ^ { ( t ) } \Big ( \tilde { \ell } _ { r } ^ { ( t ) } \Big ) \ = \ 0$ for some direction $d .$ An open forward direction implies $\ell _ { r } ^ { ( t ) } \geq 1$ , so it closes only for the proposal $0 ;$ likewise, an open backward direction implies $\ell _ { r } ^ { ( t ) } \leq \dot { L } _ { r } - 1$ so it closes only for the proposal $L _ { r }$ . Hence every closing proposal is extreme, $\tilde { \ell } _ { r } ^ { ( t ) } \in \dot { \left\{ 0 , L _ { r } \right\} } . \mathbf { B y } \left( 1 3 \right)$ , neither a clearing road nor a bridge road ever proposes closing: the former proposes its target lanes, whose prospective open lanes are its open lanes, and the latter cannot propose 0 or $L _ { r } . \mathrm { ~ A ~ }$ nonclosing proposal keeps every open direction open and therefore never removes an arc from the prospective navigable arc set. Algorithm 1 exploits this: it accepts all non-closing proposals at once, then considers the closing roads one at a time in uniformly random order, tentatively accepting each proposal and rejecting it if the prospective navigable arc set would leave S. A rejected proposal is relaxed rather than discarded: a road with at least two lanes adopts the relaxed proposal $\rho _ { r } \left( \tilde { \ell } _ { r } ^ { ( t ) } \right)$ which keeps one lane in the direction the proposal would have closed, whereas a single-lane road reverts to its current target lanes. The algorithm builds the next target lanes $\hat { \ell } ^ { ( t + 1 ) }$ in place, starting from the current target lanes ${ \hat { \ell } } ^ { ( t ) }$ . The random order prevents any road from being systematically favored when closing proposals conflict, and it makes the transition from one time step to the next stochastic, together with traffic. Algorithm 1 is a shield [13] in the sense of Section II-D.

Closing proposals conflict when each keeps the network strongly connected on its own but not together with the others, which is why they are checked one at a time against the proposals already accepted. Suppose, unlike in Fig. 1 (b), that every road of the left cycle of the example network has both directions open and is not clearing, and that $r _ { 1 }$ proposes $L _ { r _ { 1 } } = 2 $ , all of its lanes forward from $j _ { 1 }$ to $j _ { 2 }$ , and $r _ { 3 }$ proposes 0, all of its lanes backward from $j _ { 4 }$ to $j _ { 3 } ,$ , while all other roads propose their target lanes. Either closing proposal alone leaves the network strongly connected, but together they leave no arc from the cut $\{ j _ { 2 } , j _ { 3 } , j _ { 5 } , j _ { 6 } , j _ { 7 } \}$ to its complement. Whichever of the two comes first in the random order is accepted, and the other is relaxed to one lane in each direction.

Proposed normalized speed limits are always accepted and become the next active normalized speed limits. Each road then either keeps its current active lanes or completes clearing, in which case its next active lanes are its next target lanes; which of the two occurs is determined by traffic, not by the

Algorithm 1 Resolution of lane proposals at decision step   
$t \in { \mathcal { T } } .$   
Input: active lanes $\ell ^ { ( t ) }$ , target lanes $\hat { \ell } ^ { ( t ) }$ , proposed lanes   
$\tilde { \ell } ^ { ( t ) }$   
Output: next target lanes $\hat { \ell } ^ { ( t + 1 ) }$   
1: start from the target lanes: $\hat { \ell } ^ { ( t + 1 ) } \gets \hat { \ell } ^ { ( t ) }$   
2: start with an empty list of closing roads   
3: for each road $r \in \mathcal { R }$ do   
4: if $\bar { \ell } _ { r , d } ^ { ( t ) } > 0 \Rightarrow \bar { \mu } _ { r , d } ^ { ( t ) } \mathopen { } \mathclose \bgroup \left( \tilde { \ell } _ { r } ^ { ( t ) } \aftergroup \egroup \right) > 0$ for all $d \in \mathcal { D }$ then   
5: accept the non-closing proposal: $\hat { \ell } _ { r } ^ { ( t + 1 ) } \gets \tilde { \ell } _ { r } ^ { ( t ) }$   
6: else   
7: append $r$ to the list of closing roads   
8: shuffle the closing roads uniformly at random   
9: for each closing road $r ,$ in shuffled order, do   
10: tentatively accept the proposal: $\hat { \ell } _ { r } ^ { ( t + 1 ) } \gets \tilde { \ell } _ { r } ^ { ( t ) }$   
11: if ${ \mathcal { P } } ^ { ( t ) } { \Big ( } { \dot { \ell } } ^ { ( t + 1 ) } { \Big ) } { \Big ) } \neq s$ then   
12: if $\dot { L _ { r } } \geq 2$ then   
13: relax the proposal: $\hat { \ell } _ { r } ^ { ( t + 1 ) } \gets \rho _ { r } \Big ( \tilde { \ell } _ { r } ^ { ( t ) } \Big )$   
14: else   
15: revert to the target lanes: $\hat { \ell } _ { r } ^ { ( t + 1 ) } \gets \hat { \ell } _ { r } ^ { ( t ) }$   
16: return $\hat { \ell } ^ { ( t + 1 ) }$

![](images/6e26227bf068ebcfe2b3a222caa0f4e46892f032d87c0876f82d38ac39789d9d.jpg)  
Fig. 3. One decision step. The policy of the road controllers proposes an action $a ^ { ( t ) }$ , consisting of proposed lanes and proposed normalized speed limits for every road. Action resolution rejects closing proposals that would break strong connectivity, relaxing them on multi-lane roads and reverting them on single-lane roads, yielding the next target lanes $\hat { \ell } ^ { ( t + 1 ) } \colon$ ; proposed normalized speed limits are always accepted and become the next active normalized speed limits $\boldsymbol u ^ { ( t + 1 ) }$ . The traffic network applies the resulting active speed limits immediately, and each clearing road completes clearing once its clearing lanes are empty.

controllers (Fig. 3).

$$
\begin{array} { r l } & { u _ { r , d } ^ { ( t + 1 ) } = \tilde { u } _ { r , d } ^ { ( t ) } \forall ( r , d ) \in \mathcal { R } \times \mathcal { D } \ \forall t \in \mathcal { T } ; } \\ & { \ell _ { r } ^ { ( t + 1 ) } \in \left\{ \ell _ { r } ^ { ( t ) } , \hat { \ell } _ { r } ^ { ( t + 1 ) } \right\} \forall r \in \mathcal { R } \ \forall t \in \mathcal { T } ; } \end{array}\tag{18}
$$

Proposition 1: If $\vec { \mathcal { R } } ^ { ( 0 ) } \in \mathcal { S }$ , then $\vec { \mathcal { R } } ^ { ( t ) } \in \mathcal { S }$ for all $t \in { \breve { T } }$ and the navigable arc set stays in $s$ between consecutive time steps, for any admissible actions $a ^ { ( t ) } \in \mathcal { A } ^ { ( t ) }$ , any order of the closing roads in Algorithm 1, and any times at which clearing completes.

Proof: We show that the requirement carries over from time step $t \in \tau$ , where $\vec { \mathcal { R } } ^ { ( t ) } \in \vec { \mathcal { S } } ,$ , to time step t + 1 and to the period in between, in two stages.

First, the prospective navigable arc set $\mathcal { P } ^ { ( t ) } \Big ( \hat { \ell } ^ { ( t + 1 ) } \Big )$ remains in S throughout Algorithm 1. Initially, it equals $\mathring { \mathcal { R } } ^ { ( t ) } \in$

S, because $\mu _ { r , d } ^ { ( t ) } \Big ( \hat { \ell } _ { r } ^ { ( t ) } \Big ) \ : = \ : \bar { \ell } _ { r , d } ^ { ( t ) }$ for every road and direction. Accepting a non-closing proposal leaves the arc set unchanged: it keeps every open direction open, and no proposal opens a closed direction before clearing completes. A closing proposal is kept only if the arc set stays in S. When a closing proposal is rejected, reverting restores the road’s open lanes, and therefore the preceding arc set in S. Relaxing restores the preceding arc set as well, because the relaxed proposal is neither 0 nor $L _ { r }$ and is therefore not closing. Hence, once the algorithm returns, ${ \mathcal { P } } ^ { ( t ) } { \Big ( } { \hat { \ell } } ^ { ( t + 1 ) } { \Big ) } \in { \mathcal { S } }$

Second, the next navigable arc set $\vec { \mathcal { R } } ^ { ( t + 1 ) }$ contains $\mathcal { P } ^ { ( t ) } \Big ( \hat { \ell } ^ { ( t + 1 ) } \Big )$ : by (18), a road that does not complete clearing has exactly its prospective open lanes at the next time step, whereas a road that completes clearing opens all of its clearing lanes, which only adds open lanes. Since S is closed under supersets, $\vec { \mathcal { R } } ^ { ( t + 1 ) } \in \mathcal { S }$ . The same holds between the two time steps: the next target lanes and next active speed limits take effect immediately, so the open lanes in that period are at least the prospective ones, and the navigable arc set stays in S throughout. Induction on t completes the proof. □

## IV. PROPOSED ARCHITECTURE

The road controllers share a single centralized model, a transformer [14]. The name of its attention mechanism, Global Endpoint Relation-Aware Spatio-Temporal Self-Attention (GERASTSA), lists its properties. It is global: every token may attend to every road at its own and every earlier time step, not only to its neighbors in the controller graph. It is endpoint relation-aware: a learned bias, selected by how two roads are related through their endpoints, is added to every attention score, in the spirit of relation-aware attention [42] and graph transformers [15]. It is spatio-temporal: its tokens are road–time pairs, so space and time are attended to jointly, and the GERASTSA transformer adds a spatial embedding of every road and a temporal embedding of every window position to the projected input. The controller of a road is the shared model read out at that road’s tokens: the regressor of Section V-B and the heads of Section V-C act on the per-road outputs.

The graphs G and $L ( G ) \ = \ ( \mathcal { R } , \mathcal { E } )$ are those of Section III. The model processes the roads in the order of a fixed bijective road indexing ι, with road index i for road $\iota ( i )$ . Its hyperparameters are the input, output, model, and feedforward widths $F _ { \mathrm { X } } , F _ { \mathrm { Y } } , F _ { \mathrm { Z } }$ , and $F _ { \mathrm { F } } ;$ the numbers H of heads, each of head width $F _ { \mathrm { H } }$ , and N of layers; the maximum window length $W _ { \mathrm { m a x } } ;$ the maximum relation distance $\Lambda _ { \mathrm { m a x } } ;$ the maximum junction degree $\Gamma _ { \mathrm { m a x } } ;$ and the stabilizer $\varepsilon _ { \mathrm { L N } }$ of layer normalization.

$$
\begin{array} { r l } & { F _ { \mathrm { X } } , F _ { \mathrm { Y } } , F _ { \mathrm { Z } } , F _ { \mathrm { F } } , H , N , W _ { \mathrm { m a x } } , \Gamma _ { \mathrm { m a x } } \in \mathbb { Z } _ { \ge 1 } ; } \\ & { \Lambda _ { \operatorname* { m a x } } \in \mathbb { Z } _ { \ge 2 } ; \quad \varepsilon _ { \mathrm { L N } } \in \mathbb { R } _ { > 0 } ; } \\ & { H \mid F _ { \mathrm { Z } } ; \quad F _ { \mathrm { H } } = \cfrac { F _ { \mathrm { Z } } } { H } ; \quad R = | \mathcal { R } | ; } \\ & { \iota : \{ 1 , \ldots , R \}  \mathcal { R } \mathrm { ~ b i j e c t i v e } ; } \end{array}\tag{19}
$$

Bold symbols are learnable, tensor indices start at 1, and, just as (t) marks time step t, the superscripts [n] and ⟨k⟩ mark layer n and sample k. A product with a matrix, as in $X W _ { \mathrm { Q } } ,$ acts on the last axis; ⊤ swaps the last two axes, and a product of two tensors, as in $Q ^ { \prime } K ^ { \prime \top }$ , multiplies their last two axes for every index of the leading axes. A tensor added to one of higher order is repeated along the missing axes, and Softmax acts on the last axis, with $\exp ( - \infty ) = 0$ . Layer normalization LN [43] and the SwiGLU feedforward network FFN [44], with biases in all its linear maps, act on every token separately and receive their hyperparameters after a semicolon.

## A. GERASTSA Block

A GERASTSA block is the attention mechanism on its own. Its input consists of B samples of W window positions, ordered from past to present, with one token $X _ { k , w , i } \in \mathbb { R } ^ { F _ { \mathrm { Z } } }$ per sample k, position w, and road index i. An attention mask M allows (0) or forbids (−∞) attention between tokens of the same sample, and every token must be allowed to attend to some token.

$$
\begin{array} { r l } & { B , W \in \mathbb { Z } _ { \geq 1 } ; \quad X \in \mathbb { R } ^ { B \times W \times R \times F _ { \mathbb { Z } } } ; } \\ & { M \in \{ 0 , - \infty \} ^ { B \times W \times R \times W \times R } ; } \\ & { \exists ( w ^ { \prime } , i ^ { \prime } ) \in \{ 1 , . . . , W \} \times \{ 1 , . . . , R \} : } \\ & { M _ { k , w , i , w ^ { \prime } , i ^ { \prime } } = 0 \forall k \in \{ 1 , . . . , B \} } \\ & { \forall ( w , i ) \in \{ 1 , . . . , W \} \times \{ 1 , . . . , R \} ; } \end{array}\tag{20}
$$

Within a sample, attention runs over all W R tokens. The token index map θ flattens window positions and road indices, and the head-feature index map σ places the features of each head within the model width.

$$
\begin{array} { r l } & { \theta : \{ 1 , \dotsc , W \} \times \{ 1 , \dotsc , R \} \to \{ 1 , \dotsc , W R \} ; } \\ & { \theta ( w , i ) = ( w - 1 ) R + i ; } \\ & { \sigma : \{ 1 , \dotsc , H \} \times \{ 1 , \dotsc , F _ { \mathrm { H } } \} \to \{ 1 , \dotsc , F _ { \mathrm { Z } } \} ; } \\ & { \sigma ( h , f ) = ( h - 1 ) F _ { \mathrm { H } } + f ; } \end{array}\tag{21}
$$

Roads are related through their endpoints. The headjunction $\eta ( r , d )$ is the junction into which road r carries traffic in direction d. The hop set $\mathcal { N } _ { q } ( r )$ holds the roads reached from r by q hops in $L ( G )$ , and the hop distance $\boldsymbol { \lambda } ( \boldsymbol { r } , \boldsymbol { r } ^ { \prime } )$ is the fewest hops that reach $r ^ { \prime } ;$ it is finite because $L ( G )$ is connected, as G is.

$$
\begin{array} { r l } & { \eta : \mathcal { R } \times \mathcal { D } \to \mathcal { I } ; \quad \eta ( r , d ) = \left\{ \begin{array} { l l } { j _ { r } ^ { + } } & { \mathrm { i f ~ } d = \mathrm { f w e d } } \\ { j _ { r } ^ { - } } & { \mathrm { i f ~ } d = \mathrm { b w d } } \end{array} \right. ; } \\ & { \mathcal { N } _ { q } : \mathcal { R } \to 2 ^ { R } , } \\ & { \mathcal { N } _ { q } ( r ) = } \\ & { \left\{ \begin{array} { l l } { \{ r \} } & { \mathrm { i f ~ } q = 0 } \\ { \bigcup _ { r ^ { \prime } \in \mathcal { N } _ { q - 1 } ( r ) } \{ r ^ { \prime } \in \mathcal { R } \mid \{ r ^ { \prime \prime } , r ^ { \prime } \} \in \mathcal { E } \} } & { \mathrm { o t h e r w i s e } } \end{array} \right. } \\ & { \forall q \in \mathbb { Z } _ { \ge 0 } ; } \\ & { \lambda : \mathcal { R } ^ { 2 } \to \mathbb { Z } _ { \ge 0 } ; } \\ & { \lambda ( r , r ^ { \prime } ) = \operatorname* { m i n } \{ q \in \mathbb { Z } _ { \ge 0 } \mid r ^ { \prime } \in \mathcal { N } _ { q } ( r ) \} ; } \end{array}\tag{22}
$$

The relation function κ maps every ordered pair of roads to a relation in the relation set K. A road has the self relation to itself. Two adjacent roads share exactly one junction, since G is simple, and their adjacency relation is the unique pair of directions in which both flow into it; for instance, (fwd, bwd) means that r flows into the shared junction forward and $r ^ { \prime }$ backward. Any other pair has the hop distance, capped at $\Lambda _ { \mathrm { m a x } }$ , as its distance relation, much like the shortest-path spatial encoding of graph transformers [15] and the clipped relative distances of [42], so $| \mathcal { K } | = \Lambda _ { \mathrm { m a x } } + 4$ . The relation index ξ enumerates the relations, and the relation index matrix Ξ holds the relation indices of all pairs of road indices (Fig. 4).

![](images/d6eb6c58fa25604473a6d80b34e92f2631072c9764e935267f5574b78876c996.jpg)  
Fig. 4. Relations $\kappa ( r _ { 5 } , \cdot )$ on the controller graph of the example network, with $\Lambda _ { \operatorname* { m a x } } = 2$ (heavy border: $r _ { 5 } ;$ shaded: roads adjacent to $r _ { 5 } ;$ white: roads at hop distance at least 2). Roads adjacent to $r _ { 5 } .$ , with $\omega ( r _ { 5 } , \mathrm { f w d } ) = ( j _ { 3 } , j _ { 5 } )$ are told apart by the directions in which the two roads flow into their shared junction, e.g. r<sub>2</sub>, with $\omega ( r _ { 2 } , \mathrm { f w d } ) = ( j _ { 2 } , j _ { 3 } )$ , flows into $j _ { 3 }$ forward while $r _ { 5 }$ flows into it backward. Each relation $\kappa ( r _ { 5 } , r )$ selects the learned relational bias $B _ { \mathrm { R } , h , \xi } ( \kappa ( r _ { 5 } , r ) )$ of every head $h .$

$$
\begin{array} { r l } & { \mathcal { K } = \{ \mathrm { s e l f } \} \cup \mathcal { D } ^ { 2 } \cup \{ 2 , \ldots , \Lambda _ { \operatorname* { m a x } } \} ; \quad \kappa : \mathcal { R } ^ { 2 }  \mathcal { K } ; } \\ & { \kappa ( r , r ^ { \prime } ) = \{ \begin{array} { l l } { \mathrm { s e l f } } & { \mathrm { i f ~ } r = r ^ { \prime } } \\ { ( d , d ^ { \prime } ) } & { \mathrm { i f ~ } \{ r , r ^ { \prime } \} \in \mathcal { E } } \\ { \operatorname* { m i n } ( \lambda ( r , r ^ { \prime } ) , \Lambda _ { \operatorname* { m a x } } ) } & { \mathrm { o t h e r w i s e } } \end{array}  } \\ & { \mathrm { ~ w h e r e ~ } \eta ( r , d ) = \eta ( r ^ { \prime } , d ^ { \prime } ) ; } \\ & { \mathcal { E } : \mathcal { K }  \{ 1 , \ldots , | \mathcal { K } | \} \mathrm { ~ b i j e c t i v e } ; } \\ & { \Xi \in \{ 1 , \ldots , | \mathcal { K } | \} ^ { R \times R } ; } \\ & { \Xi _ { i , i ^ { \prime } } = \xi ( \kappa ( i , \mathfrak { i } ) , \iota ( i ^ { \prime } ) ) \forall ( i , i ^ { \prime } ) \in \{ 1 , \ldots , R \} ^ { 2 } ; } \end{array}\tag{23}
$$

Every token is projected to a query, a key, and a value.

$$
\begin{array} { r l } & { W _ { \mathrm { Q } } , W _ { \mathrm { K } } , W _ { \mathrm { V } } \in \mathbb R ^ { F _ { \mathrm { Z } } \times F _ { \mathrm { Z } } } ; \quad b _ { \mathrm { Q } } , b _ { \mathrm { K } } , b _ { \mathrm { V } } \in \mathbb R ^ { F _ { \mathrm { Z } } } ; } \\ & { Q = X W _ { \mathrm { Q } } + b _ { \mathrm { Q } } , \ K = X W _ { \mathrm { K } } + b _ { \mathrm { K } } , } \\ & { V = X W _ { \mathrm { V } } + b _ { \mathrm { V } } \ \in \mathbb R ^ { B \times W \times R \times F _ { \mathrm { Z } } } ; } \end{array}\tag{24}
$$

The projections are split into the H heads and flattened into token sequences.

$$
\begin{array} { r l } & { Q ^ { \prime } , K ^ { \prime } , V ^ { \prime } \in \mathbb { R } ^ { B \times H \times W R \times F _ { \mathrm { H } } } ; } \\ & { Q _ { k , h , \theta ( w , i ) , f } ^ { \prime } = Q _ { k , w , i , \sigma ( h , f ) } , } \\ & { K _ { k , h , \theta ( w , i ) , f } ^ { \prime } = K _ { k , w , i , \sigma ( h , f ) } , } \\ & { V _ { k , h , \theta ( w , i ) , f } ^ { \prime } = V _ { k , w , i , \sigma ( h , f ) } } \\ & { \forall k \in \{ 1 , \dots , B \} } \\ & { \forall ( h , f ) \in \{ 1 , \dots , H \} \times \{ 1 , \dots , F _ { \mathrm { H } } \} } \\ & { \forall ( w , i ) \in \{ 1 , \dots , W \} \times \{ 1 , \dots , R \} ; } \end{array}\tag{25}
$$

Each head learns one relational bias per relation, collected in $B _ { \mathrm { R } }$ . The bias depends only on the roads of two tokens,

so a relation is treated alike at every time lag; temporal order enters only through the mask and the temporal embedding of Section IV-B. Bias and mask are flattened like the tokens.

$$
\begin{array} { r l } & { B _ { \mathrm { R } } \in \mathbb { R } ^ { H \times | K | } ; } \\ & { B _ { \mathrm { R } } ^ { \prime } \in \mathbb { R } ^ { H \times W R \times W R } ; \quad M ^ { \prime } \in \left\{ 0 , - \infty \right\} ^ { B \times W R \times W R } ; } \\ & { B _ { \mathrm { R } , h , \theta ( w , i ) , \theta ( w ^ { \prime } , i ^ { \prime } ) } ^ { \prime } = B _ { \mathrm { R } , h , \Xi _ { i , i ^ { \prime } } } \forall h \in \{ 1 , \dots , H \} } \\ & { \forall ( w , w ^ { \prime } ) \in \{ 1 , \dots , W \} ^ { 2 } \forall ( i , i ^ { \prime } ) \in \{ 1 , \dots , R \} ^ { 2 } ; } \\ & { M _ { k , \theta ( w , i ) , \theta ( w ^ { \prime } , i ^ { \prime } ) } ^ { \prime } = M _ { k , w , i , w ^ { \prime } , i ^ { \prime } } \forall k \in \{ 1 , \dots , B \} } \\ & { \forall ( w , w ^ { \prime } ) \in \{ 1 , \dots , W \} ^ { 2 } \forall ( i , i ^ { \prime } ) \in \{ 1 , \dots , R \} ^ { 2 } ; } \end{array}\tag{26}
$$

The attention scores add the relational bias, shared by all samples, and the mask, shared by all heads, to the scaled dot products of queries and keys [14]. By (20), the attention weights are well defined, and they vanish exactly where the mask forbids attention.

$$
\begin{array} { l } { S = \displaystyle \frac { Q ^ { \prime } K ^ { \prime \top } } { \sqrt { F _ { \mathrm { H } } } } + \boldsymbol { B } _ { \mathrm { R } } ^ { \prime } + \boldsymbol { M } ^ { \prime } } \\ { \displaystyle \in \left( \mathbb { R } \cup \left\{ - \infty \right\} \right) ^ { B \times H \times W R \times W R } ; } \\ { Z ^ { \prime } = \mathrm { S o f t m a x } ( S ) V ^ { \prime } \ \in \mathbb { R } ^ { B \times H \times W R \times F _ { \mathrm { H } } } ; } \end{array}\tag{27}
$$

Merging the heads

$$
\begin{array} { r l } & { Z \in \mathbb { R } ^ { B \times W \times R \times F _ { \mathrm { Z } } } ; } \\ & { Z _ { k , w , i , \sigma ( h , f ) } = Z _ { k , h , \theta ( w , i ) , f } ^ { \prime } } \\ & { \forall k \in \{ 1 , \ldots , B \} } \\ & { \forall ( h , f ) \in \{ 1 , \ldots , H \} \times \{ 1 , \ldots , F _ { \mathrm { H } } \} } \\ & { \forall ( w , i ) \in \{ 1 , \ldots , W \} \times \{ 1 , \ldots , R \} ; } \end{array}\tag{28}
$$

and a learnable output projection

$$
\begin{array} { r l } & { { \pmb W } _ { \mathrm { Y } } \in \mathbb { R } ^ { F _ { \mathrm { Z } } \times F _ { \mathrm { Z } } } ; \quad { \pmb b } _ { \mathrm { Y } } \in \mathbb { R } ^ { F _ { \mathrm { Z } } } ; } \\ & { { \pmb Y } = { \cal Z } { \pmb W } _ { \mathrm { Y } } + { \pmb b } _ { \mathrm { Y } } \in \mathbb { R } ^ { B \times W \times R \times F _ { \mathrm { Z } } } ; } \\ & { \mathrm { G E R A S T S A B } ( X , M ) = Y ; } \end{array}\tag{29}
$$

yield the output of the block (Fig. 5). The symbols of the data flow in (20) to (29) are local to the block.

## B. GERASTSA Transformer

The GERASTSA transformer maps B histories of the network to one output vector per sample, road, and window position. Sample k ends at a time step $t _ { k } \in \breve { \tau }$ and holds the input features, defined in Section ${ \mathrm { V } } ,$ of all roads at the last $W _ { k }$ time steps up to $t _ { k }$ , ordered from past to present.

$$
\begin{array} { r l } & { B \in \mathbb { Z } _ { \ge 1 } ; } \\ & { t _ { k } \in \check { \mathcal { T } } , \ W _ { k } = \operatorname* { m i n } ( t _ { k } + 1 , W _ { \operatorname* { m a x } } ) , } \\ & { X ^ { \langle k \rangle } \in \mathbb { R } ^ { W _ { k } \times R \times F _ { \mathrm { X } } } \ \forall k \in \{ 1 , \dots , B \} ; } \end{array}\tag{30}
$$

The samples are left-padded with zeros to the longest window length W, so that their last time steps align at position W.

![](images/b3059fa94c2f36c3558c336f057107e48652c4f62cee4c8f90f702b28abedacd.jpg)  
Fig. 5. The GERASTSA block. Queries, keys, and values are split into H heads and flattened over the WR road–time tokens of each sample. The relational bias $B _ { \mathrm { R } } ^ { \prime }$ is gathered from the learned table $\scriptstyle B _ { \mathrm { R } }$ through the relation index matrix $\Xi$ and shared across all window positions and samples, and the flattened mask $M ^ { \prime }$ forbids attention as specified by M, which in the transformer is the mask of (32).

Position $w$ of sample k is valid if $w > W - W _ { k } ,$ in which case it holds time step $t _ { k } - ( W - w )$ , and padding otherwise.

$$
\begin{array} { r l } & { W = \underset { k \in \{ 1 , \dots , B \} } { \operatorname* { m a x } } W _ { k } ; \quad X \in \mathbb { R } ^ { B \times W \times R \times F _ { \mathrm { X } } } ; } \\ & { X _ { k , w , i , f } = \left\{ X _ { w - W + W _ { k } , i , f } ^ { \langle k \rangle } \quad \mathrm { i f ~ } w > W - W _ { k } \right. } \\ & { \mathrm { ~ o t h e r w i s e ~ } } \\ & { \forall k \in \{ 1 , \dots , B \} \forall w \in \{ 1 , \dots , W \} } \\ & { \forall i \in \{ 1 , \dots , R \} \forall f \in \{ 1 , \dots , F _ { \mathrm { X } } \} ; } \end{array}\tag{31}
$$

The symbols of the data flow in (30) to (37) are local to the transformer.

The attention mask is causal: a token may attend to all roads at the valid positions up to its own. It may also attend to its own position, so that padding tokens satisfy (20).

$$
\begin{array}{c} \begin{array} { r l } & { M \in \{ 0 , - \infty \} ^ { B \times W \times R \times W \times R } ; } \\ & { M _ { k , w , i , w ^ { \prime } , i ^ { \prime } } = } \\ & { \begin{array} { r l } { \left\{ 0 \right. } & { \mathrm { i f ~ } W - W _ { k } < w ^ { \prime } \leq w \lor w ^ { \prime } = w } \\ { - \infty } & { \mathrm { o t h e r w i s e } } \end{array} } \\ & { \forall k \in \{ 1 , \dots , B \} \forall ( w , w ^ { \prime } ) \in \{ 1 , \dots , W \} ^ { 2 } } \\ & { \forall ( i , i ^ { \prime } ) \in \{ 1 , \dots , R \} ^ { 2 } ; } \end{array}  \end{array}\tag{32}
$$

The junction degree $\gamma ( r , d )$ is the number of roads that meet at the head junction of road r in direction d. The spatial embedding $E _ { \mathrm { S } } ^ { \prime }$ of a road sums, over both directions, the rows that its junction degrees, capped at $\Gamma _ { \mathrm { m a x } }$ , select from the learnable tables $E _ { \mathrm { S } } ^ { d } ,$ a directional variant of the degree-based

centrality encoding of [15].

$$
\begin{array} { r l } & { \gamma : \mathcal { R } \times \mathcal { D } \to \mathbb { Z } _ { \ge 1 } ; } \\ & { \gamma ( r , d ) = | \{ r ^ { \prime } \in \mathcal { R } \mid \eta ( r , d ) \in r ^ { \prime } \} | ; } \\ & { E _ { \mathrm { S } } ^ { d } \in \mathbb { R } ^ { \Gamma _ { \operatorname* { m a x } } \times F _ { \mathrm { Z } } } \forall d \in \mathcal { D } ; \quad E _ { \mathrm { S } } ^ { \prime } \in \mathbb { R } ^ { R \times F _ { \mathrm { Z } } } ; } \\ & { E _ { \mathrm { S } , i , f } ^ { \prime } = \displaystyle \sum _ { d \in \mathcal { D } } E _ { \mathrm { S } , \operatorname* { m i n } ( \gamma ( \iota ( i ) , d ) , \Gamma _ { \operatorname* { m a x } } ) , f } ^ { d } } \\ & { \forall i \in \{ 1 , \ldots , R \} \forall f \in \{ 1 , \ldots , F _ { \mathrm { Z } } \} ; } \end{array}\tag{33}
$$

The temporal embedding $E _ { \mathrm { T } } ^ { \prime }$ gives position w row $W _ { \mathrm { m a x } } -$ $( W - w )$ of the learnable table $\pmb { { \cal E } } _ { \mathrm { T } }$ , a row determined by how many time steps w precedes the last time step. The last time step thus always receives row $W _ { \mathrm { m a x } }$ , however much its sample is padded.

$$
\begin{array} { r l } & { { \cal E } _ { \mathrm { T } } \in \mathbb { R } ^ { W _ { \operatorname* { m a x } } \times F _ { \mathrm { Z } } } ; \quad { \cal E } _ { \mathrm { T } } ^ { \prime } \in \mathbb { R } ^ { W \times F _ { \mathrm { Z } } } ; } \\ & { { \cal E } _ { \mathrm { T } , w , f } ^ { \prime } = { \cal E } _ { \mathrm { T } , W _ { \operatorname* { m a x } } - W + w , f } } \\ & { \forall w \in \{ 1 , \dots , W \} \forall f \in \{ 1 , \dots , F _ { \mathrm { Z } } \} ; } \end{array}\tag{34}
$$

A linear map lifts the input to the model width, and the temporal and spatial embeddings are added at every road and every position, respectively.

$$
\begin{array} { r l } & { { \pmb W } _ { \mathrm { X } } \in \mathbb { R } ^ { F _ { \mathrm { X } } \times F _ { \mathrm { Z } } } ; \quad { \pmb b } _ { \mathrm { X } } \in \mathbb { R } ^ { F _ { \mathrm { Z } } } ; } \\ & { { \pmb Z } ^ { [ 0 ] } = { \pmb X } { \pmb W } _ { \mathrm { X } } + { \pmb b } _ { \mathrm { X } } + { \pmb E } _ { \mathrm { T } } ^ { \prime } + { \pmb E } _ { \mathrm { S } } ^ { \prime } \in \mathbb { R } ^ { B \times W \times R \times F _ { \mathrm { Z } } } ; } \end{array}\tag{35}
$$

The transformer stacks $N$ pre-norm layers [45], each wrapping a GERASTSA block and a feedforward network in residual connections, and $Z ^ { [ n - 1 / 2 ] }$ is the residual stream halfway through layer n.

$$
\begin{array} { r l } & { Z ^ { [ n - 1 / 2 ] } = Z ^ { [ n - 1 ] } } \\ & { \phantom { { \sum } } + \mathrm { G E R A S T S A B } ^ { [ n ] } \Big ( \mathrm { L N } _ { \mathrm { A } } ^ { [ n ] } \big ( Z ^ { [ n - 1 ] } ; \varepsilon _ { \mathrm { L N } } \big ) , M \Big ) , } \\ & { Z ^ { [ n ] } = Z ^ { [ n - 1 / 2 ] } } \\ & { \phantom { { \sum } } + \mathrm { F F N } ^ { [ n ] } \Big ( \mathrm { L N } _ { \mathrm { F } } ^ { [ n ] } \big ( Z ^ { [ n - 1 / 2 ] } ; \varepsilon _ { \mathrm { L N } } \big ) ; F _ { \mathrm { F } } \Big ) } \\ & { \phantom { { \sum } } \in \mathbb { R } ^ { B \times W \times R \times F _ { \mathrm { Z } } } \forall n \in \{ 1 , \dots , N \} ; } \end{array}\tag{36}
$$

A final layer normalization and a linear map produce the output (Fig. 6).

$$
\begin{array} { r l } & { \pmb { W } _ { \mathrm { Y } } \in \mathbb { R } ^ { F _ { \mathrm { Z } } \times F _ { \mathrm { Y } } } ; \quad \pmb { b } _ { \mathrm { Y } } \in \mathbb { R } ^ { F _ { \mathrm { Y } } } ; } \\ & { Y = \mathrm { L N } _ { \mathrm { Y } } ( Z ^ { [ N ] } ; \varepsilon _ { \mathrm { L N } } ) \pmb { W } _ { \mathrm { Y } } + \pmb { b } _ { \mathrm { Y } } \in \mathbb { R } ^ { B \times W \times R \times F _ { \mathrm { Y } } } ; } \\ & { \mathrm { G E R A S T S A T } ( X ^ { \langle 1 \rangle } , \ldots , X ^ { \langle B \rangle } ) = Y ; } \end{array}\tag{37}
$$

Valid tokens never attend to padding tokens, every operation acts on each sample or token separately, and the temporal embedding is aligned to the last time step. Hence, the outputs of a sample at its valid positions are those it would receive if processed alone, regardless of padding and of the other samples; the outputs at padding positions are ignored.

No parameter belongs to a particular road: the same weights act on the tokens of all roads, which the model tells apart only by their input features, $\Xi ,$ , and $E _ { \mathrm { S } } ^ { \prime } .$ . Re-indexing the roads therefore permutes the outputs accordingly, and the output of each road does not depend on ι. Since no parameter shape depends on R or on $G ,$ the model can also be applied to other road networks, although our experiments use a single one.

![](images/84c6daf64153e812898da967723de5a7a7e8ee8a56f7b294764cc0ea59f25d6f.jpg)  
Fig. 6. The GERASTSA transformer. The histories $X ^ { \langle k \rangle }$ are left-padded to the longest window length W, and the mask M of (32) is built from their window lengths $W _ { k } .$ The embedded input $Z ^ { [ 0 ] }$ passes through N pre-norm layers, each mapping $Z ^ { [ n - 1 ] }$ to $Z ^ { [ n ] }$ with a GERASTSA block and a SwiGLU feedforward network in residual connections; a final layer normalization and a linear map produce the output Y.

## V. EXPERIMENTAL DESIGN

## A. Environment

We simulate traffic with a vectorized mesoscopic traffic simulator that runs a batch of environments in parallel. Each environment simulates a sequence of episodes, single runs of the problem over the time steps $\breve { T } ,$ , and starts the next one automatically once an episode reaches the horizon T. All vehicles are homogeneous CAVs that exchange information with the road infrastructure through vehicle-to-infrastructure communication. The simulator implements the problem of Section III: it applies the joint action of the road controllers, resolves the lane proposals with Algorithm 1, and completes the clearing of a road once the vehicles on its clearing lanes have left.

Our network is a $6 \times 6$ grid of junctions with 60 roads between horizontally and vertically adjacent junctions, every road oriented forward eastward or southward. The roads of each row and column have 2, 4, or 6 lanes, with nominal speed limits of 40, 60, and 70 km/h, respectively, so that 30 roads have two lanes, 20 four, and 10 six, 200 lanes in total.

The blocks lengthen from 300 to 350 m near the north-west corner to 650 m near the south-east corner. Every road of the grid lies on a cycle, so the network has no bridge roads and the bridge indicator of Table II is zero throughout; we keep the feature so that the model applies to other networks. A commercial zone occupies the north-west corner of the grid and a residential zone the south-east corner, and on top of a background demand, the commutes between them cause a morning rush hour toward the commercial zone, peaking at 08:00, and an evening rush hour back, peaking at 17:30. An episode simulates one day, from midnight to midnight, with a decision step every 600 s, so $T = 1 4 4$ . In fine-tuning and evaluation, every episode starts with the lanes of each road split evenly between its two directions and every speed limit nominal.

Road r has length $D _ { r } ,$ and each of its lanes holds at most $C _ { r }$ vehicles, its per-lane capacity; the hyperparameter $L _ { \mathrm { m a x } }$ bounds the total lanes of every road. Table I lists this and all other hyperparameters of our experiments. The simulator describes the traffic in each direction of a road by three quantities, computed over the active lanes of the direction. The congestion $c _ { r , d } ^ { ( t ) }$ is the number of vehicles on the mostoccupied active lane, and the density $\bar { c } _ { r , d } ^ { ( t ) }$ the mean number of vehicles per active lane, both relative to $C _ { r }$ . The normalized velocity $\bar { v } _ { r , d } ^ { ( t ) }$ is the mean velocity of the vehicles divided by the nominal speed limit. All three are 0 for a direction without active lanes, and the normalized velocity is 1 for a direction with active lanes but no vehicles, which flows freely.

Vehicles route themselves with knowledge of the current traffic: the cost of a route accounts for the congestion and the mean velocity of every road direction along it. Whenever a vehicle reaches a junction, it checks two conditions and, if either holds, reroutes from that junction along an optimal route; otherwise, it keeps its route. The first is that the next road of its remaining route has no open lane in the direction of the route, as when that direction has been closed. The second is that an optimal route costs less than $1 - \varepsilon _ { \mathrm { R } }$ times the remaining route, where $\varepsilon _ { \mathrm { R } }$ is the rerouting threshold. We chose the threshold arbitrarily; it only keeps vehicles from oscillating between routes of nearly equal cost. The first condition never strands a vehicle: since the navigable graph is strongly connected at every time step (Proposition 1), a vehicle that must reroute because a direction has been closed always has a route to the junction of its destination.

Every road is described by the 18 input features of Table II. The five features in the first four rows are static, and the others change over time. Per-direction features are listed forward first. The target and active lanes enter once each, since their backward counterparts are $L _ { r }$ minus the forward ones.

Concatenating the features yields the feature vector $x _ { r } ^ { ( t ) }$ , so the input width of Section IV is $F _ { \mathrm { X } } = 1 8$ . A sample ends at a decision step or, for the bootstrap value of Section V-C3, at the final time step $T ,$ , and its $W _ { k }$ time steps (Section IV-B) lie within its episode.

TABLE I  
HYPERPARAMETERS OF THE EXPERIMENTS.
<table><tr><td>Symbol</td><td>Hyperparameter</td><td>Value</td></tr><tr><td colspan="3">Environment</td></tr><tr><td> $L _ { \mathrm { m a x } }$ </td><td>Maximum total lanes</td><td>6</td></tr><tr><td> $\varepsilon _ { \mathrm { R } }$ </td><td>Rerouting threshold</td><td>0.1</td></tr><tr><td colspan="3">GERASTSA transformer</td></tr><tr><td> $F _ { \mathrm { Y } }$ </td><td>Output width</td><td>128</td></tr><tr><td> $F _ { \mathrm { { Z } } }$ </td><td>Model width</td><td>128</td></tr><tr><td> $F _ { \mathrm { F } }$ </td><td>Feedforward width</td><td>176</td></tr><tr><td> $H$ </td><td>Heads</td><td>8</td></tr><tr><td> $N$ </td><td>Layers</td><td>4</td></tr><tr><td> $W _ { \mathrm { m a x } }$ </td><td>Maximum window length</td><td>16</td></tr><tr><td> $\Lambda _ { \mathrm { m a x } }$ </td><td>Maximum relation distance</td><td>6</td></tr><tr><td> $\Gamma _ { \mathrm { m a x } }$ </td><td>Maximum junction degree</td><td>6</td></tr><tr><td> $\varepsilon _ { \mathrm { L N } }$ </td><td>Layer-normalization stabilizer</td><td> $1 0 ^ { - 5 }$ </td></tr><tr><td colspan="3">Pretraining</td></tr><tr><td> $\mathcal { H }$ </td><td>Horizon set</td><td>{1, 3, 6}</td></tr><tr><td colspan="3">Reward</td></tr><tr><td> $\beta _ { \mathrm { T } }$ </td><td>Trip credit</td><td>600 s</td></tr><tr><td> $\alpha _ { \mathrm { S } }$ </td><td>Own weight</td><td>0</td></tr><tr><td> $\alpha _ { \mathrm { { N } } }$ </td><td>Neighbor weight</td><td>0.3</td></tr><tr><td> $\alpha _ { \mathrm { G } }$ </td><td>Network weight</td><td>0.7</td></tr><tr><td colspan="3"> $P o l i c y$ </td></tr><tr><td> $\sigma _ { \mathrm { m i n } }$ </td><td>Minimum speed standard deviation</td><td>0.05</td></tr><tr><td> $\sigma _ { \mathrm { m a x } }$ </td><td>Maximum speed standard deviation</td><td>0.25</td></tr><tr><td colspan="3">PPO</td></tr><tr><td> $\gamma _ { \mathrm { R } }$ </td><td>Discount factor</td><td>0.993</td></tr><tr><td> $\lambda _ { \mathrm { A } }$ </td><td>GAE parameter</td><td>0.95</td></tr><tr><td> $\varepsilon _ { \mathrm { P } }$ </td><td>Clipping parameter</td><td>0.2</td></tr><tr><td> $c _ { \mathrm { V } }$ </td><td>Value coefficient</td><td>0.5</td></tr><tr><td> $c _ { \mathrm { L } }$ </td><td>Lane entropy coefficient</td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>cU</td><td>Speed entropy coefficient</td><td> $1 0 ^ { - 3 }$ </td></tr></table>

TABLE II  
INPUT FEATURES OF ROAD r AT TIME STEP t.
<table><tr><td>Feature</td><td>Value</td><td>Count</td></tr><tr><td>Road length</td><td> $D _ { r } / \operatorname* { m a x } _ { r ^ { \prime } \in \mathcal { R } } D _ { r ^ { \prime } }$ </td><td>1</td></tr><tr><td>Total lanes</td><td> $L _ { r } / L _ { \mathrm { m a x } }$ </td><td>1</td></tr><tr><td>Nominal speed limit</td><td> $s _ { r , d } ^ { ( 0 ) } / \operatorname* { m a x } _ { ( r ^ { \prime } , d ^ { \prime } ) \in \mathcal { R } \times \mathcal { D } } s _ { r ^ { \prime } , d ^ { \prime } } ^ { ( 0 ) }$ </td><td>2</td></tr><tr><td>Bridge indicator</td><td> $\mathbb { 1 } ( r \in \mathcal { R } _ {  } )$ </td><td>1</td></tr><tr><td>Congestion</td><td> $\mathbf { \Omega } _ { c } ( t )$  cr,d</td><td>2</td></tr><tr><td>Density</td><td> $\bar { c } ^ { ( \dot { t } ) } .$   $c _ { r , d }$ </td><td>2</td></tr><tr><td>Normalized velocity</td><td> $\therefore ( t )$   $v _ { r . d } ^ { ( \iota ) }$ </td><td>2</td></tr><tr><td>Active normalized speed limit</td><td> $\mathbf { \chi } _ { ( t ) } ^ { r , d }$   $u _ { r , d } ^ { \scriptscriptstyle ( * ) }$ </td><td>2</td></tr><tr><td>Open lanes</td><td> $\bar { \ell } _ { r , d } ^ { ( t ) } / L _ { r }$ </td><td>2</td></tr><tr><td>Target lanes</td><td> $\hat { \ell } _ { r } ^ { ( t ) } / L _ { r }$ </td><td>1</td></tr><tr><td>Active lanes</td><td> $\ell _ { r } ^ { ( t ) } / L _ { r }$ </td><td>1</td></tr><tr><td>Clearing indicator</td><td> $\mathbb { 1 } \left( r \in \mathcal { R } _ { \approx } ^ { ( t ) } \right)$ </td><td>1</td></tr></table>

## B. Pretraining

Pretraining is a self-supervised, multi-horizon forecasting task. From the history of the network, the model forecasts the congestion, density, and normalized velocity of every road and direction several steps ahead. As in spatio-temporal traffic forecasting [38], [39], the task encourages the model to learn how traffic propagates through the network; unlike there, it also encourages the model to learn how traffic responds to lane allocations and speed limits.

The data come from episodes simulated under a random behavior policy. Each episode starts from randomly drawn target lanes, one-way roads included, that satisfy the strongconnectivity requirement in (8), with active lanes equal to target lanes and every active normalized speed limit nominal. At randomly chosen decision steps, a road proposes a random admissible lane count, random normalized speed limits for its open directions, or both; otherwise, it proposes its current target lanes and active normalized speed limits, which leaves them unchanged. Action resolution (Section III-E) processes every proposal, so the data also cover rejected and relaxed proposals.

The encoder is a GERASTSA transformer with 561,600 parameters that receives the 18 features; its maximum window length $W _ { \mathrm { m a x } }$ spans 150 min of history, so a sample with a full window has $W _ { \mathrm { m a x } } R = 9 6 0$ road–time tokens. We write $e _ { r } ^ { ( t ) }$ for its output at road r and the last window position of a sample ending at time step t. Only this position serves as the forecast origin, as in fine-tuning: in every layer, its tokens attend to all roads at all valid positions of the window, so the stacked layers can gather the spatial and temporal information of the whole window there.

For every horizon h in the horizon set H, a per-road regressor reg forecasts the forecast quantities $y _ { r } ^ { ( t ) }$ , the three traffic quantities of Section V-A in both directions of road r. Following the direct strategy of multi-step forecasting [46], it forecasts each horizon separately rather than feeding its own forecasts back, and it does so as a residual on the current values. Besides the encoder output, it receives the one-hot horizon indicator ${ \mathbb { 1 } } _ { h }$ and the action context $p _ { r } ^ { ( t , h ) }$ , which lists the normalized target lanes and the active normalized speed limits of road r at the time steps $t + 1 , \ldots , t + h .$ , padded with zeros to the longest horizon. These result from the actions at the decision steps $t , \dots , t + h - 1$ after action resolution, so the forecasts are conditioned on the actions that take effect in between. A paired superscript (t, h) marks a quantity of the forecast from time step t at horizon h. Encoder and regressor are trained jointly to minimize the mean squared error over the six forecast quantities and over all samples of a batch, roads, and horizons whose forecast time step $t + h$ lies within the same episode.

$$
\hat { y } _ { r } ^ { ( t , h ) } = y _ { r } ^ { ( t ) } + \mathrm { r e g } \left( e _ { r } ^ { ( t ) } , p _ { r } ^ { ( t , h ) } , \mathbb { 1 } _ { h } \right)
$$

$$
\forall r \in \mathcal { R } \ \forall h \in \mathcal { H } ; \ \forall t \in \{ 0 , . . . , T - h \} ;\tag{38}
$$

$$
J _ { \mathrm { F } } = \hat { \mathbb { E } } _ { t , r , h } \left[ \frac { 1 } { 6 } \Big \| \boldsymbol { \hat { y } } _ { r } ^ { ( t , h ) } - { y } _ { r } ^ { ( t + h ) } \Big \| ^ { 2 } \right] ;
$$

After pretraining, the regressor is discarded, and the encoder initializes the fine-tuning.

## C. Fine-Tuning

Fine-tuning is an RL task in which the model learns to propose lane counts and normalized speed limits that maximize the return.

1) Reward: Over decision step t, road r completes $m _ { r } ^ { ( t ) }$ trips and accrues the delay $z _ { r } ^ { ( t ) }$ : the time spent by vehicles waiting for admission at the road and the time lost by vehicles traveling on it below its nominal speed limit. Its reward $g _ { r } ^ { ( t ) }$ credits each completed trip with $\beta _ { \mathrm { T } } > 0$ and charges the delay. The reward that the road optimizes, its mixed reward $g _ { \mathrm { M } , r } ^ { ( t ) ^ { \star } } ,$ combines its own reward, the mean reward of its neighbors in the controller graph, and the mean reward of the whole network with nonnegative weights $\alpha _ { \mathrm { S } } , \alpha _ { \mathrm { N } }$ , and $\alpha _ { \mathrm { G } }$ that sum to 1. It resembles the spatially discounted rewards of multiagent traffic signal control [30], which weight the reward of every agent by a power of its distance, but it keeps a fixed share for the whole network.

$$
\begin{array} { l } { \displaystyle { g _ { r } ^ { ( t ) } = \beta _ { \mathrm { T } } m _ { r } ^ { ( t ) } - z _ { r } ^ { ( t ) } } , } \\ { \displaystyle { g _ { \mathrm { M } , r } ^ { ( t ) } = \alpha _ { \mathrm { S } } g _ { r } ^ { ( t ) } + \frac { \alpha _ { \mathrm { N } } } { | \mathcal { N } _ { 1 } ( r ) | } \sum _ { r ^ { \prime } \in \mathcal { N } _ { 1 } ( r ) } { g _ { r ^ { \prime } } ^ { ( t ) } } } } \\ { \displaystyle { + \frac { \alpha _ { \mathrm { G } } } { R } \sum _ { r ^ { \prime } \in \mathcal { R } } { g _ { r ^ { \prime } } ^ { ( t ) } \forall r \in \mathcal { R } \forall t \in \mathcal { T } } } ; } \end{array}\tag{39}
$$

Because roads may become one-way, we keep the own weight $\alpha _ { \mathrm { S } }$ close to zero. Closing a direction sheds its traffic onto other roads: the road’s own reward can rise, since fewer vehicles travel on it and accrue delay there, while the network as a whole can lose. With a large own weight, a road could profit from this, which could push the policy toward one-way roads. Weighting the neighbors and the network makes a road bear part of the cost of the traffic it sheds. In our experiments, $\alpha _ { \mathrm { { S } } } ~ = ~ 0$ (Table I), so a road’s own reward still enters its mixed reward, but only through the network mean, with weight $\alpha _ { \mathrm { G } } / R = 0 . 7 / 6 0$ , since $\mathcal { N } _ { 1 } ( r )$ does not contain r.

2) Policy and Value Networks: The actor and the critic are separate networks, each consisting of a GERASTSA transformer and per-road heads, and both transformers are initialized from the pretrained encoder. The heads act on the outputs at the last time step of a sample (Section $\mathrm { V } { \mathrm { - } } \mathbf { A } )$ . The critic has a value head that yields the value $V _ { r } ^ { ( t ) }$ of road $r .$ The actor has a lane head and, per direction, two speed heads.

The lane head outputs one logit for every lane count from 0 to $L _ { \mathrm { m a x } } ,$ the proposed target lanes of the forward direction; the backward direction receives the remaining lanes. Since $L _ { \mathrm { m a x } }$ is a fixed hyperparameter, the lane head has a fixed output size, independent of the road and the network. Lane counts outside the admissible set of (13) are masked out of the softmax, as in invalid action masking [35], which yields the lane distribution $\pi _ { \mathrm { L } , r } ^ { ( t ) }$ over the admissible lane counts.

The speed heads of direction d output a mean offset and a scale logit. The proposed normalized speed limit follows the speed distribution $\pi _ { \mathrm { U } , r , d } ^ { ( t ) } ,$ the truncation to [0, 1] of a normal distribution whose mean is 1 plus the offset and whose standard deviation the sigmoid of the scale logit places between the bounds $\sigma _ { \mathrm { m i n } }$ and $\sigma _ { \mathrm { m a x } } .$ Since the offsets of the untrained speed heads are close to zero, shifting the mean by 1 keeps speed limits close to nominal early in fine-tuning, which is meant to stabilize its first updates.

During fine-tuning, the proposed lanes and normalized speed limits are sampled from these distributions; during execution, the modes are taken. By (14), a direction without open lanes keeps its active normalized speed limit, so its speed heads are ignored. The log-probability of the action $a _ { r } ^ { ( \dot { t } ) }$ of road r thus sums the log-probabilities of its proposed lanes and of the proposed normalized speed limits of its open directions ${ \mathcal O } _ { r } ^ { ( t ) }$ only.

$$
\begin{array} { r l } & { \log \pi _ { r } ^ { ( t ) } \left( { a _ { r } ^ { ( t ) } } \right) = \log \pi _ { \mathrm { L } , r } ^ { ( t ) } \left( \tilde { \ell } _ { r } ^ { ( t ) } \right) } \\ & { + \displaystyle \sum _ { d \in \mathcal { O } _ { r } ^ { ( t ) } } \log \pi _ { \mathrm { U } , r , d } ^ { ( t ) } \left( \tilde { u } _ { r , d } ^ { ( t ) } \right) \forall r \in \mathcal { R } \forall t \in \mathcal { T } ; } \end{array}\tag{40}
$$

3) Per-Road PPO: We train actor and critic with PPO [16], applied per road. As in independent and multi-agent PPO [33], [34], every road acts as an agent with its own probability ratio and its own advantages, while all roads share the parameters. Unlike in these methods, our actor observes the whole network (Section II-D).

Each update collects a rollout of a fixed number of consecutive decision steps from every environment. A rollout may span the end of one episode and the start of the next. The advantages $\hat { A } _ { r } ^ { ( t ) }$ follow truncated generalized advantage estimation (GAE) [16], [47], computed from the mixed rewards of the road and its values under the critic $V _ { \mathrm { o l d } , r } ^ { ( t ) }$ that collected the data, with discount factor $\gamma _ { \mathrm { R } }$ and $\mathrm { G A E }$ parameter $\lambda _ { \mathrm { A } }$ . A rollout ends at its last decision step and at the last decision step of every episode it contains; there, the recursion stops and bootstraps from the critic’s value at the next time step. Since traffic continues beyond the horizon, the end of an episode is a timelimit truncation rather than a termination [48]. The discount factor gives an effective horizon of about $1 / ( 1 - \gamma _ { \mathrm { R } } ) \approx 1 4 3$ decision steps, roughly one day. The return estimates $\hat { V } _ { r } ^ { ( t ) }$ add the values back to the advantages.

$$
\begin{array} { r l } & { \delta _ { r } ^ { ( t ) } = g _ { \mathrm { M } , r } ^ { ( t ) } + \gamma _ { \mathrm { R } } V _ { \mathrm { o l d } , r } ^ { ( t + 1 ) } - V _ { \mathrm { o l d } , r } ^ { ( t ) } , } \\ & { \hat { A } _ { r } ^ { ( t ) } = \left\{ \begin{array} { l l } { \delta _ { r } ^ { ( t ) } } & { \mathrm { ~ i f ~ } t \mathrm { ~ e n d s ~ t h e ~ r o l l o u t } } \\ { \delta _ { r } ^ { ( t ) } + \gamma _ { \mathrm { R } } \lambda _ { \mathrm { A } } \hat { A } _ { r } ^ { ( t + 1 ) } } & { \mathrm { o t h e r w i s e } } \end{array} \right. , } \\ & { \hat { V } _ { r } ^ { ( t ) } = \hat { A } _ { r } ^ { ( t ) } + V _ { \mathrm { o l d } , r } ^ { ( t ) } \ \forall r \in \mathcal { R } \ \forall t \in \mathcal { T } ; } \end{array}\tag{41}
$$

The probability ratio $\zeta _ { r } ^ { ( t ) }$ compares the probability of the action of road $r$ under the current policy with that under the policy that collected the data.

$$
\zeta _ { r } ^ { ( t ) } = \frac { \pi _ { r } ^ { ( t ) } \Big ( \boldsymbol { a } _ { r } ^ { ( t ) } \Big ) } { \pi _ { \mathrm { o l d } , r } ^ { ( t ) } \Big ( \boldsymbol { a } _ { r } ^ { ( t ) } \Big ) } \ \forall r \in \mathcal { R } \ \forall t \in \mathcal { T } ;\tag{42}
$$

It enters the clipped surrogate objective $J _ { \mathrm { P } }$ through the helper function $\chi ,$ which clips it with the clipping parameter $\varepsilon _ { \mathrm { P } }$ , and the critic regresses the return estimates through the squared

error $J _ { \mathrm { V } } ;$ both average over the decision steps of a minibatch and all roads.

$$
\begin{array} { r l } & { \chi : \mathbb { R } _ { > 0 } \times \mathbb { R } \to \mathbb { R } ; } \\ & { \chi ( x , y ) = \operatorname* { m i n } ( x y , \operatorname { c l i p } ( x , 1 - \varepsilon _ { \mathrm { P } } , 1 + \varepsilon _ { \mathrm { P } } ) y ) ; } \\ & { J _ { \mathrm { P } } = \hat { \mathbb { E } } _ { t , r } \Big [ \chi \Big ( \zeta _ { r } ^ { ( t ) } , \hat { A } _ { r } ^ { ( t ) } \Big ) \Big ] ; } \\ & { J _ { \mathrm { V } } = \hat { \mathbb { E } } _ { t , r } \Big [ \Big ( V _ { r } ^ { ( t ) } - \hat { V } _ { r } ^ { ( t ) } \Big ) ^ { 2 } \Big ] ; } \end{array}\tag{43}
$$

As in PPO, the combined objective J adds the mean entropies $\mathbb { H } _ { \mathrm { L } }$ and H<sub>U</sub> of the lane and speed distributions as a bonus [49], the latter over open directions only, and weights the terms with the coefficients $c _ { \mathrm { V } } , c _ { \mathrm { L } }$ , and $c _ { \mathrm { U } }$ . The objective is maximized.

$$
J = J _ { \mathrm { P } } - c _ { \mathrm { V } } J _ { \mathrm { V } } + c _ { \mathrm { L } } \mathbb { H } _ { \mathrm { L } } + c _ { \mathrm { U } } \mathbb { H } _ { \mathrm { U } } ;\tag{44}
$$

As the networks are separate, $J _ { \mathrm { V } }$ trains only the critic, and the other terms train only the actor.

## VI. RESULTS

We evaluate every policy on 32 days, 8 episodes for each of 4 seeds not used in training. A seed and an episode index determine the demand, which is therefore the same under every policy, so that comparisons between policies are paired by day. The learned policies take the modes of their distributions (Section V-C2). We report the return, the reward $g _ { r } ^ { ( t ) }$ summed over all roads and decision steps of a day, in vehicle-hours (veh·h), so that each completed trip counts as $\beta _ { \mathrm { T } } ~ = ~ 6 0 0$ s of avoided delay; its two components, the delay and the number of completed trips; the mean travel time of a trip from its request to its arrival and the mean admission wait it spends at its origin road; and the number of closing proposals that action resolution rejects. As a safeguard that is not part of the problem of Section III, the simulator removes every vehicle that has queued for 600 s, since a gridlocked lane could otherwise block clearing for the rest of the day; we report these removals as well. Values are means and standard errors over the 32 days, and changes against a reference are means and standard errors of the paired differences.

All variants share the same data and schedule. Pretraining runs for 30,000 steps of 32 samples on 3,584 days simulated under the behavior policy of Section V-B, and its validation error is measured on 512 further days. Fine-tuning runs for 100 PPO updates, each of which collects 8 decision steps from each of 1,024 environments, about 5,700 days in total. Before the first update, the environments run under the baseline of Section VI-B for up to one day, so that their clocks are spread uniformly over the day. Every 10 updates, the greedy policy, which takes the modes, is evaluated on 8 further days with fixed demand, which yields the learning curves below; their reference is the baseline on the same days. Each variant is trained once with the same seed, so the standard errors reflect the variation across evaluation days but not across training runs.

## A. Effect of Pretraining

Fig. 7 compares fine-tuning from the pretrained encoder with fine-tuning from an untrained, randomly initialized one, under otherwise identical settings. The pretrained encoder forecasts the traffic quantities with a validation MSE of 0.0143, which is 59%, 70%, and 77% below that of persistence, the forecast that nothing changes, at the horizons of 10, 30, and 60 min.

Panel (b) of Fig. 7 must be read with the warm-up in mind. When fine-tuning starts, each environment is in the middle of a day that began under the baseline, and a day spans 18 updates, so a training episode that ends at update $k \leq 1 8$ was controlled by the baseline until about 8k decision steps before midnight and, after that, by the early policy, whose untrained heads still propose almost at random. The return of the training episodes thus stays at the level of the baseline while the policy controls only the quiet night, up to update 4, and falls in two steps as the policy takes over the evening rush hour, at updates 5 and 6, and the morning rush hour, at updates 12 to 14: its random reallocations cost little at night but cause queues and gridlock under heavy demand. Between the two steps, the return lies near zero, and from update 14 to 18, it bottoms out at about −9,000 veh·h with the pretrained encoder and −10,000 veh·h with the untrained one. This transient stems from the staggered start, not from a collapse of the policy, and the greedy policy shows no such dip (Fig. 7 (a)). Until about update 15, the two curves nearly coincide, since most of each finished day was controlled by the baseline after an identical warm-up, and the actor’s encoder stays frozen during the first two updates, the first of which trains only the critic. From update 18 on, panel (b) tracks learning, but under sampled actions and with a lag of about 9 updates, since an episode that ends at update k was controlled by the policies of updates $k - 1 7$ to k. The run with the pretrained encoder recovers past the reference line of the baseline at update 47, and the run with the untrained encoder at update 86; since the training episodes have other demand than the days of that line, these crossings are approximate.

The greedy policy shows the same gap earlier (Fig. 7 (a)). With the pretrained encoder, it surpasses the baseline, which attains about 14,500 veh·h on the same days, from update 20 on and varies only between 16,976 and 17,656 veh·h from update 30; with the untrained encoder, it first surpasses the baseline at update 50 and varies between 12,007 and 16,983 veh·h from update 30. On the evaluation days, the final policy from the pretrained encoder attains a return of 17,653 veh·h against 17,112 veh·h from the untrained encoder, which falls short by $3 . 1 \pm 0 . 2 \%$ , and it incurs 374 veh·h of delay against 911 veh·h (Table III). Pretraining thus roughly halves the number of updates needed to surpass the baseline and, after 100 updates, leaves less than half the delay.

## B. Learned Policy

The baseline holds the initial configuration of every episode throughout the day: every road keeps its lanes split evenly between its two directions, and every speed limit stays nominal. Table III lists the metrics of all policies. Against the baseline, the learned policy raises the return by $2 2 . 9 { \pm } 0 . 3 \%$ and reduces the delay by $8 9 . 5 \pm 1 . 3 \%$ , from 3,565 to 374 veh·h per day. The mean travel time falls from 5.20 to 3.52 min, by 32%, and the mean admission wait from 25 to 3 s. The learned policy completes 583 more trips per day, 0.5% more, exactly as many as the safeguard removes from gridlocked queues under the baseline, and never under the learned policy, so the gain in return stems almost entirely from the delay.

(a) Greedy policy on the evaluation days  
![](images/e074f9bb3e4d1d73cf8736ca26de6d602ab5999972c71f23b7ffaff5bbe09058.jpg)  
(b) Training episodes

![](images/b240455f96936655d91b1cc9d0cc1de789aaf2fa2d56654aa88010a83c36a492.jpg)  
Fig. 7. Return against the PPO update when fine-tuning from the pretrained encoder and from an untrained one. (a) Greedy policy on the 8 days evaluated during fine-tuning (shaded: standard error). (b) Mean return of the training episodes that end in each update, under sampled actions; until update 18 (shaded), these episodes began under the baseline of the warm-up, and the drops at updates 5 and 6 and at updates 12 to 14 mark the policy taking over the evening and the morning rush hour. Dotted: mean return of the baseline on the days of (a).

As a second baseline, the two-way load balancer reallocates lanes in response to the observed traffic but never makes a road one-way. At each decision step, it takes as the load of each direction of a road its density times its active lanes, i.e., its vehicles in units of $C _ { r } ,$ , averaged over the last few decision steps. Unless the road is clearing or carries fewer vehicles than a threshold, it proposes to split the lanes in proportion to the loads of the two directions, rounded and kept between 1 and $L _ { r } \mathrm { ~ - ~ } 1$ , whenever this split differs from the current target lanes by at least half a lane plus a margin; every speed limit stays nominal. A grid search on 32 days drawn from the seed of fine-tuning, not from the evaluation seeds, chose the averaging window among 1, 3, and 6 decision steps, the margin among 0, 0.25, and 0.5 lanes, and the threshold among 0.01, 0.05, and 0.1 times $C _ { r }$ , for the highest mean return. The load balancer raises the return by $7 . 3 \pm 0 . 2 \%$ and reduces the delay by $2 9 . 2 \pm 0 . 9 \%$ against the fixed configuration, to 2,524 veh·h, but the safeguard still removes 535 vehicles per day, and the mean admission wait rises from 25 to 39 s. The learned policy attains a 14.5% higher return than the load balancer, with 85% less delay.

Fig. 8 shows how these differences arise over the day. Under the baseline, 99% of the delay accrues in the two rush hours, between 06:00 and 11:00 and between 15:00 and 21:00. The delay per decision step peaks at 194 veh·h around 08:30 and at 347 veh·h around 18:10, and the vehicles on the network peak at 1,658 and 2,532 at the same times. Under the learned policy, the delay per decision step never exceeds 19 veh·h, and the vehicles on the network peak at about 900 in both rush hours, while more trips are completed per decision step near the peaks: the network discharges its demand as it arrives.

Fig. 9 shows where congestion builds up. Under the baseline at 08:00, the directions toward the commercial zone in the north-west are more congested than those away from it, 0.16 against 0.07 on average, and the most congested direction, at 0.995, is the westbound road at the residential corner in the south-east, where morning trips enter the network. At 17:30, congestion is heavier and lies in the directions away from the commercial zone, 0.26 against 0.08 on average, with 20 of them above 0.3. It is highest on the roads leaving the commercial corner, up to 0.77, and on roads near the residential corner, up to 0.73. The learned policy opens on average 155 of the 200 lanes of the network toward the commercial zone at 08:00 and 159 away from it at 17:30, and 30 and 35 roads, respectively, are one-way in the peak direction on all evaluation days. No direction then exceeds a congestion of 0.19.

Over the day, the learned policy makes about 40 of the 60 roads one-way during each rush hour and almost none around midday, when only the background demand remains (Fig. 8 (d)). It hardly lowers speed limits during the rush hours, where the mean normalized speed limit stays above 0.96, and lowers them slightly after each rush hour, to about 0.92 to 0.93 (Fig. 8 (e)). Action resolution rejects 161 of its closing proposals per day, at most about 5 per decision step (Fig. 8 (f)). Of these, 89 fall between midnight and 05:00, when the policy also keeps up to 34 roads one-way with almost no traffic, and 54 fall between 06:30 and 09:30, at the onset of the morning rush hour; between 16:00 and 19:30, only 4 are rejected.

## C. Restricted Policies

To see which actions the gain rests on, we evaluate the final full model with part of its control withheld at test time, without retraining. Under lane-only control, the policy proposes the lane allocations, and every speed limit stays nominal; under speed-only control, it sets the speed limits, and every road keeps its initial lane allocation. Under two-way control, the policy may not close a direction: in the relaxed variant, each of its proposals of 0 or $L _ { r }$ is replaced by its relaxed proposal as in (17), and in the masked variant, these two allocations are excluded from the lane distribution before its mode is taken. Two-way control is evaluated both with the policy’s speed limits and with nominal ones. These evaluations show what • Collapsed relations: the four adjacency relations in $\mathcal { D } ^ { 2 }$ are collapsed into a single 1-hop relation, so that $| \kappa | =$ $\Lambda _ { \operatorname* { m a x } } + 1$ and the bias depends on the hop distance alone, no longer on the directions in which adjacent roads flow into their shared junction.

TABLE III  
PERFORMANCE ON THE 32 EVALUATION DAYS: MEANS AND STANDARD ERRORS OVER THE DAYS, WITH CHANGES IN RETURN PAIRED BY DAY. LOAD BALANCER: THE HEURISTIC BASELINE OF SECTION VI-B; FULL MODEL: THE LEARNED POLICY; UNTRAINED ENCODER: THE FULL MODEL FINE-TUNED WITHOUT PRETRAINING; THE SIX ROWS FROM LANE-ONLY ON: THE FULL MODEL RESTRICTED AT TEST TIME (SECTION VI-C); THE LAST FIVE ROWS: THE ABLATIONS (SECTION VI-D). REMOVED: VEHICLES REMOVED FROM GRIDLOCKED QUEUES BY THE SAFEGUARD; REJECTED: CLOSING PROPOSALS REJECTED BY ACTION RESOLUTION, BOTH PER DAY.
<table><tr><td>Policy</td><td>Return  $( \mathrm { { 1 0 ^ { 3 } \ v e h { \cdot } h } ) }$ </td><td>Return change (%) vs. baseline vs. full model</td><td></td><td>Delay (veh·h)</td><td>Trips  $( 1 0 ^ { \hat { 3 } } )$ </td><td>Travel time (min)</td><td>Adm. wait Removed Rejected (s)</td><td></td><td></td></tr><tr><td>Baseline</td><td> $1 4 . 3 6 \pm 0 . 0 4$ </td><td></td><td> $- 1 8 . 6 \pm 0 . 3$ </td><td> $3 5 6 5 \pm 4 3$ </td><td> $1 0 7 . 5 8 \pm 0 . 0 7$ </td><td> $5 . 2 0 \pm 0 . 0 2$ </td><td> $2 5 . 4 \pm 0 . 6$ </td><td> $5 8 3 \pm 2 3$ </td><td>0</td></tr><tr><td>Load balancer</td><td> $1 5 . 4 1 \pm 0 . 0 3$ </td><td> $+ 7 . 3 \pm 0 . 2$ </td><td> $- 1 2 . 7 \pm 0 . 2$ </td><td> $2 5 2 4 \pm 2 9$ </td><td> $1 0 7 . 6 2 \pm 0 . 0 7$ </td><td> $4 . 6 1 \pm 0 . 0 2$ </td><td> $3 9 . 0 \pm 0 . 7$ </td><td> $5 3 5 \pm 1 7$ </td><td>0</td></tr><tr><td>Full model</td><td> $1 7 . 6 5 \pm 0 . 0 1$ </td><td> $+ 2 2 . 9 \pm 0 . 3$ </td><td></td><td> $3 7 4 \pm 9$ </td><td> $1 0 8 . 1 6 \pm 0 . 0 7$ </td><td> $3 . 5 2 \pm 0 . 0 1$ </td><td> $3 . 1 \pm 0 . 1$ </td><td>0</td><td> $1 6 1 \pm 2$ </td></tr><tr><td>Untrained encoder</td><td> $1 7 . 1 1 \pm 0 . 0 3$ </td><td> $+ 1 9 . 1 \pm 0 . 3$ </td><td> $- 3 . 1 \pm 0 . 2$ </td><td> $9 1 1 \pm 2 9$ </td><td> $1 0 8 . 1 4 \pm 0 . 0 7$ </td><td> $3 . 8 8 \pm 0 . 0 2$ </td><td> $6 . 7 \pm 0 . 8$ </td><td> $1 3 \pm 4$ </td><td> $2 1 1 \pm 1 6$ </td></tr><tr><td>Lane-only</td><td> $1 7 . 8 5 \pm 0 . 0 1$ </td><td> $+ 2 4 . 3 \pm 0 . 3$ </td><td> $+ 1 . 1 \pm 0 . 1$ </td><td> $1 7 4 \pm 7$ </td><td> $1 0 8 . 1 6 \pm 0 . 0 7$ </td><td> $3 . 4 1 0 \pm 0 . 0 0 5$ </td><td> $2 . 8 \pm 0 . 1$ </td><td>0</td><td> $1 5 4 \pm 1$ </td></tr><tr><td>Speed-only</td><td> $1 4 . 0 7 \pm 0 . 0 5$ </td><td> $- 2 . 0 \pm 0 . 4$ </td><td> $- 2 0 . 3 \pm 0 . 3$ </td><td> $3 8 4 6 \pm 5 0$ </td><td> $1 0 7 . 5 0 \pm 0 . 0 7$ </td><td> $5 . 3 5 \pm 0 . 0 3$ </td><td> $2 8 . 6 \pm 0 . 6$ </td><td> $6 5 9 \pm 2 1$ </td><td>0</td></tr><tr><td>Two-way relaxed</td><td> $1 5 . 1 0 \pm 0 . 0 3$ </td><td> $+ 5 . 1 \pm 0 . 3$ </td><td> $- 1 4 . 5 \pm 0 . 2$ </td><td> $2 8 4 3 \pm 3 4$ </td><td> $1 0 7 . 6 6 \pm 0 . 0 7$ </td><td> $4 . 8 2 \pm 0 . 0 2$ </td><td> $2 7 . 5 \pm 0 . 8$ </td><td> $4 9 9 \pm 1 8$ </td><td>0</td></tr><tr><td>Two-way masked</td><td> $1 5 . 1 3 \pm 0 . 0 5$ </td><td> $+ 5 . 3 \pm 0 . 3$ </td><td> $- 1 4 . 3 \pm 0 . 2$ </td><td> $2 8 1 5 \pm 4 1$ </td><td> $1 0 7 . 6 6 \pm 0 . 0 8$ </td><td> $4 . 8 1 \pm 0 . 0 2$ </td><td> $2 6 . 1 \pm 0 . 6$ </td><td> $4 9 8 \pm 1 9$ </td><td>0</td></tr><tr><td>Two-way relaxed, lane-only</td><td> $1 5 . 5 3 \pm 0 . 0 4$ </td><td> $+ 8 . 1 \pm 0 . 4$ </td><td> $- 1 2 . 0 \pm 0 . 2$ </td><td> $2 4 1 9 \pm 4 1$ </td><td> $1 0 7 . 6 8 \pm 0 . 0 7$ </td><td> $4 . 5 9 \pm 0 . 0 2$ </td><td> $2 5 . 2 \pm 0 . 9$ </td><td> $4 7 9 \pm 1 9$ </td><td>0</td></tr><tr><td>Two-way masked, lane-only</td><td> $1 5 . 5 8 \pm 0 . 0 3$ </td><td> $+ 8 . 5 \pm 0 . 2$ </td><td> $- 1 1 . 7 \pm 0 . 2$ </td><td> $2 3 6 8 \pm 3 2$ </td><td> $1 0 7 . 6 8 \pm 0 . 0 7$ </td><td> $4 . 5 6 \pm 0 . 0 2$ </td><td> $2 3 . 9 \pm 0 . 8$ </td><td> $4 7 5 \pm 1 6$ </td><td>0</td></tr><tr><td>No relational bias</td><td> $1 6 . 2 6 \pm 0 . 0 5$ </td><td> $+ 1 3 . 2 \pm 0 . 5$ </td><td> $- 7 . 9 \pm 0 . 3$ </td><td> $1 7 4 2 \pm 5 4$ </td><td> $1 0 8 . 0 0 \pm 0 . 0 7$ </td><td> $4 . 7 8 \pm 0 . 0 3$ </td><td> $5 . 8 \pm 0 . 2$ </td><td> $1 5 9 \pm 2 1$ </td><td> $6 7 0 \pm 1 4$ </td></tr><tr><td>Collapsed relations</td><td> $1 7 . 4 7 \pm 0 . 0 2$ </td><td> $+ 2 1 . 7 \pm 0 . 3$ </td><td> $- 1 . 0 \pm 0 . 1$ </td><td> $5 4 7 \pm 1 8$ </td><td> $1 0 8 . 1 3 \pm 0 . 0 7$ </td><td> $3 . 6 2 \pm 0 . 0 1$ </td><td> $2 . 2 \pm 0 . 1$ </td><td> $1 6 \pm 8$ </td><td> ${ 2 2 5 \pm 6 }$ </td></tr><tr><td>No spatial embedding</td><td> $1 6 . 1 5 \pm 0 . 0 7$ </td><td> $+ 1 2 . 4 \pm 0 . 6$ </td><td> $- 8 . 5 \pm 0 . 4$ </td><td> $1 8 6 2 \pm 6 5$ </td><td> $1 0 8 . 0 5 \pm 0 . 0 7$ </td><td> $4 . 6 8 \pm 0 . 0 4$ </td><td> $1 1 . 5 \pm 1 . 0$ </td><td> $8 7 \pm 2 4$ </td><td> $5 6 0 \pm 1 9$ </td></tr><tr><td>Undirected spatial emb.</td><td> $1 7 . 5 1 \pm 0 . 0 3$ </td><td> $+ 2 1 . 9 \pm 0 . 3$ </td><td> $- 0 . 8 \pm 0 . 2$ </td><td> $5 1 6 \pm 2 4$ </td><td> $1 0 8 . 1 6 \pm 0 . 0 7$ </td><td> $3 . 7 8 \pm 0 . 0 1$ </td><td> $2 . 8 \pm 0 . 1$ </td><td> $0 . 2 \pm 0 . 2$ </td><td> $1 7 1 \pm 4$ </td></tr><tr><td>No temporal embedding</td><td> $1 7 . 4 7 \pm 0 . 0 5$ </td><td> $+ 2 1 . 6 \pm 0 . 4$ </td><td> $- 1 . 0 \pm 0 . 3$ </td><td> $5 5 2 \pm 4 7$ </td><td> $1 0 8 . 1 4 \pm 0 . 0 7$ </td><td> $3 . 8 6 \pm 0 . 0 3$ </td><td> $8 . 0 \pm 1 . 4$ </td><td> $5 \pm 4$ </td><td> $3 8 3 \pm 9$ </td></tr></table>

• No spatial embedding: $E _ { \mathrm { S } } ^ { \prime }$ is dropped from (35), so the junction degrees of a road no longer enter the model • No relational bias: $B _ { \mathrm { R } } ^ { \prime }$ is dropped from the attention scores in (27), so attention is not informed by any relation between roads, and roads are told apart only by their input features and the spatial embedding.

the trained policy retains under each restriction, not what a policy trained under it could reach.

Each ablation removes one component of the GERASTSA transformer (Section IV) and is pretrained and fine-tuned in the same way as the full model. Fig. 10 compares their learning curves and pretraining errors with those of the full model, and Table III their final performance.

Lane-only control does better than the full model: it raises the return by $1 . 1 { \pm } 0 . 1 \%$ and reduces the delay by $5 3 . 5 { \pm } 2 . 7 \%$ from 374 to 174 veh·h per day. Speed-only control does worse than the baseline, lowering the return by $2 . 0 { \pm } 0 . 4 \%$ and raising the delay by $7 . 9 \pm 1 . 5 \%$ . Under two-way control, too, the policy’s speed limits lower the return: with them, the relaxed and the masked variant raise the return against the baseline by $5 . 1 \pm 0 . 3 \%$ and $5 . 3 \pm 0 . 3 \%$ , and with nominal speed limits by $8 . 1 \pm 0 . 4 \%$ and $8 . 5 \pm 0 . 2 \%$ . All four two-way variants thus fall far short of the 22.9% of the full model and bracket the 7.3% of the load balancer; they leave between 2,368 and 2,843 veh·h of delay, and the safeguard removes between 475 and 499 vehicles per day.

## D. Ablations

directly, and roads are told apart only by their input features and the relational bias.

Undirected spatial embedding: instead of embedding the junction degree of a road in each direction through the directional tables $E _ { \mathrm { S } } ^ { d }$ of (33), the spatial embedding selects a single row by the total degree of the road, the number of roads that meet it at either endpoint, i.e., its degree in the controller graph $L ( G )$ . It thus no longer tells how these roads are split between the two head junctions of the road.

• No temporal embedding: $E _ { \mathrm { T } } ^ { \prime }$ is dropped from (35), so temporal order enters only through the causal mask of (32).

All ablations still improve on the baseline, by 12.4% to 21.9% in return, but they fall into two groups. Removing the relational bias or the spatial embedding lowers the return by $7 . 9 { \pm } 0 . 3 \%$ and 8.5±0.4% against the full model and multiplies the delay by about 4.7 and 5.0, to 1,742 and 1,862 veh·h. Both variants leave vehicles in gridlocked queues, 159 and 87 per day, and have far more closing proposals rejected, 670 and 560 per day against 161. Their greedy policies stay below that of the full model at every evaluation from update 30 on. Without the relational bias, the encoder also forecasts worst, with a validation MSE 7.9% above that of the full model, whereas without the spatial embedding, it forecasts as well as the full model (Fig. 10 (b)).

The other three ablations, collapsed relations, the undirected spatial embedding, and no temporal embedding, end within about 1% of the full model in return, at $- 1 . 0 \pm 0 . 1 \%$ $- 0 . 8 \pm 0 . 2 \% .$ and $- 1 . 0 \pm 0 . 3 \%$ , but their final policies incur 1.5, 1.4, and 1.5 times its delay, 547, 516, and 552 against 374 veh·h per day. Across the evaluation days, these differences far exceed their standard errors, but with one training run per variant, they also depend on when training stopped: from update 30 on, the greedy return of the full model varies by 680 veh·h, about 4%, more than these variants lose at the end. Over these earlier evaluations, the three variants differ. Without the temporal embedding, the greedy return lies below that of the full model at 6 of the 8 evaluations, by 1.0% on average, and the forecasting error lies above it at every validation from step 1,000 on, ending 2.1% higher (Fig. 10 (b)). With collapsed relations, by contrast, the greedy return exceeds that of the full model at 7 of the 8 evaluations, by 1.2% on average, and with the undirected spatial embedding at 5 of them, and both variants forecast as well as the full model, within about 0.5%.

![](images/096ec0fdc9408f626c779b3ee17fa0082dcfb2ce018a9ce94acc15bcb1cfeb09.jpg)

![](images/3cea30dad607393dd09aab20b8946dd24c5276dd122c0d14ecbc47f758d15d7a.jpg)

![](images/449b17d84f63c2d02eccd01b9c1f20dacae6b44fc0f986ed2c071d374b220a82.jpg)

![](images/47279f18cf75eacf7a4081c876ba085fd372278b2983cc1153e35cc644710ac9.jpg)

![](images/4eb3705db2512109288a6a37eaad1af8b5ec8c98144084d462c01dd9c34565f2.jpg)

![](images/255eef320b6bae2ba895d0fc3496900c5764b3e4e78142449a149bd8f579ae3f.jpg)  
Fig. 8. Metrics per decision step over the day under the baseline and the learned policy, averaged over the 32 evaluation days: (a) delay, (b) completed trips (c) vehicles on the network, (d) one-way roads, (e) mean normalized speed limit over all roads and directions, and (f) closing proposals rejected by action resolution. Dotted verticals mark 08:00 and 17:30, the times shown in Fig. 9.

C  
![](images/2581416b1c9b4ca3667fe0b1bb11bcc8e46213f529e514aafac9b6b0aac97bb7.jpg)

![](images/2469aa6cd3eecb0f4429d534bfe55440170600988df9134f9e1c1986654fbef7.jpg)  
C

![](images/c826156562dfe1d409b5f2a602c9f1df7a70f67dd0b8b5fcbb7dd292b44259f6.jpg)

![](images/57743782d16981c573380d46ba7f1ca6c6393f0370b37a66441e3eeb2b3a8c7d.jpg)  
Fig. 9. Congestion and open lanes of every road and direction at 08:00 and 17:30 under the baseline and the learned policy, averaged over the 32 evaluatio days. Each direction of a road is drawn on its right-hand side, with its luminance giving the congestion and its width the open lanes, so that a one-way road shows a single stroke. North is up; C and R mark the corners of the commercial and the residential zone.

The final evaluation thus suggests that all three components reduce delay, but only for the temporal embedding do the earlier evaluations and pretraining agree.

## VII. DISCUSSION

Pretraining likely helps because the forecasting task asks the encoder for what both the actor and the critic need: how the congestion, density, and velocity of every road will evolve over the next hour, given the lane allocations and speed limits that take effect in between. The forecasts of a road depend on the roads that feed and drain it, so to forecast well, the encoder must represent how traffic propagates through the network and how it responds when a direction is closed or slowed, and it forecasts far better than persistence, with errors 59% to 77% below it. In fine-tuning, by contrast, an action reaches the reward only indirectly and late: closing a direction diverts the traffic headed for it at once, but its lanes open in the other direction only once clearing completes, queues build and spill back over several decision steps, and with $\alpha _ { \mathrm { { S } } } = 0 .$ , the mixed reward of a road is dominated by the actions of other roads. Learning the same representation from this signal alone is slow, as the untrained encoder shows: it needed about twice as many updates to surpass the baseline and left more than twice the delay after 100 updates. The initial dip of the training episodes is alike in both runs (Fig. 7 (b)) and says nothing about the encoders: it is a transient of the staggered start, during which the actor’s encoder is frozen or has barely been trained. Both runs used the same fine-tuning schedule, which was chosen for a pretrained encoder: it freezes the encoders during the first updates and then trains them at a lower learning rate than the heads. A schedule tuned for random initialization might narrow the gap.

<table><tr><td>—0 Full model</td><td> Collapsed relations</td><td></td><td>Undirected spatial embedding</td></tr><tr><td>-- No relational bias</td><td>…No spatial embedding</td><td>· No temporal embedding</td><td></td></tr></table>

![](images/4d37846e66e1c81b15caba6d777ca4b65e3852fb346a5b1430234f604742e152.jpg)

![](images/22452ca8d0f58c9bd9e7212ba505765a6e6d87a53edde6c99b275396edbf6e7d.jpg)  
Fig. 10. Ablations of the GERASTSA transformer against the full model. (a) Return of the greedy policy against the PPO update on the 8 days evaluated during fine-tuning (shaded: standard error); dotted: mean return of the baseline on the same days. (b) Validation MSE of pretraining against the pretrainin step.

The learned policy beats the baseline by reallocating lanes in tidal flow. During each rush hour, it assigns most lanes to the peak direction and makes about two thirds of the roads one-way in that direction, and around midday, when demand is balanced, it returns to two-way roads. It thereby raises the capacity in the peak direction by more than half, to 155 and 159 of the 200 lanes instead of 100, which keeps queues from forming at the roads where trips enter the network and on the roads near the two zones, and from spilling back into gridlock. The policy learned this pattern from the reward alone; neither the behavior policy of pretraining nor the initial configuration suggests it. Rerouting makes it workable: a vehicle whose next direction closes takes another route, which strong connectivity guarantees to exist (Proposition 1), and the mean travel time still falls by a third despite such detours. One-way roads appear to be essential to this gain. Half of the roads of our grid have only two lanes, which a controller that keeps every road twoway cannot reallocate at all, and on the other roads, such a controller can open at most $L _ { r } - 1$ lanes in one direction, at most 140 of the 200 lanes in all. Within these bounds, the load balancer and the two-way restrictions of the policy leave six to eight times the delay of the learned policy, and none of them prevents gridlock. A policy trained for two-way roads might do better than the restricted one, but it would face the same bounds.

The learned speed limits, by contrast, cost return. Holding them nominal at test time removes more than half of the remaining delay, and applied to the initial configuration, they increase its delay. The delay charges every vehicle for the time it loses below the nominal speed limit, so a lower speed limit costs delay on its own road and pays off only if it prevents more delay elsewhere. In our simulator, moreover, a lane discharges at a rate that a lower speed limit can only reduce, and its capacity does not drop under congestion, so the mechanisms by which VSLs delay the onset of congestion and dampen shock waves on freeways [3] have no counterpart. The policy keeps its speed limits close to nominal, but even the small reductions in Fig. 8 (e) cost more than they save. Why fine-tuning did not remove them is not clear from our runs; possibly, the delay that a speed limit causes on its own road enters the mixed reward of that road only with the weight $\alpha _ { \mathrm { G } } / R$ (Section V-C1), which is small against the variation contributed by all other roads. Whether learned speed limits can contribute to joint control thus remains open; answering it would need a simulator with the congestion mechanisms on which VSLs act.

Action resolution, finally, rarely has to intervene during the evening rush hour: between 16:00 and 19:30, only 4 proposals are rejected, although the policy keeps 35 to 40 roads one-way around the peak, so it largely proposes closings that keep the network strongly connected. Most rejections occur when it switches many roads to one-way at once, at the onset of the morning rush hour, and at night, when lane allocations hardly affect the reward and the policy keeps up to 34 of the 60 roads one-way and proposes conflicting closings at no cost. Since every road of our grid has at least two lanes, a rejected proposal is always relaxed to the nearest allocation that keeps one lane open, which costs the policy little and may explain why it does not avoid such proposals entirely.

Among the components of attention, the relational bias matters most. Without it, attention is global but blind to which roads are adjacent: every road is related to every other in the same way, and the model must infer which roads exchange traffic from correlations in their features. This variant forecasts worst of all, and its policy incurs almost five times the delay of the full model and has four times as many closing proposals rejected, which suggests that without relations, the actor can tell neither where shed traffic goes nor which closings conflict. Telling apart the directions in which adjacent roads meet, by contrast, helped less clearly: collapsing the four adjacency relations into one left the forecasting error unchanged, and although the resulting final policy incurs 1.5 times the delay of the full model, this variant did better than the full model at most earlier evaluations. A possible reason is the uniform orientation of our grid, in which every road runs forward eastward or southward. The forward and backward features of adjacent roads then refer to the same compass directions, so the model may learn from their traffic which directions of adjacent roads queue together, and thus how spillback propagates through a shared junction, without being told. Randomizing the reference ordering of the endpoints of every road in each episode would remove this shortcut and leave the endpoint relations as the only indication of how the directions of adjacent roads connect, so training both the full model and the variant with collapsed relations under such randomization and comparing them would measure the contribution of the endpoint relations. Since the orientation of a real network is arbitrary, such randomization is also a natural step toward transfer to other networks, and irregular networks, on which the hop distance says less about how two roads interact, would test the endpoint relations further.

The spatial embedding matters for control but not for forecasting: without it, the forecasting error is unchanged, but the policy incurs five times the delay of the full model and has 3.5 times as many closing proposals rejected. The junction degrees carry information that attention cannot easily recover, since attention averages over the attended tokens with weights that sum to one and therefore cannot directly count the roads at a junction, even with the relational bias; Graphormer likewise complements attention with a degree-based centrality encoding [15]. For forecasting, the number of roads at a junction matters little once the traffic on them is observed, but for control, it bears on which closings keep the network strongly connected: a junction with few roads has few alternatives for entering and leaving it, so it is easily cut off. The evidence on how the adjacent roads are split between the two ends of a road is mixed as well: with the undirected embedding, the final policy incurs 1.4 times the delay of the full model, yet the variant led the full model at 5 of the 8 earlier evaluations. On our grid, the total degree already identifies whether a road touches a corner, lies on the boundary, leads into the interior, or lies within it, so the undirected embedding loses only which of its ends lies toward the boundary. The temporal embedding, finally, shows the most consistent benefit of the three, if a small one: without it, the forecasting error is higher throughout pretraining, the greedy return is lower at most evaluations, and the final policy incurs 1.5 times the delay of the full model and has 2.4 times as many closing proposals rejected. The benefit is possibly small because the causal mask already conveys temporal order: tokens at different positions attend to different numbers of positions, from which the stacked layers can infer how recent a position is, and the traffic of a road changes gradually over a window, so the order of its time steps can largely be read from their content. Repeating this ablation with a mask without causality, that is, comparing the model with and without the temporal embedding when every token of both may attend to all valid positions of its window, would isolate the temporal embedding. Since every window ends at its decision step and only its last position is read out, such a mask would use no information from the future, and without the temporal embedding, the model would then see the earlier time steps of a window, apart from the last one at which it is read out, as an unordered set, so this comparison would measure all the temporal order that the embedding provides.

Our evaluation has several limitations. It uses a single synthetic network with one demand pattern, a $6 \times 6$ grid with two rush hours between opposite corners, which suits the tidal pattern that the policy learned. Each variant was trained once, so differences of about 1% in return between variants, even with 1.5 times the delay, cannot be firmly attributed to the architecture; several training seeds per variant would be needed. Of our two baselines, the fixed configuration ignores the traffic, and the load balancer responds to it but keeps every road two-way; a schedule that makes roads one-way toward the commercial zone in the morning and away from it in the evening would show how much of the gain requires a policy that responds to the observed traffic. All vehicles are homogeneous CAVs that comply fully and route themselves with knowledge of the current traffic, so mixed traffic and partial compliance are not modeled. The mesoscopic simulator has no signals or turning conflicts at junctions and no capacity drop, which limits what VSLs can achieve, and its gridlock safeguard removes vehicles that would otherwise keep accruing delay, which flatters the baseline. Finally, attention over all $W _ { \mathrm { m a x } } R$ road–time tokens grows quadratically with the number of roads: our 60 roads give 960 tokens per sample, and a city network with thousands of roads would need sparser attention, for example restricted to a hop neighborhood or factorized over space and time. In our implementation, the relational bias alone doubles the time of a pretraining step, 0.21 against 0.11 s. Since no parameter of the model depends on the network (Section IV), the most direct extension is to train on several networks and test transfer to unseen ones, including real road networks.

## VIII. CONCLUSION

We formulated network-wide dynamic lane allocation and directional speed limit control with one-way roads, explicit lane clearing, and a strong-connectivity requirement on the navigable graph, and gave an action-resolution algorithm that keeps every junction reachable from every other at every time step, for any admissible actions and however long clearing takes. All road controllers share a single GERASTSA trans former, pretrained on action-conditioned traffic forecasting and fine-tuned with per-road PPO under a mixed reward. On a 6×6 grid with commuter rush hours, the learned policy reallocated lanes in tidal flow, making about two thirds of the roads oneway in the peak direction during each rush hour. It reduced the daily delay by 90% and the mean travel time by 32% against holding the initial configuration with nominal speed limits, and the delay by 85% against a tuned load balancer that keep every road two-way. Restricted to two-way roads at test time, the policy gained only about as much as the load balancer, which suggests that its gain rests on one-way roads, wherea its learned speed limits cost return: holding them nominal more than halved its delay. Pretraining roughly halved the number of updates needed to surpass the fixed configuration and left less than half the delay of fine-tuning from random initialization. The relational bias and the spatial embedding mattered most, as removing either multiplied the delay by about five. Removing the endpoint relations, the direction of the spatial embedding, or the temporal embedding left the fina policies with 1.4 to 1.5 times the delay, an effect that the earlier evaluations and pretraining corroborate only for the temporal embedding. The most promising direction for future work is to train the model on several networks, with the orientation of their roads randomized in each episode, and evaluate it on unseen ones, real road networks included, which its network independent parameters permit. Two ablations, each repeated under a changed condition and with several training seeds, ld h ib i f h d i l i and the temporal embedding: collapsing the endpoint relations with the orientation of the roads randomized in each episode, and removing the temporal embedding with a mask without causality. A schedule of one-way roads as a further baseline would show how much of the gain requires a learned policy, and a simulator with capacity drop whether learned speed limits can contribute at all.

## ACKNOWLEDGMENTS

The authors thank Kaggle for free access to GPUs, which supported the experiments of this work.

During the preparation of this work, the authors used Claude and ChatGPT to improve manuscript clarity, check

notational consistency, and assist with programming. The authors reviewed and edited all AI-assisted content and take full responsibility for the content of this publication.

## REFERENCES

[1] Transportation Research Board and National Academies of Sciences, Engineering, and Medicine, Convertible Roadways and Lanes. Washington, DC: The National Academies Press, 2004. [Online]. Available: https://nap.nationalacademies.org/catalog/23331/ convertible-roadways-and-lanes

[2] B. Wolshon, ““One-Way-Out”: Contraflow Freeway Operation for Hurricane Evacuation,” Natural Hazards Review, vol. 2, no. 3, pp. 105–112, 2001. [Online]. Available: https://ascelibrary.org/doi/abs/10. 1061/%28ASCE%291527-6988%282001%292%3A3%28105%29

[3] X.-Y. Lu and S. E. Shladover, “Review of Variable Speed Limits and Advisories: Theory, Algorithms, and Practice,” Transportation Research Record, vol. 2423, no. 1, pp. 15–23, 2014. [Online]. Available: https://doi.org/10.3141/2423-03

[4] M. Hausknecht, T.-C. Au, P. Stone, D. Fajardo, and T. Waller, “Dynamic lane reversal in traffic management,” in 2011 14th International IEEE Conference on Intelligent Transportation Systems (ITSC), 2011, pp. 1929–1934.

[5] M. W. Levin and S. D. Boyles, “A cell transmission model for dynamic lane reversal with autonomous vehicles,” Transportation Research Part C: Emerging Technologies, vol. 68, pp. 126–143, 2016. [Online]. Available: https://www.sciencedirect.com/science/article/pii/ S0968090X16000772

[6] D. Braess, A. Nagurney, and T. Wakolbinger, “On a Paradox of Traffic Planning,” Transportation Science, vol. 39, no. 4, pp. 446–450, Nov. 2005. [Online]. Available: https://doi.org/10.1287/trsc.1050.0127

[7] H. E. Robbins, “A Theorem on Graphs, with an Application to a Problem of Traffic Control,” The American Mathematical Monthly, vol. 46, no. 5, pp. 281–283, 1939. [Online]. Available: http://www.jstor.org/stable/2303897

[8] M. Duell, M. W. Levin, S. D. Boyles, and S. T. Waller, “Impact of Autonomous Vehicles on Traffic Management: Case of Dynamic Lane Reversal,” Transportation Research Record, vol. 2567, no. 1, pp. 87–94, 2016. [Online]. Available: https://doi.org/10.3141/2567-10

[9] K.-F. Chu, A. Y. S. Lam, and V. O. K. Li, “Dynamic Lane Reversal Routing and Scheduling for Connected and Autonomous Vehicles: Formulation and Distributed Algorithm,” IEEE Transactions on Intelligent Transportation Systems, vol. 21, no. 6, pp. 2557–2570, 2020.

[10] U. Gunarathna, H. Xie, E. Tanin, S. Karunasekera, and R. Borovica-Gajic, “Real-time Road Network Optimization with Coordinated Reinforcement Learning,” ACM Trans. Intell. Syst. Technol., vol. 14, no. 4, Jul. 2023. [Online]. Available: https://doi.org/10.1145/3603379

[11] S. Chen, H. Wang, and Q. Meng, “An Optimal Dynamic Lane Reversal and Traffic Control Strategy for Autonomous Vehicles,” IEEE Transactions on Intelligent Transportation Systems, vol. 23, no. 4, pp. 3804–3815, 2022.

[12] F. Jin et al., “Optimal integrated control of internal boundaries, ramp inflows, and variable speed limits for lane-free traffic of connected automated vehicles on freeways,” Transportation Research Part C: Emerging Technologies, vol. 189, p. 105728, 2026. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0968090X26002160

[13] M. Alshiekh, R. Bloem, R. Ehlers, B. Könighofer, S. Niekum, and U. Topcu, “Safe Reinforcement Learning via Shielding,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 32, no. 1, 2018. [Online]. Available: https://doi.org/10.1609/aaai.v32i1.11797

[14] A. Vaswani et al., “Attention is all you need,” in Proceedings of the 31st International Conference on Neural Information Processing Systems, ser. NIPS’17. Red Hook, NY, USA: Curran Associates Inc., 2017, pp. 6000–6010.

[15] C. Ying et al., “Do Transformers Really Perform Badly for Graph Representation?” in Advances in Neural Information Processing Systems, M. Ranzato, A. Beygelzimer, Y. Dauphin, P. S. Liang, and J. Wortman Vaughan, Eds., vol. 34. Curran Associates, Inc., 2021, pp. 28 877– 28 888. [Online]. Available: https://proceedings.neurips.cc/paper\_files/ paper/2021/file/f1c1592588411002af340cbaedd6fc33-Paper.pdf

[16] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal Policy Optimization Algorithms,” 2017. [Online]. Available: https://arxiv.org/abs/1707.06347

[17] S. Kim, S. Shekhar, and M. Min, “Contraflow Transportation Network Reconfiguration for Evacuation Route Planning,” IEEE Transactions on Knowledge and Data Engineering, vol. 20, no. 8, pp. 1115–1129, 2008.

[18] J. R. D. Frejo, I. Papamichail, M. Papageorgiou, and E. F. Camacho, “Macroscopic Modeling and Control of Reversible Lanes on Freeways,” IEEE Transactions on Intelligent Transportation Systems, vol. 17, no. 4, pp. 948–959, 2016.

[19] T. Li, N. Wang, M. Zhang, and Z. He, “Dynamic reversible lane optimization in autonomous driving environments: Balancing efficiency and safety,” Journal of Industrial and Management Optimization, vol. 20, no. 3, pp. 901–925, 2024. [Online]. Available: https://www.aimsciences.org/article/id/64ef047789e5d06909db2257

[20] F. Boesch and R. Tindell, “Robbins’s Theorem for Mixed Multigraphs,” The American Mathematical Monthly, vol. 87, no. 9, pp. 716–719, 1980. [Online]. Available: https://doi.org/10.1080/00029890.1980.11995131

[21] F. R. K. Chung, M. R. Garey, and R. E. Tarjan, “Strongly connected orientations of mixed multigraphs,” Networks, vol. 15, no. 4, pp. 477–484, 1985. [Online]. Available: https://onlinelibrary.wiley.com/doi/ abs/10.1002/net.3230150409

[22] S. Smulders, “Control of freeway traffic flow by variable speed signs,” Transportation Research Part B: Methodological, vol. 24, no. 2, pp. 111–132, 1990. [Online]. Available: https://www.sciencedirect.com/ science/article/pii/019126159090023R

[23] A. Hegyi, B. De Schutter, and H. Hellendoorn, “Model predictive control for optimal coordination of ramp metering and variable speed limits,” Transportation Research Part C: Emerging Technologies, vol. 13, no. 3, pp. 185–209, 2005. [Online]. Available: https: //www.sciencedirect.com/science/article/pii/S0968090X05000264

[24] A. Hegyi, S. P. Hoogendoorn, M. Schreuder, H. Stoelhorst, and F. Viti, “SPECIALIST: A dynamic speed limit control algorithm based on shock wave theory,” in 2008 11th International IEEE Conference on Intelligent Transportation Systems, 2008, pp. 827–832.

[25] A. Hegyi and S. P. Hoogendoorn, “Dynamic speed limit control to resolve shock waves on freeways - Field test results of the SPECIAL-IST algorithm,” in 13th International IEEE Conference on Intelligent Transportation Systems, 2010, pp. 519–524.

[26] F. Zhu and S. V. Ukkusuri, “Accounting for dynamic speed limit control in a stochastic traffic environment: A reinforcement learning approach,” Transportation Research Part C: Emerging Technologies, vol. 41, pp. 30–47, 2014. [Online]. Available: https: //www.sciencedirect.com/science/article/pii/S0968090X1400028X

[27] Z. Li, P. Liu, C. Xu, H. Duan, and W. Wang, “Reinforcement Learning-Based Variable Speed Limit Control Strategy to Reduce Traffic Congestion at Freeway Recurrent Bottlenecks,” IEEE Transactions on Intelligent Transportation Systems, vol. 18, no. 11, pp. 3204–3217, 2017.

[28] Y. Wu, H. Tan, L. Qin, and B. Ran, “Differential variable speed limits control for freeway recurrent bottlenecks via deep actor-critic algorithm,” Transportation Research Part C: Emerging Technologies, vol. 117, p. 102649, 2020. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0968090X20305647

[29] J. Yang, P. Wang, F. Golpayegani, and S. Wang, “DVS-RG: Differential Variable Speed Limits Control using Deep Reinforcement Learning with Graph State Representation,” 2024. [Online]. Available: https://arxiv.org/abs/2405.09163

[30] T. Chu, J. Wang, L. Codecà, and Z. Li, “Multi-Agent Deep Reinforcement Learning for Large-Scale Traffic Signal Control,” IEEE Transactions on Intelligent Transportation Systems, vol. 21, no. 3, pp. 1086–1095, 2020.

[31] H. Wei et al., “CoLight: Learning Network-level Cooperation for Traffic Signal Control,” in Proceedings of the 28th ACM International Conference on Information and Knowledge Management, ser. CIKM ’19. New York, NY, USA: Association for Computing Machinery, 2019, pp. 1913–1922. [Online]. Available: https://doi.org/10.1145/ 3357384.3357902

[32] F.-X. Devailly, D. Larocque, and L. Charlin, “IG-RL: Inductive Graph Reinforcement Learning for Massive-Scale Traffic Signal Control,” IEEE Transactions on Intelligent Transportation Systems, vol. 23, no. 7, pp. 7496–7507, 2022.

[33] C. Schroeder de Witt et al., “Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge?” 2020. [Online]. Available: https://arxiv.org/abs/2011.09533

[34] C. Yu et al., “The surprising effectiveness of PPO in cooperative multiagent games,” in Proceedings of the 36th International Conference on

Neural Information Processing Systems, ser. NIPS ’22. Red Hook, NY, USA: Curran Associates Inc., 2022.

[35] S. Huang and S. Ontañón, “A Closer Look at Invalid Action Masking in Policy Gradient Algorithms,” The International FLAIRS Conference Proceedings, vol. 35, May 2022. [Online]. Available: https://journals.flvc.org/FLAIRS/article/view/130584

[36] M. Jaderberg et al., “Reinforcement Learning with Unsupervised Auxiliary Tasks,” in International Conference on Learning Representations, 2017. [Online]. Available: https://openreview.net/forum?id=SJ6yPD5xg

[37] M. Schwarzer et al., “Pretraining representations for data-efficient reinforcement learning,” in Proceedings of the 35th International Conference on Neural Information Processing Systems, ser. NIPS ’21. Red Hook, NY, USA: Curran Associates Inc., 2021.

[38] Y. Li, R. Yu, C. Shahabi, and Y. Liu, “Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting,” in International Conference on Learning Representations, 2018. [Online]. Available: https://openreview.net/forum?id=SJiHXGWAZ

[39] B. Yu, H. Yin, and Z. Zhu, “Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting,” in Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence, IJCAI-18. International Joint Conferences on Artificial Intelligence Organization, Jul. 2018, pp. 3634–3640. [Online]. Available: https://doi.org/10.24963/ijcai.2018/505

[40] C. Zheng, X. Fan, C. Wang, and J. Qi, “GMAN: A Graph Multi-Attention Network for Traffic Prediction,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, no. 01, 2020, pp. 1234– 1241. [Online]. Available: https://doi.org/10.1609/aaai.v34i01.5477

[41] J. Jiang, C. Han, W. X. Zhao, and J. Wang, “PDFormer: Propagation Delay-Aware Dynamic Long-Range Transformer for Traffic Flow Prediction,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 4, 2023, pp. 4365–4373. [Online]. Available: https://doi.org/10.1609/aaai.v37i4.25556

[42] P. Shaw, J. Uszkoreit, and A. Vaswani, “Self-Attention with Relative Position Representations,” in Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), M. Walker, H. Ji, and A. Stent, Eds. New Orleans, Louisiana: Association for Computational Linguistics, Jun. 2018, pp. 464–468. [Online]. Available: https://aclanthology.org/N18-2074/

[43] J. L. Ba, J. R. Kiros, and G. E. Hinton, “Layer Normalization,” 2016. [Online]. Available: https://arxiv.org/abs/1607.06450

[44] N. Shazeer, “GLU Variants Improve Transformer,” 2020. [Online]. Available: https://arxiv.org/abs/2002.05202

[45] R. Xiong et al., “On Layer Normalization in the Transformer Architecture,” in Proceedings of the 37th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, H. Daumé III and A. Singh, Eds., vol. 119. PMLR, Jul. 2020, pp. 10 524–10 533. [Online]. Available: https://proceedings.mlr.press/v119/ xiong20b.html

[46] S. Ben Taieb, G. Bontempi, A. F. Atiya, and A. Sorjamaa, “A review and comparison of strategies for multi-step ahead time series forecasting based on the NN5 forecasting competition,” Expert Systems with Applications, vol. 39, no. 8, pp. 7067–7083, 2012. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0957417412000528

[47] J. Schulman, P. Moritz, S. Levine, M. I. Jordan, and P. Abbeel, “High-Dimensional Continuous Control Using Generalized Advantage Estimation,” in 4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings, Y. Bengio and Y. LeCun, Eds., 2016. [Online]. Available: https://doi.org/10.48550/arXiv.1506.02438

[48] F. Pardo, A. Tavakoli, V. Levdik, and P. Kormushev, “Time Limits in Reinforcement Learning,” in Proceedings of the 35th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, J. Dy and A. Krause, Eds., vol. 80. PMLR, Jul. 2018, pp. 4045–4054. [Online]. Available: https://proceedings.mlr.press/v80/pardo18a.html

[49] V. Mnih et al., “Asynchronous Methods for Deep Reinforcement Learning,” in Proceedings of The 33rd International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, M. F. Balcan and K. Q. Weinberger, Eds., vol. 48. New York, New York, USA: PMLR, Jun. 2016, pp. 1928–1937. [Online]. Available: https://proceedings.mlr.press/v48/mniha16.html