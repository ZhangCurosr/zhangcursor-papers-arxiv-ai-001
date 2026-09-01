# Learning-Assisted Congestion-Aware Route Scheduling for Semiconductor Fab Material Control Systems

Hao Yin\*, Meiqi Tu\*, Anbang Liu, Shaochong Lin, Max Z.J. Shen

Abstract—Automated material handling systems in semiconductor fabs are operated by a material control system (MCS) that must schedule a relay route for every transport command online, before execution. This is a data-driven scheduling problem in which route cost is dominated in the upper tail by queueing at heterogeneous, partially observable relay equipment, so route selection requires estimating both delivery time and congestion risk at the decision moment. This paper proposes a transport-network-aware dynamic congestion representation (TN-DCR). Built on a static directed transport graph induced by historically observed relay segments, TN-DCR combines structural route priors, multi-window network-wide congestion context, route-level bottleneck exposure, and an inductive graphaware route embedding, all constructed under a prediction-timesafety invariant that admits only information observed strictly before the prediction moment. The representation feeds separate queue- and transfer-time regressors and an ordinal multi-label classifier producing calibrated multi-threshold exceedance scores, with an empirical-Bayes stock-key residual correction reducing systematic queue-time underprediction. The predictions serve as costs in a risk-constrained route-scheduling rule that minimizes predicted delivery time subject to a bound on extremecongestion probability, embedding the learned predictors within a lightweight operations-research decision model. In a controlled closed-loop evaluation, mean delivery time falls by 16.4% and internal resource waiting time by 22.6% while throughput remains essentially unchanged.

Index Terms—Automated material handling system, material control system, congestion prediction, estimated time of arrival, risk-aware routing, semiconductor manufacturing.

Note to Practitioners—This paper is motivated by a practical difficulty in operating a semiconductor fab AMHS: whenever the MCS routes a carrier, it must commit to a relay path online, before execution, without knowing how long the carrier will wait at intermediate stockers, lifts, and overhead transport. Such waiting is the main source of long, unpredictable deliveries and of queue-time risk for time-sensitive lots, yet current practice routes by a fixed shortest-path rule that ignores the live congestion state, and high-fidelity simulation is too slow to consult per command. We provide a lightweight decision

aid that runs as each command arrives: it estimates queueing time, transfer time, and the probability of severe congestion for each candidate route, then selects a route that keeps this probability below an operator-set tolerance while minimizing predicted delivery time. Two properties matter for deployment. First, every feature is computed strictly from information observable at the decision moment, so offline evaluation reflects production behavior, and the gradient-boosted models support low-overhead candidate scoring without specialized hardware. Second, the risk ranking is concentrated: reviewing only the small fraction of highest-risk commands captures most truly congested ones, enabling selective manual intervention. Two limitations apply. The models are trained on routes chosen by the incumbent dispatcher, so predictions for rarely used routes are less reliable; the framework flags low-support routes and falls back to the existing policy instead of extrapolating. The reported gains come from a calibrated closed-loop simulator, so a staged, monitored rollout is recommended. Adapting the method to a new fab mainly requires re-estimating the transport graph, congestion statistics, and risk thresholds from that fab’s own logs.

## I. INTRODUCTION

S advanced manufacturing becomes more automated and operates at higher tempos, Automated Material Handling Systems (AMHSs) have become critical to continuous production. This is especially true in semiconductor manufacturing, where fabs operate continuously and each carrier holds a batch of high-value in-process wafers or display panels. Materialhandling delays directly increase cycle time and work-inprocess, while prolonged queueing at intermediate storage can cause queue-time (Q-time) violations and threaten yield [1], [2]. As the logistics dispatching hub, the Material Control System (MCS) converts transport requests from the Manufacturing Execution System (MES) into executable commands and schedules a relay route for each command online. This is an AI-driven scheduling problem within data-driven manufacturing and logistics operations: it combines learned route-cost estimates with an operations-research decision rule under time-varying, partially observed congestion. Unlike the local controller of a single device, the MCS coordinates heterogeneous transport and relay resources. In semiconductor display fabs, these may include combinations of Stockers, Lifts, Bays, OHSs, and OHCVs (Fig. 1). The MCS must select suitable routes and resources for individual commands while maintaining efficient flow and limiting congestion.

Route selection is more complex than static shortest-path search. A fab AMHS may contain equipment from several vendors, with different interfaces and local control logic, while some internal device queues are only partly observable. A command may pass through several equipment types, so its delivery time depends on both route structure and the timevarying state of the associated resources. Congestion also propagates through the directed network: a congested stocker or lift can back-pressure the overhead transport it feeds and change the cost of neighboring bays. Transport times therefore have a pronounced upper tail, with a small fraction of severely delayed commands causing disproportionate operational disruption and Q-time risk. Before assigning a route, the MCS must estimate the expected delivery cost and congestion risk of each candidate using only information available at that moment. Command-level ETA and congestion-risk prediction are therefore essential for congestion-aware online scheduling.

![](images/f17a71461443575c92dd8ae7123e813dfe9b2e2aed2227f9582714810bbbd748.jpg)  
Fig. 1. Illustration of congestion-aware route selection in the MCS. The MCS compares multiple feasible routes for transferring a carrier from EQP001 to EQP002. The orange STK002 denotes a congested stocker with many queued commands, which may increase the delivery delay of Route 2. The example shows why route evaluation should consider current congestion exposure at relay resources as well as static route length.

Existing methods address parts of this problem but leave a gap between structural fidelity and online deployment. Discrete-event simulation captures AMHS topology, vehicle interactions, and local queues in detail, but is too expensive for low-latency evaluation of many candidate routes per command. Gradient-boosted decision trees efficiently handle heterogeneous engineered features in tabular industrial data and are attractive for command-level ETA prediction. However, the predictor alone does not directly encode the directed transport structure through which congestion spreads, so route topology, network-wide load, and route-local exposure are only indirectly represented. Graph-based road-network ETA models, including the production system in [3], show the value of network structure. Yet they usually assume richly observed traffic and do not address prediction-time safety in an industrial MCS, where every feature must be constructed from information observable before a command is evaluated. Learning-based schedulers also show that predictive models deliver their value when embedded in an explicit decision model rather than replacing it [4], [5], but this perspective has received little attention in fab material handling.

We address this gap with a transport-network-aware dynamic congestion representation (TN-DCR), which provides the cost and risk terms for chance-constrained route scheduling in heterogeneous relay-based AMHSs. TN-DCR uses a static directed transport graph whose edges are induced by ordered node-pair traversals in historical routes. Dynamic information follows a strict prediction-time-safety invariant: every temporal feature uses only events and completed commands observed strictly before the command’s prediction moment. This invariant is enforced across network-wide, route-local, and graph-neighborhood feature construction. It aligns the representation with the information boundary of an online MCS and prevents future or concurrent information from entering offline predictions.

Under this invariant, TN-DCR combines four types of information: structural priors describing the planned route and frequently traversed segments; multi-window, as-of context summarizing recent network-wide congestion immediately before prediction; route-level bottleneck exposure based on recent edge- and node-level activity along the candidate route; and a lightweight graph-aware route representation. The last component applies mean and max pooling to route nodes and their one-hop neighborhoods, followed by a fixed untrained projection. Unlike an end-to-end graph neural network, it has no learned message-passing parameters and can be recomputed independently of the downstream models. It therefore retains an inductive feature-construction mechanism for changing route compositions while avoiding the training and serving cost of a learned graph encoder. For route segments with limited historical support, explicit indicators allow unsupported candidates to be handled conservatively.

The same TN-DCR representation supports separate models for delivery time and congestion risk. Separate regressors estimate queueing and transfer times, from which total delivery time is reconstructed. An ordinal multi-label classifier models long-tail congestion by estimating exceedance probabilities at multiple nested thresholds from the same representation; the highest-threshold output is the most selective extremecongestion indicator. An empirical-Bayes residual correction indexed by destination stock key further adjusts queue-time estimates. It reduces persistent destination-specific bias while shrinking sparsely observed stock keys toward the global model. Together, the regression and risk outputs provide the costs and constraints for a route-selection policy that limits the predicted extreme-congestion probability of the selected route. This follows chance-constrained routing under uncertainty [6], [7], but the constraint uses an online, calibrated learned estimate rather than an external distributional assumption. The framework therefore lies between conventional tabular ETA models and graph- or event-aware congestion models such as [8]: it explicitly represents network structure and dynamic congestion while retaining lightweight predictors suitable for command-level online scoring.

The main contributions are as follows:

• We formulate TN-DCR as a unified, prediction-time-safe representation for heterogeneous relay-based AMHSs.

It combines structural route information, multi-window network-wide congestion, route-local bottleneck exposure, and graph-aware neighborhood information in a common feature space. All dynamic features use only information available before prediction.

• We develop a lightweight inductive graph representation using mean/max pooling over route nodes and their one-hop neighborhoods, followed by a fixed untrained projection. It requires no learned message passing and separates graph feature construction from downstream retraining, supporting changing route compositions and low-overhead online evaluation.

• We combine TN-DCR with separate queue- and transfertime regressors, an ordinal multi-label risk model over several tail thresholds, and empirical-Bayes destination stock-key calibration for queue prediction. Their point and tail-risk estimates become the costs and constraints of a chance-constrained scheduling rule that minimizes predicted delivery time while bounding predicted extremecongestion risk. This integrates learned predictors with a conventional operations-research decision model.

• We evaluate the framework on real transport-command data from a semiconductor display fab from both predictive and scheduling perspectives. The evaluation covers regressor selection under a shared feature, training, and postprocessing pipeline, multi-threshold risk assessment, ablations of dynamic and graph-aware features, stockkey calibration, and closed-loop comparison with static shortest-path routing. This links command-level prediction quality to downstream operational performance.

The remainder of the paper is organized as follows. Section II reviews related work. Section III defines the command-level prediction and route-selection problems and the prediction-time-safety requirement. Section IV presents TN-DCR, the prediction models, stock-key calibration, and route selection. Section V reports the experiments, and Section VI gives the limitations and future work.

## II. RELATED WORK

We review four research threads related to our setting: scheduling and delivery-time estimation in fab material handling systems; learning-based transport scheduling and routing; network-aware travel-time and congestion prediction; and the use of learned predictors in optimization and risk-aware routing. These studies motivate a representation that captures network structure and dynamic congestion while supporting asynchronous, command-level MCS scheduling.

## A. Scheduling and Delivery-Time Estimation in Fab AMHS

Semiconductor manufacturing is a demanding scheduling environment with strict timing constraints. Recent studies schedule cluster tools through deep reinforcement learning and adaptive search [9], control wafer-delay variability with Kanban feedback [1], and model interacting timing requirements using dual-time Petri nets [2]. They show that timing risk, rather than mean throughput alone, is central to fab scheduling.

However, they focus on processing resources, whereas we schedule the relay path of each transport command.

AMHS performance has mainly been studied through queueing models and discrete-event simulation. Lin et al. [10] simulate connecting-transport layouts in 300-mm fabs, while Nazzal and McGinnis [11] estimate response times for closedloop multi-vehicle systems using an extended Markov-chain approximation. These methods support system-level analysis and design, but are not designed for command-specific routing under a rapidly changing network state. Repeated simulation is costly when many routes must be evaluated online, and analytical models rely on aggregate assumptions that do not capture the current route-dependent state. Liao and Wang [12] estimate AMHS delivery time with separate neural predictors for intrabay loops and combine them into a path estimate. Although this matches the physical organization of a fab, it does not jointly model dependencies between successive segments and the current state of a directed transport graph.

These issues also arise beyond 300-mm wafer fabs. In the semiconductor display fab studied here, commands traverse heterogeneous relay equipment, intermediate storage creates queueing delay that affects Q-time and cycle-time risk, and a central controller coordinates movement over a structured network. We therefore study command-level relay-path scheduling whose cost depends on topology and current resource load. Unlike general warehouse or logistics routing, this setting also involves heterogeneous equipment, partially observed vendorspecific states, and yield-relevant timing constraints.

## B. Scheduling and Routing of Automated Transport in Manufacturing and Logistics

Related work also considers automated-vehicle dispatching and flow routing in manufacturing and logistics. Famularo et al. [13] coordinate autonomous vehicles through a multilayer architecture that separates supervisory assignment from local motion control. Tresca et al. [5] repair delivery routes as new information arrives, while related work balances retrieval flows by configuring automated storage modules [14]. Learning-based methods include meta-reinforcement learning for dynamic flexible job shops [15] and sparse-reward deep reinforcement learning for cloud manufacturing [16].

Our setting differs in two ways. First, these systems usually assume that tasks and resources are directly observable. A fab AMHS combines vendor-specific relay equipment whose internal queues are only partly reported, so the relevant state must be inferred from event logs. Second, their decisions mainly assign vehicles or workstations, while an MCS selects an ordered sequence of relay resources whose cost is strongly affected by intermediate queues. We therefore retain an explicit route-cost model and learn its congestion exposure.

## C. Network-Structure-Aware Travel-Time and Congestion Prediction

Route-time models can benefit from preserving connections among transport segments instead of treating a route as independent tabular attributes. Derrow-Pinion et al. [3] develop a production-scale graph-based ETA model, while DuETA [17] includes congestion propagation; inductive neighborhood aggregation supports generalization to unseen node combinations [18]. Other work predicts the network state itself. Diffusion-convolutional recurrent models [19] and adaptiveadjacency networks [20] forecast grid-based flow or speed. Spatio-temporal graph fusion has also been used for industrial and environmental monitoring [21]. STGNPP instead models the occurrence and duration of congestion events as a marked point process on a road graph [8].

A fab MCS differs in three respects. First, each command is evaluated asynchronously at its own decision time. All candidate-route features must use only information available then; a chronological train–test split alone does not prevent look-ahead leakage. We therefore impose predictiontime safety on the representation itself. Second, delay arises mainly from queues at discrete, capacity-limited relay resources whose internal states are only partly observed, rather than from free-flowing road traffic. Third, we do not predict a full future network snapshot or the next congestion event. For each command, we estimate candidate-route queue and transfer times and multiple upper-tail queue-risk probabilities. These levels distinguish routes with similar point estimates but different severe-delay exposure. We therefore use the directed AMHS topology to build graph-aware route features for lightweight predictors instead of training an end-to-end graph ETA model.

## D. Coupling Learned Predictors With Optimization-Based Decision Models

Gradient-boosted trees, including XGBoost [22], Light-GBM [23], and CatBoost [24], provide strong nonlinear modeling for heterogeneous tabular data with lightweight training and inference. A key industrial question is how to add graph structure without replacing this prediction stack with an endto-end GNN. Ivanov and Prokhorenkova [25] jointly train GBDT and GNN components for graphs with tabular node features. TN-DCR instead constructs a fixed-length graphaware route representation through neighborhood aggregation and a fixed projection. Downstream GBDT models then use these features, separating graph feature construction from model fitting.

Another line of work combines learning with mathematical optimization. Shi et al. [4] embed reinforcement learning in an optimization framework, allowing learned decisions to be evaluated against an explicit model rather than used as a blackbox policy. Qin et al. [26] apply reinforcement learning to resource-sharing line balancing with a program-defined feasible region. Risk-aware routing offers related decision tools. Carpin [6] solves chance-constrained stochastic orienteering with Monte Carlo tree search, and Liu et al. [7] study distributionally robust chance-constrained line planning. In these studies, the chance constraint comes from an external distributional model. In our setting, a calibrated ordinal classifier produces a command-specific exceedance probability from predictiontime-safe features. This learned risk estimate directly defines the scheduling rule’s feasible set at the decision time.

Together, these threads reveal a gap. Fab scheduling studies focus on processing resources, while AMHS analytical and simulation methods do not support rapid, state-dependent evaluation of asynchronous commands. Road-network models use topology but assume richer observations and no commandspecific information boundary. Hybrid learning–optimization methods often assume that uncertainty is known in advance. TN-DCR addresses this gap by constructing prediction-timesafe structural and dynamic features on a directed transport network, encoding them for lightweight tabular predictors, and using the resulting time and tail-risk estimates as the cost and chance constraint of congestion-aware route scheduling.

## III. PROBLEM FORMULATION

We formulate congestion-aware routing in the material control system as two coupled tasks: edge-level prediction of delivery-time components and congestion risk using only information available at the route-decision moment, and path selection after aggregating these predictions over each admissible candidate. This section defines the transport system and path space, the prediction problem and its prediction-timesafety requirement, and the route-selection problem.

## A. System Model and Routing Objective

For each transport command, the MCS selects a path through heterogeneous transport and relay resources before execution. The objective is to minimize predicted delivery time while limiting exposure to severe queueing under the timevarying network state.

The structural connectivity of the AMHS is represented by a static directed transport graph $\mathcal { G } = ( \nu , \mathcal { E } )$ , with ${ \mathcal { E } } \subseteq$ $\nu \times \nu ,$ constructed from a historical reference corpus of logged transport paths. Each node $v \in \mathcal V$ denotes a physical handling or relay location, and an edge $( u , v ) \ \in \ \mathcal { E }$ indicates an observed directed transition. For each node v, let ${ \mathcal { N } } ^ { \mathrm { o u t } } ( v ) = \{ u : ( v , u ) \in { \mathcal { E } } \} , { \mathcal { N } } ^ { \mathrm { i n } } ( v ) = \{ u : ( u , v ) \in { \mathcal { E } } \}$ and ${ \mathcal { N } } ( v ) = { \mathcal { N } } ^ { \mathrm { { o u t } } } ( v ) \cup { \mathcal { N } } ^ { \mathrm { { i n } } } ( v )$ . The graph is fixed after construction from the reference-period data.

Observed connectivity alone does not determine path admissibility, since concatenating individually observed edges need not produce a path permitted by equipment-level control logic. For command $c _ { i } ,$ with origin $o _ { i } ~ \in ~ \mathcal { V }$ and destination $d _ { i } \in \mathcal V .$ , the admissible candidate-path set is $\mathcal { P } _ { i } = \mathcal { P } ( o _ { i } , d _ { i } ) \subseteq$ $\{ p : p$ is a loopless directed path from $o _ { i }$ to $d _ { i }$ in G}. It may be supplied by the lower-level routing system or generated subject to the path-feasibility and support constraints described in Section IV-C.

The dynamic operating state is represented by the timestamped event stream ${ \cal { S } } = \{ s _ { j } = ( a _ { j } , \tau _ { j } , \zeta _ { i } ) \} _ { j }$ , where $\tau _ { j } \in$ $\mathbb { R } _ { + }$ is the observation timestamp, $a _ { j } \in \mathcal { V } \dot { \cup } \mathcal { E }$ identifies the associated node or edge, and $\boldsymbol { \zeta } _ { j } \in \mathbb { R } ^ { d _ { \zeta } }$ contains its observed congestion-related quantities. These quantities include event counts and equipment states used to construct the dynamic features in Section IV-B; histories of completed edge traversals provide an additional source of as-of information.

For command $c _ { i } ,$ , let $t _ { i } \in \mathbb { R } _ { + }$ denote the prediction moment after the command is received but before path assignment and transport execution. Its static attributes are denoted by $\mathbf { { } } x _ { i } \in { }$ $\mathbb { R } ^ { d _ { x } }$ and include origin and destination descriptors, carrier and priority information, and time-of-day variables.

A candidate path $p = ( v _ { p , 0 } , v _ { p , 1 } , \ldots , v _ { p , L _ { p } } ) \in \mathcal { P } _ { i }$ satisfies $v _ { p , 0 } = o _ { i }$ and $v _ { p , L _ { p } } = d _ { i }$ . Its ordered directed-edge sequence is

$$
\boldsymbol { \mathcal { E } } ( \boldsymbol { p } ) = \left( e _ { p , 1 } , \ldots , e _ { p , L _ { p } } \right) , \qquad e _ { p , k } = \left( v _ { p , k - 1 } , v _ { p , k } \right) .\tag{1}
$$

For edge $e _ { p , k }$ , the edge-level input $\scriptstyle { \pmb { x } } _ { i , p , k }$ augments $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ with the corresponding edge descriptors and the stock-key $g _ { i , p , k }$ used by the residual correction in Section IV-B4.

The path-level times aggregate the corresponding edge-level quantities:

$$
\begin{array} { l } { { \displaystyle y _ { i } ^ { \mathrm { q u e u e } } ( p ) = \sum _ { k = 1 } ^ { L _ { p } } y _ { i , p , k } ^ { \mathrm { q u e u e } } , \qquad y _ { i } ^ { \mathrm { t r a n s } } ( p ) = \sum _ { k = 1 } ^ { L _ { p } } y _ { i , p , k } ^ { \mathrm { t r a n s } } , } } \\ { { \displaystyle y _ { i } ^ { \mathrm { t o t a l } } ( p ) = y _ { i } ^ { \mathrm { q u e u e } } ( p ) + y _ { i } ^ { \mathrm { t r a n s } } ( p ) . } } \end{array}\tag{2}
$$

Here, $y _ { i , p , k } ^ { \mathrm { q u e u e } }$ is the delay before the transport resource required by $e _ { p , k }$ becomes available, and $y _ { i , p , k } ^ { \mathrm { t r a n s } }$ is the subsequent transfer duration on that edge, both in seconds. After the selected path $p _ { i } ^ { \star }$ is executed, we omit its path argument and write $y _ { i } ^ { a } \equiv$ y<sup>a</sup>(p<sup>⋆</sup>), a ∈ {queue, trans, total}.

Because the quantities in (2) are unavailable at $t _ { i } ,$ , routing relies on edge-level time and risk predictions and their subsequent path-level aggregation.

## B. Prediction Problem

The prediction problem is defined under an explicit information boundary. Let $S _ { < t } = \{ s _ { j } \in S : \tau _ { j } < t \}$ contain the dynamic events observed strictly before t, and let $\mathcal { D } _ { < t }$ contain the edge-level transport records completed before t. The as-of information set is

$$
\mathcal { T } _ { t } = ( \mathcal { G } , S _ { < t } , \mathcal { D } _ { < t } ) .\tag{3}
$$

All reference-period statistics and model parameters are frozen before evaluation, and every time-varying feature must respect the command-specific cutoff in (3).

For command $c _ { i } ,$ , candidate path $p \in \mathcal { P } _ { i }$ , and edge $e _ { p , k } \in$ ${ \mathcal { E } } ( { \boldsymbol { p } } )$ , the feature representation satisfies

$$
\begin{array} { r } { \phi _ { i , p , k } = f _ { \phi } \left( \pmb { x } _ { i , p , k } , p , \mathcal { T } _ { t _ { i } } \right) . } \end{array}\tag{4}
$$

This prediction-time-safety invariant excludes events and completed-edge outcomes unavailable at $t _ { i \cdot }$ , thereby preventing look-ahead leakage. Fig. 2 illustrates the telemetry and completed-history cutoffs used in the implementation.

1) Delivery-Time Targets: The point-prediction task estimates queueing and transfer time separately for each edge $e _ { p , k }$ , with base outputs $\hat { y } _ { i , p , k } ^ { \mathrm { q u e u e } }$ and $\hat { y } _ { i , p , k } ^ { \mathrm { t r a n s } }$ . The stock-key calibration in Section IV-B4 is applied only to the queue estimate, producing $\tilde { y } _ { i , p , k } ^ { \mathrm { q u e u e } }$ . No separate total-time predictor is trained; path-level times are reconstructed from the edgelevel outputs in Section III-C, consistent with (2).

2) Ordinal Congestion-Risk Targets: Let $\mathcal { D } ^ { \mathrm { r e f } }$ denote the reference corpus, where record $j$ is associated with edge $\boldsymbol { e } _ { j } = ( u _ { j } , v _ { j } )$ . For $( u , v ) \in \mathcal { E } ,$ define $\mathcal { D } _ { ( u , v ) } ^ { \mathrm { r e f } } = \{ j \in \mathcal { D } ^ { \mathrm { r e f } }$ $( u _ { j } , v _ { j } ) = ( u , v ) \}$ and $N _ { ( u , v ) } = | \mathcal { D } _ { ( u , v ) } ^ { \mathrm { r e f } } |$ . For $N _ { ( u , v ) } > 0$ , the reference transfer-time scale is

$$
\mu _ { \mathrm { t r a n s } } ^ { \mathrm { e d g e } } ( u , v ) = \frac { 1 } { N _ { ( u , v ) } } \sum _ { j \in \mathcal { D } _ { ( u , v ) } ^ { \mathrm { r e f } } } y _ { j } ^ { \mathrm { t r a n s } } .\tag{5}
$$

For reference record $j ,$ define $\rho _ { j } = y _ { j } ^ { \mathrm { q u e u e } } / \mu _ { \mathrm { t r a n s } } ^ { \mathrm { e d g e } } ( u _ { j } , v _ { j } )$ For candidate edge $\begin{array} { c c l } { e _ { p , k } } & { = } & { \left( v _ { p , k - 1 } , v _ { p , k } \right) } \end{array}$ , let $\begin{array} { r l } { \mu _ { p , k } } & { { } = } \end{array}$ $\mu _ { \mathrm { t r a n s } } ^ { \mathrm { e d g e } } ( v _ { p , k - 1 } , v _ { p , k } )$ and define

$$
\rho _ { i , p , k } = \frac { y _ { i , p , k } ^ { \mathrm { q u e u e } } } { \mu _ { p , k } } .\tag{6}
$$

This ratio expresses queueing delay relative to the nominal transfer scale of the same edge, making congestion severity comparable across heterogeneous edges.

Let $\boldsymbol { B } = \{ b _ { 1 } < \cdots < b _ { M } \}$ contain thresholds estimated from the reference distribution of $\rho _ { j }$ . For percentile level $p _ { m }$

$$
b _ { m } = Q _ { p _ { m } } \left( \left\{ \rho _ { j } : j \in \mathcal { D } ^ { \mathrm { r e f } } \right\} \right) , \qquad m = 1 , \hdots , M ,\tag{7}
$$

and the corresponding edge-level ordinal label is

$$
r _ { i , p , k , m } = \mathbb { 1 } \left[ \rho _ { i , p , k } \geq b _ { m } \right] , \qquad m = 1 , \ldots , M .\tag{8}
$$

The labels are nested because $b _ { 1 } < \dots < b _ { M }$ . In seconds, the edge-specific threshold is $\theta _ { p , k , m } = \mu _ { p , k } b _ { m } , \mathrm { s o } \ r _ { i , p , k , m } =$ $\mathbb { 1 } [ y _ { i , p , k } ^ { \mathrm { q u e a e } } \geq \theta _ { p , k , m } ]$

The shared TN-DCR representation feeds separate queueand transfer-time regressors and an ordinal congestion-risk classifier, summarized by

$$
\boldsymbol { F } : ( \boldsymbol { x } _ { i , p , k } , p , \mathcal { T } _ { t _ { i } } ) \mapsto \left( \tilde { y } _ { i , p , k } ^ { \mathrm { q u e u e } } , \hat { y } _ { i , p , k } ^ { \mathrm { t r a n s } } , \{ \hat { r } _ { i , p , k , m } \} _ { m = 1 } ^ { M } \right) .\tag{9}
$$

The edge identity is included in $\scriptstyle { \pmb { x } } _ { i , p , k }$ , while $p$ provides its candidate-path context; all outputs remain associated with $e _ { p , k }$ . In particular, $\hat { r } _ { i , p , k , m } \approx P ( \rho _ { i , p , k } \geq b _ { m } \mid \pmb { x } _ { i , p , k } , p , \mathcal { T } _ { t _ { i } } )$ Consistent with the nested labels, the predicted probabilities are constrained to satisfy $\hat { r } _ { i , p , k , 1 } ~ \ge ~ \cdots ~ \ge ~ \hat { r } _ { i , p , k , M }$ , with $m = M$ representing the most selective edge-level risk. Their path-level aggregation is defined next.

## C. Route Optimization Problem

For each candidate $p \in \mathcal { P } _ { i }$ , the predictor in (9) is evaluated edge by edge while $\mathcal { T } _ { t _ { i } }$ and the command-level attributes remain fixed. Using the edge scale $\mu _ { p , k }$ defined above, let $\hat { \rho } _ { i , p , k } = \tilde { y } _ { i , p , k } ^ { \mathrm { q u e u e } } / \mu _ { p , k }$ denote the normalized predicted queueing ratio.

The edge-level time predictions are aggregated as $\begin{array} { r } { \tilde { y } _ { i } ^ { \mathrm { q u e u e } } ( p ) ~ { = } ~ \sum _ { k = 1 } ^ { L _ { p } } \tilde { y } _ { i , p , k } ^ { \mathrm { q u e u e } } } \end{array}$ and $\begin{array} { r } { \hat { y } _ { i } ^ { \mathrm { t r a n s } } ( p ) ~ = ~ \sum _ { k = 1 } ^ { \bar { L } _ { p } } \hat { y } _ { i , p , k } ^ { \mathrm { t r a n s } } } \end{array}$ Accordingly,

$$
\begin{array} { r l r } {  { \hat { y } _ { i } ^ { \mathrm { t o t a l } } ( p ) = \tilde { y } _ { i } ^ { \mathrm { q u e u e } } ( p ) + \hat { y } _ { i } ^ { \mathrm { t r a n s } } ( p ) } } \\ & { } & { = \displaystyle \sum _ { k = 1 } ^ { L _ { p } } ( \tilde { y } _ { i , p , k } ^ { \mathrm { q u e u e } } + \hat { y } _ { i , p , k } ^ { \mathrm { t r a n s } } ) . } \end{array}\tag{10}
$$

Thus, queueing severity is normalized at the edge level and converted back to seconds for the path-level delivery-time objective.

![](images/b892d97c66088110d1c094f13783a37a312dfb23995a058ff4211939a28e2941.jpg)  
Solid line: prediction moment t , at which the command has been received but no route has been assigned. Dashed line: lagged telemetry cutoff $\bar { t _ { i } } = t _ { i } - \tau _ { l a g } ,$ with τ<sub>lag</sub> = 60 s in the implementation. Shaded: not observable at t . Only the as-of information set may enter the representation, per the prediction-time-safety invariant. Time axis not to scale.  
Fig. 2. Information boundary at the path-decision moment. Minute-level telemetry is truncated at $\bar { t } _ { i } = t _ { i } - \tau _ { \mathrm { l a g } } ,$ whereas completed-edge histories are truncated at t . Both cutoffs precede path execution, and the shaded regions are unavailable at the decision moment.

At ordinal level $m ,$ the edge-level probabilities are aggregated into the path-level bottleneck-risk index

$$
\hat { r } _ { i , m } ( p ) = \operatorname* { m a x } _ { 1 \leq k \leq L _ { p } } \hat { r } _ { i , p , k , m } .\tag{11}
$$

This aggregation requires no independence assumption, and $\hat { r } _ { i , m } ( p ) \leq \varepsilon$ requires every edge in p to satisfy the probability tolerance. However, $\hat { r } _ { i , m } ( p )$ is a bottleneck-risk index rather than the calibrated probability that at least one edge on the path exceeds its threshold.

Using the highest ordinal level M, define the surviving candidate set

$$
\mathcal { P } _ { i } ^ { ( \varepsilon ) } = \left\{ p \in \mathcal { P } _ { i } : \hat { r } _ { i , M } ( p ) \leq \varepsilon \right\} .\tag{12}
$$

If $\mathcal { P } _ { i } ^ { ( \varepsilon ) } \neq \varnothing$ , the selected path is $p _ { i } ^ { \star } = \arg \operatorname* { m i n } _ { p \in \mathcal { P } _ { i } ^ { ( \varepsilon ) } } \hat { y } _ { i } ^ { \mathrm { t o t a l } } ( p )$ Otherwise, $\begin{array} { r } { p _ { i } ^ { \star } = \arg \operatorname* { m i n } _ { p \in \mathcal { P } _ { i } } \hat { r } _ { i , M } ( p ) } \end{array}$ , with lower $\hat { y } _ { i } ^ { \mathrm { t o t a l } } ( p )$ used to break ties.

This threshold-constrained rule is used throughout the paper. Because all candidate edges are evaluated using the same $\mathcal { T } _ { t _ { i } }$ path comparison preserves the prediction-time-safety invariant in (4). Candidate generation, support filtering, and implementation of the selection rule are described in Section IV-C.

## IV. METHODOLOGY

This section instantiates the formulation in Section III through TN-DCR feature construction, edge-level prediction and stock-key calibration, and candidate-path selection.

## A. Overview

Fig. 3 summarizes the framework. At prediction moment $t _ { i } ,$ TN-DCR combines command and edge attributes, candidatepath context, the reference graph, and as-of network information into an edge-level representation satisfying (4). This representation is shared by two LightGBM regressors for queue and transfer time and an ordinal multi-label LightGBM classifier, as formalized in (9). The stock-key calibration is applied to the queue estimate, after which the edge-level outputs are aggregated according to (10) and (11) for candidate-path selection.

Static graph artifacts and graph-derived feature construction are maintained independently of downstream LightGBM fitting, and the path-selection stage requires no additional model training.

## B. Prediction Model

For each evaluated edge $e _ { p , k }$ , TN-DCR decomposes the representation in (4) as

$$
\phi _ { i , p , k } = \left[ \phi _ { i , p , k } ^ { \mathrm { s t r } } \rVert \phi _ { i , p , k } ^ { \mathrm { c t x } } \rVert \phi _ { i , p , k } ^ { \mathrm { g r a p h } } \right] \in \mathbb { R } ^ { d _ { \phi } } ,\tag{13}
$$

where the three blocks encode structural information, dynamic congestion context, and graph-aware neighborhood information, respectively. Their modular construction supports the feature ablations reported in Section V-C.

1) Static Transport Graph and Structural Priors: The graph G defined in Section III-A is materialized from the reference partition after removing self-loops, invalid endpoints, and duplicate edges; traversal frequencies are estimated from the same data. It therefore represents historically observed directed transitions rather than the complete set of operationally admissible paths.

Incoming and outgoing neighbors are ordered by referenceperiod traversal frequency, and at most $K _ { \mathcal { N } } = 4 8$ neighbors are retained for graph aggregation. Nodes and edges outside this fixed reference graph are treated as unsupported.

The structural block $\phi _ { i , p , k } ^ { \mathrm { s t r } }$ includes the numbers of distinct path nodes and edges, endpoint equipment descriptors and location prefixes of $e _ { p , k }$ , whether its endpoints share a base location, element and sub-element types along $p ,$ and sparse indicators for frequently traversed edges. The latter retain $K _ { \mathrm { m a c } } = 6 0$ entries for the macro structural table and $K _ { \mathrm { c o n g } } = 8 0$ for the congestion-oriented table. These features depend only on the command, evaluated edge, candidate path, and reference-period statistics.

![](images/b06707b36274817887119f026f2f54532cdf23a8c4a705b429a5410d5077b7d7.jpg)  
Prediction-time safety: every feature and candidate score uses only information observed before the decision moment; minute-level telemetry is truncated 60 s earlier.  
Fig. 3. Overview of the TN-DCR framework. Structural, dynamic-context, and graph-aware features produce edge-level queue-time, transfer-time, and ordinal risk predictions. The corrected edge-level outputs are aggregated to evaluate and select among candidate paths.

2) Dynamic Congestion Context: The context block $\phi _ { i , p , } ^ { \mathrm { c t x } }$ k describes the network-wide and path-local state preceding the decision moment. The event stream is aggregated into oneminute buckets containing waiting- and moving-event counts, zone-busy activity, active-carrier counts, and crane-state indicators.

To exclude a potentially incomplete current bucket, telemetry uses

$$
\bar { t } _ { i } = t _ { i } - \tau _ { \mathrm { l a g } } , \qquad \tau _ { \mathrm { l a g } } = 6 0 \mathrm { s } ,\tag{14}
$$

whereas completed-edge histories use the cutoff $t _ { i } .$

For each $w \in \mathcal { W } = \{ 1 5 , 6 0 \}$ minutes, the network-wide aggregate is $\begin{array} { r c l } { { \psi _ { i } ^ { \mathrm { n e t } , w } } } & { { = } } & { { \sum _ { \tau _ { j } \in [ \bar { t } _ { i } - 6 0 w , \bar { t } _ { i } ) } \zeta _ { j } } } \end{array}$ . The half-open interval excludes observations at or after t<sup>¯</sup><sub>i</sub>, and comparison with the preceding interval of equal length captures short-term load changes.

Let $\mathcal V ( p ) = \{ v _ { p , 0 } , . . . , v _ { p , L _ { p } } \}$ , with $\mathcal { E } ( p )$ defined in (1). Over these path nodes and edges, the model computes event counts $\operatorname { c n t } ( v , w ; \bar { t } _ { i } )$ and intensity sums $\mathrm { i n t } _ { a } ( v , w ; \bar { t } _ { i } ) , \ a \ \in$ {wait, move, zone}, and aggregates them by their sum, mean, maximum, active fraction, and concentration among the most active elements.

The waiting-to-moving intensity ratio is $\eta _ { i } ^ { w } ( p )$ = $\begin{array} { r } { \left( \sum _ { v \in \mathcal { V } ( p ) } \operatorname* { i n t } _ { \mathrm { w a i t } } ( v , w ; \bar { t } _ { i } ) \right) \big / \left( \sum _ { v \in \mathcal { V } ( p ) } \operatorname* { i n t } _ { \mathrm { m o v e } } ( v , w ; \bar { t } _ { i } ) + \epsilon _ { 0 } \right) , } \end{array}$ where $\epsilon _ { 0 } > 0$ . It measures path-local waiting activity relative to movement and is distinct from the edge-level target ratio $\rho _ { i , p , k }$ . All context features use observations preceding $\bar { t } _ { i }$

3) Graph-Aware Edge Representation: Unlike the pathwide context block, $\phi _ { i , p , k } ^ { \mathrm { g r a p h } }$ depends on the static graph connectivity around $e _ { p , k }$ , independently of the remaining composition of $p .$

For each node $v ,$ temporally ordered cumulative event-count and intensity arrays are maintained. Binary search locates observations satisfying $\tau _ { j } < \bar { t } _ { i }$ , and cumulative differences over each $w \in \mathcal W$ produce the windowed node state. Concatenating the two windows gives $h _ { v } ( \bar { t } _ { i } ) \in \mathbb { R } ^ { d _ { h } }$

For $\begin{array} { c c l } { e _ { p , k } } & { = } & { \left( v _ { p , k - 1 } , v _ { p , k } \right) } \end{array}$ , define its endpoint set as $\mathcal { V } ( e _ { p , k } ) ~ = ~ \{ v _ { p , k - 1 } , v _ { p , k } \}$ and its external neighborhood as $\begin{array} { r } { \mathcal { N } ( \bar { e } _ { p , k } ) = ( \bigcup _ { v \in \mathcal { V } ( e _ { p , k } ) } \mathcal { N } ( v ) ) \backslash \mathcal { V } ( e _ { p , k } ) } \end{array}$ . At most $K _ { \mathcal { N } }$ neighbors are retained according to the reference-period ordering.

For any non-empty node set A, define

$$
\begin{array} { l } { { \displaystyle m _ { i } ^ { A , \mu } = \frac { 1 } { | A | } \sum _ { v \in A } h _ { v } ( \bar { t } _ { i } ) , } } \\ { { \displaystyle m _ { i } ^ { A , \operatorname* { m a x } } = \operatorname* { m a x } _ { v \in A } h _ { v } ( \bar { t } _ { i } ) . } } \end{array}
$$

Both vectors are set to zero when $A = \emptyset$ . Applying (15) to $\nu ( e _ { p , k } )$ and $\mathcal { N } ( e _ { p , k } )$ gives their mean and maximum states. Their mean-state contrast is $\chi _ { i , p , k } = m _ { i } ^ { \mathcal { V } ( e _ { p , k } ) , \mu } - m _ { i } ^ { \mathcal { N } ( e _ { p , k } ) , \mu }$

The four pooled vectors and $\chi _ { i , p , k }$ are concatenated into $\boldsymbol { c } _ { i , p , k } \in \mathbb { R } ^ { 5 d _ { h } }$ . To reduce skew, this vector is transformed as $\tilde { c } _ { i , p , k } = \mathrm { s i g n } ( \pmb { c } _ { i , p , k } ) \odot \mathrm { l o g } ( 1 + | \pmb { c } _ { i , p , k } | )$ and standardized using reference-period statistics as $\hat { \pmb { c } } _ { i , p , k } = ( \tilde { \pmb { c } } _ { i , p , k } - { \pmb { \mu } } _ { c } ) \oslash { \pmb { \sigma } } _ { c } .$

Graph-aware segment representation: endpoint and one-hop neighbourhood pooling

![](images/3bb9c5937e9315558cc948c3e637987cae89dfb23b04a8965f500fbcd20de285.jpg)

![](images/535a7aecaa27daea01fdabae4d19009dc3a55b1419702a00df4c42be0e21442e.jpg)  
$\varphi _ { i , p , k _ { c t x + s } }$ is concatenated with the structural block $\varphi _ { i , p , k } ^ { \phantom { } } ^ { s t r }$ and the dynamic-context block φ<sub>i,p,k</sub><sup>ctx</sup> to form $\varphi _ { i , p , k _ { , \ast } ^ { 3 } }$ the same vector feeds both regressors and the ordinal congestion-risk classifier. Because W is fixed, graph feature construction is decoupled from downstream model fitting and can be recomputed for changing route compositions rather than extrapolated to unsupported regions.  
Fig. 4. Graph-aware edge representation. Endpoint and neighboring-node states are mean/max pooled and contrasted, transformed by a fixed untrained projection, and concatenated with structural and dynamic support diagnostics.

A fixed nonlinear projection produces $\begin{array} { r l } { \phi _ { i , p , k } ^ { \mathrm { e m b } } } & { { } = } \end{array}$ tanh $\begin{array} { r l r } { ( { \bf W } ^ { \top } \hat { \pmb { c } } _ { i , p , k } ) } & { { } \in } & { \mathbb { R } ^ { d _ { e } } } \end{array}$ where $\begin{array} { r l r } { { \bf W } } & { { } \in } & { \mathbb { R } ^ { 5 d _ { h } \times d _ { e } } } \end{array}$ $W _ { a b } \sim \mathcal { N } ( 0 , 1 / ( 5 d _ { h } ) )$ , is initialized once and not optimized; $d _ { e } \ = \ 1 6 .$ . The shared projection provides a reproducible feature map for evaluated edges whose endpoint states and graph neighborhoods are represented, without implying reliable extrapolation to unsupported graph regions.

Two support diagnostics are appended to the embedding. With reference traversal count $n ^ { \mathrm { r e f } } ( e _ { p , k } )$ , the structural indicator is $s _ { p , k } ^ { \mathrm { s t r } } ~ = ~ 1 [ n ^ { \mathrm { r e f } } ( e _ { p , k } ) ~ \geq ~ \bar { n } _ { \mathrm { m i n } } ]$ . Let $\mathcal { H } ( e _ { p , k } ) \ : = \ :$ $\mathcal { V } ( e _ { p , k } ) \cup \mathcal { N } ( e _ { p , k } )$ ; the dynamic support ratio is $s _ { i , p , k } ^ { \mathrm { d y n } } = | \{ v \in$ $\mathcal { H } ( e _ { p , k } ) : h _ { v } ( \bar { t } _ { i } )$ is available $\} | / | \mathcal { H } ( e _ { p , k } )$ |. The complete block is

$$
\phi _ { i , p , k } ^ { \mathrm { g r a p h } } = [ \phi _ { i , p , k } ^ { \mathrm { e m b } } \| s _ { p , k } ^ { \mathrm { s t r } } \| s _ { i , p , k } ^ { \mathrm { d y n } } ] .
$$

4) Prediction Models and Stock-Key Residual Correction: Two separately fitted LightGBM regressors map $\phi _ { i , p , k }$ to $\hat { y } _ { i , p , k } ^ { \mathrm { q u e u e } } ~ = ~ f _ { \mathrm { q u e u e } } ( \phi _ { i , p , k } )$ and $\hat { y } _ { i , p , k } ^ { \mathrm { t r a n s } } ~ = ~ f _ { \mathrm { t r a n s } } ( \phi _ { i , p , k } )$ . No separate total-time regressor is trained.

Using the labels in (8), an ordinal multi-label LightGBM model produces $\begin{array} { r l r } { \widehat { \pmb { r } } _ { i , p , k } } & { { } = } & { f _ { \mathrm { r i s k } } ( \phi _ { i , p , k } ) } \end{array}$ = $( \hat { r } _ { i , p , k , 1 } , \dotsc , \hat { r } _ { i , p , k , M } )$ . The probabilities are calibrated on held-out reference-period data and constrained to satisfy $\hat { r } _ { i , p , k , 1 } \ge \ \cdots \ge \hat { r } _ { i , p , k , M }$ . The highest-threshold output is the most selective edge-level risk score; no separate extreme-congestion classifier is trained. The queue regressor and ordinal classifier are optimized independently, providing complementary point and exceedance-probability estimates.

Persistent stock-key queue residuals are handled using empirical-Bayes shrinkage. For a training edge record, let $\hat { y } _ { i , p , k } ^ { \mathrm { q u e u e , O O F } }$ be its out-of-fold prediction and define $\xi _ { i , p , k } =$ $y _ { i , p , k } ^ { \mathrm { { \scriptsize ~ { \mathrm { \scriptsize ~ { \sc ~ \gamma ~ } ~ } } } } } - \hat { y } _ { i , p , k } ^ { \mathrm { { \scriptsize ~ { \mathrm { \scriptsize ~ { u e u e , O O F } } } } } }$ . For stock key g, let $\mathcal { I } _ { g } = \{ ( i , p , k ) :$ $g _ { i , p , k } = g \} , n _ { g } = | \mathcal { I } _ { g } |$ , and $\begin{array} { r } { \bar { \xi } _ { g } = n _ { g } ^ { - 1 } \sum _ { ( i , p , k ) \in \mathcal { I } _ { g } } \xi _ { i , p , k } } \end{array}$ . The correction is $\begin{array} { r } { c _ { g } = \frac { n _ { g } } { n _ { a } + \kappa } \bar { \xi } _ { g } . } \end{array}$ , where $\kappa > 0$ shrinks estimates for sparsely observed keys toward zero.

The corrected edge-level queue estimate is $\begin{array} { r l } { \tilde { y } _ { i , p , k } ^ { \mathrm { q u e u e } } } & { { } = } \end{array}$ max $( 0 , \hat { y } _ { i , p , k } ^ { \mathrm { q u e u e } } + c _ { g _ { i , p , k } } )$ , with $c _ { g _ { i , p , k } } = 0$ for unseen keys; the transfer estimate is used directly. The correction targets persistent stock-key bias rather than uniform improvement across all regression metrics.

## C. Route Optimization Algorithm

The route-selection stage evaluates the admissible set $\mathcal { P } _ { i } =$ $\mathcal { P } ( o _ { i } , d _ { i } )$ . When this set is supplied by the lower-level dispatcher, it is used directly; otherwise, a bounded set is generated by loopless $K _ { p }$ -shortest-path enumeration on G, using the reference-period median edge-level delivery times as base weights.

Candidates that violate the feasibility requirements of Section III-A or the minimum structural and dynamic support requirements are removed. If none remain, the command is passed to the incumbent routing fallback rather than evaluated outside the empirical support of the prediction model.

For each remaining path, the predictor in (9) is evaluated over its constituent edges using the same command attributes and $\mathcal { T } _ { t _ { i } }$ . The resulting edge-level outputs are aggregated using (10) and (11), after which the threshold-constrained rule in Section III-C is applied with risk tolerance ε. Algorithm 1 summarizes the procedure.

Because the training edges belong to paths historically selected by the incumbent dispatcher, predictions for alternative candidates are observational rather than interventional. Support filtering limits extrapolation but does not eliminate selection bias; the policy therefore compares empirically supported candidates rather than estimating causal outcomes for arbitrary paths.

Algorithm 1 Per-command TN-DCR path selection   
Require: command $c _ { i }$ with origin $o _ { i } ,$ , destination $d _ { i } ,$ and pre  
diction moment $t _ { i } ;$ reference graph $\mathcal { G } ;$ candidate source;   
trained edge-level predictors; support thresholds; risk tol  
erance ε   
Ensure: selected path $p _ { i } ^ { \star }$   
1: obtain admissible candidate set $\mathcal { P } _ { i } = \mathcal { P } ( o _ { i } , d _ { i } )$   
2: if the dispatcher does not provide $\mathcal { P } _ { i }$ then   
3: generate at most $K _ { p }$ loopless $o _ { i } { - } d _ { i }$ paths using   
reference-period edge weights   
4: end if   
5: remove candidates violating feasibility or minimum  
support requirements   
6: if $\mathcal { P } _ { i } = \emptyset$ then   
7: return incumbent routing fallback   
8: end if   
9: for all $p \in \mathcal { P } _ { i }$ do   
10: for $k = 1 , \ldots , L _ { p }$ do   
11: construct $\phi _ { i , p , k }$ from $x _ { i , p , k } , p ,$ and $\mathcal { T } _ { t } .$   
12: evaluate the edge-level predictor in (9)   
13: end for   
14: compute $\hat { y } _ { i } ^ { \mathrm { t o t a l } } ( p )$ using (10)   
15: compute $\{ \bar { r } _ { i , m } ( p ) \} _ { m = 1 } ^ { M }$ using (11)   
16: end for   
17: form $\mathcal { P } _ { i } ^ { ( \varepsilon ) }$ using (12)   
18: if $\mathcal { P } _ { i } ^ { ( \varepsilon ) } \neq \varnothing$ then   
19: $p _ { i } ^ { \dot { \star } } \gets \mathrm { a r g } \operatorname* { m i n } _ { p \in \mathcal { P } _ { i } ^ { ( \varepsilon ) } } \hat { y } _ { i } ^ { \mathrm { t o t a l } } ( p )$   
20: else   
21: $\begin{array} { r } { p _ { i } ^ { \star } \gets \arg \operatorname* { m i n } _ { p \in \mathcal { P } _ { i } } \hat { r } _ { i , M } ( p ) } \end{array}$   
22: break ties using lower $\hat { y } _ { i } ^ { \mathrm { t o t a l } } ( p )$   
23: end if   
24: return $p _ { i } ^ { \star }$

The policy is also myopic: it uses the network state at $t _ { i }$ without anticipating how future command reassignments will alter that state. Section V-D evaluates this policy in a controlled closed-loop setting, while long-horizon fab-scale validation remains outside the scope of this study.

Internal generation considers at most $K _ { p }$ paths, with at most $L _ { \mathrm { m a x } }$ edge-level evaluations per path and $K _ { \mathcal { N } }$ retained graph neighbors per evaluation. Time-indexed histories are accessed by binary search, so candidate scoring is capped while history lookup grows only logarithmically with the event corpus. End-to-end serving latency remains an empirical deployment metric.

## V. NUMERICAL EXPERIMENTS

Using historical edge-level transport records from a semiconductor display manufacturing facility, we evaluate time regression, ordinal congestion-risk classification, feature and stock-key-correction ablations, and controlled closed-loop path selection.

## A. Experimental Setup

a) Data and prediction protocol.: The dataset contains 51,700 successful edge-level transport records collected from a production MCS in a semiconductor display fab. Each record is constructed at its command-specific prediction moment $t _ { i }$ and therefore obeys the information boundary in (4). The reference graph contains $| \nu | = 7 6 2$ nodes and $| \mathcal { E } | = 1 , 5 4 9$ directed edges; approximately 89% of adjacent node pairs occur in only one direction. Queue time is strongly rightskewed, with mean 63.3 s, median 9 s, P95 285 s, and P99 790 s. Transfer time is less skewed, with mean 116.7 s, median $7 7 { \mathrm { s } } ,$ and a P95 approximately four times its median. The earliest 80% of records (41,360) are used for training and the latest 20% (10,340) for testing. The graph, reference statistics, and all other data-derived quantities are estimated from the training period only.

b) Regression targets and models.: Separate LightGBM regressors predict edge-level queue and transfer times using the shared representation $\phi _ { i , p , k }$ in (13) [23]. The stock-key calibration in Section IV-B4 is applied only to the queue output, while the transfer output is used directly. Edge-level total time is reconstructed from the two components; no separate total-time regressor is trained.

c) Ordinal congestion-risk labels.: The edge-level ordinal targets follow the normalized ratio in (6), with transfer scales estimated from the training period using (5). We set $p _ { m } \in \{ 0 . 8 5 , 0 . 9 0 , 0 . 9 5 \}$ in (7), producing the P85, P90, and P95 labels in (8). Because these thresholds are estimated from the training distribution, their test prevalences need not equal exactly 15%, 10%, and 5%.

The ordinal multi-label LightGBM classifier in Section IV-B4 uses the same TN-DCR representation and produces calibrated edge-level probabilities constrained to be nonincreasing with congestion severity. These probabilities are evaluated as edge-level ranking scores and aggregated by (11) for path selection.

d) Input features and model configuration.: All learned models consume the same structural, dynamic-context, and graph-aware blocks in (13). Dynamic telemetry is truncated at $\bar { t } _ { i }$ according to (14), while completed-edge histories are truncated at $t _ { i } .$

e) Candidate regressors and non-learned references.: History mean and History median assign the corresponding training-period statistic to every test record. The learned candidates are Random Forest [27], ExtraTrees [28], CatBoost [24], XGBoost [22], and LightGBM. All use the same chronological split, TN-DCR representation, training and evaluation protocol, and stock-key calibration, so the comparison isolates the choice of regressor. Each learned candidate is evaluated with five fixed random seeds (42, 365, 1234, 3407, and 114514), and results are reported as mean ± standard deviation; the two history references are deterministic. Feature contributions are evaluated separately in Section V-C.

f) Evaluation metrics.: All metrics are computed on the chronological test partition. Let $j \in \{ 1 , \ldots , n \}$ index an edgelevel record, with target $y _ { j }$ , nonnegative prediction $\hat { y } _ { j } .$ , signed error $\delta _ { j } = \hat { y } _ { j } - y _ { j }$ , positive-target set $\mathcal { I } _ { + } = \{ j : y _ { j } > 0 \}$ , and

TABLE I  
EDGE-LEVEL REGRESSOR SELECTION OVER FIVE RANDOM SEEDS.
<table><tr><td>Target</td><td>Method</td><td>MAE (s)</td><td></td><td>RMSLE Cov@25 (%) Cov@50 (%)</td><td></td></tr><tr><td rowspan="7">Queue</td><td>History mean</td><td> $7 4 . 8 \pm 0 . 0$ </td><td> $2 . 3 6 2 \pm 0 . 0 0 0$ </td><td> $1 4 . 4 \pm 0 . 0$ </td><td> $2 9 . 9 \pm 0 . 0$ </td></tr><tr><td>History median</td><td> $6 8 . 4 \pm 0 . 0$ </td><td> $2 . 2 0 2 \pm 0 . 0 0 0$ </td><td> $1 4 . 0 \pm 0 . 0$ </td><td> $2 9 . 9 \pm 0 . 0$ </td></tr><tr><td>Random Forest</td><td> $6 1 . 9 \pm 0 . 2$ </td><td> $2 . 2 6 8 \pm 0 . 0 0 6$ </td><td> $2 1 . 7 \pm 0 . 2$ </td><td> $4 0 . 9 \pm 0 . 1$ </td></tr><tr><td>ExtraTrees</td><td> $6 3 . 6 \pm 0 . 2$ </td><td> $2 . 3 4 5 \pm 0 . 0 0 9$ </td><td> $2 0 . 0 \pm 0 . 2$ </td><td> $3 9 . 8 \pm 0 . 1$ </td></tr><tr><td>CatBoost</td><td> $5 1 . 1 \pm 0 . 4$ </td><td> $\mathbf { 1 . 9 1 7 \ : \pm { \ : 0 . 0 1 5 } }$ </td><td> $2 0 . 5 \pm 0 . 3$ </td><td> $4 0 . 9 \pm 0 . 1$ </td></tr><tr><td>XGBoost</td><td> $5 1 . 1 \pm 0 . 6$ </td><td> $1 . 9 2 8 \pm 0 . 0 3 2$ </td><td> $2 1 . 0 \pm 0 . 4$ </td><td> $4 1 . 2 \pm 0 . 3$ </td></tr><tr><td>LightGBM</td><td> ${ \bf 4 9 . 6 \pm 0 . 2 }$ </td><td> $1 . 9 2 0 \pm 0 . 0 1 3$ </td><td> ${ \bf \ } 2 2 . 6 \pm { \bf \delta } 0 . 2$ </td><td> ${ \bf 4 3 . 6 \pm 0 . 3 }$ </td></tr><tr><td rowspan="7">Transfer</td><td>History mean</td><td> $4 8 . 8 \pm 0 . 0$ </td><td></td><td> $4 6 . 1 \pm 0 . 0$ </td><td> $7 0 . 6 \pm 0 . 0$ </td></tr><tr><td>History median</td><td> $4 1 . 3 \pm 0 . 0$ </td><td> $0 . 4 8 5 \pm 0 . 0 0 0$   $0 . 4 8 0 \pm 0 . 0 0 0$ </td><td> $6 7 . 6 \pm 0 . 0$ </td><td> $8 6 . 0 \pm 0 . 0$ </td></tr><tr><td>Random Forest</td><td> $4 7 . 0 \pm 0 . 3$ </td><td></td><td></td><td> $7 4 . 0 \pm 0 . 4$ </td></tr><tr><td>ExtraTrees</td><td> $4 3 . 2 \pm 0 . 5$ </td><td> $0 . 4 5 2 \pm 0 . 0 0 3$ </td><td> $4 9 . 5 \pm 0 . 3$ </td><td></td></tr><tr><td>CatBoost</td><td></td><td> $0 . 4 2 0 \pm 0 . 0 0 5$ </td><td> $4 3 . 7 \pm 1 . 2$   $7 6 . 9 \pm 0 . 4$ </td><td> $7 1 . 3 \pm 0 . 8$ </td></tr><tr><td>XGBoost</td><td> $2 8 . 1 \pm 0 . 1$   $2 9 . 6 \pm 0 . 5$ </td><td> $0 . 3 2 3 \pm 0 . 0 0 1$   $0 . 3 3 4 \pm 0 . 0 0 3$ </td><td> $7 4 . 8 \pm 1 . 2$ </td><td> $9 2 . 8 \pm 0 . 3$ </td></tr><tr><td></td><td></td><td></td><td></td><td> $9 1 . 2 \pm 0 . 8$ </td></tr><tr><td rowspan="8">Total</td><td>LightGBM</td><td> $2 7 . 4 \pm { \bf 0 . 1 }$ </td><td> $\mathbf { 0 . 3 1 3 \ : \pm { \ : 0 . 0 0 1 } }$ </td><td> ${ \bf 7 7 . 9 \pm 0 . 5 }$ </td><td> ${ \bf 9 3 . 2 \pm 0 . 2 }$ </td></tr><tr><td>History mean</td><td> $1 0 3 . 0 \pm 0 . 0$ </td><td> $0 . 6 5 7 \pm 0 . 0 0 0$ </td><td> $3 1 . 2 \pm 0 . 0$ </td><td> $5 6 . 3 \pm 0 . 0$ </td></tr><tr><td>History median</td><td> $9 6 . 3 \pm 0 . 0$ </td><td> $0 . 6 4 3 \pm 0 . 0 0 0$ </td><td> $3 5 . 7 \pm 0 . 0$ </td><td> $6 3 . 6 \pm 0 . 0$ </td></tr><tr><td>Random Forest</td><td> $9 2 . 6 \pm 0 . 4$ </td><td> $0 . 5 8 4 \pm 0 . 0 0 2$ </td><td> $3 5 . 1 \pm 0 . 3$ </td><td> $6 0 . 1 \pm 0 . 2$ </td></tr><tr><td>ExtraTrees</td><td> $9 0 . 3 \pm 0 . 4$ </td><td> $0 . 5 8 9 \pm 0 . 0 0 2$ </td><td> $3 3 . 2 \pm 0 . 4$ </td><td> $5 5 . 9 \pm 0 . 2$ </td></tr><tr><td>CatBoost</td><td> $6 7 . 1 \pm 0 . 3$ </td><td> $0 . 4 4 6 \pm 0 . 0 0 2$ </td><td> $5 1 . 6 \pm 0 . 4$ </td><td> $7 9 . 1 \pm 0 . 3$ </td></tr><tr><td>XGBoost</td><td> $6 8 . 3 \pm 0 . 5$ </td><td> $0 . 4 5 4 \pm 0 . 0 0 3$ </td><td> $5 0 . 8 \pm 0 . 6$ </td><td> $7 8 . 7 \pm 0 . 7$ </td></tr><tr><td>LightGBM</td><td> ${ \bf 6 4 . 9 \pm 0 . 2 }$ </td><td> $\mathbf { 0 . 4 3 3 \ : \pm { \ : 0 . 0 0 1 } }$ </td><td> ${ \bar { \bf s } } 2 . 9 \pm { \bf 0 . 4 }$ </td><td> ${ \bf 7 9 . 7 \pm 0 . 3 }$ </td></tr></table>

$n _ { + } = | \mathcal { I } _ { + } |$ . We report

$$
\mathrm { M A E } = \frac { 1 } { n } \sum _ { j = 1 } ^ { n } | \delta _ { j } | ,
$$

$$
\mathrm { R M S L E } = \sqrt { \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \left[ \log ( 1 + \hat { y } _ { j } ) - \log ( 1 + y _ { j } ) \right] ^ { 2 } } ,
$$

$$
\mathrm { C o v @ } \tau = \frac { 1 } { n _ { + } } \sum _ { j \in \mathcal { I } _ { + } } \mathbb { 1 } \left[ \frac { | \delta _ { j } | } { y _ { j } } \leq \tau \right] , \qquad \tau \in \{ 0 . 2 5 , 0 . 5 0 \} .
$$

MAE is reported in seconds, RMSLE measures logarithmic error, and Cov@25 and Cov@50 are the percentages of positivetarget records whose absolute relative errors are within 25% and 50%. Records with $y _ { j } = 0$ are excluded from coverage because their relative errors are undefined. Lower MAE and RMSLE and higher coverage indicate better performance. For the stock-key analysis, $\begin{array} { r } { \operatorname { B i a s } = n ^ { - 1 } \sum _ { j = 1 } ^ { n } \delta _ { j } } \end{array}$ , where a negative value indicates systematic underprediction.

For each congestion level, we report prevalence, PR-AUC, ROC-AUC, P@1%, and P@5%. The latter two are the positive fractions among the top 1% and 5% of edge-level records ranked by the corresponding exceedance probability. PR-AUC emphasizes performance under class imbalance, while top-ranked precision measures the concentration of high-risk records within a limited intervention set.

## B. Prediction and Classification Results

a) Regression model selection.: As shown in Table I, LightGBM provides the strongest overall performance. For queue time, it achieves the lowest MAE $( 4 9 . 6 \pm 0 . 2 \mathrm { s } )$ and the highest Cov@25 and Cov@50 (22.6±0.2% and $4 3 . 6 { \pm } 0 . 3 \% )$ CatBoost has a marginally lower queue RMSLE $( 1 . 9 1 7 { \scriptstyle \pm 0 . 0 1 5 }$ versus $1 . 9 2 0 \pm 0 . 0 1 3 )$ , but the difference is small relative to the run-to-run variation. LightGBM achieves the best results on all transfer- and reconstructed-total-time metrics, including MAEs of $2 7 . 4 \pm 0 . 1 \mathrm { s }$ and $6 4 . 9 \pm 0 . 2 \mathrm { s } ,$ , respectively, and is therefore selected for both regressors.

TABLE II  
EDGE-LEVEL CONGESTION-RISK CLASSIFICATION OVER FIVE RANDOM SEEDS.
<table><tr><td>Level</td><td> $b _ { m }$ </td><td>Prev. (%)</td><td> $\mathrm { P R - A U C }$ </td><td>ROC-AUC</td><td>P@1%</td><td></td></tr><tr><td>P85</td><td>1.051</td><td></td><td> $1 7 . 6 \pm 0 . 0 \ 0 . 7 2 2 \pm 0 . 0 0 4$ </td><td> $0 . 9 2 8 \pm 0 . 0 0 0$ </td><td> $0 . 9 4 0 \pm 0 . 0 2 6$ </td><td> $0 . 8 9 2 \pm 0 . 0 1 1$ </td></tr><tr><td>P90</td><td>1.527</td><td></td><td> $1 1 . 8 \pm 0 . 0 \ 0 . 6 3 0 \pm 0 . 0 0 4$ </td><td> $0 . 9 2 9 \pm 0 . 0 0 0$ </td><td> $0 . 9 1 3 \pm 0 . 0 2 5$ </td><td> $0 . 7 8 2 \pm 0 . 0 0 6$ </td></tr><tr><td>P95</td><td>2.669</td><td></td><td> $5 . 9 \pm 0 . 0 \ 0 . 5 8 1 \pm 0 . 0 0 4$ </td><td> $0 . 9 4 7 \pm 0 . 0 0 1$ </td><td> $0 . 8 6 2 \pm 0 . 0 1 6$ </td><td> $0 . 6 0 2 \pm 0 . 0 0 2$ </td></tr></table>

TABLE III

EDGE-LEVEL QUEUE-TIME FEATURE ABLATION OVER FIVE RANDOM SEEDS.
<table><tr><td>Variant</td><td>MAE (s)</td><td>RMSLE</td><td> $\mathrm { C o v } @ 2 5 ~ ( \% )$ </td><td>Cov@50 (%)</td></tr><tr><td>Full</td><td> ${ \bf 4 9 . 6 \pm 0 . 2 }$ </td><td> $\mathbf { 1 . 9 2 0 \pm 0 . 0 1 3 }$ </td><td> $2 2 . 6 \pm 0 . 2$ </td><td> ${ \bf 4 3 . 6 \pm 0 . 3 }$ </td></tr><tr><td>w/o graph</td><td> $5 2 . 1 \pm 0 . 1$ </td><td> $1 . 9 4 9 \pm 0 . 0 0 6$ </td><td> $2 1 . 0 \pm 0 . 1$ </td><td> $4 2 . 2 \pm 0 . 3$ </td></tr><tr><td>w/o dynamic</td><td> $5 1 . 1 \pm 0 . 6$ </td><td> $1 . 9 7 1 \pm 0 . 0 3 7$ </td><td> ${ \bf 2 2 . 9 \pm 0 . 1 }$ </td><td> $4 2 . 4 \pm 0 . 1$ </td></tr><tr><td>w/o graph &amp; dynamic</td><td> $5 4 . 4 \pm 0 . 4$ </td><td> $2 . 0 7 5 \pm 0 . 0 2 0$ </td><td> $2 2 . 1 \pm 0 . 3$ </td><td> $4 2 . 4 \pm \ : 0 . 1$ </td></tr></table>

b) Multi-threshold congestion-risk classification.: Table II reports the edge-level ordinal results. Test prevalences differ from their nominal training quantiles because of the chronological distribution shift. Across P85, P90, and P95, PR-AUC is $0 . 7 2 2 \pm 0 . 0 0 4 , 0 . 6 3 0 \pm 0 . 0 0 4$ , and $0 . 5 8 1 { \pm } 0 . 0 0 4 .$ , while ROC-AUC remains at least 0.928. At P95, $6 0 . 2 \pm 0 . 2 \%$ of records within the highest-risk 5% are positive, compared with an overall prevalence of 5.9%. These results characterize edgelevel ranking; path selection uses the bottleneck aggregation in (11).

## C. Ablation Study

a) Graph-aware and dynamic-contextfeatures.: Table III ablates $\phi _ { i , p , k } ^ { \mathrm { g r a p h } }$ and $\phi _ { i , p , k } ^ { \mathrm { c t x } }$ while holding the remaining pipeline fixed.

The full configuration gives the lowest MAE and RM-SLE and the highest Cov@50. Removing the graph-aware or dynamic-context block increases MAE from $4 9 . 6 \pm 0 . 2 \mathrm { s }$ to $5 2 . 1 \pm 0 . 1 \mathrm { s }$ or $5 1 . 1 \pm 0 . 6 \mathrm { s } ,$ respectively. Removing both increases MAE to $5 4 . 4 \pm 0 . 4 \mathrm { s } ~ ( 9 . 7 \% )$ , raises RMSLE from $1 . 9 2 0 \pm 0 . 0 1 3$ to $2 . 0 7 5 \pm 0 . 0 2 0$ , and lowers Cov@50 from $4 3 . 6 \pm 0 . 3 \%$ to $4 2 . 4 \pm 0 . 1 \%$ Although removing dynamic context slightly improves Cov@25 from $2 2 . 6 \pm 0 . 2 \%$ to $2 2 . 9 { \pm } 0 . 1 \%$ , the full representation is strongest overall, and the larger joint degradation indicates complementary contributions from the two blocks.

b) Stock-key residual correction.: Removing the correction changes Bias from $- 9 . 6 \pm 0 . 6 \mathrm { s }$ to $- 1 7 . 8 \pm 0 . 6 \mathrm { s }$ The correction therefore reduces the magnitude of systematic underprediction by 8.2 s, or 46.1%, although the remaining negative Bias indicates residual underestimation.

## D. Closed-Loop Routing Results

In a controlled closed-loop environment, we further compare TN-DCR routing with static shortest-path selection, which is a strategy that the factory previously used in actual operation, and the experiment tests whether aggregated edgelevel predictions produce operational differences.

a) Evaluation environment.: Because historical logs contain outcomes only for paths selected by the incumbent dispatcher, counterfactual policies are evaluated in a finitecapacity discrete-event simulator fitted exclusively from the training partition. On G, each edge and relay node is modeled as an independent multi-server FCFS resource, with capacities calibrated so that the busiest resource has target utilization 0.70. A selected path reserves its constituent resources and thereby changes the queues experienced by subsequent commands. Service times, capacities, and congestion slowdowns are fitted from raw logs without using TN-DCR predictions, providing all policies with the same independent ground-truth process.

Inter-arrival gaps, batch sizes, hourly demand, origin– destination pairs, and template paths are sampled from fitted or empirical training-period distributions. Edge service times follow the best-fitting exponential, log-normal, or gamma distribution and are divided into edge-movement and receivingnode handoff components. Bounded load-dependent slowdown, neighbor spillover, and rare incidents produce heavytailed congestion. Each replication generates 500 commands and discards the first 50 as warm-up. Results are averaged over five independent replications, with every policy evaluated on identical scenarios under a paired design.

b) Delivery-time components in the simulator.: The simulator follows (2) but uses a specific accounting convention in which waiting after acquisition of the first resource belongs to transfer time. For command $c _ { i } .$ , let $t _ { i } ^ { \mathrm { a r r } } , t _ { i } ^ { \mathrm { s t a r t } }$ , and $t _ { i } ^ { \mathrm { { c o m p } } }$ denote its arrival, first-edge service start, and completion times. With the selected path implicit,

$$
y _ { i } ^ { \mathrm { q u e u e } } = t _ { i } ^ { \mathrm { s t a r t } } - t _ { i } ^ { \mathrm { a r r } } , \qquad y _ { i } ^ { \mathrm { t r a n s } } = t _ { i } ^ { \mathrm { c o m p } } - t _ { i } ^ { \mathrm { s t a r t } } .\tag{15}
$$

Thus, queue time is the pre-dispatch delay, whereas transfer time spans the selected path and includes intermediate resource waiting. We additionally report

$$
y _ { i } ^ { \mathrm { i n t w a i t } } = \left( \sum _ { r \in \mathcal { U } ( p _ { i } ^ { \star } ) } w _ { i , r } \right) - y _ { i } ^ { \mathrm { q u e u e } } ,\tag{16}
$$

where $\mathcal { U } ( p _ { i } ^ { \star } )$ is the set of traversed edge and node resources and $w _ { i , r }$ is the waiting time at resource $^ { r } \cdot$ Hence $y _ { i } ^ { \mathrm i }$ ntwait isolates the intermediate waiting contained within $y _ { i } ^ { \mathrm { t r a n s } }$

c) Compared policies.: All policies use the same candidate set $\mathcal { P } _ { i } ,$ generated by loopless $K _ { p } .$ -shortest-path enumeration with $K _ { p } = 5$ . Static shortest path minimizes the sum of fixed edge weights given by historical median service times and does not react to the live state. TN-DCR routing applies Algorithm 1 with $\varepsilon = 0 . 0 5$ . Candidate detours are limited to 1.15× the shortest static cost to prevent unbounded path-length increases. Table IV shows that TN-DCR routing reduces mean and P95 total time by 16.40% and 10.33%, respectively. Mean queue, transfer, and internal waiting times decrease by 8.2%, 18.9%, and 22.59%.

The internal-wait reduction (127.60 s) is nearly equal to the transfer-time reduction (127.40 s), indicating that most of the transfer improvement comes from avoiding intermediate contention rather than shortening the physical path. This interpretation is consistent with the detour cap and with (15)–(16). The smaller queue-time reduction reflects the stronger dependence of pre-dispatch delay on origin-resource availability. Throughput remains nearly unchanged at 295.43 versus 295.68 commands per hour. The experiment therefore demonstrates lower delivery and waiting times without an observed throughput difference under this operating condition. Absolute simulated times exceed those in the offline dataset because the simulator maintains sustained utilization and a heavy-tailed service process, and the results should therefore be interpreted as relative policy comparisons under identical calibrated scenarios.

TABLE IV  
CLOSED-LOOP PERFORMANCE AVERAGED OVER FIVE PAIRED REPLICATIONS.
<table><tr><td>Metric</td><td>Static shortest path</td><td>TN-DCR routing</td></tr><tr><td>Mean total time (s)</td><td>877.56</td><td>733.61</td></tr><tr><td>P95 total time (s)</td><td>2584.43</td><td>2317.45</td></tr><tr><td>Mean queue time (s)</td><td>202.87</td><td>186.31</td></tr><tr><td>Mean transfer time (s)</td><td>674.70</td><td>547.30</td></tr><tr><td>Mean internal wait (s)</td><td>564.75</td><td>437.15</td></tr><tr><td>Throughput (commands/h)</td><td>295.43</td><td>295.68</td></tr></table>

## E. Managerial Insights

The results provide three operational implications for MCS management.

First, the P95 ranking concentrates positive cases within a limited intervention set: $6 0 . 2 \pm 0 . 2 \%$ of the highest-scored 5% of edge-level test records are positive, compared with a 5.9% overall prevalence. This more than tenfold enrichment supports selective operator review, path reconsideration, or priority handling.

Second, the feature blocks and stock-key calibration serve distinct purposes. Removing both graph-aware and dynamiccontext features increases queue-time MAE from $4 9 . 6 \pm 0 . 2 \mathrm { s }$ to $5 4 . 4 { \pm } 0 . 4 \mathrm { s } ,$ a 9.7% degradation, although the coverage metrics are not uniformly monotonic. The stock-key calibration instead reduces Bias magnitude from 17.8±0.6 s to 9.6±0.6 s, or 46.1%, and should therefore be viewed primarily as a biascontrol mechanism.

Third, closed-loop routing reduces mean total time by 16.40%, P95 total time by 10.33%, and internal waiting by 22.59%, with nearly unchanged throughput. The dominant mechanism is avoidance of intermediate resource waiting rather than only pre-dispatch queue reduction. In deployment, support diagnostics should trigger conservative fallback for low-evidence paths, while fab-scale operation remains a separate validation step.

## VI. CONCLUSION

This paper formulated congestion-aware AMHS routing as edge-level prediction followed by candidate-path selection. TN-DCR combines structural path priors, multi-window network and path-local congestion context, and graph-aware endpoint-neighborhood information under a prediction-timesafe construction. Separate edge-level queue- and transfertime LightGBM regressors and an ordinal risk classifier share this representation, while an empirical-Bayes stock-key calibration mitigates persistent queue bias. The edge outputs are aggregated into path-level time and bottleneck-risk estimates for risk-constrained selection. Graph information is encoded through endpoint/neighborhood pooling and a fixed nonlinear projection rather than an end-to-end trained graph encoder.

Experiments on 51,700 edge-level records identify Light-GBM as the most consistent regressor over five seeds, with queue-, transfer-, and reconstructed-total-time MAEs of 49.6± 0.2 s, $2 7 . 4 \pm 0 . 1 \mathrm { s } ,$ , and $6 4 . 9 \pm 0 . 2 \mathrm { s }$ . At P95, $6 0 . 2 \pm 0 . 2 \%$ of the highest-risk 5% of test records are positive, compared with a 5.9% prevalence. Ablations support the joint value of graph-aware and dynamic-context features, while stock-key calibration reduces Bias magnitude by 46.1%. In the controlled closed-loop evaluation, TN-DCR routing reduces mean total time, P95 total time, and internal waiting by 16.40%, 10.33%, and 22.59%, respectively, with nearly unchanged throughput.

The predictors remain observational because training records arise from paths chosen by the incumbent dispatcher; support filtering limits but does not eliminate this selection bias. Moreover, the max-aggregated path risk is a bottleneck index rather than a calibrated probability of any-edge threshold exceedance, and the closed-loop evaluation covers only a limited scale, horizon, and load regime. Future work should investigate off-policy path evaluation, calibrated path-level and denser tail-risk modeling, fab-scale validation under topology and load changes, learned graph aggregation, and end-to-end serving latency.

## REFERENCES

[1] D.-H. Roh, T.-E. Lee, and C. Martinez, “Kanban feedback control for wafer delay regulation of cluster tools,” IEEE Transactions on Automation Science and Engineering, vol. 22, pp. 6822–6838, 2024.

[2] S. Zhang, R. Liu, Y. Chen, M. P. Fanti, and Z. Li, “Modeling and analysis of dual-time petri nets with application to semiconductor manufacturing systems,” IEEE Transactions on Automation Science and Engineering, vol. 22, pp. 23 769–23 783, 2025.

[3] A. Derrow-Pinion, J. She, D. Wong, O. Lange, T. Hester, L. Perez, M. Nunkesser, S. Lee, X. Guo, B. Wiltshire et al., “Eta prediction with graph neural networks in google maps,” in Proceedings of the 30th ACM international conference on information & knowledge management, 2021, pp. 3767–3776.

[4] F. Shi, Y. Meng, J. Liu, and L. Tang, “A combination feature-based reinforcement learning approach via mathematical optimization,” IEEE Transactions on Automation Science and Engineering, vol. 22, pp. 12 455–12 469, 2025.

[5] G. Tresca, H. Salem, G. Cavone, H. Zgaya-Biau, S. Ben-Othman, S. Hammadi, and M. Dotoli, “A matheuristic approach for delivery planning and dynamic vehicle routing in logistics 4.0,” IEEE Transactions on Automation Science and Engineering, vol. 22, pp. 3345–3365, 2024.

[6] S. Carpin, “Solving stochastic orienteering problems with chance constraints using monte carlo tree search,” IEEE Transactions on Automation Science and Engineering, vol. 22, pp. 7855–7869, 2024.

[7] L. Liu, W. Yang, S. Song, and Y. Zhang, “Distributionally robust chance-constrained line planning for railway systems under passenger demand uncertainty,” IEEE Transactions on Automation Science and Engineering, vol. 22, pp. 9457–9472, 2024.

[8] G. Jin, L. Liu, F. Li, and J. Huang, “Spatio-temporal graph neural point process for traffic congestion event prediction,” in Proceedings of the AAAI conference on artificial intelligence, vol. 37, no. 12, 2023, pp. 14 268–14 276.

[9] H.-J. Kim and J.-H. Lee, “Scheduling cluster tools for concurrent processing: Deep reinforcement learning with adaptive search,” IEEE Transactions on Automation Science and Engineering, vol. 22, pp. 3783– 3796, 2024.

[10] J. T. Lin, F.-K. Wang, and C.-K. Wu, “Simulation analysis of the connecting transport amhs in a wafer fab,” IEEE Transactions on Semiconductor Manufacturing, vol. 16, no. 3, pp. 555–564, 2003.

[11] D. Nazzal and L. F. McGinnis, “Expected response times for closedloop multivehicle amhs,” IEEE Transactions on Automation Science and Engineering, vol. 4, no. 4, pp. 533–542, 2007.

[12] D.-Y. Liao and C.-N. Wang, “Neural-network-based delivery time estimates for prioritized 300-mm automatic material handling operations,” IEEE Transactions on Semiconductor Manufacturing, vol. 17, no. 3, pp. 324–332, 2004.

[13] D. Famularo, G. Fortino, F. Pupo, F. Giannini, and G. Franze, “An intelligent multi-layer control architecture for logistics operations of autonomous vehicles in manufacturing systems,” IEEE Transactions on Automation Science and Engineering, vol. 22, pp. 7296–7311, 2024.

[14] G. Tresca, G. Cavone, P. Scarabaggio, R. Carli, and M. Dotoli, “A matheuristics for the configuration of automated vertical lift modules warehouses,” IEEE Transactions on Automation Science and Engineering, vol. 22, pp. 7284–7295, 2024.

[15] L. Wu, X. Li, X. Lu, Z. Feng, and Y. Jing, “An adaptive metareinforcement learning framework for dynamic flexible job shop scheduling,” IEEE Transactions on Automation Science and Engineering, vol. 22, pp. 24 036–24 052, 2025.

[16] X. Wang, Y. Laili, L. Zhang, and Y. Liu, “Hybrid task scheduling in cloud manufacturing with sparse-reward deep reinforcement learning,” IEEE Transactions on Automation Science and Engineering, vol. 22, pp. 1878–1892, 2024.

[17] J. Huang, Z. Huang, X. Fang, S. Feng, X. Chen, J. Liu, H. Yuan, and H. Wang, “Dueta: Traffic congestion propagation pattern modeling via efficient graph learning for eta prediction at baidu maps,” in Proceedings of the 31st ACM international conference on information & knowledge management, 2022, pp. 3172–3181.

[18] W. Hamilton, Z. Ying, and J. Leskovec, “Inductive representation learning on large graphs,” Advances in neural information processing systems, vol. 30, 2017.

[19] Y. Li, R. Yu, C. Shahabi, and Y. Liu, “Diffusion convolutional recurrent neural network: Data-driven traffic forecasting,” arXiv preprint arXiv:1707.01926, 2017.

[20] Z. Wu, S. Pan, G. Long, J. Jiang, and C. Zhang, “Graph wavenet for deep spatial-temporal graph modeling,” arXiv preprint arXiv:1906.00121, 2019.

[21] J. Qiao, Y. Lin, J. Bi, H. Yuan, G. Wang, and M. Zhou, “Attention-based spatiotemporal graph fusion convolution networks for water quality prediction,” IEEE Transactions on Automation Science and Engineering, vol. 22, pp. 1–10, 2024.

[22] T. Chen and C. Guestrin, “Xgboost: A scalable tree boosting system,” in Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining, 2016, pp. 785–794.

[23] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.- Y. Liu, “Lightgbm: A highly efficient gradient boosting decision tree,” Advances in neural information processing systems, vol. 30, 2017.

[24] L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, and A. Gulin, “Catboost: unbiased boosting with categorical features,” Advances in neural information processing systems, vol. 31, 2018.

[25] S. Ivanov and L. Prokhorenkova, “Boost then convolve: Gradient boosting meets graph neural networks,” arXiv preprint arXiv:2101.08543, 2021.

[26] S. Qin, W. Zeng, X. Guo, J. Wang, S. Liu, L. Qi, B. Hu, and J. Wang, “Tackling a resource-sharing hybrid disassembly line balancing problem using reinforcement learning,” IEEE Transactions on Automation Science and Engineering, 2026.

[27] L. Breiman, “Random forests,” Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.

[28] P. Geurts, D. Ernst, and L. Wehenkel, “Extremely randomized trees,” Machine learning, vol. 63, pp. 3–42, 2006.