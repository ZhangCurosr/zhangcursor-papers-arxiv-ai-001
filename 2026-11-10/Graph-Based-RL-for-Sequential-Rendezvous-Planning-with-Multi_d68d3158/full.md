# Graph-Based RL for Sequential Rendezvous Planning with Multiple Patrolling USVs

Seung Woo Han<sup>1,2</sup>, Hyeon Bin Yeo<sup>1,3</sup>, Seung Heon Oh<sup>1,2\*</sup>

<sup>1</sup> Naval Future Innovation Research Group, ROK Navy, Republic of Korea

<sup>2</sup> Seoul National University, Seoul, Republic of Korea

<sup>3</sup> Inha University, Incheon, Republic of Korea

\*Corresponding author: suenghuny@snu.ac.kr

Abstract— In patrol missions involving multiple unmanned surface vehicles (USVs), a designated command USV can support fleet management by sequentially rendezvousing with each patrolling USV. These operations require motion-feasible route planning and priority-aware rendezvous sequencing, with the objective of minimizing mission completion time and total tardiness relative to priority-dependent due dates. We propose a framework combining grid-based path planning, surrogate travel-time prediction, and graph-based reinforcement learning. The path planner accounts for ocean currents and approximates turn-rate constraints through multi-stage maneuvers, with path feasibility empirically validated through trajectory-tracking simulations. For rendezvous sequencing, a Feasible Transition Graph (FTG) represents candidate rendezvous events and temporally feasible transitions, enabling a graph neural network to exploit spatiotemporal rendezvous relationships for policy learning. A surrogate predicts travel times from compact current and motion features to avoid repeated path searches during FTG construction and policy training. Evaluation using grid-based path planning shows statistically significant average objective reductions of 10.5%, 3.5%, and 1.5% compared with greedy, MLP-RL, and 2-step lookahead greedy, respectively. The policy achieves these improvements while requiring approximately 1/35 of the latter’s computation time. It maintains its advantage over MLP-RL under unseen priority-tier settings, with larger gains as fleet size increases.

## I. INTRODUCTION

Unmanned surface vehicles (USVs) are promising platforms for supporting or replacing crewed vessels in repetitive, resource-intensive missions such as surveillance, environmental monitoring, and search operations. Large-scale, long-duration operations require periodic monitoring and information exchange, but offshore communications may face limitations in coverage, bandwidth, and reliability [1]. One practical approach is to deploy a command USV that periodically rendezvouses with individual USVs for status monitoring and shortrange data exchange.

This study considers a maritime surveillance scenario involving a command USV and multiple patrolling USVs. The command USV conducts one rendezvous with each patrolling USV in sequence, while each follows a predefined route. Each patrolling USV has a priority-dependent due date as a soft temporal constraint to encourage earlier visits to higher-priority USVs. The objective is to minimize total tardiness and mission completion time under ocean currents and turn-rate constraints.

This setting creates spatial and temporal dependencies between rendezvous decisions as the targets move along their patrol routes. Ocean currents make travel times dependent on location and direction, while turn-rate constraints expand the path-planning state from $( x , y )$ to $( x , y , \theta )$ by incorporating heading. In particular, the arrival heading affects the travel time and reachability of subsequent rendezvous. These interdependencies complicate rendezvous sequencing and require its integration with detailed path planning.

Previous multi-rendezvous studies used graphs of candidate rendezvous events and feasible transitions to schedule meetings between route-following vehicles and mobile charging vehicles. Solution methods included ILP with a TSP-based heuristic [2] and receding-horizon planning [3]. Related approaches include SMT-based optimization, receding-horizon planning, and dynamic programming [4]-[6]. However, these studies do not consider path planning under marine environmental conditions. In a marine setting, [7] used a current-aware grid to estimate travel energy and temporal transition feasibility, but its position-based representation omitted heading and turning effects. Meanwhile, marine grid-based path planners have considered turn-rate constraints and environmental effects [8], [9], but focus on individual origin-destination pairs without addressing the computational cost of repeated transition evaluations among rendezvous candidates.

Optimization-based rendezvous planning methods [2]-[7] face growing search spaces as the number of targets increases, compounded by repeated path searches to evaluate candidate transitions. Reinforcement learning (RL) has been used to learn policies that rapidly generate visit sequences for static targets, as in the TSP [10], [11]. However, these approaches typically assume static transition costs and feasibility, limiting their application to dynamic rendezvous problems, where transition feasibility and costs vary with position and time.

This study couples current-aware path planning with graphbased RL for priority-aware sequential rendezvous. A gridbased planner approximates continuous turn-rate constraints through multi-stage maneuvers, with allowable turning angles calibrated through trajectory-tracking simulations. To reduce transition evaluation costs, a surrogate-based feasible transition graph (sFTG) represents spatiotemporal relationships using predicted travel times and nominal headings. A GCN encodes the sFTG to learn a rendezvous policy, while surrogate predictions replace repeated path searches during training. The resulting policy is evaluated with grid-based path planning that incorporates ocean currents and turn-rate constraints. To the best of our knowledge, this is the first study to jointly consider these factors in sequential multi-rendezvous planning.

Experiments show statistically significant average objective reductions of 10.5%, 3.5%, and 1.5% compared with greedy, MLP-RL, and 2-step lookahead greedy, respectively.

The proposed method also runs approximately 35 times faster than the computationally intensive lookahead heuristic. Its advantage persists across priority-tier settings and widens over MLP-RL as fleet size increases, supporting the benefits of exploiting graph structure. The main contributions are as follows:

1. A current-aware grid-based path planner with a multi-stage approximation of turn-rate constraints, empirically calibrated through trajectory-tracking simulations.

2. A graph-based RL framework that exploits future rendezvous connectivity, using surrogate predictions to avoid repeated path searches during FTG construction and training.

3. Empirical evidence that the benefit of graph encoding increases with patrol fleet size and persists under grid-based path-planning evaluation.

## II.PROBLEM DEFINITION

This section formulates the problem in two stages: current-aware path planning with turn constraints and sequential rendezvous planning.

## A. Current-Aware Path Planning with Turn Constraints

The minimum-time path-planning problem for a USV under ocean currents and turn-rate constraints is formulated in $( 1 ) ‐ ( 8 )$ . Let the initial position be $\pmb { X _ { s } } = ( x _ { s } , y _ { s } )$ , the goal position be $\pmb { X } _ { f } = \left( x _ { f } , y _ { f } \right)$ , the initial heading be $\theta _ { s } ,$ , the USV speed be �, the current field be $\pmb { c } ( x , y )$ , and the maximum heading rate be $\omega _ { \mathrm { m a x } } .$ . Headings and bearings are measured counterclockwise from the positive �-axis.

$$
T _ { Q } ( X _ { s } , \theta _ { s } , X _ { f } ) : = \operatorname* { m i n } _ { \zeta ( \cdot ) , T } T\tag{1}
$$

$$
\mathrm { s . t . } \quad \dot { x } ( t ) = v c o s \theta ( t ) + c _ { x } ( x ( t ) , y ( t ) ) , \ t \in [ 0 , T ]\tag{2}
$$

$$
{ \dot { y } } ( t ) = v s i n \theta ( t ) + c _ { y } ( x ( t ) , y ( t ) ) , \ t \in [ 0 , T ]\tag{3}
$$

$$
{ \dot { \theta } } ( t ) = w ( t ) , \ t \in [ 0 , T ]\tag{4}
$$

$$
\zeta ( t ) = \big ( x ( t ) , y ( t ) , \theta ( t ) \big ) , t \in [ 0 , T ]\tag{5}
$$

$$
| w ( t ) | \leq \omega _ { m a x } , \ t \in [ 0 , T ]\tag{6}
$$

$$
( x ( 0 ) , y ( 0 ) , \theta ( 0 ) ) = ( x _ { s } , y _ { s } , \theta _ { s } )\tag{7}
$$

$$
( x ( T ) , y ( T ) ) = ( x _ { f } , y _ { f } )\tag{8}
$$

Here, �(�) denotes the vessel state trajectory, and $T _ { Q } \big ( X _ { s } , \theta _ { s } , X _ { f } \big )$ denotes the minimum travel time. Equations $( 2 ) \AA { - } ( 8 )$ define the current-affected vessel dynamics, turning constraint, and boundary conditions.

## B. Sequential Rendezvous Planning

The rendezvous planning problem determines the visit sequence of a command USV to � patrolling USVs with hierarchical priorities. The patrol route of USV � is defined as $P _ { i } = \{ p _ { i , j } \in \mathbb { R } ^ { 2 } \mid j = 1 , \ldots , m \}$ , where $t _ { i , j } ^ { n }$ denotes the time at which USV � reaches waypoint $p _ { i , j }$ for the � -th time, and $d _ { i }$ denotes its due date. Let � denote the decision step and $T ( \tau )$ the completion time of the �-th rendezvous. The travel time of the command USV is computed using the minimumtravel-time function $T _ { Q }$ defined above. Let $\pi = ( \pi _ { 1 } , \dots , \pi _ { N } ) \in$ $S _ { N }$ denote a permutation representing the visit order of all patrolling USVs. � returns the arrival-heading associated with the path defined above. The rendezvous planning problem is then formulated in (9)-(15).

$$
\begin{array} { r } { \underset { \pi \in S _ { N } } { m i n } \big [ T ( N ) + \sum _ { \tau = 1 } ^ { N } m a x ( 0 , T ( \tau ) - d _ { \pi _ { \tau } } ) \big ] } \end{array}\tag{9}
$$

$$
T ( 0 ) = 0 , \ X _ { c } ( 0 ) = X _ { 0 } , \ \theta _ { c } ( 0 ) = \theta _ { 0 }\tag{10}
$$

$$
\mathrm { f o r } \mathrm { e a c h } \tau = 1 , \dots , N , \mathrm { w h e r e } i ( \tau ) = \pi _ { \tau } ,
$$

$$
a _ { j } ^ { \tau } = T ( \tau - 1 ) + T _ { Q } ( X _ { C } ( \tau - 1 ) , \theta _ { c } ( \tau - 1 ) , p _ { i ( \tau ) , j } )\tag{11}
$$

$$
n _ { j } ^ { \tau } = m i n \{ n : ~ t _ { i ( \tau ) , j } ^ { ( n ) } \geq a _ { j } ^ { \tau } \}\tag{12}
$$

$$
j ^ { * } ( \tau ) = a r g m i n ( t _ { i ( \tau ) , j } ^ { ( n _ { j } ^ { \tau } ) } - T ( \tau - 1 ) )\tag{13}
$$

$$
T ( \tau ) = t _ { i ( \tau ) , j ^ { * } ( \tau ) } ^ { ( n _ { j ^ { * } ( \tau ) } ^ { \tau } ) } , \ X _ { c } ( \tau ) = p _ { i ( \tau ) , j ^ { * } ( \tau ) }
$$

$$
\theta _ { c } ( \tau ) = Q ( X _ { c } ( \tau - 1 ) , \theta _ { c } ( \tau - 1 ) , p _ { i ( \tau ) , j ^ { * } ( \tau ) } )\tag{14}
$$

(15)

Equation (9) minimizes the final rendezvous completion time and total tardiness across all USVs. In (11), �<sup>�</sup> denotes the earliest arrival time of the command USV at $p _ { i ( \tau ) , j }$ at decision step �. The feasible rendezvous time at $p _ { i ( \tau ) , j }$ is determined as the earliest time at which both USVs can be present at the location, allowing the command USV to wait if it arrives before the patrolling USV. Equation (13) selects the earliest feasible rendezvous point �<sup>∗</sup> along the route of the selected USV. Equations (14) and (15) then update the rendezvous time, position, and command-USV heading. The decision variable is the visit sequence �, while each rendezvous point is selected to achieve the earliest feasible meeting with the chosen USV.

## III. MODELING

## A. Current-Aware Path Planning with Turn Constraints

This section presents the proposed grid-based marine pathplanning framework, emphasizing computational tractability for repeated searches in multi-rendezvous planning (Fig. 1).

$$
I ) ~ G r i d G r a p h C o n s t r u c t i o n
$$

The operating area is discretized into a grid, and the USV is assumed to move along straight-line segments between neighboring grid points. As shown in Fig. 1(a), the neighborhood is constructed by adding candidate edges only when they do not intersect existing edges except at the current node. To incorporate turn-rate constraints, each grid point is represented by nodes $v = ( \boldsymbol { X } , \boldsymbol { \theta } )$ , where � is the position and � the arrival heading. The arrival heading depends on the incoming edge, so each grid point has one node per neighboring direction.

For a transition from $X _ { s } { \mathrm { t o } X _ { f } }$ , the required arrival heading $\theta _ { X _ { s } X _ { f } }$ is calculated using (16), where $\bar { c } _ { \bot }$ is the component of the average current between $X _ { s }$ and $X _ { f }$ perpendicular to the transition direction, positive leftward. If ∣ $\bar { c } _ { \perp } \mid \geq v$ , the transition is excluded because the required heading is infeasible.

$$
\begin{array} { r } { \theta _ { X _ { s } , X _ { f } } = \angle ( X _ { f } - X _ { s } ) + a r c s i n ( - \frac { \bar { c } _ { \bot } } { v } ) } \end{array}\tag{16}
$$

A directed edge is created only if the angular difference between the current heading $\theta _ { s }$ and the required heading $\theta _ { X _ { s } , X _ { f } }$ does not exceed the maximum allowable angle $\theta _ { \mathrm { m a x } } .$

![](images/ca1f73c640ecf56d754836faee2741a39232d59bbc867afd1679f155394a3e00.jpg)  
Fig. 1. Grid-based path planning framework under ocean currents and turn-rate constraints

Let $N ( { \pmb X } _ { s } )$ denote the neighborhood of $X _ { s }$ , and $d _ { a n g }$ the minimum absolute angular difference between two headings. The feasible neighborhood under this constraint is defined in (17). The edge cost represents the travel time between neighboring grid points under ocean currents, as illustrated in Fig. 1(b), and is calculated using (18), where $\bar { c } _ { \parallel }$ is the average current component along the transition direction.

$$
N _ { \theta _ { m a x } } ( X _ { s } , \theta _ { s } ) = \{ X _ { f } \in N ( X _ { s } ) \mid d _ { a n g } ( \theta _ { s } , \theta _ { X _ { s } , X _ { f } } ) \leq \theta _ { m a x } \}\tag{17}
$$

$$
\begin{array} { r } { L _ { X _ { s } X _ { f } } = \frac { | | X _ { f } - X _ { s } | | } { \overline { { c _ { | | } } } + \sqrt { v ^ { 2 } - \bar { c } _ { \perp } ^ { 2 } } } } \end{array}\tag{18}
$$

This graph formulation captures both current-dependent travel time and heading-dependent transition feasibility. The grid path is then obtained using standard graph search algorithms. The estimation of $\theta _ { \mathrm { { m a x } } }$ based on the continuous turnrate constraint is detailed in the following subsection.

## 2) Multi-stage maneuver approximation

This section introduces a multi-stage maneuver approximation that maps the accumulated displacement from successive small turns to an equivalent straight-line motion along a representative heading, as shown in Fig. 1(c). This approximates continuous turning motion at the grid-transition level. Assuming that heading changes at each small-time step Δ�, the turn-rate constraint is expressed in (19).

$$
d _ { a n g } ( \theta _ { k + 1 } , \theta _ { k } ) \leq \omega _ { m a x } \Delta t\tag{19}
$$

For a maneuver consisting of � time intervals, let �denote the mean heading and define the heading deviation at step � as $\delta _ { k } = \theta _ { k } - { \bar { \theta } }$ . Assuming that the headings remain close to $\bar { \theta }$ and that $\delta _ { k }$ is sufficiently small, a second-order Taylor expansion yields (20).

$$
\begin{array} { r } { \sum _ { k = 1 } ^ { N } \left[ \begin{array} { l } { c o s \theta _ { k } } \\ { s i n \theta _ { k } } \end{array} \right] \approx N \left( 1 - \frac { 1 } { 2 } V a r ( \delta ) \right) \left[ \begin{array} { l } { c o s \bar { \theta } } \\ { s i n \bar { \theta } } \end{array} \right] } \end{array}\tag{20}
$$

Equation (20) shows that the accumulated displacement of the multi-stage maneuver is approximately aligned with the mean heading ${ \bf \Xi } _ { \breve { \theta } , \bullet } ^ { \cup }$ while the displacement along that direction decreases as the heading variation increases. Thus, $\bar { \theta }$ can be interpreted as the representative direction of the accumulated motion. This result provides a mathematical basis for the above approximation.

In the absence of ocean currents, the heading required for a given grid transition can be determined geometrically. Under ocean currents, however, determining the allowable head ing change becomes more difficult because the accumulated displacement includes current-induced motion at each maneuver stage, as described in (21). Accordingly, $\theta _ { \mathrm { m a x } }$ is empirically calibrated to account for the reachability of multi-stage maneuvers under ocean-current conditions.

$$
\begin{array} { r } { \Delta X ^ { ( n ) } = v \Delta t \sum _ { k = 1 } ^ { N } \left[ { c o s \theta _ { k } } \right] + \Delta t \sum _ { k = 1 } ^ { N } c ( x _ { k } , y _ { k } ) } \end{array}\tag{21}
$$

## 3) Calibration of the Maximum Feasible Turning Angle

A grid search is conducted over candidate turning angles. For each candidate, trajectory-tracking simulations are performed under varying initial positions, headings, and oceancurrent conditions. Path tracking employs a reference strategy that probabilistically selects between a Path-Proximity Strategy, which guides the USV toward the closest point on the reference path, and a Heading-Alignment Strategy, which aligns the USV with the direction of the nearest path segment $( { \mathrm { F i g } } .$ 1(d)). Repeated simulations are used to estimate the tracking success rate $p _ { \mathrm { s u c c } } ( \theta )$ for each candidate angle, from which its two-sided 95% Wilson confidence lower bound $p _ { W } ^ { 0 . 9 5 } ( \theta )$ is computed. Using a conservative acceptance threshold of $\eta = 0 . 9 9$ , the largest candidate angle whose confidence lower bound meets the threshold is selected as the maximum empirically feasible turning angle in (22).

$$
\theta _ { m a x } = m a x \{ \theta \in \theta \mid p _ { W } ^ { 0 . 9 5 } ( \theta ) \geq \eta \}\tag{22}
$$

This procedure is intended as an empirical calibration of $\theta _ { \mathrm { m a x } }$ , rather than a new trajectory-tracking control method.

![](images/7ab49e72c189d696259307c4c3011b4581364e593f90087c61146b3d954a5d6a.jpg)  
(a) Rendezvous Scenario

![](images/18438f5e35de19b28098085c91e4267d11aa0e68592f0885d94f232bc5732872.jpg)  
(b) Feasible Transition Graph

![](images/f232bd5a7a672659a9c11e813196349f1145560d6d12f2579dda4506d00ffcb7.jpg)  
(c) Graph Convolution

![](images/d885b89a573719b88230ab1302c3141a131edbbdad70470571348ed05a852fe8.jpg)  
(d) Action Representation  
Fig. 2. Graph-Based RL framework for sequential rendezvous planning

The focus is on estimating a grid-level turning constraint from simulation-based tracking performance under continuous turn rate constraints and ocean currents.

## 4) Surrogate-based travel-time estimation

Although the grid-based path planner described above enables efficient graph search, repeatedly invoking it for each candidate rendezvous transition during RL training remains computationally expensive. To alleviate this computational burden, a low-cost surrogate model is introduced to predict USV travel time from low-dimensional features describing the origin–destination geometry and surrounding currents. Current statistics are extracted from inner and outer corridors defined with respect to the straight line connecting the origin and destination. The reference line is used only for feature extraction and does not represent the actual travel path. The resulting feature vector, consisting of current statistics, origin– destination distance, and initial heading information, is defined in (23).

$$
z _ { o d } = \left[ \mu _ { \bot } ^ { i n } , \left( \sigma _ { \bot } ^ { i n } \right) ^ { 2 } , \mu _ { \parallel } ^ { i n } , \left( \sigma _ { \parallel } ^ { i n } \right) ^ { 2 } , \Delta \theta _ { o d } , d _ { o d } , \varDelta \mu _ { \parallel } ^ { o u t - i n } , \varDelta \mu _ { \bot } ^ { o u t - i n } \right]\tag{23}
$$

Here, $\mu _ { \perp } ^ { i n }$ and $\mu _ { \parallel } ^ { i n }$ denote the mean perpendicular and parallel current components in the inner corridor, with corresponding variances $\left( { \sigma _ { \bot } ^ { i n } } \right) ^ { 2 } \mathrm { a n d } \left( { \sigma _ { \parallel } ^ { i n } } \right) ^ { 2 } . \Delta \theta _ { o d }$ and $d _ { o d }$ denote the initial heading–bearing difference and origin–destination distance, respectively. The features $\Delta \mu _ { \perp } ^ { o u t - i n }$ and $\Delta \mu _ { \parallel } ^ { o u t - i n }$ represent the differences in the mean current components between the outer and inner corridors, capturing spatial variations in the surrounding current field. The corridor widths are treated as hyperparameters. Furthermore, two prediction targets are considered: the absolute travel time and its residual with respect to the Euclidean reference time $d _ { o d } / v$

## B. Graph-based RL for Rendezvous Planning

This section presents the RL framework for sequential rendezvous planning. Candidate rendezvous events across multiple patrol routes and their feasible transitions are represented by a Feasible Transition Graph (FTG), which encodes the spatiotemporal connectivity among potential future rendezvous events. The FTG serves as the state representation, and its graph structure is encoded by a graph convolutional network (GCN) [12] to learn an RL policy for rendezvous sequencing. The overall framework is shown in Fig. 2, with its components described using a Markov Decision Process.

## 1) State representation

A node in the FTG represents a waypoint on a patrol route together with the corresponding arrival time. Following the notation in Section II-B, a node is denoted by $v _ { i , j } ^ { ( n ) } =$ $\big ( p _ { i , j } , t _ { i , j } ^ { ( n ) } \big )$ . Its feature vector consists of the waypoint position, the remaining time until the patrolling USV reaches the waypoint, and the remaining time until its due date, as defined in (24). Here, $t _ { c }$ denotes the current simulation time.

$$
\pmb { x } _ { i , j } ^ { ( n ) } = [ x _ { i , j } , y _ { i , j } , t _ { i , j } ^ { ( n ) } - t _ { c } , d _ { i } - t _ { c } ]\tag{24}
$$

Repeated patrol cycles can generate infinitely many nodes at the same locations, so a computational horizon of the next three arrivals is used to bound the graph size. The FTG includes only unvisited patrolling USVs and is updated over time in both node features and graph structure, forming a time-varying graph. Each FTG node is then connected to the earliest reachable rendezvous candidate of every other patrol ling USV. These connections represent the feasible next rendezvous options, as illustrated in Fig. 2(b). The successor node and edge set are defined in (25) and (26), respectively. For each FTG transition, the bearing from departure to destination, $\theta ^ { \prime }$ , is used as the nominal departure heading. For brevity, $T _ { Q } ( p _ { i , j } , p _ { k , l } )$ denotes $T _ { Q } \big ( p _ { i , j } , \theta ^ { \prime } , p _ { k , l } \big )$

$$
v _ { k } ^ { * } ( v _ { i , j } ^ { ( n ) } ) = a r g m i n \{ t _ { k , l } ^ { ( m ) } | T _ { Q } ( p _ { i , j } , p _ { k , l } ) \le t _ { k , l } ^ { ( m ) } - t _ { i , j } ^ { ( n ) } \}\tag{25}
$$

$$
\pmb { \varepsilon } = \{ ( v _ { i , j } ^ { ( n ) } , v _ { k } ^ { \ast } ( v _ { i , j } ^ { ( n ) } ) ) | k \neq i \}\tag{26}
$$

Conceptually, the FTG is constructed using the travel time and arrival heading obtained from detailed path planning. In practice, however, the estimated travel time from the surrogate model and a nominal heading approximation are used for efficient state generation. Accordingly, the state representation in this study employs a surrogate-based FTG, hereafter denoted as sFTG.

## 2) Graph Convolution

The sFTG is encoded by a GCN that incorporates information from reachable future rendezvous candidates. As illustrated in Fig. 2(c), directed edges represent temporal precedence, allowing each node to aggregate information from its feasible successor nodes. For a node $v = v _ { i , j } ^ { ( n ) }$ , �(�) denotes the set of reachable successor nodes. For $\mathcal { N } ( v ) = \emptyset , m _ { v }$ is set to 0. The GCN layer is defined in (27)–(28), where $h _ { v }$ and $h _ { v } ^ { \prime }$ denote the node embeddings before and after aggregation, respectively. The initial embedding $h _ { v } ^ { ( 0 ) }$ is obtained from the node features $\pmb { x } _ { i , j } ^ { ( n ) }$ , where ∥ denotes vector concatenation.

$$
\begin{array} { r } { m _ { v } = \frac { 1 } { | \mathcal { N } ( v ) | } { \sum } _ { u \in N ( v ) } E L U ( W _ { n } h _ { u } ) } \end{array}\tag{27}
$$

$$
h ^ { \prime } { } _ { v } = E L U ( W _ { o } [ E L U ( W _ { s } h _ { v } ) \parallel m _ { v } ] )\tag{28}
$$

Repeated GCN layers progressively incorporate information from multi-hop future rendezvous candidates, yielding the final representation $h _ { v } ^ { ( L ) }$ after � layers. The GCN uniformly aggregates successor-node information, with rendezvoustime and due-date features enabling the representation to encode future rendezvous urgency.

## 3) Action

At each decision step, the action selects one of the unvisited patrolling USVs for rendezvous, with its earliest reachable rendezvous node serving as the corresponding candidate. The policy distribution is computed using an actor query that incorporates the current state of the command USV and information about each candidate rendezvous, as shown in Fig. 2(d). The query is combined with the graph representation obtained from the preceding graph network to compute an action score for each candidate. The action scores are computed as defined in (29)–(30) and converted into a policy distribution over all feasible candidates using a softmax function with temperature $T = 1 . 5$ . Here, $x _ { c }$ and $y _ { c }$ denote the current position of the command $\mathrm { U S V } , \theta _ { c }$ denotes its current heading, and $\theta _ { u } ^ { a r r }$ and $\Delta t _ { u }$ denote the arrival heading and remaining time until rendezvous for candidate �, respectively.

$$
q _ { u } = [ x _ { c } , y _ { c } , \Delta t _ { u } , s i n \ \theta _ { c } , c o s \ \theta _ { c } , s i n \ \theta _ { u } ^ { a r r } , c o s \ \theta _ { u } ^ { a r r } ]\tag{29}
$$

$$
\begin{array} { r } { a _ { u } = \frac { 1 } { T } M L P _ { v a l u e } ^ { a c t o r } ( M L P _ { q } ^ { a c t o r } ( q _ { u } ) | | h _ { u } ^ { ( L ) } ) } \end{array}\tag{30}
$$

## 4) Reward

The reward at each decision step is defined as the negative sum of the elapsed time until rendezvous completion and the selected $\mathrm { U S V } ^ { \prime } \mathbf { s }$ tardiness at the rendezvous time. Because the next decision step begins immediately after each rendezvous, the elapsed-time terms accumulate to the total mission completion time, while the tardiness terms accumulate to the total tardiness. Thus, maximizing the cumulative reward is equivalent to minimizing the objective function.

## 5) Critic Network and Policy Optimization

Proximal Policy Optimization (PPO) [13] is employed to stabilize policy updates. The critic uses the same inputs as the actor but estimates the state value with separate parameters. For each candidate node $u \in V _ { \mathrm { a l i v e } } ,$ , it computes a gate $e _ { u }$ and value vector $v _ { u } ,$ whose weighted sum forms a global context vector for estimating $V ( s )$ , as defined in (31)–(33).

$$
e _ { u } = \sigma ( M L P _ { g a t e } ^ { c r i t i c } ( M L P _ { q } ^ { c r i t i c } ( q _ { u } ) \vert \vert ~ h _ { u } ^ { ( L ) } ) )\tag{31}
$$

$$
v _ { u } = M L P _ { v a l u e } ^ { c r i t i c } ( M L P _ { q } ^ { c r i t i c } ( q _ { u } ) \vert \vert \ h _ { u } ^ { ( L ) } )\tag{32}
$$

$$
\begin{array} { r } { V ( s ) = M L P _ { r e a d o u t } ^ { c r i t i c } ( \sum _ { u \in V _ { a l i v e } } e _ { u } v _ { u } ) } \end{array}\tag{33}
$$

## IV. EXPERIMENTS AND RESULTS

All experiments used the same ocean-current generation procedure and USV parameters. A time-invariant current field was generated over the 3600 m × 3600 m operating area by superimposing Lamb–Oseen vortices at large, medium, and small scales to produce smooth spatial variations. The resulting ocean-current velocity at position $\pmb { p }$ is defined in (34). Here, $g \in \{ L , M , S \}$ denotes the vortex scale, and $N _ { g }$ is the number of vortices of scale $g .$ . The vortex induced tangential velocity is given by (35).

$$
\begin{array} { r } { u _ { c } ( \pmb { p } ) = \sum _ { g \in \{ L , M , S \} } \sum _ { i = 1 } ^ { N _ { g } } u _ { g } ^ { ( i ) } \left( \pmb { p } \right) } \end{array}\tag{34}
$$

$$
\begin{array} { r } { v _ { \theta , g } ( r ) = \frac { { r } _ { g } } { 2 \pi r } \bigg [ 1 - e x p \left( - \frac { r ^ { 2 } } { a _ { g } ^ { 2 } } \right) \bigg ] , { r } _ { g } = 2 \pi R _ { g } V _ { g } } \end{array}\tag{35}
$$

Here, $a _ { g }$ is the core radius, while $R _ { g }$ and $V _ { g }$ are the reference radius and velocity used to determine the circulation strength $\Gamma _ { g }$ , respectively. The parameter $d _ { \mathrm { m i n } }$ specifies the minimum separation between vortices. The parameters used for each vortex scale are summarized in TABLE I.

TABLE I. Parameters for multiscale ocean-current generation
<table><tr><td>Vortex</td><td> $N _ { g }$ </td><td> $V _ { g } ( \mathrm { m } / \mathrm { s } )$ </td><td> $R _ { g } ( \mathbf { m } )$ </td><td> $a _ { g } ( \mathbf { m } )$ </td><td> $\underline { { d _ { m i n } ( \mathbf { m } ) } }$ </td></tr><tr><td> $u _ { L }$ </td><td>2-3</td><td>0.5</td><td>1800</td><td>900</td><td>900</td></tr><tr><td> $u _ { M }$ </td><td>3</td><td>0.4</td><td>900</td><td>550</td><td>450</td></tr><tr><td> $u _ { S }$ </td><td>6</td><td>0.4</td><td>250</td><td>350</td><td>250</td></tr></table>

The vortex parameters were selected to generate current fields with distinct spatial scales and moderate magnitudes relative to the USV cruising speed. The number of large-scale vortices was sampled uniformly, and the vortices were arranged in symmetric pairs or triangular configurations, with the entire pattern randomly rotated. Rotational directions were balanced to avoid directional bias. Across 1,000 generated current fields, the mean current speed was 0.71 m/s, with a variance of $0 . 1 5 ( \mathrm { m } / s ) ^ { 2 }$ . The USV cruised at 2.5 m/s with a maximum turn rate of $2 . 0 ^ { \circ } / s ,$ giving a turning radius of about 72 m in still water. All experiments were conducted on a workstation equipped with an Intel Core i7 CPU, 64 GB RAM, and an NVIDIA RTX 5070 GPU.

## A. Current-Aware Path Planning with Turn Constraints

## 1) Grid-Based Path Planning

The maximum turning angle $\theta _ { \mathrm { m a x } }$ was evaluated using a lightweight stochastic tracking procedure to validate the continuous realizability of grid-based paths. For 2,000 Dijkstra paths generated under random current fields, start-goal pairs, and headings, using $R = 6 0 m$ and a 24-direction neighborhood, each path underwent up to 10 stochastic tracking trials to account for controller stochasticity and was regarded as continuously realizable if at least one trial satisfied the $5$ m position and $5 ^ { \circ }$ heading tolerances. TABLE II reports the feasibility rates for each candidate $\theta _ { \mathrm { m a x } }$ . Here, $P _ { p }$ and $P _ { h }$ denote the probabilities of the Path-Proximity and Heading-Alignment strategies, respectively. The setting $\left[ P _ { p } , P _ { h } \right] =$ [0.67, 0.33] allowed the largest accepted turning angle. Using a lower bound of at least 99% from the 95% Wilson confidence interval as the criterion, $\theta _ { \mathrm { { m a x } } } = 6 0 ^ { \circ }$ was selected, achieving a feasibility rate of 99.45%.

TABLE II. Tracking success rates across policy settings
<table><tr><td> $[ P _ { p } / P _ { h } ]$ </td><td> $5 0 ^ { \circ }$ </td><td> $5 5 ^ { \circ }$ </td><td> $6 0 ^ { \circ }$ </td><td> $6 5 ^ { \circ }$ </td><td> $7 0 ^ { \circ }$ </td><td> $\theta _ { m a x }$ </td></tr><tr><td>[0.80/0.20]</td><td>99.65</td><td>99.45</td><td>99.35</td><td>99.30</td><td>90.05</td><td>55°</td></tr><tr><td>[0.75/0.25]</td><td>99.60</td><td>99.45</td><td>99.40</td><td>99.20</td><td>99.0</td><td>55°</td></tr><tr><td>[0.67/0.33]</td><td>99.60</td><td>99.50</td><td>99.45</td><td>99.25</td><td>99.00</td><td> $6 0 ^ { \circ }$ </td></tr><tr><td>[0.50/0.50]</td><td>99.30</td><td>99.15</td><td>98.80</td><td>98.60</td><td>98.55</td><td></td></tr></table>

For $\theta _ { \mathrm { { m a x } } } = 6 0 ^ { \circ }$ , cumulative feasibility rates were $9 6 . 4 \% .$ 98.85%, and 99.10% within 1, 3, and 5 trials, respectively. Each trial required only 0.38 s on average, making repeated tracking practical. Among successful trials, the minimum successful travel time was used as the continuous-tracking reference. Relative to this reference, the mean absolute percentage error (MAPE) of the grid-based travel times was 1.42%, with a 95th percentile of 5.12%, supporting the validity of the gridbased travel-time approximation.

TABLE III evaluates the effect of grid spacing � under the same experimental conditions. Here, $d _ { \operatorname* { m a x } } ^ { G }$ and $\rho$ denote the maximum distance to the nearest grid point and grid-point density in $p o i n t s / k m ^ { 2 }$ , respectively, while �̅ denotes the average computation time. As � increased, $d _ { \operatorname* { m a x } } ^ { G }$ increased and $\rho$ decreased, indicating lower spatial resolution. Meanwhile, the larger feasible $\theta _ { \mathrm { m a x } }$ helped maintain comparable path performance while computation time decreased. Accordingly, $R = 6 0 \mathrm { m }$ was selected for the multi-rendezvous experiments to balance spatial resolution and computational efficiency.

TABLE III. Grid spacing effects on path quality and spatial resolution
<table><tr><td>R(m)</td><td> $\theta _ { m a x } ( ^ { \circ } )$ </td><td> $T _ { Q } ( \mathrm { s } )$ </td><td> $d _ { \mathrm { m a x } } ^ { G } ( \mathrm { m } )$ </td><td> $\rho$ </td><td>τ(s)</td></tr><tr><td>45</td><td>50</td><td>829.7</td><td>26.0</td><td>570.2</td><td>8.47</td></tr><tr><td>60</td><td>60</td><td>832.3</td><td>34.6</td><td>320.8</td><td>2.49</td></tr><tr><td>90</td><td>80</td><td>831.7</td><td>52.0</td><td>142.6</td><td>1.69</td></tr></table>

The neighborhood size and grid geometry were also examined. The 12-direction setting had a smaller feasible $\theta _ { m a x }$ due to shorter transitions and occasionally caused path-search failures, prompting adoption of the 24-direction setting. Triangular and square grids showed comparable performance, with minor trade-offs in spatial resolution and computation time. The triangular grid was used in subsequent experiments.

## 2) Surrogate Modeling and Path Search

Three regression models were considered: ridge linear regression (LR), extreme gradient boosting (XGB), and random forest (RF). The models were trained on 4,000 random grid paths and evaluated on 1,000 independent paths, with corridor widths set to $3 R = 1 8 0 \mathrm { m }$ . TABLE IV compares the absolute (Abs) and residual (Res) prediction errors, using Euclidean travel time as a reference and Dijkstra travel time as ground truth. Residual prediction generally reduced error, and XGB-Res achieved the lowest errors for both metrics and was therefore selected.

TABLE IV. Surrogate model comparison
<table><tr><td rowspan="2">Metric</td><td colspan="2">LR</td><td colspan="2">XGB</td><td colspan="2">RF</td><td rowspan="2">Ref.</td></tr><tr><td> $\mathbf { A } \mathbf { b } \mathbf { s }$ </td><td>Res</td><td>Abs</td><td>Res</td><td>Abs</td><td>Res</td></tr><tr><td>MAPE(%) 10.84 RMSE(s)</td><td>83.30</td><td>8.39 86.52</td><td>3.94 49.84</td><td>3.37 45.46</td><td>4.83 58.10</td><td>3.89 51.87</td><td>18.09 197.86</td></tr></table>

TABLE V compares direct surrogate prediction, $\mathbf { A } ^ { * }$ with Euclidean and surrogate-based heuristics, and Dijkstra in terms of path cost and mean computation time. The mean absolute gap is computed relative to the Dijkstra path cost. Direct prediction is the fastest but does not generate an actual path. Although the heuristics do not guarantee optimality, $\mathbf { A } ^ { * }$ with the Euclidean heuristic achieved only a small path-cost gap while reducing computation time by approximately thirteenfold compared with Dijkstra. Accordingly, $\mathbf { A } ^ { * }$ with the Euclidean heuristic was adopted for subsequent grid-based path searches in rendezvous evaluation.

TABLE V. Comparison of path-planning methods
<table><tr><td>Method</td><td>Direct Prediction</td><td> $\mathbf { A } ^ { * }$ </td><td>Euclidean Surrogate- guided  $\mathbf { A } ^ { * }$ </td><td>Dijkstra</td></tr><tr><td>Run time (s)</td><td>0.001</td><td>0.195</td><td>1.652</td><td>2.490</td></tr><tr><td>Abs. gap (%)</td><td>3.023</td><td>2.113</td><td>0.883</td><td>0.0</td></tr><tr><td>Path output</td><td>No</td><td>Yes</td><td> $\mathrm { Y e s }$ </td><td>Yes</td></tr></table>

## B. Graph-Based RL for Rendezvous Planning

For the multi-rendezvous problem, two patrol-route types were considered for each fleet size, with the initial position, patrol direction, and priority of each USV randomly assigned. Type A comprises non-overlapping rectangular routes, whereas Type B comprises overlapping routes: four horizontal rectangles, four vertical rectangles, and four squares for 12 USVs, with four additional squares for 16 USVs. For 24-USV instances, both types used 12 routes, with two USVs assigned to each route at opposite starting positions. Each route was formed by sequentially connecting 18 major waypoints placed at fixed offsets from the area boundaries. A common due-date reference interval was obtained by dividing the mean rendezvous completion time of priority-agnostic greedy solutions over randomly sampled instances by the number of priority tiers. This interval was applied to all instances of that type to assign priority-dependent due dates.

Training used 12 patrolling USVs, three priority levels, and a mixture of Types A and B. The learning rate, discount factor γ, GAE parameter λ, clipping ratio ε, and value-loss coefficient were set to $2 . 5 \times 1 0 ^ { - 4 } , \dot { 1 } . 0 \dot { , } 0 . 9 5 , 0 . 2 .$ , and 0.5, respectively, with no entropy bonus. Each of 500 iterations generated six instances with eight rollouts each, yielding 48 trajectories trained for two epochs. During training, the surrogate model determined transition times and reachability, while the arrival heading was approximated by the bearing to the rendezvous point. During evaluation, the sFTG encoded the state. For each unvisited USV, candidate rendezvous locations were identified and ordered using surrogate estimates and checked sequentially by grid-based path planning until a feasible location was found. Grid-based path planning used the current heading, and the resulting arrival heading was propagated to the next decision. TABLE VI shows the network architecture.

TABLE VI. Graph-RL network architecture
<table><tr><td>Component</td><td>Architecture</td><td>Activation</td></tr><tr><td>Node embedding</td><td>Linear (4,128)</td><td>一</td></tr><tr><td>Graph encoder</td><td>3 layers  $( d _ { h } = 1 2 8 )$ </td><td>ELU</td></tr><tr><td> $M L P _ { q } ^ { a c t o r / c r i t i c }$ </td><td>MLP(7, 64, 128)</td><td>ELU</td></tr><tr><td> $M L P _ { v a l u e } ^ { a c t o r }$ </td><td>MLP(256, 128, 64, 1)</td><td>ELU</td></tr><tr><td> $M L P _ { g a t e } ^ { c r i t i c }$ </td><td>MLP(256, 128, 64, 1)</td><td>ELU, sigmoid</td></tr><tr><td> $M L P _ { v a l u e } ^ { c r i t i c }$ </td><td>MLP(256,128,128)</td><td>ELU</td></tr><tr><td> $M L P _ { r e a d o u t } ^ { c r i t i c }$ </td><td>MLP(128,64,1)</td><td>ELU</td></tr></table>

For direct comparison, MLP-RL removes the graph encoder from the proposed Graph-RL and independently scores candidates using query and node features. Greedy-IT considers incremental tardiness of all unfinished USVs during a transition, whereas Greedy-DT considers only the selected USV’s tardiness. Priority-Constrained Nearest Neighbor (PCNN) selects USVs by priority and then proximity. As a stronger heuristic, 2-step lookahead greedy (2-step LG) extends Greedy-IT to two-step sequences and executes the first action of the best sequence. Exhaustive evaluation over �! rendezvous orders is costly because each sequence requires repeated path searches. Greedy-IT and MLP-RL therefore serve as the primary baselines, while 2-step LG provides a stronger, tractable search-based comparison. Each problem setting used 60 test instances, comprising 30 each of patrol Types A and B, with stochastic RL results averaged over 10 runs per instance. The objective was normalized by 2,000 s.

TABLE VII reports Graph-RL and MLP-RL results from three seeds under the same training setting. The mean and standard deviation are calculated from the mean objective values obtained for each seed. Both methods showed low variation across seeds, while Graph-RL consistently outperformed MLP-RL under both surrogate- and grid-based evaluations. Given the low seed-to-seed variation, further evaluations used a near-average seed from TABLE VII on a separate test set.

TABLE VIII reports the mean and standard deviation across test instances for each patrol fleet sizes � and priority level �. Dark and light shading indicate $p < 0 . 0 5$ and $p < 0 . 1$ , respectively, in paired t-tests against MLP-RL. Boldface denotes the lowest mean objective value. Relative improvements were computed by averaging instance-wise relative objective reductions.

TABLE VII. Performance variability across training seeds
<table><tr><td rowspan="2">Algorithm</td><td colspan="2">Surrogate</td><td colspan="2">Grid-graph</td></tr><tr><td>Mean</td><td>Std</td><td>Mean</td><td>Std</td></tr><tr><td>Graph-RL</td><td>4.145</td><td>0.017</td><td>4.258</td><td>0.066</td></tr><tr><td>MLP-RL</td><td>4.333</td><td>0.004</td><td>4.400</td><td>0.048</td></tr></table>

Graph-RL reduced the objective by 3.50% and 10.52% on average compared with MLP-RL and Greedy-IT, respectively, both statistically significant at the 5% level. Despite not explicitly encoding future rendezvous relationships, MLP-RL and Greedy-IT remained competitive baselines. Compared with the 2-step lookahead greedy across all test instances, Graph-RL significantly reduced the objective by 1.48% at the 5% level while requiring only about 1/35 of the computation time. The higher runtime of 2-step LG results from evaluating $O ( N ^ { 3 } )$ two-step candidate sequences, making additional lookahead steps computationally demanding. Graph-RL and MLP-RL showed similar computation times, while Greedy-IT had an intermediate runtime and Greedy-DT and PCNN were faster. Path search dominated computation time for all methods. Runtime differences among the methods became more pronounced under the 3T and 4T settings, with the gap between the RL methods and the faster heuristics increasing to approximately 2.5 times that observed at 2T. Although priority does not directly affect computational cost, the resulting rendezvous sequences may lead to longer distances to subsequent candidates, increasing path-search costs.

TABLE IX shows that the advantage over MLP-RL grew with fleet size under both evaluation methods. As larger fleets introduce more potential rendezvous relationships, this trend suggests an increasing benefit from encoding future connectivity rather than encoding candidates independently. Across priority tiers, both evaluations showed the same ordering, with the largest improvement at 2T and smaller gains at 3T and 4T, likely reflecting the relatively stronger performance of MLP-RL as the improvement over Greedy-IT remained stable. The similar magnitudes across both evaluations suggest that the sFTG remains informative despite approximation and that the benefit of graph encoding is preserved.

Fig. 3 shows representative multi-rendezvous trajectories. Blue, orange, and green indicate patrol routes in descending order of priority, while the green pentagon and red × mark the command USV’s start and end points. The policy tends to visit high-priority USVs earlier while also favoring nearby rendezvous with little extra travel time.

(c) 16 USVs, 3T, Type A  
TABLE VIII. Objective values and runtime across fleet sizes and priority settings
<table><tr><td rowspan=2 colspan=1>Algorithm</td><td rowspan=1 colspan=3>N = 12</td><td rowspan=1 colspan=3>N = 16</td><td rowspan=1 colspan=3>N = 24</td><td rowspan=1 colspan=1>Mean</td></tr><tr><td rowspan=1 colspan=1>2T</td><td rowspan=1 colspan=1>3T</td><td rowspan=1 colspan=1>4T</td><td rowspan=1 colspan=1>2T</td><td rowspan=1 colspan=1>3T</td><td rowspan=1 colspan=1>4T</td><td rowspan=1 colspan=1>2T</td><td rowspan=1 colspan=1>3T</td><td rowspan=1 colspan=1>4T</td><td rowspan=1 colspan=1>runtime</td></tr><tr><td rowspan=1 colspan=1>Graph-RL</td><td rowspan=1 colspan=1> ${ \bf 3 . 8 4 \pm 0 . 7 }$ </td><td rowspan=1 colspan=1> ${ \bf 4 . 2 8 \pm 0 . 9 }$ </td><td rowspan=1 colspan=1> ${ \bf 4 . 6 5 \pm 1 . 1 }$ </td><td rowspan=1 colspan=1> ${ \bf 4 . 5 6 \pm 0 . 7 }$ </td><td rowspan=1 colspan=1> ${ \bf 5 . 6 3 \pm 1 . 1 }$ </td><td rowspan=1 colspan=1> ${ \bf 6 . 3 2 \pm 1 . 4 }$ </td><td rowspan=1 colspan=1> $7 . 4 0 \pm 2 . 4$ </td><td rowspan=1 colspan=1> ${ \bf 7 . 9 5 \pm 1 . 4 }$ </td><td rowspan=1 colspan=1> $9 . 2 5 \pm 1 . 7$ </td><td rowspan=1 colspan=1>105.37 s</td></tr><tr><td rowspan=1 colspan=1>MLP-RL</td><td rowspan=1 colspan=1> $3 . 8 6 \pm 0 . 6$ </td><td rowspan=1 colspan=1> $4 . 4 1 \pm 0 . 8$ </td><td rowspan=1 colspan=1> $4 . 7 2 \pm 0 . 8$ </td><td rowspan=1 colspan=1> $4 . 8 7 \pm 0 . 8$ </td><td rowspan=1 colspan=1> $5 . 8 0 \pm 1 . 0$ </td><td rowspan=1 colspan=1> $6 . 5 5 \pm 1 . 2$ </td><td rowspan=1 colspan=1> $7 . 8 5 \pm 2 . 2$ </td><td rowspan=1 colspan=1> $8 . 4 0 \pm 1 . 2$ </td><td rowspan=1 colspan=1> $9 . 7 6 \pm 1 . 5$ </td><td rowspan=1 colspan=1>106.63 s</td></tr><tr><td rowspan=1 colspan=1>Greedy-IT</td><td rowspan=1 colspan=1> $4 . 4 2 \pm 1 . 2$ </td><td rowspan=1 colspan=1> $5 . 1 7 \pm 1 . 5$ </td><td rowspan=1 colspan=1> $5 . 6 3 \pm 1 . 7$ </td><td rowspan=1 colspan=1> $5 . 1 1 \pm 1 . 1$ </td><td rowspan=1 colspan=1> $6 . 3 7 \pm 1 . 4$ </td><td rowspan=1 colspan=1> $6 . 9 8 \pm 1 . 5$ </td><td rowspan=1 colspan=1> $9 . 3 9 \pm 3 . 8$ </td><td rowspan=1 colspan=1> $9 . 6 3 \pm 2 . 5$ </td><td rowspan=1 colspan=1> $1 1 . 2 4 \pm 2 . 9$ </td><td rowspan=1 colspan=1> $9 0 . 2 2 \ : \mathrm { s }$ </td></tr><tr><td rowspan=1 colspan=1>Greedy-DT</td><td rowspan=1 colspan=1> $5 . 2 7 \pm 1 . 5$ </td><td rowspan=1 colspan=1> $6 . 8 1 \pm 1 . 6$ </td><td rowspan=1 colspan=1> $7 . 9 6 \pm 2 . 0$ </td><td rowspan=1 colspan=1> $6 . 2 5 \pm 1 . 6$ </td><td rowspan=1 colspan=1> $9 . 3 4 \pm 2 . 3$ </td><td rowspan=1 colspan=1> $1 0 . 4 7 \pm 2 . 5$ </td><td rowspan=1 colspan=1> $1 4 . 2 1 \pm 6 . 4$ </td><td rowspan=1 colspan=1> $1 5 . 4 9 \pm 3 . 6$ </td><td rowspan=1 colspan=1> $1 8 . 6 1 \pm 4 . 2$ </td><td rowspan=1 colspan=1>70.30 s</td></tr><tr><td rowspan=1 colspan=1>PCNN</td><td rowspan=1 colspan=1> $4 . 0 5 \pm 0 . 7$ </td><td rowspan=1 colspan=1> $5 . 0 5 \pm 1 . 3$ </td><td rowspan=1 colspan=1> $6 . 4 2 \pm 2 . 1$ </td><td rowspan=1 colspan=1> $4 . 7 9 \pm 0 . 9$ </td><td rowspan=1 colspan=1> $6 . 9 3 \pm 1 . 8$ </td><td rowspan=1 colspan=1> $9 . 8 4 \pm 4 . 0$ </td><td rowspan=1 colspan=1> $9 . 6 7 \pm 5 . 1$ </td><td rowspan=1 colspan=1> $1 0 . 3 8 \pm 3 . 1$ </td><td rowspan=1 colspan=1> $1 6 . 4 3 \pm 6 . 1$ </td><td rowspan=1 colspan=1>66.68 s</td></tr><tr><td rowspan=1 colspan=1>2-step LG</td><td rowspan=1 colspan=1> $4 . 1 8 \pm 1 . 0$ </td><td rowspan=1 colspan=1> $4 . 5 3 \pm 1 . 0$ </td><td rowspan=1 colspan=1> $4 . 8 9 \pm 1 . 1$ </td><td rowspan=1 colspan=1> $4 . 9 4 \pm 1 . 2$ </td><td rowspan=1 colspan=1> $6 . 1 1 \pm 1 . 8$ </td><td rowspan=1 colspan=1> $6 . 6 5 \pm 2 . 0$ </td><td rowspan=1 colspan=1> $8 . 0 7 \pm 2 . 4$ </td><td rowspan=1 colspan=1> $8 . 3 4 \pm 2 . 1$ </td><td rowspan=1 colspan=1> $9 . 6 9 \pm 2 . 4$ </td><td rowspan=1 colspan=1>3737.2 s</td></tr></table>

TABLE IX. Relative improvements over primary baselines under grid-based and surrogate-based evaluation
<table><tr><td rowspan="2">Comparison</td><td rowspan="2">Evaluation</td><td colspan="3">Fleet size</td><td colspan="3">Priority tier</td><td rowspan="2">Macro avg. (%)</td></tr><tr><td>N = 12</td><td>N = 16</td><td>N = 24</td><td>2T</td><td>3T</td><td>4T</td></tr><tr><td rowspan="2">Graph-RL vs. MLP-RL</td><td>Grid graph</td><td>1.29</td><td>3.86</td><td>5.35</td><td>3.79</td><td>3.44</td><td>3.27</td><td>3.50</td></tr><tr><td>Surrogate</td><td>2.91</td><td>3.82</td><td>5.08</td><td>4.88</td><td>4.50</td><td>2.44</td><td>3.94</td></tr><tr><td rowspan="2">Graph-RL vs. Greedy-IT</td><td>Grid graph</td><td>10.03</td><td>7.46</td><td>14.06</td><td>10.09</td><td>11.24</td><td>10.22</td><td>10.52</td></tr><tr><td>Surrogate</td><td>8.59</td><td>11.00</td><td>17.46</td><td>11.62</td><td>12.54</td><td>12.88</td><td>12.35</td></tr></table>

![](images/034efb2f657522e7189107551a1a723ee1ec9c0a36a88e815c54575ab1589499.jpg)

![](images/39b3c6c1327853fe7b78cd10e0695e20c2a94388341222b52f54d7fa465592f8.jpg)

![](images/dd4dfa427f4494713eb8c75f66e3cf18168d5e6778ebaadbe39e9d05504c2dc7.jpg)  
Fig. 3. Representative rendezvous trajectories generated by Graph-RL

## V.CONCLUSION

This study proposed a computationally tractable framework for multi-rendezvous planning under ocean-current and turn-rate constraints. Continuous turn-rate constraints were approximated by maximum turning-angle constraints calibrated through trajectory-tracking simulations, while a surrogate travel-time model enabled efficient construction of the sFTG for GCN-based rendezvous policy learning.

The proposed policy outperformed Greedy and MLP-RL across various problem settings and showed a statistically significant improvement over lookahead greedy while requiring only about 1/35 of its computation time. The study demonstrates that surrogate-based state representation and policy learning remain effective under evaluation with explicit path and arrival-heading computation. Nevertheless, trajectorytracking validation over complete multi-rendezvous missions and the effects of assuming a fixed command USV heading during waiting remain to be examined. Overall, the proposed framework provides an effective approach to complex multirendezvous planning under ocean currents and turn-rate constraints and can serve as a basis for future studies incorporating more realistic mission-level simulations.

## REFERENCES

[1] F. S. Alqurashi, A. Trichili, N. Saeed, B. S. Ooi, and M.-S. Alouini, “Maritime communications: A survey on enabling technologies, opportunities, and challenges,” IEEE Internet Things J., vol. 10, no. 4, pp. 3525–3547, 2023.

[2] N. Mathew, S. L. Smith, and S. L. Waslander, “Multirobot rendezvous planning for recharging in persistent tasks,” IEEE Trans. Robot., vol. 31, no. 1, pp. 128–142, 2015.

[3] Y. Li, S. Li, Y. Zhang, W. Zhang, and H. Lu, “Dynamic route planning for a USV-UAV multi-robot system in the rendezvous task with obstacles,” J. Intell. Robot. Syst., vol. 107, no. 4, Art. no. 52, 2023.

[4] T. Kundu and I. Saha, “Mobile recharger path planning and recharge scheduling in a multi-robot environment,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2021, pp. 3635–3642.

[5] N. Goulet and B. Ayalew, “Dynamic charging rendezvous and motion planning for a multi-AGV team including a mobile charging host,” IEEE Trans. Robot., 2025.

[6] A. H. Eker, A. Öncü, and H. I. Bozma, “Optimal rendezvous scheduling for charging coordination between aerial–ground vehicles,” IEEE Trans. Aerosp. Electron. Syst., vol. 61, no. 5, pp. 13811–13826, 2025.

[7] B. Li, B. R. Page, J. Hoffman, B. Moridian, and N. Mahmoudian, “Rendezvous planning for multiple AUVs with mobile charging stations in dynamic currents,” IEEE Robot. Autom. Lett., vol. 4, no. 2, pp. 1653–1660, 2019.

[8] H. Kim, D. Kim, J. U. Shin, H. Kim, and H. Myung, “Angular rateconstrained path planning algorithm for unmanned surface vehicles,” Ocean Eng., vol. 84, pp. 37–44, 2014.

[9] T. Lee, H. Kim, H. Chung, Y. Bang, and H. Myung, “Energy efficient path planning for a marine surface vehicle considering heading angle,” Ocean Eng., vol. 107, pp. 118–131, 2015.

[10] I. Bello, H. Pham, Q. V. Le, M. Norouzi, and S. Bengio, “Neural combinatorial optimization with reinforcement learning,” in Proc. Int. Conf. Learn. Representations (ICLR), 2017.

[11] W. Kool, H. van Hoof, and M. Welling, “Attention, learn to solve routing problems!” in Proc. Int. Conf. Learn. Representations (ICLR), 2019.

[12] T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” in Proc. Int. Conf. Learn. Representations (ICLR), 2017.

[13] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” arXiv:1707.06347, 2017.