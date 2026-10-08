# MeshSIPP: Efficient Lattice Planning in Dynamic Environment

Marat Agranovskiy and Konstantin Yakovlev

Abstract— Autonomous navigation in dynamic environments requires computing spatiotemporal trajectories that satisfy nonholonomic motion constraints. When the trajectories of the moving obstacles are predictable or known, a promising approach is to rely on the combination of state lattices constructed from precomputed feasible motion primitives and Safe Interval Path Planning – a search-based algorithm with strong theoretical guarantees. While this approach yields feasible paths, the rich primitive sets needed for smooth navigation induce a large branching factor, which becomes costly when coupled with time-dependent obstacle intervals. To this end, we present MeshSIPP, an efficient planner that removes the computational bottleneck by exploiting the fact that many primitives sweep the same regions and can therefore be validated together. MeshSIPP propagates primitives as spatial bundles, screens them with lightweight bounding-interval checks, and defers the expensive exact departure-time search until a primitive reaches its terminal state. A time-aware pruning rule additionally discards redundant space-time branches early in the search. We prove that the resulting search is complete and optimal. Extensive experiments over more than 6,000 benchmark instances and real-time ROS 2 simulations show that MeshSIPP achieves up to a 3× speedup over state-of-the-art spatiotemporal planners.

## I. INTRODUCTION

Autonomous navigation in environments with dynamic obstacles is a fundamental challenge in mobile robotics. In many practical applications, such as warehouse logistics, the future movements of dynamic obstacles are either explicitly known or can be accurately predicted. Utilizing this predictive information enables a global spatiotemporal planning paradigm, allowing the robot to compute high-quality collision-free paths ahead of time. Beyond avoiding dynamic obstacles, practical deployment requires strict adherence to vehicle kinematics, as non-holonomic platforms (such as carlike or differential-drive vehicles) cannot turn in place and require smoothly curved, kinematically feasible trajectories.

A widespread approach to handle kinodynamic constraints in robotics is sampling-based path planning [1], [2], [3]. Planners of this family naturally operate in continuous spaces and rely on the randomized decomposition of the problem into smaller ones.

They support complex vehicle constraints via customized steering functions and are particularly advantageous in high dimensional configuration spaces, e.g., for robotic manipulators. Moreover, extensions like ST-RRT\* [4] incorporate a temporal dimension to avoid dynamic obstacles. Still, these methods provide only probabilistic guarantees of completeness and optimality and may struggle to find paths in confined configuration workspaces.

![](images/3b3aebfc36e05db482ac56e9ade4a53b67a0d6580c8c9ff56c4d90618e82e83e.jpg)  
Fig. 1: Example of the path planning problem we are interested in. Within a 2D workspace (green: static obstacles, white: free space), the robot with state (x, y, θ) must compute a collision-free path composed of smoothly connected motion primitives (alternating colors) to reach the goal while avoiding dynamic obstacles.

Conversely, lattice-based planners rely on the discretization of the workspace/configuration space and provide strong theoretical guarantees with respect to this discretization. They are often preferable for systems with fewer degrees of freedom, such as mobile robots, where planning primarily involves (x, y, θ). These methods reason over motion primitives – precomputed, kinodynamically feasible motions used to construct the final path (see Fig. 1). Stacked motion primitives form a state lattice, i.e., a graph where vertices correspond to the robot’s states and edges to the feasible transitions between them. A shortest path on this graph can be obtained by the search algorithms like A\* [5].

To effectively extend this approach to dynamic environments, one can utilize Safe Interval Path Planning (SIPP) [6] that reduces the temporal search space by grouping obstaclefree time periods into contiguous safe intervals. This abstraction has been successfully extended to any-angle pathfinding [7], strict acceleration profiles [8], quadrotors [9], highdimensional manipulators [10], anytime planning [11], etc.

However, directly integrating state lattices with SIPP, as in [12], introduces a severe computational bottleneck. To produce smooth and expressive trajectories, rich motionprimitive libraries are required and this leads to a large branching factor. In dynamic environments, this is further amplified by the fragmentation of safe intervals at each state, resulting in a multiplicative increase in search complexity. As a consequence, straightforward combinations of latticebased and safe interval planning become computationally burdensome in complex, obstacle-rich scenarios.

To this end, we introduce MeshSIPP, a novel kinodynamic spatiotemporal planning algorithm that transforms the conventional state lattice into a specialized mesh graph [13]. Because many primitives sweep the same areas, MeshSIPP groups them into coherent bundles and propagates each bundle incrementally along their shared footprint, establishing spatial validity once for the whole group. Extending this idea to dynamic environments is not straightforward: primitives within a bundle have different execution durations, so evaluating exact temporal constraints during propagation would fragment the bundle and destroy the structure that makes it efficient. We resolve this with a lazy evaluation scheme that defers the expensive continuous departure search until a primitive reaches its terminal state, combined with a time-aware pruning rule that discards redundant space-time branches early in the search.

Our contributions are: (i) MeshSIPP, a bundle-based safeinterval search with lazy temporal evaluation; (ii) a bounded, time-aware terminal pruning rule, together with a proof that it preserves completeness and optimality with respect to the given workspace discretization and control set; (iii) an evaluation over 6,000+ instances against four baselines, in which MeshSIPP returns solutions of identical cost to SIPP at a 100% success rate while running $2 - 3 \times$ faster; and (iv) a real-time ROS/Gazebo validation on a TurtleBot3, replanning in 50 ms per cycle amidst 20 non-cooperative moving obstacles.

## II. PROBLEM STATEMENT

Consider a mobile robot, approximated by a bounding circle of radius R, operating in a planar 2D workspace $\mathcal { W } \subset \mathbb { R } ^ { 2 }$ . Its continuous state is defined by its coordinates and heading angle: $\mathbf { s } = ( x , y , \theta ) \in \mathcal { W } \times S ^ { 1 }$ . The underlying motion is subject to non-holonomic kinematic constraints expressed through the dynamic model $\dot { \mathbf { s } } = f ( \mathbf { s } , \mathbf { u } )$ , where u denotes the bounded control inputs. As is typical for mobile platforms, these dynamics are translationally invariant, being independent of the absolute spatial coordinates $( x , y )$

State Lattice and Motion Primitives. To formulate continuous planning as a graph search, we define a discrete set of headings $\Theta = \{ \theta _ { 1 } , \dots , \theta _ { k } \} \subset S ^ { 1 }$ and a finite control set of template motion primitives. Each template specifies an initial heading $\theta \in \Theta$ , a control sequence ${ \bf \delta u } ( t )$ , and an execution duration $\Delta t { \bf \Omega } > { \bf \Omega } 0$ . Due to the translational invariance of the dynamics, integrating $\dot { \mathbf { s } } = f ( \mathbf { s } , \mathbf { u } )$ from an initial state $( x , y , \theta )$ over $\Delta t$ produces a kinematically feasible trajectory segment whose continuous geometric shape is independent of the starting coordinates $( x , y )$ . Consequently, rather than reasoning about time-varying control inputs, these templates can be treated directly as precomputed, invariant trajectory segments that the robot can execute from any spatial location.

Instantiating these templates at a common origin, such as $( 0 , 0 )$ , yields a finite set of relative endpoint displacements that can be embedded into a regular spatial grid (see Fig. 2). This grid is not unique, as any of its refinements preserve the alignment of all primitive endpoints. We choose a resolution that reflects the robot’s sensing and localization capabilities, using the resulting grid as the environmental occupancy map.

To close this construction under transitions, we additionally design each template so that its terminal heading also belongs to Θ. We can therefore define the discrete state space as $S ~ = ~ \{ ( i , j , \theta ) ~ \mid ~ ( i , j ) ~ \in ~ \mathbb { Z } ^ { 2 } , ~ \theta ~ \in ~ \Theta \}$ , where $( i , j )$ corresponds to the cell centers of the chosen grid.

![](images/56ad4d244de68a22e0cf270c0539bf68e7577db69ce5d8178bddb5c535d8b4ef.jpg)  
Fig. 2: Control set with regular grid: totaling 384 primitives across 16 discrete orientations. Two colors are used for visual clarity.

A template with initial heading θ can be instantiated at any state $( i , j , \theta )$ . Because its relative displacement is gridaligned and its terminal heading belongs to Θ, the resulting trajectory terminates at another state in S. Thus, template instantiations define directed edges between discrete states and form a regular state lattice. A path in this graph represents a sequence of smoothly connected primitives, yielding a continuous, kinematically feasible global trajectory whose expressiveness scales with the richness of the control set.

Environment Representation. We represent the environment on the same occupancy grid introduced above, and distinguish between static and dynamic obstacles:

• Static Environment: A binary occupancy map $M ,$ where $M _ { i , j } = 1$ if cell $( i , j )$ is blocked, and 0 otherwise.

• Dynamic Environment: A spatio-temporal occupancy map D, where each cell $( i , j )$ is associated with a set $\bar { D _ { i , j } } ~ = ~ \{ [ t _ { i n } ^ { ( 1 ) } , t _ { o u t } ^ { ( 1 ) } ) , [ t _ { i n } ^ { ( 2 ) } , t _ { o u t } ^ { ( 2 ) } ) , . . . \}$ of disjoint intervals, indicating when the cell is occupied by a moving obstacles. These intervals are precomputed by dilating dynamic obstacles by the robot radius R, sweeping them along their trajectories, and rasterizing the resulting spatio-temporal volumes onto the grid.

Spatial-Temporal Trace. As the robot executes a template primitive, its center sequentially traverses a set of grid cells. For each visited cell $c ,$ we record its coordinate offset $\Delta c =$ $( \Delta i _ { c } , \Delta j _ { c } )$ from the primitive’s origin and the relative time interval $[ \tau _ { i n } ^ { c } , \tau _ { o u t } ^ { c } ) \subseteq [ 0 , \Delta t )$ during which the center resides within c. When the primitive is instantiated at $( i , j , \theta )$ with absolute start time t, this local trace shifts accordingly. The execution is collision-free if all cells in the trace are statically free in M and its absolute intervals do not overlap with any obstacle intervals: $\begin{array} { r } { [ t + \tau _ { i n } ^ { c } , t + \tau _ { o u t } ^ { c } ) \cap D _ { i + \Delta i _ { c } , j + \Delta j _ { c } } = \emptyset . } \end{array}$

Objective. A valid path is a sequence of collision-free motion primitives, optionally interleaved with waits in safe intervals. Each action must start from the discrete state reached by the previous one, ensuring kinematic continuity. Starting from $s _ { 0 } = ( i _ { 0 } , j _ { 0 } , \theta _ { 0 } )$ at time $t _ { 0 } .$ , we seek a path to $s _ { f } = ( i _ { f } , j _ { f } , \theta _ { f } )$ that minimizes the arrival time $t _ { f }$

## III. BACKGROUND: SAFE INTERVAL PATH PLANNING

To effectively navigate dynamic environments where inplace waiting is permitted to avoid obstacles, several methods exist. Standard space-time search algorithms operate over an augmented state space $( x , y , \theta , t )$ . By modeling waiting actions through discrete time increments $( t ~  ~ t + \Delta t )$ these methods often suffer from severe state-space explosion, especially over long planning horizons or in continuous time. Safe Interval Path Planning (SIPP) [6] mitigates this inefficiency by abstracting the continuum of free time at each spatial configuration into a finite set of maximal, contiguous, collision-free windows called $s a f e$ intervals. Over this interval-augmented space, a search state is defined as $\boldsymbol { p } ~ = ~ \langle i , j , \theta , I \rangle$ , and SIPP functions similarly to $\mathbf { A } ^ { * }$ (or Dijkstra’s algorithm) to find a path to each state $p$ that minimizes the arrival time $g ( p ) = t _ { a r r } \in I$ . This formulation enables a dominance property: if multiple paths reach the same spatial configuration within the same safe interval $I ,$ only the earliest arrival time $t _ { a r r }$ needs to be maintained. Any later one is inherently dominated, as the robot could instead arrive earlier and safely wait in place within I until any later moment. This allows SIPP to prune redundant trajectories and bound the search space by the number of safe intervals, regardless of the amount or duration of wait actions.

Consequently, a single spatial configuration gives rise to multiple search states, each corresponding to different time windows (before, between, or after the passage of dynamic obstacles). Maintaining all of them is essential for completeness and optimality, as arriving at an intermediate configuration in a later interval may be the only way to access a downstream safe interval that yields a faster overall path to the goal. Given this representation, the SIPP search proceeds by iteratively expanding states $p \ = \ \langle i , j , \theta , I \rangle$ to generate successors. To do so, all applicable motion primitives m from the current configuration are evaluated. For each target configuration $( i ^ { \prime } , j ^ { \prime } , \theta ^ { \prime } )$ resulting from $m _ { : }$ , the algorithm retrieves the list of available safe intervals. For every target interval $I ^ { \prime } { . }$ , we compute the earliest possible departure time $\scriptstyle t _ { \mathrm { d e p } }$ from the current state — which directly minimizes the arrival time in $I ^ { \prime } -$ such that the robot successfully departs from the current interval, arrives within the target interval, and avoids all dynamic obstacles along the path (see Fig. 3, left). Formally, a valid departure time $\scriptstyle t _ { \mathrm { d e p } }$ must satisfy:

1) $t _ { d e p } \geq t _ { a r r } ,$ , enforcing temporal causality;

2) $t _ { d e p } + m . \tau _ { o u t } ^ { ( 0 , 0 ) } \leq I . t _ { e n d } ,$ , ensuring the agent vacates the origin cell before the current interval I closes, where $\overline { { m . \tau } } _ { o u t } ^ { ( 0 , 0 ) }$ is the time required to leave the cell;

3) $t _ { d e p } + \Delta t _ { m } \in I ^ { \prime } .$ , ensuring arrival at the target cell within the destination interval $I ^ { \prime } { . }$ where $\Delta t _ { m }$ is the execution duration of primitive $m$

This step generates a successor state $p ^ { \prime } = \langle i ^ { \prime } , j ^ { \prime } , \theta ^ { \prime } , I ^ { \prime } \rangle$ with an updated arrival time $g ( p ^ { \prime } ) = t _ { d e p } + \Delta t _ { m }$

We encapsulate the calculation of the time $t _ { d e p } ^ { * }$ within a function GETEARLIESTDEPARTURE $( ( i , j ) , m , t _ { m i n } , t _ { m a x } )$ While its specific implementation is environment-dependent, for a grid-based state space, it can be achieved by projecting all temporal constraints back to the moment of departure. Specifically, the function iterates over all cells traversed by the primitive $m ,$ retrieves the occupied intervals of dynamic obstacles in those cells, and projects them backward in time according to the primitive’s spatial-temporal trace. This projection yields a set of forbidden departure time blocks at the origin cell. By merging and sorting these invalid blocks, the algorithm identifies the earliest free gap that falls within the allowable departure window $[ t _ { m i n } , t _ { m a x } ]$

![](images/bcc4de1564b19219a8ddf6f88e585d2449d3605f2f3193d2c04d6fb93629873e.jpg)  
Fig. 3: Left: Finding the optimal departure time $t _ { \mathrm { d e p } } ^ { \ast }$ for a single motion primitive. Colored curves represent the same primitive shifted across continuous times: gray trajectories collide with dynamic obstacle intervals (black blocks), red options are safe, and the purple trajectory denotes the earliest safe departure we are interested in. Right: Extended cells in the mesh graph, representing a grid cell (magenta) combined with a bundle of motion primitives (teal) passing through it at a specific 0-based spatial trace index k. The bottom panel depicts $( i , \dot { j } , \Psi )$ at index $k = 3$ . The top panel shows its second-order successor $( \dot { i } ^ { \prime } , j ^ { \prime } , \Psi ^ { \prime } )$ at $k = 5 ,$ , retaining only the primitives whose spatial traces match the bundle’s exact sequence of cells (orange ones) up to this point. The remainder are pruned (gray dashed lines) as their traces diverged at an earlier step. Red arrows indicate the endpoints of the surviving primitives in the bundle.

## IV. METHOD: MESHSIPP

The proposed method combines the interval-based reasoning of SIPP with a structural transformation of the search space (state lattice) into a specialized mesh graph. Conducting a search directly over this representation enables early pruning and mitigates redundant state expansions.

## A. Mesh Graph

To reason about motion primitives at the individual gridcell level, we adopt the mesh graph formulation introduced in [13]. In conventional state-lattice planning, search nodes represent discrete states $( i , j , \theta )$ and edges represent full motion primitives. In contrast, the mesh graph decomposes the primitive execution cell-by-cell along its spatial trace, allowing primitives with overlapping footprints to be evaluated simultaneously in bundles within the same search node.

Formally, nodes in the mesh graph are extended cells, defined as tuples $\boldsymbol { u } = ( i , j , \Psi )$ , where $( i , j )$ specifies the 2D grid cell and Ψ represents a primitive configuration:

$$
\Psi = \left\{ ( m _ { 1 } , k ) , ( m _ { 2 } , k ) , \ldots , ( m _ { n } , k ) \right\} .
$$

Here, each pair $( m , k ) ~ \in ~ \Psi$ indicates that an instance of the template primitive m (from a precomputed control set) traverses cell $( i , j )$ as the k-th cell (0-based index) in its spatial trace. Thus, Ψ captures the information about a specific bundle of primitives simultaneously passing through cell $( i , j )$ at identical spatial offsets k (see Fig. 3).

To establish equivalence with the conventional state lattice, extended cells are categorized into two types:

• Initial Extended Cells: Characterized by a configuration $\Psi _ { \boldsymbol \theta } = \{ ( m _ { 1 } , 0 ) , \dots , ( m _ { r } , 0 ) \}$ containing all control-set primitives originating at orientation θ with step index $k \ = \ 0 .$ . An initial cell $( i , j , \Psi _ { \theta } )$ serves as the direct functional counterpart to a standard lattice state $( i , j , \theta )$

• Regular Extended Cells: Others: intermediate cells $( k >$ 0) along the spatial traces of active primitives from a bundle, as depicted in Fig. 3 (right).

Transitions in the mesh graph propagate active bundles cell-by-cell along the spatial traces of their constituent primitives. To generate a successor configuration for an immediate neighbor, the algorithm filters the current bundle Ψ: it retains only the primitives whose next swept cell matches that neighbor, while primitives routing elsewhere are excluded. For instance, in Fig. 3 (right, bottom panel), the bundle at trace index $k = 3$ physically diverges. If the search follows the rightward branch and advances two steps, it reaches a second-order successor $( i ^ { \prime } , j ^ { \prime } , \Psi ^ { \prime } )$ at $k ~ = ~ 5$ (top panel). The primitives that routed along the alternate branch are inherently pruned from $\Psi ^ { \prime }$ (illustrated by gray dashed lines). This local filtering mechanism ensures that primitive bundles remain grouped as long as they share a spatial path, splitting only when their traces physically diverge.

When a primitive $m \in \Psi$ completes its spatial trace, it reaches a specific discrete endpoint state $\left( i _ { m } , j _ { m } , \theta _ { m } \right)$ (indicated by red arrows in Fig. 3, right). Reaching this terminal state initiates a new bundle, meaning the succeeding node is represented as an initial extended cell $( i _ { m } , j _ { m } , \Psi _ { \theta _ { m } } )$ Edge costs in the mesh graph are assigned as follows:

• Transitions into regular extended cells carry a cost 0.

• Transitions into initial extended cells are assigned the full cost of the completed primitive, $\mathrm { C o s t } ( m )$ . Exactly one primitive completes upon entering any given initial cell, ensuring this edge cost is uniquely defined.

Formally, we encapsulate the entire neighbor-generation process in a function MESHSUCCESSORS $( i , j , \Psi )$ , which yields the set of all valid successor extended cells $( i ^ { \prime } , j ^ { \prime } , \Psi ^ { \prime } )$

Crucially, paths between initial extended cells are in bijection with motion-primitive sequences on the state lattice, preserving both total cost and exact spatial trace.

## B. MeshSIPP: Lazy Temporal Evaluation

Integrating continuous temporal constraints directly into the spatial mesh graph requires careful handling of primitive execution times. Because primitives within a single bundle possess distinct durations $\Delta t .$ , evaluating exact dynamic collisions cell-by-cell would prematurely fragment the bundles, diminishing the core efficiency of the mesh graph structure. To preserve the structural advantage of simultaneous primitive evaluation, we introduce a lazy temporal evaluation scheme. This approach decouples spatial traversal from strict dynamic interval validation: spatial propagation along regular cells utilizes lightweight bounding checks, while the exact continuous departure search is deferred until primitives reach their endpoints at initial cells.

Search Space Elements. To support this delayed evaluation, we augment the spatial extended cells of the mesh graph with the temporal structure of SIPP.

Definition 1 (MeshSIPP State): A state in the MeshSIPP search space is defined as a tuple $\begin{array} { c c l } { u } & { = } & { \langle i , j , \Psi , s _ { \mathrm { r o o t } } \rangle } \end{array}$ where $( i , j , \Psi )$ is the spatial extended cell, and $s _ { \mathrm { r o o t } } =$ $\left. i _ { 0 } , j _ { 0 } , \theta _ { 0 } , I _ { 0 } \right.$ is the base SIPP state from which the current bundle Ψ originated. Thus, $\big ( i _ { 0 } , j _ { 0 } , \Psi _ { \theta _ { 0 } } \big )$ corresponds to the last initial extended cell traversed on the path up to u. The g-value (current time) of the state u is defined entirely by the arrival time of its root state, $\begin{array} { r } { \mathrm { i . e . , } g ( u ) = g ( s _ { \mathrm { r o o t } } ) = t _ { \mathrm { a r r } } . } \end{array}$

In this augmented space, the root state $S _ { \mathrm { r o o t } }$ acts as a temporal anchor. As the bundle propagates through intermediate cells, time remains logically anchored at $t _ { \mathrm { a r r } } ,$ and transition costs are inherently zero. Time advances and costs accumulate exclusively when a primitive terminates and a new temporal anchor is instantiated.

Successor Generation and Conservative Pruning. Algorithm 1 details the successor generation process. For a state u, the algorithm first retrieves the spatial successors from the underlying mesh graph via MESHSUCCESSORS (Line 3).

If a transition yields a REGULAR cell (Line 4), we apply conservative temporal pruning (Lines 5–8). We estimate a bounding window $[ t _ { \mathrm { m i n } } , t _ { \mathrm { m a x } } )$ representing the earliest possible arrival and latest possible departure of any primitive within the bundle $\Psi ^ { \prime }$ traversing the current cell. If dynamic obstacles $( D _ { i ^ { \prime } j ^ { \prime } } )$ completely subsume this window, the transition is pruned. Otherwise, it forms a new regular state with zero transition cost (Lines 9–10).

Conversely, reaching an INITIAL successor (past Line 11) means a primitive m has completed, triggering the deferred exact evaluation. For each destination safe interval $I ^ { \prime } ,$ , we compute a bounded departure window $[ t _ { 1 } , t _ { 2 } ]$ relative to $s _ { \mathrm { r o o t } }$ (Lines 18–19) and execute the continuous departure search (Line 20). If a valid departure time $\scriptstyle t _ { \mathrm { d e p } }$ is found, we establish a new temporal anchor $s _ { \mathrm { n e w } }$ and assign the true transition cost as the required wait time at the root cell plus the primitive’s execution duration: $\left( { t _ { \mathrm { d e p } } } - { t _ { \mathrm { a r r } } } \right) + m . \Delta t .$

Heuristic. As in other informed search algorithms, we use a heuristic to estimate the remaining cost to the goal and focus the search on promising states. Let $\widehat { h } _ { \mathrm { S I P P } }$ denote an admissible SIPP heuristic; as is common, it may ignore the temporal component and estimate only the remaining motion cost. For an initial MeshSIPP state, we use the heuristic value of its corresponding SIPP state. For a regular state $u \ : = \ : \langle i , j , \Psi , s _ { \mathrm { r o o t } } \rangle$ , each $( m , k ) \ \in \ \Psi$ defines a discrete endpoint state $s _ { m }$ obtained by completing $m .$ . Since the cost of a primitive is incurred only at its endpoint, we set

$$
h ( u ) = \operatorname* { m i n } _ { ( m , k ) \in \Psi } \left\{ \Delta t _ { m } + \widehat { h } _ { \mathrm { S I P P } } ( s _ { m } ) \right\} .
$$

If $\widehat { h } _ { \mathrm { S I P P } }$ also accounts for safe intervals, the minimum is additionally taken over all reachable safe intervals of the

Algorithm 1 Generating Successors of a MeshSIPP State   
Input: MeshSIPP state $u = \langle i , j , \Psi , s _ { r o o t } \rangle$ with time $t _ { a r r }$   
Output: Set of successors and transition costs (v, cost)   
1: Successors ← ∅   
2: for all $\langle i ^ { \prime } , j ^ { \prime } , \Psi ^ { \prime } \rangle \in \mathbf { M E S H S U C C E S S O R S } ( i , j , \Psi )$ do   
3: if $\Psi ^ { \prime }$ is REGULAR then   
4: $\begin{array} { r } { t _ { m i n } = t _ { a r r } + \operatorname* { m i n } _ { ( m , k ) \in \Psi ^ { \prime } } ( \tau _ { i n } ^ { k } ) } \end{array}$   
5: $\begin{array} { r } { t _ { m a x } = t _ { a r r } + \operatorname* { m a x } _ { ( m , k ) \in \Psi ^ { \prime } } ( \tau _ { o u t } ^ { k } ) } \end{array}$   
6: $\mathbf { i f } \ [ t _ { m i n } , t _ { m a x } ) \subset I _ { b }$ for some $I _ { b } \in D _ { i ^ { \prime } j ^ { \prime } }$ then   
7: continue ▷ Fully covered by obstacles   
8: $v _ { i n t }  \langle i ^ { \prime } , j ^ { \prime } , \Psi ^ { \prime } , s _ { r o o t } \rangle$   
9: Successors.add $( ( v _ { i n t } , 0 ) )$   
10: continue   
11: $\theta ^ { \prime } \gets$ heading from which $\Psi ^ { \prime }$ originates ▷ $\Psi ^ { \prime } = \Psi _ { \theta ^ { \prime } }$   
12: m ← the primitive that just terminated to form $\Psi ^ { \prime }$   
13: $p \gets ( s _ { r o o t } . i , ~ s _ { r o o t } . j )$ ▷ Origin cell   
14: $\tau ^ { * } \gets m . \tau _ { o u t } ^ { ( 0 , 0 ) }$ ▷ Time to clear origin   
15: SafeInterva $1 \mathrm { s }  \mathrm { R } _ { \ge 0 } \setminus D _ { i ^ { \prime } , j ^ { \prime } }$   
16: for all $I ^ { \prime } = [ t _ { s t a r t } , t _ { e n d } ] \bar { \bf \Phi } \in \bar { \bf S } .$ afeIntervals do   
17: $t _ { 1 } \gets \mathrm { m a x } ( t _ { a r r } , ~ t _ { s t a r t } - m . \Delta t )$   
18: $t _ { 2 } \gets \operatorname* { m i n } ( s _ { r o o t } . I . t _ { e n d } - \tau ^ { * } , ~ t _ { e n d } - m . \Delta t )$   
19: $t _ { d e p } \gets \mathbf { G } \mathbf { \mathrm { E T } } \mathbf { E } \mathbf { \mathrm { A } }$ RLIESTDEPARTUR $\mathfrak { z } ( p , m , t _ { 1 } , t _ { 2 } )$   
20: $\mathbf { i f } \ t _ { 1 } < t _ { 2 }$ and $t _ { d e p } \neq \infty$ then   
21: $s _ { n e w } \gets \langle i ^ { \prime } , j ^ { \prime } , \theta ^ { \prime } , I ^ { \prime } \rangle$   
22: $v _ { t e r m } \gets \langle i ^ { \prime } , j ^ { \prime } , \Psi ^ { \prime } , s _ { n e w } \rangle$   
23: $t _ { a r r } ^ { n e w } \gets t _ { d e p } + m . \Delta t$ $\mathsf { \Pi } \triangleright = g ( s _ { n e w } )$   
24: $c o s t \gets ( t _ { d e p } - t _ { a r r } ) + m . \Delta t$   
25: Successors.add ((v<sub>term</sub>, cost))   
26: return Successors

primitive endpoints.

## C. Early Terminal Pruning in MeshSIPP

A profound advantage of transitioning to a cell-by-cell search space (mesh graph) is the structural capability to prune unpromising branches prematurely. In a static environment, a regular search node can be safely discarded if all its spatial endpoints are already expanded. Since any valid path from this node must pass through one of these endpoints, its further exploration is redundant. Extending this principle to SIPP, however, introduces a temporal challenge: a single spatial endpoint $( i , j , \theta )$ encompasses multiple safe intervals.

To maintain theoretical optimality without exhaustive global interval checks, we introduce a bounded, time-aware pruning mechanism. For a regular state $u = \langle i , j , \Psi , s _ { \mathrm { r o o t } } \rangle$ with arrival time $t _ { \mathrm { a r r } } .$ the physically possible arrival times at any spatial endpoint via a primitive m are strictly confined to a continuous window $W _ { m } = [ t _ { \mathrm { m i n } } , t _ { \mathrm { m a x } } ]$ . It reflects the constraint that motion cannot initiate before the agent’s arrival $- \ t _ { \mathrm { m i n } } = t _ { \mathrm { a r r } } + m . \Delta t - \mathrm { o r }$ later than the interval’s closure, accounting for the time required to fully vacate the root grid cell: $t _ { \mathrm { m a x } } = s _ { \mathrm { r o o t } } . I . t _ { \mathrm { e n d } } - m . \tau _ { \mathrm { o u t } } ^ { ( 0 , 0 ) } + m . \Delta t$

To safely prune u, every safe interval I intersecting $W _ { m }$ must be explored – meaning its corresponding SIPP state has already been expanded by the search (which, under $\mathbf { A } ^ { * }$ properties, guarantees its optimal cost is securely fixed), or its current arrival time matches the theoretical absolute minimum of the interval, $I . t _ { \mathrm { s t a r t } }$ . If even one intersecting interval remains unexplored, the endpoint might yield a better path, and pruning is immediately aborted.

Algorithm 2 MeshSIPP Early Terminal Pruning   
Input: Regular $u = \langle i , j , \Psi , s _ { r o o t } \rangle$ with arrival time $t _ { a r r }$   
Output: true if u is pruned, false otherwise   
1: $\Delta t _ { m a x } \gets \operatorname* { m a x } _ { m \in \gamma \in \gamma \rho \nu \rho ^ { + } } ( m . \Delta t )$ ▷ Slowest primitive   
2: for all endpoint $( i ^ { \prime } , j ^ { \prime } , \theta ^ { \prime } )$ for each primitive $m \in \Psi$ do   
3: $t _ { m i n } \gets t _ { a r r } + m . \Delta t$ ▷ $W _ { m } = [ t _ { m i n } , t _ { m a x } ]$   
4: $t _ { m a x } \gets s _ { r o o t } . I . t _ { e n d } - m . \tau _ { o u t } ^ { ( 0 , 0 ) } + m . \Delta t$   
5: $\mathcal { T }  ( \mathbb { R } _ { \ge 0 } \setminus D _ { i ^ { \prime } , j ^ { \prime } } ) \cap [ t _ { m i n } , t _ { m a x } ]$ ▷ Reachable   
6: if I is empty then   
7: continue ▷ Arrival within $W _ { m }$ is blocked   
8: $I _ { l a s t } \gets$ last interval in I ▷ Nearest to $t _ { m a x }$   
9: if not $\mathrm { I s E x p L O R E D } \big ( \langle i ^ { \prime } , j ^ { \prime } , \theta ^ { \prime } , I _ { l a s t } \rangle \big )$ then   
10: return false ▷ Found unexplored interval   
11: $t _ { c u t o f f } \gets \operatorname* { m i n } ( t _ { m a x } , ~ t _ { a r r } + \Delta t _ { m a x } )$   
12: $\mathcal { T } _ { 0 } $ intervals in I before $t _ { c u t o f f }$   
13: for all $I \in \mathcal { T } _ { 0 }$ (checked backward) do   
14: if not $\operatorname { I s E x p L O R E D } ( \langle i ^ { \prime } , j ^ { \prime } , \theta ^ { \prime } , I \rangle )$ then   
15: return false   
16: for all $I \in \mathcal { T } \setminus \mathcal { T } _ { 0 } \setminus \left\{ I _ { l a s t } \right\}$ (checked backward) do   
17: if not ISEXPLORED $( \langle \dot { \iota } ^ { \prime } , j ^ { \prime } , \theta ^ { \prime } , I \rangle )$ then   
18: return false   
19: return true ▷ All reachable intervals are explored

Theorem 1: MeshSIPP is complete and optimal, yielding identical solutions (in terms of path cost and spatial traces) to the standard SIPP algorithm over the same control set.

Proof Sketch: The proof establishes that augmenting the mesh graph with temporal anchors $( s _ { \mathrm { r o o t } } )$ and deferred continuous evaluation yields a dynamic search space isomorphic to the standard SIPP state lattice with respect to optimal paths. We first prove the correspondence for a single motion primitive. Specifically, MeshSIPP constructs a chain of states whose grid projections form exactly the primitive’s spatial trace, ensuring identical interactions with static obstacles in both representations. Once the trace reaches the primitive’s endpoint, MeshSIPP performs the exact continuous temporal evaluation, as in SIPP, yielding the same departure window, successor safe interval, and transition cost. This correspondence extends to arbitrary sequences of primitives by concatenating the corresponding MeshSIPP chains. Finally, the pruning rules do not break this correspondence: conservative pruning removes only branches whose entire reachable temporal window is occupied (Algorithm 1, Line 7), while early terminal pruning removes only branches whose reachable intervals have already been explored.

## D. Heuristic Interval Ordering

While verifying all intervals within $W _ { m }$ guarantees optimality, a sequential scan introduces a computational bottleneck. To accelerate the discovery of unexplored intervals – and thereby abort the pruning check faster – we evaluate the intervals in a heuristically permuted order.

Our strategy is designed to fail-fast by exploiting the chronological expansion properties of A\*. Since later time intervals are typically explored deeper in the search tree, scanning reachable intervals in reverse chronological order maximizes the probability of immediately encountering an unexplored state. Furthermore, a powerful bound can be established based on the expansion status of the latest reachable interval. If $I _ { \mathrm { l a s t } }$ (nearest to $t _ { \mathrm { m a x } } )$ has already been explored, it implies that the search previously reached this spatial cell via some primitive m from a predecessor state expanded at some earlier time $t ^ { \prime } \leq t _ { \mathrm { a r r } }$ . The same spatial trajectory with zero (or minimal) waiting would have reached earlier safe intervals at arrival times starting from $t ^ { \prime } + \Delta t$ . Since $t ^ { \prime } \le t _ { a r r } ,$ we can strictly bound these early arrival times by defining a cutoff threshold: $t ^ { \prime } + \Delta t \leq t _ { a r r } + \operatorname* { m a x } _ { m \in \mathrm { C o n t r o l } \mathrm { S e t } } ( \Delta t ) = :$ $t _ { c u t o f f }$ . Consequently, all intervals from $t _ { m a x }$ down to $t _ { c u t o f f }$ are highly likely to have already been expanded via these zero-wait continuations, unless a dynamic obstacle has blocked those earlier arrivals.

This logic is implemented in Algorithm 2. First, it evaluates $I _ { \mathrm { l a s t } } - \mathrm { i f }$ unexplored, the pruning check aborts immediately (Lines $8 \mathrm { - } 1 0 )$ . Second, it jumps to $t _ { \mathrm { c u t o f f } }$ and scans backward toward $t _ { \mathrm { m i n } }$ (Lines 11–15). Finally, it verifies any remaining intervals between $t _ { \mathrm { c u t o f f } }$ and $I _ { \mathrm { l a s t } }$ , also in backward order (Lines 16–18). This permuted sequence dramatically minimizes average memory accesses. Because the entire window $W _ { m }$ is comprehensively verified before a state is conclusively pruned (Line 19), the fail-fast heuristic strictly preserves the optimality guarantees of the algorithm.

## V. EXPERIMENTS

To evaluate the performance, scalability, and execution efficiency of MeshSIPP, we conducted both numerical benchmarks and ROS2/Gazebo simulations.

## A. Numerical experiments

In the experiments we use the grid maps from the Moving AI benchmark [14]: Denver 1 256 (city layout, 256×256), arena (open space with structured static obstacles, 49×49), and empty 64x64 (unobstructed grid, 64 × 64). Dynamic obstacles were simulated as independent agents moving along predefined trajectories. For each obstacle density, we generated sets of problem instances with randomly assigned valid start and goal configurations. In total, more than 6,000 evaluation runs were executed to ensure representativeness.

The motion primitive control set was constructed following the methodology of [15] (Fig. 2). Robot orientations are discretized into 16 headings and for each one the control set provides 24 primitives with lengths ranging from 1 to 9 grid units, each terminating at a cell center and aligned with the valid discrete orientations. Motion along all primitives is executed at a constant speed of 0.1 cells per tick, where a tick represents the base simulation time step.

Baselines. We compare MeshSIPP algorithm against four baselines that share the same underlying kinematic limits (maximum velocity and turning radius).

• SIPP: The standard Safe Interval Path Planning algorithm with motion primitives using the same control set from Fig. 2. Both SIPP and MeshSIPP were implemented in C++ to provide a meaningful comparison.

• ST-RRT\*: A spatiotemporal variant of the probabilistically optimal RRT\* algorithm [4], connecting sampled states $( x , y , t )$ via constant-velocity linear segments. We utilized the standard implementation from the Open Motion Planning Library (OMPL v1.7.0 for C++) [16]. The time budget was set generously to max $( 1 . 0 , 2 \times T _ { S I P P } )$ seconds, where $T _ { S I P P }$ is the SIPP baseline runtime.

• ST-RRT\*-Dubins: An extension of the previous baseline where the spatial space is replaced with Dubins space. Consequently, the algorithm samples fourdimensional states $( x , y , \theta , t )$ and connects them via Dubins curves, natively respecting the agent’s kinematic constraints. Due to the increased computational complexity of the steering function, the time budget was doubled relative to standard ST-RRT\*.

• ORCA: A two-stage approach widely used in robotics. A path is first planned ignoring dynamic obstacles. Then Optimal Reciprocal Collision Avoidance (ORCA) local planner [17] tracks the path while reactively avoiding dynamic agents. We utilized the official implementation from the RVO2 library (also in C++) [18]. Notably, while standard ORCA assumes cooperative collision avoidance where all agents share the responsibility equally, the dynamic obstacles in our setup are strictly non-cooperative. To account for this, we adjusted the ORCA velocity update factor, increasing it from the standard 0.5 to 1.0, thereby assigning the entire avoidance maneuver exclusively to the ego-agent.

Results. The experimental results are summarized in Fig. 4(a) and Table I. As anticipated, the MeshSIPP algorithm consistently achieves identical optimal path costs (1.0 ratio) and a 100% success rate across all evaluated instances, matching the SIPP baseline. At the same time, it delivers a substantial reduction in computation time. The performance graphs demonstrate a stable speedup of approximately 2 to 3 times over the baseline SIPP. Crucially, this efficiency gain remains highly consistent across different map types and varying quantities of dynamic obstacles, demonstrating the scalability and reproducibility of MeshSIPP.

In contrast, the performance of ORCA depends heavily on obstacle density. Being a purely reactive local planner that bypasses complex graph searches, ORCA exhibits exceptional speed in sparse scenarios. In these simpler cases, ORCA frequently achieves a 100% success rate and nearoptimal trajectory costs. However, as the density of dynamic obstacles increases, its performance sharply degrades. In heavily populated environments, the high density of dynamic agents forces the reactive planner into continuous evasive maneuvers and oscillatory behavior, substantially increasing both the final path cost (execution time of the trajectory) and the total computation time required by ORCA to simulate and calculate the path itself. Because the obstacles are non-cooperative and strictly follow their predefined trajectories, they frequently corner the ego-robot into unavoidable collisions. This fundamental limitation is reflected in a drastically reduced success rate, dropping to 67.6% on the empty 64 64 map with 80 obstacles and 70.1% on the Denver 1 256 map with 200 obstacles.

![](images/67b5c08c3626cc903c0f197f886f147168981d4fc41ac708cc70208c4c66b706.jpg)  
Fig. 4: Performance and trajectory evaluation of the tested algorithms. (a) Median runtime relative to standard SIPP across three map topologies with varying dynamic obstacle counts. (b) Snapshots of trajectory execution at t = 250.0 ticks on the arena map populated with 20 dynamic obstacles. The ego-agent (red circle, radius 1.0) navigates from start (42, 7) to goal (14, 37).

TABLE I: Median cost ratio and success rate (Cost / SR) relative to standard SIPP.
<table><tr><td rowspan="2">Algorithm</td><td colspan="3">arena</td><td colspan="3">empty-64_64</td><td colspan="3">Denver_1_256</td></tr><tr><td>N = 0</td><td>N = 5</td><td>N = 35</td><td>N = 0</td><td>N = 20</td><td>N = 80</td><td>N = 0</td><td>N = 50</td><td>N = 200</td></tr><tr><td>SIPP</td><td>1.0 / 100%</td><td>1.0 / 100%</td><td>1.0 /  100%</td><td>1.0 / 100%</td><td>1.0 / 100%</td><td>1.0 /  100%</td><td>1.0 /  100%</td><td>1.0 /  100%</td><td>1.0 / 100%</td></tr><tr><td>MeshSIPP</td><td>1.0 / 100%</td><td>1.0 / 100%</td><td>1.0 / 100%</td><td>1.0 / 100%</td><td>1.0 / 100%</td><td>1.0 / 100%</td><td>1.0 / 100%</td><td>1.0 / 100%</td><td>1.0 / 100%</td></tr><tr><td>ORCA</td><td>1.0 / 100%</td><td>1.0 / 100%</td><td>1.3 / 90.2%</td><td>1.0 / 99.6%</td><td>1.1 / 100%</td><td>1.6 / 67.6%</td><td>1.0 / 98.5%</td><td>1.1 / 98.6%</td><td>1.3 / 70.1%</td></tr><tr><td>ST-RRT*</td><td>1.8 / 87.8%</td><td>1.6 / 87.4%</td><td>1.9 / 88.0%</td><td>1.7 / 91.8%</td><td>1.7 / 91.2%</td><td>2.0 / 90.3%</td><td>2.6 / 89.9%</td><td>2.2 / 89.7%</td><td>2.0 / 85.9%</td></tr><tr><td>ST-RRT*-Dubins</td><td>2.2 / 80.0%</td><td>2.3 / 79.3%</td><td>3.3 / 54.3%</td><td>1.8 / 89.3%</td><td>2.1 / 87.1%</td><td>3.9 / 68.8%</td><td>2.4 / 85.9%</td><td>2.4 / 84.2%</td><td>2.5 /  76.5%</td></tr></table>

For the sampling-based baselines, ST-RRT\* and ST-RRT\*- Dubins, execution time is bounded by a predefined budget, making direct runtime comparisons uninformative. Despite receiving generous computational budgets, both variants frequently fail to reach a 100% success rate. Unlike ORCA, where a failure denotes a physical collision during execution, a failure in the sampling-based planners indicates an inability to discover a collision-free space-time trajectory before the time limit expires. When successful, the stochastic nature of RRT\* inherently yields suboptimal paths, which is reflected in the elevated median cost ratios (up to 3.9×) in Table I.

The runtime speedup profiles in Fig. 4(a) provide additional context regarding the scalability of the samplingbased methods. Because the RRT\* variants were allotted a minimum base budget of 1.0 second, their relative speedup appears near zero in simpler scenarios where the deterministic SIPP resolves the problem in milliseconds. However, in the most computationally demanding scenarios, such as the Denver 1 256 map with extensive dynamic traffic, SIPP’s runtime increases significantly. Consequently, the relative speedups of the ST-RRT\* variants rise to meet their theoretical budget caps $( \frac 1 2$ and $\textstyle { \frac { 1 } { 4 } }$ of SIPP’s runtime). This convergence highlights that while sampling-based methods carry an excessive overhead for standard planar spatiotemporal navigation, their fixed-budget scaling becomes competitive as the combinatorial complexity of the exact search space approaches extremes, reinforcing their standard application in highly complex, high-dimensional configuration spaces (such as multi-DOF manipulators).

Sample trajectory executions and their total costs are illustrated in Fig. 4(b). Indeed, SIPP and MeshSIPP produce smooth lowest-cost paths.

## B. Real-Time Validation in ROS 2 and Gazebo

To validate MeshSIPP in a realistic physics engine, we deployed the planner within a Robot Operating System (ROS 2) [19], [20] and Gazebo [21] simulation. The physical environment and system architecture were strictly configured to test the planner’s real-time adaptability.

• Workspace: A 10 × 10 meter room, discretized into a 100 × 100 occupancy grid. This yields a spatial resolution of 0.1 meters per cell, utilized for both internal planning and RViz visualization.

• Kinematics: The planner utilizes the motion primitives from Fig. 2. Time resolution is fixed at 1 tick = 0.08 seconds. Consequently, the planned movement speed across grid cells is 0.1 meters per 0.08 seconds, equating to 1.25 m/s (approx. 4.5 km/h) – a velocity representative of a fast pedestrian or a standard service robot.

• Agents: The ego-robot is modeled as a standard Turtle-Bot3 Waffle Pi (radius 0.22 meters) [22]. The environment is densely populated with 20 dynamic obstacles, modeled as moving cylinders with a radius of 10 cm and a height of 1.5 meters.

• System Architecture: A custom ROS 2 node manages dynamic obstacles, invokes MeshSIPP, publishes the generated trajectory to RViz, and tracks the resulting path via a PID wheel velocity controller. Standard localization is handled by the Nav2 [23] stack.

![](images/7d87ac0405c15a6984f8d30320f233425934ec16e030a828195fe2b0b0f5b5a1.jpg)  
Fig. 5: Real-time navigation in the ROS 2 and Gazebo simulation. Left: The 3D physical environment in Gazebo, featuring the Turtle-Bot3 ego-agent, dynamic obstacles (red cylinders), and simulated 2D LiDAR scans (blue rays). Right: The corresponding RViz visualization showing the occupancy grid, inflated obstacle footprints, and the planned MeshSIPP trajectory (orange/blue primitives).

To bridge the gap between deterministic planning and real-world perception limits, we imposed a strict planning horizon. The robot is only permitted to predict dynamic obstacle trajectories for the next 200 ticks (16 seconds). Any time state beyond this 16-second threshold is assumed to be free of dynamic obstacles. To compensate for this limited foresight, the robot triggers a mandatory replan every 3 seconds. Throughout the simulation, MeshSIPP demonstrated high efficiency, requiring an average of only 50 ms of computation time for each replanning cycle.

By utilizing this receding-horizon strategy, the trajectory dynamically adapts and shifts over time as the ego-robot closes the distance to the moving obstacles. This allows the robot to seamlessly weave through dense, non-cooperative traffic and successfully reach the goal without any collisions.

This adaptive behavior is visualized in Fig. 5. The top panels display the early stages of navigation, where distant obstacles fall beyond the temporal horizon and the robot plans a relatively direct route. The bottom panels illustrate the adapted trajectory later in the run, showcasing the planner heavily modifying the route to evade the approaching obstacle cluster. Full continuous execution of this scenario is available in the supplementary demonstration video.

## VI. CONCLUSION

In this paper, we presented MeshSIPP, an efficient spatiotemporal path planning algorithm that integrates statelattice motion primitives with Safe Interval Path Planning (SIPP). By searching over a mesh graph with timeaware pruning, the method eliminates redundant branches while maintaining completeness and optimality. Evaluations demonstrate a consistent 3× runtime speedup over standard lattice SIPP with identical trajectory quality.

Future work will extend MeshSIPP to multi-agent systems via decentralized or prioritized frameworks and validate the algorithm on physical robots in real-world scenarios.

## REFERENCES

[1] S. Karaman and E. Frazzoli, “Incremental sampling-based algorithms for optimal motion planning,” ArXiv, vol. abs/1005.0416, 2010.

[2] S. LaValle and J. Kuffner, “Randomized kinodynamic planning.” I. J. Robotic Res., vol. 20, pp. 378–400, 01 2001.

[3] S. Karaman and E. Frazzoli, “Sampling-based algorithms for optimal motion planning,” The International Journal of Robotics Research, vol. 30, no. 7, pp. 846–894, 2011.

[4] F. Grothe, V. N. Hartmann, A. Orthey, and M. Toussaint, “St-rrt\*: Asymptotically-optimal bidirectional motion planning through spacetime,” in 2022 International Conference on Robotics and Automation (ICRA), 2022, pp. 3314–3320.

[5] P. E. Hart, N. J. Nilsson, and B. Raphael, “A formal basis for the heuristic determination of minimum cost paths,” IEEE Transactions on Systems Science and Cybernetics, vol. 4, no. 2, pp. 100–107, 1968.

[6] M. Phillips and M. Likhachev, “Sipp: Safe interval path planning for dynamic environments,” in 2011 IEEE International Conference on Robotics and Automation, 2011, pp. 5628–5635.

[7] K. Yakovlev and A. Andreychuk, “Any-angle pathfinding for multiple agents based on sipp algorithm,” in International Conference on Automated Planning and Scheduling, vol. 27, no. 1, 2017, pp. 586– 594.

[8] Z. A. Ali and K. Yakovlev, “Safe interval path planning with kinodynamic constraints,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 10, pp. 12 330–12 337, Jun. 2023.

[9] S. Huang, Y. Wu, Y. Tao, and V. Kumar, “Safe interval motion planning for quadrotors in dynamic environments,” in 2025 IEEE International Conference on Robotics and Automation, 2025, pp. 2780–2786.

[10] N. Kerimov, A. Onegin, and K. Yakovlev, “Safe interval randomized path planning for manipulators,” in International Conference on Automated Planning and Scheduling, vol. 35, no. 1, 2025, pp. 213–217.

[11] V. Narayanan, M. Phillips, and M. Likhachev, “Anytime safe interval path planning for dynamic environments,” in IEEE/RSJ International Conference on Intelligent Robots and Systems, 2012, pp. 4708–4715.

[12] M. Agranovskiy, “Path planning with motion primitives in dynamic environments: Sipp on lattices,” in Interactive Collaborative Robotics. Springer Nature Switzerland, 2027, pp. 406–420.

[13] M. Agranovskiy and K. Yakovlev, “MeshA\*: Efficient path planning with motion primitives,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 43, pp. 36 785–36 792, Mar. 2026.

[14] N. Sturtevant, “Benchmarks for grid-based pathfinding,” Transactions on Computational Intelligence and AI in Games, vol. 4, no. 2, pp. 144 – 148, 2012.

[15] B. Nagy and A. Kelly, “Trajectory generation for car-like robots using cubic curvature polynomials,” in Field and Service Robotics (FSR 01), Helsinki, Finland, June 2001.

[16] I. Sucan, M. Moll, and L. Kavraki, “The open motion planning library,” IEEE Robotics & Automation Magazine, vol. 19, pp. 72–82, 12 2012.

[17] J. van den Berg, S. Guy, M. Lin, and D. Manocha, Reciprocal n-Body Collision Avoidance, 04 2011, vol. 70, pp. 3–19.

[18] J. Snape and lithander, “snape/rvo2: v2.0.2,” May 2016.

[19] M. Quigley, K. Conley, B. P. Gerkey, J. Faust, T. Foote, J. Leibs, R. Wheeler, and A. Y. Ng, “Ros: an open-source robot operating system,” in ICRA Workshop on Open Source Software, 2009.

[20] Y. Maruyama, S. Kato, and T. Azumi, “Exploring the performance of ros2,” in Proceedings of the 13th International Conference on Embedded Software, ser. EMSOFT ’16. New York, NY, USA: Association for Computing Machinery, 2016.

[21] N. Koenig and A. Howard, “Design and use paradigms for gazebo, an open-source multi-robot simulator,” in 2004 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) (IEEE Cat. No.04CH37566), vol. 3, 2004, pp. 2149–2154 vol.3.

[22] E. Guizzo and E. Ackerman, “The turtlebot3 teacher [resources hands on],” IEEE Spectrum, vol. 54, no. 8, pp. 19–20, 2017.

[23] S. Macenski, F. Martin, R. White, and J. Gines Clavero, “The marathon´ 2: A navigation system,” in 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2020.