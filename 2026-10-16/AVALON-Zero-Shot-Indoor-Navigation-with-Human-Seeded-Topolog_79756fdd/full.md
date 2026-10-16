# AVALON: Zero-Shot Indoor Navigation with Human-Seeded Topological Memory

Jing Xia<sup>1</sup>, Changyan Li<sup>1</sup>, Yuduo Wang<sup>1</sup>, Guiyin Tian<sup>1</sup>, Bo Wang<sup>2</sup>, Yifan Liao<sup>1</sup>, Mingjun Li<sup>1</sup>, Shiyan Hu<sup>1</sup>

Abstract— Indoor service robots often enter unmapped buildings whose layouts and task constraints are already familiar to users. We address this human-familiar, robot-unseen setting with AVALON, a topological navigation framework organized around human-seeded layered spatial memory, nodeverified topological execution, and task-triggered, evidencegated human–robot memory co-updating. The robot has no prior on-site traversal or site-specific training. Deployment requires only a floor plan, place clues, and an initial topology-odometry alignment. Visual depth point-goal navigation leverages verified node arrivals to perform translational re-anchoring, without requiring a robot-constructed global SLAM map. Human–robot memory co-updating fuses complementary sources of information. Human operators specify task constraints that are unobservable via vision and report offline environmental modifications, whereas the robot provides physical-state evidence through active in-situ validation. Task relevance governs whether human-reported environmental changes undergo robot validation during the active mission. Field-collected perceptual evidence gates memory updates, maintaining an explicit separation between human-specified permissions and robot-observed traversability. Our experiments on a Unitree Go2 show that AVALON with X-NavDP completes a 12.81 m Intersection-to-Dining mission in 46.4 s (5/5) with no mapping, 40.9% less robot time than DimOS-Go2 (78.5 s, $4 / 5 ) \ r _ { ; }$ , while policy-only X-NavDP times out in all of the 5 runs. These demonstrate the superiority of the proposed topological navigation framework.

## I. INTRODUCTION

Indoor service robots often enter unmapped buildings whose layouts, routes, and closures are already familiar to users. This human-familiar, robot-unseen setting offers a remarkable opportunity for rapid deployment of robots. Users can supply a floor plan and delivery instructions before the first mission (Fig. 1). The robot must turn these priors into executable navigation memory and detect disagreements with observations.

Dedicated pre-mapping and online topological exploration [1]–[3] both require robot data collection that may delay service. Existing CAD, BIM [4], and user knowledge offer alternative spatial priors, but these must be grounded in observations and updated as conditions change.

In this paper, we propose AVALON (Asymmetric Visual-Architectural Layered Online Navigation), built around seed, execute, and co-update. A spatial memory framework is constructed based on floor-plan topological skeletons with visibility-aware constrained traversal hops, and paired with editable task memory that shares unified node and edge identifiers. Multi-modal contextual clues, including casual imagery, linguistic descriptions, and RoadBook annotations, are explicitly associated with these topological primitives. The topological skeleton provides global route planning priors for the robot, while the attached multi-modal clues serve as verification signatures to validate the arrival at target topological nodes. Local point-goal control governs the robot’s inter-node navigation, and node arrival is rigorously validated via fused visual-semantic observations and odometry measurements.

Successful arrival confirmation enables translational reanchoring and progressive route execution. In case of visual observation mismatch with the expected node template, the robot adaptively refines its localization hypothesis. For ob ject search tasks, real-time perceptual inference is deployed to verify object presence within the corresponding spatial regions, and the collected perceptual evidence is archived to support subsequent downstream missions. Robot onboard observations continuously record environmental physical states, whereas human operators supplement task-specific constraints and update unobserved environmental variations. Benefiting from the initialized prior memory seed, the system can execute initial missions without standalone dedicated mapping procedures. Furthermore, human-robot collaborative memory updating maintains the timeliness and accuracy of the memory bank, accommodating dynamic changes in workspace layouts and task specifications.

AVALON replaces globally precise localization with human-seeded, task-sufficient spatial grounding. Topology specifies where to go, sparse semantic anchors re-ground route progress, and terminal perception confirms what the task seeks. AVALON turns human-held spatial knowledge into executable memory for a robot’s first mission by coupling a human-seeded structural prior, node-verified local execution, and task-triggered evidence acquisition through stable graph IDs. The main contributions of this paper can be summarized as follows.

1) Human-Seeded Layered Spatial Memory. We propose to decouple a floor-plan skeleton from editable task memory, and use shared node-edge identifiers to associate human-provided clues, task constraints, and robot sensory evidence. This design eliminates the requirement for a dedicated pre-mapping traversal prior to deployment.

2) Node-Verified Topological Execution. The robot navigates along human-seeded routes via local graph transitions. Node-associated visual and textual clues validate arrival and re-anchor execution progress. Mismatches between real observations and expected node signatures trigger incremental revision of the localization hypothesis.

![](images/781d3bd7f5ccdcbfa4aa37723e6ac15c7a3d719f6789f6664836dce2dbb07cd2.jpg)

![](images/6839b22030d8dde0a8955c5798d10c989e56cdb959927f88cabdaf320a3c3d34.jpg)  
Fig. 1. Standard mapping-then-execute versus AVALON: a user seeds a floor-plan graph and the robot executes the first mission without a mapping pass. Measured time is in Table II.

TABLE I  
COMPARISON ON HOW FIRST-MISSION SPATIAL MEMORY IS OBTAINED AND USED, WHERE MEMORY CO-UPDATING DENOTES WHETHER USER ANDTHE ROBOT WRITE BACK TO THE SAME NODE/EDGE MEMORY IN THE FIRST MISSION.
<table><tr><td>Method</td><td>Spatial prior</td><td>Robot mapping</td><td>Localization</td><td>Memory co-updating</td></tr><tr><td>FloNa</td><td>floor plan</td><td>no</td><td>metric pose</td><td>no</td></tr><tr><td>osmAG-Nav</td><td>CAD</td><td>no</td><td>LiDAR pose</td><td>no</td></tr><tr><td>RTAB-Map</td><td>none</td><td>dedicated mapping</td><td>metric map</td><td>no</td></tr><tr><td>NTS</td><td>none</td><td>exploration</td><td>node</td><td>no</td></tr><tr><td>AVALON</td><td>CAD + user clues</td><td>no (seed)</td><td>node verify</td><td>yes</td></tr></table>

3) Task-Triggered, Evidence-Gated Human-Robot Memory Co-Updating. Graph identifiers are shared across execution and memory maintenance modules. Human permissions remain authoritative, while change notifications invoke on-site physical validation only when relevant to the active task. Physical world states are updated exclusively using spatially localized robot evidence, and verified execution outcomes refresh the spatial memory graph to support subsequent missions.

4) Experimental Validation. We validate our proposed closed-loop pipeline on a Dining-to-Meeting-Room navigation scenario involving a human-reported cart at the entryway. On a Unitree Go2, AVALON with X-NavDP completes a 12.81 m Intersection-to-Dining mission in 46.4 s (5/5) with no mapping, 40.9% less robot time than DimOS-Go2 [5] (78.5 s, 4/5), while policy-only X-NavDP times out in all of the 5 runs. In the five AVALON + X-NavDP Dining-to-Meeting-Room trials reported in Table II, user reports of environmental changes trigger on-site robot verification before the entrance node condition is updated. AVALON successfully completes the report–verification–update sequence and subsequently reaches the meeting room in all five trials. These demonstrate the superiority of the proposed topological navigation framework.

## II. RELATED WORK

Table I compares the first-mission memory in terms of the categories including what prior exists, whether the robot must map, how it localizes, and whether user later edit the same memory with the robot.

## A. Floor Plans and CAD Priors

Floor plans have been used to localize a camera [6], [7] and to condition a learned navigation policy [8]. Imperfect drawings can also guide exploration [9]. Other user-provided sketches include posted signs [10], [11] and hand-drawn maps [12]. CAD and BIM can initialize a workspace model before any robot visit [4], [13].

osmAG-Nav demonstrates how CAD priors can support planning on the passage graph together with structure-aware LiDAR localization [14]. In contrast, AVALON addresses a different first-deployment question, i.e., how architectural priors and user-held task knowledge can become executable shared memory before the robot traverses the site. It converts a floor plan into a visibility-constrained point-goal skeleton, while stable node and edge IDs bind user photos, language descriptions, and RoadBook notes to that structure. During execution, these clues verify semantic progress and gate translational re-anchoring. The same IDs subsequently organize robot evidence and user edits. AVALON therefore couples first-mission routing, semantic grounding, and human-robot memory co-updating without a dedicated robot mapping pass or LiDAR-map localization.

## B. Two-Stage Mapping and Exploration

A common first-mission pattern is two-stage, i.e., the robot maps or explores, then executes. Metric SLAM such as RTAB-Map builds a global map in a dedicated pass and later localizes against it [15]. Topological methods such as NTS, TSGM, and HGR grow a node-edge graph while exploring [1]–[3]. SSMG-Nav and MG-Nav likewise construct semantic or geometric memories online [16], [17]. OPENNAVMAP fuses multi-session image trajectories into a shared graph with full-pose estimates [18]. In all of these cases the robot must traverse the site before it has an executable map. In contrast, AVALON seeds the backbone from the floor plan so the first mission already has nodes and edges. Verified arrivals re-anchor translation at those nodes.

## C. Map-Free Policies

Goal-directed policies can run with no site map [19]–[23], and zero-shot semantic navigation seeks objects in unseen scenes [24]–[26]. AVALON interfaces with NavDP and X-NavDP through robot-frame point goals and evaluates both policies without its graph. These policies receive no user floor plan and are not treated as object-goal benchmarks. osmAG-LLM uses map-guided object confirmation, while Bee-Nav separates transit from terminal visual guidance [27], [28]. AVALON instead reaches a graph-associated candidate region and confirms the requested object from current observations. Timed trials are room-to-room with terminal confirmation, not open-ended search. Here, zero-shot means no on-site mapping or site-specific training, given only a human-provided floor plan and initial topology-odometry alignment.

## D. Human-Robot Memory Updates

Human-in-the-loop semantic SLAM lets operators correct geometry and labels on a robot-built map [29]. In contrast, AVALON’s co-updating is at task level. Human users encode navigational permissions and environmental updates that are inherently unobservable via onboard vision, while the robot grounds memory entries with physical occupancy measurements acquired in situ. Human-reported modifications relevant to the active task undergo on-site robotic validation before physical-state memory is rewritten; other reports remain pending until a later relevant task. We validate this closed-loop human–robot memory update pipeline in realworld Dining-to-Meeting-Room navigation deployments.

## III. METHOD

## A. Problem Formulation and System Overview

Assuming that a CAD or floor-plan drawing is given, we build a node-edge skeleton $G \ = \ ( V , E )$ with stable IDs from it. Place photos, language descriptions, and RoadBook notes are then attached to the nodes on this graph. A single alignment between the topology frame and robot odometry is required at initialization. The robot has no on-site traversal, site-specific training, or robot-built global SLAM map. A mission specifies a room, an object description, or a reference image [26], which the memory associates with one or more graph nodes. For room navigation, success is declared when the destination node passes arrival verification. For object search, verified arrival at a candidate node determines where to search, but success is declared only after live perception detects the requested object.

Figure 2 outlines the three stages of user-seeded memory construction, verified execution, and evidence-gated coupdating. Section III-B first builds the CAD-derived skeleton, then attaches human clues. Section III-C follows the graph through a robot-frame point-goal interface, instantiated here with RGB-D local policies [22], [23], and re-anchors translation at verified semantic nodes. Section III-D keeps human permissions separate from physical state and accepts physical-state updates only with localized robot evidence (Fig. 3).

## B. Human-Seeded Layered Spatial Memory

In the construction, a CAD or floor-plan raster first produces the node-edge skeleton, and photos, language descriptions, and RoadBook notes are attached by users to the nodes in the graph. Route connectivity is therefore separate from the evidence used to recognize places.

1) Structural backbone: In contrast to CAD-to-osmAG’s area-passage representation [13], our visibility-constrained skeleton construction produces straight local hops. Wall thresholding, exterior flood fill, and connected-component filtering extract a walkable pixel set $\mathcal { F } \subset \mathbb { Z } ^ { 2 }$ from a floorplan raster with known resolution and origin. We skeletonize ${ \mathcal F } ,$ group endpoints and junctions into vertices, and prune short or low-clearance branches. When an endpoint chord leaves ${ \mathcal { F } } _ { : }$ intermediate waypoints are inserted along the chain. Shortcutting removes a waypoint only if the resulting segment remains in ${ \mathcal F } .$ The vertex set V comprises these endpoints, junctions, and inserted waypoints. Edges in E join successive vertices on retained chains, not every mutually visible pair. Each hop is a free-space raster segment:

$$
( u , v ) \in E \implies \ell ( q _ { u } , q _ { v } ) \subseteq \mathcal { F } ,\tag{1}
$$

where $q _ { u } , q _ { v } \in \mathcal { F }$ are node pixels and $\ell ( q _ { u } , q _ { v } )$ is the Bresenham segment between them [30]. Metric positions $p _ { u } , p _ { v }$ follow from the same resolution and origin. Route costs can be generic such as Euclidean length $w _ { u v } = \| p _ { u } - p _ { v } \| _ { 2 }$ . These geometric constraints support route generation but do not replace footprint clearance or online obstacle checks. After E is formed, a grid visibility preview on $\mathcal { F }$ may annotate each vertex with a VGA openness score [31] copied from nearby free-space cells. Hop visibility in (1) (whether a straight edge stays in F) is distinct from this per-node openness (how much free space is visible from that vertex).

2) Node roles and memory attachment: Users then attach photos, language descriptions, or RoadBook notes to nodes, and can change node roles and edit edges between deployments, while keeping the same IDs. The editable graph distinguishes semantic nodes from connection waypoints. Both store metric positions and participate in routing. Users can use the openness score as a clue when assigning these roles. Visually open junctions are preferred as recognition anchors, while low-openness samples are often left as waypoints. The score does not itself label a node or authorize re-anchoring. Semantic nodes may carry descriptions, visual clues, and reference or subsequently attached images. They can be queried for place recognition and become position-correction anchors after confirmation. Verification is gated on stored images, not on openness. Nodes without images, including connection waypoints, are used only for walking and do not trigger verification or translational re-anchoring. Geometric subdivision therefore does not imply a new recognizable place. Semantic references and later observations remain separately stored.

![](images/676e2093921b389283a4d23a3602e5ddeb512f04693e833e5df5166f08778ae0.jpg)  
Fig. 2. AVALON system overview. Users seed a stable structural skeleton from CAD/floor plans and attach task-level spatial memory (casual photos, text, and RoadBook notes) on stable IDs. In actual operations, the robot follows the seeded graph with local motion, verifies semantic-node arrival, and re-anchors translation without pre-mapping. Verified visits write back to the same IDs. Users can edit the same memory between deployments.

3) Execution interface: Routing uses the full graph, whereas semantic candidate selection uses only semantic nodes. Semantic adjacency traverses connection-waypoint chains and stops at the next semantic node, preserving candidate relationships without requiring every intermediate waypoint to be recognizable. RoadBook notes, attached after skeleton construction, record route preferences and nonvisual task constraints on node IDs, keeping human permissions separate from observed physical conditions. Current execution uses route preferences and structured rule parsing and enforcement remain extensions. This representation supplies routes and recognition evidence to Section III-C, and shared update targets to Section III-D.

## C. Node-Verified Topological Execution

The system develops a hierarchical execution pipeline as follows. The topological graph schedules sequential node transitions, a local navigation policy executes point-goal tracking, and semantically verified arrivals correct translational alignment. For object-search tasks, terminal perceptual validation independently confirms target presence to complete mission execution.

1) Graph Route and Point-Goal Execution: Given a task associated with one or more nodes, we compute a route on G from the current topological neighborhood to a goalassociated node. Semantic nodes and connection waypoints both participate in routing. Only semantic nodes later serve as recognition anchors. RoadBook entries may bias route costs but do not authorize unsafe motion.

Deployment initializes planar alignment between topology frame M and odometry O. For the current destination v with map position $p _ { v } ^ { M }$ , fixed initial rotation $R _ { 0 }$ , and latest translational correction $t _ { k }$ , the robot-frame goal is

$$
g _ { t } ^ { B } = R _ { O B } ( t ) ^ { \top } \big ( R _ { 0 } p _ { v } ^ { M } + t _ { k } - p _ { B } ^ { O } ( t ) \big ) ,\tag{2}
$$

where $p _ { B } ^ { O }$ and $R _ { O B }$ are odometric position and body-toodometry rotation, respectively. This interface can use a local navigation policy that consumes the robot-frame goal $g _ { t } ^ { B }$ and onboard observations to produce local motion. We instantiate it with NavDP [22] and X-NavDP [23], which map live RGB-D observations and $g _ { t } ^ { B }$ to short-horizon waypoints. Reaching a geometric waypoint advances the route but does not verify node identity.

2) Semantic Verification and Position-Only Re-Anchoring: Upon entering a semantic-node verification region, the robot stops and holds position until verification is accepted or rejected. Live multi-view observations are compared with stored descriptions and images. Candidates are restricted to the last confirmed semantic node and its semantic neighbors, which are reached by traversing connection-waypoint chains. Therefore, geometrically nearby but topologically unrelated places are not treated as alternatives.

A vision-language model (VLM) compares the stopped robot’s multi-view observations $\{ I _ { t } ^ { ( j ) } \} _ { j = 1 } ^ { J }$ with the stored image-text reference set $\mathcal { R } _ { v }$ of the active candidate.Its match decision and the odometry-range gate are

$$
\begin{array} { r l } & { M _ { t } ( v ) = \mathbf { 1 } \Big [ f _ { \mathrm { V L M } } \big ( \{ I _ { t } ^ { ( j ) } \} _ { j = 1 } ^ { J } , \mathcal { R } _ { v } \big ) = \mathrm { m a t c h } \Big ] , } \\ & { O _ { t } ( v ) = \mathbf { 1 } \big [ \alpha \ell _ { k } ( v ) \le d _ { k } ^ { \mathrm { o d o m } } \le \beta \ell _ { k } ( v ) \big ] , } \\ & { A _ { t } ( v ) = M _ { t } ( v ) \wedge O _ { t } ( v ) , } \end{array}\tag{3}
$$

where 1[·] is the indicator function that outputs one if the enclosed condition is true and zero otherwise, $\ell _ { k } ( v )$ is the nominal route length from the last verified anchor to $v , \ d _ { k } ^ { \mathrm { o d o m } }$ is the odometric distance accumulated over the same interval, and $0 < \alpha \leq 1 \leq \beta$ are configurable deployment parameters defining the admissible distance range. The gates are conjunctive rather than weighted. That is, only $A _ { t } ( v ) = 1$ authorizes re-anchoring and later reference promotion. Geometric proximity alone remains unverified. Identity contradictions trigger relocalization over a broader candidate set. A blockage at a confidently identified edge withholds that hop without invalidating location. Uncertain localization cannot authorize a confirmed edge update. Note that nodes encode position but not orientation. At verified node $v _ { k }$ and acceptance time $\tau _ { k }$

$$
t _ { k } = p _ { B } ^ { O } ( \tau _ { k } ) - R _ { 0 } p _ { v _ { k } } ^ { M }\tag{4}
$$

updates subsequent goals in (2) while leaving $R _ { 0 }$ fixed. Heading continues to come from odometry. Because recognition is asynchronous, the correction uses odometry at acceptance rather than at image capture, and stale results are discarded if the robot has left the trigger region. Position corrections are excluded from traveled distance.

3) Task-Sufficient Spatial Grounding: Verified anchors interrupt translational drift such that local execution need not rely on a globally accurate pose. Let $\varepsilon _ { \mathrm { v e r } }$ bound the planar residual between the stopped robot and the true node anchor after verification, including the node acceptance region and stopping error. Let $\varepsilon _ { M }$ bound target-position error induced by floor-plan node coordinates, map scale, and the initial topology-odometry registration. Between correctly verified anchors, after actual travel $s \leq L _ { k }$ , denote the upper bound of planar position error by $\varepsilon _ { \mathrm { v e r } } + \rho s$ , the upper bound of map-position error by $\varepsilon _ { M }$ , the upper bound of heading error by $\bar { \theta } \le \pi$ , and the upper bound of true local-goal distance by $D _ { \mathrm { m a x } }$ . Using Equation (2), comparing $g _ { t } ^ { B }$ with the true robot-frame goal $\bar { g } _ { t } ^ { B }$ one has

$$
\lVert g _ { t } ^ { B } - \bar { g } _ { t } ^ { B } \rVert \leq \varepsilon _ { \mathrm { v e r } } + \rho L _ { k } + \varepsilon _ { M } + 2 D _ { \operatorname* { m a x } } \sin ( \bar { \theta } / 2 ) .\tag{5}
$$

The preceding relations are operationally coupled. The point goal in (2) uses the latest translational correction, which is updated by (4) only when the arrival gate in (3) accepts a semantic node. Let $\Delta _ { \pi , c }$ denote the empirical local-goal perturbation tolerance of local policy π under scene condition c. A conditional compatibility requirement for verification interval k is

$$
B _ { k } = \varepsilon _ { \mathrm { v e r } } + \rho L _ { k } + \varepsilon _ { M } + 2 D _ { \operatorname* { m a x } } \sin ( \bar { \theta } / 2 ) \leq \Delta _ { \pi , c } .\tag{6}
$$

Equation (5) supplies the bound $B _ { k }$ under the stated assumptions. Neither it nor (6) guarantees navigation success. Motivated by capability matching in feedback roadmaps and funnel composition [32], [33], the condition makes explicit the interface between accumulated perturbation and localpolicy capability. AVALON limits the translational contribution $\rho L _ { k }$ by starting a new interval after each correctly verified semantic anchor. Geometric waypoints do not delimit this interval. The current system neither estimates $\Delta _ { \pi , c }$ online nor enforces (6) during skeleton construction. The tolerance must be calibrated empirically for each policy and scene. Position-only re-anchoring does not remove yaw or map error. From the above, one can formulate the tackled conditional feasibility problem as follows.

$$
\begin{array} { r l } { \mathbf { F i n d } \ } & { v _ { k } , \tau _ { k } , t _ { k } } \\ { \mathrm { s . t . } } & { A _ { t } \big ( v _ { k } \big ) = M _ { t } \big ( v _ { k } \big ) \wedge O _ { t } \big ( v _ { k } \big ) = 1 , } \\ & { t _ { k } = p _ { B } ^ { O } \big ( \tau _ { k } \big ) - R _ { 0 } p _ { v _ { k } } ^ { M } , } \\ & { B _ { k } = \varepsilon _ { \mathrm { v e r } } + \rho L _ { k } + \varepsilon _ { M } + 2 D _ { \operatorname* { m a x } } \sin ( \bar { \theta } / 2 ) \leq \Delta _ { \pi , c } . } \end{array}
$$

After the arrival at goal-associated topological nodes is verified by our proposed technique, an onboard openvocabulary detector in [34] will be applied to analyze the real-time observations to localize the target object. The object-search mission completes exclusively upon successful target detection, as a single negative observation is insufficient to assert object absence and necessitates further perceptual sampling. Notably, this semantic confirmation operates without precise global pose estimates.

## D. Task-Triggered, Evidence-Gated Human-Robot Memory Co-Updating

AVALON links memory maintenance to execution through verified graph IDs. Human and robot inputs exhibit asymmetric authority. Specifically, user configure permissions and log environment changes beyond robot observations, whereas the robot supplies localized physical evidence. A pending update becomes a sensing objective when its corresponding node or edge affects the current route, allowing execution to selectively refresh memory for subsequent missions. During a mission, the operational model $\mathcal { M } _ { t } = ( V , E , \Phi _ { t } )$ updates only the state features $\Phi _ { t } .$ , while the deployed topology $( V , E )$ remains fixed. Authorized users may revise the graph before a subsequent deployment.

For each graph element $x \in V \cup E$ , its task memory separates three fields:

$$
\phi _ { t } ( x ) = \big ( s _ { t } ( x ) , \ell _ { t } ( x ) , p _ { t } ( x ) \big ) ,\tag{8}
$$

where $s _ { t } ( x ) \in$ {unknown, clear, blocked} is the robotobservable physical condition, $\ell _ { t } ( x )$ is an open-vocabulary semantic description, and $p _ { t } ( x ) \in$ {allowed, forbidden} is a human-authorized task permission. These fields are decoupled by update authority, not assumed statistically independent. Human reports remain pending hypotheses over affected fields rather than accepted physical state.

![](images/46d7ae044b12f1540749a108cf1d6dee30a254e7610dd4eca621067f558297f1.jpg)  
Fig. 3. Shared permission and physical state at one doorway. Panel titles and memory boxes state the four cases. The text gives the update rules.

1) Complementary authority: Localized robot evidence may revise $s _ { t } ( x )$ and attach an observed $\ell _ { t } ( x )$ , but cannot modify $p _ { t } ( x )$ . Authorized users set permissions, but permission does not establish physical clearance. Updates are fieldspecific, i.e., a confident clearance assessment may revise $s _ { t } ( x )$ while the obstacle label remains uncertain, whereas a semantic correction does not itself establish traversability. A robot-observed blockage always stops execution.

2) Task-relevant verification: If the affected node or edge is required by the current route, the robot checks it during that mission. Otherwise, the report remains pending for a later relevant task. Onboard open-vocabulary detections identify object-related conditions, while depth and local feasibility checks assess immediate geometric clearance. These checks update memory evidence but never replace the online safety controller.

3) Evidence-gated reuse: A physical-state update requires a resolved graph ID and localized robot evidence. A doorway observation may update its entrance-node condition, whereas edge traversability additionally requires both endpoints and evidence covering that hop. Node-reference promotion still requires $A _ { t } ( v ) \ = \ 1$ from Section III-C. Records retain source, time, and evidence. Stale or conflicting revisions remain pending. Memory identifies where evidence is needed, and verified execution determines which observations can revise it without replacing online safety checks.

## IV. EXPERIMENTS

## A. Experimental Setup

Experiments use a Unitree Go2 with an Insta360 X3 360<sup>◦</sup> camera, an Intel RealSense D455F RGB-D camera, and an NVIDIA Jetson AGX Orin. NavDP and X-NavDP send RGB-D observations over ROS 2 to an RTX 5090 workstation, which returns local waypoints at 6 Hz. Semanticnode verification calls Qwen3.5-Omni [35] through its API. YOLO-World [34] runs on the Jetson at 15 Hz for objectrelated memory verification and endpoint target confirmation (a red trash bin in Dining or a chair in the Meeting Room). Odometry uses the Unitree /odom interface. The office scene has an 8-node floor-plan graph (24.11 m central trunk) seeded with 5 user photos. Timed routes are Intersectionto-Dining (12.81 m, 6 nodes) and Dining-to-Meeting-Room (20.43 m, 7 nodes). The five AVALON + X-NavDP Diningto-Meeting-Room trials in Table II also evaluate memory co-updating; their 78.7 s includes cart-report verification and memory update (Fig. 5(b)). Each method is evaluated in five trials from a comparable start pose. A trial is marked successful if the robot reaches the designated goal region. AVALON requires no on-site robot mapping pass and topology–odometry alignment is initialized once at deployment time.

![](images/dd2c405d72464797fad51b4ce36e58f8709c468c76210b9e119a9e4f14cc4e3e.jpg)  
Fig. 4. Left: Go2 sensing platform. Right: equirectangular view, four cubemap projections, and RGB-D observations for node verification and local execution.

Baselines. DimOS-Go2, based on the DimOS agentive operating system [5], maps a region slightly larger than the task workspace (onboard odometry, WebRTC LiDAR, VoxelBlockGrid with column carving, height/slope costmap, replanning A\*, no loop closure) at 0.55 m/s, then navigates. Policy-only NavDP [22] and X-NavDP [23] receive one route-destination point goal without the AVALON graph, node verification, or re-anchoring. We pair each with its AVALON-enabled counterpart under the same route, 0.50 m/s speed limit, and 120 s timeout.

Metrics. We report success rate, mean execution time over successful trials, and robot deployment time $T _ { \mathrm { d e p } } ~ =$ $T _ { \mathrm { m a p p i n g } } + T _ { \mathrm { e x e c u t i o n } }$ . For all methods, success requires reaching the goal region and detecting its target with YOLO-World within 120 s. If all trials fail, execution and deployment equal the timeout. Human-prior production, map I/O, and pose initialization are excluded from $T _ { \mathrm { d e p } }$

## B. Room-to-Room Deployment

Table II compares first-mission robot time and pairs each local policy with and without AVALON. With the X-NavDP backend, AVALON completes the Intersection-to-Dining mission in 46.4 s versus 78.5 s for DimOS, 40.9% less, because it skips on-site mapping. Per-route execution is slower (46.4 s vs. 33.3 s): AVALON pauses at semantic nodes to verify identity, and the 0.05 m/s speed-limit gap alone does not explain the difference. The unsuccessful DimOS (a) Walk past the office booth, continue into the dining room, and stop beside the dining table.

![](images/155a266694ad7fbdd923cd1c2ee61c07c9dbad80b76fdc015c9e29a9582b871b.jpg)  
Fig. 5. Qualitative navigation and memory co-updating. (a) Intersection-to-Dining first-mission route. (b) Dining-to-Meeting-Room co-updating case: a user reports a cart at the office entrance, the robot checks the location, records the entrance node as clear, and reaches the meeting room.

TABLE II  
FIRST-MISSION RESULTS ON TWO ROUTES IN THE SAME OFFICE (FIVE TRIALS PER METHOD AND ROUTE), WHERE “POLICY-ONLY” REMOVES THE AVALON GRAPH, VERIFICATION, AND RE-ANCHORING.
<table><tr><td>Method</td><td>Success rate</td><td>Mapping (s)</td><td>Execution (s)</td><td>Deployment (s)</td></tr><tr><td colspan="5">Intersection → Dining (12.81 m)</td></tr><tr><td>DimOS-Go2</td><td>4/5 (80%)</td><td>45.2</td><td>33.3</td><td>78.5</td></tr><tr><td>NavDP (policy-only)</td><td>0/5 (0%)</td><td>0</td><td>120</td><td>120</td></tr><tr><td>AVALON + NavDP</td><td>3/5 (60%)</td><td>0</td><td>83.4</td><td>83.4</td></tr><tr><td>X-NavDP (policy-only)</td><td>0/5 (0%)</td><td>0</td><td>120</td><td>120</td></tr><tr><td>AVALON + X-NavDP</td><td>5/5 (100%)</td><td>0</td><td>46.4</td><td>46.4</td></tr><tr><td colspan="5">Dining → Meeting Room (20.43 m)</td></tr><tr><td>DimOS-Go2</td><td>5/5 (100%)</td><td>60.2</td><td>52.3</td><td>112.5</td></tr><tr><td>NavDP (policy-only)</td><td>0/5 (0%)</td><td>0</td><td>120</td><td>120</td></tr><tr><td>AVALON + NavDP</td><td>1/5 (20%)</td><td>0</td><td>93.6</td><td>93.6</td></tr><tr><td>X-NavDP (policy-only)</td><td>0/5 (0%)</td><td>0</td><td>120</td><td>120</td></tr><tr><td>AVALON + X-NavDP</td><td>5/5 (100%)</td><td>0</td><td>78.7</td><td>78.7</td></tr></table>

trial drifted in odometry and entered an unintended dead-end corridor. On the 20.43 m Dining-to-Meeting-Room route, AVALON with X-NavDP achieves 5/5 success in 78.7 s deployment time, compared with 112.5 s and 5/5 success for DimOS-Go2. Policy-only X-NavDP receives one routedestination point goal; all five trials on each route reach the 120 s timeout. Figure 5(a) shows the Intersection-to-Dining route.

## C. Robustness to Odometry Perturbations

Table II establishes the benefit of the graph-guided interface. Table III assesses the sensitivity of full AVALON to odometry perturbations while keeping the graph, route, X-NavDP backend, semantic-node verification, 0.50 m/s speed limit, and 120 s timeout fixed. The perturbation model adds zero-mean Gaussian increment noise and a random-walk bias before accumulating the noisy pose. Its dimensionless parameter odom noise strength controls the noise scale and is set to 0.02 and 0.04 for the low and high conditions, respectively. This experiment evaluates robustness to increasing noise rather than isolating the effect of re-anchoring.

TABLE III  
SENSITIVITY OF FULL AVALON TO CONTROLLED ODOMETRY NOISE. ALL SETTINGS USE THE SAME GRAPH, 12.81 M ROUTE, AND X-NAVDP LOCAL POLICY (FIVE TRIALS). EXECUTION TIME AVERAGES SUCCESSFUL TRIALS ONLY.
<table><tr><td>Odometry setting</td><td>odom_noise_strength</td><td>Success rate</td><td>Execution (s)</td></tr><tr><td>Nominal</td><td>0</td><td>5/5 (100%)</td><td>46.4</td></tr><tr><td>Perturbed (low)</td><td>0.02</td><td>3/5 (60%)</td><td>44.1</td></tr><tr><td>Perturbed (high)</td><td>0.04</td><td>1/5 (20%)</td><td>60.1</td></tr></table>

## D. Memory Co-Updating Case Study

In this experiment, a node ID is specified, while an ambiguous edge requires the other endpoint. Across five Diningto-Meeting-Room deployments (Fig. 5(b)), a user reports a cart at the office entrance. In each trial, the robot travels to that node, observes that the doorway is clear, records the entrance-node condition, and reaches the meeting room. The report triggers verification rather than being treated as confirmed occupancy. This node-level observation does not certify an unobserved edge. These five deployments illustrate how a reported change is verified before it updates memory.

## V. CONCLUSION

This paper proposes AVALON, a topology-first indoor navigation framework that decouples task execution from rigid global map alignment. A floor-plan-derived topological skeleton paired with editable task memory enables hierarchical mission execution. Topological routing outputs robotframe point goals for a local navigation policy, and verified node arrivals update translational alignment without requir ing yaw correction. For object-search tasks, terminal perceptual validation confirms target presence in contrast to treating node arrival as task completion. Verified robot observations feed into reference-memory updates and enable auditable memory refinement. On a Unitree Go2 first-mission deployment (12.81 m Intersection-to-Dining, workspace trunk 24.11 m, five trials), AVALON with X-NavDP succeeds in all five runs in 46.4 s without robot mapping, 40.9% less robot time than DimOS-Go2 (45.2 s mapping + 33.3 s mean execution, $4 / 5$ success). Policy-only X-NavDP fails in all five trials under a 120 s timeout, pacing near the start without a long-horizon route. DimOS’s unsuccessful trial drifted in odometry into a dead end. In the five AVALON + X-NavDP Dining-to-Meeting-Room trials reported in Table II, user reports of environmental changes trigger on-site robot verification before the entrance node condition is updated. AVALON successfully completes the report–verification– update sequence and subsequently reaches the meeting room in all five trials. These demonstrate the superiority of the proposed topological navigation framework.

## REFERENCES

[1] D. S. Chaplot, D. Gandhi, S. Gupta, A. Gupta, and R. Salakhutdinov, “NTS: Neural topological SLAM for visual navigation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 12 875–12 884, arXiv:2005.12256.

[2] N. Kim, O. Kwon, H. Yoo, Y. Choi, J. Park, and S. Oh, “Topological semantic graph memory for image-goal navigation,” in Proc. Conf. Robot Learn. (CoRL), ser. Proc. Mach. Learn. Res., vol. 205. PMLR, 2023, pp. 393–402.

[3] P. Chen, G. Zhang, J. Ma, and Q. Li, “Hypothesis graph refinement: Hypothesis-driven exploration with cascade error correction for embodied navigation,” arXiv preprint arXiv:2604.04108, 2026.

[4] M. A. Vega Torres, A. Braun, and A. Borrmann, “BIM-SLAM: Integrating BIM models in multi-session SLAM for lifelong mapping using 3D LiDAR,” in Proc. Int. Symp. Autom. Robot. Construct. (ISARC), 2023, arXiv:2408.15870.

[5] Dimensional OS, “DimOS: The agentive operating system for physical space,” ver. 0.0.13.post1, GitHub, Jul. 2026, [Online]. Available: https: //github.com/dimensionalOS/dimos.

[6] H. Howard-Jenkins, J.-R. Ruiz-Sarmiento, and V. A. Prisacariu, “LaLaLoc: Latent layout localisation in dynamic, unvisited environments,” arXiv preprint arXiv:2104.09169, 2021.

[7] H. Howard-Jenkins and V. A. Prisacariu, “LaLaLoc++: Global floor plan comprehension for layout localisation in unvisited environments,” in Proc. Eur. Conf. Comput. Vis. (ECCV), ser. LNCS, vol. 13687, 2022, pp. 693–709.

[8] J. Li, W. Huang, Z. Wang, W. Liang, H. Di, and F. Liu, “FloNa: Floor plan guided embodied visual navigation,” arXiv preprint arXiv:2412.18335, 2025.

[9] M. Luperto, D. Fusi, N. A. Borghese, and F. Amigoni, “Robot exploration using knowledge of inaccurate floor plans,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2019.

[10] N. Zimmerman, J. Loo, A. Agrawal, and D. Hsu, “SignLoc: Robust localization using navigation signs and public maps,” arXiv preprint arXiv:2508.18606, 2025.

[11] N. Zimmerman, J. Loo, B. Koh, Z. Wang, and D. Hsu, “Sign-Scene: Visual sign grounding for mapless navigation,” arXiv preprint arXiv:2602.12686, 2026.

[12] A. H. Tan, A. Fung, H. Wang, and G. Nejat, “Mobile robot navigation using hand-drawn maps: A vision language model approach,” arXiv preprint arXiv:2502.00114, 2025.

[13] J. Zhang, S. Wu, X. Ma, and S. Schwertfeger, “Generation of indoor open street maps for robot navigation from CAD files,” arXiv preprint arXiv:2507.00552, 2025.

[14] Y. Zhang, J. Zhang, C. Li, F. Xie, and S. Schwertfeger, “osmAG-Nav: A hierarchical semantic topometric navigation stack for robust lifelong indoor autonomy,” arXiv preprint arXiv:2603.28271, 2026.

[15] M. Labbe and F. Michaud, “RTAB-Map as an open-source lidar and´ visual SLAM library for large-scale and long-term online operation,” Journal of Field Robotics, vol. 36, no. 8, pp. 1418–1438, 2019.

[16] H. Niu, L. Zhang, X. Ji, R. Ying, P. Liu, and F. Wen, “SSMG-Nav: Enhancing lifelong object navigation with semantic skeleton memory graph,” arXiv preprint arXiv:2603.01813, 2026.

[17] B. Wang, J. Lin, C. Liu, X. Hu, Y. Yu, T. Liu, Z. Wang, and X. Qi, “MG-Nav: Dual-scale visual navigation via sparse spatial memory,” arXiv preprint arXiv:2511.22609, 2025.

[18] J. Jiao, C. Liu, J. Yu, B. Liu, Q. Zhang, Y. Wang, and D. Kanoulas, “OPENNAVMAP: Multi-session appearance-based topometric mapping for scalable visual navigation,” arXiv preprint arXiv:2601.12291, 2026.

[19] D. Shah, A. Sridhar, A. Bhorkar, N. Hirose, and S. Levine, “GNM: A general navigation model to drive any robot,” arXiv preprint arXiv:2210.03370, 2023.

[20] D. Shah, A. Sridhar, N. Dashora, K. Stachowicz, K. Black, N. Hirose, and S. Levine, “ViNT: A foundation model for visual navigation,” arXiv preprint arXiv:2310.07896, 2023.

[21] A. Sridhar, D. Shah, C. Glossop, and S. Levine, “NoMaD: Goal masked diffusion policies for navigation and exploration,” arXiv preprint arXiv:2306.14846, 2023.

[22] W. Cai, J. Peng, Y. Yang, Y. Zhang, M. Wei, H. Wang, Y. Chen, T. Wang, and J. Pang, “NavDP: Learning sim-to-real navigation diffusion policy with privileged information guidance,” arXiv preprint arXiv:2505.08712, 2025.

[23] T. Yang, Y. Zeng, W. Cai, Y. Yang, J. Peng, H. Cheng, J. Pang, and T. Wang, “X-NavDP: Generalizing navigation diffusion policy to novel behavior and embodiments with group q-score reweighted matching,” arXiv preprint arXiv:2607.28560, 2026.

[24] N. Yokoyama, S. Ha, D. Batra, J. Wang, and B. Bucher, “VLFM: Vision-language frontier maps for zero-shot semantic navigation,” arXiv preprint arXiv:2312.03275, 2024.

[25] J. Zhang, K. Wang, R. Xu, G. Zhou, Y. Hong, X. Fang, Q. Wu, Z. Zhang, and H. Wang, “NaVid: Video-based VLM plans the next step for vision-and-language navigation,” arXiv preprint arXiv:2402.15852, 2024.

[26] H. Yin, X. Xu, L. Zhao, Z. Wang, J. Zhou, and J. Lu, “UniGoal: Towards universal zero-shot goal-oriented navigation,” arXiv preprint arXiv:2503.10630, 2025.

[27] F. Xie, S. Schwertfeger, and H. Blum, “osmAG-LLM: Zero-shot openvocabulary object navigation via semantic maps and large language models reasoning,” IEEE Robotics and Automation Letters, 2026, accepted manuscript, arXiv:2507.12753v2.

[28] D. Ou, J. J. Hagenaars, M. R. Jankowski, M. V. M. Firlefyn, C. De Wagter, F. T. Muijres, J. Degen, and G. C. H. E. de Croon, “Efficient robot navigation inspired by honeybee learning flights,” Nature, 2026.

[29] L. Ribeiro, M. Shaheer, M. Fernandez-Cortizas, A. Tourani, H. Voos, and J. L. Sanchez-Lopez, “Human interaction for collaborative semantic SLAM using extended reality,” arXiv preprint arXiv:2509.14949, 2025.

[30] P. E. Black, “Bresenham’s algorithm,” Dictionary of Algorithms and Data Structures, National Institute of Standards and Technology, Dec. 2020, [Online]. Available: https://xlinux.nist.gov/dads/HTML/ bresenham.html.

[31] A. Turner, M. Doxa, D. O’Sullivan, and A. Penn, “From isovists to visibility graphs: A method for the analysis of architectural space,” Environment and Planning B: Planning and Design, vol. 28, no. 1, pp. 103–121, 2001.

[32] A. A. Agha-mohammadi, S. Chakravorty, and N. M. Amato, “FIRM: Sampling-based feedback motion-planning under motion uncertainty and imperfect measurements,” The International Journal of Robotics Research, vol. 33, no. 2, pp. 268–304, Feb. 2014.

[33] A. Majumdar and R. Tedrake, “Funnel libraries for real-time robust feedback motion planning,” The International Journal of Robotics Research, 2017.

[34] T. Cheng, L. Song, Y. Ge, W. Liu, X. Wang, and Y. Shan, “YOLO-World: Real-time open-vocabulary object detection,” arXiv preprint arXiv:2401.17270, 2024.

[35] Qwen Team, “Qwen3.5-Omni technical report,” arXiv preprint arXiv:2604.15804, 2026.