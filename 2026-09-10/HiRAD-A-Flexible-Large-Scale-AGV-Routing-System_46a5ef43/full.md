# HiRAD: A Flexible Large-Scale AGV Routing System

Yunjie Huang

Ruizhong Wu

Mengxuan Zhang

Frodo Kin Sun Chan

Yan Nei Law

Lei Li

yhuang863@connect.hkust-gz.edu.cn

## Ab<sub>s</sub>t<sub>rac</sub>t

Automatic Guided Vehicles (AGVs)substantially boost warehouse throughput, but routing large-scale AGV fleets remains challenging. Classical Multi-Agent Pathfinding solvers sufer from exploding combinatorial complexity and super-quadratic runtime, while relying on idealized grid or piecewise-linear motion models that mismatch real-world kinematics. Recent Reinforcement Learning (RL) solutions improve flexibility via decentralized agent policies but depend on discretized spatiotemporal representations, require millions of episodes to converge, and incur full-map observation at every step, which leads to large models, slow convergence, and high inference latency that violates real-time industrial control constraints. To address these bottlenecks, we propose HiRAD, a hierarchical RL framework for continuous-space AGV routing with real-time guarantees, with 1) a step-level spatiotemporal representation that translates continuous motion into a diferentiable RL problem, 2) a hierarchical strategy that splits heading choice from velocity control to reduce the action space, and 3) an asynchronous event-driven decision pipeline further lowers inference complexity from �(�<sup>2</sup>) to �(�) and cuts per-step latency by as much as 71 %. Across random graphs and two warehouse maps, HiRAD reduces makespan by 45 %–63 % and shortens end-to-end runtime.

## Ke<sub>y</sub>words

Multi-agent, Pathfinding, Reinforcement learning, AGV, Routing

## 1 I<sub>n</sub>t<sub>ro</sub>d<sub>uc</sub>ti<sub>on</sub>

The rapid growth of e-commerce and AI-driven logistics has propelled Automated Guided Vehicles (AGV) to a critical role in warehouse operations, which transports items from storage to packaging zones to meet large-scale order demands [5, 13, 52]. A large AGV fleet can significantly cut labor costs [10], especially during peak shopping events when billions of orders are processed daily.

Despite widespread deployment, AGV systems face fundamental limitations [6, 28, 49, 50], which must meet four base requirements: 1) Collision-free operation as the safety baseline; 2) Flexibility to handle ad-hoc failures and dynamic adjustments to tasks, layouts, or vehicles; 3) Kinematic Constraints to adhere to physical movement laws in continuous environments; and 4) Eficiency to complete tasks as quickly as possible. However, achieving both flexibility and kinematic constraints simultaneously remains a challenge.

To improve flexibility, routing algorithms must adapt quickly to environmental changes, but kinematic constraints complicate computations, often leading to problem simplifications in discrete environments: space is divided into grids, and AGVs move instantly between grids at discrete time intervals, sacrificing realism. This can even violate collision-free guarantees; for instance, Figure 2 illustrates a collision scenario caused by discrete assumptions that fail to capture real-world motion.

![](images/d13d099dd0ad7fbf8620f688c0ccb708a62aeb47931f59b2fc0a28f77203b567.jpg)  
(a)

![](images/7310ad3b7edacb3744a4dbd8de270cab6d78a3e72164a4ca0e29dec98fcc04e1.jpg)  
(b)

![](images/9962fb398b5d4ed171ab742f77dc114daf5766badf47739dc9aca2cc31effcf3.jpg)  
(c)  
Figure 1: (a) Snapshot of an AGV Operating Scene ; (b) Physical AGVs; (c) Warehouse Ma<sub>p</sub> (Green Triangles: Picku<sub>p</sub> locations, Blue Triangles: Dropof locations).

Conceptually, AGV routing is an NP-Hard Multi-Agent Path Finding (MAPF) problem [53]. Early robotic solutions planned paths for all AGVs in advance to avoid collisions [6, 28], but they were computationally slow and impractical for scaling. Faster pruningbased algorithms [43, 44], like OHSMD [49], introduced continuous kinematic constraints but lacked flexibility, as deviations in AGV paths required system-wide re-planning. To address flexibility, reinforcement learning (RL) trains AGVs to independently find paths [12, 33, 35]. However, RL methods scale poorly, requiring days to train for large fleets and sacrificing kinematic constraints. PRIMAL [41] is the only RL-based solution to support over 1,000 AGVs, but it operates in discrete environments and treats collisions as penalties rather than strict constraints, limiting its practicality.

In summary, achieving both flexibility and kinematic constraints remains a key challenge for improving AGV eficiency. Since fullpath planning conflicts with flexibility, we adopt the RL-based paradigm for adaptability, extend it to satisfy kinematic constraints eficiently using data management techniques, and propose the Hig<sup>h</sup>- R<sub>eso</sub>l<sub>u</sub>ti<sub>o</sub>n <sub>a</sub>nd Hi<sub>e</sub>r<sub>a</sub>r<sub>c</sub>hi<sub>ca</sub>l R<sub>ou</sub>tin<sub>g</sub> <sub>w</sub>ith A<sub>sy</sub>n<sub>c</sub>hr<sub>o</sub>n<sub>ous</sub> Decision Framewor<sup>k</sup> (HiRAD) with the following componentes.

Firstly, RL’s step-based decision-making conflicts with continuous environments. To address this, we propose High-Resolution Spatiotemporal Modeling, which uses finer-grained discrete space and time aligned with AGV kinematics, efectively simulating a

![](images/8e90da5385781f49534a0eba35537f4edf7cb84a0d0936ba5db22c34ee7d62cc.jpg)  
Fi<sub>gu</sub>r<sub>e</sub> 2: AGV C<sub>o</sub>nfli<sub>c</sub>t<sub>s</sub> in Di<sub>sc</sub>r<sub>e</sub>t<sub>e a</sub>nd C<sub>o</sub>ntin<sub>uous</sub> M<sub>ov</sub>in<sub>g</sub> continuous environment. Grids are divided into smaller cells, time into finer intervals, and AGV movements into intermediate velocity states, allowing smooth ac-/deceleration to mimic real-world dynamics. However, it inevitably enlarges the search space and decision-making time, while the higher number of steps demands faster decisions, making real-time operation even more challenging.

Secondly, to reduce the action search space, we propose Alternating Decision-Velocity Control $( A l t - D V C ) ,$ , which first decouples direction and velocity decisions and reduces their transition sizes. The Macro-Routing Layer operates on a coarse grid map using RLbased decision-making to generate global paths, while the Micro-Velocity Control Layer runs on a fine-grained map with predefined velocity modes, reducing action frequency and computational cost by dynamically switching between modes. Additionally, Alt-DVC becomes the first RL-based method to avoid collisions. Static constraints treat AGVs as dynamic obstacles to prevent overlapping positions, and priority scheduling resolves path conflicts by assigning right-of-way, reducing congestion and wait times. This approach minimizes collision risks, equipment wear, and task interruptions, significantly improving operational eficiency and safety.

Finally, the above constraints and priority handling make decisionmaking time larger the AGV’s physical execution time, causing loss of control in practice. This arises from the high complexity of existing observation and priority scheduling methods. To address this, we propose Asynchronous Decision-Velocity Control (Asy-DVC) optimized with Observation Pruning and Map-Oriented Priority to reduce complexity from quadratic to linear. This ensures decisionmaking remains faster than execution time, enabling eficient and seamless AGV control in real-world warehouse environments.

Our contributions are summarized below:

• We propose the HiRAD to enable a large-scale AGVs collision-free routing system to work in kinematic constraint environments flexibly and eficiently;

• We propose a high-resolution spatiotemporal model that adapts the step-based RL framework to the continuous environment;

• We propose a hierarchical Alternating Decision-Velocity Control to decouple the direction and speed decision-making to reduce the search space and avoid collisions;

• We propose Asynchronous Decision-Velocity Control with observation pruning and map-oriented priority to further reduce the decision-making complexity for real-life use;

• Evaluations with extensive experiments show that our approach outperforms the state-of-the-art solutions.

## 2 R<sub>e</sub>l<sub>a</sub>t<sub>e</sub>d W<sub>or</sub>k

## 2.1 Sin<sub>g</sub>le-A<sub>g</sub>ent Pathfindin<sub>g</sub>

These methods focus on single AGV path planning and difer in environmental information accessibility [32, 54]. In local pathfinding, the environment is partially known via LiDAR or visual sensors, resembling general robotics but not AGVs. Global pathfinding, where the full environment is known, employs models like Voronoi diagrams [3, 20, 47, 48], probabilistic maps [19], and geometric methods [30]. Sampling-based algorithms like RRT [26] and RRT\* [23] incrementally build search spaces. For AGV warehouses, environments are modeled as grids [22] or topological graphs [8], enabling graph algorithms like Dijkstra’s [14] and $A ^ { * }$ [21], and their dynamic variants �<sup>∗</sup> [46], Lifelong �<sup>∗</sup> [25], and �<sup>∗</sup>-lite [24], to adjust for new obstacles, but they struggle with multi-agent, collision-free, optimal pathfinding due to slow computation. Path indexes like CH [18, 39] and HL [1, 9, 38, 55–57] are fast but cannot handle collisions.

## 2.2 Multi-A<sub>g</sub>ent Pathfindin<sub>g</sub> MAPF

Multi-agent scenarios aim to coordinate AGV fleets eficiently without conflicts. A simple approach is to use single-agent algorithms, detect conflicts, and re-route dynamically [2, 11, 40], but re-routing is time-consuming and reduces eficiency. Classical optimization techniques like Ant Colony [31], Particle Swarm [37], and Genetic Algorithms [45] are unsuitable for real-time MAPF due to slowness and batch processing. CBS [6, 11, 27] and ODrM\* [16] improve multi-agent search eficiency but require simultaneous planning for all agents, limiting online application. Cooperative A<sup>∗</sup> [4] plans paths individually but is restricted to discrete environments and lacks support for kinematics or lifelong learning. OHSMD [50, 51], derived from time-dependent routing [29], operates in continuous environments with kinematic constraints and lifelong scheduling, but its centralized system requires global re-planning after incidents, making it inflexible and impractical.

## 2<sub>.</sub>3 RL<sub>-</sub>b<sub>ase</sub>d P<sub>a</sub>thfi<sub>n</sub>di<sub>ng</sub>

Recent advancements have integrated deep reinforcement learning (DRL) and graph neural networks (GNNs) into pathfinding, ofering greater flexibility by identifying optimal actions based on the current environment and single-agent training. CADRL [7] replaced traditional online computations with ofline-trained value networks for decentralized, non-communicating multi-robot settings. Ding [15] introduced a hierarchical RL framework using LiDAR data, with high-level risk assessment and low-level action decisions, while [34] addressed multi-robot collision avoidance using 512-dimensional LiDAR data, target positions, and robot velocities as inputs, with continuous translational and rotational velocities as outputs. However, these studies were limited to AGV fleets of 32 or fewer, where classical algorithms remain efective. The PRIMAL framework [41] scaled to 1024 agents, and occasionally 2048, marking a milestone in large-scale MAPF with RL-based approaches. Despite this success, PRIMAL operates in discrete settings, limiting its ability to simulate real-world AGV motion dynamics.

## 3 P<sub>ro</sub>bl<sub>em</sub> F<sub>ormu</sub>l<sub>a</sub>ti<sub>on</sub>

In an AGV network, the environment is modeled as a grid map � consisting of� × � square grids, where each AGV occupies a single grid at any given time. AGVs move between grids to complete tasks ${ \tau _ { i } } = \langle s _ { i } , d _ { i } \rangle$ , where $s _ { i }$ and $d _ { i }$ are the start and destination positions. AGVs can perform five actions: moving to adjacent grids or staying in place $( a ^ { t } = \{ \uparrow , \right. , \downarrow , \left. , \odot \} )$ . To account for kinematic constraints,

![](images/640076241d0af40957e065247a4ea789647a789a140e98577409e693eb86039c.jpg)

Fi<sub>g</sub>ure 3: Hi<sub>g</sub>h-Resolution S<sub>p</sub>atiotem<sub>p</sub>oral Modelin<sub>g</sub> we additionally associate each time step with a scalar velocity $\boldsymbol { v } _ { i } ^ { t }$ (under a high-resolution modeling). Thus, the objective of our Continuous Reinforcement Learning Multi-Agent Pathfinding (C-RL-MAPF) problem is defined to compute collision-free paths $\mathcal { P } =$ $\{ p _ { i } = \langle ( \boldsymbol { x } _ { i } ^ { t } , \overline { { \boldsymbol { y } } } _ { i } ^ { t } , \boldsymbol { v } _ { i } ^ { t } ) \rangle \} _ { i = 1 } ^ { N }$ for all � AGVs that minimize the makespan (maximum path length) and flowtime (total path length). $\langle ( x _ { i } ^ { t } , y _ { i } ^ { t } ) \rangle$ is the location of the �-th AGV at time step � with the velocity $\boldsymbol { v } _ { i } ^ { t } .$ The discrete action specifies the intended neighboring cell, while $\boldsymbol { v } _ { i } ^ { t }$ determines how this intent is realized over Δ�.

## 4 Hi<sub>g</sub>h-Resolution S<sub>p</sub>atiotem<sub>p</sub>oral Modelin<sub>g</sub>

To balance the trade-of between the coarse-grained discrete modeling and the computationally expensive on continuous modeling, we propose the High-Resolution Spatiotemporal (HRS) modeling. HRS integrates a fine-grained spatial map (HR-Map) with a finer temporal decision-making step (HR-Step), enabling discrete modeling to approximate continuous environments eficiently.

High-Resolution Map and Step. The definitions are described as:

Definition 1 (High-Resolution HR-Map). Given a grid network �, we increase the spatial resolution by dividing each grid into $c \times c$ smaller cells (as Fig. 3), resulting in an HR-Map $M _ { c } \in$ $\mathbb { R } ^ { ( m c ) \times ( k c ) }$ , where $c > 1 .$ . The original grid map $M = M _ { c = 1 }$ is referred to as a Coarse Map.

Definition 2 (High-Resolution HR-Step). Given a decision interval � in a coarse map, we divide it into $\begin{array} { r } { \Delta t { } = { } \frac { \delta } { c } } \end{array}$ HR-Steps such that AGVs make decisions at higher temporal resolution.

In the HR-Map, AGVs can move between cells at varying speeds, which correspond to consecutive cell movements in HR-Steps: acceleration, constant velocity, or deceleration. This discrete yet finegrained modeling is equivalent to continuous environments while maintaining computational eficiency.

HRS RL-MAPF Modeling. HRS RL-MAPF extends discrete RL-MAPF by incorporating velocity into the action space $a ^ { t } = ( \mathbf { d } a ^ { t } , \boldsymbol { v } ^ { t } )$ where d�<sup>�</sup> represents direction choice in the classical MAPF problem and $v ^ { t }$ represents the number of cells moved per HR-Step. The velocity can increase, remain constant, or decrease between steps, reflecting acceleration and deceleration.

For a C-RL-MAPF where $c  \infty ,$ the complexity for an RLbased method would increase from (� $\mathbf { \nabla } \times k \mathbf { \bar { \nabla } } \times \mathbf { \bar { \ p d a } } | ) ^ { \vert \mathcal { N } \vert }$ to (�� × <sub>�</sub>� $\times a \times | v | ) ^ { | N | }$ as the Expanded range of velocity $v ^ { t }$ and action directions adds complexity to the policy optimization. The reward function $\mathcal { R } ( s ^ { t } , a ^ { t } )$ also trapped in reducing positional error � = $\lVert ( x ^ { t } , y ^ { t } ) - ( x ^ { * } , y ^ { * } ) \rVert$ to ensure AGVs stop precisely at grid centers.

Therefore, HRS modeling could achieve continuous, large-scale AGV routing while controlling complexity: instead of letting � → ∞ and freely expanding $v ^ { t }$ , HRS discretizes motion into a small set of kinematically feasible velocity modes and uses HR-step-based decisions to limit the efective action/velocity dimensions. In this way, HRS preserves continuous feasibility and coordination, while reducing both the spatiotemporal state growth and the joint action explosion compared with a naive continuous formulation.

## 5 HiRAD AGV R<sub>ou</sub>tin<sub>g</sub> Fr<sub>a</sub>m<sub>ewo</sub>rk

Figure 4 illustrates our HiRAD framework, which decouples AGV navigation into direction and velocity decisions. At the macro level, we propose a Macro RL-Routing Layer (orange) that plans collisionfree moving directions toward the goal using coarse-map information, where obstacles include both static structures and other AGVs treated as dynamic obstacles. At the micro level, we introduce a Micro Velocity-Control Layer (green) to execute safe, fine-grained motions on the HR-Map. These two layers are integrated by the Alternating Decision–Velocity Control (Alt-DVC) module (yellow) to drive the AGV, while the Asynchronous Decision–Velocity Control (Asy-DVC) module (blue) further reduces decision frequency to improve inference eficiency. We next detail each component.

## 5.1 Macro-RL-Routin<sub>g</sub> La<sub>y</sub>er

Macro-RL-Routing performs coarse-level navigation by outputting collision-free moving directions on the coarse grid. To avoid expensive multi-AGV training, we train a single-agent policy on randomly generated maps $M _ { c = 1 }$ and reuse it as the macro routing module. At each step, the policy receives the partial observation state $z ^ { t } \approx s ^ { t }$ which includes a local window $\hat { W ^ { t } } \in \mathbb { R } ^ { b \times b }$ and a guidance vector $\vec { u } ^ { t } \in \mathbb R ^ { 3 }$ , where � is the number of grids. Both of them are encoded and concatenated as the observation representation �, then fed into an LSTM to model temporal context. At last of module, a projection head produces the action distribution $\pi ( \boldsymbol { a } ^ { t } \mid \boldsymbol { s } ^ { t } )$ (direction decision) and the state value $V ( s ^ { t } )$ (critic signal). Training adopts curriculum learning (from small obstacle-free maps to large dense ones) with step/collision penalties and a dificulty-scaled terminal goal reward to encourage long-horizon routing. The specific training process and reward settings are displayed in Appendix C.1.

We adopt the Macro-layer, which embeds the well-trained routing policy $\pi ^ { * }$ to generate the direction actions for all AGVs. Both the view window and the guidance vector are extended from the single-AGV form to accommodate all AGVs. The observation becomes $W \in \mathbb { R } ^ { | N | \times b \times b }$ , and the guidance vector becomes $\vec { u } \in \mathbb { R } ^ { | N | \times 3 }$ , enabling simultaneous decision-making for the entire fleet. As AGVs may traverse multiple fine-grained cells across grid boundaries under the HR-map, we must treat AGVs as dynamic obstacles occupying spatial regions. To handle this, we define AGV-specific exclusion zones as follows:

Definition 3 (Exclusion Zone). The Exclusion Zone for a specific AGV � is denoted as $\mathcal { E } _ { i } \subseteq M _ { c = 1 }$ ensures safe and eficient navigation, which consists offive grids: the two grids ahead in the AGV’s moving direction (active zone) and the grids to its left, right, and rear (passive zones).

The exclusion zone for each AGV consists of five grids that indicate its occupied and potentially occupied regions. These zones

![](images/4b86e72ed38606a12f6612bf16a2653d8d249d75cfd12f3006d726d79ab30000.jpg)  
Fi<sub>gure</sub> 4<sub>:</sub> HiRAD<sub>:</sub> Hi<sub>g</sub>h<sub>-</sub>R<sub>eso</sub>l<sub>u</sub>ti<sub>on an</sub>d Hi<sub>erarc</sub>hi<sub>ca</sub>l R<sub>ou</sub>ti<sub>ng w</sub>ith A<sub>sync</sub>h<sub>ronous</sub> D<sub>ec</sub>i<sub>s</sub>i<sub>on</sub> F<sub>ramewor</sub>k<sub>.</sub>

![](images/753f8171156dcee0af36cf8fda55c1e642e2fce1aad799ebfb3cf48eef84baa2.jpg)  
Fi<sub>g</sub>ure 5: Velocit<sub>y</sub> Ma<sub>pp</sub>in<sub>g</sub>. The time units in <sub>y</sub>ellow circles re<sub>p</sub>resent the HR-Ste<sub>p</sub>s durin<sub>g</sub> which the Macro-RL-R<sub>ou</sub>ti<sub>ng ma</sub>k<sub>es</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>ons.</sub> Th<sub>e</sub> ti<sub>me un</sub>it<sub>s</sub> i<sub>n re</sub>d<sub>, cen</sub>t<sub>ere</sub>d i<sub>n</sub> circles<sub>,</sub> re<sub>p</sub>resent the HR-Ste<sub>p</sub>s to execute the Micro-Velocit<sub>y</sub>- Control decisions for switchin<sub>g</sub> velocit<sub>y</sub> modes. are used to proactively warn other AGVs of intended movements, enabling complete collision avoidance at the decision level.

## 5.2 Micro Velocit<sub>y</sub>-Control La<sub>y</sub>er

Micro Velocity-Control is a lightweight module that adjusts the AGV travel velocity online according to local motion requirements recommended by the Macro-Routing layer. We explain the finegrained control on the HR-map with resolution factor � = 10, where velocity is mapped to the distance traveled within the minimum control interval Δ� (e.g., � = 0.2 means moving 0.2 grids, i.e., 2 HRcells, per Δ�). To satisfy kinematic constraints and enable precise stopping at grid centers, we predefine several velocity modes that follow a three-stage profile Acceleration–(Max)Speed–Deceleration, covering typical cases such as moving to the next grid (Mvx), continuous cruising (Mvv), decelerating to stop (Mvz), and move to the second grid (Mxx) as Figure 5.

Here, velocity is converted to the distance an AGV can travel within the smallest time unit Δ�. For example, a velocity of 0.2 means that the AGV moves 0.2 grids (i.e., 2 HR-cells) per Δ�, whereas a velocity of 0 means staying at the grid center. The specific velocity distributions and behavioral characteristics of the three velocity modes are detailed as follows:

(1) Move to Next Grid (Mvx): This mode applies when the AGV needs to move only one grid. The velocity pattern is [0, 0.1, 0.2, 0.2, 0.2, 0.2, 0.1, 0], meaning the AGV slightly accelerates, then moves at a constant speed, and finally decelerates to a stop. It takes 7 Δ� to achieve the discrete movement;

(2) Continuous Motion (Mvv): This mode is used when there is no obstacle ahead, allowing the AGV to accelerate to its maximum velocity and maintain high-velocity movement. The velocity pattern follows [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5, ...], thereby improving operational eficiency. At the maximum velocity, it takes 2 Δ� to traverse a grid. In addition, it is guaranteed that at every HR-Step Δ� at 0.5 speed, the AGV is at the grid center or boundary;

(3) Move to zero (Mvz): This mode is applied when the AGV needs to stop at the next grid after moving at maximum speed. The velocity pattern follows [..., 0.5, 0.5, 0.5, 0.5, 0.4, 0.3, 0.2, 0.1, 0], where the AGV maintains its peak speed initially and then decelerates uniformly to a full stop. The first velocity decision is made at time step � + 0, where the AGV chooses to maintain its maximum velocity. Upon reaching the center of the next grid at � + 3, it initiates deceleration, reducing speed at a constant rate to ensure it comes to still exactly at the grid center.

(4) Move to Second Grid (Mxx): This mode applies when the AGV needs to stop at the second grid ahead. The velocity pattern is [0, 0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1, 0], allowing for greater acceleration followed by a smooth deceleration to ensure stable movement and precise stopping. The AGV cannot reach its maximum speed in this mode.

Speed modes guaranteed real-time applicability conditions for HR-step-based decision making:

Lemma 1 (RL Real-Time Applicability Conditions). Given an HRS-Modeling and its corresponding velocity mapping, the AGVs can run continuously under kinematic constraints in real life only when the following conditions are satisfied:

(1) The AGV stays at the grid center when it stops;

(2) The velocity mode satisfies kinematic constraints of three stages Acceleration-(Max)Speed-Deceleration;

(3) The decision time Δ� is shorter than the moving time unit.

With Lemma 1, our HRS modeling and the HR-step-based control admit continuous execution under kinematic constraints:

Theorem 1 (HRS-Continuous Eqivalence). The proposed HRS modeling, combined with HR-step-based decision-making, enables the AGV to run continuously under kinematic constraints.

The proof is described in Appendix C.2. Finally, while we use $c = 1 0$ as an illustrative setting, the specific resolution and mode parameters can be calibrated to diferent physical systems; the conditions in Lemma 1 remain unchanged.

## 5<sub>.</sub>3 Alt<sub>e</sub>rn<sub>a</sub>tin<sub>g</sub> D<sub>ec</sub>i<sub>s</sub>i<sub>o</sub>n–V<sub>e</sub>l<sub>oc</sub>it<sub>y</sub> C<sub>o</sub>ntr<sub>o</sub>l (Alt-DVC)

Alt-DVC is the online synchronous execution mechanism that alternates macro direction planning and micro velocity adjustment at every HR-step Δ�. At each cycle, the Macro-RL-Routing layer outputs a desired discrete direction $a ^ { t } = \pi ^ { * } ( a \mid s ^ { t } )$ , and the Micro Velocity-Control layer maps it into feasible continuous motion on the HR-map $M _ { c = 1 0 }$ under kinematic constraints. Unlike standard RL-MAPF (where agents move one grid per step and efectively “stop” at each cell), Alt-DVC must additionally decide (i) conflict resolution among moving AGVs with diferent velocities, and (ii) when to decelerate so that an AGV can stop at an appropriate grid center for reorientation. To prevent collisions, Alt-DVC detects potential collisions by checking whether an $\operatorname { A G V } { \boldsymbol { s } }$ active zone intersects any other AGV’s exclusive zone (Definition 3); such conflicts are then resolved online via priority scheduling, where higher-priority AGVs may proceed while lower-priority ones treat the higher-priority exclusive zones as obstacles and stop before entering. Priorities are ranked primarily by velocity and then refined by local obstaclebased tie-breaking. In the following, we first define collision, then present how to schedule the priority of AGVs to avoid collisions, and finally present how to make the decision on deceleration.

Definition 4 (Potential Collisions Condition). Given any AGV � with position $l _ { i } ^ { t } = ( x _ { i } ^ { t } , y _ { i } ^ { t } )$ at timestep �, and each AGV keeping an exclusive zone $\varepsilon _ { i }$ (as defined in Definition 3), a potential collision is identified occurs when $A G V _ { i } { ' } ;$ s active zone intersects with the exclusive zone of any other $A G V _ { j }$ where $j \neq i .$

It should be noted that the above condition only detects the potential collision, which is subject to being resolved by the following priority scheduling.

AGV Scheduling Priority. We allow the AGV with higher pri ority to enter the Exclusion Zones of the lower-priority AGVs. Then, AGVs with lower priority must treat the Exclusion Zone of a higher-priority AGV as obstacles, stopping before entering it. This mechanism always allows the higher AGV to keep running, so it can achieve smooth AGV movements without collisions. The AGV priority is determined based on the following criteria:

(a) Velocity-Based Priority: Collisions at higher velocities can lead to severe property damage, and it takes a longer time and space for a faster AGV to stop, so the priority is first ranked by velocities;

(b) Obstacle-Awareness Priority: An AGV with fewer obstacles within its observation window � has greater maneuverability. For example, suppose a set of AGVs is clustered together; then the AGVs on the outline of the cluster should move away to bring more space for the inner AGVs. Thus, the AGVs with fewer obstacles have higher priority;

(c) Local-Flexibility Priority: When there is still a tie in the priority, then we shrink the view window incrementally to compare the number of the nearby obstacles until one has fewer obstacles.

Additionally, Alt-DVC triggers deceleration in two situations: passive deceleration when required by priority-based collision avoidance, and active deceleration when continuing forward would increase the distance to the goal, using a constant-time heuristic comparing the goal distances of the next and second-next grids (�<sub>�</sub> vs. $d _ { x x } )$ , so that the AGV stops at the next grid and allows the macro layer to update direction.

Case1 Passive Dece<sup>l</sup>eration: When an AGV has potential collisions (Definition 4) with a lower priority, it has to stop passively to avoid collision.

Case2 Active Dece<sup>l</sup>eration: When the current moving direction does not lead straight to the target location, then the AGV needs to stop and let the RL-routing modify the direction at an appropriate position before continuing toward it. One straightforward way to adjust the deceleration is according to the RL-Routing decision. If the recommended action is not aligned with the current AGV’s heading, then the AGV should start decelerating and turn to the recommended heading direction. However, due to the objective of the RL-based method being shorter, the path length finding without considering stopping and turning takes more time. ${ \mathrm { S o } } ,$ frequent stopping and starting will prolong the running time. Another solution adopted by PRIMAL is utilizing $A ^ { * }$ to compute a path on the clean network as guidance at each step �. But it has a very high complexity (�(ℓ × (�� log �� + 4 · ��)) where ℓ is the path length, �� is the grid number, and 4 · �� is the edge number) and also a large result path space to choose from. In addition, it is against the flexibility motivation of RL. Therefore, we take a distance-based heuristic to achieve this eficiently and efectively. Specifically, we compare the distance between the second grid ahead and the target, denoted as $d _ { x x } ,$ with the distance between the first grid ahead and the target, denoted as $d _ { x } .$ . If $d _ { x x } > d _ { x } ,$ it means that the AGV is moving away from the target and should be assigned a stop flag, requiring it to stop at the next grid before reorienting itself. The complexity of the distance-based heuristic is reduced to constant.

## 6 Inf<sub>e</sub>r<sub>e</sub>n<sub>ce</sub> Efi<sub>c</sub>i<sub>e</sub>n<sub>cy</sub> O<sub>p</sub>timiz<sub>a</sub>ti<sub>o</sub>n

Given � AGVs, Alt-DVC executes macro direction inference and micro velocity/conflict resolution at each HR-step Δ�. In a straightforward synchronous implementation, all AGVs upload local observations $s ^ { t } = ( W ^ { t } , \vec { u } ^ { t } )$ to query the shared policy, and the controller aggregates these local windows to support global priority scheduling, leading to $O ( n b ^ { 2 } )$ data per step (and $O ( n b ^ { 2 } T )$ over � steps) for observation preparation. Moreover, priority-based conflict checking requires each AGV to be tested against higher-priority ones, resulting in $1 + \cdot \cdot \cdot + n = O ( n ^ { 2 } )$ pairwise checks per step $( \mathrm { i . e . , } O ( n ^ { 2 } T )$ over an episode). These costs motivate our inference-eficiency optimization in the following.

## 6<sub>.</sub>1 A<sub>sy</sub>n<sub>c</sub>hr<sub>o</sub>n<sub>ous</sub> D<sub>ec</sub>i<sub>s</sub>i<sub>o</sub>n–V<sub>e</sub>l<sub>oc</sub>it<sub>y</sub> C<sub>o</sub>ntr<sub>o</sub>l (Asy-DVC)

As analyzed above, the synchronous Alt-DVC inference incurs $O ( n b ^ { 2 } T + n ^ { 2 } T )$ time, which may make the decision latency longer than the movement time and thus violate the real-time applicability conditions (Lemma 1). To enable real-time execution, we propose $A s y – D V C ,$ which replaces Alt-DVC with two pruning techniques that reduce both observation and priority-check overhead.

6.1.1 Observation Pruning. The key observation is that, due to kinematic constraints, an AGV with nonzero velocity cannot instantaneously change heading; directional decisions only become actionable when the AGV is able to stop and reorient. Therefore, $\operatorname { A s y - D V C }$ skips policy inference for moving AGVs and only queries the macro policy for the subset of AGVs that are at decision points (typically when $v = 0 )$ . Moving AGVs directly proceed with microlevel velocity evaluation (primarily predicting whether to stop ahead), while only a smaller subset performs the $O ( b ^ { 2 } )$ observation and-inference step. Under our velocity-mode design (Fig. 5), this reduces the observation cost from $O ( n b ^ { 2 } T )$ in Alt-DVC to about $O ( { \textstyle { \frac { 1 } { 4 } } } n b ^ { 2 } T )$ in Asy-DVC (i.e., on average one macro decision every four HR-steps).

6.1.2 Map-Oriented Priority. Alt-DVC resolves conflicts via AGVcentric pairwise checking, yielding $O ( n ^ { 2 } )$ checks per cycle and $O ( n ^ { 2 } T )$ overall. Asy-DVC removes this quadratic factor by using a map-oriented index on the coarse map $M _ { c = 1 } { \mathrm { : } }$ at each cycle, we insert all AGVs into a grid-indexed priority structure (per grid, keep AGVs sorted by priority); when $\mathrm { A G V } _ { i }$ evaluates a candidate motion, it only looks up the relevant destination grid(s) and compares against the top-priority entry, accepting the move if it remains highest-priority locally, otherwise slowing/stopping or deferring to re-planning. This replaces global pairwise comparisons with constant-time grid lookups, reducing priority handling from $O ( n ^ { 2 } T )$ to �(��).

Overa<sup>ll</sup> comp<sup>l</sup>exity. Combining the two prunings, Alt-DVC has observation complexity $O ( n b ^ { 2 } T )$ and priority complexity $O ( n ^ { 2 } T )$ for a total of $O ( n b ^ { 2 } T + n ^ { 2 } T )$ . In contrast, Asy-DVC reduces these to $O \big ( \textstyle { \frac { 1 } { 4 } } n b ^ { 2 } T \big )$ (observation pruning) and $O ( n T )$ (map-oriented priority), yielding an overall complexity of $O \big ( \textstyle { \frac { 1 } { 4 } } n b ^ { 2 } T + n T \big )$ .

## 7 Ex<sub>p</sub>eriment

## 7.1 Ex<sub>p</sub>eriment Setu<sub>p</sub>

7.1.1 Experiment Environment. All experiments are conducted on a Linux server with two Intel Xeon Platinum 8375C 2.9GHz, 500 GB memory, and two GeForce RTX 4090 GPUs. The proposed model and RL-baselines are built with Pytorch 2.1.1 and Python 3.8.19, and ICBS and OHSMD are implemented in C++ with full optimization.

7.1.2 AGV Networks and Experiment Designs. We set the highresolution factor $c = 1 0$ and observation range $b = 1 0$ grids. The RL training hyper-parameters are $\gamma = 0 . 9$ and $\lambda = 0 . 5 .$ . The algorithms are tested on the following three types of networks:

![](images/f10a6c730f8751661a5b0090e1aca0379e3baf2853792b79bc6fa18c590e8817.jpg)  
Fi<sub>g</sub>ure 6: AGV Network La<sub>y</sub>outs

Random Map: We test on three worlds of size {40, 80, 160} with a fleet size of {64, 128, 256, 512, 1024, 2048}. The source and goals are generated randomly for each AGV. For each feasible parameter combination, we conducted 10 trials with randomly initialized maps, running each trial 10 times and averaging the results for comparison. We excluded certain infeasible scenarios: tests were not conducted with 256 or more AGVs in the world 40, nor with 1024 agents in the world 80.

Warehouse Layout-A: This layout comes from real-life warehouses [49], where the pickup location is on the top, and the delivery locations are at the bottom. There is no fixed obstacle as it requires more AGVs to operate together. We test on two diferent network sizes: $G _ { 1 }$ with 16×214 grids, and $G _ { 2 }$ with 32×428 grids. The top and bottom rows are designated for 100 pick-up and drop-of points, with one grid separating the adjacent counterparts. The fleet sizes are {100, 200, 300, 400}. Some cases are infeasible, like more than 200 AGVs on map $G _ { 1 }$ . The tasks are lifelong in the warehouse, so the AGVs continuously travel between pick-up and drop-of locations to complete 15,000 tasks set together.

To ensure unbiased evaluation, all methods were tested under the same map set and task set.

7.1.3 Baselines. We only compare with the baselines that can scale to our experiment setting. For the graph searching baselines, we compare with i) ICBS [6], which is an eficient extension of the classic CBS [42], and ii) ODrM<sup>∗</sup> [17]. For the lifelong tasks, we compare with OHSMD [50]. For the RL-based methods, we compare with the state-of-the-art PRIMAL [41], as it is the only one that can scale to the large fleet. Finally, we use HiRAD-Asy to denote the optimized version with the Asy-DVC, and HiRAD-Alt to denote the original Alt-DVC version.

7.1.4 Metrics. For the discrete methods, we compare with the following metrics: (1) RunTime (RT): The total code running time to complete the path-planning for all agents; (2) MakeSpan (MS (Steps): The number of HR-Steps required for the last agent to reach its goal, from start to finish; (3) FlowPath (FP): The total distance traveled by all agents to reach their respective goals . To ensure a fair comparison, we multiply the step count of the discrete algorithms by the high-resolution step count required to move one grid in HiRAD, as our method uses higher spatial resolution for discrete movements.

For the continuous method OHSMD, we evaluate with (1) Run-Time (RT): Same as the previous one; (2) Makespan (MS (time)): The total time taken for the last agent to reach its goal, from start

![](images/3305a012bd682da54edcc0e248f3c3939d776153837029daa189589f73ddc1bd.jpg)  
Fi<sub>g</sub>ure 7: Performance of the AGV Routin<sub>g</sub> Methods with Dif<sub>e</sub>r<sub>e</sub>nt Fl<sub>ee</sub>t Siz<sub>es o</sub>n R<sub>a</sub>nd<sub>o</sub>m M<sub>aps</sub>

![](images/68fb83653d629281011f497fbe24f520aa7d5c675eef5e4672e8b1bbb070020b.jpg)  
Fi<sub>g</sub>ure 8: Flexibilit<sub>y</sub> Influence Com<sub>p</sub>arison  
to finish; (3) FlowTime (FT): The sum of the times taken for all agents to reach their respective targets.

## 7.2 Random Ma<sub>p</sub> Testin<sub>g</sub>

Fig. 7 presents the performance of the on diferent map sizes and diferent fleet sizes. ⊗ indicates that the scenario is not feasible, and ‘−’ denotes that no result was obtained within the limited time of 600s. For the learning-based methods, since the process operates through an observe-decision-execute cycle, we use the endurance time. Specifically, if no additional AGV reaches its desti nation within the defined endurance time $\tau _ { t } ,$ , the planning process terminates. At this point, the current completion time (runtime) and the number of agents that have reached their destination (success number) are recorded. This ensures that the algorithm does not exceed the predefined time limit, while still providing meaningful results within the given $\tau _ { t } . \tau _ { t } = 5 ( s )$ for the number of agents less than 2048 and $\tau _ { t } = 1 5 ( s )$ for � = 2048.

In terms of scalability, the RL-based methods can scale to a much larger fleet than the classical methods. Because ICBS and ODrM<sup>∗</sup> fail to provide results when the number of AGVs exceeds 64, the table does not display results for 128-2048 AGVs. Additionally, we also tested the performance of CBS, but it was unable to provide results for 64 AGV, so its results are not shown in the table. Besides, OHSMD is explicitly designed for Warehouse A, so we do not test it here. In a map size of 160, both ICBS and ODrM<sup>∗</sup> can only plan paths for up to 64 AGVs. However, PRIMAL and HiRAD can handle path planning for up to 2048 AGVs.

![](images/3b1a13b757c33bafbd0b151900308c9089f078ec86ad7bd5d783ac81d0f51641.jpg)  
Fi<sub>g</sub>ure 9: Performance of Lifelon<sub>g</sub> Tasks in Warehouses

In terms of the path quality and eficiency, when the fleet size is 64, ICBS achieves the shortest makespan and flow path, providing the highest path quality. However, it requires 446.92s to compute. On the other hand, ODrM<sup>∗</sup> improves search eficiency by performing dimensionality reduction on the space, with $\mathrm { O D r M } _ { \epsilon = 1 . 5 } ^ { * }$ completing the path planning for all AGVs in 12.01s, and $\mathrm { O D r M } _ { \epsilon = 1 0 } ^ { * }$ requiring even less time, just 4.98s. $\mathrm { O D r M } _ { \epsilon = 1 0 } ^ { * }$ is the fastest classical method in terms of runtime. However, both PRIMAL and HiRAD ofer faster runtime speeds than $\mathrm { O D r M } _ { \epsilon = 1 0 } ^ { * } ,$ , while the Flow Path does not degrade significantly. Additionally, HiRAD, due to its integration with the velocity control module, can further achieve a shorter makespan while maintaining high eficiency.

Compared to PRIMAL, HiRAD achieves lower runtime and maintains a lower makespan in most cases. This is because HiRAD allows some AGVs to operate at high velocity, enabling them to complete the pathfinding task much more quickly than when taking discrete, step-by-step actions. On a map of size 40, HiRAD achieves a makespan that is approximately 3-4 times faster than PRIMAL. On a map of size 80, HiRAD is about 2-3 times faster, and on a map of size 160, HiRAD achieves a makespan that is approximately 2 times faster. This demonstrates the eficiency of HiRAD in handling larger-scale pathfinding tasks while maintaining competitive performance in terms of makespan.

It is worth noting that both PRIMAL and HiRAD can successfully plan completion paths for almost all AGVs in all scenarios. However, when the AGV number increases to 2048 on map 160, the number of targets that PRIMAL can successfully reach within the endurance time $\tau _ { t }$ drops significantly. In contrast, HiRAD consistently shows an increase in the number of AGVs that reach their destination, demonstrating its scalability and eficiency for large AGV fleets.

## 7.3 Warehouse Lifelon<sub>g</sub> Task Testin<sub>g</sub>

In this section, we test on the two warehouses with lifelong tasks. As shown in Figure 9, OHSMD performs best on map G1. This is not only because it is implemented in C++ but also because it is specially optimized for G1. However, the primary limitation of OHSMD is its inflexibility: its performance would deteriorate in unexpected situations. Figure 8 illustrates the running behavior of the RL-based methods and OHSMD. Suppose the horizontal line represents the time spent on the paths of a set of AGVs. The RL-based methods determine the actions step by step before the movement. Therefore, whenever an interruption happens on any one of the AGVs, it does not afect the others. However, OHSMD has to plan the routes for all the AGVs in serial. More importantly, it requires the actual AGVs to follow planned routes spatially and temporally exactly. When any AGV has a little diference in action (due to the network delay, mechanical residual accumulation, or other failures), the whole system has to halt to avoid collisions. Then, the system has to re-plan the remaining routes again based on the current AGVs’ locations. Therefore, we further provide a simulation of 10% of AGVs encountering issues as OHSMD-10%, which has longer computation times and makespan on the large network than the RL-based methods. This demonstrates the importance of flexibility and the robustness of the proposed methods.

T<sub>a</sub>bl<sub>e</sub> 1: RL Tr<sub>a</sub>inin<sub>g</sub>
<table><tr><td>Map Size | Small Large Large Large</td><td></td><td></td><td></td><td></td><td>Large</td><td>Large</td></tr><tr><td>Obstacle Density</td><td>0</td><td>0</td><td>0</td><td>0.2</td><td>0.2</td><td>0.2</td></tr><tr><td>Reward</td><td>25</td><td>25</td><td>55</td><td>55</td><td>55</td><td>25-&gt;55</td></tr><tr><td>IAM</td><td>N</td><td>N</td><td>N</td><td>N</td><td>Y</td><td>Y</td></tr><tr><td>Curriculum</td><td>N</td><td>N</td><td>N</td><td>N</td><td>N</td><td>Y</td></tr><tr><td>SR1†</td><td>98%</td><td>76%</td><td>94%</td><td>96%</td><td>99%</td><td>99%</td></tr><tr><td>Average Steps</td><td>33</td><td>102</td><td>107</td><td>115</td><td>110</td><td>99</td></tr><tr><td>Convergence</td><td>4.1h</td><td>24h</td><td>18.5h</td><td>21h</td><td>17h</td><td>20h</td></tr><tr><td>SR2†</td><td>86.8%</td><td>71%</td><td>90%</td><td>95%</td><td>99%</td><td>99%</td></tr></table>

<sup>†</sup> SR1 is measured by testing diferent maps under the same settings as the training environment, recording the rate of cases where AGVs successfully reach their targets. SR2 is evaluated on a fixed large map (160×160) with 1024 AGVs, calculating the success rate of AGVs reaching their destinations.

Compared to other learning-based methods, HiRAD-Alt and PRI-MAL struggle to scale to a larger number of AGVs and bigger maps due to their slow runtimes. In contrast, the optimized HiRAD-Asy demonstrates higher time eficiency, enabling it to scale efectively with more AGVs and larger maps.

Overall, these results highlight HiRAD’s strong generalization capability and scalability across various warehouse environments. By eficiently handling larger fleet sizes and diverse spatial layouts, HiRAD-Asy proves to be a more robust and adaptable solution for lifelong multi-AGV pathfinding. This adaptability makes it wellsuited for real-world applications where dynamic and complex environments demand flexible yet eficient routing strategies.

## 7.4 RL-Routin<sub>g</sub> Trainin<sub>g</sub> Performance

Since global path planning decisions are driven by reinforcement learning, the efectiveness of the strategy is highly dependent on the diversity ofthe environment ofthe training. Table 3 presents the test performance of models trained with diferent reward settings, obstacle densities, and action masks on maps of the same size as those used for training. The metrics include success rate (SR1), and average step count (equal to path length). Additionally, the table reports the success rate (SR2) on a specified larger random map with dimensions � = 160, � = 1024. In this section, we define small maps as those with a size between 10 and 40, and large maps as those ranging from 40 to 100.

T<sub>a</sub>bl<sub>e</sub> 2<sub>:</sub> Abl<sub>a</sub>ti<sub>on</sub> St<sub>u</sub>d<sub>y</sub>
<table><tr><td rowspan=3 colspan=1># AgentsMap Size |</td><td rowspan=3 colspan=1>Metrics</td><td rowspan=3 colspan=2>64    25680 160</td><td></td></tr><tr><td rowspan=2 colspan=1>40 480</td><td></td></tr><tr><td rowspan=1 colspan=1>1024160</td></tr><tr><td rowspan=2 colspan=1>HiRAD</td><td rowspan=2 colspan=1>RTMSCA</td><td rowspan=1 colspan=1>3.1 3.8</td><td rowspan=1 colspan=1>28 34</td><td rowspan=1 colspan=1>146</td></tr><tr><td rowspan=1 colspan=1>103 2760 0</td><td rowspan=1 colspan=1>425 7370  0</td><td rowspan=1 colspan=1>7990</td></tr><tr><td rowspan=2 colspan=1>-priority</td><td rowspan=2 colspan=1>RTMSCA</td><td rowspan=2 colspan=1>3.3 3.8176 2764 1</td><td rowspan=2 colspan=1>22 36411 730227</td><td rowspan=1 colspan=1>159</td></tr><tr><td rowspan=1 colspan=1>863255</td></tr><tr><td rowspan=1 colspan=1>-eclusiveZone</td><td rowspan=1 colspan=1>RTMSCA</td><td rowspan=1 colspan=1>2.6 4.3164 28655 27</td><td rowspan=1 colspan=1>19 X359 X500 268</td><td rowspan=1 colspan=1>XX3937</td></tr></table>

Table 3: Com<sub>p</sub>arison <sub>w</sub>ith Coo<sub>p</sub>erative DHC
<table><tr><td></td><td colspan="4">HiRAD</td></tr><tr><td></td><td>RT</td><td>MS</td><td>FP</td><td>SN</td></tr><tr><td>40</td><td>1.11</td><td>192</td><td>1932</td><td>64</td></tr><tr><td>80</td><td>2.41</td><td>426.8</td><td>3792</td><td>64</td></tr><tr><td>一</td><td colspan="4">DHC</td></tr><tr><td></td><td>RT</td><td>MS</td><td>FP</td><td>SN</td></tr><tr><td>40</td><td>8</td><td>475</td><td>1850</td><td>64</td></tr><tr><td>80</td><td>24.5</td><td>895</td><td>3630</td><td>64</td></tr></table>

From Table 3, it can be observed when the reward is set to 25 for small maps, SR1, the average steps and the convergence time perform well. However, the trained model does not generalize efectively to large maps, where SR2 shows a significant decline. Setting a small reward directly on the large map prolongs the learning process, increasing the convergence time. Meanwhile, its efect on test maps and multi-agent scenarios remains limited, as both SR1 and SR2 exhibit suboptimal performance. Increasing the reward for large maps reduces the convergence time while significantly improving SR1 and SR2. This demonstrates that the AGV can successfully find a path in such cases, albeit not optimally. Increasing the obstacle density in the training phase enhances map diversity and raises the learning dificulty for the agent. When the obstacle density reaches 0.2, the convergence time increases; however, the success rate in multi-agent scenarios SR2 improves.

Given the increased dificulty of exploration in complex maps to reach the goal, we adopt Invalid Action Masking (IAM) during training to filter out non-compliant actions and reduce inefective explorations, thereby accelerating model convergence. This not only improves training eficiency but also efectively mitigates path redundancy caused by random decision-making, resulting in more optimized path generation. Ultimately, we introduced curriculum learning, where small maps are initially trained with small rewards. The dificulty of pathfinding is then gradually increased by expanding the map size, adding more obstacles, and raising the reward. This approach leads to better quality paths with a higher success rate while maintaining a reasonable convergence time.

## 8 C<sub>o</sub>n<sub>c</sub>l<sub>us</sub>i<sub>o</sub>n

Enabling the AGVs to run eficiently, conflict-free, and flexibly in a real-life warehouse has been a dream for a long time. The existing solutions cannot be used in practice because of their discrete assumption or centralized planning fashion. In this work, the proposed HiRAD framework fills the gap between RL-based routing and practice for the first time. The HRS modeling enables the step-based RL run in continuous environments, the two-layer mechanism reduces the RL search space dramatically, and the asyn chronous optimization further improves the inference eficiency such that RL-based AGV can finally run in practice. Experiments validate the efectiveness and eficiency of the proposed framework.

## R<sub>e</sub>f<sub>erences</sub>

[1] Takuya Akiba, Yoichi Iwata, and Yuichi Yoshida. 2013. Fast exact shortest-path distance queries on large networks by pruned landmark labeling. In Proceedings of the 2013 ACM SIGMOD International Conference on Management of Data. 349–360.

[2] M BAILEY and K OVERTON. 1990. Automated aircraft engine costing using artificial intelligence. In 26th Joint Propulsion Conference. 1887.

[3] Max Barer, Guni Sharon, Roni Stern, and Ariel Felner. 2014. Suboptimal variants of the conflict-based search algorithm for the multi-agent pathfinding problem. In Proceedings ofthe International Symposium on Combinatorial Search, Vol. 5. 19–27.

[4] Zahy Bnaya and Ariel Felner. 2014. Conflict-Oriented Windowed Hierarchical Cooperative A\*. In 2014 IEEE International Conference on Robotics and Automation (ICRA). 3743–3748.

[5] Robert Bogue. 2016. Growth in e-commerce boosts innovation in the warehouse robot market. Industrial Robot: An International Journal 43, 6 (2016), 583–587.

[6] Eli Boyarski, Ariel Felner, Roni Stern, Guni Sharon, Oded Betzalel, David Tolpin, and Eyal Shimony. 2015. Icbs: The improved conflict-based search algorithm for multi-agent pathfinding. In Proceedings ofthe International Symposium on Combinatorial Search, Vol. 6. 223–225.

[7] Yu Fan Chen, Miao Liu, Michael Everett, and Jonathan P How. 2017. Decentralized non-communicating multiagent collision avoidance with deep reinforcement learning. In 2017 IEEE international conference on robotics and automation (ICRA). IEEE, 285–292.

[8] Robert T Chien, Ling Zhang, and Bo Zhang. 1984. Planning collision-free paths for robotic arm among obstacles. IEEE transactions on pattern analysis and machine intelligence 1 (1984), 91–96.

[9] Edith Cohen, Eran Halperin, Haim Kaplan, and Uri Zwick. 2003. Reachability and distance queries via 2-hop labels. SIAM J. Comput. 32, 5 (2003), 1338–1355.

[10] Larissa Custodio and Ricardo Machado. 2020. Flexible automated warehouse: a literature review and an innovative framework. The International Journal of Advanced Manufacturing Technology 106 (2020), 533–558.

[11] Peng Dai, Daniel Weld, et al. 2011. Artificial intelligence for artificial artificial intelligence. In Proceedings ofthe AAAIConference on Artificial Intelligence, Vol. 25. 1153–1160.

[12] Mehul Damani, Zhiyao Luo, Emerson Wenzel, and Guillaume Sartoretti. 2021. PRIMAL \_2: Pathfinding via reinforcement and imitation multi-agent learning lifelong. IEEE Robotics and Automation Letters 6, 2 (2021), 2666–2673.

[13] Amandeep Dhaliwal. 2020. The rise of automation and robotics in warehouse management. In Transforming management using artificial intelligence techniques. CRC Press, 63–72

[14] Edsger W Dijkstra. 1959. A note on two problems in connexion with graphs. Numerische mathematik 1, 1 (1959), 269–271.

[15] Wenhao Ding, Shuaijun Li, Huihuan Qian, and Yongquan Chen. 2018. Hierarchical reinforcement learning framework towards multi-agent navigation. In 2018 IEEE international conference on robotics and biomimetics (ROBIO). IEEE, 237–242

[16] Cornelia Ferner, Glenn Wagner, and Howie Choset. 2013. ODrM\* optimal multi robot path planning in low dimensional search spaces. In 2013 IEEE international conference on robotics and automation. IEEE, 3854–3859.

[17] Cornelia Ferner, Glenn Wagner, and Howie Choset. 2013. ODrM\* optimal multi robot path planning in low dimensional search spaces. In 2013 IEEE International Conference on Robotics and Automation. 3854–3859.

[18] Robert Geisberger, Peter Sanders, Dominik Schultes, and Daniel Delling. 2008. Contraction hierarchies: Faster and simpler hierarchical routing in road networks. In International workshop on experimental and eficient algorithms. Springer, 319– 333.

[19] Roland Geraerts and Mark H Overmars. 2004. A comparative study of probabilistic roadmap planners. In Algorithmic foundations of robotics V. Springer, 43–57.

[20] Clara Gomez, Marius Fehr, Alex Millane, Alejandra C Hernandez, Juan Nieto, Ramon Barber, and Roland Siegwart. 2020. Hybrid topological and 3d dense mapping through autonomous exploration for large indoor environments. In 2020 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 9673–9679.

[21] Peter E Hart, Nils J Nilsson, and Bertram Raphael. 1968. A formal basis for the heuristic determination of minimum cost paths. IEEE transactions on Systems Science and Cybernetics 4, 2 (1968), 100–107.

[22] William E Howden. 1968. The sofa problem. The computer journal 11, 3 (1968), 299–301.

[23] Sertac Karaman and Emilio Frazzoli. 2011. Sampling-based algorithms for optimal motion planning. The international journal ofrobotics research 30, 7 (2011), 846– 894.

[24] Sven Koenig and Maxim Likhachev. 2005. Fast replanning for navigation in unknown terrain. IEEE Transactions on Robotics 21, 3 (2005), 354–363.

[25] Sven Koenig, Maxim Likhachev, and David Furcy. 2004. Lifelong planning A<sup>∗</sup>. Artificial Intelligence 155, 1-2 (2004), 93–146.

[26] Steven LaValle. 1998. Rapidly-exploring random trees: A new tool for path planning. Research Report 9811 (1998).

[27] Jiaoyang Li, Ariel Felner, Eli Boyarski, Hang Ma, and Sven Koenig. 2019. Improved Heuristics for Multi-Agent Path Finding with Conflict-Based Search.. In IJCAI, Vol. 2019. 442–449.

[28] Jiaoyang Li, Wheeler Ruml, and Sven Koenig. 2021. Eecbs: A bounded-suboptimal search for multi-agent path finding. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 35. 12353–12362

[29] Lei Li, Wen Hua, Xingzhong Du, and Xiaofang Zhou. 2017. Minimal on-road time route scheduling on time-dependent graphs. Proceedings ofthe VLDB Endowment 10, 11 (2017), 1274–1285.

[30] Xiao Liang, Guanglei Meng, Yimin Xu, and Haitao Luo. 2018. A geometrical path planning method for unmanned aerial vehicle in 2D/3D complex environment. Intelligent Service Robotics 11 (2018), 301–312.

[31] Andrei Lissovoi and Carsten Witt. 2013. Runtime analysis of ant colony optimization on dynamic shortest path problems. In Proceedings ofthe 15th annual conference on Genetic and evolutionary computation. 1605–1612.

[32] Lixing Liu, Xu Wang, Xin Yang, Hongjie Liu, Jianping Li, and Pengfei Wang. 2023. Path planning techniques for mobile robots: Review and prospect. Expert Systems with Applications (2023), 120254.

[33] Zhe Liu, Hesheng Wang, Huanshu Wei, Ming Liu, and Yun-Hui Liu. 2020. Prediction, planning, and coordination of thousand-warehousing-robot networks with motion and communication uncertainties. IEEE Transactions on Automation Science and Engineering 18, 4 (2020), 1705–1717.

[34] Pinxin Long, Tingxiang Fan, Xinyi Liao, Wenxi Liu, Hao Zhang, and Jia Pan. 2018. Towards optimally decentralized multi-robot collision avoidance via deep reinforcement learning. In 2018 IEEE international conference on robotics and automation (ICRA). IEEE, 6252–6259.

[35] Abderraouf Maoudj and Anders Lyhne Christensen. 2022. Decentralized multi agent path finding in warehouse environments for fleets of mobile robots with limited communication range. In International Conference on Swarm Intelligence. Springer, 104–116.

[36] Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. 2016. Asynchronous methods for deep reinforcement learning. In International conference on machine learning. PmLR, 1928–1937.

[37] Mohammad Aijaz Mohiuddin, Salman A Khan, and Andries P Engelbrecht. 2016. Fuzzy particle swarm optimization algorithms for the open shortest path first weight setting problem. Applied Intelligence 45 (2016), 598–621.

[38] Dian Ouyang, Lu Qin, Lijun Chang, Xuemin Lin, Ying Zhang, and Qing Zhu. 2018. When hierarchy meets 2-hop-labeling: Eficient shortest distance queries on road networks. In Proceedings of the 2018 International Conference on Management of Data. 709–724.

[39] Dian Ouyang, Long Yuan, Lu Qin, Lijun Chang, Ying Zhang, and Xuemin Lin. 2020. Eficient shortest path index maintenance on dynamic road networks with theoretical guarantees. Proceedings ofthe VLDB Endowment 13, 5 (2020), 602–615.

[40] Chadi F Riman and Pierre E Abi-Char. 2024. Novel Fuzzy Reinforcement Algorithm for Mobile Robot Navigation in Automated Storage. International Journal ofMechanical Engineering and Robotics Research 13, 2 (2024).

[41] Guillaume Sartoretti, Justin Kerr, Yunfei Shi, Glenn Wagner, T. K. Satish Kumar, Sven Koenig, and Howie Choset. 2019. PRIMAL: Pathfinding via Reinforcement and Imitation Multi-Agent Learning. IEEE Robotics and Automation Letters 4, 3 (2019), 2378–2385.

[42] Guni Sharon, Roni Stern, Ariel Felner, and Nathan R Sturtevant. 2015. Conflictbased search for optimal multi-agent pathfinding. Artificial Intelligence 219 (2015), 40–66.

[43] Dingyuan Shi, Yongxin Tong, Zimu Zhou, Ke Xu, Wenzhe Tan, and Hongbo Li. 2022. Adaptive Task Planning for Large-Scale Robotized Warehouses. In 2022 IEEE 38th International Conference on Data Engineering (ICDE). IEEE.

[44] Dingyuan Shi, Nan Zhou, Yongxin Tong, Zimu Zhou, Yi Xu, and Ke Xu. 2023. Collision-Aware Route Planning in Warehouses Made Eficient: A Strip-based Framework. In 2023 IEEE 39th International Conference on Data Engineering (ICDE). IEEE.

[45] Hamed Shorakaei, Mojtaba Vahdani, Babak Imani, and Ali Gholami. 2016. Optimal cooperative path planning of unmanned aerial vehicles by a parallel genetic algorithm. Robotica 34, 4 (2016), 823–836.

[46] Anthony Stentz. 1994. Optimal and eficient path planning for partially-known environments. In Proceedings ofthe 1994 IEEE international conference on robotics and automation

[47] Jonas Stenzel, Dennis Lünsch, and Lea Schmitz. 2021. Automated topology creation for global path planning of large AGV fleets. In 2021 IEEE International Intelligent Transportation Systems Conference (ITSC). IEEE, 3373–3380.

[48] Qi Wang, Markus Wulfmeier, and Bernardo Wagner. 2016. Voronoi-based heuristic for nonholonomic search-based path planning. In Intelligent Autonomous Systems 13: Proceedings of the 13th International Conference IAS-13. Springer, 445–458.

[49] Ruizhong Wu, Mengxuan Zhang, Shuxin Wang, Frodo Kin Sun Chan, Yan Nei Law, and Lei Li. 2025. Continuous Lifelong Conflict-Aware AGV Routing with Kinematic Constraint. Proceedings ofthe VLDB Endowment (2025).

[50] Ruizhong Wu, Mengxuan Zhang, Shuxin Wang, Frodo Kin Sun Chan, Yan Nei Law, and Lei Li. 2025. A Lifelong Conflict-Aware AGV Routing System. In Australasian Database Conference. Springer, 447–462.

[51] Ruizhong Wu, Tianqi Zhang, Yunjie Huang, and Lei Li. 2026. A Demonstration of Continuous Lifelong Conflict-Aware AGV Routing with Kinematic Constraints. Proceedings of the VLDB Endowment (2026).

[52] Peter R Wurman, Rafaello D’Andrea, and Mick Mountz. 2008. Coordinating hundreds of cooperative, autonomous vehicles in warehouses. AI magazine 29, 1 (2008), 9–9.

[53] Jingjin Yu and Steven LaValle. 2013. Structure and intractability of optimal multirobot path planning on graphs. In Proceedings ofthe AAAIConference on Artificial Intelligence, Vol. 27. 1443–1449.

[54] Han-ye Zhang, Wei-ming Lin, and Ai-xia Chen. 2018. Path planning for the mobile robot: A review. Symmetry 10, 10 (2018), 450.

[55] Mengxuan Zhang, Lei Li, Wen Hua, Rui Mao, Pingfu Chao, and Xiaofang Zhou. 2021. Dynamic hub labeling for road networks. In 2021 IEEE 37th International Conference on Data Engineering (ICDE). IEEE, 336–347.

[56] Mengxuan Zhang, Lei Li, Wen Hua, and Xiaofang Zhou. 2021. Eficient 2-hop labeling maintenance in dynamic small-world networks. In 2021 IEEE 37th International Conference on Data Engineering (ICDE). IEEE, 133–144.

[57] Mengxuan Zhang, Lei Li, and Xiaofang Zhou. 2021. An experimental evaluation and guideline for path finding in weighted dynamic network. Proceedings ofthe VLDB Endowment 14, 11 (2021), 2127–2140.

## A P<sub>re</sub>li<sub>m</sub>i<sub>nary</sub>

We list discrete the AGV network and discrete MAPF problem at first in follow part. Then we introduce the RL-based MAPF modeling in the discrete and continuous environments with a discussion of their pro and cons.

## A<sub>.</sub>1 AGV Grid N<sub>e</sub>t<sub>wo</sub>rk <sub>a</sub>nd Di<sub>sc</sub>r<sub>e</sub>t<sub>e</sub> MAPF P<sub>ro</sub>bl<sub>em</sub>

The AGV networks are physically designed as a map � of� × � square grids ofthe same size, where � and � are the row and column numbers. We use the coordinate $( i , j )$ to represent the center of a grid � at row � and column $j ,$ with the upper-left grid’s coordinate being (0, 0), and the bottom-right grid’s being $( m - 1 , k - 1 )$ . A fleet of AGVs is denoted as $N = \{ A G V _ { i } \}$ }. Each vehicle is approximated as a circle in shape, with a diameter that is nearly the same as the grid size. ���<sub>�</sub>’ location at time � is denoted as $l _ { i } ^ { t } = ( x _ { i } ^ { t } , y _ { i } ^ { t } )$ . As the AGV’s size is roughly the same as a $\mathrm { g r i d } _ { \mathrm { \ell } }$ one grid can only be occupied by one AGV at the same time �.

The grids on � have the following types: 1) Static Obstacles O that cannot be entered or occupied by any AGV at any time; 2) Dynamic Obstacles $\mathcal { L } = \{ l _ { i } ^ { t } \}$ that are all grids occupied by AGVs; 3) Vacant Grids that AGVs can move into; 4) Start $s _ { i }$ and Destination $d _ { i }$ positions that form the task ${ \tau } _ { i } = { \langle s _ { i } , d _ { i } \rangle }$ for $A G V _ { i : }$ , where $s _ { i } ~ =$ $( x _ { i } ^ { s } , y _ { i } ^ { s } ) , d _ { i } = ( x _ { i } ^ { d } , y _ { i } ^ { d } )$

From time $t \mathrm { ~ t o ~ } t + 1 , A G V _ { i }$ can take five actions d� $= \ \{ \longrightarrow \} :$ $( 0 , 1 ) , \longleftarrow : ( 0 , - 1 ) , \uparrow : ( - 1 , 0 ) , \downarrow : ( 1 , 0 ) , \odot : ( 0 , 0 ) \}$ , i.e., moving to the neighboring vacant grids or staying at the current grid. Then its next position $l _ { i } ^ { t + 1 } = \bar { ( x _ { i } ^ { t + 1 } , y _ { i } ^ { t + 1 } ) }$ is determined by $( x _ { i } ^ { t } , y _ { i } ^ { t } ) + \mathbf { d } a _ { i }$ A path of ���<sub>�</sub> is an ordered sequence of visiting grids $\begin{array} { r } { p _ { i } = \langle s _ { i } = } \end{array}$ $l _ { i } ^ { 0 } , l _ { i } ^ { 1 } \cdots , l _ { i } ^ { d } = d _ { i } \rangle$ . It should be noted that $l _ { i } ^ { t }$ could be the same as $\boldsymbol { l } _ { i } ^ { t + 1 }$ as the AGV can stay at the same location. The time it takes for an AGV to finish its task is the number of visited grids in the path and is denoted as |�<sub>�</sub> |. Given a set of tasks, its ma<sup>k</sup>espan ��� $\left( \left| p _ { i } \right| \right)$ is the time to finish them all, and <sup>fl</sup>owtime $\sum \left. p _ { i } \right.$ is the sum of each individual task. Now we are ready to define the discrete MAPF problem as follows:

Definition 5 (Discrete MAPF). Given a grid network $M ,$ a set of tasks $\left\{ \tau _ { i } \right\}$ and the corresponding AGVs N, the discrete MAPF aims to compute a collision-free path set $\mathcal { S } = \{ p _ { i } \} _ { i = 1 } ^ { | N | }$ such that their makespan and flowtime are as small as possible.

## A<sub>.</sub>2 Di<sub>scre</sub>t<sub>e</sub> RL<sub>-</sub>MAPF M<sub>o</sub>d<sub>e</sub>li<sub>ng</sub>

One approach to solving the discrete MAPF problem is to utilize reinforcement learning. It treats each AGV as an agent and makes moving decisions based on the environment. Specifically, AGVs need to 1) observe their surroundings to avoid conflicts, and 2) know their destinations to plan routes. Specifically, an $\mathrm { A G V } _ { i } ^ { \phantom { \dagger } } \mathfrak { s }$ s surrounding $b \times b$ grids are collected into a local view window as a matrix $W _ { i : }$ , where each matrix element is either 0 (vacant grid) or 1 (obstacles). Its destination indicated by a vector $\vec { u } = \langle \mathbf { d } x , \mathbf { d } y , \Delta d \rangle \in \mathbb { R } ^ { 3 }$ where d� $= x _ { i } ^ { d } - x _ { i } ^ { t }$ and $\mathbf { d } y = y _ { i } ^ { d } - y _ { i } ^ { t }$ represent the horizontal and vertical displacements, and the scalar $\Delta d = \sqrt { ( \mathbf { d } x ) ^ { 2 } + ( \mathbf { d } y ) ^ { 2 } }$ is the Euclidean distance to the destination. These two components form the observations space $\mathcal { Z } = ( W , \vec { u } )$ , with each $\mathrm { A G V } _ { i } ^ { \phantom { \dagger } } ,$ s denoted as $z _ { i } ^ { t } = ( W _ { i } ^ { t } , { \vec { u } } _ { i } ^ { t } )$ at time step �.

The discrete RL-MAPF problem could be modeled as a Partially Observation Markov Decision Processes (POMDPs), represented by the quintuple (S, A, �, R, �), under a partially observable setting.

(1) State Space S is the set of all possible observation configurations $z$ that an AGV may encounter. That is, we treat $s = z$ , where each distinct observation is regarded as a unique state<sup>1</sup>. State $s ^ { t } \in S$ represents the environment state at time step �.

(2) Action Space $\mathcal { R } = \mathrm { d o m } ( { \bf d } a ) = \{ \uparrow , \right. , \downarrow , \left. , \odot \}$ are the decisions that the AGV can execute. The action chosen by $\mathrm { A G V } _ { i }$ at the time step � is denoted as $a _ { i } ^ { t } \in \mathcal { A } _ { : }$ , and the corresponding state transit is expressed as $s _ { i } ^ { t } \xrightarrow { a _ { i } ^ { t } } s _ { i } ^ { t + 1 }$ , with the location of AGV moving from $l _ { i } ^ { t }$ to $l _ { i } ^ { t } + \mathbf { d } a _ { i } ^ { t }$ and other observation information updated after the action is executed. An action is invalid if the target grid is out of bounds, blocked by a static obstacle, or occupied by another AGV. Then its transition is either rejected or penalized. In all valid cases, any change in position or surroundings caused by taking an action leads to a new state $s ^ { t + 1 } \neq s ^ { t }$

(3) State–Transition Probability $P ( s ^ { t + 1 } \mid s ^ { t } , a ^ { t } )$ denotes the probability that the environment transitions to state $s ^ { t + 1 }$ when action �<sup>�</sup> is executed in state $s ^ { t }$ . If the move is rejected, then $s ^ { t + 1 } = s ^ { t }$

(4) Reward Function $\mathcal { R } ( s ^ { t } , a ^ { t } )$ is a designer–specified mapping that assigns a scalar score $r _ { i } ^ { t }$ to executing action $a _ { i } ^ { t }$ in state $s _ { i } ^ { t }$ for specific $\mathrm { A G V } _ { i ; }$ , thereby encoding task objectives $( e . g .$ shorter steps cost) into a numerical feedback signal. Positive values indicate desirable behaviors, such as reaching a goal, whereas negative values penalize undesirable ones, like moving away from the destination and colliding with obstacles or other agents.

(5) Policy � $: S  \mathcal { A }$ is a mapping from states to actions that specifies the decision rule an AGV follows. It is learned and iteratively refined through reinforcement learning to maximize expected cumulative rewards $\textstyle \sum _ { t = 0 } ^ { T } r _ { i } ^ { t }$ . During training, the policy is continuously updated based on interactions with the environment, gradually converging toward an optimal strategy $\pi ^ { * }$

Now we are ready to define the RL version of the discrete MAPF problem as follows:

Definition 6 (Discrete RL-MAPF). Given a grid network �, a set of tasks $\left\{ \tau _ { i } \right\}$ and the corresponding AGVs $N ,$ the discrete RL-MAPF aims to find an optimal policy $\pi ^ { * }$ that schedules a collisionfree path set $\mathcal { P }$ step by step such that their makespan is as small as possible.

Because the flow time $\Sigma | p _ { i } |$ is equivalent to the number of action decisions that all AGVs made, the time complexity of inference is linear to $\Sigma | p _ { i } |$ . Although it has lower complexity, its result cannot be implemented in real life.

## A<sub>.</sub>3 C<sub>on</sub>ti<sub>nuous</sub> RL<sub>-</sub>MAPF M<sub>o</sub>d<sub>e</sub>li<sub>ng</sub>

In practice, both time and space are continuous. The moving actions of AGVs must follow kinematic constraints, such as velocity $v ,$ acceleration a, and deceleration −a: it first accelerates at a constant a until reaching its maximum velocity $v _ { \mathrm { m a x } }$ or $ { \boldsymbol { \upsilon } } ^ { \mathrm { ~ ~ < ~ } }  { \boldsymbol { v } } _ { \mathrm { m a x } }$ when the distance is not suficient to accelerate to maximum speed, then maintains that velocity and finally decelerates with the deceleration −a to a complete stop precisely at the center of a grid for turning or keeping stationary. Therefore, path $\mathscr { p } _ { i }$ in Continuous MAPF should be reformulated as $\boldsymbol { p } _ { i } = \langle ( \boldsymbol { x } ^ { 0 } , y ^ { 0 } , v ^ { 0 } ) , \ldots , ( \boldsymbol { x } ^ { t } , y ^ { t } , v ^ { t } ) \rangle$ that satisfies the following constraints:

(1) The next position $( x ^ { t + 1 } , y ^ { t + 1 } )$ is determined by $( x ^ { t } , y ^ { t } ) + \mathbf { d } a ^ { t } *$ $v ^ { t } ,$ , where $a ^ { t } \in \mathcal { A }$ , d� $= \{ \right. : ( 0 , 1 ) , \left. : ( 0 , - 1 ) , \uparrow : ( - 1 , 0 ) ,$ ↓: $( 1 , 0 ) , \odot : ( 0 , 0 ) \}$ } is the unit vector directional set, and $v ^ { t }$ is the velocity;

(2) Velocity changes follow $\lvert v ^ { t + 1 } - v ^ { t } \rvert \leq \mathbf { a } _ { \mathrm { m a x } }$ , where $\mathbf { a } _ { \mathrm { m a x } }$ is the max acceleration/deceleration.

Definition 7 (Continuous RL-MAPF). Given a grid environment �, a set ofAGVs N and a set oftasks $\left\{ \tau _ { i } \right\}$ , continuous MAPF aims to compute a valid path set $\mathcal { P } = \{ p _ { i } \} _ { i = 1 } ^ { | N | } , i \in N$ for them such that the time for each AGV to finish its task is as short as possible with the kinematic constraints.

However, in continuous environments, theoretically, the decisions can be made at any point in time, which means the potential number of decision points is infinite within any finite interval. Therefore, in practice, continuous time is often discretized for computational purposes, resulting in a large number of very small time steps. The finer the discretization, the more decision points there are, potentially leading to a much larger number of decisions compared to a purely discrete time setting. It works fine in the single-agent environment, but the complexity grows linearly with the number of agents in the multi-agent environment, so no RL-MAPF solution aims to solve this continuous problem.

## B Hi<sub>g</sub>h-Resolution S<sub>p</sub>atialtem<sub>p</sub>oral Modelin<sub>g</sub>

In this section, we supply our Hight-Resolution Spatialtemporal modeling mentioned in Sec. 4.

## B.1 Hi<sub>g</sub>h-Resolution Ma<sub>p</sub> and Ste<sub>p</sub>

The HRS is composed of two components: a high-resolution map that the AGVs navigate spatially, and a high-resolution step that the AGVs make decisions temporally. In the following, we define these two components formally.

Definition 8 (High-Resolution HR-Map). Given a grid network $M ,$ we increase the grid resolution to $c , c > 1$ by dividing each grid into $c \times c$ smaller cells to obtain a High-Resolution Map $M _ { c } \in \mathbb { R } ^ { m \times k }$ . The original $M = M _ { c = 1 } \in \mathbb { R } ^ { m \times k }$ is called a Coarse Map.

As shown in Figure 3, we divide each small grid into $c \times c = c ^ { 2 }$ smaller cells. Each cell’s size is shaped in $\textstyle { \frac { 1 } { c } } \times { \frac { 1 } { c } }$ . Note that $M _ { c = 1 }$ and $M _ { c > 1 }$ are two precision definitions for the same network. The only diference between them is the minimum unit. Next, we increase the temporal resolution accordingly:

Definition 9 (High-Resolution HR-Step). Given every � time that AGV needs to make a decision to move on the coarse map, we divide � into $\begin{array} { r } { \Delta t { } = \frac { \delta } { c } } \end{array}$ HR-Steps such that AGVs need to make in the HR-Map.

The reason for dividing the map and steps into the same smallest unit is to guarantee that one Δ� decision can make an AGV move at least one cell. Meanwhile, it should be noted that the HRS modeling is not equivalent to an �� × �� cells discrete modeling. This is because, firstly, when an AGV stops or rotates, it still needs to stay at the center of the grids (the $( c / 2 , c / 2 ) – \mathrm { c e l l } )$ , not any arbitrary cell. Secondly, in each HR-Step, the AGV can move more than one cell at higher speeds: When an AGV moves an increasing number of cells consecutively, it is accelerating; when it moves an decreasing number of cells, then it is decelerating; it can also moves at a constant number cells when it travels at a constant speed. In this way, the HRS-Modeling is equivalent to continuous modeling, and we discussed the detailed optimal velocity mapping in Sec. 5.2.

## B<sub>.</sub>2 HRS RL-MAPF M<sub>o</sub>d<sub>e</sub>lin<sub>g</sub>

Now we are ready to model the RL-MAPF in the proposed HRS environment. Specifically, the action $a ^ { t } = ( \mathbf { d } a ^ { t } , \boldsymbol { v } ^ { t } )$ also combine both direction and velocity. Diferent from the continuous modeling, the velocity $v ^ { t }$ corresponds to the number of cells, with neighboring step’s velocity diference being 1 (accelerating), 0 (constant speed), or -1 (decelerating).

Although it looks similar to the discrete RL-MAPF except for the new velocity constraint, it is more complicated to train from the following aspects: 1) The action space $| \mathcal { A } |$ grows from |d�| to |d�|×|�|. As the discrete resolution ofvelocity |�| increases, the AGV faces greater dificulty in policy optimization due to the enlarged combinatorial space; 2) The state space dimension grows substantially. If each state of $\mathrm { A G V } _ { i }$ is represented as $( s _ { i } ^ { t } = ( x _ { i } ^ { t } , y _ { i } ^ { t } , a _ { i } ^ { t } , v _ { i } ^ { t } )$ , the total number of states |S| grows from $( m \times k \times | \mathbf { d } a | ) ^ { | N | }$ to (��×��× $| \mathbf { d } a | \times | \boldsymbol { v } | ) ^ { | N | }$ , which makes the exploration much harder; 3) The reward design becomes more complex. Since AGVs are required to stop precisely at the center of a target grid, the reward function $\mathcal { R } ( s ^ { t } , a ^ { t } )$ depends on the positional error $e = \| ( x ^ { t } , y ^ { t } ) - ( x ^ { * } , y ^ { * } ) \|$ where $( x ^ { * } , y ^ { * } )$ is the center coordinates of a grid. Directly controlling velocity may cause overshooting or oscillations, making it harder for � to converge smoothly; 4) The learning process itself is afected. The convergence time of reinforcement learning is approximately proportional to the product of the state and action space sizes, $\mathcal { T } _ { \mathrm { c o n v } } \propto | S | \times | \mathcal { A } |$ . Hence, the enlarged space results in slower convergence and ineficient exploration, reducing the $\operatorname { A G V } { \boldsymbol { \varsigma } } _ { s }$ ability to learn a stable policy.

Table 4: Reward Settin<sub>g</sub>
<table><tr><td>Action</td><td>Move  $( \uparrow / \downarrow / \left. / \right. )$ </td><td>Collision</td><td>Stay (©)</td><td>Find goal</td></tr><tr><td>Reward</td><td>-0.1</td><td>-2.0</td><td>-0.3</td><td> $+ 5 \longrightarrow + 2 5 \longrightarrow + 5 5$ </td></tr></table>

In terms ofinference, the number ofdecision-making is � times of the discrete version, while on the other hand, each decision-making time, which has much higher complexity as analyzed above, needs to be faster than 1/� of the discrete version.

## C Macro-RL-Routin<sub>g</sub> La<sub>y</sub>er

This layer aims to direct the AGV to reach the destination without collision on the coarse map. We give the specifical training process in the following part. and

## C.1 Macro-RL-Routin<sub>g</sub> Trainin<sub>g</sub>

As analyzed in Section B.2, full-fidelity training with hundreds of AGVs is computationally expensive, so we first learn a single-agent policy on a coarse abstraction of the map $M _ { c = 1 }$ generated randomly. Specifically, the generated maps are constructed such that each grid cell has a probability $p \in \{ 0 , 0 . 1 , 0 . 2 \}$ of containing an obstacle. The start and destination positions of the AGV are randomly assigned under the constraint that no obstacle occupies them and a valid passage exists between them.

To accelerate convergence, we implement the curriculum learning, where the dificulty increases progressively throughout the training process. Initially, the AGV is trained on easier maps with $m = k = 1 0 { \mathrm { ~ a n d ~ } } p = 0$ (no obstacles). As training progresses, the dificulty increases gradually, with the AGV encountering more complex environments, eventually reaching the most challenging configuration where � = � = 100 (largest training map) and $p = 0 . 2$ (highest obstacle density).

As dificulties increase, the reward setting also changes. This is because a fixed goal reward may not provide suficient incentive for optimal pathfinding. As Table 4 shows, An AGV receives a reward of -0.1 for moving up (↑), down (↓), left (←), or right (→), while stopping incurs a penalty of -0.3, and collision made the penalty of -2.0. Upon reaching its goal, the AGV is rewarded with a value of +5 $\mathrm { t o } + 5 5$ , dynamically determined by the training hardness. In small maps, a reward of +5 is adequate. But as the map size increases, the accumulated movement penalties $( e . g . , - 0 . 1 \times 2 0 0 = - 2 0 $ on map size with $1 0 0 \times 1 0 0 )$ can outweigh a small completion reward (+5 reward), reducing the AGV’s motivation to reach the goal eficiently. Therefore, the goal reward is dynamically scaled from +5 to +55 based on the environment size and dificulty, ensuring that AGVs remain suficiently incentivized to complete longer paths in larger maps.

The employed RL-routing network is built upon the Actor-Critic AC framework [36]. The deep neural network was employed to approximate the $_ \mathrm { A G V } \mathbf { \vec { \mathbf { \sigma } } } _ { \mathbf { { s } } }$ policy �. As illustrated in Figure 4, the state $s ^ { t } = z ^ { t }$ includes the local view window $W ^ { t } ~ \in ~ \mathbb { R } ^ { b \times b }$ with the guidance vector $\vec { u } ^ { t } \in \mathbb R ^ { 3 }$ . They were first encoded and then concatenated to form a feature � at each time step �. The sequential features � are processed through the LSTM layer to generate two critical outputs: 1) Predicted Action Distribution $( \pi ^ { t } ( a ^ { t } \mid s ^ { t } ) )$ : A probability distribution over the action space A. Each probability $\pi ^ { t } ( a ^ { t } \mid s ^ { t } )$ indicates the likelihood of selecting action $a ^ { t }$ under the current policy $\pi ^ { t }$ based on the state �<sup>�</sup>. 2) Action Value $( V ^ { t } ( a ^ { t } | s ^ { t } ) )$ $\mathrm { A }$ scalar estimate of the expected cumulative rewards when starting from state $s ^ { t }$ and taking the action $a ^ { t }$ following the current policy $\pi ^ { t }$ thereafter. It serves as a baseline or critic signal in policy gradient methods to guide the update toward the optimal policy $\pi ^ { * }$

## C<sub>.</sub>2 Alt<sub>e</sub>rn<sub>a</sub>tin<sub>g</sub> D<sub>ec</sub>i<sub>s</sub>i<sub>o</sub>n-V<sub>e</sub>l<sub>oc</sub>it<sub>y</sub> C<sub>o</sub>ntr<sub>o</sub>l

We identify the following conditions such that the HR-Step-based decision-making RL AGV can run in real life:

Lemma 2 (RL Real-Time Applicability Conditions). Given an HRS-Modeling and its corresponding velocity mapping, the AGVs can run continuously under kinematic constraints in real life only when the following conditions are satisfied:

(1) The AGV stays at the grid center when it stops;

(2) The velocity mode satisfies kinematic constraints of three stages Acceleration-(Max)Speed-Deceleration;

(3) The decision time Δ� is shorter than the moving time unit.

The first condition ensures the AGV movements still follow the grid network design. The second condition ensures the velocity mode follows the physical law and corresponds to the real-life move ment. The third condition is the RL eficiency constraint, which is crucial in real-life applications. If it takes a longer time to make decisions than to move, then the decision becomes invalid, and the conflict cannot be avoided among the moving AGVs. Now, we can connect our HRS modeling with the realistic environment:

Theorem 2 (HRS-Continuous Eqivalence). The proposedHRS modeling, combinedwith HR-step-basedRL decision-making, enables the AGV to run continuously under kinematic constraints.

Proof. We prove our design satisfies Lemma 1. Firstly, the Mvx and Mxx all stop at the first and second neighbor grids by default. As for Mvv, when it reaches the speed 0.4, it is in the center of the second grid (1.0 grids and 4 Δ�). Then, it could 1) take another 1.0 grids and 5 Δ� to stop at the third neighbor grids’ center (as Mxx mode), or 2) take another 2 Δ� to move to the next grid’s center place and decide to decelerate or not. If deceleration is chosen, the AGV switches to the Mvz mode and comes to a full stop at the next grid center following a constant deceleration. Therefore, all three modes satisfy Condition 1. Secondly, because the velocity increases and decreases follow the same acceleration parameter and together they satisfy the Acceleration-(Max)Speed-Deceleration pattern, it is easy to implement in real life. Finally, as the third condition is about computational eficiency, we will address it in Section 6 by reducing the complexity. □

It should be noted that we use our � = 10 design as an example, so the actual resolution parameters and velocity mode are subject to change depending on the physical system. However, regardless of the changes, the conditions of Lemma 1 remain unchanged, and the parameters can be obtained through measurement and alignment.