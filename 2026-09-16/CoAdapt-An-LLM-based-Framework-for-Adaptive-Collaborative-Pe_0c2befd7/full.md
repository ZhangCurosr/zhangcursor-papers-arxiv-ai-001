# CoAdapt: An LLM-based Framework for Adaptive Collaborative Perception in IIoT Robotic Swarms

Houssam Hajj Hassan\*, Antonia Maria Masucci\*, Lynda Zitoune†, Salah Eddine Elayoubi†

\*Orange Innovation, Châtillon, France.

† Université Paris-Saclay, CNRS, CentraleSupélec, Laboratoire des Signaux et Systèmes, France.

{houssam.hajjhassan, antoniamaria.masucci} @orange.com

lynda.zitoune@l2s.centralesupelec.fr, salaheddine.elayoubi@centralesupelec.fr

Abstract—Industrial IoT environments increasingly deploy autonomous mobile robots for tasks such as material handling, product assembly, or infrastructure inspection. In such deployments, collaborative perception enables robots to share LiDAR observations and collectively construct a richer model of their environment than an individual agent could produce alone. However, industrial environments are dynamic spaces where robot positions shift continuously, network bandwidth fluctuates, and the marginal contribution of robots to perception quality varies at runtime. Existing collaborative perception approaches are designed for static participation assumptions and cannot adapt to these dynamics without sacrificing either detection precision or communication efficiency. This paper presents CoAdapt, an adaptive collaborative perception framework for IIoT robotic swarms in which a Large Language Model (LLM) serves as a runtime fusion controller, jointly deciding which robots participate in the fusion process and which fusion algorithm to apply based on the current spatial configuration and network state. The LLM reasons over structured natural language descriptions of the scene derived from raw LiDAR point clouds, requiring no taskspecific training and generalizing to unseen swarm topologies. Evaluated on the OPV2V benchmark across 25 scenarios, our approach achieves a 38% reduction in communication cost while maintaining detection precision comparable to static baseline approaches.

Index Terms—Robot swarms, IIoT, Collaborative perception, Self-adaptation, Data fusion, Large Language Models

## I. INTRODUCTION

The proliferation of autonomous robotic systems in industrial environments has given rise to a new class of cyberphysical deployments, in which large numbers of mobile agents equipped with heterogeneous sensors, constrained computational resources, and wireless communication interfaces, must collectively perceive and act upon a shared environment [1]. These robotic swarms are increasingly central to IIoT applications such as warehouse automation, infrastructure inspection, and autonomous logistics, where individual robots operate under severe perceptual limitations: restricted fields of view, sensor occlusions caused by obstacles or other agents, and range constraints that degrade single-agent detection at distance [2], [3]. Enabling such swarms to operate autonomously without continuous human supervision or centralized orchestration is a fundamental objective of modern IoT system design, and a prerequisite for scalable deployment in dynamic, unstructured industrial environments.

Collaborative perception [4]–[6] addresses these limitations by enabling robots to share sensory information, constructing a richer, more complete representation of the environment than an individual agent could produce alone. The dominant sensing modality in autonomous vehicle and robotic swarm deployments is LiDAR-based object-detection, which produces observations as a cluster of point coordinates. In this context, collaborative perception takes the form of cooperative data fusion, where point cloud observations from multiple robots are aggregated. Three broad data fusion paradigms have emerged in the literature. Early fusion [4] aggregates raw point clouds before any feature extraction, maximizing information sharing at the cost of high communication volume. Late fusion [6] shares only final detection outputs (bounding boxes and confidence scores), minimizing bandwidth usage but discarding inter-agent complementarity. Intermediate fusion [5], [7], [8] shares learned feature representations at an intermediate network layer, achieving a favorable tradeoff between precision and communication cost. Yet all three paradigms treat the fusion strategy as a static design choice where the participating robot set and fusion algorithm are fixed at deployment, blind to runtime variations in network load or scene conditions. This rigidity becomes particularly problematic in dynamic IIoT swarm deployments, where environmental conditions (e.g., occlusions, robot positioning) vary at runtime, the network state fluctuates, and the marginal contribution of each robot to perception quality is neither fixed nor uniform. Agents with overlapping fields of view introduce redundancy without improving coverage, whereas agents far from the region of interest add communication cost without detection benefit. Moreover, such relationships vary widely at runtime: as robots move, the utility of each agent's contribution changes continuously, and a participation decision that was optimal may become suboptimal seconds later. Addressing this challenge therefore requires jointly and continuously deciding which robots participate in the fusion process, and which fusion strategy to apply, in response to the current environment and network state.

Existing work addresses subsets of this problem in isolation, with the majority of them either fixing the fusion paradigm at design time [5], [6] or adapting feature transmission within a fixed intermediate fusion architecture [9], [10], without considering runtime switching between fusion paradigms or joint adaptation of the participant set. Static subset selection methods [7], [8], [11] determine participation at deployment time based on fixed topology assumptions, and degrade as robots move. Network-aware fusion methods [9] reduce communication volume but optimize independently of applicationlevel precision requirements and runtime network fluctuations. Learning approaches [12] have shown promise for adaptive cooperative perception but require extensive environmentspecific training and generalize poorly to novel topologies. However, none of these approaches jointly adapt participant selection and fusion strategy at runtime in response to changing network and environmental conditions.

In this paper, we propose CoAdapt, an adaptive collaborative perception framework that jointly optimizes participant selection and fusion strategy at runtime, balancing detection precision against communication cost in dynamic IIoT swarm deployments. Our approach introduces a Scene Abstraction Module that converts raw LiDAR point cloud data into structured natural language descriptions of the swarm's spatial configuration, enabling a Large Language Model (LLM) to serve as an autonomic fusion controller. At each control cycle, the LLM receives a description of the current environment state, and produces a joint decision that includes the subset of robots that participate in the fusion process, and the fusion algorithm to apply. CoAdapt enables adaptive collaborative perception in IIoT systems through the following key contributions:

– An adaptive collaborative perception framework that jointly adapts participant selection and fusion strategy at runtime in response to dynamic network and environmental conditions.   
– An LLM-driven fusion policy mechanism that reasons over structured natural language abstractions of LiDAR point cloud data, swarm geometry, and network state to produce joint participant selection and fusion algorithm decisions without task-specific training or exhaustive search.

– An open-source prototype implementation evaluated on the OPV2V benchmark, demonstrating significant communication cost reduction while maintaining comparable detection precision.

The rest of the paper is organized as follows. Section II provides a brief background about collaborative perception and surveys related work. Section III presents the adaptive data fusion problem in dynamic robot swarms. Section IV overviews the architecture of CoAdapt, and Section V presents the experimental evaluation of our approach. Section VI concludes the paper with future directions.

## II. RELATED WORK

This section provides first a brief background on collaborative perception, then surveys existing state-of-the-art approaches for communication-efficient collaborative perception, and self-adaptive robotic systems.

a) Collaborative perception and data fusion: Collaborative perception has emerged as a promising paradigm for overcoming the inherent limitations of single-agent sensing in multi-robot systems [3], [6], [13], [14]. By aggregating observations across agents, cooperative approaches significantly extend effective sensing range and mitigate occlusion effects that degrade single-robot perception. Three broad fusion paradigms have been established in the literature: early fusion, which aggregates raw sensor data (in the form of LiDAR point clouds, i.e., a set of data points in 3D space) before feature extraction [4]; late fusion, which shares final detection outputs [6]; and intermediate fusion, which exchanges learned feature representations at an intermediate network layer [5], [7], [8]. Intermediate fusion has attracted significant attention as it achieves a favorable tradeoff between detection precision and communication overhead. Recent work has begun to address the communication efficiency of collaborative perception more explicitly. F-Cooper [4] introduced the first featurelevel fusion pipeline; V2VNet [3] applied spatially-aware graph neural networks for joint perception and prediction. AttFuse [6] and CoBEVT [5] extend existing models with attention mechanisms, achieving state-of-the-art precision on benchmark datasets. Who2com [7] and When2com [8] learn to select communication partners and timing based on requestresponse matching. DiscoNet [12] employs knowledge distillation to compress intermediate representations, reducing transmission volume without sacrificing detection quality.

b) Communication-efficient collaborative perception: A parallel line of work explicitly targets the communication cost of collaborative fusion. Where2comm [9] proposes spatial confidence maps that allow agents to transmit only perceptually critical features, achieving over 100, 000× reduction in communication volume on benchmark datasets while matching prior methods in detection precision. V2X-ViT [15] introduces heterogeneous multi-agent attention to reduce redundant transmissions. SyncNet [16] addresses latency-aware fusion under asynchronous communication by tackling temporal misalignment. CodeFilling [17] addresses the communication bottleneck by using a codebook-based representation to transmit compact integer codes instead of high-dimensional feature maps across robots. Point Cluster [18] proposes a compact message unit that decouples low-level structural and highlevel semantic information, enabling strategic point sampling to preserve scene structure at a lower communication cost than that of feature map transmission. CoSDH [19] combines supply-demand-aware region selection with an intermediatelate hybrid fusion mode, where late fusion compensates for accuracy degradation when bandwidth constraints make intermediate fusion insufficient.

While these approaches reduce communication overhead within a fixed fusion paradigm, none jointly adapts participant selection and fusion strategy at runtime in response to fluctuating network conditions and swarm dynamics.

c) Self-adaptive robotic systems: Beyond perceptionspecific methods, a growing body of work addresses selfadaptation and autonomous decision-making in robotic systems more broadly. ReBeT [20] extends Behavior Trees with explicit quality requirement nodes to enable architecturebased self-adaptation in ROS2 robotic systems, allowing a robot to modify its software architecture at runtime to satisfy non-functional objectives such as energy consumption and safety. Atik et al. [21] focus on battery degradationaware task and charge allocation algorithms that maximize mission revenue and battery lifespan in autonomous ground robots. Swarmalators [22] addresses the practical challenge of implementing swarmalator dynamics on physical robots by deriving analytical bounds on the discrete-time step size that guarantee non-oscillatory convergence to emergent spatial patterns. Aguzzi et al. [23] propose a decentralized runtime replanning approach grounded in aggregate programming, where continuously evolving fields represent mission state and enable robots to collectively adapt their task plan in response to failures without centralized coordination. Large Language Models have also been used in robotics for task planning [24]–[26], scene understanding [27], and multi-agent coordination [28]. Notably, Nascimento et al. [29] proposed integrating LLMs into MAPE-K-based multi-agent systems, demonstrating that GPT-based models can serve as autonomic controllers for self-adaptive behavior. In the context of autonomous driving, V2V-LLM [30] explored multi-modal LLM reasoning for cooperative driving decisions. Li et al. [31] provide a comprehensive overview of generative AI within self-adaptive systems, specifically mapping LLM capabilities onto MAPE-K loop phases.

![](images/f7ad7b55424f6e5f791b3c85fd81a0408f09baf01a447bf55cc4fc796bc0f50e.jpg)  
(a) Geometric redundancy: R5 overlaps R6 (b) Dynamic occlusion: R5 uniquely covers (c) Network-QoS tension: bandwidth drop and R4. a blind spot, making it indispensable. forces algorithm switch.  
Fig. 1: Three snapshots of an IIoT robot swarm exposing distinct challenges in adaptive collaborative perception. Blue robots are active participants; gray robots are excluded.

Despite this growing body of work, no existing approach applies LLM-based reasoning to the joint selection of fusion algorithm and participant subset in a collaborative perception system, nor does any prior work use LiDAR-derived point cloud data as the reasoning substrate for such decisions.

## III. THE ADAPTIVE DATA FUSION PROBLEM

In this section, we first provide a motivating scenario to highlight self-adaptation challenges in dynamic IIoT robot swarms. We then formally present the adaptive data fusion problem.

## A. Motivating Scenario

To highlight collaborative perception challenges in dynamic robotic swarms, we consider a smart manufacturing plant in the context of Industry 4.0 environments. Consider an IIoT factory floor in which a swarm of several autonomous mobile robots (AMRs) $\{ R _ { 1 } , \ldots , R _ { 6 } \}$ collaboratively perform object detection to support material handling operations. Each robot is equipped with a LiDAR sensor and communicates over a shared wireless network. The factory floor contains fixed infrastructure such as shelving units, conveyor belts, and machinery that create persistent occlusion zones, as well as moving obstacles such as forklifts and human workers that generate transient occlusions at runtime.

Figure 1 illustrates three snapshots of this deployment, each exposing a distinct challenge that a static data fusion policy cannot resolve. In Figure 1a, robots R1, R2, and R3 are tightly co-located near a conveyor belt, and therefore their fields of view substantially overlap. Fusing all three observations from R1, R2, and R3 produces minimal coverage gain relative to fusing only R1 and R3, which occupy more complementary positions. The same applies for R4, R5, and R6, which are clustered next to each other. Transmitting R2, R4, and R6's full point clouds adds a significant load on the network without improving detection precision. A static full-participation policy cannot identify this redundancy; an adaptive policy must reason over the current context and robot configuration to remove R2, R4, and R6 from the participant set.

Figure 1b shows how transient occlusions affect which robots participate in the fusion process. As a forklift approaches R4's field of view, R4's contribution becomes uniquely valuable since no other robot is seeing the new obstacle. At the same time, R5 navigates around the forklift and its field of view temporarily covers a blind spot invisible to all other robots. In this context, excluding R5's observation from the fusion process causes a missed detection. The optimal participant set at this timestep $\{ R _ { 1 } , R _ { 3 } , R _ { 4 } , R _ { 5 } \}$ differs from that of Figure 1a. In addition, as network utilization increases, sending compressed feature maps (intermediate fusion) instead of raw point clouds (early fusion) becomes the more appropriate approach.

Figure 1c highlights a case where telemetry traffic from industrial sensors on the factory floor leads to an increase in network utilization to 85%. Under these conditions, transmitting the raw point clouds or even compressed feature maps becomes infeasible within the real-time latency budget. Late fusion reduces communication volume at the cost of slightly degrading detection precision. A reduced participant set consisting of $\{ R _ { 2 } , R _ { 4 } \}$ enables reducing communication data volume while satisfying detection precision constraints. Identifying this combination requires reasoning jointly over the network state, the current robot positions, and the precisioncommunication tradeoff. No existing collaborative perception system performs this joint reasoning at runtime.

These three snapshots define the core challenge that this paper solves, where the optimal fusion policy including which robots participate in the fusion process, is a function of the current environment state, the detection precision, and the network conditions, all of which change continuously during runtime operation. Solving this challenge requires a system capable of autonomous and joint decision-making across all three dimensions.

## B. Problem Formulation

We now formalize the collaborative perception problem for a robotic swarm operating under dynamic situations. We consider a swarm of N autonomous mobile robots $\mathcal { R } =$ $\{ r _ { 1 } , r _ { 2 } , \dots , r _ { N } \}$ operating on a shared factory floor environment. Each robot $r _ { i }$ is equipped with a LiDAR sensor and a wireless communication interface. At each discrete timestep $t \in \tau .$ robot $r _ { i }$ acquires a point cloud observation $O _ { i } ( t ) \in$ $\mathbb { R } ^ { K _ { i } ( t ) \times 3 }$ , where $K _ { i } ( t )$ is the number of points returned by the sensor at timestep t, and each point is represented by its 3D Cartesian coordinates $( x , y , z )$ . Robots communicate over a shared wireless network with capacity $C ( t )$ , which may vary over time due to competing traffic and channel conditions.

At time t, the full system is represented as $\begin{array} { r l } { S ( t ) } & { { } = } \end{array}$ $( S ^ { e n v } ( t ) , S ^ { n e t } ( t ) , S ^ { a p p } ( t ) )$ , where:

• Environment state $\mathbf { \mathit { S } } ^ { e n v } ( t ) = \{ ( x _ { i } ( t ) , \theta _ { i } ( t ) ) \} _ { i = 1 } ^ { N }$ , where $x _ { i } ( t ) \in \mathbb { R } ^ { 2 }$ and $\theta _ { i } ( t ) \in [ 0 , 2 \pi )$ denote the position and heading of robot $r _ { i } .$ The environment state implicitly encodes inter-robot distances, field-of-view overlaps, and occlusion geometry induced by the swarm's current spatial configuration.

• Network state $S ^ { n e t } ( t ) = C ( t )$ , representing the available communication bandwidth at timestep t.

• Application state $S ^ { a p p } ( t ) = Q _ { m i n }$ , the minimum detection precision required by the application, expressed as average precision at a fixed Intersection over Union (IoU) threshold (AP@IoU). We treat this as a time-invariant requirement set at deployment.

Let $\mathcal { A } ~ = ~ \{ \alpha _ { 1 } , \alpha _ { 2 } , \ldots , \alpha _ { M } \}$ denote the set of candidate fusion algorithms available to the system. In our model, A includes early fusion, late fusion, and a set of intermediate fusion models. A fusion policy π is a mapping:

$$
\pi : { \mathcal { S } } ( t ) \mapsto ( P ( t ) , \alpha ( t ) )\tag{1}
$$

where $P ( t ) \subseteq \mathcal { R } , | P ( t ) | \geq 1$ is the selected subset of robots participating in the fusion process, and $\alpha ( t ) \in { \mathcal { A } }$ is the chosen fusion algorithm. The policy is re-evaluated at each control period and may produce a different decision as ${ \mathbf { } } S ( t )$ evolves.

To make the objective concrete, we define the network capacity C explicitly, which denotes the data volume (in bytes)

transmitted over the network at timestep t. For a participant subset $P ( t )$ of size $n = | P ( t ) |$ under fusion algorithm α:

$$
\mathcal { C } ( P ( t ) , \alpha ( t ) ) = \sum _ { r _ { i } \in P ( t ) } \delta ( \alpha , r _ { i } ( t ) )\tag{2}
$$

where $\delta ( \alpha , r _ { i } )$ is the per-robot transmission size under algorithm α:

$$
\delta ( \alpha , r _ { i } ) = \left\{ \begin{array} { l l } { K _ { i } \cdot 3 \cdot b _ { f l o a t } } & { \mathrm { i f ~ } \alpha = \mathrm { e a r l y ~ f u s i o n } } \\ { d _ { f e a t } \cdot b _ { f l o a t } } & { \mathrm { i f ~ } \alpha \in \mathrm { i n t e r m e d i a t e ~ f u s i o n } } \\ { d _ { d e t } \cdot b _ { f l o a t } } & { \mathrm { i f ~ } \alpha = \mathrm { l a t e ~ f u s i o n } } \end{array} \right.\tag{3}
$$

where $b _ { f l o a t }$ is the byte size of a floating-point value, $d _ { f e a t }$ is the feature map size, and $d _ { d e t }$ is the size of the detection output vector.

Let $\mathcal { Q } ( P ( t ) , \alpha ( t ) )$ denote the object detection precision achieved by applying algorithm α to the fused observations of subset $P$ at time t, measured as $\mathbf { A P } @ \mathbf { I o U }$ . We formulate the objective as finding a policy $\pi ^ { * }$ that maximizes the expected detection precision while minimizing communication cost, subject to the application QoS constraints:

$$
\begin{array} { r l } & { \pi ^ { * } = \underset { \pi } { \arg \operatorname* { m a x } } \mathbb { E } \bigl [ \mathcal { Q } ( P ( t ) , \alpha ( t ) ) - \lambda \cdot \mathcal { C } ( P ( t ) , \alpha ( t ) ) \bigr ] } \\ & { } \\ & { \mathrm { s . ~ t . ~ } \quad \mathcal { Q } ( P ( t ) , \alpha ( t ) ) \geq Q _ { m i n } \qquad \forall t } \end{array}\tag{4}
$$

where $\lambda > 0$ is a scalar weighting factor that governs the precision-communication tradeoff. Note that C depends on both the number and identity of participating robots $P ( t )$ , and the fusion algorithm chosen $\alpha ( t )$ , which determines the type and volume of data transmitted.

The decision space of π at each timestep is $\left( 2 ^ { N } - 1 \right) \times \left| \mathcal { A } \right|$ reflecting all non-empty subsets of R paired with each candidate fusion algorithm. While manageable by exhaustive search in isolation, three properties make real-time optimization complex in practice. First, the detection precision $\mathcal { Q } ( P ( t ) , \alpha ( t ) )$ cannot be evaluated analytically. It requires running the full fusion and detection pipeline, making exhaustive online evaluation infeasible within real-time latency budgets. Second, the relationship between $S ^ { e n v } ( t )$ and $\mathcal { Q }$ is non-linear and environment-dependent: which robots are geometrically complementary depends on occlusion patterns that change continuously as robots move and the scene evolves. Third, jointly reasoning over all three state components $( S ^ { e n v } , S ^ { n e t } , S ^ { a p p } )$ simultaneously requires a policy capable of multi-objective reasoning, which neither heuristic selection rules nor singleobjective optimization methods handle naturally.

These properties motivate the use of an LLM as the reasoning engine for $\pi ;$ rather than evaluating Q directly, the LLM approximates the mapping from $S ( t )$ to $( P ( t ) , \alpha ( t ) )$ through structured natural language reasoning over a compact scene abstraction, as described in Section IV.

## IV. SYSTEM ARCHITECTURE

Existing collaborative perception systems treat fusion strategy and participant selection as fixed design-time decisions, leaving the system blind to the runtime dynamics that most affect performance, such as shifting swarm topology, transient occlusions, and fluctuating network conditions. We address this by framing collaborative fusion control as an autonomic decision problem and solving it with a large language model acting as the Plan component of a MAPE-K feedback loop [32]. The key insight is that given a sufficiently structured description of the current system state, an LLM can reason jointly over geometric, application-level, and network objectives, producing a fusion policy $( P ( t ) , \alpha ( t ) )$ without exhaustive search and without retraining for new environments. This is made possible through our Scene Abstraction Module (SAM), which converts raw LiDAR point clouds and network telemetry into a compact JSON representation that exposes the information relevant to the fusion decision. Figure 2 shows CoAdapt's four-layer architecture. We describe each layer in detail below.

![](images/2777d7613f8d719281800331c83ed2243cbd6c9a59f52d0c63abf41156101c72.jpg)  
Fig. 2: The CoAdapt architecture.

![](images/e4a06d6c9cb6085b494873381adce1a0a36386002c8918fd0a0f5336ea202ac6.jpg)  
Fig. 3: CoAdapt's integration within the MAPE-K loop.

a) Physical Layer: Each robot $r _ { i } \in \mathcal { R }$ is equipped with a LiDAR sensor producing point cloud observation $O _ { i } ( t ) \in$ $\mathbb { R } ^ { K _ { i } \times 3 }$ at regular intervals (e.g., 10Hz). We consider that a leader robot (also called ego robot [9], [10]) is the one executing the fusion policy, deciding which other robots to include as participants, and producing the final detection output. Robots broadcast lightweight state beacons to the ego robot containing their current position $x _ { i } ( t )$ and heading $\theta _ { i } ( t )$ . In addition, we implement a Scene Abstraction Module (SAM) that enables each robot to produce structured descriptions of the scene it observes, based on the produced LiDAR observation. The

SAM addresses a fundamental interface problem where raw point clouds cannot be interpreted by LLMs. The SAM bridges this gap by transforming heterogeneous physical observations into a structured natural language representation that encodes the geometric and contextual information needed for fusion policy reasoning. The SAM produces a description of objects perceived by robots including their position relative to the observing robot, their size and their density. More information about how the SAM is implemented is provided in Section V. Robots periodically share their scene descriptions with the ego robot. The ego robot also simultaneously monitors network conditions, measuring instantaneous available bandwidth C(t) and per-link latency across the swarm. These raw observations, including physical observations and network metrics, are forwarded to the Perception Layer at each control cycle.

b) Perception Layer: The Perception Layer serves as the integration point between raw physical observations and the reasoning process. Upon receiving the physical layer outputs at each control cycle, the ego robot fuses the individual scene descriptions from participating robots $P ( t )$ with their reported positions $x _ { i } ( t )$ and heading $\theta _ { i } ( t )$ , and the measured network metrics $C ( t )$ into a unified scene representation. This fusion produces a context-aware characterization of the environment that includes spatial relationships between robots, perceived object descriptions, and network conditions. The resulting representation is structured as a natural language prompt and forwarded to the Reasoning Layer, where it serves as the informational basis for the LLM fusion policy.

c) Reasoning Layer: The Reasoning Layer is responsible for adapting the data fusion algorithm and selecting participating robots at runtime. A single LLM instance runs on the ego robot and serves as the autonomic fusion controller. At each control cycle, the ego robot constructs a prompt composed of three elements: (i) a fixed system prompt encoding the optimization objective in natural language (maximize the detection precision subject to application goals and network constraints), along with the fusion algorithm vocabulary and output format specification; (ii) the current scene descriptions from robots; and (iii) a structured output constraint requiring the response to be valid JSON. The LLM reasons jointly over swarm geometry, network state, application requirements, and prior performance, producing a fusion policy decision in the following format:

```json
{
"participants": ["R1", "R3", "R4"],
"algorithm": "intermediate",
"reason":"R4 provides unique occlusion
coverage; bandwidth sufficient for
intermediate fusion with 3 agents"
}
```

The reason field includes a human-readable explanation of the LLM's decision, providing a diagnostic signal for detecting system reasoning failures for IIoT operators.

d) Network Control Layer: CoAdapt includes a Network Control Layer operating above the Reasoning Layer, responsible for wireless resource management at a coarser timescale. While the Reasoning Layer adapts the fusion policy to the current network state C(t) by selecting participant subsets and fusion algorithms that fit within available bandwidth, it does so without modifying the network configuration itself. The Network Control Layer closes this gap by actively adjusting network resources in response to application demands. For instance, this can include assigning traffic priority to fusion data streams or reallocating bandwidth from non-critical channels when sustained congestion prevents the Reasoning Layer from meeting $Q _ { m i n }$ . This layer operates at a slower timescale than the per-cycle decisions of the Reasoning Layer, intervening only when network conditions drift beyond what fusion policy adaptation alone can compensate for. This two-layer design reflects a deliberate separation of concerns. The Reasoning Layer handles faster, environment-driven adaptation of the perception policy; the Network Control Layer handles slower, demand-driven adaptation of the network infrastructure.

Integration within the MAPE-K loop. CoAdapt instantiates a MAPE-K control loop over the collaborative perception pipeline, as shown in Figure 3. The Monitor () phase collects raw observations from the physical layer at each control cycle that includes scene descriptions from participating robots, their positions, and measured network metrics. The Analyze (A) phase constructs the unified scene representation in the Perception Layer, integrating perceptual and network state into a structured prompt. The Plan (©) phase is realized by the Reasoning Layer, where the LLM reasons over the current knowledge base to produce a fusion policy π. The Execute () phase applies the selected fusion algorithm over the chosen participant subset, producing the final detection output. The shared Knowledge base () includes the up-todate optimization objective, set of available fusion algorithms A, and QoS requirement $Q _ { m i n } .$ and is injected into the LLM system prompt at each control cycle.

## V. EXPERIMENTAL EVALUATION

This section presents the experimental evaluation of CoAdapt. We start first by describing the experimental setup including the dataset, data fusion models, implementation of the SAM, and LLM and bandwidth configurations. We then evaluate CoAdapt's ability to effectively select robots that participate in the fusion process under dynamic scene and network conditions. We then proceed to show how CoAdapt adapts the fusion strategy as a function of bandwidth, and finally compare the communication cost-precision tradeoff of CoAdapt against rule-based and default approaches. The code used for the experimental evaluation, along with the fusion models, datasets, configurations, and results, are publicly available at https://github.com/houssamhh/coadapt.

## A. Experimental Setup

Dataset. We evaluate CoAdapt on the OPV2V dataset [6], [33], a large-scale simulated collaborative perception benchmark dataset collected using the CARLA [34] simulator across diverse urban and suburban scenarios. OPV2V is among the few publicly available datasets providing multi-vehicle LiDAR observations with up to six simultaneous agents per scenario. This is a critical requirement for evaluating swarm-level fusion policy decisions, as datasets with a higher number of agent counts remain scarce in the collaborative perception literature [35]. The OPV2V dataset contains 73 scenarios simulated across 6 road types and 9 cities in CARLA. Note that in our evaluation, we only consider scenarios containing 3 vehicles or more to effectively evaluate CoAdapt's participant selection ability.

Fusion Models. We evaluate CoAdapt's fusion policy against three static baselines representing each fusion paradigm, implemented within the OpenCOOD framework [6], [36]. OpenCOOD provides a benchmark framework for evaluating collaborative perception algorithms against existing state-ofthe-art approaches and datasets, including OPV2V. In our evaluation, we select the Cooper model [37] for early fusion, attentive fusion using the VoxelNet [38] detection network for intermediate fusion, and late fusion using the Point Pillars model [39]. These represent the best performing models (in terms of precision) available in OpenCOOD. CoAdapt selects among these three algorithms at runtime. Note that CoAdapt's architecture is agnostic to specific fusion models: any collaborative perception model can be substituted for the fusion backends evaluated here, without modification to the adaptation mechanism. Fusion inference models run on an NVIDIA RTX 3090 GPU with 24GB of memory.

Scene Abstraction Module (SAM). The SAM is implemented using Open3D [40] for point cloud processing. For each robot's LiDAR observation at each frame, the SAM first removes the ground plane via RANSAC-based plane segmentation [41], isolating non-ground points. The remaining points are then clustered using DBSCAN [42]. Each remaining cluster is described by its 3D centroid position relative to the ego vehicle, its Euclidean distance to the ego vehicle, its axisaligned bounding box dimensions, and its point density (points per unit volume). These per-object descriptions are serialized into structured natural language and appended to the LLM prompt alongside the robot's global position.

LLM Configuration. We evaluate 4 open-source LLMs: Gemma 4 (31B), GPT-OSS (20B and 120B), and Llama3.3 (70B). All models are deployed locally on an NVIDIA H100 GPU with 80GB of memory. We select a control cycle window of 50 frames, which corresponds to a timestep of 5 seconds. At each control cycle, the ego robot constructs a prompt comprising the current scene descriptions from all in-range vehicles, their positions, and the current network state, and queries the LLM for a fusion policy decision.

Bandwidth Configuration. Since OPV2V does not provide network measurements, we simulate time-varying bandwidth conditions using a two-component model designed to reflect realistic IIoT factory floor dynamics. We map morning shift hours (06:00 am–12:00 pm) to a high-bandwidth tier of 55 Mbps, afternoon shift hours (12:00 pm–6:00 pm) to a medium tier of 20 Mbps, and evening shift hours (6:00 pm–12:00 am) to a low tier of 5 Mbps. Time information is already encoded in scenario names in OPV2V and can readily be extracted. To simulate network fluctuations, we add a sinusoidal fluctuation of amplitude 15% of the base tier.

![](images/fe9cc507370831fe49566f85976447a1df5403a0dbd44727db079e6e849c3888.jpg)

(a) Gemma 4 31B.  
![](images/17cb05ff37cf8b873b6abf00c1f8cb033cb5eb6b77d0703bd436e3ccfe009de8.jpg)  
(c) GPT-OSS 120B.

![](images/0fc389bd40162976bb2f2c2c48fffbc6d49d2893ccc5f321a5db972a8fd4f760.jpg)

(b) Llama3.3 70B.  
![](images/5a2f0651549d7647aeeeb5ca2a092fb85a66726fc036990808222e4329c19e83.jpg)  
(d) GPT-OSS 20B.  
Fig. 4: Number of selected CAVs vs. network bandwidth.

## B. Evaluating Participant Selection

We first evaluate the ability of CoAdapt to adapt participant selection in response to varying network conditions. Figure 4 shows the number of Connected Autonomous Vehicles (CAVs) selected by each LLM across all evaluation scenarios, alongside the simulated available bandwidth. The default baseline represents existing approaches that always include all in-range CAVs, irrespective of network conditions or scene geometry.

CoAdapt consistently reduces participant count relative to the default full-participation baseline, pruning redundant CAVs based on their spatial contribution and the available network bandwidth. All four tested LLMs consistently select fewer CAVs than the default approach across scenarios, achieving an average reduction of 26% in participant count (11% for Gemma 4 31B, 7% for GPT-OSS 20B, 6% for GPT-OSS 120B, and 40% for Llama3.3). This shows that the models respond to bandwidth variations, selecting fewer participants during low-bandwidth periods and maintaining higher participation when bandwidth is abundant. This network-aware behavior is most pronounced in Gemma 4 and Llama 3.3, which exhibit stronger correlation between bandwidth drops and participant reduction. GPT-OSS 20B and 120B show similar selection patterns to each other, suggesting that model scale alone does not determine selection behavior within the same model family.

To illustrate that participant reduction reflects strategic spatial reasoning rather than indiscriminate exclusion, Figure 5 shows the spatial configuration of selected and excluded CAVs for a representative scenario¹. The ego vehicle is positioned at the center of its 45m communication range, which corresponds to the communication range used in the OPV2V dataset. Figure 5 shows that the CoAdapt-selected CAVs are often those that provide geometrically complementary coverage, while excluded CAVs tend to be spatially clustered near already-selected participants, introducing redundant observations without coverage gain. This pattern is consistent across three models: Gemma 4, GPT-OSS 120B and Llama 3.3. All three exclude the CAV furthest from providing unique coverage, though they differ in which specific agents they retain. This shows that the LLMs successfully internalize spatial reasoning over the scene descriptions produced by the SAM, selecting participants based on coverage complementarity rather than proximity or availability alone. GPT-OSS 20B, on the other hand, retains spatially redundant CAVs that provide overlapping coverage, suggesting that smaller models may lack the spatial reasoning capacity to reliably identify complementary configurations. This points to a limitation of purely LLM-driven selection, and motivates the integration of a deterministic fallback component as part of future work to handle cases where LLM reasoning produces suboptimal participant sets.

## C. Precision Evaluation under Dynamic Bandwidth

We now analyze how CoAdapt adapts the fusion strategy under dynamic network conditions. Figure 6 shows the fusion strategy selected by each LLM across all evaluation scenarios, alongside the simulated available bandwidth.

CoAdapt dynamically switches between fusion paradigms across control cycles, with the selected algorithm reflecting the trade-off between available bandwidth and detection precision. We can notice that Gemma 4 and Llama3.3 models exhibit the richest adaptive behavior, switching across all three fusion paradigms in response to bandwidth variations. They often select intermediate fusion during high-bandwidth periods and fall back to late fusion when bandwidth is constrained. Early fusion is selected by Gemma 4 at only three control cycles, when bandwidth is high enough to accommodate sending raw point clouds over the network.

![](images/a1d277ce36df312dbfa21f6322cd685791f1ec27060e612a7ad7d247a0f50c4c.jpg)  
(a) Gemma4.

![](images/e99dd948c5c7f2ec5b48bf9813efb93c24e261a2fe2e564e2a3ca465366ba7d6.jpg)  
(b) Llama 3.3 70B.

![](images/59e0d51581572747b5ca56df3771bbf51257506b5c9890c8f9ebed53d811d6e3.jpg)  
(c) GPT-OSS 20B.

![](images/1da166b3f0c4070844f3c9e568da425e6d894dfad15dd45d8cf65acb072bead7.jpg)  
(d) GPT-OSS 120B.

Fig. 5: Spatial positions of CoAdapt-selected CAVs.  
![](images/8409a06022a919dda490124e1c84d024322ed283b5197e100d961f9d7b4d666f.jpg)  
(a) Gemma 4 31B.

![](images/2ef67e91aa5bb069307b5bbfd7944fe4dfdabc25ec613fa02ba1047734270d24.jpg)

![](images/a1f59df0693927e4019cafd04477a8945c821ebe50e6d6c0c18b5f0856b21ca4.jpg)  
(c) GPT-OSS-120B.

(b) Llama3.3 70B.  
![](images/4d030bb8e87ef79a3fbbb72d6d61c2fdc30352786b8bef1a33f8e293e389bad2.jpg)  
(d) GPT-OSS-20B.

Fig. 6: CoAdapt-selected fusion strategy vs. available bandwidth.  
![](images/d43edca362307e0a00dfb672d490e128fd09e858e2e5a84a566005abdf9ff58a.jpg)  
Fig. 7: Average communication cost / frame vs. average precision.

GPT-OSS models, on the other hand, show a more conservative approach, selecting almost exclusively late fusion across scenarios. They rarely select intermediate fusion, and never select early fusion. This conservative behavior is consistent with the high communication cost of early fusion in practice, which makes it an unattractive choice under realistic wireless network constraints. This suggests a tendency to default to lower-cost, lower-risk options under uncertainty. Nonetheless, the strategy selections remain generally coherent with the network conditions: late fusion is a valid and communicationefficient choice, and the models do not select early fusion when bandwidth is low.

We next evaluate the communication cost-precision tradeoff. For comparison purposes, we implement a rule-based approach that selects intermediate fusion when enough bandwidth (more than 20 Mbps) is available, and late fusion otherwise. We compare CoAdapt against the rule-based approach, and existing approaches that use a static intermediate fusion policy regardless of bandwidth fluctuations. We measure the average precision at an Intersection over Union (IoU) threshold of 0.7 (AP@0.7), which constitutes a standard metric for 3D object detection that measures the fraction of detected objects whose predicted bounding boxes sufficiently overlap with ground truth annotations. Figure 7 shows the average communication cost per frame vs. the average precision for CoAdapt with the 4 used LLMs against the rule-based and default approach. Because GPT-OSS models almost always choose late fusion, their average communication cost is significantly low: both models have an average cost of less than 10 KB / frame. This is expected, as late fusion models are very communicationefficient, transmitting only detection boxes in the form of textual data. However, they achieve the lowest precisions, at an AP@0.7 of 0.766 and 0.774 for GPT-OSS 20B and GPT-OSS 120B, respectively. Gemma 4, on the other hand, achieves a favorable communication cost-precision tradeoff, with an AP@0.7 of 0.792 against an average communication cost of 152.2 KB/frame. This constitutes an 11% reduction in communication load at a cost of a 2.3% precision drop compared to rule-based approaches, which represents a comparable precision at lower communication cost. In terms of precision, Llama 3.3 achieves the best precision at an AP@0.7 of 0.862, for a communication cost that is 38% lower than default approaches. These results demonstrate that LLM-driven fusion policy selection offers a favorable and configurable tradeoff between communication cost and detection precision.

![](images/341493e2c195d8294d4997f9a6ce932656cd992525329cfc11c72f6b430a5afc.jpg)

![](images/bb2ed283c3c3cad5e14363cf1a145d43c6c1cb4e6e2b32deb76385d70489a19a.jpg)

![](images/03e2d1b4fed3e6d58b485172e3ccd7a3e92916dfe86e4c3df570b37db8d43918.jpg)

![](images/378d59d0f2054ac73668873067d3d63c4c112f0bfe7f60b2679b9ae03d0eabbe.jpg)

![](images/89e33c8054d88330377e6b259044ae9aa6c8b19d644b7d824193db45627a72ff.jpg)  
(a) Gemma 4 31B.

![](images/0089f9a239472cf8915e1727a4657914ebf50b354ad8bae2b7d08599e95d7dde.jpg)  
(b) Llama 3.3 70B.

![](images/763f1edddf85f3efdf5dfe2c25238f39190fbb6cb2fee14dad3843fc86009349.jpg)

![](images/18766537935a9e380ff216fc6bec746c4983812fa5bce66c82ac1ac5b2725fe3.jpg)

![](images/6c5a8b0c12e4a8e361ff40609841b1d65850e4b641611c9c312f0a833b39ea65.jpg)  
(c) GPT-OSS 20B.

![](images/6506fcaa407385b0fc938253bbb498776dea7a37daad43dab9eafd7e5793beac.jpg)  
(d) GPT-OSS 120B.  
Fig. 8: Average communication cost / frame / fusion strategy vs. precision.

We further show the communication cost and AP@0.7 of CoAdapt compared to the default full-participation baseline, broken down by fusion paradigm and model. Figure 8 groups scenarios by the fusion algorithm selected by the CoAdapt at each control window. Across all models and fusion paradigms, CoAdapt-selected participant subsets consistently reduce communication cost compared to the default baseline while maintaining comparable detection precision. In intermediate fusion windows, CoAdapt achieves substantial cost reductions of 50% with GPT-OSS models, 42% with the Llama model, and 38% with the Gemma model, by pruning redundant participants. AP@0.7 values are comparable between CoAdapt and default configuration, confirming that participant pruning does not meaningfully degrade precision. A similar pattern holds in late fusion windows, where the already low communication cost of late fusion is further reduced by selecting a smaller participant subset, again without significant precision loss compared to the default approach.

Note that precision drops in scenarios S12 and S25 are attributable to scene-level difficulty inherent to those scenarios (which is demonstrated by a precision drop for the default approach), rather than suboptimal participant selection by CoAdapt. This confirms that the LLM policy does not introduce additional precision risk beyond what the underlying fusion models already exhibit under challenging conditions.

## D. Limitations

While CoAdapt demonstrates promising results in using LLMs in adaptive collaborative perception, several limitations must be acknowledged:

Synthetic Bandwidth Model. The bandwidth conditions used in this evaluation are synthetically derived from scenario timestamps in OPV2V, mapping shift hours to predefined tiers with sinusoidal fluctuations. While this model captures the temporal structure of industrial network load, it does not reflect the full complexity of real IIoT wireless environments, which exhibit bursty interference, spatial variation, and protocol-level dynamics. Validating CoAdapt against real network traces or a real testbed is part of our future work.

Dataset Constraints. The evaluation is limited by the scarcity of collaborative perception datasets supporting high CAV counts. OPV2V provides up to six simultaneous agents per scenario, which, while sufficient to demonstrate the adaptive behavior of CoAdapt, does not reflect the scale of large industrial swarm deployments where tens of robots may operate concurrently. The decision space grows exponentially with the number of agents, as described in Section III, and scenarios with larger participant sets are the ones where the LLM's reasoning ability can be showcased against static approaches. The development of large-scale multi-agent perception datasets is an open challenge for the collaborative perception community [35], and future evaluation of our framework will depend on the availability of such datasets.

Absence of a Deterministic Fallback. Our system currently relies entirely on LLM reasoning for participant selection and fusion strategy decisions. Even though our evaluation shows that tested LLMs produced valid outputs in response to all queries, no deterministic fallback exists for edge cases where the model produces suboptimal or invalid output. As demonstrated by GPT-OSS 20B's failure to identify spatially redundant CAVs in certain scenarios, smaller models may lack the spatial reasoning capacity required for reliable operation.

A hybrid architecture combining LLM-driven reasoning with a rule-based safety layer would improve robustness and is a natural direction for future work.

LLM Inference Latency. While CoAdapt focuses on demonstrating the feasibility and capabilities of LLM-driven autonomic adaptation for collaborative perception, inference latency represents a practical constraint that must be addressed before deployment in real-time control loops. Our evaluation showed that the mean LLM response times range from 21 seconds for Gemma 4 31B to 44 seconds for Llama 3.3 70B. Several approaches exist to reduce LLM inference latency, including model quantization [43], knowledge distillation into smaller surrogate models [44], and speculative decoding [45], and their integration into the control loop is left as future work.

## VI. CONCLUSION AND FUTURE WORK

This paper presents CoAdapt, an adaptive collaborative perception framework for IIoT robotic swarms. We rely on Large Language Models as runtime fusion controllers, jointly selecting the participant subset and fusion algorithm at each control cycle based on the current spatial configuration and network state. We implement a Scene Abstraction Module to bridge the gap between raw LiDAR point cloud data and LLM reasoning by converting heterogeneous physical observations into structured natural language scene descriptions. CoAdapt instantiates a MAPE-K control loop over the collaborative fusion pipeline, enabling self-adaptive behavior without task-specific training or exhaustive online search. Experimental evaluation on the OPV2V benchmark across 25 scenarios demonstrates that CoAdapt consistently reduces communication cost compared to the static full-participation baseline, while maintaining detection precision comparable to the underlying fusion models.

Our future work will focus on three extensions. First, we will implement the Network Control Layer, enabling active adaptation of the wireless infrastructure to support higher robot participation when application precision requirements cannot be met through fusion policy adjustment alone. Second, we will introduce a constraint validation layer that verifies LLM outputs against physical and operational constraints before execution, coupled with a rule-based fallback policy that activates when the LLM produces invalid or low-confidence decisions. Third, we will evaluate the framework on physical robotic platforms and real-world network traces to assess the sim-to-real transferability of our approach.

## DATA AVAILABILITY

Our code, datasets and results are publicly available at: https://github.com/houssamhh/coadapt.

## ACKNOWLEDGMENT

This work is supported by the CANCUN project funded by the French National Research Agency (ref. ANR-24-CE25- 3560).

[1] O. Vermesan, A. Bröring, E. Tragos, M. Serrano, D. Bacciu, S. Chessa, C. Gallicchio, A. Micheli, M. Dragone, A. Saffiotti et al., "Internet of Robotic Things-Converging Sensing/Actuating, Hyperconnectivity, Artificial Intelligence and IoT Platforms," in Cognitive hyperconnected digital transformation. River Publishers, 2022, pp. 97–155.

[2] E. Debie, K. Kasmarik, and M. Garratt, "Swarm Robotics: A Survey from a Multi-tasking Perspective," ACM Computing Surveys, vol. 56, no. 2, pp. 1–38, 2023.

[3] T.-H. Wang, S. Manivasagam, M. Liang, B. Yang, W. Zeng, J. Tu, and R. Urtasun, "V2VNet: Vehicle-to-Vehicle Communication for Joint Perception and Prediction," in European Conference on Computer Vision (ECCV), 2020.

[4] Q. Chen, X. Ma, S. Tang, J. Guo, Q. Yang, and S. Fu, "F-Cooper: Feature Based Cooperative Perception for Autonomous Vehicle Edge Computing System Using 3D Point Clouds," in ACM/IEEE Symposium on Edge Computing (SEC), 2019.

[5] R. Xu, Z. Tu, H. Xiang, W. Shao, B. Zhou, and J. Ma, "CoBEVT: Cooperative Bird's Eye View Semantic Segmentation with Sparse Transformers," in Conference on Robot Learning (CoRL), 2022.

[6] R. Xu, H. Xiang, X. Xia, X. Han, J. Li, and J. Ma, "OPV2V: An Open Benchmark Dataset and Fusion Pipeline for Perception with Vehicle-to-Vehicle Communication," in IEEE International Conference on Robotics and Automation (ICRA), 2022.

[7] Y.-C. Liu, J. Tian, C.-Y. Ma, N. Glaser, C.-W. Kuo, and Z. Kira, “"Who2com: Collaborative Perception via Learnable Handshake Communication," in 2020 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020, pp. 6876–6883.

[8] Y.-C. Liu, J. Tian, N. Glaser, and Z. Kira, "When2com: Multi-agent Perception via Communication Graph Grouping," in Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition, 2020, pp. 4106–4115.

[9] Y. Hu, S. Fang, Z. Lei, Y. Zhong, and S. Chen, "Where2comm: Communication-Efficient Collaborative Perception via Spatial Confidence Maps," in Advances in Neural Information Processing Systems (NeurIPS), 2022.

[10] D. Yang, K. Yang, Y. Wang, J. Liu, Z. Xu, R. Yin, P. Zhai, and L. Zhang, “How2comm: Communication-Efficient and Collaborationpragmatic Multi-agent Perception," Advances in Neural Information Processing Systems, vol. 36, pp. 25 151–25 164, 2023.

[11] F. Golpayegani, Z. Sahaf, I. Dusparic, and S. Clarke, "Participant Selection for Short-term Collaboration in Open Multi-agent Systems," Simulation Modelling Practice and Theory, vol. 83, pp. 149–161, 2018.

[12] Y. Li, S. Ren, P. Wu, S. Chen, C. Feng, and W. Zhang, "Learning Distilled Collaboration Graph for Multi-agent Perception," Advances in Neural Information Processing Systems, vol. 34, pp. 29 541–29 552, 2021.

[13] R. Hadidi, J. Cao, M. Woodward, M. S. Ryoo, and H. Kim, "Distributed Perception by Collaborative Robots," IEEE Robotics and Automation Letters, vol. 3, no. 4, pp. 3709–3716, 2018.

[14] Y. Han, H. Zhang, H. Li, Y. Jin, C. Lang, and Y. Li, "Collaborative Perception in Autonomous Driving: Methods, Datasets, and Challenges," IEEE Intelligent Transportation Systems Magazine, vol. 15, no. 6, pp. 131–151, 2023.

[15] R. Xu, H. Xiang, Z. Tu, X. Xia, M.-H. Yang, and J. Ma, "V2X-ViT: Vehicle-to-Everything Cooperative Perception with Vision Transformer," European Conference on Computer Vision (ECCV), 2022.

[16] Z. Lei, S. Ren, Y. Hu, W. Zhang, and S. Chen, "Latency-aware Collaborative Perception," in European Conference on Computer Vision. Springer, 2022, pp. 316–332.

[17] Y. Hu, J. Peng, S. Liu, J. Ge, S. Liu, and S. Chen, "Communicationefficient Collaborative Perception via Information Filling with Codebook," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 15 481–15 490.

[18] Z. Ding, J. Fu, S. Liu, H. Li, S. Chen, H. Li, S. Zhang, and X. Zhou, “Point Cluster: A Compact Message Unit for Communication-efficient Collaborative Perception," in The Thirteenth International Conference on Learning Representations, 2025.

[19] J. Xu, Y. Zhang, Z. Cai, and D. Huang, "CoSDH: Communicationefficient Collaborative Perception via Supply-demand Awareness and Intermediate-late Hybridization," in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 6834–6843.

[20] E. Alberts, I. Gerostathopoulos, V. Stoico, and P. Lago, “"ReBeT: Architecture-based Self-adaptation of Robotic Systems through Behavior Trees," in 2024 IEEE International Conference on Autonomic Computing and Self-Organizing Systems (ACSOS). IEEE, 2024, pp. 1–10.

[21] S. T. Atik, D. Grosu, and M. Brocanelli, “Sustainability-aware Online Task and Charge Allocation for Autonomous Ground Robot Fleets," in 2024 IEEE International Conference on Autonomic Computing and Self-Organizing Systems (ACSOS). IEEE, 2024, pp. 11–20.

[22] U. Schilcher, M. Schref, and C. Bettstetter, "Swarmalators with Discrete-Time Coupling: Which Step Size to Choose?" in 2025 IEEE International Conference on Autonomic Computing and Self-Organizing Systems (ACSOS). IEEE, 2025, pp. 112–121.

[23] G. Aguzzi, M. Baiardi, A. Cortecchia, B. Miloradovic, A. Papadopoulos, D. Pianini, and M. Viroli, “A Field-based Approach for Runtime Replanning in Swarm Robotics Missions," in 2025 IEEE International Conference on Autonomic Computing and Self-Organizing Systems (AC-SOS). IEEE, 2025, pp. 1–10.

[24] W. Huang, P. Abbeel, D. Pathak, and I. Mordatch, "Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents," in International Conference on Machine Learning (ICML), 2022, pp. 9118–9147.

[25] S. S. Kannan, V. L. Venkatesh, and B.-C. Min, "Smart-LLM: Smart Multi-agent Robot Task Planning using Large Language Models," in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 12 140–12 147

[26] J. Ao, F. Wu, Y. Wu, A. Swiki, and S. Haddadin, "LLM-as-BT-Planner: Leveraging LLMs for behavior tree generation in robot task planning," in 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2025, pp. 1233–1239.

[27] S. Liu, J. Zhang, R. X. Gao, X. V. Wang, and L. Wang, "Vision-language Model-driven Scene Understanding and Robotic Object Manipulation," in 2024 IEEE 20th International Conference on Automation Science and Engineering (CASE). IEEE, 2024, pp. 21–26.

[28] T. Yang, P. Feng, Q. Guo, J. Zhang, X. Zhang, J. Ning, X. Wang, and Z. Mao, “AutoHMA-LLM: Efficient Task Coordination and Execution in Heterogeneous Multi-agent Systems using Hybrid Large Language Models," IEEE Transactions on Cognitive Communications and Networking, vol. 11, no. 2, pp. 987–998, 2025.

[29] N. Nascimento, P. Alencar, and D. Cowan, “"Self-adaptive Large Language Model (LLM)-based Multiagent Systems," in 2023 IEEE International Conference on Autonomic Computing and Self-Organizing Systems Companion (ACSOS-C). IEEE, 2023, pp. 104–109.

[30] H.-k. Chiu, R. Hachiuma, C.-Y. Wang, S. F. Smith, Y.-C. F. Wang, and M.-H. Chen, “V2V-LLM: Vehicle-to-vehicle cooperative autonomous driving with multi-modal large language models," arXiv preprint arXiv:2502.09980, 2025.

[31] J. Li, M. Zhang, N. Li, D. Weyns, Z. Jin, and K. Tei, "Generative AI for Self-Adaptive Systems: State of the Art and Research Roadmap," ACM Transactions on Autonomous and Adaptive Systems, 2024.

[32] J. O. Kephart and D. M. Chess, “The Vision of Autonomic Computing," Computer, vol. 36, no. 1, pp. 41–50, 2003.

[33] https://mobility-lab.seas.ucla.edu/opv2v/.

[34] https://carla.org/.

[35] N. Wang, D. Shang, Y. Gong, X. Hu, Z. Song, L. Yang, Y. Huang, X. Wang, and J. Lu, "Collaborative Perception Datasets for Autonomous Driving: A Review," IEEE Sensors Journal, 2025.

[36] https://github.com/DerrickXuNu/OpenCOOD.

[37] Q. Chen, S. Tang, Q. Yang, and S. Fu, "Cooper: Cooperative Perception for Connected Autonomous Vehicles based on 3D Point Clouds," in 2019 IEEE 39th International Conference on distributed computing systems (ICDCS). IEEE, 2019, pp. 514–524.

[38] Y. Zhou and O. Tuzel, “VoxelNet: End-to-end Learning for Point Cloud based 3D Object Detection," in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 4490–4499.

[39] A. H. Lang, S. Vora, H. Caesar, L. Zhou, J. Yang, and O. Beijbom, "PointPillars: Fast Encoders for Object Detection from Point Clouds," in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 12 697–12705.

[40] Q.-Y. Zhou, J. Park, and V. Koltun, "Open3D: A Modern Library for 3D Data Processing," arXiv preprint arXiv:1801.09847, 2018.

[41] L. Li, F. Yang, H. Zhu, D. Li, Y. Li, and L. Tang, "An Improved RANSAC for 3D Point Cloud Plane Segmentation Based on Normal Distribution Transformation Cells," Remote Sensing, vol. 9, no. 5, p. 433, 2017.

[42] D. Deng, "DBSCAN Clustering Algorithm Based on Density," in 2020 7th international forum on electrical engineering and automation (IFEEA). IEEE, 2020, pp. 949–953.

[43] J. Lin, J. Tang, H. Tang, S. Yang, W.-M. Chen, W.-C. Wang, G. Xiao, X. Dang, C. Gan, and S. Han, "AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration," Proceedings of machine learning and systems, vol. 6, pp. 87–100, 2024.

[44] X. Xu, M. Li, C. Tao, T. Shen, R. Cheng, J. Li, C. Xu, D. Tao, and T. Zhou, “A Survey on Knowledge Distillation of Large Language Models," arXiv preprint arXiv:2402.13116, 2024.

[45] H. Xia, Z. Yang, Q. Dong, P. Wang, Y. Li, T. Ge, T. Liu, W. Li, and Z. Sui, "Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding," Findings of the Association for Computational Linguistics: ACL 2024, pp. 7655–7671, 2024.