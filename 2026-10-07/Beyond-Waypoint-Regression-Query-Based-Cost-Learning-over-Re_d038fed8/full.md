# Beyond Waypoint Regression: Query-Based Cost Learning over Reachable Ego Futures for End-to-End Driving

Ahmed Abouelazm<sup>1,2</sup>, Rupert Polley<sup>1,2</sup>, Qingyuan Zhang<sup>1</sup>, Yin Wu<sup>2,3</sup>, Philip Schörner<sup>1</sup>, Carl Esselborn<sup>3</sup>, and J. Marius Zöllner<sup>1,2</sup>

<sup>1</sup> FZI Research Center for Information Technology (abouelazm@fzi.de) 2 Karlsruhe Institute of Technology <sup>3</sup> CARIAD SE

Abstract. End-to-end planners based on waypoint regression achieve strong open-loop accuracy, but they primarily learn to mimic expert geometry and remain dificult to adapt to deployment-time safety constraints. We propose a query-based cost-learning framework that estimates bounded costs for dynamically reachable ego trajectory queries, rather than dense BEV cells or a small regressed trajectory set. Compact joint scene tokens capture coherent multimodal agent futures, while contingency-aware cost aggregation and cost-guided intra-cluster MPPI mixing convert the learned cost topology into feasible ego plans. On nuScenes, our method improves over prior cost-estimation planners such as ST-P3 and NMP, outperforms most regression baselines in collision rate, while remaining competitive in L2, and retaining an interpretable cost interface. On real-world driving logs, the proposed planner reduces collision rates compared with SparseDrive and Alpamayo without finetuning, while maintaining a diverse set of candidate trajectories.

Keywords: End-to-end driving · Decision Making · Motion Planning

## 1 Introduction

End-to-End (E2E) driving has progressed from direct sensor-to-control learning [2, 11] to unified systems that jointly learn perception, prediction, and planning [5, 10]. Recent sparse representations make these systems more scalable by allocating capacity to compact task- or instance-level tokens instead of dense raster features [12,13,24]. However, the final planning interface remains a bottleneck: decisions should remain risk-aware, interpretable, physically feasible, and adaptable to deployment-time safety constraints.

Waypoint regression approaches are efective because they directly predict a short horizon of future ego motions. However, imitation objectives primarily reward spatial similarity to expert demonstrations and do not explicitly distinguish safe behavior from collision-causing hard negatives [26]. Consequently, regression planners are dificult to inspect or repair, may produce infeasible behavior under distribution shift, and weakly incorporate safety through post-hoc rescoring of a limited top-k set. Cost maps ofer a more abstract alternative by exposing a risk landscape that can be optimized, repaired, and combined with external costs at runtime.

![](images/517797a56c9914bd43b93761e7e15ae3204a7ef0195375e34fc1cded60bd9ee0.jpg)  
Fig. 1: Multi-view images feed a sparse perception encoder with temporal instance memory. The proposed planner combines joint scene representations and reachable ego queries to estimate bounded costs and refine ego plans through intra-cluster mixing.

Existing E2E cost-map planners such as Neural Motion Planner (NMP) [31] and ST-P3 [9], however, estimate dense rasterized costs over many unreachable cells, spreading supervision away from feasible trajectories. Their flexibility has therefore often come at the cost of lower open-loop planning accuracy than direct regression. Moreover, many planners rely on marginal motion modes, which can ignore cross-agent correlations and miss scene-consistent interaction outcomes [23]. This motivates a cost representation that preserves the optimization benefits of cost maps while avoiding dense supervision over irrelevant space.

We propose a vectorized, query-based E2E cost planner that narrows this gap by learning a cost estimation over the reachable ego trajectories rather than over a dense spatial grid. The model estimates mode-conditioned costs for sampled ego trajectory queries from compact joint scene tokens built from vectorized map and agent representations. It then aggregates costs over coherent future scene modes with a contingency strategy and refines trajectories through cluster-based MPPI-style mixing guided by the learned cost topology. This design retains the flexibility of cost optimization while approaching regression-style accuracy and improving transfer to real-world driving. The key contributions of this work are:

– Query-based cost estimation over sampled reachable ego trajectories instead of dense raster grids.

– Compact joint scene tokens that condition ego costs on coherent multimodal futures for contingency planning.

Cost-guided intra-cluster mixing for continuous reachable-space coverage beyond top-k selection.

– Ranking-based cost learning objective driven by expert trajectories, collision risk, and of-road violations.

## 2 Related Work

E2E autonomous driving has advanced along two closely related directions: richer scene representations and more structured planning interfaces. Our work lies at their intersection, using compact sparse scene tokens while addressing the limitations of direct regression by estimating costs over reachable ego trajectories.

## 2.1 Sparse and vectorized end-to-end planning.

E2E driving has moved from intermediate-representation and sensor-fusion models, such as DSDNet [32] and TransFuser [5], toward planning-oriented architectures such as UniAD [10] and DriveTransformer [12]. In parallel, vectorized and sparse representations have reduced the computation cost of dense BEV representations. MapTR introduced structured queries for online map reconstruction [19]; VAD, SparseDrive, and SparseAD extend this idea to end-to-end driving with compact map and agent queries [13, 24, 33]. While these methods improve scene encoding and planning eficiency, the final decision is commonly obtained through trajectory regression. They also often reason over marginal agent modes, which can fail to capture coherent scene understanding and complex interactive futures [23]. In contrast, our planner learns a cost landscape over reachable ego trajectories conditioned on joint scene tokens, so each action is scored against coherent multimodal scene hypotheses rather than independently selected futures.

## 2.2 Cost-based and safety-aware planning.

Cost-based planners provide an interpretable interface between perception and action by exposing a cost landscape that can be optimized or repaired. Neural Motion Planner [31], QUAD [1], and MP3 [4] estimate costs from BEV or mapconditioned features and score sampled ego trajectories, while ST-P3 learns a spatial-temporal BEV cost volume for multi-camera planning [9]. These works demonstrate the value of learned costs, but dense BEV cost supervision disperses learning over large regions that are irrelevant to the ego vehicle’s reachable motion. This can make optimization less stable, blur the distinction between reachable and unreachable costs, and has often led to weaker planning performance than direct regression.

In addition, cost learning is connected to Inverse Reinforcement Learning (IRL), which treats expert demonstrations as evidence of an underlying reward or cost function. Maximum-entropy IRL formalizes this idea by recovering costs whose induced trajectory distribution matches expert behavior [30], with later extensions to spatial-temporal costmaps [15,16,25]. However, these methods usually supervise costs indirectly through expert likelihood or policy matching, often requiring an inner planning or policy-inference loop. On the other hand, purely supervised trajectory regression has safety limitations: Wang et al. [26] show that distance-to-expert losses fail to separate safe and collision-prone alternatives. We instead learn a query-based cost landscape in which expert behavior, collision risk, and of-road violations supervise the ordering of reachable ego queries.

## 3 Methodology

## 3.1 Overview and Problem Formulation

Given a history of multi-view images $\mathcal { T } ~ = ~ \{ I _ { t , n } ~ \in ~ \mathbb { R } ^ { 3 \times H _ { 0 } \times W _ { 0 } } \} _ { t = - T _ { \mathrm { h i s t } } } ^ { 0 } , n ~ =$ $1 , \ldots , N _ { I }$ , the current ego state $\mathbf { x } _ { 0 } .$ , and a high-level navigation command $c ,$ the planner outputs a future ego trajectory $\hat { \tau } = \left\{ \hat { \mathbf { x } } _ { t } \right\} _ { t = 1 } ^ { T _ { \mathrm { p l a n } } }$ in the ego coordinate frame. Trajectory-regression planners learn this mapping directly, producing a small set of probabilistic ego futures. In contrast, we formulate planning as cost estimation over dynamically reachable ego trajectory queries. This preserves the interpretability of cost-based planning while avoiding dense supervision over regions the ego vehicle cannot execute.

Figure 1 summarizes the proposed architecture, which follows four stages. First, a sparse perception encoder extracts compact agent and map tokens from the image queue. Second, a joint scene prediction module summarizes multimodal agent futures into joint scene tokens $\begin{array} { r } { { \cal { S } } = \{ ( { \bf s } _ { j } , \pi _ { j } ) \} _ { j = 1 } ^ { N _ { s } } , } \end{array}$ , where ${ \bf s } _ { j }$ denotes a future scene-mode token, $\pi _ { j }$ its probability, and $N _ { s }$ the number of scene modes. Third, conditioned on the ego state $\mathbf { x } _ { \mathrm { 0 } }$ and navigation command $c ,$ a kinematic rollout sampler generates a clustered reachable query set $\mathcal { Q } = \{ ( \tau _ { i } , g _ { i } ) \} _ { i = 1 } ^ { N } ,$ , where $\tau _ { i }$ is a feasible ego trajectory and $g _ { i }$ denotes an intention cluster induced by the sampler. Each trajectory is then embedded as an ego query. Finally, the model predicts a bounded cost for each ego query, scene mode, and future timestep,

$$
\mathbf { C } _ { \theta } = \left[ C _ { \theta } ( \tau _ { i } , \mathbf { s } _ { j } , t ) \right] \in \left[ C _ { \operatorname* { m i n } } , C _ { \operatorname* { m a x } } \right] ^ { N \times N _ { s } \times T _ { \operatorname* { p l a n } } } .\tag{1}
$$

where $\theta$ denotes the learned model parameters. We refer to $\mathbf { C } _ { \theta }$ as a querybased cost estimate: rather than assigning costs to dense BEV cells, it defines a cost landscape over the reachable ego trajectory manifold. The final plan is obtained by aggregating costs over time and scene modes with a contingencyaware risk criterion, refining low-cost candidates through cost-guided trajectory mixing within intention clusters, and selecting the refined trajectory with the lowest resulting cost.

## 3.2 Vectorized Map and Agent Encoding

To eficiently encode the image queue I into compact map and agent features, we build on the SparseDrive image encoder and symmetric sparse perception module [24]. At each timestep, a ResNet-FPN [20] image encoder maps the multi-view images to multi-scale image features

$$
\mathcal { F } = \left\{ \mathbf { F } _ { r } \in \mathbb { R } ^ { N _ { I } \times D \times H _ { r } \times W _ { r } } \right\} _ { r = 1 } ^ { N _ { F } } ,\tag{2}
$$

where D is the feature dimension and $N _ { F }$ is the number of feature scales. The sparse perception module then learns two distinct instance sets. Static map elements are encoded as map features and polyline anchors (M, L), with M ∈ $\mathbb { R } ^ { N _ { m } \times D }$ and $\mathbf { L } \in \mathbb { R } ^ { N _ { m } \times P \times 2 }$ . Dynamic agents are encoded as agent features and 3D anchors $( \mathbf { A } , \mathbf { B } )$ , with $\mathbf { A } \ \doteq \ \mathbb { R } ^ { N _ { a } \times D }$ and $\mathbf { B } \in \mathbb { R } ^ { N _ { a } \times 1 1 }$ . Unlike BEV-based encoders [9], SparseDrive does not lift image features into a dense BEV volume. Instead, each sparse instance uses its anchor to gather image evidence from $\mathcal { F }$ through deformable feature aggregation. After confidence-based selection, the map tokens M and agent tokens A provide the initial vectorized context for the joint prediction and cost estimation modules.

## 3.3 Joint Scene Representation

Starting from the selected map and agent tokens, temporal propagation and attention-based interactions refine the features with latent temporal and local scene context. However, cost-based planning also requires compact hypotheses about how the scene may evolve. Predicting agents independently can produce futures that are individually likely but mutually inconsistent [28]. Accordingly, we propose $N _ { s }$ joint scene modes, each assigned a scene-level probability, so the cost estimator can reason over coherent scene contingencies rather than independent marginal futures.

Each scene mode is represented by a learnable embedding ${ \bf e } _ { j } \in \mathbb { R } ^ { D }$ shared across all agents. We form mode-conditioned agent tokens by adding this embedding to each selected agent token, $\tilde { \mathbf { a } } _ { a , j } = \mathbf { a } _ { a } + \mathbf { e } _ { j }$ , yielding $\tilde { \mathbf { A } } \in \breve { \mathbb { R } ^ { N _ { a } \times N _ { s } \times D } }$ A scene-level token is produced by cross-attending the mode embedding to its mode-conditioned agent tokens:

$$
\begin{array} { r } { { \bf s } _ { j } = \mathrm { F F N } \Big ( \mathrm { M H C A } \left( Q = { \bf e } _ { j } , { \cal K } = \{ { \tilde { \bf a } } _ { a , j } \} _ { a = 1 } ^ { N _ { a } } , { \cal V } = \{ { \tilde { \bf a } } _ { a , j } \} _ { a = 1 } ^ { N _ { a } } \right) \Big ) . } \end{array}\tag{3}
$$

The resulting token ${ \bf s } _ { j }$ summarizes one coherent future scene mode, and its probability is predicted by $\pi _ { j } ~ = ~ \mathrm { s o f t m a x } _ { j } ( \psi _ { \mathrm { s c e n e } } ( \mathbf { s } _ { j } ) )$ , where $\psi _ { \mathrm { s c e n e } }$ is a lightweight scene-logit head. To encourage these scene tokens to represent jointly consistent futures, the mode-conditioned agent tokens regress mode-specific agent trajectories, while the scene tokens predict the mixture weights over modes [22]. Let $\textbf { Y } = { \{ \mathbf { y } _ { a , t } \} }$ denote the future positions of the selected agents for $t \ =$ $1 , \ldots , T _ { \mathrm { p r e d } }$ . We model their joint future distribution as

$$
p ( \mathbf { Y } \mid \mathcal { T } ) = \sum _ { j = 1 } ^ { N _ { s } } \pi _ { j } \prod _ { a = 1 } ^ { N _ { a } } \prod _ { t = 1 } ^ { T _ { \mathrm { p r e d } } } \mathcal { N } \big ( \mathbf { y } _ { a , t } ; \mu _ { a , j , t } , \boldsymbol { \Sigma } _ { a , j , t } \big ) .\tag{4}
$$

The joint prediction branch is trained with the negative log-likelihood of Eq. 4, encouraging the scene-mode probabilities to assign mass to the hypotheses that best explain the observed joint future.

## 3.4 Reachable Ego Queries and Cost Estimation

Figure 2 details how joint scene tokens and reachable ego queries are combined to estimate and refine query costs. The previous modules provide a compact description of the current scene and its plausible joint futures. We build on this representation to define the planning interface itself: instead of regressing a few ego motions, we learn costs over explicit reachable ego queries. The ego branch builds a context token for the current decision by pooling an ego feature from the front-camera feature map and refining it through temporal attention, agent interaction, and cross-attention to selected map tokens. We then add a learned embedding of the navigation command c, yielding an ego context token $\mathbf { h } _ { e } \in \mathbb { R } ^ { D }$ that encodes the current ego state, scene context, and route intent.

![](images/78208be73768a6cac971e2dace0155447cc57f49ccca84ae86649fb115802388.jpg)  
Fig. 2: Detailed view of the proposed planner. Joint scene tokens encode multimodal futures, while the ego branch samples and encodes reachable trajectory queries. A lightweight head predicts scene-conditioned costs with contingency aggregation, optional inference-time external-cost repair, and cost-guided intra-cluster mixing.

Reachable Query Sampling. To avoid the diversity collapse often associated with top-k selection, we generate reachable queries with intention-aware clustering. Given the current ego state $\mathbf { x } _ { \mathrm { 0 } }$ and command $c ,$ a kinematic rollout sampler produces $\mathcal { Q } = \{ ( \tau _ { i } , g _ { i } ) \} _ { i = 1 } ^ { N }$ , where $\tau _ { i } = \{ \mathbf { x } _ { i , t } \} _ { t = 1 } ^ { T _ { \mathrm { p l a n } } }$ is a dynamically feasible ego trajectory and $g _ { i } \in \{ 1 , \ldots , N _ { g } \}$ denotes its intention cluster. The clusters preserve coverage over distinct maneuver families, while the samples within each cluster explore local motion variations. Each trajectory contributes its geometry and temporal structure through a coordinate and time embedding,

$$
\mathbf { r } _ { i , t } = \phi _ { \tau } ( \mathrm { P E } ( \mathbf { x } _ { i , t } ) ) + \gamma _ { t } ,\tag{5}
$$

where $\phi _ { \tau }$ is a lightweight projection and $\gamma _ { t }$ denotes the timestep embedding.

For cost estimation, the three inputs play complementary roles: $\mathbf { h } _ { e }$ provides the current ego-scene context, $\mathbf { r } _ { i , t }$ provides the spatial location and timing of a candidate ego motion, and ${ \bf s } _ { j }$ provides a compact hypothesis of the surrounding scene future. We combine them additively to form a cost query for trajectory $i ,$ scene mode $j ,$ and timestep t:

$$
{ \bf z } _ { i , j , t } ^ { \mathrm { c o s t } } = { \bf r } _ { i , t } + { \bf h } _ { e } + { \bf s } _ { j } .\tag{6}
$$

A lightweight cost head maps these queries to bounded costs,

$$
C _ { \theta } ( \tau _ { i } , \mathbf { s } _ { j } , t ) = C _ { \operatorname* { m i n } } + \sigma \big ( \psi _ { \mathrm { c o s t } } ( \mathbf { z } _ { i , j , t } ^ { \mathrm { c o s t } } ) \big ) ( C _ { \operatorname* { m a x } } - C _ { \operatorname* { m i n } } ) ,\tag{7}
$$

where $\psi _ { \mathrm { c o s t } }$ is implemented as a small prediction head shared across trajectory queries. This defines a query-based cost map only over executable ego trajectories, avoiding dense BEV cells unreachable by the vehicle.

Risk-aware Aggregation. The per-mode costs must be collapsed without prematurely committing to a single future scene. This is important in interactive settings, where early ego actions should remain safe under multiple plausible agent responses, while later costs can account for the probability of each contingency [17]. We therefore use a risk-aware contingency aggregation. Let $b = \lfloor T _ { b } / \varDelta t \rfloor$ denote the branching index, where $T _ { b }$ is the branching time in seconds and $\varDelta t$ is the planning timestep. Before this index, the planner uses the worst scene-mode cost; afterward, it follows the learned scene probabilities:

$$
\begin{array} { r } { \bar { C } _ { i , t } = \left\{ \begin{array} { l l } { \operatorname* { m a x } _ { j } C _ { \theta } ( \tau _ { i } , \mathbf { s } _ { j } , t ) , } & { t \leq b , } \\ { \sum _ { j = 1 } ^ { N _ { s } } \pi _ { j } C _ { \theta } ( \tau _ { i } , \mathbf { s } _ { j } , t ) , } & { t > b . } \end{array} \right. } \end{array}\tag{8}
$$

Cost-guided Intra-cluster Mixing. The aggregated costs define a local cost topology over the sampled reachable set. Pure sampling provides broad coverage, but the hard selection remains limited by the resolution of the sampled set and can therefore sufer from high displacement error. Selecting only the global lowest-cost samples can also collapse to a single maneuver family. We address these limitations with cost-guided intra-cluster mixing: intention clusters preserve global maneuver diversity, while the learned risk-aware costs refine each cluster toward its low-cost region.

Let $\mathcal { T } _ { q } = \{ i : g _ { i } = q \}$ denote the samples in intention cluster $q .$ . For mixing, we use a flexible cost profile $R _ { i , t }$ derived from $\bar { C } _ { i , t }$ . This profile uses average trajectory cost by default, with per-timestep cost as an alternative. Within each cluster, this profile is converted into MPPI-style weights [29],

$$
w _ { i , t } ^ { ( q ) } = \frac { \exp \bigl ( - \bigl ( R _ { i , t } - R _ { q , t } ^ { \operatorname* { m i n } } \bigr ) / \lambda \bigr ) } { \sum _ { m \in \mathcal { I } _ { q } } \exp \bigl ( - \bigl ( R _ { m , t } - R _ { q , t } ^ { \operatorname* { m i n } } \bigr ) / \lambda \bigr ) } , \qquad R _ { q , t } ^ { \operatorname* { m i n } } = \operatorname* { m i n } _ { m \in \mathcal { I } _ { q } } R _ { m , t } ,\tag{9}
$$

where $\lambda$ controls how strongly the refinement follows the lowest-cost samples. Given the sampled control sequence ${ \bf { u } } _ { i , t }$ associated with each trajectory, the refined control for cluster $q$ is computed directly as

$$
\tilde { \mathbf { u } } _ { q , t } = \sum _ { i \in \mathcal { I } _ { q } } w _ { i , t } ^ { ( q ) } \mathbf { u } _ { i , t } , \qquad \tilde { \tau } _ { q } = f _ { \mathrm { k i n } } \big ( \big \{ \tilde { \mathbf { u } } _ { q , t } \big \} _ { t = 1 } ^ { T _ { \mathrm { p l a n } } } \big ) ,\tag{10}
$$

where $f _ { \mathrm { k i n } }$ denotes the ego kinematic rollout. The resulting set $\{ \tilde { \tau } _ { q } \}$ retains one candidate per intention while using the learned safety-aware costs to refine controls beyond the original samples. The same weights define a cluster-desirability proxy for the mixed-trajectory cost,

$$
\tilde { C } _ { q } = \frac { 1 } { T _ { \mathrm { p l a n } } } \sum _ { t = 1 } ^ { T _ { \mathrm { p l a n } } } \sum _ { i \in \mathcal { I } _ { q } } w _ { i , t } ^ { ( q ) } R _ { i , t } .\tag{11}
$$

This proxy preserves a single diferentiable path through which trajectory ranking, classification, and regression shape sample costs and fusion, without rescoring the mixed trajectory. For final candidate selection and classification supervision, we use the negative cluster costs as logits, $p _ { q } ^ { \mathrm { e g o } } =$ softma $\mathfrak { c } _ { q } \big ( - \tilde { C } _ { q } \big )$ , such that lower-cost refined candidates receive higher probability.

## 3.5 Cost Ranking and Candidate Supervision

During training, we retain the standard SparseDrive losses for map and object detection and supervise the joint scene modes with the negative log-likelihood in Eq. 4. We then add a planner-specific objective for the query-based cost estimation. This planning objective supervises two complementary parts of the planner: the cost topology over reachable queries and the final candidate selected after intra-cluster refinement. We propose the following planning loss as

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { p l a n } } = \lambda _ { \mathrm { r a n k } } \mathcal { L } _ { \mathrm { r a n k } } + \lambda _ { \mathrm { c l s } } \mathcal { L } _ { \mathrm { c l s } } + \lambda _ { \mathrm { r e g } } \mathcal { L } _ { \mathrm { r e g } } . } \end{array}\tag{12}
$$

Here, $\mathcal { L } _ { \mathrm { r a n k } }$ trains the cost map before final selection, while $\mathcal { L } _ { \mathrm { c l s } }$ and $\mathcal { L } _ { \mathrm { r e g } }$ supervise the refined candidates $\{ \tilde { \tau } _ { q } \} _ { q = 1 } ^ { N _ { g } }$ <sub>1</sub>.

Safety-aware cost ranking. Imitation losses based only on expert distance may assign similar supervision to safe alternatives and unsafe hard negatives [26]. We therefore train the learned costs to separate the expert trajectory $\tau ^ { \star }$ from sampled reachable trajectories using a novel safety-aware margin. The expert trajectory is queried only during training, not at inference. Unlike ST-P3- or NMP-style objectives [9, 31], which mainly rank candidates through expert distance, our margin uses collision and boundary violations to determine the required separation between learned costs. For a sampled trajectory $\tau _ { i }$ , let $C ^ { \star }$ and $C _ { i }$ denote the predicted costs of the expert and sampled trajectory under the chosen cost aggregation. The ranking violation and loss are

$$
v _ { i } = \left[ C ^ { \star } - C _ { i } + m _ { i } \right] _ { + } , \qquad \mathcal { L } _ { \mathrm { r a n k } } = \frac { 1 } { | \mathcal { K } _ { h } | } \sum _ { i \in \mathcal { K } _ { h } } v _ { i } .\tag{13}
$$

where $m _ { i } = m ( d _ { i } , \rho _ { i } ^ { \mathrm { c o l } } , \rho _ { i } ^ { \mathrm { b d } } )$ is a bounded margin that grows with the distance $d _ { i }$ to the expert and increases sharply when $\tau _ { i }$ collides or leaves the drivable boundary. Intuitively, the loss enforces $C _ { i } \geq C ^ { \star } + m _ { i } \colon$ candidates farther from the expert require a higher cost, and collision or boundary violations increase the margin even when the trajectory remains geometrically close to the expert. Rather than using only the single largest violation, $\kappa _ { h }$ denotes the indices of the top- $\boldsymbol { \cdot } \boldsymbol { k } _ { h }$ violations in a scene. This spreads gradients over informative negatives, producing a smoother topology and providing more stable guidance for the costweighted cluster mixing.

Candidate selection and refinement. After intra-cluster mixing, each intention cluster yields one refined candidate $\tilde { \tau } _ { q }$ . The classification loss encourages the cluster-cost logits to select the candidate closest to the expert, while the regression loss pulls the selected candidate toward the expert trajectory. Together, the ranking loss shapes the intra-cluster cost landscape used for mixing, and the classification and regression losses preserve accurate inter-cluster selection while allowing the refined candidates to continuously cover the expert distribution.

## 3.6 Inference-time Cost Repair

At inference time, we propose a repair mechanism that incorporates external safety or rule signals without retraining the model. Since our planner predicts bounded costs over explicit ego queries, these signals can directly override unsafe regions of the learned cost landscape. Let $E _ { i , t } \in [ 0 , 1 ]$ denote an external penalty for trajectory $\tau _ { i }$ at timestep $t ,$ obtained, for example, from a trafic-rule monitor or vehicle-to-everything (V2X) trafic-light data. We perform cost repair by saturating externally unsafe query states toward the maximum cost,

$$
\bar { C } _ { i , t } ^ { \mathrm { r e p a i r } } = \left( 1 - E _ { i , t } \right) \bar { C } _ { i , t } + E _ { i , t } C _ { \operatorname* { m a x } } .\tag{14}
$$

The repaired costs are then used in the same cluster-wise mixing described above. Thus, trajectories marked unsafe by external signals receive low mixing weights, and intention clusters whose trajectories are consistently penalized obtain high cluster costs and are unlikely to be selected. This gives the planner a practical repair mechanism for correcting learned costs at deployment while preserving the reachable-query structure.

## 4 Experimental Setup

Dataset and protocol. We evaluate on the nuScenes dataset [3], which contains 1000 driving scenes with six surround-view cameras and keyframe annotations at 2Hz. Using the train/validation split, the model observes a queue of four multi-view frames and produces a 3s future ego plan for open-loop evaluation. For zero-shot transfer, we use a subset of 465 driving logs from FZI-AURA [21], each lasting 20s, totaling 2.58 hours of driving, about 3× the duration of the nuScenes validation split. The logs were collected with the CoCar NextGen research vehicle [7] in Germany across diferent times of day, lighting conditions, and trafic densities. To reduce domain shift, we select six surround-view cameras and process the logs following a nuScenes-inspired protocol with manually annotated 3D bounding boxes and semantic segmentation labels. The vehicle uses a sensor rig similar in layout but not identical to nuScenes, resulting in slightly diferent camera intrinsics and extrinsics.

Implementation details. Our model uses SparseDrive-S [24] as the sparse perception backbone with a ResNet-50 FPN image encoder, 900 agent anchors, 100 map anchors, and 256-dimensional instance features. All variants of our model are trained with AdamW using a learning rate of $3 \times 1 0 ^ { - 4 }$ , weight decay of $1 0 ^ { - 3 }$ , and an efective batch size of 48 on two NVIDIA RTX 6000 Ada GPUs. The query-cost planner uses $N _ { s } = 6$ joint scene modes, N = 512 sampled ego trajectories, and $N _ { g } = 1 6$ intention clusters with 32 samples per cluster. Costs are bounded to [0, 100], contingency aggregation uses a $T _ { b }$ of 1s, and the safetyranking loss averages the $\mathrm { t o p } { - } k _ { h } = 3 2$ violations.

Metrics and baselines. We report displacement error and collision rate at 1s, 2s, and 3s, together with their averages. For zero-shot real-world transfer, we additionally report candidate diversity for all models. Diversity is measured by the mean pairwise average distance and endpoint distance among candidate trajectories, capturing separation between maneuver intentions. For variants of our end-to-end model, we also report cost AUC and cost gap, which measure how well the learned costs separate unsafe candidates from safe ones. Unsafe candidates are those that collide or leave the drivable area. An AUC of 0.5 corresponds to random ordering and 1.0 to perfect separation. The cost gap is the diference between the mean cost of unsafe candidates and the mean cost of safe candidates; larger positive values indicate stronger safety-aligned separation.

## 5 Evaluation

## 5.1 Comparison to State of the Art

Table 1 compares open-loop planning performance on nuScenes. Compared with prior cost-based planners, our method substantially improves both displacement error and collision rate. Our query cost representation concentrates learning on reachable ego motions, the safety-margin loss separates unsafe hard negatives, and cost-guided cluster mixing converts the learned topology into accurate final plans. Against ST-P3, the average L2 error drops from 2.11m to 0.71m, while the average collision rate drops from 0.71% to 0.07%. This closes much of the gap to recent regression-based planners while preserving the advantages of an explicit cost interface. Notably, our method achieves a lower average collision rate than SparseDrive-S and remains close to the larger SparseDrive-B variant, while also outperforming most regression baselines such as VAD [13] and Drive-Transformer [12], suggesting that learning costs over reachable queries provides a useful balance between accurate imitation and safety-aware planning.

## 5.2 Zero-shot Real-world Transfer

Table 2 evaluates zero-shot transfer to real-world driving logs collected in Germany, without fine-tuning any method on this data. Alpamayo 1.5 [27] is a large vision-language driving model trained on substantially broader data, so we use it as a strong imitation-oriented reference rather than a lightweight planner baseline. Compared with SparseDrive-S, our method reduces the average L2 distance from 4.22m to 2.31m and the average collision rate from 2.73% to 0.68%. Alpamayo achieves the lowest displacement error, but uses a 10B-parameter model and requires 1.87s per frame on an NVIDIA RTX 3090 GPU. In contrast, our 86.2M-parameter planner achieves the lowest collision rate, runs in real time, and maintains higher candidate diversity for downstream repair and selection. These results suggest that reachable-query cost learning provides a favorable balance between safety, diversity, and eficiency under real-world domain shift.

Table 1: Open-loop planning performance on nuScenes under the UniAD metrics. † denotes LiDAR-based methods.
<table><tr><td rowspan="2">Method</td><td colspan="3">L2 (m) ↓</td><td colspan="2">Collision (%)↓</td><td colspan="2"></td></tr><tr><td>1s</td><td>2s 3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s Avg.</td></tr><tr><td colspan="7">Cost-based methods</td></tr><tr><td>NMP [31]†</td><td></td><td>2.31</td><td></td><td></td><td>1.92</td><td></td></tr><tr><td>SA-NMP [31]† ST-P3 [9]</td><td></td><td>2.05</td><td>2.11</td><td>0.23 0.62</td><td>1.59 1.27</td><td></td></tr><tr><td>Ours</td><td>1.33 2.11 0.28 0.65</td><td>2.90 1.21</td><td>0.71</td><td>0.00 0.02</td><td>0.18</td><td>0.71 0.07</td></tr><tr><td colspan="7"></td></tr><tr><td>Regression-based methods FF [8]†</td><td>1.20</td><td>2.54</td><td>1.43</td><td>0.06 0.17</td><td>1.07</td><td>0.43</td></tr><tr><td>EO [14]†</td><td>0.55 0.67 1.36</td><td>2.78</td><td>1.60</td><td>0.04 0.09</td><td>0.88</td><td>0.33</td></tr><tr><td>UniAD [10]</td><td>0.48 0.74</td><td>1.07</td><td>0.76</td><td>0.12 0.13</td><td>0.28</td><td>0.17</td></tr><tr><td>VAD-Tiny [13]</td><td>0.46</td><td></td><td>0.78</td><td>0.21 0.35</td><td>0.58</td><td>0.38</td></tr><tr><td></td><td>0.76</td><td>1.12</td><td>0.72</td><td>0.07 0.17</td><td></td><td></td></tr><tr><td>VAD-Base [13]</td><td>0.41 0.70</td><td>1.05</td><td></td><td></td><td>0.41</td><td>0.22</td></tr><tr><td>BEVPlanner [18]</td><td>0.27 0.54</td><td>0.90</td><td>0.57</td><td></td><td></td><td></td></tr><tr><td>DriveTransformer [12]</td><td>0.19 0.34</td><td>0.66</td><td>0.40</td><td>0.03 0.10</td><td>0.21</td><td>0.11</td></tr><tr><td>SparseDrive-S [24]</td><td>0.29 0.58</td><td>0.96</td><td>0.61</td><td>0.01</td><td>0.05 0.18</td><td>0.08</td></tr><tr><td>SparseDrive-B [24]</td><td>0.29 0.55</td><td>0.91</td><td>0.58</td><td>0.01 0.02</td><td>0.13</td><td>0.06</td></tr></table>

Table 2: Zero-shot real-world transfer. No method is fine-tuned on the real-world logs. Latency is measured on a workstation with a RTX 3090 GPU.
<table><tr><td rowspan="2">Method</td><td rowspan="2"></td><td colspan="4">L2 (m) ↓</td><td colspan="4">Collision (%)↓</td><td colspan="2">Diversity (m) ↑</td><td rowspan="2">Params.</td><td rowspan="2">Latency (ms) ↓</td></tr><tr><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>Avg. Endpoint</td><td></td></tr><tr><td>SparseDrive-S [24]</td><td></td><td>2.38</td><td>4.17</td><td>6.11</td><td>4.22</td><td>0.53</td><td>2.57</td><td>5.08</td><td>2.73</td><td>1.77</td><td>3.41</td><td>85.9M</td><td>115</td></tr><tr><td>Alpamayo 1.5 [27]</td><td></td><td>0.64</td><td>1.41</td><td>2.39</td><td>1.48</td><td>0.03</td><td>0.90</td><td>1.79</td><td>0.87</td><td>1.47</td><td>3.25</td><td>10.0B</td><td>1878</td></tr><tr><td>Ours</td><td></td><td>0.85</td><td>2.12</td><td>3.98</td><td>2.31</td><td>0.01</td><td>0.40</td><td>1.50</td><td>0.68</td><td>5.54</td><td>11.88</td><td>86.2M</td><td>104</td></tr></table>

## 5.3 Safety Supervision and Cost Topology

We next isolate how the safety margin shapes the learned cost topology. Table 3 first studies how the collision term in the margin is computed during training. All variants use the same architecture and cost planner at inference; only the collision signal used to compute the training margin changes. At test time, the margin itself is not required, since its efect is absorbed into the learned cost topology. Ground-truth collision provides an oracle target, while top-1 predicted collision supervises the margin using only the most likely future scene. In contrast, contingency collision risk is computed over multimodal predicted scene futures, allowing lower-probability but safety-critical futures to influence the learned costs.

The results show a tradeof between accuracy and safety. Top-1 predicted collision achieves the lowest displacement error, suggesting that supervising with the most likely predicted future is well aligned with imitation accuracy, but it also gives the highest average collision rate. Using predicted contingency risk slightly increases L2 error, but reduces the average collision rate from 0.12% to

Table 3: Ablation of collision supervision for the safety margin on nuScenes.
<table><tr><td rowspan="2">Collision supervision</td><td colspan="4">L2 (m) ↓</td><td colspan="4">Collision (%)↓</td></tr><tr><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><td>Ground-truth collision</td><td>0.27</td><td>0.63</td><td>1.16</td><td>0.69</td><td>0.00</td><td>0.06</td><td>0.25</td><td>0.10</td></tr><tr><td>Top-1 predicted collision</td><td>0.25</td><td>0.59</td><td>1.10</td><td>0.64</td><td>0.02</td><td>0.07</td><td>0.26</td><td>0.12</td></tr><tr><td>Contingency collision risk 0.28</td><td></td><td>0.65</td><td>1.21</td><td>0.71</td><td>0.00</td><td>0.02</td><td>0.18</td><td>0.07</td></tr></table>

0.07%. This indicates that multimodal collision supervision encourages the latent scene tokens to encode collision-relevant futures and yields a cost topology that is better aligned with safety-aware planning.

Having chosen contingency collision risk as the default supervision, Tables 4 and 5 evaluate whether the resulting cost topology is specific to the proposed ranking objective. All variants use the same architecture, reachable-query planner, candidate losses, and inference procedure; only $\mathcal { L } _ { \mathrm { r a n k } }$ is replaced. We first compare against ranking losses adapted from NMP [31] and ST-P3 [9], and then ablate the sources used to form our safety margin.

Table 4 shows that the choice of cost supervision mainly changes the safety structure of the learned topology. The ST-P3-style loss achieves the lowest nuScenes L2 loss, but its weaker cost separation results in a higher collision rate than the proposed margin. In contrast, our safety-margin objective improves nuScenes collision from 0.12% to 0.07% relative to the ST-P3-style loss, while increasing cost AUC from 0.79 to 0.86 and cost gap from 7.80 to 45.70. The same trend holds under real-world transfer: the proposed objective yields the lowest collision rate and the largest cost gap, indicating that the learned safety ordering generalizes beyond the training domain.

Table 4: Comparison of cost-supervision objectives.
<table><tr><td rowspan="2">Objective Lrank</td><td colspan="4">nuScenes</td><td colspan="4">Real-world</td></tr><tr><td>L2↓</td><td></td><td></td><td>Coll. ↓ Cost AUC ↑ Cost Gap ↑</td><td>L2 ↓</td><td>Coll. ↓</td><td></td><td>Cost AUC ↑ Cost Gap ↑</td></tr><tr><td>NMP-style loss [31]</td><td>0.76</td><td>0.33</td><td>0.70</td><td>6.60</td><td>2.29</td><td>0.84</td><td>0.59</td><td>3.99</td></tr><tr><td>ST-P3-style loss [9]</td><td>0.61</td><td>0.12</td><td>0.79</td><td>7.80</td><td>3.26</td><td>0.83</td><td>0.61</td><td>4.24</td></tr><tr><td>Proposed safety margin</td><td>0.71</td><td>0.07</td><td>0.86</td><td>45.70</td><td>2.31</td><td>0.68</td><td>0.63</td><td>16.18</td></tr></table>

Table 5 further decomposes the proposed margin. This evaluates the contribution of expert-distance and safety terms, and whether combining them yields a cost topology that is both safety-aware and useful for final planning. The distance-only margin keeps the trajectories close to the expert but leaves many unsafe hard negatives weakly separated, resulting in higher collision rates on both nuScenes and real-world logs. Using only collision and of-road terms produces the strongest raw separation, but it also degrades displacement and does not translate into the best final collision rate. Combining distance with collision and of-road terms gives the best planning tradeof: the learned costs remain anchored to expert-like reachable motions while still penalizing unsafe alternatives, yielding the lowest collision rate in both domains.

Table 5: Ablation of safety-margin sources.
<table><tr><td rowspan="2">Margin source</td><td colspan="4">nuScenes</td><td colspan="4">Real-world</td></tr><tr><td>L2↓</td><td>Coll. ↓</td><td></td><td>Cost AUC ↑ Cost Gap ↑ | L2 ↓</td><td></td><td></td><td></td><td>Coll. ↓ Cost AUC ↑ Cost Gap ↑</td></tr><tr><td>Distance</td><td>0.72</td><td>0.20</td><td>0.71</td><td>10.51</td><td>2.36</td><td>1.71</td><td>0.56</td><td>4.89</td></tr><tr><td>Collision + off-road</td><td>0.88</td><td>0.11</td><td>0.87</td><td>51.25</td><td>2.34</td><td>1.48</td><td>0.66</td><td>23.57</td></tr><tr><td>All</td><td>0.71</td><td>0.07</td><td>0.86</td><td>45.70</td><td>2.31</td><td>0.68</td><td>0.63</td><td>16.18</td></tr></table>

## 5.4 Qualitative Results of External Cost Repair

Figure 3 illustrates the inference-time cost repair introduced in Section 3.6. As an example of external costs, we use the conflict area of a red trafic light received from V2X infrastructure. SparseDrive produces a low-diversity proposal set in which all candidates enter this area (Fig. 3b). Since the proposals already violate the rule, post-hoc rescoring alone cannot recover a rule-compliant trajectory. Our planner instead applies the external penalty to the aggregated query-cost profile before intra-cluster mixing. This changes the cost topology used to form the final candidates, suppressing unsafe queries and producing safe low-cost candidates, highlighted in green in Fig. 3c, that respect the trafic-light rule.

## 5.5 Qualitative Robustness to Misaligned Navigation Commands

Figure 4 qualitatively evaluates the behavior of the planners under a temporally misaligned high-level navigation command. The scene contains an S-bend before a highway merge: the route will soon require a right turn, but the vehicle must first follow a left-hand bend bounded by lane dividers. Thus, the command is not semantically wrong, but executing the right-turn intent immediately would cut into the divider. SparseDrive prematurely shifts its selected trajectory toward the upcoming right maneuver and collides with the divider, as seen in Fig. 4b. This reflects a limitation of command-conditioned trajectory regression: the generated proposal follows the route intent in geometry, but does not explicitly optimize a local road-compliance cost.

Our planner samples command-conditioned reachable trajectories with the kinematic rollout sampler and evaluates them using the learned query-cost landscape. Trajectories that conflict with the current lane geometry receive high learned cost, so the selected trajectory follows the low-cost, road-compliant option through the left bend before later committing to the rightward route intent, as illustrated in Fig. 4c. This example shows that query-based cost estimation can use navigation commands while still selecting the safest locally feasible trajectory in the candidate set.

![](images/368d209a109461b9d0b4b2cf2bf0bfa7513942b824effed8e8257f2c9f831254.jpg)  
(a) Camera view

![](images/d49216639ab76d8dd0c30ef1fdf65657689fc4364f8832affefedd606d38cb7a.jpg)  
(b) SparseDrive BEV

![](images/e1011f412b1a283bae1dd790ffa68d8a69b474dbd9d61a27031c680a9497a8ce.jpg)  
(c) Ours BEV

Fig. 3: Qualitative comparison using the conflict area of a red trafic light received from a V2X unit. SparseDrive’s low-diversity proposals already violate the rule, so post-hoc rescoring cannot recover a rule-compliant trajectory. Our planner applies the external penalty to the aggregated query-cost profile, producing safe low-cost candidates.  
![](images/29772d0a59a421ad8937322f7ae16d655916ff0ecde17ffee3348cc1286794f6.jpg)  
(a) Projected final trajectory candidates

![](images/64ee16053760221476318ba64117444f149fefd5b651435f5403534c9df00f9a.jpg)  
(b) SparseDrive BEV

![](images/f3547ad0844c330e4cd28b15c4a2377a1c06c3622fb6a75a23fba621c067842e.jpg)  
(c) Ours BEV  
Fig. 4: Qualitative example under a temporally premature right-turn command. Panel (a) shows the final selected trajectory candidates projected into the camera views, with ours in green and SparseDrive in red. SparseDrive follows the premature command into the lane divider, while our planner selects the locally safe left bend.

## 6 Conclusion

This work revisits cost-based planning as an end-to-end driving interface by learning bounded costs over dynamically reachable ego trajectories rather than dense BEV grids. This shifts supervision toward actions the vehicle can actually execute, while retaining a cost representation that can be inspected, optimized, and repaired. Compact joint scene tokens, safety-aware ranking, and cost-guided cluster mixing support reasoning over coherent future contingencies and produce safe ego plans. On nuScenes, the approach substantially improves over previous cost-based planners and remains competitive with strong regression-based approaches. Zero-shot transfer to FZI-AURA demonstrates efectiveness without fine-tuning, while the cost interface provides a mechanism for inference-time repair using external safety signals.

## Acknowledgements

The research leading to these results is funded by the German Federal Ministry for Economic Afairs and Energy within the project “NXT GEN AI METHODS – Generative Methoden für Perzeption, Prädiktion und Planung". The authors would like to thank the consortium for the successful cooperation. The authors acknowledge support by the state of Baden-Württemberg through bwHPC.

## References

1. Biswas, S., Casas, S., Sykora, Q., Agro, B., Sadat, A., Urtasun, R.: Quad: Querybased interpretable neural motion planning for autonomous driving. In: 2024 IEEE International Conference on Robotics and Automation (ICRA). pp. 14236–14243. IEEE (2024)

2. Bojarski, M., Yeres, P., Choromanska, A., Choromanski, K., Firner, B., Jackel, L., Muller, U.: Explaining how a deep neural network trained with end-to-end learning steers a car. arXiv preprint arXiv:1704.07911 (2017)

3. Caesar, H., Bankiti, V., Lang, A.H., Vora, S., Liong, V.E., Xu, Q., Krishnan, A., Pan, Y., Baldan, G., Beijbom, O.: nuscenes: A multimodal dataset for autonomous driving. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 11621–11631 (2020)

4. Casas, S., Sadat, A., Urtasun, R.: Mp3: A unified model to map, perceive, predict and plan. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 14398–14407 (2021)

5. Chitta, K., Prakash, A., Jaeger, B., Yu, Z., Renz, K., Geiger, A.: Transfuser: Imitation with transformer-based sensor fusion for autonomous driving. IEEE Transactions on Pattern Analysis and Machine Intelligence 45(11), 12878–12895 (2022)

6. Fawcett, T.: An introduction to roc analysis. Pattern recognition letters 27(8), 861–874 (2006)

7. Heinrich, M., Zipfl, M., Uecker, M., Ochs, S., Gontscharow, M., Fleck, T., Doll, J., Schörner, P., Hubschneider, C., Zofka, M.R., et al.: Cocar nextgen: a multipurpose platform for connected autonomous driving research. In: 2024 IEEE 27th International Conference on Intelligent Transportation Systems (ITSC). pp. 482– 489. IEEE (2024)

8. Hu, P., Huang, A., Dolan, J., Held, D., Ramanan, D.: Safe local motion planning with self-supervised freespace forecasting. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 12732–12741 (2021)

9. Hu, S., Chen, L., Wu, P., Li, H., Yan, J., Tao, D.: St-p3: End-to-end vision-based autonomous driving via spatial-temporal feature learning. In: European Conference on Computer Vision. pp. 533–549. Springer (2022)

10. Hu, Y., Yang, J., Chen, L., Li, K., Sima, C., Zhu, X., Chai, S., Du, S., Lin, T., Wang, W., et al.: Planning-oriented autonomous driving. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 17853– 17862 (2023)

11. Hubschneider, C., Bauer, A., Weber, M., Zöllner, J.M.: Adding navigation to the equation: Turning decisions for end-to-end vehicle control. In: 2017 IEEE 20th international conference on intelligent transportation systems (ITSC). pp. 1–8. IEEE (2017)

12. Jia, X., You, J., Zhang, Z., Yan, J.: Drivetransformer: Unified transformer for scalable end-to-end autonomous driving. arXiv preprint arXiv:2503.07656 (2025)

13. Jiang, B., Chen, S., Xu, Q., Liao, B., Chen, J., Zhou, H., Zhang, Q., Liu, W., Huang, C., Wang, X.: Vad: Vectorized scene representation for eficient autonomous driving. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 8340–8350 (2023)

14. Khurana, T., Hu, P., Dave, A., Ziglar, J., Held, D., Ramanan, D.: Diferentiable raycasting for self-supervised occupancy forecasting. In: European Conference on Computer Vision. pp. 353–369. Springer (2022)

15. Lee, K., Isele, D., Theodorou, E.A., Bae, S.: Risk-sensitive mpcs with deep distributional inverse rl for autonomous driving. In: 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). pp. 7635–7642. IEEE (2022)

16. Lee, K., Isele, D., Theodorou, E.A., Bae, S.: Spatiotemporal costmap inference for mpc via deep inverse reinforcement learning. IEEE Robotics and Automation Letters 7(2), 3194–3201 (2022)

17. Li, T., Zhang, L., Liu, S., Shen, S.: Multimodal integrated prediction and decisionmaking with adaptive interaction modality explorations. In: 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). pp. 13674–13681. IEEE (2025)

18. Li, Z., Yu, Z., Lan, S., Li, J., Kautz, J., Lu, T., Alvarez, J.M.: Is ego status all you need for open-loop end-to-end autonomous driving? In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 14864– 14873 (2024)

19. Liao, B., Chen, S., Wang, X., Cheng, T., Zhang, Q., Liu, W., Huang, C.: Maptr: Structured modeling and learning for online vectorized hd map construction. arXiv preprint arXiv:2208.14437 (2022)

20. Lin, T.Y., Dollár, P., Girshick, R., He, K., Hariharan, B., Belongie, S.: Feature pyramid networks for object detection. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. pp. 2117–2125 (2017)

21. Polley, R., Heinrich, M., Schörner, P., Uecker, M., Ochs, S., Fleck, T., Zofka, M.R., Zöllner, J.M.: FZI-AURA: A Multimodal Autonomous Driving Dataset (2026), https://huggingface.co/datasets/fzi-forschungszentrum-informatik/FZI-AURA

22. Shi, S., Jiang, L., Dai, D., Schiele, B.: Mtr++: Multi-agent motion prediction with symmetric scene modeling and guided intention querying. IEEE Transactions on Pattern Analysis and Machine Intelligence 46(5), 3955–3971 (2024)

23. Sun, Q., Huang, X., Gu, J., Williams, B.C., Zhao, H.: M2i: From factored marginal trajectory prediction to interactive prediction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 6543–6552 (2022)

24. Sun, W., Lin, X., Shi, Y., Zhang, C., Wu, H., Zheng, S.: Sparsedrive: End-to-end autonomous driving via sparse scene representation. In: 2025 IEEE International Conference on Robotics and Automation (ICRA). pp. 8795–8801. IEEE (2025)

25. Triest, S., Castro, M.G., Maheshwari, P., Sivaprakasam, M., Wang, W., Scherer, S.: Learning risk-aware costmaps via inverse reinforcement learning for of-road navigation. In: 2023 IEEE International Conference on Robotics and Automation (ICRA). pp. 924–930. IEEE (2023)

26. Wang, J., Hua, Z., Liu, X., Xing, Z., Tian, H., Ma, K., Ye, H., Chen, G., Chen, L., Zhang, Q.: Beyond imitation: Learning safe end-to-end autonomous driving from hard negatives. arXiv preprint arXiv:2605.19771 (2026)

27. Wang, Y., Luo, W., Bai, J., Cao, Y., Che, T., Chen, K., Chen, Y., Diamond, J., Ding, Y., Ding, W., et al.: Alpamayo-r1: Bridging reasoning and action prediction for generalizable autonomous driving in the long tail. arXiv preprint arXiv:2511.00088 (2025)

28. Weng, E., Hoshino, H., Ramanan, D., Kitani, K.: Joint metrics matter: A better standard for trajectory forecasting. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 20315–20326 (2023)

29. Williams, G., Aldrich, A., Theodorou, E.A.: Model predictive path integral control: From theory to parallel computation. Journal of Guidance, Control, and Dynamics 40(2), 344–357 (2017)

30. Wulfmeier, M., Ondruska, P., Posner, I.: Maximum entropy deep inverse reinforcement learning. arXiv preprint arXiv:1507.04888 (2015)

31. Zeng, W., Luo, W., Suo, S., Sadat, A., Yang, B., Casas, S., Urtasun, R.: End-to-end interpretable neural motion planner. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8660–8669 (2019)

32. Zeng, W., Wang, S., Liao, R., Chen, Y., Yang, B., Urtasun, R.: Dsdnet: Deep structured self-driving network. In: European Conference on Computer Vision. pp. 156–172. Springer (2020)

33. Zhang, D., Wang, G., Zhu, R., Zhao, J., Chen, X., Zhang, S., Gong, J., Zhou, Q., Zhang, W., Wang, N., et al.: Sparsead: Sparse query-centric paradigm for eficient end-to-end autonomous driving. arXiv preprint arXiv:2404.06892 (2024)

## Supplementary Material for Beyond Waypoint Regression: Query-Based Cost Learning over Reachable Ego Futures for End-to-End Driving

## A Additional Method Details

## A.1 Reachable Query Sampling

The proposed method uses a clustered set of reachable ego trajectories $\mathcal { Q } =$ $\{ ( \tau _ { i } , g _ { i } ) \} _ { i = 1 } ^ { N }$ as the queries of the learned cost landscape. Here, we describe how these queries are generated before cost evaluation. The goal is to cover distinct maneuver intentions while keeping every query tied to a physically executable control sequence. Following sampling-based cost planners such as NMP [31] and ST-P3 [9], we use sampled ego trajectories as the support for cost estimation, but organize them as command-conditioned reachable control clusters. The sampler first defines coarse reference controls for a set of intention clusters, then samples temporally smooth control perturbations around these references.

Given the current ego speed $v _ { 0 }$ and navigation command $^ { c , }$ we first define a uniform grid of $N _ { g }$ reference intentions over target final speeds and heading changes. The target final speeds $\{ v _ { \mathrm { t a r } } ^ { q } \}$ span a bounded interval starting from zero up to a command-independent speed limit around the current velocity. The heading changes $\{ \varDelta \psi _ { q } \}$ are command-conditioned:

$$
\begin{array} { r } { \varDelta \psi _ { q } \in \left\{ \begin{array} { l l } { [ - \psi _ { \operatorname* { m a x } } , 0 ] , } & { c = \mathrm { r i g h t } , } \\ { [ 0 , \psi _ { \operatorname* { m a x } } ] , } & { c = \mathrm { l e f t } , } \\ { [ - \psi _ { \operatorname* { m a x } } / 2 , \psi _ { \operatorname* { m a x } } / 2 ] , } & { c = \mathrm { s t r a i g h t } . } \end{array} \right. } \end{array}\tag{15}
$$

Each grid point defines a reference intention. Rather than using it as a geometric anchor, we convert it into constant reference controls over the planning horizon:

$$
a _ { q } ^ { 0 } = \frac { v _ { \mathrm { t a r } } ^ { q } - v _ { 0 } } { T _ { \mathrm { p l a n } } \varDelta t } , \qquad \omega _ { q } ^ { 0 } = \frac { \varDelta \psi _ { q } } { T _ { \mathrm { p l a n } } \varDelta t } , \qquad \mathbf { u } _ { q , t } ^ { 0 } = ( \omega _ { q } ^ { 0 } , a _ { q } ^ { 0 } ) .\tag{16}
$$

Thus, each cluster represents a coarse control intention, such as maintaining speed, slowing down, or turning with diferent heading changes.

To improve low-speed coverage, the sampler also includes one auxiliary slowmotion cluster. In total, intra-cluster mixing operates over $\tilde { N } _ { g } = N _ { g } + 1$ clusters. To explore local variations around each intention, we draw $N _ { p } - 1$ perturbed control sequences around each cluster control reference and use the auxiliary cluster for the remaining low-speed samples, giving $N = N _ { g } N _ { p }$ total queries. For trajectory i assigned to cluster $g _ { i } ,$ , the sampled control at timestep t is

$$
\mathbf { u } _ { i , t } = \mathbf { u } _ { g _ { i } , t } ^ { 0 } + \epsilon _ { i , t } , \qquad \mathbf { u } _ { i , t } = ( \omega _ { i , t } , a _ { i , t } ) .\tag{17}
$$

We use temporally correlated perturbations instead of independent per-step noise, which produces smoother control sequences and avoids unrealistically jittery trajectories:

$$
\begin{array} { r } { \epsilon _ { i , t } = \alpha \epsilon _ { i , t - 1 } + \sqrt { 1 - \alpha ^ { 2 } } \sigma \odot \mathbf { z } _ { i , t } , \qquad \alpha = \exp ( - \Delta t / t _ { c } ) , } \end{array}\tag{18}
$$

where $\mathbf z _ { i , t } \sim \mathcal { N } ( \mathbf 0 , \mathbf I )$ , σ sets the yaw-rate and acceleration noise scales, and $t _ { c }$ controls the correlation time.

Finally, each sampled control sequence is rolled out with the kinematic model,

$$
\begin{array} { r } { \tau _ { i } = f _ { \mathrm { k i n } } \left( \left\{ \mathbf { u } _ { i , t } \right\} _ { t = 1 } ^ { T _ { \mathrm { p l a n } } } \right) , \qquad \tau _ { i } = \left\{ \mathbf { x } _ { i , t } \right\} _ { t = 1 } ^ { T _ { \mathrm { p l a n } } } . } \end{array}\tag{19}
$$

For clarity, the rollout can be written as a unicycle update for $t = 1 , \ldots , T _ { \mathrm { p l a n } } ,$ initialized at the current ego state:

$$
v _ { i , t } = v _ { i , t - 1 } + a _ { i , t } \varDelta t , \qquad \psi _ { i , t } = \psi _ { i , t - 1 } + \omega _ { i , t } \varDelta t ,\tag{20}
$$

$$
\mathbf { x } _ { i , t } = \mathbf { x } _ { i , t - 1 } + v _ { i , t - 1 } \left[ \cos \psi _ { i , t - 1 } \right] { \varDelta t } + \frac { 1 } { 2 } a _ { i , t } \left[ \cos \psi _ { i , t - 1 } \right] { \varDelta t } ^ { 2 } .\tag{21}
$$

This corresponding closed-form control integration for nonzero yaw rate is used. The resulting query set is

$$
\boldsymbol { \mathcal { Q } } = \{ ( \tau _ { i } , \mathbf { u } _ { i } , g _ { i } ) \} _ { i = 1 } ^ { N } , \qquad N = N _ { g } N _ { p } , \qquad g _ { i } \in \{ 1 , \ldots , \tilde { N } _ { g } \} ,\tag{22}
$$

which is passed to the query-cost planner.

Figure 5 visualizes the resulting command-conditioned query sets for representative straight, left, and right commands from the nuScenes dataset.

## A.2 Cost-Guided Intra-cluster Mixing

Intra-cluster mixing can be written as a weighted average of sampled controls. Here, we give a more detailed view, which is equivalent to the main paper but makes the MPPI structure explicit. After contingency aggregation, each sampled trajectory has a per-timestep cost profile $\bar { C } _ { i , t }$ and belongs to a mixing cluster $\mathcal { I } _ { q } = \{ i : g _ { i } = q \}$ , with $q = 1 , \ldots , \tilde { N } _ { q }$ . For mixing, we define a score profile $R _ { i , t }$ from the learned costs. In the trajectory-level variant, the same score is used at every timestep,

$$
R _ { i , t } = \frac { 1 } { T _ { \mathrm { p l a n } } } \sum _ { t ^ { \prime } = 1 } ^ { T _ { \mathrm { p l a n } } } \bar { C } _ { i , t ^ { \prime } } , \qquad \forall t ,\tag{23}
$$

whereas in the timestep-level variant we use $R _ { i , t } = \bar { C } _ { i , t }$ directly. The first option follows the overall quality of each sampled trajectory, while the second recomputes weights at each timestep, so diferent samples can dominate diferent parts of the fused control sequence.

Within each cluster, this score profile is converted into the same MPPI-style softmax weights as defined in the main paper, with the cluster-wise minimum score subtracted for numerical stability. The temperature λ controls the sharpness of the update: smaller values concentrate the refinement around the lowestcost samples, while larger values average over more of the local cost topology.

![](images/5ca69f39eb6130f200c37480d0c3a2b50d0d16f877a23443e95996cf7d5a1daf.jpg)  
(c) Right command  
Fig. 5: Command-conditioned reachable query sampling. The kinematic sampler generates clustered, dynamically executable ego trajectories that cover the diverse intentions implied by the navigation command while preserving local control variations within each cluster.

Rather than averaging trajectory coordinates, our method fuses control perturbations around the reference control of each cluster, then rolls out the fused controls with the ego kinematic model to keep the refined candidate feasible.

$$
\varDelta \mathbf { u } _ { i , t } = \mathbf { u } _ { i , t } - \mathbf { u } _ { g _ { i } , t } ^ { 0 }\tag{24}
$$

denote its deviation from the reference control of its intention cluster. The refined control sequence for cluster $q$ is then

$$
\tilde { \mathbf { u } } _ { q , t } = \mathbf { u } _ { q , t } ^ { 0 } + \sum _ { i \in \mathcal { I } _ { q } } w _ { i , t } ^ { ( q ) } \varDelta \mathbf { u } _ { i , t } .\tag{25}
$$

Because the weights sum to one within each cluster and timestep, this residual form is equivalent to directly averaging the sampled controls with the same MPPI weights. The residual form simply highlights that the update is performed around the command-conditioned control intention. Finally, the refined candidate is obtained by rolling out the fused controls,

$$
\tilde { \tau } _ { q } = f _ { \mathrm { k i n } } \left( \{ \tilde { \mathbf { u } } _ { q , t } \} _ { t = 1 } ^ { T _ { \mathrm { p l a n } } } \right) .\tag{26}
$$

This produces one continuous candidate per intention cluster, preserving maneuver diversity while using the learned cost topology to reduce the discretization error of the sampled query set.

## A.3 Cost-Guided Inference and Repair Algorithm

Algorithm 1 summarizes how predicted query costs become a selected ego plan. Contingency aggregation combines scene-mode costs, and optional external penalties modify them before fusion. Mean trajectory costs determine within-cluster weights used to mix controls, roll out candidates, and compute cluster-desirability scores for selection. $T = T _ { \mathrm { p l a n } } , b = \lfloor T _ { b } / \varDelta t \rfloor$ , and $E _ { i , t } = 0$ disables external repair.

```latex
Algorithm 1 Cost aggregation, repair, and intra-cluster fusion
Input: Sampled controls ${ \bf { u } } _ { i , t } ,$ cluster IDs $g _ { i } ,$ query costs $C _ { i , j , t } ,$ scene probabilities $\pi _ { j } ,$
optional penalties $E _ { i , t }$
Parameters: Horizon T, branch index $b ,$ temperature $\lambda > 0 ,$ cost bound $C _ { \mathrm { m a x } }$
Output: Selected plan $\hat { \tau }$ and candidates $\{ ( \tilde { \tau } _ { q } , \tilde { C } _ { q } ) \}$
1: for each query i and timestep $t = 1 , \dots , T$ do
2: $\begin{array} { r } { \bar { C } _ { i , t } \gets \left\{ \underset { - } { \operatorname* { m a x } } _ { j } C _ { i , j , t } , \quad t \leq b , \right. } \end{array}$
$\begin{array} { r } { \mathbb { U } \mathbb { \Lambda } ^ {  } \bigcup _ { j } \pi _ { j } C _ { i , j , t } , \quad t > b } \end{array}$
3: $\bar { C } _ { i , t } \gets ( 1 - E _ { i , t } ) \bar { C } _ { i , t } + E _ { i , t } C _ { \operatorname* { m a x } }$
4: end for
5: $\begin{array} { r } { R _ { i }  T ^ { - 1 } \sum _ { t = 1 } ^ { T } \bar { C } _ { i , t } } \end{array}$ for every query $_ { i }$
6: for each nonempty cluster $q$ do
7: $\begin{array} { r } { \mathcal { I } _ { q }  \{ i : g _ { i } = q \} , \quad R _ { q } ^ { \mathrm { m i n } }  \mathrm { m i n } _ { i \in \mathcal { I } _ { q } } R _ { i } } \end{array}$
8: w<sub>i</sub> (q) ← $\begin{array} { r } { \overline { { \sum _ { m \in \mathcal { I } _ { q } } \exp ( - ( R _ { m } - R _ { q } ^ { \operatorname* { m i n } } ) / \lambda ) } } } \end{array}$ $\mathrm { e x p } ( - ( R _ { i } - R _ { q } ^ { \mathrm { m i n } } ) / \lambda )$ $i \in \mathcal { I } _ { q }$
9: $\begin{array} { r } { \tilde { \mathbf { u } } _ { q , t }  \sum _ { i \in \mathcal { I } _ { q } } w _ { i } ^ { ( q ) } \mathbf { u } _ { i , t } } \end{array}$ for all t
10: $\tilde { \tau } _ { q } \gets f _ { \mathrm { k i n } } ( \{ \tilde { \mathbf { u } } _ { q , t } \} _ { t = 1 } ^ { T } )$
11: $\begin{array} { r } { \tilde { C } _ { q } \gets \sum _ { i \in \mathcal { I } _ { q } } w _ { i } ^ { ( q ) } R _ { i } } \end{array}$
12: end for
13: $q ^ { \star } \gets \arg \operatorname* { m i n } _ { q } \tilde { C } _ { q }$
14: return $\hat { \tau } = \tilde { \tau } _ { q ^ { \star } } , \{ ( \tilde { \tau } _ { q } , \tilde { C } _ { q } ) \}$
```

The rollout uses the sampler’s initialization and kinematic model. We fuse controls rather than trajectory coordinates, and use the cluster-desirability proxy ${ \tilde { C } } _ { q }$ without re-evaluating fused trajectories through the learned cost head. External penalties therefore influence both candidate construction and selection without retraining. The expert trajectory is used only during training.

## A.4 Network Architecture Details.

We provide the implementation details of the lightweight cost head used to score the reachable ego queries. Given trajectory embeddings $\mathbf { r } _ { i , t }$ , ego context $\mathbf { h } _ { e } .$ , and scene tokens $\{ \mathbf { s } _ { j } \} _ { j = 1 } ^ { N _ { s } }$ , the model forms the cost-query tensor by broadcasting

$$
\begin{array} { r } { \mathbf { z } ^ { \mathrm { c o s t } } \in \mathbb { R } ^ { B \times N \times N _ { s } \times T _ { \mathrm { p l a n } } \times D } , \qquad \mathbf { z } _ { i , j , t } ^ { \mathrm { c o s t } } = \mathbf { r } _ { i , t } + \mathbf { h } _ { e } + \mathbf { s } _ { j } . } \end{array}
$$

The shared cost head is a two-layer MLP applied to every trajectory, scene mode, and timestep:

$$
\begin{array} { r } { \psi _ { \mathrm { c o s t } } ( \mathbf { z } ) = \mathbf { W } _ { 2 } \mathrm { R e L U } ( \mathbf { W } _ { 1 } \mathbf { z } + \mathbf { b } _ { 1 } ) + \mathbf { b } _ { 2 } . } \end{array}
$$

Its scalar output is mapped to $[ C _ { \mathrm { m i n } } , C _ { \mathrm { m a x } } ]$ using the bounded sigmoid cost mapping described in the method; in our experiments, we set $C _ { \mathrm { m i n } } = 0$ and $C _ { \mathrm { m a x } } = 1 0 0$

## B Additional Experimental Setup Details

## B.1 Real World Driving Logs Details.

The real-world evaluation set contains 465 driving logs from FZI-AURA collected with the CoCar NextGen research vehicle in Germany across diferent times of day, lighting conditions, and trafic densities. Each log has a duration of 20s, resulting in 2.58 h of raw driving data and 93,000 raw samples at 10 Hz. The logs cover diferent locations, times of day, lighting conditions, and trafic densities. The logs include synchronized camera images, ego poses, calibrated sensor transformations, and both 3D bounding-box annotations and semantic labels at 2Hz keyframes.

We match the nuScenes planning protocol and evaluate on 2Hz keyframes. Across the 465 logs, the dataset contains 725,459 annotated 3D boxes. The labeled keyframes contain an average of 40 boxes, a median of 33, and a range of 1 to 415. The dominant categories are cars, bicycles, persons, and trucks. Frames near the end of a log are excluded from full-horizon planning metrics when the required 3 s of future trajectory are unavailable.

## B.2 SparseDrive-format preprocessing.

SparseDrive and our method are evaluated with six surround-view cameras, mapped to the nuScenes-style camera slots. The five forward and side-facing cameras have a native resolution of $1 9 2 0 \times 1 2 0 0$ , while the rear wide-angle camera has a native resolution of $2 5 9 2 \times 2 0 4 8$ . All image inputs are resized and cropped to $7 0 4 \times 2 5 6$ pixels.

To reduce visual domain shift caused by diferent camera mounting heights and fields of view, we use a fixed vertical crop ofset. This crop choice aligns the horizon and road layout more closely with the nuScenes image distribution. Since SparseDrive is trained in the nuScenes top-lidar reference convention, we define a virtual top-lidar frame on the research vehicle to provide the same planner coordinate interface. This frame follows the nuScenes axis convention, with the x axis pointing right and the y axis pointing forward; the translation from the virtual top-lidar frame to the vehicle base frame is (0.94, 0.00, 1.84) m.

High-level navigation commands are generated from the recorded future ego motion using the same virtual frame. For each evaluated frame, we transform the ego displacement after 3 s into the virtual top-lidar frame and apply the SparseDrive lateral-threshold rule. Displacements greater than 2 m are labeled right, displacements below −2 m are labeled left, and the remaining frames are labeled straight. This yields a command distribution of 8% left, 83.5% straight, and 8.5% right. For Alpamayo, we convert the same command labels into text prompts so both methods receive the same high-level intent.

## B.3 Alpamayo Inference Details.

We evaluate Alpamayo 1.5 [27] as a large imitation-oriented vision-language driving reference rather than as a lightweight real-time planner baseline. Alpamayo is evaluated using only the front-medium camera. It receives four image history frames and 16 ego-history states sampled at 10 Hz, corresponding to 1.6 s of egostate history. In contrast, SparseDrive and our method receive four multi-view frames at 2 Hz. The same navigation-command heuristic is used for Alpamayo as for the SparseDrive-format methods, with the discrete labels converted into natural-language instructions: “Turn left”, “Go straight”, or “Turn right”. Alpamayo inference is run on the 10 Hz raw logs with stride 5, giving the same 2 Hz evaluation rate as the labeled keyframes.

Predicted trajectories are generated for a 3 s horizon and exported into the same SparseDrive-style result format used by the other methods. Alpamayo 1.5 can sample one or more future trajectories, but the released model does not expose a model-defined trajectory-level cost, confidence, likelihood, or score for ranking those samples. Therefore, L2 and collision are computed using the first decoded trajectory. For candidate-diversity evaluation, we collect 16 sampled trajectories per frame. The samples are generated sequentially due to GPU memory limitations. For the real-world transfer table in the main paper, latency is measured with a single generated trajectory per frame.

## B.4 Evaluation Metric Details.

We report the standard open-loop L2 and collision metrics at 1s, 2s, and 3s. Unless otherwise stated, these metrics are computed on the final selected trajectory for each planner. SparseDrive selects the trajectory with the highest planning score, whereas our method selects the refined candidate with the lowest aggregated query cost. Following the uniAD [10] protocol, each reported collision value is cumulative over the 0.5s evaluation steps up to the corresponding horizon. Collision is evaluated by intersecting the predicted ego footprint with future annotated 3D bounding boxes in the ego-aligned evaluation frame. Map-based road-compliance signals are used outside the reported open-loop L2 and collision metrics. On nuScenes, boundary violations are used during training to construct the safety-aware ranking margin: a violation is counted when the predicted ego trajectory, evaluated with footprint ofsets, crosses drivable-area boundary polylines extracted from lane and road-segment polygons. On the real-world logs, where the same HD-map protocol is not available, we build an accumulated semantic-lidar road-support map for qualitative visualization and sanity checks.

Additionally, we report candidate-level metrics for diversity and learned cost separation. For a fixed scene s, let $\{ \tilde { \tau } _ { q } \} _ { q = 1 } ^ { K _ { s } }$ denote the refined candidate trajectories, where $\tilde { \tau } _ { q } = \{ \tilde { \mathbf { x } } _ { q , t } \} _ { t = 1 } ^ { T _ { \mathrm { p l a n } } }$

Candidate diversity. We measure candidate diversity as the pairwise displacement between candidates produced for the same scene. We report both an average-displacement version and an endpoint-displacement version:

$$
\begin{array} { r l } & { \mathrm { D i v } _ { \mathrm { A v g } } ( s ) = \cfrac { 2 } { K _ { s } ( K _ { s } - 1 ) } \displaystyle \sum _ { q < q ^ { \prime } } \frac { 1 } { T _ { \mathrm { p l a n } } } \displaystyle \sum _ { t = 1 } ^ { T _ { \mathrm { p l a n } } } \| \tilde { \mathbf { x } } _ { q , t } - \tilde { \mathbf { x } } _ { q ^ { \prime } , t } \| _ { 2 } , } \\ & { \mathrm { D i v } _ { \mathrm { E n d p o i n t } } ( s ) = \cfrac { 2 } { K _ { s } ( K _ { s } - 1 ) } \displaystyle \sum _ { q < q ^ { \prime } } \| \tilde { \mathbf { x } } _ { q , T _ { \mathrm { p l a n } } } - \tilde { \mathbf { x } } _ { q ^ { \prime } , T _ { \mathrm { p l a n } } } \| _ { 2 } . } \end{array}
$$

We average both values over scenes. $\mathrm { D i v _ { A v g } }$ captures the average spread of the candidates over the full planning horizon, while $\mathrm { D i v } _ { \mathrm { E n d p o i n t } }$ emphasizes endpoint separation and more directly reflects diversity between maneuver intentions.

Cost separation. For cost-based variants, we evaluate whether the learned costs separate unsafe candidates from safe candidates. A candidate is unsafe if it collides with another actor or leaves the drivable area at any valid timestep. Let $\mathcal { U } _ { s }$ and $\mathcal { R } _ { s }$ denote the unsafe and safe refined-candidate index sets in scene $s ,$ respectively, and let ${ \tilde { C } } _ { q }$ denote the final cluster cost of refined candidate $\tilde { \tau } _ { q } .$ Following the standard probabilistic interpretation of ROC-AUC [6], cost AUC measures how often an unsafe candidate is ranked above a safe candidate. Since higher costs correspond to less desirable trajectories, it estimates $\begin{array} { r } { \operatorname* { P r } ( \tilde { C } _ { u } > \tilde { C } _ { r } ) + \frac { 1 } { 2 } \operatorname* { P r } ( \tilde { C } _ { u } = \tilde { C } _ { r } ) } \end{array}$ , where $u \sim \mathcal { U } _ { s }$ and $r \sim \mathcal { R } _ { s }$ . For scenes containing at least one safe and one unsafe candidate, we compute this empirical AUC using the following ranking form:

$$
\mathrm { A U C } _ { s } = \frac { 1 } { \lvert \mathscr { U } _ { s } \rvert \lvert \mathscr { R } _ { s } \rvert } \sum _ { u \in \mathscr { U } _ { s } } \sum _ { r \in \mathscr { R } _ { s } } \left[ \mathbb { 1 } _ { \{ \tilde { C } _ { u } > \tilde { C } _ { r } \} } + \frac { 1 } { 2 } \mathbb { 1 } _ { \{ \tilde { C } _ { u } = \tilde { C } _ { r } \} } \right] .
$$

The equality term follows the standard half-credit convention for ties in $\mathrm { A U C } ;$ exact equality is typically rare for continuous learned costs, but the term makes the metric well-defined for bounded or numerically equal scores. Thus, an AUC of 0.5 indicates random ordering, while 1.0 indicates perfect separation. We also report the cost gap,

$$
\mathrm { G a p } _ { s } = \frac { 1 } { \left| \mathcal { U } _ { s } \right| } \sum _ { u \in \mathcal { U } _ { s } } \tilde { C } _ { u } - \frac { 1 } { \left| \mathcal { R } _ { s } \right| } \sum _ { r \in \mathcal { R } _ { s } } \tilde { C } _ { r } .
$$

Larger positive gaps indicate that unsafe candidates receive substantially higher costs than safe candidates on average. The reported AUC and gap are averaged over scenes where both safe and unsafe candidates exist.

## C Additional Experiments

## C.1 Branching Time.

Table 6 evaluates the branching time that separates worst-case from probabilityweighted scene-cost aggregation. At $T _ { b } = 0$ , probability-weighted aggregation is used throughout, while at $T _ { b } ~ = ~ 3 \mathrm { s }$ , worst-case aggregation covers the full planning horizon. Pure expectation can underweight low-probability, high-cost futures, whereas a long worst-case interval can favor conservative responses even when the corresponding scene mode is unlikely.

The default $T _ { b } = 1 \mathrm { s }$ achieves both the lowest average L2 error (0.71 m) and the lowest average collision rate (0.07%) among the ablated branching times. Extending the worst-case interval to two or three seconds degrades both metrics, so more conservative cost aggregation does not automatically yield fewer collisions. These results support a short worst-case prefix for near-term decisions, followed by probability-weighted costs that distinguish the likelihood of later outcomes.

Table 6: Ablation of Branching-time on nuScenes.
<table><tr><td rowspan="2"> $T _ { b }$  (s)</td><td colspan="4">L2 (m) ↓</td><td colspan="3">Collision (%)↓</td></tr><tr><td>1s</td><td>2s</td><td>3s</td><td> $\operatorname { A v g } .$ </td><td>1s</td><td>2s 3s</td><td>Avg.</td></tr><tr><td>0</td><td>0.33</td><td>0.80</td><td>1.50</td><td>0.88</td><td>0.02</td><td>0.14 0.35</td><td>0.17</td></tr><tr><td>1</td><td>0.28</td><td>0.65</td><td>1.21</td><td>0.71</td><td>0.00</td><td>0.02 0.18</td><td>0.07</td></tr><tr><td>2</td><td>0.30</td><td>0.69</td><td>1.28</td><td>0.76</td><td>0.02</td><td>0.07 0.29</td><td>0.13</td></tr><tr><td>3</td><td>0.32</td><td>0.78</td><td>1.47</td><td>0.86</td><td>0.03</td><td>0.16 0.37</td><td>0.19</td></tr></table>

## C.2 Plan Selection and Rescoring.

Table 7 compares global lowest-cost selection, best-per-cluster selection, MPPI mixing, and MPPI mixing with learned rescoring of the fused trajectories. Global selection chooses the lowest-cost sample without preserving a candidate for each intention, while best-per-cluster selection maintains a representative sample from each cluster. MPPI instead uses the learned safety-aware costs to mix controls within each intention cluster and rolls out the fused controls with the kinematic model, allowing refinement beyond individual samples without averaging distinct maneuver families.

Compared with global selection, MPPI reduces average L2 from 0.77 to 0.71 m and collision from 0.13% to 0.07%. Best-per-cluster selection achieves lower L2 (0.68 m), but its higher collision rate (0.11%) highlights the separation between matching the expert trajectory and avoiding collisions. The mixing results therefore support using the learned cost topology for safety-aware refinement rather than only selecting fixed samples.

The rescoring experiment tests whether evaluating the fused trajectory with the learned cost head improves selection over the cluster-desirability proxy. The proxy combines sample costs using the fusion weights, whereas rescoring directly evaluates the resulting trajectory. Rescoring increases average L2 from 0.71 to 0.73 m and average collision from 0.07% to 0.15% relative to the default. Empirical analysis shows that the cluster-desirability proxy is strongly correlated with the learned cost evaluated directly on the mixed trajectory, with Pearson and Spearman correlation coeficients of 0.921 and 0.923, respectively.

Table 7: Ablation of Plan selection and rescoring on nuScenes.
<table><tr><td rowspan="2">Variant</td><td colspan="4">L2 (m) ↓</td><td colspan="4">Collision (%) ↓</td></tr><tr><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><td>Global selection</td><td>0.29</td><td>0.70</td><td>1.30</td><td>0.77</td><td>0.02</td><td>0.07</td><td>0.31</td><td>0.13</td></tr><tr><td>Best per cluster</td><td>0.26</td><td>0.63</td><td>1.14</td><td>0.68</td><td>0.01</td><td>0.05</td><td>0.28</td><td>0.11</td></tr><tr><td>MPPI</td><td>0.28</td><td>0.65</td><td>1.21</td><td>0.71</td><td>0.00</td><td>0.02</td><td>0.18</td><td>0.07</td></tr><tr><td>MPPI + rescoring</td><td>0.28</td><td>0.67</td><td>1.22</td><td>0.73</td><td>0.01</td><td>0.12</td><td>0.31</td><td>0.15</td></tr></table>

## C.3 Planning Loss Components.

Table 8 ablates the planner-specific losses while keeping the same reachablequery architecture. The safety margin corresponds to $\mathcal { L } _ { \mathrm { r a n k } }$ , while candidate classification and candidate regression correspond to ${ \mathcal L } _ { \mathrm { c l s } }$ and $\mathcal { L } _ { \mathrm { r e g } }$ of the training objective ${ \mathcal { L } } _ { \mathrm { p l a n } }$ . The safety-margin ranking loss alone already learns a meaningful safe-versus-unsafe ordering, as reflected by a cost AUC of 0.81 and a cost gap of 36.17, but the final plan remains limited by weak cluster-level supervision. Adding candidate classification improves displacement error by encouraging the planner to select a more expert-like intention cluster, although this alone does not improve collision. Adding candidate regression reduces collision relative to the safety-margin-only variant, but without the classification loss, it does not reliably select the best cluster and therefore hurts displacement. Using all three losses gives the best overall behavior: the ranking loss shapes the cost topology, classification improves inter-cluster selection, and regression refines the selected candidate geometry.

Table 8: Ablation of planning loss components on nuScenes.
<table><tr><td rowspan="2">Variant</td><td colspan="4">L2 (m) ↓</td><td colspan="4">Collision (%) ↓</td><td rowspan="2"></td><td rowspan="2">Cost AUC ↑ Cost Gap ↑</td></tr><tr><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><td>Safety margin</td><td>0.35 0.89</td><td></td><td>1.72</td><td>0.99</td><td>0.00</td><td>0.151</td><td>0.41</td><td>0.19</td><td>0.81</td><td>36.17</td></tr><tr><td>+ candidate classification</td><td></td><td>0.34 0.82</td><td>1.54</td><td>0.90</td><td>0.04</td><td>0.19</td><td>0.44</td><td>0.22</td><td>0.81</td><td>38.64</td></tr><tr><td>+ candidate regression</td><td></td><td>0.38 1.00</td><td>1.91</td><td>1.10</td><td>0.04</td><td>0.11</td><td>0.31</td><td>0.15</td><td>0.82</td><td>35.76</td></tr><tr><td>+ classification + regression</td><td></td><td>0.28 0.65</td><td>1.21</td><td>0.71</td><td>0.00</td><td>0.02</td><td>0.18</td><td>0.07</td><td>0.86</td><td>45.70</td></tr></table>

## C.4 Hard-Negative Aggregation.

Table 9 studies two choices in the safety-ranking loss: the collision signal used to define the margin, and how hard negatives are aggregated. The max variant backpropagates only through the single largest ranking violation in each scene, while the Top-k variant averages the largest $k _ { h }$ violations. Thus, max aggregation focuses on the hardest negative but can produce sparse gradients, whereas Top-k spreads supervision over multiple unsafe or near-unsafe candidates.

For ground-truth collision supervision, Top-k aggregation consistently improves the learned cost topology and final planning, reducing L2 from 0.76 to 0.69 and increasing the cost gap from 41.67 to 46.66. A similar trend appears for contingency collision risk, where Top-k aggregation reduces the average collision rate from 0.13% to 0.07% and improves both cost AUC and cost gap. This suggests that when the collision signal is reliable or contingency-aware, spreading gradients across several hard negatives yields a smoother and more useful cost topology than optimizing only the worst single violation.

The top-1 predicted collision signal behaves diferently: max aggregation yields a lower collision rate than Top-k, likely because a single predicted future provides a narrower, noisier safety target. In that setting, averaging many hard negatives can overemphasize candidates that are unsafe only under the most likely predicted future, rather than under the broader multimodal risk. Overall, the best result is obtained by combining contingency collision risk with Top-k aggregation, supporting our default choice of supervising the ranking loss with multiple informative negatives under multimodal scene risk.

Table 9: Ablation of hard-negative aggregation for safety-margin training on nuScenes.
<table><tr><td rowspan="2">Collision signal</td><td rowspan="2">Negative agg.</td><td colspan="4">L2 (m) ↓</td><td colspan="4">Collision (%) ↓</td><td rowspan="2">Cost AUC ↑ Cost Gap ↑</td><td rowspan="2"></td></tr><tr><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><td>Ground-truth collision</td><td>Max</td><td>0.29</td><td>0.69</td><td>1.30</td><td>0.76</td><td>0.01</td><td>0.06</td><td>0.27</td><td>0.11</td><td>0.85</td><td>41.67</td></tr><tr><td>Ground-truth collision</td><td>Top-k</td><td>0.27</td><td>0.63</td><td>1.16</td><td>0.69</td><td>0.00</td><td>0.06</td><td>0.25</td><td>0.10</td><td>0.86</td><td>46.66</td></tr><tr><td>Top-1 predicted collision</td><td>Max</td><td>0.27</td><td>0.62</td><td>1.16</td><td>0.68</td><td>0.00</td><td>0.04</td><td>0.19</td><td>0.08</td><td>0.88</td><td>42.56</td></tr><tr><td>Top-1 predicted collision</td><td>Top-k</td><td>0.27</td><td>0.61</td><td>1.12</td><td>0.67</td><td>0.04</td><td>0.12</td><td>0.23</td><td>0.13</td><td>0.87</td><td>45.71</td></tr><tr><td>Contingency collision risk</td><td>Max</td><td>0.28</td><td>0.66</td><td>1.23</td><td>0.72</td><td>0.01</td><td>0.10</td><td>0.28</td><td>0.13</td><td>0.84</td><td>41.22</td></tr><tr><td>Contingency collision risk</td><td>Top-k</td><td>0.28</td><td>0.65</td><td>1.21</td><td>0.71</td><td>0.00</td><td>0.02</td><td>0.18</td><td>0.07</td><td>0.86</td><td>45.70</td></tr></table>

## C.5 Cost Profile for Cluster Mixing.

Table 10 evaluates which learned cost profile is used to compute the MPPIstyle cluster weights. The mean trajectory cost first averages the cost over the horizon and uses a single score for each sampled trajectory, giving the refinement a trajectory-level preference. The per-timestep cost instead computes weights independently at each future step, which is more local but can combine controls from diferent samples at diferent timesteps.

The two variants obtain the same cost AUC, indicating a similar safe-versusunsafe ordering. The diference appears in how this topology is used for refinement: the mean trajectory cost reduces the average L2 from 0.77 to 0.71 and the average collision from 0.14% to 0.07%. This suggests that intra-cluster mixing benefits from a temporally consistent trajectory-level preference. Per-timestep weighting is more flexible, but in a finite sampled cluster, it can assemble a control sequence from samples that are locally low-cost at diferent times but do not form a coherent maneuver. We therefore use the mean trajectory cost as the default profile for cost-guided intra-cluster mixing.

Table 10: Ablation of cost profiles for cost-guided cluster mixing on nuScenes.
<table><tr><td rowspan="3">Mixing score</td><td colspan="4">L2 (m) ↓</td><td colspan="4">Collision (%)↓</td><td rowspan="3">Cost AUC ↑ Cost Gap ↑</td><td rowspan="3"></td></tr><tr><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><td>Mean trajectory cost</td><td>0.28</td><td>0.65</td><td>1.21</td><td>0.71</td><td>0.00</td><td>0.02</td><td>0.18</td><td>0.07</td><td>0.86</td><td>45.70</td></tr><tr><td>Per-timestep cost</td><td>0.30</td><td>0.71</td><td>1.32</td><td>0.77</td><td>0.02</td><td>0.10</td><td>0.30</td><td>0.14</td><td>0.86</td><td>42.99</td></tr></table>

## C.6 Scene-Cost Aggregation.

Table 11 ablates how the scene-mode costs are aggregated before cluster mixing. Top-1 aggregation uses only the most likely scene mode, worst-case aggregation takes the maximum over scene modes, mean aggregation treats all modes uniformly, and probability-weighted aggregation follows the predicted scene probabilities. The contingency strategy combines these views temporally: it uses worstcase costs before the branching time and probability-weighted costs afterward.

The aggregation strategy has a clear efect on both planning accuracy and safety. Top-1 and probability-weighted aggregation commit strongly to likely scene modes and can therefore underemphasize low-probability but safety-critical futures. Worst-case aggregation avoids this optimism, but applying the maximum over all scene modes for the full horizon can be overly conservative and does not improve collision in this setting. Uniform mean aggregation provides a moderate compromise, reducing collision relative to Top-1 and worst-case aggregation, but it ignores the diferent roles of early and late planning timesteps.

The contingency aggregation performs best across all metrics. By using worstcase costs before the branching time, it keeps early ego actions safe under multiple plausible futures. After the branching time, probability-weighted aggregation allows the planner to account for the likelihood of each scene mode. This temporal structure improves average L2 from 0.85–0.88 to 0.71 and reduces collision from 0.13–0.19% to 0.07%, while producing the strongest cost separation.

Table 11: Ablation of scene-cost aggregation on nuScenes.
<table><tr><td rowspan="2">Aggregation</td><td colspan="4">L2 (m) ↓</td><td colspan="4">Collision (%)↓</td><td rowspan="2">Cost AUC ↑ Cost Gap ↑</td><td rowspan="2"></td></tr><tr><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><td>Top-1</td><td>0.33</td><td>0.79</td><td>1.50</td><td>0.87</td><td>0.03</td><td>0.17</td><td>0.37</td><td>0.19</td><td>0.81</td><td>33.99</td></tr><tr><td>Worst Case</td><td>0.32</td><td>0.78</td><td>1.47</td><td>0.86</td><td>0.03</td><td>0.16</td><td>0.37</td><td>0.19</td><td>0.81</td><td>37.81</td></tr><tr><td>Mean</td><td>0.32</td><td>0.77</td><td>1.45</td><td>0.85</td><td>0.00</td><td>0.09</td><td>0.30</td><td>0.13</td><td>0.82</td><td>38.97</td></tr><tr><td>Probability-weighted</td><td>0.33</td><td>0.80</td><td>1.50</td><td>0.88</td><td>0.02</td><td>0.14</td><td>0.35</td><td>0.17</td><td>0.81</td><td>37.97</td></tr><tr><td>Contingency</td><td>0.28</td><td>0.65</td><td>1.21</td><td>0.71</td><td>0.00</td><td>0.02</td><td>0.18</td><td>0.07</td><td>0.86</td><td>45.70</td></tr></table>

## D Additional Qualitative Results

## D.1 nuScenes Examples.

Figure 6 shows representative scenes covering car following, yielding, lane-change, and overtaking behavior from nuScenes. These examples complement the quantitative results by illustrating how the reachable-query planner evaluates a diverse set of ego futures in interactive trafic scenes and selects low-cost candidates that remain consistent with the local road geometry and surrounding agents.

## D.2 Zero-shot Real-World Examples

Figure 7 shows representative zero-shot transfer examples on our real-world driving logs, covering stop-and-go trafic, interaction with vulnerable road users, intersection turning, and highway following. These examples complement the quantitative real-world evaluation by illustrating that the learned cost topology remains meaningful under a diferent sensor setup, camera domain, and road layout. Each row shows the surround camera inputs, the BEV cost visualization over top-k reachable ego candidates, and the ground-truth ego trajectory together with semantic map support. In the middle BEV panel, the selected final trajectory candidate is highlighted in dark green with a black outline.

![](images/37d4d2275250dc6df3763983448eca8a1ea00e220b76e8d830d57bd3a7532897.jpg)  
(a) Following a leading vehicle

![](images/2e783c724e2f37a38770dad85ff7c0e20affae15cdaa4d6eafc180853fce0310.jpg)

![](images/83ec99ca90b6f5cefb95b5acd5164d3ba2e03b8709c924a209053d5e3a276a2e.jpg)  
(b) Yielding to oncoming trafic

![](images/c4a70f6241c034788c4a6881727a6b22d3a874ca73633a26b809acc4eee50f08.jpg)

![](images/c3b6c5d9fb936c9035b876d36aa464c117567ce3246b618edc9761217795767a.jpg)  
(c) Lane change Maneuver

![](images/dce3d874be49b41bfb7286492bd29f3a66675f4ff3600cefbf7a7df1dc2744f2.jpg)

![](images/aa85038431f9c4cafbb131c95f173fc22f2c9b0136fdd457ca4edc862bb4ad67.jpg)  
(d) Overtaking Maneuver

![](images/b3eab27644e90f5ae6eafc444c13a383402d02013c1b5a2e649b0543eefdd955.jpg)  
Fig. 6: Qualitative nuScenes examples. The camera views and BEV visualizations show representative interactive scenes in which the learned cost topology guides selection among reachable ego candidates.

![](images/6ce735d0a86037a4140a92c8c2b83a832cf05dcde8c538e95078b846e560a96f.jpg)

![](images/139a95113d4ad5cf80aa4731b8c6a557aa162ed768e7a6bcd7b8cd3454781a4b.jpg)

![](images/a7d2388026d2f1217da223760cb4076f2ea6fcfe978dd7f9da1ef28aa9cc63a8.jpg)

![](images/c9c24211bf2d0422d240150112e9bab208f3e3dd94d773acaa4f1e6c933a3a3c.jpg)  
(a) Reacting to a stopping lead vehicle

![](images/9082b3bf57ca7b87798067902a9dc65aee380c6af8223a05dfbcc27a1aeb6535.jpg)

![](images/7532c20a0101e05fa9b11bd8d8e1c6e34de198850a71abf03ba92647006a4d40.jpg)

![](images/67e173af565c026cb49552553fe5df2e0b7d511a796cd0b761a4871d84da7d4f.jpg)  
(b) Reacting to a pedestrian

![](images/653cb7b4e731dc72c9ab4f67e30995492c19239e737433ee1d29f380abb8b4d6.jpg)

![](images/b0bd710d13969c3429668b6a3a8ffcb145a5b52a23dfb61c29fe56c86d5e8be8.jpg)

![](images/e62ca2aab1687c93abfb87d493cf3c8caa036071e5d958b7eedb652dab420c1f.jpg)  
(c) Turning left at an intersection

![](images/af7eeedc8e47bce03f08a94d6c051c491c5c9bfc195d36a7f834f163dc483380.jpg)

![](images/94fdbea811e0f8fb82ad3a29cf8a7160746ea81c667a1a418bc4f0bec7fddd15.jpg)  
(d) Stop-and-go trafic on a highway  
Fig. 7: Zero-shot qualitative examples on real-world driving logs. Each row shows surround-view camera predictions, the BEV cost visualization with the selected trajectory, and ground-truth boxes with ego trajectory and semantic map support.