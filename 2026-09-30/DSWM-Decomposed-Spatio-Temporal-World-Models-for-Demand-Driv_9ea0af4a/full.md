# DSWM: Decomposed Spatio-Temporal World Models for Demand-Driven UAV Base Station Repositioning

Shengjie Zhong, Graduate Student Member, IEEE, Zhongliang Zhao, Jingxuan Chen, Xianbin Cao, Xinmei Qiang, Dapeng O. Wu, Fellow, IEEE, Tony Q. S. Quek, Fellow, IEEE

This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible.

Abstract—Uncrewed aerial vehicle base stations (UAV-BSs) are expected to cover traffic demand that shifts across space and time, yet most repositioning schemes either re-solve an optimization problem per slot or learn reactive policies without an explicit demand model. We cast demand-driven fleet repositioning as latent-space decision-time planning and propose DSWM, a decomposed spatio-temporal world model: an agentic controller that perceives the demand field through a rolling observation window, retains operational context in a latent recurrent state, reasons about candidate motions by imagined rollouts under an uncertainty penalty, and coordinates the fleet through replanned first actions. DSWM learns a recurrent state-space model shaped by an exponential-moving-average (EMA) based latent predictive objective with variance regularization. It attaches a differentiable service simulator that replays the association, probabilistic lineof-sight channel, and Shannon rate chain inside latent rollouts. Planning uses a cross-entropy method whose imagined demand is anchored on the current observation window with mixing coefficient $\rho \ = \ 0 . 9 5 .$ On a unified pipeline over three real datasets (Milan CDR (call detail record), Shanghai Telecom, YJMob100K) and 14 methods including five reproduced IEEE baselines, DSWM attains weekday served ratios of 0.889, 0.908, and 0.898, ranking first among non-ablated configurations on every dataset. On Milan it improves over the strongest nonlearning baseline (Greedy, 0.780) by 0.109, a margin that comes from decision-time use of observations rather than prediction accuracy.

Index Terms—6G mobile communication, agentic AI, reinforcement learning, uncrewed aerial vehicles (UAVs), world model.

## I. INTRODUCTION

as a fast way to add capacity where ground infrastructure is absent or saturated [1], [2]. UAV-BSs fly at low altitude, establish line-of-sight (LoS) links with high probability, and can be repositioned within minutes [1]. Recent surveys argue that such aerial access layers must be driven by generative and predictive models of network state rather than static rules [3], [2]. The practical question is therefore not whether to move the fleet, but how to decide where each UAV-BS should be at every slot so that a demand map that drifts over hours is covered. This is an agentic control problem: the controller must perceive a partially observed demand field, retain operational context over hours, reason under uncertainty about delayed consequences, and coordinate a fleet under coupled energy and coverage constraints. Two properties make this decision hard. Demand is both spatially uneven and temporally nonstationary, with commute peaks, weekend shifts, and holiday anomalies. And each repositioning decision has delayed consequences: a move that takes minutes pays off only after the demand arrives, while battery limits force charging detours that remove capacity for an hour. Any controller must therefore trade current service against future service under dynamics it cannot observe directly.

Three lines of work address this question, and each leaves a specific gap. First, optimization-based designs decompose placement, trajectory, and resource allocation into subproblems solved by block coordinate descent (BCD) or convex relaxations [4], [5], [6], [7]. They are strong within a static or slowly changing demand snapshot, but they re-solve a nonconvex problem per decision and degrade when the demand drifts between solves. Second, model-free reinforcement learning (RL) and multi-agent RL (MARL) learn reactive policies for trajectory and resource control [8], [9], [10], [11], [12], [13], [14], [15], [16]. These policies carry no explicit demand model, so they need many environment samples and generalize weakly out of distribution (OOD). Third, two-stage pipelines first predict traffic with a spatio-temporal forecaster and then plan against the prediction [17], [18], [19], [20], [21]. Prediction error passes directly into the plan, and the forecasting objective (per-cell accuracy) is misaligned with the coverage objective (demand-weighted service).

![](images/57e963237dff14ee59751f7cc5d300f9df2cc90c4739a1a95b719a9fe929bb16.jpg)  
Fig. 1. Scenario and motivation. A fleet of K UAV-BSs at altitude $h _ { \mathrm { f l y } }$ covers a gridded demand map x<sub>t</sub> that drifts over the day (commute bimodal pattern). Because the demand map is strongly structured in space and time, the bottleneck for coverage is how the current observation window is exploited at decision time, not raw forecasting accuracy. DSWM therefore plans in a learned latent space with the imagination anchored on the current observation.

Our key insight, illustrated in Fig. 1, is that urban demand has strong spatio-temporal structure and is largely predictable, so the bottleneck is not prediction accuracy but how observations are used at decision time. World models answer exactly this question: they learn latent dynamics and let the agent plan by imagination at each step [22], [23], [24]. If a planner anchors its imagined demand on the current observation window, it can in principle match a planner that sees the future. In a preliminary smaller-scale setting (K=4, without the battery model), anchoring improved both splits by +0.018 over the non-anchored variant at no cost (replicated from two checkpoints), while demand-forecasting enhancements were net negative; we report these as motivation—the main-table evidence appears in Section V-D. Based on this insight, we propose DSWM (Decomposed Spatio-Temporal World Model). DSWM learns a recurrent state-space model (RSSM) of the demand-service dynamics, shaped by an EMAtarget latent predictive objective with a variance-margin regularizer (JEPA/VICReg-style) [25] and variance regularization. A decomposed service head replays the environment’s service physics—association, probabilistic LoS, Shannon rate, and capped per-cell fulfillment—as a differentiable simulator inside latent rollouts, and an ensemble disagreement term penalizes uncertain branches. A cross-entropy method (CEM) planner scores H-slot imagined trajectories with the demand anchored on the current observation with mixing coefficient $\rho = 0 . 9 5$ , and executes only the first action.

The contributions are summarized as follows:

• Modeling paradigm (Section IV, Fig. 2). We formalize demand-driven UAV-BS fleet repositioning as latentspace decision-time planning and propose DSWM, an RSSM world model shaped by a latent predictive regularization objective. To our knowledge this is the first world-model formulation of UAV-BS repositioning with an explicit coverage objective.

• Architecture (Section IV-C, Fig. 3). We design a decomposed demand-service physics head, a differentiable replica of the service simulator, with an ensemble uncertainty penalty. Ablation attributes +0.120 of served ratio to the decomposed head (0.769 vs. 0.889) and −0.004 to the uncertainty term, which we report as an insurance mechanism rather than a gain source.

• Algorithm (Section IV-D, Fig. 4). We propose observation-anchored CEM decision-time planning with anchor mixing coefficient $\rho \ : = \ : 0 . 9 5$ . DSWM improves over the strongest non-learning baseline by 0.109 on Milan (0.889 vs. 0.780); in a preliminary smaller-scale setting (K=4, without the battery model), anchoring improved both splits by +0.018 over the non-anchored variant at no cost (replicated from two checkpoints).

• Benchmark evidence (Section V, Tables III and IV). We build a unified pipeline over three real datasets (Milan CDR, Shanghai Telecom, YJMob100K) with 14 methods, including five recent IEEE baselines (2023–2026), and a three-way leakage audit. DSWM ranks first among nonablated configurations on all three datasets.

A further motivation shapes our evaluation design. Comparisons of learning-based network controllers are often undermined by three silent leaks: random train/test splits that let the model memorize periodicity, normalization statistics computed on the full dataset, and evaluation protocols that differ across methods. We therefore fix temporal splits, derive all scaling factors from the training segment, give every method the identical observation set, and subject the full pipeline to a readonly leakage audit (Section V-B). We regard this protocol as part of the contribution, because the margin we report (0.109 over the strongest baseline on Milan) is meaningful only if the comparison is clean.

The remainder of the paper is organized as follows. Section II reviews related work. Section III presents the system model and problem formulation. Section IV details DSWM. Section V reports experiments. Section VI concludes.

## II. RELATED WORK

## A. UAV-BS Deployment and Trajectory Design

Early work characterized the air-to-ground channel and derived optimal altitudes for coverage [5], [1]. Zeng and Zhang introduced the rotary-wing power model and optimized trajectories for energy efficiency [4]. Optimizationbased formulations then scaled to multi-UAV settings: Sun et al. [6] minimized a weighted sum of delay, energy, and load imbalance via BCD with Karush–Kuhn–Tucker (KKT) bisection and successive convex approximation (SCA); Jia et al. [7] combined K-means and Voronoi pre-deployment with whale optimization for hierarchical swarms. Related formulations addressed distributionally robust aerial edge computing [26], wireless-powered data acquisition [27], two-timescale mobile edge computing (MEC) scheduling [28], sensingcommunication-computing trajectories [29], covert integrated sensing and communication (ISAC) beamforming [30], [31], and self-adjusting slicing [32]; learning-based variants replace the solver with a policy network [33], [34], [35], [36], [37], [38], [39], [40]. The gap in this line is per-slot re-solving against a demand snapshot: when demand drifts between solves, BCD-type methods are the most sensitive in our benchmark (Section V-E).

## B. Spatio-Temporal Demand Prediction

Cellular and mobility demand prediction spans graph networks, Transformers, and large language models: MVSTGN fused multi-view spatial-temporal graphs for cellular traffic [17]; Gu et $a l .$ predicted city-level traffic with a spatialtemporal Transformer [18]; probabilistic graph networks quantified travel demand uncertainty [19]; recent work used confounder representations [41], traffic LLMs [20], and diffusionbased text-to-traffic generation [21]. These models output a forecast that a downstream planner must trust; the gap is objective misalignment, since per-cell forecasting accuracy does not measure demand-weighted coverage. We keep prediction inside a decision loop and show that decision-time use of observations, not forecast skill, is the binding constraint.

## C. World Models and Decision-Time Planning

World models learn latent dynamics and plan by imagination [42], [22]; TD-MPC combined model predictive control with temporal-difference learning [23]; the Dreamer line scaled world-model agents across domains [43], [24], [44]. Self-supervised objectives [45], [25], [46] stabilize representation learning without reconstruction heuristics, and general agents have been argued to require world models [47]. In wireless, MobiWorld built world models of mobile networks for traffic and channel simulation [48], and internal-inference frameworks moved learning inside the network [49]. The gap is that existing wireless world models target prediction or simulation, not closed-loop repositioning with a physical service model inside the imagination. Offline model-based RL work such as MOPO [50] motivates our uncertainty penalty, which we apply to imagined rollouts rather than policy training.

## D. Closest Works and Differentiation

Four works are closest to ours, and the distinctions are structural. GA-MATR [13] learns a graph-attention MARL policy with a fairness regularizer; it is model-free, so it cannot imagine counterfactual demand, yet it is the strongest learning baseline under distribution shift in our benchmark (Section V-E), though still far below DSWM on weekdays. HRL-TPRA [16] decomposes the problem hierarchically in time with pointer-network trajectory planning and per-UAV resource actors; the hierarchy is compute-bound and ranked last among learning baselines in our setting (0.485). JTORATC [6] solves a weighted BCD problem per decision window; it carries no learned demand model and suffers OOD backfire (holiday 0.495 vs. weekday 0.583). MobiWorld [48] builds a generative world model of wireless networks but does not close the loop with a service-physics-aware planner. DSWM differs from all four in that the service physics is differentiable inside the world model, so gradients and planning scores both flow through the true coverage mechanism.

## III. SYSTEM MODEL AND PROBLEM FORMULATION

## A. Scenario and Assumptions

We consider a city-wide aerial access layer. A fleet of $K \ = \ 1 6 \ { \mathrm { U A V  – B S s } }$ hovers at altitude $h _ { \mathrm { f f y } } = 1 0 0$ m over a service area rasterized into $G$ cells (G=400 for Milan) of side c (235 m for Milan). Time is slotted into $T _ { \mathrm { s l o t } } ~ = ~ 1 0$ min slots; one episode is one day of $T = 1$ 44 slots. The demand in slot t is a nonnegative map $\boldsymbol { x } _ { t } \in \mathbb { R } _ { + } ^ { G }$ , normalized by the 99th percentile of the training segment. We adopt the following assumptions, each of which matches the deployed environment used in all experiments.

(A1) Fixed flight altitude. All UAV-BSs operate at a constant altitude of $h _ { \mathrm { f f y } } ~ = ~ 1 0 0$ m. This isolates the horizontal repositioning problem, as optimal altitude results are well-established [5].

(A2) UAV mobility. A UAV travels at most $D _ { \mathrm { m a x } } = 1 5 0 0$ m per 10-minute slot. This fits comfortably within the $V _ { \mathrm { m a x } } = 2 5$ m/s speed limit of consumer-grade platforms, requiring an average speed of only 2.5 m/s.

(A3) Nearest-available association. Each cell is served by the closest active UAV-BS. This practical Voronoi-style coverage [7] simplifies the model by decoupling user association from the learning problem.

(A4) Homogeneous rate demand. Each normalized demand unit corresponds to $r _ { 0 } = 0 . 5$ Mbps per user. This isolates coverage optimization from traffic heterogeneity, which is typically handled by separate prediction modules [17].

(A5) Battery constraints. Since finite endurance dictates short-horizon planning [4], we explicitly model charging cycles. A UAV dropping below a 15% reserve returns to a quadrant station, charges for 6 slots (1 hour), and redeploys.

Table I lists the main notation.

## B. Channel Model

Let $d _ { g , k } = \sqrt { h _ { \mathrm { H y } } ^ { 2 } + \Vert p _ { t } ^ { k } - c _ { g } \Vert ^ { 2 } }$ be the distance between UAV k at position $p _ { t } ^ { k }$ in slot t and the center $c _ { g }$ of cell $^ { g , }$ and let $\theta _ { g , k } = \arcsin ( h _ { \mathrm { f l y } } / d _ { g , k } )$ be the elevation angle, converted to degrees before evaluating (1). The LoS probability follows the standard sigmoid model [5], [1]:

$$
P _ { \mathrm { l o s } } ( \theta _ { g , k } ) = \frac { 1 } { 1 + a _ { \mathrm { e n v } } \exp \big ( - b _ { \mathrm { e n v } } \big ( \theta _ { g , k } - a _ { \mathrm { e n v } } \big ) \big ) } ,\tag{1}
$$

where $a _ { \mathrm { e n v } }$ and $b _ { \mathrm { e n v } }$ are environment constants. The expected path loss in dB mixes LoS and non-LoS (NLoS) components with excess losses $\eta _ { \mathrm { L o S } }$ and $\eta _ { \mathrm { N L o S } } ;$ the constants a<sub>env</sub>, $b _ { \mathrm { e n v } }$ and the excess-loss terms η<sub>LoS</sub>, η<sub>NLoS</sub> follow the urban parametrization of [5]:

$$
\begin{array} { r l } & { \mathrm { P L } _ { g , k } = 2 0 \log _ { 1 0 } \Bigl ( \frac { 4 \pi f _ { c } d _ { g , k } } { c _ { 0 } } \Bigr ) } \\ & { \qquad + P _ { \mathrm { l o s } } ( \theta _ { g , k } ) \eta _ { \mathrm { L o S } } + \bigl ( 1 - P _ { \mathrm { l o s } } ( \theta _ { g , k } ) \bigr ) \eta _ { \mathrm { N L o S } } , } \end{array}\tag{2}
$$

TABLE I MAIN NOTATION
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $t , T$ </td><td>slot index; slots per day (T=144)</td></tr><tr><td> $T _ { \mathrm { s l o t } }$ </td><td>slot duration (10 min)</td></tr><tr><td> $k , K$ </td><td>UAV-BS index; fleet size (K=16)</td></tr><tr><td> $g , G$ </td><td>cell index; number of cells (G=400)</td></tr><tr><td>C</td><td>cell side length (235/470/500 m)</td></tr><tr><td> $x _ { t } \in \mathbb { R } _ { + } ^ { G }$ </td><td>normalized demand map</td></tr><tr><td> $s _ { t } ^ { k } = ( \stackrel { \cdot } { p _ { t } ^ { k } } , b _ { t } ^ { k } , \xi _ { t } ^ { k } )$ </td><td>UAV k state (position, battery, charge)</td></tr><tr><td> $\bar { a _ { t } } \in [ - \bar { 1 } , 1 ] ^ { \bar { K } }$  ×3</td><td>action (∆x, ∆y, bw_logit)</td></tr><tr><td> $h _ { \mathrm { f l y } } , \operatorname { \bar { D } } _ { \mathrm { m a x } } , \operatorname { \bar { V } } _ { \mathrm { m a x } }$ </td><td>altitude; move limit; speed limit</td></tr><tr><td> $B , \  N _ { 0 } , \ f _ { c }$   $\gamma , E$ </td><td>bandwidth; noise spectral density; carrier frequency planner discount; ensemble size</td></tr><tr><td> $\dot { N } _ { \mathrm { c e m } } , N _ { \mathrm { e l i t e } }$ </td><td>CEM samples; elites</td></tr><tr><td> $m , \sigma _ { 0 }$ </td><td>CEM sampling mean; initial sampling std</td></tr><tr><td>ν</td><td>EMA coefficient of the teacher</td></tr><tr><td>Ot</td><td>observation (past w=6 demand, fleet, time)</td></tr><tr><td> $h _ { t } , z _ { t }$ </td><td>GRU deterministic / stochastic latent state</td></tr><tr><td> $q _ { \phi } , p _ { \theta }$ </td><td>posterior / prior of the RSSM</td></tr><tr><td> $\bar { \mathrm { S R } _ { t } } \bar { \in } [ 0 , 1 ]$ </td><td>served ratio (aggregate demand satisfaction)</td></tr><tr><td> $r _ { 0 } , U _ { \mathrm { p e a k } }$ </td><td>per-user rate demand; satisfaction threshold</td></tr><tr><td> $E _ { t } , \dot { E _ { \mathrm { h o v e r } } } , E _ { \mathrm { b a t t } }$ </td><td>slot / hover / battery energy</td></tr><tr><td> $P _ { \mathrm { l o s } }$ </td><td>LoS probability</td></tr><tr><td> $R _ { g , k }$ </td><td>Shannon rate of link  $( g , k )$ </td></tr><tr><td> $H , \rho$ </td><td>horizon; anchor mixing coefficient</td></tr><tr><td> $u _ { t } , \lambda _ { u }$ </td><td>ensemble uncertainty; penalty weight</td></tr><tr><td> $r _ { s }$ </td><td>Spearman rank correlation</td></tr></table>

with carrier frequency $f _ { c } = 2$ GHz and light speed $c _ { 0 }$ . With transmit power $P _ { k } = 0 . 5 \mathrm { ~ W ~ }$ , allocated bandwidth $B _ { g , k }$ , and noise spectral density $N _ { 0 } = 1 0 ^ { - 2 0 . 4 }$ W/Hz, the Shannon rate of link $( g , k )$ is

$$
R _ { g , k } = B _ { g , k } \log _ { 2 } \Bigl ( 1 + \frac { P _ { k } 1 0 ^ { - \mathrm { { P L } } _ { g , k } / 1 0 } } { N _ { 0 } B _ { g , k } } \Bigr ) ,\tag{3}
$$

where $\begin{array} { l } { \displaystyle \sum _ { q } B _ { g , k } ~ \leq ~ B } \end{array}$ with $B ~ = ~ 2 0$ MHz per UAV-BS. The chain $( 1 ) – ( 3 )$ is deterministic given positions and is differentiable almost everywhere; Section IV-C reuses it inside the world model.

## C. Energy Consumption and Battery State Machine

Rotor power follows the model of Zeng and Zhang [4]:

$$
\begin{array} { r l r } {  { P ( V ) = P _ { 0 } \Big ( 1 + \frac { 3 V ^ { 2 } } { U _ { \mathrm { t i p } } ^ { 2 } } \Big ) + P _ { i } \Big ( \sqrt { 1 + \frac { V ^ { 4 } } { 4 v _ { 0 } ^ { 4 } } } - \frac { V ^ { 2 } } { 2 v _ { 0 } ^ { 2 } } \Big ) ^ { 1 / 2 } } } \\ & { } & { + \ \frac { 1 } { 2 } d _ { 0 } \varrho _ { \mathrm { a i r } } s A V ^ { 3 } , } \end{array}\tag{4}
$$

where $V$ is speed, $P _ { 0 }$ and $P _ { i }$ are blade profile and induced power in hover, $U _ { \mathrm { t i p } }$ is tip speed, $v _ { 0 }$ is mean rotor induced velocity, and $d _ { 0 } , \varrho _ { \mathrm { a i r } } , s , A$ are fuselage drag ratio, air density, rotor solidity, and disc area $( \rho$ is reserved for the anchor mixing coefficient). The per-UAV slot energy is $E _ { t } ^ { k } =$ $P ( V _ { t } ^ { k } ) T _ { \mathrm { s l o t } }$ , and $\begin{array} { r } { E _ { t } = \sum _ { k } E _ { t } ^ { k } } \end{array}$ is the fleet total; hover energy $E _ { \mathrm { h o v e r } } = P ( 0 ) T _ { \mathrm { s l o t } }$ is the reference. Because repositioning spans at most 1500 m per 10-min slot, hovering dominates consumption; this physical fact explains the narrow energy spread $( \sim 3 \% )$ across methods in Section V-D. The battery obeys $b _ { t + 1 } ^ { k } = b _ { t } ^ { k } - E _ { t } ^ { k }$ while airborne; when $b _ { t } ^ { k } < 0 . 1 5 E _ { \mathrm { b a t t } }$ with $E _ { \mathrm { { b a t t } } } = 1 { , } 9 7 2 { , } 8 0 0 \ ]$ J (548 Wh, DJI M300-class), UAV k enters returning, flies to its quadrant station, charges for 6 slots, and redeploys at full charge.

## D. Demand-Service Model

Cell g requests an aggregate rate $x _ { t } ^ { g } r _ { 0 }$ . Under nearestavailable association (A3), the served demand of cell g in slot

$$
\begin{array} { r l } & { t \mathrm { i s } } \\ & { y _ { t } ^ { g } = \operatorname* { m i n } \bigl ( x _ { t } ^ { g } r _ { 0 } , \ R _ { g , k ^ { * } ( g ) } \bigr ) , \qquad k ^ { * } ( g ) = \arg \operatorname* { m i n } _ { k \in \mathcal { A } _ { t } } \lVert p _ { t } ^ { k } - c _ { g } \rVert , } \end{array}\tag{5}
$$

where $\boldsymbol { A } _ { t }$ is the set of airborne, non-returning UAV-BSs and $R _ { g , k }$ is from (3). The served ratio aggregates fulfilled demand over the map:

$$
\mathrm { S R } _ { t } = \frac { \sum _ { g } y _ { t } ^ { g } } { \sum _ { g } x _ { t } ^ { g } r _ { 0 } } = \frac { \sum _ { g } x _ { t } ^ { g } \operatorname* { m i n } \bigl ( 1 , \ R _ { g , k ^ { * } ( g ) } / ( x _ { t } ^ { g } r _ { 0 } ) \bigr ) } { \sum _ { g } x _ { t } ^ { g } } .\tag{6}
$$

$\mathrm { S R } _ { t }$ is demand-weighted, so covering a hotspot matters more than covering a sparse cell. A cell counts as satisfied when its fulfillment ratio reaches $U _ { \mathrm { p e a k } } = 0 . 8 ;$ this threshold is used only to mark satisfied cells in the Jain fairness statistic reported in Section V-D. The reward used by learning methods is

$$
\mathcal { R } _ { t } = \mathrm { S R } _ { t } - \beta \left( 1 - \mathrm { S R } _ { t } \right) - \alpha \frac { E _ { t } } { E _ { \mathrm { h o v e r } } K } ,
$$

with $\alpha = \beta = 1 . 0$ in all experiments.

(7)

## E. Problem Formulation

The operator seeks a policy π mapping observations to actions that maximizes cumulative served ratio over an episode:

$$
\left( \mathrm { P 1 } \right) : \quad \operatorname* { m a x } _ { \pi } \mathbb { E } \Big [ \sum _ { t = 1 } ^ { T } \mathrm { S R } _ { t } \Big ]\tag{8}
$$

$$
\mathrm { s . t . } \ \lVert \Delta p _ { k } ^ { t } \rVert \leq D _ { \operatorname* { m a x } } , \ \lVert \Delta p _ { k } ^ { t } \rVert / T _ { \mathrm { s l o t } } \leq V _ { \operatorname* { m a x } } ,\tag{C1}
$$

$$
b _ { t + 1 } ^ { k } = b _ { t } ^ { k } - E _ { t } ^ { k } , \ b _ { t } ^ { k } < 0 . 1 5 E _ { \mathrm { b a t t } } \Rightarrow \mathrm { r e t u r n } ,\tag{C2}
$$

$$
\begin{array} { r } { \sum _ { g } B _ { g , k } \le B , \forall k , t . } \end{array}\tag{C3}
$$

Constraint (C1) enforces kinematics, (C2) the battery state machine, and (C3) the per-UAV bandwidth budget. The observation $o _ { t }$ contains the past $w = 6$ demand maps, the current demand map, fleet position/battery/charge states, and a timephase encoding; all methods in Section V share this identical information set.

(P1) is hard for three reasons, and each maps to one DSWM component. First, the demand map is time-varying and partially observed through a short window, so per-slot re-optimization chases a moving target; DSWM answers this with a learned world model that carries demand dynamics in latent state. Second, the objective couples a nonconvex airto-ground channel with association and battery constraints, so the composite problem is nonconvex and gradients through solvers are unavailable; DSWM answers this with a decomposed differentiable service head that makes the coverage mechanism differentiable. Third, credit assignment spans slots: a repositioning decision pays off hours later, after charging and drift; DSWM answers this with CEM planning over a horizon of $H = 1 5$ slots (2.5 h) anchored on the current observation.

## IV. PROPOSED METHOD: DSWM

## A. Overview

Fig. 2 shows the data flow. At each slot, the encoder compresses the observation $o _ { t }$ and the RSSM updates its latent state $\left( h _ { t } , z _ { t } \right)$ . The CEM planner samples candidate action sequences, rolls them forward for H slots inside the world model, converts each imagined latent state into a servedratio estimate through the decomposed service head, penalizes ensemble disagreement, and executes the first action of the best sequence. Training is offline: sequences from a replay buffer of pre-collected transitions update the RSSM, the demand ensemble, the latent predictive objective, and the service head end to end. The same information set $( o _ { t }$ and internal states) is available to DSWM and to every baseline.

![](images/5474f12e5b2d591988285a7c99350775d4beeac03d816cbaa109354b9315c2b1.jpg)  
Fig. 2. DSWM framework. The encoder and RSSM maintain a latent state from the observation window. The CEM planner imagines H-slot rollouts, anchors the imagined demand on the current observation with mixing coefficient $\rho = 0 . 9 5$ , scores each rollout with the decomposed service head minus an uncertainty penalty, and executes the first action. Training (dashed) is offline over replayed sequences.

## B. RSSM World Model with Latent Predictive Regularization

The world model is an RSSM [22], [24] with a deterministic GRU (gated recurrent unit) state $h _ { t }$ and a diagonal-Gaussian stochastic state $z _ { t } .$

$$
h _ { t } = f _ { \theta } ( h _ { t - 1 } , z _ { t - 1 } , a _ { t - 1 } ) ,\tag{9}
$$

$$
\mathrm { p r i o r : } \quad p _ { \theta } ( z _ { t } \mid h _ { t } ) = { \mathcal N } \big ( \mu _ { \theta } ( h _ { t } ) , \sigma _ { \theta } ^ { 2 } ( h _ { t } ) \big ) ,\tag{10}
$$

$$
\mathrm { p o s t e r i o r : \quad } q _ { \phi } ( z _ { t } \mid h _ { t } , o _ { t } ) = { \cal N } \big ( \mu _ { \phi } ( h _ { t } , e _ { \psi } ( o _ { t } ) ) ,\tag{11}
$$

$$
\sigma _ { \phi } ^ { 2 } ( h _ { t } , e _ { \psi } ( o _ { t } ) ) ) ,
$$

where $e _ { \psi }$ is the observation encoder. Imagination uses the prior; training uses the posterior. The Kullback–Leibler (KL) term uses free bits with the Dreamer-style balancing weights 0.8/0.2 [24]:

$$
{ \cal L } _ { \mathrm { K L } } = 0 . 8 D _ { \mathrm { K L } } \ ( \mathrm { s g } [ q _ { \phi } ] \| p _ { \theta } ) + 0 . 2 D _ { \mathrm { K L } } \big ( q _ { \phi } \| \mathrm { s g } [ p _ { \theta } ] \big ) ,\tag{12}
$$

where each per-dimension KL is clipped from below by a freebits floor and $\mathrm { s g } [ \cdot ]$ stops gradients; the 0.8-weighted term trains the prior toward the stopped posterior, and the 0.2-weighted term trains the posterior toward the stopped prior. Free bits prevent posterior collapse, which we found to be the main stability risk during training scans.

Two implementation choices: the demand ensemble operates in symlog space, s $\operatorname { y m l o g } ( x ) = \operatorname { s i g n } ( x ) \ln ( 1 + | x | )$ , because demand is heavy-tailed and a linear-scale loss would fit the mean and ignore hotspots; and a GRU deterministic path instead of a Transformer backbone, since the observation window is short $( w \ = \ 6$ slots), the planning-relevant state is Markovian given $h _ { t } ,$ and recurrence keeps the imagination forward pass cheap for CEM. Reconstruction-free shaping follows the joint-embedding predictive (JEPA) view [25]. Instead of reconstructing the demand map (which forces the model to spend capacity on heavy-tailed magnitudes irrelevant to coverage), DSWM predicts in representation space. Let $\hat { h } _ { t + 1 } \ = \ g _ { \omega } ( h _ { t } )$ be the one-step prediction of the recurrent latent; the target $\bar { h } _ { t + 1 }$ is produced by an exponential-movingaverage (EMA) teacher, a delayed copy of the encoder–RSSM stack whose parameters <sup>¯</sup>θ track the online parameters θ after every gradient step by

$$
{ \bar { \theta } }  \nu { \bar { \theta } } + ( 1 - \nu ) \theta ,\tag{13}
$$

and $\bar { h } _ { t + 1 }$ denotes the recurrent latent produced by this teacher. The latent prediction loss is

$$
L _ { \mathrm { p r e d } } = \left. g _ { \omega } ( h _ { t } ) - \mathrm { s g } [ \bar { h } _ { t + 1 } ] \right. _ { 2 } ^ { 2 } ,\tag{14}
$$

where $g _ { \omega }$ is a latent predictor conditioned on the recurrent state and ν is the EMA coefficient. Thus (14) and the first term of (15) are the same loss, written once and reused, not two different objectives. A joint-embedding loss alone admits the collapsing solution $f _ { \theta } \equiv \mathrm { c o n s t }$ , which makes (14) identically zero but destroys all information. We prevent collapse with a per-dimension variance margin in the style of VICReg [51],

$$
\begin{array} { r } { L _ { \mathrm { L P R } } = \mathbb { E } \left. \hat { h } _ { t + 1 } - \mathrm { s g } [ \bar { h } _ { t + 1 } ] \right. _ { 2 } ^ { 2 } } \\ { + \lambda _ { \mathrm { v a r } } \displaystyle \sum _ { j } \operatorname* { m a x } \left( 0 , m _ { \mathrm { v a r } } - \mathrm { S t d } _ { j } ( h ) \right) , } \end{array}\tag{15}
$$

where the first term instantiates (14) on the recurrent latent $( \hat { h } _ { t + 1 } = g _ { \omega } ( h _ { t } ) , \ \bar { h } _ { t + 1 }$ the EMA target), $\mathrm { S t d } _ { j }$ is taken over the batch dimension for latent dimension $j ,$ and $m _ { \mathrm { v a r } }$ is the variance margin.

This objective replaced an earlier reconstruction term after an internal scan: it raised the weekday served ratio from 0.878 to 0.888 and stabilized the OOD split. The total world-model loss is

$$
{ \cal L } _ { \mathrm { W M } } = { \cal L } _ { \mathrm { d e m } } + \beta _ { \mathrm { K L } } { \cal L } _ { \mathrm { K L } } + \lambda _ { L } { \cal L } _ { \mathrm { L P R } } + { \cal L } _ { \mathrm { s e r v e } } ,\tag{16}
$$

where $L _ { \mathrm { d e m } }$ is the demand-head loss and $L _ { \mathrm { s e r v e } }$ is the servicehead loss defined next.

## C. Decomposed Demand-Service Physics Head

Fig. 3 illustrates the service-physics chain that the head replays differentiably—association, probabilistic LoS channel, Shannon rate, and capped per-cell fulfillment. A demand ensemble of E heads predicts the next-slot demand map in symlog space, $\hat { x } _ { t + 1 } ^ { ( e ) } \stackrel { \cdot } { = } \mathrm { s y m l o g } ^ { - 1 } \big ( D _ { e } ( h _ { t + 1 } , z _ { t + 1 } ) \big )$ , and their mean $\hat { x } _ { t + 1 }$ is the demand estimate. The head then replays the service physics of Section III as a differentiable simulator: given imagined UAV positions decoded from the action sequence, it computes association by nearest available UAV, evaluates (1)–(3) for each link, caps per-cell fulfillment, and aggregates:

$$
\widehat { \mathrm { S R } } _ { t + 1 } = \frac { \sum _ { g } \hat { x } _ { t + 1 } ^ { g } \operatorname* { m i n } \bigl ( 1 , \hat { R } _ { g , k ^ { * } ( g ) } / ( \hat { x } _ { t + 1 } ^ { g } r _ { 0 } ) \bigr ) } { \sum _ { g } \hat { x } _ { t + 1 } ^ { g } } .\tag{17}
$$

Because every step of (17) is differentiable, gradients flow through the coverage mechanism back into the world model; the model is therefore trained on the quantity the operator cares about, not on a proxy. The service loss is $L _ { \mathrm { s e r v e } } ~ =$ $\mathbb { E } ( \widehat { \mathrm { S R } } _ { t + 1 } - \mathrm { S R } _ { t + 1 } ) ^ { 2 }$ on replayed transitions. Removing this decomposition and regressing $\mathrm { S R } _ { t }$ directly from latent state costs 0.120 of served ratio (Section V-F), which confirms that the physics, not the latent capacity, carries the coverage signal.

Ensemble disagreement defines the uncertainty of an imagined step:

$$
u _ { t + 1 } = { \mathrm { S t d } } _ { e } \left( { \hat { x } } _ { t + 1 } ^ { ( e ) } \right) { \mathrm { a g g r e g a t e d ~ o v e r ~ c e l l s , } }\tag{18}
$$

and the planner subtracts $\lambda _ { u } u _ { \tau }$ from imagined rewards $( \lambda _ { u } =$ 1.0 in production), following the pessimism principle of offline model-based RL [50]. Removing this term changes the score by only +0.004 (Section V-F): the uncertainty penalty acts as insurance against rare overconfident branches, not as a gain source.

## D. Observation-Anchored CEM Decision-Time Planning

Fig. 4 illustrates the mechanism behind the planner: openloop imagination accumulates error along the horizon, while re-anchoring each rollout to the current observation keeps the estimate grounded. At slot t, the planner samples $N _ { \mathrm { c e m } }$ action sequences $\scriptstyle a _ { t : t + H - 1 }$ from a diagonal Gaussian, rolls each through the prior (9)–(10), and scores the rollout as

$$
J \big ( a _ { t : t + H - 1 } \big ) = \sum _ { \tau = t + 1 } ^ { t + H } \gamma ^ { \tau - t } \big ( \widehat { \mathrm { S R } } _ { \tau } - \lambda _ { u } u _ { \tau } \big ) ,\tag{19}
$$

where $H = 1 5$ slots (2.5 h) and $\gamma$ is the planner discount. The top $N _ { \mathrm { e l i t e } }$ sequences refit the sampling distribution, and after a fixed number of iterations the first action of the best sequence is executed.

The key mechanism is observation anchoring. With mixing coefficient $\rho ,$ the imagined demand at each rollout step is blended with the current observed demand map (persist anchoring):

Algorithm 1 Per-Slot Observation-Anchored CEM Planning   
Require: observation o<sub>t</sub>; latent state $( h _ { t - 1 } , z _ { t - 1 } ) ;$ horizon H; an  
chor $\rho = 0 . 9 5 ;$ samples $N _ { \mathrm { c e m } } ;$ elites $N _ { \mathrm { e l i t e } } ;$ iterations I   
1: Encode $o _ { t }$ and update posterior: $h _ { t } , z _ { t } \gets ( 9 ) _ { : } ( 1 1 )$   
2: Initialize sampling mean m $ 0 ,$ std $\Sigma  \sigma _ { 0 } ^ { 2 }$ I over $\scriptstyle a _ { t : t + H - 1 }$   
3: for $i = 1$ to I do   
4: Sample $N _ { \mathrm { c e m } }$ sequences $\{ a _ { t : t + H - 1 } ^ { ( n ) } \} \sim { \mathcal { N } } ( m , \Sigma )$   
5: for each sequence n do   
6: Roll out $\hat { ( } h _ { \tau } , z _ { \tau } )$ for H steps via $( 9 ) , ( 1 0 )$   
7: Predict $\hat { x } _ { \tau _ { - } } ^ { ( e ) } .$ , anchor by (20) with $\rho = 0 . 9 5$   
8: Evaluate SRc <sub>τ</sub> by (17) and $u _ { \tau }$ by (18)   
9: Score $J ^ { ( n ) }$ by (19)   
10: end for   
11: Select $N _ { \mathrm { e l i t e } }$ top sequences; refit $m , \Sigma$ to elites   
12: end for   
13: return first action $a _ { t } ^ { * }$ of the best-scoring sequence

$$
\hat { x } _ { \tau } \gets \rho x _ { t } + ( 1 - \rho ) \hat { x } _ { \tau } .\tag{20}
$$

Equation (20) is a deterministic convex interpolation; we use $\rho \ : = \ : 0 . 9 5$ , so the estimate leans on the current observation with a small forecast weight. Even so, the score J still reflects multi-slot consequences that no myopic rule can see— fleet positions, battery depletion, forced returns, and charging occupancy propagated through the service simulator—while compounding demand error along the horizon is removed. In a preliminary smaller-scale setting $( K { = } 4 ,$ without the battery model), this choice improved both splits by +0.018 over the non-anchored variant at no cost (replicated from two checkpoints), and demand-forecasting enhancements were net negative; the interpretation is that the current observation already carries most of the actionable signal, so the model evaluates consequences of motions under it rather than outforecasting the environment. Algorithm 1 gives the per-slot procedure; an actor/critic direct mode exists as a fallback but is not used in the reported results.

## E. Training Procedure

Algorithm 2 summarizes offline training. A heuristic precollection phase gathers 60,000 transitions on training days only; the replay buffer never sees test-period data. Training then runs 30,000 gradient steps (learning rate $1 0 ^ { - 3 } )$ , which an internal scan showed is well past the plateau (curves flatten near 20k steps). Each step samples sequences, encodes them, unrolls the RSSM, and applies the four loss terms of (16): demand MSE in symlog space, free-bits KL (12), latent predictive regularization (15), and the service loss, plus an uncertainty calibration term for the ensemble. The EMA teacher updates after each gradient step. The production configuration is identical across all evaluated datasets, with no per-dataset tuning.

## F. Complexity Analysis

At decision time, DSWM performs $N _ { \mathrm { c e m } } \times I$ imagined rollouts of H steps each $( N _ { \mathrm { c e m } }$ candidates, I elite-refit iterations, initial std $\sigma _ { 0 } ;$ production values are hard-coded in the released experiment.py configuration). Every imagined step costs one GRU update, E demand-head evaluations, and one pass of the differentiable service simulator, all of which are dense tensor operations that batch across the $N _ { \mathrm { c e m } }$ samples. Perslot planning cost is therefore $O ( N _ { \mathrm { c e m } } I H )$ imagination forwards with a small constant, and it does not grow with the nonconvexity of the problem. In contrast, the BCD baseline JTORATC [6] invokes an SCA solver per decision window; measured solve times exceeded 8 s per slot in our $K = 1 6$ setting, which forced the documented degradation rule (one solve per 6 slots with reduced SCA iterations). DSWM’s planning cost is fixed per slot and independent of solver convergence, which suits online repositioning.

(a) Open-loop imagination: errors compound with horizon  
![](images/07bc5c1fe1470ab6b8e0a3e1690f5f5600efc22cec1a8bbec79ba451d5169155.jpg)

![](images/42586e26cbb3deb3d5256c2d98cb7c216a94cc20c70359355bd7ebd04b28c627.jpg)

![](images/62d11b3a9f36ae6361db9e153c75ce8ea26bb9729859654c81f668757d745e6d.jpg)

![](images/7914565444dc04c0c8bdf6f84fe01eb50c029689122f8e5c8a017d63c642c8e5.jpg)  
Fig. 3. Decomposed demand-service physics head. An E-head ensemble predicts the next demand map in symlog space; a differentiable service simulator replays association → probabilistic LoS (1) → Shannon rate (3) → capped per-cell fulfillment → aggregate SRc (17). Ensemble disagreement defines th uncertainty u<sub>t</sub>.

![](images/f08bf7787698b5232313d817ea0fa4c41ca7ffd2f9f954f894c441ddb7886782.jpg)  
(b) Observation anchoring ( =1): re-plan every slot

![](images/47e075eddf626bc1a90372482e6d311191482890e2b5f24fc19523212b250dc0.jpg)  
Fig. 4. Observation-anchored CEM planning. Candidate action sequences are imagined for $H = 1 5$ slots through the RSSM prior; at each imagined step the demand estimate is anchored on the current observation with mixing coefficient $\rho = 0 . 9 5$ (20); rollouts are scored by J in (19); elites refit the sampling distribution and the first action is executed.

Algorithm 2 DSWM Offline Training   
Require: training-days environment; collect steps $= 6 0 { , } 0 0 0 ;$ gradi  
ent steps = 30,000; $\mathrm { l r } = 1 0 ^ { - 3 }$   
1: Pre-collect 60,000 transitions into replay (training days only)   
2: for step = 1 to 30,000 do   
3: Sample a batch of sequences from replay   
4: Encode observations; unroll RSSM via $( 9 ) - ( 1 1 )$   
5: Demand ensemble: symlog MSE loss $L _ { \mathrm { d e m } } \left( E _ { \right. }$ heads, mean)   
6: Free-bits KL loss $L _ { \mathrm { K L } }$ by (12) (weights 0.8/0.2)   
7: Latent predictive regularization L<sub>LPR</sub> by (15) (EMA teacher   
+ variance margin)   
8: Decomposed service loss $L _ { \mathrm { s e r v e } }$ via (17); uncertainty calibra  
tion via (18)   
9: Update parameters with ∇L<sub>WM</sub> (16); EMA-update teacher   
10: end for

## V. EXPERIMENTS

## A. Datasets and Preprocessing

We evaluate on three real datasets with one unified pipeline; Fig. 5 visualizes the demand statistics, and the splits are declared per dataset below.

Milan (main dataset). The Telecom Italia CDR dataset [52] (2013, internet modality), cropped to $2 0 \times 2 0$ cells of $c =$ 235 m. Training: Nov. 1–10 (10 days); testing: Dec. 21–Jan. 1 (12 days), containing the Christmas–New Year OOD week. Demand is normalized by the training-segment 99th percentile. Milan demand is extremely dispersed: the top 5% cells carry only 13% of demand, with a clear commute bimodal pattern.

Shanghai Telecom. A public cellular dataset (2014-06– 2014-11; 6.95M raw records, cleaned to 6.15M). Demand is the number of Internet session starts per cell per slot. Sparsity-adaptive preprocessing coarsens cells to $c = 4 7 0$ m with a 3-slot sliding-window density estimate (zero-value rate 85.3%, i.e., genuinely sparse sessions). Training: June– October (153 days, including the merged validation segment); testing: November (30 days). Weekends form the holiday (OOD) split.

YJMob100K. The YJMob100K dataset [53] (Zenodo 10142719, 2023), 111.5M rows. We declare the semantics explicitly: demand is the number of distinct user IDs present per cell per slot—an approximately 5% sampled mobility proxy, not traffic bytes. Native 30-min slots are conservatively de-aggregated into 10-min slots (uniform within-cell division by 3, no information injected), and the densest $9 \times 9$ cells at $c = 5 0 0$ m are cropped $( 4 . 5 \times 4 . 5 $ km, Milan-matched scale). Training: days 0–55; testing: days 56–74 (19 days). Week structure is recovered from autocorrelation (peak lag 7); pseudo-weekend labels (21 days) are inferred and declared as such. Intensity is aligned across datasets by scaling factors $( \kappa = 3 . 8 1$ for Shanghai, $\kappa = 1 . 7 4 9$ for YJMob100K).

![](images/bf79f4ce30d142270b303008567123224ccfd7842c732a884ee4b32832ea8ec5.jpg)

![](images/f6cc5db99528304e1cb5016f35ec08a4d53e756e729121d8c4210e1cee046107.jpg)  
Fig. 5. Exploratory data analysis. Left: daily demand cycle (weekday vs. weekend; Milan shows a commute bimodal pattern). Right: daily total drift across the observation period, with the temporal train/test split and the Christmas–New Year OOD week marked. The spatial demand map is shown in Fig. 1. Milan demand is the most dispersed (top 5% cells carry 13% of demand, vs. 22% for YJMob100K), and we show in Section V-I that this dispersion is what makes dynamic repositioning most valuable here.

![](images/6bbfd40d2cfabb6dc2461900cc3d5f6537e63eb5ad3462cad42fa20574bfbb8e.jpg)

![](images/45c6bfb901b5fac36cedce6dc90e950874deca8a34816a0f0c6ef9e8b7af1572.jpg)  
Fig. 6. OOD analysis over the test period. (a) per-method OOD degradation (weekday minus holiday); (b) per-day served-ratio scatter, including the holiday OOD week. DSWM is the most stable method and scores higher on holidays (0.903) than weekdays (0.889). GA-MATR attains the highest holiday score among learning baselines (0.854) and, together with TD-MPC, is one of only two learning baselines whose holiday score surpasses Greedy; JTORATC suffers OOD backfire (0.495 vs. 0.583). Degradation values are computed from unrounded per-day series.

## B. Setup and Evaluation Protocol

Table II lists the scenario parameters with their justification. All learning methods receive the identical observation set (past $\ w \ = \ \mathrm { ~ 6 ~ }$ demand maps, current demand, fleet position/battery/charge, time phase), and all actions are $( \Delta x , \Delta y , \mathrm { b w \_ l o g i t } ) \in [ - 1 , 1 ] ^ { K \times 3 }$

The evaluation protocol is designed for credibility: temporal train/test splits (random splits forbidden); fixed evaluation days, weekday (in-distribution, ID) {0, 1, 2} and holiday (OOD) {3, 4, 10, 11} (for Milan, Dec. 21–23 and Dec. 24– 25/31/Jan. 1); 5 seeds on Milan and 3 per dataset. We report ±std where the spread changes adjacent-ranking interpretation (DSWM, GA-MATR, JTORATC); per-day variability appears in Fig. 6(b).

Two engineering details support reproducibility: environment outputs drift by order $1 0 ^ { - 4 }$ across days (BLAS/CPU variation), so all regression comparisons use frozen same-day references; and checkpoints are written only by the Milan full run (cross-dataset runs persist metrics only), so all reported numbers come from persisted per-seed result files, never from logs or probe curves.

## C. Baseline Lineup

Fourteen methods are compared. Seven are learning-based: PPO, SAC, MADDPG, TD-MPC [23], and three state-ofthe-art baselines re-implemented from recent works—GA-MATR [13] (graph-attention MARL with fairness regularizer), HRL-TPRA [16] (hierarchical trajectory planning with per-UAV resource actors), and EMORL-TCTO [14] (evolutionary multi-objective RL). Four are optimization-based or classical: JTORATC [6] (weighted BCD with KKT bisection and SCA, re-implemented), INS-WOA [7] (K-means/Voronoi predeployment with whale optimization, re-implemented), Static (K-means placement), and Greedy (per-slot BCD). Together with GA-MATR, HRL-TPRA, and EMORL-TCTO above, five IEEE baselines (2023–2026) are reproduced from their papers.

![](images/8da97b0ea33f67e7b436a209c0e4f59a67c14a6b91ab722b68bce6996c14f5a7.jpg)

![](images/600f8b1f91ea1de2977873ca87005160a501bd81f71e9232b46b1586fa0b8c52.jpg)

![](images/2aa4e6d44b79eb7c6b7b70127d12b32678c0ad2dceaf8c03194cbec0d059efb6.jpg)  
Fig. 7. Milan main results for all 14 methods: (a) served ratio (weekday ID / holiday OOD), (b) energy per slot, (c) energy efficiency. Served ratio is the only strongly discriminative metric; energy differs by at most ∼3%. Fairness is reported in the text.  
TABLE III

TABLE II  
SCENARIO PARAMETERS AND JUSTIFICATION  
MILAN MAIN RESULTS: SERVED RATIO (WEEKDAY ID / HOLIDAY OOD), 5 SEEDS
<table><tr><td>Parameter</td><td>Value</td><td>Basis</td></tr><tr><td>Fleet size K</td><td>16</td><td>city-wide; {2, 8} scaling</td></tr><tr><td>Altitude hfly / speed Vmax</td><td>100 m / 25 m/s</td><td>low-altitude [5]</td></tr><tr><td>Move limit  $\mathrm { \bar { \it D } _ { m a x } } / \mathrm { \ s l o t }$ </td><td>1500 m / 10 min</td><td>per-slot motion limit</td></tr><tr><td>Carrier  $f _ { c }$  / bandwidth B</td><td>2 GHz /  20 MHz</td><td>sub-6GHz access</td></tr><tr><td>Power / noise  $N _ { 0 }$ </td><td>0.5 W / 10−20.4</td><td>W/Hz small-cell</td></tr><tr><td>Rate demand  $r _ { \mathrm { 0 } } ~ /$  threshold  $U _ { \mathrm { p e a k } }$ </td><td>0.5 Mbps / 0.8</td><td>per-user</td></tr><tr><td>Battery  $E _ { \mathrm { b a t t } }$ </td><td>548Wh</td><td>DJI M300-class</td></tr><tr><td>Reserve / recharge</td><td>0.15 / 6 slots</td><td>forced return; 1 h charge</td></tr><tr><td>Charging stations</td><td>4</td><td>one per quadrant</td></tr><tr><td>Reward weights  $\alpha , \beta$ </td><td>1.0, 1.0</td><td>equal coverage/energy</td></tr><tr><td>Obs. window w / horizon H</td><td>6 / 15 slots</td><td>1 h hist.  $/ 2 . 5$  h ahead</td></tr><tr><td>Anchor  $\rho ~ / ~$  uncertainty  $\lambda _ { u }$ </td><td>0.95 /  1.0</td><td>dual-ckpt; abl. Section V-F</td></tr><tr><td>Learning rate</td><td> $1 0 ^ { - 3 }$ </td><td>fixed across datasets</td></tr><tr><td>Gradient steps</td><td> $^ { 3 0 , 0 0 0 }$ </td><td>20k-step plateau observed</td></tr></table>

DSWM and two internal ablations (w/o unc, w/o dec) complete the lineup.

## D. Main Results

Table III and Fig. 7 report the Milan results. Two findings stand out.

First, DSWM ranks first among non-ablated configurations (the w/o-uncertainty variant is analyzed in Section V-F) with $0 . 8 8 9 \pm 0 . 0 0 5$ on weekdays and 0.903 on holidays, 0.109 above the strongest external baseline (Greedy, 0.780). Second, energy (1210–1247 kJ, ∼3% spread) and fairness discriminate weakly. All methods attain Jain’s index no lower than 0.93 on weekdays and 0.88 on holidays; the geometrybased methods (Static, Greedy, INS-WOA) reach 0.999– 1.000, since the per-UAV demand-proportional bandwidth split guarantees fairness structurally, and only the more stochastic RL methods incur measurable loss (JTORATC 0.930/0.888, EMORL-TCTO 0.946/0.978, HRL-TPRA 0.956/0.968, MAD-DPG 0.978/0.990, weekday/holiday). Hover power dominates the Zeng model (4) and the battery state machine homogenizes flight patterns, so served ratio is the metric that separates policies; we report it as primary.

Greedy, GA-MATR, and TD-MPC cluster between 0.769 and 0.780 on weekdays: per-slot re-optimization and shorthorizon forecasts capture most of the attainable service under regular commute demand. INS-WOA (0.744) sits between Static (0.736) and Greedy, consistent with its Voronoi predeployment inheriting K-means’ sensitivity to demand shape.

<table><tr><td>Method</td><td>Class</td><td>Weekday</td><td>Holiday</td></tr><tr><td>DSWM (ours)</td><td>world model</td><td>0.889±0.005</td><td>0.903</td></tr><tr><td>w/o unc</td><td>ablation</td><td>0.894</td><td>0.902</td></tr><tr><td>w/o dec</td><td>ablation</td><td>0.769</td><td>0.787</td></tr><tr><td>Greedy (BCD)</td><td>model-free</td><td>0.780</td><td>0.792</td></tr><tr><td>GA-MATR [13]</td><td>MARL</td><td>0.770±0.025</td><td>0.854</td></tr><tr><td>TD-MPC [23]</td><td>model-based</td><td>0.769</td><td>0.805</td></tr><tr><td>INS-WOA [7]</td><td>optimization</td><td>0.744</td><td>0.752</td></tr><tr><td>Static (K-means)</td><td>deployment</td><td>0.736</td><td>0.748</td></tr><tr><td>PPO</td><td>model-free RL</td><td>0.723</td><td>0.791</td></tr><tr><td>SAC</td><td>model-free RL</td><td>0.682</td><td>0.757</td></tr><tr><td>MADDPG</td><td>MARL</td><td>0.648</td><td>0.737</td></tr><tr><td>EMORL-TCTO [14] multi-obj. RL</td><td></td><td>0.614</td><td>0.751</td></tr><tr><td>JTORATC [6]</td><td>BCD optim.</td><td>0.583±0.110</td><td>0.495</td></tr><tr><td>HRL-TPRA [16]</td><td>hier. RL</td><td>0.485</td><td>0.586</td></tr></table>

The hierarchical and multi-objective RL methods (EMORL-TCTO 0.614, HRL-TPRA 0.485) trail the table; Section V-J reports the diagnostic chain behind these negative results.

## E. OOD Analysis

Fig. 6 shows per-day served ratios across the Christmas– New Year OOD week. DSWM is the most stable method under shift: its holiday score (0.903) exceeds its weekday score. GA-MATR attains the highest holiday score among learning baselines (0.854, up from 0.770) and, together with TD-MPC (0.769→0.805), is one of only two learning baselines surpassing Greedy (0.792) on holidays—graph attention with a fairness regularizer has genuine value under shift; even so, it trails DSWM by 0.119 on weekdays. JTORATC shows the reverse failure mode (holiday 0.495 below weekday 0.583), an OOD backfire consistent with BCD re-solves locking onto stale demand snapshots. Model-free RL baselines (PPO, SAC, MADDPG) also gain 0.07–0.09 on holidays, but from weekday scores 0.17–0.24 below DSWM’s; DSWM improves by 0.014 from the highest base.

## F. Ablation

Fig. 8 reports the two structural ablations and a counterfactual demand-scaling intervention do(s). Removing the decomposed service head and regressing $\mathrm { S R } _ { t }$ directly from the latent state (w/o dec) drops the weekday score from 0.889 to 0.769 (−0.120), returning the method to Greedy’s level: the coverage gain comes from the differentiable service physics, not latent capacity. Removing the uncertainty penalty (w/o unc) raises the weekday score by 0.004 (0.894 vs. 0.889), with the holiday score essentially unchanged (0.902 vs. 0.903): in our setting the term is insurance against overconfident imagined branches, not a gain source, consistent with offline model-based RL [50] where pessimism matters most near the data boundary. The counterfactual do(s) probe in Fig. 8(b) scales the imagined demand by $s \in \{ 0 . 5 , 2 \}$ at planning time: DSWM stays near its s=1 reference under both scalings and degrades less than the w/o-dec variant, which again points to the differentiable service physics—not latent capacity—as the robustness source.

![](images/7012a3ba1e1c80f4da9006ff1c26261c7aa16ff6eaf74e6a3ccf5521b7d34beb.jpg)

![](images/f4b25d9a49886558d74e9fc18c68d92186a4cccfc70d52d2274abc2ad62208d2.jpg)

Fig. 8. Ablation and counterfactual analysis on Milan. Left: removing the decomposed service head costs 0.120 of served ratio (0.769 vs. 0.889), while removing the uncertainty penalty gains 0.004 (insurance, not a gain source). Right: counterfactual demand-scaling do(s) intervention (imagined demand scaled by s; dashed/dotted references at s=1).  
![](images/c7788d38247e58641e7556f9f9a316cdb23fa6c39eabb2e16713896354af83ea.jpg)  
Fig. 9. Sample efficiency on Milan. Curves are probe-based and diagnostic only (fixed test-day probe; no early stopping or model selection). DSWM’s prob flattens near 20k gradient steps; the production budget is 30k steps.

![](images/8606f4eb372f4e34a2a248da8da2125e9577108df263603e7d25591408af7639.jpg)

![](images/d4fe779da35bd71d1847cbc89a4464be429189300edb4c48f8fae80d325252ca.jpg)

![](images/aa07979fd819fa6dce47f1e43b1edda888c634a8b1d6ed75d43936a89999c876.jpg)  
Fig. 10. Scalability in fleet size $K \in \{ 2 , 8 , 1 6 \}$ and demand intensity on Milan. DSWM’s advantage grows with fleet size, where coordination is hardest for myopic baselines.

## G. Sample Efficiency and Scalability

Fig. 9 plots served ratio against environment steps (log scale) under an explicitly stated probe protocol: probe-based and diagnostic only (fixed test-day probe; fixed budget with no early stopping or model selection; probe numbers never enter Tables III or IV). The x-axis counts environment interaction, including the 60,000 pre-collected transitions. DSWM’s probe curve flattens early, which motivated the 30k-step gradient budget. Fig. 10 varies the fleet size $K \in \{ 2 , 8 , 1 6 \}$ and demand intensity: DSWM’s margin over Greedy grows with K, since coordinating a 16-UAV fleet is exactly where myopic methods lose most. Fig. 11 shows the training diagnostics (world-model loss, demand MSE, KL, and latent-prediction similarity terms); these use the same test-day probe protocol and are diagnostic

![](images/055b6cd4461c0b8e9edb3c7fa150d07329caa2ea73af12bd88b9da07dc0b048d.jpg)

![](images/26c01921cf78606d15792be2a3d2069f5b72fa791fb0832e90154256f0d339f3.jpg)

![](images/a1c95b9b74097aa5ab400ab1b7704e9791c8d5e8a650729394256dcedb3cc9bf.jpg)

![](images/102869db115700efbc1cef2b1513a01399daac0520ee2c19dff290fa4aee74f0.jpg)  
Fig. 11. Training diagnostics of DSWM: world-model loss, demand MSE, free-bits KL, and latent-prediction similarity terms. Probe curves use the testday diagnostic protocol (probe-based, diagnostic only).

![](images/770e964161f1a6db7e22c941815c89df6d1dd676544be02599dced0909da5adf.jpg)

![](images/a02a62ee68949127bcabda4afb0f5369948bda7c33cbca3c5ff33e035061a082.jpg)  
Fig. 12. World-model quality on the Milan test period. (a) Demand RMSE vs. imagination rollout step. (b) Spearman correlation $r _ { s }$ between imagined and realized returns; each point is one seed–evaluation-day pair.

only.

## H. World Model Quality

Fig. 12 evaluates the learned model along two axes: demand RMSE over the imagination horizon, and per-seed Spearman rank correlation $r _ { s }$ between imagined and realized returns. Demand RMSE stays flat (0.128 Mbps at one step, 0.119 Mbps at 24 slots), and the imagined-return ranking correlates moderately with the realized one $( r _ { s } = 0 . 5 2 0 \pm 0 . 0 8 7$ ; each point in Fig. 12(b) is one seed–evaluation-day pair): the model preserves where demand concentrates far better than exact magnitudes—the quantity anchored planning consumes.

TABLE IV  
CROSS-DATASET RESULTS: WEEKDAY / HOLIDAY SERVED RATIO (MEAN; 5 SEEDS FOR MILAN, 3 SEEDS FOR SHANGHAI AND YJMOB; COLUMNS: MILAN (2013, CDR), SHANGHAI (2014, SESSIONS), YJMOB (2023, 4.5 KM)). PER-SEED STD WHERE IT AFFECTS ADJACENT RANKINGS: DSWM ±0.006 (SHANGHAI), ±0.002 (YJMOB); GA-MATR ±0.058 (SHANGHAI), ±0.039 (YJMOB).
<table><tr><td>Method</td><td>Milan</td><td>Shanghai</td><td>YJMob</td></tr><tr><td>DSWM (ours)</td><td>0.889/0.903</td><td>0.908/0.862</td><td>0.898/0.891</td></tr><tr><td>TD-MPC</td><td>0.769/0.805</td><td>0.875/0.839</td><td>0.818/0.831</td></tr><tr><td>PPO</td><td>0.723/0.791</td><td>0.810/0.748</td><td>0.879/0.870</td></tr><tr><td>GA-MATR</td><td>0.770/0.854</td><td>0.745/0.708</td><td>0.820/0.790</td></tr><tr><td>Greedy</td><td>0.780/0.792</td><td>0.746/0.741</td><td>0.762/0.760</td></tr><tr><td>INS-WOA</td><td>0.744/0.752</td><td>0.816/0.788</td><td>0.753/0.756</td></tr><tr><td>Static</td><td>0.736/0.748</td><td>0.437/0.420</td><td>0.654/0.622</td></tr></table>

## I. Cross-Dataset Generalization

Table IV and Fig. 13 extend the comparison to Shanghai and YJMob (2023, 4.5km)<sup>1</sup> under the identical production configuration with no per-dataset tuning. DSWM ranks first among non-ablated configurations on all three datasets: 0.908 on Shanghai (+0.033 over TD-MPC), 0.898 on YJMob100K (+0.019 over PPO), and 0.889 on Milan (+0.109 over Greedy). The margin ordering is informative: Milan’s demand is the most dispersed (top 5% cells: 13%) at a 4.7-km scale, where dynamic repositioning is most valuable and DSWM’s margin largest; YJMob100K demand is more concentrated (22%) and Shanghai’s extremely sparse (85.3% zero cells), where static or myopic coverage already captures much of the achievable service and margins shrink. Spatial scale and dispersion thus moderate the value of dynamic repositioning. Two failure cases support the same reading from the negative side: on Shanghai, Static collapses to 0.437 because K-means placement fails on 85%-sparse demand; on YJMob100K, learning methods regain their normal ordering once slot physics and geographic scale match Milan’s.

## J. Discussion: Lessons Learned

The development process produced findings that we report as lessons, including negative results.

1) Per-agent embedding checks are mandatory for MARL baselines. Our first GA-MATR re-implementation collapsed: two global graph-attention layers flattened per-UAV embeddings (cross-agent std = 0), degenerating multi-agent learning into a single-agent policy; only a variance check and symmetry-breaking fixes recovered 0.770.

2) Hierarchical complexity can be a liability. HRL-TPRA was kernel-launch bound (serial pointer decoding plus 16 per-UAV actor updates per macro-slot); even after a 5.7× speedup it ranked last among learning baselines (0.485).

3) Environment drift requires frozen same-day references. Identical code rerun on a different day deviates by $\sim 1 0 ^ { - 4 }$ (BLAS/CPU variation).

![](images/0e6540d8bf720c14250fb9e2e7db2c122feda41162da6adb4d76faa442b0f750.jpg)  
Fig. 13. Cross-dataset comparison (Milan (2013, CDR), Shanghai (2014, sessions), YJMob (2023, 4.5km); 7 methods; 5 seeds for Milan, 3 for Shanghai/YJMob). DSWM ranks first among non-ablated configurations on all three datasets under one configuration with no per-dataset tuning.

4) Model-free methods are not always safe. Static (Kmeans) collapses to 0.437 on 85%-sparse Shanghai demand.

5) OOD strength exists outside world models. GA-MATR’s graph attention with a fairness term yields real robustness under shift (Section V-E).

6) Pre-deployment inherits clustering bias. INS-WOA’s Voronoi stage inherits K-means’ sensitivity to demand shape, landing mid-table on all three datasets.

## K. Qualitative Visualization

Figs. 14–16 give qualitative evidence. Fig. 14 compares the world model’s predicted demand map against ground truth over the full Milan city grid, together with the error map: predictions track the commute-driven hotspot migration, and errors concentrate at cell boundaries during peak transitions. Figs. 15 and 16 show full-region demand, ground truth, and weekday/weekend contrasts for Shanghai and YJMob100K; the weekday–weekend shift that defines each dataset’s OOD split is visible as a spatial redistribution rather than a uniform intensity change, which is the regime where anchored planning helps most.

## L. Limitations

Two limitations qualify our claims. First, YJMob100K demand is a distinct-user presence proxy (5% sampling), not traffic bytes, so cross-dataset conclusions about absolute served ratios should be read with the declared semantics. Second, energy and fairness discriminate weakly under hoverdominated power and battery constraints (∼3% energy spread, Jain 0.888–1.000), so our claims rest on served ratio.

## VI. CONCLUSION

We formulated demand-driven UAV-BS fleet repositioning as a decision-time planning problem in a learned latent space and proposed DSWM, a world model of the demand-service dynamics built around three mutually reinforcing designs. The decomposed demand-service physics head makes the coverage mechanism differentiable inside the model, so the model is trained on the quantity the operator actually cares about rather than a latent proxy. The latent predictive regularization objective keeps the learned dynamics stable and collapsefree without reconstructing heavy-tailed demand maps. And observation-anchored CEM planning grounds every imagined rollout in the current observation instead of an open-loop forecast, which proves to be the decisive choice: across three real-world datasets, a broad lineup of learning, optimization, and classical baselines, and both in-distribution and holiday out-of-distribution weeks, DSWM consistently delivered the best demand coverage, including under demand regimes much sparser or more concentrated than the training distribution. The evidence supports the paper’s central thesis: for this class of problems, the binding constraint on coverage is not how accurately demand is forecast, but how the current observation is exploited at decision time. The same design guideline carries over to other partially observed network control problems where the physical service model is known but the exogenous demand process is not, and more broadly to agentic network controllers that must perceive, remember, reason, and coordinate under uncertainty. Future work includes byte-accurate demand semantics on mobility datasets, heterogeneous fleets, and energy-aware altitude control.

## REFERENCES

[1] M. Mozaffari, W. Saad, M. Bennis, Y.-H. Nam, and M. Debbah, “A tutorial on UAVs for wireless networks: Applications, challenges, and open problems,” IEEE Communications Surveys & Tutorials, vol. 21, no. 3, pp. 2334–2360, 2019.

[2] Z. Ning, T. Li, Y. Wu, X. Wang, Q. Wu, F. R. Yu, and S. Guo, “6G communication new paradigm: The integration of autonomous aerial vehicles and intelligent reflecting surfaces,” IEEE Communications Surveys & Tutorials, vol. 27, no. 6, pp. 3382–3416, 2025.

[3] F. Khoramnejad and E. Hossain, “Generative AI for the optimization of next-generation wireless networks: Basics, state-of-the-art, and open challenges,” IEEE Communications Surveys & Tutorials, vol. 27, no. 6, pp. 3483–3525, 2025.

[4] Y. Zeng and R. Zhang, “Energy-efficient UAV communication with trajectory optimization,” IEEE Transactions on Wireless Communications, vol. 16, no. 6, pp. 3747–3760, 2017.

[5] A. Al-Hourani, S. Kandeepan, and S. Lardner, “Optimal LAP altitude for maximum coverage,” IEEE Wireless Communications Letters, vol. 3, no. 6, pp. 569–572, 2014.

![](images/e8b02a5f78de361f7c49416f93b6cc09588594c8c8726f7fa6daf6442ec42e16.jpg)

![](images/3a8c3f70b92f9666ac204d851fcbe610b9ea10938010f8380396c9a8727665a0.jpg)

![](images/89e1b3ec0ff41ea4633c98ade60d6843bcd73ed388eb57ba3ebe343e3a741cae.jpg)

![](images/160095b5e3231b2c8206985f3a2c7c3d55ed0c6e96558072b3135fceac167471.jpg)  
Fig. 14. World-model prediction visualization on Milan. From left to right: full-city context, ground-truth demand map, predicted demand map, and per-cell error. Predictions track the commute-driven hotspot migration; errors concentrate at cell boundaries during peak transitions.

![](images/e4007e9172262e9a31182ffc948f6768780e49df14bacdc91e5363c38832b3c5.jpg)

![](images/2390c3e9f1c5d4fd1e0fc457310d497f4c28f7eef4178eccac0df59e4ce8a41c.jpg)

![](images/e176b0d02a4c379079e2f949b0fc61e148a00d270d99b9aaacd00437733ad879.jpg)

![](images/65e46d2338a01c3ce43055de1fc37bed7178cedef08260df2b59bd1c6ebfeda9.jpg)  
Fig. 15. Shanghai dataset visualization. From left to right: full-region demand, ground truth, weekday profile, and weekend (OOD) profile. The weekday– weekend shift is a spatial redistribution of sparse sessions (85.3% zero cells), not a uniform intensity change.

![](images/e7406664e3fbef6fb69fbe3947c85f2ccddfb173e0c8fa3be8d4b7e6e4d2dd0c.jpg)

![](images/df8ca75be4a7f0cef248cc2043d260e8edb0aa11dde0a0ec2e5cc7b41bcfe228.jpg)

![](images/12129ef5db5148ce23219132dccca92be24e883c2d56974717d806233df33d46.jpg)

![](images/305227c925c936c6aef8dd4d903f544e34ffadc483640f1ba45ebf5a273ea631.jpg)  
Fig. 16. YJMob100K dataset visualization. From left to right: full-region demand, ground truth, weekday profile, and pseudo-weekend (OOD) profile. Demand semantics are distinct-user presence counts (a 5% sampled mobility proxy), as declared in Section V-A; weekend labels are inferred from the autocorrelationrecovered week structure.

[6] G. Sun, Y. Wang, Z. Sun, Q. Wu, J. Kang, D. Niyato, and V. C. Leung, “Multi-objective optimization for multi-UAV-assisted mobile edge computing,” IEEE Transactions on Mobile Computing, vol. 23, no. 12, pp. 14 803–14 820, 2024.

[7] Z. Jia, J. He, L. He, M. Sheng, J. Liu, Q. Wu, and Z. Han, “Dynamic trajectory optimization and power control for hierarchical UAV swarms in 6G aerial access network,” IEEE Transactions on Wireless Communications, 2025.

[8] J. Cui, Y. Liu, and A. Nallanathan, “Multi-agent reinforcement learningbased resource allocation for UAV networks,” IEEE Transactions on Wireless Communications, vol. 19, no. 2, pp. 729–743, 2019.

[9] Z. Qin, Z. Liu, G. Han, C. Lin, L. Guo, and L. Xie, “Distributed UAV-BSs trajectory optimization for user-level fair communication service with multi-agent deep reinforcement learning,” IEEE Transactions on Vehicular Technology, vol. 70, no. 12, pp. 12 290–12 301, 2021.

[10] L. Wang, K. Wang, C. Pan, W. Xu, N. Aslam, and A. Nallanathan, “Deep reinforcement learning based dynamic trajectory control for UAV-assisted mobile edge computing,” IEEE Transactions on Mobile Computing, vol. 21, no. 10, pp. 3536–3550, 2021.

[11] N. Zhao, Z. Ye, Y. Pei, Y.-C. Liang, and D. Niyato, “Multi-agent deep reinforcement learning for task offloading in UAV-assisted mobile edge computing,” IEEE Transactions on Wireless Communications, vol. 21, no. 9, pp. 6949–6960, 2022.

[12] Z. Ning, Y. Yang, X. Wang, Q. Song, L. Guo, and A. Jamalipour, “Multiagent deep reinforcement learning based UAV trajectory optimization for differentiated services,” IEEE Transactions on Mobile Computing, vol. 23, no. 5, pp. 5818–5834, 2023.

[13] Z. Feng, D. Wu, M. Huang, and C. Yuen, “Graph-attention-based reinforcement learning for trajectory design and resource assignment in

multi-UAV-assisted communication,” IEEE Internet of Things Journal, vol. 11, no. 16, pp. 27 421–27 434, 2024.

[14] F. Song, H. Xing, X. Wang, S. Luo, P. Dai, Z. Xiao, and B. Zhao, “Evolutionary multi-objective reinforcement learning based trajectory control and task offloading in UAV-assisted mobile edge computing,” IEEE Transactions on Mobile Computing, vol. 22, no. 12, pp. 7387– 7405, 2022.

[15] Y. Wang, Y. Wan, W. Zuo, S. Wang, Y.-C. Wu, C. Xu, and H. Arslan, “LAGS: Low-altitude gaussian splatting with groupwise heterogeneous graph learning,” IEEE Transactions on Vehicular Technology, 2026.

[16] W. Yuan, S. Chen, H. He, Y. Hou, S. Chen, X. Tan, and J. Yang, “Hierarchical reinforcement learning based joint trajectory planning and resource allocation in UAV-assisted IoT-sensor networks,” IEEE Transactions on Communications, 2025.

[17] Y. Yao, B. Gu, Z. Su, and M. Guizani, “MVSTGN: A multi-view spatial-temporal graph network for cellular traffic prediction,” IEEE Transactions on Mobile Computing, vol. 22, no. 5, pp. 2837–2849, 2021.

[18] B. Gu, J. Zhan, S. Gong, W. Liu, Z. Su, and M. Guizani, “A spatialtemporal transformer network for city-level cellular traffic analysis and prediction,” IEEE Transactions on Wireless Communications, vol. 22, no. 12, pp. 9412–9423, 2023.

[19] Q. Wang, S. Wang, D. Zhuang, H. Koutsopoulos, and J. Zhao, “Uncertainty quantification of spatiotemporal travel demand with probabilistic graph neural networks,” IEEE Transactions on Intelligent Transportation Systems, vol. 25, no. 8, pp. 8770–8781, 2024.

[20] C. Liu, K. H. Hettige, Q. Xu, C. Long, S. Xiang, G. Cong, Z. Li, and R. Zhao, “ST-LLM+: Graph enhanced spatio-temporal large language models for traffic prediction,” IEEE Transactions on Knowledge and Data Engineering, vol. 37, no. 8, pp. 4846–4859, 2025.

[21] C. Zhang, Y. Zhang, Q. Shao, B. Li, Y. Lv, X. Piao, and B. Yin, “ChatTraffic: Text-to-traffic generation via diffusion model,” IEEE Transactions on Intelligent Transportation Systems, vol. 26, no. 2, pp. 2656–2668, 2024.

[22] D. Hafner, T. Lillicrap, I. Fischer, R. Villegas, D. Ha, H. Lee, and J. Davidson, “Learning latent dynamics for planning from pixels,” in International Conference on Machine Learning. PMLR, 2019, pp. 2555–2565.

[23] N. Hansen, X. Wang, and H. Su, “Temporal difference learning for model predictive control,” arXiv preprint arXiv:2203.04955, 2022.

[24] D. Hafner, J. Pasukonis, J. Ba, and T. Lillicrap, “Mastering diverse control tasks through world models,” Nature, vol. 640, no. 8059, pp. 647–653, 2025.

[25] M. Assran, A. Bardes, D. Fan, Q. Garrido, R. Howes, M. Muckley, A. Rizvi, C. Roberts, K. Sinha, A. Zholus et al., “V-JEPA 2: Selfsupervised video models enable understanding, prediction and planning,” arXiv preprint arXiv:2506.09985, 2025.

[26] Z. Jia, C. Cui, C. Dong, Q. Wu, Z. Ling, D. Niyato, and Z. Han, “Distributionally robust optimization for aerial multi-access edge computing via cooperation of UAVs and HAPs,” IEEE Transactions on Mobile Computing, vol. 24, no. 10, pp. 10 853–10 867, 2025.

[27] Z. Ning, H. Ji, X. Wang, E. C. Ngai, L. Guo, and J. Liu, “Joint optimization of data acquisition and trajectory planning for UAV-assisted wireless powered Internet of Things,” IEEE Transactions on Mobile Computing, vol. 24, no. 2, pp. 1016–1030, 2024.

[28] Z. Sun, G. Sun, Q. Wu, L. He, S. Liang, H. Pan, D. Niyato, C. Yuen, and V. C. Leung, “TJCCT: A two-timescale approach for UAV-assisted mobile edge computing,” IEEE Transactions on Mobile Computing, vol. 24, no. 4, pp. 3130–3147, 2024.

[29] S. Peng, B. Li, L. Liu, Z. Fei, and D. Niyato, “Trajectory design and resource allocation for multi-UAV-assisted sensing, communication, and edge computing integration,” IEEE Transactions on Communications, vol. 73, no. 4, pp. 2847–2861, 2024.

[30] D. Deng, W. Zhou, X. Li, D. B. da Costa, D. W. K. Ng, and A. Nallanathan, “Joint beamforming and UAV trajectory optimization for covert communications in ISAC networks,” IEEE Transactions on Wireless Communications, vol. 24, no. 2, pp. 1016–1030, 2024.

[31] G. Cheng, X. Song, Z. Lyu, and J. Xu, “Networked ISAC for lowaltitude economy: Coordinated transmit beamforming and UAV trajectory design,” IEEE Transactions on Communications, vol. 73, no. 8, pp. 5832–5847, 2025.

[32] X. Li, W. Mao, X. Xu, Y. Liu, H. Zhang, W. Huangfu, and K. Long, “Self-adjusting network slicing for dynamic heterogeneous task offloading in UAV-enabled mobile edge computing,” IEEE Transactions on Cognitive Communications and Networking, vol. 12, pp. 673–687, 2025.

[33] Y. Bai, B. Xie, Y. Liu, Z. Chang, and R. Jantti, “Dynamic UAV¨ deployment in multi-UAV wireless networks: A multimodal-featurebased deep reinforcement learning approach,” IEEE Internet of Things Journal, vol. 12, no. 12, pp. 18 765–18 778, 2025.

[34] L. T. Hoang, C. T. Nguyen, H. D. Le, and A. T. Pham, “Adaptive 3D placement of multiple UAV-mounted base stations in 6G airborne small cells with deep reinforcement learning,” IEEE Transactions on Networking, vol. 33, no. 4, pp. 1989–2004, 2025.

[35] C. Liu, Y. Zhong, R. Wu, S. Ren, S. Du, and B. Guo, “Deep reinforcement learning based 3D-trajectory design and task offloading in UAVenabled MEC system,” IEEE Transactions on Vehicular Technology, vol. 74, no. 2, pp. 3185–3195, 2024.

[36] Z. Liu, J. Zhang, Y. Zeng, and B. Ai, “Energy-efficient multi-agent reinforcement learning for UAV trajectory optimization in cell-free massive MIMO networks,” IEEE Transactions on Wireless Communications, vol. 24, no. 7, pp. 5917–5930, 2025.

[37] G. Chen, G. Zhao, C. Xu, Z. Han, and S. Yu, “Spatiotemporalaware deep reinforcement learning for multi-UAV cooperative coverage in emergency deterministic communications,” IEEE Transactions on Vehicular Technology, 2025.

[38] I. A. Meer, K.-L. Besser, M. Ozger, D. Schupke, H. V. Poor, and C. Cavdar, “Hierarchical multi-agent DRL based dynamic cluster reconfiguration for UAV mobility management,” IEEE Transactions on Cognitive Communications and Networking, 2025.

[39] H. Peng and X. Shen, “Multi-agent reinforcement learning based resource management in MEC-and UAV-assisted vehicular networks,” IEEE Journal on Selected Areas in Communications, vol. 39, no. 1, pp. 131–141, 2020.

[40] J. Chen, X. Cao, P. Yang, M. Xiao, S. Ren, Z. Zhao, and D. O. Wu, “Deep reinforcement learning based resource allocation in multi-UAVaided MEC networks,” IEEE Transactions on Communications, vol. 71, no. 1, pp. 296–309, 2022.

[41] J. Ji, W. Zhang, J. Wang, and C. Huang, “Seeing the unseen: Learning basis confounder representations for robust traffic prediction,” in Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1, 2025, pp. 577–588.

[42] D. Ha and J. Schmidhuber, “World models,” arXiv preprint arXiv:1803.10122, 2018.

[43] D. Hafner, T. Lillicrap, M. Norouzi, and J. Ba, “Mastering Atari with discrete world models,” arXiv preprint arXiv:2010.02193, 2020.

[44] D. Hafner, W. Yan, and T. Lillicrap, “Training agents inside of scalable world models,” arXiv preprint arXiv:2509.24527, 2025.

[45] R. Balestriero and Y. LeCun, “LeJEPA: Provable and scalable self-supervised learning without the heuristics,” arXiv preprint arXiv:2511.08544, 2025.

[46] G. Zhou, H. Pan, Y. LeCun, and L. Pinto, “Dino-WM: World models on pre-trained visual features enable zero-shot planning,” arXiv preprint arXiv:2411.04983, 2024.

[47] J. Richens, D. Abel, A. Bellot, and T. Everitt, “General agents contain world models,” arXiv preprint arXiv:2506.01622, 2025.

[48] H. Chai, Y. Yuan, and Y. Li, “MobiWorld: World models for mobile wireless network,” arXiv preprint arXiv:2507.09462, 2025.

[49] R. Ding, F. Zhou, Q. Wu, and D. W. K. Ng, “From external interaction to internal inference: An intelligent learning framework for spectrum sharing and UAV trajectory optimization,” IEEE Transactions on Wireless Communications, vol. 23, no. 9, pp. 12 099–12 114, 2024.

[50] T. Yu, G. Thomas, L. Yu, S. Ermon, J. Y. Zou, S. Levine, C. Finn, and T. Ma, “MOPO: Model-based offline policy optimization,” Advances in Neural Information Processing Systems, vol. 33, pp. 14 129–14 142, 2020.

[51] Y. Chen, A. Bardes, Z. Li, and Y. LeCun, “Intra-instance vicreg: Bag of self-supervised image patch embedding,” arXiv preprint arXiv:2206.08954, vol. 2, 2022.

[52] G. Barlacchi, M. De Nadai, R. Larcher, A. Casella, C. Chitic, G. Torrisi, F. Antonelli, A. Vespignani, A. Pentland, and B. Lepri, “A multi-source dataset of urban life in the city of Milan and the Province of Trentino,” Scientific Data, vol. 2, no. 1, p. 150055, 2015.

[53] T. Yabe, K. Tsubouchi, T. Shimizu, Y. Sekimoto, K. Sezaki, E. Moro, and A. Pentland, “YJMob100K: City-scale and longitudinal dataset of anonymized human mobility trajectories,” Scientific Data, vol. 11, no. 1, p. 397, 2024.