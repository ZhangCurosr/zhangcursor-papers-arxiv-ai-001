# Dual Process Motion Planning

Jiayi Yan<sup>1</sup>, Francesco Fabiano<sup>2</sup>, Alessandro Abate<sup>2</sup>

<sup>1</sup>The Chinese University of Hong Kong, Shenzhen

jiayiyan@link.cuhk.edu.cn

<sup>2</sup>University of Oxford

francesco.fabiano@cs.ox.ac.uk, alessandro.abate@cs.ox.ac.uk

## Abstract

Robotic systems are deeply embedded in both industry and everyday life, where they are expected to act with speed, precision, and reliability. Classical control and planning methods have long delivered strong guarantees, but often at the cost of computational eficiency and adaptability. More recently, learning-based approaches have shown promise in overcoming these limitations, enabling agents to leverage experience to accelerate decision-making and address previously intractable problems. In this work, we bridge these two approaches through a neuro-symbolic perspective on nonlinear motion planning. Inspired by the Thinking Fast and Slow paradigm, we introduce a dual-process architecture that combines the strengths of robust reasoning and learning. Our framework integrates state-of-the-art symbolic solvers as a “System-2” component with experience-driven “System-1” modules. A metacognitive controller dynamically orchestrates their interaction, selecting when to rely on fast intuition versus slower, more precise reasoning. By evaluating the framework across diverse nonlinear benchmark environments, we demonstrate that this architecture yields consistent gains in planning eficiency, accuracy, and generalization, while promoting reuse across tasks. The results suggest that tightly coupling learning with structured reasoning ofers a scalable path toward more capable and adaptive robotic systems.

## Introduction

Motion planning is a core component of robotic systems and underpins a wide range of applications, including autonomous driving (Paden et al. 2016) and multi-agent coordination (Hegde and Panagou 2016). Its primary objective is to compute a high-quality, collision-free trajectory that connects a start state to a goal state while respecting system dynamics and constraints. However, real-world environments are often high-dimensional, continuous, and dynamically constrained, which makes motion planning computationally challenging, especially under strict real-time requirements (LaValle 2006; Karaman and Frazzoli 2011).

Classical motion planning methods can be broadly categorized into experience-based planning using ofline trajectory libraries and neural networks, and online optimization-based planning. Experience-based methods, such as those leveraging neural networks, amortize planning into a learned policy or trajectory generator that maps observations directly to controls or waypoints at runtime, enabling fast online inference and interpolation beyond a finite trajectory library, albeit typically with weaker guarantees (Ichter, Harrison, and Pavone 2017). Their efectiveness depends critically on selecting a trajectory that matches the current scenario, which is dificult in complex environments, hence often leading to suboptimal or unsafe performance.

In contrast, online optimization-based methods, including Model Predictive Control (MPC) (Mayne et al. 2000) and Control Barrier Function (CBF)-based safety filters (Ames et al. 2019), compute trajectories by solving constrained optimization problems that explicitly account for system dynamics and environmental constraints. While these methods reliably provide solutions, they incur in computational overhead, limiting their applicability in time-critical settings.

To bridge this gap, recent work has explored learningaugmented motion planning, where data-driven models are used to improve eficiency and performance (Mansard et al. 2018; Bjelonic et al. 2022; Carvalho et al. 2024). These methods improve convergence and solution quality, but they typically rely on a fixed pipeline that often hinders their eficiency. To address this limitation, we draw inspiration from a recent AI paradigm Fabiano et al. (2025) that is in turn informed by the dual-system theory Kahneman (2011). We propose Dual-MP, a dual-system motion-planning framework that exploits between fast experience-based and slower online solving.

We instantiate this idea within a SOFAI-style architecture (Pallagani et al. 2025). Dual-MP inherits the modular nature ofSOFAI: it uses both fast, experience-based and slow, deliberate solvers. We refer to the former category as System-1 (S1) and to the latter as System-2 (S2). These are arbitrated through a metacognitive (MC) agent. Dual-MP is equipped with a Neural Network S1: a neural policy trained from successful trajectories, and two general S2 online solvers: one based on MPC and one on CBF. The code used in this work is available online<sup>1</sup>.

We summarize our main contributions below:

• We propose Dual-MP, a modular S1/S2 architecture for nonlinear motion planning that arbitrates fast neural plan ning and deliberate symbolic solving.

• We instantiate the framework with a neural S1 policy and two nonlinear S2 solvers, based on MPC and CBFs, under a common MC planner interface.

• We add continual learning, where successful trajectories are reused to retrain S1 and improve future fast planning.

• We evaluate the framework across diverse nonlinear benchmark families, reporting success, runtime, S1/S2 usage, and trajectory quality.

## Related work

Classical Motion Planning. Current classical motion planning methods are primarily divided into two popular categories: experience-based planning using neural networks and online optimization-based planning.

Experience-based planning typically uses neural networks to amortize planning from previously solved problems, learning a direct mapping from observations, goals, and local state information to actions, waypoints, or full trajectories (Ichter, Harrison, and Pavone 2017; Fishman et al. 2023). Such methods can provide low-latency inference and strong empirical performance in complex environments by reusing structure learned from expert demonstrations, simulation data, or successful planning experience (Wang et al. 2021). While effective, such approaches typically do not preserve explicit models of dynamics, safety constraints, or solver confidence at inference time, which limits their interpretability and their ability to decide when the output should be trusted.

In contrast to experience-based planning methods, online optimization-based methods, including MPC and CBF-based safety filters, compute trajectories or controls by solving constrained optimization problems at runtime (Mayne et al. 2000; Ames et al. 2017). These methods directly incorporate system dynamics and environmental constraints, yielding high-quality and dynamically feasible solutions. However, they often incur significant computational overhead, which makes real-time deployment challenging in complex or high-dimensional scenarios.

Learning-Augmented Motion Planning. To mitigate this, recent work seeks to combine the eficiency of ofline planning with the accuracy of online optimization through learning-augmented hybrid approaches. A common strategy is to use learned models or ofline datasets to warm-start online solvers. For example, Memory-of-Motion learns a mapping from task descriptors to state-control trajectories and uses the resulting memory to initialize nonlinear predictive control (Mansard et al. 2018). Bjelonic et al. (2022) use ofline motion libraries as reference costs for online MPC, which allows long-horizon ofline behaviors to be executed through short-horizon feedback optimization. Transformerbased motion planners learn to restrict or guide the search space from prior data (Johnson et al. 2022), while difusionbased planners learn multimodal trajectory priors that can be sampled or adapted during planning (Carvalho et al. 2024). Closely related to our dual-system motivation, Fridovich-Keil et al. (2018) propose a “Planning, Fast and Slow” framework in which ofline safety computation enables safe switching among online planners. Their method provides a strong safety-aware planning module; in our terminology, such a framework can be viewed as a possible instantiation of S2.

Despite these advances, a key limitation of existing hybrid approaches is that they usually invoke the online solver regardless of how well the ofline or learned prior matches the current scenario. This leads to unnecessary computation in cases where a retrieved or neural trajectory is already suficient. More fundamentally, these methods lack a principled mechanism for deciding when online optimization is required. Dual-MP is designed to address exactly this allocation problem.

## Background

## Nonlinear Model Predictive Control

Model Predictive Control (MPC) is a widely used optimal control framework for motion planning under dynamic and environmental constraints (Mayne et al. 2000). At each time step, MPC solves a finite-horizon optimization problem to compute a control sequence that minimizes a cost function while satisfying system dynamics and constraints.

Consider nonlinear dynamics $\begin{array} { r c l } { \dot { x } } & { = } & { f ( x , u ) } \end{array}$ , or, after discretization, $x _ { k + 1 } ~ = ~ F _ { d } ( x _ { k } , u _ { k } )$ , where $\boldsymbol { x } _ { k } ~ \in ~ \mathbb { R } ^ { n }$ is the state and $u _ { k } \in \mathbb { R } ^ { m }$ is the control. Given the current state $x _ { 0 }$ and goal $x _ { g } .$ , nonlinear MPC solves min $\begin{array} { r } { \iota _ { u _ { 0 : N _ { h } - 1 } } \sum _ { k = 0 } ^ { N _ { h } - 1 } \ell ( x _ { k } , u _ { k } ; x _ { g } ) + V _ { f } ( x _ { N _ { h } } ; x _ { g } ) } \end{array}$ subject to $x _ { k + 1 } = F _ { d } ( x _ { k } , u _ { k } )$ $x _ { k } ~ \in ~ \mathcal { X } _ { \mathrm { f r } \epsilon }$ <sub>e</sub>, $u _ { k } \in \mathcal { U }$ . The stage cost is typically chosen as $\ell ( x _ { k } , u _ { k } ; x _ { g } ) \ = \ \| x _ { k } \ -$ $x _ { g } \| _ { Q } ^ { 2 } + \| u _ { k } \| _ { R } ^ { 2 }$ , where $Q \succeq 0$ and $R \succ 0$ . Obstacle avoidance is encoded through nonlinear state constraints defining the collision-free set $\chi _ { \mathrm { f r e e } }$ MPC is accurate because it optimizes over future trajectories, but repeatedly solving the resulting nonlinear program can be computationally expensive.

## Control Barrier Functions

Control Barrier Functions (CBFs) provide a complementary approach for enforcing safety constraints in control systems (Ames et al. 2019). Unlike MPC, the CBF-QP does not optimize an entire future trajectory; instead, it acts as a safety filter that enforces local obstacle-avoidance constraints.

For a nonlinear control-afine system ${ \dot { x } } = f ( x ) + g ( x ) u .$ let the safe set for obstacle i be $\mathcal { C } _ { i } = \{ x : h _ { i } ( x ) \geq 0 \}$ , where $h _ { i } ( x )$ is positive outside the obstacle and negative inside it. Forward invariance of $\mathcal { C } _ { i }$ can be encouraged by enforcing $L _ { f } h _ { i } ( x ) + L _ { g } h _ { i } ( x ) u + \gamma h _ { i } ( x ) \geq 0$ , where $L _ { f } \dot { h _ { i } }$ and $L _ { g } h _ { i }$ are Lie derivatives and $\gamma > 0$ controls how aggressively the controller moves away from the safety boundary.

At each timestep, the CBF controller solves the quadratic program min $_ { \iota } \parallel u - u _ { \mathrm { r e f } } \parallel ^ { 2 }$ subject to $L _ { f } h _ { i } ( x ) + L _ { g } h _ { i } ( x ) u +$ $\gamma h _ { i } ( x ) \geq 0 , \quad \forall i , \qquad u \in \mathcal { U }$ . Here, $u _ { \mathrm { r e f } }$ is a nominal control input, such as a goal-directed command or a command proposed by the neural System-1 policy.

## Thinking Fast and Slow in AI

Kahneman (2011) describe human decision making as the interaction between two complementary processes: a fast, intuitive, experience-driven System-1 and a slower, deliberative, reasoning-based System-2. Recent AI architectures, including SOFAI (Pallagani et al. 2025; Fabiano et al. 2025), adapt this dual-process principle to machine decision-making by combining fast and slow solvers with a metacognitive agent that arbitrates them (Ganapini et al. 2021). These architectures have proven successful in tackling settings closely related to motion planning, such as classical planning and constrained grid navigation, as shown in Fabiano et al. (2025).

Formally, SOFAI defines a decision architecture composed of three components: (i) a set of fast S1 solvers, typically data-driven and experience-based; (ii) a set of slow S2 solvers, based on explicit symbolic reasoning; and (iii) a centralized metacognitive (MC) controller. Incoming problem instances automatically trigger one or more S1 solvers, which produce candidate solutions together with confidence estimates. The MC controller then decides whether to accept the S1 proposal or invoke an S2 solver. This decision is performed in two stages: a lightweight assessment that evaluates whether the expected solution quality satisfies a task-dependent threshold under resource constraints, followed, when necessary, by a cost-benefit comparison between S1 and S2 execution. S2 reasoning is activated only if its expected improvement compensates for the additional computational cost. A key aspect of this framework is that S1 behavior is not static. Through metacognition, solutions produced or validated by S2 can be used to improve the fast solver over time, efectively distilling deliberative reasoning into reactive policies. In this sense, the architecture supports an iterative refinement process in which expensive symbolic reasoning is gradually amortized into eficient inference, enabling the system to adapt to recurring problem distributions while reducing reliance on S2 computation.

## Dual-MP

As mentioned above, Dual-MP solves motion planning through metacognitive arbitration. A planning query is defined as $q = ( f , \mathcal { O } , x _ { 0 } , x _ { g } , \mathcal { X } , \mathcal { U } )$ , where f denotes the nonlinear system dynamics, O is the obstacle map, $x _ { 0 }$ and $x _ { g }$ are the start and goal states, X is the workspace, and $\bar { \mathcal { U } }$ is the admissible control set. In this work, we consider discrete-time nonlinear reach–avoid problems of the form $x _ { t + 1 } = F _ { d } ( x _ { t } , u _ { t } )$ , where $F _ { d }$ is the discretized nonlinear dynamics used by the planner.

In the following, we define the main components of Dual-MP, namely the S1 and S2 solvers, as well as the functionalities required from the MC module. In particular, as detailed in Section , we consider two basic Dual-MP configurations obtained by combining the S1 solver with two S2 solvers.

## Neural S1

Our System-1 is implemented as a neural reactive policy. This follows the common approach of imitation learning to predict low-level controls from local environment observations and goal information (Ichter, Harrison, and Pavone 2017).

Given the current rollout context, local obstacle information, nonlinear dynamics features, and the goal direction, the policy predicts a control input directly: $u _ { t } ^ { \mathrm { S 1 } } \ =$ $\pi _ { \boldsymbol { \theta } } \big ( c _ { t } , s , d _ { t } , g _ { t } \big )$ , where $c _ { t }$ is a fixed-length window of recent states in a local coordinate frame, s encodes the local obstacle situation, $d _ { t }$ encodes the nonlinear dynamics at the current state, and $g _ { t }$ is the local goal vector.

The policy uses a lightweight convolutional architecture. A one-dimensional convolutional encoder processes the recent state context $c _ { t } .$ , while separate multilayer encoders process the situation vector, dynamics features, and goal vector. The resulting features are concatenated and passed through a feedforward control head:

$$
h _ { t } = [ \phi _ { c } ( c _ { t } ) , \phi _ { s } ( s ) , \phi _ { d } ( d _ { t } ) , \phi _ { g } ( g _ { t } ) ] , \qquad u _ { t } ^ { \mathrm { S 1 } } = \pi _ { \theta } ( h _ { t } ) .
$$

This design keeps S1 inexpensive at runtime while still conditioning the control prediction on both geometry and nonlinear dynamics.

As mentioned, S1 is trained by imitation from successful System-2 trajectories. In particular, the bootstrap demonstrations are solved ofline by System-2, and the System-2 recoveries are collected online during continual learning. For each state-control pair, the supervised target is the expert control $\boldsymbol { u } _ { t } ^ { \star }$ , with auxiliary single-step losses that encourage accurate one-step motion, correct direction, and progress toward the goal, and a multi-step rollout term that penalizes the drift accumulated when the policy is run in closed loop:

$$
\begin{array} { r l } & { \mathcal { L } ( \theta ) = \lambda _ { u } \ell _ { u } + \lambda _ { x } \ell _ { x } + \lambda _ { \mathrm { d i r } } \ell _ { \mathrm { d i r } } + \lambda _ { \mathrm { s p d } } \ell _ { \mathrm { s p d } } } \\ & { \qquad + \lambda _ { \mathrm { p r o g } } \ell _ { \mathrm { p r o g } } + \lambda _ { \mathrm { r o l l } } \ell _ { \mathrm { r o l l } } + \lambda _ { \mathrm { s m o } } \ell _ { \mathrm { s m o } } + \lambda _ { \mathrm { o b s } } \ell _ { \mathrm { o b s } } . } \end{array}
$$

Here, $\ell _ { u }$ is a Huber control-imitation loss, $\ell _ { x }$ penalizes one-step prediction error under the nonlinear dynamics, $\ell _ { \mathrm { d i r } }$ encourages alignment with the expert displacement, $\ell _ { \mathrm { s p d } }$ matches step magnitude, and $\ell _ { \mathrm { p r o g } }$ penalizes insuficient progress toward the goal. The remaining three terms are evaluated on a diferentiable H-step rollout of the policy under the nonlinear dynamics, run once every N supervised batches $( H = N = 8 ) \colon \ell _ { \mathrm { r o l l } }$ penalizes deviation of the rolledout positions from the expert positions, $\ell _ { \mathrm { s m o } }$ penalizes large control changes, and $\ell _ { \mathrm { o b s } }$ penalizes proximity to obstacles along the rollout.

At execution time, S1 rolls out the learned policy under the nonlinear dynamics. A lightweight one-step safety filter selects among the predicted control, goal-directed controls, scaled controls, and zero control, keeping only candidates whose next segment is collision-free. The resulting trajectory is then verified for collision freedom and goal reach. In the SOFAI setting, S2 is invoked only when the S1 rollout fails this verification.

## S2 Solvers: CBF and MPC

Dual-MP can be instantiated with two state-of-the-art S2 solvers:

• safe\_control: a control barrier function (CBF)- based safety filter (Kim, Beard, and Panagou 2025)

• acados: a nonlinear model predictive control (MPC) planner (Verschueren et al. 2021)

Let us note that both solvers are implemented using their standard formulations from the literature.

## The Metacognitive Module

Dual-MP follows the SOFAI principle of coordinating fast S1 solvers and slower deliberative S2 solvers through a metacognitive arbitration module (Pallagani et al. 2025). Depending on the Dual-MP configuration, the S1 and S2 components are instantiated diferently; in our experiments, the two configurations are obtained by pairing the S1 solver with two S2 solvers introduced above.

For each new scenario, the metacognitive module first invokes the configured S1 solver. The proposed rollout is accepted only if it passes verification (collision freedom and goal reach - cf. above). If the S1 proposal is rejected, Dual-MP invokes the configured S2 fallback, provided that the available time budget is suficient relative to the estimated problem dificulty. If S2 does not return a satisfactory solution within the budget, Dual-MP falls back to the best available S1 proposal. Successful S2 rollouts are stored in memory and periodically used to improve S1.

The metacognitive module relies on three quantities to properly arbitrate the two systems: problem dificulty, solution correctness, and solver confidence. Dificulty is a scenario-level quantity used to characterize the planning instance; correctness measures the quality of a proposed rollout; and confidence estimates whether an S1 proposal is worth verifying. Following the SOFAI principle, S2 solvers are treated as deliberative solvers and assigned confidence 1, while S1 confidence is computed as described above.

Correctness. Given a rollout ${ \tau } = \{ x _ { t } \} _ { t = 0 } ^ { T } ,$ , correctness is set to 1 if the trajectory reaches the goal and remains collision-free. Otherwise, we assign a soft correctness score based on path length, terminal goal error, and collision penalty: Correct $\begin{array} { r } { ( \tau ; q ) \ = \ \frac { 1 } { 1 + L ( \tau ) + e _ { g } ( \tau ) + \lambda _ { \mathrm { c o l } } N _ { \mathrm { c o l } } ( \tau ) } } \end{array}$ , where $\begin{array} { r } { L ( \tau ) = \sum _ { t = 0 } ^ { T - 1 } \| x _ { t + 1 } - x _ { t } \| _ { 1 } , \qquad e _ { g } ( \tau ) = \| x _ { T } - x _ { g } \| _ { 2 } . } \end{array}$ Here, $N _ { \mathrm { c o l } } ( \tau )$ is a bounded collision penalty and $\lambda _ { \mathrm { c o l } }$ controls the relative cost of unsafe states. In the experiments reported in this paper, we use a strict correctness threshold of 1.0, requiring both goal reaching and collision avoidance. More permissive thresholds are supported by the SOFAI framework and may be useful under strict time constraints.

Dificulty. We define the dificulty of a query scenario using simple features to maintain a lightweight process. Let O denote the set of rectangular obstacles and let W denote the workspace. We first compute the obstacle occupancy $\begin{array} { r } { \eta _ { \mathrm { o c c } } = \overset { \mathbf { \bar { \sum } } _ { O _ { j } \in \mathcal { O } } \mathrm { a r e a } ( O _ { j } ) } { \mathrm { a r e a } ( \mathcal { W } ) + \epsilon } } \end{array}$ . Let $x _ { \mathrm { m i d } } = ( x _ { 0 } + x _ { g } ) / 2$ be the midpoint between start and goal, and let $d _ { \mathrm { c l e a r } } \ =$ min ${ \sf 1 } _ { O _ { j } \in \mathcal { O } } \mathrm { d i s t } ( x _ { \mathrm { m i d } } , O _ { j } )$ be a clearance estimate from this midpoint to the nearest obstacle. The dificulty is then

$$
\begin{array} { c } { { D ( q ) = \eta _ { \mathrm { o c c } } \left( d _ { \mathrm { c l e a r } } + \epsilon \right) ^ { - 1 } \left( 1 + \log ( 1 + T ) \right) } } \\ { { \times \left( 1 + \operatorname* { m i n } \left\{ 1 0 , \left( \epsilon _ { g } + \epsilon \right) ^ { - 1 } \right\} \right) } } \\ { { \times \left( 1 + \operatorname* { m i n } \left\{ 1 0 , \left( \epsilon _ { \mathrm { s a f e } } + \epsilon \right) ^ { - 1 } \right\} \right) , } } \end{array}
$$

where $T$ is the planning horizon, $\epsilon _ { g }$ is the goal tolerance, and $\epsilon _ { \mathrm { s a f e } }$ is the collision margin. The first factor captures geometric clutter and clearance, while the remaining factors account for horizon length, goal precision, and safety strictness.

## SOFAI Continual Learning

A key advantage of SOFAI-inspired architectures is that S1 solvers can improve over time. Dual-MP supports this by storing successfully generated S2 trajectories and using them to enrich the corresponding S1 solver. Specifically, when S2 successfully solves a query, the resulting trajectory is added to the online dataset. For Neural S1, accumulated S2 trajectories are distilled periodically after a fixed number of episodes by fine-tuning the policy on both the original ofline dataset and the newly collected S2 successes.

## Experimental Evaluation

All experiments were run on an Intel Xeon Gold 6248 CPU server with 40 cores and 125 GiB of memory, running Ubuntu 22.04.5 LTS. The source code, benchmark data, and additional details, including ofline training times, will be provided in the appendix for space reasons.

We evaluate Dual-MP on reach–avoid motion-planning tasks in two-dimensional continuous environments. Each instance is defined by nonlinear dynamics $x _ { t + 1 } = F _ { d } ( x _ { t } , u _ { t } )$ a start state $x _ { 0 } , \mathrm { a }$ goal state $x _ { g } ,$ rectangular obstacles O, control bounds $u ,$ and workspace bounds X. A rollout counts as successful only if it is both collision-free and goal-reaching; no partial credit is given. Our experiments address four questions:

• Q1: Can Dual-MP improve state-of-the-art planning?

• Q2: Does Dual-MP preserve reliability?

• Q3: How does continual learning afect S1?

• Q4 Does warm-starting System-2 from the rejected S1 trajectory help System-2 solving?

We evaluate our approach on six benchmark families, each stressing a diferent aspect of reach–avoid planning:

• Large sparse (LS): a few large obstacles; tests longhorizon obstacle avoidance in an open workspace.

• Dense clutter (DC): many small rectangles; tests local collision avoidance under frequent re-planning pressure.

• Serial walls (SW): several wall-with-gap structures in sequence; tests repeated constrained passages, where a single missed gap is unrecoverable.

• Maze branching (MB): intersecting wall structures with multiple candidate routes; tests global route commitment.

• Long slalom (LSM): a long corridor with alternating ofset barriers; tests sustained tracking over a horizon far longer than the MPC prediction window.

• Bugtrap (BT): concave trap layouts; tests robustness to locally attractive but globally poor routes.

Instances within a family share the workspace scale, the start/goal pair, and the reach–avoid objective, and vary in obstacle layout and in the parameters of the nonlinear drift field. For every family, we generate three disjoint instance sets from separate seeds: a training set of 100 instances used only to bootstrap S1, an evaluation set of 500 instances used for the results in Tables 2 and 3, and a held-out probe set of 500 instances used exclusively to measure the efect of continual learning. The probe set is never trained on and is identical across all blocks and all methods, so probe curves are directly comparable over time.

All of the generated instances are run on six diferent solvers/Dual-MP configurations, listed in Table 1. The three

Table 1: Planners configuration.
<table><tr><td>Name</td><td>System-1</td><td>System-2</td></tr><tr><td>NN</td><td>Neural policy</td><td></td></tr><tr><td>CBF</td><td></td><td>CBF</td></tr><tr><td>MPC</td><td></td><td>MPC</td></tr><tr><td>DMP-CBF</td><td>Neural policy</td><td>CBF</td></tr><tr><td>DMP-MPC</td><td>Neural policy</td><td>MPC</td></tr><tr><td>DMP-Warm</td><td>Neural policy</td><td>MPC (S1 warm start)</td></tr></table>

Dual-MP variants difer only in the System-2 they escalate to and in whether that solver is warm-started from the rejected S1 trajectory.

The various approaches are then evaluated on three main metrics: (i) Success rate, which is the fraction of instances whose returned rollout is collision-free and reaches the goal. (ii) Runtime which represents the time to find a solution: for Dual-MP it is the sum of the S1 call and, when escalation occurs, the S2 call, so escalated instances are charged for both. We report the mean and the 90th percentile. (iii) Trajectory Quality evaluated only on successful rollouts, so that it never trades of against the success rate. We use a duration-invariant index that is the geometric mean of three sub-scores, each in (0, 1] and each normalised against a scenario-defined rather than a solver-defined reference: $\begin{array} { r l r } { Q } & { = } & { \left( \eta _ { \mathrm { p a t h } } \cdot \eta _ { \mathrm { s m o o t h } } \cdot \eta _ { \mathrm { c l e a r } } \right) ^ { 1 / 3 } . \eta _ { \mathrm { p a t h } } = L _ { \mathrm { r e f } } / L } \end{array}$ is the standard path-optimality ratio of the executed length L against the shortest collision-free path $L _ { \mathrm { r e f } } ,$ obtained by grid search around the obstacles (Sucan, Moll, and Kavraki 2012). $\eta _ { \mathrm { s m o o t h } } = \mathrm { S P A R C } _ { \mathrm { m i n - j e r k } } / \mathrm { S P A R C }$ is the spectral arc length of the speed profile expressed relative to an ideal minimum– jerk movement (Flash and Hogan 1985), so a minimum-jerk trajectory scores 1.0; spectral arc length is used because it is the only smoothness measure that is simultaneously valid and reliable (Balasubramanian, Melendez-Calderon, and Burdet 2012). $\eta _ { \mathrm { c l e a r } } = \operatorname* { m i n } ( d _ { \mathrm { m i n } } / r _ { \mathrm { b o d y } } , 1 )$ scores the worst obstacle clearance against one body radius, capped so that excess conservatism earns nothing. The geometric mean prevents a near-collision or a chattering control from being compensated elsewhere. Two properties matter for the comparisons below. First, Q is invariant to how fast the trajectory is traversed, so an aggressive controller cannot buy quality by saturating its actuator. Second, because the reference is the scenario’s own shortest path, Q is comparable across families of very diferent dificulty.

For Dual-MP we additionally report how many successes were produced by S1 alone versus by the S2 fallback.

## Training Protocol

Bootstrap. For each family and each System-2 solver, we solve the 100 training instances with that solver and retain only the collision-free, goal-reaching trajectories. The base S1 policy is trained on those demonstrations by behaviour cloning augmented with a short diferentiable rollout loss (horizon 8, applied every 8 supervised batches). Crucially, the bootstrap teacher is matched to the System-2 the arm will escalate to: DMP-CBF starts from a CBF-taught policy and

DMP-MPC from an MPC-taught one.

Continual learning. The 500 evaluation instances are shufled once (fixed seed) and presented in five sequential blocks of 100. After each block, S1 is retrained, warm-started from the previous block’s weights, on the union of the bootstrap demonstrations and all System-2 recoveries collected so far. Recoveries are obtained in DAgger fashion (Ross, Gordon, and Bagnell 2010): on instances where S1 was rejected, we relabel states visited by the S1 rollout with the System-2 solver, so the policy receives corrective actions on its own state distribution rather than only on the expert’s. A replay fraction of 0.6 keeps the fixed bootstrap demonstrations in the mixture and limits drift toward the increasingly hard, failure-biased recovery set. The retrained model is evaluated on the frozen probe set before the next block begins.

## Results

We now present the experimental results of our paper and use these to answer the four research questions we introduced above. The main results are presented in Table 2, which reports results macro-averaged over the six families for the six configurations, and in Table 3 that breaks the same runs down per family.

(Q1) Can Dual-MP improve state-of-the-art planning? Yes. Dual-MP improves the MPC baseline by filtering out easy instances with S1 and invoking MPC only when needed, which reduces the number of expensive optimisation calls while improving overall success. This yields the strongest success-runtime trade-ofamong the compared methods. The largest gains occur in large sparse and dense clutter environments, where S1 frequently solves instances without escalation. Gains are limited in serial walls, maze branching, and long slalom, where most instances still require MPC. By contrast, DMP-CBF is slower than CBF alone because the cost of an S1 rollout outweighs the savings from avoiding an already inexpensive fallback. Thus, the value of dual-process arbitration is determined primarily by the runtime gap between S1 and S2, rather than by S1 accuracy alone.

(Q2) Does Dual-MP preserve reliability? Yes. Across the completed families, both Dual-MP variants match or improve the success rate of their respective S2 fallback (Tables 2 and 3). This follows from the verification gate: S1 is accepted only when its rollout is collision-free and reaches the goal, while rejected rollouts are delegated to S2. S1 can also solve some instances missed by the corresponding fallback solver, particularly in constrained environments.

The main trade-of is trajectory quality. MPC produces the highest-quality trajectories because it explicitly optimises path eficiency and obstacle clearance, whereas accepted S1 rollouts prioritise fast, feasible completion. Consequently, Dual-MP may return trajectories that are longer or pass closer to obstacles than solutions produced by MPC alone. Dual-MP is therefore most suitable when reducing online planning time is more important than retaining the trajectory quality of every MPC solution.

(Q3) How does continual learning afect S1? Table 4 reports the cumulative changes in S1 success rate and trajectory quality, while Figures 1 and 2 show their evolution across training blocks.

Table 2: Aggregate results for all methods averaged over the six environment families (500 instances per family).
<table><tr><td>Method</td><td>Succ. (%)</td><td>Mean RT (ms)</td><td>P90 RT (ms)</td><td>Mean Q</td><td>P90 Q</td><td>S1/S2 Solves</td></tr><tr><td>NN</td><td>36.4</td><td>230</td><td>305</td><td>0.601</td><td>0.817</td><td>182 / 0</td></tr><tr><td>CBF</td><td>82.0</td><td>657</td><td>1448</td><td>0.795</td><td>0.844</td><td>0/475</td></tr><tr><td>MPC</td><td>84.8</td><td>5708</td><td>7092</td><td>0.844</td><td>0.875</td><td>0/496</td></tr><tr><td>DMP-CBF</td><td>84.1</td><td>983</td><td>2279</td><td>0.734</td><td>0.850</td><td>204 / 217</td></tr><tr><td>DMP-MPC</td><td>85.1</td><td>4979</td><td>8086</td><td>0.780</td><td>0.877</td><td>141 / 285</td></tr><tr><td>DMP-Warm</td><td>83.6</td><td>5008</td><td>8161</td><td>0.782</td><td>0.877</td><td>140 / 278</td></tr></table>

Table 3: Per-family results on instance sets (n = 500). LS: large sparse, DC: dense clutter, SW: serial walls, MB: maze branching, LSM: long slalom, BT: bugtrap.
<table><tr><td>Method</td><td>LS</td><td>DC</td><td>SW</td><td>MB</td><td>LSM</td><td>BT</td></tr><tr><td colspan="7">Success rate (%)</td></tr><tr><td>NN</td><td>82.8</td><td>74.4</td><td>12.0</td><td>19.2</td><td>13.0</td><td>17.0</td></tr><tr><td>CBF</td><td>96.0</td><td>94.0</td><td>59.7</td><td>65.0</td><td>98.2</td><td>78.8</td></tr><tr><td>MPC</td><td>98.2</td><td>94.0</td><td>77.5</td><td>69.5</td><td>76.2</td><td>93.2</td></tr><tr><td>DMP-CBF</td><td>97.0</td><td>94.8</td><td>65.6</td><td>70.0</td><td>98.2</td><td>79.2</td></tr><tr><td>DMP-MPC</td><td>98.2</td><td>95.2</td><td>77.8</td><td>69.6</td><td>76.6</td><td>93.4</td></tr><tr><td>DMP-Warm</td><td>98.2</td><td>95.2</td><td>77.2</td><td>67.8</td><td>74.6</td><td>88.8</td></tr><tr><td colspan="7">Mean planning runtime (ms)</td></tr><tr><td>NN</td><td>112</td><td>135</td><td>230</td><td>255</td><td>460</td><td>185</td></tr><tr><td>CBF</td><td>121</td><td>234</td><td>934</td><td>961</td><td>1258</td><td>437</td></tr><tr><td>MPC</td><td>3439</td><td>3688</td><td>4886</td><td>6140</td><td>12316</td><td>3781</td></tr><tr><td>DMP-CBF</td><td>349</td><td>449</td><td>1131</td><td>1212</td><td>1974</td><td>783</td></tr><tr><td>DMP-MPC</td><td>1473</td><td>1670</td><td>4866</td><td>5988</td><td>12490</td><td>3385</td></tr><tr><td>DMP-Warm</td><td>1440</td><td>1715</td><td>4909</td><td>6048</td><td>12458</td><td>3476</td></tr><tr><td colspan="7">Mean trajectory quality Q</td></tr><tr><td>NN</td><td>0.781</td><td>0.750</td><td>0.580</td><td>0.484</td><td>0.425</td><td>0.586</td></tr><tr><td>CBF</td><td>0.817</td><td>0.817</td><td>0.778</td><td>0.772</td><td>0.801</td><td>0.787</td></tr><tr><td>MPC</td><td>0.860</td><td>0.861</td><td>0.836</td><td>0.826</td><td>0.827</td><td>0.853</td></tr><tr><td>DMP-CBF</td><td>0.794</td><td>0.763</td><td>0.696</td><td>0.679</td><td>0.741</td><td>0.729</td></tr><tr><td>DMP-MPC</td><td>0.761</td><td>0.716</td><td>0.793</td><td>0.802</td><td>0.817</td><td>0.792</td></tr><tr><td>DMP-Warm</td><td>0.762</td><td>0.709</td><td>0.814</td><td>0.790</td><td>0.817</td><td>0.796</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Continual learning improves S1, but the type of improvement depends on the S2 teacher. The CBF-taught policy achieves substantial gains in success rate as fallback experience accumulates, increasing the fraction of instances solved directly by S1 and reducing reliance on CBF. The MPCtaught policy primarily improves trajectory quality and increases success in relatively open environments, such as large sparse and dense clutter, but provides limited or inconsistent success gains in more constrained families.

This diference is in the “teacher” capabilities: MPC demonstrations often contain long-horizon, globally routedependent plans that a reactive policy with local observations cannot reliably reproduce after small closed-loop deviations. CBF, by contrast, provides short, local corrective actions that are easier for S1 to imitate.

Table 4: Continual learning on the held-out (500)-instance probe set. “Base” is the bootstrap policy before any online experience. B0–B4 show cumulative performance after each (100)-instance block. ∆ is the change from Base to B4, and ∆Q is the corresponding change in mean S1 trajectory quality.
<table><tr><td>Family</td><td>Arm</td><td>Base</td><td>B0</td><td>B1</td><td>B2</td><td>B3</td><td>B4</td><td>Δ</td><td>∆Q</td></tr><tr><td>LS</td><td>CBF</td><td>80.4</td><td>80.8</td><td>86.0</td><td>85.0</td><td>90.0</td><td>90.0</td><td>+9.6</td><td>+0.007</td></tr><tr><td></td><td>MPC</td><td>54.2</td><td>64.8</td><td>71.8</td><td>73.8</td><td>75.2</td><td>72.4</td><td>+18.2</td><td>+0.113</td></tr><tr><td>DC</td><td>CBF</td><td>72.8</td><td>74.6</td><td>82.0</td><td>84.2</td><td>85.4</td><td>85.2</td><td>+12.4</td><td>-0.008</td></tr><tr><td></td><td>MPC</td><td>58.2</td><td>65.6</td><td>64.6</td><td>66.8</td><td>68.8</td><td>70.8</td><td>+12.6</td><td>+0.104</td></tr><tr><td>SW</td><td>CBF</td><td>11.2</td><td>15.8</td><td>20.8</td><td>24.6</td><td>27.6</td><td>30.4</td><td>+19.2</td><td>-0.028</td></tr><tr><td></td><td>MPC</td><td>12.6</td><td>9.2</td><td>5.2</td><td>5.0</td><td>6.2</td><td>4.4</td><td>-8.2</td><td>+0.044</td></tr><tr><td>MB</td><td>CBF</td><td>21.2</td><td>19.4</td><td>24.0</td><td>26.8</td><td>23.8</td><td>24.2</td><td>+3.0</td><td>+0.062</td></tr><tr><td></td><td>MPC</td><td>7.0</td><td>6.2</td><td>4.0</td><td>5.2</td><td>5.6</td><td>4.2</td><td>-2.8</td><td>+0.061</td></tr><tr><td>LSM</td><td>CBF</td><td>13.0</td><td>16.8</td><td>16.6</td><td>18.8</td><td>19.0</td><td>24.8</td><td>+11.8</td><td>+0.006</td></tr><tr><td>BT</td><td>MPC</td><td>0.6</td><td>0.8</td><td>0.6</td><td>2.0</td><td>2.0</td><td>2.6</td><td>+2.0</td><td>+0.041</td></tr><tr><td></td><td>CBF</td><td>19.2</td><td>23.0</td><td>28.8</td><td>30.6</td><td>36.4</td><td>42.4</td><td>+23.2</td><td>-0.020</td></tr><tr><td></td><td>MPC</td><td>22.2</td><td>19.4</td><td>23.6</td><td>25.0</td><td>24.6</td><td>27.2</td><td>+5.0</td><td>+0.085</td></tr><tr><td>Mean</td><td>CBF</td><td>36.3</td><td>38.4</td><td>43.0</td><td>45.0</td><td>47.0</td><td>49.5</td><td>+13.2</td><td>+0.003</td></tr><tr><td></td><td>MPC</td><td>25.8</td><td>27.7</td><td>28.3</td><td>29.6</td><td>30.4</td><td>30.3</td><td>+4.5</td><td>+0.075</td></tr></table>

Table 5: Warm-start ablation on instances escalated by both MPC arms. Times are mean System-2 solver time; “S2 solves” counts successful fallbacks.
<table><tr><td>Family</td><td>n</td><td>Cold (s)</td><td>Warm (s)</td><td>Δ</td><td>S2 solves</td></tr><tr><td>LS</td><td>137</td><td>3.489</td><td>3.492</td><td>-0.003</td><td>128 / 128</td></tr><tr><td>DC</td><td>150</td><td>3.764</td><td>3.771</td><td>-0.007</td><td>126 / 126</td></tr><tr><td>SW</td><td>449</td><td>4.813</td><td>4.803</td><td>+0.010</td><td>339 /337</td></tr><tr><td>MB</td><td>461</td><td>5.935</td><td>6.067</td><td>-0.133</td><td>310 / 300</td></tr><tr><td>LSM</td><td>477</td><td>11.096</td><td>11.057</td><td>+0.040</td><td>370 /362</td></tr><tr><td>BT</td><td>364</td><td>3.870</td><td>4.001</td><td>-0.131</td><td>331 / 311</td></tr></table>

(Q4) Does warm-starting System-2 from the rejected S1 trajectory help System-2 solving? Naive warm-starting from a rejected S1 rollout is not useful as shown in Table 5. Initialising MPC with the rejected trajectory does not give a consistent runtime benefit, and it can even steer the optimiser toward the same locally poor region that caused S1 to fail. Although warm-starting occasionally helps, the efect is unreliable and can reduce fallback success on the hardest instances. We therefore do not use rejected S1 rollouts as unconditional warm starts. While Dual-MP does compute an S1 confidence score, we did not tune it for warm-start selection because it was not a suficiently reliable proxy for feasibility, so we omitted that mechanism. We leave this as future work.

![](images/3b19ddf1d399a1d361145b923f34adc5a39014cda8276333b4da3c0dcf40b107.jpg)  
Figure 1: Success rate of S1 on the 500-instance probe set as continual learning accumulates experience. Solid curves show the CBF- and MPC-taught policies.

![](images/0cdd38e2d1777326c28c4ae6f654eaa3918c88980ed0eb287d64068fde83a9df.jpg)  
Figure 2: Mean quality Q of successful S1 rollouts on the 500-instance probe set as continual learning accumulates experience. Quality is evaluated only on successful rollouts.

## Discussion

Although Dual-MP is only an initial step toward dual-process motion planning, it delivers promising results; it outperforms the standalone baselines in the reported success and eficiency metrics. This suggests that verified arbitration is not just a conceptual framework, but a practical way to combine learned policies with optimization-based solvers.

The results show that Dual-MP is most efective when the two systems have distinct costs and complementary strengths. A fast S1 policy is most valuable when it avoids expensive MPC calls, but it ofers little benefit when the fallback is already cheap, as with CBF. This means that success rate alone is not suficient to evaluate a hybrid planner: the runtime balance between S1 and S2 largely determines whether arbitration is worthwhile. Exact rollout verification is central to this behavior. It allows S1 to produce low-latency feasible solutions without letting unsafe or incomplete predictions replace the fallback solver. When S1 fails, S2 takes over, and successful recoveries can then be reused for continual learning. This makes the architecture practical without requiring the neural policy to be perfectly safe in isolation, as long as it is paired with a correct fallback and a verification gate.

Similarly, we note that continual learning is most efective when the teacher’s behavior is compatible with the student’s policy class. CBF provides short, local corrections that improve S1 coverage, whereas MPC often produces longer, route-dependent behaviors that are harder for a reactive policy to reproduce reliably in closed loop. This suggests several promising directions for future work: (i) training S1 on MPC trajectories and pairing it with CBF, (ii) exploring other combinations of fast and slow components, and (iii) using a portfolio metacontroller to select among multiple S2 planners rather than relying on a single fallback. More generally, future work could optimize diferent objectives depending on the application, including planning time, trajectory quality, success rate, or even accepting slightly suboptimal solutions when that yields a better overall trade-of. In the same spirit, better knowledge transfer across environment families could reduce bootstrap cost, either by sharing an S1 policy across tasks or by making continual learning more selective when ofline retraining budget is limited. Rejected S1 rollouts should likewise be used more selectively as MPC initializations; a stronger confidence or feasibility estimate could help determine when a proposal is close enough to serve as a useful warm start and when MPC should instead solve the problem from scratch.

Overall, this work provides an initial but promising step toward optimizing not only the individual tools, but also their composition. It shows that the interaction between learned policies, symbolic solvers, verification, and retraining can itself be designed and improved.

## Conclusion

We presented Dual-MP, a dual-process motion-planning architecture that combines a fast neural System-1 policy with a symbolic System-2 fallback. Dual-MP accepts S1 rollouts only when they are collision-free and goal-reaching; otherwise, it delegates planning to either MPC or CBF. Successful S2 recoveries are then reused to improve S1 through continual learning.

Dual-MP with MPC strikes the best balance between success rate and runtime: it achieves the highest success rate while remaining substantially faster than standalone MPC. DMP-CBF is also improved in success rate, but it is slightly slower than CBF because the added S1 rollout overhead outweighs part of the savings. This supports the case for dualprocess as it can deliver measurable gains in planning eficiency and performance. At the same time, the results show that continual learning can reduce dependence on System-2. Conversely, naive warm-starting of MPC from rejected S1 trajectories provides no consistent benefit. Overall, Dual-MP ofers a promising demonstration that fast-and-slow architecture is a practical mechanism for combining the speed of learned policies with the robustness of optimization-based planning.

## References

Ames, A. D.; Coogan, S.; Egerstedt, M.; Notomista, G.; Sreenath, K.; and Tabuada, P. 2019. Control Barrier Func-

tions: Theory and Applications. In 2019 18th European Control Conference (ECC), 3420–3431.

Ames, A. D.; Xu, X.; Grizzle, J. W.; and Tabuada, P. 2017. Control Barrier Function Based Quadratic Programs for Safety Critical Systems. IEEE Transactions on Automatic Control, 62(8): 3861–3876.

Balasubramanian, S.; Melendez-Calderon, A.; and Burdet, E. 2012. A Robust and Sensitive Metric for Quantifying Movement Smoothness. IEEE Transactions on Biomedical Engineering, 59(8): 2126–2136.

Bjelonic, M.; Grandia, R.; Geilinger, M.; Harley, O.; Medeiros, V.; Pajovic, V.; Jelavic, E.; Coros, S.; and Hutter, M. 2022. Ofline motion libraries and online MPC for advanced mobility skills. The International Journal of Robotics Research, 41: 903–924.

Carvalho, J.; Le, A. T.; Baierl, M.; Koert, D.; and Peters, J. 2024. Motion Planning Difusion: Learning and Planning of Robot Motions with Difusion Models. arXiv:2308.01557.

Fabiano, F.; Ganapini, M. B.; Loreggia, A.; Mattei, N.; Murugesan, K.; Pallagani, V.; Rossi, F.; Srivastava, B.; and Venable, K. B. 2025. Thinking Fast and Slow in Human and Machine Intelligence. Commun. ACM, 68(8): 72–79.

Fishman, A.; Murali, A.; Eppner, C.; Peele, B.; Boots, B.; and Fox, D. 2023. Motion Policy Networks. In Liu, K.; Kulic, D.; and Ichnowski, J., eds., Proceedings of The 6th Conference on Robot Learning, volume 205 of Proceedings ofMachine Learning Research, 967–977. PMLR.

Flash, T.; and Hogan, N. 1985. The coordination of arm movements: an experimentally confirmed mathematical model. In Journal ofNeuroscience.

Fridovich-Keil, D.; Herbert, S. L.; Fisac, J. F.; Deglurkar, S.; and Tomlin, C. J. 2018. Planning, Fast and Slow: A Framework for Adaptive Real-Time Safe Trajectory Planning. arXiv:1710.04731.

Ganapini, M. B.; Campbell, M.; Fabiano, F.; Horesh, L.; Lenchner, J.; Loreggia, A.; Mattei, N.; Rossi, F.; Srivastava, B.; and Venable, K. B. 2021. Thinking Fast and Slow in AI: the Role of Metacognition. arXiv:2110.01834.

Hegde, R.; and Panagou, D. 2016. Multi-agent motion planning and coordination in polygonal environments using vector fields and model predictive control. In 2016 European Control Conference (ECC), 1856–1861.

Ichter, B.; Harrison, J.; and Pavone, M. 2017. Learning Sampling Distributions for Robot Motion Planning. CoRR, abs/1709.05448.

Johnson, J. J.; Kalra, U. S.; Bhatia, A.; Li, L.; Qureshi, A. H.; and Yip, M. C. 2022. Motion Planning Transformers: A Motion Planning Framework for Mobile Robots. arXiv:2106.02791.

Kahneman, D. 2011. Thinking, Fast and Slow. Farrar, Straus and Giroux.

Karaman, S.; and Frazzoli, E. 2011. Sampling-based Algorithms for Optimal Motion Planning. CoRR, abs/1105.1186.

Kim, T.; Beard, R. W.; and Panagou, D. 2025. How to Adapt Control Barrier Functions? A Learning-Based Approach with Applications to a VTOL Quadplane. In IEEE Conference on Decision and Control (CDC).

LaValle, S. M. 2006. Planning Algorithms. Cambridge University Press. ISBN 9780511546877.

Mansard, N.; DelPrete, A.; Geisert, M.; Tonneau, S.; and Stasse, O. 2018. Using a Memory of Motion to Eficiently Warm-Start a Nonlinear Predictive Controller. In 2018 IEEE International Conference on Robotics and Automation (ICRA), 2986–2993.

Mayne, D.; Rawlings, J.; Rao, C.; and Scokaert, P. 2000. Constrained model predictive control: Stability and optimality. Automatica, 36(6): 789–814.

Paden, B.; Cáp, M.; Yong, S. Z.; Yershov, D. S.; and Frazzoli, E. 2016. A Survey of Motion Planning and Control Techniques for Self-Driving Urban Vehicles. IEEE Trans. Intell. Veh., 1(1): 33–55.

Pallagani, V.; Loreggia, A.; Fabiano, F.; Srivastava, B.; Rossi, F.; and Horesh, L. 2025. SOFAI Lab: A Hands-On Guide to Building Neurosymbolic Systems with Metacognitive Control. In AAAI Conference on Artificial Intelligence.

Ross, S.; Gordon, G. J.; and Bagnell, J. A. 2010. No-Regret Reductions for Imitation Learning and Structured Prediction. CoRR, abs/1011.0686.

Sucan, I. A.; Moll, M.; and Kavraki, L. E. 2012. The Open Motion Planning Library. IEEE Robotics and Automation Magazine, 19(4): 72–82.

Verschueren, R.; Frison, G.; Kouzoupis, D.; Frey, J.; van Duijkeren, N.; Zanelli, A.; Novoselnik, B.; Albin, T.; Quirynen, R.; and Diehl, M. 2021. acados – a modular open-source framework for fast embedded optimal control. Mathematical Programming Computation.

Wang, J.; Zhang, T.; Ma, N.; Li, Z.; Ma, H.; Meng, F.; and Meng, M. Q.-H. 2021. A survey of learning-based robot motion planning. IET Cyber-Systems and Robotics, 3(4): 302–314.

## Computational Resources and Experimental Configuration

## Computational resources

All experiments were run on an Intel Xeon Gold 6248 CPU server with 40 cores and 125 GiB of memory, running Ubuntu 22.04.5 LTS.

Overall, generating the six nonlinear benchmark suites, training the System 1 policies, collecting DAgger recovery demonstrations, and evaluating all System 1/System 2 configurations required several CPU-hours of computation across the experimental batches. A detailed training-time breakdown is provided in Table 6.

<table><tr><td>Environment Family</td><td>Baseline S1 Training</td><td>DAgger Collection</td><td>CL S1 Training</td></tr><tr><td>Large sparse (LS)</td><td>~6s</td><td>11 s (8–14)</td><td>~66 s</td></tr><tr><td>Dense clutter (DC)</td><td>~84 s</td><td>12 s (11–13)</td><td>~96 s</td></tr><tr><td>Serial walls (SW)</td><td>~18 s</td><td>32 s (28–35)</td><td>~24 s</td></tr><tr><td>Maze branching (MB)</td><td>~27 s</td><td>39 s (37–43)</td><td>~18 s</td></tr><tr><td>Long slalom (LSM)</td><td>~90 s</td><td>80 s (75–87)</td><td>~102 s</td></tr><tr><td>Bugtrap (BT)</td><td>~2 s</td><td>23 s (21–24)</td><td>~36 s</td></tr></table>

Table 6: Approximate training and dagger collecting time per environment family.

## Parameters and Hyperparameters

Tables 7 and 8 summarize the model parameters and the training, benchmark, and reproducibility hyperparameters used in our experiments.

<table><tr><td>Parameter Description</td><td>Value</td></tr><tr><td>Channel dimensions of the three-layer Conv1d context encoder.</td><td> $d _ { \mathrm { c t x } } \to 6 4 \to 1 2 8 \to 1 2 8$ </td></tr><tr><td>Kernel size of the Conv1d context encoder.</td><td>3</td></tr><tr><td>Pooling operation applied to the encoded context.</td><td>Adaptive average pooling</td></tr><tr><td>Widths of the situation and dynamics encoders, respectively.</td><td>128 / 128</td></tr><tr><td>Width of the goal encoder.</td><td>64</td></tr><tr><td>Dimensions of the multilayer perceptron prediction head.</td><td>320→256→256→2</td></tr><tr><td>Hidden-layer width of the neural System 1 policy.</td><td>256</td></tr><tr><td>Number of channels in the final convolutional layer.</td><td>128</td></tr><tr><td>Dropout probability used throughout the policy</td><td>0.05</td></tr></table>

Table 7: Values and descriptions of the neural System 1 model parameters.

<table><tr><td>Hyperparameter Description</td><td>Value</td></tr><tr><td>Optimizer used to train the neural System 1 policy.</td><td>AdamW</td></tr><tr><td>Learning rate and weight-decay coefficient.</td><td> $1 0 ^ { - 4 } / 1 0 ^ { - 5 }$ </td></tr><tr><td>Training batch size and validation-data fraction.</td><td>64 / 0.10</td></tr><tr><td>Maximum gradient norm used for gradient clipping.</td><td>5.0</td></tr><tr><td>Supervised training objective and its transition parameter.</td><td>Huber loss, δ = 1.0</td></tr><tr><td>Number of continual-learning training epochs.</td><td>12</td></tr><tr><td>Number of baseline training epochs.</td><td>35</td></tr><tr><td>Random seeds used for training, evaluation, and probing.</td><td>7 / 8 /700</td></tr><tr><td>Continual-learning block size and scenario-ordering seed.</td><td>100 / 42</td></tr><tr><td>Numbers of evaluation and DAgger workers.</td><td>16 / 16</td></tr><tr><td>System 1 integration step and nominal rollout horizon.</td><td>0.075 s / 900 steps</td></tr><tr><td>Situation-grid resolution and corridor buffer.</td><td> $\mathrm { 2 5 \times 2 5 / 2 c e l l s }$ </td></tr><tr><td>Differentiable rollout horizon and application frequency.</td><td>8 steps / every 8 batches</td></tr><tr><td>Safety-filter mode used during runtime and differentiable rollouts.</td><td>Policy</td></tr><tr><td>Replay fraction allocated to bootstrap demonstrations.</td><td>0.60</td></tr><tr><td>Number of DAgger states sampled from each scenario.</td><td>4</td></tr><tr><td>Relative weights assigned to bootstrap and DAgger samples.</td><td></td></tr><tr><td>Planning timeout for each scenario.</td><td>1.0 / 1.0 60 s</td></tr></table>

Table 8: Key training, benchmark, and reproducibility hyperparameters.

## Visualising Trajectory Quality

The trajectory quality is described by a score Q that takes the geometric mean of path eficiency, smoothness, and obstacleclearance score. A higher score corresponds to a shorter, smoother, and/or better-cleared trajectory. Figure 3 shows the neura System 1 (NN), System 2 CBF, and System 2 MPC trajectories and their corresponding scores on the same scenario.

![](images/a38ac4c1c564910a87d6503f6644e60035b4d437c6a431d40f6c4ae31648d599.jpg)  
(a) S1-NN. Q = 0.592.

![](images/335e4093acc7a0c8745cae4ee711aed19b8ebdeeeb3977e6fdc8a73f14dc22b0.jpg)  
(b) S2-CBF. Q = 0.802.

![](images/e45b6dadb243dcca49cf27c2ae22269880ae2b0c518fe84a5926f5e49bebf80b.jpg)  
(c) S2-MPC. Q = 0.870.  
Figure 3: Trajectory-quality comparison on Bugtrap scenario 6. Higher Q indicates a more eficient, smoother, and better-cleared trajectory.

## Visualising the Benchmark Families

Figure 4 provides a visual index of the six nonlinear two-dimensional obstacle-avoidance families and the trajectory computed by System 2 MPC. Large Sparse and Dense Clutter vary obstacle density; Serial Walls and Maze Branching introduce structured barriers and route choices; Long Slalom creates a long alternating corridor; and Bugtrap places the robot in a concave trapping geometry.

![](images/3276f4af34e72871e0d5a693961f787170a80aad90531f48606f556050a6eade.jpg)  
(a) Large Sparse (LS).

![](images/016124c5b1a016035fe597af8ba50648e29a53ec2026b3b87e991d3e83160364.jpg)  
(b) Dense Clutter (DC).

![](images/f7f068f2d821f771ab70a43131cae3ac0ea60788a7339eda3b7698bd31593124.jpg)  
(c) Serial Walls (SW).

![](images/c02e8fa9084a06c67f2199a617cba0242ac8234044058290489ca6cd94193f47.jpg)  
(d) Maze Branching (MB).

![](images/f025afe2751daa5e28084037738821fe977e5616caf443416744eb1681d94d0a.jpg)  
(e) Long Slalom (LSM).

![](images/8e2a7ca50bb03861c789366415a1c950d14b577f86326dd8f1c62914acf4d1e1.jpg)  
(f) Bugtrap (BT).

Figure 4: Representative nonlinear motion-planning environments used in the evaluation. Each panel shows an example map from one benchmark family.