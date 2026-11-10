Workplace

# HolidayFactory 1.0

# Simulating, Generating, and Evaluating Industrial Robot Tasks

Holiday Robotics<sup>†</sup>

## Abstract

Robots are typically evaluated in episodes that begin from a reset, while industrial work consists of repeated jobs whose outcomes affect subsequent operations. Evaluating such work at scale requires simulation of task-critical physical interactions, construction of industrial environments, and measurements of performance as scene state persists. We present HOLIDAYFACTORY 1.0, a framework for simulating, generating, and evaluating industrial robot work. Given a human-specified workplace and task, the framework defines a world pairing a scenario of scene variants and process logic with an evaluation defining a job, metrics, and a rubric. HOLIDAYFACTORY 1.0 comprises three components. HOLIDAY-SIM extends Newton’s SDF-based hydroelastic contact with additional constitutive laws and a semi-implicit primal-dual solver, couples stateful actuator dynamics to articulated motion, and derives tactile and joint observations from simulated states. Agentic world construction generates simulation-ready assets and scene variants using geometric and interaction checks, and proposes task-conditioned evaluations for human confirmation and executable checks. 24/7-BENCH introduces seven industrial tasks in which jobs are evaluated without resetting the scene, measuring completed, failed, and unfinished jobs, throughput, latency, and motor energy. Controlled studies characterize the proposed contact, actuator, and tactile models and examine the effectiveness of verification-guided world construction. Experiments on scene and evaluation generation further reveal the limitations of model critics and the difficulty of ensuring that generated evaluations faithfully capture task requirements. We examine job-level evaluations with scripted trajectories (e.g., code-as-policy) but have not yet compared policies over persistent runs.

![](images/bb2db7672990414a3d0eec625db56814f93fdd9878bf2e836fdfc32e87c992a4.jpg)

![](images/93298e922280af0d414800ec2a7e6bbd1747555fd01181e07f2836cc7533a895.jpg)

![](images/14078c2d09443b216ce6c226c02c4c6d982084f8d23a0531b6ac57c10e2bbafd.jpg)

Repeated runs  
![](images/4ebd3fbbdd09b458b80127ffa6373177a373f37a179a0c1d07be1f62b0208665.jpg)

![](images/d9f7edafe11f0aaf1bdc924e515685562f7ca3a0a3b263155db1f3425114375b.jpg)  
Conveyor loading

![](images/d4797c5418f21eb67952772e22979d10b1736c503fd24978c07cf28e85162f30.jpg)  
Flow-rack loading

![](images/fc221471d681fc2087ea9c0630465011f579376b7b7f3d632ad43d816bad5aa2.jpg)  
Cart push

![](images/5dbbbcf7c0f0c5ba53cab0baa387c04c67837da87975b0bb4dff929d1624a8c1.jpg)  
Jig loading

![](images/53ab2a73e2901ceeb3bcaaf62400dd9df53203cabddd0d589eaf9c85b6be9d44.jpg)  
Press tending

![](images/5f56a8814e79e25fe6402cb545dc0460d6cdaeb10adbf7d94e90eea9a59ecfa4.jpg)  
Valve operation

![](images/2bef9bcb3e4b6023ca255a83d97e8c9fb755d1c41527aa09671416f1cdf78418.jpg)  
Barcode routing

Figure 1. An overview of HOLIDAYFACTORY 1.0. Human-specified workplaces and tasks guide agents to generate assets, scene variants, and process logic, and to propose evaluations for human confirmation. HOLIDAY-SIM simulates contact, actuation, and sensor responses. The top row is schematic, and the bottom panel shows 24/7-BENCH including seven industrial robot tasks in simulation.

## Contents

1 Introduction 4   
1.1 Research questions and organization   
2 Overview   
2.1 Preliminaries   
2.2 HOLIDAY-SIM   
2.3 Agentic world 6   
2.4 24/7-BENCH   
3 HOLIDAY-SIM 7   
3.1 SDF-hydroelastic contact 8   
3.1.1 Geometry and constitutive laws 8   
3.1.2 Semi-implicit contact solver 11   
3.1.3 Magnetic tactile sensing 13   
3.2 Stateful actuator 14   
3.2.1 Drive and transmission models 14   
3.2.2 State integration and dynamics coupling 16   
3.2.3 Joint telemetry 17   
4 Agentic world 17   
4.1 Constraint-driven asset generation 17   
4.1.1 Constraints and routing 17   
4.1.2 Mesh and code-native generation 18   
4.1.3 Verification and revision 18   
4.1.4 Appearance randomization 20   
4.1.5 Experiments 20   
4.2 Scenario generation . 22   
4.2.1 Inverse scene generation <sup>22</sup><sub>26</sub>   
4.2.2 Photorealistic background   
4.2.3 Process logic generation . 28   
4.3 Evaluation generation 28   
4.3.1 Proposal and confirmation . 29   
4.3.2 Compilation and verification 29   
4.3.3 Questions and verification feedback . 30   
5 24/7-BENCH 31   
5.1 Tasks 31   
5.2 Evaluation 32   
5.2.1 Jobs 33   
5.2.2 Metrics 33   
5.2.3 Rubrics 34   
5.3 Evaluation across variants 34   
6 Discussion and open questions 34   
6.1 Value of richer physical simulation . 34   
6.2 Limits of automated verification 35   
6.3 Expressiveness and diversity of generated scenes 35   
6.4 Evolving role of coding agents . 36   
6.5 Physics-grounded supervision for learned evaluators . 36   
6.6 Statistical reliability of reset-free evaluation 36   
7 Conclusion 37   
8 Contributors and acknowledgments 38   
8.1 Contributors 38   
8.2 Acknowledgments 38   
A SDF-hydroelastic contact details 45   
A.1 Collision geometry and memory 45   
A.2 Pressure law and solver discretization 45   
A.3 Contact and solver studies . 46   
A.4 Magnetic tactile response kernel 48   
B Actuator model and integration details 50   
B.1 Drive and transmission models 50   
B.2 State integration and dynamics coupling 51   
C Asset generation details 52   
C.1 Constraint types 52   
C.2 SDF construction and collider layout 52   
C.3 Models, budgets, and thresholds 53   
D Scenario generation details 55   
D.1 Inverse scene generation 55   
D.1.1 Models and budgets 55   
D.1.2 Experimental protocols . 55   
D.2 Background generation 57   
D.3 Re-rendering recorded trajectories 60   
D.4 Process logic representation 60   
D.5 Verification thresholds across components 60   
D.6 Post-hoc rubric evaluation 61   
E Evaluation generation details 63   
E.1 Rejections at compilation 63   
E.2 Verification checks 63   
E.3 Generation study protocol 64   
E.4 Feedback examples 65   
F 24/7-BENCH details 67   
F.1 Task selection 67   
F.2 Robot and simulation settings 67   
F.3 Evaluations of the seven tasks 68   
F.4 Visual evidence 70

## 1 Introduction

Robotics has a simple scaling problem. The most trustworthy evaluation happens on the real robot, but real-robot evaluation is slow, expensive, difficult to parallelize, and hard to reproduce. Each trial consumes hardware time and operator attention, scenes must be prepared and reset, and failures can damage the robot or its surroundings. These costs grow quickly for generalist policies that must be tested across many tasks, objects, and environments. Recent work therefore treats simulation not only as a training tool but also as an evaluation substrate that can be regenerated and replayed at much larger scale (Li et al., 2024b; Jain et al., 2026; Jangir et al., 2026).

Most simulation benchmarks built for this purpose, however, emphasize homes and everyday activities. BEHAVIOR-1K defines 1,000 human-centered activities across houses, gardens, restaurants, offices, and related settings, while RoboCasa and RoboCasa365 focus on large-scale everyday manipulation in household environments (Li et al., 2024a; Nasiriany et al., 2024; 2026). Other benchmarks study whole-body humanoid control or long-horizon indoor mobile manipulation (Sferrazza et al., 2024; Jaafar et al., 2025). These suites cover much of general-purpose embodied behavior, but each scores an episode that starts from a reset. We instead focus on repetitive industrial work.

Factory work is rarely a single isolated pick-and-place episode, and it is where robots are already deployed at scale. Factories installed more than 600,000 industrial robots in 2025, and the operational stock reached a record 5 million (International Federation of Robotics, 2026e). Parts keep arriving, racks fill and empty, and the next job depends on the state left by previous jobs, a condition that reset-free robot learning has begun to address (Sharma et al., 2022; 2023; Wu et al., 2026). A failed grasp may require a retry, a different stance, an intervention, or a safe stop, and naive retries often repeat the same mistake (Hao et al., 2026). Throughput matters, as in bin picking, where Dex-Net 4.0 reports more than 300 mean picks per hour alongside its reliability (Mahler et al., 2019), but so do unsafe contacts, dropped objects, blocked workspaces, recovery time, and the accumulated effect of smal errors. Small per-job failure rates also compound. Under the simplifying assumption of independent outcomes, a policy with a 95% single-job success rate completes 20 consecutive jobs only 36% of the time, while a 99% rate yields 82%. Persistent scene state can additionally make job outcomes dependent.

“The race is not always to the swift.”

— The Hare & the Tortoise, Aesop for Children

We introduce HOLIDAYFACTORY 1.0, a framework for simulating, generating, and evaluating industrial robot work. Given a human-specified workplace W and tasks T, each task is associated with a world ξ = (S, E) (§ 2). The scenario S comprises scene variants and process logic, while the evaluation E defines a job, its metrics, and a rubric. The world specification is fixed before policy evaluation, while the physical state evolves across jobs within each run without a scene reset. Agentic construction produces and verifies assets and scene variants, writes process logic, and supports the generation of evaluations from task requirements. The current framework requires human confirmation of generated evaluation proposals. HOLIDAY-SIM provides the simulated dynamics and sensor observations through which a separately chosen policy π interacts with a scene.

## 1.1 Research questions and organization

HOLIDAYFACTORY 1.0 addresses a central research question:

How can industrial robot tasks be transformed into physically executable worlds

in which robot performance can be evaluated over repeated jobs?

We approach it through three research questions.

• (RQ1) How to simulate the contact, actuation, and sensing that determine physical outcomes

• (RQ2) How to construct task-conditioned worlds that pair executable scenarios with their evaluations

• (RQ3) How to measure robot performance across repeated jobs as the physical state persists

(RQ1) Simulation must represent whether a part seats in a pocket with little clearance, whether a grasp holds, and whether the motors can deliver the required effort. Differences between simulated and real interactions can affect the reliability of simulation-based evaluation (Torne et al., 2024; Han et al., 2026; Jain et al., 2026; Jiang et al., 2026). (RQ2) Industrial tasks require scenarios that support their intended operations. Each scenario must contain task-conditioned assets, valid scene configurations, and executable process logic. Scaling their construction requires verifying task-relevant constraints during generation. (RQ3) Evaluation needs to translate continuous execution into measurements of work. Jobs require well-defined start, completion, and failure conditions, with consistent metrics and reporting criteria across policies and scene variants.

HOLIDAYFACTORY 1.0 makes three contributions:

• HOLIDAY-SIM (§ 3). A physics engine that extends hydroelastic contact on Newton<sup>2</sup>, adds stateful actuators, and resolves both with a custom solver. Contact patches also support built-in magnetic tactile sensing, while actuator and joint states provide telemetry as policy observations.

• Agentic world (§ 4). Coding agents generate and refine task-conditioned assets and scene variants through verification-guided generation, using executable checks to enforce task-relevant constraints (§ 4.1, § 4.2). The framework also generates visual backgrounds (§ 4.2.2), while a separate agent proposes evaluation for human confirmation and deterministic compilation (§ 4.3).

• 24/7-BENCH (§ 5). Seven industrial tasks define repeated jobs on persistent state. Their job outcomes and any run-level success criteria use privileged simulator state. The benchmark reports throughput, job latency, failed and unfinished jobs, and motor energy per completed job (§ 5.2). Runs are grouped by scene variant and by the variation axes a comparison declares (§ 5.3).

The experiments assess physical and tactile responses, asset and scene generation, and the verification and semantic agreement of generated evaluations. They do not yet establish end-to-end generation of the seven benchmark worlds or compare policies across those tasks.

The remainder of this paper is organized as follows. Section 2 establishes the common terminology and framework overview. Section 3 presents HOLIDAY-SIM for physical simulation. Section 4 describes agentic world construction and verification. Section 5 introduces 24/7-BENCH for evaluating persistent industrial work. Section 6 discusses limitations and open research questions.

## 2 Overview

## 2.1 Preliminaries

A workplace W is a human-specified description of an industrial site, including its equipment, the objects it handles, and where the robot operates. A workplace may host multiple tasks, each specifying the work requested of a robot policy. A task $\tau$ is expressed in natural language or structured fields, such as continuously loading totes onto a conveyor. Figure 2 illustrates four tasks within an inbound conveyor workplace.

For each task T, a world $\xi = ( S , E )$ pairs a scenario $S$ with an evaluation E. The scenario $S = ( s _ { 1 : N _ { s } } , \ell )$ consists of scene variants $s _ { 1 } , \ldots , s _ { N _ { i } }$ and process logic ℓ. Each scene specifies the robot and objects, their geometry, physical parameters, initial state, background, and cameras. The process logic governs workplace operations through continuous effects, such as conveyor motion, triggered events, such as press activation, and queues that deliver or remove parts. Following scenario-based testing in automated driving, a scene describes a snapshot of the environment and a scenario its evolution over time (Ulbrich et al., 2015). Our scenario corresponds to a logical scenario, while each run instantiates a concrete scenario using one scene variant $s _ { i }$ and process logic ℓ (Menzel et al., 2018).

A job is a unit of requested work, such as moving one tote from a pallet onto a conveyor, with its own progress and outcome. The evaluation $E = ( j , m _ { 1 : N _ { m } } , c )$ defines the job j through events on privileged simulator state marking its start, completion, and failure. The metrics $m _ { 1 : N _ { m } }$ aggregate job records and events into quantities such as completed jobs, job latency, and energy. The rubric c specifies run termination, reporting rules, and optional success criteria (§ 5.2).

A run executes a policy $\pi$ on task $\tau$ in one scene variant $s _ { i }$ of its world. An evaluation regime, defined separately from the world, specifies how scene variants and experimental conditions are sampled across runs (§ 5.3). The same evaluation applies to different policies, while their observations, control authority, and callable skills are part of the experimental condition. The world specification remains fixed during policy evaluation, although the scene state evolves within each run. Table 1 summarizes the terminology.

For a run in scene variant $s = s _ { i }$

$$
x _ { t + 1 } = F _ { s , \ell } ( x _ { t } , u _ { t } ) , \qquad o _ { t } = O _ { s } ( x _ { t } ) , \qquad u _ { t } = \pi ( o _ { 0 : t } , T ) , \qquad Q = \mathrm { E v a l } _ { E } ( \tau ) ,\tag{1}
$$

where $x _ { t } , u _ { t }$ , and $o _ { t }$ denote the simulation state, action, and observation, respectively; $F _ { s , \ell }$ advances the state under scene s and process logic $\ell ,$ while $O _ { s }$ produces the policy observations, including images and nonvisual sensor readings, and $\tau = ( x _ { 0 : N _ { \mathrm { s t e p } } } , o _ { 0 : N _ { \mathrm { s t e p } } } , u _ { 0 : N _ { \mathrm { s t e p } } - 1 } )$ ) is the recorded trajectory over $N _ { \mathrm { s t e p } }$ steps. The evaluation report Q contains job records, metric values, the termination reason, and, when required by the rubric, a verdict

![](images/04e285055764bce19e5551c3fdbcbe401f36a9f8a4150632abaf2724ee4daaab.jpg)  
Figure 2. From a human-specified workplace and its tasks to the world for continuous loading. Agents construct scene variants and propose evaluations, using feedback from deterministic checks to revise their outputs. Process logic is written by an agent, and evaluation proposals require human confirmation.

Table 1. Terminology of HOLIDAYFACTORY 1.0.
<table><tr><td>Term</td><td>Definition</td></tr><tr><td>Workplace W</td><td>Human-specified description of an industrial site, its equipment, objects, and robot workspace</td></tr><tr><td>Task  $\tau$ </td><td>Human-specified instruction given to a robot policy.</td></tr><tr><td>World ξ</td><td>Scenario and evaluation associated with one task.</td></tr><tr><td>Scenario S</td><td>Executable environment of a task, made of scene variants and the process logic that changes them over time.</td></tr><tr><td>Scene si</td><td>One variant of the objects and robot, with their physical parameters and placement, and the background.</td></tr><tr><td>Process logic l</td><td>What the site does on its own, such as per-step effects and changes triggered by a condition.</td></tr><tr><td>Evaluation E</td><td>Job definition, metrics, and rubric for a task.</td></tr><tr><td>Job j</td><td>Requested piece of work defined by start, completion, and failure events on privileged state.</td></tr><tr><td>Unit</td><td>Object that a job acts on, such as one tote.</td></tr><tr><td>Metric m</td><td>Quantity aggregated from job records and events, such as completed jobs, latency, or energy.</td></tr><tr><td>Rubric c</td><td>Run termination and reporting rules, with optional success criteria.</td></tr><tr><td>Policy π</td><td>Method under test, selected independently of the world.</td></tr></table>

Y. A report-only rubric does not assign a verdict. Core execution symbols are summarized in Table 2. Generated backgrounds affect $O _ { s }$ but not $F _ { s , \ell } ,$ allowing recorded trajectories to be re-rendered without repeating the physical simulation.

Embodiment. We use FRIDAY, a wheeled humanoid robot about 1.8 m tall with a differential-drive base, a five-joint waist, two seven-joint arms, a three-joint head, and two five-finger hands whose palms and inner finger surfaces carry compliant tactile skin, 75 joints in all. Head and wrist cameras, together with an IMU, LiDAR, odometry, joint telemetry, and the tactile skin, supply the observations $o _ { t } \ ( \ S ^ { \lceil \rceil } \mathbb { F } . 2 )$

## 2.2 HOLIDAY-SIM

HOLIDAY-SIM provides the physical state transitions and sensor responses used to execute robot policies in HOLIDAYFACTORY 1.0 (§ 3). It extends Newton’s sparse-SDF-based hydroelastic contact with contact constitutive models, a semi-implicit primal-dual solver, and stateful actuator dynamics. The contact patches also provide inputs to magnetic tactile sensing, while the actuator and mechanism states produce joint telemetry.

Each scene $s _ { i }$ supplies the geometry and physical parameters for these models, while process logic ℓ governs workplace operations during execution.

## 2.3 Agentic world

An agentic pipeline generates task-conditioned assets, scene variants, and evaluation proposals (§ 4). Asset generation produces simulation-ready objects and checks their geometry and task-relevant interactions in HOLIDAY-SIM. Scenario generation assembles these assets into scene variants with physical parameters and photorealistic backgrounds, while a scenario-authoring agent writes the process logic ℓ. Evaluation generation proposes a job, metrics, and rubric for human confirmation before compilation and executable checks. Formally,

Table 2. Core execution notation. World components are defined in Table 1; local physical quantities are defined where used.
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $N _ { s } , N _ { m }$ </td><td>Numbers of scene variants and metric definitions.</td></tr><tr><td> $x _ { t } , u _ { t } , o _ { t }$ </td><td>Simulator state, policy action, and observation at run step t.</td></tr><tr><td>π</td><td>Policy mapping observation history and task to an action.</td></tr><tr><td> $F _ { s , \ell } , O _ { s }$ </td><td>State transition under scene s and process logic l, and observation model for s.</td></tr><tr><td> $\tau , N _ { \mathrm { s t e p } }$ </td><td>Recorded trajectory and its actual number of steps.</td></tr><tr><td> $\mathcal { Q } = \mathrm { E v a l } _ { E } ( \tau )$ </td><td>Evaluation report under specification E.</td></tr><tr><td>Y</td><td>Optional run verdict: pass, fail, or undecided.</td></tr><tr><td> $\Delta t , h , N _ { \mathrm { s u b } }$ </td><td>Frame interval, physics substep interval, and substeps per frame;  $h = \Delta t / N _ { \mathrm { s u b } }$ </td></tr><tr><td> $T _ { \mathrm { r u n } } , t _ { \mathrm { r u n } }$ </td><td>Maximum run duration and actual elapsed simulation time.</td></tr><tr><td> $\tau _ { \mathrm { j o b } }$ </td><td>Timeout duration measured from a job&#x27;s start.</td></tr><tr><td> $N _ { \mathrm { g o a l } } , N _ { \mathrm { d o n e } }$ </td><td>Target and actual completed-job counts.</td></tr><tr><td> $N _ { \mathrm { e n v } }$ </td><td>Number of parallel simulation instances.</td></tr></table>

![](images/6cecfc7d7ccb6fa6a79685254ba75594ff77cd49b7d3712f7b6be0d3c99235ed.jpg)  
Figure 3. An overview of HOLIDAY-SIM. Illustrations are schematic, not simulator outputs.

$$
S \sim G ( { \mathcal T } , { \mathcal W } ) \mathrm { w i t h } V _ { \mathrm { s c e n } } ( S ) = 1 , E \sim A ( { \mathcal T } , S ) \mathrm { w i t h } V _ { \mathrm { e v a l } } ( E , S ) = 1 ,\tag{2}
$$

where G assembles scene variants with the specified process logic, A generates evaluations with human confirmation, and $V _ { \mathrm { s c e n } }$ and $V _ { \mathrm { e v a l } }$ denote their respective checks.

## 2.4 24/7-BENCH

24/7-BENCH comprises seven industrial tasks across seven workplaces (§ 5). Each task involves repeated jobs within a scene whose state persists between jobs. Some tasks use conveyors, events, or queues to supply and process work, while others operate on a fixed initial set of units. The benchmark records completed, failed, timed-out, and unfinished jobs, reporting throughput, job latency, violations, and motor energy under specified evaluation regimes. A rubric may additionally define a run-level success criterion. Table 3 compares its execution horizon, scoring units, and metrics with representative simulation benchmarks.

## 3 HOLIDAY-SIM

HOLIDAY-SIM combines Newton’s sparse-SDF hydroelastic contact with the constitutive models and a semi-implicit primal-dual solver, and couples stateful actuators to articulated dynamics (Figure 3). For contact, it models pressure, damping, and friction over finite patches and resolves their coupled response within each simulation step (§ 3.1). For actuation, it models motor controllers, electrical limits and thermal dynamics, transmissions, and internal friction, contributing joint effort and impedance to the articulated solve (§ 3.2).

Table 3. Comparison of representative robot simulation benchmarks.
<table><tr><td>Benchmark</td><td>Scale</td><td>Environment</td><td>Horizon</td><td>Simulator</td><td>Scored unit</td><td>Metrics</td></tr><tr><td>LIBERO (Liu et al., 2023a)</td><td>130 tasks</td><td>Tabletop</td><td>30s</td><td>MuJoCo (robosuite)</td><td>Episode</td><td>Success rate</td></tr><tr><td>ManiSkill3 (Tao et al., 2025)</td><td>12 domains</td><td>Tabletop Indoor</td><td>2.5-12.5s†</td><td>PhysX (SAPIEN)</td><td>Episode</td><td>Success rate Episode return</td></tr><tr><td>RoboCasa365 (Nasiriany et al., 2026)</td><td>365 tasks</td><td>Kitchen</td><td>15 s-6 min</td><td>MuJoCo (robosuite)</td><td>Episode</td><td>Success rate</td></tr><tr><td>BEHAVIOR-1K (Li et al., 2024a)</td><td>50 tasks</td><td>Household</td><td>2.4-29 min†</td><td>PhysX (OmniGibson) Episode</td><td></td><td>Goal completion Execution time Motion distance</td></tr><tr><td>RoboTwin 2.0 (Chen 50 tasks et al., 2026a)</td><td></td><td>Tabletop</td><td>400-1,700 steps†</td><td>PhysX (SAPIEN)</td><td>Episode</td><td>Success rate</td></tr><tr><td>HumanoidBench (Sferrazza et al., 2024)</td><td>27 tasks</td><td>Obstacle courses Indoor</td><td>10-20 s</td><td>MuJoCo</td><td>Episode</td><td>Success rate Episode return</td></tr><tr><td>Factory (Narang et al., 2022)</td><td>3 tasks</td><td>Assembly cell</td><td>1.7 s-2.3 min</td><td>PhysX (Isaac Gym)</td><td>Episode</td><td>Success rate</td></tr><tr><td>24/7-BENCH</td><td>7 workplaces Open task set</td><td>Industrial cell</td><td>∼ hours</td><td>HOLIDAY-SIM</td><td>Job Task</td><td>Success rate Throughput Job latency Failures Violations</td></tr></table>

Entries that the cited papers do not state follow each benchmark’s released code and documentation (ManiSkill Contributors, 2026a;b), and the BEHAVIOR-1K entries follow its 2025 challenge (BEHAVIOR Team, 2025). <sup>†</sup>Episode limits of the tabletop gripper tasks of ManiSkill3. BEHAVIOR sets each limit at twice the mean human demonstration time of the task from the 2025 challenge demonstrations. A RoboTwin step is one policy action, a planned motion of variable duration.

## 3.1 SDF-hydroelastic contact

Precision industrial manipulation depends on contact fidelity. A machined part must seat in a jig pocket with little clearance, a tote must rest stably on a stack, and a grasp must hold a heavy box without slipping or twisting. Representative benchmarks use convex decomposition and sparse point contacts (Table 4), which can fill in narrow features and collapse distributed loads onto a handful of points. Hydroelastic contact, introduced by Elandt et al. (2019) and implemented in Drake (Tedrake and the Drake Development Team, 2019; Masterjohn et al., 2022) with tetrahedral volume meshes, addresses the distributed-load limitation by integrating a pressure field over a finite contact patch. Preparing such meshes for arbitrary CAD assets can be laborious, and contact queries can be costly at benchmark scale. Newton (The Newton Contributors, 2025) implements the same pressure-balance principle on sparse signed distance fields (SDFs), avoiding tetrahedral meshing.

Building on Newton, HOLIDAY-SIM addresses three challenges in precision industrial manipulation: contact modeling, numerical stability, and memory efficiency. First, it extends Newton’s linear pressure law with compression hardening, compression-dependent damping, speed-dependent friction, and planar ground contact (§ 3.1.1). Second, a semi-implicit primal-dual solver resolves stiff, coupled contacts across articulated bodies (§ 3.1.2). Third, asset-level SDF budgets and instance-chunked collision control memory consumption, which can otherwise be dominated by the finest-resolution asset (§ 4.1.3 and § A.1). The resulting contact patches also support magnetic tactile sensing (§ 3.1.3).

## 3.1.1 Geometry and constitutive laws

HOLIDAY-SIM adopts Newton’s sparse-grid SDF representation to model nonconvex contact geometry, including narrow gaps, at lower memory cost than uniformly fine volumetric grids (§ A.1). For a body A, the signed distance field $\phi _ { A } ( \mathbf { x } )$ is negative inside the body and positive outside. It also supplies the depth-dependent pressure field of the hydroelastic model through Eq. (4). Factory (Narang et al., 2022) also uses SDFs to represent tight-clearance assembly geometry, but resolves contact at points and has no pressure patches.

Where two bodies overlap, the contact surface is the set of points at which their pressure fields balance (Elandt et al., 2019):

$$
\begin{array} { r } { S _ { A B } = \left\{ \mathbf { x } \in \Omega _ { A } \cap \Omega _ { B } \mid p _ { A } ( \mathbf { x } ) = p _ { B } ( \mathbf { x } ) \right\} , } \end{array}\tag{3}
$$

where $\Omega _ { A }$ and $\Omega _ { B }$ are the volumes occupied by the two bodies. Newton extracts this surface directly from the sparse-grid SDFs, using hierarchical voxel pruning and marching cubes, and can reduce the patch to a few representative contacts by moment matching while preserving its friction moment.

HOLIDAY-SIM specifies two mappings around this surface, from signed distance to pressure and from a pressure patch to contact force and friction. With a limited set of contacts, the force that supports an object and the moment that resists its tilting depend on how many points the simulator keeps. With a pressure patch, both follow from where and how deeply the bodies overlap. Both mappings matter across the range of stiffness in industrial contact, from stiff machined parts to compliant pads and objects. On the FRIDAY hand, compliant fingertips hold a slender tapered part through patch pressure and friction while it rolls. With point contacts the same motion drops the part (Figure 4).

Table 4. Rigid-body contact formulations in representative robot simulation benchmarks. Asterisks mark the default solver of each simulator.
<table><tr><td>System</td><td>Simulator engine</td><td>Contact geometry</td><td>Contact physics</td><td>Contact solver</td></tr><tr><td>LIBERO (Liu et al., 2023a)</td><td>MuJoCo (robosuite)</td><td>Convex decomp.</td><td>Compliant point</td><td>Newton*</td></tr><tr><td>ManiSkill3 (Tao et al., 2025)</td><td>PhysX (SAPIEN)</td><td>Convex decomp.</td><td>Point</td><td>TGS*</td></tr><tr><td>RoboCasa365 (Nasiriany et al., MuJoCo (robosuite) 2026)</td><td></td><td>Convex decomp.</td><td>Compliant point</td><td>Newton*</td></tr><tr><td>BEHAVIOR-1K (Li et al., 2024a)</td><td>PhysX (OmniGibson / Isaac Sim)</td><td>Convex decomp.</td><td>Point</td><td>TGS*</td></tr><tr><td>RoboTwin 2.0 (Chen et al., 2026a)</td><td>PhysX (SAPIEN)</td><td>Convex decomp.</td><td>Point</td><td>TGS*</td></tr><tr><td>HumanoidBench (Sferrazza et al., 2024)</td><td>MuJoCo</td><td>Convex decomp.</td><td>Compliant point</td><td>Newton*</td></tr><tr><td>Factory (Narang et al., 2022)</td><td>PhysX (Isaac Gym)</td><td>SDF</td><td>Point</td><td>TGS</td></tr><tr><td>HOLIDAYFACTORY 1.0</td><td>HOLIDAY-SIM</td><td>SDF</td><td>Hydroelastic</td><td>CDMM</td></tr></table>

TGS denotes temporal Gauss-Seidel, CDMM denotes Constrained Dynamics via Method of Multipliers (§ 3.1.2), and Newton in the solver column is MuJoCo’s Newton solver, unrelated to the Newton engine. Where the cited papers do not state a setting, the entry follows the benchmark’s released assets and configuration and the engine documentation (Google DeepMind, 2026; NVIDIA, 2022). These entries list only the configurations used for comparison, and each engine supports other contact representations as well.

For the first mapping, signed distance becomes pressure through a constitutive law, which is linear by default:

$$
p _ { A } ( \mathbf { x } ) = k _ { h , A } \left[ - \phi _ { A } ( \mathbf { x } ) \right] _ { + } , \qquad [ z ] _ { + } = \operatorname* { m a x } ( z , 0 ) .\tag{4}
$$

Here, $\phi _ { A }$ is the signed distance after any surface offset, and $k _ { h , A } ,$ , in $\mathrm { { N } / \mathrm { { m } ^ { 3 } } }$ , relates pressure to depth. Unlike a point-contact stiffness, the resulting force also grows with the contact area. Measurements on a silicone robotic fingerpad show that force rises from 1.1 to 11.7 N as the indentation grows from 1.6 to 4.3 mm and the contact area spreads from 48 to $8 7 \mathsf { m m ^ { 2 } }$ (Achilli et al., 2022). To represent additional depth-dependent stiffening, HOLIDAY-SIM uses the following law for interior depth $x _ { A } = [ - \phi _ { A } ( \mathbf { x } ) ] _ { + }$

$$
p _ { A } ( \mathbf { x } ) = k _ { h , A } x _ { A } \left[ 1 + \beta _ { A } \left( \frac { x _ { A } } { \ell _ { A } } \right) ^ { 2 } \right] , \qquad \beta _ { A } \geq 0 , \quad \ell _ { A } > 0 .\tag{5}
$$

Here, $k _ { h , A }$ sets the initial pressure gradient, $\beta _ { A }$ the amount of hardening, and $\ell _ { A }$ a characteristic depth. Setting $\beta _ { A } = 0$ recovers the linear law. The law does not model the interior of the pad. The depth $\ell _ { A }$ need not equal the pad thickness, and plasticity, creep, and lateral bulging are not represented (§ A.2).

The same mapping covers the limit of an infinitely stiff body. Against rigid ground, pressure balance places the contact surface on the plane, so HOLIDAY-SIM integrates the compliant body’s pressure directly over it without building a ground SDF. An unbounded plane needs no volumetric field, yet wheels still need patch-level pressure and friction because slip speeds vary across the footprint and produce a net force and moment. For a ground plane $\Pi _ { g }$ with unit normal $\mathbf { n } _ { \Pi _ { g } }$ pointing toward the body, the patch and elastic normal force are

$$
{ \mathcal { S } } _ { A \Pi _ { g } } = \Omega _ { A } \cap \Pi _ { g } , \qquad { \bf f } _ { n } ^ { \mathrm { e l } } = { \bf n } _ { \Pi _ { g } } \int _ { S _ { A \Pi _ { g } } } p _ { A } ( { \bf x } ) \mathrm { d } A ,\tag{6}
$$

evaluated with two-dimensional quadrature over the footprint.

For the second mapping, a pressure patch becomes contact force by integrating the traction over the contact surface:

$$
\mathbf { f } _ { B  A } = \int _ { S _ { A B } } \mathbf { t } _ { B  A } ( \mathbf { x } ) \mathrm { d } A , \qquad \tau _ { B  A , \mathbf { o } } = \int _ { S _ { A B } } ( \mathbf { x } - \mathbf { o } ) \times \mathbf { t } _ { B  A } ( \mathbf { x } ) \mathrm { d } A ,\tag{7}
$$

where $\mathbf { t } _ { B  A }$ is the total traction that body B exerts on body A, including friction and damping, and o is a reference point. Because the patch grows as the bodies press together, the net force rises faster than linearly even under a linear pressure law. When a resting object tilts, the pressure shifts toward the lower side and produces a moment that opposes the tilt. The model represents local compliance between nominally rigid bodies. Bulk deformation is not modeled. Because the patch is reduced to representative contacts, force and moment are approximated, with accuracy set by grid resolution and contact reduction.

![](images/e8249d95f49d81257e046ddb6f1d6d84209e4460aeb11c151f398fe502e7daa2.jpg)

![](images/ec8e7f69142113000f70937a97826e1e84024b4495d9594e883513a03cb538b5.jpg)  
Figure 4. FRIDAY fingertips rolling a slender tapered part under the same finger motion. Top, point contacts on convex pieces. In this configuration, the part drops at 2.4 s. Bottom, sparse-grid SDFs with hydroelastic patches hold the part through the whole motion. The right column shows the contact on the part at 1.67 s, a single point against a pressure patch.

HOLIDAY-SIM adds Hunt-Crossley damping (Hunt and Crossley, 1975), which grows with compression and helps a part settle. A purely elastic contact would return all stored energy, so a dropped part would keep bouncing. This damping removes energy in proportion to both compression and approach speed. The stiffness $K _ { i }$ of each representative contact is derived from the pressure patch and held fixed within each substep. For penetration $\delta _ { i } \geq 0$ and normal velocity $v _ { n , i ; }$ , positive when the bodies separate, $f _ { n , i } = K _ { i } \delta _ { i } \left[ 1 - \alpha v _ { n , i } \right] _ { + }$ with zero force outside contact and a dissipation coefficient $\alpha ,$ in $\mathrm { s / m }$ , shared across contacts. The corresponding damping coefficient $c _ { \mathrm { e f f } , i } = K _ { i } \alpha \delta _ { i }$ vanishes at first touch and grows as the contact is compressed, so the same approach speed is damped more strongly in a deeper contact. Loading and unloading therefore follow different force curves without any stored material history, and the force never becomes tensile.

HOLIDAY-SIM applies friction over the whole patch to model a grasp’s resistance to twisting. Each representative contact applies friction bounded by its share of the normal load at its own lever arm, approximating the resisting torque from the pressure distribution. A point-contact manifold can also resist twisting through tangential forces at spatially separated contacts, with the resulting torque depending on their locations and load distribution. HOLIDAY-SIM also lets this friction weaken once sliding starts, so that an interface can hold at rest yet resist less while it slides. With normal force $f _ { n , i } \geq 0$ , tangential force $\mathbf { f } _ { t , i } ,$ , and slip speed $v _ { \mathrm { s l i p } , i } = \| \mathbf { v } _ { t , i } \|$ at contact $i ,$ the friction law satisfies

$$
\begin{array} { r l } & { \mu _ { i } ( v _ { \mathrm { s l i p } , i } ) = \mu _ { s , i } \left[ r + ( 1 - r ) e ^ { - v _ { \mathrm { s l i p } , i } / v _ { c } } \right] , } \\ & { \qquad \| \mathbf { f } _ { t , i } \| \leq \mu _ { i } ( v _ { \mathrm { s l i p } , i } ) f _ { n , i } , \qquad \mathbf { f } _ { t , i } = - \mu _ { i } ( v _ { \mathrm { s l i p } , i } ) f _ { n , i } \frac { \mathbf { v } _ { t , i } } { v _ { \mathrm { s l i p } , i } } , \qquad v _ { \mathrm { s l i p } , i } > 0 . } \end{array}\tag{8}
$$

Here, $\mu _ { s , i }$ is the static coefficient, $r \in [ 0 , 1 ]$ sets the sliding coefficient $r \mu _ { s , i } ,$ and $v _ { c } > 0$ is the transition speed. The same exponential decay is used in continuum contact modeling (COMSOL, 2025). Setting $r = 1$ recovers

Spatial load distribution

(a) Contact patch at 0.5 mm indentation

![](images/2b400c042236c528a3c9872ff68a7b45c2b82184cab63fc860430a5b0eaebf83.jpg)  
(d) Twist resistance  
Compression-dependent dissipation

![](images/27eab792cdb77e43e25ccf59371e3595a4ac45fef80cf963f60a083fb0146eeb.jpg)  
Compression hardening  
(c) Local pressure law

![](images/fbf0819e1dd65f230b2c742d6cc1532745042cc7abe6a7370d64bc5f500188b9.jpg)

![](images/94cd2b8aaec80ef34e4b06833ecdf0a8f6b07196f66fc5ca2325e039093034a5.jpg)

(e) Damping  
![](images/868a08c67c53c849413b2e350afc1c0a788b8b422ac6776a4e492d56bed849da.jpg)

(f) Elastic force  
![](images/c1dbff181ee2e7126676bbbe9d4f6039557e6c777a9bba5dd64f7601ab29592c.jpg)  
Figure 5. FRIDAY fingertip contact response. Left, (a) pressure patch at 0.5 mm indentation and (d) effective friction radius under pure twist, for HOLIDAY-SIM and a point-contact configuration. Middle, (b) prescribed loading and unloading loops and (e) the effective damping they imply. Right, (c) depth-hardening pressure laws for several β and (f) the elastic forces they integrate to. Dashed curves are quadrature references of the same law, and the first two columns use the linear law.

constant Coulomb friction. The parameters $r$ and $v _ { c }$ are currently shared across contacts. Because representative contacts can stick or slip independently, the patch also gives a discrete approximation of partial slip. Tangentia elastic deformation, micro-slip, and rolling resistance are not resolved.

Sparse-grid SDFs are used for all volumetric collision geometry, including simple shapes such as cubes, so that every contact receives the same hydroelastic treatment, and only the ground uses the planar path above.

Contact probes. Controlled probes isolate three responses of the contact model from any benchmark task and compare each with a quadrature of the same constitutive law. Under pure twist of the FRIDAY index fingertip against a rigid plane, the effective friction radius $r _ { \mathrm { e f f } } = | \tau _ { n } | / ( \mu N )$ is the pressure-weighted distance from the rotation axis for normal load N and resisting torque $\tau _ { n } ( \mathsf { F i g u r e 5 } ( \mathsf { a } , \mathsf { d } ) )$ ). A dense quadrature of the linear pressure law gives a radius that increases with indentation. HOLIDAY-SIM, using Newton’s moment-matching reduction, follows this trend. The tested point-contact configuration gives a larger, nearly constant radius. Under prescribed loading and unloading of the same fingertip, the computed damping grows with indentation as the Hunt-Crossley law predicts, unlike a constant damper (Figure 5(b,e)). Direct quadrature shows that raising $\beta$ keeps the response to light contact unchanged and increases resistance at depth (Figure 5(c,f)). This stiffening of the local law is separate from the growth of the patch noted above. On a simplified differential-drive rig, the planar hydroelastic path is compared with a rim-point baseline across wheel widths (Figure 6). The loss of yaw gain with width is the turn-slip resistance that tire models integrate over the footprint (Pacejka, 2012) and that point-contact kinematics of skid-steered robots omit (Mandow et al., 2007). Here it follows from the same patch contact as every other body, without a tire model or a hand-set torsional friction coefficient. Tire deformation, tread, and hysteresis are not modeled. These probes compare the model with a numerical reference and involve no hardware measurement.

## 3.1.2 Semi-implicit contact solver

The contact laws above specify the forces that arise from local deformation and motion. The solver must resolve these forces together with the motion of the contacting bodies. Hydroelastic patches produce many coupled contacts on each body or articulated chain, with stiffnesses ranging from compliant pads to near-rigid supports. Within a fixed iteration budget, numerical errors can cause resting parts to rattle, grasped objects to slip, or supported loads to sink.

HOLIDAY-SIM uses a semi-implicit discretization to limit the nonlinear coupling within each solve while retaining the dependence of contact forces on the unknown next-step motion. Contact geometry and selected constitutive quantities are held fixed within each substep, while stiffness and damping enter the implicit force–motion relation.

(a) FRIDAY  
![](images/5ca62d85c4f046285a1705e434971a86ab80764dff30ee61e60602383f8b7b3f.jpg)

(b) Contact patch on the drive wheel  
![](images/64eeb377f5826a1fbce9bdf0875f8cad4c2e9ed9b434823ad244ee65663e1003.jpg)

(c) In-place rotation  
![](images/099bca19bccb6235bb5a22d12a1d4e7ac23fe0fea637d1f48351d96546a41029.jpg)  
Figure 6. Wheel-ground contact. (a) F with its right drive wheel marked behind the base skirt. (b) The hydroelastic pressure sampled on the authored wheel profile under a static load, drawn through the wheel and stretched along the rolling direction for visibility. Rim contacts place one point at each tread edge instead. (c) Yaw gain in the two-wheel width sweep.

This avoids solving simultaneously for evolving contact geometry, material coefficients, and body motion; geometry and coefficients are refreshed at the next substep. It folds Hunt-Crossley damping into the implicit solve through a single contact time scale, so stiffness and damping act together on the unknown next-step velocity (§ A.2).

Friction, including its speed weakening, is solved in the same step without adding nonlinearity to the iteration. The patch-dependent resistance to sliding and twisting (§ 3.1.1) is resolved together with the next-step motion. The coefficient $\mu _ { i } ^ { k } = \mu _ { i } ( \| \mathbf { v } _ { t , i } ^ { k } \| )$ is evaluated from the local slip speed at the start of each substep and held fixed during the solve $( \ S \ A . 2 )$

The resulting system is solved by Constrained Dynamics via Method of Multipliers (CDMM), a primal-dual iteration that retains both motion and contact reactions as unknowns. The comparison by Macklin et al. (2020) shows that large stiffness ratios can impede primal descent, whereas large mass ratios can impede dual descent. This motivates the variable choice but does not guarantee convergence of the HOLIDAY-SIM solver. The algorithmic core of CDMM builds on prior work (Lee et al., 2025) that separates system dynamics from local contact relations using auxiliary contact variables and augmented-Lagrangian multiplier updates. HOLIDAY-SIM specializes the local response to the hydroelastic stiffnesses and semi-implicit Hunt-Crossley law above.

Splitting motion from contact lets CDMM handle many stiff hydroelastic contacts without a dense contact solve. The motion update accounts for how contact forces act through articulated dynamics, and the contact update enforces the local material and friction response. Repeated exchange between the two brings predicted motion and contact reactions into agreement, so coupling between contacts is retained. Each articulation or free body is solved with an augmented inertia matrix whose factorization is reused across iterations while the geometry, implicit coefficients, and augmentation parameter remain fixed. Local contact responses are evaluated in parallel.

A patch is resampled at every step, so the same support can reappear as new contact samples even while the physical interaction persists. HOLIDAY-SIM warm-starts such samples. It reuses previous impulses for matched contacts in the updated contact frame and estimates impulses for new or unmatched contacts from their compression, stiffness, damping, and approximate inertial response. Together with the current generalized velocity and the contact slack variables reconstructed from it, these estimates initialize both the motion and reaction sides of the solve. Substepping complements this initialization by refreshing contact geometry and material coefficients more frequently. The same time interval can be divided into smaller substeps, each with fewer solver iterations. Macklin et al. (2019) show in XPBD that this allocation can reduce constraint error and numerical damping. In HOLIDAY-SIM, successive substeps warm-start from matched reactions. Additional substeps nevertheless require repeated collision processing and factorization, so the balance between substeps and solver iterations depends on the scene.

The numerical changes to CDMM are guided by the inertia behind each contact, which determines both how its update should be scaled and how stiff a response the time step can resolve. Contact preconditioning uses this inertia to improve convergence. Contacts on light parts and heavy articulated mechanisms can converge at different rates, so HOLIDAY-SIM scales each contact block by an approximate inertial metric that incorporates contact directions and connectivity, applying the same scalar to its normal and tangential coordinates so that the isotropic friction cone is not distorted. The scaling is applied consistently to the dynamics and local contact updates and leaves the material law unchanged. Near-rigid regularization uses it to keep stiff contacts stable.

![](images/4d3fa23049fd9a3b505d13ae1dc06afd7471b3c6c5d86e2e165b06cdc7e6acb9.jpg)  
Figure 7. Battery lifting with CDMM (top) and MuJoCo (bottom) at $h = 1 / 2 4 0 0 s$ using the same collision pipeline, initial state, and joint position targets. Columns show t = 0, 1.150, 1.183, and 1.185 s, the last two being five substeps apart. CDMM completes the lift, whereas the MuJoCo state jumps abruptly and becomes nonfinite at t = 1.189 s.

Shallow contacts whose stiffness exceeds what their inertia and the time step can resolve are softened, while deeper compression recovers the nominal stiffness (§ A.2).

Comparison with MuJoCo. We compare CDMM with Newton’s MuJoCo backend in a battery-lifting task involving compliant fingertips, lightweight finger links, and a substantially heavier rigid battery. Both backends share the SDF-hydroelastic collision pipeline, initial state, and joint targets, but retain different contact treatments. CDMM completes more lifts under the tested configurations (Table 5). However, the grasp and targets were tuned using CDMM, so the comparison does not isolate solver performance. Figure 7 illustrates an additional rollout.

Table 5. Battery-lift success across four time steps, with five rollouts per step size.
<table><tr><td>Backend</td><td>Iter. cap</td><td>Lifted</td></tr><tr><td>MuJoCo</td><td>15</td><td>1/20</td></tr><tr><td>MuJoCo</td><td>100</td><td>0/20</td></tr><tr><td>CDMM</td><td>32</td><td>20/20</td></tr></table>

## 3.1.3 Magnetic tactile sensing

Tactile sensing captures contact onset and load changes that may be obscured from vision. The FRIDAY hand uses compliant skin with embedded magnets whose contact-induced displacement changes the magnetic field measured by Hall sensors (Bhirangi et al., 2021; Tomo et al., 2018). Existing simulators model skin deformation using finite-element methods (Si et al., 2024) or approximate contact-response models (Si and Yuan, 2022; Akinola et al., 2025; Dang et al., 2026). HOLIDAY-SIM instead uses hydroelastic contact patches (§ 3.1.1) to generate magnetic tactile responses, sharing the contact representation between dynamics and sensing.

The patches give the sensor its load distribution without a second contact computation. The same net load applied over a broad face, a narrow ridge, or two separate patches deforms the skin differently, and the patches already resolve these differences down to the SDF and patch resolution. The sensor reconstructs elastic pressure from each contact’s depth and stiffness with the linear law of Eq. (4) and weights it by contact area to obtain the distributed normal load.

Following HydroShear (Dang et al., 2026), the sensor turns these loads into magnet displacements by superposing spatially weighted contributions and does not solve volumetric elasticity. With reconstructed pressure $p _ { i }$ and area A at contact position $\mathbf { x } _ { i }$ , the displacement of magnet m is

$$
\mathbf { u } _ { m } ^ { \mathrm { p r e s s } } \approx \sum _ { i } \mathbf { G } _ { m } ( \mathbf { x } _ { i } , \mathbf { n } _ { i } ) p _ { i } A _ { i } \approx \int _ { S } \mathbf { G } _ { m } ( \mathbf { x } , \mathbf { n } ) p ( \mathbf { x } ) \mathrm { d } A ,\tag{9}
$$

where $\mathbf { G } _ { m }$ , in $\mathrm { m } / \mathrm { N } _ { \mathrm { i } }$ , is the magnet’s displacement per unit normal load at $\mathbf { x } _ { i }$ along the inward normal $\mathbf { n } _ { i }$ representing inward compression and radial deformation around the loaded region.

Sliding contact also shears the skin, and HOLIDAY-SIM stores the shear history on the sensor. Each query point on the sensor surface records its contact position at onset as a reference attached to the object. The tangential

![](images/9170090067ba68529bf9e2ea8ff5b4ff47612bd8f79613ec644970cd801a3fbb.jpg)  
Figure 8. Two contacts with the same net load but different patch distributions on the FRIDAY tactile skin. The reconstructed pressure displaces the embedded magnets differently and produces different Hall readings.

displacement of that reference adds to magnet displacement and resets when the query loses contact. Unlike HydroShear, which tracks predefined object-surface points with Coulomb-limited history (Dang et al., 2026), this sensor-side history needs no tactile samples prepared for each new object (§ A.4).

Hall readings follow from the displaced magnets through the point-dipole field:

$$
\mathbf { B } ( \mathbf { r } ) = { \frac { \mu _ { 0 } } { 4 \pi r ^ { 3 } } } \left[ 3 ( \mathbf { m } \cdot { \hat { \mathbf { r } } } ) { \hat { \mathbf { r } } } - \mathbf { m } \right] ,\tag{10}
$$

where B is the flux density from one magnet with moment m, held fixed in the sensor frame, r points from the displaced magnet to the Hall sensor, $r = \| \mathbf { r } \| , { \hat { \mathbf { r } } } = \mathbf { r } / r$ , and $\mu _ { 0 }$ is the vacuum permeability. Each Hall sensor sums these contributions in its own frame.

The coupling is one-way. The reconstructed deformation does not feed back into collision geometry or the dynamics, and the sensor reads reconstructed elastic loading rather than the solver’s final reactions (§ A.4).

Figure 8 shows two contacts of equal net load whose different patch distributions give different Hall readings, which a single net-force sensor model cannot reproduce.

Patch-dependent readings. We pressed three rigid probes against a FRIDAY fingertip pad until each reached the same net normal load of 1 N (Figure 9). The Hall readings differ with the shape and position of the probe, whereas a model driven by net force alone would return one reading for all three. These are model outputs with default kernel parameters, so the figure shows what the sensor model distinguishes rather than how closely it matches hardware.

## 3.2 Stateful actuator

The motor, its drive, and its transmission decide which forces the robot can deliver and how quickly it responds. The benchmarks in Table 6 use joint-level torque commands or PD drives without internal actuator states, which capture the commanded behavior but omit the dynamics that govern tracking, sustained force, and motion near rest. HOLIDAY-SIM carries the actuation chain from a commanded target to the delivered joint effort with states advanced at every step. These are the shaped command and error integral of the drive, the winding current and temperature of the motor, the chamber effort of a single-acting transmission, and the bristle deflection of the internal friction (§ 3.2.1). The articulated solve receives these states through their implicit tangents (§ 3.2.2), and the joint telemetry a policy observes is reported from the same states (§ 3.2.3).

## 3.2.1 Drive and transmission models

HOLIDAY-SIM models command latency, first-order target filtering, and position slew limits, using either motorposition PID with current or voltage commands or cascaded joint-position and velocity control. The drive includes gravity and friction compensation, current feedforward, and saturation-aware anti-windup.

![](images/152e12f3f558f1741cbb577ae8dbfda5d965db77f542f3987bf6991d59dbbcb1.jpg)  
Figure 9. Simulated Hall responses of a FRIDAY fingertip pad to three rigid probes at the same net normal load of 1 N, viewed from the contact side with the fingertip up. The orange region is the hydroelastic contact patch, black dots mark the Hal sensors, and arrows show the in-plane field change on a common scale. The out-of-plane component is not drawn.

Table 6. Actuation models in representative robot simulation benchmarks.
<table><tr><td>System</td><td>Actuation model</td><td>Actuation characteristics</td><td>Internal actuator states</td></tr><tr><td>LIBERO</td><td>Ideal torque/servo actuation</td><td>Torque limits</td><td>None modeled</td></tr><tr><td>RoboCasa365</td><td>Ideal torque/servo actuation</td><td>Torque limits</td><td>None modeled</td></tr><tr><td>ManiSkill3</td><td>Ideal PD drives</td><td>Torque limits</td><td>None modeled</td></tr><tr><td>HOLIDAYFACTORY 1.0</td><td>Embedded control → motor/transmission</td><td>Electrical limits; transmission and friction dynamics</td><td>Electrical, thermal, friction, hydraulic, and controller states</td></tr></table>

“Ideal” denotes actuation without separate drive dynamics, and its output is still limited.

Motor current and electromagnetic torque follow

$$
L \dot { i } = V - R ( T ) i - k _ { e } \omega _ { m } , \qquad \tau _ { \mathrm { e m } } = k _ { t } i ,\tag{11}
$$

where $L$ is winding inductance, $R ( T )$ temperature-dependent resistance, i current, $V$ terminal voltage, $\omega _ { m }$ motor speed, and $k _ { e }$ and $k _ { t }$ the back-EMF and torque constants. Current limits and the terminal-voltage range, bounded by bus voltage and maximum duty cycle, constrain the output. The inductance delays the current buildup, and torque falls once back-EMF consumes the available voltage. A lumped thermal state heats the winding through copper loss $i ^ { 2 } R ( T )$ and cools it toward ambient. The resulting increase in winding resistance can reduce available torque when voltage-limited. Current, terminal power, copper loss, and temperature are available as diagnostics (§ B.1).

A motor’s effect on joint motion depends on its transmission, so the model includes geometric transmissions as well as constant gearing. With an offset slider-crank, the same motor produces different joint torque and reflected inertia across the joint range. Writing motor position as a function of joint position gives

$$
\theta _ { m } = \phi ( q ) , \qquad r ( q ) = \frac { \mathrm { d } \phi } { \mathrm { d } q } , \qquad \omega _ { m } = r ( q ) \dot { q } , \qquad I _ { \mathrm { r e f } } ( q ) = I _ { r } r ( q ) ^ { 2 } ,\tag{12}
$$

where $I _ { r }$ is rotor inertia and the reflected-inertia relation applies while the motor is coupled. Motor-side effort is mapped through the local transmission ratio $r ( q )$ and a scalar efficiency, and the dynamics include the inertia bias $I _ { r } r ( q ) r ^ { \prime } ( q ) \dot { q } ^ { 2 }$ , with $r ^ { \prime } ( q ) = \mathrm { d } r / \mathrm { d } q$ , that the changing reflected inertia produces (§ B.1).

Small commands and motion reversals expose internal friction that viscous damping cannot represent, so the model adds motor-side LuGre friction (Canudas de Wit et al., 1995). Unlike the memoryless contact-friction coefficient of § 3.1.1, LuGre friction carries a bristle state that represents presliding deformation and historydependent resistance, and a Stribeck envelope governs the transition toward sliding. Together these reproduce small-load holding, breakaway, and reversal. Optional temporally correlated variation of the LuGre parameters, a low-speed torque disturbance, and motor-angle-dependent cogging add the variability seen in precision motion and position holding (§ B.1).

(a) Single-acting thumb mechanism  
![](images/a8f97065f654deae6067904975727fc1fe9822a93758650e02cecba97c4a4e25.jpg)

![](images/4c22b19581d52b7f9ff3599f51ee51f753c0a6fe728ba9561a176b8673435e15.jpg)  
Figure 10. Single-acting drive and passive return in the FRIDAY thumb example. (a) Snapshots from one cycle of the single-acting hydraulic finger. The colored linkage is a kinematic visualization. (b) Joint motion, motor current, and joint-side torques over two cycles. Shaded intervals command release. Solid curves use the reduced hydraulic model with the hand’s parameters, and dashed curves replace it with a coupling switch on the sign of the requested torque, which drops the motor’s damping for one step at release and re-engages the motor at its current limit.

The FRIDAY fingers use single-acting hydraulic drives. A motor drives a piston through a lead screw, hydraulic pressure closes the finger, and a return spring opens it. The drive can exert force only in the closing direction. HOLIDAY-SIM models this response with a nonnegative chamber-effort state that builds and decays with first-order dynamics, and flow damping that resists motion in both directions (§ B.1). The return spring and damper act through the transmission independently of electrical saturation. Without chamber dynamics, a coupling switch based on the sign of the requested torque approximates the unilateral drive.

In the FRIDAY thumb example, the motor drives the joint toward its target, while commanded release allows chamber effort to decay as the spring reopens the joint against flow resistance (Figure 10). The drive and return torques capture the asymmetric response absent from a bidirectional servo model. In contrast, the coupling-switch approximation drops motor damping for one step at release.

## 3.2.2 State integration and dynamics coupling

Actuators can have high stiffness and internal dynamics that evolve much faster than the robot’s motion. Applying their effects as explicit joint forces can therefore make the result strongly dependent on the step size. HOLIDAY-SIM updates internal states locally and incorporates the dominant inertial, stiffness, and damping effects into the articulated solve through a semi-implicit contribution. This couples the actuator response to the robot’s motion without a fully nonlinear electromechanical solve.

Contact and actuators are solved together, so the actuator’s impedance contributes to the contact response. Reflected rotor inertia and the controller and passive terms enter the articulated dynamics used by the contact iteration, schematically as

$$
\mathbf { M } _ { \mathrm { a u g } } = \mathbf { M } + \mathrm { d i a g } \left( I _ { \mathrm { r e f } } + h D _ { \mathrm { a c t } } + h ^ { 2 } K _ { \mathrm { a c t } } \right) .\tag{13}
$$

Here, M includes the joint armature, h is the substep duration, and $K _ { \mathrm { a c t } }$ and $D _ { \mathrm { a c t } }$ collect local implicit coefficients expressed in joint coordinates, arising from controller response, passive springs and damping, and a nonnegative approximation to the LuGre stiffness. Corresponding force corrections accompany these terms.

The controller’s contribution to the implicit response depends on the control mode, so the same numerical gain need not produce the same joint impedance. Voltage-controller gains are converted to motor effort through $k _ { t } / R ( T )$ and scaled by the current that builds within the substep. Back-EMF adds implicit damping where the limits leave it active, while cascade gains act in joint coordinates.

Implicit stiffness and damping terms are retained when the output is clipped, providing regularization as a drive saturates or a single-acting drive switches to passive return. For the controller, this applies at a current or voltage limit. Similarly, hydraulic flow damping is retained across the zero-output boundary, because a return spring or external load can reverse motion within a substep.

The LuGre term enters the solve through a state- and direction-dependent stiffness approximation. After the solve, the bristle state advances with the solved joint velocity, which removes the artificial creep of advancing it with the pre-solve velocity. Electrical and thermal states use exponential updates with their driving quantities held fixed over the substep.

## 3.2.3 Joint telemetry

HOLIDAY-SIM generates joint telemetry from actuator and mechanism states, with configurable sampling and measurement errors. The reported channels include joint position, estimated velocity, motor current, and joint effort. Encoder resolution, filtering, calibration error, and correlated noise can obscure small motions, delay velocity estimates, and cause reported effort to differ from delivered joint torque.

Motor position is quantized at the encoder resolution and mapped to joint position through the local transmission ratio, with an optional joint-position bias. Velocity is estimated from successive samples over the actual elapsed interval and optionally low-pass filtered. Outputs are held between samples, so a faster physics rate does not raise the telemetry rate.

Effort can be reported from motor current, as on hardware. With measured current $\hat { i } ,$ torque constant $k _ { t }$ transmission ratio $r ( q )$ from Eq. (12), and efficiency η, the reported effort is

$$
\hat { \tau } _ { \mathrm { c u r r e n t } } = a _ { \tau } \eta r ( q ) k _ { t } \hat { i } + b _ { \tau } + \epsilon _ { \tau } ,\tag{14}
$$

where $a _ { \tau }$ and $b _ { \tau }$ are calibration scale and bias, and $\epsilon _ { \tau }$ is optional correlated error. Because internal friction and passive mechanisms sit between current and output, this estimate differs from the net joint effort. An alternative channel reports the computed net effort before calibration and noise.

Observation errors stay separate from physical disturbances. $\mathsf { A n }$ Ornstein-Uhlenbeck process adds correlated error to the effort channel without changing the forces on the robot. These outputs reach a policy only when a run exposes them as policy inputs, and their parameters are recorded with the benchmark settings (§ F.2).

## 4 Agentic world

A task-conditioned world $\xi = ( S , E )$ needs assets that support the interactions of the task, scene variants with process logic, and an evaluation consistent with both, each verified before use. ERIS generates and verifies assets (§ 4.1), inverse scene generation builds and verifies scene variants and adds backgrounds (§ 4.2), and evaluation generation turns the task into a confirmed, compiled, and checked evaluation (§ 4.3).

## 4.1 Constraint-driven asset generation

We introduce ERIS, a constraint-driven asset generation system that selects, constructs, verifies, and revises each asset according to the guarantees its downstream interaction requires, not according to its object category. Figure 11 illustrates the range of rigid and articulated assets produced by the system.

Visual fidelity and diversity alone are insufficient for industrial manipulation. Manipulation tasks involve tight fixtures, part-to-part contact, and articulated mechanisms, where small errors in dimensions or clearances decide whether an interaction is feasible. A visually plausible fixture can still block insertion if its mating geometry is slightly off. Collision geometry that is finer than needed or badly partitioned inflates memory when many simulation instances run in parallel.

Assets from 3D repositories (Chang et al., 2015; Deitke et al., 2023) and generative models provide broad semantic and visual coverage, but are not verified against a particular task. ERIS verifies each asset against its task on the representation the simulator consumes and revises it from the measured violations before it enters a scene (Figure 12).

## 4.1.1 Constraints and routing

Given a natural-language task instruction or asset request, ERIS first infers a structured set of task constraints $\widehat { \mathcal { R } } _ { \mathrm { t a s k } }$ using an LLM. The constraints are drawn from a fixed set of constraint types covering dimensions and counts, wall thickness, clearance and access, articulation, functional behavior, and appearance (§ C.1). Each type specifies the fields needed to instantiate it and, when applicable, its corresponding executable geometry check, articulation intent, or functional simulation claim.

ERIS has two generators that differ in how geometry is produced. A mesh generator reconstructs a surface from a generated image, and a code-native generator writes an executable construction program. Only the latter can declare joints, and together they provide three branches: mesh, code-rigid, and code-articulated. Given $\widehat { \mathcal { R } } _ { \mathrm { t a s k } }$ branch selection is deterministic:

![](images/1bd0db4e8c6518789655f5821871772e119129dcf40a09714dc14d0b6e061a55.jpg)  
Figure 11. Example assets constructed by ERIS. Red arrows indicate articulated degrees of freedom.

$$
\begin{array} { r } { \boldsymbol { b } = \rho ( \widehat { \mathcal { R } } _ { \mathrm { t a s k } } ) , \qquad \boldsymbol { b } \in \{ \mathsf { M E S H } , \mathsf { C O D E - R I G I D } , \mathsf { C O D E - A R T I C U L A T E D } \} , } \end{array}\tag{15}
$$

where $\rho$ selects a generation branch from typed task constraints. Requests prioritizing shape or appearance diversity without explicit geometric or interaction constraints route to mesh. Requests specifying dimensions, counts, thicknesses, or clearances route to code-rigid. Requests requiring relative motion or articulation route to code-articulated, regardless of other constraints.

Routing depends on task requirements rather than object category. For example, the same container can route to mesh for visual diversity, code-rigid for prescribed fixture clearances, or code-articulated when movable parts are required.

## 4.1.2 Mesh and code-native generation

The two generators trade visual detail for explicit control. The mesh generator chains a text-to-image model, an image-conditioned 3D mesh generator (Chen et al., 2026b), and post-processing (Oh et al., 2025). It inherits the shape detail of generative models but exposes no construction parameters. The code-native generator, shared by both code branches, writes an executable Blender program that defines geometry, named parts, and their hierarchy, so dimensions, counts, and clearances are set explicitly. In the code-articulated branch, the program also declares joints over the part hierarchy with their type, axis, limits, and drive parameters. All three branches ultimately produce assets with geometry, physical parameters, and verified collision representations, together with joint declarations where articulation is required. The mesh generator keeps the texture of the generative model, whereas the code-native generator assigns part-aware materials during generation and a canonical PBR appearance after the asset passes verification.

## 4.1.3 Verification and revision

ERIS places verification inside the generation loop. Each round of the loop runs up to four verification stages on the asset in Table 7, and feeds every failure back to the generator before the next round. The stages execute checks derived from the applicable task constraints and the checks that HOLIDAY-SIM always requires, such as construction validity, hierarchy, and collision representation. Which stages apply depends on the branch and on the capabilities the asset declares (Table 8).

Within a round, ERIS runs every stage the asset can reach rather than stopping at the first violation, so that all violations are revised together. The generator receives each violation with its measured value, and the checks that already pass so that it does not break them. Generation succeeds once a round passes every check, and otherwise ERIS reports the remaining violations after K rounds.

![](images/d8193319a3ebc6fc19c3828e53feefcbab1e21c4eaa3823b86d873e788b7d664.jpg)  
Figure 12. An overview of ERIS. A natural-language task instruction is first converted into a structured set of typed task constraints, from which ERIS deterministically selects one of three generation branches: mesh, code-rigid, or code-articulated. All three branches emit a common simulation-ready format and undergo the applicable verification stages. Measured failures and diagnostics are returned through a branch-specific revision path. Green marks the stages that apply only to articulated assets and the verified outputs.

Table 7. Verification stages in ERIS with the checks most directly tied to task feasibility and downstream simulation.
<table><tr><td>Stage</td><td>Representative checks</td><td>Example failure</td></tr><tr><td>Conformance</td><td>categorical attributes, dimensions, counts</td><td>rack has 3 shelves where 4 are requested</td></tr><tr><td>Collider</td><td>resolution admissibility, SDF fidelity, layout preservation</td><td>thin tote wall vanishes after SDF baking</td></tr><tr><td>Articulation</td><td>joint type, limits, motion range, binding</td><td>drawer binds before reaching its limit</td></tr><tr><td>Interaction</td><td>insertion, rolling, transport, retention</td><td>peg jams on insertion</td></tr></table>

Table 8. Verification stages by generation branch. ✓denotes a standard stage, “task” a task-dependent stage, and – a non-applicable stage.

<table><tr><td>Branch</td><td>Conf.</td><td>Coll.</td><td>Artic.</td><td>Inter.</td></tr><tr><td>Mesh</td><td>√</td><td>√</td><td>一</td><td></td></tr><tr><td>Code-rigid</td><td>√</td><td>√</td><td>一</td><td>task</td></tr><tr><td>Code-articulated</td><td>√</td><td>√</td><td>√</td><td>task</td></tr></table>

Conformance. Conformance checks whether the generated asset satisfies the properties requested by the task instruction. For the mesh branch, a VLM gives a discrete value for each categorical property, such as the presence of handles and openings, from turntable renders of the asset. When a value differs from the request, ERIS updates the asset’s label to match instead of rejecting it.

Code-native assets expose named parts and explicit geometric parameters, allowing conformance to be measured directly against requested dimensions, counts, clearances, and other structured constraints. Each check reports the requested and the measured value, and the code-native generator revises the program from them.

Collider. Assets in HOLIDAYFACTORY 1.0 are consumed by HOLIDAY-SIM, whose contact model represents colliders using sparse-grid SDFs (§ 3.1). Collider construction must preserve task-relevant geometry while controlling collision memory. Coarse SDFs can erase thin structures, introduce perforations, or fragment connected geometry, compromising downstream interactions. Excessively fine SDFs increase field storage and collision-buffer requirements, particularly across parallel simulation instances.

Collider construction seeks to minimize predicted collision memory while preserving task-relevant geometry. Let Π partition an asset’s parts into colliders, $\Delta _ { \mathrm { v o x } }$ denote their SDF voxel sizes, and $M _ { \mathrm { S D F } } ( \Pi , \pmb { \Delta } _ { \mathrm { v o x } } )$ estimate collision memory. We formulate the objective as

$$
\operatorname* { m i n } _ { ( \Pi , \Delta _ { \mathrm { v o x } } ) \in \mathcal { F } _ { \mathrm { S D F } } } M _ { \mathrm { S D F } } ( \Pi , \Delta _ { \mathrm { v o x } } ) .\tag{16}
$$

where $\mathcal { F } _ { \mathrm { S D F } }$ contains configurations that pass the collider checks (Table 9).

ERIS searches for feasible, low-memory representations through iterative validation. For a fixed collider layout, geometric constraints bound the admissible voxel sizes. ERIS selects a resolution for each collider, bakes its SDF in HOLIDAY-SIM, and checks the resulting collision geometry. Failed validation triggers a more conservative parameterization and rebaking until the checks pass or the admissible range is exhausted. For multipart assets,

Table 9. Checks of the collider verification stage.
<table><tr><td>Check</td><td>Measurement</td><td>Action on failure</td></tr><tr><td>Resolution admissibility</td><td>geometric voxel-size bounds</td><td>choose a finer admissible voxel size</td></tr><tr><td>SDF fidelity</td><td>surface, boundary, and topology checks</td><td>refine and rebake</td></tr><tr><td>Layout preservation</td><td>per-part wall and surface retention</td><td>keep failed part separate</td></tr></table>

ERIS also searches for lower-cost groupings of eligible static parts, rejecting merges that fail per-part geometry checks. Only when these searches fail does ERIS return the measured violations for source-level revision.

The three checks enter collider construction at different points (Table 9). Resolution admissibility is a pre-bake geometric check that constrains the initial SDF parameters using analytic bounds derived from collider scale, measured wall thickness, and declared functional features. SDF fidelity is a post-bake quantitative check that evaluates surface retention, boundary validity, and topology changes in the resulting HOLIDAY-SIM collider. Layout preservation is a merge-specific geometric check that measures per-part wall and surface retention after multiple source parts have been grouped into a shared collider. We provide further details on this stage in § C.2.

Articulation. ERIS uses the articulation intent derived from task constraints to guide the generation of joint declarations. ERIS verifies that the resulting joint declarations are valid and dynamically realizable. It checks the joint type, axis, pivot, and limits, and tests the corresponding revolute or prismatic motion for geometric binding. The articulated asset is then exercised in HOLIDAY-SIM. Driven joints run under their declared drives, and passive joints receive a controlled test input. ERIS verifies that each joint achieves a prescribed fraction of its expected motion without exceeding joint limits or binding.

Interaction. ERIS evaluates task-relevant physical interactions using the baked SDF colliders in HOLIDAY-SIM. Geometric clearance in the source meshes, for example, does not guarantee successful insertion after SDF conversion. Each simulation probe specifies the interacting objects, their initial configuration, and applied inputs. Generic probes test load support, transport, path following, and payload retention, while dedicated probes test interactions such as insertion, seating, and rolling. For insertion, a part is placed above its mating fixture and allowed to settle under gravity, after which its final pose is checked against prescribed seating tolerances. A constraint passes only when the measured outcome satisfies its acceptance criterion.

Branch-specific revision. ERIS revises failed assets through branch-specific operations guided by verification feedback. Let $X _ { t }$ denote the editable asset state at round t for generation branch b, including geometry and, where applicable, joint declarations and physical parameters. The revision step is

$$
\boldsymbol { X } _ { t + 1 } = \mathcal { F } _ { b } \left( \boldsymbol { X } _ { t } , \widehat { \mathcal { R } } _ { \mathrm { t a s k } } , \boldsymbol { \mathcal { C } } _ { t } ^ { - } , \boldsymbol { \mathcal { C } } _ { t } ^ { + } \right) ,\tag{17}
$$

where $\mathcal { F } _ { b }$ is the branch-specific revision function, $\mathcal { C } _ { t } ^ { - }$ contains measured violations and diagnostics, and $\mathcal { C } _ { t } ^ { + }$ records satisfied constraints to help prevent regressions.

For code-native assets, ERIS edits the executable construction program, including joint declarations and physical parameters where applicable. For mesh assets, correctable collider defects, such as thin walls, are repaired through surface offsetting, remeshing, and SDF rebaking. Conformance failures originating from image generation or 3D reconstruction, and collider failures without reliable local corrections, instead trigger regeneration.

## 4.1.4 Appearance randomization

Before finalizing an asset, ERIS can optionally apply appearance randomization to produce multiple visual variants while preserving the verified geometry, articulation, and collision state. The frozen geometry is UV-parameterized and packed into a shared atlas (Young, 2018), after which a texture-generation model (Bao et al., 2026) generates multiple texture variants on the same asset.

## 4.1.5 Experiments

We conduct five controlled studies: branch selection, verification, revision feedback, LLM backbones, and generation efficiency. Unless otherwise noted, the code-native generator uses claude-opus-5-5 as its LLM backbone at high reasoning effort, and ERIS runs for at most K = 4 rounds (§ C.3). The backbone study additionally evaluates claude-fable-5-1, gpt-6-astra, gpt-6.1-sol, and gpt-6-luna. Each comparison varies one factor while holding other applicable settings fixed.

Table 10. Branch selection under different downstream requirements. Mesh and Code-native entries report success rates over the evaluation unit shown in the second column. Selected generator denotes the generator to which ERIS routes the corresponding request.
<table><tr><td>Probe</td><td>Evaluation unit</td><td>Mesh</td><td>Code-native</td><td>Selected generator</td></tr><tr><td>Visual shape fidelity</td><td>VLM judgments</td><td>80%</td><td>0%</td><td>Mesh</td></tr><tr><td>Tote: vertical walls, no handles</td><td>geometric checks</td><td>80%</td><td>100%</td><td>Code</td></tr><tr><td>Tote: wall ≥ 8 mm</td><td>geometric checks</td><td>75%</td><td>100%</td><td>Code</td></tr><tr><td>Rack / peg: requested pocket count</td><td>structural checks</td><td>37.5%</td><td>100%</td><td>Code</td></tr><tr><td>Rack / peg: diameter within ±3 mm</td><td>geometric checks</td><td>0%</td><td>100%</td><td>Code</td></tr><tr><td>Rack / peg: all parts fully seated</td><td>functional checks</td><td>0%</td><td>100%</td><td>Code</td></tr></table>

Table 11. First-round verification failures among 72 assets. Unique rejections indicate failures at only one stage.  
Table 12. Post-bake SDF collider re-authoring. Re-authoring does not imply asset rejection.
<table><tr><td>Verification stage</td><td>Rejected</td><td>Unique</td></tr><tr><td>Any stage</td><td>34/72 (47%)</td><td></td></tr><tr><td>Conformance</td><td>28/72 (39%)</td><td>24/72 (33%)</td></tr><tr><td>Collider</td><td>10/72 (14%)</td><td>4/72 (6%)</td></tr><tr><td>Articulation</td><td>2/72 (3%)</td><td>0/72</td></tr><tr><td>Interaction</td><td>2/72 (3%)</td><td>0/72</td></tr></table>

<table><tr><td>Simulator-side intervention</td><td>Incidence</td></tr><tr><td>Colliders requiring re-authoring</td><td>346/1597 (22%)</td></tr><tr><td>Assets with re-authored colliders</td><td>39/72 (54%)</td></tr></table>

An asset succeeds by round t when all applicable executable checks have passed; required checks that cannot be evaluated count as failures. We report Success@t as the fraction of evaluation units passing verification by round t. For revision experiments, Recovery@t measures the fraction of units unsuccessful after round t that pass in round t + 1. Paired comparisons use matched evaluation units. Generation efficiency is measured by generation rounds and estimated API-equivalent model cost per solved asset and per recovered first-round failure. The verification study reports stage-wise rejection rates, unique rejections, and the frequency of post-bake collider re-authoring.

Branch selection. We compare mesh and code-native generation across visual, geometric, and functional requirements matched to their intended capabilities (Table 10).

For visual fidelity, each generator produces five assets from each of six categories with complex, non-primitive geometry, yielding 30 assets per generator. Both receive only the object category and its longest dimension. A blinded VLM judge (claude-fable-5-1) evaluates four-view renders using four labels: detailed and realistic, simplified, distorted, and cannot tell. Each asset is judged twice with reshuffled label order, yielding 60 judgments per generator. Separate tote and rack probes assess dimensions, part structure, mating geometry, and functional seating.

On these complex everyday objects, the mesh generator is rated detailed and realistic in 80% of judgments, compared with 0% for code-native generation, although objects from both generators remain recognizable. Conversely, code-native generation satisfies all tested dimensional, structural, and mating-geometry requirements, including functional seating. Across three end-to-end routing trials per request, ERIS consistently selects the generator matched to the specified requirements.

Verification. We evaluate first-round verification on 72 assets, including 32 from the revision feedback study and 40 from the claude-opus-5-5 backbone study. We examine failures across four stages: conformance, collider, articulation, and interaction. We also measure SDF collider re-authoring following simulator-side validation.

Verification rejects 47% of first-round assets, with conformance as the primary source of rejection (Table 11). Separately, 22% of SDF colliders require re-authoring after their initial bake (Table 12). These results distinguish verification

Table 13. Revision feedback ablation. All variants share the same first-round asset.
<table><tr><td>Metric</td><td>Independent regeneration feedback</td><td>Coarse</td><td>ERIS</td></tr><tr><td>Success@1</td><td>0.63</td><td>0.63</td><td>0.63</td></tr><tr><td>Recovery@1</td><td>0.33 (4/12)</td><td>0.17 (2/12)</td><td>0.92 (11/12)</td></tr><tr><td>Success@2</td><td>0.75</td><td>0.69</td><td>0.97</td></tr><tr><td>Recovery@2</td><td>0.38 (3/8)</td><td></td><td>0.40 (4/10) 1.00 (1/1)</td></tr><tr><td>Success@3</td><td>0.84</td><td>0.81</td><td>1.00</td></tr></table>

that rejects an asset from simulator-side validation that revises its collision representation before acceptance.

Revision feedback. We compare three revision strategies on 32 tasks with matched first-round assets. Independent regeneration discards the failed asset and starts anew. Coarse feedback retains the editable program but receives only a binary pass/fail signal. ERIS additionally uses measured constraint violations and diagnostics.

Of the 12 first-round failures, ERIS repairs 11 after one revision and the remaining one after two, reaching 100% final success (Table 13). Independent regeneration reaches 84% final success, while coarse feedback achieves 81%. Retaining the editable program without measured feedback therefore does not improve success over independent regeneration.

LLM backbones. To test whether the benefit of measurementgrounded revision generalizes across different LLM backbones, we compare ERIS with independent regeneration under the same verifier and round budget for each backbone (Table 14). We evaluate 20 tasks with two seeds each, for 40 paired units per backbone. Each pair shares the same first-round asset; after a failure, the raw baseline generates a fresh independent sample from the original request, whereas ERIS revises the existing asset using measured verification feedback. Across the evaluated LLM backbones, ERIS consistently improves Success@4 by 25–35 per-

Table 14. Revision across LLM backbones. All values report Success@4 with $K = 4$ rounds; ∆ is the absolute improvement over independent regeneration.
<table><tr><td>LLM backbone</td><td>Raw</td><td>ERIS</td><td>∆</td></tr><tr><td>claude-opus-5-5 (high)</td><td>0.65 (26/40)</td><td>1.00 (40/40)</td><td>+0.35</td></tr><tr><td>claude-fable-5-1</td><td>0.70 (28/40)</td><td>0.95 (38/40)</td><td>+0.25</td></tr><tr><td>gpt-6-astra</td><td>0.65 (26/40)</td><td>0.95 (38/40)</td><td>+0.30</td></tr><tr><td>gpt-6.1-sol</td><td>0.58 (23/40)</td><td>0.93 (37/40)</td><td>+0.35</td></tr><tr><td>gpt-6-luna</td><td>0.43 (17/40)</td><td>0.75 (30/40)</td><td>+0.33</td></tr></table>

centage points over independent regeneration. This consistent gain despite substantial differences in baseline success rates suggests that the benefit of measurement-grounded revision is not specific to a particular LLM backbone.

Generation efficiency. Closed-loop repair need not increase generation cost if measurements reduce the number of unsuccessful generation rounds. We compare ERIS with two baselines (Table 15). One-shot performs exactly one generation round per evaluation unit and makes no further attempt after a failure, while independent regeneration repeatedly resubmits the original request without using feedback from previous rounds. We evaluate both overall generation efficiency and the marginal efficiency of recovering failures after the first round. Compared with independent regeneration, ERIS achieves higher final success while requiring fewer generation rounds and lower model cost per solved asset, with larger efficiency gains when considering only the recovery of first-round failures.

Table 15. Generation efficiency. Overall measures are normalized by solved assets, and repair measures by recovered first-round failures. Costs are estimated API-equivalent model costs.
<table><tr><td></td><td colspan="3">Independent</td></tr><tr><td>Metric</td><td>One-shot</td><td>regeneration</td><td>ERIS</td></tr><tr><td>Success@4</td><td>18/40</td><td>26/40</td><td>40/40</td></tr><tr><td>Overall efficiency</td><td></td><td></td><td></td></tr><tr><td>Rounds / solved</td><td>2.22</td><td>3.81</td><td>1.70</td></tr><tr><td>Cost / solved</td><td>$2.59</td><td>$5.00</td><td>$2.48</td></tr><tr><td>Repair efficiency</td><td></td><td></td><td></td></tr><tr><td>Rounds / recovered</td><td></td><td>7.4</td><td>1.3</td></tr><tr><td>Cost / recovered</td><td>1</td><td>$10.43</td><td>$2.39</td></tr></table>

## 4.2 Scenario generation

We introduce a pipeline for generating task-conditioned scenarios and evaluate its scene-generation and background components. Building a scene manually requires placing the robot and objects and checking whether the configuration satisfies the task requirements. Repeating this process across the many variants needed for a benchmark is difficult to scale. Inverse scene generation (ISG) therefore uses staged verification and feedback to construct scene variants that satisfy task-relevant constraints.

ISG generates task-conditioned scene variants from a task T and its workplace W (§ 4.2.1). A separate pipeline adds photorealistic backgrounds (§ 4.2.2). A scenario-authoring agent can also write the process logic governing workplace operations (§ 4.2.3).

Figure 13 summarizes the scenario-generation pipeline, and Table 16 distinguishes its outputs and verification methods. We evaluate scene generation and background construction without a robot policy, and each experiment is reported with its component. For the background study, the agent generated the packaging scene from a supplied scene description and applied static verification. A person reviewed the result (Figure 17).

## 4.2.1 Inverse scene generation

Each generated scene variant must satisfy a set of hard constraints $\mathcal { C } _ { \mathrm { s c e n e } } ,$ , including catalog asset availability, noninterpenetrating placements, valid support relations, and the presence of task-required objects. The admissible set $\boldsymbol { S } _ { \mathrm { a d m } }$ consists of scene configurations satisfying every constraint in $\mathcal { C } _ { \mathrm { s c e n e } } .$

We use generate-then-verify as a baseline for scene-specification generation. Given a scene description, the agent generates a complete scene specification, which is resolved into scene variants and checked against the requirements. If the candidate fails verification, it is discarded and the agent starts a new attempt from the origina description, without access to the previous candidate or its failure feedback.

![](images/ec3459896091aa374c6f3d7a4921bd9ccb088babbfb4c73783b896d18b4ae9b6.jpg)  
Figure 13. Overview of scenario generation. Left, inverse scene generation revises the task plan and then the scene specification until each passes its checks and critics. Middle, a Gaussian-splat background is generated from a prompt and the verified work cell is placed on its free floor. Right, process logic keeps the work going across jobs without a reset.

Table 16. Components of scenario generation, their outputs, verification, and reported measures. Process logic is schema validated when loaded.
<table><tr><td>Component</td><td>Output</td><td>Verification</td><td>Reported measures</td></tr><tr><td>Task planning</td><td>Task plan, a predicate-based graph of work steps</td><td>Deterministic plan checks, then plan critic</td><td>Acceptance, cost</td></tr><tr><td>Scene generation</td><td>Scene specification and resolved variants</td><td>Deterministic scene checks, then scene critics</td><td>Acceptance, validity, fidelity, cost</td></tr><tr><td>Background generation</td><td>Panorama, Gaussian splat, work-cell placement</td><td>Image, view, and stability checks; model assessment</td><td>Rubric, stability</td></tr><tr><td>Process logic authoring</td><td>Logic with effects, events, queues, and idle joint targets</td><td>Schema validation when the scenario is loaded</td><td></td></tr></table>

In contrast, ISG generates a task plan and a scene specification sequentially, verifying each before proceeding to the next. The generation agent first summarizes the task steps, measurable success signals, required objects, spatial relations, and variation axes. The agent uses these preparatory descriptions to generate a task plan $z _ { 1 } = \Gamma$ and a scene specification $z _ { 2 } = \hat { S }$ . The task plan specifies symbolic work steps rather than robot motion trajectories. The scene specification defines the physical configurations needed by those steps and is resolved into scene variants $s _ { 1 : N _ { s } }$

At each stage, deterministic checks, which run as code without a language model, test the candidate. For the task plan, this is the plan check described below. For the scene specification, the checks verify required objects and relations and static validity across all resolved variants. Language models then serve as a plan critic or scene critics, corresponding to PLANCRITIC and SCENECRITICS in the algorithm. These critics run in separate calls after the deterministic checks pass and assess whether the candidate meets the task intent and the criteria for that stage. A candidate is accepted only when the checks pass and the critics approve it. Otherwise, the agent revises the candidate using the failed checks, measured violations where available, and the critics’ concerns.

The task, workplace, and preparatory descriptions remain fixed throughout revision. At each stage, revision retains the accepted outputs of earlier stages. Each stage has a fixed attempt budget. This use of feedback is related to iterative refinement with model feedback (Madaan et al., 2023) and coding agents that verify and roll back scene programs (Li et al., 2026).

Procedural generators sample layouts from hand-written rules or constraint programs (Deitke et al., 2022; Raistrick et al., 2024), language models map descriptions to layouts (Feng et al., 2023; Yang et al., 2024b), and recent agentic systems add simulation-ready assets and physical checks, sometimes conditioned on a task or language specification (Xia et al., 2026; Pfaff et al., 2026; Wang et al., 2026a). Task-generation systems propose a task and then build a scene for it (Wang et al., 2024; Hua et al., 2024).

Algorithm 1. Inverse scene generation   
Input: Task T of workplace W, catalog C, positive budgets $R _ { \mathrm { p l a n } } , R _ { \mathrm { s c e n e } } ,$ variants $N _ { s }$   
Output: Scene specification $\hat { S }$ with variants $\begin{array} { r } { s _ { 1 : N _ { s } } , } \end{array}$ or a typed stop   
Calls retain prior candidates and feedback in context; typed stops terminate the procedure.   
1: O ← OUTLINE(T, W); P ← SCENEPLAN(O)   
2: $\Gamma  \emptyset ; \quad \hat { S }  \emptyset ; \quad e  \emptyset$   
3: for $r = 1 , \ldots , R _ { \mathrm { p l a n } }$ do   
4: Γ ← TASKPLAN(P, Γ, e); e ← PLANCHECK(Γ, P) ▷ deterministic checks   
5: ${ \mathfrak { i } } e = \emptyset$ then e ← PLANCRITIC(Γ, P)   
6: if $e = \emptyset$ then break   
7: end if   
8: end for   
9: if $e \neq \emptyset$ then return BLOCKED   
10: end if   
11: for $r = 1 , \ldots , R _ { \mathrm { s c e n e } }$ do   
12: $\hat { S } \gets \mathsf { S P E C l F Y } ( P , \Gamma , \mathcal { C } , \hat { S } , e )$ ▷ asset selection and authoring   
13: $\left( s _ { 1 : N _ { s } } , e \right) \gets \mathtt { S c E N E C H E C K } ( \hat { S } , P , \Gamma , N _ { s } )$ ▷ static, resolve, variant checks   
14: if $e = \emptyset$ then $e  \mathsf { S C E }$ NECRITICS $( \hat { S } , s _ { 1 : N _ { s } } , \Gamma )$   
15: if e = ∅ then return $\hat { S } , s _ { 1 : N _ { s } }$   
16: end if   
17: ESCALATE(e) ▷ repair, restart with context, or stop   
18: end for   
19: return BUDGET

Algorithm 1 gives the procedure. The agent first attempts to fix the failure using the feedback. If the same failure recurs, the stage restarts in a fresh context. A third occurrence ends the run.

Task plan. Given a person-supplied task T and its workplace W (§ 2.1), the agent authors a task plan Γ as a directed graph of work steps. Nodes specify predicate preconditions and effects, and edges specify success, failure, or recovery transitions. Its initial predicate state is $\sigma _ { \mathrm { i n i t } }$ , its goal is $^ { g , }$ and its invariants are I. We denote the states reachable under the plan by $\mathrm { R e a c h } _ { \Gamma } ( \sigma _ { \mathrm { i n i t } } )$ and the objects and relations required by the plan by Req(Γ). A valid task plan requires that the goal does not already hold in the initial state, that at least one reachable state satisfies the goal, and that every reachable state preserves the invariants:

$$
\sigma _ { \mathrm { i n i t } } \not \vdash g , \qquad \exists \sigma \in \mathrm { R e a c h } _ { \Gamma } ( \sigma _ { \mathrm { i n i t } } ) : \sigma \vdash g , \qquad \forall \sigma \in \mathrm { R e a c h } _ { \Gamma } ( \sigma _ { \mathrm { i n i t } } ) : \sigma \vdash \mathbb { Z } .\tag{18}
$$

Before any geometry exists, the plan check explores up to $1 0 ^ { 4 }$ reachable predicate states to check goal reachability, invariant violations, dead ends, and cycle closure. It also checks representability in the scene specification. This bounded search does not establish the universal condition in Eq. (18) when the reachable state space is not fully explored. Each hard requirement is assigned to a deterministic check that must pass. A requirement that no check can verify, such as a hard visibility or reachability relation, is rejected at this stage. The plan critic evaluates the task plan in a fresh context and accepts it only with a score of at least 0.8 and no open concern. The agent revises the task plan until its deterministic checks pass and the plan critic approves ${ \dot { \mathsf { I } } } { \boldsymbol { \mathsf { I } } } ,$ or the budget is exhausted.

Scene specification. After task-plan acceptance, the agent selects catalog assets and authors a scene specification ${ \hat { S } } ,$ a typed representation of the workspace, objects and poses, optional background, and variation axes. If a required asset is missing from the catalog, scene generation stops unless the run allows the agent to generate new assets (Table 19). A compiler emits USD. The resolver $s _ { k } = { \mathsf { R E S O L V E } } ( { \hat { S } } , { \mathrm { s e e d } } _ { k } )$ samples only declared variations within the task plan’s pose bounds, producing the scene variants $s _ { 1 : N _ { s } }$ from one specification. Every resolved variant must contain the objects and relations required by the task plan and pass static verification. Static checks cover asset and reference validity, spawn collisions, support relations, and geometric tolerances. We allow at most 5 mm of overlap or ground penetration and 1 cm of support clearance. The resolved variants must also differ only along the declared variation axes. After the scene specification and its resolved variants pass all deterministic scene checks, the scene critics assess them against the task plan.

A scenario stores collision-buffer sizes for the contacts it can produce at once, and the batch size is chosen from the resulting memory estimate before allocation (§ A.1).

The following CNC request illustrates the task steps and success signals used to construct a task plan.

## Example task and generated steps

Task. Generate a long-horizon CNC tending workcell that repeatedly loads two shaft families, unloads finished parts, routes failed gauge results to quarantine, replenishes an empty infeed tray, and performs a fixture changeover between batches. Carry machine occupancy, tool-wear state, and completed-part count across cycles without resetting the whole cell.

Step load\_blank. Pick a shaft blank from the active infeed tray and load it into the lathe chuck. Success: blank seated in chuck within tolerance, door closed, machine occupancy=true.

Step gauge\_route. Gauge part and route to finished tray or quarantine bin. Success: passed part in finished tray and count incremented, or failed part in quarantine bin.

Step fixture\_changeover. Between batches, swap chuck fixture set for the other shaft family. Success: new fixture mounted, old fixture returned to rack, active family switched.

In this example, the task plan uses predicates such as chuck\_empty and carries tool wear and part counts across eight cycles.

Metrics. We evaluate generation without a robot policy using acceptance, static validity, specification fidelity, and cost.

1. Acceptance, the number of runs that reach each stage. A candidate is fully checked if it passes every deterministic check and accepted if the critics of its stage also approve it.

2. Static validity, the number of candidates whose specification passes static verification.

3. Specification fidelity, agreement with a reference specification on three tiers (two boxes and one item on a table, three items on each of three shelves, sixteen items in two trays), scored on nine criteria (Appendix D.1.2).

4. Cost, wall time and model calls per accepted scene, and mean wall time per run.

The specification-generation trials isolate scene-specification generation by starting from supplied scene descriptions and omitting task-plan generation (Table 31). Success in these trials means meeting the reference-based fidelity criteria. It is distinct from acceptance by the plan and scene critics in full ISG. When planners are compared, acceptance is also reported under a fixed critic, because a critic that shares the planner’s model can be lenient (Figure 15, Table 17).

Ablations of staged verification and feedback. We separately test the effects of task-plan verification in the full pipeline and feedback-based repair within scene-specification generation. The scene-only verification ablation still generates a task plan but omits its deterministic checks and the plan critic. It therefore differs from the specification-generation trials, which omit task-plan generation entirely. The generate-then-verify baseline independently resamples scene specifications without feedback from previous failures. Scene-only verification reaches a scene specification in all ten runs and three times faster. However, only 1 run yields a candidate that passes every deterministic check, against 5 with full ISG, and 8 of its 10 unchecked plans fail the plan check when tested afterwards (Figure 14, left). Within scene-specification generation, removing failure feedback reduces the number of successful trials. Generate-then-verify solves 3 and 2 of 9 trials within five attempts with gpt-6-astra and gpt-6.1-sol, against 9 and 7 with feedback (Figure 14, right).

Ablation over frontier models. Scene acceptance depends substantially on the model critic. Under a shared claude-opus-5-5 critic, gpt-6.1-sol and claude-sonnet-5-5 achieve identical acceptance rates despite substantial differences under their own critics (Table 17). Re-evaluating the same candidates reveals disagreements over stand-in assets, primitive fixtures, and support gaps (Figure 15). Consequently, the cost per accepted scene also depends on critic selection (Table 32). In separate specification-generation trials, verification feedback increases success from 4/54 first attempts to 40/54 within five attempts (Table 18). However, gemini-4-argon fails the easiest tier on object positioning. These experiments use ten tasks from a separate 30-task industrial corpus (§ D.1.2), rather than the seven benchmark tasks (Table 22).

Bottlenecks. Missing assets and limited scene-specification expressivity constrain inverse scene generation. With gpt-6.1-sol, 8/10 runs yield accepted plans, but only 1/10 produces an accepted scene (Table 18). Increasing the generation budget does not resolve asset availability, which blocks most runs (Table 19).

![](images/af19e313d7a5ee36f895fa1866f6376ad1c9239b28b3b973afe3fbb75dda501a.jpg)  
Figure 14. Ablations of staged verification and feedback. Left, full ISG with gpt-6.1-sol on the ten tasks of Table 18, with and without the plan check and plan critic. Valid plan applies the plan check retrospectively when it was skipped. Right, specification-generation trials solved within five attempts, with and without feedback.

![](images/32bcd1fe961d83b8362355e57838039bdad9ccbd45e9b8b07448f15fd7de4ce1.jpg)  
Figure 15. Accepted scenes when the fully checked candidates of Table 18 are re-judged by a fixed critic. Parentheses give the number of such candidates, Own is the planner’s own critic, and color shows the accepted fraction.

Table 17. Planning with a critic shared by both planners. Each row runs the planner on the ten tasks of Table 18 with the same budgets, and Critic names the model that serves every plan and scene critic call. Rows with the planner’s own critic repeat Table 18. Min is the mean wall time per run in minutes.
<table><tr><td>Planner</td><td>Critic</td><td>Plan</td><td>Scene spec.</td><td>Checks</td><td>Scene</td><td>Min</td></tr><tr><td>gpt-6.1-sol</td><td>Own</td><td>8/10</td><td>8/10</td><td>5/10</td><td>1/10</td><td>27</td></tr><tr><td>gpt-6.1-sol</td><td>claude-opus-5-5</td><td>7/10</td><td>7/10</td><td>4/10</td><td>3/10</td><td>28</td></tr><tr><td>claude-sonnet-5-5</td><td>Own</td><td>10/10</td><td>10/10</td><td>10/10</td><td>7/10</td><td>12</td></tr><tr><td>claude-sonnet-5-5</td><td>claude-opus-5-5</td><td>9/10</td><td>9/10</td><td>9/10</td><td>3/10</td><td>22</td></tr></table>

Accepted scenes often rely on primitive substitutes for missing fixtures (Figure 16). Because scene critics inspect catalog descriptions rather than renders, they can overlook mismatches such as a vise-like fixture represented by a teapot. Explicitly specifying the required affordances of generated assets enables one additional accepted scene (Figure 16). The remaining failures expose requirements that the current scene specification cannot express (Table 19).

Rubric judges score the candidates after generation, separately from the plan and scene critics. Asset fit largely drives their scores, which cluster in a narrow range (§ D.6).

Threats to validity. Specification fidelity is measured against one reference per tier derived from the evaluation rubric and does not establish broader scene quality. In Table 18, each planner uses critics from its own model, so scene acceptance reflects both generation quality and critic decisions. The critic-swap and shared-critic experiments partially separate these effects (Figure 15, Table 17), although the shared-critic comparison uses only one run per task and model. The model families also differ in their agent interfaces. Without an independent human audit, critic agreement does not establish the suitability of accepted scenes for the requested tasks. Finally, plan checks explore at most $1 0 ^ { 4 }$ predicate states, and static geometric checks are approximate.

## 4.2.2 Photorealistic background

Background generation runs separately from ISG. After scene verification, we place each simulated cell inside generated 3D surroundings rendered as Gaussian splats. They form the background of a scene variant and therefore enter only its observation model $O _ { s }$ (§ 2.1). Before a background is added, the scene contains task-relevant geometry but leaves the surrounding space empty. Real-to-sim methods reconstruct each scene from captured data (Torne et al., 2024; Han et al., 2026; Jiang et al., 2026; Wang et al., 2026b), which for a facility requires physical access and a capture session. We therefore generate the surroundings with 3D world generators (World Labs, 2025; HunyuanWorld Team, 2025; Yang et al., 2025) and render them as 3D Gaussian splats (Kerbl et al., 2023), as in simulators that pair splat rendering with rigid-body physics (Qureshi et al., 2025; Jia et al., 2026; Kim et al., 2026). Unlike systems that align a collision mesh to the splat (Kim et al., 2026), the generated background is used for rendering only. A generated interior is not a measured facility, even when the

Table 18. Frontier-model ablation for scene generation. Planning counts, over ten tasks, runs with an accepted plan, runs with at least one candidate that passes every deterministic check, and runs that end in an accepted scene, judged by critics of the planner’s model, and the mean wall time per run in minutes. Wall times include provider latency.
<table><tr><td rowspan="2">Model (effort)</td><td colspan="5">Specification generation</td><td colspan="4">Full ISG</td></tr><tr><td>Easy</td><td>Medium</td><td>Hard</td><td>1st</td><td>Attempts</td><td>Plan</td><td>Checks</td><td>Scene</td><td>Min</td></tr><tr><td>gpt-6-astra (high)</td><td>3/3</td><td>3/3</td><td>3/3</td><td>2/9</td><td>2.44</td><td>5/10</td><td>0/10</td><td>0/10</td><td>29</td></tr><tr><td>gpt-6.1-sol (medium)</td><td>3/3</td><td>3/3</td><td>1/3</td><td>0/9</td><td>2.57</td><td>8/10</td><td>5/10</td><td>1/10</td><td>27</td></tr><tr><td>gpt-6-luna (medium)</td><td>2/3</td><td>0/3</td><td>1/3</td><td>0/9</td><td>3.33</td><td>3/10</td><td>2/10</td><td>0/10</td><td>13</td></tr><tr><td>claude-opus-5-5 (high)</td><td>1/3</td><td>3/3</td><td>3/3</td><td>0/9</td><td>3.29</td><td>9/10</td><td>8/10</td><td>2/10</td><td>29</td></tr><tr><td>claude-sonnet-5-5 (medium)</td><td>3/3</td><td>2/3</td><td>3/3</td><td>2/9</td><td>3.00</td><td>10/10</td><td>10/10</td><td>7/10</td><td>12</td></tr><tr><td>gemini-4-argon (high)</td><td>0/3</td><td>3/3</td><td>3/3</td><td>0/9</td><td>2.67</td><td>10/10</td><td>8/10</td><td>6/10</td><td>22</td></tr></table>

![](images/ec0d551098d32198ffc97527ee628eac2d8fda01d427cb2f0213b15c5d504ef2.jpg)  
(a) CNC tending, variant 1

![](images/5c61f5cf2b698e9acb6488d2445b5b256d35434b1efd5387e8b89f1dbcb62181.jpg)  
(b) CNC tending, variant 2

![](images/42511a78a7818d0b5634dc45401240548f08a8e0be9912706cb244c512205ec5.jpg)  
(c) CNC cell detail

![](images/4df08634b4462a9e422bb7fb4d12cd07c59e16831c661a156658009247faa968.jpg)  
(d) Secondary packaging

![](images/f944800f53d99c4730be80f3fc7d78d56e9461aa22dad1daa89f01cd0750c2e9.jpg)  
(e) Adhesive application

![](images/c64cabf7a53190f4d679c873b5665c6748ca7019cb5f55db1b0e01ab822b6125.jpg)  
(f) Go/no-go gauging

![](images/2f1cdb4abed4dbdf3ef4a296107ec9b77fd8fd75991e9c35877ddb48fe40429c.jpg)  
(g) CNC chuck exchange

![](images/32376cf430a7e25523a2785888968f19d36a5ef837d91fa7400458bba8e55cde.jpg)  
(h) Generated billet  
Figure 16. Scenes accepted by inverse scene generation under each planner’s own critics. (a–c) CNC tending with two scene variants and a six-step plan generated by claude-sonnet-5-5. (d) Secondary packaging with a six-step plan generated by claude-sonnet-5-5. (e, f) Scenes generated by gpt-6.1-sol. (g, h) CNC chuck exchange with an asset generated during scene construction (Table 19). A stricter critic rejects (a–d) (Figure 15). All panels show static renders at t = 0 with generated Gaussian-splat backgrounds (§ 4.2.2) and manually placed FRIDAY.

## generator is conditioned on a reference photograph.

Background generation. Prompts are sampled from eight facility presets, such as electronics, food, automotive, and logistics plants, over 17 design factors such as ceiling, lighting, racks, and density (Table 34). They require free floor space for the separately simulated work cell, since a plausible hall can still leave no room for it (§ D.2, Figure 23). We assess the panorama before reconstructing it in 3D, as DreamScene360 does during iterative visual assessment and prompt refinement (Zhou et al., 2024). Panorama-to-3D reconstruction produces a Gaussian splat (Zhou et al., 2024; Xiong et al., 2025) through one of the following generator paths. The proprietary path sends a panorama or reference image to a hosted world-generation service (World Labs, 2025) and returns a splat. The open-source path chains an image-to-panorama diffusion model (Yang et al., 2025), a multi-view generator that renders camera paths through the panorama, and 3DGS optimization on those views (Team HY-World et al., 2026), in the spirit of PanoDreamer (Xiong et al., 2025). A lighter variant lifts the panorama with monocular panoramic depth (Jiang et al., 2025), as DreamScene360 does with a different depth model (Zhou et al., 2024). The provider records little beyond model and seed, whereas the open-source chain records every seed, checkpoint, and rejected view. Both hallucinate geometry the panorama does not show.

We estimate a six-plane room enclosure to remove out-of-bounds Gaussians, align the floor and walls, and set metric scale. We then search background transforms that place the work-cell footprint on free floor, ranking them by minimum clearance with a safety margin while keeping the simulated scene coordinates fixed. Distinct translations and yaw angles provide placements for scenario variants. Appendix D.2 gives the procedure and parameters, with an example in Figure 24.

Verification. Verification first checks the panorama and background views, then evaluates the background with the work cell in place. The checks cover alignment and scale, footprint clearance, and overlap with task-relevant geometry. A recorded rollout is re-rendered with the background to check missing-background coverage and the temporal stability of fixed regions along the camera trajectories. Where a stage uses a VLM judge, deterministic checks run first and every judged criterion must pass (Table 38). Generation and diagnostic renders can be inspected before qualification. Installation requires both panorama and background-view QA to pass, followed by qualification on integrated scene views and rollout videos. Skipping VLM evaluation leaves the result unverified and does not permit installation. We re-render recorded trajectories under different backgrounds without rerunning the policy, processing variants in chunks to limit peak memory usage (§ D.3).

Table 19. Removing bottlenecks of inverse scene generation one at a time. Each row runs gpt-6.1-sol on the same ten industrial tasks and counts runs with an accepted plan, runs that reach a scene specification, runs with at least one candidate that passes every deterministic check, and runs that end in an accepted scene.
<table><tr><td>Condition</td><td>Plan</td><td>Scene spec.</td><td>Checks</td><td>Scene</td></tr><tr><td>Default budget (40 min, 40 calls)</td><td>8/10</td><td>8/10</td><td>5/10</td><td>1/10</td></tr><tr><td>+ Larger budget (2 h, 80 calls)</td><td>9/10</td><td>9/10</td><td>2/10</td><td>1/10</td></tr><tr><td>+ Asset generation</td><td>9/10</td><td>9/10</td><td>2/10</td><td>0/10</td></tr><tr><td>+ Affordance format in prompt</td><td>9/10</td><td>9/10</td><td>3/10</td><td>2/10</td></tr></table>

Results. We evaluate the backgrounds using the packaging scene generated with human review (Figure 17). A blind VLM judge compares the proprietary and open-source backgrounds on nine matched frames (§ D.2, Table 36). It prefers the proprietary splat in all 18 pairwise comparisons and scores it 2.85 against 1.78 on a five-point rubric. The judge rates neither background fully usable, and the comparison rests on one reference image and one judge model. These post-hoc rubric judgments are separate from the pipeline verification gates. The preference results show a relative advantage for the proprietary background in this comparison, but do not establish that either background is ready for use. Retention in the background catalog does not change that assessment.

We automatically place the packaging cell in the free floor space. Figure 17 shows the cell in nine of the eleven generated backgrounds.

## 4.2.3 Process logic generation

Process logic ℓ governs workplace operations within a scenario $S = ( s _ { 1 : N _ { s } } , \ell )$ , including equipment behavior and work supply during execution. The scenario-authoring agent can specify ℓ using four typed components: effects, events, queues, and idle targets (§ D.4). Of the seven benchmark tasks, conveyor loading uses one belt and one queue, flow-rack loading three belts and one queue, press tending five events, and barcode routing one belt and two events.

Schema validation checks the representation when a scenario is loaded. Finalization additionally checks event construction, queue configuration, and effect-parameter bindings for each scene variant, rejecting unsupported effect types. These checks establish structural validity but not whether the logic implements the intended industrial process.

## 4.3 Evaluation generation

From the same task that drives scenario generation, a coding agent also generates the evaluation $E =$ $( j , m _ { 1 : N _ { m } } , c )$ , so that each world $\xi ~ = ~ ( S , E )$ of Eq. (2) is produced as a whole. We split this generation into two stages (Figure 18). The agent proposes an interpretation of the task that the person who wrote it confirms, and a compiler with no model in the loop turns the confirmed proposal into executable checks that are tested against the scenario before the world enters the benchmark.

The agent composes E from the predicates and metrics of § 5.2 and cannot add new ones.

A generated reward can be checked against a separately defined performance metric (Ma et al., 2024; Xie et al., 2024b; Yu et al., 2023). A generated evaluation defines how performance is assessed, so it can compile and run while applying criteria that do not match the intended task.

We distinguish two outcomes. A generated evaluation passes verification when it loads and composes as an executable evaluation and then passes the three checks of § 4.3.2 against its scenario. Its verdicts agree with the intended task when they match labeled recordings, which verification cannot show. § 4.3.3 measures both for evaluations generated with and without questions.

![](images/a76c3ce964471e7f0d6e3d34b07f73718dc59be97fa6f359e002457832d604b5.jpg)

![](images/8ad3fa911c208d9680cc8bfb7e95b8b6553a79690130f1544ba5984b4ec63206.jpg)  
(a)  
(b)

![](images/43e304a2a11bf4f43ca2095acc42bf19eb02db26af2c7c4d4e5045b1d896bc93.jpg)

![](images/2ad5ccf6424bfd7a29dae4b22f81ebb1dfe50fd48edc2637f1b96bfcd2aea9bf.jpg)

![](images/ebf5d94ba9398b5df32c6dcbd6d3c8786ffe41fafbdc8dcbe5ba661bb149db96.jpg)

(c)  
![](images/5e645e3f0edbdc4edb942837f4357d2645ca2c41eebd5f3a2588e171224ad801.jpg)

(d)  
![](images/e467a6ed9f2c7383dc09f0b25ccbc15f92e02ac725cf57c268913c9ac33485d0.jpg)

(e)  
![](images/45f022a6b6f612868a772d566398f81d7170395b0d54d7df26aff81a1d78f704.jpg)  
(g)  
(h)

(f)  
![](images/a1558e54ded62437d4fae395204ca400b6ed74becb38c05939253844544bf8a8.jpg)  
(i)  
Figure 17. One cell under nine backgrounds, nine scene variants with identical dynamics. The packaging cell is placed by clearance search (§ D.2) in nine generated Gaussian-splat warehouses. Geometry, assets, robot pose, and camera are identical across panels, so the variants share $F _ { s , \ell }$ and differ only in $O _ { s }$ (Eq. (1)).

## 4.3.1 Proposal and confirmation

The agent first reads the task and the scenario and asks about what they leave undecided. A question is allowed only if its answer changes what is measured or what counts as passing, and only if the scenario does not already answer it. Typical questions concern what completes a job, which tolerance applies, which outcomes count as failure, how long a job may take, and whether the work repeats. The agent then reviews the answers and may ask follow-up questions.

The agent proposes an evaluation consisting of a job, its metrics, and a rubric. The rubric specifies when a run ends and what it reports. It may also define a success criterion.

Tasks within a workplace can share a job definition and metrics while using different rubrics. At the inbound conveyor station, single tote transfer specifies a run-level success criterion, whereas conveyor loading reports metrics without assigning a verdict. Both count jobs consistently, allowing comparison across runs using their metric distributions. Table 20 illustrates the questions, answers, and evaluation proposals for both tasks.

The evaluation reads the full privileged state, but a task can require more than that state expresses. Each condition drawn from the task is therefore labeled by what can judge it. It is physics when a predicate on privileged state measures it, visual when only a camera can show it, and unmeasured when neither can. These labels show the person where the evaluation falls short of the task, instead of hiding the gap behind a plausible check. The proposal also marks which names of the scenario the policy may be given. Names that appear in the task, such as a tote, are visible to the policy, and all others, such as a target region, are reserved for the evaluator.

The person confirms the proposal as a whole for one scenario variant, and a change to the scenario requires a new confirmation.

## 4.3.2 Compilation and verification

A deterministic compiler converts the confirmed proposal into an executable evaluation without model calls or simulator rollouts. It resolves scenario references, instantiates supported predicates and events, binds them to job start, completion, and failure conditions, and assembles the specified metrics and rubric. The confirmed proposal and model outputs are retained for recompilation. Only simulator-state predicates can determine success. Visua requirements produce event-bound evidence records for post-run assessment (§ F.4), but cannot enter success criteria. Unsupported or unmeasured requirements block compilation rather than being replaced by proxies (§ E.1).

![](images/9c72de73fa6626000e4f6af233ade97a2c4fd15ab3398c438d94dcce34fd22e8.jpg)  
Figure 18. Evaluation generation. The agent asks what the task and scenario leave undecided and proposes a job, metrics, and a rubric, which the person who wrote the task confirms. A compiler without a model turns the confirmed proposal into an evaluation, rejects what it cannot measure, and checks the result against the scenario.

Table 20. Evaluation generation for two tasks at the inbound conveyor station. Both tasks share a job and metrics but use different rubrics. Answers are italicized.
<table><tr><td>Step</td><td>Content</td></tr><tr><td>Task 1 Task 2</td><td>Single tote transfer: “Pick up the tote presented on the pallet and place it upright on the running conveyor.&quot; Conveyor loading: “Pick up each tote presented on the pallet and place it upright on the running conveyor.&quot;</td></tr><tr><td>Question 1</td><td>Does a job complete upon placement on the belt or entry into the X-ray inlet? Entry into the inlet.</td></tr><tr><td>Question 2</td><td>What tilt is allowed?At most 10°. When does a run end? After one job or 1 min for Task 1, and after 5 min for Task 2.</td></tr><tr><td>Question 3</td><td>Starts when a tote is presented in the pick region, requires upright placement supported by the conveyor, and</td></tr><tr><td>Shared job</td><td>completes upon entry into the X-ray inlet. All conditions use simulator state.</td></tr><tr><td>Shared metrics</td><td>Completed and failed jobs, throughput, latency, stalls, protocol violations, upright placements, and inlet entries.</td></tr><tr><td>Rubric 1</td><td>Ends after one job or 1 min. Passes with one completed job and no protocol violations.</td></tr><tr><td>Rubric 2</td><td>Runs for 5 min without a success criterion. Reports completed jobs and throughput.</td></tr></table>

Compiled evaluations undergo three checks against their scenarios without requiring a policy or human intervention (Table 21). Prerequisite guards check loading, scenario references, and time limits. The static check examines the executable definitions, including those edited by hand. The reset check tests initial job conditions over 4 s of uncontrolled physics in each scene variant. The idle check runs the evaluation without robot control to detect jobs completed by the scenario alone. For example, widening the conveyor-loading start region produces four completed jobs in an idle run, which the other checks do not reject. However, all three checks can accept an incorrect upright tolerance because they do not assess task intent (§ E.2).

Failures return diagnostic feedback to the agent for revision, with at most three candidates including the initia attempt. In question-guided generation, the agent revises the proposal for renewed human confirmation, or simulated-user confirmation in our study, while retaining previous answers. In direct generation, it revises the executable evaluation. Because the loop receives no labeled trajectories or verdict feedback, passing verification does not establish agreement with task intent. Unresolved failures terminate the loop without an accepted evaluation.

The three checks establish consistency with the scenario while they do not determine whether an evaluation captures task intent. The following study assesses this agreement using labeled recordings (§ 6.5).

## 4.3.3 Questions and verification feedback

We evaluate single tote transfer (Table 20) under four conditions, comparing direct evaluation generation with question-guided generation, each with and without verification feedback. In question-guided generation, a simulated user answers clarification questions and confirms proposals using a fixed task-intent specification. Four models perform three trials per condition, with up to three candidates when feedback is enabled. Evaluations are assessed against 15 labeled recordings withheld from generation and revision (§ E.3).

Table 21. Three policy-independent checks for compiled evaluations. Reset and idle checks run in every scene variant without robot control.
<table><tr><td>Check</td><td>Execution</td><td>Rejects when</td><td>Does not detect</td></tr><tr><td>Static</td><td>No simulation</td><td>Invalid definitions, unjustified event rules, or time limits beyond the run</td><td>Incorrect regions or tolerances</td></tr><tr><td>Reset</td><td>4 s after reset</td><td>Expected job units fail to satisfy or maintain their start conditions</td><td>Overly broad start regions or later state changes</td></tr><tr><td>Idle</td><td>Rollout to termination</td><td>Jobs complete without robot action, the run passes, or start events recur during open jobs</td><td>Errors requiring policy actions or task-intent labels</td></tr></table>

![](images/124383a752ba08f753f92e07b91da9e320ecc119ecc1b9218c054c90d2d0db5b.jpg)

![](images/57e4ce7225f98d1e1b7d768d714a12e5af126d83d4867195aa130281682673e5.jpg)

<table><tr><td rowspan=1 colspan=4>(c) Failures judged successful (FP)</td></tr><tr><td rowspan=1 colspan=1>3/33</td><td rowspan=1 colspan=1>3/33</td><td rowspan=1 colspan=1>0/33</td><td rowspan=1 colspan=1>0/33</td></tr><tr><td rowspan=1 colspan=1>3/33</td><td rowspan=1 colspan=1>3/33</td><td rowspan=1 colspan=1>0/33</td><td rowspan=1 colspan=1>0/33</td></tr><tr><td rowspan=1 colspan=1>3/33</td><td rowspan=1 colspan=1>3/33</td><td rowspan=1 colspan=1>0/33</td><td rowspan=1 colspan=1>0/33</td></tr><tr><td rowspan=1 colspan=1>0/33</td><td rowspan=1 colspan=1>0/33</td><td rowspan=1 colspan=1>0/11</td><td rowspan=1 colspan=1>0/22</td></tr><tr><td rowspan=1 colspan=1>withoutfeedback f</td><td rowspan=1 colspan=1>witheedback f</td><td rowspan=1 colspan=2>without  witheedback feedback</td></tr><tr><td rowspan=1 colspan=4>Direct         Questions</td></tr></table>

Figure 19. Evaluations generated for single tote transfer, three trials per model and condition. (a) Trials whose evaluation passes the guards and the three checks on the first scene variant. (b) Successful recordings judged failed, out of four per executable evaluation. (c) Failed recordings judged successful, out of eleven per executable evaluation. Panels (b) and (c) include evaluations that verification rejects.

Verification feedback increases direct evaluation acceptance from 6/12 to 12/12 but does not correct any of the 21 misclassified recording pairs (Figure 19). These errors arise from ambiguous completion criteria and additional conditions imposed by the models. Question-guided generation recovers the intended completion event and uprightness tolerance, achieving agreement with all assessed labels. It produces 10/12 accepted evaluations without feedback and 11/12 with feedback.

These results distinguish executable correctness from agreement with task intent. However, the study covers one task, one scene variant, and 15 reused recordings without isolated violations of individual requirements. The simulated user does not establish the reliability of human confirmation, and question-guided generation changes elicitation, review, and compilation together, preventing attribution of the improvement to any single component.

## 5 24/7-BENCH

## 5.1 Tasks

The seven tasks cover the industrial work that robots already do and that employs many people (Table 22). BEHAVIOR-1K chooses its household activities by surveying people (Li et al., 2024a), but no such survey exists for factory work, so the selection follows robot deployment and employment. Handling and assembling are among the largest applications of industrial robots (International Federation of Robotics, 2024), the matching occupations employ millions of people in each major robot market (§ F.1) and are highly exposed to automation (Frey and Osborne, 2017), and robotics research targets the same work (Pantano et al., 2022; Correll et al., 2018; Nationa Institute of Standards and Technology, 2026).

Conveyor loading and flow-rack loading are repeated pick and place, and the other tasks add whole-body pushing of a loaded cart, filling and exchange of a fixture, a press started through a two-hand control (International Organization for Standardization, 2019), process equipment that the robot operates and waits on, as mobile robots do in chemistry laboratories (Burger et al., 2020), and routing by a label that the robot reads. Welding, dispensing, and processing are established robot applications (International Federation of Robotics, 2024) but are left for a future benchmark. The generators build further workplaces and tasks, such as the CNC tending cell of § 4.2.1, and these sources decide which tasks enter the benchmark.

Table 22. The seven tasks in 24/7-BENCH. A job is the unit of work that the evaluation counts, and the carried state is what earlier jobs leave for later ones. Robot application is the matching class of industrial robot applications (Internationa Federation of Robotics, 2020), and occupation is the matching group of workers. The task instructions are in § F.3.
<table><tr><td>Family</td><td>Task</td><td>One job</td><td>Carried state</td><td>Robot application</td><td>Occupation</td></tr><tr><td rowspan="3">Material flow</td><td>Conveyor loading</td><td>Tote, pallet → belt</td><td>Totes on pallet, belt</td><td>Picking, placing</td><td>Material movers</td></tr><tr><td>Flow-rack loading</td><td>Container, stack → lane</td><td>Filled lanes, stack</td><td>Picking, placing</td><td>Material movers</td></tr><tr><td>Cart push</td><td>Carton, table → dock Part in stored jig;</td><td>Cart loads, docks</td><td>Transport†</td><td>Material movers</td></tr><tr><td>Assembly</td><td>Jig loading</td><td>empty jig seated Stamped part</td><td>Filled jigs</td><td>Assembling</td><td>Assemblers</td></tr><tr><td>Machine tending</td><td>Press tending</td><td>on outfeed</td><td>Press, stamped parts</td><td>Press tending</td><td>Press operators</td></tr><tr><td>Process operation</td><td>Valve operation</td><td>Beaker, spout → output</td><td>Valve, filled beakers</td><td>一</td><td>Plant operators</td></tr><tr><td>Inspection</td><td>Barcode routing</td><td>Parcel → coded tote</td><td>Routed parcels</td><td>Inspection, triage</td><td>Sorters</td></tr></table>

<sup>†</sup>Service robots for transport and logistics (International Federation of Robotics, 2026c).

![](images/886ce218d2c1572fba1f15904e188e0788ccfe9af6249e1e053c6e62c4909eda.jpg)  
(a) Conveyor loading

![](images/ac58d5ebc3a02659cf9f000a9d8220c3bc43234422f0ee8feff08b1d3337ddef.jpg)  
(b) Flow-rack loading

![](images/77c069b52b9c65b2eee883c00522588841351eb9c51874a3a651125f904dc54d.jpg)  
(c) Cart push

![](images/fa10c03cdbd80545f6c4288689990f2081708b7800eb91214be4fee10d45b0f2.jpg)  
(d) Jig loading

![](images/4ad1ddae6a0c81619701e19fad86d37ee2d12375114d0ee3f2ad4208b13f6de8.jpg)  
(e) Press tending

![](images/158d81aacbdc05d9bc2978d3594cc6311e98876927c3aaeea5a4dfd57fd2018f.jpg)  
(f) Valve operation

![](images/af3aa2a30ee6c80c6db01a2cec4e7dd978a1c3b6379521b041937e90575c98dc.jpg)  
(g) Barcode routing  
Figure 20. The seven benchmark tasks in photoreal factory halls with FRIDAY. (a–c) Material flow. (d) Assembly. (e) Machine tending. (f) Process operation. (g) Inspection.

Every task runs on persistent state. No scenario restores its scene between jobs, and finished units stay in it unless the process logic removes them. Process logic acts in four tasks: belts move units in conveyor loading, flow-rack loading, and barcode routing, queues return or restock units in the first two, and events drive the press in press tending and stop and restart the belt in barcode routing, while cart push, jig loading, and valve operation start from a fixed set of units. A run has a maximum duration $T _ { \mathrm { r u n } }$ and may end earlier under its rubric, for example when a target number of jobs is completed or an inactivity limit is reached. Completed-job counts and elapsed time are measured outcomes. Evaluation uses the resulting trajectory regardless of the policy’s internal structure.

The scenario representation supports scene variants that differ in the mass, placement, or appearance of their units and fixtures (§ 4.2). The scenes are generated from each task and its workplace in photorealistic factory halls (Figure 20), those of cart push, jig loading, and barcode routing by inverse scene generation (§ 4.2.1). Process logic specifies equipment behavior and unit supply (§ 4.2.3).

## 5.2 Evaluation

Policies act on observations $o _ { t }$ , whereas 24/7-BENCH determines job outcomes and any run-level verdicts from privileged simulator state $x _ { t }$ . Observation-based assessments, such as visual judgments, provide supplementary evidence but do not affect the current benchmark outcomes. An evaluation $E = ( j , m _ { 1 : N _ { m } } , c )$ defines the job j by events on privileged state, the metrics $m _ { 1 : N _ { m } }$ aggregated from job records, and the rubric c, which sets when a run ends, what it reports, and, optionally, which outcomes count as success (Table 1). Evaluations of the same task can share a job and metrics while differing in the rubric, for example one that requires a single transfer within a time limit and one without a success criterion that counts the transfers of a fixed interval.

(a) Run record  
![](images/f5f94fc961f4bdcdd2b8393709ac214d3ecb27728c0c4b34c8ecc462fc0848c1.jpg)

![](images/4401dec2e569d110c79e672da77155411dbec352adb5fcb27dfd34b5e28dd2b1.jpg)

![](images/7b67509b290043efb6f9b0e66c7d0269c3094d15c1d26dd1e2e0d7b19150afb1.jpg)

![](images/4b7c7bbd8b44608826426881ac60176b1706c44d7c4c23728db8ab4adc113b7d.jpg)  
Figure 21. Run evidence and variant-based scorecard.(Left) Press-cycle record with state-based events at 60 Hz, showing a cycle-level verdict rather than the full outfeed-task outcome. (Right) Schematic scorecard with aggregate metrics, per-variant results, and condition-axis summaries. O, S, P, and B denote object, spatial, physics, and background conditions.

Table 23. Job events over the seven tasks.
<table><tr><td>Task</td><td>Start</td><td>Required</td><td>Completion</td></tr><tr><td>Conveyor loading</td><td>On pallet</td><td>Upright on belt</td><td>Entered tunnel</td></tr><tr><td>Flow-rack loading</td><td>In stack</td><td>In aisle</td><td>Upright in lane</td></tr><tr><td>Cart push</td><td>Staged</td><td>Left table, crossed aisle</td><td>In dock, loading deck clear</td></tr><tr><td>Jig loading</td><td>In tray</td><td>Left tray, in bench jig</td><td>Jig stored, empty jig seated</td></tr><tr><td>Press tending</td><td>On infeed</td><td>Nested, two-hand, stamped</td><td>Part on outfeed</td></tr><tr><td>Valve operation</td><td>On input side</td><td>Filled</td><td>On output side, valve shut</td></tr><tr><td>Barcode routing</td><td>Scanned</td><td>Off belt end</td><td>In destination tote</td></tr></table>

## 5.2.1 Jobs

A job is an occurrence of work. In conveyor loading, a tote that arrives on the pallet starts a job, which completes when the tote, upright on the belt, enters the X-ray tunnel, and because scanned totes return to the pallet, the same tote takes part in successive jobs. Jobs are delimited by events on privileged state. A predicate is a true-or-false test of the instantaneous state, such as a unit inside a region, upright within a tilt bound, at rest, or supported by another body, or a joint within a range (Table 48), and its truth value is determined by thresholding a satisfaction margin at zero, following the robustness semantics of temporal logic (Fainekos and Pappas, 2009; Donzé and Maler, 2010). An event occurs when a predicate has held for $K _ { \mathrm { h o l d } }$ consecutive steps, the bounded always operator of signal temporal logic (Maler and Nickovic, 2004), so that contact noise does not trigger it,

$$
r _ { t } = b _ { t } \big ( r _ { t - 1 } + 1 \big ) , \qquad e _ { t } = { \bf 1 } \big [ r _ { t } = K _ { \mathrm { h o l d } } \big ] , \qquad e _ { - 1 } = 0 ,\tag{19}
$$

for the predicate value $b _ { t } = b ( x _ { t } ) \in \{ 0 , 1 \}$ , evaluated for all $N _ { \mathrm { e n v } }$ simulation instances in parallel.

A job completes at its completion event once its required events have occurred since its start (Table 23). It fails at a failure event, which overrides a completion in the same step, times out at its limit $\tau _ { \mathrm { j o b } }$ , or is censored if the run ends first. A start event for a unit whose job is still open is a violation, as is an event outside any job, such as a hand inside the die while the ram is away from home.

## 5.2.2 Metrics

We measure sustained performance through throughput, job outcomes, latency, and motor energy. Throughput is the number of completed jobs per simulated hour, 3600 $N _ { \mathrm { d o n e } } / t _ { \mathrm { r u n } } .$ , where $t _ { \mathrm { r u n } }$ is measured in seconds, as in bin picking (Mahler et al., 2019). Policy comparisons hold the evaluation and work supply fixed.

Completion latency is reported alongside failed, timed-out, and censored jobs. Jobs still open at the end of a run are right-censored (Kaplan and Meier, 1958), while failures are competing outcomes (Fine and Gray, 1999). Reporting latency only for completed jobs can favor policies that finish easier jobs.

HOLIDAY-SIM measures electrical power at the motor terminals of each actuator. We report drawn and regenerated energy, peak drawn power, and drawn energy per completed job. The latter includes energy spent on failed attempts and is undefined when no job completes. These measures exclude computing and sensing loads and are reported separately from throughput.

## 5.2.3 Rubrics

A rubric specifies run termination, reporting, and optional success criteria. Conveyor loading reports performance over a fixed interval without a success criterion. For rubrics requiring $N _ { \mathrm { g o a l } }$ completed jobs, a run terminates upon reaching the target, the time limit $T _ { \mathrm { r u n } } ,$ or a specified inactivity limit. A run passes only if it reaches the target without failures, timeouts, stalls, or violations (Eq. (45)). Passing reflects only the encoded predicates, which may miss placement defects outside their specified tolerances.

Verification with scripted trajectories. We further test the evaluations of six tasks, excluding conveyor loading, using 185 region checks, 20 start-state probes, and 32 scripted trajectories with predefined outcomes. All tests match their expected results after correcting spurious start events during object transport (§ F.3). These results establish agreement on the tested cases, not complete task coverage or policy performance.

## 5.3 Evaluation across variants

Evaluation is organized around scene variants, each specifying a complete task configuration. The placements within a variant encode relationships established during scene construction. Moving a fixture may require corresponding changes to part placement, support, access space, and target regions. Using the variant as the evaluation unit retains these coordinated choices, subject to the verification limits of § 4.2.1.

An evaluation regime declares the variation axes and conditions represented by its variants, such as object identity, spatial configuration, physical properties, and visual surroundings (Xie et al., 2024a; Pumacay et al., 2024). Outcomes are grouped by these axes and conditions and aggregated across the evaluated set. Background-only variants share the dynamics $F _ { s , \ell }$ and differ in the observation model $O _ { s }$ (§ 4.2.2). Isolating one axis requires matched variants that hold the other conditions fixed.

Policies should be compared on the same declared variants, with initial conditions and randomization settings drawn from the same distribution (Snyder et al., 2026). Repeated runs provide observations within each variant, and the comparison specifies how variants are sampled and weighted in aggregate results. Jobs within a run share the state left by earlier jobs, so the run is the unit of statistical analysis (Agarwal et al., 2021; Cameron and Miller, 2015). Repeat counts and uncertainty estimates are reported alongside job outcomes, throughput, latency, and motor-terminal energy. For a fixed evaluation, physical outcomes can be recomputed from a stored trajectory (Figure 21).

## 6 Discussion and open questions

## 6.1 Value of richer physical simulation

Richer physical models allow a benchmark to capture contact, actuator, and sensor phenomena that simpler models omit. However, their value depends on how well they reproduce real hardware behavior, whether they improve policy comparisons, and whether the additional physical information can be used to evaluate the policy.

How accurately do the models reproduce real hardware? This paper examines the contact, actuator, and sensor models against their specified behavior and does not validate them against hardware measurements (Oberkampf and Roy, 2010). The fingertip studies compare with quadrature of the same pressure law (§ A.3), and the wheel study with ideal differential-drive kinematics (§ A.3). The battery comparison and the finger-rolling and tactile examples show behavior under the stated settings and have no independent reference (§ 3). No result has been compared with measurements from a physical robot.

Which physical details change rankings? Contact models, actuator dynamics, time steps, and solver resolution can affect task outcomes, but their influence on policy rankings remains unmeasured and underexplored. The central question is therefore which physical phenomena and levels of numerical accuracy are necessary to reproduce policy rankings observed on real hardware, and which can be simplified without changing those rankings.

What distinguishes gentle from damaging handling? Job completion and motor energy do not fully capture how objects are handled. A part may be placed gently or dropped into a fixture while satisfying the same success criteria. Contact-derived measures such as impact impulse, approach speed, peak pressure, accumulated slip, and transverse insertion force could distinguish these behaviors when measured for individual parts and job phases. Force and pressure limits are used in human–robot contact safety (Haddadin et al., 2009; Internationa Organization for Standardization, 2016), while damage-aware manipulation considers object integrity alongside grasp success (Xie et al., 2024c). However, these quantities do not predict damage without material-specific thresholds, and their values can depend on numerical resolution. Which measures can reliably distinguish acceptable from damaging handling without penalizing forces required by the task?

## 6.2 Limits of automated verification

What should decide scene acceptance? Asset-level verification establishes whether an asset satisfies its specified requirements, but not whether it supports the intended interaction once placed in a scene. Scene-leve checks introduce additional constraints on placement and spatial relations, yet they do not establish whether the intended interactions are feasible, including whether the robot can reach and grasp the objects. Model critics provide additional judgments, yet the same candidates that pass deterministic checks receive different acceptance decisions depending on the critic (Figure 15), a known weakness of model judges (Zheng et al., 2023). It remains an open question to determine which aspects of scene suitability require additional evidence such as reachability analysis or rendered observations.

How can evaluations be validated with limited labels? A related limitation arises in evaluation generation. Even when an evaluation passes verification, its criteria may not faithfully reflect the intended operation. In the evaluation generation study of § 4.3.3, feedback from verification raises acceptance without changing a single verdict, because the loop receives no labels, and the errors that remain come from what the instruction leaves unstated. These results rest on one task and fifteen recordings. More generally, model judges can share errors because language-model errors are correlated across models (Kim et al., 2025). Success detection and dense progress estimates from vision-language models (Du et al., 2023; Ma et al., 2025) face the same limit, because agreement among them is not evidence of correctness.

One possibility is to compare different forms of verifiers generated from the same instruction. Beyond the predicates on simulator state in 24/7-BENCH, instructions can be expressed as temporal logic (Cosler et al., 2023; Chen et al., 2023; Liu et al., 2023b), reward code (Yu et al., 2023; Ma et al., 2024; Xie et al., 2024b), or a learned success detector or reward model (Du et al., 2023; Lee et al., 2026; Liang et al., 2026). These verifiers may fail in different ways, and disagreements among them could reveal misinterpreted requirements. However, disagreement alone does not establish which judgment is correct, and different verifiers may share the same errors. This suggests using verifier disagreement to identify executions worth labeling, and combining a limited number of independent labels with model judgments to assess the reliability of generated evaluations (Angelopoulos et al., 2023; Kossen et al., 2021).

## 6.3 Expressiveness and diversity of generated scenes

Do generated scenes remain diverse? A generator that passes every check can still return the same few scenes. The checks of § 4.2.1 require distinct layouts, while its family diversity is defined but not quantified here. Moreover, distinct layouts do not necessarily demand different capabilities from a policy. Repeated revision may also favor layouts and assets that are easier to verify, potentially reducing diversity, although this effect has not yet been measured. Quality-diversity search keeps an archive of good solutions that differ along chosen descriptors (Mouret and Clune, 2015), open-ended and unsupervised environment design generate environments that stay novel and solvable (Wang et al., 2019; Dennis et al., 2020), and similarity kernels measure diversity without a reference set (Friedman and Dieng, 2023). The challenge is to identify which variations in geometry, physical properties, and appearance meaningfully change what a policy must do, rather than merely how a scene looks.

Do scene variations predict real-world performance? The value of generated scene variants for policy evaluation depends in part on whether performance across them predicts performance in the real world (Li et al., 2024b). This has not been exhaustively tested. Scenes reconstructed from real cells (Torne et al., 2024; Han et al., 2026) offer one way to test it, because scores on a reconstruction can be compared with trials in the cell it was built from. Such comparisons could help determine which physical and visual variations are needed for evaluations on generated scenes to predict real-world policy performance.

## 6.4 Evolving role of coding agents

Which parts of the harness remain necessary as coding agents improve? The staged checks, typed specifications, and model critics in our pipeline are designed to guide coding agents through world generation, but their value may change as agents become more capable. Prior work shows that agent interfaces and scaffolds affect performance on software tasks (Yang et al., 2024a) and in robot control (Fu et al., 2026; Berman et al., 2026), and that a simple pipeline without an agent loop can match elaborate software agents (Xia et al., 2025). In inverse scene generation, a change in how the prompt states affordances helps more than a larger budget (Table 19). These results establish the importance of pipeline design under the tested conditions, but not whether its individual components will remain necessary as coding agents improve. Some may continue to provide essential guarantees, while others may become unnecessary overhead or limit what stronger agents can achieve.

A related question is how much of the harness itself can be delegated to coding agents. For example, ERIS infers the requirements of an asset from the task, but the language that states them and the check behind each of its terms are designed by hand. Letting the generation agent add terms, measurements, and probes from a fixed set of primitives would let verification grow with the benchmark, but a requirement written by the agent that also generates the asset is not an independent check of it, much as language models favor their own outputs when they judge them (Panickssery et al., 2024).

How should the pipeline be tested? As discussed above, some parts of the harness may become unnecessary as coding agents improve, so its value must be shown again with every change rather than assumed. The comparison of staged verification against its baselines (Figure 14) can serve as a fixed internal benchmark that every change of agent, prompt, or harness must pass, as software agents are compared on fixed task suites (Jimenez et al., 2024). Agents also change between versions, so a generated world is reproducible only with the agent and harness that generated it. A small, fixed test suite could make these comparisons practical, provided it can distinguish improvements in the agent from improvements in the harness.

Can world generation evaluate coding agents? Coding agents are now evaluated on embodied work, as controllers that write programs for a robot (Fu et al., 2026) or act through a visual interface (Hu et al., 2026). 24/7-BENCH adds two roles for them. Its tasks can score a coding agent as the policy under test, with the same evaluations as any other policy, and generating a world from a task is itself a programming problem whose output passes or fails checks that need no labels (§ 4.2). Verified world generation could provide a separate measure of coding-agent capability, although its relationship to robot-control performance remains unexplored.

## 6.5 Physics-grounded supervision for learned evaluators

Reset-free execution provides information that isolated episodes do not capture. Job events reveal how work progresses across repeated operations, including stalls, failures, and their effects on subsequent jobs. These records can provide supervision for learned progress estimators, but they also complicate policy comparison because successive jobs share the same evolving physical state.

Can physics calibrate value functions? Learned models now serve as value functions that estimate progress from frames (Ma et al., 2025; Budzianowski et al., 2025), as reward models for reinforcement learning (Zhang et al., 2025; Zhai et al., 2025; Lee et al., 2026; Liang et al., 2026), and as general models that label progress for robot learning (Gemini 4 Argon; Rao (2026)). Most are trained or scored against progress taken from frame order or from human labels, but frame order assumes that progress grows with time, which fails for runs that stall or regress, and human labels are costly at the length of a shift. Physics gives the progress of every job without labels, over runs long enough for stalls, regressions, and compounding errors to occur, so 24/7-BENCH can measure where an estimate drifts from completed work, as most whole-run judges of § F.4 do when they accept a tote that lies upside down. Physics-defined progress could therefore provide supervision for learned value functions, although its usefulness in real cells without privileged simulator state remains to be established.

## 6.6 Statistical reliability of reset-free evaluation

How many runs separate two policies? How reliably the benchmark can distinguish between policies remains unknown, because no policies have yet been compared on the seven tasks. Jobs within one run depend on the state left by earlier jobs, so treating them as independent observations would underestimate uncertainty. The interval estimates recommended for policy evaluation (Agarwal et al., 2021; Kress-Gazit et al., 2024) must therefore use runs as the statistical unit or account for dependencies among jobs within each run, although cluster-robust uncertainty estimates can be unreliable with few runs (Cameron and Miller, 2015). Sequential tests can reduce evaluation cost by stopping once sufficient evidence is collected, for binary success (Snyder et al.,

2025) and continuous outcomes (Snyder et al., 2026), but must also account for this dependence. Establishing reliable policy comparisons will require measuring run-to-run variability and how the precision of estimated policy differences depends on the number and duration of runs.

## 7 Conclusion

We introduce HOLIDAYFACTORY 1.0, a framework for constructing task-conditioned industrial worlds and evaluating repeated robot jobs under persistent scene state. HOLIDAY-SIM extends Newton’s SDF-based hydroelastic contact with constitutive laws and a semi-implicit primal-dual solver, couples stateful actuators to articulated dynamics, and provides simulated tactile and joint observations. Verification-guided world construction generates and revises assets and scene variants, incorporates photorealistic backgrounds and process logic, and compiles human-confirmed evaluation proposals into job definitions, metrics, and rubrics. 24/7-BENCH defines seven industrial tasks that measure job outcomes, throughput, latency, and motor energy without resetting the scene between jobs.

Controlled studies characterize physical and sensor responses, demonstrate the benefits of measured feedback in asset generation, and reveal the dependence of scene acceptance on model critics. Evaluation-generation experiments further show that passing executable checks does not establish agreement with task intent. Scripted trajectories test the benchmark’s job-level evaluations, but policy comparisons over persistent runs remain untested.

These findings identify several open challenges. The physical models require hardware validation, and their effects on policy rankings remain unknown. Generated worlds must be assessed for task feasibility and meaningful diversity beyond executable validity. Verification harnesses must also be evaluated independently of the coding agents they support. Physics-derived job records offer potential supervision for learned evaluators, but their usefulness beyond simulation remains to be established. Most importantly, reliable policy comparison requires accounting for dependencies among successive jobs and measuring uncertainty across repeated runs.

We believe simulation benchmarks should extend beyond household tasks to capture the repeated, statedependent work of industrial robots. We hope the methods, empirical findings, and open questions presented in this paper contribute to the open-source robotics ecosystem by informing the construction of industrial simulation environments, task-conditioned world generation, and persistent evaluation protocols.

## 8 Contributors and acknowledgments

## 8.1 Contributors

Within each role, contributors are listed alphabetically by given name and then by family name, and the order does not indicate relative contribution.

Project Leads Jeongmin Lee, Taehyeon Kim

Core Contributors Dohyun Jang, Minji Lee, Seonghun Oh

Contributors Dongwoo Park, Geonhyup Lee, Jaehyung Kim, Jaemin Yoon, Jinseop Bok, Jungwoo Kim, Minseok Yang, Philsik Chang, Soomin Lee

## 8.2 Acknowledgments

We thank ChanHee “Luke” Song (NVIDIA), Justin Yu (UC Berkeley), Minyoung Hwang (MIT), and Seungjae Lee (University of Maryland) for careful discussions and for their insights into recent developments in the field.

## References

G. M. Achilli, S. Logozzo, M. Malvezzi, and M. C. Valigi. Contact mechanics analysis of a soft robotic fingerpad. Frontiers in Mechanical Engineering, 8:966335, 2022. doi: 10.3389/fmech.2022.966335. URL https://www.frontiersin.org/journals/mec hanical-engineering/articles/10.3389/fmech.2022.966335/full.

R. Agarwal, M. Schwarzer, P. S. Castro, A. Courville, and M. G. Bellemare. Deep reinforcement learning at the edge of the statistical precipice. In Advances in Neural Information Processing Systems, 2021. URL https://arxiv.org/abs/2108.13264.

I. Akinola, J. Xu, J. Carius, D. Fox, and Y. Narang. TacSL: A library for visuotactile sensor simulation and learning. IEEE Transactions on Robotics, 41:2645–2661, 2025. doi: 10.1109/TRO.2025.3547267.

A. N. Angelopoulos, S. Bates, C. Fannjiang, M. I. Jordan, and T. Zrnic. Prediction-powered inference. Science, 382(6671): 669–674, 2023. doi: 10.1126/science.adi6000.

J. Bao, H. Chen, L. Zhu, C. Liu, R. Zhang, K. Luo, Z. Hu, W. Chen, Y. Yin, X. Wang, Z. Lin, J. Zhang, and X. Han. LumiTex: Towards high-fidelity PBR texture generation with illumination context. In International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=CDwG0Bebfo.

BEHAVIOR Team. 2025 BEHAVIOR Challenge: Evaluation and rules, 2025. URL https://behavior.stanford.edu/challenge/arch ive/2025/evaluation.html. Accessed November 4, 2026.

S. Berman, M. Ilie, J. Deng, and D. Freeman. How claude performs on robotics tasks. Anthropic Frontier Red Team research report, 2026. URL https://www.anthropic.com/research/claude-plays-robotics.

R. Bhirangi, T. Hellebrekers, C. Majidi, and A. Gupta. ReSkin: Versatile, replaceable, lasting tactile skins. In Conference on Robot Learning, volume 164 of Proceedings of Machine Learning Research, pages 587–597. PMLR, 2021. URL https://proceedings.mlr.press/v164/bhirangi22a.html.

P. Budzianowski, E. Wisnios, M. Tyrolski, G. Góral, I. Kulakov, V. Petrenko, and K. Walas. OpenGVL - benchmarking visua´ temporal progress for data curation. arXiv preprint arXiv:2509.17321, 2025. URL https://arxiv.org/abs/2509.17321.

B. Burger, P. M. Maffettone, V. V. Gusev, C. M. Aitchison, Y. Bai, X. Wang, X. Li, B. M. Alston, B. Li, R. Clowes, N. Rankin, B. Harris, R. S. Sprick, and A. I. Cooper. A mobile robotic chemist. Nature, 583(7815):237–241, 2020. doi: 10.1038/s41586 -020-2442-2.

A. C. Cameron and D. L. Miller. A practitioner’s guide to cluster-robust inference. Journal of Human Resources, 50(2):317–372, 2015. doi: 10.3368/jhr.50.2.317.

C. Canudas de Wit, H. Olsson, K. J. Åström, and P. Lischinsky. A new model for control of systems with friction. IEEE Transactions on Automatic Control, 40(3):419–425, 1995. doi: 10.1109/9.376053.

A. X. Chang, T. Funkhouser, L. Guibas, P. Hanrahan, Q. Huang, Z. Li, S. Savarese, M. Savva, S. Song, H. Su, J. Xiao, L. Yi, and F. Yu. ShapeNet: An information-rich 3D model repository. arXiv preprint arXiv:1512.03012, 2015.

T. Chen, Z. Chen, B. Chen, Z. Cai, Y. Liu, Z. Li, Q. Liang, X. Lin, Y. Ge, Z. Gu, W. Deng, Y. Guo, T. Nian, X. Xie, Q. Chen, K. Su, T. Xu, G. Liu, M. Hu, H.-a. Gao, K. Wang, Z. Liang, Y. Qin, X. Yang, P. Luo, and Y. Mu. RoboTwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. In Internationa Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 13673–13699, 2026a. URL https://proceedings.mlr.press/v306/chen26j.html.

X. Chen, F.-J. Chu, P. Gleize, K. J. Liang, A. Sax, H. Tang, W. Wang, M. Guo, T. Hardin, X. Li, et al. SAM 3D: 3Dfy anything in images. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7220–7232, 2026b. URL https: //openaccess.thecvf.com/content/CVPR2026/html/Chen\_SAM\_3D\_3Dfy\_Anything\_in\_Images\_CVPR\_2026\_paper.html.

Y. Chen, R. Gandhi, Y. Zhang, and C. Fan. NL2TL: Transforming natural languages to temporal logics using large language models. In Conference on Empirical Methods in Natural Language Processing, 2023. doi: 10.18653/v1/2023.emnlp-main. 985.

COMSOL. Tangential contact with friction. COMSOL Multiphysics 6.4 Documentation, 2025. URL https://doc.comsol.com/6.4 /doc/com.comsol.help.sme/sme\_ug\_theory.06.088.html. Accessed October 5, 2026.

N. Correll, K. E. Bekris, D. Berenson, O. Brock, A. Causo, K. Hauser, K. Okada, A. Rodriguez, J. M. Romano, and P. R. Wurman. Analysis and observations from the first Amazon picking challenge. IEEE Transactions on Automation Science and Engineering, 15(1):172–188, 2018. doi: 10.1109/TASE.2016.2600527.

M. Cosler, C. Hahn, D. Mendoza, F. Schmitt, and C. Trippel. nl2spec: Interactively translating unstructured natural language to temporal logics with large language models. In Computer Aided Verification, Lecture Notes in Computer Science, 2023. doi: 10.1007/978-3-031-37703-7\_18.

A. T. Dang, J. Lee, M. Mukadam, X. A. Wu, B. Bucher, M. Nambi, and N. Fazeli. HydroShear: Hydroelastic shear simulation for tactile sim-to-real reinforcement learning. In Robotics: Science and Systems, 2026. doi: 10.15607/RSS.2026.XXII.155. URL https://www.roboticsproceedings.org/rss22/p155.html.

M. Deitke, E. VanderBilt, A. Herrasti, L. Weihs, K. Ehsani, J. Salvador, W. Han, E. Kolve, A. Kembhavi, and R. Mottaghi. ProcTHOR: Large-scale embodied AI using procedural generation. In Advances in Neural Information Processing Systems, volume 35, 2022. URL https://proceedings.neurips.cc/paper\_files/paper/2022/hash/27c546ab1e4f1d7d638e6a8dfbad9a0 7-Abstract-Conference.html.

M. Deitke, D. Schwenk, J. Salvador, L. Weihs, O. Michel, E. VanderBilt, L. Schmidt, K. Ehsani, A. Kembhavi, and A. Farhadi. Objaverse: A universe of annotated 3D objects. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13142–13153, 2023.

M. Dennis, N. Jaques, E. Vinitsky, A. Bayen, S. Russell, A. Critch, and S. Levine. Emergent complexity and zero-shot transfer via unsupervised environment design. In Advances in Neural Information Processing Systems, 2020. URL https://arxiv.org/abs/2012.02096.

A. Donzé and O. Maler. Robust satisfaction of temporal logic over real-valued signals. In Formal Modeling and Analysis of Timed Systems, Lecture Notes in Computer Science, pages 92–106. Springer, 2010. doi: 10.1007/978-3-642-15297-9\_9.

Y. Du, K. Konyushkova, M. Denil, A. Raju, J. Landon, F. Hill, N. de Freitas, and S. Cabi. Vision-language models as success detectors. In Conference on Lifelong Learning Agents, volume 232 of Proceedings of Machine Learning Research, pages 120–136, 2023.

R. Elandt, E. Drumwright, M. Sherman, and A. Ruina. A pressure field model for fast, robust approximation of net contact force and moment between nominally rigid objects. In IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 8238–8245. IEEE, 2019. doi: 10.1109/IROS40897.2019.8968548. URL https://doi.org/10.1109/IROS40897.2019.8 968548.

Eurostat. Employed persons by detailed occupation (ISCO-08 two-digit level). Dataset lfsa\_egai2d, updated 10 September 2026, 2026. URL https://ec.europa.eu/eurostat/databrowser/view/lfsa\_egai2d/default/table.

G. E. Fainekos and G. J. Pappas. Robustness of temporal logic specifications for continuous-time signals. Theoretical Computer Science, 410(42):4262–4291, 2009. doi: 10.1016/j.tcs.2009.06.021.

W. Feng, W. Zhu, T.-j. Fu, V. Jampani, A. Akula, X. He, S. Basu, X. E. Wang, and W. Y. Wang. LayoutGPT: Compositiona visual planning and generation with large language models. In Advances in Neural Information Processing Systems, 2023. URL https://arxiv.org/abs/2305.15393.

J. P. Fine and R. J. Gray. A proportional hazards model for the subdistribution of a competing risk. Journal of the American Statistical Association, 94(446):496–509, 1999. doi: 10.1080/01621459.1999.10474144.

M. A. Fischler and R. C. Bolles. Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography. Communications of the ACM, 24(6):381–395, 1981. doi: 10.1145/358669.358692. URL https://graphics.stanford.edu/courses/cs233-21-spring/ReferencedPapers/ransac.pdf.

C. B. Frey and M. A. Osborne. The future of employment: How susceptible are jobs to computerisation? Technological Forecasting and Social Change, 114:254–280, 2017. doi: 10.1016/j.techfore.2016.08.019.

D. Friedman and A. B. Dieng. The Vendi score: A diversity evaluation metric for machine learning. Transactions on Machine Learning Research, 2023. URL https://arxiv.org/abs/2210.02410.

L. Fu, J. Yu, K. El-Refai, E. Kou, H. Xue, H. Huang, W. Xiao, L. Fei-Fei, G. Shi, J. Wu, S. S. Sastry, Y. Zhu, K. Goldberg, and L. Fan. CaP-X: A framework for benchmarking and improving coding agents for robot manipulation. In International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 32155–32212, 2026. URL https://proceedings.mlr.press/v306/fu26u.html.

Google DeepMind. MuJoCo documentation: Computation, 2026. URL https://mujoco.readthedocs.io/en/stable/computation/in dex.html. Accessed November 4, 2026.

S. Haddadin, A. Albu-Schäffer, and G. Hirzinger. Requirements for safe robots: Measurements, analysis and new insights. The International Journal of Robotics Research, 28(11–12):1507–1527, 2009. doi: 10.1177/0278364909343970.

X. Han, J. Yu, M. Liu, Y. Chen, X. Lyu, Y. Tian, B. Wang, W. Zhang, and J. Pang. Re<sup>3</sup>Sim: Generating high-fidelity simulation data via 3D-photorealistic real-to-sim for robotic manipulation. In IEEE International Conference on Robotics and Automation, pages 20781–20788, 2026. doi: 10.1109/ICRA57385.2026.11695858. URL https://arxiv.org/abs/2502.08645.

H. Hao, S. N. Syed, J. Ichnowski, and J. Schneider. FAR: Failure-aware retry for test-time recovery and continual policy improvement. arXiv preprint arXiv:2607.01111, 2026. URL https://arxiv.org/abs/2607.01111.

H. Hu, J. Gao, P. Sundaresan, S. Sharma, J. Bohg, and D. Sadigh. VIA: Visual interface agent for robot control. arXiv preprint arXiv:2607.11119, 2026. URL https://arxiv.org/abs/2607.11119.

P. Hua, M. Liu, A. Macaluso, Y. Lin, W. Zhang, H. Xu, and L. Wang. GenSim2: Scaling robot data generation with multi-modal and reasoning LLMs. In Conference on Robot Learning, volume 270 of Proceedings of Machine Learning Research, pages 5030–5066, 2024. URL https://proceedings.mlr.press/v270/hua25a.html.

K. H. Hunt and F. R. E. Crossley. Coefficient of restitution interpreted as damping in vibroimpact. Journal of Applied Mechanics, 42(2):440–445, 1975. doi: 10.1115/1.3423596.

HunyuanWorld Team. HunyuanWorld 1.0: Generating immersive, explorable, and interactive 3D worlds from words or pixels. arXiv preprint arXiv:2507.21809, 2025. URL https://arxiv.org/abs/2507.21809.

International Federation of Robotics. World robotics 2020 – industrial robots, chapter 1: Introduction, sources and methods. Technical report, International Federation of Robotics, 2020. URL https://ifr.org/img/worldrobotics/WR\_Industrial\_Robots\_2 020\_Chapter\_1.pdf. Section 1.7.4, classification of applications.

International Federation of Robotics. World robotics 2024: Press conference presentation. Slide deck, 2024. URL https: //ifr.org/img/worldrobotics/Press\_Conference\_2024.pdf.

International Federation of Robotics. World robotics 2025: Industrial robots. Technical report, International Federation of Robotics, 2025. URL https://ifr.org/img/worldrobotics/Executive\_Summary\_WR\_2025\_Industrial\_Robots.pdf.

International Federation of Robotics. Robot density surges in Europe, Asia, and Americas. Press release, 8 April 2026, 2026a. URL https://ifr.org/ifr-press-releases/news/robot-density-surges-in-europe-asia-and-americas.

International Federation of Robotics. European union’s industrial robot stock hits record 700,000 units. Press release, 24 September 2026, 2026b. URL https://ifr.org/downloads/press\_docs/EN-2026-SEP-14-IFR\_Press\_Release\_WR-EU-27.pdf.

International Federation of Robotics. Global sales of professional service robots surge 24%. Press release, 30 September 2026, 2026c. URL https://ifr.org/ifr-press-releases/news/global-sales-of-professional-service-robots-surge-24-percent.

International Federation of Robotics. U.S. now second-largest robotics market, following China. Press release, 24 September 2026, 2026d. URL https://ifr.org/downloads/press\_docs/EN-2026-SEP-24-IFR\_Press\_Release\_WR-USA.pdf.

International Federation of Robotics. Five million robots now operate in factories globally: World robotics 2026 report. Press release, 24 September 2026, 2026e. URL https://ifr.org/ifr-press-releases/news/five-million-robots-now-operate-in-factor es-globally.

International Organization for Standardization. Robots and robotic devices – collaborative robots. Technical Specification ISO/TS 15066:2016, ISO, Geneva, 2016. URL https://www.iso.org/standard/62996.html.

International Organization for Standardization. Safety of machinery – two-hand control devices – principles for design and selection. International Standard ISO 13851:2019, ISO, Geneva, 2019. URL https://www.iso.org/standard/70295.html.

A. Jaafar, S. S. Raman, S. Harithas, Y. Wei, S. Juliani, A. Wernerfelt, B. Quartey, I. Idrees, J. X. Liu, and S. Tellex. λ: A benchmark for data-efficiency in long-horizon indoor mobile manipulation robotics. In IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 8942–8949, 2025. doi: 10.1109/IROS60139.2025.11246789.

A. Jain, M. Zhang, K. Arora, W. Chen, M. Torne, M. Z. Irshad, S. Zakharov, Y. Wang, S. Levine, C. Finn, W.-C. Ma, D. Shah, A. Gupta, and K. Pertsch. PolaRiS: Scalable real-to-sim evaluations for generalist robot policies. In Robotics: Science and Systems, 2026. doi: 10.15607/RSS.2026.XXII.062. URL https://www.roboticsproceedings.org/rss22/p062.html.

Y. Jangir, Y. Zhang, K. Yamazaki, C. Zhang, K.-H. Tu, T.-W. Ke, L. Ke, Y. Bisk, and K. Fragkiadaki. RobotArena ∞: Scalable robot benchmarking via real-to-sim translation. In International Conference on Learning Representations, 2026. URL https: //proceedings.iclr.cc/paper\_files/paper/2026/hash/2aa3da3c1463ee2cdaaee94be4f8ba3f-Abstract-Conference.html.

H. Ji, Y. Lee, and D. Lee. TE-SDF: Tetra-encoded signed distance field for memory-efficient and accurate collision detection. In Robotics: Science and Systems, 2026. doi: 10.15607/RSS.2026.XXII.171. URL https://www.roboticsproceedings.org/rs s22/p171.html.

Y. Jia, H. Zhang, Z. Zhang, L. Han, J. Wu, M. Yu, Z. Wang, et al. GS-Playground: A high-throughput photorealistic simulator for vision-informed robot learning. In Robotics: Science and Systems, 2026. doi: 10.15607/RSS.2026.XXII.093. URL https://www.roboticsproceedings.org/rss22/p093.html.

G. Jiang, H. Chang, R.-Z. Qiu, Y. Liang, M. Ji, J. Zhu, Z. Dong, X. Zou, and X. Wang. GSWorld: Closed-loop photorealistic simulation suite for robot manipulation. In IEEE International Conference on Robotics and Automation, pages 384–391, 2026. doi: 10.1109/ICRA57385.2026.11697084. URL https://arxiv.org/abs/2510.20813.

H. Jiang, Z. Song, Z. Lou, R. Xu, and M. Tan. Depth anything in 360<sup>◦</sup>: Towards scale invariance in the wild. arXiv preprint arXiv:2512.22819, 2025. URL https://arxiv.org/abs/2512.22819.

C. E. Jimenez, J. Yang, A. Wettig, S. Yao, K. Pei, O. Press, and K. Narasimhan. SWE-bench: Can language models resolve real-world GitHub issues? In International Conference on Learning Representations, 2024.

E. L. Kaplan and P. Meier. Nonparametric estimation from incomplete observations. Journal of the American Statistical Association, 53(282):457–481, 1958. doi: 10.1080/01621459.1958.10501452.

B. Kerbl, G. Kopanas, T. Leimkühler, and G. Drettakis. 3D Gaussian Splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4):1–14, 2023. doi: 10.1145/3592433.

E. M. Kim, A. Garg, K. Peng, and N. Garg. Correlated errors in large language models. In International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 30038–30066, 2025. URL https://proceedings.mlr.press/v267/kim25e.html.

H. Kim, T. Chen, J. Sun, L. W. Osterberg, Q. Chen, K. Wang, and M. Schwager. LEGS: Fine-tuning teleop-free VLAs for humanoid loco-manipulation in an embodied Gaussian splatting world. arXiv preprint arXiv:2606.01458, 2026. URL https://arxiv.org/abs/2606.01458

J. Kossen, S. Farquhar, Y. Gal, and T. Rainforth. Active testing: Sample-efficient model evaluation. In International Conference on Machine Learning, 2021. URL https://arxiv.org/abs/2103.05331.

H. Kress-Gazit, K. Hashimoto, N. Kuppuswamy, P. Shah, P. Horgan, G. Richardson, S. Feng, and B. Burchfiel. Robot learning as an empirical science: Best practices for policy evaluation. arXiv preprint arXiv:2409.09491, 2024. URL https://arxiv.org/abs/2409.09491.

J. Lee, M. Lee, S. Park, J. Yun, and D. Lee. Variations of augmented Lagrangian for robotic multicontact simulation. IEEE Transactions on Robotics, 41:3852–3869, 2025. doi: 10.1109/TRO.2025.3577410. URL https://doi.org/10.1109/TRO.2025 .3577410.

T. Lee, A. Wagenmaker, K. Pertsch, P. Liang, S. Levine, and C. Finn. RoboReward: General-purpose vision-language reward models for robotics. arXiv preprint arXiv:2601.00675, 2026. URL https://arxiv.org/abs/2601.00675.

C. Li, R. Zhang, J. Wong, C. Gokmen, S. Srivastava, R. Martín-Martín, et al. BEHAVIOR-1K: A human-centered, embodied AI benchmark with 1,000 everyday activities and realistic simulation. arXiv preprint arXiv:2403.09227, 2024a. URL https://arxiv.org/abs/2403.09227.

X. Li, K. Hsu, J. Gu, O. Mees, K. Pertsch, H. R. Walke, C. Fu, I. Lunawat, I. Sieh, S. Kirmani, S. Levine, J. Wu, C. Finn, H. Su, Q. Vuong, and T. Xiao. Evaluating real-world robot manipulation policies in simulation. In Conference on Robot Learning, volume 270 of Proceedings of Machine Learning Research, pages 3705–3728, 2024b. URL https://proceedings.mlr.press/ v270/li25c.html.

X. Li, P. Shi, M. Dong, S. Zhang, Z. Xu, D. Lee, S. Chang, Y. Xiang, L. Pan, and J. Jiang. LEGO-Anything: Coding agents for 3D scene reconstruction. arXiv preprint arXiv:2609.36380, 2026. URL https://arxiv.org/abs/2609.36380.

A. Liang, Y. Korkmaz, J. Zhang, M. Hwang, A. Anwar, S. Kaushik, A. Shah, A. S. Huang, L. Zettlemoyer, D. Fox, Y. Xiang, A. Li, A. Bobu, A. Gupta, S. Tu, E. Biyik, and J. Zhang. Robometer: Scaling general-purpose robotic reward models via trajectory comparisons. In Robotics: Science and Systems, 2026.

B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone. LIBERO: Benchmarking knowledge transfer for lifelong robot learning. In Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, volume 36, 2023a. URL https://proceedings.neurips.cc/paper\_files/paper/2023/hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datas ets\_and\_Benchmarks.html.

J. X. Liu, Z. Yang, I. Idrees, S. Liang, B. Schornstein, S. Tellex, and A. Shah. Grounding complex natural language commands for temporal tasks in unseen environments. In Conference on Robot Learning, volume 229 of Proceedings of Machine Learning Research, pages 1084–1110, 2023b. URL https://proceedings.mlr.press/v229/liu23d.html.

Y. J. Ma, W. Liang, G. Wang, D.-A. Huang, O. Bastani, D. Jayaraman, Y. Zhu, L. Fan, and A. Anandkumar. Eureka: Human-level reward design via coding large language models. In International Conference on Learning Representations, 2024.

Y. J. Ma, J. Hejna, A. Wahid, C. Fu, D. Shah, J. Liang, Z. Xu, S. Kirmani, P. Xu, D. Driess, T. Xiao, J. Tompson, O. Bastani, D. Jayaraman, W. Yu, T. Zhang, D. Sadigh, and F. Xia. Vision language models are in-context value learners. In International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/54854cf15a2 4fff9f5134a8641136fe4-Abstract-Conference.html.

M. Macklin. Warp: A high-performance Python framework for GPU simulation and graphics. NVIDIA GPU Technology Conference (GTC), Mar. 2022. URL https://github.com/NVIDIA/warp.

M. Macklin, K. Storey, M. Lu, P. Terdiman, N. Chentanez, S. Jeschke, and M. Müller. Small steps in physics simulation. In ACM SIGGRAPH/Eurographics Symposium on Computer Animation, pages 1–7. Association for Computing Machinery, 2019. doi: 10.1145/3309486.3340247. URL https://doi.org/10.1145/3309486.3340247.

M. Macklin, K. Erleben, M. Müller, N. Chentanez, S. Jeschke, and T.-Y. Kim. Primal/dual descent methods for dynamics. Computer Graphics Forum, 39(8):89–100, 2020. doi: 10.1111/cgf.14104. URL https://doi.org/10.1111/cgf.14104.

A. Madaan, N. Tandon, P. Gupta, S. Hallinan, L. Gao, S. Wiegreffe, U. Alon, N. Dziri, S. Prabhumoye, Y. Yang, S. Gupta, B. P. Majumder, K. Hermann, S. Welleck, A. Yazdanbakhsh, and P. Clark. Self-Refine: Iterative refinement with self-feedback. In Advances in Neural Information Processing Systems, 2023. URL https://arxiv.org/abs/2303.17651.

J. Mahler, M. Matl, V. Satish, M. Danielczuk, B. DeRose, S. McKinley, and K. Goldberg. Learning ambidextrous robot grasping policies. Science Robotics, 4(26):eaau4984, 2019. doi: 10.1126/scirobotics.aau4984.

O. Maler and D. Nickovic. Monitoring temporal properties of continuous signals. In Formal Techniques, Modelling and Analysis of Timed and Fault-Tolerant Systems, Lecture Notes in Computer Science, pages 152–166. Springer, 2004. doi: 10.1007/978-3-540-30206-3\_12.

A. Mandow, J. L. Martínez, J. Morales, J. L. Blanco, A. García-Cerezo, and J. González. Experimental kinematics for wheeled skid-steer mobile robots. In IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 1222–1227, 2007. doi: 10.1109/IROS.2007.4399139.

ManiSkill Contributors. ManiSkill: Learning from demonstrations—evaluation, 2026a. URL https://maniskill.readthedocs.io/en latest/user\_guide/learning\_from\_demos/setup.html. Accessed November 4, 2026.

ManiSkill Contributors. ManiSkill: Table-top 2 finger gripper tasks, 2026b. URL https://maniskill.readthedocs.io/en/latest/tasks /table\_top\_gripper/index.html. Accessed November 4, 2026.

J. Masterjohn, D. Guoy, J. Shepherd, and A. Castro. Velocity level approximation of pressure field contact patches. IEEE Robotics and Automation Letters, 7(4):11593–11600, 2022. doi: 10.1109/LRA.2022.3203845.

T. Menzel, G. Bagschik, and M. Maurer. Scenarios for development, test and validation of automated vehicles. In IEEE Intelligent Vehicles Symposium, pages 1821–1827, 2018. doi: 10.1109/IVS.2018.8500406.

Ministry of Data and Statistics, Republic of Korea. Economically active population survey: Employment trends, december and annual 2025. Statistical table 4, employed persons by occupation, 2026. URL https://mods.go.kr/board.es?act=view&bid=2 10&list\_no=443017&mid=a10301030200.

J.-B. Mouret and J. Clune. Illuminating search spaces by mapping elites, 2015. URL https://arxiv.org/abs/1504.04909.

Y. Narang, K. Storey, I. Akinola, M. Macklin, P. Reist, L. Wawrzyniak, Y. Guo, A. Moravanszky, G. State, M. Lu, A. Handa, and D. Fox. Factory: Fast contact for robotic assembly. In Robotics: Science and Systems, 2022. doi: 10.15607/RSS.2022.XV III.035. URL https://www.roboticsproceedings.org/rss18/p035.html.

S. Nasiriany, A. Maddukuri, L. Zhang, A. Parikh, A. Lo, A. Joshi, A. Mandlekar, and Y. Zhu. RoboCasa: Large-scale simulation of household tasks for generalist robots. In Robotics: Science and Systems, 2024. doi: 10.15607/RSS.2024.XX.050. URL https://www.roboticsproceedings.org/rss20/p050.html.

S. Nasiriany, S. Nasiriany, A. Maddukuri, and Y. Zhu. RoboCasa365: A large-scale simulation framework for training and benchmarking generalist robots. In International Conference on Learning Representations, 2026. URL https: //openreview.net/forum?id=tQJYKwc3n4.

National Bureau of Statistics of China. China Population Census Yearbook 2020. China Statistics Press, 2022. URL https://www.stats.gov.cn/sj/pcsj/rkpc/7rp/zk/html/B0407.xls. Table 4-7, employed population by occupation, long-form sample.

National Institute of Standards and Technology. Assembly performance metrics and test methods. NIST Engineering Laboratory. Accessed October 2026, 2026. URL https://www.nist.gov/el/intelligent-systems-division-73500/robotic-graspin g-and-manipulation-assembly/assembly.

NVIDIA. PhysX SDK 5.1.2 documentation: Advanced collision detection, 2022. URL https://nvidia-omniverse.github.io/Phy sX/physx/5.1.2/docs/AdvancedCollisionDetection.html#persistent-contact-manifold-pcm. Section on persistent contact manifolds. Accessed November 4, 2026.

W. L. Oberkampf and C. J. Roy. Verification and Validation in Scientific Computing. Cambridge University Press, 2010. doi: 10.1017/CBO9780511760396.

S. Oh, X. Yuan, X. Wei, R. Shi, F. Xiang, M. Liu, and H. Su. PaMO: Parallel mesh optimization for intersection-free low-poly modeling on the GPU. Computer Graphics Forum, 44(7):e70267, 2025. doi: 10.1111/cgf.70267.

H. B. Pacejka. Tire and Vehicle Dynamics. Butterworth-Heinemann, 3 edition, 2012. doi: 10.1016/C2010-0-68548-8.

A. Panickssery, S. R. Bowman, and S. Feng. LLM evaluators recognize and favor their own generations. In Advances in Neural Information Processing Systems, 2024. URL https://arxiv.org/abs/2404.13076.

M. Pantano, T. Eiband, and D. Lee. Capability-based frameworks for industrial robot skills: A survey. In IEEE International Conference on Automation Science and Engineering, pages 2355–2362, 2022. doi: 10.1109/CASE49997.2022.9926648. URL https://arxiv.org/abs/2203.00538.

N. Pfaff, T. Cohn, S. Zakharov, R. Cory, and R. Tedrake. SceneSmith: Agentic generation of simulation-ready indoor scenes. In International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 98499–98567, 2026. URL https://proceedings.mlr.press/v306/pfaff26a.html.

W. Pumacay, I. Singh, J. Duan, R. Krishna, J. Thomason, and D. Fox. THE COLOSSEUM: A benchmark for evaluating generalization for robotic manipulation. In Robotics: Science and Systems, 2024. doi: 10.15607/RSS.2024.XX.133.

M. N. Qureshi, S. Garg, F. Yandun, D. Held, G. Kantor, and A. Silwal. SplatSim: Zero-shot Sim2Real transfer of RGB manipulation policies using Gaussian splatting. In IEEE International Conference on Robotics and Automation, pages 6502–6509, 2025. doi: 10.1109/ICRA55743.2025.11128339. URL https://arxiv.org/abs/2409.10161.

A. Raistrick, L. Mei, K. Kayan, D. Yan, Y. Zuo, B. Han, H. Wen, M. Parakh, S. Alexandropoulos, L. Lipson, Z. Ma, and J. Deng. Infinigen indoors: Photorealistic indoor scenes using procedural generation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21783–21794, 2024. URL https://openaccess.thecvf.com/content/CVPR2024/html/Raistric k\_Infinigen\_Indoors\_Photorealistic\_Indoor\_Scenes\_using\_Procedural\_Generation\_CVPR\_2024\_paper.html.

K. Rao. Gemini 4 argon for robotics. Google for Developers Blog, 2026. URL https://developers.googleblog.com/gemini-4-arg on-for-robotics/.

C. Sferrazza, D.-M. Huang, X. Lin, Y. Lee, and P. Abbeel. HumanoidBench: Simulated humanoid benchmark for whole-body locomotion and manipulation. In Robotics: Science and Systems, 2024. doi: 10.15607/RSS.2024.XX.061. URL https://www.roboticsproceedings.org/rss20/p061.html.

A. Sharma, K. Xu, N. Sardana, A. Gupta, K. Hausman, S. Levine, and C. Finn. Autonomous reinforcement learning: Formalism and benchmarking. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id= nkaba3ND7B5.

A. Sharma, A. M. Ahmed, R. Ahmad, and C. Finn. Self-improving robots: End-to-end autonomous visuomotor reinforcement learning. In Conference on Robot Learning, volume 229 of Proceedings of Machine Learning Research, pages 3292–3308, 2023. URL https://proceedings.mlr.press/v229/sharma23b.html.

Z. Si and W. Yuan. Taxim: An example-based simulation model for GelSight tactile sensors. IEEE Robotics and Automation Letters, 7(2):2361–2368, 2022. doi: 10.1109/LRA.2022.3142412.

Z. Si, G. Zhang, Q. Ben, B. Romero, Z. Xian, C. Liu, and C. Gan. DIFFTACTILE: A physics-based differentiable tactile simulator for contact-rich robotic manipulation. In International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=eJHnSg783t.

D. Snyder, A. Hancock, A. Badithela, E. Dixon, P. Miller, R. Ambrus, A. Majumdar, M. Itkina, and H. Nishimura. Is your imitation learning policy better than mine? Policy comparison with near-optimal stopping. In Robotics: Science and Systems, 2025. doi: 10.15607/RSS.2025.XXI.077.

D. Snyder, A. Badithela, N. Matni, G. J. Pappas, A. Majumdar, M. Itkina, and H. Nishimura. Beyond binary success: Sample-efficient and statistically rigorous robot policy comparison. In Robotics: Science and Systems, 2026. doi: 10.15607/RSS.2026.XXII.076. URL https://www.roboticsproceedings.org/rss22/p076.html.

Statistics Bureau of Japan. Annual report on the labour force survey 2025, employed persons by occupation. e-Stat, table I-A-7, 2026. URL https://www.e-stat.go.jp/stat-search/file-download?statInfId=000040440930&fileKind=0.

S. Tao, F. Xiang, A. Shukla, Y. Qin, X. Hinrichsen, X. Yuan, C. Bao, X. Lin, Y. Liu, T.-K. Chan, Y. Gao, X. Li, T. Mu, N. Xiao, A. Gurha, V. N. Rajesh, Y. W. Choi, Y.-R. Chen, Z. Huang, R. Calandra, R. Chen, S. Luo, and H. Su. Demonstrating GPU parallelized robot simulation and rendering for generalizable embodied AI with ManiSkill3. In Robotics: Science and Systems, 2025. doi: 10.15607/RSS.2025.XXI.021. URL https://www.roboticsproceedings.org/rss21/p021.html. Preprint: ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI.

Team HY-World, C. Cao, X. Zuo, Z. Wang, et al. HY-World 2.0: A multi-modal world model for reconstructing, generating, and simulating 3D worlds. arXiv preprint arXiv:2604.14268, 2026. URL https://arxiv.org/abs/2604.14268.

R. Tedrake and the Drake Development Team. Drake: Model-based design and verification for robotics, 2019. URL https://drake.mit.edu.

The Newton Contributors. Newton: GPU-accelerated physics simulation for robotics and simulation research, 2025. URL https://github.com/newton-physics/newton. Software. Accessed October 3, 2026.

T. P. Tomo, A. Schmitz, W. K. Wong, H. Kristanto, S. Somlor, J. Hwang, L. Jamone, and S. Sugano. Covering a robot fingertip with uSkin: A soft electronic skin with distributed 3-axis force sensitive elements for robot hands. IEEE Robotics and Automation Letters, 3(1):124–131, 2018. doi: 10.1109/LRA.2017.2734965.

M. Torne, A. Simeonov, Z. Li, A. Chan, T. Chen, A. Gupta, and P. Agrawal. Reconciling reality through simulation: A real-to sim-to-real approach for robust manipulation. In Robotics: Science and Systems, 2024. doi: 10.15607/RSS.2024.XX.015. URL https://www.roboticsproceedings.org/rss20/p015.html.

S. Ulbrich, T. Menzel, A. Reschka, F. Schuldt, and M. Maurer. Defining and substantiating the terms scene, situation, and scenario for automated driving. In IEEE International Conference on Intelligent Transportation Systems, pages 982–988, 2015. doi: 10.1109/ITSC.2015.164.

U.S. Bureau of Labor Statistics. Occupational employment and wages, may 2025. News release USDL-26-0725, nationa estimates, 2026. URL https://www.bls.gov/news.release/ocwage.htm.

R. Wang, J. Lehman, J. Clune, and K. O. Stanley. Paired open-ended trailblazer (POET): Endlessly generating increasingly complex and diverse learning environments and their solutions, 2019. URL https://arxiv.org/abs/1901.01753.

X. Wang, L. Liu, T. Ding, A. Choi, C. Huang, M. Zhao, Z. Li, J. Jiang, C. Yu, S. Liu, W. Xu, and Z. Su. EmbodiedGen V2: An agentic, simulation-ready 3D world engine for embodied AI. arXiv preprint arXiv:2607.07459, 2026a. URL https://arxiv.org/abs/2607.07459.

Y. Wang, Z. Xian, F. Chen, T.-H. Wang, Y. Wang, K. Fragkiadaki, Z. Erickson, D. Held, and C. Gan. RoboGen: Towards unleashing infinite data for automated robot learning via generative simulation. In International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 51936–51983, 2024. URL https://proceedings. mlr.press/v235/wang24cc.html.

Y.-J. Wang, J. Li, S. Chen, T. E. Truong, P. Xu, P. Abbeel, R. Duan, K. Sreenath, A. Kanazawa, C. Sferrazza, G. Shi, and K. Liu. VLK: Learning humanoid loco-manipulation from synthetic interactions in reconstructed scenes. arXiv preprint arXiv:2606.30645, 2026b. URL https://arxiv.org/abs/2606.30645.

World Labs. Marble. https://marble.worldlabs.ai, 2025.

P.-Y. Wu, D. C.-J. Fang, D. Cheng, and T.-W. Ke. CRONOS: Benchmarking non-episodic reinforcement learning for robotic manipulation. Preprint, 2026. URL https://embodiedai-ntu.github.io/cronos/.

C. S. Xia, Y. Deng, S. Dunn, and L. Zhang. Demystifying LLM-based software engineering agents. Proceedings of the ACM on Software Engineering, 2(FSE):801–824, 2025. doi: 10.1145/3715754.

H. Xia, X. Li, Z. Li, Q. Ma, J. Xu, M.-Y. Liu, Y. Cui, T.-Y. Lin, W.-C. Ma, S. Wang, S. Song, and F. Wei. SAGE: Scalable agentic 3D scene generation for embodied AI. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22358–22368, 2026. URL https://openaccess.thecvf.com/content/CVPR2026/html/Xia\_SAGE\_Scalable\_Agentic\_3D\_Sce ne\_Generation\_for\_Embodied\_AI\_CVPR\_2026\_paper.html.

A. Xie, L. Lee, T. Xiao, and C. Finn. Decomposing the generalization gap in imitation learning for visual robotic manipulation. In IEEE International Conference on Robotics and Automation, pages 3153–3160, 2024a. doi: 10.1109/ICRA57147.2024 .10611331. URL https://arxiv.org/abs/2307.03659.

T. Xie, S. Zhao, C. H. Wu, Y. Liu, Q. Luo, V. Zhong, Y. Yang, and T. Yu. Text2Reward: Reward shaping with language models for reinforcement learning. In International Conference on Learning Representations, 2024b.

W. Xie, M. Valentini, J. Lavering, and N. Correll. DeliGrasp: Inferring object properties with LLMs for adaptive grasp policies. In Conference on Robot Learning, volume 270 of Proceedings of Machine Learning Research, pages 1290–1309, 2024c. URL https://proceedings.mlr.press/v270/xie25a.html.

Z. Xie. WorldGen: Generate any 3D scene in seconds. GitHub repository, 2025. URL https://github.com/ZiYang-xie/WorldGen.

Z. Xiong, Z. Chen, Z. Li, Y. Xu, and N. Jacobs. PanoDreamer: Consistent text to 360-degree scene generation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 295–304, 2025. URL https: //arxiv.org/abs/2504.05152.

J. Yang, C. E. Jimenez, A. Wettig, K. Lieret, S. Yao, K. Narasimhan, and O. Press. SWE-agent: Agent-computer interfaces enable automated software engineering. In Advances in Neural Information Processing Systems, 2024a.

Y. Yang, F.-Y. Sun, L. Weihs, E. VanderBilt, A. Herrasti, W. Han, J. Wu, N. Haber, R. Krishna, L. Liu, C. Callison-Burch, M. Yatskar, A. Kembhavi, and C. Clark. Holodeck: Language guided generation of 3D embodied AI environments. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16227–16237, 2024b. URL https://arxiv.org/ab s/2312.09067.

Z. Yang, W. Ge, Y. Li, J. Chen, H. Li, M. An, F. Kang, H. Xue, B. Xu, Y. Yin, E. Li, Y. Liu, Y. Wang, H.-X. Guo, and Y. Zhou. Matrix-3D: Omnidirectional explorable 3D world generation. arXiv preprint arXiv:2508.08086, 2025. URL https://arxiv.org/abs/2508.08086.

J. Young. xatlas: Mesh parameterization / UV unwrapping library. https://github.com/jpcy/xatlas, 2018. GitHub repository.

W. Yu, N. Gileadi, C. Fu, S. Kirmani, K.-H. Lee, M. Gonzalez Arenas, H.-T. L. Chiang, T. Erez, L. Hasenclever, J. Humplik, B. Ichter, T. Xiao, P. Xu, A. Zeng, T. Zhang, N. Heess, D. Sadigh, J. Tan, Y. Tassa, and F. Xia. Language to rewards for robotic skill synthesis. In Conference on Robot Learning, volume 229 of Proceedings ofMachine Learning Research, pages 374–404, 2023.

S. Zhai, Q. Zhang, T. Zhang, F. Huang, H. Zhang, M. Zhou, S. Zhang, L. Liu, S. Lin, and J. Pang. A vision-languageaction-critic model for robotic real-world reinforcement learning. arXiv preprint arXiv:2509.15937, 2025. URL https: //arxiv.org/abs/2509.15937.

J. Zhang, Y. Luo, A. Anwar, S. A. Sontakke, J. J. Lim, J. Thomason, E. Biyik, and J. Zhang. ReWiND: Language-guided rewards teach robot policies without new demonstrations. In Conference on Robot Learning, volume 305 of Proceedings of Machine Learning Research, pages 460–488, 2025. URL https://proceedings.mlr.press/v305/zhang25a.html.

L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. P. Xing, H. Zhang, J. E. Gonzalez, and I. Stoica. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, 2023. URL https://arxiv.org/abs/2306.05685.

S. Zhou, Z. Fan, D. Xu, H. Chang, P. Chari, T. Bharadwaj, S. You, Z. Wang, and A. Kadambi. DreamScene360: Unconstrained text-to-3D scene generation with panoramic Gaussian splatting. In European Conference on Computer Vision, pages 324–342, 2024. doi: 10.1007/978-3-031-72658-3\_19. URL https://arxiv.org/abs/2404.06903.

## A SDF-hydroelastic contact details

This appendix gives numerical details of the contact model and solver of § 3.1.1 and 3.1.2, followed by the protocols of the contact studies and the response kernel of the magnetic tactile sensor of § 3.1.3. Choices of collision query, field storage, and surface discretization affect cost and numerical approximation, but do not by themselves introduce additional materia phenomena.

## A.1 Collision geometry and memory

Newton’s sparse-grid SDF representation uses a coarse grid and packed high-resolution subgrids near the surface. Concen trating high-resolution storage near the surface avoids allocating a uniformly fine grid throughout the surrounding volume.

For SDF hydroelastic contact, hierarchical pruning identifies candidate cells, and marching cubes extracts the zero level set of the pressure difference. Surface quadrature and contact reduction then provide the contact constraints described in the main text. The cost of this path depends on overlap volume, grid resolution, and the number of generated surface elements.

SDF resolution trades geometric fidelity, including surface features and narrow clearances, against memory consumption.   
Four mechanisms reduce collision memory in batched simulation.

Per-scenario collision budget. A generated scenario also fixes how much collision memory one simulation instance needs. Each asset already carries an SDF resolution chosen under a memory model when it is generated (§ 4.1.3), and the remaining memory requirements depend on the scenario. GPU simulation gains throughput by stepping many instances at once (Tao et al., 2025), and collision buffers grow with the number of instances and of contact pairs. We therefore size the collision buffers for the contacts that a scenario can produce at once rather than for every possible pair and store the sizes with the scenario. The batch size is then chosen from the resulting memory estimate before allocation, instead of shrinking the batch after an out-of-memory failure. The stored sizes hold only if no buffer overflows during a rollout. This per-scenario budget further reduces memory without losing contacts in rollouts that reach the grasp.

Shared structure across simulation instances. Simulation instances that share the robot’s joint structure may hold different scene variants. Identical meshes and catalog assets are stored once per batch, but a signed-distance field is shared only when instances reuse the same loaded asset.

Asset-level SDF budget. Each part’s signed-distance resolution is chosen when the asset enters the catalog, so that no single detailed asset dominates collision memory. HOLIDAY-SIM sizes the buffer of every hydroelastic contact pair (Elandt et al., 2019) by the most finely resolved part in the scene, so one detailed part inflates the memory of all pairs. Compact field encodings trade memory against accuracy (Ji et al., 2026). We keep the encoding, take the coarsest resolution between 32 and 1 mm that represents the part, and refine further only while the part adds at most 1 MB per instance. Parts above 500 MB per instance are rejected. For assets generated by ERIS, collision construction follows § C.2.

Instance-chunked collision. Collision buffers are sized for a chunk of simulation instances rather than the whole batch, while the solver still steps all instances together. In deterministic mode the contact set equals that of a single pass.

## A.2 Pressure law and solver discretization

This subsection gives the evaluation of the nonlinear pressure law of Eq. (5) and the discretization behind the contact solver of § 3.1.2.

Nonlinear pressure evaluation. For contact-surface extraction, we extend the nonnegative interior pressure defined by Eq. (5) outside the body as $p _ { A } = - k _ { h , A } \phi _ { A }$ for $\phi _ { A } \ge 0$ . This signed extension is continuously differentiable at the surface and remains monotone in signed distance. It is used to locate pressure-equilibrium crossings and does not introduce tensile contact forces. The nonlinear law is evaluated at voxel corners and quadrature locations, while marching-cubes edge interpolation still approximates the contact surface and requires adequate spatial resolution.

The compression-hardening preset of Eq. (5) targets the SDF-SDF and SDF-rigid-plane paths. The reduced force-derived stiffness is held fixed within each dynamics step, so introducing nonlinear pressure does not make the time integration fully implicit in the nonlinear constitutive law.

Semi-implicit Hunt-Crossley damping. To fold Hunt-Crossley damping into the implicit solve, we predict penetration as $\delta ^ { k + 1 } \approx \dot { \delta } ^ { k } - h v _ { n } ^ { k + 1 }$ , with normal relative velocity $v _ { n }$ positive on separation, and freeze only the penetration factor in the damping coefficient at its current value. For an active penetrating contact, with patch-derived stiffness K fixed during the substep and before near-rigid regularization, the update is

$$
\begin{array} { r l } & { f _ { n } ^ { k + 1 } = \left[ K ( \delta ^ { k } - h v _ { n } ^ { k + 1 } ) - K \alpha \delta ^ { k } v _ { n } ^ { k + 1 } \right] _ { + } } \\ & { ~ = K \left[ \delta ^ { k } - \tau _ { \mathrm { c o n t a c t } } ^ { k } v _ { n } ^ { k + 1 } \right] _ { + } , \qquad \tau _ { \mathrm { c o n t a c t } } ^ { k } = h + \alpha \delta ^ { k } . } \end{array}\tag{20}
$$

Here, h is the substep duration and $[ x ] _ { + } = \operatorname* { m a x } ( x , 0 )$ enforces a nonnegative normal force. Freezing the penetration factor removes the quadratic velocity term that would arise from a fully implicit Hunt-Crossley product, yielding a monotone, piecewise-affine normal force law for fixed contact geometry. This retains penetration-dependent dissipation while allowing the normal response to be solved jointly with friction and articulated dynamics. Contacts with $\delta ^ { k } \leq 0$ are inactive, so the velocity-dependent term is not applied before penetration.

Discrete friction law. Tangential reactions and the normal forces that bound them remain coupled unknowns. At convergence, the discrete contact law satisfies

$$
\| \mathbf { f } _ { t , i } ^ { k + 1 } \| \leq \mu _ { i } ^ { k } f _ { n , i } ^ { k + 1 } , \qquad \mathbf { f } _ { t , i } ^ { k + 1 } \cdot \mathbf { v } _ { t , i } ^ { k + 1 } = - \mu _ { i } ^ { k } f _ { n , i } ^ { k + 1 } \| \mathbf { v } _ { t , i } ^ { k + 1 } \| ,\tag{21}
$$

where i denotes the contact index. These conditions allow a nonzero sticking force at zero slip velocity and recover friction opposing the next-step motion during sliding. Because each contact samples the local tangential velocity, the same solve resolves both net sliding resistance and the frictional moment generated by twisting across the patch. Freezing the geometry and coefficient preserves the existing local Coulomb update without freezing the friction direction or the normal-force-dependent bound. The coefficient is refreshed at each substep, so rapid transitions between sticking and sliding remain sensitive to the time step. This is a lagged approximation to the continuous speed-dependent law, not a fully implicit treatment of speed weakening.

Near-rigid regularization. A contact with material stiffness $K _ { i }$ has a mass metric $m _ { \mathrm { c o n t a c t } , i } ,$ a numerical estimate of the inertia behind the contact propagated along the load path, not its exact effective mass. The reference stiffness $K _ { \mathrm { r e f } , i } =$ $m _ { \mathrm { c o n t a c t , } i } ( \vartheta / h ) ^ { 2 }$ relates the correction to a dimensionless threshold ϑ. When $K _ { i } > K _ { \mathrm { r e f } , i }$ , the effective stiffness is

$$
K _ { \mathrm { e f f } , i } = K _ { \mathrm { r e f } , i } \left( \frac { K _ { i } } { K _ { \mathrm { r e f } , i } } \right) ^ { \chi _ { i } } , \qquad \chi _ { i } = \mathrm { m i n } \left( \frac { \delta _ { i } } { \delta _ { \mathrm { a l l o w } } } , 1 \right) .\tag{22}
$$

Here, $\delta _ { \mathrm { a l l o w } }$ is the penetration allowance, and contacts below the reference stiffness remain unchanged. This geometric interpolation softens shallow contact while recovering the original stiffness for deeper compression, with additional damping accompanying the softening. Its mass metric is independent of the preconditioner settings, so numerical scaling does not redefine the contact response. Unlike preconditioning, this correction intentionally modifies the discrete material response near contact onset.

Regularization modifies both the stiffness and the contact time scale in Eq. (20). In this paragraph, $\delta _ { i }$ denotes the current penetration ${ \delta } _ { i } ^ { k }$ , and the time index on the contact coefficients is omitted for brevity. We denote the regularized counterpart of $\tau _ { \mathrm { c o n t a c t } , i } ^ { k } = h + \alpha \delta _ { i } ^ { k }$ by $\tau _ { \mathrm { e f f } , i }$ and define it, for a penetrating contact, as

$$
\tau _ { \mathrm { e f f } , i } = \left\{ \begin{array} { l l } { \displaystyle \operatorname* { m a x } \left( h + \alpha \delta _ { i } , h + 2 \left[ \sqrt { \frac { m _ { \mathrm { c o n t a c t } , i } } { K _ { \mathrm { e f f } , i } } } - \sqrt { \frac { m _ { \mathrm { c o n t a c t } , i } } { K _ { i } } } \right] \right) , } & { K _ { i } > K _ { \mathrm { r e f } , i } , } \\ { h + \alpha \delta _ { i } , } & { K _ { i } \le K _ { \mathrm { r e f } , i } . } \end{array} \right.\tag{23}
$$

The second argument of the maximum imposes a damping floor associated with softening and increases the time scale only when it exceeds $h + \alpha \delta _ { i }$ . With these regularized coefficients, the normal contact law becomes

$$
\begin{array} { r l } & { f _ { n , i } ^ { k + 1 } = \left[ K _ { \mathrm { e f f } , i } \left( \delta _ { i } ^ { k } - \tau _ { \mathrm { e f f } , i } v _ { n , i } ^ { k + 1 } \right) \right] _ { + } } \\ & { \qquad = \left[ K _ { \mathrm { e f f } , i } \left( \delta _ { i } ^ { k } - h v _ { n , i } ^ { k + 1 } \right) - K _ { \mathrm { e f f } , i } ( \tau _ { \mathrm { e f f } , i } - h ) v _ { n , i } ^ { k + 1 } \right] _ { + } . } \end{array}\tag{24}
$$

The first term is the elastic response evaluated at the predicted next-step penetration, and the second is the damping term. Its coefficient is therefore $K _ { \mathrm { e f f } , i } ( \tau _ { \mathrm { e f f } , i } - h )$ . When regularization is disabled, $K _ { \mathrm { e f f } , i } = K _ { \mathrm { i } }$ <sub>i</sub> and $\tau _ { \mathrm { e f f } , i } = h + \alpha \delta _ { i } ^ { k }$ , recovering Eq. (20). Nonpenetrating contacts remain inactive.

## A.3 Contact and solver studies

Four studies follow. Each compares the engine with a numerical reference of the same law or with another backend as configured, and none compares with hardware measurements.

Fingertip patch friction. Figure $5 ( \mathsf { a } , \mathsf { d } )$ uses the right-index distal pad of the FRIDAY hand model, isolated from the articulated hand and rigidly reoriented toward a plane without rescaling. Only the pad participates in contact, and the distal housing is visual context. The pad uses $k _ { h } = 1 0 ^ { 8 } \mathrm { N } / \mathrm { m } ^ { 3 } , \mu = 0 . 6$ , a 0.5 mm SDF voxel against the asset default of 2 mm, 0.25 mm quadrature spacing, moment-matching contact reduction, $\alpha = 1 0 ^ { - 6 } \mathrm { s / m }$ , deterministic contact generation, and 128 solver iterations without warm starting or near-rigid regularization. The pressure law is linear, $p = k _ { h } [ - \phi ] +$ , and the Coulomb coefficient is constant. At indentations of 0.25, 0.5, and 1 mm, direct signed-distance queries against the original triangles followed by 0.05 mm planar quadrature give normal loads of 3.39, 7.84, and 18.81 N. Pure rotation at 0.1 rad/s is prescribed about the normal axis through each reference pressure center. Integrating the local sliding tractions gives reference effective radii of 4.71, 5.07, and 5.57 mm.

HOLIDAY-SIM evaluates one dynamics step of 1 ms from each prescribed state. The test body of $1 0 ^ { 4 }$ kg with diagonal inertia $1 0 ^ { 4 }$ kg m<sup>2</sup> in zero gravity keeps the pad from moving during the force probe. At the three indentations, normal-load errors are 0.58–1.26% and effective-radius errors are 0.23–0.52%. With default reduction at the same resolution, effective-radius errors are 6.33–10.74%. Moment matching targets the friction moment, so this agreement checks that design rather than grasp fidelity in general.

The point-contact configuration of panel (d) uses the same pad mesh through a convex collision representation with a few contact points per pair, matched to each reference normal load by pose bisection, and its effective-radius errors are 32.1– 57.6%. An additional nonnegative torsional-friction coefficient fitted at the middle load reaches its zero lower bound because the manifold already produces more torque than the reference. This configuration does not represent every point-contact setup.

Fingertip compression and dissipation. Figure $^ { 5 ( \mathsf { b } , \mathsf { e } ) }$ uses the same isolated pad, test body, stiffness, voxel, quadrature spacing, and solver settings, with default contact reduction and $\alpha = 2 0 \mathrm { { s } / \mathrm { { m } } }$ . Each prescribed pose and velocity is reset before one solver step, so these are constitutive probes rather than a continuously actuated indentation rollout.

For panel (b), the prescribed cycle is $\delta = \delta _ { \mathrm { m a x } } ( 1 - \cos \theta ) / 2$ with $\delta _ { \mathrm { m a x } } = 1$ mm and $\dot { \delta } = v _ { \operatorname* { m a x } }$ sin $\cdot \theta ,$ sampled at 129 phases. Peak speeds of 5, 10, and 20 mm/s give loop work of 1.429, 2.857, and 5.712 mJ, respectively, 10.8% above direct pressure-law quadrature with the same velocity factor. The excess comes from the implicit treatment of the elastic force, which is evaluated at the end-of-step penetration and therefore adds khv to the loading branch and subtracts it from the unloading branch, where k is the local force gradient, h the step, and v the indentation speed. The negligible-dissipation baseline $( \alpha = 1 0 ^ { - 6 } \mathrm { s / m } )$ at 20 mm/s shows the same term as a residual loop work of 0.596 mJ, and at a 0.1 ms step the fastest-cycle excess falls to 0.4%.

Panel (e) holds approach and release speeds at matched magnitudes for each of 31 indentations from 0.25 to 1 mm and extracts

$$
c _ { \mathrm { e f f } } ( \delta ) = \frac { F _ { \mathrm { l o a d } } ( \delta , + v ) - F _ { \mathrm { u n l o a d } } ( \delta , - v ) } { 2 v } ,\tag{25}
$$

where $F _ { \mathrm { l o a d } }$ and $F _ { \mathrm { u n l o a d } }$ are the normal forces on the loading and unloading branches and positive v denotes increasing indentation. $\mathsf { A t } \ v = 1 0 \mathsf { m m / s }$ , computed coefficients are 80.4, 175.9, and 396.7 N s/m at indentations of 0.25, 0.5, and 1 mm, 18.7, 12.2, and 5.4% above $\alpha F _ { \mathrm { e l } }$ , where $F _ { \mathrm { e l } }$ is the elastic normal force. The offset is the implicit term of the previous paragraph, which shifts the coefficient by kh, and it falls below 1.2% at a 0.1 ms step. Repeating at 5 and 20 mm/s changes the extracted coefficient by at most 1.5%. The analytical constant-damping comparator has $c = 1 5 6 . 8 2 \mathrm { N s / m } .$ , calibrated to the reference at 0.5 mm. The comparator is a lumped constant-c Kelvin-Voigt damper in parallel with the same elastic response, not a separate engine measurement or every distributed viscoelastic model.

Panels (c,f) are separate predictions of Eq. (5), using $\beta \in \{ 0 , 0 . 5 , 1 , 2 \}$ , fixed $k _ { h } = 1 0 ^ { 8 } \mathrm { N } / \mathrm { m } ^ { 3 }$ , and ℓ = 1 mm. These illustrative parameters are not fitted to a physical pad. For 41 indentations from 0 to 1 mm, original-mesh signed-distance queries provide local interior depth, and planar quadrature integrates each pressure law. These curves show equal initial local pressure gradients and increasing resistance at greater compression, and are computed offline, independently of the solver preset described in $\ S \mathsf { A } . 2$

Wheel-ground contact. Figure 6 uses a simplified differential-drive rig with the base dimensions of FRIDAY rather than the complete robot. Two cylindrical wheels hang from a box chassis on preloaded vertical springs, and four freely rotating swivel casters, two of them on a rear pivot, share the load, for a total mass of about 135 kg. Wheel radius is 70 mm and half-track b is 262.2 mm. Wheel density is held at $1 2 0 0 \mathrm { k g / m ^ { 3 } }$ as width varies, so wheel mass varies with width. Each rollout lasts 4 s at a 2 ms step with at most 32 solver iterations and warm starting, $k _ { h } = 1 0 ^ { 1 0 } \mathrm { N } / \mathrm { m } ^ { 3 }$ , a 1 mm SDF voxel target, 0.5 mm quadrature spacing, and $\mu = 0 . 6$ for the wheels, casters, and ground. Both contact methods use the same rig and velocity-servo settings at each width. The rim-point baseline removes the axial midpoint from the current analytic contact manifold, so it is not a comparison against the current midpoint-enhanced manifold.

After 1 s of settling, opposite wheel-velocity targets command an ideal in-place yaw rate of 1 rad/s. Over the final second, unwrapped chassis yaw gives the computed yaw rate, normalized by the mean kinematic rate computed from achieved wheel speeds. Widths are 10, 30, 53, 70, 88, and 120 mm, with one rollout per method and width.

The pressure illustration in panel (b) is a separate static probe on the authored profile of the FRIDAY drive wheel at the indentation estimated for a 300 N load, and its samples before contact reduction integrate to 266.6 N. The visual mesh is narrower than the 88 mm collision cylinder of the robot asset, which the width sweep in panel (c) uses.

Comparison with MuJoCo. We compare CDMM with Newton’s MuJoCo backend on FRIDAY lifting its own battery in two configurations. The first keeps each backend’s default contact treatment across time steps (Table 24). The second is the stiff-contact case of Figure 7. In the first configuration, both backends start from the same grasp state and follow the same joint position targets through the same controller, with MuJoCo’s native collision detection disabled and contacts generated at each substep from the current state by the shared reduced SDF-hydroelastic pipeline. The 8.2 s path raises the wrist by about 25 cm and is followed by a 2 s hold at the final target. The hand and battery use $k _ { h } = 1 0 ^ { 1 2 } \mathrm { N } / \mathrm { m } ^ { 3 } , \mu = 1 . 0$ , SDF voxels of 1.61–2.00 mm for the fingers and 1.91–1.92 mm for the battery, default contact reduction, and warm starting, with CDMM using at most 32 iterations and $\alpha = 2 0 \mathrm { s } / $ m and MuJoCo using at most 15 or 100 iterations, and all recorded model parameters match between backends.

CDMM uses near-rigid regularization and Hunt-Crossley damping. MuJoCo uses positive-format contact parameters, whose time constants are floored at twice the time step, with an elliptic friction cone and an impedance ratio of 1000. The two contact laws therefore differ, so these results compare the backends as configured and do not isolate the numerical solver. We count a rollout as lifted when the final battery rise exceeds 10 cm and as jittering when the root-mean-square third difference of the battery position, sampled at 60 Hz over the final 1.8 s, exceeds 2 mm. We run five rollouts per setting. Contact generation is not forced to be deterministic, and repeated rollouts differ.

Table 24. Lifted and jittering rollouts out of five per setting, written as lifted / jittering. All 60 rollouts remain finite.
<table><tr><td>Time step</td><td>CDMM, at most 32</td><td>MuJoCo, at most 15</td><td>MuJoCo, at most 100</td></tr><tr><td>1/240s</td><td>5/0</td><td>0/0</td><td>0/0</td></tr><tr><td>1/480s</td><td>5/0</td><td>0/0</td><td>0/0</td></tr><tr><td>1/960 s</td><td>5/0</td><td>0/0</td><td>0/0</td></tr><tr><td>1/1920s</td><td>5/0</td><td>1/1</td><td>0/0</td></tr></table>

CDMM lifts the battery in all 20 rollouts, and none of them jitters. MuJoCo lifts it in one of 40 rollouts, at $h = 1 / 1 9 2 0 \mathsf { s }$ , with a cap of 15 iterations, and that battery jitters with a value of 5.5 mm. The largest value among the CDMM rollouts is 1.3 mm, and the threshold is set after inspecting these results. Raising the MuJoCo iteration cap from 15 to 100 does not produce a lift in these rollouts.

Figure 7 uses a separate configuration at $h = 1 / 2 4 0 0 s$ with an iteration cap of 32 in both backends. It starts from a different input with a 1 s hold, a commanded 15 cm wrist lift over 0.35 s, and a 3 s final hold. To reduce differences in the normal-contact model, we disable near-rigid regularization in both backends and assign each patch contact to MuJoCo through direct-format parameters built from its patch stiffness and Hunt-Crossley damping. These parameters bypass the time-constant floor, and the mapping relies on a reference-mass approximation, so friction, integration, and the articulated contact response remain backend-specific. CDMM completes the 4.35 s rollout with a final battery rise of about 13.3 cm. MuJoCo raises the battery by about 5.8 cm before the hand and battery poses change abruptly near $t = 1 . 1 8 5 \mathsf { s } .$ , and the first nonfinite state occurs at t = 1.18875 s. This is one rollout per backend, chosen after a parameter search, and the outcome depends on stiffness, contact mapping, and time step. These observations characterize the stated configurations rather than general solver superiority, and we make no runtime claim.

## A.4 Magnetic tactile response kernel

The tactile model adopts spatially weighted displacement superposition from HydroShear (Dang et al., 2026). We adapt this formulation to the discrete magnets of the FRIDAY hand using loads reconstructed from hydroelastic contact patches. This subsection specifies the resulting pressure-to-displacement mapping and describes the sensor-side shear history and coupling assumptions.

Pressure-induced displacement. All positions and normals are expressed in the sensor frame. Let q denote the ${ \bf q } _ { m }$ undeformed position of magnet $m ,$ and let $\mathbf { x } _ { i } , \mathbf { n } _ { i } ,$ , and A denote the position, inward unit normal, and area of contact contribution i. Each contribution carries an elastic pressure p on the extracted pressure-equilibrium surface, reconstructed from its depth and stiffness with the linear law of Eq. (4), giving a normal load $F _ { i } = p _ { i } A _ { i }$ .

The tangential offset from contact i to magnet m and its Gaussian weight are

$$
\mathbf { r } _ { m i } = ( I - \mathbf { n } _ { i } \mathbf { n } _ { i } ^ { \mathsf { T } } ) ( \mathbf { q } _ { m } - \mathbf { x } _ { i } ) , \qquad w _ { m i } = \exp \left( - { \frac { \| \mathbf { r } _ { m i } \| ^ { 2 } } { 2 \ell ^ { 2 } } } \right) ,\tag{26}
$$

where I is the three-dimensional identity matrix and $\ell > 0$ sets the lateral response scale. The vector response per unit normal load is

$$
\begin{array} { r l } & { \mathbf { G } _ { m i } = e ^ { - d _ { m } / \lambda } \Bigg [ c _ { n } w _ { m i } \mathbf { n } _ { i } + c _ { t } \left( \varrho e ^ { 1 / 2 } \frac { \mathbf { r } _ { m i } } { \ell } w _ { m i } + ( 1 - \varrho ) \frac { 2 \ell \mathbf { r } _ { m i } } { \| \mathbf { r } _ { m i } \| ^ { 2 } + \ell ^ { 2 } } \right) \Bigg ] , } \\ & { \mathbf { u } _ { m } ^ { \mathrm { p r e s s } } = \sum _ { i } \mathbf { G } _ { m i } F _ { i } , } \end{array}\tag{27}
$$

where $c _ { n }$ and $^ { c _ { t } , }$ in $\mathrm { m } / \mathrm { N } ,$ , control normal and radial compliance, and the magnet depth $d _ { m }$ and attenuation length λ determine depth attenuation. The parameter $\varrho \in [ 0 , 1 ]$ blends the local Gaussian and longer-range radial responses. This phenomenological kernel does not resolve volumetric elasticity or deformation caused by rigid magnet inclusions.

Sensor-side queries and shear history. Shear deformation is tracked at query points on the sensor surface. At contact onset, each query stores its position in the contacted object’s local frame as a reference. As the object moves, the reference’s displacement relative to the sensor surface is projected onto the local tangent plane and spatially weighted to estimate magnet displacement.

Each query maintains its own contact history, which resets upon contact loss. An optional Coulomb limit with its own friction coefficient clips the accumulated tangential displacement and moves the reference accordingly. It is off by default, so a query sticks until it loses contact.

Table 25. Pressure-response kernel parameters and their default values. Lengths are reported in SI units.
<table><tr><td>Parameter</td><td>Meaning</td><td>Default</td></tr><tr><td> $c _ { n }$ </td><td>Normal compliance</td><td> $1 0 ^ { - 5 } \mathrm { m } / \mathrm { N }$ </td></tr><tr><td> $c _ { t }$ </td><td>Radial compliance</td><td> $1 0 ^ { - 5 } \mathrm { m } / \mathrm { N }$ </td></tr><tr><td> $\varrho$ </td><td>Local/global radial blend</td><td>0.5</td></tr><tr><td> $\ell$ </td><td>Lateral response scale</td><td> $2 \times 1 0 ^ { - 3 } \mathrm { m }$ </td></tr><tr><td> $\lambda$ </td><td>Depth attenuation length</td><td> $1 0 ^ { - 3 } \mathrm { m }$ </td></tr><tr><td> $d _ { m }$ </td><td>Configured magnet depth</td><td> $1 . 5 \times 1 0 ^ { - 3 } \mathrm { m }$ </td></tr></table>

Coupling to contact dynamics. The tactile model reconstructs elastic loading from collision geometry rather than using the solver’s final contact reactions. Damping, solver friction, near-rigid regularization, and nonlinear pressure stiffening therefore do not directly enter the sensor response, although they can affect it through the resulting motion and contact geometry.

Sensor deformation is not fed back into the dynamics.

Parameters and calibration scope. Table 25 lists the default parameters of Eq. (27). These are initialization values, not experimentally calibrated benchmark settings. No calibration result or accuracy claim is implied by the defaults.

Same-load probes. Figure 9 uses the right-index distal pad of Appendix A.3, isolated from the articulated hand, with the linear pressure law at $k _ { h } = 1 0 ^ { 8 } \mathrm { N } / \mathrm { m } ^ { 3 }$ and the default parameters of Table 25. The magnet array is centered across the pad width as in the FRIDAY hand design, with a 5 mm pitch. Each probe is a rigid body with $\bar { k } _ { h } = 1 0 ^ { \mathrm { i } 2 } \mathrm { N } / \mathrm { m } ^ { 3 }$ and a flat face. The first is a 12 by 10 mm face. The second is a 3 by 10 mm ridge, offset 4.5 mm toward the fingertip and 1.5 mm across the pad. The third is a pair of 4 by 4 mm posts, 9 mm apart and offset 1.5 mm across the pad.

Probes are posed kinematically and no solver step is taken. Collision and sensor evaluation use a 0.125 mm SDF grid without contact reduction, and bisection on the normal offset brings the net normal load to within 0.2% of 1 N. The resulting indentations are 0.14, 0.44, and 0.38 mm, below the 1.5 mm magnet depth. Direct quadrature of the pad height field agrees with these loads within 2%. Faces on the probe sides raise the total face load above the net normal load by about 10% for the ridge and the posts.

The cosine similarity of the Hall readings between probe pairs is 0.67 to 0.80, against 1 for a model driven by net force alone. It changes by at most 0.01 on a 0.25 mm grid. These values depend on probe placement, and moving the ridge 4.5 mm changes them by more than 0.1, so they illustrate the response and do not measure the discrimination of the sensor. The field magnitudes are not meaningful under default parameters. The study does not involve hardware measurements.

## B Actuator model and integration details

This appendix gives the local state updates and parameter definitions of the actuator model in $\ S 3 . 2$ . The equations describe reduced actuator models, not resolved electromagnetic fields or hydraulic circuits. Unless stated otherwise, geometry and driving quantities are sampled at the beginning of a substep of duration h.

## B.1 Drive and transmission models

Electrical and thermal states. The winding resistance is evaluated from the current temperature as $R ^ { k } = \operatorname* { m a x } ( R _ { 0 } [ 1 +$ $\alpha _ { R } ( T ^ { k } - T _ { 0 } ) ] , R _ { \mathrm { m i n } } )$ , where $R _ { 0 }$ is the resistance at the reference temperature $T _ { 0 } ,$ α is its temperature coefficient, and $R _ { \operatorname* { m i n } } > 0$ prevents a degenerate circuit. After control and voltage limiting determine $V ^ { k }$ , the current update is

$$
\begin{array} { r } { \begin{array} { c } { i _ { \infty } ^ { k } = \frac { V ^ { k } - k _ { e } \omega _ { m } ^ { k } } { R ^ { k } } , } \\ { \tilde { i } ^ { k + 1 } = i _ { \infty } ^ { k } + ( i ^ { k } - i _ { \infty } ^ { k } ) e ^ { - R ^ { k } h / L } , } \\ { i ^ { k + 1 } = \mathrm { c l i p } ( \tilde { i } ^ { k + 1 } , i _ { \mathrm { m i n } } , i _ { \mathrm { m a x } } ) . } \end{array} } \end{array}\tag{28}
$$

Here, $i _ { \mathrm { m i n } }$ and $i _ { \mathrm { m a x } }$ are the current limits. For $L = 0$ , the pre-limit current is $i _ { \infty } ^ { k }$ . The exponential update is exact for the frozen linear circuit before current clipping, not for the complete coupled motion and controller response. In the current-commanded path, the requested voltage includes back-EMF compensation, whereas the direct-voltage controller does not cancel back-EMF. Both paths remain subject to the available voltage range.

With copper loss $P _ { \mathrm { C u } } ^ { k } = ( i ^ { k + 1 } ) ^ { 2 } R ^ { k }$ held fixed, the lumped winding temperature advances as

$$
\begin{array} { r } { T ^ { k + 1 } = T _ { \infty } ^ { k } + ( T ^ { k } - T _ { \infty } ^ { k } ) e ^ { - h / ( R _ { \mathrm { t h } } C _ { \mathrm { t h } } ) } , \qquad T _ { \infty } ^ { k } = T _ { \mathrm { a m b } } + R _ { \mathrm { t h } } P _ { \mathrm { C u } } ^ { k } . } \end{array}\tag{29}
$$

Here, $R _ { \mathrm { t h } }$ and $C _ { \mathrm { t h } }$ are thermal resistance and capacity, and $T _ { \mathrm { a m b } }$ is the ambient temperature. The new temperature affects resistance on the following substep. Reported terminal power is the signed quantity $\dot { V ^ { k } } i ^ { k + 1 }$ . It is not a battery-energy mode or a substep-integrated energy balance.

The control loops advance at each simulation substep. Separate firmware sampling rates and communication jitter are not modeled. Battery discharge, automatic bus-voltage sag, and thermal protection logic are outside the electrical model.

Transmission geometry and reflected inertia. Constant gearing uses $\phi ( q ) = n q$ with gear ratio $n .$ For the offset slidercrank, let a be crank radius, b connecting-rod length, e slider-axis offset, c motor rotation per slider travel, and $\epsilon \in \{ - 1 , 1 \}$ the selected geometric branch. With $\theta = q + q _ { 0 }$ for a joint offset $q _ { 0 }$ , the motor position is

$$
\phi ( q ) = c \left[ a \sin \theta + \epsilon \sqrt { b ^ { 2 } - ( e - a \cos \theta ) ^ { 2 } } \right] .\tag{30}
$$

The local ratio $r = \phi ^ { \prime } ( q )$ maps velocity, while the forward-drive efficiency η scales the transmitted motor effort $\tau _ { m }$ to the joint effort $\tau _ { q } = \eta r \tau _ { m }$ before any unilateral output restriction. This scalar efficiency is not a detailed direction-dependent loss model for forward and backdriven operation.

The coupled rotor kinetic energy is $\textstyle { \frac { 1 } { 2 } } I _ { r } r ( q ) ^ { 2 } { \dot { q } } ^ { 2 }$ . Its contribution to the joint equation therefore contains both $I _ { r } r ^ { 2 } \ddot { q }$ and $I _ { r } r r ^ { \prime } \dot { q } ^ { 2 }$ The first enters the mass matrix and the second enters the inertial bias, without electrical clipping or efficiency scaling. Invalid slider-crank geometry disables transmission output, and the validity check is not a regularization of near-singular transmission ratios. Joint ranges should avoid geometric singularities.

The transmission is kinematic, so backlash and elastic deformation of gears or linkages are not resolved.

LuGre friction. For motor speed $\omega _ { m }$ , the deterministic LuGre model uses a bristle deflection z and a Stribeck torque envelope $g \dot { . }$

$$
\begin{array} { c } { g ( \omega _ { m } ) = \tau _ { c } + ( \tau _ { s } - \tau _ { c } ) e ^ { - ( \omega _ { m } / \omega _ { s } ) ^ { 2 } } , } \\ { \dot { z } = \omega _ { m } - \lambda z , \qquad \lambda = \frac { \sigma _ { 0 } | \omega _ { m } | } { g ( \omega _ { m } ) } , } \\ { \tau _ { f } = \sigma _ { 0 } z + \sigma _ { 1 } \dot { z } + B \omega _ { m } . } \end{array}\tag{31}
$$

Here, $\sigma _ { 0 }$ and $\sigma _ { 1 }$ are bristle stiffness and damping, B is viscous damping, and $\tau _ { s } , \tau _ { c }$ , and $\omega _ { s }$ set the static level, sliding level, and transition speed. A positive floor is applied to g. At fixed speed, the state update is

$$
z ^ { k + 1 } = e ^ { - \lambda h } z ^ { k } + \frac { 1 - e ^ { - \lambda h } } { \lambda } \omega _ { m } ,\tag{32}
$$

with the continuous limit $z ^ { k + 1 } = z ^ { k } + h \omega _ { m }$ as $\lambda  0$

Low-speed variability. The optional variability models use independently seeded, temporally correlated processes. Each normalized Ornstein-Uhlenbeck state follows

$$
\begin{array} { r } { x ^ { k + 1 } = a _ { \mathrm { O U } } x ^ { k } + \sqrt { 1 - a _ { \mathrm { O U } } ^ { 2 } } \epsilon _ { \mathrm { O U } } ^ { k } , \qquad a _ { \mathrm { O U } } = e ^ { - h / \tau _ { \mathrm { n o i s e } } } , \qquad \epsilon _ { \mathrm { O U } } ^ { k } \sim \mathcal { N } ( 0 , 1 ) , } \end{array}\tag{33}
$$

where $\tau _ { \mathrm { n o i s e } }$ is the correlation time. The normalized process has unit variance at stationarity, and resetting it to zero does not start it in its stationary distribution.

For friction variation, the log-scale amplitude depends on joint speed $v = \dot { q }$ through $\ell ( v ) = \ell _ { 0 } [ \rho + ( 1 - \rho ) e ^ { - ( v / v _ { f } ) ^ { 2 } } ]$ . The scale applied during friction-force and implicit-coefficient evaluation is

$$
\begin{array} { r } { s _ { f } = \mathrm { c l i p } \left( \exp \left[ \ell ( v ) x ^ { k + 1 } - \frac { 1 } { 2 } \ell ( v ) ^ { 2 } \right] , s _ { \mathrm { m i n } } , s _ { \mathrm { m a x } } \right) . } \end{array}\tag{34}
$$

It jointly multiplies $\sigma _ { 0 } , \sigma _ { 1 } , \tau _ { c }$ , and $\tau _ { s } .$ , preserving their ratios. The parameters $\ell _ { 0 } , v _ { f }$ , and $\rho$ control the low-speed amplitude, velocity scale, and high-speed fraction, and $s _ { \mathrm { m i n } }$ and $s _ { \mathrm { m a x } }$ bound the scale. The exponential correction gives unit mean before clipping for a stationary normalized Gaussian state and a fixed envelope, but clipping and velocity-state dependence need not preserve this mean.

A separate process y produces a joint-effort disturbance $d _ { \mathrm { e f f o r t } } = \sigma _ { d } e ^ { - \left( v / v _ { d } \right) ^ { 2 } } y ^ { k + 1 }$ with amplitude $\sigma _ { d }$ and velocity scale $v _ { d } ,$ which is subtracted from the actuator output. Its amplitude, correlation time, and seed are set independently. Unlike passive friction, this additive disturbance can inject mechanical energy. Both extensions are empirical descriptions of low-speed variability rather than microscopic contact models.

Reduced single-acting response. Let $s _ { \mathrm { d r i v e } } \in \{ - 1 , 1 \}$ specify the permitted drive direction and $u ^ { k }$ be the joint-side drive effort before the unilateral hydraulic response. A nonnegative chamber-effort state H is updated and applied as

$$
\begin{array} { r l } & { H ^ { k + 1 } = a _ { h } H ^ { k } + ( 1 - a _ { h } ) [ s _ { \mathrm { d r i v e } } u ^ { k } ] _ { + } , \qquad a _ { h } = e ^ { - h / \tau _ { h } } , } \\ & { } \\ & { \tau _ { \mathrm { d r i v e } } = s _ { \mathrm { d r i v e } } [ H ^ { k + 1 } - D _ { h } s _ { \mathrm { d r i v e } } \dot { q } ^ { k } ] _ { + } . } \end{array}\tag{35}
$$

The time constant $\tau _ { h }$ represents effort buildup and decay, and $D _ { h }$ sets joint-side flow damping. The passive return spring and damper are added independently of this unilateral drive restriction. The implicit solve retains the flow-damping term across the zero-output boundary as described in § 3.2.2. Eq. (35) defines the evaluated output, not an exact implicit solution of that boundary.

The model is a joint-effort-level approximation, not a resolved hydraulic circuit with explicit valves, chamber volumes, or fluid compressibility.

## B.2 State integration and dynamics coupling

Implicit coupling approximations. Transmission geometry and local coefficients are held fixed during each substep, so the coupling of § 3.2.2 does not solve all electrical, thermal, and mechanical states simultaneously. The coefficients retained through saturation are not the exact derivatives of the clipped output law, and the unilateral hydraulic transition is regularized without locating its event time. Controller anti-windup addresses accumulated tracking error, while the implicit terms address the numerical response within the substep. The split update does not remove temporal-discretization error, so rapid changes in transmission geometry, saturation, or friction still require adequate substep resolution.

Bristle stiffness in the articulated solve. During the articulated solve, the implicit motor-side bristle stiffness uses the nonnegative approximation

$$
K _ { z } = \sigma _ { 0 } \mathrm { c l i p } \left( e ^ { - \lambda h } \left[ 1 - \frac { \mathrm { s g n } ( \omega _ { m } ) \sigma _ { 0 } z ^ { k } } { g ( \omega _ { m } ) } \right] , 0 , 2 \right) .\tag{36}
$$

This approximation holds the Stribeck envelope fixed and is not a full derivative of the nonlinear friction law. After the solve, Eq. (32) is evaluated from the old bristle state using the solved velocity and the frozen transmission ratio. The pre-solve provisional state is not integrated a second time.

## C Asset generation details

This appendix provides additional details for § 4.1.

## C.1 Constraint types

As described in § 4.1.1, the structured set of task constraints $\widehat { \mathcal { R } } _ { \mathrm { t a s k } }$ is instantiated from a fixed vocabulary of constraint types. Table 26 summarizes the seven categories and their representative constraint types. Each type specifies the fields needed to instantiate it, and the resulting constraints are used by the rule-based router $\rho$ for branch selection.

Table 26. Constraint types used by ERIS, grouped into seven categories. Representative types are shown for each category.
<table><tr><td>Category</td><td>Representative constraint types</td><td>Description</td></tr><tr><td>Property</td><td>dimension, part_count, min_thickness, appearance</td><td>Dimensions, counts, wall thickness, and appear- ance</td></tr><tr><td>Access</td><td>open_top, access, graspable</td><td>Openings and manipulation accessibility</td></tr><tr><td>Fit</td><td>fits, clearance, aligns, ...</td><td>Geometric compatibility, fit, and clearance</td></tr><tr><td>Articulation</td><td>dof, actuated, reaches_state, ...</td><td>Relative motion and actuation</td></tr><tr><td>Support</td><td>supports, retains, stable, ...</td><td>Load support, retention, and stability</td></tr><tr><td>Transport</td><td>transports, guides, presents, ..</td><td>Object transport and presentation</td></tr><tr><td>Interaction</td><td>inserts, removes, engages</td><td>Part-to-part interactions</td></tr></table>

## C.2 SDF construction and collider layout

This section provides additional details on the collider verification stage summarized in Table 9 (§ 4.1.3). We describe adaptive SDF authoring and simulator-side validation in HOLIDAY-SIM, the collision-cost model, and bake-verified collider-layout optimization.

Adaptive SDF authoring and validation. Before baking each collider, ERIS applies the resolution-admissibility check to determine which SDF voxel sizes are geometrically admissible. It then authors the SDF parameters used by HOLIDAY-SIM, most importantly the voxel size that determines collision resolution. For each collider $c _ { i } ,$ , ERIS derives an upper bound on voxel size from geometric information in the constructed asset. Let $\ell _ { i } ^ { \mathrm { m i n } }$ denote the minimum side length of the collider’s axis-aligned bounding box, $w _ { i }$ a measured wall thickness when available, and $f _ { i }$ the scale of an explicitly declared collision feature. These quantities constrain the resolution according to the collider’s overall scale, locally thin structures, and declared functional features, respectively. The factors $\kappa _ { \mathrm { s i z e } } , \kappa _ { \mathrm { w a l l } }$ , and $\kappa _ { \mathrm { f e a t } }$ specify the corresponding sampling requirements:

$$
u _ { i } = \operatorname* { m i n } \left\{ \frac { \ell _ { i } ^ { \mathrm { m i n } } } { \kappa _ { \mathrm { s i z e } } } , \frac { w _ { i } } { \kappa _ { \mathrm { w a l l } } } , \frac { f _ { i } } { \kappa _ { \mathrm { f e a t } } } \right\} .\tag{37}
$$

ERIS then initializes the search at the coarsest supported voxel size $\Delta _ { \mathrm { v o x } , i }$ that satisfies $\Delta _ { \mathrm { v o x } , i } \leq u _ { i }$ , subject to any additional simulator- or layout-specific caps, where smaller voxel sizes correspond to finer SDF resolutions. Terms corresponding to unavailable measurements or declarations are omitted, so $u _ { i }$ is determined by the most restrictive applicable geometric constraint. The resolution-admissibility check therefore excludes candidate voxel sizes that are too coarse relative to known geometric scales, without attempting to predict the exact surface produced by SDF discretization.

ERIS therefore performs a second, complementary check after conversion. HOLIDAY-SIM bakes the collider, and the SDF fidelity check runs on the resulting simulation representation. Whereas resolution admissibility evaluates the sampling conditions before baking, SDF fidelity evaluates the geometry that the simulator will actually use. Let $\boldsymbol { S } _ { i }$ denote samples drawn from the source surface, and $\Sigma _ { i }$ denote the resulting SDF isosurface. We define the surface coverage $C _ { i }$ and surface area $A _ { i }$ as:

$$
C _ { i } = \frac { 1 } { | S _ { i } | } \sum _ { x \in S _ { i } } \mathbf { 1 } [ d ( x , \Sigma _ { i } ) < 2 \Delta _ { \mathrm { v o x } , i } ] , \qquad A _ { i } = \mathrm { A r e a } ( \Sigma _ { i } ) .
$$

Here, $d ( x , \Sigma _ { i } )$ is the distance from x to $\Sigma _ { i }$ . Let $E _ { i }$ denote the number of open boundaries of $\Sigma _ { i } ,$ and let $g _ { i }$ and $n _ { i }$ denote its genus and connected-component count, respectively. A candidate representation passes the SDF-fidelity check when

$$
\frac { C _ { i } } { C _ { i } ^ { \mathrm { r e f } } } \geq \tau _ { \mathrm { c o v } } , \qquad \frac { A _ { i } } { A _ { i } ^ { \mathrm { r e f } } } \geq \tau _ { \mathrm { a r e a } } \qquad ( \mathsf { s u r f a c e r e t e n t i o n } ) ,
$$

$$
E _ { i } \leq \tau _ { \mathrm { o p e n } } ~ ( { \mathrm { b o u n d a r y ~ v a l i d i t y } } ) ,\tag{38}
$$

$$
g _ { i } - g _ { i } ^ { + } \leq \Delta _ { g } , \qquad n _ { i } - n _ { i } ^ { + } \leq \Delta _ { n } \quad ( \mathrm { t o p o l o g y ~ g u a r d } ) .
$$

Coverage and area provide complementary measures of surface retention. $C _ { i }$ detects regions of the source surface that are no longer represented nearby in the baked geometry, whereas $A _ { i }$ detects substantial surface loss that can occur even when the remaining baked surface stays close to the source. Surface collapse can preserve high coverage while reducing area, whereas localized geometric loss can reduce coverage even when the total baked surface area remains large. Open boundaries identify invalid extracted surfaces, while increases in genus and connected-component count detect spurious perforation and fragmentation, respectively. Here, “ref” denotes the finest available reference bake, while $( \cdot ) ^ { + }$ denotes the corresponding quantity at the next finer voxel size. When a candidate fails the SDF-fidelity check, ERIS moves to a more conservative SDF parameterization and rebakes the collider.

Collision-cost model. We now specify the cost term $M _ { \mathrm { S D F } }$ in Eq. (16). A valid collision representation may still be unnecessarily expensive for large-scale simulation. Let $N = | \Pi |$ denote the number of colliders in a layout, let $B _ { i }$ denote the block count of collider i, and let $B _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { i } B _ { i }$ . For a reference number D of dynamic bodies, ERIS estimates the collision-memory contribution of the asset with a simulator-derived model:

$$
M _ { \mathrm { S D F } } = a _ { 1 } \sum _ { i } B _ { i } + a _ { 2 } D N B _ { \mathrm { m a x } } + a _ { 3 } D N + a _ { 4 } N .\tag{39}
$$

Here, $a _ { 1 }$ to $a _ { 4 }$ are coefficients derived from HOLIDAY-SIM’s collision-memory model. The terms capture different sources of collision cost: finer SDF resolutions increase the aggregate block count $\textstyle \sum _ { i } B _ { i }$ , retaining many independent colliders increases $N ,$ and merging parts can increase $B _ { \mathrm { m a x } }$ by placing a spatially larger region into a single collider. Consequently, SDF resolution and collider partition cannot generally be optimized independently.

Collider-layout optimization. For multipart code-native assets, Π in Eq. (16) introduces an additional optimization variable. Keeping every part separate can increase collider-dependent memory overhead, while merging parts too aggressively can cause small or thin source parts to be distorted or partially absorbed by the discretized union. ERIS therefore searches for lower-cost groupings of eligible static parts and subjects the selected merged layout to a separate layout-preservation check after baking.

Moving parts, conveying surfaces, parts carrying explicitly declared collision features, non-closed geometry, and parts belonging to different rigid bodies are excluded from merging. The remaining eligible parts form a graph in which edges connect touching parts. For each candidate grouping, ERIS reapplies the resolution-admissibility rule to the geometry of the union and recomputes its block count, since merging changes both the spatial extent and geometric scales of the collider being baked.

ERIS greedily considers unions of adjacent groups under several deterministic orderings. A merge is considered only when the resulting union forms a valid connected solid, satisfies the per-collider block constraint

$$
B _ { i } \leq B _ { \mathrm { c a p } } ,\tag{40}
$$

and reduces the predicted collision cost in Eq. (39). The block cap prevents a locally cost-reducing merge from producing an individually oversized collider. The original per-part layout is retained as a candidate throughout the search, so reducing collider count alone is never sufficient to select a merge.

Predicted cost alone is insufficient because merging changes the geometry that is ultimately discretized. ERIS therefore bakes the selected layout and applies the layout-preservation check to each source part participating in a merged collider. For each source part $j ,$ area-weighted surface samples from both the source union and the baked collider are attributed to the nearest source part whose surface lies within $2 \Delta _ { \mathrm { v o x } , i }$ and has a sufficiently aligned normal. Let $s _ { j } ^ { \mathrm { s r c } }$ and $s _ { j } ^ { \mathrm { b a k e } }$ denote the resulting surface-sample shares attributed to part $j$ before and after baking. From the attributed samples, we estimate local wall thickness using the first percentile of distances to opposing surfaces, denoted by $w _ { j } ^ { \mathrm { s r c } }$ and $w _ { j } ^ { \mathrm { b a k e } }$ , respectively. We quantify local geometric integrity using

$$
\frac { w _ { j } ^ { \mathrm { b a k e } } } { w _ { j } ^ { \mathrm { s r c } } } \geq \tau _ { \mathrm { w a l l } } , \qquad \frac { s _ { j } ^ { \mathrm { b a k e } } } { s _ { j } ^ { \mathrm { s r c } } } \geq \tau _ { \mathrm { s h a r e } } .\tag{41}
$$

Here, $\tau _ { \mathrm { w a l l } }$ specifies the minimum wall-retention ratio, while $\tau _ { \mathrm { s h a r e } }$ specifies the minimum retained surface-share ratio. The former detects local squeezing or necking, whereas the latter detects parts that are partially erased or absorbed by the merged SDF even when their remaining thickness is preserved. These failures can be hidden by collider-level statistics when the affected part occupies only a small fraction of a merged collider, motivating the part-wise preservation check.

Parts for which the preservation measurement is inconclusive are treated conservatively as failures. Parts that fail the check or yield inconclusive measurements are forced to remain separate, after which the remaining eligible parts are laid out and baked again. If no verified merged layout is obtained within the retry budget, ERIS falls back to the original per-part layout. The retry budget limits repeated baking in the simulator and does not relax the preservation requirements. Algorithm 2 summarizes this bake-verified optimization.

When adaptive SDF construction and collider-layout optimization still cannot produce an acceptable collision representation, the measurements return through the representation-specific revision path of § 4.1.3.

## C.3 Models, budgets, and thresholds

The generators, model settings, and common generation budget used in the ERIS experiments are summarized in Table 27. All branches allow up to $K = 4$ generation rounds, including the initial generation and subsequent revisions. Observed

Algorithm 2. Bake-verified collider-layout optimization   
Input: Eligible static parts A, touching graph G, block cap $B _ { \mathrm { c a p } }$ , retry budget R, greedy orderings O   
Output: Bake-verified layout Π, or the per-part layout $\Pi _ { \mathrm { p a r t } }$   
1: $\mathcal { C } _ { \mathrm { m e r g e } }  \emptyset$ ▷ parts forced to remain separate   
2: for $\boldsymbol { r } = 1 , \ldots , R$ do   
3: $\mathcal { P }  \{ \Pi _ { \mathrm { p a r t } } \}$   
4: for greedy ordering $o \in \mathcal { O }$ do   
5: Π<sub>o</sub> ← GREEDYMERGE(A, G, C<sub>merge</sub>, o, M<sub>SDF</sub>, B<sub>cap</sub>)   
6: ${ \mathcal { P } } \gets { \mathcal { P } } \cup \{ \Pi _ { o } \}$   
7: end for   
8: Π ← arg min ′<sub>∈P</sub> M<sub>SDF</sub>(Π<sup>′</sup>)   
9: Πb ← BAKE(Π) ▷ HOLIDAY-SIM SDF bake   
10: F ← VERIFYPARTS(Π, Π)b ▷ part-wise layout preservation   
11: if F = ∅ then   
12: return Π   
13: end if   
14: $\mathcal { C } _ { \mathrm { m e r g e } }  \mathcal { C } _ { \mathrm { m e r g e } } \cup \mathcal { F }$ ▷ failed or unjudged parts stay separate   
15: end for   
16: return $\Pi _ { \mathrm { p a r t } }$

generation cost and wall-clock time are reported separately in Table 28, since they depend strongly on the structure and interaction complexity of the requested asset.

The fixed hyperparameters used for collider construction and verification are reported in Table 29. Task-specific geometric and interaction tolerances are instantiated from the corresponding task constraints and are therefore not included as globa hyperparameters.

Table 27. Generation configuration and round budgets used by ERIS.
<table><tr><td>Branch</td><td>Generator</td><td>Model setting</td><td>Budget</td></tr><tr><td>Mesh</td><td>FLUX.2-klein-9B → BiRefNet → SAM 3D → PaMO</td><td>pipeline defaults</td><td> $K = 4$ </td></tr><tr><td>Code-Rigid</td><td>executable Blender program</td><td>claude-opus-5-5 (high)</td><td> $K = 4$ </td></tr><tr><td>Code-Articulated</td><td>Blender program + joint declaration</td><td>claude-opus-5-5 (high)</td><td> $K = 4$ </td></tr></table>

Table 28. Observed generation cost and wall-clock time for the code-native ERIS pipeline measured with claude-opus-5-5.
<table><tr><td>Task class</td><td>Wall-clock / unit</td><td>Cost / unit</td></tr><tr><td>Static</td><td>1.9 min</td><td>$1.11</td></tr><tr><td>Insertion</td><td>1.8 min</td><td>$0.80</td></tr><tr><td>Articulated</td><td>12.5 min</td><td>$3.13</td></tr><tr><td>Transport</td><td>21.1 min</td><td>$4.29</td></tr><tr><td>Machinery</td><td>11.2 min</td><td>$2.78</td></tr><tr><td>Overall</td><td>10.1 min</td><td>$2.48</td></tr></table>

Table 29. Fixed hyperparameters used for collider construction and verification.
<table><tr><td>Stage</td><td>Parameters</td><td>Values</td></tr><tr><td rowspan="2">Resolution admissibility</td><td>Ksize, Kwall, Kfeat</td><td>1.25, 1, 2</td></tr><tr><td>Supported voxel sizes</td><td>{48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1} mm</td></tr><tr><td>SDF fidelity</td><td>Tcov, Tarea, Topen  $\Delta _ { g } , \Delta _ { n }$ </td><td>0.99, 0.80, 0 2, 2</td></tr><tr><td rowspan="2">Collider layout</td><td> $B _ { \mathrm { c a p } } .$  Twall, Tshare</td><td>1900, 0.90, 0.25</td></tr><tr><td>Maximum collider-layout optimization rounds</td><td>8</td></tr></table>

## D Scenario generation details

This appendix gives the models, budgets, prompts, protocols, and verification thresholds behind § 4.2, together with the background pipeline, re-rendering, and the post-hoc rubric evaluation. Values are taken from the system’s default parameters and the stored records of the generation runs. Quoted prompts in this and the following appendices are reproduced verbatim, with line breaks adjusted for typesetting.

## D.1 Inverse scene generation

## D.1.1 Models and budgets

Model calls return typed JSON and use read-only file access. Codex-served models use a command-line agent, while Claude models use an adapter described under Controls below (Table 30). The planner first summarizes the request in an outline and then compiles it separately into a scene specification and a task plan. The scene-specification prompt forbids task steps and process state so that the two remain separate. A run stops when the same checks fail a third time or when an identical candidate is resubmitted.

Table 30. Models and budgets for inverse scene generation in Table 18 and Table 19, with settings for asset generation.
<table><tr><td>Stage</td><td>Model (effort)</td><td>Budget and settings</td></tr><tr><td>Inverse scene generation</td><td>gpt-6-astra (high), gpt-6.1-sol (medium), gpt-6-luna (medium), claude-opus-5-5 (high) claude-sonnet-5-5 (medium),</td><td>900 s per call, 2,400 s and 40 calls per run, or 7,200 s and 80 calls with the larger budget, at most 6 scene-specification, 8 plan, and 3 outline rounds, 2 scene critics in fresh contexts, asset generation off unless stated</td></tr><tr><td>Asset generation in the loop</td><td>gemini-4-argon (high) claude-opus-5-5 (high),</td><td>Parametric path of the asset generator, at most 1,800 s per</td></tr><tr><td></td><td>claude-sonnet-5-5 fallback</td><td>request, used only in the last two rows of Table 19</td></tr></table>

The scenes in Figure 16 come from runs in Table 18 and Table 19. The packaging scene in Figure 17 is written by claude-opus-5-5 as a coding agent and reviewed by a person (§ D.2).

## D.1.2 Experimental protocols

Table 31 summarizes the experimental settings alongside full ISG and the human review used for the background study.

Table 31. Input and assessment settings for scene generation. Full ISG starts from a task and workplace. The other settings start from a supplied scene description and omit task-plan generation.
<table><tr><td></td><td>Full ISG</td><td>Scene description</td><td>Human review</td></tr><tr><td>Input</td><td>Task and workplace</td><td>Scene description</td><td>Scene description</td></tr><tr><td>Task-plan stage</td><td>Generated and checked</td><td>Not run</td><td>Not run</td></tr><tr><td>Scene specification</td><td>Agent-authored</td><td>Agent-authored</td><td>Agent-authored</td></tr><tr><td>Deterministic checks</td><td>Plan and scene checks</td><td>Scene checks</td><td>Static verification</td></tr><tr><td>Assessment</td><td>Plan critic and scene critics</td><td>Reference-based fidelity checks</td><td>Human review</td></tr></table>

Specification-generation trials. These trials measure scene-specification writing alone. The input already describes the scene, so no task plan is generated, and each result is scored against a reference specification. Each model writes scene specifications for the three fidelity tiers of § 4.2.1, with three cold-start trials per tier and at most five attempts per trial. Each attempt starts a fresh coding agent that writes a scene specification from the supplied description. The result is scored against the tier’s reference specification on item count, object set, target placement, alignment, an empty container where the tier requires one, catalog asset identity, position within 3 cm, support contact within 1 cm and penetration within 5 mm, and overlap. After a failed attempt, the next agent receives the previous specification and a critic report that names each failed criterion, the object involved, and the kind or direction of the error, without reference coordinates. The reference specifications follow the rubric of each tier and meet every criterion when scored against themselves. Agents run in a sandbox that is read-only except for a working directory per attempt, the reference specifications are hidden, and every attempt is checked for reads of benchmark definitions.

## Example critic report after a failed attempt (verbatim excerpt)

position: box\_left: placed moderately off the expected place; it should move toward the viewer’s right (+Y).   
contact: apple\_in\_left\_box: sunk into its support (penetrating it moderately).

Full ISG trials. Each model runs every stage of inverse scene generation on the same ten tasks with a 2,400 s and 40-call budget per run and asset generation disabled, so that asset supply is held fixed across generation models. Each task is written together with its workplace as one request, and the ten are every third entry of an industrial corpus of 15 atomic and 15 long-horizon tasks. The atomic tasks are CNC chuck exchange, adhesive bead application, mixed-SKU case packing, go/no-go gauging with reject, and battery-cell tray loading. The long-horizon tasks are a mixed-part CNC tending shift, secondary packaging line clearance, injection-molding machine tending, a preventive-maintenance route, and a multi-station pump build. We report the number of runs with an accepted plan, with at least one candidate that passes every deterministic check, and with an accepted scene. Table 32 gives the validity and cost of these runs.

Bottleneck removal. Table 19 repeats the full ISG protocol with gpt-6.1-sol, the Codex-served model with the most accepted plans in Table 18, adding one change per condition. The conditions add a larger budget of 7,200 s and 80 calls, then asset generation, and then a stated affordance format. With asset generation, a missing asset triggers the parametric path of the asset generator described above, and each run writes its new assets to its own scratch copy of the catalog, so runs cannot reuse each other’s assets. The asset-generation condition adds to the supplier prompt the evidence fields that the acceptance check reads. The last condition further requires that the words of each required affordance appear in the description or tags of a registered asset, because acceptance matches them literally. An asset counts as accepted only after it passes geometry, conversion, mass, collision, and resting checks in simulation, in addition to this match. Runs whose asset supplier reaches a usage limit of its model provider, two in the asset-generation condition and one in the last condition, are repeated once in a fresh catalog.

Critic swap. For Figure 15, we take the last fully checked candidate of every run in Table 18, 25 in total, and re-judge each with gpt-6.1-sol (medium), claude-opus-5-5 (high), and claude-sonnet-5-5 (medium). Each critic receives, in two fresh contexts, the prompt of the original run rebuilt from its stored plan and candidate, and accepts only if both return a pass with a score of at least 0.8 and no open concern. A stand-in asset is one whose catalog category, read from the asset evidence of the candidate, differs from the category of the planned entity. A critic counts as blocking a stand-in asset when it raises that asset as a concern in at least one of its two passes. The critics see the same evidence and disagree on how much category fidelity matters, as the following excerpts show.

## Three critics on the same raw billet (verbatim excerpts)

claude-sonnet-5-5, pass. Optional, not a defect: the billet asset is catalogued as a toy yellow cylinder [. . . ] and is scaled 2x. Its geometry fits a raw cylindrical billet, so this is acceptable at category fidelity.

claude-opus-5-5, revise. Because the catalog evidence contradicts the industrial category the plan requires [. . . ], the scene shows a toy block standing in for machining stock.

gpt-6.1-sol, revise. raw\_billet uses a cataloged yellow toy building block with a matte finish. Scaling it by 2 does not make it a raw steel billet.

Shared critic. Table 17 repeats the full ISG protocol for gpt-6.1-sol and claude-sonnet-5-5 with every plan and scene critic call served by claude-opus-5-5 (high) in a fresh context, with unchanged prompts and schemas, while all other calls stay on the planner model. Across the 20 runs, all 104 critic calls are served by claude-opus-5-5 and all 299 other calls by the planner model.

Controls. The catalog, thresholds, tasks, and budgets are fixed across models. In the generation runs, critics use the planner’s model except in the shared-critic experiment. Claude models run the same procedure through an adapter that replaces the command-line agent and reads files with read-only search tools instead of a read-only shell. The adapter checks each structured output against the schema after generation, whereas the Codex-served models receive the schema as a strict output format. gemini-4-argon runs the same procedure through its own command-line agent with read-only file tools and its thinking level set to high, receives the schema in the prompt with the same check after generation, and executes one run at a time. One claude-opus-5-5 run ends when its output violates a cross-field rule of the schema, and it counts as a failed run.

Table 32. Validity and cost of inverse scene generation over the ten runs of Table 18. Candidates are proposed scene specifications, with the number that pass static verification and the number that pass every deterministic check. Calls and minutes are per accepted scene. Dashes mark models without an accepted scene.
<table><tr><td>Model (effort)</td><td>Candidates</td><td>Static</td><td>All checks</td><td>Calls</td><td>Min</td></tr><tr><td>gpt-6-astra (high)</td><td>1</td><td>0</td><td>0</td><td>一</td><td>一</td></tr><tr><td>gpt-6.1-sol (medium)</td><td>14</td><td>10</td><td>8</td><td>186</td><td>274</td></tr><tr><td>gpt-6-luna (medium)</td><td>11</td><td>6</td><td>6</td><td>一</td><td>一</td></tr><tr><td>claude-opus-5-5 (high)</td><td>24</td><td>21</td><td>16</td><td>105</td><td>147</td></tr><tr><td>claude-sonnet-5-5 (medium)</td><td>40</td><td>34</td><td>31</td><td>28</td><td>17</td></tr><tr><td>gemini-4-argon (high)</td><td>31</td><td>17</td><td>8</td><td>32</td><td>37</td></tr></table>

## D.2 Background generation

The procedure in Algorithm 3 combines background generation, post-processing, work-cell placement, and verification. Table 33 lists the models of the proprietary and open-source approaches compared in § 4.2.2. The presets organize the factors of Table 34 into plausible facility types and allow variation within each preset. The presets cover customer industries that IFR tracks for industrial robots, namely electronics, automotive, metal and machinery, and food and beverage (International Federation of Robotics, 2025). The factors vary lighting, surface appearance, and surrounding objects because visual changes can affect manipulation-policy performance (Xie et al., 2024a). Figure 22 shows a generated background in use, and Figure 23 shows one that fails because it places people and equipment where the work cell must go.

Table 33. Background-generation models. gpt-5.6-sol was the Codex default model at the time of the background study.
<table><tr><td>Stage</td><td>Model</td><td>Parameters</td></tr><tr><td colspan="3">Shared</td></tr><tr><td>Reference image</td><td>gemini-3.1-flash-image</td><td>Optional, 16:9, 1K, at most three attempts, not used in the paired comparison</td></tr><tr><td>Image and view checks</td><td>gpt-5.6-sol (medium)</td><td>Panorama with four 90° crops, five renders of the splat</td></tr><tr><td>Rubric judge</td><td>gpt-5.6-sol (medium)</td><td>Nine matched frames for each background in the paired comparison, four criteria scored 1 to 5, three shuffled repeats, pairwise comparison in both orders</td></tr><tr><td colspan="3">Proprietary</td></tr><tr><td>Background</td><td>marble-1.1-plus</td><td>Image input, seed 17, full resolution, 4.32 M Gaussians</td></tr><tr><td>Open-source</td><td></td><td></td></tr><tr><td>Panorama</td><td>Matrix-3D image-to-panorama (Yang et al., 2025), FLUX.1-Fill-dev, WorldGen LoRA (Xie, 2025)</td><td>2048 × 1024, 50 steps, guidance 30, 100° field of view, seed 7341</td></tr><tr><td>Multi-view</td><td>HY-World 2.0 WorldStereo</td><td>Three views, forward, left, and right paths, 8 kept and 1 rejected, 262 cameras</td></tr><tr><td>3DGS</td><td>HY-World 2.0 trainer</td><td>4,000 steps, seed 29211, 10⁶ initial points, degree-0 color, depth and normal losses, 3.78 M Gaussians, scale 0.631</td></tr><tr><td colspan="3">Rendering</td></tr><tr><td>Robot camera</td><td>Newton tiled camera</td><td>Fast Gaussian mode, transmittance 0.49, at most 20 hits</td></tr><tr><td>Figures and checks</td><td>RTX path</td><td>per ray 4 samples per pixel, denoising, 64 warm-up frames</td></tr></table>

Algorithm 3. Background generation and placement   
Input: Preset or prompt c, a 3D scene reconstruction pipeline, footprint samples F, recorded work-cell rollout r with camera   
configuration, attempts $R _ { \mathrm { b g } }$   
Output: Placed splat $( \mathcal { G } , \theta ^ { \star } )$ , or the last failed check   
A failed check ends the attempt at its stage. Stages are ordered Panorama, Lift, PostProcess.   
1: s ← PANORAMA ▷ stage to rerun   
2: for $k = 1 , \ldots , R _ { \mathrm { b g } }$ do   
3: if s = PANORAMA then u ← PROMPT(c); I ← PANORAMA(u) ▷ checks: requirements, image   
4: if s ≤ LIFT then $\mathcal { G }  \mathsf { L I F T } ( I )$ ▷ 3D scene reconstruction   
5: G ← POSTPROCESS(G) ▷ check: background views   
6: $\theta ^ { \star } \gets \mathsf { P L A C E W o R K C E L L } ( \mathcal { G } , \mathcal { F } )$ ▷ footprint clearance (§ D.2)   
7: VERIFYINTEGRATEDVIEWS(G, θ<sup>⋆</sup>, r) ▷ checks: coverage, stability   
8: if every check passes then return $( \mathcal { G } , \theta ^ { \star } )$   
9: f ← the first failed check; s ← ROUTE(f) ▷ alignment or outliers rerun PostProcess. Inconsistent views rerun Lift with   
new input. A failed image check, no free floor, or missing coverage rerun Panorama with a new source. A stability failure   
fixes the renderer instead   
10: if f has failed twice with the same input then break ▷ the candidate is closed   
11: end for   
12: return the last failed check

![](images/9e91728091baafe5ff2a0435e84a97afd7f33ffed217c3f9b001a3f400822f4f.jpg)  
(a) Generated 360<sup>◦</sup> panorama (equirectangular)

![](images/2c8058224e811a64394c6e64102a378e77b21bc5359029b2d0df3d9d80ce5115.jpg)  
(b) FRIDAY at a shelf rack inside the splat of (a)  
Figure 22. A generated background in use. (a) Panorama produced by the proprietary 3D world generator, an enclosed hal with an empty central floor as the prompt requires. (b) FRIDAY inside the reconstructed splat of (a), replaying a recorded box-stacking motion. The splat is rendered by the tiled camera and the simulated racks, cartons, and robot by ray tracing, and the two are composited.

Table 34. Design space of background generation: 17 factors, with the number of options in parentheses. Presets: modern general, electronics, food, cleanroom, automotive, machining, logistics hybrid, and small automated.
<table><tr><td>Group</td><td>Factors</td></tr><tr><td>Structure</td><td>Ceiling (5), floor (6), wall (6), window (5), door (5)</td></tr><tr><td>Lighting</td><td>Lighting (5), ceiling equipment (4)</td></tr><tr><td>Perimeter</td><td>Conveyor (5), shelving (5), cart (4), boxes (5), production equipment (6), safety (5)</td></tr><tr><td>Arrangement</td><td>Layout (5), density (3), condition (5), palette (7)</td></tr></table>

Post-processing a generated splat. We estimate the room boundaries as a six-plane enclosure (Table 35 and Figure 24). Floor and ceiling are fitted by angle-constrained RANSAC (Fischler and Bolles, 1981). Each wall is found by scanning inward and measuring, at each scan position $^ { c , }$ the fraction of the cross-section that Gaussians block, $\begin{array} { r } { B ( c ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } ( 1 - e ^ { - \tau _ { k } ( c ) } ) } \end{array}$ over K cross-section samples with optical depth $\tau _ { k } ( c )$ , so that large, opaque Gaussians count more than small, faint ones. A Gaussian is removed when its one-sigma extent lies beyond face j by more than a margin $m _ { j }$

$$
\begin{array} { r } { \mathbf { n } _ { j } ^ { \top } { \pmb \mu } _ { i } - \left\| \mathrm { d i a g } ( \mathbf s _ { i } ) R _ { i } ^ { \top } \mathbf { n } _ { j } \right\| _ { 2 } - c _ { j } > m _ { j } . } \end{array}\tag{42}
$$

Here, $\mu _ { i } , \mathbf { s } _ { i } ,$ , and $R _ { i }$ are the center, per-axis scales, and rotation of Gaussian $i ,$ and face j is the plane $\mathbf { n } _ { j } ^ { \top } \mathbf { x } = c _ { j }$ with outward unit normal $\mathbf { n } _ { j } .$ For each face, we also record whether it is observed or, when detection fails, placed at the 1st or 99th percentile of the Gaussian positions along its normal.

![](images/938564d56ef4fb910dfca8374b25b478cffd58ae7492c13e1cf0f2dd058b599a.jpg)

![](images/d4d923963f6edbf47a521530294d411f92be505786b742dcdbd01e0b4698d527.jpg)

(c) Resulting 3D background  
![](images/9e5b220cce796f11b4890fff8ee8111bcf684a4782d4b072933719001169cda4.jpg)  
Figure 23. A background-generation failure relative to the benchmark requirements. Selected prompt excerpts specify a factory with a conveyor, numerous workstations, and a prominent pillar. The generated panorama and the resulting background contain human figures and central equipment, conflicting with the no-people and open-center requirements for work-cell placement. Panel (c) is cropped and enlarged, and amber markers indicate two human figures.

![](images/80daa5928656459ab3ee40c7339f080ada85d3bbcac1bcdab8fa99197e1cf83b.jpg)

(b) Estimated enclosure  
![](images/d64690843ac6036a5fe7d0bc9e00cc7da798a6de0be9441ef5655bb6f498996a.jpg)

![](images/682013bf8cf1be90618a14961397e473723ff03a51bcfa3928b84896b808e043.jpg)  
Figure 24. Preparation of generated backgrounds for work-cell placement. (a) Exterior view of a generated Gaussian splat. (b) Estimated room enclosure, shown by the red and green overlay. (c) Interior view with the identified placement region, outlined in blue. The estimated enclosure provides the geometric reference for coordinate alignment and out-of-bounds filtering, and the floor rectangle identifies the region available for work-cell placement.

Table 35. Post-processing of a generated splat, with default parameters.
<table><tr><td>Step</td><td>Method</td><td>Defaults</td></tr><tr><td>Faces</td><td>Floor and ceiling by RANSAC on the lowest and highest deciles, walls by blocked fraction B(c)</td><td>Normal within  $2 0 ^ { \circ } ,$  maxc  $B ( c ) \geq 0 . 1 3$ </td></tr><tr><td>Alignment</td><td>Floor to z = 0, yaw from wall azimuths folded by 90°</td><td></td></tr><tr><td>Floaters</td><td>Eq. (42)</td><td>mj: 0.2 (floor, ceiling), 0.1 (walls) of center distance</td></tr><tr><td>Placement region</td><td>Largest free rectangle, with the origin moved to its center</td><td>Band 0.2–1.2 m, wall margin 1 m</td></tr></table>

Work-cell placement. We build a 0.25 m occupancy grid from opaque splats 0.15 to 2.5 m above the floor, treating unscanned space as occupied. Candidate work-cell positions are sampled on a 0.5 m grid with rotations in 90<sup>◦</sup> increments. For each candidate, we measure the distance to the nearest occupied grid cell at sampled points covering the work-cell footprint, including the robot’s spawn poses and a 1.5 m margin. We reject candidates with any sample in occupied space and select the remaining candidate with the largest minimum distance. If no candidate remains, placement fails. For the variants of one scenario, we keep several well-separated work-cell poses with at least 0.5 m clearance, so they occupy different parts of the same building.

Packaging scene for the comparison. A person supplies the following brief for the packaging scene used in the background study (Figure 17).

## Packaging-cell brief

A home-appliance packaging cell with an infeed conveyor of boxed washing machines, two wooden pallets, a flow rack of small-parts totes, a packing table, and a cart. Leave a clear 1.5 m aisle for a mobile manipulator.

Table 36. Proprietary and open-source backgrounds generated from the same reference image and compared on nine matched frames. Rubric scores come from a blind VLM judge over three shuffled repeats, and pairwise preference counts wins over both presentation orders. Better values are in bold. Stability is the temporal-stability score (limit 1.0, lower is better). The proprietary splat has 4.32 M Gaussians and is kept in the background catalog, and the open-source splat has 3.78 M and is not.
<table><tr><td colspan="7">Rubric score (1 to 5)</td></tr><tr><td>Background</td><td>Mean</td><td>Detail</td><td>Integrity</td><td>Realism</td><td>Consistency</td><td>Preferred (of 18)</td><td>Usable (of 27)</td><td>Stability</td></tr><tr><td>Proprietary API</td><td>2.85</td><td>3.00</td><td>2.67</td><td>2.41</td><td>3.33</td><td>18</td><td>0</td><td>0.37</td></tr><tr><td>Open-source chain</td><td>1.78</td><td>1.67</td><td>1.78</td><td>1.33</td><td>2.33</td><td>0</td><td>0</td><td>0.98</td></tr></table>

Table 37. Chunking reduces the measured peak GPU memory of batched re-rendering. A 24-frame recording with one simulation instance and three cameras is re-rendered with a different lighting environment per variant on a 24 GB GPU.
<table><tr><td>Variants</td><td>Per chunk</td><td>Peak memory (GiB)</td><td>Images/s</td></tr><tr><td>4</td><td>4</td><td>4.5</td><td>129.0</td></tr><tr><td>16</td><td>16</td><td>9.8</td><td>135.0</td></tr><tr><td>32</td><td>32</td><td>15.9</td><td>140.6</td></tr><tr><td>32</td><td>8</td><td>6.4</td><td>130.3</td></tr></table>

The agent searches the catalog, writes the scene specification, applies static verification, and repairs it at most twice. A person reviews the scene in place of model critics. This scene is used only to study backgrounds, because its bounding-box checks flag 2 mm gaps as overlaps. The full results of the comparison in $\ S 4 . 2 . 2$ are reported in Table 36. The rubric judgments are post hoc and do not decide installation.

## D.3 Re-rendering recorded trajectories

Render targets grow with simulation instances, cameras, and background variants, so the chunk size is fixed before allocation, instead of shrinking the batch after an out-of-memory failure. Variants that share their objects and robot and differ only in background or lighting share $F _ { s , \ell }$ and differ only in $O _ { s }$ , so a recorded trajectory can be re-rendered under each of them without rerunning the policy $( \ S 2 . 1 )$ . Demonstrations are replayed in splat renderers in the same way (Kim et al., 2026). We choose the number of variants per chunk using pixel and video-writer budgets. For $N _ { \mathrm { e n v } }$ instances and C cameras with image width $w _ { c }$ and height $h _ { c } ,$ , each variant requires $p = N _ { \mathrm { e n v } } \sum _ { c = 1 } ^ { C } w _ { c } h _ { c }$ pixels and $N _ { \mathrm { e n v } } C$ writers, giving a chunk size of

$$
B = \operatorname* { m a x } ( 1 , \ \operatorname* { m i n } ( \left\lfloor P _ { \operatorname* { m a x } } / p \right\rfloor , \ \left\lfloor K / ( N _ { \mathrm { e n v } } C ) \right\rfloor ) ) ,\tag{43}
$$

where $P _ { \mathrm { m a x } } = 1 6$ megapixels by default and $K = 2 5 6$ . If a single variant requires more than K writers, rendering stops before replay. If it exceeds the pixel budget, the renderer warns and proceeds with a chunk of one variant, which can still exhaust GPU memory. Each chunk replays the trajectory once and releases its sensors before the next begins. For 32 variants, chunks of eight reduce peak memory from 15.9 to 6.4 GiB at 7% lower throughput (Table 37).

## D.4 Process logic representation

The scenario-authoring agent writes the four typed parts of process logic ℓ in a process-logic file using role placeholders to bind scenario entities. An effect is a continuous process applied at every step. The only type currently supported is a conveyor belt, whose speed events can change. An event pairs a trigger with one or more actions applied in order. The trigger is a predicate of § 5.2 that fires while it holds, on its rising edge, or after it has held for a dwell time, and each action teleports a body, sets a joint target, or sets the speed of an effect. A queue delivers bodies first in, first out. It names its member bodies, the slots where they wait, and the region and pose where each is delivered, and it can wait for an upstream belt, delay each dispatch, and let a removed body re-enter. An idle target sets the initial joint target of articulated equipment, such as a press ram parked at the top of its stroke. Changes that must happen together, such as removing a blank and placing the stamped part, form one event with two actions rather than two events with the same trigger.

## D.5 Verification thresholds across components

Table 38 lists the reported thresholds and resource limits in § 4.2, including the checks at each stage of background generation.   
Deterministic checks run first, every judged criterion must pass, and these checks determine installation eligibility.

Table 38. Thresholds of the checks in § 4.2.
<table><tr><td>Check</td><td>Threshold</td></tr><tr><td>Static geometry</td><td>Fixture bounding-box overlap and ground penetration at most 5 mm, specified clearance of an item above its support at most 1 cm</td></tr><tr><td>Plan</td><td>Up to 10⁴ reachable predicate states, critic score at least 0.8 with no open concern</td></tr><tr><td>Variants</td><td>At least two distinct resolved layouts when variation is requested, every variant collision-checked</td></tr><tr><td>Specification fidelity</td><td>Position within 3 cm, contact gap within 1 cm, penetration within 5 mm, at most five attempts</td></tr><tr><td>Background prompt</td><td>Ten requirements present</td></tr><tr><td>Panorama geometry</td><td>Aspect ratio 2:1 ± 0.01, width at least 1,024 pixels, seam ratio at most 4.0, pole standard deviation at most 24 (8-bit)</td></tr><tr><td>Panorama VLM</td><td>Valid panorama, open center, plausible, no people</td></tr><tr><td>Splat</td><td>Single splat</td></tr><tr><td>View VLM</td><td>Intact geometry, open center, plausible, no people, consistent with the panorama</td></tr><tr><td>Background coverage</td><td>Missing background at most 2% and at most 1 percentage point above the scene without background, with sharpness, contrast, and clipping checks</td></tr><tr><td>Temporal stability</td><td>Mean frame-to-frame RGB difference at most 1.0 per fixed region, region standard deviation at least 2, at least 99% valid frames</td></tr><tr><td>Placement</td><td>Splat opacity at least 0.3, obstacle band 0.15 to 2.5 m, grid 0.25 m, margin 1.5 m, step 0.5 m, yaw in 90° steps, extra poses at least 0.5 m clear and separated by yaw or 3 m</td></tr><tr><td>Collision memory</td><td>Resolution from 32 to 1 mm, refined while a part costs at most 1 MB per simulation instance, parts above 500 MB per instance rejected</td></tr><tr><td>Re-rendering</td><td>Pixel budget of 16 megapixels per chunk by default, at most 256 open video writers</td></tr></table>

## D.6 Post-hoc rubric evaluation

Generated scenes and generated evaluations are scored by three judges, gpt-6-astra (high), claude-opus-5-5 (high), and gemini-4-argon (high), each under three personas. Every input is text only, with model names, paths, run identifiers, and timestamps removed and the inputs presented in shuffled order. All judges receive identical prompts and return typed JSON with an integer score from 1 to 5 and a rationale for each criterion and overall. Each prompt states one persona, the artifact type, and the criteria of that type with their anchors (Table 39), and asks for a rationale of one to three sentences before each score. These judges are used only for post-hoc analysis. The criteria use the word contract for an evaluation.

## Personas (verbatim)

Process engineer. You are a senior industrial process engineer with many years of experience designing and commissioning robotic work cells. You judge the artifact as you would before signing off a cell design or a cell acceptance test.

Evaluation researcher. You are a robot-learning evaluation researcher who designs benchmarks and success criteria for manipulation policies. You judge the artifact as you would before using it to compare policies in a paper.

Adversarial QA reviewer. You are an adversarial QA reviewer. Assume the artifact may be wrong and actively look for ways it is wrong: contradictions, missing pieces, loopholes, mismatches with the request. Score using the anchors below and ground each deduction in specific evidence from the text; do not deduct for speculative problems that you cannot point to in the artifact.

![](images/a4d7bfdcbfb01988ab6ff408b74c29dbc87fd5c3cad69e3cd971e41e300ec1dd.jpg)  
Figure 25. Rubric judges, scores averaged over three personas. (a) AUC of each scene criterion for predicting acceptance of the 25 candidates by the planner’s critic. (b) Overall-score difference for each defect that execution misses, with the number of defective evaluations in parentheses.

Table 39. Rubric criteria. Each criterion in the prompt also carries anchors for scores 1, 3, and 5.
<table><tr><td>Criterion</td><td>Question to the judge</td></tr><tr><td>Scenes</td><td></td></tr><tr><td>process_realism</td><td>Would this cell, its equipment, and its material flow exist on a real factory floor?</td></tr><tr><td>task_conditioning</td><td>Does the scene enable exactly the requested work, no more and no less?</td></tr><tr><td>entity_asset_fit</td><td>Is each entity the right kind of thing, at the right scale, with an asset whose category matches its role?</td></tr><tr><td>layout_coherence</td><td>Are positions, supports, clearances, access, and reach coherent?</td></tr><tr><td>meaningful_variation</td><td>Do the variation axes match what the request asks to vary, with meaningful ranges?</td></tr><tr><td>completeness</td><td>Is every element and constraint named in the request present and honoured?</td></tr><tr><td>Evaluations</td><td></td></tr><tr><td>task_faithfulness</td><td>Does the contract measure what the task text asks, including every allowed route to success?</td></tr><tr><td>requirement_coverage</td><td>Is each requirement owned by a predicate, event, job, metric, or acceptance rule?</td></tr><tr><td>measurement_validity</td><td>Are regions, tolerances, hold durations, and thresholds consistent with the task and physically sensible?</td></tr><tr><td>failure_coverage</td><td>Would the contract report failure for the plausible ways the task can go wrong?</td></tr><tr><td>robustness</td><td>Does the contract stay correct under noise and resist degenerate or exploiting executions?</td></tr><tr><td>clarity</td><td>Is the contract understandable and internally consistent, with its choices declared?</td></tr></table>

Scene results. Plan and scene critics accept or reject candidates during generation. Rubric judges score the candidates afterward for analysis. The pooled scores of gpt-6-astra and claude-opus-5-5 align closely with acceptance by a shared critic, but are less predictive of outcomes under the original pipelines. Asset fit largely drives the scores, while layout coherence provides little discrimination (Figure 25a). Scores cluster in a narrow range, agreement between judges is modest, and generated assets improve local fit without substantially affecting overall scores. The third judge, gemini-4-argon, scores higher than both with a wider spread across personas and agrees more with claude-opus-5-5 than with gpt-6-astra. It also separates acceptance by asset fit and variation but barely by layout coherence.

## E Evaluation generation details

This appendix gives the details behind § 4.3, namely the reasons for which the compiler rejects a proposal, the three checks of verification with an audit of them, the protocol of the generation study, and examples of its feedback.

## E.1 Rejections at compilation

Table 40 groups the reasons for which the compiler of § 4.3.2 rejects a confirmed proposal.

Table 40. Reasons for which the compiler rejects a confirmed proposal. A rejection returns the reason to the person and produces no evaluation.

<table><tr><td>Group</td><td>The proposal is rejected when</td></tr><tr><td>Names</td><td>a name is missing from the scenario, is not declared in the proposal, does not resolve to an object in the scene, or is reserved for the evaluator although the task gives it to the policy.</td></tr><tr><td>Predicates and metrics</td><td>a predicate has invalid parameters, or a metric is not one the evaluator can compute.</td></tr><tr><td>Job lifecycle</td><td>the start, completion, and failure events do not form a valid job, or the rubric of an episodic task ends on anything other than one job.</td></tr><tr><td>Authority</td><td>a visual requirement is not bound to events, a success criterion refers to a visual or unmea- sured requirement, or a requirement that the task needs is labeled unmeasured.</td></tr></table>

## E.2 Verification checks

This subsection states the three checks of Table 21 and reports a controlled evaluation-mutation audit on the nominal first scene variant of conveyor loading. An evaluation that fails the static check is not simulated. The reset and idle checks share one environment per scene variant, and an evaluation is accepted when it passes all three.

Static check. The check first composes the job, metrics, and rubric again, so an error in the schema, in a reference, or in the ownership of a requirement fails it. It then tests the composed evaluation against the rules of Table 41. A single-step completion is reported as an advisory, independently of required sustained events or protocol-violation thresholds. The persistence needed for completion belongs to task intent and is evaluated against labeled trajectories, not inferred from a one-frame event. Events on both edges require a rubric that tolerates no protocol violation, and a sustained event does not justify this lifecycle. A declared justification must name a condition of the evaluation and connect the job start to a required event, so a justification cannot outlive the condition it is written for.

Table 41. Rules of the static check. Reported conditions appear in the result of the check but do not reject the evaluation.
<table><tr><td>Rule</td><td>Rejects when</td><td>Admits or only reports when</td></tr><tr><td>Single-step completion</td><td>never</td><td>the completion event holds for one step</td></tr><tr><td>Events on both edges</td><td>a predicate has events on its rising and its falling edge, or its declared justification names no condition of the evaluation</td><td>the rubric tolerates no protocol violation</td></tr><tr><td>Nonpositive drop threshold</td><td>a drop predicate uses a height threshold z ≤ 0 and no region failure is declared</td><td>the evaluation declares it a guard only, which is reported</td></tr><tr><td>Time limit beyond the run</td><td>the job timeout or the stall limit exceeds the</td><td>the limit equals the length of the run, which is</td></tr><tr><td></td><td>length of the run</td><td>reported</td></tr><tr><td>No failure event</td><td>never</td><td>no failure event is declared, which is reported</td></tr></table>

Reset check. The check runs 4 s of physics with the robot uncontrolled, at four substeps, in every scene variant. It runs once with every body at its nominal position and once at each corner of the randomization range of body positions, and the axes that it leaves at nominal are listed in its result. It covers the units that the job names and, when a queue delivers the units as in conveyor loading, the units that the queue delivers. It fails when a unit expected to start a job is outside its start condition at reset, when the queue delivers no unit at reset, or when a unit leaves its start condition during the 4 s. For each unit initiall inside the start condition, the implementation records the first step at which it becomes inactive, if any.

Idle check. In the same environment, each evaluation then runs from reset to its termination with no action, at its time limit or its stall limit. The check fails when a job completes, when the run passes the success criterion, or when a start event fires on a job that is already open. When the units miss their start condition at reset, the idle run is not judged, because the reset check already reports it.

Table 42. Verification outcomes on the nominal first conveyor-loading variant. Expected is the rejection stage specified before physics, and None predicts no rejection. NR means not run, and NJ means not judged because Reset fails, despite an executed rollout. Accepted refers only to this nominal scope. The static-only control has no final acceptance decision.
<table><tr><td>Case</td><td>Expected</td><td>Static</td><td>Reset</td><td>Idle</td><td>Accepted</td></tr><tr><td>Conveyor-loading control</td><td>None</td><td>Pass</td><td>Pass</td><td>Pass</td><td>Yes</td></tr><tr><td>Undeclared completion event</td><td>Static</td><td>Fail</td><td>NR</td><td>NR</td><td>No</td></tr><tr><td>Static-rule control (two-frame completion)</td><td>None</td><td>Pass</td><td>NR</td><td>NR</td><td>一</td></tr><tr><td>Add falling start-predicate event</td><td>Static</td><td>Fail</td><td>NR</td><td>NR</td><td>No</td></tr><tr><td>Add drop failure at floor (z = 0)</td><td>Static</td><td>Fail</td><td>NR</td><td>NR</td><td>No</td></tr><tr><td>Stall limit: 3,600 to 3,601 steps</td><td>Static</td><td>Fail</td><td>NR</td><td>NR</td><td>No</td></tr><tr><td>Start region excludes presented tote</td><td>Reset</td><td>Pass</td><td>Fail</td><td>NJ</td><td>No</td></tr><tr><td>Start region: 0.2 mm vertical slab</td><td>Reset</td><td>Pass</td><td>Pass</td><td>Pass</td><td>Yes</td></tr><tr><td>Start region widened to whole hall</td><td>Idle</td><td>Pass</td><td>Pass</td><td>Fail</td><td>No</td></tr><tr><td>Completed-job threshold: 1 to 0</td><td>Idle</td><td>Fail</td><td>NR</td><td>NR</td><td>No</td></tr><tr><td>Upright tolerance: 10° to 45°</td><td>None</td><td>Pass</td><td>Pass</td><td>Pass</td><td>Yes</td></tr><tr><td>Upright tolerance:  $1 0 ^ { \circ }$  to 1°</td><td>None</td><td>Pass</td><td>Pass</td><td>Pass</td><td>Yes</td></tr></table>

Controlled evaluation mutations. The audit in Table 42 changes evaluation documents while keeping the scene, robot, and physics fixed. The control is an evaluation written for the audit rather than the benchmark evaluation of conveyor loading. It uses a 3,600-step fixed-duration run at 60 Hz and requires at least one completed job with zero protocol violations. No policy or model is run. The first 240 steps of the passive rollout provide the 4 s reset check, and the same rollout continues to 60 s for the idle check, with four physics substeps. The dual-edge test uses a separate static control whose completion requires two frames and whose rubric allows at most one protocol violation, which removes the zero-violation waiver. The mutant adds the falling start-predicate event. The static control passes and the mutant fails the rule on events at both edges. This control is not submitted for physical acceptance.

The control passes all three checks. Five mutations are rejected by the static check (including schema validation), one by the reset check, and one by the idle check. The zero-job threshold is rejected by schema validation before the idle check because acceptance requires a positive completed-job threshold. The whole-hall start region passes the static and reset checks but completes four jobs without robot action and satisfies the rubric during the 60 s rollout, where the control completes none. The narrow start slab passes, contrary to the predeclared reset-rejection expectation, because the presented tote is initially inside it and never leaves during the reset window. No settling departure is established, so this outcome is neither evidence of a missed violation nor a test of departure-detection sensitivity. In contrast, the distant start region rejects an initial eligibility failure, and its Idle result is excluded because the evaluation fails the reset check. Both changes to the upright tolerance pass all three checks. Verification does not receive the tolerance that the person intends and therefore does not reject these semantic changes. This audit does not measure their verdict errors on tilted trajectories or estimate a statistical detection rate.

Scope. The audit covers one nominal variant without randomization offsets and does not establish acceptance across al variants or the seven tasks.

## E.3 Generation study protocol

The four models of Table 43 run at the following effort levels, claude-opus-5-5 (high), gpt-6.1-sol (medium), gpt-6-astra (high), and gpt-6-luna (medium), and the simulated user is gpt-6.1-sol (medium). Each model runs three fresh conversations for direct and for question-guided generation, which are repeated model calls rather than controlled random seeds. The 24 generation trials, twelve with direct evaluation generation and twelve with question-guided generation, yield the 48 condition outcomes of the table because the first candidate of each trial also forms its condition without feedback. The study makes 91 mode calls, 18 in direct evaluation generation, 43 in question-guided generation, and 30 by the simulated user, who answers 64 questions in fifteen answer rounds and reviews fifteen proposals, approving thirteen. In direct evaluation generation, the mode receives the task instruction, neutral interface documentation, and the public scenario, and only the simulated user sees the private description of the intended task. Neither the generating models nor the simulated user sees the held-out recordings, their labels, or any verdict. Twenty-nine distinct executable candidates are scored on all fifteen recordings, and candidates shared between conditions are reused in the reported totals. The recorded costs include the simulated user and the compiler diagnostics, which in this implementation carry the records of earlier calls, so one failed trial spends 247,042 input tokens in its last proposal call.

Table 43. Evaluations generated for single tote transfer, by model and condition, three trials each. Accepted counts trials whose evaluation passes the guards and the three checks on the first scene variant. TP, FN, FP, and TN count pairs of an executable evaluation and one of the fifteen labeled recordings, including evaluations that verification rejects, and a trial without an executable evaluation stays unscored.
<table><tr><td>Model</td><td>Condition</td><td>Accepted</td><td>TP</td><td>FN</td><td>FP</td><td>TN</td></tr><tr><td>claude-opus-5-5</td><td>Direct, no feedback</td><td>3/3</td><td>12</td><td>0</td><td>3</td><td>30</td></tr><tr><td></td><td>Direct, with feedback</td><td>3/3</td><td>12</td><td>0</td><td>3</td><td>30</td></tr><tr><td></td><td>Questions, no feedback</td><td>3/3</td><td>12</td><td>0</td><td>0</td><td>33</td></tr><tr><td></td><td>Questions, with feedback</td><td>3/3</td><td>12</td><td>0</td><td>0</td><td>33</td></tr><tr><td>gpt-6.1-sol</td><td>Direct, no feedback</td><td>0/3</td><td>12</td><td>0</td><td>3</td><td>30</td></tr><tr><td></td><td>Direct, with feedback</td><td>3/3</td><td>12</td><td>0</td><td>3</td><td>30</td></tr><tr><td></td><td>Questions, no feedback</td><td>3/3</td><td>12</td><td>0</td><td>0</td><td>33</td></tr><tr><td></td><td>Questions, with feedback</td><td>3/3</td><td>12</td><td>0</td><td>0</td><td>33</td></tr><tr><td>gpt-6-astra</td><td>Direct, no feedback</td><td>0/3</td><td>12</td><td>0</td><td>3</td><td>30</td></tr><tr><td></td><td>Direct, with feedback</td><td>3/3</td><td>12</td><td>0</td><td>3</td><td>30</td></tr><tr><td></td><td>Questions, no feedback</td><td>3/3</td><td>12</td><td>0</td><td>0</td><td>33</td></tr><tr><td></td><td>Questions, with feedback</td><td>3/3</td><td>12</td><td>0</td><td>0</td><td>33</td></tr><tr><td>gpt-6-luna</td><td>Direct, no feedback</td><td>3/3</td><td>0</td><td>12</td><td>0</td><td>33</td></tr><tr><td></td><td>Direct, with feedback</td><td>3/3</td><td>0</td><td>12</td><td>0</td><td>33</td></tr><tr><td></td><td>Questions, no feedback</td><td>1/3</td><td>4</td><td>0</td><td>0</td><td>11</td></tr><tr><td></td><td>Questions, with feedback</td><td>2/3</td><td>8</td><td>0</td><td>0</td><td>22</td></tr></table>

## E.4 Feedback examples

These examples come from the generation study reported in Table 43, not from new trials. The excerpts below are shortened paraphrases of saved outputs and feedback. They illustrate the repair process rather than provide additional independent evidence.

Questions, with feedback: repairing event order. gpt-6-luna trial 1 asks which physical condition defines completion, what tilt counts as upright, and whether evidence must persist. The simulated user answers that uprightness and physical support on the conveyor must be observed together before or at inlet entry, with a tilt of at most 10<sup>◦</sup>, and that one step of confirmation suffices. Conveyor placement alone does not complete the job. These details are supplied through the answers and are absent from the original instruction. The first candidate is rejected for malformed predicate-parameter JSON. Table 44 shows the exchange for candidates 2 and 3.

Table 44. Saved proposal repair in gpt-6-luna trial 1 (Questions, with feedback). Text is condensed, and field names identify the corresponding saved proposal. The first candidate, omitted from this excerpt, fails JSON validation.
<table><tr><td>Stage</td><td>Relevant content</td></tr><tr><td>Candidate 2</td><td>transfer_complete combines support, on-top geometry, uprightness, and inlet membership in one conjunction. required_before_complete is empty, and the separately defined transfer_qualification is unused.</td></tr><tr><td>Simulated-user feedback</td><td>Candidate 2 accepts qualification only at entry and not earlier, and it also completes when qualification first holds after entry. Record qualification as an event, require it before or on completion, and complete on the rising inlet-entry event.</td></tr><tr><td>Candidate 3</td><td>transfer_qualified becomes a one-step rising event for the support/upright conjunction and is added to required_before_complete. transfer_complete becomes inlet membership alone, with completion on its one-step rising event.</td></tr></table>

The simulated user approves candidate 3, which compiles and passes the static, reset, and idle checks on the nomina first variant. It produces TP=4, FN=0, FP=0, and TN=11 on the frozen recordings. The temporal repair is requested by simulated-user review, not by the three verification checks or by held-out verdict feedback.

Feedback does not guarantee acceptance. In gpt-6-luna trial 2, the simulated user approves candidates 2 and 3, but both retain a required pickup condition labeled unmeasured. The compiler returns required\_unmeasured for pickup\_unmeasured. The three-candidate budget ends without an executable evaluation. This is the failed Questions trial in Table 43, whose fifteen recording pairs remain unscored.

## F 24/7-BENCH details

This appendix gives the settings, the evaluations, and the reporting behind § 5.

## F.1 Task selection

The work behind the seven tasks employs millions of people in each of the five largest robot markets (Table 45), with robot data from the IFR (International Federation of Robotics, 2026e;d;b;a) and employment from national labor surveys (Eurostat, 2026; Statistics Bureau of Japan, 2026; Ministry of Data and Statistics, Republic of Korea, 2026; U.S. Bureau of Labor Statistics, 2026) and the 2020 census of China (National Bureau of Statistics of China, 2022). Occupation groups differ between countries. For example, Japan’s handling group includes cleaning and South Korea’s counts all elementary work, so the table supports which tasks enter the benchmark but does not weight them.

Table 45. Industrial robots and matching workers by market. Installations are for 2025, density is robots per 10,000 manufacturing employees in 2024, and workers are counted in the national occupation groups closest to handling and to production, with their share of employment.
<table><tr><td>Region</td><td>Robots installed</td><td>Robot density</td><td>Handling workers</td><td>Production workers</td></tr><tr><td>China</td><td>354,000</td><td>166</td><td>6.1%</td><td>15.8%</td></tr><tr><td>European Union</td><td>60,500</td><td>231</td><td>5.33 M (2.6%)</td><td>5.90 M (2.8%)</td></tr><tr><td>Japan</td><td>36,200</td><td>446</td><td>5.05 M (7.4%)</td><td>8.44 M (12.4%)</td></tr><tr><td>South Korea</td><td>30,000</td><td>1,220</td><td>3.92 M (13.6%)</td><td>2.94 M (10.2%)</td></tr><tr><td>United States</td><td>38,400</td><td>307</td><td>2.95 M (1.9%)</td><td>8.57 M (5.5%)</td></tr></table>

## F.2 Robot and simulation settings

FRIDAY is the wheeled humanoid robot used throughout this paper, simulated from a single USD description (Table 46).

Table 46. FRIDAY robot model. Counts exclude fixed joints and the floating base.

<table><tr><td>Component</td><td>Description</td></tr><tr><td>Mobile base</td><td>Differential drive, two driven wheels on prismatic suspension, four passive casters</td></tr><tr><td>Torso</td><td>Five-joint waist</td></tr><tr><td>Arms</td><td>Two arms with seven revolute joints each</td></tr><tr><td>Hands</td><td>Two five-finger hands with 20 joints each</td></tr><tr><td>Head</td><td>Three joints</td></tr><tr><td>Active joints</td><td>64 (2 wheels, 5 waist, 14 arm, 40 hand, and 3 head joints)</td></tr><tr><td>Passive joints</td><td>11 (2 suspension and 9 caster joints)</td></tr><tr><td>Overall</td><td>75 joints in total, about 1.8 m tall in the default configuration</td></tr><tr><td>Cameras</td><td>Head and wrist cameras</td></tr><tr><td>Base sensors</td><td>IMU, LiDAR, and odometry</td></tr><tr><td>Joint telemetry</td><td>Joint state</td></tr><tr><td>Tactile sensing</td><td>Magnetic tactile skin on the palms and inner finger surfaces</td></tr></table>

The robot’s magnetic tactile skin produces Hall field vectors from contact-induced magnet displacements (§ 3.1.3 and § A.4). These readings are available to a policy but are not used by the benchmark’s success criteria, which are evaluated from privileged simulator state.

Table 47 lists the default timing and actuation settings for FRIDAY. All actuator controllers produce motor-current commands.   
Joint-specific parameters are defined in the robot configuration, and individual experiments specify any overrides.

Table 47. Default simulation and actuation settings for FRIDAY.

<table><tr><td>Setting</td><td>Configuration</td></tr><tr><td>Frame/control period</td><td>1/60 s</td></tr><tr><td>Control interface</td><td>Joint position and velocity targets</td></tr><tr><td>Physics substeps</td><td>4 per frame (1/240 s each)</td></tr><tr><td>Evaluation rate</td><td>60 Hz</td></tr></table>

## F.3 Evaluations of the seven tasks

This subsection states the predicates (Table 48), the margins, the temporal operators, and the job semantics behind § 5.2, and then the evaluation of each task in Table 23.

Table 48. Predicate vocabulary. All predicates read privileged state and return one value per simulation instance.
<table><tr><td>Group</td><td>Predicates</td></tr><tr><td>Object state</td><td>In a region, upright within a tilt bound, below a height, at rest</td></tr><tr><td>Relation</td><td>Touching, supported by, on top of, next to, stable on (on top of, supported by, and at rest)</td></tr><tr><td>Joint</td><td>Within a range, completed a stroke</td></tr><tr><td>Composition</td><td>Conjunction</td></tr></table>

State and margins. Let $x _ { t }$ be the privileged state at step t of a run sampled at $\Delta t = 1 / 6 0 \mathsf { s }$ . The default $N _ { \mathrm { s u b } } = 4$ physics substeps per frame have duration $h = \Delta t / N _ { \mathrm { s u b } } = 1 / 2 4 0 \mathsf { s }$ . For a run of $N _ { \mathrm { s t e p } }$ frames, $t _ { \mathrm { r u n } } = N _ { \mathrm { s t e p } } \Delta t$ , which can be shorter than $T _ { \mathrm { r u n } } . \mathsf { A }$ unit u has pose $( \mathbf { p } _ { u } , \mathbf { R } _ { u } ) \in S E ( 3 )$ and twist $\left( \mathbf { v } _ { u } , \omega _ { u } \right)$ , and joint j has position $q _ { j } . \mathsf { A }$ region $\mathcal { R } = ( b , \mathbf { c } , \mathbf { h } )$ is a box with center c and half extent $\mathbf { h } \in \mathbb { R } _ { > 0 } ^ { 3 }$ in the frame of a body b with pose $\left( \mathbf { p } _ { b } , \mathbf { R } _ { b } \right)$ , the world for fixed regions and a jig or a tote for moving ones. The margins are

$$
\begin{array} { r l r l } & { \mu _ { \mathrm { i n } } ( u , \mathcal { R } ) = \underset { i \in \{ 1 , 2 , 3 \} } { \operatorname* { m i n } } \left( h _ { i } - \left| \mathbf { e } _ { i } ^ { \top } \left( \mathbf { R } _ { b } ^ { \top } \left( \mathbf { p } _ { u } - \mathbf { p } _ { b } \right) - \mathbf { c } \right) \right| \right) , \quad } & & { \qquad \mu _ { \mathrm { t o p } _ { \alpha } } ( u ) = \alpha - \mathrm { a r c c o s } \left( \mathbf { e } _ { z } ^ { \top } \mathbf { R } _ { u } \mathbf { e } _ { z } \right) , } \\ & { \mu _ { \mathrm { s t } } ( u ) = \operatorname* { m i n } \left( \epsilon _ { v } - \left\| \mathbf { v } _ { u } \right\| , \epsilon _ { \omega } - \left\| \omega _ { u } \right\| \right) , \quad } & & { \qquad \mu _ { [ a , b ] } ( q _ { j } ) = \operatorname* { m i n } \left( q _ { j } - a , b - q _ { j } \right) , } \end{array}\tag{44}
$$

with the standard basis ${ \bf e } _ { 1 } , { \bf e } _ { 2 } , { \bf e } _ { 3 } = { \bf e } _ { z }$ , tilt bound $\alpha , \epsilon _ { v } = 0 . 0 2 { \sf m } / { \sf s } , \epsilon _ { \omega } = 0 .$ 1 rad/s, and $a = - \infty \mathsf { o r } b = \infty$ for one-sided joint bounds. Each atomic predicate thresholds its margin at zero, $\mathsf { i } \mathsf { n } ( u , \mathcal { R } ) = \mathbf { 1 } [ \mu _ { \mathsf { i } \mathsf { n } } ( u , \mathcal { R } ) \geq 0 ]$ and likewise for up and joint bounds, while rest is strict, $\mathsf { s t } ( u ) = \mathbf { 1 } [ \mu _ { \mathsf { s t } } ( u ) > 0 ]$ . A union of regions takes the maximum of the margins and a conjunction the minimum, so composite truth values can likewise be determined by thresholding the resulting margin, retaining the strictness of the constituent tests. For a set U of bodies, c $\begin{array} { r } { \mathsf { n t } ( U , \mathcal { R } ) = \sum _ { u \in U } \mathsf { i n } ( u , \mathcal { R } ) } \end{array}$ enters through integer bounds $n \leq \mathsf { c n t } ( U , { \mathcal { R } } ) \leq n ^ { \prime }$ Margins are used for their sign, and their magnitudes are not compared across predicates with different units. A geometric margin can also express distance to a boundary, for example −0.04 m for a tote whose origin rests 4 cm outside its region.

Temporal operators. For a Boolean sequence $b _ { t }$ , set $b _ { t } ~ = ~ 0$ for $t < 0$ and define the hold operator as $H _ { K _ { \mathrm { h o l d } } } b _ { t } =$ $\begin{array} { r } { \operatorname* { m i n } _ { 0 \leq s < K _ { \mathrm { h o l d } } } b _ { t - s } } \end{array}$ . The rising event $\uparrow _ { \Delta }$ b with $\Delta = K _ { \mathrm { h o l d } }$ ∆t fires at step t when $H _ { K _ { \mathrm { h o l d } } } b _ { t } \wedge \neg H _ { K _ { \mathrm { h o l d } } } b _ { t - 1 }$ , which equals $H _ { K _ { \mathrm { h o l d } } } b _ { t } \wedge \neg b _ { t - K _ { \mathrm { h o l d } } }$ and is what Eq. (19) computes, the falling event is $\downarrow _ { \Delta } b = \uparrow _ { \Delta } ( 1 - b )$ , and the subscript is omitted for $K _ { \mathrm { h o l d } } = 1$ . In the task formulas below, the temporal operator applies to the entire bracketed predicate. Durations written as 0.17 s denote 10 evaluation steps (10/60 s).

Jobs and success criterion. A job of unit u is given by a start event $e _ { \mathrm { s t a r t } }$ , required events $e _ { \mathrm { r e q } , 1 } , \ldots , e _ { \mathrm { r e q } , m } ,$ a completion event $e _ { \mathrm { c o m p l e t e } } ,$ failure events $e _ { \mathrm { f a i l } , 1 } , \ldots , e _ { \mathrm { f a i l } , l } ,$ , and a limit $\tau _ { \mathrm { j o b } }$ . It opens at the step $t _ { 0 }$ at which $e _ { \mathrm { s t a r t } } ( u )$ fires while no job of u is open, and a start event of u during an open job counts as a protocol violation. At every later step the job fails if some $e _ { \mathrm { f a i l } , j } ( u )$ fires, and otherwise completes i $\textsf { f } e _ { \mathrm { c o m p l e t e } } ( u )$ fires and every $e _ { \mathrm { r e q } , i } ( u )$ has fired since t<sub>0</sub>. A job still open when $( t - t _ { 0 } ) \Delta t \geq \tau _ { \mathrm { j o b } }$ times out, and one still open when the run ends is censored. Every job also fails when its unit remains in the floor region, $\uparrow _ { \Delta } \mathsf { i n } ( u , \mathcal { R } _ { \mathrm { f l o o r } } )$ with $\Delta$ of 0.5 or 1 s, or leaves the world, $z _ { u } = { \mathbf { e } } _ { z } ^ { \top } { \mathbf { p } } _ { u } < z _ { \operatorname* { m i n } }$ . An event $e _ { \mathrm { m o n i t o r } }$ that no job references, such as a hand in the die while the ram is away from home, can fire at any step, also after the job of its unit completes. A metric counts its firings $N _ { \mathrm { m o n i t o r } ; }$ , as for any other event of the evaluation. A run with $N _ { \mathrm { d o n e } }$ completed jobs ends at $N _ { \mathrm { d o n e } } = N _ { \mathrm { g o a l } }$ , at the limit $T _ { \mathrm { r u n } } .$ , or after 50 s to 3 min without an event, and

$$
\mathrm { p a s s } \iff N _ { \mathrm { d o n e } } \geq N _ { \mathrm { g o a l } } \wedge N _ { \mathrm { f a i l } } = N _ { \mathrm { t i m e o u t } } = N _ { \mathrm { s t a l l } } = N _ { \mathrm { v i o l a t i o n } } = 0 \wedge N _ { \mathrm { m o n i t o r } } = 0 .\tag{45}
$$

Here, $N _ { \mathrm { f a i l } } , N _ { \mathrm { t i m e o u t } } , N _ { \mathrm { s t a l l } }$ , and $N _ { \mathrm { v i o l a t i o n } }$ count failed jobs, timed-out jobs, stalls, and protocol violations. A rubric that only reports, as in conveyor loading, has no $N _ { \mathrm { g o a l } }$ and no success criterion, and its run ends $\mathrm { a t } T _ { \mathrm { r u n } } . N _ { \mathrm { g o a l } } , T _ { \mathrm { r u n } } , \tau _ { \mathrm { j o b } }$ , and the inactivity limit are parameters of the evaluation, not constants of the benchmark, and an evaluation may set its own. Table 49 lists the values of the seven reference evaluations used below.

Properties. Three properties follow from these definitions. (P1) For a fixed evaluation and initial job state, the verdict can be recomputed from timestamped events and the run termination time. The event times preserve the information needed for job timeouts and inactivity limits, and the termination time determines censoring. (P2) A predicate excursion shorter than $K _ { \mathrm { h o l d } }$ steps fires no event, a run of at least $K _ { \mathrm { h o l d } }$ steps fires exactly one, and its time stamp lags the transition by exactly $K _ { \mathrm { h o l d } } - 1$ steps. (P3) Every job ends in exactly one of the states completed, failed, timed out, or censored, a failure overrides a completion in the same step, and a job completes only after all of its required events, so a unit that reaches its destination by another route does not count.

Table 49. Rubric parameters of the seven reference evaluations. A report-only rubric has no job target and no job limit.
<table><tr><td>Task</td><td> $N _ { \mathrm { g o a l } }$   $T _ { \mathrm { r u n } }$ </td><td>(min)</td><td> $\tau _ { \mathrm { j o b } }$  (min)</td></tr><tr><td>Conveyor loading</td><td>一</td><td>5</td><td>一</td></tr><tr><td>Flow-rack loading</td><td>3</td><td>10</td><td>7</td></tr><tr><td>Cart push</td><td>8</td><td>20</td><td>18</td></tr><tr><td>Jig loading</td><td>8</td><td>20</td><td>18</td></tr><tr><td>Press tending</td><td>3</td><td>12</td><td>9</td></tr><tr><td>Valve operation</td><td>3</td><td>10</td><td>7</td></tr><tr><td>Barcode routing</td><td>5</td><td>12</td><td>6</td></tr></table>

Conveyor loading. The instruction reads “Pick up each tote presented on the pallet and place it upright on the running conveyor.” A queue of five totes returns every tote that enters the X-ray tunnel to the pallet, so the same totes serve every job. A tote opens a job when the queue delivers it to the pallet. It must be upright on and supported by the belt, and the job completes when the tote enters the tunnel, a single-step event because the queue recycles the tote at once. The job has no limit $\tau _ { \mathrm { j o b } }$ of its own, and the rubric runs for 5 min and only reports. Single tote transfer shares this job, with a rubric that ends after one job or 1 min and passes with one completed job and no protocol violations (Table 20).

$$
\begin{array} { r l } & { e _ { \mathrm { s t a r t } } ( u ) = \uparrow \mathsf { i n } ( u , \mathcal { R } _ { \mathrm { p a l l e t } } ) , } \\ & { \quad e _ { \mathrm { r e q } } ( u ) = \uparrow \big ( \mathsf { o n } ( u , \mathrm { b e l t } ) \wedge \mathsf { s u p } ( u , \mathrm { b e l t } ) \wedge \mathsf { u p } _ { 1 0 ^ { \circ } } ( u ) \big ) , } \\ & { e _ { \mathrm { c o m p l e t e } } ( u ) = \uparrow \mathsf { i n } ( u , \mathcal { R } _ { \mathrm { t u n n e l } } ) , } \end{array}
$$

where on and sup are the relations on top of and supported by in Table 48.

Flow-rack loading. The instruction reads “Move each container from the pallet stack into a lane of the flow rack, upright.” A container opens a job in its slot of the pallet stack and crosses the aisle, and the job completes when the container is upright in any of the 12 lanes $\mathcal { R } _ { 1 } , \ldots , \mathcal { R } _ { 1 2 }$

$$
\begin{array} { r l } & { e _ { \mathrm { s t a r t } } ( \boldsymbol { u } ) = \boldsymbol { \uparrow } _ { 0 . 5 \mathrm { s } } \mathsf { i n } ( \boldsymbol { u } , \mathcal { R } _ { \mathrm { s t a c k } } ( \boldsymbol { u } ) ) , } \\ & { \quad e _ { \mathrm { r e q } } ( \boldsymbol { u } ) = \boldsymbol { \uparrow } _ { 0 . 1 7 \mathrm { s } } \mathsf { i n } ( \boldsymbol { u } , \mathcal { R } _ { \mathrm { a i s l e } } ) , } \\ & { e _ { \mathrm { c o m p l e t e } } ( \boldsymbol { u } ) = \boldsymbol { \uparrow } _ { 0 . 5 \mathrm { s } } \left[ \mathsf { i n } \big ( \boldsymbol { u } , \bigcup _ { \ell = 1 } ^ { 1 2 } \mathcal { R } _ { \ell } \big ) \wedge \mathsf { u p } _ { 1 5 ^ { \circ } } ( \boldsymbol { u } ) \right] , } \\ & { \quad e _ { \mathrm { f a i l } } ( \boldsymbol { u } ) = \boldsymbol { \downarrow } _ { 0 . 5 \mathrm { s } } \mathsf { u p } _ { 4 5 ^ { \circ } } ( \boldsymbol { u } ) . } \end{array}
$$

Cart push. The instruction reads “Load the cartons from the staging tables onto a hand cart, push it to a dock bay, and clear the loading deck.” A carton opens a job on its staging table, leaves the table, and crosses the aisle, and the job completes when the carton is upright and at rest in a dock bay and no carton of the set U remains on the loading deck. The completion condition checks that the loading deck is clear but does not verify that an empty cart has returned. A carton that leaves its bay fires an event e<sub>monitor</sub>.

$$
\begin{array} { r l } & { e _ { \mathrm { s t a r t } } ( u ) = \uparrow _ { 0 . 5 \mathrm { s } } \mathrm { i n } ( u , \mathcal { R } _ { \mathrm { s t a g e } } ( u ) ) , } \\ & { e _ { \mathrm { r e q } , 1 } ( u ) = \downarrow _ { 0 . 1 7 \mathrm { s } } \mathrm { i n } ( u , \mathcal { R } _ { \mathrm { s t a g e } } ( u ) ) , } \\ & { e _ { \mathrm { r e q } , 2 } ( u ) = \uparrow _ { 0 . 5 \mathrm { s } } \mathrm { i n } ( u , \mathcal { R } _ { \mathrm { a i s l e } } ) , } \\ & { e _ { \mathrm { c o m p l e t e } } ( u ) = \uparrow _ { 1 \mathrm { s } } \big [ \mathrm { i n } ( u , \mathcal { R } _ { \mathrm { d o c k } } ) \wedge \mathsf { u p } _ { 1 5 ^ { \circ } } ( u ) \wedge \mathsf { s t } ( u ) \wedge \mathsf { c n t } ( U , \mathcal { R } _ { \mathrm { l o a d } } ) = 0 \big ] , } \\ & { e _ { \mathrm { m o n i t o r } } ( u ) = \downarrow _ { 0 . 5 \mathrm { s } } \mathrm { i n } ( u , \mathcal { R } _ { \mathrm { d o c k } } ) . } \end{array}
$$

Jig loading. The instruction reads “Fill the pockets of the bench jig with parts from the tray, store the full jig on the rack, and seat an empty jig.” A part opens a job in the input tray, leaves it, and is seated in a pocket of the bench jig. The job completes when the part’s jig is stored on a rack shelf while a jig of the set G is seated on the bench and no part of the set U remains there. Pockets $\mathcal { R } _ { \mathrm { p o c k e t } }$ move with their jigs. The evaluation does not enforce a pocket-filling order. A stored part that leaves its jig, a jig on the floor, and a jig tilted beyond 30<sup>◦</sup> fire events e<sub>monitor</sub>.

$$
\begin{array} { r l } & { \boldsymbol { e } _ { \mathrm { s t a r t } } ( u ) = \boldsymbol { \uparrow } _ { 0 , \mathrm { 5 s } } \mathrm { i n } ( u , \mathcal { R } _ { \mathrm { t r a y } } ) , } \\ & { \boldsymbol { e } _ { \mathrm { r e q , 1 } } ( u ) = \boldsymbol { \downarrow } _ { 0 , \mathrm { 1 7 s } } \mathrm { i n } ( u , \mathcal { R } _ { \mathrm { t r a y } } ) , } \\ & { \boldsymbol { e } _ { \mathrm { r e q , 2 } } ( u ) = \boldsymbol { \uparrow } _ { 1 \mathrm { s } } \left[ \mathrm { i n } ( u , \mathcal { R } _ { \mathrm { p o c k e t } } ) \wedge \mathrm { i n } ( u , \mathcal { R } _ { \mathrm { b e n e h } } ) \right] , } \\ & { \boldsymbol { e } _ { \mathrm { c o m p l e t e } } ( u ) = \boldsymbol { \uparrow } _ { 1 \mathrm { s } } \left[ \mathrm { i n } ( u , \mathcal { R } _ { \mathrm { p o c k e t } } ) \wedge \mathrm { i n } ( u , \mathcal { R } _ { \mathrm { s h e l f } } ) \wedge \mathrm { s t } ( u ) \wedge \mathrm { c n t } ( G , \mathcal { R } _ { \mathrm { s e a t } } ) \geq 1 \wedge \mathrm { c n t } ( U , \mathcal { R } _ { \mathrm { b e n e h } } ) = 0 \right] . } \end{array}
$$

Press tending. The instruction reads “Load a blank into the die, start the press with both buttons, and move the stamped part to the outfeed table.” A blank opens a job on the infeed table, is nested in the die, is pressed with both buttons while nested, and is stamped, when the press event replaces it by its stamped part $u ^ { \prime } .$ The job completes when the origin of $u ^ { \prime }$ remains in the outfeed region for 1 s. This region is a 7 cm height band above the table. The dwell is a proxy for placement at rest and does not include a velocity check. A hand or wrist link of the set H inside the die while the ram is away from home fires an event $e _ { \mathrm { m o n i t o r } } .$ . Here, q<sub>L</sub> and $q _ { \mathrm { R } }$ are the positions of the two button joints and $q _ { \mathrm { r a m } }$ that of the ram.

$$
\begin{array} { r l } & { e _ { \mathrm { s t a r t } } ( u ) = \uparrow _ { 0 . 5 \mathrm { s } } \mathrm { i n } ( u , \mathcal { R } _ { \mathrm { i n f e e d } } ) , } \\ & { e _ { \mathrm { r e q } , 1 } ( u ) = \uparrow _ { 0 . 5 \mathrm { s } } \mathrm { i n } ( u , \mathcal { R } _ { \mathrm { n e s t } } ) , } \\ & { e _ { \mathrm { r e q } , 2 } ( u ) = \uparrow _ { 0 . 1 \mathrm { s } } \big [ \mathrm { i n } ( u , \mathcal { R } _ { \mathrm { n e s t } } ) \wedge q _ { \mathrm { L } } \leq - 1 9 \mathrm { m m } \wedge q _ { \mathrm { R } } \leq - 1 9 \mathrm { m m } \big ] , } \\ & { e _ { \mathrm { r e q } , 3 } ( u ) = \uparrow _ { 0 . 5 \mathrm { s } } \mathrm { s t a m p e d } ( u ) , } \\ & { e _ { \mathrm { c o m p l e t e } } ( u ) = \uparrow _ { 1 \mathrm { s } } \mathrm { i n } ( u ^ { \prime } , \mathcal { R } _ { \mathrm { o u t f e e d } } ) , } \\ & { e _ { \mathrm { m o n i t o r } } = \uparrow _ { 0 . 0 5 \mathrm { s } } \big [ \mathrm { c n t } ( H , \mathcal { R } _ { \mathrm { d i e } } ) \geq 1 \wedge q _ { \mathrm { r a m } } \leq - 2 0 \mathrm { m m } \big ] . } \end{array}
$$

Valve operation. The instruction reads “Place a beaker under the spout, open the valve until it is filled, close the valve, and move the beaker to the output side.” A beaker opens a job on the input side, and the job completes when the beaker is upright and at rest on the output side with the valve closed. Because the simulator does not model the liquid, filling is proxied by holding the beaker upright under the spout with the valve open for 3 s. Opening the valve with no beaker of the set U under the spout fires an event e , where $q _ { \mathrm { v a l v e } }$ is the valve joint angle.

$$
\begin{array} { r l } & { e _ { \mathrm { s t a r t } } ( u ) = \uparrow _ { 0 . 5 \mathrm { s } } \mathrm { i n } ( u , \mathcal { R } _ { \mathrm { i n p u t } } ) , } \\ & { \qquad e _ { \mathrm { r e q } } ( u ) = \uparrow _ { 3 \mathrm { s } } \left[ \mathrm { i n } ( u , \mathcal { R } _ { \mathrm { s p o u t } } ) \wedge \mathsf { u p } _ { 1 0 ^ { \circ } } ( u ) \wedge q _ { \mathrm { v a l v e } } \geq 6 0 ^ { \circ } \right] , } \\ & { e _ { \mathrm { c o m p l e t e } } ( u ) = \uparrow _ { 1 \mathrm { s } } \left[ \mathrm { i n } ( u , \mathcal { R } _ { \mathrm { o u t p u t } } ) \wedge \mathsf { u p } _ { 1 5 ^ { \circ } } ( u ) \wedge \mathsf { s t } ( u ) \wedge q _ { \mathrm { v a l v e } } \leq 5 ^ { \circ } \right] , } \\ & { e _ { \mathrm { f a i l } } ( u ) = \downarrow _ { 0 . 2 5 \mathrm { s } } \mathsf { u p } _ { 3 0 ^ { \circ } } ( u ) , } \\ & { e _ { \mathrm { m o n i t o r } } = \uparrow _ { 0 . 1 \mathrm { s } } \left[ \mathrm { c n t } ( U , \mathcal { R } _ { \mathrm { s p o u t } } ) = 0 \wedge q _ { \mathrm { v a l v e } } \geq 1 0 ^ { \circ } \right] . } \end{array}
$$

Barcode routing. The instruction reads “Put each scanned parcel into the tote assigned to its barcode, and a parcel with an unreadable code into the exception tote.” A parcel opens a job once it passes the scan gantry. It then leaves the end of the belt, and the job completes when the parcel is at rest in the tote $\mathcal { R } _ { \mathrm { d e s t } } ( u )$ that the routing table assigns to its code, the exception tote for an unreadable code. Resting in any other tote fails the job, and a parcel that leaves its assigned tote fires an event e<sub>monitor</sub>.

$$
\begin{array} { r l } & { e _ { \mathrm { s t a r t } } ( u ) = \uparrow _ { 0 . 5 \mathrm { s } } \mathsf { i n } ( u , \mathcal { R } _ { \mathrm { s c a n } } ) , } \\ & { \qquad e _ { \mathrm { r e q } } ( u ) = \downarrow _ { 0 . 1 7 \mathrm { s } } \mathsf { i n } ( u , \mathcal { R } _ { \mathrm { e n d } } ) , } \\ & { e _ { \mathrm { c o m p l e t e } } ( u ) = \uparrow _ { 1 \mathrm { s } } \bigl [ \mathsf { i n } ( u , \mathcal { R } _ { \mathrm { d e s t } } ( u ) ) \wedge \mathsf { s t } ( u ) \bigr ] , } \\ & { \qquad e _ { \mathrm { f a i l } } ( u ) = \uparrow _ { 1 \mathrm { s } } \bigl [ \mathsf { i n } ( u , \mathcal { R } _ { \mathrm { o t h e r } } ( u ) ) \wedge \mathsf { s t } ( u ) \bigr ] , } \\ & { e _ { \mathrm { m o n i t o r } } ( u ) = \downarrow _ { 0 . 5 \mathrm { s } } \mathsf { i n } ( u , \mathcal { R } _ { \mathrm { d e s t } } ( u ) ) . } \end{array}
$$

Checks with scripted trajectories. The evaluations of the six tasks other than conveyor loading are also tested with scripted trajectories, which stand in for a policy and are therefore not part of acceptance (§ 4.3.2). All 185 region checks against scene bodies match, the 20 start-state probes behave as expected, including a false result for each plausible non-start state, and 32 scripted trajectories labeled with their intended verdict reproduce all 32 labels (Table 50). The replay exposes one defect that the static and reset checks miss. Carry paths that dip back into a start volume 15 to 25 cm tall re-fire the start event and count protocol violations in nominal runs, 5 for cart push and 3 for barcode routing. Holding every start volume within 5 cm of the resting height and the start event for 30 frames removes all false alarms. No policy completes the jobs yet, so the replay uses scripted trajectories instead of recorded rollouts.

## F.4 Visual evidence

All seven benchmark evaluations determine job outcomes and run verdicts from privileged simulator state. The visual progress estimates reported below are diagnostic and carry no weight in these outcomes. We distinguish this use from an optional evaluator extension that can include explicitly declared visual conditions in the success criterion. That extension is not used by the seven benchmark evaluations, and its visual accuracy is not established here.

Optional visual conditions. The evaluator can bind a visual condition to frame windows selected by named events and combine its results with physical conditions when this additional rule is declared before execution. This differs from the evaluation generation pipeline of § 4.3, whose compiler records visual requirements as event-bound evidence but rejects their use in a success criterion. Physical predicates retain their simulator-derived values, and visual evidence cannot replace them. For a visual condition, model judgments or reviewed annotations resolve the available evidence, while missing or unresolved results remain unknown. The number of satisfying frames is therefore bounded by $[ L _ { \kappa } , U _ { \kappa } ]$ , where $L _ { \kappa }$ counts known satisfying frames and $U _ { \kappa }$ also includes unresolved frames. A declared threshold passes or fails only when these bounds determine the result, and otherwise it remains undecided. For an evaluation using this extension, let $K _ { \mathrm { e v a l } } =  { K _ { \mathrm { p h y s } } } \cup  { K _ { \mathrm { v i s } } }$ be its declared condition set. The outcomes of condition κ and of the run are

Table 50. Further verification of the evaluations of six tasks. Region checks place a scene body inside and outside each region, start-state probes move one item to a plausible non-start pose, and replay traces are scripted trajectories whose verdict is declared before the run, one success and the rest violations.
<table><tr><td>Task</td><td>Region checks</td><td>Start probes</td><td>Replay traces</td></tr><tr><td>Flow-rack loading</td><td>33/33</td><td>3/3</td><td>5/5</td></tr><tr><td>Cart push</td><td>30/30</td><td>4/4</td><td>5/5</td></tr><tr><td>Jig loading</td><td>25/25</td><td>3/3</td><td>5/5</td></tr><tr><td>Press tending</td><td>23/23</td><td>3/3</td><td>6/6</td></tr><tr><td>Valve operation</td><td>15/15</td><td>3/3</td><td>717</td></tr><tr><td>Barcode routing</td><td>59/59</td><td>4/4</td><td>4/4</td></tr><tr><td>All six</td><td>185/185</td><td>20/20</td><td>32/32</td></tr></table>

$$
y _ { \kappa } = A _ { \kappa } \left( \mathrm { A g g r e g a t e } ( \mathcal D ( x _ { 0 : N _ { \mathrm { s t e p } } } ) ) ) \right) \left( \kappa \in K _ { \mathrm { p h y s } } \right) , \qquad y _ { \kappa } = A _ { \kappa } \left( [ L _ { \kappa } , U _ { \kappa } ] \right) \left( \kappa \in K _ { \mathrm { v i s } } \right) , \qquad Y = \operatorname* { m i n } _ { \kappa \in K _ { \mathrm { c v a l } } } y _ { \kappa } ,\tag{46}
$$

where D detects events, Aggregate aggregates them into jobs and metrics, $A _ { \kappa }$ applies the threshold of condition κ, and the minimum follows the order fail < undecided < pass. Thus, a physical failure remains a failure, while a run that passes its physical conditions may fail or remain undecided on an additional visual condition. This rule specifies how evidence affects the verdict, not how accurately a visual judge recognizes the condition.

Visual judge. A visual judge receives one criterion and one ordered frame window, for example whether the tote remains visibly upright as it enters the X-ray tunnel, and returns satisfied or violated only when the frames show it, not visible when an occluder hides the relevant object, and unclear when the visible evidence does not resolve the criterion, with a rationale that names the visible basis rather than a reconstructed account of hidden robot actions. Uprightness is measured from simulator state in the actual conveyor-loading evaluation, and this example does not report a validation of visual acceptance.

Whole-run progress judges. The evaluator computes progress from the number of completed jobs and, within a job, the fraction of required events that have occurred, using the event definitions in the evaluation specification without frame-leve progress labels. A vision-language model that estimates progress from frames acts as a value function (Ma et al., 2025; Rao, 2026), and we record such whole-run judgments beside physical progress as a diagnostic. Following the in-context progress-judging setup of Ma et al. (2025), four models assess 36 labeled runs of line unloading, a diagnostic task that reverses conveyor loading: 8 successes and 28 failures (Table 51). Each run supplies 48 frames, and the judges return frame-level progress and an outcome using the prompt structure given at the end of this section. We compare outcomes with the physical evaluation and progress with completed-job counts and an elapsed-time baseline. The three models gpt-6.1-sol, gpt-6-astra, and claude-opus-5-5 call the same 8 failures successful and agree with one another on at least 34 of the 36 outcomes. gemini-4-argon calls 7 failures successful, 6 of them among those 8. It rejects a tote left on its side and the upside-down tote of Figure 26, which the others accept, and accepts a tote wedged at $4 2 ^ { \circ }$ , which they reject. All four agree on 32 of the 36 outcomes, so their errors are largely but not entirely shared. Across the four models, 25–29% of failed runs are judged successful. Their progress estimates correlate with completed-job counts more strongly than elapsed time does for three of the four judges, but whole-run visual judgments cannot substitute for the physical conditions in these trials, and they say nothing about the accuracy of the optional event-bound visual conditions above. Calibrated estimates may still direct a reviewer to runs worth inspecting.

Progress judge prompt. The judge receives the goal of the run, the frame indices with their times in the run, and five progress anchors from 0 (nothing relevant has changed) to 1 (the goal is visibly satisfied and stable), is told that the physics predicates decide success, and returns per-frame progress with a rationale and a whole-run outcome as JSON. With shuffled frames, the order and times are withheld.

Validating the optional visual condition path requires independently labeled conditions, including physical failures, visual failures, and missing or occluded evidence. A comparison with whole-run judging should hold the visual model and evidence budget fixed and report false acceptance, false rejection, and unresolved rates. For $N > 0$ assessed intervals, $N _ { a }$ non-abstaining answers, and $E _ { a }$ errors among those answers, coverage is $N _ { a } / N$ and conditional error is $E _ { a } / N _ { a }$ when $N _ { a } > 0$ . Both are needed because abstaining on difficult cases can reduce error without improving coverage. No such validation of event-bound visual conditions is reported here.

![](images/7d0d6e4205f5c0ab0c64af7187c2d882b1eef021881d006066dcaa8510e51a5f.jpg)

![](images/31cc12ab5b36c32bd99415e069ff04c37eda635e7830316480aef3167d297f98.jpg)  
Figure 26. Progress estimated by proprietary models against progress from physics on line unloading, conveyor loading in reverse.

Table 51. Progress judges against physics on line unloading, over 36 labeled runs. τ is the Kendall rank correlation between judged progress and the completed-job count over 48 frames, step-up the fraction of job completions at which judged progress rises by at least 0.1, agreement the fraction of runs whose judged outcome matches physics among those the judge does not leave unclear, and false success the fraction of failed runs judged achieved. With shuffled frames the outcome is undefined. Time ramp reports the elapsed fraction of the run as progress.
<table><tr><td>Judge (effort)</td><td>Frames</td><td>Tb</td><td>Step-up</td><td>Agreement</td><td>False success</td></tr><tr><td>Time ramp</td><td>Ordered</td><td>0.53</td><td>一</td><td>一</td><td>一</td></tr><tr><td>gpt-6.1-sol (medium)</td><td>Ordered</td><td>0.68</td><td>27%</td><td>78%</td><td>29%</td></tr><tr><td></td><td>Shuffled</td><td>0.73</td><td>52%</td><td>一</td><td>一</td></tr><tr><td>gpt-6-astra (high)</td><td>Ordered</td><td>0.69</td><td>26%</td><td>78%</td><td>29%</td></tr><tr><td></td><td>Shuffled</td><td>0.73</td><td>58%</td><td>一</td><td>一</td></tr><tr><td>claude-opus-5-5 (high)</td><td>Ordered</td><td>0.55</td><td>50%</td><td>76%</td><td>29%</td></tr><tr><td></td><td>Shuffled</td><td>0.61</td><td>52%</td><td>一</td><td>一</td></tr><tr><td>gemini-4-argon (high)</td><td>Ordered</td><td>0.63</td><td>44%</td><td>81%</td><td>25%</td></tr><tr><td></td><td>Shuffled</td><td>0.65</td><td>50%</td><td>一</td><td>一</td></tr></table>