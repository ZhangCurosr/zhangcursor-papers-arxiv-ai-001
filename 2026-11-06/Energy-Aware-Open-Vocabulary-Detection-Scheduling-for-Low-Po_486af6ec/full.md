# Energy-Aware Open-Vocabulary Detection Scheduling for Low-Power Robotic Autonomy

Abdul Basit and Muhammad Shafique

Abstract— Open-vocabulary object detection (OVD) lets robots search for run-time object queries, but fixed-rate inference spends embedded compute on low-value frames during idle motion, map revisits, or stable viewpoints. This paper presents an energy-aware OVD scheduler for low-power mobile robots. The scheduler uses odometry, map change, and detector confidence to switch among low-, mid-, and high-power YOLO-World-S modes on a Jetson Orin Nano. We evaluate the energyquality trade-off on an AgileX LIMO using 40 real navigation trials: two target objects, five placements per target-policy pair, and four perception policies. Across bottle and cup trials, the scheduler records 8/10 mission success compared with 7/10 for each fixed-rate baseline, while reducing mean Jetson input power (a Jetson-side power-draw metric) by 5.2% relative to an 8 Hz high-rate baseline and lowering achieved OVD call rate by 34.0%. It does not minimize mission-window Jetson input energy because longer missions can erase instantaneous power savings. A separate 60 s stationary microbenchmark isolates the perception workload: the scheduler lowers mean power from 8.045 W to 7.301 W and energy from 484.7 J to 438.1 J relative to the 8 Hz baseline. These results support robot-state-aware OVD scheduling as a promising power-management mechanism with competitive task reliability, while showing that missionwindow Jetson input energy must be read together with task progress and runtime.

## I. INTRODUCTION

Open-vocabulary object detection (OVD) lets mobile robots ground run-time object queries, such as “find the bottle near the doorway,” in their camera stream instead of searching only fixed trained labels. This capability is enabled by vision-language models and open-vocabulary detection/grounding systems, including CLIP [1], OWL-ViT [2], GLIP [3], Grounding DINO [4], Detic [5], and YOLO-World [6]. In robotics, similar representations support language-grounded maps, open-vocabulary scene graphs, and object-centric navigation [7]–[10].

The deployment problem is that OVD is expensive on embedded robots. A low-power robot must share a small battery across the embedded GPU, RGB-D camera, LiDAR, wireless stack, controller, and drive base. Mean Jetson input power is therefore a practical Jetson-side power-draw metric, especially for robots that keep monitoring or waiting after a target event. A fixed high detector rate gives fast response but wastes power during idle motion, map revisits, or lowinformation views. A fixed low-rate or quantized detector saves power but can delay target acceptance or miss short visual opportunities. The right operating point changes during the mission.

This motivates mission-level perception scheduling: each inference should be worth its energy at the current robot state. We ask whether an embedded robot can reduce Jetson input power for OVD while maintaining mission completion by scheduling detector rate from robot state. Our hypothesis is that low-value frames can be sampled sparsely, while candidate-target phases need high-rate response. A stationary frame in a stable map has lower marginal value than a frame near a plausible target; once a target appears, aggressive throttling is risky.

Existing work covers pieces of the problem but not the full intersection. OVD papers emphasize zero-shot recognition, language-grounded navigation emphasizes semantic goal reaching, and adaptive video analytics emphasizes throughput or resource budgets for passive streams. Our focus is more targeted: deciding when an embedded robot should spend OVD compute during search. As summarized in Table I, prior works generally do not combine OVD, robotstate-aware adaptation, Jetson input-power/energy accounting, and task-level metrics such as success rate and time-todetect (TTD).

We present an energy-aware OVD scheduler that treats detector invocations as mission-level computational resources. The scheduler uses odometry, map changes, and detector confidence to adaptively switch among low-, mid-, and highpower operating modes. We evaluate YOLO-World-S on a Jetson Orin Nano mobile robot against fixed-rate 8 Hz, 5 Hz, and 2 Hz baselines across bottle and cup search tasks. Fig. 1 illustrates the overall runtime flow.

Our main contributions are:

• We formulate energy-aware OVD as a mission-level scheduling problem that trades Jetson input power and detector invocation rate against search success and TTD, rather than only reducing per-frame inference cost.

• We design a finite-state runtime that switches OVD rate from robot-native signals with explicit thresholds and hysteresis, including a low-power configuration for lowvalue frames.

• We implement the scheduler with YOLO-World-S on an AgileX LIMO equipped with a Jetson Orin Nano and compare it against fixed-rate high-, medium-, and low-rate baselines.

• We report a real-robot evaluation with 40 navigation trials across two queried objects and a separate stationary microbenchmark that isolates perception power.

• We show empirically that lower mean power does not necessarily imply lower mission-window Jetson input energy, because slower or less reliable perception can increase navigation time.

The main finding is that lower instantaneous detector power is only one part of the robot objective. The scheduler reduces mean Jetson input power relative to the high-rate baseline and shows slightly higher aggregate success in the evaluation logs, but mission-window Jetson input energy still depends strongly on runtime. We therefore report power, energy, success, TTD, and time-domain traces rather than optimizing energy in isolation.

## II. RELATED WORK

Table I positions representative prior work by this paper’s evaluation axes. The comparison is qualitative: the proposed scheduler builds on OVD models but studies their coupling with robot-state-aware scheduling, Jetson inputpower/energy accounting, and task-quality metrics.

TABLE I  
QUALITATIVE COVERAGE OF REPRESENTATIVE RELATED WORK.
<table><tr><td>Work</td><td></td><td></td><td>OVD Robot Adapt. R/E Task</td><td></td><td></td></tr><tr><td>OWL-ViT/GLIP/GDINO [2]–[4]</td><td></td><td>×</td><td>X</td><td>X</td><td>×</td></tr><tr><td>Detic/YOLO-World [5], [6]</td><td></td><td>×</td><td>X</td><td></td><td>×</td></tr><tr><td>VLMaps/LM-Nav/ROVER [7], [10], [11]</td><td></td><td></td><td>X</td><td>X</td><td></td></tr><tr><td>OV scene graphs [8], [9]</td><td></td><td></td><td>X</td><td></td><td></td></tr><tr><td>Active/NBV perception [12], [13]</td><td>×</td><td></td><td></td><td></td><td></td></tr><tr><td>NoScope/Chameleon [14], [15]</td><td>×</td><td>×</td><td></td><td></td><td>X</td></tr><tr><td>ApproxDet [16]</td><td>×</td><td>×</td><td></td><td></td><td>×</td></tr><tr><td>Proposed scheduler</td><td>了</td><td></td><td></td><td></td><td></td></tr></table>

Note: Robot = robot task evaluation; Adapt. = runtime adaptation; $\overline { { R / E = { \mathrm { r e } } } } -$ source/energy metric; Task = success/TTD. ✓= covered, ▲= partial/indirect, ×= not a primary focus.

Modern OVD systems inherit generalization from imagetext pretraining. CLIP [1] introduced transferable visionlanguage representations, and OWL-ViT [2], GLIP [3], RegionCLIP [17], Detic [5], Grounding DINO [4], and YOLO-World [6] adapt them to region-level localization. These works emphasize recognition accuracy, vocabulary generalization, and frame latency, but rarely ask how often an embedded robot should invoke OVD during a mission.

Open-vocabulary perception also supports semantic robot behavior. VLMaps [7], LM-Nav [10], open-vocabulary scene graphs [8], ConceptGraphs [9], ROVER [11], and RoboAuditor [18] use language-grounded perception for navigation, mapping, planning, object search, or inspection. These systems show the utility of open vocabularies for robots, but not the scheduling and energy cost of the detector.

Efficient inference and adaptive perception provide the closest systems context. Approximate video analytics adapts resolution, model size, or sampling rate for passive camera workloads [14]–[16]; edge systems partition or accelerate neural computation [19]–[21]; continual-learning surveys highlight deployment adaptation challenges in autonomous systems [22]; and active/NBV perception decides where or when to sense using task state or information gain [12], [13]. In contrast, a mobile OVD robot has mission-native signals such as odometry, map change, and target confidence, and a missed target can change task outcome. These systems are complementary but optimize frame-level or edge-cloud decisions, whereas our evaluation couples OVD scheduling to closed-loop success, TTD, and mission-window power. This paper targets that intersection by treating OVD calls as mission resources and evaluating robot-state-aware scheduling with Jetson input energy, success, and TTD.

## III. METHODOLOGY

## A. Problem Formulation

We consider a differential-drive mobile robot performing language-guided object search using an embedded openvocabulary detector (OVD). For each mission, the robot is provided with a target query q and autonomously searches the environment until either the task is completed or a mission duration $\tau _ { \mathrm { m a x } }$ is reached. Perception and navigation operate continuously throughout the mission, while detector execution is governed by the proposed energy-aware scheduling policy. We define TTD as the elapsed time from mission start to the first detection of the queried object; this event is used as an evaluation metric and does not terminate the mission. At time t, the perception runtime selects $a ( t ) =$ $( f ( t ) , c _ { \mathrm { r e q } } ( t ) )$ , where $f ( t )$ is the requested OVD rate and $c _ { \mathrm { r e q } } ( t ) \ \in$ {FP16, INT8} is the requested inference mode. The mission-window Jetson input energy is

$$
E _ { \mathrm { m } } = \int _ { 0 } ^ { \mathcal { T } } P _ { \mathrm { j e t s o n } } ( t ) d t ,\tag{1}
$$

where $\tau \leq \tau _ { \operatorname* { m a x } }$ is the realized duration. Because energy alone can be minimized by never running the detector, we measure search quality using success rate and time-to-detect,

$$
\mathrm { T T D } = t _ { \mathrm { f i r s t ~ c o n f } } - t _ { \mathrm { s t a r t } } .\tag{2}
$$

where $t _ { \mathrm { f i r s t ~ c o n f } }$ is the first target detection above $\tau _ { \mathrm { c o n f } } .$ TTD is a perception-timing metric, not the success label. Aggregate TTD is reported over successful runs only; failures remain in success counts, runtime, and mission-window Jetson input energy.

The design objective is to reduce Jetson input power and detector workload while maintaining success and acceptable TTD relative to a high-rate detector baseline. We also report $E _ { \mathrm { m } }$ to capture the effect of mission duration. Unlike offline detector compression, the scheduler is conditioned on robot state and spends computation unevenly across the mission.

## B. System Overview

Fig. 1 shows the runtime. The robot runs ROS Noetic with GMapping and move base. The OVD node receives RGB frames from an Astra/Dabai U3 RGB-D camera and evaluates YOLO-World-S for the queried class. Requested rates are upper bounds: the camera continues streaming, but frames reach the detector only when the scheduler permits a new OVD call. A state monitor subscribes to odometry, the occupancy grid, and detector outputs. A logger records scheduler events, poses, detector calls, mission outcomes, and Jetson power from tegrastats.

The scheduler uses three robot-state signals:

• Motion: linear speed $v ( t )$ from odometry, used to identify idle or nearly idle intervals.

• Map change: normalized occupancy-grid entropy change $\Delta H ( t )$ , used as a proxy for whether new space is being observed.

• Detection confidence and target proximity: best target confidence $p ( t )$ from the OVD node and proximity to a recent target estimate, used to enter a high-rate candidate-response mode.

![](images/d38d10faa7a9b062682cae00963e7022f2b8ec6551275c73a8d4e55132c5d75e.jpg)  
Fig. 1. Proposed energy-aware OVD runtime. Robot sensing and mapping produce motion, map-change, confidence, and target-distance signals. The tri-state scheduler maps these signals to LOW/MID/HIGH detector modes, while the logging pipeline records Jetson input power, mission-window energy, inference count, success, TTD, and mode occupancy.

For an occupancy grid window $\Omega _ { t } .$ , normalized map entropy is computed as

$$
\begin{array} { l } { \displaystyle { H ( t ) = \frac { 1 } { | \Omega _ { t } | \log 2 } \sum _ { i \in \Omega _ { t } } h _ { i } ( t ) , } } \\ { \displaystyle { h _ { i } ( t ) = - p _ { i } ( t ) \log p _ { i } ( t ) - ( 1 - p _ { i } ( t ) ) \log ( 1 - p _ { i } ( t ) ) . } } \end{array}\tag{3}
$$

Unknown cells use $p _ { i } = 0 . 5$ , and known cells use normalized occupancy probability clipped to $[ \epsilon , 1 - \epsilon ]$ . We use a 4.0 m × 4.0 m robot-centered window; if successive GMapping maps differ in extent, ∆H uses overlapping indexed cells. The scheduler updates at $r _ { s } ~ = ~ 2 0$ Hz, so $\Delta H ( t ) = | H ( t ) - \Delta H ( t ) |$ $H ( t - \Delta t _ { s } ) |$ with $\Delta t _ { s } ~ = ~ 1 / r _ { s }$ . Detector confidence $p ( t )$ updates after each OVD call. Only fresh detector output can satisfy the confidence branch of $M  H ;$ the distance branch uses an unexpired target estimate. The estimate comes from RGB-D box-center back-projection to the map frame, LiDAR snapping within 0.3 m, and exponential filtering with $\alpha = 0 . 3 ;$ after 30 s without update, $d _ { \mathrm { t a r g e t } } = \infty$

## C. Tri-State Scheduling Policy

The proposed scheduler uses a finite-state policy with hysteresis. Each state has a fixed target rate and preferred inference mode:

• LOW: requested INT8. Used after sustained evidence that the robot is nearly idle and the local map is stable. This state samples frames sparsely to retain target awareness without continuous high-rate detection.

• MID: FP16. Used during normal navigation and mapping, where new viewpoints arrive but there is no confident target.

• HIGH: FP16. Used when a target candidate is detected or the robot is near a recent target estimate, prioritizing high-rate response over energy.

HIGH denotes a high-power candidate-response state rather than a distinct long-term tracking mode. The scheduler operates deterministically: transitions to HIGH are triggered by moderate evidence of the target, whereas entry into LOW is permitted only after sustained inactivity and persistent evidence of map stability.

For compactness, let L, M, and H denote LOW, MID, and HIGH. The transition rules are:

$$
L \to M : v > v _ { \mathrm { i d l e } } \lor \Delta H > \tau _ { H } ,\tag{4}
$$

$$
M  H : p \geq \tau _ { \mathrm { h i g h } } \lor d _ { \mathrm { t a r g e t } } \leq d _ { \mathrm { n e a r } } ,\tag{5}
$$

$$
H  M : p < \tau _ { \mathrm { h y s t } } \mathrm { f o r } K _ { \mathrm { l o s t } } \mathrm { e v a l s . } ,\tag{6}
$$

$$
M  L : v < v _ { \mathrm { i d l e } } \ \wedge \ \Delta H < \tau _ { H } \ \mathrm { f o r } \ T _ { \mathrm { l o w } } .\tag{7}
$$

Table II lists parameters used for all reported runs. They are fixed rather than tuned per trajectory: the system enters HIGH at moderate confidence but requires persistent low confidence before leaving.

TABLE II  
SCHEDULER PARAMETERS USED IN THE REPORTED EXPERIMENTS.
<table><tr><td>Parameter</td><td>Value</td><td>Purpose</td></tr><tr><td> $r _ { s }$ </td><td>20 Hz</td><td>Scheduler update rate</td></tr><tr><td> $\bar { f _ { L } } / f _ { M } / f _ { H }$ </td><td>1.5/4/8 Hz</td><td>Requested OVD rates</td></tr><tr><td> $\ell _ { \Omega }$ </td><td>4.0 m</td><td>Local grid-window side</td></tr><tr><td> $v _ { \mathrm { i d l e } }$ </td><td>0.03 m/s</td><td>Idle-motion threshold</td></tr><tr><td> $\underline { { \tau } } _ { H }$ </td><td>0.02</td><td>Map-change threshold</td></tr><tr><td> $T _ { \mathrm { l o w } }$ </td><td>2.0 s</td><td>Stable interval before LOW</td></tr><tr><td> $\tau _ { \mathrm { h i g h } }$ </td><td>0.25</td><td>Confidence to enter HIGH</td></tr><tr><td>Tconf</td><td>0.25</td><td>Detection acceptance threshold</td></tr><tr><td> $\tau _ { \mathrm { i n t } 8 }$ </td><td>0.20</td><td>INT8 detector threshold</td></tr><tr><td> $\tau _ { \mathrm { h y s t } }$ </td><td>0.20</td><td>Confidence to leave HIGH</td></tr><tr><td> $d _ { \mathrm { n e a r } }$ </td><td>2.0 m</td><td>Near-target HIGH trigger</td></tr><tr><td> $K _ { \mathrm { l o s t } }$ </td><td>5 evals.</td><td>HIGH hysteresis window</td></tr></table>

## D. Low-Cost Inference Mode

The perception runtime maintains both the high-rate YOLO-World-S inference path and the INT8 TensorRT engine in memory, selecting the appropriate execution mode according to the scheduler state. INT8 inference is used exclusively in LOW mode, where frames are processed at reduced precision to support coarse target awareness with lower computational cost. MID and HIGH modes use the FP16 inference path to provide higher-fidelity localization and response. The INT8 decision threshold, $\tau _ { \mathrm { i n t } 8 } ,$ is applied only within the LOW-mode detector, whereas missionlevel detection decisions and time-to-detection (TTD) are evaluated using the common confidence threshold $\tau _ { \mathrm { c o n f } }$ . All reported experiments were conducted with the INT8 engine available and without precision fallback.

## E. Metrics

For mission runs we log success, TTD, runtime, path length, OVD invocations $N _ { \mathrm { i n f } }$ , average OVD rate $N _ { \mathrm { i n f } } / \tau$ state occupancy, mean Jetson input power, and missionwindow energy. State occupancy is the realized time fraction in LOW/MID/HIGH using the corresponding raw labels. Power is scoped to the Jetson module, not the full robot battery: the analysis uses tegrastats VDD IN total W at a 50 ms requested sampling interval, with GPU and SoC rails used only as diagnostics. For every reported trial, the power trace is aligned to mission start/end wall time and energy is integrated by a trapezoidal sum over valid inwindow samples. Failed missions remain in all-run averages because they consume real time and power.

## IV. EXPERIMENTAL SETUP

## A. Robot Platform

![](images/fd04038a199162d7e0ee9da6efb05a0a79bf09043f4f7a66a702f643189e8c73.jpg)  
Fig. 2. Real-robot arena with fixed LIMO start pose and bottle/cup target placements reused across policies.

Experiments use an AgileX LIMO differential-drive robot with an Astra/Dabai U3 RGB-D camera, 2D LiDAR, and NVIDIA Jetson Orin Nano 8 GB. The robot runs ROS Noetic, GMapping, and move base. The OVD node runs YOLO-World-S with one queried target class. Final navigation trials use “bottle” and “cup” queries to test scheduling across two targets.

## B. Policies Compared

We compare four perception policies:

• Base-8Hz: fixed 8 Hz high-power mode (TRACK).

• Base-5Hz: fixed 5 Hz mid-power mode (EXPLORE).

• Base-2Hz: fixed 2 Hz low-power INT8 mode (SLEEP).

• Proposed: tri-state low-/mid-/high-power scheduling. The paper reports LOW/MID/HIGH as conceptual modes and keeps SLEEP/EXPLORE/TRACK as labels where needed. Achieved rates can be below requested values because detector latency and gating limit calls, so results report measured invocation rates.

## C. Mission Protocol

The real-robot evaluation comprises 40 trials spanning two target objects, four perception policies, and five repetitions for each object–policy combination. As shown in Fig. 2, all trials begin from a fixed robot pose, with each target placed at one of five predefined locations. The same placements are reused across policies to support controlled comparison. Bottle and cup trials are conducted in separate batches;

between trials, the target is moved to the next location and the navigation stack is reinitialized.

Each trial has a maximum duration of 180 s. Experiments are executed in policy-specific blocks rather than randomized order, so gradual changes in battery state, illumination, or system temperature may introduce temporal drift. Each policy is therefore evaluated across all five placements, and aggregate statistics are reported to reduce sensitivity to individual trial conditions. For every trial, the navigation stack is restarted, system power is monitored externally using tegrastats, and mission outcomes are recorded by the energy-aware navigation framework. The mission-level success field in the corresponding mission summary record is used for evaluation.

In addition to the navigation trials, we perform a 60 s stationary microbenchmark for each policy to characterize perception-related power consumption independently of robot motion and navigation workload. During this benchmark, the camera, detector or scheduler, and power logger remain active, while navigation, SLAM, move base, and velocity-command execution are disabled. For the proposed policy, scripted inputs exercise the scheduler across low-, mid-, and high-response phases, whereas each fixed-rate baseline remains in its assigned operating mode. This benchmark isolates the contribution of detector execution rate and scheduler state composition from variability introduced by the navigation stack.

## D. Metrics and Statistics

We use the metrics in Section III-E. Navigation results cover all 40 trials, with aggregate summaries over bottle and cup. TTD is averaged over successful runs only, while Jetson energy and runtime are averaged over all runs because failures consume real time and power. The stationary benchmark is reported separately as a mechanism check. With five placements per object-policy pair, we report descriptive rather than inferential statistics.

## V. RESULTS AND DISCUSSION

## A. Navigation Results

TABLE III  
REAL-ROBOT NAVIGATION SUMMARY OVER BOTTLE AND CUP TRIALS(TEN TRIALS PER POLICY). LOW-, MID-, AND HIGH-POWER MODES.
<table><tr><td>Policy</td><td>Succ.</td><td> $\bar { P } _ { J }$  (W)</td><td> $E _ { \mathrm { m } }$  (J)</td><td>(s)</td><td>Run TTDsucc (s)</td><td>Hz</td><td>L/M/H (%)</td></tr><tr><td>Base-8Hz 7/10</td><td></td><td>9.35</td><td>866.9</td><td>92.8</td><td>25.9</td><td>4.74</td><td>0/0/100</td></tr><tr><td>Base-5Hz 7/10</td><td></td><td></td><td>8.92 1019.8114.2</td><td></td><td>53.6</td><td></td><td>3.39 0/100/0</td></tr><tr><td>Base-2Hz 7/10 8.38 1155.0 140.4</td><td></td><td></td><td></td><td></td><td>0.9</td><td></td><td>1.48 100/0/0</td></tr><tr><td>Proposed 8/10 8.86 1114.0 125.9</td><td></td><td></td><td></td><td></td><td>10.9</td><td></td><td>3.12 25/60/15</td></tr></table>

Note: $E _ { \mathrm { m } }$ is mission-window Jetson input energy; TTD is conditional on success. L/M/H maps raw log labels to low-/mid-/high-power modes.

Table III summarizes the 40 real-robot navigation trials. The proposed scheduler has the highest aggregate success count, 8/10, while each fixed-rate baseline succeeds in 7/10 trials. It reduces mean Jetson input power from 9.35 W for Base-8Hz to 8.86 $\mathbf { W } ,$ a 5.2% reduction, and lowers achieved OVD rate from 4.74 Hz to 3.12 Hz. Runtime is the missionwindow duration over all runs; $\mathrm { T T D } _ { \mathrm { s u c c } }$ measures only first accepted detection on successful runs. Base-2Hz has the lowest mean power, 8.38 W, but the longest mean runtime and largest mean Jetson input energy. Accordingly, the navigation result is best read as power/workload reduction at competitive task reliability, not as a statistically large reliability improvement.

![](images/0f48ef1823ff73ad1f5e6780c37144a212db50a28e90d198f459fb18ec5c5e28.jpg)

![](images/7c816dc01a450f28c17813684770d18cf3e99020b42b9f958c5c54e32edc6288.jpg)

Fig. 3. Policy dashboard over the 40 navigation trials. The proposed scheduler has the highest aggregate success count and lower mean Jetson input power than Base-8Hz, while Jetson input energy is dominated by runtime.  
![](images/0dc4c2acfec8ad9075e5476798c15400a33799692d98cd2f7c9b2ff9c8c5fc57.jpg)  
Fig. 4. Mission runtime versus Jetson input energy for all 40 trials. Circles are successes, crosses failures, and open diamonds condition means; dashed lines show constant-power contours.

Fig. 3 visualizes the same summary. The main claim is a power-management trade-off, not guaranteed $E _ { \mathrm { m } }$ minimization. Relative to Base-8Hz, the scheduler reduces mean power and detector calls, yet mean $E _ { \mathrm { m } }$ is higher because missions last 35.8% longer on average. Base-2Hz is clearer: it cuts mean power by 10.3%, but runtime rises by 51.4%, producing the largest mean $E _ { \mathrm { m } }$ . Its low $\mathrm { T T D } _ { \mathrm { s u c c } }$ reflects successful easy cases, not generally faster search. Fig. 4 shows why energy-aware OVD must be evaluated as a closed-loop workload.

## B. Time-Domain Scheduler Behavior

![](images/d08e1a61bccc7bc5082a8d8a84e04c78888de6233c7250dea982394d74b91744.jpg)  
Fig. 5. Compact proposed-scheduler trace from the 60 s scripted stationary run. Colored regions show state, orange ticks show detector calls, the blue curve is Jetson input power, and the red dashed line is mean power.

Fig. 5 shows the proposed scheduler in the 60 s scripted stationary run. This is not a navigation trial: it isolates detector/scheduler power from path length, obstacle avoidance, and mission duration. The run has 7.30 W mean Jetson input power, consumes 438.1 J, executes 202 detector calls, and spends 50/38/13% of the horizon in LOW/MID/HIGH modes. The power step between LOW and MID/HIGH makes the mechanism behind Table IV visible in time.

![](images/6bb996fbcfb73c08840a7ce961540890696c47006c34ecb12bbee2bc8176a0ab.jpg)  
Fig. 6. Representative adaptive navigation trace showing Jetson input power/energy, scheduler state and detector calls, detector confidence with the 0.25 threshold, and path length.

Fig. 6 complements the stationary trace with a closedloop run. It shows the same scheduler while the robot moves, accumulates path length, and receives intermittent target-confidence events. The run is illustrative, exposing timing relationships among power, state, confidence, and path progress. Fig. 7 adds visual context from two run videos. Early frames show stable or low-value viewpoints where LOW is appropriate, middle frames show obstacleaware navigation where MID maintains periodic OVD, and late frames show target-adjacent candidate response where HIGH is appropriate. The figure is qualitative; quantitative occupancy and timing come from synchronized logs in Tables III and IV and Fig. 6.

## C. Stationary Power Mechanism

The stationary microbenchmark isolates detector and scheduler power from navigationrelated variability. OVD frequency is computed from detector invocations within the same time window used for power and energy inte-

TABLE IV  
STATIONARY MICROBENCHMARK OVER THE POWER-INTEGRATION WINDOW.
<table><tr><td>Policy</td><td> $\bar { P } _ { J }$  E  $( \mathbf { W } )$  (J)</td><td>Hz #Inf. L/M/H</td><td>(%)</td></tr><tr><td>Base-8Hz 8.045 Base-5Hz 7.578456.64.40</td><td>484.76.67</td><td>402 0/0/100</td><td></td></tr><tr><td>Base-2Hz 6.847 410.8 1.90 Proposed 7.301 438.1 3.37</td><td></td><td></td><td>265 0/100/0 114 100/0/0 202 50/38/13</td></tr></table>

Note: OVD Hz is computed as #Inf./mission-window duration.

gration. The fixed-rate baselines exhibit the expected monotonic reduction in power as the detector rate decreases from Base-8Hz to Base-5Hz and Base-2Hz.

The proposed scheduler operates between the Base-5Hz and Base-2Hz baselines, reflecting its mixed operating profile: approximately 50% of the benchmark is spent in LOW, 37.6% in MID, and 12.5% in HIGH. Relative to Base-8Hz, the proposed policy reduces mean power by 9.2% and energy consumption over the 60 s benchmark by 9.6%.

![](images/26c77f85c44abbe5f61985a3909147ecef214ad611d6f3bed4eb454b3fde5bc3.jpg)

Fig. 7. Qualitative video-frame progression for representative bottle and cup searches. Blue R marks the robot and red T marks the target; LOW/MID/HIGH labels are qualitative stage estimates, not synchronized scheduler labels.  
![](images/c3ed56db513e9d11dc28919efb4d60c44598f1ddfa8c1c02d428a36a73682a9e.jpg)

![](images/e08291ebdd4b51871ea4a37db518f66fe8cb0365b229abf8b6e4d2f47ac01dc7.jpg)  
Fig. 8. Object-level success and state occupancy, mapped to LOW/MID/HIGH power modes. In the top plot, solid bars denote bottle and hatched bars denote cup.

## D. Object-Level Observations

The combined result hides object-level variation. On bottle trials, Base-8Hz succeeds in 5/5 runs, the proposed scheduler in 4/5, and the 5 Hz and 2 Hz baselines in 3/5 each. On cup trials, the proposed scheduler, Base-5Hz, and Base-2Hz each succeed in 4/5, while Base-8Hz succeeds in 2/5. Fig. 8 shows this split and state mix. Thus, the adaptive policy has the highest aggregate success count here, but the sample is too small to claim universal dominance.

## VI. CONCLUSION

This paper presented an energy-aware OVD scheduler for low-power mobile robots. It treats detector frames as mission resources: low-value frames can be sampled slowly, while candidate-target phases benefit from high-rate response. In 40 real AgileX LIMO navigation trials across bottle and cup searches, the scheduler shows 8/10 success compared with 7/10 for each fixed-rate baseline and reduces mean Jetson input power by 5.2% relative to Base-8Hz. The same trials show the key caution: lower mean power does not guarantee lower mission-window Jetson energy, because longer runtime can dominate the integral. A stationary microbenchmark confirms the mechanism, with 9.6% lower 60 s energy than Base-8Hz. Future work will broaden objects and environments, add external labels, randomize trials, measure wholerobot energy, and study motion-predictive scheduling.

## REFERENCES

[1] A. Radford, et al., “Learning transferable visual models from natural language supervision,” in Proceedings of the 38th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 139. PMLR, 2021, pp. 8748–8763.

[2] M. Minderer, et al., “Simple open-vocabulary object detection,” in Computer Vision – ECCV 2022, ser. Lecture Notes in Computer Science. Springer, 2022, pp. 728–755.

[3] L. H. Li, et al., “GLIP: Grounded language-image pre-training,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 10 965–10 975.

[4] S. Liu, et al., “Grounding DINO: Marrying DINO with grounded pretraining for open-set object detection,” in Computer Vision – ECCV 2024, ser. Lecture Notes in Computer Science. Springer, 2024, pp. 38–55.

[5] X. Zhou, et al., “Detecting twenty-thousand classes using image-level supervision,” in Computer Vision – ECCV 2022, ser. Lecture Notes in Computer Science. Springer, 2022, pp. 350–368.

[6] T. Cheng, et al., “YOLO-World: Real-time open-vocabulary object detection,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 16 901–16 911.

[7] C. Huang, et al., “Visual language maps for robot navigation,” in 2023 IEEE International Conference on Robotics and Automation (ICRA), 2023, pp. 10 608–10 615.

[8] A. Werby, et al., “Hierarchical open-vocabulary 3d scene graphs for language-grounded robot navigation,” in Robotics: Science and Systems XX, ser. RSS2024. Robotics: Science and Systems Foundation, July 2024.

[9] Q. Gu, et al., “ConceptGraphs: Open-vocabulary 3d scene graphs for perception and planning,” in 2024 IEEE International Conference on Robotics and Automation (ICRA), 2024, pp. 5021–5028.

[10] D. Shah, et al., “LM-Nav: Robotic navigation with large pre-trained models of language, vision, and action,” in Proceedings of The 6th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 205. PMLR, 2023, pp. 492–504.

[11] A. Basit, N. Pudasaini, and M. Shafique, “ROVER: Autonomous openvocabulary object searching in unexplored environments using VLMdriven scene understanding,” in 2025 International Joint Conference on Neural Networks (IJCNN), 2025, pp. 1–8.

[12] R. Bajcsy, “Active perception,” Proceedings ofthe IEEE, vol. 76, no. 8, pp. 966–1005, 1988.

[13] C. Korbach, et al., “Next-best-view estimation based on deep reinforcement learning for active object classification,” 2021, arXiv:2110.06766.

[14] D. Kang, et al., “NoScope: Optimizing neural network queries over video at scale,” Proceedings of the VLDB Endowment, vol. 10, no. 11, pp. 1586–1597, 2017.

[15] J. Jiang, et al., “Chameleon: Scalable adaptation of video analytics,” in Proceedings of the 2018 Conference of the ACM Special Interest Group on Data Communication, 2018, pp. 253–266.

[16] R. Xu, et al., “ApproxDet: Content and contention-aware approximate object detection for mobiles,” in Proceedings of the 18th Conference on Embedded Networked Sensor Systems, 2020, pp. 449–462.

[17] Y. Zhong, et al., “RegionCLIP: Region-based language-image pretraining,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 16 793–16 803.

[18] W. Cai, L. Huang, and Z. Zou, “Roboauditor: Goal-oriented robotic system for assessing energy-intensive indoor appliance via visual language models,” in Proceedings of the 10th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation, ser. BuildSys ’23. New York, NY, USA: Association for Computing Machinery, 2023, pp. 130–139.

[19] Y. Kang, et al., “Neurosurgeon: Collaborative intelligence between the cloud and mobile edge,” in Proceedings of the Twenty-Second International Conference on Architectural Support for Programming Languages and Operating Systems, 2017, pp. 615–629.

[20] S. Han, et al., “EIE: Efficient inference engine on compressed deep neural network,” in 2016 ACM/IEEE 43rd Annual International Symposium on Computer Architecture (ISCA), 2016, pp. 243–254.

[21] Y.-H. Chen, et al., “Eyeriss: An energy-efficient reconfigurable accelerator for deep convolutional neural networks,” in 2016 IEEE International Solid-State Circuits Conference (ISSCC), 2016, pp. 262– 263.

[22] K. Shaheen, et al., “Continual learning for real-world autonomous systems: Algorithms, challenges and frameworks,” Journal of Intelligent & Robotic Systems, vol. 105, no. 1, p. 9, 2022.