# GraspBench: Execution-Aware Cross-Robot Evaluation of 6-DoF Grasp Generation in MuJoCo

Abdul Basit<sup>∗</sup>, Bibek Poudel<sup>∗</sup>, and Muhammad Shafique

Abstract— Modern 6-DoF grasp generators are often evaluated using grasp scores, dataset-level metrics, or single-robot demonstrations, yet proposal quality alone does not determine whether a grasp is executable by a particular robot. We present GraspBench, a controlled MuJoCo protocol for evaluating complete grasp-generation pipelines after robot-specific inversekinematics filtering, motion planning, and fixed approach-closelift execution. GraspBench evaluates six configured pipelines across four robot embodiments, five diagnostic objects, and a structured tabletop pose grid, yielding 540 scenarios per method and 3,240 method-scenario evaluations in total. Missing outputs are conservatively counted as failures, and success follows the binary lift criterion produced by each method’s execution runner. GraspGen achieves the highest overall success rate (73.1%), followed by EconomicGrasp (63.5%), while robot- and object-conditioned analyses reveal substantial variation across embodiments and object geometries. GPG produces the largest native candidate sets but achieves 44.3% success, indicating that proposal volume alone does not guarantee robot-executable grasps. Cluster-aware paired intervals support the principal method differences while not estimating repeated-execution uncertainty. The results characterize complete configured pipelines on fixed scenarios rather than intrinsic model quality or stochastic repeatability, showing that grasping performance depends jointly on the generator, robot embodiment, scene, and execution protocol.

## I. INTRODUCTION

Reliable robotic grasping remains a central requirement for practical manipulation. Recent 6-DoF grasp generators can produce dense or diverse candidate grasps from point clouds, and large-scale datasets have accelerated progress in grasp prediction [1], [2]. Yet a high-scoring grasp proposal is not necessarily executable by a specific robot. The final outcome depends on how the predicted grasp distribution interacts with object geometry, workspace placement, robot kinematics, approach feasibility, collision constraints, and the execution controller. A well-ranked proposal may be unreachable for one arm, feasible for another, or invalid during approach and lift.

This issue becomes sharper across robot embodiments. Differences in reach, joint limits, wrist geometry, and gripper placement can change which candidates survive feasibility filtering. Aggregate success can therefore hide structured failures: a method may perform well on box-like shapes but degrade on thin or concave geometries, succeed near the table center but fail near workspace boundaries, or generate many candidates that downstream constraints discard. These observations motivate evaluating grasp generation as part of an execution pipeline rather than as pose prediction alone.

Existing grasp resources provide indispensable scale and standardization. GraspNet-1Billion offers dense RGB-D grasp annotations and benchmark metrics [1]; ACRONYM provides millions of simulated grasps over thousands of objects [2]; and Dex-Net demonstrated the value of synthetic grasp analysis for robust manipulation policies [3]. In parallel, grasp generators such as GPG, the geometric candidate generator associated with GPD [4], Contact-GraspNet [5], AnyGrasp [6], EconomicGrasp [7], GraspGen [8], and HGGD [9] have advanced geometric, contact-based, dense, budget-aware, generative, and heatmap-guided grasp proposal mechanisms. These works address important parts of the grasping problem, but dataset-level annotations and method-specific demonstrations do not fully answer a complementary question: how do different generators behave when their outputs are routed through a common scenarioindexedfeasibility and simulated-execution structure? Unlike dataset-level grasp benchmarks that evaluate pose annotations or generator outputs independently of a fixed execution stack, GraspBench fixes the robot, object, workspace, and execution protocol so that generated grasps can be compared under robot-conditioned feasibility constraints.

We study this question through four research questions:

• How much do 6-DoF grasp-generator rankings change across robot embodiments?

• Which object, workspace, and pose conditions expose failures hidden by aggregate success?

• How do method-native candidate-set sizes relate descriptively to simulated execution success?

• How does failure-aware accounting affect denominator consistency and reported aggregate performance when methods fail to return outputs?

GraspBench evaluates a shared robot-object-workspacepose grid, enabling paired comparisons across AUBO i5, Franka, KUKA iiwa, and PiPER; five diagnostic objects; workspace positions; and object pitch. For each scenario, a generator proposes grasps, robot-specific filtering selects the highest-ranked feasible candidate, the motion planner produces the robot trajectory, and MuJoCo executes a fixed approach-close-lift sequence scored by its recorded binary lift outcome.

The evaluated pipelines span dense perception (AnyGrasp), contact-based prediction (Contact-GraspNet), geometric sampling (GPG), budget-aware generation (EconomicGrasp), generative modeling (GraspGen), and heatmap-guided detection (HGGD). We evaluate execution consequences under scenario-aligned downstream stages, not training recipes or implementation efficiency.

Conservative scenario-level accounting counts missing outputs as failures instead of removing them. This prevents missing scenarios or repeated rows from changing the common denominator. Normalized records retain available candidate counts, scores, condition identities, and failure labels for reproducible scenario-level analysis.

Three observations emerge. First, method gaps and rankings depend on robot embodiment and object geometry. Second, native proposal volume has no apparent monotonic relation with execution success, although differing budgets preclude a causal candidate-count comparison. Third, missing and explicit no-grasp outputs are execution-relevant outcomes; counting them as failures provides a conservative common denominator rather than excluding absent records.

The main contributions of this paper are:

• Execution protocol: a controlled MuJoCo benchmark protocol for evaluating 6-DoF grasp generators across multiple robot embodiments under a scenario-aligned IK-filtered execution pipeline.

• Scenario diagnostics: a structured robot-objectworkspace-pose grid and paired scenario-level analysis that reveal robot-, object-, workspace-, and poseconditioned failure patterns.

• Failure-aware study: a conservative evaluation of six configured grasp-generation pipelines showing that aggregate and condition-specific execution outcomes can diverge.

The study is intentionally limited to controlled, singleobject simulation. Its five geometries are diagnostic rather than representative of broad object diversity, and the opaque glass model does not test transparent-object perception. GraspBench therefore complements, rather than replaces, cluttered-scene benchmarks, candidate-level Success@K analysis, repeated stochastic trials, and real-robot validation.

## II. BACKGROUND AND RELATED WORK

## A. 6-DoF Grasp Generation and Datasets

Large-scale grasp datasets and simulation pipelines have played a major role in modern grasping. Dex-Net demonstrated that synthetic grasp analysis can train robust grasping policies [3], and the YCB object set helped standardize manipulation objects and models [10]. GraspNet-1Billion introduced a large-scale RGB-D benchmark for general object grasping [1], while ACRONYM provided millions of simulated grasps over thousands of objects [2]. Together, these resources provide broad object coverage, large-scale data, and standardized candidate evaluation.

Several families of 6-DoF grasp generators are relevant to our study. GPD estimates grasp poses directly from point clouds using local geometric evidence [4], and PointNet-GPD learns grasp scoring from point sets [11]. Contact-GraspNet predicts contact-based 6-DoF grasps from cluttered scenes [5]. AnyGrasp emphasizes robust and efficient grasp perception across spatial and temporal domains [6]. EconomicGrasp studies efficient candidate generation under a 6-DoF detection framework [7], while HGGD uses heatmapguided grasp detection [9]. Generative approaches, including 6-DOF GraspNet [12] and GraspGen [8], model distributions over grasp poses and can produce diverse hypotheses. Their distinct proposals motivate robot-specific, complete-pipeline evaluation.

GraspBench is also related to broader simulation benchmarks for manipulation. Robosuite provides modular Mu-JoCo environments for robot-learning evaluation [13], RL-Bench defines a large set of language-conditioned manipulation tasks [14], Meta-World studies multi-task and metareinforcement-learning manipulation [15], and ManiSkill and SAPIEN provide scalable simulated manipulation environments and assets [16], [17]. These benchmarks emphasize policy learning, environment diversity, or general manipulation tasks. GraspBench instead diagnoses generators under scenario-aligned tabletop feasibility and execution.

TABLE I  
EVALUATION-SCOPE COMPARISON WITH REPRESENTATIVE GRASPING DATASETS, GENERATORS, AND MANIPULATION BENCHMARKS.
<table><tr><td>Work</td><td>Role</td><td></td><td></td><td>Scale Exec. ×-robot Cond. Fail.</td><td></td><td></td></tr><tr><td>Dex-Net [3]</td><td>synthetic policy/data</td><td></td><td></td><td>×</td><td>△</td><td>×</td></tr><tr><td>GraspNet [1]</td><td>RGB-D benchmark</td><td></td><td>X</td><td>×</td><td>△</td><td>×</td></tr><tr><td>ACRONYM [2]</td><td>simulated grasp data</td><td></td><td>△</td><td>×</td><td>×</td><td>×</td></tr><tr><td>Robosuite [13]</td><td>manipulation sim</td><td></td><td></td><td>△</td><td></td><td>×</td></tr><tr><td>RLBench [14]</td><td>task benchmark</td><td></td><td></td><td>△</td><td></td><td>×</td></tr><tr><td>GPD/GPG [4]</td><td>geom. generator</td><td>×</td><td>7</td><td>×</td><td>△</td><td>×</td></tr><tr><td>CGN [5]</td><td>contact generator</td><td>△</td><td>△</td><td>×</td><td>△</td><td>×</td></tr><tr><td>AnyGrasp [6]</td><td>dense perception</td><td>△</td><td></td><td>×</td><td>△</td><td>×</td></tr><tr><td></td><td>EconomicGrasp [7] budget-aware detector</td><td>△</td><td>△</td><td>×</td><td>△</td><td>×</td></tr><tr><td>GraspGen [8]</td><td>generative model</td><td>△</td><td>△</td><td>△</td><td>△</td><td>×</td></tr><tr><td>HGGD [9]</td><td>heatmap-guided detector</td><td>△</td><td></td><td>×</td><td></td><td>×</td></tr><tr><td>GraspBench</td><td>execution diagnostic protocol</td><td>△</td><td></td><td></td><td></td><td></td></tr></table>

Scale=large benchmark; Exec.=robot/physics execution; ×-robot=cross-embodiment; <sub>Cond.=conditioned analysis; Fail.=failure accounting. Symbols:</sub> √ <sub>=core, △ =par-</sub> tial/indirect, × =not primary (not necessarily absent).

Table I summarizes the intended positioning of Grasp-Bench relative to these resources and representative graspgeneration methods. While large-scale datasets emphasize object coverage and standardized grasp scoring, and methodspecific studies evaluate individual generators and, in some cases, real-robot performance, GraspBench focuses on controlled execution-level diagnosis across multiple robot embodiments, object poses, and workspace conditions, with explicit failure accounting. This execution-focused evaluation complements large-scale grasp benchmarks and real-robot validation.

## B. Execution-Aware Evaluation

Robot execution introduces constraints that grasp scores alone do not capture. Candidate poses must satisfy reachability, IK feasibility, collision avoidance, motion planning, gripper closure, and successful lifting under the recorded outcome criterion. MuJoCo provides a controlled physics environment for studying these interactions [18]. GraspBench evaluates each configured generator after robot-specific feasibility and a fixed high-level tabletop execution sequence. The objective is not to claim real-world transfer, but to reveal where proposal rankings and robot-executable outcomes diverge.

![](images/02edb9e8fda98e028687f0cbb7082437e4ef21d29b414c1113d1c34596ebc561.jpg)  
Fig. 1. GraspBench pipeline: proposal, robot-specific feasibility filtering, OMPL planning, MuJoCo execution, and outcome logging. Missing outputs and scenarios with no feasible grasp count as failures; Contact-GN denotes Contact-GraspNet (CGN).

## III. GRASPBENCH DESIGN

## A. Benchmark Objective

GraspBench is designed to answer a system-level question: given a grasp generator, how often does it produce an executable grasp for a specific robot, object, and tabletop configuration? To isolate this question, all main results are evaluated under an IK-filtered execution protocol. The executed grasp is the highest-ranked feasible candidate selected by the benchmark pipeline from the method’s scored or ordered proposals. Candidate ordering is method-native: scores determine order only within a pipeline and score magnitudes are never compared across methods. No common candidate budget or oracle Success@K analysis is claimed. A feasible candidate is one accepted by the benchmark’s robotspecific IK, reachability, and approach-constraint selection stage. The normalized records expose scenario outcomes and, where available, tested IK candidate counts; the reported metrics operate at the scenario level. A scenario is counted as successful only if the selected grasp completes the fixed MuJoCo approach-close-lift trial and the corresponding execution runner records a successful binary lift outcome. If a method fails to return an output for a scenario, the scenario is counted as a failure. Figure 1 summarizes the stages.

## B. Scenario Grid

Table II summarizes the benchmark grid. Each method is evaluated over four robots, five diagnostic objects, three distance bands, three lateral positions, and three pitch values, giving 540 intended scenarios. In MuJoCo world coordinates, Near, Mid, and Far correspond to $x ~ = ~ 0 . 4 5 , ~ 0 . 5 5$ , and 0.65 m; Left, Center, and Right correspond to $y = 0 . 1 0 ,$

0, and −0.10 m; pitch is applied about the object’s x-axis. Object origins use z = 0.55 m (0.555 m for the cylinder). The robots are AUBO i5, Franka Emika Panda, KUKA LBR iiwa 14, and AgileX PiPER, which provide different arm geometries, reachable workspaces, wrist configurations, and simulated end-effectors. The object set contains an airplane model, glass, cube, cylinder, and bottle, spanning simple convex shapes, elongated shapes, and thin or concave geometries. The distance, lateral-position, and pitch grid is designed to probe reachability, workspace-boundary effects, and orientation sensitivity. Because grippers, camera mounts, controllers, grasp frames, and collision models are robot-specific, cross-robot differences characterize complete embodiment configurations and cannot be attributed to arm reach alone.

![](images/7bb79576da7421ec787ae24aa78993259e9a063f75a7848387b5112a2c96cf0a.jpg)  
Fig. 2. Diagnostic GraspBench object set.  
Figures 2, 3, and 4 visualize the benchmark factors.

TABLE II  
GRASPBENCH SCENARIO GRID USED IN THE MAIN EVALUATION.
<table><tr><td>Category</td><td>Values Count</td></tr><tr><td>Robots</td><td>AUBO i5, Franka, KUKA iiwa, PiPER 4</td></tr><tr><td>Methods</td><td>AG, CGN, Econ., GPG, GG, HGGD 6</td></tr><tr><td>Objects</td><td>airplane, glass, cube, cylinder, bottle 5</td></tr><tr><td>Distance</td><td>Near, Mid, Far 3</td></tr><tr><td>Lateral</td><td>Left, Center, Right 3</td></tr><tr><td>Pitch</td><td>-45°,0°,45° 3</td></tr><tr><td>Scenarios/method Full grid</td><td>540</td></tr><tr><td>Method-scenarios 540 × 6 methods</td><td>3240</td></tr></table>

Method abbreviations: AG = AnyGrasp, CGN = Contact-GraspNet, Econ. = EconomicGrasp, and GG = GraspGen.

![](images/a200b9ad426b2df24983e7b350118bc0f34066c1c78f79db0170fa78dbdd3695.jpg)

Fig. 3. Evaluated robot configurations; each includes its arm, gripper, camera, controller, frame adapter, and collision model. “Piper” denotes AgileX PiPER.  
![](images/a6622638c7cd62bc8a13e7e4d91a03fb310af6467dfaed54eed8cfe25d9d2715.jpg)  
Workspace cells combine distance band and lateral position; object pitch is evaluated at -45 deg, 0 deg, and +45 deg

Fig. 4. Tabletop distance, lateral-position, and pitch grid for conditionspecific analysis.

## C. Methods

We evaluate six grasp-generation pipelines: AnyGrasp (AG), Contact-GraspNet (CGN), EconomicGrasp (Econ.), GPG, the geometric candidate generator associated with GPD, GraspGen (GG), and HGGD. These methods span dense detection, contact-based prediction, geometric candidate generation, budget-aware/fixed-budget sampling, generative grasp modeling, and heatmap-guided detection. The analysis focuses on the evaluation layer: method outputs retain their native within-method ordering, are mapped to the same scenario keys, and pass through the same failure-aware accounting. Accordingly, the results compare configured endto-end pipelines under a scenario-aligned IK-filtered execution structure rather than training procedures, hardwarenormalized runtime, or model architectures in isolation.

## D. Failure-Aware Data Treatment

We use three failure terms consistently: a missing output has no raw scenario record, a no-grasp output explicitly records an empty/no-grasp result, and an other observed failure records unsuccessful execution without a recoverable first-failure stage. Missing entries indicate scenarios for which no method output is available; because such entries cannot be verified as successful executions, they are conservatively counted as failures. This yields an equal intended denominator of 540 scenarios for every method. Duplicate observed rows are collapsed at the scenario level to avoid overweighting repeated entries. This policy prevents optimistic success estimates that could arise if missing scenarios were excluded from the denominator, particularly when missingness is non-random, and makes the comparison reproducible from the scenario grid.

## IV. EXPERIMENTAL SETUP

## A. Simulation Environment

All experiments are evaluated in a MuJoCo-based tabletop setting. Each trial places one target object at a specified distance band, lateral position, and pitch. A grasp-generation method proposes candidate grasps, the pipeline applies robotspecific feasibility filtering, and the selected feasible grasp passes through motion planning before simulated execution. The primary outcome is binary grasp success after the fixed approach-close-lift execution. Because the accompanying records encode lift success as a binary scenario-level outcome, we evaluate the recorded outcome directly and do not introduce an additional post hoc height or stability threshold.

Each robot uses one arm-mounted MuJoCo camera with 60<sup>◦</sup> vertical field of view; RGB-D frames are back-projected into partial point clouds in the camera frame. The benchmark uses synthetic depth without added sensor or calibration noise and does not test transparent-surface sensing. Camera placement and resulting self-occlusion are embodimentspecific. Method adapters retain their native preprocessing and candidate ordering.

The execution wrapper maps camera-relative 4×4 grasp poses into each robot’s base/end-effector frames. It uses damped-least-squares IK (up to 2000 iterations), a 0.02 m positional acceptance threshold, joint limits, and MuJoCo contact checks for robot collision. Motion planning uses OMPL RRTConnect [19] with a 2 s timeout and 80 interpolated samples for reach and 40 each for approach and lift. Planner choice was not varied as an experimental factor, so its effect is not analyzed separately. AUBO i5 and KUKA iiwa use AG95 end effectors, whereas Franka and PiPER use their model-specific grippers; approach/lift offsets, gains, closure commands, and motion durations are consequently robot-specific rather than controlled arm-only variables. Full per-robot values are specified in the execution implementation.

All methods use the same intended scenario grid and the same high-level feasibility, planning, and approach-close-lift stages for a fixed embodiment. Success is analyzed using the binary lift label produced by each execution runner. Grasp-Gen and AnyGrasp automatically mark lifts of at least 0.10 m as successful, with recorded manual judgment for lower lifts;

IK-FILTERED RESULTS. MISSING OUTPUTS FAIL; INTERVALS ARE WILSON GRID SUMMARIES; CANDIDATES ARE NATIVE MEAN/MEDIAN.

the remaining runners directly record the final binary lift outcome. This preserves each pipeline’s implemented success decision while applying common scenario-level accounting.

The benchmark protocol uses a fixed grid rather than randomized clutter. This choice makes cross-method and cross-robot comparisons interpretable: every method is evaluated against the same intended scenario set, allowing paired scenario-level analysis. The normalized table contains one row per intended method-scenario key; missing raw method records are explicitly marked and counted as failures. The records do not encode repeated stochastic trials or random seeds. Therefore, the present study estimates scenario-level behavior rather than repeatability under simulator noise. The current study does not claim robustness to clutter, occlusion, or real-world transfer. Matched ablations without IK filtering or lift judgment, repeated seeded executions, and cluttered scenes require new candidate-level runs and are outside the present scope.

## B. Analysis Artifacts

The records provide scenario-level execution outcomes, candidate counts, selected scores where available, tested IK candidates for some methods, and logged pipeline time. Candidate-level trace analysis and finer collision-stage decomposition are planned extensions.

## C. Metrics

The main metric is success rate:

$$
S R = \frac { N _ { \mathrm { s u c c e s s } } } { N _ { \mathrm { s c e n a r i o s } } } ,\tag{1}
$$

where $N _ { \mathrm { { s c e n a r i o s } } } ~ = ~ 5 4 0$ for every method in the main comparison. We additionally report robot-conditioned success, object-conditioned success, pitch-conditioned success, native candidate counts, returned-record availability, success conditional on a returned record, and failure accounting. Missing outputs are included in the denominator as failures. Wilson 95% intervals summarize binary outcomes over the finite scenario grid; they are not confidence intervals for repeated stochastic execution.

For pairwise method comparisons, we compute perscenario differences on the common 540-scenario grid. A positive value for method A − B indicates that A succeeded where B failed more often than the reverse. To account for dependence among poses of the same embodiment and object, 95% percentile intervals use 50,000 bootstrap resamples of the 20 robot-object clusters; all 27 pose/workspace outcomes within a sampled cluster remain together. These intervals describe sensitivity across scenario groups, not execution repeatability.

## D. Candidate Records and Runtime Scope

Candidate counts describe method-native outputs before method-specific truncation and feasibility testing; confidence calibration, suppression, and sampling budgets are not standardized. We therefore do not use them for controlled efficiency or Success@K claims. Logged times are not comparable across implementations because hardware, software, and stage boundaries differ.

## V. RESULTS AND DISCUSSION

## A. Overall IK-Filtered Execution Performance

Table III summarizes the main results. The configured GraspGen pipeline achieves the highest success rate, with 395 successes out of 540 scenarios (73.1%). Economic-Grasp ranks second, with 343 successes (63.5%). AnyGrasp, HGGD, and GPG range from 44.3–48.0%, versus 33.0% for Contact-GraspNet.

TABLE III
<table><tr><td>Method</td><td>Success n/540 (%)</td><td>95% int. (%)</td><td>Missing</td><td>Cand. mean/med.</td><td>IK max</td></tr><tr><td>GraspGen</td><td>395 (73.1)</td><td>[69.3, 76.7]</td><td>0</td><td>296/304</td><td>50</td></tr><tr><td>EconomicGrasp</td><td>343 (63.5)</td><td>[59.4, 67.5]</td><td>0</td><td>1024/1024</td><td>50</td></tr><tr><td>AnyGrasp</td><td>259 (48.0)</td><td>[43.8, 52.2]</td><td>1</td><td>391/365</td><td>80</td></tr><tr><td>HGGD</td><td>249 (46.1)</td><td>[41.9, 50.3]</td><td>5</td><td>6/6</td><td>20</td></tr><tr><td>GPG</td><td>239 (44.3)</td><td>[40.1, 48.5]</td><td>0</td><td>7414/8088</td><td>一</td></tr><tr><td>Contact-GraspNet</td><td>178 (33.0)</td><td>[29.1, 37.0]</td><td>31</td><td>23/23</td><td>一</td></tr></table>

IK max is the maximum tested-candidate count observed in available logs; “–” denotes an unrecorded field, not zero tests.

Figure 5 shows the intended-grid results and clusteraware paired analysis. Resampling robot-object groups gives GraspGen paired advantages of 9.6 percentage points over EconomicGrasp (95% interval 0.9–19.6), 25.2 over Any-Grasp (15.0–35.2), 28.9 over GPG (17.2–39.6), and 40.2 over Contact-GraspNet (30.9–49.3). HGGD–AnyGrasp (−1.9, −16.1–11.9) and GPG–AnyGrasp (−3.7, −14.3–6.3) cross zero. These intervals summarize variation across 20 robotobject clusters and do not measure repeated-execution uncertainty.

## B. Cross-Robot Sensitivity

Figure 6(a) shows substantial variation across robot configurations. GraspGen reaches 85.2% on AUBO i5 and 83.0% on PiPER, but drops to 51.9% on KUKA iiwa. Economic-Grasp is more uniform, ranging from 57.0% on PiPER to 68.9% on AUBO i5 and Franka. GPG performs similarly on AUBO i5 and KUKA iiwa (53.3% and 54.1%) but is lower on Franka and PiPER. HGGD records nonzero success on AUBO i5, Franka, and KUKA iiwa, but zero successes on PiPER. All 135 HGGD/PiPER scenarios are explicitly logged as no-grasp outputs; the records do not localize the cause to generation, frame adaptation, or feasibility filtering. We therefore describe a configuration-level failure pattern, not intrinsic HGGD quality. Robot scale may be one contributing factor, but the available records do not isolate it. As a sensitivity check, excluding PiPER gives HGGD 249/405 successes (61.5%) across the remaining robots; over the same non-PiPER subset, GraspGen reaches 69.9% and Economic-Grasp reaches 65.7%, and these two pipelines remain ahead of HGGD on this subset.

![](images/fef4e3e7216ded53fc350116f9e775ae713fea5d93c1dadb5f14b4b4a3fd9553.jpg)

(b) Paired difference (pp), clustered bootstrap  
![](images/436f337308fd4420cee2fe96b03532967859d0d646ff39300416eb29a69e234a.jpg)

(c) Candidates vs. success  
![](images/b2cd202a7cf4e156ce4b3392916a06606e1117e5201d2b89cdba49453651eaa1.jpg)  
Fig. 5. Execution results: (a) success with Wilson intervals; (b) clustered paired differences; (c) native candidates versus success (descriptive because budgets differ). Abbreviations follow Table II.

This variation characterizes complete embodiment configurations: arm kinematics, end effector, camera and base mounting, grasp-frame adapter, controller, and collision model vary jointly. Isolating the contribution of arm reach requires a controlled component-level ablation.

## C. Object and Pose Sensitivity

Figure 6(b) shows object-conditioned success. Cube is the easiest object overall, while the glass is the hardest. Grasp-Gen performs strongly on cube, cylinder, bottle, and airplane model, but drops to 38.0% on the glass. EconomicGrasp is similarly strong on cube (84.3%), cylinder (80.6%), and bottle (77.8%), but reaches only 23.1% on the glass. Thus, even the strongest methods remain sensitive to thin or difficult geometries. Qualitatively, some successful glass trials used top- or rim-directed grasps rather than potentially more tasksuitable stem or side contacts. Binary lift success therefore need not indicate an optimal or semantically appropriate grasp.

Pitch-conditioned results in Fig. 6(c) reveal additional structure. GraspGen peaks at zero pitch (81.7%) and decreases at ±45<sup>◦</sup>. EconomicGrasp remains high at −45<sup>◦</sup> and 0<sup>◦</sup> but drops at 45<sup>◦</sup>. AnyGrasp also decreases at $4 5 ^ { \circ }$ , while GPG improves at $4 5 ^ { \circ }$ . Workspace effects are weaker than robot and object effects in this grid: averaged over methods, success is 49.2% for Near, 52.8% for Mid, and 52.0% for Far placements, with lateral averages of 49.4% for Center, 53.7% for Left, and 50.9% for Right. Thus, workspace factors are most useful here as paired scenario controls, while object pose and robot embodiment produce clearer diagnostic variation.

![](images/7e2fd4ade6ed688a5a048ce469c403cbe7c8c1b8e99092a00b9b9275797fd34a.jpg)  
(c) Pitch (%)

(b) Object (%)  
![](images/a317bbea80b4f56b1bcc73a0865f640e7e243b2a0f95cbee6e0b06d337a791fa.jpg)

![](images/834a716a8ac2bf9c19214ae6dce894513195699f2665202b953663bacec749a0.jpg)

(d) Failures  
![](images/65bcae3e2bcb7d38e5f0aa861955f589239343a91a48fe78b0ce332be17c6bb6.jpg)  
Fig. 6. Diagnostics by (a) robot, (b) object, (c) pitch, and (d) failure type: other observed (amber), no grasp (red), and missing output (purple). Abbreviations follow Table II.

## D. Failure Accounting

Figure 6(d) reports failure accounting. Contact-GraspNet has 31 missing outputs; these are counted as failures and reduce its final success rate to 33.0%. HGGD has five missing outputs and 148 no-grasp outputs, including all PiPER scenarios. AnyGrasp has one missing output. This conservative accounting prevents optimistic estimates that could result from excluding missing scenarios. The reported labels distinguish missing, no-grasp, and other observed failures; recording the first execution stage would enable finer attribution to IK, collision, planning, closure, drop, or lift.

TABLE IV  
RETURNED-RECORD AVAILABILITY AND SUCCESS (%). END-TO-END SR USES ALL 540 INTENDED SCENARIOS.
<table><tr><td>Method</td><td>Returned records</td><td>E2E SR SRlrecord (%)</td><td>(%)</td></tr><tr><td>GraspGen</td><td>540 (100.0)</td><td>73.1</td><td>73.1</td></tr><tr><td>EconomicGrasp</td><td>540 (100.0)</td><td>63.5</td><td>63.5</td></tr><tr><td>AnyGrasp</td><td>539 (99.8)</td><td>48.0</td><td>48.1</td></tr><tr><td>HGGD</td><td>535 (99.1)</td><td>46.1</td><td>46.5</td></tr><tr><td>GPG</td><td>540 (100.0)</td><td>44.3</td><td>44.3</td></tr><tr><td>Contact-GraspNet 509 (94.3)</td><td></td><td>33.0</td><td>35.0</td></tr></table>

Stage-conditional success after IK or planning is unavailable because those denominators were not recorded consistently.

Table IV shows that conditioning on a returned record changes success by only 0.1 percentage points for AnyGrasp, 0.4 for HGGD, and 2.0 for Contact-GraspNet. HGGD’s dominant recorded issue is therefore the explicit no-grasp output rather than missing records. The unavailable stage denominators prevent stronger attribution to IK, planning, or execution control.

## E. Implications

The results support several practical conclusions. First, execution-aware benchmarking can change method interpretation: the highest-scoring grasp proposal is not necessarily reachable, IK-feasible, or plannable for a particular robot. This motivates explicit robot-specific filtering, although its marginal effect is not isolated because the records do not support a matched no-IK ablation. Its cost is also not quantified because timing was not measured on common hardware. Across the configured pipelines, no monotonic relation is apparent between native candidate count and success; differing budgets and filtering preclude causal interpretation. Partial point-cloud observations can make object extent and grasp pose estimation ambiguous, suggesting that geometric context, camera-view context, or multi-view perception may be needed for difficult objects. Finally, conservative handling of missing outputs and duplicate records is necessary for defensible benchmark reporting. These findings motivate condition-specific, execution-aware, failure-aware reporting.

## VI. CONCLUSION AND LIMITATIONS

GraspBench provides an execution-aware MuJoCo protocol for six grasp pipelines, four robots, five diagnostic objects, and a structured tabletop grid. Common scenarios, missing-output failures, and duplicate collapse support conservative paired comparisons. The configured GraspGen pipeline leads at 73.1%, followed by EconomicGrasp at 63.5%, while robot- and object-conditioned results vary substantially. GPG generates the largest native candidate sets but reaches 44.3% success, showing that proposal volume alone does not ensure robot-executable grasps; candidate counts remain descriptive because budgets differ. Thus, grasp performance is a systems property of the generator, embodiment, scene, and execution stack. Conclusions apply to complete configured pipelines and one execution per scenario in single-object simulation; isolating individual components requires controlled ablations. Future work includes repeats, stage traces, matched budgets, clutter, and real transfer.

## REFERENCES

[1] H.-S. Fang, C. Wang, M. Gou, and C. Lu, “GraspNet-1Billion: A largescale benchmark for general object grasping,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 11 444– 11 453.

[2] C. Eppner, A. Mousavian, and D. Fox, “ACRONYM: A large-scale grasp dataset based on simulation,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2021, pp. 6222–6227.

[3] J. Mahler, M. Matl, V. Satish, M. Danielczuk, B. DeRose, S. McKinley, and K. Goldberg, “Learning ambidextrous robot grasping policies,” Sci. Robot., vol. 4, no. 26, 2019, art. no. eaau4984.

[4] A. ten Pas, M. Gualtieri, K. Saenko, and R. Platt, “Grasp pose detection in point clouds,” Int. J. Robot. Res., vol. 36, no. 13–14, pp. 1455–1473, 2017.

[5] M. Sundermeyer, A. Mousavian, R. Triebel, and D. Fox, “Contact-GraspNet: Efficient 6-DoF grasp generation in cluttered scenes,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2021, pp. 13 438–13 444.

[6] H.-S. Fang, C. Wang, H. Fang, M. Gou, J. Liu, H. Yan, W. Liu, Y. Xie, and C. Lu, “AnyGrasp: Robust and efficient grasp perception in spatial and temporal domains,” IEEE Trans. Robot., vol. 39, no. 5, pp. 3929–3945, 2023.

[7] X.-M. Wu, J.-F. Cai, J.-J. Jiang, D. Zheng, Y.-L. Wei, and W.-S. Zheng, “An economic framework for 6-DoF grasp detection,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2024, pp. 357–375.

[8] A. Murali, B. Sundaralingam, Y.-W. Chao, W. Yuan, J. Yamada, M. Carlson, F. Ramos, S. Birchfield, D. Fox, and C. Eppner, “Grasp-Gen: A diffusion-based framework for 6-DOF grasping with ongenerator training,” 2025, arXiv preprint arXiv:2507.13097.

[9] S. Chen, W. Tang, P. Xie, W. Yang, and G. Wang, “Efficient heatmapguided 6-Dof grasp detection in cluttered scenes,” IEEE Robot. Autom. Lett., vol. 8, no. 8, pp. 4895–4902, 2023.

[10] B. Calli, A. Singh, A. Walsman, S. Srinivasa, P. Abbeel, and A. M. Dollar, “The YCB object and model set: Towards common benchmarks for manipulation research,” in Proc. Int. Conf. Adv. Robot. (ICAR), 2015, pp. 510–517.

[11] H. Liang, X. Ma, S. Li, M. Görner, S. Tang, B. Fang, F. Sun, and J. Zhang, “PointNetGPD: Detecting grasp configurations from point sets,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2019, pp. 3629– 3635.

[12] A. Mousavian, C. Eppner, and D. Fox, “6-DOF GraspNet: Variational grasp generation for object manipulation,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2019, pp. 2901–2910.

[13] Y. Zhu, J. Wong, A. Mandlekar, R. Martín-Martín, A. Joshi, K. Lin, A. Maddukuri, S. Nasiriany, and Y. Zhu, “robosuite: A modular simulation framework and benchmark for robot learning,” 2020, arXiv preprint arXiv:2009.12293.

[14] S. James, Z. Ma, D. R. Arrojo, and A. J. Davison, “RLBench: The robot learning benchmark & learning environment,” IEEE Robot. Autom. Lett., vol. 5, no. 2, pp. 3019–3026, 2020.

[15] T. Yu, D. Quillen, Z. He, R. Julian, K. Hausman, C. Finn, and S. Levine, “Meta-World: A benchmark and evaluation for multi-task and meta reinforcement learning,” in Proc. Conf. Robot Learn. (CoRL), vol. 100. PMLR, 2020, pp. 1094–1100.

[16] T. Mu, Z. Ling, F. Xiang, D. Yang, X. Li, S. Tao, Z. Huang, Z. Jia, and H. Su, “ManiSkill: Generalizable manipulation skill benchmark with large-scale demonstrations,” in Proc. NeurIPS Datasets Benchmarks Track, 2021.

[17] F. Xiang, Y. Qin, K. Mo, Y. Xia, H. Zhu, F. Liu, M. Liu, H. Jiang, Y. Yuan, H. Wang, L. Yi, A. X. Chang, L. J. Guibas, and H. Su, “SAPIEN: A SimulAted part-based interactive ENvironment,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 11 097–11 107.

[18] E. Todorov, T. Erez, and Y. Tassa, “MuJoCo: A physics engine for model-based control,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2012, pp. 5026–5033.

[19] I. A. Sucan, M. Moll, and L. E. Kavraki, “The open motion planning library,” IEEE Robot. Autom. Mag., vol. 19, no. 4, pp. 72–82, 2012.