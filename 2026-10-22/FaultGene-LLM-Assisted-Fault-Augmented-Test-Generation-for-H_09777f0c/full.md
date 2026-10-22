# FaultGene: LLM-Assisted Fault-Augmented Test Generation for Hardware-in-the-Loop Validation of Autonomous Driving Systems

1<sup>st</sup> Chaoran Yuan

Center for Embedded and Cyber-Physical Systems

University of California, Irvine

Irvine, USA

chaory2@uci.edu

2<sup>nd</sup> Fadi kurdahi

Center for Embedded and Cyber-Physical Systems

University of California, Irvine

Irvine, USA

kurdahi@uci.edu

Abstract—Reliable validation of autonomous driving cyber-physical systems requires testing not only nominal driving behavior but also failures arising from faults in sensing and embedded computation. Physical road testing provides realistic system behavior but is costly and difficult to reproduce, while simulation enables scalable evaluation but cannot fully capture hardware-dependent behavior and fault propagation. Hardware-in-the-loop (HIL) validation bridges this gap by executing the driving stack on real embedded hardware within a simulated environment. However, preparing fault-augmented scenarios for HIL testing remains largely a manual process, limiting test coverage, scalability, and efficiency.

We present FaultGene, an LLM-assisted framework for automated generation of executable fault-augmented tests for HIL fault-resilience validation. Given a baseline Scenic/CARLA scenario and a structured fault specification, FaultGene generates a fault-augmented test case that captures the effects of HIL-level fault injection. During offline refinement, semantic-consistency checks and execution-level feedback are used to correct candidates that fail to preserve the intended scenario context or produce the desired fault-induced degradation, and to construct high-quality supervision for fine-tuning. The resulting generator then produces fault-augmented tests directly at deployment time without requiring manually engineered reference executions or repeated refinement loops.

On held-out scenario–fault combinations, FaultGene improves semantic alignment and executability over oneshot LLM generation across both modular and end-toend autonomous driving architectures. It also preserves the main sensor- and compute-fault sensitivity trends observed in HIL evaluation, capturing hardware-level fault effects beyond simulation-based verification and falsification alone. Compared with conventional manual fault-scenario engineering, FaultGene reduces estimated fault-scenario preparation time by 72×, enabling scalable HIL-based dependability validation.

## I. INTRODUCTION

The rapid advancement of Automated Driving Systems (ADS) has created an urgent need for testing environments that can reliably expose safety-critical failures [1]. Road testing provides high-fidelity evaluation but is costly, time-consuming, difficult to reproduce, and limited in its ability to systematically exercise rare fault conditions. Simulation-based testing offers scalable and reproducible alternatives through

controllable scenario generation using platforms such as CARLA [2]. Existing probabilistic, constraintbased, and search-based approaches [3], [4] further expand the scenario space for systematic verification. However, simulation alone cannot fully capture hardware-dependent effects, such as timing variations, scheduling jitter, and embedded computation faults, motivating hardware-in-the-loop (HIL) testing, where the driving stack executes on real hardware within a simulated environment.

Despite these advantages, preparing faultaugmented scenarios for HIL testing remains a practical bottleneck. A fault-oriented test must define both the driving context and the intended sensor or compute fault while preserving scenario semantics and producing meaningful system-level effects. Manually constructing scenarios across fault types, activation times, and severities is labor-intensive and difficult to scale.

Large language models (LLMs) offer a potential solution by translating high-level specifications into executable scenario programs [5]. However, one-shot generation may produce scenarios that violate the intended context, fail to compile or execute, or fail to expose the specified fault meaningfully. These limitations motivate combining LLM-based generation with execution-guided refinement grounded in HIL behavior.

This paper presents FaultGene, an LLM-assisted framework for automated generation of faultaugmented HIL test cases. Given a baseline Scenic [3]/CARLA scenario and a structured specification of fault type, activation timing, and severity, FaultGene generates a fault-augmented test while preserving the underlying driving context. The framework integrates structured LLM generation with execution-guided refinement in the Autonomous Driving Digital Twin (ADDT) environment [6]. HIL executions provide fault-grounded reference behavior and system-level feedback that are used offline to refine generated scenarios and construct supervision for fine-tuning the generator. After tuning, the generator produces fault-augmented tests in a single pass without reference executions or iterative refinement during deployment.

We evaluate FaultGene on held-out scenario–fault combinations across modular and end-to-end AD architectures. Experimental results show improved scenario validity and executability over one-shot LLM generation, while the generated tests reproduce the main sensor- and compute-fault sensitivity trends observed in ADDT. Compared with conventional manual fault-scenario engineering, FaultGene reduces faultengineering effort by 72× per fault type.

In summary, this work makes the following contributions:

• An LLM-assisted framework for automatically transforming baseline driving scenarios and structured fault specifications into executable faultaugmented test cases;

• A feedback-driven refinement and tuning mechanism that uses execution feedback as supervision to improve semantic consistency and faultinduced system behavior;

• An empirical evaluation showing improved scenario validity, ADDT-consistent fault trends, and higher testing efficiency compared with conventional HIL workflows.

## II. RELATED WORKS

Digital-twin-based hardware-in-the-loop (HIL) validation enables controlled evaluation of autonomous driving systems under realistic sensing, compute, and communication disturbances. Digital twins model the vehicle, environment, and traffic context [7], [8], while HIL co-simulation executes perception, planning, and control on real embedded hardware, exposing timing effects, scheduling jitter, and fault propagation that simulation-only studies may miss [9]–[11]. Fault injection has been used to assess neural-network-based autonomous driving components: APPRAISER [12] and DNN accelerator fault-injection frameworks [13] examine model-level vulnerability, while AVFI and Kayotee [14], [15] inject faults across perception, planning, and control in simulated environments. However, these approaches generally do not jointly address faultoriented scenario generation and realistic embedded execution.

Autonomous-driving scenario generation has explored probabilistic and constraint-based specification, systematic search, and optimization. Scenic [3] enables probabilistic and constraint-based scenario specification, while VerifAI [4] performs simulationbased verification and falsification over parameterized scenario spaces; their combination has been used to identify challenging scenarios [16], alongside optimization- and adversarial-search-based approaches [17]. Works have also explored learning scenario representations from driving videos, such as [18].

More recently, LLMs have been applied to drivingscenario description and generation. DriveGPT4 [19] and Talk2Drive [20] support language-based driving representation and interaction, vision-language approaches derive structured scenarios from driving videos [21], and DriveArena [22] and LMDrive [23] explore LLM-based scenario expansion and reasoning over traffic interactions and environmental conditions. Nevertheless, these methods typically rely on predefined scenario spaces or text-/simulation-level generation and do not directly construct fault-augmented scenarios from structured hardware-fault specifications or validate their fault-induced effects on real embedded hardware. FaultGene addresses this gap by treating fault-augmented test generation as a program-level generation and refinement problem grounded in HIL execution.

## III. HIL-BASED AUTONOMOUS DRIVING VALIDATION

In this section, we introduce the Autonomous Driving Digital Twin (ADDT) [6], the HIL validation environment used throughout this work.

## A. Hardware-in-the-Loop Setup

As shown in Fig. 1, ADDT consists of a simulation system and an AD computing system connected through Ethernet. The simulator streams sensor data to the AD system and receives control commands, enabling closed-loop execution in the virtual environment.

1) Digital-Twin Simulation System: Software. The simulation system is built on CARLA [2], an opensource AD simulator based on Unreal Engine [24]. OpenSCENARIO [25] and Scenic [3] specify and parameterize the traffic scenarios.

Hardware. The simulator runs on a workstation with an Intel i7-9800X CPU and an NVIDIA RTX 2080 Ti GPU.

2) AD Computing System: The AD stack (AD architectures are described in Section V) runs on an NVIDIA Jetson Xavier or Orin, processing six 480p or three 720p camera streams at 10 Hz and producing control commands at 100 Hz.

3) Testing Scenarios: We evaluate three representative driving scenarios: vehicle following, largecurvature turning, and overtaking. Fig. 2 shows their configurations and parameters, including vehicle speeds and inter-vehicle distances.

## B. Fault Injection Mechanism

We evaluate resilience by injecting faults into the sensing and computing paths of the HIL pipeline.

Sensor fault injection. ADDT models temporal and spatial sensing faults:

1) Temporal faults: Controlled timestamp delays are introduced into sensor streams. Delays exceeding the perception model’s out-of-order tolerance are treated as frame drops.

![](images/797e9357034f1d2603348df8478068ccbd43ac59c2aa5438ea8c7527a890a42c.jpg)  
Fig. 1: Overview of the ADDT hardware-in-the-loop setup.

![](images/39287df09eb583dbac513504bb9430abc5d0535f8255555435aa9e8e58a7fb33.jpg)  
Fig. 2: Driving scenarios used for evaluation.

2) Spatial faults: Position and orientation perturbations are applied to the ground-truth extrinsic parameters to emulate sensor misalignment with controllable severity.

Computing hardware fault injection. ADDT supports closed-loop CPU and GPU fault injection for studying memory and data-path errors [26], [27]. CPU faults are injected with LLFI [28], while GPU faults are injected with NVBitFI [29]. Both tools introduce register-level bit flips at specified locations and values, allowing the resulting errors to propagate through subsequent AD computations. This approach follows prior fault-injection studies [26], [27].

## IV. FAULTGENE: AUTOMATED FAULT-AUGMENTED TEST GENERATION

FaultGene is an LLM-based framework for generating executable fault-augmented tests for autonomousdriving fault-resilience validation. An overview of the framework is shown in Fig. 3. Given a baseline Scenic/CARLA scenario and a structured fault specification, it produces constrained scenario variants that can be executed in CARLA or through a CARLAbased HIL setup, such as the ADDT HIL environment described in Section III.

FaultGene has two phases. During offline framework construction, a prompt-based generator produces candidate fault-augmented Scenic programs, while ScenarioFeat extracts semantic attributes from execution videos. Generated executions are compared with manually engineered ADDT HIL references, and detected mismatches guide prompt refinement and supervision construction. After fine-tuning, the deployed generator produces tests in one pass without accessing manually engineered references, execution feedback, or refinement loops.

A. ScenarioFeat: Video-Based Scenario Feature Extraction

ScenarioFeat is a prompt-engineered module that extracts semantic features from manually engineered reference and FaultGene-generated executions. It uses Qwen2.5-VL-7B-Instruct as its base vision-language model.

For an execution video V, ScenarioFeat extracts a predefined semantic feature vector describing the driving context and safety-relevant behavior, including road environment, surrounding-object interactions, obstacle relationships, and vehicle maneuvers such as cut-in behavior:

$$
C ( V ) = [ c _ { 1 } ( V ) , c _ { 2 } ( V ) , \ldots , c _ { n } ( V ) ] ,\tag{1}
$$

where $c _ { i } ( V ) \in [ 0 , 1 ]$ denotes the predicted confidence score for feature category i.

ScenarioFeat is prompt-engineered using representative execution videos and expected feature annotations. It does not generate scenarios; instead, it provides a structured semantic representation for comparing generated executions with manually engineered references.

![](images/2dc64a48efc85ebfd5866ef5608b2537672390243ac0f5952c8563fe2bdc562b.jpg)  
Fig. 3: Overview of FaultGene. (a) Pipeline overview showing offline refinement and deployment-time generation steps. (b) Detailed offline refinement process, where FaultGene generates fault-augmented Scenic scenarios, compares their executions with manually engineered HIL references using ScenarioFeat, and uses the resulting feedback for refinement and fine-tuning.

## B. FaultGene: Prompt-Based Scenic Generation

FaultGene is a text-only LLM, instantiated with Qwen3-14B-Instruct, that transforms a baseline Scenic scenario and structured fault specification into a faultaugmented Scenic program. Formally,

$$
S _ { F } = \pi _ { \theta } ( S _ { 0 } , F ) ,\tag{2}
$$

where $S _ { 0 }$ is the baseline scenario, $F$ is the fault specification, and $\pi _ { \theta }$ is the generator.

The input includes $S _ { 0 } , F _ { : }$ , and constraints requiring valid Scenic syntax and preservation of the original driving context. The fault specification defines the fault type, activation condition, and severity. FaultGene modifies the scenario while preserving its road layout, actor identities, and intended maneuver semantics. Because the output is structured Scenic code rather than free-form text, it can be syntax-checked and executed directly in CARLA or CARLA-based HIL setups such as ADDT.

## C. Reference-Guided Similarity and Execution Checks

Each generated scenario $S _ { F }$ is executed through the HIL loop with the under-test AD system and compared with the corresponding manually faultengineered ADDT HIL execution. The reference provides offline HIL grounding for supervision construction and is not required at deployment.

Let $V _ { \mathrm { r e f } }$ and $V _ { F }$ denote the reference and generated execution videos. ScenarioFeat extracts their semantic feature vectors:

$$
C ( V _ { \mathrm { r e f } } ) = [ \ldots , c _ { n } ( V _ { \mathrm { r e f } } ) ] , \qquad C ( V _ { F } ) = [ \ldots , c _ { n } ( V _ { F } ) ] .\tag{3}
$$

The feature mismatch vector is defined as

$$
\begin{array} { r } { \mathbf { d } _ { C } ( V _ { \mathrm { r e f } } , V _ { F } ) = [ c _ { 1 } ( V _ { \mathrm { r e f } } ) - c _ { 1 } ( V _ { F } ) , \ldots , } \\ { c _ { n } ( V _ { \mathrm { r e f } } ) - c _ { n } ( V _ { F } ) ] . } \end{array}\tag{4}
$$

A generated execution is semantically inconsistent if

$$
| { \bf d } _ { C } ( V _ { \mathrm { r e f } } , V _ { F } ) _ { i } | > \tau _ { i }\tag{5}
$$

for any feature i. The threshold $\tau _ { i }$ controls the tradeoff between semantic preservation and scenario diversity and is used only during offline refinement.

We separately assess fault relevance using nearcollision events, inter-vehicle clearance, maneuver interruption, collision outcome, and task completion. Let $M ( S _ { F } )$ denote the resulting metric vector. A scenario is fault-relevant if

$$
M ( S _ { F } ) \in \Omega ,\tag{6}
$$

where Ω denotes safety-relevant degradation without trivial invalidity, such as deadlock, early termination, or failure to trigger the fault.

Together, these checks determine whether the scenario preserves the intended context, implements the requested fault, and allows it to manifest during execution.

## D. Offline Iterative Refinement and Supervised Fine-Tuning

Candidates that fail either semantic or faultrelevance checks, or fail to compile or execute successfully, are refined offline. The mismatch is converted into structured feedback covering Scenic syntax and compilation, physical or semantic validity, faultspecification consistency, and execution-level fault manifestation. Depending on the failure, the feedback may request corrected actor placement, adjusted activation timing, stronger fault severity, or removal of unintended deadlocks and early termination.

Let $\Delta ( S _ { F } )$ denote the aggregated semantic and execution mismatch relative to the manually engineered reference. Starting from $S _ { F } ^ { ( 0 ) }$ , the generator produces the next candidate as

$$
S _ { F } ^ { ( k + 1 ) } = \mathrm { R e f i n e } \left( S _ { 0 } , F , S _ { F } ^ { ( k ) } , \Delta ( S _ { F } ^ { ( k ) } ) \right) ,\tag{7}
$$

until

$$
\vert \mathbf { d } _ { C } ( V _ { \mathrm { r e f } } , V _ { F } ^ { ( k ) } ) _ { i } \vert \leq \tau _ { i } , \quad \forall i ,\tag{8}
$$

and

$$
M ( S _ { F } ^ { ( k ) } ) \in \Omega .\tag{9}
$$

Here, Refine(·) denotes rule-guided LLM regeneration using the original baseline scenario, fault specification, current candidate, and structured feedback prompts. It is not a reinforcement-learning policy or an external optimizer that directly edits the Scenic program.

The resulting scenario $S _ { F } ^ { \star }$ is retained as the corrected target for $( S _ { 0 } , F )$ . Repeating this process over $N$ framework-construction examples yields

$$
\mathcal { D } _ { \mathrm { r e f } } = \left\{ \left( S _ { 0 } ^ { ( j ) } , F ^ { ( j ) } , S _ { F } ^ { \star ( j ) } \right) \right\} _ { j = 1 } ^ { N } .\tag{10}
$$

We then fine-tune the original generator π<sub>θ</sub> on $\mathcal { D } _ { \mathrm { r e f } }$ using supervised fine-tuning (SFT). Each training example uses the baseline scenario and fault specification $( S _ { 0 } , F )$ as input and the corresponding refined Scenic program $S _ { F } ^ { \star }$ as the target. The trainable LoRA parameters are optimized with teacher-forced causal language modeling:

$$
\mathcal { L } _ { \mathrm { S F T } } = - \sum _ { t = 1 } ^ { | S _ { F } ^ { \star } | } \log P _ { \theta } ( s _ { t } ^ { \star } \mid S _ { 0 } , F , s _ { < t } ^ { \star } ) ,\tag{11}
$$

where $s _ { t } ^ { \star }$ is the t-th token of $S _ { F } ^ { \star }$ and $s _ { < t } ^ { \star }$ denotes its preceding tokens. The baseline scenario and fault specification provide the conditioning context, while the loss is computed over the target Scenic program.

Although the loss is evaluated token by token, the model is conditioned on the complete $( S _ { 0 } , F )$ pair and the preceding program context. Across diverse training examples, the shared parameters learn reusable transformations, including which scenario elements to preserve, how to encode the specified fault, and how to adjust actor timing and interactions to produce valid and fault-relevant Scenic programs. The objective therefore learns the conditional mapping

$$
( S _ { 0 } , F ) \longmapsto S _ { F } ^ { \star } .\tag{12}
$$

The refined target $S _ { F } ^ { \star }$ is used rather than the manually engineered HIL reference because it converts execution-level behavioral supervision into an inputaligned program target. Since $S _ { F } ^ { \star }$ is generated from the $( S _ { 0 } , F )$ pair and refined against the reference, SFT can directly learn the desired program transformation, following the broader paradigm of using validated or refined generated outputs as supervision [30], [31].

This SFT process absorbs the offline corrections into the trainable LoRA parameters, producing the finalized generator $\pi _ { \theta ^ { \star } }$ . Its generalization is evaluated on heldout scenario–fault combinations excluded from refinement and fine-tuning.

E. Deployment-Time Fault-Augmented Test Generation

After fine-tuning, the deployed model generates

$$
S _ { F } ^ { \mathrm { d e p l o y } } = \pi _ { \theta ^ { \star } } ( S _ { 0 } , F ) ,\tag{13}
$$

where $\theta ^ { \star }$ denotes the fine-tuned model parameters and $( S _ { 0 } , F )$ is a new baseline scenario–fault specification pair. The generated Scenic program is syntaxchecked and compiled for execution with CARLA or a CARLA-based HIL setup.

Because the offline corrections are absorbed through fine-tuning, deployment does not require a manually engineered reference, ScenarioFeat comparison, execution feedback, or repeated refinement. FaultGene therefore generates fault-augmented tests in one pass.

## V. EXPERIMENTAL SETUP

Simulation Environment. We use CARLA [2], an open-source platform for the development and validation of autonomous driving systems. CARLA provides a realistic physics engine and high-fidelity environmental rendering, enabling the generation of reproducible driving scenarios for our evaluation.

Scenario Description. We use Scenic [3] to describe driving scenarios. Its Python-like syntax provides a structured representation that is well suited for LLM-based scenario generation and prompt engineering. Scenic also integrates directly with CARLA and supports the specification of complex vehicle configurations, traffic interactions, and environmental conditions, making it suitable for program-level scenario generation.

Autonomous Driving Systems Under Test. We evaluate FaultGene on two complementary autonomous driving architectures: a modular stack that explicitly separates perception, planning, and control, and an end-to-end system that directly maps sensor observations to driving commands. This distinction allows us to evaluate FaultGene across different system organizations and fault-propagation paths.

Modular AD stack: The modular system consists of a FastBEV [32] perception module, an MLP-based planner trained on nuPlan [33], and a PID controller. The modules communicate through ROS [34] and execute on the embedded HIL platform, enabling evaluation of fault propagation across modules.

End-to-end PilotNet: The end-to-end system uses PilotNet [35], a CNN-based model designed for lanekeeping that directly maps front-camera images to steering commands. We use PilotNet as a simple representative of end-to-end driving, providing a complementary architecture for evaluating FaultGene through an end-to-end driving pipeline without explicit perception, planning, and control modules.

Dataset split and evaluation protocol. We construct the refinement dataset from 600 baseline scenario–fault combinations and reserve 200 disjoint combinations for held-out evaluation. No held-out is used during refinement or fine-tuning.

TABLE I: Semantic alignment, compilation success, and execution success for modular and end-to-end systems.
<table><tr><td>System</td><td>Method</td><td>Semantic Alignment Score*</td><td>Compilation Success(%)</td><td>Execution Success Among Compiled(%)</td></tr><tr><td>Modular</td><td>One-shot LLM</td><td>0.45</td><td>68</td><td>81</td></tr><tr><td>Modular</td><td>Fine-tuned FaultGene Generator</td><td>0.89</td><td>96</td><td>93</td></tr><tr><td>End-to-end</td><td>One-shot LLM</td><td>0.43</td><td>71</td><td>74</td></tr><tr><td>End-to-end</td><td>Fine-tuned FaultGene Generator</td><td>0.86</td><td>94</td><td>90</td></tr></table>

\*The semantic alignment score is computed as the average feature agreement between the generated and reference executions.

TABLE II: Comparison of sensor-fault sensitivity trends between ADDT and FaultGene under different camera frame drop rates. Values report mission success rate (%).
<table><tr><td>System</td><td>Source</td><td>Scenario</td><td>1%</td><td>2%</td><td>5%</td><td>10%</td></tr><tr><td rowspan="5">Modular</td><td rowspan="3">ADDT</td><td>Large-curv. turning</td><td>100.00%</td><td>99.58%</td><td>96.25%</td><td>88.75%</td></tr><tr><td>Vehicle following</td><td>99.58%</td><td>99.58%</td><td>97.92%</td><td>94.16%</td></tr><tr><td>Overtaking</td><td>97.08%</td><td>87.50%</td><td>73.75%</td><td>54.58%</td></tr><tr><td rowspan="3">FaultGene</td><td>Large-curv. turning</td><td>99.58%</td><td>98.60%</td><td>94.83%</td><td>86.18%</td></tr><tr><td>Vehicle following</td><td>97.92%</td><td>96.05%</td><td>95.91%</td><td>92.37%</td></tr><tr><td>Overtaking</td><td>94.83%</td><td>85.25%</td><td>71.69%</td><td>62.33%</td></tr><tr><td rowspan="4">End-to-end</td><td rowspan="2">ADDT</td><td>Lane following</td><td>100.00%</td><td>100.00%</td><td>94.16%</td><td>90.33%</td></tr><tr><td>Large-curv. turning</td><td>98.39%</td><td>98.39%</td><td>92.58%</td><td>84.60%</td></tr><tr><td rowspan="2">FaultGene</td><td>Lane following</td><td>100.00%</td><td>99.37%</td><td>95.27%</td><td>88.45%</td></tr><tr><td>Large-curv. turning</td><td>99.17%</td><td>98.12%</td><td>91.86%</td><td>82.56%</td></tr></table>

Fine-Tuning Setup. FaultGene is fine-tuned using QLoRA-based supervised fine-tuning [36]. The training set consists of 600 baseline scenario–fault combinations, each paired with its corresponding refined Scenic program S<sup>⋆</sup><sub>F</sub> from refinement. During fine-tuning, the pretrained Qwen3-14B base model is kept frozen and loaded in 4-bit quantized form, while trainable LoRA adapters are optimized on these input– target pairs.

## VI. RESULTS AND EVALUATION

In this section, we evaluate FaultGene on held-out scenario–fault combinations from three perspectives: scenario validity and executability, consistency with known fault-sensitivity trends, and overall testing efficiency.

## A. Scenario Validity, Executability, and the Effect of Iterative Refinement and Fine-Tuning

We first evaluate whether fine-tuned FaultGene generator produces valid and executable tests and how iterative refinement and fine-tuning affect semantic alignment and execution quality. A test is considered valid if it compiles successfully in Scenic and executes without runtime failure. Table I summarizes the results for a modular autonomous driving stack and an endto-end system based on PilotNet [35], demonstrating that the same framework supports different AD architectures.

Compilation validity and executability. For the modular stack, one-shot generation without refinement achieves only 68% compilation success, with failures mainly caused by syntax errors or invalid parameter combinations. Among compiled scenarios,

81% complete closed-loop execution, while the remaining failures primarily result from physically infeasible initializations, including overlapping vehicles and deadlocked traffic configurations. With fine-tuned FaultGene, compilation success rises to 96% and execution success to 93%, indicating that executionguided correction and fine-tuning resolve many invalid cases.

A similar pattern appears for the end-to-end system. Without refinement, 71% of generated scenarios compile successfully, and 74% of compiled scenarios complete execution. With fine-tuned FaultGene, compilation and execution success increase to 94% and 90%, respectively, showing that the refinement benefits both modular and end-to-end systems.

Semantic alignment. ScenarioFeat measures whether generated fault-augmented scenarios preserve the intended behavior by scoring predefined scenario feature categories. As shown in Table I, one-shot generation achieves semantic alignment scores of 0.45 and 0.43 for the modular and end-to-end systems, respectively. Fine-tuned FaultGene improves these scores to 0.89 and 0.86, indicating that refinement and fine-tuning improve both executability and preservation of the intended scenario structure and behavior.

## B. Consistency with ADDT Fault Trends

A key question is whether FaultGene-generated tests preserve the system-level fault-sensitivity trends previously observed in ADDT. We therefore compare the qualitative sensor- and compute-fault behaviors exposed by FaultGene with those reported in ADDT. The objective is not to reproduce identical mission success values, but to preserve relative vulnerability patterns across scenarios and software modules.

Sensor-fault consistency. Table II compares ADDT and FaultGene under different camera frame-drop rates. In both evaluations, the impact of sensor faults is scenario dependent: vehicle following is comparatively robust, large-curve turning shows moderate degradation, and overtaking is the most sensitive. Although the absolute success rates differ, FaultGene preserves this overall ordering as the drop rate increases.

In particular, overtaking degrades more rapidly than vehicle following and large-curve turning, reflecting the stronger dependence of interaction-heavy maneuvers on temporally consistent sensing. This agreement indicates that FaultGene preserves the scenariodependent sensor-fault behavior observed in ADDT.

TABLE III: Estimated deployment-stage efficiency for 1,000 total scenario runs and 20 fault types on the same HIL platform.
<table><tr><td>Metric</td><td>Manual ADDT</td><td> $\mathbf { F a u l t G e n e + A D D T ^ { * } }$ </td><td>FaultGene benefit</td></tr><tr><td>Fault-scenario preparation</td><td>Manual engineering</td><td>Single-pass generation</td><td>Automated preparation</td></tr><tr><td>Preparation per fault type</td><td>12 h</td><td>10 min</td><td>72× preparation-time speedup</td></tr><tr><td>Preparation for 20 fault types</td><td>240 h</td><td>3.33 h</td><td>72× preparation-time speedup</td></tr><tr><td>Setup + execution for 1,000 runs</td><td>200 h</td><td>200 h</td><td>Unchanged HIL run time</td></tr><tr><td>Total campaign time</td><td>440 h</td><td>203.33 h</td><td>2.16× speedup; 53.8% less time</td></tr></table>

Assumptions: Each scenario run requires 2 min of setup and 10 min of HIL execution. Manual fault-scenario preparation is estimated at 12 h per fault type, while FaultGene requires 10 min per fault type for single-pass generation using the fine-tuned generator. Thus, for R total runs and K fault types, $T _ { \mathrm { m a n u a l } } \dot { = } 0 . 2 R + 1 2 \dot { K }$ and $T _ { \mathrm { F G } } = 0 . 2 R \dot { + } \dot { K } / 6$ hours. \*FaultGene generation is followed by execution on the same ADDT HIL platform.

![](images/664668b936ec9f42626200250d51e24547cf4bf957c575730245a3df816b8a1a.jpg)  
Fig. 4: Comparison of compute-fault sensitivity trends between ADDT and FaultGene on the modular AD system across 3 driving scenarios. For each scenario, mission success rate is shown under bit-flip faults injected into perception, planning, and control nodes with 1, 3, and 5 injected faults.

The same consistency is observed for the end-toend system: FaultGene follows the ADDT degradation trends for both normal lane keeping and large-curve turning across increasing drop rates. Since PilotNet is designed primarily for lane-keeping, the end-to-end evaluation focuses on these two scenarios.

Compute-fault consistency. We evaluate computefault consistency on the modular architecture. Fig. 4 compares compute-fault sensitivity across the three scenarios and multiple fault locations. FaultGene preserves the main trend observed in ADDT: faults injected into later pipeline stages cause greater missionsuccess degradation, while perception faults are comparatively less damaging. This pattern is especially clear in large-curvature turning and overtaking, where control-node faults cause the sharpest degradation as the number of injected faults increases.

Across modules, perception faults generally produce smaller mission-success drops, planning faults have an intermediate effect, and control faults are the most harmful because they directly affect vehicle actuation. Although the absolute values differ, FaultGene follows the same progression across scenarios, modules, and fault counts, indicating that the generated tests preserve the fault-propagation patterns observed under ADDT HIL execution.

Scenario-specific sensitivity. FaultGene also preserves the scenario-level conclusions reported in ADDT. Overtaking shows greater sensitivity to planning faults because lane-change-like maneuvers depend strongly on timely decision making. Largecurvature turning is particularly vulnerable to control faults, where late-stage actuation errors quickly cause failure. Vehicle following is more stable overall but retains the same module-level ranking.

## C. Testing Efficiency

Table III estimates the deployment-stage efficiency of FaultGene for a campaign of 1,000 scenario runs across 20 fault types. FaultGene reduces fault-scenario preparation from 12 h to 10 min per fault type, yielding a 72× preparation-time speedup. With the same HIL setup and execution time for both workflows, the estimated total campaign time decreases from 440 h to 203.33 h, corresponding to a 2.16× speedup and 53.8% reduction in total time.

ADDT and FaultGene use the same hardware configuration, including a simulation workstation, an NVIDIA Jetson Xavier/Orin, and standard HIL networking, with an estimated hardware cost below \$5,000. In contrast, an autonomous driving test vehicle, such as a Tesla Model Y, typically costs more than \$50,000, excluding insurance, maintenance, calibration, and safety personnel. Thus, FaultGene retains ADDT’s approximately 10× hardware-cost advantage over physical testing while reducing deployment-stage test-preparation effort. Compared with simulation-only testing, it executes the AD stack on real embedded hardware, enabling evaluation of hardware-dependent effects such as compute faults, timing behavior, and fault propagation.

## VII. CONCLUSION

This paper presented FaultGene, an LLM-based framework for generating fault-augmented tests for HIL resilience validation. Given a baseline Scenic/CARLA scenario and structured fault specification, FaultGene generates executable tests for closedloop evaluation across modular and end-to-end AD architectures. Manual ADDT references and execution feedback are used only for offline supervision construction and fine-tuning, while deployment requires neither reference executions nor iterative refinement. On held-out scenario–fault combinations, FaultGene improves semantic alignment, compilation, and execution success over one-shot generation while preserving the main HIL sensor- and compute-fault sensitivity trends.

FaultGene reduces manual Scenic editing and faultaugmented tests construction to providing a baseline scenario and fault specification. Although evaluated on ADS, the framework can be adapted to other CPS by replacing the scenario representation, execution metrics, and fault-injection interfaces with domain-specific counterparts. Similarly, although the current evaluation uses LLFI and NVBitFI, FaultGene is not tied to these tools and can incorporate other fault-injection backends. A current limitation is that timing effects are evaluated after generation rather than incorporated into the generation objective. Future work will integrate HIL timing measurements and broader fault-injection backends into the feedback process to expose both safety- and timing-critical failures.

## REFERENCES

[1] W. Huang, K. Wang, Y. Lv, and F. Zhu, “Autonomous vehicles testing methods review,” in 2016 IEEE 19th International Conference on Intelligent Transportation Systems (ITSC). IEEE Press, 2016, p. 163–168. [Online]. Available: https://doi.org/10.1109/ITSC.2016.7795548

[2] A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, and V. Koltun, “CARLA: An open urban driving simulator,” in Proceedings of the 1st Annual Conference on Robot Learning, 2017, pp. 1–16.

[3] D. J. Fremont, T. Dreossi, S. Ghosh, X. Yue, A. L. Sangiovanni-Vincentelli, and S. A. Seshia, “Scenic: a language for scenario specification and scene generation,” in Proceedings of the 40th ACM SIGPLAN Conference on Programming Language Design and Implementation. ACM, Jun. 2019, p. 63–78. [Online]. Available: http://dx.doi.org/10.1145/3314221.3314633

[4] T. Dreossi, D. J. Fremont, S. Ghosh, E. Kim, H. Ravanbakhsh, M. Vazquez-Chanlatte, and S. A. Seshia, “Verifai: A toolkit for the design and analysis of artificial intelligence-based systems,” 2019. [Online]. Available: https://arxiv.org/abs/1902.04245

[5] Y. Zhao, J. Zhou, D. Bi, T. Mihalj, J. Hu, and A. Eichberger, “A survey on the application of large language models in scenario-based testing of automated driving systems,” 2026. [Online]. Available: https://arxiv.org/abs/2505.16587

[6] B. Yu, C. Yuan, Z. Wan, J. Tang, F. Kurdahi, and S. Liu, “Addt – a digital twin framework for proactive safety validation in autonomous driving systems,” 2025. [Online]. Available: https://arxiv.org/abs/2504.09461

[7] A. Rasheed, O. San, and T. Kvamsdal, “Digital twin: Values, challenges and enablers from a modeling perspective,” IEEE Access, vol. 8, pp. 21 980–22 012, 2020.

[8] R. J. Somers, J. A. Douthwaite, D. J. Wagg, N. Walkinshaw, and R. M. Hierons, “Digital-twin-based testing for cyber–physical systems: A systematic literature review,” Inf. Softw. Technol., vol. 156, no. C, Apr. 2023. [Online]. Available: https://doi.org/10.1016/j.infsof.2022.107145

[9] X. Zhou, M. Li, Y. Zhang, and H. Wang, “Safety-oriented digital twins for autonomous driving validation,” in IEEE Intelligent Vehicles Symposium (IV), 2023, pp. 1245–1252.

[10] F. Mihalic, M. Trunti ˇ c, and A. Hren, “Hardware-in-the-loopˇ simulations: A historical overview of engineering challenges,” Electronics, vol. 11, no. 15, 2022. [Online]. Available: https://www.mdpi.com/2079-9292/11/15/2462

[11] M. Abboush, C. Knieke, and A. Rausch, “A virtual testing framework for real-time validation of automotive software systems based on hardware in the loop and fault injection,” Sensors, vol. 24, no. 11, p. 3733, 2024.

[12] M. Taheri, M. H. Ahmadilivani, M. Jenihhin, M. Daneshtalab, and J. Raik, “Appraiser: Dnn fault resilience analysis employing approximation errors,” in International Symposium on Design and Diagnostics of Electronic Circuits and Systems (DDECS), May 2023, also available at arXiv:2305.19733. [Online]. Available: http://www.es.mdu.se/publications/6682-

[13] J. Ordonez and C. Yang, “Enhancing dnn accelerator integrity˜ via selective and permuted recomputation,” arXiv preprint, 2025, preprint, https://arxiv.org/abs/2212.02649.

[14] S. Jha, S. S. Banerjee, J. Cyriac, Z. T. Kalbarczyk, and R. K. Iyer, “Avfi: Fault injection for autonomous vehicles,” 2018 48th Annual IEEE/IFIP International Conference on Dependable Systems and Networks Workshops (DSN-W), pp. 55–56, 2018.

[15] S. Jha, T. Tsai, S. Hari, M. Sullivan, Z. Kalbarczyk, S. W. Keckler, and R. K. Iyer, “Kayotee: A fault injection-based system to assess the safety and reliability of autonomous vehicles to faults and errors,” in IEEE International Workshop on Automotive Reliability & Test, July 2019, also available as arXiv:1907.01024.

[16] K. Viswanadha, F. Indaheng, J. Wong, E. Kim, E. Kalvan, Y. Pant, D. J. Fremont, and S. A. Seshia, “Addressing the ieee av test challenge with scenic and verifai,” in 2021 IEEE International Conference on Artificial Intelligence Testing (AITest), 2021, pp. 136–142.

[17] J. Wang, A. Pun, J. Tu, S. Manivasagam, A. Sadat, S. Casas, M. Ren, and R. Urtasun, “Advsim: Generating safety-critical scenarios for self-driving vehicles,” in Proceedings of the

IEEE/CVF Conference on Computer Vision and Pattern Recog nition (CVPR), June 2021, pp. 9909–9918.

[18] A. Harder and M. Behl, “Automated traffic scenario description extraction using video transformers,” in 2024 Design, Automation & Test in Europe Conference & Exhibition (DATE), 2024, pp. 1–6.

[19] Z. Xu, Y. Zhang, E. Xie, Z. Zhao, Y. Guo, K.-Y. K. Wong, Z. Li, and H. Zhao, “Drivegpt4: Interpretable endto-end autonomous driving via large language model,” 2024. [Online]. Available: https://arxiv.org/abs/2310.01412

[20] C. Cui, Z. Yang, Y. Zhou, Y. Ma, J. Lu, L. Li, Y. Chen, J. Panchal, and Z. Wang, “Personalized autonomous driving with large language models: Field experiments,” 2024. [Online]. Available: https://arxiv.org/abs/2312.09397

[21] Y. Fu, A. Jain, X. Di, X. Chen, and Z. Mo, “Drivegenvlm: Real-world video generation for vision language model based autonomous driving,” 2024. [Online]. Available: https://arxiv.org/abs/2408.16647

[22] X. Yang, L. Wen, Y. Ma, J. Mei, X. Li, T. Wei, W. Lei, D. Fu, P. Cai, M. Dou, B. Shi, L. He, Y. Liu, and Y. Qiao, “Drivearena: A closed-loop generative simulation platform for autonomous driving,” 2024. [Online]. Available: https://arxiv.org/abs/2408.00415

[23] H. Shao, Y. Hu, L. Wang, S. L. Waslander, Y. Liu, and H. Li, “Lmdrive: Closed-loop end-to-end driving with large language models,” 2023. [Online]. Available: https://arxiv.org/abs/2312.07488

[24] “Unreal engine,” https://www.unrealengine.com/en-US/, accessed: 2023-12-10.

[25] “Openscenario,” https://www.asam.net/standards/detail/openscenario/v200/, accessed: 2023-12-10.

[26] Z. Wan, Y. Gan, B. Yu, S. Liu, A. Raychowdhury, and Y. Zhu, “The vulnerability-adaptive protection paradigm,” Communications of the ACM, vol. 67, no. 9, pp. 66–77, 2024.

[27] S. Jha, S. Banerjee, T. Tsai, S. K. Hari, M. B. Sullivan, Z. T. Kalbarczyk, S. W. Keckler, and R. K. Iyer, “Ml-based fault injection for autonomous vehicles: A case for bayesian fault injection,” in 2019 49th annual IEEE/IFIP international conference on dependable systems and networks (DSN). IEEE, 2019, pp. 112–124.

[28] Q. Lu, M. Farahani, J. Wei, A. Thomas, and K. Pattabiraman, “Llfi: An intermediate code-level fault injection tool for hardware faults,” in 2015 IEEE International Conference on Software Quality, Reliability and Security. IEEE, 2015, pp. 11–16.

[29] T. Tsai, S. K. S. Hari, M. Sullivan, O. Villa, and S. W. Keckler, “Nvbitfi: Dynamic fault injection for gpus,” in 2021 51st Annual IEEE/IFIP International Conference on Dependable Systems and Networks (DSN). IEEE, 2021, pp. 284–291.

[30] Y. Wang, Y. Kordi, S. Mishra, A. Liu, N. A. Smith, D. Khashabi, and H. Hajishirzi, “Self-instruct: Aligning language models with self-generated instructions,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), A. Rogers, J. Boyd-Graber, and N. Okazaki, Eds. Toronto, Canada: Association for Computational Linguistics, Jul. 2023, pp. 13 484–13 508. [Online]. Available: https://aclanthology.org/2023.acl-long.754/

[31] J. Huang, S. Gu, L. Hou, Y. Wu, X. Wang, H. Yu, and J. Han, “Large language models can self-improve,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, H. Bouamor, J. Pino, and K. Bali, Eds. Singapore: Association for Computational Linguistics, Dec. 2023, pp. 1051–1068. [Online]. Available: https://aclanthology.org/2023.emnlp-main.67/

[32] B. Huang, Y. Li, E. Xie, F. Liang, L. Wang, M. Shen, F. Liu, T. Wang, P. Luo, and J. Shao, “Fast-bev: Towards realtime on-vehicle bird’s-eye view perception,” 2023. [Online]. Available: https://arxiv.org/abs/2301.07870

[33] H. Caesar, J. Kabzan, K. S. Tan, W. K. Fong, E. Wolff, A. Lang, L. Fletcher, O. Beijbom, and S. Omari, “Nuplan: A closed-loop ml-based planning benchmark for autonomous vehicles,” in CVPR ADP3 Workshop, 2021.

[34] M. Quigley, K. Conley, B. Gerkey, J. Faust, T. Foote, J. Leibs, R. Wheeler, and A. Ng, “Ros: an open-source robot operating system,” in ICRA Workshop on Open Source Software, vol. 3, 2009, pp. 1–6.

[35] M. Bojarski, C. Chen, J. Daw, A. Degirmenci, J. Deri,˘ B. Firner, B. Flepp, S. Gogri, J. Hong, L. Jackel, Z. Jia, B. Lee, B. Liu, F. Liu, U. Muller, S. Payne, N. K. N. Prasad, A. Provodin, J. Roach, T. Rvachov, N. Tadimeti, J. van Engelen, H. Wen, E. Yang, and Z. Yang, “The nvidia pilotnet experiments,” 2020. [Online]. Available: https://arxiv.org/abs/2010.08776

[36] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, “Qlora: Efficient finetuning of quantized llms,” in Advances in Neural Information Processing Systems, vol. 36, 2023.