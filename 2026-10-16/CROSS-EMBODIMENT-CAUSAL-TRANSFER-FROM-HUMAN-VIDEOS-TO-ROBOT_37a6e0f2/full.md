# CROSS-EMBODIMENT CAUSAL TRANSFER FROM HUMAN VIDEOS TO ROBOT MANIPULATION

Xingyu Zhang<sup>1,2</sup>, Yingchao Li<sup>3</sup>, Peizheng Guo<sup>2,1</sup>, Jie Hu<sup>3</sup>, Changwen Zheng<sup>2</sup>, Wenwen Qiang<sup>2∗</sup>

<sup>1</sup>University of Chinese Academy of Sciences <sup>2</sup>Institute of Software, Chinese Academy of Sciences <sup>3</sup>Meituan

Correspondence: qiangwenwen@iscas.ac.cn

October 16, 2026

## ABSTRACT

Scaling robot manipulation models requires large amounts of interaction data, yet collecting diverse real-robot experience remains costly. Human manipulation videos, in contrast, provide abundant and diverse task experience that could potentially benefit robot learning. However, differences in morphology, action spaces, and control mechanisms make human motions unsuitable as direct supervision for robot actions, raising a scientific question: what knowledge in human videos should be transferred across embodiments to robot manipulation? To address this question, we analyze subtask transfer through the lens of causal beams, focusing on which contextual conditions and interaction dependencies should be retained to support local task effects. Based on this perspective, we propose Causal-Beam-Guided Subtask Transfer (CBST). CBST learns manipulation-related representations from human videos and grounds them in task-level semantic variables using robot demonstrations. Within a predefined manipulation model, causal-beam constraints guide contextual information selection by requiring retained assignments to sustain local mechanism outputs while preserving nontrivial responsiveness to retained inputs. The selected semantic information conditions a world-action model, allowing the robot to generate actions from its own observations without requiring human–robot motion correspondence. Experiments on robot manipulation benchmarks show improvements in task success over the corresponding world-action model baselines.

## 1 Introduction

Recent robot manipulation models are evolving toward World Action Models (WAMs) that jointly model future scene evolution and robot behavior, enabling predictions of interaction outcomes [Kim et al., 2026, Li et al., 2026]. However, scaling real-world robot data requires dedicated hardware, human operation, and physical deployment, making broad coverage of tasks, objects, and scenes expensive. In contrast, human manipulation videos are abundant and capture diverse task processes and object interactions, making them an attractive source of additional experience for robot learning [Khazatsky et al., 2025, Ma et al., 2026, Wang et al., 2023]. Yet humans and robots differ substantially in morphology, action spaces, and control mechanisms, so concrete human actions cannot be directly reused as robot supervision [Bahl et al., 2022, Xu et al., 2023, Kim et al., 2025a]. This leads to the central question of this work: what knowledge in human manipulation videos should be transferred across embodiment differences to robot manipulation?

A complete human manipulation sequence contains two types of information: object-interaction relations that support task progress, and concrete motion realizations that are closely tied to the human embodiment. For long-horizon manipulation tasks, we first organize continuous execution into local task processes using temporally extended subtasks, thereby avoiding the requirement that the robot reproduce human motion step by step. However, defining a subtask only specifies the local scope within which transfer takes place; it does not further determine which information within that scope should be preserved together with the task experience. Retaining only the task intention or boundary states of a subtask may discard the conditions and interaction relations that determine how a local effect is realized, whereas retaining the complete human execution trajectory reintroduces substantial embodiment-specific motion details.

![](images/fe7b18e843003791c9b0930d57a41b1252875a9f4715aeb50f8bc456c32cfbfc.jpg)  
Figure 1: The cross-embodiment transfer problem. Human and robot demonstrations share task objectives but differ in execution. Examples of placing a bowl, opening a lid, and lifting a book motivate the question: what task information should transfer beyond embodiment-specific motion and goal endpoints?

We study this selective information-preservation problem through the lens of causal beams [Pearl, 2009]. Given a structural model and a specific context, a causal beam allows part of the inputs to be projected out in a contextdependent manner, while requiring the actual values of the retained variables to sustain the actual outcome of the current mechanism and the projected mechanism to retain nontrivial responsiveness to its retained inputs. This perspective motivates separating task-level abstraction from contextual information selection. Task-level variables provide an abstraction of execution, while causal-beam conditions constrain which semantic dependencies are retained within a specified model.

Based on this, we propose Causal-Beam-Guided Subtask Transfer (CBST). CBST first learns manipulation-related temporal representations through human-video pretraining and grounds them, under robot demonstration supervision, into a candidate SCM instantiated from shared manipulation templates. A contextual subtask-conditioning module then selects a contextual causal beam from the candidate SCM according to the current execution context, producing the local task effect, relevant conditions, and their effect-relevant dependencies. This structural information conditions the target robot WAM, while concrete actions are still generated from the robot’s current state and its own action space. CBST therefore combines human-video representation transfer with causal-beam-constrained selection of contextual subtask information for robot execution.

Our contributions are summarized as: (1) We study human-to-robot learning from the perspective of cross-embodiment transfer, focusing on which task-progress structures should remain transferable after embodiment-specific motion is discarded; (2) We propose Causal-Beam-Guided Subtask Transfer (CBST), which combines a shared candidate manipulation SCM with learned contextual causal beams to preserve effect-relevant conditions and dependencies for conditioning robot WAMs; (3) We systematically evaluate CBST on WAMs including LingBot-VA and Cosmos Policy, examining the contributions of human video pretraining, context-specific causal selection, and their effects on robot control.

## 2 Related Works

Human-to-robot transfer aims to exploit abundant human demonstrations while addressing the embodiment gap between human and robot execution. Motion-oriented approaches seek representations that remain useful across embodi ments while retaining execution information. WHIRL aligns human and robot videos to infer task intent and initializes robot policy learning from third-person human demonstrations [Bahl et al., 2022], while EgoMimic uses egocentric human videos with 3D hand tracking, cross-domain alignment, and joint training on human and robot demonstrations [Kareer et al., 2024]. Latent Action Pretraining (LAPA) instead learns discrete latent actions from videos without robot action labels and subsequently maps them to robot actions using limited robot data [Ye et al., 2025]. Other approaches move toward higher-level transferable representations. MimicPlay learns latent plans from large-scale human play videos and uses them to guide a low-level robot controller [Wang et al., 2023]; XSkill discovers cross-embodiment skill prototypes from unlabeled human and robot videos and maps them to robot actions with a conditional policy [Xu et al., 2023]; Translating Robot Skills learns unsupervised correspondences between skill sequences across morphologically different agents [Shankar et al., 2022]; and UniSkill learns embodiment-agnostic skill representations that allow skills extracted from human video prompts to guide robot policies trained on robot data [Kim et al., 2025a]. More recently, Gen2Act generates human future videos for novel tasks and conditions a robot policy on the generated motion information [Bharadhwaj et al., 2024].

Temporal abstraction organizes long-horizon behavior into reusable units beyond primitive actions. The options framework formalizes temporally extended policies through initiation sets, internal policies, and termination conditions [Sutton et al., 1999], while later robotic methods model skill feasibility and composition, such as SayCan [Ahn et al., 2022] and relational skill-precondition learning. Another line of work represents task progress through intermediate subgoals: Video Language Planning generates hierarchical visual and language plans [Du et al., 2023], TaKSIE conditions visual subgoal generation on task progress [Kang and Kuo, 2024], and Latent Space Backward Planning recursively constructs intermediate latent goals from the final task goal [Liu et al., 2025]. More recent worldmodel methods integrate such temporal abstraction directly into planning: Hierarchical Planning with Latent World Models learns dynamics at multiple temporal scales and uses high-level latent subgoals to guide low-level planning [Zhang et al., 2026], while SAGE similarly exploits temporally extended latent subgoals for action-sequence generation [Cheng et al., 2026].

World models learn predictive dynamics to support action selection and planning. Early latent-dynamics approaches such as PlaNet perform planning directly in learned latent spaces [Hafner et al., 2019], while more recent methods extend this idea to visual robot manipulation: DINO-WM predicts future visual features for zero-shot planning [Zhou et al., 2025], V-JEPA 2-AC learns action-conditioned latent dynamics for model-predictive control [Assran et al., 2025], and IRASim models fine-grained action-conditioned visual futures for simulation and planning [Zhu et al., 2025]. Recent World-Action Models further unify future prediction with action generation. WorldVLA jointly models visual and action tokens in an autoregressive framework [Cen et al., 2025]; Cosmos Policy generates robot actions together with future-state images and value estimates [Kim et al., 2026]; LingBot-VA jointly learns future-frame prediction and policy execution in a shared visual–action latent space [Li et al., 2026]; and τ -WM further combines video prediction, action generation, and action evaluation within a unified model [Zhou et al., 2026].

## 3 Problem Definition and Analysis

## 3.1 Problem Formulation

Robot manipulation is a sequential decision-making process: given a task instruction $^ { g , }$ the robot generates actions based on the available information and obtains new observations through interaction with the environment. Let $s _ { t }$ denote the robot observation at time $t ,$ comprising visual input and available proprioceptive measurements, and let $a _ { t }$ denote a robot action. The interaction history is $h _ { t } = ( s _ { 0 } , a _ { 0 } , \ldots , a _ { t - 1 } , s _ { t } )$ . We consider a world-action model (WAM) that jointly models robot actions and future observations conditioned on the interaction history $h _ { t }$ and task instruction $g .$

Given $h _ { t }$ and $^ { g , }$ the WAM generates an action chunk and predicts future observations through the joint conditional distribution $p _ { \theta } ( A _ { t } , S _ { t } ^ { + } \ | \ h _ { t } , g )$ . Here, $A _ { t } = ( a _ { t } , \dotsc , a _ { t + H - 1 } )$ denotes an action chunk of length $H ,$ and $S _ { t } ^ { + } =$ $\left( s _ { t + \delta } \right) _ { \delta \in \mathcal { T } _ { H } }$ denotes the future-observation sequence at offsets $\dot { \mathcal { T } _ { H } } \subseteq \{ 1 , \dots , H \}$ . After executing the action chunk, the robot updates the interaction history using newly acquired observations, forming a closed-loop control process.

## 3.2 Causal Beams

We use a recursive structural causal model $M = ( U , V , { \mathcal { F } } )$ , where each non-root variable $V _ { i } \in V$ is determined by a structural mechanism $V _ { i } = f _ { i } ( \mathrm { P A } _ { i } , U )$ . A context $U = u$ determines the actual values $V _ { i } ( u )$ . For the causal-beam construction, the parent set of each mechanism is partitioned as

$$
\mathrm { P A } _ { i } = C _ { i } \dot { \cup } N _ { i } ,\tag{1}
$$

where $C _ { i }$ denotes the retained parents and $N _ { i }$ the remaining parents. The retained parents must sustain the actual output:

$$
f _ { i } ( C _ { i } ( u ) , n _ { i } , u ) = V _ { i } ( u ) , \qquad \forall n _ { i } \in \mathrm { V a l } ( N _ { i } ) .\tag{2}
$$

(b) Human and robot execution  
![](images/b90f047f3b11109043c7fad971bb6c307f180ddb36a334aa3107555396566c69.jpg)  
Figure 2: (a) Candidate local manipulation SCM for a fixed task goal, showing the endogenous parent structure. X and I are boundary inputs, and the mechanisms for D and Y may differ across embodiments. Variable groups are instantiated into task-specific variables before beam selection. (b) Illustrative human and robot motion realizations of the same bowl-placement task.

In this work, we use the natural-beam specialization of the general causal-beam construction [Pearl, 2009]. The remaining inputs are fixed at their actual values, giving the projected mechanism

$$
f _ { i } ^ { u } ( c _ { i } ) = f _ { i } ( c _ { i } , N _ { i } ( u ) , u ) .\tag{3}
$$

The projection must remain nontrivially responsive to its retained inputs:

$$
\exists c _ { i } ^ { \prime } \in \mathrm { V a l } ( C _ { i } ) \quad \mathrm { s . t . } \quad f _ { i } ^ { u } ( c _ { i } ^ { \prime } ) \neq V _ { i } ( u ) .\tag{4}
$$

Thus, the actual assignments of the retained parents sustain the current mechanism’s output, while the projected mechanism is nonconstant in the retained inputs considered jointly. This local responsiveness condition does not establish that every retained parent is individually necessary or that each retained upstream variable has an intervention effect on the downstream effect Y in the composed model. A collection of such projected mechanisms forms a natural beam only when their retained and projected variables are mutually compatible across mechanisms. The full compatibility conditions are given in Appendix A.

## 3.3 Causal Analysis of Cross-Embodiment Transfer

Humans and robots may accomplish the same task through different motion realizations because of differences in morphology, action spaces, and control mechanisms. This motivates describing local task processes using semantic variables for task conditions, manipulation attempts, object-state changes, and effects, without requiring step-by-step correspondence between human and robot motion. Defining these variables, however, does not determine which conditions and dependencies should be retained in a particular execution context. We use causal-beam conditions to guide this contextual selection within a specified task-level structural model.

To formalize the task-relevant structure under consideration, we use a candidate SCM as Fig.2(a) for a local manipulation process. We assume that human and robot executions of the same local manipulation can be described using a shared set of task-level variable semantics and candidate dependency relations, while their embodiment-specific structural functions may differ. This shared schema defines the hypothesis space in which cross-embodiment structural correspondence is considered; it does not assert that every candidate dependency is active in every execution context. Sharing this schema does not guarantee that a beam valid in one embodiment remains valid in another; the beam conditions must be evaluated relative to the instantiated mechanisms and execution context. Let X denote task-relevant conditions at the beginning of the interval, I the manipulation attempt, D the resulting object-state change, and Y the realized local effect:

$$
X = U _ { X } , I = U _ { I } , D = f _ { D } ^ { e } ( X , I , U _ { D } ) , Y = f _ { Y } ^ { e } ( X , D , U _ { Y } ) ,\tag{5}
$$

where $e \in \{ h , r \}$ denotes the embodiment. Thus, humans and robots share the task-level variables and the hypothesis space of possible dependencies, but not necessarily the concrete mechanisms that realize these dependencies. The mechanisms $f _ { D } ^ { e }$ and $f _ { Y } ^ { e }$ may therefore differ between humans and robots. The distinction between I and D separates an attempted manipulation from the object change that is actually realized. This schema specifies candidate taskrelevant variable roles and dependencies rather than a complete physical model; task-specific variables are instantiated from the corresponding manipulation specification. Within this candidate structure, causal-beam selection determines which dependencies are retained in a particular execution context.

![](images/b4e0c9e6cb7b7e438a1a05bf5fbbf353eba081c2337a84239716cfd98ea146e5.jpg)  
Figure 3: Overview of Causal-Beam-Guided Subtask Transfer (CBST). Human-video pretraining and robot adaptation support state prediction and contextual beam selection. The resulting semantic conditions guide a frozen world– action model to predict actions and future states. Solid and dashed arrows denote inference and training/supervision, respectively.

Consider moving a bowl into a drawer. Let B indicate whether a transport motion is executed, G whether the grasp is effective, O whether the drawer entrance is accessible, D whether the bowl is carried by the transport motion, and $Y$ whether it reaches the target region. Here $X = ( G , O )$ and $I = B$ . A simplified local model is

$$
D = B \wedge G , Y = D \wedge O .\tag{6}
$$

In a successful context with $B = G = O = 1$ , no proper subset of $\{ B , G \}$ sustains $D = 1$ X̂, and no proper subset of $\{ D , O \}$ sustains $Y = 1 .$ . A projection of the original mechanism that omits one of these parents may still match the observed endpoint, but it is not a valid sustaining projection in this context. In contrast, when transport is attempted with an ineffective grasp, $B = 1 , G = 0 , O = \mathrm { i }$ , one valid natural projection retains G for D and D for Y, yielding $D = G$ and $Y = \bar { D }$ . Thus, the same candidate SCM can admit different valid projections under different execution contexts. Appendix A provides the full verification of the causal-beam conditions. This example highlights two points relevant to transfer. First, matching an observed endpoint does not imply that the dependencies producing that endpoint have been preserved. Second, the structure retained for a local process is context dependent.

This motivates using causal-beam conditions to constrain which task-level dependencies are preserved in the transferred representation, with contextual selection performed under the robot-side reference mechanisms rather than by directly reusing a human-side beam.

Definition 3.1 (Causal Subtask Representation). Given a candidate manipulation SCM M for a local task process and a context u, let $M _ { u }$ denote a causal beam of M. We define the causal subtask representation as

$$
\sigma = ( Y ( u ) , \mathcal { C } _ { u } , \mathcal { R } _ { u } ) ,\tag{7}
$$

where $Y ( u )$ denotes the realized local effect, $\mathcal { C } _ { u } = \{ ( C _ { i } , C _ { i } ( u ) ) \} _ { i }$ collects the retainedparents and their actual values, and $\mathcal { R } _ { u }$ records the parent–child relations retained within the projected local mechanisms. The projected mechanisms $f _ { i } ^ { u }$ are used to determine whether a retained parent set satisfies the causal-beam conditions, but the representation transfers the retained task-level dependencies rather than requiring identical projectedfunctions across embodiments. Excluding a parent from the transferred dependency set does not remove its role in the projection context: its fixed value can still determine how the projected mechanism responds to changes in the retained inputs

The local task goal specifies the desired change, whereas σ describes dependencies supporting the observed loca effect in a given context. An unrealized goal alone does not distinguish ongoing execution from failure.

## 4 Method

We propose Causal-Beam-Guided Subtask Transfer (CBST). Human-video pretraining initializes a visual–temporal module, which is then adapted with robot demonstrations to estimate task-level semantic variables from robot observations. Using these estimates, a learned selector chooses contextual parent assignments subject to causal-beam constraints within a predefined manipulation model. The selected assignments are rendered as language conditions for WAM, which combines them with the robot’s native observations to generate actions and future observation predictions.

## 4.1 Overall Architecture

Following the task-level formulation in Section 3.3, CBST maps robot observations to semantic estimates, selects contextual dependencies, and uses the resulting subtask information to condition action generation. Given the interaction history $h _ { t }$ and task instruction g, a visual–temporal encoder and semantic prediction heads produce

$$
z _ { t } = E _ { \eta } ( h _ { t } , g ) , \hat { \mathbf { v } } _ { t } = P _ { \omega } ( z _ { t } ) ,\tag{8}
$$

where $z _ { t }$ summarizes the observed manipulation process and $\hat { \mathbf { v } } _ { t }$ estimates the conditions, manipulation attempts, object-state changes, and effects defined by the local SCM. Discretizing these predictions gives the contextual assignment $\bar { \bf v } _ { t }$ . Predicted states and task-specified prerequisite rules also determine the currently admissible goals $\hat { G } _ { t }$

The estimated context is then used to instantiate the causal-beam conditions in Section 3.2. Under fixed reference mechanisms $f _ { i } ^ { \mathrm { r e f } }$ , let $\ddot { \mathcal { A } } _ { i , t }$ contain the parent subsets satisfying sustenance and responsiveness at $\bar { \mathbf { v } } _ { t } . ~ \mathbf { A }$ learned selector assigns probabilities $p _ { i , t } ( C )$ from $z _ { t }$ and the semantic predictions, and jointly selects compatible subsets:

$$
\begin{array} { r l } & { \{ C _ { i , t } ^ { * } \} _ { i } \in \underset { \{ C _ { i } \} _ { i } } { \arg \operatorname* { m a x } } \sum _ { i } \log p _ { i , t } ( C _ { i } ) } \\ & { \qquad \mathrm { s u b j e c t ~ t o } \quad C _ { i } \in \hat { \mathcal { A } } _ { i , t } , \quad ( \mathrm { P A } _ { i } \setminus C _ { i } ) \cap C _ { j } = \varnothing \quad \mathrm { f o r ~ a l l } \ i \neq j . } \end{array}\tag{9}
$$

For a feasible selection, the retained assignments and relations form $\hat { \sigma } _ { t } = ( \bar { Y } _ { t } , \hat { \mathcal { C } } _ { t } , \hat { \mathcal { R } } _ { t } )$ , an estimated instance of the causal subtask representation in Definition 3.1. Its validity is relative to the reference mechanisms and estimated context. A deterministic renderer combines the task instruction, admissible goals, and selected subtask information into a contextual description. The text encoder and conditional generative network complete the data flow:

$$
\boldsymbol { r } _ { t } = \mathcal { R } ( \boldsymbol { g } , \hat { G } _ { t } , \hat { \sigma } _ { t } ) , \boldsymbol { c } _ { t } = T _ { \boldsymbol { \theta } } ( \boldsymbol { r } _ { t } ) , ( A _ { t } , \hat { S } _ { t } ^ { + } ) \sim p _ { \boldsymbol { \theta } } ( \cdot \mid \mathcal { O } _ { t } ^ { W } , \boldsymbol { c } _ { t } ) .\tag{10}
$$

Here, $\mathcal { O } _ { t } ^ { W }$ denotes the robot-observation condition, $A _ { t }$ the generated action chunk, and $\hat { S } _ { t } ^ { + }$ the predicted future observations. The renderer includes realized effects and selected variable assignments; retained relations guide field selection but are not explicitly serialized. If beam selection is infeasible, its fields are omitted while task and state conditioning are retained. The WAM and its text encoder remain frozen. After action execution, new observations update $h _ { t } .$ , and the estimation–selection–generation process repeats.

## 4.2 Learning Objectives and Training

Following the analysis in Section 3.3, training grounds the task-level semantic variables and learns contextual parent selection under causal-beam constraints. We first initialize the visual–temporal encoder through action-category classification on human videos:

$$
\begin{array} { r } { \mathcal { L } _ { h } = \mathbb { E } _ { ( o ^ { h } , y ^ { h } ) } \left[ \operatorname { C E } \left( H ( E _ { \eta } ( o ^ { h } ) ) , y ^ { h } \right) \right] , } \end{array}\tag{11}
$$

where $o ^ { h }$ is a video clip, $y ^ { h }$ its category label, and H a classification head. The pretrained encoder then initializes robot adaptation. Robot demonstrations supervise task states and interval variables through $\mathcal { L } _ { \mathrm { s e m } } = \mathcal { L } _ { \mathrm { s t a t e } } + \lambda _ { v } \mathcal { L } _ { \mathrm { v a r } }$ , with each component normalized over valid labels. An auxiliary mechanism-consistency loss ${ \mathcal { L } } _ { \mathrm { m e c h } }$ encourages predicted parent variables to reproduce annotated outputs under the fixed reference mechanisms. This loss uses valid, referenceconsistent annotations, while other valid labels remain available for semantic supervision. The reference mechanisms remain frozen during training.

Structural supervision follows the sustenance and responsiveness conditions in Section 3.2. For an annotated context $v ,$ finite-domain enumeration under the fixed reference mechanisms produces the acceptable parent subsets $\mathcal { A } _ { i } ( v )$ The selector learns to assign probability mass to this set, while a separate regularizer favors compact selections:

$$
\mathcal { L } _ { \mathrm { b e a m } } = - \sum _ { i \in \mathcal { Z } ( v ) } \log \sum _ { C \in A _ { i } ( v ) } p _ { i } ( C ) , \mathcal { L } _ { \mathrm { s i z e } } = \sum _ { i \in \mathcal { Z } ( v ) } \mathbb { E } _ { C \sim p _ { i } ( \cdot \vert C \in A _ { i } ( v ) ) } [ \frac { \vert C \vert } { \vert \mathrm { P A } _ { i } \vert } ] .\tag{12}
$$

Table 1: Success rates (%) on RoboTwin 2.0..
<table><tr><td>Method</td><td>Easy</td><td>Hard</td></tr><tr><td>X-VLA</td><td>72.9</td><td>72.8</td></tr><tr><td>π0</td><td>65.9</td><td>58.4</td></tr><tr><td>π0.5</td><td>82.7</td><td>76.8</td></tr><tr><td>Motus</td><td>88.7</td><td>87.0</td></tr><tr><td>LingBot-VA</td><td>92.93</td><td>91.55</td></tr><tr><td>LingBot-VA + CBST</td><td>94.00</td><td>92.54</td></tr></table>

Table 2: Success rates (%) on LIBERO-Long.
<table><tr><td>Method Long SR (%)</td></tr><tr><td>X-VLA 97.6</td></tr><tr><td>Cosmos Policy (Predict2) 97.6</td></tr><tr><td>LingBot-VA 98.5 ± 0.5 LingBot-VA + CBST 98.60</td></tr><tr><td>Cosmos Policy (Predict2.5) 96.6</td></tr><tr><td>Predict2.5 + CBST 97.33</td></tr></table>

LIBERO: Turn on the stove and place the moka pot  
![](images/0bc8a5794de223483bbd12824209db7269f2ae82ae2320460dd11f58652b1874.jpg)  
Initial state 0.00 s

![](images/270f5b8218d714a2b696f0540e6afef3414872ed4a011afb2206f87788534b30.jpg)  
Operate switch 0.75 s

![](images/104db34fb11a391156af82d4ff69dea4a4af1295401f9ca96bd8610a715eb98d.jpg)  
Stove on

![](images/3dbbc6cc4983b64d4f78cefe46be9442c668a3a29d13abe865200372ef9eb4c6.jpg)  
Grasp pot 2.65 s

![](images/0956fb6f069ddb9551b66baedbefd336da5b64da2ed23236d0971fa7a1725445.jpg)  
Transport pot 3.42 s

![](images/b48db783e2e09a49ee3fef5fe79e9050eda5ac88f16cb03312f7390e07850e41.jpg)  
Place on stove  
Figure 4: A LingBot-VA + CBST execution: operating the switch, turning on the stove, and placing the moka pot on it. Frames follow temporal order; timestamps indicate video playback time.

Here, I(v) contains mechanisms with valid, reference-consistent annotations, complete variable domains, and nonempty acceptable sets. Other valid annotations remain available for semantic supervision. The beam loss permits multiple acceptable subsets; the size regularizer expresses a compactness preference rather than an additional beam-validity condition. Cross-mechanism compatibility is enforced by the joint selection procedure in the preceding subsection. The robot adaptation objective is

$$
{ \mathcal { L } } _ { \mathrm { r o b o t } } = { \mathcal { L } } _ { \mathrm { s e m } } + \lambda _ { m } { \mathcal { L } } _ { \mathrm { m e c h } } + \lambda _ { b } { \mathcal { L } } _ { \mathrm { b e a m } } + \lambda _ { \mathrm { s i z e } } { \mathcal { L } } _ { \mathrm { s i z e } } .\tag{13}
$$

Training updates the trainable encoder components, semantic prediction heads, and selector. The reference mech anisms, WAM, and native text encoder remain frozen. At inference, beam validity is checked using the discrete estimated context, and compatible selections are used to construct the action-generation condition.

## 5 Experiments

## 5.1 Experimental Setup

We pretrain CBST’s ResNet18–GRU visual-temporal encoder on Something-Something V2 (SSv2) [Goyal et al., 2017] and adapt the planner using robot demonstrations, keeping the WAM and its text encoder frozen. We evaluate LingBot-VA on RoboTwin 2.0 and LIBERO, and Cosmos Policy (Predict2.5) on LIBERO and RoboCasa [Chen et al., 2025, Liu et al., 2023, Nasiriany et al., 2024]. Comparisons include the underlying WAMs and representative published policies, using officially reported baseline scores [Li et al., 2026, Kim and Gu, 2026]. We report task success rates (%) under official success criteria, averaged across evaluated tasks. Dataset composition, baseline descriptions, implementation and hardware configurations, and evaluation protocols are provided in Appendix B.

## 5.2 Results on Simulation Benchmarks

We evaluate CBST with two frozen WAM backbones across three simulation benchmarks (Tables 1–3). On RoboTwin 2.0, LingBot-VA + CBST achieves 94.00% and 92.54% success under Easy and Hard settings. On LIBERO-Long, CBST achieves 98.60% with LingBot-VA and 97.33% with Cosmos Policy (Predict2.5). On RoboCasa, Predict2.5 + CBST reaches 72.44% with 50 demonstrations per task, compared with 71.1% for the backbone. Figure 4 illustrates a CBST execution on LIBERO. Full baseline comparisons, detailed results, and additional visualizations appear in Appendices C.

Table 3: Success rates (%) on RoboCasa.
<table><tr><td>Method</td><td>Demos/task</td><td>Average SR</td></tr><tr><td>π0</td><td>300</td><td>62.5</td></tr><tr><td>GR00T-N1.5</td><td>300</td><td>64.1</td></tr><tr><td>Video Policy</td><td>300</td><td>66.0</td></tr><tr><td>FLARE</td><td>300</td><td>66.4</td></tr><tr><td>Cosmos2</td><td>50</td><td>67.1</td></tr><tr><td>Cosmos2.5</td><td>50</td><td>71.1</td></tr><tr><td>Cosmos2.5 + CBST</td><td>50</td><td>72.44</td></tr></table>

Table 4: Ablations on RoboTwin Easy.
<table><tr><td>Component</td><td>Variant</td><td>SR (%)</td></tr><tr><td>Reference</td><td>LingBot-VA CBST</td><td>92.93 94.00</td></tr><tr><td>Pretraining</td><td>w/o human pretraining Shuffled frames</td><td>92.80 92.94</td></tr><tr><td>Conditioning</td><td>State-only Full structure</td><td>92.82 93.42</td></tr><tr><td>Objective</td><td>Random selection w/o  $\mathcal { L } _ { \mathrm { m e c h } }$ </td><td>92.56 93.18</td></tr></table>

![](images/da27d56d2898fb699aa19cf2ece051606e9914cd8995ad9fb40907eaaa4bae2e.jpg)  
Figure 5: CBST execution of cup placement on the AgileX platform. One arm lifts the cup and transfers it to the receiving gripper, which transports it to the target location and releases it on the tabletop. Handover enables crossworkspace transport but is not the task goal. Labels describe observed events, and timestamps indicate video playback time.

## 5.3 Ablation Studies

We examine human-video pretraining, contextual structural selection, and mechanism-consistency supervision on RoboTwin Easy with LingBot-VA as the frozen backbone (Table 4). Pretraining variants remove human-video initialization or shuffle its temporal order; conditioning variants use state-only information, the full candidate structure, or size-matched random subsets; the objective variant removes mechanism-consistency supervision. Robot adaptation data and evaluation settings remain fixed, while conditioning variants share the same trained predictor. More results across both datasets, detailed variant definitions, and computational costs are provided in Appendix D.

## 5.4 Real-World Experiments

We evaluate CBST on an AgileX dual-arm platform across ten object-placement tasks, with 20 trials per task and method. Each task specifies an object and a fixed target location. In cross-workspace configurations, no single arm can reach both the object’s initial position and its destination, requiring coordinated transport through a shared handover region. We compare Cosmos Policy with the Predict2.5 backbone against the same backbone augmented with CBST, and measure full-task success after object release. Task definitions, deployment settings, and the complete result table are provided in Appendix E. Figure 5 shows a recorded CBST cup-placement execution, including the intermediate handover. Additional sequences appear in Appendix E.4.

## 5.5 Human-Video Pretraining

We examine the role of temporal ordering during human-video pretraining by comparing ordered and frame-shuffled SSv2 clips. Both variants use the same ResNet18–GRU architecture, action labels, sampled frame content, and optimization budget; only the temporal order of the pretraining inputs differs. Figure 6 shows the training classification loss and final validation top-1 accuracy after three epochs. Ordered pretraining achieves 25.88% accuracy, compared with 22.15% for the shuffled control, supporting the utility of temporal order for SSv2 action recognition.

We further evaluate the downstream contribution of this pretraining through the robot ablations in Table 4. Under the same robot adaptation protocol, ordered pretraining yields 94.00% success on RoboTwin Easy, compared with 92.94% for shuffled pretraining and 92.80% without human-video pretraining. These comparisons support the practical benefit of ordered human-video pretraining for robot control. The pretrained representation initializes the robot planner; robotspecific prediction heads and structural selection are learned during adaptation.

![](images/8d39a0036d11234ba03279c1ee01434d378ac8ece41b21197879156fb401853a.jpg)  
Figure 6: Temporal-order comparison for ResNet18–GRU pretraining on SSv2. (a) Training classification loss: bold lines show sample-weighted means over non-overlapping blocks of 120 updates; faint traces show the original values. (b) Final top-1 accuracy on 24,777 validation clips after three epochs, with one run per condition.

## 6 Conclusion

We presented Causal-Beam-Guided Subtask Transfer (CBST), which uses a causal-beam perspective to guide contextual information selection for human-to-robot transfer. Human-video pretraining initializes manipulation-related representations, while robot demonstrations ground these representations in task-level semantic variables. Within a predefined manipulation model, causal-beam conditions constrain which contextual assignments are retained and supplied to a frozen world-action model for closed-loop control. Experiments on robot manipulation benchmarks show task-success improvements over the corresponding backbone baselines.

## References

Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Daniel Ho, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Irpan, Eric Jang, Rosario Jauregui Ruano, Kyle Jeffrey, Sally Jesmonth, Nikhil J Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Kuang-Huei Lee, Sergey Levine, Yao Lu, Linda Luu, Carolina Parada, Peter Pastor, Jornell Quiambao, Kanishka Rao, Jarek Rettinghouse, Diego Reyes, Pierre Sermanet, Nicolas Sievers, Clayton Tan, Alexander Toshev, Vincent Vanhoucke, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Mengyuan Yan, and Andy Zeng. Do as i can, not as i say: Grounding language in robotic affordances, 2022. URL https://arxiv.org/abs/ 2204.01691.

Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba, Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, and Nicolas Ballas. V-jepa 2: Self-supervised video models enable understanding, prediction and planning, 2025. URL https://arxiv.org/abs/2506.09985.

Shikhar Bahl, Abhinav Gupta, and Deepak Pathak. Human-to-robot imitation in the wild, 2022. URL https:// arxiv.org/abs/2207.09450.

Homanga Bharadhwaj, Debidatta Dwibedi, Abhinav Gupta, Shubham Tulsiani, Carl Doersch, Ted Xiao, Dhruv Shah, Fei Xia, Dorsa Sadigh, and Sean Kirmani. Gen2act: Human video generation in novel scenarios enables generalizable robot manipulation, 2024. URL https://arxiv.org/abs/2409.16283.

Hongzhe Bi, Hengkai Tan, Shenghao Xie, Zeyuan Wang, Shuhe Huang, Haitian Liu, Ruowen Zhao, Yao Feng, Chendong Xiang, Yinze Rong, Hongyan Zhao, Hanyu Liu, Zhizhong Su, Lei Ma, Hang Su, and Jun Zhu. Motus: A unified latent action world model, 2025. URL https://arxiv.org/abs/2512.13030.

Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, and Ury Zhilinsky. π : A vision-language-action flow model for general robot control, 2026. URL https: //arxiv.org/abs/2410.24164.

Jun Cen, Chaohui Yu, Hangjie Yuan, Yuming Jiang, Siteng Huang, Jiayan Guo, Xin Li, Yibing Song, Hao Luo, Fan Wang, Deli Zhao, and Hao Chen. Worldvla: Towards autoregressive action world model, 2025. URL https: //arxiv.org/abs/2506.21539.

Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Zixuan Li, Qiwei Liang, Xianliang Lin, Yiheng Ge, Zhenyu Gu, Weiliang Deng, Yubin Guo, Tian Nian, Xuanbing Xie, Qiangyu Chen, Kailun Su, Tianling Xu, Guodong Liu, Mengkang Hu, Huan ang Gao, Kaixuan Wang, Zhixuan Liang, Yusen Qin, Xiaokang Yang, Ping Luo, and Yao Mu. Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation, 2025. URL https://arxiv.org/abs/2506.18088.

Letian Cheng, Qi Zhang, and Yisen Wang. Sage: Subgoal-conditioned action generation for latent world model planning, 2026. URL https://arxiv.org/abs/2607.17973.

Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and¨ Yoshua Bengio. Learning phrase representations using RNN encoder–decoder for statistical machine translation. In Alessandro Moschitti, Bo Pang, and Walter Daelemans, editors, Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1724–1734, Doha, Qatar, October 2014. Association for Computational Linguistics. doi: 10.3115/v1/D14-1179. URL https://aclanthology.org/D14-1179/.

Yilun Du, Mengjiao Yang, Pete Florence, Fei Xia, Ayzaan Wahid, Brian Ichter, Pierre Sermanet, Tianhe Yu, Pieter Abbeel, Joshua B. Tenenbaum, Leslie Kaelbling, Andy Zeng, and Jonathan Tompson. Video language planning, 2023. URL https://arxiv.org/abs/2310.10625.

Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim,´ Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, Florian Hoppe, Christian Thurau, Ingo Bax, and Roland Memisevic. The ”something something” video database for learning and evaluating visual common sense, 2017. URL https://arxiv.org/abs/1706.04261.

Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels, 2019. URL https://arxiv.org/abs/1811.04551.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition, 2015. URL https://arxiv.org/abs/1512.03385.

Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Walke, Anna Walling, Haohuan Wang, Lili Yu, and Ury Zhilinsky. π<sub>0.5</sub>: a vision-language-action model with open-world generalization, 2025. URL https://arxiv. org/abs/2504.16054.

Xuhui Kang and Yen-Ling Kuo. Incorporating task progress knowledge for subgoal generation in robotic manipulation through image edits, 2024. URL https://arxiv.org/abs/2410.11013.

Simar Kareer, Dhruv Patel, Ryan Punamiya, Pranay Mathur, Shuo Cheng, Chen Wang, Judy Hoffman, and Danfei Xu. Egomimic: Scaling imitation learning via egocentric video, 2024. URL https://arxiv.org/abs/2410.24221.

Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, Peter David Fagan, Joey Hejna, Masha Itkina, Marion Lepert, Yecheng Jason Ma, Patrick Tree Miller, Jimmy Wu, Suneel Belkhale, Shivin Dass, Huy Ha, Arhan Jain, Abraham Lee, Youngwoon Lee, Marius Memmel, Sungjae Park, Ilija Radosavovic, Kaiyuan Wang, Albert Zhan, Kevin Black, Cheng Chi, Kyle Beltran Hatch, Shan Lin, Jingpei Lu, Jean Mercat, Abdul Rehman, Pannag R Sanketi, Archit Sharma, Cody Simpson, Quan Vuong, Homer Rich Walke, Blake Wulfe, Ted Xiao, Jonathan Heewon Yang, Arefeh Yavary, Tony Z. Zhao, Christopher Agia, Rohan Baijal, Mateo Guaman Castro, Daphne Chen, Qiuyu Chen, Trinity Chung, Jaimyn Drake, Ethan Paul Foster, Jensen Gao, Vitor Guizilini, David Antonio Herrera, Minho Heo, Kyle Hsu, Jiaheng Hu, Muhammad Zubair Irshad, Donovon Jackson, Charlotte Le, Yunshuang Li, Kevin Lin, Roy Lin, Zehan Ma, Abhiram Maddukuri, Suvir Mirchandani, Daniel Morton, Tony Nguyen, Abigail O’Neill, Rosario Scalise, Derick Seale, Victor Son, Stephen Tian, Emi Tran, Andrew E. Wang, Yilin Wu, Annie Xie, Jingyun Yang, Patrick Yin, Yunchu Zhang, Osbert Bastani, Glen Berseth, Jeannette Bohg, Ken Goldberg, Abhinav Gupta, Abhishek Gupta, Dinesh Jayaraman, Joseph J Lim, Jitendra Malik, Roberto

Mart´ın-Mart´ın, Subramanian Ramamoorthy, Dorsa Sadigh, Shuran Song, Jiajun Wu, Michael C. Yip, Yuke Zhu, Thomas Kollar, Sergey Levine, and Chelsea Finn. Droid: A large-scale in-the-wild robot manipulation dataset, 2025. URL https://arxiv.org/abs/2403.12945.

Hanjung Kim, Jaehyun Kang, Hyolim Kang, Meedeum Cho, Seon Joo Kim, and Youngwoon Lee. Uniskill: Imitating human videos via cross-embodiment skill representations, 2025a. URL https://arxiv.org/abs/2505.08787.

Moo Jin Kim and Jinwei Gu. Cosmos policy: Fine-tuning video models for visuomotor control and planning. https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post\_training/predict2/ cosmos\_policy/post\_training.html, January 2026. NVIDIA Cosmos Cookbook.

Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success, 2025b. URL https://arxiv.org/abs/2502.19645.

Moo Jin Kim, Yihuai Gao, Tsung-Yi Lin, Yen-Chen Lin, Yunhao Ge, Grace Lam, Percy Liang, Shuran Song, Ming-Yu Liu, Chelsea Finn, and Jinwei Gu. Cosmos policy: Fine-tuning video models for visuomotor control and planning. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview. net/forum?id=wPEIStHxYH.

Lin Li, Qihang Zhang, Yiming Luo, Shuai Yang, Ruilin Wang, Fei Han, Mingrui Yu, Zelin Gao, Nan Xue, Xing Zhu, Yujun Shen, and Yinghao Xu. Causal world modeling for robot control, 2026. URL https://arxiv.org/abs/ 2601.21998.

Junbang Liang, Pavel Tokmakov, Ruoshi Liu, Sruthi Sudhakar, Paarth Shah, Rares Ambrus, and Carl Vondrick. Video generators are robot policies, 2025. URL https://arxiv.org/abs/2508.00795.

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning, 2023. URL https://arxiv.org/abs/2306.03310.

Dongxiu Liu, Haoyi Niu, Zhihao Wang, Jinliang Zheng, Yinan Zheng, Zhonghong Ou, Jianming Hu, Jianxiong Li, and Xianyuan Zhan. Efficient robotic policy learning via latent space backward planning, 2025. URL https: //arxiv.org/abs/2505.06861.

Junyi Ma, Erhang Zhang, Haoran Yang, Ditao Li, Chenyang Xu, Guangming Wang, and Hesheng Wang. Robot learning from human videos: A survey, 2026. URL https://arxiv.org/abs/2604.27621.

Soroush Nasiriany, Abhiram Maddukuri, Lance Zhang, Adeet Parikh, Aaron Lo, Abhishek Joshi, Ajay Mandlekar, and Yuke Zhu. Robocasa: Large-scale simulation of everyday tasks for generalist robots, 2024. URL https: //arxiv.org/abs/2406.02523.

NVIDIA, :, Johan Bjorck, Fernando Castaneda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi ”Jim” Fan,˜ Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, Joel Jang, Zhenyu Jiang, Jan Kautz, Kaushil Kundalia, Lawrence Lao, Zhiqi Li, Zongyu Lin, Kevin Lin, Guilin Liu, Edith Llontop, Loic Magne, Ajay Mandlekar, Avnish Narayan, Soroush Nasiriany, Scott Reed, You Liang Tan, Guanzhi Wang, Zu Wang, Jing Wang, Qi Wang, Jiannan Xiang, Yuqi Xie, Yinzhen Xu, Zhenjia Xu, Seonghyeon Ye, Zhiding Yu, Ao Zhang, Hao Zhang, Yizhou Zhao, Ruijie Zheng, and Yuke Zhu. Gr00t n1: An open foundation model for generalist humanoid robots, 2025. URL https://arxiv.org/abs/2503.14734.

J. Pearl. Causality. Causality: Models, Reasoning, and Inference. Cambridge University Press, 2009. ISBN 9780521895606. URL https://books.google.com/books?id=f4nuexsNVZIC.

Tanmay Shankar, Yixin Lin, Aravind Rajeswaran, Vikash Kumar, Stuart Anderson, and Jean Oh. Translating robot skills: Learning unsupervised skill correspondences across robots. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, Proceedings ofthe 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 19626–19644. PMLR, 17–23 Jul 2022. URL https://proceedings.mlr.press/v162/shankar22a.html.

Richard S. Sutton, Doina Precup, and Satinder Singh. Between mdps and semi-mdps: A framework for temporal abstraction in reinforcement learning. Artificial Intelligence, 112(1):181–211, 1999. ISSN 0004-3702. doi: https:// doi.org/10.1016/S0004-3702(99)00052-1. URL https://www.sciencedirect.com/science/article/pii/ S0004370299000521.

Chen Wang, Linxi Fan, Jiankai Sun, Ruohan Zhang, Li Fei-Fei, Danfei Xu, Yuke Zhu, and Anima Anandkumar. Mimicplay: Long-horizon imitation learning by watching human play, 2023. URL https://arxiv.org/abs/ 2302.12422.

Mengda Xu, Zhenjia Xu, Cheng Chi, Manuela Veloso, and Shuran Song. Xskill: Cross embodiment skill discovery, 2023. URL https://arxiv.org/abs/2307.09955.

Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Se June Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Bill Yuchen Lin, Lars Liden, Kimin Lee, Jianfeng Gao, Luke Zettlemoyer, Dieter Fox, and Minjoon Seo. Latent action pretraining from videos. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=VYOe2eBQeh.

Wancong Zhang, Basile Terver, Artem Zholus, Soham Chitnis, Harsh Sutaria, Mido Assran, Randall Balestriero, Amir Bar, Adrien Bardes, Yann LeCun, and Nicolas Ballas. Hierarchical planning with latent world models, 2026. URL https://arxiv.org/abs/2604.03208.

Jinliang Zheng, Jianxiong Li, Zhihao Wang, Dongxiu Liu, Xirui Kang, Yuchun Feng, Yinan Zheng, Jiayin Zou, Yilun Chen, Jia Zeng, Ya-Qin Zhang, Jiangmiao Pang, Jingjing Liu, Tai Wang, and Xianyuan Zhan. X-vla: Soft-prompted transformer as scalable cross-embodiment vision-language-action model, 2025. URL https://arxiv.org/abs/ 2510.10274.

Gaoyue Zhou, Hengkai Pan, Yann LeCun, and Lerrel Pinto. DINO-WM: World models on pre-trained visual features enable zero-shot planning. In Forty-second International Conference on Machine Learning, 2025. URL https: //openreview.net/forum?id=D5RNACOZEI.

Pengfei Zhou, Shengcong Chen, Di Chen, Jiaxu Wang, Rongjun Jin, Bingwen Zhu, Yike Pan, Songen Gu, Kuanning Wang, Shufeng Nan, Xingyu Qiu, Chenhao Qiu, Pu Yang, Yunuo Cai, Jianxiong Gao, Yifan Li, Yanwei Fu, Xiangyu Yue, Zhi Chen, and Jianlan Luo. τ<sub>0</sub>-wm: A unified video-action world model for robotic manipulation, 2026. URL https://arxiv.org/abs/2606.01027.

Fangqi Zhu, Hongtao Wu, Song Guo, Yuxiao Liu, Chilam Cheang, and Tao Kong. Irasim: A fine-grained world model for robot manipulation, 2025. URL https://arxiv.org/abs/2406.14540.

## AI Usage Statement

In this work, we used generative AI tools for grammar and language polishing. They were not involved in research ideation, methodology, analysis, or substantive writing. The authors take full responsibility for all contents of the paper.

## A Details of Causal Beams

## A.1 Formal Details of Causal Beam Construction

We consider a recursive structural causal model $M = ( U , V , { \mathcal { F } } )$ , where U denotes the exogenous variables, V the endogenous variables, and $\mathcal { F }$ the set of structural mechanisms. Each non-root endogenous variable $V _ { i }$ is determined by $\bar { V _ { i } } = f _ { i } ( \mathrm { P A } _ { i } , U )$ , where $\mathrm { P A } _ { i } \subseteq V$ denotes the endogenous parents of $V _ { i } . \mathrm { ~ A ~ }$ context $U = u$ determines the actual value $V _ { i } ( u )$ of every endogenous variable. For a variable set $Z ,$ we write $Z _ { W  w } ( u )$ for its value in the original model under the intervention $W \stackrel { - } {  } w$ and context u.

Definition A.1 (Causal Beam [Pearl, 2009]). Given a recursive structural causal model $M = ( U , V , { \mathcal { F } } )$ and a context u, a causal beam $M _ { u } = \left( u , V , \mathcal { F } ^ { u } \right)$ is constructed by projecting the structural mechanisms asfollows.

For each projected non-root mechanism $V _ { i } = f _ { i } ( \mathrm { P A } _ { i } , U )$ , partition its parent set as $\mathrm { P A } _ { i } = C _ { i } \dot { \cup } N _ { i }$ , where $C _ { i }$ denotes the retained parents and $N _ { i }$ the remaining parents. The actual values of the retained parents must sustain the actual output:

$$
f _ { i } ( C _ { i } ( u ) , n _ { i } , u ) = V _ { i } ( u ) , \qquad \forall n _ { i } \in \mathrm { V a l } ( N _ { i } ) .\tag{14}
$$

Next, select $W _ { i } \subseteq N _ { i }$ and an assignment $w _ { i }$ . Let $N _ { i , W _ { i }  w _ { i } } ( u )$ denote the values attained by the remaining parents $N _ { i }$ in the original model under the intervention $W _ { i }  w _ { i }$ and context u. The projected mechanism is

$$
f _ { i } ^ { u } ( c _ { i } ) = f _ { i } ( c _ { i } , N _ { i , W _ { i }  w _ { i } } ( u ) , u ) .\tag{15}
$$

The projected mechanism must remain nontrivially responsive to the retained inputs: there exists $c _ { i } ^ { \prime } \in \mathrm { V a l } ( C _ { i } )$ such that $\mathsf { \bar { f } } _ { i } ^ { u } ( c _ { i } ^ { \prime } ) \neq V _ { i } ( u )$ . Thus, the projection cannot simply replace the original mechanism with a constantfunction that always returns the actual output.

Selections across different mechanisms must additionally be compatible. In particular, $N _ { i } \cap C _ { j } \ : = \ : \emptyset \ : f o r \ : i \ : \neq \ : j ,$ and whenever intervention assignments overlap across mechanisms, they must assign the same value to every shared intervened variable. The compatible projected mechanisms, together with the retained assignment rules for root variables, constitute the causal beam $\bar { M } _ { u } \stackrel { \cdot } { = } ( u , V , \mathcal { F } ^ { u } )$

The sustaining condition requires the actual values $C _ { i } ( u )$ to preserve the actual mechanism output under arbitrary assignments to the remaining parents $N _ { i }$ . The responsiveness condition serves a different role: it requires the projected mechanism to preserve a nontrivial dependence on its retained inputs. These two requirements distinguish a causal projection from a representation that merely records the observed output.

A special case is obtained when no non-actual contingency is introduced.

Definition A.2 (Natural Beam [Pearl, 2009]). A causal beam is called natural when $W _ { i } = \emptyset f o r$ every projected mechanism. In this case, the remaining parents are fixed at their actual values under context u, and the projected mechanism reduces to $f _ { i } ^ { u } ( c _ { i } ) = f _ { i } ( c _ { i } , \bar { N _ { i } } ( u ) , u )$

A variable that is fixed in a particular projection is not thereby declared irrelevant in general. The projection is defined relative to the specified model and context, and a different context may admit a different set of retained parents and hence a different projected mechanism.

## A.2 An Illustrative Construction of a Causal Beam

We illustrate the construction using a binary structural causal model with three root variables $B , G , O$ and two nonroot variables $D , Y$ , with mechanisms $D = B \land G$ and $Y = D \land O$ . The parent sets are therefore $\mathrm { P A } _ { D } = \{ B , G \}$ and $\mathrm { P A } _ { Y } = \{ D , O \}$ .

We first consider a context $u _ { 1 }$ with $B = 1 , G = 1 , O = 1$ , which implies $D = 1 , Y = 1$ . For the mechanism $D = B \land G ,$ retaining only B does not sustain the actual output because setting $G = 0$ changes D from 1 to 0. Likewise, retaining only G fails because setting $B = 0$ changes the output. A valid choice is therefore $C _ { D } = \{ B , G \}$ and $N _ { D } = \emptyset$ . The corresponding projection is $f _ { D } ^ { u _ { 1 } } ( b , g ) = b \wedge g .$ . It preserves the actual output at $b = g = 1$ and remains nontrivially responsive because, for example, $f _ { D } ^ { \bar { u _ { 1 } } } ( 0 , 1 ) \stackrel {  } { = } 0 \stackrel {  } { \neq } 1 = D ( u _ { 1 } )$

Table 5: Datasets and evaluation scope. The RoboCasa entry refers to the 24-task protocol used by Cosmos Policy, rather than the complete RoboCasa task collection.
<table><tr><td>Dataset</td><td>Role in CBST</td><td>Scope / WAM</td></tr><tr><td>SSv2</td><td>Human-video representation pretraining</td><td>174 action categories; planner initialization</td></tr><tr><td>RoboTwin 2.0</td><td>Robot adaptation and evaluation</td><td>50 tasks; LingBot-VA</td></tr><tr><td>LIBERO-Long</td><td>Robot adaptation and evaluation</td><td>10 tasks; LingBot-VA and Cosmos Policy (Pre- dict2.5)</td></tr><tr><td>RoboCasa</td><td>Robot adaptation and evaluation</td><td>24 tasks; Cosmos Policy (Predict2.5)</td></tr></table>

Table 6: Official SSv2 split sizes. These are release-level counts; the number of clips retained after preprocessing is reported separately in Table 14.
<table><tr><td>Split</td><td>Videos</td><td>Annotation availability</td></tr><tr><td>Training</td><td>168,913</td><td>Action labels</td></tr><tr><td>Validation</td><td>24,777</td><td>Action labels</td></tr><tr><td>Test</td><td>27,157</td><td>Labels withheld</td></tr><tr><td>Total</td><td>220,847</td><td>174 action categories</td></tr></table>

For $Y = D \land O$ , the same reasoning gives $C _ { Y } = \{ D , O \}$ and $N _ { Y } = \varnothing$ , with $f _ { Y } ^ { u _ { 1 } } ( d , o ) = d \wedge o .$ . Again, the actual output is preserved, while changing either retained input appropriately can change the projected output. Because both excluded parent sets are empty, the cross-mechanism compatibility condition is satisfied. The resulting natural beam therefore retains the original mechanisms for this context.

Now consider a second context $u _ { 2 }$ with $B = 1 , G = 0 , O = 1$ , which gives $D = 0 , Y = 0$ . For $D = B \land G .$ , choose $C _ { D } = \{ G \}$ and $N _ { D } = \{ B \}$ . The sustaining condition holds because $f _ { D } ( b , 0 ) = b \wedge 0 = 0$ for all $b \in \{ 0 , 1 \}$ . Using a natural projection fixes B at its actual value $B ( u _ { 2 } ) = 1$ , yielding $f _ { D } ^ { u _ { 2 } } ( g ) = f _ { D } ( 1 , g ) = g$ . The projection remains nontrivially responsive since $f _ { D } ^ { u _ { 2 } } ( 1 ) = 1 \ne 0 = \dot { D } ( \dot { u } _ { 2 } )$

For $Y = D { \land } O$ , choose $C _ { Y } = \{ D \}$ and $N _ { Y } = \{ O \}$ . The sustaining condition is satisfied because $f _ { Y } ( 0 , o ) = 0 \land o = 0$ for all $o \in \{ 0 , 1 \}$ . Fixing O at its actual value $O ( u _ { 2 } ) = 1$ gives $\mathsf { \bar { f } } _ { Y } ^ { u _ { 2 } } ( d ) = f _ { Y } ( d , 1 ) = d ,$ , which is also nontrivially responsive because $f _ { Y } ^ { u _ { 2 } } ( { \bar { 1 } } ) = 1 \not = 0 = Y ( u _ { 2 } )$

The cross-mechanism compatibility conditions are satisfied because $N _ { D } \cap C _ { Y } = \{ B \} \cap \{ D \} = \emptyset$ and $N _ { Y } \cap C _ { D } =$ $\{ O \} \cap \{ G \} = \emptyset$ . No non-actual intervention assignments are introduced, so there are no additional assignment consistency constraints to check. The resulting natural beam contains the projected mechanisms $D = G$ and ${ \check { Y } } = D$

The two contexts therefore yield different valid projections of the same underlying structural causal model. In $u _ { 1 }$ sustaining the actual successful outputs requires retaining both parents of each mechanism, whereas in $u _ { 2 } .$ , the actual zero values allow smaller retained parent sets. The construction is context specific: fixing B and O in the second projection does not imply that these variables are generally irrelevant, only that their actual values can be treated as fixed within that particular projection.

## B Experimental Details

## B.1 Datasets and Benchmark Composition

We distinguish the data used to pretrain the planner from the robot demonstrations used for adaptation and the environments used for closed-loop evaluation. Table 5 summarizes their roles.

## B.1.1 Something-Something V2

SSv2 contains short videos of everyday human–object interactions [Goyal et al., 2017]. Its 174 labels describe actions and their outcomes using object-independent templates. The official release contains 220,847 videos, divided into 168,913 training videos, 24,777 validation videos, and 27,157 test videos. Test labels are not publicly provided.

Table 7: Representative action distinctions in SSv2. Descriptions are abbreviated semantic summaries rather than verbatim labels or an exhaustive class list.
<table><tr><td>Interaction type</td><td>Representative distinctions</td></tr><tr><td>Object displacement</td><td>Directional movement; motion toward or away from the camera.</td></tr><tr><td>Placement and removal</td><td>Moving objects onto surfaces, into receptacles, or away from their previous locations.</td></tr><tr><td>Lifting and release</td><td>Partial versus complete lifting; holding versus releasing.</td></tr><tr><td>Contact and motion</td><td>Contact that moves an object versus contact with little observable displacement.</td></tr><tr><td>Support and stability</td><td>Tilting a support with or without causing the supported object to fall.</td></tr><tr><td>Attachment</td><td>Successful attachment versus an unsuccessful attempt.</td></tr><tr><td>Deformation</td><td>Bending, breaking, folding, unfolding, and squeezing.</td></tr><tr><td>Visibility</td><td>Covering or uncovering an object.</td></tr><tr><td>Object interaction</td><td>Collisions with different resulting motions.</td></tr><tr><td>Action completion</td><td>Actual execution versus pretending or attempting without corresponding effect.</td></tr></table>

Table 8: The 50 RoboTwin tasks grouped by the horizon labels used in the LingBot-VA report. Names are abbreviated from the official task identifiers.
<table><tr><td>Horizon</td><td>Tasks Task names</td><td></td></tr><tr><td>1</td><td>30</td><td>Adjust Bottle; Beat Block Hammer; Click Alarmclock; Click Bell; Dump Bin Bigbin; Grab Roller; Lift Pot; Move Can Pot; Move Pillbottle Pad; Move Playingcard Away; Move Stapler Pad; Open Laptop; Open Microwave; Place A2B Left; Place A2B Right; Place Bread Basket; Place Container Plate; Place Empty Cup; Place Fan; Place Mouse Pad; Place Object Scale; Place Object Stand; Place Phone Stand; Place Shoe; Press Stapler; Rotate QRcode; Shake Bottle Horizontally; Shake</td></tr><tr><td>2</td><td>15</td><td>Bottle; Stamp Seal; Turn Switch. Handover Block; Handover Mic; Hanging Mug; Pick Diverse Bottles; Pick Dual Bottles; Place Bread Skillet; Place Burger Fries; Place Can Basket; Place Cans Plasticbox; Place Dual Shoes; Place Object Basket; Put Object Cabinet; Scan Object; Stack Blocks Two; Stack Bowls Two.</td></tr><tr><td>3</td><td>5</td><td>Blocks Ranking RGB; Blocks Ranking Size; Put Bottles Dustbin; Stack Blocks Three; Stack Bowls Three.</td></tr></table>

Action coverage. The label space includes object displacement, placement, containment, deformation, and interactions with different observable outcomes. Table 7 summarizes representative distinctions relevant to manipulation. The groups are descriptive summaries, not an alternative official taxonomy. Training retains the original 174-class classification target.

Use in CBST. Human pretraining optimizes action classification. The visual encoder, temporal module, and representation projection initialize the robot-side planner; robot prediction heads and structural selectors are introduced during adaptation. Human category labels are not treated as robot action labels or as direct supervision for causal parent sets.

For the temporal-order control, sampled frames are shuffled within each clip while preserving the clip label, sampled frame content, and training budget. The no-human-pretraining variant uses the same robot adaptation procedure without SSv2 pretraining. Its visual initialization is specified separately to distinguish absence of human-video pretraining from random initialization.

## B.1.2 RoboTwin 2.0

RoboTwin 2.0 provides simulated bimanual manipulation tasks [Chen et al., 2025]. We use the 50-task evaluation set reported for LingBot-VA. The tasks cover object transport, handover, placement, stacking, tool interaction, and articulated-object manipulation. Easy and Hard results are reported separately.

Table 8 lists the complete task set, grouped by the horizon annotations in the LingBot-VA evaluation table [Li et al., 2026]. These annotations describe task groups; they are distinct from the action-chunk length and the simulator episode limit.

Evaluation settings. Easy and Hard follow the configurations associated with the reported LingBot-VA results. Hard introduces additional variation in object poses and scene configurations. We retain the benchmark-defined success

Table 9: LIBERO suite composition. LIBERO-90 and LIBERO-Long are the two partitions of LIBERO-100, not additional copies of that collection.
<table><tr><td>Suite</td><td>Tasks</td><td>Main variation</td></tr><tr><td>Spatial</td><td>10</td><td>Spatial relations identifying a target object.</td></tr><tr><td>Object</td><td>10</td><td>Object identity under a shared placement goal.</td></tr><tr><td>Goal</td><td>10</td><td>Desired outcomes within related environments.</td></tr><tr><td>LIBERO-90</td><td>90</td><td>Diverse manipulation tasks.</td></tr><tr><td>Long / LIBERO-10</td><td>10</td><td>Tasks requiring extended execution.</td></tr></table>

Table 10: Task composition of the Spatial, Object, and Goal suites. Shared instructions are factored out for compactness.
<table><tr><td>Suite</td><td>Task variants</td></tr><tr><td>Spatial</td><td>Move the black bowl to the plate. The bowl starts: between plate and ramekin; beside ramekin; at table center; on cookie box; inside top drawer; on ramekin; beside cookie box; on stove; beside plate; on cabinet.</td></tr><tr><td>Object</td><td>Place each target in the basket: alphabet soup; cream cheese; salad dressing; barbecue sauce; ketchup; tomato sauce; butter; milk; chocolate pudding; orange juice.</td></tr><tr><td>Goal</td><td>Open middle drawer; bowl to stove; wine bottle to cabinet top; bowl inside top drawer; bowl to cabinet top; plate toward stove front; cream cheese into bowl; activate stove; bowl to plate; wine bottle to rack.</td></tr></table>

checks. Overall performance averages the task success rates over all 50 tasks; horizon-specific results use only the tasks in the corresponding group.

The robot embodiment, camera configuration, demonstration budget, and episode limits used for CBST are specified in Tables 14 and 15.

## B.1.3 LIBERO

LIBERO evaluates knowledge transfer in language-conditioned robot manipulation [Liu et al., 2023]. Its released collections include Spatial, Object, Goal, and LIBERO-100. LIBERO-100 is partitioned into LIBERO-90 and the 10-task LIBERO-Long set, also registered as libero 10.

We evaluate CBST on all four LIBERO suites—Spatial, Object, Goal, and Long—with both WAM backbones. Each suite contains ten tasks. The main text focuses on LIBERO-Long, while Appendix C.2 reports results on all four suites. Table 9 distinguishes these evaluation suites from the broader data collection.

Spatial, Object, and Goal. Table 10 summarizes the task variants in the three short-horizon suites. LIBERO provides scene and language specifications together with visual observations and robot proprioception. The released demonstrations include workspace and wrist views.

Long-horizon tasks. Table 11 lists all ten tasks used for CBST evaluation. Task identifiers and success predicates follow the official benchmark; the descriptions below are shortened for readability.

Data and evaluation. Robot demonstrations are partitioned by trajectory before observation windows are constructed. Planner inputs use observations available up to the current decision time, together with robot proprioception and task information. Simulator state may provide adaptation labels, but is not an input to the learned planner during evaluation.

We retain the official task success predicates. Success rates are averaged over the ten Long tasks. Scores for the complete four-suite benchmark, when quoted from prior work, retain the aggregation reported by that source.

Table 11: The ten LIBERO-Long tasks. LR, K, and S denote living-room, kitchen, and study scenes, respectively.
<table><tr><td>ID</td><td>Scene</td><td>Goal summary</td></tr><tr><td>L1</td><td>LR2</td><td>Basket: alphabet soup and tomato sauce.</td></tr><tr><td>L2</td><td>LR2</td><td>Basket: cream cheese and butter.</td></tr><tr><td>L3</td><td>K3</td><td>Activate stove; place moka pot on it.</td></tr><tr><td>L4</td><td>K4</td><td>Black bowl inside bottom drawer; drawer closed.</td></tr><tr><td>L5</td><td>LR5</td><td>White mug on left plate; yellow-and-white mug on right plate.</td></tr><tr><td>L6</td><td>S1</td><td>Book in caddy&#x27;s rear compartment.</td></tr><tr><td>L7</td><td>LR6</td><td>White mug on plate; pudding to its right.</td></tr><tr><td>L8</td><td>LR1</td><td>Basket: alphabet soup and cream cheese.</td></tr><tr><td>L9</td><td>K8</td><td>Both moka pots on stove.</td></tr><tr><td>L10</td><td>K6</td><td>Yellow-and-white mug inside microwave; door closed.</td></tr></table>

Table 12: The 24 RoboCasa tasks used in the Cosmos Policy evaluation protocol. Task identifiers follow its environment release. PnP denotes pick-and-place.
<table><tr><td>Family</td><td>Count</td><td>Tasks</td></tr><tr><td>Pick-and-place</td><td>8</td><td>PnPCounterToCab; PnPCabToCounter; PnPCounterToSink; PnPSinkToCounter; Pn- PCounterToMicrowave; PnPMicrowaveToCounter; PnPCounterToStove; PnPStove- ToCounter.</td></tr><tr><td>Cabinet doors</td><td>4</td><td>OpenSingleDoor; CloseSingleDoor; OpenDoubleDoor; CloseDoubleDoor.</td></tr><tr><td>Drawers</td><td>2</td><td>OpenDrawer; CloseDrawer.</td></tr><tr><td>Sink</td><td>3</td><td>TurnOnSinkFaucet; TurnOffSinkFaucet; TurnSinkSpout.</td></tr><tr><td>Stove</td><td>2</td><td>TurnOnStove; TurnOffStove.</td></tr><tr><td>Coffee</td><td>3</td><td>CoffeeSetupMug; CoffeeServeMug; CoffeePressButton.</td></tr><tr><td>Microwave</td><td>2</td><td>TurnOnMicrowave; TurnOffMicrowave.</td></tr></table>

## B.1.4 RoboCasa

RoboCasa provides kitchen environments for learning everyday manipulation skills [Nasiriany et al., 2024]. We use the 24-task protocol adopted by Cosmos Policy [Kim et al., 2026]. This evaluation set contains seven task families, listed in Table 12; it excludes kitchen navigation.

Task coverage. Pick-and-place tasks vary the source and destination fixture. Door and drawer tasks require changing articulated states. Sink, stove, and microwave tasks involve fixture controls, whereas coffee tasks cover mug positioning, retrieval, and button operation.

Reference protocol. The Cosmos Policy report uses 50 human-teleoperated demonstrations per task. Its evaluation includes five scenes per task, with ten trials per scene and three evaluation seeds, giving 3,600 trials over the 24 tasks. Evaluation uses unseen object instances, and two of the five scenes introduce styles absent from training.

These quantities describe the published reference protocol. CBST’s adaptation data and rollout counts are specified separately below. Demonstration budgets remain explicit in the result tables because the published baselines use different amounts of data.

## B.2 Baseline Details

We include two types of comparison methods: the WAM backbones to which CBST is attached, and external robot manipulation methods reported on the same benchmarks. Only LingBot-VA and Cosmos Policy (Predict2.5) serve as CBST backbones.

LingBot-VA. LingBot-VA jointly generates visual futures and robot actions using an autoregressive video–action model [Li et al., 2026]. We use it as the underlying WAM for RoboTwin and LIBERO-Long.

Table 13: Roles of the principal comparison methods. R, L, and C denote RoboTwin, LIBERO, and RoboCasa. Benchmark entries indicate the comparisons included in this paper, not the complete capabilities of each method.
<table><tr><td>Method</td><td>Model / interface</td><td>Role</td><td>Benchmark</td></tr><tr><td>LingBot-VA</td><td>Video-action generation</td><td>Backbone</td><td>R, L</td></tr><tr><td>Cosmos Policy-Predict2.5</td><td>Joint action and future prediction</td><td>Backbone</td><td>L, C</td></tr><tr><td>Cosmos Policy-Predict2</td><td>Earlier video-model configuration</td><td>Reference</td><td>L, C</td></tr><tr><td>X-VLA</td><td>Soft-prompted VLA</td><td>Reference</td><td>R, L</td></tr><tr><td>π0</td><td>Flow-based VLA</td><td>Reference</td><td>R, L, C</td></tr><tr><td>π0.5</td><td>Generalist VLA</td><td>Reference</td><td>R, L</td></tr><tr><td>Motus</td><td>Unified latent-action world model</td><td>Reference</td><td>R</td></tr><tr><td>OpenVLA-OFT</td><td>Adapted VLA with action chunking</td><td>Reference</td><td>L</td></tr><tr><td>GR00T-N1.5</td><td>Vision-language and diffusion action model</td><td>Reference</td><td>CC</td></tr><tr><td>Video Policy</td><td>Coupled video and action generation</td><td>Reference</td><td></td></tr></table>

Cosmos Policy. Cosmos Policy adapts a pretrained video model to generate actions together with future-state and value predictions [Kim et al., 2026]. CBST uses the Predict2.5 configuration. The earlier Predict2 configuration is retained as an additional published reference, with its results identified separately [Kim and Gu, 2026].

X-VLA. X-VLA uses a flow-based action model and soft prompts to accommodate heterogeneous embodiments and data sources [Zheng et al., 2025].

π and $\pi _ { 0 . 5 } . \quad \pi _ { 0 }$ combines pretrained vision–language representations with flow-based action generation [Black et al., 2026]. $\pi _ { 0 . 5 }$ extends this line with heterogeneous training sources and semantic supervision for broader generalization [Intelligence et al., 2025].

Motus. Motus combines understanding, video generation, and action modeling within a unified latent-action framework [Bi et al., 2025].

OpenVLA-OFT. OpenVLA-OFT applies an optimized adaptation recipe to OpenVLA, including parallel action decoding, action chunking, and continuous action prediction [Kim et al., 2025b].

GR00T-N1.5. GR00T-N1.5 combines vision–language encoding with a diffusion action module and updates the model and training recipe of GR00T-N1 [NVIDIA et al., 2025].

Video Policy. Video Policy couples video generation with action generation, using visual prediction to support robot policy learning [Liang et al., 2025].

Sources of reported results. RoboTwin comparisons follow the LingBot-VA report. LIBERO comparisons combine the published LingBot-VA and Cosmos Policy tables. RoboCasa comparisons follow Cosmos Policy, with Predict2.5 results obtained from the official Cosmos Cookbook.

We preserve the reported model version, demonstration budget, evaluation setting, and uncertainty notation. When two sources report different results for the same method, the corresponding source is identified explicitly; entries from different sources are not combined into a single synthetic result.

## B.3 Implementation Details

Visual-temporal encoder. The planner uses a ResNet18 visual encoder [He et al., 2015] and a GRU temporal module [Cho et al., 2014]. Available camera views share the visual encoder, and their features are averaged at each sampled timestep. The resulting feature is combined with robot proprioception and task features before temporal aggregation. Observation histories combine sparse earlier observations with denser recent observations.

Human pretraining and robot adaptation. Human pretraining uses action-category classification. The pretrained encoder and representation projection initialize the robot planner, while robot-specific semantic heads and parent-set selectors are newly initialized.

Table 14: Planner architecture and training configuration.
<table><tr><td>Item</td><td>Configuration</td></tr><tr><td>Visual encoder</td><td>ResNet18</td></tr><tr><td>Temporal encoder</td><td>GRU</td></tr><tr><td>Camera-feature fusion</td><td>Mean pooling over available camera views</td></tr><tr><td>Human pretraining</td><td>174-class action classification on SSv2 using cross-entropy loss</td></tr><tr><td>Transferred components</td><td>Visual encoder, temporal module, and representation projection</td></tr><tr><td>Robot prediction heads</td><td>Semantic states, interval variables, and candidate parent subsets Observed-prefix inputs with sparse earlier observations and</td></tr><tr><td>Observation history</td><td>dense recent observations</td></tr><tr><td>Robot adaptation</td><td>Update the final visual stage, temporal module, representation projection, semantic heads, and parent-set selectors</td></tr><tr><td>Frozen components</td><td>Earlier visual stages, reference mechanisms, WAM, and native text encoder</td></tr><tr><td>Conditioning interface</td><td>Deterministic text rendering followed by the WAM&#x27;s native text encoder</td></tr><tr><td>Robot data split</td><td>Split by demonstration trajectory before constructing observa- tion windows</td></tr><tr><td>Missing supervision</td><td>Mask unknown labels in the corresponding losses</td></tr></table>

During robot adaptation, the final visual stage, temporal module, representation projection, semantic heads, and selectors are optimized using the objectives defined in the method section. The earlier visual stages, reference mechanisms, WAM, and native text encoder remain frozen. Trajectory splits are constructed before sampling observation windows.

Semantic and structural supervision. Robot demonstration replay provides labels for semantic states, interval variables, and realized effects. Unknown labels are masked rather than assigned negative values. Candidate dependencies are instantiated from the task specification and shared manipulation templates.

For structural supervision, parent subsets are evaluated within the declared finite variable domains using frozen reference mechanisms. Multiple acceptable subsets are retained when they satisfy the structural conditions. Examples lacking the required labels or variable domains are excluded from the corresponding structural loss.

Conditioning and execution. A deterministic renderer preserves the original task instruction and appends predicted task information and selected dependency fields. The WAM’s native text encoder processes the resulting prompt. The WAM retains its original observation encoder, action representation, and generation procedure.

The planner updates the condition from newly available observations before the next action request. Evaluation does not provide the learned planner with future observations or simulator ground-truth semantic states. Any oracle condition is identified separately.

## B.4 Evaluation Metrics and Protocols

Task success. Let $s _ { k , e } \in \{ 0 , 1 \}$ denote the official success indicator for episode e of task $k ,$ with $N _ { k }$ evaluated episodes. The task success rate is

$$
\mathrm { S R } _ { k } = \frac { 1 0 0 } { N _ { k } } \sum _ { e = 1 } ^ { N _ { k } } { s _ { k , e } } .\tag{16}
$$

The aggregate score over $K$ tasks is

$$
\mathrm { S R } _ { \mathrm { a v g } } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathrm { S R } _ { k } .\tag{17}
$$

When every task has the same number of episodes, this equals the success rate pooled over all episodes.

RoboTwin Easy and Hard are aggregated separately. LIBERO-Long averages over ten tasks, and the RoboCasa protocol averages over 24 tasks. Published baseline values retain the aggregation and uncertainty conventions of their respective sources.

Table 15: Evaluation scope and episode budgets. RoboTwin counts apply separately to Easy and Hard. LIBERO-Long and RoboCasa counts aggregate evaluation seeds. Each WAM is evaluated independently.
<table><tr><td></td><td>RoboTwin 2.0</td><td colspan="2">LIBERO-Long</td><td>RoboCasa</td></tr><tr><td>Item</td><td>LingBot-VA</td><td>LingBot-VA</td><td>Predict2.5</td><td>Predict2.5</td></tr><tr><td>Tasks</td><td>50</td><td>10</td><td>10</td><td>24</td></tr><tr><td>Episodes per task</td><td>100</td><td>150</td><td>150</td><td>150</td></tr><tr><td>Total episodes</td><td>5,000</td><td>1,500</td><td>1,500</td><td>3,600</td></tr></table>

Table 16: RoboTwin success rates (%) grouped by task horizon. Each cell reports Easy / Hard. Values follow Table 1 of [Li et al., 2026]; X-VLA<sup>∗</sup> is adopted there from Motus.
<table><tr><td>Method</td><td>Horizon 1</td><td>Horizon 2</td><td>Horizon 3</td><td>All 50 tasks</td></tr><tr><td>X-VLA*</td><td>81.6 / 82.5</td><td>59.3 / 55.9</td><td>61.2 / 66.0</td><td>72.9 / 72.8</td></tr><tr><td>π0</td><td>66.5 / 61.6</td><td>66.1 / 54.7</td><td>61.6 / 50.2</td><td>65.9 / 58.4</td></tr><tr><td> $\pi _ { 0 . 5 }$ </td><td>85.1 / 80.2</td><td>79.3 / 73.0</td><td>78.6 / 67.4</td><td>82.7 / 76.8</td></tr><tr><td>Motus</td><td>91.0 / 90.6</td><td>85.2 / 80.9</td><td>85.0 / 84.2</td><td>88.7 / 87.0</td></tr><tr><td>LingBot-VA</td><td>94.18 / 93.56</td><td>90.35 / 86.95</td><td>93.22 / 93.28</td><td>92.93 / 91.55</td></tr><tr><td> $\mathbf { L i n g B o t - V A + C B S T }$ </td><td>94.73 / 93.97</td><td>91.93 / 88.80</td><td>95.80 / 95.20</td><td>94.00 / 92.54</td></tr></table>

Evaluation configurations. Table 15 records the protocol for each CBST evaluation. Action-chunk length, executed actions per request, and episode limits are specified separately because they determine different aspects of closed-loop execution.

Training and evaluation randomness. Training seeds and evaluation seeds are recorded separately. Repeated roll out seeds assess variation in evaluation, whereas independently trained planners assess variation in learning. The meaning of any reported error bar is specified in the corresponding table or figure caption.

## C Additional Simulation Results

This section expands the benchmark comparisons in Section 5.2. Published values retain their original precision and aggregation.

## C.1 RoboTwin 2.0

Table 16 reports the horizon-grouped results [Li et al., 2026]. Easy and Hard results remain separate. Source-reported aggregate values are retained without recomputing them from the rounded per-task entries.

## C.2 LIBERO

Table 17 combines the four-suite comparisons reported by LingBot-VA and Cosmos Policy, together with the official Predict2.5 results. Identical entries are consolidated, while source-dependent differences are preserved as separate rows. In particular, the reported $\pi _ { 0 }$ averages and UniVLA results differ between the two papers.CBST results are reported for Spatial, Object, Goal, and Long with both WAM backbones. For CBST, the Average column is the arithmetic mean of the four suite success rates; source-reported baseline averages are preserved.

## C.3 RoboCasa

Table 18 includes all methods in the RoboCasa comparison reported by Cosmos Policy, together with its Predict2.5 update. Training demonstrations per task are retained because the methods use different data budgets. The CBST entry reports the robot data used by the underlying policy and the additional planner, with overlapping demonstrations counted only once.

Table 17: Complete published LIBERO comparison. All values are success rates (%). L: Table 2 of [Li et al., 2026]; C: Table 1 of [Kim et al., 2026]; K: official Cosmos Cookbook [Kim and Gu, 2026]. Source-reported averages and uncertainty values are preserved.
<table><tr><td>Method</td><td>Source</td><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td><td>Average</td></tr><tr><td>Octo</td><td>L</td><td>78.9</td><td>85.7</td><td>84.6</td><td>51.1</td><td>75.1</td></tr><tr><td>Seer</td><td>L</td><td>一</td><td>一</td><td></td><td>87.7</td><td></td></tr><tr><td>MoDE</td><td>L</td><td>一</td><td>一</td><td>一</td><td>94.0</td><td>一</td></tr><tr><td>SuSIE</td><td>L</td><td></td><td></td><td></td><td>76.3</td><td></td></tr><tr><td>SpatialVLA</td><td>L</td><td>88.2</td><td>89.9</td><td>78.6</td><td>55.5</td><td>78.1</td></tr><tr><td>TraceVLA</td><td>L</td><td>84.6</td><td>85.2</td><td>75.1</td><td>54.1</td><td>74.8</td></tr><tr><td>CoT-VLA</td><td>L</td><td>87.5</td><td>91.6</td><td>87.6</td><td>69.0</td><td>81.1</td></tr><tr><td>ThinkAct</td><td>L</td><td>88.3</td><td>91.4</td><td>87.1</td><td>70.9</td><td>84.4</td></tr><tr><td>SmolVLA</td><td>L</td><td>93.0</td><td>94.0</td><td>91.0</td><td>77.0</td><td>88.8</td></tr><tr><td>CronusVLA</td><td>L</td><td>97.3</td><td>99.6</td><td>96.9</td><td>94.0</td><td>97.0</td></tr><tr><td>FLOWER</td><td>L</td><td>97.1</td><td>96.7</td><td>95.6</td><td>93.5</td><td>95.7</td></tr><tr><td>GR00T-N1</td><td>L</td><td>94.4</td><td>97.6</td><td>93.0</td><td>90.6</td><td>93.9</td></tr><tr><td>π0</td><td>L</td><td>96.8</td><td>98.8</td><td>95.8</td><td>85.2</td><td>94.1</td></tr><tr><td>π0</td><td>C</td><td>96.8</td><td>98.8</td><td>95.8</td><td>85.2</td><td>94.2</td></tr><tr><td>π0+FAST</td><td>L</td><td>96.4</td><td>96.8</td><td>88.6</td><td>60.2</td><td>85.5</td></tr><tr><td>π0.5</td><td>C</td><td>98.8</td><td>98.2</td><td>98.0</td><td>92.4</td><td>96.9</td></tr><tr><td>OpenVLA</td><td>L</td><td>84.7</td><td>88.4</td><td>79.2</td><td>53.7</td><td>76.5</td></tr><tr><td>OpenVLA-OFT</td><td>L, C</td><td>97.6</td><td>98.4</td><td>97.9</td><td>94.5</td><td>97.1</td></tr><tr><td>DD-VLA</td><td>L</td><td>97.2</td><td>98.6</td><td>97.4</td><td>92.0</td><td>96.3</td></tr><tr><td>UniVLA</td><td>L</td><td>95.4</td><td>98.8</td><td>93.6</td><td>94.0</td><td>95.4</td></tr><tr><td>UniVLA</td><td>C</td><td>96.5</td><td>96.8</td><td>95.6</td><td>92.0</td><td>95.2</td></tr><tr><td>X-VLA</td><td>L</td><td>98.2</td><td>98.6</td><td>97.8</td><td>97.6</td><td>98.1</td></tr><tr><td>Diffusion Policy</td><td>C</td><td>78.3</td><td>92.5</td><td>68.3</td><td>50.5</td><td>72.4</td></tr><tr><td>Dita</td><td>C</td><td>97.4</td><td>94.8</td><td>93.2</td><td>83.6</td><td>92.3</td></tr><tr><td>UVA</td><td>C</td><td>一</td><td></td><td></td><td>90.0</td><td></td></tr><tr><td>Video Policy</td><td>C</td><td></td><td></td><td></td><td>94.0</td><td></td></tr><tr><td>CogVLA</td><td>C</td><td>98.6</td><td>98.8</td><td>96.6</td><td>95.4</td><td>97.4</td></tr><tr><td>LingBot-VA</td><td>L</td><td>98.5 ± 0.3</td><td>99.6 ± 0.3</td><td>97.2 ± 0.2</td><td>98.5 ± 0.5</td><td>98.5</td></tr><tr><td>Cosmos Policy (Predict2)</td><td>C</td><td>98.1</td><td>100.0</td><td>98.2</td><td>97.6</td><td>98.5</td></tr><tr><td>Cosmos Policy (Predict2.5)</td><td>K</td><td>98.7</td><td>99.7</td><td>98.3</td><td>96.6</td><td>98.33</td></tr><tr><td>LingBot-VA + CBST</td><td>Ours</td><td>99.0</td><td>99.6</td><td>97.73</td><td>98.60</td><td>98.73</td></tr><tr><td>Predict2.5 + CBST</td><td>Ours</td><td>99.27</td><td>99.8</td><td>98.73</td><td>97.33</td><td>98.78</td></tr></table>

## C.4 Additional Qualitative Examples

Figures 7 provide three additional simulation task sequences. Each example contains temporally ordered frames from a recorded execution. Task names and recording sources are specified in the captions. Annotations describe observable object interactions and state changes; model predictions are included only when corresponding outputs are available.

## D Additional Ablation Studies

This section specifies the ablation variants and extends the analysis in Section 5.3. We distinguish changes to pretraining, robot adaptation objectives, and inference-time conditioning. This distinction identifies which comparisons require independently trained planners and which can be evaluated using a shared checkpoint.

## D.1 Variant Definitions

Common configuration. Unless explicitly changed, variants use the same robot demonstration split, observation inputs, candidate variable domains, reference mechanisms, renderer, and frozen WAM checkpoint. Retrained variants use the same robot adaptation budget and checkpoint-selection rule. Inference-only variants share the full CBST checkpoint and differ solely in the information rendered for the WAM.

Table 18: Complete RoboCasa comparison over 24 tasks. Baseline scores and demonstration counts follow [Kim et al., 2026]. Predict2.5 results follow the official Cosmos Cookbook [Kim and Gu, 2026]. Synthetic data are identified separately.
<table><tr><td>Method</td><td>Demos per task</td><td>Average SR (%)</td></tr><tr><td>GR00T-N1</td><td>300</td><td>49.6</td></tr><tr><td>UVA</td><td>50</td><td>50.0</td></tr><tr><td>DP-VLA</td><td>3000</td><td>57.3</td></tr><tr><td>GR00T-N1 + DreamGen</td><td>300 (+10000 synthetic)</td><td>57.6</td></tr><tr><td>GR00T-N1 + DUST</td><td>300</td><td>58.5</td></tr><tr><td>UWM</td><td>1000</td><td>60.8</td></tr><tr><td>π0</td><td>300</td><td>62.5</td></tr><tr><td>GR00T-N1.5</td><td>300</td><td>64.1</td></tr><tr><td>Video Policy</td><td>300</td><td>66.0</td></tr><tr><td>FLARE</td><td>300</td><td>66.4</td></tr><tr><td>GR00T-N1.5 + HAMLET</td><td>300</td><td>66.4</td></tr><tr><td>Cosmos Policy (Predict2)</td><td>50</td><td>67.1</td></tr><tr><td>Cosmos Policy (Predict2.5)</td><td>50</td><td>71.1</td></tr><tr><td>Predict2.5 + CBST</td><td>50</td><td>72.44</td></tr></table>

(a) RoboTwin: Stack three bowls  
![](images/1124d8118c9aa6e7442ced2d73cc80e757a8b2a70af9a238d1831c0310aa2d1f.jpg)  
Figure 7: Additional recorded LingBot-VA executions:(a) stacking three bowls in RoboTwin, (b) handing over a block in RoboTwin, and (c) placing a black bowl in a drawer and closing it in LIBERO. Each row shows six temporally ordered frames;timestamps indicate video playback time.

Without human-video pretraining. This variant removes the human-video classification stage. The visual encoder retains the common initialization specified in Appendix B.3, and components normally initialized through human pretraining use their corresponding untrained initialization. Robot heads and selectors follow the same initialization procedure as full CBST. The comparison holds the robot adaptation budget fixed, rather than equating the total compute spent across human pretraining and adaptation.

Table 19: Definitions of the principal ablation variants. “Retrain” refers to robot-side planner adaptation; the WAM remains frozen in every variant.
<table><tr><td>Variant</td><td>Retrain</td><td>Change relative to CBST</td></tr><tr><td>Without human pretraining</td><td>Yes</td><td>Omit SSv2 pretraining while retaining the common visual initialization and robot adaptation procedure.</td></tr><tr><td>Shuffled human pretraining</td><td>Yes</td><td>Shuffle sampled frames within each human clip before pretraining; retain labels, frame content, and pretraining budget.</td></tr><tr><td>State-only</td><td>No</td><td>Render admissible goals and predicted semantic states without the se- lected interaction-dependency fields.</td></tr><tr><td>Full structure</td><td>No</td><td>Render all candidate parent fields instead of the contextually selected subset.</td></tr><tr><td>Random selection</td><td>No</td><td>Render randomly selected candidate parent fields, matching the learned subset cardinality.</td></tr><tr><td>Without mechanism consis- tency</td><td>Yes</td><td>Set  $\lambda _ { m } = 0$  while preserving the remaining supervision and reference mechanisms.</td></tr></table>

Shuffled human-video pretraining. We randomly permute the sampled frame order within each human clip while keeping its original action label. The sampled images, class distribution, architecture, optimization settings, and number of pretraining updates are otherwise unchanged. Robot adaptation subsequently uses correctly ordered robot observations. This control evaluates the contribution of temporal ordering during human pretraining.

State-only conditioning. Using the full CBST checkpoint, we retain the original task instruction, admissible goals, and predicted semantic-state fields. Fields describing the selected parent dependencies and associated intervalinteraction variables are omitted from the rendered condition. The underlying state predictions remain unchanged, so this comparison isolates the information presented to the WAM rather than differences in state-estimation training.

Full-structure conditioning. For every mechanism, we replace the selected parent set with the complete candidate parent set, $C _ { i , t } = \mathrm { P A } _ { i }$ . Parent values are obtained from the same predictions as in CBST. Variable naming, field ordering, update frequency, and text encoding remain unchanged. This variant evaluates whether selecting a contextspecific subset is useful beyond supplying all available structural fields.

Random selection. For mechanism i at time $t ,$ let $k _ { i , t }$ be the number of parents retained by CBST. We sample a subset uniformly from the candidate subsets of $\mathrm { P A } _ { i }$ having cardinality $k _ { i , t }$ . The sampled subset uses the same predicted variable values and renderer as CBST. Sampling does not use ground-truth task outcomes or reference-validity labels. Random-selection seeds are fixed before evaluation and reported separately from training and rollout seeds.

Without mechanism consistency. We remove ${ \mathcal { L } } _ { \mathrm { m e c h } }$ from robot adaptation by setting $\lambda _ { m } = 0$ . Semantic-state supervision, interval-variable supervision, and beam supervision remain unchanged. The reference mechanisms are still used to construct acceptable parent sets; the ablation therefore removes the mechanism-consistency training signal without removing the structural supervision source.

## D.2 Results across Backbones and Benchmarks

Table 20 extends the principal ablations to WAM backbones and associated benchmarks. RoboTwin Easy and Hard are reported separately. Published backbone scores are included as references and are not treated as paired measurement from the ablation runs.

## D.3 Temporal Architecture

We compare GRU, Transformer, temporal convolutional network (TCN), and MLP temporal modules while retaining the same visual encoder, observation samples, robot supervision, and WAM interface. The MLP receives the sampled frame features in their fixed temporal order through concatenation; the other modules process the corresponding feature

Table 20: Principal ablation results across benchmarks. All values are success rates (%). RoboTwin and LingBot-VA reference values follow [Li et al., 2026]; Predict2.5 reference values follow [Kim and Gu, 2026].
<table><tr><td></td><td colspan="3">LingBot-VA</td></tr><tr><td>Variant</td><td>RoboTwin Easy</td><td>RoboTwin Hard</td><td>LIBERO-Long</td></tr><tr><td>Backbone</td><td>92.93</td><td>91.55</td><td>98.5 ± 0.5</td></tr><tr><td>CBST</td><td>94.00</td><td>92.54</td><td>98.60</td></tr><tr><td>Without human pretraining</td><td>92.80</td><td>91.38</td><td>98.00</td></tr><tr><td>Shuffled human pretraining</td><td>92.94</td><td>91.54</td><td>98.47</td></tr><tr><td>State-only</td><td>92.82</td><td>91.36</td><td>98.13</td></tr><tr><td>Full structure</td><td>93.42</td><td>91.96</td><td>98.53</td></tr><tr><td>Random selection</td><td>92.56</td><td>91.08</td><td>96.60</td></tr><tr><td>Without mechanism consistency</td><td>93.18</td><td>91.70</td><td>98.53</td></tr></table>

Table 21: Temporal architecture comparisons on LIBERO-Long. Parameter counts refer to the planner; latency refers to a planner update with the same input specification and accelerator.
<table><tr><td>Temporal module</td><td>Parameters</td><td>LingBot-VA SR</td><td>Predict2.5 SR</td><td>Planner latency</td></tr><tr><td>MLP</td><td>12.0</td><td>98.53</td><td>97.13</td><td>9</td></tr><tr><td>TCN</td><td>12.3</td><td>98.53</td><td>97.20</td><td>11</td></tr><tr><td>GRU</td><td>12.5</td><td>98.60</td><td>97.33</td><td>12</td></tr><tr><td>Transformer</td><td>13.2</td><td>98.66</td><td>97.53</td><td>13</td></tr></table>

sequence. Architecture-specific widths and depths, parameter counts, and optimization settings are reported with the results.

The comparison uses the same data and training schedule. Parameter counts are reported explicitly rather than assuming that the architectures have identical capacity.

## D.4 Training and Validation Curves

We report human-pretraining and robot-adaptation curves separately. Human-pretraining plots include classification loss and validation accuracy for correctly ordered and shuffled clips. Robot-adaptation plots separate semantic-state, interval-variable, mechanism-consistency, and structural-selection objectives, together with the corresponding heldout diagnostics. Curves use the original logged values. Training steps and epochs are identified explicitly, and resumed training segments retain their boundaries. Duplicate logging streams are not counted as additional optimization steps. Model selection follows the rule specified in Appendix B.3.

## D.5 Sensitivity to Observation History

We vary observation history along two axes. First, we change the number of sampled frames while holding the lookback duration fixed, testing sampling density. Second, we change the lookback duration while holding the number of sampled frames fixed, testing temporal coverage. The current observation is retained in every setting. Other planner and WAM settings remain unchanged. Each sampling configuration is used consistently during adaptation and evaluation. The tested values and the default configuration are specified in Table 22.

## D.6 Sensitivity to Structural Compactness

We vary the size-regularization coefficient $\lambda _ { \mathrm { s i z e } }$ while retaining the same semantic supervision, reference mechanisms, and acceptable-set construction. A separate planner is trained for each coefficient using the same adaptation budget. Setting $\bar { \lambda _ { \mathrm { s i z e } } } = 0$ removes the explicit compactness preference while preserving the structural objectives. For each setting, we report control success, the mean retained-parent fraction, and the number of text tokens delivered to the WAM. Token counts are measured using each backbone’s native tokenizer. Structural compactness and prompt length are reported separately because equal parent counts need not produce equal token counts.

Table 22: Sensitivity-analysis configurations. Default values correspond to the full CBST configuration.
<table><tr><td>Factor</td><td>values</td><td>Reference Fixed quantities</td><td></td></tr><tr><td>Sampled frame count</td><td>{4, 8, 16, 32}</td><td>16</td><td>Lookback duration and temporal sampling strategy.</td></tr><tr><td>Lookback duration (s)</td><td>{1, 2, 4, 8}</td><td>4</td><td>Frame count and recent/history allocation ra- tio.</td></tr><tr><td> $\lambda _ { \mathrm { s i z e } }$ </td><td> $\{ 0 , 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , 1 0 ^ { - 1 } \}$ </td><td> $1 0 ^ { - 2 }$ </td><td>Other loss weights, reference mechanisms, and training budget.</td></tr></table>

Table 23: Task inventory.
<table><tr><td>ID</td><td>Task</td><td>Placement objective and success condition</td></tr><tr><td>T1</td><td>Remote-control placement</td><td>Move the remote from the tabletop into the designated open container; it must remain fully inside after gripper release.</td></tr><tr><td>T2</td><td>Cup placement</td><td>Move the cup to the designated tabletop region and leave it stable after release; handover may be used to connect the source and target workspaces.</td></tr><tr><td>T3</td><td>Plate racking</td><td>Place the plate into the designated rack slot, with the rack supporting the plate after both grippers disengage.</td></tr><tr><td>T4</td><td>Apple placement</td><td>Place an apple inside the designated fruit basket and release it.</td></tr><tr><td>T5</td><td>Banana placement</td><td>Place a banana within the designated tray and release it.</td></tr><tr><td>T6</td><td>Orange placement</td><td>Place an orange inside the designated bowl and release it.</td></tr><tr><td>T7</td><td>Bottle placement</td><td>Place a bottle upright within the designated tabletop region and release it.</td></tr><tr><td>T8</td><td>Sponge placement</td><td>Place a sponge inside the designated storage box and release it.</td></tr><tr><td>T9</td><td>Spoon placement</td><td>Place a spoon within the designated cutlery tray and release it.</td></tr><tr><td>T10</td><td>Block placement</td><td>Place a block inside the designated box and release it.</td></tr></table>

## D.7 Computational Cost

We measure the cost of the additional planner and conditioning path separately from WAM generation. Measurements use a fixed accelerator, input specification, batch size, and action-generation configuration for each backbone. Warmup iterations are excluded, and accelerator operations are synchronized at the declared timing boundaries.

Planner time includes visual-temporal encoding, semantic prediction, and parent-set selection. Conditioning time separates text rendering from native text encoding. End-to-end request latency is measured directly rather than inferred by summing components that may overlap. Simulator stepping and video recording are excluded from policy-request timing.

## E Real-World Experimental Details

## E.1 Platform and Task Formulation

The platform consists of two AgileX manipulators with parallel-jaw grippers operating over a shared tabletop. Each task is specified by a manipulated object and a fixed target region or receptacle. The controller must acquire the object, transport it, and establish the required final placement. When a single arm cannot reach both the source and destination, the arms coordinate through their overlapping reachable workspace. No particular handover trajectory is prescribed by the task. The supplied videos show a fixed external recording view.

## E.2 Task Definitions

Table 23 describes the ten task entries. All names specify placement outcomes rather than a required handover.Target locations are fixed within a task configuration; any changes across configurations must be documented separately from variation in initial object pose. The detailed results in Table 24.

Table 24: Detailed real-world results. Each method has 20 trials per task. Gains are measured in percentage points.
<table><tr><td></td><td></td><td colspan="2">Cosmos Policy (Predict2.5)</td><td colspan="2">+CBST</td><td></td></tr><tr><td>ID</td><td>Task</td><td>Successes</td><td>SR (%)</td><td>Successes</td><td>SR (%)</td><td>Gain</td></tr><tr><td>T1</td><td>Remote</td><td>(17/20)</td><td>85</td><td>(18/20)</td><td>90</td><td>5</td></tr><tr><td>T2</td><td>Cup</td><td>(14/20)</td><td>70</td><td>(17/20)</td><td>85</td><td>15</td></tr><tr><td>T3</td><td>Plate</td><td>(13/20)</td><td>65</td><td>(15/20)</td><td>75</td><td>10</td></tr><tr><td>T4</td><td>Apple</td><td>(10/20)</td><td>50</td><td>(12/20)</td><td>60</td><td>10</td></tr><tr><td>T5</td><td>Banana</td><td>(15/20)</td><td>75</td><td>(16/20)</td><td>80</td><td>5</td></tr><tr><td>T6</td><td>Orange</td><td>(8/20)</td><td>40</td><td>(9/20)</td><td>45</td><td>5</td></tr><tr><td>T7</td><td>Bottle</td><td>(17/20)</td><td>85</td><td>(17/20)</td><td>85</td><td>0</td></tr><tr><td>T8</td><td>Sponge</td><td>(16/20)</td><td>80</td><td>(17/20)</td><td>85</td><td>5</td></tr><tr><td>T9</td><td>Spoon</td><td>(11/20)</td><td>55</td><td>(14/20)</td><td>70</td><td>15</td></tr><tr><td>T10</td><td>Block</td><td>(16/20)</td><td>80</td><td>(18/20)</td><td>90</td><td>10</td></tr></table>

## E.3 Baseline and Evaluation Protocol

The comparison pairs Cosmos Policy with a Predict2.5 backbone and its CBST-augmented counterpart. The intended controlled comparison holds the task-adapted low-level policy checkpoint, robot observations, control settings, and evaluation conditions fixed, while adding CBST’s contextual structural conditioning. The official Cosmos Policy ALOHA experiments provide a related tabletop-manipulation reference [Kim et al., 2026]; those published real-world results use Predict2, and the official Cookbook states that Predict2.5 was not trained on ALOHA [Kim and Gu, 2026]. Those results are therefore not substituted for evaluation on our AgileX tasks.

Each method is evaluated for 20 trials on each of ten tasks. This specifies the number of executions, not the number of distinct layouts or random seeds. Initial object poses, target geometry, and the mapping of initializations between methods must be documented to establish comparability. For the final protocol, the same initialization set should be used for both methods, with scene resets between executions and balanced method order.

Success criteria. Full-task success requires the designated object to occupy its specified target region or receptacle, satisfy the task-specific placement requirement, and remain stable after gripper release within the evaluation time limit. Handover is neither an additional task objective nor an independently rewarded outcome. Grasping or transferring an object without completing its final placement does not constitute success. The handling of autonomous retries, intermediate drops, and human interventions must follow the same documented rule for both methods

Metrics. Let $S _ { m , k }$ denote the number of successful trials for method m on task k. With 20 trials per task, the task success rate and ten-task average are

$$
\mathrm { S R } _ { m , k } = 1 0 0 { \frac { S _ { m , k } } { 2 0 } } , \qquad \overline { { \mathrm { S R } } } _ { m } = { \frac { 1 } { 1 0 } } \sum _ { k = 1 } ^ { 1 0 } \mathrm { S R } _ { m , k } = 1 0 0 { \frac { \sum _ { k = 1 } ^ { 1 0 } S _ { m , k } } { 2 0 0 } } .\tag{18}
$$

Single-task rates therefore change in increments of 5 percentage points, while the ten-task average changes in increments of 0.5 percentage points. Improvements are reported as percentage-point differences. A mean computed over the three visualized tasks is explicitly labeled as a subset mean and is not used as the ten-task result. Stage-level annotations, if available, distinguish acquisition, transport, handover where applicable, and final placement; they are diagnostic and do not replace the full-task metric.

## E.4 Qualitative Executions

The three supplied recordings are identified by the authors as CBST executions. Each visualization contains six original video frames selected around visible manipulation events. In the cup-placement sequence shown in the main text, the initiating arm lifts the cup and transfers it to the receiving arm for final tabletop placement. Figure 8 shows plate transport into a rack and remote-control placement into a container. In the plate sequence, grasp transfer enables transport between arm workspaces; in the remote sequence, one arm completes the placement while the other remains inactive. These examples illustrate execution strategies and do not measure their frequency across the ten-task suite.

The figure labels describe visible events rather than predicted states or causal dependencies. All selected frames preserve the source image content. The encoded timestamps and frame rate are used only to identify frames, not to infer control frequency or execution latency. Selected successful videos provide qualitative examples and are not used to estimate task success rates.

(b) Plate racking  
![](images/49e98a672396b84bd5fba2ca3fa44f33a5b55c733bf02a0621bb9b13bae1bf60.jpg)  
Figure 8: Additional real-world executions on the AgileX platform: (b) plate racking and (c) remote-control placement. Frames are shown in temporal order, with timestamps indicating video playback time. Annotations describe observed manipulation events.