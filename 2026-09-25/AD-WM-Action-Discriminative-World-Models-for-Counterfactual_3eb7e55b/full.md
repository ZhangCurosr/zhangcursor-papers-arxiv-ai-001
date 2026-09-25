# AD-WM: Action-Discriminative World Models for Counterfactual Model Predictive Control

Jiabin Qiu<sup>∗</sup>, Zixuan Chen<sup>∗,†</sup>, Hongye Cao,

Jieqi Shi, Jing Huo<sup>†</sup>, Yang Gao

Nanjing University

<sup>∗</sup>Equal contribution. <sup>†</sup>Corresponding authors.

Abstract— Latent world models are typically trained to predict factual transitions, whereas model predictive control (MPC) must compare alternative actions from the same state. A model can therefore achieve low factual prediction error yet poorly distinguish candidate actions. We introduce AD-WM, an action-discriminative joint-embedding world model for counterfactual MPC. AD-WM combines residual latent dynamics with predictor-level action-recovery regularization, using inverse dynamics and a normalized recovery objective motivated by conditional mutual information. Both objectives encourage planning transitions to preserve action information; their auxiliary heads are discarded at test time, leaving MPC unchanged. On OGBench-Cube, AD-WM improves hard-start success from 3.7% to 52.0% over a matched LeWM baseline and improves mean success over the reproduced baseline in four of five simulation environments. Planning diagnostics show that factual prediction error and whole-bank action ranking do not follow the closed-loop success ordering, whereas CEMaligned elite regret tracks success more closely. With a frozen V-JEPA 2 encoder and matched DROID post-training, AD-WM also improves zero-shot transfer to our Franka setup, increasing basic pick-and-place success from 42.2% to 71.1% without lab-specific adaptation. These results suggest that world models for planning should preserve action-dependent differences needed for counterfactual selection, rather than optimize factual prediction accuracy alone. More videos and code are available at https://ad-wm.github.io/.

## I. INTRODUCTION

World models enable agents to predict the consequences of candidate actions before executing them. For visionbased control, methods such as PlaNet, Dreamer, and TD-MPC learn latent dynamics from visual observations and use predicted trajectories to guide action selection [1]–[6]. More recently, joint-embedding predictive architectures (JEPAs) have enabled future representation prediction without pixel reconstruction [7], [8]. Building on this approach, DINO-WM, PLDM, LeWorldModel (LeWM), and V-JEPA 2-AC use predicted features to plan toward visual goals [9]–[12]. These methods evaluate candidate action sequences through latent rollouts and select actions whose predicted outcomes approach the goal.

Accurate prediction of recorded transitions, however, does not guarantee useful comparisons between alternative actions. Training supervises the outcome of the action actually taken, whereas planning compares what would happen under different actions from the same state. Fig. 1 illustrates this distinction through a cube-grasping example. When visual representations are dominated by persistent scene content, a predictor can achieve low error by largely preserving the current representation. Yet its predictions may obscure the smaller action-dependent changes that distinguish approaching the cube from moving away. Such predictions fit observed transitions but provide limited guidance for goaldirected action selection.

![](images/47443aea1564f2db910ba8ad67e2eebfeb2bcf785041fe06135b12a117805394.jpg)  
Fig. 1: Action discrimination for planning. Low prediction error can mask differences between candidate actions. AD-WM preserves action-dependent changes to guide goaldirected selection.

This motivates a key requirement for planning: latent dynamics should preserve the action-dependent differences needed for counterfactual comparison. Factual supervision anchors predictions to observed outcomes but does not explicitly enforce action recoverability. Recovering the action from the current and predicted next representations provides a complementary training signal. Applied to model-generated transitions, this constraint directly shapes the dynamics that MPC uses to compare candidate actions. We therefore focus on predictor-level action discrimination.

We propose AD-WM, which combines residual latent prediction with predictor-level action recovery for counterfactual MPC. The residual predictor estimates latent increments relative to the current state. Action recovery uses two related objectives: inverse dynamics and normalized recovery motivated by conditional mutual information. By default, both operate on the current and predicted next representations. In simulation, the encoder and predictor are trained jointly. At deployment, the auxiliary heads are discarded,

leaving MPC unchanged.

We also examine whether the learned dynamics support useful candidate selection. The cross-entropy method (CEM) retains a small set of high-scoring action sequences, called the elite set, to guide subsequent search [13]. A model can rank most candidates correctly while misjudging those retained for further search. We therefore introduce planning-aligned diagnostics that assess selected elites using environment-realized outcomes on shared candidate sequences. In controlled Cube experiments, elite regret is more closely associated with closed-loop success than factual prediction error or whole-bank ranking. Across five simulation environments, AD-WM improves mean success over matched LeWM in four environments. It also improves zeroshot transfer to a real robot without lab-specific adaptation.

Our contributions are threefold:

• We identify a mismatch between factual prediction accuracy and counterfactual action comparison in JEPAbased MPC. Our elite-selection diagnostics better reflect closed-loop success in controlled Cube experiments.

• We introduce AD-WM, which combines residual latent prediction with predictor-level action recovery to preserve action information without modifying the MPC planner.

• We evaluate AD-WM across five simulation environments and a real Franka robot. Baseline comparisons, controlled ablations, and planning diagnostics establish its control benefits and examine the factors supporting transfer.

## II. RELATED WORK

Latent World Models for Control. Learned dynamics support model-based control by predicting the consequences of actions before execution [14], [15]. Existing approaches use these predictions for latent-space planning, behavior learning through imagined rollouts, or visual foresight [1], [2], [5], [16], [17]. More recent work explores Transformer and diffusion architectures for world modeling [18], [19]. Robot world models also investigate action-conditioned prediction and counterfactual behavior [20]. These approaches differ in how their predictions guide control. Our work focuses on reward-free, image-goal MPC learned from offline data. In this setting, predicted latent transitions directly determine how the planner compares candidate action sequences. We therefore study whether these transitions preserve the action-dependent differences needed for selection.

Feature-predictive world models. Joint-embedding predictive architectures (JEPAs) predict future representations without explicit observation reconstruction [7], [8]. DINO-WM, PLDM, LeWM, and V-JEPA 2 use learned or pretrained visual features for goal-directed planning [9]–[12]. LeWM is our closest end-to-end simulation baseline, while V-JEPA 2-AC provides pretrained dynamics for our realrobot study. JEPA’s emphasis on slowly varying features [21] raises the question of whether predictions adequately capture smaller action-dependent changes needed for control. Recent extensions address complementary limitations of JEPAbased control. Fast-LeWM improves rollout efficiency, while Sub-JEPA regularizes representations for stable end-to-end learning [22], [23]. INTACT and Qantara introduce actiongeneration and control interfaces beyond standard CEM planning [24], [25]. Delta-JEPA promotes action sensitivity by recovering actions from observed latent displacements [26]. Our normalized recovery regularizes predicted transitions, as does the default inverse objective. Combined with residual prediction, these objectives shape latent dynamics for counterfactual action comparison without modifying the MPC planner.

Action-aware representation learning. Inverse dynamics provides a training signal for learning action-relevant representations from observed state transitions [27], [28]. Related work studies Markov state abstractions and action-sufficient representations for control [29], [30]. These approaches motivate retaining information that is useful for predicting and influencing environment dynamics. We apply this principle directly at the predictor level. Our inverse dynamics and normalized action-recovery objectives operate on the current and predicted next representations. Their gradients therefore constrain the model-generated transitions that MPC rolls out and compares during planning. This complements factual prediction supervision with an explicit objective for preserving action information in predicted transitions.

## III. METHOD

Fig. 2 illustrates the training and planning pipeline of AD-WM. During training, residual latent prediction and action-recovery objectives jointly shape action-discriminative transitions. At test time, MPC uses the learned dynamics to compare candidate action sequences, with the auxiliary heads discarded. We first formulate the task, then detail the model, training objectives, and planning procedure. Finally, we introduce diagnostics for evaluating candidate selection.

## A. Problem Formulation

Given an offline dataset $\mathcal { D } = \{ ( o _ { t } , a _ { t } , o _ { t + 1 } ) \}$ of images and continuous actions, we learn an encoder ${ \boldsymbol { z } } _ { t } ~ = ~ e _ { \phi } ( { \boldsymbol { o } } _ { t } )$ and latent dynamics $\hat { z } _ { t + 1 } ~ = ~ F _ { \theta } ( z _ { t } , a _ { t } )$ . At test time, the agent receives a current image and a goal image $^ { g , }$ encoded as $z _ { g } = e _ { \phi } ( g )$ . MPC evaluates candidate action sequences $a \ = \ a _ { t : t + H - 1 }$ using the predicted terminal cost ${ \hat { c } } ( a ) \ =$ $\lVert \hat { z } _ { t + H } ( z _ { t } , a ) - z _ { g } \rVert _ { 2 } ^ { 2 }$ . It executes the first action block and replans without parameter updates. Low factual prediction error alone does not ensure useful action comparison. For transitions $z _ { t + 1 } = z _ { t } + \delta _ { t }$ with small increments, an actionindependent predictor $F _ { 0 } ( z , a ) ~ = ~ z$ incurs only $\mathbb { E } \Vert \delta _ { t } \Vert _ { 2 } ^ { 2 }$ expected error. Yet it assigns all action sequences the same terminal cost. We therefore seek latent dynamics that preserve action-dependent differences needed for planning.

## B. Residual Latent Prediction

Successive visual representations often share substantial state information. We let the current representation carry this shared information and train the predictor to estimate the

![](images/af79e85475c344396ab92aab40d5ca9aff3f96197a6130951a1aaa010409c72d.jpg)  
Fig. 2: Training and planning with AD-WM. Left: residual dynamics predicts increments supervised by encoded successors. Inv and MI encourage action recovery from predicted transitions. Simulation jointly trains the encoder and predictor with SIGReg. Robot post-training freezes the encoder and omits SIGReg. Right: CEM scores rollouts by terminal cost $\| \hat { z } _ { t + H } - z _ { g } \| _ { 2 } ^ { 2 }$ , executes the first action block, and replans. Auxiliary heads are discarded at deployment.

remaining change. Given a learned action embedding $e _ { t } =$ $\psi _ { \rho } ( a _ { t } )$ , AD-WM predicts:

$$
\Delta \hat { z } _ { t } = f _ { \theta } ( z _ { t } , e _ { t } ) , \qquad \hat { z } _ { t + 1 } = z _ { t } + \Delta \hat { z } _ { t } .\tag{1}
$$

The prediction is supervised by the encoded successor:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { p r e d } } = \| \hat { z } _ { t + 1 } - z _ { t + 1 } \| _ { 2 } ^ { 2 } , \qquad z _ { t + 1 } = e _ { \phi } \big ( o _ { t + 1 } \big ) . } \end{array}\tag{2}
$$

This equivalently matches $\Delta \hat { z } _ { t }$ to the encoded displacement $z _ { t + 1 } - z _ { t }$ . Absolute and residual prediction share the same optimum in an unconstrained function class. The residual parameterization changes the learning bias by explicitly modeling local latent changes. However, it does not itself require these changes to retain action information. We address this through predictor-level action recovery.

## C. Predictor-Level Action Recovery

We encourage actions to be recoverable from the current and predicted next representations. Applying recovery to model-generated transitions directly shapes the dynamics used by MPC. Prediction supervision anchors these transitions to observed outcomes, while recovery discourages the predictor from ignoring action information.

Inverse dynamics. The default inverse head recovers the action embedding from predicted transition endpoints:

$$
\hat { e } _ { t } = g _ { \omega } ( z _ { t } , \hat { z } _ { t + 1 } ) , \qquad \mathcal { L } _ { \mathrm { i n v } } = \| \hat { e } _ { t } - \mathrm { s g } ( e _ { t } ) \| _ { 2 } ^ { 2 } .\tag{3}
$$

The operator $\mathrm { s g }$ detaches the recovery target. The predicted endpoint remains differentiable, allowing auxiliary gradients to train the predictor and action encoder $\psi _ { \rho }$ through $\hat { z } _ { t + 1 }$ Thus, recovery constrains generated transitions rather than only observed state representations. Section IV-C examines alternative inverse inputs.

Normalized action recovery. Inv operates on learned action embeddings at their current scale. Normalized recovery uses standardized targets and a Gaussian recovery objective motivated by conditional mutual information (MI). We normalize embeddings componentwise as $\bar { e } _ { t } = \mathrm { s g } [ ( e _ { t } -$ $\mu _ { B } ) / \operatorname* { m a x } ( \sigma _ { B } , 1 0 ^ { - 6 } ) ]$ , where $\mu _ { B }$ and $\sigma _ { B }$ are the batch mean and standard deviation. This controls target scale for likelihood training with fixed unit covariance.

A separate head receives $[ z _ { t } , \hat { z } _ { t + 1 } ]$ and defines $q _ { \eta } ( \cdot \ |$ $z _ { t } , \hat { z } _ { t + 1 } ) = \mathcal { N } ( \mu _ { \eta } , I )$ . Using a standard Gaussian as a fixed regularization reference, we minimize:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { M I } } = \ - \log q _ { \eta } ( \bar { e } _ { t } \ \vert \ z _ { t } , \hat { z } _ { t + 1 } ) } \\ & { \quad \quad \ + \ \beta D _ { \mathrm { K L } } ( q _ { \eta } ( \cdot \ \vert \ z _ { t } , \hat { z } _ { t + 1 } ) \| \mathcal { N } ( 0 , I ) ) . } \end{array}\tag{4}
$$

With unit covariance, this reduces, up to an additive constant, to:

$$
\begin{array} { r } { \frac { 1 } { 2 } \| \bar { \boldsymbol { e } } _ { t } - \mu _ { \eta } \| _ { 2 } ^ { 2 } + \frac { \beta } { 2 } \| \mu _ { \eta } \| _ { 2 } ^ { 2 } . } \end{array}
$$

The first term recovers standardized action embeddings. The second shrinks the predicted mean toward zero. For fixed state/action representations and target normalization, the recovery log-likelihood is the variational term in a Barber– Agakov lower bound on $I ( \bar { e } _ { t } ; \hat { z } _ { t + 1 } \mid z _ { t } ) [ 3 1 ]$ . This motivates the recovery term, with additional KL regularization during joint training.

## D. Training and Planning

In simulation, the encoder and predictor are trained jointly with:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { p r e d } } + \lambda _ { \mathrm { s i g } } \mathcal { L } _ { \mathrm { s i g } } + \lambda _ { \mathrm { i n v } } \mathcal { L } _ { \mathrm { i n v } } + \lambda _ { \mathrm { M I } } \mathcal { L } _ { \mathrm { M I } } .\tag{5}
$$

Prediction supervision matches observed outcomes, while the recovery objectives encourage predicted transitions to retain action information. We use the same SIGReg representation regularizer as LeWM [11]. Robot post-training freezes the encoder and omits SIGReg (Section IV-E).

At deployment, both auxiliary heads are discarded. MPC embeds candidate actions with $\psi _ { \rho }$ and recursively applies Eq. (1). CEM retains the sequences with lowest predicted terminal cost and refits its sampling distribution to this elite set. After search, the agent executes the first action block and replans from the new observation. The encoder architecture and CEM procedure remain unchanged.

## E. Planning-Facing Diagnostics

To assess whether predicted transitions support useful action selection, we evaluate global ranking and elite quality on a shared candidate bank $\mathcal { A } ~ = ~ \{ a ^ { ( \bar { i } ) } \} _ { i = 1 } ^ { N }$ . For each sequence, cˆ(a) is its predicted terminal cost. Its realized cost $c ^ { \star } ( a )$ is obtained by executing the sequence from the same initial environment state and encoding the final image. These realized costs are used only for simulation diagnostics, outside training and MPC. Each model uses its own encoder for both costs, so the metrics assess alignment within its planning space.

Global ranking. Counterfactual action discriminability (CAD) measures agreement across the full candidate bank:

$$
\mathrm { C A D } ( z _ { t } ) = \rho _ { S } \left( \{ \hat { c } ( a ^ { ( i ) } ) \} _ { i = 1 } ^ { N } , \{ c ^ { \star } ( a ^ { ( i ) } ) \} _ { i = 1 } ^ { N } \right) ,\tag{6}
$$

where $\rho _ { S }$ is Spearman rank correlation. This includes candidates outside the elite set.

Elite quality. Let $E _ { k } ( c )$ denote the k lowest-cost candidates under c. Define the realized cost range $\begin{array} { r l } { D _ { \mathbf { \mathcal { A } } } } & { { } = } \end{array}$ $\begin{array} { r } { \operatorname* { m a x } _ { a \in \mathcal { A } } c ^ { \star } ( a ) - \operatorname* { m i n } _ { a \in \mathcal { A } } c ^ { \star } ( a ) } \end{array}$ , floored at $1 0 ^ { - 8 }$ . Normalized best-in-elite regret is:

$$
R _ { k } = \frac { \operatorname* { m i n } _ { a \in E _ { k } ( \hat { c } ) } c ^ { \star } ( a ) - \operatorname* { m i n } _ { a \in \mathcal { A } } c ^ { \star } ( a ) } { D _ { \mathcal { A } } } .\tag{7}
$$

Low $R _ { k }$ indicates that the predicted elite set retains at least one near-best candidate. Since CEM refits its proposal using all elites, we also define elite-mean regret:

$$
\bar { R } _ { k } = \frac { \sum _ { a \in E _ { k } ( \hat { c } ) } { c ^ { \star } ( a ) } - \sum _ { a \in E _ { k } ( c ^ { \star } ) } { c ^ { \star } ( a ) } } { k D _ { A } } .\tag{8}
$$

This compares the mean realized cost of predicted elites with that of the realized top-k set. Both regrets normalize selection penalties by the bank’s realized cost range. They characterize candidate selection, while closed-loop success remains the final control outcome.

![](images/63ca451663451d7afe2869fc19f8647e3c2169b6a62c4744b9ae365831e38a0f.jpg)  
Fig. 3: The simulation environments.

## IV. EXPERIMENTS

We organize our experiments around four questions: Q1: How well does AD-WM perform across simulation tasks and configuration shifts? Q2: How do residual prediction and action recovery contribute to control performance? Q3: Which prediction and selection metrics best reflect closedloop success? Q4: Can AD-WM transfer to a real robot without lab-specific adaptation?

## A. Experimental Setup

Benchmarks and Baselines. We follow LeWM’s simulation environment selection [11], evaluating on Cube, Reacher, TwoRoom, and PushT, and additionally include Scene from OGBench [32]. As shown in Fig. 3, these environments cover robotic manipulation, reaching, navigation, and planar pushing. Cube serves as the primary benchmark for controlled ablations and planning diagnostics. We retain the original evaluation protocols for crossenvironment comparisons, except for Scene, and introduce additional hard-start protocols on Cube. Our main baseline is a matched LeWM reproduction [11]. We also evaluate released Cube checkpoints from Fast-LeWM [22], Sub-JEPA [23], and INTACT [24], retaining their native inference interfaces. Controlled ablations share images, encoder size, training budget, and MPC settings, varying only transition parameterization and auxiliary training choices.

Evaluation protocols. Cube includes Original, which samples dataset states, and hard-start protocols P00–P04. Hard starts select tabletop states without gripper contact, with minimum goal distance 0.05 and end-effector–cube distance 0.02. P00–P04 apply cube xy perturbations of radii $0 . 0 0 / 0 . 0 1 / 0 . 0 2 / 0 . 0 3 / 0 . 0 4 _ { \mathrm { : } }$ clipped to $x \in [ 0 . 3 0 , 0 . 5 5 ]$ and $y \in [ - 0 . 3 0 , 0 . 3 0 ]$ . P00 therefore differs from Original even without perturbation. Each protocol uses 50 episodes per checkpoint with evaluation seed 42. Hard-start success (HS) averages P00–P04. Unless stated otherwise, Cube results are means and population standard deviations (SDs) over seeds 3072, 4096, and 6144. Reacher, TwoRoom, and PushT use original protocols. Scene uses 200 hard-start episodes per checkpoint, balanced across Button, Cube, Drawer, and Window. Starts differ from goals in one component, with changedetection tolerance 0.04 and minimum target displacement and end-effector–target distance 0.08 for geometric filters.

TABLE I: Cube success (%) on matched starts. Fast-LeWM and Sub-JEPA use one checkpoint. Others report mean±population SD over three. External methods retain native inference. Bold marks the highest mean.
<table><tr><td>Method</td><td>Inference</td><td>Original</td><td>P0O</td><td>P01</td><td>P02</td><td>P03</td><td>P04</td></tr><tr><td>LeWM</td><td>CEM</td><td>73.3±2.5</td><td>8.0±2.8</td><td>4.7±0.9</td><td>4.7±1.9</td><td>1.3±1.9</td><td>0.0±0.0</td></tr><tr><td>Fast-LeWM [22]</td><td>CEM</td><td>80.0</td><td>24.0</td><td>20.0</td><td>10.0</td><td>2.0</td><td>0.0</td></tr><tr><td>Sub-JEPA [23]</td><td>CEM</td><td>78.0</td><td>34.0</td><td>20.0</td><td>10.0</td><td>10.0</td><td>10.0</td></tr><tr><td>INTACT [24]</td><td>Pure CEM</td><td>74.7±0.9</td><td>20.0±3.3</td><td>21.3±9.3</td><td>13.3±2.5</td><td>6.7±1.9</td><td>3.3±2.5</td></tr><tr><td>INTACT</td><td>Actor-CEM</td><td>94.7±2.5</td><td>78.7±3.4</td><td>72.0±5.7</td><td>56.0±4.3</td><td>33.3±12.4</td><td>10.7±2.5</td></tr><tr><td>INTACT</td><td>Direct</td><td>98.7±1.9</td><td>99.3±0.9</td><td> $\mathbf { 9 9 . 3 } \pm \mathbf { 0 . 9 }$ </td><td>88.7±1.9</td><td>35.3±3.4</td><td>10.0±0.0</td></tr><tr><td>AD-WM</td><td>CEM</td><td>90.7±3.4</td><td>74.0±2.8</td><td>68.0±4.3</td><td>56.0±1.6</td><td>36.7±5.2</td><td>25.3±3.4</td></tr></table>

We report success as percentages and factual latent mean squared error (MSE) at one step and averaged over the first five rollout steps (local MSE). Counterfactual action discriminability (CAD) measures whole-bank ranking agreement; $R _ { 3 0 }$ and $\bar { R } _ { 3 0 }$ denote normalized best-in-elite and elite-mean regret for 30 selected candidates (Section III-E). We use $\rho$ for Spearman rank correlation.

Training details. Our simulation setup follows LeWM [11], using 224 × 224 images, training/evaluation histories of 3/1, and frame skip 5. The trainable ViT-tiny encoder [33] has latent dimension 192. The predictor has depth 6, 16 heads, and MLP dimension 2048. Inv and MI heads use 1024-hidden-unit MLPs with BatchNorm. Training uses bf16, 10 epochs, AdamW, batch size 128, learning rate $5 \times 1 0 ^ { - 5 }$ , weight decay $1 0 ^ { - 3 } .$ , and $\lambda _ { \mathrm { s i g } } = 0 . 0 9 .$ All simulation environments use $\lambda _ { \mathrm { i n v } } = 0 . 1 , \lambda _ { \mathrm { M I } } = 0 . 0 1$ and $\beta = 0 . 0 1$ , except Scene, where $\lambda _ { \mathrm { M I } } = 1 0 ^ { - 4 }$ . Primary weights were specified before sensitivity analysis.

Planning details. Matched comparisons use CEM [13] with 300 candidates, 30 elites, 30 iterations, horizon 5, receding horizon 5, and action block 5. Actions are clipped to environment bounds and scored by terminal latent distance. Cube and Scene use goal offset 25 and a 50-step budget. Sub-JEPA and INTACT CEM modes use the same candidate count, elite count, iterations, horizon, and action block. Fast-LeWM uses horizon 1 and action block 25. INTACT Direct performs no search. External Cube comparisons share starts, episode counts, goal offset, and budget, but do not control training and inference as the LeWM-AD-WM pair does.

## B. Simulation Results

Main results. To answer Q1, Table I compares methods on identical Cube starts. AD-WM outperforms matched LeWM across all protocols. INTACT Direct performs best from Original through P02, while its Actor-CEM mode improves over Pure CEM through P03. On P03, INTACT Direct, Actor-CEM, and AD-WM have similar means relative to checkpoint variation. On P04, AD-WM reaches 25.3%, compared with 10.0% and 10.7% for Direct and Actor-CEM. These results indicate stronger performance under larger perturbations, but differences in training and inference prevent attributing the crossover to search strategy alone.

Configuration shift. P00–P01 serve as near-distribution controls, while P02–P04 probe configuration tails within empirical cube xy support. We quantify shift using Euclidean nearest-neighbor distances to training states from other episodes. Joint coordinates concatenate cube xy and end-effector xyz, while relative coordinates use end-effectorminus-cube xyz. Reference 95th-percentile thresholds use 20,000 tabletop/non-contact training states per seed. For P02–P04, 2.0%/38.7%/72.0% exceed the joint threshold, and 52.7%/80.7%/95.3% exceed the relative threshold.

TABLE II: Cube ablations (%, three seeds). A/B show mean±population SD, C/D show means. HS averages P00– P04. Blue marks proposed/default settings in A/C/D. Bold indicates column-best means in A and row-best means in C/D.
<table><tr><td colspan="4">A. Component contributions</td></tr><tr><td>Variant</td><td colspan="2">Original ↑</td><td>HS ↑</td></tr><tr><td>LeWM  $\mathrm { A b s . + I n v + M I }$  Res</td><td> $7 3 . 3 \pm 2 . 5$   $8 3 . 3 \pm 0 . 9$   $8 2 . 7 \pm 2 . 5$ </td><td></td><td> $3 . 7 \pm 1 . 4$   $1 4 . 4 \pm 2 . 9$   $3 4 . 7 \pm 1 . 6$ </td></tr><tr><td colspan="4"> $\mathrm { R e s } + \mathrm { I n v }$   $8 3 . 3 \pm 1 . 9$   $\mathrm { R e s } + \mathrm { M I }$   $8 9 . 3 \pm 3 . 8$ </td></tr><tr><td colspan="4"> $\mathbf { A D - W M }$   ${ \bf 9 0 . 7 \pm 3 . 4 }$ </td></tr><tr><td colspan="4">B. MI gains across inverse inputs (HS)</td></tr><tr><td colspan="4">Input No MI Predicted endpoints  $3 7 . 1 \pm 4 . 7$ </td></tr><tr><td colspan="4">Encoded endpoints  $4 5 . 1 \pm 4 . 4$ </td></tr><tr><td colspan="4">Predicted increment  $3 9 . 9 \pm 2 . 1$ </td></tr><tr><td colspan="4">C. MI weight  $( \lambda _ { \mathrm { i n v } } = 0 . 1 )$ </td></tr><tr><td colspan="4">λMI 0 .005 .01 .015 .03 90.7 91.3 88.0</td></tr><tr><td colspan="4">Original 83.3 89.3 90.7 52.0</td></tr><tr><td colspan="4">HS 37.1 50.9</td></tr><tr><td colspan="4">D. Inv weight  $( \lambda _ { \mathrm { M I } } = 0 . 0 1 ) $ </td></tr><tr><td colspan="4">λinv 0 .05 .10</td></tr><tr><td colspan="4">Original 89.3 90.0 HS</td></tr><tr><td colspan="4">90.7 54.7 57.2 52.0</td></tr></table>

MI-enabled variants in A use $\lambda _ { \mathrm { M I } } = 0 . 0 1 .$ In B, $\lambda _ { \mathrm { i n v } } = 0 . 1$ and MI always uses predicted endpoints. Encoded endpoints remove only the direct Inv gradient through the predicted successor.

Cross-environment performance. Fig. 4(a) shows higher mean success than reproduced LeWM in four of five environments: Cube (73.3% → 90.7%), Reacher (76.7% → 83.3%), TwoRoom (90% → 98%), and Scene (35.5% → 39.5%). PushT decreases from 94% to 92%. All environments except Scene use original protocols. Reported external results are shown separately, including LeWM’s reported 86% on Reacher. Our improvement claim concerns the matched reproduction. Scene gains concentrate in Drawer (63.3% → 69.3%) and Window (46.7% → 55.3%), both improving in all three matched seed comparisons. Button is similar and

LeWM (reproduced) AD-WM (ours)  
![](images/876c5b74c52d2767b1a5d474301f55df32fc10acb0b32b275cca8b7da44be91d.jpg)  
(a) Cross-environment performance.

LeWM (reproduced) AD-WM (ours)  
![](images/524eb13529e82082fdb2393637be61ffa5c781b943aeba328106f5a1839f75a6.jpg)  
(b) Scene subtask performance.  
Fig. 4: Cross-environment and subtask performance. (a) Matched LeWM and AD-WM results, with reported baselines shown separately. Scene (\*) uses balanced hard starts, while other environments use original protocols. (b) Scene results over three seeds and 50 episodes per subtask, with $\lambda _ { \mathrm { M I } } = 1 0 ^ { - 4 }$ . Error bars show source SDs in (a) and population SDs in (b).

Cube remains difficult. The overall gain remains unresolved by the paired seed test $( p = 0 . 1 3 )$

## C. Ablation Studies

Table II addresses Q2 through component ablations (A), MI gains across inverse inputs (B), and post-hoc weight sensitivity (C/D).

Component contributions. Panel A identifies residual prediction and MI as the largest contributors. LeWM achieves 3.7% HS. Adding Inv+MI to absolute prediction raises this to 14.4%, while residual prediction alone reaches 34.7%. Res+Inv achieves 37.1%, Res+MI 54.7%, and AD-WM 52.0%. Inv gives a smaller mean gain over Res and does not improve upon Res+MI at its default weight.

MI gains across inverse inputs. Panel B varies Inv inputs while MI always uses predicted endpoints. MI adds 14.9/15.6/20.4 mean HS points across predicted endpoints, encoded endpoints, and predicted increments. The latter two reach 60.7/60.3%, versus 52.0% for the default; predicted increments still backpropagate Inv through the predictor. Thus, B tests Inv-input robustness, not MI-input robustness.

Weight sensitivity. With Inv fixed at 0.1, all tested nonzero MI weights (0.005–0.05) yield higher mean HS: 50.9–65.2% versus 37.1% (Panel C). MI peaks at 0.03; the pre-specified 0.01 remains primary. With MI fixed at 0.01, Inv weights 0.05/0.20 exceed the no-Inv mean, but the default 0.10 does not (Panel D). Inv’s effect is smaller and weight-dependent.

## D. Planning Diagnostics

Evaluation setup. To answer Q3, Table III compares prediction error, candidate selection, and HS. Factual errors use 61,999 held-out clips with three-frame context. Candidate selection evaluates five default-weight variants over three seeds on 64 shared cases with 300 candidates each. Banks contain four reference sequences (expert, zero, negatedexpert, and reversed-expert) and 99/99/98 near/medium/far perturbations. Actions and simulator endpoints are shared, while costs use each model’s encoder. The default elite size $k = 3 0$ matches CEM, with $k = 1 5 , 6 0$ testing robustness.

Prediction and selection quality. LeWM has the lowest factual MSE but also the lowest HS; AD-WM reaches 52.0% HS despite the highest MSE. These errors depend on representation and scale. Across 15 model–seed observations, HS correlates with CAD at $\rho = - 0 . 3 9 9$ and with negative $R _ { 3 0 }$ and $\bar { R } _ { 3 0 }$ at 0.863 and 0.810, respectively. The regret associations persist at $k = 1 5 , 6 0$ . Hierarchical paired bootstrap over seeds and cases (20,000 resamples) gives successive $R _ { 3 0 }$ reductions along LeWM–Res–Res+Inv–AD-WM of 0.028 [0.014, 0.043], 0.002 [−0.003, 0.009], and 0.014 [0.009, 0.023] (95% CIs). The Inv-only increment remains unresolved; Res+MI and AD-WM have similar mean regret. From LeWM to AD-WM, realized-best candidate retention falls from 0.198 to 0.135; a nearly tied alternative can still yield low regret. These associations concern fixedbank selection, not adaptive CEM trajectories.

Local dynamics and search geometry. We examine the first five steps of 25-step rollouts and 20 CEM landscapes on a 31 × 31 grid along the first two principal directions of the final elites. From LeWM to Res, latent-increment MSE decreases from 0.0067 to 0.0055, and the fraction of consecutive predicted increments with negative cosine similarity falls from 0.242 to 0.146. From LeWM to AD-WM, the distance between the CEM center and the lowestcost grid point decreases from 0.827 to 0.130, while mean elite-to-center distance falls from 0.448 to 0.082. These results are consistent with more stable local updates and more concentrated search.

## E. Real-Robot Transfer

Setup and post-training. To address Q4, we evaluate transfer on the Franka setup in Fig. 5(a). We compare AD-WM with V-JEPA 2-AC [12] using the same frozen V-JEPA 2 ViT-G encoder, filtered cadene/droid 1.0.1 data [34], predictor architecture, training schedule, CEM planner, clipping, and deployment stack. Only transition parameterization and auxiliary objectives differ. No laboratory images or demonstrations are used for adaptation.

TABLE III: Prediction, selection, and control metrics on Cube (mean±population SD, three seeds). Local MSE averages five rollout steps. MSE values are multiplied by $1 0 ^ { 3 } .$ . Shared banks use N = 300 and k = 30. Blue marks AD-WM. Bold marks the best displayed mean per metric, including ties.
<table><tr><td>Variant</td><td colspan="2">Factual prediction</td><td colspan="3">Candidate selection</td><td>Control</td></tr><tr><td></td><td>One-step MSE ↓</td><td>Local MSE ↓</td><td>CAD↑</td><td> $R _ { 3 0 } \downarrow$ </td><td> $\bar { R } _ { 3 0 \mathrm { ~ \downarrow ~ } }$ </td><td>HS (%) ↑</td></tr><tr><td>LeWM</td><td> ${ \bf 2 . 7 2 \pm 0 . 0 6 }$ </td><td> ${ \bf 1 0 . 3 6 \pm 0 . 4 8 }$ </td><td> $0 . 4 6 0 \pm 0 . 0 1 0$ </td><td> $0 . 0 7 4 \pm 0 . 0 0 6$ </td><td> $0 . 2 9 8 \pm 0 . 0 1 0$ </td><td> $3 . 7 \pm 1 . 4$ </td></tr><tr><td>Res</td><td> $3 . 6 1 \pm 0 . 2 0$ </td><td> $1 1 . 1 7 \pm 0 . 6 1$ </td><td> $0 . 4 6 9 \pm 0 . 0 0 5$ </td><td> $0 . 0 4 6 \pm 0 . 0 0 3$ </td><td> $0 . 2 5 0 \pm 0 . 0 0 9$ </td><td> $3 4 . 7 \pm { 1 . 6 }$ </td></tr><tr><td>Res + Inv</td><td> $3 . 4 3 \pm 0 . 1 1$ </td><td> $1 1 . 0 0 \pm 0 . 2 8$ </td><td> $\mathbf { 0 . 4 7 0 \pm 0 . 0 0 5 }$ </td><td> $0 . 0 4 3 \pm 0 . 0 0 2$ </td><td> $0 . 2 4 5 \pm 0 . 0 0 5$ </td><td> $3 7 . 1 \pm 4 . 7$ </td></tr><tr><td>Res + MI</td><td> $4 . 2 0 \pm 0 . 0 6$ </td><td> $1 5 . 1 7 \pm 0 . 0 4$ </td><td> $0 . 4 5 4 \pm 0 . 0 0 2$ </td><td> $\mathbf { 0 . 0 2 9 \pm 0 . 0 0 2 }$ </td><td> $\mathbf { 0 . 2 2 8 \pm 0 . 0 0 2 }$ </td><td> ${ \bf 5 4 . 7 \pm 3 . 0 }$ </td></tr><tr><td>AD-WM</td><td> $4 . 2 7 \pm 0 . 1 0$ </td><td> $1 5 . 7 7 \pm 0 . 2 0$ </td><td>_  $0 . 4 4 8 \pm 0 . 0 1 1$ </td><td> $\mathbf { 0 . 0 2 9 \pm 0 . 0 0 1 }$ </td><td>_  $0 . 2 2 9 \pm 0 . 0 0 5$ </td><td>_  $5 2 . 0 \pm 3 . 1$ </td></tr></table>

![](images/f74acfce465c460c6927b89647212e6ac9e7746a4abcf54339d7393582d45d95.jpg)  
Fig. 5: Real-robot evaluation. (a) Franka setup with a single camera. (b) Basic pick-and-place, complex-object, and specifiedtarget protocols use 45, 10, and 27 trials per model. (c) Matched examples illustrate grasping and target selection. Both models share image goals, execution stages, gripper handling, and action clipping.

The predictor has dimension 1024, depth 24, and 16 heads. Post-training uses four A800 GPUs, bf16, 315 epochs, 300 iterations per epoch, batch size 8 per GPU, and peak learning rate $4 . 2 5 \times 1 0 ^ { - 4 }$ . Auxiliary weights are $\lambda _ { \mathrm { i n v } } = 0 . 1 , \lambda _ { \mathrm { M I } } =$ $5 \times 1 0 ^ { - 5 }$ , and $\beta = 0 . 1$ , without tuning on Franka evaluation results. Actions are DROID-style 7D Cartesian/gripper deltas. Each replan predicts one latent transition using 800 CEM samples, 10 elites, and 10 iterations.

Evaluation protocol. A single side/rear RealSense camera provides eight frames at 4 fps with a 256 × 256 crop. MPC follows manually supplied grasp, move, and place image goals, with 10/10/4 execution steps. Cartesian deltas are clipped to norm 0.075 and converted to absolute base-frame targets. Gripper commands are thresholded, with forced closing during grasp and opening during place. This evaluates short-horizon MPC within a structured subgoal pipeline.

Each model receives 45 basic pick-and-place trials (three objects, five start-goal settings, three repeats), 10 complexobject trials, and 27 specified-target trials. The total is 82 trials per model and 164 overall. No trials are excluded, and safety stops count as failures. Evaluation uses separate, nonrandomized model blocks.

Results. Fig. 5(b) illustrates three protocols: pick-andplace across object colors and shapes, placing a Labubu doll in a basket, and selecting objects through different goal images. Table IV shows higher basic pick-and-place success, from 42.2% to 71.1%. All three objects improve $( 3 3 . 3 \%  6 0 . 0 \% , 4 6 . 7 \%  7 3 . 3 \%$ , and $4 6 . 7 \%  8 0 . 0 \% )$ Complex-object and specified-target results favor AD-WM, though smaller samples limit precision. In Fig. 5(c), AD-WM grasps the cube and selects the requested trash bin, while V-JEPA 2-AC misses the grasp or moves the wrong target. Abnormal motion denotes uncontrolled or implausible movement. Specified-target metrics distinguish moving the requested object from completing lift-and-place. These results support transfer within the shared deployment pipeline.

TABLE IV: Zero-shot Franka results (count/total). Abnormal motion counts adverse events. Bold marks better results.
<table><tr><td>Protocol Metric</td><td></td><td>V-JEPA 2-AC AD-WM</td><td></td></tr><tr><td>Basic</td><td>Success ↑</td><td>19/45</td><td>32/45</td></tr><tr><td rowspan="2">Complex Success ↑</td><td></td><td>2/10</td><td>5/10</td></tr><tr><td>Abnormal motion  $\downarrow$ </td><td>6/10</td><td>2/10</td></tr><tr><td rowspan="2">Target</td><td>Target moved ↑</td><td>14/27</td><td>21/27</td></tr><tr><td>Lift-and-place ↑</td><td>9/27</td><td>17/27</td></tr></table>

## V. CONCLUSION AND LIMITATIONS

We present AD-WM, an action-discriminative world model for counterfactual MPC. It combines residual pre diction with predictor-level action recovery to preserve action information without changing the planner. Experiments show gains over matched LeWM in four of five simulation environments and transfer to Franka without lab-specific adaptation. Cube ablations identify residual prediction and MI as the largest contributors, while Inv has a smaller, weight-dependent effect on mean success. Cube diagnostics show that elite regret tracks success more closely than factual error or whole-bank ranking.

Diagnostics remain limited to Cube, and Scene’s gain is unresolved over three seeds. External comparisons differ in inference and checkpoint counts. Simulation and robot posttraining use different auxiliary weights. Robot evaluation uses manual subgoals and non-randomized trials with one site, camera, and backbone. Future work examines broader settings and reduces reliance on manual subgoals.

## REFERENCES

[1] D. Hafner, T. Lillicrap, I. Fischer, R. Villegas, D. Ha, H. Lee, and J. Davidson, “Learning latent dynamics for planning from pixels,” in Proceedings of the 36th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 97. PMLR, 2019, pp. 2555–2565.

[2] D. Hafner, T. Lillicrap, J. Ba, and M. Norouzi, “Dream to control: Learning behaviors by latent imagination,” in International Conference on Learning Representations, 2020.

[3] D. Hafner, T. Lillicrap, M. Norouzi, and J. Ba, “Mastering atari with discrete world models,” in International Conference on Learning Representations, 2021.

[4] D. Hafner, J. Pasukonis, J. Ba, and T. Lillicrap, “Mastering diverse control tasks through world models,” Nature, vol. 640, pp. 647–653, 2025.

[5] N. Hansen and X. Wang, “Temporal difference learning for model predictive control,” in Proceedings of the 39th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 162. PMLR, 2022, pp. 8383–8396.

[6] N. Hansen, H. Su, and X. Wang, “TD-MPC2: Scalable, robust world models for continuous control,” in International Conference on Learning Representations, 2024.

[7] M. Assran, Q. Duval, I. Misra, P. Bojanowski, P. Vincent, M. Rabbat, Y. LeCun, and N. Ballas, “Self-supervised learning from images with a joint-embedding predictive architecture,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 15 619–15 629.

[8] A. Bardes, Q. Garrido, J. Ponce, X. Chen, M. Rabbat, Y. LeCun, M. Assran, and N. Ballas, “Revisiting feature prediction for learning visual representations from video,” Transactions on Machine Learning Research, 2024.

[9] G. Zhou, H. Pan, Y. LeCun, and L. Pinto, “DINO-WM: World models on pre-trained visual features enable zero-shot planning,” in Proceedings of the 42nd International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 267. PMLR, 2025, pp. 79 115–79 135.

[10] V. Sobal, W. Zhang, K. Cho, R. Balestriero, T. G. J. Rudner, and Y. LeCun, “Learning from reward-free offline data: A case for planning with latent dynamics models,” 2025.

[11] L. Maes, Q. Le Lidec, D. Scieur, Y. LeCun, and R. Balestriero, “LeWorldModel: Stable end-to-end joint-embedding predictive architecture from pixels,” 2026.

[12] M. Assran, A. Bardes, D. Fan et al., “V-JEPA 2: Self-supervised video models enable understanding, prediction and planning,” 2025. [Online]. Available: https://arxiv.org/abs/2506.09985

[13] R. Y. Rubinstein and D. P. Kroese, The Cross-Entropy Method: A Unified Approach to Combinatorial Optimization, Monte-Carlo Simulation and Machine Learning. Springer, 2004.

[14] D. Ha and J. Schmidhuber, “World models,” arXiv preprint arXiv:1803.10122, 2018. [Online]. Available: https://arxiv.org/abs/ 1803.10122

[15] R. S. Sutton, “Dyna, an integrated architecture for learning, planning, and reacting,” ACM SIGART Bulletin, vol. 2, no. 4, pp. 160–163, 1991.

[16] M. Watter, J. T. Springenberg, J. Boedecker, and M. Riedmiller, “Embed to control: A locally linear latent dynamics model for control from raw images,” in Advances in Neural Information Processing Systems, vol. 28, 2015.

[17] C. Finn and S. Levine, “Deep visual foresight for planning robot motion,” in IEEE International Conference on Robotics and Automation, 2017, pp. 2786–2793.

[18] V. Micheli, E. Alonso, and F. Fleuret, “Transformers are sampleefficient world models,” in International Conference on Learning Representations, 2023. [Online]. Available: https://openreview.net/ forum?id=vhFu1Acb0xb

[19] E. Alonso, A. Jelley, V. Micheli, A. Kanervisto, A. Storkey, T. Pearce, and F. Fleuret, “Diffusion for world modeling: Visual details matter in atari,” in Advances in Neural Information Processing Systems, 2024. [Online]. Available: https://openreview.net/forum?id=NadTwTODgC

[20] S. Gao, W. Liang, K. Zheng, A. Malik, S. Ye, S. Yu, W.-C. Tseng, Y. Dong, K. Mo, C.-H. Lin, Q. Ma, S. Nah, L. Magne, J. Xiang, Y. Xie, R. Zheng, D. Niu, Y. L. Tan, K. R. Zentner, G. Kurian, S. Indupuru, P. Jannaty, J. Gu, J. Zhang, J. Malik, P. Abbeel, M.-Y. Liu, Y. Zhu, J. Jang, and L. J. Fan, “DreamDojo: A generalist robot world model from large-scale human videos,” 2026.

[21] V. Sobal, S. V. Jyothir, S. Jalagam, N. Carion, K. Cho, and Y. LeCun, “Joint embedding predictive architectures focus on slow features,” 2022.

[22] Y. Gao and X. Xu, “Fast LeWorldModel,” arXiv preprint arXiv:2606.26217, 2026.

[23] K. Zhao, D. Nie, Y. Lin, Z. Luo, Y. Gu, D.-P. Fan, and D. Zeng, “Sub-JEPA: Subspace gaussian regularization for stable end-to-end world models,” 2026.

[24] J. Sun, H. Zhao, and G. Zhang, “INTACT: Isomorphic intent-to-action learning for search-free world models,” 2026.

[25] R. Rakhimov, G. Bredis, Y. Maksyuta, and D. Gavrilov, “Qantara: Bridge-flow training for multi-paradigm JEPA control,” 2026.

[26] Z. Zhang, Y. Wang, Z. Guan, Y. Yang, B. Shi, T. Zong, H. Yi, G. Chao, X. Chen, T. Yang, C. Bao, T. Yu, J. Zhou, and J. Xu, “Delta-JEPA: Learning action-sensitive world models via latent difference decoding,” 2026.

[27] P. Agrawal, A. V. Nair, P. Abbeel, J. Malik, and S. Levine, “Learning to poke by poking: Experiential learning of intuitive physics,” in Advances in Neural Information Processing Systems, vol. 29, 2016.

[28] D. Pathak, P. Agrawal, A. A. Efros, and T. Darrell, “Curiosity-driven exploration by self-supervised prediction,” in Proceedings of the 34th International Conference on Machine Learning, 2017.

[29] C. Allen, N. Parikh, O. Gottesman, and G. Konidaris, “Learning markov state abstractions for deep reinforcement learning,” in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 8229–8241.

[30] B. Huang, C. Lu, L. Leqi, J. M. Hernandez-Lobato, C. Glymour,´ B. Scholkopf, and K. Zhang, “Action-sufficient state representation ¨ learning for control with structural constraints,” in Proceedings of the 39th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 162. PMLR, 2022, pp. 9260– 9279.

[31] D. Barber and F. V. Agakov, “Information maximization in noisy channels: A variational approach,” in Advances in Neural Information Processing Systems, vol. 16, 2003.

[32] S. Park, K. Frans, B. Eysenbach, and S. Levine, “OGBench: Bench-

marking offline goal-conditioned reinforcement learning,” in International Conference on Learning Representations, 2025.

[33] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” in International Conference on Learning Representations, 2021. [Online]. Available: https://openreview.net/forum?id=YicbFdNTTy

[34] A. Khazatsky, K. Pertsch, S. Nair, A. Balakrishna, S. Dasari, S. Karamcheti, S. Nasiriany, M. K. Srirama, L. Y. Chen, K. Ellis et al., “DROID: A large-scale in-the-wild robot manipulation dataset,” in Robotics: Science and Systems, 2024. [Online]. Available: https://arxiv.org/abs/2403.12945