# IMPACT-VLA: Interaction-aware Multimodal Propagation Attribution via Counterfactual Trajectories for Vision-Language-Action Policies

Jinwoong Kim and Sangjin Park<sup>∗</sup>

Abstract— Vision-Language-Action (VLA) policies perform robot manipulation tasks using multimodal inputs such as visual observations, proprioceptive states, and language instructions. However, it remains unclear at which execution stages each modality contributes to final task success and how the effects of input interventions propagate through subsequent states, observations, and actions. Existing attribution approaches primarily measure local sensitivity or temporally aggregated importance, limiting their ability to capture phase-dependent modality contributions and cross-phase dependencies. To address this limitation, we propose Interaction-aware Multimodal Propagation Attribution via Counterfactual Trajectories for Vision-Language-Action Policies (IMPACT-VLA). IMPACT-VLA constructs behavioral phases from action transitions in a successful reference rollout, aligns them with policy query boundaries, and defines phase-modality blocks as attribution units. It then performs closed-loop counterfactual re-execution to quantify each block’s contribution to final task success. We further analyze cross-phase non-additive interactions and trajectory propagation while distinguishing behavioral from functional recovery. Applying IMPACT-VLA to 30 robot manipulation tasks from the LIBERO benchmark using OpenVLA-OFT, we observed dominant-modality transitions across behavioral phases in 25 tasks (83.3%), and closed-loop attribution identified task-critical information more faithfully than Static Action Perturbation. We also found that later-block marginal gains for negatively interacting pairs increased by approximately 3.3× under early-phase input replacement, and that functional recovery could occur without behavioral recovery. These results provide an execution-level analysis of when multimodal inputs support task success and how their contributions become conditionally coupled across closed-loop execution.

## I. INTRODUCTION

Vision-Language-Action (VLA) policies perform a wide range of robot manipulation tasks using multimodal inputs such as visual observations, proprioceptive states, and language instructions [1], [2]. As their performance and generalization improve, understanding which input information contributes to task success during actual execution has become increasingly important. The role of a modality may vary across execution stages, and an input change at a particular time can affect subsequent environment states, observations, and action generation [3]. Therefore, interpreting VLA policies requires analyzing not only which modalities are important, but also when they are needed and how their influence propagates during closed-loop execution.

Existing work on embodied policy interpretability has identified salient observations, important states, or internal action-related representations, while perturbation-, gradient-, and Shapley-based methods quantify input contributions across individual or coalition contexts [4], [5]. However, in sequential robot policies, an intervention affects not only the immediate action but also subsequent states and observations, requiring input contributions to be evaluated in closed loop with respect to final task success. Moreover, modality-level analysis can obscure stage-wise variation, whereas timestep level analysis may not match the actual intervention unit. This is particularly important for chunk-based VLAs, where a single query generates multiple actions, requiring a temporal unit that reflects action transitions while remaining aligned with the action-chunk structure [6], [7].

Even with an appropriate temporal unit, independently measuring each phase is insufficient to explain dependen cies across execution. Earlier- and later-phase information may interact in complementary or redundant ways, and the contribution of later inputs may change when earlier information is unavailable [8]. In addition, trajectory reconvergence and recovery of final task success may represent distinct phenomena. Therefore, cross-phase interactions and separate measures of behavioral and functional recovery are needed to characterize post-intervention propagation and compensation.

To this end, we propose Interaction-aware Multimodal Propagation Attribution via Counterfactual Trajectories for Vision-Language-Action Policies (IMPACT-VLA). IMPACT-VLA constructs behavioral phases from action transitions in a successful reference rollout, aligns them with policy query boundaries, and defines phase-modality blocks as attribution units. It then re-executes the policy in closed loop from intervention-altered environment states to quantify each block’s contribution to final task success across coalition contexts. Cross-phase non-additive interactions, trajectory propagation, and separate behavioral and functional recovery measures are further used to analyze information dependencies and compensation across execution stages.

Experiments on OpenVLA-OFT across 30 LIBERO tasks show phase-dependent modality transitions, improved closed-loop faithfulness over immediate action sensitivity, and conditional cross-phase dependencies. We further observe that early-phase input replacement increases laterblock marginal gains for negatively interacting pairs by approximately 3.3×, and that functional recovery can occur without behavioral recovery.

The main contributions of this work are as follows:

• We define phase-modality blocks aligned with actual action transitions and the action-chunk structure as attribution units, providing a temporal unit for analyzing when each modality contributes to robot task success.

• We formulate multimodal information use as a global phase-resolved closed-loop counterfactual attribution problem, placing blocks from different phases in a shared coalition space while re-executing the policy from intervention-altered environment states.

• Using this shared cross-phase formulation, we analyze non-additive interactions and trajectory propagation while distinguishing behavioral recovery from functional recovery to characterize conditional dependencies and compensation following early information loss.

## II. RELATED WORK

## A. Interpretability of Embodied Policies

Early interpretability studies of reinforcement learning and robot policies identified observations, states, or behavioral events that influence policy decisions using perturbationand saliency-based analyses [9]. In robot manipulation, multimodal execution traces have also been combined with language models to diagnose failures and generate corrective strategies [10]. These approaches broaden interpretation from individual observations to execution-level behavior, but primarily focus on salient states, behavioral events, or failure causes.

Recent work has applied mechanistic interpretability directly to VLA models by identifying and intervening on latent representations associated with actions or task progress [11], [12]. Sparse autoencoders further decompose hidden activations into interpretable features and relate them to actions or trajectory events [13], [14]. However, these approaches do not directly quantify when each input modality is required during execution or how its contribution to final task success propagates through subsequent states and actions. Addressing this requires intervention on actual multimodal inputs and analysis of their closed-loop effects.

## B. Input Attribution and Evaluation

Perturbation- and gradient-based methods such as Occlusion, LIME, RISE, Grad-CAM, and Integrated Gradients attribute model predictions to individual input components [15]–[19]. However, individual perturbations or gradients do not explicitly characterize combinatorial dependencies between redundant or complementary inputs.

Shapley-based methods instead quantify marginal contributions across coalitions [20]. SHAP generalizes this formulation to model predictions [21], while TimeSHAP and WindowSHAP extend it to temporal inputs [4], [22], and SVERL evaluates state-feature contributions with respect to long-term policy performance [5]. The Shapley–Taylor interaction index further characterizes non-additive dependencies such as complementarity and redundancy [8].

Deletion and insertion test faithfulness by removing or restoring inputs [17], while retraining-based methods address perturbation shift [23]. TimeSHAP/WindowSHAP operate on temporal inputs and SVERL on state features tied to long-term performance; in contrast, our players are phase–modality blocks evaluated by re-executing the policy–environment loop. This captures intervention-induced changes in subsequent states and observations while enabling cross-phase interaction, propagation, and recovery analysis.

## C. Robot Behavior Segmentation

Robot behavior segmentation has primarily focused on discovering reusable skills or subtasks from demonstrations. Transition State Clustering identifies recurring transition states, while DDCO and CompILE jointly learn segmentation and latent skills or options [24]–[26]. These methods target behavioral structure shared across demonstrations and typically require repeated demonstrations or additional learning, whereas our setting requires post hoc segmentation of a single execution from an already trained policy.

Change-point detection provides an alternative without separate skill learning. CUSUM and related methods detect statistical changes in temporal signals [27], including multivariate settings using channel-wise change statistics [28]. However, statistically detected boundaries are not necessarily valid intervention boundaries for robot policies.

This distinction is particularly important for chunk-based policies, where one policy query generates multiple future actions [6]. A changepoint may therefore occur inside an action chunk, where the corresponding input cannot be independently perturbed. We consequently require a temporal unit that preserves changes in executed behavior while remaining aligned with policy-query and action-chunk boundaries.

## III. METHODOLOGY

Fig. 1 illustrates the overall analysis procedure of IMPACT-VLA. A successful execution in which all input modalities are provided normally is used as the reference rollout, and the rollout is segmented into behavioral phases based on transitions in the actions actually executed. Each combination of a phase and a modality forms a phasemodality block. The policy and environment are then reexecuted in closed loop while replacing the inputs of selected blocks to estimate each block’s contribution to final task success. The same counterfactual re-executions are also used to analyze cross-phase non-additive interactions, interventioninduced trajectory propagation, and recovery.

## A. Problem Formulation

We denote the VLA policy by π and the set of input modalities by $\mathcal { M } = \{ m _ { 1 } , . . . , m _ { M } \}$ . We consider a chunkbased VLA that generates an action chunk of length K at each policy query and executes the actions sequentially [6]. The multimodal observation at query q and the generated action chunk are represented as follows.

$$
\boldsymbol { o } _ { q } = ( o _ { q } ^ { m } ) _ { m \in \mathcal { M } } , \qquad \hat { \boldsymbol { A } } _ { q } = \pi ( \boldsymbol { o } _ { q } ) = ( \hat { a } _ { q , 0 } , \dots , \hat { a } _ { q , K - 1 } ) ,\tag{1}
$$

where $d _ { a }$ denotes the action dimensionality and $\hat { a } _ { q , k } \in \mathbb { R } ^ { d _ { a } }$ Let $a _ { t }$ denote the action actually applied at control timestep t. The environment state $x _ { t }$ evolves according to the executed action and stochastic factor $\epsilon _ { t } .$

$$
x _ { t + 1 } = f ( x _ { t } , a _ { t } , \epsilon _ { t } ) .\tag{2}
$$

![](images/5313ce511b77ef1c6a5e5320376dca3e3bb1ccbf996221236ea190035feb8158.jpg)  
Fig. 1. Overview of the IMPACT-VLA framework.

We denote the resulting closed-loop execution from initial state $x _ { 0 }$ by τ , with task success $Y ( \tau ) \in \{ 0 , 1 \}$ . Because an intervention can affect subsequent states, observations, and actions, the policy–environment interaction is re-executed after intervention until termination.

The analysis is based on a successful reference rollout $\tau ^ { \mathrm { r e f } }$ in which all modalities are provided without intervention and $Y ( \tau ^ { \mathrm { r e f } } ) = 1$ . Phase construction uses the actions actually executed in this rollout, whose length is denoted by $T ^ { \mathrm { r e f } }$

## B. Phase Construction

Because the role of a modality may vary across execution stages, we segment the reference rollout into behavioral phases based on executed action transitions. These phases are execution-aligned intervention units, not semantic skill labels. Since action channels differ in scale, each channel is normalized by its standard deviation over the reference rollout, and the normalized action of channel j is denoted by $\tilde { a } _ { t , j } ^ { \mathrm { r e f } }$

For a candidate changepoint c within an interval $[ u , v )$ , the CUSUM statistic for channel j is defined from the difference between the mean actions before and after c.

$$
C _ { j } ( c ; u , v ) = \sqrt { \frac { n _ { L } n _ { R } } { n _ { L } + n _ { R } } } \left| \bar { \tilde { a } } _ { j , [ u , c ) } - \bar { \tilde { a } } _ { j , [ c , v ) } \right| ,\tag{3}
$$

where $n _ { L } = c - u$ and $\begin{array} { r } { n _ { R } = v - c . } \end{array}$ , and $\bar { \tilde { a } } _ { j , [ u , c ) }$ and $\bar { \tilde { a } } _ { j , [ c , v ) }$ denote the normalized action means over the corresponding intervals. For each channel, we retain the strongest admissible changepoint exceeding $\lambda _ { C } = 5 . 0$ , with the minimum segment length set to $L _ { \mathrm { m i n } } = 8$ action steps.

Channel-wise transitions within K timesteps are merged into a single group. Let $\gamma _ { h }$ denote the CUSUM-strengthweighted representative location of group h. Because interventions can be applied independently only at policy-query boundaries, each representative transition is aligned to the

nearest query boundary.

$$
\tilde { \beta } _ { h } = K \cdot \mathrm { r o u n d } \Big ( \frac { \gamma _ { h } } { K } \Big ) .\tag{4}
$$

If multiple transitions align to the same boundary, the stronger one is retained, and any phase shorter than one action chunk is merged with an adjacent phase. The final boundaries are denoted by $0 = \beta _ { 0 } < \beta _ { 1 } < \cdots < \beta _ { P } = T ^ { \mathrm { r e f } }$ The behavioral phases and phase-modality blocks are defined as follows.

$$
\begin{array} { c } { \mathcal { T } _ { p } ^ { \mathrm { r e f } } = [ \beta _ { p - 1 } , \beta _ { p } ) , } \\ { B = \{ ( p , m ) \mid p = 1 , \ldots , P , m \in \mathcal { M } \} . } \end{array}\tag{5}
$$

The total number of blocks is $\begin{array} { r } {  { N } =  { P } | \mathcal { M } | } \end{array}$ . The same modality in different phases is treated as a distinct attribution player, and the reference phase boundaries remain fixed across all counterfactual rollouts.

## C. Closed-Loop Attribution

A subset $S \subseteq B$ contains the blocks for which the original inputs are preserved. We denote the replacement function for modality m by $r _ { m } ^ { \rho }$ , where $\rho$ represents a replacement configuration.

For a counterfactual query $q ,$ let $\bar { o } _ { q } ^ { m }$ denote the preintervention input obtained from the current environment state, and let $p ( q )$ denote the behavioral phase corresponding to the execution time of query $q .$ For queries occurring beyond the reference horizon $T ^ { \mathrm { r e f } }$ , we set $\begin{array} { r } { p ( q ) \ = \ P , } \end{array}$ so that the final behavioral phase remains active until the counterfactual rollout terminates. The input actually provided to the policy is defined as follows.

$$
o _ { q , S , \rho } ^ { m } = \left\{ \begin{array} { l l } { \bar { o } _ { q } ^ { m } , } & { ( p ( q ) , m ) \in S , } \\ { r _ { m } ^ { \rho } ( \bar { o } _ { q } ^ { m } ) , } & { ( p ( q ) , m ) \notin S . } \end{array} \right.\tag{6}
$$

Importantly, stored reference observations are not reused. Each $\bar { o } _ { q } ^ { m }$ is newly acquired from the current environment state, which may have been altered by preceding interventions. Therefore, the effects of an intervention propagate in closed loop through subsequent states, observations, and action generation.

All counterfactual rollouts start from the same initial environment state as the corresponding reference rollout. We denote the closed-loop rollout generated under coalition S, replacement configuration $\rho ,$ and rollout randomness ω by $\tau ^ { \dot { S } , \rho , \omega }$ , and define the coalition value as the expected final task success.

$$
V _ { \rho } ( S ) = \mathbb { E } _ { \omega } [ Y ( \tau ^ { S , \rho , \omega } ) ] .\tag{7}
$$

In practice, each coalition occurrence within a sampled permutation is evaluated once, with all prefix coalitions sharing the same rollout seed. Thus, marginal contributions are paired within each permutation, while different permutations use reproducibly generated seeds. Thus, attribution reflects each block’s contribution to final task success rather than immediate action sensitivity.

The Shapley contribution of block $b \in B$ is defined as its average marginal contribution across all coalition contexts [20].

$$
\phi _ { b } ^ { \rho } = \sum _ { S \subseteq B \setminus \{ b \} } \frac { | S | ! ( N - | S | - 1 ) ! } { N ! } \big [ V _ { \rho } ( S \cup \{ b \} ) - V _ { \rho } ( S ) \big ] .\tag{8}
$$

To reduce the cost of directly evaluating all coalitions, we use uniformly sampled block permutations [29]. For permutation $\sigma _ { \ell } ,$ let $\mathrm { P r e d } _ { \sigma _ { \ell } } ( b )$ denote the set of blocks preceding b. The sampled Shapley contribution is computed as follows.

$$
\hat { \phi } _ { b } ^ { \rho } = \frac { 1 } { L } \sum _ { \ell = 1 } ^ { L } \left[ V _ { \rho } ( \mathrm { P r e d } _ { \sigma _ { \ell } } ( b ) \cup \{ b \} ) - V _ { \rho } ( \mathrm { P r e d } _ { \sigma _ { \ell } } ( b ) ) \right] .\tag{9}
$$

where L is the number of sampled permutations. In the LIBERO experiments, we use $L \ = \ 1 2 8$ , with one rollout seed per permutation. Block contributions are organized in an attribution map whose two axes are behavioral phase and input modality.

## D. Cross-Phase Interaction

Shapley contributions quantify the average marginal contribution of each block, but they do not directly represent non-additive dependencies that arise when two blocks from different phases are provided jointly. Let $i = ( p _ { i } , m _ { i } )$ be an earlier block and $j = ( p _ { j } , m _ { j } )$ be a later block, with $p _ { i } < p _ { j }$ The cross-phase interaction measures the extent to which the joint effect of the two blocks deviates from the sum of their individual effects under the same coalition context.

$$
\begin{array} { r l } & { I _ { i , j } ^ { \rho } = \mathbb { E } _ { S \sim \nu } \big [ V _ { \rho } ( S \cup \{ i , j \} ) - V _ { \rho } ( S \cup \{ i \} ) } \\ & { \qquad - V _ { \rho } ( S \cup \{ j \} ) + V _ { \rho } ( S ) \big ] , } \end{array}\tag{10}
$$

where $S \subseteq B \backslash \{ i , j \}$ , and ν is a context sampling distribution constructed so that different coalition sizes are represented evenly. Interaction contexts are sampled independently of the permutations used for Shapley attribution. $I _ { i , j } ^ { \rho }$ is not an approximation of the Shapley interaction index; rather, it is a context-averaged finite-difference effect. A positive value is consistent with complementarity, in which the joint effect of the two blocks exceeds the sum of their individual effects, whereas a negative value is consistent with redundancy or a suppressive interaction. This allows us to analyze how earlier- and later-phase information interact to support final task success.

## E. Propagation and Recovery

Cross-phase interaction measures dependencies between phases with respect to task success, but does not directly explain how deviations induced by an early intervention propagate and are compensated for later. Let $\bar { \boldsymbol { z } _ { t } ^ { \mathrm { r e f } } } \in \mathbb { R } ^ { 3 }$ and $\bar { z } _ { t } ^ { S , \bar { \rho } }$ denote the 3D end-effector positions at control timestep t in the reference and counterfactual rollouts, respectively. The trajectories are compared at the same absolute control timesteps; thus, large intervention-induced temporal delays are not time-warped and may contribute to measured deviation.

If a counterfactual rollout terminates earlier than the reference rollout, its last observed end-effector position is held constant until the reference horizon and denoted by $\tilde { z } _ { t } ^ { S , \rho }$ If it extends beyond the reference horizon, execution continues under the final-phase assignment defined in Sec. III-C, while the additional portion is excluded from trajectory comparison. Let $\mathcal { T } _ { p } ^ { \mathrm { r e f } }$ denote the timesteps belonging to phase $p .$ The phase-level trajectory distance is defined as follows.

$$
D _ { p } ( S , \rho ) = \frac { 1 } { | \mathcal { T } _ { p } ^ { \mathrm { r e f } } | } \sum _ { t \in \mathcal { T } _ { p } ^ { \mathrm { r e f } } } \Vert \tilde { z } _ { t } ^ { S , \rho } - z _ { t } ^ { \mathrm { r e f } } \Vert _ { 2 } .\tag{11}
$$

Using the full-input condition as a baseline, we track the excess deviation caused by replacing an early block. For an early block i and a later block j in phase $p _ { j }$ , the deviation immediately before $j$ becomes active is defined as follows.

$$
E _ { \mathrm { p r e } } ( i , j ) = D _ { p _ { j } - 1 } ( S _ { \mathrm { e a r l y } } , \rho ) - D _ { p _ { j } - 1 } ( S _ { \mathrm { f u l l } } , \rho ) .\tag{12}
$$

Recovery quantities are evaluated over 32 paired rollouts, and recovery analysis is restricted to ordered pairs with finite $E _ { \mathrm { p r e } } ( i , j ) ~ > ~ 0 .$ This pre-damage gate is applied only to recovery metrics.

Behavioral Recovery. We consider $S _ { \mathrm { f u l l } } = B , S _ { \mathrm { e a r l y } } =$ $B \setminus \{ i \} , S _ { \mathrm { l a t e } } = B \setminus \{ j \}$ , and $S _ { \mathrm { b o t h } } = B \setminus \{ i , j \}$ . For phase $p \geq p _ { j }$ , the recovery effect attributable to the later block is defined in a difference-in-differences form.

$$
\begin{array} { r l } & { \mathrm { R e c } _ { p } ^ { \rho } ( i  j ) = [ D _ { p } ( S _ { \mathrm { b o t h } } , \rho ) - D _ { p } ( S _ { \mathrm { l a t e } } , \rho ) ] } \\ & { \qquad - [ D _ { p } ( S _ { \mathrm { e a r l y } } , \rho ) - D _ { p } ( S _ { \mathrm { f u l l } } , \rho ) ] . } \end{array}\tag{13}
$$

Behavioral Recovery is identified when the pre-damage gate is satisfied, the excess deviation decreases after j becomes active, and $\operatorname { R e c } _ { p } ^ { \rho } ( i \  \ j ) \ > \ 0$ . This separates recovery attributable to j from natural trajectory convergence.

Functional Recovery. Because trajectory convergence does not necessarily imply recovery of final task success, we separately measure whether later block j compensates for the loss of earlier block i. The functional recovery effect is defined as follows.

$$
\begin{array} { r l } { \mathrm { F R e c } _ { \rho } ( i  j ) = [ V _ { \rho } ( S _ { \mathrm { e a r l y } } ) - V _ { \rho } ( S _ { \mathrm { b o t h } } ) ] } & { } \\ { - [ V _ { \rho } ( S _ { \mathrm { f u l l } } ) - V _ { \rho } ( S _ { \mathrm { l a t e } } ) ] . } \end{array}\tag{14}
$$

FRec is the negative pairwise finite difference at the fullcoalition context, whereas Eq. (10) averages that quantity across contexts; it is therefore a full-information compensation criterion, not an independent interaction measure. Functional Recovery requires the pre-damage gate and that j increases final success after replacement of i.

$$
V _ { \rho } ( S _ { \mathrm { e a r l y } } ) - V _ { \rho } ( S _ { \mathrm { b o t h } } ) > 0 .\tag{15}
$$

Its success-preserving effect must also be stronger than under the full-input condition.

$$
\mathrm { F R e c } _ { \rho } ( i \to j ) > 0 .\tag{16}
$$

Behavioral Recovery reflects trajectory stabilization, whereas Functional Recovery reflects task-level compensation; the two may occur jointly or independently.

## IV. EXPERIMENTS

We evaluate IMPACT-VLA in controlled settings and on LIBERO [30], covering attribution accuracy, phasedependent modality contributions, closed-loop faithfulness, cross-phase interactions, propagation, and recovery.

## A. Experimental Setup

For each task, we use OpenVLA-OFT [31] with a fixed initial state and reference seed. One successful execution defines the phase structure, while all analyses use repeated closed-loop counterfactual re-executions. Phasemodality blocks follow Sec. III-B, with reference boundaries fixed across counterfactual rollouts.

Mean intervention is used for primary attribution, with gray, blur, and noise for robustness. For third-person and wrist vision, mean uses a suite-specific pixel-wise mean image; gray sets RGB to 128, blur uses Gaussian radius 12, and noise uses fixed-seed Gaussian noise centered at the current image mean with standard deviation 40. Proprioception uses the suite-specific mean 8-D state vector and language an empty string. Mean baselines use all frames from all 10 tasks in each suite, three initial states, and five rollouts per state. For each task, interaction screening uses 32 contexts/pair under the mean kernel; the strongest positive and negative edges, the remaining largest-|I| edge, and two near-zero controls are re-evaluated on 32 held-out contexts/pair. Propagation/recovery uses 32 paired rollout seeds per condition.

Baselines are LOO, Static Action Perturbation, and Random. LOO measures the success decrease when one block in the full coalition is replaced, Static Action Perturbation uses the immediate action change after perturbation, and Random assigns a random ranking. For structural ablations, Phase-restricted Shapley uses within-phase modalities as players with other phases unperturbed, Modality-only Shapley uses one player per modality across the episode, and Uniform query-aligned phases preserve the IMPACT-VLA phase count but place boundaries uniformly at queryaligned positions. All faithfulness comparisons use the mean intervention.

Each task is the statistical unit. We aggregate within task and report task-macro means; method comparisons use task-paired differences, and brackets denote 95% percentile

CIs from 10,000 task-cluster bootstrap resamples. Table I summarizes the configuration.

TABLE I  
EXPERIMENTAL CONFIGURATION.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Policy</td><td>OpenVLA-OFT,  $K = 8$ </td></tr><tr><td>Evaluation</td><td>30 tasks: Spatial 10, Object 10, Goal 10</td></tr><tr><td>Modalities</td><td>Third-person, wrist, proprioception, language</td></tr><tr><td>Phase counts</td><td> $P = \dot { 3 } ; 1 , P = 4 ; 9 , \dot { P } = 5 ; 1 5 , P = \tilde { 6 } ; \tilde { 5 }$ </td></tr><tr><td>Max horizon</td><td>Spatial 220, Object 280, Goal 300</td></tr><tr><td>Intervention kernels Mean, gray, blur, noise</td><td></td></tr><tr><td>Attribution</td><td>128 permutations/kernel</td></tr><tr><td>Interaction</td><td>32 contexts/pair + 32 held-out</td></tr><tr><td>Faithfulness</td><td> $k = \{ 1 , 2 , \bar { 3 , 4 , 5 } \}$  , 16 replicates/task</td></tr><tr><td>Statistics</td><td>10,000 task-cluster bootstrap, 95% CI</td></tr></table>

## B. Controlled Validation of Attribution Estimation

Because the ground-truth contribution of each phasemodality block cannot be directly observed during actual LIBERO execution, we validated the accuracy of the sampled Shapley estimator in a stochastic closed-loop game in which all coalitions can be enumerated. We considered three scenarios: Additive with independent block effects, Delayed Credit with dependencies between temporally separated blocks, and Genuine Interaction with explicit non-additive cross-phase effects. We varied the permutation sampling budget over 2, 6, 12, 24, 128, and 512 and repeated each condition 64 times. Attribution accuracy was evaluated using the mean absolute error (MAE) relative to the exact-enumeration Shapley values and the Spearman correlation with the ground-truth ranking.

![](images/52fc7e96180bdcfc20a9120b665fb6cd0f02d55ae5668cde3f84f3c7a57c6cc1.jpg)

![](images/c347596442971c7b394c41e71ea6828cbcc46e16112af8cd80c63efe345c9266.jpg)  
Fig. 2. Controlled attribution validation. Top: MAE. Bottom: Spearman correlation.

As shown in Fig. 2, estimation error decreased and ranking agreement increased as the sampling budget grew across all three scenarios. At the 128-permutation budget used in the LIBERO analysis, the MAE remained below 0.03 and the Spearman correlation remained above 0.89 in all scenarios. For L sampled permutations over N blocks, the estimator requires $L ( N + 1 )$ coalition evaluations, whereas exact subset-based computation requires $2 ^ { N }$ unique coalition values. Thus, for the block counts considered here, the sampled estimator provides reliable attribution and ranking estimates at substantially lower evaluation cost.

## C. Phase-Dependent Modality Attribution

We analyze how modality contributions vary across behavioral phases in LIBERO rollouts. Fig. 3 projects phase-level Shapley contributions under the primary mean intervention onto normalized rollout progress. Each rollout endpoint is normalized to 100%, and each colored segment represents the contribution of the corresponding behavioral phase rather than a timestep-wise attribution.

![](images/3b4abe6e6154d6a022e6746c549c9d25c93769fda643eb13e22b2e80d78bd326.jpg)

normalized trajectory progress (%)  
![](images/313d47531ea0c4b1342d7967c0eea41a99b9e70a5bbd2f168f0865857a852743.jpg)

normalized trajectory progress (%)  
![](images/0a75cde48f5f37eea59cbb72724221c41aed70371a10d5ef11535ad4481bbddc.jpg)  
normalized trajectory progress (%)  
Fig. 3. Phase-dependent modality attribution. Top: Spatial. Middle: Object. Bottom: Goal.

In LIBERO-Spatial, wrist vision contributes prominently during early execution, while proprioception becomes more important in later stages. In LIBERO-Object, wrist vision is relatively important early, whereas third-person vision increases toward task completion. In LIBERO-Goal, language contributes prominently in early stages, followed by increased contributions from wrist vision and proprioception. These patterns indicate that modality dependence varies across behavioral phases rather than remaining constant throughout an episode.

This variation is also observed at the task level. Of the 30 tasks, 25 (83.3%) exhibit at least one dominant-modality change between adjacent phases, and 37 of 92 dominant-set changes (40.2%) show no overlap between the sets before and after the transition. Modalities with identical attribution values are treated as a joint dominant set. Thus, phasedependent modality transitions recur across LIBERO tasks rather than being limited to representative examples.

To assess sensitivity to the replacement choice, we repeat the attribution procedure using gray, blur, and noise interventions. Table II summarizes pairwise agreement across the four intervention kernels.

TABLE II  
ATTRIBUTION ROBUSTNESS ACROSS INTERVENTION KERNELS.
<table><tr><td>Metric</td><td>Pairwise range</td></tr><tr><td>Spearman correlation</td><td>0.734–0.816</td></tr><tr><td>Top-3 overlap</td><td>0.857–0.933</td></tr></table>

Across kernels, task-level Spearman correlations range from 0.734 to 0.816 and Top-3 overlap from 0.857 to 0.933. Although magnitudes depend on the replacement and may reflect distribution shift, important-block rankings remain consistent; we therefore emphasize ranking structure rather than absolute attribution values.

## D. Closed-Loop Faithfulness

We evaluate whether attribution rankings reflect actual task outcomes using deletion and insertion [17]. For each method, the top-k ranked blocks are progressively selected and the policy is re-executed in closed loop from the same initial state. Deletion replaces the selected blocks and measures the resulting loss in full-information success, whereas insertion preserves only the selected blocks and measures success recovery from the empty-coalition condition. Table III reports partial area under the curve (pAUC) over the evaluated k values.

TABLE III  
CLOSED-LOOP FAITHFULNESS.
<table><tr><td>Method</td><td>Deletion pAUC</td><td>Insertion pAUC</td></tr><tr><td>IMPACT-VLA</td><td>0.810 [0.746, 0.861]</td><td>0.104 [0.050, 0.163]</td></tr><tr><td>Phase-restricted</td><td></td><td>0.796 [0.743, 0.846] 0.082 [0.035, 0.144]</td></tr><tr><td>LOO</td><td>0.780 [0.697, 0.851] 0.017 [0.002, 0.035]</td><td></td></tr><tr><td>Static Action Pert.</td><td></td><td>0.670 [0.548, 0.777] 0.030 [0.002, 0.062]</td></tr><tr><td>Random</td><td>0.278 [0.228, 0.325] 0.005 [0.001, 0.009]</td><td></td></tr></table>

IMPACT-VLA achieved the highest pAUC for both metrics. Paired improvements over Static Action Perturbation had confidence intervals excluding zero for both deletion and insertion, indicating that closed-loop attribution identifies task-critical information more faithfully than immediate action sensitivity.

Compared with LOO, IMPACT-VLA showed no clear deletion difference but higher insertion performance. Absolute insertion pAUC remained low across methods, so we interpret it as a relative ranking test under limited information. Differences from Phase-restricted Shapley were not statistically clear; however, the global formulation is necessary to evaluate how later-block contributions depend on earlier information availability, as analyzed in Sec. IV-E.

We further evaluate temporal structure using Modalityonly Shapley and Uniform query-aligned phases. For phaseresolved methods, block contributions are aggregated by modality before selecting the top modality. Table IV reports the top-modality removal effect, i.e., the decrease in closedloop success after replacing the modality identified as most important by each method.

TABLE IV  
STRUCTURAL ABLATION OF TEMPORAL PHASE RESOLUTION.
<table><tr><td>Method</td><td>Top-modality removal effect</td></tr><tr><td>Modality-only Shapley</td><td>0.915 [0.830, 0.980]</td></tr><tr><td>Uniform query-aligned phases</td><td>0.931 [0.854, 0.984]</td></tr><tr><td>Action-transition-aligned phases</td><td>0.948 [0.883, 0.993]</td></tr></table>

Uniform query-aligned phases outperformed Modalityonly Shapley, while Action-transition-aligned phases provided further improvement; both paired differences had confidence intervals excluding zero. Thus, temporal resolution is useful, and action-transition-aligned boundaries better identify phase-specific modality importance under this removalbased evaluation.

## E. Cross-Phase Interaction and Recovery

Phase-dependent attribution does not directly reveal nonadditive dependencies between blocks from different phases. We therefore compute interactions for all ordered crossphase block pairs under the primary mean intervention. Fig. 4 shows task-level interactions between early and late modalities. Positive interactions are interpreted as consistent with complementary relations, whereas negative interactions are interpreted as consistent with redundancy or suppressive dependencies.

![](images/f6472f163ca419a75ac84d5cdcda383fd16271b72e3f11ee40266703696f0c97.jpg)  
Fig. 4. Cross-phase modality interactions.

The wrist-vision-to-proprioception pair exhibited the strongest positive interaction, while the task-macro fraction of negative cross-phase pairs was approximately 31%. Across all 1,046 cross-phase pairs with negative screening interaction under the mean kernel (30 tasks), the task-macro laterblock marginal gain increased from 0.025 [0.017, 0.033] to 0.082 [0.072, 0.093] when the early block was replaced (paired ∆=+0.057 [0.052, 0.063]), corresponding to a 3.3× ratio.

Table V summarizes held-out validation. Complementary interactions remained positive with high sign consistency, indicating that earlier information reliably enhanced later contributions. Negative interactions persisted at the aggregate level but showed weaker edge-level consistency, suggesting more context-dependent redundancy. Negative controls remained near zero, confirming that these dependencies are not artifacts of arbitrary block pairing. These results support reproducible cross-phase dependencies, particularly complementarity, but do not reveal how early information loss propagates through execution.

TABLE V  
HELD-OUT INTERACTION VALIDATION.
<table><tr><td>Group</td><td>Edges</td><td>Held-out interaction</td><td>Sign agreement</td></tr><tr><td>Complementary</td><td>52</td><td>+0.182 [0.136, 0.226]</td><td>0.900 [0.780, 1.000]</td></tr><tr><td>Redundant</td><td>38</td><td>-0.066 [-0.106, -0.029]</td><td>0.520 [0.320, 0.720]</td></tr><tr><td>Negative control</td><td>60</td><td>+0.006 [-0.009, 0.021]</td><td></td></tr></table>

Fig. 5 tracks intervention-induced trajectory deviation across behavioral progress. Complementary interactions showed substantial deviation after the early phase followed by partial reduction near the end, possibly reflecting corrective behavior using subsequent observations, whereas mean deviation for redundant interactions increased from 2.4 cm to 10.5 cm. This shows that early information loss can propagate across subsequent closed-loop execution and exhibit different dynamics depending on the interaction type.

![](images/e8a49475a8e2cf1e75749d0c65369cd5b8f85d19db8308fb05ef819bfab2b79e.jpg)  
Fig. 5. Phase-wise propagation of intervention-induced trajectory deviation.

In Table VI, we evaluate Behavioral and Functional Recovery for candidates passing the pre-damage gate. Behavioral Recovery measures trajectory-level reduction in deviation, whereas Functional Recovery measures task-level compensation by later-phase information; Both layers denote cases satisfying both criteria. After gating, 42 complementary edges from 30 tasks, 29 redundant edges from 29 tasks, and 46 negative-control edges from 29 tasks remained.

Behavioral Recovery was frequent even in the negativecontrol group, so trajectory re-convergence alone is insufficient evidence of functional compensation. The task-macro Functional Recovery rate was 81.3% in the redundant group after the pre-damage gate; given its full-coalition definition, we interpret this as context-specific compensation rather than an independent interaction finding. The task-macro rate of Functional Recovery without Behavioral Recovery was 62.5% in this group, indicating that later information can support task success without restoring the reference trajectory.

TABLE VI  
RECOVERY RATES ACROSS INTERACTION GROUPS.
<table><tr><td>Recovery metric</td><td>Complementary</td><td>Redundant</td><td>Negative control</td></tr><tr><td>Behavioral recovery</td><td>20.0% [8.0, 32.0]</td><td>22.9% [8.3, 39.6]</td><td>50.0% [31.3, 68.8]</td></tr><tr><td>Functional recovery</td><td>6.0% [0.0, 16.0]</td><td>81.3% [64.6, 95.8]</td><td>18.8% [8.3, 31.3]</td></tr><tr><td>Both layers</td><td>2.0% [0.0, 6.0]</td><td>18.8% [6.3, 33.3]</td><td>6.3% [0.0, 12.5]</td></tr></table>

## V. CONCLUSION

We proposed IMPACT-VLA for analyzing when multimodal inputs contribute during closed-loop VLA execution and how their effects interact across behavioral phases. Experiments on LIBERO showed phase-dependent modality contributions and greater closed-loop faithfulness than Static Action Perturbation, while differences from Phase-restricted Shapley were not statistically clear. Later-phase contributions depended on earlier information availability, and functional compensation could occur without trajectory re-convergence, motivating temporally structured closed-loop analysis rather than independent local sensitivity.

The evaluation is limited to OpenVLA-OFT in LIBERO, with phase structure defined from one successful reference execution per task; results should therefore be interpreted relative to those references. Future work should test alternative references, VLA architectures, failure trajectories, and real robots while reducing coalition-evaluation cost.

## REFERENCES

[1] B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, et al., “Rt-2: Visionlanguage-action models transfer web knowledge to robotic control,” in Proceedings of the 7th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 229, 2023, pp. 2165–2183.

[2] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, et al., “Openvla: An open-source vision-language-action model,” in Proceedings of the 8th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 270, 2025, pp. 2679–2713.

[3] H. Ichiwara, H. Ito, K. Yamamoto, H. Mori, and T. Ogata, “Modality attention for prediction-based robot motion generation: Improving interpretability and robustness of using multi-modality,” IEEE Robotics and Automation Letters, vol. 8, no. 12, pp. 8271–8278, 2023.

[4] J. Bento, P. Saleiro, A. F. Cruz, M. A. T. Figueiredo, and P. Bizarro, “Timeshap: Explaining recurrent models through sequence perturbations,” in Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2021, pp. 2565–2573.

[5] D. Beechey, T. M. Smith, and O. S¸ ims¸ek, “Explaining reinforcement<sup>¨</sup> learning with shapley values,” in Proceedings of the 40th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 202, 2023, pp. 2003–2014.

[6] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn, “Learning fine-grained bimanual manipulation with low-cost hardware,” in Proceedings of Robotics: Science and Systems, 2023.

[7] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song, “Diffusion policy: Visuomotor policy learning via action diffusion,” The International Journal of Robotics Research, vol. 44, no. 10-11, pp. 1684–1704, 2025.

[8] M. Sundararajan, K. Dhamdhere, and A. Agarwal, “The shapley taylor interaction index,” in Proceedings ofthe 37th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 119, 2020, pp. 9259–9268.

[9] S. Greydanus, A. Koul, J. Dodge, and A. Fern, “Visualizing and understanding atari agents,” in Proceedings of the 35th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 80, 2018, pp. 1792–1801.

[10] J. Liu, C. Li, G. Wang, L. Lee, K. Zhou, S. Chen, C. Xiong, J. Ge, R. Zhang, and S. Zhang, “Self-corrected multimodal large language model for end-to-end robot manipulation,” arXiv preprint arXiv:2405.17418, 2024.

[11] B. Haon, K. C. Stocking, I. Chuang, and C. Tomlin, “Mechanistic ¨ interpretability for steering vision-language-action models,” in Proceedings of the 9th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 305, 2025, pp. 2743–2762.

[12] A. Bhardwaj, E. W. Duan, P. Dan, W.-C. Ma, and P. Culbertson, “Decoding task progress from vla representations,” arXiv preprint arXiv:2608.13474, 2026.

[13] A. Swann, L. McGranahan, H. Buurmeijer, M. K. III, and M. Schwager, “Sparse autoencoders reveal interpretable and steerable features in vla models,” arXiv preprint arXiv:2603.19183, 2026.

[14] X. Jin, A. Chatterjee, P. Kumar, and R. Paleja, “Event-grounded sparse autoencoders for vision-language-action policies,” arXiv preprint arXiv:2605.17204, 2026.

[15] M. D. Zeiler and R. Fergus, “Visualizing and understanding convolutional networks,” in European Conference on Computer Vision, 2014, pp. 818–833.

[16] M. T. Ribeiro, S. Singh, and C. Guestrin, “Why should i trust you?: Explaining the predictions of any classifier,” in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016, pp. 1135–1144.

[17] V. Petsiuk, A. Das, and K. Saenko, “Rise: Randomized input sampling for explanation of black-box models,” in British Machine Vision Conference, 2018.

[18] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-cam: Visual explanations from deep networks via gradient-based localization,” in Proceedings of the IEEE International Conference on Computer Vision, 2017, pp. 618–626.

[19] M. Sundararajan, A. Taly, and Q. Yan, “Axiomatic attribution for deep networks,” in Proceedings of the 34th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 70, 2017, pp. 3319–3328.

[20] L. S. Shapley, “A value for n-person games,” in Contributions to the Theory of Games II. Princeton University Press, 1953, pp. 307–317.

[21] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in Advances in Neural Information Processing Systems, vol. 30, 2017.

[22] A. Nayebi, S. Tipirneni, C. K. Reddy, B. Foreman, and V. Subbian, “Windowshap: An efficient framework for explaining time-series classifiers based on shapley values,” Journal of Biomedical Informatics, vol. 144, p. 104438, 2023.

[23] S. Hooker, D. Erhan, P.-J. Kindermans, and B. Kim, “A benchmark for interpretability methods in deep neural networks,” in Advances in Neural Information Processing Systems, vol. 32, 2019.

[24] S. Krishnan et al., “Transition state clustering: Unsupervised surgical trajectory segmentation for robot learning,” The International Journal of Robotics Research, vol. 36, no. 13-14, pp. 1595–1618, 2017.

[25] S. Krishnan, R. Fox, I. Stoica, and K. Goldberg, “Ddco: Discovery of deep continuous options for robot learning from demonstrations,” in Proceedings ofthe 1st Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 78, 2017, pp. 418–437.

[26] T. Kipf et al., “Compile: Compositional imitation learning and execution,” in Proceedings ofthe 36th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 97, 2019, pp. 3418–3428.

[27] P. Fryzlewicz, “Wild binary segmentation for multiple change-point detection,” The Annals of Statistics, vol. 42, no. 6, pp. 2243–2281, 2014.

[28] H. Cho, “Change-point detection in panel data via double cusum statistic,” Electronic Journal of Statistics, vol. 10, no. 2, pp. 2000– 2038, 2016.

[29] J. Castro, D. Gomez, and J. Tejada, “Polynomial calculation of the´ shapley value based on sampling,” Computers & Operations Research, vol. 36, no. 5, pp. 1726–1730, 2009.

[30] B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone, “Libero: Benchmarking knowledge transfer for lifelong robot learning,” in Advances in Neural Information Processing Systems, vol. 36, 2023, pp. 44 776–44 791.

[31] M. J. Kim, C. Finn, and P. Liang, “Fine-tuning vision-language-action models: Optimizing speed and success,” in Proceedings of Robotics: Science and Systems, 2025.