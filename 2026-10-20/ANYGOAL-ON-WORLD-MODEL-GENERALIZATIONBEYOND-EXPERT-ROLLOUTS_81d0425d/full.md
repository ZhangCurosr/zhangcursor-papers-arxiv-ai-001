# ANYGOAL: ON WORLD MODEL GENERALIZATIONBEYOND EXPERT ROLLOUTS

Heejeong Nam<sup>1,2</sup> Divyat Mahajan<sup>3</sup> Quentin Le Lidec<sup>4</sup> Randall Balestriero<sup>1</sup> David Klindt<sup>2</sup> <sup>1</sup>Brown University <sup>2</sup>Cold Spring Harbor Laboratory <sup>3</sup>Mila, Universite de Montr´ eal´ <sup>4</sup>New York University

<sup></sup> Website <sup>§</sup> Code

## ABSTRACT

The goal of a world model is to capture the dynamics of an environment. This has the great appeal that it allows agents to plan and simulate a course of actions with the help of a world model. For these uses to be reliable, however, the learned dynamics must behave well beyond the specific state–action combinations observed in expert rollouts. Despite this, the relationship between data coverage, dynamics generalization, and downstream planning performance remains underexplored. We study this relationship through the lens of compositional generalization: can a world model recombine familiar state and action factors into valid dynamics? To answer this, we introduce AnyGoal, a benchmark of goal-shifted task variants built from widely used world-model environments, together with an open-source library. Each task is designed to be within reach of a world model that has learned the underlying dynamics: individual state factors and actions remain within their training support, but their combinations are underrepresented in expert trajectories. Across multiple environments and world-model families, we find that current models consistently exhibit failures under these recombinations. By examining dynamics and representation failures across model families, interaction regimes, and evaluation metrics, we study where current world models generalize and where challenges remain for goal-shifted prediction and planning. Together, this framework, benchmark, and analysis provide a basis for evaluating progress toward world models that generalize to new planning tasks.

Generalization Failure Beyond Training Goal

![](images/1ec9e49b21d9fa95041b3325c849c43bcb79ef6bd3b6fd0baa8d0f8d51037dcc.jpg)

![](images/718971486bb4205f1f781b827bc4e0d96a9e0505dbc66b03cd43d36ff6304796.jpg)  
Figure 1: Example of generalization failure under goal shift. A world model trained on Push-T performs poorly on Avoid-T, despite Avoid-T requiring only familiar 2D navigation and having individually familiar states and actions. In Push-T, action interventions reveal reduced model sensitivity in biased regions, suggesting that the model learns shortcuts rather than reusable dynamics.

Expert Rollouts Couple States, Objects, and Actions  
![](images/063a3754a73faa24465b4f5c5d945f50787c5d56a9ae6a67d70cafff077f528e.jpg)  
Expert Rollouts always move agent toward the block.  
Agent-Block Distance  
World Models cannot correctly recombine familiar action to familiar state.

## 1 INTRODUCTION

A key goal of physical intelligence is the ability to plan and perform new tasks. Routine tasks, such as driving to work, can be automatized and executed without much cognitive effort. Yet novel tasks requires us to plan and simulate the consequences of our actions. World models (Ha & Schmidhuber, 2018; Maes et al., 2026b; Ivashkov et al., 2026; Lu et al., 2026; Zhu et al., 2024; Zhou et al., 2025) aim to learn action-conditioned dynamics of environment by predicting how the world evolves under actions rather than task-specific behavior (Sutton & Barto, 2018). Therefore, a good world model should not merely memorize behavioral biases present in the training distribution. However, indistribution evaluation offers limited evidence of reusable dynamics. Figure 1 illustrates a simple but strong example of this caveat. In Push-T, the agent must push a block to a target pose, so expert rollouts strongly couple far-from-block states with actions that move the agent toward the block. A model trained on Push-T may therefore learn this rollout regularity rather than the action-conditioned effect of the supplied action, as reflected by the action-sensitivity plot in Fig. 1.When evaluated on Avoid-T, which requires avoiding the block using familiar 2D navigation, the model fails.

![](images/f1f5e9d6a8e622070266bf4ce631d3ef87ad27c462b8e8ab440bd53203296c48.jpg)  
Figure 2: Overview of AnyGoal. Each task preserves the environment dynamics of the corresponding base tasks and keep states and actions individually within training coverage. Throughout the paper, AnyGoal is used solely as a test set for evaluating generalization.

If the training data fully covered the relevant joint state–action space, then a model could in principle predict correctly on that support. However, such coverage is unrealistic in many control problems, especially when the relevant space is high-dimensional. First, offline datasets for world-model training are often collected from goal-directed policies that reach task-relevant states (Burda et al., 2019), which can introduce strong state–action correlations. One possible remedy is to mix data from multiple policies (Zhao et al., 2026). However, even with such mixtures, it can be difficult to determine whether the resulting dataset sufficiently covers the relevant state–action space, especially when the true reachable set is unknown. Even when a coverage gap is identified, collecting targeted rollouts to fill it can be difficult in offline settings. This makes it necessary to study generalization under incomplete coverage, rather than assuming that data scale alone will resolve the issue.

Generalization of world models has been studied to some extent. Closest to our setting, Hansen & Wang (2026) study hallucination detection and mitigation in world models, while we evaluate generalization in a statistically controlled space with explicit state-action support checks. Other OOD evaluations introduce states or factors outside the training support (Maes et al., 2026a; Wang et al., 2026; Li et al., 2025). Such settings are valuable for testing robustness to novel factors or extrapolation, but they may go beyond the generalization expected from a reusable dynamics model.

What remains missing is a systematic evaluation of the generalization that a good world model should support. First, we formalize this criterion as compositional generalization (Wiedemer et al., 2024b) and link it to goal shifts and behavior bias in the training data. We further show that the bias is empirically realized through targeted action interventions on familiar but under-covered state– action combinations, eliciting degenerate model responses. We then propose AnyGoal (Fig. 2), a benchmark of goal-shifted task variants built from widely used world-model environments. To remain in the compositional-generalization regime, AnyGoal preserves the observed marginal support of states, actions, task-relevant factors, and physical rules, while changing the joint combinations induced by the training behavior distribution and targeting under-covered state–action regions. With AnyGoal, we test whether current world models can generalize to new tasks without requiring extrapolation to unseen states, actions, or visual factors. Across this controlled setting, we observe broad failures across modeling paradigms, including generative world models (Zhu et al., 2024; Lu et al., 2026) and JEPA-style architectures (Maes et al., 2026b; Ivashkov et al., 2026; Zhou et al.,

2025). Our results then characterize this degradation across several axes, showing that failures can arise from both dynamics prediction and representation learning, and that their severity depends on task complexity, evaluation metric, and model family. Finally, we outline a theoretical direction toward compositional generalization for world models, framing what future models may need in order to move beyond behavior-biased rollouts.

## 2 RELATED WORKS

Compositional generalization diagnostics and benchmarks. Compositional generalization tests whether a model can recombine familiar components into unseen configurations, a key challenge when possible compositions exceed finite data coverage. Early sequence-to-sequence benchmarks such as SCAN (Lake & Baroni, 2018) and CFQ (Keysers et al., 2020) exposed this failure mode through controlled compositional splits. The same diagnostic was subsequently adopted in vision and multimodal settings, with benchmarks such as CLEVR (Johnson et al., 2017), Winoground (Thrush et al., 2022), and CVR (Zerroug et al., 2022) testing whether familiar objects, attributes, and relations can be composed correctly. These failures remain prominent in modern large-scale models: ConceptMix (Wu et al., 2024) shows that text-to-image models degrade as prompts require more familiar concepts to be composed jointly, while recent vision–language work continues to reveal failures in object–attribute and relational composition (Assouel et al., 2026). At the same time, compositional evaluation is becoming broader and more systematic, with scalable protocols for probing generalization across controlled compositions (Camposampiero et al., 2026), and similar shifts being used as diagnostics in mechanistic interpretability (Pacela et al., 2026). AnyGoal brings this perspective to world models, where goal-directed trajectories naturally induce restricted joint coverage over otherwise familiar physical states and actions.

World-model generalization and support mismatch. Recent world-model work has studied generalization through object-centric compositional dynamics (Zhao et al., 2022), composable causal components for unseen environments (Wang & Huang, 2025), multi-task world models (Georgiev et al., 2025), environment-level evaluation queries (Warrier et al., 2026), actionconditioned planning (Zhang & Du, 2026), and coverage-related hallucination failures (Hansen & Wang, 2026). However, less is known about whether expert-trained world models generalize to new goal-induced recombinations of familiar states and actions. This question is closely connected to support mismatch in offline reinforcement learning, where fixed behavior datasets can lead methods to fail when evaluating poorly covered state–action regions (Fujimoto et al., 2019; Kumar et al., 2019; Levine et al., 2020). Offline RL addresses support mismatch by constraining or regularizing policies toward covered regions (Gao et al., 2025b;a; Laroche et al., 2019; Chen et al., 2025; Kumar et al., 2020; Tang et al., 2025). Model-based offline RL extends this issue to learned dynamics, motivating uncertainty- or pessimism-based objectives for poorly covered regions (Yu et al., 2020; Kidambi et al., 2020; Yu et al., 2021; Rigter et al., 2022). D4RL further shows that dataset composition strongly affects offline policy improvement (Fu et al., 2021). AnyGoal is complementary to these approaches: it evaluates whether the world model learns reusable dynamics under goal shifts.

## 3 COMPOSITIONAL GENERALIZATION FOR WORLD MODELS

When should we say that a world model has learned reusable dynamics, rather than only the training behavior? In this section, we formalize this criterion as compositional generalization.

Notation. We use uppercase letters for random variables and lowercase letters for their realizations. Let $\boldsymbol { z } _ { t } = ( z _ { 1 , t } , \dots , z _ { K , t } )$ denote true underlying state of the environment, where each component corresponds to a semantic factor of variation<sup>1</sup> (Scholkopf et al., 2021), such as object position¨ or end-effector pose. We model observations as $s _ { t } = f ( z _ { t } )$ , where f maps $z _ { t }$ to the RGB observation $s _ { t }$ presented to the world model. An action-paired trajectory $\tau = ( s _ { 0 } , a _ { 0 } , s _ { 1 } , a _ { 1 } , \dots , s _ { H } )$ can therefore equivalently be represented through the corresponding sequence of physical factors and actions, $( z _ { 0 } , a _ { 0 } , z _ { 1 } , a _ { 1 } , \dotsc , z _ { H } )$ , where H denotes horizon.

## 3.1 ACTION-CONDITIONED PREDICTION CAN DEGENERATE INTO BEHAVIOR PREDICTION

An action-conditioned world model should learn a transition predictor $\hat { T }$ that receives ${ s _ { t } } ^ { 2 }$ and an action ${ { a } _ { t } } ,$ and predicts how the observation changes under the supplied action, $\hat { T } ( s _ { t } , a _ { t } )$ ≈ $s _ { t + 1 }$ We consider the common offline setting in which data are collected by a behavior policy π, so that the observed action satisfies $a _ { t } \sim \pi ( \cdot \mid s _ { t } )$ . Thus, the observed transitions reflect the state–action pairings induced by π, rather than arbitrary interventions at each state:

$$
T ^ { \pi } ( s _ { t + 1 } \mid s _ { t } ) = \int T ( s _ { t + 1 } \mid s _ { t } , a ) \pi ( a \mid s _ { t } ) d a .\tag{1}
$$

A predictor trained only on such data may degenerate into an action-ignorant model for state regions with restricted conditional action support, fitting the observed transitions by approximating

$$
{ \hat { T } } ( s _ { t + 1 } \mid s _ { t } , a _ { t } ) \approx T ^ { \pi } ( s _ { t + 1 } \mid s _ { t } ) ,\tag{2}
$$

where the dependence on the supplied action $a _ { t }$ is effectively lost.

## 3.2 COMPOSITIONAL DISTRIBUTION SHIFTS IN WORLD MODELS

Now, let $g _ { \mathrm { t r } }$ and $g _ { \mathrm { t e } }$ denote the training and test goals. Each goal induces a behavior policy, $\pi _ { \mathrm { t r } }$ or $\pi _ { \mathrm { t e } } ,$ which generates actions along the same environment dynamics T. We denote the resulting trajectory distributions by $p _ { \mathrm { t r } }$ and $p _ { \mathrm { t e } } ,$ , respectively. We define the supports of physical factors and actions observed across trajectories, writing the training joint state–action support as

$$
\mathcal { C } ^ { \mathrm { t r } } : = \bigcup _ { t = 0 } ^ { H - 1 } \operatorname* { s u p p } _ { p _ { \mathrm { t r } } } \bigl ( Z _ { 1 , t } , \dotsc , Z _ { K , t } , A _ { t } \bigr ) .\tag{3}
$$

We also define the marginal support of each factor as $\begin{array} { r } { \mathcal { Z } _ { k } ^ { \mathrm { t r } } : = \bigcup _ { t = 0 } ^ { H - 1 } \operatorname { s u p p } _ { p _ { \mathrm { t r } } } ( Z _ { k , t } ) } \end{array}$ , and the action support as $\begin{array} { r } { \mathcal { A } ^ { \mathrm { t r } } : = \bigcup _ { t = 0 } ^ { H - 1 } \operatorname { s u p p } _ { p _ { \mathrm { t r } } } ( A _ { t } ) } \end{array}$ . These supports pool over temporal position: a physical factor is considered observed during training if it occurs at any point along a trajectory drawn from $p _ { \mathrm { t r } } .$ The test supports ${ \mathcal { C } } ^ { \mathrm { t e } } , { \mathcal { Z } } _ { k } ^ { \mathrm { t e } }$ , and $\mathcal { A } ^ { \mathrm { t e } }$ are defined analogously under $p _ { \mathrm { t e } }$ . We define the combinatorial space as the Cartesian product of the marginal training supports:

$$
\mathcal { C } ^ { \times } : = \left( \prod _ { k = 1 } ^ { K } \mathcal { Z } _ { k } ^ { \mathrm { t r } } \right) \times \mathcal { A } ^ { \mathrm { t r } } .\tag{4}
$$

![](images/554702880b1c6dec0f9c1783ff88bea0b1ae44c2952ba030dc320aa859b1d34f.jpg)  
Figure 3: Compositional support structure. ${ \bar { \mathcal { C } } } ^ { \mathrm { t e } } \subseteq { \mathcal { C } } ^ { \times }$ and ${ \dot { C } } ^ { t e } \not \subset { \mathcal { C } } ^ { \mathrm { t r } } { \dot { 1 } }$ s necessary.

We study compositional distribution shifts (Liu et al., 2023; Mahajan et al., 2025) where the underlying transition dynamics are invariant across the train and test distribution, i.e., $T _ { \mathrm { t r } } ( s _ { t + 1 } ~ $ $s _ { t } , a _ { t } ) = T _ { \mathrm { t e } } ( s _ { t + 1 } \ | \ s _ { t } , a _ { t } )$ , but the test support introduces state, action pairs not present in the training support $( { \mathcal { C } } ^ { \mathrm { t e } } ~ \not \subset ~ { \mathcal { C } } ^ { \mathrm { t r } } )$ . Any-Goal targets the combinatorial gap, regions of the combinatorial space $\mathcal { C } ^ { \bar { \times } }$ that are absent from the training joint support ${ \mathcal { C } } ^ { \mathrm { t r } }$ . Since test trajectories need not cover the entire gap or lie only within it, we require ${ \mathcal { C } } ^ { \mathrm { t e } } \subseteq { \mathcal { C } } ^ { \times }$ and choose goals with substantial mass in under-covered regions of ${ \mathcal { C } } ^ { \times }$ . This means that the test distribution does not introduce novel physical factors or actions, only novel combinations of observed physical factors and actions. Importantly, training-distribution accuracy does not identify the extension of the learned transition function ${ \mathrm { t o } } { \mathcal { C } } ^ { \times } \backslash { \mathcal { C } } ^ { \mathrm { t r } } ;$ : predictors that agree on the training support may disagree on novel compositions (Wiedemer et al., 2024b). This motivates evaluating the combinatorial gap directly. We discuss more about why it is an essential test in Appendix A.

![](images/87386c73e6739ffad4f335ee9388eef74dcfc9846aeb494dbe665863e8ebe3b8.jpg)  
Figure 4: AnyGoal targets the combinatorial gap. (a) Example factor-wise support check for Avoid-T relative to Push-T using KDE. Darker regions indicate higher density, and gray regions mark areas whose density is below 1% of the peak reference density. (b) Estimated overlap between each target variant and the base-task training distribution via KDE.

## 4 ANYGOAL: CONTROLLED GOAL-SHIFT EVALUATION

We introduce AnyGoal, a benchmark for testing whether world models generalize beyond the goaldirected trajectories on which they are trained. AnyGoal uses the controlled goal shifts discussed in Sec. 3.2 and builds on four widely used world-model tasks: TwoRoom (Sobal et al., 2025), Push T (Chi et al., 2025), DMC Manipulator Insert-Peg (Tassa et al., 2018), and OGBench Cube (Park et al., 2025). AnyGoal contains eight variants: one free-space variant per base task and one or two contact-rich variants except TwoRoom. We release 400 episodes per variant, with data-collection tools for generating additional datasets. Figure 2 summarizes the tasks, with details in Appendix B.

## 4.1 ANYGOAL TARGETS THE COMBINATORIAL GAP

Validating $c ^ { \mathsf { t e } } \subseteq c ^ { \times }$ For each base–target task pair, we apply three factor-wise support checks to verify that the target distribution remains marginally in-distribution relative to its reference distribution. The range check requires all query samples to lie within the coordinate-wise reference range. The Mahalanobis check uses the reference mean and covariance, thresholds distances at the reference 0.99-quantile, and requires at least 95% of query samples to fall below it. The KDE check thresholds reference densities at the 0.01-quantile and requires at least 95% of query samples to lie above it. The final support verdict is the conjunction of all three checks. All AnyGoal variants are designed to pass these factor-wise checks, which indicates that all tasks lie within the combinatorial space of training data. Appendix C provides the full details of pass-criterion definitions, the verified factor groups for each base task, and the complete score of range, Mahalanobis, and KDE results.

Measuring combinatorial-gap coverage of $c ^ { \mathbf { t e } }$ We then estimate how much each target distribution overlaps with the base-task joint support. Unlike the factor-wise checks above, this joint test is meant to diagnose the combinatorial gap: individual factors may be marginally familiar while their full state–action combination is rare under the training distribution. For each base–target pair, we concatenate the action, agent state, and task-object or target state, fit a full-dimensional Gaussian KDE to the reference vectors, and set the density threshold to the empirical 0.01-quantile of log-densities. We then report the fraction of target vectors whose log-density exceeds this threshold. Lower joint scores indicate less overlap with the base-task joint distribution and greater coverage of the combinatorial gap. Full vector definitions and KDE implementation details are in Appendix D.

## 4.2 RELATIVE EVALUATION AGAINST BASE TASKS

We evaluate rollout prediction error and downstream MPC planning. Models are trained only on the base dataset and evaluated on both base and goal-shifted trajectories to measure relative changes. Rollout-error changes more directly diagnose whether action-conditioned dynamics generalize to unseen state–action combinations. However, planning success remains important. For JEPA-style models, it provides complementary evidence that the representation retains task-relevant information. For generative models, low rollout error can miss small but task-critical observation changes. We therefore report both metrics over seeds 0, 1, 2. Evaluation details are in Appendix H

Table 1: End-to-End World models and oracle state predictors evaluated in our experiments.
<table><tr><td>Model</td><td>Family</td><td>Representation</td><td>Action conditioning</td></tr><tr><td colspan="4">End-to-End World Models</td></tr><tr><td>LeWorldModel</td><td>JEPA</td><td>Learned latent</td><td>AdaLN</td></tr><tr><td>Sensorimotor WM</td><td>JEPA</td><td>Learned latent</td><td>AdaLN</td></tr><tr><td>DINO-WM</td><td>JEPA</td><td>Frozen DINO features</td><td>Feature concatenation</td></tr><tr><td>DriftWorld</td><td>Generative</td><td>Pixels</td><td>Conditional generator</td></tr><tr><td>IRASim</td><td>Generative</td><td>Frozen SDXL VAE</td><td>DiT conditioning</td></tr><tr><td colspan="4">Predictor-Only Models</td></tr><tr><td>AdaLN</td><td>Transformer</td><td>Oracle state</td><td>AdaLN</td></tr><tr><td>Action-Concatenated</td><td>Transformer</td><td>Oracle state</td><td>Feature concatenation</td></tr><tr><td>Toy-DriftWorld</td><td>Generative</td><td>Oracle state</td><td>Conditional generator</td></tr></table>

Prediction Error. We measure MSE in each model’s native prediction space, which may be a learned latent space, oracle state space, or pixel space depending on the model class. We evaluate autoregressive prediction over horizons of 5 and 25 environment steps, with frame skip of 5. For each seed, we sample 100 episodes and average over all valid starting points within each episode.

MPC Planning Success Rate. We evaluate downstream control using the same CEM-based MPC implementation from Stable-WorldModel (Maes et al., 2026a) for all models. We use identical planning configurations, with a planning horizon of 25 environment steps. The full 25-step action sequence is executed before replanning.

## 4.3 ANYGOAL LIBRARY AND EVALUATION INTERFACE

We release AnyGoal as an open-source Python library for reproducible goal-shift evaluation. The library provides dataset access, task-specific data collection, and standardized rollout-prediction and MPC-planning evaluation across supported world models. Its unified CLI allows researchers to evaluate the same base-trained model on both the original task and unseen goals, with explicit model, dataset, and seed selection. The library also exposes Python APIs for dataset collection, model training, and evaluation. We provide examples and reproducibility instructions in Appendix K, with com plete documentation available online.

```shell
# Original training goal
anygoal eval planning --model lewm --dataset pusht --seed 0
# Unseen goal
anygoal eval planning --model lewm --dataset avoidt --seed 0
```  
Figure 5: The same base-trained model is evaluated on Push-T and Avoid-T using the standardized planning interface.

## 5 EXPERIMENT AND RESULTS

We evaluate a diverse model suite spanning JEPA-style and generative objectives, learned and frozen visual encoders, and action-conditioning mechanisms (Tab. 1). To separate dynamics learning from visual representation failure, we also include oracle-state predictors which replace visual encoders with oracle states. For consistency, all models use frame skip 5 and one-step prediction. Model details and oracle-state variables for each environment are provided in Appendix F, and the full numerical results are summarized in Appendix I.1.

End-to-End World Models. LeWorldModel (LeWM) (Maes et al., 2026b) is an end-to-end JEPA that jointly learns a visual encoder and an action-conditioned transformer predictor from pixels. It uses SIGReg (Balestriero & LeCun, 2025) for anti-collapse regularization and conditions the predictor through AdaLN (Peebles & Xie, 2023). Sensorimotor World Model (SMWM) (Ivashkov et al., 2026) uses the same encoder–predictor family as LeWM with an inverse-dynamics objective that encourages representations to retain controllable changes. DINO-WM (Zhou et al., 2025) decouples perception from dynamics by using a frozen DINOv2 (Oquab et al., 2024) patch encoder and predicting future patch representations, with actions concatenated to the patch tokens. DriftWorld (Lu et al., 2026) is a generative model that uses an action-conditioned spatial–temporal U-Net to predict future observations directly from observation history, actions, and noise. IRASim (Zhu et al., 2024) is a latent generative model that encodes observations with a frozen pretrained VAE and uses a Diffusion Transformer (Peebles & Xie, 2023) to predict future latents conditioned on action trajectories.

![](images/44e93e1374ece9671a150e01411dea3eadde608560fc54bec019dff74864ad23.jpg)

![](images/d36fe5e13aafbe7859dd30ff4ce4ef9baae8d6cb50d5a7b0ea9a4b6609ca8451.jpg)  
Figure 6: Oracle-state predictors reveal dynamics-level failures. Even with task-relevant state factors provided directly, goal shifts still degrade prediction and planning, showing that the AnyGoal gap cannot be reduced to visual representation failure. See Appendix I.1for absolute metrics.

Oracle-State Predictor. AdaLN predictor follows the architecture used by LeWM and SMWM. Action-Concatenated predictor follows the DINO-WM predictor, where an embedded action is concatenated to the patch representation before being processed by a causal transformer. Toy-DriftWorld is adapted from DriftWorld’s action-conditioned generative backbone. We use a reduced DriftWorld style model and replace pixel inputs and outputs with true state inputs and next-state targets.

## 5.1 DIMENSIONS OF GOAL-SHIFT PERFORMANCE

Goal-shift performance cannot be summarized by a single number. The same degradation can reflect different sources depending on whether it appears in prediction or planning, in latent or pixel space, in free-space or contact-rich tasks, and with or without oracle state access. We therefore organize the results around five recurring dimensions for interpreting the results:

1. Failure source: Predictor failures can be isolated from encoder failures via oracle-state predictors.

2. Evaluation metric: Rollout prediction and planning are showing different aspects.

3. Task complexity: Free-space and contact-rich tasks impose different dynamics challenges.

4. Representation quality: Learned representations sometimes obscure task-relevant dynamics, although they sometimes can improve planning geometry.

5. Prediction space: JEPA-style and generative models show distinct trends, but comparisons should be interpreted cautiously because prediction errors are measured in different spaces.

## 5.2 DYNAMICS GENERALIZATION BEYOND REPRESENTATION FAILURE

To examine dynamics generalization, we use the oracle states provided by AnyGoal, which expose task-relevant factors for action-conditioned next-step prediction and remove the need to infer them from observations under goal shifts. Overall, oracle-state predictors show that the AnyGoal gap cannot be reduced to visual representation failure.

Oracle-state predictors reveal an interaction-dependent dynamics gap. Figure 6 shows that removing visual abstraction does not yield a uniform outcome across AnyGoal variants. In freespace variants such as TR-Route, Avoid-T, Avoid-Peg, and Cube-Hop, MPC performance is preserved or improves relative to the corresponding base task. In contrast, contact-rich variants consistently degrade, despite using the same privileged state inputs. This suggests that the difficulty is not merely access to the relevant state factors (Klindt et al., 2026; Shi et al., 2026), but whether the goal shift requires precise object interaction. This asymmetry is consistent with prior work showing that compositional generalization requires constraints on the learned function. Additive or interactionfree settings are especially favorable, since factor-specific mechanisms can be learned from observed compositions and reused under novel recombinations (Dong & Ma, 2022; Lachapelle et al., 2023; Wiedemer et al., 2024a; Mahajan et al., 2025). This intuition aligns with the independent causal mechanisms principle, under which reusable mechanisms remain autonomous as their joint occurrence changes (Parascandolo et al., 2018). Contact-rich variants depart from this favorable regime: action effects may no longer decompose into reusable factor-specific mechanisms. Compositional generalization may still be possible, but only with additional structure on these interactions (Brady et al., 2025). The same asymmetry appears in rollout prediction. Contact-rich variants show much larger error increases, often by one to three orders of magnitude, even with oracle states. Free-space variants degrade far less, and in some cases match or improve on the base-task error. Thus, removing representation learning does not make goal-shifted dynamics prediction uniformly reliable.

![](images/fc6550e084b2cfcc81d7509919cc7a4c646805af9cbfa05ca452e019403d2276.jpg)

![](images/f6dc55d7d65f1d9e3126d55d613c719b886edb3413989ced57f0d87efc01dbd1.jpg)  
Figure 7: End-to-end degradation reflects representation effects. Compared with oracle-state predictors, end-to-end models degrade more under goal shifts, suggesting additional failures from observation-level shift and learned representation geometry. See Appendix I.1for absolute metrics.

Metric alignment depends on task complexity. We additionally observe that rollout error and planning performance sometimes diverge in free-space variants where planning can tolerate imperfect prediction. The closer agreement in contact-rich settings may reflect the structure of the planning problem. Rollout error measures whether the model predicts the full trajectory accurately, whereas MPC only needs to find some action sequence that reaches the goal, regardless of the exact intermediate path. Free-space tasks often admit many such trajectories, while contact-rich tasks allow fewer viable sequences because success also depends on the correct object interaction.

## 5.3 REPRESENTATION EFFECTS IN END-TO-END WORLD MODELS

End-to-end world models can degrade on both metric under the goal shift even when oracle-state predictors remain stable. Such cases show that the AnyGoal gap is not solely a failure of actionconditioned dynamics, but can also reflect encoder failure under unseen factor combinations. Due to space constraints, we discuss two main observations, and more are in Appendix I.3.

![](images/493b55c7e07949b243807dedb89a938f4f8ea8f04c4361871e898853f40975de.jpg)  
(a) TwoRoom-Route Evaluation  
(b) Examples of TwoRoom 2D t-SNE Projection  
Figure 8: Representation geometry matters under goal shifts. (a) The observations remain visually in support, but end-to-end models degrade while oracle-state predictors remain stable. (b) The 2D t-SNE projections illustrate that learned representations can organize the same underlying agent positions with substantially different geometry, which can affect prediction and planning.

Factor recombination can induce observation-level shifts. AnyGoal controls generalization at the level of physical factors, but end-to-end models observe rendered inputs $s = f ( z )$ . Recombining marginally familiar factors can therefore create observations outside the joint visual support of expert rollouts, adding an encoder-inference challenge beyond dynamics prediction. (See Appendix I.2 for details.) Crucially, correct factor inference on training observations need not extend to unseen compositions. Even when factors are identifiable in distribution, an encoder may fail to invert novel recombined observations without compositional-consistency constraints (Wiedemer et al., 2024a). Thus, end-to-end world models face a representation-level generalization problem before dynamics prediction is considered. End-to-end models must therefore infer task-relevant factors from observations generated by unseen factor combinations, in addition to learning dynamics.

Latent geometry shapes degradation under goal shifts. TwoRoom provides a useful case where end-to-end degradation cannot be attributed to an observation-level shift. Because agent position is the only changing factor and TR-Route positions remain marginally within the TwoRoom training support, the corresponding observations stay visually in support, enabling a controlled comparison between end-to-end models and oracle-state predictors. Figure 8 (a) summarizes results from TwoRoom and TR-Route across different model families. Even without an apparent visual shift, end-to-end models degrade whereas oracle-state predictors remain stable. This indicates that representation quality still matters: the learned latent space may fail to preserve the agent location precisely, introduce unnecessary dimensions or nuisance variation, or organize states in a geometry that is unfavorable for prediction and planning. Thus, even visually in-support goal shifts can fail if the learned latent geometry is not aligned with the task-relevant state geometry.

## 5.4 DIAGNOSING BEHAVIOR-POLICY SHORTCUTS THROUGH ACTION SENSITIVITY

As discussed in Sec. 3.1, a world model trained on expert rollouts may exploit correlations between states and the actions typically taken by the expert, rather than learning the effects of the supplied action. We investigate whether this failure mode is reflected in the learned models by measuring their sensitivity to interventional action inputs.

Interventional action sensitivity. To test whether the model responds to the supplied action rather than relying on the behavior policy’s preferred transition, we intervene on the action input while holding the state fixed. For each training state s with observed action a, we replace a with actions from a fixed interventional set $\mathcal { A } _ { \mathrm { c f } }$ and measure the resulting change in the model’s predicted representation:

$$
S _ { \theta } ( s ) = \frac { 1 } { \vert \mathcal { A } _ { \mathrm { c f } } \vert } \sum _ { a ^ { \prime } \in \mathcal { A } _ { \mathrm { c f } } } \Vert \phi _ { \theta } ( s , a ) - \phi _ { \theta } ( s , a ^ { \prime } ) \Vert _ { 2 } ,\tag{5}
$$

where $\phi _ { \theta } ( s , a )$ denotes the model representation used for the sensitivity measurement. A smaller score indicates that the model is less sensitive to changes in the supplied action, although this does not necessarily imply incorrect predictions. We evaluate this score across training states and relate it to an environment-specific coordinate along which expert behavior is strongly biased. More details are in Appendix E.

Goal bias and reduced action sensitivity. Figure 9 shows that action sensitivity varies systematically with the state regions associated with expert behavior. In TwoRoom, expert trajectories favor motion toward the narrow passage. In Push-T, the agent typically approaches the block before pushing it. In Insert-Peg, the agent carries the grasped peg toward the target and in Cube, vertical actions are prevalent at low end-effector heights. Across these environments, we observe reduced sensitivity to interventional actions in regions where the training behavior is strongly constrained by the goal and these observations are consistent with the degeneracy described in Sec. 3.1.

![](images/a3855ee9b36515bf28c3f60f3a58c859df1e5fc2b453ae7037257f0654deb653.jpg)  
Figure 9: Goal-biased training support is associated with reduced action sensitivity. Each panel plots model action sensitivity against an environment-specific bias coordinate induced by the expert policy. The analysis suggests that the model can partially rely on a goal-directed prior rather than the supplied action.

## 6 TOWARDS WORLD MODELS THAT GENERALIZE COMPOSITIONALLY

Our oracle-state experiments show that compositional failures cannot be attributed only to visual representation learning. Even when task-relevant physical state is provided directly, goal shifts remain difficult, particularly when prediction involves contact-rich interactions. This raises a broader question for future work: what assumptions on the dynamics, data, and learned representation are sufficient for a world model to generalize compositionally? We see two distinct challenges.

1. Identifying mechanisms. Fitting training trajectories does not ensure that a predictor has learned reusable action-conditioned mechanisms. This becomes harder in interaction-rich dynamics where action effects can vary with state and regime. Characterizing when such mechanisms can be identified from offline trajectories remains an open problem.

2. Extrapolation beyond the training support. Even if the correct mechanisms are identified, their behavior on novel state–action compositions is generally not determined without additional structural assumptions. Interaction-free dynamics provide a favorable case for compositional generalization because factor effects can often be reused independently, but world models must also handle interaction-rich dynamics where action effects vary with the joint physical configuration and regime. In this setting, extrapolation is fundamentally harder because the relevant interactions may change across compositions, and is a fruitful direction for future work.

A simple illustrative case. One favorable case for extrapolation in interaction-rich dynamics arises when, within each interaction regime, each transition mechanism depends only on a small subset of the state and action variables. Consider regime-dependent dynamics of the following form $z _ { t + 1 } = T _ { r _ { t } } ( z _ { t } , a _ { t } )$ , where $r _ { t }$ denotes an interaction regime, such as free motion, contact, or grasping. Both the autonomous state evolution and the effect of the action may therefore change across regimes. Suppose that, within a fixed regime $r ,$ the transition of each next-state factor $z _ { k , t + 1 }$ depends only on a subset of the coordinates of $\left( \boldsymbol { z } _ { t } , \boldsymbol { a } _ { t } \right)$ . Writing $\boldsymbol { x } _ { t } = \left( \boldsymbol { z } _ { t } , \boldsymbol { a } _ { t } \right)$ , this sparse dependence can be expressed as follows, $\| \nabla _ { x } T _ { k , r _ { t } } ( x ) \| _ { 0 } \ll \mathrm { d i m } ( x )$

Let $u _ { k }$ denote the relevant state and action coordinates for k-th transition mechanism $T _ { k , r _ { t } } ,$ , so that we may write $z _ { k , t + 1 } = T _ { k , r _ { t } } ( u _ { k , t } )$ . This allows a simple case where compositional shift in the full state–action space need not require extrapolation for each transition mechanism.

$$
{ \mathcal { C } } ^ { \mathrm { t e } } \not \subseteq { \mathcal { C } } ^ { \mathrm { t r } } , \qquad { \mathrm { w h i l e } } \qquad { \mathcal { U } } _ { k } ^ { \mathrm { t e } } \subseteq { \mathcal { U } } _ { k } ^ { \mathrm { t r } } \quad \forall k
$$

where $\mathcal { U } _ { k } ^ { \mathrm { t r } }$ and $\mathcal { U } _ { k } ^ { \mathrm { t e } }$ denote the train and test supports of $u _ { k }$ for the fixed regime r. Thus, the complete state–action composition may be unseen, while every input relevant to predicting each next-state factor remains within its training support. Hence, if the corresponding mechanisms have been correctly learned, we can generalize to novel compositions as we do not need to extrapolate any individual mechanism beyond its training support. Appendix J (Proposition 1) gives a formal statement.

## 7 CONCLUSION

We introduced AnyGoal, a benchmark for evaluating whether world models generalize to new be havioral goals by recombining familiar states and actions. Across a diverse set of models, goal-shif generalization remains difficult even when the relevant state and action factors are marginally covered by the training distribution. Together, these analyses show that goal-shift failures are heteroge neous, reflecting interactions among representation, dynamics prediction, task structure, and evaluation metric. More broadly, AnyGoal connects world-model evaluation with compositional generalization. It asks for prediction on novel joint state–action combinations within the product of marginal training supports, a reasonable but unresolved requirement for reusable dynamics. Our theoretical perspective suggests that global coverage of all state–action combinations is neither realistic nor necessary. Instead, world models need coverage of local mechanism inputs, together with objectives or architectures that identify action-conditioned mechanisms rather than behavior-policy shortcuts. By isolating goal-induced recombination as a failure mode, AnyGoal provides a controlled basis fo evaluating progress toward world models that learn reusable dynamics beyond expert rollouts.

## ACKNOWLEDGMENTS

David Klindt acknowledges the CSHL GPU cluster with assistance from the US National Institutes of Health Grant S10OD028632-01. Divyat Mahajan acknowledges support via FRQNT doctoral scholarship (https://doi.org/10.69777/354785) for his graduate studies.

## REFERENCES

Rim Assouel, Pietro Astolfi, Florian Bordes, Michal Drozdzal, and Adriana Romero-Soriano. Object-centric binding in contrastive language-image pretraining. Advances in Neural Information Processing Systems, 38:131418–131458, 2026.

Randall Balestriero and Yann LeCun. Lejepa: Provable and scalable self-supervised learning without the heuristics, 2025. URL https://arxiv.org/abs/2511.08544.

Jack Brady, Julius von Kugelgen, S¨ ebastien Lachapelle, Simon Buchholz, Thomas Kipf, and´ Wieland Brendel. Interaction asymmetry: A general principle for learning composable abstractions. In International Conference on Learning Representations, volume 2025, pp. 46606–46660, 2025.

Yuri Burda, Harrison Edwards, Amos Storkey, and Oleg Klimov. Exploration by random network distillation. In International Conference on Learning Representations, 2019. URL https: //openreview.net/forum?id=H1lJJnR5Ym.

Giacomo Camposampiero, Pietro Barbiero, Michael Hersche, Roger Wattenhofer, and Abbas Rahimi. Scalable evaluation and neural models for compositional generalization. Advances in Neural Information Processing Systems, 38:7341–7401, 2026.

Tianyuan Chen, Ronglong Cai, Faguo Wu, and Xiao Zhang. Active: Offline reinforcement learning via adaptive imitation and in-sample v-ensemble. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (eds.), International Conference on Learning Representations, volume 2025, pp. 77269–77287, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file/ c06f788963f0ce069f5b2dbf83fe7822-Paper-Conference.pdf.

Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. Int. J. Rob. Res., 44(10–11):1684–1704, September 2025. ISSN 0278-3649. doi: 10.1177/ 02783649241273668. URL https://doi.org/10.1177/02783649241273668.

Kefan Dong and Tengyu Ma. First steps toward understanding the extrapolation of nonlinear models to unseen domains. arXiv preprint arXiv:2211.11719, 2022.

Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, and Sergey Levine. D4rl: Datasets for deep data-driven reinforcement learning, 2021. URL https://arxiv.org/abs/2004.07219.

Scott Fujimoto, David Meger, and Doina Precup. Off-policy deep reinforcement learning without exploration. In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 2052–2062. PMLR, 09–15 Jun 2019. URL https://proceedings.mlr. press/v97/fujimoto19a.html.

Chen-Xiao Gao, Chenyang Wu, Mingjun Cao, Chenjun Xiao, Yang Yu, and Zongzhang Zhang. Behavior-regularized diffusion policy optimization for offline reinforcement learning. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 18630–18657. PMLR, 13–19 Jul 2025a. URL https://proceedings.mlr.press/v267/gao25q.html.

Yunkai Gao, Jiaming Guo, Fan Wu, and Rui Zhang. Policy constraint by only support constraint for offline reinforcement learning, 2025b. URL https://arxiv.org/abs/2503.05207.

Ignat Georgiev, Varun Giridhar, Nick Hansen, and Animesh Garg. Pwm: Policy learning with multi-task world models. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (eds.), International Conference on Learning Representations, volume 2025, pp. 13737–13757, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file/ 240ea1741b205ea295721d55184ac43b-Paper-Conference.pdf.

David Ha and Jurgen Schmidhuber. Recurrent world models facilitate policy evolution. In¨ S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018. URL https://proceedings.neurips.cc/paper\_files/paper/2018/ file/2de5d16682c3c35007e4e92982f1a2ba-Paper.pdf.

Nicklas Hansen and Xiaolong Wang. Hallucination in world models is predictable and preventable. arXiv preprint arXiv:2606.27326, 2026.

Petr Ivashkov, Randall Balestriero, and Bernhard Scholkopf. Sensorimotor world models: Percep-¨ tion for action via inverse dynamics, 2026. URL https://arxiv.org/abs/2606.20104.

Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Li Fei-Fei, C. Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), July 2017.

Daniel Keysers, Nathanael Scharli, Nathan Scales, Hylke Buisman, Daniel Furrer, Sergii Kashubin,¨ Nikola Momchev, Danila Sinopalnikov, Lukasz Stafiniak, Tibor Tihon, Dmitry Tsarkov, Xiao Wang, Marc van Zee, and Olivier Bousquet. Measuring compositional generalization: A comprehensive method on realistic data. In ICLR, 2020. URL https://arxiv.org/pdf/1912. 09713.pdf.

Rahul Kidambi, Aravind Rajeswaran, Praneeth Netrapalli, and Thorsten Joachims. Morel: Modelbased offline reinforcement learning. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 21810–21823. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper\_ files/paper/2020/file/f7efa4f864ae9b88d43527f4b14f750f-Paper.pdf.

David Klindt, Yann LeCun, and Randall Balestriero. When does lejepa learn a world model?, 2026. URL https://arxiv.org/abs/2605.26379.

Aviral Kumar, Justin Fu, Matthew Soh, George Tucker, and Sergey Levine. Stabilizing off-policy q-learning via bootstrapping error reduction. In H. Wallach,

H. Larochelle, A. Beygelzimer, F. d'Alche-Buc, E. Fox, and R. Garnett (eds.),´ Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper\_files/paper/2019/ file/c2073ffa77b5357a498057413bb09d3a-Paper.pdf.

Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine. Conservative q-learning for offline reinforcement learning. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 1179–1191. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper\_files/ paper/2020/file/0d2b2061826a5df3221116a5085a6052-Paper.pdf.

Sebastien Lachapelle, Divyat Mahajan, Ioannis Mitliagkas, and Simon Lacoste-Julien. Additive de-´ coders for latent variables identification and cartesian-product extrapolation. Advances in Neural Information Processing Systems, 36:25112–25150, 2023.

Brenden Lake and Marco Baroni. Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. In Jennifer Dy and Andreas Krause (eds.), Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp. 2873–2882. PMLR, 10–15 Jul 2018. URL https: //proceedings.mlr.press/v80/lake18a.html.

Romain Laroche, Paul Trichelair, and Remi Tachet Des Combes. Safe policy improvement with baseline bootstrapping. In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 3652–3661. PMLR, 09–15 Jun 2019. URL https://proceedings. mlr.press/v97/laroche19a.html.

Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. Offline reinforcement learning: Tutorial, review, and perspectives on open problems, 2020. URL https://arxiv.org/abs/ 2005.01643.

Shuang Li, Yihuai Gao, Dorsa Sadigh, and Shuran Song. Unified Video Action Model. In Proceedings of Robotics: Science and Systems, LosAngeles, CA, USA, June 2025. doi: 10.15607/RSS. 2025.XXI.074.

Yuejiang Liu, Alexandre Alahi, Chris Russell, Max Horn, Dominik Zietlow, Bernhard Scholkopf,¨ and Francesco Locatello. Causal triplet: An open challenge for intervention-centric causal representation learning. In Conference on Causal Learning and Reasoning, pp. 553–573. PMLR, 2023.

Susie Lu, Haonan Chen, Weirui Ye, and Yilun Du. Driftworld: Fast world modeling through drifting, 2026. URL https://arxiv.org/abs/2607.15065.

Lucas Maes, Quentin Le Lidec, Dan Haramati, Nassim Massaudi, Damien Scieur, Yann LeCun, and Randall Balestriero. stable-worldmodel-v1: Reproducible world modeling research and evaluation, 2026a. URL https://arxiv.org/abs/2602.08968.

Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, and Randall Balestriero. Leworld model: Stable end-to-end joint-embedding predictive architecture from pixels, 2026b. URL https://arxiv.org/abs/2603.19312.

Divyat Mahajan, Mohammad Pezeshki, Charles Arnal, Ioannis Mitliagkas, Kartik Ahuja, and Pascal Vincent. Compositional risk minimization. In Proceedings of the 42nd International Conference on Machine Learning, pp. 42503–42553. PMLR, 2025. URL https://proceedings.mlr. press/v267/mahajan25a.html.

Maxime Oquab, Timothee Darcet, Th´ eo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khali-´ dov, Pierre Fernandez, Daniel HAZIZA, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum?id=a68SUt6zFt. Featured Certification.

Vitoria Barin Pacela, Shruti Joshi, Isabela Camacho, Simon Lacoste-Julien, and David Klindt. Stop´ probing, start coding: Why linear probes and sparse autoencoders fail at compositional generalisation. arXiv preprint arXiv:2603.28744, 2026.

Giambattista Parascandolo, Niki Kilbertus, Mateo Rojas-Carulla, and Bernhard Scholkopf. Learning ¨ independent causal mechanisms. In International Conference on Machine Learning, pp. 4036– 4044. PMLR, 2018.

Seohong Park, Kevin Frans, Benjamin Eysenbach, and Sergey Levine. Ogbench: Benchmarking offline goal-conditioned rl. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (eds.), International Conference on Learning Representations, volume 2025, pp. 94937–94982, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file/ ecd92623ac899357312aaa8915853699-Paper-Conference.pdf.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 4195–4205, October 2023.

Marc Rigter, Bruno Lacerda, and Nick Hawes. Rambo-rl: Robust adversarial modelbased offline reinforcement learning. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems, volume 35, pp. 16082–16097. Curran Associates, Inc., 2022. doi: 10.52202/ 068431-1170. URL https://proceedings.neurips.cc/paper\_files/paper/ 2022/file/6691c5e4a199b72dffd9c90acb63bcd6-Paper-Conference.pdf.

B. Scholkopf, F. Locatello, S. Bauer, N. R. Ke, N. Kalchbrenner, A. Goyal, and Y. Bengio. Toward¨ causal representation learning. Proceedings of the IEEE, 109(5):612–634, 2021. doi: 10.1109/ JPROC.2021.3058954. URL https://ieeexplore.ieee.org/stamp/stamp.jsp? arnumber=9363924. \*equal contribution.

Yuhong Shi, Zhenhao Chu, Jie Wei, Jun Hao, Jianyi Liu, and Jingwen Fu. Overcoming statistical bias in action-controllable world models, 2026. URL https://arxiv.org/abs/2608.04653.

Uladzislau Sobal, Wancong Zhang, Kyunghyun Cho, Randall Balestriero, Tim G. J. Rudner, and Yann LeCun. Learning from reward-free offline data: A case for planning with latent dynamics models. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (eds.), Advances in Neural Information Processing Systems, volume 38, Main Conference, pp. 43905–43941. Curran Associates, Inc., 2025. doi: 10.52202/ 085713-1465. URL https://proceedings.neurips.cc/paper\_files/paper/ 2025/file/3e7cf447f21cd11c846463affefce665-Paper-Conference.pdf.

Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. The MIT Press, Cambridge, MA, 2nd edition, 2018.

Cheng Tang, Zhishuai Liu, and Pan Xu. Robust offline reinforcement learning with linearly structured f-divergence regularization. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 58842–58882. PMLR, 13–19 Jul 2025. URL https: //proceedings.mlr.press/v267/tang25f.html.

Yuval Tassa, Yotam Doron, Alistair Muldal, Tom Erez, Yazhe Li, Diego de Las Casas, David Budden, Abbas Abdolmaleki, Josh Merel, Andrew Lefrancq, Timothy Lillicrap, and Martin Riedmiller. Deepmind control suite, 2018. URL https://arxiv.org/abs/1801.00690.

Tristan Thrush, Ryan Jiang, Max Bartolo, Amanpreet Singh, Adina Williams, Douwe Kiela, and Candace Ross. Winoground: Probing vision and language models for visio-linguistic compositionality. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recogni tion (CVPR), pp. 5238–5248, June 2022.

Xinyue Wang and Biwei Huang. Modeling unseen environments with language-guided composable causal components in reinforcement learning. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu

(eds.), International Conference on Learning Representations, volume 2025, pp. 48784–48812, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file/ 79d86433c2acd12b6fa98553435d226e-Paper-Conference.pdf.

Ying Wang, Oumayma Bounou, Yann LeCun, and Mengye Ren. Adajepa: An adaptive latent world model, 2026. URL https://arxiv.org/abs/2606.32026.

Archana Warrier, Dat Nguyen, Michelangelo Naim, Moksh Jain, Yichao Liang, Karen Schroeder, Cambridge Yang, Joshua B. Tenenbaum, Sebastian Josef Vollmer, Kevin Ellis, and Zenna Tavares. Benchmarking world-model learning with environment-level queries. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/forum?id= Ny6UYZ8ysN.

Thaddaus Wiedemer, Jack Brady, Alexander Panfilov, Attila Juhos, Matthias Bethge, and Wieland¨ Brendel. Provable compositional generalization for object-centric learning. In International Conference on Learning Representations, volume 2024, pp. 20137–20164, 2024a.

Thaddaus Wiedemer, Prasanna Mayilvahanan, Matthias Bethge, and Wieland Brendel. Composi-¨ tional generalization from first principles. Advances in Neural Information Processing Systems, 36, 2024b.

Xindi Wu, Dingli Yu, Yangsibo Huang, Olga Russakovsky, and Sanjeev Arora. Conceptmix: A compositional image generation benchmark with controllable difficulty. Advances in Neural Information Processing Systems, 37:86004–86047, 2024.

Tianhe Yu, Garrett Thomas, Lantao Yu, Stefano Ermon, James Zou, Sergey Levine, Chelsea Finn, and Tengyu Ma. Mopo: Model-based offline policy optimization. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 14129–14142. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper\_files/paper/2020/ file/a322852ce0df73e204b7e67cbbef0d0a-Paper.pdf.

Tianhe Yu, Aviral Kumar, Rafael Rafailov, Aravind Rajeswaran, Sergey Levine, and Chelsea Finn. Combo: Conservative offline model-based policy optimization. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan (eds.), Advances in Neural Information Processing Systems, volume 34, pp. 28954–28967. Curran Associates, Inc., 2021. URL https://proceedings.neurips.cc/paper\_files/paper/2021/ file/f29a179746902e331572c483c45e5086-Paper.pdf.

Aimen Zerroug, Mohit Vaishnav, Julien Colin, Sebastian Musslick, and Thomas Serre. A benchmark for compositional visual reasoning. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems, volume 35, pp. 29776–29788. Curran Associates, Inc., 2022. doi: 10.52202/068431-2159. URL https://proceedings.neurips.cc/paper\_files/paper/2022/file/ c08ee8fe3d19521f3bfa4102898329fd-Paper-Datasets\_and\_Benchmarks. pdf.

Xiangcheng Zhang and Yilun Du. World action planner: Generalizable decision-making with actionconditioned world models, 2026. URL https://arxiv.org/abs/2607.27599.

Linfeng Zhao, Lingzhi Kong, Robin Walters, and Lawson L.S. Wong. Toward compositional generalization in object-oriented world modeling. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (eds.), Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pp. 26841–26864. PMLR, 17–23 Jul 2022. URL https://proceedings.mlr.press/ v162/zhao22b.html.

Yi Zhao, Aidan Scannell, Wenshuai Zhao, Yuxin Hou, Tianyu Cui, Le Chen, Dieter Buchler, Arno¨ Solin, Juho Kannala, and Joni Pajarinen. Efficient reinforcement learning by guiding world models with non-curated data. In C. Vondrick, B. Hariharan, C. Raffel, L. Pinto, D. Yang, and A. Faust (eds.), International Conference on Learning Representations, volume 2026, pp. 79246–79280, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file/ 7ff013b7e372ba5b790352ccd6908f03-Paper-Conference.pdf.

Gaoyue Zhou, Hengkai Pan, Yann LeCun, and Lerrel Pinto. DINO-WM: World models on pretrained visual features enable zero-shot planning. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=D5RNACOZEI.

Fangqi Zhu, Hongtao Wu, Song Guo, Yuxiao Liu, Chi-Hou Cheang, and Tao Kong. Irasim: A fine-grained world model for robot manipulation. 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 9834–9844, 2024. URL https://api.semanticscholar. org/CorpusID:270620594.

## Supplementary Material AnyGoal: On World Model Generalization Beyond Expert Rollouts

A. Compositional Generalization: Essential World-Model Test 18   
B. AnyGoal Task Details 18   
B.1 Two Room and Its Variants 18   
B.2 PushT and Its Variants 19   
B.3 Insert-Peg and Its Variants 19   
B.4 OGBench Cube and Its Variants 19   
C. Statistical Test 1: Marginal Distribution Support 20   
C.1 Distribution Support Criteria . 20   
C.2 Verified Factor Groups . 21   
C.3 Support-Verification Results 21   
D. Statistical Test 2: Full-Dimensional Joint Distribution Support 23   
E. Interventional Action Set 24   
F. Model Explanation 25   
F.1 End-to-End World Models 26   
F.2 Predictor-Only Models from True States . 27   
G. Training Details 28   
H. Detailed Planning and Evaluation Configuration 28   
H.1 Planning Evaluation Criteria . 29   
H.2 Configuration . 30   
I. Results 30   
I.1 Full Results 30   
I.2 Factor Support versus Observation Support 31   
I.3 Additional Observations 32   
J. Toward Compositional Generalization in World Models 32   
J.1 Identifying Reusable Mechanisms on the Training Support 33   
J.2 Extrapolating Identified Mechanisms to Novel Compositions 34   
K. AnyGoal Library and Reproducibility 35   
K.1 Installation and Dataset Access 36   
K.2 World-Model Evaluation . 37   
K.3 Reproducibility and Extensibility 37

## A COMPOSITIONAL GENERALIZATION: ESSENTIAL WORLD-MODEL TEST

We formalize why our compositional evaluation is an essential test for world models. The key distinction is between extrapolating to unseen factors and recombining observed ones. Good actionconditioned world model should predict the effect of an observed action on familiar physical factors, even when that particular combination was not selected by the behavior policy.

For clarity, consider a deterministic oracle-state setting with true controlled dynamics $z _ { t + 1 } =$ $T ^ { \star } ( z _ { t } , a _ { t } )$ . Suppose the offline dataset is collected by a behavior policy $a = \pi _ { E } ( z )$ . Then the observed state–action support lies on the policy graph,

$$
\Gamma _ { \pi _ { E } } = \{ ( z , a ) : a = \pi _ { E } ( z ) \} .\tag{6}
$$

Thus, expert rollouts cover only the state–action pairings induced by the behavior policy, not arbi trary interventions with observed actions.

Zero Training Error Does Not Imply Action-Conditioned Dynamics. A predictive world model trained on $( z , a , z ^ { \prime } ) \sim p _ { \mathrm { t r } }$ is constrained only on the state–action pairs generated by the behavior policy. Since observed actions satisfy $a = \pi _ { E } ( z )$ , the training loss constrains $\hat { T } ( z , \pi _ { E } ( z ) )$ , but not $\hat { T } ( z , a )$ for alternative observed actions at the same state. Consequently, the action-ignorant predictor

$$
\hat { T } _ { \mathrm { d e g } } ( z , a ) = T ^ { \star } ( z , \pi _ { E } ( z ) )\tag{7}
$$

achieves zero prediction error on the behavior dataset while ignoring the queried action a. Thus, expert-rollout prediction cannot distinguish a controlled dynamics model from a predictor that only models behavior-policy transitions.

Compositional Evaluation Requires Action-Conditioned Dynamics Compositional evaluation rules out this degenerate solution. Assume actions have nonzero effect within the observed marginal supports: for a set of oracle states with positive measure, there exist observed actions $a , a ^ { \prime } \in \mathcal { A } ^ { \mathrm { t r } }$ such that

$$
T ^ { \star } ( z , a ) \neq T ^ { \star } ( z , a ^ { \prime } ) .
$$

Now consider any action-ignorant predictor ${ \hat { T } } ( z , a ) = h ( z )$ . If it were correct on the relevant compositional state–action space, then for every observed action $a \in \mathcal { A } ^ { \mathrm { t r } } , h ( z ) = T ^ { \star } ( z , a )$ . Therefore, for any two observed actions $a , a ^ { \prime }$

$$
T ^ { \star } ( z , a ) = h ( z ) = T ^ { \star } ( z , a ^ { \prime } ) ,\tag{8}
$$

which contradicts the action-relevance assumption. Hence, no action-ignorant model can be correct on recombinations of observed state and action factors when actions causally affect the next state.

Interpretation. Expert rollouts test whether a model predicts what happens under the behavior policy. Compositional state–action evaluation instead tests whether the model predicts what happens when observed actions are applied to familiar physical factors, including combinations not generated by the behavior policy. This is an essential requirement for model-based control, since planning must evaluate candidate actions other than the behavior action.

## B ANYGOAL TASK DETAILS

## B.1 TWO ROOM AND ITS VARIANTS

TwoRoom is a simple navigation environment with two rooms connected by a narrow slot. The agent must move through the slot and reach a target location, with little to no meaningful interaction with the environment beyond collision with walls. Because the dynamics are essentially planar motion, this environment isolates whether models can preserve action-conditioned movement under goalbiased trajectory data.

TwoRoom Route is an interaction-light navigation variant. The agent must follow a piecewise-linear triangular route defined by a start point, an intermediate waypoint, and an invisible endpoint, rather than moving directly through the slot toward the target. Episodes mix clean, noisy, and variablespeed actions within the observed action support. This variant tests whether the model preserves planar action-conditioned dynamics when familiar positions and actions are recombined outside the dominant slot-crossing behavior of the base dataset.

## B.2 PUSHT AND ITS VARIANTS

PushT is a planar manipulation environment where the agent must push a T-shaped block into a fixed T-shaped goal pose. Unlike TwoRoom, the base task already requires meaningful contact, object translation, and object rotation. The expert trajectories are strongly goal-directed, coupling agent motion with the block pose and the target configuration. This environment tests whether world models learn action-conditioned contact dynamics rather than only the dominant expert pushing manifold.

AvoidT is a non-interactive variant. The agent must avoid touching the physical T-block and move to one of four corner regions. This variant keeps the agent and action factors familiar, but changes the required behavior from goal-directed pushing to free-space navigation around the object. It primarily tests whether the model preserves the action effect when the expert pushing objective is no longer appropriate.

StackT is an interaction-rich variant. The physical T-block must be placed so that its head overlaps with the fixed goal T, while its rotation is opposite to the original PushT goal orientation, creating a plus-shaped target configuration. This variant requires contact and rotation dynamics under a goal that differs from the original training objective.

PushT Anti-goal is also interaction-rich and directly reverses the dominant goal bias in the base dataset. The physical T-block is initialized near the original PushT goal pose, but the agent must push it away from the fixed goal region and move it toward one of four corner regions, with rotation opposite to the original goal orientation. This tests whether the model can predict familiar pushing dynamics when the required behavior contradicts the original expert trajectory distribution.

## B.3 INSERT-PEG AND ITS VARIANTS

Insert-Peg is a planar manipulation environment where an arm must move a peg toward a sampled slot and insert it into the target opening. The task involves articulated robot motion, object contact, and alignment between the peg and the slot. Expert trajectories are strongly structured around successful insertion behavior. This environment tests whether models can learn action-conditioned manipulation dynamics under goal-biased contact-rich data.

Avoid-Peg is an interaction-light variant. The arm must move around the target box in a specified hugging direction while avoiding contact with the peg. This changes the goal-conditioned arm behavior without requiring peg manipulation, testing whether the model can predict familiar arm motion under a non-insertion objective.

Extract-Peg is an interaction-rich variant. The arm must extract the peg from the target box, move it through a waypoint, and place it at a destination, with episodes balanced between left- and rightside box placements. This reverses the dominant insertion behavior while keeping the peg, box, workspace, and action space familiar.

## B.4 OGBENCH CUBE AND ITS VARIANTS

Cube is a robot manipulation environment where the end-effector must interact with a cube and move it toward a target configuration. The base behavior follows a structured pick-and-place pattern, including approach, grasp, lift, transport, and release. This creates a goal-biased trajectory distribution in which robot motion, gripper state, cube pose, and target location are tightly coupled. The environment tests whether models learn manipulation dynamics beyond this dominant expert sequence.

CubeHop is a non-interactive variant. The robot does not intentionally move the cube. Instead, the end-effector moves between waypoint and target positions while gradually changing its yaw. This variant changes the goal-conditioned end-effector behavior while avoiding object interaction. It tests whether the model can predict familiar robot motion and yaw changes outside the dominant cubemanipulation trajectory distribution.

CubeDiagonal is an interaction-rich variant. The robot manipulates the cube using a diagonal grasp configuration, where the gripper contacts opposite edges of the cube instead of the standard parallel faces. This introduces a novel contact regime with different grasp geometry while keeping the ob ject, task objective, and action space unchanged. It tests whether the model can compose familiar manipulation factors under unseen interaction configurations.

## C STATISTICAL TEST 1: MARGINAL DISTRIBUTION SUPPORT

## C.1 DISTRIBUTION SUPPORT CRITERIA

We describe the statistical criteria used to decide whether a query distribution remains inside the support of a reference distribution. Let $P = \{ p _ { n } \} _ { n = 1 } ^ { N }$ denote the reference or training point set, and let $\bar { Q } = \{ q _ { m } \} _ { m = 1 } ^ { M }$ denote the query point set.

For each factor, we compute the empirical mean $\textstyle \mu _ { P } \ = \ { \frac { 1 } { N } } \sum _ { n = 1 } ^ { N } p _ { n }$ and regularized covariance $\Sigma _ { P } = \mathrm { C o v } ( P ) + \lambda I ,$ where $\lambda > 0$ is a small diagonal regularizer used to ensure that the covariance matrix is invertible. This is particularly important for low-rank or manifold-valued factors, such as circular variables represented by sine and cosine coordinates.

Now we measure three metric respectively. First, the Mahalanobis criterion checks whether query points remain within the reference distribution’s main covariance structure.

Empirical Mahalanobis criterion. The empirical squared Mahalanobis distance of a point x from the reference distribution is

$$
d _ { P } ^ { 2 } ( x ) = ( x - \mu _ { P } ) ^ { \top } \Sigma _ { P } ^ { - 1 } ( x - \mu _ { P } ) .\tag{9}
$$

We compute the reference threshold as the 99th percentile of the empirical Mahalanobis distances of the reference points, $\tau _ { 0 . 9 9 } = \mathrm { Q u a n t i l e } _ { 0 . 9 9 } ^ { \cdot } ( \{ d _ { P } ^ { 2 } ( p _ { n } ) \} _ { n = 1 } ^ { N } )$ . The query set passes the Mahalanobis support criterion if at least 95% of its points lie below this reference threshold:

$$
\mathrm { M a h a l a n o b i s } _ { 0 . 9 9 } ( Q ; P ) = \left[ \frac { 1 } { M } \sum _ { m = 1 } ^ { M } { \bf 1 } \big [ d _ { P } ^ { 2 } ( q _ { m } ) \leq \tau _ { 0 . 9 9 } \big ] \geq 0 . 9 5 \right] .\tag{10}
$$

For a one-dimensional factor, this criterion reduces to the regularized squared z-score $d _ { P } ^ { 2 } ( x ) =$ $( x - \mu _ { P } ) ^ { 2 } / ( \sigma _ { P } ^ { 2 } + \lambda )$

Second, the KDE criterion checks whether query points lie in regions with sufficient empirical reference density, beyond an ellipsoidal approximation.

Histogram-binned KDE criterion. We estimate a reference density $\rho _ { P }$ with a histogram-binned Gaussian KDE fit only on the reference set $P .$ Query points are then evaluated under this fixed density estimate. The histogram grid is constructed from the pooled reference–query range with a small padding, and Gaussian smoothing uses Scott’s rule with a lower bandwidth clip to avoid degenerate bins.

After fitting the KDE, we evaluate the density of all reference samples and set the threshold to the empirical first percentile,

$$
\delta _ { 0 . 0 1 } = \mathrm { Q u a n t i l e } _ { 0 . 0 1 } \left( \{ \rho _ { P } ( p _ { n } ) \} _ { n = 1 } ^ { N } \right) .
$$

The query set passes the KDE criterion if at least 95% of query points have density no smaller than this reference-sample threshold:

$$
\mathrm { K D E } _ { 0 . 0 1 } ( Q ; P ) = \left[ \frac { 1 } { M } \sum _ { m = 1 } ^ { M } { \bf 1 } [ \rho _ { P } ( q _ { m } ) \geq \delta _ { 0 . 0 1 } ] \geq 0 . 9 5 \right] .\tag{11}
$$

For periodic yaw variables, KDE smoothing is applied with wrapped boundary conditions.

Lastly, the range criterion checks whether query points avoid direct coordinate-wise extrapolation beyond the observed marginal range.

Observed-range criterion. As a complementary support check, we compute an axis-wise observed-range test. For each query point $q _ { m }$ , we check whether all of its coordinates lie inside

the coordinate-wise minimum and maximum of the reference set, i.e., $q _ { m } \in [ \operatorname* { m i n } ( P )$ , max(P)] iff min $\phantom { } _ { \ i } p _ { n , j } \leq q _ { m , j } \leq \operatorname* { m a x } _ { n } p _ { n , j }$ for all $j .$ . The observed-range criterion is

$$
\mathrm { r a n g e } ( Q ; P ) = \left[ { \frac { 1 } { M } } \sum _ { m = 1 } ^ { M } \mathbf { 1 } [ q _ { m } \in [ \operatorname* { m i n } ( P ) , \operatorname* { m a x } ( P ) ] ] \geq 0 . 9 9 \right] .\tag{12}
$$

For circular variables, such as yaw, the observed-range criterion is ignored because the interval boundary is representation-dependent. In that case, the Mahalanobis and KDE criteria are used without the range check.

The final verdict is the conjunction of all non-ignored criteria:

$$
\operatorname { p a s s } ( Q ; P ) = \operatorname { r a n g e } ( Q ; P ) \wedge \operatorname { M a h a l a n o b i s } _ { 0 . 9 9 } ( Q ; P ) \wedge \operatorname { K D E } _ { 0 . 0 1 } ( Q ; P ) .\tag{13}
$$

If the observed-range criterion is ignored for a circular factor, the verdict is computed over the remaining criteria:

$$
\mathrm { p a s s } ( Q ; P ) = \mathrm { M a h a l a n o b i s } _ { 0 . 9 9 } ( Q ; P ) \wedge \mathrm { K D E } _ { 0 . 0 1 } ( Q ; P ) .\tag{14}
$$

This design separates marginal factor support from full trajectory-distribution typicality. The support checks are applied at the level of semantic factor groups to verify that each physical factor remains within the reference support. Combinatorial novelty can still arise when individually supported factors appear in new joint configurations that are rare or absent under the expert trajectory distribution.

## C.2 VERIFIED FACTOR GROUPS

We apply the three support criteria above to task-relevant factor groups rather than to the full joint state–action vector. This verifies marginal coverage of the physical factors and actions used in each goal-shifted variant, while allowing the full combinations to remain novel. Table 2 lists the factor groups checked for each reference–query pair.

Table 2: Every independent factor is in-distribution with respect to the reference dataset. Factor groups used for support verification. For each reference dataset $P ,$ we compare the corresponding goal-shifted variant datasets $Q$ against the listed task-relevant factors..
<table><tr><td>Reference dataset P</td><td>Query dataset Q</td><td>Factor groups checked</td></tr><tr><td>TwoRoom</td><td>TR-Route</td><td>Agent position (2D); action position delta (2D)</td></tr><tr><td>PushT</td><td>Avoid-T; Stack-T; Push-T Antigoal</td><td>Agent position (2D); block position (2D); action (2D); block angle.</td></tr><tr><td>Insert-Peg</td><td>Avoid-Peg; Extract-Peg</td><td>Arm root angle; Arm shoulder angle; Arm elbow angle; Arm wrist angle; Finger angle; Fingertip angle; Thumb angle; Thumbtip angle; hand, peg, and target position (2D); hand, peg, and target orientation (2D) ; 5 motor</td></tr><tr><td>Cube</td><td>Cube-Hop; Cube-Diagonal</td><td>commands (1D) independently. End-effector position (3D); block position (3D); block quaternion (4D); action position delta (3D); gripper opening; gripper contact; end-effector yaw; action yaw delta; action gripper delta. ; 6 joint pos independently</td></tr></table>

## C.3 SUPPORT-VERIFICATION RESULTS

Table 3 reports the full support-verification results for the factor groups in Table 2. For each reference–query pair, we report the pass rate under the Mahalanobis, KDE, and observed-range criteria, together with the final conjunction verdict. All checked factors pass, confirming that the AnyGoal variants remain marginally in distribution with respect to the corresponding base datasets.

Table 3: Distribution-support verification between each reference dataset (P) and query dataset (Q). A dash indicates that the range criterion is not applied to circular variables.(All Passed)
<table><tr><td rowspan=1 colspan=12>P         Q               Factor of variants              Mahalanobis 0.99 KDE RangeFinal verdict</td></tr><tr><td rowspan=1 colspan=12>agent pos (2D)                       0.970       1.000 1.000     passTR-RouteTwoRoomaction (2D)                          1.000      0.957 1.000     pass</td></tr><tr><td rowspan=1 colspan=12>block angle                          0.970      0.963            passagent pos (2D)                       0.968      0.984 1.000     passAvoid-TPushTblock pos (2D)                       1.000      0.990 1.000     passaction (2D)                          1.000      0.995 1.000     pass</td></tr><tr><td rowspan=1 colspan=12>block angle                          0.998      0.989            passagent pos (2D)                       0.992      0.989 1.000    passPushT    Stack-Tblock pos (2D)                       1.000      0.999 1.000     passaction (2D)                          1.000      0.954 1.000     pass</td></tr><tr><td rowspan=2 colspan=12>block angle                          1.000      0.988            passagent pos (2D)                       0.988      0.986 1.000     passPushT    Push-T Antigoalblock pos (2D)                       1.000       1.000 1.000    passaction (2D)                          1.000      0.989 1.000     pass</td></tr><tr><td rowspan=1 colspan=3></td></tr><tr><td rowspan=6 colspan=12>arm root angle                       0.980      0.963 1.000     passarm shoulder angle                   1.000      0.999 1.000     passarm elbow angle                     0.999      0.998 1.000    passarm wrist angle                      1.000      1.000 1.000     passfinger angle                          1.000       1.000 1.000     passfingertip angle                       0.996      0.998 1.000     passthumb ângle                         1.000       1.000 1.000     passthumbtip angle                       0.994       0.997 1.000     passhand position (x,z)                   0.978      1.000 1.000     passInsert-Peg Avoid-Peg      hand orientation (qw,qy)             1.000      1.000 1.000     passpeg position (x,z)                    1.000       1.000 1.000    passpeg orientation (qw,qy)              0.999      0.999 1.000     passtarget position (x,z)                  1.000      1.000 1.000     passtarget orientation (qw,qy)            1.000      1.000 1.000     passmotor command 0                   1.000      0.982 1.000     passmotor command 1                   1.000       1.000 1.000     passmotor command 2                   1.000      0.991 1.000</td></tr><tr><td rowspan=1 colspan=1>1.000pass</td></tr><tr><td rowspan=1 colspan=1>1.000pass</td></tr><tr><td rowspan=1 colspan=1>1.000pass</td></tr><tr><td rowspan=1 colspan=1>1.000pass</td></tr><tr><td rowspan=3 colspan=11>motor command 3                   1.000       1.000motor command 4                   1.000      1.000</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1>1.000     pass</td></tr><tr><td rowspan=1 colspan=1>1.000     pass</td></tr><tr><td rowspan=1 colspan=11>arm root angle                       0.996      0.975</td><td rowspan=1 colspan=1>1.000     pass</td></tr><tr><td rowspan=1 colspan=11>arm shoulder angle                  0.989      0.983</td><td rowspan=1 colspan=1>1.000</td></tr><tr><td rowspan=7 colspan=11>arm elbow angle                     0.988      0.988arm wrist angle                      1.000      0.998finger angle                          1.000       1.000fingertip angle                       0.989      0.991thumb angle                         1.000      0.997thumbtip angle                      0.966      0.960hand position (x,z)                   1.000      0.998</td><td rowspan=1 colspan=1>1.000</td></tr><tr><td rowspan=1 colspan=1>pass</td></tr><tr><td rowspan=1 colspan=1>pass</td></tr><tr><td rowspan=1 colspan=1>1.000     pass</td></tr><tr><td rowspan=1 colspan=1>1.000     pass</td></tr><tr><td rowspan=1 colspan=1>1.000    pass</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>1.000    pass</td></tr><tr><td rowspan=1 colspan=8>Insert-Peg Extract-Peg</td><td rowspan=1 colspan=3>hand orientation (qw,qy)             1.000      1.000</td><td rowspan=1 colspan=1>1.000     pass</td></tr><tr><td rowspan=1 colspan=5></td><td rowspan=6 colspan=7>peg position (x,z)                    0.998      0.997peg orientation (qw,qy)              0.997      0.997target position (x,z)                  0.983      0.978target orientation (qw,qy)            1.000      1.000motor command 0                   1.000      1.000motor command 1                   1.000      1.000</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=1>0.9970.997</td><td rowspan=1 colspan=1>pass</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=1>0.9830.978</td><td rowspan=1 colspan=1>pass</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=1>1.0001.000</td><td rowspan=1 colspan=1>pass</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=1>1.0001.000</td><td rowspan=1 colspan=1>pass</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=1>1.0001.000</td><td rowspan=1 colspan=1>1.000     pass</td></tr><tr><td rowspan=1 colspan=8></td><td rowspan=1 colspan=2>motor command 2</td><td rowspan=1 colspan=1>1.000      0.987</td><td rowspan=1 colspan=1>1.000    pass</td></tr><tr><td rowspan=2 colspan=11>motor command 4                   1.000      1.000</td><td rowspan=1 colspan=1>motor command 3</td></tr><tr><td rowspan=1 colspan=1>1.000    pass</td></tr><tr><td rowspan=1 colspan=11>joint pos 0                           0.995       0.995</td><td rowspan=1 colspan=1>1.000    pass</td></tr><tr><td rowspan=1 colspan=8>j</td><td rowspan=1 colspan=3>oint pos 1                           0.997       0.998</td><td rowspan=1 colspan=1>1.000     pass</td></tr><tr><td rowspan=3 colspan=8>jj</td><td rowspan=1 colspan=3>joint pos 2                           0.974      0.974</td><td rowspan=1 colspan=1>1.000     pass</td></tr><tr><td rowspan=1 colspan=3>oint pos 3                           0.990      0.991</td><td rowspan=1 colspan=1>1.000     pass</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=3>oint pos 4                           1.000      1.000</td><td rowspan=1 colspan=1>1.000     pass</td></tr><tr><td rowspan=3 colspan=12>gripper opening                     0.980      1.000 1.000     passCube     Cube-HopEE yaw                             0.958      0.961            passdelta yaw                            1.000       1.000 1.000    pass</td></tr><tr><td rowspan=1 colspan=4></td></tr><tr><td rowspan=1 colspan=4></td></tr></table>

Continued on next page

Table 3 continued.
<table><tr><td> $P$ </td><td>Q</td><td>Factor of variants</td><td>Mahalanobis 0.99</td><td>KDE</td><td>Range</td><td>Final verdict</td></tr><tr><td rowspan="9"></td><td rowspan="9"></td><td>delta gripper EE pos (3D)</td><td>1.000</td><td>1.000 0.970</td><td>1.000 1.000</td><td>pass</td></tr><tr><td></td><td>0.969</td><td></td><td></td><td>pass</td></tr><tr><td>block pos (3D)</td><td>1.000</td><td>1.000</td><td>1.000</td><td>pass</td></tr><tr><td>block quat (4D)</td><td>1.000</td><td>1.000</td><td>1.000</td><td>pass</td></tr><tr><td>delta pos (3D)</td><td>1.000</td><td>1.000</td><td>1.000</td><td>pass</td></tr><tr><td>joint pos 0</td><td>0.989</td><td>0.990</td><td>1.000</td><td></td></tr><tr><td></td><td>0.992</td><td>0.991</td><td>1.000</td><td>pass</td></tr><tr><td>joint pos 1 joint pos 2</td><td>0.990</td><td>0.990</td><td>1.000</td><td>pass pass</td></tr><tr><td>joint pos 3</td><td>0.991</td><td>0.992</td><td>1.000</td><td>pass</td></tr><tr><td rowspan="9">Cube Cube-Diagonal</td><td>joint pos 4</td><td>0.978</td><td>0.979</td><td>1.000</td><td></td><td>pass</td></tr><tr><td>joint pos 5</td><td>1.000</td><td>1.000</td><td>1.000</td><td></td><td>pass</td></tr><tr><td>gripper opening</td><td>0.989</td><td>0.984</td><td>1.000</td><td></td><td>pass</td></tr><tr><td>gripper contact</td><td>1.000</td><td></td><td>0.985</td><td>1.000</td><td>pass</td></tr><tr><td>EE yaw</td><td>1.000</td><td></td><td>1.000</td><td></td><td>pass</td></tr><tr><td>delta yaw</td><td>1.000</td><td>0.992</td><td></td><td>1.000</td><td>pass</td></tr><tr><td>delta gripper</td><td>0.995</td><td>0.995</td><td>1.000</td><td></td><td>pass</td></tr><tr><td>EE pos (3D)</td><td>0.989</td><td>0.990</td><td></td><td>1.000</td><td>pass</td></tr><tr><td>block pos (3D)</td><td>0.991</td><td>0.990</td><td></td><td>1.000</td><td>pass</td></tr><tr><td></td><td>block quat (4D)</td><td></td><td></td><td></td><td>1.000</td><td></td></tr><tr><td></td><td>delta pos (3D)</td><td></td><td>0.997 0.989</td><td>0.987 0.990</td><td>1.000</td><td>pass pass</td></tr></table>

## D STATISTICAL TEST 2: FULL-DIMENSIONAL JOINT DISTRIBUTION SUPPORT

We complement the factor-wise support tests in Table 3 with a full-dimensional joint test. The factorwise tests verify whether each task-relevant factor or factor group is marginally supported by the reference data. In contrast, the joint test asks whether the complete state–action combination lies in a high-density region of the reference distribution. This test uses the concatenated vectors directly, without coordinate marginalization or worst-pair reduction.

For each transition, let a denote the action, $s _ { \mathrm { a g e n t } }$ the agent-observed state, and $s _ { \mathrm { o t h e r } }$ the task-object and target state. We evaluate three joint vectors:

$$
z _ { \mathrm { a g e n t } } = s _ { \mathrm { a g e n t } } , \qquad z _ { \mathrm { a l l } } = [ s _ { \mathrm { a g e n t } } , s _ { \mathrm { o t h e r } } ] , \qquad z _ { a , \mathrm { a l l } } = [ a , s _ { \mathrm { a g e n t } } , s _ { \mathrm { o t h e r } } ] .
$$

Only rows that are finite in every coordinate of the corresponding vector are retained. Thus, terminal rows with undefined actions contribute to the state-only vectors but not to the action–state vector.

Table 4: Components and dimensionalities of the full-dimensional joint KDE.
<table><tr><td>Family</td><td> $s _ { \mathrm { a g e n t } }$ </td><td> $s _ { \mathrm { a l l } }$ </td><td> $[ a , s _ { \mathrm { a l l } } ]$ </td></tr><tr><td>TwoRoom</td><td>position (2D)</td><td>position (2D)</td><td>4D</td></tr><tr><td>PushT</td><td>agent position (2D)</td><td>+ block pose (5D total)</td><td>7D</td></tr><tr><td>Cube</td><td>EE/gripper state (6D)</td><td>+ block pose (13D total)</td><td>18D</td></tr><tr><td>Manipulator</td><td>joints + hand pose (20D)</td><td>+ peg/target poses (28D total)</td><td>33D</td></tr></table>

For each reference vector set, we quantize each coordinate using reference-based limits with small padding, and assign zero density to target values outside these limits. We then whiten the quantized vectors using the full reference covariance with a small diagonal regularizer. This preserves correlations between action and state coordinates when computing kernel distances.

We fit a full-dimensional Gaussian KDE to the whitened reference vectors using Scott’s bandwidth rule. The density is evaluated without reference subsampling, dimension dropping, nearest-neighbor truncation, or pairwise marginalization. The density threshold is calibrated only from the reference samples:

$$
\tau = Q _ { 0 . 0 1 } \big ( \{ \log \widehat { p } ( x _ { i } ) \} _ { i = 1 } ^ { N } \big ) .\tag{15}
$$

For target vectors $Y = \{ y _ { j } \} _ { j = { \mathrm { \scriptscriptstyle { \frac { M } { \it { 1 2 } } } } } } ^ { M }$ , we report the fraction whose log-density exceeds this referencesample threshold:

$$
\mathrm { I D } ( Y ; X ) = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \mathbf { 1 } [ \log \widehat { p } ( y _ { j } ) \geq \tau ] .\tag{16}
$$

The joint score is therefore a full-vector KDE support percentage. Individual factors can pass their marginal support tests while the concatenated vector has a low joint score, indicating that the corresponding combination of familiar values is rare under the reference distribution.

## E INTERVENTIONAL ACTION SET

For each environment, we choose a fixed interventional action set that remains within the empirical action support while covering the task-relevant intervention directions as uniformly as possible. For environments with direct 2D controls, this is a regular grid in action space. For higher-dimensional manipulation actions, we vary only the dimensions needed for the intended spatial intervention and keep the remaining dimensions fixed.

TwoRoom. For TwoRoom, actions are two-dimensional position deltas, so interventional interventions are defined directly in the action space. We use a fixed $3 \times 3$ grid that is uniformly spread over the two action dimensions and includes the zero action as the center intervention.

Selected TwoRoom interventional actions.

$$
\mathcal { A } _ { \mathrm { c f } } ^ { \mathrm { T w o R o m } } = \left\{ \begin{array} { c c } { - 1 } & { - 1 } \\ { - 1 } & { 0 } \\ { - 1 } & { 1 } \\ { 0 } & { - 1 } \\ { 0 } & { 0 } \\ { 0 } & { 1 } \\ { 1 } & { - 1 } \\ { 1 } & { 0 } \\ { 1 } & { 1 } \end{array} \right\} = \{ - 1 , 0 , 1 \} \times \{ - 1 , 0 , 1 \} .
$$

Each row is a 2D action.

Push-T. For Push-T, actions are also two-dimensional planar controls. We therefore use a fixed $3 \times 3$ grid in the planar action space. The grid is chosen to provide evenly distributed in-distribution interventions at the empirical action scale of the dataset.

Selected Push-T interventional actions.

$$
\mathcal { A } _ { \mathrm { c f } } ^ { \mathrm { p u s h T } } = \left\{ \begin{array} { r r r } { 0 } & { 0 } & { 0 } \\ { 0 . 2 5 } & { 0 } \\ { - 0 . 2 5 } & { 0 } \\ { 0 . 2 5 } & { - 0 . 2 5 } \\ { 0 . 2 5 } & { 0 . 2 5 } \\ { - 0 . 2 5 } & { - 0 . 2 5 } \\ { - 0 . 2 5 } & { 0 . 2 5 } \\ { 0 } & { - 0 . 2 5 } \\ { 0 } & { 0 . 2 5 } \end{array} \right\} .
$$

Each row is a 2D planar action.

Insert-Peg. Unlike the 2D navigation and planar pushing tasks, Insert-Peg actions are 5- dimensional motor commands rather than direct task-space displacements. Therefore, interventional actions cannot be chosen by simply placing actions on a Euclidean grid in task space. Instead, we select actions by their realized effect on the hand endpoint.

We first choose a representative state near the center of the expert-state distribution. Each candidate action is sampled from the dataset and repeated for five environment control steps from this same state. We then measure the resulting hand endpoint in the x-z plane. To ensure that interventions remain in distribution, we retain only actions whose 20-nearest-neighbor distance in the raw 5D action space is below the 95th-percentile threshold estimated from held-out dataset actions.

Among the retained actions, we select a fixed set of nine interventional actions whose realized endpoints form the widest approximately uniform 3×3 grid. The zero action is assigned to the center, and the remaining actions are chosen to maximize grid spacing while keeping each endpoint within 10% of its corresponding ideal grid point. The resulting grid spacing is 0.0358, with a maximum endpoint mismatch of 9.95% of the grid spacing.

Selected Insert-Peg actions.

$$
\mathcal { A } _ { \mathrm { c f } } ^ { \mathrm { I n s e r t } } = \left\{ \begin{array} { r r r r r r } { 0 . 6 7 2 8 } & { - 0 . 9 8 1 4 } & { - 0 . 9 9 3 2 } & { - 1 . 0 0 0 0 } & { 0 . 9 9 0 0 } \\ { 0 . 1 3 5 3 } & { - 1 . 0 0 0 0 } & { 0 . 6 5 5 7 } & { - 0 . 9 9 7 2 } & { 1 . 0 0 0 0 } \\ { - 0 . 6 0 2 3 } & { - 0 . 1 2 0 4 } & { 0 . 3 1 4 2 } & { 0 . 9 9 4 3 } & { - 1 . 0 0 0 0 } \\ { 1 . 0 0 0 0 } & { - 0 . 8 0 0 2 } & { - 1 . 0 0 0 0 } & { - 1 . 0 0 0 0 } & { 0 . 9 7 3 9 } \\ { 0 . 0 0 0 0 } & { 0 . 0 0 0 0 } & { 0 . 0 0 0 0 } & { 0 . 0 0 0 0 } & { 0 . 0 0 0 0 } \\ { - 0 . 7 7 9 7 } & { 1 . 0 0 0 0 0 } & { - 0 . 6 6 7 2 } & { 0 . 6 5 5 3 } & { - 1 . 0 0 0 0 0 } \\ { 0 . 9 1 3 0 } & { - 0 . 0 0 8 2 } & { - 1 . 0 0 0 0 } & { - 0 . 9 8 2 0 } & { 1 . 0 0 0 0 } \\ { - 0 . 0 5 1 3 } & { 1 . 0 0 0 0 0 } & { - 1 . 0 0 0 0 } & { 0 . 5 7 0 4 } & { 0 . 1 9 3 6 } \\ { - 0 . 5 8 3 3 } & { 0 . 8 9 5 8 } & { 0 . 8 2 3 4 } & { 0 . 0 5 8 2 } & { 0 . 0 1 6 1 8 } \end{array} \right\}
$$

Each row is a 5D motor command. The zero action is the center intervention.

Cube. For Cube, the raw action is five-dimensional, but the interventional protocols are designed to probe spatial sensitivity of end-effector motion. We therefore vary only translational action dimensions and keep ∆yaw and ∆gripper fixed at zero. The horizontal protocol uses an x-y grid.

Selected Cube horizontal interventional actions.

$$
\mathcal { A } _ { \mathrm { c f } } ^ { \mathrm { C u b e - x y } } = \left\{ \begin{array} { c c c c c } { - 0 . 5 4 2 7 } & { - 0 . 9 5 0 4 } & { 0 } & { 0 } & { 0 } \\ { - 0 . 5 4 2 7 } & { 0 } & { 0 } & { 0 } & { 0 } \\ { - 0 . 5 4 2 7 } & { 0 . 9 5 0 4 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { - 0 . 9 5 0 4 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 . 9 5 0 4 } & { 0 } & { 0 } & { 0 } \\ { 0 . 5 4 2 7 } & { - 0 . 9 5 0 4 } & { 0 } & { 0 } & { 0 } \\ { 0 . 5 4 2 7 } & { 0 } & { 0 } & { 0 } & { 0 } \\ { 0 . 5 4 2 7 } & { 0 . 9 5 0 4 } & { 0 } & { 0 } & { 0 } \end{array} \right\} .
$$

Each row is a normalized 5D Cube action $( a _ { x } , a _ { y } , a _ { z } , \Delta \mathrm { y a w } , \Delta \mathrm { g r i p p e r } )$

Implementation Detail. Within orignal LeWM architecture, we measure action sensitivity using hidden representations from the predictor blocks. Let $f _ { l }$ denote the last-token hidden representation after the l-th ConditionalBlock,

$$
f _ { l } = \mathrm { o u t p u t } \left[ : , \begin{array} { r r r } { - 1 , } & { : } \end{array} \right]
$$

obtained from a forward hook on model.predictor.transformer.layers[layer idx]. Thus, $f _ { l }$ is not the representation immediately after AdaLN modulation; it is the output of the full AdaLN-conditioned transformer block, including the attention and MLP residual updates. In all reported layer-wise analyses, we average the sensitivity scores across these block outputs. For the final prediction representation, we use pred $[ : , \quad - 1 , \quad : ]$ , the output of model.predict(...) after the final pred proj layer.

## F MODEL EXPLANATION

Our model selection is designed to span several substantially different approaches to visual world modeling rather than to concentrate on a single architectural family. In particular, we evaluate both joint-embedding predictive architectures (JEPAs) and generative visual world models. Within the JEPA family, we include methods that jointly learn perception and dynamics from the offline dataset as well as a method that keeps a large pretrained visual encoder frozen. Within the generative family, we include both a direct action-conditioned generator and a latent video diffusion model with an explicit visual encoder–decoder. This diversity is important for our central question: whether goal-induced combinatorial bias is specific to a particular representation or generative objective, or can arise more broadly when action-conditioned prediction is trained on a biased offline distribution. For consistency, all models use frame skip 5, history length 3, and one-step prediction, except Sensorimotor-WM, where we keep the original history length 1 to preserve its inverse-dynamics setup.

## F.1 END-TO-END WORLD MODELS

LeWorldModel (LeWM). LeWorldModel (Maes et al., 2026b) is an end-to-end joint-embedding predictive architecture trained directly from pixels and actions. An image encoder maps observations to compact latent representations, and an autoregressive transformer predictor models the next latent representation conditioned on the action. Actions are injected into each transformer layer through adaptive layer normalization (AdaLN-zero). Unlike approaches that rely on a frozen encoder, stopgradient target network, or exponential-moving-average encoder, LeWM jointly optimizes its en coder and predictor.

A central design choice in LeWM is its anti-collapse regularization. In addition to the nextembedding mean-squared prediction loss, LeWM applies the Sketched-Isotropic-Gaussian Regularizer (SIGReg), which encourages the empirical latent distribution to match an isotropic Gaussian. The resulting model is particularly useful for our study because it represents a recent example of a fully end-to-end JEPA in which both the visual representation and dynamics predictor are shaped by the same offline trajectories.

Sensorimotor World Model (SMWM). The Sensorimotor World Model (Ivashkov et al., 2026) is also an end-to-end JEPA, but uses a different principle to prevent representation collapse. Rather than prescribing the marginal geometry of the latent representation, SMWM introduces an inverse dynamics model that predicts the executed action from consecutive latent states. The encoder, forward dynamics model, and inverse model are optimized jointly using forward-prediction and inversedynamics losses.

The inverse objective requires consecutive representations to preserve information that is useful for recovering the action that produced the transition. Consequently, the learned representation is explicitly biased toward controllable, action-relevant structure. Architecturally, SMWM adopts the LeWM encoder and transformer forward predictor, including AdaLN-zero action conditioning, while replacing Gaussian latent regularization with inverse dynamics as its anti-collapse mechanism. This makes SMWM particularly relevant to our benchmark: it provides a strong test of whether explicitly encouraging action-aware representations is sufficient to prevent failures induced by goal-biased state–action coverage.

DINO World Model (DINO-WM). DINO-WM (Zhou et al., 2025) separates representation learning from dynamics learning. Instead of jointly learning an image encoder from the offline control dataset, it uses pretrained DINOv2 patch features as the observation representation and keeps the DINOv2 encoder frozen during both world-model training and evaluation. A causal ViT-based transition model then predicts future patch representations from a history of latent observations and actions.

DINO-WM also differs in how actions enter the predictor. The action is first mapped through an MLP and the resulting action representation is concatenated to each spatial patch representation before the transformer processes the latent state. Thus, compared with LeWM and SMWM, DINO-WM provides a useful separation between perception and dynamics: the visual representation is inherited from large-scale pretraining, while only the action-conditioned transition model is learned from the potentially goal-biased offline trajectories. This lets us test whether combinatorial failures persist even when the visual encoder itself is not learned from the biased task dataset.

DriftWorld. DriftWorld (Lu et al., 2026) represents a substantially different, reconstruction-based family of world models. It is an action-conditioned generative model based on drifting rather than a JEPA or iterative diffusion model. Its core generator is a spatial–temporal U-Net that receives observation history, Gaussian noise, and a future action sequence and generates the corresponding future visual observations. Actions condition the U-Net frame-by-frame through FiLM, such tha each future frame is explicitly conditioned on its corresponding action.

For relatively simple simulated domains such as PushT, DriftWorld operates directly in pixel space and does not require a learned visual state encoder between the observations and the actionconditioned generator. Its drifting objective moves generated samples toward ground-truth future observations and away from generated negatives, enabling future video generation in a single forward pass at inference time. DriftWorld therefore provides an important contrast to the latent JEPA models: prediction and visual generation are implemented by a single action-conditioned generative backbone rather than an explicit encoder followed by a latent dynamics predictor.

IRASim. IRASim (Zhu et al., 2024) represents the latent generative end of our model suite. It formulates world modeling for robot manipulation as trajectory-conditioned video generation. Frames are first compressed using a pretrained VAE, after which a Diffusion Transformer (DiT) models future video latents conditioned on historical observations and a robot action trajectory. The generated latents are subsequently decoded back to RGB observations.

A defining component of IRASim is its fine-grained action conditioning. Rather than conditioning the complete video only on a global trajectory embedding, IRASim can associate individual actions with their corresponding generated frames through frame-level adaptive normalization. This architecture gives us a generative model that is structurally different from DriftWorld: visual prediction occurs in an encoded latent space, the dynamics generator is a diffusion transformer, and an explicit encoder–decoder surrounds the dynamics model.

## F.2 PREDICTOR-ONLY MODELS FROM TRUE STATES

End-to-end evaluation mixes perception and dynamics-prediction errors. To isolate the latter, we use predictor-only models that replace the visual encoder with oracle states from simulator metadata and proprioceptive signals. These oracle states contain task-relevant physical variables, such as positions, orientations, agent or gripper configurations, and other environment-specific state factors needed for one-step prediction.

We refer to these inputs as oracle state representations rather than unique or minimal Markov states. Their role is operational: they provide the predictor with relevant physical information directly, removing visual representation learning as the explanation for prediction failure. We instantiate three predictor families derived from the end-to-end models above. The two transformer-based predictoronly models are trained with next-state mean-squared error,

$$
\mathcal { L } _ { \mathrm { p r e d } } = \left. \hat { s } _ { t + 1 } - s _ { t + 1 } \right. _ { 2 } ^ { 2 } .\tag{17}
$$

Toy-DriftWorld instead retains the drifting particle loss used by its generative backbone.

AdaLN Transformer Predictor. Our first predictor follows the forward-model architecture shared by LeWM and SMWM. Ground-truth state factors replace the learned visual embedding, while the action is injected into the transformer through AdaLN-zero. The transformer predicts the next true state directly. Because LeWM and SMWM use the same forward-predictor family, a single predictor-only architecture represents both models in this controlled setting. Their endto-end distinction—SIGReg versus inverse-dynamics representation regularization—is intentionally removed once the encoder is replaced by the oracle state.

Action-Duplicated Transformer Predictor. Our second predictor follows DINO-WM’s transition-model conditioning strategy. The true state replaces DINO patch features, while an embedded action is attached to the state representation before processing by a causal transformer. This preserves the central architectural distinction of DINO-WM’s predictor: action information enters through the predictor input rather than modulating transformer layers through AdaLN. The model is again trained only to regress the next true state with MSE.

DriftWorld-Style Generative Predictor. Our third predictor is derived from the actionconditioned generative backbone of DriftWorld. Because DriftWorld does not require a separate visual encoder for the simple simulated setting, its U-Net constitutes the principal predictive computation. For the predictor-only experiment, we use a reduced-size version of this backbone and replace pixel observations and outputs with the corresponding true-state representation. The predictor retains the drifting particle loss while operating on oracle-state inputs and next-state targets.

## F.2.1 ORACLE STATES

Oracle-state predictors replace RGB observations with task-relevant physical state variables from each base environment.

Table 5: Oracle state variables used by predictor-only models for each base environment.
<table><tr><td>Base environment</td><td>Dim.</td><td>Oracle state variables</td></tr><tr><td>TwoRoom</td><td>2D</td><td>Agent position  $( x , y )$ </td></tr><tr><td>Push-T</td><td>10D</td><td>Goal position  $( x , y )$  , goal orientation (sin θ, cos θ), block position  $( x , y )$  , block orientation (sin θ, cos θ), and agent position  $\bar { ( } x , y )$ </td></tr><tr><td>Cube</td><td>15D</td><td>End-effector position  $( x , y , z )$  , end-effector yaw (cos θ, sin θ), gripper opening, gripper contact, gripper velocity, block position</td></tr><tr><td>Manipulator Insert-Peg</td><td>33D</td><td> $( x , y , z )$  , and block quaternion  $( w , x , y , z )$  Arm joint positions, touch sensors, hand pose  $( x , z , q _ { w } , q _ { y } )$  , peg pose  $( x , z , q _ { w } , q _ { y } )$  , and target pose  $( x , z , q _ { w } , q _ { y } )$ </td></tr></table>

## F.2.2 MODEL ARCHITECTURE DETAILS

For LeWM (Maes et al., 2026b) and Sensorimotor World-Model (Ivashkov et al., 2026), we use the authors’ official implementations. For DINO-WM (Zhou et al., 2025), we use the Stable-WorldModel (Maes et al., 2026a) implementation without proprioceptive inputs. For DriftWorld (Lu et al., 2026), we use the authors’ implementation with two minor modifications: we integrate frame skip into the action-conditioning interface and add a pixel-space cost function for MPC planning, without changing the main architecture. We use the Push-T model configuration for TwoRoom and Push-T, and the Robomimic model configuration for the Cube and Manipulator environments. For IRASim (Zhu et al., 2024), we use the authors’ implementation with the same frame-skip integration and pixel-space planning cost, and use IRASim-XL/2 for all environments.

For the oracle-state experiments, we use three toy predictor families derived from the corresponding end-to-end modeling motifs. Toy-AdaLN keeps the LeWM transformer size unchanged, replacing only the pixel-latent input and output with true-state projections. Toy Action-Duplicate preserves the DINO-WM-style action duplication, causal masking, and shifted prediction design, but adapts it to scalar true-state tokens with a smaller 192D predictor instead of the original high-dimensional patchtoken predictor. Toy-DriftWorld preserves the stochastic particle-generation objective and drifting loss, but replaces the image U-Net with a compact vector-valued residual FiLM MLP operating on oracle states.

## G TRAINING DETAILS

AdaLN predictors and Action duplicated predictors use AdamW, batch size 256, weight decay $1 0 ^ { - 3 }$ linear warmup followed by cosine decay, gradient clipping at 1.0, and an MSE loss over all shifted state predictions. Toy-DriftWorld is trained with AdamW, batch size 128, learning rate $1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 2 }$ , a 500-step warmup followed by a constant schedule, gradient clipping at 2.0, and the drifting particle loss. For end-to-end world models, we follow the original settings.

## H DETAILED PLANNING AND EVALUATION CONFIGURATION

All planning experiments use the same MPC evaluation pipeline based on the Stable World Model library Maes et al. (2026a), with Cross-Entropy Method (CEM) solver.

Table 6: Training details and selected checkpoints for end-to-end world models.
<table><tr><td>Dataset</td><td>Model</td><td>LR</td><td>Configured duration</td><td>Selected checkpoint</td></tr><tr><td>Push-T</td><td>LeWM</td><td></td><td></td><td>official LeWM checkpoint</td></tr><tr><td>Push-T</td><td>Sensorimotor-WM</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>10 epochs</td><td>epoch 10</td></tr><tr><td>Push-T</td><td>DINO-WM</td><td></td><td></td><td>official DINO-WM checkpoint</td></tr><tr><td>Push-T</td><td>DriftWorld</td><td> $5 \times 1 0 ^ { - 4 }$ </td><td>max 500k steps</td><td>step 500k</td></tr><tr><td>Push-T</td><td>IRASim</td><td> $5 \times 1 0 ^ { - 5 }$ </td><td>max 300k steps</td><td>step 300k</td></tr><tr><td>Cube</td><td>LeWM</td><td></td><td></td><td>official LeWM checkpoint</td></tr><tr><td>Cube</td><td>Sensorimotor-WM</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>10 epochs</td><td>epoch 10</td></tr><tr><td>Cube</td><td>DINO-WM</td><td></td><td></td><td>official DINO-WM checkpoint</td></tr><tr><td>Cube</td><td>DriftWorld</td><td> $5 \times 1 0 ^ { - 5 }$ </td><td>300-epoch cap</td><td>step 100k</td></tr><tr><td>Cube</td><td>IRASim</td><td> $5 \times 1 0 ^ { - 5 }$ </td><td>max 300k steps</td><td>step 300k</td></tr><tr><td>TwoRoom</td><td>LeWM</td><td></td><td></td><td>official LeWM checkpoint</td></tr><tr><td>TwoRoom</td><td>Sensorimotor-WM</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>10 epochs</td><td>epoch 10</td></tr><tr><td>TwoRoom</td><td>DINO-WM</td><td></td><td></td><td>official DINO-WM checkpoint</td></tr><tr><td>TwoRoom</td><td>DriftWorld</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>300-epoch cap</td><td>step 500k</td></tr><tr><td>TwoRoom</td><td>IRASim</td><td> $5 \times 1 0 ^ { - 5 }$ </td><td>max 300k steps</td><td>step 300k</td></tr><tr><td>Insert-Peg</td><td>LeWM</td><td> $5 \times 1 0 ^ { - 5 }$ </td><td>10 epochs</td><td>epoch 10</td></tr><tr><td>Insert-Peg</td><td>Sensorimotor-WM</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>10 epochs</td><td>epoch 10</td></tr><tr><td>Insert-Peg</td><td>DINO-WM</td><td> $5 \times 1 0 ^ { - 4 }$ </td><td>10 epochs</td><td>epoch 10</td></tr><tr><td>Insert-Peg</td><td>DriftWorld</td><td> $5 \times 1 0 ^ { - 5 }$ </td><td>max 100k steps</td><td>step 100k</td></tr><tr><td>Insert-Peg</td><td>IRASim</td><td> $5 \times 1 0 ^ { - 5 }$ </td><td>max 300k steps</td><td>step 300k</td></tr></table>

Table 7: Training details for the selected oracle-state predictor checkpoints.
<table><tr><td>Dataset</td><td>Model</td><td>selected LR</td><td>Configured duration</td><td>Selected checkpoint</td></tr><tr><td>Push-T</td><td>Toy-AdaLN</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td>100 epochs</td><td>epoch 100</td></tr><tr><td>Push-T</td><td>Toy-Action-Duplicate</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td>100 epochs</td><td>epoch 100</td></tr><tr><td>Push-T</td><td>Toy-DriftWorld</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>50 epochs, max 500k steps</td><td>step 500k</td></tr><tr><td>Cube</td><td>Toy-AdaLN</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td>100 epochs</td><td>epoch 100</td></tr><tr><td>Cube</td><td>Toy-Action-Duplicate</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td>100 epochs</td><td>epoch 100</td></tr><tr><td>Cube</td><td>Toy-DriftWorld</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>50 epochs, max 500k steps</td><td>step 500k</td></tr><tr><td>TwoRoom</td><td>Toy-AdaLN</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td>100 epochs</td><td>epoch 100</td></tr><tr><td>TwoRoom</td><td>Toy-Action-Duplicate</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td>100 epochs</td><td>epoch 100</td></tr><tr><td>TwoRoom</td><td>Toy-DriftWorld</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>50 epochs, max 500k steps</td><td>epoch 50</td></tr><tr><td>Insert-Peg</td><td>Toy-AdaLN</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td>100 epochs</td><td>epoch 100</td></tr><tr><td>Insert-Peg</td><td>Toy-Action-Duplicate</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td>100 epochs</td><td>epoch 100</td></tr><tr><td>Insert-Peg</td><td>Toy-DriftWorld</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>50 epochs, max 500k steps</td><td>epoch 50</td></tr></table>

## H.1 PLANNING EVALUATION CRITERIA

Table 8 summarizes the success criteria used for each environment. Some details are below.

TwoRoom. For TwoRoom, the goal is the dataset future agent position. A rollout is successful when $\| p _ { t } - p _ { g } \| _ { 2 } < 1 6$ , where $p _ { t }$ is the current agent position and $p _ { g }$ is the goal agent position. Distances are measured in pixels in the 224 × 224 environment coordinate system. Target and door positions are not used.

PushT. For PushT, the position error is computed over the 4D vector [agent , agent , block<sub>x</sub>, block<sub>y</sub>]. A rollout is successful when both $\| s _ { t } ^ { \mathrm { p o s } } - s _ { g } ^ { \mathrm { p o s } } \| _ { 2 } ~ < ~ 2 0$ and $d _ { \mathrm { a n g l e } } ( \theta _ { t } , \theta _ { g } ) \stackrel { \textstyle \sim } { < } \pi / 9$ hold at the same timestep. The angle distance accounts for object symmetries:

Table 8: Planning success criteria. Goals are defined by the dataset future state 25 raw steps after the initial state. An episode is successful if the criterion is satisfied at least once within 50 environment steps.
<table><tr><td>Environment</td><td>Planning success criterion</td></tr><tr><td>TwoRoom</td><td>The 2D Euclidean distance between the agent position and the goal agent position is less than 16 pixels. Target and door positions are not used.</td></tr><tr><td>PushT</td><td>The 4D Euclidean error over agent and block positions is less than 20 pixels, and the block angle error is less than  $\pi / 9$  radians. The angle error accounts for object</td></tr><tr><td>Cube</td><td>symmetries. The rollout must simultaneously satisfy: end-effector 3D position error  $\leq 0 . 0 4 \mathrm { { m } , }$  end- effector yaw error  $\leq 0 . 1 0$  rad, and cube 3D position error  $\leq 0 . 0 4$  m. Cube orientation, gripper opening, contact, and velocity are not used.</td></tr><tr><td>Insert Peg</td><td>The rollout must simultaneously satisfy: hand 2D position error  $\leq 0 . 0 4 1$  m, hand planar angle error  $\leq 0 . 1 0$  rad, peg 2D position error  $\leq 0 . 0 4 \ \mathrm { m } .$  and peg planar angle error  $\leq 0 . 1 0$  rad. Arm joints, touch, velocity, and the physical target or slot pose are not used.</td></tr></table>

square and + shapes use $9 0 °$ symmetry, Z and I shapes use $1 8 0 ^ { \circ }$ symmetry, and other shapes use 360<sup>◦</sup>.

Cube. For Cube, the goal is defined by the dataset future end-effector pose and cube position. A rollout is successful when $\| p _ { t } ^ { e e } \mathrm { ~ - ~ } p _ { g } ^ { e e } \| _ { 2 } ~ \leq ~ 0 . 0 4 \mathrm { m }$ , | wrap $( \psi _ { g } ^ { e e } \ : - \ : \psi _ { t } ^ { e e } ) | ~ \leq ~ 0 . 1 0 \mathrm { r a d }$ , and $\| p _ { t } ^ { c u b e } - p _ { q } ^ { c u b e } \| _ { 2 } \leq 0 . 0 4$ m all hold at the same timestep. Cube orientation, gripper opening, contact, and velocity are not used. Once satisfied, episode-level success remains true. Cube rollouts are not terminated early by this criterion.

Manipulator Insert Peg. For Manipulator Insert Peg, the goal is defined by the dataset future hand and peg poses. Positions are measured in the $( x , z )$ plane, and planar angles are computed from quaternion components $( q _ { w } , q _ { y } )$ as $2 \operatorname { a t a n 2 } ( q _ { y } , q _ { w } )$ . A rollout is successful when the hand position, hand angle, peg position, and peg angle errors are all within their thresholds: 0.04 m for positions and 0.10 rad for angles. Arm joints, touch sensors, velocity, and the physical target or slot pose are not used. This dataset-goal success metric differs from the native task success based on the DeepMind Control shaped reward.

## H.2 CONFIGURATION

We evaluated all true-state dynamics models using the same goal-conditioned receding-horizon controller. For each episode, the initial state and goal were sampled from the same dataset trajectory with a temporal offset of 25 environment steps. Planning was performed in the normalized action space using the cross-entropy method (CEM). Each candidate consisted of five action blocks, where each block contained five consecutive environment actions, yielding an effective planning horizon of 25 environment steps. At every planning round, CEM evaluated 300 candidate sequences for 30 iterations and refitted a factorized Gaussian distribution to the 30 lowest-cost elites. The objective was the squared Euclidean distance between the terminal predicted state and the goal in the model’s normalized state space. The final mean sequence was executed open-loop for all five action blocks before replanning. Episodes were evaluated for at most 50 environment steps, corresponding to at most two planning rounds.

## I RESULTS

## I.1 FULL RESULTS

This section reports the full numerical results used in the main analysis. We include absolute MPC planning success rates complementing the relative comparisons shown in the main text, as well as table of relative prediction errors with absolute base-environment rollout errors. For the five-step

prediction error, corresponding to 25 environment steps, Avoid-T is omitted because its episodes are too short to support this horizon.

Table 9: Absolute base-task rollout MSE at 5- and 25-step prediction horizons.
<table><tr><td colspan="6">Base-Env Horizon LeWM SMWM DINO-WM DriftWorld IRASim AdaLN Action Dup. Toy-Drift</td></tr><tr><td colspan="2">5 TwoRoom</td><td>0.065188 0.000058 0.033841</td><td>0.000154</td><td>0.003838 0.00000466</td><td>0.0000144 0.000051</td></tr><tr><td colspan="2">25</td><td>0.511843 0.000754 0.23277</td><td>0.002164</td><td>0.023646 0.000145</td><td>0.000636 0.00111671</td></tr><tr><td colspan="2">5 PushT</td><td>0.00743 0.02933 0.05124</td><td>0.001331</td><td>0.005813 0.000005</td><td>0.000028 0.000066</td></tr><tr><td colspan="2">25 0.16739</td><td>0.23039 0.48649</td><td>0.018599</td><td>0.050163 0.000229</td><td>0.001068 0.00388751</td></tr><tr><td colspan="2">5</td><td>0.0007170.002913 0.172737</td><td>0.001448</td><td>0.00939 0.003526</td><td>0.009356 0.054486</td></tr><tr><td colspan="2">Insert Peg 25</td><td>0.010678 0.024729 1.410328</td><td>0.013113</td><td>0.0593 0.053553</td><td></td></tr><tr><td colspan="2">5</td><td>0.00431 0.00203 0.15249</td><td>0.00095</td><td>0.01027</td><td>0.149251 0.404549</td></tr><tr><td colspan="2">Cube 25</td><td>0.07875 0.0193 0.89757</td><td>0.011023 0.072107</td><td>0.000056 0.000486</td><td>0.000045 0.0521936 0.000364 0.458214</td></tr></table>

Table 10: Base-relative one-step (5 environment step) prediction error. Values report the error ratio with respect to the corresponding base task.
<table><tr><td colspan="6">End-to-End World Models</td><td colspan="4">Oracle State Predictors</td></tr><tr><td>Env</td><td>Variant</td><td colspan="8">LeWM SMWM DINO-WM DriftWorld IRASim AdaLN Action Dup. Toy-Drift</td></tr><tr><td colspan="2">TwoRoom TR-Route</td><td>14.56</td><td>19.11</td><td>6.56</td><td>0.012</td><td>0.9</td><td>0.045</td><td>1.03</td><td>0.014</td></tr><tr><td rowspan="2"></td><td>AvoidT</td><td>4.14</td><td>5.2</td><td>5.77</td><td>0.71</td><td>0.77</td><td>1.26</td><td>1.25</td><td>1.32</td></tr><tr><td>StackT</td><td>11.33</td><td>3.13</td><td>3.38</td><td>2.54</td><td>1.36</td><td>501.14</td><td>113.55</td><td>70.24</td></tr><tr><td rowspan="2">PushT</td><td>Anti-goal</td><td>6.23</td><td>2.66</td><td>2.83</td><td>1.64</td><td>1.09</td><td>54.74</td><td>12.85</td><td>19.96</td></tr><tr><td>Avoid-Peg</td><td>6.83</td><td>2.54</td><td>1.3</td><td>7.01</td><td>7.1</td><td>1.45</td><td>0.62</td><td>0.08</td></tr><tr><td rowspan="2"></td><td>Insert Peg Extract-Peg</td><td>4.43</td><td>2.96</td><td>1.30</td><td>2.59</td><td>1.09</td><td>53.53</td><td>20.01</td><td>3.55</td></tr><tr><td>CubeHop</td><td>27.83</td><td>93.7</td><td>2.8</td><td>1.38</td><td>1.19</td><td>91.94</td><td>1.03</td><td>0.058</td></tr><tr><td rowspan="2">Cube</td><td>Cube Diagonal</td><td>1.71</td><td>5.8</td><td>1.12</td><td>1.09</td><td>0.97</td><td>940.34</td><td>1026.48</td><td>0.58</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 11: Base-relative five-step (25 environment step) prediction error. Values report the error ratio with respect to the corresponding base task.
<table><tr><td colspan="6">end-to-end world models</td><td colspan="4">oracle state predictors</td></tr><tr><td>Env</td><td>Variant</td><td>LeWM SMWM DINO-WM DriftWorld IRASim AdaLN Action Dup. Toy-Drift</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">TwoRoom TR-Route</td><td>11.2 41.6</td><td>9.66</td><td>0.01</td><td>0.82</td><td>0.062</td><td>0.816</td><td>0.01</td></tr><tr><td rowspan="2">PushT</td><td>StackT</td><td>9.83</td><td>4.56 3.8</td><td>2.34</td><td>1.44</td><td>158.5</td><td>47.57</td><td>22.09</td></tr><tr><td>Anti-goal</td><td>8.49 4.27</td><td>3.94</td><td>1.72</td><td>1.28</td><td>29.97</td><td>12.68</td><td>8.96</td></tr><tr><td rowspan="2"></td><td>Avoid-Peg</td><td>9.73</td><td>16.69</td><td>2.8</td><td>1.14</td><td>1.12</td><td>0.67 0.42</td><td>0.09</td></tr><tr><td>Insert Peg Extract-Peg</td><td>4.08 7.15</td><td>1.78</td><td>2.75</td><td>1.19</td><td>34.39</td><td>12.52</td><td>4.49</td></tr><tr><td rowspan="2">Cube</td><td>CubeHop</td><td>29.64</td><td>139.07</td><td>4.12</td><td>1.45</td><td>1.37</td><td>245.45 3.47</td><td>0.102</td></tr><tr><td>Cube Diagonal</td><td>1.55</td><td>11.38 1.14</td><td></td><td>1.05</td><td>0.97 841.69</td><td>1202.22</td><td>0.46</td></tr></table>

## I.2 FACTOR SUPPORT VERSUS OBSERVATION SUPPORT

Our compositional shifts are defined at the level of physical factors: test queries recombine factor values that are individually observed in the training data. However, this does not imply that the resulting observations are in-distribution for an end-to-end encoder.

Let ${ \mathcal { Z } } ^ { \mathrm { t r } }$ denote the joint support of physical factors under the training trajectories, and let $\mathcal { Z } ^ { \times } =$ $\textstyle \prod _ { k = 1 } ^ { K } { \mathcal { Z } } _ { k _ { . } } ^ { \mathrm { t r } }$ denote the Cartesian product of their marginal supports. A compositional query may satisfy ${ \hat { { \mathcal { Z } } } } ^ { \mathrm { t e } } \subseteq { \mathcal { Z } } ^ { \times }$ while still containing factor combinations absent from ${ \mathcal { Z } } ^ { \mathrm { t r } }$

Table 12: Absolute planning success rate (%) across different models and tasks.
<table><tr><td>Task</td><td>LeWM SMWM DINO-WM DriftWorld</td><td></td><td></td><td></td><td>IRASim</td><td>AdaLN</td><td>Action Dup.</td><td>Drift Toy</td></tr><tr><td>TwoRoom TR-Route</td><td>89.30 60.70</td><td>100.00 64.70</td><td>100.00 68.00</td><td>90.00 76.00</td><td>76.00 63.30 5.30</td><td>100.00 100.00 92.70</td><td>99.30 100.00</td><td>100.00 100.00</td></tr><tr><td>Push-T AvoidT StackT Push-T Antigoal</td><td>84.70 3.30 20.70 34.00</td><td>84.70 4.70 20.00 44.00</td><td>74.00 6.70 8.70 27.30</td><td>43.30 0.70 10.70 14.70</td><td>0.00 2.00 5.30</td><td>100.00 70.70 79.30</td><td>92.00 100.00 66.70 81.30</td><td>89.30 99.30 54.00 75.30</td></tr><tr><td>Insert-Peg Avoid-Peg Extract-Peg</td><td>26.67 31.33 8.00</td><td>70.00 32.67 23.33</td><td>35.33 28.67 15.33</td><td>63.33 68.00 23.33</td><td>2.43 34.00 9.33</td><td>74.00 92.00 16.66</td><td>71.33 89.33 25.33</td><td>71.33 80.67 19.33</td></tr><tr><td>Cube Cube-Hop Cube-Diagonal</td><td>30.70 17.30 28.00</td><td>72.70 40.70 53.30</td><td>60.00 51.30 58.70</td><td>30.70 64.70 42.00</td><td>10.70 12.70 9.33</td><td>30.00 95.30 20.00</td><td>20.70 51.30 12.00</td><td>25.30 79.30 22.00</td></tr></table>

End-to-end world models observe $s = f ( z )$ , where $f$ is the rendering map. Thus the corresponding observation supports are $S ^ { \mathrm { t r } } = f ( \mathcal { Z } ^ { \mathrm { t r } } )$ and $S ^ { \times } = f ( \mathcal { Z } ^ { \times } )$ . When $\bar { \boldsymbol { f } }$ mixes factors, unseen factor combinations in $\mathcal { Z } ^ { \times } \setminus \mathcal { Z } ^ { \mathrm { t r } }$ can induce observations in ${ \dot { S } } ^ { \times } \setminus S ^ { \mathrm { t r } }$

Therefore, a query can be factor-wise in support while still creating an observation-level shift for an end-to-end encoder. This is why we separately analyze oracle-state predictors: they remove the encoder-inference problem and isolate whether the learned dynamics model can generalize across recombined state-action factors.

## I.3 ADDITIONAL OBSERVATIONS

Prediction Space Affects Metric Interpretation We do not use the rollout results to directly rank JEPA-style and generative world models. These models predict in different spaces: JEPA-style models are evaluated in latent space, whereas generative models are evaluated in pixel or reconstruction space. As a result, absolute prediction errors are not directly comparable across model families, and we primarily interpret rollout degradation relative to each model’s own base-task performance.

Under this relative comparison, generative models often show smaller rollout degradation under some goal shifts, while several JEPA-style models show larger increases. However, this tendency does not translate consistently to planning: a model can show stable rollout error but poor MPC success, or larger rollout degradation while remaining competitive in control. We therefore treat prediction-space effects as an interpretation caveat rather than a model-family ranking. The results suggest that rollout robustness and downstream control utility are related but distinct properties.

Oracle state access does not guarantee better planning. Interestingly, access to the oracle state does not always lead to stronger planning performance. In Cube, several end-to-end models achieve higher planning success than their oracle-state counterparts, both on the base task and on CubeDiagonal. This should not be interpreted as evidence that learned latents are generally superior to physical state representations, since the end-to-end and oracle predictors differ in more than just their input space.

Rather, these results suggest that state observability and planning suitability are distinct properties. Oracle states preserve the underlying physical factors, but the resulting geometry may not be the most favorable one for the planner’s cost and optimization procedure. Conversely, a learned representation may, in some cases, emphasize task-relevant variation or compress nuisance factors in a way that makes goal-directed optimization easier. We therefore treat planning performance as depending not only on how much state information is available, but also on how that information is organized in the representation space.

## J TOWARD COMPOSITIONAL GENERALIZATION IN WORLD MODELS

Our experiments empirically show that removing visual abstraction is not sufficient for reliable goal-shift generalization: free-space dynamics often transfer across new state–action compositions, whereas contact-rich dynamics remain substantially harder even when the predictor receives taskrelevant physical state directly. This raises a dynamics-level question: what structure must a world model learn for its dynamics to remain reusable under compositional shifts?

Ideally, we want to solve AnyGoal by learning reusable dynamics that extrapolate to the combinatorial space. That is, we want a learned transition function $\hat { T }$ that matches the observed training transitions on ${ \mathcal { C } } ^ { \mathrm { t r } }$ , while also predicting correctly on novel compositions in ${ \mathcal { C } } ^ { \times } \backslash { \mathcal { C } } ^ { \mathrm { t r } }$ . We separate this question into two parts: identifying reusable mechanisms on the training support, and extrapolating those mechanisms to novel compositions.

Note. The discussion below should be read as a conditional statement, not a claim that reusable mechanisms are identifiable from expert rollouts alone. Rather, we give one possible sufficient condition: if sparse regime-conditioned local mechanisms are identified on training support and test goals query only locally covered active inputs, then the model can generalize to globally novel state–action compositions.

## J.1 IDENTIFYING REUSABLE MECHANISMS ON THE TRAINING SUPPORT

To discuss identifiability, we first need to specify what structure the true transition may have. One observation from our results is that world models often face multiple regimes, such as free motion, contact, grasping, or collision.

In a single-regime setting with little interaction, such as free-space navigation of an agent, one may expect a transition of the form $z _ { t + 1 } = f ( z _ { t } ) + g ( a _ { t } )$ . In this favorable case, identifiability and compositional generalization are easier to obtain, for example through an additive function class (Dong & Ma, 2022; Lachapelle et al., 2023; Wiedemer et al., 2024b;a; Mahajan et al., 2025).

However, world-model dynamics are often regime-dependent and more complex than a purely additive and separated form. The action effect is usually better written as $g _ { r } ( z _ { t } , a _ { t } )$ rather than $g ( a _ { t } )$ meaning that the transition cannot be decomposed into simple independent state-only and actiononly terms. A simple additive form is therefore too limited for the setting we want to characterize.

We instead consider a factorized transition with regime-dependent local mechanisms. For each factor $k ,$ let

$$
z _ { k , t + 1 } = T _ { k , r _ { t } } ( u _ { k , t } ) , \qquad u _ { k , t } = \big ( z _ { \mathrm { p a } ( k ) , t } , a _ { \mathrm { p a } ( k ) , t } \big ) ,
$$

where $\operatorname { p a } ( k )$ denotes the local parents of factor $k ,$ and $r _ { t } \in \{ 1 , \ldots , R \}$ denotes a discrete local regime. The transition kernel is specified as follows

$$
T _ { r _ { t } } ( z _ { t } , a _ { t } ) = \big ( T _ { 1 , r _ { t } } ( u _ { 1 , t } ) , \dots , T _ { K , r _ { t } } ( u _ { K , t } ) \big ) .
$$

Since multiple regime–mechanism decompositions can explain the same training transitions, we next state the favorable assumptions under which the discussion applies: a Markov, goal-invariant oracle state, and training-support identification of the relevant local mechanisms up to observational equivalence.

Assumption 1 (Markov and invariant oracle state) The oracle state $z _ { t }$ is Markov, and the transition law is invariant across training and test goals:

$$
p ( z _ { t + 1 } \mid z _ { \leq t } , a _ { \leq t } ) = p ( z _ { t + 1 } \mid z _ { t } , a _ { t } ) , \qquad T _ { \mathrm { t r } } ( z _ { t + 1 } \mid z _ { t } , a _ { t } ) = T _ { \mathrm { t e } } ( z _ { t + 1 } \mid z _ { t } , a _ { t } ) .
$$

Thus, goal shifts change the behavior distribution, but not the environment dynamics.

Assumption 2 (Identification of local mechanisms) For each state factor $k ,$ suppose that the interaction regime $r _ { t }$ and the local input $u _ { k , t } ~ = ~ u _ { k } ( z _ { t } , a _ { t } )$ are observed or identifiable from the training transitions, and that the corresponding regime can be consistently inferred at prediction time.

$$
z _ { k , t + 1 } = T _ { k , r _ { t } } ( u _ { k , t } ) , \qquad \hat { z } _ { k , t + 1 } = \hat { T } _ { k , r _ { t } } ( u _ { k , t } ) .
$$

For each factor and regime, define the training support

$$
\mathcal { U } _ { k , r } ^ { \mathrm { t r } } : = \bigcup _ { t = 0 } ^ { H - 1 } \operatorname { s u p p } _ { p _ { \mathrm { t r } } } ( U _ { k , t } \mid R _ { t } = r ) .
$$

Let $\mathcal { L } _ { \mathrm { t r } } ( \tilde { T } )$ denote the population objective used to train the world model. We assume that this objective identifies the regime-conditioned mechanisms on their training support: any population minimizer T<sup>˜</sup> satisfies

$$
\tilde { T } _ { k , r } ( u ) = T _ { k , r } ( u ) , \qquad \forall u \in \mathcal { U } _ { k , r } ^ { \mathrm { t r } } ,
$$

for every $k , r ,$ up to observational equivalence and regime relabeling.

For a model $\hat { T }$ learned from finite data, we allow estimation and optimization error

$$
\epsilon _ { k , r } : = \operatorname* { s u p } _ { u \in \mathcal { U } _ { k , r } ^ { \mathrm { t r } } } \| \hat { T } _ { k , r } ( u ) - T _ { k , r } ( u ) \| _ { 2 } ,
$$

and assume $\epsilon _ { k , r } \to 0$ as the amount of relevant training data grows and the training objective is optimized consistently.

## J.2 EXTRAPOLATING IDENTIFIED MECHANISMS TO NOVEL COMPOSITIONS

Training-support identification does not by itself imply compositional generalization. Two mechanisms can agree on all observed training transitions and still disagree on a novel state–action composition induced by a goal shift. Recombination therefore requires additional structure that separates coordinates that affect a mechanism from coordinates that are only part of the global state.

Additive or separable functions provide one favorable case, which allow us to reuse the learned mechanisms for each factor $z _ { k , t + 1 }$ , but contact-rich world-model dynamics are often regimedependent rather than purely additive. We propose a simpler scenario where each transition mechanisms $T _ { k , r _ { t } }$ depends sparsely on the inputs $s _ { t } , a _ { t }$ . For a fixed regime $r ,$ let $\boldsymbol { x } _ { t } = \left( \boldsymbol { z } _ { t } , \boldsymbol { a } _ { t } \right)$ , then

$$
z _ { k , t + 1 } = T _ { k , r _ { t } } ( x _ { t } ) .
$$

Sparsity of $\nabla _ { x } T _ { k , r _ { t } } ( x )$ implies that only a subset of coordinates affects the k-th factor transition, while the remaining coordinates may change the global composition without changing this local mechanism.

Assumption 3 (Sparse local mechanisms) For each factor k and regime $^ { r , }$ , there exists a fixed coordinate set $P _ { k } \subseteq \{ 1 , \dots , \dim ( x ) \}$ , with $| P _ { k } | \ll$ dim(x), such that

$$
T _ { k , r } ( x ) = T _ { k , r } ( x _ { P _ { k } } ) .
$$

Equivalently,

$$
{ \frac { \partial T _ { k , r } ( x ) } { \partial x _ { j } } } = 0 \qquad \forall j \not \in P _ { k } .
$$

We write $u _ { k , t } = x _ { P _ { k } , t } f o r$ the inputs relevant to the k-th transition mechanism.

Sparse local mechanism can allow us to create favorable cases when a goal shift creates a novel global composition without creating novel local mechanism inputs.

Assumption 4 (Local support coverage) Let

$$
\mathcal { U } _ { k , r } ^ { \mathrm { t e } } : = \bigcup _ { t = 0 } ^ { H - 1 } \operatorname { s u p p } _ { p _ { \mathrm { t e } } } ( U _ { k , t } \mid R _ { t } = r )
$$

denote the test support of the inputs of transition mechanism for factor k. For every factor k, assume

$$
\mathcal { U } _ { k , r } ^ { \mathrm { t e } } \subseteq \mathcal { U } _ { k , r } ^ { \mathrm { t r } } .
$$

Thus, the global state–action composition may be novel, but every local mechanism is queried only on local inputs covered during training

Intuition. Within a fixed regime, sparse local support separates active & inactive coordinates. The active coordinates $u _ { k }$ determine the transition mechanism for the factor $z _ { k }$ , while coordinates outside $u _ { k }$ can be recombined to create globally novel state–action compositions without changing that local mechanism. Thus, the sparser the mechanism, the larger the set of globally novel compositions that can be covered by the same local transition rule.

Proposition 1 (Local active-coordinate coverage controls global error) Let $\mathcal { L } _ { \mathrm { t r } } ( \tilde { T } )$ denote the learning objective used to train the world model. Suppose that we perfectly optimize the world model on training support, then under assumption 1 and 2, we identify the learned model such that the true transition and learned transition share the same regime-conditioned local decomposition,

$$
T ( z _ { t } , a _ { t } ) = \big ( T _ { 1 , r _ { t } } ( u _ { k , t } ) , \dots , T _ { K , r _ { t } } ( u _ { k , t } ) \big ) ,
$$

and

$$
\hat { T } ( z _ { t } , a _ { t } ) = \big ( \hat { T } _ { 1 , r _ { t } } ( u _ { k , t } ) , \dots , \hat { T } _ { K , r _ { t } } ( u _ { k , t } ) \big ) .
$$

Assume that, for every factor k and regime r, we satisfy the assumption of sparse local mechanisms (assumption 3) and local support coverage (assumption 4).

Then, for any test composition $( z _ { t } , a _ { t } ) \in \mathcal { C } ^ { \mathrm { t e } }$

$$
\left\| \hat { T } ( z _ { t } , a _ { t } ) - T ( z _ { t } , a _ { t } ) \right\| _ { 2 } \leq \left( \sum _ { k = 1 } ^ { K } \epsilon _ { k , r _ { t } } ^ { 2 } \right) ^ { 1 / 2 } .
$$

where

$$
\epsilon _ { k , r } : = \operatorname* { s u p } _ { u \in \mathcal { U } _ { k } ^ { \mathrm { t r } } } \left\| \hat { T } _ { k , r } ( u ) - T _ { k , r } ( u ) \right\| _ { 2 } .
$$

Consequently, $i f \ \epsilon _ { k , r } \ \to \ 0$ for all $k , r ,$ then the world model generalizes to novel combinations present in the test support.

Proof. Take any test composition $( z _ { t } , a _ { t } ) \in \mathcal { C } ^ { \mathrm { t e } }$ with interaction regime r<sub>t</sub>. By Assumption 4, for every factor k,

$$
u _ { k , t } \in \mathcal { U } _ { k , r _ { t } } ^ { \mathrm { t e } } \subseteq \mathcal { U } _ { k , r _ { t } } ^ { \mathrm { t r } } .
$$

Thus, each local mechanism is evaluated within its regime-conditioned training support. By Assumption 2,

$$
\left\| \hat { T } _ { k , r _ { t } } ( u _ { k , t } ) - T _ { k , r _ { t } } ( u _ { k , t } ) \right\| _ { 2 } \leq \epsilon _ { k , r _ { t } } .
$$

Since the true and learned transitions share the same factorized decomposition,

$$
\begin{array} { r l r } {  { \Big \| \hat { T } ( \boldsymbol { z } _ { t } , \boldsymbol { a } _ { t } ) - T ( \boldsymbol { z } _ { t } , \boldsymbol { a } _ { t } ) \Big \| _ { 2 } ^ { 2 } = \displaystyle \sum _ { k = 1 } ^ { K } \Big \| \hat { T } _ { k , r _ { t } } ( \boldsymbol { u } _ { k , t } ) - T _ { k , r _ { t } } ( \boldsymbol { u } _ { k , t } ) \Big \| _ { 2 } ^ { 2 } } } \\ & { } & { \leq \displaystyle \sum _ { k = 1 } ^ { K } \epsilon _ { k , r _ { t } } ^ { 2 } . } \end{array}
$$

Taking the square root completes the proof.

Discussion. This construction should be read as an illustrative sufficient condition rather than a complete solution. Sparse local mechanisms reduce a high-dimensional global coverage problem to smaller local coverage problems, but they do not by themselves solve mechanism identification, regime discovery, or extrapolation to genuinely new local interactions. The broader challenge is to develop world-model architectures, objectives, and data-collection strategies that identify reusable action-conditioned mechanisms and impose enough structure on their interactions to support generalization beyond expert rollouts.

## K ANYGOAL LIBRARY AND REPRODUCIBILITY

We release AnyGoal as an open-source Python library for collecting goal-shifted datasets, training world models, and evaluating rollout prediction and MPC planning. The library provides both Python APIs and a unified CLI, allowing researchers to reproduce the benchmark experiments and evaluate additional models under the same protocol. AnyGoal builds on Stable-WorldModel (Maes et al., 2026a) and retains the original model implementations and numerical protocols. Complete installation instructions and API references are available in the online documentation.

## K.1 INSTALLATION AND DATASET ACCESS

The library can be installed with evaluation dependencies using uv.

```batch
pip install uv
uv venv --python 3.10
source .venv/bin/activate
uv pip install "anygoal[eval]"
```

The default installation provides dataset access, collection, validation, and simulation environments, while the eval and all extras add evaluation and training dependencies, respectively.

The CLI exposes a registry of supported datasets and models.

# Inspect available datasets and models   
anygoal list datasets   
anygoal list models   
# Inspect default data and checkpoint paths   
anygoal config path data   
anygoal config path checkpoints

Datasets and pretrained checkpoints can be downloaded from Hugging Face through the artifact interface.

```shell
# Download training and test datasets
anygoal artifacts download --dataset pusht
anygoal artifacts download --dataset avoidt
# Download a pretrained model checkpoint
anygoal artifacts download --model lewm
```

Model integrations use the corresponding upstream implementations. Their source code can be installed separately.

anygoal models list   
anygoal models install lewm

Dataset Collection Beyond the released evaluation episodes, AnyGoal provides tools for generating additional goal-shifted trajectories. Dataset collection is available through both Python and CLI interfaces, which use the same in-package collectors.

The following example collects 400 Avoid-T episodes using eight parallel environments.

```python
import anygoal
dataset = anygoal.collect(
"avoidt",
num_episodes=400,
num_parallel=8,
seed=0,
```

The equivalent collection procedure can be invoked from the terminal.

```shell
anygoal collect avoidt --num-episodes 400 --num-parallel 8
# Collect another goal-shifted variant
anygoal collect cubediagonal --output /path/to/cubediagonal.h5
```

## K.2 WORLD-MODEL EVALUATION

AnyGoal provides standardized interfaces for evaluating a base-trained world model on both its original task and goal-shifted variants. The same model checkpoint can be used for rollout prediction and downstream MPC planning, enabling comparisons across evaluation metrics.

Planning evaluation. The following commands evaluate LeWM on Push-T and Avoid-T.

```shell
# Original training goal
anygoal eval planning --model lewm --dataset pusht --seed 0
# Unseen goal
anygoal eval planning --model lewm --dataset avoidt --seed 0
```

Rollout evaluation. The corresponding rollout-prediction evaluation uses the same model and dataset identifiers.

```shell
anygoal eval rollout --model lewm --dataset pusht --seed 0
anygoal eval rollout --model lewm --dataset avoidt --seed 0
```

By default, the evaluation commands resolve the checkpoint associated with the selected model and base-task family through the shared artifact catalog. Users can also specify a checkpoint explicitly with --checkpoint or download missing artifacts with --download.

The evaluation interfaces additionally support explicit random seeds, allowing comparisons across repeated runs.

## K.3 REPRODUCIBILITY AND EXTENSIBILITY

AnyGoal organizes datasets, checkpoints, and supported model implementations through shared registries. Dataset and model identifiers, artifact locations, and evaluation configurations can be specified explicitly, reducing ambiguity when reproducing experiments. The library exposes its core functionality through Python APIs, including anygoal.collect, anygoal.train, anygoal.evaluate rollout, and anygoal.plan. These interfaces complement the CLI and support integration into existing research workflows.

The released benchmark contains 400 episodes per goal-shifted variant, while the collection tools allow researchers to construct additional evaluation datasets. Together with the standardized evaluation interfaces, this enables future world models to be tested on the same controlled goal shifts without reproducing the environment and evaluation infrastructure. The source code, configuration files, and implementation instructions are available at github.com/galilai-group/AnyGoal.