# FocusVLA: Future-Intent Guided Dynamic View Routing for Vision-Language-Action Models

Huixin Xu<sup>1,2,3,4</sup> Ziya Yu<sup>1,2,3,5</sup> Peng Li<sup>2,3,1,∗</sup>

<sup>1</sup> Nanjing Institute of Software Technology

<sup>2</sup> Institute of Software, Chinese Academy of Sciences

<sup>3</sup> University of Chinese Academy of Sciences, Nanjing

<sup>4</sup> Nanjing University of Posts and Telecommunications <sup>5</sup> Hohai University Corresponding author

![](images/d9aec351115ba581cc0c7df1af437dca89a89642b1b52b841c80958178756480.jpg)

![](images/1e6c50b53f8fe9f250260de5026e1d218d69f8d1107d0d155122ff0295a7c5ce.jpg)  
Figure 1: Overview of FocusVLA. Static multi-view fusion can cause view-dependency inertia, making the policy rely on a fixed view across task stages. FocusVLA predicts future action intent from multimodal observations and action history, and uses it to guide dynamic view routing during action generation, improving multi-view utilization in both simulation and real-world tasks.

Abstract: Vision-language-action (VLA) models have become an important paradigm for robotic manipulation by unifying visual perception, language understanding, and action generation. Since reliable manipulation requires sufficient observation of objects, end-effectors, and their spatial relationships, multi-view inputs are widely used to provide complementary spatial information. However, our experiments show that existing VLA models do not fully exploit such multiview complementarity. The policy tends to persistently rely on the wrist view, lacking the ability to dynamically adjust global-local view dependency across manipulation stages, which limits stage adaptability and robustness. To address this issue, we propose FocusVLA, a stage-aware multi-view routing framework. FocusVLA predicts future interaction intent from historical actions and uses it to guide dynamic view routing during action generation. By adaptively coordinating global and local observations across different task stages, FocusVLA improves multi-view utilization. On RoboTwin 2.0, FocusVLA improves the average success rate over the strongest baseline π by 10.10 pp. In real-world evaluation, FocusVLA improves the success rate from 71.7% to 85.0%, achieving a 13.3% gain over π<sub>0.5</sub>. Moreover, the intent prediction and dynamic routing modules of FocusVLA can be integrated into existing VLA frameworks without modifying their core architectures.

Keywords: Robotic manipulation, Imitation learning, VLA models

## 1 Introduction

Recent Vision-Language-Action (VLA) models [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] have shown strong potential for generalist robotic manipulation by unifying visual perception, language understanding, and action generation. In real-world manipulation, robust action generation often requires visual observations that can handle occlusions, viewpoint variations, and fine-grained interactions. Therefore, multi-view inputs are widely used to provide complementary visual information [12, 13, 14]: third-person cameras provide global scene context for target localization and motion planning, while wrist cameras provide local cues for alignment, grasping, and contact-rich execution. However, existing VLA models typically fuse visual observations with language instructions and robot states into a fixed context for action generation. Although this paradigm effectively integrates multimodal information, it rarely considers that the policy’s visual dependency may change across manipulation stages.

Our empirical analysis further shows that, under the static fusion paradigm, the model is prone to view-dependency inertia during action generation: the policy may develop a strong reliance on a particular view at an early task stage according to the current manipulation demand. Although such reliance may be reasonable at the early stage, the lack of explicit awareness of task progress causes the early-formed observation preference to persist into later stages, making it difficult for the policy to dynamically adjust the contributions of different views to action generation according to chang ing manipulation requirements. This phenomenon suggests that improving multi-view utilization requires not only fusing multi-view information, but also enabling the policy to dynamically select and coordinate different view observations during action execution. Motivated by these observations, we aim to mitigate the stage-view misalignment caused by static fusion. Achieving this goal requires answering two key questions: (1) how can a VLA policy acquire a representation that reflects task progress and use it to infer the visual requirements of upcoming manipulation stages? (2) how can such a representation be transformed into effective multi-view routing signals without disrupting the pretrained VLA backbone?

To answer these questions, we propose FocusVLA, a stage-aware dynamic view routing framework for multi-view VLA policies, as shown in Fig. 1. FocusVLA predicts future action intent from multimodal observations and short-term action history, and uses it to generate action-step-wise routing weights that dynamically coordinate visual features from different views during action generation. The framework is non-invasive, requiring no modification to the visual encoder, language model, or action-generation backbone. We evaluate FocusVLA in both simulation and real-world settings. The simulation experiments cover all 50 RoboTwin [15] tasks, and the real-world experiments include 6 representative manipulation tasks with diverse object configurations, viewpoint variations, and interaction requirements. Results show that FocusVLA consistently outperforms static multiview fusion baselines, demonstrating its effectiveness in mitigating stage-view misalignment and improving policy adaptability.

In summary, our contributions are: (1) We identify and quantify a temporal failure in static multiview VLA policies: view dependence formed at earlier stages persists after visual requirements change, causing stage–view misalignment. (2) We propose FocusVLA, a task-progress-aware visual modulation framework that converts future action intent into action-step-wise view-routing weights for non-invasive dynamic multi-view modulation. (3) We evaluate FocusVLA on all 50 RoboTwin simulation tasks and 6 real-world manipulation tasks, demonstrating improved adaptability and robustness over static multi-view baselines.

## 2 Related Works

## 2.1 Vision-Language-Action Models.

Vision-Language-Action (VLA) models unify visual perception, language understanding, and action generation for robotic control [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]. Recent models have improved generalist manipulation through diverse action generation paradigms, model scaling, and data scaling [27, 28]. However, most existing VLA models still follow a relatively fixed perception-to-action pipeline. As shown in Fig. 2, visual observations are first encoded into image tokens or visual features, fused with language instructions and robot states, and then provided as context to the action generation module [2, 3, 7, 17].

While this paradigm effectively integrates multimodal information, it rarely considers that the policy’s visual dependency may change across different manipulation stages. This limitation becomes especially important in multi-view settings, where global and local views provide complementary but stagedependent information [12, 13, 14, 29, 30, 31]. Unlike prior works that mainly improve action decoders or scale VLA backbones, FocusVLA introduces a

![](images/c70f6cba489a454219fd7700d51961a139471c12a52fa22e44dafd2b5ce2954a.jpg)  
Figure 2: Fusion pathway of multi-view perception in VLA models.

lightweight, architecture-agnostic view-routing module on the action-generation side. By predicting future action intent, FocusVLA dynamically modulates global and local visual contexts, enabling stage-aware multi-view adaptation without modifying existing VLA architectures.

## 2.2 Multi-View Perception and Dynamic Visual Routing.

Multi-view perception provides complementary information for robotic manipulation [12, 14]: third-person views capture global scene context, while wrist views provide fine-grained hand–object interaction cues. Existing methods exploit such observations through representation learning, 3D reconstruction, feature aggregation, cross-attention, or perspective-adaptive fusion [32, 33, 34, 35, 36, 37, 38]; recent VLA-oriented works further demonstrate the value of multi-view inputs for action generation [39, 40, 41, 42]. Most methods nevertheless adopt fixed view organization or generic fusion. BFA [13] addresses redundancy from equally treating current multi-view features by predicting vision-based view weights and fusing the features before an imitation policy. FocusVLA instead addresses temporal mismatch between learned view dependence and changing future action requirements in chunked VLA. The two methods differ in four key aspects: (1) prediction horizon: BFA produces one current-view allocation, whereas FocusVLA predicts H step-specific future routes from multimodal context; (2) fusion site: BFA fuses views before the policy, whereas Fo cusVLA performs per-step fusion within action generation; (3) label construction: BFA uses offline VLM annotations and task-specific mappings, whereas FocusVLA derives task-agnostic labels from trajectories; and (4) intent token: FocusVLA introduces chunk-level intent conditioning absent from BFA. Together, these designs enable future-intent-guided coordination of global and local visual contexts without altering the core VLA backbone.

## 3 Method

## 3.1 Motivation: Diagnosing Stage-Invariant View Dependency in Multi-View VLA

To diagnose view usage in existing multi-view VLA models, we analyze the $\pi _ { 0 . 5 }$ [2] multiview baseline by computing per-step attention ratios over global and local visual contexts. At action step t, $a _ { t } ^ { \mathrm { g l o b a l } }$ and $a _ { t } ^ { \mathrm { l o c a l } }$ denote the normalized attention ratios for the global view and aggregated local views, respectively, with $a _ { t } ^ { \mathrm { g l o b a l } } + \overline { { a } } _ { t } ^ { \mathrm { l o c a l } } = 1$ . When multiple local cameras are used, their attention ratios are summed as $a _ { t } ^ { \mathrm { l o c a l } }$ , allowing a unified comparison between global and local reliance. We quantify view usage with two metrics. The first is the Dominance Margin, which measures global-

![](images/794d3ee15e456cfe94dac03712fb78f502ada490396198205b7b5e1e544118d6.jpg)  
Figure 3: Motivation analysis of stage-insensitive view dependency in multi-view VLA models.

local imbalance:

$$
D _ { t } = \left| a _ { t } ^ { \mathrm { g l o b a l } } - a _ { t } ^ { \mathrm { l o c a l } } \right| , \qquad D = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } D _ { t } .\tag{1}
$$

The second is the temporal variation of local-view reliance:

$$
L _ { t } = a _ { t } ^ { \mathrm { l o c a l } } , \qquad \Delta L _ { t } = L _ { t } - L _ { t - 1 } , \qquad V _ { \Delta L } = \mathrm { V a r } \left( \{ \Delta L _ { t } \} _ { t = 2 } ^ { T } \right) .\tag{2}
$$

A larger D indicates stronger dominance of one visual context, while a smaller $V _ { \Delta L }$ suggests weaker adjustment of visual reliance over time. Empirically, both successful and failed rollouts exhibit imbalanced global-local reliance, typically biased toward the local visual context. This suggests that local-view preference itself is common and not necessarily harmful. However, failed rollouts show a larger Dominance Margin and a smaller $V _ { \Delta L }$ , indicating stronger and less adaptive local-view dependency over time.

Since attention ratios do not necessarily imply causal stage-conditioned view masking experiment on the Stack Blocks Three task, using the $\pi _ { 0 . 5 }$ [2] multiview baseline. For each masking setting, we evaluate 20 rollouts. During global planning stages, we separately mask the global view and the local view. If the policy effectively uses global observations for plan-

Table 1: Stage-conditioned View Masking.
<table><tr><td>Model</td><td>Masking Mode</td><td>SR</td><td>∆SR</td></tr><tr><td>Baseline</td><td>No Mask</td><td>85.0%</td><td></td></tr><tr><td>Baseline</td><td>Mask Global@Planning</td><td>65.0%</td><td> $- 2 0 . 0 \mathrm { p p }$ </td></tr><tr><td>Baseline</td><td>Mask Local@Planning</td><td>20.0%</td><td>-65.0 pp</td></tr></table>

ning, masking the global view should cause a larger performance drop. However, Table 1 shows the opposite pattern: masking the global view reduces the success rate from 85.0% to 65.0%, whereas masking the local view causes a much larger drop to 20.0%. This suggests that the static multi-view baseline remains highly dependent on local observations even during global-planning stages.

Together, the attention analysis and masking intervention provide complementary evidence for stage-insensitive view dependency. The attention metrics indicate that failed rollouts exhibit stronger local-view dominance and weaker temporal adjustment of visual reliance, while the intervention re sults further show that the baseline fails to shift its reliance toward the global view when global scene context is needed. This limitation motivates FocusVLA, which uses future intent to dynami cally regulate global and local visual contexts during action generation.

## 3.2 The FocusVLA Model Architecture

As shown in Fig. 4, FocusVLA is built upon the $\pi _ { 0 . 5 } \ [ 2 ]$ multi-view VLA backbone. Given global and local observations $o _ { t } = \{ I _ { t } ^ { g } , I _ { t } ^ { l } \}$ , a language instruction $L ,$ robot state $s _ { t } .$ , and short-term action history $h _ { t } .$ , FocusVLA predicts a future action chunk $A _ { t } = [ a _ { t } , a _ { t + 1 } , \dotsc , a _ { t + H - 1 } ]$ . Different from static multi-view fusion, FocusVLA uses future action trends to guide view dependency, allowing visual reliance to evolve with task progress. It consists of two modules: Future Intent Bridge, which predicts future interaction intent and produces a chunk-level intent token with stepwise routing weights $\alpha _ { 1 : H }$ , and Action-Side Dynamic Routing, which uses these signals to regulate the contributions of global and local visual contexts during action generation. This design enables stage-aware multi-view modulation without modifying the pretrained VLA backbone.

## 3.2.1 Future Intent Bridge

Future Intent Bridge aims to infer future interaction trends from the current multimodal context. Given the vision-language features from the VLA prefix backbone, together with the robot state and short-term action history, the module first encodes recent action dynamics into a temporal summary. It then performs cross-modal attention, using language features as queries and visual/state-history features as keys and values, to obtain a contextualized representation for future intent prediction.

Based on this representation, the bridge predicts a step-wise intent sequence over the future action horizon, $I _ { t } = [ \dot { i } _ { t } ^ { ( 1 ) } , i _ { t } ^ { ( 2 ) } , \dots , i _ { t } ^ { ( H ) } ]$ , where $i _ { t } ^ { ( k ) } \in [ 0 , 1 ]$ denotes the probability that the k-th future action step requires local interaction-oriented visual evidence. Accordingly, $1 - i _ { t } ^ { ( k ) }$ can be interpreted as the probability of workspace-level motion, such as target localization or object transport.

![](images/cf2a0525890d99f9822d554e73e563aa9c83fb27b7a0e6a0cc09429e84943c1c.jpg)  
Figure 4: Architecture of FocusVLA. Given multi-view observations, language instructions, robot states, and action history, FocusVLA predicts future action intent and uses it to guide action-side dynamic view routing for adaptive multi-view feature modulation.

The predicted intent sequence is used in two ways. First, it is aggregated into a chunk-level intent token, $z _ { \mathrm { i n t e n t } } = \mathrm { M L P } _ { \mathrm { t o k e n } } ( \mathrm { P o o l } ( I _ { t } ) )$ ), which summarizes the overall future interaction trend. Second, each step-wise intent prediction is converted into global–local routing weights through a routing function, $\alpha ^ { ( k ) } = f _ { \mathrm { r o u t e } } ( i _ { t } ^ { ( k ) } )$ , where $\alpha ^ { ( k ) } = [ \alpha _ { g } ^ { ( k ) } , \alpha _ { l } ^ { ( k ) } ]$ consists of nonnegative weights that sum to one and controls the relative contributions of global and local visual contexts for the k-th future action step. In our implementation, $f _ { \mathrm { r o u t e } }$ is a fixed bounded linear mapping that retains complementary information from both views.

During training, we construct trajectory-derived weak supervision labels $y _ { t } ^ { ( k ) } \in \{ 0 , 1 \}$ from endeffector trajectories, avoiding manual stage annotations. Based on the observation that manipulation trajectories alternate between workspace-level motion and local interaction, we initialize all time steps as movement-oriented labels 0, and then identify interaction-oriented intervals from end effector dynamics. When gripper-closing signals are available, a short window before closing is treated as a high-confidence interaction interval; otherwise, interaction intervals are detected as lowspeed local motions enclosed by higher-speed movements, using translational speed, accumulated path length, and axis-dominance cues. Short low-motion pauses that resemble transport pauses are filtered out. Steps within detected interaction intervals are labeled as 1, while the rest remain 0. These labels are used only as weak supervision, rather than hard semantic stage annotations. At inference time, FocusVLA directly predicts continuous intent probabilities from multimodal context and action history, and converts them into step-wise routing weights. Detailed label construction procedures are provided in Appendix A.

## 3.2.2 Action-Side Dynamic Routing

Action-Side Dynamic Routing applies the stage-aware control signals produced by the Future Intent Bridge, including the chunk-level intent token and step-wise routing weights, to the actiongeneration process. Specifically, the action backbone first generates action features over the future horizon conditioned on noisy action tokens [47], the chunk-level intent token, and the diffusion timestep embedding injected through adaptive RMS normalization [48, 49]. These action features are then used as queries to perform cross-attention with global and local visual features, extracting the corresponding global action context $c _ { g } ^ { ( k ) }$ and local action context $c _ { l } ^ { ( k ) }$ for each future action step k. Unlike methods that directly modify multi-view representations inside the visual encoder, Fo cusVLA performs view modulation on the action-generation side, allowing visual information to be selected and fused according to the requirements of each action step. Given the step-wise routing weight $\alpha ^ { ( k ) } = [ \alpha _ { g } ^ { ( k ) } , \alpha _ { l } ^ { ( k ) } ]$ predicted by the Future Intent Bridge, the global and local visual contexts are fused as

$$
c ^ { ( k ) } = \alpha _ { g } ^ { ( k ) } c _ { g } ^ { ( k ) } + \alpha _ { l } ^ { ( k ) } c _ { l } ^ { ( k ) } .\tag{3}
$$

Table 2: Evaluation on RoboTwin 2.0 simulation tasks. We report success rates on all 50 RoboTwin [15] tasks, grouped by task horizon. FocusVLA achieves the best overall performance and shows consistent improvements across diverse manipulation tasks.
<table><tr><td colspan="10">Short Horizon Tasks</td><td colspan="3">Click</td></tr><tr><td>Model</td><td>Click Alarmclock</td><td>Press Stapler</td><td>Turn Switch</td><td>Adjust Bottle</td><td>Grab Roller</td><td>Lt</td><td>Scan Object</td><td>Rotate Qrcode</td><td>Place Object Scale</td><td>Bell</td><td>Avg.</td></tr><tr><td>RDT [4]</td><td>61.0%</td><td>41.0%</td><td>35.0%</td><td>81.0%</td><td>74.0%</td><td>72.0%</td><td>4.0%</td><td>50.0%</td><td>1.0%</td><td>80.0%</td><td>49.90%</td></tr><tr><td>π0 [3]</td><td>63.0%</td><td>62.0%</td><td>27.0%</td><td>90.0%</td><td>96.0%</td><td>84.0%</td><td>18.0%</td><td>68.0%</td><td>10.0%</td><td>44.0%</td><td>56.20%</td></tr><tr><td>π0.5 [2]</td><td>28.0%</td><td>85.0%</td><td>46.0%</td><td>100.0%</td><td>100.0%</td><td>95.0%</td><td>35.0%</td><td>82.0%</td><td>56.0%</td><td>4.0%</td><td>63.10%</td></tr><tr><td>FocusVLA</td><td>65.0%</td><td>91.0%</td><td>53.0%</td><td>100.0%</td><td>100.0%</td><td>98.0%</td><td>60.0%</td><td>66.0%</td><td>71.0%</td><td>36.0%</td><td>74.00%</td></tr><tr><td colspan="10">Medium Horizon Tasks</td><td></td></tr><tr><td colspan="10">Move Can Move Pillbottle Move Playingcard Move Stapler</td></tr><tr><td>Model</td><td>Pot</td><td>Pad</td><td>Away</td><td>Pad</td><td>Place Mouse Pad</td><td>Place A2B Left</td><td>Place A2B Right</td><td>Place Phone Stand</td><td>Place Object Stand</td><td>Place Object Basket</td><td>Avg.</td></tr><tr><td>RDT [4]</td><td></td><td>8.0%</td><td>43.0%</td><td>2.0%</td><td>1.0%</td><td>3.0%</td><td>1.0%</td><td>15.0%</td><td>15.0%</td><td>33.0%</td><td>14.60%</td></tr><tr><td>π0 [3]</td><td>25.0% 58.0%</td><td>21.0%</td><td>53.0%</td><td>0.0%</td><td>7.0%</td><td>31.0%</td><td>27.0%</td><td>35.0%</td><td>36.0%</td><td>16.0%</td><td>28.40%</td></tr><tr><td>π0.5 [2] FocusVLA</td><td>75.0%</td><td>50.0%</td><td>83.0%</td><td>20.0%</td><td>35.0%</td><td>62.0%</td><td>10.0%</td><td>60.0%</td><td>77.0%</td><td>77.0%</td><td>54.90%</td></tr><tr><td></td><td>93.0%</td><td>79.0%</td><td>95.0%</td><td>35.0%</td><td>47.0%</td><td>67.0%</td><td>32.0%</td><td>63.0%</td><td>82.0%</td><td>85.0%</td><td>67.80%</td></tr><tr><td colspan="10">Medium-Long Horizon Tasks</td></tr><tr><td>Model</td><td>Place Cans</td><td>Place Bread</td><td>Place Bread</td><td>Place Burger Place Container</td><td></td><td>Place Can</td><td>Place Empty</td><td>Place</td><td>Place</td><td>Place Dual</td><td></td></tr><tr><td></td><td>Plasticbox</td><td>Basket</td><td>Skillet</td><td>Fries</td><td>Plate</td><td>Basket</td><td>Cup</td><td>Fan</td><td>Shoe</td><td>Shoes</td><td>Avg.</td></tr><tr><td>RDT [4] π0 [3]</td><td>19.0%</td><td>10.0%</td><td>5.0%</td><td>50.0%</td><td>78.0%</td><td>19.0%</td><td>56.0%</td><td>12.0%</td><td>35.0%</td><td>4.0%</td><td>28.80%</td></tr><tr><td>π0.5 [2]</td><td>34.0%</td><td>17.0%</td><td>23.0%</td><td>80.0%</td><td>88.0%</td><td>41.0%</td><td>37.0%</td><td>20.0%</td><td>28.0%</td><td>15.0%</td><td>38.30%</td></tr><tr><td>FocusVLA</td><td>60.0% 90.0%</td><td>74.0% 77.0%</td><td>74.0% 81.0%</td><td>90.0% 95.0%</td><td>95.0% 97.0%</td><td>76.0% 87.0%</td><td>97.0% 99.0%</td><td>44.0% 58.0%</td><td>54.0% 63.0%</td><td>56.0% 60.0%</td><td>72.00% 80.70%</td></tr><tr><td colspan="10">Long Horizon Tasks</td><td></td></tr><tr><td colspan="10"></td></tr><tr><td></td><td>Stack Blocks</td><td>Stack Blocks</td><td>Stack Bowls</td><td>Stack Bowls</td><td>Pick Dual</td><td>Pick Diverse</td><td>Put Bottles</td><td>Blocks Ranking Blocks Ranking</td><td></td><td>Beat Block</td><td></td></tr><tr><td>Model</td><td>Two</td><td>Three</td><td>Two</td><td>Three</td><td>Bottles</td><td>Bottles</td><td>Dustbin</td><td>Size</td><td>Rgb</td><td>Hammer</td><td>Avg.</td></tr><tr><td>RDT [4]</td><td>21.0%</td><td>2.0%</td><td>76.0%</td><td>51.0%</td><td>42.0%</td><td>2.0%</td><td>21.0%</td><td>0.0%</td><td>3.0%</td><td>77.0%</td><td>29.50%</td></tr><tr><td>π0 [3]</td><td>42.0%</td><td>17.0%</td><td>91.0%</td><td>64.0%</td><td>57.0%</td><td>27.0%</td><td>88.0%</td><td>7.0%</td><td>19.0%</td><td>43.0%</td><td>45.50%</td></tr><tr><td>π0.5 [2] FocusVLA</td><td>89.0% 99.0%</td><td>60.0% 69.0%</td><td>84.0% 92.0%</td><td>65.0% 74.0%</td><td>100.0% 100.0%</td><td>84.0% 87.0%</td><td>64.0% 96.0%</td><td>42.0% 53.0%</td><td>88.0% 90.0%</td><td>95.0% 96.0%</td><td>77.10% 85.60%</td></tr><tr><td colspan="10">Extra-Long Horizon / Dexterous Tasks</td><td></td></tr><tr><td></td><td>Shake</td><td>Shake Bottle</td><td></td><td>Hanging</td><td>Handover</td><td>Handover</td><td>Dump Bin</td><td>Open</td><td>Put Object</td><td>Open</td><td></td></tr><tr><td>Model</td><td>Bottle</td><td>Horizontally</td><td>Stap</td><td>Mug</td><td>Block</td><td>Mic</td><td>Bigbin</td><td>Laptop</td><td>Cabinet</td><td>Microwave</td><td>Avg.</td></tr><tr><td>RDT [4]</td><td>74.0%</td><td>84.0%</td><td>1.0%</td><td>23.0%</td><td>45.0%</td><td>90.0%</td><td>64.0%</td><td>59.0%</td><td>33.0%</td><td>37.0%</td><td>51.00%</td></tr><tr><td>π0 [3]</td><td>97.0%</td><td>99.0%</td><td>3.0%</td><td>11.0%</td><td>45.0%</td><td>98.0%</td><td>83.0%</td><td>85.0%</td><td>68.0%</td><td>80.0%</td><td>66.90%</td></tr><tr><td>π0.5 [2] FocusVLA</td><td>98.0%</td><td>99.0% 100.0%</td><td>60.0%</td><td>15.0% 25.0%</td><td>52.0%</td><td>48.0% 94.0%</td><td>86.0%</td><td>93.0% 98.0%</td><td>55.0%</td><td>64.0% 61.0%</td><td>67.00%</td></tr><tr><td>Overall Avg.</td><td colspan="2">100.0%</td><td colspan="2">60.0%</td><td colspan="2">63.0%</td><td colspan="2">92.0%</td><td colspan="2">72.0%</td><td colspan="2">76.50%</td></tr><tr><td></td><td colspan="2">RDT: 34.76%</td><td colspan="2">π0: 47.06%</td><td colspan="2">π0.5: 66.82%</td><td colspan="2"></td><td colspan="2">FocusVLA:76.92%</td><td colspan="2">+10.10%</td></tr></table>

The fused context is then injected back into the corresponding action feature through a residual connection and further processed by a lightweight action refinement module for final action prediction. Through this step-wise routing mechanism, FocusVLA dynamically adjusts the relative contributions of global and local visual information during action generation according to the predicted task progress. This design preserves the stability of the pretrained visual backbone while mitigating the limitation of static fusion, where view dependency is difficult to adapt across different task stages.

## 3.3 Training Objective

FocusVLA is trained with a joint objective for action generation and future-intent prediction. Given the future action target $U _ { t }$ and intent labels $Y _ { t } = [ y _ { t } ^ { ( 1 ) } , \dots , y _ { t } ^ { ( H ) } ]$ , the Future Intent Bridge predicts the intent sequence $I _ { t } = [ i _ { t } ^ { ( 1 ) } , \dots , i _ { t } ^ { ( H ) } ]$ , which produces the chunk-level intent token and routing weights $\alpha _ { 1 : H }$ . Conditioned on these signals, the action backbone predicts the denoised action output $V _ { t }$ . We optimize FocusVLA with a joint loss consisting of an action generation loss $\| V _ { t } - U _ { t } \| _ { 2 } ^ { 2 }$ and an intent prediction loss $\begin{array} { r } { \lambda _ { \mathrm { i n t e n t } } \overset { 1 } { H } \sum _ { k = 1 } ^ { H } \mathrm { B C E } ( i _ { t } ^ { ( k ) } , \bar { y } _ { t } ^ { ( k ) } ) } \end{array}$ , where the first term supervises action generation and the second term supervises future-intent prediction with binary labels. To stabilize training, we first warm up the intent prediction branch for $S _ { w }$ steps, and then jointly optimize both objectives with occasional teacher forcing to improve training stability.

## 4 Experiment

In this section, we evaluate FocusVLA through simulation, real-world experiments, spatial rearrangement tests, visualization analysis, and ablation studies. We aim to answer the following questions: RQ1: How does FocusVLA compare with existing VLA baselines on large-scale simulation tasks? RQ2: Does FocusVLA improve real-world manipulation performance? RQ3: Can FocusVLA adapt to unseen spatial rearrangements? RQ4: Does FocusVLA dynamically adjust view dependency across task stages? RQ5: How do its key components contribute to performance? To ensure a fair comparison, all baseline models and FocusVLA are trained with the same demonstration data, the same number of optimization steps, and evaluated under identical test seeds and evaluation protocols across all tasks and experimental settings, without additional data or tuning.

![](images/f9b54e2a0409db34ffb2c7a3e607e87a6f8b4258e0e4c093ce85bc3fa980c941.jpg)

![](images/0e67df33e29c88732479633873897163f0d52003fe7b6aacc19748793c127faa.jpg)  
Figure 5: Real-world evaluation setup and tasks. We evaluate FocusVLA on six real-world tasks using a Piper arm with global and local RealSense observations, with 30 trials per task, demonstrating its effectiveness in physical multi-view manipulation.

<table><tr><td></td><td></td><td></td><td>Task</td><td>Change</td><td>π0</td><td>π0.5</td><td>ours</td></tr><tr><td></td><td><img src="images/fbe5f4971d9e651c383aa9c13cd7b36ef614fa5a4278425f249e3225aeb552b7.jpg"/></td><td></td><td>Task1</td><td>Pos.swap</td><td>14/30(46.7%)</td><td>18/30(60.0%)</td><td>24/30(80.0%)</td></tr><tr><td></td><td></td><td></td><td>Task2</td><td>Distractor</td><td>0/30(0.0%)</td><td>3/30(10.0%)</td><td>7/30(23.3%)</td></tr><tr><td></td><td></td><td></td><td>Task3</td><td>Distractor</td><td>9/30(30.0%)</td><td>14/30(46.7%)</td><td>17/30(56.7%)</td></tr></table>

Figure 6: Execution adaptability under spatial changes.

## 4.1 Simulation Evaluation

We evaluate FocusVLA on the full RoboTwin 2.0 benchmark [15] with 50 manipulation tasks, using 100 trials per task and reporting success rates. Under the same protocol, we compare against RDT, $\pi _ { 0 }$ [3], and $\pi _ { 0 . 5 }$ [2]. As shown in Table 2, FocusVLA achieves the best overall performance, reaching an average success rate of 76.92% and outperforming the strongest baseline by 10.10 percentage points. The gains are more pronounced on medium-horizon and long-horizon tasks, suggesting that FocusVLA is especially beneficial in complex manipulation scenarios where the importance of global and local observations changes over time. We further analyze this behavior through visualization and ablation studies in the following sections.

## 4.2 Real-World Evaluation

We conduct real-world experiments on a Piper robotic arm equipped with two Intel RealSense D435i cameras for global and local observations. As shown in Fig. 5, we evaluate FocusVLA on six representative manipulation tasks with diverse object configurations and interaction requirements, con ducting 30 trials for each task. FocusVLA achieves an average success rate of 85.0%, outperforming $\pi _ { 0 . 5 }$ [2] and $\pi _ { 0 }$ [3] by 13.3 and 25.6 percentage points, respectively, and obtains the best performance across all six tasks. These results suggest that future-intent-guided dynamic view routing improves the robustness of multi-view VLA policies in real-world robot manipulation.

## 4.3 Execution Adaptability under Spatial Rearrangement

As shown in Fig. 6, we evaluate the execution adaptability of FocusVLA under unseen spatial layouts. Object positions are rearranged, requiring the policy to adapt target localization, approach paths, and fine-grained interactions to the current scene rather than relying on fixed layouts or trajectories from training. Results show that FocusVLA is more stable than static multi-view fusion baselines, suggesting that future action intent modeling and dynamic view routing help the policy better adapt to stage-dependent visual requirements under spatial changes.

![](images/25733194993a27aabddaf5301cdf0ca2aa251e7dc32d55e3d7cd9f1336133e29.jpg)

![](images/4efa68634d2d9ad8ea258c6aadf8f73cd036e188591285b81705cdb690b4a1a2.jpg)  
Figure 7: Visualization of stage-aware view routing. We visualize the global/local routing weights of FocusVLA on the Stack Blocks Three task. FocusVLA shows adaptive view reallocation across task stages.

Table 3: Ablations and Future-Intent Contribution Analysis.  
Tasks (2 seeds each; 100 trials/task/seed): put bottles dustbin, place phone stand, move can pot, beat block hammer, place shoe, open laptop, move pillbottle pad, place a2b left, and handover block.
<table><tr><td colspan="4">open_laptop, move_pillbottle_pad, place_a2b_left, and handover_block.</td></tr><tr><td>Factor</td><td>Method</td><td>Control</td><td>SR [95% CI] (%)</td></tr><tr><td>Signal</td><td>Learned Static</td><td>Globally shared learned allocation; step-wise application</td><td>67.5 [65.3, 69.6]</td></tr><tr><td>Signal</td><td>Reactive-Vision</td><td>Current-vision routing; step-wise application</td><td>57.4 [55.1, 59.7]</td></tr><tr><td>Signal</td><td>Time + State</td><td>Time/state routing; step-wise application</td><td>58.5 [56.2, 60.8]</td></tr><tr><td>Supervision</td><td>E2E Future</td><td>Action-loss-only routing</td><td>55.7 [53.4, 58.0]</td></tr><tr><td>Intent Token</td><td>Routing Only</td><td>Intent-supervised routing without the chunk-level intent token</td><td>63.3 [61.0, 65.5]</td></tr><tr><td>Regularizer</td><td>Random Masking</td><td>RDT-style train-only random view dropout</td><td>54.6 [52.3, 56.9]</td></tr><tr><td>Full</td><td>FocusVLA</td><td>Multimodal future intent; step-wise routing</td><td>73.7 [71.6, 75.7]</td></tr></table>

## 4.4 Visualization Analysis

Figure 7 visualizes the global/local view routing weights of FocusVLA on the Stack Blocks Three task. FocusVLA shows clear stage-aware view reallocation: it relies more on the global view during early target localization and motion planning, and gradually shifts toward the local view during grasping and alignment. During stacking, the routing weights become more balanced, indicating that the model jointly uses global spatial context and local manipulation cues when needed. Overall, the visualization shows that FocusVLA adapts its view usage across task stages rather than maintaining a fixed view bias, consistent with the evolving visual demands of manipulation.

## 4.5 Ablation Study

To isolate the contribution of FocusVLA, we conduct matched ablations on nine tasks using two training seeds and 100 trials per task per seed; all variants are trained for 20K steps. Table 3 examines the role of FocusVLA’s design choices along four dimensions. First, we replace multimodal future intent with a static view preference, current visual information, or time and state cues to verify the necessity of future intent as the routing signal. Second, we remove explicit future-intent supervision and train the model end-to-end using only the action loss to evaluate the effectiveness of the proposed supervision scheme and rule out the possibility that the performance gains arise merely from increased model capacity. Third, we remove the chunk-level intent token to assess its effectiveness and its complementarity with step-wise routing. Fourth, we replace intent-guided routing with random view dropout to verify the necessity of structured intent guidance for dynamic view modulation and determine whether its benefits can be explained by generic view regularization.

Variants. Full uses multimodal future-intent supervision, step-wise action-side routing, and a chunk-level intent token. Learned Static retains the global–local fusion structure and intent supervision but replaces sample- and step-dependent routing with a globally shared learnable ratio. Reactive-Vision conditions the step-wise router only on current visual features, while Time+State uses only the execution time and robot state. E2E Future retains the routing architecture and intenttoken pathway but removes explicit future-intent supervision, learning the routing variables solely through the action loss. Routing Only removes the chunk-level intent token while retaining intentsupervised step-wise routing. Finally, Random Masking replaces intent-guided routing with RDTstyle random view dropout during training.

Routing signals and temporal adaptation. As shown in Table 3, Full achieves a mean success rate of 73.7%, outperforming Learned Static (67.5%) by 6.2 pp. This gap shows that a globally shared view preference cannot accommodate sample- and step-dependent visual requirements. Full also exceeds Reactive-Vision (57.4%) and Time+State (58.5%) by 16.3 and 15.2 pp, respectively. Thus, neither current visual relevance nor time/state shortcuts can substitute for multimodal future intent when determining the visual requirements of future actions.

Supervision objective. To distinguish the effect of explicit future-intent supervision from that of additional model capacity, E2E Future retains the same routing architecture, step-wise application, and intent-token pathway as Full but is optimized only through the action loss. Its success rate drops to 55.7%, 18.0 pp below Full, demonstrating that the improvement does not arise merely from introducing an additional routing network; explicit future-intent supervision is essential for learning effective routes.

Step- and chunk-level intent conditioning. Removing the chunk-level intent token while retaining intent-supervised step-wise routing reduces performance from 73.7% to 63.3% (Routing Only), a drop of 10.4 pp. This result shows that the two intent pathways are complementary: step-wise routes determine the view allocation for each future action, while the chunk-level token provides the action generator with an overall estimate of task progress.

Comparison with view-dropout regularization. RDT-style Random Masking obtains 54.6% success rate, trailing Full by 19.1 pp. Although random view dropout can reduce reliance on a fixed view during training, it neither predicts which view will be required nor coordinates view allocation across future action steps. Its lower performance therefore indicates that generic view regularization cannot replace intent-guided routing. Overall, these ablations attribute FocusVLA’s gains to three complementary components: multimodal prediction of future visual requirements, explicitly supervised step-wise routing, and chunk-level intent conditioning. The improvements cannot be explained by a fixed global–local preference, current-context or time/state shortcuts, additional model capacity, or random view regularization.

## 5 Conclusion

We propose FocusVLA, a stage-aware dynamic view routing framework for multi-view VLA policies. By predicting future action intent and converting it into step-wise routing weights, FocusVLA dynamically coordinates global and local visual features during action generation, mitigating the view-dependency inertia caused by static multi-view fusion. Experiments on RoboTwin and realworld manipulation tasks show that FocusVLA consistently outperforms static multi-view baselines across diverse task horizons and physical scenarios. Visualization results further demonstrate that FocusVLA adapts its view usage across task stages, supporting global planning and local interaction. These results suggest that future-intent-guided routing is an effective strategy for improving the adaptability and robustness of multi-view VLA models.

## 6 Limitations

Although FocusVLA improves stage-aware multi-view utilization, there remain several promising directions for future exploration. FocusVLA demonstrates that future-intent-guided routing is an effective and lightweight way to introduce task-progress awareness into multi-view VLA models, opening a practical path toward more adaptive visual reasoning in robotic manipulation. Nevertheless, future intent is only one possible representation of manipulation progress. We hope this work encourages further exploration of richer progress-aware signals, such as explicit subgoals, objectlevel interaction states, task-phase descriptions, or language-grounded intermediate plans. Extending such progress-aware routing mechanisms to cross-embodiment settings with diverse robots, camera configurations, and real-world environments is also an important direction for future work.

## Acknowledgments

This work was supported by the Science and Technology Major Project of Jiangsu Province (Grant No. BG2024041). We gratefully acknowledge this support, which facilitated the research, system development, and experimental validation presented in this work. We sincerely thank the Area Chair and anonymous reviewers for their careful reading, thoughtful questions, and constructive suggestions, which helped us clarify the motivation, strengthen the empirical evaluation, and improve the presentation of this work.

## References

[1] M. J. Kim, C. Finn, and P. Liang. Fine-tuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645, 2025.

[2] P. Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, et al. π<sub>0.5</sub>: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

[3] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al. π : A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

[4] S. Liu, L. Wu, B. Li, H. Tan, H. Chen, Z. Wang, K. Xu, H. Su, and J. Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation. In International Conference on Learning Representations, volume 2025, pages 29982–30009, 2025.

[5] J. Bjorck, F. Castaneda, N. Cherniadev, X. Da, R. Ding, L. Fan, Y. Fang, D. Fox, F. Hu,˜ S. Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.

[6] W. Wu, F. Lu, Y. Wang, S. Yang, S. Liu, F. Wang, Q. Zhu, H. Sun, Y. Wang, S. Ma, et al. A pragmatic vla foundation model. arXiv preprint arXiv:2601.18692, 2026.

[7] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

[8] S. Yang, H. Li, B. Wang, Y. Chen, Y. Tian, T. Wang, H. Wang, F. Zhao, Y. Liao, and J. Pang. Instructvla: Vision-language-action instruction tuning from understanding to manipulation. arXiv preprint arXiv:2507.17520, 2025.

[9] Y. Fan, P. Ding, S. Bai, X. Tong, Y. Zhu, H. Lu, F. Dai, W. Zhao, Y. Liu, S. Huang, et al. Long-vla: Unleashing long-horizon capability of vision language action model for robot manipulation. arXiv preprint arXiv:2508.19958, 2025.

[10] D. Qu, H. Song, Q. Chen, Y. Yao, X. Ye, Y. Ding, Z. Wang, J. Gu, B. Zhao, D. Wang, et al. Spatialvla: Exploring spatial representations for visual-language-action model. arXiv preprint arXiv:2501.15830, 2025.

[11] Z. Zhang, H. Xu, Z. Yang, C. Yue, Z. Lin, H.-a. Gao, Z. Wang, and H. Zhao. Ta-vla: Elucidating the design space of torque-aware vision-language-action models. arXiv preprint arXiv:2509.07962, 2025.

[12] Y. Seo, J. Kim, S. James, K. Lee, J. Shin, and P. Abbeel. Multi-view masked world models for visual robotic manipulation. In International Conference on Machine Learning, pages 30613–30632. PMLR, 2023.

[13] Z. Lan, W. Mao, H. Li, L. Wang, T. Wang, H. Fan, and O. Yoshie. Bfa: Best-feature-aware fusion for multi-view fine-grained manipulation. IEEE Robotics and Automation Letters, 2025.

[14] R. Jangir, N. Hansen, S. Ghosal, M. Jain, and X. Wang. Look closer: Bridging egocentric and third-person views with transformers for robotic manipulation. IEEE Robotics and Automation Letters, 7(2):3046–3053, 2022.

[15] T. Chen, Z. Chen, B. Chen, Z. Cai, Y. Liu, Z. Li, Q. Liang, X. Lin, Y. Ge, Z. Gu, et al. Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. arXiv preprint arXiv:2506.18088, 2025.

[16] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022.

[17] B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, F. Xia, J. Wu, P. Wohlhart, S. Welker, A. Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pages 2165–2183. PMLR, 2023.

[18] O. M. Team, D. Ghosh, H. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, T. Kreiman, C. Xu, et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024.

[19] H. Wu, Y. Jing, C. Cheang, G. Chen, J. Xu, X. Li, M. Liu, H. Li, and T. Kong. Unleashing largescale video generative pre-training for visual robot manipulation. In International Conference on Learning Representations, volume 2024, pages 10641–10662, 2024.

[20] C.-L. Cheang, G. Chen, Y. Jing, T. Kong, H. Li, Y. Li, Y. Liu, H. Wu, J. Xu, Y. Yang, et al. Gr-2: A generative video-language-action model with web-scale knowledge for robot manipulation. arXiv preprint arXiv:2410.06158, 2024.

[21] J. Wen, Y. Zhu, J. Li, Z. Tang, C. Shen, and F. Feng. Dexvla: Vision-language model with plug-in diffusion expert for general robot control. arXiv preprint arXiv:2502.05855, 2025.

[22] M. Shukor, D. Aubakirova, F. Capuano, P. Kooijmans, S. Palma, A. Zouitine, M. Aractingi, C. Pascal, M. Russi, A. Marafioti, et al. Smolvla: A vision-language-action model for affordable and efficient robotics. arXiv preprint arXiv:2506.01844, 2025.

[23] G. R. Team, S. Abeyruwan, J. Ainslie, J.-B. Alayrac, M. G. Arenas, T. Armstrong, A. Balakrishna, R. Baruch, M. Bauza, M. Blokzijl, et al. Gemini robotics: Bringing ai into the physical world. arXiv preprint arXiv:2503.20020, 2025.

[24] Q. Zhao, Y. Lu, M. J. Kim, Z. Fu, Z. Zhang, Y. Wu, Z. Li, Q. Ma, S. Han, C. Finn, et al. Cotvla: Visual chain-of-thought reasoning for vision-language-action models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1702–1713, 2025.

[25] W. Zhang, H. Liu, Z. Qi, Y. Wang, X. Yu, J. Zhang, R. Dong, J. He, H. Wang, Z. Zhang, et al. Dreamvla: a vision-language-action model dreamed with comprehensive world knowledge. Advances in Neural Information Processing Systems, 38:24195–24228, 2026.

[26] G. Lu, W. Guo, C. Zhang, Y. Zhou, H. Jiang, Z. Gao, Y. Tang, and Z. Wang. Vla-rl: Towards masterful and general robotic manipulation with scalable reinforcement learning. arXiv preprint arXiv:2505.18719, 2025.

[27] A. O’Neill, A. Rehman, A. Maddukuri, A. Gupta, A. Padalkar, A. Lee, A. Pooley, A. Gupta, A. Mandlekar, A. Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6892–6903. IEEE, 2024.

[28] A. Khazatsky, K. Pertsch, S. Nair, A. Balakrishna, S. Dasari, S. Karamcheti, S. Nasiriany, M. K. Srirama, L. Y. Chen, K. Ellis, et al. Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024.

[29] P.-L. Guhur, S. Chen, R. G. Pinel, M. Tapaswi, I. Laptev, and C. Schmid. Instruction-driven history-aware policies for robotic manipulations. In Conference on Robot Learning, pages 175–187. PMLR, 2023.

[30] A. Goyal, J. Xu, Y. Guo, V. Blukis, Y.-W. Chao, and D. Fox. Rvt: Robotic view transformer for 3d object manipulation. In Conference on Robot Learning, pages 694–710. PMLR, 2023.

[31] M. Shridhar, L. Manuelli, and D. Fox. Perceiver-actor: A multi-task transformer for robotic manipulation. In Conference on Robot Learning, pages 785–799. PMLR, 2023.

[32] T. Gervet, Z. Xian, N. Gkanatsios, and K. Fragkiadaki. Act3d: 3d feature field transformers for multi-task robotic manipulation. arXiv preprint arXiv:2306.17817, 2023.

[33] C. Acar, K. Binici, A. Tekirdag, and Y. Wu. Visual-policy learning through multi-camera view˘ to single-camera view knowledge distillation for robot manipulation tasks. IEEE Robotics and Automation Letters, 9(1):691–698, 2023.

[34] Y. Lin, J. Tremblay, S. Tyree, P. A. Vela, and S. Birchfield. Multi-view fusion for multi-level robotic scene understanding. In 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 6817–6824. IEEE, 2021.

[35] I. Akinola, J. Varley, and D. Kalashnikov. Learning precise 3d manipulation from multiple uncalibrated cameras. In 2020 IEEE International Conference on Robotics and Automation (ICRA), pages 4616–4622. IEEE, 2020.

[36] H. Chen, R. Yang, J. Zhang, X. Wen, Y. Chen, D. Yu, C. Bai, and Z. Wang. Temporal consistent multi-view perception for robust embodied manipulation. Pattern Recognition, page 112177, 2025.

[37] J. Mao, J. Guan, Y. Tang, Q. Hu, Z. Li, J. Yu, Y. Mao, Y. Sun, S. Liu, and X. Ju. Omnid: Generalizable robot manipulation policy via image-based bev representation. arXiv preprint arXiv:2508.11898, 2025.

[38] T.-W. Ke, N. Gkanatsios, and K. Fragkiadaki. 3d diffuser actor: Policy diffusion with 3d scene representations. arXiv preprint arXiv:2402.10885, 2024.

[39] J. Bian, Z. Zhang, Z. Liang, S. Zheng, S. Zhang, R. Shen, C. Yang, and A. Hou. Vla-lpaf: Lightweight perspective-adaptive fusion for vision-language-action to enable more unconstrained robotic manipulation. arXiv preprint arXiv:2509.18183, 2025.

[40] Q. Fan, Z. Li, Y. Lu, W. Chen, Q. Shen, X.-x. Long, Y. Cai, T. Lu, S. Wang, and X. Cao. Peafowl: Perception-enhanced multi-view vision-language-action for bimanual manipulation. arXiv preprint arXiv:2601.17885, 2026.

[41] D. Li, S. He, J. Tian, Y. Zhang, W. Xie, M. Cao, D. Liu, Z. Li, T. Hui, R. Huang, et al. Uni-sight: An e2e vision-language-action system unifying multi-view alignment and multi-modal fusion. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 7142–7151, 2025.

[42] M. S. Sarowar, O. Tariq, and S. Kim. Gst-vla: Structured gaussian spatial tokens for 3d depthaware vision-language-action models. arXiv preprint arXiv:2603.09079, 2026.

[43] S. Jain and B. C. Wallace. Attention is not explanation. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 3543–3556, 2019.

[44] S. Serrano and N. A. Smith. Is attention interpretable? In Proceedings of the 57th annual meeting of the association for computational linguistics, pages 2931–2951, 2019.

[45] H. Chefer, S. Gur, and L. Wolf. Transformer interpretability beyond attention visualization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 782–791, 2021.

[46] S. Abnar and W. Zuidema. Quantifying attention flow in transformers. In Proceedings of the 58th annual meeting of the association for computational linguistics, pages 4190–4197, 2020.

[47] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, 44(10-11):1684–1704, 2025.

[48] B. Zhang and R. Sennrich. Root mean square layer normalization. Advances in neural information processing systems, 32, 2019.

[49] W. Peebles and S. Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.

## Appendix

## A. Automatic Intent Label Construction

Algorithm 1 Automatic Intent Label Construction   
1: for each episode do   
2: compute smoothed translational speed and rotation for both arms   
3: combine the two arms by taking the maximum speed/rotation at each frame   
4: detect gripper-closing windows if gripper signals are available   
5: if any valid closing window exists then   
6: expand each closing run by a 16-frame lookback   
7: merge same-arm windows with $\mathrm { g a p } \le 6$   
8: use the resulting windows as interaction seeds   
9: else   
10: compute high-speed threshold as the 70th percentile of speed   
11: compute low-speed threshold as the 45th percentile of speed   
12: extract high-speed segments of length $\geq 3$   
13: extract low-speed segments of length $\geq 6$   
14: keep only low-speed segments that are surrounded by high-speed motion   
15: reject low-speed segments whose path length exceeds the adaptive cap unless they are   
strongly axis-dominant   
16: add press-like candidates that occur late, follow an approach, and have small path length   
17: add handover-like windows based on end-effector proximity and cooperative motion   
18: end if   
19: merge nearby interaction windows with gap tolerance 12   
20: split each merged interaction by high-speed subsegments   
21: demote non-dominant high-speed chunks to global movement   
22: remove short interaction valleys that are better explained as transport pauses   
23: trim interaction windows around valid gripper-closing events when available   
24: assign label 1 to all frames inside final interaction windows   
25: assign label 0 to all remaining frames   
26: save the binary label sequence   
27: end for

We automatically construct binary intent labels from demonstration trajectories without manual phase annotation. The label generation pipeline first applies an end-pose segmentation backend to each episode and then converts the resulting temporal segments into frame-wise supervision. The segmentation operates on the end-effector motion of both arms and uses a set of deterministic heuristics to distinguish global movement from interaction.

Let $x _ { t } ^ { a } \in \mathbb { R } ^ { 3 }$ and $q _ { t } ^ { a } \in \mathbb { R } ^ { 4 }$ denote the translation and quaternion orientation of arm $a \in \{ \mathrm { l e f t , r i g h t } \}$ at time step t. We compute the per-arm translational speed and rotation increment as

$$
v _ { t } ^ { a } = \left| \left| x _ { t } ^ { a } - x _ { t - 1 } ^ { a } \right| \right| _ { 2 } , \qquad r _ { t } ^ { a } = 2 \operatorname { a r c c o s } \left( \left| \langle q _ { t - 1 } ^ { a } , q _ { t } ^ { a } \rangle \right| \right) ,\tag{4}
$$

and then smooth both signals using a 5-tap moving average. The fused motion signal is obtained by taking the maximum over the two arms:

$$
v _ { t } = \operatorname* { m a x } \Bigl ( v _ { t } ^ { \mathrm { l e f t } } , v _ { t } ^ { \mathrm { r i g h t } } \Bigr ) , \qquad r _ { t } = \operatorname* { m a x } \Bigl ( r _ { t } ^ { \mathrm { l e f t } } , r _ { t } ^ { \mathrm { r i g h t } } \Bigr ) .\tag{5}
$$

Gripper-driven interaction seeds. When gripper signals are available, interaction detection is dominated by closing events. We detect a closing event when the per-step gripper difference satisfies

$$
\Delta g _ { t } < - 0 . 0 3\tag{6}
$$

for at least 4 consecutive frames. Closures occurring before the first 12 frames are ignored as initialization noise. Each valid closing run is expanded by a 16-frame lookback window, and nearby

windows of the same arm are merged if their temporal gap is at most 6 frames. If any gripper-closing window exists, these windows are directly treated as interaction seeds and the remaining timeline is assigned to global movement unless further trimming is needed.

Motion-driven interaction seeds. If no valid gripper-close evidence is available, we fall back to motion-based segmentation. We define high-speed frames using the 70th percentile of the smoothed speed signal and low-speed frames using the 45th percentile:

$$
\mathrm { h i g h – s p e e d : } v _ { t } \geq p _ { 7 0 } ( v ) , \qquad \mathrm { l o w \mathrm { - } s p e e d : } v _ { t } \leq p _ { 4 5 } ( v ) .\tag{7}
$$

High-speed segments must last at least 3 frames, while low-speed segments must last at least 6 frames. A low-speed segment $[ s , e ]$ is considered an interaction candidate only if it is temporally surrounded by high-speed motion, i.e., there exists high-speed activity within a 20-frame lookback before s and within a 20-frame lookahead after e.

For each low-speed candidate, we compute its fused path length

$$
P ( s , e ) = \sum _ { t = s } ^ { e } \operatorname* { m a x } \Bigl ( \| x _ { t } ^ { \mathrm { l e f t } } - x _ { t - 1 } ^ { \mathrm { l e f t } } \| _ { 2 } , \| x _ { t } ^ { \mathrm { r i g h t } } - x _ { t - 1 } ^ { \mathrm { r i g h t } } \| _ { 2 } \Bigr ) ,\tag{8}
$$

and compare it against an adaptive path cap

$$
P _ { \mathrm { m a x } } = \mathrm { m a x } \big ( 0 . 0 2 , 2 \cdot p _ { 8 0 } \big ( \{ P ( s , e ) \} _ { \mathrm { l o w - s p e e d s e g m e n t s } } \big ) \big ) .\tag{9}
$$

A low-speed segment is rejected if $P ( s , e ) > P _ { \operatorname* { m a x } }$ and its dominant-axis ratio is below 0.78. The dominant-axis ratio is defined as the fraction of motion energy along the most active axis within the segment. Intuitively, this rule preserves compact, axis-consistent local adjustments while filtering out long translational motions.

Press-like and handover-like cases. Some tasks do not exhibit a clear gripper-closing event but still contain local interactions such as pressing or handover. If no interaction candidate is found and no closing window exists, we additionally keep a late low-speed segment as a press-like interaction if it satisfies all of the following conditions: (i) its start index is at least 12, (ii) it is preceded by a high-speed approach within a 30-frame lookback, and (iii) its fused path length is smaller than $\operatorname* { m a x } ( 0 . 0 1 2 , 0 . 8 P _ { \operatorname* { m a x } } )$ . We further allow a short low-speed hold tail after such a press-like interaction when the following low-speed segment starts within 6 frames and its length is no more than 14 frames.

We also add handover-style windows based on near-field dual-arm coordination. These windows are detected when the distance between the two end-effectors becomes sufficiently small and the two arms move cooperatively. A stricter handover detector uses an arm-distance threshold clipped to [0.16, 0.25], requires a minimum overlap of motion share, and expands each candidate window by 6 frames. A looser fallback detector uses a distance threshold clipped to [0.22, 0.30] and retains only windows whose motion share, activity mean, and activity peak exceed adaptive thresholds. Nearby handover windows are merged if their gap is within 8 frames.

Interaction cleanup and priority rules. After collecting all interaction seeds, we merge nearby or overlapping interaction windows with a 12-frame gap tolerance. To prevent large translations from being mislabeled as interaction, each merged interaction window is further split by high-speed subsegments using a movement threshold at the 72nd percentile of $v _ { t }$ . High-speed subsegments shorter than 7 frames are ignored, and interaction fragments shorter than 6 frames are discarded. Within such high-speed subsegments, we keep them as interaction only if their dominant-axis ratio is at least 0.78; otherwise, they are demoted to global movement.

We then apply three additional reclassification rules to remove spurious short pauses. First, a short interaction valley between two strong global segments is converted to global if its duration is below 16 frames and its fused path length is sufficiently small. Second, if an interaction valley is short (at most 18 frames), low-path, and its neighboring global segments move along the same axis and in the same direction, it is also reclassified as global. Third, a short low-motion valley between two strong global transport segments is demoted to global if its length is at most 12 frames and its path length is below 0.03. Finally, when gripper-closing evidence exists, interaction windows are trimmed around the closing intervals so that only the frames consistent with the closing event are preserved.

Given the final interaction intervals $\mathcal { T } = \{ [ s _ { k } , e _ { k } ] \} _ { k = 1 } ^ { K }$ , we define the binary intent label sequence $\mathbf { y } \in \{ 0 , 1 \} ^ { T }$ as

$$
y _ { t } = { \left\{ \begin{array} { l l } { 1 , } & { { \mathrm { i f ~ } } t \in \bigcup _ { k = 1 } ^ { K } [ s _ { k } , e _ { k } ] , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } , } \end{array} \right. } \qquad t = 1 , \ldots , T .\tag{10}
$$

Here, $y _ { t } = 1$ denotes local interaction and $y _ { t } = 0$ denotes global movement. The same deterministic procedure is applied to every episode, making the generated labels fully reproducible.

## B. Training and Evaluation Configurations

We train FocusVLA separately on the RoboTwin simulated benchmark and the real-world bench mark. Within each benchmark, we collect 50 demonstration trajectories per task and jointly train a single policy across all tasks in that benchmark. In other words, simulation tasks are trained in one multi-task setting, while real-world tasks are trained in another, with no data mixing across the two domains. Training is conducted on 8 NVIDIA A100 80GB GPUs for 100,000 optimization steps.

To stabilize optimization, we adopt an intent warmup strategy. Specifically, during the initial warmup phase, we optimize the future intent branch first so that the model can learn a reliable interaction-pattern prior before being used to guide action generation. After warmup, we jointly optimize the action-generation objective and the intent-prediction objective.

Since interaction frames are typically less frequent than global movement frames, we further apply a class-balanced intent loss to mitigate label imbalance. Let $p _ { t }$ denote the predicted intent probability at time step t and $y _ { t } \in \{ 0 , 1 \}$ denote the corresponding binary label, where $y _ { t } ~ = ~ 1$ indicates interaction and $y _ { t } ~ = ~ 0$ indicates global movement. The intent loss is formulated as a weighted binary cross-entropy:

$$
\mathcal { L } _ { \mathrm { i n t e n t } } = - \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \left( \lambda _ { + } y _ { t } \log p _ { t } + \lambda _ { - } ( 1 - y _ { t } ) \log ( 1 - p _ { t } ) \right) ,\tag{11}
$$

where $\lambda _ { + }$ and $\lambda _ { - }$ are class weights for interaction and movement, respectively. In practice, these weights are set to balance the contribution of the two classes and prevent the model from being dominated by the majority movement labels.

The overall training objective is

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { a c t } } + \lambda _ { \mathrm { i n t e n t } } \mathcal { L } _ { \mathrm { i n t e n t } } ,\tag{12}
$$

where $\mathcal { L } _ { \mathrm { a c t } }$ denotes the action-generation loss and $\lambda _ { \mathrm { { i n t e n t } } }$ controls the relative importance of the intent supervision.

During evaluation, we report the success rate on the corresponding task-specific test episodes using the checkpoint obtained from the respective domain-specific training. Unless otherwise stated, the evaluation protocol follows the standard benchmark setting described in the main paper.

## C. Future Intent Bridge Details

The Future Intent Bridge is designed to infer how the interaction mode evolves over the future action horizon from the current multimodal context. Given the contextualized language features from the VLA prefix backbone, together with visual features and state-history features, we use language tokens as queries and visual/state-history tokens as keys and values in a cross-modal attention layer. This design allows the bridge to selectively retrieve scene evidence and recent motion cues that are most relevant to the instructed task.

The resulting multimodal representation is then fed into a temporal intent head, which predicts a step-wise intent sequence over the future horizon,

$$
I _ { t } = [ i _ { t } ^ { ( 1 ) } , i _ { t } ^ { ( 2 ) } , \dots , i _ { t } ^ { ( H ) } ] ,\tag{13}
$$

where $i _ { t } ^ { ( k ) } \in [ 0 , 1 ]$ denotes the predicted probability that the k-th future action step requires local interaction-oriented visual evidence. Unlike a discrete stage label, this sequence provides a continuous estimate of how interaction demand evolves across the future horizon.

To convert the intent sequence into downstream control signals, we further derive two outputs from it. First, we compress the future intent sequence into a chunk-level intent token, which serves as a coarse semantic condition for the action suffix. Second, we map the predicted intent probabilities to step-wise routing weights,

$$
\alpha ^ { ( k ) } = [ \alpha _ { g } ^ { ( k ) } , \alpha _ { l } ^ { ( k ) } ] ,\tag{14}
$$

where ${ \alpha } _ { g } ^ { ( k ) }$ and $\alpha _ { l } ^ { ( k ) }$ denote the relative emphasis on global and local visual evidence for the kth future step, respectively. These routing weights are obtained by linearly mapping the predicted intent probabilities into the allowed global/local range and are subsequently used by the action-side routing module.

## D. Action-Side Dynamic Routing Details

Action-Side Dynamic Routing applies the stage-aware control signals produced by the Future Intent Bridge to the action generation process. After the action backbone processes the noisy action tokens under the conditioning of the chunk-level intent token and the diffusion timestep embedding, it produces action features for the future horizon. These action features are then used as queries to attend to global and local visual features separately, yielding two context streams for each future step:

$$
c _ { g } ^ { ( k ) } , \qquad c _ { l } ^ { ( k ) } , \qquad k = 1 , \dots , H .\tag{15}
$$

Different from static multi-view fusion, we perform view modulation on the action side, so that the model can decide at each future step which visual source should contribute more strongly. Given the routing weights $\alpha ^ { ( k ) } = [ \alpha _ { g } ^ { ( k ) } , \alpha _ { l } ^ { ( \bar { k } ) } ]$ , the two visual contexts are fused as

$$
c ^ { ( k ) } = \alpha _ { g } ^ { ( k ) } c _ { g } ^ { ( k ) } + \alpha _ { l } ^ { ( k ) } c _ { l } ^ { ( k ) } .\tag{16}
$$

The fused context is then injected back into the corresponding action feature through a gated resid ual connection, producing an enhanced action representation for final prediction. This design allows FocusVLA to dynamically shift its dependence between global and local views across different manipulation stages, while preserving the pretrained visual backbone and avoiding rigid, hand-designed view selection rules.

## E. BFA-Related Controls

We further conduct matched controls to distinguish FocusVLA from BFA along four dimensions: prediction horizon, fusion site, label construction, and intent conditioning. Within this comparison, all variants use the same four-task subset and evaluation protocol.

Prediction horizon. BFA estimates one view allocation from current vision and applies the resulting fused representation to the entire action chunk. It therefore cannot condition routing on instruction-, state-, or history-dependent requirements or explicitly adapt the allocation across future steps within the chunk. Reactive-Vision retains step-wise routing but uses only current vision, whereas Reactive $( H _ { r } { = } 1 )$ retains multimodal intent conditioning but predicts one route shared by the full H-step chunk. Full outperforms these variants by 18.2 and 24.0 pp, respectively, supporting the contributions of multimodal future-intent conditioning and anticipatory step-specific routing.

Table 4: BFA-related controls.
<table><tr><td colspan="5">Tasks (100 trials/task): open_microwave, pick_diverse_bottles, stack_bowls_two, and pick_dual_bottles.</td></tr><tr><td>Variant</td><td>Routing signal</td><td>Application</td><td>Token</td><td>SR [95% CI] (%)</td></tr><tr><td>Reactive-Vision</td><td>Current vision</td><td>Step-wise</td><td>Yes</td><td>61.3 [56.4, 65.9]</td></tr><tr><td>Reactive (Hr=1)</td><td>MM intent</td><td>Shared</td><td>Yes</td><td>55.5 [50.6, 60.3]</td></tr><tr><td>Feature-side</td><td>MM intent</td><td>Pre-policy</td><td>Yes</td><td>67.0 [62.2, 71.4]</td></tr><tr><td>BFA-style Labels</td><td>BFA-style</td><td>Step-wise</td><td>Yes</td><td>72.0 [67.4, 76.2]</td></tr><tr><td>Reactive w/o Token</td><td>MM intent</td><td>Shared</td><td>No</td><td>49.8 [44.9, 54.6]</td></tr><tr><td>Routing Only</td><td>MM intent</td><td>Step-wise</td><td>No</td><td>71.5 [66.9, 75.7]</td></tr><tr><td>Full</td><td>MM intent</td><td>Step-wise</td><td>Yes</td><td>79.5 [75.3, 83.2]</td></tr></table>

Fusion site. BFA merges multi-view features before the policy, whereas FocusVLA maintains separate global and local action contexts and fuses them for each predicted action step. With routing, intent supervision, and the chunk-level token fixed, Feature-side moves only the fusion operation to the pre-policy location used by BFA. Its 12.5-pp gap from Full isolates the benefit of performing view fusion within action generation.

Label construction. BFA uses offline VLM annotations of robot states together with task-specific rule mappings. Because BFA provides neither usable code nor complete labeling rules, we implement a paper-based BFA-style baseline using GPT-4o arm-state annotations and task-specific mappings. With step-wise routing and the intent token fixed, Full exceeds BFA-style Labels by 7.5 pp. The human-agreement evaluation in Table 5 further supports the accuracy of our task-agnostic trajectory labels.

Intent token. Unlike BFA, FocusVLA provides the action generator with a chunk-level intent token in addition to step-wise routing weights. Adding this token improves Reactive (H<sub>r</sub>=1) over Reactive w/o Token by 5.7 pp and Full over Routing Only by 8.0 pp. These matched comparisons show that chunk-level intent conditioning benefits both chunk-shared and step-wise routing.

Together, these controls show that anticipatory routing, action-side fusion, trajectory-derived super vision, and chunk-level intent conditioning each provide distinct gains.

## F. Intent-Label Evaluation

We evaluate the trajectory-derived intent labels from three perspectives: agreement with human annotations, held-out intent-prediction accuracy, and robustness to label definitions and prediction errors. The robustness experiments use the same four tasks as the BFA-related controls, with 50 trials per task.

Human agreement. Using human annotations as reference, our task-agnostic trajectory labels achieve 82.9% F1, exceeding the paper-based BFA-style labels by 17.8 points. This result indicates that our labels better capture stage-specific visual requirements without relying on offline VLM annotations or task-specific rule mappings.

Intent-prediction accuracy. On held-out trajectories, the Future Intent Bridge achieves 78.4% F1 at a 0.5 classification threshold and 91.2% AUROC. Qualitative inspection shows that most prediction errors slightly shift transition boundaries rather than confuse the underlying interaction stages, suggesting that the predicted scores preserve the temporal structure required for step-wise routing.

Robustness to label definitions. We construct loose and strict variants by relaxing or tightening the local-interaction criteria used for trajectory labeling. Relative to the default definition, these variants alter 45.2% and 27.6% of the labels, respectively. Nevertheless, their success rates are 75.5% and 74.0%, compared with 78.0% under the default labels, corresponding to a maximum change of 4.0 pp. Thus, downstream control performance is relatively insensitive to the precise heuristic thresholds used to construct the intent labels.

Table 5: Intent-label evaluation.
<table><tr><td>Metric</td><td>BFA-style</td><td>Ours</td><td>Effect</td></tr><tr><td>Human F1 ↑ (%)</td><td>65.1</td><td>82.9</td><td>+17.8</td></tr><tr><td>Intent F1/AUROC (%)</td><td>N/A</td><td>78.4/91.2</td><td>N/A</td></tr><tr><td>Thresh. SR (D/L/S) (%)</td><td>N/A</td><td>78.0/75.5/74.0</td><td>≤ 4.0 pp</td></tr><tr><td>Infer. SR (0/10/20%) (%)</td><td>N/A</td><td>78.0/73.5/75.5</td><td> $\leq 4 . 5 \ : \mathrm { p p }$ </td></tr></table>

Robustness to prediction errors. To evaluate whether FocusVLA depends on perfectly accurate intent estimates, we independently corrupt 10% and 20% of the predicted intents at inference. The corresponding success rates are 73.5% and 75.5%, differing from the uncorrupted result by at most 4.5 pp. FocusVLA therefore remains effective under moderate intent-prediction errors.

Overall, our trajectory-derived supervision aligns more closely with human annotations than the BFA-style alternative, supports accurate held-out intent prediction, and remains robust to both labeldefinition choices and moderate prediction errors. Generalization to unseen camera layouts remains future work.

## G. Architecture and Implementation Details

Architecture. FocusVLA augments $\pi _ { 0 . 5 }$ with a Future Intent Bridge and an action-side router. In the Bridge, the Temporal State-History Encoder linearly embeds the current robot state into a 2,048-D state token. It separately processes the previous 30 actions and their temporal differences using two local temporal blocks, followed by softmax pooling to produce a 2,048-D history token. Together with the multimodal vision-language context, these tokens are processed by four-head, 2,048-D cross-attention. Masked-mean pooling aggregates the valid features, and a 2048 → 4096 → 30 MLP predicts intent scores for $H = 3 0$ future action steps. The action-side router uses four-head, 1,024- D cross-attention to construct separate global and local action contexts, which are fused for each predicted action step using the corresponding routing weight. The Bridge additionally provides a chunk-level intent token to condition the overall action generation process.

Training and computational overhead. The intent objective uses a loss weight of $\lambda _ { \mathrm { i n t e n t } } = 1$ and a positive-class weight of $w _ { + } = 1 5$ . Training begins with a 5K-step intent-only warm-up. From steps 5K to 15K, the teacher-forcing probability is annealed from 0.77 to 0.3 and then remains at 0.3 for the rest of training; teacher forcing is disabled at inference. FocusVLA adds 70.13M parameters to $\pi _ { 0 . 5 }$ . At batch size $B = 1$ on one NVIDIA A100, its per-chunk latency is 92.78 ms, compared with 79.59 ms for $\pi _ { 0 . 5 } ,$ , corresponding to an increase of 13.19 ms (16.6%).

Bounded routing weights. For each sample and future step, the Bridge predicts an intent logit from multimodal context. A sigmoid converts this logit into a probability, which is linearly mapped to a view weight strictly within (0.25, 0.75); the complementary view receives the remaining weight, so the two weights sum to one. This task-invariant mapping can favor one view by up to a 3:1 ratio while retaining at least 25% of the complementary view, thereby preserving complementary evidence and avoiding brittle 0/1 selection. The bounds and scalar mapping are shared across tasks; context dependence arises from the sample- and step-specific intent predictions rather than from task-specific endpoints. At the reporting precision, 64.19% of held-out $\alpha _ { g }$ values remain strictly interior, while the remaining values round to 0.25 or 0.75. These endpoint occurrences therefore represent confident, near-binary intent predictions rather than a fixed routing policy.