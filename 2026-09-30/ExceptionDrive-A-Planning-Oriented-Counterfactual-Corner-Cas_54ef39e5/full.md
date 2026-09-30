# ExceptionDrive: A Planning-Oriented Counterfactual Corner-Case Benchmark for Autonomous Driving

Ziyi Luo<sup>1</sup>, Zhe Sun<sup>1</sup>, Yehao Lu<sup>1</sup>, Lei Zhou<sup>2</sup>, Lisheng Wu<sup>2</sup>, Xuewei Li<sup>1</sup>, Zequn Qin<sup>1</sup>, and Xi <sub>Li</sub><sup>∗</sup>1

<sup>1</sup>College of Computer Science and Technology, Zhejiang University, Hangzhou, China <sup>2</sup>Yinwang Intelligent Technology Co., Ltd., Shenzhen, China

## Abstract

Average performance on routine driving benchmarks does not establish planner reliability under rare, safety-critical hazards. We proposed ExceptionDrive, a counterfactual planning benchmark that uses VLM-assisted screening, localized multi-view editing, and quality auditing to insert hazards into real nuScenes scenes while preserving their context. Its 21 tasks span six safety families and define hazard or conflict regions, local safety constraints, and acceptable responses. Because hazard insertion can invalidate the recorded human trajectory, our reference-free protocol evaluates edited predictions using Unsafe Rate (UR), Hazard Clearance Compliance (HCC), Hazard Proximity Response (HPR), and Counterfactual Trajectory Shift (CTS), which measure core-region intrusion, clearance compliance, clearance relative to a prescribed margin, and counterfactual trajectory change. Seven representative planners frequently intrude into hazard regions or provide insuficient clearance. We also develop a Reminder Agent that, without sample-specific task labels, converts visual evidence and the shared taxonomy into structured records of hazard presence, type, and a recommended high-level strategy. The agent neither predicts trajectories nor controls the vehicle; its records guide a VLM-based decision agent. In zero-shot experiments, the reminders improve strategy accuracy and reduce under-warning.

![](images/a5f65ff7c377ebe584dc507e20e5abaaf1ca6b3f6bb91b5df15ae30f142ce6ad.jpg)  
Figure 1: Overview of ExceptionDrive. Our framework generates localized counterfactual hazards in real driving scenes, evaluates planner responses without requiring an expert reference trajectory for the edited scene, and produces structured safety-aware reminders. The benchmark includes 21 planning tasks across six planning-safety families.

## 1 Introduction

Autonomous-driving planners have achieved strong performance on standard benchmarks, yet these benchmarks remain dominated by routine, low-risk scenes. Average-case performance therefore does not establish reliability under rare, safety-critical hazards, such as an occluded pedestrian entering the ego path or a collapsed road surface blocking the lane.

Existing resources provide limited support for controlled planner evaluation in such cases. Real-world datasets preserve realistic observations but contain few controllable hazards, whereas simulation introduces a domain gap. Corner-case datasets often target perception, anomaly recognition, or semantic reasoning rather than trajectory compliance. A planning benchmark should instead combine realistic scenes, controlled hazard insertion, paired source–edited observations, spatial hazard annotations, and trajectory-level evaluation.

Hazard insertion also invalidates conventional trajectory supervision: the recorded human trajectory may intersect the new hazard and no longer provide a valid reference. Moreover, slowing, stopping, yielding, and bypassing may all be acceptable under diferent constraints. Each task should therefore define local safety constraints and a set of acceptable responses rather than prescribe a unique expert trajectory.

We proposed ExceptionDrive, a planning-oriented counterfactual benchmark built by inserting localized hazards into real nuScenes observations. Its construction pipeline combines VLM-assisted scene screening, localized multi-view editing, quality review, and BEV hazard localization while preserving the surrounding context. The benchmark contains 21 tasks in six safety families. Overall, ExceptionDrive comprises 12,440 source–edited pairs: 3,181 quality-approved benchmark pairs and 9,259 auxiliary pairs. Source–edited pairs use identical model settings and non-visual inputs, allowing the efect of the localized edit to be isolated.

Our hazard-centric protocol evaluates edited trajectories without expert references. Unsafe Rate (UR) measures core-region intrusion, and Hazard Clearance Compliance (HCC) measures compliance with the required clearance. Hazard Proximity Response (HPR) diagnoses clearance relative to the prescribed margin, while Counterfactual Trajectory Shift (CTS) measures the change between source and edited predictions. HCC and UR are the primary safety metrics; HPR and CTS are complementary diagnostics. The protocol evaluates local hazard compliance rather than complete route validity or closed-loop safety.

Experiments on representative planners reveal frequent intrusion and insuficient clearance. We also develop a lightweight Reminder Agent that maps visual evidence and the shared task taxonomy to a structured record containing hazard status, type, relative location, risk level, supporting evidence, the applicable safety constraint, and a recommended high-level strategy. Without receiving sample-specific task labels or target strategies, the reminder supports high-level decision making but does not predict trajectories or control the vehicle.

Our main contributions are threefold:

1. We proposed ExceptionDrive, a planning-oriented counterfactual corner-case benchmark created through localized multi-view editing of real driving scenes. It enables controlled insertion of safetycritical hazards while preserving the surrounding driving context as much as possible.

2. We established a planning-oriented task taxonomy and hazard-centric evaluation framework. The framework organizes 21 tasks into six planning-safety families, defines each task through local safety constraints and acceptable risk-mitigation responses, and evaluates edited-scene trajectories without requiring expert reference trajectories. HCC, UR, HPR, and CTS respectively characterize clearance compliance, core-region intrusion, hazard-proximity response, and counterfactual trajectory change.

3. We systematically evaluated representative planners using the proposed benchmark and reveal frequent failures in local hazard compliance. As an auxiliary application, we also introduced a lightweight Reminder Agent and found that structured task-grounded risk records improve high-level strategy accuracy and reduce under-warning in zero-shot VLM experiments.

## 2 Related Work

## 2.1 VLM-Assisted Data Generation for Autonomous Driving

Autonomous-driving data come from real-world logs, simulation, and generative models. Real-world datasets preserve authentic distributions but are costly and dominated by routine driving Caesar et al. [2020], Wilson et al. [2023], Sun et al. [2020], Geiger et al. [2012]; simulation ofers controllability but introduces domain gaps Dosovitskiy et al. [2017], Jia et al. [2024], Gong et al. [2024]; and generative models synthesize scenes conditioned on text, layout, or geometry Gao et al. [2024], Wen et al. [2024], Li et al. [2024], Wang et al. [2024].

VLMs support scene captioning, semantic annotation, prompt generation, and quality assessment Sima et al. [2024], Chen et al. [2025], Wang et al. [2025], Hao et al. [2026]. Rather than synthesizing entire scenes, ExceptionDrive uses VLMs to select task-compatible real scenes, generate location-aware editing instructions, and review edited samples. This localized approach retains the source scene as a within-scene control while seeking to isolate the intended hazard.

## 2.2 Corner-Case Taxonomies and Dataset Construction

Corner cases span multiple levels, from sensor or appearance shifts and unusual objects to rare scene configurations and safety-critical interactions Breitenstein et al. [2020], Heidecker et al. [2021], Rösch et al. [2022]. Existing benchmarks can be broadly grouped by their evaluation targets. Perception-oriented datasets assess road-anomaly segmentation or unknown-object detection Pinggera et al. [2016], Blum et al. [2021],

Chan et al. [2021], Li et al. [2022a], Gong et al. [2024]; event- and reasoning-oriented datasets collect accidents or annotate rare scenes with semantic descriptions and driving suggestions Fang et al. [2019], Chen et al. [2025]; and system-level benchmarks evaluate driving policies in interactive simulation Dosovitskiy et al. [2017], Jaeger et al. [2023], Xu et al. [2022], Jia et al. [2024].

Real-world datasets preserve visual realism but provide limited control over rare hazards, whereas simulation ofers controllability at the cost of a domain gap. ExceptionDrive complements these settings with planner-level counterfactual testing on paired real scenes.

## 3 Method

![](images/b2c954c701f2aec5b541f86a672c7ff550189673c8192a4add016c4e78c2594a.jpg)  
Figure 2: Counterfactual corner-case construction pipeline. An edit assistant uses the task definition and real-scene context to produce task-aware positive and negative prompts. A localized driving-scene editor inserts the target hazard while preserving viewpoint, road geometry, lighting, trafic context, and unrelated background. A review assistant then audits task correctness, physical plausibility, background preservation, realism, and action determinacy. Accepted source–edit pairs provide hazard-localized, planning-oriented samples for evaluation.

## 3.1 Task Taxonomy

An object- or appearance-based corner-case taxonomy is insuficient for planner evaluation because risk depends on the hazard’s location, its conflict with the ego vehicle, and the feasible responses. For example, the same vehicle may require stopping when it blocks the ego lane, yielding when it occupies an intersection conflict region, or performing a constrained bypass when suficient adjacent space is available. Conversely, visually diferent hazards may impose the same planning constraint, such as avoiding a non-drivable region. Constructing planning-oriented counterfactual cases therefore requires an operational task specification that jointly defines the applicable driving context, the localized hazard intervention, the spatial safety constraint, and the range of acceptable risk-mitigation responses.

We define 21 tasks and organize them into six planning-safety families: Sudden Emergence, Lane

Blockage, Bypass and Gap Conflict, Signal and Rule Conflict, Intersection Conflict, and Road-Surface Hazard. For each task, we specify scene-suitability conditions, rejection rules, edit content, target-location constraints, the corresponding hazard or conflict region, and acceptable risk-mitigation responses. These responses do not prescribe a single mandatory maneuver. Instead, they describe task-level options such as avoiding a localized risk region, stopping before a constraint boundary, yielding to conflicting trafic, or bypassing only when suficient space is available. The resulting specifications provide a common basis for scene screening, localized editing, quality review, hazard-region construction, geometric planner evaluation, and high-level strategy evaluation.

## 3.2 Dataset Construction

ExceptionDrive uses task-conditioned local counterfactual editing to introduce planning-relevant hazards while preserving the surrounding real-world context. The task specifications in Section 3.1 guide four stages: scene screening, localized multi-view editing, quality review, and hazard-region localization.

For each scene–task combination, a VLM assesses whether the synchronized nuScenes views and scene description satisfy the task-specific suitability conditions. It examines the road layout, available space, occlusion pattern, and potential insertion location. For suitable combinations, the VLM selects the relevant views, specifies an ego-relative target location, and generates view-specific editing instructions. A combination is rejected if the hazard would violate task constraints, create an ambiguous planning conflict, or require substantial changes to the original scene.

An instruction-driven editor then combines global preservation requirements with the task-level hazard description and scene-specific location instructions. It modifies only the target region while preserving the viewpoint, road geometry, lane markings, illumination, weather, unrelated trafic participants, and background. For hazards visible in multiple synchronized views, view-specific instructions promote cross-view coherence, which is subsequently assessed during VLM review; no separate geometric multi-view consistency module is used. The resulting source–edited pair is intended to difer primarily in the inserted hazard.

Each edit undergoes task-aware quality review. A VLM reviewer assesses hazard correctness, physical plausibility, visibility, cross-view consistency, source-scene preservation, and clarity of the planning conflict. Low-scoring edits and a random subset of accepted edits are also inspected manually. Only pairs satisfying both visual-quality and task-validity requirements are retained.

Finally, we construct a hazard region $H _ { i }$ in ego coordinates for each accepted sample. BEVFormer-based localization Li et al. [2022b] provides four ground-plane corners defining $H _ { i }$ as a closed BEV quadrilateral. For physical hazards, $H _ { i }$ represents the occupied or non-drivable region; for signal, right-of-way, and intersection tasks, it represents the task-defined conflict or no-go region. Each region is verified against the intended edit location, multi-view evidence, camera calibration, and task definition, and serves as the spatial reference for hazard-centric evaluation.

## 3.3 Paired Counterfactual Planner Evaluation

After a safety-critical hazard is inserted into a real driving scene, the original human trajectory may pass through the newly introduced hazard and therefore cannot serve as a valid reference trajectory for the edited scene. We therefore evaluate whether the predicted trajectory satisfies the task-defined local safety constraints, rather than using an unique trajectory label. Let the corresponding source and edited trajectories, $\tau _ { i } ^ { o }$ and $\tau _ { i } ^ { e }$ for scene � be:

$$
\tau _ { i } ^ { o } = \left\{ ( x _ { i , t } ^ { o } , y _ { i , t } ^ { o } ) \right\} _ { t = 0 } ^ { T } , \tau _ { i } ^ { e } = \left\{ ( x _ { i , t } ^ { e } , y _ { i , t } ^ { e } ) \right\} _ { t = 0 } ^ { T } .\tag{1}
$$

The source prediction $\tau _ { i } ^ { o }$ is used only as a counterfactual reference for diagnosing behavioral changes, and the safety of $\tau _ { i } ^ { e }$ is evaluated by its geometric relationship with the task-defined hazard region $H _ { i }$

## (a) Hazard Clearance Compliance (HCC)

![](images/83da1a7e346aa183ccb8ba36b0b664506e90a48ba84d38954f65b88bfd1f5583.jpg)

## Clearance Regimes (if no intrusion, urᵢ = 0)

![](images/0fb852795095b0609e6416b403616a829fa499893a16ecd6b75c740b04fb1b5e.jpg)

$$
\mathrm { H C C } = 0
$$

![](images/7b1f3753dbe1f18e8cd83f06d8b5362229ac011702176fd38c3c916979e68748.jpg)

$$
0 < \mathrm { H C C } < 1
$$

$$
\mathrm { d } _ { \mathrm { i } } \geq \mathrm { d } \_ { \mathrm { o } } \mathrm { a f e }
$$

![](images/2e26bf29c0457f75ec6cc6bde75ec51065f5611f0e818a0bf078764294cfe1d7.jpg)

$$
\mathtt { H C C } = 1 \ ( \mathtt { s a t u r a t e s } )
$$

## (b) Hazard Proximity Response (HPR)

![](images/28204578500154dc0b7a25ab72df780e3690dffb0f6b1daccfd4b2ab41f46e2a.jpg)

![](images/8c5f345ae5359847cd46742146771dd9aec7ba1a7e982903a31f00f5902c304d.jpg)

![](images/dd170d7b9a54c63acbfb24faf1ad0637bf06764fa94b3bb0ba8216442ba56d0b.jpg)

![](images/67e09959a964f3f9fc3b2b9ad3387508d1581b9ee238c2df2a2ca91459d3583c.jpg)

![](images/8f6eeff2dcb125125f60ef9afb26fd6ffbba3779749994618d889d23eb7cb2ed.jpg)  
Figure 3: Illustration of the core hazard-centric evaluation metrics used in ExceptionDrive. The blue collision tube is evaluated relative to the task-defined hazard $H _ { i }$ and its required clearance boundary $d _ { \mathrm { s a f e } } .$ . (a) Hazard Clearance Compliance (HCC) quantifies whether the trajectory maintains the prescribed clearance margin. (b) Hazard Proximity Response (HPR) characterizes whether the avoidance response is close to the minimum suficient clearance.

Counterfactual Trajectory Shift (CTS) measures how strongly the planner changes its predicted trajectory after the hazard is introduced:

$$
C T S = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { 1 } { T + 1 } \sum _ { t = 0 } ^ { T } \sqrt { \left( x _ { i , t } ^ { e } - x _ { i , t } ^ { o } \right) ^ { 2 } + \left( y _ { i , t } ^ { e } - y _ { i , t } ^ { o } \right) ^ { 2 } } .\tag{2}
$$

and its longitudinal and lateral components are given in the supplementary material. CTS diagnoses the planner’s sensitivity to the counterfactual hazard, but does not by itself determine whether the resulting response is safe or directionally appropriate.

Unsafe Rate (UR) measures direct intrusion into the task-defined core hazard region. We construct an ego collision tube $\mathbf { c o l l } _ { i }$ from the edited trajectory using oriented vehicle footprints, with details given in the supplementary material. Let $H _ { i }$ denote the point set of the hazard region defined for scene �. We define the unsafe-region intrusion indicator as:

$$
u r _ { i } = \mathbb { I } \left[ \mathbf { c o l l } _ { i } \cap H _ { i } \neq \emptyset \right] ,\tag{3}
$$

where I[·] is the indicator function, which equals one when the enclosed condition holds and zero otherwise. The dataset-level Unsafe Rate is:

$$
U R = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } u r _ { i } .\tag{4}
$$

A lower UR indicates that the planner enters the localized hazard or no-go region in fewer edited cases.

Avoiding direct intrusion is necessary but insuficient, because a safe trajectory may still pass unacceptably close to the hazard. Therefore, Hazard Clearance Compliance (HCC) is used as our primary metric to measure geometric safety-compliance . Let $d _ { i } = d ( \mathbf { c o l l } _ { i } , H _ { i } ) , d _ { i } \geq 0$ denote the minimum Euclidean distance between the ego collision tube and the hazard region. Given the required safety clearance $d _ { \mathrm { s a f e } }$ , HCC is defined as:

$$
H C C = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( 1 - u r _ { i } ) \cdot \operatorname* { m i n } \left( \frac { d _ { i } } { d _ { \mathrm { s a f e } } } , 1 \right) .\tag{5}
$$

HCC is bounded in [0, 1]. It is zero when the trajectory intrudes into the hazard region, increases with clearance for non-overlapping trajectories, and saturates once the required safety clearance is reached.

Hazard Proximity Response (HPR): although a trajectory should maintain suficient clearance, its distance from the hazard also provides information about the magnitude of the avoidance response. We define HPR as:

$$
H P R = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( 1 - u r _ { i } ) \cdot \frac { d _ { \mathrm { s a f e } } } { | d _ { i } - d _ { \mathrm { s a f e } } | + d _ { \mathrm { s a f e } } } .\tag{6}
$$

HPR is zero when the trajectory enters the hazard region. Among safety-compliant trajectories, it reaches one when the minimum clearance equals $d _ { \mathrm { s a f e } }$ and decreases as the trajectory deviates farther from this clearance. Thus, HPR is used to diagnose whether safe avoidance is close to the minimum suficient response. In particular, a high HPR does not by itself imply safety compliance, because trajectories with $d _ { i } < d _ { \mathrm { s a f e } }$ may still obtain a positive HPR value; their insuficient clearance is explicitly captured by HCC.

Illustrated in Fig. 3, CTS, UR, HCC, and HPR together characterize counterfactual sensitivity, direct hazard-region violation, spatial-clearance compliance, and clearance proportionality, respectively. Safety comparisons primarily rely on high HCC and low UR, whereas CTS and HPR provide complementary diagnostic information and should not be interpreted independently as monotonic safety scores.

## 3.4 Task-Grounded Risk Reminder

The planner evaluation described above diagnoses whether a predicted trajectory responds safely to a localized counterfactual hazard, but it does not directly provide an interpretable explanation of the detected risk. We therefore explore a secondary use of the ExceptionDrive taxonomy by constructing a lightweight task-grounded risk Reminder Agent for VLM-based driving agents. The Reminder Agent neither predicts trajectories nor controls the vehicle. Instead, it converts task-relevant visual evidence into a structured risk record that supports high-level driving strategy selection.

Given a driving scene, the reminder first identifies potential hazard evidence and associates it with the applicable risk definitions in the task taxonomy. It then produces a structured record containing hazard presence, hazard type, and a recommended high-level strategy. The reminder has access to the task taxonomy as a collection of risk definitions, but is not provided with the ground-truth task identity or its target strategy for the current sample. The final driving strategy is predicted by a VLM-based driving agent from a predefined strategy space, including actions such as maintaining the current path, slowing down, stopping, yielding, or preparing to bypass.

## 4 Experiments

![](images/ec3ef4e72102c74c58878be1407f0592e6236abd03a1885e9dda65a047c842dd.jpg)

![](images/9917d5f5a54136f8f7f0812ad3cdf93739f079b6868172db6a4a2fa9c17233d6.jpg)  
Figure 4: Representative samples from ExceptionDrive across six safety families. Each example pairs an original image with its counterfactual edit. Red dashed boxes mark the inserted local hazards. The edits preserve the original viewpoint, road layout, illumination, weather, and surrounding trafic context while modifying only the task-relevant risk factor.

## 4.1 Dataset Statistics and Quality Analysis

Starting from 850 nuScenes scenes and 21 predefined tasks, we obtain 17,850 scene–task combinations. Scene applicability filtering identifies 3,446 suitable pairs, accounting for 19.31% of all combinations. After localized editing and VLM-based quality review, 3,181 samples are retained, corresponding to an acceptance rate of 92.31%, and form the final benchmark.

To assess the reliability of the automated review, we randomly inspect 168 VLM-approved samples, covering 5.28% of the benchmark. Among them, 163 pass manual inspection, yielding a 97.02% acceptance rate and indicating a high manual confirmation rate among the audited VLM-approved samples.

<table><tr><td>Model</td><td>HCC%↑</td><td>UR%↓</td><td>HPR%↑</td><td>CTS</td><td>CTSx</td><td>CTSy</td></tr><tr><td>Impromptu VLA Chi et al. [2026]</td><td>19.64</td><td>78.92</td><td>18.87</td><td>2.20</td><td>1.86</td><td>0.70</td></tr><tr><td>OmniDrive Wang et al. [2025]</td><td>9.52</td><td>83.36</td><td>9.84</td><td>1.40</td><td>1.34</td><td>0.22</td></tr><tr><td>AutoVLA Zhou et al. [2026]</td><td>10.37</td><td>81.93</td><td>11.26</td><td>1.97</td><td>1.67</td><td>0.64</td></tr><tr><td>DrivoR Kirby et al. [2026]</td><td>4.86</td><td>92.13</td><td>5.42</td><td>1.25</td><td>1.19</td><td>0.22</td></tr><tr><td>LightEMMA Qiao et al. [2025]</td><td>18.66</td><td>79.51</td><td>18.04</td><td>1.68</td><td>1.61</td><td>0.23</td></tr><tr><td>ST-P3 Hu et al. [2022]</td><td>21.03</td><td>78.11</td><td>18.59</td><td>1.57</td><td>1.55</td><td>0.24</td></tr><tr><td>VAD Jiang et al. [2023]</td><td>13.89</td><td>80.26</td><td>13.47</td><td>1.68</td><td>1.47</td><td>0.51</td></tr></table>

Table 1: Main planner evaluation results on the proposed corner-case benchmark. HCC is the primary metric and measures whether the planner produces the task-defined safe response in edited hazardous scenes. UR measures unsafe response into the annotated hazard region. HPR measures the eficiency of hazard avoidance. CTS reports the magnitude of counterfactual trajectory shift between source and edited inputs, with longitudinal and lateral components denoted by $\mathrm { C T S } _ { x }$ and CTS<sub>�</sub>.
<table><tr><td>Planner</td><td>Emergence</td><td>Blockage</td><td>Bypass</td><td>Signal</td><td>Inter.</td><td>Surface</td></tr><tr><td>autoVLA</td><td>9.90</td><td>12.86</td><td>10.22</td><td>9.78</td><td>9.92</td><td>9.56</td></tr><tr><td>Impromptu VLA</td><td>19.63</td><td>20.28</td><td>19.98</td><td>20.14</td><td>17.86</td><td>19.95</td></tr><tr><td>DrivoR</td><td>4.31</td><td>4.59</td><td>2.61</td><td>5.49</td><td>3.94</td><td>8.27</td></tr><tr><td>ST-P3</td><td>20.02</td><td>19.86</td><td>17.48</td><td>22.92</td><td>24.75</td><td>21.17</td></tr></table>

Table 2: Family-level HCC. The table reports planner response correctness across the six safety families, revealing which types of corner cases are most challenging for each model. Higher HCC indicates a larger fraction of edited hazardous scenes in which the planner produces the task-defined safe response.

Fig. 1 shows the distribution across six safety families. Sudden Emergence, Lane Blockage, Bypass and Gap Conflict, Signal and Rule Conflict, Intersection Conflict, and Road-Surface Hazard account for 25.56%, 19.75%, 24.96%, 3.60%, 9.10%, and 17.03%, respectively. Signal and Rule Conflict is the least frequent because it requires specific intersection layouts, trafic-light states, and right-of-way contexts.

To support failure analysis, qualitative studies, and future expansion, we construct an auxiliary set in addition to the benchmark evaluation set. Following the task distribution of the final benchmark, we select 8,994 additional scene–task pairs from outside the 3,446 screened benchmark candidates and process them directly with the editing model. These pairs are combined with the 265 edits rejected during benchmark quality review, yielding an auxiliary set of 9,259 pairs. The complete ExceptionDrive collection therefore contains 12,440 source–edited pairs: 3,181 quality-approved pairs in the benchmark evaluation set and 9,259 pairs in the auxiliary set. The auxiliary set is not used for the reported benchmark evaluation.

Fig. 4 shows representative original–edited pairs. The approved samples introduce localized task-relevant hazards while largely preserving irrelevant scene content, enabling controlled counterfactual evaluation.

## 4.2 Planner Evaluation on Corner Cases

We evaluate seven representative planners on the final benchmark: Impromptu VLA, OmniDrive, AutoVLA, DrivoR, LightEMMA, ST-P3, and VAD. For each original–edited pair, the navigation command, historical information, and inference configuration remain unchanged. Model-specific input modalities and adaptations to the edited views are described in the supplementary material.

Table 1 shows limited local hazard compliance across all planners: the highest HCC is 21.03%, and every UR exceeds 78%. ST-P3 performs best, with the highest HCC (21.03%) and lowest UR (78.11%), whereas DrivoR obtains the lowest HCC (4.86%) and highest UR (92.13%). Impromptu VLA produces the largest CTS (2.20) without achieving the best safety performance, confirming that trajectory change indicates sensitivity to the edit but not necessarily a safe response. We therefore use HCC and UR as the primary safety metrics and treat HPR and CTS as complementary diagnostics.

Table 2 shows that ST-P3 achieves the highest HCC in most families, while DrivoR reaches only 2.61% on Bypass and Gap Conflict. For Signal and Rule Conflict and Intersection Conflict, HCC measures spatial compliance with task-defined regions rather than complete trafic-rule understanding.

## 4.3 Qualitative Analysis

![](images/8a535f27ca475dbaf381f7f3e598081426380601a5a8bd4ce7909b64d6a9c8f2.jpg)  
Figure 5: Qualitative planner responses across the six ExceptionDrive safety families. For each family, (a) shows the source scene and (b) the edited counterfactual scene. Red dashed boxes denote the inserted hazards, while blue, orange, and green curves show trajectories from AutoVLA, Impromptu VLA, and DrivoR, respectively.

Fig. 5 presents representative planning cases and illustrates three common failure patterns. First, some models follow trajectories close to those predicted in the original scene and directly enter the risk region, resulting in increasing UR and decreaseing HCC. Second, some models react to the edit but shift in an inappropriate direction or by an insuficient amount, producing a large CTS without achieving safe avoidance. Third, some trajectories avoid direct intrusion but remain too close to the risk boundary, leading to a relatively lower UR but limited HCC.

These cases show that intrusion, insuficient clearance, and counterfactual trajectory change are related but non-equivalent phenomena. The qualitative results therefore complement, rather than replace, the quantitative metrics.

## 4.4 Reminder Agent Evaluation

We further evaluate task-grounded reminders as a downstream application of ExceptionDrive, using Qwen2.5- VL-7B Bai et al. [2023] as the final driving agent. The baseline and reminder-augmented settings use identical visual inputs, candidate strategies, and base instructions. The baseline predicts directly from the scene, whereas the augmented setting additionally receives the structured reminder defined in Section 3.4, generated by an auxiliary VLM from the same visual input and shared task taxonomy. Neither setting receives human-annotated risk regions, sample-specific safe actions, or quality-review labels.

To avoid ambiguity in free-form evaluation, Qwen2.5-VL-7B selects one of five predefined strategies, denoted A–E. A prediction is correct if it belongs to the task-defined acceptable response set. We report strategy accuracy and under warning rate, where the latter measures the fraction of hazardous cases in which the selected action is insuficiently cautious.

<table><tr><td>Method</td><td></td><td>Accuracy ↑ Under-warning Rate ↓</td></tr><tr><td>Qwen2.5VL-7B</td><td>51.86</td><td>45.53</td></tr><tr><td>Qwen2.5VL-7B + Reminder</td><td>70.71</td><td>29.18</td></tr></table>

Table 3: High-level strategy evaluation for the zero-shot VLM driving agent.

Table 3 shows that the reminder improves strategy accuracy from 51.86% to 70.71% and reduces the miss rate from 45.53% to 29.18%. These results demonstrate the value of ExceptionDrive for both planner diagnosis and structured risk communication to VLM-based driving agents.

## 5 Conclusion

In this paper, we proposed ExceptionDrive, a planning oriented counterfactual benchmark for evaluating autonomous-driving planners under rare and safety-critical local hazards. ExceptionDrive constructs 21 tasks across six planning-safety families by inserting localized hazards into compatible real-world driving observations while preserving the original scene context. To address the absence of valid expert trajectories after hazard insertion, we proposed a hazard-centric evaluation protocol that assesses hazard intrusion, clearance compliance, avoidance response, and counterfactual behavioral changes. In addition, we developed a lightweight Reminder Agent to transform visual hazard evidence into structured risk records for high-level decision making. Experiments with representative planners demonstrate that existing systems frequently fail to maintain suficient safety clearance, while the Reminder Agent improves zero-shot strategy prediction and reduces under-warning. These results highlight the value of ExceptionDrive for systematically analyzing planner behavior under localized visual interventions that are critical to safety.

## References

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.

Hermann Blum, Paul-Edouard Sarlin, Juan Nieto, Roland Siegwart, and Cesar Cadena. The fishyscapes benchmark: Measuring blind spots in semantic segmentation. International Journal ofComputer Vision, 129(11):3119–3135, 2021.

Jasmin Breitenstein, Jan-Aike Termöhlen, Daniel Lipinski, and Tim Fingscheidt. Systematization of corner cases for visual perception in automated driving. In 2020 IEEE Intelligent Vehicles Symposium (IV), pages 1257–1264. IEEE, 2020.

Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11621–11631, 2020.

Robin Chan, Krzysztof Lis, Svenja Uhlemeyer, Hermann Blum, Sina Honari, Roland Siegwart, Pascal Fua, Mathieu Salzmann, and Matthias Rottmann. Segmentmeifyoucan: A benchmark for anomaly segmentation. arXiv preprint arXiv:2104.14812, 2021.

Kai Chen, Yanze Li, Wenhua Zhang, Yanxin Liu, Pengxiang Li, Ruiyuan Gao, Lanqing Hong, Meng Tian, Xinhai Zhao, Zhenguo Li, et al. Automated evaluation of large vision-language models on self-driving corner cases. In 2025 IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV), pages 7817–7826. IEEE, 2025.

Haohan Chi, Huan-ang Gao, Ziming Liu, Jianing Liu, Chenyu Liu, Jinwei Li, Kaisen Yang, Yangcheng Yu, Zeda Wang, Wenyi Li, et al. Impromptu vla: Open weights and open data for driving vision-language-action models. Advances in Neural Information Processing Systems, 38, 2026.

Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, and Vladlen Koltun. Carla: An open urban driving simulator. In Conference on robot learning, pages 1–16. PMLR, 2017.

Jianwu Fang, Dingxin Yan, Jiahuan Qiao, Jianru Xue, He Wang, and Sen Li. Dada-2000: Can driving accident be predicted by driver attentionƒ analyzed by a benchmark. In 2019 IEEE Intelligent Transportation Systems Conference (ITSC), pages 4303–4309. IEEE, 2019.

Ruiyuan Gao, Kai Chen, Enze Xie, Lanqing Hong, Zhenguo Li, Dit-Yan Yeung, and Qiang Xu. Magicdrive: Street view generation with diverse 3d geometry control. In International Conference on Learning Representations, volume 2024, pages 22841–22860, 2024.

Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? the kitti vision benchmark suite. In 2012 IEEE conference on computer vision and pattern recognition, pages 3354–3361. IEEE, 2012.

Lei Gong, Yu Zhang, Yingqing Xia, Yanyong Zhang, and Jianmin Ji. Sdac: a multimodal synthetic dataset for anomaly and corner case detection in autonomous driving. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 1914–1922, 2024.

Ruiyang Hao, Bowen Jing, Haibao Yu, and Zaiqing Nie. Styledrive: Towards driving-style aware benchmarking of end-to-end autonomous driving. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 4627–4635, 2026.

Florian Heidecker, Jasmin Breitenstein, Kevin Rösch, Jonas Löhdefink, Maarten Bieshaar, Christoph Stiller, Tim Fingscheidt, and Bernhard Sick. An application-driven conceptualization of corner cases for perception in highly automated driving. arXiv preprint arXiv:2103.03678, 2021.

Shengchao Hu, Li Chen, Penghao Wu, Hongyang Li, Junchi Yan, and Dacheng Tao. St-p3: End-to-end vision-based autonomous driving via spatial-temporal feature learning. In European Conference on Computer Vision, pages 533–549. Springer, 2022.

Bernhard Jaeger, Kashyap Chitta, and Andreas Geiger. Hidden biases of end-to-end driving models. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 8240–8249, 2023.

Xiaosong Jia, Zhenjie Yang, Qifeng Li, Zhiyuan Zhang, and Junchi Yan. Bench2drive: Towards multi-ability benchmarking of closed-loop end-to-end autonomous driving. Advances in Neural Information Processing Systems, 37:819–844, 2024.

Bo Jiang, Shaoyu Chen, Qing Xu, Bencheng Liao, Jiajie Chen, Helong Zhou, Qian Zhang, Wenyu Liu, Chang Huang, and Xinggang Wang. Vad: Vectorized scene representation for eficient autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8340–8350, 2023.

Ellington Kirby, Alexandre Boulch, Yihong Xu, Yuan Yin, Gilles Puy, Éloi Zablocki, Andrei Bursuc, Spyros Gidaris, Renaud Marlet, Florent Bartoccioni, et al. Driving on registers. arXiv preprint arXiv:2601.05083, 2026.

Kaican Li, Kai Chen, Haoyu Wang, Lanqing Hong, Chaoqiang Ye, Jianhua Han, Yukuai Chen, Wei Zhang, Chunjing Xu, Dit-Yan Yeung, et al. Coda: A real-world road corner case dataset for object detection in autonomous driving. In European Conference on Computer Vision, pages 406–423. Springer, 2022a.

Xiaofan Li, Yifu Zhang, and Xiaoqing Ye. Drivingdifusion: Layout-guided multi-view driving scenarios video generation with latent difusion model. In European Conference on Computer Vision, pages 469–485. Springer, 2024.

Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Qiao Yu, and Jifeng Dai. Bevformer: Learning bird’s-eye-view representation from multi-camera images via spatiotemporal transformers. arXiv preprint arXiv:2203.17270, 2022b.

Peter Pinggera, Sebastian Ramos, Stefan Gehrig, Uwe Franke, Carsten Rother, and Rudolf Mester. Lost and found: detecting small road hazards for self-driving vehicles. In 2016 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 1099–1106. IEEE, 2016.

Zhijie Qiao, Haowei Li, Zhong Cao, and Henry X Liu. Lightemma: Lightweight end-to-end multimodal model for autonomous driving. arXiv preprint arXiv:2505.00284, 2025.

Kevin Rösch, Florian Heidecker, Julian Truetsch, Kamil Kowol, Clemens Schicktanz, Maarten Bieshaare, Bernhard Sick, and Christoph Stiller. Space, time, and interaction: A taxonomy of corner cases in trajectory datasets for automated driving. In 2022 IEEE Symposium Series on Computational Intelligence (SSCI), pages 86–93. IEEE, 2022.

Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Jens Beißwenger, Ping Luo, Andreas Geiger, and Hongyang Li. Drivelm: Driving with graph visual question answering. In European conference on computer vision, pages 256–274. Springer, 2024.

Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, et al. Scalability in perception for autonomous driving: Waymo open dataset. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 2446–2454, 2020.

Shihao Wang, Zhiding Yu, Xiaohui Jiang, Shiyi Lan, Min Shi, Nadine Chang, Jan Kautz, Ying Li, and Jose M Alvarez. Omnidrive: A holistic vision-language dataset for autonomous driving with counterfactual reasoning. In Proceedings ofthe computer vision and pattern recognition conference, pages 22442–22452, 2025.

Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, Jiagang Zhu, and Jiwen Lu. Drivedreamer: Towards real-world-drive world models for autonomous driving. In European conference on computer vision, pages 55–72. Springer, 2024.

Yuqing Wen, Yucheng Zhao, Yingfei Liu, Fan Jia, Yanhui Wang, Chong Luo, Chi Zhang, Tiancai Wang, Xiaoyan Sun, and Xiangyu Zhang. Panacea: Panoramic and controllable video generation for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6902–6912, 2024.

Benjamin Wilson, William Qi, Tanmay Agarwal, John Lambert, Jagjeet Singh, Siddhesh Khandelwal, Bowen Pan, Ratnesh Kumar, Andrew Hartnett, Jhony Kaesemodel Pontes, et al. Argoverse 2: Next generation datasets for self-driving perception and forecasting. arXiv preprint arXiv:2301.00493, 2023.

Chejian Xu, Wenhao Ding, Weijie Lyu, Zuxin Liu, Shuai Wang, Yihan He, Hanjiang Hu, Ding Zhao, and Bo Li. Safebench: A benchmarking platform for safety evaluation of autonomous vehicles. Advances in Neural Information Processing Systems, 35:25667–25682, 2022.

Zewei Zhou, Tianhui Cai, Seth Zhao, Yun Zhang, Zhiyu Huang, Bolei Zhou, and Jiaqi Ma. Autovla: A vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. Advances in Neural Information Processing Systems, 38:27920–27956, 2026.