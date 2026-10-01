# AVERT-VLN: Abstention-aware Visual Error Recovery and Training for Vision-and-Language Navigation

Minrui Liu<sup>1,†</sup>, Jingke Wang<sup>1,†</sup>, Yuehao Huang<sup>1</sup>, Hao Su<sup>1</sup>, Jiajun Lü<sup>1</sup>, Yukai Ma<sup>1,‡</sup>, Yong Liu<sup>1,∗</sup>

Abstract— Deploying vision-and-language navigation (VLN) agents in unseen environments remains challenging because unfamiliar layouts and visual conditions can cause autonomous execution to go off track. Rather than relying on continuous human supervision, a practical strategy is to let the agent selectively request corrective guidance when needed, use it to recover the ongoing task, and reuse the resulting corrective interactions to improve subsequent navigation. We propose Abstention-aware Visual Error Recovery and Training for Vision-and-Language Navigation (AVERT-VLN), a closed-loop framework that uses a plug-in vision-language Monitor for online human-assisted recovery and offline preference learning. The Monitor operates separately from navigation decision generation and assesses instruction–execution consistency from the instruction, visual history, and current observation. To train the Monitor for deviation recognition, we construct LOSTNAV DATASET with 20K counterfactual risk trajectories and rule-based deviation labels. The Monitor is first fine-tuned on 40K normal trajectories to assess instruction progress and then jointly fine-tuned on normal and risk trajectories to recognize semantic deviations. At runtime, Asynchronous Sidecar Monitoring evaluates execution alongside the navigation model. When the controller accepts a LOST verdict, it suspends autonomous execution and requests human guidance for recovery. For offline policy improvement, Trajectory-Anchored Preference Learning converts deviationassociated failures into decision-level preference pairs under shared decision contexts, restricting supervision to the decisions targeted for correction. Under human-assisted evaluation, the full AVERT-VLN system achieves success rates of 76.2% and 66.3% on the val-unseen splits of R2R-CE and RxR-CE, respectively. The same monitoring and human-assisted recovery interface also improves success rates across the three evaluated navigation architectures. More visualizations are available on our project page: https://minrui-liu.github.io/avert-vln/.

## I. INTRODUCTION

Vision-and-language navigation (VLN) enables embodied agents to follow natural-language route instructions using visual observations, supporting tasks such as delivery and guidance [1], [2]. Recent navigation systems leverage visionlanguage models (VLMs) to interpret instructions, ground navigation targets, and generate navigation decisions [3]–[9]. Nevertheless, deployment in unseen environments exposes VLN agents to layouts and visual conditions not covered during training, where execution may drift from the instruction and continued autonomous actions can compound such errors [10]–[12]. A practical deployment system should therefore detect when execution goes off track, request lightweight human correction when needed rather than rely on continuous supervision, and reuse deployment failures and corrections to improve subsequent navigation.

![](images/f2cfaf974969bca66fa5c298c171fca699493991dc3c66cac18e4b6362f80cd1.jpg)  
Fig. 1. Conceptual comparison of conventional VLN and AVERT-VLN. AVERT-VLN forms a closed deployment loop in which the Monitor detects off-track execution, human guidance supports recovery when needed, and the resulting corrections are reused for offline policy improvement.

Prior work improves deployed navigation models through environment-specific adaptation and feedback-driven learning. GSA-VLN [10] studies continual scene adaptation through repeated navigation in persistent environments. ATENA [11] uses episodic feedback for active test-time adaptation, while user-feedback-driven adaptation [12] incorporates episodelevel success confirmations and goal-level corrections into model updates. Policy improvement from deployment experience is related to, but distinct from, deciding when to request assistance during an ongoing episode. Just Ask [13] enables agents to request assistance through either a confidencebased trigger or an explicit ASK action learned through reinforcement learning, and reuses human–agent interaction histories for further policy learning. These lines of work motivate a complementary question: how can a VLN agent detect when its execution goes off track, seek human assistance to recover the ongoing task, and turn deployment failures into supervision for improving navigation in unseen environments?

To address this question, we propose Abstention-aware Visual Error Recovery and Training for Vision-and-Language Navigation (AVERT-VLN), a closed-loop framework for continuous VLN deployment that connects off-track detection, human-assisted recovery, and failure-driven policy improvement through a shared vision-language Monitor, as illustrated in Fig. 1. The Monitor assesses instruction– execution consistency outside the navigation model’s actiongeneration process, using the instruction, visual history, and current observation. To train this assessor, we construct LOSTNAV DATASET with 20K counterfactual risk trajectories and rule-based deviation labels. Starting from Qwen3.5- 4B [14], we first fine-tune the Monitor on 40K normal trajectories to assess instruction progress, then jointly on normal and risk trajectories to recognize semantic deviations. At runtime, Asynchronous Sidecar Monitoring evaluates execution alongside the navigation model, allowing navigation to proceed while monitoring requests are pending. When the controller accepts a LOST verdict, it suspends autonomous execution and requests a pixel goal in the current view or a local turning command from a human operator. After the corrective action is executed and a new observation becomes available, autonomous navigation and monitoring resume.

For offline policy improvement, a deviation verdict alone does not specify which decision to revise or which alternative to prefer. We therefore develop Trajectory-Anchored Preference Learning, which uses Monitor judgments to locate candidate decision anchors associated with observed deviations. At each anchor, corrective feedback obtained during human-assisted recovery is used to construct a taskconsistent preferred decision, which is paired with the original model output under the same decision context comprising the instruction, visual history, and current observation. We optimize the navigation model with a decision-focused objective based on direct preference optimization (DPO) [15], applying preference supervision only to the decision outputs targeted for correction. Together, Asynchronous Sidecar Monitoring and Trajectory-Anchored Preference Learning connect online recovery with subsequent policy improvement.

## Our contributions are summarized as follows:

• We introduce AVERT-VLN, a closed-loop framework for VLN deployment connecting off-track detection, humanassisted recovery, and failure-driven policy improvement through a shared vision-language Monitor.

• We develop Asynchronous Sidecar Monitoring for abstention and human-assisted recovery, and Trajectory-Anchored Preference Learning to turn deviationassociated failures and corrective feedback into decisionlevel preference pairs under shared decision contexts.

• We construct LOSTNAV DATASET for Monitor training and LOSTAWARE BENCHMARK for evaluating semantic deviation recognition. Experiments on the val-unseen splits of R2R-CE and RxR-CE show human-assisted recovery gains across three navigation architectures, while autonomous ablations isolate the effect of failurederived preference learning from online recovery.

## II. RELATED WORK

## A. Execution Assessment and Interactive Assistance

Prior VLN work has explored execution assessment and interactive assistance for handling ambiguous or off-track states. VNLA [16] enables an agent to query an advisor for language subgoals, whereas Just Ask [13] requests assistance through either an action-confidence threshold or a learned ASK action. Alongside interactive assistance, progress estimation guides navigation [17], while recent VLM-based methods reason about agent state or semantic progress to condition navigation decisions [18], [19]. Navigation Heads detects path deviations from internal attention heads in a frozen navigation model and invokes a lightweight rollback policy for recovery [20]. In contrast, our Monitor operates outside the navigation model’s native action-generation process and provides a shared instruction–execution assessment used by Asynchronous Sidecar Monitoring for human-assisted recovery across navigation architectures.

## B. Learning from Navigation Failures

Learning from navigation failures depends on where to anchor corrective supervision and which target to assign. CorrectNav [21] derives self-correction supervision for perception and action decisions from mined error trajectories, whereas BudVLN [22] retrospectively reanchors corrective supervision to valid historical states. SACA [23] uses stepaware assessment to identify valid prefixes and divergence points, extracting dense corrective supervision from failed trajectories. AeroDPO [24] uses simulator rollback and privileged interventions to construct collision-avoidance preference pairs. In contrast, our Trajectory-Anchored Preference Learning uses Monitor judgments to locate deviation-associated decision anchors and corrective feedback to construct taskconsistent preference pairs under shared decision contexts, while restricting preference optimization to the decision outputs targeted for correction.

## III. METHOD

## A. Overview

AVERT-VLN provides a closed-loop framework for VLN deployment in unseen environments. A shared vision-language Monitor assesses instruction–execution consistency while the navigation model retains its native action-generation process. During execution, Asynchronous Sidecar Monitoring uses Monitor verdicts to coordinate abstention and human-assisted recovery. For subsequent policy improvement, Trajectory-Anchored Preference Learning uses deviation-associated failures and corrective feedback to construct decision-level preference supervision under shared decision contexts.

## B. Counterfactual Risk-Trajectory Construction

We define a navigation deviation as an observable inconsistency between the instruction and the agent’s target selection or navigation behavior. Geometric distance to a reference trajectory is insufficient for identifying such deviations: valid alternative routes may depart from the reference path, whereas an incorrect target may remain geometrically nearby. Accordingly, we construct training data covering instructionconsistent and deviating executions. We assign a LOST label only when execution exhibits a verifiable conflict with the instruction, and pair it with a diagnostic description grounded in the instruction, visual history, and current observation. As shown in Fig. 3, LOSTNAV DATASET follows a threestep pipeline comprising intervention generation, candidate validation, and simulator rollout with annotation.

![](images/2a3cf21d7fdd570b3f6c5cc94ef37e0e457ac8183d3e18f7aca7051afa665269.jpg)  
Fig. 2. Online recovery and offline preference learning in AVERT-VLN. (1) Asynchronous Sidecar Monitoring assesses execution alongside the navigation model and triggers abstention and human-assisted recovery upon a LOST verdict. (2) Trajectory-Anchored Preference Learning traces deviations to the nearest preceding PASS boundary, pairs correction-derived preferred with the original output under a shared context, and updates the navigation model offline using decision-focused DPO.

1) Failure Case Generation: We use the instance and region dictionaries from VL-LN Bench [25] to identify scene entities for region- and object-level interventions. At a fixed execution state, we intervene on target selection and simulate the counterfactual continuation. To generate controlled deviations at different semantic levels, we factor each intervention into a behavior mode $b \in B$ and a target granularity $g \in \mathcal G .$ , where B = {explore, exploit} and $\mathcal { G } = \{ \mathrm { r e g i o n } , \mathrm { o b j e c t } \}$ . The behavior mode controls target commitment: under explore, the intervention keeps the agent searching despite a visible subgoal, whereas under exploit, it redirects the agent toward an incorrect target. The granularity specifies whether the intervention operates at the region or object level, yielding four intervention types in $B \times \mathcal { G }$

2) Candidate Validation: Before simulator rollout, structural checks verify episode–instruction correspondence, action–observation alignment, temporal ordering, and valid simulator-state restoration. Physical and semantic checks verify the selected entity is annotated, visible, navigable from the restored state, and consistent with the intended intervention. Only candidates passing both checks are executed.

3) Simulator Rollout and Annotation: Executing the validated interventions in the simulator produces $\mathcal { D } _ { \mathrm { l o s t } }$ , comprising 20K counterfactual risk trajectories with rendered observations. Rule-based criteria assign deviation labels, while doubao-seed-2-0-pro-260215 generates the corresponding diagnostic descriptions. Scene metadata and privileged navigation states are used only for data construction and verification and are never provided as Monitor inputs.

## C. Monitor Training

Assessing instruction–execution consistency requires tracking instruction progress over the visual history and interpreting the current observation in that context. Let $M _ { \phi }$ denote the Monitor with parameters $\phi .$ We initialize the Monitor from Qwen3.5-4B [14] and fine-tune it in two stages. Both stages condition on the instruction $\mathcal { T } ,$ , a history $\mathcal { H } _ { t }$ of at most eight temporally ordered frames, and the current observation $o _ { t }$ We structure the diagnostic description as

$$
r _ { t } = r _ { t } ^ { \mathrm { h i s t } } \oplus r _ { t } ^ { \mathrm { c u r } } ,\tag{1}
$$

where $r _ { t } ^ { \mathrm { h i s t } }$ summarizes instruction progress supported by the visual history, $r _ { t } ^ { \mathrm { c u r } }$ assesses the current observation against that progress, and $\oplus$ denotes text concatenation.

1) Stage 1: Normal-Trajectory Reasoning: Using the same annotator as for the risk trajectories, we annotate 40K normal trajectories with diagnostic descriptions to form $\mathcal { D } _ { \mathrm { p a s s } } .$ We first fine-tune the Monitor on this dataset to model instruction progress and diagnose the current state under normal execution.

2) Stage 2: Joint Pass/Lost Fine-Tuning: Starting from the Stage-1 checkpoint, we jointly fine-tune the Monitor on $\mathcal { D } _ { \mathrm { p a s s } } \cup \mathcal { D } _ { \mathrm { l o s t } }$ . Normal examples retain the same diagnostic supervision as in Stage 1, while risk examples introduce semantic-deviation diagnostics and the corresponding LOST verdicts. Joint training therefore exposes the Monitor to both instruction-consistent and deviating executions under a common input–output format.

At inference time, the Monitor generates a diagnostic response from which the controller obtains a PASS or LOST verdict, without a separate classification head.

![](images/6de2be19b4e208fd8ebc384bf81b33bbd8dbbc269bb4393b58b92f03732ef114.jpg)  
Fig. 3. Construction of LOSTNAV DATASET. Simulator rollouts of validated target-selection interventions yield counterfactual risk trajectories with rule-based deviation labels and diagnostic annotations.

## D. Asynchronous Sidecar Monitoring

As shown in the upper module of Fig. 2, Asynchronous Sidecar Monitoring closes the online recovery loop by coupling Monitor-based deviation assessment with human intervention, without modifying the navigation model’s native action-generation process. The navigation model issues its native navigation commands, while the controller executes them and the Monitor asynchronously evaluates the resulting observations in micro-batches. Given the instruction I, visual history $\mathcal { H } _ { t }$ of the active execution, and current observation $o _ { t } .$ , the Monitor generates

$$
v _ { t } = M _ { \phi } ( \mathbb { Z } , \mathcal { H } _ { t } , o _ { t } ) ,\tag{2}
$$

where $v _ { t }$ is a diagnostic containing a deviation verdict $z _ { t } \in$ {PASS, LOST} and supporting diagnostic text $r _ { t }$

The commit frontier denotes the latest monitoring boundary such that all committed verdicts up to that boundary are PASS. Previously executed actions are not reverted; recovery starts from the current execution state. Committing a LOST verdict halts unexecuted actions and invalidates pending monitoring requests associated with the current execution, triggering human-assisted recovery. A remote human supplies a pixel goal in the current view or a local turning command, which the controller converts into an executable local action. Navigation and asynchronous monitoring resume once the corrective action yields a new observation.

## E. Trajectory-Anchored Preference Learning

For offline policy improvement, Trajectory-Anchored Preference Learning converts deviation-associated failures and corrective feedback into decision-level preferences, as shown in the lower module of Fig. 2. Because a LOST verdict identifies a failure but not which decision to revise or which alternative to prefer, we use Monitor judgments to locate decision anchors and corrective feedback to construct taskconsistent preference pairs under shared decision contexts.

1) Anchored Preference Construction: For each policygenerated trajectory receiving a LOST verdict, we trace the detected deviation to the nearest preceding boundary judged PASS and treat it as a decision anchor. At the anchor context x, comprising the instruction, visual history, and current observation, corrective feedback provides a taskconsistent preferred decision $y ^ { + }$ , while the original output at the same decision stage serves as $y ^ { - }$ . Specifically, a pixel-goal correction defines the preferred visual-grounding decision, whereas a local-turn correction defines the preferred direction selection decision. Each sample $d = ( x , y ^ { + } , y ^ { - } , k ) \in \mathcal { D } _ { \mathrm { p r e f } }$ additionally records the corrected stage k, corresponding to direction selection or visual grounding.

2) Decision-Focused Optimization: We initialize the trainable policy $\pi _ { \theta }$ and frozen reference policy $\pi _ { \mathrm { r e f } }$ from the same System 2 checkpoint of DualVLN [5]. For each sample $\boldsymbol { d } ~ = ~ ( x , y ^ { + } , y ^ { - } , k )$ , the stage $k$ determines a candidatespecific mask m selecting only tokens from the decision stage targeted for correction; m is instantiated separately for each candidate and shared between $\pi _ { \theta }$ and $\pi _ { \mathrm { r e f } }$ when scoring that candidate. Both policies score each candidate y under the same context x and mask m using

$$
s _ { \pi } ( y \mid x ; m ) = { \frac { \sum _ { j } m _ { j } \log \pi ( y _ { j } \mid x , y _ { < j } ) } { \sum _ { j } m _ { j } } } ,\tag{3}
$$

where $m _ { j } \in \{ 0 , 1 \}$

The reference-adjusted preference margin is

$$
\begin{array} { r l } & { \Delta _ { \theta } ( d ) = \left[ s _ { \pi _ { \theta } } ( y ^ { + } \mid x ; m ) - s _ { \pi _ { \theta } } ( y ^ { - } \mid x ; m ) \right] } \\ & { \phantom { \frac { d } { d } } - \left[ s _ { \pi _ { \mathrm { r e f } } } ( y ^ { + } \mid x ; m ) - s _ { \pi _ { \mathrm { r e f } } } ( y ^ { - } \mid x ; m ) \right] . } \end{array}\tag{4}
$$

We optimize a decision-focused objective based on DPO:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { p o l i c y } } = - \mathbb { E } _ { d \sim \mathcal { D } _ { \mathrm { p r e f } } } \Big [ \log \sigma ( \beta \Delta _ { \theta } ( d ) ) } \\ & { \qquad + \lambda s _ { \pi _ { \theta } } ( y ^ { + } \mid x ; m ) \Big ] . } \end{array}\tag{5}
$$

Here, σ is the sigmoid function, $\beta > 0$ scales the margin, and $\lambda \geq 0$ weights the preferred-output likelihood term.

## IV. EXPERIMENTS

## A. Experimental Setup

We evaluate the Monitor’s ability to recognize instruction deviations, the effect of preference learning on autonomous navigation, and the benefit of online monitoring with humanassisted recovery across navigation architectures.

1) Navigation Benchmarks and Metrics: We evaluate closed-loop navigation on Room-to-Room in Continuous Environments (R2R-CE) and Room-across-Room in Continuous Environments (RxR-CE), using their validation splits in unseen environments (val-unseen) [2], [42]. We report navigation error (NE), oracle success (OS), success rate (SR),

TABLE I  
NAVIGATION PERFORMANCE ON R2R-CE AND RXR-CE VAL-UNSEEN SPLITS.
<table><tr><td rowspan="2">Method</td><td colspan="4">Observation</td><td rowspan="2"></td><td colspan="3">R2R-CE Val-Unseen</td><td colspan="4">RxR-CE Val-Unseen</td></tr><tr><td>Pano.</td><td>Odo.</td><td>Depth</td><td>S.RGB</td><td>NE↓</td><td>OS ↑</td><td>SR ↑ SPL ↑</td><td>NE↓</td><td>SR ↑</td><td>SPL ↑</td><td>nDTW ↑</td></tr><tr><td>HPN+DN* [26]</td><td>√</td><td>√</td><td>√</td><td></td><td>6.31</td><td>40.0</td><td>36.0</td><td>34.0</td><td></td><td>一</td><td>一</td><td>一</td></tr><tr><td>CMA* [27]</td><td>√</td><td>√</td><td>√</td><td></td><td>6.20</td><td>52.0</td><td>41.0</td><td>36.0</td><td>8.76</td><td>26.5</td><td>22.1</td><td>47.0</td></tr><tr><td>GridMM* [28]</td><td>√</td><td>√</td><td>√</td><td></td><td>5.11</td><td>61.0</td><td>49.0</td><td>41.0</td><td>一</td><td>一</td><td>一</td><td></td></tr><tr><td>ETPNav* [29]</td><td>√</td><td>√</td><td>√</td><td></td><td>4.71</td><td>65.0</td><td>57.0</td><td>49.0</td><td>5.64</td><td>54.7</td><td>44.8</td><td>61.9</td></tr><tr><td>ScaleVLN* [30]</td><td>√</td><td>√</td><td>√</td><td></td><td>4.80</td><td></td><td>55.0</td><td>51.0</td><td>一</td><td></td><td>一</td><td></td></tr><tr><td>InstructNav [31]</td><td>√</td><td>√</td><td>√</td><td>√</td><td>6.89</td><td></td><td>31.0</td><td>24.0</td><td></td><td></td><td>一</td><td></td></tr><tr><td>R2R-CMTP [32]</td><td>√</td><td>√</td><td>√</td><td></td><td>7.90</td><td>38.0</td><td>26.4</td><td>22.7</td><td></td><td></td><td>一</td><td></td></tr><tr><td>LAW [33]</td><td></td><td>√</td><td>√</td><td>√</td><td>6.83</td><td>44.0</td><td>35.0</td><td>31.0</td><td>10.90</td><td>8.0</td><td>8.0</td><td>38.0</td></tr><tr><td>CM2 [34]</td><td></td><td>√</td><td>√</td><td>√</td><td>7.02</td><td>41.5</td><td>34.3</td><td>27.6</td><td>一</td><td>一</td><td>一</td><td>一</td></tr><tr><td>WS-MGMap [35]</td><td></td><td>√</td><td>√</td><td>√</td><td>6.28</td><td>47.6</td><td>38.9</td><td>34.3</td><td>1</td><td>1</td><td>1</td><td></td></tr><tr><td>ETPNav + FF [36]</td><td></td><td>√</td><td>√</td><td>√</td><td>5.95</td><td>55.8</td><td>44.9</td><td>30.4</td><td>8.79</td><td>25.5</td><td>18.1</td><td></td></tr><tr><td>Seq2Seq [2]</td><td></td><td></td><td>√</td><td>√</td><td>7.77</td><td>37.0</td><td>25.0</td><td>22.0</td><td>12.10</td><td>13.9</td><td>11.9</td><td>30.8</td></tr><tr><td>CMA [2]</td><td></td><td></td><td>√</td><td>√</td><td>7.37</td><td>40.0</td><td>32.0</td><td>30.0</td><td>一</td><td>一</td><td>一</td><td></td></tr><tr><td>NaVid [4]</td><td></td><td></td><td></td><td>√</td><td>5.47</td><td>49.1</td><td>37.4</td><td>35.9</td><td>一</td><td>一</td><td>一</td><td></td></tr><tr><td>MapNav [37]</td><td></td><td></td><td></td><td>√</td><td>4.93</td><td>53.0</td><td>39.7</td><td>37.2</td><td>一</td><td>1</td><td>一</td><td></td></tr><tr><td>NaVILA [38]</td><td></td><td></td><td></td><td>√</td><td>5.22</td><td>62.5</td><td>54.0</td><td>49.0</td><td>6.77</td><td>49.3</td><td>44.0</td><td>58.8</td></tr><tr><td>UniNaVid [39]</td><td></td><td></td><td></td><td>√</td><td>5.58</td><td>53.3</td><td>47.0</td><td>42.7</td><td>6.24</td><td>48.7</td><td>40.9</td><td></td></tr><tr><td>BudVLN [22]</td><td></td><td></td><td></td><td>√</td><td>4.74</td><td>65.6</td><td>57.6</td><td>51.1</td><td>5.79</td><td>56.1</td><td>46.6</td><td>63.2</td></tr><tr><td>SACA [23]</td><td></td><td></td><td></td><td>√</td><td>4.57</td><td>64.9</td><td>60.3</td><td>55.1</td><td>4.90</td><td>60.3</td><td>49.8</td><td>62.1</td></tr><tr><td>StreamVLN [40]</td><td></td><td></td><td></td><td>√</td><td>4.98</td><td>64.2</td><td>56.9</td><td>51.9</td><td>6.22</td><td>52.9</td><td>46.0</td><td>61.9</td></tr><tr><td>InternVLA-N1 (System 2) [41]</td><td></td><td></td><td></td><td>√</td><td>4.89</td><td>60.6</td><td>55.4</td><td>52.1</td><td>6.41</td><td>49.5</td><td>41.8</td><td>62.6</td></tr><tr><td>DualVLN [5]</td><td></td><td></td><td></td><td>√</td><td>4.05</td><td>70.7</td><td>64.3</td><td>58.5</td><td>4.58</td><td>61.4</td><td>51.8</td><td>70.0</td></tr><tr><td>AVERT-VLN† (StreamVLN)</td><td></td><td></td><td></td><td>√</td><td>3.31</td><td>82.3</td><td>63.7</td><td>42.6</td><td>5.58</td><td>55.0</td><td>47.0</td><td>63.7</td></tr><tr><td>AVERT-VLN† (InternVLA-N1)</td><td></td><td></td><td></td><td>√</td><td>2.97</td><td>78.2</td><td>69.9</td><td>28.7</td><td>4.53</td><td>57.8</td><td>32.7</td><td>57.3</td></tr><tr><td>AVERT-VLN† (DualVLN)</td><td></td><td></td><td></td><td>√</td><td>2.61</td><td>80.7</td><td>75.5</td><td>62.3</td><td>3.93</td><td>65.6</td><td>43.0</td><td>71.7</td></tr><tr><td>AVERT-VLN (DPO + DualVLN)‡</td><td></td><td></td><td></td><td>√</td><td>2.60</td><td>82.5</td><td>76.2</td><td>63.4</td><td>3.83</td><td>66.3</td><td>48.0</td><td>72.0</td></tr></table>

<sup>∗</sup> Methods using pretrained waypoint predictors. <sup>†</sup> Monitor and human-assisted recovery with released navigation checkpoints and no additional navigation model training. <sup>‡</sup> DPO-trained System 2 with subsequently fine-tuned System 1 in the DualVLN architecture, evaluated with Monitor and human-assisted recovery. Shading groups rows by architecture.

## B. Implementation Details

success weighted by path length (SPL) [43], and normalized dynamic time warping (nDTW) [44]. NE is measured in meters; the remaining navigation scores are percentages. Table I also records the observation requirements of each method: panoramic observations (Pano.), odometry (Odo.), depth, and single-view RGB (S.RGB).

2) LOSTAWARE BENCHMARK: We evaluate deviation recognition on paired PASS/LOST samples from English RxR-CE [2], [42]. The benchmark pairs failed Follower trajectories with Guide reference trajectories for the same scene, instruction, and goal. Guides describe prescribed routes while traversing them, whereas Followers navigate independently from these instructions. Within failed Follower trajectories, we identify candidate states using geometric deviation, failure to recover, inconsistency with subsequent Guide regions, and object-context mismatch. States from temporally stable high-score segments are paired with Guide states at matched normalized trajectory progress. Modelassisted review and human adjudication retain 1,095 pairs, each containing one LOST Follower state and one PASS Guide state. All 2,190 samples provide the instruction, visual history, and current observation and are reserved for evaluation. We report recall and F1 with LOST as the positive class.

1) Monitor Training: We initialize the Monitor from Qwen3.5-4B and perform full-parameter fine-tuning on eight NVIDIA H20 GPUs. Following Section III-C, Stage 1 trains on the normal-trajectory dataset $\mathcal { D } _ { \mathrm { p a s s } }$ for one epoch, and Stage 2 continues for one epoch on $\mathcal { D } _ { \mathrm { p a s s } } \cup \mathcal { D } _ { \mathrm { l o s t } }$

2) Navigation Training: For Trajectory-Anchored Preference Learning, we optimize System 2 of DualVLN. We train it for two epochs using the objective in Eq. (5), with $\beta = 0 . 5$ and a preferred-output weight of $\lambda = 0 . 1$

3) Online Monitoring: The navigation model and Monitor run through the Asynchronous Sidecar Monitoring interface. Monitoring requests use a maximum batch size of four and a coalescing timeout of 0.05 s. A committed LOST verdict triggers human-assisted recovery using a pixel goal or a local turning command, as described in Section III-D.

## C. Experimental Results

1) Navigation Performance: Table I evaluates AVERT-VLN across three navigation architectures on the R2R-CE and RxR-CE val-unseen splits. The <sup>†</sup> configurations add the shared Monitor and human-assisted recovery to StreamVLN [40], InternVLA-N1 (System 2) [41], and DualVLN [5] checkpoints without additional navigation-model training. The final <sup>‡</sup> configuration uses the DualVLN dual-system architecture. System 2 is optimized through Trajectory-Anchored Preference Learning, followed by fine-tuning of the corresponding System 1 to adapt low-level execution to the updated System 2. The resulting model is evaluated with the same monitoring and human-assisted recovery.

![](images/7cf78ab6ebf29f57f2f846dbb973ad996b17ee06f7f12d7099a113d74bbacc3d.jpg)

![](images/84c89a2ac7ea82be49c94f19961eac9adb5758b98109836686b52af2c33a77b8.jpg)

![](images/702cae23116791d0a91650dea89351b77e41eca95550cb216a5cea2e59e6eeaa.jpg)  
(a) Diagnostic reasoning  
(b) Lost-state recognition across 18 models  
Fig. 4. Diagnostic reasoning and lost-state recognition. (a) Six-dimensional diagnostic comparison of Monitor-4B and six selected baselines. Annotations report Monitor scores and gaps to the highest displayed baseline score in each dimension. (b) Lost-state detection recall and F1 (%) for all 18 models on the 2,190 states in LOSTAWARE BENCHMARK, with LOST as the positive class. Higher is better for both metrics.

The full configuration achieves SRs of 76.2% on R2R-CE and 66.3% on RxR-CE. Compared with DualVLN using the same monitoring and recovery interface, SR increases by 0.7 points on both benchmarks, while SPL increases by 1.1 and 5.0 points, respectively. The improvement is pronounced in success-weighted path efficiency, particularly on RxR-CE, alongside a modest increase in task completion.

Across the three released navigation architectures, online monitoring and human-assisted recovery increase SR and reduce NE on both benchmarks. SR gains range from 6.8 to 14.5 percentage points on R2R-CE and from 2.1 to 8.3 points on RxR-CE, supporting the applicability of the shared interface across the evaluated architectures. For InternVLA-N1 on RxR-CE, nDTW decreases from 62.6 to 57.3, indicating lower agreement with the reference trajectory. Backtracking and corrective movements during recovery may contribute to this reduction in trajectory fidelity.

2) Lost-State Recognition and Diagnostic Reasoning: We compare Monitor-4B with 17 open-source VLMs [45]– [50]. Monitor-4B achieves the highest recall and F1 at 63.38% and 59.52%, respectively, as shown in Fig. 4(b). These exceed its Qwen3.5-4B initialization by 34.43 and 21.35 percentage points, and Qwen3.5-27B, the strongest F1 baseline, by 20.37 and 9.73 points, respectively. Its precision is 56.10%, compared with 59.10% for Qwen3.5- 27B, reflecting a precision–recall trade-off.

TABLE II  
COMPONENT ABLATION ON RXR-CE VAL-UNSEEN.
<table><tr><td>Configuration</td><td>NE↓</td><td>SR↑</td><td>OS ↑</td><td>nDTW ↑</td></tr><tr><td>System 2</td><td>5.77</td><td>52.88</td><td>65.30</td><td>64.16</td></tr><tr><td>System 2 + SFT</td><td>5.69</td><td>53.41</td><td>65.30</td><td>64.20</td></tr><tr><td> $\mathrm { S y s t e m } \ 2 + \mathrm { D P O }$ </td><td>5.76</td><td>55.03</td><td>65.56</td><td>63.21</td></tr><tr><td>System 2 + Monitor</td><td>4.53</td><td>57.78</td><td>68.71</td><td>57.25</td></tr><tr><td>System 2 + DPO + Monitor</td><td>4.67</td><td>60.21</td><td>70.29</td><td>57.13</td></tr></table>

We use GPT-5.6 Luna to score Monitor-4B and six baselines on the same samples using the same judging prompt. The six-dimensional rubric evaluates consistency with prior observations (history consistency), grounding in visible objects and spatial relations (spatial grounding), identification of completed and remaining instruction steps (instruction progress), explanation of instruction–execution conflicts (deviation diagnosis), observational support for diagnostic claims (evidence faithfulness), and the usefulness of the diagnosis for correcting execution (recovery usefulness). Scores are on a 0–100 scale. Monitor-4B obtains the highest reported scores across all six dimensions in Fig. 4(a). The largest gaps occur in history consistency and recovery usefulness, where it scores 74.93 and 61.44, exceeding the strongest baseline in each dimension by 21.45 and 11.17 points, respectively.

## D. Component Ablation

We use System 2 of DualVLN as the baseline for component ablations on RxR-CE val-unseen (Table II). Combining DPO with online monitoring and human-assisted recovery yields 60.21% SR. Removing online recovery or preference learning reduces SR by 5.18 or 2.43 points, respectively.

You are in a bedroom facing towards the wall, turn slightly left and move forward towards the bed. You can see an open arch which is just beside the bed, move forward and pass through it. Now turn slightly right and move forward onto the carpet. You can see an open door right in front of you which is just beside a cupboard. Enter through the door and move forward. You are now standing in between two wooden cupboards facing towards the wall and this is your end point.

![](images/f9e2ce847b108ce8a1f7e02e5878724858951d52eff6c609157c457d14ecb018.jpg)  
Fig. 5. Qualitative comparison of autonomous navigation before and after DPO. Two selected RxR-CE val-unseen episodes are shown for the original System 2 policy (Original) and its DPO-trained counterpart (+DPO), with online monitoring and human assistance disabled. Each row presents six chronologically ordered observations and the final trajectory map; columns are not temporally aligned across rows. Gold borders mark shared observations; arrows indicate camera motions or navigation actions.

Additional SFT controls for the effect of extra training data. With online monitoring and human assistance disabled, DPO outperforms additional SFT by 1.62 percentage points in SR, supporting failure-derived preference learning for autonomous navigation. In the selected cases shown in Fig. 5, +DPO uses the specified landmarks to choose its direction and reaches the instructed destination, completing tasks that Original fails to finish. Online monitoring with human-assisted recovery also improves task success for System 2 without preference training by enabling correction during execution.

## V. CONCLUSION

In this work, we presented AVERT-VLN, a continuous framework for deploying VLN agents in unseen environments. It detects off-track execution, triggers human-assisted recovery for the ongoing task, and converts deployment failures into decision-level preference supervision for policy improvement. Its asynchronous monitoring and recovery interface preserves the navigation model’s native action-generation process, enabling integration across different navigation architectures. Experiments on R2R-CE and RxR-CE show that humanassisted recovery improves task success across multiple navigation architectures, while autonomous ablations confirm that failure-driven preference learning improves subsequent navigation without online assistance. Together, these results demonstrate the value of coupling selective intervention with failure-driven learning so that deployment failures can be recovered online and reused to improve future navigation.

## REFERENCES

[1] P. Anderson, Q. Wu, D. Teney et al., “Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments,” in 2018 IEEE/CVF conference on computer vision and pattern recognition. IEEE, 2018, pp. 3674–3683.

[2] J. Krantz, E. Wijmans, A. Majumdar, D. Batra, and S. Lee, “Beyond the nav-graph: Vision-and-language navigation in continuous environments,” in European Conference on Computer Vision. Springer, 2020, pp. 104–120.

[3] D. Zheng, S. Huang, L. Zhao, Y. Zhong, and L. Wang, “Towards learning a generalist model for embodied navigation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 13 624–13 634.

[4] J. Zhang, K. Wang, R. Xu et al., “Navid: Video-based vlm plans the next step for vision-and-language navigation,” arXiv preprint arXiv:2402.15852, 2024.

[5] M. Wei, C. Wan, P. Peng et al., “Ground slow, move fast: A dualsystem foundation model for generalizable vision-language navigation,” in International Conference on Learning Representations, vol. 2026, 2026, pp. 12 380–12 396.

[6] Y. Huang, L. Liu, S. Lei et al., “Cogddn: A cognitive demand-driven navigation with decision optimization and dual-process thinking,” in

Proceedings of the 33rd ACM International Conference on Multimedia, 2025, pp. 5237–5246.

[7] Y. Huang, Y. Wu, X. Zhang et al., “Wnm-3d: A world navigation model with 3d scene conditioning for closed-loop vln,” arXiv preprint arXiv:2608.07267, 2026.

[8] X. Li, X. Zhang, Y. Huang et al., “Gn0: Toward a unified paradigm for generation, evaluation, and policy learning in visual-language navigation,” arXiv preprint arXiv:2606.03682, 2026.

[9] H. Su, Y. Huang, Y. Ma, Y. Liu, and J. Lv, “Sage-nav: Leveraging llm planning and alignment fusion for hierarchical scene graph-guided navigation,” arXiv preprint arXiv:2606.25497, 2026.

[10] H. Hong, Y. Qiao, S. Wang, J. Liu, and Q. Wu, “General scene adaptation for vision-and-language navigation,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 69 956–69 983.

[11] H. Ko, S. J. Kim, G. Oh et al., “Active test-time vision-language navigation,” Advances in Neural Information Processing Systems, vol. 38, pp. 44 756–44 775, 2026.

[12] Y. Yu, X. Li, H. Mahmood et al., “User-feedback-driven adaptation for vision-and-language navigation,” IEEE Transactions on Multimedia, 2026.

[13] T.-C. Chi, M. Shen, M. Eric, S. Kim, and D. Hakkani-Tur, “Just ask: An interactive learning framework for vision and language navigation,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, no. 3, 2020, pp. 2459–2466.

[14] Qwen Team, “Qwen3.5-4B model card,” Hugging Face, 2026. [Online]. Available: https://huggingface.co/Qwen/Qwen3.5-4B

[15] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn, “Direct preference optimization: Your language model is secretly a reward model,” Advances in neural information processing systems, vol. 36, pp. 53 728–53 741, 2023.

[16] K. Nguyen, D. Dey, C. Brockett, and B. Dolan, “Vision-based navigation with language-based assistance via imitation learning with indirect intervention,” in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2019, pp. 12 519– 12 529.

[17] C.-Y. Ma, J. Lu, Z. Wu et al., “Self-monitoring navigation agent via auxiliary progress estimation,” arXiv preprint arXiv:1901.03035, 2019.

[18] W. Guo, X. Xu, Y. Liu et al., “Awarevln: Reasoning with self-awareness for vision-language navigation,” arXiv preprint arXiv:2605.22816, 2026.

[19] S. Wang, Y. Wang, G. Lian et al., “Progress-Think: Semantic progress reasoning for vision-language navigation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 4076–4086.

[20] J. Jeong, E. Zhu, J. Lin et al., “Your vision-language-action model already has attention heads for path deviation detection,” arXiv preprint arXiv:2603.13782, 2026.

[21] Z. Yu, Y. Long, Z. Yang et al., “CorrectNav: Self-correction flywheel empowers vision-language-action navigation model,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 22, 2026, pp. 18 737–18 745.

[22] G. He, Z. Liu, K. Xu et al., “Nipping the drift in the bud: Retrospective rectification for robust vision-language navigation,” arXiv preprint arXiv:2602.06356, 2026.

[23] H. Li, R. Liu, H. Fan, and Y. Yang, “Let’s reward step-by-step: Step-aware contrastive alignment for vision-language navigation in continuous environments,” in European Conference on Computer Vision. Springer, 2026, pp. 482–500.

[24] P. Xu, C. Wang, and S. Wan, “AeroDPO: Unleashing lightweight UAV navigation with high-fidelity perception and automated preference optimization,” arXiv preprint arXiv:2608.07557, 2026.

[25] W. Huang, S. Zhu, M. Wei et al., “VL-LN Bench: Towards longhorizon goal-oriented navigation with active dialogs,” arXiv preprint arXiv:2512.22342v1, 2025.

[26] J. Krantz, A. Gokaslan, D. Batra, S. Lee, and O. Maksymets, “Waypoint models for instruction-guided navigation in continuous environments,” in 2021 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2021, pp. 15 142–15 151.

[27] Y. Hong, Z. Wang, Q. Wu, and S. Gould, “Bridging the gap between learning in discrete and continuous environments for vision-andlanguage navigation,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2022, pp. 15 418– 15 428.

[28] Z. Wang, X. Li, J. Yang, Y. Liu, and S. Jiang, “Gridmm: Grid memory map for vision-and-language navigation,” in 2023 IEEE/CVF

International Conference on Computer Vision (ICCV). IEEE, 2023, pp. 15 579–15 590.

[29] D. An, H. Wang, W. Wang et al., “Etpnav: Evolving topological planning for vision-language navigation in continuous environments,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 7, pp. 5130–5145, 2024.

[30] Z. Wang, J. Li, Y. Hong et al., “Scaling data generation in vision-andlanguage navigation,” in 2023 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2023, pp. 11 975–11 986.

[31] Y. Long, W. Cai, H. Wang, G. Zhan, and H. Dong, “Instructnav: Zero-shot system for generic instruction navigation in unexplored environment,” arXiv preprint arXiv:2406.04882, 2024.

[32] K. Chen, J. K. Chen, J. Chuang, M. Vázquez, and S. Savarese, “Topological planning with transformers for vision-and-language navigation,” in 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2021, pp. 11 271–11 281.

[33] S. Raychaudhuri, S. Wani, S. Patel, U. Jain, and A. Chang, “Languagealigned waypoint (law) supervision for vision-and-language navigation in continuous environments,” in Proceedings of the 2021 conference on empirical methods in natural language processing, 2021, pp. 4018– 4028.

[34] G. Georgakis, K. Schmeckpeper, K. Wanchoo et al., “Cross-modal map learning for vision and language navigation,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2022, pp. 15 439–15 449.

[35] P. Chen, D. Ji, K. Lin et al., “Weakly-supervised multi-granularity map learning for vision-and-language navigation,” Advances in Neural Information Processing Systems, vol. 35, pp. 38 149–38 161, 2022.

[36] Z. Wang, X. Li, J. Yang, Y. Liu, and S. Jiang, “Sim-to-real transfer via 3d feature fields for vision-and-language navigation,” arXiv preprint arXiv:2406.09798, 2024.

[37] L. Zhang, X. Hao, Q. Xu et al., “Mapnav: A novel memory representation via annotated semantic maps for vlm-based vision-andlanguage navigation,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2025, pp. 13 032–13 056.

[38] A.-C. Cheng, Y. Ji, Z. Yang et al., “Navila: Legged robot vision-language-action model for navigation,” arXiv preprint arXiv:2412.04453, 2024.

[39] J. Zhang, K. Wang, S. Wang et al., “Uni-navid: A video-based visionlanguage-action model for unifying embodied navigation tasks,” arXiv preprint arXiv:2412.06224, 2024.

[40] M. Wei, C. Wan, X. Yu et al., “Streamvln: Streaming vision-andlanguage navigation via slowfast context modeling,” arXiv preprint arXiv:2507.05240, 2025.

[41] Intern Robotics, Shanghai AI Laboratory, “InternVLA-N1: An open dual-system vision-language navigation foundation model with learned latent plans,” Technical report, 2025. [Online]. Available: https://internrobotics.github.io/internvla-n1.github.io/static/ pdfs/InternVLA\_N1.pdf

[42] A. Ku, P. Anderson, R. Patel, E. Ie, and J. Baldridge, “Roomacross-room: Multilingual vision-and-language navigation with dense spatiotemporal grounding,” in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, 2020, pp. 4392–4412.

[43] P. Anderson, A. Chang, D. S. Chaplot et al., “On evaluation of embodied navigation agents,” arXiv preprint arXiv:1807.06757, 2018.

[44] G. Ilharco, V. Jain, A. Ku, E. Ie, and J. Baldridge, “General evaluation for instruction conditioned navigation using dynamic time warping,” arXiv preprint arXiv:1907.05446, 2019.

[45] S. Bai et al., “Qwen2.5-VL technical report,” arXiv preprint arXiv:2502.13923, 2025.

[46] S. Bai et al., “Qwen3-VL technical report,” arXiv preprint arXiv:2511.21631, 2025.

[47] Qwen Team, “Qwen3.5: Towards native multimodal agents,” Official Qwen Blog, 2026. [Online]. Available: https://qwen.ai/blog?id=qwen3.5

[48] Qwen Team, “Qwen3.6 model collection,” Hugging Face, 2026. [Online]. Available: https://huggingface.co/collections/Qwen/qwen36

[49] J. Zhu, W. Wang, Z. Chen et al., “Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models,” arXiv preprint arXiv:2504.10479, 2025.

[50] W. Wang, Z. Gao, L. Gu et al., “Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency,” arXiv preprint arXiv:2508.18265, 2025.