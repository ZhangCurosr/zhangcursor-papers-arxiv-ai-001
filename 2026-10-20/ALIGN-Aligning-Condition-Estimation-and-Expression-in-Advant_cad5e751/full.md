# ALIGN: Aligning Condition Estimation and Expression in Advantage-Conditioned Behavior Cloning

Zechen Wang<sup>3\*</sup> Zhenqi Qiu<sup>1,3\*</sup> Houlong Xiong<sup>2,3\*</sup> Suohang Zhang<sup>4</sup> Yiyu Ren<sup>3</sup> Wanting Xu<sup>1</sup> Hongfei Niu<sup>3</sup> Chengyang He<sup>5</sup> Ge Sun<sup>5†</sup> Ran Cheng<sup>3</sup> Qian Zhu<sup>3†</sup>

<sup>1</sup> ShanghaiTech University <sup>2</sup> Shanghai Jiao Tong University <sup>3</sup> PrimeBot <sup>4</sup> Zhejiang University <sup>5</sup> National University of Singapore Equal contribution; <sup>†</sup> Corresponding authors

Abstract: Human-collected robot manipulation data often mix successful behavior with retries, pauses, recoveries, and failures. Advantage-Conditioned Behavior Cloning (AC-BC) can exploit such mixed-quality data, but only if the conditioning signal is both accurately constructed from offline trajectories and effectively incorporated into the policy’s action-vector-field prediction. Existing pipelines can weaken this signal by thresholding estimator-derived advantage scores into noisy conditions, and by routing conditions through prompts that weakly affect the action vector field. We introduce ALIGN, a condition-aligned AC-BC framework for vision-language-action policies. ALIGN directly predicts categorical pairwise conditions in the policy-facing label space and injects them into a DiT action expert through AdaRMS modulation fused with flow time. We evaluate ALIGN on deliberately mixed real-robot datasets for ARX X5 bowl stacking and Franka key hanging, each combining standard demonstrations with retry-heavy trajectories. ALIGN improves condition fidelity, strengthens conditional vector-field separability, and achieves 90.0% and 56.7% closed-loop success, compared with 56.7% and 33.3% for the strongest AC-BC baseline. These results indicate that aligning conditions from estimation to action prediction is an effective way to extract robust manipulation policies from data in which success, recovery and error coexist.

![](images/afa478f796ca42a39fd3e830925f778f6f4723e27358c5c23d39cc236d4891ea.jpg)  
Figure 1: Overview of ALIGN. Left: condition estimation. Given a historical observation, the current observation, robot states, and the task instruction, the direct categorical estimator predicts the AC-BC condition $\hat { c } _ { t }$ from an ordered observation pair. Right: AC-BC policy training with estimated conditions. The current observation stream is encoded by the VLM, while AdaRMS combines $\hat { c } _ { t }$ with the flow-time embedding to generate per-layer modulation parameters for the DiT action expert, so the predicted action vector field is explicitly conditioned on the estimated training condition.

10th Conference on Robot Learning (CoRL 2026), Austin, Texas, USA.

Keywords: robot learning, vision-language-action models, imitation learning, advantage-conditioned behavior cloning, flow-matching policies

## 1 Introduction

Robot manipulation datasets collected from human teleoperation are rarely clean: besides successful execution, they contain pauses, retries, recoveries, and erroneous actions. This heterogeneity is important when adapting vision-language-action (VLA) models, which have emerged as strong robot policy backbones [1, 2, 3, 4, 5], including diffusion- and flow-based variants for multimodal action distributions [6, 7]. Advantage-Conditioned Behavior Cloning (AC-BC) offers a natural way to extract useful behavior from such mixed-quality offline data [8, 9, 10]: it estimates behavior quality, converts it into a policy-facing condition, trains imitation conditioned on that label, and deploys the policy under the desired high-quality condition. In principle, conditioning can retain useful recovery behavior while learning from both positive- and negative-conditioned examples. Filtering with the same classifier can also retain recovery frames predicted as positive; the distinction is that filtering discards predicted-negative examples. In practice, its benefit depends on whether the condition is both correctly constructed from data and strongly expressed by the policy.

We argue that existing VLA-based AC-BC pipelines can lose this conditioning signal at both stages. First, the condition is often constructed indirectly: a continuous value, progress, or reward score is evaluated at two frames, differenced, thresholded, and then mapped to a discrete positive/negative label [10, 11, 12]. This indirect conversion is fragile: continuous value/progress estimates are imperfect predictions, differencing can amplify local prediction noise, and thresholding turns nearboundary errors into discrete condition flips. Our temporal-gap supervision likewise represents local task progress, rather than independently annotated action quality or calibrated advantage. We study how this progress-based signal is converted into the condition consumed by AC-BC. The supervision used to define this signal is a separate design axis from estimator parameterization and condition expression. Second, even a correct condition may be weakly expressed. Prompt-based conditioning is convenient for VLA policies, but the condition must pass through the language or vision-language context before reaching the flow-based action expert. If changing the condition barely changes the predicted action vector field, the learned policy has little ability to act differently under different conditions.

We introduce ALIGN, an AC-BC framework that aligns the policy-facing condition from estimation to action prediction. For condition construction, ALIGN replaces post-hoc thresholding of scalar value/progress differences with a direct categorical pairwise estimator, $p _ { \phi } ( c \mid o _ { i } , o _ { j } )$ , trained to predict the same discrete condition consumed by the policy. For condition expression, ALIGN fuses the categorical condition with the flow-time embedding and injects it into the DiT action expert through AdaRMS modulation, giving the condition a direct and repeated influence on vectorfield computation rather than relying only on prompt-level conditioning. Figure 1 summarizes this estimation-to-expression pipeline.

We evaluate ALIGN using standard and retry-heavy successful real-robot demonstrations for policy training; failed trajectories are used only for held-out condition evaluation. Our contributions are threefold:

• We formulate AC-BC for mixed-quality robot data as two coupled problems: condition construction and condition expression, diagnosed by AC-BC Condition Fidelity (ACF) and Conditional Vector-Field Separability (CVFS).

• We propose a direct categorical pairwise condition estimator that predicts the policy-facing AC-BC condition directly, reducing conversion error relative to value/progress thresholding.

• We introduce AdaRMS action-expert injection for flow-based VLA policies and validate ALIGN across two real-robot embodiments under controlled standard and retry-heavy training mixtures, showing improved condition fidelity, stronger condition-dependent vector-field separation, and higher closed-loop success.

## 2 Related Work

Vision-language-action models for robot control. Modern robot policies increasingly use vision-language-action (VLA) backbones that combine perception, language, robot state, and action prediction, spanning embodied multimodal models, scalable transformer policies, crossembodiment data, open-source generalists, and foundation-model-coordinated visuomotor policies [13, 14, 1, 15, 16, 2, 17, 18, 19, 20, 4, 5]. Diffusion- and flow-based variants further model actions through generative dynamics [6, 21, 7, 22, 23]. These backbones improve action modeling but leave open how mixed-quality segments should be selected or conditioned during imitation.

Advantage-conditioned imitation and policy extraction. Offline RL and policy extraction use learned returns, values, rewards, or advantages to bias imitation, including AWR/AWAC, CRR/IQL, and return- or goal-conditioned sequence models [8, 9, 24, 25, 26, 27]. Robot variants learn rewards or progress for data selection, online RL, policy steering, and evaluation [28, 29, 30, 31, 32, 33, 34, 35, 12]; ReWiND improves policies with language-guided rewards without new demonstrations [36], and GR-RL filters progress-positive transitions before VLA specialization [37]. Other work exploits imperfect or mixed-quality data by self-supervised failure scoring, segment selection, value-guided retrials, or human recovery/correction trajectories [38, 39, 40, 41]. Closest to our setting, $\pi _ { 0 . 6 } ^ { * }$ converts learned score differences into positive/negative VLA conditions $[ 1 0 ] ; \chi _ { 0 }$ predicts a stageconditioned pairwise advantage before thresholding it into a binary optimality label [11]. ARM also uses pairwise supervision, but for advantage reward modeling rather than the local condition consumed by AC-BC [42]. ALIGN is complementary: instead of treating value, progress, reward, or advantage as a continuous score for filtering, reweighting, or later thresholding, we directly predict the categorical AC-BC condition and study how it enters the action expert.

Conditioning interfaces for generative action policies. Condition estimation is distinct from condition expression. VLA policies can express advantage through prompts or special tokens [10], while generative policies provide lower-level conditioning through timestep modulation, featurewise affine transforms, or conditional vector-field guidance [43, 44, 45]. We compare action-expert condition-injection interfaces using downstream success and CVFS; among the tested mechanisms, AdaRMS gives the strongest vector-field separation and closed-loop performance.

## 3 Method

Condition-Mediated Decomposition. We formulate AC-BC as a condition-mediated policy extraction process with two coupled stages:

$$
\begin{array} { r l r l } & { c _ { t } = g _ { \phi } ( \mathcal { D } _ { \operatorname* { m i x } } ) , } & & { \mathrm { c o n d i t i o n ~ c o n s t r u c t i o n } , } \\ & { v _ { t } = v _ { \theta } ( x _ { t } , t , o _ { t } , c _ { t } ) , } & { \mathrm { c o n d i t i o n ~ e x p r e s s i o n } . } \end{array}
$$

Here, $\mathcal { D } _ { \mathrm { m i x } }$ denotes mixed-quality offline trajectories, $c _ { t }$ is the policy-facing AC-BC condition, and v is the vector field predicted by the flow-matching action expert. This decomposition makes explicit two places where the conditioning signal can be weakened. The first is condition construction: whether mixed trajectory evidence, including retries, pauses, recoveries, and failures, is mapped to the correct discrete condition consumed by the policy. The second is condition expression: whether changing this condition produces a meaningful change in the features and vector field that determine action prediction. Under this view, successful AC-BC requires not only estimating behavior quality, but also preserving the resulting condition as an effective control variable inside the policy. ALIGN instantiates this principle by directly estimating categorical pairwise conditions in the policy-facing label space and injecting them into the DiT action expert through AdaRMS modulation fused with flow time. Section 4 evaluates these two stages using held-out condition fidelity, vector-field diagnostics, and downstream robot success.

## 3.1 Direct Categorical Condition Estimation

Mapping estimator outputs to AC-BC conditions. Many robot-learning systems estimate continuous task-quality signals [11, 34, 29, 35], but AC-BC uses them only after conversion into a discrete policy-facing condition. For the $\pi _ { 0 . 6 } ^ { * }$ -style and $\chi _ { 0 } \mathrm { - s t y l }$ e baselines, we follow the original advantage-threshold protocols [10, 11]: each method computes an advantage-style score on the local comparison $\left( o _ { t - N } , o _ { t } \right)$ , with $N = 5 0$ , and thresholds that score using the 30% rule. This shared score-to-condition conversion is the condition-construction bottleneck evaluated by ACF, because continuous-score errors near the threshold become policy-facing condition errors. Appendix B.2 gives the full formulas, threshold construction, and implementation details.

Direct categorical condition estimator. We replace post-hoc thresholding of a predicted scalar score with categorical estimation in the condition space consumed by AC-BC. On standard datacollection trajectories, a locally monotone value or progress target makes the value-difference rule $V ( o _ { j } ) - V ( o _ { i } ) > \tau _ { \mathrm { V } }$ correspond, up to local progress scale, to a minimum forward temporal gap $j - i > \tau$ . We set this temporal-gap threshold $\tan { \tau } = 3 0$ frames and use it to supervise pairwise conditions before estimator inference. For a randomly sampled same-trajectory pair $( o _ { i } , o _ { j } )$ ,

$$
\tilde { c } _ { i , j } = g _ { \tau } ( j - i ) , \qquad \mathcal { L } _ { \mathrm { c l s } } = - \log p _ { \phi } ( \tilde { c } _ { i , j } \mid o _ { i } , o _ { j } )
$$

where $g _ { \tau }$ maps the signed frame gap to a categorical condition. After training, we run the estimator on each trajectory to infer frame-level conditions for policy learning. For timestep $t ,$ this applies the fixed local comparison used by AC-BC:

$$
\hat { c } _ { t } ^ { \phi } = \underset { c \in \mathcal { C } } { \arg \operatorname* { m a x } } p _ { \phi } ( c \mid o _ { t - N } , o _ { t } ) , \qquad N = 5 0
$$

Thus, the threshold is used to construct categorical supervision rather than to discretize a noisy scalar prediction after inference. We use binary labels as the main AC-BC interface because they match the deployment-time imitation decision, while the three-class variant tests whether finer forward/neutral/backward granularity helps or adds condition-to-policy expressivity. Appendix B.3 details the design rationale and gap rules, and Appendix A.5 gives the full train/evaluation/deploymen condition flow. This pairwise supervision is related in form to ARM [42], but our estimator targets the local AC-BC condition rather than a global progress model for downstream weighting.

## 3.2 Condition Expression in the Action Expert

By predicting the categorical AC-BC condition directly, the estimator above improves condition fidelity, i.e., the accuracy of the predicted policy-facing condition. However, high condition fidelity is not sufficient unless the policy expresses the condition in the features that determine the action vector field. We therefore view condition expression as a measurable property of the vector-field predictor: under matched open-loop probes, changing only c should induce condition-separable changes in $v _ { \theta } ( x _ { t } , t , o _ { t } , c )$ . Section 4.3 later quantifies this property with Conditional Vector-Field Separability (CVFS), a separability diagnostic for whether different injection mechanisms produce conditiondependent vector-field changes. Textual advantage prompts provide the most direct baseline, e.g., “Advantage: positive” or “Advantage: negative” [10]. For a flow-based action policy, however, this path is indirect: the condition is first processed by the language or vision-language backbone and only later reaches the action expert. We therefore treat condition injection as a progression: keep the condition in the prompt/prefix stream, move it past the VLM backbone into the action-side representation, repeat it throughout the action expert, and finally use it to modulate normalized action-embedding features. This progression is motivated by denoising-path conditioning in robot diffusion policies [6] and adaptive feature modulation in generative transformers [43, 44]; ALIGN instantiates the strongest tested stage with AdaRMS.

AdaRMS action-expert injection. We use AdaRMS because the policy-facing effect of a condition is not merely the presence of a text token, but a change in the vector field computed by the action expert. The action expert already uses the flow-time embedding to select a denoising-stage-specific computation; we treat c as an additional index of that computation. We map the scalar condition label to the flow-time embedding dimension and add it to the processed flow-time embedding; the summed embedding then parameterizes the AdaRMS scale, shift, and gate at each action-expert layer.

This design gives the condition a direct and repeated path into vector-field prediction: at a fixed observation, noisy action, and flow time, changing c changes the modulation applied to actionembedding features. Compared with prompt conditioning, the condition no longer has to survive only through the language/VLM pathway. Compared with a one-time suffix embedding shift, AdaRMS affects both feature amplitude and residual update strength throughout the action expert. The experiments evaluate prompt-level and post-backbone alternatives using both CVFS and closed-loop success, with implementation details in Appendix B.4.

## 4 Experiments

Our evaluation is organized around two questions:

• Does direct categorical estimation yield more faithful AC-BC conditions than $\pi _ { 0 . 6 } ^ { * } -$ and χ<sub>0</sub>-style thresholding?

• Given the same predicted condition, does action-expert injection improve vector-field expression and closed-loop success?

![](images/4ae8e330fb5ea4175a29e37066440d01eccda43b4b4feeee6bbe454bd4f2cd54.jpg)  
Figure 2: Validation-set episodes for ARX X5 bowl stacking and Franka key hanging.

## 4.1 Experimental Setting

We evaluate on two real-robot tasks (Figure 2): ARX X5 bowl stacking and Franka key hanging. For each task, policy training mixes 100 standard trajectories with 100 retry-heavy trajectories containing both incorrect segments and recovery, testing whether AC-BC can preserve useful recovery evidence while down-conditioning low-quality frames. ACF uses task-specific held-out validation sets $\mathcal { D } _ { \mathrm { v a l } }$ collected and annotated for both ARX X5 and Franka, with standard, retry, and failure behavior excluded from training. Closed-loop success is measured over 30 fixed-setup rollouts per policy variant, using the same object poses, initial states, and scene setups across methods for paired comparison. Appendix A gives hardware, data, condition-flow, and hyperparameter details.

We report ACF-Precision, ACF-Recall, and ACF-F1 for estimators, closed-loop success for policies, and CVFS for condition-injection methods.

## 4.2 Estimator Evaluation

To answer the first question, we compare our direct estimator $p _ { \phi } ( c \mid o _ { t - N } , o _ { t } )$ with the $\pi _ { 0 . 6 } ^ { * }$ -style and $\chi _ { 0 }$ -style AC-BC baselines. For a fair ACF comparison, all estimator outputs are evaluated in the same binary AC-BC condition space: continuous-value baselines are thresholded into $\{ + , - \}$ , and direct tri-class predictions are collapsed into {+, −}.

AC-BC Condition Fidelity. ACF measures agreement between the induced condition $\hat { c } _ { t }$ and validation label $c _ { t }$ . We report positive-class precision, recall, and F1: precision penalizes false + assignments on stagnant or regressive frames, while recall measures whether useful progress remains available for imitation. ACF is diagnostic; closed-loop success remains the policy metric. Table 1 reports these metrics for each estimator on the two held-out validation sets. The direct binary estimator achieves the highest F1 on both tasks and the highest recall, improving positive-frame recovery while maintaining high precision and the tri-class variant is close but slightly weaker.

Condition construction remains imperfect for all estimators on non-monotonic robot trajectories. Figure 3 visualizes frame-level condition disagreement on a held-out trajectory: each method deviates from the validation labels(GT) at some timesteps, while the direct categorical estimator yields fewer temporal condition errors relative to thresholded continuous-score baselines. Because manual retry/failure boundaries inevitably encode annotation priors, Appendix A.4 perturbs these boundaries and shows that the direct estimator achieves the best F1 across the tested annotation rules.

Table 1: Condition-estimator ACF on held-out ARX X5 and Franka validation sets.
<table><tr><td></td><td></td><td colspan="3">ARX X5 bowl stacking</td><td colspan="3">Franka key hanging</td></tr><tr><td>Estimator</td><td>Output space</td><td>Precision</td><td>Recall</td><td>F1</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>π0.6-style value</td><td>continuous → {+, −}</td><td>98.1%</td><td>71.1%</td><td>82.5%</td><td>91.4%</td><td>85.8%</td><td>88.5%</td></tr><tr><td>Xo-style stage advantage</td><td>continuous → {+, −}</td><td>97.8%</td><td>70.9%</td><td>82.3%</td><td>92.7%</td><td>75.6%</td><td>83.3%</td></tr><tr><td>Direct tri-class</td><td>categorical → {+, −}</td><td>97.6%</td><td>79.0%</td><td>87.3%</td><td>90.6%</td><td>86.5%</td><td>88.5%</td></tr><tr><td>Direct binary-class (ALIGN)</td><td>{+, −}</td><td>97.7%</td><td>79.2%</td><td>87.5%</td><td>90.7%</td><td>89.1%</td><td>89.9%</td></tr></table>

![](images/e855de56f0a942d8ab5cc08b6c69f67d4fcec678595975029b6ea0dda806777b.jpg)  
Figure 3: Estimator-induced AC-BC conditions on a held-out trajectory.

## 4.3 Policy Evaluation and Injection Analysis

To answer the second question, we isolate the effect of condition injection by fixing the condition estimator and all policy-training settings, and varying only the mechanism by which the predicted condition is fed into the policy.

Conditional Vector-Field Separability. Inspired by Fisher-style separability and variance-ratio criteria [46, 47], CVFS is an open-loop diagnostic for the condition-to-policy stage. A probe is one fixed policy query, consisting of an observation $o _ { i } .$ , a noisy action $x _ { t , i }$ , and a flow time $t _ { i } ;$ for each probe, we vary only the condition. An effective injection mechanism should then produce condition-specific changes in the predicted action vector field. For probe i, condition $c \in { \mathcal { C } } .$ , and $K = | \mathcal { C } |$ , define

$$
\begin{array} { c c } { { v _ { i } ^ { c } = v _ { \theta } ( x _ { t , i } , t _ { i } , o _ { i } , c ) , \displaystyle } } & { { \delta _ { i } ^ { c } = v _ { i } ^ { c } - \frac { 1 } { K } \sum _ { c ^ { \prime } \in \mathcal { C } } v _ { i } ^ { c ^ { \prime } } , \displaystyle } } \\ { { { \mu } _ { c } = \displaystyle { \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \delta _ { i } ^ { c } } , \displaystyle } } & { { \mu = \displaystyle { \frac { 1 } { K } \sum _ { c \in \mathcal { C } } \mu _ { c } } } } \end{array}
$$

where M is the number of probes. We compute between-condition and within-condition scatter,

$$
B = \frac { 1 } { K - 1 } \sum _ { c \in \mathcal { C } } \| \mu _ { c } - \mu \| _ { 2 } ^ { 2 } , \qquad W = \frac { 1 } { K ( M - 1 ) } \sum _ { c \in \mathcal { C } } \sum _ { i = 1 } ^ { M } \| \delta _ { i } ^ { c } - \mu _ { c } \| _ { 2 } ^ { 2 }
$$

and define

$$
\mathrm { C V F S } = { \frac { B } { B + W + \epsilon } }
$$

with numerical constant $\epsilon > 0 .$ . CVFS is high when vector-field changes cluster coherently by condition, and low when the condition is ignored or expressed as inconsistent perturbations. Thus, CVFS does not replace closed-loop success; it measures whether the policy exposes a usable conditiondependent control direction before rollout.

![](images/f47d16ff22db9c8aa8b65f774de3c5cea0eb4c583b88056bd84ff14d53dca0da.jpg)

Figure 4: Condition-dependent vector-field separation across injection mechanisms. Left: t-SNE of vector-field effects at flow time t = 0.2. Right: CVFS over flow time, where stronger post-backbone injection yields larger separation than prompt conditioning.  
![](images/37a50da5b2487f9c10e8dcdbcbe18125791ff116c9e2b70151ca19061ca0d933.jpg)  
Figure 5: Condition-injection ablation with a fixed direct estimator. Bars report flow-time-averaged CVFS and 30-rollout closed-loop success for ARX X5 bowl stacking and Franka key hanging. Hatched bars denote CVFS on the left axis, while solid bars denote success rate on the right axis.

Figure 4 motivates the injection study: prompt conditioning is the natural AC-BC baseline, but its vector-field separation is limited after the condition passes through the language/VLM pathway. We therefore test action-proximal injection using both CVFS and downstream success. The ablation in Figure 5 suggests a positive association between flow-time-averaged CVFS and closed-loop success: methods with larger CVFS generally correspond to higher success, with AdaRMS highest on both.

Condition-Injection Ablation. We fix the direct estimator and vary only the injection mechanism, ordered from prompt/prefix conditioning to action-side, layer-wise, and AdaRMS modulation. This progression reflects our hypothesis that useful AC-BC conditions must be expressed inside the action expert rather than merely encoded in the prompt. It separates gains from bypassing the VLM pathway, repeated action-expert conditioning, and adaptive feature modulation; implementation details are in Appendix B.4.

Table 2 is organized around two experimental controls. The first block fixes prompt/default injection to compare condition construction. Adding retry-heavy trajectories to SFT does not produce a consistent observed improvement across the two tasks. Under the same prompt interface, the direct binary estimator reaches 70.0% on ARX X5 and 40.0% on Franka, with higher observed success than the thresholded baselines and the tri-class variant. The second block fixes this estimator and compares injection mechanisms. ARX X5 success rises from 70.0% with prompts to 90.0% with

Table 2: Closed-loop policy success over 30 rollouts on ARX bowl stacking and Franka key hanging.
<table><tr><td>Policy variant</td><td>Condition source</td><td>Injection</td><td>ARX bowl stacking</td><td>Franka key hanging</td></tr><tr><td colspan="5">Condition construction comparison</td></tr><tr><td>SFT, 100 demonstrations</td><td>none</td><td>none</td><td>46.7%</td><td>6.7%</td></tr><tr><td>SFT, 100 demonstrations + 100 retry trajectories</td><td>none</td><td>none</td><td>50.0%</td><td>3.3%</td></tr><tr><td>π0.6-style threshold</td><td>continuous progress</td><td>prompt/default</td><td>50.0%</td><td>26.7%</td></tr><tr><td>Xo-style threshold</td><td>continuous stage advantage</td><td>prompt/default</td><td>56.7%</td><td>33.3%</td></tr><tr><td>2-class direct estimator 3-class direct estimator</td><td>binary categorical</td><td>prompt</td><td>70.0%</td><td>40.0%</td></tr><tr><td></td><td>tri-class categorical</td><td>prompt</td><td>66.7%</td><td>20.0%</td></tr><tr><td colspan="5">Condition injection comparison with the direct binary estimator</td></tr><tr><td>2-class direct estimator</td><td>binary categorical</td><td>prompt</td><td>70.0%</td><td>40.0%</td></tr><tr><td>2-class direct estimator</td><td>binary categorical</td><td>embedding addition</td><td>76.7%</td><td>40.0%</td></tr><tr><td>2-class direct estimator</td><td>binary categorical</td><td>per-layer additive</td><td>83.3%</td><td>40.0%</td></tr><tr><td>2-class direct estimator</td><td>binary categorical</td><td>outcome token</td><td>76.7% 90.0%</td><td>53.3% 56.7%</td></tr><tr><td>ALIGN(ours)</td><td>binary categorical</td><td>AdaRMS</td><td></td><td></td></tr><tr><td colspan="5">Three-class condition-space extension</td></tr><tr><td>3-class direct estimator</td><td>tri-class categorical</td><td>AdaRMS</td><td>86.7%</td><td>33.3%</td></tr></table>

AdaRMS, and Franka success rises from 40.0% to 56.7%. These point estimates are consistent with contributions from both condition construction and expression, but do not establish that the two components independently improve closed-loop success.

Multitask Categorical Condition Injection. The CVFS analysis above indicates that actionexpert injection makes a categorical variable more explicit in the predicted vector field than prompt conditioning alone. We therefore ask whether this effect is specific to AC-BC advantage labels or reflects broader categorical steerability. We probe this question on the Franka platform in a balanced three-task setting covering bottle-cap unscrewing, bowl stacking, and key hanging: the instruction-only policy receives task-specific language prompts, while the injected policy retains the same prompts and uses AdaRMS to inject the task category through the same pathway used for advantage conditions. Matched task-category injection improves bottle-cap unscrewing from 0.0% to 73.3%, bowl stacking from 26.7% to 80.0%, and key hanging from 0.0% to 50.0%. Appendix C.2 reports the full deployment sweep and further analysis.

## 5 Limitations

ALIGN remains task-specific: the estimator is trained within the studied manipulation tasks and does not target semantic generalization across unseen tasks, instructions, or embodiments. We train from standard data-collection trajectories with temporal-gap labels and reserve manually annotated retry/failure segments for validation to avoid leakage and task-design-specific annotation bias. Consequently, the estimator does not learn directly from labeled negative behavior; incorporating retry, failure, and non-progressive data requires label rules that control human and data-collection bias. Bi nary conditions cannot rank finer quality differences within the positive class. The binary/three-class comparison probes discretization of the same progress-based signal, rather than comparing binary conditioning with calibrated continuous-quality conditioning; it does not isolate label ambiguity from the policy’s ability to use finer conditions. CVFS is an open-loop diagnostic: even when it correlates with success here, separable condition-dependent vector fields need not always improve behavior. Our injection study is also not exhaustive; AdaRMS is strongest among the tested post-backbone mechanisms, while other action-expert interfaces may yield more expressive conditioning.

## 6 Conclusion

We presented ALIGN, a condition-aligned AC-BC framework for extracting robust manipulation policies from mixed-quality robot data. ALIGN targets two key bottlenecks in VLA-based AC-BC: condition construction and condition expression. For condition construction, it replaces thresholded scalar value or progress differences with a direct categorical pairwise estimator that predicts the policyfacing condition. For condition expression, it injects the categorical condition into the DiT action expert through AdaRMS modulation fused with flow time, allowing the condition to directly influence action-vector-field prediction. Across two real-robot manipulation tasks with deliberately mixed standard and retry-heavy trajectories, ALIGN improves closed-loop success while also increasing condition fidelity and conditional vector-field separability. Within the studied tasks and temporal-gap supervision, these results support directly estimating the policy-facing categorical condition and expressing it through action-expert modulation. A supplementary multitask probe further suggests that CVFS and action-expert injection can expose and strengthen categorical steerability beyond advantage labels. Future work should broaden condition definitions and retry/failure supervision without annotation bias.

## Acknowledgments

We thank Youchao Zhang, Youzhang He and Wenqun Wang for their insightful discussion.

## References

[1] B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, F. Xia, J. Wu, P. Wohlhart, S. Welker, A. Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pages 2165–2183. PMLR, 2023.

[2] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

[3] Physical Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, et al. π<sub>0.5</sub>: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

[4] Gemini Robotics Team, S. Abeyruwan, J. Ainslie, J.-B. Alayrac, M. G. Arenas, T. Armstrong, A. Balakrishna, R. Baruch, M. Bauza, M. Blokzijl, et al. Gemini robotics: Bringing ai into the physical world. arXiv preprint arXiv:2503.20020, 2025.

[5] Gemini Robotics Team, A. Abdolmaleki, S. Abeyruwan, J. Ainslie, J.-B. Alayrac, M. G. Arenas, A. Balakrishna, N. Batchelor, A. Bewley, J. Bingham, et al. Gemini robotics 1.5: Pushing the frontier of generalist robots with advanced embodied reasoning, thinking, and motion transfer. arXiv preprint arXiv:2510.03342, 2025.

[6] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, 44(10-11):1684–1704, 2025.

[7] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al. π<sub>0</sub>: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

[8] X. B. Peng, A. Kumar, G. Zhang, and S. Levine. Advantage-weighted regression: Simple and scalable off-policy reinforcement learning. arXiv preprint arXiv:1910.00177, 2019.

[9] A. Nair, A. Gupta, M. Dalal, and S. Levine. Awac: Accelerating online reinforcement learning with offline datasets. arXiv preprint arXiv:2006.09359, 2020.

[10] Physical Intelligence, A. Amin, R. Aniceto, A. Balakrishna, K. Black, K. Conley, G. Connors, J. Darpinian, K. Dhabalia, J. DiCarlo, D. Driess, M. Equi, A. Esmail, Y. Fang, C. Finn, C. Glossop, T. Godden, I. Goryachev, L. Groom, H. Hancock, K. Hausman, G. Hussein, B. Ichter, S. Jakubczak, R. Jen, T. Jones, B. Katz, L. Ke, C. Kuchi, M. Lamb, D. LeBlanc, S. Levine, A. Li-Bell, Y. Lu, V. Mano, M. Mothukuri, S. Nair, K. Pertsch, A. Z. Ren, C. Sharma, L. X. Shi, L. Smith, J. T. Springenberg, K. Stachowicz, W. Stoeckle, A. Swerdlow, J. Tanner, M. Torne, Q. Vuong, A. Walling, H. Wang, B. Williams, S. Yoo, L. Yu, U. Zhilinsky, and Z. Zhou. $\pi _ { 0 . 6 } ^ { * } \colon$ a vla that learns from experience, 2025. URL https://arxiv.org/abs/2511.14759.

[11] C. Yu, C. Sima, G. Jiang, H. Zhang, H. Mai, H. Li, H. Wang, J. Chen, K. Wu, L. Chen, et al. χ<sub>0</sub>: Resource-aware robust manipulation via taming distributional inconsistencies. arXiv preprint arXiv:2602.09021, 2026.

[12] Q. Chen, J. Yu, M. Schwager, P. Abbeel, Y. Shentu, and P. Wu. Sarm: Stage-aware reward modeling for long horizon robot manipulation. arXiv preprint arXiv:2509.25358, 2025.

[13] D. Driess, F. Xia, M. S. Sajjadi, C. Lynch, A. Chowdhery, B. Ichter, A. Wahid, J. Tompson, Q. Vuong, T. Yu, et al. Palm-e: An embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023.

[14] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022.

[15] A. O’Neill, A. Rehman, A. Maddukuri, A. Gupta, A. Padalkar, A. Lee, A. Pooley, A. Gupta, A. Mandlekar, A. Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6892–6903. IEEE, 2024.

[16] Octo Model Team, D. Ghosh, H. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, T. Kreiman, C. Xu, et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024.

[17] M. Shukor, D. Aubakirova, F. Capuano, P. Kooijmans, S. Palma, A. Zouitine, M. Aractingi, C. Pascal, M. Russi, A. Marafioti, et al. Smolvla: A vision-language-action model for affordable and efficient robotics. arXiv preprint arXiv:2506.01844, 2025.

[18] R. Cai, J. Guo, X. He, P. Jin, J. Li, B. Lin, F. Liu, W. Liu, F. Ma, K. Ma, et al. Xiaomi-robotics-0: An open-sourced vision-language-action model with real-time execution. arXiv preprint arXiv:2602.12684, 2026.

[19] J. Bjorck, F. Castañeda, N. Cherniadev, X. Da, R. Ding, L. Fan, Y. Fang, D. Fox, F. Hu, S. Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.

[20] C. He, G. Sun, Y. Bai, J. Lu, J. Zhao, and G. Sartoretti. Falcon: Actively decoupled visuomotor policies for loco-manipulation with foundation-model-based coordination. arXiv preprint arXiv:2512.04381, 2025.

[21] S. Liu, L. Wu, B. Li, H. Tan, H. Chen, Z. Wang, K. Xu, H. Su, and J. Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation. In International Conference on Learning Representations, volume 2025, pages 29982–30009, 2025.

[22] Z. Zhang, H. Li, Y. Dai, Z. Zhu, L. Zhou, C. Liu, D. Wang, F. E. Tay, S. Chen, Z. Liu, et al. From spatial to actions: Grounding vision-language-action model in spatial foundation priors. arXiv preprint arXiv:2510.17439, 2025.

[23] Y. Hu, J. Zhang, Y. Luo, Y. Guo, X. Chen, X. Sun, K. Feng, Q. Lu, S. Chen, Y. Zhang, et al. Bagelvla: Enhancing long-horizon manipulation via interleaved vision-language-action generation. arXiv preprint arXiv:2602.09849, 2026.

[24] Z. Wang, A. Novikov, K. Zolna, J. S. Merel, J. T. Springenberg, S. E. Reed, B. Shahriari, N. Siegel, C. Gulcehre, N. Heess, et al. Critic regularized regression. Advances in Neural Information Processing Systems, 33:7768–7778, 2020.

[25] I. Kostrikov, A. Nair, and S. Levine. Offline reinforcement learning with implicit q-learning. arXiv preprint arXiv:2110.06169, 2021.

[26] L. Chen, K. Lu, A. Rajeswaran, K. Lee, A. Grover, M. Laskin, P. Abbeel, A. Srinivas, and I. Mordatch. Decision transformer: Reinforcement learning via sequence modeling. Advances in neural information processing systems, 34:15084–15097, 2021.

[27] S. Emmons, B. Eysenbach, I. Kostrikov, and S. Levine. Rvs: What is essential for offline rl via supervised learning? arXiv preprint arXiv:2112.10751, 2021.

[28] A. Wagenmaker, M. Nakamoto, Y. Zhang, S. Park, W. Yagoub, A. Nagabandi, A. Gupta, and S. Levine. Steering your diffusion policy with latent space reinforcement learning. arXiv preprint arXiv:2506.15799, 2025.

[29] A. Liang, Y. Korkmaz, J. Zhang, M. Hwang, A. Anwar, S. Kaushik, A. Shah, A. S. Huang, L. Zettlemoyer, D. Fox, et al. Robometer: Scaling general-purpose robotic reward models via trajectory comparisons. arXiv preprint arXiv:2603.02115, 2026.

[30] Y. J. Ma, S. Sodhani, D. Jayaraman, O. Bastani, V. Kumar, and A. Zhang. Vip: Towards universal visual reward and representation via value-implicit pre-training. arXiv preprint arXiv:2210.00030, 2022.

[31] Y. J. Ma, V. Kumar, A. Zhang, O. Bastani, and D. Jayaraman. Liv: Language-image representations and rewards for robotic control. In International Conference on Machine Learning, pages 23301–23320. PMLR, 2023.

[32] S. Sontakke, J. Zhang, S. Arnold, K. Pertsch, E. Bıyık, D. Sadigh, C. Finn, and L. Itti. Roboclip: One demonstration is enough to learn robot policies. Advances in Neural Information Processing Systems, 36:55681–55693, 2023.

[33] T. W. Ayalew, X. Zhang, K. Y. Wu, T. Jiang, M. Maire, and M. R. Walter. Progressor: A perceptually guided reward estimator with self-supervised online refinement. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10297–10306, 2025.

[34] H. Tan, S. Chen, Y. Xu, Z. Wang, Y. Ji, C. Chi, Y. Lyu, Z. Zhao, X. Chen, P. Co, et al. Robodopamine: General process reward modeling for high-precision robotic manipulation. arXiv preprint arXiv:2512.23703, 2025.

[35] S. Chen, C. Harrison, Y.-C. Lee, A. J. Yang, Z. Ren, L. J. Ratliff, J. Duan, D. Fox, and R. Krishna. Topreward: Token probabilities as hidden zero-shot rewards for robotics. arXiv preprint arXiv:2602.19313, 2026.

[36] J. Zhang, Y. Luo, A. Anwar, S. A. Sontakke, J. J. Lim, J. Thomason, E. Biyik, and J. Zhang. Rewind: Language-guided rewards teach robot policies without new demonstrations. arXiv preprint arXiv:2505.10911, 2025.

[37] Y. Li, X. Ma, J. Xu, Y. Cui, Z. Cui, Z. Han, L. Huang, T. Kong, Y. Liu, H. Niu, et al. Gr-rl: Going dexterous and precise for long-horizon robotic manipulation. arXiv preprint arXiv:2512.01801, 2025.

[38] K. Wu, N. Liu, Z. Zhao, D. Qiu, J. Li, Z. Che, Z. Xu, and J. Tang. Learning from imperfect demonstrations with self-supervision for robotic manipulation. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 16899–16906. IEEE, 2025.

[39] J. Chen, H. Fang, H.-S. Fang, and C. Lu. Towards effective utilization of mixed-quality demonstrations in robotic manipulation via segment-level selection and optimization. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 16884–16891. IEEE, 2025.

[40] M. Du, A. Khazatsky, T. Gerstenberg, and C. Finn. To err is robotic: Rapid value-based trial-and-error during deployment. arXiv preprint arXiv:2406.15917, 2024.

[41] Z. Hu, R. Wu, N. Enock, J. Li, R. Kadakia, Z. Erickson, and A. Kumar. Rac: Robot learning for long-horizon tasks by scaling recovery and correction. arXiv preprint arXiv:2509.07953, 2025.

[42] Y. Mao, Z. Yu, W. Mao, Y. Li, Q. Hu, Z. Lan, M. Zhu, and H. Chen. Arm: Advantage reward modeling for long-horizon manipulation. arXiv preprint arXiv:2604.03037, 2026.

[43] W. Peebles and S. Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.

[44] E. Perez, F. Strub, H. De Vries, V. Dumoulin, and A. Courville. Film: Visual reasoning with a general conditioning layer. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018.

[45] J. Ho and T. Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

[46] R. A. Fisher. The use of multiple measurements in taxonomic problems. Annals of eugenics, 7 (2):179–188, 1936.

[47] T. Calinski and J. Harabasz. A dendrite method for cluster analysis.´ Communications in Statistics-theory and Methods, 3(1):1–27, 1974.

## A Hardware Platforms, Dataset Construction, and Data Usage

## A.1 Hardware Platforms

We evaluate ALIGN on two dual-arm tabletop manipulation platforms with different embodiments and control interfaces. Both systems provide the policy with synchronized multi-view RGB observations, robot proprioception, and a task instruction. Figure 6 shows the physical platforms, and Table 3 summarizes their hardware interfaces.

ARX X5. The ARX X5 setup follows an ALOHA-style bimanual manipulation layout. It contains two 6-DoF robot arms, each equipped with a 1-DoF parallel gripper. The visual stream is collected at 30 Hz from three RGB cameras: one front-facing camera covering the tabletop workspace and two wrist-mounted cameras providing local views near the end effectors. Robot commands use 100 Hz position control. We use this platform for the three-bowl stacking task.

Franka Research 3. The Franka setup consists of two Franka Research 3 arms arranged as a dual-arm tabletop system. Each arm provides 7-DoF manipulation and is equipped with a parallel gripper. As in the ARX setting, the workspace is observed at 30 Hz through one front-facing RGB camera and two wrist-mounted RGB cameras. Commands are issued at 200 Hz through the Franka Control Interface under Cartesian impedance control. We use this platform for the key-hanging task.

![](images/98ff17659eaf97269fc7f760baf1e0cc946ce71d21d1430d395e3d52a316ee7d.jpg)  
ARX X5

![](images/c3e868c2a0a3b0a381ec4ea203f8455ac95b69701a074f309753b9363f57b597.jpg)  
Franka  
Figure 6: Real-robot hardware platforms used in our experiments. Left: the ARX X5 platform used for bowl stacking. Right: the Franka platform used for key hanging.

Table 3: Real-robot hardware and observation interfaces used in our experiments.
<table><tr><td>Platform</td><td>Robot embodiment</td><td>End effectors</td><td>Cameras</td><td>Control</td></tr><tr><td>ARX X5</td><td>Dual 6-DoF arms</td><td>Dual 1-DoF parallel grippers</td><td>Front RGB + two wrist RGB, 30 Hz</td><td>100 Hz position control</td></tr><tr><td>Franka Research 3</td><td>Dual 7-DoF arms</td><td>Dual parallel grippers</td><td>Front RGB + two wrist RGB, 30 Hz</td><td>200 Hz Cartesian impedance</td></tr></table>

For policy learning, each observation $o _ { t }$ absorbs the multi-view images, robot state, and task instruction; the paper therefore uses the same observation notation across hardware platforms even though their proprioceptive and control interfaces differ. During closed-loop evaluation, each policy variant is tested on the same fixed task apparatus and the same suite of setup configurations within a platform, so comparisons are not confounded by method-specific changes to camera placement, robot embodiment, or scene layout.

## A.2 Robot Tasks and Data Splits

We evaluate on two real-robot manipulation tasks. The ARX bowl-stacking task uses an ARX X5 arm and requires stacking three bowls in a specified order. The Franka key-hanging task uses a Franka arm and requires placing a key onto a hook. For each task, policy training deliberately combines 100 standard data-collection trajectories with 100 retry-heavy trajectories in which retry actions are inserted by design. These retry trajectories contain useful recovery behavior together with incorrect or stagnant segments, creating mixed-quality training data for testing whether AC-BC can extract recovery evidence more selectively than SFT.

Estimator evaluation uses a separate validation set $\mathcal { D } _ { \mathrm { v a l } }$ that is excluded from both estimator training and policy training. For each task, $\mathcal { D } _ { \mathrm { v a l } }$ contains 100 additional standard data-collection trajectories, 100 additional trajectories with deliberately inserted retry actions, and 50 additional trajectories with deliberately designed failure actions. Closed-loop policy success is measured over 30 robot rollouts per policy variant. A rollout is counted as successful only if the task objective is completed without human intervention.

## A.3 Validation Annotation Protocol

The validation set is designed to test whether an estimator induces the correct frame-level AC-BC condition under standard, retry, and failure behavior. Annotators mark the start frame τ of each designed retry or failure event. The marked events correspond to the deliberately inserted action patterns in the data-collection protocol; therefore, the labels are tied to known intervention points rather than inferred from estimator predictions.

We use short post-event windows to define the binary validation condition. Retry events label the next $K _ { r } = 2 0$ frames as negative, and failure events label the next $K _ { f } = 3 0$ frames as negative:

$$
c _ { t } = \left\{ { \begin{array} { l l } { - , } & { t \in [ \tau , \tau + K _ { r } ] { \mathrm { ~ f o r ~ r e t r y ~ e v e n t s } } } \\ { - , } & { t \in [ \tau , \tau + K _ { f } ] { \mathrm { ~ f o r ~ f a i l u r e ~ e v e n t s } } } \\ { + , } & { { \mathrm { o t h e r w i s e } } } \end{array} } \right.
$$

where + denotes useful imitation evidence and − denotes stagnant or regressive evidence for binary condition evaluation. When multiple annotated windows overlap, the negative label takes precedence. These labels are used only for held-out estimator evaluation and are not used to train the gap-supervised estimator.

## A.4 Annotation-Boundary Robustness of the Direct Estimator

Frame-level retry and failure boundaries are manually specified and therefore can introduce unavoidable boundary bias. To test whether the estimator ranking is tied to one particular annotation rule, we perturb the validation labels in three ways and recompute ACF. The baseline rule is the manual window definition used in the main text. The ignore boundary rule removes m frames on each side of each annotated boundary from evaluation, so ambiguous transition frames do not count as either positive or negative. The expand bad window rule increases each negative retry/failure interval by m frames on both sides, making the bad-label definition more conservative. The shrink bad window rule removes m frames from the beginning and end of each negative interval, keeping only the central part of the annotated bad segment. Tables 4 and 5 report ACF-F1 for the ARX X5 and Franka validation sets, respectively. We report baseline precision/recall/F1 for both validation sets in Table 1; the appendix perturbations show that the direct binary estimator remains the strongest or tied-strongest condition estimator under the tested manual-boundary variants.

Table 4: Annotation-boundary robustness of ACF-F1 on the held-out ARX X5 validation set. The raw row gives the baseline manual labels reported in Table 1; other rows perturb the manually annotated retry/failure boundaries to account for boundary bias.
<table><tr><td>Annotation rule</td><td> $\pi _ { 0 . 6 } ^ { * } \mathrm { F } 1$ </td><td>F1  $\chi _ { 0 }$ </td><td>Binary F1</td><td>Tri F1</td><td>Best F1</td></tr><tr><td>Baseline manual labels</td><td>82.5</td><td>82.3</td><td>87.5</td><td>87.3</td><td>Binary</td></tr><tr><td>Ignore boundary by 1 frame</td><td>83.2</td><td>83.0</td><td>88.4</td><td>88.3</td><td>Binary</td></tr><tr><td>Ignore boundary by 5 frames</td><td>83.3</td><td>83.2</td><td>88.6</td><td>88.4</td><td>Binary</td></tr><tr><td>Ignore boundary by 10 frames</td><td>83.5</td><td>83.3</td><td>88.7</td><td>88.5</td><td>Binary</td></tr><tr><td>Expand bad window by 1 frame</td><td>82.5</td><td>82.2</td><td>87.5</td><td>87.3</td><td>Binary</td></tr><tr><td>Expand bad window by 5 frames</td><td>82.5</td><td>82.2</td><td>87.5</td><td>87.3</td><td>Binary</td></tr><tr><td>Expand bad window by 10 frames</td><td>82.6</td><td>82.2</td><td>87.4</td><td>87.2</td><td>Binary</td></tr><tr><td>Shrink bad window by 1 frame</td><td>82.4</td><td>82.3</td><td>87.5</td><td>87.3</td><td>Binary</td></tr><tr><td>Shrink bad window by 5 frames</td><td>82.4</td><td>82.3</td><td>87.6</td><td>87.4</td><td>Binary</td></tr><tr><td>Shrink bad window by 10 frames</td><td>82.4</td><td>82.3</td><td>87.6</td><td>87.5</td><td>Binary</td></tr></table>

Table 5: Annotation-boundary robustness of ACF-F1 on the held-out Franka validation set. The main text uses the baseline manual labels; other rows perturb the manually annotated retry/failure boundaries to account for boundary bias.
<table><tr><td>Annotation rule</td><td> $\pi _ { 0 . 6 } ^ { * }$  F1</td><td> $\chi _ { 0 } \mathrm { F } 1$ </td><td>Binary F1</td><td>Tri F1</td><td>Best F1</td></tr><tr><td>Baseline manual labels</td><td>88.5</td><td>83.3</td><td>89.9</td><td>88.5</td><td>Binary</td></tr><tr><td>Ignore boundary by 2 frames</td><td>89.2</td><td>83.9</td><td>89.5</td><td>89.2</td><td>Binary</td></tr><tr><td>Ignore boundary by 5 frames</td><td>90.1</td><td>84.8</td><td>90.4</td><td>90.1</td><td>Binary</td></tr><tr><td>Ignore boundary by 10 frames</td><td>91.7</td><td>86.3</td><td>92.0</td><td>91.7</td><td>Binary</td></tr><tr><td>Expand bad window by 2 frames</td><td>88.4</td><td>83.3</td><td>88.7</td><td>88.4</td><td>Binary</td></tr><tr><td>Expand bad window by 5 frames</td><td>88.3</td><td>83.4</td><td>88.5</td><td>88.2</td><td>Binary</td></tr><tr><td>Expand bad window by 10 frames</td><td>88.1</td><td>83.4</td><td>88.1</td><td>87.8</td><td> $\pi _ { 0 . 6 } ^ { * } / \mathrm { B i n a r y }$ </td></tr><tr><td>Shrink bad window by 2 frames</td><td>88.6</td><td>83.2</td><td>88.9</td><td>88.6</td><td>Binary</td></tr><tr><td>Shrink bad window by 5 frames</td><td>88.7</td><td>83.1</td><td>89.1</td><td>88.8</td><td>Binary</td></tr><tr><td>Shrink bad window by 10 frames</td><td>88.9</td><td>82.9</td><td>89.5</td><td>89.1</td><td>Binary</td></tr></table>

## A.5 Condition Flow

Algorithm 1: Condition generation and use.

Input: standard trajectories $\mathcal { D } _ { \mathrm { s t d } } .$ , policy-training trajectories $\mathcal { D } _ { \mathrm { p o l } }$ , local horizon $N = 5 0$ , condition vocabulary C, temporal-gap rule $g _ { \tau }$ with $\tau = 3 0$ frames.

Output: conditioned policy $\pi _ { \boldsymbol { \theta } } ( a \mid o , c _ { \mathrm { g o a l } } )$

1. Estimator training. Randomly sample two frames $( o _ { i } , o _ { j } )$ from the same standard datacollection trajectory in $\mathcal { D } _ { \mathrm { s t d } }$ , without sorting them by time. Assign the weak condition label

$$
\tilde { c } _ { i , j } = g _ { \tau } ( j - i )
$$

and train the estimator with cross-entropy,

$$
\mathcal { L } _ { \mathrm { c l s } } = - \log p _ { \phi } ( \tilde { c } _ { i , j } \mid o _ { i } , o _ { j } )
$$

Deliberately collected retry and failure trajectories are not used for this gap-based estimator training.

2. Policy-training condition assignment. For each frame $t \in \mathcal { D } _ { \mathrm { p o l } }$ , the direct estimator assigns

$$
\hat { c } _ { t } ^ { \phi } = \underset { c \in \mathcal { C } } { \arg \operatorname* { m a x } } p _ { \phi } ( c \mid o _ { t - N } , o _ { t } )
$$

Value-based baselines instead compute $\hat { c } _ { t } ^ { V }$ by thresholding $V ( o _ { t } ) - V ( o _ { t - N } )$ . The induced condition is attached to the policy-training tuple and supplied through the selected prompt or injection mechanism.

3. Policy training. Train the flow-matching policy by behavior cloning on conditioned tuples $\left( o _ { t } , a _ { t } , \hat { c } _ { t } \right)$ . All policy variants use matched data, action representation, initialization, and optimizer settings unless otherwise stated.

4. Closed-loop deployment. The estimator is not run online during robot rollouts. The policy is queried with a fixed desired condition $c _ { \mathrm { g o a l } }$ . For binary policies, $c _ { \mathrm { g o a l } } = +$ . For multi-class policies, $c _ { \mathrm { g o a l } }$ is the most positive progress class.

## A.6 How Conditions Are Used

The same condition variable has three distinct roles. During estimator training, it is a weak label derived from temporal order in standard trajectories. During policy training, it is an offline label attached to each frame and consumed by the policy through a prompt or action-expert injection mechanism. During deployment, it is not re-estimated online; instead, the policy is run under the desired successful condition. This separation avoids leakage from the validation annotations into estimator training and clarifies that the estimator is used to construct policy-training conditions, not as an online closed-loop module.

## B Value/Progress Baselines, Model Architectures, and Training Details

## B.1 Policy Fine-Tuning Setup

All policy variants are fine-tuned from the same $\pi _ { 0 . 5 }$ -style VLA checkpoint with a flow-matching action expert. The observation contains multi-view images, the robot state, and the task instruction, and the policy predicts an action chunk of horizon $H = 5 0$ . Across variants, we keep the backbone, action representation, data split, optimizer, and fine-tuning schedule fixed; only the condition source or the condition-injection mechanism changes.

## B.2 Continuous-Score Baseline Implementation

We treat value and progress-difference baselines under the same value-function interface. Let $V _ { \psi } ( o _ { t } )$ denote a scalar estimator trained on standard data-collection trajectories. Depending on the baseline, the regression target $y _ { t }$ may correspond to a value or progress annotation, and the resulting score is converted to the same AC-BC condition through a local advantage-style test. The estimator is trained with

$$
\mathcal { L } _ { V } = \left\| V _ { \psi } ( o _ { t } ) - y _ { t } \right\| _ { 2 } ^ { 2 }
$$

where $y _ { t }$ is the scalar supervision provided by the corresponding baseline. This keeps the comparison focused on how a continuous estimator is converted into a policy-facing condition, rather than on the particular semantics of its scalar target.

For a queried timestep $t ,$ the value-based baselines compute an $N .$ -step advantage proxy using the same local comparison pair $\left( o _ { t - N } , o _ { t } \right)$

$$
A _ { N } ( t ) = R _ { t - N } ^ { N } + V _ { \psi } ( o _ { t } ) - V _ { \psi } ( o _ { t - N } ) , \qquad R _ { t - N } ^ { N } = \sum _ { k = t - N } ^ { t - 1 } r _ { k }
$$

The $\pi _ { 0 . 6 } ^ { * } { \ - } \mathrm { { s t y } }$ le baseline fixes $N = 5 0$ . We set $\epsilon _ { \mathrm { a d v } }$ by applying the original 30% advantage-threshold rule to $A _ { N } ( t )$ , without tuning it on the held-out validation labels. Since $R _ { t - N } ^ { N }$ is fixed, thresholding $A _ { N } ( t )$ is equivalent to thresholding the value difference:

$$
A _ { N } ( t ) > \epsilon _ { \mathrm { a d v } } \quad \implies \quad V _ { \psi } ( o _ { t } ) - V _ { \psi } ( o _ { t - N } ) > \eta , \qquad \eta = \epsilon _ { \mathrm { a d v } } - R _ { t - N } ^ { N }
$$

The induced binary AC-BC condition is therefore

$$
\hat { c } _ { t } ^ { V } = \left\{ \begin{array} { l l } { + , } & { V _ { \psi } ( o _ { t } ) - V _ { \psi } ( o _ { t - N } ) > \eta , } \\ { - , } & { \mathrm { o t h e r w i s e } } \end{array} \right. \quad \quad N = 5 0
$$

The χ<sub>0</sub>-style baseline is separated from this value-difference conversion. Following the Stage Advantage view, let $g _ { i }$ denote the semantic stage at the earlier frame. Instead of learning a perframe value and subtracting two independent predictions, the stage-advantage estimator is trained on same-episode pairs to predict a pairwise advantage target $A _ { i , j } ^ { \chi }$

$$
\mathcal { L } _ { \chi } = \big \| f _ { \psi } \big ( o _ { i } , o _ { j } , g _ { i } \big ) - A _ { i , j } ^ { \chi } \big \| _ { 2 } ^ { 2 } .
$$

For the local AC-BC comparison used to label frame $t ,$ this gives the continuous pairwise score

$$
\begin{array} { r } { A _ { \psi } ^ { \chi } ( t ) = f _ { \psi } \big ( o _ { t - N } , o _ { t } , g _ { t - N } \big ) . } \end{array}
$$

This avoids the value-baseline noise source $V _ { \psi } ( o _ { t } ) - V _ { \psi } ( o _ { t - N } )$ , but it still produces an intermediate continuous advantage score. We therefore apply the same 30% advantage-threshold protocol to obtain the binary AC-BC interface,

$$
\hat { c } _ { t } ^ { \chi } = \left\{ \begin{array} { l l } { + , } & { A _ { \psi } ^ { \chi } ( t ) > \epsilon _ { \chi } , } \\ { - , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

The two baselines therefore threshold different advantage quantities. The value/progress baseline thresholds an N-step advantage formed from a value difference, whereas the χ -style baseline thresholds the predicted pairwise stage advantage $A _ { \psi } ^ { \chi } ( t )$ directly. ACF is computed after thresholding because the resulting $\hat { c } _ { t }$ , not the raw scalar value, is the condition consumed by AC-BC. ALIGN removes this thresholded-score interface by training $p _ { \phi } ( c \mid o _ { t - N } , o _ { t } )$ directly in the categorical AC-BC condition space. Thus, Table 1 evaluates the final discrete condition consumed by AC-BC, rather than the raw continuous score before thresholding.

## B.3 Direct Categorical Estimator

The direct estimator receives two randomly sampled frames from the same trajectory. The sampled frames are not ordered before labeling; the label is determined by their signed temporal difference using a temporal-gap threshold of $\tau = 3 0$ frames. Each observation contains the same information used by the policy: images, robot state, and task instruction. In the notation, the instruction is absorbed into $o _ { t }$

We consider binary and three-class variants for different reasons. The binary design is the main setting because it matches the downstream AC-BC interface: the policy ultimately needs to distinguish frames that should be imitated under the desired condition from frames that should be down-conditioned. It is therefore intentionally conservative: only pairs with sufficiently forward temporal progress are labeled positive, while reverse or near-stationary pairs are not treated as positive evidence. The threeclass design is a diagnostic label-space variant. It separates clearly backward pairs from temporally ambiguous pairs, allowing us to test whether preserving a neutral middle state improves condition construction or whether the additional class mainly introduces ambiguity for policy conditioning.

For the binary estimator, the condition vocabulary is $\mathcal { C } _ { 2 } = \{ + , - \}$ . Let $\Delta = j - i$ . The binary temporal-gap rule is

$$
g _ { \tau } ^ { ( 2 ) } ( \Delta ) = \left\{ \begin{array} { l l } { + , } & { \Delta > \tau , } \\ { - , } & { \mathrm { o t h e r w i s e } } \end{array} \right.
$$

We use $\tau = 3 0$ frames in all direct-estimator experiments. Thus, only a sufficiently forward pair is labeled $+ ;$ reverse pairs and temporally close pairs are labeled −.

For the three-class estimator, the condition vocabulary is $\mathcal { C } _ { 3 } = \{ \mathrm { g o o d } , \mathrm { m i d } , \mathrm { b a d } \}$ . The label rule is

$$
g _ { \tau } ^ { ( 3 ) } ( i , j ) = \left\{ \begin{array} { l l } { \mathrm { g o o d } , } & { j - i > \tau , } \\ { \mathrm { b a d } , } & { i - j > \tau , } \\ { \mathrm { m i d } , } & { | j - i | \leq \tau } \end{array} \right.
$$

where the middle class represents temporally ambiguous pairs. The estimator is trained with crossentropy on the corresponding label, $\tilde { c } _ { i , j } = g _ { \tau } ^ { ( 2 ) } ( j - i )$ for the binary model and $\tilde { c } _ { i , j } = g _ { \tau } ^ { ( 3 ) } ( i , j )$ for the three-class model. For binary ACF evaluation, three-class outputs are collapsed by treating good as + and both mid and bad ${ \mathrm { a s } } - .$

## B.4 Condition Injection Implementations

All injection variants receive the same scalar condition label $s _ { c } .$ . For binary policies, $s _ { c } = 1 . 0$ denotes + and $s _ { c } = 0 . 0$ denotes −. Multi-class policies use ordered scalar labels, with the most positive class queried at deployment. Let $f _ { c } ( s _ { c } )$ denote the condition embedding MLP, $m _ { t } ( t )$ the processed flow-time embedding, and $h _ { l }$ the action-embedding hidden state at action-expert layer l.

Prompt conditioning. Prompt conditioning appends the condition as a textual suffix to the task instruction, $\mathrm { e . g . }$ ., “Advantage: positive” or “Advantage: negative”. This leaves the policy architecture unchanged, but routes the condition through the language or vision-language backbone before it can affect action-embedding features.

Outcome-token conditioning. Outcome-token conditioning maps $s _ { c }$ through the condition MLP with prefix-token hidden width and appends the resulting vector as one additional prefix token after the image and language tokens:

$$
q _ { c } = f _ { c } ^ { \mathrm { t o k } } ( s _ { c } )
$$

Suffix action embeddings can attend to this prefix token through the usual prefix-to-suffix attention pathway.

Embedding addition. Embedding addition maps $s _ { c }$ to the action-expert width and adds the resulting vector to the initial suffix action embeddings before action-expert processing. In the $\pi _ { 0 . }$ 5 implementation, each action embedding is the linear projection of the noisy action:

$$
h _ { 0 } ^ { a } = W _ { a } x _ { t } + f _ { c } ^ { \mathrm { a d d } } ( s _ { c } )
$$

This can be viewed as prompt-style condition information placed directly on the action side after the VLM prefix.

Per-layer additive conditioning. Per-layer additive conditioning first combines the sinusoidal time embedding and condition embedding,

$$
r ( c , t ) = f _ { \mathrm { t i m e } } ^ { \mathrm { a d d } } ( [ \mathrm { P E } ( t ) , f _ { c } ( s _ { c } ) ] )
$$

and then injects this vector at every action-expert layer through a learned layer-wise scale:

$$
h _ { l } ^ { \prime } = h _ { l } + \alpha _ { l } \odot r ( c , t )
$$

where $\alpha _ { l }$ is the learned scale vector for layer l. This is the post-backbone additive baseline: the condition is persistent across layers, but it enters only as an additive residual.

AdaRMS conditioning. AdaRMS is the main injection mechanism used by ALIGN. The implementation first computes a sinusoidal flow-time embedding with periods in $[ 4 \times 1 0 ^ { - 3 } , 4 ]$ , passes it through a two-layer SiLU MLP, and adds an MLP embedding of the scalar condition label:

$$
m _ { t } ( t ) = \mathrm { S i L U } \big ( W _ { t } ^ { 2 } \mathrm { S i L U } ( W _ { t } ^ { 1 } \mathrm { P E } ( t ) ) \big ) , \qquad z ( c , t ) = m _ { t } ( t ) + f _ { c } ( s _ { c } )
$$

Each AdaRMS site in the action expert then maps $z ( c , t )$ to scale, shift, and gate parameters:

$$
\begin{array} { r l } & { \gamma _ { l } , \beta _ { l } , g _ { l } = D _ { l } ( z ( c , t ) ) , } \\ & { \qquad \tilde { h } _ { l } = ( 1 + \gamma _ { l } ) \odot \mathrm { R M S N o r m } ( h _ { l } ) + \beta _ { l } , } \\ & { \qquad h _ { l + 1 } = h _ { l } + g _ { l } \odot F _ { l } ( \tilde { h } _ { l } ) } \end{array}
$$

Here l indexes an AdaRMS site, $D _ { l }$ is the dense modulation layer inside the corresponding actionexpert RMSNorm, and $F _ { l }$ denotes the associated attention or feed-forward sub-block. Thus, unlike embedding addition, AdaRMS lets the condition alter both feature normalization and residual update strength throughout vector-field prediction.

## B.5 Metric Definitions

ACF measures condition-level agreement on the held-out validation set after every estimator output is mapped into $\{ + , - \}$ :

$$
\mathrm { A C F - P r e c i s i o n } = P ( c _ { t } = + \mid \hat { c } _ { t } = + ) , \qquad \mathrm { A C F - R e c a l l } = P ( \hat { c } _ { t } = + \mid c _ { t } = + )
$$

We also report their harmonic mean,

$$
\mathrm { A C F  – F 1 } = \frac { 2 \mathrm { A C F  – P r e c i s i o n A C F – R e c a l l } } { \mathrm { A C F – P r e c i s i o n + A C F – R e c a l l } }
$$

Precision penalizes false-positive high-condition assignments on retry or failure frames, recall measures whether useful frames are preserved as positive imitation evidence, and F1 summarizes the precision–recall tradeoff.

The main text defines CVFS. In implementation, each matched probe fixes the observation, noisy action, and flow time, sweeps all conditions, and subtracts the probe-specific mean vector field before computing the scatter ratio. Unless otherwise stated, CVFS is evaluated over $t \in \{ 0 , 0 . 1 , 0 . 2 , \ldots , 0 . 9 , 1 . 0 \}$ and averaged over flow time for the bar chart.

## B.6 Training Hyperparameters

Table 6: Policy fine-tuning settings.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Policy backbone</td><td>π0.5-style VLA policy with a flow-matching action expert</td></tr><tr><td>Action horizon</td><td>50</td></tr><tr><td>Training precision</td><td>bfloat16</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>AdamW betas / epsilon</td><td> $\left( 0 . 9 , 0 . 9 5 \right) / 1 0 ^ { - 8 }$ </td></tr><tr><td>Weight decay</td><td>ì0−10</td></tr><tr><td>Learning-rate schedule</td><td>cosine decay</td></tr><tr><td>Peak / final learning rate</td><td> $2 . 5 \times 1 0 ^ { - 5 } / 2 . 5 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Batch size</td><td>256</td></tr><tr><td>Training steps</td><td>5k for conditioned policy fine-tuning</td></tr><tr><td>Default inference denoising steps</td><td>10 Euler steps</td></tr></table>

Table 7: ALIGN and $\pi _ { 0 . 6 } ^ { * }$ -style estimator settings.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Pair sampling</td><td>randomly sample two frames from the same trajectory without sorting their temporal order</td></tr><tr><td>Direct-estimator objective</td><td> $\bar { g _ { \tau } ^ { ( 2 ) } } ( j - i )$   $g _ { \tau } ^ { ( 3 ) } ( i , j )$  cross-entropy on binary or three-class labels</td></tr><tr><td>Value/progress objective</td><td>MSE regression for  $V _ { \psi } ( o _ { t } )$  on scalar value or progress target</td></tr><tr><td>Local comparison horizon</td><td> $N = 5 0$ </td></tr><tr><td>Retry validation window</td><td> $K _ { r } = 2 0$  frames</td></tr><tr><td>Failure validation window</td><td> $K _ { f } = 3 0$  frames</td></tr><tr><td>Binary injection label values Positive-label rule</td><td> $+ \dot { = } 1 . 0 , \ - = 0 . 0$  rank advantage scores; label the top 70% as positive and</td></tr><tr><td>Three-class injection label values</td><td>the rest as negative  $\mathrm { g o o d } = 1 . 0 , \ \mathrm { m i d } = 0 . 8 , \ \mathrm { b a d } = 0 . 0$ </td></tr><tr><td> $\mathrm { A L I G N } / \pi _ { 0 . 6 } ^ { * }$  -style estimator opti- AdamW, mizer</td><td> $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 5 ) , \epsilon = 1 0 ^ { - 8 }$ </td></tr><tr><td> $\mathrm { A L I G N } / \pi _ { 0 . 6 } ^ { * } .$  -style estimator learn- cosine decay, ing rate</td><td> $2 . 5 \times 1 0 ^ { - 5 } \to 2 . 5 \times 1 0 ^ { - 6 }$ </td></tr><tr><td> $\mathrm { A L I G N } / \pi _ { 0 . 6 } ^ { * } { \mathrm { - s t y l } }$  e estimator train- 50k ing steps</td><td></td></tr></table>

Table 8: Matched χ<sub>0</sub>-style estimator settings.
<table><tr><td>Setting</td><td>Value used in our reproduction</td></tr><tr><td>Pair sampling for SA</td><td>randomly sample two frames from the same trajectory without sorting their temporal order</td></tr><tr><td>SA supervision target</td><td>pairwise stage-advantage target  $A _ { i , j } ^ { \chi }$ </td></tr><tr><td>Stage annotation</td><td>semantic sub-goals; 4 stages for ARX X5 bowl stacking and 2 stages for Franka key hanging</td></tr><tr><td>Positive-label rule</td><td>rank local stage-advantage scores; label the top 30% as positive and the rest as negative</td></tr><tr><td>Local comparison horizon</td><td>N = 50, same as ALIGN and  $\pi _ { 0 . 6 } ^ { * } { \ - } { \mathrm { - s t y l e } }$  baselines</td></tr><tr><td>Optimizer Learning rate</td><td>AdamW,  $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 5 ) , \tilde { \epsilon } = 1 0 ^ { - 8 }$ </td></tr><tr><td></td><td>cosine decay,  $2 . 5 \times 1 0 ^ { - 5 } \to 2 . 5 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Training steps</td><td>50k</td></tr></table>

## C Additional Multitask Condition-Injection Analysis

## C.1 Multitask Setup

The main experiments use CVFS to show that action-expert injection increases the vector-field separability induced by a categorical AC-BC condition. As a supplementary behavioral test of this steerability signal, we consider a multitask setting in which the injected categorical variable represents task identity rather than an estimated advantage condition. Task identity is one concrete categorical control variable; this setting asks whether the same injection pathway can make it behaviorally salient beyond the task-specific language prompt.

The multitask policy is trained and evaluated on the Franka platform, using 200 trajectories per task for bottle-cap unscrewing, bowl stacking, and key hanging. Both the prompt-only baseline and the injected policy retain task-specific language instructions. The injected policy additionally conditions on the task category through AdaRMS, using the same categorical-condition embedding and actionexpert modulation path as the advantage-conditioned policy. During evaluation, the injected policy uses the matched category for the queried task.

## C.2 Multitask Categorical Condition Injection

Table 9: Multitask categorical-condition injection on the Franka platform. Each row reports closedloop success on the evaluated physical task. Prompt-only uses only the task-specific language prompt, while AdaRMS uses the same prompt together with the matched task category.

<table><tr><td>Task</td><td>Prompt-only success</td><td>AdaRMS success</td></tr><tr><td>Unscrew bottle cap</td><td>0.0%</td><td>73.3%</td></tr><tr><td>Stack bowls</td><td>26.7%</td><td>80.0%</td></tr><tr><td>Hang key</td><td>0.0%</td><td>50.0%</td></tr></table>

Table 9 reports the matched-category evaluation for each physical task. On matched open-loop multitask probes, AdaRMS increases CVFS from 0.1247 under prompt-only conditioning to 0.4032, indicating that the injected category is more clearly represented in the action vector field. The accompanying success-rate gains suggest that CVFS is aligned with whether the policy learns to use the categorical signal for task-specific behavior.

## D Additional Experiments and Data Details

## D.1 Additional Controls on AGI G2

We evaluate an additional open-wardrobe-door task on AGI G2, with two sub-goals: engaging the handle and opening the door. Table 10 reports results over 50 rollouts per method.

The Filtered-SFT control uses the same classifier to discard predicted-negative frames, which comprise 5.7% in this setting. Filtering can retain recovery frames predicted as positive, whereas AC-BC conditions on both positive and negative examples. The $\pi _ { 0 . 6 } ^ { * }$ -style estimator with AdaRMS provides a separate control that changes the estimator while holding AdaRMS injection fixed.

Table 10: Additional AGI G2 results, with 50 rollouts per method. The first sub-goal is engaging the handle. The final column is the mean time-to-success over successful rollouts only.
<table><tr><td>Method</td><td>Success (%)</td><td>First sub-goal (%)</td><td>Mean time (s)</td></tr><tr><td>Demo</td><td>28.0</td><td>36.0</td><td>50.2</td></tr><tr><td>Demo + retry</td><td>34.0</td><td>46.0</td><td>51.7</td></tr><tr><td> $\pi _ { 0 . 6 } ^ { * }$  thresh. + AdaRMS</td><td>64.0</td><td>70.0</td><td>42.5</td></tr><tr><td>Filtered-SFT</td><td>76.0</td><td>86.0</td><td>50.5</td></tr><tr><td>ALIGN</td><td>86.0</td><td>90.0</td><td>43.2</td></tr></table>

## D.2 Single-Task Data Composition

All policy-training trajectories terminate successfully. The retry-heavy trajectories are collected end-to-end rather than spliced, with one or two deliberate retries followed by recovery. Table 11 summarizes the data pools for each single-task setting. All condition estimators use the same additional standard trajectories, disjoint from policy-training and held-out ACF data. The held-out ACF set is also excluded from policy training; failed trajectories appear only in this evaluation set. Across the three single-task datasets, the fraction of predicted-negative frames ranges from 4.0% to 5.7%, with 5.7% on G2. The collected data include natural placement, lighting, and collector variation.

Table 11: Trajectory counts per single-task setting. The estimator-training, policy-training, and held-out ACF pools are disjoint.
<table><tr><td>Data pool</td><td>Composition</td><td>Total</td></tr><tr><td>Estimator training</td><td>400 standard</td><td>400</td></tr><tr><td>Policy training</td><td>100 standard + 100 retry-heavy</td><td>200</td></tr><tr><td>Held-out ACF</td><td>100 standard + 100 retry + 50 failure</td><td>250</td></tr></table>

## D.3 Uncertainty of the Main Policy Results

Table 12 adds success counts and 95% Wilson intervals to every entry in Table 2. The construction and expression comparisons retain the point-estimate interpretation stated in the main text.

## D.4 Condition Granularity

Table 13 compares binary, three-class, and four-class conditions. Finer discretization of the same signal does not improve these point estimates. This comparison does not test continuous conditioning under calibrated graded-quality supervision. Within the positive class, binary labels do not rank marginal recovery against clean expert motion; recovering a calibrated continuous value lies outside the estimator objective and would require additional supervision.

Table 12: Success counts and 95% Wilson score intervals for every policy entry in Table 2, with $n = 3 0$ rollouts per policy and task. Each cell reports $k / n ;$ success percentage [95% interval], with interval endpoints also in percent. SFT, standard uses 100 standard trajectories; SFT, standard + retry uses 100 standard and 100 retry-heavy trajectories. These are marginal intervals for each policy, not confidence intervals for paired method differences
<table><tr><td>Policy</td><td>ARX X5</td><td>Franka</td></tr><tr><td colspan="3">Condition construction comparison</td></tr><tr><td>SFT, standard</td><td>14/30; 46.7 [30.2, 63.9]</td><td>2/30; 6.7 [1.8, 21.3]</td></tr><tr><td>SFT, standard + retry</td><td>15/30; 50.0 [33.2, 66.8]</td><td>1/30; 3.3 [0.6, 16.7]</td></tr><tr><td>π0.6, prompt/default</td><td>15/30; 50.0 [33.2, 66.8]</td><td>8/30; 26.7 [14.2, 44.4]</td></tr><tr><td>Xo, prompt/default</td><td>17/30; 56.7 [39.2, 72.6]</td><td>10/30; 33.3 [19.2, 51.2]</td></tr><tr><td>Direct binary, prompt</td><td>21/30; 70.0 [52.1, 83.3]</td><td>12/30; 40.0 [24.6, 57.7]</td></tr><tr><td>Direct tri-class, prompt</td><td>20/30; 66.7 [48.8, 80.8]</td><td>6/30; 20.0 [9.5, 37.3]</td></tr><tr><td colspan="3">Condition injection comparison with the direct binary estimator</td></tr><tr><td>Direct binary, prompt</td><td>21/30; 70.0 [52.1, 83.3]</td><td>12/30; 40.0 [24.6, 57.7]</td></tr><tr><td>Direct binary, embedding addition</td><td>23/30; 76.7 [59.1, 88.2]</td><td>12/30; 40.0 [24.6, 57.7]</td></tr><tr><td>Direct binary, per-layer additive</td><td>25/30; 83.3 [66.4, 92.7]</td><td>12/30; 40.0 [24.6, 57.7]</td></tr><tr><td>Direct binary, outcome token</td><td>23/30; 76.7 [59.1, 88.2]</td><td>16/30; 53.3 [36.1, 69.8]</td></tr><tr><td>ALIGN, AdaRMS</td><td>27/30; 90.0 [74.4, 96.5]</td><td>17/30; 56.7 [39.2, 72.6]</td></tr><tr><td colspan="3">Three-class condition-space extension</td></tr><tr><td>Direct tri-class, AdaRMS</td><td>26/30; 86.7 [70.3, 94.7]</td><td>10/30; 33.3 [19.2, 51.2]</td></tr></table>

Table 13: Closed-loop success for different condition granularities. Four-class results are reported for prompt injection only.
<table><tr><td>Injection</td><td>Condition space</td><td>ARX X5 (%)</td><td>Franka (%)</td></tr><tr><td>Prompt</td><td>Binary</td><td>70.0</td><td>40.0</td></tr><tr><td>Prompt</td><td>Three-class</td><td>66.7</td><td>20.0</td></tr><tr><td>Prompt</td><td>Four-class</td><td>66.7</td><td>16.7</td></tr><tr><td>AdaRMS</td><td>Binary</td><td>90.0</td><td>56.7</td></tr><tr><td>AdaRMS</td><td>Three-class</td><td>86.7</td><td>33.3</td></tr></table>

## D.5 Additional Diagnostics

In an exploratory analysis across the six condition-injection variants in Figure 5, CVFS and success are positively correlated: ARX has $r = 0 . 9 5 , p = 0 . 0 0 3$ , and Franka has $r = 0 . 9 3 , p = 0 . 0 0 7$ , using two-sided Pearson tests. ACF and CVFS remain mechanism diagnostics rather than substitutes for closed-loop uncertainty.

Ten additional G2 distractor trajectories, comprising five standard and five retry trajectories, yield an ACF-F1 of 87.1%, compared with 87.6% on the standard G2 held-out set. This check evaluates the estimator’s condition fidelity on these additional trajectories.

## D.6 Scope of the Evidence

Supervision remains a distinct design axis. Our temporal-gap labels encode local task progress, and ACF measures task-specific condition fidelity; neither establishes general-purpose semantic action-quality or calibrated value estimation.

Flow matching can represent a multimodal $p ( a \mid o , c )$ in principle, so sharing a condition need not force deterministic averaging. However, mode preservation at matched observations remains unverified. Joint multitask and mixed-quality training, interacting steering signals, and policy classes beyond the single flow-matching VLA backbone studied here remain untested.