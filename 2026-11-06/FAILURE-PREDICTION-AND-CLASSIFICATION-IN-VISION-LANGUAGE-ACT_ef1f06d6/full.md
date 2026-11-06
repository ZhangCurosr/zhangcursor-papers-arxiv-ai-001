# FAILURE PREDICTION AND CLASSIFICATION IN VISION-LANGUAGE-ACTION MODELS VIA CALIBRATED LATENT PROBING

Michal P. Podolinsky <sup>1</sup> Yunhao Yang <sup>1</sup> Daniel Little <sup>2</sup> Hong Lu <sup>2</sup> Rohan Siva <sup>1</sup> Olin Goudey <sup>2</sup> Neel P. Bhatt <sup>3</sup> Matthias Scheutz <sup>2</sup> Ufuk Topcu <sup>1</sup>

## ABSTRACT

Vision-Language-Action models (VLAs) enable general-purpose embodied intelligence, but their performance is brittle under distribution shift, where unanticipated failures can damage equipment or endanger people. Deploying VLAs in real-world systems therefore requires a runtime monitor that is timely, calibrated, and informative while being efficient enough to run on the edge. Existing methods fail to fulfill these requirements. We present VLA-TUG, a lightweight probe over the latent layers of a VLA backbone that improves early failure detection and bounds the missed-detection and false-alarm rates, with minimal inference overhead. We show that supervising on the pre-failure segment of a trajectory increases early failure detection at matched false-positive rates by up to 30% over the strongest learned baseline. Under shift to unseen tasks, the bounds on missed-detection and false-alarm rates remain valid in up to 86% of runs, versus at most 21% for baselines. Unlike existing methods, VLA-TUG predicts the failure mode to enable corrective actions, raising top-3 accuracy on unseen tasks by up to 44% over a majority baseline. VLA-TUG enables real-time monitoring where the well-timed and informative probe supports lightweight steering that improves task success. We release a dataset of 3,180 annotated manipulation episodes to support future work on latent failure detection in VLAs at vla-tug.github.io.

## 1 INTRODUCTION

Vision-Language-Action models (VLAs) (Physical Intelligence et al., 2025; 2026; NVIDIA et al., 2025) are strong candidates for general-purpose embodied intelligence. By pairing a VLM backbone pre-trained on internet-scale data with an action expert fine-tuned on demonstration trajectories, they excel at complex dexterous tasks such as tightening screws or assembling cardboard boxes. Yet their performance is brittle and can degrade significantly under distribution shift, such as changes in lighting, novel textures, or rare object combinations (Wang et al., 2026; Fei et al., 2025). A VLA can therefore fail unexpectedly, especially as task and environment novelty increases, and such failures can be costly, damaging materials and robots, or outright dangerous to humans. This brittleness has slowed the adop tion of VLAs in factories, homes, and public spaces.

Among multiple solutions, one solution that the robotics community is exploring is VLA steering. The idea of steering comes from research on LLMs, where activations are selectively modified to nudge a model towards a desired output (Turner et al., 2024). Steering VLAs is particularly interesting because it can unlock behaviors that cannot be reached through prompting alone (Buurmeijer et al., 2026; Swann et al., 2026). Various steering methods have recently been proposed (Li et al., 2026b; Liu et al., 2026a; Tan et al., 2026; Swann et al., 2026; Cui et al., 2026; Jeong et al., 2026; Li, 2025), but they rarely use a calibrated probe to decide when to steer. Some steer continuously (Li et al., 2026b; Liu et al., 2026a; Cui et al., 2026), some rely on humanspecified offline triggers (Swann et al., 2026; Li, 2025), and others are gated by a one-sided calibrated trigger (Tan et al., 2026; Jeong et al., 2026). Overall, they lack an efficient and principled trigger that would make them deployable in real-world systems.

Research on failure detection for VLAs is becoming increasingly mainstream (Gu et al., 2025; Agia et al., 2024; Ma et al., 2026; Huang et al., 2026). In a nutshell, these methods compute a failure score from signals available in the VLA, often its latent variables (Gu et al., 2025) or predicted actions (Agia et al., 2024; Huang et al., 2026; Ma et al., 2026), that encode how likely the policy is to fail at the current task. A detection threshold is then calibrated to give probabilistic guarantees on the detector’s performance (Gu et al., 2025). However, these guarantees often cover only the false-positive rate (FPR), i.e. the rate of successes misclassified as failures, and not a target true-positive rate (TPR), i.e. the rate of correctly classified failures, which is arguably even more important in safety-critical settings where a single failure could harm a person or damage costly equipment. These methods are also seldom evaluated systematically against ground-truth failure times, which we argue is crucial to assess whether they can prevent failures before they happen and thus enable VLA steering. Finally, they cannot predict failure modes (grasping, pushing, opening), so steering cannot be targeted towards the root cause and would require human interventions.

![](images/2ae082fd7c140462f0c1e75d2b64090ef864302aad7cf06ae63d2b3fd19fee5a.jpg)  
Figure 1. We introduce VLA-TUG, a probe on the VLA backbone embeddings that predicts failures and their mode. (a) A recurrent encoder maps the embedding into a latent failure space; a score head and a calibrated trigger raise the alarm, and a mode head then predicts the failure mode. (b) VLA-TUG alarms before the failure, allowing early stopping or intervention by an expert model.

We argue that steering would be best supported by a probe that is (T.) timely, detecting failures before they occur; (I.) informative, outputting semantic information beyond a numerical score; (C.) calibrated, achieving the desired error rates; and (E.) efficient to train, calibrate, and deploy.

We address this with VLA-TUG<sup>1</sup>, a well-behaved probe designed to support VLA steering in the real world. We train VLA-TUG on the latent layers of two VLAs (Physical Intelligence et al., 2025; NVIDIA et al., 2025) using 228 distinct tasks in two simulation environments and 8 real-world tasks, and evaluate it against ground-truth failure times. We show that supervision should focus on the pre-failure part of the trajectory to improve early failure detection, and that calibrating the detection threshold to guarantee both TPR and

FPR simultaneously yields guarantees that hold far more often, including on unseen tasks. Building on these findings, we propose a real-time architecture that incorporates VLA-TUG to autonomously support the detection and prevention of VLA failures. Our contributions are as follows:

• VLA-TUG: A timely, informative, and efficient probe for early failure detection. On unseen tasks, our probe improves the area under the early-detection curve by up to 30% over the strongest learned baseline. VLA-TUG also predicts the failure mode, raising the top-3 accuracy on unseen tasks by up to 44% over a majority baseline, at under 1 ms per step.

• Guarantees on false alarms and missed detections. We certify FPR and $\mathrm { T P R } _ { \mathrm { e a r l y } }$ targets jointly; both held in every certified run, compared to 57–71% of runs for baselines. Under distribution shift to unseen tasks, the false-alarm target still held in 40–86% of certified runs, compared to 0–21% for baselines.

• A deployable failure-monitoring system. Through real-world experiments, we demonstrate that integrating the timely failure predictions and associated modes with steering strategies can lead to enhanced success rates.

• An annotated VLA failure dataset. We release a dataset of ≈289k policy-step VLA backbone embeddings from annotated 3,180 episodes of diverse tasks.

## 2 RELATED WORK

Existing methods can detect failures, but their alarms are not necessarily timely, informative, or sufficiently calibrated. Our method addresses these limitations through failureaware representations, onset-based detection, and joint statistical guarantees.

Existing works and their limitations. Existing VLA failure detectors leverage different signals to identify potential failures, including external assessments from world models or LLM judges (Liu et al., 2026b; Li et al., 2026b), action consistency across sampled or consecutive action chunks (Agia et al., 2024; Huang et al., 2026), distributional deviations in policy inputs, outputs, or embeddings (Xu et al., 2025; Gu et al., 2025), and internal policy representations (Gu et al., 2025; Rajaprakash et al., 2026; Ma et al., 2026; Chen, 2026; Park et al., 2026; Francis-Meretzki et al., 2026).

These approaches have three key limitations. (1) Detection performance is commonly evaluated at fixed time points using metrics such as AUROC or balanced accuracy (Gu et al., 2025; Huang et al., 2026), without explicitly measuring whether alarms precede the actual failure onset. Moreover, supervision ranges from episode-level failure labels assigned to all timesteps (Gu et al., 2025), to heuristic or human-refined timestep labels (Ma et al., 2026), to failurefree training (Xu et al., 2025), without explicitly accounting for the timing of failures. (2) Most detectors produce scalar risk scores without identifying the underlying failure modes. While semantic supervision and representation interpretability have been explored for embodied policies (Xu et al., 2024; Swann et al., 2026; Buurmeijer et al., 2026), they do not directly establish informative failure alarms. (3) Existing calibration methods typically control a single criterion. Conformal methods provide false-positive guarantees for sequential failure detection (Gu et al., 2025; Rajaprakash et al., 2026; Park et al., 2026), calibrate anomaly or failure scores (Xu et al., 2025; Ma et al., 2026), or provide guarantees for one-shot policy compliance and embodied decision making (Yang et al., 2025a; Chen et al., 2026; Bhatt et al., 2025; Yang et al., 2025b). However, these guarantees do not jointly account for false alarms and timely failure detection.

VLA-TUG addresses the limitations. We address these limitations by developing a failure detector that is timely, informative, andjointly calibrated. Following the internalrepresentation probing direction, we reuse the VLA’s latent representations and align them with failure-mode semantics, enabling the detector to identify both potential failures and their types. To evaluate timeliness, we measure detection relative to annotated failure onsets and investigate supervision strategies that explicitly account for when failures occur. To provide statistical guarantees, we adopt Learnthen-Test (Angelopoulos et al., 2022) to jointly calibrate multiple detection criteria, including the false-positive rate (FPR) and early true-positive rate $( \mathrm { T P R } _ { \mathrm { e a r l y } } )$ . Existing failure detectors primarily raise alarms without steering, while steering methods rely on predefined schedules or calibrated triggers. In contrast, to the best of our knowledge, VLA-TUG is the first framework to combine joint FPR and early-TPR calibration, failure-onset-aware evaluation, and failure-mode prediction for calibrated failure-triggered steering. Table 1 compares VLA-TUG against the existing works.

Table 1. Comparison with related methods. FPG / TPG: formal calibrated guarantee on the false-positive / true-positive rate. EE: early evaluation, i.e. a per-episode criterion against annotated failure time; ◦ denotes partial use. FM: predicts the failure mode. Trigger: what decides when steering happens. A dash means n/a.
<table><tr><td>Method</td><td>Trigger</td><td>FPG</td><td>TPG</td><td>EE</td><td>FM</td></tr><tr><td>Failure detectors (no steering)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SAFE (Gu et al., 2025)</td><td>one-sided conformal</td><td>V</td><td>×</td><td> $\circ ^ { \dagger }$ </td><td>×</td></tr><tr><td>ActProbe (Huang et al., 2026)</td><td>split conformal</td><td>√</td><td>×</td><td>x</td><td>×</td></tr><tr><td>STAC (Agia et al., 2024)</td><td>one-sided conformal</td><td>√</td><td>X</td><td>×</td><td>X</td></tr><tr><td>FailureSpot (Ma et al., 2026)</td><td>one-sided conformal</td><td>√</td><td>×</td><td>o†</td><td>×</td></tr><tr><td>Steering methods</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VLA-Pilot (Li et al., 2026b)</td><td>always on</td><td></td><td></td><td></td><td></td></tr><tr><td>VLS (Liu et al., 2026a)</td><td>always on⁸</td><td></td><td></td><td></td><td></td></tr><tr><td>DreamSteer (Cui et al., 2026)</td><td>fixed period</td><td></td><td></td><td></td><td></td></tr><tr><td>SAE-VLA (Swann et al., 2026)</td><td>human-specified</td><td></td><td></td><td></td><td></td></tr><tr><td>TLI (Li, 2025)</td><td>human-specified</td><td></td><td></td><td></td><td></td></tr><tr><td>RL2-VLA (Tan et al., 2026)</td><td>one-sided conformal</td><td>V</td><td>×</td><td>×</td><td>×</td></tr><tr><td>LFP (Jeong et al., 2026)</td><td>one-sided conformal</td><td>干</td><td>×</td><td>×</td><td>×</td></tr><tr><td>VLA-TUG (Ours)</td><td>jointly calibrated</td><td></td><td>√</td><td>√</td><td>V</td></tr></table>

† Annotated failure time used only in aggregate or qualitatively. ‡ Timeliness measured via normalized detection time and early-window AUC relative to the episode length; no annotated failure time. § Threshold on VLM-generated rewards. ∓ Class-conditional bound on harmful steering.

## 3 PROBLEM STATEMENT

In this section, we start by giving several broad definitions used to describe how failure detectors for VLAs function. Next, we give a mathematical formulation of our problem and then introduce how VLA-TUG addresses it.

## 3.1 Definitions

To facilitate comparisons across methods, we give general definitions for the concepts of failure score, detection threshold, detection time, failure time, and failure mode.

• Failure score. The score $s _ { t } \in \mathbb { R } .$ , defined at every timestep $t ,$ encodes the probe’s belief that the VLA is failing. It need not be a probability in the strict sense.

• Detection threshold. The threshold $h ^ { \star }$ over $s _ { t }$ separates failures from successes.

• Detection time. The time $t _ { \mathrm { d e t } } = \operatorname* { m i n } \{ t : s _ { t } \geq h ^ { \star } \}$ i

the first time the score crosses the threshold.

• Failure time. The time $t _ { \mathrm { f a i l } }$ , which we also call the failure onset, is when the policy makes an obvious mistake or its progress stalls to the point where an intervention, human or otherwise, is needed.

• Failure mode. The label $f _ { \mathrm { m o d e } }$ is a high-level semantic category assigned to each failure, such as grasping failure, placement failure, or no progress.

These definitions are general, and their exact form varies slightly across methods. The interpretation of $s _ { t }$ depends on the training strategy, but a higher score usually signals failure and a lower score success. The threshold $h ^ { \star }$ may be fixed or vary with time, and it is often calibrated with probabilistic guarantees on a set of calibration rollouts $\mathcal { D } _ { \mathrm { c a l } }$ If $s _ { t } < h ^ { \star }$ for all t, the detection time $t _ { \mathrm { d e t } }$ is undefined, and the rollout raises no alarm. The failure time definition follows (Gu et al., 2025).

## 3.2 Problem Formulation

Consider a VLA policy $\mathbf { f } _ { \mathrm { V L A } } ( \cdot )$ and a rollout $\tau _ { i }$ of $T$ policy steps. At each policy step $t \in \{ 1 , \dots , T \} , \mathbf { f } _ { \mathrm { V L A } } ( \cdot )$ takes as input RGB observations $\mathbf { o } _ { t }$ (agent and wrist cameras) and a natural language prompt p, and produces a chunk of m control actions $\mathbf { a } _ { t ; 0 : m - 1 }$ . We denote the VLA’s internal embedding at step t as $\mathbf { e } _ { t } .$ , so that $\tau _ { i } = \{ ( \mathbf { p } , \mathbf { o } _ { t } , \mathbf { e } _ { t } , \mathbf { a } _ { t ; 0 : m - 1 } ) \} _ { t = 1 } ^ { T } .$ Each rollout has an episode-level label $Y _ { i } \in \{ 0 , 1 \}$ , where $Y _ { i } = 1$ marks a failure and $Y _ { i } = 0$ a success. If $Y _ { i } = 1$ the rollout also has a failure time $t _ { \mathrm { f a i l } }$ and a failure mode $f _ { \mathrm { m o d e } } \in \mathcal { F }$ , where $\mathcal { F }$ is a fixed taxonomy of failure modes.

Problem. Given a training set and a calibration set $\mathcal { D } _ { \mathrm { c a l } } =$ $\{ \tau _ { i } \}$ , and access to the internal embeddings $\mathbf { e } _ { t }$ of $\mathbf { f } _ { \mathrm { V L A } } ( \cdot )$ predict an imminent task failure before it happens, i.e. $t _ { \mathrm { d e t } } \le t _ { \mathrm { f a i l } }$ where $t _ { \mathrm { d e t } }$ is the detection time, and predict the failure mode $f _ { \mathrm { m o d e } }$ in a manner that holds up empirically on tasks unseen during calibration.

We propose VLA-TUG, a probe that is trained and calibrated to detect failures before they happen. It is:

(T.) Timely. We propose a training and evaluation strategy that accounts for the failure time and improves early failure detection.

(C.) Calibrated. We introduce a risk formulation that jointly bounds the false-positive rate and the early miss rate $1 - \mathrm { T P R } _ { \mathrm { e a r l y } }$ , and examine how well these targets hold on unseen tasks.

(I.) Informative. We design VLA-TUG to predict the failure mode, giving the semantic information needed to choose a corrective action.

(E.) Efficient. VLA-TUG adds minimal latency and parameters to the VLA, and is cheap to train and calibrate.

## 4 METHODOLOGY

In this section we introduce VLA-TUG. In subsection 4.1 we present the probe and how we train it, and in subsection 4.2 we show how we calibrate its alarm threshold and what guarantees this gives.

## 4.1 Failure Detection Through Latent Space Probing

Building on existing works (Gu et al., 2025), we instantiate the probe as a combination of LSTMs and MLPs, the goal being to capture a non-linear pattern in the backbone embeddings that generalizes beyond the seen episodes. We briefly introduce the model inputs, followed by a discussion of the model architecture and finally the training strategy.

We use VLA backbone embeddings as the inputs. We explore two distinct layers: backbone layer 18 (after normalization) in $\pi _ { 0 . 5 }$ and layer 16 in GR00T-N1.7, focusing on the latter in particular. At every policy step t we average the backbone tokens of each modality (agent camera, wrist camera, instructions) and concatenate the three 2048- dimensional vectors into a single embedding $e _ { t } \in \mathbb { R } ^ { 6 1 4 4 }$

We leverage a combination of LSTMs and MLPs. A single-layer LSTM with 256 hidden units reads the sequence of embeddings and produces a latent state $\boldsymbol { u } _ { t } = \mathrm { L S T M } ( \boldsymbol { e } _ { 1 : t } )$ at every policy step. Three MLP heads read this shared latent state (see Figure 1). The failure score head predicts the failure score, $s _ { t } = \sigma ( g _ { s } ( u _ { t } ) )$ ). To make the probe informative, the mode head predicts a distribution over the 10 failure modes $f _ { \mathrm { m o d e } } , m _ { t } = \mathrm { s o f t m a x } ( g _ { m } ( u _ { t } ) )$ . Following (Xu et al., 2024), an auxiliary head, $c _ { t } = g _ { c } ( u _ { t } )$ , aligns the latent state with a CLIP embedding of the description of the current action. Each head g is an MLP with one hidden layer of 64 ReLU units. The probe has 6.6M parameters.

We train the probe to predict features that precede a failure. We hypothesize that training on post-failure embeddings teaches the probe what already failed trajectories look like, not what trajectories about to fail look like. If so, to predict failures the probe must be trained on the embeddings $e _ { t }$ before the failure time $t _ { \mathrm { f a i l } }$ (pre-onset supervision), rather than after it (Ma et al., 2026) or at all times of failed rollouts (Gu et al., 2025). To test this, we compare four ways of labeling the embeddings of failed rollouts (Figure 2; the mathematical formulation is in App. A.2). Successful rollouts are labeled 0 in all four. The window w in onset and post-onset is the time span before $t _ { \mathrm { f a i l } }$ that is labeled as failure. Pre-onset and onset truncate the rollout at the failure time, while post-onset and trajectory keep all steps. We exclude the steps after $t _ { \mathrm { f a i l } }$ in the first two strategies instead of labeling them 0: these embeddings already show a failed rollout, so labeling them as successes would give the probe contradicting training signals.

![](images/f04283490e0d8bfdf4d81b91282475ef3845000582507e0f95766a7128824b0f.jpg)  
Failure (y<sub>t</sub>=1) No failure (y<sub>t</sub>=0) Excluded from the loss  
Figure 2. How the embeddings $e _ { t }$ of a failed rollout are labelled for the four supervision strategies.

We optimize three objectives. With the labels above, the probe is trained with

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { s c o r e } } + \lambda _ { m } \mathcal { L } _ { \mathrm { m o d e } } + \lambda _ { c } \mathcal { L } _ { \mathrm { C L I P } } , } \end{array}\tag{1}
$$

with $\lambda _ { m } = \lambda _ { c } = 1 . { \mathcal { L } } _ { \mathrm { s c o r e } }$ is a binary cross-entropy over all steps that are not excluded from the loss, weighted by the ratio of negative to positive steps in the training set. ${ \mathcal { L } } _ { \mathrm { m o d e } }$ is a cross-entropy against the true failure mode, applied to the positive steps of failed rollouts and weighted to balance the modes. ${ \mathcal { L } } _ { \mathrm { C L I P } }$ is an auxiliary contrastive loss that aligns $c _ { t }$ with the CLIP embedding of the step’s description. We train with Adam (learning rate $3 \times 1 0 ^ { - 4 }$ , batch size 32 rollouts, at most 100 epochs) on standardized inputs, with early stopping on the validation $\mathcal { L } _ { \mathrm { s c o r e } }$ (patience 15).

## 4.2 Uncertainty Quantification and Guarantees

Where previous works only provided guarantees of falsepositive rates (i.e. the rate of successful rollouts that were flagged as failures) (Gu et al., 2025; Rajaprakash et al., 2026), they did not provide guarantees on the false-negative rate (i.e. the rate of unsuccessful trajectories that were not flagged). We will first formulate the risks and show how to calibrate the detection threshold $h ^ { \star }$ . In an effort to make these bounds interpretable to the practitioner, we compute the probability of failure given that a rollout was flagged.

VLA-TUG guarantees both FPR and $\mathrm { T P R } _ { \mathrm { e a r l y } }$ . In general, the FPR is an operational metric that we wish to minimize to prevent unnecessary steering, the TPR is a safety metric that we wish to maximize to prevent undetected failures. We can control these risks using the Learn-then-Test framework (Angelopoulos et al., 2022). Let $h ^ { \star }$ be a constant detection threshold, and $s _ { t }$ the failure score produced by our probe at each timestep t for each rollout $\tau .$ . Furthermore, let the detection time $t _ { \mathrm { d e t } }$ be the first time the failure score $s _ { t }$ exceeds the threshold $h ^ { \star }$ . Finally, the ground truth failure time $t _ { \mathrm { f a i l } }$ remains the first time t when failure becomes apparent or a human operator would choose to take over. We propose to formulate the above mentioned risks as follows:

$$
\mathrm { F P R } ( h ) = \mathbb { E } _ { \tau \sim \mathcal { D } _ { \mathrm { c a l } } ^ { \mathrm { s u c c } } } [ { \mathbf { 1 } \big \{ \exists t _ { \mathrm { d e t } } : s _ { t } ( \tau ) \ge h \big \} } ] ,\tag{2}
$$

$$
\mathrm { T P R } _ { \mathrm { e a r l y } } ( h ) = \mathbb { E } _ { \tau \sim \mathcal { D } _ { \mathrm { c a l } } ^ { \mathrm { f a i l } } } [ { \mathbf 1 } \{ \exists t \mathrm { d e t } \leq t _ { \mathrm { f a i l } } : s _ { t } ( \tau ) \geq h \} ] ,\tag{3}
$$

where $\mathcal { D } _ { \mathrm { c a l } } ^ { \mathrm { s u c c } }$ and $\mathcal { D } _ { \mathrm { c a l } } ^ { \mathrm { f a i l } }$ are the sets of successful and failed rollouts $\tau ,$ respectively, in the calibration set $\mathcal { D } _ { \mathrm { c a l } }$ . We are interested in finding the set Λ<sup>ˆ</sup> of values $h ^ { \star }$ that satisfy:

$$
\begin{array} { r } { P \Big ( \mathrm { F P R } ( h ) \leq \alpha , \mathrm { ~ T P R } _ { \mathrm { e a r l y } } ( h ) \geq \beta ~ \forall h \in \hat { \Lambda } \Big ) \geq 1 - \delta , } \end{array}\tag{4}
$$

where $\alpha \in [ 0 , 1 ]$ and $\beta \in \ [ 0 , 1 ]$ are user-selected rates. This guarantee can be understood as saying that any $h \in { \hat { \Lambda } }$ guarantees that the false-positive rate (FPR) is at most α and the early true-positive rate $( \mathrm { T P R } _ { \mathrm { e a r l y } } )$ is at least $\beta ,$ with high-probability of no less than $1 - \delta$ . We will see, in the experimental section, that this translates to guarantees that empirically behave better under distribution shift. Following (Angelopoulos et al., 2022) we select a set of candidate values $\hat { \Lambda } ;$ from this set Λ<sup>ˆ</sup> of values h we select the value $h ^ { \star }$ that maximizes $\mathrm { T P R } _ { \mathrm { e a r l y } }$

Probability of failure given that an alarm is raised. Since our goal is a probe that can be deployed in real-world systems, we complement the formal $\mathrm { T P R } _ { \mathrm { e a r l y } }$ and FPR guarantees with an actionable interpretation: a bound on the positive predictive value (PPV). The PPV is the probability that failure occurs given score $s _ { t }$ exceeds the threshold $h ^ { \star }$ Existing calibration methods (Gu et al., 2025) cannot give such a bound, since they lack guarantees on the TPR.

Let FAILURE denote rollout-level failure, and let EXCEED denote the event that the score $s _ { t }$ exceeds the threshold $h ^ { \star }$ at some time t. The PPV is P(FAILURE | EXCEED). Define the failure prevalence $\pi = P ( { \mathrm { F A I L U R E } } )$ , the true positive rate TPR = P(EXCEED | FAILURE), and the false positive rate FPR = P(EXCEED | NOT FAILURE). By Bayes’ rule,

$$
P ( { \mathrm { F A I L U R E } } \mid { \mathrm { E x C E E D } } ) = { \frac { \pi \cdot { \mathrm { T P R } } } { \pi \cdot { \mathrm { T P R } } + ( 1 - \pi ) \cdot { \mathrm { F P R } } } } .\tag{5}
$$

Detections that occur before the failure onset are a subset of all detections, so $\mathrm { T P R } \geq \mathrm { T P R } _ { \mathrm { e a r l y } }$ Our guarantees give $\mathrm { T P R } _ { \mathrm { e a r l y } } \geq \beta$ and $\mathrm { F P R } \le \alpha$ . Since the PPV is nondecreasing in TPR and non-increasing in FPR, substituting the worst-case values yields the conservative bound:

$$
P ( { \mathrm { F A I L U R E } } \mid { \mathrm { E X C E E D } } ) \ \geq \ { \frac { \pi \cdot \beta } { \pi \cdot \beta + ( 1 - \pi ) \cdot \alpha } } .\tag{6}
$$

The bound increases with $\pi ,$ so a lower bound on the prevalence suffices. Given an estimate of $\pi ,$ , this gives practitioners an intuitive reading of the guarantees.

## 5 DATASET

To evaluate VLA-TUG, we create a labeled dataset of rollouts of two VLA policies, $\pi _ { 0 . 5 }$ (Physical Intelligence et al., 2025) and GR00T-N1.7 (NVIDIA et al., 2025), with failure labels and semantic descriptions of the robot’s actions. We first describe the benchmarks and the splits, and then the labeling pipeline.

![](images/babbd91f41059a2ac8e8862d4022911a8407c8dbb25e9571d0a742ec8cf9c9aa.jpg)  
Figure 3. A successful (top) and a failed (bottom) rollout with their timeline,failure time, andfailure mode labels.

Benchmarks and splits. We use LIBERO (Liu et al., 2023), LIBERO-Occ (Li et al., 2026a), and LIBERO-X (Wang et al., 2026), as well as real-robot tasks. Both policies succeed on most LIBERO episodes, so we combine it with LIBERO-Occ to obtain enough failures. We propose splits of increasing distribution shift to evaluate generalization in a principled way; Table 2 gives how the tasks are held out and their counts. In LIBERO-X, each task has a skill, the manipulation behavior (such as pushing a button or stacking objects), and a signature, the combination of its skill, target class, relation, landmark, and horizon. The splits are:

NEW-TASK holds out individual tasks; their skill and signature can still occur in the training tasks, the shift is mild.

NEW-SIGNATURE holds out all tasks with a given signature: the components are familiar, but their combination is new.

NEW-SKILL holds out a complete skill, so the tested behavior is absent from training: this is the largest shift of the three.

Efficient labeling pipeline. We label each rollout from its video with Gemini (gemini-3.5-flash), as in Figure 3. Every rollout receives a timeline label, and failed rollouts also receive a failure mode label and a failure time label. The timeline is a short keyword phrase per 3-second window that we use for CLIP alignment. The failure mode $f _ { \mathrm { m o d e } }$ comes from a 10-way taxonomy F. The failure time $t _ { \mathrm { f a i l } }$ is the earliest point at which the mistake responsible for the failure becomes apparent or, following (Gu et al., 2025), the time a human operator would take over if none is apparent.

Table 2. Task count by dataset and split.
<table><tr><td>Dataset</td><td>Split</td><td>Seen tasks</td><td>Unseen tasks</td></tr><tr><td>LIBERO</td><td>leave-three-out</td><td>7</td><td>3</td></tr><tr><td rowspan="3">LIBERO-X</td><td> $\mathrm { N E W  – T A S K }$ </td><td>175</td><td>43</td></tr><tr><td> $\mathrm { N E W - S I G N A T U R E }$ </td><td>162</td><td>56</td></tr><tr><td> $\mathrm { N E W  – S K I L L } ^ { \star }$ </td><td>159-209</td><td>9-59</td></tr><tr><td>Real-world</td><td>leave-one-out</td><td>7</td><td>1</td></tr></table>

⋆ There are 8 skills, each with a different seen/unseen split.

We open-source this dataset including over 288,000 embeddings from 3,180 labelled rollouts at vla-tug.github.io.

## 6 EXPERIMENTAL RESULTS

In this section, we answer five questions based on our problem formulation (section 3): (Q1. Timeliness) Does training on features that precede the failure onset improve early failure detection on unseen tasks (subsection 6.1)? (Q2. Calibration) Can VLA-TUG jointly bound the FPR and $\mathrm { T P R } _ { \mathrm { e a r l y } }$ , with bounds that hold on seen tasks and degrade gracefully on unseen tasks (subsection 6.2)? (Q3. Informativeness) Can VLA-TUG predict the failure mode based on VLA backbone embeddings (subsection 6.3)? (Q4. Efficiency) Are the cost and time of labeling, training, and inference appropriate for deployment in real-world systems (subsection 6.4)? (Q5. Steering) Can VLA-TUG, as a timely, informative, calibrated, and efficient probe, support VLA steering (subsection 6.5)? We ablate our design choices in App. D.4.

## 6.1 Failure Detection and Timeliness Evaluation

We evaluate VLA-TUG against a series of baselines using metrics that depend on the ground-truth failure time. Firstly, we describe how we split the data. Secondly, we review the baselines. Thirdly, we formally define our evaluation metrics. Lastly, we discuss the results.

Data split. For every run, we split the episodes as in Figure 4. The training set fits the probe, the validation set selects the checkpoint, and the threshold is chosen on the calibration set. The calibration and held-out seen sets are exchangeable, and our guarantees apply to the held-out seen set. The unseen set, which the probe and the threshold never see, measures robustness to distribution shift. On LIBERO-X, the held-out tasks of the NEW-TASK and NEW-SIGNATURE splits are the same for every seed, because they are defined by the semantic axis described in section 5; the seed changes only the shuffle of the seen episodes and the initialization. The NEW-SKILL folds and the LIBERO-10 seeds hold out different tasks.

Table 3. Early failure detection on seen and unseen tasks: area under the TPR early vs. FPR curve, averaged over splits. Seen: held-out episodes of training tasks; unseen: held-out tasks. Cal.: calibrator (MC max-conformal, FC functional conformal). Cost on LIBERO-X (lower is better): fit time, memory to score a new episode, and per-step latency on a GPU and a CPU. Dashes: not available. Best in bold, second best in italics.
<table><tr><td></td><td>Cal.</td><td colspan="2">LIBERO-10  $\pi _ { 0 . 5 }$ </td><td colspan="2">LIBERO-10 GR0OT</td><td colspan="2">LIBERO-X task</td><td colspan="2">LIBERO-X signature</td><td colspan="2">LIBERO-X skill</td><td colspan="2">Real robot</td><td colspan="4">Cost (LIBERO-X)</td></tr><tr><td>Method</td><td></td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td><td>Fit (s) State (MB)</td><td>GPU (ms)</td><td>CPU (ms)</td></tr><tr><td>SAFE-LSTM</td><td>FC</td><td>75.5</td><td>34.9</td><td>74.1</td><td>51.2</td><td>64.6</td><td>40.4</td><td>64.7 53.7</td><td>41.2</td><td>62.5</td><td>30.1 44.5</td><td></td><td>25.1</td><td>43.5</td><td>26.2 6.3</td><td>0.066 0.86</td></tr><tr><td>SAFE-MLP</td><td>FC</td><td>74.1</td><td>39.9</td><td>62.9</td><td>39.2</td><td>59.3</td><td>42.3</td><td></td><td>46.2</td><td>56.8</td><td>23.5 40.1</td><td>20.2</td><td>114.5</td><td></td><td>0.034 0.0</td><td>0.032</td></tr><tr><td>STAC</td><td>MC</td><td>33.8</td><td>36.4</td><td>24.2</td><td>17.5</td><td>21.0</td><td>28.0 25.6</td><td></td><td>28.7 24.6</td><td></td><td>21.5</td><td></td><td>0.0</td><td></td><td>21.30</td><td></td></tr><tr><td>ActProbe</td><td>MC</td><td>39.3</td><td>26.7</td><td>32.4</td><td>30.1</td><td>12.6</td><td>22.7 11.9</td><td></td><td>13.9 20.3</td><td>13.6</td><td>=</td><td>=</td><td>7.1</td><td>0.09</td><td>0.15</td><td>0.11</td></tr><tr><td>FailureSpot</td><td>FC</td><td>61.9</td><td>36.2</td><td>36.9</td><td>44.7</td><td>26.3</td><td>20.4 28.0</td><td></td><td>26.2 30.4</td><td>19.6</td><td></td><td></td><td>9.2</td><td>6.4</td><td>0.033</td><td>0.029</td></tr><tr><td>Mahalanobis</td><td>FC</td><td>85.7</td><td>40.2</td><td>70.7</td><td>29.2</td><td>69.6</td><td>30.6 64.4 65.4</td><td></td><td>22.3 71.2</td><td>15.9</td><td>43.0</td><td>19.6</td><td>15.2</td><td>302.0</td><td>0.41</td><td>5.37</td></tr><tr><td>Euclidean</td><td>FC</td><td>73.1 35.7</td><td>24.2</td><td>56.0</td><td>23.1</td><td>70.5</td><td>28.5</td><td></td><td>19.7 68.5</td><td></td><td>19.0 45.1</td><td>8.3</td><td>0.4</td><td>3056</td><td>13.19</td><td>368.83</td></tr><tr><td>PCA-KMeans</td><td>FC</td><td></td><td>12.9</td><td>44.5</td><td>20.0</td><td>31.7</td><td>25.0 35.3</td><td></td><td>32.5 37.8</td><td></td><td>20.7 24.0</td><td>19.4</td><td>39.9</td><td>1.6</td><td>0.12</td><td>0.055</td></tr><tr><td>VLA-TUG (Ours)</td><td>MC</td><td>75.1</td><td>47.9</td><td>70.5</td><td>47.4</td><td>67.9</td><td>40.2 67.0</td><td></td><td>47.0 67.8</td><td></td><td>39.0 47.6</td><td>25.8</td><td>6.8</td><td>26.5</td><td>0.12</td><td>0.94</td></tr></table>

![](images/65aea3b72d7e8a20baeb5a8a8475cacb0ecf9dcba26c444ead772d38412851c9.jpg)  
Figure 4. Task- and episode-level data splits. Episodes from seen tasks are divided into training (65%), validation (15%), calibration (10%), and held-out seen (10%) sets. Unseen tasks form a separate evaluation set. Task-level proportions are schematic.

## Baselines. We consider the following baselines:

Embedding distance (Mahalanobis, Euclidean, PCA-KMeans). A step is scored by how much closer its backbone embedding is to failed than to successful training steps.

Action consistency (STAC, ActProbe, FailureSpot). Steps are scored based on the discrepancy between predicted and executed steps.

Latent (SAFE). A step is scored by a learned projection of its backbone embedding.

Timeliness Metrics. We evaluate VLA-TUG against three key metrics: TPR, $\mathrm { T P R } _ { \mathrm { e a r l y } }$ , and FPR. Existing works often evaluate their methods against AUROC (Gu et al., 2025; Huang et al., 2026; Ma et al., 2026) without accounting for the true failure times, making it hard to distinguish timely detections. The TPR and FPR follow their standard definitions. The $\mathrm { T P R } _ { \mathrm { e a r l y } }$ is the rate of failures detected before the failure time, i.e. $t _ { \mathrm { d e t } } \le t _ { \mathrm { f a i l } }$ . We additionally define $\mathrm { A U R O C _ { \mathrm { e a r l y } } }$ as the area under the $\mathrm { T P R } _ { \mathrm { e a r l y } } – \mathrm { F P R }$ curve. We compute it on the support $\mathrm { F P R } \in [ 0 , 0 . 4 ]$ , normalized by its width 0.4 (reported in %), since a detector that raises false alarms on more than 40% of successful rollouts is im-

practical to deploy. We formally define these metrics in App.   
C.1.

Results and Discussion. We ask whether training on features that precede the failure onset improves early failure detection on unseen tasks. Our hypothesis is that it does: post-failure features only describe failures the probe has already seen, so training on them keeps it from learning a general notion of what leads to a failure, while features before the onset carry the cues that come before one. We compare VLA-TUG against the baselines in Table 3 and Figure 5. On unseen tasks, our domain of interest, VLA-TUG is the best method in four of six settings: LIBERO-10 with π<sub>0.5</sub>, NEW-SIGNATURE, NEW-SKILL, and the real robot. Against SAFE-LSTM, the strongest learned baseline, its margin grows from −1% on NEW-TASK to +14% on NEW-SIGNATURE and +30% on NEW-SKILL. On the real robot, the lead is small, and VLA-TUG does not lead on LIBERO-10 with GR00T. On seen tasks, the results are mixed. We ablate the label in App. D.4: replacing the pre-onset label with the other variants lowers early detection on LIBERO-X.

Answer to Q1. Timeliness. Yes: for most unseen settings. Training on features that precede the failure onset lets VLA-TUG raise the alarm at or before the failure onset more often than the baselines, and on LIBERO-X the gap widens as the test tasks move away from the training tasks.

## 6.2 Calibration Evaluation

We validate the theoretical results from section 4 on seen tasks and examine how they hold under distribution shift on unseen tasks. We calibrate our probe with Learn-then-Test (LTT, $\delta = 0 . 1 )$ at seven false-positive-rate (FPR) targets (10 to 50%), each certified together with a TPR early target, on the 14 LIBERO-X runs (3 NEW-TASK seeds, 3 NEW-SIGNATURE seeds, and 8 NEW-SKILL folds). We compare against max conformal (MC) (Agia et al., 2024) and functional conformal (FC) (Gu et al., 2025) calibration (the complete table is in App. D.1). We draw three conclusions:

![](images/34fb88d49dbb713f9ddc6127f59364bc52232c98bd390ce27a406368a981b37e.jpg)  
Figure 5. Early detection on seen (top) and unseen (bottom) tasks: the share of failures detected at or before the failure time $( \mathrm { T P R } _ { \mathrm { e a r l y } } )$ against the false-positive rate, for VLA-TUG and the learned and action-based baselines. Each curve is the mean over the splits of a setting, and the area under it up to 40% FPR is the metric of Table 3. Panel titles give the benchmark, the policy, and the held-out split.

• LTT bounds the $\mathrm { T P R } _ { \mathrm { e a r l y } }$ together with the FPR, which hold in all certified runs on seen tasks and in 40–86% (FPR) and 57–90% (TPR early) of runs on unseen tasks; baselines give no guarantee on the TPR.

• LTT states upfront whether the system can support the required guarantee whereas existing methods can give a false sense of safety. LTT abstains on 4 to 7 of the 14 runs, depending on the target (FPR targets of 15– 50%), which tells us before deployment that the model cannot support the requested rates. MC always returns a threshold, but it meets the FPR target in only 57–71% of runs on seen tasks and in at most 21% on unseen tasks.

• Our LTT targets give a lower bound on the positive predictive value (PPV), the probability that an alarm is a true failure. It is 58–71% across the FPR targets of the full table and holds in every certified run on seen tasks and in 70–100% of runs on unseen tasks. Methods without a TPR guarantee cannot give it, and it tells the practitioner what an alarm means.

On unseen tasks, the guarantees hold only empirically, since new samples are not exchangeable with the calibration set, which applies to all three methods. The shares above are trends over 14 runs, not precise rates.

Answer to Q2. Calibration. Yes: the rates we calibrate for are the rates we get on seen tasks. Under distribution shift, they hold in far more runs than with MC and FC, and when a target cannot be supported, LTT abstains instead of silently missing it. Together with the PPV bound, this makes VLA-TUG easier to integrate into real-world systems.

## 6.3 Evaluation of Informativeness

Here we focus on failure mode prediction, a crucial element to enable failure prevention through steering. Failure modes are strongly imbalanced, so we report balanced accuracy, the mean over classes of the share of their failures predicted correctly. Since existing baselines do not predict failure modes, we compare against a majority baseline and focus on LIBERO-X.

On LIBERO-X, the probe improves top-1 accuracy over the majority baseline by up to 149% on seen and 79% on unseen tasks, and top-3 accuracy by a smaller but consistent margin on both. On the real robot, it improves top-1 by 171% and 8% on seen and unseen tasks, respectively, and stays well above the baseline in top-3 on both.

The practical consequence is that the probe narrows a failure down to three candidate classes, which are more likely than the majority baseline to include the true one. Different failure modes carry different risks, so a practitioner can use these candidates to select the level of response, for example responding to the riskiest of the three predicted classes (a stop, a retry, or a handover to a human) instead of applying the same intervention to every failure.

Answer to Q3. Informativeness. Yes: backbone embeddings carry enough information to predict the failure class above a majority baseline, which lets a practitioner match the response to the predicted risk.

## 6.4 Practical Considerations

Any real-world system must balance performance ((T.) Timeliness, (C.) Calibration, (I.) Informativeness) with the cost of

![](images/a5f38cc3c3bb2c97d58e039fbc46583dc1418a1a43327e5865b11154ce0bd141.jpg)

(a) Example steering for the simulated task “place the largest bowl inside the left compartment of the bowl drainer.” The gripper drops the bowl into the right compartment with no steering and successfully drops it in the left compartment when steered towards the target location.<sub>c</sub>o<sup>r</sup>  
![](images/060d8d2c334e056cd5a95853a136f5d1fd2952072b0f4ea0e8ca40c28c662c6e.jpg)  
(b) Example steering for the real-world task “put the mushroom in the plate.” The gripper misses the mushroom without steering and successfully grasps it when steered towards the mushroom.

Figure 6. Steering demonstrations showing failure scores and corrections for simulated and real-world tasks. When the probe’s predicted failure score rises above the threshold, the VLA policy is automatically paused. For the simulated bowl task, we find that nudging the end-effector toward the drop-off location can help lead to success. For the real-world mushroom task, we find that nudging the wris camera towards the target grasp object can help. Future work could systematically explore steering strategies. Refer to Figure 17a fo more steering strategies and Figure 17b for another real world steering example.

building and running it ((E.) Efficiency). We first evaluate the cost of labeling the dataset, then of training the probe, calibrating it, and running it at inference time.

Dataset. Labeling costs about one cent per failed episode (under \$50 for all annotations, including queries not used in this paper). Labels are therefore cheap, and practitioners will often already have unlabeled rollouts from system deployment.

Training. VLA-TUG trains in 6.8 s on one RTX 4090, against 43.5 s for SAFE-LSTM, and holds 6.6M parameters (26.5 MB), about 0.2% of the VLA. It is therefore cheap enough to retrain whenever the policy or the task set changes, or to train online as new rollouts are produced.

Calibration. Calibrating takes under 2 ms per target pair on a CPU, so VLA-TUG can be re-calibrated if system requirements change, or online along with the probe.

Inference. The VLA outputs a chunk of m actions per call (8 at 20 Hz on LIBERO-X), so a policy step lasts 400 ms, and the probe reads the embedding of each call. It adds 0.12 ms per step on the GPU (0.2% of the 54 ms GR00T policy call) and 0.9 ms on four CPU threads, so its score is available before the chunk is executed and its latency is not observable: the step period is about 7× the combined VLA and probe time in simulation and about 2× on the real robot on our GPU.

Faster baselines offer diminishing benefit, since the policy

frequency bounds how fast a detector needs to be, while slower ones such as k-NN and STAC add noticeable latency to every step. VLA-TUG costs about the same as the cheapest learned baselines and far less than k-NN and STAC, while providing improved informativeness and timeliness on most unseen settings, which makes it a practical option for real-world deployment.

Answer to Q4. Efficiency. Yes: labeling, training, calibrating, and running the probe introduce minimal cost, and on a GPU the probe adds under 1% to the policy call and fits within the step period.

## 6.5 Steering Demonstration

We demonstrate how the four properties of VLA-TUG: (T.) timeliness, (C.) calibration, (I.) informativeness, and (E.) efficiency, can jointly support failure prevention through VLA steering. Our goal is to illustrate how an effective failure predictor can provide the signals needed to guide interventions rather than develop a steering method itself. We consider a steering architecture in which an expert model (e.g. a VLM) can invoke failure-recovery tools in response to the probe’s predictions. For our experiments, we implement these tools using failure-mode-specific heuristics in simulation and demonstrate corresponding interventions on a real robot.

For example, in the simulated rollout shown in Figure 6a, the task is “place the largest bowl inside the left compartment of the bowl drainer.” The probe raises an alarm before the bowl is mistakenly placed in the compartment on the right. After steering the end-effector towards the left side, the policy is resumed and the rollout succeeds. Without intervention, the bowl would be placed on the right side and the rollout fails.

Our real-world experiments similarly demonstrate that the joint properties of VLA-TUG help in steering a VLA away from failure. Figure 6b illustrates this with the “put the mushroom on the plate” task. Without VLA-TUG, the gripper misses the mushroom and hits the table. With it turned on, however, the probe predicts the imminent grasp failure (which would likely happen), and corrective action is taken. In the example, the corrective action is to pause the VLA, let an expert model shift trajectory to a successful one, and resume the VLA.

Such a strategy is possible with a failure detector like VLA-TUG, which anticipates the failure before it happens, rather than after it is too late. Our approach is agnostic towards the steering architecture. We leave exploration of robust steering strategies for future work. Nevertheless, these examples demonstrate the applicability of VLA-TUG to be utilized in safe and principled robotic systems.

Answer to Q5. Steering. Yes: Our proof-of-concept demonstrations show how VLA-TUG can provide the timing and failure mode information needed for targeted interventions during VLA execution. Timeliness creates an opportunity to intervene, calibration controls when interventions are triggered, informativeness guides which intervention is applied, and efficiency allows the probe to operate alongside the VLA.

## 7 CONCLUSION

We introduced VLA-TUG, a lightweight probe on the latent layers of a VLA that is timely, calibrated, informative, and efficient, making it well suited for real-world deployment. It is timely: supervising on the pre-failure part of the trajectory drives early detection, and the probe’s performance compared to the strongest learned baseline increases as the shift from the training tasks increases. It is calibrated: calibrating the FPR and the $\mathrm { T P R } _ { \mathrm { e a r l y } }$ jointly provides guarantees that hold in far more runs than the baselines under shift. It is informative: beyond a failure score, it names the likely failure mode. It is efficient: the probe adds under 1% to the cost of a policy call. Our steering demonstrations, in simulation and on a real robot, show that VLA-TUG can inform targeted interventions, and we release 3,180 annotated episodes to support further work on latent failure detection.

Limitations. First, our calibration guarantees rely on exchangeability between the calibration and test episodes, which weakens as the distribution shift grows. Empirically, we observe that the degradation is gradual. Second, we do not analyze the choice of source backbone embedding systematically. Third, the steering component, in which an expert model intervenes on a predicted failure, is heuristic and not evaluated in the scope of this work.

Future Work. Looking forward, first, we will adapt the probe and its calibration guarantees online, which our preliminary experiments suggest is computationally feasible. Second, we will maintain and extend our embedding dataset with additional backbones and action-expert layers to determine which layers contain the most generalizable failure prediction signal. Third, we will explore agentic frameworks as the expert model in our setup, combining fast and local inference of a VLA with general-purpose intervention and steering from a cloud-based LLM, triggered by a principled detector such as VLA-TUG.

## REFERENCES

Agia, C., Sinha, R., Yang, J., Cao, Z.-a., Antonova, R., Pavone, M., and Bohg, J. Unpacking failure modes of generative policies: Runtime monitoring of consistency and progress. arXiv preprint arXiv:2410.04640, 2024.

Angelopoulos, A. N., Bates, S., Candes, E. J., Jordan,\` M. I., and Lei, L. Learn then Test: Calibrating predictive algorithms to achieve risk control. arXiv preprint arXiv:2110.01052, 2022.

Bhatt, N. P., Yang, Y., Siva, R., Milan, D., Topcu, U., and Wang, Z. Know where you’re uncertain when planning with multimodal foundation models: A formal framework. arXiv preprint arXiv:2411.01639, 2025.

Buurmeijer, H., Alonso, C. A., Swann, A., and Pavone, M. Observing and controlling features in vision-languageaction models. arXiv preprint arXiv:2603.05487, 2026.

Chen, J. AEGIS: A backup reflex for physical AI. arXiv preprint arXiv:2606.06660, 2026.

Chen, L., Lyu, Z., and Beksi, W. J. ReconVLA: An uncertainty-guided and failure-aware vision-languageaction framework for robotic control. arXiv preprint arXiv:2604.16677, 2026.

Cui, H., Arnaud, S., Majumdar, A., Dugas, D., Aljalbout, E., Desingh, K., Jatavallabhula, K. M., and Meier, F. DREAMSTEER: Latent world models can steer VLA policies during deployment without any finetuning. arXiv preprint arXiv:2607.02865, 2026.

Fei, S., Wang, S., Shi, J., Dai, Z., Cai, J., Qian, P., Ji, L., He, X., Zhang, S., Fei, Z., Fu, J., Gong, J., and Qiu, X. LIBERO-Plus: In-depth robustness analysis of vision-language-action models. arXiv preprint arXiv:2510.13626, 2025.

Francis-Meretzki, S., Mutti, M., Romano, Y., and Tamar, A. Temporal difference calibration in sequential tasks: Application to vision-language-action models. arXiv preprint arXiv:2604.20472, 2026.

Gu, Q., Ju, Y., Sun, S., Gilitschenski, I., Nishimura, H., Itkina, M., and Shkurti, F. SAFE: Multitask failure detection for vision-language-action models. arXiv preprint arXiv:2506.09937, 2025.

Huang, B., Li, X., Wang, X., Mi, L., Hao, Z., Wang, W., Wu, H., Li, K., Liu, Y., and Cao, T. ActProbe: Actionspace probe for early failure detection of generative robot policies. arXiv preprint arXiv:2606.08508, 2026.

Jeong, H. J., Swamy, G., and Bajcsy, A. Learning what to say to your VLA: Mostly harmless vision language action model steering. arXiv preprint arXiv:2606.12299, 2026.

Li, Q. VLAs are confined yet capable of generalizing to novel instructions. arXiv preprint arXiv:2505.03500, 2025.

Li, T., Zhang, J., Wang, S., Huang, X., and Wei, Z. LIBERO-Occ: Evaluating and improving vision-language-action models under scene-induced occlusion via viewpoint imagination. arXiv preprint arXiv:2606.10862, 2026a.

Li, Z., Liu, J., Dong, Z., Teng, T., Rouxel, Q., Caldwell, D., and Chen, F. Towards deploying VLA without fine-tuning: Plug-and-play inferencetime VLA policy steering via embodied evolutionary diffusion. IEEE Robotics and Automation Letters, 11(5):6234–6241, 2026b. doi: 10.1109/LRA. 2026.3678455. URL https://ieeexplore.ieee. org/document/11457243/.

Liu, B., Zhu, Y., Gao, C., Feng, Y., Liu, Q., Zhu, Y., and Stone, P. LIBERO: Benchmarking knowledge transfer for lifelong robot learning. arXiv preprint arXiv:2306.03310, 2023.

Liu, S., Singh, I. S., Xu, Y., Duan, J., and Krishna, R. VLS: Steering pretrained robot policies via vision-language models. arXiv preprint arXiv:2602.03973, 2026a.

Liu, Y., Sun, P., Chao, X., Yang, Z., Xie, Y., Zhang, L., Li, S., Tang, C., Chen, F., Zhang, X.-P., and Ding, W. CheckVLA: Execution-time verification with actionconditioned world model for long-horizon mobile manipulation. arXiv preprint arXiv:2607.26789, 2026b.

Ma, J., Liu, Z., and Zhu, Y. FailureSpot: Label-efficient timestamp-level failure detection for vision-languageaction models. arXiv preprint arXiv:2609.04277, 2026.

NVIDIA, Bjorck, J., Castaneda, F., Cherniadev, N., Da, X.,˜ Ding, R., Fan, L. J., Fang, Y., Fox, D., Hu, F., Huang, S., Jang, J., Jiang, Z., Kautz, J., Kundalia, K., Lao, L., Li, Z., Lin, Z., Lin, K., Liu, G., Llontop, E., Magne, L., Mandlekar, A., Narayan, A., Nasiriany, S., Reed, S., Tan, Y. L., Wang, G., Wang, Z., Wang, J., Wang, Q., Xiang, J., Xie, Y., Xu, Y., Xu, Z., Ye, S., Yu, Z., Zhang, A., Zhang, H., Zhao, Y., Zheng, R., and Zhu, Y. GR00T N1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.

Park, S., Li, W., Oh, C., Yeh, S., Kira, Z., Hagenow, M., and Li, S. Hide-and-seek in trajectories: Discovering failure signals for VLA runtime monitoring. arXiv preprint arXiv:2605.30834, 2026.

Physical Intelligence, Black, K., Brown, N., Darpinian, J., Dhabalia, K., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., Galliker, M. Y., Ghosh, D., Groom, L., Hausman, K., Ichter, B., Jakubczak, S., Jones, T., Ke, L., LeBlanc, D., Levine, S., Li-Bell, A., Mothukuri, M., Nair, S., Pertsch, K., Ren, A. Z., Shi, L. X., Smith, L., Springenberg, J. T., Stachowicz, K., Tanner, J., Vuong, Q., Walke, H., Walling, A., Wang, H., Yu, L., and Zhilinsky,

U. $\pi _ { 0 . 5 } { : }$ a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

Physical Intelligence, Ai, B., Amin, A., Aniceto, R., Balakrishna, A., Balke, G., Black, K., Bokinsky, G., Cao, S., Charbonnier, T., Choudhary, V., Collins, F., Conley, K., Connors, G., Darpinian, J., Dhabalia, K., Dhaka, M., DiCarlo, J., Driess, D., Equi, M., Esmail, A., Fang, Y., Finn, C., Glossop, C., Godden, T., Goryachev, I., Groom, L., Habeeb, H., Hancock, H., Hausman, K., Hussein, G., Hwang, V., Ichter, B., Jacobsen, C., Jakubczak, S., Jen, R., Jones, T., Kammerer, G., Katz, B., Ke, L., Khadikov, M., Kuchi, C., Lamb, M., LeBlanc, D., LeCount, B., Levine, S., Li, X., Li-Bell, A., Lialin, V., Liang, Z., Lim, W., Lu, Y., Luo, E., Mano, V., Marwaha, N., Mongush, A., Murphy, L., Nair, S., Patterson, T., Pertsch, K., Ren, A. Z., Schelske, G., Sharma, C., Shi, B., Shi, L. X., Smith, L., Springenberg, J. T., Stachowicz, K., Stoeckle, W., Tang, J., Tanner, J., Tekeste, S., Torne, M., Vedder, K., Vuong, Q., Walling, A., Wang, H., Wang, J., Wang, X., Whalen, C., Whitmore, S., Williams, B., Xu, C., Yoo, S., Yu, L., Zhang, W., Zhang, Z., and Zhilinsky, U. π<sub>0.7</sub>: a steerable generalist robotic foundation model with emergent capabilities. arXiv preprint arXiv:2604.15483, 2026.

Rajaprakash, H., Prajapati, A., Xue, R., Anwar, A., and Thomason, J. SAFECAST: Robust failure detection for VLA policies with contrast-set training and calibration. arXiv preprint arXiv:2608.04246, 2026.

Swann, A., McGranahan, L., Buurmeijer, H., Kennedy, M., and Schwager, M. Sparse autoencoders reveal interpretable and steerable features in VLA models. arXiv preprint arXiv:2603.19183, 2026.

Tan, D. M. S., Shailesh, S., Iyer, S., Teo, W. W. J., Ju, Y., Gu, Q., and Sartoretti, G. RL<sup>2</sup>-VLA: Adaptive RL latent compositional steering with test-time scaling for vision-language-action models. arXiv preprint arXiv:2607.26991, 2026.

Turner, A. M., Thiergart, L., Leech, G., Udell, D., Vazquez, J. J., Mini, U., and MacDiarmid, M. Steering language models with activation engineering. arXiv preprint arXiv:2308.10248, 2024.

Wang, G., Zhang, C., Liu, Q., Zhang, J., Cai, J., Liu, J., and Liu, X. LIBERO-X: Robustness litmus for vision-language-action models. arXiv preprint arXiv:2602.06556, 2026.

Xu, C., Nguyen, T. K., Dixon, E., Rodriguez, C., Miller, P., Lee, R., Shah, P., Ambrus, R., Nishimura, H., and Itkina, M. Can we detect failures without failure data? Uncertainty-aware runtime failure detection for imitation learning policies. arXiv preprint arXiv:2503.08558, 2025.

Xu, Y., Hu, Y., Zhang, Z., Meyer, G. P., Mustikovela, S. K., Srinivasa, S., Wolff, E. M., and Huang, X. VLM-AD: End-to-end autonomous driving through vision-language model supervision. arXiv preprint arXiv:2412.14446, 2024.

Yang, Y., Bhatt, N. P., Samineni, P., Siva, R., Wang, Z., and Topcu, U. RepV: Safety-separable latent spaces for scalable neurosymbolic plan verification. arXiv preprint arXiv:2510.26935, 2025a.

Yang, Y., Hu, Y., Ye, M., Zhang, Z., Lu, Z., Xu, Y., Topcu, U., and Snyder, B. Uncertainty-guided enhancement on driving perception system via foundation models. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pp. 8752–8758, 2025b. doi: 10.1109/ICRA55743.2025. 11127651. URL https://ieeexplore.ieee. org/document/11127651/.

## A METHOD DETAILS

This section gives the details behind section 4.

## A.1 Probe and Training

Inputs. At every policy step we average the backbone tokens of the agent camera, of the wrist camera, and of the (unpadded) language prompt, and concatenate the three means into $e _ { t } \in \mathbb { R } ^ { \bar { 6 } 1 4 4 }$ . No other feature of the backbone is used. All models in this paper read this vector, the baselines included.

Probe. A single-layer LSTM with 256 hidden units reads $e _ { 1 : t } .$ Three heads read its state, each an MLP with one hidden layer of 64 ReLU units: the failure score head (one sigmoid output), the failure mode head (logits over the 10 modes), and the semantic head (a 64-dimensional vector).

Semantic (CLIP) head. The target space of the semantic head is a PCA of the CLIP text embeddings of the timeline phrases (clip-vit-base-patch32), reduced to 64 dimensions with a PCA fitted on the training split only. The loss is InfoNCE with temperature 0.07: the normalized output of a step must pick the target of the step’s phrase among the targets of all phrases of the training split. A step whose phrase is not among them is left out of this term. At inference the head is only a 64-dimensional MLP output; no CLIP encoder runs.

Loss and training. $\mathcal { L } _ { \mathrm { s c o r e } }$ is a binary cross-entropy over the steps that are not excluded, weighted by the ratio of negative to positive steps of the training set. ${ \mathcal { L } } _ { \mathrm { m o d e } }$ is a cross-entropy with class weights that balance the modes, on the positive steps of failed rollouts. Both auxiliary terms have weight 1. Early stopping follows $\mathcal { L } _ { \mathrm { s c o r e } }$ on the validation episodes only, so the auxiliary heads never choose the checkpoint.

Labeling window. The window w of the onset and postonset labels, and the window of the mode read-out (subsection C.2), is 2.4 s: 6 policy steps for GR00T on LIBERO, 20 on the real robot, and 10 steps (2.5 s) for $\pi _ { 0 . 5 }$ <sub>5</sub>.

## A.2 Supervision Strategies

Let $e _ { t }$ be the backbone embedding at step t and $y _ { t } ~ \in$ $\{ 0 , 1 , \emptyset \}$ its target, where $y _ { t } = \mathcal { D }$ excludes the step from the loss. Let ${ Y _ { i } } ~ \in ~ \{ 0 , 1 \}$ be the episode label (success, failure) and w the width of the window around $t _ { \mathrm { f a i l } }$ . For a failed rollout $( Y _ { i } = 1 )$ the four strategies of Figure 2 are

$$
\begin{array} { r } { P r e - o n s e t . \quad y _ { t } = \left\{ \begin{array} { l l } { 1 } & { t \leq t _ { \mathrm { f a i l } } , } \\ { \emptyset } & { t > t _ { \mathrm { f a i l } } , } \end{array} \right. } \end{array}
$$

$$
O n s e t . y _ { t } = \left\{ \begin{array} { r l } { 1 } & { t \in [ t _ { \mathrm { f a i l } } - w , t _ { \mathrm { f a i l } } ] , } \\ { 0 } & { t < t _ { \mathrm { f a i l } } - w , } \\ { \emptyset } & { t > t _ { \mathrm { f a i l } } , } \end{array} \right.
$$

$$
P o s t \ – o n s e t . \quad y _ { t } = \left\{ { 1 } \quad t \geq t _ { \mathrm { f a i l } } - w , \right.
$$

$$
T r a j e c t o r y . \quad y _ { t } = 1 \quad { \mathrm { f o r ~ a l l ~ } } t .
$$

In all four, $y _ { t } = 0$ for every step of a successful rollout.   
Excluding a step $( y _ { t } = \boldsymbol { \mathcal { D } } )$ is the same as giving it zero loss.   
Trajectory is the labeling of SAFE.

## A.3 Calibration and Guarantees

Alarm rule and risks. For a constant threshold h the alarm time of an episode is $t _ { \mathrm { d e t } } ( h ) = \operatorname* { m i n } \{ t : s _ { t } \geq h \}$ The $\mathrm { F P R } ( h )$ is the probability that a successful episode has an alarm, and $\mathrm { T P R } _ { \mathrm { e a r l y } } ( h )$ the probability that a failed episode has its first alarm at or before $t _ { \mathrm { f a i l } }$

Learn-then-Test. We follow LTT (Angelopoulos et al., 2022) with two risks, the FPR and the early miss rate $1 - \mathrm { T P R } _ { \mathrm { e a r l y } }$ (Section 2.4 there). A threshold h is tested against the null that it violates either target, and its p-value is the larger of the two p-values (Proposition 6). Both losses are binary, so each p-value is an exact binomial tail (Section 3.2): the number of calibration failures without an alarm at or before $t _ { \mathrm { f a i l } }$ is compared with Bi ${ \mathrm { _ { 1 } } } ( n _ { \mathrm { f } } , 1 - \beta )$ and the number of calibration successes with any alarm with $\mathrm { B i n } ( n _ { \mathrm { s } } , \alpha )$ . We combine the p-values by multi-start fixedsequence testing (Section 2.3.1, Algorithm 1, Proposition 3) over a grid of 181 thresholds sigmoid $( z ) , z \in [ - 9 , 9 ]$ , ordered from low to high, with $S = 4 0$ starts and $\delta = 0 . 1$ , i.e. a level of 0.0025 per walk. By Theorem 1 of Angelopoulos et al. (2022), the set Λ<sup>ˆ</sup> of accepted thresholds satisfies $\begin{array} { r } { \mathbb { P } \big ( \operatorname* { s u p } _ { h \in \hat { \Lambda } } \{ R ( h ) \} \le \alpha \big ) \ge 1 - \delta } \end{array}$ , where R stands for each of our two risks with its target; that is, with probability at least $1 - \delta$ every accepted threshold meets both targets, for episodes exchangeable with the calibration episodes. We select the accepted threshold with the highest calibration $\mathrm { T P R } _ { \mathrm { e a r l y } }$ (ties: lowest calibration FPR, then lowest threshold), and abstain if none is accepted.

Max-conformal (MC). For STAC (whose accumulated scores are non-negative, so the maximum over steps is the final score), for ActProbe (as in its paper), and for our probe in the main table, we use max-conformal prediction. It takes the maximum score $M _ { i }$ of each of the n calibration successes and sets the constant threshold to the k-th smallest of them, with $k = \lceil ( n + 1 ) ( 1 - \alpha ) \rceil$ , and $_ { \mathrm { t o } + \infty \mathrm { i f } \ k } >$ n. For a new success exchangeable with the calibration successes, its maximum has a uniformly distributed rank, so the false-alarm probability is at most α, and conservative by less than $1 / ( n + 1 )$ . It says nothing about the TPR.

Functional conformal (FC). For SAFE and the methods that follow its calibration (FailureSpot and the embeddingdistance baselines), we use the one-sided functional conformal band of SAFE (Gu et al., 2025). The band is a time-varying threshold built from the score trajectories of the calibration successes: the mean trajectory gives its shape, the local spread its width over time, and a conformal quantile of the successes’ deviations sets its level so that a success stays below it with probability about 1 − α.

Which method gets which calibrator. SAFE-MLP, SAFE-LSTM, FailureSpot, and the three embeddingdistance baselines use the SAFE band, as in their own work. STAC, ActProbe, and our probe use max-conformal.

## B DATASETS AND LABELING

## B.1 Benchmarks and Splits

LIBERO-10 with occlusions. We use the 10-suite of LIBERO-Occ (Li et al., 2026a), an extension of LIBERO (Liu et al., 2023) that occludes the agent-view camera. Both policies fail rarely on the original benchmark, so we mix occluded and unoccluded episodes in equal parts to obtain enough failures (250 of each per policy).

LIBERO-X. LIBERO-X (Wang et al., 2026) introduces five levels of distribution shift. We use levels L1–L3 and select 218 tasks that cover diverse skills. Table 2 gives the task counts of the splits.

Real robot. We collected demonstrations for 8 tabletop manipulation tasks on a Kinova Gen3 arm (Table 4) and fine-tuned GR00T on them. The evaluation data are 399 rollouts of the fine-tuned policy (49 to 50 per task), 254 of them failures; Table 5 gives their failure modes.

Table 4. The eight real-world tasks and the number of demonstrations used to fine-tune the policy (510 in total).
<table><tr><td>Task instruction</td><td>Third-person view</td><td>Demos</td></tr><tr><td>Close the drawer</td><td><img src="images/28fd0edfd70f9b7918c5c4a6e389bdac7eac7bb1ef859755e4dcc2a8ab271c9b.jpg"/></td><td>50</td></tr><tr><td>Flip the basket upright</td><td><img src="images/9fca0a449e5750e2e584ae0a45d6853de77117047832eeb0215cdd0314a87980.jpg"/></td><td>50</td></tr><tr><td>Push the pineapple off the stove</td><td><img src="images/b72b8d7ebc99f529ab6cca1187db0031562cb8bed31651fa05589ae043ed0db0.jpg"/></td><td>100</td></tr><tr><td>Put the mushroom on the plate</td><td>上</td><td>50</td></tr><tr><td>Put the tomato in the bin</td><td></td><td>100</td></tr><tr><td>Stack the yellow block on the wooden block</td><td><img src="images/68ce18368848e86760ce093593fbef49eeff0279b62279e37b970e0b67510245.jpg"/></td><td>60</td></tr><tr><td>Take the eggplant off the plate</td><td><img src="images/8d934f86b27f5c36fe64643fbc215a45380ad09a549aee6acf359968a45fef8b.jpg"/></td><td>50</td></tr><tr><td>Take the pineapple out of the drawer</td><td><img src="images/311d80d20658d405ee5c5a325aadefc04adf8ff445ce2efe2d098fd2234cd5fd.jpg"/></td><td>50</td></tr></table>

Construction of the splits. The episodes of the held-out tasks form the unseen set. The episodes of the seen tasks are split 65/15/20% into train, validation, and test. The test part is halved, into the calibration set and the held-out seen set. The calibration sets of the LIBERO-X runs hold 158 to 209 episodes (66 to 90 successes and 92 to 120 failures). On the real robot each run holds one task out, so the calibration part has only 12 to 14 successes, a max-conformal threshold exists only for α of about 0.07 and above, and the curves of all methods start at an FPR of about 8%. The area up to 40% FPR is still defined and treats every method the same, but it is noisy. The seed changes the shuffle of the seen episodes and the initialization of the learned models; for the new-task and new-signature splits the unseen episodes are identical in the three seeds, so their spread is not a spread over different unseen tasks.

## B.2 Labeling Pipeline

We label each rollout with Gemini (gemini-3.5-flash) from its video, in three passes. A coarse pass over the whole video of a failed rollout returns the failure mode and a failure time; a second pass on a slowed clip around that time refines the failure time to a single frame; and a third pass on the same video session gives the timeline: a short keyword phrase (2–6 words, e.g. “reaching for $\mathrm { \ m u g ^ { \prime } ) }$ for each 3-second window, which provides the targets of the semantic head. The failure time $t _ { \mathrm { f a i l } }$ is the earliest point at which the mistake responsible for the failure becomes apparent. Following (Gu et al., 2025), if there is no single mistake but a human operator would take over (for example when the robot is stuck), we use that time, and if the robot behaves plausibly but is too slow, the failure time is the end of the rollout.

The failure mode is one of 10 classes: wrong object (acts on an object or control other than the one the task specifies); wrong target (right object, wrong destination); grasp failure (never acquires the correct object securely); object displacement (knocks over, pushes away, or drops an object); placement or insertion failure (right object and target, wrong final pose); open close failure (right container, but not the required open or closed state); press failure (right button or knob, but no effect); stuck or no progress (looping, oscillating, hovering, or freezing); timeout or insufficient progress (purposeful but too slow, no identifiable mistake); and other. The full definitions and the prompts are on the project page, vlatug.github.io, and on the card of each dataset.

## B.3 Dataset Statistics and Release

Table 5 gives, for each policy and dataset, the share of each failure mode among the failures and the mean and standard deviation of its failure time.

We release the embeddings, the labels, the prompts, and the splits of the simulated datasets, 3,180 episodes, at vlatug.github.io.

## C EXPERIMENTAL SETUP

## C.1 Baselines

We re-implement all baselines on the same inputs and splits as our probe. Where a baseline has a configuration grid, we run the whole grid on every split and select, for every setting, the configuration with the highest mean area on the validation episodes, never looking at the unseen tasks; ties go to the earlier configuration of the grid. LIBERO-10 and the real robot pool their runs into one choice per method. On LIBERO-X, SAFE uses one configuration for the three kinds of splits, as in its own protocol, and every other method chooses per kind of held-out set. Our probe uses one architecture in every setting.

Embedding distance. Mahalanobis, Euclidean, and PCA-KMeans are SAFE’s embedding baselines (Gu et al., 2025). The score of a step is $D _ { \mathrm { s u c c } } ( x ) - D _ { \mathrm { f a i l } } ( x )$ , where $D _ { c }$ is the Mahalanobis distance to a Gaussian fitted to the steps of class c, the mean Euclidean distance to the k nearest stored steps of that class (Euclidean, also called k-NN), or the distance to the nearest of a few k-means centroids in a PCAreduced space (PCA-KMeans); it can be accumulated over the steps. We run a sweep of 26 configurations (Mahalanobis 2, Euclidean 6, PCA-KMeans 18).

Action consistency. STAC (Agia et al., 2024) compares the unexecuted tail of the previous plan with the first actions of the new plan, using the K = 10 plans the policy samples at every call and the maximum mean discrepancy with an RBF kernel; nothing is learned. We run 8 variants (MMD over 3, 6, or 7 action dimensions, with the step score or accumulated; and STAC-Single, SAFE’s single-plan variant). ActProbe (Huang et al., 2026) is a small LSTM on two action signals and an embedding of the task instruction; we train it with the recipe of its repository and select its checkpoint with its own validation criterion. FailureSpot (Ma et al., 2026) has no public code, so we implemented it from the paper: a detector on the pooled features is trained on weak targets computed from action consistency and magnitude, and then fine-tuned on timestamp labels of the 15% most uncertain episodes.

Latent. SAFE-MLP and SAFE-LSTM (Gu et al., 2025) are written from SAFE’s code with its losses and its labels: every step of a failed rollout is positive, i.e. our trajectory label. We run SAFE’s learning-rate and regularization grid, 12 configurations per model. SAFE trains for a fixed number of epochs without validation; we stop early after 50 epochs without improvement of its validation loss (cap: 1,000 epochs) and keep the best epoch, so that all learned methods are treated alike.

## C.2 Metrics

We define three events for a rollout: detection $\mathcal { A } = \{ t _ { \mathrm { d e t } } \le$ $t _ { \mathrm { m a x } } \mid Y = 1 \}$ , early detection $\mathcal { A } _ { \mathrm { { e a r l y } } } = \{ t _ { \mathrm { d e t } } \ \leq \ t _ { \mathrm { f a i l } } \ |$ $Y = 1 !$ , and no alarm $\mathcal { N } = \{ t _ { \mathrm { d e t } } = \mathrm { \bar { n u l l } \mid \mathcal { Y } = 0 } \}$ . We count A as a true positive (TP) and $\boldsymbol { A } _ { \mathrm { e a r l y } }$ as an early true positive $( \mathrm { T P _ { e a r l y } ) ; }$ the complements are the false negatives. The TPR and FPR follow their standard definitions, and

$$
\mathrm { T P R } _ { \mathrm { e a r l y } } = \frac { \mathrm { T P } _ { \mathrm { e a r l y } } } { \mathrm { T P } _ { \mathrm { e a r l y } } + \mathrm { F N } _ { \mathrm { e a r l y } } } ,\tag{7}
$$

$$
\mathrm { A U R O C _ { \mathrm { e a r l y } } = \frac { 1 } { 0 . 4 } \int _ { 0 } ^ { 0 . 4 } T P R _ { \mathrm { e a r l y } } ( F P R ) d F P R , }\tag{8}
$$

Table 5. Failure-mode distribution and mean ± std failure time (s) per mode, by policy and dataset.
<table><tr><td></td><td colspan="2"> $\pi _ { 0 . 5 } \mathrm { L I B E R O - 1 0 - O c c }$ </td><td colspan="2">GR00T LIBERO-10-Occ</td><td colspan="2">GR00T LIBERO-X</td><td colspan="2">GR00T real robot</td></tr><tr><td>Mode</td><td>% of failures</td><td> ${ \mathrm { T i m e ~ } } ( { \mathrm { s } } )$ </td><td> $\% { \mathrm { o f ~ f a i l u r e s } }$ </td><td> ${ \mathrm { T i m e ~ } } ( { \mathrm { s } } )$ </td><td>% of failures</td><td>Time (s)</td><td>% of failures</td><td>Time (s)</td></tr><tr><td>stuck_or_no_progress</td><td>24.5</td><td> $1 0 . 3 \pm { 3 . 5 }$ </td><td>30.2</td><td> $8 . 6 \pm 3 . 7$ </td><td>14.1</td><td> $1 5 . 5 \pm 1 0 . 7$ </td><td>12.2</td><td> $3 . 8 \pm 3 . 0$ </td></tr><tr><td>object_displacement</td><td>18.2</td><td> $1 1 . 9 \pm 5 . 3$ </td><td>17.8</td><td> $\smash { 1 1 . 8 \pm 6 . 7 }$ </td><td>10.8</td><td> $1 5 . 1 \pm 1 0 . 9$ </td><td>9.4</td><td> $4 . 9 \pm 2 . 3$ </td></tr><tr><td>placement_or_insertion</td><td>13.6</td><td> $1 5 . 2 \pm 4 . 2$ </td><td>16.0</td><td>12.9 ± 7.0</td><td>9.9</td><td> $2 1 . 9 \pm 1 0 . 2$ </td><td>11.4</td><td> $8 . 3 \pm 4 . 5$ </td></tr><tr><td>grasp_failure</td><td>17.3</td><td> $9 . 6 \pm 5 . 4$ </td><td>13.0</td><td> $8 . 6 \pm 4 . 3$ </td><td>9.0</td><td> $1 2 . 0 \pm 9 . 7$ </td><td>56.7</td><td> $4 . 7 \pm 2 . 3$ </td></tr><tr><td>open_close_failure</td><td>7.3</td><td> $1 2 . 6 \pm 4 . 8$ </td><td>9.5</td><td> $1 3 . 9 \pm 3 . 1$ </td><td>4.7</td><td> $1 8 . 7 \pm 1 1 . 5$ </td><td>7.5</td><td> $3 . 8 \pm 1 . 7$ </td></tr><tr><td>wrong-target</td><td>13.6</td><td> $6 . 8 \pm 4 . 9$ </td><td>8.9</td><td> $4 . 9 \pm 4 . 4$ </td><td>12.3</td><td> $1 4 . 3 \pm 9 . 7$ </td><td>0.0</td><td></td></tr><tr><td>wrong-object</td><td>0.0</td><td></td><td>1.8</td><td> $5 . 3 \pm 3 . 1$ </td><td>37.0</td><td> $5 . 8 \pm 7 . 2$ </td><td>1.2</td><td> $2 . 0 \pm 0 . 8$ </td></tr><tr><td>press_failure</td><td>2.7</td><td> $4 . 4 \pm 1 . 4$ </td><td>1.8</td><td> $1 0 . 7 \pm 4 . 8$ </td><td>1.5</td><td> $2 6 . 8 \pm 1 8 . 5$ </td><td>0.0</td><td></td></tr><tr><td>timeout</td><td>2.7</td><td> $2 5 . 9 \pm 0 . 0$ </td><td>1.2</td><td> $2 5 . 9 \pm 0 . 0$ </td><td>0.2</td><td> $6 0 . 0 \pm 0 . 0$ </td><td>1.6</td><td> $6 . 1 \pm 1 . 1$ </td></tr><tr><td>other</td><td>0.0</td><td></td><td>0.0</td><td></td><td>0.4</td><td> $8 . 6 \pm 7 . 4$ </td><td>0.0</td><td></td></tr><tr><td>n failed / total</td><td colspan="2">110/500</td><td colspan="2">169/500</td><td colspan="2">1258/2180</td><td colspan="2">254/399</td></tr></table>

so that $\mathrm { A U R O C _ { \mathrm { e a r l y } } }$ is reported in % and a detector with $\mathrm { T P R } _ { \mathrm { e a r l y } } = 1 0 0 \%$ everywhere scores 100.

## C.3 Full Results

Table 6 gives Table 3 with the standard deviation over the splits, and Figure 7 shows the curves of every setting and method. On the real robot the standard deviation is over the 40 splits (8 held-out tasks, 5 seeds).

## D RESULTS IN DETAIL

## D.1 Calibration and Guarantees

This section gives the details behind subsection 6.2.

Setup. We run this analysis on LIBERO-X only. LIBERO-10 has only a few tens of failures per split, and the real robot has only 12 to 14 calibration successes per run, too few for LTT to certify any $\mathrm { T P R } _ { \mathrm { e a r l y } }$ target together with an FPR target in at least half of the runs; its LTT rows would be empty. The 14 runs are the 3 NEW-TASK, 3 NEW-SIGNATURE, and 8 NEW-SKILL splits. We use the FPR targets 10, 15, 20, 25, 30, 40, and 50%. For each of them, we report the highest $\mathrm { T P R } _ { \mathrm { e a r l y } }$ target that LTT certifies together with it in at least half of the runs, searched on a grid from 10% to 95% in steps of 5%. LTT uses $\delta = 0 . 1$ and $S = 4 0$ starting points for its fixed-sequence walks (subsection A.3). Table 7 gives for each FPR target the shares of runs in which the targets hold on the held-out seen and unseen episodes, the realized rates, and the PPV bound, for LTT and for the two calibrators of the baselines, max-conformal (MC) and functional conformal (FC).

How to read Table 7. Runs is the share of runs for which LTT certifies the targets; the other columns of an LTT row are over these runs. Holds is the share of runs in which the target is met on the evaluated episodes. The PPV bound is $\beta \pi / ( \beta \pi + \alpha ( 1 - \pi ) )$ with π the prevalence of the calibration set, and PPV holds is the share of runs in which the realized PPV is at least the bound (computed with the calibration set’s prevalence, or with that of the evaluated set).

PPV bound. The bound follows from the two LTT targets, so for VLA-TUG it comes directly from the FPR and $\mathrm { T P R } _ { \mathrm { e a r l y } }$ targets we calibrate for. It depends on the prevalence π, which we estimate on the calibration set. On seen tasks the estimate is stable. Under shift it can drift on individual NEW-SKILL folds, where the share of failures can differ a lot between the calibration set and the held-out tasks; this is what limits the bound on unseen tasks, and using the unseen set’s own π removes most of the misses. A moving average of π over the observations at deployment time could deal with more severe shift.

Unseen tasks. On the NEW-TASK and NEW-SIGNATURE splits, the three runs of each split share the same unseen tasks and differ only in the calibration data and the initialization. The held-out tasks define the shift each split is designed to test, so we cannot redraw them between runs. Only the eight NEW-SKILL folds hold out different tasks. The shares on unseen tasks therefore rest on fewer distinct unseen sets than runs, so they indicate trends and not precise rates.

## D.2 Informativeness

This section gives the details behind subsection 6.3.

Classes. The failure modes are strongly imbalanced, so we report a balanced accuracy: the mean over classes of the share of their failures predicted correctly. We merge the two smallest classes repeatedly, once per dataset, until every class holds at least 5% of the failures. The rule uses label frequencies only. The mode head is not retrained: a merged class gets the sum of the probabilities of its modes. The accuracy is computed on the predictions pooled over the splits of a setting, since a split has few failures of a rare class. We exclude LIBERO-10: it has too few failures for a per-class accuracy, since even after merging its smallest classes hold only 6 to 16 failures, so one failure changes a class’s accuracy by several points.

Table 6. Table 3 with the standard deviation over the splits: area under TPR early vs. FPR up to 40% (in %). Dashes: the action chunks the method needs were not saved. Best in bold, second best in italics.
<table><tr><td></td><td></td><td colspan="2">LIBERO-10 π0.5</td><td colspan="2">LIBERO-10 GR0OT</td><td colspan="2">LIBERO-X task</td><td colspan="2">LIBERO-X signature</td><td colspan="2">LIBERO-X skill</td><td colspan="2">Real robot</td><td colspan="4">Cost (LIBERO-X)</td></tr><tr><td>Method</td><td>Cal.</td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td><td>Fit (s) State (MB)</td><td>GPU (ms)</td><td>CPU (ms)</td><td></td></tr><tr><td>SAFE-LSTM</td><td>FC</td><td>75.5±13.4</td><td>34.9±3.5</td><td>74.1±13.0</td><td>51.2±8.4</td><td>64.6±3.5</td><td>40.4±2.3</td><td>64.7±7.4</td><td>41.2±7.1</td><td>62.5±2.4</td><td>30.1±8.5</td><td>44.5±13.5</td><td>25.1±18.8</td><td>43.5</td><td>26.2</td><td>0.066</td><td>0.86</td></tr><tr><td>SAFE-MLP</td><td>FC</td><td>74.1±2.0</td><td>39.9±15.1</td><td>62.9±18.2</td><td>39.2±9.4</td><td>59.3±2.0</td><td>42.3±2.9</td><td>53.7±2.4</td><td>46.2±2.3</td><td>56.8±6.0</td><td>23.5±12.4</td><td>40.1±15.9</td><td>20.2±16.6</td><td>114.5</td><td>6.3</td><td>0.034</td><td>0.032</td></tr><tr><td>STAC</td><td>MC</td><td>33.8±5.1</td><td>36.4±5.3</td><td>24.2±7.5</td><td>17.5±9.8</td><td>21.0±2.3</td><td>28.0±0.1</td><td>25.6±3.7</td><td>28.7±0.3</td><td>24.6±5.0</td><td>21.5±8.0</td><td></td><td></td><td>0.0</td><td>0.0</td><td>21.30</td><td></td></tr><tr><td>ActProbe</td><td>MC</td><td>39.3±11.3</td><td>26.7±2.6</td><td>32.4±16.8</td><td>30.1±9.3</td><td>12.6±4.2</td><td>22.7±4.6</td><td>11.9±3.1</td><td>13.9±1.5</td><td>20.3±9.4</td><td>13.6±7.1</td><td></td><td></td><td>7.1</td><td>0.09</td><td>0.15</td><td>0.11</td></tr><tr><td>FailureSpot</td><td>FC</td><td>61.9±3.0</td><td>36.2±21.0</td><td>36.9±19.8</td><td>44.7±18.6</td><td>26.3±2.1</td><td>20.4±1.2</td><td>28.0±2.5</td><td>26.2±5.6</td><td>30.4±3.4</td><td>19.6±7.9</td><td></td><td></td><td>9.2</td><td>6.4</td><td>0.033</td><td>0.029</td></tr><tr><td>Mahalanobis</td><td>FC</td><td>85.7±4.1</td><td>40.2±15.5</td><td>70.7±6.4</td><td>29.2±8.7</td><td>69.6±6.2</td><td>30.6±2.2</td><td>64.4±3.0</td><td>22.3±4.0</td><td>71.2±3.1</td><td>15.9±9.6</td><td>43.0±16.5</td><td>19.6±2.5</td><td>15.2</td><td>302.0</td><td>0.41</td><td>5.37</td></tr><tr><td>Euclidean</td><td>FC</td><td>73.1±4.4</td><td>24.2±10.6</td><td>56.0±6.8</td><td>23.1±11.4</td><td>70.5±4.2</td><td>28.5±3.0</td><td>65.4±2.1</td><td>19.7±0.8</td><td>68.5±4.8</td><td>19.0±6.5</td><td>45.1±14.2</td><td>8.3±12.8</td><td>0.4</td><td>3056</td><td>13.19</td><td>368.83</td></tr><tr><td>PCA-KMeans</td><td>FC</td><td>35.7±12.2</td><td>12.9±5.1</td><td>44.5±8.7</td><td>20.0±0.0</td><td>31.7±3.4</td><td>25.0±1.2</td><td>35.3±1.8</td><td>32.5±1.7 37.8±7.7</td><td></td><td>20.7±7.1</td><td>24.0±11.4</td><td>19.4±7.7</td><td>39.9</td><td>1.6</td><td>0.12</td><td>0.055</td></tr><tr><td>VLA-TUG (Ours) MC</td><td></td><td>75.1±12.5</td><td>47.9±20.9</td><td>70.5±10.4</td><td></td><td></td><td></td><td>47.4±13.5 67.9±5.0 40.2±1.2 67.0±6.6</td><td>47.0±0.6 67.8±4.8</td><td></td><td>39.0±7.0 47.6±10.9</td><td></td><td>25.8±11.6</td><td>6.8</td><td>26.5</td><td>0.12</td><td>0.94</td></tr></table>

![](images/e4408739946ad982390b4ed1d32ee668c60fa7e049329bf570bcc60983ef74a7.jpg)  
Figure 7. $\mathrm { T P R } _ { \mathrm { e a r l y } }$ against FPR for every setting and method, on seen (top) and unseen (bottom) tasks; mean over the splits of a setting.

Prediction. A failed episode is predicted at an anchor step: the class with the highest mean probability over the 2.4 s window before the anchor and the anchor itself (the top three classes for top-3). The anchor is either the probe’s first alarm (max-conformal at $\alpha = 0 . 2 ;$ episodes without an alarm are not scored) or the true failure time, which is an upper bound that a deployed system can reach. The baselines predict the most common training classes (always) or guess uniformly among the training classes.

## D.3 Cost and Latency

This section gives the details behind subsection 6.4.

How the VLA runs. At each policy step the VLA receives the camera images and the prompt and produces a chunk of m actions (m = 8 on LIBERO-X, 5 for $\pi _ { 0 . 5 }$ on LIBERO-10, 6 on the real robot), which the robot executes at its control rate (20 Hz in simulation, 50 Hz on the real robot). A policy step therefore lasts m divided by the control rate: 400 ms on LIBERO-X, 250 ms for $\pi _ { 0 . 5 } ,$ and 120 ms on the real robot. The probe reads the embedding of each call and must return its score well within one step.

Measurement. All times are measured on one RTX 4090 and 32 CPU threads. The fit time is the training time of the probe or the SAFE models, or the time to fit the statistics of the distance baselines. The state is what a fitted method must keep to score a new episode: the weights, the stored training steps of the k-NN, or the statistics of the distance baselines. The latency is the online latency: the time to turn one new policy step into a score, one embedding at a time and with the recurrent state of the previous step, after a warm-up and with the device synchronized, as the median over up to 2,000 steps. The CPU is timed with four threads. Online latency is the one that matters for deployment.

Baselines. The k-NN baseline has to keep every training step and compare each new step with all of them, so its cost grows with the training set; Mahalanobis keeps two 6144 × 6144 matrices. STAC needs the K = 10 plans of the policy at every call, which no other method needs, so its latency is its scoring time plus the time to sample them.

Probe. On the GPU the LSTM takes most of the time; the mode head and the CLIP head add little, and on the CPU they add nothing measurable. The probe is not the fastest online method (SAFE-LSTM and the MLP detectors are faster), but it is far below the duration of a policy step.

Table 7. Learn-then-Test (LTT), max-conformal (MC), and functional conformal (FC) at the same FPR target, over the 14 LIBERO-X runs (δ = 0.1). LTT certifies the TPR-early target of the second column together with the FPR target, and abstains otherwise; its percentages are over the certified runs. PPV bound: βπ/(βπ + α(1 − π)).
<table><tr><td colspan="3"></td><td colspan="4">Seen tasks</td><td colspan="2">PPV PPV holds</td><td colspan="5">Unseen tasks PPV</td><td rowspan="2">PPV holds (own π)</td></tr><tr><td></td><td>FPR target TPR-early target PPV bound Method</td><td></td><td>Runs</td><td>FPR holds TPR holds</td><td></td><td>FPR TPR early</td><td>PPV</td><td>holds (own π)</td><td>FPR holds TPR holds</td><td></td><td>FPR TPR early PPV</td><td></td><td>holds</td></tr><tr><td>10%</td><td></td><td>LTT</td><td>0% (0/14)</td><td></td><td>一</td><td>一</td><td></td><td>一</td><td></td><td>一</td><td></td><td></td><td>一</td><td></td></tr><tr><td rowspan="3"></td><td></td><td>Max conformal (MC)</td><td>100% (14/14)</td><td>57%</td><td>9.3±4.4</td><td>53.4±12.7</td><td>90.1</td><td></td><td>50%</td><td></td><td>15.6±12.3</td><td>35.3±20.5 80.7</td><td></td><td></td></tr><tr><td></td><td>Functional conformal (FC)</td><td>100% (14/14)</td><td>71%</td><td>9.8±5.8</td><td>33.9±17.2</td><td>90.7</td><td></td><td>36%</td><td></td><td>14.9±9.4</td><td>26.3±16.2 84.3</td><td></td><td></td></tr><tr><td>58.1 LTT</td><td></td><td>50% (7/14)</td><td>100%</td><td>100% 5.3±3.1</td><td>39.2±8.0</td><td>92.5 100%</td><td>100%</td><td>86%</td><td>71%</td><td>9.8±11.5</td><td>27.1±14.7 85.8</td><td>100%</td><td>100%</td></tr><tr><td>15%</td><td>15%</td><td>Max conformal (MC)</td><td>100% (14/14)</td><td>57%</td><td>14.2±7.3</td><td>64.8±9.0</td><td>88.1</td><td></td><td>14%</td><td></td><td>25.8±12.3</td><td>50.7±15.7 77.5</td><td></td><td></td></tr><tr><td rowspan="3">20%</td><td>一</td><td>一 Functional conformal (FC)</td><td>100% (14/14)</td><td>64%</td><td>13.2±6.1</td><td>40.0±13.9</td><td>88.9</td><td></td><td>29%</td><td>一</td><td>21.9±11.9</td><td>32.8±18.5 81.4</td><td></td><td></td></tr><tr><td>35%</td><td>70.7 LTT</td><td>50% (7/14)</td><td>100%</td><td>100% 10.4±3.5</td><td>58.1±5.4</td><td>89.7 100%</td><td>100%</td><td>71%</td><td>57%</td><td>15.5±10.4</td><td>37.0±16.4</td><td>81.1 100%</td><td>100%</td></tr><tr><td></td><td>Max conformal (MC)</td><td>100% (14/14)</td><td>57%</td><td>19.2±7.2</td><td>72.9±5.8</td><td>85.5</td><td></td><td>21%</td><td></td><td>37.2±13.8 61.8±13.9</td><td>73.6</td><td>一</td><td></td></tr><tr><td rowspan="3">25%</td><td>一</td><td>Functional conformal (FC)</td><td>100% (14/14)</td><td>79%</td><td>18.0±6.1</td><td>54.4±11.2</td><td>86.6</td><td></td><td>21%</td><td></td><td>30.7±11.1</td><td>47.0±16.7 77.9</td><td>一</td><td></td></tr><tr><td>35%</td><td>65.7 LTT</td><td>57% (8/14)</td><td>100%</td><td>100% 12.4±3.7</td><td>64.5±5.2</td><td>88.8 100%</td><td>100%</td><td>75%</td><td>75%</td><td>22.7±11.6</td><td>49.2±18.9</td><td>79.9 100%</td><td>100%</td></tr><tr><td></td><td>Max conformal (MC)</td><td>100% (14/14)</td><td>57%</td><td>25.1±8.1</td><td>80.0±5.0</td><td>83.1</td><td></td><td>7%</td><td></td><td>48.7±14.3</td><td>72.1±11.8 70.2</td><td></td><td></td></tr><tr><td>30% 45%</td><td></td><td>一 Functional conformal (FC)</td><td>100% (14/14)</td><td>79%</td><td>22.7±7.0</td><td>63.8±12.0</td><td>84.7</td><td></td><td>7%</td><td></td><td>42.5±12.1 57.5±16.1</td><td>74.3</td><td></td><td></td></tr><tr><td rowspan="3">一</td><td></td><td>67.4 LTT</td><td>50% (7/14)</td><td>100%</td><td>100% 15.2±3.5</td><td>68.2±4.6</td><td>87.4 100%</td><td>100%</td><td>86%</td><td>57%</td><td>24.5±6.6</td><td>50.6±12.6</td><td>77.9 100%</td><td>100%</td></tr><tr><td></td><td>Max conformal (MC)</td><td>100% (14/14)</td><td>57%</td><td>28.6±7.9</td><td>83.1±4.5</td><td>81.6</td><td></td><td>0%</td><td></td><td>56.9±13.4</td><td>77.3±10.2</td><td>67.9</td><td></td></tr><tr><td></td><td>Functional conformal (FC)</td><td>100% (14/14)</td><td>71%</td><td>27.8±7.4</td><td>71.2±11.1</td><td>82.5</td><td></td><td>7%</td><td></td><td>54.1±15.3</td><td>67.3±15.0 70.7</td><td></td><td></td></tr><tr><td>40%</td><td>55%</td><td>65.8 LTT</td><td>71% (10/14)</td><td>100%</td><td>100% 24.1±7.3</td><td>78.4±5.7</td><td>83.5 100%</td><td>100%</td><td>60%</td><td>80%</td><td>43.3±14.6</td><td>66.6±13.8 70.5</td><td>80%</td><td>100%</td></tr><tr><td rowspan="3">50%</td><td></td><td>Max conformal (MC)</td><td>100% (14/14)</td><td>64% 64%</td><td>37.6±6.2 38.7±7.3</td><td>87.5±3.1 82.0±8.1</td><td>77.7</td><td></td><td>0%</td><td></td><td>69.8±14.1</td><td>84.9±7.9</td><td>65.0</td><td></td></tr><tr><td>65%</td><td>一 Functional conformal (FC)</td><td>100% (14/14)</td><td></td><td></td><td></td><td>77.8</td><td></td><td></td><td>0%</td><td>74.0±15.0</td><td>80.2±8.1</td><td>65.0</td><td></td></tr><tr><td>64.3 LTT</td><td></td><td>71% (10/14)</td><td>100%</td><td>100% 31.1±5.4</td><td>84.5±3.3</td><td>80.2 100%</td><td>100%</td><td>40%</td><td>90%</td><td>59.2±18.7</td><td>77.6±10.7 67.7</td><td>70%</td><td>60%</td></tr><tr><td rowspan="3"></td><td></td><td>Max conformal (MC)</td><td>100% (14/14)</td><td>71%</td><td>46.1±6.2</td><td>92.1±2.2</td><td>74.6</td><td></td><td>0%</td><td></td><td>79.2±13.2</td><td>89.8±6.8 63.0</td><td></td><td></td></tr><tr><td></td><td>Functional conformal (FC)</td><td>100% (14/14)</td><td>50%</td><td>49.0±7.2</td><td>90.6±3.3</td><td>73.8</td><td></td><td></td><td>0%</td><td>83.1±12.6</td><td>87.2±5.9 62.4</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 8. Balanced top-1 and top-3 failure-mode accuracy (%), pooled over the splits of each kind. Shaded rows: our probe, by labe variant; the oracle rows use the true failure time and are an upper bound. Best in bold, second best in italics, among the other rows.
<table><tr><td rowspan="2"></td><td colspan="8">Top-1 accuracy (%)</td><td colspan="8">Top-3 accuracy (%)</td></tr><tr><td colspan="2">task</td><td colspan="2">signature</td><td colspan="2">skill</td><td colspan="2">Real robot</td><td colspan="2">task</td><td colspan="2">signature</td><td colspan="2">skill</td><td colspan="2">Real robot</td></tr><tr><td>Method</td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td><td></td><td>Seen Unseen</td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td></tr><tr><td>Most common</td><td>14.3</td><td>14.3</td><td>14.3</td><td>14.3</td><td>14.3</td><td>14.3</td><td>20.0</td><td>20.0</td><td>42.9</td><td>42.9</td><td>42.9</td><td>42.9</td><td>43.9</td><td>39.0</td><td>60.4</td><td>32.6</td></tr><tr><td>Random</td><td>14.3</td><td>14.3</td><td>14.3</td><td>14.3</td><td>14.3</td><td>14.3</td><td>20.0</td><td>20.0</td><td>42.9</td><td>42.9</td><td>42.9</td><td>42.9</td><td>42.9</td><td>42.9</td><td>60.0</td><td>60.0</td></tr><tr><td colspan="9">VLA-TUG (Ours), own detection (first alarm)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>onset_pre</td><td>34.2</td><td>25.5</td><td>35.6</td><td>22.6</td><td>33.6</td><td>20.3</td><td>54.3</td><td>21.7</td><td>65.1</td><td>59.9</td><td>69.8</td><td>56.3</td><td>69.4</td><td>56.3</td><td>88.2</td><td>70.4</td></tr><tr><td>onset_traj</td><td>39.3</td><td>25.9</td><td>37.1</td><td>29.9</td><td>39.7</td><td>27.0</td><td>57.2</td><td>15.8</td><td>71.8</td><td>58.8</td><td>71.4</td><td>61.1</td><td>69.8</td><td>59.7</td><td>87.7</td><td>58.6</td></tr><tr><td>onset</td><td>28.0</td><td>19.3</td><td>32.8</td><td>19.5</td><td>25.0</td><td>17.2</td><td>54.9</td><td>13.2</td><td>63.8</td><td>54.6</td><td>59.3</td><td>50.4</td><td>56.0</td><td>48.1</td><td>88.0</td><td>57.8</td></tr><tr><td>onset_post</td><td>31.1</td><td>19.8</td><td>32.1</td><td>22.1</td><td>32.5</td><td>25.2</td><td>56.6</td><td>18.8</td><td>65.3</td><td>55.7</td><td>69.6</td><td>58.0</td><td>65.2</td><td>57.2</td><td>91.4</td><td>54.4</td></tr><tr><td colspan="9">VLA-TUG (Ours), oracle onset (upper bound)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>onset_pre</td><td>41.3</td><td>31.0</td><td>41.0</td><td>30.2</td><td>42.4</td><td>30.0</td><td>68.4</td><td>12.1</td><td>73.0</td><td>65.0</td><td>78.4</td><td>66.8</td><td>72.9</td><td>64.8</td><td>92.5</td><td>57.1</td></tr><tr><td>onset_traj</td><td>36.1</td><td>29.3</td><td>39.8</td><td>29.9</td><td>39.1</td><td>27.4</td><td>66.1</td><td>12.4</td><td>69.5</td><td>59.3</td><td>73.3</td><td>57.1</td><td>71.5</td><td>59.4</td><td>93.8</td><td>53.1</td></tr><tr><td>onset</td><td>37.5</td><td>30.5</td><td>39.7</td><td>29.2</td><td>37.1</td><td>27.1</td><td>70.2</td><td>11.4</td><td>74.2</td><td>67.5</td><td>75.0</td><td>66.9</td><td>70.1</td><td>62.7</td><td>92.6</td><td>60.0</td></tr><tr><td>onset_post</td><td>23.0</td><td>20.0</td><td>28.0</td><td>22.2</td><td>33.0</td><td>21.4</td><td>67.8</td><td>16.3</td><td>64.8</td><td>59.2</td><td>69.6</td><td>56.2</td><td>66.0</td><td>58.8</td><td>94.5</td><td>52.9</td></tr></table>

Calibration. Per target pair, max-conformal takes 0.1 ms, functional conformal 0.7 ms, and LTT 1.3 ms on a CPU. The calibration sets hold 158 to 209 labeled episodes, of which 92 to 120 are failures.

## D.4 Ablations

We ablate the supervision strategy, the auxiliary heads, and the failure time used for evaluation.

Supervision strategy. Table 9 compares our pre-onset label (every step before the failure time is positive) with the window before the onset only, every step of the failed rollout, and the window before the onset together with all steps after it. The pre-onset label is the best, or within the spread across splits, in nearly every setting. On LIBERO-X the alternatives lose 24% to 50% of the detection area on average, and are lower on every kind of held-out split, seen and unseen; labeling the steps after the onset as positive is worst. On LIBERO-10 the differences are smaller (−1% to −14% on average), within the spread across splits for π<sub>0.5</sub>, and clear on seen tasks with GR00T.

Auxiliary heads. We include the mode and CLIP heads because their labels already exist for every failure. We find no consistent benefit from either on detection (Table 9): removing the heads changes the area by less than the spread across splits in most settings. For mode prediction (Table 10), the CLIP head changes top-1 accuracy by less than one point on average, while training the mode head jointly with the failure score adds three to four points over a mode-only probe. The CLIP alignment might help some subpopulations and hurt others, which we leave to future work.

Sensitivity to the failure time. Moving the failure time later makes the metric easier, since more alarms then count as early, so we check how much our results depend on it.

We score the saved scores again with the failure time of every failed episode moved 1 and 2 s later, with the same thresholds and without retraining (Table 11). Every method gains, but late detectors gain more than VLA-TUG, which already alarms early: on unseen LIBERO-X VLA-TUG gains 2.1 to 2.4 points of area over 2 s and SAFE-LSTM 3.5 to 4.0. On LIBERO-X our rank among the methods does not change, and our margins hold, so these results do not depend on the exact failure time. On the real robot, where our lead is a single point, SAFE-LSTM overtakes VLA-TUG at 1 s for the same reason.

## E QUALITATIVE RESULTS

## E.1 Latent Space

VLA-TUG encodes the VLA backbone embedding of each step with an LSTM into a latent state, which three MLP heads read: the failure score, the failure mode, and a CLIP embedding. The following figures show four of these spaces in two dimensions: the backbone embedding itself (the input), the LSTM state, and the hidden layers of the failure score head and of the failure mode head. Each is projected onto its first two principal components, fitted on the prefailure steps of the training episodes only, and drawn for the training episodes and for the unseen tasks of the LIBERO-X NEW-TASK split. The points are the steps up to the failure of failed episodes and all steps of successful episodes; the ellipses are the 1σ ellipses of a Gaussian fitted to each class, drawn over the points. The classes are the outcome (success in blue, failure in red) or the failure mode. The mode head is trained on failures only and has only a mode plot.

![](images/ce49a7b2849fb74f77665643dbb1923a144360c7d9e1b730029d8b9971609de3.jpg)  
Figure 8. VLA backbone embedding, by outcome.

![](images/d826b2b2651a42252c881a0a33d4c11017a66ab20954ee17f659041cf652d027.jpg)  
Figure 9. VLA backbone embedding, by failure mode.

Table 9. Ablations of our probe in the simulated settings: area under TPR early vs. FPR up to 40% (in %), mean ± standard deviation over the splits, and in brackets the change relative to our probe (shaded). Best in bold, second best in italics, within each block.
<table><tr><td></td><td colspan="2">LIBERO-10 π0.5</td><td colspan="2">LIBERO-10 GR0OT</td><td colspan="2">LIBERO-X task</td><td colspan="2">LIBERO-X signature</td><td colspan="2">LIBERO-X skill</td></tr><tr><td></td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td></tr><tr><td colspan="9">Heads (LSTM, pre-onset label)</td><td></td></tr><tr><td>Modes + CLIP (ours)</td><td>75.1±12.5</td><td>47.9±20.9</td><td>70.5±10.4</td><td>47.4±13.5</td><td>67.9±5.0</td><td> $4 0 . 2 \pm 1 . 2$ </td><td>67.0±6.6</td><td>47.0±0.6</td><td>67.8±4.8</td><td>39.0±7.0</td></tr><tr><td>No head</td><td> $8 2 . 3 { \scriptstyle \pm 6 . 4 } \ ( + 1 0 \% )$ </td><td> $5 I . I { \pm } 4 . 0 ~ ( + 7 \% )$ </td><td> $7 0 . 1 { \pm } 1 1 . 0 \ ( - 1 \% )$ </td><td> $5 5 . 1 { \pm } 5 . 5 \ ( + 1 6 \% )$ </td><td> $6 7 . 1 { \pm } 1 . 8 \ ( - 1 \% )$ </td><td> $4 2 . 7 { \scriptstyle \pm 1 . 5 \ ( + 6 \% ) }$ </td><td> ${ 6 8 . 3 2 5 . 7 \ ( + 2 \% ) }$ </td><td> $4 5 . 0 { \pm } 1 . 3 \ ( - 4 \% )$ </td><td> $6 9 . 2 \pm 4 . 4 \ ( + 2 \% )$ </td><td> $4 3 . 6 { \pm } 1 1 . 5 \ : ( + 1 2 \% )$ </td></tr><tr><td>Modes only CLIP only</td><td> $8 0 . 4 { \pm } 8 . 4 \ ( + 7 \% )$ </td><td>48.3±8.8 (+1%)</td><td> $7 0 . 9 { \scriptstyle \pm 8 . 2 } \ ( + 0 \% )$ </td><td> $5 2 . 9 { \scriptstyle \pm 8 . 4 } ( + 1 2 \% )$ </td><td>69.4±1.0 (+2%)</td><td> $3 9 . 9 { \scriptstyle \pm 3 . 8 \ ( - 1 \% ) }$ </td><td> $6 8 . 0 { \pm } 8 . 8 \ ( + 2 \% )$ </td><td> $4 6 . 6 { \pm } 3 . 5 \ ( - 1 \% )$ </td><td>68.3±3.5 (+1%)</td><td>43.7±10.2 (+12%)</td></tr><tr><td></td><td>78.2±11.4 (+4%)</td><td>58.3±15.3 (+22%)</td><td>72.6±4.8 (+3%)</td><td>44.4±12.7 (−6%)</td><td> $6 6 . 1 { \scriptstyle \pm 3 . 1 \ ( - 3 \% ) }$ </td><td> $4 I . 4 2 2 . 4 \ : ( + 3 \% )$ </td><td> $6 7 . 3 { \pm } 8 . 3 \ ( + 1 \% )$ </td><td> $4 6 . 2 { \pm } 2 . 8 \ ( - 2 \% )$ </td><td>67.8±5.5 (+0%)</td><td> $3 8 . 5 { \scriptstyle \pm 9 . 2 \ ( - 1 \% ) }$ </td></tr><tr><td colspan="9">Training strategy (modes + CLIP heads)</td><td></td><td></td></tr><tr><td>onset_pre (ours)</td><td>75.1±12.5</td><td>47.9±20.9</td><td>70.5±10.4</td><td>47.4±13.5</td><td>67.9±5.0</td><td>40.2±1.2</td><td>67.0±6.6</td><td>47.0±0.6</td><td>67.8±4.8</td><td>39.0±7.0</td></tr><tr><td>onset_traj</td><td>77.6±6.5 (+3%)</td><td>52.0±14.9 (+9%)</td><td> $6 4 . 5 { \scriptstyle \pm 2 . 0 \ ( - 9 \% ) }$ </td><td>43.8±10.9 (−8%)</td><td> $4 9 . 7 { \scriptstyle \pm 2 . 4 \ ( - 2 7 \% ) }$ </td><td>38.4±2.9 (−5%)</td><td>48.0±6.8 (−28%)</td><td>31.3±3.1 (−33%)</td><td> $5 2 . 1 { \scriptstyle \pm 4 . 3 \ ( - 2 3 \% ) }$ </td><td>21.6±3.9 (−45%)</td></tr><tr><td>onset</td><td> $7 3 . 3 { \scriptstyle \pm 3 . 5 } \ ( - 2 \% )$ </td><td>45.4±14.6 (−5%)</td><td>60.3±7.7 (−15%)</td><td> $3 9 . 5 { \scriptstyle \pm 1 0 . 7 \ ( - 1 7 \% ) }$ </td><td>48.8±1.4 (−28%)</td><td>39.9±6.6 (−1%)</td><td>47.6±11.8(−29%)</td><td>36.6±5.7 (−22%)</td><td>46.4±5.4 (−32%)</td><td> $2 5 . 5 { \scriptstyle \pm 9 . 0 \ ( - 3 5 \% ) }$ </td></tr><tr><td>onset_post</td><td>73.5±4.3 (−2%)</td><td>50.2±11.0 (+5%)</td><td>37.4±3.0 (-47%)</td><td>41.8±11.8 (-12%)</td><td> $2 4 . 1 { \pm } 2 . 4 \ ( - 6 5 \% )$ </td><td> $3 1 . 3 { \pm } 0 . 9 \ ( - 2 2 \% )$ </td><td>28.3±3.6 (−58%)</td><td>25.0±0.3 (−47%)</td><td>30.6±2.5 (−55%)</td><td> $1 7 . 0 { \pm } 3 . 4 \ ( - 5 6 \% )$ </td></tr></table>

Table 10. What the CLIP head and the score term add to failure-mode prediction: top-1 and top-3 accuracy (in %), mean ± standard deviation over the splits. Our probe is shaded. Best in bold, second best in italics, per column and anchor.
<table><tr><td rowspan="3"></td><td colspan="8">Top-1 accuracy (%)</td><td colspan="8">Top-3 accuracy (%)</td></tr><tr><td colspan="2">task</td><td colspan="2">signature</td><td colspan="2">skill</td><td colspan="2">Real robot</td><td colspan="2">task</td><td colspan="2">signature</td><td colspan="2">skill</td><td colspan="2">Real robot</td></tr><tr><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td></tr><tr><td colspan="2">At the first alarm of our probe</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Score + modes + CLIP (ours)</td><td>41.0±1.4</td><td>27.2±2.5</td><td>37.3±7.2</td><td>23.7±0.7</td><td>42.3±5.3</td><td>32.2±4.3</td><td>57.9±10.6</td><td>61.5±44.0</td><td>76.0±2.6</td><td>65.9±5.7</td><td>74.2±1.0</td><td>62.8±1.8</td><td>74.6±3.4</td><td>63.2±5.6</td><td>93.3±7.3</td><td>100.0±0.0</td></tr><tr><td>Score + modes</td><td>40.6±3.7</td><td>25.8±2.5</td><td>44.0±4.6</td><td>25.6±0.8</td><td>42.5±5.3</td><td>28.0±7.0</td><td>57.4±11.8</td><td>64.1±45.4</td><td>76.4±1.5</td><td>64.0±2.8</td><td>75.6±2.8</td><td>63.3±1.0</td><td>75.7±2.9</td><td>61.9±7.9</td><td>91.9±7.0</td><td>100.0±0.0</td></tr><tr><td>Modes + CLIP</td><td>37.3±5.8</td><td>23.7±1.4</td><td>29.3±11.1</td><td>17.7±9.3</td><td>44.0±6.4</td><td>30.5±5.3</td><td>55.0±14.2</td><td>59.0±42.8</td><td>72.3±7.7</td><td>64.9±3.3</td><td>62.6±16.4</td><td>54.3±13.4</td><td>75.3±5.2</td><td>61.0±5.6</td><td>90.4±7.9</td><td>100.0±0.0</td></tr><tr><td>Modes</td><td>36.5±6.0</td><td>23.7±3.2</td><td>35.7±5.7</td><td>20.6±4.8</td><td>40.9±6.7</td><td>30.1±4.7</td><td>53.8±12.5</td><td>56.4±41.8</td><td>71.9±7.8</td><td>63.6±1.4</td><td>72.2±6.6</td><td>57.3±7.3</td><td>71.5±3.4</td><td>61.6±8.7</td><td>94.7±7.2</td><td>100.0±0.0</td></tr><tr><td colspan="2">At the true failure onset (upper bound)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Score + modes + CLIP (ours)</td><td>49.2±5.7</td><td>40.6±4.5</td><td>47.7±9.0</td><td>40.9±3.1</td><td>48.6±3.5</td><td>39.6±8.0</td><td>61.9±8.4</td><td>23.7±22.0</td><td>80.3±3.6</td><td>73.6±2.6</td><td>81.7±5.8</td><td>77.5±1.5</td><td>77.4±4.0</td><td>74.7±6.7</td><td>95.9±4.6</td><td>62.4±35.1</td></tr><tr><td>Score + modes</td><td>54.7±3.2</td><td>39.6±5.0</td><td>49.1±4.5</td><td>39.0±0.4</td><td>48.4±3.7</td><td>40.4±8.0</td><td>59.4±5.0</td><td>25.7±21.1</td><td>82.5±3.5</td><td>72.6±0.7</td><td>79.6±3.8</td><td>74.5±1.7</td><td>77.9±3.0</td><td>73.4±8.1</td><td>96.4±3.2</td><td>62.8±33.3</td></tr><tr><td>Modes + CLIP</td><td>42.4±5.3</td><td>37.0±7.7</td><td>33.7±15.9</td><td>31.2±17.2</td><td>49.4±5.8</td><td>39.8±6.4</td><td>62.1±8.8</td><td>19.5±19.8</td><td>77.3±10.2</td><td>69.1±5.3</td><td>69.2±15.6</td><td>61.4±23.2</td><td>77.5±4.7</td><td>72.2±5.9</td><td>96.4±3.9</td><td>59.5±32.9</td></tr><tr><td>Modes</td><td></td><td>48.9±5.6 38.1±8.7</td><td>42.7±7.2</td><td></td><td>35.0±12.0 49.8±4.9 40.8±5.5</td><td></td><td>59.4±7.6</td><td>32.4±26.3</td><td>76.4±7.4</td><td>68.8±5.4</td><td>74.2±5.3</td><td>67.4±13.2</td><td>77.4±4.0</td><td>73.2±7.5</td><td>96.4±3.3</td><td>66.1±34.8</td></tr></table>

Table 11. Early-detection area (in %) on the unseen episodes when the failure time is moved later by 0, 1, and 2 s; nothing is retrained. Ours is shaded. Best in bold, second best in italics.
<table><tr><td></td><td colspan="3"> $\mathrm { L I B E R O  – 1 0 \ \pi _ { 0 . 5 } }$ </td><td colspan="3">LIBERO-10 GR0OT</td><td colspan="3">LIBERO-X task</td><td colspan="3">LIBERO-X signature</td><td colspan="3">LIBERO-X skill</td><td colspan="3">Real robot</td></tr><tr><td>Method</td><td>0s</td><td>+1 s</td><td>+2s</td><td>0s</td><td>+1 s</td><td>+2s</td><td>0s</td><td>+1s</td><td>+2s</td><td>0s</td><td>+1 s</td><td>+2s</td><td>0s</td><td>+1 s</td><td>+2s</td><td>0s</td><td>+1 s</td><td>+2s</td></tr><tr><td>SAFE-LSTM</td><td>34.9</td><td>37.2</td><td>39.2</td><td>51.2</td><td>53.1</td><td>55.0</td><td>40.4</td><td>42.1</td><td>44.4</td><td>41.2</td><td>42.8</td><td>44.9</td><td>30.1</td><td>31.7</td><td>33.6</td><td>25.1</td><td>26.8</td><td>28.9</td></tr><tr><td>SAFE-MLP</td><td>39.9</td><td>42.1</td><td>43.9</td><td>39.2</td><td>40.6</td><td>43.9</td><td>42.3</td><td>43.3</td><td>44.7</td><td>46.2</td><td>47.2</td><td>47.9</td><td>23.5</td><td>24.3</td><td>25.7</td><td>20.2</td><td>21.4</td><td>22.8</td></tr><tr><td>STAC</td><td>36.4</td><td>43.0</td><td>53.2</td><td>17.5</td><td>22.0</td><td>27.6</td><td>28.0</td><td>29.5</td><td>32.2</td><td>28.7</td><td>30.6</td><td>34.0</td><td>21.5</td><td>22.6</td><td>25.1</td><td></td><td></td><td></td></tr><tr><td>ActProbe</td><td>26.6</td><td>31.0</td><td>37.2</td><td>30.1</td><td>33.3</td><td>38.6</td><td>22.7</td><td>24.7</td><td>27.5</td><td>13.9</td><td>16.1</td><td>18.8</td><td>13.6</td><td>14.3</td><td>15.7</td><td></td><td></td><td></td></tr><tr><td>FailureSpot</td><td>36.2</td><td>39.4</td><td>42.7</td><td>44.7</td><td>46.5</td><td>46.6</td><td>20.4</td><td>20.9</td><td>21.7</td><td>26.2</td><td>27.2</td><td>29.1</td><td>19.6</td><td>20.8</td><td>22.8</td><td></td><td></td><td></td></tr><tr><td>Mahalanobis</td><td>40.2</td><td>42.1</td><td>45.9</td><td>29.2</td><td>30.4</td><td>32.6</td><td>30.6</td><td>31.7</td><td>34.0</td><td>22.3</td><td>23.1</td><td>24.4</td><td>15.9</td><td>16.6</td><td>17.4</td><td>19.6</td><td>19.7</td><td>19.8</td></tr><tr><td>Euclidean</td><td>24.2</td><td>28.2</td><td>31.2</td><td>23.1</td><td>25.4</td><td>30.4</td><td>28.5</td><td>29.7</td><td>31.1</td><td>19.7</td><td>20.8</td><td>22.5</td><td>19.0</td><td>20.0</td><td>21.4</td><td>8.3</td><td>9.7</td><td>12.2</td></tr><tr><td>PCA-KMeans</td><td>12.9</td><td>12.9</td><td>13.8</td><td>20.0</td><td>20.0</td><td>20.0</td><td>25.0</td><td>25.3</td><td>25.9</td><td>32.5</td><td>34.0</td><td>36.4</td><td>20.7</td><td>21.5</td><td>23.1</td><td>19.4</td><td>19.9</td><td>20.3</td></tr><tr><td>VLA-TUG (Ours)</td><td>47.9</td><td>52.5</td><td>55.4</td><td>47.4</td><td>49.1</td><td>49.3</td><td>40.2</td><td>41.3</td><td>42.3</td><td>47.0</td><td>48.2</td><td>49.1</td><td>39.0</td><td>40.5</td><td>41.4</td><td>25.8</td><td>26.3</td><td>26.5</td></tr></table>

![](images/737de2a259c417f65972376b3b66a61cddda62d55c0c5ce2113594db8f9c8dc9.jpg)  
Figure 10. Probe LSTM state, by outcome.

![](images/1c7771d9896a17e1d76979cd413cb3f6ac85c6e7cf9b3a06ee8bbe96c74c9579.jpg)  
Figure 11. Probe LSTM state, by failure mode.

![](images/d755bdbbda459d31a8f963c0190876d3850223d20a8705ef506149622d1e1b67.jpg)

![](images/acfe4876b505b82952f61e872a424def8eb48c544c86e12affd35cb42307ad43.jpg)

![](images/0e3c63d4d53e41a784f24af78e5b7e36235efbe7cfe596af1e3f22d00f4cad01.jpg)  
Figure 12. Failure score head layer, by outcome.  
Figure 13. Failure score head layer, by failure mode.

## E.2 Latent Trajectories

To see how an episode moves through these spaces, Figure 15 and Figure 16 follow one success and one failure step by step, for episodes of seen and of unseen tasks. Each panel shows one space in its first two principal components: (a) the VLA backbone embedding, (b) the probe’s LSTM state, and the hidden layers of (c) the failure score head and (d) the failure mode head. The points and ellipses are those of the episodes of the set shown (the training episodes, or the unseen tasks), grouped by outcome and failure mode, and the mode head is shown on failures only. The principal components are always those fitted on the training episodes. A trajectory starts at the ring; a failure is dashed and ends at the cross, at its failure step.

## E.3 Steering Demonstrations

We use steering only to show how the signal of VLA-TUG could be used; the heuristics are simple and are not evaluated. When the score crosses the calibrated threshold, the policy is paused, an intervention chosen from the predicted failure mode is applied, and the policy resumes.

Simulation. We implemented three heuristics for LIBERO-X. For a wrong-target or placement/insertion failure, the end effector is displaced by a small distance towards the destination. For a grasp failure, it is displaced laterally, perpendicular to its recent motion. For every other predicted mode, the policy replans from the current state. Figure 17a shows the three on the task “place the white-yellow mug on top of the shelf”: after the alarm, the probe predicts the failure mode, the matching intervention is applied, and the policy resumes and succeeds, whereas without intervention the mug is placed on the table and the task fails.

Real robot. In the task “put the mushroom on the plate”, the gripper misses the mushroom and hits the table without the probe; the probe predicts a grasp failure, and an expert would steer the wrist camera towards the grasp target and resume the policy. In the task “flip the basket upright”, we start with the basket standing on its tall end, outside the training demonstrations; an expert might move the gripper along the −z direction, so that the joint states are in the distribution of the flip motion, as the demonstrations are collected with the basket on its long side (Figure 17b).

![](images/12edc39bb16063a56c13c910bdd8d62183d6505f3b14107a4fd9c4dc63eb3d2b.jpg)  
Figure 14. Failure mode head layer, by failure mode (failures only).

![](images/cb57dcecb61f0cce5d24ec21f13b502a83bdc05a7afc90f23b002806b8bcaa77.jpg)  
Figure 15. Seen tasks (training episodes). A success (green, “pick up the bowl on the plate and place it to the left of the bowl drainer”) and an object displacement failure (red, “place the striped book in the top region of the shelf”).

![](images/62bb8da2452045b05bd81b2c529dd9a9b93dace4a961ce1b0d880d996f04b557.jpg)  
Figure 16. Unseen tasks. A success (green, “place the red bowl in the basket”) and a press failure (red, “turn on the flat stove”).

![](images/5465b9ce25e20844e3e26937b0404dc9ce71a44389db172956662520945f906e.jpg)

(a) Simulation: “place the white-yellow mug on top of the shelf.”  
![](images/25c84fd3b2e93ac243c0816a8aa994627a327fba3bc8daef09970a94d2a92558.jpg)  
(b) Real robot: “flip the basket upright.”  
Figure 17. Steering strategies, in simulation and on the real robot.