# Hierarchical Residual Policy Learning for Real-World Mobile Manipulation with Sparse Human Guidance

Zhefei Gong<sup>1∗</sup> Hefeifei Jiang<sup>2</sup> Zehao Wang<sup>3</sup> Jiachen Li<sup>3†</sup>

<sup>1</sup>Northeastern University <sup>2</sup>University of California, Riverside <sup>3</sup>Georgia Institute of Technology

Project Website: https://hire-moma.github.io/

Abstract: Mobile Manipulation (MoMa) is a cornerstone of autonomous robotic systems, yet real-world fine-tuning remains a formidable challenge because of high-dimensional action spaces and intricate coupling between heterogeneous modules. In this paper, we propose HiRe-MoMa (Hierarchical Residual Mobile Manipulation), a systematic framework for stable and sample-efficient real-world fine-tuning of complex MoMa systems. At its core, HiRe-MoMa adopts a Hierarchical Residual Architecture that fine-tunes a frozen base policy. A Policy-Agnostic Residual learns additive corrections atop the base policy, while Hierarchical Decoupling splits the residual into mobile-base and manipulator-arm heads with unidirectional conditioning, mitigating gradient interference and aligning corrections with the platform’s physical hierarchy. To improve sample efficiency in data-expensive real-world settings, we propose Expert-Guided Value Alignment, which leverages sparse human interventions with under 20% training coverage to reshape the critic’s value landscape, effectively guiding autonomous exploration through expert priors. Extensive real-world experiments on the TIAGo++ platform demonstrate that HiRe-MoMa consistently enhances task performance with only around 30 minutes of online interaction, regardless of the base policy architecture. To our knowledge, this is among the first frameworks to enable efficient on-robot fine-tuning for mobile manipulation in the real world, providing a practical paradigm for deploying complex robotic systems in dynamic, humancentric environments. Further details can be found on our project website.

Keywords: Real-world reinforcement learning, Human-robot interaction

## 1 Introduction

Mobile manipulation is a key step toward multi-purpose robotic systems that operate in humancentric environments. Performing long-horizon and multi-object tasks, from domestic chores to industrial logistics, requires the coordinated operation of heterogeneous modules such as mobile bases, robotic arms, and sensory heads. Despite its potential, autonomous mobile manipulation remains challenging due to high-dimensional action spaces, tightly coupled inter-module dynamics, and the non-stationarity of real-world deployment. Recent advancements have leveraged Vision-Language Models (VLMs) for semantic reasoning [1, 2, 3, 4, 5], generative modeling for robust imitation learning [6, 7], and simulation-to-real reinforcement learning for skill acquisition [8]. However, a single-shot trained policy often fails to generalize to the stochasticity of diverse real-world deployment scenarios. Post-training or fine-tuning is thus essential to bridge the performance gap. While fine-tuning has seen success in isolated manipulation or locomotion tasks [9, 10], its application to the coupled, high-dimensional domain of mobile manipulation remains largely unexplored.

![](images/f9ed802d3c67e0e1b7ff40121de9fb53fa86b9fea052f978683fc716aabec8a3.jpg)  
Figure 1: HiRe-MoMa System Overview. (a) Training Paradigm: A real-world dual-unit configuration consisting of an onboard controller for real-time interaction and a computing unit for policy optimization and action prediction. (b) Coordinated Execution: The red dashed lines illustrate the stringent spatiotemporal precision required for base-arm synchronization. In mobile manipulation, marginal tracking deviations are often magnified by kinematic coupling, causing even routine grasping maneuvers to fail.

Real-world fine-tuning, however, is shaped by a series of fundamental trade-offs. Imitation learning (IL), particularly DAgger-based approaches [11], offers a straightforward optimization objective but is constrained by inherent expert data scarcity and a lack of autonomous exploration, which often leads to limited generalization on out-of-distribution states and compounding errors at deployment. Reinforcement learning (RL) enables broader state-space exploration [12] but faces significant obstacles when paired with modern generative base policies, whose intractable log-likelihoods preclude standard policy gradient methods [13]. Residual RL offers a promising and practical middle ground by learning additive corrections atop frozen base policies for model-agnostic adaptation [14, 15], yet extending it to mobile manipulation introduces a distinct challenge: inter-module coupling. As shown in Fig. 1, a task as routine as grasping becomes difficult in the MoMa context: it requires tight spatiotemporal coordination between the mobile base and the arm, and any misalignment in the base’s approach rapidly propagates as a sizable pose error at the manipulator. Without explicit structural priors, the high-dimensional action space and non-stationary inter-module dynam ics often hinder convergence and ultimately prevent the precision required for physical interaction.

In this paper, we propose HiRe-MoMa (Hierarchical Residual Mobile Manipulation), a framework for real-world fine-tuning of mobile manipulation systems, built around three components. First, our Policy-Agnostic Residual Learning paradigm adds a learnable residual atop a frozen base policy, sidestepping the intractable log-likelihoods of generative architectures and allowing the same procedure to refine diverse pre-trained policies. Second, to address this base-arm coupling specific to mobile manipulation, we introduce a Hierarchical Decoupling structure with unidirectional conditioning: the manipulator’s residual head is conditioned on the base’s correction, reflecting the system’s physical hierarchy. Third, an Expert-Guided Value Alignment loss leverages sparse human interventions to reshape the critic’s value landscape toward expert-preferred behaviors, improving sample efficiency without overriding RL exploration. HiRe-MoMa is realized as an asynchronous actor-learner system on a single robot: an onboard rollout worker executes the policy while a central learner optimizes the residual on a workstation, enabling on-robot fine-tuning in the real world.

## Overall, our main contributions are as follows:

1) Systematic Fine-tuning Framework: We present an asynchronous, model-agnostic framework for the real-world fine-tuning of mobile manipulation systems, enabling efficient adaptation in this high-dimensional and coupled robotic domain with 25–40 minutes of online interaction. 2) Hierarchical Residual Architecture: We introduce independent residual policies for distinct modules, coupled with a unidirectional conditioning mechanism to ensure stable and logically coherent coordinated control. 3) Expert-Guided Alignment: We develop an auxiliary human intervention alignment loss that leverages sparse expert guidance (≤20% of collected data) to improve sample efficiency and accelerate convergence. 4) Empirical Validation on Diverse MoMa Tasks: We validate HiRe-MoMa on three real-world TIAGo++ tasks of increasing complexity, demonstrating consistent improvements across two distinct base policy architectures (Diffusion Policy and ACT).

![](images/aa0e586e4b84704de984e1df2c48c1e3a60c58a3f41f8464cfa8a77023cd024b.jpg)  
Figure 2: HiRe-MoMa Framework enables policy-agnostic real-world fine-tuning of mobile manipulators through an asynchronous residual reinforcement learning paradigm. (Right) Shared-Control Rollout: an onboard rollout worker augments a frozen base policy with learnable residuals; sparse human interventions during critical or hazardous phases are stored in an expert buffer. (Left) Asynchronous Central Learner: the resid ual policy is optimized along two pillars: a) a Hierarchical Residual Architecture (Sec. 4.1) that decouples the residual into mobile and manipulator heads with unidirectional conditioning, embedding the platform’s physi cal hierarchy directly into the policy; and b) an Expert-Guided Value Alignment loss (Sec. 4.2) that fuses sparse environment rewards with sparse human corrections to reshape the critic toward expert-preferred behaviors.

## 2 Related Work

Real-World Reinforcement Learning offers a principled path to bridging the sim-to-real gap [16], with sample efficiency as the central challenge. The SERL framework [17] demonstrated the efficacy of off-policy RL for robot fine-tuning, inspiring subsequent residual correction methods [18, 19]. Existing residual and HIL systems [20, 18, 14, 15, 19] fine-tune fixed-base arms, where base–arm gradient interference does not arise; HiRe-MoMa instead targets policy-agnostic, on-robot finetuning of a whole-body mobile manipulator, without simulation-pre-trained latent action spaces [21].

Policy Post-Training of base policies [22, 23] via RL is hampered by intractable log-likelihoods, motivating surrogate objectives [13, 24, 25, 26] or policy-agnostic RL [27, 18]. Residual RL offers a practical middle ground by learning additive corrections atop frozen policies [19, 28, 29, 30, 31], with human-in-the-loop variants [32, 33, 34] further mitigating distributional drift. Yet simple residuals cannot address the cross-module coupling inherent in MoMa: the spatiotemporal dependency between base and arm demands structured decoupling that flat residual methods do not provide.

Mobile Manipulation poses unique challenges owing to its high-dimensional action space and tight base-arm coupling [6, 35]. Most RL methods still rely on simulation-pre-trained priors [8, 36] or expensive skill-discovery phases [21, 37] to bridge the sim-to-real gap. HiRe-MoMa instead operates directly in the raw action space of any pre-trained policy, modeling hierarchical inter-module dependencies under a sparse success-only reward without manual reward shaping.

Human-in-the-Loop corrections serve as safety buffers [20, 9], imitation data sources [38], or auxiliary supervision signals [39, 40, 41] in robot RL. Beyond these roles, interventions implicitly encode expert value preferences [9]; HiRe-MoMa exploits this by treating corrections as functional constraints on the critic’s value surface, directly reshaping the value landscape to reflect expert priorities. A broader discussion of related work on these four topics is provided in App. E.

## 3 Problem Formulation

We study the problem of fine-tuning pre-trained base policies π<sup>base</sup> [42, 22, 43] for complex robotic tasks in unstructured real-world environments. While Supervised Fine-Tuning (SFT) via imitation learning offers a stable convergence path, its efficacy is fundamentally bounded by the coverage and quality of pre-collected demonstrations, which are typically scarce and sub-optimal in practice. Conversely, Reinforcement Learning (RL) enables the agent to extend beyond the constraints of static datasets through active exploration, fostering improved robustness and task proficiency. This online refinement is especially important for high-dimensional systems such as mobile manipulators, where conventional zero-shot transfer leaves a persistent deployment-time gap. We formalize the fine-tuning process as a Partially Observable Markov Decision Process (POMDP), defined by the tuple $\mathcal { M } = ( \mathcal { S } , \mathcal { A } , \mathcal { O } , \mathcal { P } , R , \gamma )$ . S represents the underlying hidden environment states, and $A \subset \mathbb { R } ^ { d }$ denotes the high-dimensional continuous action space (e.g., joint velocities or Cartesian poses). The observation space O comprises multi-modal sensory inputs, including RGB images and robot proprioception, while the transition dynamics $\mathcal { P } ( s ^ { \prime } | s , a )$ are governed by real-world physical interactions. To minimize reward engineering and remain task-agnostic, we adopt a sparse binary reward function $R ( s , a ) \in \{ 0 , 1 \}$ , where 1 denotes task success and 0 otherwise. The objective is to refine the policy π to maximize the expected cumulative discounted return: $J ( \pi ) = \mathbb { E } _ { \tau \sim \pi } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } R ( s _ { t } , a _ { t } ) \right]$ where $\gamma \in ( 0 , 1 ]$ is the discount factor. This formulation drives policy refinement toward higher task success rates and improved temporal efficiency under real-world execution.

## 4 Method

To facilitate efficient fine-tuning for high-dimensional mobile manipulation, we propose HiRe-MoMa (Hierarchical Residual Mobile Manipulation), a model-agnostic framework designed for stable real-world adaptation via human-robot interaction $( \mathrm { F i g } . 2 )$ . The framework consists of three components: (i) a policy-agnostic residual that learns additive corrections atop a frozen prior π<sup>base</sup> without structural assumptions on the base policy; (ii) a hierarchical decoupling mechanism that mitigates optimization interference between heterogeneous modules (e.g., mobile base and manipulator arm) by enforcing a unidirectional dependency aligned with the system’s physical hierarchy; (iii) an expert-guided value alignment loss that leverages sparse human interventions to reshape the critic’s value landscape, improving sample efficiency under safe real-world exploration. We detail each component below; full implementation details are provided in App. D.

## 4.1 Hierarchical Residual Architecture

Policy-Agnostic Residual Learning. Fine-tuning pre-trained policies via reinforcement learning often entails restrictive structural assumptions, particularly when involving intractable log-likelihood estimations [13, 25]. To achieve structural agnosticism across arbitrary base policies $\pi ^ { \mathrm { b a s e } }$ , we introduce an auxiliary residual policy $\pi _ { \theta } ^ { \mathrm { r e s } }$ that learns to predict additive corrections atop a frozen prior. Given the high sample complexity of real-world interactions, we adopt an off-policy, value-based paradigm [20] utilizing Temporal Difference (TD) learning [44]. To stabilize value estimation and mitigate erratic exploration, $Q _ { \phi }$ is optimized via a conservative target derived from the minimum of a critic ensemble.

$$
\begin{array} { r } { Q _ { \phi } ( \mathbf { o } _ { t } , \mathbf { a } _ { t } ) \xleftarrow { } r _ { t } + \gamma \mathbb { E } _ { \mathbf { o } _ { t + 1 } \sim \mathcal { P } } \left[ Q _ { \phi ^ { \prime } } ( \mathbf { o } _ { t + 1 } , \mathbf { a } _ { t + 1 } ^ { \mathrm { e x e } } ) \right] , \quad \mathrm { w h e r e } \quad \mathbf { a } _ { t + 1 } ^ { \mathrm { e x e } } = \mathbf { a } _ { t + 1 } ^ { \mathrm { b a s e } } + \pi _ { \theta } ^ { \mathrm { r e s } } ( \mathbf { o } _ { t + 1 } , \mathbf { a } _ { t + 1 } ^ { \mathrm { b a s e } } ) . } \end{array}\tag{1}
$$

The residual policy $\pi _ { \theta } ^ { \mathrm { r e s } }$ is optimized to maximize the expected return while remaining localized around the base policy’s trajectory via entropy regularization H [45]:

$$
\mathbb { E } _ { \pi _ { \theta } ^ { \mathrm { r e s } } } [ Q _ { \phi } ( \mathbf { o } _ { t } , \mathbf { a } _ { t } ^ { \mathrm { e x e } } ) ] + \alpha \mathcal { H } ( \pi _ { \theta } ^ { \mathrm { r e s } } ( \cdot | \mathbf { o } _ { t } ) ) , \quad \mathrm { w h e r e } \quad \mathbf { a } _ { t } ^ { \mathrm { e x e } } = \mathbf { a } _ { t } ^ { \mathrm { b a s e } } + \pi _ { \theta } ^ { \mathrm { r e s } } ( \mathbf { o } _ { t } , \mathbf { a } _ { t } ^ { \mathrm { b a s e } } ) .\tag{2}
$$

where α is an adaptive temperature parameter. To stabilize exploration and prevent corrections from exceeding the residual module’s capacity, we constrain $\| \mathbf { a } ^ { \mathrm { r e s } } \| \leq \epsilon$ with $\epsilon \in \mathsf { \Gamma } ( 0 , 1 ]$ . The resulting residual policy $\pi ^ { \mathrm { r e s } }$ remains agnostic to $\pi ^ { \mathrm { b a s e } 9 } \mathrm { s }$ architecture, providing flexibility across heterogeneous robotic platforms (Alg. 1, Rollout Worker and Critic Update blocks). This module also serves as the building block for the hierarchical decoupling introduced next.

Hierarchical Decoupling. Mobile manipulation requires high-dimensional coordination across heterogeneous modules (mobile base, manipulator arm, torso). Treating the joint action space as a monolithic output couples the gradients of these modules, which leads to interference and unstable optimization. Mobile manipulation tasks naturally exhibit a unidirectional inter-module dependency: the manipulator’s reaching motion is conditioned on the base’s positioning rather than the reverse. Such structural priors have been shown to align fine-tuned corrections with the system’s physical hierarchy [6, 35]. Leveraging our modular residual design, we first decouple the action space into independent components, denoted as ${ \bf a } ^ { \mathrm { r e s } } = [ { \bf a } _ { \mathrm { m o } } ^ { \mathrm { r e s } } , { \bf a } _ { \mathrm { m a } } ^ { \mathrm { r e s } } ]$ for the mobile base and manipulator, respectively. This decomposition mitigates the gradient interference inherent in highdimensional RL by reducing the cross-module optimization conflict, which is critical for training stability. We then encode this dependency through unidirectional conditioning at each timestep t:

$$
\begin{array} { r } { \mathbf { a } _ { t , \mathrm { m o } } ^ { \mathrm { r e s } } = \pi _ { \theta _ { \mathrm { m o } } } ^ { \mathrm { r e s } } ( \mathbf { o } _ { t } , \mathbf { a } _ { t , \mathrm { m o } } ^ { \mathrm { b a s e } } ) \to \mathbf { a } _ { t , \mathrm { m a } } ^ { \mathrm { r e s } } = \pi _ { \theta _ { \mathrm { m a } } } ^ { \mathrm { r e s } } ( \mathbf { a } _ { t , \mathrm { m o } } ^ { \mathrm { r e s } } , \mathbf { o } _ { t } , \mathbf { a } _ { t , \mathrm { m a } } ^ { \mathrm { b a s e } } ) . } \end{array}\tag{3}
$$

By treating the mobile correction $\mathbf { a } _ { t , \mathrm { m o } } ^ { \mathrm { r e s } }$ as an upstream conditioning context for the manipulator policy $\pi _ { \theta _ { \mathrm { m a } } } ^ { \mathrm { r e s } }$ and its corresponding critic $Q _ { \phi _ { \mathrm { m a } } }$ , we enforce a unidirectional structural constraint that mirrors the physical coupling of the system. This design mitigates the non-stationarity of joint MoMa training by aligning arm refinements with the platform’s evolving coordinate frame, yielding more coherent coordination and stable convergence (Alg. 2, highlighted in light blue; full pseudocode in App. B). The conditioning chain naturally extends to deeper hierarchies, accommodating additional degrees of freedom (DoF) or specialized modules as task complexity grows.

## 4.2 Expert-Guided Value Alignment

The high dimensionality and exploration cost of real-world mobile manipulation pose significant challenges for vanilla RL, even with constrained residual spaces. To accelerate convergence and enhance safety, we integrate human interaction into the training loop via a shared-control paradigm. Human experts can intervene during critical or high-risk task stages to provide corrective guidance, which is stored in a dedicated intervention buffer. During training, we perform balanced sampling [20] from both the autonomous exploration buffer and the human intervention buffer to align the agent’s learning with expert priors. Rather than direct action imitation (e.g., Behavior Cloning), which is sensitive to teleoperation noise, we use human corrections to reshape the value landscape. This corresponds to the Expert-Guided Critic Alignment block in Alg. 1 (lines 28–29, highlighted in light blue): an auxiliary supervised objective on $Q _ { \phi }$ that provides a stable, conservative target [39]. The objective minimizes the following preference loss:

$$
\lambda _ { \mathrm { i m i } } \mathbb { E } _ { \mathbf { o } _ { t } , \mathbf { a } _ { t } ^ { \mathrm { r e s } } , \mathbf { a } _ { t } ^ { \mathrm { h } } } \left[ \log \left( 1 + \exp \left( Q _ { \phi } ( \mathbf { o } _ { t } , \mathbf { a } _ { t } ^ { \mathrm { r e s } } ) - Q _ { \phi } ( \mathbf { o } _ { t } , \mathbf { a } _ { t } ^ { \mathrm { h } } ) \right) \right) \right] ,\tag{4}
$$

where $\mathbf { a } _ { t } ^ { \mathrm { h } }$ represents the expert correction from the human-intervened policy $\pi ^ { \mathrm { r e s , h } }$ , and $\lambda _ { \mathrm { i m i } }$ scales the auxiliary supervision. This formulation encourages the critic to be optimistic toward human actions while remaining pessimistic toward the agent’s autonomous exploration. By embedding human preferences directly into the proxy value function, the agent aligns with expert priors more efficiently, leading to faster and more stable convergence in complex environments.

## 5 Experiments

We evaluate HiRe-MoMa through real-world experiments on the TIAGo++ mobile manipulator, designed to isolate the contribution of each method component and to test adaptability across basepolicy architectures and long-horizon tasks. Our study addresses the following Research Questions: [RQ1] Performance & Generality: Does HiRe-MoMa outperform existing residual refinement baselines across diverse architectures in a model-agnostic manner?

[RQ2] Hierarchical Architecture: Does Hierarchical Decoupling (splitting base/arm heads with unidirectional conditioning) improve success rate and reduce episode length on MoMa tasks?

[RQ3] Value Alignment: Does the Expert-Guided Value Alignment loss further improve sample efficiency and final performance on top of Hierarchical Decoupling?

[RQ4] Interpretability: Can the predicted residual corrections be qualitatively validated through visualization to ensure they provide physically meaningful refinements?

Table 1: Comparison with Fine-Tuning Baselines. All methods use a frozen DP as the base policy. +HG-DAgger [32] overfits to sparse human corrections, and +Flat Residual (equivalent to HIL-SERL [20] Residual) is limited by inter-module gradient interference in its monolithic action space. The full HiRe-MoMa consis tently outperforms both in success rate and execution efficiency. Metrics are averaged over 20 trials per task.
<table><tr><td rowspan="2">Task</td><td rowspan="2">Training Time</td><td colspan="4">Success Rate (%)</td><td colspan="4">Episode length (steps)</td></tr><tr><td>Base</td><td>+HG-DAgger [32]</td><td>+Flat Residual [20]</td><td>HiRe-MoMa</td><td>Base</td><td>+HG-DAgger</td><td>+Flat Residual</td><td>HiRe-MoMa (Full)</td></tr><tr><td>Place Only</td><td>(mins) 25</td><td>(DP) 50.0</td><td>10.0 (-80%)</td><td>60.0 (+20%)</td><td>(Full) 95.0 (+90%)</td><td>(DP) 342</td><td>[32] 420 (0.81x)</td><td>[20] 329 (1.03x)</td><td>235 (1.46x)</td></tr><tr><td>Sweep Litter</td><td>30</td><td>45.0</td><td>5.0 (-89%)</td><td>40.0 (-11%)</td><td>90.0 (+100%)</td><td>359</td><td>478 (0.75x)</td><td>394 (0.91x)</td><td>271 (1.32x)</td></tr><tr><td>Pick Place</td><td>40</td><td>22.5</td><td>5.0 (-78%)</td><td>27.5 (+22%)</td><td>40.0 (+78%)</td><td>664</td><td>742 (0.89x)</td><td>649 (1.02x)</td><td>552 (1.20x)</td></tr><tr><td>Average</td><td>31.7</td><td>39.2</td><td>6.7 (-83%)</td><td>42.5 (+8%)</td><td>75.0 (+91%)</td><td>455.0</td><td>546.7 (0.82x)</td><td>457.3 (0.99x)</td><td>352.7 (1.29x)</td></tr></table>

Table 2: Build-Up Ablation of HiRe-MoMa Components. Each column adds one method component on top of the previous: +Flat Residual is a single residual head over the joint action space (equivalent to HIL-SERL [20] Residual); +Hierarchical Decoupling splits the residual into base/arm heads with unidirectional conditioning (Sec. 4.1); +Alignment is the full HiRe-MoMa with Expert-Guided Value Alignment (Sec. 4.2). Results show base policy → fine-tuned performance on tasks Place Only and Sweep Litter.
<table><tr><td rowspan="2">Task</td><td rowspan="2">Training Time</td><td colspan="3">Success Rate (%)</td><td colspan="3">Episode length (steps)</td></tr><tr><td>+Flat</td><td>+Hierarchical</td><td>+Alignment</td><td>+Flat</td><td>+Hierarchical</td><td>+Alignment</td></tr><tr><td>Place Only</td><td>(mins) 25</td><td>Residual 50→60</td><td>Decoupling 50→85</td><td>(Full) 50→95</td><td>Residual 342→329</td><td>Decoupling 342→283</td><td>(Full) 342→235</td></tr><tr><td>Sweep Litter</td><td>30</td><td>45→40</td><td>45→85</td><td>45→90</td><td>359→394</td><td>359→304</td><td>359→271</td></tr><tr><td>Average</td><td>27.5</td><td>47.5→50.0</td><td>47.5→85.0</td><td>47.5→92.5</td><td>350.5→361.5</td><td>350.5→293.5</td><td>350.5→253.0</td></tr></table>

![](images/f74c0c52eb78ed30bf4a7bfe4bbfae00dfb9016b108ca51432a04a15d8254486.jpg)  
Figure 3: Real-World Experimental Tasks. Our evaluation encompasses three tasks, Place Only, Sweep Litter, and Pick Place, that demand varying levels of inter-module coordination. These tasks are particularly challenging for mobile manipulators as they require the simultaneous optimization of base positioning and arm reaching, where minor misalignments in the platform’s pose propagate as significant errors at the end-effector.

## 5.1 Experimental Setup

Platform. We evaluate HiRe-MoMa on the TIAGo++ mobile manipulator with the torso and head fixed, leaving a 17-D action space (3-D base velocity; per arm, a 6-D end-effector delta pose and a 1-D gripper), of which each task uses a subset. Human interventions are delivered through a Meta Quest 3 teleoperation interface [46] with absolute priority over the residual policy; for every method, interventions are collected online by the same operator under an identical, method-blind protocol. Following HIL-SERL [20], rollout and learning run asynchronously (Fig. 1; more details in App. C).

Baselines and Ablation Build-Up. We compare against two baselines that share the same residual learning paradigm but differ in supervision: +HG-DAgger [32] learns a residual via MSE on human corrections, and +Flat Residual (equivalent to HIL-SERL Residual [20]) is an off-policy RL baseline operating on a monolithic residual vector. To isolate the contribution of each HiRe-MoMa component, we evaluate the following build-up chain atop a frozen base policy: (i) +Flat Residual: single residual head over the joint action space (no hierarchy); (ii) +Hierarchical Decoupling: splits the residual into independent base/arm heads with unidirectional conditioning (Sec. 4.1); (iii) +Alignment (Full HiRe-MoMa): adds the Expert-Guided Value Alignment loss atop hierarchical decoupling (Sec. 4.2). We use Diffusion Policy (DP) [22] as the default base policy and additionally evaluate ACT [47] to verify policy-agnostic generalization, as shown in Tab. 3.

![](images/99ce03203d6ac3bfa1e5f68f50835c61567e625c134d291ea20c0a8f084c3665.jpg)  
Figure 4: Training Dynamics. Comparison across methods over training time, where we construct the figure along the actual training time. (a) Success rate: HiRe-MoMa(full) attains a consistently higher success rate throughout training, validating the effectiveness of our design combining hierarchical residual architecture and alignment. (b) Human intervention steps: HiRe-MoMa requires fewer human interventions and converges faster than the baselines, and the comparison with and without alignment (+Hierarchical Decoupling) demonstrates the effectiveness of the extra alignment loss.

Tasks and Protocol. We design three real-world tasks of increasing complexity (Fig. 3): Place Only (single-arm placement), Sweep Litter (dual-arm coordination), and Pick Place (long-horizon base–arm synchronization), with maximum episode lengths of 500, 500, and 800 steps at 10 Hz, beyond which a trial counts as a failure. Pick Place uses a phased success metric (0.5 for grasp, 1.0 for placement); the others use binary success. Each task is evaluated over 20 trials with randomized initialization (identical in training and evaluation), and we report both success rate and episode length. The residual policy and critic are MLPs (hidden 256) trained with off-policy SAC [20]; the residual is constrained by $\lVert \mathbf { a } ^ { \mathrm { r e s } } \rVert \leq \epsilon = 0 . 3$ , and the alignment-loss weight is $\lambda _ { \mathrm { i m i } } = 1 0 ^ { - 3 }$ . Per-task action dimensions, demo counts, and remaining hyperparameters are listed in App. C–D.

## 5.2 Comparative Evaluation: Performance and Generality

As shown in Tab. 1, the full HiRe-MoMa reaches 95% success on Place Only and 90% on Sweep Litter within 25–40 minutes of real-world fine-tuning, and lifts the long-horizon Pick Place from 22.5% (frozen base) to 40% (failure modes in App. C.4). In contrast, +HG-DAgger suffers from overfitting and erratic corrections under purely imitation-based supervision, while +Flat Residual (HIL-SERL Residual [20]) shows only marginal improvement: its monolithic action space cannot resolve the coupled base-arm dynamics. HiRe-MoMa also consistently reduces episode length, indicating improved execution efficiency. To further verify the framework’s ver-

satility, we evaluate HiRe-MoMa using ACT [47] (a Transformer-based policy) as the prior. Results in Tab. 3 show a 94% relative improvement in success rate. While the absolute performance is slightly bounded by the base policy’s initial coverage, consistent with recent findings [15, 28], the consistent gains across disparate architectures (ACT and DP) support RQ1.

Table 3: Policy-Agnostic Generalization. Our HiRe-MoMa yields consistent gains when applied to ACT [47], demonstrating architecture-independent efficacy. While the performance ceiling is inherently tied to the base policy’s quality, the results confirm HiRe-MoMa’s capacity to fine-tune diverse base policies.
<table><tr><td rowspan="2">Task</td><td rowspan="2">Training Time (mins)</td><td colspan="2">Success Rate (%)</td><td colspan="2">Episode length (steps)</td></tr><tr><td>Base (ACT)</td><td>HiRe-MoMa (Full)</td><td>Base (ACT)</td><td>HiRe-MoMa (Full)</td></tr><tr><td>Place Only</td><td>25</td><td>40</td><td>80 (+100%)</td><td>381</td><td>303 (1.26x)</td></tr><tr><td>Sweep Litter</td><td>30</td><td>40</td><td>75 (+88%)</td><td>362</td><td>297 (1.22x)</td></tr><tr><td>Average</td><td>27.5</td><td>40.0</td><td>77.5 (+94%)</td><td>371.5</td><td>300.0 (1.24x)</td></tr></table>

## 5.3 Build-Up Ablation: Hierarchical Decoupling and Value Alignment

Tab. 2 reports each component’s contribution along the build-up chain, and Fig. 4(a) visualizes the corresponding training-time success-rate curves on Place Only and Sweep Litter, where the gain from +Hierarchical Decoupling is clearly visible (line). Effect of Hierarchical Decoupling (RQ2). Moving from +Flat Residual to +Hierarchical Decoupling raises the average success rate from 50.0% to 85.0% and shortens episode length from 361.5 to 293.5 steps. Splitting the residual into base/arm heads with unidirectional conditioning mitigates the gradient interference of a monolithic action space, allowing each sub-policy to converge more effectively, which is a finding consistent with recent studies on modular visuomotor learning [43]. The arm head, conditioned on the base correction, generates refinements that are reactive to the platform’s displacement, mirroring the system’s physical hierarchy and stabilizing coordinated behavior.

To further disentangle the contributions of decomposition and unidirectional conditioning within Hierarchical Decoupling, we additionally evaluate a +Decomposition-Only variant; Fig. 5 indicates that decomposition contributes the majority of the gain, with conditioning adding a smaller, task-dependent improvement (full breakdown in App. A.1). Effect of Expert-Guided Value Alignment (RQ3). Adding the alignment loss on top of hierarchical decoupling (+Alignment, Full HiRe-MoMa) further improves the average success rate to 92.5% and reduces episode length to 253.0 steps, while the human-intervention ratio decreases monotonically over training when the alignment loss is added (Fig. 4(b), line; see also App. D.4). By penalizing the critic for deviating from expert preferences, the value landscape is reshaped to favor concise corrections over redundant RL exploration, enabling the robot to complete tasks with fewer steps and less human supervision.

![](images/28bc7cd131b8857eb7db5d6924ca98bf25aba180d8bab6a83ec82fff380dec20.jpg)  
Figure 5: Disentangling Decomposition and Conditioning. Training-time success-rate curves for +Flat Residual, +Decomposition-Only, and +Hierarchical Decoupling on Place Only and Sweep Litter; see App. A.1.

## 5.4 Qualitative Analysis: Interpretability and Visualization

To probe what the residual policy actually learns, we visualize per-dimension corrections on the Place Only task (Fig. 6), focusing on the base’s forward velocity and the arm’s downward placement motion. The corrections concentrate at critical decision boundaries: even bounded by $\| \mathbf { a } ^ { \mathrm { r e s } } \| \leq \epsilon ,$ they compensate for the base’s inertial drift and the arm’s kinematic inaccuracies, refining the base policy’s coarse trajectory into more precise control signals. The pattern indicates that the learned correction policy is physically interpretable and consistent with the platform’s hierarchy (RQ4).

![](images/71dab4a8a969bf90e1d44c54e674ac10236349fddc6283fd738416835d23e298.jpg)

![](images/cd3bfd7b7fc62a49994d421619910cafa039d4b48016e96cae5bbe1db8886a94.jpg)  
Figure 6: Residual Correction Visualization. Base longitudinal velocity and manipulator vertical position throughout the Place Only task. HiRe-MoMa applies coherent corrections at critical trajectory phases, most notably during base acceleration and the manipulator’s lay-down motion.

## 6 Limitations and Future Work

Our evaluation covers a single platform (TIAGo++) with 20 trials per condition and no seed sweeps, and Pick Place remains bounded by platform hardware. The two residual heads share one sparse reward without explicit hierarchical credit assignment [48, 49], so arm failures and human-assisted successes may bias the base critic; training also needs a small warm-up buffer of successful on-robot rollouts. Finally, each run requires a supervising operator, and the policies are not yet reliable for unsupervised deployment around people. Future work includes contact-rich tasks with multimodal sensing, richer inter-module structures, and validation on more platforms. See App. F for details.

## 7 Conclusion

We presented HiRe-MoMa, a real-world residual fine-tuning framework for mobile manipulation that combines a policy-agnostic residual, hierarchical decoupling with unidirectional conditioning, and an expert-guided value-alignment loss. Across three TIAGo++ tasks, HiRe-MoMa raises average success from 39.2% to 75% within 25–40 min of online interaction, shortens execution by 1.29×, and remains effective across base architectures with less than 20% human intervention.

## References

[1] R. Shah, A. Yu, Y. Zhu, Y. Zhu, and R. Mart´ın-Mart´ın. Bumble: Unifying reasoning and acting with vision-language models for building-wide mobile manipulation. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 13337–13345. IEEE, 2025.

[2] H. Jiang, Y. Chaudhary, Y. Wang, Z. Wang, R. Sharma, M. Mehta, Y. Zhou, L. Sun, Z. Fan, Z. Tu, and J. Li. NavTrust: Benchmarking trustworthiness for embodied navigation. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2026.

[3] X. Zhang, Z. Wang, Z. Li, J. Yao, and J. Li. CommCP: Efficient multi-agent coordination via LLM-based communication with conformal prediction. In 2026 IEEE International Conference on Robotics and Automation (ICRA), pages 8613–8620. IEEE, 2026.

[4] M. Yan, Y. Wang, Z. Liu, and J. Li. RDD: Retrieval-based demonstration decomposer for planner alignment in long-horizon tasks. In Advances in Neural Information Processing Systems, volume 38, pages 52848–52873, 2025.

[5] T. Chakraborty, U. Ghosh, A. E. Gongora, R. Glatt, Y. Dong, J. Li, A. K. Roy-Chowdhury, and C. Song. Toward autonomous laboratory safety monitoring with vision language models: Learning to see hazards through scene structure. arXiv preprint arXiv:2602.00414, 2026.

[6] Y. Jiang, R. Zhang, J. Wong, C. Wang, Y. Ze, H. Yin, C. Gokmen, S. Song, J. Wu, and L. Fei-Fei. BEHAVIOR robot suite: Streamlining real-world whole-body manipulation for everyday household activities. In Proceedings ofThe 9th Conference on Robot Learning, volume 305 of Proceedings ofMachine Learning Research, pages 1246–1281. PMLR, 2025.

[7] U. Ghosh, D. S. Raychaudhuri, J. Li, K. Karydis, and A. Roy-Chowdhury. Robust offline imitation learning from diverse auxiliary data. Transactions on Machine Learning Research, 2025.

[8] J. Hu, P. Stone, and R. Mart´ın-Mart´ın. Causal Policy Gradient for Whole-Body Mobile Manipulation. In Proceedings of Robotics: Science and Systems, Daegu, Republic of Korea, July 2023.

[9] A. Amin, R. Aniceto, A. Balakrishna, K. Black, K. Conley, G. Connors, J. Darpinian, K. Dhabalia, J. DiCarlo, D. Driess, et al. π<sup>∗</sup><sub>0.6</sub>: a VLA that learns from experience. arXiv preprint arXiv:2511.14759, 2025.

[10] L. Smith, J. C. Kew, X. B. Peng, S. Ha, J. Tan, and S. Levine. Legged robots that keep on learning: Fine-tuning locomotion policies in the real world. In 2022 International Conference on Robotics and Automation (ICRA), pages 1593–1599. IEEE, 2022.

[11] S. Ross, G. Gordon, and D. Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics (AISTATS), pages 627–635. JMLR Workshop and Conference Proceedings, 2011.

[12] J. Liu, F. Gao, B. Wei, X. Chen, Q. Liao, Y. Wu, C. Yu, and Y. Wang. What can RL bring to VLA generalization? an empirical study. In Advances in Neural Information Processing Systems, volume 38, pages 107724–107754, 2025.

[13] T. Zhang, C. Yu, S. Su, and Y. Wang. ReinFlow: Fine-tuning flow matching policy with online reinforcement learning. In Advances in Neural Information Processing Systems, volume 38, pages 117701–117738, 2025.

[14] Y. Jiang, C. Wang, R. Zhang, J. Wu, and L. Fei-Fei. TRANSIC: Sim-to-real policy transfer by learning from online correction. In Proceedings ofThe 8th Conference on Robot Learning, volume 270 of Proceedings ofMachine Learning Research, pages 1691–1729. PMLR, 2025.

[15] L. Ankile, A. Simeonov, I. Shenfeld, M. Torne, and P. Agrawal. From imitation to refinement – residual RL for precise assembly. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 1–8. IEEE, 2025.

[16] C. Xu, Q. Li, J. Luo, and S. Levine. RLDG: Robotic Generalist Policy Distillation via Reinforcement Learning. In Proceedings of Robotics: Science and Systems, Los Angeles, CA, USA, June 2025.

[17] J. Luo, Z. Hu, C. Xu, Y. L. Tan, J. Berg, A. Sharma, S. Schaal, C. Finn, A. Gupta, and S. Levine. SERL: A software suite for sample-efficient robotic reinforcement learning. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 16961–16969. IEEE, 2024.

[18] Y. Chen, S. Tian, S. Liu, Y. Zhou, H. Li, and D. Zhao. ConRFT: A reinforced fine-tuning method for VLA models via consistency policy. In Proceedings of Robotics: Science and Systems, Los Angeles, CA, USA, June 2025.

[19] L. Ankile, Z. Jiang, R. Duan, G. Shi, P. Abbeel, and A. Nagabandi. Residual off-policy RL for finetuning behavior cloning policies. In 2026 IEEE International Conference on Robotics and Automation (ICRA), pages 5789–5796. IEEE, 2026.

[20] J. Luo, C. Xu, J. Wu, and S. Levine. Precise and dexterous robotic manipulation via humanin-the-loop reinforcement learning. Science Robotics, 10(105):eads5033, 2025.

[21] J. Hu, P. Stone, and R. Mart´ın-Mart´ın. SLAC: Simulation-pretrained latent action space for whole-body real-world RL. In Proceedings ofThe 9th Conference on Robot Learning, volume 305 of Proceedings of Machine Learning Research, pages 2966–2982. PMLR, 2025.

[22] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, 44(10-11):1684–1704, 2025.

[23] W. Zhao, P. Ding, Z. Min, Z. Gong, S. Bai, H. Zhao, and D. Wang. VLAS: Vision-languageaction model with speech instructions for customized robot manipulation. In International Conference on Learning Representations (ICLR), 2025.

[24] K. Chen, Z. Liu, T. Zhang, Z. Guo, S. Xu, H. Lin, H. Zang, X. Li, Q. Zhang, Z. Yu, et al. $\pi _ { \mathrm { R L } } \colon$ Online RL fine-tuning for flow-based vision-language-action models. arXiv preprint arXiv:2510.25889, 2025.

[25] Y. Wang, L. Wang, Y. Jiang, W. Zou, T. Liu, X. Song, W. Wang, L. Xiao, J. Wu, J. Duan, et al. Diffusion actor-critic with entropy regulator. In Advances in Neural Information Processing Systems, volume 37, pages 54183–54204, 2024.

[26] H. Ma, T. Chen, K. Wang, N. Li, and B. Dai. Efficient online reinforcement learning for diffusion policy. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pages 41837–41853. PMLR, 13– 19 Jul 2025.

[27] M. S. Mark, T. Gao, G. G. Sampaio, M. K. Srirama, A. Sharma, C. Finn, and A. Kumar. Policy agnostic RL: Offline RL and online RL fine-tuning of any class and backbone. arXiv preprint arXiv:2412.06685, 2024.

[28] Z. Gong, S. Lyu, P. Ding, W. Xiao, and D. Wang. Robust online residual refinement via Koopman-guided dynamics modeling. In 2026 IEEE International Conference on Robotics and Automation (ICRA), pages 9034–9041. IEEE, 2026.

[29] Z. Wang, H. Jiang, S. Dong, Y. Wang, H. Qiu, and J. Li. Drive My Way: Preference alignment of vision-language-action model for personalized driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 25204–25214, June 2026.

[30] U. Ghosh, D. S. Raychaudhuri, J. Li, K. Karydis, and A. Roy-Chowdhury. Reducing oracle feedback with vision-language embeddings for preference-based RL. In 2026 IEEE International Conference on Robotics and Automation (ICRA), pages 20829–20836, 2026.

[31] M. Gao, X. Li, Q. Yan, X. Sun, X. Zhang, C.-M. Huang, and J. Li. GUIDES: Guidance using instructor-distilled embeddings for pre-trained robot policy enhancement. In 2026 IEEE International Conference on Robotics and Automation (ICRA), pages 14679–14686. IEEE, 2026.

[32] M. Kelly, C. Sidrane, K. Driggs-Campbell, and M. J. Kochenderfer. HG-DAgger: Interactive imitation learning with human experts. In 2019 International Conference on Robotics and Automation (ICRA), pages 8077–8083. IEEE, 2019.

[33] Z. Hu, R. Wu, N. Enock, J. Li, R. Kadakia, Z. Erickson, and A. Kumar. RaC: Robot learning for long-horizon tasks by scaling recovery and correction. IEEE Transactions on Robotics, 42: 2843–2860, 2026.

[34] H. Kim, K. Lee, J. Park, J. Li, and J. Park. Human implicit preference-based policy finetuning for multi-agent reinforcement learning in USV swarm. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 18653–18659. IEEE, 2025.

[35] S. Chen, J. Liu, S. Qian, H. Jiang, Z. Liu, C. Gu, X. Li, C. Hou, P. Wang, Z. Wang, R. Zhang, and S. Zhang. AC-DiT: Adaptive coordination diffusion transformer for mobile manipulation. In Advances in Neural Information Processing Systems, volume 38, pages 71369–71397, 2025.

[36] Y. Zhang, T. Liang, Z. Chen, Y. Ze, and H. Xu. Catch it! learning to catch in flight with mobile dexterous hands. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 14385–14391. IEEE, 2025.

[37] R. Mendonca, B. Bucher, J. Wang, and D. Pathak. Continuously improving mobile manipulation with autonomous real-world RL. In Proceedings ofThe 8th Conference on Robot Learning, volume 270 of Proceedings of Machine Learning Research, pages 5204–5219. PMLR, 2025.

[38] H. Liu, S. Nasiriany, L. Zhang, Z. Bao, and Y. Zhu. Robot Learning on the Job: Human-inthe-Loop Autonomy and Learning During Deployment. In Proceedings of Robotics: Science and Systems, Daegu, Republic of Korea, July 2023.

[39] Q. Li, Z. Peng, and B. Zhou. Efficient learning of safe driving policy via human-AI copilot optimization. In International Conference on Learning Representations (ICLR), 2022.

[40] Y. Wang, L. Wang, Y. Du, B. Sundaralingam, X. Yang, Y.-W. Chao, C. Perez-D’Arpino,´ D. Fox, and J. Shah. Inference-time policy steering through human interactions. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 15626–15633. IEEE, 2025.

[41] Y. Korkmaz and E. Bıyık. MILE: Model-based intervention learning. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 15673–15679. IEEE, 2025.

[42] Physical Intelligence et al. $\pi _ { 0 . 5 } \colon$ a vision-language-action model with open-world generalization. In Proceedings of The 9th Conference on Robot Learning, volume 305 of Proceedings of Machine Learning Research, pages 17–40. PMLR, 2025.

[43] Z. Gong, P. Ding, S. Lyu, S. Huang, M. Sun, W. Zhao, Z. Fan, and D. Wang. CARP: Visuomotor policy learning via coarse-to-fine autoregressive prediction. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 13460–13470, 2025.

[44] R. S. Sutton. Learning to predict by the methods of temporal differences. Machine Learning, 3(1):9–44, 1988.

[45] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 1861–1870. PMLR, 2018.

[46] S. Dass, W. Ai, Y. Jiang, S. Singh, J. Hu, R. Zhang, P. Stone, B. Abbatematteo, and R. Mart´ın-Mart´ın. TeleMoMa: A modular and versatile teleoperation system for mobile manipulation. In 2nd Workshop on Mobile Manipulation and Embodied Intelligence at ICRA 2024, 2024.

[47] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. In Proceedings of Robotics: Science and Systems, Daegu, Republic of Korea, July 2023.

[48] A. Levy, G. Konidaris, R. Platt, and K. Saenko. Learning multi-level hierarchies with hindsight. In International Conference on Learning Representations (ICLR), 2019.

[49] O. Nachum, S. Gu, H. Lee, and S. Levine. Data-efficient hierarchical reinforcement learning. In Advances in Neural Information Processing Systems, volume 31, 2018.

[50] Y. J. Ma, W. Liang, V. Som, V. Kumar, A. Zhang, O. Bastani, and D. Jayaraman. LIV: Language-image representations and rewards for robotic control. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 23301–23320. PMLR, 2023.

[51] K. Yu, S. Zhang, H. Soora, F. Huang, H. Huang, P. Tokekar, and R. Gao. GenFlowRL: Shaping rewards with generative object-centric flow in visual reinforcement learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 13183–13192, 2025.

[52] J. Zhang, Y. Luo, A. Anwar, S. A. Sontakke, J. J. Lim, J. Thomason, E. Bıyık, and J. Zhang. ReWiND: Language-guided rewards teach robot policies without new demonstrations. In Proceedings of The 9th Conference on Robot Learning, volume 305 of Proceedings of Machine Learning Research, pages 460–488. PMLR, 2025.

[53] A. Sharma, R. Ahmad, and C. Finn. A state-distribution matching approach to non-episodic reinforcement learning. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 19645–19657. PMLR, 2022.

[54] A. Sharma, A. M. Ahmed, R. Ahmad, and C. Finn. Self-improving robots: End-to-end autonomous visuomotor reinforcement learning. In Proceedings of The 7th Conference on Robot Learning, volume 229 of Proceedings of Machine Learning Research, pages 3292– 3308. PMLR, 2023.

[55] Z. Zhou, P. Atreya, Y. L. Tan, K. Pertsch, and S. Levine. AutoEval: Autonomous evaluation of generalist robot manipulation policies in the real world. In Proceedings ofThe 9th Conference on Robot Learning, volume 305 of Proceedings ofMachine Learning Research, pages 1997– 2017. PMLR, 2025.

[56] H. Zhu, A. Gupta, A. Rajeswaran, S. Levine, and V. Kumar. Dexterous manipulation with deep reinforcement learning: Efficient, general, and low-cost. In 2019 International Conference on Robotics and Automation (ICRA), pages 3651–3657. IEEE, 2019.

[57] P. Wu, A. Escontrela, D. Hafner, P. Abbeel, and K. Goldberg. DayDreamer: World models for physical robot learning. In Proceedings of The 6th Conference on Robot Learning, volume 205 of Proceedings of Machine Learning Research, pages 2226–2240. PMLR, 2023.

[58] Z. Hu, A. Rovinsky, J. Luo, V. Kumar, A. Gupta, and S. Levine. REBOOT: Reuse data for bootstrapping efficient real-world dexterous manipulation. In Proceedings of The 7th Conference on Robot Learning, volume 229 of Proceedings of Machine Learning Research, pages 1930–1949. PMLR, 2023.

[59] S. Huang, Z. A. Zhang, T. Liang, Y. Xu, Z. Kou, C. Lu, G. Xu, Z. Xue, and H. Xu. MENTOR: Mixture-of-experts network with task-oriented perturbation for visual reinforcement learning. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 26143–26161. PMLR, 2025.

[60] M. Wu, L. Wu, Y. Wu, W. Huang, H. Fan, Z. Hu, H. Geng, J. Li, J. Ying, L. Yang, Y. Chen, and H. Dong. SimLauncher: Launching sample-efficient real-world robotic reinforcement learning via simulation pre-training. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 7933–7940. IEEE, 2025.

[61] Z. Zhou, A. Peng, Q. Li, S. Levine, and A. Kumar. Efficient online reinforcement learning finetuning need not retain offline data. In International Conference on Learning Representations (ICLR), 2025.

[62] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

[63] A. Bahety, A. Balaji, B. Abbatematteo, and R. Mart´ın-Mart´ın. SafeMimic: Towards safe and autonomous human-to-robot imitation for mobile manipulation. In Proceedings of Robotics: Science and Systems, Los Angeles, CA, USA, June 2025.

[64] L. Gong, F. Bahrani, Y. Zhou, A. Banayeeanzade, J. Li, and E. Bıyık. AutoFocus-IL: VLMbased saliency maps for data-efficient visual imitation learning without extra human annotations. In 2026 IEEE International Conference on Robotics and Automation (ICRA), pages 18665–18672. IEEE, 2026.

[65] S. Nasiriany, A. Maddukuri, L. Zhang, A. Parikh, A. Lo, A. Joshi, A. Mandlekar, and Y. Zhu. RoboCasa: Large-scale simulation of household tasks for generalist robots. In Proceedings of Robotics: Science and Systems, Delft, Netherlands, July 2024.

[66] C. Li, R. Zhang, J. Wong, C. Gokmen, S. Srivastava, R. Mart´ın-Mart´ın, C. Wang, G. Levine, M. Lingelbach, J. Sun, et al. BEHAVIOR-1K: A benchmark for embodied AI with 1,000 everyday activities and realistic simulation. In Proceedings of The 6th Conference on Robot Learning, volume 205 of Proceedings of Machine Learning Research, pages 80–93. PMLR, 2023.

## Appendix

## A Additional Experiments

## A.1 Disentangling Decomposition and Conditioning

The +Hierarchical Decoupling column in Tab. 2 combines two sub-operations introduced in Sec. 4.1. (i) decomposition: splitting the residual into independent base/arm heads $\begin{array} { r l } { ( \mathbf { a } ^ { \mathrm { r e s } } } & { { } = } \end{array}$ $[ \mathbf { a } _ { \mathrm { m o } } ^ { \mathrm { r e s } } , \mathbf { a } _ { \mathrm { m a } } ^ { \mathrm { r e s } } ] )$ ; and (ii) unidirectional conditioning: feeding the base correction into the arm head (Eq. (3)). To isolate their individual contributions, we additionally evaluate a +Decomposition-Only variant in which the base and arm heads are trained independently without inter-module information flow. Tab. 4 reports the results.

Table 4: Decomposition vs. Conditioning within Hierarchical Decoupling. Decomposition alone provides the dominant gain by mitigating gradient interference (+30 pt average success rate). Conditioning adds a targeted improvement (+5 pt on average) that manifests on the coupled dual-arm task (Sweep Litter).
<table><tr><td rowspan="2">Task</td><td colspan="3">Success Rate (%)</td><td colspan="3">Episode Length (steps)</td></tr><tr><td>+Flat</td><td>+Decomposition</td><td>+Hierarchical</td><td>+Flat</td><td>+Decomposition</td><td>+Hierarchical</td></tr><tr><td></td><td>Residual</td><td>Only</td><td>Decoupling</td><td>Residual</td><td>Only</td><td>Decoupling</td></tr><tr><td>Place Only</td><td>50→60 45→40</td><td>50→85</td><td>50→85</td><td>342→329</td><td>342→291</td><td>342→283</td></tr><tr><td>Sweep Litter</td><td>47.5→50.0</td><td>45→75</td><td>45→85</td><td>359→394</td><td>359→322</td><td>359→304</td></tr><tr><td>Average</td><td></td><td>47.5→80.0</td><td>47.5→85.0</td><td>350.5→361.5</td><td>350.5→306.5</td><td>350.5→293.5</td></tr></table>

Decomposition raises the average success rate from 50.0% to 80.0% (+30 pt) and reduces average episode length from 361.5 to 306.5 steps, confirming that the primary benefit stems from eliminat ing gradient interference between heterogeneous modules, in line with prior observations in modular visuomotor learning [43]. Adding unidirectional conditioning yields a further gain to 85.0% (+5 pt) with 13 fewer steps on average. Notably, the conditioning gain is task-dependent: negligible on Place Only (single-arm, 85% → 85%) but pronounced on Sweep Litter (dual-arm, 75% → 85%), where base displacement simultaneously affects both arms’ reachability. This indicates that conditioning provides targeted improvements that scale with inter-module coupling complexity, comple mentary to the broader stabilization achieved by decomposition alone. We therefore view conditioning as a coupling-dependent refinement on top of decomposition rather than the sole source of the gain; both are components of Hierarchical Decoupling.

## A.2 Training Steps vs. Wall-Clock Time

To enable a fair comparison of training efficiency, we align all methods on a common wall-clock budget and report the corresponding training-step progression in Fig. 7. HG-DAgger advances through training steps faster owing to its supervised objective, but this does not translate into higher success rates (Fig. 4(a)), reflecting overfitting to sparse human corrections. Among the RL-based methods, HiRe-MoMa incurs only a modest overhead relative to +Flat Residual [20], roughly 1–2× the per-step compute, since the residual is split into separate base and arm policies, an acceptable cost given the consistent gains in success rate and episode length reported in Tab. 1. The wall-clock gap in Fig. 7 thus has two additive sources: each SAC update back-propagates through a critic ensemble with min-ensemble TD targets and separate actor and temperature updates, making +Flat Residual roughly an order of magnitude slower per step than HG-DAgger’s single regression step, and HiRe-MoMa’s extra 1–2× comes from evaluating its base and arm heads sequentially.

## B Pseudo Code

We provide pseudocode for the full system: Alg. 1 details the SAC-style training loop with the Expert-Guided Value Alignment loss, and Alg. 2 specifies the Hierarchical Decoupling design. The corresponding method derivation appears in Sec. 4.

Algorithm 1 HiRe-MoMa: SAC-Style Residual Policy w/ HIL   
1: Input: Pre-trained base policy $\pi ^ { \mathrm { b a s e } }$ , warm-up buffer $\mathcal { D } _ { \mathrm { w a r m } }$ of successful on-robot rollouts,   
human expert $\pi ^ { \mathrm { r e s , h } }$   
2: Initialize: Residual policy $\pi _ { \theta } ^ { \mathrm { r e s } }$ , Ensemble critics $\boldsymbol { Q } _ { \phi _ { 1 , . . . N } } ,$ , Temperature $\alpha ,$ Target entropy $\bar { \mathcal { H } } . \mathsf { \Omega } \mathsf { \Omega } \mathsf { \Omega }$   
See Alg. 2 for the module-level expansion.   
3: Initialize: Replay buffer $\mathcal { D } _ { \mathrm { r e p l a y } }  \emptyset .$ , and expert buffer $\mathcal { D } _ { \mathrm { e x p e r t } }  \mathcal { D } _ { \mathrm { w a r m } } .$ ▷ Not $\pi ^ { \mathrm { b a s e } 9 } \mathrm { s }$ training   
data; see App. D.3.   
4: Initialize: Target networks $\phi _ { i } ^ { \prime }  \phi _ { i }$   
# — Rollout Worker —   
5: Reset environment and receive initial observation $\mathbf { o } _ { t } .$   
6: for each step t do   
7: Compute base action $\mathbf { a } _ { t } ^ { \mathrm { b a s e } } \sim \pi ^ { \mathrm { b a s e } } ( \mathbf { o } _ { t } ) .$   
8: if Human intervenes then   
9: Obtain residual ${ \bf a } _ { t } ^ { \mathrm { r e s } } = \pi ^ { \mathrm { r e s , h } } ( { \bf o } _ { t } , { \bf a } _ { t } ^ { \mathrm { b a s e } } )$   
10: Flag $\mathbb { I } _ { \mathrm { h } } = 1 .$   
11: else   
12: Sample residual $\mathbf { a } _ { t } ^ { \mathrm { r e s } } \sim \pi _ { \theta } ^ { \mathrm { r e s } } \big ( \mathbf { o } _ { t } , \mathbf { a } _ { t } ^ { \mathrm { b a s e } } \big )$   
13: Flag $\overset { \cdot } { \mathbb { I } } _ { \mathrm { h } } = 0 .$   
14: end if   
15: Execute $\mathbf { a } _ { t } ^ { \mathrm { e x e } } = \mathbf { a } _ { t } ^ { \mathrm { b a s e } } + \mathbf { a } _ { t } ^ { \mathrm { r e s } }$ , observe $( r _ { t } , \mathbf { o } _ { t + 1 } , d _ { t } )$   
16: Store $( \mathbf { o } _ { t } , \mathbf { \bar { a } } _ { t } ^ { \mathrm { b a s e } } , \mathbf { \bar { a } } _ { t } ^ { \mathrm { e x e } } , r _ { t } , \bar { d } _ { t } , \mathbf { o } _ { t + 1 } , \mathbf { a } _ { t + 1 } ^ { \mathrm { b a s e } } )$ in $\mathcal { D } _ { \mathrm { r e p l a y } } .$   
17: if $\mathbb { I } _ { \mathrm { h } } = 1$ then add to $\mathcal { D } _ { \epsilon }$ <sub>xpert</sub>.   
18: end for   
# — Central Learner —   
19: for each update step do   
20: Evenly sample $\mathbf { \bar { \boldsymbol { B } } } _ { \mathrm { r e p l a y } } \sim \mathcal { D } _ { \mathrm { r e p l a y } }$ and $\begin{array} { r } { B _ { \mathrm { e x p e r t } } \sim \mathcal { D } _ { \mathrm { e x p e r t } } . } \end{array}$   
21: # — Actor Update (using $B \dot { = } B _ { \mathrm { r e p l a y } } \cup \dot { B } _ { \mathrm { e x p e r t } } ) \dot { - }$   
22: Optimize θ through $\begin{array} { r } { \big [ \frac { 1 } { N } \sum _ { i } ^ { N } Q _ { \phi _ { i } } \big ( \mathbf { o } _ { t } , \pi _ { \theta } ^ { \mathrm { r e s } } \big ( \mathbf { o } _ { t } , \mathbf { a } _ { t } ^ { \mathrm { b a s e } } \big ) + \mathbf { a } _ { t } ^ { \mathrm { b a s e } } \big ) - \alpha \log \pi _ { \theta } ^ { \mathrm { r e s } } \big ( \mathbf { a } _ { t } ^ { \mathrm { r e s } } | \mathbf { o } _ { t } \big ) \big ] } \end{array}$   
23: Optimize α through $\left[ - \alpha ( \log \pi _ { \boldsymbol { \theta } } ^ { \mathrm { r e s } } ( \mathbf { a } _ { t } ^ { \mathrm { r e s } } | \mathbf { o } _ { t } ) + \bar { \mathcal { H } } ) \right]$   
24: # — Critic Update (using $B ) -$   
25: $\mathbf { a } _ { t + 1 } ^ { \mathrm { e x e } }  \mathbf { a } _ { t + 1 } ^ { \mathrm { b a s e } } + \mathbf { a } _ { t + 1 } ^ { \mathrm { r e s } } = \mathbf { \Breve { a } } _ { t + 1 } ^ { \mathrm { b a s e } } + \pi _ { \theta } ^ { \mathrm { r e s } } ( \mathbf { o } _ { t + 1 } , \mathbf { a } _ { t + 1 } ^ { \mathrm { b a s e } } )$   
26: Compute $y = r _ { t } + \gamma ( 1 - \dot { d _ { t } } )$ [min<sub>i</sub> $\dot { Q } _ { \phi _ { i } ^ { \prime } } \big ( \mathbf { o } _ { t + 1 } , \mathbf { a } _ { t + 1 } ^ { \mathrm { e x e } } \big ) \big ] .$   
27: Update $\phi _ { i }$ by minimizing $\begin{array} { r } { \sum _ { i } ^ { N } \| Q _ { \phi _ { i } } ( \mathbf { o } _ { t } , \mathbf { a } _ { t } ^ { \mathrm { e x e } } ) - y \| ^ { 2 } . } \end{array}$   
28: # — Expert-Guided Critic Alignment (applied to $B _ { \mathrm { e x p e r t } }$ only) —   
29: Additionally optimize $\phi _ { i }$ through $\begin{array} { r } { \sum _ { i } \log ( 1 + \exp ( Q _ { \phi _ { i } } \ \left( \mathbf { o } _ { t } , \pi _ { \theta } ^ { \mathrm { r e s } } ( \mathbf { o } _ { t } , \mathbf { a } _ { t } ^ { \mathrm { b a s e } } ) + \mathbf { a } _ { t } ^ { \mathrm { b a s e } } \right) - } \end{array}$   
$Q _ { \phi _ { i } } \left( \mathbf { o } _ { t } , \mathbf { a } _ { t } ^ { \mathrm { e x e } } ) \right) \big )$ ▷ See Sec. 4.2 for details.   
30: Update target critic networks: $\phi _ { i } ^ { \prime }  \rho \phi _ { i } ^ { \prime } + ( 1 - \rho ) \phi _ { i }$   
31: Synchronize latest policy θ to rollout worker.   
32: end for

Algorithm 2 HiRe-MoMa: Hierarchical Decoupling Design   
1: Initialize: Separate residual policies $\pi _ { \theta _ { \mathrm { m o } } } ^ { \mathrm { r e s } } , \pi _ { \theta _ { \mathrm { m a } } } ^ { \mathrm { r e s } }$ and critics $Q _ { \phi _ { \mathrm { m o } } } , Q _ { \phi _ { \mathrm { m a } } } .$ ▷ Critic ensembles   
omitted for brevity.   
2: Notation: $\mathbf { a } _ { t } = [ \mathbf { a } _ { t , \mathrm { m o } } , \mathbf { a } _ { t , \mathrm { m a } } ] .$   
# — Hierarchical Action Generation —   
3: Mobile Base Residual: $\mathbf { a } _ { t , \mathrm { m o } } ^ { \mathrm { r e s } } \sim \pi _ { \theta _ { \mathrm { m o } } } ^ { \mathrm { r e s } } \bigl ( \mathbf { o } _ { t } , \mathbf { a } _ { t , \mathrm { m o } } ^ { \mathrm { b a s e } } \bigr ) .$   
4: Manipulator Arm Residual: $\begin{array} { r } { \mathbf { a } _ { t , \mathrm { m a } } ^ { \mathrm { r e s } } \sim \pi _ { \theta _ { \mathrm { m a } } } ^ { \mathrm { r e s } } ( \mathbf { a } _ { t , \mathrm { m o } } ^ { \mathrm { r e s } } , \mathbf { o } _ { t } , \mathbf { a } _ { t , \mathrm { m a } } ^ { \mathrm { b a s e } } ) , } \end{array}$   
5: Manipulator Arm Critic: $Q _ { \phi _ { \mathrm { m a } } } \big ( \mathbf { a } _ { t , \mathrm { m o } } ^ { \mathrm { r e s } } , \mathbf { o } _ { t } , \mathbf { a } _ { t , \mathrm { m a } } ^ { \mathrm { e x e } } \big ) .$   
6: Combined Executed Action: $\begin{array} { r } { \mathbf { a } _ { t _ { \cdot } } ^ { \mathrm { e x e } } = \mathbf { a } _ { t } ^ { \mathrm { b a s e } } + [ \mathbf { a } _ { t , \mathrm { m o } } ^ { \mathrm { r e s } } , \mathbf { a } _ { t , \mathrm { m a } } ^ { \mathrm { r e s } } ] . } \end{array}$   
# — Hierarchical Policy Optimization —   
7: Update Base and Arm modules independently with unidirectional flow. ▷ Decouples base-arm   
gradient interference during fine-tuning.   
8: Update $\theta _ { \mathrm { m o } }$ via $Q _ { \phi _ { \mathrm { m o } } }$ using standard gradient.   
9: Update $\theta _ { \mathrm { m a } }$ via $Q _ { \phi _ { \mathrm { m a } } }$ while conditioning on $\mathbf { a } _ { t , \mathrm { m o } } ^ { \mathrm { r e s } } .$

![](images/8569a54d0d8e464e89a153ff8fcfbb7c1a3eb44cd4a10ea3afefe40518ebfb4f.jpg)

![](images/a82c881b2c642e50174a0cbf3cb913f9668733cfdb0188de3cdc2f973d4cee76.jpg)  
+HG-DAgger +Flat Residual +Decomposition Only +Hierarchical Decoupling +HiRe-MoMa (Full)  
Figure 7: Training-Step Progression vs. Wall-Clock Time. HG-DAgger advances through training steps faster than the SAC-based residual methods, but its higher step throughput does not translate into higher success rates (Fig. 4(a)). HiRe-MoMa adds only a modest per-step overhead over +Flat Residual (1–2×) while delivering clear gains in success rate and episode length.

## C Experimental Setup

## C.1 Hardware and Control Interface

We evaluate HiRe-MoMa on the TIAGo++ mobile manipulator, which features two 7-DoF arms, a liftable torso, a wheeled mobile base, and a head equipped with an RGB camera. To manage the complexity of real-world RL, we fix the torso and head poses, leaving a 17-dimensional action space: a 3-D base velocity and, per arm, a 6-D end-effector delta pose plus a 1-D discrete gripper command, of which each task uses a subset (App. C.3). Base and arm commands are output jointly at every step, so the base and arms move simultaneously rather than in alternating phases. Visua feedback is provided by a head-mounted RGB camera at 208 × 160 resolution. For human-in-theloop (HIL) integration, we employ a teleoperation system [46] based on the Meta Quest 3. The expert simultaneously provides velocity commands for the base and end-effector offsets in Euler space, which are then resolved into joint-space deltas via Inverse Kinematics (IK). During sharedcontrol rollouts, human interventions preempt the residual policy with absolute priority; arbitration runs locally on the robot’s onboard unit (Fig. 1) for low-latency response.

## C.2 Distributed Asynchronous Training

We adopt a distributed asynchronous training paradigm inspired by HIL-SERL [20], decomposed into three loosely coupled components: Rollout Worker: an asynchronous actor process on the robot (not a separate robot station) that executes actions and collects transitions at 10 Hz; all experiments use a single TIAGo++, although the shared replay buffer would also admit additional robots. Central Learner: performs continuous policy and critic optimization on a workstation (NVIDIA RTX 4090) from the shared replay buffer, asynchronously to data collection; the rollout worker syncs the latest actor checkpoint every 50 environment steps. Inference Node: a standalone service that continuously provides frozen base-policy predictions as conditioning observations for the residual agent. Communication between the onboard unit and the workstation runs over TCP/IP, providing decoupled synchronization with low end-to-end latency.

## C.3 Tasks and Evaluation Protocol

We evaluate on three real-world tasks of increasing complexity (Fig. 3). In each trial we randomize the robot’s initial pose (±30<sup>◦</sup> orientation, 1.5 m starting distance from the table where applicable) and manually perturb object placements (bread, plate, litter) by roughly 10–20 cm around nominal positions, using the same randomization for training and evaluation: (I) Place Only (Single-Arm). The robot starts holding bread in its right gripper and must navigate to the table and precisely place the bread onto a plate. Action space: $\mathcal { A } \in \mathbb { R } ^ { 1 0 }$ (6-D right-arm end-effector delta + 1-D gripper + 3-D base); maximum episode length: 500 steps (50 s at 10 Hz). (II) Sweep Litter (Dual-Arm). TIAGo++ coordinates a broom (right arm) and a dustpan (left arm) to sweep litter into the bag. Action space: $\mathcal { A } \in \mathbb { R } ^ { 1 5 }$ (two 6-D end-effector deltas + 3-D base; the two gripper dimensions are dropped because both hands hold their tools throughout); maximum episode length: 500 steps. This task tests highdimensional bilateral coordination. (III) Pick Place (Long-Horizon). Starting empty-handed at 1.5 m from the source table, the robot must (1) approach and grasp the bread, (2) transition between two tables, and (3) place it on a target plate. Action space: $\mathcal { A } \in \mathbb { R } ^ { 1 0 }$ (6-D end-effector delta + 1-D gripper + 3-D base); maximum episode length: 800 steps (80 s). This is the most demanding task, requiring tight base–arm spatiotemporal synchronization (Fig. 1).

For each task we run 20 independent trials with randomized initialization. A trial is marked as failure if it exceeds the maximum step limit or violates task-specific safety constraints. Success is taskspecific: Place Only: the bread must be released and stabilized within the target plate. Sweep Litter: all target litter must be fully swept into the dustpan bag; any litter remaining outside or displaced beyond the workspace is a failure. Pick Place: we use a phased success metric—successful grasping and lifting yield a partial score of 0.5, and successful placement on the target plate yields the full score 1.0. For all tasks we report success rate (%) and episode length (steps to completion) averaged over the 20 trials.

## C.4 Failure Modes and Pick Place Ablation

Before fine-tuning, the frozen DP fails 10/20 trials on Place Only (6 base mis-positioning, 4 wrong release height) and 11/20 on Sweep Litter (7 broom–dustpan misalignment from base drift, 4 incom plete sweeps), i.e., mostly base-positioning errors that propagate to the arm; HiRe-MoMa reduces these to $1 / 2 0$ and 2/20. On Pick Place, the remaining failures stem mainly from hardware: the arm’s narrow workspace turns small base errors after the table-to-table transition into IK-infeasible grasp targets, and the casters stick after prolonged operation, both beyond what bounded residuals $( \epsilon = 0 . 3 )$ can absorb. Because the long table-to-table transition aggravates these hardware issues and the 800-step episodes are costly to repeat across ablated variants, Pick Place is reported only in the main comparison (Tab. 1) and omitted from the build-up ablation (Tab. 2).

## D Implementation Details

## D.1 Observations and Actions

The observation space consists of egocentric RGB images $( 2 0 8 \times 1 6 0 )$ and $a \sim 2 0$ -dimensional proprioceptive vector (whose exact dimension is task-dependent and kept consistent across all baselines). We use a relative action space, with all continuous dimensions normalized to $[ - 1 , 1 ]$ . The gripper is controlled discretely; all other dimensions are continuous. The residual correction magnitude is bounded by $\| \mathbf { a } ^ { \mathrm { r e s } } \| \leq \epsilon$ with $\epsilon = 0 . 3 $ , as introduced in Sec. 4.1.

## D.2 Network Architecture and Optimization

Both actor and critic of the residual policy are parameterized by MLPs with hidden dimension 256. We use an off-policy SAC framework [20] with a critic ensemble for conservative target estimation $( \mathrm { A l g . \ 1 } )$ . The auxiliary alignment-loss weight is fixed at $\lambda _ { \mathrm { i m i } } = 1 0 ^ { - 3 }$ across all tasks; target critic networks are updated via Polyak averaging following standard SAC. The base policy remainsfrozen throughout fine-tuning to preserve the pre-trained priors. The residual bound ϵ trades correction capacity against exploration stability: too small an ϵ under-corrects the base policy, while too large an ϵ enlarges the exploration space and destabilizes SAC, consistent with prior residual RL [14, 15]; $\epsilon = 0 . 3$ works well on all three tasks.

## D.3 Base-Policy Pre-Training and Warm-Up Buffer

We pre-train two frozen base policies, Diffusion Policy (DP) [22] and ACT [47], on human demonstrations: 80 trajectories for Place Only, 100 for Sweep Litter, and 200 for Pick Place. Each base model is trained for 1,000 epochs under the standard configuration of its respective architecture. For DP we set observation horizon 1, prediction horizon 8, and action horizon 6, following standard visuomotor practice [43]; ACT is configured with matching chunk sizes. These demonstrations are used only to pre-train $\pi ^ { \mathrm { b a s e } }$ . For fine-tuning, following HIL-SERL [20], we instead warm-start the expert buffer with ∼500–600 transitions from successful episodes collected on the robot by rolling out the frozen base policy, with operator takeover when needed $( \mathcal { D } _ { \mathrm { w a r m } }$ in Alg. 1); HiRe-MoMa thus never accesses the base policy’s training data.

## D.4 Human Intervention Protocol and Budget

A single trained operator supervises all training runs, blind to the method being trained, and intervenes when failure is imminent but still recoverable (e.g., the gripper starts releasing while misaligned with the plate, or the broom is extended well wide of the litter). Interventions are collected online within each method’s own run under this identical protocol, so all methods receive the same intervention opportunities. We measure the budget by the intervention coverage, i.e., the fraction of human-intervened transitions among all collected transitions: it stays below 20% on all tasks and decreases from roughly 15–20% early in training to 0–3% near convergence, as interventions concentrate at task-critical phases (Fig. 4(b)). Since interventions reshape the critic rather than being cloned, the method tolerates moderate teleoperation noise; a systematic study of operator expertise is left to future work.

## E Related Work

## E.1 Real-World Reinforcement Learning

Real-world RL aims to bridge the sim-to-real gap through direct physical interaction [16]. Among the persistent bottlenecks, three lines of work are particularly relevant. (i) Reward specification. To reduce the burden of reward engineering, recent studies have leveraged vision-language representations [50], video-based goal rewards [51], and language-conditioned prediction models [52] as alternatives to hand-crafted dense rewards. HiRe-MoMa instead adopts a sparse binary success reward, removing the need for these auxiliary signals. (ii) Autonomous learning loops. Frameworks such as MEDAL [53, 54] and other self-resetting systems [55] use forward-backward policy pairs to automate task resets. While complementary, these techniques are orthogonal to our focus on residual fine-tuning. (iii) Sample efficiency. Early on-policy deployments [56] demonstrated feasibility, after which the field moved toward model-based [57] and off-policy [58] architectures. More recent advances include Mixture-of-Experts gradient deconfliction [59], simulation-based pretraining [60], and warm-start RL for offline-to-online transitions [61]. The SERL framework [17] consolidated many of these ideas into a practical off-policy paradigm that has inspired follow-on residual fine-tuning systems [18, 19]. HiRe-MoMa builds on this off-policy foundation, but extends it to the unique coupling structure of mobile manipulation, where prior approaches based on simulation-pre-trained latent action spaces [21] are computationally expensive and lack adaptability.

## E.2 Policy Post-Training and Residual RL

Post-training has emerged as a key paradigm for refining base policies derived from large-scale Vision-Language-Action models [23] or specialized imitation-learning frameworks [22]. Interactive imitation learning [32, 33] incorporates human supervision to mitigate distributional drift, while RL extends agent capability beyond the offline dataset through autonomous exploration. However, generative base policies pose a technical hurdle: their intractable log-likelihoods preclude standard policy gradient [62] or value-based [45] updates. To address this, recent work has formulated surrogate objectives [13, 24] or alternative entropy measures [25, 26] to bypass explicit likelihood estimation; nonetheless, these approaches typically require architecture-specific modifications and limit cross-platform portability. Policy-agnostic RL [27] offers a more general route but often suffers from training instability and over-conservatism [18]. Residual RL offers a practical alternative by learning additive corrections atop frozen base policies [19, 28]; HiRe-MoMa builds on this paradigm and introduces a structured decoupling tailored to MoMa’s cross-module coupling.

## E.3 Mobile Manipulation

Mobile manipulation has been studied along several complementary axes. Recent approaches have leveraged Vision-Language Models [31, 3] for high-level reasoning [1] or large-scale human video datasets [63] to scale base capabilities, while imitation-learning studies show that exploiting hierarchical relationships within modular action spaces is critical for downstream performance [6, 35]. In the RL domain, however, this structural insight remains under-explored: most methods rely on extensive simulation and hand-crafted constraints, such as restricted latent action spaces, to manage the sim-to-real gap [8, 36]. Even when real-world RL is used for continuous adaptation [37], the vast search space often necessitates task-specific priors that limit cross-scenario generalization, or computationally expensive skill-discovery phases [21]. In contrast, HiRe-MoMa operates directly in the raw action space of any pre-trained policy and avoids manually tuned reward shaping [8] by exploiting the system’s hierarchical dependencies under a sparse, success-only reward.

## E.4 Human-in-the-Loop Robot Learning

Human input [34, 64] has long served as a supervisory signal for aligning agent behavior with expert intent [40, 41]. In the RL setting, interventions are traditionally used as safety buffers to prevent policy collapse during exploration [20, 9], treated as data sources for imitation [38], or used to construct auxiliary losses for denser supervision [39]. Beyond these conventional roles, expert corrections also encode implicit information about the relative value of state-action pairs. HiRe-MoMa exploits this perspective by treating corrections as functional constraints on the critic’s value surface, complementing rather than replacing the standard imitation pipeline.

## F Limitations and Future Work

Limitations. Beyond the single-platform scope (TIAGo++, 20 trials per condition, no seed sweeps, build-up ablation on two of three tasks) and the hardware-bound Pick Place results (App. C.4), HiRe-MoMa has three methodological limitations. First, its hierarchy is spatial rather than temporal: unlike hierarchical RL methods that assign credit across temporal levels via hindsight relabeling [48] or off-policy subgoal correction [49], both heads act at the same timescale and share one sparse reward. Since a base pose is only good relative to the arm’s ability to act from it, we model this coupling implicitly by conditioning the arm on the base residual instead of assigning per-module credit. As a result, an arm failure also lowers the value of a correct base motion, and humanassisted successes can inflate $Q _ { \phi _ { \mathrm { m o } } }$ at base poses where the autonomous arm would fail; sparse interventions limit but do not remove this bias, since balanced sampling keeps them in every batch. Second, although HiRe-MoMa never accesses the base policy’s training data, it needs a small warmup buffer of successful on-robot rollouts per task (App. D.3). Third, an MLP residual conditioned on the current observation may miss history-dependent corrections of strongly non-Markovian base policies (e.g., action chunking); a history buffer or recurrent head is a natural extension.

While HiRe-MoMa is effective on the evaluated tasks, several directions remain open for future:

Adaptive Regularization. The residual bound ϵ = 0.3 and the expert-guided alignment weight $\lambda _ { \mathrm { i m i } } = 1 0 ^ { - 3 }$ are currently fixed across tasks, which suffices for our evaluated settings but is unlikely to be optimal across broader task distributions. Developing adaptive weighting mechanisms, such as learnable coefficients, curriculum-based scheduling, or human-in-the-loop prior scheduling, could dynamically adjust the regularization strength based on the agent’s real-time performance and task complexity, potentially improving robustness across diverse task distributions.

Multimodal Sensing Integration. Our framework primarily utilizes visual and proprioceptive feedback. However, many fine-grained manipulation tasks are inherently contact-rich, requiring precise force and tactile sensing to ensure safety and dexterity. Integrating multimodal fusion (e.g., visuotactile or force-torque sensors) into the residual learning process is a critical next step for scaling to complex, industrial-grade applications.

Richer Inter-Module Structures. HiRe-MoMa’s hierarchical decoupling currently uses a two-level structure (base → arm), which fits the evaluated tasks but does not directly extend to bi-manual coordination with active torso and head control, or to multi-stage long-horizon tasks with temporal dependencies. Such configurations may benefit from deeper or non-tree-shaped inter-module structures; exploring latent representations or graph-based conditioning mechanisms could capture richer relational dependencies. Larger workspaces with obstacles would further call for collision-aware base corrections, while multi-stage semantic reasoning is best delegated to a stronger base policy (e.g., a VLA [23]), with HiRe-MoMa providing the low-level whole-body corrections.

Cross-Platform and Simulation Validation. To enhance reproducibility and broaden adoption, replicating the HiRe-MoMa protocol in high-fidelity simulators (e.g., Robocasa [65], Isaac Sim, Behavior-1K [66]) and validating on additional mobile manipulation platforms (e.g., Stretch, Fetch) would clarify the framework’s generalizability and surface platform-specific tuning requirements. A platform with a wider arm workspace and more reliable base actuation would also lift the hardware ceiling observed on Pick Place (App. C.4).