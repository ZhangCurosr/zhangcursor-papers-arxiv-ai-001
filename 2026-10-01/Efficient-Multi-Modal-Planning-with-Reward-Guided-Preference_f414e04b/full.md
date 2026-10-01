# Efficient Multi-Modal Planning with Reward-Guided Preference Optimization for Autonomous Driving

Chenglin Chen, Lujia Wang, Xinhu Zheng, Jun Ma, Haoang Li<sup>∗</sup>

Abstract— Safe and efficient trajectory planning is essential in autonomous driving. However, existing end-to-end approaches often fall short in both computational efficiency and safety guarantees. Methods based on imitation learning suffer from causal confusion, while rule-based scoring approaches often incur heavy computational overhead and suffer from objective misalignment. Additionally, preference-based methods rely on strict pairwise annotations, limiting data utilization. To overcome these limitations, we propose EMPlan, an efficient multi-modal trajectory planning method powered by reward-guided fine-tuning. We design a hybrid architecture that combines sparse anchors with an offset refinement module for efficient multi-modal trajectory prediction. Sparse anchors provide coarse trajectory candidates with low latency, which are subsequently refined by the offset module for higher prediction accuracy. To enhance safety without incurring additional inference costs, we adopt a two-stage training paradigm consisting of pretraining and reward-guided fine-tuning. During finetuning, we leverage rule-based reward signals and unpaired preference supervision to refine the pretrained policy toward safer trajectory selection. We evaluate EMPlan on the nonreactive NAVSIM benchmark, where it strikes a favorable balance between planning accuracy and efficiency, demonstrating superior performance under real-time constraints.

## I. INTRODUCTION

Ensuring safety, comfort, and compliance with traffic rules are the fundamental objectives for an autonomous driving (AD) system. Recently, the end-to-end (E2E) AD has emerged as a promising solution to achieve these goals [1], [2], [3], [4]. This paradigm allows the entire system to be optimized jointly, thereby eliminating the information loss inherent in traditional modular designs. Nevertheless, the training objective of most existing E2E systems remains rooted in imitation learning (IL), where policies are trained in an open-loop manner by simply aligning predicted trajectories with expert data [5], [6]. IL-based methods often fail in long-tail situations due to causal confusion [7] and distribution shifts [8], [9], [10] in the training data. Furthermore, the objective of IL primarily aims to mimic expert data, often neglecting critical driving requirements such as safety, comfort, and adherence to traffic rules.

To alleviate these limitations, recent approaches attempt to refine IL-pretrained policies by incorporating auxiliary signals, such as rule-based scores [11], [12] or preferencebased labels [13], [14], [15], [16], [17], to explicitly align the model with safe or logical driving behaviors. Although these approaches are generally more reliable than pure imitation, they still suffer from limitations in computational efficiency and safety guarantees. First, methods incorporating rule-based scores typically adopt anchor-based architecture. To ensure sufficient action space coverage, they require a dense set of anchors, resulting in prohibitive computational latency. Moreover, directly regressing rule-based scores often causes objective misalignment, as it encourages the model to fit absolute values rather than learning preference relationships necessary for reliable trajectory selection. Second, preference-based methods attempt to better capture driving logic through trajectory-level comparisons. However, they generally rely on strict pairwise annotations (e.g., chosen vs. rejected) within each scenario. Such pairwise supervision is sparse and data-inefficient, as learning is restricted to limited trajectory pairs while other valid trajectory candidates are ignored.

![](images/f78c4ababbc7fa5439c19683c3d7b737841d33dda6be89e2841fbd69bec27d68.jpg)  
Figure 1. Overview of our EMPlan, built upon a hybrid architecture and powered by reward-guided fine-tuning. (a) The hybrid architecture combines sparse anchors with an offset module for efficient multi-modal trajectory prediction. Reward-guided fine-tuning further aligns trajectory selection with safety preferences. (b) Our model achieves better performance in efficiency and accuracy compared with representative methods [2], [3], [11], [12]. (c) Compared to score-based methods (i.e., WoTE [12]), our model assigns lower scores to human-like but unsafe trajectories (red) and higher scores to safe ones (green).

To overcome these limitations, we propose EMPlan (see Fig. 1), a multi-modal trajectory planning method that unifies efficient trajectory generation with safety-aware trajectory selection. Our model is built upon a hybrid architecture and enhanced by reward-guided fine-tuning. First, we design a hybrid architecture to improve efficiency. Unlike traditional methods that exhaustively enumerate thousands of anchors, we utilize sparse anchors to generate coarse trajectory candidates, which are then refined by a lightweight offset module. This hybrid design preserves action space coverage while significantly reducing computational overhead. Second, to enhance safety without incurring additional inference costs, we employ a two-stage training paradigm consisting of pretraining and reward-guided fine-tuning. In the pretraining stage, we design a score-based policy learning objective that integrates trajectory imitation, rule-based supervision, and auxiliary BEV segmentation to better handle data imbalance and improve representation learning. In the fine-tuning stage, we optimize the pretrained policy using reward signals instead of rule-based scores. This aligns the training objective with the trajectory selection criterion at inference, reducing the mismatch between optimization and decision-making. However, reward supervision typically operates on individual trajectories and fails to explicitly model relative comparisons between candidates. We therefore incorporate preferencebased signals to capture relative relationships within each scenario. To alleviate the reliance on strict pairwise annotations in preference learning, we utilize unpaired trajectory samples with preference annotations (e.g., desirable and undesirable samples). This approach improves the utilization of available trajectory data, aligning the policy with safe driving behaviors while avoiding the constraints of strict pairwise supervision.

In summary, our main contributions are as follows:

• We propose EMPlan, an efficient multi-modal trajectory planning method powered by reward-guided fine-tuning, which can achieve real-time inference and improve overall planning accuracy.

• We design a hybrid architecture combining sparse anchors with an offset module to refine trajectories, overcoming discrete coverage gaps at low computational cost.

• We introduce an unpaired, reward-guided fine-tuning strategy that adapts preference learning to anchor-based trajectory evaluation, enabling effective policy alignment with multi-objective driving criteria.

## II. RELATED WORKS

## A. End-to-End Autonomous Driving

Benefiting from large-scale driving datasets and advances in transformer architectures [18], E2E AD has progressed rapidly in recent years. Early works, such as Transfuser [2], emphasize multi-modal sensor fusion to learn stronger scene representations. Moving towards planning-oriented design, UniAD [3] unifies full-stack tasks within a single network to enhance planning performance. To further improve efficiency, VAD [19] explores compact vectorized scene representations. Despite architectural differences, these approaches typically regress a single-mode trajectory conditioned on the ego vehicle. To address the inherent multimodality of driving behaviors, VADv2 [4] introduces a predefined anchor vocabulary and learns to assign scores to candidate trajectories, enabling multi-mode prediction. Building upon this idea, HydraMDP [11] distills rule-based knowledge from simulators into the model to enhance trajectory scoring. WoTE [12] leverages a world model to generate future BEV representations as intermediate supervision, improving downstream trajectory prediction. Nevertheless, such score-based models are optimized by assigning supervision scores to each anchor in isolation, while ignoring the relative preference relations among different trajectory candidates. More recently, diffusion models have been explored for multi-modal trajectory prediction. Despite their strong generative capability, diffusion-based approaches [20], [21] typically incur high memory overhead and inference latency, which hinder their deployment in real-time safety-critical AD systems. In contrast, we propose a lightweight hybrid architecture model for multimodal trajectory prediction, further enhanced by reward-guided fine-tuning to encourage safer planning behaviors.

## B. Policy Fine-Tuning in Autonomous Driving

Recently, policy fine-tuning has emerged as a promising direction for improving E2E AD, mainly through closed-loop reinforcement learning (RL) and preference-based optimization. RL-based methods benefit from interactive environment signals but are still in an early stage for AD, as they rely heavily on domain-specific reward engineering and often exhibit unstable training. To bridge the simulation-to-reality gap, RAD [9] introduces a 3D Gaussian Splatting-based RL framework for training in high-fidelity environments. In parallel, both AlphaDrive [22] and TrajHF [23] adopt Group Relative Policy Optimization [24] for policy optimization, where the former incorporates vision-language reasoning, and the latter aligns diffusion planners with human driving preferences. Furthermore, ReCogDrive [25] leverages simulator-assisted RL to refine trajectory generation toward safer and more human-like behaviors. More recently, DriveDPO [17] fine-tunes anchor-based planners via DPO, but it relies on strict pairwise preference annotations, which may leave valuable unpaired trajectories underutilized. In contrast, our method adopts unpaired preference data, allowing each candidate trajectory to contribute independently to preference optimization. Our goal is to improve the alignment between predicted scores and actual driving quality without requiring explicit pair construction. This is particularly beneficial under a multi-candidate selection setting, where candidate trajectories can be independently evaluated rather than selectively paired.

## III. METHOD

Given the sensor observation O, the goal of the model is to forecast a T-step ego-vehicle’s future trajectory $\tau =$ $\{ ( x _ { t } , y _ { t } , \theta _ { t } ) \} _ { t = 1 } ^ { T }$ , where each waypoint denotes the vehicle’s position and headings at time step t. To handle the multimodal nature of real-world driving, we follow the paradigm in VADv2 [4] and treat motion planning as a discrete selection problem over a predefined vocabulary of trajectory anchors $\bar { \mathcal { V } } ~ = ~ \{ \tau _ { i } ~ \in ~ \bar { \mathbb { R } ^ { T \times 3 } } \} _ { i = 1 } ^ { N }$ , where each anchor $\tau _ { i }$ represents a $T \cdot$ -step waypoint sequence. The policy assigns a score to each anchor, and the anchor with the highest score is selected as the predicted trajectory.

Standard supervised training optimizes these scores independently for each anchor, which ignores relative quality among candidates and requires a dense anchor set to cover the action space, leading to high computational cost. To address these limitations, we aim to 1) efficiently generate trajectories without exhaustive anchor enumeration and 2) leverage relative reward signals to refine the policy toward safer trajectory selection. The following sections detail the overall architecture and key optimization strategies.

![](images/cabfd67f0eac694ffcfe44e3a83f08daa3c371d6e7ebf575b67949cdbba173d6.jpg)  
Figure 2. Overview of the network architecture. Given multi-view images, the perception module extracts BEV representations, which are fed into a trajectory decoder to predict trajectory offsets and reward scores based on ego states and sparse anchors. The offset head refines anchor trajectories, and the trajectory with the highest aggregated reward is chosen as the final output. During training, the model is first pretrained with imitation and rule-based supervision and subsequently optimized via reward-guided fine-tuning, which integrates reward loss and preference loss to align trajectory selection with safety preferences.

## A. Hybrid Architecture of Our Policy Model

As shown in Fig. 2, our method is structured into two networks: a Perception module for scene understanding and a Trajectory Decoder for multimodal trajectory generation. For the perception module, we extract features from multi-view RGB images and transform them into BEV representations at two distinct scales. One scale is utilized for BEV semantic segmentation with auxiliary supervision to enhance spatial awareness. The other is combined with learnable positional embeddings to form scene tokens $E _ { \mathrm { e n v } }$ , which provide rich, scene-level context for the subsequent trajectory decoder. The core of our architecture lies in the design of the trajectory decoder. To balance computational efficiency and planning accuracy, we decompose trajectory generation into coarse discrete mode representation and continuous offset refinement. The sparse anchors represent distinct maneuver modes, while the offset head predicts a scene-conditioned residual for each anchor, reducing the quantization error imposed by a finite anchor vocabulary. In contrast, denseanchor methods obtain finer discrete coverage by increasing the number of anchors, which enlarges the candidate vocabulary and raises the computation of trajectory decoding. Unlike diffusion planners, our refinement is completed in a single feed-forward pass without iterative denoising.

Specifically, we begin by constructing a sparse set of trajectory anchors $\mathcal { V } = \overline { { \{ \tau _ { i } \in \mathbb { R } ^ { T \times 3 } \} } } _ { i = 1 } ^ { N }$ by applying k-means clustering to the NAVSIM training dataset, following prior works [4], [11], [12]. These anchors are transformed into a high-dimensional feature space through Fourier positional encoding to obtain the anchor embeddings:

$$
\Gamma ( \mathcal { V } ) = \left[ \mathcal { V } , \phi _ { 0 } ( \mathcal { V } ) , \phi _ { 1 } ( \mathcal { V } ) , \cdots , \phi _ { L - 1 } ( \mathcal { V } ) \right] \in \mathbb { R } ^ { N \times T \times ( 3 + 6 L ) }
$$

$$
\phi _ { k } ( \mathcal { V } ) = \left[ \sin ( 2 ^ { k } \pi \mathcal { V } ) , \cos ( 2 ^ { k } \pi \mathcal { V } ) \right]\tag{1a}
$$

(1b)

In addition, the ego state and high-level driving command are also encoded and fused with these anchor embeddings to form the final planning tokens $E _ { \mathrm { p l a n } }$ for trajectory decoding. The scene tokens $E _ { \mathrm { e n v } }$ and planning tokens $E _ { \mathrm { p l a n } }$ are then fed into a Planning Transformer to perform cross-attention, producing a set of scene-conditioned planning tokens:

$$
\hat { E } _ { \mathrm { p l a n } } = \mathrm { T r a n s f o r m e r D e c o d e r } \left( Q = E _ { \mathrm { p l a n } } , K , V = E _ { \mathrm { e n v } } \right)\tag{2}
$$

Finally, the context-aware planning tokens $\hat { E } _ { \mathrm { { p l a n } } }$ are passed to two separate prediction heads: an offset head and a scorer head. Since the predefined trajectory anchors form a sparse and discrete motion prior, the ground-truth trajectories may not perfectly align with them. To address this, an offset head predicts per-anchor offsets to refine the coarse trajectories. The refined trajectories are given by:

$$
\hat { \mathcal { V } } = \mathcal { V } + \mathrm { O f f s e t H e a d } \left( \hat { E } _ { \mathrm { p l a n } } \right)\tag{3}
$$

In parallel with trajectory refinement, the scorer head evaluates the quality of each refined trajectory. Specifically, it outputs an imitation-based score $S _ { \mathrm { i m } }$ to measure humanlikeness and a set of rule-based scores $S _ { \mathrm { s i m } }$ to assess safety and comfort constraints. These per-anchor scores are further unified via a reward aggregation module into a single reward for each trajectory. During inference, the optimal trajectory is chosen by maximizing the predicted reward. During training, the predicted scores together with the refined trajectories are supervised to optimize the policy in the first stage.

## B. Score-Based Policy Learning

This section details the supervision and loss formulation for first-stage policy learning. Following established practices [11], [26], we employ a trajectory regression loss ${ \mathcal { L } } _ { \mathrm { t r a j } }$ with a winner-takes-all strategy and an imitation-based score loss ${ \mathcal { L } } _ { \mathrm { i m } }$ to encourage human-like trajectory prediction.

BEV Segmentation Loss. We introduce an auxiliary BEV semantic segmentation task to enhance scene representation. The BEV map contains multiple categories (e.g., background, road, lane centerline, vehicles) with severe class imbalance, as background and road dominate the scene. Unlike prior work [2], [11] that adopts cross-entropy loss, we supervise the predicted map using focal loss ${ \mathcal { L } } _ { \mathrm { m a p } }$ to emphasize hard and minority categories.

Rule-Based Score Loss. To enforce safety and physical constraints beyond imitation, we supervise the predicted rule-based scores $S _ { \mathrm { s i m } }$ using precomputed outputs $S _ { \mathrm { s i m } } ^ { * }$ from NAVSIM simulator [27]. The simulator provides five rulebased metrics: No At-Fault Collision (NC), Drivable Area Compliance (DAC), Time-to-Collision (TTC), Comfort (C), and Ego Progress (EP). The distribution of rule-based scores is heavily skewed, as the majority of anchors correspond to low-quality behaviors. To mitigate this imbalance, we adopt a Sigmoid Focal Loss (SFLoss) for supervision. Notably, for the EP score, we introduce a conditional progress mask. Since EP measures agent progress but is set to zero when either NC or DAC is zero, the mask ensures that EP loss is applied only for trajectories that are safe and legal. This prevents the model from being misled by unsafe behavior while still learning meaningful progress signals. As a result, the loss is defined as:

$$
\mathcal { L } _ { \mathrm { s i m } } = \left\{ \begin{array} { l l } { \mathrm { M a s k } _ { \mathrm { E P } } \cdot \mathrm { S F L o s s } \left( S _ { \mathrm { s i m } } , S _ { \mathrm { s i m } } ^ { * } \right) , } & { \mathrm { i f ~ \mathrm { s i m } = E P } } \\ { \mathrm { S F L o s s } \left( S _ { \mathrm { s i m } } , S _ { \mathrm { s i m } } ^ { * } \right) , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{4}
$$

where Mas $\mathfrak { s } _ { \mathrm { E P } } = ( S _ { \mathrm { N C } } > 0 . 5 ) \wedge ( S _ { \mathrm { D A C } } > 0 . 5 )$ . Finally, the total loss in the first stage is defined as:

$$
{ \mathcal { L } } _ { t o t a l } = \lambda _ { \mathrm { { m a p } } } { \mathcal { L } } _ { \mathrm { { m a p } } } + \lambda _ { \mathrm { { t r a j } } } { \mathcal { L } } _ { \mathrm { { t r a j } } } + \lambda _ { \mathrm { { i m } } } { \mathcal { L } } _ { \mathrm { { i m } } } + \lambda _ { \mathrm { { s i m } } } { \mathcal { L } } _ { \mathrm { { s i m } } }\tag{5}
$$

where $\lambda _ { \operatorname* { m a p } } , \lambda _ { \operatorname { t r a j } } , \lambda _ { \operatorname { i m } } , \lambda _ { \sin }$ balance the contributions of the respective loss terms.

## C. Reward-Guided Fine-Tuning

The first-stage policy learning yields a pretrained reference policy $\pi _ { \mathrm { r e f } } .$ , which predicts per-anchor scores and refined trajectories. However, regressed absolute scores can be noisy and fail to explicitly model relative preferences among candidate trajectories. In reality, driving decisions rely more on relative preferences than absolute scoring. To bridge this gap, we introduce a reward-guided fine-tuning strategy (see Fig. 3) that combines two supervision signals: i) reward supervision over all candidate trajectories and ii) preferencebased optimization by unpaired trajectory samples with preference annotations. Preference-based optimization captures trajectory preferences but is prone to training instability and distribution drift when applied alone. Inspired by [8], we introduce reward supervision as an auxiliary loss to mitigate this issue, which also helps to keep the training objective consistent with trajectory selection. In particular, we break down this process into two key components: reward design and preference-based optimization in Stage II.

![](images/bcf1e727afe424527c21723bfe1cab336720126a73e3708fd234571c63aef126.jpg)  
Figure 3. Illustration of our two-stage training pipeline. In Stage I, π<sub>θ</sub> is pretrained using expert demonstrations and rule-based scores to learn human-like driving priors. In Stage II, rule-based scores are combined to compute a trajectory-level reward loss across all candidates. In addition, unpaired preference data are used to compute preference loss (see Eq. (10)), enabling the model to learn safer and more reliable trajectory selection.

Reward Design. During reward-guided fine-tuning, we employ the reward aggregation module to unify the peranchor scores predicted by the policy model into a single scalar reward for each trajectory. For each candidate trajectory y under scene x, the reward $R ( x , y )$ is computed as a weighted combination of the imitation score $S _ { \mathrm { i m } }$ and rulebased scores $S _ { \mathrm { s i m } } \in \{ S _ { \mathrm { N C } } , S _ { \mathrm { D A C } } , S _ { \mathrm { E P } } , S _ { \mathrm { T T C } } , S _ { \mathrm { C } } \}$ in log space:

$$
\begin{array} { r l } & { R ( x , y ) = \omega _ { 1 } \log { S _ { \mathrm { i m } } } + \omega _ { 2 } \log { S _ { \mathrm { N C } } } + \omega _ { 3 } \log { S _ { \mathrm { D A C } } } } \\ & { \phantom { = } + \omega _ { 4 } \log \frac { 5 S _ { \mathrm { E P } } + 5 S _ { \mathrm { T T C } } + 2 S _ { \mathrm { C } } } { 1 2 } } \end{array}\tag{6}
$$

where $\{ \omega _ { i } \} _ { i = 1 } ^ { 4 }$ are the hyperparameters that balance driving style and safety. We use log-space aggregation to reflect the conjunctive nature of the safety-related criteria, so that high EP, TTC, or C scores cannot compensate for nearzero NC or DAC scores. Unlike a conventional weighted sum, which allows compensation across all components, the weighted sum of logarithms is equivalent to the logarithm of a weighted geometric product. Consequently, when any individually logged factor approaches zero, the aggregated score decreases sharply, strongly penalizing the corresponding trajectory. These scores serve as unified reward signals for both direct supervised training and subsequent preference-based optimization.

Preference-Based Optimization. To further align the policy with robust driving behaviors, we introduce Kahneman–Tversky Optimization (KTO) [28], which encourages desirable outputs while penalizing undesirable ones using unpaired samples rather than strict pairwise data. To enable preference-based fine-tuning, our first step is to construct preference samples. As illustrated in Fig. 4, we utilize the pretrained model $\pi _ { \mathrm { r e f } }$ to generate multiple trajectories and feed them into an offline evaluator to obtain trajectory-level imitation scores, rule-based scores, and aggregated rewards. We restrict preference partitioning to the top 25% of candidates ranked by L2 similarity to the expert demonstration, since the remaining trajectories deviate substantially from expert behavior and are therefore excluded from fine-grained preference construction. These human-like trajectories are then partitioned into three disjoint sets according to two empirically defined reward thresholds, $\tau _ { \mathrm { h i g h } } ~ = ~ - 0 . 6$ and $\tau _ { \mathrm { l o w } } = - 1 . 0$ . Trajectories with rewards above the threshold τ<sub>high</sub> form the desirable set $\mathcal { D } ^ { + }$ , corresponding to strictly safe behaviors. Those with rewards below the threshold $\tau _ { \mathrm { l o w } }$ form the undesirable set $\mathcal { D } ^ { - }$ , corresponding to safetyviolating behaviors. The remaining trajectories constitute the reference set $\mathcal { D } ^ { \mathrm { r e f } }$ , which contains safe yet suboptimal behaviors and is used to estimate the reference point as the average performance of the reference model. This partition enables the model to capture finer-grained preferences among trajectories with similar quality, providing more informative supervision for trajectory ranking than rule-based scores.

![](images/961088b370c009039a217319bd96db8d7bf2569abc360ef8e44de33a58073009.jpg)  
Figure 4. Illustration of our construction of reward signal and preference data. The policy model pretrained in Stage I first generates candidate trajectories, which are then evaluated by an offline evaluator to obtain reward scores. The resulting rewards are used both for dense reward supervision and for constructing unpaired preference data by selecting $\mathcal { D } ^ { + } , \dot { \mathcal { D } } ^ { - } , \mathcal { D } ^ { \mathrm { r e f } }$

Second, we define a relative reward and formulate the corresponding loss accordingly. We use the relative reward $r _ { \theta } ( x , y )$ to measure the preference of the current policy for output $y$ relative to the reference policy under scene x. We denote $\pi _ { \boldsymbol { \theta } } ( \boldsymbol { y } \mid \boldsymbol { x } )$ as the probability assigned to the trajectory $y$ by the current policy, obtained by normalizing the aggregated rewards $R ( x , y )$ over the set of trajectory anchors V using a softmax operation:

$$
\pi _ { \theta } ( y \mid x ) = { \frac { \exp { \bigl ( } R ( x , y ^ { \prime } ) { \bigr ) } } { \sum _ { y ^ { \prime } \in \mathcal { V } } \exp { \bigl ( } R ( x , y ^ { \prime } ) { \bigr ) } } }\tag{7}
$$

Similarly, $\pi _ { \mathrm { r e f } } ( y \mid x )$ denotes the corresponding probability assigned by the frozen reference policy. Thus, the relative reward is given by:

$$
r _ { \boldsymbol { \theta } } ( x , y ) = \log \pi _ { \boldsymbol { \theta } } ( y \mid x ) - \log \pi _ { \mathrm { r e f } } ( y \mid x )\tag{8}
$$

A larger $r _ { \theta } ( x , y )$ indicates that π<sub>θ</sub> assigns a higher preference to y compared to $\pi _ { \mathrm { r e f } } ,$ while a smaller value suggests the opposite.

We further estimate reference points to stabilize training. KTO was originally proposed for text generation, where the output space is continuous and many desirable samples are available. In contrast, anchor-based trajectory prediction operates over a discrete candidate set, in which a large portion of sampled trajectories may be infeasible or unsafe. Consequently, computing the reference point over all predicted trajectories can be biased, as low-quality samples may dominate the expectation. To mitigate this, we estimate the reference point z<sub>0</sub> as the average relative reward over the reference set:

$$
z _ { 0 } = \mathbb { E } _ { y ^ { \prime } \sim \mathcal { D } ^ { \mathrm { r e f } } } \left[ r _ { \theta } ( x , y ^ { \prime } ) \right]\tag{9}
$$

Finally, we formulate the preference loss objective:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { K T O } } = \mathbb { E } _ { ( x , y ) } \left[ w ( y ) \left( 1 - v ( x , y ) \right) \right] , } \\ & { v ( x , y ) = \left\{ \sigma \left( \beta \left( r _ { \theta } ( x , y ) - z _ { 0 } \right) \right) , \mathrm { i f ~ } y \in \mathcal { D } ^ { + } \right. } \\ & { \left. \sigma \left( \beta \left( z _ { 0 } - r _ { \theta } ( x , y ) \right) \right) , \mathrm { i f ~ } y \in \mathcal { D } ^ { - } \right. } \end{array}\tag{10}
$$

where $w ( y )$ denotes the sample weight and is set to $\lambda _ { D }$ or $\lambda _ { U }$ for desirable or undesirable samples, respectively. $\sigma ( \cdot )$ denotes the sigmoid function, and $\beta ~ > ~ 0$ controls the sensitivity of the objective to deviations of the relative reward from the reference point. This objective encourages the policy to increase the relative likelihood of desirable trajectories while suppressing undesirable ones with respect to the reference distribution.

## IV. EXPERIMENT

## A. Experimental Setup

Dataset. We conduct experiments on NAVSIM [27], an E2E driving benchmark built from the OpenScene [32] redistributed version of nuPlan [33]. NAVSIM provides 2 Hz sensor inputs, along with semantic BEV maps and 3D bounding box annotations of objects. NAVSIM provides 1192 scenes in the navtrain split for training and validation, and 136 scenes in the navtest split for testing.

Metrics. Following prior work [11], [12], [21], we use the Predictive Driving Model Score (PDMS) as the primary evaluation metric under a non-reactive simulation setting. PDMS is computed from five sub-metrics: NC, DAC, TTC, C, and EP. The final score is defined as $\mathrm { P D M S } = \mathrm { N C } \times$ $\mathrm { D A C } \times ( \mathrm { 5 E P + 5 T T C + 2 C } ) / 1 2$

Implementation Details. We use concatenated 1024×256 multi-view RGB images for perception input, following the commonly adopted setting in Transfuser [2]. We use V2- 99 [34] as the backbone to extract image features. We set $N = 2 5 6$ predefined trajectory anchors. The reward weights $w _ { 2 } ~ = ~ w _ { 3 } ~ = ~ 0 . 5 , ~ w _ { 4 } ~ = ~ 1 . 0$ are set informed by the relative importance of components in the PDMS metric, and a smaller weight $w _ { 1 } = 0 . 1$ is assigned to the imitation term to prioritize safety over style imitation. Training is performed on a platform with two NVIDIA RTX A6000 GPUs. For the first-stage training, we set $\lambda _ { \mathrm { m a p } } ~ = ~ 1 0 . 0 , ~ \lambda _ { \mathrm { t r a j } } ~ = ~ 1 . 0$ $\lambda _ { \mathrm { i m } } = 1 . 0 ,$ and $\lambda _ { \mathrm { { s i m } } } = \{ \mathrm { { N C : 5 . 0 , D A C : 6 . 0 , E P : 3 . 0 , } }$ TTC : 3.0, C : 1.0} to balance the loss term. We pretrain the model for 35 epochs with AdamW at a learning rate of $1 \times 1 0 ^ { - 4 }$ and a batch size of 16 per GPU. In the fine-tuning stage, we further optimize the policy for 10 epochs with a learning rate of $5 \times 1 0 ^ { - 5 }$ and a batch size of 16 per GPU in an iterative manner. During inference, the model predicts 8- waypoint trajectories over a 4-second horizon.

TABLE I  
PERFORMANCE COMPARISON WITH REPRESENTATIVE METHODS ON THE NAVTEST SPLIT. C: CAMERA. L: LIDAR.
<table><tr><td>Method</td><td>Input</td><td>Supervision</td><td>Anchor</td><td>NC↑</td><td>DAC↑</td><td>EP↑</td><td>TTC↑</td><td>C↑</td><td>PDMS↑ (Overall)</td></tr><tr><td>Human</td><td></td><td>=</td><td></td><td>100.0</td><td>100.0</td><td>87.5</td><td>100.0</td><td>99.9</td><td>94.8</td></tr><tr><td>PDM-closed</td><td>Perception GT</td><td>=</td><td></td><td>94.6</td><td>99.8</td><td>89.9</td><td>86.9</td><td>99.9</td><td>89.1</td></tr><tr><td>Transfuser [2]</td><td>C&amp;L</td><td>Human</td><td></td><td>97.7</td><td>92.8</td><td>79.2</td><td>92.8</td><td>100.0</td><td>84.0</td></tr><tr><td>UniAD [3]</td><td>C</td><td>Human</td><td></td><td>97.8</td><td>91.9</td><td>78.8</td><td>92.9</td><td>100.0</td><td>83.4</td></tr><tr><td>VADv2 [4]</td><td>C</td><td>Human</td><td></td><td>97.9</td><td>91.7</td><td>77.6</td><td>92.9</td><td>100.0</td><td>83.0</td></tr><tr><td>PARA-Drive [29]</td><td>C</td><td>Human</td><td></td><td>97.9</td><td>92.4</td><td>79.3</td><td>93.0</td><td>99.8</td><td>84.0</td></tr><tr><td>LAW [30]</td><td>C</td><td>Human</td><td></td><td>96.4</td><td>95.4</td><td>81.7</td><td>88.7</td><td>99.9</td><td>84.6</td></tr><tr><td>DRAMA [31]</td><td>C&amp;L</td><td>Human</td><td></td><td>98.0</td><td>93.1</td><td>80.1</td><td>94.8</td><td>100.0</td><td>86.5</td></tr><tr><td>DiffusionDrive [21]</td><td>C&amp;L</td><td>Human</td><td></td><td>98.2</td><td>96.2</td><td>94.7</td><td>82.2</td><td>100.0</td><td>88.1</td></tr><tr><td>Hydra-MDP [11]</td><td>C&amp;L</td><td>Human &amp; Rule</td><td>8192</td><td>98.3</td><td>96.0</td><td>78.7</td><td>94.6</td><td>100.0</td><td>86.5</td></tr><tr><td>WoTE [12]</td><td>C&amp;L</td><td>Human &amp; Rule</td><td>256</td><td>98.5</td><td>96.8</td><td>81.9</td><td>94.9</td><td>99.9</td><td>88.3</td></tr><tr><td>EMPlan (Ours)</td><td>C</td><td>Human &amp; Rule</td><td>256</td><td>98.4</td><td>97.5</td><td>84.2</td><td>94.9</td><td>100.0</td><td>89.6</td></tr></table>

![](images/65596fec8dc3cf76419bc964203a88df9eead9142b11939417af2e01385aeda2.jpg)  
Figure 5. Qualitative comparison of predicted trajectories across challenging scenarios. We present qualitative comparisons with TransFuser [2], Hydra-MDP [11], and the SOTA WoTE [12]. In (a), both TransFuser and Hydra-MDP fail to respond properly to the red traffic signal, causing rear-end collisions with the leading vehicle. In addition, all baselines exhibit lane deviations in (b). WoTE tends to be overly conservative, resulting in stagnation behaviors in (c). In (d), TransFuser and Hydra-MDP proceed straight toward the construction obstruction, while WoTE avoids collision but fails to commit to the required right turn. In contrast, our EMPlan successfully executes all maneuvers.

## B. Comparisons with State-of-the-art Methods

To evaluate the effectiveness of EMPlan, we perform experiments on the NAVSIM navtest split against opensourced representative planning baselines. These include the imitation-based Transfuser [2], the diffusion-based DiffusionDrive [21], and the scoring-based Hydra-MDP [11] and WoTE [12], where WoTE represents the previous state-ofthe-art (SOTA). As shown in Table I, our model achieves the best PDMS of 89.6, indicating superior overall driving performance. Among the baselines, Transfuser (84.0) suffers from causal confusion and often produces unsafe behaviors in complex interactions, including collisions and drivablearea violations, as shown in Figs. 5(a, b). DiffusionDrive (88.1) achieves the highest EP but underperforms our method on critical safety indicators, particularly NC and DAC. This highlights that PDMS provides a more reliable evaluation of driving performance, whereas a sub-metric alone can be misleading. Scoring-based methods such as Hydra-MDP (86.5) and WoTE (88.3) reduce collision rates but still struggle with consistent lane-keeping (see Fig. 5(b)), making them highly prone to off-road incidents. More importantly, their safety gains largely stem from overly conservative behavior, resulting in unnecessary braking and stagnated behavior as reflected by lower EP. As illustrated in the Uturn scenario in Fig. 5(c), both Hydra-MDP and WoTE fail to complete the maneuver and remain stalled, whereas our EMPlan successfully executes the turn. As further illustrated in the right-turn scenario with road construction in Fig. 5(d), Transfuser and Hydra-MDP fail to recognize the construction cones and proceed straight, resulting in collisions. Although WoTE avoids an immediate collision, continuing straight remains potentially unsafe given the obstruction ahead. In contrast, our EMPlan executes a smooth and decisive right turn, demonstrating its ability to suppress trajectories that appear human-like but carry latent safety risks.

TABLE II  
EFFICIENCY COMPARISON BETWEEN ANCHOR-BASED METHODS. ALL MODELS WERE TESTED ON AN NVIDIA RTX A6000.
<table><tr><td>Method</td><td>PDMS↑</td><td>Latency↓</td></tr><tr><td>Hydra-MDP [11]</td><td>86.5</td><td>104.7 ms</td></tr><tr><td>WoTE [12]</td><td>88.2</td><td>28.3 ms</td></tr><tr><td>EMPlan (Ours)</td><td>89.6</td><td>21.4 ms</td></tr></table>

TABLE III  
TABLE IV  
TABLE V  
ABLATION STUDY FOR OFFSET REFINEMENT AND SFLOSS.
<table><tr><td>offset</td><td>SFLoss</td><td>PDMS↑</td></tr><tr><td>x</td><td>x</td><td>81.6</td></tr><tr><td>√</td><td>x</td><td>87.8</td></tr><tr><td>√</td><td>√</td><td>88.3</td></tr></table>

ANALYSIS ON NUMBER OF PREDEFINED ANCHORS N.  
ABLATION STUDY OF KTO WITH AUXILIARY SUPERVISION (SUP.).
<table><tr><td>N</td><td>PDMS↑</td><td>Latency↓</td><td>Param.</td></tr><tr><td>128</td><td>88.2</td><td>17.5 ms</td><td></td></tr><tr><td>256</td><td>88.3</td><td>21.4 ms</td><td>113M</td></tr><tr><td>512</td><td>88.4</td><td>24.8 ms</td><td></td></tr></table>

<table><tr><td>Method</td><td>PDMS↑</td></tr><tr><td>w/o Fine-tuning</td><td>88.3</td></tr><tr><td>KTO + Score Sup.</td><td>89.0</td></tr><tr><td>KTO + Reward Sup.</td><td>89.6</td></tr></table>

Table II compares our EMPlan with publicly available representative anchor-based methods with respect to inference efficiency. Notably, our method achieves the lowest latency (21.4 ms) while maintaining the highest PDMS score (89.6). This efficiency gain is primarily attributed to our hybrid architecture, which bypasses the heavy computational burden from dense anchors. In addition, EMPlan relies solely on camera inputs and operates at real-time inference speed, further enhancing its suitability for real-world deployment.

## C. Ablation Study

Table III presents the ablation results of offset-based trajectory refinement and SFLoss. Adding offset results in a notable performance gain, improving PDMS from 81.6 to 87.8. This is primarily because the base model relies on sparse predefined anchors that cannot fully cover diverse driving behaviors, while offset refinement enables fine-grained trajectory adjustments beyond discrete anchor limitations. Incorporating SFLoss further increases PDMS to 88.3 by emphasizing hard samples during training.

Table IV analyzes the effect of the number of predefined anchors N. As Nincreases from 128 to 512, PDMS improves slightly from 88.2 to 88.4, indicating that a denser anchor set provides better coverage of diverse driving behaviors. However, increasing N also leads to noticeably higher inference latency (from 17.5 ms to 24.8 ms), which may hinder efficiency in closed-loop evaluation. Considering both accuracy and efficiency, we adopt N = 256, as it provides a favorable trade-off between accuracy and efficiency.

![](images/2b9aaf49a404bd81b323567a9351567f8431ae873eeb49a513aa79d1d820bfec.jpg)  
Figure 6. Visualization of reward distributions over candidate trajectories, colored by reward value. Unsafe trajectories (e.g., collisions or off-road driving, red circles) are consistently assigned low rewards, while feasible trajectories near the ground truth (yellow circles) receive higher rewards, demonstrating that the reward-guided policy effectively suppresses unsafe behaviors while preserving multimodal diversity.

Table V evaluates different fine-tuning objectives. Compared to the model without fine-tuning, incorporating KTO with auxiliary supervision consistently improves PDMS. In particular, KTO complements the supervision loss by encouraging the policy to favor more desirable trajectories over undesirable ones, while the auxiliary supervision provides dense training signals for stable optimization. Moreover, reward supervision achieves the best performance (89.6), outperforming score supervision (89.0) used in Stage I. This suggests that the aggregated reward signal offers more taskaligned guidance, as it better matches the final trajectory selection criterion used during inference.

To better understand how the model ranks trajectory candidates, Fig. 6 presents the reward distributions over predicted trajectories across diverse scenarios. The learned reward exhibits a clear separation between safe and unsafe trajectories while preserving multimodal feasible solutions.

For instance, in the left-turn scenario (see Fig. 6(c)), trajectories corresponding to the intended turning behavior receive higher rewards, whereas alternative non-turning trajectories are consistently suppressed. Similar patterns are observed in Figs. 6(a, b, d), where trajectories leading to collision or offroad driving are assigned low rewards, while those closer to the ground truth obtain relatively higher rewards.

## V. CONCLUSION

We present EMPlan, an efficient multi-modal trajectory planning framework based on a hybrid architecture and reward-guided fine-tuning. Our method enables real-time multi-modal trajectory generation while improving planning performance beyond pure imitation learning. Experiments on the non-reactive NAVSIM benchmark demonstrate that EMPlan achieves state-of-the-art performance with competitive inference efficiency. A limitation, however, is that our reward-guided training relies on simulator-based evaluation, which prevents us from directly scaling to benchmarks that lack compatible simulation environments. In future work, we plan to explore simulator-free reward modeling strategies to improve training scalability, and we will also explore reactive closed-loop evaluation to better assess how the model behaves when other agents respond to its actions.

## REFERENCES

[1] S. Hu, L. Chen, P. Wu, H. Li, J. Yan, and D. Tao, “ST-P3: Endto-end vision-based autonomous driving via spatial-temporal feature learning,” in ECCV, 2022, pp. 533–549.

[2] K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger, “TransFuser: Imitation with transformer-based sensor fusion for autonomous driving,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 11, pp. 12 878–12 895, 2022.

[3] Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, et al., “Planning-oriented autonomous driving,” in CVPR, 2023, pp. 17 853–17 862.

[4] S. Chen, B. Jiang, H. Gao, B. Liao, Q. Xu, Q. Zhang, C. Huang, W. Liu, and X. Wang, “VADv2: End-to-end vectorized autonomous driving via probabilistic planning,” arXiv preprint arXiv:2402.13243, 2024.

[5] D. Chen and P. Krähenbühl, “Learning from all vehicles,” in CVPR, 2022, pp. 17 222–17 231.

[6] M. Bansal, A. Krizhevsky, and A. Ogale, “ChauffeurNet: Learning to drive by imitating the best and synthesizing the worst,” arXiv preprint arXiv:1812.03079, 2018.

[7] R. Geirhos, J.-H. Jacobsen, C. Michaelis, R. Zemel, W. Brendel, M. Bethge, and F. A. Wichmann, “Shortcut learning in deep neural networks,” Nat. Mach. Intell., vol. 2, no. 11, pp. 665–673, 2020.

[8] Y. Lu, J. Fu, G. Tucker, X. Pan, E. Bronstein, R. Roelofs, B. Sapp, B. White, A. Faust, S. Whiteson, et al., “Imitation is not enough: Robustifying imitation with reinforcement learning for challenging driving scenarios,” in IROS, 2023, pp. 7553–7560.

[9] H. Gao, S. Chen, B. Jiang, B. Liao, Y. Shi, X. Guo, Y. Pu, H. Yin, X. Li, X. Zhang, et al., “RAD: Training an end-to-end driving policy via large-scale 3DGS-based reinforcement learning,” arXiv preprint arXiv:2502.13144, 2025.

[10] S. Ross, G. Gordon, and D. Bagnell, “A reduction of imitation learning and structured prediction to no-regret online learning,” in AISTATS, 2011, pp. 627–635.

[11] Z. Li, K. Li, S. Wang, S. Lan, Z. Yu, Y. Ji, Z. Li, Z. Zhu, J. Kautz, Z. Wu, et al., “Hydra-MDP: End-to-end multimodal planning with multi-target hydra-distillation,” arXiv preprint arXiv:2406.06978, 2024.

[12] Y. Li, Y. Wang, Y. Liu, J. He, L. Fan, and Z. Zhang, “End-to-end driving with online trajectory evaluation via BEV world model,” arXiv preprint arXiv:2504.01941, 2025.

[13] P. F. Christiano, J. Leike, T. Brown, M. Martic, S. Legg, and D. Amodei, “Deep reinforcement learning from human preferences,” NeurIPS, vol. 30, 2017.

[14] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al., “Training language models to follow instructions with human feedback,” NeurIPS, vol. 35, pp. 27 730–27 744, 2022.

[15] W. Xiao, Z. Wang, L. Gan, S. Zhao, Z. Li, R. Lei, W. He, L. A. Tuan, L. Chen, H. Jiang, et al., “A comprehensive survey of direct preference optimization: Datasets, theories, variants, and applications,” arXiv preprint arXiv:2410.15595, 2024.

[16] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn, “Direct preference optimization: Your language model is secretly a reward model,” NeurIPS, vol. 36, pp. 53 728–53 741, 2023.

[17] S. Shang, Y. Chen, Y. Wang, Y. Li, and Z. Zhang, “DriveDPO: Policy learning via safety dpo for end-to-end autonomous driving,” arXiv preprint arXiv:2509.17940, 2025.

[18] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” NeurIPS, vol. 30, 2017.

[19] B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu, C. Huang, and X. Wang, “VAD: Vectorized scene representation for efficient autonomous driving,” in ICCV, 2023, pp. 8340–8350.

[20] X. Li, Y. Zhang, and X. Ye, “DrivingDiffusion: Layout-guided multiview driving scenarios video generation with latent diffusion model,” in ECCV, 2024, pp. 469–485.

[21] B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, et al., “DiffusionDrive: Truncated diffusion model for end-to-end autonomous driving,” in CVPR, 2025, pp. 12 037– 12 047.

[22] B. Jiang, S. Chen, Q. Zhang, W. Liu, and X. Wang, “AlphaDrive: Unleashing the power of vlms in autonomous driving via reinforcement learning and reasoning,” arXiv preprint arXiv:2503.07608, 2025.

[23] D. Li, J. Ren, Y. Wang, X. Wen, P. Li, L. Xu, K. Zhan, Z. Xia, P. Jia, X. Lang, et al., “Finetuning generative trajectory model with reinforcement learning from human feedback,” arXiv preprint arXiv:2503.10434, 2025.

[24] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al., “DeepSeekMath: Pushing the limits of mathematical reasoning in open language models,” arXiv preprint arXiv:2402.03300, 2024.

[25] Y. Li, K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. Wang, et al., “ReCogDrive: A reinforced cognitive framework for end-to-end autonomous driving,” arXiv preprint arXiv:2506.08052, 2025.

[26] Z. Zhou, J. Wang, Y.-H. Li, and Y.-K. Huang, “Query-centric trajectory prediction,” in CVPR, 2023, pp. 17 863–17 873.

[27] D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, et al., “NAVSIM: Datadriven non-reactive autonomous vehicle simulation and benchmarking,” NeurIPS, vol. 37, pp. 28 706–28 719, 2024.

[28] K. Ethayarajh, W. Xu, N. Muennighoff, D. Jurafsky, and D. Kiela, “Model alignment as prospect theoretic optimization,” in ICML, 2024.

[29] X. Weng, B. Ivanovic, Y. Wang, Y. Wang, and M. Pavone, “Para-Drive: Parallelized architecture for real-time autonomous driving,” in CVPR, 2024, pp. 15 449–15 458.

[30] Y. Li, L. Fan, J. He, Y. Wang, Y. Chen, Z. Zhang, and T. Tan, “Enhancing end-to-end autonomous driving with latent world model,” arXiv preprint arXiv:2406.08481, 2024.

[31] C. Yuan, Z. Zhang, J. Sun, S. Sun, Z. Huang, C. D. W. Lee, D. Li, Y. Han, A. Wong, K. P. Tee, et al., “DRAMA: An efficient end-to-end motion planner for autonomous driving with mamba,” arXiv preprint arXiv:2408.03601, 2024.

[32] S. Peng, K. Genova, C. Jiang, A. Tagliasacchi, M. Pollefeys, T. Funkhouser, et al., “OpenScene: 3D scene understanding with open vocabularies,” in CVPR, 2023, pp. 815–824.

[33] H. Caesar, J. Kabzan, K. S. Tan, W. K. Fong, E. Wolff, A. Lang, L. Fletcher, O. Beijbom, and S. Omari, “nuPlan: A closed-loop mlbased planning benchmark for autonomous vehicles,” arXiv preprint arXiv:2106.11810, 2021.

[34] Y. Lee, J.-w. Hwang, S. Lee, Y. Bae, and J. Park, “An energy and gpu-computation efficient backbone network for real-time object detection,” in CVPRW, 2019.