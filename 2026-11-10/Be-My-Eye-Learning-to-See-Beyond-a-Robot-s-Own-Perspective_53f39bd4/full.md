# Be My Eye : Learning to See Beyond a Robot’s Own Perspective

Astribot Team

research@astribot.com

Project Page: www.astribot.com/AI/CoMind

Author List in Contributions

## Abstract

Robots operating in the real world are inherently subject to partial observability. This limitation is particularly pronounced in multi-robot collaboration, where an individual robot’s observations may be insufficient for effective task execution. A central challenge is therefore how to integrate complementary observations from the environment, particularly partner robots, to inform collaborative decision-making. We argue that inter-robot information sharing should go beyond passively transmitting raw observations or generic visual features. Instead, robots should selectively extract and encode task-relevant information for sharing based on their task objectives and current execution progress. To this end, we propose a task-oriented active context sharing framework that encodes robot observations into compact context representations, preserving information critical to collaborative decision-making under a constrained communication budget. Our framework adopts a two-stage training paradigm. In the first stage, onboard robot observations and task instructions are jointly encoded into compact context representations. Task-oriented future prediction and soft task-progress correspondence learning guide this collaboration-agnostic compression process to preserve information relevant to individual robot’s task objectives, state transitions, and execution progress. In the second stage, the learned context representations are integrated into a pretrained Vision-Language-Action (VLA) policy, enabling bidirectional context exchange between robots. Each robot integrates complementary information from its partner with its local observations to guide action generation. Supervision of the resulting actions, in turn, jointly refines the shared representations and the policy, aligning information sharing with the requirements of collaborative decision-making. Our controlled experiments demonstrate substantial performance gains under constrained communication budgets, while three real-world experiments further validate the approach in complex dual-robot collaborative tasks. These results highlight a shift from passive visual information transmission toward active, task-oriented context sharing, in which compact representations are learned to directly support collaborative decision-making.

## 1. Introduction

“Gaze following into joint attention, social manipulation into cooperative communication, group activity into collaboration.”

— Michael Tomasello and Malinda Carpenter, Shared Intentionality

Partial observability is a fundamental constraint on perception in the physical world. For robots, limited fields of view, sensor placement, spatial constraints, and environmental occlusions restrict access to the information needed to execute complex tasks. In multi-robot settings, this limitation also creates an opportunity: robots can exploit complementary observations of their shared environment. Consider two robots collaboratively transporting a large cabinet through a narrow passage. The cabinet may obscure critical regions, leaving obstacles or clearance visible to one robot but hidden from the other. These observations can directly affect whether its partner should slow down, adjust its posture, or change direction. Similar dependencies arise in warehousing, assembly, and everyday collaborative manipulation, where information distributed across robots jointly informs task execution. We argue that the potential of multirobot systems lies not only in their collective physical capabilities but also in the complementary perspectives afforded by distributed sensing. Translating this perceptual complementarity into coordinated action requires each robot to integrate the observations of its partners into its own decision-making, a fundamental challenge and opportunity in tightly coupled multi-robot collaboration.

A common approach to multi-robot coordination relies on centralized planning, in which a high-level planner decomposes global objectives into subtasks or commands that individual robots execute using local observations. Although effective for task allocation and high-level coordination, systems that communicate primarily through task descriptions or discrete instructions provide limited support for sharing fine-grained perceptual information. Object states, spatial relationships, and execution progress observed by one robot may therefore remain unavailable to another robot’s action policy. This limitation is particularly pronounced in dynamic, partially observable environments, where effective collaboration requires continuous adaptation to environmental changes and partners’ actions. Without access to complementary perceptual information, coordination is often restricted to parallel task execution or sequential handoffs, rather than tightly coupled behaviors grounded in shared situational awareness.

Conditioning a robot’s policy on its partners’ visual observations provides a natural mechanism for coupling perception and action across robots. However, transmitting raw images or generic dense visual features incurs substantial communication and computational costs and introduces redundant or task-irrelevant information. Moreover, the value of an observation depends on task objectives, robot roles, and execution progress: visual details essential to one decision may be irrelevant to another. Generic visual compression may preserve appearance or semantic content without retaining the information most useful for collaborative action. The central challenge is therefore to selectively encode local observations into compact representations that preserve their relevance to collaborative decision-making. We refer to this process as active context sharing. Here, active denotes task-conditioned information compression, rather than selection of communication timing or recipients; these decisions may still be managed by a centralized planner. Active context sharing aims to preserve task-relevant perceptual content together with relationships among task objectives, observed states, and execution progress, providing partners with informative context under a limited representation budget.

In this work, we propose a two-stage active context sharing framework for multi-robot collaboration. In Stage 1, we introduce a task-oriented context compression module that uses learnable queries to aggregate frozen multi-camera visual features into compact latent tokens conditioned on task instructions. We train this module with two complementary objectives to preserve task-relevant visual information and capture the structure of task execution: (i) task-oriented future prediction encourages the latent tokens to encode information predictive of future visual states, emphasizing task-relevant regions and state transitions; and (ii) soft task-progress correspondence learning structures the latent space according to task identity and execution progress, promoting inter-task discriminability and intra-task temporal continuity. In Stage 2, we perform joint context optimization and collaborative policy learning. Context tokens produced by the Stage 1 encoder are projected into the hidden space of a pretrained Vision-Language-Action (VLA) policy and integrated into its multi-modal prefix, conditioning action generation alongside local visual observations, task instructions, and proprioceptive states. This interface supports variable numbers of shared tokens without modifying the actiongeneration architecture. Joint optimization of the context encoder and collaborative policy allows the receiving robot’s action supervision to refine the sending robot’s representations for collaborative decision-making. Symmetric encoding and policy interfaces enable bidirectional sharing, allowing each robot to incorporate complementary context from its partners.

We systematically evaluate our framework across diverse real-world dual-robot manipulation tasks involving visual occlusion, spatial separation, state-dependent interactions, and role coordination, all of which require robots to leverage complementary observations for successful execution. We compare our approach against communication-free policies and alternative information-sharing strategies under varying communication budgets, while conducting ablation studies to assess the contribution of key design choices.

## Our main contributions are as follows:

1. Active context sharing framework for multi-robot collaboration. We propose a twostage training framework that encodes each robot’s local observations into compact, taskconditioned latent representations, which are integrated into partner robots’ action policies and jointly optimized through downstream action supervision. This approach complements high-level coordination with shared perceptual context, enabling information from complementary observations to directly inform collaborative action generation.

2. Progress-aware, task-conditioned observation compression. We introduce a query-based compression module supporting joint and per-camera visual encoding. Task-oriented future prediction and soft task-progress correspondence learning jointly encourage the retention of predictive visual information, inter-task discriminability, and intra-task temporal continuity under a fixed representation budget.

3. Joint context optimization for collaborative policy learning. We integrate compressed context into pretrained VLA policies through a flexible conditioning interface supporting variable token budgets and bidirectional context sharing. Action supervision from the receiving robot jointly optimizes the context encoder and policy, aligning shared representations with the requirements of collaborative decision-making.

4. Real-world evaluation and representation analysis. We systematically evaluate our framework on dual-robot manipulation tasks through comparisons against no-sharing and alternative information-sharing baselines, communication-budgets, and component-wise ablation studies.

![](images/9d4f16b914667f73882c156f6f16ee695cb97759d3579f27a80842390820eb40.jpg)  
Figure 1 | Overview of the two-stage active context sharing framework. In Stage 1, multicamera observations and task instructions are compressed into shared context, optimized through weighted future visual feature prediction and soft-label contrastive learning. In Stage 2, robots bidirectionally exchange local context and condition their VLA policies on shared context to generate continuous actions for collaborative decision-making.

## 2. Method

## 2.1. Problem Formulation and Framework Overview

We study active context sharing for multi-robot collaboration, where robots exchange taskrelevant latent representations to complement each other’s local observations and improve collaborative decision-making. Throughout this paper, �, � denote robot indices, � denotes the time step, � denotes the camera index, � denotes the visual patch index, and �, � denote sample indices within a training batch.

At time �, robot � receives local visual observations from � cameras:

$$
\mathbf { V } _ { i , t } = \left\{ \mathbf { V } _ { i , t } ^ { ( c ) } \right\} _ { c = 1 } ^ { C } , \qquad \mathbf { V } _ { i , t } ^ { ( c ) } \in \mathbb { R } ^ { H \times W \times 3 } ,\tag{1}
$$

where � and � denote the image height and width, respectively, along with a language instruc-

tion $\ell _ { i }$ and proprioceptive state $\mathbf { s } _ { i , t }$ .

Due to limited fields of view, spatial constraints, and environmental occlusions, local observations may not provide sufficient information for collaborative task execution. To address this, a collaborating robot � encodes its multi-camera observations and task instruction into a task-conditioned latent representation $\mathbf { Z } _ { j , t } ,$ which serves as additional context for robot �’s Vision-Language-Action (VLA) policy:

$$
\mathbf { A } _ { i , t } = \pi _ { \theta } \left( \mathbf { V } _ { i , t } , \ell _ { i } , \mathbf { s } _ { i , t } , \mathbf { Z } _ { j , t } ^ { p r o j } \right) , \qquad \mathbf { A } _ { i , t } \in \mathbb { R } ^ { H _ { a } \times d _ { a } } ,\tag{2}
$$

where $\pi _ { \theta }$ denotes the VLA policy, ${ \bf Z } _ { j , t } ^ { p r o j }$ represents the shared latent context projected into the policy’s hidden space, and $H _ { a }$ and $\dot { d _ { a } }$ denote the action prediction horizon and per-step action dimension, respectively.

We refer to the task-conditioned encoding of local observations into shared latent representations as active context sharing. Here, active characterizes how shared information is constructed, rather than autonomous decisions about when or with whom to communicate.

Our framework adopts symmetric, bidirectional latent sharing. For two collaborating robots, A and $\scriptstyle \mathrm { \mathrm { B } , }$ their action predictions are given by

$$
{ \bf A } _ { A , t } = \pi _ { \theta } \left( { \bf V } _ { A , t } , \ell _ { A } , { \bf s } _ { A , t } , { \bf Z } _ { B , t } ^ { p r o j } \right) ,\tag{3}
$$

$$
\mathbf { A } _ { B , t } = \pi _ { \theta } \left( \mathbf { V } _ { B , t } , \ell _ { B } , \mathbf { s } _ { B , t } , \mathbf { Z } _ { A , t } ^ { p r o j } \right) .\tag{4}
$$

Each robot thus both contributes to and benefits from shared contextual information, without requiring predefined leader–follower roles.

Our framework adopts a two-stage training paradigm. In Stage 1, task-conditioned context tokens are optimized to preserve information relevant to task objectives and state transitions under a limited representation budget, while promoting inter-task discriminability and intratask temporal continuity. In Stage 2, these tokens are integrated into a pretrained VLA policy, and downstream action supervision jointly refines the shared context representations and policy for collaborative decision-making.

## 2.2. Task-Oriented Multi-Camera Context Encoding

Given a set of multi-camera observations $\mathbf { V } _ { i , t } = \{ \mathbf { V } _ { i , t } ^ { ( c ) } \} _ { c = 1 } ^ { C }$ and a task instruction $\ell _ { i } ,$ our objective is to encode the local observations of robot � into a compact context representation that preserves information relevant to the current task. Frozen visual and language encoders extract generic feature representations from the observations and task instructions, respectively:

$$
\mathbf { F } _ { i , t } ^ { ( c ) } = f _ { \mathrm { v i s } } ( \mathbf { V } _ { i , t } ^ { ( c ) } ) , \qquad \mathbf { L } _ { i } = f _ { \mathrm { t e x t } } ( \ell _ { i } ) ,\tag{5}
$$

where $\mathbf { F } _ { i , t } ^ { ( c ) } \in \mathbb { R } ^ { N _ { p } \times d _ { f } }$ and $\mathbf { L } _ { i } \in \mathbb { R } ^ { N _ { \ell } \times d _ { \ell } }$ denote the visual and linguistic features, respectively.

To extract task-relevant information under a fixed representation budget, a trainable context encoder $E _ { \phi }$ aggregates visual and linguistic features using � learnable queries $\mathbf { Q } \in \mathbb { R } ^ { K \times d _ { \mathrm { i } } }$ <sup>�</sup> . The resulting task-conditioned context tokens are given by:

$$
\begin{array} { r l r } { \mathbf { Z } _ { i , t } = E _ { \phi } \left( \mathbf { Q } , \{ \mathbf { F } _ { i , t } ^ { ( c ) } \} _ { c = 1 } ^ { C } , \mathbf { L } _ { i } \right) , } & { } & { \mathbf { Z } _ { i , t } \in \mathbb { R } ^ { K \times d _ { z } } . } \end{array}\tag{6}
$$

The context encoder supports both joint and per-camera query allocation. Joint encoding uses a shared set of queries to aggregate observations from all cameras into � context tokens. Percamera encoding assigns $K _ { c }$ queries to each camera � and concatenates the resulting tokens, yielding a total token budget of:

$$
K = \sum _ { c = 1 } ^ { C } K _ { c } .\tag{7}
$$

In both configurations, robots exchange the resulting context representation $\mathbf { Z } _ { i , t }$ rather than raw visual observations or generic visual features.

Finally, a learnable projection module $P _ { \rho }$ maps the context tokens into the hidden representation space of the downstream VLA policy:

$$
\begin{array} { r } { \mathbf { Z } _ { i , t } ^ { p r o j } = P _ { \rho } ( \mathbf { Z } _ { i , t } ) , \qquad \mathbf { Z } _ { i , t } ^ { p r o j } \in \mathbb { R } ^ { K \times d _ { \pi } } , } \end{array}\tag{8}
$$

where $d _ { \pi }$ denotes the hidden dimension of the VLA policy. The projected context tokens serve as additional conditioning during policy learning, supporting collaborative decision-making.

## 2.3. Stage 1: Collaboration-Agnostic Context Compression

Stage 1 focuses on optimizing a compact, task-oriented representation of each robot’s local observations under a fixed token budget, without assuming or modeling any specific collaboration pattern. We achieve this through two complementary objectives: (i) task-oriented future feature prediction, which encourages the representation to preserve predictive information about task-relevant visual state transitions, and (ii) soft-label contrastive learning, which structures the representation space according to task identity and execution-stage proximity. Together, these objectives yield compact, generalizable representations that capture task-relevant dynamics and preserve meaningful relationships across tasks and execution stages, providing a reusable basis for downstream collaborative policy learning.

## 2.3.1. Task-Oriented Future Prediction

Given a temporal offset Δ, we define the future visual feature target for camera � as

$$
\mathbf { F } _ { i , t + \Delta } ^ { ( c ) } = f _ { \mathrm { v i s } } \left( \mathbf { V } _ { i , t + \Delta } ^ { ( c ) } \right) , \qquad \mathbf { F } _ { i , t + \Delta } ^ { ( c ) } \in \mathbb { R } ^ { N _ { p } \times d _ { f } } .\tag{9}
$$

The frozen visual encoder maps future observations to a fixed feature space, providing selfsupervised prediction targets without exposing future observations to the context encoder.

A predictor $D _ { \psi }$ estimates future visual features for all cameras from the current context tokens:

$$
\widehat { \mathbf { F } } _ { i , t + \Delta } = D _ { \psi } ( \mathbf { Z } _ { i , t } ) = \left\{ \widehat { \mathbf { F } } _ { i , t + \Delta } ^ { ( c ) } \right\} _ { c = 1 } ^ { C } .\tag{10}
$$

To prioritize task-relevant visual content, we compute patch-level relevance scores using a pretrained vision-language model with aligned visual and textual representations. Specifically, we measure the similarity between each visual patch and the corresponding task instruction in the aligned embedding space. The resulting similarity scores are normalized to [0, 1] and converted into bounded weights for future visual feature prediction. Let

$$
r _ { n } ^ { ( c , p ) } \in [ 0 , 1 ]\tag{11}
$$

denote the normalized relevance of patch $p$ from camera � for training sample �. We map these scores to bounded weights:

$$
w _ { n } ^ { ( c , p ) } = w _ { \mathrm { m i n } } + ( w _ { \mathrm { m a x } } - w _ { \mathrm { m i n } } ) r _ { n } ^ { ( c , p ) } ,\tag{12}
$$

where $0 \leq w _ { \mathrm { m i n } } \leq w _ { \mathrm { m a x } } .$ . The weights are detached from the computational graph during prediction-loss computation, preventing gradients from this objective from propagating through the relevance-scoring branch.

Let $\Omega _ { n }$ denote the set of valid camera-patch pairs for sample �. The normalized squared prediction error for each patch is:

$$
e _ { n } ^ { ( c , p ) } = \frac { 1 } { d _ { f } } \left\| \widehat { \mathbf { F } } _ { n } ^ { ( c , p ) } - \mathbf { F } _ { n } ^ { ( c , p ) } \right\| _ { 2 } ^ { 2 } .\tag{13}
$$

For a batch of $B _ { \mathrm { p r e d } }$ valid samples, the task-relevance-weighted prediction objective is:

$$
\mathcal { L } _ { \mathrm { p r e d } } = \frac { 1 } { B _ { \mathrm { p r e d } } } \sum _ { n = 1 } ^ { B _ { \mathrm { p r e d } } } \frac { \displaystyle \sum _ { ( c , p ) \in \Omega _ { n } } w _ { n } ^ { ( c , p ) } e _ { n } ^ { ( c , p ) } } { \displaystyle \sum _ { ( c , p ) \in \Omega _ { n } } w _ { n } ^ { ( c , p ) } } .\tag{14}
$$

This objective assigns greater predictive importance to task-relevant patches, encouraging the context encoder to preserve information predictive of task-relevant visual state transitions under a fixed token budget.

## 2.3.2. Soft Task-Progress Correspondence Learning

To impose additional structure on the context representation space, we construct soft correspondence targets based on task identity and execution progress. Given a multi-camera observation $\mathbf { V } _ { n } ,$ we generate two independently augmented views:

$$
\widetilde { \mathbf { V } } _ { n } ^ { ( \nu ) } = \mathcal { T } _ { \nu } ( \mathbf { V } _ { n } ) , \qquad \nu \in \{ 1 , 2 \} ,\tag{15}
$$

where $\mathcal { T } _ { \nu }$ denotes a stochastic vision augmentation operator. All augmented views are encoded using the same task instruction and context encoder:

$$
{ \bf Z } _ { n } ^ { \left( \nu \right) } = E _ { \phi } \left( { \bf Q } , \widetilde { { \bf F } } _ { n } ^ { \left( \nu \right) } , { \bf L } _ { n } \right) ,\tag{16}
$$

where $\widetilde { \mathbf { F } } _ { n } ^ { \left( \nu \right) }$ denotes the visual features extracted from the augmented observations $\widetilde { \mathbf { V } } _ { n } ^ { ( \nu ) }$

Normalized contrastive embeddings are obtained by pooling the context tokens, applying a projection head $g _ { \xi } ,$ , and performing $\ell _ { 2 }$ normalization:

$$
\mathbf { h } _ { n } ^ { ( \nu ) } = \frac { g _ { \xi } \left( \mathrm { P o o l } ( \mathbf { Z } _ { n } ^ { ( \nu ) } ) \right) } { \left\| g _ { \xi } \left( \mathrm { P o o l } ( \mathbf { Z } _ { n } ^ { ( \nu ) } ) \right) \right\| _ { 2 } } .\tag{17}
$$

The projection head is used only for contrastive supervision; the complete context tokens remain the representation exchanged between robots.

![](images/ef48b6221860cabd03f443395fe7019ec459222d030fd93d055a4794035ac5c0.jpg)  
Figure 2 | Soft Task-Progress Correspondence Learning. For each observation, augmented views are encoded by the task-conditioned context encoder and treated as positive pairs. Taskand progress-based affinities define the similarity matrix for contrastive learning, promoting cross-view consistency, inter-task discriminability, and intra-task temporal continuity.

Let $B _ { \mathrm { c o n } }$ denote the contrastive batch size. For sample $n ,$ let $y _ { n }$ denote its task identity and $\gamma _ { n } \in$ [0, 1] its normalized execution progress. We compute the execution progress as $\gamma _ { n } = t _ { n } / ( T _ { n } - 1 )$ where $t _ { n }$ is the zero-based frame index of sample � and $T _ { n }$ is the total number of frames in its demonstration trajectory.We define the pairwise affinity between samples � and � as:

$$
S _ { n m } = \mathbb { I } [ y _ { n } = y _ { m } ] \kappa ( \gamma _ { n } , \gamma _ { m } ) ,\tag{18}
$$

where I[·] is the indicator function and � is a nonnegative similarity kernel that decreases with execution-progress distance. Execution progress is used exclusively to construct training targets and is not provided to the context encoder or communicated between robots.

We normalize the affinity matrices row-wise to obtain directional target distributions:

$$
\mathbf { R } ^ { 1 \to 2 } = \operatorname { R o w N o r m } ( \mathbf { S } ) , \qquad \mathbf { R } ^ { 2 \to 1 } = \operatorname { R o w N o r m } ( \mathbf { S } ^ { \top } ) .\tag{19}
$$

Here, RowNorm(·) denotes row-wise normalization, with a suitable safeguard for rows whose affinity sums are zero.

To balance instance-level correspondence with progress-based similarity, we define the soft targets as:

$$
\mathbf { P } ^ { d } = \alpha \mathbf { I } + ( 1 - \alpha ) \mathbf { R } ^ { d } , \qquad d \in \{ 1 \to 2 , 2 \to 1 \} ,\tag{20}
$$

where $\alpha \in [ 0 , 1 ]$ controls the relative contribution of the two components. Since both I and $\mathbf { R } ^ { d }$ are row-stochastic, $\mathbf { P } ^ { d }$ is also row-stochastic.

The predicted cross-view correspondence distributions are computed using temperature-scaled cosine similarities:

$$
Q _ { n m } ^ { 1 \to 2 } = \frac { \exp \left( ( \mathbf { h } _ { n } ^ { ( 1 ) } ) ^ { \top } \mathbf { h } _ { m } ^ { ( 2 ) } / \tau \right) } { \displaystyle \sum _ { j = 1 } ^ { B _ { \mathrm { c o n } } } \exp \left( ( \mathbf { h } _ { n } ^ { ( 1 ) } ) ^ { \top } \mathbf { h } _ { j } ^ { ( 2 ) } / \tau \right) } ,\tag{21}
$$

$$
Q _ { n m } ^ { 2 \to 1 } = \frac { \exp \left( ( \mathbf { h } _ { n } ^ { ( 2 ) } ) ^ { \top } \mathbf { h } _ { m } ^ { ( 1 ) } / \tau \right) } { \displaystyle \sum _ { j = 1 } ^ { B _ { \mathrm { c o n } } } \exp \left( ( \mathbf { h } _ { n } ^ { ( 2 ) } ) ^ { \top } \mathbf { h } _ { j } ^ { ( 1 ) } / \tau \right) } ,\tag{22}
$$

where $\tau > 0$ is the contrastive temperature.

The bidirectional soft-label contrastive objective is defined as:

$$
\mathcal { L } _ { \mathrm { c o n } } = - \frac { 1 } { 2 B _ { \mathrm { c o n } } } \sum _ { d \in \{ 1  2 , 2  1 \} } \sum _ { n = 1 } ^ { B _ { \mathrm { c o n } } } \sum _ { m = 1 } ^ { B _ { \mathrm { c o n } } } P _ { n m } ^ { d } \log Q _ { n m } ^ { d } .\tag{23}
$$

This objective encourages consistent representations across augmented views while aligning samples according to task identity and execution-stage proximity. Samples from the same task and nearby execution stages receive greater target affinity, whereas samples from different tasks are assigned zero affinity before instance-level mixing. The resulting supervision promotes task-discriminative representations with structured temporal relationships.

The overall Stage 1 objective is:

$$
\mathcal { L } _ { \mathrm { S 1 } } = \lambda _ { \mathrm { p r e d } } \mathcal { L } _ { \mathrm { p r e d } } + \lambda _ { \mathrm { c o n } } \mathcal { L } _ { \mathrm { c o n } } ,\tag{24}
$$

where $\lambda _ { \mathrm { p r e d } }$ and $\lambda _ { \mathrm { c o n } }$ control the relative contributions of future feature prediction and soft-label contrastive learning, respectively.

## 2.4. Stage 2: Joint Context Optimization and Collaborative Policy Learning

For illustration, we consider robot � as the context sender and robot � as the context receiver. Following Stage 1, the context encoder generates task-oriented context tokens from the sender’s current multi-camera observations and task instruction:

$$
{ \bf Z } _ { j , t } = E _ { \phi } \left( { \bf Q } , { \bf F } _ { j , t } , { \bf L } _ { j } \right) .\tag{25}
$$

A projection module then maps the context tokens into the hidden space of the VLA policy:

$$
{ \bf Z } _ { j , t } ^ { p r o j } = { \cal P } _ { \rho } \left( { \bf Z } _ { j , t } \right) .\tag{26}
$$

The receiving robot � integrates the shared tokens with its local multi-camera observations, task instruction, and proprioceptive state as inputs to the VLA policy. Conditioned on these inputs, the policy predicts the action sequence $\mathbf { A } _ { i , t }$ as:

$$
\mathbf { A } _ { i , t } = \pi _ { \theta } \left( \mathbf { V } _ { i , t } , \ell _ { i } , \mathbf { s } _ { i , t } , \mathbf { Z } _ { j , t } ^ { p r o j } \right) .\tag{27}
$$

For each sender–receiver direction $j \to i ,$ we define the Stage 2 training objective as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { S 2 } } ^ { j  i } = \mathcal { L } _ { \mathrm { V L A } } ^ { i } + \eta \mathcal { L } _ { \mathrm { S 1 } } ^ { j } , } \end{array}\tag{28}
$$

where $\mathcal { L } _ { \mathrm { V L A } } ^ { i }$ denotes the action learning objective of the receiving robot, $\mathcal { L } _ { \mathrm { S 1 } } ^ { j }$ denotes the auxiliary Stage 1 objective computed from the sender’s observations and task instruction, and $\eta \geq 0$ controls the contribution of the auxiliary loss.

During joint optimization, the downstream action loss propagates through the shared context tokens to the sender’s encoder, adapting the transmitted representations to the receiver’s collaborative action generation. Meanwhile, future visual feature prediction and soft-label contrastive learning serve as auxiliary objectives that preserve task-relevant perceptual information and the structural relationships between task identity and execution progress. Thus, Stage 2 optimizes the representations not merely for perceptual compression, but also for their utility in downstream collaborative decision-making.

We adopt a symmetric parameterization for the two communication directions, $A  B$ and $B  A ,$ using shared encoder, projection, and policy parameters. At inference time, the two robots independently generate their task-conditioned context tokens:

$$
{ \bf Z } _ { A , t } = E _ { \phi } \left( { \bf Q } , { \bf F } _ { A , t } , { \bf L } _ { A } \right) , \qquad { \bf Z } _ { B , t } = E _ { \phi } \left( { \bf Q } , { \bf F } _ { B , t } , { \bf L } _ { B } \right) .\tag{29}
$$

Each robot then conditions its policy on the context received from its partner:

$$
{ \bf A } _ { A , t } = \pi _ { \theta } \left( { \bf V } _ { A , t } , \ell _ { A } , { \bf s } _ { A , t } , { \bf Z } _ { B , t } ^ { p r o j } \right) ,\tag{30}
$$

$$
\mathbf { A } _ { B , t } = \pi _ { \theta } \left( \mathbf { V } _ { B , t } , \ell _ { B } , \mathbf { s } _ { B , t } , \mathbf { Z } _ { A , t } ^ { p r o j } \right) .\tag{31}
$$

This formulation enables bidirectional context sharing, allowing each robot to supplement its local observations with task-relevant information from its partner’s complementary observations. The future feature prediction decoder $D _ { \psi }$ and contrastive projection head $g _ { \xi }$ are used only during training and are omitted during cross-robot communication and action inference.

## 3. Experiments

## 3.1. Experimental Setup

We evaluate the proposed framework on real-world dual-robot manipulation tasks using two Astribot S1 robots (Gao et al., 2025). At each decision step, each robot receives its local camera observations, task instruction, and proprioceptive state, which are jointly encoded into taskconditioned context tokens and subsequently exchanged between the two robots. Each robot integrates the received context into a pretrained $\pi _ { 0 . 5 }$ policy to generate continuous actions. The framework does not impose predefined sender–receiver roles: both robots can simultaneously encode, exchange, and utilize contextual information from their partners, with their respective active and supporting roles emerging dynamically from the current task stage.

During training, demonstrations collected from both robots are consolidated into a unified dataset, enabling the model to learn context encoding and action generation across different role configurations. Unless otherwise specified, all methods share the same data and action pre-processing pipelines, policy initialization, and optimization hyperparameters to ensure a controlled and fair comparison. The default context token budget is set to � = 32.

![](images/320f6400c607b08e864d3d68591377bfbbdd4a66c4271b7bc34399cb694f02ba.jpg)  
Figure 3 | Representative rollouts of designed collaborative tasks. Keyframes highlight critical moments of inter-robot context sharing during task execution.

We design two dual-robot tasks to evaluate the effectiveness of cross-robot context sharing. The first is an occluded-target selection task, which evaluates whether shared context tokens can convey target information that is unavailable in the acting robot’s local observations. The second is an alternating leader-follower marker retrieval task, which further investigates whether target identity and the partner’s execution progress can support bidirectional information sharing and continuous collaborative control. The former minimizes the influence of action coordination, isolating the effectiveness of cross-robot information transfer, whereas the latter requires the two robots to alternate between leader and follower roles within a single task sequence, evaluating whether shared context can directly inform action generation. Representative rollouts are shown in Fig. 3.

The main comparison settings are defined as follows: (1) No Sharing, where each robot relies exclusively on its local visual observations, task instruction, and proprioceptive state; (2) Generic SigLIP, where the partner’s visual tokens are directly shared without task-oriented compression; (3) Ours-Frozen, where the context tokens learned in Stage 1 are shared while the context encoder remains frozen during policy training; and (4) Ours-Joint, where the context encoder and VLA policy are jointly optimized, allowing downstream action supervision to further shape the information encoded in the shared latent tokens.

Table 1 | Occluded-target selection results.

<table><tr><td>Method</td><td>Target accuracy (%)</td></tr><tr><td>No Sharing</td><td>38.9</td></tr><tr><td>Generic SigLIP</td><td>72.2</td></tr><tr><td>Ours-Joint</td><td>88.9</td></tr></table>

## 3.2. Occluded-Target Selection

The occluded-target selection task involves two robots standing face-to-face, with three boxes placed between them. A human repeatedly places a blue plate into one of the boxes. This targetdefining action is visible to Robot A but occluded from Robot B. For each experimental setting, we conduct 18 trials. This task provides a controlled evaluation of whether shared context tokens can convey target information unavailable from the acting robot’s local observations. Robot A remains stationary and does not participate in object manipulation, but has access to the visual cues needed to identify the correct target box. Robot B performs the pick-and-place operation, while its local observations alone are insufficient to disambiguate the destination. Successful task completion therefore requires Robot B to leverage the context tokens shared by Robot A to resolve the target ambiguity and select the correct box.

The information-providing robot is intentionally kept stationary to minimize confounding factors arising from action synchronization, physical interaction, and control errors. This task is designed not to evaluate the execution of basic pick-and-place skills, but to isolate whether the shared representations can convey the target-defining information needed by an otherwise capable manipulation policy to select the correct destination among the three available boxes.

We use target-box selection accuracy as the primary evaluation metric. Table 1 summarizes the results across different settings. Ours-Joint substantially outperforms No Sharing, demonstrating that compact context tokens effectively convey complementary, target-relevant information beyond the acting robot’s local observations. This interpretation is further supported by the be havior under No Sharing: the robot tends to select the nearest box, suggesting that its decisions rely primarily on local spatial cues rather than the target-defining information available from its partner. To evaluate the effectiveness of our representation against high-bandwidth visual sharing, we further compare Ours-Joint with Generic SigLIP, which serves as a visual-sharing reference with a token budget of 160. Despite using only 32 shared context tokens, Ours-Joint achieves a target selection accuracy of 88.9%, outperforming Generic SigLIP by 16.7 percentage points. This result suggests that simply transmitting more visual information does not necessarily lead to better collaborative decisions. Instead, task-oriented context compression enables more effective transmission of target-defining information, substantially improving cross-robot information sharing while reducing the token budget by 80%.

## 3.3. Alternating Leader–Follower Marker Retrieval

This task is designed to evaluate whether shared context tokens convey both target identity and the partner’s execution progress, thereby supporting bidirectional information sharing and continuous collaboration during role switching. Three colored markers (red, blue, and black) are placed in front of each robot, A and B. Each trial consists of two consecutive stages, during which the robots alternate between Leader and Follower roles. Taking Robot A as the initial Leader, in the first stage, Robot A follows its task instruction to select and grasp a marker of the specified color $c _ { A }$ . Robot B acts as the Follower and must infer from Robot A’s shared context whether the grasp has been completed and when to grasp a marker of the same color. Robot A remains stationary throughout this stage. In the second stage, the roles are reversed. Robot B becomes the Leader and selects and grasps a marker of a different color $c _ { B } ,$ where $c _ { B } \neq c _ { A } ,$ from the two remaining colors. Robot A then acts as the Follower, using Robot B’s shared context to infer the selected color and determine when to initiate its grasp. A complete sequence initiated by Robot A is denoted as ABBA, representing the order in which the two robots initiate their actions. Importantly, each robot’s language instruction specifies only the target color for its own Leader stage. As a Follower, it receives neither an explicit target-color instruction nor an action-trigger command. Instead, it must infer two key pieces of information from its partner’s shared context tokens: target identity, i.e., the selected marker color, and execution progress, i.e., whether the partner has completed the grasp and when to initiate its own action. We evaluate all six ordered combinations of distinct marker colors, with both Robot A and Robot B serving as the initial Leader. Each configuration is repeated four times, resulting in 48 trials.

Table 2 | Alternating leader–follower marker-retrieval results.
<table><tr><td>Method</td><td>B follows A (%)</td><td>A follows B (%)</td><td>Trigger (%)</td><td>Sequence Success Rate (%)</td></tr><tr><td>No Sharing</td><td>16.7</td><td>33.3</td><td>25</td><td>0</td></tr><tr><td>Generic SigLIP</td><td>100</td><td>100</td><td>100</td><td>100</td></tr><tr><td>Ours-Joint</td><td>100</td><td>100</td><td>100</td><td>100</td></tr></table>

Under No Sharing, the initial Leader successfully grasps the marker specified by its instruction and remains stationary. The initial Follower typically waits for a period before initiating its action. However, without access to its partner’s target selection or execution status, it tends to retrieve the color specified in its own subsequent Leader-stage instruction rather than match its partner’s current target. This behavior suggests that the policy retains basic grasping capabilities and can learn coarse temporal patterns, such as delayed action initiation, but cannot reliably condition its behavior on the partner’s target identity and real-time execution progress.

We evaluate collaboration using four metrics: color-matching accuracy when Robot B follows Robot A, color-matching accuracy when Robot A follows Robot B, correct-trigger rate, and overall two-stage sequence success rate. A trial is considered successful only when both stages are completed in the correct order and with the corresponding target colors. Together, these metrics assess bidirectional target inference, progress-aware action triggering, and their combined effectiveness in completing the full collaborative sequence.

Normalized Process Score. To quantify partial completion of the alternating leader–follower sequence, we assign 0.25 points for each successful Leader grasp, 0.5 points for each correctly triggered Follower action, and 1.0 point for each correctly matched Follower grasp.

As shown in Table 2, No Sharing achieves low color-matching accuracy in both following directions (16.7% for B follows A and 33.3% for A follows B), together with a trigger accuracy of only 25% and zero full-sequence successes. These results reveal that recognizing the task structure and learning coarse temporal patterns are insufficient for reliable role-switching collaboration: the Follower must access both the partner’s target identity and execution progress to coordinate its actions correctly. In contrast, both Generic SigLIP and Ours-Joint achieve 100% across all four metrics, demonstrating reliable bidirectional target inference and progress-aware action triggering. Notably, Ours-Joint matches the performance of Generic SigLIP while using only 32 shared context tokens, compared with 160 visual tokens. This result demonstrates that task-oriented context compression preserves the information necessary for successful bidirectional collaboration with an 80% reduction in communication budget.

## 3.4. Ablation of Stage 1 Training Objectives

This section investigates the contributions of the two representation-learning objectives, taskoriented future prediction and task-progress contrastive learning, on the Alternating Leader–Follower Marker Retrieval task. We consider two ablation variants: w/o Prediction, which removes future prediction while retaining the complete contrastive objective, and w/o Contrastive, which removes both self-pair correspondence and task-conditioned progress correspondence while retaining future prediction. The full method combines both objectives.

Each variant undergoes Stage 1 pretraining with its corresponding objectives, followed by joint Stage 2 training. The context encoder remains trainable in all settings. During Stage 2, receiverside action supervision is combined with the auxiliary objectives retained by each variant, while the removed objectives remain disabled. This comparison isolates the contribution of each representation-learning objective within the complete two-stage training pipeline.

Table 3 | Ablation of Stage 1 Training Objectives.
<table><tr><td>Setting</td><td>Process Score (%)</td></tr><tr><td>Full</td><td>100</td></tr><tr><td>w/o Prediction</td><td>96.7</td></tr><tr><td>w/o Contrastive</td><td>68.9</td></tr></table>

As shown in Table 3, removing contrastive learning substantially reduces the Process Score from 100.0% to 68.9%, whereas removing future prediction leads to a smaller decrease to 96.7%. This asymmetry suggests that task-progress contrastive learning plays a more critical role in this task, where successful collaboration requires the Follower to distinguish both the partner’s target identity and execution phase. By enforcing task-conditioned correspondence across observations, contrastive learning helps organize shared representations according to task identity and progress. In comparison, future prediction encourages the preservation of predictive information about relevant visual dynamics. Although its individual contribution is smaller, the full method still outperforms the contrastive-only variant, indicating that predictive supervision provides complementary benefits beyond task-progress alignment. Together, the two objectives guide the representation to learn both what information to preserve and how to organize it for collaborative decision-making.

## 3.5. Joint Optimization and Communication Capacity

## 3.5.1. Frozen Context vs. Joint Optimization

We further compare Ours-Frozen and Ours-Joint on the Alternating Leader-Follower Marker Retrieval task to investigate whether downstream action supervision can further refine the representations learned in Stage 1. Ours-Frozen keeps the Stage 1 context encoder fixed and trains only the VLA policy conditioned on the shared context tokens. In contrast, Ours-Joint allows action gradients to propagate through the shared context tokens into the context encoder, while retaining the Stage 1 future prediction and progress correspondence objectives. This comparison examines both the transferability of Stage 1 representations to downstream collaborative tasks and the extent to which action supervision can further adapt these representations to support continuous collaborative decision-making.

Table 4 | Frozen context vs. joint optimization.
<table><tr><td>Setting</td><td>Process Score (%)</td></tr><tr><td>Ours-Frozen</td><td>83.3</td></tr><tr><td>Ours-Joint</td><td>100</td></tr></table>

As shown in Table 4, Ours-Frozen achieves a Process Score of 83.3%, indicating that the representations learned in Stage 1 already capture useful information for downstream collaborative decision-making without further adaptation. Joint Stage 2 training further improves the score to 100% with a performance gain of 16.7 percentage points. This improvement demonstrates that receiver-side action supervision further adapts the shared context representations to the requirements of collaborative policy execution.

## 3.5.2. Context Token Budget

To investigate the relationship between communication capacity and task performance, we vary the number of shared context tokens, $K \in \{ 4 , 1 6 , 3 2 \}$ , on the Alternating Leader-Follower Marker Retrieval task, using Generic SigLIP as a high-bandwidth visual-sharing reference. All token-budget variants of our method share the same feature dimension and training protocol, differing only in the number of transmitted tokens. We evaluate each setting using the Process Score and report the corresponding token count and communication cost.Communication cost is calculated based on the transmitted token representations. Each token contains 1152 floatingpoint values stored in FP32 format (4 bytes). The communication cost per direction and decision step is therefore computed as $K \times 1 1 5 2 \times 4$ bytes, excluding protocol overhead.

Table 5 | Context token budget vs. task success rate.
<table><tr><td>Setting</td><td># Tokens</td><td>Process Score (%)</td><td>Bytes/step (KiB)</td></tr><tr><td>Ours-K=4</td><td>4</td><td>90.5</td><td>18</td></tr><tr><td>Ours-K=16</td><td>16</td><td>97.6</td><td>72</td></tr><tr><td>Ours-K=32</td><td>32</td><td>100</td><td>144</td></tr><tr><td>Generic SigLIP</td><td>160</td><td>100</td><td>720</td></tr></table>

As shown in Table 5, increasing the context token budget consistently improves task performance, revealing a trade-off between communication capacity and collaborative decision-making. Notably, even with only 4 tokens, our method achieves a Process Score of 90.5%, indicating that task-relevant information can be encoded in a highly compact representation. Increasing the budget to 16 tokens further improves the score to 97.6%, while 32 tokens yields 100% performance, matching Generic SigLIP with 80% fewer transmitted tokens. This progression suggests that additional tokens progressively improve the preservation of decision-relevant information, with diminishing performance gains at larger budgets. We therefore adopt � = 32 as the default, as it is the smallest evaluated budget that matches the high-bandwidth reference. Overall, these results demonstrate that task-oriented context compression achieves comparable collaborative performance with substantially lower communication overhead.

## 3.6. Qualitative Analysis

To further demonstrate the applicability of our approach across different collaborative scenarios, we present three representative dual-robot tasks in Fig. 4: (1) refrigerator restocking, illustrating efficient information exchange between robots; (2) box assembly, demonstrating complementary perception and collaborative manipulation under occlusion; and (3) blanket folding and packing, illustrating synchronized dual-robot actions during continuous task execution. These tasks represent different collaborative requirements, including information exchange, occlusion-aware collaboration, and synchronized actions. Representative keyframes illustrate the execution process of each task.

To further examine visual attention during collaboration, we visualize the patch-level attention scores extracted from the seventh layer of the VLA policy during action generation. Specifically, we pair the main-camera observations of Robots A and B at the same timestep and visualize their corresponding attention distributions. Regions with higher attention scores indicate the visual areas emphasized by each policy during action generation. By comparing the two views at the same collaborative moment, we examine whether the robots attend to complementary task-relevant information, such as target objects and interaction regions.

As illustrated in Fig. 5, Robot A first opens the refrigerator and shares contextual information about the target placement locations with Robot B through the shared context tokens, with Fanta assigned to the upper shelf and Coke to the lower shelf. $\operatorname { A t } t _ { 1 } ,$ the refrigerator and the Fanta bottle on the table receive higher attention, corresponding to the Fanta restocking stage. At �<sub>2</sub> and $t _ { 3 } ,$ the highlighted regions shift toward the refrigerator and the Coke bottle on the table as the task progresses to Coke restocking. By $t _ { 4 } ,$ neither the Fanta nor the Coke inside the refrigerator exhibits prominent attention responses, and no salient activation remains on the table, corresponding to the stage of checking for remaining relevant drinks. These observations suggest that the two robots exhibit complementary visual attention patterns from their respective viewpoints, with attention distributions changing consistently with task progression.

Overall, the task demonstrations in Fig. 4 illustrate the applicability of shared context across different dual-robot collaborative scenarios, while the attention visualizations in Fig. 5 provide additional insights into how the robots attend to task-relevant visual information across different stages of collaboration. These attention maps serve as qualitative diagnostic evidence rather than direct causal explanations of how specific visual regions or shared context tokens influence action generation. Together with the quantitative comparisons and ablation studies, these qualitative observations help characterize the role of shared context in supporting multi-robot collaborative decision-making.

![](images/9211a84c71fc2bfa5392debb0c7acac223d97a6a9134cfe8f555247379fd0054.jpg)

![](images/7bf529b3f9ff17b52ef2d1c8c9fee0fb8ac2fd286de273b1cd45a2ce46504b46.jpg)

![](images/7074c0dbac435e2fca9049bd5ded725cbf076fb82c18beff81b34e6daa846b4a.jpg)

## Restock the fridge from the top shelf down

![](images/7ea750c792ec3cc682d486524f178977e530c37302b00c9b5e4682302f02a88d.jpg)  
Robot B spots the upper-level Fanta via A’s view, pick it up to hand to A.

![](images/927672e921d688bb7d80dac2dadea95e14b9f57221cd869126ac8aa9e2dbd339.jpg)  
Robot A spots the Fanta handed over by B, turn to fetch it.

![](images/c6197d3eed1394d75d1e6682c05f08e707df623ee30758cbf76969d0d680aa46.jpg)  
Robot B decides to stock Coke based on the fridge and table state.

![](images/04337c6f967677384bdeb4e084f8909cc1b0b036a543abfabdc2e87a5a98dee4.jpg)  
Robot A

![](images/f1b5ae7f2b732008a2dae380173ed6e8a200ef9ae79e219ace5d3144375906e5.jpg)

![](images/9ce4b32d9828aae9179c3a5cf4d91c858bee5b2cfff763cda97fcc16ee212fd2.jpg)

![](images/4db97b73e0c7d616f67b52e98b0a848420d5689d7c48cfcba1961eb7cef46e5b.jpg)

![](images/be1d6ecdfd4a954411a6b2437865ac83ecc6a7227c0981c37ff1c404f1f1acea.jpg)  
Robot A closes the fridge based on B’s view of the table.

![](images/91ccffa5e1fe2e677f6635ca9c97dffd3416fa39642e94704a1dc0c7489edf8e.jpg)

![](images/a4cf4f73f068b0985c175df30f6490c73a5b7155d30f901d997a4a08e767fbbb.jpg)

![](images/993079450253b5742d48d5f5ea2d6f4e05cf487fdcfcba6a17fb1fcef0de44cd.jpg)

![](images/bee129d0fed6d51c2b04560350f4de64126d710f0550fa10bea6a895b20446a5.jpg)

![](images/10843a7e2a41cdd1cdad6038a32e20b65ccac0a6179dda03ea66024e2cd6989b.jpg)

![](images/88f20efbaaa1d572fd4e69ac3b4ac7acce22a76a297e4f9dd710243074c27b5a.jpg)

![](images/aef369dcd7b54649406e49278ddc345b1c940dfd807c8340273ad799f4f9c0b4.jpg)

## Assemble the box with your partner

![](images/a50f7c385f90a0329dbfa1d7b051ce625ef93684a8bad3fcd2a76e1b94f973ff.jpg)

![](images/84327dc7e32df5fb94bea84eb97610620ec714c1af2f3543cdfe785c2be61f7a.jpg)

![](images/c520f3f3c425e13012f187f6933a8ab7d27ffacace5eb980a32c70bbc55c7199.jpg)  
Robots A and B Robot A stabilizes the opposite work together to push the box open.

![](images/2ae0cdc3cbff7d52ce5821e8113f8337ffdaa3c1624f2eb352f2900a9ec276e5.jpg)  
Robot B uses A’s view to assemble the box top together.

![](images/9c3009de18b3dc777c240bd17f4b9b9e90cbda3eb2aa08e4f88bc6ae2e9dbda2.jpg)  
Robot A

![](images/cfecce495161536f94807f9c34a84df6055175fd5fe3f2acff79043f22688f67.jpg)  
Robot B straightens the box.

![](images/7f9535afe64fc83f2cf08fc2e3a211bd0befbe8328c76e450c8dc1f7924607db.jpg)

![](images/1ad67fc8c8b4226a245e4135d260be15fd85e900b18485c8011772b7113cee70.jpg)

![](images/32392e47c197ef5b7654827223b068d2fe9561ba0cdf75d6e990e5c09e3cb377.jpg)

![](images/c3d0e7a09710eacb67d8b8b757a0a02b021cf6559a16ddc5a192835c0bab5b86.jpg)  
Robot B

![](images/7ec430077b732f3e612952917cb22c212cc5bde9ea0b10762fd0c7c4692c5b23.jpg)

![](images/7baebc97ee82f2aa32143945c4caae9a70d2c8b5c6d4dfdc53f1c43a188d3a0c.jpg)

![](images/7cab961991acbd8f757110ae806790beaf143fa52d3dc229a1557d9b40d33723.jpg)  
Robot B opens its gripper only after receiving confirmation that A is holding the opposite side.

![](images/faefcd1b7d4dfe0199ab0cb3ab307d6ed72fbbb5a1f41c836f46acd9fcf4ae5e.jpg)

![](images/bac83980876cdf25fe5640d553ef1f2609aea502a1e61a6fd29a50a40553993c.jpg)

## Fold and pack the blanket with your partner

![](images/7b16792637633927e0e54e18d40523b32f1087eb66a857d0f0eeb165c060c7a8.jpg)

![](images/b37670d4d438b412c2c117eb4a2de18080f47d7844108e90e6ea65c4777a2e56.jpg)  
Robots A and B observe each other and synchronously fold the blanket into thirds.

![](images/8bbf02a5c8d813b04f722713ada890bae84eb0c02f3e7a87aad7ae5e2578bb62.jpg)

![](images/16c0fce9f597cb093206d14e9027c05b9b2977a327397afe911595323942e367.jpg)  
Robot A moves to the foot of the bed after B folds the blanket.

![](images/620c625295b4290b03cef7c2d30b91e9b6997547950a58a9b31e1b92e2dac782.jpg)

![](images/5d7e50953876e8fd427fb192e4793d724c81907d0eb0192ac39020de8c39b38d.jpg)  
Robot B uses A’s view to pack the blanket into the box.

![](images/5f4c500e1ce32c3185dedbf68e946c73991bf1c94a04fc02e47471ded82858b1.jpg)  
Robot A

![](images/8a47b83f809f366b85bf1a8021a0213b017829053ec9549919b4fc786e7942d6.jpg)

![](images/507865e6d8c2c5abd72a4bde354687cf2ff0eab15bf7273514c89e64cf28407a.jpg)

![](images/f66ab032150d3709a93aebf37c76d2c890788d3513ff16795ed4a3161d032666.jpg)

![](images/46b61320c34ebe40f6e922e654480d1dd397d39862edfa2b89868e1de2773a3c.jpg)  
Robot B

![](images/ba65ec20f3452a37c77c14feb227cb48ad3b0fe67dcaf2e5af43f0e511aaf572.jpg)

![](images/64923b0327553323697c8b900d4ce94ebb45aaf195ce341e8cdbaae0b44ec374.jpg)

![](images/c49f239c6b7faee0338f3517fb0154dd6780990300615fc9a2b982c5d773276e.jpg)

![](images/f1ff95d80727075af9688f533858478768f51328908b53d4d42c4d9b0a1d7936.jpg)  
Figure 4 | Representative rollouts of more collaborative tasks. Keyframes highlight critical moments of inter-robot context sharing during task execution.

![](images/e5c9eb878b027588da1c5d18584f3a96afc14776e336d27c3f10854188678d69.jpg)  
Figure 5 | Paired visual attention maps of Robots A and B during refrigerator restocking at four key moments. Each column shows the two robots’ main-camera observations at the same timestep, with brighter regions indicating higher attention scores.

## 4. Related Work

## 4.1. Vision-Language-Action Models

Recent Vision-Language-Action (VLA) models have advanced generalist manipulation through pretrained semantic representations, diverse training data, and multi-modal learning objectives. Pretrained vision-language backbones enable semantic knowledge transfer to robot control (Black et al., 2024; Driess et al., 2023; Intelligence et al., 2025; Zitkovich et al., 2023), while large-scale, cross-embodiment training extends task and platform coverage (Kim et al., 2024; Li et al., 2024; Liu et al., 2024; O’Neill et al., 2024; Team et al., 2024). Multi-modal co-training and auxiliary visual guidance further strengthen perception–action grounding (Brohan et al., 2022; Intelligence et al., 2025; Li et al., 2025). Recent approaches also introduce intermediate representations and explicit reasoning to bridge semantic understanding and action generation (Huang et al., 2025; Lee et al., 2025; Sun et al., 2024; Tang et al., 2025, 2026; Zawalski et al., 2024; Zhao et al., 2025). Despite training across diverse embodiments and tasks, these policies typically control a single robot using its local visual observations, language instructions, and proprioceptive state. Helix (Figure AI, 2025) extends this paradigm by using shared model weights to control two humanoid robots in a collaborative, long-horizon manipulation task. While this demonstrates the potential of generalist VLAs for multi-robot collaboration, how to organize, compress, and exchange task-relevant perceptual information remains insufficiently explored as a design problem. Our work addresses this cross-robot perceptual interface by encoding each robot’s multi-camera observations into compact, task-conditioned latent representations that directly condition its partner’s action policy. This formulation makes perceptual context sharing an explicit component of collaborative policy learning.

## 4.2. Task Orchestration for Multi-Robot Collaboration

LLM- and VLM-based agents are able to support multi-robot collaboration through task decomposition, capability matching, skill invocation, execution monitoring, and failure recovery. By incorporating execution feedback, these systems adapt their plans and coordination strategies to environmental changes and unexpected outcomes. InstAgent (Luo et al., 2026) extends this paradigm with recursive self-improvement (RSI) for heterogeneous robot teams. Its agentic behavior trees encode capability allocation, inter-robot dependencies, and task handover conditions. Following an execution failure, the system revises the relevant collaboration subtree, validates the revision through re-execution, and retains successful modifications for subsequent reuse. These approaches primarily address task orchestration and execution management through high-level coordination interfaces, using shared memory and iterative refinement to improve task allocation, execution ordering, and failure recovery. Our work investigates the complementary problem of cross-robot perceptual context sharing during policy execution. Agentic orchestration and RSI can thus manage high-level coordination, while our framework supplies shared perceptual context for fine-grained collaborative action generation.

## 4.3. Collaborative Perception

Collaborative perception mitigates partial observability by enabling agents to exchange complementary information. In autonomous driving, V2VNet (Wang et al., 2020) shares compressed intermediate features for perception and motion prediction, while When2com (Liu et al., 2020) and Where2comm (Hu et al., 2022) improve communication efficiency through learned agent selection and spatially selective feature exchange, respectively. These approaches primarily optimize communication for perceptual objectives rather than downstream action generation. In embodied manipulation, Sequential Asymmetric Imitation (Chen et al., 2026c) learns coordinated behaviors without explicit inter-robot communication, whereas MAAP (Chen et al., 2026a) integrates complementary wrist-camera observations for collaborative control. Recent approaches explore explicit cross-robot information exchange. LatentToM (He et al., 2025) learns individual and team-consensus representations for explicit communication and implicit partner-state inference. DuoMind (Zhou et al., 2026) coordinates VLA policies through semantic information from a VLM orchestrator, while Latent Telepathy (Wang et al., 2026) shares frozen perceptual latents to provide task-relevant information unavailable from local observations. Our work learns compact, task-conditioned visual representations that directly condition a partner’s VLA policy. Action supervision from the receiving robot jointly optimizes the context encoder and policy, aligning shared perceptual representations with collaborative action generation under a limited representation budget.

## 4.4. Predictive and Progress-Aware Representation Learning

Predictive representation learning models environmental dynamics through future-state prediction in latent space. V-JEPA2 (Assran et al., 2025) learns dynamics through joint-embedding prediction and supports planning with an action-conditioned latent world model. FLARE (Zheng et al., 2025) aligns policy features with future observation representations, while LaWAM (Chen et al., 2026b) uses compact latent visual subgoals to guide VLA policies. These approaches introduce predictive information into robot control without requiring pixel-level future generation. Task progress provides a complementary signal for structuring embodied representations. XIRL (Zakka et al., 2021) learns progress-sensitive embeddings from cross-embodiment videos and derives rewards from latent distances to goal states. VIP (Ma et al., 2023b) and LIV (Ma et al., 2023a) exploit temporal structure and value-based objectives for representation learning and reward modeling, with LIV additionally incorporating language goals. PROGRESSOR (Ayalew et al., 2024) estimates progress relative to initial and goal observations, while ProgVLA (Kim et al., 2026) and ProgressVLA (Yan et al., 2026) incorporate progress prediction or guidance into long-horizon VLA policies. Whereas these methods primarily use progress for reward modeling or individual-robot control, our work uses it to structure shared perceptual representations for multi-robot collaboration. A partner’s execution phase can inform whether a robot should wait, proceed, or coordinate a handover. We combine task-oriented future prediction with soft task-progress correspondence learning to preserve predictive visual information while promoting inter-task discriminability and intra-task temporal continuity. Progress serves only as a training signal, rather than an explicit prediction or communication target. The resulting compact representations directly condition a partner’s VLA policy, linking predictive and progress-aware representation learning to collaborative action generation.

## 5. Conclusion, Limitations and Future Work

We propose a two-stage active context sharing framework for multi-robot collaboration, in which each robot encodes its multi-camera observations and task instruction into a compact latent representation that is directly integrated into its partner’s VLA policy. The first stage learns shared representations that capture task-relevant information and execution-state structure, while the second stage further aligns these representations with collaborative decision-making through downstream action supervision. Extensive real-world dual-robot experiments demonstrate that the proposed framework provides decision-relevant information unavailable from local observations, improves target selection under visual occlusion, and enables bidirectional sharing of target identity and execution progress in alternating leader–follower tasks. Further ablations and communication-budget analyses validate the effectiveness of the two-stage training strategy and joint optimization, demonstrating that compact task-oriented tokens can achieve performance comparable to high-bandwidth generic visual features at substantially lower communication cost.

Although our experiments demonstrate the effectiveness of the proposed framework, the semantic content of the shared tokens remains underexplored. In particular, how visual semantics, task states, and execution-stage information are encoded in the latent representations, and how these representations influence the receiver’s subsequent decisions and actions, remain open questions for further investigation.

Future work will explore extensions to larger robot teams, asynchronous and selective information sharing, and multi-modal collaborative representations that incorporate temporal memory, tactile sensing, and proprioception. More broadly, our results point toward a paradigm in which multi-robot collaboration moves beyond exchanging high-level commands or generic visual features, enabling robots to actively distill and communicate compact, task-relevant context shaped jointly by the sender’s task objectives and the receiver’s decision requirements.

## 6. Contributions

• Contributors: Baifu Huang, Shilin Fang

• Project Lead: Jianan Wang

## References

M. Assran, A. Bardes, D. Fan, Q. Garrido, R. Howes, M. Komeili, M. Muckley, A. Rizvi, C. Roberts, K. Sinha, A. Zholus, S. Arnaud, A. Gejji, A. Martin, F. R. Hogan, D. Dugas,

P. Bojanowski, V. Khalidov, P. Labatut, F. Massa, M. Szafraniec, K. Krishnakumar, Y. Li, X. Ma, S. Chandar, F. Meier, Y. LeCun, M. Rabbat, and N. Ballas. V-JEPA 2: Selfsupervised video models enable understanding, prediction and planning, 2025. URL https://arxiv.org/abs/2506.09985.

T. Ayalew, X. Zhang, K. Y. Wu, T. Jiang, M. Maire, and M. R. Walter. PROGRESSOR: A perceptually guided reward estimator with self-supervised online refinement, 2024. URL https://arxiv.org/abs/2411.17764.

K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al. �0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022.

B. N. Y. Chen, L. Kang, H. Zhou, X. Song, Z. Zhang, J. Ma, and Y. Qin. MAAP: Multi-agent active perception for collaborative manipulation, 2026a. URL https://arxiv.org/abs/26 09.21929.

J. Chen, K. Wang, K. Chen, S. Chen, F. Gao, W. Tang, Z. Li, W. Liu, Z. Yao, B. Li, Y. Xu, and C. Yu. LaWAM: Latent world action models for efficient dynamics-aware robot policies, 2026b. URL https://arxiv.org/abs/2606.15768.

Y. Chen, R. Qiu, Z. Li, Y. Zhou, G. Ren, and W. Zhi. Robots that collaborate: Sequential asymmetric imitation for learning coupled robot policies, 2026c. URL https://arxiv.org/ abs/2606.16490.

D. Driess, F. Xia, M. S. Sajjadi, C. Lynch, A. Chowdhery, A. Wahid, J. Tompson, Q. Vuong, T. Yu, W. Huang, et al. Palm-e: An embodied multimodal language model. 2023.

Figure AI. Helix: A vision-language-action model for generalist humanoid control. Official technical blog, February 2025. URL https://www.figure.ai/news/helix. Accessed 2026-10-26.

G. Gao, J. Wang, J. Zuo, J. Jiang, J. Zhang, X. Zeng, Y. Zhu, L. Ma, K. Chen, M. Sheng, et al. Towards human-level intelligence via human-like whole-body manipulation. arXiv preprint arXiv:2507.17141, 2025.

C. He, G. M. S. Camps, X. Liu, M. Schwager, and G. A. Sartoretti. Latent theory of mind: A decentralized diffusion architecture for cooperative manipulation. In Proceedings of the 9th Conference on Robot Learning, volume 305 of Proceedings of Machine Learning Research, pages 392–405. PMLR, 2025. URL https://proceedings.mlr.press/v305/he25a.htm l.

Y. Hu, S. Fang, Z. Lei, Y. Zhong, and S. Chen. Where2comm: Communication-efficient collaborative perception via spatial confidence maps. In Advances in Neural Information Processing Systems, volume 35, 2022. URL https://proceedings.neurips.cc/paper\_files/p aper/2022/hash/1f5c5cd01b864d53cc5fa0a3472e152e-Abstract-Conference.ht ml.

C.-P. Huang, Y.-H. Wu, M.-H. Chen, Y.-C. F. Wang, and F.-E. Yang. Thinkact: Vision-languageaction reasoning via reinforced visual latent planning. arXiv preprint arXiv:2507.16815, 2025.

P. Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, et al. �<sub>0.5</sub>: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

S. Kim, J. Choi, S. Baek, and J.-M. Renders. ProgVLA: Progress-aware robot manipulation skill learning, 2026. URL https://arxiv.org/abs/2605.28231.

J. Lee, J. Duan, H. Fang, Y. Deng, S. Liu, B. Li, B. Fang, J. Zhang, Y. R. Wang, S. Lee, et al. Molmoact: Action reasoning models that can reason in space. arXiv preprint arXiv:2508.07917, 2025.

P. Li, Y. Wu, Z. Xi, W. Li, Y. Huang, Z. Zhang, Y. Chen, J. Wang, S.-C. Zhu, T. Liu, et al. Controlvla: Few-shot object-centric adaptation for pre-trained vision-language-action models. arXiv preprint arXiv:2506.16211, 2025.

Q. Li, Y. Liang, Z. Wang, L. Luo, X. Chen, M. Liao, F. Wei, Y. Deng, S. Xu, Y. Zhang, et al. Cogact: A foundational vision-language-action model for synergizing cognition and action in robotic manipulation. arXiv preprint arXiv:2411.19650, 2024.

S. Liu, L. Wu, B. Li, H. Tan, H. Chen, Z. Wang, K. Xu, H. Su, and J. Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation. arXiv preprint arXiv:2410.07864, 2024.

Y.-C. Liu, J. Tian, N. Glaser, and Z. Kira. When2com: Multi-agent perception via communication graph grouping. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4106–4115, 2020. URL https://openaccess.thecvf.com/co ntent\_CVPR\_2020/html/Liu\_When2com\_Multi-Agent\_Perception\_via\_Communic ation\_Graph\_Grouping\_CVPR\_2020\_paper.html.

H. Luo, J. Qin, J. Cao, Y. Huang, G. Yang, X. Chen, H. Lu, T. Gui, Y. Shi, and J. Wang. InstAgent: A recursively self-improving agent operating system for heterogeneous robot collaboration, 2026. URL https://arxiv.org/abs/2610.23098.

Y. J. Ma, V. Kumar, A. Zhang, O. Bastani, and D. Jayaraman. LIV: Language-image representations and rewards for robotic control. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 23301– 23320. PMLR, 2023a. URL https://proceedings.mlr.press/v202/ma23b.html.

Y. J. Ma, S. Sodhani, D. Jayaraman, O. Bastani, V. Kumar, and A. Zhang. VIP: Towards universal visual reward and representation via value-implicit pre-training. In International Conference on Learning Representations, 2023b. URL https://arxiv.org/abs/2210.00030.

A. O’Neill, A. Rehman, A. Maddukuri, A. Gupta, A. Padalkar, A. Lee, A. Pooley, A. Gupta, A. Mandlekar, A. Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6892–6903. IEEE, 2024.

Q. Sun, P. Hong, T. D. Pala, V. Toh, U. Tan, D. Ghosal, S. Poria, et al. Emma-x: An embodied multimodal action model with grounded chain of thought and look-ahead spatial reasoning. arXiv preprint arXiv:2412.11974, 2024.

P. Tang, S. Xie, B. Sun, B. Huang, K. Luo, H. Yang, W. Jin, and J. Wang. Mind to hand: Purposeful robotic control via embodied reasoning. arXiv preprint arXiv:2512.08580, 2025.

P. Tang, S. Xie, B. Huang, B. Sun, H. Yang, K. Luo, W. Jin, S. Fang, and J. Wang. Towards predictive, aligned, and scalable robot learning. arXiv preprint arXiv:2607.11270, 2026.

O. M. Team, D. Ghosh, H. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, T. Kreiman, C. Xu, et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024.

H. Wang, H. Zheng, and C. Wu. Latent telepathy: Multi-robot communication with selfsupervised perceptual latents, 2026. URL https://arxiv.org/abs/2609.23269.

T.-H. Wang, S. Manivasagam, M. Liang, B. Yang, W. Zeng, and R. Urtasun. V2VNet: Vehicle-tovehicle communication for joint perception and prediction. In Proceedings of the European Conference on Computer Vision, pages 605–621, 2020. doi: 10.1007/978-3-030-58536-5\_36. URL https://doi.org/10.1007/978-3-030-58536-5\_36.

H. Yan, Q. Li, J. Yang, and Y. Mu. ProgressVLA: Progress-guided diffusion policy for visionlanguage robotic manipulation, 2026. URL https://arxiv.org/abs/2603.27670.

K. Zakka, A. Zeng, P. Florence, J. Tompson, J. Bohg, and D. Dwibedi. XIRL: Cross-embodiment inverse reinforcement learning, 2021. URL https://arxiv.org/abs/2106.03911.

M. Zawalski, W. Chen, K. Pertsch, O. Mees, C. Finn, and S. Levine. Robotic control via embodied chain-of-thought reasoning. arXiv preprint arXiv:2407.08693, 2024.

Q. Zhao, Y. Lu, M. J. Kim, Z. Fu, Z. Zhang, Y. Wu, Z. Li, Q. Ma, S. Han, C. Finn, et al. Cot-vla: Visual chain-of-thought reasoning for vision-language-action models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1702–1713, 2025.

R. Zheng, J. Wang, S. Reed, J. Bjorck, Y. Fang, F. Hu, J. Jang, K. Kundalia, Z. Lin, L. Magne, A. Narayan, Y. L. Tan, G. Wang, Q. Wang, J. Xiang, Y. Xu, S. Ye, J. Kautz, F. Huang, Y. Zhu, and L. Fan. FLARE: Robot learning with implicit world modeling. In Proceedings of the 9th Conference on Robot Learning, volume 305 of Proceedings of Machine Learning Research, pages 3952–3971. PMLR, 2025. URL https://proceedings.mlr.press/v305/zheng25 a.html.

H. Zhou, D. Gao, H. Wang, B. Lynch, B. Zhao, Q. Ma, R. Goyal, and J. Zhang. DuoMind: Enabling distributed multi-robot coordination with semantic communication, 2026. URL https://arxiv.org/abs/2610.02161.

B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, F. Xia, J. Wu, P. Wohlhart, S. Welker, A. Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pages 2165–2183. PMLR, 2023.