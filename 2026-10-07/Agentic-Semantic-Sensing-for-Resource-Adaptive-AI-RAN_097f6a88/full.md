# Agentic Semantic Sensing for Resource-Adaptive AI-RAN

Zhongqin Wang, Member, IEEE, Xiaoqi Zhang, Student Member, IEEE, Nan Yang, Member, IEEE, Kai Wu, Member, IEEE, J. Andrew Zhang, Senior Member, IEEE, Y. Jay Guo, Life Fellow, IEEE

Abstract—Semantic sensing (SemS) acquires task-relevant information rather than reconstructing complete physical information. Existing SemS formulations typically operate open loop: sensing configurations and observation schedules are fixed before inference and cannot respond to evolving task-level evidence. We propose Agentic SemS, a closed-loop framework for AIenabled radio access networks (AI-RANs) that controls sensing within a communication-feasible profile set. A profile-conditioned causal Transformer updates the semantic belief from streaming observations, while key–value caching enables efficient state updates across profile changes without repeatedly processing the complete history. A semantic utility network estimates the tasklevel benefit of acquiring the next observation block under each feasible profile after accounting for sensing cost. The resulting continuation utilities jointly support next-profile selection and semantic early exit, adapting sensing configuration and duration to evolving evidence. The expected semantic gain is further related to conditional mutual information, providing a valueof-information interpretation of continued online sensing. Experiments on Widar3.0 with six emulated sensing profiles show that, in comparison with full-sequence High, the resource-efficient Agentic setting reduces normalized cumulative sensing cost by 25.33% while achieving 85.79% Macro-F1. At the same utility checkpoint, semantic early exit provides a further 12.35% cost reduction over adaptive sensing without early exit, with a 0.97- percentage-point Macro-F1 decrease.

Index Terms—Semantic sensing, agentic AI, AI-enabled radio access network (AI-RAN), adaptive sensing, semantic early exit.

## I. INTRODUCTION

N <sup>EXT-GENERATION</sup> <sup>wireless</sup> <sup>networks</sup> <sup>are</sup> <sup>expected</sup> <sup>to</sup> provide connectivity, perceive their environment, and adapt operation to evolving task requirements. Integrating artificial intelligence (AI) into the radio access network (RAN), known as AI-RAN [1], [2], enables learning-driven sensing, inference, and resource control, while data-driven architectures support adaptive resource management and service provisioning [3]. Emerging agentic AI closes the perception–decision– action loop by updating task-relevant knowledge from observations, evaluating feasible actions, and using selected actions to shape subsequent environmental interactions. Integrated sensing and communication (ISAC) extends this capability by allowing sensing and communication to share spectrum, signals, energy, and hardware resources [4]–[8]. Within such systems, information requirements for reliable inference can vary with task progress, environmental dynamics, and wireless conditions. This gives rise to two coupled resource-control decisions: how the sensing configuration should be adapted when additional information is required, and whether further sensing should be performed at all. These decisions determine the resource cost of continued acquisition and the duration over which sensing resources are consumed.

Conventional wireless sensing includes physical-parameter estimation and learning-based task inference. Classical signal processing approaches estimate physical parameters such as propagation delays, Doppler shifts, angles of arrival (AoAs), and path gains [9], [10]. Learning-based methods map received signal strength indicator (RSSI) [11], channel state information (CSI) [12], and Doppler features [13] directly to task labels or task-specific estimates. Recent work has explored technologyagnostic recognition across heterogeneous radio-frequency (RF) modalities [14] and resource-adaptive inference under varying channel and computation budgets [15]. These methods can achieve strong inference, but their sensing configurations and observation schedules are typically fixed before inference. Such open-loop operation cannot respond to changes in the informativeness of newly acquired observations, task uncertainty, environmental dynamics, or wireless conditions. Consequently, a fixed sensing configuration may expend excessive resources when lower-cost observations are already sufficient, while a fixed observation horizon may continue acquisition after reliable inference has become possible or terminate before enough task-relevant evidence has been collected. Addressing this mismatch requires sensing decisions to be guided by taskrelevant information rather than a predetermined strategy.

Semantic sensing (SemS) [16] provides such a taskoriented perspective by focusing sensing on information relevant to downstream inference rather than complete physicalinformation reconstruction. This direction is closely related to semantic communication, which moves beyond exact bit reconstruction and emphasizes the delivery of task-relevant meaning [17], [18]. Token-based semantic communication further exploits cross-modal context to improve communication efficiency [19], while related task-oriented work in integrated sensing, communication, and computation allocates radio and computing resources according to edge-learning or inference performance rather than reconstruction fidelity [20]. Existing SemS formulations establish the principle of taskrelevant information acquisition, but sensing configurations and observation procedures are generally determined before inference. As a result, the task-level information revealed during sensing does not directly influence how subsequent observations are acquired. Closing this feedback loop would allow the evolving inference state to adapt the sensing profile and observation duration according to the information required for reliable inference, thereby reducing unnecessary sensingside radio and processing resource expenditure.

Recent studies on feedback-driven ISAC and agentic AI examine closed-loop control of predictive beamforming, target-aware transmission, spectrum allocation, power control, and communication–sensing resource management [21]– [26]. Adaptive-interval CSI estimation reduces measurement overhead by deciding whether communication and radar CSI should be re-estimated or predicted from temporal correlation [27], while feedback-aware cognitive-radio designs jointly control spectrum sensing and access under resource constraints [28]. These approaches use channel states, target dynamics, traffic conditions, or network-level objectives to drive adaptation. In contrast, they do not use the evolving tasklevel inference state as feedback for subsequent sensing. This leaves open the question of how semantic inference itself can guide which sensing profile should be used next and, when appropriate, whether further sensing is necessary.

In this work, we propose Agentic SemS, a closed-loop framework that uses evolving semantic information to control subsequent sensing in AI-RANs. A sensing profile specifies the observation quality, signal representation, temporal configuration, and sensing-resource usage of an observation block, while the RAN defines the set of profiles that satisfy the underlying communication requirements. A profile-conditioned causal Transformer [29] processes streaming observations and incrementally updates the semantic belief, which represents the current task-level inference state. Key–value (KV) caching preserves the accumulated observation context across profile transitions, enabling efficient incremental inference without repeatedly processing the complete observation history [30]. At each decision epoch, a semantic utility network evaluates the feasible sensing profiles according to the current semantic state and their predicted contribution to subsequent inference. The agent then selects the profile for the next observation block, allowing the sensing configuration and associated resource usage to evolve with the task-level evidence. Semantic early exit provides an additional duration-control mechanism, allowing the agent to terminate sensing when further acquisition is predicted to provide limited benefit. Repeating this belief–utility–action cycle establishes a closed feedback loop between semantic inference and sensing-resource adaptation.

Our main contributions are summarized as follows.

• We propose Agentic SemS, a closed-loop semantic sensing framework that uses the evolving semantic belief as feedback for subsequent acquisition. Within a communication-feasible profile set, a continuation-utility formulation evaluates the predicted task-level benefit and sensing-resource cost of further observation and supports both next-profile selection and semantic early exit. The expected semantic gain is linked to conditional mutual information, providing a value-of-information (VoI) interpretation of continued sensing.

• We provide a reference implementation of Agentic SemS that realizes the framework as a streaming sensing system. It covers profile-dependent sensing front ends, the architectures and training procedures of the semantic belief and utility networks, and KV-cached online operation for profile adaptation and semantic early exit, while allowing its front ends and profile context to adapt across sensing modalities and configurations in diverse deployments.

• We evaluate Agentic SemS on Widar3.0 [31] with six emulated sensing profiles and controlled comparisons of profile adaptation and semantic early exit.<sup>1</sup> In comparison with full-sequence High, the resource-efficient Agentic setting reduces normalized cumulative sensing cost by 25.33% while achieving 85.79% Macro-F1, demonstrating that the two control mechanisms can reduce sensingresource usage while maintaining task performance.

The paper is organized as follows. Section II reviews SemS and motivates its closed-loop extension. Section III presents the Agentic SemS framework. Section IV describes the system design and implementation, while Section V presents the experimental setup and results. Section VI concludes the paper.

## II. PRELIMINARIES AND MOTIVATION

This section briefly reviews the open-loop SemS formulation and motivates its extension to closed-loop Agentic SemS.

## A. Semantic Sensing Model

Semantic sensing (SemS) [16] provides a task-oriented sensing paradigm that focuses information acquisition on downstream inference rather than complete environmental reconstruction. Let $\mathcal { A } = \{ a _ { 1 } , \ldots , a _ { N _ { \mathrm { p } } } \}$ denote the sensing-profile space, where each profile specifies a sensing configuration and its associated resource usage. In an AI-RAN system, communication-service requirements constrain the sensing configurations that can be used. We represent the resulting feasible profile set as

$$
\begin{array} { r } { \mathcal { A } ^ { \mathrm { c o m } } = \left\{ a _ { p } \in \mathcal { A } : Q _ { \mathrm { c o m } } ( a _ { p } ) \geq Q _ { \mathrm { c o m } } ^ { \mathrm { m i n } } \right\} , } \end{array}\tag{1}
$$

where $Q _ { \mathrm { c o m } } ( a _ { p } )$ is the communication quality of profile ${ a _ { p } } ,$ and $Q _ { \mathrm { c o m } } ^ { \mathrm { m i n } }$ is the minimum communication-quality requirement. Accordingly, $A ^ { \mathrm { c o m } }$ is the communication-feasible profile set considered throughout the framework and is supplied by the underlying RAN to the sensing controller.

A generic profile-dependent SemS acquisition model is

$$
\begin{array} { r } { \mathbf { x } _ { s } ^ { ( p ) } = \mathcal { F } _ { \Theta _ { \mathrm { T x } } } \left( \mathbf { x } _ { \mathrm { r e f } } ; a _ { p } \right) , \qquad a _ { p } \in \mathcal { A } ^ { \mathrm { c o m } } , } \end{array}\tag{2}
$$

where $\mathbf { x } _ { \mathrm { r e f } }$ is the reference signal, $\Theta _ { \mathrm { T x } }$ contains encoder parameters, and $a _ { p }$ identifies the active profile. A profile may specify subcarrier usage, sampling rate, signal quality, perblock temporal configuration, and signal representation.

The sensing signal interacts with the physical environment and propagation channel to produce

$$
\mathbf { y } _ { s } ^ { ( p ) } = \Psi _ { \mathrm { c h } } \left( S , \mathbf { x } _ { s } ^ { ( p ) } \right) + \mathbf { n } _ { s } ,\tag{3}
$$

where $S$ denotes the latent physical state, $\Psi _ { \mathrm { c h } } ( \cdot )$ represents the sensing-channel mapping, and $\mathbf { n } _ { s }$ denotes aggregate noise. The semantic decoder directly estimates the task output as

$$
\hat { T } = \mathcal { F } _ { \Theta _ { \mathrm { R x } } } \left( \mathbf { y } _ { s } ^ { \left( p \right) } \right) ,\tag{4}
$$

without reconstructing task-irrelevant physical information.

![](images/e0641f2d3f2e7729bde60bccaa0842bcaabbbb63c620ee1862f608b57c46288d.jpg)  
Fig. 1. Closed-loop architecture of Agentic SemS. The belief network updates the semantic state from streaming observations, while the utility network scores feasible profiles. The predicted utilities jointly support semantic early exit and, when sensing continues, selection of the next feasible profile.

## B. From Open-Loop SemS to Closed-Loop Agentic SemS

The model in (2)–(4) captures profile-dependent taskoriented sensing but fixes the sensing configuration and observation procedure, preventing feedback from the evolving task-level inference state. Consequently, information revealed during sensing does not directly control subsequent acquisition or whether sensing should continue. As sensing proceeds, the value of observations changes with accumulated evidence, environmental dynamics, and wireless conditions. A predetermined profile may waste resources when lowercost observations suffice, while a fixed schedule cannot adapt duration to evolving inference needs. Agentic SemS closes this loop by using the semantic belief to guide sensing. At each epoch, the agent selects a feasible profile as evidence accumulates. Semantic early exit terminates sensing when further acquisition has low predicted value, jointly adapting sensing configuration and duration to task requirements.

## III. AGENTIC SEMANTIC SENSING FRAMEWORK

This section presents how Agentic SemS represents the evolving task-level information, evaluates feasible sensing profiles, and controls subsequent sensing.

## A. System Overview

1) Closed-Loop Semantic Sensing Task: We consider a semantic sensing task with target $T .$ For classification, $T =$ $y ~ \in ~ \{ 1 , \ldots , C \}$ , whereas for regression, $T \mathbf { \Psi } = \textbf { y } \in \mathbb { R } ^ { D _ { \mathbf { y } } }$ where C and $D _ { \mathrm { y } }$ denote the number of classes and the output dimension, respectively. As shown in Fig. 1, Agentic SemS processes observations under the active sensing profile and updates the task-level inference state. At each decision epoch, the agent selects a feasible sensing profile for the next observation block. Semantic early exit adapts sensing duration according to the predicted value of further acquisition. The resulting belief–utility–action sequence closes the loop, so each sensing decision affects subsequent observations.

2) Sensing Profiles and Feasible Actions: Using the sensing-profile space A and communication-feasible subset $A ^ { \mathrm { c o m } }$ defined in Section II, each profile specifies an operating point through its observation, configuration, and resource attributes. Let $s _ { t }$ denote the profile active at observation step t, and let $\chi _ { t } ( a _ { p } , a _ { s _ { t } } ) ~ \in ~ \{ 0 , 1 \}$ indicate whether profile $a _ { p }$ is available from the current operating state. The feasible continuation profiles at decision epoch t are

$$
\mathcal { A } _ { t } ^ { \mathrm { f e a s } } = \left\{ a _ { p } \in \mathcal { A } ^ { \mathrm { c o m } } : \chi _ { t } ( a _ { p } , a _ { s _ { t } } ) = 1 \right\} ,\tag{5}
$$

where $\chi _ { t } ( \cdot )$ captures sensing-hardware availability and reconfiguration constraints at decision epoch t.

3) Semantic Belief and State: At observation step t, active profile $a _ { s _ { t } }$ is described by context $\mathbf { q } _ { s _ { t } }$ and used to acquire observation $\mathbf { o } _ { t } .$ . The belief network processes accumulated observations with corresponding profile contexts and produces a latent representation $\mathbf { r } _ { t }$ and a task-dependent semantic belief $\mathbf { b } _ { t } .$ . For classification, $\mathbf { b } _ { t }$ represents the posterior distribution over the semantic classes; for regression, it may represent the predictive estimate together with its uncertainty. Together, $\mathbf { r } _ { t }$ and $\mathbf { b } _ { t }$ constitute the semantic state $\mathbf { u } _ { t } ,$ which combines accumulated task-relevant information with the current inference to guide subsequent profile selection and stopping.

4) Joint Profile Selection and Semantic Early Exit: For each $a _ { p } \ \in \ A _ { t } ^ { \mathrm { f e a s } }$ , the encoder maps its context $\mathbf { q } _ { p }$ to an embedding $\mathbf { g } _ { p } .$ The semantic utility network combines $\mathbf { g } _ { p }$ with the semantic state $\mathbf { u } _ { t }$ to estimate the utility of acquiring the next observation block under profile $a _ { p } .$ . The predicted utilities support two coupled decisions: selecting a sensing profile for the next observation block and determining whether further sensing is worthwhile. If sensing continues, the selected profile is applied to the next observation block; otherwise, sensing terminates and the current task prediction is returned. Each new block updates the semantic state before subsequent profileselection and stopping decisions during online operation.

## B. Semantic Belief Network

The belief network is the perception and state-update module of Agentic SemS and uses each coherent processing interval (CPI) as an observation unit. A CPI aggregates a short sequence of consecutive measurements, providing sufficient temporal support for extracting motion-sensitive spectral or statistical features, such as Doppler features, while preserving the temporal resolution required for online inference. The resulting CPI-level observation provides a common sequential input unit across sensing profiles. At each CPI, the belief network processes the observation together with its active-profile context and incrementally updates the latent representation $\mathbf { r } _ { t }$ and semantic belief $\mathbf { b } _ { t }$ . The resulting state supplies task-level information for subsequent sensing decisions.

1) CPI-Level Input Representation: At CPI t, the profiledependent preprocessing module receives measurements m<sub>t</sub> acquired under sensing profile $a _ { s _ { t } }$ . Depending on the sensing interface and active profile, m<sub>t</sub> may comprise raw inphase/quadrature (I/Q) samples, channel-response measurements such as CSI, or compact radio indicators such as RSSI, reference signal received power (RSRP), reference signal received quality (RSRQ), and signal-to-interference-plus-noise ratio (SINR). Their modality, structure, and dimension may vary across profiles according to the sensing hardware and the information exposed by the active configuration. The sensing profile therefore determines which information is available and the resources required to acquire it.

2) Signal Preprocessing and Observation Encoding: Profile-dependent preprocessing of $\mathbf { m } _ { t }$ under $a _ { s _ { t } }$ suppresses measurement artifacts and produces the CPI observation

$$
\mathbf { o } _ { t } = \mathcal { G } _ { \mathrm { o } } \left( \mathbf { m } _ { t } ; a _ { s _ { t } } \right) \in \mathbb { R } ^ { N _ { \mathrm { o b s } } } ,\tag{6}
$$

where $\mathcal { G } _ { \mathrm { o } } ( \cdot )$ denotes the profile-dependent preprocessing and feature-extraction function. For example, CSI measurements may be affected by timing offsets, residual carrierfrequency offsets, receiver-chain phase distortions, automaticgain-control variations, static multipath components, and interference. Although different profiles may provide different measurements and acquisition settings, their outputs are mapped to the common dimension $N _ { \mathrm { o b s } }$ <sub>s</sub>.

The resulting observation is mapped to a common embedding through a learnable observation encoder:

$$
\mathbf { z } _ { t } = \mathcal { E } _ { \theta _ { \mathrm { o } } } ( \mathbf { o } _ { t } ) \in \mathbb { R } ^ { N _ { \mathrm { B } } } ,\tag{7}
$$

where $\mathcal { E } _ { \theta _ { \mathrm { o } } } ( \cdot )$ is an $N _ { \mathrm { B } }$ -dimensional observation encoder.

3) Profile-Context Encoding: Each sensing profile $a _ { p }$ is associated with a context vector $\mathbf { q } _ { p }$ that provides a compact numerical description of its sensing configuration and resource requirements, rather than the sensed data itself. Depending on the system, this context may include the sensing modality, sampling rate, observation quality, and temporal configuration; the sensing-side use of subcarriers, bandwidth, antennas, or beams; and the associated normalized sensing-resource cost. This context enables the belief network to interpret observations according to how they were acquired and allows the utility network to evaluate candidate profiles under their respective sensing conditions and resource requirements. The profile context is encoded as

$$
\mathbf { g } _ { p } = \mathcal { E } _ { \theta _ { \mathrm { A } } } \left( \mathbf { q } _ { p } \right) \in \mathbb { R } ^ { N _ { \mathrm { B } } } ,\tag{8}
$$

where $\mathcal { E } _ { \theta _ { \mathrm { A } } } ( \cdot )$ denotes the profile encoder. The same encoder represents the active profile for semantic belief updates and the candidate profiles for subsequent utility evaluation.

4) Semantic Belief Update and Prediction: At CPI t, the observation embedding, active-profile embedding, and positional embedding are fused through element-wise addition:

$$
\mathbf { x } _ { t } = \mathbf { z } _ { t } + \mathbf { g } _ { s _ { t } } + \mathbf { p } _ { t } \in \mathbb { R } ^ { N _ { \mathrm { B } } } ,\tag{9}
$$

where $\mathbf { p } _ { t }$ denotes the positional embedding. This construction conditions each observation on its acquisition profile.

For online sensing decisions, the belief model uses causal attention, such that the representation at CPI t is computed only from the current and preceding profile-conditioned inputs, without access to future observations:

$$
{ \bf r } _ { t } = \mathcal { F } _ { \boldsymbol { \theta } _ { \mathrm { B } } } \left( { \bf x } _ { 1 : t } \right) ,\tag{10}
$$

where $\mathbf { r } _ { t }$ denotes the latent semantic representation at CPI t. Because each input $\mathbf { x } _ { t }$ includes the corresponding activeprofile context, $\mathbf { r } _ { t }$ encodes both the accumulated observations and the sensing conditions under which they were acquired.

For classification, the semantic belief is obtained as

$$
\mathbf b _ { t } = \mathrm { s o f t m a x } \left( \mathbf W _ { \mathrm { c l s } } \mathbf r _ { t } + \beta _ { \mathrm { c l s } } \right) \in \mathbb R ^ { C } ,\tag{11}
$$

where $\mathbf { W } _ { \mathrm { c l s } }$ and $\beta _ { \mathrm { c l s } }$ are learnable parameters, and $\begin{array} { r l } { \mathbf { b } _ { t } } & { { } = } \end{array}$ $[ b _ { t , 1 } , \ldots , b _ { t , C } ] ^ { \mathsf { T } }$ contains the class-posterior probabilities. For regression, the same latent representation can be coupled with a task-specific prediction head that outputs the estimate and associated uncertainty. For clarity, the following formulation uses classification to illustrate the framework, although Agentic SemS is not limited to classification tasks. The resulting $\mathbf { r } _ { t }$ and $\mathbf { b } _ { t }$ form the semantic state $\mathbf { u } _ { t }$ defined in Section III-A.

5) Weighted-Prefix Learning Objective: For training sequence $n ,$ , let $L _ { n }$ denote the number of CPIs and $y _ { n } \in$ $\{ 1 , \ldots , C \}$ its ground-truth class. The belief $\mathbf { b } _ { n , t }$ represents the prediction after observing the first t CPIs. To align intermediate supervision with online decision epochs, supervision is applied at intervals of K CPIs and begins only after at least a fraction $\rho _ { \mathrm { m i n } }$ of the sequence has been observed. This excludes very short prefixes that may contain insufficient task-relevant information. The resulting supervised prefix set is

$$
\mathcal { P } _ { n } = \left\{ t \in \{ K , 2 K , 3 K , . . . \} : t < L _ { n } , \quad \frac { t } { L _ { n } } \geq \rho _ { \operatorname* { m i n } } \right\} .\tag{12}
$$

Here, $K$ is the decision interval and $\rho _ { \mathrm { m i n } }$ the minimum supervised observation fraction. The complete sequence receives separate supervision. The prefix set is constructed during training, when the complete sequence length $L _ { n }$ is known. During online operation, observations are processed as they arrive, and the model does not require $L _ { n }$

Each intermediate prefix is assigned a weight that increases with the observed fraction of the sequence:

$$
w _ { n , t } = \left( \frac { t } { L _ { n } } \right) ^ { \kappa } , \qquad t \in \mathcal { P } _ { n } ,\tag{13}
$$

where κ controls the relative emphasis on later intermediate prefixes. Later prefixes generally provide more reliable semantic evidence; assigning them larger weights stabilizes training while retaining supervision at earlier decision epochs.

For classification, the loss is defined as

$$
\ell _ { \mathrm { c l s } } \left( \mathbf { b } _ { n , t } , y _ { n } \right) = - \log \left[ \mathbf { b } _ { n , t } \right] _ { y _ { n } } ,\tag{14}
$$

the normalized intermediate-prefix loss is

$$
\mathcal { L } _ { \mathrm { p r e f i x } } ^ { ( n ) } = \frac { 1 } { \displaystyle \sum _ { t \in \mathcal { P } _ { n } } w _ { n , t } } \sum _ { t \in \mathcal { P } _ { n } } w _ { n , t } \ell _ { \mathrm { c l s } } \left( \mathbf { b } _ { n , t } , y _ { n } \right) ,\tag{15}
$$

and the training objective for sequence n is

$$
\begin{array} { r } { \mathcal L ^ { ( n ) } = \omega _ { \mathrm { f } } \ell _ { \mathrm { c l s } } \left( \mathbf b _ { n , L _ { n } } , y _ { n } \right) + \left( 1 - \omega _ { \mathrm { f } } \right) \mathcal L _ { \mathrm { p r e f i x } } ^ { ( n ) } , } \end{array}\tag{16}
$$

where $\omega _ { \mathrm { f } }$ balances full-sequence and prefix supervision.

## C. Semantic Utility Network

The semantic utility network evaluates how useful each feasible sensing profile is for the next observation block. Given the current semantic state, it predicts the utility of continuing sensing under each candidate profile. At each decision epoch, these utilities determine both continued acquisition and the profile used to perform it.

1) Candidate-Conditioned Utility Input: At decision epoch t, the latent representation $\mathbf { r } _ { t }$ summarizes accumulated taskrelevant information, while the semantic belief $\mathbf { b } _ { t }$ represents the current task-level inference. For each candidate profile $a _ { p } \in \mathcal { A } _ { t } ^ { \mathrm { f e a s } }$ , the shared profile encoder produces its embedding $\mathbf { g } _ { p } ~ = ~ \mathcal { E } _ { \theta _ { \mathrm { A } } } ( \mathbf { q } _ { p } )$ . The candidate-conditioned utility input is $\mathbf { v } _ { t } ^ { ( p ) } = \mathrm { c o n c a t } ( \mathbf { r } _ { t } , \mathbf { b } _ { t } , \mathbf { g } _ { p } )$ , where concat(·) joins the three feature vectors into a single input vector. This representation lets the utility network consider the semantic evidence and sensing characteristics of each candidate profile.

2) Semantic Utility Prediction: A utility network shared across all candidate profiles predicts

$$
U _ { t } ^ { ( p ) } = \mathcal { F } _ { \theta _ { \mathrm { U } } } \left( \mathbf { v } _ { t } ^ { ( p ) } \right) , \qquad a _ { p } \in \mathcal { A } _ { t } ^ { \mathrm { f e a s } } ,\tag{17}
$$

where $U _ { t } ^ { ( p ) } \in \mathbb { R }$ represents the predicted utility of acquiring the next observation block under profile $a _ { p } . \mathrm { ~ A ~ }$ larger value indicates greater expected semantic improvement after accounting for sensing cost. When sensing continues, the agent applies the highest-utility candidate to the next block.

3) Semantic Utility Objective: After training, the semantic belief network is fixed to construct supervised utility targets. The candidate-profile semantic gain is defined as the task-loss reduction obtained from one additional observation block:

$$
G _ { n , t } ^ { ( p ) } = \ell _ { \mathrm { s e m } } \left( y _ { n } , \mathbf { b } _ { n , t } \right) - \ell _ { \mathrm { s e m } } \left( y _ { n } , \mathbf { b } _ { n , t ^ { \prime } } ^ { ( p ) } \right) .\tag{18}
$$

Here, $\ell _ { \mathrm { s e m } } ( \cdot )$ denotes the task loss. At decision epoch $t < L _ { n }$ of training sequence $n , \ \mathbf { b } _ { n , t }$ is the current belief, and $\mathbf { b } _ { n , t ^ { \prime } } ^ { ( p ) }$ is produced by the fixed belief network after processing the next observation block under $a _ { p } \in \mathcal { A } _ { t } ^ { \mathrm { f e a s } }$ , where $t ^ { \prime } > t$ is the next decision epoch. The additional block and groundtruth target are used only for utility-target construction, not during online sensing. For classification, we use cross-entropy, $\ell _ { \mathrm { s e m } } ( y , \mathbf { b } ) = - \log b _ { y }$ , yielding

$$
G _ { n , t } ^ { ( p ) } = \log \frac { b _ { n , t ^ { \prime } , y _ { n } } ^ { ( p ) } } { b _ { n , t , y _ { n } } } .\tag{19}
$$

A positive value indicates that the additional observation reduces task loss, whereas a negative value denotes semantic degradation. For other tasks, $\ell _ { \mathrm { s e m } }$ is task-specific.

Let $C _ { n , t } ^ { ( p ) }$ denote the normalized sensing-resource cost of acquiring the next observation block under candidate profile $a _ { p }$ . This cost is represented by a system-dependent function:

$$
C _ { n , t } ^ { ( p ) } = \mathcal { C } \left( a _ { p } , t ^ { \prime } - t \right) .\tag{20}
$$

Here, $t ^ { \prime } - t$ is the candidate-block duration, and $\mathcal { C } ( \cdot )$ measures its sensing-resource cost. The supervised utility target combines this cost with the semantic gain:

$$
U _ { n , t } ^ { \mathrm { t a r g e t , } ( p ) } = G _ { n , t } ^ { ( p ) } - \lambda _ { \mathrm { c o s t } } C _ { n , t } ^ { ( p ) } .\tag{21}
$$

Here, $\lambda _ { \mathrm { c o s t } } \geq 0$ balances semantic gain and sensing cost.

The utility network is then trained on the decision–candidate samples using the mean-squared-error (MSE) objective:

$$
\mathcal { L } _ { \mathrm { u t i l i t y } } = \frac { 1 } { \vert S _ { \mathrm { U } } \vert } \sum _ { ( n , t , p ) \in S _ { \mathrm { U } } } \left( U _ { n , t } ^ { ( p ) } - U _ { n , t } ^ { \mathrm { t a r g e t } , ( p ) } \right) ^ { 2 } ,\tag{22}
$$

where $S _ { \mathrm { U } }$ denotes the utility-training set. With this objective, the learned utility approximates the conditional target mean given the semantic state and candidate context. Section III-D relates this conditional mean to VoI.

4) Profile Selection and Semantic Early Exit: At decision epoch t, $\dot { U } _ { t } ^ { ( p ) }$ predicts the utility of acquiring the next block under candidate profile $a _ { p } .$ After acquisition, the semantic state and candidate utilities are updated. Their maximum across feasible profiles is defined as

$$
U _ { t } ^ { \operatorname* { m a x } } = \operatorname* { m a x } _ { a _ { p } \in \mathcal { A } _ { t } ^ { \mathrm { f e a s } } } U _ { t } ^ { ( p ) } .\tag{23}
$$

When sensing continues, the agent selects

$$
a _ { t } ^ { * } = \operatorname * { a r g m a x } _ { a _ { p } \in \mathcal { A } _ { t } ^ { \mathrm { f e a s } } } U _ { t } ^ { ( p ) } ,\tag{24}
$$

and applies $a _ { t } ^ { * }$ to the next observation block. The selected profile therefore adapts the sensing-resource usage of subsequent acquisition according to the evolving semantic state.

Let $N _ { t } ^ { \mathrm { o b s } }$ denote the number of acquired CPIs at decision epoch t. To prevent premature termination, semantic early exit is enabled after $N _ { \mathrm { m i n } }$ CPIs and uses the stopping rule

$$
N _ { t } ^ { \mathrm { o b s } } \geq N _ { \operatorname* { m i n } } \quad \mathrm { a n d } \quad U _ { t } ^ { \mathrm { m a x } } \leq \tau .\tag{25}
$$

The agent then returns the current prediction; otherwise, it continues with $a _ { t } ^ { * }$ . Before $N _ { t } ^ { \mathrm { o b s } }$ reaches $N _ { \mathrm { m i n } }$ , the controller keeps sensing while allowing profile adaptation at each decision epoch. The parameter $\lambda _ { \mathrm { c o s t } }$ controls the semantic–cost balance; $N _ { \mathrm { m i n } }$ and τ govern early-exit eligibility.

## D. VoI Interpretation of the Continuation Utility

The semantic-gain target admits a VoI interpretation [32]. For classification, let $Y$ denote the random class label, $\mathcal { H } _ { t } =$ $h _ { t }$ the observed history, and $Z _ { t } ^ { ( p ) }$ the next block acquired under feasible profile $a _ { p } \in \mathcal { A } _ { t } ^ { \mathrm { f e a s } }$ . We assume exact posteriors for the following VoI analysis:

$$
{ \mathbf { b } } _ { t } = P ( Y \mid \mathcal { H } _ { t } = h _ { t } ) , \quad  { \mathbf { b } } _ { t ^ { \prime } } ^ { ( p ) } = P ( Y \mid \mathcal { H } _ { t } = h _ { t } , Z _ { t } ^ { ( p ) } ) .\tag{26}
$$

Let $G _ { t } ^ { ( p ) }$ denote the population-level counterpart of the sample-wise semantic gain in (18). Under the ideal-posterior assumption, its expected log-loss reduction measures the information supplied by the next block.

Proposition 1 (Semantic VoI of the Next Observation Block). Under (26), the expected log-loss reduction produced by candidate profile $a _ { p }$ equals the conditional mutual information between the task label and the next observation block:

$$
\begin{array} { r } { \mathbb { E } \left[ G _ { t } ^ { ( p ) } \mid \mathcal { H } _ { t } = h _ { t } \right] = I \left( Y ; Z _ { t } ^ { ( p ) } \mid \mathcal { H } _ { t } = h _ { t } \right) \ge 0 . } \end{array}\tag{27}
$$

Proof. Conditioned on the observed history $\mathcal { H } _ { t } ~ = ~ h _ { t }$ , the expected semantic gain under profile $a _ { p }$ is

$$
\begin{array} { r l } & { \mathbb { E } \left[ G _ { t } ^ { ( p ) } \mid \mathcal { H } _ { t } = h _ { t } \right] = \mathbb { E } _ { Z _ { t } ^ { ( p ) } , Y \mid h _ { t } } \left[ \log \frac { P ( Y \mid h _ { t } , Z _ { t } ^ { ( p ) } ) } { P ( Y \mid h _ { t } ) } \right] } \\ & { \quad = \mathbb { E } _ { Z _ { t } ^ { ( p ) } \mid h _ { t } } \left[ D _ { \mathrm { K L } } ( P ( Y \mid h _ { t } , Z _ { t } ^ { ( p ) } ) \mid \mid P ( Y \mid h _ { t } ) ) \right] } \\ & { \quad = I \left( Y ; Z _ { t } ^ { ( p ) } \mid \mathcal { H } _ { t } = h _ { t } \right) . } \end{array}\tag{28}
$$

Here, $D _ { \mathrm { K L } } ( \cdot | | \cdot )$ denotes the Kullback–Leibler (KL) divergence. Since the KL divergence is nonnegative, the expected semantic gain is also nonnegative, although an individual realized gain may be negative. This proves the result. □

Let $C _ { t } ^ { ( p ) }$ denote the corresponding candidate-block cost in the population formulation. For candidate profile $a _ { p } .$ , this formulation defines the ideal cost-adjusted VoI as

$$
V _ { t } ^ { ( p ) } = I \left( Y ; Z _ { t } ^ { ( p ) } \mid \mathcal { H } _ { t } = h _ { t } \right) - \lambda _ { \mathrm { c o s t } } C _ { t } ^ { ( p ) } .\tag{29}
$$

Thus, under the ideal-posterior assumption, $V _ { t } ^ { \left( p \right) } > 0$ when the next block provides more expected task information than the sensing-resource penalty. This relation provides a theoretical basis for the learned utility.

Remark 1 (VoI Interpretation Versus Learned Utility). Under the MSE objective in (22), the population-optimal predictor for the candidate-conditioned input $\mathbf { \bar { v } } _ { t } ^ { ( p ) }$ is

$$
U ^ { * } \left( \mathbf { v } _ { t } ^ { ( p ) } \right) = \mathbb { E } \left[ U _ { t } ^ { \mathrm { t a r g e t } , ( p ) } \mid \mathbf { v } _ { t } ^ { ( p ) } \right] .\tag{30}
$$

Here, $U _ { t } ^ { \mathrm { t a r g e t } , ( p ) }$ is the population counterpart of the supervised target in (21), constructed using the fixed belief network. The learned utility therefore estimates the expected task-loss reduction minus sensing cost, conditioned on the semantic state and candidate context. Its interpretation in terms of conditional mutual information holds under the ideal-posterior assumption above. Ground-truth labels and future candidate observations are used only for utility-target construction during training and are not included in the online utility input.

## IV. SYSTEM DESIGN OF AGENTIC SEMS

This section presents a reference implementation of Agentic SemS, covering its network architectures, training, and online operation. Section V evaluates this implementation on a specific sensing task using multiple sensing-profile configurations.

## A. Semantic Belief Network Implementation

1) CPI-Level Feature Extraction: At CPI t, the reference sensing front end receives the measurement tuple

$$
\mathbf { m } _ { t } = \left( \mathbf { C } _ { t } , \mathbf { Q } _ { t } \right) ,\tag{31}
$$

where $\mathbf { C } _ { t } = \{ \mathbf { C } _ { t , n } \} _ { n = 1 } ^ { N _ { \mathrm { p k t } , t } }$ is the CSI packet sequence, and each $\mathbf { C } _ { t , n } \in \mathbb { C } ^ { N _ { \mathrm { { s c } , { t } } } \times M _ { t } }$ contains measurements over $N _ { \mathrm { s c } , t }$ retained subcarriers and $M _ { t }$ receive antennas. CSI is widely used in wireless sensing because it preserves fine-grained motioninduced channel variations across subcarriers and antennas, supporting delay–Doppler feature extraction [8]. The compactradio-measurement sequence is $\mathbf { Q } _ { t } = \{ \mathbf { d } _ { t , \ell } \} _ { \ell = 1 } ^ { L _ { t } }$ , where $\mathbf { d } _ { t , \ell } \in$ $\mathbb { R } ^ { N _ { \mathrm { W } , t } }$ contains $N _ { \mathrm { W , i } }$ <sub>t</sub> radio indicators, such as received-power, link-quality, and interference measurements. The active profile determines which of these inputs are available.

For CSI extraction, we follow the phase-compensation and feature-construction procedures in [13]. Phase compensation mitigates timing- and carrier-frequency-offset variations, while temporal-mean removal suppresses static components. Adaptive spectral processing combines 32 range bins from a rangedomain fast Fourier transform (FFT) with 128 Doppler bins from a temporal FFT, forming a range–Doppler spectrum over the intervals supported by the sensing configuration. Compact radio measurements are processed through an appropriate statistical or spectral front end to extract task-relevant temporal variations [11]. In the reference implementation, the selected compact-radio sequence is mean-centered to suppress static components, and a 128-point FFT extracts its CPIlevel temporal spectral feature. Both front ends produce a 128-dimensional CPI-level spectral feature. This feature is augmented with seven spectrum-derived statistics: log standard deviation, log total energy, log peak magnitude, negative- and positive-frequency energy ratios, normalized spectral centroid, and normalized spectral width, resulting in a 135-dimensional CPI observation. The reference encoder in (7) takes the form

$$
\begin{array} { r } { \mathcal { E } _ { \theta _ { \mathrm { o } } } ( \mathbf { o } _ { t } ) = \mathbf { W } _ { \mathrm { o } } \mathrm { L N } ( \mathbf { o } _ { t } ) + \beta _ { \mathrm { o } } , } \end{array}\tag{32}
$$

where LN(·) denotes layer normalization, $\mathbf { W } _ { \mathrm { o } } \in \mathbb { R } ^ { 1 9 2 \times 1 3 5 }$ is the learnable projection matrix, and $\beta _ { \mathrm { o } } \in \mathbb { R } ^ { 1 9 2 }$ is the learnable bias vector. The encoder projects the 135-dimensional observation to a 192-dimensional embedding.

2) Profile-Context Construction: Each sensing profile is represented by a fixed-dimensional context

$$
\begin{array} { r } { \mathbf q _ { p } = \left[ \pmb { \xi } _ { p } ^ { \mathsf { T } } , c _ { p } , \pmb { \iota } _ { p } ^ { \mathsf { T } } \right] ^ { \mathsf { T } } \in \mathbb { R } ^ { N _ { \mathbf { q } } } . } \end{array}\tag{33}
$$

Here, $\pmb { \xi } _ { p } ~ \in ~ \mathbb { R } ^ { N _ { \xi } }$ contains numerical sensing-configuration attributes, such as the sampling rate, signal-to-noise ratio (SNR), subcarrier usage, and temporal configuration. Before encoding, continuous attributes are scaled to comparable numerical ranges because they may have different physical units and magnitudes. The scalar $c _ { p } \in [ 0 , 1 ]$ is a normalized per-CPI sensing-resource score associated with profile ${ a _ { p } } .$ . The binary vector $\iota _ { p } \ \in \ \{ 0 , 1 \} ^ { N _ { \iota } }$ indicates the available measurement branches and applicable configuration attributes of profile $a _ { p }$ The attribute set is determined by the sensing interface and remains fixed across profiles within a deployment. A twolayer profile encoder with a 64-dimensional hidden layer maps $\mathbf { q } _ { p }$ to a 192-dimensional embedding shared by the belief and utility networks. After joint training with the belief network, the encoder is frozen and reused by the utility network.

3) Network Architecture: The 192-dimensional observation, active-profile, and positional embeddings form $\mathbf { x } _ { t }$ in (9). The causal Transformer $\mathcal { F } _ { \theta _ { \mathrm { B } } }$ uses four self-attention layers with six heads, a 192-dimensional model, and a 768- dimensional feed-forward layer. At each CPI $t ,$ it outputs $\mathbf { r } _ { t } ~ \in \mathbb { R } ^ { 1 9 2 }$ , and the classification head maps $\mathbf { r } _ { t }$ to the $C -$ dimensional semantic belief $\mathbf { b } _ { t } \in \mathbb { R } ^ { C }$

4) Two-Stage Weighted-Prefix Training: The weightedprefix objective is evaluated at intermediate prefixes separated by $K = 4 ~ \mathrm { C P I s }$ . In this implementation, consecutive CPIs use overlapping windows with a 32-ms stride; hence, $K = 4$ places decision epochs 128 ms apart. Prefix supervision begins after 20% of a sequence is observed, and the prefix weights use $w _ { n , t } = ( t / L _ { n } ) ^ { \kappa }$ with $\kappa = 2$ . Training proceeds in two stages. Stage 1 uses fixed-profile trajectories to learn reliable representations at intermediate prefixes and complete sequences, with $\omega _ { \mathrm { f } } = 0 . 7$ . Stage 2 improves robustness to online profile changes using $\omega _ { \mathrm { f } } ~ = ~ 0 . 5$ , with 30% fixed-profile and 70% mixed-profile trajectories generated using feasible adjacent profile transitions between observation blocks. Stage 1 trains for 100 epochs with a learning rate of $1 0 ^ { - 3 }$ , and Stage 2 trains for 50 epochs with a learning rate of $1 0 ^ { - 4 }$ . Both stages use batch size 128, AdamW with weight decay $1 0 ^ { - 4 }$ , cosine annealing to $1 0 ^ { - 6 }$ , gradient clipping with a maximum norm of 1, and random seed 42.

5) KV-Cached Inference: For the causal Transformer, the previously computed key and value tensors are retained as

$$
\mathcal { C } _ { t } ^ { \mathrm { K V } } = \left\{ \mathbf { K } _ { 1 : t } ^ { \left( \ell \right) } , \mathbf { V } _ { 1 : t } ^ { \left( \ell \right) } \right\} _ { \ell = 1 } ^ { L _ { \mathrm { B } } } ,\tag{34}
$$

where $L _ { \mathrm { B } }$ is the number of Transformer layers. A newly acquired CPI is processed incrementally according to

$$
\left( \mathbf { r } _ { t } , \mathcal { C } _ { t } ^ { \mathrm { K V } } \right) = \mathcal { F } _ { \boldsymbol { \theta } _ { \mathrm { B } } } \left( \mathbf { x } _ { t } ; \mathcal { C } _ { t - 1 } ^ { \mathrm { K V } } \right) .\tag{35}
$$

In the reference implementation, streaming inference uses a KV cache for at most 128 CPI-level tokens, with each token corresponding to one acquired CPI input $\mathbf { x } _ { t } .$ . The cache capacity is an implementation parameter and can be increased for applications requiring a longer sensing horizon, subject to available memory. The cache also persists across profile transitions because the stored tokens retain their profile embeddings.

## B. Semantic Utility Network and Online Agent Implementation

1) Candidate-Conditioned Utility Input: With the semantic belief network fixed, the utility network concatenates the 192- dimensional latent representation $\mathbf { r } _ { t } ,$ , C-dimensional semantic belief $\mathbf { b } _ { t } .$ , and 192-dimensional candidate-profile embedding $\mathbf { g } _ { p }$ for each $a _ { p } \in \mathcal { A } _ { t } ^ { \mathrm { f e a s } }$ at every decision epoch. This forms the (384 + C)-dimensional candidate-conditioned input ${ \bf v } _ { t } ^ { \left( p \right) }$

2) Utility Network Architecture: The candidate-conditioned input is processed by a lightweight multilayer perceptron with hidden dimensions of 256 and 128. The first linear layer is followed by LayerNorm and a Gaussian error linear unit (GELU) [33], while the second hidden layer also uses GELU. A final linear layer produces a continuation utility. The same network is shared across all candidate profiles.

3) Utility-Target Generation: Utility targets are generated at decision epochs spaced by $K = 4$ CPIs. At each epoch, a candidate is evaluated using the next four-CPI observation block, and a training sample is constructed only when this complete block is available. For each feasible candidate, the target follows (19) and (21). Here, $c _ { p }$ denotes the normalized per-CPI sensing-resource cost, so the candidate block has cost $C _ { n , t } ^ { ( p ) } = 4 c _ { p }$ . Because this factor of four is fixed and common to all candidates, the reported implementation-level $\lambda _ { \mathrm { c o s t } }$ absorbs this factor, and the utility target uses $c _ { p }$ directly. Cumulative sensing cost remains calculated at the CPI level. Because $\lambda _ { \mathrm { c o s t } }$ is part of the supervised target, each value requires regenerated targets and a separately trained utility checkpoint. The parameters $N _ { \mathrm { m i n } }$ and τ are adjusted only at inference.

4) Utility Network Training: The utility network is trained for 30 epochs using record-level sampling. Each epoch samples 2,000 training records; for each record, all decision epochs with a complete next observation block and their feasible candidates contribute candidate-level training examples. Training uses the MSE objective in (22) and AdamW with a learning rate of $1 0 ^ { - 3 }$ , weight decay of $1 0 ^ { - 4 }$ , cosine annealing to $1 0 ^ { - 5 }$ an accumulation size of 128 records, and random seed 42. For each $\lambda _ { \mathrm { c o s t } }$ , validation retains the minimum-MSE checkpoint.

5) Online Profile Adaptation and Semantic Early Exit: The online agent maintains the active profile, current semantic state, and Transformer KV cache as persistent runtime states. In the reference controller, the sensing profiles are ordered by resource usage and the transition rule in (5) is instantiated as

$$
\mathcal { A } _ { t } ^ { \mathrm { f e a s } } = \left\{ a _ { p } \in \mathcal { A } ^ { \mathrm { c o m } } : \left| p - s _ { t } \right| \leq 1 \right\} ,\tag{36}
$$

which permits the current profile and its available adjacent lower- and higher-resource profiles. Stay retains the current profile, whereas Down and Up select the adjacent profiles. This transition rule is an implementation choice, not a requirement of Agentic SemS. The controller uses only acquired CPI history, active-profile context, and feasible candidates; it receives neither the complete record length nor future observations. The belief network processes each acquired CPI incrementally, while the utility network evaluates continuation profiles after each observation block is acquired. If sensing continues, the highest-utility candidate is selected for the next block. After acquisition, the semantic state and candidate utilities are updated. The KV cache is retained across profile transitions and reuses the observed context. Profile adaptation is available from the first decision epoch, whereas semantic early exit becomes eligible after $N _ { \mathrm { m i n } }$ has been reached. Thereafter, sensing continues under the selected profile when the maximum candidate utility exceeds $\tau ;$ otherwise, it stops and returns the current task prediction as the semantic output. Evaluation also terminates when a complete next block is unavailable or the selected profile reaches its configured CPI budget; these limits are enforced by the sensing environment rather than provided to the agent as future information.

## V. EXPERIMENTAL EVALUATION

We evaluate the reference implementation on a six-class WiFi gesture-recognition task. The dataset, sensing-profile realizations, controller configuration, and evaluation protocol are described first, followed by the experimental results.

## A. Experimental Setup

1) Dataset and Partitioning: We use Widar3.0 [31] for six-class WiFi gesture recognition. It was collected in the

TABLE I  
ORDERED SENSING PROFILES AND NORMALIZED PER-CPI SENSING-RESOURCE COSTS UNDER THE REFERENCE WEIGHTING.
<table><tr><td rowspan=2 colspan=1>Profile</td><td rowspan=1 colspan=1>Feature</td><td rowspan=1 colspan=1>Subcarriers</td><td rowspan=1 colspan=1>Sampling rate</td><td rowspan=1 colspan=1>Effective SNR</td><td rowspan=1 colspan=1>CPI length</td><td rowspan=1 colspan=1>CPI stride</td><td rowspan=1 colspan=1>CPI budget</td><td rowspan=1 colspan=1>Reference cost</td></tr><tr><td rowspan=1 colspan=1> $\psi$ </td><td rowspan=1 colspan=1> $\overline { { N _ { \mathrm { s c } } } }$ </td><td rowspan=1 colspan=1> $\overline { { F _ { s } \ ( \mathrm { k H z } ) } }$ </td><td rowspan=1 colspan=1> $\overline { { \Delta \gamma \ ( \mathrm { d B } ) } }$ </td><td rowspan=1 colspan=1> $\overline { { W _ { \mathrm { C P I } } } }$ </td><td rowspan=1 colspan=1> $I _ { \mathrm { C P I } }$ </td><td rowspan=1 colspan=1> $L _ { \mathrm { s e q } }$ </td><td rowspan=1 colspan=1> $c _ { p } ( \mathbf { w } _ { \mathrm { r e f } } )$ </td></tr><tr><td rowspan=1 colspan=1>Lightweight</td><td rowspan=1 colspan=1>RSSI</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>0.25</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1>0.236</td></tr><tr><td rowspan=1 colspan=1>Low</td><td rowspan=1 colspan=1>CSI</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>0.25</td><td rowspan=1 colspan=1>25</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>48</td><td rowspan=1 colspan=1>0.528</td></tr><tr><td rowspan=1 colspan=1>Low-Medium</td><td rowspan=1 colspan=1>CSI</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>25</td><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1>0.655</td></tr><tr><td rowspan=1 colspan=1>Medium</td><td rowspan=1 colspan=1>CSI</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>96</td><td rowspan=1 colspan=1>0.720</td></tr><tr><td rowspan=1 colspan=1>Medium-High</td><td rowspan=1 colspan=1>CSI</td><td rowspan=1 colspan=1>25</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>128</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>96</td><td rowspan=1 colspan=1>0.936</td></tr><tr><td rowspan=1 colspan=1>High</td><td rowspan=1 colspan=1>CSI</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>128</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>128</td><td rowspan=1 colspan=1>1.000</td></tr></table>

![](images/2fba2aa8f14b6fb0ced15c06fec80a3a1ecde0fe53148ab5c504841914bc886e.jpg)  
Fig. 2. Micro-Doppler spectrograms under the six sensing profiles, ordered from Lightweight to High from left to right.

5-GHz band with Intel 5300 network interface cards, one transmitter, and multiple receivers. Each receiver provides CSI over 30 subcarriers and three receive antennas, together with RSSI at a 1-kHz packet rate. It contains 165,724 records of six gestures: Clap, Draw-O, Draw-Zigzag, Push & Pull, Slide, and Sweep, from 16 subjects across three environments, five locations, and five orientations. Each record contains one pre-segmented gesture sequence under diverse motion and propagation conditions. A class-stratified 70/10/20 split yields 116,006 training, 16,572 validation, and 33,146 test records with comparable proportions. All six profile realizations of each source record remain in the same partition to prevent cross-profile leakage. The training split fits both networks, validation selects their checkpoints and controller parameters $( \lambda _ { \mathrm { c o s t } } , N _ { \mathrm { m i n } } , \tau )$ , and testing is used only for final evaluation.

2) Emulated Sensing Profiles: We construct six ordered sensing profiles from a prescribed set feasible under the communication constraints, as summarized in Table I. For the WiFi realization evaluated here, $\psi \in \{ \mathrm { C S I } , \mathrm { R S S I } \}$ identifies the sensing feature type. The Lightweight profile uses the RSSI field provided by Widar3.0 as its compact received-power measurement, whereas the other profiles use CSI to preserve fine-grained subcarrier responses. For the CSI-based profiles, the range–Doppler spectrum is averaged over the 32 retained range bins to obtain a 128-dimensional micro-Doppler vector. The five CSI-based profiles retain $N _ { \mathrm { s c } } \in \{ 1 0 , 1 5 , 2 0 , 2 5 , 3 0 \}$ subcarriers. Across the six profiles, sampling rates $F _ { s } \in \ U$ {0.25, 0.5, 1} kHz are obtained by downsampling the original 1-kHz Widar3.0 measurement stream, and effective signalto-noise ratio (SNR) settings $\Delta \gamma ~ \in ~ \{ 2 0 , 2 5 , 3 0 \}$ dB are applied where applicable. Larger configurations preserve richer sensing information at greater measurement and processing demand; the High profile retains the original observation. Each Widar3.0 record contributes one receive-antenna stream shared across all six profile realizations.

CPI lengths $W _ { \mathrm { C P I } } ~ \in ~ \{ 3 2 , 6 4 , 1 2 8 \}$ packets use quarterlength strides $I _ { \mathrm { C P I } } \in \{ 8 , 1 6 , 3 2 \}$ packets. Together with the profile-dependent sampling rate, each CPI spans 128 ms and produces one CPI-level observation every 32 ms across all profiles. The sequence length $L _ { \mathrm { s e q } } \in \{ 4 8 , 6 4 , 9 6 , 1 2 8 \}$ specifies the maximum CPI budget configured for each emulated profile. Because Widar3.0 provides pre-segmented gesture records, each profile realization is also bounded by the corresponding source-record duration. For each source recording, all profile observations are generated offline and temporally aligned over their common interval so that the same CPI represents the same underlying time interval, enabling candidatespecific utility targets to be constructed from the subsequent observation block. During online evaluation, the controller processes only observations from its selected profiles and updates the semantic state sequentially until semantic early exit or an environment-imposed availability limit is reached. Fig. 2 compares the resulting micro-Doppler spectrograms; higher profiles show clearer structure and richer information.

3) Normalized Sensing-Resource Cost: We quantify the relative sensing-resource requirements of profile $a _ { p }$ using

$$
\phi _ { p } = \left[ \frac { N _ { \mathrm { s c } , p } } { 3 0 } , \frac { F _ { s , p } } { 1 \mathrm { \ k H z } } , \widetilde { \gamma } _ { p } , \frac { W _ { \mathrm { C P I } , p } } { 1 2 8 } , \eta _ { \psi , p } \right] ^ { \mathsf { T } } ,\tag{37}
$$

where the components represent normalized subcarrier usage, sampling rate, observation quality, CPI length, and feature representation. For $\Delta \gamma _ { p } \in \{ 2 0 , 2 5 , 3 0 \}$ dB, $\widetilde { \gamma } _ { p } = \Delta \gamma _ { p } / ( 3 5 ~ \mathrm { d B } )$ while $\widetilde \gamma _ { p } = 1$ for the undegraded High profile. The representation component is $\eta _ { \psi , p } = 1$ for CSI and 0.2 for RSSI. The 35-dB denominator and the factor 0.2 are fixed reference scales used consistently across all experiments; they define relative profile costs rather than calibrated energy ratios. The RSSI profile has zero subcarrier usage, and CPI stride is omitted because it is determined by CPI length.

For nonnegative resource weights $\mathbf { w } = [ w _ { 1 } , \ldots , w _ { 5 } ] ^ { \mathsf { T } }$ satisfying $\textstyle \sum _ { i = 1 } ^ { 5 } { w _ { j } } = 1$ , the normalized sensing-resource cost is $c _ { p } ( \mathbf { w } ) = \mathbf { w } ^ { \mathsf { T } } \phi _ { p }$ . The weights are fixed rather than learned. The reference weights $\mathbf { w } _ { \mathrm { r e f } }$ are obtained by normalizing $[ 1 , 1 , 0 . 7 5 , 0 . 5 , 1 ] ^ { \mathsf { T } }$ . This setting gives comparable importance to subcarrier usage, sampling rate, and feature representation, with smaller contributions from observation quality and CPI length. The resulting $c _ { p }$ is a relative resource index rather than a measurement of physical energy consumption. Table I reports $c _ { p } ( \mathbf { w } _ { \mathrm { r e f } } )$ for each profile, with unit cost for High. Cumulative sensing cost sums these per-CPI costs over the acquired sequence, using the same weights in all experiments. For this WiFi realization, the context in (33) has 11 entries: the six configuration attributes in Table I, the per-CPI cost $c _ { p } ,$ and four binary applicability indicators for compact measurements, CSI, subcarrier count, and SNR.

TABLE II  
RECOGNITION AND RESOURCE USE ON THE TEST SET FOR FIXED AND ADAPTIVE SENSING CONTROLS.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Profile policy</td><td rowspan=1 colspan=1> $\lambda _ { \mathrm { c o s t } }$ </td><td rowspan=1 colspan=1>T</td><td rowspan=1 colspan=1>Macro-F1 (%)</td><td rowspan=1 colspan=1> $\overline { { P } }$ </td><td rowspan=1 colspan=1> ${ \overline { { R } } } _ { \mathrm { o b s } } \ ( \% )$ </td><td rowspan=1 colspan=1> $\overline { { C } }$ </td></tr><tr><td rowspan=1 colspan=1>Full-sequence High</td><td rowspan=1 colspan=1>Fixed High</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Disabled</td><td rowspan=1 colspan=1>90.70</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>100.00</td><td rowspan=1 colspan=1>1.000</td></tr><tr><td rowspan=1 colspan=1>Fixed High with early exit</td><td rowspan=1 colspan=1>Fixed High</td><td rowspan=1 colspan=1>0.005</td><td rowspan=1 colspan=1>-0.05</td><td rowspan=1 colspan=1>90.16</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>88.51</td><td rowspan=1 colspan=1>0.885</td></tr><tr><td rowspan=1 colspan=1>Full-sequence Medium-High</td><td rowspan=1 colspan=1>Fixed Medium-High</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Disabled</td><td rowspan=1 colspan=1>87.62</td><td rowspan=1 colspan=1>0.936</td><td rowspan=1 colspan=1>100.00</td><td rowspan=1 colspan=1>0.936</td></tr><tr><td rowspan=1 colspan=1>Fixed Medium-High with early exit</td><td rowspan=1 colspan=1>Fixed Medium-High</td><td rowspan=1 colspan=1>0.005</td><td rowspan=1 colspan=1>-0.025</td><td rowspan=1 colspan=1>86.80</td><td rowspan=1 colspan=1>0.936</td><td rowspan=1 colspan=1>86.90</td><td rowspan=1 colspan=1>0.813</td></tr><tr><td rowspan=1 colspan=1>Adaptive profiles without early exit</td><td rowspan=1 colspan=1>Adaptive</td><td rowspan=1 colspan=1>0.005</td><td rowspan=1 colspan=1>Disabled</td><td rowspan=1 colspan=1>86.76</td><td rowspan=1 colspan=1>0.852</td><td rowspan=1 colspan=1>100.00</td><td rowspan=1 colspan=1>0.852</td></tr><tr><td rowspan=1 colspan=1>Agentic SemS reference</td><td rowspan=1 colspan=1>Adaptive</td><td rowspan=1 colspan=1>0.005</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>85.79</td><td rowspan=1 colspan=1>0.866</td><td rowspan=1 colspan=1>86.74</td><td rowspan=1 colspan=1>0.747</td></tr><tr><td rowspan=1 colspan=1>Agentic SemS high-F1</td><td rowspan=1 colspan=1>Adaptive</td><td rowspan=1 colspan=1>0.005</td><td rowspan=1 colspan=1>-0.20</td><td rowspan=1 colspan=1>86.75</td><td rowspan=1 colspan=1>0.852</td><td rowspan=1 colspan=1>99.87</td><td rowspan=1 colspan=1>0.851</td></tr></table>

![](images/0d764d564805f2c63293ac8d553f97580661e877536c6fa06ffbe5990f661f59.jpg)  
Fig. 3. Macro-F1 versus sensing cost across fixed and adaptive strategies.

4) Controller Parameter Selection: Controller parameters are selected exclusively on the validation set. We evaluate six utility checkpoints corresponding to $\lambda _ { \mathrm { c o s t } } \in$ {0.005, 0.01, 0.025, 0.05, 0.075, 0.10} and nine τ values from −0.20 to 0 in increments of 0.025. Based on the validation analysis, $N _ { \mathrm { m i n } } = 8 ~ \mathrm { C P I s }$ is used for all exit-enabled settings. Validation selects $( \lambda _ { \mathrm { c o s t } } , \tau ) = ( 0 . 0 0 5 , - 0 . 2 0 )$ for the high-F1 setting and (0.005, 0) for the resource-efficient reference setting. All parameters are fixed before test evaluation. Adaptive variants use the same randomly sampled initial profile for each record, generated with seed 42.

## B. Evaluation Metrics

Recognition performance is evaluated using Macro-F1. For record $n ,$ let $t _ { n } ^ { \mathrm { s t o p } }$ denote the number of acquired CPIs, $L _ { n }$ the number of valid CPIs in the complete record, $s _ { n , t }$ the profile active at CPI $t ,$ and $c _ { \mathrm { H } }$ the per-CPI cost of High. Unlike the profile-level budget $L _ { \mathrm { s e q } } , \ L _ { n }$ is determined by the available duration of record n and is used only for metric normalization, not as a controller input. We define the normalized average profile cost and observation ratio as

$$
P _ { n } = \frac { 1 } { t _ { n } ^ { \mathrm { s t o p } } } \sum _ { t = 1 } ^ { t _ { n } ^ { \mathrm { s t o p } } } \frac { c _ { s _ { n , t } } } { c _ { \mathrm { H } } } , \qquad R _ { n } = \frac { t _ { n } ^ { \mathrm { s t o p } } } { L _ { n } } ,\tag{38}
$$

and the normalized cumulative sensing cost as

$$
C _ { n } = \frac { 1 } { L _ { n } c _ { \mathrm { H } } } \sum _ { t = 1 } ^ { t _ { n } ^ { \mathrm { s t o p } } } c _ { s _ { n , t } } = P _ { n } R _ { n } .\tag{39}
$$

Here, $P _ { n }$ characterizes the average sensing-profile cost during acquisition, $R _ { n }$ measures the fraction of the complete sequence that is observed, and $C _ { n }$ captures their combined effect on cumulative sensing-resource use.

For a test set of N records, we report $\begin{array} { r } { \overline { { P } } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } P _ { n } , } \end{array}$ $\begin{array} { r } { \begin{array} { r c l } { \overline { { R } } _ { \mathrm { o b s } } } & { = } & { \frac { 1 } { N } \sum _ { n = 1 } ^ { N } R _ { n } , \overline { { C } } } \end{array} = \begin{array} { r c l } { \frac { 1 } { N } \sum _ { n = 1 } ^ { N } C _ { n } } \end{array} } \end{array}$ , and $\begin{array} { r l } { \overline { { L } } _ { \mathrm { o b s } } } & { { } = } \end{array}$ $\begin{array} { r } { \frac { 1 } { N } \sum _ { n = 1 } ^ { N } t _ { n } ^ { \mathrm { s i o p } } } \end{array}$ . In comparison with full-sequence High, $1 - { \overline { { P } } } .$ $1 \mathrm { ~ - ~ } \overline { { R } } _ { \mathrm { o b s } } .$ and $1 - \overline { { C } }$ quantify reductions in average profile cost, observation duration, and cumulative sensing cost, respectively. These metrics distinguish the effects of profile adaptation and semantic early exit while also capturing their combined resource impact. We additionally report the semantic early-exit rate and the proportions of continuation decisions that retain, increase, or decrease the sensing profile.

## C. Evaluation Protocol and Baselines

The evaluation separates adaptive profile selection from semantic early exit. Full-sequence High is the common reference, using the highest-resource configuration without early termination. Fixed High with semantic early exit varies only observation duration, whereas adaptive profiles without early exit vary only the profile sequence. The complete Agentic SemS controller combines both mechanisms; full-sequence Medium-High and its early-exit variant provide lower-resource fixed-profile references. All exit-enabled methods use the same $N _ { \mathrm { m i n } } = 8 – \mathrm { C P I }$ minimum-observation setting without access to the future record boundary. The threshold $\tau$ is selected on validation data and fixed before test evaluation. For each fixed-profile early-exit baseline, the continuation set contains only the selected fixed profile, and its predicted continuation utility is compared with τ. Higher-F1 and Lower-cost points characterize the resulting duration-control tradeoffs.

## D. Overall Recognition–Resource Performance

1) Fixed-Profile Recognition and Cost: Fig. 3 shows the six full-sequence fixed-profile baselines evaluated on the test set as blue triangular markers connected by the dashed curve. All baselines use the same final causal Transformer checkpoint and differ only by profile. From Lightweight to High, Macro-F1 increases from 51.02% to 62.75%, 74.19%, 80.95%, 87.62%, and 90.70%, respectively. The corresponding normalized cumulative costs are $\begin{array} { r l } { \overline { { C } } } & { { } = } \end{array}$ {0.236, 0.528, 0.655, 0.720, 0.936, 1.000}. Since each baseline uses a single profile throughout the complete sequence, its normalized cumulative cost is identical to the corresponding normalized per-CPI cost in Table I. This recognition–cost hierarchy provides the controller with multiple operating points for adapting sensing-resource use to the evolving semantic state.

![](images/65700d93de5e006b0732eb3003a474bea189638853d38485c8523129ed7a3591.jpg)  
Fig. 4. High-profile Macro-F1 comparison across belief networks.

2) Adaptive Profile Selection: All controller settings are fixed using validation data before the test results reported here are evaluated. Building on the fixed-profile baselines above, Table II separates the effects of adaptive profile selection and semantic early exit. Adaptive profiles without early exit observe the complete sequence, with $\overline { { R } } _ { \mathrm { { o b s } } } = 1 0 0 \%$ , but reduce the average profile cost and normalized cumulative sensing cost to 0.852. In comparison with full-sequence High, this setting still achieves 86.76% Macro-F1 at 14.81% lower cost. With $\overline { { C } } = 0 . 8 5 1$ and a 99.87% observation ratio, the high-F1 Agentic setting achieves 86.75% Macro-F1, and most of its resource reduction comes from profile adaptation.

3) Semantic Early Exit: With the sensing profile fixed at High, semantic early exit at $\tau = - 0 . 0 5$ retains 90.16% Macro-F1, only 0.54 percentage points below full-sequence High, while reducing the observation ratio and normalized cumulative cost from 100% and 1.000 to 88.51% and 0.885. Since $\overline { { P } } ~ = ~ 1$ , the savings arise entirely from shorter observation duration. This controlled comparison confirms that semantic early exit alone can reduce sensing use with a small recognition loss. The Agentic SemS reference further combines profile adaptation with early exit at τ = 0. Under the same $\lambda _ { \mathrm { c o s t } } ~ = ~ 0 . 0 0 5$ checkpoint, early exit reduces the observation ratio from 100% to 86.74% and cost from 0.852 to 0.747, a further 12.35% relative reduction, while Macro-F1 decreases by only 0.97 percentage points from 86.76% to 85.79%. In comparison with full-sequence High, the complete controller reduces cost by 25.33% while achieving 85.79% Macro-F1. Thus, adaptation lowers acquisition cost, whereas early exit removes low-benefit observations.

4) Per-Gesture Results: Table III examines the recognition– cost tradeoff of the resource-efficient Agentic SemS reference setting across gesture classes. In comparison with fullsequence High, cumulative sensing cost is reduced for every class, with reductions ranging from 12.36% for Clap to 33.72% for Sweep. The corresponding per-class F1 gaps range from 2.46 to 7.76 percentage points. These differences reflect the joint effects of adaptive profile selection and semantic early exit on the observation trajectories of individual gestures. Ground-truth labels are used only to compute the reported perclass statistics and remain unavailable to the online controller.

TABLE III  
PER-CLASS F1 AND NORMALIZED CUMULATIVE SENSING COST FOR FULL-SEQUENCE HIGH AND THE AGENTIC SEMS REFERENCE SETTING.
<table><tr><td rowspan=1 colspan=1>Gesture</td><td rowspan=1 colspan=1>Full-seq.High F1 (%)</td><td rowspan=1 colspan=1>Agenticreference (%)</td><td rowspan=1 colspan=1>∆F1 (pp)</td><td rowspan=1 colspan=1>C</td><td rowspan=1 colspan=1>Cost reductionvs. High (%)</td></tr><tr><td rowspan=1 colspan=1>Clap</td><td rowspan=1 colspan=1>92.00</td><td rowspan=1 colspan=1>89.54</td><td rowspan=1 colspan=1>-2.46</td><td rowspan=1 colspan=1>0.876</td><td rowspan=1 colspan=1>12.36</td></tr><tr><td rowspan=1 colspan=1>Draw-O</td><td rowspan=1 colspan=1>90.48</td><td rowspan=1 colspan=1>85.23</td><td rowspan=1 colspan=1>-5.25</td><td rowspan=1 colspan=1>0.775</td><td rowspan=1 colspan=1>22.53</td></tr><tr><td rowspan=1 colspan=1>Draw-Zigzag</td><td rowspan=1 colspan=1>94.95</td><td rowspan=1 colspan=1>90.74</td><td rowspan=1 colspan=1>-4.21</td><td rowspan=1 colspan=1>0.674</td><td rowspan=1 colspan=1>32.57</td></tr><tr><td rowspan=1 colspan=1>Push &amp; Pull</td><td rowspan=1 colspan=1>90.15</td><td rowspan=1 colspan=1>85.35</td><td rowspan=1 colspan=1>-4.80</td><td rowspan=1 colspan=1>0.783</td><td rowspan=1 colspan=1>21.68</td></tr><tr><td rowspan=1 colspan=1>Slide</td><td rowspan=1 colspan=1>87.01</td><td rowspan=1 colspan=1>82.04</td><td rowspan=1 colspan=1>-4.97</td><td rowspan=1 colspan=1>0.734</td><td rowspan=1 colspan=1>26.60</td></tr><tr><td rowspan=1 colspan=1>Sweep</td><td rowspan=1 colspan=1>89.59</td><td rowspan=1 colspan=1>81.83</td><td rowspan=1 colspan=1>-7.76</td><td rowspan=1 colspan=1>0.663</td><td rowspan=1 colspan=1>33.72</td></tr><tr><td rowspan=1 colspan=1>Overall Macro</td><td rowspan=1 colspan=1>90.70</td><td rowspan=1 colspan=1>85.79</td><td rowspan=1 colspan=1>-4.91</td><td rowspan=1 colspan=1>0.747</td><td rowspan=1 colspan=1>25.33</td></tr></table>

## E. Belief Network and Motivation for Adaptive Control

1) Belief-Network Comparison: Under the High profile, we compare independently trained MobileViT-XXS [34], gated recurrent unit (GRU) [35], and KV-cached causal Transformer models to determine a suitable architecture for the belief network. For this comparison, all models use the same 49,718- record Widar3.0 subset with a 70/30 train/test split. Fig. 4 reports Macro-F1 values of 91.93%, 77.40%, and 91.29%, respectively. MobileViT-XXS treats each pre-segmented record as a single noncausal input, whereas the GRU and causal Transformer process it as an ordered sequence of CPI observations. The causal Transformer approaches the performance of MobileViT-XXS while supporting causal prefix processing and incremental KV-cached inference required for closed-loop sensing control. Agentic SemS therefore employs the causal Transformer as its belief network.

2) Profile-Dependent Recognition: Fig. 5 presents the complete-sequence MobileViT-XXS confusion matrices across the six sensing profiles, illustrating how class separability changes with the available sensing information. Recognition generally improves from Lightweight to High, although some gesture classes retain strong recognition under lower-resource profiles. The highest-resource profile is therefore not always necessary, motivating profile adaptation according to the evolving semantic evidence. The online Agentic SemS controller uses the causal Transformer selected above.

3) Semantic-Gain Analysis: Fig. 6 examines the semantic gains used to construct the utility targets. The left plot shows the gain distributions for the six profiles and their aggregate, while the right plot reports the proportions of negative, numerically zero, and positive gains. Across 29,941 candidate evaluations from 1,000 deterministic class-stratified validation records, 62.77% of the realized gains are positive, 36.66% are negative, and 0.57% are numerically zero. The overall mean and median are 0.115 and 0.036, respectively. A positive gain indicates that the next observation block increases the probability assigned to the correct class, whereas a negative gain indicates that the updated belief assigns it a lower probability. These gains characterize individual continuations rather than the intrinsic quality of a sensing profile. The same profile can therefore produce different gains depending on the current semantic state and the acquired observation. Negative gains for individual observations do not contradict the nonnegative expected VoI in Section III-D, since the latter is defined as an expectation over possible future observations.

![](images/1c7d64248e59334da75f0c9812b2fcf854ab283bee09054782003e52e4cf80f5.jpg)

![](images/312085376dfe33792484f09d5c08fcf4a4917e9e91c97bcf3791174ba427b9d3.jpg)

![](images/522e888cab90afc9bb0f15f96a355a219136d0a1163d1a2d933fd7862c580bf1.jpg)

![](images/cdf2c2bff619adb5036cd0cee42bd341b1cd1d899f948164fb8e80bfec358205.jpg)

![](images/21f63ac35ee9bedabc04d6597ac8a2d03916f31e25eec75d6528b736442975e8.jpg)

![](images/819a7f52edd8f7ea1a1a978a00c717d36ca687b1cac7c3f60107dada94b24fde.jpg)

![](images/23b7795ca6d9586ed1f17983e32045b24817f546e8b2f489fd1ce1bfcf216de2.jpg)  
Fig. 5. Class-normalized confusion matrices of the complete-sequence MobileViT-XXS models across the six emulated sensing profiles.  
Fig. 6. Semantic-gain distributions across candidate profiles. Fig. 7. Macro-F1 versus observation fraction. Fig. 8. Validation tradeoff across $\lambda _ { \mathrm { c o s t } }$ and τ.  
VALIDATION SENSITIVITY TO THE MINIMUM OBSERVATION BUDGET FORFIXED $\lambda _ { \mathrm { c o s t } } = 0 . 0 0 5$ AND $\tau = 0 .$

The observed variation motivates candidate-conditioned utility prediction rather than fixed profile-upgrade decisions.

VALIDATION EFFECT OF τ FOR THE FIXED $\lambda _ { \mathrm { c o s t } } = 0 . 0 0 5$ UTILITY CHECKPOINT WITH $N _ { \mathrm { m i n } } = 8 \mathrm { C P I s }$
<table><tr><td rowspan=1 colspan=1> $\underline { { N _ { \mathrm { m i n } } } }$ (CPIs)</td><td rowspan=1 colspan=1>Macro-F1 (%)</td><td rowspan=1 colspan=1> $\overline { { R } } _ { \mathrm { o b s } }$ (%)</td><td rowspan=1 colspan=1> $\overline { { \overline { { C } } } }$ </td><td rowspan=1 colspan=1>Exit rate (%)</td><td rowspan=1 colspan=1>Average exit CPI</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>22.22</td><td rowspan=1 colspan=1>12.03</td><td rowspan=1 colspan=1>0.0899</td><td rowspan=1 colspan=1>97.50</td><td rowspan=1 colspan=1>4.60</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>85.92</td><td rowspan=1 colspan=1>87.03</td><td rowspan=1 colspan=1>0.7487</td><td rowspan=1 colspan=1>37.93</td><td rowspan=1 colspan=1>35.87</td></tr><tr><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>85.92</td><td rowspan=1 colspan=1>87.03</td><td rowspan=1 colspan=1>0.7487</td><td rowspan=1 colspan=1>37.93</td><td rowspan=1 colspan=1>35.87</td></tr><tr><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>85.94</td><td rowspan=1 colspan=1>87.11</td><td rowspan=1 colspan=1>0.7495</td><td rowspan=1 colspan=1>37.88</td><td rowspan=1 colspan=1>35.96</td></tr></table>

## F. Parameter Selection and Early Exit

4) Recognition versus Observed Prefix: Fig. 7 shows how test-set recognition evolves as more of a High-profile sequence is observed. The 20%, 40%, 60%, 80%, and 100% prefix settings correspond to average realized observation fractions of 24.03%, 43.94%, 64.09%, 83.46%, and 100%, respectively, yielding Macro-F1 values of 39.68%, 63.66%, 78.12%, 86.85%, and 90.70%. Improvements are largest early and diminish near the complete sequence, showing that additional sensing becomes less valuable as evidence accumulates and motivating continuation decisions from the current belief state.

1) Role of the Minimum Observation Budget: The minimum observation budget $N _ { \mathrm { m i n } }$ and the early-exit threshold τ govern complementary parts of stopping. The former determines the earliest decision epoch at which semantic early exit is permitted, whereas the latter determines whether the predicted continuation utility is sufficiently low to stop thereafter. Table IV evaluates $N _ { \mathrm { m i n } }$ on all 16,572 validation records with $\lambda _ { \mathrm { c o s t } } = 0 . 0 0 5$ and $\tau = 0$ . Allowing semantic early exit at the first utility evaluation with $N _ { \operatorname* { m i n } } = 4$ causes 97.50% of the records to exit early and reduces Macro-F1 to 22.22%. This shows that four CPIs often provide insufficient semantic evidence for reliable stopping, causing premature exits before the belief becomes informative. Increasing $N _ { \mathrm { m i n } }$ to 8 CPIs raises Macro-F1 to 85.92% with a normalized cumulative cost of 0.7487. Results at 8 and 12 CPIs are identical, while increasing the budget to 16 CPIs changes Macro-F1 by only

<table><tr><td rowspan=1 colspan=1>Exit setting</td><td rowspan=1 colspan=1>Macro-F1 (%)</td><td rowspan=1 colspan=1> $\overline { { P } }$ </td><td rowspan=1 colspan=1> $\overline { { R } } _ { \mathrm { o b s } }$ (%)</td><td rowspan=1 colspan=1> $\overline { { C } }$ </td><td rowspan=1 colspan=1>Exit rate (%)</td></tr><tr><td rowspan=1 colspan=1>τ = −0.20</td><td rowspan=1 colspan=1>86.74</td><td rowspan=1 colspan=1>0.852</td><td rowspan=1 colspan=1>99.89</td><td rowspan=1 colspan=1>0.851</td><td rowspan=1 colspan=1>0.41</td></tr><tr><td rowspan=1 colspan=1>τ = −0.15</td><td rowspan=1 colspan=1>86.72</td><td rowspan=1 colspan=1>0.854</td><td rowspan=1 colspan=1>99.35</td><td rowspan=1 colspan=1>0.848</td><td rowspan=1 colspan=1>2.27</td></tr><tr><td rowspan=1 colspan=1>τ = −0.10</td><td rowspan=1 colspan=1>86.66</td><td rowspan=1 colspan=1>0.856</td><td rowspan=1 colspan=1>96.62</td><td rowspan=1 colspan=1>0.825</td><td rowspan=1 colspan=1>12.39</td></tr><tr><td rowspan=1 colspan=1>τ = −0.05</td><td rowspan=1 colspan=1>86.48</td><td rowspan=1 colspan=1>0.860</td><td rowspan=1 colspan=1>92.31</td><td rowspan=1 colspan=1>0.790</td><td rowspan=1 colspan=1>25.16</td></tr><tr><td rowspan=1 colspan=1> $\tau = - 0 . 0 2 5$ </td><td rowspan=1 colspan=1>86.19</td><td rowspan=1 colspan=1>0.862</td><td rowspan=1 colspan=1>89.88</td><td rowspan=1 colspan=1>0.771</td><td rowspan=1 colspan=1>31.31</td></tr><tr><td rowspan=1 colspan=1>τ = 0</td><td rowspan=1 colspan=1>85.92</td><td rowspan=1 colspan=1>0.865</td><td rowspan=1 colspan=1>87.03</td><td rowspan=1 colspan=1>0.749</td><td rowspan=1 colspan=1>37.93</td></tr></table>

TABLE VI  
EARLY-EXIT SETTINGS UNDER FIXED PROFILES ON THE TEST SET.
<table><tr><td rowspan=1 colspan=1>Profile</td><td rowspan=1 colspan=1>Exit setting</td><td rowspan=1 colspan=1>T</td><td rowspan=1 colspan=1>Macro-F1 (%)</td><td rowspan=1 colspan=1> $\overline { { R } } _ { \mathrm { o b s } }$ (%)</td><td rowspan=1 colspan=1> $\overline { { C } }$ </td></tr><tr><td rowspan=1 colspan=1>High</td><td rowspan=1 colspan=1>Disabled</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>90.70</td><td rowspan=1 colspan=1>100.00</td><td rowspan=1 colspan=1>1.000</td></tr><tr><td rowspan=1 colspan=1>High</td><td rowspan=1 colspan=1>Higher-F1</td><td rowspan=1 colspan=1>-0.15</td><td rowspan=1 colspan=1>90.60</td><td rowspan=1 colspan=1>98.45</td><td rowspan=1 colspan=1>0.985</td></tr><tr><td rowspan=1 colspan=1>High</td><td rowspan=1 colspan=1>Lower-cost</td><td rowspan=1 colspan=1>-0.05</td><td rowspan=1 colspan=1>90.16</td><td rowspan=1 colspan=1>88.51</td><td rowspan=1 colspan=1>0.885</td></tr><tr><td rowspan=1 colspan=1>Medium-High</td><td rowspan=1 colspan=1>Disabled</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>87.62</td><td rowspan=1 colspan=1>100.00</td><td rowspan=1 colspan=1>0.936</td></tr><tr><td rowspan=1 colspan=1>Medium-High</td><td rowspan=1 colspan=1>Higher-F1</td><td rowspan=1 colspan=1>-0.125</td><td rowspan=1 colspan=1>87.48</td><td rowspan=1 colspan=1>97.45</td><td rowspan=1 colspan=1>0.912</td></tr><tr><td rowspan=1 colspan=1>Medium-High</td><td rowspan=1 colspan=1>Lower-cost</td><td rowspan=1 colspan=1>-0.025</td><td rowspan=1 colspan=1>86.80</td><td rowspan=1 colspan=1>86.90</td><td rowspan=1 colspan=1>0.813</td></tr></table>

TABLE VII

SENSITIVITY TO THE INITIAL SENSING PROFILE ON THE TEST SET.
<table><tr><td rowspan=1 colspan=1>Initial profile</td><td rowspan=1 colspan=1>Macro-F1 (%)</td><td rowspan=1 colspan=1> $\overline { { P } }$ </td><td rowspan=1 colspan=1> $\overline { { R } } _ { \mathrm { o b s } }$ (%)</td><td rowspan=1 colspan=1> $\overline { { C } }$ </td></tr><tr><td rowspan=1 colspan=1>Lightweight</td><td rowspan=1 colspan=1>80.09</td><td rowspan=1 colspan=1>0.719</td><td rowspan=1 colspan=1>91.65</td><td rowspan=1 colspan=1>0.661</td></tr><tr><td rowspan=1 colspan=1>Random</td><td rowspan=1 colspan=1>85.79</td><td rowspan=1 colspan=1>0.866</td><td rowspan=1 colspan=1>86.74</td><td rowspan=1 colspan=1>0.747</td></tr><tr><td rowspan=1 colspan=1>High</td><td rowspan=1 colspan=1>88.85</td><td rowspan=1 colspan=1>0.961</td><td rowspan=1 colspan=1>83.71</td><td rowspan=1 colspan=1>0.800</td></tr></table>

0.02 percentage points and cumulative cost by less than 0.001. We therefore select $N _ { \mathrm { m i n } } = 8$ as the smallest tested budget that avoids premature stopping at the first decision epoch while retaining later early-exit opportunities. This budget covers two four-CPI observation blocks, allowing the controller to adapt the sensing profile after CPI 4 and evaluate semantic early exit from CPI 8 onward. Since $N _ { \mathrm { m i n } }$ is applied only during inference, changing it does not require retraining either network. All exit-enabled comparisons therefore use $N _ { \mathrm { m i n } } = 8$

2) Validation-Set Search over $\lambda _ { \mathrm { c o s t } }$ and τ: The validation search evaluates six separately trained utility checkpoints using $\lambda _ { \mathrm { c o s t } } = 0 . 0 0 5 , 0 . 0 1$ , and 0.025–0.10 in increments of 0.025, with nine τ values from −0.20 to 0 in increments of 0.025. Because $\lambda _ { \mathrm { c o s t } }$ changes the supervised utility target, each value requires a separate checkpoint, whereas τ varies during inference without retraining. Fig. 8 summarizes all 54 combinations: each curve represents one checkpoint, its markers denote different τ values, and the dashed line connects the best observed recognition–cost tradeoffs. Increasing τ generally increases early stopping and lowers cost and Macro-F1. The highest validation Macro-F1, 86.74%, occurs at $\lambda _ { \mathrm { c o s t } } = 0 . 0 0 5$ and $\tau = - 0 . 2 0$ , defining the high-F1 setting. At this checkpoint, τ = 0 defines the resource-efficient reference by lowering normalized cumulative cost from 0.851 to 0.749.

3) Effect ofthe Early-Exit Threshold: Table V fixes $\lambda _ { \mathrm { c o s t } } =$ 0.005 and $N _ { \mathrm { m i n } } = 8$ while varying τ to isolate semantic early exit. As τ increases from −0.20 to 0, the exit rate rises from 0.41% to 37.93%, the observation ratio falls from 99.89% to 87.03%, and normalized cumulative cost falls from 0.851 to 0.749. Macro-F1 decreases from 86.74% to 85.92%. Thus, τ yields a 12.0% relative validation-cost reduction for a 0.82- percentage-point Macro-F1 decrease. The smaller change in average profile cost indicates that τ primarily controls sensing duration rather than profile selection.

## G. Fixed-Profile Early-Exit Baselines

Table VI isolates sensing-duration control by enabling semantic early exit while keeping the sensing profile fixed. All exit-enabled settings use $\lambda _ { \mathrm { c o s t } } = 0 . 0 0 5$ and $N _ { \mathrm { m i n } } = 8$ . Under fixed High, the Higher-F1 setting achieves 90.60% Macro-F1 with an observation ratio of 98.45%, whereas the Lower-cost setting reduces the observation ratio to 88.51% while retaining 90.16% Macro-F1. A similar trend is observed under fixed Medium-High. Because the sensing profile remains fixed, the reduction in cumulative sensing cost is attributable entirely to shorter sensing duration. Under fixed High, the Lower-cost setting reduces cumulative cost by 11.49% with a decrease of only 0.54 percentage points in Macro-F1. Under fixed Medium-High, the corresponding relative cost reduction is 13.14% with a decrease of 0.82 percentage points. These controlled results show that semantic early exit reduces sensingresource use independently of adaptive profile selection.

TABLE VIII  
NEXT-PROFILE PROBABILITIES ON THE TEST SET, CONDITIONAL ON CONTINUATION, FOR THE REFERENCE CONTROLLER.
<table><tr><td rowspan=1 colspan=1>Current</td><td rowspan=1 colspan=1>Lightweight</td><td rowspan=1 colspan=1>Low</td><td rowspan=1 colspan=1>Low-Medium</td><td rowspan=1 colspan=1>Medium</td><td rowspan=1 colspan=1>Medium-High</td><td rowspan=1 colspan=1>High</td></tr><tr><td rowspan=1 colspan=1>Lightweight</td><td rowspan=1 colspan=1>39.5</td><td rowspan=1 colspan=1>60.5</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td></tr><tr><td rowspan=1 colspan=1>Low</td><td rowspan=1 colspan=1>31.8</td><td rowspan=1 colspan=1>10.7</td><td rowspan=1 colspan=1>57.5</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td></tr><tr><td rowspan=1 colspan=1>Low-Medium</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>20.9</td><td rowspan=1 colspan=1>2.8</td><td rowspan=1 colspan=1>76.4</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td></tr><tr><td rowspan=1 colspan=1>Medium</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>19.9</td><td rowspan=1 colspan=1>9.0</td><td rowspan=1 colspan=1>71.1</td><td rowspan=1 colspan=1>一</td></tr><tr><td rowspan=1 colspan=1>Medium-High</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>24.0</td><td rowspan=1 colspan=1>5.3</td><td rowspan=1 colspan=1>70.7</td></tr><tr><td rowspan=1 colspan=1>High</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>9.9</td><td rowspan=1 colspan=1>90.1</td></tr></table>

## H. Controller Behavior

1) Initial-Profile Sensitivity: Table VII evaluates the sensitivity of the reference controller to its initial sensing profile. Starting from Lightweight produces the lowest cumulative sensing cost of 0.661, but also the lowest Macro-F1 of 80.09%. Starting from High increases Macro-F1 to 88.85% while increasing cumulative sensing cost to 0.800. The random initialization used in the main reference experiments lies between these cases, achieving 85.79% Macro-F1 with $\overline { { C } } \ : = \ : 0 . 7 4 7$ Because profile transitions are restricted to adjacent profiles and occur only at decision epochs, initialization affects early resource use and shifts the resulting recognition–cost operating point. Subsequent utility-based decisions can nevertheless revise the initial allocation throughout the observation sequence.

2) Profile Transitions: Table VIII reports the next-profile probabilities for continuation decisions of the reference controller, excluding stop decisions and record-end carryover. Across 304,371 continuation decisions, retaining the current profile, moving to a higher-resource profile, and moving to a lower-resource profile account for 54.45%, 30.82%, and 14.72%, respectively. Although upward transitions are more frequent than downward transitions in aggregate, both directions occur throughout online sensing. At the interior profiles, the controller frequently transitions to an adjacent profile as the semantic state evolves, whereas High remains unchanged in 90.12% of its continuation decisions. The presence of both upward and downward transitions shows that adaptation is not a one-way resource-reduction strategy. When the predicted continuation utility favors additional sensing resources, the controller can move to a higher-resource profile; at other decision epochs, it can move to a lower-resource profile to reduce sensing cost. Together with the stay decisions, these transitions demonstrate that sensing-resource allocation is repeatedly adjusted according to the evolving semantic state rather than following a predetermined profile schedule.

## VI. CONCLUSION

Agentic SemS is a closed-loop framework that adapts sensing profiles and observation duration to evolving semantic evidence. A profile-conditioned causal Transformer maintains semantic belief, while a candidate-conditioned utility network evaluates profiles. Adaptive profile selection controls acquisition cost, whereas semantic early exit controls sensing duration. Widar3.0 experiments demonstrate their complementary roles. In comparison with full-sequence High, the resource-efficient Agentic setting achieves 85.79% Macro-F1 while reducing normalized cumulative sensing cost by 25.33%. Adaptive profile selection alone reduces cumulative sensing cost by 14.81%, and semantic early exit at the same utility checkpoint provides a further 12.35% relative reduction over adaptive sensing without early exit, with Macro-F1 decreasing by only 0.97 percentage points. Semantic feedback thus jointly controls sensing configuration and duration. Future work will dynamically update the communication-feasible profile set and evaluate end-to-end energy, latency, signaling overhead, and control overhead on physical AI-RAN platforms.

## REFERENCES

[1] L. Kundu, X. Lin, R. Gadiyar, J.-F. Lacasse, and S. Chowdhury, “AI-RAN: Transforming RAN with AI-driven computing infrastructure,” IEEE Communications Magazine, vol. 64, no. 1, pp. 168–174, 2026.

[2] M. Polese, N. Mohamadi, S. D’Oro, L. Bonati, and T. Melodia, “Beyond connectivity: An open architecture for AI-RAN convergence in 6G,” IEEE Communications Magazine, vol. 64, no. 7, pp. 90–95, 2026.

[3] C. Fang, S. Guo, Z. Wang, H. Huang, H. Yao, and Y. Liu, “Data-driven intelligent future network: Architecture, use cases, and challenges,” IEEE Communications Magazine, vol. 57, no. 7, pp. 34–40, 2019.

[4] J. A. Zhang, M. L. Rahman, X. Huang, Y. J. Guo, S. Chen, and R. W. Heath, “Perceptive mobile networks: Cellular networks with radio vision via joint communication and radar sensing,” IEEE Vehicular Technology Magazine, vol. 16, no. 2, pp. 20–30, 2021.

[5] J. A. Zhang, F. Liu, C. Masouros, R. W. Heath, Jr., Z. Feng, L. Zheng, and A. Petropulu, “An overview of signal processing techniques for joint communication and radar sensing,” IEEE Journal of Selected Topics in Signal Processing, vol. 15, no. 6, pp. 1295–1315, 2021.

[6] K. Wu, J. A. Zhang, and Y. J. Guo, Joint Communications and Sensing: From Fundamentals to Advanced Techniques. Hoboken, NJ, USA: John Wiley & Sons, 2022.

[7] F. Liu, Y. Cui, C. Masouros, J. Xu, T. X. Han, Y. C. Eldar, and S. Buzzi, “Integrated sensing and communications: Toward dual-functional wireless networks for 6G and beyond,” IEEE Journal on Selected Areas in Communications, vol. 40, no. 6, pp. 1728–1767, 2022.

[8] K. Wu, Z. Wang, S.-L. Chen, J. A. Zhang, and Y. J. Guo, “ISAC: From human to environmental sensing,” IEEE Journal of Selected Topics in Electromagnetics, Antennas and Propagation, vol. 1, no. 1, pp. 84–98, 2025.

[9] K. Wu, J. Pegoraro, F. Meneghello, J. A. Zhang, J. O. Lacruz, J. Widmer, F. Restuccia, M. Rossi, X. Huang, D. Zhang, G. Caire, and Y. J. Guo, “Sensing in bistatic ISAC systems with clock asynchronism: A signal processing perspective,” IEEE Signal Processing Magazine, vol. 41, no. 5, pp. 31–43, September 2024.

[10] B. Li, W. Yuan, F. Liu, N. Wu, and S. Jin, “OTFS-based ISAC: How delay-doppler channel estimation assists environment sensing?” IEEE Wireless Communications Letters, vol. 13, no. 12, pp. 3563–3567, December 2024.

[11] Z. Wang, J. A. Zhang, K. Wu, and Y. J. Guo, “Rethinking RSSI for WiFi sensing,” npj Wireless Technology, vol. 2, p. 45, 2026.

[12] H. Wang, Z. Wang, and J. A. Zhang, “WiFi passive human tracking with multi-point differential CSI,” IEEE Transactions on Mobile Computing, vol. 25, no. 9, pp. 14 972–14 987, 2026.

[13] Z. Wang, J. A. Zhang, K. Wu, M. Xu, and Y. J. Guo, “Towards SISO bistatic sensing for ISAC,” arXiv preprint arXiv:2508.12614, 2025.

[14] C. Yang, X. Wang, and S. Mao, “TARF: Technology-agnostic RF sensing for human activity recognition,” IEEE Journal of Biomedical and Health Informatics, vol. 27, no. 2, pp. 636–647, 2023.

[15] A. Waqas and S. Coleri, “SNR and resource adaptive deep JSCC for distributed IoT image classification,” in 2025 IEEE 36th Annual International Symposium on Personal, Indoor and Mobile Radio Communications (PIMRC), 2025, pp. 1–6.

[16] X. Zhang, J. A. Zhang, C. Liu, W. Yuan, and G. Y. Li, “Semantic sensing: A task-oriented paradigm,” in ICC 2026–IEEE International Conference on Communications, 2026, pp. 1–6.

[17] Z. Lu, R. Li, K. Lu, X. Chen, E. Hossain, Z. Zhao, and H. Zhang, “Semantics-empowered communications: A tutorial-cum-survey,” IEEE Communications Surveys & Tutorials, vol. 26, no. 1, pp. 41–79, 2024.

[18] Y. E. Sagduyu, T. Erpek, A. Yener, and S. Ulukus, “Joint sensing and semantic communications with multi-task deep learning,” IEEE Communications Magazine, vol. 62, no. 9, pp. 74–81, 2024.

[19] L. Qiao, M. B. Mashhadi, Z. Gao, R. Tafazolli, M. Bennis, and D. Niyato, “Token communications: A large model-driven framework for cross-modal context-aware semantic communications,” IEEE Wireless Communications, vol. 32, no. 5, pp. 80–88, 2025.

[20] H. Xing, G. Zhu, D. Liu, H. Wen, K. Huang, and K. Wu, “Task-oriented integrated sensing, computation and communication for wireless edge AI,” IEEE Network, vol. 37, no. 4, pp. 135–144, 2023.

[21] Z. Du, F. Liu, W. Yuan, C. Masouros, Z. Zhang, S. Xia, and G. Caire, “Integrated sensing and communications for V2I networks: Dynamic predictive beamforming for extended vehicle targets,” IEEE Transactions on Wireless Communications, vol. 22, no. 6, pp. 3612–3627, 2023.

[22] X. Zhang, W. Yuan, C. Liu, J. Wu, and D. W. K. Ng, “Predictive beamforming for vehicles with complex behaviors in ISAC systems: A deep learning approach,” IEEE Journal of Selected Topics in Signal Processing, vol. 18, no. 5, pp. 828–841, 2024.

[23] Y. Xiao, G. Shi, and P. Zhang, “Toward agentic AI networking in 6G: A generative foundation model-as-agent approach,” IEEE Communications Magazine, vol. 63, no. 9, pp. 68–74, 2025.

[24] R. Zhang, S. Tang, Y. Liu, D. Niyato, Z. Xiong, S. Sun, S. Mao, and Z. Han, “Toward agentic AI: Generative information retrieval inspired intelligent communications and networking,” IEEE Communications Magazine, vol. 64, no. 1, pp. 197–204, 2026.

[25] F. Jiang, C. Pan, L. Dong, K. Wang, O. A. Dobre, and M. Debbah, “From large AI models to agentic AI: A tutorial on future intelligent communications,” arXiv preprint arXiv:2505.22311, 2025.

[26] W. Xie, G. Sun, R. Zhang, X. Liu, Y. Liu, J. Wang, D. Niyato, and P. Zhang, “Agentic AI for integrated sensing and communication: Analysis, framework, and case study,” arXiv preprint arXiv:2512.15044, 2025.

[27] J. Chen and X. Wang, “Learning-based intermittent CSI estimation with adaptive intervals in integrated sensing and communication systems,” IEEE Journal of Selected Topics in Signal Processing, vol. 18, no. 5, pp. 917–932, 2024.

[28] A. E. Shafie and A. K. Sultan, “Optimal random access and random spectrum sensing for an energy harvesting cognitive radio,” in 2012 IEEE 8th International Conference on Wireless and Mobile Computing, Networking and Communications (WiMob), 2012, pp. 403–410.

[29] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,” in Advances in Neural Information Processing Systems, vol. 30, 2017, pp. 5998–6008.

[30] R. Pope, S. Douglas, A. Chowdhery, J. Devlin, J. Bradbury, A. Levskaya, J. Heek, K. Xiao, S. Agrawal, and J. Dean, “Efficiently scaling transformer inference,” in Proceedings of Machine Learning and Systems, vol. 5, 2023, pp. 606–624.

[31] Y. Zhang, Y. Zheng, K. Qian, G. Zhang, Y. Liu, C. Wu, and Z. Yang, “Widar3.0: Zero-effort cross-domain gesture recognition with Wi-Fi,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 11, pp. 8671–8688, 2022.

[32] S. Kleinegesse, C. Drovandi, and M. U. Gutmann, “Sequential bayesian experimental design for implicit models via mutual information,” Bayesian Analysis, vol. 16, no. 3, pp. 773–802, 2021.

[33] D. Hendrycks and K. Gimpel, “Gaussian error linear units (GELUs),” arXiv preprint arXiv:1606.08415, 2016.

[34] S. Mehta and M. Rastegari, “MobileViT: Light-weight, general-purpose, and mobile-friendly vision transformer,” in International Conference on Learning Representations (ICLR), 2022.

[35] K. Cho, B. van Merrienboer, C. Gulcehre, D. Bahdanau, F. Bougares,¨ H. Schwenk, and Y. Bengio, “Learning phrase representations using RNN encoder–decoder for statistical machine translation,” in Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP). Doha, Qatar: Association for Computational Linguistics, 2014, pp. 1724–1734.