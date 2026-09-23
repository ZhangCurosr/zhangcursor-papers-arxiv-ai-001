# OmniFysics-Nano-V2 Technical Report: Understanding the Physical World Across Modalities

Yizhou Liu<sup>∗</sup>, Jinghang Han<sup>∗</sup>, Kaixiang Qiu, Qi He, Minghao Han, Yue Jiang, Xujia Chen, Wei Zou, Shunli Wang<sup>§</sup>, Lihua Zhang<sup>§</sup>, Dingkang Yang<sup>†,§</sup>

Physical Superintelligence Lab, Fysics AI

College of Intelligent Robotics and Advanced Manufacturing, Fudan University

<sup>∗</sup>Equal contribution, <sup>†</sup>Project lead, <sup>§</sup>Corresponding author

## Abstract

Omni-modal models have expanded multimodal interaction across vision, audio, speech, and language. However, their training is predominantly organized around semantic descriptions and general-purpose objectives, leaving physical attributes, interaction states, and causal mechanisms only partially specified. This gap is not simply a matter of modality coverage: adding more modalities does not by itself provide the supervision needed to connect observations with the physical structure of the world. We present OmniFysics-Nano-V2, a compact omni-modal model for physical-world perception and understanding. The model supports image, video, audio, speech, and text inputs within a shared reasoning framework, together with text and speech generation. To address the lack of explicit physical supervision, we construct a dual-branch physics-aware data pipeline that grounds salient objects in structured physical attributes and aligns visual changes with acoustic events, intermediate responses, and interaction outcomes. To address homogeneous training objectives, we curate reinforcement-learning prompts by reward diversity and adopt a two-stage Group Relative Policy Optimization curriculum that progresses from general task correctness to fine-grained physical perceptual reasoning. Experiments across multimodal, audio-visual, and physical reasoning benchmarks show that the proposed data and training strategy improves physical-world understanding while preserving broad omni-modal competence. The proposed model achieves leading result on 17 of 21 benchmarks against SOTA omni-modal models. By equipping AI systems with both omni-modal and physical-world perception capabilities, OmniFysics-Nano-V2 is poised to become a cornerstone of next-generation Physical AI.

Date: September 23, 2026

Corresponding: dicken@fyscis.ai, lihuazhang@fudan.edu.cn

Page: https://github.com/Fysics-AI/OmniFysics-Nano-V2

Hugging Face: https://huggingface.co/Fysics-AI/OmniFysics-Nano-V2

## 1 Introduction

Understanding the physical world requires more than recognizing objects, understanding scenes, or following language instructions. Currently, most Multimodal Large Language Models (MLLMs) [4, 26] focus primarily on vision-language interaction in general-purpose scenarios. Despite the promising performance of these models on visual recognition, Visual Question Answering (VQA), and instruction following [1, 3, 51], there remains a fundamental gap compared with physical-world understanding. A model endowed with physical-reasoning capabilities should resemble a human agent (a) Multi-modal understanding model with coarse-grained physical perception capability.

![](images/1b2233fbfe02dab0afcd948baac09f235449a985fa92d162da67d5870090ed92.jpg)  
Figure 1 Illustration of general multimodal models and the proposed OmniFysics-Nano-V2. In contrast to generic datasets and training strategies, we design a physics-aware data pipeline that incorporates both static-property and dynamic-event data. Moreover, we adopt multi-stage reinforcement learning to boost the model’s physical perception capabilities.

equipped with rich physical prior knowledge. It can infer the static physical properties of real-world objects, as well as comprehend dynamic physical events observed in video inputs. Physical perception and omni-modal understanding constitute indispensable foundations for the next generation of artificial intelligence [21].

Several existing omni-modal models have gone beyond the conventional vision-language interaction paradigm and are capable of performing understanding tasks under interleaved multi-modal information. For instance, closed-source models include GPT-4o [38] and Gemini [17], while open-source ones cover Qwen2.5-Omni [53], Qwen3-Omni [54], OmniVinci [56] and MiniCPM-o-4.5 [12]. Existing omni-modal systems are generally trained on web-scale semantic data and optimized for transcription, captioning, instruction following. They therefore learn to align modalities at the level of meaning, but receive little explicit supervisory information to identify specific physical properties, temporally organize an interaction, or distinguish a physically grounded explanation from a plausible description. These studies fully reveal the complementary nature of information across diferent modalities. For example, visual signals enable the inference of object geometry and motion, while audio information allows more precise localization of event timings. However, merely adding more modalities cannot directly endow models with physical understanding capabilities.

We argue that two fundamental factors currently limit the physical understanding capabilities of omni-modal models: Absent Physical Supervision and Homogeneous Training Strategies. At the level of dataset and supervisory information, generic multi-modal datasets [7, 15, 18] predominantly provide semantic descriptions of objects and events, while physical attributes, interaction states, and causal mechanisms are often implicit, ambiguous, or completely absent. Therefore, models struggle to associate cross-modal observations with the latent physical properties and laws governing real-world interactions. At the level of training strategy, existing models [3, 53, 54] are typically trained using general-purpose and largely homogeneous objectives. These objectives tend to prioritize general semantic understanding and cross-modal alignment over the explicit modeling of physical dynamics and causal relationships, thereby encouraging models to exploit superficial semantic correlations rather than acquire robust physical reasoning abilities.

To remedy Absent Physical Supervision, we build a physics-aware data pipeline with complementary Static and Dynamic branches. The static branch anchors salient objects to a structured physical prototype bank, augments them with intrinsic-property annotations, and enforces plausible value ranges and cross-property compatibility. Rather than treating frames in isolation, the dynamic branch focuses on event-centered video clips, aligning visible motion and state transitions with acoustic transients to trace an interaction from its initial state, through intermediate responses, to its final outcome. Together, the two branches connect object identity and intrinsic physical properties to the visual and acoustic evidence through which those properties are revealed during interaction.

![](images/ddb2e9a4066f9573cd4273e28efd68a4e63555c891aa8f8eaa19a3220b937dab.jpg)  
Figure 2 Demonstration of OmniFysics-Nano-V2 model capability and performance. The proposed physics-aware data pipeline and eficient policy-aware training strategies endow the model with a powerful ability to understand the physical world.

At the training stage, we move beyond Homogeneous Training Strategies with an eficient, policy-aware two-stage curriculum. Before optimization, we sample four stochastic rollouts for each candidate prompt and compare their rewards. Only prompts that produce reward variation are retained, as they provide meaningful within-group learning signals. Applied to mathematical, image–text, video, and mixed-modal reinforcement learning data, this reward-diversity filter removes over half of the candidate samples and saves more than 2,600 GPU-hours. We then organize GRPO into a two-stage curriculum: the first stage improves answer accuracy on general tasks, while the second further strengthens physical understanding through format-specific rewards. By combining selective data curation with curriculum-based optimization, the resulting policy learns not only to make accurate predictions but also to support its decisions with physically meaningful evidence.

In this paper, we propose OmniFysics-Nano-V2, a compact 4B omni-modal model for physical-world perception and understanding. Figure 1 illustrates the diferences between general multimodal models and our model. The model supports the understanding of image, video, audio, speech, and text inputs, alongside text and audio generation capabilities. With the help of physics-aware data pipeline and eficient two-stage policy-aware training strategy, our model achieves superior performance across multi-modal perception and physical perception tasks. Figure 2(a) illustrates the model’s capabilities in omni-modal data processing and physical understanding in detail. We evaluated on 21 benchmarks that cover general multimodal, audio, omni-modal / video, physical understanding, mathematical reasoning and physical reasoning benchmarks. Our model achieves SOTA performance on 17 benchmarks with 4B model size, even against 7B-scale baseline models. Figure 2(b) demonstrates the superior performance of OmniFysics-Nano-V2 on the Omni-modal Understanding Benchmarks and Physical Perception Benchmarks. Notably, our model achieves 98.27% on FysicsEval Understanding [21] and 59.42% on PhysUniBench [50], surpassing the state-of-the-art baselines by 5.57% and 11.42%, respectively.

## Our contributions are summarized as follows:

• We present OmniFysics-Nano-V2, a compact 4B omni-modal model that supports holistic understanding of images, videos, audios, speeches, and texts with speech output capability.

• We design a dual-branch physical supervision pipeline that complements static physical attribute grounding and dynamic physical event modeling.

• We develop a reward-diversity filtering for policy-aware RL data curation, and a two-stage GRPO strategy that optimizes general task performance and enhances fine-grained physical reasoning via intermediate perception supervision.

• The proposed model achieves leading result on 17 of 21 benchmarks against SOTA omni-modal models, with ablations showing substantial improvements over SFT and large reductions in RL data and compute.

## 2 Related Work

## 2.1 Omni-Modal Foundation Models

Contemporary omni-modal foundation models seek to place text, vision, audio, and speech within a single perceptual and generative system. GPT-4o [38] demonstrated end-to-end processing of text, image, audio, and video inputs together with text, audio, and image generation, establishing low-latency speech interaction as a central capability of native multimodal modeling. The Qwen-Omni family develops this direction through time-aligned audiovisual encoding and a Thinker–Talker architecture. Qwen2.5-Omni [53] supports streaming text and speech generation, Qwen3-Omni [54] improves long-context perception and modality balance, and Qwen3.5-Omni [48] further scales audiovisual training, context length, temporal grounding, and multilingual speech generation. Open models have pursued comparable breadth from diferent architectural perspectives. Baichuan-Omni-1.5 [29] combines separate visual and acoustic encoders with joint multimodal alignment, OmniVinci [56] strengthens audiovisual fusion through shared latent alignment and explicit temporal encoding, and MiniCPM-o 4.5 [12] organizes perception and speech generation on a common temporal axis to support full-duplex interaction. These systems have substantially advanced modality coverage, streaming response, and cross-modal instruction following, but their notion of unification remains largely functional. Training objectives and evaluations are designed to determine whether heterogeneous signals can be understood and whether coherent responses can be produced across modalities. They seldom require the model to recover latent physical quantities, maintain an explicit account of how object states evolve, or identify the physical event that causally links a visual change with an acoustic observation. Strong omni-modal interaction does not yet amount to a structured model of the physical world.

## 2.2 Physical Data Generation and Supervision

Physical data are distinguished from general multimodal data by the constraints their annotations place on the state or transition underlying an observation. A visual description records what appears in a scene, whereas physical supervision relates that appearance to intrinsic properties, initial conditions, interactions, intermediate responses, or outcomes. Controlled simulation has been the principal source of such supervision because it exposes variables that are dificult to measure in ordinary video. IntPhys [43] evaluates violations of object permanence and continuity, PHYRE [5] tests goal-directed intervention, CLEVRER [57] supports causal and counterfactual reasoning over collisions, and Physion [6] makes prediction depend on latent properties including mass, friction, elasticity, and deformability. Procedural platforms such as Kubric [19] and PhysInOne [67] extend this setting with larger scene collections and denser annotations of geometry, motion, trajectories, and material parameters. Their labels are exact within the simulator, although the corresponding observations remain bounded by the chosen assets, parameter ranges, contact models, and rendering assumptions. Real-world resources ofer complementary evidence. Physics 101 [52] and PhysVid [39] associate controlled interaction videos with measured object properties, while ObjectFolder [16] records visual, acoustic, and tactile responses from real and neural objects. These measurements are physically meaningful but rely on specialized acquisition protocols that limit scale and interaction diversity. Current physical supervision is consequently distributed across separate forms of evidence. Simulator datasets provide latent states without fully realistic observations, real-world datasets provide authentic dynamics with sparse property labels, and multisensory datasets rarely annotate complete state transitions. Object attributes, localized events, intermediate responses, final outcomes, and synchronized audiovisual evidence are still seldom available within the same example, leaving the physical cause of an observed transition only partially specified.

## 2.3 Physics-Oriented Reinforcement Learning

Physics-oriented reinforcement learning has emerged from the broader use of verifiable rewards for reasoning. GRPO [45] estimates relative advantages from multiple responses to the same prompt without a learned critic, while subsequent variants such as DAPO [59] and GSPO [65] improve optimization stability through dynamic sampling, asymmetric clipping, and sequence-level objectives. This paradigm has also been extended to multimodal reasoning, where Visual-RFT [33], VLM-R1 [46], Vision-R1 [24], and R1-VL [61] use automatically verifiable, rule-based, or step-wise rewards to strengthen visual reasoning without relying on a learned critic. Building on these advances, recent work has begun to incorporate physical criteria into group-relative optimization. Cosmos-Reason1 [2] converts physical-common-sense and embodied-reasoning data into multiple-choice problems and optimizes answer and format rewards. HCM-GRPO [23] mines dificult samples for physical-plausibility discrimination, reflecting evidence that group-relative optimization benefits from prompts whose sampled responses exhibit meaningful reward variation [40], while Physics-R1 [55] further incorporates verifiable answer and unit-consistency rewards for visual physics reasoning. Process-oriented studies supplement terminal accuracy with judgments of physical principles, units, reasoning quality, visual attention, or an explicitly constructed physical model [31, 62].

Verifiability in these methods is often obtained by narrowing the task to answer matching, static plausibility classification, or rubric-based assessment of diagrammatic problems. Such rewards can distinguish a correct response from an incorrect one, but they provide limited evidence that the response rests on the correct material property, contact event, or state transition observed in the input. The group-relative signal is also weak when all rollouts from a prompt receive the same reward, which makes prompt selection part of the optimization problem rather than a separate data-preparation choice. Interaction-based policy learning further shows that improvements within a training environment do not necessarily yield physical knowledge that transfers to related settings [8]. Existing strategies have improved answer-level reliability, while perceptually grounded and temporally resolved reward signals for multimodal physical reasoning remain underdeveloped.

## 3 Training Data Construction

Our training-data pipeline contains two components. We first construct physics-aware supervised data to provide explicit object-property and physical-event supervision, and then curate reinforcement-learning data according to the response distribution of the current policy. Figure 3 provides an overview of the modality and task distribution of the final training corpus.

## 3.1 Physics-Aware Data Construction

Generic image and video corpora describe objects and actions but rarely provide numerical properties or cross-modal physical explanations. We therefore construct complementary static and dynamic data: the static branch performs object filtering, profiling, prototype retrieval, and property refinement to characterize material, body type, and intrinsic properties, while the dynamic branch performs event filtering, clip selection, and dense multimodal annotation to capture interactions, state changes, and acoustic evidence. Branch-specific templates then

![](images/effa75ca36cc43bdd53d4715631a7a9b70adfeeaef0323724bc3ff0fb25f2672.jpg)  
Figure 3 Training-data distribution across modalities and task categories.

guide the same commercial large language model to convert both forms of metadata into question–answer pairs, bridging semantic perception and physical reasoning as shown in Figure 4.

## 3.1.1 Static Physical Data Construction

Dataset Filtering. As shown in the first block of Figure 4(a), we begin with object-level image annotations and remove invalid bounding boxes, living subjects, and severely incomplete objects. This step retains non-living objects whose visible extent is suficient for estimating material and physical properties, while preserving the surrounding image as contextual evidence.

Physical Profiling. For every retained object, its image $I _ { i }$ and target bounding box $b _ { i j }$ are provided to a commercial multimodal large model. The model returns a structured profile containing the object category $c _ { i j }$ , material $m _ { i j }$ , body type $\tau _ { i j }$ , and a set of applicable numerical properties:

$$
\begin{array} { r } { \mathcal { M } _ { \mathrm { c o m } } ( I _ { i } , b _ { i j } ) = \left( c _ { i j } , m _ { i j } , \tau _ { i j } , \{ ( k , \hat { p } _ { i j k } , u _ { k } ) \} _ { k \in \mathcal { K } _ { i j } } \right) , } \end{array}\tag{1}
$$

where $\hat { p } _ { i j k }$ and $u _ { k }$ denote the estimated value and canonical unit of property $k .$ . Depending on the object and body type, $\mathcal { K } _ { i j }$ may include mass, density, stifness, friction, restitution, Young’s modulus, Poisson’s ratio, viscosity, surface tension, and yield stress. The values in this initial profile are produced directly by the model from the image and bounding box; the prototype bank is not used at this stage.

Prototype Retrieval. We build a reference bank containing 1,209 object–material prototypes. Each prototype stores normalized category and material keywords, body type, and real-value references or valid intervals compiled from material handbooks and representative product specifications. Given the predicted category and material, we perform keyword matching over the two indices and retrieve

$$
r _ { i j } ^ { \star } = \arg \operatorname* { m a x } _ { r \in { \mathcal { R } } } \left[ \lambda _ { c } J ( { \mathcal { W } } _ { c _ { i j } } , { \mathcal { W } } _ { c _ { r } } ) + \lambda _ { m } J ( { \mathcal { W } } _ { m _ { i j } } , { \mathcal { W } } _ { m _ { r } } ) \right] ,\tag{2}
$$

where $\mathcal { W }$ denotes a normalized keyword set and $J ( \cdot , \cdot )$ is Jaccard overlap. Category keywords identify a functionally similar object prototype, while material keywords provide the corresponding material-level reference. The matched entry supplies a reference value and a valid interval $\left[ l _ { r k } , u _ { r k } \right]$ for each applicable property.

Property Refinement. We first filter out attributes that are irrelevant to the inferred body type, such as viscosity for a rigid chair. Each remaining estimate is then compared with the interval of the matched prototype. An out-of-range value is returned to the same commercial model together with the reference property and interval for one re-estimation:

$$
\begin{array} { r l } & { \tilde { p } _ { i j k } = \mathrm { R e f i n e } _ { \mathcal { M } _ { \mathrm { c o m } } } ( I _ { i } , b _ { i j } , \hat { p } _ { i j k } , [ l _ { r k } , u _ { r k } ] ) , } \\ & { p _ { i j k } ^ { \star } = \left\{ \begin{array} { l l } { \hat { p } _ { i j k } , } & { \hat { p } _ { i j k } \in [ l _ { r k } , u _ { r k } ] , } \\ { \tilde { p } _ { i j k } , } & { \tilde { p } _ { i j k } \in [ l _ { r k } , u _ { r k } ] , } \\ { \varnothing , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{3}
$$

Only one refinement pass is performed. If a required value remains outside the valid interval after receiving the reference, the entire object sample is discarded; otherwise, the accepted and re-estimated values are merged into its final physical profile.

Training Data Construction. The refined object metadata are organized with static-data templates and provided to the commercial large language model, which summarizes them into question–answer pairs for physical-property estimation and property-grounded reasoning. The corresponding image is attached to each generated instruction as its visual input.

## 3.1.2 Dynamic Physical Data Construction

Coarse Filtering. As shown in the first block of Figure 4(b), we construct a hierarchical physical-event library comprising six coarse categories and 26 fine-grained event types. Each event type is associated with a set of retrieval keywords, which are matched against the available video titles, descriptions, and tags to remove videos without an explicit physical event. We further expand the retained keywords into short phrases that combine an object, an interaction, and an observable result.

Clip Selection. Long videos often contain only a short interval relevant to the event key. We divide each candidate video into overlapping temporal clips and use Qwen3-VL-Embedding-8B [27] to encode both the clips and the expanded phrases in a shared embedding space. For clip $\nu _ { i }$ and phrase $q _ { j }$ , their matching score is

$$
s _ { i j } = \frac { E _ { \nu } ( \nu _ { i } ) ^ { \top } E _ { t } ( q _ { j } ) } { \| E _ { \nu } ( \nu _ { i } ) \| _ { 2 } \| E _ { t } ( q _ { j } ) \| _ { 2 } } , \qquad \nu _ { j } ^ { \star } = \arg \operatorname* { m a x } _ { \nu _ { i } } s _ { i j } ,\tag{4}
$$

where $E _ { \nu }$ and $E _ { t }$ are the video and text encoders. The top-matched neighboring clips are merged to preserve the pre-event state, interaction, and visible consequence.

Dense Multimodal Annotations. The selected clip, sampled video frames, and synchronized audio waveform are jointly provided to the commercial multimodal large model. The model first produces separate descriptions of the visual and acoustic streams. The visual description records actions, object states, trajectories, and visible text, while the audio description identifies speech, sound events, and their timestamps. It then combines the two streams with physical priors to describe the underlying interaction, state change, and relevant physical properties. We keep the three descriptions separate so that the physical interpretation can be checked against the original visual and acoustic evidence.

For an event $e$ with visual onset $t _ { e } ^ { \mathrm { v } }$ and acoustic onset $t _ { e } ^ { \mathrm { a } } ,$ , we associate the two observations only when their temporal discrepancy satisfies

$$
\Delta t _ { e } = \left| t _ { e } ^ { \mathrm { v } } - t _ { e } ^ { \mathrm { a } } \right| \leq \delta _ { e } ,\tag{5}
$$

where $\delta _ { e }$ is an event-dependent tolerance. The aligned record therefore preserves the video description, audio description, physical interpretation, and their time anchors instead of collapsing them into a generic video caption.

![](images/b664eacfbec33b9b79ef0660d134683c022acea920d562d64a43fb06fc7e11bd.jpg)  
Figure 4 Physics-aware data construction. (a) The static branch filters images, identifies objects and materials, retrieves object–material prototypes, and refines physical properties through compatibility and range checks before constructing training examples. (b) The dynamic branch retrieves event-centered video clips and aligns visual descriptions, acoustic evidence, and physical analysis into temporally grounded supervision of interactions and state changes. Together, the two branches connect intrinsic object properties with their observable efects during physical events.

Training Data Construction. The aligned video, audio, and physical annotations are organized with dynamic-data templates and provided to the commercial large language model, which summarizes them into question–answer pairs for event description, temporal reasoning, and physical explanation. The source video and audio are attached to the generated instruction as its multimodal input.

## 3.2 Policy-Aware RL Data Curation

Unlike supervised examples, the utility of an RL prompt depends on both the current policy and the optimization rule. GRPO ranks responses sampled for the same prompt through group-normalized rewards (Eq. (17)). If every response in a sampled group receives the same reward, all normalized advantages are zero, so that group provides no reward-based preference for the corresponding update. We therefore curate the RL corpus according to the response distribution of a fixed rollout policy rather than treating every candidate prompt as equally informative. Figure 5 summarizes this procedure across candidate pools spanning text, image, video, and audio inputs.

Group-Wise Rollout Evaluation. Let $\mathcal { D } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ denote a candidate pool, where $x _ { i }$ is a prompt and $y _ { i }$ is its reference answer. For each $x _ { i } .$ , a fixed policy $\pi _ { \mathrm { r o l l } }$ generates � stochastic responses under the same decoding configuration, with � = 4 by default. Each response is scored independently by the evaluator $\mathcal { R } _ { \tau _ { i } }$ associated with its task type $\tau _ { i } \dot { : }$

$$
\begin{array} { r l } & { \hat { y } _ { i } ^ { ( k ) } \sim \pi _ { \mathrm { r o l l } } ( \cdot \mid x _ { i } ) , } \\ & { r _ { i } ^ { ( k ) } = \mathcal { R } _ { \tau _ { i } } \left( x _ { i } , \hat { y } _ { i } ^ { ( k ) } , y _ { i } \right) , \qquad k = 1 , \dots , K . } \end{array}\tag{6}
$$

Keeping the rollout policy and decoding configuration fixed ensures that the reward diversity within each rollout group is induced by the policy’s stochastic response behavior under a consistent sampling process.

Reward-Diversity Selection. To identify prompts that provide informative group-relative learning signals, we retain a

![](images/157147a49532d90523746c4239e9d8d1b7c4ec14a158d891b1d55f3e656883a5.jpg)  
Figure 5 Rollout-based reward-diversity filtering for RL data curation. For each prompt, a fixed policy generates � stochastic responses (� = 4 by default), which are scored by task-specific reward evaluators. A prompt is retained only if its rollout group contains more than one distinct reward value. Uniform-reward groups are discarded; for binary rewards, these correspond to all-correct or all-incorrect responses. The retained corpus provides within-group reward variation for subsequent GRPO training.

prompt only if its sampled responses receive at least two distinct reward values. With ${ \cal S } _ { i } = \{ r _ { i } ^ { ( k ) } \} _ { k = 1 } ^ { K }$ , the selection rule and resulting corpus are

$$
\begin{array} { r } { q _ { i } = \mathbb { I } [ | S _ { i } | > 1 ] , \quad } \\ { \mathcal { D } _ { \mathrm { R L } } = \{ ( x _ { i } , y _ { i } ) \in \mathcal { D } \mid q _ { i } = 1 \} . } \end{array}\tag{7}
$$

For a binary correctness reward, this retains groups containing both correct and incorrect responses and removes both all-correct and all-incorrect groups. For a graded or composite reward, the same rule retains any group with more than one attained score. The criterion thus mirrors the requirement of group-relative optimization: a retained prompt exhibits an observable reward ordering under the rollout policy.

Task-Specific Reward Instantiation. The selection rule is shared across data sources, while the evaluator follows the intended training objective. Mathematical and symbolic answers are checked for equivalence, with normalized string matching as a fallback. Multiple-choice image–text, video, and mixed-modal tasks use exact option matching, whereas free-form answers use their task-specific normalized or graded matching rules; malformed or degenerate generations are rejected.

Candidate data for progressive multimodal RL are scored by final-answer correctness, while the physical-perception data used in the fine-grained RL phase are scored with the answer, intermediate-perception, and format terms defined in Eq. (21). Thus, task-specific evaluators determine response quality, and the shared diversity rule determines whether that quality diference is observable within a rollout group.

## 4 Method

## 4.1 Model Architecture

As illustrated in Figure 6, OmniFysics-Nano-V2 follows a unified perception–reasoning–expression design. Modality interfaces first map heterogeneous observations into a common token space, where a shared causal transformer performs cross-modal reasoning. The generated answer states are then reused to condition speech synthesis through cross-attention. This separation keeps semantic reasoning in a single pathway while allowing the same response to be expressed as either text or speech.

![](images/a59c30ac007905897a7a315c8bce84eeb5eebd7c8e32d4bd7ddaee83baca3e2f.jpg)  
Figure 6 Architecture of OmniFysics-Nano-V2. Text, visual, and acoustic features are packed into a unified sequence for a shared causal backbone; temporally corresponding visual and acoustic groups are interleaved for synchronized video–audio input. For speech output, the generated answer is normalized and retokenized with the speech tokenizer. The resulting text embeddings query the projected answer states through cross-attention, producing a semantic conditioning sequence for speech-token generation and waveform reconstruction.

## 4.1.1 Unified Omni-Modal Sequence

Let $x _ { \mathrm { { t x t } } } , x _ { \mathrm { { v i s } } }$ , and $x _ { \mathrm { a u d } }$ denote text, image or video, and audio inputs. Their representations are

$$
\begin{array} { r l } & { X _ { \mathrm { t x t } } = E _ { \mathrm { t x t } } ( x _ { \mathrm { t x t } } ) , } \\ & { X _ { \mathrm { v i s } } = M _ { \mathrm { v i s } } ( E _ { \mathrm { v i s } } ( x _ { \mathrm { v i s } } ) ) , } \\ & { X _ { \mathrm { a u d } } = P _ { \mathrm { a u d } } ( E _ { \mathrm { a u d } } ( x _ { \mathrm { a u d } } ) ) , } \end{array}\tag{8}
$$

where $E _ { \mathrm { t x t } } , E _ { \mathrm { v i s } }$ , and $E _ { \mathrm { a u d } }$ are the text, visual, and acoustic encoders, respectively; $M _ { \mathrm { v i s } }$ is the visual merger; and $P _ { \mathrm { a u d } }$ projects acoustic features to the shared hidden dimension. Images and videos use the same visual interface, preserving spatial layout and frame order, while audio and speech share the acoustic interface. Modality-specific boundary embeddings explicitly mark the extent of every non-text segment.

The modality-specific representations are inserted at their corresponding positions in the conversational sequence, delimited by modality boundary tokens, and jointly processed by the shared causal transformer $F _ { \theta }$ . Consequently, each response token can condition on textual, visual, and acoustic evidence within a single autoregressive context.

For synchronized video–audio input, the visual and acoustic streams are divided into temporally corresponding feature groups and interleaved as

$$
[ b _ { \mathrm { v i s } } , b _ { \mathrm { a u d } } , X _ { \mathrm { v i s } } ^ { ( 1 ) } , X _ { \mathrm { a u d } } ^ { ( 1 ) } , \ldots , X _ { \mathrm { v i s } } ^ { ( N ) } , X _ { \mathrm { a u d } } ^ { ( N ) } , e _ { \mathrm { v i s } } , e _ { \mathrm { a u d } } ] .\tag{9}
$$

Each group may contain multiple tokens; the superscript indexes a shared temporal interval rather than a one-to-one token correspondence. This ordering places temporally related visual and acoustic evidence in proximity while preserving the internal structure of each modality, making event-level correspondences directly accessible to the causal backbone.

## 4.1.2 Semantic Alignment for Speech Generation

Let $a = \left( a _ { 1 } , \ldots , a _ { T _ { a } } \right)$ be the textual answer generated by the reasoning backbone and $H _ { a } \in \mathbb { R } ^ { T _ { a } \times d }$ its final-layer hidden states. Rather than conditioning speech generation on the rendered answer alone, we retain these answer states as an

additional semantic representation. A trainable MLP $P _ { \phi }$ maps them to the speech-conditioning space, followed by feature normalization:

$$
Z _ { a } = \mathrm { N o r m } \left( P _ { \phi } ( H _ { a } ) \right) .\tag{10}
$$

In parallel, detokenization recovers the response string from the generated text tokens. Text normalization then converts the raw response into a canonical, human-readable and speakable form by removing non-spoken formatting and rewriting symbols or numerals when necessary. The normalized text is subsequently processed by the speech-side tokenizer:

$$
C = \tau _ { \mathrm { s p } } ( N ( \mathrm { D e t o k } _ { \mathrm { t x t } } ( a ) ) ) , \qquad E _ { C } = \mathrm { E m b } _ { \mathrm { s p } } ( C ) ,\tag{11}
$$

where $C = \left( c _ { 1 } , \dots , c _ { T _ { c } } \right)$ . Because the language and speech tokenizers segment the same answer diferently, $T _ { a }$ and $T _ { c }$ need not match and their representations cannot be aligned position by position. We therefore apply cross-attention between the speech-side text embeddings and the projected answer states:

$$
U = E _ { C } + { \mathrm { C r o s s A t t n } } ( E _ { C } , Z _ { a } ) .\tag{12}
$$

The residual connection preserves the speech-side textual representation, while cross-attention adds context from the complete sequence of answer states. The resulting sequence $U$ is combined with the speech start token, control-text embeddings, and task identifier to form the conditioning prefix

$$
\Pi = [ e _ { \mathrm { s o s } } , e _ { \mathrm { c t r l } } , U , e _ { \mathrm { t a s k } } ] .\tag{13}
$$

Conditioned on Π, the speech-token decoder $D _ { \psi }$ predicts a discrete speech-token sequence autoregressively until the end-of-sequence token is reached. After token generation is complete, the codec decoder $G _ { \xi }$ converts the discrete sequence into the output waveform:

$$
\begin{array} { r l } & { s _ { t } \sim p _ { \psi } ( \cdot \mid s _ { < t } , \Pi ) , \quad t = 1 , \dots , T _ { s } , } \\ & { \hat { w } = G _ { \xi } ( s _ { 1 : T _ { s } } ) . } \end{array}\tag{14}
$$

This decomposition assigns semantic transfer to cross-attention, sequence modeling to the speech-token decoder, and waveform reconstruction to the codec decoder. It also allows speech generation to reuse the states that produced the written answer rather than recovering its semantics from text alone.

## 4.1.3 Training Objective

The model is trained with the causal language-modeling objective, with the speech term activated only during the final output-alignment stage:

$$
\mathcal { L } _ { \mathrm { S F T } } = \mathcal { L } _ { \mathrm { t e x t } } + \lambda _ { \mathrm { s p } } \mathcal { L } _ { \mathrm { s p e e c h } } ,\tag{15}
$$

where $\lambda _ { \mathrm { s p } } = 0$ for multimodal supervised fine-tuning and is nonzero for speech alignment. This objective keeps the main model unchanged while the output branch learns its own interface.

## 4.2 Training Strategy

Our training strategy combines supervised fine-tuning (SFT) and reinforcement learning (RL) within a staged curriculum. We first define the optimization objectives shared across stages and then describe how they are assigned to diferent model components and data mixtures.

## 4.2.1 Optimization Objectives

Supervised Fine-Tuning. Given a multimodal context � and target response $a = \left( a _ { 1 } , \ldots , a _ { T _ { a } } \right)$ , SFT minimizes the standard autoregressive next-token loss

$$
\mathcal { L } _ { \mathrm { t e x t } } ( \theta ) = - \sum _ { t = 1 } ^ { T _ { a } } \log p _ { \theta } ( a _ { t } \mid a _ { < t } , x ) .\tag{16}
$$

This objective provides token-level supervision for aligning modality interfaces and learning joint multimodal responses in Stage 1.

![](images/d3e9a0146e1539fbc66a9590a1d51b106316969fa1c9536c4720e6d60fda5df8.jpg)  
Figure 7 Training curriculum of OmniFysics-Nano-V2. Stage 1 comprises Audio Alignment (Stage 1-1), Cross-Modal Alignment (Stage 1-2), and Omni-Modal Joint Training (Stage 1-3). Stage 2 first performs Progressive Multimodal RL with answer-level rewards (Stage 2-1), then applies Fine-Grained RL for Physical Perceptual Reasoning with final-answer, intermediate-perception, and format rewards (Stage 2-2). Stage 3 performs Audio Generation Training while keeping the multimodal reasoner and codec decoder frozen. Flame and snowflake symbols denote trainable and frozen components, respectively.

Group Relative Policy Optimization. For each input �, the behavior policy $\pi _ { \theta _ { \mathrm { o l d } } }$ samples a group of � responses $\begin{array} { r } { y ( x ) = \{ y _ { i } \} _ { i = 1 } ^ { G } } \end{array}$ . After assigning each response a stage-specific reward $r _ { i } = R ( x , y _ { i } )$ , we normalize rewards within the group to obtain

$$
A _ { i } = \frac { r _ { i } - \mu _ { \mathscr { G } } } { \sigma _ { \mathscr { G } } + \delta } , \qquad \mu _ { \mathscr { G } } = \frac { 1 } { G } \sum _ { j = 1 } ^ { G } r _ { j } ,\tag{17}
$$

where $\sigma _ { \mathcal { G } }$ is the group-wise standard deviation and � prevents division by zero. We maximize the clipped sequence-level objective

$$
\mathcal { T } _ { \mathrm { G R P O } } = \mathbb { E } \Bigg [ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \operatorname* { m i n } \Big ( \rho _ { i } A _ { i } , \mathrm { c l i p } ( \rho _ { i } , 1 - \epsilon , 1 + \epsilon _ { \mathrm { h i g h } } ) A _ { i } \Big ) \Bigg ] ,\tag{18}
$$

Here $\rho _ { i }$ is computed over the complete response rather than independently for each token. Group normalization makes the update depend on the relative quality of responses sampled for the same input, consistent with the reward-diversity filtering used to construct the RL corpus.

Speech Generation. For target speech tokens $s _ { 1 : T _ { s } }$ , let $y _ { 1 : T _ { s } } = s _ { 1 : T _ { s } }$ and append $y _ { T _ { s } + 1 } = s _ { \mathrm { e o s } }$ . Conditioned on the prefix Π from Eq. (13), the answer projector and cross-attention parameters � are jointly optimized with the speech-token decoder parameters � through

$$
\mathcal { L } _ { \mathrm { s p e e c h } } ( \psi , \omega ) = - \sum _ { t = 1 } ^ { T _ { s } + 1 } \log p _ { \psi , \omega } ( y _ { t } \mid y _ { < t } , \Pi ) .\tag{19}
$$

This objective jointly optimizes the answer-conditioning module � and the speech-token decoder $\psi$

## 4.2.2 Stage-Wise Training

Figure 7 summarizes our three-stage curriculum. Stage 1 establishes multimodal input alignment and joint understanding through three supervised phases. Stage 2 optimizes the resulting policy through progressive and fine-grained reinforcement learning. Stage 3 adds speech generation after the text policy has been optimized. Every phase is initialized from the preceding checkpoint. This ordering separates interface alignment, policy optimization, and output adaptation, reducing interference between objectives that act on diferent parts of the model.

• Stage 1: Supervised Multimodal Training. All three phases optimize the SFT objective in Eq. (16), while progressively expanding the trainable components and supervision mixture.

Stage 1-1: Audio Alignment. We first use ASR supervision to align the acoustic representation with the shared language space. The acoustic and visual encoders, visual merger, and causal backbone remain frozen; only the audio projector and newly introduced audio-boundary embeddings are optimized. Constraining the update to the audio interface prevents the initial alignment objective from overwriting pretrained linguistic and visual representations.

Stage 1-2: Cross-Modal Alignment. Starting from the audio-aligned checkpoint, we introduce image and video supervision and optimize the visual mergers together with the audio projector and modality-boundary embeddings. Captioning, OCR, ASR, and video-caption tasks provide direct alignment signals, while the modality encoders and causal backbone remain fixed. This phase brings visual and acoustic features into the language model’s representational space before joint reasoning is learned.

Stage 1-3: Omni-Modal Joint Training. After the modality interfaces have been aligned, we unfreeze the causal backbone and optimize it jointly with the visual mergers, audio projector, and boundary embeddings; the modality encoders remain frozen. The training mixture covers image, video, audio, speech, text, embodied, and physics-oriented instructions. At this point, the objective moves beyond interface alignment: the shared transformer learns to combine appearance, temporal change, acoustic evidence, and language context within a single autoregressive response.

• Stage 2: Reinforcement Learning. Supervised likelihood training encourages imitation but does not directly optimize verifiable reasoning outcomes. We therefore apply two consecutive GRPO phases. They share the same policy-update rule but difer in curriculum and reward granularity: Stage 2-1 uses answer-level correctness across increasingly diverse modalities, whereas Stage 2-2 adds explicit feedback for intermediate physical perception and response format.

Stage 2-1: Progressive Multimodal RL. This phase uses a binary final-answer reward. Let $\varepsilon ( y )$ extract the predicted answer from response �, let $a ^ { \star }$ be the reference answer, and let $\boldsymbol { { \mathcal { M } } } _ { \boldsymbol { \tau } ( \boldsymbol { x } ) }$ denote the verifier associated with the task type �(�). We define

$$
R _ { \mathrm { a n s } } ( x , y ) = \mathbb { I } \big [ M _ { \tau ( x ) } \big ( \mathcal { E } ( y ) , a ^ { \star } \big ) = 1 \big ] .\tag{20}
$$

For mathematical and symbolic tasks, $\boldsymbol { { M } _ { \tau } }$ checks symbolic equivalence and falls back to normalized string matching when parsing fails. Multiple-choice tasks require an exact option match, while OCR and free-form multimodal tasks compare normalized answers using their task-specific evaluators. The verifier therefore changes with the answer format, but the optimization signal remains final-answer correctness.

Training proceeds from mathematical and logical reasoning to image and video reasoning, and finally to mixed-modality problems involving visual and acoustic evidence. The first phase establishes reasoning and answer verification in a controlled text setting; subsequent phases introduce spatial, temporal, and cross-modal dependencies. Throughout this curriculum, the GRPO update and answer-level reward principle remain fixed, while the input complexity progressively increases.

Stage 2-2: Fine-Grained RL for Physical Perceptual Reasoning. An answer-only reward cannot distinguish a response that identifies the relevant physical evidence but makes a downstream error from one that never extracts the required evidence. We therefore introduce denser supervision for image–audio and video tasks involving motion, physical properties, materials, and physical phenomena. The reward combines final-answer correctness, intermediate physical perception, and format compliance:

$$
\begin{array} { r } { R = \lambda _ { \mathrm { a n s } } R _ { \mathrm { a n s } } + \lambda _ { \mathrm { m i d } } R _ { \mathrm { m i d } } + \lambda _ { \mathrm { f m t } } R _ { \mathrm { f m t } } , } \\ { R _ { \mathrm { m i d } } = \lambda _ { \mathrm { p r o p } } R _ { \mathrm { p r o p } } + \lambda _ { \mathrm { m a t } } R _ { \mathrm { m a t } } , \qquad } \end{array}\tag{21}
$$

where all weights are nonnegative, $\lambda _ { \mathrm { a n s } } + \lambda _ { \mathrm { m i d } } + \lambda _ { \mathrm { f m t } } = 1$ , and $\lambda _ { \mathrm { p r o p } } + \lambda _ { \mathrm { m a t } } = 1 . ~ R _ { \mathrm { a n s } }$ evaluates the final decision using the task-specific verifier. $R _ { \mathrm { p r o p } }$ evaluates the requested physical property or attribute value, or agreement with an annotated

<table><tr><td>Model</td><td>Size</td><td>Modality</td><td>MMBench-V1.1</td><td>MMStar</td><td>MMMU</td><td>HallusionBench</td><td>AI2D</td></tr><tr><td>Ovis2.5 [35]</td><td>2B</td><td>VL</td><td>79.20</td><td>67.70</td><td>58.70</td><td>58.50</td><td>85.00</td></tr><tr><td>SAIL-VL2 [58]</td><td>2B</td><td>VL</td><td>80.10</td><td>64.00</td><td>49.30</td><td>51.10</td><td>83.10</td></tr><tr><td>Ovis-U1 [49]</td><td>3B</td><td>VL</td><td>77.90</td><td>61.30</td><td>50.60</td><td>55.80</td><td>85.60</td></tr><tr><td>Qwen3.5-4B [41]</td><td>4B</td><td>VL</td><td>84.30</td><td>69.70</td><td>77.60</td><td>59.90</td><td>84.60</td></tr><tr><td>Qwen3-VL-4B-Instruct [3]</td><td>4B</td><td>VL</td><td>82.20</td><td>63.70</td><td>56.40</td><td>55.20</td><td>78.20</td></tr><tr><td>Qwen2.5-Omni-3B [53]</td><td>3B</td><td>Omni</td><td>77.80</td><td>55.70</td><td>53.10</td><td>40.21</td><td>79.50</td></tr><tr><td>Qwen2.5-Omni-7B [53]</td><td>7B</td><td>Omni</td><td>81.80</td><td>64.00</td><td>59.20</td><td>44.90</td><td>83.20</td></tr><tr><td>OmniVinci [56]</td><td>7B</td><td>Omni</td><td>88.50</td><td>64.50</td><td>49.70</td><td>33.00</td><td>91.50</td></tr><tr><td>OmniFysics-Nano-V2</td><td>4B</td><td>Omni</td><td>85.53</td><td>75.27</td><td>62.11</td><td>62.49</td><td>88.60</td></tr></table>

Table 1 General multimodal understanding on MMBench-V1.1, MMStar, MMMU, HallusionBench, and AI2D. Models are grouped as vision-language (VL) or omni-modal (Omni); Size is the reported parameter count in billions (B). Higher scores are better. Bold and underlined values denote the best and second-best results, respectively, in each column; the shaded row is our model.

numerical interval for quantitative predictions, while $R _ { \mathrm { m a t } }$ evaluates material recognition and associated attributes. $R _ { \mathrm { f m t } }$   
checks structural compliance and the presence of required fields; it does not substitute for semantic correctness.

For image–audio tasks, the intermediate reward evaluates whether the response identifies both the queried property and the material evidence needed to support it. For video tasks, it evaluates physical attributes inferred from temporal observations. These intermediate terms expose which perceptual step failed and provide a more localized learning signal than final-answer correctness alone.

• Stage 3: Audio Generation Training. After reinforcement learning, we freeze the modality encoders, shared causal backbone, and codec decoder so that speech adaptation cannot alter the learned multimodal policy. Only the answer projector and cross-attention parameters � and the speech-token decoder parameters $\psi$ are jointly updated using the speech-generation objective in Eq. (19). Because this objective is applied only after text-policy optimization, it adapts the speech-output path while preserving both the learned multimodal reasoner and waveform decoder.

## 4.2.3 Joint Optimization with Diferentiable Physics Engine

We further couple OmniFysics-Nano-V2 with our in-house diferentiable physics engine to calibrate the object-level physical properties inferred from visual observations. The model first estimates mass, density, static and kinetic friction coeficients, and restitution for each visible object. Together with the perceived scene geometry and initial object states, these predictions are used to parameterize the corresponding instances in the physics engine. The engine then performs forward simulation under the same initial conditions and action sequence as the reference interaction, producing the simulated evolution of object positions, motion trajectories, and velocities.

The simulated dynamics are compared with the reference observations in terms of terminal position, complete trajectory, and velocity. Because the physical engine is diferentiable, the resulting discrepancy can be propagated backward through the sequence of physical transitions to the predicted physical parameters. These parameters are iteratively corrected while remaining within physically valid ranges, after which the updated simulation is evaluated again. This recurring process forms a perception–simulation–calibration loop that uses observed motion to refine the model’s initial estimates, converting visual physical priors into physically consistent, simulation-ready parameters for downstream policy optimization.

## 5 Experiments and Results

## 5.1 Experimental Setup

Model and Training. We initialize the shared backbone and visual interface from Qwen3.5-4B [41], the acoustic encoder from Whisper Medium [42], and the speech-token and codec decoders from CosyVoice3 [13]. These pretrained components retain their native tokenization and input-processing schemes, while the newly introduced modality projectors and answer-conditioning modules are optimized according to the stage-wise training strategy. Supervised training is implemented with VeOmn [36], whereas reinforcement learning is conducted with MS-SWIFT [64]. All training stages are performed on 64 NVIDIA H100 GPUs using bfloat16 precision.

¡ Audio Understanding Benchmarks  
![](images/25edcc0e2a36e348a3e64c0960482caa6238a27e54de2d1b6bd2c911e8f7f6fb.jpg)

Figure 8 Comparison of Audio Understanding Performance on MMAU and MMAR. Purple hatched bars denote our model.
<table><tr><td>Model</td><td>Size</td><td>OmniBench</td><td>WorldSense</td><td>Daily-Omni</td><td>FysicsWorld</td><td>Video-MME</td></tr><tr><td>Unified-IO-2 L [34]</td><td>1B</td><td>27.06</td><td>23.30</td><td>27.40</td><td>45.34</td><td>45.20</td></tr><tr><td>Unified-IO-2 XL [34]</td><td>3B</td><td>38.00</td><td>24.70</td><td>28.30</td><td>47.62</td><td>46.80</td></tr><tr><td>Qwen2.5-Omni-3B [53]</td><td>3B</td><td>45.18</td><td>44.45</td><td>51.35</td><td>51.49</td><td>62.00</td></tr><tr><td>Qwen2.5-Omni-7B [53]</td><td>7B</td><td>56.13</td><td>45.40</td><td>53.42</td><td>58.58</td><td>64.30</td></tr><tr><td>OmniVinci [56]</td><td>7B</td><td>46.47</td><td>48.23</td><td>66.50</td><td>55.52</td><td>68.20</td></tr><tr><td>Unified-IO-2 XXL [34]</td><td>7B</td><td>33.98</td><td>25.90</td><td>28.24</td><td>47.62</td><td>54.40</td></tr><tr><td>OmniFysics-Nano-V2</td><td>4B</td><td>60.95</td><td>60.32</td><td>85.24</td><td>60.87</td><td>80.89</td></tr></table>

Table 2 Omni-modal and video understanding on OmniBench, WorldSense, Daily-Omni, FysicsWorld, and Video-MME. Size is the reported parameter count in billions (B). Higher scores are better. Bold and underlined values denote the best and second-best results, respectively, in each column; the shaded row is our model.

Evaluation. We evaluate five capability groups: general multimodal understanding, audio understanding, omni-modal and video understanding, physical-world understanding and reasoning, and mathematical and physical reasoning, using each benchmark’s native metric. We also conduct a human evaluation of generated speech.

## 5.2 Main Results

OmniFysics-Nano-V2 exhibits its clearest gains on temporal, cross-modal, and physics-oriented evaluation. It ranks first among the listed systems on the representative benchmarks in Figure 2(b). The largest margins appear on Daily-Omni and PhysUniBench, followed by MMStar and MMAR; the gains on OmniBench, PhysBench, and PhyX are smaller but remain positive. The complete comparisons in Tables 1–4 and Figure 8 further show that these targeted strengths are achieved while maintaining competitive performance on general-purpose visual-language benchmarks.

## 5.2.1 General Multimodal Understanding

OmniFysics-Nano-V2 is strongest among the listed omni-modal models on MMStar [9], MMMU [60], and Hal lusionBench [20], with scores of 75.27, 62.11, and 62.49, respectively (Table 1). Relative to Qwen2.5-Omni-7B, the corresponding gains are 11.27, 2.91, and 17.59 points, together with a 3.73-point gain on MMBench-V1.1 [32].

<table><tr><td rowspan="2">Model</td><td rowspan="2">Size</td><td colspan="3">FysicsEval</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2">PhysBench PAI-Bench QuantiPhy PhysUniBench</td></tr><tr><td></td><td>Prediction Reasoning Understanding</td><td></td></tr><tr><td>Ovis2.5 [35]</td><td>2B</td><td>20.40</td><td>2.46</td><td>89.50</td><td>43.80</td><td>42.70</td><td>29.30</td><td>37.00</td></tr><tr><td>SAIL-VL2 [58]</td><td>2B</td><td>21.90</td><td>2.58</td><td>84.70</td><td>44.40</td><td>48.10</td><td>25.60</td><td>37.30</td></tr><tr><td>Ovis-U1 [49]</td><td>3B</td><td>6.60</td><td>2.22</td><td>81.90</td><td>26.40</td><td>21.20</td><td>28.60</td><td>38.10</td></tr><tr><td>Qwen2.5-Omni-3B [53]</td><td>3B</td><td>18.10</td><td>1.71</td><td>87.50</td><td>35.50</td><td>50.60</td><td>28.20</td><td>33.40</td></tr><tr><td>Qwen3.5-4B [41]</td><td>4B</td><td>24.00</td><td>2.33</td><td>92.70</td><td>48.00</td><td>30.80</td><td>27.90</td><td>48.00</td></tr><tr><td>Qwen3-VL-4B-Instruct [3]</td><td>4B</td><td>24.80</td><td>2.24</td><td>87.70</td><td>41.20</td><td>50.60</td><td>29.00</td><td>41.90</td></tr><tr><td>Qwen2.5-Omni-7B [53]</td><td>7B</td><td>27.90</td><td>2.13</td><td>86.30</td><td>46.30</td><td>53.00</td><td>34.40</td><td>46.40</td></tr><tr><td>OmniVinci [56]</td><td>7B</td><td>14.30</td><td>2.07</td><td>88.30</td><td>45.80</td><td>52.20</td><td>21.90</td><td>41.86</td></tr><tr><td>OmniFysics-Nano-V2</td><td>4B</td><td>45.09</td><td>3.26</td><td>98.27</td><td>50.07</td><td>54.61</td><td>40.82</td><td>59.42</td></tr></table>

Table 3 Physical-world understanding and reasoning for vision-language and omni-modal models. FysicsEval is decomposed into prediction, reasoning, and understanding, and is reported alongside PhysBench, PAI-Bench, QuantiPhy, and PhysUniBench. Size is the reported parameter count in billions (B). Higher scores are better. Bold and underlined values denote the best and second-best results, respectively, in each column; the shaded row is our model.

![](images/399f47547c7907bcab1e45f14a95518e12f34abc57ea050e46b7713485e69e85.jpg)  
Figure 9 Pairwise human preference (%) for generated speech from OmniFysics-Nano-V2 and native CosyVoice3, measured by naturalness, clarity, and semantic faithfulness.

OmniVinci remains strongest on MMBench-V1.1 and AI2D, whereas the Qwen3.5-4B reference leads on MMMU. The resulting profile highlights a clear advantage on several omni-modal general-understanding tasks while preserving competitive performance across the broader visual-language suite.

## 5.2.2 Audio Understanding

Audio-input evaluation likewise highlights the model’s parameter eficiency. OmniFysics-Nano-V2 reaches 76.80 on MMAU [44] and 66.10 on MMAR [37], exceeding Qwen2.5-Omni-7B by 5.30 and 9.40 points and Kimi-Audio by 8.60 and 27.31 points, respectively (Figure 8). It therefore provides the strongest audio-understanding results among the listed systems despite having fewer parameters than the 7B baselines, suggesting that the proposed physical-perception training transfers beyond visual inputs.

## 5.2.3 Omni-Modal and Video Understanding

The largest improvements emerge on tasks that require temporal or cross-modal evidence integration. OmniFysics-Nano-V2 ranks first among the listed models on all five benchmarks in Table 2, surpassing Qwen2.5-Omni-7B on OmniBench [30], WorldSense [22], Daily-Omni [68], FysicsWorld [25], and Video-MME [14] by 4.82, 14.92, 31.82, 2.29, and 16.59 points, respectively. The particularly large gains on Daily-Omni, Video-MME, and WorldSense align with the visual–acoustic integration targeted by the model.

## 5.2.4 Physics Understanding and Reasoning

Physics-oriented evaluation provides the most direct test of the proposed supervision, and the advantage is consistent across task types. OmniFysics-Nano-V2 obtains 45.09, 3.26, and 98.27 on FysicsEval [21] Prediction, Reasoning, and Understanding, improving over Qwen2.5-Omni-7B by 17.19, 1.13, and 11.97 points (Table 3). It also leads all listed models on PhysBench [11], PAI-Bench [66], QuantiPhy [28], and PhysUniBench [50]; relative to the same 7B baseline, the gains are 3.77, 1.61, 6.42, and 13.02 points. These gains span physical properties, perception, and grounded reasoning, indicating that the proposed supervision benefits multiple forms of physical-world evaluation.

## 5.2.5 Mathematical and Physical Reasoning

Beyond perception-focused benchmarks, OmniFysics-Nano-V2 retains an advantage on reasoning-intensive tasks. It reaches 13.33 on AIME25 [63], 28.13 on TheoremQA [10], and 48.63 on PhyX [47] (Table 4). Relative to Qwen2.5- Omni-7B, the gains on TheoremQA and PhyX are 2.63 and 1.96 points, while the AIME25 score increases from 10.00 to 13.33. The strongest improvements therefore occur on the theorem-level and physically grounded tasks that align with the proposed training and evaluation design.

<table><tr><td>Model</td><td>Size</td><td>AIME25</td><td>AIME26</td><td>TheoremQA PhyX</td><td></td></tr><tr><td>Qwen2.5-Omni-3B[53]</td><td>3B</td><td></td><td>3.33</td><td>20.13</td><td>32.33</td></tr><tr><td>Qwen2.5-Omni-7B[53]</td><td>7B</td><td>10.00</td><td>10.00</td><td>25.50</td><td>46.67</td></tr><tr><td>Baichuan-Omni-1.5 [29]</td><td>7B</td><td>一</td><td></td><td>14.13</td><td>37.40</td></tr><tr><td>OmniVinci[56]</td><td>7B</td><td></td><td></td><td>12.00</td><td>35.23</td></tr><tr><td>MiniCPM-o-4_5[12]</td><td>9B</td><td>3.33</td><td>3.33</td><td>23.38</td><td>32.17</td></tr><tr><td>OmniFysics-Nano-V2</td><td>4B</td><td>13.33</td><td>3.33</td><td>28.13</td><td>48.63</td></tr></table>

Table 4 Mathematical and physically grounded reasoning results on AIME25, AIME26, TheoremQA, and PhyX. Model size is reported in billions of parameters (B), and higher scores indicate better performance. A “-” indicates that the model did not produce a valid response on the corresponding benchmark; hence, no score is reported. Bold and underlined values denote the best and second-best distinct results, respectively, with ties receiving the same formatting. The shaded row highlights our model.

## 5.2.6 Speech Output Quality

The advantage extends from input understanding to generated speech. In the paired human evaluation shown in Figure 9, OmniFysics-Nano-V2 is preferred to native CosyVoice3 on all three criteria, with margins of 32.8 percentage points for naturalness, 17.4 for clarity, and 42.4 for semantic faithfulness. The largest margin occurs in semantic faithfulness, showing that the answer-conditioned speech path preserves the content of the multimodal response while also improving perceived naturalness and clarity.

## 5.3 Ablation Study

## 5.3.1 Static and Dynamic Physical Supervision

As shown in Table 5, we compare Full SFT with variants that remove Static Physical Data, Dynamic Physical Data, or both. All variants share the same initialization, model, optimizer, and epoch-level training recipe, and are evaluated before policy optimization.

The two supervision branches make complementary contributions. Full SFT is the strongest variant on all nine reported benchmarks, reaching 46.72 on PhysBench, 38.72 on QuantiPhy, 56.67 on PhysUniBench, 58.45 on FysicsWorld, 52.54 on OmniBench, 82.58 on Daily-Omni, 59.80 on MMAR, 82.40 on MMBench-V1.1, and 72.82 on MMStar; relative to removing both branches, these results improve by 0.74, 8.52, 7.47, 7.60, 2.78, 10.13, 2.18, 0.49, and 0.61 points, respectively. Static supervision has its clearest efect on property-oriented evaluation: removing it lowers QuantiPhy by 7.47 points and also reduces PhysUniBench, FysicsWorld, OmniBench, and Daily-Omni by 0.88, 0.84, 1.26, and 0.97 points, respectively. Dynamic supervision is more important for temporal and cross-modal behavior, as removing it lowers FysicsWorld by 7.03 points, OmniBench by 2.23 points, and Daily-Omni by 9.40 points. It is also the primary contributor to MMAR: removing dynamic supervision lowers the score by 2.30 points, whereas removing static supervision lowers it by 0.70 points. Relative to the variant without either branch, dynamic supervision alone improves

<table><tr><td rowspan="2">Training Variant</td><td rowspan="2">Static Dynamic</td><td rowspan="2"></td><td colspan="3">Physics Und.</td><td colspan="2">General Multimodal Und.</td></tr><tr><td>PhysBench</td><td>QuantiPhy</td><td>PhysUniBench</td><td>MMBench-V1.1</td><td>MMStar</td></tr><tr><td>w/o Static &amp; Dynamic Physical Data</td><td></td><td></td><td>45.98</td><td>30.20</td><td>49.20</td><td>81.91</td><td>72.21</td></tr><tr><td>w/o Static Physical Data</td><td></td><td>√</td><td>46.64</td><td>31.25</td><td>55.79</td><td>81.90</td><td>72.53</td></tr><tr><td>w/o Dynamic Physical Data</td><td>√</td><td></td><td>46.53</td><td>37.10</td><td>55.53</td><td>81.87</td><td>72.45</td></tr><tr><td>Full SFT</td><td>√</td><td>√</td><td>46.72</td><td>38.72</td><td>56.67</td><td>82.40</td><td>72.82</td></tr></table>

<table><tr><td rowspan="2">Training Variant</td><td rowspan="2">Static Dynamic</td><td rowspan="2"></td><td colspan="3">Omni-Modal Und.</td><td>Audio Und.</td></tr><tr><td>FysicsWorld</td><td>OmniBench</td><td>Daily-Omni</td><td>MMAR</td></tr><tr><td>w/o Static &amp; Dynamic Physical Data</td><td></td><td></td><td>50.85</td><td>49.76</td><td>72.45</td><td>57.62</td></tr><tr><td>w/o Static Physical Data</td><td></td><td>√</td><td>57.61</td><td>51.28</td><td>81.61</td><td>59.10</td></tr><tr><td>w/o Dynamic Physical Data</td><td>√</td><td></td><td>51.42</td><td>50.31</td><td>73.18</td><td>57.50</td></tr><tr><td>Full SFT</td><td>√</td><td>√</td><td>58.45</td><td>52.54</td><td>82.58</td><td>59.80</td></tr></table>

Table 5 Ablation of static and dynamic physical data. Checkmarks indicate the included data branch. All variants use the same initialization and training recipe and are evaluated after SFT and before RL. Bold and underlined values denote the best and second-best results, respectively.
<table><tr><td>Model</td><td>Input</td><td>OmniBench</td><td>WorldSense</td><td>Daily-Omni</td><td>FysicsWorld</td><td>Video-MME</td></tr><tr><td rowspan="3">Qwen2.5-Omni-3B [53]</td><td>A</td><td>33.01</td><td>40.05</td><td>49.12</td><td>27.66</td><td>49.85</td></tr><tr><td>V</td><td>34.15</td><td>33.76</td><td>39.79</td><td>23.57</td><td>49.67</td></tr><tr><td>A+V</td><td>45.18</td><td>44.45</td><td>51.35</td><td>51.49</td><td>62.00</td></tr><tr><td rowspan="3">Qwen2.5-Omni-7B [53]</td><td>A</td><td>32.49</td><td>41.95</td><td>50.14</td><td>29.71</td><td>52.37</td></tr><tr><td>V</td><td>36.51</td><td>34.21</td><td>41.88</td><td>24.38</td><td>53.52</td></tr><tr><td>A+V</td><td>56.13</td><td>45.40</td><td>53.42</td><td>58.58</td><td>64.30</td></tr><tr><td rowspan="3">OmniVinci [56]</td><td>A</td><td>34.94</td><td>43.08</td><td>51.61</td><td>28.80</td><td>57.26</td></tr><tr><td>V</td><td>37.39</td><td>41.80</td><td>52.79</td><td>26.33</td><td>62.85</td></tr><tr><td>A+V</td><td>46.47</td><td>48.23</td><td>66.50</td><td>55.52</td><td>68.20</td></tr><tr><td rowspan="3">OmniFysics-Nano-V2</td><td>A</td><td>45.45</td><td>42.55</td><td>54.34</td><td>27.37</td><td>54.78</td></tr><tr><td>V</td><td>43.61</td><td>46.09</td><td>55.21 85.24</td><td>20.45</td><td>63.30</td></tr><tr><td>A+V</td><td>60.95</td><td>60.32</td><td></td><td>60.87</td><td>80.89</td></tr></table>

Table 6 Missing-modality evaluation on omni-modal benchmarks. A, V, and A+V retain the audio stream, visual stream, and both streams, respectively; the textual question is unchanged. Higher scores are better. Rows for our model are shaded, and bold values denote its full-input results.

MMAR by 1.48 points, while static supervision alone changes it by −0.12 points; combining both branches yields the best score of 59.80. This result indicates that static supervision does not benefit MMAR in isolation but provides a complementary gain when paired with dynamic supervision, consistent with the combined model’s strongest overall physical and omni-modal profile.

## 5.3.2 Joint Audio–Visual Understanding

Understanding the physical world requires audio and visual information to be perceived and interpreted jointly; either modality alone provides only a partial account of real-world events. As shown in Table 6, joint audio–visual input outperforms both single-modality settings for every model and benchmark. For OmniFysics-Nano-V2, retaining both streams improves over audio-only input by 15.50–33.50 points and over vision-only input by 14.23–40.42 points across the five benchmarks. The stronger isolated modality also varies by task: audio is more informative on OmniBench and FysicsWorld, whereas vision is stronger on WorldSense, Daily-Omni, and Video-MME. This task-dependent pattern reflects the distinction raised in the Introduction: vision captures objects, geometry, and motion, while audio provides event timing, contact, and material-response cues. These observations must be understood together to connect what happens, when it happens, and the physical cause behind it. The consistent degradation after removing either stream therefore confirms that simultaneous audio–visual understanding is essential for understanding the physical world.

![](images/e8a6c1a77a9969a40b723375724d76a53b675024b5815e27cab053147c66841e.jpg)

Figure 10 Successive checkpoints across ten understanding and reasoning benchmarks. SFT denotes the Stage 1-3 checkpoint; Progressive RL adds Stage 2-1 answer-correctness optimization; Fine-Grained RL further adds Stage 2-2 physical-perception and format rewards. Each row continues training from the preceding row. Higher scores are better.
<table><tr><td>Training Data Type</td><td>Reduction Rate</td><td>Saved GPU-Hours</td></tr><tr><td>Mathematical Reasoning</td><td>73.99%</td><td>1,239</td></tr><tr><td>Image-Text Perception</td><td>44.15%</td><td>416.3</td></tr><tr><td>Video Perception</td><td>69.79%</td><td>530.2</td></tr><tr><td>Multimodal Perception</td><td>57.69%</td><td>502.2</td></tr></table>

Table 7 Reward-diversity filtering across four RL data categories. Reduction rate is the fraction of candidate samples removed, and saved GPU-hours is the estimated avoided training compute. The estimates total 2,687.7 GPU-hours.

## 5.3.3 Progressive Reinforcement Learning

Figure 10 reports the successive checkpoints obtained after SFT, Progressive Multimodal RL, and Fine-Grained RL. Each row continues from the preceding checkpoint, allowing the two policy-optimization stages to be examined separately. Relative to SFT, the final checkpoint improves nine of the ten benchmarks. The gains are largest on PhyX (+17.80 points), OmniBench (+8.41), and HallusionBench (+7.79), with additional improvements on MMBench-V1.1, MMAU, MMAR, PhysBench, PAI-Bench, and TheoremQA. WorldSense changes by only -1.06 points, indicating that the final policy retains its overall level of temporal understanding while improving the other reported capabilities.

The two stages exhibit distinct efects. Progressive Multimodal RL raises TheoremQA and PhyX by 4.88 and 16.94 points over SFT, while the intermediate checkpoint is lower on OmniBench and PhysBench. Fine-Grained RL subsequently increases OmniBench, WorldSense, PhysBench, and PAI-Bench by 21.09, 1.75, 13.08, and 7.99 points relative to the intermediate checkpoint, and further improves HallusionBench and MMAR by 7.40 and 5.00 points. TheoremQA decreases by only 0.25 points in this stage and remains 4.63 points above SFT. These results support a staged optimization strategy in which answer-level policy improvement is followed by intermediate physical-perception feedback to recover cross-modal performance and strengthen physical reasoning.

## 5.3.4 Reward-Diversity Filtering

Reward-diversity filtering substantially reduces the RL corpus while retaining prompts that provide a non-degenerate group-relative signal. For each candidate prompt, four stochastic responses are sampled and scored; a prompt is retained only when its rollout group contains more than one attained reward value. Figure 5 illustrates this criterion, and Table 7 reports its category-level efect. The filter removes 73.99% of Mathematical Reasoning, 44.15% of Image–Text Perception, 69.79% of Video Perception, and 57.69% of Multimodal Perception samples. These reductions correspond to estimated savings of 1,239, 416.3, 530.2, and 502.2 GPU-hours, respectively, or 2,687.7 GPU-hours in total. The results show that the filtering rule removes groups that cannot provide a relative reward ordering and concentrates policy optimization on prompts for which the sampled responses expose an observable quality diference.

![](images/4d8771dc0fe47201403eeaaadd0d6f00e469a57b2773d26135f9202ec69fcb5a.jpg)  
Figure 11 Qualitative case studies of physical property prediction and multimodal physical understanding.

## 5.4 Case Study

To illustrate the behavior underlying the aggregate results, Figure 11 compares a static object–property query with a bowling clip accompanied by its soundtrack. In Figure 11(a), both models identify the laptop, mug, and table, whereas OmniFysics-Nano-V2 additionally provides object-level boxes and a structured profile of mass, density, stifness, and Young’s modulus. These values are prior-informed estimates rather than measurements; the example therefore demonstrates grounded output structure, while quantitative reliability is evaluated by the physics benchmarks. In Figure 11(b), both models recover the visible bowling sequence, but OmniFysics-Nano-V2 also associates vocalizations and pin-crash sounds with the interaction stages and explains the outcome using kinetic-energy transfer, friction, sliding, rotation, and mass diferences. This audio–visual evidence chain is consistent with Table 6, where A+V outperforms either single modality on every reported benchmark.

## 6 Application in Embodied Scenarios

Beyond ofline benchmark evaluation, we further integrate OmniFysics-Nano-V2 into an embodied manipulation pipeline that connects scene perception, physics-aware property estimation, and engine-in-the-loop policy optimization. Figure 12 shows the progress. The robot first perceives the objects and their spatial configuration in the workspace, while its wrist-mounted RGB camera provides a close-range view from the manipulation perspective. Taking only the wrist-camera RGB image as input, OmniFysics-Nano-V2 predicts object-level physical properties for the visible objects, including mass, density, static and kinetic friction coeficients, and restitution. These predictions convert visual appearance into structured physical priors and, together with the perceived object configuration, are used to parameterize the corresponding object instances in the physics engine. This process produces a task-specific simulation environment in which object motion and contact evolve according to the predicted dynamics and contact parameters.

![](images/70cbb38235375fff28dabfcbd0d1ebd09a72d64a1faa4a3f9b04651a677f15f5.jpg)  
Figure 12 The proposed OmniFysics-Nano-V2 is integrated with our in-house diferentiable physics engine, enabling continual self-evolution of the embodied agent.

OmniFysics-Nano-V2 therefore does not directly generate robot actions, but instead serves as an interface between visual scene perception and simulation-ready physical states. Within the parameterized environment, the policy first explores candidate interactions and then performs autonomous rollouts to evaluate diferent actions under the estimated object properties and contact conditions. Each rollout is filtered according to task success: successful trajectories are retained as valid training samples, whereas failed trajectories are excluded from the current training set. The filtered successful trajectories are subsequently used to train and update the policy, which then returns to the simulation environment for another round of exploration and autonomous rollout. This recurring cycle of physics-engine simulation, policy exploration, autonomous rollout, success filtering, and policy training constitutes an engine-in-the-loop policy self-evolution process. By providing object-specific physical conditions for policy optimization, this pipeline reduces reliance on repeated trial and error with the physical robot and turns the model’s physical understanding into actionable support for contact-rich manipulation and scalable embodied-data construction.

## 7 Conclusion

We presented OmniFysics-Nano-V2, a compact omni-modal model for physical-world understanding that unifies image, video, audio, speech, and text within a shared reasoning framework and supports answer-conditioned speech generation. The proposed approach combines static object–property supervision, dynamic audio–visual event supervision, reward-diversity filtering, and progressive policy optimization. Experiments show that these components improve complementary aspects of physical intelligence: static supervision supports property-oriented reasoning, dynamic supervision strengthens temporal and cross-modal understanding, and joint audio–visual perception provides more complete evidence for physical interactions. The resulting model achieves strong performance across omni-modal, video, physical-world, and reasoning benchmarks while retaining general multimodal competence, demonstrating the value of explicitly supervising the evidence chain from physical observations to grounded answers. Beyond ofline evaluation, we further explore coupling OmniFysics-Nano-V2 with a diferentiable physics engine, where object-level property estimates parameterize simulation and are refined according to discrepancies between simulated and observed trajectories. This model–engine interaction provides a framework for connecting omni-modal physical perception with physically grounded state prediction and embodied policy optimization. By equipping AI systems with both omni-modal and physical-world perception capabilities, we believe that OmniFysics-Nano-V2 has the potential to serve as a cornerstone of next-generation Physical AI, enabling agents to perceive, simulate, reason, and reliably interact with the real world.

## References

[1] Xiang An, Yin Xie, Feilong Tang, Yunyao Yan, Huajie Tan, Didi Zhu, Changrui Chen, Xiuwei Zhao, Bin Qin, Kaicheng Yang, et al. Llava-onevision-2: Towards next-generation perceptual intelligence. arXiv preprint arXiv:2605.25979, 2026.

[2] Alisson Azzolini, Junjie Bai, Hannah Brandon, Jiaxin Cao, Prithvijit Chattopadhyay, Huayu Chen, Jinju Chu, Yin Cui, Jenna Diamond, Yifan Ding, et al. Cosmos-reason1: From physical common sense to embodied reasoning. arXiv preprint arXiv:2503.15558, 2025.

[3] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Wenbin Ge, Chunjiang Ge, Zhifang Guo, Qidong Huang, Jie Huang, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

[4] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

[5] Anton Bakhtin, Laurens van der Maaten, Justin Johnson, Laura Gustafson, and Ross Girshick. Phyre: A new benchmark for physical reasoning. In Advances in Neural Information Processing Systems, volume 32, pages 5082–5093, 2019.

[6] Daniel M. Bear, Elias Wang, Damian Mrowca, Felix J. Binder, Hsiao-Yu Fish Tung, R. T. Pramod, Cameron Holdaway, Sirui Tao, Kevin Smith, Fan-Yun Sun, Li Fei-Fei, Nancy Kanwisher, Joshua B. Tenenbaum, Daniel L. K. Yamins, and Judith E. Fan. Physion: Evaluating physical prediction from vision in humans and machines. arXiv preprint arXiv:2106.08261, 2022.

[7] Ali Furkan Biten, Ruben Tito, Andr\` es Mafla, Lluis Gomez, Mar´ c¸al Rusinol, C.V. Jawahar, Ernest Valveny, and Dimosthenis˜ Karatzas. Scene text visual question answering. In 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pages 4290–4300, 2019.

[8] Luca M Schulze Buschof, Konstantinos Voudouris, Can Demircan, and Eric Schulz. Can vision language models learn intuitive physics from interaction? arXiv preprint arXiv:2602.06033, 2026.

[9] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin and Feng Zhao. Are we on the right way for evaluating large vision-language models? In Advances in Neural Information Processing Systems, volume 37, pages 27056–27087, 2024.

[10] Wenhu Chen, Ming Yin, Max Ku, Pan Lu, Yixin Wan, Xueguang Ma, Jianyu Xu, Xinyi Wang, and Tony Xia. Theoremqa: A theorem-driven question answering dataset. In Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7889–7901, 2023.

[11] Wei Chow, Jiageng Mao, Boyi Li, Daniel Seita, Vitor Guizilini, and Yue Wang. Physbench: Benchmarking and enhancing vision-language models for physical world understanding. arXiv preprint arXiv:2501.16411, 2025.

[12] Junbo Cui, Bokai Xu, Chongyi Wang, Tianyu Yu, Weiyue Sun, Yingjing Xu, Tianran Wang, Zhihui He, Wenshuo Ma, Tianchi Cai, Jiancheng Gui, Luoyuan Zhang, Xian Sun, Fuwei Huang, Moye Chen, Zhuo Lin, et al. Minicpm-o 4.5: Towards real-time full-duplex omni-modal interaction. arXiv preprint arXiv:2604.27393, 2026.

[13] Zhihao Du, Changfeng Gao, Yuxuan Wang, Fan Yu, Tianyu Zhao, Hao Wang, Xiang Lv, Hui Wang, Chongjia Ni, Xian Shi, Keyu An, Guanrou Yang, Yabin Li, Yanni Chen, Zhifu Gao, Qian Chen, Yue Gu, Mengzhe Chen, Yafeng Chen, Shiliang Zhang, Wen Wang, and Jieping Ye. Cosyvoice 3: Towards in-the-wild speech generation via scaling-up and post-training. arXiv preprint arXiv:2505.17589, 2025.

[14] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, Peixian Chen, Yanwei Li, Shaohui Lin, Sirui Zhao, Ke Li, Tong Xu, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24108–24118, 2025.

[15] Chaoyou Fu, Haozhi Yuan, Yuhao Dong, Yi-Fan Zhang, Yunhang Shen, Xiaoxing Hu, Xueying Li, Jinsen Su, Chengwu Long, Xiaoyao Xie, Yongkang Xie, Xiawu Zheng, Xue Yang, Haoyu Cao, Yunsheng Wu, Ziwei Liu, Xing Sun, Caifeng Shan, and Ran He. Video-mme-v2: Towards the next stage in benchmarks for comprehensive video understanding. arXiv preprint arXiv:2604.05015, 2026.

[16] Ruohan Gao, Yiming Dou, Hao Li, Tanmay Agarwal, Jeannette Bohg, Yunzhu Li, Li Fei-Fei, and Jiajun Wu. The object folder benchmark: Multisensory learning with neural and real objects. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 17276–17286. IEEE, 2023.

[17] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M. Dai, Anja Hauth, Katie Millican, David Silver, et al. Gemini: A family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

[18] Kaixiong Gong, Kaituo Feng, Bohao Li, Yibing Wang, Mofan Cheng, Shijia Yang, Jiaming Han, Benyou Wang, Yutong Bai, Zhuoran Yang, and Xiangyu Yue. Av-odyssey bench: Can your multimodal llms really understand audio-visual information? arXiv preprint arXiv:2412.02611, 2024.

[19] Klaus Gref, Francois Belletti, Lucas Beyer, Carl Doersch, Yilun Du, Daniel Duckworth, David J Fleet, Dan Gnanapragasam, Florian Golemo, Charles Herrmann, et al. Kubric: A scalable dataset generator. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3749–3761, 2022.

[20] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. Hallusionbench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14375–14385, 2024.

[21] Minghao Han, Dingkang Yang, Yue Jiang, Yizhou Liu, and Lihua Zhang. Omnifysics: Towards physical intelligence evolution via omni-modal signal processing and network optimization. arXiv preprint arXiv:2602.07064, 2026.

[22] Jack Hong, Shilin Yan, Jiayin Cai, Xiaolong Jiang, Yao Hu, and Weidi Xie. Worldsense: Evaluating real-world omnimodal understanding for multimodal llms. In International Conference on Learning Representations (ICLR), 2026.

[23] Zhiyuan Hu, Zheng Sun, Yi Wei, and Long Yu. Physical plausibility reasoning via hcm-grpo: Empowering compact model for superior performance. Pattern Recognition, 180:114108, 2026.

[24] Wenxuan Huang, Bohan Jia, Shaosheng Cao, Zheyu Ye, Fei zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. In International Conference on Learning Representations, pages 63794–63812, 2026.

[25] Yue Jiang, Dingkang Yang, Minghao Han, Jinghang Han, Zizhi Chen, Yizhou Liu, Mingcheng Li, Peng Zhai, and Lihua Zhang. Fysicsworld: A unified full-modality benchmark for any-to-any understanding, generation, and reasoning. arXiv preprint arXiv:2512.12756, 2025.

[26] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.

[27] Mingxin Li, Yanzhao Zhang, Dingkun Long, Keqin Chen, Sibo Song, Shuai Bai, Zhibo Yang, Pengjun Xie, An Yang, Dayiheng Liu, et al. Qwen3-vl-embedding and qwen3-vl-reranker: A unified framework for state-of-the-art multimodal retrieval and ranking. arXiv preprint arXiv:2601.04720, 2026.

[28] Puyin Li, Tiange Xiang, Ella Mao, Shirley Wei, Xinye Chen, Adnan Masood, Li Fei-Fei, and Ehsan Adeli. Quantiphy: A quantitative benchmark evaluating physical reasoning abilities of vision-language models. arXiv preprint arXiv:2512.19526, 2025.

[29] Yadong Li, Jun Liu, Tao Zhang, Song Chen, Tianpeng Li, Zehuan Li, Lijun Liu, Lingfeng Ming, Guosheng Dong, Da Pan, et al. Baichuan-omni-1.5 technical report. arXiv preprint arXiv:2501.15368, 2025.

[30] Yizhi Li, Ge Zhang, Yinghao Ma, Ruibin Yuan, Kang Zhu, Hangyu Guo, Yiming Liang, Jiaheng Liu, Zekun Wang, Jian Yang, Siwei Wu, Xingwei Qu, Jinjie Shi, Xinyue Zhang, Zhenzhu Yang, et al. Omnibench: Towards the future of universal omni-language models. arXiv preprint arXiv:2409.15272, 2024.

[31] Derek Lilienthal, Manisha Mukherjee, and Sameera Horawalavithana. Reward design for physical reasoning in vision-language models. arXiv preprint arXiv:2604.13993, 2026.

[32] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281, 2023.

[33] Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visual-rft: Visual reinforcement fine-tuning. arXiv preprint arXiv:2503.01785, 2025.

[34] Jiasen Lu, Christopher Clark, Sangho Lee, Zichen Zhang, Savya Khosla, Ryan Marten, Derek Hoiem, and Aniruddha Kembhavi. Unified-io 2: Scaling autoregressive multimodal models with vision, language, audio and action. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 26439–26455, 2024.

[35] Shiyin Lu, Yang Li, Yu Xia, Yuwei Hu, Shanshan Zhao, Yanqing Ma, Zhichao Wei, Yinglun Li, Lunhao Duan, Jianshan Zhao, Yuxuan Han, Haijun Li, Wanying Chen, Junke Tang, et al. Ovis2.5 technical report. arXiv preprint arXiv:2508.11737, 2025.

[36] Qianli Ma, Yaowei Zheng, Zhelun Shi, Zhongkai Zhao, Bin Jia, Ziyue Huang, Zhiqi Lin, Youjie Li, Jiacheng Yang, Yanghua Peng, et al. Veomni: Scaling any modality model training with model-centric distributed recipe zoo. arXiv preprint arXiv:2508.02317, 2025.

[37] Ziyang Ma, Yinghao Ma, Yanqiao Zhu, Chen Yang, Yi-Wen Chao, Ruiyang Xu, Wenxi Chen, Yuanzhe Chen, Zhuo Chen, Jian Cong, Kai Li, Keliang Li, Siyou Li, Xinfeng Li, et al. Mmar: A challenging benchmark for deep reasoning in speech, audio, music, and their mix. arXiv preprint arXiv:2505.13032, 2025.

[38] OpenAI. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

[39] Saurabh Pathak, Elahe Arani, Mykola Pechenizkiy, and Bahram Zonooz. Physvid: Physics aware local conditioning for generative video models. arXiv preprint arXiv:2603.26285, 2026.

[40] Benjamin Pikus, Pratyush Ranjan Tiwari, and Burton Ye. Hard examples are all you need: Maximizing grpo post-training under annotation budgets. arXiv preprint arXiv:2508.14094, 2025.

[41] Qwen Team. Qwen3.5 technical report. arXiv preprint, 2026.

[42] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In International Conference on Machine Learning (ICML), volume 202, pages 28492–28518, 2023.

[43] Ronan Riochet, Mario Ynocente Castro, Mathieu Bernard, Adam Lerer, Rob Fergus, Veronique Izard, and Emmanuel Dupoux.´ Intphys: A framework and benchmark for visual intuitive physics reasoning. arXiv preprint arXiv:1803.07616, 2020.

[44] Sakshi Sakshi, Utkarsh Tyagi, Sonal Kumar, Ashish Seth, Ramaneswaran Selvakumar, Oriol Nieto, Ramani Duraiswami, Sreyan Ghosh, and Dinesh Manocha. Mmau: A massive multi-task audio understanding and reasoning benchmark. In International Conference on Learning Representations (ICLR), volume 2025, pages 84929–84964, 2025.

[45] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

[46] Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, Ruochen Xu, and Tiancheng Zhao. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025.

[47] Hui Shen, Taiqiang Wu, Qi Han, Yunta Hsieh, Jizhou Wang, Yuyue Zhang, Yuxin Cheng, Zijian Hao, Yuansheng Ni, Xin Wang, Zhongwei Wan, Kai Zhang, Wendong Xu, Jing Xiong, et al. Phyx: Does your model have the “wits” for physical reasoning? arXiv preprint arXiv:2505.15929, 2025.

[48] Qwen Team. Qwen3.5-omni technical report. arXiv preprint arXiv:2604.15804, 2026.

[49] Guo-Hua Wang, Shanshan Zhao, Xinjie Zhang, Liangfu Cao, Pengxin Zhan, Lunhao Duan, Shiyin Lu, Minghao Fu, Jianshan Zhao, Yang Li, and Qing-Guo Chen. Ovis-u1 technical report. arXiv preprint arXiv:2506.23044, 2025.

[50] Lintao Wang, Encheng Su, Jiaqi Liu, Pengze Li, Jiabei Xiao, Wenlong Zhang, Xinnan Dai, Xi Chen, Yuan Meng, Lei Bai, Wanli Ouyang, Shixiang Tang, Aoran Wang, and Xinzhu Ma. Physunibench: A multi-modal physics reasoning benchmark at undergraduate level. arXiv preprint arXiv:2506.17667, 2026. URL https://arxiv.org/abs/2506.17667.

[51] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and eficiency. arXiv preprint arXiv:2508.18265, 2025.

[52] Jiajun Wu, Joseph J Lim, Hongyi Zhang, Joshua B Tenenbaum, and William T Freeman. Physics 101: Learning physical object properties from unlabeled videos. In BMVC, volume 2, page 7, 2016.

[53] Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, Bin Zhang, Xiong Wang, Yunfei Chu, and Junyang Lin. Qwen2.5-omni technical report. arXiv preprint arXiv:2503.20215, 2025.

[54] Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, Yuanjun Lv, Yongqi Wang, et al. Qwen3-omni technical report. arXiv preprint arXiv:2509.17765, 2025.

[55] Shan Yang. Physics-r1: An audited olympiad corpus and recipe for visual physics reasoning. arXiv preprint arXiv:2605.14040, 2026.

[56] Hanrong Ye, Chao-Han Huck Yang, Arushi Goel, Wei Huang, Ligeng Zhu, Yuanhang Su, Sean Lin, An-Chieh Cheng, Zhen Wan, Jinchuan Tian, Yuming Lou, Dong Yang, Zhijian Liu, et al. Omnivinci: Enhancing architecture and data for omni-modal understanding llm. arXiv preprint arXiv:2510.15870, 2025.

[57] Kexin Yi, Chuang Gan, Yunzhu Li, Pushmeet Kohli, Jiajun Wu, Antonio Torralba, and Joshua B. Tenenbaum. Clevrer: Collision events for video representation and reasoning. arXiv preprint arXiv:1901.01442, 2020.

[58] Weijie Yin, Yongjie Ye, Fangxun Shu, Yue Liao, Zijian Kang, Hongyuan Dong, Haiyang Yu, Dingkang Yang, Jiacong Wang, Han Wang, Wenzhuo Liu, Xiao Liang, Shuicheng Yan, and Chao Feng. Sail-vl2 technical report. arXiv preprint arXiv:2509.14033, 2025.

[59] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, et al. Dapo: An open-source llm reinforcement learning system at scale. In Advances in Neural Information Processing Systems, volume 38, 2025.

[60] Xiang Yue, Yuansheng Ni, Tianyu Zheng, Kai Zhang, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9556–9567, 2024.

[61] Jingyi Zhang, Jiaxing Huang, Huanjin Yao, Shunyu Liu, Xikun Zhang, Shijian Lu, and Dacheng Tao. R1-vl: Learning to reason with multimodal large language models via step-wise group relative policy optimization. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 1859–1869, 2025.

[62] Ye Zhang, Xuehang Guo, Rui Pan, Pengfei Yu, Denghui Zhang, Manling Li, and Qingyun Wang. Decoupled physical modeling and execution for physics reasoning. arXiv preprint arXiv:2608.22126, 2026.

[63] Yifan Zhang and Math-AI Team. American invitational mathematics examination (aime), 2025.

[64] Yuze Zhao, Jintao Huang, Jinghan Hu, Xingjun Wang, Yunlin Mao, Daoze Zhang, Zeyinzi Jiang, Zhikai Wu, Baole Ai, Ang Wang, Wenmeng Zhou, and Yingda Chen. Swift:a scalable lightweight infrastructure for fine-tuning. arXiv preprint arXiv:2408.05517, 2024.

[65] Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, and Junyang Lin. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.

[66] Fengzhe Zhou, Jiannan Huang, Jialuo Li, Deva Ramanan, and Humphrey Shi. Pai-bench: A comprehensive benchmark for physical ai. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21522–21536, 2026.

[67] Siyuan Zhou, Hejun Wang, Hu Cheng, Jinxi Li, Dongsheng Wang, Junwei Jiang, Yixiao Jin, Jiayue Huang, Shiwei Mao, Shangjia Liu, et al. Physinone: Visual physics learning and reasoning in one suite. arXiv preprint arXiv:2604.09415, 2026.

[68] Ziwei Zhou, Rui Wang, Zuxuan Wu, and Yu-Gang Jiang. Daily-omni: Towards audio-visual reasoning with temporal alignment across modalities. arXiv preprint arXiv:2505.17862, 2026.