# A Mathematical Theory of Pragmatic Information

Kai Niu, Senior Member, IEEE, Ping Zhang, Fellow, IEEE

## Abstract

We propose a mathematical theory of pragmatic information that unifies communication, control, and decision-making within a single coherent framework. At its core lies the isoteleia mapping, which formalizes equifinality, the principle that distinct semantic paths leading to the same optimal action are pragmatically equivalent, and this induces a three-tier hierarchy of syntactic, semantic, and pragmatic information where each successive abstraction discards task-irrelevant distinctions. We develop a complete set of information-theoretic measures at the pragmatic level, including pragmatic entropy, up/down pragmatic mutual information, channel capacity, and rate-distortion, and prove three coding theorems: lossless source coding, channel coding, and rate-distortion, which generalize Shannon’s classical results. We introduce pragmatic value of information (VoI) and pragmatic cost of information (CoI) as decisiontheoretic duals to rate-distortion and capacity, respectively, and formulate a Lagrangian dual framework for cross-layer optimization. The resulting pragmatic efficiency bound $\mathcal E _ { p } ( \lambda ) = \operatorname* { s u p } _ { R } [ \Phi _ { p } ( R ) - \lambda \operatorname { C o I } _ { p } ( R ) ]$ quantifies the maximum net utility that any resource-constrained intelligent system can extract from its environment, thereby establishing a fundamental limit on purposeful behavior—a behavioral capacity that generalizes Shannon’s symbol-level capacity to the realm of goal-directed action. Extensions to continuous messages yield closed-form expressions for Gaussian channels and sources, while dynamic settings are addressed through a Bellman equation for sequential decision-making. This framework provides a rigorous foundation for task-oriented communication, networked control, autonomous systems, and embodied AI, shifting the focus from symbol fidelity to the effectiveness of information in guiding

actions and offering a unified mathematical language for the design of next-generation intelligent systems.

## Index Terms

Pragmatic Information Theory, Isoteleia Mapping, Reification Mapping, Pragmatic Entropy, Pragmatic Mutual Information, Pragmatic Channel Capacity, Pragmatic Rate-Distortion, Value of Information, Cost of Information, Pragmatic Lagrangian

## I. INTRODUCTION

## A. Background and Motivation

Classic information theory (CIT), established by C. E. Shannon in 1948 [1, 2], was a monumental achievement that laid the mathematical foundations for modern communication. By quantifying information through entropy, mutual information, channel capacity, and rate-distortion functions, CIT provided fundamental limits on data compression and reliable transmission over noisy channels. These theoretical bounds have guided the development of practical coding schemes over the past seven decades, with techniques such as Huffman coding, arithmetic coding, turbo codes, LDPC codes, and polar codes approaching the Shannon limits [1, 45, 46]. In parallel, the emergence of cybernetics and control theory established complementary foundations for understanding and designing dynamical systems. Wiener’s cybernetics [4] first unified the concepts of communication and control, while Tsien’s engineering cybernetics [6] and Bellman’s dynamic programming [5] provided powerful tools for optimal decision-making under uncertainty. Kalman’s filtering theory [7] further bridged estimation and control, enabling real-time state estimation from noisy observations. Decision theory, particularly through the work of Stratonovich on the value of information (VoI) [8, 9], addressed the fundamental question of how much utility can be gained from acquiring information before making a decision. Despite their shared intellectual roots, these three pillars—communication theory, control theory, and decision theory—have largely developed along independent trajectories, with limited cross-fertilization among them.

Weaver [3] first articulated that communication involves not only the accurate transmission of symbols (Level A) but also the conveyance of meaning (Level B) and the effectiveness of conduct (Level C). This layered view has motivated numerous efforts to formalize semantic information. Early attempts to formalize semantic information include Carnap and Bar-Hillel’s logical approach based on propositional logic [10], Floridi’s theory of strongly semantic information [11], and fuzzy entropy formulations by De Luca and Termini [13, 14] and Wu [15]. Zhong [16, 17] and Lu [18, 19] independently generalized Shannon’s information theory to the semantic and pragmatic levels. Bao et al. [12] extended the logical framework to derive semantic source and channel coding theorems. More recently, Liu et al. [20] and others [28–30] investigated rate-distortion frameworks for semantic information. However, a systematic information-theoretic treatment of semantic communication remained elusive until the work of Niu and Zhang [21–23], who established a rigorous mathematical theory of semantic communication centered on the concept of synonymy, the principle that the same meaning can be expressed by many different syntactic realizations. Through the synonymous mapping, they introduced semantic entropy, up/down semantic mutual information, semantic channel capacity, and semantic rate-distortion functions, proving corresponding coding theorems that demonstrate performance gains over classical counterparts.

Despite these pioneering efforts, a systematic information-theoretic treatment of pragmatic information has remained largely absent. Early philosophical inquiries [24] and historical surveys [25] provided valuable conceptual foundations but stopped short of developing a rigorous mathematical framework. The quantitative formulation by Weinberger [26, 27], while introducing a formal definition of pragmatic information as the change in an agent’s action distribution, was primarily developed within the context of biological evolution and did not establish fundamental coding theorems, rate-distortion limits, or channel capacity characterizations. More critically, none of these prior works provided a unified framework that integrates pragmatic information with syntactic and semantic layers. Consequently, a comprehensive mathematical theory capable of quantifying, compressing, and reliably transmitting pragmatic information in engineered communication systems has been lacking until now.

The recent surge of interest in semantic communication, driven by advances in deep learning and the growing demand for intelligent communications, has further highlighted the limitations of the classical syntactic paradigm. Deep learning-based semantic communication systems have demonstrated significant performance improvements in tasks such as image transmission, speech communication, and text understanding [32–36]. These systems extract and transmit semantic features relevant to the task, moving beyond bit-level fidelity to meaning-level effectiveness. The convergence of communication and artificial intelligence has opened new possibilities for task-oriented communication systems that prioritize the end goal over exact symbol reconstruction. Zhang et al. [31] and Niu et al. [36] have articulated the vision of wisdom-evolutionary and primitive-concise communication networks for 6G, where semantic information plays a central role.

Equally important is the convergence of communication and control, extensively studied in networked control systems (NCS) and cyber-physical systems (CPS), where joint design of communication and control policies is essential for stability and performance under bandwidth constraints. Foundational results on this co-design have been established by Xiao et al. [40], with comprehensive surveys provided by Park et al. [38], Wang et al. [39], Zhao et al. [41], Negi and Chakrabortty [43], and Lu and Guo [42]. Tishby and Polani [37] incorporated informationtheoretic costs into the Bellman equation from a perception-action perspective, though their focus remained on action complexity rather than physical communication resources. Critically, these works treat information at the syntactic level, emphasizing signal fidelity over semantic or pragmatic content.

At a deeper level, modern information-driven systems, whether they are large language models interacting with users, autonomous vehicles navigating complex environments, embodied AI agents performing physical tasks, or human learners acquiring knowledge, share a common operational structure that can be abstracted as a perception-communication-decision-execution (PCDE) closed loop. In such systems, information is not merely transmitted for its own sake; it is used to guide decisions and actions that ultimately determine system performance. The pragmatic dimension of information, its effect on conduct, is therefore paramount. The following examples illustrate the ubiquity of the PCDE loop and the central role of pragmatic information across diverse domains.

Large Language Models and AI Agents. In a typical interaction with a large language model (LLM) or AI agent, the user submits a query (perception), the model processes the query through its internal reasoning (communication), generates a response or takes an action (decision), and the user applies the response to their task (execution). The pragmatic value of the response lies not in its syntactic correctness but in its effectiveness in helping the user achieve their goal—whether solving a problem, generating code, or making a decision. The same response may be pragmatically equivalent or entirely different depending on the user’s context and intent. Figure 1 illustrates this PCDE loop, where pragmatic information bridges the gap between

![](images/b4277052e532d816e66e5558d9fe5d80872d2cec5bc843b94e6e9476cdbf42f4.jpg)  
Fig. 1: The perception-communication-decision-execution (PCDE) loop in large language model (LLM) and AI agent interactions. The user query (perception) is processed by the LLM (communication/understanding), which generates a response (decision) that drives the user’s subsequent action (execution). Pragmatic information resides in the effectiveness of the response in achieving the user’s intended goal.

## meaning and effective action.

Autonomous Driving. An autonomous vehicle operates within a complex PCDE loop: sensors (cameras, LiDAR, radar) perceive the surrounding environment (perception), the vehicle communicates with other vehicles and infrastructure via V2X (communication), the planning module decides on acceleration, braking, and steering (decision), and the actuators execute the planned actions (execution). The effectiveness of the system is measured by safety, efficiency, and comfort—all pragmatic objectives that depend on the correct interpretation of sensory data and the appropriate selection of actions. The same set of sensor readings may lead to different actions depending on the context (e.g., yielding to a pedestrian vs. proceeding through an intersection), highlighting the need for pragmatic reasoning. Figure 2 depicts this loop, where pragmatic information—the optimal action class—is the key to safe and efficient navigation.

Embodied AI and Robotics. Embodied AI agents and robots operate in physical environments, where perception, communication, decision, and execution are tightly coupled. Sensors perceive the environment (perception), internal representations communicate state and intent across modules (communication), the control policy decides on the next action (decision), and motors and actuators execute the motion (execution). The success of the system depends on the pragmatic effectiveness of the actions—whether the robot grasps the object, navigates to the target, or completes the task. The same sensory input may lead to different actions depending on the task goal, illustrating the context-dependent nature of pragmatic information. Figure 3 shows this loop, where pragmatic information is the bridge between perception and effective physical action.

![](images/3b40caab84993274e4d7eb4359f86c86a7c02ee13df1045ffc0cf84cae17331c.jpg)  
Fig. 2: The perception-communication-decision-execution loop in autonomous driving. Sensors perceive the environment (perception), the vehicle communicates with other vehicles and infrastructure (communication), the planning module decides on a trajectory (decision), and the actuators execute the maneuver (execution). Pragmatic information—the correct action to take—is the ultimate objective that transcends individual sensor measurements.

Human Learning and Education. Human learning follows a PCDE loop where instructional content is perceived (perception), communicated through explanation, demonstration, and feedback (communication), the learner decides on a learning strategy (decision), and applies it through practice and application (execution). The effectiveness of instruction depends on its pragmatic value—whether it enables the learner to perform better in real-world tasks. The same instructional content may have high pragmatic value for one learner but low value for another, depending on prior knowledge and learning context. Figure 4 illustrates this loop, where pragmatic information is the knowledge that translates into improved performance.

![](images/f1c4cdead31f95f5770fe24633b779fcc4f60266065844c5d0ede573b78c3628.jpg)  
Fig. 3: The perception-communication-decision-execution loop in embodied AI and robotics. Sensors perceive the environment (perception), internal representations communicate state information (communication), the policy selects an action (decision), and the actuators interact with the physical world (execution). Pragmatic information is embodied in the action’s effectiveness in achieving the task objective.

These four examples, spanning AI, autonomous systems, robotics, and human cognition, share a common underlying structure: information flows through a perception-communication-decisionexecution (PCDE) loop, and the ultimate measure of success is the effectiveness of the resulting actions. In each domain, pragmatic information, the information that makes a difference to decisions and actions, is the central concern. The value of information (VoI) quantifies the utility gain from information, while the cost of information (CoI) captures the resources needed to acquire and transmit it. Despite the ubiquity of this structure across diverse domains, no unified mathematical framework exists for describing, analyzing, and optimizing pragmatic information systems. Classical information theory emphasizes symbol fidelity; control theory emphasizes system stability; and decision theory emphasizes utility maximization. None provides a comprehensive account of the interplay among perception, communication, decision, and execution. A unified theory is therefore needed to bridge these disciplines and to provide a rigorous foundation for the design of modern information-driven systems.

![](images/956eadba84901161aeccbad8330fc6d52fe102666a3803cb6e1467c184efb80c.jpg)  
Fig. 4: The perception-communication-decision-execution loop in human learning. Instructional content is perceived (perception), communicated through explanation and feedback (communication), the learner decides on a learning strategy (decision), and applies it to practice (execution). Pragmatic information is the knowledge that effectively improves performance, beyond mere factual recall.

## B. Contributions and Organization

This paper presents a mathematical theory of pragmatic information that unifies communication, control, and decision-making within a single coherent framework. The proposed theory extends classical information theory by incorporating the pragmatic dimension of information, its effect on conduct, while subsuming semantic information theory and classical information theory as special cases. By introducing the isoteleia mapping, which formalizes the principle of equifinality (distinct semantic paths converging to the same optimal action), we establish a three-tier hierarchy of syntactic, semantic, and pragmatic information that provides a complete characterization of information in goal-directed systems.

![](images/930845c8d7b646a70e9a912eee35ffead7b3bdeabcc6a023bd2ee6c973da5950.jpg)  
Fig. 5: Three-tier framework of pragmatic information theory and its integration with communication, control, and decision.

Figure 5 illustrates the three-tier hierarchy of information and its integration with the PCDE (perception-communication-decision-execution) closed loop. Syntactic information, at the bottom, is the domain of classical information theory and concerns the faithful transmission of symbols. Semantic information, the middle layer captured by the synonymous mapping, formalizes the principle that the same meaning can be expressed by many different syntactic realizations. At the top lies pragmatic information, the focus of this paper, which concerns the effectiveness of information in guiding actions. The isoteleia mapping groups semantic classes that lead to the same optimal terminal action, embodying the principle of equifinality. Their composition, the reification mapping, directly bridges pragmatic purposes to syntactic signals, enabling the materialization of intent into physical communication. The PCDE loop connects the Physical World through perception, communication, decision, and execution, with pragmatic information serving as the essential link that closes the loop, answering not only “What signal should be sent?” (syntax) and “What does it mean?” (semantics), but fundamentally “What difference does this information make to the decisions and actions of the receiver?” (pragmatics). This unified framework positions pragmatic information theory as a natural generalization that subsumes both classical and semantic information theories, while simultaneously integrating decision theory and cybernetics into a single coherent structure. The three theories are nested and compatible: pragmatic information theory contains semantic information theory as a special case when the isoteleia mapping is trivial, and both contain classical information theory when the synonymous mapping is also trivial. This hierarchical compatibility ensures backward compatibility with existing theories while extending their reach into decision-making and control.

![](images/89e35f00d26aa0ddf49a6afc80dcf48516269621deeb433dd07164e38b3dea40.jpg)  
Fig. 6: The framework of pragmatic information theory.

As illustrated in Fig. 6, the main contributions of this paper are as follows:

1) Unified Pragmatic Information System Framework. We develop a comprehensive system model for pragmatic information, integrating Shannon’s communication paradigm with cybernetic control. The system is structured as a perception-communication-decisionexecution (PCDE) closed loop, comprising ten core modules organized into two functional subsystems: the Cognitive-Regulation Subsystem (CRS), which handles reasoning, intent formulation, and value/cost evaluation, and the Symbolic-Conduction Subsystem (SCS), which handles encoding, transmission, and decoding. The reification mapping $g = f \circ e$ serves as the critical bridge between these subsystems, enabling the direct materialization of pragmatic intent into syntactic signals.

2) Information-Theoretic Measures for Pragmatic Information. We define a complete set of information-theoretic measures at the pragmatic level, including pragmatic entropy $H _ { p } ( \underline { { W } } )$ up/down pragmatic mutual information $I ^ { p } ( \underline { { X } } ; \underline { { Y } } )$ and $I _ { p } ( \underline { { X } } ; \underline { { Y } } )$ , pragmatic channel capacity $C _ { p } ,$ and pragmatic rate-distortion function $R _ { p } ( D )$ . We establish the fundamental hierarchies that relate these measures to their syntactic and semantic counterparts: $H _ { p } ( \underline { { W } } ) \leq H _ { s } ( \tilde { W } ) \leq$ H(W), $C _ { p } \geq C _ { s } \geq C$ , and $R _ { p } ( D ) \leq R _ { s } ( D ) \leq R ( D )$ . These hierarchies quantify the performance gains achievable by exploiting pragmatic abstraction.

3) Pragmatic Value of Information and Pragmatic Cost of Information. We introduce the pragmatic value of information (VoI) as the utility gain obtained from task-relevant information, and the pragmatic cost of information (CoI) as the minimum resource expenditure required to convey information at a given pragmatic rate. We establish the dualities between VoI and the rate-distortion function, and between CoI and the channel capacity, and develop single-sided extensions for perceptual and expressive paths. These measures provide a decision-theoretic and economic interpretation of the information-theoretic limits.

4) Lagrangian Dual Framework and Pragmatic Shadow Price for Cross-Layer Optimization. We develop a unified Lagrangian dual framework that integrates the rate-distortion-value duality (down loop) and the capacity-cost duality (up loop) into a coherent cross-layer optimization principle. The global pragmatic Lagrangian $\mathcal { L } _ { \mathrm { g l o b a l } } ( R ; \lambda ) = \Phi _ { p } ( R ) - \lambda \operatorname { C o I } _ { p } ( R )$ expresses the net benefit as value minus cost, with the optimality condition $\begin{array} { r l } { \lambda \operatorname { C o l } _ { p } ^ { \prime } ( R ^ { * } ) = } \end{array}$ $\Phi _ { p } ^ { \prime } ( R ^ { * } )$ balancing marginal value against marginal cost. The resulting pragmatic efficiency functional $\begin{array} { r } { \mathcal { E } _ { p } ( \lambda ) = \operatorname* { s u p } _ { R } [ \Phi _ { p } ( R ) - \lambda \operatorname { C o I } _ { p } ( R ) ] } \end{array}$ defines a fundamental behavioral limit for any resource-constrained intelligent system—a behavioral capacity that parallels Shannon’s physical capacity, but replaces symbol fidelity with the effectiveness of information in guiding actions. Central to this framework is the shadow price λ, which serves as a unified currency that converts physical resource consumption—such as transmit power, bandwidth, and latency—into utility-equivalent units. In static settings, λ determines the optimal operating point where marginal value equals marginal cost; in dynamic settings, it acts as a real-time scarcity signal that continuously adapts system behavior to varying channel conditions, power budgets, and task priorities. This framework provides a principled, practical, and dimensionally consistent approach for resource allocation and system design in task-oriented communication systems, bridging information-theoretic limits with decisiontheoretic objectives without ad hoc heuristics.

5) Coding Theorems for Pragmatic Communication. We prove three fundamental coding theorems for pragmatic information systems: (i) the pragmatic lossless source coding theorem, establishing that the minimum rate for lossless pragmatic compression is the pragmatic entropy $H _ { p } ( \underline { { W } } )$ ; (ii) the pragmatic channel coding theorem, establishing that the maximum reliable transmission rate is the pragmatic capacity $C _ { p } ;$ ; and (iii) the pragmatic rate-distortion coding theorem, establishing that the minimum rate for lossy pragmatic compression with distortion D is $R _ { p } ( D )$ . These theorems generalize Shannon’s classical coding theorems [1, 2] and the semantic coding theorems of Niu and Zhang [21, 22].

6) Continuous-Domain Pragmatic Information Measures. We extend the pragmatic information measures to continuous messages through the concept of isoteleic volumes Ω, which are the continuous analogues of pragmatic equivalence classes. We derive closed-form expressions for the pragmatic capacity of the Gaussian channel $\begin{array} { r } { C _ { p } = \frac { 1 } { 2 } \log ( \Omega ^ { 4 } ( 1 + P / \sigma ^ { 2 } ) ) } \end{array}$ and the pragmatic rate-distortion function for Gaussian sources $\begin{array} { r } { R _ { p } ( D ) = \frac { 1 } { 2 } \log ( P / ( \Omega ^ { 4 } D ) ) } \end{array}$ , along with the corresponding CoI and VoI functions. We also derive the band-limited capacity $C _ { p } = B \log ( \Omega ^ { 4 } ( 1 + P / ( N _ { 0 } B ) ) )$

7) Joint Optimization of Communication, Control, and Decision. We formulate the joint optimization of information, control, and decision-making in both static and dynamic settings. For the static case, we derive the optimality conditions for the global pragmatic Lagrangian, showing that the optimal policy is the posterior-maximizing Bayes decision rule and the optimal rate satisfies marginal value equals marginal cost. For the dynamic case, we derive the Bellman equation for the dynamic pragmatic Lagrangian, providing a principled framework for co-designing communication and control in sequential decision-making systems.

8) Generalization and Unification. We demonstrate that the pragmatic information theory framework seamlessly subsumes classical communication systems (when semantic and pragmatic layers are trivialized), closed physical systems (with explicit utility functions), embodied and interactive systems (with multiple agents), and open social and cognitive systems (with implicit and evolving utility functions). This establishes the pragmatic information system as a universal reference model for analyzing and designing any system that involves goal-directed information processing under uncertainty.

The remainder of the paper is organized as follows. Section II presents the pragmatic information system model, including the system architecture, the three-tier hierarchy, the reification mapping, and the axiomatic foundations. Section III defines pragmatic entropy and its joint and conditional counterparts, establishing the fundamental hierarchies. Section IV introduces pragmatic relative entropy and up/down pragmatic mutual information. Section V defines pragmatic channel capacity and pragmatic rate-distortion functions, including single-sided extensions. Section VI establishes the pragmatic value of information and pragmatic cost of information, along with their Lagrangian dual framework. Section VII proves the pragmatic lossless source coding theorem. Section VIII proves the pragmatic channel coding theorem. Section IX proves the pragmatic rate-distortion coding theorem. Section X extends the pragmatic information measures to continuous messages, deriving the pragmatic capacity of Gaussian channels and the pragmatic rate-distortion function for Gaussian sources, along with the corresponding CoI and VoI functions. Section XI addresses the joint optimization of communication, control, and decision-making in pragmatic information systems. Section XII concludes the paper.

## II. PRAGMATIC INFORMATION SYSTEM AND ISOTELEIA MAPPING

This section presents the theoretical foundations of the pragmatic information system. We establish a unified framework that integrates Shannon’s communication paradigm with cybernetic control, introducing the three-tier hierarchy of syntactic, semantic, and pragmatic information. Central to this framework is the Isoteleia Mapping, which formalizes the principle of equifinality, that is, distinct semantic paths converging to the same optimal action.

## A. Notation Conventions

We adopt standard information-theoretic notation throughout. Calligraphic letters $( \mathcal { X } , \mathcal { Y } , \mathcal { W } , \mathcal { V } )$ denote alphabets; uppercase letters $( X , Y , W )$ are random variables, lowercase $( x , y , w )$ their realizations. Semantic and pragmatic counterparts are marked with a tilde $( \tilde { W } )$ and an underline (W), respectively. Probabilities are written as $P ( \cdot )$ or $p ( \cdot )$ , expectations as $\mathbb { E } [ \cdot ]$ , and log(·) denotes the logarithm to base 2. The main symbols are summarized in Table I; additional notation is introduced as needed in the text.

TABLE I: Principal symbols, meanings, and intuitive explanations
<table><tr><td>Symbol</td><td>Meaning and Intuitive Explanation</td></tr><tr><td> $x , y , w , \nu$ </td><td>Alphabets: collections of possible symbols</td></tr><tr><td> $X , Y , W , V  { / } x , y , w , v$ </td><td>Random variables / realizations at syntactic layer</td></tr><tr><td> $\tilde { X } , \tilde { Y } , \tilde { W } , \tilde { V } / \underline { { X } } , \underline { { Y } } , \underline { { W } } , \underline { { V } }$ </td><td>Semantic layer (meanings) / Pragmatic layer (actions)</td></tr><tr><td> $P ( \cdot ) { \mathrm { ~ o r ~ } } p ( \cdot ) , \mathbb { E } [ \cdot ] , \log ( \cdot )$ </td><td>Probability, expectation, log (bits/sebits/prabits)</td></tr><tr><td> $H ( W ) , H _ { s } ( \tilde { W } ) , H _ { p } ( \underline { { W } } )$ </td><td>Entropy hierarchy: symbol → meaning → action uncertainty</td></tr><tr><td> $H ( \cdot | \cdot )$ </td><td>Conditional entropy: residual uncertainty given context</td></tr><tr><td> $I ( X ; Y )$ </td><td>Syntactic mutual information: symbol-level dependence</td></tr><tr><td> $I _ { s } ( \tilde { X } ; \tilde { Y } ) , I ^ { s } ( \tilde { X } ; \tilde { Y } )$ </td><td>Down/Up semantic MI: meaning dependence (marginal/joint)</td></tr><tr><td> $I _ { p } ( \underline { { X } } ; \underline { { Y } } ) , I ^ { p } ( \underline { { X } } ; \underline { { Y } } )$ </td><td>Down/Up pragmatic MI: action dependence (marginal/joint)</td></tr><tr><td> $C , C _ { s } , C _ { p }$ </td><td>Capacity hierarchy: max reliable symbol/meaning/action rate</td></tr><tr><td> $R ( D ) , R _ { s } ( D ) , R _ { p } ( D )$ </td><td>Rate-distortion hierarchy: min symbol/meaning/action rate for distortion D</td></tr><tr><td> $f : \tilde { \mathcal { W } } \to 2 ^ { \mathcal { W } }$ </td><td>Synonymous mapping: one meaning → many signals</td></tr><tr><td> $e : \underline { { \mathcal { W } } } \to 2 ^ { \tilde { \mathcal { W } } }$ </td><td>Isoteleia mapping: one action → many meanings</td></tr><tr><td> $g = f \circ e : \underline { { \mathcal { W } } }  2 ^ { \mathcal { W } }$ </td><td>Reification mapping: one action → all realizing signals</td></tr><tr><td> $g ^ { n } , f ^ { n } , u ^ { n } , \mathcal { U } ^ { n }$ </td><td>Sequential extensions: block-level mappings and sequences</td></tr><tr><td>Ω</td><td>Average isoteleic volume: avg size of action-equivalence set (continuous)</td></tr><tr><td> $\mathrm { V o I } _ { p } ( R ) ~ \mathrm { o r } ~ \Phi _ { p } ( R )$ </td><td>Pragmatic value of information: max utility gain from rate R</td></tr><tr><td> $\operatorname { C o I } _ { p } ( R )$ </td><td>Pragmatic cost of information: min resource (power/bandwidth) for rate R</td></tr><tr><td> $\mathcal { L } _ { \mathrm { S D V } } , \mathcal { L } _ { \mathrm { C P C } } , \mathcal { L } _ { \mathrm { g l o b a l } }$ </td><td>Lagrangians: down-loop value—cost, up-loop rate—cost, global net benefit</td></tr><tr><td> $\lambda , \gamma$ </td><td>Lagrange multipliers (shadow prices): convert resource cost to utility</td></tr><tr><td> $V ( S )$ </td><td>Value function in dynamic programming: optimal expected future reward</td></tr></table>

For sequences and conditional quantities, we adopt the usual conventions: $u ^ { n } = ( u _ { 1 } , \ldots , u _ { n } )$ $\mathcal { U } ^ { n }$ denotes the n-fold Cartesian product, and $f ^ { n }$ is the sequential extension of a mapping $f .$ Conditional entropy and mutual information are written as $H ( \cdot | \cdot )$ and $I ( \cdot ; \cdot | \cdot )$ , with the convention 0 log 0 = 0. Syntactic quantities are measured in bits, semantic in sebits, and pragmatic in prabits.

## B. Pragmatic Information System

The pragmatic information system is a perception-communication-decision-execution (PCDE) closed-loop architecture that integrates the classic Shannon communication paradigm with a cybernetic control framework. Its fundamental purpose is not merely to transmit symbols, but to ensure that the received information leads to terminal actions that maximize a predefined task utility. In this section, we present the system model in detail, describe its processing flow with state-space dynamics, formulate its axiomatic foundations, and discuss its broad applicability across diverse domains.

1) System Structure and Processing Flow: The system consists of ten core modules that operate in a sequential cascade, forming a complete perception–communication–decision–execution loop. These modules are: (1) Physical World / Task Environment, (2) Task Intent Initiator, (3) Source, (4) Reification Mapping, (5) Encoder, (6) Channel, (7) Decoder, (8) Demapping, (9) Destination, and (10) Task Execution Responder. Figure 7 provides a conceptual illustration of the overall architecture.

We now describe each module and the information flow that connects them.

Module 1: Physical World / Task Environment: The system operates within a Physical World characterized by a Plant with state $S _ { t } \in S$ at discrete time t, governed by:

$$
S _ { t + 1 } = F _ { m } ( S _ { t } , A _ { t } ) + Z _ { t } ,\tag{1}
$$

where $A _ { t } \in { \mathcal { A } }$ is the action applied by the actuator, $F _ { m } : \mathcal { S } \times \mathcal { A }  \mathcal { S }$ is the nominal transition dynamics, and $Z _ { t }$ represents exogenous disturbances. Observations are obtained through a sensor model:

$$
Y _ { t } = F _ { o } ( S _ { t } ) + V _ { t } ,\tag{2}
$$

with $F _ { o } : S  \mathcal { V }$ the observation function and $V _ { t }$ measurement noise.

Module 2: Task Intent Initiator: The Task Intent Initiator compares the observed state (or belief) $\hat { S } _ { t }$ with a task goal $G _ { t }$ to generate a task intent:

$$
T _ { t } = \Phi ( \hat { S } _ { t } , G _ { t } ) ,\tag{3}
$$

where $\Phi$ encodes the system’s strategy. This module answers: “What should be done?”

![](images/c848b314a12b3dc548c3fc8e226609f92ca9e779c442f3a70531cf6b298aa678.jpg)  
Fig. 7: The block diagram of Pragmatic Information System with PCDE loop, showing the interaction between the Physical World, the Cognitive-Regulation Subsystem (CRS), and the Symbolic-Conduction Subsystem (SCS).

Module 3: Source: The Source is a three-tier hierarchical structure corresponding to the pragmatic, semantic, and syntactic levels [2, 3]. The Pragmatic Source W first maps the intent $T _ { t }$ to a pragmatic symbol $w \in \underline { { \mathcal W } }$ representing the optimal action class, answering: “What is the best terminal action?” This is then expanded by the Semantic Source $\tilde { \mathcal W }$ into a semantic symbol $\tilde { w } \in \tilde { \mathcal { W } }$ justifying the action, answering: “Why is this action justified?” Finally, the Syntactic Source W maps to a syntactic symbol $w \in \mathcal W$ for transmission, answering: “What signal should be sent?”

Module 4: Reification Mapping: The Reification Mapping bridges the Pragmatic and Syntactic layers as the composition of the isoteleia and synonymous mappings:

$$
g \triangleq f \circ e : \underline { { \mathcal { W } } } \to \mathcal { P } ( \mathcal { P } ( \mathcal { W } ) ) ,\tag{4}
$$

where $e : \underline { { \mathcal { W } } } \to \mathcal { P } ( \tilde { \mathcal { W } } )$ maps a pragmatic symbol to all semantic justifications (equifinality), and $f : \tilde { \mathcal { W } } \to \mathcal { P } ( \mathcal { W } )$ maps a semantic symbol to all syntactic realizations (synonymy). Thus,

$$
g ( \underline { { w } } ) = \bigcup _ { \tilde { w } \in e ( \underline { { w } } ) } f ( \tilde { w } ) \subset \mathcal { W } ,\tag{5}
$$

enumerating all syntactic realizations that fulfill a given pragmatic purpose.

Module 5: Encoder: The Encoder maps a block of n syntactic symbols to channel input symbols:

$$
\phi : \mathcal { W } ^ { n } \to \mathcal { X } ^ { n } ,\tag{6}
$$

performing a unified source-channel encoding operation designed to maximize end-to-end task utility.

Module 6: Channel: The Channel is the physical medium characterized by the conditional distribution $P ( Y _ { t } | X _ { t } )$ , subject to resource constraints such as power, bandwidth, and delay.

Module 7: Decoder: The Decoder recovers an estimate of the transmitted syntactic sequence:

$$
\psi : \mathcal { V } ^ { n } \to \mathcal { W } ^ { n } ,\tag{7}
$$

ensuring that the recovered symbols preserve semantic and pragmatic content for effective decision-making.

Module 8: Demapping: The Demapping module reverses the Reification process: recovered syntactic symbols are first mapped to estimated semantic symbols via an inverse synonymous mapping, then to estimated pragmatic intentions via an inverse isoteleia mapping, recovering meaning and purpose from the received signals.

Module 9: Destination: The Destination is the hierarchical counterpart to the Source. The Syntactic Destination V receives decoded syntactic symbols; the Semantic Destination $\tilde { \nu }$ reconstructs semantic interpretations; and the Pragmatic Destination V reconstructs the intended pragmatic purpose, answering: “What was the intended purpose?”

Module 10: Task Execution Responder: The Task Execution Responder (Actuator) translates the recovered pragmatic intention wˆ into a physical action:

$$
A _ { t } = \mu ( \underline { { \hat { w } } } _ { t } ) ,\tag{8}
$$

where $\mu : \underline { { \mathcal { V } } } \to \mathcal { A }$ maps pragmatic intentions to actions, closing the perception-communicationdecision-execution loop via Eq. (1).

The effectiveness of the entire process is evaluated by the Value of Information (VoI) and Cost of Information (CoI). The VoI quantifies the expected utility gain:

$$
\mathrm { V o I } = \mathbb { E } _ { S , Y } [ U ( S _ { t } , \mu ( \underline { { \hat { w } _ { t } } } ) ) ] - \mathbb { E } _ { S } [ U ( S _ { t } , \mu ( \underline { { w } } _ { 0 } ^ { * } ) ) ] ,\tag{9}
$$

where $\underline { w } _ { 0 } ^ { * }$ is the optimal action based on prior beliefs alone. The CoI quantifies the minimum resource expenditure required to convey pragmatic information at a given rate. Their joint consideration enables principled trade-off optimization between task performance and resource consumption, a central theme of the pragmatic Lagrangian framework developed in Section VI.

2) Functional Subsystems: The ten modules are naturally grouped into two functional subsystems: the Cognitive-Regulation Subsystem (CRS) and the Symbolic-Conduction Subsystem (SCS).

The CRS handles high-level reasoning, intent formulation, and information value evaluation. It comprises the Task Intent Initiator (generating intent from state and goals), the Pragmatic Source (formalizing intent as pragmatic symbols), the Pragmatic Destination (reconstructing pragmatic intention), and the Task Execution Responder (translating intention into action). The isoteleia mapping e embodies equifinality by grouping semantic justifications that lead to the same optimal action. The CRS answers: “What should be done, and is it worth doing?”

The SCS handles encoding, transmission, and decoding. It comprises the Semantic Source (generating semantic symbols from pragmatic intent), the Syntactic Source (generating syntactic symbols from semantic meanings), the Encoder (converting symbols to physical signals), the Decoder (recovering symbols from received signals), and the Syntactic and Semantic Destinations (reconstructing meanings along the reverse path). The synonymous mapping f captures synonymy by mapping each meaning to multiple syntactic realizations. The SCS answers: “How should it be expressed and transmitted?”

The reification mapping $g = f$ ◦ e serves as the critical bridge between these subsystems, allowing CRS intent to be directly materialized into syntactic signals by the SCS.

3) Axiomatic Foundations: The operation of the pragmatic information system is governed by three fundamental axioms that ensure its mathematical tractability, general applicability, and consistency with both information theory and control theory.

Axiom 1: The Probability Axiom: The system operates in a world where all observations, states, and intentions are subject to irreducible uncertainty. This uncertainty is quantified by probability measures. Formally, for any time t, the joint distribution over the system variables is:

$$
P ( S _ { t } , W _ { t } , Y _ { t } ) \in \Delta ( S \times \mathcal { W } \times \mathcal { Y } ) ,\tag{10}
$$

where $\Delta$ denotes the space of all probability distributions. The channel is characterized by the conditional distribution $P ( Y _ { t } | X _ { t } )$ . This axiom ensures that all information-theoretic quantities—entropy, mutual information, channel capacity, rate-distortion functions, and VoI/VoC—are well-defined and computable.

Axiom 2: The Decision Axiom: All deliberate actions of the system are governed by the principle of maximizing expected utility. Formally, the utility function is defined as:

$$
U : \mathcal { S } \times \mathcal { A }  \mathbb { R } ,\tag{11}
$$

which assigns a real-valued scalar to each state-action pair $( S _ { t } , A _ { t } )$ , representing the immediate reward or cost associated with taking action $A _ { t }$ in state $S _ { t }$ . For any decision node where the system has access to information I, it chooses an action $a ^ { * }$ such that:

$$
a ^ { * } = \arg \operatorname* { m a x } _ { a \in \mathcal { A } } \mathbb { E } _ { P ( S _ { t } \mid I ) } [ U ( S _ { t } , a ) ] .\tag{12}
$$

This axiom underpins the isoteleia mapping, as it defines the equivalence classes of semantic states that lead to the same optimal action (the same telos). It also underpins the Value of Information (VoI), which measures the utility gain obtained from additional information, and the Pragmatic Lagrangian Functional, which formalizes the trade-off between information value and communication cost.

Axiom 3: The Markov Axiom: The system’s dynamics satisfy the Markov property:

$$
P ( S _ { t + 1 } | S _ { t } , S _ { t - 1 } , \ldots , A _ { t } , A _ { t - 1 } , \ldots ) = P ( S _ { t + 1 } | S _ { t } , A _ { t } ) .\tag{13}
$$

This property allows the system to be modeled as a Partially Observable Markov Decision Process (POMDP). Furthermore, the Markov property implies that the system can be decomposed across time: by removing the time indices, the dynamic control problem can be reduced to a static communication and encoding problem. In this static formulation, the channel coding and source coding theorems can be applied directly to characterize the fundamental limits of information transmission, treating each time step independently while preserving the overall structure of the pragmatic system. This temporal decomposition forms the basis for the coding theorems developed in subsequent sections.

4) Generality and Scope: A Unified Framework for Information-Driven Systems: The pragmatic information system provides a universal mathematical language for describing any system that involves goal-directed information processing under uncertainty. The framework spans a continuous spectrum of systems, ranging from pure symbol transmission to tightly coupled physical control loops, and further to open-ended social and cognitive interactions.

We organize the application domains into four broad classes, ordered according to the degree of closure and the nature of the utility function. At one extreme lies the classical communication paradigm, where utility is defined solely by symbol fidelity. Moving along the spectrum, we encounter closed physical systems with well-defined utility functions; embodied and interactive systems with multiple agents; and finally, open social and cognitive systems where utility is implicit, dynamic, and subject to interpretation. This progression reflects increasing complexity in the relationship between information, meaning, and value.

Case I: Classical Communication Systems: When the Semantic and Pragmatic layers are trivialized—i.e., when utility depends solely on symbol fidelity (mean squared error, Hamming distance, or bit error rate)—the isoteleia and synonymous mappings reduce to identity mappings. The three-tier source collapses to a single syntactic source, and the system reverts to the familiar source-channel separation architecture. All classical results, including Shannon’s source coding, channel coding, and rate-distortion theorems, emerge as special cases. The framework thus serves not as a replacement but as a rigorous generalization of classical information theory, extending its reach from “how accurately can symbols be transmitted?” to “how effectively does the received meaning affect conduct in the desired way?”

Case II: Closed Physical Systems: Here the Physical World is governed by well-defined physical laws, the utility function is explicitly specified and measurable (e.g., tracking error, energy consumption, or task success rate), and system behavior is constrained by strict physical and temporal limits. The framework naturally embeds the control-theoretic feedback loop via the Plant, Task Intent Initiator, and Actuator. Representative domains include networked control systems, robotics and industrial automation, autonomous vehicles and intelligent transportation, UAV swarm control, smart grids, and healthcare cyber-physical systems. In all these cases, VoI provides a rigorous metric for quantifying the trade-off between communication resource consumption and closed-loop performance, while the isoteleia mapping captures how diverse sensory inputs converge to the same control or navigation command, enabling robust decision-making under noise and delays.

Case III: Embodied and Interactive Systems: In this class, the physical world remains central, but the system now involves multiple agents (human or artificial) with possibly distinct sensors, actuators, and world-views. Utility may be shared or conflicting, and communication becomes strategic or collaborative rather than a mere data pipe. Representative domains include embodied AI and autonomous agents, multi-agent systems and collaborative AI, human-machine interaction (HMI), and intelligent decision support systems. The framework provides a formal model for signaling games, negotiation protocols, and shared intentionality. The isoteleia mapping captures how different perceptual or contextual states lead to the same joint action or collaborative outcome, while VoI quantifies the benefit of sharing information among agents with potentially asymmetric information and conflicting objectives.

Case IV: Open Social and Cognitive Systems: Here the physical world recedes into the background, and the focus shifts to internal cognitive processes or emergent social dynamics. Utility is no longer a single measurable objective but may be implicit, dynamic, subjective, or even contradictory across agents. Representative domains include human learning and education, human communication and language, social and economic systems (where agents exchange signals to coordinate actions under uncertainty), biological and cognitive systems (aligning with predictive processing and active inference), and recommendation and content delivery systems. The framework offers a rigorous foundation for modeling information exchange without assuming complete rationality, common knowledge, or stable utility functions. The isoteleia mapping captures the human cognitive ability to generalize from diverse experiences to coherent principles, while VoI quantifies the effectiveness of communication in reducing uncertainty and improving decisions under subjective and evolving objectives.

The Unity of the Framework: Despite the apparent diversity across these four cases, they all share a common mathematical core: a state space S, a utility function $U : \mathcal { S } \times \mathcal { A }  \mathbb { R }$ , a three-tier information hierarchy (Syntactic, Semantic, Pragmatic), communication constraints, and a VoI metric guiding resource allocation and decision-making. The spectrum from Case I to Case IV is not a rigid partition but a continuum: classical communication systems can be enhanced with semantic and pragmatic layers; closed physical systems can involve human operators; and social systems can incorporate physical sensors and actuators. The pragmatic information system thus serves as a unified reference model for analyzing and designing systems that range from physical actuators to social networks, from individual agents to entire economies, and from fixed utility functions to dynamically evolving human values.

## C. Design Principles of Pragmatic Information Systems

Building upon the system architecture and the axiomatic foundations established in the preceding subsections, we now articulate three fundamental design principles that guide the analysis, synthesis, and operation of pragmatic information systems. These principles are not arbitrary engineering heuristics; they are direct consequences of the mathematical structure of the pragmatic framework and the nature of information as it flows from the physical world through semantic interpretation to pragmatic action.

Principle I: Pragmatic Imperceptibility and Isoteleia: Pragmatic information, the ultimate purpose or terminal action that a communication system is intended to achieve, is not directly observable. It cannot be read from the received signal as one would read a bit string. Instead, pragmatic information must be inferred or evaluated indirectly through the lens of a utility function, which measures the consequences of actions taken in response to received information. Consequently, the fundamental characteristic of pragmatic information is Isoteleia: different semantic interpretations, and even different syntactic realizations, are pragmatically equivalent if they lead to the same optimal terminal action under the given utility. This is precisely the equivalence relation captured by the isoteleia mapping $e : \underline { { \mathcal { W } } } \to \mathcal { P } ( \tilde { \mathcal { W } } )$ , which partitions the semantic space into classes of meanings that share the same optimal action. The Value of Information (VoI) serves as the primary metric for quantifying pragmatic information, measuring the utility gain obtained from information. This principle implies that the system must be designed around the utility function, not around the fidelity of symbol reconstruction; performance metrics should be defined in terms of the quality of the resulting actions; and communication resources should be allocated according to the marginal Value of Information they provide.

Principle II: Semantic Imperceptibility and Synonymy: Semantic information, the meaning or interpretation conveyed by a signal, is also not directly observable. It cannot be extracted from the signal by simple demodulation or decoding. Instead, semantic information must be inferred from the observed syntactic symbols through a process of pattern recognition, contextual reasoning, and knowledge-based interpretation. Consequently, the fundamental characteristic of semantic information is synonymy: the same meaning can be expressed by many different syntactic realizations. This equivalence relation is captured by the synonymous mapping $f : \tilde { \mathcal { W } } \to \mathcal { P } ( \mathcal { W } )$ which partitions the syntactic space into equivalence classes of symbols that share the same semantic interpretation. This principle implies that the system should exploit synonymy by allowing lossy compression at the syntactic level, as long as the semantic content is preserved; that the decoder must incorporate contextual information to resolve ambiguities; and that the encoder should extract and transmit semantic features relevant to the task, rather than attempting to reconstruct the original signal.

Principle III: Backward Compatibility and Unification: The pragmatic information system must be backward compatible with both classical communication systems and classical control systems. This means that the pragmatic framework is not a replacement for existing theories but a strict generalization that contains them as special cases. When the pragmatic and semantic layers are trivialized, that is, when the utility function depends solely on symbol fidelity and the isoteleia and synonymous mappings become identity mappings, the pragmatic framework must collapse exactly into the classical Shannon communication paradigm. Similarly, when communication is ignored, the framework must collapse into classical control theory. This principle imposes that the three-tier hierarchy must be implemented in a modular fashion, allowing each layer to be independently analyzed, optimized, or bypassed; that the system must include explicit mechanisms to detect when the utility function reduces to syntactic fidelity and automatically switch to classical operation; and that the encoder and decoder must support both pragmatic encoding and syntactic encoding to interface with legacy infrastructure.

Together, these three principles establish the philosophical and mathematical foundation for the design of pragmatic information systems. They ensure that the system is simultaneously task-effective (Principle I), meaning-aware (Principle II), and infrastructure-compatible (Principle III).

## D. Pragmatic Information and Isoteleia Mapping

We now turn to the purely structural account of the mappings that bridge the three layers of information: Syntactic, Semantic, and Pragmatic.

1) The Nature of Pragmatic Information: In classical information theory, information is defined syntactically: it concerns the statistical properties of symbols and their transmission, independent of meaning or purpose. Semantic information, as formalized in the framework of Niu and Zhang [21, 22], extends this by considering the meaning conveyed by symbols: it concerns the relationship between symbols and their interpretations.

Pragmatic information, in contrast, concerns the use or purpose of information. It addresses the fundamental question: “What difference does this information make to the decisions and actions of the receiver?” Pragmatic information is inherently value-laden: it is defined not by what symbols mean, but by what they achieve. Two messages that convey different syntactic symbols and even different semantic meanings may carry the same pragmatic information if they lead the receiver to the same optimal terminal action. Conversely, two messages that convey identical semantic content may carry different pragmatic information if they are interpreted in different contexts or lead to different decisions.

The defining characteristic of pragmatic information is therefore Isoteleia, the principle of equifinality: distinct semantic paths converging to the same ultimate end, or telos. This principle captures the essence of goal-directed communication: the receiver’s objective is not to reconstruct the sender’s exact message, but to extract from it the guidance necessary to achieve a desired outcome.

2) Foundations: Power Sets, Equivalence Relations, and Fibers: To rigorously formalize the relationships among syntactic, semantic, and pragmatic information, we introduce the mathematical structures of power sets, equivalence relations, and fibers. Let W be the syntactic alphabet (the set of all possible physical symbols or signals). A semantic partition of W is a collection of disjoint subsets whose union is W, where each subset groups symbols that share the same meaning. Equivalently, a semantic partition is induced by an equivalence relation $\sim _ { s }$ on W: $w _ { 1 } \sim _ { s } w _ { 2 }$ iff they convey the same semantic meaning. The quotient set $\mathcal { W } / \sim _ { s }$ is the semantic alphabet, denoted by $\tilde { \mathcal { W } }$

The collection of all subsets of W is its power set, denoted $\mathcal { P } ( \mathcal { W } )$ . The semantic partition

$\Pi _ { s e m }$ is a subset of $\mathcal { P } ( \mathcal { W } )$

$$
\Pi _ { s e m } \triangleq \mathcal { W } / \sim _ { s } = \left\{ \tilde { w } _ { 1 } , \tilde { w } _ { 2 } , \hdots , \tilde { w } _ { K } \right\} \subset \mathcal { P } ( \mathcal { W } ) .\tag{14}
$$

Each element $\tilde { w } _ { k } \in \Pi _ { s e m }$ is a semantic fiber (or synonymous class): it is a subset of W containing all syntactic symbols that carry the same meaning.

Similarly, a pragmatic partition is a collection of disjoint subsets of the semantic alphabet, induced by an equivalence relation $\sim _ { p }$ on $\tilde { \mathcal { W } } \colon \tilde { w } _ { i } \sim _ { p } \tilde { w } _ { j }$ iff they lead to the same optimal terminal action. The quotient set $\tilde { \mathcal { W } } / \sim _ { p }$ is the pragmatic alphabet, denoted by W. The pragmatic partition $\Pi _ { p r a g }$ is a subset of the power set of $\tilde { \mathcal { W } } \mathrm { : }$

$$
\Pi _ { p r a g } \triangleq \tilde { \mathcal { W } } / \sim _ { p } = \left\{ \underline { { w _ { 1 } } } , \underline { { w _ { 2 } } } , \hdots , \underline { { w _ { L } } } \right\} \subset \mathcal { P } ( \tilde { \mathcal { W } } ) \subset \mathcal { P } ( \mathcal { P } ( \mathcal { W } ) ) .\tag{15}
$$

Thus, each element ${ \underline { w } _ { l } } \in \Pi _ { p r a g }$ is a set of semantic fibers, i.e., a subset of $\mathcal { P } ( \mathcal { W } )$ , forming a second-order power set. The fibers at each level form a hierarchy: syntactic fibers are individual elements $w \in \mathcal W ;$ ; semantic fibers are subsets $\tilde { w } \subset \mathcal { W } ;$ ; and pragmatic fibers are subsets $\underline { w } \subset \tilde { \mathcal W } \subset \mathcal P ( \mathcal { W } )$ , i.e., collections of semantic fibers. This hierarchy is the mathematical backbone of pragmatic information theory.

3) Formal Definitions of the Mappings: We now formally define the two fundamental mappings—the synonymous mapping and the isoteleia mapping—along with their composite, the reification mapping.

Definition 1 (Synonymous Mapping). Let W be the syntactic alphabet and $\tilde { \mathcal W }$ be the semantic alphabet. Let $\sim _ { s }$ be an equivalence relation on W defined by meaning identity. The Synonymous Mapping

$$
f : \tilde { \mathcal { W } } \to 2 ^ { \mathcal { W } }\tag{16}
$$

is defined for each semantic symbol $\tilde { w } \in \tilde { \mathcal { W } }$ as:

$$
f ( \tilde { w } ) \triangleq \{ w \in \mathscr { W } | w \sim _ { s } \tilde { w } \} \subset \mathscr { W } .\tag{17}
$$

Here, $2 ^ { w }$ denotes the power set of W. The mapping is one-to-many in the forward direction: a single meaning corresponds to a set of syntactic realizations.

Definition 2 (Isoteleia Mapping). Let $\tilde { \mathcal W }$ be the semantic alphabet and let $U : \mathcal { W } \times \mathcal { A } $ R be a utility function. For each semantic symbol $\tilde { w } \in \tilde { \mathcal { W } }$ , define its induced optimal action as:

$$
a ^ { * } ( \tilde { w } ) \triangleq \arg \operatorname* { m a x } _ { a \in \mathcal { A } } \mathbb { E } _ { W \sim P ( W \mid \tilde { w } ) } \left[ U ( W , a ) \right] .\tag{18}
$$

Define an equivalence relation $\sim _ { p }$ on $\tilde { \mathcal W }$ by $\tilde { w } _ { i } \sim _ { p } \tilde { w } _ { j } \iff a ^ { * } ( \tilde { w } _ { i } ) = a ^ { * } ( \tilde { w } _ { j } )$ . The Isoteleia Mapping

$$
e : \underline { { \mathcal { W } } } \to 2 ^ { \tilde { \mathcal { W } } }\tag{19}
$$

is defined for each pragmatic class $\underline { w } \in \underline { { \mathcal { W } } } \triangleq \tilde { \mathcal { W } } / \sim _ { p }$ as:

$$
e ( \underline { { w } } ) \triangleq \Big \{ \tilde { w } \in \tilde { \mathcal { W } } \Big | a ^ { * } \big ( \tilde { w } \big ) = \underline { { w } } \Big \} \subset \tilde { \mathcal { W } } .\tag{20}
$$

The mapping is one-to-many in the forward direction: a single pragmatic purpose corresponds to a set of semantic justifications.

Definition 3 (Dynamic Compatible Isoteleia Mapping). For sequential decision-making problems (Section XI), the optimal action must account for future consequences. Let $V : \Delta ( \mathcal { S } )  \mathbb { R }$ be the value function over belief states. We define the dynamic optimal action for a semantic symbol w˜ as:

$$
a ^ { * } ( \tilde { w } , V ) \triangleq \arg \operatorname* { m a x } _ { a \in \mathcal { A } } \mathbb { E } _ { S | \tilde { w } } \left[ U ( S , a ) + \gamma \sum _ { s ^ { \prime } } P ( s ^ { \prime } | S , a ) V ( b _ { s ^ { \prime } } ^ { \prime } ) \right] ,\tag{21}
$$

where $b _ { s ^ { \prime } } ^ { \prime }$ is the updated belief after transitioning to $s ^ { \prime } .$ Consequently, the dynamic isoteleia mapping is:

$$
e _ { V } ( \underline { { w } } ) \triangleq \left\{ \tilde { w } \in \tilde { \mathcal { W } } \mid a ^ { * } ( \tilde { w } , V ) = \underline { { w } } \right\} .\tag{22}
$$

When $\gamma = 0$ (static decision), we have $e _ { V } ( { \underline { { w } } } ) \equiv e ( { \underline { { w } } } )$ , recovering the static Definition 2. All dynamic Bellman equations in Section XI-B implicitly adopt this conditional extension of the isoteleia mapping.

Definition 4 (Reification Mapping). The Reification Mapping

$$
g : \underline { { \mathcal { W } } } \to 2 ^ { \mathcal { W } }\tag{23}
$$

is defined as the composition of the isoteleia and synonymous mappings:

$$
g ( \underline { { w } } ) \triangleq \bigcup _ { \tilde { w } \in e ( \underline { { w } } ) } f ( \tilde { w } ) \subset \mathcal { W } .\tag{24}
$$

This mapping directly associates a pragmatic purpose with the complete set of syntactic symbols that can realize it.

Figure 8 illustrates the three-tier quotient mapping structure. The isoteleia mapping $e : \underline { { \mathcal { W } } } \to 2 ^ { \tilde { \mathcal { W } } }$ groups semantic classes that lead to the same optimal action, forming pragmatic equivalence classes. The synonymous mapping $f : \tilde { \mathcal { W } } \to 2 ^ { \mathcal { W } }$ associates each semantic meaning with a set of syntactic realizations. Their composition, the reification mapping $g = f \circ e : \underline { { \mathcal { W } } }  2 ^ { \mathcal { W } }$ , directly links a pragmatic purpose to all possible syntactic signals. The resulting pragmatic classes are evaluated by the utility function U and mapped to physical actions $A _ { t } \in A .$ , forming the closed loop of pragmatic information processing.

![](images/2a1986d094b8e637f926ca2d434ab6b9ad0b342f70e92ca8bec9807b2763762f.jpg)  
Fig. 8: The hierarchical mapping from pragmatic classes to semantic classes and syntactic symbols, with correspondence to the action space A and the utility evaluation U.

4) The Five Relations Between Layers: The three mappings give rise to five fundamental relations. The isoteleia mapping e is a pragmatic-to-semantic relation: a single pragmatic class corresponds to multiple semantic classes. Conversely, each semantic class maps to exactly one pragmatic class via the inverse isoteleia, ensuring the partition is well-defined. The synonymous mapping f relates semantic to syntactic: a single meaning corresponds to multiple syntactic symbols, while each syntactic symbol belongs to exactly one semantic class via the inverse synonymous mapping. Finally, the reification mapping g directly relates pragmatic purposes to syntactic signals, associating a single purpose with the union of all syntactic symbols in its associated semantic classes.

5) Consistency Condition: Semantic Refinement: A critical condition must be satisfied for the isoteleia mapping to be well-defined: every semantic class w˜ must be pragmatically homogeneous, meaning that for any two syntactic symbols $w _ { 1 } , w _ { 2 } \in \tilde { w }$ , we must have $a ^ { * } ( w _ { 1 } ) = a ^ { * } ( w _ { 2 } )$ . If this fails, the semantic equivalence is too coarse, and we must refine the semantic partition by splitting w˜ into smaller classes, each homogeneous with respect to the optimal action. This operation, termed semantic refinement, is defined as:

$$
\tilde { w } \to \{ \tilde { w } ^ { ( 1 ) } , \tilde { w } ^ { ( 2 ) } , \dots \} , \quad \tilde { w } ^ { ( j ) } \triangleq \{ w \in \tilde { w } \mid a ^ { * } ( w ) = a _ { j } \} .\tag{25}
$$

Only after such refinement is the isoteleia mapping uniquely defined.

6) Quotient-Space Structure: The syntactic, semantic, and pragmatic layers are successive quotients of the same original set W: $\mathcal { W } \stackrel { \sim _ { s } } { \longrightarrow } \tilde { \mathcal { W } } \stackrel { \sim _ { p } } { \longrightarrow } \mathcal { W }$ . Syntactic information is the finest description, retaining all symbol distinctions. Semantic information is the quotient by meaning equivalence, discarding differences that do not affect meaning. Pragmatic information is the quotient by action equivalence, discarding semantic differences that do not affect optimal actions. Thus, the three layers are different levels of abstraction of the same underlying signal, with the pragmatic layer representing the coarsest, most task-relevant description. This quotientspace perspective provides the mathematical foundation for all subsequent information-theoretic analyses.

## III. PRAGMATIC ENTROPY

In this section, we begin with pragmatic entropy, the fundamental measure of decision uncertainty, and then extend to joint and conditional pragmatic entropies, establishing their compatibility with syntactic and semantic counterparts.

## A. Formal Definition of Pragmatic Entropy

Pragmatic entropy is the fundamental measure of decision uncertainty in a pragmatic information system. Unlike syntactic entropy, which measures uncertainty about which symbol will be transmitted, or semantic entropy, which measures uncertainty about which meaning is conveyed, pragmatic entropy measures the uncertainty about which terminal action will be optimal, that is, the uncertainty that directly affects the system’s ability to achieve its task objectives. This measure is derived from the utility function U through the Isoteleia Mapping $e : \underline { { \mathcal { W } } } \to 2 ^ { \tilde { \mathcal { W } } }$ , which groups semantic symbols into pragmatic equivalence classes based on their induced optimal actions.

Let W be the syntactic alphabet with probability mass function P(W), and let $\tilde { \mathcal W }$ be the semantic alphabet induced by the Synonymous Mapping $f : \tilde { \mathcal { W } } \to 2 ^ { \mathcal { W } }$ . Let the Isoteleia Mapping $e : \underline { { \mathcal { W } } } \to 2 ^ { \tilde { \mathcal { W } } }$ partition the semantic space according to the utility function $U : \mathcal { W } \times \mathcal { A }  \mathbb { R }$ , as defined in Eq. (11).

For each pragmatic symbol $w \in { \underline { { w } } } .$ , its probability is given by:

$$
P ( \underline { { w } } ) = \sum _ { \tilde { w } \in e ( \underline { { w } } ) } P ( \tilde { w } ) = \sum _ { \tilde { w } \in e ( \underline { { w } } ) } \sum _ { w \in f ( \tilde { w } ) } P ( w ) .\tag{26}
$$

Definition 5 (Pragmatic Entropy). The pragmatic entropy of a pragmatic source W is defined as:

$$
H _ { p } ( \underline { { W } } ) \triangleq - \sum _ { \underline { { w } } \in \underline { { W } } } P ( \underline { { w } } ) \log P ( \underline { { w } } ) ,\tag{27}
$$

where $P ( \underline { w } )$ is given by Eq. (26). The unit of pragmatic entropy is the pragmatic bit (Prabit), reflecting that it measures uncertainty about terminal actions rather than about symbols or meanings.

Equivalently, pragmatic entropy can be expressed directly in terms of the syntactic distribution and the composite Reification Mapping $g = f \circ e : \underline { { \mathcal { W } } }  2 ^ { \mathcal { W } }$ :

$$
H _ { p } ( \underline { { W } } ) = - \sum _ { \underline { { w } } \in \mathcal { W } } \left( \sum _ { w \in g ( \underline { { w } } ) } P ( w ) \right) \log \left( \sum _ { w \in g ( \underline { { w } } ) } P ( w ) \right) .\tag{28}
$$

The pragmatic entropy quantifies the minimum average number of pragmatic bits required to specify the optimal terminal action. It represents the irreducible decision uncertainty that must be resolved through communication and control.

We now establish the fundamental properties of pragmatic entropy. These properties are direct consequences of the quotient-space structure and the deterministic nature of the Isoteleia and Synonymous mappings.

Lemma 1 (Non-negativity of Pragmatic Entropy). The pragmatic entropy is non-negative:

$$
H _ { p } ( \underline { { W } } ) \geq 0 .\tag{29}
$$

Equality holds if and only if the pragmatic alphabet has a single element, i.e., there is only one possible optimal action regardless of the state.

Proof: By definition, $P ( \underline { w } ) \in [ 0 , 1 ]$ for all $w \in \underline { { \mathcal { W } } }$ , and $\begin{array} { r } { \sum _ { \underline { { w } } } P ( \underline { { w } } ) = 1 } \end{array}$ . Thus − log $P ( \underline { { w } } ) \geq 0$ and the weighted sum is non-negative. Equality holds if and only if $P ( \underline { w } ) = 1$ for some w and $P ( \underline { { w } } ^ { \prime } ) = 0$ for all $w ^ { \prime } \ne w ,$ , which means the pragmatic alphabet has cardinality 1. □

Lemma 2 (Pragmatic Entropy Hierarchy). Let $H ( W )$ denote the syntactic entropy (classical Shannon entropy), $H _ { s } ( \tilde { W } )$ denote the semantic entropy as defined in semantic information theory, and $H _ { p } ( \underline { { W } } )$ denote the pragmatic entropy. Then:

$$
H _ { p } ( \underline { { W } } ) \leq H _ { s } ( \tilde { W } ) \leq H ( W ) .\tag{30}
$$

Proof: We prove the two inequalities separately.

First inequality: $H _ { p } ( \underline { { W } } ) \leq H _ { s } ( \tilde { W } )$

Since the Isoteleia Mapping $e : \underline { { \mathcal { W } } } \to 2 ^ { \tilde { \mathcal { W } } }$ partitions the semantic alphabet $\tilde { \mathcal W }$ into disjoint equivalence classes, the pragmatic symbol W is a deterministic function of the semantic symbol $\tilde { W }$ . That is, there exists a deterministic function $q : \tilde { \mathcal { W } } \to \mathcal { W }$ such that ${ \underline { { W } } } = q ( { \tilde { W } } )$ . By the data processing inequality for entropy, the entropy of a function of a random variable cannot exceed the entropy of the original variable:

$$
H _ { p } ( { \cal W } ) = H ( q ( { \tilde { \cal W } } ) ) \leq H ( { \tilde { \cal W } } ) = H _ { s } ( { \tilde { \cal W } } ) .\tag{31}
$$

Equality holds if and only if the isoteleia mapping is injective (no two semantic classes are merged into the same pragmatic class).

Second inequality: $H _ { s } ( \tilde { W } ) \le H ( W )$ . This is the fundamental result of semantic information theory, established in [21, 22].

Combining both inequalities yields the desired hierarchy.

This hierarchy captures the essence of pragmatic information theory: each successive layer of abstraction reduces uncertainty by discarding distinctions that are irrelevant to the current level of analysis. The syntactic layer retains all symbol distinctions; the semantic layer discards distinctions that do not affect meaning; and the pragmatic layer discards distinctions that do not affect the optimal terminal action.

We now establish the maximum entropy principle for pragmatic entropy, which characterizes the upper bound of decision uncertainty given the pragmatic alphabet.

Lemma 3 (Maximum Pragmatic Entropy). Let W be the pragmatic alphabet with cardinality $| \underline { { \boldsymbol { \mathcal { W } } } } | = L$ . The pragmatic entropy satisfies:

$$
H _ { p } ( \underline { { W } } ) \leq \log L .\tag{32}
$$

Equality holds if and only if the pragmatic symbols are uniformly distributed, i.e., $P ( \underline { { w } } ) = 1 / L$ for all $w \in { \underline { { w } } }$

Proof: Applying the standard entropy maximization argument, we know that the entropy is maximized when the distribution is uniform. Therefore, $H _ { p } ( \underline { { W } } ) \leq \log L$ , with equality if and only if $P ( \underline { { w } } ) = 1 / L$ for all $w \in { \underline { { w } } }$ □

Corollary 1 (Hierarchy of Maximum Entropies). Let $L _ { p } = | \underline { { \mathcal { W } } } | , L _ { s } = | \tilde { \mathcal { W } } | .$ , and $L = | \mathcal { W } |$ denote the cardinalities of the pragmatic, semantic, and syntactic alphabets, respectively. Since the Isoteleia and Synonymous mappings are quotient mappings, we have $L _ { p } \leq L _ { s } \leq L$ . Consequently:

$$
\log L _ { p } \leq \log L _ { s } \leq \log L .\tag{33}
$$

Combining with Lemma 2, we obtain the full hierarchy of maximum entropies:

$$
\begin{array} { r } { H _ { p } ( \underline { { W } } ) \leq \log L _ { p } \leq \log L _ { s } \leq \log L . } \end{array}\tag{34}
$$

Remark 1. The maximum pragmatic entropy log $L _ { p }$ is strictly less than the maximum semantic entropy log $L _ { s }$ whenever the Isoteleia Mapping is non-injective $( i . e .$ , when multiple semantic classes merge into the same pragmatic class). This reflects the fundamental principle that the goal-directed compression of semantic distinctions reduces the theoretical upper bound of decision uncertainty. In the extreme case where all semantic classes map to a single pragmatic class (i.e., the same action is optimal for all states), we have $L _ { p } = 1$ and $H _ { p } ( \underline { { W } } ) = 0$ , indicating that there is no decision uncertainty whatsoever.

The maximum pragmatic entropy lemma provides a theoretical limit on the amount of decision uncertainty that can exist in a pragmatic information system. It also serves as a design guideline: to maximize the information content of pragmatic communication, the system should be designed such that the pragmatic symbols are as uniformly distributed as possible, within the constraints imposed by the utility function and the environment.

Example 1 (Autonomous Vehicle Decision-Making).

We now present a concrete numerical example to illustrate the calculation and interpretation of pragmatic entropy, comparing it with syntactic and semantic entropies. Consider an autonomous vehicle approaching an intersection. The vehicle’s perception system observes three possible traffic light states, which serve as the syntactic alphabet $\mathcal { W } = \{ w _ { 1 } , w _ { 2 } , w _ { 3 } \}$ with probabilities $P ( W ) = \{ 0 . 6 , 0 . 3 , 0 . 1 \}$ , representing green, yellow, and red $l i g h t s ,$ , respectively.

The vehicle’s semantic interpretation groups traffic lights into two meaning classes: $\tilde { w } _ { 1 } =$ “Proceed with caution” (green and yellow) with $P ( \tilde { w } _ { 1 } ) = 0 . 9$ , and $\tilde { w } _ { 2 } = \mathrm { \cdots } \mathrm { { s t o p } ^ { \prime \prime } ( r e d ) }$ with $P ( \tilde { w } _ { 2 } ) = 0 . 1$ . The decision utility $U ( w , a )$ for actions $a _ { 1 } = \mathrm { G o }$ and $a _ { 2 } \ = \ { \mathsf { S t o p } }$ is given in Table II.

TABLE II: Utility function $U ( w , a )$ for autonomous vehicle decision example.
<table><tr><td>State w</td><td> $\mathrm { G o } \ ( a _ { 1 } )$ </td><td>Stop (a2)</td></tr><tr><td>w1 (green)</td><td>+10</td><td>-5</td></tr><tr><td>w2 (yellow)</td><td>-2</td><td>+1</td></tr><tr><td>W3 (red)</td><td>-100</td><td>+5</td></tr></table>

For each semantic class, the expected utility is $\mathbb { E } [ U | \tilde { w } _ { 1 } ] = ( 0 . 6 / 0 . 9 ) \cdot 1 0 + ( 0 . 3 / 0 . 9 ) \cdot ( - 2 ) = 6$ and $\mathbb { E } [ U | \tilde { w } _ { 2 } ] = ( 0 . 1 / 0 . 1 ) \cdot ( - 1 0 0 ) = - 1 0 0$ . Thus, the optimal action for $\tilde { w } _ { 1 }$ is Go $( 6 > - 5 )$ while for ${ \tilde { w } } _ { 2 }$ it is Stop $( - 1 0 0 < + 5 )$ . The Isoteleia Mapping induces two pragmatic classes: $\underline { w } _ { 1 } = \mathrm { ^ { * } G o ^ { , * } }$ containing $\tilde { w } _ { 1 }$ with $P ( \underline { { w } } _ { 1 } ) = 0 . 9$ , and $\underline { { w } } _ { 2 } = \mathrm { \cdots } \mathrm { s t o p } ^ { \mathrm { \prime \mathrm { \prime } } }$ containing ${ \tilde { w } } _ { 2 }$ with $P ( \underline { { w } } _ { 2 } ) = 0 . 1$

The syntactic entropy is $H ( W ) = - [ 0 . 6 \log { 0 . 6 + 0 . 3 \log { 0 . 3 + 0 . 1 \log { 0 . 1 } } } ] \approx 1 . 2 9 5$ bits. The semantic entropy is $H _ { s } ( \tilde { W } ) = - [ 0 . 9 \log { 0 . 9 } + 0 . 1 \log { 0 . 1 } ] \approx 0 . 4 6 9$ sebits. The pragmatic entropy is $H _ { p } ( \underline { { W } } ) = - [ 0 . 9 \log { 0 . 9 } + 0 . 1 \log { 0 . 1 } ] \approx 0 . 4 6 9$ prabits. In this example, $H _ { p } = H _ { s }$ because each semantic class maps to a distinct pragmatic class (the Isoteleia Mapping is injective), yet the hierarchy $H _ { p } = 0 . 4 6 9 < 1 . 2 9 5 = H ( W )$ still holds.

The syntactic entropy of 1.295 bits represents the uncertainty about the specific traffic light color observed. The semantic entropy of 0.469 sebits captures the uncertainty about whether the vehicle should “Proceed with caution” or “Stop” based on the meaning of the observation. The pragmatic entropy of 0.469 prabits reflects the uncertainty about the optimal terminal action, which ultimately determines the vehicle’s behavior. The reduction from syntactic to pragmatic entropy (from 1.295 to 0.469 bits) demonstrates that the vehicle does not need to distinguish between green and yellow lights to make the correct decision; it only needs to distinguish between “Go” and “Stop” situations.

This example demonstrates how the pragmatic entropy measure provides a principled way to quantify the minimum decision uncertainty that must be resolved through communication and control, thereby establishing the theoretical foundation for resource allocation and system design in pragmatic information systems.

## B. Pragmatic Joint Entropy and Conditional Entropy

Building upon the definition of pragmatic entropy for a single variable, we now extend the framework to multiple semantic/pragmatic variables. This extension is essential for analyzing the dependencies and information flows between different components of a pragmatic information system, such as the relationship between the transmitted pragmatic intention and the received pragmatic interpretation.

To formalize the joint behavior of two pragmatic sources, we first extend the Isoteleia Mapping to the joint and conditional cases. Let W and V be two pragmatic alphabets, with associated semantic alphabets $\tilde { \mathcal W }$ and $\tilde { \mathcal { V } } ,$ respectively. Let the corresponding syntactic alphabets be W and V, with a joint distribution $P ( W , V )$ .

Definition 6 (Joint Isoteleia Mapping). The joint Isoteleia Mapping

$$
e _ { w v } : \underline { { { \mathcal { W } } } } \times \underline { { { \mathcal { V } } } } \longrightarrow 2 ^ { \tilde { \mathcal { W } } \times \tilde { \mathcal { V } } }\tag{35}
$$

is defined by its preimage: for each pair of pragmatic symbols $( { \underline { { w } } } , { \underline { { v } } } ) \in { \underline { { \mathcal { W } } } } \times { \underline { { \mathcal { V } } } } .$ , the image is the set of all semantic pairs $( \tilde { w } , \tilde { v } )$ that jointly induce the same optimal terminal actions:

$$
\begin{array} { r } { e _ { w v } ( \underline { { w } } , \underline { { v } } ) \triangleq \left\{ ( \tilde { w } , \tilde { v } ) \in \tilde { \mathcal { W } } \times \tilde { \mathcal { V } } \bigg | \operatorname * { m a x } _ { a \in A } \mathbb { E } [ U ( W , a ) | \tilde { w } ] = \underline { { w } } , \mathrm { ~ a r g \operatorname* { m a x } _ { \boldsymbol { b } \in \mathcal { B } } \mathbb { E } } [ U ( V , \boldsymbol { b } ) | \tilde { v } ] = \underline { { v } } \right\} \in V , } \end{array}\tag{36}
$$

This mapping partitions the joint semantic space $\tilde { \mathcal { W } } \times \tilde { \mathcal { V } }$ into equivalence classes according to the pair of optimal actions.

Definition 7 (Conditional Isoteleia Mapping). Given a specific pragmatic symbol $w \in \underline { { \mathcal W } }$ , the conditional Isoteleia Mapping

$$
e _ { v | w } ( \underline { { v } } | \underline { { w } } ) : \underline { { \mathcal { V } } } \longrightarrow 2 ^ { \tilde { \mathcal { V } } }\tag{37}
$$

is defined as the set of semantic symbols v˜ that, together with the given w, lead to the optimal action v:

$$
e _ { v | w } ( \underline { { v } } | \underline { { w } } ) \triangleq \left\{ \widetilde { v } \in \widetilde { \mathcal { V } } \bigg | \arg \operatorname* { m a x } _ { b \in \mathcal { B } } \mathbb { E } [ U ( V , b ) | \widetilde { v } , \underline { { w } } ] = \underline { { v } } \right\} .\tag{38}
$$

This mapping captures the pragmatic equivalence of the second variable conditioned on the first pragmatic class.

The joint and conditional synonymous mappings have been established in the semantic information theory literature [21, 22]. The joint synonymous mapping partitions the syntactic product space according to meaning identity, while the conditional synonymous mapping partitions the syntactic space of one variable given the semantic class of the other. We will leverage these existing definitions in the subsequent entropy formulations.

With the joint and conditional isoteleia mappings defined, we can now introduce the corresponding entropy measures.

Definition 8 (Pragmatic Joint Entropy). Let (W, V ) be a pair of pragmatic random variables derived from the syntactic pair $( W , V )$ via the joint isoteleia mapping $e _ { w v }$ . The pragmatic joint entropy is defined as the Shannon entropy of the joint pragmatic distribution:

$$
H _ { p } ( \underline { { W } } , \underline { { V } } ) \triangleq - \sum _ { w \in \mathcal { W } } \sum _ { v \in \mathcal { V } } P ( \underline { { w } } , \underline { { v } } ) \log P ( \underline { { w } } , \underline { { v } } ) ,\tag{39}
$$

where the joint pragmatic probabilities are obtained by aggregating over all semantic and syntactic fibers:

$$
P ( \underline { { w } } , \underline { { v } } ) = \sum _ { \substack { ( \tilde { w } , \tilde { v } ) \in e _ { u v } ( \underline { { w } } , v ) } } P ( \tilde { w } , \tilde { v } ) = \sum _ { \substack { ( \tilde { w } , \tilde { v } ) \in e _ { u v } ( \underline { { w } } , v ) } } \sum _ { \substack { ( w , v ) \in f _ { u v } ( \tilde { w } , \tilde { v } ) } } P ( w , v ) ,\tag{40}
$$

with $f _ { u v }$ denoting the joint Synonymous Mapping.

Definition 9 (Pragmatic Conditional Entropy). There are two natural variants of pragmatic conditional entropy, depending on whether the conditioning variable is syntactic or pragmatic.

(a) Pragmatic conditional entropy of V given the syntactic variable W:

$$
H _ { p } ( \underline { { V } } | W ) \triangleq - \sum _ { w \in \mathcal W } \sum _ { \underline { { v } } \in \underline { { \mathcal V } } } P ( w ) P ( \underline { { v } } | w ) \log P ( \underline { { v } } | w ) ,\tag{41}
$$

where $\begin{array} { r } { P ( \underline { { v } } | w ) = \sum _ { \tilde { v } \in e ( \underline { { v } } ) } P ( \tilde { v } | w ) } \end{array}$ is the pragmatic distribution of V conditioned on a specific syntactic observation w.

(b) Pragmatic conditional entropy of V given the pragmatic variable W:

$$
H _ { p } ( \underline { { V } } | \underline { { W } } ) \triangleq - \sum _ { \underline { { w } } \in \underline { { W } } } \sum _ { \underline { { v } } \in \underline { { \mathcal { V } } } } P ( \underline { { w } } , \underline { { v } } ) \log P ( \underline { { v } } | \underline { { w } } ) ,\tag{42}
$$

where $P ( \underline { { v } } | \underline { { w } } ) = P ( \underline { { w } } , \underline { { v } } ) / P ( \underline { { w } } )$

Similarly, one can define $H _ { p } ( \underline { { W } } | V )$ and $H _ { p } ( \underline { { W } } | \underline { { V } } )$ by symmetry. The choice between conditioning on syntactic or pragmatic variables depends on the application: $H _ { p } ( \underline { { V } } | W )$ measures the residual decision uncertainty after observing the exact syntactic signal, while $H _ { p } ( \underline { { V } } | \underline { { W } } )$ measures the uncertainty of one decision given the other decision.

Analogous to semantic entropy, pragmatic entropy does not satisfy the exact additive chain rule of classical entropy; instead, it satisfies a series of inequalities that reflect the coarsening of information. We first establish the chain rule for a pair of pragmatic variables.

Theorem 1 (Chain Rule for Binary Pragmatic Entropy). For any two pragmatic variables W and V derived from syntactic variables W and V via the isoteleia and synonymous mappings, the following chain of inequalities holds:

$$
H _ { p } ( \underline { { W } } ) + H _ { p } ( \underline { { V } } | W ) \le H _ { p } ( \underline { { W } } , \underline { { V } } ) \le H ( W ) + H _ { p } ( \underline { { V } } | W ) ,\tag{43}
$$

and symmetrically,

$$
H _ { p } ( \underline { { V } } ) + H _ { p } ( \underline { { W } } | V ) \le H _ { p } ( \underline { { W } } , \underline { { V } } ) \le H ( V ) + H _ { p } ( \underline { { W } } | V ) .\tag{44}
$$

Proof: Let $g _ { W } : \underline { { \mathcal { W } } } \to 2 ^ { \mathcal { W } }$ and $g _ { V } : \underline { { \mathcal { V } } } \to 2 ^ { \mathcal { V } }$ be the reification mappings, so that ${ \underline { { W } } } = g _ { W } ^ { - 1 } ( W )$ and $\underline { { { V } } } = g _ { V } ^ { - 1 } ( V )$ are deterministic functions of W and V , respectively.

For any $w \in { \underline { { w } } }$ , define its preimage set

$$
\mathcal { W } _ { \underline { { w } } } \triangleq \{ w \in \mathcal { W } : g _ { W } ( w ) = \underline { { w } } \} ,
$$

and similarly $\mathcal { V } _ { \underline { { v } } } \triangleq \{ v \in \mathcal { V } : g _ { V } ( v ) = \underline { { v } } \}$ . The aggregated probabilities are

$$
p ( \underline { { w } } ) = \sum _ { w \in \mathcal { W } _ { \underline { { w } } } } p ( w ) , \qquad p ( \underline { { w } } , \underline { { v } } ) = \sum _ { w \in \mathcal { W } _ { \underline { { w } } } } \sum _ { v \in \mathcal { V } _ { \underline { { v } } } } p ( w , v ) .
$$

For a fixed w, the conditional distribution of V given ${ \underline { { W } } } = { \underline { { w } } }$ is

$$
p ( \underline { { v } } \mid \underline { { w } } ) = \frac { p ( \underline { { w } } , \underline { { v } } ) } { p ( \underline { { w } } ) } = \sum _ { w \in \mathcal { W } _ { \underline { { w } } } } \frac { p ( w ) } { p ( \underline { { w } } ) } \cdot p ( \underline { { v } } \mid w ) ,\tag{45}
$$

where $\begin{array} { r } { p ( \underline { { v } } \mid w ) \triangleq \sum _ { v \in \mathcal { V } _ { v } } p ( v \mid w ) } \end{array}$ . Thus, for each $\underline { w } ,$ , the distribution $p ( \cdot \mid { \underline { { w } } } )$ is a mixture of the distributions $\{ p ( \cdot \mid w ) : w \in \mathcal { W } _ { \underline { { w } } } \}$ with weights $\frac { p ( w ) } { p ( \underline { { w } } ) }$

Since the Shannon entropy is concave,

$$
H \big ( p ( \cdot \mid \underline { { w } } ) \big ) \geq \sum _ { w \in \mathcal { W } _ { \underline { { w } } } } \frac { p ( w ) } { p ( \underline { { w } } ) } H \big ( p ( \cdot \mid w ) \big ) .\tag{46}
$$

Multiplying by $p ( \underline { { w } } )$ and summing over w:

$$
\sum _ { \underline { { w } } } p ( \underline { { w } } ) H \big ( p ( \cdot \mid \underline { { w } } ) \big ) \geq \sum _ { \underline { { w } } } \sum _ { w \in \mathcal { W } _ { \underline { { w } } } } p ( w ) H \big ( p ( \cdot \mid w ) \big ) .\tag{47}
$$

The left-hand side is $H _ { p } ( \underline { { V } } \mid \underline { { W } } )$ , and the right-hand side is $H _ { p } ( \underline { { V } } \mid W )$ . Hence,

$$
H _ { p } ( \underline { { { V } } } \mid \underline { { { W } } } ) \geq H _ { p } ( \underline { { { V } } } \mid W ) .\tag{48}
$$

Since W and $\underline V$ are ordinary random variables, the classical chain rule applies:

$$
H _ { p } ( \underline { { W } } , \underline { { V } } ) = H _ { p } ( \underline { { W } } ) + H _ { p } ( \underline { { V } } \mid \underline { { W } } ) .\tag{49}
$$

Substituting (48) into (49) yields the left inequality of (43):

$$
H _ { p } ( \underline { { W } } , \underline { { V } } ) \geq H _ { p } ( \underline { { W } } ) + H _ { p } ( \underline { { V } } \mid W ) .\tag{50}
$$

For the right inequality of (43), since W is a function of W,

$$
H _ { p } ( { \underline { { W } } } , { \underline { { V } } } ) = H ( { \underline { { W } } } , { \underline { { V } } } ) \leq H ( W , { \underline { { V } } } ) = H ( W ) + H ( { \underline { { V } } } \mid W ) = H ( W ) + H _ { p } ( { \underline { { V } } } \mid W ) ,\tag{51}
$$

which proves the right inequality.

The symmetric chain (44) follows by interchanging $W$ and $V .$ , noting that $\underline { { V } }$ is a function of $V ,$ and applying the same mixture-entropy argument to $p ( \cdot \mid \underline { { v } } )$ . This completes the proof. □

The chain rule (Eq. 43) reveals that pragmatic joint entropy lies between two bounds: the lower bound is the sum of the marginal pragmatic entropy of $W$ and the pragmatic uncertainty of V given the syntactic observation $W ;$ the upper bound is the sum of the syntactic entropy of W and the same pragmatic conditional term. This intermediate position reflects the fact that pragmatic abstraction reduces uncertainty, but not as much as semantic abstraction.

Theorem 2 (Hierarchy of Binary Joint Entropies). For any two syntactic variables W and V with their associated semantic variables $\tilde { W } , \tilde { V }$ and pragmatic variables W, V , the following inequality holds:

$$
H _ { p } ( \underline { { W } } , \underline { { V } } ) \le H _ { s } ( \tilde { W } , \tilde { V } ) \le H ( W , V ) .\tag{52}
$$

Moreover, equality in the first inequality holds if and only if the isoteleia mapping is injective on the joint semantic space (i.e., no two distinct semantic pairs are merged into the same pragmatic pair). Equality in the second holds if and only if the joint synonymous mapping is injective (no two distinct syntactic pairs share the same semantic meaning).

Proof: We prove the two inequalities separately.

First inequality: $H _ { p } ( \underline { { W } } , \underline { { V } } ) \le H _ { s } ( \tilde { W } , \tilde { V } )$

By definition, the pragmatic variables are deterministic functions of the semantic variables via the joint Isoteleia mapping:

$$
( \underline { { W } } , \underline { { V } } ) = e _ { u v } ^ { - 1 } ( \tilde { W } , \tilde { V } ) ,\tag{53}
$$

where $e _ { u v } ^ { - 1 }$ is the inverse of the isoteleia mapping that collapses each semantic equivalence class to a single pragmatic symbol. Since (W, V ) is a deterministic function of $( \tilde { W } , \tilde { V } )$ , the joint entropy of a function of a random variable cannot exceed the joint entropy of the original variable. Applying the data processing inequality for entropy:

$$
H _ { p } ( \underline { { W } } , \underline { { V } } ) = H ( e _ { u v } ^ { - 1 } ( \tilde { W } , \tilde { V } ) ) \leq H ( \tilde { W } , \tilde { V } ) = H _ { s } ( \tilde { W } , \tilde { V } ) .\tag{54}
$$

Equality holds if and only if the mapping is injective, i.e., no two distinct semantic pairs are merged.

Second inequality: $H _ { s } ( \tilde { W } , \tilde { V } ) \leq H ( W , V )$

Similarly, the semantic variables are deterministic functions of the syntactic variables via the joint synonymous mapping:

$$
( \tilde { W } , \tilde { V } ) = f _ { u v } ^ { - 1 } ( W , V ) ,\tag{55}
$$

where $f _ { u v } ^ { - 1 }$ collapses each syntactic equivalence class to a single semantic symbol. Hence,

$$
H _ { s } ( \tilde { W } , \tilde { V } ) = H ( f _ { u v } ^ { - 1 } ( W , V ) ) \leq H ( W , V ) .\tag{56}
$$

Equality holds if and only if the joint synonymous mapping is injective.

Combining both inequalities yields the desired hierarchy.

We now extend this result to sequences of pragmatic variables. To do so, we first define the sequential extension of the joint reification mapping.

Definition 10 (Sequential Joint Reification Mapping). Let g<sub>WV</sub> : $\mathcal { W } \times \underline { { \mathcal { V } } } \to 2 ^ { \mathcal { W } \times \mathcal { V } }$ be the joint Reification Mapping that partitions the joint syntactic space $\mathcal { W } \times \mathcal { V }$ into pragmatic equivalence

classes. Its n-th extension

$$
g _ { W V } ^ { n } : { \underline { { \mathcal { W } } } } ^ { n } \times { \underline { { \mathcal { V } } } } ^ { n } \longrightarrow 2 ^ { { \mathcal { W } } ^ { n } \times { \mathcal { V } } ^ { n } }\tag{57}
$$

is defined by

$$
g _ { W V } ^ { n } ( { \underline { { w } } } ^ { n } , { \underline { { v } } } ^ { n } ) \triangleq \prod _ { i = 1 } ^ { n } g _ { W V } ( { \underline { { w } } } _ { i } , { \underline { { v } } } _ { i } ) ,\tag{58}
$$

where $\underline { w } ^ { n } = ( \underline { w } _ { 1 } , \dots , \underline { w } _ { n } )$ and $\underline { { v } } ^ { n } = ( \underline { { v } } _ { 1 } , \ldots , \underline { { v } } _ { n } )$ are pragmatic sequences, and the product denotes the Cartesian product of sets. In other words, a pragmatic sequence pair $( { \underline { { w } } } ^ { n } , { \underline { { v } } } ^ { n } )$ corresponds to the set of all syntactic sequence pairs $( w ^ { n } , v ^ { n } )$ such that for every position $i ,$ $( w _ { i } , v _ { i } ) \in g _ { W V } ( \underline { { w } } _ { i } , \underline { { v } } _ { i } )$

The marginal sequential pragmatic variables $W ^ { n }$ and $\underline { { V } } ^ { n }$ are then obtained by projecting from the joint pragmatic sequence. The probability of a pragmatic sequence pair is given by

$$
P ( \underline { { w } } ^ { n } , \underline { { v } } ^ { n } ) = \sum _ { ( w ^ { n } , v ^ { n } ) \in g _ { W V } ^ { n } ( \underline { { w } } ^ { n } , \underline { { v } } ^ { n } ) } P ( w ^ { n } , v ^ { n } ) .\tag{59}
$$

Theorem 3 (Chain Rule for Sequential Pragmatic Entropy). Let $( \underline { { W } } _ { 1 } , \underline { { W } } _ { 2 } , \dots , \underline { { W } } _ { n } )$ be a sequence of pragmatic variables derived from the syntactic sequence $( W _ { 1 } , W _ { 2 } , \ldots , W _ { n } )$ via the sequential joint Reification Mapping. Define the sequential partial entropy

$$
\tilde { H } _ { p } ( \underline { { W } } _ { 1 } ^ { m } , W _ { m + 1 } ^ { n } ) \triangleq \sum _ { k = 1 } ^ { m } H _ { p } ( \underline { { W } } _ { k } | \underline { { W } } _ { 1 } ^ { k - 1 } ) + \sum _ { k = m + 1 } ^ { n } H _ { p } ( W _ { k } | \underline { { W } } _ { 1 } ^ { m } , W _ { m + 1 } ^ { k - 1 } ) ,\tag{60}
$$

where the notation $\underline { W } _ { 1 } ^ { m }$ denotes the block $( \underline { { W } } _ { 1 } , \dots , \underline { { W } } _ { m } )$ , and $W _ { m + 1 } ^ { n }$ denotes the block $( W _ { m + 1 } , \ldots , W _ { n } )$ . Then the following chain of inequalities holds:

$$
\sum _ { k = 1 } ^ { n } H _ { p } ( { \underline { { W } } } _ { k } | W _ { 1 } ^ { k - 1 } ) \leq H _ { p } ( W _ { 1 } ^ { n } ) \leq { \tilde { H } } _ { p } ( { \underline { { W } } } _ { 1 } ^ { n - 1 } , W _ { n } ) \leq \cdots \leq { \tilde { H } } _ { p } ( W _ { 1 } , W _ { 2 } ^ { n } ) \leq H _ { p } ( W _ { 1 } ^ { n } ) ,\tag{61}
$$

where the last term $H _ { p } ( W _ { 1 } ^ { n } )$ denotes the syntactic entropy of the full sequence.

Proof: The proof follows by iteratively applying the binary chain rule to each adjacent pair in the sequence. Starting from the full pragmatic joint entropy $H _ { p } ( \underline { { W } } _ { 1 } ^ { n } )$ , we use the binary chain rule to bound it in terms of the entropy of the first $n - 1$ variables and the last variable, either conditioned on the syntactic or pragmatic prefix. By gradually replacing pragmatic conditioning with syntactic conditioning, we obtain the chain of inequalities. The details are analogous to the semantic sequential chain rule established in [21, 22], with the substitution of pragmatic entropies. □

Theorem 4 (Hierarchy of Sequential Joint Entropies). Let $W ^ { n } = ( W _ { 1 } , \ldots , W _ { n } )$ be a syntactic sequence, with associated semantic sequence $\tilde { W } ^ { n }$ and pragmatic sequence $W ^ { n }$ obtained by applying the Synonymous and Isoteleia mappings element-wise (or, equivalently, by the sequential joint Reification Mapping). Then:

$$
H _ { p } ( \underline { { W } } ^ { n } ) \ \leq \ H _ { s } ( \tilde { W } ^ { n } ) \ \leq \ H ( W ^ { n } ) ,\tag{62}
$$

where $H _ { p } ( \underline { { W } } ^ { n } ) , ~ H _ { s } ( \tilde { W } ^ { n } )$ , and $H ( W ^ { n } )$ denote the joint entropies of the entire sequences.

Proof: The proof follows directly from the element-wise deterministic mappings. Since the sequential joint reification mapping is defined as the product of the individual mappings, the entire pragmatic sequence $\underline { W } ^ { n }$ is a deterministic function of the semantic sequence $\tilde { W } ^ { n }$ , and $\tilde { W } ^ { n }$ is a deterministic function of the syntactic sequence $W ^ { n }$ . Therefore:

$$
H _ { p } ( \underline { { W } } ^ { n } ) = H ( e ^ { - n } ( \tilde { W } ^ { n } ) ) \leq H ( \tilde { W } ^ { n } ) = H _ { s } ( \tilde { W } ^ { n } ) ,\tag{63}
$$

and

$$
H _ { s } ( \tilde { W } ^ { n } ) = H ( f ^ { - n } ( W ^ { n } ) ) \leq H ( W ^ { n } ) .\tag{64}
$$

Combining these yields the desired hierarchy.

Corollary 2 (Hierarchy of Sequential Conditional Entropies). For any m with $1 \leq m \leq n$ , the sequential partial entropies also satisfy the same ordering:

$$
\tilde { H } _ { p } ( \underline { { { W } } } _ { 1 } ^ { m } , W _ { m + 1 } ^ { n } ) ~ \leq ~ \tilde { H } _ { s } ( \tilde { W } _ { 1 } ^ { m } , W _ { m + 1 } ^ { n } ) ~ \leq ~ H ( W ^ { n } ) ,\tag{65}
$$

where the partial entropies are defined as in the chain rule $( E q . \ ( 6 0 ) )$ . This follows because the partial entropies are linear combinations of conditional entropies, each of which obeys the single-variable hierarchy.

Remark 2. The hierarchy theorems reveal a layered structure of uncertainty: syntactic entropy captures all symbol-level distinctions; semantic entropy removes meaningless syntactic variations; and pragmatic entropy further removes decision-irrelevant semantic distinctions. Each successive coarsening reduces the joint entropy, providing a principled foundation for multi-level source coding and compression in pragmatic communication systems.

Example 2 (Multi-Vehicle Autonomous Driving Scenario).

We now present a concrete numerical example to illustrate the computation and interpretation ofpragmatic joint and conditional entropies, with particular emphasis on distinguishing pragmatic entropy from semantic entropy. Consider two autonomous vehicles, Vehicle A and Vehicle B, approaching an intersection. Each vehicle must decide whether to “Go” or “Stop” based on its observed traffic light state and, potentially, the state of the other vehicle.

Let the syntactic state of Vehicle A be $W \in \{ w _ { 1 } , w _ { 2 } , w _ { 3 } , w _ { 4 } \}$ representing green, yellow, red-short, and red-long, respectively; Vehicle B’s state V has the same interpretation. The joint probability distribution $P ( W , V )$ is given in Table III, with marginals $P ( w _ { 1 } ) = 0 . 2 3 , ~ P ( w _ { 2 } ) =$ 0.22, $P ( w _ { 3 } ) = 0 . 2 5 , \ P ( w _ { 4 } ) = 0 . 3 0$ , and similarly for V.

TABLE III: Joint probability distribution $P ( W , V )$ for two vehicles.
<table><tr><td>P(W, V)</td><td> $v _ { 1 }$  (green)  $v _ { 2 }$  (yellow)</td><td> $v _ { 3 }$  (red-short)</td><td> $v _ { 4 }$  (red-long)</td></tr><tr><td>w1 (green)</td><td>0.12 0.06</td><td>0.03</td><td>0.02</td></tr><tr><td> $w _ { 2 }$  (yellow)</td><td>0.06 0.10</td><td>0.04</td><td>0.02</td></tr><tr><td> $w _ { 3 }$  (red-short)</td><td>0.03 0.04</td><td>0.12</td><td>0.06</td></tr><tr><td> $w _ { 4 }$  (red-long)</td><td>0.02 0.02</td><td>0.06</td><td>0.20</td></tr></table>

The synonymous mapping groups the syntactic states into three semantic classes: $\tilde { w } _ { 1 } = \tilde { v } _ { 1 } =$ “Proceed with caution” (green and yellow) with probability 0.45, $\tilde { w } _ { 2 } = \tilde { v } _ { 2 } = { } ^ { \ast \mathrm { { \cdot } } } \mathrm { S t o p \mathrm { - \ s h o r t ^ { \prime \prime } } }$ (redshort) with probability 0.25, and $\tilde { w } _ { 3 } = \tilde { v } _ { 3 } = { } ^ { \ast } \mathrm { S t o p \mathrm { - l o n g } ^ { \ast } }$ (red-long) with probability 0.30. Under a safety-first utility, the optimal action for $\tilde { w } _ { 1 }$ is “Go”, while for both ${ \tilde { w } } _ { 2 }$ and ${ \tilde { w } } _ { 3 }$ it is “Stop”. The isoteleia mapping therefore merges the two stop classes into a single pragmatic class: $\underline { w } _ { 1 } = \mathrm { ^ { * } G o ^ { , * } }$ containing $\tilde { w } _ { 1 }$ with probability 0.45, and $\underline { { w } } _ { 2 } = \mathrm { \cdots } \mathrm { s t o p } ^ { 3 }$ containing ${ \tilde { w } } _ { 2 }$ and ${ \tilde { w } } _ { 3 }$ with probability 0.55. The same mapping applies to Vehicle B.

Aggregating the syntactic probabilities yields the semantic joint distribution $P ( \tilde { W } , \tilde { V } )$ (Table IV) and the pragmatic joint distribution P(W, V ) (Table V). The syntactic joint entropy is $H ( W , V ) \approx$ 3.6318 bits; the semantic joint entropy is $H _ { s } ( \tilde { W } , \tilde { V } ) \approx 2 . 7 5 6 5$ sebits; and the pragmatic joint entropy is $H _ { p } ( \underline { { W } } , \underline { { V } } ) \approx 1 . 7 5 0 9$ prabits, confirming the hierarchy $H _ { p } \leq H _ { s } \leq H$ . Marginal entropies are $H ( \tilde { W } ) \approx 1 . 5 3 9 0$ bits and $H _ { p } ( \underline { { W } } ) \approx 0 . 9 9 2 8$ prabits.

The pragmatic conditional entropy is $H _ { p } ( \underline { { V } } | \underline { { W } } ) = H _ { p } ( \underline { { W } } , \underline { { V } } ) - H _ { p } ( \underline { { W } } ) = 0 . 7 5 8 1$ prabits, while

TABLE IV: Joint semantic distribution $P ( \tilde { W } , \tilde { V } )$
<table><tr><td> $P ( \tilde { W } , \tilde { V } )$ </td><td> $\tilde { v } _ { 1 } \ ( \mathrm { P r o c e d } )$ </td><td> $\tilde { v } _ { 2 } \ ( \mathrm { S t o p - s h o r t } )$ </td><td> ${ \tilde { v } } _ { 3 } \ ( \mathrm { S t o p \mathrm { - l o n g } } )$ </td></tr><tr><td> $\tilde { w } _ { 1 }$  (Proceed)</td><td>0.34</td><td>0.07</td><td>0.04</td></tr><tr><td> $\tilde { w } _ { 2 }$  (Stop-short)</td><td>0.07</td><td>0.12</td><td>0.06</td></tr><tr><td> ${ \tilde { w } } _ { 3 }$  (Stop-long)</td><td>0.04</td><td>0.06</td><td>0.20</td></tr></table>

TABLE V: Joint pragmatic distribution P(W, V ).
<table><tr><td> $P ( \underline { { W } } , \underline { { V } } )$ </td><td> $\underline { { v } } _ { 1 } ~ ( \mathrm { G o ) }$ </td><td> $\underline { { v } } _ { 2 } \ ( \mathrm { S t o p } )$ </td></tr><tr><td> $\underline { w } _ { 1 } ~ ( \mathrm { G o } )$ </td><td>0.34</td><td>0.11</td></tr><tr><td> $\underline { { w } } _ { 2 } \ ( \mathrm { S t o p } )$ </td><td>0.11</td><td>0.44</td></tr></table>

$H _ { p } ( \underline { { V } } | W ) \approx 0 . 9 1 2 3$ prabits. The binary chain rule is verified: $H _ { p } ( \underline { { W } } ) + H _ { p } ( \underline { { V } } | W ) = 1 . 9 0 5 1 \geq$ $H _ { p } ( \underline { { W } } , \underline { { V } } ) = 1 . 7 5 0 9$ , and $H _ { p } ( \underline { { W } } , \underline { { V } } ) = 1 . 7 5 0 9 \leq H ( W ) + H _ { p } ( \underline { { V } } | W ) = 1 . 9 8 5 6 + 0 . 9 1 2 3 =$ 2.8979. These results (summarized in Table VI) demonstrate that pragmatic abstraction, by merging semantically distinct but action-equivalent states, reduces joint uncertainty beyond semantic abstraction.

TABLE VI: Comparison of joint and conditional entropies.
<table><tr><td>Entropy Measure</td><td>Value (bits/sebits/prabits)</td></tr><tr><td>Syntactic Joint Entropy  $H ( W , V )$ </td><td>3.6318</td></tr><tr><td>Semantic Joint Entropy  $H _ { s } ( \tilde { W } , \tilde { V } )$ </td><td>2.7565</td></tr><tr><td>Pragmatic Joint Entropy  $H _ { p } ( \underline { { W } } , \underline { { V } } )$ </td><td>1.7509</td></tr><tr><td>Pragmatic Marginal Entropy  $H _ { p } ( \underline { { W } } )$ </td><td>0.9928</td></tr><tr><td>Pragmatic Conditional Entropy  $H _ { p } ( \underline { { V } } | \underline { { W } } )$ </td><td>0.7581</td></tr><tr><td>Pragmatic Conditional Entropy  $H _ { p } ( \underline { { V } } | W )$ </td><td>0.9123</td></tr></table>

This example illustrates the key insight: pragmatic entropy is strictly less than semantic entropy when the isoteleia mapping merges distinct semantic classes that share the same optimal action. The two types of “stop” (short red and long red) are semantically distinct but pragmatically equivalent, resulting in a joint pragmatic entropy (1.7509 prabits) that is significantly lower than the joint semantic entropy (2.7565 sebits). This reduction quantifies the decision-irrelevant information that can be discarded in a pragmatic communication system.

## IV. PRAGMATIC RELATIVE ENTROPY AND PRAGMATIC MUTUAL INFORMATION

We now introduce the information-theoretic analogues of relative entropy and mutual information at the pragmatic level. These measures quantify the divergence between probability distributions and the statistical dependence between variables, respectively, when the comparison and dependency are evaluated at the level of terminal actions rather than symbols or meanings.

## A. Pragmatic Relative Entropy

In classical information theory, the relative entropy (Kullback-Leibler divergence) [45] between two probability mass functions $p ( w )$ and $q ( w )$ over a syntactic alphabet W is defined as $\begin{array} { r } { D ( p \| q ) = \sum _ { w } p ( w ) \log \frac { p ( w ) } { q ( w ) } } \end{array}$ . Semantic information theory [21, 22] extends this by introducing semantic relative entropies that aggregate probabilities over synonymous sets, thereby measuring divergence at the meaning level. In pragmatic information theory, we further aggregate over pragmatic equivalence classes induced by the isoteleia mapping, so that the divergence is evaluated with respect to the terminal actions that the variables induce. This yields three forms of pragmatic relative entropy: full, partial (Type I), and partial (Type II), which correspond to different ways of combining the pragmatic aggregation with the syntactic or semantic distributions.

Let W be a syntactic alphabet with two probability mass functions $p ( w )$ and $q ( w ) , w \in \mathcal { W }$ Let $\tilde { \mathcal W }$ be the associated semantic alphabet induced by the synonymous mapping $f : \tilde { \mathcal { W } } \to 2 ^ { \mathcal { W } }$ and let W be the pragmatic alphabet induced by the isoteleia mapping $e : \underline { { \mathcal { W } } } \to 2 ^ { \tilde { \mathcal { W } } }$ according to a given utility function $U : \mathcal { W } \times \mathcal { A }  \mathbb { R }$ . The pragmatic partition of W is given by the equivalence relation $\sim _ { p }$ defined in Eq. (15), which groups syntactic symbols that lead to the same optimal action. Equivalently, the pragmatic classes $w \in { \underline { { w } } }$ are the atoms of the partition $\Pi _ { p r a g } = \mathcal { W } / \sim _ { p }$ . For each pragmatic class w, its probability under a distribution $p$ is defined as

$$
p _ { p } ( \underline { { w } } ) \triangleq \sum _ { w \in g ( \underline { { w } } ) } p ( w ) ,\tag{66}
$$

where $g = f \circ e : \underline { { \mathcal { W } } }  2 ^ { \mathcal { W } }$ is the reification mapping. Similarly, we define $q _ { p } ( \underline { { w } } )$ for the distribution q.

Definition 11 (Pragmatic Relative Entropy). Given two probability mass functions $p ( w )$ and $q ( w )$ on the syntactic alphabet W, and the pragmatic partition induced by the reification mapping g, we define the following three forms of pragmatic relative entropy:

(a) Full Pragmatic Relative Entropy:

$$
D _ { p } ( p _ { p } \| q _ { p } ) \triangleq \sum _ { { \underline { { w } } } \in { \underline { { w } } } } p _ { p } ( { \underline { { w } } } ) \log { \frac { p _ { p } ( { \underline { { w } } } ) } { q _ { p } ( { \underline { { w } } } ) } } .\tag{67}
$$

(b) Partial Pragmatic Relative Entropy (Type I):

$$
D _ { p } ( p _ { p } \| q ) \triangleq \sum _ { \underline { { w } } \in \underline { { \mathcal { W } } } } \sum _ { w \in g ( \underline { { w } } ) } p ( w ) \log \frac { p _ { p } ( \underline { { w } } ) } { q ( w ) } .\tag{68}
$$

(c) Partial Pragmatic Relative Entropy (Type II):

$$
D _ { p } ( p \| q _ { p } ) \triangleq \sum _ { \underline { { w } } \in \underline { { \mathcal { W } } } } \sum _ { w \in g ( \underline { { w } } ) } p ( w ) \log \frac { p ( w ) } { q _ { p } ( \underline { { w } } ) } .\tag{69}
$$

These definitions mirror the semantic relative entropies [21, 22] but with the aggregation performed over pragmatic classes instead of semantic classes. The full pragmatic relative entropy compares the pragmatic marginal distributions directly. The partial Type I compares the aggregated pragmatic distribution of $p$ against the fine-grained syntactic distribution of $q ,$ while the partial Type II compares the fine-grained syntactic distribution of $p$ against the aggregated pragmatic distribution of $q .$ As in the semantic case, $D _ { p } ( p \vert \vert q _ { p } )$ may be negative; in practice, one can take its positive part $( D _ { p } ( p \| q _ { p } ) ) ^ { + }$

We now establish the fundamental properties of pragmatic relative entropy. These properties are analogous to those of semantic relative entropy, with the pragmatic partition replacing the semantic partition.

Theorem 5 (Pragmatic Information Inequality). Let p and q be two probability mass functions on W. Given the Reification Mapping $^ { g , }$ the following inequalities hold:

$$
\left\{ \begin{array} { l l } { D _ { p } ( p _ { p } \| q _ { p } ) \geq 0 , } \\ { D _ { p } ( p _ { p } \| q ) \geq D _ { p } ( p _ { p } \| p ) , } \\ { D _ { p } ( p \| q _ { p } ) \geq D _ { p } ( p \| p _ { p } ) , } \end{array} \right.\tag{70}
$$

with equality in the first inequality if and only if $p _ { p } ( \underline { { w } } ) = q _ { p } ( \underline { { w } } )$ for all $w \in { \underline { { W } } } ,$ ; equality in the second if and only if $p ( w ) = q ( w )$ for all w within each pragmatic class; equality in the third if and only if $p _ { p } ( \underline { { w } } ) = q _ { p } ( \underline { { w } } )$ for all w.

Proof: For the first inequality, by Jensen’s inequality applied to the negative of $D _ { p } ( p _ { p } \| q _ { p } )$ we have

$$
- D _ { p } ( p _ { p } \| q _ { p } ) = \sum _ { \underline { { w } } } p _ { p } ( \underline { { w } } ) \log \frac { q _ { p } ( \underline { { w } } ) } { p _ { p } ( \underline { { w } } ) }\tag{71}
$$

$$
\leq \log \sum _ { \underline { { w } } } p _ { p } ( \underline { { w } } ) \frac { q _ { p } ( \underline { { w } } ) } { p _ { p } ( \underline { { w } } ) } = \log \sum _ { \underline { { w } } } q _ { p } ( \underline { { w } } ) = 0 .\tag{72}
$$

Thus $D _ { p } ( p _ { p } \| q _ { p } ) \geq 0$ , with equality iff $p _ { p } = q _ { p }$

For the second inequality, we compute

$$
D _ { p } ( p _ { p } \| q ) - D _ { p } ( p _ { p } \| p ) = \sum _ { \underline { { w } } } \sum _ { w \in g ( \underline { { w } } ) } p ( w ) \log \frac { p _ { p } ( \underline { { w } } ) } { q ( w ) } - \sum _ { \underline { { w } } } \sum _ { w \in g ( \underline { { w } } ) } p ( w ) \log \frac { p _ { p } ( \underline { { w } } ) } { p ( w ) }\tag{73}
$$

$$
= \sum _ { { \underline { { w } } } } \sum _ { w \in g ( { \underline { { w } } } ) } p ( w ) \log { \frac { p ( w ) } { q ( w ) } }\tag{74}
$$

$$
= D ( p \| q ) \geq 0 .\tag{75}
$$

The last inequality is the classical Gibbs inequality, with equality iff $p = q$ pointwise. Hence $D _ { p } ( p _ { p } \| q ) \ge D _ { p } ( p _ { p } \| p )$

For the third inequality,

$$
D _ { p } ( p | | q _ { p } ) - D _ { p } ( p | | p _ { p } ) = \sum _ { { \underline { { w } } } } \sum _ { w \in g ( { \underline { { w } } } ) } p ( w ) \log { \frac { p ( w ) } { q _ { p } ( { \underline { { w } } } ) } } - \sum _ { { \underline { { w } } } } \sum _ { w \in g ( { \underline { { w } } } ) } p ( w ) \log { \frac { p ( w ) } { p _ { p } ( { \underline { { w } } } ) } }\tag{76}
$$

$$
= \sum _ { \underline { { w } } } \sum _ { w \in g ( \underline { { w } } ) } p ( w ) \log \frac { p _ { p } ( \underline { { w } } ) } { q _ { p } ( \underline { { w } } ) }\tag{77}
$$

$$
\begin{array} { r } { = D _ { p } ( p _ { p } \| q _ { p } ) \ge 0 . } \end{array}\tag{78}
$$

Thus $D _ { p } ( p \| q _ { p } ) \ge D _ { p } ( p \| p _ { p } )$ . This completes the proof.

Corollary 3 (Ordering of Pragmatic Relative Entropies). The three forms of pragmatic relative entropy satisfy the following chain of inequalities:

$$
D _ { p } ( p \| q _ { p } ) \leq D _ { p } ( p _ { p } \| q _ { p } ) \leq D _ { p } ( p _ { p } \| q ) .\tag{79}
$$

Proof: Detailed derivations are analogous to the semantic case shown in [21, 22]. □

Corollary 4 (Relation to Classical and Semantic Relative Entropies). Let $D ( p \| q )$ denote the classical relative entropy, and let $D _ { s } ( p _ { s } \| q _ { s } ) , \ D _ { s } ( p _ { s } \| q ) , \ D _ { s } ( p \| q _ { s } )$ denote the full and partial

semantic relative entropies as defined in [21, 22]. Then the following relationships hold:

$$
\begin{array} { r } { D _ { p } ( p \| q _ { p } ) \leq D _ { s } ( p \| q _ { s } ) \leq D ( p \| q ) \leq D _ { s } ( p _ { s } \| q ) \leq D _ { p } ( p _ { p } \| q ) , } \end{array}\tag{80}
$$

where $p _ { s }$ and $q _ { s }$ denote the semantic aggregations, and $p _ { p } , q _ { p }$ denote the pragmatic aggregations. More precisely, we have the two separate chains:

$$
D _ { p } ( p \| q _ { p } ) \leq D _ { s } ( p \| q _ { s } ) \leq D ( p \| q ) ,\tag{81}
$$

and

$$
\begin{array} { r } { D ( p \| q ) \leq D _ { s } ( p _ { s } \| q ) \leq D _ { p } ( p _ { p } \| q ) . } \end{array}\tag{82}
$$

Proof: The semantic relative entropies are defined by aggregating over the semantic partition, which is finer than the pragmatic partition (since multiple semantic classes may merge into one pragmatic class). By the data processing inequality for relative entropy [45] (or by applying the chain of inequalities analogous to the semantic case), the coarser aggregation (pragmatic) yields a smaller or equal divergence when the first argument is aggregated and the second is fine-grained, and a larger or equal divergence when the first is fine-grained and the second is aggregated. The classical relative entropy lies between the semantic partial entropies as established in [21, 22]. Combining these facts yields the desired hierarchy. □

Theorem 6 (Convexity). The pragmatic relative entropies $D _ { p } ( p _ { p } \| q _ { p } ) , \ D _ { p } ( p _ { p } \| q )$ , and $D _ { p } ( p \vert \vert q _ { p } )$ are convex in the pair of distributions $( p , q )$ . That is, for any two pairs $( p _ { 1 } , q _ { 1 } )$ and $( p _ { 2 } , q _ { 2 } )$ , and for $0 \leq \theta \leq 1$ , we have

$$
\left\{ \begin{array} { r } { D _ { p } ( \theta p _ { p , 1 } + ( 1 - \theta ) p _ { p , 2 } \| \theta q _ { p , 1 } + ( 1 - \theta ) q _ { p , 2 } ) \leq \theta D _ { p } ( p _ { p , 1 } \| q _ { p , 1 } ) + ( 1 - \theta ) D _ { p } ( p _ { p , 2 } \| q _ { p , 2 } ) , } \\ { D _ { p } ( \theta p _ { p , 1 } + ( 1 - \theta ) p _ { p , 2 } \| \theta q _ { 1 } + ( 1 - \theta ) q _ { 2 } ) \leq \theta D _ { p } ( p _ { p , 1 } \| q _ { 1 } ) + ( 1 - \theta ) D _ { p } ( p _ { p , 2 } \| q _ { 2 } ) , } \\ { D _ { p } ( \theta p _ { 1 } + ( 1 - \theta ) p _ { 2 } \| \theta q _ { p , 1 } + ( 1 - \theta ) q _ { p , 2 } ) \leq \theta D _ { p } ( p _ { 1 } \| q _ { p , 1 } ) + ( 1 - \theta ) D _ { p } ( p _ { 2 } \| q _ { p , 2 } ) . } \end{array} \right.\tag{83}
$$

Proof: The proof follows from the joint convexity of the classical KL divergence and the fact that the pragmatic aggregation is a linear operation on probabilities. Each of the three pragmatic relative entropies can be expressed as a linear combination of classical KL divergences between appropriate fine-grained or aggregated distributions, and the convexity is inherited. Detailed derivations are analogous to the semantic case presented in [21, 22]. □

Remark 3. The pragmatic relative entropy measures the difference between two probability distributions at the level of terminal actions. The full pragmatic relative entropy $D _ { p } ( p _ { p } \| q _ { p } )$ quantifies how much the decision-oriented distributions differ. The partial measures allow comparison between a decision-level distribution and a syntactic-level distribution, which is useful in scenarios where one belief is expressed in actions (e.g., a control policy) and the other in raw observations. The hierarchy in Eq. (80) shows that pragmatic abstraction—by discarding distinctions that do not affect the optimal action—reduces the divergence when comparing fine-grained against coarse-grained distributions, and increases the divergence when comparing coarse-grained against fine-grained, perfectly mirroring the information-theoretic processing inequality for relative entropy under coarse-graining.

Subsequentlly, we will extend these concepts to mutual information, where we define up and down pragmatic mutual information and establish their relationships with syntactic and semantic mutual information.

## B. Pragmatic Mutual Information

In semantic information theory, two distinct forms of mutual information—the up and down semantic mutual information—were introduced to account for the non-additive nature of semantic entropy [21, 22]. Analogously, we define up and down pragmatic mutual information, which capture the reduction in decision uncertainty from different perspectives. Furthermore, we extend the framework to four single-sided pragmatic mutual information measures that characterize the information flow when only one side of the communication link is coarsened to the pragmatic level. These measures are essential for analyzing perception, expression, and reconstruction in task-oriented systems.

1) Definitions of Up and Down Pragmatic Mutual Information: Let W and V be two syntactic alphabets with a joint probability mass function $P ( W , V )$ , and let the associated pragmatic alphabets be W and V induced by the joint reification mapping

$$
g _ { W V } : { \underline { { { \mathcal { W } } } } } \times \underline { { { \mathcal { V } } } } \longrightarrow 2 ^ { 2 w \times \mathcal { V } } ,\tag{84}
$$

which partitions the joint syntactic space $\mathcal { W } \times \mathcal { V }$ into pragmatic equivalence classes. The pragmatic variables are then defined as ${ \underline { { W } } } = g _ { W V } ^ { - 1 } ( W , V )$ and $\underline { { { V } } } = g _ { W V } ^ { - 1 } ( W , V )$ . More precisely, the pragmatic pair (W, V ) is the unique pragmatic class to which the syntactic pair (W, V )

belongs. Equivalently, the joint reification mapping directly induces the joint pragmatic distribution by aggregating probabilities over the joint syntactic cells:

$$
P ( \underline { { w } } , \underline { { v } } ) = \sum _ { ( w , v ) \in g _ { W V } ( \underline { { w } } , \underline { { v } } ) } P ( w , v ) .\tag{85}
$$

The marginal pragmatic distributions are obtained by further summing over the other variable:

$$
P ( \underline { { w } } ) = \sum _ { \underline { { v } } ^ { \prime } } P ( \underline { { w } } , \underline { { v } } ^ { \prime } ) , \quad P ( \underline { { v } } ) = \sum _ { \underline { { w } } ^ { \prime } } P ( \underline { { w } } ^ { \prime } , \underline { { v } } ) .\tag{86}
$$

Definition 12 (Pragmatic Mutual Information). The up pragmatic mutual information is defined as

$$
\begin{array} { l } { I ^ { p } ( \underline { { W } } ; \underline { { V } } ) \triangleq \displaystyle \sum _ { w \in \mathcal { W } } \sum _ { v \in \mathcal { V } } P ( w , v ) \log \frac { P ( \underline { { w } } , \underline { { v } } ) } { P ( w ) P ( v ) } } \\ { \quad = H ( W ) + H ( V ) - H _ { p } ( \underline { { W } } , \underline { { V } } ) , } \end{array}\tag{87}
$$

where $( { \underline { { w } } } , { \underline { { v } } } ) = g _ { W V } ( w , v )$ is the pragmatic class assigned to the syntactic pair $( w , v )$

The down pragmatic mutual information is defined as

$$
\begin{array} { l } { I _ { p } ( \underline { { W } } ; \underline { { V } } ) \triangleq \displaystyle \sum _ { w \in \mathcal { W } } \sum _ { v \in \mathcal { V } } P ( w , v ) \log \frac { P ( w ) P ( v ) } { P ( \underline { { w } } , \underline { { v } } ) } } \\ { \quad = H _ { p } ( \underline { { W } } ) + H _ { p } ( \underline { { V } } ) - H ( W , V ) , } \end{array}\tag{88}
$$

We adopt the canonical definition for all subsequent rate-distortion and Lagrangian formulations: $I _ { p } ( { \underline { { W } } } ; { \underline { { V } } } ) = \operatorname* { m a x } { \{ 0 , ~ H _ { p } ( { \underline { { W } } } ) + H _ { p } ( { \underline { { V } } } ) - H ( W , V ) \} } = ( I _ { p } ) ^ { + }$ . The positive-part operation ensures that the rate-distortion function $R _ { p } ( D )$ is strictly convex and bounded below by 0, resolving the ill-posed minimization that would otherwise arise.

The up pragmatic mutual information measures the reduction in the syntactic uncertainty of W and V when their joint pragmatic structure is taken into account. It can be interpreted as the maximum amount of information about the terminal actions that can be extracted from the syntactic variables. The down pragmatic mutual information measures the reduction in pragmatic uncertainty when the joint syntactic structure is known; it can be seen as the minimum amount of pragmatic dependence that survives after coarse-graining.

We now establish the fundamental relationships among the pragmatic, semantic, and syntactic mutual information measures. For reference, the semantic mutual informations are defined

as [21, 22]:

$$
I ^ { s } ( \tilde { W } ; \tilde { V } ) = H ( W ) + H ( V ) - H _ { s } ( \tilde { W } , \tilde { V } ) ,\tag{89}
$$

$$
I _ { s } ( \tilde { W } ; \tilde { V } ) = H _ { s } ( \tilde { W } ) + H _ { s } ( \tilde { V } ) - H ( W , V ) .\tag{90}
$$

Theorem 7 (Hierarchy of Mutual Information). For any two syntactic variables W, V with associated semantic variables W , <sup>˜</sup> V<sup>˜</sup> and pragmatic variables W, V , the following chain of inequalities holds:

$$
I _ { p } ( { \underline { { W } } } ; { \underline { { V } } } ) \leq I _ { s } ( \tilde { W } ; \tilde { V } ) \leq I ( W ; V ) \leq I ^ { s } ( \tilde { W } ; \tilde { V } ) \leq I ^ { p } ( { \underline { { W } } } ; { \underline { { V } } } ) .\tag{91}
$$

Proof: The inequalities follow directly from the entropy hierarchy established in Section III. Specifically, $I _ { p } \leq I _ { s }$ because pragmatic variables are deterministic functions of semantic variables, implying $H _ { p } ( \underline { { W } } ) \le H _ { s } ( \tilde { W } )$ and $H _ { p } ( \underline { { V } } ) \le H _ { s } ( \tilde { V } ) ; I _ { s } \le I$ because $H _ { s } ( \tilde { W } ) \leq H ( W )$ and $H _ { s } ( \tilde { V } ) \leq H ( V ) ; I \leq I ^ { s }$ because $H _ { s } ( \tilde { W } , \tilde { V } ) \le H ( W , V )$ ; and $I ^ { s } \leq I ^ { p }$ because $H _ { p } ( \underline { { W } } , \underline { { V } } ) \leq$ $H _ { s } ( \tilde { W } , \tilde { V } )$ . Combining these yields the desired chain. □

Beyond the inequality chain, we can derive explicit additive decompositions that reveal the incremental contributions of semantic and pragmatic coarse-graining.

Theorem 8 (Additive Decomposition of Up Pragmatic Mutual Information). The up pragmatic mutual information can be decomposed into the up semantic mutual information plus a pragmatic increment:

$$
I ^ { p } ( { \underline { { W } } } ; { \underline { { V } } } ) = I ^ { s } ( \tilde { W } ; \tilde { V } ) + \Delta _ { p } ^ { u p } ( { \underline { { W } } } , { \underline { { V } } } ) ,\tag{92}
$$

where the pragmatic increment is

$$
\Delta _ { p } ^ { u p } ( \underline { { W } } , \underline { { V } } ) \triangleq H _ { s } ( \tilde { W } , \tilde { V } ) - H _ { p } ( \underline { { W } } , \underline { { V } } ) \geq 0 .\tag{93}
$$

Furthermore, the up pragmatic mutual information can also be decomposed relative to the classical mutual information:

$$
I ^ { p } ( \underline { { W } } ; \underline { { V } } ) = I ( W ; V ) + \Delta _ { s } ^ { u p } ( \tilde { W } , \tilde { V } ) + \Delta _ { p } ^ { u p } ( \underline { { W } } , \underline { { V } } ) ,\tag{94}
$$

where

$$
\Delta _ { s } ^ { u p } ( \tilde { W } , \tilde { V } ) \triangleq H ( W , V ) - H _ { s } ( \tilde { W } , \tilde { V } ) \geq 0 ,\tag{95}
$$

is the semantic increment (the reduction in joint entropy due to semantic coarse-graining).

Proof: From the definitions, $I ^ { p } = H ( W ) + H ( V ) - H _ { p } ( \underline { { W } } , \underline { { V } } )$ . Adding and subtracting $H _ { s } ( \tilde { W } , \tilde { V } )$ :

$$
{ \cal I } ^ { p } = [ H ( W ) + H ( V ) - H _ { s } ( \tilde { W } , \tilde { V } ) ] + [ H _ { s } ( \tilde { W } , \tilde { V } ) - H _ { p } ( \underline { { W } } , \underline { { V } } ) ] = { \cal I } ^ { s } + \Delta _ { p } ^ { u p } .\tag{96}
$$

For the second decomposition, add and subtract $H ( W , V )$ :

$$
\begin{array} { r l } & { I ^ { p } = [ H ( W ) + H ( V ) - H ( W , V ) ] } \\ & { \qquad + \left[ H ( W , V ) - H _ { s } ( \tilde { W } , \tilde { V } ) \right] } \\ & { \qquad + \left[ H _ { s } ( \tilde { W } , \tilde { V } ) - H _ { p } ( \underline { { W } } , \underline { { V } } ) \right] } \\ & { \qquad = I + \Delta _ { s } ^ { u p } + \Delta _ { p } ^ { u p } . } \end{array}\tag{97}
$$

The non-negativity follows from the entropy hierarchy.

Theorem 9 (Additive Decomposition of Down Pragmatic Mutual Information). The down pragmatic mutual information can be decomposed into the down semantic mutual information minus a pragmatic loss:

$$
I _ { p } ( \underline { { W } } ; \underline { { V } } ) = I _ { s } ( \tilde { W } ; \tilde { V } ) - \Delta _ { p } ^ { d o w n } ( \underline { { W } } , \underline { { V } } ) ,\tag{98}
$$

where the pragmatic loss is

$$
\Delta _ { p } ^ { d o w n } ( { \underline { { W } } } , { \underline { { V } } } ) \triangleq [ H _ { s } ( \tilde { W } ) - H _ { p } ( { \underline { { W } } } ) ] + [ H _ { s } ( \tilde { V } ) - H _ { p } ( { \underline { { V } } } ) ] \geq 0 .\tag{99}
$$

Furthermore, the down pragmatic mutual information can be decomposed relative to the classical mutual information:

$$
I _ { p } ( \underline { { { W } } } ; \underline { { { V } } } ) = I ( W ; V ) - \Delta _ { s } ^ { d o w n } ( \tilde { W } , \tilde { V } ) - \Delta _ { p } ^ { d o w n } ( \underline { { { W } } } , \underline { { { V } } } ) ,\tag{100}
$$

where

$$
\Delta _ { s } ^ { d o w n } ( \tilde { W } , \tilde { V } ) \triangleq [ H ( W ) - H _ { s } ( \tilde { W } ) ] + [ H ( V ) - H _ { s } ( \tilde { V } ) ]  \geq 0 ,\tag{101}
$$

is the semantic loss.

Proof: From $I _ { p } = H _ { p } ( { \underline { { W } } } ) + H _ { p } ( { \underline { { V } } } ) - H ( W , V )$ . Adding and subtracting $H _ { s } ( \tilde { W } )$ and $H _ { s } ( \tilde { V } )$

$$
\begin{array} { r l } & { I _ { p } = [ H _ { s } ( \tilde { W } ) + H _ { s } ( \tilde { V } ) - H ( W , V ) ] } \\ & { \qquad - \left[ H _ { s } ( \tilde { W } ) - H _ { p } ( \underline { { W } } ) \right] } \\ & { \qquad - \left[ H _ { s } ( \tilde { V } ) - H _ { p } ( \underline { { V } } ) \right] } \\ & { \qquad - I _ { s } - \Delta _ { p } ^ { d o w n } . } \end{array}\tag{102}
$$

For the classical decomposition, add and subtract $H ( W )$ and $H ( V )$

$$
\begin{array} { l } { { I _ { p } = [ H ( W ) + H ( V ) - H ( W , V ) ] } } \\ { { \ } } \\ { { \ } } \\ { { \displaystyle ~ - \left[ H ( W ) - H _ { s } ( \tilde { W } ) \right] } } \\ { { \ } } \\ { { \displaystyle ~ - \left[ H ( V ) - H _ { s } ( \tilde { V } ) \right] } } \\ { { \ } } \\ { { \displaystyle ~ - \left[ H _ { s } ( \tilde { W } ) - H _ { p } ( \underline { { { W } } } ) \right] } } \\ { { \ } } \\ { { \displaystyle ~ - \left[ H _ { s } ( \tilde { V } ) - H _ { p } ( \underline { { { V } } } ) \right] } } \\ { { \ } } \\ { { \displaystyle = I - \Delta _ { s } ^ { d o w n } - \Delta _ { p } ^ { d o w n } . } } \end{array}\tag{103}
$$

The non-negativity follows from the entropy hierarchy.

Remark 4. The additive decompositions reveal a clear structural interpretation: the up pragmatic mutual information adds the information gains from each layer of abstraction, while the down pragmatic mutual information subtracts the information losses. The classical mutual information serves as the central reference point. The inequalities $I _ { p } \leq I _ { s } \leq I \leq I ^ { s } \leq I ^ { p }$ follow immediately from the non-negativity of the increments/losses.

2) Single-Sided Pragmatic Mutual Information: In many engineering scenarios, the coarsegraining to the pragmatic level occurs on only one side of the communication link. For instance, in perception tasks, the source $W$ is coarsened to $W$ (the action), while the observation $V$ remains syntactic. Conversely, in expressive or reconstructive tasks, the source remains syntactic while the reconstruction or observation is coarsened to $\underline V$ (the action label). To capture these asymmetric situations, we define four single-sided pragmatic mutual information measures.

Let $W$ and $V$ be syntactic variables with joint distribution $P ( W , V )$ , and let W and $\underline V$ be the corresponding pragmatic variables induced by the respective reification mappings. The following four measures are defined.

Definition 13 (Single-Sided Pragmatic Mutual Information).

1) Perceptual Pragmatic Mutual Information (PPMI):

$$
I _ { p } ( \underline { { W } } ; V ) \triangleq H _ { p } ( \underline { { W } } ) + H ( V ) - H ( \underline { { W } } , V ) = H _ { p } ( \underline { { W } } ) - H _ { p } ( \underline { { W } } | V ) = I ( \underline { { W } } ; V ) ,\tag{104}
$$

This measure quantifies how much information the syntactic observation V carries about the pragmatic action W.

2) Prospective Perceptual Pragmatic Mutual Information (PPPMI):

$$
I ^ { p } ( \underline { { W } } ; V ) \triangleq H ( W ) + H ( V ) - H ( \underline { { W } } , V ) ,\tag{105}
$$

This measure provides an upper bound on the perceptual information when the syntactic entropy of the source is retained in the marginal term.

3) Expressive Pragmatic Mutual Information (EPMI):

$$
I _ { p } ( W ; \underline { { V } } ) \triangleq H ( W ) + H _ { p } ( \underline { { V } } ) - H ( W , \underline { { V } } ) = H ( W ) - H ( W | \underline { { V } } ) = I ( W ; \underline { { V } } ) ,\tag{106}
$$

This measure quantifies how much information about the syntactic source W is contained in the pragmatic label V .

4) Prospective Expressive Pragmatic Mutual Information (PEPMI):

$$
I ^ { p } ( W ; \underline { { { V } } } ) \triangleq H ( W ) + H ( V ) - H ( W , \underline { { { V } } } ) ,\tag{107}
$$

This measure provides an upper bound on the expressive information when the syntactic entropy of the observation is retained.

The naming reflects their roles: “Perceptual” refers to the direction from observations to actions; “Expressive” refers to the direction from actions to source reconstruction. The prefix “Prospective” indicates that the measure uses the syntactic (larger) marginal entropy, thus providing an upper bound.

These four measures, together with the classical mutual information $I ( W ; V )$ and the two bilateral pragmatic mutual informations $I _ { p } ( \underline { { W } } ; \underline { { V } } )$ and $I ^ { p } ( \underline { { W } } ; \underline { { V } } )$ , form a complete lattice of information quantities that cover all combinations of coarse-graining (none, one side, both sides) and marginal entropy type (syntactic or pragmatic).

Theorem 10 (Hierarchy of Single-Sided and Bilateral Measures). For any syntactic variables W, V with associated pragmatic variables W, V , the following inequalities hold:

$$
I _ { p } ( \underline { { W } } ; \underline { { V } } ) \le I _ { p } ( \underline { { W } } ; V ) \le I ( W ; V ) \le I ^ { p } ( \underline { { W } } ; V ) \le I ^ { p } ( \underline { { W } } ; \underline { { V } } ) ,\tag{108}
$$

$$
I _ { p } ( \underline { { W } } ; \underline { { V } } ) \le I _ { p } ( W ; \underline { { V } } ) \le I ( W ; V ) \le I ^ { p } ( W ; \underline { { V } } ) \le I ^ { p } ( \underline { { W } } ; \underline { { V } } ) .\tag{109}
$$

Moreover, the cross comparisons between the left chain (perceptual) and the right chain (expressive) are generally incomparable, except that any measure from the lower (pragmatic marginal) side is bounded above by any measure from the upper (syntactic marginal) side:

$$
I _ { p } ( \underline { { W } } ; V ) \le I ( W ; V ) \le I ^ { p } ( W ; \underline { { V } } ) , \quad a n d \quad I _ { p } ( W ; \underline { { V } } ) \le I ( W ; V ) \le I ^ { p } ( \underline { { W } } ; V ) .\tag{110}
$$

Proof: The left chain in Eq. (108) follows from four successive inequalities. First, $I _ { p } ( \underline { { W } } ; \underline { { V } } ) \leq$ $I _ { p } ( \underline { { W } } ; V )$ because V is a function of V (data processing inequality). Second, $I _ { p } ( \underline { { W } } ; V ) \ \leq$ $I ( W ; V )$ because W is a function of W. Third, $I ( W ; V ) \le I ^ { p } ( \underline { { W } } ; V )$ since $I ^ { p } ( \underline { { W } } ; V ) \ =$ $I ( W ; V ) + [ H ( W ) - H ( \underline { { W } } ) ] - H ( W | \underline { { W } } , V ) \geq I ( W ; V )$ . Fourth, $I ^ { p } ( \underline { { W } } ; V ) \le I ^ { p } ( \underline { { W } } ; \underline { { V } } )$ because $T ^ { p } ( \underline { { W } } ; \underline { { V } } ) = I ^ { p } ( \underline { { W } } ; V ) + [ H ( V ) - H ( \underline { { V } } ) ] - H ( V | \underline { { W } } , \underline { { V } } ) \geq I ^ { p } ( \underline { { W } } ; V )$ . The right chain in Eq. (109) follows by symmetry.

The cross bounds between perceptual and expressive measures are direct consequences of the fact that any single-sided measure with a pragmatic marginal entropy (PPMI or EPMI) is bounded above by the classical mutual information, while any single-sided measure with a syntactic marginal entropy (PPPMI or PEPMI) is bounded below by the classical mutual information. The incomparability between PPMI and EPMI (and similarly between PPPMI and PEPMI) can be demonstrated by constructing examples where one exceeds the other, depending on the relative losses of coarse-graining on each side. □

3) Sequential Chain Rules for Pragmatic Mutual Information: We now extend the pragmatic mutual information to sequences, including the single-sided measures. Let $( W ^ { n } , V ^ { n } )$ be a pair of syntactic sequences, and let $( \underline { { W } } ^ { n } , \underline { { V } } ^ { n } )$ be the corresponding pragmatic sequences obtained by applying the joint Reification Mapping element-wise. Define the sequential partial entropy as in Section III.B:

$$
\tilde { H } _ { p } ( \underline { { W } } _ { 1 } ^ { m } , W _ { m + 1 } ^ { n } ) \triangleq \sum _ { k = 1 } ^ { m } H _ { p } ( \underline { { W } } _ { k } | \underline { { W } } _ { 1 } ^ { k - 1 } ) + \sum _ { k = m + 1 } ^ { n } H _ { p } ( W _ { k } | \underline { { W } } _ { 1 } ^ { m } , W _ { m + 1 } ^ { k - 1 } ) ,\tag{111}
$$

with analogous notation for mixed partial entropies involving V and $V .$

Theorem 11 (Sequential Chain Rule for Down Pragmatic Mutual Information). For a syntactic sequence pair $( W ^ { n } , V ^ { n } )$ and its pragmatic counterpart, the down pragmatic mutual information satisfies the following chain of inequalities:

$$
I _ { p } ( { \underline { { W } } } ^ { n } ; { \underline { { V } } } ^ { n } ) \ \leq \ I _ { p } ( { \underline { { W } } } _ { 1 } ^ { n - 1 } , W _ { n } ; { \underline { { V } } } ^ { n } ) \ \leq \ \cdots \ \leq \ I _ { p } ( { \underline { { W } } } _ { 1 } , W _ { 2 } ^ { n } ; { \underline { { V } } } ^ { n } ) \ \leq \ I ( W ^ { n } ; V ^ { n } ) ,\tag{112}
$$

where

$$
I _ { p } ( \underline { { W } } _ { 1 } ^ { m } , W _ { m + 1 } ^ { n } ; \underline { { V } } ^ { n } ) \triangleq \tilde { H } _ { p } ( \underline { { W } } _ { 1 } ^ { m } , W _ { m + 1 } ^ { n } ) + H _ { p } ( \underline { { V } } ^ { n } ) - H ( W ^ { n } , V ^ { n } ) .\tag{113}
$$

Proof: The proof follows the same steps as the semantic sequential chain rule (Theorem 8 in [21, 22]), with the semantic entropies replaced by pragmatic entropies. By the hierarchy of sequential entropies (Theorem 3 in Section III-B), we have

$$
\tilde { H } _ { p } ( \underline { { W } } _ { 1 } ^ { m } , W _ { m + 1 } ^ { n } ) \leq \tilde { H } _ { p } ( \underline { { W } } _ { 1 } ^ { m - 1 } , W _ { m } ^ { n } ) \leq \cdots \leq H ( W ^ { n } ) .\tag{114}
$$

Substituting these into the definition of $I _ { p }$ for the mixed blocks yields the desired chain. □

Similarly, for the up pragmatic mutual information we have:

Theorem 12 (Sequential Chain Rule for Up Pragmatic Mutual Information).

$$
I ( W ^ { n } ; V ^ { n } ) \ \le \ I ^ { p } ( { \underline { { W } } } _ { 1 } , W _ { 2 } ^ { n } ; { \underline { { V } } } ^ { n } ) \ \le \ \cdots \ \le \ I ^ { p } ( { \underline { { W } } } _ { 1 } ^ { n - 1 } , W _ { n } ; { \underline { { V } } } ^ { n } ) \ \le \ I ^ { p } ( { \underline { { W } } } ^ { n } ; \underline { { V } } ^ { n } ) ,\tag{115}
$$

where

$$
I ^ { p } ( \underline { { W } } _ { 1 } ^ { m } , W _ { m + 1 } ^ { n } ; \underline { { V } } ^ { n } ) \triangleq H ( W ^ { n } ) + H ( V ^ { n } ) - \tilde { H } _ { p } ( \underline { { W } } _ { 1 } ^ { m } , W _ { m + 1 } ^ { n } , \underline { { V } } ^ { n } ) ,\tag{116}
$$

with $\tilde { H } _ { p }$ denoting the corresponding mixed sequential entropy.

Proof: Analogous to the semantic case, using the fact that replacing a syntactic variable by its pragmatic counterpart increases the up mutual information due to the decrease in the joint entropy term. □

For the single-sided measures, we can also derive sequential chain rules. For example, the perceptual pragmatic mutual information $I _ { p } ( \underline { { W } } ^ { n } ; V ^ { n } )$ can be bounded by sequentially replacing $W ^ { \bullet } \mathbf { s }$ with W’s:

Theorem 13 (Sequential Chain Rule for Perceptual Pragmatic Mutual Information).

$$
I _ { p } ( { \underline { { W } } } ^ { n } ; V ^ { n } ) \le I _ { p } ( { \underline { { W } } } _ { 1 } ^ { n - 1 } , W _ { n } ; V ^ { n } ) \le \cdots \le I _ { p } ( { \underline { { W } } } _ { 1 } , W _ { 2 } ^ { n } ; V ^ { n } ) \le I ( W ^ { n } ; V ^ { n } ) ,\tag{117}
$$

where the mixed quantities are defined analogously using the corresponding conditional entropies.

Similarly, for the expressive pragmatic mutual information $I _ { p } ( W ^ { n } ; \underline { { V } } ^ { n } )$ , we have

$$
I _ { p } ( W ^ { n } ; \underline { { { V } } } ^ { n } ) \leq I _ { p } ( W ^ { n } ; \underline { { { V } } } _ { 1 } ^ { n - 1 } , V _ { n } ) \leq \cdots \leq I _ { p } ( W ^ { n } ; \underline { { { V } } } _ { 1 } , V _ { 2 } ^ { n } ) \leq I ( W ^ { n } ; V ^ { n } ) .\tag{118}
$$

The up versions (PPPMI and PEPMI) satisfy the reverse chain inequalities, starting from the classical mutual information and increasing towards the upper bounds.

Theorem 14 (Sequential Chain Rule for Prospective Perceptual Pragmatic Mutual Information).

$$
I ( W ^ { n } ; V ^ { n } ) \ \le \ I ^ { p } ( \underline { { W } } _ { 1 } , W _ { 2 } ^ { n } ; V ^ { n } ) \ \le \ \cdots \ \le \ I ^ { p } ( \underline { { W } } _ { 1 } ^ { n - 1 } , W _ { n } ; V ^ { n } ) \ \le \ I ^ { p } ( \underline { { W } } ^ { n } ; V ^ { n } ) .\tag{119}
$$

Analogous chains hold for PEPMI by replacing $V ' \mathbf { s }$ with $\underline { { V } } ^ { \ } \mathbf { s }$ in the mixed blocks.

Finally, the hierarchy of sequential mutual informations extends to all seven measures, with the same partial order as in the single-letter case, for any fixed n.

## Example 3 (Multi-Vehicle Scenario).

We now revisit the multi-vehicle autonomous driving scenario from Section III-B to compute the classical, semantic, and pragmatic mutual informations, and also the four single-sided measures. Recall the joint syntactic distribution $P ( W , V )$ given in Table III. The syntactic states are traffic light colors, with four possible values: green, yellow, red-short, red-long. The semantic mapping groups green and yellow into “Proceed”, red-short into “Stop-short”, and red-long into “Stop-long”. The pragmatic mapping further merges the two stop classes into a single “Stop” action, as the utility function treats them identically. The joint Reification Mapping g<sub>WV</sub> thus maps the $4 \times 4$ syntactic pairs to a $2 \times 2$ pragmatic partition: “Go” (green/yellow combined) and “Stop” (both red types combined).

The syntactic joint distribution is given in Table III and the corresponding semantic and pragmatic joint distributions are given in Tables IV and V, respectively.

From earlier computations in Example 2, we can calculate the mixed entropies $H ( W , \underline { { V } } ) =$ $H ( \underline { { W } } , V ) \approx 2 . 7 3 2 6$ bits. Further, we need $H ( \underline { W } )$ and $H ( \underline { { V } } )$ which are both 0.9928 (since marginal pragmatic entropies equal). Now compute the single-sided measures:

1. PPMI: $I _ { p } ( \underline { { W } } ; V ) = H _ { p } ( \underline { { W } } ) + H ( V ) - H ( \underline { { W } } , V ) = 0 . 9 9 2 8 + 1 . 9 8 5 6 - 2 . 7 3 2 6 = 0 . 2 4 5 8$ bits. 2. PPPMI: $I ^ { p } ( \underline { { { W } } } ; V ) = H ( W ) + H ( V ) - H ( \underline { { { W } } } , V ) = 1 . 9 8 5 6 + 1 . 9 8 5 6 - 2 . 7 3 2 6 = 1 . 2 3 8 6$ bits. 3. EPMI: $I _ { p } ( W ; \underline { { { V } } } ) = H ( W ) + H _ { p } ( \underline { { { V } } } ) - H ( W , \underline { { { V } } } ) = 1 . 9 8 5 6 + 0 . 9 9 2 8 - 2 . 7 3 2 6 = 0 . 2 4 5 8$ bits (same as PPMI due to symmetry in this example, but in general they differ). 4. PEPMI: $I ^ { p } ( W ; \underline { { { V } } } ) = H ( W ) + H ( V ) - H ( W , \underline { { { V } } } ) = 1 . 9 8 5 6 + 1 . 9 8 5 6 - 2 . 7 3 2 6 = 1 . 2 3 8 6$ bits (same as PPPMI).

The classical mutual information $I ( W ; V ) = 0 . 3 3 9 4$ bits. The down bilateral $I _ { p } ( \underline { { W } } ; \underline { { V } } ) = 0$ (positive part). The up bilateral $I ^ { p } ( \underline { { W } } ; \underline { { V } } ) = 2 . 2 2 0 3$ prabits. The results are summarized in Table VII.

TABLE VII: Comparison of all mutual information measures for the two-vehicle scenario.
<table><tr><td>Mutual Information</td><td>Value (bits/sebits/prabits)</td></tr><tr><td>Down Pragmatic  $I _ { p } ( \underline { { W } } ; \underline { { V } } )$ </td><td>0 (positive part)</td></tr><tr><td>Down Semantic  $I _ { s } ( \tilde { W } ; \tilde { V } )$ </td><td>0 (positive part)</td></tr><tr><td>Perceptual Pragmatic  $I _ { p } ( \underline { { W } } ; V )$ </td><td>0.2458 prabits</td></tr><tr><td>Expressive Pragmatic  $I _ { p } ( W ; \underline { { V } } )$ </td><td>0.2458 prabits</td></tr><tr><td>Classical  $I ( W ; V )$ </td><td>0.3394 bits</td></tr><tr><td>Prospective Perceptual  $I ^ { p } ( \underline { { W } } ; V )$ </td><td>1.2386 bits</td></tr><tr><td>Prospective Expressive  $I ^ { p } ( W ; \underline { { V } } )$ </td><td>1.2386 bits</td></tr><tr><td>Up Semantic  $I ^ { s } ( \tilde { W } ; \tilde { V } )$ </td><td>1.2147 sebits</td></tr><tr><td>Up Pragmatic  $I ^ { p } ( \underline { { W } } ; \underline { { V } } )$ </td><td>2.2203 prabits</td></tr></table>

We observe that the single-sided measures lie between the down bilateral and the classical mutual information (for the lower ones) or between the classical and the up bilateral (for the upper ones), confirming the lattice structure. The equality of PPMI and EPMI in this symmetric example is coincidental; in general they differ.

This example demonstrates the utility of the single-sided measures in quantifying the information loss or gain when only one side of the communication is coarsened to the pragmatic level. They provide a finer-grained analysis of task-oriented information flow compared to the bilateral measures alone.

## V. PRAGMATIC CHANNEL CAPACITY AND PRAGMATIC RATE DISTORTION

We introduce the pragmatic channel capacity and rate-distortion function as the fundamental limits of pragmatic communication, paralleling their classical and semantic counterparts. These quantities characterize the maximum reliable transmission rate and the minimum rate for a given task distortion, respectively, and reveal performance gains from pragmatic abstraction. We further extend to single-sided perceptual and expressive measures for asymmetric sensing-actuation systems.

## A. Pragmatic Channel Capacity

The pragmatic channel capacity quantifies the maximum rate at which pragmatic information—the optimal terminal actions—can be reliably transmitted over a given communication channel. It is defined as the supremum of the up pragmatic mutual information between the input pragmatic variable and the output pragmatic variable, maximized over the input distribution and the joint reification mapping.

Consider a discrete memoryless channel with input alphabet $x ,$ , output alphabet ${ \mathcal { V } } ,$ and transition probability $P ( \boldsymbol { Y } | \boldsymbol { X } )$ . Let the associated semantic alphabets be $\tilde { \mathcal X }$ and $\tilde { \mathcal { V } } .$ , and the pragmatic alphabets be $\mathcal { X }$ and $\underline { { \boldsymbol { \psi } } } ,$ , induced by the joint reification mapping $g _ { X Y } : \underline { { \mathcal { X } } } \times \underline { { \mathcal { Y } } } \longrightarrow 2 ^ { \mathcal { X } \times \mathcal { Y } }$ , which partitions the joint syntactic space into pragmatic equivalence classes.

Definition 14 (Pragmatic Channel Capacity). The pragmatic channel capacity is defined as

$$
C _ { p } \triangleq \operatorname* { m a x } _ { p ( x ) } \operatorname* { m a x } _ { g _ { X Y } } I ^ { p } ( \underline { { X } } ; \underline { { Y } } ) ,\tag{120}
$$

where $I ^ { p } ( \underline { { { X } } } ; \underline { { { Y } } } ) = H ( X ) + H ( Y ) - H _ { p } ( \underline { { { X } } } , \underline { { { Y } } } )$ is the up pragmatic mutual information, and the maximization is over all input distributions $p ( x )$ and all joint reification mappings $g _ { X Y }$

For reference, the semantic channel capacity [21, 22] and the classical (syntactic) channel capacity [1, 45] are defined as:

$$
\left\{ \begin{array} { l l } { C _ { s } \triangleq \underset { p ( x ) } { \mathrm { m a x } } \underset { f _ { X Y } } { \mathrm { m a x } } I ^ { s } ( \tilde { X } ; \tilde { Y } ) , } \\ { C \triangleq \underset { p ( x ) } { \mathrm { m a x } } I ( X ; Y ) , } \end{array} \right.\tag{121}
$$

where $f _ { X Y }$ denotes the joint synonymous mapping.

The following theorem establishes the hierarchy among the three capacities.

Theorem 15 (Hierarchy of Channel Capacities). The pragmatic, semantic, and syntactic channel capacities satisfy:

$$
C \ \leq \ C _ { s } \ \leq \ C _ { p } .\tag{122}
$$

Proof: The inequality $C \leq C _ { s }$ is a known result of semantic information theory [21, 22], following from the fact that $I ( X ; Y ) \leq I ^ { s } ( \tilde { X } ; \tilde { Y } )$ for any joint synonymous mapping.

To prove $C _ { s } \leq C _ { p } ,$ note that for any input distribution and any joint Synonymous Mapping $f _ { X Y }$ , we can define a corresponding joint Reification Mapping $g _ { X Y }$ that is coarser than $f _ { X Y }$

(i.e., it merges semantic classes that lead to the same optimal action). For such $^ { g , }$ we have $H _ { p } ( \underline { { X } } , \underline { { Y } } ) \le H _ { s } ( \tilde { X } , \tilde { Y } )$ , hence

$$
I ^ { p } ( \underline { { X } } ; \underline { { Y } } ) = H ( X ) + H ( Y ) - H _ { p } ( \underline { { X } } , \underline { { Y } } ) \geq H ( X ) + H ( Y ) - H _ { s } ( \tilde { X } , \tilde { Y } ) = I ^ { s } ( \tilde { X } ; \tilde { Y } ) .\tag{123}
$$

Therefore, the maximum over $g _ { X Y }$ of $I ^ { p }$ is at least the maximum over $f _ { X Y }$ of $I ^ { s } ,$ , yielding $C _ { p } \geq C _ { s }$ □

Beyond the inequality, we can quantify the capacity gains explicitly.

Theorem 16 (Additive Decomposition of Pragmatic Capacity). The pragmatic channel capacity can be decomposed as

$$
\begin{array} { r } { C _ { p } = C + \Delta _ { p s } ^ { C } , } \end{array}\tag{124}
$$

where

$$
\Delta _ { p s } ^ { C } \triangleq \Delta _ { s } ^ { C } + \Delta _ { p } ^ { C } \geq 0 ,\tag{125}
$$

with

$$
\left\{ \begin{array} { l l } { \Delta _ { s } ^ { C } \triangleq \underset { p ( x ) } { \mathrm { m a x } } \underset { f _ { X Y } } { \mathrm { m a x } } \left[ H ( X , Y ) - H _ { s } ( \tilde { X } , \tilde { Y } ) \right] \geq 0 , } \\ { \Delta _ { p } ^ { C } \triangleq \underset { p ( x ) } { \mathrm { m a x } } \underset { g _ { X Y } } { \mathrm { m a x } } \left[ H _ { s } ( \tilde { X } , \tilde { Y } ) - H _ { p } ( \underline { { X } } , \underline { { Y } } ) \right] \geq 0 . } \end{array} \right.\tag{126}
$$

Equivalently, the pragmatic capacity can also be expressed relative to the semantic capacity as

$$
\begin{array} { r } { C _ { p } = C _ { s } + \Delta _ { p } ^ { C } . } \end{array}\tag{127}
$$

Proof: From the additive relation of the up mutual information:

$$
I ^ { p } = I ^ { s } + \Delta _ { p } ^ { u p } ,\tag{128}
$$

with $\Delta _ { p } ^ { u p } = H _ { s } ( \tilde { X } , \tilde { Y } ) - H _ { p } ( \underline { { X } } , \underline { { Y } } )$ . Similarly, $I ^ { s } = I + \Delta _ { s } ^ { u p }$ , where $\Delta _ { s } ^ { u p } = H ( X , Y ) - H _ { s } ( \tilde { X } , \tilde { Y } )$ Combining these and taking the maximum over $p ( x ) , f _ { X Y }$ , and $g _ { X Y }$ yields the decomposition.

Remark 5. The classical capacity C is the fundamental physical limit determined by the channel transition probability $P ( \boldsymbol { Y } | \boldsymbol { X } )$ , representing the maximum rate at which syntactic symbols can be reliably transmitted. The additional semantic gain $\Delta _ { s } ^ { C }$ and pragmatic gain $\Delta _ { p } ^ { C }$ are not violations of this physical limit; rather, they reflect the fact that semantic and pragmatic communication systems relax the requirement of exact symbol reconstruction. $B y$ allowing errors that do not affect meaning (semantic) or that do not affect the optimal terminal action (pragmatic), the system can effectively transmit more “useful” information per physical bit. The total pragmatic capacity $C _ { p } = C + \Delta _ { p s } ^ { C }$ thus represents the capacity measured in pragmatic bits—the maximum rate of reliably conveying terminal actions—which can exceed the syntactic bit rate C because each pragmatic bit may correspond to multiple syntactic bits that are semantically or pragmatically equivalent.

## B. Pragmatic Rate Distortion

The pragmatic rate-distortion function characterizes the minimum rate required to describe a source such that the resulting distortion, measured in terms of task utility loss, does not exceed a given threshold. Unlike classical rate distortion, which uses a syntactic distortion measure (e.g., mean squared error), the pragmatic distortion is defined directly on the terminal actions.

Let $X \sim p ( x )$ be a syntactic source, and let $\tilde { X }$ and $\underline { { \boldsymbol X } }$ be the associated semantic and pragmatic variables induced by the Synonymous and Isoteleia mappings. The decoder produces a reconstruction $\hat { X }$ with associated semantic $\hat { \tilde { X } }$ and pragmatic $\underline { { \hat { X } } }$ , where the reconstructions are obtained via a test channel $p ( \hat { x } | x )$ . The pragmatic distortion measure $d _ { p } ( \underline { { x } } , \underline { { \hat { x } } } )$ quantifies the cost of representing the pragmatic symbol $\underline { { x } }$ by $\underline { { \hat { x } } } ;$ in practice, it can be defined as the utility loss:

$$
d _ { p } ( \underline { { x } } , \underline { { \hat { x } } } ) = \operatorname* { m a x } _ { a } \mathbb { E } [ U ( X , a ) | \underline { { x } } ] - \operatorname* { m a x } _ { a } \mathbb { E } [ U ( X , a ) | \underline { { \hat { x } } } ] ,\tag{129}
$$

or any other non-negative function that reflects the decision degradation.

Definition 15 (Pragmatic Rate Distortion). The pragmatic rate-distortion function is defined as

$$
R _ { p } ( D ) \triangleq \operatorname* { m i n } _ { g _ { X } , g _ { \hat { X } } } \operatorname* { m i n } _ { \substack { p ( \hat { x } | x ) : \mathbb { E } [ d _ { p } ( \underline { { X } } , \hat { \underline { { X } } } ) ] \leq D } } I _ { p } ( \underline { { X } } ; \underline { { \hat { X } } } ) ,\tag{130}
$$

where $I _ { p } ( \underline { { X } } ; \underline { { \hat { X } } } ) = H _ { p } ( \underline { { X } } ) + H _ { p } ( \underline { { \hat { X } } } ) - H ( X , \hat { X } )$ is the down pragmatic mutual information $( I _ { p } ( \underline { { X } } ; \underline { { \hat { X } } } ) = ( I _ { p } ) ^ { + }$ is non-negative), and the minimization is over all joint reification mappings $g _ { X }$ and $g _ { \hat { X } }$ for the source and reconstruction, and over all test channels satisfying the distortion constraint.

The semantic rate-distortion function $R _ { s } ( D )$ [21, 22] and the classical rate-distortion function

$R ( D )$ [1, 45] are defined analogously:

$$
\begin{array} { r } { \left\{ { R _ { s } } ( D ) \triangleq \underset { f _ { X } , f _ { \hat { X } } } { \mathrm { m i n } } \underset { p ( \hat { x } \mid x ) : \mathbb { E } [ d _ { s } ( \hat { X } , \hat { \hat { X } } ) ] \leq D } { \mathrm { m i n } } I _ { s } ( \hat { X } ; \hat { \hat { X } } ) , \right. } \\ { \left. R ( D ) \triangleq \underset { p ( \hat { x } \mid x ) : \mathbb { E } [ d ( X , \hat { X } ) ] \leq D } { \mathrm { m i n } } I ( X ; \hat { X } ) , \right. } \end{array}\tag{131}
$$

where $d _ { s }$ is a semantic distortion measure (e.g., based on meaning), and $d$ is a syntactic distortion measure (e.g., squared error). The following hierarchy holds.

Theorem 17 (Hierarchy of Rate-Distortion Functions). For any $D \geq 0 _ { : }$

$$
R _ { p } ( D ) \leq R _ { s } ( D ) \leq R ( D ) .\tag{132}
$$

Proof: The inequality $R _ { s } ( D ) \leq R ( D )$ is a known result of semantic information theory [21, 22], following from the fact that $I _ { s } ( \tilde { X } ; \hat { \tilde { X } } ) \leq I ( X ; \hat { X } )$

To prove $R _ { p } ( D ) \leq R _ { s } ( D )$ , note that for any Synonymous Mappings $f _ { X } , f _ { \hat { X } }$ and any test channel, we can define coarser Reification Mappings $g _ { X } , g _ { \hat { X } }$ that merge semantic classes with identical optimal actions. For such coarser mappings, we have

$$
I _ { p } ( \underline { { { X } } } ; \underline { { { \hat { X } } } } ) = I _ { s } ( \tilde { X } ; \hat { \tilde { X } } ) - \Delta _ { p } ^ { d o w n } ,\tag{133}
$$

where $\Delta _ { p } ^ { d o w n } = [ H _ { s } ( \tilde { X } ) - H _ { p } ( \underline { { X } } ) ] + [ H _ { s } ( \tilde { \underline { { X } } } ) - H _ { p } ( \underline { { \hat { X } } } ) ] \geq 0$ . Therefore, the minimum over $g _ { X } , g _ { \hat { X } }$ of $I _ { p }$ is no larger than the minimum over $f _ { X } , f _ { \hat { X } }$ of $I _ { s } ,$ , provided the distortion constraints are compatible (which holds by defining $d _ { p }$ appropriately from $d _ { s } )$ . Hence $R _ { p } ( D ) \leq R _ { s } ( D )$ □

The gains can also be expressed additively.

Theorem 18 (Additive Decomposition of Pragmatic Rate Distortion). The pragmatic rate distortion function can be written as

$$
R _ { p } ( D ) = R ( D ) - { \Delta } _ { p s } ^ { R } ( D ) ,\tag{134}
$$

where

$$
\Delta _ { p s } ^ { R } ( D ) \triangleq \Delta _ { s } ^ { R } ( D ) + \Delta _ { p } ^ { R } ( D ) \geq 0 ,\tag{135}
$$

with

$$
\left\{ \begin{array} { l l } { \Delta _ { s } ^ { R } ( D ) \triangleq \displaystyle \operatorname* { m a x } _ { f _ { X } , f _ { \hat { X } } } \Big \{ [ H ( X ) - H _ { s } ( \tilde { X } ) ] + [ H ( \hat { X } ) - H _ { s } ( \hat { \tilde { X } } ) ] \Big \} \geq 0 , } \\ { \Delta _ { p } ^ { R } ( D ) \triangleq \displaystyle \operatorname* { m a x } _ { g _ { X } , g _ { \hat { X } } } \Big \{ [ H _ { s } ( \tilde { X } ) - H _ { p } ( \underline { { X } } ) ] + [ H _ { s } ( \hat { \tilde { X } } ) - H _ { p } ( \underline { { \hat { X } } } ) ] \Big \} \geq 0 . } \end{array} \right.\tag{136}
$$

Equivalently, the pragmatic rate distortion can be expressed relative to the semantic rate distortion as

$$
R _ { p } ( D ) = R _ { s } ( D ) - \Delta _ { p } ^ { R } ( D ) .\tag{137}
$$

Proof: From the additive relation $I _ { p } = I _ { s } - \Delta _ { p } ^ { d o w n }$ , where $\Delta _ { p } ^ { d o w n }$ is the pragmatic loss defined above, and $I _ { s } = I - \Delta _ { s } ^ { d o w n }$ , combining these and taking the minimum over the test channel and the mappings yields the decomposition. □

Remark 6. The classical rate-distortion function $R ( D )$ represents the minimum syntactic bit rate required to achieve a given syntactic distortion. The semantic loss $\Delta _ { s } ^ { R } ( D )$ and pragmatic loss $\Delta _ { p } ^ { R } ( D )$ reflect the reductions in rate achievable by allowing distortions at the semantic and pragmatic levels, respectively. These reductions reflect the fact that the distortion measure itself has been redefined to capture task-relevant fidelity rather than symbol-level fidelity. By tolerating syntactic or semantic errors that do not affect the terminal action, the pragmatic rate-distortion function $R _ { p } ( D )$ can be significantly lower than $R ( D )$ , providing a principled theoretical foundation for task-oriented compression.

## C. Single-Sided Pragmatic Achievable Rates and Rate-Distortion Functions

The bilateral pragmatic capacity and rate-distortion functions defined above assume that both the source and the reconstruction (or input and output) are coarsened to the pragmatic level. However, in many practical systems—such as perception, sensing, or actuation—only one side of the communication link is subject to pragmatic abstraction. To address such asymmetric scenarios, we introduce four single-sided measures: two rate-distortion functions (perceptual and expressive) and two achievable rates (prospective perceptual and prospective expressive). These measures provide a finer-grained analysis of task-oriented information flows when either the observation or the action label is coarsened.

1) Perceptual Pragmatic Rate-Distortion Function: The perceptual pragmatic rate-distortion function captures the fundamental limit of compressing an observation V to extract information about the pragmatic action W, where the source W itself remains syntactic. It is defined using the down perceptual pragmatic mutual information $I _ { p } ( \underline { { W } } ; V )$

Definition 16 (Perceptual Pragmatic Rate-Distortion Function). Let W be the pragmatic variable associated with the source W, and let V be an observation. The perceptual pragmatic ratedistortion function is defined as

$$
R _ { p } ^ { p e r c } ( D ) \triangleq \operatorname* { m i n } _ { P ( \hat { W } | V ) : \mathbb { E } [ d _ { p } ( { \underline { { W } } } , \hat { \underline { { W } } } ) ] \leq D } I _ { p } ( { \underline { { W } } } ; V ) ,\tag{138}
$$

where $\underline { { \hat { W } } }$ is an estimate of W based on V, and $d _ { p }$ is a distortion measure on pragmatic symbols.

This function quantifies the minimum rate (in bits per observation) required to achieve a given perceptual distortion D when the receiver is only interested in the optimal action class. It is particularly relevant in perception systems where the goal is to infer the correct action from sensory data, and the raw sensory stream can be compressed by exploiting the action equivalence classes.

2) Expressive Pragmatic Rate-Distortion Function: The expressive pragmatic rate-distortion function captures the fundamental limit of representing a syntactic source W using a pragmatic label V (e.g., a control command or action label) that is then used to reconstruct the original source. It is defined using the down expressive pragmatic mutual information $I _ { p } ( W ; \underline { { V } } )$

Definition 17 (Expressive Pragmatic Rate-Distortion Function). Let W be a syntactic source, and let V be a pragmatic variable (action label) that is a function of W. The expressive pragmatic rate-distortion function is defined as

$$
R _ { p } ^ { e x p r } ( D ) \triangleq \operatorname* { m i n } _ { P ( \hat { W } | \underline { { V } } ) : \mathbb { E } [ d _ { s } ( W , \hat { W } ) ] \leq D } I _ { p } ( W ; \underline { { V } } ) ,\tag{139}
$$

where $\hat { W }$ is a reconstruction of W based on V, and $d _ { s }$ is a distortion measure on the syntactic symbols (or on the corresponding semantics).

This function quantifies the minimum rate (in bits per action label) required to achieve a given reconstruction distortion when the only information available about the source is the pragmatic action label. It is relevant in actuation or control systems where a low-bandwidth command channel must convey enough information to reconstruct the intended state or message.

3) Prospective Perceptual Pragmatic Achievable Rate: The prospective perceptual pragmatic achievable rate is the dual of the perceptual rate-distortion function. It quantifies the maximum rate at which information about the pragmatic action W can be reliably conveyed through an observation V, when the receiver has access to the full syntactic observation but is restricted to using only the pragmatic information. It is defined using the up perceptual pragmatic mutual information $I ^ { p } ( \underline { { W } } ; V )$ .

Definition 18 (Prospective Perceptual Pragmatic Achievable Rate). Let W be the pragmatic variable and V the observation. The prospective perceptual pragmatic achievable rate is defined as

$$
C _ { p } ^ { p - p e r c } ( P ) \triangleq \operatorname* { m a x } _ { p ( v | \underline { { w } } ) : \mathbb { E } [ c ( { W } ) ] \leq P } I ^ { p } ( \underline { { W } } ; V ) ,\tag{140}
$$

where c(W) is $a$ cost function on the pragmatic source $( e . g .$ , sensing power or bandwidth), and the maximization is over all conditional distributions $p ( v | \underline { { w } } )$ satisfying the resource constraint.

This achievable rate represents the maximum amount of pragmatic information that can be extracted from the observation when the observation itself is not coarsened. It provides an upper bound on the rate at which a sensor can convey actionable information, given a resource budget. The superscript “p-perc” denotes “prospective perceptual”.

4) Prospective Expressive Pragmatic Achievable Rate: The prospective expressive pragmatic achievable rate is the dual of the expressive rate-distortion function. It quantifies the maximum rate at which a pragmatic label $\underline { { V } }$ can convey information about the syntactic source W, when the receiver has access to the full syntactic source but is restricted to using only the pragmatic label. It is defined using the up expressive pragmatic mutual information $I ^ { p } ( W ; \underline { { V } } )$

Definition 19 (Prospective Expressive Pragmatic Achievable Rate). Let W be the syntactic source and $\underline { { V } }$ the pragmatic label. The prospective expressive pragmatic achievable rate is defined as

$$
C _ { p } ^ { p - e x p r } ( P ) \triangleq \operatorname* { m a x } _ { p ( \underline { { v } } | w ) : \mathbb { E } \left[ c ( W ) \right] \leq P } I ^ { p } ( W ; \underline { { V } } ) ,\tag{141}
$$

where $c ( W )$ is a cost function on the source $( e . g .$ ., transmit power), and the maximization is over all conditional distributions $p ( \underline { { v } } | w )$ satisfying the resource constraint.

This achievable rate represents the maximum rate at which a command or action label can convey information about the source, given a resource budget. It is relevant in scenarios where a low-dimensional action space is used to encode high-dimensional source information, and the system designer wants to know the best possible expressive performance under resource limits. The superscript “p-expr” denotes “prospective expressive”.

The four single-sided measures are related through the data processing inequality and the mutual information hierarchy, yielding the following bounds:

$$
\left\{ \begin{array} { l l } { R _ { p } ^ { \mathrm { p e r c } } ( D ) \geq R _ { p } ( D ) , } \\ { R _ { p } ^ { \mathrm { e x p r } } ( D ) \geq R _ { p } ( D ) , } \\ { C _ { p } ^ { \mathrm { p - p e r c } } ( P ) \leq C _ { p } ( P ) , } \\ { C _ { p } ^ { \mathrm { p - e x p r } } ( P ) \leq C _ { p } ( P ) . } \end{array} \right.\tag{142}
$$

These inequalities indicate that coarsening only one side preserves more syntactic distinctions, thus requiring higher rates for the same distortion and yielding lower achievable rates for the same resource.

Such single-sided measures are particularly useful in asymmetric systems, including:

• Perceptual (e.g., cameras, LIDAR): inferring actions from observations without coarsening the observation.

• Expressive (e.g., remote control, teleoperation): conveying intent via low-dimensional action labels.

• Edge AI: feature extraction at sensors (perceptual) and reconstruction at the central processor (expressive).

Together with their bilateral counterparts, these measures provide a complete pragmatic information theoretic framework for designing task-oriented communication systems with asymmetric constraints.

## VI. PRAGMATIC COST OF INFORMATION AND PRAGMATIC VALUE OF INFORMATION

In this section, we introduce the pragmatic value of information (VoI) and the pragmatic cost of information (CoI) as the benefit and resource-cost duals to the rate-distortion and capacity functions, respectively. We then combine these dualities into a Lagrangian framework for crosslayer optimization of pragmatic communication systems.

## A. Pragmatic Value of Information

The concept of the value of information was originally introduced by Stratonovich [8, 9] in the context of statistical decision theory, where it measures the maximum expected utility gain that a decision-maker can obtain by acquiring additional information before making a decision.

Classical VoI is defined as the difference between the expected utility with the observation and the expected utility without it, providing a quantitative basis for evaluating whether information acquisition is worthwhile.

The pragmatic value of information extends this classical notion to the pragmatic level by considering both the state and the observation through the lens of the isoteleia mapping. Unlike the classical VoI, which treats the raw state $X$ and raw observation $Y$ as the basis for decisionmaking, the pragmatic VoI recognizes that what ultimately matters is not the exact state or the exact observation, but their pragmatic implications—the optimal actions they induce. Both the state and the observation are therefore coarsened to their pragmatic equivalence classes via the isoteleia mapping, reflecting the principle of equifinality: distinct states or observations that lead to the same optimal action are pragmatically equivalent

Let $X \in { \mathcal { X } }$ be the state variable, $A \in { \mathcal { A } }$ the action, and $U ( X , A )$ the utility function. Let $\underline { { \boldsymbol X } }$ and $\underline { { Y } }$ be the pragmatic variables induced by the isoteleia mappings $e _ { X } : \underline { { { \mathcal { X } } } } \to 2 ^ { \tilde { \mathcal { X } } }$ and $e _ { Y } : \underline { { \mathcal { V } } } \to 2 ^ { \tilde { \mathcal { V } } }$ , respectively (composed with the synonymous mappings). The pragmatic variable $\underline { { \boldsymbol X } }$ groups states that lead to the same optimal action, while $\underline { { Y } }$ groups observations that are pragmatically equivalent. The receiver, upon observing the pragmatic observation $\underline { { Y } } ,$ chooses the action that maximizes the expected utility conditioned on $\underline { { Y } } ,$ where the expectation is taken over the pragmatic state $\underline { { X } } \mathrm { : }$

$$
a ^ { * } ( \underline { { Y } } ) = \arg \operatorname* { m a x } _ { a \in \mathcal { A } } \mathbb { E } _ { \underline { { X } } | \underline { { Y } } } [ U ( X , a ) ] ,\tag{143}
$$

where $X$ is any representative state in the pragmatic class $\underline { { X } } .$ , and the expectation is well-defined because all states in the same pragmatic class yield the same optimal action.

The baseline utility, without any information, is obtained by choosing the prior-optimal action based on the pragmatic state distribution:

$$
a _ { 0 } ^ { * } = \arg \operatorname* { m a x } _ { a \in \mathcal { A } } \mathbb { E } _ { \underline { { X } } } [ U ( X , a ) ] , \qquad U _ { 0 } = \mathbb { E } _ { \underline { { X } } } [ U ( X , a _ { 0 } ^ { * } ) ] .\tag{144}
$$

Definition 20 (Pragmatic Value of Information). The pragmatic value of information is defined as the expected utility gain obtained when the receiver makes decisions based on the pragmatic observation $\underline { { Y } } ,$ where both the state and the observation have been coarsened to their pragmatic equivalence classes:

$$
\begin{array} { r } { \mathrm { V o I } _ { p } \triangleq \mathbb { E } _ { \underline { { X } } , \underline { { Y } } } \big [ U ( X , a ^ { * } ( \underline { { Y } } ) ) \big ] - U _ { 0 } , } \end{array}\tag{145}
$$

where the expectation is taken over the joint distribution of the pragmatic state and pragmatic observation, and X is any representative state in the pragmatic class $\underline { { X } } .$ . This measures the utility gain when both the state and the observation are evaluated at the pragmatic level, that is, when decisions are based on the optimal action classes rather than on raw states or observations.

This bilateral coarse-graining is essential for a proper pragmatic characterization of information value. By coarsening both the state and the observation, the pragmatic VoI captures the decisionrelevant information that survives after discarding all distinctions that do not affect the optimal action. The classical VoI, which operates on raw states and observations, is a special case of the pragmatic VoI when the isoteleia mappings are trivial (identity).

The pragmatic VoI satisfies the following basic properties.

Theorem 19 (Non-negativity). $\mathrm { V o I } _ { p } \geq 0 .$ , with equality if and only if the pragmatic observation Y provides no useful information for improving the decision, i.e., $a ^ { * } ( \underline { { Y } } ) = a _ { 0 } ^ { * }$ almost surely.

Proof: For any realization of $\underline { { Y } } ,$ the optimal posterior action yields an expected utility at least as high as the prior-optimal action:

$$
\operatorname* { m a x } _ { a } \mathbb { E } _ { \underline { { X } } | \underline { { Y } } } [ U ( X , a ) ] \geq \mathbb { E } _ { \underline { { X } } | \underline { { Y } } } [ U ( X , a _ { 0 } ^ { * } ) ] .\tag{146}
$$

Taking expectation over Y and subtracting $U _ { 0 } = \mathbb { E } _ { \underline { { X } } } [ U ( X , a _ { 0 } ^ { * } ) ]$ yields the result.

Theorem 20 (Monotonicity under Refinement). If the pragmatic partition of the observation is refined $( i . e . , \ \underline { { Y } } _ { 2 }$ is a finer pragmatic partition than $\underline { { Y } } _ { 1 } ) ,$ , then

$$
\operatorname { V o I } _ { p } ( \underline { { Y } } _ { 1 } ) \leq \operatorname { V o I } _ { p } ( \underline { { Y } } _ { 2 } ) .\tag{147}
$$

This follows directly from the data processing inequality for utility-based decisions, as a finer pragmatic observation provides no less information about the pragmatic state than a coarser one.

Theorem 21 (Concavity in Rate). When the pragmatic observation is obtained through a channel with rate R, the maximum achievable pragmatic value, defined as

$$
\Phi _ { p } ( R ) \triangleq \operatorname* { m a x } _ { P ( \underline { { Y } } | \underline { { X } } ) : I _ { p } ( \underline { { X } } ; \underline { { Y } } ) \leq R } \mathrm { V o I } _ { p } ,\tag{148}
$$

where $I _ { p } ( \underline { { X } } ; \underline { { Y } } )$ is the down pragmatic mutual information between the pragmatic state and the pragmatic observation, is a non-decreasing concave function of R.

Lemma 4 (Constructive Concavity of the Upper Concave Envelope). Let

$$
\mathcal { R } = \left\{ ( R , V ) \ : | \ : R \geq 0 , \ : V \leq \Phi _ { p } ( R ) \right\}
$$

be the achievable rate–value region of the pragmatic information system, where $\Phi _ { p } ( R )$ is the original (possibly non-concave) value function. Define its upper concave envelope (i.e., the upper boundary of the convex hull) as

$$
\overline { { \Phi } } _ { p } ( R ) \triangleq \operatorname* { s u p } \big \{ V \mid ( R , V ) \in \operatorname { C o n v } ( \mathcal { R } ) \big \} ,\tag{149}
$$

where Conv(R) denotes the convex hull of R. Then $\overline { { \Phi } } _ { p } ( R )$ is a non-decreasing concave function on $R \geq 0 _ { : }$ , and it satisfies $\overline { { \Phi } } _ { p } ( R ) \geq \Phi _ { p } ( R )$ for all R. If R is already convex, then equality holds.

Proof: Let $( R _ { 1 } , V _ { 1 } )$ and $( R _ { 2 } , V _ { 2 } )$ be any two points in Conv(R) lying on the upper boundary, i.e., $V _ { 1 } = \overline { { \Phi } } _ { p } ( R _ { 1 } )$ and $V _ { 2 } = \overline { { \Phi } } _ { p } ( R _ { 2 } )$ . For any $\lambda \in [ 0 , 1 ]$ , consider the convex combination

$$
( R _ { \lambda } , V _ { \lambda } ) = \lambda ( R _ { 1 } , V _ { 1 } ) + ( 1 - \lambda ) ( R _ { 2 } , V _ { 2 } ) = \big ( \lambda R _ { 1 } + ( 1 - \lambda ) R _ { 2 } , \lambda V _ { 1 } + ( 1 - \lambda ) V _ { 2 } \big ) .
$$

By definition of the convex hull, $( R _ { \lambda } , V _ { \lambda } ) \in \mathrm { C o n v } ( \mathscr { R } )$ . Hence, by the supremum property of $\overline { { \Phi } } _ { p } ( R _ { \lambda } )$ , we have

$$
\begin{array} { r } { \overline { { \Phi } } _ { p } \big ( \lambda R _ { 1 } + ( 1 - \lambda ) R _ { 2 } \big ) \geq \lambda \overline { { \Phi } } _ { p } ( R _ { 1 } ) + ( 1 - \lambda ) \overline { { \Phi } } _ { p } ( R _ { 2 } ) , } \end{array}
$$

which is exactly the definition of concavity. Monotonicity is obvious because a larger rate allows a larger set of achievable values; and since the convex hull contains the original set, $\overline { { \Phi } } _ { p } ( R ) \geq \Phi _ { p } ( R )$ . If R is itself convex, its convex hull coincides with ${ \mathcal { R } } ,$ so equality holds. □

Remark 7. The key distinction between the pragmatic VoI and the classical VoI lies in the level of abstraction at which both the state and the observation are evaluated. Classical VoI treats every distinction in the state and observation as potentially valuable; pragmatic VoI recognizes that only distinctions that affect the optimal action matter. By coarsening both the state and the observation to their pragmatic equivalence classes, the pragmatic VoI provides a more parsimonious and task-relevant measure of information value. This bilateral coarse-graining also aligns naturally with the pragmatic rate-distortion and capacity-cost dualities established in Sections VI.A and VI.B.

The pragmatic value of information is the natural decision-theoretic dual to the pragmatic rate-distortion function. Recall from Section V-B that the pragmatic rate-distortion function is defined as $\begin{array} { r } { R _ { p } ( D ) = \operatorname* { m i n } _ { P ( \hat { X } | X ) : \mathbb { E } [ d _ { p } ( \underline { { X } } , \hat { \underline { { X } } } ) ] \leq D } I _ { p } ( \underline { { X } } ; \underline { { \hat { X } } } ) } \end{array}$ , where $d _ { p }$ is a distortion measure defined on pragmatic symbols.

The duality manifests through the following variational relationship.

Theorem 22 (Variational Duality). For a given utility function $U ,$ there exists a concave function $\Psi _ { p }$ such that

$$
\Phi _ { p } ( R ) = \operatorname* { m a x } _ { D \geq 0 } \big \{ \Psi _ { p } ( D ) - \lambda R _ { p } ( D ) \big \} ,\tag{150}
$$

where $\lambda > 0$ is a Lagrange multiplier. Conversely,

$$
R _ { p } ( D ) = \operatorname* { m a x } _ { \lambda \geq 0 } \big \{ \lambda D - \Phi _ { p } ^ { * } ( \lambda ) \big \} ,\tag{151}
$$

where $\Phi _ { p } ^ { * }$ is the concave conjugate of $\Phi _ { p } .$ This establishes a Legendre–Fenchel duality between the value-rate function $\Phi _ { p } ( R )$ and the rate-distortion function $R _ { p } ( D )$ .

Example 4 (Autonomous Vehicle (Continued)). We revisit the two-vehicle autonomous driving scenario from Sections III.B and IV.B. The state X is the traffic light state (four possible values), and the observation Y is the corresponding signal. The pragmatic mapping merges “Stop-short” and “Stop-long” into a single “Stop” action. The utility is 10 for correct action, 0 otherwise. The baseline utility (prior-optimal action “Stop”) is $U _ { 0 } = 1 1$ (combined for two vehicles). The computed pragmatic VoI is: $\mathrm { V o I } _ { p } \approx 0 . 5$ utils. This low value reflects the fact that the binary pragmatic observation (Go/Stop) resolves little decision uncertainty. In contrast, the classical VoI (using full syntactic signal) is about 4.2 utils.

The duality with rate-distortion can be illustrated by noting that to achieve a pragmatic distortion D (e.g., probability of wrong action), the required rate $R _ { p } ( D )$ must be at least the rate that enables the corresponding VoI. For instance, if we require $\mathrm { V o I } _ { p } \geq 4 . 0$ , the rate must exceed the value $R _ { p } ( D ^ { * } )$ where $D ^ { * }$ is the distortion corresponding to that utility level.

## B. Pragmatic Cost of Information

While the value of information quantifies the benefit, the pragmatic cost of information quantifies the resource expenditure required to convey information at a given pragmatic rate. Unlike VoI, which depends on the utility function and the task, CoI is determined solely by the physical communication channel and the pragmatic abstraction (the isoteleic volume).

Consider a communication channel with input X, output $Y .$ , and a joint reification mapping $g _ { X Y } : \underline { { \mathcal { X } } } \times \underline { { \mathcal { Y } } } \to 2 ^ { \mathcal { X } \times \mathcal { Y } }$ that defines the pragmatic equivalence classes. Let the resource consumption be measured by a cost function $c ( x )$ on the channel input (e.g., transmit power $\mathbb { E } [ | X | ^ { 2 } ]$ , or for discrete channels, the number of channel uses or energy per symbol).

Lemma 5 (Concavity of the Pragmatic Capacity–Resource Function). Let $P \geq 0$ denote the available communication resource (e.g., power) and define the pragmatic capacity under resource constraint P as

$$
C _ { p } ( P ) \triangleq \operatorname* { m a x } _ { p ( x ) : \mathbb { E } \left[ c ( X ) \right] \leq P } I ^ { p } ( \underline { { X } } ; \underline { { Y } } ) ,\tag{152}
$$

where $c ( X )$ is a non-negative cost function and $I ^ { p } ( \underline { { X } } ; \underline { { Y } } )$ is the up-pragmatic mutual information.   
Then $C _ { p } ( P )$ is a non-decreasing concave function of P.

Proof: For fixed channel and pragmatic mapping, the mutual information $I ^ { p } ( \underline { { X } } ; \underline { { Y } } )$ is a concave function of the input distribution $p ( x )$ (this follows from the joint convexity of the KL divergence and the fact that the pragmatic aggregation is a linear operation on probabilities). The resource constraint $\mathbb { E } [ c ( X ) ] \leq P$ defines a convex set of input distributions.

Let $P _ { 1 } , P _ { 2 } \ge 0$ and $\lambda \in [ 0 , 1 ]$ . Choose distributions $p _ { 1 }$ and $p _ { 2 }$ that attain the maxima for $P _ { 1 }$ and $P _ { 2 }$ , respectively, i.e.,

$$
C _ { p } ( P _ { i } ) = I ^ { p } ( p _ { i } ) \quad ( i = 1 , 2 ) , \qquad \mathbb { E } _ { p _ { i } } [ c ( X ) ] \leq P _ { i } .
$$

Consider the mixture distribution $p _ { \lambda } = \lambda p _ { 1 } + ( 1 - \lambda ) p _ { 2 }$ . Its average cost is

$$
\begin{array} { r } { \mathbb { E } _ { p _ { \lambda } } [ c ( X ) ] = \lambda \mathbb { E } _ { p _ { 1 } } [ c ( X ) ] + ( 1 - \lambda ) \mathbb { E } _ { p _ { 2 } } [ c ( X ) ] \le \lambda P _ { 1 } + ( 1 - \lambda ) P _ { 2 } , } \end{array}
$$

so $p _ { \lambda }$ is feasible for the resource level $\lambda P _ { 1 } + ( 1 - \lambda ) P _ { 2 }$ . By concavity of $I ^ { p }$ in the distribution,

$$
I ^ { p } ( p _ { \lambda } ) \geq \lambda I ^ { p } ( p _ { 1 } ) + ( 1 - \lambda ) I ^ { p } ( p _ { 2 } ) = \lambda C _ { p } ( P _ { 1 } ) + ( 1 - \lambda ) C _ { p } ( P _ { 2 } ) .
$$

Since $C _ { p } ( \lambda P _ { 1 } + ( 1 - \lambda ) P _ { 2 } )$ is the maximum over all feasible distributions, we have

$$
C _ { p } \big ( \lambda P _ { 1 } + ( 1 - \lambda ) P _ { 2 } \big ) \ge I ^ { p } ( p _ { \lambda } ) \ge \lambda C _ { p } ( P _ { 1 } ) + ( 1 - \lambda ) C _ { p } ( P _ { 2 } ) .
$$

Thus $C _ { p }$ is concave. Monotonicity follows immediately from the fact that a larger resource set contains smaller ones. □

The pragmatic cost of information is defined as the inverse of this capacity-resource relationship.

Definition 21 (Pragmatic Cost of Information). For a given pragmatic rate $R ,$ the pragmatic cost of information is the minimum resource consumption required to achieve that rate:

$$
\operatorname { C o I } _ { p } ( R ) \triangleq \operatorname* { m i n } _ { p ( x ) : I ^ { p } ( \underline { { X } } ; \underline { { Y } } ) \geq R } \mathbb { E } [ c ( X ) ] .\tag{153}
$$

This definition is completely general: the resource $c ( X )$ can be power, bandwidth, time, number of channel uses, or any other cost measure. It does not depend on the utility function or the task; it is purely a property of the physical channel and the pragmatic abstraction.

The pragmatic cost of information satisfies the following fundamental properties.

Theorem 23 (Non-negativity and Monotonicity). $\mathrm { C o I } _ { p } ( R ) \geq 0 \mathrm { : }$ , with $\mathrm { C o I } _ { p } ( 0 ) = 0 .$ . It is strictly increasing in R for $R > 0 .$

Lemma 6 (Convexity of the Inverse of an Increasing Concave Function). Let $f : \mathbb { R } ^ { + } \to \mathbb { R } ^ { + }$ be a strictly increasing concave function. Then its inverse function $f ^ { - 1 }$ is convex on the range of $f .$

Proof: Let $R _ { 1 } , R _ { 2 } \geq 0$ be in the range of $f ,$ and let $P _ { 1 } = f ^ { - 1 } ( R _ { 1 } ) , P _ { 2 } = f ^ { - 1 } ( R _ { 2 } )$ . For any $\lambda \in [ 0 , 1 ]$ , define

$$
R _ { \lambda } = \lambda R _ { 1 } + ( 1 - \lambda ) R _ { 2 } , \qquad P _ { \lambda } = \lambda P _ { 1 } + ( 1 - \lambda ) P _ { 2 } .
$$

By concavity of $f ,$

$$
f ( P _ { \lambda } ) \geq \lambda f ( P _ { 1 } ) + ( 1 - \lambda ) f ( P _ { 2 } ) = \lambda R _ { 1 } + ( 1 - \lambda ) R _ { 2 } = R _ { \lambda } .
$$

Since $f$ is strictly increasing, applying $f ^ { - 1 }$ to both sides preserves the inequality:

$$
P _ { \lambda } \geq f ^ { - 1 } ( R _ { \lambda } ) .
$$

Substituting the definition of $P _ { \lambda }$ yields

$$
\lambda f ^ { - 1 } ( R _ { 1 } ) + ( 1 - \lambda ) f ^ { - 1 } ( R _ { 2 } ) \geq f ^ { - 1 } \big ( \lambda R _ { 1 } + ( 1 - \lambda ) R _ { 2 } \big ) ,
$$

which is exactly the definition of convexity.

Theorem 24 (Convexity of the Pragmatic Cost of Information). Let $C _ { p } ( P )$ be the pragmatic capacity as a function of the resource P. Assume that $C _ { p } ( P )$ is concave and strictly increasing in P. Then the pragmatic cost of information $\mathrm { C o I } _ { p } ( R ) = C _ { p } ^ { - 1 } ( R )$ is a convex function of R on the achievable rate region.

Proof: By Lemma 6, the inverse of any strictly increasing concave function is convex. Applying this result with $f = C _ { p }$ immediately gives that $\mathrm { C o I } _ { p } ( R ) = C _ { p } ^ { - 1 } ( R )$ is convex.

For completeness, we restate the argument in detail. Let $R _ { 1 } , R _ { 2 }$ be two rates and let $P _ { i } =$ $\operatorname { C o I } _ { p } ( R _ { i } )$ for i = 1, 2. For any $\lambda \in [ 0 , 1 ]$ , set

$$
R _ { \lambda } = \lambda R _ { 1 } + ( 1 - \lambda ) R _ { 2 } , \qquad P _ { \lambda } = \lambda P _ { 1 } + ( 1 - \lambda ) P _ { 2 } .
$$

Because $C _ { p }$ is concave,

$$
C _ { p } ( P _ { \lambda } ) \geq \lambda C _ { p } ( P _ { 1 } ) + ( 1 - \lambda ) C _ { p } ( P _ { 2 } ) = \lambda R _ { 1 } + ( 1 - \lambda ) R _ { 2 } = R _ { \lambda } .
$$

Since $C _ { p }$ is strictly increasing, its inverse is also strictly increasing, so

$$
P _ { \lambda } \ge C _ { p } ^ { - 1 } ( R _ { \lambda } ) = \operatorname { C o I } _ { p } ( R _ { \lambda } ) .
$$

Therefore,

$$
\lambda \mathrm { C o I } _ { p } ( R _ { 1 } ) + ( 1 - \lambda ) \mathrm { C o I } _ { p } ( R _ { 2 } ) \geq \mathrm { C o I } _ { p } \big ( \lambda R _ { 1 } + ( 1 - \lambda ) R _ { 2 } \big ) ,
$$

which proves convexity.

Remark 8. The convexity established above is weak convexity. Strict convexity would require $C _ { p } ( P )$ to be strictly concave (i.e., no linear segments) and the strict increasing assumption to hold globally. In practice, when zero-cost capacity plateaus exist, $\operatorname { C o I } _ { p } ( R )$ may have linear segments and hence is convex but not strictly convex.

Theorem 25 (Duality with Pragmatic Channel Capacity). The pragmatic cost of information and the pragmatic channel capacity are strict inverses:

$$
\operatorname { C o I } _ { p } ( R ) = C _ { p } ^ { - 1 } ( R ) , \qquad C _ { p } ( P ) = \operatorname { C o I } _ { p } ^ { - 1 } ( P ) .\tag{154}
$$

Thus, they form a perfect mathematical dual pair: one gives the maximum rate for a given resource, the other gives the minimum resource for a given rate.

## C. Lagrangian Duality

Having established the two dual pairs, (rate-distortion, value) and (capacity, cost), we now combine them into a unified Lagrangian framework. This framework allows cross-layer optimization where the rate R serves as the common variable linking the source (decision) side and the channel (transmission) side.

1) Source-Distortion-Value Lagrangian (Down Loop): The down loop concerns the trade-off between the distortion (or decision quality) and the rate required to achieve it. The Lagrangian for this loop is formulated by combining the rate-distortion function $R _ { p } ( D )$ and the value-rate function $\Phi _ { p } ( R )$

Definition 22 (Source-Distortion-Value Pragmatic Lagrangian). For a given distortion level D, the down-loop Lagrangian is

$$
{ \mathcal { L } } _ { \mathrm { S D V } } ( D ; \lambda _ { d } ) \triangleq \Phi _ { p } \big ( R _ { p } ( D ) \big ) - \lambda _ { d } R _ { p } ( D ) ,\tag{155}
$$

where $\lambda _ { d } > 0$ is the Lagrange multiplier representing the marginal cost of rate (in utility units per bit).

Maximizing $\mathcal { L } _ { \mathrm { S D V } }$ over D yields the optimal operating point that balances the value obtained from reducing distortion against the rate cost. The first-order condition is:

$$
\frac { d \Phi _ { p } } { d R } \cdot R _ { p } ^ { \prime } ( D ) - \lambda _ { d } R _ { p } ^ { \prime } ( D ) = 0 \quad \Longrightarrow \quad \frac { d \Phi _ { p } } { d R } = \lambda _ { d } ,\tag{156}
$$

since $R _ { p } ^ { \prime } ( D ) \neq 0$ . Thus, at optimality, the marginal value of rate equals the marginal rate cost. 2) Channel-Power-Cost Lagrangian (Up Loop): The up loop concerns the trade-off between the rate and the physical resource cost. Using the pragmatic capacity $C _ { p } ( P )$ and the cost function $P ( R ) = \mathrm { C o I } _ { p } ( R )$ , we define:

Definition 23 (Channel-Power-Cost Pragmatic Lagrangian). For a given resource budget P, the up-loop Lagrangian is

$$
\mathcal { L } _ { \mathrm { C P C } } ( P ; \gamma ) \triangleq \gamma C _ { p } ( P ) - P ,\tag{157}
$$

where $\gamma > 0$ is the Lagrange multiplier representing the shadow price of the resource (in utility units per unit resource).

Equivalently, in terms of rate R, using $P = \operatorname { C o I } _ { p } ( R )$ , the up-loop Lagrangian becomes

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { C P C } } ( R ; \gamma ) = \gamma R - \mathrm { C o I } _ { p } ( R ) . } \end{array}\tag{158}
$$

Maximizing over P (or R) gives the first-order condition:

$$
\gamma C _ { p } ^ { \prime } ( P ) = 1 \quad \Longrightarrow \quad \gamma = \frac { 1 } { C _ { p } ^ { \prime } ( P ) } = \mathrm { C o I } _ { p } ^ { \prime } ( R ) .\tag{159}
$$

Thus, at optimality, the shadow price of the resource equals the marginal cost of increasing the rate.

3) Global Cross-Layer Lagrangian and Behavioral Capacity: The two loops are coupled through the common rate R: the down loop requires a certain rate to achieve a given utility, while the up loop supplies that rate at a certain resource cost. The global optimization problem is to maximize the net benefit (value minus cost) over the rate. To unify the two layers, we introduce a global Lagrangian that incorporates a shadow price λ (in utility units per unit resource) to convert physical resource cost into utility-equivalent cost:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { g l o b a l } } ( R ; \lambda ) = \Phi _ { p } ( R ) - \lambda \mathrm { C o I } _ { p } ( R ) , } \end{array}\tag{160}
$$

where $\lambda > 0$ is the Lagrange multiplier representing the shadow price of the physical resource (e.g., power). It serves as the conversion factor between resource consumption and utility.

Theorem 26 (Concavity of the Global Pragmatic Lagrangian). Assume that the pragmatic value function $\Phi _ { p } ( R )$ is concave and the pragmatic cost function $\operatorname { C o I } _ { p } ( R )$ is convex on $R \geq 0$ . Then, for every fixed Lagrange multiplier $\lambda \geq 0 ,$ , the global pragmatic Lagrangian

$$
\mathcal { L } _ { g l o b a l } ( R ; \lambda ) \triangleq \Phi _ { p } ( R ) - \lambda \operatorname { C o I } _ { p } ( R )
$$

is a concave function of R.

Proof: Let $R _ { 1 } , R _ { 2 } \geq 0$ and let $\theta \in \left[ 0 , 1 \right]$ . Define $R _ { \theta } = \theta R _ { 1 } + ( 1 - \theta ) R _ { 2 }$

By concavity of $\Phi _ { p } ,$

$$
\Phi _ { p } ( R _ { \theta } ) \geq \theta \Phi _ { p } ( R _ { 1 } ) + ( 1 - \theta ) \Phi _ { p } ( R _ { 2 } ) .\tag{161}
$$

By convexity of $\mathrm { C o I } _ { p } .$

$$
\operatorname { C o I } _ { p } ( R _ { \theta } ) \leq \theta \operatorname { C o I } _ { p } ( R _ { 1 } ) + ( 1 - \theta ) \operatorname { C o I } _ { p } ( R _ { 2 } ) .\tag{162}
$$

Multiplying (162) by −λ (which is non-positive) reverses the inequality:

$$
- \lambda \mathrm { C o I } _ { p } ( R _ { \theta } ) \ge - \lambda \theta \mathrm { C o I } _ { p } ( R _ { 1 } ) - \lambda ( 1 - \theta ) \mathrm { C o I } _ { p } ( R _ { 2 } ) .\tag{163}
$$

Adding (161) and (163) yields

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { g l o b a l } } ( R _ { \theta } ; \lambda ) = \Phi _ { p } ( R _ { \theta } ) - \lambda \mathrm { C o I } _ { p } ( R _ { \theta } ) } \\ & { \qquad \ge \theta \big [ \Phi _ { p } ( R _ { 1 } ) - \lambda \mathrm { C o I } _ { p } ( R _ { 1 } ) \big ] + ( 1 - \theta ) \big [ \Phi _ { p } ( R _ { 2 } ) - \lambda \mathrm { C o I } _ { p } ( R _ { 2 } ) \big ] } \\ & { \qquad = \theta \mathcal { L } _ { \mathrm { g l o b a l } } ( R _ { 1 } ; \lambda ) + ( 1 - \theta ) \mathcal { L } _ { \mathrm { g l o b a l } } ( R _ { 2 } ; \lambda ) . } \end{array}\tag{164}
$$

Inequality (164) is precisely the definition of concavity.

Corollary 5. Under the assumptions of Theorem 26, if $\mathcal { L } _ { g l o b a l }$ is differentiable, then the first-order condition

$$
\Phi _ { p } ^ { \prime } ( R ^ { * } ) = \lambda \mathrm { C o I } _ { p } ^ { \prime } ( R ^ { * } )\tag{165}
$$

is sufficient for $R ^ { * }$ to be a global maximizer of $\mathcal { L } _ { g l o b a l } ( \cdot ; \lambda )$

Proof: For a differentiable concave function, any point at which the derivative vanishes is a global maximum. Hence (165) yields the global optimum. □

This condition states that the marginal value of increasing the rate (in utility per bit) must equal the marginal cost of increasing the rate (in utility per bit), where the marginal cost is the product of the resource shadow price and the marginal resource consumption per bit. When the physical cost $\operatorname { C o I } _ { p } ( R )$ is already expressed in utility units (i.e., the shadow price is normalized to unity), we recover the simplified form $\mathcal { L } _ { \mathrm { g l o b a l } } ( R ) = \Phi _ { p } ( R ) - \mathrm { C o I } _ { p } ( R )$ and $\Phi _ { p } ^ { \prime } ( R ^ { * } ) = \mathrm { C o I } _ { p } ^ { \prime } ( R ^ { * } )$ In general, however, the explicit λ is required to maintain dimensional consistency.

Theorem 27 (Inner Duality of the Two Lagrangians). Let $\lambda _ { d }$ be the multiplier from the down-loop (marginal cost of rate) and $\gamma$ be the multiplier from the up-loop (shadow price of resource). At the global optimum defined by (165), the following relationships hold:

$$
\gamma = \lambda , \qquad \lambda _ { d } = \lambda \operatorname { C o I } _ { p } ^ { \prime } ( R ^ { * } ) = \Phi _ { p } ^ { \prime } ( R ^ { * } ) .\tag{166}
$$

Thus, the global multiplier λ coincides with the up-loop shadow price $\gamma ,$ while the down-loop marginal rate cost $\lambda _ { d }$ equals the marginal value of rate, which is also $\lambda { \mathrm { C o I } } _ { p } ^ { \prime } ( R ^ { * } )$ . This establishes a consistent duality between the decision-theoretic and physical-layer perspectives.

Proof: From the down-loop optimality condition, $\Phi _ { p } ^ { \prime } ( R ^ { * } ) = \lambda _ { d }$ . From the up-loop condition, $\gamma = \operatorname { C o I } _ { p } ^ { \prime } ( R ^ { * } )$ . The global condition (165) gives $\Phi _ { p } ^ { \prime } ( R ^ { * } ) = \lambda \mathrm { C o I } _ { p } ^ { \prime } ( R ^ { * } )$ . Combining, we get $\lambda _ { d } = \lambda \gamma$ , and since $\gamma = \operatorname { C o I } _ { p } ^ { \prime } ( R ^ { * } )$ ), it follows that $\lambda _ { d } = \lambda \gamma$ . To satisfy the global condition, we must have $\lambda = \gamma$ (otherwise the marginal value equation would not hold for all R). Hence, $\lambda = \gamma$ and $\lambda _ { d } = \lambda \mathrm { C o I } _ { p } ^ { \prime } ( R ^ { \ast } ) = \Phi _ { p } ^ { \prime } ( R ^ { \ast } )$ □

Definition 24 (Pragmatic Efficiency Bound (Behavioral Capacity)). For a given resource shadow price $\lambda \geq 0$ , the pragmatic efficiency functional $\mathcal { E } _ { p } ( \lambda )$ is defined as the supreme net benefit

achievable by the system:

$$
\mathcal { E } _ { p } ( \lambda ) \triangleq \operatorname* { s u p } _ { R \geq 0 } \left[ \Phi _ { p } ( R ) - \lambda \cdot \operatorname { C o I } _ { p } ( R ) \right] .\tag{167}
$$

This quantity represents the maximum net utility that an intelligent agent can extract from its environment per unit of time, after accounting for the physical cost of sensing, communication, and computation. It constitutes the behavioral capacity of the system under resource pricing λ, and serves as the master performance metric for any resource-constrained goal-directed system.

Remark 9. When the pragmatic rate R is continuous and the function $\Phi _ { p } ( R ) - \lambda \mathrm { C o I } _ { p } ( R )$ is strictly concave (which holds under the assumptions of Theorem 26), the supremum in (167) is attained at a unique point $R ^ { * }$ satisfying the marginal condition $\Phi _ { p } ^ { \prime } ( R ^ { * } ) = \lambda \mathrm { C o I } _ { p } ^ { \prime } ( R ^ { * } )$ . Consequently, in all operational contexts, we may write

$$
\mathcal { E } _ { p } ( \lambda ) = \operatorname* { m a x } _ { R \geq 0 } \mathcal { L } _ { \mathrm { g l o b a l } } ( R ; \lambda ) ,
$$

where the maximum is understood in the sense of the unique global maximizer. In the absence of resource constraints $( \lambda = 0 ) , \ : \mathcal { E } _ { p } ( 0 )$ reduces to $\operatorname { s u p } _ { R } \Phi _ { p } ( R )$ , i.e., the system’s performance under unlimited resources.

The Lagrangian dual framework provides a principled method for resource allocation in pragmatic systems:

1) Decision-driven design: Given a target utility level, the down-loop Lagrangian determines the minimum rate required. The corresponding distortion D and rate R are obtained by solving $\operatorname* { m a x } _ { D } \mathcal { L } _ { \mathrm { S D V } }$

2) Resource-limited design: Given a resource budget P, the up-loop Lagrangian determines the maximum achievable rate. The optimal rate is found by maximizing $\mathcal { L } _ { \mathrm { C P C } }$

3) Global optimum: When both utility and cost are considered, for a given shadow price λ (or equivalently, a resource budget constraint), the rate $R ^ { * }$ from (165) yields the maximal net benefit. This rate can be implemented by choosing the appropriate distortion level (via source coding) and resource allocation (via channel coding) such that $R _ { p } ( D ^ { * } ) = C _ { p } ( P ^ { * } ) = R ^ { * }$

The framework also enables adaptive operation: as the channel quality or task requirements change, the system can re-optimize the Lagrangian to adjust the operating point.

In summary, the Lagrangian duality between the source-distortion-value and channel-power-cost Lagrangians provides a complete mathematical characterization of the fundamental trade-offs in pragmatic communication systems, unifying Shannon’s physical-layer limits with the decisiontheoretic value of information, while maintaining dimensional consistency through the explicit shadow price λ.

## D. Single-Sided Pragmatic Cost of Information and Pragmatic Value of Information

The single-sided pragmatic rate-distortion functions and achievable rates defined in Section V-C have natural dual counterparts in terms of information value and cost. These single-sided VoI and CoI measures capture the utility gains and resource expenditures when only one side of the communication link is subject to pragmatic abstraction. They provide a finer-grained analysis for asymmetric systems, such as sensing-to-action (perceptual) and command-to-execution (expressive) scenarios.

1) Perceptual Pragmatic Value of Information: The perceptual pragmatic value of information quantifies the maximum expected utility gain obtainable from an observation V when the decision is based on the pragmatic action $\underline { { W } } .$ It is the dual of the perceptual pragmatic rate-distortion function $R _ { p } ^ { \mathrm { p e r c } } ( D )$ , and is defined using the down perceptual pragmatic mutual information $I _ { p } ( \underline { { W } } ; V )$ .

Definition 25 (Perceptual Pragmatic Value of Information). For a given rate constraint $R ,$ the perceptual pragmatic value of information is defined as

$$
\operatorname { V o I } _ { p } ^ { p e r c } ( R ) \triangleq \operatorname* { m a x } _ { { P ( V | { \underline { { W } } } ) } : I _ { p } ( { \underline { { W } } } ; { V } ) \le R } \operatorname { V o I } _ { p } ,\tag{168}
$$

where $\mathrm { V o I } _ { p }$ is the pragmatic value defined in $E q .$ . (145), and the maximization is over all conditional distributions $P ( V | \underline { W } )$ satisfying the rate constraint.

This quantity represents the maximum utility gain achievable when the observation is constrained to carry at most R bits of information about the pragmatic action. It is non-decreasing and concave in $R ,$ and satisfies the variational duality:

$$
\mathrm { V o I } _ { p } ^ { \mathrm { p e r c } } ( R ) = \operatorname* { m a x } _ { D > 0 } \big \{ \Psi _ { p } ^ { \mathrm { p e r c } } ( D ) - \lambda R _ { p } ^ { \mathrm { p e r c } } ( D ) \big \} ,\tag{169}
$$

where $\Psi _ { p } ^ { \mathrm { p e r c } }$ is a concave function determined by the utility, and $\lambda > 0$ is the Lagrange multiplier.

2) Expressive Pragmatic Value of Information: The expressive pragmatic value of information quantifies the maximum expected utility gain obtainable from a pragmatic label $\underline V$ when reconstructing the syntactic source $W .$ . It is the dual of the expressive pragmatic rate-distortion function $R _ { p } ^ { \mathrm { e x p r } } ( D )$ , and is defined using the down expressive pragmatic mutual information $I _ { p } ( W ; \underline { { V } } )$

Definition 26 (Expressive Pragmatic Value of Information). For a given rate constraint $R ,$ the expressive pragmatic value of information is defined as

$$
\operatorname { V o I } _ { p } ^ { e x p r } ( R ) \triangleq \operatorname* { m a x } _ { \substack { P ( \underline { { V } } | W ) : I _ { p } ( W ; \underline { { V } } ) \le R } } \operatorname { V o I } _ { p } ,\tag{170}
$$

where the maximization is over all conditional distributions $P ( \underline { { V } } | W )$ satisfying the rate constraint and $\mathrm { V o I } _ { p }$ is evaluated with respect to the reconstructed source.

This quantity measures the utility gain achieved by using the pragmatic label as a compressed representation of the source, subject to a rate limit. It is also non-decreasing and concave in $R ,$ and satisfies the dual variational relationship with $R _ { p } ^ { \mathrm { e x p r } } ( D )$

3) Prospective Perceptual Pragmatic Cost of Information: The prospective perceptual pragmatic cost of information quantifies the minimum resource expenditure required to achieve a given level of perceptual pragmatic information rate. It is the dual of the prospective perceptual pragmatic achievable rate $C _ { p } ^ { \mathrm { p - p e r c } } ( P )$ , and is defined using the up perceptual pragmatic mutual information $I ^ { p } ( \underline { { W } } ; V )$

Definition 27 (Prospective Perceptual Pragmatic Cost of Information). For a given pragmatic rate $R ,$ the prospective perceptual pragmatic cost of information is defined as

$$
\mathrm { C o I } _ { p } ^ { p \cdot p e r c } ( R ) \stackrel { \triangle } { = } \operatorname* { m i n } _ { P ( V | { \cal W } ) : I ^ { p } ( { \underline { { W } } } ; V ) \geq R } { \mathbb { E } } [ c ( { \underline { { W } } } ) ] ,\tag{171}
$$

where $c ( \underline { { W } } )$ is the cost function on the pragmatic source $( e . g .$ , sensing power), and the minimization is over all conditional distributions $P ( V | \underline { W } )$ satisfying the rate constraint.

This quantity represents the minimum resource cost required to extract at least $R$ bits of pragmatic information from the observation. It is non-decreasing and convex in $R ,$ and is the strict inverse of the prospective perceptual achievable rate:

$$
\mathrm { C o I } _ { p } ^ { \mathrm { p - p e r c } } ( R ) = \big ( C _ { p } ^ { \mathrm { p - p e r c } } \big ) ^ { - 1 } ( R ) .\tag{172}
$$

4) Prospective Expressive Pragmatic Cost of Information: The prospective expressive pragmatic cost of information quantifies the minimum resource expenditure required to achieve a given level of expressive pragmatic information rate. It is the dual of the prospective expressive pragmatic achievable rate $C _ { p } ^ { \mathrm { p - e x p r } } ( P )$ , and is defined using the up expressive pragmatic mutual information $I ^ { p } ( W ; \underline { { V } } )$

Definition 28 (Prospective Expressive Pragmatic Cost of Information). For a given pragmatic rate R, the prospective expressive pragmatic cost of information is defined as

$$
\mathrm { C o I } _ { p } ^ { p - e x p r } ( R ) \triangleq \operatorname* { m i n } _ { P ( \underline { { V } } | W ) : I ^ { p } ( W ; \underline { { V } } ) \geq R } \mathbb { E } [ c ( W ) ] ,\tag{173}
$$

where $c ( W )$ is the cost function on the source $( e . g .$ ., transmit power), and the minimization is over all conditional distributions $P ( \underline { { V } } | W )$ satisfying the rate constraint.

This quantity represents the minimum resource cost required to convey at least R bits of source information through the pragmatic label. It is non-decreasing and convex in R, and is the strict inverse of the prospective expressive achievable rate:

$$
\mathrm { C o I } _ { p } ^ { \mathrm { p - e x p r } } ( R ) = \big ( C _ { p } ^ { \mathrm { p - e x p r } } \big ) ^ { - 1 } ( R ) .\tag{174}
$$

5) Relationships and Duality Structure: The four single-sided VoI and CoI measures form a complete duality framework that parallels the bilateral case, with the following key relationships:

1) Rate-Distortion ↔ Value Duality: For both perceptual and expressive paths, the value function is the Legendre–Fenchel dual of the corresponding rate-distortion function:

$$
\left\{ \begin{array} { l l } { \mathrm { V o I } _ { p } ^ { \mathrm { p e r c } } ( R ) = \displaystyle \operatorname* { m a x } _ { D \geq 0 } \big \{ \Psi _ { p } ^ { \mathrm { p e r c } } ( D ) - \lambda R _ { p } ^ { \mathrm { p e r c } } ( D ) \big \} , } \\ { \mathrm { V o I } _ { p } ^ { \mathrm { e x p r } } ( R ) = \displaystyle \operatorname* { m a x } _ { D > 0 } \big \{ \Psi _ { p } ^ { \mathrm { e x p r } } ( D ) - \lambda R _ { p } ^ { \mathrm { e x p r } } ( D ) \big \} . } \end{array} \right.\tag{175}
$$

2) Capacity ↔ Cost Duality: For both prospective perceptual and expressive paths, the cost function is the inverse of the corresponding achievable rate:

$$
\left\{ { \begin{array} { l l } { { \mathrm { C o I } } _ { p } ^ { \mathrm { p - p e r c } } ( R ) = \left( C _ { p } ^ { \mathrm { p - p e r c } } \right) ^ { - 1 } ( R ) , } \\ { { \mathrm { C o I } } _ { p } ^ { \mathrm { p - e x p r } } ( R ) = \left( C _ { p } ^ { \mathrm { p - e x p r } } \right) ^ { - 1 } ( R ) . } \end{array} } \right.\tag{176}
$$

3) Comparison with Bilateral Counterparts: Single-sided measures are bounded by their bilateral counterparts:

$$
\left\{ \begin{array} { c c } { \mathrm { V o I } _ { p } ^ { \mathrm { p e r c } } ( R ) \leq \mathrm { V o I } _ { p } ( R ) , } & { \mathrm { V o I } _ { p } ^ { \mathrm { e x p r } } ( R ) \leq \mathrm { V o I } _ { p } ( R ) , } \\ { \mathrm { C o I } _ { p } ^ { \mathrm { p - p e r c } } ( R ) \geq \mathrm { C o I } _ { p } ( R ) , } & { \mathrm { C o I } _ { p } ^ { \mathrm { p - e x p r } } ( R ) \geq \mathrm { C o I } _ { p } ( R ) , } \end{array} \right.\tag{177}
$$

where ${ \mathrm { V o I } } _ { p } ( R )$ and $\operatorname { C o I } _ { p } ( R )$ are the bilateral functions defined in Sections VI.A and VI.B.

TABLE VIII: Summary of bilateral and single-sided pragmatic information measures.
<table><tr><td>Measure</td><td>Abbrev.</td><td>Definition / Expression</td><td>Scenario</td></tr><tr><td>Bilateral VoI</td><td> ${ \mathrm { V o I } } _ { p } ( R )$ </td><td> $\operatorname* { m a x } _ { P ( \underline { { Y } } | \underline { { X } } ) : I _ { p } \leq R } \mathrm { V o I } _ { p }$ </td><td>Symmetric, both ends coarsened</td></tr><tr><td>Bilateral CoI</td><td> $\operatorname { C o I } _ { p } ( R )$ </td><td> $\operatorname* { m i n } _ { p ( x ) : I ^ { p } \geq R } \mathbb { E } [ c ( X ) ]$ </td><td>Symmetric, both ends coarsened</td></tr><tr><td>Perceptual VoI</td><td> $\mathrm { V o I } _ { p } ^ { \mathrm { p e r c } } ( R )$ </td><td> $\operatorname* { m a x } _ { P ( V | \underline { { W } } ) : I _ { p } ( \underline { { W } } ; V ) \leq R } \mathrm { V o I } _ { p }$ </td><td>Sensing: observation to action</td></tr><tr><td>Expressive VoI</td><td> $\mathrm { V o I } _ { p } ^ { \mathrm { e x p r } } ( R )$ </td><td> $\operatorname* { m a x } _ { P ( \underline { { V } } | W ) : I _ { p } ( W ; \underline { { V } } ) \leq R } \mathrm { V o I } _ { p }$ </td><td>Actuation: command to reconstruction</td></tr><tr><td>Prospective Per-</td><td> $\mathrm { C o I } _ { p } ^ { \mathrm { p - p e r c } } ( R )$ </td><td> $\operatorname* { m i n } _ { P ( V \mid W ) : I ^ { p } ( \underline { { W } } ; V ) \geq R } \mathbb { E } [ c ( \underline { { W } } ) ]$ </td><td>Sensing with full syntac-</td></tr><tr><td>ceptual CoI Prospective Ex-</td><td> $\mathrm { C o I } _ { p } ^ { \mathrm { p - e x p r } } ( R )$ </td><td></td><td>tic observation Actuation with full syn-</td></tr><tr><td>pressive CoI</td><td></td><td> $\operatorname* { m i n } _ { P ( \underline { { V } } | W ) : I ^ { p } ( W ; \underline { { V } } ) \geq R } \mathbb { E } [ c ( W ) ]$ </td><td>tactic source</td></tr></table>

Table VIII consolidates all bilateral and single-sided pragmatic information measures, their defining mutual informations, and their applicability in symmetric versus asymmetric systems. The perceptual path (sensing) coarsens only the source side, whereas the expressive path (actuation) coarsens only the reconstruction side. The prospective variants use the up mutual information and thus provide upper bounds on achievable rates.

Figure 9 illustrates the duality structure of pragmatic information measures, encompassing both bilateral and single-sided quantities. At the core lie the two fundamental dual pairs: $R _ { p } ( D )  \operatorname { V o I } _ { p } ( R )$ and $C _ { p } ( P )  \mathrm { C o I } _ { p } ( R )$ . Extending this duality to asymmetric scenarios, $R _ { p } ^ { \mathrm { p e r c } } ( D )  \mathrm { V o I } _ { p } ^ { \mathrm { p e r c } } ( R )$ and $R _ { p } ^ { \mathrm { e x p r } } ( D )  \operatorname { V o I } _ { p } ^ { \mathrm { e x p r } } ( R )$ capture the rate–value trade-off when only one side is coarsened. On the capacity–cost side, $C _ { p } ^ { \mathrm { p - p e r c } } ( P ) \  \ \mathrm { C o I } _ { p } ^ { \mathrm { p - p e r c } } ( R )$ and $C _ { p } ^ { \mathrm { p - e x p r } } ( P )  \mathrm { C o I } _ { p } ^ { \mathrm { p - e x p r } } ( R )$ complete the duality lattice, which collectively characterizes tradeoffs between rate, distortion, capacity, cost, and value in both symmetric and asymmetric pragmatic communication systems.

The single-sided VoI and CoI measures provide a precise toolkit for analyzing asymmetric pragmatic systems:

![](images/48cf5b8602bb018ba6ca0da46c6f31f2d514474f35bad643670c0281b8f36ec8.jpg)  
Fig. 9: Duality of pragmatic information measures.

• Perceptual systems (sensing): $\mathrm { V o I } _ { p } ^ { \mathrm { p e r c } }$ measures the utility gain of a sensor under a rate constraint, while $\mathrm { C o I } _ { p } ^ { \mathrm { p - p e r c } }$ quantifies the sensing power required to achieve a certain perceptual information rate.

• Expressive systems (actuation): $\mathrm { V o I } _ { p } ^ { \mathrm { e x p r } }$ measures the utility gain of a command channel under a rate constraint, while $\mathrm { C o I } _ { p } ^ { \mathrm { p - e x p r } }$ quantifies the transmit power required to convey a certain expressive information rate.

• Joint optimization: The single-sided measures can be combined with the bilateral Lagrangian framework to design systems where sensing and actuation have different resource constraints and utility contributions.

Together with the bilateral value and cost functions, these single-sided counterparts complete the pragmatic information-theoretic spectrum, enabling a comprehensive analysis of task-oriented communication systems with asymmetric constraints.

## VII. PRAGMATIC LOSSLESS SOURCE CODING

In this section, we investigate the problem of pragmatic lossless source coding. We extend the asymptotic equipartition property (AEP) to the pragmatic domain and introduce the pragmatic typical set, the isoteleia typical set, and the reified typical set. These concepts serve as the mathematical foundation for proving the pragmatic lossless source coding theorem.

## A. Asymptotic Equipartition Property and Pragmatic Typical Set

Specifically, we establish the syntactic, semantic, and pragmatic AEP theorems. Based on these results, we define the pragmatic typical set, the isoteleia typical set, and the reified typical set, and analyze their cardinality and probability properties. The hierarchical structure of these typical sets reveals that pragmatic abstraction, by merging semantic distinctions that do not affect the optimal terminal action, further reduces the number of typical sequences compared to semantic abstraction, thereby enabling more efficient lossless compression at the pragmatic level.

To establish the asymptotic equipartition property at the pragmatic level, we first introduce the sequential extensions of the fundamental mappings that bridge the syntactic, semantic, and pragmatic layers.

Let W be the syntactic alphabet with probability mass function $P ( W )$ , and let $\tilde { \mathcal W }$ be the semantic alphabet induced by the synonymous mapping $f : \tilde { \mathcal { W } } \to 2 ^ { \mathcal { W } }$ . Let W be the pragmatic alphabet induced by the isoteleia mapping $e : \underline { { \mathcal { W } } } \to 2 ^ { \tilde { \mathcal { W } } }$ according to a given utility function $U : \mathcal { W } \times \mathcal { A }  \mathbb { R }$ . The reification mapping $g = f \circ e : \underline { { \mathcal { W } } }  2 ^ { \mathcal { W } }$ directly bridges the pragmatic and syntactic layers.

Definition 29 (Sequential Synonymous Mapping). Let $f : \tilde { \mathcal { W } } \to 2 ^ { \mathcal { W } }$ be the synonymous mapping. Its n-th sequential extension $f ^ { n } : { \tilde { \mathcal { W } } } ^ { n } \longrightarrow 2 ^ { \mathcal { W } ^ { n } }$ is defined by

$$
f ^ { n } ( \tilde { w } ^ { n } ) \triangleq \prod _ { k = 1 } ^ { n } f ( \tilde { w } _ { k } ) = \{ w ^ { n } \in \mathcal W ^ { n } : w _ { k } \in f ( \tilde { w } _ { k } ) , k = 1 , \dots , n \} ,\tag{178}
$$

where $\tilde { w } ^ { n } = ( \tilde { w } _ { 1 } , \dots , \tilde { w } _ { n } )$

Definition 30 (Sequential Isoteleia Mapping). Let $e : \underline { { \mathcal { W } } } \to 2 ^ { \tilde { \mathcal { W } } }$ be the isoteleia mapping. Its n-th sequential extension $e ^ { n } : \underline { { { \mathcal { W } } } } ^ { n } \longrightarrow 2 ^ { \tilde { \mathcal { W } } ^ { n } }$ is defined by

$$
e ^ { n } ( \underline { { w } } ^ { n } ) \triangleq \prod _ { k = 1 } ^ { n } e ( \underline { { w } } _ { k } ) = \left\{ \tilde { w } ^ { n } \in \tilde { \mathcal { W } } ^ { n } : \tilde { w } _ { k } \in e ( \underline { { w } } _ { k } ) , k = 1 , \dots , n \right\} ,\tag{179}
$$

where $\underline { w } ^ { n } = ( \underline { w } _ { 1 } , \dots , \underline { w } _ { n } )$

Definition 31 (Sequential Reification Mapping). Let $g = f \circ e : \underline { { \mathcal { W } } }  2 ^ { \mathcal { W } }$ be the reification mapping. Its n-th sequential extensiong ${ \ v O } ^ { n } : \underline { { \mathcal { W } } } ^ { n } \longrightarrow 2 ^ { \mathcal { W } ^ { n } }$ is defined as the composition of the sequential isoteleia and synonymous mappings:

$$
g ^ { n } ( \underline { { w } } ^ { n } ) \triangleq ( f ^ { n } \circ e ^ { n } ) ( \underline { { w } } ^ { n } ) = \bigcup _ { \tilde { w } ^ { n } \in e ^ { n } ( \underline { { w } } ^ { n } ) } f ^ { n } ( \tilde { w } ^ { n } ) = \prod _ { k = 1 } ^ { n } g ( \underline { { w } } _ { k } ) .\tag{180}
$$

Equivalently,

$$
g ^ { n } ( \underline { { w } } ^ { n } ) = \{ w ^ { n } \in \mathcal { W } ^ { n } : w _ { k } \in g ( \underline { { w } } _ { k } ) , k = 1 , \dots , n \} .\tag{181}
$$

1) Three-Tier AEP Theorems: We now state the AEP at each of the three layers. The syntactic AEP is the classical result [1, 45]. The semantic AEP was established in [21, 22]. The pragmatic AEP follows analogously.

Theorem 28 (Three-Tier AEP). Let $( W _ { 1 } , W _ { 2 } , \ldots )$ be an i.i.d. syntactic sequence drawn according to $P ( W )$ , and let $( \tilde { W } _ { 1 } , \tilde { W } _ { 2 } , \dots )$ and $( \underline { { W } } _ { 1 } , \underline { { W } } _ { 2 } , \dots )$ be the associated semantic and pragmatic sequences under the sequential Synonymous Mapping $f ^ { n }$ and the sequential Reification Mapping $g ^ { n } = f ^ { n } \circ e ^ { n }$ , respectively. Then the following convergences hold in probability:

$$
\left\{ \begin{array} { l l } { - \displaystyle \frac 1 n \log P ( W _ { 1 } , \dots , W _ { n } ) \longrightarrow H ( W ) , } \\ { \displaystyle - \frac 1 n \log P ( \tilde { W } _ { 1 } , \dots , \tilde { W } _ { n } ) \longrightarrow H _ { s } ( \tilde { W } ) , } \\ { \displaystyle - \frac 1 n \log P ( \underline { { W } } _ { 1 } , \dots , \underline { { W } } _ { n } ) \longrightarrow H _ { p } ( \underline { { W } } ) , } \end{array} \right.\tag{182}
$$

where

$$
P ( W ^ { n } ) = \prod _ { k = 1 } ^ { n } P ( W _ { k } ) , \quad P ( \tilde { w } ^ { n } ) = \prod _ { k = 1 } ^ { n } P ( \tilde { w } _ { k } ) , \quad P ( \tilde { w } _ { k } ) = \sum _ { w _ { k } \in f ( \tilde { w } _ { k } ) } P ( w _ { k } ) ,
$$

and

$$
P ( \underline { { w } } ^ { n } ) = \prod _ { k = 1 } ^ { n } P ( \underline { { w } } _ { k } ) , \quad P ( \underline { { w } } _ { k } ) = \sum _ { w _ { k } \in g ( \underline { { w } } _ { k } ) } P ( w _ { k } ) .
$$

The syntactic convergence is the classical result [1, 45]; the semantic and pragmatic convergences follow by applying the weak law of large numbers to the aggregated probabilities over synonymous and reified equivalence classes, respectively.

The following two AEP theorems characterize the asymptotic relationship between adjacent layers; they follow directly from the three AEP theorems above.

Theorem 29 (Isoteleia AEP).

$$
- \frac { 1 } { n } \left[ \log P ( \tilde { W } ^ { n } ) - \log P ( \underline { { W } } ^ { n } ) \right] \longrightarrow H _ { s } ( \tilde { W } ) - H _ { p } ( \underline { { W } } )\tag{183}
$$

in probability.

Theorem 30 (Reified AEP).

$$
- \frac { 1 } { n } \left[ \log P ( W ^ { n } ) - \log P ( \underline { { W } } ^ { n } ) \right] \longrightarrow H ( W ) - H _ { p } ( \underline { { W } } )\tag{184}
$$

in probability.

2) Definitions of the Hierarchical Typical Sets: We now define the typical sets at each layer and the equivalence classes that partition them.

Definition 32 (Syntactic, Semantic, and Pragmatic Typical Sets). For $\epsilon > 0 ,$ , the syntactically typical set $A _ { \epsilon } ^ { ( n ) }$ , the semantically typical set $\tilde { A } _ { \epsilon } ^ { \left( n \right) }$ , and the pragmatic typical set $\underline { { A } } _ { \epsilon } ^ { ( n ) }$ are respectively defined as

$$
\left\{ \begin{array} { l l } { { A _ { \epsilon } ^ { ( n ) } \triangleq \left\{ w ^ { n } \in  { \mathcal { W } } ^ { n } : \left| - \displaystyle \frac { 1 } { n } \log P ( w ^ { n } ) - H ( W ) \right| < \epsilon \right\} , } } \\ { { \tilde { A } _ { \epsilon } ^ { ( n ) } \triangleq \left\{ \tilde { w } ^ { n } \in  { \mathcal { W } } ^ { n } : \left| - \displaystyle \frac { 1 } { n } \log P ( \tilde { w } ^ { n } ) - H _ { s } ( \tilde { W } ) \right| < \epsilon \right\} , } } \\ { { A _ { \epsilon } ^ { ( n ) } \triangleq \left\{ \underline { { w } } ^ { n } \in  { \mathcal { W } } ^ { n } : \left| - \displaystyle \frac { 1 } { n } \log P ( \underline { { w } } ^ { n } ) - H _ { p } ( \underline { { W } } ) \right| < \epsilon \right\} . } } \end{array} \right.\tag{185}
$$

Definition 33 (Synonymous Typical Set). For a given semantic typical sequence $\tilde { w } ^ { n } \in \tilde { A } _ { \epsilon } ^ { ( n ) }$ , the synonymous typical set $S _ { \epsilon } ^ { ( n ) } ( \tilde { w } ^ { n } )$ is defined as the set of syntactic sequences that map to $\tilde { w } ^ { n }$ under $f ^ { n }$ and are syntactically typical:

$$
\begin{array} { r l } & { S _ { \epsilon } ^ { ( n ) } ( \tilde { w } ^ { n } ) \triangleq \Bigg \{ w ^ { n } \in \mathcal { W } ^ { n } : w ^ { n } \in f ^ { n } ( \tilde { w } ^ { n } ) , } \\ & { \qquad \bigg | - \frac { 1 } { n } \log P ( w ^ { n } ) - H ( W ) \bigg | < \epsilon , } \\ & { \qquad \bigg | - \frac { 1 } { n } \log \frac { P ( w ^ { n } ) } { P ( \tilde { w } ^ { n } ) } - \big ( H ( W ) - H _ { s } ( \tilde { W } ) \big ) \bigg | < \epsilon \Bigg \} . } \end{array}\tag{186}
$$

The collection $\{ S _ { \epsilon } ^ { ( n ) } ( \tilde { w } ^ { n } ) \} _ { \tilde { w } ^ { n } \in \tilde { A } _ { \epsilon } ^ { ( n ) } }$ forms a partition of $A _ { \epsilon } ^ { ( n ) }$ :

$$
A _ { \epsilon } ^ { ( n ) } = \bigcup _ { \tilde { w } ^ { n } \in \tilde { A } _ { \epsilon } ^ { ( n ) } } S _ { \epsilon } ^ { ( n ) } ( \tilde { w } ^ { n } ) ,\tag{187}
$$

with disjointness for distinct $\tilde { w } ^ { n }$

Definition 34 (Isoteleia Typical Set). For a given pragmatic typical sequence $\underline { w } ^ { n } \in \underline { A } _ { \epsilon } ^ { ( n ) }$ , the Isoteleia typical set $E _ { \epsilon } ^ { ( n ) } ( \underline { { w } } ^ { n } )$ is defined as the set of semantic sequences that map to $\underline { w } ^ { n }$ under

$e ^ { n }$ and are semantically typical:

$$
\begin{array} { r l } & { E _ { \epsilon } ^ { ( n ) } ( \underline { { w } } ^ { n } ) \triangleq \Bigg \{ \tilde { w } ^ { n } \in \tilde { \mathcal { W } } ^ { n } : \tilde { w } ^ { n } \in e ^ { n } ( \underline { { w } } ^ { n } ) , } \\ & { \qquad \bigg | - \frac { 1 } { n } \log P ( \tilde { w } ^ { n } ) - H _ { s } ( \tilde { W } ) \bigg | < \epsilon , } \\ & { \qquad \bigg | - \frac { 1 } { n } \log \frac { P ( \tilde { w } ^ { n } ) } { P ( \underline { { w } } ^ { n } ) } - \big ( H _ { s } ( \tilde { W } ) - H _ { p } ( \underline { { W } } ) \big ) \bigg | < \epsilon \Bigg \} . } \end{array}\tag{188}
$$

The collection $\{ E _ { \epsilon } ^ { ( n ) } ( { \underline { { w } } } ^ { n } ) \} _ { { \underline { { w } } } ^ { n } \in { \underline { { A } } } _ { \epsilon } ^ { ( n ) } }$ forms a partition of $\tilde { A } _ { \epsilon } ^ { \left( n \right) }$ :

$$
\tilde { A } _ { \epsilon } ^ { ( n ) } = \bigcup _ { \underline { { w } } ^ { n } \in \underline { { A } } _ { \epsilon } ^ { ( n ) } } E _ { \epsilon } ^ { ( n ) } ( \underline { { w } } ^ { n } ) ,\tag{189}
$$

with disjointness for distinct $w ^ { n }$

Definition 35 (Reified Typical Set). For a given pragmatic typical sequence $\underline { w } ^ { n } \in \underline { A } _ { \epsilon } ^ { ( n ) }$ , the Reified typical set $B _ { \epsilon } ^ { ( n ) } ( \underline { { w } } ^ { n } )$ is defined as the set of syntactic sequences that map to $w ^ { n }$ under the composite mapping $g ^ { n } = f ^ { n } \circ e ^ { n }$ and are syntactically typical:

$$
\begin{array} { r l } & { B _ { \epsilon } ^ { ( n ) } ( \underline { { w } } ^ { n } ) \triangleq \Bigg \{ w ^ { n } \in \mathscr { W } ^ { n } : w ^ { n } \in g ^ { n } ( \underline { { w } } ^ { n } ) , } \\ & { \qquad \bigg | - \frac { 1 } { n } \log P ( w ^ { n } ) - H ( W ) \bigg | < \epsilon , } \\ & { \qquad \bigg | - \frac { 1 } { n } \log \frac { P ( w ^ { n } ) } { P ( \underline { { w } } ^ { n } ) } - \big ( H ( W ) - H _ { p } ( \underline { { W } } ) \big ) \bigg | < \epsilon \Bigg \} . } \end{array}\tag{190}
$$

The collection $\{ B _ { \epsilon } ^ { ( n ) } ( \underline { { w } } ^ { n } ) \} _ { \underline { { w } } ^ { n } \in \underline { { A } } _ { \epsilon } ^ { ( n ) } }$ forms a partition of $A _ { \epsilon } ^ { ( n ) }$ :

$$
A _ { \epsilon } ^ { ( n ) } = \bigcup _ { { \underline { { w } } } ^ { n } \in \underline { { A } } _ { \epsilon } ^ { ( n ) } } B _ { \epsilon } ^ { ( n ) } ( { \underline { { w } } } ^ { n } ) ,\tag{191}
$$

with disjointness for distinct $w ^ { n }$

The hierarchical relationships among the pragmatic, semantic, and syntactic typical sets, as well as their corresponding equivalence classes, are illustrated schematically in Fig. 10. In this two-level hierarchical partition: the synonymous mapping first refines the syntactic typical set into synonym classes; the isoteleia mapping then groups these classes into coarser Reified classes, each corresponding to a single pragmatic action. This nested structure directly reflects the composition $g ^ { n } = f ^ { n } \circ e ^ { n }$ and underpins the entropy reduction $H _ { p } ( \underline { { W } } ) \le H _ { s } ( \tilde { W } ) \le H ( W )$ . Moreover, each reified typical set is the union of synonymous typical sets over all semantic sequences in the corresponding isoteleia typical set:

$$
B _ { \epsilon } ^ { ( n ) } ( \underline { { w } } ^ { n } ) = \bigcup _ { \tilde { w } ^ { n } \in E _ { \epsilon } ^ { ( n ) } ( \underline { { w } } ^ { n } ) } S _ { \epsilon } ^ { ( n ) } ( \tilde { w } ^ { n } ) .
$$

![](images/30a180d2db514b48032e3befd02dd788faf4b3cabaad234e0b2c64ab911f50b3.jpg)  
Fig. 10: Hierarchical structure of syntactic, semantic, and pragmatic typical sets.

3) Properties of the Typical Sets: The syntactic and semantic typical sets satisfy the standard properties established in [1, 45] and [21, 22], respectively. For completeness, we list them below without proof.

Theorem 31 (Properties of the Syntactic Typical Set). For sufficiently large n:

1) For any $w ^ { n } \in A _ { \epsilon } ^ { ( n ) }$

$$
2 ^ { - n ( H ( W ) + \epsilon ) } \leq P ( w ^ { n } ) \leq 2 ^ { - n ( H ( W ) - \epsilon ) } .\tag{192}
$$

2) $\operatorname* { P r } \{ A _ { \epsilon } ^ { ( n ) } \} > 1 - \epsilon .$

$$
3 ) \ ( 1 - \epsilon ) 2 ^ { n ( H ( W ) - \epsilon ) } \leq | A _ { \epsilon } ^ { ( n ) } | \leq 2 ^ { n ( H ( W ) + \epsilon ) } .
$$

Theorem 32 (Properties of the Semantic Typical Set). For sufficiently large n:

1) For any $\tilde { w } ^ { n } \in \tilde { A } _ { \epsilon } ^ { ( n ) }$

$$
2 ^ { - n ( H _ { s } ( \tilde { W } ) + \epsilon ) } \leq P ( \tilde { w } ^ { n } ) \leq 2 ^ { - n ( H _ { s } ( \tilde { W } ) - \epsilon ) } .\tag{193}
$$

2) $\mathrm { P r } \{ \tilde { A } _ { \epsilon } ^ { ( n ) } \} > 1 - \epsilon .$

3) (1 − ϵ)2<sup>n(Hs(W˜</sup> <sup>)−ϵ)</sup> ≤ |A<sup>˜(n)</sup><sub>ϵ</sub> | ≤ 2<sup>n(Hs(W˜</sup> <sup>)+ϵ)</sup>.

We now establish the properties of the pragmatic typical set.

Theorem 33 (Properties of the Pragmatic Typical Set). For sufficiently large n:

1) For any ${ \underline { w } } ^ { n } \in { \underline { { A } } } _ { \epsilon } ^ { ( n ) }$

$$
2 ^ { - n ( H _ { p } ( { \underline { { W } } } ) + \epsilon ) } \leq P ( { \underline { { w } } } ^ { n } ) \leq 2 ^ { - n ( H _ { p } ( { \underline { { W } } } ) - \epsilon ) } .\tag{194}
$$

2) $\operatorname* { P r } \{ \underline { { A } } _ { \epsilon } ^ { ( n ) } \} > 1 - \epsilon .$

3) (1 − ϵ)2<sup>n(Hp(W)−ϵ)</sup> ≤ |A<sup>(n)</sup>| ≤ 2<sup>n(Hp(W)+ϵ)</sup>.

The proof of Theorem 33 follows the standard AEP counting argument. For the sake of readability, the detailed summation steps are provided in Appendix A.

Next, we state and prove the properties of the isoteleia typical set.

Theorem 34 (Properties of the Isoteleia Typical Set). For any ${ \underline { w } } ^ { n } \in { \underline { { A } } } _ { \epsilon } ^ { ( n ) }$ , for sufficiently large n: 1) For any $\tilde { w } ^ { n } \in E _ { \epsilon } ^ { ( n ) } ( \underline { { w } } ^ { n } )$

$$
2 ^ { - n ( H _ { s } ( \tilde { W } ) - H _ { p } ( \underline { { W } } ) + \epsilon ) } \leq \frac { P ( \tilde { w } ^ { n } ) } { P ( \underline { { w } } ^ { n } ) } \leq 2 ^ { - n ( H _ { s } ( \tilde { W } ) - H _ { p } ( \underline { { W } } ) - \epsilon ) } .\tag{195}
$$

2)

$$
2 ^ { n ( H _ { s } ( \tilde { W } ) - H _ { p } ( { \underline { { W } } } ) - \epsilon ) } \leq | E _ { \epsilon } ^ { ( n ) } ( { \underline { { w } } } ^ { n } ) | \leq 2 ^ { n ( H _ { s } ( \tilde { W } ) - H _ { p } ( { \underline { { W } } } ) + \epsilon ) } .\tag{196}
$$

The details of the proof are provided in Appendix B.

The synonymous typical set properties are known from [21, 22]; we restate them for reference.

Theorem 35 (Properties of the Synonymous Typical Set). For any $\tilde { w } ^ { n } \in \tilde { A } _ { \epsilon } ^ { ( n ) }$ , for sufficiently large n:

1) For any $w ^ { n } \in S _ { \epsilon } ^ { ( n ) } ( \tilde { w } ^ { n } )$

$$
2 ^ { - n ( H ( W ) - H _ { s } ( \tilde { W } ) + \epsilon ) } \leq \frac { P ( w ^ { n } ) } { P ( \tilde { w } ^ { n } ) } \leq 2 ^ { - n ( H ( W ) - H _ { s } ( \tilde { W } ) - \epsilon ) } .\tag{197}
$$

2)

$$
2 ^ { n ( H ( W ) - H _ { s } ( \tilde { W } ) - \epsilon ) } \leq | S _ { \epsilon } ^ { ( n ) } ( \tilde { w } ^ { n } ) | \leq 2 ^ { n ( H ( W ) - H _ { s } ( \tilde { W } ) + \epsilon ) } .\tag{198}
$$

Finally, we prove the properties of the Reified typical set.

Theorem 36 (Properties of the Reified Typical Set). For any ${ \underline { w } } ^ { n } \in { \underline { { A } } } _ { \epsilon } ^ { ( n ) }$ , for sufficiently large n: 1) For any $w ^ { n } \in B _ { \epsilon } ^ { ( n ) } ( \underline { { w } } ^ { n } )$

$$
2 ^ { - n ( H ( W ) - H _ { p } ( \underline { { W } } ) + \epsilon ) } \leq \frac { P ( w ^ { n } ) } { P ( \underline { { w } } ^ { n } ) } \leq 2 ^ { - n ( H ( W ) - H _ { p } ( \underline { { W } } ) - \epsilon ) } .\tag{199}
$$

2)

$$
2 ^ { n ( H ( W ) - H _ { p } ( { \underline { { W } } } ) - \epsilon ) } \leq | B _ { \epsilon } ^ { ( n ) } ( { \underline { { w } } } ^ { n } ) | \leq 2 ^ { n ( H ( W ) - H _ { p } ( { \underline { { W } } } ) + \epsilon ) } .\tag{200}
$$

The details of the proof are provided in Appendix C.

Corollary 6 (Hierarchy of Typical Set Sizes). The cardinalities of the typical sets satisfy

$$
| \underline { { A } } _ { \epsilon } ^ { ( n ) } | \le | \tilde { A } _ { \epsilon } ^ { ( n ) } | \le | A _ { \epsilon } ^ { ( n ) } | ,\tag{201}
$$

and the size of a Reified typical set is the product of the sizes of the corresponding Isoteleia and Synonymous typical sets:

$$
| B _ { \epsilon } ^ { ( n ) } ( \underline { { w } } ^ { n } ) | = | E _ { \epsilon } ^ { ( n ) } ( \underline { { w } } ^ { n } ) | \cdot | S _ { \epsilon } ^ { ( n ) } ( \tilde { w } ^ { n } ) | ,\tag{202}
$$

for any $\tilde { w } ^ { n } \in E _ { \epsilon } ^ { ( n ) } ( \underline { { w } } ^ { n } )$ , reflecting the composition $g ^ { n } = f ^ { n } \circ e ^ { n }$

Remark 10. The hierarchy $| \underline { { A } } _ { \epsilon } ^ { ( n ) } | \leq | \tilde { A } _ { \epsilon } ^ { ( n ) } |$ reflects the fact that the isoteleia mapping merges semantic distinctions that do not affect the optimal action, thereby reducing the number of distinct pragmatic typical sequences. Similarly, the synonymous mapping reduces the number of semantic typical sequences relative to syntactic ones. The reified typical set $B _ { \epsilon } ^ { ( n ) } ( \underline { { w } } ^ { n } )$ directly collects all syntactic sequences that realize a given pragmatic purpose, and its size is approximately $2 ^ { n ( H ( W ) - H _ { p } ( \underline { { W } } ) ) }$ , which is larger than a synonymous typical set by a factor of $2 ^ { n ( H _ { s } ( \tilde { W } ) - H _ { p } ( \underline { { W } } ) ) }$ This additional freedom is the source of compression gains in pragmatic lossless source coding.

## B. Pragmatic Lossless Source Coding Theorem

We establish the fundamental limit of pragmatic lossless source coding using a three-tier architecture that refines symbols through meanings to actions, yet only the pragmatic index is transmitted. The code maps each pragmatic typical sequence to a unique codeword, allowing arbitrary syntactic and semantic realizations. Using random coding and typical set properties, we prove that pragmatic entropy is the limit: rates above it are achievable with vanishing error, while rates below fail. We also relate this to the down pragmatic value of information, showing that the full decision utility can only be obtained when the rate reaches the pragmatic entropy.

1) Three-Tier Pragmatic Source Code Model: We consider a discrete memoryless syntactic source W with alphabet W and probability mass function $P ( W )$ . Let $\tilde { W }$ and $W$ be the associated semantic and pragmatic variables induced by the synonymous mapping $f$ and the isoteleia mapping e, respectively, with the reification mapping $g = f \circ e$ . The source generates an i.i.d. sequence $W ^ { n } = ( W _ { 1 } , \ldots , W _ { n } )$ according to $\begin{array} { r } { P ( W ^ { n } ) = \prod _ { k = 1 } ^ { n } P ( W _ { k } ) } \end{array}$ . The semantic sequence $\tilde { W } ^ { n }$ and pragmatic sequence $W ^ { n }$ are obtained by applying the sequential mappings $f ^ { n }$ and $g ^ { n }$ element-wise.

In the pragmatic lossless source coding scenario, the encoder may operate in three stages corresponding to the three layers of information, but in the order from pragmatic to semantic to syntactic: it first identifies the pragmatic purpose (optimal action class) from the source, then selects a semantic interpretation that justifies that action, and finally chooses a syntactic realization that conveys that meaning. However, since the reification mapping $g ^ { n } = f ^ { n } \circ e ^ { n }$ directly gives the pragmatic class of any syntactic sequence, the encoder can equivalently apply $g ^ { n }$ directly and then encode the resulting pragmatic index. The decoder, upon receiving the index, outputs an arbitrary syntactic sequence from the corresponding reified typical set $B _ { \epsilon } ^ { ( n ) } ( \underline { { w } } ^ { n } )$ , thereby recovering the pragmatic purpose without needing to reconstruct the exact syntactic or semantic details.

Definition 36 (Pragmatic Lossless Source Code). An $( M , n )$ pragmatic lossless source code consists of:

1) A pragmatic index set ${ \mathcal { T } } _ { p } = \{ 1 , 2 , \dots { } , M _ { p } \}$ , where each index $i _ { p } \in \mathcal { T } _ { p }$ corresponds to a distinct pragmatic purpose (optimal action class). Additionally, a reification index set $\mathcal { T } _ { r } = \{ 1 , 2 , \dots \cdot , M _ { r } \}$ , where each index $i _ { r } \in \mathcal { I } _ { r }$ enumerates the distinct syntactic realizations within each reified typical set. The total number of syntactic sequences that can be represented is $M = M _ { p } \cdot M _ { r }$

2) An encoding function that operates in three stages:

a) Pragmatic-to-semantic mapping (isoteleia): For a given pragmatic index $i _ { p } ,$ the encoder first selects a semantic interpretation from the corresponding isoteleia typical set $E _ { \epsilon } ^ { ( n ) } ( \underline { { w } } _ { i _ { p } } )$ where the index j enumerates the semantic sequences within that set.

b) Semantic-to-syntactic mapping (synonymous): For the chosen semantic sequence $\tilde { w } ^ { j }$ , the encoder then chooses a syntactic realization from the synonymous typical set $S _ { \epsilon } ^ { ( n ) } ( \tilde { w } ^ { j } )$ where the index k enumerates the syntactic sequences within that set.

c) Reification index selection: The encoder finally selects a reification index $i _ { r } \in \mathcal { Z } _ { r }$ , which corresponds to a specific syntactic sequence $w ^ { n }$ from the reified typical set $B _ { \epsilon } ^ { ( n ) } ( \underline { { w } } _ { i _ { p } } )$ . The reified typical set is the union of all synonymous typical sets corresponding to semantic sequences in the isoteleia typical set for that pragmatic index.

Equivalently, the encoding function $\phi : \mathcal { W } ^ { n } \to \mathcal { I } _ { p } \times \mathcal { I } _ { r }$ maps each syntactic sequence $w ^ { n }$ to a pair of indices $( i _ { p } , i _ { r } )$ such that $w ^ { n } \in B _ { \epsilon } ^ { ( n ) } ( \underline { { w } } _ { i _ { p } } )$ and $i _ { r }$ identifies the specific syntactic sequence within that reified typical set. Pragmatically, only the pragmatic index $i _ { p }$ needs to be transmitted to preserve the action; the reification index $i _ { r }$ is optional and determines syntactic fidelity.

3) A decoding function ψ : $: T _ { p } \to \mathcal { W } ^ { n }$ that, upon receiving an index $i _ { p } ,$ , outputs a representative syntactic sequence $\hat { w } ^ { n }$ belonging to the reified typical set $B _ { \epsilon } ^ { ( n ) } ( \underline { { w } } _ { i _ { p } } )$ . If the reification index $i _ { r }$ is also transmitted, the decoder can output the exact syntactic sequence identified by $i _ { r }$ ; otherwise, it may choose any syntactic sequence from the corresponding synonymous typical sets, as all share the same pragmatic meaning.

The pragmatic code rate is defined as $R = { \textstyle { \frac { 1 } { n } } } \log _ { 2 } M _ { p }$ (prabits per source symbol), and the reification rate is defined as $\begin{array} { r } { R _ { r } = \frac { 1 } { n } \log _ { 2 } M _ { r } } \end{array}$ (bits per source symbol). The total syntactic code rate is therefore $\begin{array} { r } { R _ { s y n } = R + R _ { r } = \frac { 1 } { n } \log _ { 2 } M } \end{array}$ . The error probability for a given code is

$$
P _ { e } ^ { ( n ) } = \operatorname* { P r } \{ \psi ( \phi ( W ^ { n } ) ) \notin B _ { \epsilon } ^ { ( n ) } ( \underline { { W } } ^ { n } ) \} ,\tag{203}
$$

$i . e .$ , the probability that the decoded syntactic sequence does not belong to the correct reified typical set, and hence the pragmatic purpose is not preserved.

This coding architecture is illustrated in Fig. 11. The encoder first applies the Reification Mapping $g ^ { n }$ to determine the pragmatic class of the observed syntactic sequence, then encodes that class index. The decoder recovers the index and outputs an arbitrary syntactic sequence from the corresponding reified typical set, which indicates that the decoder’s output may differ syntactically from the input but preserves the pragmatic meaning. The intermediate semantic layer is implicit in the composition $g ^ { n } = f ^ { n } \circ e ^ { n }$ , and the system can be viewed as a cascade of synonymous and isoteleia mappings.

$$
\xrightarrow { i _ { p } } \left[ \int ^ { n } \circ e ^ { n } \left[ \begin{array} { c } { W ^ { n } } \\ { \Phi } \\ { \phi } \end{array} \right] ^ { W ^ { n } } \left[ \begin{array} { c } { X ^ { n } \left( i _ { p } , j \right) } \\ { i _ { p } \in \left\{ 1 , \cdots , 2 ^ { n R } \right\} } \\ { j \in \left\{ 1 , \cdots , 2 ^ { n R } \right\} } \end{array} \right] ^ { - \left[ \begin{array} { c } { D \mathrm { e c o d e r } } \\ { W } \end{array} \right] ^ { \widehat { W } ^ { n } } \left[ \begin{array} { c } { \hat { e } ^ { - n } \circ f ^ { - n } } \\ { \hat { e } ^ { - n } \circ f ^ { - n } } \end{array} \right] ^ { \widehat { i } _ { p } } \right] }
$$

Fig. 11: Block diagram of pragmatic lossless source coding. The encoder applies the reification mapping $g ^ { n }$ to map the syntactic sequence to its pragmatic class and encodes the index. The decoder outputs a representative syntactic sequence from the corresponding reified typical set.

Theorem 37 (Pragmatic Lossless Source Coding Theorem). Let W be a discrete memoryless syntactic source with associated pragmatic variable W having pragmatic entropy $H _ { p } ( \underline { { W } } )$ . For any $\epsilon > 0$ and pragmatic code rate $R { : }$

1) Achievability: If $R > H _ { p } ( \underline { { W } } )$ , then for sufficiently large n, there exists an $\left( 2 ^ { n \left( R + R _ { p } \right) } , n \right)$ pragmatic lossless source code with error probability $P _ { e } ^ { ( n ) } < \epsilon ,$ , where $R _ { p }$ is the rate of the reified typical set $( i . e .$ , the number of syntactic realizations per pragmatic class).

2) Converse: If $R < H _ { p } ( \underline { { W } } )$ , then for any $( 2 ^ { n ( R + R _ { p } ) } , n )$ pragmatic lossless source code, the error probability $P _ { e } ^ { ( n ) }$ tends to 1 as $n  \infty$

Moreover, the pragmatic Value of Information (VoI) achieved by the compressed index is maximized $( i . e . ,$ , equals the VoI obtained from the full pragmatic variable W) if and only if $R \geq H _ { p } ( \underline { { W } } )$

Proof: Achievability: Fix $\epsilon > 0$ and choose $R > H _ { p } ( \underline { { W } } )$ . By the properties of the pragmatic typical set (Theorem 33), the pragmatic typical set $\underline { { A } } _ { \epsilon } ^ { ( n ) }$ satisfies

$$
( 1 - \epsilon ) 2 ^ { n ( H _ { p } ( \underline { { W } } ) - \epsilon ) } \leq | \underline { { A } } _ { \epsilon } ^ { ( n ) } | \leq 2 ^ { n ( H _ { p } ( \underline { { W } } ) + \epsilon ) } .
$$

Since $R > H _ { p } ( \underline { { W } } )$ , we can select n large enough such that $2 ^ { n R } \geq | \underline { { { A } } } _ { \epsilon } ^ { ( n ) } |$ . We construct a one-to-one mapping from each pragmatic typical sequence $\underline { w } ^ { n } \in \underline { A } _ { \epsilon } ^ { ( n ) }$ to a distinct pragmatic index $i _ { p } \in \{ 1 , \ldots , 2 ^ { n R } \}$ . For any syntactic sequence $w ^ { n }$ , the encoder computes its pragmatic class ${ \underline { { w } } } ^ { n } = g ^ { n } ( w ^ { n } )$ . If ${ \underline { w } } ^ { n } \in { \underline { { A } } } _ { \epsilon } ^ { ( n ) }$ , it sends the assigned index $i _ { p } ;$ otherwise, it sends a default index (e.g., 1).

Each pragmatic index $i _ { p }$ corresponds to a reified typical set $B _ { \epsilon } ^ { ( n ) } ( \underline { { w } } _ { i _ { p } } )$ of size approximately $2 ^ { n R _ { p } }$ , where $R _ { p } = H ( W ) - H _ { p } ( \underline { { W } } )$ is the maximum rate of the reified typical set. The decoder receives the index $i _ { p }$ and outputs an arbitrary syntactic sequence from the corresponding reified typical set $B _ { \epsilon } ^ { ( n ) } ( \underline { { w } } _ { i _ { p } } )$ (which is non-empty by definition). This ensures that the pragmatic purpose is correctly recovered whenever ${ \underline { { W } } } ^ { n } \in { \underline { { A } } } _ { \epsilon } ^ { ( n ) }$ . By Theorem 33, $\operatorname* { P r } \{ \underline { { W } } ^ { n } \in \underline { { A } } _ { \epsilon } ^ { ( n ) } \} > 1 - \epsilon$ , so the error probability is bounded by ϵ. Hence, for any $R > H _ { p } ( \underline { { W } } )$ , there exists a code with arbitrarily small error probability.

The total syntactic code rate is $R _ { s y n } = R + R _ { p }$ . When $R _ { p } = 0 ~ ( \mathrm { i . e . }$ ., each pragmatic class is represented by a single syntactic sequence), the total rate approaches $H _ { p } ( \underline { { W } } )$ . When $R _ { p }$ increases to its maximum value $H ( W ) - H _ { p } ( \underline { { W } } )$ , the total rate approaches the syntactic entropy $H ( W )$ , recovering the classical lossless source coding limit. The intermediate values of $R _ { p }$ correspond to different levels of syntactic redundancy preservation within each pragmatic equivalence class.

Converse: Assume a code with pragmatic rate $R < H _ { p } ( \underline { { W } } )$ and error probability $P _ { e } ^ { ( n ) }$ . Let $\mathcal { T } _ { p }$ be the set of pragmatic indices, with $| \mathcal { T } _ { p } | = 2 ^ { n R }$ . Define the decoding sets ${ \mathcal { D } } _ { i _ { p } } = \{ w ^ { n } : \phi ( w ^ { n } ) = i _ { p } \}$ and their corresponding pragmatic classes $\underline { { \mathcal { D } } } _ { i _ { p } } = \{ \underline { { w } } ^ { n } : g ^ { n } ( w ^ { n } ) = \underline { { w } } ^ { n }$ for some $w ^ { n } \in \mathcal { D } _ { i _ { p } } \}$ Since each decoding set is assigned to a single pragmatic index, the decoder can correctly recover the pragmatic purpose only if the true $\underline { { W } } ^ { n }$ belongs to $\underline { { \mathcal { D } } } _ { i _ { p } }$ . The total number of distinct pragmatic classes that can be correctly decoded is at most $| \mathcal { T } _ { p } | = 2 ^ { n R }$ . To achieve small error probability, these classes must cover most of the probability mass of $\underline { { W } } ^ { n }$ . By the properties of the pragmatic typical set, the number of pragmatic typical sequences is approximately $2 ^ { n H _ { p } ( \underline { { W } } ) }$ If $R < H _ { p } ( \underline { { W } } )$ , then $2 ^ { n R } < ( 1 - \epsilon ) 2 ^ { n ( H _ { p } ( { \underline { { W } } } ) - \epsilon ) }$ for large $n ,$ so at most a fraction of the pragmatic typical sequences can be assigned distinct indices. Consequently, for any code, the probability that the true pragmatic sequence falls into an unassigned class or is mapped to a wrong index is at least $1 - 2 ^ { - n ( H _ { p } ( \underline { { W } } ) - R - \epsilon ) }$ , which tends to 1 as $n \to \infty$ . Thus, $P _ { e } ^ { ( n ) } \to 1$ . This proves the converse.

Connection to Pragmatic VoI: The pragmatic Value of Information achieved by the compressed

index $I _ { p }$ is defined as

$$
\mathrm { V o I } _ { p } ( I _ { p } ) = \mathbb { E } _ { W , I _ { p } } \big [ U ( W , a ^ { * } ( \underline { { W } } ( I _ { p } ) ) ) \big ] - U _ { 0 } ,
$$

where $a ^ { * } ( \underline { { W } } ( I _ { p } ) )$ is the optimal action based on the recovered pragmatic class, and $U _ { 0 }$ is the baseline utility without any information. Since $I _ { p }$ is a function of the encoded index, it carries at most $n R$ bits of information about the source. By the data processing inequality, the mutual information between the pragmatic variable W and $I _ { p }$ is bounded by

$$
I ( \underline { W } ; I _ { p } ) \le H ( I _ { p } ) \le n R .
$$

To achieve the maximum pragmatic value $\mathrm { V o I } _ { p } ^ { \mathrm { m a x } } { \ - } \mathrm { i } . \mathrm { e } .$ ., the same utility as if the true pragmatic variable W were directly observed—the index $I _ { p }$ must preserve all pragmatically relevant distinctions. This requires that W be recoverable from $I _ { p } ,$ , which in turn necessitates $I ( \underline { { W } } ; I _ { p } ) =$ $H _ { p } ( \underline { { W } } )$ . However, this equality can hold only if $n R \geq H _ { p } ( \underline { { W } } )$ , i.e., $R \geq H _ { p } ( \underline { { W } } )$ . If $R < H _ { p } ( \underline { { W } } )$ then $I ( \underline { { W } } ; I _ { p } ) \le n R < H _ { p } ( \underline { { W } } )$ , implying that some uncertainty about the optimal action class remains unresolved. Consequently, the expected utility conditioned on $I _ { p }$ is strictly less than the utility obtained under perfect pragmatic knowledge, and the achievable VoI is strictly below $\operatorname { V o I } _ { p } ^ { \operatorname* { m a x } }$ . Conversely, when $R \geq H _ { p } ( \underline { { W } } )$ , the encoder can simply transmit a lossless description of the pragmatic index, and the decoder recovers W perfectly, thereby attaining the full pragmatic VoI. Thus, the minimum rate for full pragmatic value extraction is exactly $H _ { p } ( \underline { { W } } )$ □

Remark 11. The pragmatic lossless source coding theorem generalizes Shannon’s classical theorem (when isoteleia and synonymous mappings are trivial, yielding $H _ { p } ( \underline { { W } } ) = H ( W ) )$ and the semantic theorem (when only the synonymous mapping is nontrivial, $H _ { p } ( \underline { { W } } ) = H _ { s } ( \tilde { W } ) )$ The hierarchy $H _ { p } ( \underline { { W } } ) \le H _ { s } ( \tilde { W } ) \le H ( W )$ quantifies the compression gains from pragmatic abstraction.

The reified typical set $B _ { \epsilon } ^ { ( n ) } ( \underline { { w } } ^ { n } )$ admits a two-level decomposition via the composition $g ^ { n } =$ $f ^ { n } \circ e ^ { n } .$ : it partitions into isoteleia typical sets $E _ { \epsilon } ^ { ( n ) } ( \underline { { w } } ^ { n } )$ (grouping semantic justifications that lead to the same action) and, within each, synonymous typical sets $S _ { \epsilon } ^ { ( n ) } ( \tilde { w } ^ { n } )$ (grouping syntactic realizations of the same meaning). This structure underpins the additional rate $R _ { p } ,$ which offers a flexible trade-off: setting $R _ { p } = 0$ achieves the pure pragmatic limit, while increasing $R _ { p }$ preserves syntactic details within each pragmatic class, recovering the classical limit as $R _ { p } \to H ( W ) - H _ { p } ( \underline { { W } } )$

The optimal encoding strategy compresses directly to the pragmatic index, discarding all syntactic and semantic distinctions that do not affect the optimal action — consistent with the design principles of pragmatic information systems (Section II). The pragmatic VoI provides a direct performance metric: for a given rate R, the achievable utility is maximized when the code preserves pragmatic equivalence classes, establishing a principled trade-off between communication rate and task performance in applications such as autonomous driving, remote control, and industrial automation.

## VIII. PRAGMATIC CHANNEL CODING

In this section, we study pragmatic information transmission over noisy channels. Extending the joint AEP via the joint reification mapping, we define the jointly reified typical set that groups syntactic input-output pairs by their common pragmatic purpose. Using random coding and joint typical decoding, we prove the pragmatic channel coding theorem: the pragmatic capacity equals the maximum achievable rate and extends the classical capacity by tolerating errors that do not affect optimal actions.

## A. Jointly Asymptotic Equipartition Property and Jointly Reified Typical Set

To establish the fundamental limits of pragmatic channel coding, we first extend the asymptotic equipartition property to the joint distribution of channel input and output sequences at the pragmatic level. Unlike the semantic approach, which relies on the joint synonymous mapping, we directly employ the joint reification mapping, which is the composition of the joint synonymous and joint isoteleia mappings, to aggregate syntactic sequence pairs into pragmatic equivalence classes. This approach provides a direct characterization of the equivalence of input-output pairs with respect to the optimal terminal actions.

Consider a discrete memoryless channel with input alphabet $x ,$ output alphabet Y, and transition probability $P ( \boldsymbol { Y } | \boldsymbol { X } )$ . Let $\tilde { \mathcal X }$ and $\tilde { \mathcal { V } }$ be the associated semantic alphabets, and let $\underline { { \mathcal { X } } }$ and Y be the pragmatic alphabets induced by the isoteleia mappings $e _ { X } : \underline { { { \mathcal { X } } } } \to 2 ^ { \tilde { \mathcal { X } } }$ and $e _ { Y } : \underline { { \mathcal { V } } } \to 2 ^ { \tilde { \mathcal { V } } }$ , respectively. The synonymous mappings $f _ { X } : \tilde { \mathcal { X } } \to 2 ^ { \mathcal { X } }$ and $f _ { Y } : \tilde { \mathcal { V } }  2 ^ { \mathcal { V } }$ map semantic symbols to sets of syntactic symbols. The reification mappings $g _ { X } = f _ { X } \circ e _ { X } : \underline { { \mathcal { X } } } \to 2 ^ { \mathcal { X } }$ and $g _ { Y } = f _ { Y } \circ e _ { Y } : \underline { { \mathcal { Y } } }  2 ^ { \mathcal { Y } }$ directly bridge the pragmatic and syntactic layers.

For the joint behavior of channel input and output, we define the joint reification mapping, the joint synonymous mapping, and the joint isoteleia mapping. Their sequential extensions are defined element-wise.

Definition 37 (Joint Mappings for Sequences). Let $f _ { X Y } : \tilde { \mathcal { X } } \times \tilde { \mathcal { Y } }  2 ^ { { \mathcal { X } } \times \mathcal { Y } }$ be the joint synonymous mapping, and let $e _ { X Y } : \underline { { { \mathcal { X } } } } \times \underline { { { \mathcal { Y } } } } \to 2 ^ { \tilde { \mathcal { X } } \times \tilde { \mathcal { Y } } }$ be the joint isoteleia mapping. The joint reification mapping $g _ { X Y } = f _ { X Y } \circ e _ { X Y } : \underline { { { \mathcal { X } } } } \times \underline { { { \mathcal { Y } } } } \to 2 ^ { \mathcal { X } \times \mathcal { Y } }$ is their composition. Their n-th sequential extensions are defined by:

$$
\left\{ \begin{array} { l l } { \displaystyle f _ { X Y } ^ { n } ( \tilde { x } ^ { n } , \tilde { y } ^ { n } ) \triangleq \prod _ { k = 1 } ^ { n } f _ { X Y } ( \tilde { x } _ { k } , \tilde { y } _ { k } ) , } \\ { \displaystyle e _ { X Y } ^ { n } ( \underline { { x } } ^ { n } , \underline { { y } } ^ { n } ) \triangleq \prod _ { k = 1 } ^ { n } e _ { X Y } ( \underline { { x } } _ { k } , \underline { { y } } _ { k } ) , } \\ { \displaystyle g _ { X Y } ^ { n } ( \underline { { x } } ^ { n } , \underline { { y } } ^ { n } ) \triangleq ( f _ { X Y } ^ { n } \circ e _ { X Y } ^ { n } ) ( \underline { { x } } ^ { n } , \underline { { y } } ^ { n } ) = \prod _ { k = 1 } ^ { n } g _ { X Y } ( \underline { { x } } _ { k } , \underline { { y } } _ { k } ) . } \end{array} \right.\tag{204}
$$

Thus, $g _ { X Y } ^ { n }$ partitions the joint syntactic space $\mathcal { X } ^ { n } \times \mathcal { Y } ^ { n }$ into equivalence classes according to the pragmatic pair $( \underline { x } ^ { n } , y ^ { n } )$

We now define the relevant typical sets. Let $P ( X , Y ) = P ( X ) P ( Y | X )$ denote the joint distribution induced by the channel. The syntactically jointly typical set $A _ { \epsilon } ^ { ( n ) }$ is defined as in classical information theory [1, 45]:

$$
\begin{array} { l } { { \displaystyle { \cal A } _ { \epsilon } ^ { ( n ) } \triangleq \Big \{ ( x ^ { n } , y ^ { n } ) \in \mathcal { X } ^ { n } \times \mathcal { Y } ^ { n } : \left. - \frac { 1 } { n } \log P ( x ^ { n } ) - H ( X ) \right. < \epsilon , } } \\ { { \displaystyle ~ \left. - \frac { 1 } { n } \log P ( y ^ { n } ) - H ( Y ) \right. < \epsilon , } } \\ { { \displaystyle ~ \left. - \frac { 1 } { n } \log P ( x ^ { n } , y ^ { n } ) - H ( X , Y ) \right. < \epsilon \Big \} } . } \end{array}\tag{205}
$$

For the pragmatic layer, we define the pragmatically jointly typical set, which aggregates probabilities over reified equivalence classes.

Definition 38 (Pragmatically Jointly Typical Set). The pragmatically jointly typical set $\underline { { A } } _ { \epsilon } ^ { ( n ) }$ is

the set of pragmatic sequence pairs $( \underline { { x } } ^ { n } , y ^ { n } ) \in \underline { { \mathcal { X } } } ^ { n } \times \underline { { \mathcal { Y } } } ^ { n }$ such that

$$
\begin{array} { r l } & { \underline { { A } } _ { \epsilon } ^ { ( n ) } \triangleq \Bigg \{ ( \underline { { x } } ^ { n } , \underline { { y } } ^ { n } ) : \Bigg | - \frac { 1 } { n } \log P ( \underline { { x } } ^ { n } ) - H _ { p } ( \underline { { X } } ) \Bigg | < \epsilon , } \\ & { \qquad \Bigg | { - \frac { 1 } { n } \log P ( \underline { { y } } ^ { n } ) - H _ { p } ( \underline { { Y } } ) } \Bigg | < \epsilon , } \\ & { \qquad \Bigg | { - \frac { 1 } { n } \log P ( \underline { { x } } ^ { n } , \underline { { y } } ^ { n } ) - H _ { p } ( \underline { { X } } , \underline { { Y } } ) } \Bigg | < \epsilon \Bigg \} , } \end{array}\tag{206}
$$

where

$$
P ( \underline { { x } } ^ { n } , \underline { { y } } ^ { n } ) = \prod _ { k = 1 } ^ { n } P ( \underline { { x } } _ { k } , \underline { { y } } _ { k } ) , \quad P ( \underline { { x } } _ { k } , \underline { { y } } _ { k } ) = \sum _ { ( x _ { k } , y _ { k } ) \in g _ { X Y } ( \underline { { x } } _ { k } , \underline { { y } } _ { k } ) } P ( x _ { k } , y _ { k } ) .\tag{207}
$$

The marginal pragmatic probabilities $P ( \underline { x } ^ { n } )$ and $P ( y ^ { n } )$ are obtained by summing over the other variable.

For a given pragmatic pair, the jointly reified typical set collects all syntactic pairs that share that pragmatic interpretation and are syntactically jointly typical. This set is naturally decomposed into synonymous typical sets corresponding to each semantic interpretation in the isoteleia typical set.

Definition 39 (Jointly Reified Typical Set). For a given pragmatic typical pair $( \underline { x } ^ { n } , \underline { y } ^ { n } ) \in \underline { A } _ { \epsilon } ^ { ( n ) }$ the jointly reified typical set $B _ { \epsilon } ^ { ( n ) } ( \underline { { x } } ^ { n } , y ^ { n } )$ with the syntactically jointly typical sequences $( x ^ { n } , y ^ { n } )$ is defined as the set of n-sequence pairs $( x ^ { n } , y ^ { n } ) \in \mathcal { X } ^ { n } \times \mathcal { Y } ^ { n }$ such that the following conditions

hold:

$$
\begin{array} { r l } { E _ { \mathrm { c } } ^ { ( \lambda ) } ( \mathcal { Q } , \mathcal { Z } ^ { * } ) = \Bigg \{ \ln ^ { + } x ^ { \lambda } + \mathcal { Q } ^ { ( \lambda ) } \times \mathcal { X } ^ { \lambda ^ { 2 } + } \Bigg | - \frac { 1 } { \alpha } \ln \mathcal { R } \Bigg \} ( x , x ^ { \lambda } ) - \mathcal { R } ( \lambda ) } & { \leq x _ { \lambda } } \\ & { \Bigg \lvert - \ln \mathrm { l i d } \mathcal { P } \Bigg ( \theta _ { \lambda } ^ { ( \lambda ) } - \mathcal { R } ( \mathcal { X } ) \Bigg \rvert \leq x _ { \lambda } } \\ & { \Bigg \lvert - \frac { 1 } { \alpha } \ln \mathrm { l i d } \mathcal { P } \Bigg ( \theta _ { \lambda } ^ { ( \lambda ) } - \theta _ { \lambda } ^ { ( \lambda ) } - \mathcal { R } ( \mathcal { X } ) \Bigg \rvert \leq x _ { \lambda } } \\ & { \Bigg \lvert - \frac { 1 } { \alpha } \ln \mathrm { l i d } \mathcal { P } \Bigg ( x _ { \lambda } ^ { ( \lambda ) } - \mathcal { R } _ { \lambda } ( \mathcal { X } ) \Bigg \rvert \leq x _ { \lambda } } \\ & { \Bigg \lvert - \frac { 1 } { \alpha } \ln \mathrm { l i d } \mathcal { P } \Bigg ( \mathcal { X } \Bigg ) - \mathcal { R } _ { \lambda } ( \mathcal { X } ) \Bigg \rvert \leq x _ { \lambda } } \\ & { \Bigg \lvert - \frac { 1 } { \alpha } \ln \mathrm { l i d } \mathcal { P } \Bigg ( \theta _ { \lambda } ^ { ( \lambda ) } - \mathcal { R } _ { \lambda } ( \mathcal { X } ) \Bigg \rvert \leq x _ { \lambda } } \\ & { \Bigg \lvert - \ln \mathrm { l i d } \mathcal { R } \Bigg ( \mathcal { P } \Bigg ) - \mathcal { R } _ { \lambda } ( \mathcal { X } ) \Bigg \rvert \leq x _ { \lambda } } \\ & { \Bigg \lvert \mathrm { l i d } \mathcal { R } \Bigg ( \mathcal { P } \Bigg ( \theta _ { \lambda } ^ { ( \lambda ) } , \Phi _ { \lambda } ^ { ( \lambda ) } - \mathcal { R } _ { \lambda } ( \mathcal { X } ) \Bigg \rvert \leq x _ { \lambda } } \\ \end{array}\tag{208}
$$

where

$$
P ( ( \underline { { x } } ^ { n } , \underline { { y } } ^ { n } ) \to ( x ^ { n } , y ^ { n } ) ) \triangleq \left\{ \begin{array} { l l } { \displaystyle \frac { P ( x ^ { n } , y ^ { n } ) } { P ( \underline { { x } } ^ { n } , \underline { { y } } ^ { n } ) } , } & { i f ( x ^ { n } , y ^ { n } ) \in g _ { X Y } ^ { n } ( \underline { { x } } ^ { n } , \underline { { y } } ^ { n } ) , } \\ { 0 , } & { o t h e r w i s e . } \end{array} \right.\tag{209}
$$

The first three conditions ensure syntactic joint typicality of the sequence pair $( x ^ { n } , y ^ { n } )$ . The fourth through sixth conditions ensure pragmatic joint typicality of the sequence pair $( \underline { { x } } ^ { n } , y ^ { n } )$ . The seventh condition enforces the reification typicality, i.e., the difference between the syntactic joint entropy and the pragmatic joint entropy is ϵ-close to the entropy reduction $H ( X , Y ) - H _ { p } ( \underline { { X } } , \underline { { Y } } )$

Equivalently, the jointly reified typical set can be expressed as the union of jointly synonymous typical sets over all semantic pairs in the corresponding isoteleia typical set:

$$
B _ { \epsilon } ^ { ( n ) } ( \underline { { x } } ^ { n } , \underline { { y } } ^ { n } ) = \bigcup _ { ( \tilde { x } ^ { n } , \tilde { y } ^ { n } ) \in E _ { \epsilon } ^ { ( n ) } ( \underline { { x } } ^ { n } , \underline { { y } } ^ { n } ) } S _ { \epsilon } ^ { ( n ) } ( \tilde { x } ^ { n } , \tilde { y } ^ { n } ) ,\tag{210}
$$

where $E _ { \epsilon } ^ { ( n ) } ( \underline { { x } } ^ { n } , y ^ { n } )$ denotes the isoteleia typical set, and $S _ { \epsilon } ^ { ( n ) } ( \tilde { x } ^ { n } , \tilde { y } ^ { n } )$ denotes the jointly synonymous typical set. Thus, the jointly reified typical set partitions the syntactically jointly typical set according to the equivalence classes induced by the composition of the joint isoteleia and joint synonymous mappings, i.e., the joint reification mapping.

Under the sequential joint reification mapping $g _ { X Y } ^ { n }$ , the syntactically jointly typical set $A _ { \epsilon } ^ { ( n ) }$ can be partitioned into jointly reified typical sets:

$$
A _ { \epsilon } ^ { ( n ) } = \bigcup _ { ( \underline { { x } } ^ { n } , \underline { { y } } ^ { n } ) \in \underline { { A } } _ { \epsilon } ^ { ( n ) } } B _ { \epsilon } ^ { ( n ) } ( \underline { { x } } ^ { n } , \underline { { y } } ^ { n } ) ,\tag{211}
$$

with disjointness for distinct pragmatic typical pairs.

Fig. 12 illustrates the hierarchical relationship among the syntactically jointly typical set $A _ { \epsilon } ^ { ( n ) }$ the pragmatically jointly typical set $\underline { { A } } _ { \epsilon } ^ { ( n ) }$ , and the jointly reified typical sets $B _ { \epsilon } ^ { ( n ) } ( \underline { { x } } ^ { n } , y ^ { n } )$ . The joint reification mapping $g _ { X Y } ^ { n } = f _ { X Y } ^ { n } \circ e _ { X Y } ^ { n }$ maps each syntactic sequence pair $( x ^ { n } , y ^ { n } )$ to its corresponding pragmatic pair $( \underline { { x } } ^ { n } , \underline { { y } } ^ { n } )$ . Under the joint reification mapping, $A _ { \epsilon } ^ { ( n ) }$ is partitioned into disjoint jointly reified typical sets, each corresponding to a distinct pragmatic typical pair $( \underline { { x } } ^ { n } , \underline { { y } } ^ { n } )$ in $\underline { { A } } _ { \epsilon } ^ { ( n ) }$ . Each jointly reified typical set $B _ { \epsilon } ^ { ( n ) } ( \underline { { x } } ^ { n } , \underline { { y } } ^ { n } )$ is itself the union of jointly synonymous typical sets $S _ { \epsilon } ^ { ( n ) } ( \tilde { x } ^ { n } , \tilde { y } ^ { n } )$ over all semantic pairs in the isoteleia typical set $E _ { \epsilon } ^ { ( n ) } ( \underline { { x } } ^ { n } , y ^ { n } )$ , reflecting the composition $g _ { X Y } ^ { n } = f _ { X Y } ^ { n } \circ e _ { X Y } ^ { n }$ . The dashed circles within each reified set represent the individual synonymous typical sets, which group syntactic pairs sharing the same semantic interpretation. This nested structure visually demonstrates that the pragmatic abstraction merges semantically distinct but pragmatically equivalent pairs, thereby reducing the number of equivalence classes from $| A _ { \epsilon } ^ { ( n ) } | \ \mathrm { t o } \ | \underline { { { A } } } _ { \epsilon } ^ { ( n ) } |$ , consistent with the entropy reduction $H _ { p } ( \underline { { X } } , \underline { { Y } } ) \le H _ { s } ( \tilde { X } , \tilde { Y } ) \le H ( X , Y )$ established in Section III.

We now establish the AEP properties of these typical sets. The syntactically joint AEP is the classical result [1, 45]; we restate it for reference.

Theorem 38 (Syntactically Joint AEP). Let $( X ^ { n } , Y ^ { n } )$ be a sequence pair of length n drawn i.i.d. according to $P ( x ^ { n } , y ^ { n } )$ . Then for sufficiently large n:

1) $\operatorname* { P r } \{ ( X ^ { n } , Y ^ { n } ) \in A _ { \epsilon } ^ { ( n ) } \} > 1 - \epsilon .$

2) $( 1 - \epsilon ) 2 ^ { n ( H ( X , Y ) - \epsilon ) } \leq | A _ { \epsilon } ^ { ( n ) } | \leq 2 ^ { n ( H ( X , Y ) + \epsilon ) } .$

3) ${ \cal I } f \left( \dot { X } ^ { n } , \dot { Y } ^ { n } \right) \sim { \cal P } ( x ^ { n } ) { \cal P } ( y ^ { n } )$ are independent sequences with the same marginals, then

$$
( 1 - \epsilon ) 2 ^ { - n ( I ( X ; Y ) + 3 \epsilon ) } \leq \operatorname* { P r } \{ ( \dot { X } ^ { n } , \dot { Y } ^ { n } ) \in A _ { \epsilon } ^ { ( n ) } \} \leq 2 ^ { - n ( I ( X ; Y ) - 3 \epsilon ) } .
$$

The pragmatically joint AEP follows by applying the weak law of large numbers to the aggregated pragmatic probabilities.

![](images/11f41735e98fc4ea29f27ac4e4f465ad2c708f5a84b2d69273ad6bb9f88df053.jpg)  
Fig. 12: The joint reification mapping $g _ { X Y } ^ { n }$ maps syntactic pairs $( x ^ { n } , y ^ { n } )$ to pragmatic classes $( \underline { { x } } ^ { n } , \underline { { y } } ^ { n } )$ . Each pragmatic typical pair corresponds to a jointly reified typical set $B _ { \epsilon } ^ { ( n ) } ( \underline { { x } } ^ { n } , \underline { { y } } ^ { n } )$ which is a union of synonymous typical sets over the isoteleia typical set.

Theorem 39 (Pragmatically Joint AEP for Transmission). Let $( \underline { { X } } ^ { n } , \underline { { Y } } ^ { n } )$ be a pragmatic sequence pair of length n drawn i.i.d. according to $P ( \underline { x } ^ { n } , y ^ { n } )$ , with the associated syntactic sequence pair $( X ^ { n } , Y ^ { n } )$ under the joint reification mapping $g _ { X Y } ^ { n }$ . Then for sufficiently large n:

1) $\operatorname* { P r } \{ ( \underline { { X } } ^ { n } , \underline { { Y } } ^ { n } ) \in \underline { { A } } _ { \epsilon } ^ { ( n ) } \} > 1 - \epsilon .$

2) For any $( \underline { { x } } ^ { n } , y ^ { n } ) \in \underline { { A } } _ { \epsilon } ^ { ( n ) }$

$$
2 ^ { - n ( H _ { p } ( \underline { { X } } , \underline { { Y } } ) + \epsilon ) } \leq P ( \underline { { x } } ^ { n } , y ^ { n } ) \leq 2 ^ { - n ( H _ { p } ( \underline { { X } } , \underline { { Y } } ) - \epsilon ) } .
$$

$$
3 ) \ ( 1 - \epsilon ) 2 ^ { n ( H _ { p } ( \underline { { X } } , \underline { { Y } } ) - \epsilon ) } \leq | \underline { { A } } _ { \epsilon } ^ { ( n ) } | \leq 2 ^ { n ( H _ { p } ( \underline { { X } } , \underline { { Y } } ) + \epsilon ) } .
$$

4) ${ \cal I } f \left( \dot { X } ^ { n } , \dot { Y } ^ { n } \right) \sim { \cal P } ( x ^ { n } ) { \cal P } ( y ^ { n } )$ are independent sequences with the same marginals as $X ^ { n }$ and $Y ^ { n }$ , and $( \dot {  { \boldsymbol { X } } } ^ { n } , \dot {  { \boldsymbol { Y } } } ^ { n } )$ are the corresponding pragmatic sequences, then

$$
( 1 - \epsilon ) 2 ^ { - n ( I ^ { p } ( \underline { { X } } ; \underline { { Y } } ) + 3 \epsilon ) } \leq \operatorname* { P r } \{ ( \underline { { \dot { X } } } ^ { n } , \underline { { \dot { Y } } } ^ { n } ) \in \underline { { A } } _ { \epsilon } ^ { ( n ) } \} \leq 2 ^ { - n ( I ^ { p } ( \underline { { X } } ; \underline { { Y } } ) - 3 \epsilon ) } .\tag{212}
$$

The details of the proof are provided in Appendix D.

Finally, we establish the properties of the jointly reified typical set and its decomposition.

Theorem 40 (Properties of the Jointly Reified Typical Set). For any pragmatic typical pair $( \underline { x } ^ { n } , \underline { y } ^ { n } ) \in \underline { A } _ { \epsilon } ^ { ( n ) }$ , for sufficiently large n:

1) For any $( x ^ { n } , y ^ { n } ) \in B _ { \epsilon } ^ { ( n ) } ( \underline { { x } } ^ { n } , y ^ { n } )$

$$
2 ^ { - n ( H ( X , Y ) - H _ { p } ( \underline { { X } } , \underline { { Y } } ) + \epsilon ) } \leq \frac { P ( x ^ { n } , y ^ { n } ) } { P ( \underline { { x } } ^ { n } , \underline { { y } } ^ { n } ) } \leq 2 ^ { - n ( H ( X , Y ) - H _ { p } ( \underline { { X } } , \underline { { Y } } ) - \epsilon ) } .
$$

2)

$$
2 ^ { n ( H ( X , Y ) - H _ { p } ( \underline { { X } } , \underline { { Y } } ) - \epsilon ) } \leq | B _ { \epsilon } ^ { ( n ) } ( \underline { { x } } ^ { n } , \underline { { y } } ^ { n } ) | \leq 2 ^ { n ( H ( X , Y ) - H _ { p } ( \underline { { X } } , \underline { { Y } } ) + \epsilon ) } .
$$

The details of the proof are provided in Appendix E.

Corollary 7 (Decomposition of the Jointly Reified Typical Set). The jointly reified typical set can be decomposed as the union of synonymous typical sets over the isoteleia typical set:

$$
B _ { \epsilon } ^ { ( n ) } ( \underline { { x } } ^ { n } , \underline { { y } } ^ { n } ) = \bigcup _ { ( \tilde { x } ^ { n } , \tilde { y } ^ { n } ) \in E _ { \epsilon } ^ { ( n ) } ( \underline { { x } } ^ { n } , \underline { { y } } ^ { n } ) } S _ { \epsilon } ^ { ( n ) } ( \tilde { x } ^ { n } , \tilde { y } ^ { n } ) .
$$

Consequently, its cardinality satisfies

$$
| B _ { \epsilon } ^ { ( n ) } ( \underline { { { x } } } ^ { n } , \underline { { { y } } } ^ { n } ) | \approx | E _ { \epsilon } ^ { ( n ) } ( \underline { { { x } } } ^ { n } , \underline { { { y } } } ^ { n } ) | \cdot | S _ { \epsilon } ^ { ( n ) } ( \tilde { x } ^ { n } , \tilde { y } ^ { n } ) | ,\tag{213}
$$

where the approximation holds for any $\tilde { x } ^ { n } , \tilde { y } ^ { n }$ in the isoteleia typical set, and the product reflects the composition $g _ { X Y } ^ { n } = f _ { X Y } ^ { n } \circ e _ { X Y } ^ { n }$

Remark 12. The jointly reified typical set $B _ { \epsilon } ^ { ( n ) } ( \underline { { x } } ^ { n } , y ^ { n } )$ groups together all syntactically jointly typical pairs that share the same pragmatic interpretation. Its decomposition into synonymous and isoteleia typical sets reveals the two-level abstraction: the isoteleia mapping merges semantic differences that do not affect the optimal actions, while the synonymous mapping merges syntactic differences that do not affect the meaning. The cardinality of a jointly reified typical set is approximately 2<sup>n(H(X,Y</sup> <sup>)−Hp(X,Y</sup> <sup>))</sup>, which is larger than a single synonymous typical set by a factor of $2 ^ { n ( H _ { s } ( { \tilde { X } } , { \tilde { Y } } ) - H _ { p } ( \underline { { { X } } } , \underline { { { Y } } } ) ) }$ , reflecting the additional syntactic freedom gained when only the pragmatic purpose needs to be preserved. When randomly selecting an independent pair of syntactic sequences, the probability that they fall into the same jointly reified typical set (i.e., are pragmatically jointly typical) is about $2 ^ { - n I ^ { p } ( \underline { { X } } ; \underline { { Y } } ) }$ , which is larger than the classical joint typicality probability $2 ^ { - n I ( X ; Y ) }$ , reflecting the pragmatic capacity gain.

## B. Pragmatic Channel Coding Theorem

We establish the fundamental limit for pragmatic transmission over noisy channels using a two-level code: the encoder selects a pragmatic index (optimal action) and then a syntactic codeword from its reified typical set; the decoder recovers only the index, tolerating syntactic errors. Via random coding and joint typical decoding based on the jointly reified typical set, we prove that the pragmatic channel capacity is the maximum achievable rate. We also connect this result to the pragmatic cost of information.

Definition 40 (Pragmatic Channel Code). An $( M _ { p } , M _ { r } , n )$ pragmatic channel code consists of:

1) A pragmatic index set ${ \mathcal { T } } _ { p } = \{ 1 , 2 , \dots { } \} M _ { p } \}$ , where each index $i _ { p } \in \mathcal { I } _ { p }$ corresponds to a distinct pragmatic purpose (optimal action class).

2) A reification index set ${ \mathcal { T } } _ { r } = \{ 1 , 2 , \dots { } , M _ { r } \}$ , where each index $i _ { r } \in \mathcal { T } _ { r }$ enumerates the syntactic codewords within each reified typical set.

3) An encoding function $\phi : \mathbb { Z } _ { p } \times \mathbb { Z } _ { r } \to \mathcal { X } ^ { n }$ that maps each pair $( i _ { p } , i _ { r } )$ to a codeword $x ^ { n } ( i _ { p } , i _ { r } )$ such that for each pragmatic index $i _ { p } ,$ the set of codewords $\{ x ^ { n } ( i _ { p } , i _ { r } ) : i _ { r } \in \mathcal { T } _ { r } \}$ lies entirely within the corresponding reified typical set $B _ { \epsilon } ^ { ( n ) } ( \underline { { x } } _ { i _ { p } } ^ { n } , \underline { { y } } _ { i _ { p } } ^ { n } )$ for some typical output pragmatic sequence $\underline { { y } } _ { i _ { p } } ^ { n }$ . (The decoder will recover $i _ { p }$ without needing the exact $i _ { r \cdot } )$

4) A decoding function $\psi : \mathcal { V } ^ { n } \to \mathcal { T } _ { p }$ that maps the received sequence to a pragmatic index.

The pragmatic code rate is $\begin{array} { r } { R _ { p } = \frac { 1 } { n } \log _ { 2 } M _ { p } } \end{array}$ (prabits per channel use), and the reification rate is $\begin{array} { r } { R _ { r } = \frac { 1 } { n } \log _ { 2 } M _ { r } } \end{array}$ (bits per channel use). The total syntactic rate is

$$
R _ { t o t } = R _ { p } + R _ { r } = \frac { 1 } { n } \log _ { 2 } ( M _ { p } M _ { r } ) .\tag{214}
$$

For a given code, the error probability is defined as

$$
{ \cal P } _ { e } ^ { ( n ) } = \operatorname* { P r } \{ \psi ( Y ^ { n } ) \neq i _ { p } \} ,\tag{215}
$$

where $i _ { p }$ is the intended pragmatic index, and $Y ^ { n }$ is the channel output when the corresponding codeword is transmitted.

![](images/1707ae5e5df6669067d090f9f15991c2795f98dd10f5dc62d84ca90e6a27a7a2.jpg)

Fig. 13: Block diagram of pragmatic channel coding. The encoder selects a pragmatic index $i _ { p }$ and a reification index $i _ { r } ,$ then transmits the corresponding codeword. The decoder recovers only the pragmatic index $\hat { i } _ { p } ,$ ignoring syntactic variations within the reified typical set.

Figure 13 depicts the process of pragmatic channel coding. The encoder first selects a pragmatic index $i _ { p } \in \mathcal { I } _ { p }$ corresponding to the intended optimal action class. Given $i _ { p } ,$ the encoder then chooses a reification index $i _ { r } \in \mathcal { I } _ { r }$ , which selects a specific syntactic codeword from the reified typical set $B _ { \epsilon } ^ { ( n ) } ( \underline { { x } } _ { i _ { p } } ^ { n } , \underline { { y } } _ { i _ { p } } ^ { n } )$ . The transmitted codeword $x ^ { n } ( i _ { p } , i _ { r } )$ is sent over the channel. The decoder recovers only the pragmatic index $\hat { i } _ { p }$ via jointly typical decoding based on the jointly reified typical set, ignoring syntactic variations within the reified typical set. Thus, the two-level structure separates the pragmatic purpose from its syntactic realizations, enabling reliable transmission at rates up to the pragmatic channel capacity.

Theorem 41 (Pragmatic Channel Coding Theorem). Let $C _ { p }$ be the pragmatic channel capacity defined in $E q .$ . (120) of Section V-A, i.e.,

$$
C _ { p } = \operatorname* { m a x } _ { p ( x ) } \operatorname* { m a x } _ { g _ { X Y } } I ^ { p } ( \underline { { X } } ; \underline { { Y } } ) ,\tag{216}
$$

where $I ^ { p } ( \underline { { { X } } } ; \underline { { { Y } } } ) = H ( X ) + H ( Y ) - H _ { p } ( \underline { { { X } } } , \underline { { { Y } } } )$ is the up pragmatic mutual information. For any $\epsilon > 0$ and total rate $R _ { t o t } .$

1) Achievability: If $R _ { t o t } < C _ { p } ,$ then for sufficiently large n, there exists an $( 2 ^ { n ( R _ { p } + R _ { r } ) } , n )$ pragmatic channel code with error probability $P _ { e } ^ { ( n ) } < \epsilon .$

2) Converse: If $R _ { t o t } > C _ { p } ,$ , then for any $( 2 ^ { n ( R _ { p } + R _ { r } ) } , n )$ pragmatic channel code, the error probability $P _ { e } ^ { ( n ) }$ is bounded away from zero for sufficiently large n.

Consequently, the pragmatic cost of information $\operatorname { C o I } _ { p } ( R )$ is finite for $R \leq C _ { p }$ and becomes unbounded for $R > C _ { p } ,$ establishing $C _ { p }$ as the fundamental threshold for resource-limited pragmatic communication.

Proof: Achievability: Fix $\epsilon > 0$ and choose rates $R _ { p } , R _ { r }$ such that $R _ { \mathrm { t o t } } = R _ { p } + R _ { r } < C _ { p }$ . By the definition of $C _ { p } ,$ there exists an input distribution $p ( x )$ and a joint reification mapping $g _ { X Y }$ such that $I ^ { p } ( \underline { { X } } ; \underline { { Y } } ) > R _ { \mathrm { t o t } }$ . Let $R = I ^ { p } ( \underline { { X } } ; \underline { { Y } } ) - \delta$ for some small $\delta > 0$ , so that $R _ { \mathrm { t o t } } < R < C _ { p }$

We construct a random codebook C of size $2 ^ { n ( R _ { p } + R _ { r } ) }$ by drawing each codeword independently according to $\begin{array} { r } { P ( X ^ { n } ) = \prod _ { k = 1 } ^ { n } P ( X _ { k } ) } \end{array}$ . These codewords are then partitioned uniformly into $2 ^ { n R _ { p } }$ groups, each corresponding to a pragmatic index $i _ { p } ,$ with $2 ^ { n R _ { r } }$ codewords per group. This partition is based on the reification mapping: all codewords in group $i _ { p }$ are chosen from the reified typical set $B _ { \epsilon } ^ { ( n ) } ( \underline { { x } } _ { i _ { p } } ^ { n } , \underline { { y } } _ { i _ { p } } ^ { n } )$ for some typical output pragmatic sequence.

The encoder, upon receiving a pragmatic message $i _ { p } ,$ , selects one of the $2 ^ { n R _ { r } }$ codewords in group $i _ { p }$ (any choice is sufficient, as all share the same pragmatic purpose) and transmits it. The decoder uses the jointly typical decoding rule based on the jointly reified typical set: upon receiving $Y ^ { n }$ , it searches for the unique pragmatic index $i _ { p }$ such that there exists a codeword $x ^ { n }$ in group $i _ { p }$ with $( x ^ { n } , Y ^ { n } ) \in B _ { \epsilon } ^ { ( n ) } ( \underline { { x } } _ { i _ { p } } ^ { n } , \underline { { y } } _ { i _ { p } } ^ { n } )$ , i.e., the pair is jointly reified typical. If no such index exists, or if multiple exist, an error is declared.

We now analyze the error probability. Assume codeword $x ^ { n } ( 1 , 1 )$ (from group $i _ { p } = 1 )$ is transmitted. Define the following events:

$E _ { 0 } \colon$ The transmitted codeword and the received sequence are not jointly reified typical. By the jointly reified typical set properties (Theorem 39), the probability of this event tends to 0 as $n \to \infty$ , i.e., $P ( E _ { 0 } ) \leq \epsilon / 2$ for large n.

$E _ { i }$ for $i _ { p } \neq 1$ : There exists some codeword in a wrong pragmatic group that is jointly reified typical with the received sequence. For a fixed wrong group $i _ { p } ,$ , the number of codewords is $2 ^ { n R _ { r } }$ , and each is independent of the received sequence. By the probability bound in Theorem 39, the probability that a given codeword from a wrong group is jointly typical with $Y ^ { n }$ is at most $2 ^ { - n ( I ^ { p } ( \underline { { X } } ; \underline { { Y } } ) - 3 \epsilon ) }$ . Hence, by the union bound,

$$
P ( E _ { i } ) \leq 2 ^ { n R _ { r } } \cdot 2 ^ { - n ( I ^ { p } ( \underline { { X } } ; \underline { { Y } } ) - 3 \epsilon ) } = 2 ^ { - n ( I ^ { p } ( \underline { { X } } ; \underline { { Y } } ) - R _ { r } - 3 \epsilon ) } .
$$

Summing over all $2 ^ { n R _ { p } } - 1$ wrong groups gives

$$
\begin{array} { c } { { P \displaystyle \left( \bigcup _ { i _ { p } \neq 1 } E _ { i } \right) \leq 2 ^ { n R _ { p } } \cdot 2 ^ { - n \left( I ^ { p } ( \underline { { { X } } } ; \underline { { { Y } } } ) - R _ { r } - 3 \epsilon \right) } } } \\ { { } } \\ { { = 2 ^ { - n \left( I ^ { p } ( \underline { { { X } } } ; \underline { { { Y } } } ) - R _ { p } - R _ { r } - 3 \epsilon \right) } } } \\ { { } } \\ { { = 2 ^ { - n \left( I ^ { p } ( \underline { { { X } } } ; \underline { { { Y } } } ) - R _ { \mathrm { t o t } } - 3 \epsilon \right) } . } } \end{array}\tag{217}
$$

Since $R _ { \mathrm { t o t } } < I ^ { p } ( \underline { { X } } ; \underline { { Y } } )$ , this probability tends to 0 as $n \to \infty$ . Therefore, the total error probability $\begin{array} { r } { P _ { e } ^ { ( n ) } \leq P ( E _ { 0 } ) + \sum _ { i _ { p } \neq 1 } P ( E _ { i } ) < \epsilon } \end{array}$ for sufficiently large n. This proves the achievability.

Converse: Assume a code with total rate $R _ { \mathrm { t o t } } > C _ { p }$ and error probability $P _ { e } ^ { ( n ) }$ . Let $W _ { p }$ be the uniformly distributed pragmatic message, and let $\hat { W } _ { p }$ be the decoder’s estimate. $\boldsymbol { \mathrm { B y } }$ Fano’s inequality,

$$
H ( W _ { p } | \hat { W } _ { p } ) \leq 1 + P _ { e } ^ { ( n ) } \log M _ { p } = 1 + P _ { e } ^ { ( n ) } n R _ { p } .
$$

Using the data processing inequality and the fact that $\underline { { X } } ^ { n }$ is a function of $W _ { p }$ and the codebook, we have

$$
n R _ { p } = H ( W _ { p } ) \leq I ( W _ { p } ; Y ^ { n } ) + H ( W _ { p } | Y ^ { n } ) \leq I ( \underline { { X } } ^ { n } ; Y ^ { n } ) + 1 + P _ { e } ^ { ( n ) } n R _ { p } .
$$

Now, by the definition of pragmatic capacity and the chain rule for up pragmatic mutual information (Theorem 12), we have $I ( \underline { { X } } ^ { n } ; Y ^ { n } ) \leq I ^ { p } ( \underline { { X } } ^ { n } ; \underline { { Y } } ^ { n } ) \leq n C _ { p }$ . Thus,

$$
n R _ { p } \leq n C _ { p } + 1 + P _ { e } ^ { ( n ) } n R _ { p } \implies R _ { p } ( 1 - P _ { e } ^ { ( n ) } ) \leq C _ { p } + \frac { 1 } { n } .
$$

Since the total rate $R _ { \mathrm { t o t } } = R _ { p } + R _ { r }$ , and $R _ { r } \geq 0$ , we get

$$
R _ { \mathrm { t o t } } - C _ { p } \leq P _ { e } ^ { ( n ) } R _ { p } + { \frac { 1 } { n } } + R _ { r } .\tag{218}
$$

Letting $n  \infty$ and assuming $P _ { e } ^ { ( n ) } \to 0$ , we obtain $R _ { \mathrm { t o t } } \ \leq \ C _ { p } ,$ a contradiction. Hence, if $R _ { \mathrm { t o t } } > C _ { p }$ , the error probability cannot tend to zero. This proves the converse.

Connection to Pragmatic Cost of Information: The pragmatic cost of information $\operatorname { C o I } _ { p } ( R )$ is the minimum resource required to achieve rate R. Since reliable transmission is possible iff $R \leq C _ { p } , \operatorname { C o I } _ { p } ( R )$ is finite for $R \leq C _ { p }$ and increases to infinity as $R \to C _ { p } .$ . Thus, $C _ { p }$ is the threshold beyond which no finite resource can guarantee reliable pragmatic communication. □

Remark 13. The pragmatic channel coding theorem generalizes Shannon’s classical theorem (when isoteleia and synonymous mappings are trivial, $C _ { p } = C )$ and the semantic theorem (when only the synonymous mapping is nontrivial, $C _ { p } = C _ { s } )$ . By tolerating errors that do not affect the optimal action, pragmatic coding achieves a strictly larger capacity.

The code structure employs a two-level decomposition via the reification mapping $g _ { X Y } ^ { n } =$ $f _ { X Y } ^ { n } \circ e _ { X Y } ^ { n }$ : the pragmatic index set $\mathcal { T } _ { p }$ (rate $R _ { p } )$ identifies optimal actions, while the reification index set $\mathcal { T } _ { r }$ (rate $R _ { r } )$ enumerates syntactic codewords within each pragmatic class. The total rate decomposes as $R _ { t o t } = R _ { p } + R _ { r }$ , with the decoder recovering only $i _ { p } .$ . This two-level structure can be further refined into three levels (pragmatic, isoteleia, synonymous) when intermediate semantic information is also of interest, but the two-level form suffices for capacity achievability.

## IX. PRAGMATIC LOSSY SOURCE CODING

In this section, we investigate the fundamental limits of pragmatic lossy source coding, where the goal is to compress a source while preserving its pragmatic utility rather than its exact syntactic or semantic content. We extend the asymptotic equipartition property to the joint distribution of the source and its reconstruction at the pragmatic level. By introducing the reification mapping for both the source and the reconstruction sequences, we define the jointly reified typical set, which characterizes the equivalence classes of source-reconstruction pairs that share the same pragmatic interpretation. Using random coding and jointly typical encoding based on this typical set, we prove the pragmatic rate-distortion coding theorem, which establishes that the pragmatic rate-distortion function $R _ { p } ( D )$ is the fundamental limit for task-oriented compression.

## A. Pragmatic Distortion and Jointly Typical Set

To establish the fundamental limits of pragmatic lossy source coding, we first extend the asymptotic equipartition property to the joint distribution of the source sequence and its reconstruction at the pragmatic level. Unlike the semantic approach, which relies on the joint synonymous mapping, we directly employ the reification mapping, which is the composition of the synonymous and isoteleia mappings, to aggregate syntactic sequences into pragmatic equivalence classes. This approach provides a direct characterization of the equivalence of source and reconstruction pairs with respect to the optimal terminal actions.

Consider a discrete memoryless source X with alphabet $\mathcal { X }$ and probability mass function $P ( X )$ . Let $\tilde { X }$ be the associated semantic variable and $\varDelta$ be the pragmatic variable induced by the isoteleia mapping $e _ { X } : \underline { { { \mathcal { X } } } } \to 2 ^ { \tilde { \mathcal { X } } }$ and the synonymous mapping $f _ { X } : \tilde { \mathcal { X } } \to 2 ^ { \mathcal { X } }$ , with the reification mapping $g _ { X } = f _ { X } \circ e _ { X } : \underline { { \mathcal { X } } } \to 2 ^ { \mathcal { X } }$ . Similarly, let $\hat { X }$ be the reconstruction variable with associated semantic $\tilde { \hat { X } }$ and pragmatic $\hat { X } ,$ , induced by the reification mapping $g _ { \hat { X } } : \underline { { \hat { X } } } \to 2 ^ { \hat { \mathcal { X } } }$

For lossy source coding, the encoder produces a reconstruction sequence ${ \hat { X } } ^ { n }$ based on a test channel $P ( { \hat { X } } ^ { n } | X ^ { n } )$ . The pragmatic distortion measure $d _ { p } ( \underline { { x } } , \underline { { \hat { x } } } )$ quantifies the loss of utility when a pragmatic symbol $\underline { { x } }$ is represented by $\underline { { \hat { x } } } .$ In practice, it can be defined as the utility loss:

$$
d _ { p } ( \underline { { x } } , \underline { { \hat { x } } } ) = \operatorname* { m a x } _ { a } \mathbb { E } [ U ( X , a ) | \underline { { x } } ] - \operatorname* { m a x } _ { a } \mathbb { E } [ U ( X , a ) | \underline { { \hat { x } } } ] ,\tag{219}
$$

or any other non-negative function that reflects the degradation in decision quality. The average pragmatic distortion is then

$$
\bar { d } _ { p } = \mathbb { E } \left[ d _ { p } ( \underline { { X } } , \underline { { \hat { X } } } ) \right] = \sum _ { \underline { { x } } , \underline { { \hat { x } } } } P ( \underline { { x } } , \underline { { \hat { x } } } ) d _ { p } ( \underline { { x } } , \underline { { \hat { x } } } ) .\tag{220}
$$

We now define the relevant typical sets. Let $P ( X , { \hat { X } } ) = P ( X ) P ( { \hat { X } } | X )$ denote the joint distribution induced by the test channel. The syntactically jointly typical set $A _ { \epsilon } ^ { ( n ) }$ is defined as in classical information theory [1, 45]:

$$
\begin{array} { l } { { A _ { \epsilon } ^ { ( n ) } \triangleq \Big \{ ( x ^ { n } , \hat { x } ^ { n } ) \in \mathcal { X } ^ { n } \times \hat { \mathcal { X } } ^ { n } : \left| - \displaystyle \frac { 1 } { n } \log P ( x ^ { n } ) - H ( X ) \right| < \epsilon , } } \\ { { \left| - \displaystyle \frac { 1 } { n } \log P ( \hat { x } ^ { n } ) - H ( \hat { X } ) \right| < \epsilon , } } \\ { { \left| - \displaystyle \frac { 1 } { n } \log P ( x ^ { n } , \hat { x } ^ { n } ) - H ( X , \hat { X } ) \right| < \epsilon \Big \} . } } \end{array}\tag{221}
$$

For the pragmatic layer, we define the pragmatically jointly typical set, which aggregates probabilities over reified equivalence classes.

Definition 41 (Pragmatically Jointly Typical Set for Source and Reconstruction). The pragmatically jointly typical set $\underline { { A } } _ { \epsilon } ^ { ( n ) }$ is the set of pragmatic sequence pairs $( \underline { { x } } ^ { n } , \underline { { \hat { x } } } ^ { n } ) \in \underline { { \mathcal { X } } } ^ { n } \times \underline { { \hat { \mathcal { X } } } } ^ { n }$ such that

$$
\begin{array} { l } { \displaystyle \underline { { A } } _ { \epsilon } ^ { ( n ) } \triangleq \Big \{ ( \underline { { x } } ^ { n } , \underline { { \hat { x } } } ^ { n } ) : \left. - \frac { 1 } { n } \log P ( \underline { { x } } ^ { n } ) - H _ { p } ( \underline { { X } } ) \right. < \epsilon , } \\ { \displaystyle \left. - \frac { 1 } { n } \log P ( \underline { { \hat { x } } } ^ { n } ) - H _ { p } ( \underline { { \hat { X } } } ) \right. < \epsilon , } \\ { \displaystyle \left. - \frac { 1 } { n } \log P ( \underline { { x } } ^ { n } , \underline { { \hat { x } } } ^ { n } ) - H _ { p } ( \underline { { X } } , \underline { { \hat { X } } } ) \right. < \epsilon \Big \} , } \end{array}\tag{222}
$$

where

$$
P ( \underline { { x } } ^ { n } , \hat { \underline { { x } } } ^ { n } ) = \prod _ { k = 1 } ^ { n } P ( \underline { { x } } _ { k } , \hat { \underline { { x } } } _ { k } ) , \quad P ( \underline { { x } } _ { k } , \hat { \underline { { x } } } _ { k } ) = \sum _ { ( \underline { { x } } _ { k } , \hat { x } _ { k } ) \in g _ { X } \hat { x } ( \underline { { x } } _ { k } , \hat { \underline { { x } } } _ { k } ) } P ( x _ { k } , \hat { x } _ { k } ) ,\tag{223}
$$

with $g _ { X \hat { X } } = g _ { X } \times g _ { \hat { X } }$ being the joint reification mapping for the source and reconstruction. The marginal pragmatic probabilities are obtained by summing over the other variable.

For a given pragmatic pair, the jointly reified typical set collects all syntactic source reconstruction pairs that share that pragmatic interpretation and are syntactically jointly typical.

Definition 42 (Jointly Reified Typical Set for Source and Reconstruction). For a given pragmatic typical pair $( \underline { x } ^ { n } , \underline { x } ^ { n } ) \in \underline { A } _ { \epsilon } ^ { ( n ) }$ , the jointly reified typical set $B _ { \epsilon } ^ { ( n ) } ( \underline { { x } } ^ { n } , \underline { { \hat { x } } } ^ { n } )$ is defined as the set of

syntactic pairs $( x ^ { n } , \hat { x } ^ { n } ) \in \mathcal X ^ { n } \times \hat { \mathcal X } ^ { n }$ such that the following conditions hold:

$$
\begin{array} { r l } { B _ { c } ^ { ( n ) } ( \underline { { x } } ^ { n } , \underline { { x } } ^ { n } ) = \left\{ \left( \begin{array} { l } { - 1 } \\ { \cdots } \\ { - 1 } \end{array} \right) \underline { { \ln } } \mathrm { t o r } P ( \underline { { x } } ^ { n } ) - H ( X ) \right\} \Bigg \leq \epsilon , } & { } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \times \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \times \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \times \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \times \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \times \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \times \quad } \\ &  \quad \quad \quad \quad \quad  \end{array}\tag{224}
$$

The first three conditions ensure syntactic joint typicality. The fourth and fifth conditions enforce individual reification typicality for the source and reconstruction sequences. The sixth condition enforces joint reification typicality. The last condition guarantees that the syntactic pair maps to the given pragmatic pair under the joint reification mapping.

Under the sequential joint reification mapping $g _ { X { \hat { X } } } ^ { n }$ , the syntactically jointly typical set $A _ { \epsilon } ^ { ( n ) }$ can be partitioned into jointly reified typical sets:

$$
A _ { \epsilon } ^ { ( n ) } = \bigcup _ { ( \underline { { x } } ^ { n } , \underline { { \hat { x } } } ^ { n } ) \in \underline { { A } } _ { \epsilon } ^ { ( n ) } } B _ { \epsilon } ^ { ( n ) } ( \underline { { x } } ^ { n } , \underline { { \hat { x } } } ^ { n } ) ,\tag{225}
$$

with disjointness for distinct pragmatic typical pairs.

As shown in Fig. 14, the joint reification mapping $g _ { X \hat { X } } ^ { n } = f _ { X \hat { X } } ^ { n } \circ e _ { X \hat { X } } ^ { n }$ first collapses syntactic sequence pairs into pragmatic equivalence classes according to the optimal actions, yielding the pragmatically jointly typical set $\underline { { A } } _ { \epsilon } ^ { ( n ) }$ . For each pragmatic pair, the preimage is the jointly reified typical set $B _ { \epsilon } ^ { ( n ) }$ , which collects all syntactic pairs sharing the same pragmatic purpose. Within $B _ { \epsilon } ^ { ( n ) }$ , the isoteleia mapping further groups semantic interpretations that lead to the same action, while the synonymous mapping collects syntactic realizations of each meaning. Thus, $B _ { \epsilon } ^ { ( n ) }$ is the union of synonymous typical sets over the isoteleia typical set. This hierarchical coarsening reduces the number of equivalence classes from $| A _ { \epsilon } ^ { ( n ) } |$ to $| \underline { { A } } _ { \epsilon } ^ { ( n ) } |$ |, consistent with the entropy hierarchy $H _ { p } ( \underline { { X } } , \underline { { \hat { X } } } ) \le H _ { s } ( \tilde { X } , \hat { \tilde { X } } ) \le H ( X , \hat { X } )$ , and provides the basis for the pragmatic rate-distortion theorem.

![](images/43f6f084aa1251b65d365b67d0e2dbb72db2b721e363365bf49d59150375ec62.jpg)  
Fig. 14: Hierarchical typical sets for pragmatic lossy source coding. The joint reification mapping $g _ { { } _ { X } \hat { X } } ^ { n } = f _ { { } _ { X } \hat { X } } ^ { n } \circ e _ { { } _ { X } \hat { X } } ^ { n }$ partitions the syntactic joint typical set $A _ { \epsilon } ^ { ( n ) }$ into disjoint jointly reified typical sets $B _ { \epsilon } ^ { ( n ) } ( \underline { { x } } ^ { n } , \underline { { \hat { x } } } ^ { n } )$ , each corresponding to a pragmatic pair in $\underline { { A } } _ { \epsilon } ^ { ( n ) }$ . Each $B _ { \epsilon } ^ { ( n ) }$ is further decomposed into synonymous typical sets $S _ { \epsilon } ^ { ( n ) }$ over the isoteleia typical sets $E _ { \epsilon } ^ { ( n ) }$

We now establish the AEP properties of these typical sets. The syntactically joint AEP is the classical result [1, 45]; we restate it for reference.

Theorem 42 (Syntactically Joint AEP). Let $( X ^ { n } , { \hat { X } } ^ { n } )$ be a sequence pair of length n drawn i.i.d. according to $P ( x ^ { n } , { \hat { x } } ^ { n } )$ . Then for sufficiently large n:

1) $\operatorname* { P r } \{ ( X ^ { n } , { \hat { X } } ^ { n } ) \in A _ { \epsilon } ^ { ( n ) } \} > 1 - \epsilon .$

2) $( 1 - \epsilon ) 2 ^ { n ( H ( X , \hat { X } ) - \epsilon ) } \leq | A _ { \epsilon } ^ { ( n ) } | \leq 2 ^ { n ( H ( X , \hat { X } ) + \epsilon ) } .$

3) ${ \cal I } f \left( \dot { X } ^ { n } , \dot { X } ^ { n } \right) \sim { \cal P } ( x ^ { n } ) { \cal P } ( \hat { x } ^ { n } )$ are independent sequences with the same marginals, then

$$
( 1 - \epsilon ) 2 ^ { - n ( I ( X ; \hat { X } ) + 3 \epsilon ) } \leq \operatorname* { P r } \{ ( \dot { X } ^ { n } , \hat { \dot { X } } ^ { n } ) \in A _ { \epsilon } ^ { ( n ) } \} \leq 2 ^ { - n ( I ( X ; \hat { X } ) - 3 \epsilon ) } .
$$

The pragmatically joint AEP follows by applying the weak law of large numbers to the aggregated pragmatic probabilities.

Theorem 43 (Pragmatically Joint AEP for Compression). Let $( \underline { { \boldsymbol X } } ^ { n } , \hat { \underline { X } } ^ { n } )$ be a pragmatic sequence pair of length n drawn i.i.d. according to $P ( \underline { x } ^ { n } , \underline { x } ^ { n } )$ , with the associated syntactic sequence pair

$( X ^ { n } , { \hat { X } } ^ { n } )$ under the joint reification mapping $g _ { X { \hat { X } } } ^ { n }$ . Then for sufficiently large n:

1) $\operatorname* { P r } \{ ( \underline { { X } } ^ { n } , \underline { { \hat { X } } } ^ { n } ) \in \underline { { A } } _ { \epsilon } ^ { ( n ) } \} > 1 - \epsilon .$

2) For any $( \underline { x } ^ { n } , \underline { x } ^ { n } ) \in \underline { A } _ { \epsilon } ^ { ( n ) }$

$$
2 ^ { - n ( H _ { p } ( \underline { { X } } , \underline { { \hat { X } } } ) + \epsilon ) } \leq P ( \underline { { x } } ^ { n } , \underline { { \hat { x } } } ^ { n } ) \leq 2 ^ { - n ( H _ { p } ( \underline { { X } } , \underline { { \hat { X } } } ) - \epsilon ) } .
$$

3) $( 1 - \epsilon ) 2 ^ { n ( H _ { p } ( \underline { { X } } , \underline { { \hat { X } } } ) - \epsilon ) } \leq | \underline { { A } } _ { \epsilon } ^ { ( n ) } | \leq 2 ^ { n ( H _ { p } ( \underline { { X } } , \underline { { \hat { X } } } ) + \epsilon ) } .$

4) ${ \cal I } f \left( \dot { X } ^ { n } , \dot { X } ^ { n } \right) \sim { \cal P } ( x ^ { n } ) { \cal P } ( \hat { x } ^ { n } )$ are independent sequences with the same marginals as $X ^ { n }$ and ${ \hat { X } } ^ { n } ,$ , and $( \dot { \underline { { X } } } ^ { n } , \hat { \dot { \underline { { X } } } } { } ^ { n } )$ are the corresponding pragmatic sequences, then

$$
( 1 - \epsilon ) 2 ^ { - n ( I _ { p } ( \underline { { X } } ; \underline { { \hat { X } } } ) + 3 \epsilon ) } \leq \operatorname* { P r } \{ ( \underline { { \dot { X } } } ^ { n } , \underline { { \hat { X } } } ^ { n } ) \in \underline { { A } } _ { \epsilon } ^ { ( n ) } \} \leq 2 ^ { - n ( I _ { p } ( \underline { { X } } ; \underline { { \hat { X } } } ) - 3 \epsilon ) } .
$$

The details of the proof are provided in Appendix F.

Finally, we establish the properties of the jointly reified typical set.

Theorem 44 (Properties of the Jointly Reified Typical Set). For any pragmatic typical pair $( \underline { x } ^ { n } , \underline { x } ^ { n } ) \in \underline { A } _ { \epsilon } ^ { ( n ) }$ , for sufficiently large n:

1) For any $( x ^ { n } , \hat { x } ^ { n } ) \in B _ { \epsilon } ^ { ( n ) } ( \underline { { x } } ^ { n } , \underline { { \hat { x } } } ^ { n } )$

$$
2 ^ { - n ( H ( X , \hat { X } ) - H _ { p } ( \underline { { X } } , \underline { { \hat { X } } } ) + \epsilon ) } \leq \frac { P ( x ^ { n } , \hat { x } ^ { n } ) } { P ( \underline { { x } } ^ { n } , \underline { { \hat { x } } } ^ { n } ) } \leq 2 ^ { - n ( H ( X , \hat { X } ) - H _ { p } ( \underline { { X } } , \underline { { \hat { X } } } ) - \epsilon ) } .
$$

2)

$$
2 ^ { n ( H ( X , \hat { X } ) - H _ { p } ( \underline { { X } } , \underline { { \hat { X } } } ) - \epsilon ) } \leq | B _ { \epsilon } ^ { ( n ) } ( \underline { { x } } ^ { n } , \underline { { \hat { x } } } ^ { n } ) | \leq 2 ^ { n ( H ( X , \hat { X } ) - H _ { p } ( \underline { { X } } , \underline { { \hat { X } } } ) + \epsilon ) } .
$$

Proof: Property (1) follows directly from the definition of $B _ { \epsilon } ^ { ( n ) } ( \underline { { x } } ^ { n } , \underline { { \hat { x } } } ^ { n } )$ . The cardinality bounds are obtained by the same summation argument as in Theorem 36 of Section VIII-A. □

Remark 14. The jointly reified typical set $B _ { \epsilon } ^ { ( n ) } ( \underline { { x } } ^ { n } , \underline { { \hat { x } } } ^ { n } )$ groups together all syntactically jointly typical source-reconstruction pairs that share the same pragmatic interpretation. Its definition directly employs the reification mapping $g _ { X \hat { X } } ^ { n } = f _ { X \hat { X } } ^ { n } \circ e _ { X \hat { X } } ^ { n }$ , which is the composition of the joint synonymous and joint isoteleia mappings. Consequently, the jointly reified typical set can be decomposed as the union of jointly synonymous typical sets over the isoteleia typical set:

$$
B _ { \epsilon } ^ { ( n ) } ( \underline { { x } } ^ { n } , \underline { { \hat { x } } } ^ { n } ) = \bigcup _ { ( \tilde { x } ^ { n } , \hat { \tilde { x } } ^ { n } ) \in E _ { \epsilon } ^ { ( n ) } ( \underline { { x } } ^ { n } , \underline { { \hat { x } } } ^ { n } ) } S _ { \epsilon } ^ { ( n ) } ( \tilde { x } ^ { n } , \hat { \tilde { x } } ^ { n } ) ,
$$

where $E _ { \epsilon } ^ { ( n ) } ( \underline { { x } } ^ { n } , \underline { { \hat { x } } } ^ { n } )$ is the isoteleia typical set and $S _ { \epsilon } ^ { ( n ) } ( \tilde { x } ^ { n } , \hat { \tilde { x } } { } ^ { n } )$ is the jointly synonymous typical set. This decomposition reveals the two-level abstraction: the isoteleia mapping merges semantic differences that do not affect the optimal actions, while the synonymous mapping merges syntactic differences that do not affect the meaning. The cardinality of a jointly reified typical set is approximately $2 ^ { n ( H ( X , \hat { X } ) - H _ { p } ( \underline { { X } } , \underline { { \hat { X } } } ) ) }$ , which is larger than a single synonymous typical set by a factor of $2 ^ { n ( H _ { s } ( \tilde { X } , \hat { \tilde { X } } ) - H _ { p } ( \underline { { { X } } } , \hat { \underline { { { X } } } } ) ) }$ , reflecting the additional syntactic freedom gained when only the pragmatic purpose needs to be preserved. When randomly selecting an independent pair of syntactic sequences, the probability that they fall into the same jointly reified typical set (i.e., are pragmatically jointly typical) is about $2 ^ { - n I _ { p } ( \underline { { X } } ; \underline { { \hat { X } } } ) }$ , which is larger than the classical joint typicality probability $2 ^ { - n I ( X ; { \hat { X } } ) }$ , reflecting the pragmatic rate-distortion gain.

## B. Pragmatic Rate-Distortion Coding Theorem

We now investigate the fundamental limit of pragmatic lossy source coding. As depicted in Fig. 15, with the help of the reification mapping $g ^ { n } = f ^ { n } \circ e ^ { n }$ , a pragmatic index $i _ { p }$ is mapped into the syntactic source sequence $X ^ { n } ( i )$ . Considering the distortion requirement, the encoder selects a suitable reconstruction codeword $\hat { X } ^ { n } ( j )$ from a pragmatic class to represent the source, then sends the indices to the receiver. The decoder outputs the reconstruction, and after demapping recovers the estimated pragmatic index.

![](images/b6a8277bcb4f3c3776e7dc2d80a43279b9a65de4972a0b5ee57d43f16a4ace68.jpg)  
Fig. 15: Block diagram of pragmatic lossy source coding.

For a lossy source coding system, let $p ( { \hat { X } } | X )$ be the test channel and $x , { \hat { x } }$ denote the source and reconstruction syntactic alphabets. The conditional distribution for the n-th extension is

$$
p ( \hat { x } ^ { n } | x ^ { n } ) = \prod _ { k = 1 } ^ { n } p ( \hat { x } _ { k } | x _ { k } ) .\tag{226}
$$

Definition 43 (Pragmatic Lossy Source Code). An $( M , n )$ code for pragmatic lossy source coding consists of the following parts:

(1) A pragmatic index set $T _ { p } = \{ 1 , \ldots , M _ { p } \}$ and a source reification index set $\mathcal { T } _ { r } = \{ 1 , \dots , M _ { r } \}$ a reconstruction reification index set $\begin{array} { c c l } { { \mathcal { Z } _ { c } } } & { { = } } & { { \left\{ 1 , . . . , M _ { c } \right\} } } \end{array}$ . The syntactic source and reconstruction index sets are $\mathcal { T } = \{ 1 , \dots , M \}$ and $\mathcal { T } ^ { \prime } = \{ 1 , \dots , M ^ { \prime } \}$

(2) A reification mapping $g _ { X } ^ { n } : \underline { { \mathcal { X } } } ^ { n } \to \mathcal { X } ^ { n }$ generates the set of source sequences, namely the pragmatic sourcebook $\mathcal { S } = \{ X ^ { n } ( 1 ) , \ldots , X ^ { n } ( M ^ { \prime } ) \}$ . This book is partitioned into reified source subsets $S _ { r }$ according to the pragmatic index.

(3) An encoding function $\phi : \mathcal { X } ^ { n } \to \hat { \mathcal { X } } ^ { n }$ generates the reconstruction codebook

$\mathcal { C } = \{ \hat { X } ^ { n } ( 1 ) , \ldots , \hat { X } ^ { n } ( M ) \}$ , partitioned into pragmatic codeword subsets $\mathcal { C } _ { p }$ via the reification mapping $g _ { \hat { X } } ^ { n }$ .

(4) A decoding function $\psi : \hat { \mathcal { X } } ^ { n } \to \hat { \mathcal { X } } ^ { n }$ outputs the reconstruction ${ \hat { X } } ^ { n }$

(5) After demapping, $g _ { \hat { X } } ^ { n } ( { \hat { X } } ^ { n } ) = { \hat { i } } _ { p } ,$ , the estimated pragmatic index is obtained. Both $\psi$ and $g _ { \hat { X } } ^ { n }$ are deterministic.

The quotient sets ${ \cal S } / g _ { X } ^ { n } = \{ { \cal S } _ { r } \}$ and ${ \mathcal C } / g _ { \hat { X } } ^ { n } = \{ { \mathcal C } _ { p } \}$ have sizes $| S / g _ { X } ^ { n } | = M _ { r }$ and $| { \mathcal C } / g _ { \hat { X } } ^ { n } | = M _ { p }$ Define the pragmatic code rate $\begin{array} { r } { R = { \frac { 1 } { n } } \log _ { 2 } M _ { p } } \end{array}$ , the source reification rate $\begin{array} { r } { R _ { r } = \frac { 1 } { n } \log _ { 2 } M _ { r } } \end{array}$ , and the reconstruction reification rate $\begin{array} { r } { R _ { c } = \frac { 1 } { n } \log _ { 2 } M _ { c } } \end{array}$ . We assume each reified set has equal size: $| S _ { r } | = M ^ { \prime } / M _ { r } = 2 ^ { n R _ { r } }$ and $| { \mathcal { C } } _ { p } | = M / M _ { p } = 2 ^ { n R _ { c } }$ . The total syntactic rate is $R _ { \mathrm { t o t } } = R + R _ { c }$

Theorem 45 (Pragmatic Rate-Distortion Coding Theorem). Given an i.i.d. syntactic source $X \sim p ( x )$ with associated pragmatic variable $X$ under the reification mapping $g _ { X }$ , and a bounded pragmatic distortion measure $d _ { p } ( \underline { { x } } , \underline { { \hat { x } } } )$ , define the pragmatic rate-distortion function as

$$
R _ { p } ( D ) = \operatorname* { m i n } _ { g _ { X } , g _ { \hat { X } } \atop \mathbb { E } [ d _ { p } ( \underline { { X } } , \underline { { \hat { X } } } ) ] \leq D } I _ { p } ( \underline { { X } } ; \underline { { \hat { X } } } ) ,\tag{227}
$$

where $I _ { p } ( \underline { { X } } ; \underline { { \hat { X } } } ) = H _ { p } ( \underline { { X } } ) + H _ { p } ( \underline { { \hat { X } } } ) - H ( X , \hat { X } )$ is the down pragmatic mutual information.

For any $\epsilon > 0 .$

1) Achievability: If $R > R _ { p } ( D )$ , then for sufficiently large n, there exists an $\left( 2 ^ { n \left( R + R _ { c } \right) } , n \right)$ code such that the average pragmatic distortion satisfies $\bar { d } _ { p } \leq D + \epsilon$

2) Converse: If $R < R _ { p } ( D )$ , then for any $( 2 ^ { n ( R + R _ { c } ) } , n )$ code, the average distortion $\bar { d } _ { p } > D - \epsilon$ for sufficiently large n.

Moreover, the pragmatic Value of Information (VoI) achieved by the reconstruction is maximized $( i . e .$ , equals the VoI obtained from the full pragmatic variable X) if and only if $R \ge R _ { p } ( D )$ If $R < R _ { p } ( D )$ , the achievable pragmatic VoI is strictly less than the full value, and the gap is determined by the rate shortfall.

Proof: Achievability: Fix $\epsilon > 0$ and choose a test channel $p ( \hat { x } | x )$ such that $\mathbb { E } [ d _ { p } ( \underline { { X } } , \underline { { \hat { X } } } ) ] \leq$ $D / ( 1 + \epsilon )$ . Let $I _ { p } = I _ { p } ( \underline { { X } } ; \underline { { \hat { X } } } )$ and select a pragmatic rate R satisfying $R _ { p } ( D ) < R < I _ { p } - 3 \epsilon$

Generate a pragmatic sourcebook $\boldsymbol { s }$ of size $2 ^ { n ( R + R _ { r } ) }$ by drawing sequences independently from $\begin{array} { r } { p ( x ^ { n } ) = \prod _ { k = 1 } ^ { n } p ( x _ { k } ) } \end{array}$ . Partition it uniformly into $2 ^ { n R }$ groups $S _ { r } ( i _ { r } )$ according to the reification mapping $g _ { X } ^ { n }$ , each of size $2 ^ { n R _ { \eta } }$ , such that $S _ { r } ( i _ { r } ) \subset B _ { \epsilon } ^ { ( n ) } ( \underline { { x } } _ { i _ { r } } )$ for some representative pragmatic sequence.

Generate a reconstruction codebook $\mathcal { C }$ of size $2 ^ { n ( R + R _ { c } ) }$ independently from $\begin{array} { r } { p ( \hat { x } ^ { n } ) = \prod _ { k = 1 } ^ { n } p ( \hat { x } _ { k } ) } \end{array}$ where $\begin{array} { r } { p ( \hat { x } ) = \sum _ { x } p ( x ) p ( \hat { x } | x ) } \end{array}$ . Partition it uniformly into $2 ^ { n R }$ groups $\mathcal { C } _ { p } ( i _ { p } )$ according to $g _ { \hat { X } } ^ { n }$ each of size $2 ^ { n R _ { c } }$ , with $\mathcal { C } _ { p } ( i _ { p } ) \subset B _ { \epsilon } ^ { ( n ) } ( \underline { { \hat { x } } } _ { i _ { p } } )$

Given a source sequence $X ^ { n }$ , the encoder first finds the unique source reification group $S _ { r } ( w _ { r } )$ containing it. Then, based on the distortion criterion, it searches for a pragmatic index $w _ { p }$ and a codeword $\hat { X } ^ { n } ( w _ { p } , l ) \in \mathcal { C } _ { p } ( w _ { p } )$ such that $( X ^ { n } , \hat { X } ^ { n } ( w _ { p } , l ) ) \in B _ { \epsilon } ^ { ( n ) }$ (jointly reified typical). If more than one such pair exists, choose the smallest index; if none, set $w _ { p } = 1$ and $l = 1$ . The encoder sends the indices $( w _ { p } , l )$

Define the error event $\mathcal { E }$ as the event that the chosen pair is not jointly reified typical. It can be decomposed into

$$
\mathcal { E } = \mathcal { E } _ { 1 } \cup \mathcal { E } _ { 2 } ,
$$

where $\mathcal { E } _ { 1 } = \{ X ^ { n } \notin A _ { \epsilon } ^ { ( n ) } ( X ^ { n } ) \}$ , and

$$
\mathcal { E } _ { 2 } = \Big \{ X ^ { n } \in \mathcal { S } _ { r } ( w _ { r } ) , ~ \hat { X } ^ { n } ( w _ { p } , l ) \in \mathcal { C } _ { p } ( w _ { p } ) , ~ ( X ^ { n } , \hat { X } ^ { n } ( w _ { p } , l ) ) \not \in B _ { \epsilon } ^ { ( n ) } , ~ \forall w _ { p } , l \Big \} .
$$

By the AEP, $\operatorname* { P r } ( { \mathcal { E } } _ { 1 } ) \to 0 { \mathrm { ~ a s ~ } } n \to \infty$ . For ${ \mathcal { E } } _ { 2 }$ , using the fact that $S _ { r } ( w _ { r } ) \subset B _ { \epsilon } ^ { ( n ) } ( \underline { { x } } _ { w _ { r } } )$ and $\mathcal { C } _ { p } ( w _ { p } ) \subset B _ { \epsilon } ^ { ( n ) } ( \underline { { \hat { x } } } _ { w _ { p } } )$ , we have

$$
\begin{array} { r l } & { P ( \mathcal { E } _ { 2 } ) = \displaystyle \sum _ { x ^ { n } \in A _ { \epsilon } ^ { ( n ) } } p ( x ^ { n } ) \prod _ { w _ { p } = 1 } ^ { 2 ^ { n R } } \operatorname* { P r } \left( ( X ^ { n } , \hat { X } ^ { n } ( w _ { p } , l ) ) \notin B _ { \epsilon } ^ { ( n ) } \Big | X ^ { n } = x ^ { n } \right) } \\ & { \qquad \le \displaystyle \sum _ { x ^ { n } \in A _ { \epsilon } ^ { ( n ) } } p ( x ^ { n } ) \left[ 1 - \operatorname* { P r } \left( ( X ^ { n } , \hat { X } ^ { n } ( 1 , l ) ) \in B _ { \epsilon } ^ { ( n ) } \right) \right] ^ { 2 ^ { n R } } . } \end{array}\tag{228}
$$

Since $x ^ { n } \in A _ { \epsilon } ^ { \left( n \right) }$ and $\hat { X } ^ { n } ( 1 , l )$ is drawn independently from $p ( \hat { x } ^ { n } )$ , by the pragmatically joint AEP (Theorem 43), the probability that the pair is jointly reified typical is bounded below by

$$
\operatorname* { P r } \left( ( X ^ { n } , \hat { X } ^ { n } ( 1 , l ) ) \in B _ { \epsilon } ^ { ( n ) } \right) \geq ( 1 - \epsilon ) 2 ^ { - n ( I _ { p } + 3 \epsilon ) } ,\tag{229}
$$

for sufficiently large n. This bound follows from the fact that the probability that an independent pair falls into the pragmatically jointly typical set $\underline { { A } } _ { \epsilon } ^ { ( n ) }$ is at least $( 1 - \epsilon ) 2 ^ { - n ( I _ { p } + 3 \epsilon ) }$ , and $B _ { \epsilon } ^ { ( n ) } \subseteq \underline { { A } } _ { \epsilon } ^ { ( n ) }$

Using $( 1 - x ) ^ { m } \leq e ^ { - m x }$ for $x \in [ 0 , 1 ]$ and $m \geq 0$ , we obtain

$$
\begin{array} { r } { P ( \mathcal { E } _ { 2 } ) \le \left( 1 - ( 1 - \epsilon ) 2 ^ { - n ( I _ { p } + 3 \epsilon ) } \right) ^ { 2 ^ { n R } } } \\ { \le \exp \left( - ( 1 - \epsilon ) 2 ^ { n ( R - I _ { p } - 3 \epsilon ) } \right) . } \end{array}\tag{230}
$$

Since $R < I _ { p } - 3 \epsilon$ , the exponent tends to $- \infty$ , hence $P ( \mathcal { E } _ { 2 } )  0$ . Therefore the total encoding error probability $P _ { e } ^ { ( n ) } \to 0$

Conditioned on the success event $\mathcal { E } ^ { c } \ ( \mathrm { i . e . }$ , the pair is jointly reified typical), the average distortion is bounded by

$$
\mathbb { E } [ d _ { p } ( \underline { { X } } ^ { n } , \underline { { \hat { X } } } ^ { n } ) \mid \mathcal { E } ^ { c } ] \leq ( 1 + \epsilon ) \mathbb { E } [ d _ { p } ( \underline { { X } } , \underline { { \hat { X } } } ) ] \leq D .
$$

Let $d _ { \mathrm { m a x } }$ be the maximum distortion. The overall expected distortion satisfies

$$
\mathbb { E } [ d _ { p } ( \underline { { X } } ^ { n } , \underline { { \hat { X } } } ^ { n } ) ] \le P _ { e } ^ { ( n ) } d _ { \operatorname* { m a x } } + ( 1 - P _ { e } ^ { ( n ) } ) D \le D + \epsilon ,\tag{231}
$$

for sufficiently large $n .$ . This proves achievability.

Converse: Assume a sequence of $( 2 ^ { n ( R + R _ { c } ) } , n )$ codes with average distortion $\bar { d } _ { p } \leq D$ . Let $W _ { p }$ be the uniform pragmatic message (index $i _ { p } )$ , and let $\hat { W } _ { p }$ be its estimate at the decoder. We will show that $R \geq R _ { p } ( D )$ must hold.

First, by the data processing inequality and the fact that $W _ { p } \to X ^ { n } \to { \hat { X } } ^ { n }$ forms a Markov chain (where ${ \hat { X } } ^ { n }$ is the reconstruction), we have

$$
I ( W _ { p } ; X ^ { n } ) \geq I ( \underline { { \hat { X } } } ^ { n } ( W _ { p } ) ; X ^ { n } ) ,\tag{232}
$$

where $\underline { { \hat { X } } } ^ { n } ( W _ { p } )$ denotes the pragmatic reconstruction associated with the chosen codeword. Since $W _ { p }$ is a function of $\underline { { \hat { X } } } ^ { n } ( W _ { p } )$ , we have $H ( W _ { p } ) \leq I ( W _ { p } ; X ^ { n } ) + H ( W _ { p } \mid X ^ { n } )$ . However, a more direct route is used in [21].

We proceed as follows:

$$
n R = H ( W _ { p } ) \geq I ( W _ { p } ; X ^ { n } )\tag{233}
$$

$$
\geq I ( \underline { { \hat { X } } } ^ { n } ( W _ { p } ) ; X ^ { n } )\tag{234}
$$

$$
\geq I _ { p } ( \underline { { X } } ^ { n } ; \underline { { \hat { X } } } ^ { n } ) .\tag{235}
$$

Inequality (234) follows from the data processing inequality because $\hat { \underline { X } } ^ { n }$ is a function of $W _ { p }$ (through the codebook). Inequality (235) is a pragmatic analogue of Lemma 10 in [21, 22], which states that the mutual information between the syntactic source and the pragmatic reconstruction is at least the down pragmatic mutual information between the pragmatic source and the pragmatic reconstruction. Indeed, by the definition of down pragmatic mutual information and the fact that the reconstruction variable is a function of the channel output, one can verify that $I ( X ^ { n } ; { \hat { \underline { { X } } } } ^ { n } ) \geq$ $I _ { p } ( \underline { { X } } ^ { n } ; \underline { { \hat { X } } } ^ { n } )$ (this follows from the entropy hierarchy and the data processing inequality).

Since the source is memoryless, the down pragmatic mutual information satisfies

$$
I _ { p } ( \underline { { { X } } } ^ { n } ; \underline { { { \hat { X } } } } ^ { n } ) = \sum _ { k = 1 } ^ { n } I _ { p } ( \underline { { { X } } } _ { k } ; \underline { { { \hat { X } } } } _ { k } ) ,\tag{236}
$$

because the pragmatic variables are also memoryless due to the deterministic reification mapping.

By the definition of the pragmatic rate-distortion function $R _ { p } ( D )$ , for each $k ,$ we have

$$
I _ { p } ( \underline { { X } } _ { k } ; \underline { { \hat { X } } } _ { k } ) \ge R _ { p } \left( \mathbb { E } [ d _ { p } ( \underline { { X } } _ { k } , \underline { { \hat { X } } } _ { k } ) ] \right) .\tag{237}
$$

Summing over $k$ and using the convexity of $R _ { p } ( D )$ , we obtain

$$
\sum _ { k = 1 } ^ { n } I _ { p } ( \underline { { X } } _ { k } ; \underline { { \hat { X } } } _ { k } ) \geq \sum _ { k = 1 } ^ { n } R _ { p } \left( \mathbb { E } [ d _ { p } ( \underline { { X } } _ { k } , \underline { { \hat { X } } } _ { k } ) ] \right) \geq n R _ { p } \left( \frac { 1 } { n } \sum _ { k = 1 } ^ { n } \mathbb { E } [ d _ { p } ( \underline { { X } } _ { k } , \underline { { \hat { X } } } _ { k } ) ] \right) .\tag{238}
$$

The average distortion over the entire sequence is $\begin{array} { r } { \bar { d } _ { p } ~ = ~ \frac { 1 } { n } \sum _ { k = 1 } ^ { n } \mathbb { E } [ d _ { p } ( \underline { { X } } _ { k } , \underline { { \hat { X } } } _ { k } ) ] ~ \leq ~ D } \end{array}$ by assumption. Since $R _ { p } ( D )$ is non-increasing, $R _ { p } ( \bar { d } _ { p } ) \geq R _ { p } ( D )$ . Therefore,

$$
I _ { p } ( \underline { { X } } ^ { n } ; \underline { { \hat { X } } } ^ { n } ) \geq n R _ { p } ( \bar { d } _ { p } ) \geq n R _ { p } ( D ) .\tag{239}
$$

Combining (233)–(235) with (239), we obtain

$$
n R \geq n R _ { p } ( D ) \quad \Longrightarrow \quad R \geq R _ { p } ( D ) .\tag{240}
$$

Thus, if $R < R _ { p } ( D )$ , the assumption $\bar { d } _ { p } \leq D$ leads to a contradiction. Hence, for any code with $R < R _ { p } ( D )$ , the average distortion must satisfy $\bar { d } _ { p } > D - \epsilon$ for sufficiently large n. This completes the converse.

Connection to Pragmatic Value of Information The pragmatic Value of Information achieved by the reconstruction $\hat { \underline { X } }$ is

$$
\operatorname { V o I } _ { p } ( \underline { { \hat { X } } } ) = \mathbb { E } _ { X , \hat { X } } \left[ U ( X , a ^ { * } ( \underline { { \hat { X } } } ) ) \right] - U _ { 0 } ,\tag{241}
$$

where $a ^ { * } ( \hat { \underline { { X } } } )$ is the optimal action based on the reconstructed pragmatic class. The full VoI (as if $\underline { { \boldsymbol { X } } }$ were directly observed) is attained if and only if the reconstruction preserves all pragmatic distinctions, which requires $I _ { p } ( \underline { { X } } ; \underline { { \hat { X } } } ) = H _ { p } ( \underline { { X } } )$ . This occurs only when $R \geq H _ { p } ( \underline { { X } } )$ . More generally, for a given distortion level D, the full VoI is attainable iff $R \geq R _ { p } ( D )$ . If $R < R _ { p } ( D )$ the loss in pragmatic information reduces the achievable utility. □

Remark 15. The pragmatic rate-distortion theorem generalizes both the classical rate-distortion theorem [1, 45] and the semantic rate-distortion theorem [21, 22]. When the reification mapping $g$ is trivial (identity), $R _ { p } ( D ) = R ( D )$ ; when only the synonymous mapping f is non-trivial, we recover $R _ { p } ( D ) = R _ { s } ( D )$ . In general, the hierarchy $R _ { p } ( D ) \leq R _ { s } ( D ) \leq R ( D )$ holds. Although our code uses three levels (pragmatic + two reification rates), the reification mapping $g = f$ ◦ e implicitly includes the semantic layer. The extra reification rates $R _ { r }$ and $R _ { c }$ allow flexible trade-$o f f s .$ setting them to zero achieves the pure pragmatic compression limit $R _ { p } ( D )$ , while increasing them preserves additional syntactic or semantic details, recovering the classical or semantic limits. This modularity is consistent with the design principles of pragmatic information systems.

## X. PRAGMATIC INFORMATION MEASURE OF CONTINUOUS MESSAGE

In this section, we extend the pragmatic information measures to continuous messages, completing the theoretical framework with both bilateral and single-sided characterizations. Building upon the isoteleic volume formalism, we derive the pragmatic capacity and rate-distortion functions for Gaussian channels and sources, respectively. More importantly, we introduce the corresponding pragmatic cost of information (CoI) and pragmatic value of information (VoI) for both bilateral and single-sided cases, establishing the full duality spectrum: CoI is the inverse of the achievable rate functions, while VoI is the Legendre-Fenchel dual of the rate-distortion functions. These results provide a complete information-theoretic and decision-theoretic characterization of task-oriented communication over continuous alphabets, encompassing perceptual (observationto-action) and expressive (action-to-reconstruction) asymmetric scenarios.

## A. Pragmatic Entropy and Pragmatic Mutual Information for Continuous Message

In this subsection, we extend the pragmatic information measures, such as pragmatic entropy and pragmatic mutual information, to the domain of continuous messages. Unlike the syntactic layer, which deals with exact signal values, or the semantic layer, which deals with meanings, the pragmatic layer is concerned with the consequences of actions. For continuous signals, the equivalence is determined by whether different realizations lead to the same optimal action, as defined by a utility function. This gives rise to the concept of isoteleic volumes, which are the continuous analogues of the pragmatic equivalence classes (fibers) introduced in Section II.

The coding theorems established in Sections VII–IX are developed for discrete, finite alphabets, where the syntactic, semantic, and pragmatic layers are characterized by quotient spaces of finite cardinality. For continuous alphabets, the pragmatic equivalence classes induced by the isoteleia mapping become uncountable sets, and the discrete entropy measures must be replaced by their differential counterparts. To bridge the discrete and continuous formulations, we adopt the standard quantization argument: partition the continuous space W into small cells of volume $\Delta ,$ , apply the discrete theory to the resulting finite alphabet, and then take the limit $\Delta \to 0$ . Under this limiting procedure, the quotient-space structure of the isoteleia mapping converges to the continuous notion of isoteleic volumes Ω, which quantify the average measure of signal space that maps to the same optimal action. The entropy hierarchy $H _ { p } \leq H _ { s } \leq H$ is preserved in the limit, with the discrete pragmatic entropy giving way to the differential pragmatic entropy defined below. Consequently, all subsequent Gaussian channel and source characterizations are consistent with—and indeed inherit the operational meaning of—the discrete coding theorems proved earlier.

To formalize pragmatic information for continuous messages, we first define the isoteleia mapping for continuous variables and introduce the notion of isoteleic volumes.

Definition 44 (Isoteleic Volume). Let W be a continuous random variable with alphabet $\mathcal { W } \subseteq \mathbb { R } ^ { d }$ and probability density function $p ( w )$ . Let $\tilde { \mathcal W }$ be the associated semantic alphabet, and let W be the pragmatic alphabet. The pragmatic variable W is induced by the isoteleia mapping $e : \underline { { \mathcal { W } } } \to 2 ^ { \tilde { \mathcal { W } } }$ , which, in turn, is derived from a utility function $U ( s , a )$ defined on states and actions.

For each pragmatic symbol ${ \underline { { w } } } _ { i } \in { \underline { { w } } } .$ , its isoteleic volume $\Omega _ { i }$ is defined as the total measure of the set of all semantic (and hence syntactic) realizations that map to the same optimal action. Formally,

$$
\Omega _ { i } \triangleq | \{ w \in \mathcal { W } : a ^ { * } ( w ) = \underline { { w } } _ { i } \} | ,\tag{242}
$$

where $a ^ { * } ( w )$ is the optimal action for the realization w, defined by

$$
a ^ { * } ( w ) = \arg \operatorname* { m a x } _ { a \in \mathcal { A } } \mathbb { E } _ { S | w } [ U ( S , a ) ] .\tag{243}
$$

The collection $\{ \Omega _ { i } \} _ { i = 1 } ^ { | \mathcal { W } | }$ forms a partition of the support of W, i.e., $\textstyle \bigcup _ { i } \Omega _ { i } = { \mathcal { W } }$ and $\Omega _ { i } \cap \Omega _ { j } = \varnothing$ for $i \neq j$

Analogous to the average synonymous length S in semantic information theory, we define the average isoteleic volume as the expectation of the isoteleic volume under the distribution of W.

Definition 45 (Average Isoteleic Volume). The average isoteleic volume Ω is defined as

$$
\Omega \triangleq \mathbb { E } _ { w \sim p ( w ) } \left[ \left| \Omega _ { a ^ { * } ( w ) } \right| \right] = \sum _ { i = 1 } ^ { | \mathcal { W } | } \operatorname* { P r } ( \underline { { w } } _ { i } ) \cdot | \Omega _ { i } | ,\tag{244}
$$

where $\begin{array} { r } { \mathrm { P r } ( \underline { { w } } _ { i } ) = \int _ { \Omega _ { i } } p ( w ) d w } \end{array}$ is the probability mass of the pragmatic class $\underline { w } _ { i } .$

The average isoteleic volume Ω quantifies the “compression” achieved by grouping different continuous signals into the same pragmatic class. A larger Ω indicates that more distinct signals lead to the same optimal action, thus reducing decision uncertainty.

1) Pragmatic Entropy for Continuous Messages:

We now define the pragmatic entropy for a continuous random variable. The definition is derived by quantizing the continuous space and taking the limit, analogous to the derivation of differential entropy.

Definition 46 (Pragmatic Entropy for Continuous Variables). Let W be a continuous random variable with probability density function $p ( w )$ , supported on $\mathcal { W } \subseteq \mathbb { R } ^ { d }$ . Let W be the associated pragmatic variable induced by the isoteleia mapping, with average isoteleic volume Ω. The pragmatic entropy of W is defined as

$$
H _ { p } ( \underline { { W } } ) \triangleq - \int _ { \mathcal { W } } p ( w ) \log p ( w ) d w - \mathbb { E } [ \log \Omega ] ,\tag{245}
$$

where $\begin{array} { r } { \mathbb E [ \log \Omega ] = \sum _ { i } \operatorname* { P r } ( \underline { { w } } _ { i } ) \log \left| \Omega _ { i } \right| } \end{array}$ is the expectation of the logarithm of the isoteleic volume.

Remark 16. The first term in Eq. (245) is the classical differential entropy $h ( W ) \ =$ $- \int p ( w ) \log p ( w ) d w$ . The second term, $\mathbb { E } [ \log \Omega ]$ , accounts for the reduction in uncertainty due to pragmatic abstraction. Unlike the discrete case, the pragmatic differential entropy $H _ { p } ( \underline { { W } } )$ is not an absolute measure of information but a relative one, defined up to a reference measure.

To maintain a meaningful non-negative pragmatic entropy (in prabits), the average isoteleic volume Ω must satisfy: $\mathbb { E } [ \log \Omega ] \le h ( W )$ . When the isoteleia mapping is trivial (i.e., each signal leads to a unique action, $\Omega = 1 )$ , the pragmatic entropy reduces to the differential entropy: $H _ { p } ( \underline { { W } } ) = h ( W )$

Corollary 8 (Lower Bound on Pragmatic Entropy). The pragmatic entropy is lower bounded by

$$
H _ { p } ( \underline { { W } } ) \geq h ( W ) - \log \Omega .\tag{246}
$$

Equality holds when the isoteleic volumes are chosen proportionally to the probability mass of the pragmatic classes, i.e., when the partition is optimal.

Example 5 (Uniform Distribution). Let W be uniformly distributed over an interval of length L, i.e., $p ( w ) = 1 / L$ for $w \in [ 0 , L ]$ . Suppose the isoteleia mapping partitions this interval into K equal isoteleic volumes, each of size $\Omega _ { i } = L / K$ . Then the average isoteleic volume is $\Omega = L / K$ The pragmatic entropy is

$$
H _ { p } ( \underline { { { W } } } ) = - \int _ { 0 } ^ { L } \frac { 1 } { L } \log \frac { 1 } { L } d w - \log \frac { L } { K } = \log K \ p r a b i t s .\tag{247}
$$

This result is analogous to the semantic entropy for a uniformly distributed source, but here the interpretation is in terms of optimal actions rather than meanings.

Example 6 (Gaussian Distribution). Let $W \sim { \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ be a Gaussian random variable. Under an optimal isoteleia mapping with average isoteleic volume Ω, the pragmatic entropy is

$$
\begin{array} { l } { { H _ { p } ( \underline { { { W } } } ) = \mathbb { E } [ - \log p ( W ) ] - \log \Omega } } \\ { ~ } \\ { { \displaystyle = \frac { 1 } { 2 } \log ( 2 \pi e \sigma ^ { 2 } ) - \log \Omega } } \\ { { \displaystyle ~ = \frac { 1 } { 2 } \log \left( \frac { 2 \pi e \sigma ^ { 2 } } { \Omega ^ { 2 } } \right) ~ p r a b i t s . } } \end{array}\tag{248}
$$

This shows that the pragmatic entropy of a Gaussian source decreases logarithmically with the average isoteleic volume.

The definitions can be extended to multiple continuous variables. Let (W, V) be a pair of continuous random variables with joint density $p ( w , v )$ , and let (W, V) be their associated pragmatic variables, induced by a joint isoteleia mapping with average isoteleic volume $\Omega _ { w v }$

Definition 47 (Joint Pragmatic Entropy for Continuous Variables). The joint pragmatic entropy $o f \left( \underline { { W } } , \underline { { V } } \right)$ is defined as

$$
H _ { p } ( \underline { { W } } , \underline { { V } } ) \triangleq - \int _ { \mathscr { W } \times \mathscr { V } } p ( w , v ) \log p ( w , v ) d w d v - \mathbb { E } [ \log \Omega _ { w v } ] ,\tag{249}
$$

where $\Omega _ { w v }$ is the average joint isoteleic volume.

Similarly, the conditional pragmatic entropy of V given W is defined as

$$
H _ { p } ( \underline { { V } } | W ) \triangleq - \int _ { \mathscr { W } \times \mathscr { V } } p ( w , v ) \log p ( v | w ) d w d v - \mathbb { E } [ \log \Omega _ { v } ] ,\tag{250}
$$

where $\Omega _ { v }$ is the average isoteleic volume for V.

2) Pragmatic Mutual Information for Continuous Variables:

We now define the pragmatic mutual information for continuous variables. As in the discrete case, there are two variants: the up pragmatic mutual information and the down pragmatic mutual information.

Definition 48 (Up Pragmatic Mutual Information for Continuous Variables). Let $( W , V )$ be a pair of continuous random variables with joint density $p ( w , v )$ . Let W and $\underline { { V } }$ be their pragmatic counterparts, induced by a joint isoteleia mapping with average isoteleic volumes $\Omega _ { w }$ and $\Omega _ { v }$ The up pragmatic mutual information is defined as

$$
\begin{array} { l } { \displaystyle { I ^ { p } ( \underline { { W } } ; \underline { { V } } ) \triangleq H ( W ) + H ( V ) - H _ { p } ( \underline { { W } } , \underline { { V } } ) } } \\ { \displaystyle { \quad \quad = - \int _ { W \times \mathscr { V } } p ( w , v ) \log \frac { p ( w ) p ( v ) } { p ( w , v ) } d w d v + \mathbb { E } [ \log ( \Omega _ { w } \Omega _ { v } ) ] . } } \end{array}\tag{251}
$$

Definition 49 (Down Pragmatic Mutual Information for Continuous Variables). The down pragmatic mutual information is defined as

$$
\begin{array} { l } { \displaystyle I _ { p } ( \underline { W } ; \underline { V } ) \triangleq H _ { p } ( \underline { W } ) + H _ { p } ( \underline { V } ) - H ( W , V ) } \\ { \displaystyle \quad \quad = - \int _ { \mathscr W \times \mathscr V } p ( w , v ) \log \frac { p ( w ) p ( v ) } { p ( w , v ) } d w d v - \mathbb E [ \log ( \Omega _ { w } \Omega _ { v } ) ] . } \end{array}\tag{252}
$$

Remark 17. The first term in both (251) and (252) is the classical mutual information $I ( W ; V )$ The difference lies in the second term. If $\mathbb { E } [ \log ( \Omega _ { w } \Omega _ { v } ) ] \geq 0 .$ , then the up pragmatic mutual information is greater than or equal to the classical mutual information, while the down pragmatic mutual information is less than or equal to it. In practice, the down pragmatic mutual information may be negative; in such cases, one can take its positive part, $( I _ { p } ( \underline { { W } } ; \underline { { V } } ) ) ^ { + }$

The relationship between these measures is summarized by the following hierarchy, which mirrors the discrete case:

$$
I _ { p } ( \underline { { W } } ; \underline { { V } } ) \leq I ( W ; V ) \leq I ^ { p } ( \underline { { W } } ; \underline { { V } } ) .\tag{253}
$$

Example 7 (Gaussian Variables). Let (W, V ) be jointly Gaussian with correlation coefficient $\rho$ and marginal variances $\sigma _ { w } ^ { 2 }$ and $\sigma _ { v } ^ { 2 } .$ . The classical mutual information is $I ( W ; V ) = - \textstyle { \frac { 1 } { 2 } } \log ( 1 - \rho ^ { 2 } )$ Under a joint isoteleia mapping with average volumes $\Omega _ { w }$ and $\Omega _ { v } ,$ , the up and down pragmatic mutual informations are

$$
I ^ { p } ( { \underline { { W } } } ; { \underline { { V } } } ) = - { \frac { 1 } { 2 } } \log ( 1 - \rho ^ { 2 } ) + \log ( \Omega _ { w } \Omega _ { v } ) ,\tag{254}
$$

$$
I _ { p } ( \underline { { W } } ; \underline { { V } } ) = - \frac { 1 } { 2 } \log ( 1 - \rho ^ { 2 } ) - \log ( \Omega _ { w } \Omega _ { v } ) .\tag{255}
$$

This demonstrates how pragmatic abstraction can either enhance or reduce the measured information, depending on whether the coarsening is applied to the joint entropy or to the marginal entropies.

This completes the extension of pragmatic entropy and mutual information to continuous messages. In the following subsections, we will apply these measures to derive the pragmatic capacity of the Gaussian channel and the pragmatic rate-distortion function for Gaussian sources, along with their associated cost and value counterparts.

## B. Pragmatic Capacity of Gaussian Channel and Pragmatic Cost of Information

We now investigate the pragmatic capacity of the additive white Gaussian noise (AWGN) channel and its dual—the pragmatic cost of information. The received signal at time k is modeled as

$$
Y _ { k } = X _ { k } + Z _ { k } ,\tag{256}
$$

with $X _ { k }$ being the channel input, $Z _ { k } \sim { \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ i.i.d. Gaussian noise, and $Y _ { k }$ the output. The channel is subject to an average power constraint $\mathbb { E } [ X ^ { 2 } ] \le P$

Let $\varDelta$ and $\underline { { Y } }$ be the pragmatic variables associated with the input and output, respectively, induced by the isoteleia mappings $e _ { X }$ and $e _ { Y }$ (and their compositions with the synonymous mappings) according to a given utility function. The pragmatic channel capacity is defined as the

maximum achievable rate of pragmatic information transmission, i.e.,

$$
C _ { p } \triangleq \operatorname* { m a x } _ { p ( x ) } \operatorname* { m a x } _ { g _ { X Y } } I ^ { p } ( \underline { { X } } ; \underline { { Y } } ) ,\tag{257}
$$

where $g _ { X Y } = f _ { X Y } \circ e _ { X Y }$ is the joint reification mapping, and $I ^ { p } ( \underline { { { X } } } ; \underline { { { Y } } } ) = H ( X ) + H ( Y ) -$ $H _ { p } ( \underline { { X } } , \underline { { Y } } )$ is the up pragmatic mutual information.

For the Gaussian channel, the input distribution that maximizes the classical mutual information is $X \sim { \mathcal { N } } ( 0 , P ) $ . Under the pragmatic framework, we assume that the joint isoteleia mapping partitions the input-output space such that the average isoteleic volumes for X and Y are $\Omega _ { x }$ and $\Omega _ { y } ,$ respectively. Without loss of generality, we consider a symmetric case where $\Omega _ { x } = \Omega _ { y } = \Omega$ with $\Omega \geq 1$ being the average isoteleic volume. This volume represents the average measure of the set of signals that lead to the same optimal terminal action.

Using the continuous version of the up pragmatic mutual information derived in (251), we have

$$
\begin{array} { l } { { \displaystyle I ^ { p } ( \underline { { { X } } } ; \underline { { { Y } } } ) = I ( X ; Y ) + \mathbb { E } [ \log ( \Omega _ { x } \Omega _ { y } ) ] } } \\ { { \displaystyle \quad = \frac { 1 } { 2 } \log \left( 1 + \frac { P } { \sigma ^ { 2 } } \right) + \log ( \Omega ^ { 2 } ) } } \\ { { \displaystyle \quad = \frac { 1 } { 2 } \log \left( \Omega ^ { 4 } \left( 1 + \frac { P } { \sigma ^ { 2 } } \right) \right) . } } \end{array}\tag{258}
$$

Thus, the pragmatic capacity of the Gaussian channel (per real dimension) is given by

$$
C _ { p } = \frac { 1 } { 2 } \log \left( \Omega ^ { 4 } \left( 1 + \frac { P } { \sigma ^ { 2 } } \right) \right) \mathrm { ~ p r a b i t s ~ p e r ~ c h a n n e l ~ u s e } .\tag{259}
$$

By the capacity-cost duality established in Section VI, the bilateral pragmatic cost of information is the inverse of the capacity function. From Eq. (259), solving for P as a function of R yields:

$$
\mathrm { C o I } _ { p } ( R ) = \sigma ^ { 2 } \left( { \frac { 2 ^ { 2 R } } { \Omega ^ { 4 } } } - 1 \right) ~ \mathrm { f o r } ~ R \ge 0 .\tag{260}
$$

This represents the minimum transmit power required to achieve a pragmatic rate R (in prabits per channel use) over the AWGN channel.

For asymmetric systems where only one side is coarsened, we have two additional cost measures, that is, perceptual/expressive pragmatic cost of information. For the perceptual path (observation $Y$ carries information about pragmatic action X), the prospective perceptual achievable rate from Section V-C is:

$$
C _ { p } ^ { \mathrm { p - p e r c } } ( P ) = \frac { 1 } { 2 } \log \left( 1 + \frac { P } { \sigma ^ { 2 } } \right) + \log \Omega = \frac { 1 } { 2 } \log \left( \Omega ^ { 2 } \left( 1 + \frac { P } { \sigma ^ { 2 } } \right) \right) .\tag{261}
$$

The inverse gives the perceptual pragmatic cost of information:

$$
\mathrm { C o I } _ { p } ^ { \mathrm { p - p e r c } } ( R ) = \sigma ^ { 2 } \left( \frac { 2 ^ { 2 R } } { \Omega ^ { 2 } } - 1 \right) .\tag{262}
$$

This is the minimum sensing power required to extract R prabits of pragmatic information from the observation, when the observation itself is not coarsened.

For the expressive path (pragmatic label $\underline { { Y } }$ conveys information about source X), the prospective expressive achievable rate is:

$$
C _ { p } ^ { \mathrm { p - e x p r } } ( P ) = \frac { 1 } { 2 } \log \left( \Omega ^ { 2 } \left( 1 + \frac { P } { \sigma ^ { 2 } } \right) \right) ,\tag{263}
$$

which is identical to the perceptual case under symmetry. Thus,

$$
\mathrm { C o I } _ { p } ^ { \mathrm { p - e x p r } } ( R ) = \sigma ^ { 2 } \left( \frac { 2 ^ { 2 R } } { \Omega ^ { 2 } } - 1 \right) .\tag{264}
$$

Remark 18 (Comparison of CoI Measures). The three CoI measures satisfy the hierarchy:

$$
\mathrm { C o I } _ { p } ( R ) \leq \mathrm { C o I } _ { p } ^ { p - p e r c } ( R ) = \mathrm { C o I } _ { p } ^ { p - e x p r } ( R ) ,\tag{265}
$$

since $\Omega ^ { 4 } \geq \Omega ^ { 2 } f o r \Omega \geq 1$ . This confirms the general principle: bilateral pragmatic abstraction (coarsening both sides) is more efficient than single-sided abstraction, requiring less power to achieve the same pragmatic rate.

We now extend the result to the band-limited Gaussian channel. Consider a Gaussian channel with bandwidth B (Hz), two-sided power spectral density $N _ { 0 } / 2$ , and signal power constraint P. The classical Shannon capacity for this channel is

$$
C = B \log \left( 1 + { \frac { P } { N _ { 0 } B } } \right) \quad { \mathrm { b i t s ~ p e r ~ s e c o n d } } .\tag{266}
$$

By the Shannon-Nyquist sampling theorem, a band-limited signal of duration $T$ can be represented by $2 B T$ independent samples per real dimension. The total pragmatic capacity over the 2BT dimensions gives:

$$
C _ { p } = B \log \left( \Omega ^ { 4 } \left( 1 + { \frac { P } { N _ { 0 } B } } \right) \right) { \mathrm { ~ p r a b i t s ~ p e r ~ s e c o n d } } .\tag{267}
$$

The corresponding bilateral CoI (power per second) is:

$$
\mathrm { C o I } _ { p } ( R ) = N _ { 0 } B \left( \frac { 2 ^ { R / B } } { \Omega ^ { 4 } } - 1 \right) \quad \mathrm { ( w a t t s ) } .\tag{268}
$$

For the single-sided cases, the perceptual and expressive CoI become:

$$
\mathrm { C o I } _ { p } ^ { \mathsf { p - p e r c } } ( R ) = \mathrm { C o I } _ { p } ^ { \mathsf { p - e x p r } } ( R ) = N _ { 0 } B \left( \frac { 2 ^ { R / B } } { \Omega ^ { 2 } } - 1 \right) .\tag{269}
$$

Example 8 (Numerical comparison of pragmatic cost measures). Let $\sigma ^ { 2 } = 1$ for the discrete-time case, and for the band-limited case set B = 1 Hz, $N _ { 0 } = 1$ . Take $\Omega = 2 .$ . The table below compares the minimum required power (in units of $\sigma ^ { 2 }$ or watts) for various target rates R (in prabits per channel use or prabits/s) under bilateral and single-sided coarsening. Negative values indicate that zero power suffices (the rate lies below the zero-power capacity).

TABLE IX: CoI values for different scenarios and rates.
<table><tr><td>Scenario</td><td>CoI expression</td><td> $R = 1$ </td><td> $R = 2$ </td><td> $R = 3$ </td></tr><tr><td>Discrete bilateral</td><td> $\sigma ^ { 2 } ( 2 ^ { 2 R } / \Omega ^ { 4 } - 1 )$ </td><td>-0.75 (0)</td><td>0 (0)</td><td>3</td></tr><tr><td>Discrete perceptual/expressive</td><td> $\sigma ^ { 2 } ( 2 ^ { 2 R } / \Omega ^ { 2 } - 1 )$ </td><td>0 (0)</td><td>3</td><td>15</td></tr><tr><td>Band-limited bilateral</td><td> $N _ { 0 } B ( 2 ^ { R / B } / \Omega ^ { 4 } - 1 )$ </td><td>−0.875 (0)</td><td>−0.75 (0)</td><td>−0.5 (0)</td></tr><tr><td>Band-limited perceptual/expressive</td><td> $N _ { 0 } B ( 2 ^ { R / B } / \Omega ^ { 2 } - 1 )$ </td><td>−0.5 (0)</td><td>0 (0)</td><td>1</td></tr></table>

For the discrete bilateral case, the zero-power capacity is $\begin{array} { r } { C _ { p } ( 0 ) = \frac { 1 } { 2 } \log ( 1 6 ) = 2 } \end{array}$ prabits per channel use, so any $R \leq 2$ requires zero power. For the discrete perceptual path, $C _ { p } ^ { \mathrm { p - p e r c } } ( 0 ) =$ $\textstyle { \frac { 1 } { 2 } } \log ( 4 ) = 1$ prabit, thus $R = 1$ is free but $R = 2$ costs 3 units. In the band-limited case with $B = 1$ , the bilateral zero-power capacity is $C _ { p } ( 0 ) = \log ( 1 6 ) = 4$ prabits/s, hence $R = 3$ is still free; the perceptual zero-power capacity is log(4) = 2 prabits/s, so $R = 2$ is free and $R = 3$ requires 1 W. These numerical results illustrate that bilateral abstraction (coarsening both source and observation) yields the greatest power savings for a given pragmatic rate.

Remark 19. The pragmatic capacity and CoI formulas reveal that the isoteleic volume Ω acts as a power amplification factor. In the band-limited case, $\Omega ^ { 4 }$ effectively multiplies the signal-to-noise ratio, while $\Omega ^ { 2 }$ multiplies it for single-sided cases. This provides a clear design guideline: to minimize transmit power for a given pragmatic rate, one should exploit pragmatic abstraction on both the source and the observation sides whenever possible.

## C. Pragmatic Rate Distortion of Gaussian Source and Pragmatic Value of Information

As a dual to the channel capacity problem, we now investigate the pragmatic rate-distortion function for a Gaussian source and its corresponding pragmatic value of information. This function characterizes the minimum rate required to represent a continuous source such that the distortion, measured at the pragmatic level, does not exceed a given threshold.

Consider a Gaussian source $X \sim { \mathcal { N } } ( 0 , P ) $ with variance $P .$ . Let $\hat { X }$ be the reconstruction of $X$ and let $\varDelta$ and $\hat { \underline { X } }$ be the corresponding pragmatic variables induced by the isoteleia mappings $e _ { X }$ and $e _ { \hat { X } }$ (and their compositions with synonymous mappings). The distortion measure is defined at the pragmatic level as the mean squared error between the optimal actions corresponding to the source and its reconstruction:

$$
d _ { p } ( \underline { { x } } , \underline { { \hat { x } } } ) \triangleq \left( a ^ { * } ( \underline { { x } } ) - a ^ { * } ( \underline { { \hat { x } } } ) \right) ^ { 2 } ,\tag{270}
$$

where $a ^ { * } ( \underline { { x } } )$ is the optimal action for the pragmatic class $\underline { { x } } .$ For the Gaussian source, we adopt the standard mean squared error (MSE) distortion measure, which is widely used and has a well-known rate-distortion characterization. Specifically, we consider the distortion constraint E $\left[ ( X - { \hat { X } } ) ^ { 2 } \right] \leq D$ , which, under appropriate isoteleia mappings, translates to a pragmatic distortion constraint.

The pragmatic rate-distortion function is defined as

$$
R _ { p } ( D ) \triangleq \operatorname* { m i n } _ { g _ { X } , g _ { \hat { X } } } \operatorname* { m i n } _ { \substack { p ( \hat { x } | x ) : \mathbb { E } [ d _ { p } ( \underline { { X } } , \underline { { \hat { X } } } ) ] \leq D } } I _ { p } ( \underline { { X } } ; \underline { { \hat { X } } } ) ,\tag{271}
$$

where $I _ { p } ( \underline { { X } } ; \underline { { \hat { X } } } ) = H _ { p } ( \underline { { X } } ) + H _ { p } ( \underline { { \hat { X } } } ) - H ( X , \hat { X } )$ is the down pragmatic mutual information.

For a Gaussian source with variance $P _ { \mathrm { : } }$ , the classical rate-distortion function is $R ( D ) =$ ${ \scriptstyle { \frac { 1 } { 2 } } } \log ( P / D )$ for $0 \le D \le P$ . Under the pragmatic framework, the isoteleia mapping groups signals that lead to the same optimal action, effectively reducing the “volume” of the source space that needs to be distinguished. Let $\Omega _ { x }$ and $\Omega _ { \hat { x } }$ be the average isoteleic volumes for the source and reconstruction, respectively. Assuming symmetry, we set $\Omega _ { x } = \Omega _ { \hat { x } } = \Omega$ , where $\Omega \geq 1$ is the average isoteleic volume.

Using the continuous expression for the down pragmatic mutual information derived in Eq.

(252), we have

$$
\begin{array} { l } { { \displaystyle { I _ { p } ( \underline { { X } } ; \underline { { \hat { X } } } ) = I ( X ; \hat { X } ) - \mathbb { E } [ \log ( \Omega _ { x } \Omega _ { \hat { x } } ) ] } } } \\ { { \displaystyle \quad \geq \frac { 1 } { 2 } \log \left( \frac { P } { D } \right) - \log ( \Omega ^ { 2 } ) } } \\ { { \displaystyle \quad = \frac { 1 } { 2 } \log \left( \frac { P } { \Omega ^ { 4 } D } \right) , } } \end{array}\tag{272}
$$

where the inequality follows from the fact that for any test channel satisfying the MSE constraint $\mathbb { E } [ ( X - \hat { X } ) ^ { 2 } ] \le D$ , the classical mutual information satisfies $I ( X ; \hat { X } ) \ge \textstyle { \frac { 1 } { 2 } } \log ( P / D )$ . The equality is achieved when the test channel is Gaussian and the isoteleia mapping is optimal (i.e., partitions the space proportionally to the probability mass of the pragmatic classes).

Thus, the pragmatic rate-distortion function for a Gaussian source is given by

$$
R _ { p } ( D ) = \left\{ \begin{array} { l l } { \displaystyle \frac { 1 } { 2 } \log \left( \frac { P } { \Omega ^ { 4 } D } \right) , } & { 0 \leq D \leq \frac { P } { \Omega ^ { 4 } } , } \\ { 0 , } & { D > \displaystyle \frac { P } { \Omega ^ { 4 } } . } \end{array} \right.\tag{273}
$$

By the rate-distortion-value duality established in Section VI, the bilateral pragmatic value of information is the Legendre-Fenchel dual of the rate-distortion function. For the Gaussian quadratic utility case, the value-rate function $\Phi _ { p } ( R )$ is obtained by solving the distortion-rate relationship.

From Eq. (273), the distortion as a function of rate R is:

$$
D _ { p } ( R ) = \frac { P } { \Omega ^ { 4 } } 2 ^ { - 2 R } , \quad 0 \leq R < \infty .\tag{274}
$$

For a quadratic utility $U ( X , A ) = - ( X - A ) ^ { 2 }$ , the maximum expected utility at rate R is:

$$
\Phi _ { p } ( R ) = - D _ { p } ( R ) = - \frac { P } { \Omega ^ { 4 } } 2 ^ { - 2 R } .\tag{275}
$$

The baseline utility (without information) is $U _ { 0 } = - P$ . Therefore, the bilateral pragmatic value of information is:

$$
\operatorname { V o I } _ { p } ( R ) = \Phi _ { p } ( R ) - U _ { 0 } = P \left( 1 - { \frac { 2 ^ { - 2 R } } { \Omega ^ { 4 } } } \right) .\tag{276}
$$

This is the maximum utility gain achievable when both the source and the reconstruction are coarsened pragmatically.

For asymmetric systems, we have two single-sided value measures corresponding to the perceptual and expressive rate-distortion functions from Section V-C. Due to symmetry, these two cases yield identical expressions. For either side, the rate-distortion function is

$$
R _ { p } ^ { \mathrm { s i d e } } ( D ) = \frac 1 2 \log \left( \frac { P } { \Omega ^ { 2 } D } \right) , \quad 0 \leq D \leq \frac { P } { \Omega ^ { 2 } } ,\tag{277}
$$

with corresponding distortion-rate function $\begin{array} { r } { D _ { p } ^ { \mathrm { s i d e } } ( R ) ~ = ~ \frac { P } { \Omega ^ { 2 } } 2 ^ { - 2 R } } \end{array}$ . The pragmatic value of information for both the perceptual and expressive paths is therefore

$$
\operatorname { V o I } _ { p } ^ { \mathrm { p e r c } } ( R ) = \operatorname { V o I } _ { p } ^ { \mathrm { e x p r } } ( R ) = P \left( 1 - { \frac { 2 ^ { - 2 R } } { \Omega ^ { 2 } } } \right) .\tag{278}
$$

We shall refer to either of these as the single-sided VoI.

Remark 20 (Comparison of VoI Measures). The three VoI measures satisfy the hierarchy:

$$
\mathrm { V o I } _ { p } ^ { p e r c } ( R ) = \mathrm { V o I } _ { p } ^ { e x p r } ( R ) \le \mathrm { V o I } _ { p } ( R ) ,\tag{279}
$$

since $1 / \Omega ^ { 2 } \leq 1 / \Omega ^ { 4 }$ for $\Omega \geq 1$ . This is consistent with the general principle that bilateral pragmatic abstraction (coarsening both the source and the reconstruction) yields higher utility gain for a given rate than single-sided abstraction, because it discards more decision-irrelevant information, allowing the same rate to be used more efficiently for the task.

Example 9 (Numerical comparison of VoI and distortion for Gaussian source). Let $P = 1$ and $\Omega = 2 .$ . The table below compares the pragmatic value of information VoI(R) and the corresponding minimal distortion $D ( R )$ for bilateral, perceptual, and expressive coarsening at selected rates R (in prabits). Recall that under symmetry, perceptual and expressive values are identical.

TABLE X: VoI and distortion for Gaussian source with $P = 1$ $\Omega = 2$
<table><tr><td rowspan=1 colspan=1>Scenario</td><td rowspan=1 colspan=1>Measure</td><td rowspan=1 colspan=1> $R = 0$ </td><td rowspan=1 colspan=1>R = 1</td><td rowspan=1 colspan=1>R = 2</td></tr><tr><td rowspan=1 colspan=1>Bilateral</td><td rowspan=1 colspan=1> ${ \mathrm { V o I } } _ { p } ( R )$  $D _ { p } ( R ) = P / \Omega ^ { 4 } \cdot 2 ^ { - 2 R }$ </td><td rowspan=1 colspan=1>0 $1 / 1 6 = 0 . 0 6 2 5$ </td><td rowspan=1 colspan=1>0.984375 $1 / 6 4 \approx 0 . 0 1 5 6 2 5$ </td><td rowspan=1 colspan=1>0.999023 $1 / 2 5 6 \approx 0 . 0 0 3 9 0 6$ </td></tr><tr><td rowspan=2 colspan=1>Perceptual / Expressive</td><td rowspan=2 colspan=1> $\mathrm { V o I } _ { p } ^ { \mathrm { p e r c / e x p r } } ( R )$  $D _ { p } ^ { \mathrm { p e r c / e x p r } } ( R ) = P / \Omega ^ { 2 } \cdot 2 ^ { - 2 R }$ </td><td rowspan=2 colspan=1>0 $1 / 4 = 0 . 2 5$ </td><td rowspan=1 colspan=1>0.9375</td><td rowspan=1 colspan=1>0.996094</td></tr><tr><td rowspan=1 colspan=1> $1 / 1 6 = 0 . 0 6 2 5$ </td><td rowspan=1 colspan=1> $1 / 6 4 \approx 0 . 0 1 5 6 2 5$ </td></tr></table>

At R = 1, bilateral coarsening yields a VoI of 0.984 utils and distortion 0.0156, while singlesided coarsening gives 0.9375 utils and distortion 0.0625 — a fourfold reduction in distortion for the same rate. Equivalently, to achieve the single-sided distortion of 0.0625 with bilateral coarsening, only $R = 0$ is needed (since $D _ { p } ( 0 ) = 0 . 0 6 2 5 )$ , saving 1 prabit. As $R \to \infty$ , both VoI approach 1 (perfect action reconstruction) and distortion tends to 0. The classical case $\Omega = 1$ is recovered by setting $\Omega = 1$ , which gives $\mathrm { V o I } ( R ) = 1 - 2 ^ { - 2 R }$ , strictly smaller than any pragmatic VoI for finite R.

This example quantifies the benefit of coarsening both ends: bilateral pragmatic compression achieves lower distortion and higher utility gain than single-sided compression at the same rate, highlighting the efficiency of task-oriented abstraction.

Remark 21. The closed-form expressions for CoI and VoI derived above assume that the action space is continuous Euclidean, enabling the use of MSE as the distortion measure. For discrete action spaces (e.g., classification tasks), the corresponding CoI and VoI functions must be derived using discrete distortion measures such as Hamming distance or utility loss, which generally do not admit such simple closed forms. The general methodology, however, remains the same: CoI is the inverse of the achievable rate functions, and VoI is the Legendre-Fenchel dual of the rate-distortion functions, both extending naturally to the single-sided cases via the perceptual and expressive mutual informations.

The pragmatic cost of information and pragmatic value of information for Gaussian channels and sources complete the bilateral and single-sided duality spectrum in the continuous domain, providing a comprehensive framework for task-oriented communication system design with explicit resource-utility trade-offs.

## XI. JOINT OPTIMIZATION OF PRAGMATIC INFORMATION SYSTEM

In this section, we present the pragmatic source-channel coding theorem, which extends the classical separation principle to the pragmatic domain and establishes that reliable transmission is possible iff the pragmatic entropy (or pragmatic rate-distortion function) does not exceed the pragmatic channel capacity. This condition is equivalent to the existence of a scheme that achieves the full pragmatic VoI while keeping the pragmatic CoI within the resource budget, thereby bridging information-theoretic limits with decision-theoretic objectives. We then formulate a unified optimization over communication rate, control law, and pragmatic abstraction in closedloop systems, and derive optimality conditions via the pragmatic Lagrangian for co-designing communication and control under resource constraints.

## A. Pragmatic Source-Channel Coding

We now consider the joint source-channel coding problem in the pragmatic domain, where the goal is to transmit a syntactic source over a noisy channel such that the receiver can recover the pragmatic intention rather than the exact source symbols. This problem generalizes Shannon’s classical source-channel coding theorem by relaxing the requirement of exact symbol reconstruction and allowing errors that do not affect the optimal action.

Consider a discrete memoryless syntactic source W with alphabet W and probability mass function $P ( W )$ . Let $\tilde { W }$ and W be the associated semantic and pragmatic variables induced by the synonymous mapping $f : \tilde { \mathcal { W } } \to 2 ^ { \mathcal { W } }$ and the isoteleia mapping $e : \underline { { \mathcal { W } } } \to 2 ^ { \tilde { \mathcal { W } } }$ , respectively, with the reification mapping $g = f \circ e : \underline { { \mathcal { W } } }  2 ^ { \mathcal { W } }$ . The source generates an i.i.d. sequence $W ^ { n }$ according to $\begin{array} { r } { P ( W ^ { n } ) = \prod _ { k = 1 } ^ { n } P ( W _ { k } ) } \end{array}$

Let the channel be a discrete memoryless channel with input alphabet $x ,$ output alphabet Y, and transition probability $P ( \boldsymbol { Y } | \boldsymbol { X } )$ . The pragmatic channel capacity $C _ { p }$ is defined as in Section V-A: $C _ { p } \triangleq \operatorname* { m a x } _ { p ( x ) } \operatorname* { m a x } _ { g _ { X Y } } I ^ { p } ( \underline { { X } } ; \underline { { Y } } )$ , where $I ^ { p } ( \underline { { X } } ; \underline { { Y } } )$ is the up pragmatic mutual information, and $g _ { X Y }$ is the joint reification mapping.

![](images/a2f6e2b90ced8944e8ea27bf96c77bf5556652393615924dfba0ee2561fba0cc.jpg)  
Fig. 16: Block diagram of pragmatic source channel coding.

Figure 16 illustrates the block diagram of pragmatic source-channel coding. At the transmitter side, the syntactic source $W ^ { n }$ first passes through the reification mapping $g ^ { n } = f ^ { n } \circ e ^ { n }$ , which collapses semantically equivalent syntactic symbols into pragmatic classes $W ^ { n }$ representing the optimal terminal actions. These pragmatic classes are then encoded into channel codewords $X ^ { n }$ via a pragmatic channel encoder, which may exploit the structure of the pragmatic equivalence classes to achieve higher rates than classical coding. The codewords are transmitted over the noisy channel, producing the received sequence $Y ^ { n }$ . At the receiver side, the pragmatic channel decoder recovers an estimate of the pragmatic index ${ \hat { W } } ^ { n }$ from the channel output, ignoring syntactic and semantic variations that do not affect the terminal action. The demapping operation $g ^ { - n }$ then maps the estimated pragmatic index to a representative syntactic sequence ${ \hat { W } } ^ { n }$ , which is passed to the destination. The entire system is evaluated by the error probability $P _ { e } ^ { ( n ) } =$ $\operatorname* { P r } \{ g ^ { n } ( \psi _ { n } ( Y ^ { n } ) ) \neq g ^ { n } ( W ^ { n } ) \}$ , i.e., the probability that the recovered pragmatic intention differs from the true one. This architecture embodies the separation principle: the pragmatic source code (reification mapping) and the pragmatic channel code can be designed independently while preserving asymptotic optimality.

A pragmatic source-channel code of block length n consists of:

1) An encoding function $\phi _ { n } : \mathcal { W } ^ { n } \to \mathcal { X } ^ { n }$ that maps each source sequence to a channel input sequence, possibly exploiting the reification mapping $g ^ { n }$ to first map the source to its pragmatic class;

2) A decoding function $\psi _ { n } : \mathcal { V } ^ { n } \to \mathcal { W } ^ { n }$ that maps each channel output to an estimated source sequence (or directly to an estimated pragmatic index), followed by demapping $g ^ { n }$ to recover the pragmatic intention.

The error probability is defined as the probability that the decoded pragmatic intention differs from the true pragmatic intention:

$$
P _ { e } ^ { ( n ) } \triangleq \operatorname* { P r } \left\{ g ^ { n } ( \psi _ { n } ( Y ^ { n } ) ) \neq g ^ { n } ( W ^ { n } ) \right\} .\tag{280}
$$

We now state the fundamental theorem of pragmatic source-channel coding.

Theorem 46 (Pragmatic Source-Channel Coding Theorem). Let W be a discrete memoryless syntactic source with associated pragmatic variable W having pragmatic entropy $H _ { p } ( \underline { { W } } )$ , and let $p ( y | x )$ be a discrete memoryless channel with pragmatic capacity $C _ { p } .$

1) Lossless Case: If $H _ { p } ( \underline { { W } } ) < C _ { p } ,$ , then for any $\epsilon > 0$ , there exists, for sufficiently large n, a sequence of pragmatic source-channel codes with error probability $P _ { e } ^ { ( n ) } < \epsilon .$ . On the contrary, if $H _ { p } ( \underline { { W } } ) > C _ { p }$ , then for any sequence of pragmatic source-channel codes, the error probability is bounded away from zero for sufficiently large n.

2) Lossy Case: For a given pragmatic distortion measure $d _ { p } ( \underline { { w } } , \underline { { \hat { w } } } )$ and distortion constraint $D , \ i f \ R _ { p } ( D ) < C _ { p }$ , then there exists a sequence of pragmatic source-channel codes such that $\mathbb { E } [ d _ { p } ( \underline { { W } } , \underline { { \hat { W } } } ) ] \leq D$ . On the contrary, if $R _ { p } ( D ) > C _ { p } ,$ then no such sequence exists.

Proof: We outline the proof following the standard separation principle, adapted to the pragmatic domain.

Achievability: Assume $H _ { p } ( \underline { { W } } ) < C _ { p } .$ . Choose $\epsilon > 0$ such that $H _ { p } ( \underline { { W } } ) + \epsilon < C _ { p } - \epsilon$ . By the pragmatic lossless source coding theorem (Theorem 37), there exists a pragmatic source code that maps the source sequences to pragmatic indices at rate $R _ { s } < H _ { p } ( \underline { { W } } ) + \epsilon$ with error probability less than $\epsilon / 2$ . By the pragmatic channel coding theorem (Theorem 41), since $R _ { s } < C _ { p } ,$ there exists a pragmatic channel code that transmits these indices over the channel with error probability less than $\epsilon / 2$ . By the union bound, the overall error probability is less than ϵ. The decoder first recovers the pragmatic index from the channel output, then outputs a representative syntactic sequence from the corresponding reified typical set.

Converse: Suppose a sequence of codes exists with $P _ { e } ^ { ( n ) } \to 0$ . Following the standard Fano’s inequality argument adapted to the pragmatic domain, we have

$$
H _ { p } ( \underline { { W } } ) \leq \frac { 1 } { n } I ( \underline { { W } } ^ { n } ; \underline { { Y } } ^ { n } ) + \delta _ { n } ,\tag{281}
$$

where $\delta _ { n } \to 0$ as $P _ { e } ^ { ( n ) } \to 0$ . By the definition of pragmatic channel capacity and the data processing inequality,

$$
\frac { 1 } { n } I ( \underline { { W } } ^ { n } ; \underline { { Y } } ^ { n } ) \leq C _ { p } .\tag{282}
$$

Thus, $H _ { p } ( \underline { { W } } ) \leq C _ { p } ,$ which contradicts $H _ { p } ( \underline { { W } } ) > C _ { p } .$ . The proof for the lossy case follows analogously using the pragmatic rate-distortion function $R _ { p } ( D )$ □

The pragmatic source-channel coding theorem has a direct interpretation in terms of the pragmatic value of information (VoI) and pragmatic cost of information (CoI) established in Section VI.

The condition $H _ { p } ( \underline { { W } } ) < C _ { p } \ ( \mathrm { o r } \ R _ { p } ( D ) < C _ { p } )$ can be equivalently stated as the existence of a rate R such that:

$$
\operatorname { V o I } _ { p } ( R ) = \operatorname { V o I } _ { p } ^ { \operatorname* { m a x } } \quad { \mathrm { a n d } } \quad \operatorname { C o I } _ { p } ( R ) \leq P _ { \operatorname* { m a x } } ,\tag{283}
$$

where $\mathrm { V o I } _ { p } ^ { \mathrm { m a x } }$ is the maximum achievable pragmatic value (the utility gain obtained when all pragmatic distinctions are preserved), and $P _ { \mathrm { m a x } }$ is the available physical resource budget (e.g., maximum transmit power). This follows from the monotonicity of VoI and CoI:

• The pragmatic value of information ${ \mathrm { V o I } } _ { p } ( R )$ is non-decreasing in R and saturates at its maximum when $R \geq H _ { p } ( \underline { { W } } )$ (for lossless transmission) or $R \ \geq \ R _ { p } ( D )$ (for lossy transmission).

• The pragmatic cost of information $\operatorname { C o I } _ { p } ( R )$ is non-decreasing and convex in R, representing the minimum resource required to achieve rate R.

Therefore, the achievability condition $H _ { p } ( \underline { { W } } ) < C _ { p }$ guarantees that there exists a rate R such that:

$$
\operatorname { V o I } _ { p } ( R ) = \operatorname { V o I } _ { p } ^ { \operatorname* { m a x } } \quad { \mathrm { a n d } } \quad \operatorname { C o I } _ { p } ( R ) \leq P _ { \operatorname* { m a x } } .\tag{284}
$$

Conversely, if $H _ { p } ( \underline { { W } } ) > C _ { p }$ , then even at the maximum achievable rate $C _ { p } ,$ the system cannot transmit enough information to preserve all pragmatic distinctions, resulting in a loss of pragmatic value:

$$
\mathrm { V o I } _ { p } ( C _ { p } ) < \mathrm { V o I } _ { p } ^ { \mathrm { m a x } } .\tag{285}
$$

This gap represents the pragmatic value loss due to the channel bottleneck.

Corollary 9 (Pragmatic Value-Cost Characterization). A pragmatic communication system achieves the full pragmatic value of information if and only if the transmission rate R satisfies:

$$
R \geq H _ { p } ( \underline { { W } } ) ( l o s s l e s s ) o r R \geq R _ { p } ( D ) ( l o s s y ) ,\tag{286}
$$

and the available physical resources $P _ { \mathrm { m a x } }$ satisfy:

$$
\operatorname { C o I } _ { p } ( R ) \leq P _ { \operatorname* { m a x } } .\tag{287}
$$

Remark 22. The pragmatic source-channel coding theorem provides a rigorous theoretical foundation for task-oriented communication. Unlike classical source-channel coding, which requires the syntactic entropy H(W) to be less than the syntactic channel capacity C, the pragmatic version relaxes this condition to $H _ { p } ( \underline { { W } } ) \leq C _ { p } .$ Since $H _ { p } ( \underline { { W } } ) \leq H ( W )$ and $C _ { p } \geq C _ { : }$ the pragmatic condition is strictly weaker than the classical one. This explains the performance gains observed in semantic and pragmatic communication systems: by tolerating syntactic and semantic errors that do not affect the optimal action, the system can operate reliably in regimes where classical communication would fail.

Example 10 (Autonomous Driving Revisited). Consider the two-vehicle autonomous driving scenario from Sections III and IV, where the syntactic source has entropy $H ( W ) \approx 1 . 9 8 5 6$ bits per symbol, while the pragmatic entropy is $H _ { p } ( \underline { { W } } ) \approx 0 . 9 9 2 8$ prabits per symbol. Suppose the communication channel has syntactic capacity $C = 1 . 5$ bits per channel use. Classical source-channel coding would fail because $H ( W ) > C .$ . However, if the pragmatic capacity $C _ { p } \geq 0 . 9 9 2 8$ (which holds for any channel with $C _ { p } \geq C )$ , the pragmatic source-channel coding theorem guarantees reliable transmission of the optimal action (Go/Stop) even though the exact traffic light symbols cannot be reliably transmitted. This demonstrates the power of pragmatic abstraction in overcoming physical communication bottlenecks.

1) Separation Principle and the Imperative of Joint Optimization: The pragmatic sourcechannel coding theorem (Theorem 46) is proved via a separation architecture, which builds on two independent results:

1) The pragmatic lossless source coding theorem (Section VII) guarantees that the pragmatic source can be compressed to rate $H _ { p } ( \underline { { W } } )$ (or $R _ { p } ( D )$ in the lossy case) without losing any pragmatically relevant information.

2) The pragmatic channel coding theorem (Section VIII) guarantees that any rate up to $C _ { p }$ can be reliably transmitted over the noisy channel.

Concatenating the pragmatic source code and the pragmatic channel code yields an end-to-end system that achieves reliable pragmatic transmission if and only if the source rate does not exceed the channel capacity, i.e.,

$$
H _ { p } ( \underline { { W } } ) \leq C _ { p } \quad ( \mathrm { l o s s l e s s } ) \quad \mathrm { o r } \quad R _ { p } ( D ) \leq C _ { p } \quad ( \mathrm { l o s s y } ) .
$$

This constitutes the pragmatic separation principle—the asymptotic bedrock of the theory.

In the classical information-theoretic setting where the blocklength n tends to infinity and no constraints are imposed on encoding/decoding complexity, the separation principle is strictly optimal. It allows the system designer to optimize the pragmatic source code (e.g., feature extraction, action classification) and the pragmatic channel code (e.g., modulation, error correction) independently, while guaranteeing that the concatenated system achieves the fundamental limit. This modularity is a direct consequence of the asymptotic equipartition property and the joint typicality arguments used in the proofs, and it is fully consistent with the backward-compatibility requirement of Principle III in Section II.

The optimality of separation, however, is an asymptotic statement. In practical systems, where the blocklength n is finite, or when the encoder and decoder are subject to computational complexity, delay, or memory constraints, the separation architecture is strictly suboptimal with respect to the end-to-end task utility. In such regimes, the optimal code cannot be factored into independent source and channel components; instead, the encoder must jointly consider the source statistics, the channel characteristics, and the decision utility to minimize the overall distortion or maximize the expected reward. This fundamental gap between asymptotic theory and engineering reality is precisely the motivation for the joint optimization framework that follows.

This observation directly justifies the “joint optimization” framework developed in Sections VI. While the separation principle identifies the ultimate performance bounds (capacities and ratedistortion functions), the optimal allocation of finite resources—such as transmit power, latency, bandwidth, or model size—requires solving the coupled Lagrangian:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { g l o b a l } } ( R ; \lambda ) = \Phi _ { p } ( R ) - \lambda \mathrm { C o I } _ { p } ( R ) , } \end{array}
$$

where the same rate R simultaneously determines the coarsening level of the isoteleia mapping (source-side abstraction) and the error-protection strength (channel-side coding). The cross-layer coupling therefore does not lie in the code structure, which can be separated asymptotically, but in the rate-resource trade-off, which must be jointly optimized to maximize the net benefit. Thus, the separation principle provides the theoretical bedrock, while the Lagrangian framework delivers the practical tool for engineering finite-dimensional, resource-constrained systems.

Remark 23. The pragmatic separation principle generalizes both the classical separation theorem (when the isoteleia and synonymous mappings are trivial) and the semantic separation theorem (when only the synonymous mapping is non-trivial). It unifies all previous results as special cases and offers new insights into the design of task-oriented communication systems, while the accompanying joint optimization framework addresses the real-world constraints that lie beyond the asymptotic ideal.

2) Gaussian Source over Gaussian Channel: To illustrate the joint optimization framework in a concrete setting, we now specialize to the canonical case of a Gaussian source transmitted over an additive white Gaussian noise (AWGN) channel. This example demonstrates how the pragmatic cost of information (CoI) and pragmatic value of information (VoI) jointly determine the optimal operating point of a task-oriented communication system.

Consider a Gaussian source $X \sim { \mathcal { N } } ( 0 , P ) $ with variance P, transmitted over an AWGN channel with noise variance $\sigma ^ { 2 }$ and power constraint $\mathbb { E } [ | X | ^ { 2 } ] \le P _ { \mathrm { t x } }$ . The channel output is $Y = X + Z$ where $Z \sim { \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ . Let the isoteleic volume be $\Omega \geq 1$ , representing the pragmatic abstraction that groups signals leading to the same optimal action.

From Sections X-B and X-C, we have the following closed-form expressions for the continuous Gaussian case:

• Pragmatic channel capacity (per real dimension):

$$
C _ { p } ( P _ { \mathrm { t x } } ) = \frac { 1 } { 2 } \log { \left( \Omega ^ { 4 } \left( 1 + \frac { P _ { \mathrm { t x } } } { \sigma ^ { 2 } } \right) \right) } \mathrm { p r a b i t s ~ p e r ~ c h a n n e l ~ u s e } .\tag{288}
$$

• Bilateral pragmatic cost of information (inverse of capacity):

$$
\mathrm { C o I } _ { p } ( R ) = \sigma ^ { 2 } \left( \frac { 2 ^ { 2 R } } { \Omega ^ { 4 } } - 1 \right) \mathrm { f o r } R \geq 0 .\tag{289}
$$

• Pragmatic rate-distortion function:

$$
R _ { p } ( D ) = \frac { 1 } { 2 } \log \left( \frac { P } { \Omega ^ { 4 } D } \right) , \quad 0 \leq D \leq \frac { P } { \Omega ^ { 4 } } .\tag{290}
$$

• Bilateral pragmatic value of information (for quadratic utility $U ( X , A ) = - ( X - A ) ^ { 2 } )$

$$
\mathrm { V o I } _ { p } ( R ) = P \left( 1 - \frac { 2 ^ { - 2 R } } { \Omega ^ { 4 } } \right) .\tag{291}
$$

The pragmatic source-channel coding theorem (Theorem 46) states that reliable transmission of pragmatic information is possible if and only if

$$
R _ { p } ( D ) \leq C _ { p } ( P _ { \mathrm { t x } } ) .\tag{292}
$$

Substituting the closed-form expressions from (288) and (290), the feasibility condition becomes

$$
\frac { 1 } { 2 } \log { \left( \frac { P } { \Omega ^ { 4 } D } \right) } \leq \frac { 1 } { 2 } \log { \left( \Omega ^ { 4 } \left( 1 + \frac { P _ { \mathrm { t x } } } { \sigma ^ { 2 } } \right) \right) } .\tag{293}
$$

Simplifying, we obtain the equivalent condition:

$$
D \geq \frac { P } { \Omega ^ { 8 } \left( 1 + P _ { \mathrm { t x } } / \sigma ^ { 2 } \right) } .\tag{294}
$$

Here the factor $\Omega ^ { 8 }$ results from the product of the pragmatic gains at both ends: the source– reconstruction pair contributes $\Omega ^ { 4 }$ through the rate–distortion function, and the input–output pair contributes another $\Omega ^ { 4 }$ through the channel capacity, under the assumption of equal isoteleic volumes for both the source and the channel.

This inequality reveals the fundamental trade-off: for a given transmit power $P _ { \mathrm { t x } }$ and pragmatic abstraction Ω, the minimum achievable pragmatic distortion is

$$
D _ { \mathrm { m i n } } ( P _ { \mathrm { t x } } ) = \frac { P } { \Omega ^ { 8 } \left( 1 + P _ { \mathrm { t x } } / \sigma ^ { 2 } \right) } .\tag{295}
$$

Equivalently, for a target distortion D, the minimum required transmit power is

$$
P _ { \mathrm { t x } } ^ { \mathrm { m i n } } ( D ) = \sigma ^ { 2 } \left( \frac { P } { \Omega ^ { 8 } D } - 1 \right) .\tag{296}
$$

The joint feasibility condition can also be expressed directly in terms of VoI and CoI. Recall that the full pragmatic value $\mathrm { V o I } _ { p } ^ { \mathrm { m a x } } = P$ is achieved when $R \geq R _ { p } ( D )$ (i.e., the rate is sufficient to meet the distortion requirement). The required rate to achieve a given VoI level is obtained by inverting (291):

$$
R _ { \mathrm { V o I } } ( V ) = \frac { 1 } { 2 } \log \left( \frac { 1 } { \Omega ^ { 4 } ( 1 - V / P ) } \right) , \quad 0 \leq V \leq P .\tag{297}
$$

The system achieves the target VoI V if and only if the available power satisfies

$$
\operatorname { C o I } _ { p } ( R _ { \mathrm { V o I } } ( V ) ) \leq P _ { \mathrm { t x } } .\tag{298}
$$

Substituting the CoI expression (289), this becomes

$$
\sigma ^ { 2 } \left( { \frac { 2 ^ { 2 R _ { \mathrm { V o I } } ( V ) } } { \Omega ^ { 4 } } } - 1 \right) \leq P _ { \mathrm { t x } } .\tag{299}
$$

Since $2 ^ { 2 R _ { \mathrm { V o I } } ( V ) } = 1 / ( \Omega ^ { 4 } ( 1 - V / P ) )$ , we obtain

$$
\sigma ^ { 2 } \left( { \frac { 1 } { \Omega ^ { 8 } ( 1 - V / P ) } } - 1 \right) \leq P _ { \mathrm { t x } } .\tag{300}
$$

Solving for V , the maximum achievable pragmatic value for a given transmit power is

$$
\mathrm { V o I } _ { p } ^ { \operatorname * { m a x } } ( P _ { \mathrm { t x } } ) = P \left( 1 - \frac { 1 } { \Omega ^ { 8 } ( 1 + P _ { \mathrm { t x } } / \sigma ^ { 2 } ) } \right) .\tag{301}
$$

This expression directly quantifies the value-cost trade-off : the utility gain increases with transmit power but saturates at $P$ as $P _ { \mathrm { t x } } \to \infty$ . The pragmatic abstraction Ω amplifies the effective signal-to-noise ratio, reducing the power required to achieve a given utility level. For example, when Ω = 1 (no pragmatic abstraction), (301) reduces to the classical value-cost relationship:

$$
\mathrm { V o I } ^ { \mathrm { m a x } } ( P _ { \mathrm { t x } } ) = P \left( 1 - { \frac { 1 } { 1 + P _ { \mathrm { t x } } / \sigma ^ { 2 } } } \right) = { \frac { P P _ { \mathrm { t x } } } { P _ { \mathrm { t x } } + \sigma ^ { 2 } } } ,\tag{302}
$$

which is the well-known SNR-dependent utility gain for a Gaussian channel with quadratic utility.

Example 11. Let $P = 1 , \ \sigma ^ { 2 } = 1$ , and consider three values of the isoteleic volume: $\Omega = 1$ (classical), Ω = 1.5, and $\Omega = 2 .$ . For a target VoI of $V = 0 . 9 \ ( i . e . , \ 9 0 \%$ of the maximum utility),

the required transmit powers are:

$$
\Omega = 1 : { \cal P } _ { t x } = \sigma ^ { 2 } \left( \frac { 1 } { \Omega ^ { 8 } ( 1 - V / P ) } - 1 \right) = \frac { 1 } { 0 . 1 } - 1 = 9 \ ( u n i t s ) ,\tag{303}
$$

$$
\Omega = 1 . 5 : P _ { t x } = \frac { 1 } { 1 . 5 ^ { 8 } \cdot 0 . 1 } - 1 \approx \frac { 1 } { 2 5 . 6 3 \cdot 0 . 1 } - 1 \approx 2 . 9 0 ,\tag{304}
$$

$$
\Omega = 2 : \quad P _ { t x } = { \frac { 1 } { 2 ^ { 8 } \cdot 0 . 1 } } - 1 = { \frac { 1 } { 2 5 . 6 } } - 1 \approx - 0 . 9 6 \ ( z e r o \ p o w e r \ s u f f i c e s ) .\tag{305}
$$

With $\Omega = 2 .$ , the pragmatic abstraction alone $( P _ { \mathrm { t x } } = 0 )$ already achieves $\mathrm { V o I } = P ( 1 - 1 / \Omega ^ { 8 } ) =$ $1 - 1 / 2 5 6 \approx 0 . 9 9 6$ , exceeding the target 0.9. This demonstrates that pragmatic abstraction can dramatically reduce or eliminate the need for transmit power in task-oriented communication.

For asymmetric systems where only one side is coarsened, the perceptual and expressive CoI/VoI functions from Section VI-D yield analogous joint feasibility conditions. For the perceptual path, the achievable VoI under power constraint is

$$
\mathrm { V o I } _ { p } ^ { \mathrm { p e r c , \ m a x } } ( P _ { \mathrm { t x } } ) = P \left( 1 - \frac { 1 } { \Omega ^ { 4 } ( 1 + P _ { \mathrm { t x } } / \sigma ^ { 2 } ) } \right) ,\tag{306}
$$

which is strictly less than the bilateral case (301) for any finite $\Omega > 1$ and $P _ { \mathrm { t x } }$ , since the effective SNR amplification is $\Omega ^ { 4 }$ rather than $\Omega ^ { 8 }$ . This quantifies the performance loss when only the source or only the reconstruction is coarsened, confirming the benefits of coarsening both sides of the communication link.

Remark 24. The Gaussian example illustrates the full power of the pragmatic informationtheoretic framework: it provides a unified, analytically tractable model for jointly optimizing communication resources (power), decision performance (distortion/utility), and pragmatic abstraction (isoteleic volume). The closed-form expressions reveal the precise quantitative relationships among these quantities, enabling system designers to make principled trade-offs in task-oriented communication system design.

## B. Communication-Control-Decision Joint Optimization

We now extend the Lagrangian dual framework from the static source-channel coding setting to the general closed-loop system where communication, control, and decision-making are jointly optimized. In many practical systems—such as autonomous vehicles, robotic control, and industrial automation—the communication rate, control policy, and pragmatic abstraction must be co-designed to maximize the overall task performance under resource constraints. We present two formal theorems that characterize the optimal trade-offs in both static (single-step) and dynamic (multi-step) settings, with explicit optimality conditions that bridge information-theoretic limits and decision-theoretic objectives.

1) Static Joint Optimization: Lagrangian Approach and Single-Sided Extensions:

We first consider a single-step decision problem where the system observes a state $S \in S$ takes an action $A \in A$ , and receives a utility $U ( S , A )$ . The observation is obtained through a communication channel that conveys a pragmatic message $\underline { { Y } }$ about the state, with a rate R that determines the quality of the pragmatic information. The receiver chooses a policy $\pi : \underline { { Y } }  A$ that maps the pragmatic observation to an action. The joint optimization problem is to maximize the expected utility subject to a communication cost constraint.

Theorem 47 (Static Joint Optimization). For a single-step decision problem with state S, action A, and utility $U ( S , A )$ , let the pragmatic observation Y be obtained at rate R satisfying $I _ { p } ( { \underline { { S } } } ; { \underline { { Y } } } ) \leq R$ and let the communication cost be given by the pragmatic cost of information $\operatorname { C o I } _ { p } ( R )$ . The Lagrangian relaxation of the constrained utility maximization problem

$$
\operatorname* { m a x } _ { \pi , P ( \underline { { Y } } | S ) } \ \mathbb { E } _ { S , \underline { { Y } } } [ U ( S , \pi ( \underline { { Y } } ) ) ] \quad s u b j e c t \ t o \quad \operatorname { C o I } _ { p } ( R ) \leq P _ { \operatorname* { m a x } }\tag{307}
$$

yields the global pragmatic Lagrangian

$$
{ \mathcal L } _ { \mathrm { g l o b a l } } ( \pi , R ; \lambda ) = \mathbb { E } _ { S , \underline { { Y } } } [ U ( S , \pi ( \underline { { Y } } ) ) ] - \lambda { \mathrm { C o I } } _ { p } ( R ) ,\tag{308}
$$

where $\lambda > 0$ is the Lagrange multiplier (shadow price of communication resources). The optimal policy $\pi ^ { * }$ and optimal rate $R ^ { * }$ satisfy the following necessary and sufficient conditions:

$$
\pi ^ { * } ( \underline { { Y } } ) = \arg \operatorname* { m a x } _ { a } \mathbb { E } _ { S | \underline { { Y } } } [ U ( S , a ) ] ,\tag{309}
$$

$$
\lambda \operatorname { C o I } _ { p } ^ { \prime } ( R ^ { * } ) = \frac { \partial } { \partial R } \mathbb { E } _ { S , \underline { { Y } } } [ U ( S , \pi ^ { * } ( \underline { { Y } } ) ) ] ,\tag{310}
$$

where the derivative in (310) is understood in the sense of marginal utility gain with respect to the rate. Moreover, the value of the Lagrangian at the optimum equals the maximum net benefit, that is behavioral capacity:

$$
\operatorname* { m a x } _ { \pi , R } { \mathcal L } _ { \mathrm { g l o b a l } } ( \pi , R ; \lambda ) = \operatorname* { m a x } _ { P ( \underline { { Y } } | S ) } \left\{ \mathbb { E } _ { S , \underline { { Y } } } [ U ( S , a ^ { * } ( \underline { { Y } } ) ) ] - \lambda \operatorname { C o I } _ { p } ( R ) \right\} ,\tag{311}
$$

with $a ^ { * } ( \underline { { Y } } ) = \arg \operatorname* { m a x } _ { a } \mathbb { E } _ { S | \underline { { Y } } } [ U ( S , a ) ]$

Sketch of Proof: The Lagrangian is concave in π and R under mild regularity conditions (e.g., U bounded, $\mathrm { C o I } _ { p }$ convex). The first-order conditions follow from the concavity and the fact that the optimal policy is the Bayes decision rule. The marginal utility gain in (310) is obtained by differentiating the optimal value with respect to R, which equals the rate of improvement of the expected utility when the constraint on mutual information is relaxed. The global optimum is achieved when the marginal benefit of increasing the rate equals its marginal cost, scaled by the shadow price. □

Corollary 10 (Single-Sided Lagrangian Extensions). For asymmetric systems where only one side of the communication link is coarsened, the corresponding single-sided Lagrangians are obtained by substituting the appropriate CoI functions:

• Perceptual pragmatic Lagrangian: ${ \mathcal { L } } _ { \mathrm { p e r c } } ( \pi , R ; \lambda ) = \mathbb { E } _ { S , \underline { { Y } } } [ U ( S , \pi ( \underline { { Y } } ) ) ] - \lambda \mathrm { C o I } _ { p } ^ { p e r c } ( R )$ , where $\mathrm { C o l } _ { p } ^ { p e r c } ( R )$ is the perceptual cost (sensing power for rate R).

• Expressive pragmatic Lagrangian: ${ \mathcal { L } } _ { \mathrm { e x p r } } ( \pi , R ; \lambda ) = \mathbb { E } _ { S , \underline { { Y } } } [ U ( S , \pi ( \underline { { Y } } ) ) ] - \lambda \mathrm { C o I } _ { p } ^ { e x p r } ( R )$ , where $\mathrm { C o I } _ { p } ^ { e x p r } ( R )$ is the expressive cost (transmit power for rate R).

The optimality conditions for each case are identical to (309) and (310) with the corresponding CoI function.

Example 12 (Linear Quadratic Gaussian (LQG) with Perceptual Cost). Consider a scalar state $S \sim \mathcal { N } ( 0 , \sigma _ { s } ^ { 2 } )$ , quadratic utility $U ( S , A ) = - ( S - A ) ^ { 2 }$ , and perceptual cost ${ \mathrm { C o I } } _ { p } ^ { p e r c } ( R ) =$ $\sigma ^ { 2 } ( 2 ^ { 2 R } / \Omega ^ { 2 } - 1 )$ with isoteleic volume Ω. The Lagrangian $\mathcal { L } _ { \mathrm { p e r c } } ( P _ { t x } ; \lambda ) = - { \sigma _ { s } ^ { 2 } } / { ( 1 + \Omega ^ { 2 } P _ { t x } / \sigma ^ { 2 } ) } -$ $\lambda P _ { t x }$ yields the optimal power

$$
P _ { t x } ^ { * } = \operatorname* { m a x } \left\{ \sigma ^ { 2 } \left( \frac { \sqrt { \sigma _ { s } ^ { 2 } } } { \Omega \sqrt { \lambda } } - 1 \right) , 0 \right\} .\tag{312}
$$

This closed-form solution illustrates the trade-off between control performance and communication cost: the optimal power decreases as the shadow price λ increases, and the pragmatic abstraction Ω amplifies the effective signal-to-noise ratio.

Example 13 (Gaussian Source over Gaussian Channel: Closed-Form Bilateral Pragmatic Lagrangian). We now instantiate Theorem 47 for the canonical case of a Gaussian source transmitted over an AWGN channel, where both the source and the observation are coarsened pragmatically (bilateral abstraction). This provides a closed-form verification of the static joint

optimization conditions.

Let the state (source) be $X \sim { \mathcal { N } } ( 0 , P ) $ and the pragmatic observation be Y obtained through an AWGN channel with noise variance $\sigma ^ { 2 }$ and transmit power $P _ { \mathrm { t x } }$ . The isoteleic volume $\Omega \geq 1$ captures the pragmatic equivalence. From Section XI-A2, the bilateral pragmatic value and cost functions are (cf. (291) and (289)):

$$
\Phi _ { p } ( R ) = P \left( 1 - \frac { 2 ^ { - 2 R } } { \Omega ^ { 4 } } \right) ,\tag{313}
$$

$$
\mathrm { C o I } _ { p } ( R ) = \sigma ^ { 2 } \left( \frac { 2 ^ { 2 R } } { \Omega ^ { 4 } } - 1 \right) ,\tag{314}
$$

where R is the pragmatic rate (prabits per channel use). Substituting into the global Lagrangian of Theorem 47 (after maximizing over the policy π, which yields $\Phi _ { p } ( R ) )$ , we obtain the explicit Lagrangian:

$$
{ \mathcal { L } } ( R ) = \Phi _ { p } ( R ) - \lambda \operatorname { C o I } _ { p } ( R ) = P \left( 1 - { \frac { 2 ^ { - 2 R } } { \Omega ^ { 4 } } } \right) - \lambda \sigma ^ { 2 } \left( { \frac { 2 ^ { 2 R } } { \Omega ^ { 4 } } } - 1 \right) .\tag{315}
$$

Equivalently, symmetrically:

$$
{ \mathcal { L } } ( R ) = ( P + \lambda \sigma ^ { 2 } ) - { \frac { 1 } { \Omega ^ { 4 } } } \left( P \cdot 2 ^ { - 2 R } + \lambda \sigma ^ { 2 } \cdot 2 ^ { 2 R } \right) .\tag{316}
$$

![](images/53c600a0ab9115ed7340be25edcf1f9a5bf5f09fd0175a49d388daaaea650b38.jpg)  
(a) Net benefit maximization: $\Phi _ { p } ( R )$ $\lambda { \mathrm { C o I } } _ { p } ( R )$ , and ${ \mathcal { L } } ( R ) .$ Parameters: $\Omega = 2 , \lambda = 0 . 5 , P / \sigma ^ { 2 } = 1$

![](images/7d95a9824f7ddbdda4570b041cd3ee657bb9ffa26e7d32332092416554207624.jpg)  
(b) Comparison of $\mathcal { L } ( R )$ for different Ω. Parameters: $P / \sigma ^ { 2 } = 1 , \lambda = 0 . 5 .$  
Fig. 17: Global pragmatic Lagrangian and the effect of pragmatic abstraction.

Figure 17 illustrates the global Lagrangian structure. Subfigure (a) shows the pragmatic value, the weighted cost, and the resulting net benefit, with the optimal rate $R ^ { * }$ marked where marginal value equals marginal cost. Subfigure (b) demonstrates that increasing Ω uniformly shifts the net benefit curve upward while leaving $R ^ { * }$ unchanged, confirming that pragmatic abstraction amplifies the effective signal-to-noise ratio without altering the optimal communication rate.

Applying the first-order optimality condition from Theorem 47 $( \mathcal { L } ^ { \prime } ( R ^ { * } ) = 0 )$

$$
\frac { d \mathcal { L } } { d R } = \frac { 2 \ln 2 } { \Omega ^ { 4 } } \left( P \cdot 2 ^ { - 2 R ^ { * } } - \lambda \sigma ^ { 2 } \cdot 2 ^ { 2 R ^ { * } } \right) = 0 ,\tag{317}
$$

we obtain the optimal rate:

$$
R ^ { * } = \frac { 1 } { 4 } \log _ { 2 } \left( \frac { P } { \lambda \sigma ^ { 2 } } \right) ^ { + } ,\tag{318}
$$

where $( x ) ^ { + } = \operatorname* { m a x } \{ x , 0 \}$ ensures non-negativity. The maximized Lagrangian (net benefit) is then:

$$
\mathcal L ^ { * } = \mathcal L ( R ^ { * } ) = P + \lambda \sigma ^ { 2 } - \frac { 2 \sqrt { P \lambda \sigma ^ { 2 } } } { \Omega ^ { 4 } } ,\tag{319}
$$

provided $P > \lambda \sigma ^ { 2 }$ ; otherwise $R ^ { * } = 0$ and $\mathcal { L } ^ { * } = 0$ (the system opts not to communicate).

This maximized net benefit is precisely the behavioral capacity $\mathcal { E } _ { p } ( \lambda )$ defined in Definition 24. Substituting $\lambda = ( P / \sigma ^ { 2 } ) 2 ^ { - 4 R ^ { * } }$ into (319) yields the behavioral capacity as a function of the optimal rate:

$$
\mathcal { E } _ { p } ( R ^ { * } ) = P \left( 1 + 2 ^ { - 4 R ^ { * } } - \frac { 2 \cdot 2 ^ { - 2 R ^ { * } } } { \Omega ^ { 4 } } \right) , \quad R ^ { * } \geq 0 .\tag{320}
$$

Equation (319) and (320) explicitly characterize the behavioral capacity in terms of the shadow price and the optimal communication rate, respectively.

Figure 18 further characterizes the behavioral capacity. Subfigure (a) shows how the capacity increases with the invested communication rate, with diminishing returns and saturation at the perfect-information utility P. A larger Ω shifts the curve upward, indicating that the same rate yields higher net utility under stronger pragmatic abstraction. Subfigure (b) reveals the decay of capacity as resources become more expensive; the zero-capacity threshold marks the point where communication ceases. Larger Ω flattens the decline, demonstrating that abstraction provides robustness against rising resource costs.

This closed-form solution explicitly confirms the two optimality conditions of Theorem 47:

1) The optimal policy $\pi ^ { * }$ is the Bayes decision rule (implicit in the definition of $\Phi _ { p } ( R ) )$ ;

2) The marginal value equals the marginal cost: $\Phi _ { p } ^ { \prime } ( R ^ { * } ) = \lambda \mathrm { C o I } _ { p } ^ { \prime } ( R ^ { * } )$ , which is precisely the balance expressed in (317).

![](images/a5f75e0e542c577bd5bcdaad33014c9cfb16221915daa9727b4af4a2280b945f.jpg)  
(a) Behavioral capacity $\mathcal { E } _ { p }$ vs. optimal pragmatic rate $R ^ { * }$ . Larger Ω yields higher capacity, saturating at $P .$ Parameters: $P = 1 , \sigma ^ { 2 } = 1$

![](images/733d032629e69b893df3277ef71245391f75bfa707d7bc8c05b44ff2786e4a13.jpg)  
(b) Behavioral capacity $\mathcal { E } _ { p }$ vs. shadow price λ. Capacity decreases with λ and vanishes at $\lambda = P / \sigma ^ { 2 }$ . Larger Ω reduces sensitivity. Parameters: $P = 1 , \sigma ^ { 2 } = 1$  
Fig. 18: Behavioral capacity as a function of optimal rate and resource shadow price.

Moreover, the expression (319) reveals the four-way trade-off among source power $P ,$ , channel noise $\sigma ^ { 2 }$ , resource shadow price $\lambda ,$ and pragmatic abstraction $\Omega .$ When $\Omega = 1$ , it recovers the classical Gaussian benchmark; when $\Omega > 1$ , the effective SNR is amplified by $\Omega ^ { 4 }$ on both sides, drastically reducing the power needed to achieve a given net benefit.

## 2) Dynamic Joint Optimization: Sequential Decision Making:

In dynamic systems, the state evolves over time according to a controlled Markov process, and the decision-maker must balance immediate rewards against future consequences. The system is modeled as a Partially Observable Markov Decision Process (POMDP) with communication constraints. At each time step $t ,$ the system state $S _ { t }$ evolves as $S _ { t + 1 } \sim P ( \cdot | S _ { t } , A _ { t } )$ , and the decision-maker receives a pragmatic observation $\underline { { Y } } _ { t }$ about $S _ { t }$ at a rate $R _ { t }$ that incurs a cost $\mathrm { C o I } _ { p } ( R _ { t } )$ . The goal is to find a policy $\pi = \{ \pi _ { t } \} _ { t \geq 0 }$ mapping the history of pragmatic observations to actions, maximizing the expected discounted sum of utilities subject to a total communication resource budget.

Theorem 48 (Dynamic Joint Optimization for POMDP). For an infinite-horizon discounted POMDP with state space ${ \mathcal { S } } ,$ action space ${ \mathcal A } ,$ observation space ${ \mathcal { V } } ,$ transition kernel $P ( s ^ { \prime } | s , a )$

observation kernel $P ( \boldsymbol { y } | \boldsymbol { s } ^ { \prime } , R )$ (where the communication rate R affects the observation quality), utility $U ( s , a )$ , discount factor $\gamma \in ( 0 , 1 )$ , and pragmatic communication cost $\operatorname { C o I } _ { p } ( R )$ per step, the Lagrangian relaxation of the constrained expected discounted utility maximization

$$
\operatorname* { m a x } _ { \pi , \{ R _ { t } \} } \ \mathbb { E } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } U ( S _ { t } , A _ { t } ) \right] \quad s u b j e c t \ t o \quad \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } \mathrm { C o I } _ { p } ( R _ { t } ) \le C _ { t o t a l }\tag{321}
$$

yields the dynamic pragmatic Lagrangian

$$
\mathcal { L } _ { \mathrm { d y n } } ( \pi , \{ R _ { t } \} ; \lambda ) = \mathbb { E } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } \left( U ( S _ { t } , A _ { t } ) - \lambda \mathrm { C o l } _ { p } ( R _ { t } ) \right) \right] ,\tag{322}
$$

where $\lambda > 0$ is the Lagrange multiplier for the total resource constraint.

Let $b \in \Delta ( S )$ denote the belief state (the posterior distribution over the state given the history of actions and observations). The optimal policy and rate schedule satisfy the Bellman equation [5] in belief-state form:

$$
V ( b ) = \operatorname* { m a x } _ { a , R \geq 0 } \left\{ \sum _ { s } b ( s ) U ( s , a ) - \lambda \operatorname { C o I } _ { p } ( R ) + \gamma \sum _ { s , s ^ { \prime } , y } b ( s ) P ( s ^ { \prime } | s , a ) P ( y | s ^ { \prime } , R ) V ( b _ { y } ^ { \prime } ) \right\} ,\tag{323}
$$

where $b _ { y } ^ { \prime }$ is the updated belief state after observing y (and taking action a with rate $R )$ :

$$
b _ { y } ^ { \prime } ( s ^ { \prime } ) = \frac { P ( y | s ^ { \prime } , R ) \sum _ { s } P ( s ^ { \prime } | s , a ) b ( s ) } { \sum _ { \tilde { s } } P ( y | \tilde { s } , R ) \sum _ { s } P ( \tilde { s } | s , a ) b ( s ) } .\tag{324}
$$

The optimal action $a ^ { * }$ and optimal rate $R ^ { * }$ at belief b are characterized by the following first-order conditions:

$$
a ^ { * } = \arg \operatorname* { m a x } _ { a } \left\{ \sum _ { s } b ( s ) U ( s , a ) + \gamma \sum _ { s } b ( s ) \sum _ { s ^ { \prime } } P ( s ^ { \prime } | s , a ) \sum _ { y } P ( y | s ^ { \prime } , R ^ { * } ) V ( b _ { y } ^ { \prime } ) \right\} ,\tag{325}
$$

$$
\lambda \mathrm { C o I } _ { p } ^ { \prime } ( R ^ { * } ) = \frac { \partial } { \partial R } \left( \gamma \sum _ { s } b ( s ) \sum _ { s ^ { \prime } } P ( s ^ { \prime } | s , a ^ { * } ) \sum _ { y } P ( y | s ^ { \prime } , R ) V ( b _ { y } ^ { \prime } ) \right) \Bigg | _ { R = R ^ { * } } ,\tag{326}
$$

where $b _ { y } ^ { \prime }$ is given by (324) and depends on $a ^ { * }$ and R. The derivative in (326) accounts for the improvement in future value due to better state estimation (sharper belief) resulting from a higher rate R.

Sketch of Proof: The proof follows from the principle of optimality applied to the belief-state MDP, which is equivalent to the original POMDP. Before applying the Bellman equation, we

first establish its well-definedness. Define the Bellman operator $\tau$ on the space of bounded continuous functions on $\Delta ( S )$ by

$$
( T V ) ( b ) \triangleq \operatorname* { m a x } _ { a , R \geq 0 } \left\{ \sum _ { s } b ( s ) U ( s , a ) - \lambda { \mathrm { C o I } } _ { p } ( R ) + \gamma \sum _ { s , s ^ { \prime } , y } b ( s ) P ( s ^ { \prime } | s , a ) P ( y | s ^ { \prime } , R ) V ( b _ { y } ^ { \prime } ) \right\} .
$$

For any two bounded functions $V _ { 1 } , V _ { 2 }$ and any belief $b ,$ the transition kernels sum to 1, which gives

$$
| ( T V _ { 1 } ) ( b ) - ( T V _ { 2 } ) ( b ) | \leq \gamma \operatorname* { s u p } _ { a , R } \sum _ { s , s ^ { \prime } , y } b ( s ) P ( s ^ { \prime } | s , a ) P ( y | s ^ { \prime } , R ) | V _ { 1 } ( b _ { y } ^ { \prime } ) - V _ { 2 } ( b _ { y } ^ { \prime } ) | \leq \gamma \| V _ { 1 } - V _ { 2 } \| _ { \infty } .
$$

Taking the supremum over b yields $\| T V _ { 1 } - T V _ { 2 } \| _ { \infty } \leq \gamma \| V _ { 1 } - V _ { 2 } \| _ { \infty }$ . Since $\gamma < 1 , \tau$ is a contraction. $\mathtt { B y }$ the Banach fixed-point theorem, there exists a unique fixed point $V ^ { * }$ in the space of bounded continuous functions, ensuring that the belief-state Bellman equation (323) is well-defined. The dynamic isoteleia mapping $e _ { V ^ { \ast } }$ defined through $V ^ { * }$ is then uniquely determined and recovers the static mapping e when $\gamma = 0$

With this established, the augmented reward process $U ( S _ { t } , A _ { t } ) - \lambda { \mathrm { C o I } } _ { p } ( R _ { t } )$ leads to the belief-state Bellman equation (323), where the expectation over the next belief accounts for both state transition and observation likelihood. The optimality conditions (325) and (326) are obtained by differentiating the right-hand side of (323) with respect to $a$ and $R ,$ respectively, under suitable regularity conditions (e.g., concavity of the value function and convexity of $\mathrm { C o I } _ { p } )$ The coupling between action and rate appears through the belief update, as both influence the future belief distribution. □

To avoid a conceptual gap between the static hierarchical framework (Section II–IX) and the dynamic POMDP formulation, we emphasize that the pragmatic variable S appearing in the Bellman equation (324) is induced by the dynamic isoteleia mapping $e _ { V }$ defined in Definition 3. The resulting pragmatic equivalence classes are therefore functions of the future value function $V .$ In the steady-state case, $e _ { V }$ collapses to the static mapping e used in the coding theorems, ensuring that the hierarchical entropy inequalities remain valid as limiting bounds.

Remark 25 (Interpretation and Design Insights). The dynamic Lagrangian framework provides several key insights for system design:

1) Adaptive Resource Allocation: The optimal rate $R ^ { * }$ varies with the belief state, allocating more resources when the belief is diffuse (high uncertainty) $o r$ when future rewards are particularly sensitive to estimation accuracy. This yields a value-of-information-guided scheduling policy.

2) Belief-Dependent Optimization: Unlike the original erroneous formulation, the correct belief-state Bellman equation explicitly incorporates the observation model and the impact of R on the quality of future beliefs. The action and rate decisions are coupled through the belief update, reflecting the inherent trade-off between control and sensing/communication.

3) Connection to CoI and VoI: The term $\lambda { \mathrm { C o I } } _ { p } ( R )$ represents the instantaneous communication cost, while the future value $V ( b _ { y } ^ { \prime } )$ encodes the long-term utility gain from improved state information. The optimal policy balances immediate andfuture value, mirroring the economic principle of marginal benefit equals marginal cost.

Example 14 (Linear Quadratic Gaussian (LQG) with Linear Communication Cost). Consider a scalar linear system $S _ { t + 1 } = a S _ { t } + b A _ { t } + W _ { t }$ with $W _ { t } \sim \mathcal { N } ( 0 , \sigma _ { w } ^ { 2 } )$ , quadratic utility ${ \cal U } ( S _ { t } , A _ { t } ) =$ $- q S _ { t } ^ { 2 } - r A _ { t } ^ { 2 }$ , and communication cost $\mathrm { C o I } _ { p } ( R _ { t } ) = c R _ { t }$ (linear in rate). Assume Gaussian observations with precision proportional to $R _ { t } ,$ i.e., $Y _ { t } = S _ { t } + Z _ { t } , Z _ { t } \sim { \mathcal { N } } ( 0 , \sigma ^ { 2 } / R _ { t } )$ . The belief state is Gaussian, characterized by its mean and variance. The optimal control law remains linear in the estimated state (separation principle holds for LQG with quadratic cost and Gaussian noise). The optimal rate $R ^ { * }$ is constant over time when the system is stationary, and can be found by solving a Riccati equation augmented with the rate penalty. The steady-state rate decreases with the shadow price λ and depends on the system dynamics through the controllability and observability parameters. This example demonstrates how the dynamic Lagrangian framework provides a tractable method for co-designing communication and control in practical systems.

The dynamic joint optimization framework completes the pragmatic information-theoretic spectrum, providing a unified mathematical tool for designing task-oriented communication and control systems that optimally trade off information value, physical resource consumption, and long-term performance.

## XII. CONCLUSIONS

## A. Summary of Contributions

We have established a mathematical theory of pragmatic information that unifies communication, control, and decision-making within a single coherent framework. The theory is built upon the isoteleia mapping $e : \underline { { \mathcal { W } } } \to 2 ^ { \tilde { \mathcal { W } } }$ , which formalizes equifinality—the principle that distinct semantic paths converging to the same optimal action are pragmatically equivalent. This induces a three-tier hierarchy of syntactic, semantic, and pragmatic information, with each successive abstraction discarding task-irrelevant distinctions.

The main contributions are threefold. First, we have developed a complete set of informationtheoretic measures at the pragmatic level—entropy $H _ { p } ( \underline { { W } } )$ , up/down mutual information $I ^ { p } ( \underline { { X } } ; \underline { { Y } } )$ and $I _ { p } ( \underline { { X } } ; \underline { { Y } } )$ , channel capacity $C _ { p } ,$ and rate-distortion function $R _ { p } ( D )$ —and established the fundamental hierarchies $H _ { p } \leq H _ { s } \leq H , C _ { p } \geq C _ { s } \geq C$ , and $R _ { p } \leq R _ { s } \leq R$ . Second, we have introduced pragmatic value of information (VoI) and pragmatic cost of information (CoI) as decision-theoretic and economic duals to rate-distortion and capacity, respectively, and formulated a Lagrangian dual framework for cross-layer optimization. The pragmatic efficiency functional $\begin{array} { r } { \mathcal { E } _ { p } ( \lambda ) = \operatorname* { s u p } _ { R } [ \Phi _ { p } ( R ) - \lambda \operatorname { C o I } _ { p } ( R ) ] } \end{array}$ defines the ultimate behavioral capacity of any resourceconstrained intelligent system. Third, we have proved three coding theorems—lossless source coding, channel coding, and rate-distortion coding—that generalize Shannon’s classical results and their semantic counterparts. We have further extended the framework to continuous messages, deriving closed-form Gaussian expressions, and formulated joint optimization of communication, control, and decision-making in both static and dynamic settings.

## B. Engineering Implications and Deployment Pathways

The pragmatic information theory developed in this paper has profound implications for a wide range of engineering disciplines and applications. By shifting the focus from symbol fidelity to the effectiveness of information in guiding actions, the framework provides a rigorous mathematical language for task-oriented communication, networked control systems, autonomous systems, embodied AI, human-machine interaction, and social and cognitive systems. The Value of Information (VoI) and Cost of Information (CoI) offer a unified economic and decisiontheoretic basis for resource allocation, enabling system designers to balance utility gains against physical resource consumption. The Lagrangian dual framework provides a principled approach for cross-layer optimization, bridging the gap between information-theoretic limits and engineering practice. The framework also offers new perspectives on emerging challenges such as semantic communication, edge AI, intelligent transportation, and human learning, where information must be evaluated by its effectiveness in achieving goals rather than its syntactic accuracy.

Looking toward engineering practice, a critical path for deploying the proposed theory lies in the realization of the isoteleia mapping—the core mathematical object that groups semantically distinct signals into pragmatically equivalent action classes. In real-world scenarios where the utility function U is not explicitly specified, the isoteleia mapping can be acquired implicitly through large-scale data-driven learning. Specifically, deep neural networks can serve as universal approximators of the optimal action function, effectively inducing the equivalence relation from data without requiring an analytic utility model. This paradigm aligns naturally with inverse reinforcement learning and reinforcement learning from human feedback (RLHF), where the network learns to map high-dimensional observations (semantic inputs) to optimal action classes (pragmatic outputs) by minimizing task-oriented loss functions.

Furthermore, in practical deployment, the isoteleia mapping should not be a fixed, static structure. It must adapt dynamically to changing environmental contexts and evolving task objectives. For instance, in autonomous driving, the mapping from a visual scene to the action “brake” may depend critically on the current speed, road condition, and surrounding traffic. This context-dependent behavior suggests an extension of the isoteleia mapping to a conditional form depending on the contextual state or scene encoding. In terms of neural implementation, this can be achieved through context-gating mechanisms or conditional neural networks that modulate the mapping based on the current environment.

Equally important is the principle of determinism in decision-making. For safety-critical systems, the mapping from observation to action should be deterministic: a given input must produce a unique, consistent output. This ensures reproducibility, facilitates fault diagnosis, and aligns with the arg max operation in the decision axiom. In practice, this can be realized by deploying deterministic policies at inference time, or by using techniques such as Gumbel-Softmax with temperature annealing to ensure that the learned mapping converges to a hard, deterministic decision boundary.

To concretize these ideas into a deployable pipeline, we envision an end-to-end pragmatic information system consisting of three functional stages: (i) an offline training phase, where a deep neural network is trained on large-scale task-specific data to approximate the conditional isoteleia mapping; (ii) a context-adaptive inference phase, where the network dynamically adjusts its decision boundaries based on real-time contextual inputs; and (iii) an online adaptation mechanism that continuously refines the mapping through reinforcement learning or online fine-tuning, enabling the system to respond to non-stationary environments. This pipeline directly addresses the practical challenges of implementing the theory in real-world applications, providing a bridge between the mathematical abstraction of the isoteleia mapping and the constraints of physical hardware and real-time computation.

## C. Limitations and Future Directions

Several limitations of the current framework should be acknowledged. The theory assumes stationarity and ergodicity, which may not hold in social, economic, or biological systems where statistical properties evolve over time. The coding theorems are asymptotic in nature; finiteblocklength analyses and the corresponding dispersion bounds remain to be developed. The framework primarily addresses single-agent or centralized settings; multi-agent systems with conflicting or adversarial utilities require game-theoretic extensions. The dynamic optimization relies on belief-state Bellman equations that are computationally intractable in high dimensions, necessitating approximate solution methods. Finally, while deep neural networks offer a practical path for learning isoteleia mappings, they introduce approximation errors and generalization gaps not captured by the current theoretical guarantees.

Looking forward, several directions merit further investigation. The extension to non-stationary and non-ergodic processes calls for time-varying entropies and online adaptive coding strategies. Finite-blocklength analysis would provide dispersion bounds and practical guidelines for system design. Multi-agent and networked pragmatic information systems—including collaborative autonomous fleets, adversarial communication, and competitive economic interactions—require the development of game-theoretic equilibrium concepts and incentive-compatible protocols. The integration of deep learning with pragmatic coding demands provably efficient algorithms with finite-sample guarantees and convergence rates. Approximate dynamic programming techniques, such as deep reinforcement learning and function approximation, can be brought to bear on the high-dimensional belief-state optimization problem. Beyond these technical directions, the framework opens avenues for cross-disciplinary applications in cognitive science, linguistics, and economics, where information exchange is inherently goal-directed and context-dependent.

We believe that the mathematical theory of pragmatic information will open a new chapter in information theory, providing a unified language for the design of next-generation intelligent systems that operate not merely as symbol transmitters but as goal-directed agents that extract value from information under resource constraints.

## REFERENCES

[1] C. E. Shannon, “A Mathematical Theory of Communication,” The Bell System Technical Journal, vol. 27, pp. 379-423, 623-656, July, Oct., 1948.

[2] C. E. Shannon and W. Weaver, The Mathematical Theory of Communication, The University of Illinois Press, 1949.

[3] W. Weaver, “Recent Contributions to the Mathematical Theory of Communication,” ETC: A Review of General Semantics, pp. 261-81, 1953.

[4] N. Wiener, Cybernetics: Or Control and Communication in the Animal and the Machine. Cambridge, MA, USA: The MIT Press, 1948.

[5] R. Bellman, “The Theory of Dynamic Programming,” Bulletin of the American Mathematical Society, vol. 60, pp. 503–515, 1954.

[6] H. S. Tsien, Engineering Cybernetics. Englewood Cliffs, NJ, USA: Prentice-Hall, 1954.

[7] R. E. Kalman, “A New Approach to Linear Filtering and Prediction Problems,” Journal of Basic Engineering, Transactions of the ASME, vol. 82, no. 1, pp. 35–45, Mar. 1960.

[8] R. L. Stratonovich, “On value of information,” Izvestiya of USSR Academy of Sciences, Technical Cybernetics, vol. 5, no. 1, pp. 3-12, 1965.

[9] R. L. Stratonovich, Theory of Information and its Value. Cham, Switzerland: Springer, 2020.

[10] R. Carnap, Y. Bar-Hillel, “An Outline of A Theory of Semantic Information,” RLE Technical Reports 247, Research Laboratory of Electronics, Massachusetts Institute of Technology, Cambridge MA,1952.

[11] L. Floridi, “Outline of A Theory of Strongly Semantic Information,” Minds and machines, vol. 14, no. 2, pp. 197-221, 2004.

[12] J. Bao, P. Basu, M. Dean, et al., “Towards A Theory of Semantic Communication,” IEEE Network Science Workshop, West Point, NY, USA, Jun. 2011.

[13] A. D. Luca, S. Termini, “A Definition of A Non-probabilistic Entropy In The Setting of Fuzzy Sets,” Information and Control, vol. 20, pp. 301-312, 1972.

[14] A. D. Luca, S. Termini, “Entropy of L-Fuzzy Sets,” Information and Control, vol. 24, pp. 55-73, 1974.

[15] W. Wu, “General Source and General Entropy,” Journal of Beijing University of Posts and Telecommunications, vol. 5, no. 1, pp. 29-41, 1982.

[16] Y. Zhong, “Comprehensive measure of information”, Journal of Beijing University of Posts and Telecommunications, vol. 9, no. 2, pp. 12–19, 1986.

[17] ——, “Intelligence oriented comprehensive information theory——In memory of the 50th anniversary of Shannon information theory”, Journal of Beijing University of Posts and Telecommunications, vol. 21, no. 4, pp. 1–6, 1998.

[18] C. Lu, “Coding significance of generalized entropy and generalized mutual information”, Journal of China Institute of Communications, vol. 15, no. 6, pp. 37–44, 1994.

[19] ——, “A semantic generalization of Shannon’s information theory and applications,” Entropy, vol. 27, no. 5, p. 461, 2025.

[20] J. Liu, W. Zhang, and H. V. Poor, “A Rate-Distortion Framework for Characterizing Semantic Information,” 2021 IEEE International Symposium on Information Theory (ISIT), Melbourne, Australia,2021.

[21] K. Niu and P. Zhang, “A mathematical theory of semantic communication.” Journal on Communications, Vol. 45, No. 6, pp. 7-59, July 2024. https://www.joconline.com.cn/en/ article/doi/10.11959/j.issn.1000-436x.2024111/

[22] ——, The Mathematical Theory of Semantic Communication, Springer Nature, 2025. https: //link.springer.com/book/10.1007/978-981-96-5132-0

[23] P. Zhang , K. Niu, Z. Liang et al. “Beyond shannon: Semantic information theory and methodology,” IEEE Transactions on Network Science and Engineering, vol. 13, pp. 8062- 8079, 2026.

[24] C. F. von Weizsacker and E. von Weizs ¨ acker, “Wiederaufnahme der begrifflichen Frage:¨ Was ist Information?,” Nova Acta Leopoldina, vol. 37, no. 206, pp. 535–555, 1972.

[25] D. Gernert, “Pragmatic information: Historical exposition and general overview,” Mind and Matter, vol. 4, no. 2, pp. 141–167, 2006.

[26] E. D. Weinberger, “A theory of pragmatic information and its application to the quasi-species model of biological evolution,” BioSystems, vol. 66, no. 3, pp. 105–119, Aug. 2002.

[27] E. D. Weinberger, “Towards a theory of pragmatic information,” arXiv preprint, arXiv:2403.12324, 2024.

[28] T. Guo, Y. Wang, et al., “Semantic Compression with Side Information: A Rate-Distortion

Perspective,” arXiv preprint arXiv:2208.06094, 2022.

[29] Y. Shao, Q. Cao, and D. Gunduz, “A Theory of Semantic Communication,” arXiv preprint arXiv:2212.01485, 2022.

[30] J. Tang, Q. Yang, and Z. Zhang, “Information-Theoretic Limits on Compression of Semantic Information,” arXiv preprint arXiv:2306.02305, 2023.

[31] P. Zhang, W. Xu, H. Gao, et al., “Toward Wisdom-Evolutionary and Primitive-Concise 6G: A New Paradigm of Semantic Communication Networks,” Engineering, vol. 8, no. 1, pp. 60-73, 2022.

[32] G. Shi, Y. Xiao, Y. Li, et al., “From Semantic Communication to Semantic-aware Networking: Model, Architecture, and Open Problems,” IEEE Communications Magazine, vol. 59, no. 8, pp. 44-50, 2021.

[33] Z. Qin, X. Tao, J. Lu, et al., “Semantic Communications: Principles and Challenges,” arXiv preprint arXiv:2201.01389, 2021.

[34] H. Xie, Z. Qin, X. Tao, et al., “Task-oriented Multi-user Semantic Communications,” IEEE Journal on Selected Areas in Communications, vol. 40, no. 9, pp. 2584-2597, 2022.

[35] D. Gud¨ uz, Z. Qin, et al., “Beyond Transmitting Bits: Context, Semantics, and Task-oriented¨ Communications,” IEEE Journal on Selected Areas in Communications, vol. 41, no. 1, pp. 5-41, Jan. 2023.

[36] K. Niu, J. Dai, S. Yao, et al., “A paradigm shift towards semantic communications,” IEEE Communications Magazine, vol. 60, no. 11, pp. 113-119, 2022.

[37] N. Tishby and D. Polani, “Information Theory of Decisions and Actions,” in Perception-Action Cycle, New York, NY, USA: Springer, 2011, pp. 601–636.

[38] P. Park, S. C. Ergen, C. Fischione, C. Lu, and K. H. Johansson, “Wireless network design for control systems: A survey,” IEEE Commun. Surveys Tuts., vol. 20, no. 2, pp. 978–1013, Second Quarter 2018.

[39] Y. Wang, S. Wu, C. Lei, J. Jiao, and Q. Zhang, “A review on wireless networked control system: The communication perspective,” IEEE Internet Things J., vol. 11, no. 5, pp. 7499–7524, Mar. 2024.

[40] L. Xiao, M. Johansson, H. Hindi, S. Boyd, and A. Goldsmith, “Joint optimization of communication rates and linear systems,” IEEE Trans. Autom. Control, vol. 48, no. 1, pp. 148–153, Jan. 2003.

[41] G. Zhao, M. A. Imran, Z. Pang, Z. Chen, and L. Li, “Toward real-time control in future wireless networks: Communication-control co-design,” IEEE Commun. Mag., vol. 57, no. 2, pp. 138–144, Apr. 2019.

[42] Z. Lu and G. Guo, “Control and communication scheduling co-design for networked control systems: A survey,” Int. J. Syst. Sci., vol. 54, no. 1, pp. 189–203, 2023.

[43] N. Negi and A. Chakrabortty, “Optimal co-designs of communication and control in bandwidth-constrained cyber-physical systems,” Automatica, vol. 142, art. no. 110288, Aug. 2022.

[44] I. Molchanov, Theory of Random Sets, 2nd edition. Probability Theory and Stochastic Modelling, Springer, 2017.

[45] T. Cover and J. Thomas, Elements of Information Theory, New York: Wiley, 1991.

[46] A. El Gamal and Y. H. Kim, Network Information Theory, Cambridge: Cambridge University Press, 2011.

## APPENDIX

## A. Proof of Theorem 33

Proof: Property (1) follows directly from the definition of $\underline { { A } } _ { \epsilon } ^ { ( n ) }$ . By Theorem 28, the probability of the event $\underline { W } ^ { n } \in \underline { A } _ { \epsilon } ^ { ( n ) }$ tends to 1 as $n \to \infty$ . Thus for any $\epsilon > 0$ , there exists $n _ { 0 }$ such that for all $n \geq n _ { 0 } , \operatorname* { P r } \{ \underline { { W } } ^ { n } \in \underline { { A } } _ { \epsilon } ^ { ( n ) } \} > 1 - \epsilon .$ , proving property (2). Summing over the set $\underline { { A } } _ { \epsilon } ^ { ( n ) }$ , using property (2) and the bounds from property (1), we obtain

$$
\begin{array} { r } { ( 1 - \epsilon ) 2 ^ { n ( H _ { p } ( \underline { { W } } ) - \epsilon ) } \leq | \underline { { A } } _ { \epsilon } ^ { ( n ) } | \leq 2 ^ { n ( H _ { p } ( \underline { { W } } ) + \epsilon ) } , } \end{array}\tag{327}
$$

which proves property (3).

## B. Proof of Theorem 34

Proof: Property (1) follows directly from the definition of $E _ { \epsilon } ^ { ( n ) } ( \underline { { w } } ^ { n } )$ . To prove the left inequality of property (2), we write

$$
\begin{array} { r l } & { P ( \underline { w } ^ { n } ) = \displaystyle \sum _ { \tilde { w } ^ { n } \in E _ { \epsilon } ^ { ( n ) } ( \underline { w } ^ { n } ) } P ( \tilde { w } ^ { n } ) } \\ & { \qquad \le \displaystyle \sum _ { \tilde { w } ^ { n } \in E _ { \epsilon } ^ { ( n ) } ( \underline { w } ^ { n } ) } P ( \underline { w } ^ { n } ) 2 ^ { - n ( H _ { s } ( \tilde { W } ) - H _ { p } ( \underline { W } ) - \epsilon ) } } \\ & { \qquad \quad = P ( \underline { w } ^ { n } ) | E _ { \epsilon } ^ { ( n ) } ( \underline { w } ^ { n } ) | 2 ^ { - n ( H _ { s } ( \tilde { W } ) - H _ { p } ( \underline { W } ) - \epsilon ) } . } \end{array}\tag{328}
$$

Dividing by $P ( \underline { { w } } ^ { n } ) > 0$ yields

$$
| E _ { \epsilon } ^ { ( n ) } ( { \underline { { w } } } ^ { n } ) | \geq 2 ^ { n ( H _ { s } ( \tilde { W } ) - H _ { p } ( { \underline { { W } } } ) - \epsilon ) } .\tag{329}
$$

For the right inequality, we have

$$
\begin{array} { r l } & { P ( \underline { w } ^ { n } ) = \displaystyle \sum _ { \tilde { w } ^ { n } \in E _ { \epsilon } ^ { ( n ) } ( \underline { w } ^ { n } ) } P ( \tilde { w } ^ { n } ) } \\ & { \qquad \ge \displaystyle \sum _ { \tilde { w } ^ { n } \in E _ { \epsilon } ^ { ( n ) } ( \underline { w } ^ { n } ) } P ( \underline { w } ^ { n } ) 2 ^ { - n ( H _ { s } ( \tilde { W } ) - H _ { p } ( \underline { W } ) + \epsilon ) } } \\ & { \qquad \quad = P ( w ^ { n } ) | E _ { \epsilon } ^ { ( n ) } ( \underline { w } ^ { n } ) | 2 ^ { - n ( H _ { s } ( \tilde { W } ) - H _ { p } ( \underline { W } ) + \epsilon ) } , } \end{array}\tag{330}
$$

so

$$
| E _ { \epsilon } ^ { ( n ) } ( { \underline { { w } } } ^ { n } ) | \leq 2 ^ { n ( H _ { s } ( \tilde { W } ) - H _ { p } ( { \underline { { W } } } ) + \epsilon ) } .\tag{331}
$$

This completes the proof.

## C. Proof of Theorem 36

Proof: Property (1) follows directly from the definition of $B _ { \epsilon } ^ { ( n ) } ( \underline { { w } } ^ { n } )$ . To prove the left inequality of property (2), we write

$$
\begin{array} { l } { \displaystyle P ( \underline { w } ^ { n } ) = \sum _ { w ^ { n } \in B _ { \epsilon } ^ { ( n ) } ( \underline { w } ^ { n } ) } P ( w ^ { n } ) } \\ { \displaystyle \qquad \leq \sum _ { w ^ { n } \in B _ { \epsilon } ^ { ( n ) } ( \underline { w } ^ { n } ) } P ( \underline { w } ^ { n } ) 2 ^ { - n ( H ( W ) - H _ { p } ( \underline { W } ) - \epsilon ) } } \\ { \displaystyle \qquad = P ( \underline { w } ^ { n } ) | B _ { \epsilon } ^ { ( n ) } ( \underline { w } ^ { n } ) | 2 ^ { - n ( H ( W ) - H _ { p } ( \underline { W } ) - \epsilon ) } . } \end{array}\tag{332}
$$

Dividing by $P ( \underline { { w } } ^ { n } ) > 0$ gives

$$
| B _ { \epsilon } ^ { ( n ) } ( { \underline { { w } } } ^ { n } ) | \geq 2 ^ { n ( H ( W ) - H _ { p } ( { \underline { { W } } } ) - \epsilon ) } .\tag{333}
$$

For the right inequality,

$$
\begin{array} { r l } & { P ( \underline { w } ^ { n } ) = \displaystyle \sum _ { w ^ { n } \in B _ { \epsilon } ^ { ( n ) } ( \underline { w } ^ { n } ) } P ( w ^ { n } ) } \\ & { \qquad \ge \displaystyle \sum _ { w ^ { n } \in B _ { \epsilon } ^ { ( n ) } ( \underline { w } ^ { n } ) } P ( \underline { w } ^ { n } ) 2 ^ { - n ( H ( W ) - H _ { p } ( \underline { W } ) + \epsilon ) } } \\ & { \qquad = P ( \underline { w } ^ { n } ) | B _ { \epsilon } ^ { ( n ) } ( \underline { w } ^ { n } ) | 2 ^ { - n ( H ( W ) - H _ { p } ( \underline { W } ) + \epsilon ) } , } \end{array}\tag{334}
$$

so

$$
| B _ { \epsilon } ^ { ( n ) } ( { \underline { { w } } } ^ { n } ) | \leq 2 ^ { n ( H ( W ) - H _ { p } ( { \underline { { W } } } ) + \epsilon ) } .\tag{335}
$$

This completes the proof.

## D. Proof of Theorem 39

Proof: Properties (1)-(3) follow directly from the definition of $\underline { { A } } _ { \epsilon } ^ { ( n ) }$ and the weak law of large numbers applied to the i.i.d. pragmatic sequence pair, with the aggregated probabilities over the reified equivalence classes. For property (4), since $\dot { X } ^ { n }$ and $\dot { Y } ^ { n }$ are independent, we have

$$
\operatorname* { P r } \{ ( \underline { { \dot { X } } } ^ { n } , \underline { { \dot { Y } } } ^ { n } ) \in \underline { { A } } _ { \epsilon } ^ { ( n ) } \} = \sum _ { ( \underline { { x } } ^ { n } , \underline { { y } } ^ { n } ) \in \underline { { A } } _ { \epsilon } ^ { ( n ) } } P ( \underline { { x } } ^ { n } ) P ( \underline { { y } } ^ { n } ) .
$$

Using the bounds on $P ( \underline { x } ^ { n } )$ and $P ( y ^ { n } )$ from the entropy hierarchy $H _ { p } ( \underline { { X } } ) \leq H _ { s } ( \tilde { X } ) \leq H ( X )$ and the size bound on $\underline { { A } } _ { \epsilon } ^ { ( n ) }$ , we obtain

$$
\operatorname* { P r } \{ ( \underline { { \dot { X } } } ^ { n } , \dot { \underline { { Y } } } ^ { n } ) \in \underline { { A } } _ { \epsilon } ^ { ( n ) } \} \le 2 ^ { n ( H _ { p } ( \underline { { X } } , \underline { { Y } } ) + \epsilon ) } 2 ^ { - n ( H ( X ) - \epsilon ) } 2 ^ { - n ( H ( Y ) - \epsilon ) } = 2 ^ { - n ( I ^ { p } ( \underline { { X } } ; \underline { { Y } } ) - 3 \epsilon ) } .
$$

Similarly,

$$
\begin{array} { r } { \operatorname* { P r } \{ ( \dot { \underline { X } } ^ { n } , \dot { \underline { Y } } ^ { n } ) \in \underline { A } _ { \epsilon } ^ { ( n ) } \} \geq ( 1 - \epsilon ) 2 ^ { n ( H _ { p } ( { \underline { X } } , { \underline { Y } } ) - \epsilon ) } 2 ^ { - n ( H ( { \underline { X } } ) + \epsilon ) } 2 ^ { - n ( H ( { \underline { Y } } ) + \epsilon ) } = ( 1 - \epsilon ) 2 ^ { - n ( I ^ { p } ( { \underline { X } } ; { \underline { Y } } ) + 3 \epsilon ) } . } \end{array}
$$

This proves property (4).

## E. Proof of Theorem 40

Proof: Property (1) follows directly from the definition of $B _ { \epsilon } ^ { ( n ) } ( \underline { { x } } ^ { n } , y ^ { n } )$ . To prove the left inequality of property (2), we write

$$
\begin{array} { r l } & { P ( \underline { { x } } ^ { n } , \underline { { y } } ^ { n } ) = \displaystyle \sum _ { ( x ^ { n } , y ^ { n } ) \in B _ { \epsilon } ^ { ( n ) } ( \underline { { x } } ^ { n } , \underline { { y } } ^ { n } ) } P ( x ^ { n } , y ^ { n } ) } \\ & { \qquad \le \displaystyle \sum _ { ( x ^ { n } , y ^ { n } ) \in B _ { \epsilon } ^ { ( n ) } ( \underline { { x } } ^ { n } , \underline { { y } } ^ { n } ) } P ( \underline { { x } } ^ { n } , \underline { { y } } ^ { n } ) 2 ^ { - n ( H ( X , Y ) - H _ { p } ( \underline { { X } } , \underline { { Y } } ) - \epsilon ) } } \\ & { \qquad \quad ( x ^ { n } , y ^ { n } ) \in B _ { \epsilon } ^ { ( n ) } ( \underline { { x } } ^ { n } , \underline { { y } } ^ { n } ) } \\ & { \qquad = P ( \underline { { x } } ^ { n } , y ^ { n } ) | B _ { \epsilon } ^ { ( n ) } ( \underline { { x } } ^ { n } , y ^ { n } ) | 2 ^ { - n ( H ( X , Y ) - H _ { p } ( \underline { { X } } , \underline { { Y } } ) - \epsilon ) } . } \end{array}
$$

Dividing by $P ( \underline { { x } } ^ { n } , y ^ { n } ) > 0$ yields the lower bound. The right inequality follows similarly by reversing the inequality. □

## F. Proof of Theorem 43

Proof: Properties (1)-(3) follow directly from the definition of $\underline { { A } } _ { \epsilon } ^ { ( n ) }$ and the weak law of large numbers applied to the i.i.d. pragmatic sequence pair, with the aggregated probabilities over the reified equivalence classes. For property (4), since $\dot { X } ^ { n }$ and ${ \hat { \dot { X } } } ^ { n }$ are independent, we have

$$
\operatorname* { P r } \{ ( \underline { { \dot { X } } } ^ { n } , \underline { { \hat { X } } } ^ { n } ) \in \underline { { A } } _ { \epsilon } ^ { ( n ) } \} = \sum _ { ( \underline { { x } } ^ { n } , \underline { { \hat { x } } } ^ { n } ) \in \underline { { A } } _ { \epsilon } ^ { ( n ) } } P ( \underline { { x } } ^ { n } ) P ( \underline { { \hat { x } } } ^ { n } ) .
$$

Using the bounds on $P ( \underline { x } ^ { n } )$ and $P ( \underline { { \hat { x } } } ^ { n } )$ from the entropy hierarchy $H _ { p } ( \underline { { X } } ) \leq H _ { s } ( \tilde { X } ) \leq H ( X )$ and the size bound on $\underline { { A } } _ { \epsilon } ^ { ( n ) }$ , we obtain the desired exponential bounds. □