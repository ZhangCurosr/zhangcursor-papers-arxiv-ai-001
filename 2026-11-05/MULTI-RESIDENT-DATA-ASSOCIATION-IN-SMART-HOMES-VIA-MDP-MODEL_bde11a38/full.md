# MULTI-RESIDENT DATA ASSOCIATION IN SMART HOMES VIA MDP MODELING AND BEHAVIOR CLONING

AUTHOR VERSION

Xi Chen,<sup>1,2,</sup> Julien Cumin,<sup>2,</sup> Fano Ramparany,<sup>2,</sup> Dominique Vaufreydaz<sup>1,</sup> <sup>,</sup>

<sup>1</sup> Univ. Grenoble Alpes, CNRS, Grenoble INP, LIG, 38000 Grenoble, France

<sup>2</sup> Orange Research

## ABSTRACT

Ambient sensors offer a privacy-preserving solution for activity recognition in smart homes, yet they face a critical challenge in multi-resident environments known as Data Association, which aims to identify the resident triggering a specific anonymous sensor event. Existing approaches fundamentally fail to maintain global consistency by often yielding conflicting associations for silent residents. Additionally, these methods frequently suffer from severe trajectory fragmentation caused by the ontological mismatch of applying continuous physical motion models to discrete logical behaviors. To address these limitations, we propose a novel framework that reformulates data association as a sequential decision-making task within a Markov Decision Process (MDP). We solve the defined MDP via Behavior Cloning (BC) to avoid complex reward engineering by directly learning from expert demonstrations. We instantiate the policy using two complementary architectures, including a lightweight Transformer that treats association as a multi-label classification task over linearized event histories, and a fine-tuned Large Language Model (LLM) that reformulates the problem as conditional text generation to leverage pre-trained semantic knowledge. Extensive evaluations on three diverse datasets demonstrate that our approach significantly outperforms stateof-the-art baselines, particularly in resolving concurrent multi-resident events. Furthermore, the LLM-based policy shows preliminary cross-environment transfer potential under benchmark settings, while its robustness in real deployments still requires further validation.

## Keywords:

## 1 Introduction

Ambient sensors serve as a practical foundation for intelligent smart-home services because they are low-power, unobtrusive, and privacy-preserving, making them particularly attractive for daily residential monitoring compared to more intrusive modalities such as cameras or wearables. These properties have fueled a wave of sophisticated research into high-leve behavioral analytics, including health monitoring, home safety, and building automation [1, 3, 10, 26, 31, 34]. However, a fundamental mismatch emerges when these applications move beyond controlled laboratory studies: most experimenta settings simplify the problem to single-resident households, whereas real-world homes are commonly shared by multiple occupants. Under these realistic multi-resident conditions, the identity-blind nature of ambient observations becomes a fundamental and shared barrier. In all of these settings, the practical value of sensor data depends entirely on recovering resident-level context rather than merely detecting that some occupant triggered a sensor. This inability to isolate residentspecific event streams has effectively blocked the practical implementation of advanced services. Specifically, clinically meaningful monitoring of behavioral changes [5, 21, 34], resident-specific anomaly detection [2, 20, 31, 32], and personalized building automation [3, 26, 28, 43] all remain largely unusable in shared homes because they cannot reliably attribute activities or preferences to a specific target individual. Without a robust mechanism to separate interleaved sensor streams and reconstruct individual trajectories, ambient intelligence remains confined to unrealistic single-occupant assumptions that fail to meet the personalization and reliability requirements of true residential deployment. More broadly, this challenge sits at the systems boundary between sensing and personalized interaction: ubiquitous computing applications can be embedded, privacy-preserving, and continuously available in the home, but they remain fundamentally limited if they cannot maintain resident-specific context under realistic multi-occupant conditions.

This technical gap stems from the inherent nature of ambient sensors, which typically provide sparse, fixed-location, and event-driven observations without explicit identity labels. When several residents share the same environment, their sensor events become temporally interleaved, thereby obscuring the continuity of each individual’s behavior. The task of recovering separated individual trajectories from these interleaved observations is called Data Association [7, 8, 11, 15, 24, 39, 40, 42]. Compared with indoor localization, which focuses on continuously estimating coordinates [23], Data Association operates on sensor events that are semantically richer but much sparser. This higher semantic level is particularly valuable for downstream applications because it is more directly tied to meaningful activities and interactions. For example, knowing that a resident is in the kitchen does not by itself reveal whether the resident is cooking or washing dishes. In contrast, observing that the stove’s power sensor reports an increase provides stronger semantic evidence that the resident may be cooking. The goal of Data Association is therefore to disentangle these sparse symbolic events into coherent resident-level event streams so that downstream semantic inference, such as activity recognition [7], can be carried out on each recovered trajectory.

Existing Data Association methods largely fall into event-centric and global state paradigms, yet both face significant structural limitations. Event-centric approaches [12, 19] classify observations with limited context. However, by lacking a persistent memory of “silent” residents, they fundamentally fail to maintain global consistency, often producing conflicting associations that violate multi-resident physical constraints. In practical shared-home settings, such inconsistencies can blur who performed which activity, directly undermining resident-specific interpretation. Conversely, global state approaches attempt to track the full environment but struggle with modeling capability. Classic discrete filtering methods [42] suffer from combinatorial explosion in the hypothesis space and are typically restricted to first-order Markovian transitions, limiting their ability to capture long-term behavioral dependencies. Continuous embedding approaches [39, 40] circumvent this combinatorial bottleneck by projecting discrete events into a continuous latent space to leverage efficient approximations (e.g., GM-PHD filters). However, this paradigm introduces an ontological mismatch: by enforcing linear dynamic model on discrete symbolic behaviors, they suffer from covariance divergence during sparse sensor activations, leading to trajectory fragmentation. In practice, these fragmented trajectories can break resident-level behavioral summaries and destabilize downstream monitoring or anomaly attribution. Most recently, LLM-based approaches [13] have revisited discrete reasoning but primarily rely on frozen pre-trained knowledge rather than actively utilizing sensor data for training, resulting in weak adaptation to domainspecific distributions. These limitations highlight the need for a principled framework that synergizes (i) an extensible, history-augmented discrete state that transcends first-order Markovian limits to capture long-term context, and (ii) datadriven transition dynamics capable of resolving complex association ambiguities via learned domain knowledge.

In this paper, we address these requirements by reformulating multi-resident Data Association as a Markov Decision Process (MDP) [30]. Our key shift is conceptual: instead of treating each sensor event as an isolated classification target or estimating assignments through recursive filtering alone, we cast the problem as sequential decision-making over a historyaugmented global state. In our formulation, the state explicitly summarizes the evolving assignment history of all residents, and each action selects the resident set responsible for the current event. This perspective is well suited to shared homes because it enables the policy to incorporate long-range contextual dependencies, maintain logical consistency across residents, and naturally handle the dynamic evolution of household occupancy. Moreover, by explicitly organizing the task around states, actions, and sequential decisions, the MDP provides a principled framework for learning-based data association. In practice, however, specifying a reliable reward function for online deployment is difficult, whereas expert trajectories are already available in annotated datasets. We therefore solve the proposed MDP through Behavior Cloning (BC) [27], treating labeled event streams as demonstrations of expert assignment decisions. This yields a practical data-driven learning strategy while preserving the sequential semantics of the formulation. To study how different policy classes exploit this framework, we instantiate the policy in two complementary ways: a lightweight Transformer [38] that learns temporal assignment patterns from scratch, and a fine-tuned Large Language Model (LLM) [45] that casts resident assignment as conditional generation and can better transfer across heterogeneous sensor environments.

We evaluate our proposed framework on three diverse public datasets: CASAS [36], MARBLE [6], and MuRAL [14], covering a wide range of multi-resident scenarios and sensing configurations. Extensive experiments show that our MDP-based approach, trained via Behavior Cloning, consistently outperforms state-of-the-art baselines [13, 39] by a substantial margin. In particular, it better maintains global consistency, assigns ambiguous events through beam search over candidate trajectories and resolves concurrent multi-resident events that prior methods struggle to handle. Furthermore, the LLM-instantiated policy exhibits strong cross-environment generalization, demonstrating that semantic knowledge can transfer across heterogeneous sensor environments without extensive retraining. Finally, a simulated downstream health-monitoring analysis shows that these gains in data association also translate into substantially lower activity-duration estimation errors, further supporting the practical value of our method for resident-level health applications. In summary, our main contributions are as follows:

• We reformulate multi-resident Data Association as a history-aware MDP, providing a unified decision-making framework that explicitly models assignment continuity over time. This formulation moves beyond isolated event classification and first-order filtering, and makes it possible to address ambiguous and overlapping multi-resident events in a principled way.

• We develop a practical Behavior Cloning solution to this MDP and instantiate it with two complementary policy classes: a lightweight Transformer for efficient data-driven assignment and a fine-tuned LLM for stronger semantic transfer across environments. This design avoids reward engineering while supporting both accurate trajectory recovery and improved generalization under heterogeneous sensor settings.

• We validate the framework on three public datasets (CASAS, MARBLE, and MuRAL), where it consistently surpasses strong baselines across multiple metrics. Beyond benchmark gains, our results show tangible practical value: the method better preserves resident-level consistency, improves the handling of ambiguous and concurrent events, and reduces downstream activity-duration estimation errors in a simulated health-monitoring setting.

## 2 Related Work

## 2.1 Practical Limitations of Ambient-based Smart Home Applications

Ambient sensors are an important modality for understanding residents’ daily activities in smart homes. Their practical appeal stems from several well-known advantages: they are non-intrusive, low-power, privacy-preserving, and easy to deploy throughout the home. These characteristics also contribute to their user acceptability. Chan et al. [10] reported that inhome monitoring technologies show moderately positive acceptability overall, particularly for ambient sensors, and highlighted evidence that ambient sensors are better accepted than wearable alternatives. Precisely because of these advantages, ambient sensors have been widely adopted to support downstream applications such as health monitoring [5, 10, 21, 34], home safety and security [2, 20, 31, 32], and building automation and energy management [3, 26, 28, 43]. At the core of these applications lies the extraction of contextual information, with Activities of Daily Living (ADLs) serving as a particularly important semantic layer.

This dependence on ADL-level context is especially evident in healthcare-oriented smart-home systems, where the goal is continuous, resident-specific monitoring of clinically meaningful behavioral changes. Aramendi et al. [5], for example, transformed ambient sensor streams into activity labels and derived indicators such as cooking duration, meal times, and nighttime bathroom frequency to predict functional health decline in older adults. Giannios et al. [21] used ambient events and ADL information to quantify deviations in meal-preparation behavior for distinguishing healthy residents from those with mild cognitive impairment, while Riboni et al. [34] combined activity classification and anomaly detection to support the early identification of cognitive decline. Yet all of these pipelines depend on individual-level attribution: the system must know whose meals are becoming irregular, whose nocturnal bathroom visits are increasing, and whose routines are deviating from baseline. This dependence becomes a major obstacle in shared homes. Aramendi et al. explicitly discarded all two-resident samples because activities from different occupants could not be reliably separated; Giannios et al. similarly noted that the lack of individual-level ADL recognition confines their framework to single-resident settings; and Riboni et al. identified the separation of individual activities in multi-resident environments as necessary future work. At a broader level, Chan et al. [10] surveyed 46 studies on smart-home health monitoring and found that evidence for multi-person ambient sensing remains extremely limited: only a small minority reported any multi-person detection accuracy, the few ambient-sensor studies addressing multi-person settings focused mainly on basic presence or identity discrimination rather than clinically meaningful ADL-based outcomes, and no clinical study using ambient sensors provided dedicated validation in realistic shared-home settings. Collectively, these findings suggest that the main bottleneck is not the lack of clinical value, but the unresolved difficulty of reliably isolating each resident’s activities in multi-resident homes.

A similar set of challenges plagues domestic safety and building automation. Recent surveys [20, 32] explicitly point out that existing anomaly-detection methods remain overwhelmingly anchored to single-occupant assumptions. As these studies emphasize, the inability to reliably separate individual event trajectories makes it difficult to analyze resident-specific anomalies (e.g., falls), distinguish genuine behavioral changes from multi-user interference, or detect external intrusions with confidence.

This limitation also extends to energy-saving applications [3, 28, 43], where ambient sensing is used to control lighting and heating, ventilation, and air conditioning (HVAC) systems. Lashkari et al. [26] noted that most literature neglects realistic shared-home conditions involving concurrent behaviors and heterogeneous comfort preferences. Without clean trajectory separation, the system cannot personalize comfort or optimize energy load effectively: apparent vacancies may merely reflect hand-offs between occupants, and energy-control decisions may become inefficient or inappropriate. Consequently, across both safety and energy domains, the recovery of resident-level trajectories remains a fundamental barrier to practical deployment.

The collective evidence from these diverse applications points to a common practical limitation, namely the inability to distinguish between occupants in multi-resident settings. This constraint directly hinders the translation of promising research findings into widespread real-world deployment. Consequently, existing studies emphasize a pressing need for solutions capable of reliably attributing sensor events to specific individuals. This task is formally defined as Data Association [7, 8, 11, 24, 39, 42]. Rather than a mere technical refinement, Data Association represents the fundamental enabling step that allows healthcare, safety, and automation services to move beyond unrealistic single-occupancy assumptions. Prior empirical studies [12, 15, 40] reinforce this by showing that robust data association can substantially improve multi-resident ADL recognition performance, thereby serving as a necessary foundation for deploying intelligent ambient services in realistic shared homes.

## 2.2 Multi-Resident Data Association

Following a detailed review of Data Association methods developed over the past two decades, a clear methodological dichotomy emerges based on the scope of system state modeling: event-centric approaches, which focus on characterizing individual observations in isolation or local context, and global state approaches, which attempt to maintain a holistic representation of all residents in the environment.

## 2.2.1 Event-Centric and Partial State Approaches

Earlier approaches primarily treated multi-resident data association as a direct classification problem, minimizing the reliance on state maintenance. A classic example is the work by Crandall and Cook [17, 18], who applied a Naive Bayes (NB) classifier to assign each sensor event to the most likely resident. However, this stateless approach is inherently limited by its inability to capture context. For instance, if the system assigns an event in the bedroom to Alice, a subsequent event in the living room is physically less likely to be Alice’s. A stateless classifier, lacking memory of Alice’s recent location, fails to utilize this spatial constraint. To address this, subsequent works introduced temporal modeling, yet they stopped short of a full system representation, adopting instead a partial state formulation. Methods by Crandall et al. [19] and Singla et al. [36] utilized Hidden Markov Models (HMMs) where the hidden state represented only the identity or activity associated with the current observation. Similarly, joint models [16, 24] using CRFs [24] or HMM [16] inferred identity and activity simultaneously but remained fundamentally event-centric.

The critical limitation shared by these methods is that the state representation collapses to a single variable tuple associated only with the active resident triggering the sensor. Consequently, the system maintains no memory of residents who are currently silent $( \mathrm { i . e . , }$ not triggering sensors). For example, if Alice is sleeping (silent) while Bob is cooking (active), these partial state models effectively lose track of Alice. By redefining “state” strictly as a hidden attribute of the current event rather than the configuration of the environment, these approaches functionally regress toward stateless classification, lacking the global view necessary to reason about temporal continuity for all residents simultaneously. This limitation extends even to modern deep sequence models like Seq2Res [12]: while they expand the observation window from a single event to a sequence block, they still lack a persistent global memory, effectively treating the task as a batch translation problem that suffers from state forgetting over long durations.

## 2.2.2 Global State Approaches

Recognizing the necessity of tracking the full environment, Wilson and Atkeson [42] established the foundational definition of Global State modeling. They formalized the system state at time t as the collection of all individual resident states:

$$
{ \cal S } _ { t } = \{ s _ { t } ^ { 1 } , s _ { t } ^ { 2 } , . . . , s _ { t } ^ { K } \} ,
$$

where $s _ { t } ^ { i }$ represents the discrete location and activity of resident i.

To estimate this complex joint state, they proposed a Rao-Blackwellized Particle Filter (RBPF) framework. Crucially, this approach conceptually decomposes the multi-resident Data Association task into three sub-tasks: (1) global state representation, which maintains the joint configuration of all residents; (2) individual state transition, where each resident’s dynamics are modeled as an independent HMM, allowing motion patterns to be learned from simpler single-resident data; and (3)joint state hypothesisfiltering, which uses particles to sample data associations and weight them against observations. This formulation enables a modular and semantically accurate approach to reasoning. However, despite its theoretical elegance this approach suffers from a severe computational bottleneck: combinatorial explosion within the hypothesis filtering step. As the number of residents and sensor events increases, the discrete hypothesis space for data association grows exponentially. Consequently, an impractical number of particles is required to cover the valid association possibilities, making the approach computationally intractable for larger-scale deployments.

To overcome this barrier, recent methods such as sMRT [39] and GAMUT [40] proposed a paradigm shift. Instead of modeling the system in its native discrete space, they project discrete sensor events into a continuous latent space via representation learning (e.g., Word2Vec). This transformation allows the global state to be defined as a set of continuous coordinate vectors $X _ { t } ~ = ~ \{ z _ { t } ^ { 1 } , \ldots , z _ { t } ^ { K _ { t } } \}$ }, enabling the use of the Gaussian Mixture Probability Hypothesis Density (GM-PHD) filter. The GM-PHD filter scales linearly with the number of targets and naturally handles time-varying resident numbers, effectively solving the computational tractability issue. Nevertheless, this shift to continuous embedding introduces a fundamental ontological mismatch. Human activity is inherently logical and symbolic, which is driven by intent (e.g., Bedroom → Kitchen) rather than by physical inertia. By enforcing a continuous representation, these methods impose artificial kinematic constraints (e.g., constant velocity) on discrete logical jumps. This theoretical flaw manifests mathematically as covariance divergence. Due to the temporal sparsity of smart home events, the error covariance matrix of the linear dynamic model grows monotonically during long intervals between observations. When the uncertainty exceeds the filter’s threshold, valid trajectories are terminated, and subsequent events are mistakenly initialized as new tracks. Thus, while continuous methods solve the scalability problem, they do so at the cost of semantic fidelity, leading to trajectory fragmentation caused by applying physical motion models to discrete symbolic behaviors. In practical terms, this fragmentation is especially problematic for the applications discussed above: once a resident’s trajectory is repeatedly broken or reassigned, downstream systems lose the continuity needed to interpret prolonged routines, attribute anomalies to the correct individual, or trigger personalized services with confidence in multi-resident homes.

Most recently, LAHAR [13] has revisited the discrete global state paradigm, serving as a modern semantic successor to Wilson and Atkeson’s framework. Instead of relying on computationally expensive particle filters, LAHAR leverages LLMs to explicitly represent the global state in natural language. By summarizing each resident’s past state and using the LLM’s reasoning capabilities to infer the next state $S _ { t + 1 }$ given a new event, LAHAR effectively bypasses the combinatorial complexity of traditional filtering while preserving the logical fidelity of human behavior. However, while this approach resolves the semantic issues of continuous models, it introduces new challenges intrinsic to LLMs: prohibitive computational costs, high inference latency, and the under-utilization of available sensor training data, as the model relies more on pre-trained knowledge than on learning from domain-specific distributions.

## 2.2.3 Synthesis: Towards a Decision-Making Paradigm

Synthesizing these historical developments, our approach is fundamentally premised on a strict adherence to the discrete Global State formulation [42], which we regard as the prerequisite for maintaining multi-resident logical consistency. However, while we adopt this representation, we identify a critical structural limitation in the frameworks historically used to estimate it. Whether utilizing classical discrete methods (e.g., RBPF [42]) or modern continuous embeddings (e.g., sMRT [39]), existing approaches predominantly operate within the Bayesian Filtering paradigm. Consequently, they are bound by the rigid First-Order Markov assumption, relying recursively on the immediate previous state to estimate the current one. This restriction inherently limits their ability to capture complex, long-term behavioral dependencies. Furthermore, each faces specific hurdles: discrete filters suffer from combinatorial explosion, while continuous filters incur an ontological mismatch by imposing physical motion models on logical behaviors.

To transcend these dual limitations—preserving the rigor of the global state while breaking the first-order barrier—we propose shifting the methodological perspective from recursive filtering to sequential decision-making. This insight naturally leads to our Markov Decision Process (MDP) [30] formulation. By explicitly augmenting the state with the assignment history, our framework enables the policy to reason about rich contexts, thereby combining the semantic fidelity of the global state with the computational power of data-driven learning.

## 3 Problem Definition and Method Overview

## 3.1 Problem Definition

Prior work often uses the term Data Association informally without a precise, task-level definition, which limits rigorous discussion and reasoning about objectives, assumptions, and evaluation. To better illustrate our objectives and methodology, we therefore first provide a formal definition of the Data Association problem.

We first define a sensory environment as a set $\mathrm { E } = \{ o _ { i } \} _ { 1 \leq i \leq I }$ , where each $o _ { i }$ represents an ambient sensor along with its associated information (e.g., type, location, and capabilities) and I is the total number of sensors in this environment.

Within a sensory environment, a set of subjects $\mathrm { U } = \{ u _ { k } \} _ { 1 \leq k \leq K }$ can enter, exit, and perform various activities. Here, K denotes the total number of distinct subjects that have appeared up to time t. As time evolves, K is a non-decreasing function of t, since new subjects may appear but previously observed subjects remain counted.

A sensor event $e _ { t }$ occurred in the environment E is defined as a tuple:

$$
e _ { t } = ( t , o _ { i } , c ) ,\tag{1}
$$

where:

• t is the timestamp at which the sensor state change occurs.

$o _ { i } \in$ E is an ambient sensor in the environment.

• c is the change of the sensor state (e.g., activation, deactivation, or a value update).

Given a sequence of sensor events recorded from the environment, denoted as $[ e _ { t } ] _ { 1 \leq t \leq T }$ , where $T$ is the length of this sequence, the Data Association problem consists in constructing a mapping M which maps each event $e _ { t }$ to its corresponding residents $\mathcal { U } _ { t }$ . Formally, we have:

$$
\ M _ { U } : e _ { t } \mapsto \mathcal { U } _ { t } ,\tag{2}
$$

where $\mathcal { U } _ { t } \subseteq \mathrm { U }$ is a subset of the total subject set $\mathrm { U } = \{ u _ { k } \} _ { 1 \leq k \leq K }$

Depending on the sensor modality and the nature of the event, U<sub>t</sub> can be an empty set, a single subject, or multiple subjects. The empty-set case corresponds to events not attributable to any resident, which we uniformly treat as noise. Such noise may arise from sensor malfunction, measurement errors, or natural environmental changes (e.g., daily variations in temperature or illumination that trigger thermometer or light sensors). A single-subject case corresponds to actions such as one person opening a door, whereas multi-subject cases occur when several people interact within a shared space.

Example: To illustrate the above definitions, we consider a simple scenario: Alice opens the bedroom door and walks out, while Bob stands up from a chair. Then, both Alice and Bob walk together to the apartment’s main door and open it. Suppose there are three sensors installed in the environment: a bedroom door sensor $D _ { 1 } ,$ , a chair pressure sensor $C _ { 1 } .$ , and a main door sensor $D _ { 2 }$ . The environment can thus be defined as: $\mathrm { E } = \{ D _ { 1 } , C _ { 1 } , D _ { 2 } \}$ . In this scenario, three sensor events occur in sequence: $[ e _ { 1 } = ( 1 , D _ { 1 } , 0 { \bf P } { \bf E } { \bf N } ) , e _ { 2 } = ( 2 , C _ { 1 } , 0 { \bf F } { \bf F } ) , e _ { 3 } = ( 3 , D _ { 2 } , { \dot { \bf O } } { \bf P } { \bf E } { \bf N } ) ]$ . The goal of Data Association is to construct a mapping M from events to residents such that: $M ( e _ { 1 } ) = \{ A l i c e \} , M ( e _ { 2 } ) = \{ B o b \} , M ( e _ { 3 } ) = \{ A l i c e , B o b \}$

## 3.2 Method Overview

Figure 1 illustrates the detailed workflow of our proposed framework. The process begins with the MDP Formulation (Section 4), where the system constructs a structured state containing the resident history and the current event. To process this symbolic state, we process it into a linearized token sequence ${ \bf { \bar { X } } } _ { t } .$ The policy model is instantiated as a Transformer network (Section 5.3) to generate assignment probabilities $\widehat { Y } _ { t }$ . In the Inference loop, the selected action updates the global history, transitioning the system to the next state. In the Training loop, the framework follows the Behavior Cloning paradigm (Section 5): it utilizes ground truth assignments $Y _ { t }$ to compute the Binary Cross Entropy loss, optimizing the policy model via backpropagation. In addition to the architecture depicted here in Figure 1, we further implement an alternative policy based on LLMs, which reformulates the task as conditional text generation (Section 6).

![](images/3cfbd741e5124790daf565d8e9e7d0e87cb4b89775b3e168779d976061dffa09.jpg)  
Figure 1: Overview of our proposed MDP-based Data Association method, trained using Behavior Cloning.

## 4 Problem Modeling via MDP

In this section, we present how the Data Association problem can be formally modeled as an MDP. We first motivate this formulation in Section 4.1 by discussing the decision-making and sequential properties of the task. In Section 4.2, we then provide the basic definition of an MDP. Finally, in Section 4.3, we formally describe how to model Data Association as an MDP, laying the foundation for solving the problem through MDP-based approaches in the subsequent sections.

## 4.1 Rationale for the MDP Formulation

We adopt the MDP formulation because the Data Association problem exhibits the following properties:

1. Data Association is fundamentally a decision-making problem. Data Association is not simply a real-time filtering problem. Filtering usually focuses only on estimating the current state, whereas Data Association often involves longhorizon decision-making. For example, when two residents pass through the main door at the same time, an anonymous Main Door OPEN event alone may be insufficient to determine whether the event should be assigned to one resident or to both. The system may need to postpone commitment in the hypothesis space and reason about future possibilities, such as subsequent sensor activations in different rooms, before it can retrospectively determine the current assignment. This property of using future evidence to inform the present gives Data Association a fundamentally decision-making character.

2. Data Association is inherently sequential. Assignment decisions exhibit a strong chain effect. Because of physical continuity and the logical structure of daily routines, the association made at the current step directly constrains the space of plausible assignments at subsequent steps. For example, if the system determines that a Bathroom Door CLOSE event is triggered by a given resident, then under spatiotemporal consistency the immediately following Bathroom Door OPEN event should be much more likely to be assigned to the same resident. This direct shaping of future decisions by earlier actions is a fundamental sequential property of the problem.

3. These properties naturally align with the MDP framework. Taken together, the sequential decision-making nature of Data Association matches the mathematical structure of a Markov Decision Process. Its decision-making aspect corresponds to the ability of an MDP to search for an optimal policy that maximizes long-term return, while its sequential dependence corresponds to the way state transitions dynamically constrain future actions. By modeling

Data Association as an MDP, we obtain a principled formalism for representing resident history, current observations, and the long-term consequences of present assignments in a unified framework.

## 4.2 Definition of MDP

An agent is formally defined as an entity that makes decisions and takes actions within an environment, aiming to maximize long-term cumulative reward. An MDP is a mathematical framework that models the interaction between an agent and its environment in sequential decision-making settings.

A typical MDP can be defined by a tuple $( S , A , P , R )$ , where

• $S$ is the state space, representing all possible states of the agent,

• A is the action space, representing all actions that the agent is allowed to execute,

$P ( s ^ { \prime } | s , a )$ is the state transition model, the probability of reaching state $s ^ { \prime }$ from state s by taking action $^ { a , }$

$R ( s , a , s ^ { \prime } )$ is the reward model, the immediate reward for transitioning from s to $s ^ { \prime }$ via action a.

The agent’s objective is to optimize a decision policy $\pi : S  A$ , which maps states to actions, such that the expected cumulative reward over a sequence of actions is maximized:

$$
\pi ^ { * } = \arg \operatorname* { m a x } _ { \pi } \mathbb { E } \left[ \sum _ { t = 0 } ^ { T } R ( s _ { t } , \pi ( s _ { t } ) , s _ { t + 1 } ) \right] ,\tag{3}
$$

where t is the index of discrete time step and $T$ is the maximum of time steps. In this study, we only consider a finite number of decision steps.

## 4.3 Adapting MDP to Data Association

Under the MDP framework, the Data Association problem is modeled by the following components:

State At any event step t, the state $s _ { t }$ encodes the assignment history prior to $t ,$ denoted as $h _ { < t }$ , along with the current event $e _ { t }$ to be assigned. Formally, we define the state as:

$$
s _ { t } = ( h _ { < t } , e _ { t } ) .\tag{4}
$$

Here, $h _ { < t }$ serves as a dynamic record, implemented as a mapping from each resident to their assigned event sequence (cf. state example in Figure 1). This state representation captures both past context and the current decision point, and thus satisfies the Markov property.

Action As shown in the example of Figure 1, at each step t, the agent takes an action $a _ { t }$ , assigning the current event $e _ { t }$ to a non-empty subset of residents:

$$
a _ { t } : e _ { t } \mapsto \widehat { \mathcal { U } } _ { t } \subseteq \mathbb { U } _ { t } , \quad \widehat { \mathcal { U } } _ { t } \neq \emptyset ,\tag{5}
$$

where $\mathbb { U } _ { t } = \{ u _ { 1 } , \ldots , u _ { n } , u _ { \mathrm { n e w } } \}$ includes all known residents and a placeholder $u _ { \mathrm { n e w } }$ for dynamically introducing new ones.

This formulation supports both single- and multi-resident assignments, enabling the model to handle overlapping activities and adapt to scenarios with an unknown or growing number of residents.

Transition Model Given a state $s _ { t } = ( h _ { < t } , e _ { t } )$ and an action $a _ { t } .$ , the environment deterministically transitions to the next state by updating the assignment history and moving to the next event. Specifically, the current event $e _ { t }$ is appended to the event list of each assigned resident in $\widehat { \mathcal { U } } _ { t } = a _ { t } ( e _ { t } )$ , yielding a new history $h _ { < t + 1 }$ (cf. the updated state in Figure 1). If $\widehat { \mathcal { U } } _ { t }$ includes a resident not previously seen in $h _ { < t }$ , a new resident is created accordingly and labeled using the next available alphabetical ID. The next unprocessed event $e _ { t + 1 }$ becomes the new focus of assignment. Formally:

$$
s _ { t + 1 } = ( h _ { < t + 1 } , e _ { t + 1 } ) .\tag{6}
$$

This transition is deterministic and satisfies the Markov property: all necessary information for future decisions is encapsulated in the updated state.

Reward The reward function quantifies the immediate correctness of the agent’s decision at each step. To strictly align the MDP framework with the objective of accurate tracking, we define an oracle reward using the indicator function:

$$
R ( s _ { t } , a _ { t } ) = \mathbb { 1 } ( a _ { t } = a _ { t } ^ { * } ) ,\tag{7}
$$

where $a _ { t } ^ { * }$ denotes the ground-truth resident assignment for event $e _ { t } .$ Under this binary formulation, the agent receives a reward of 1 solely when the assigned resident set matches the ground truth exactly, and 0 otherwise. This definition establishes the fundamental metric for evaluating individual transitions.

Policy A policy defines a strategy for Data Association, mapping the current state $s _ { t }$ to an assignment action $a _ { t } .$ . Formally, we denote this deterministic mapping as $\pi ( s _ { t } ) = a _ { t }$ . Executing a policy π over a sequence of sensor events $[ e _ { t } ] _ { 1 \leq t \leq T }$ induces a complete assignment trajectory.

The ultimate goal of Data Association is to recover the ground-truth mapping $M$ , which corresponds to an optimal policy $\pi ^ { * }$ that assigns the correct resident set at every step:

$$
\forall t \in \{ 1 , . . . , T \} , \quad M ( e _ { t } ) = \pi ^ { * } ( s _ { t } ) = a _ { t } ^ { * } .\tag{8}
$$

Consequently, the Data Association problem can be cast as a policy optimization task. The objective is to find a policy $\pi ^ { \mathrm { { o p t } } }$ that approximates $\pi ^ { * }$ by maximizing the expected cumulative reward:

$$
\pi ^ { \mathrm { o p t } } = \arg \operatorname* { m a x } _ { \pi } \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } R ( s _ { t } , \pi ( s _ { t } ) ) \right] .\tag{9}
$$

Since the reward R is defined as the binary correctness of each step, maximizing the cumulative reward is mathematically equivalent to maximizing the total assignment accuracy over the event stream. This formulation provides a rigorous theoretical basis for seeking the optimal association strategy.

## 5 Policy Learning via Behavior Cloning

## 5.1 Rationale for Behavior Cloning

Having formulated Data Association as an MDP in the previous section, the goal of this section is to solve the resulting decision problem in practice. In principle, many solution strategies are possible, including explicit search methods such as Monte Carlo Tree Search (MCTS), standard Reinforcement Learning (RL), and imitation-based policy learning. As the first work, to our knowledge, to cast multi-resident Data Association explicitly into this MDP form, our immediate objective is not to exhaust this algorithmic space, but to validate the formulation itself and establish a strong, transparent baseline for future work. We therefore seek a solver that is reliable, easy to interpret, and suitable for revealing both the strengths and the limitations of the proposed state–action–transition interface.

To this end, we adopt BC as the optimization strategy for the proposed MDP. Crucially, the assignment labels in annotated smart-home datasets already spec ${ \mathrm { f y } } ,$ at each decision step, the action that a human expert would regard as the correct attribution under full supervision. In other words, the dataset directly provides expert demonstrations of how the MDP should be solved. Rather than rediscovering this policy indirectly through online trial-and-error, we can therefore learn a parameterized policy $\pi _ { \theta }$ by directly imitating these expert actions, which is the most data-efficient and conceptually direct strategy for our setting. This choice allows us to validate the MDP formulation in a controlled and interpretable manner, while avoiding confounding the contribution of the formulation with the additional instability of reward engineering and exploration. Furthermore, as is common practice across many domains of sequential decision-making, utilizing BC to provide a crucial “warm start” is often necessary before RL can efficiently explore vast combinatorial action spaces. Thus, our BC approach serves as an essential first step, providing a solid foundation and a strong baseline upon which more advanced RL- or search-based solvers can be investigated in future work.

## 5.2 Learning Objective

To implement the Behavior Cloning framework, we translate the MDP objective, which is maximizing the cumulative reward (i.e., assignment accuracy), into a supervised optimization task. Since the reward is binary, maximizing the expected reward is equivalent to maximizing the probability that the agent selects the ground-truth action $a _ { t } ^ { * }$

Therefore, we aim to optimize the policy parameters θ such that the probability mass assigned to the expert’s actions is maximized. This corresponds to maximizing the log-likelihood of the observed expert trajectories under the learned policy distribution:

$$
\theta ^ { * } = \arg \operatorname* { m a x } _ { \theta } \sum _ { t = 1 } ^ { T } \log P _ { \theta } ( a _ { t } = \mathcal { U } _ { t } \mid s _ { t } ) .\tag{10}
$$

Action Space Instantiation While our general MDP formulation (Section 4.3) supports dynamic resident discovery via a growing set $\mathbb { U } _ { t }$ , implementing the policy using standard neural architectures requires a fixed output dimension. Therefore, for this learning instantiation, we define a fixed universal resident pool $\mathbb { U } = \{ u _ { 1 } , \ldots , u _ { K } \}$ , where K represents the maximum capacity of trackable residents in the environment. Consequently, the action $a _ { t }$ is restricted to selecting a subset from this predefined universe U. This simplification creates a consistent label space for the multi-label classifier while remaining sufficient to cover all residents present in the target datasets.

Optimization To make the optimization tractable, we decompose the set-level prediction into individual resident probabilities. We assume that the assignment probability of each resident in the fixed pool U is conditionally independent given the state $s _ { t }$ . This assumption is justified because the state representation $s _ { t }$ (comprising the event and assignment history) is designed to implicitly encode the complex inter-dependencies and context. Consequently, the log-likelihood factorizes as:

![](images/4ccc00a777647bd009dae4d37232e873e11a06dc773c3d6386e689883363a33b.jpg)  
Figure 2: Architecture of the Transformer-based policy network. The model processes the linearized event sequence and outputs resident assignment probabilities via a multi-label classification head.

$$
\log P _ { \theta } ( \mathcal { U } _ { t } \mid s _ { t } ) = \sum _ { u _ { k } \in \mathcal { U } _ { t } } \log P _ { \theta } ( u _ { k } \mid s _ { t } ) + \sum _ { u _ { k } \not \in \mathcal { U } _ { t } } \log ( 1 - P _ { \theta } ( u _ { k } \mid s _ { t } ) ) .\tag{11}
$$

We formalize this using binary labels $y _ { t , k } = \mathbb { 1 } ( u _ { k } \in \mathcal { U } _ { t } )$ , where $y _ { t , k } = 1$ indicates that resident $u _ { k }$ is responsible for the event at step t, and 0 otherwise. Substituting Eq. 11 into the objective yields the multi-label Binary Cross-Entropy (BCE) loss:

$$
\mathcal { L } ( \theta ) = - \sum _ { t = 1 } ^ { T } \sum _ { k = 1 } ^ { K } \Big [ y _ { t , k } \log P _ { \theta } ( \boldsymbol { u } _ { k } \mid \boldsymbol { s } _ { t } ) + ( 1 - y _ { t , k } ) \log ( 1 - P _ { \theta } ( \boldsymbol { u } _ { k } \mid \boldsymbol { s } _ { t } ) ) \Big ] .\tag{12}
$$

Minimizing $\mathcal { L } ( \boldsymbol { \theta } )$ trains the network to act as a set of K parallel binary classifiers. During inference, the simplest decoder constructs the predicted assignment set $\widehat { \mathcal { U } } _ { t }$ by selecting all residents whose predicted probability $P _ { \theta } ( u _ { k } \mid s _ { t } )$ exceeds a decision threshold $\tau \left( \mathbf { e . g . , } \tau = 0 . 5 \right)$ . This threshold-based mechanism naturally aligns with the MDP action space, enabling the agent to output flexible resident combinations based solely on learned confidence scores, ranging from empty sets representing noise to multi-resident groups. We later revisit this decoding rule in Section 5.5, where we promote it from a one-shot per-step decision to a search over assignment trajectories.

## 5.3 Policy Network Architecture

To implement the policy $\pi _ { \theta } .$ , we design a neural network that maps the structured MDP state into resident assignment probabilities. As illustrated in Figure 2, the architecture follows a three-stage processing pipeline: State Encoding, Context Modeling, and the Policy Head.

State Encoding The first module converts the complex, multi-resident state $s _ { t } = ( h _ { < t } , e _ { t } )$ into a unified tensor representation.   
We employ a linearization strategy to flatten the hierarchical structure while preserving semantic boundaries.

Let U be the fixed universe of K potential residents. At step t, with n identified residents, we construct a token sequence $X _ { t }$ by concatenating the local event history of each resident followed by the current event $e _ { t }$ . To distinguish between resident streams, we insert special delimiter tokens $< u _ { k } > .$ . The resulting sequence is:

$$
X _ { t } = \left[ e _ { 1 } ^ { u _ { 1 } } , \ldots , e _ { L } ^ { u _ { 1 } } , < u _ { 1 } > , \quad \ldots , \quad e _ { 1 } ^ { u _ { n } } , \ldots , e _ { L } ^ { u _ { n } } , < u _ { n } > , \quad e _ { t } \right] .
$$

Here, L is the context window size. This sequence is passed to an embedding layer, which projects discrete tokens (sensor events, delimiters, and special markers) into a continuous vector space. These dense embeddings serve as the input to the subsequent sequence model.

![](images/d1017b9e6c56c5c5723957ce3f192797128890699219c97f0527d20d4f271bae.jpg)  
Figure 3: The training data preparation pipeline, illustrating sequence extraction, resident canonicalization, and permutation augmentation.

Context Modeling The core backbone is a stack of Transformer encoder layers designed to capture temporal and semantic dependencies. To maximize training stability and representation power, we adopt a modern Transformer variant incorporating Pre-RMSNorm, SwiGLU activations, and Rotary Positional Embeddings $( \mathrm { R o P E } )$ . These architectural enhancements enable the model to effectively encode relative temporal distances and complex feature interactions. Through the self-attention mechanism, the current event token $e _ { t }$ queries the entire history sequence, aggregating relevant behavioral cues from all identified residents to form a comprehensive decision context.

Policy Head The final module maps the contextualized features to the assignment action space. We extract the hidden state corresponding to the last token position (the current event $\boldsymbol { e } _ { t } )$ as the summary representation $h _ { t }$ . This vector encapsulates the global context necessary for the decision.

We apply a final RMSNorm to $h _ { t }$ and pass it through a linear projection layer to produce logits for all K defined resident indices. The probabilities are computed via a sigmoid activation:

$$
P _ { \theta } ( u _ { k } \mid s _ { t } ) = \sigma ( W _ { k } ^ { \top } h _ { t } + b _ { k } ) , \quad k \in \{ 1 , . . . , K \} .
$$

To handle the dynamic nature of resident discovery within this fixed topology, we interpret the outputs logically: indices 1 to n correspond to known residents, and index $n + 1$ represents a potential new resident. Outputs at indices $k > n + 1$ are masked during inference. This design allows the agent to output flexible resident combinations—including single assignments, multi-resident groups, or new resident discoveries—based strictly on the learned confidence scores.

## 5.4 Training Data Curation

We curate a supervised dataset by transforming raw sensor logs into structured history-label pairs. As illustrated in Figure 3, this process involves sequence extraction, resident canonicalization, and permutation-based augmentation to ensure model robustness.

Sequence Extraction We process the chronological event stream using a sliding window approach. For a session $S =$ $[ e _ { 1 } , \ldots , e _ { T } ]$ , we generate a training instance for every time step t. Events $e _ { 1 : t - 1 }$ constitute the available history, while $e _ { t }$ serves as the prediction target. This creates a dense set of examples reflecting the evolving context at every decision point.

Resident Canonicalization and Grouping To ensure the model learns generalizable behavioral patterns rather than memorizing specific resident IDs, we map residents to a canonical index space. For each instance at step t, we identify the subset of n residents present in the history. These residents are mapped to local indices $\{ 1 , \ldots , n \}$ based on their order of appearance or an arbitrary sequence. We then group historical events by resident and truncate each subsequence to the most recent L events.

Permutation Augmentation A critical challenge in using sequence models for multi-resident data is that the set of residents has no intrinsic order, yet the input $X _ { t }$ requires a sequential concatenation. This introduces an artificial positional bias. To enforce permutation invariance, we employ data augmentation. For each training instance, we generate multiple variations by enumerating random permutations of the n resident blocks in the input sequence. The target label vector is permuted accordingly. This strategy explicitly trains the model to associate the current event $e _ { t }$ with specific behavioral histories regardless of their position in the input tensor, thereby achieving the desired permutation-equivariant behavior.

Label Construction The target label is encoded as a K-dimensional multi-hot binary vector. The i-th entry is set to 1 if the current event belongs to the resident mapped to canonical index i. If the event originates from a resident not present in the history (a new resident), we activate the $( n + 1 ) – \mathrm { t h }$ entry. This representation supports multi-label supervision and aligns with the dynamic resident discovery logic of the network head.

Token Indexing Finally, all symbolic tokens, including sensor events and resident delimiters, are mapped to integers using a global vocabulary. This vocabulary encompasses all distinct sensor states and delimiter types, producing the final tensor indices for the embedding layer.

## 5.5 Inference via Beam Search over Assignment Trajectories

As argued in Section 4, Data Association differs from pure filtering because the current assignment may remain ambiguous and only become clear after several future events provide disambiguating context. Relying solely on the greedy decoder is still insufficient: by committing to a per-step argmax, it discards alternative hypotheses before future evidence arrives. To preserve this deferred-commitment capability, we reformulate inference as a search over assignment trajectories $\mathbf { a } = ( a _ { 1 } , \dots , a _ { T } )$ and use beam search as a tractable approximation that keeps the top-B partial trajectories ranked by accumulated likelihood. When $B { = } 1$ , beam search reduces to the original greedy decoder; when $B { > } 1$ , it retains multiple plausible hypotheses that can be revised as later events arrive, without modifying the learned policy $\pi _ { \theta }$

Formally, at step $t ,$ the multi-label policy yields a per-slot Bernoulli probability $p _ { t } ^ { ( k ) } ~ = ~ P _ { \theta } ( u _ { k } ~ \vert ~ s _ { t } )$ . By the same conditional-independence factorization as in Eq. 11, the joint probability of any candidate action $a _ { t } \subseteq \mathbb { U }$ can be written as

$$
P _ { \theta } ( a _ { t } \mid s _ { t } ) = \prod _ { u _ { k } \in a _ { t } } p _ { t } ^ { ( k ) } \cdot \prod _ { u _ { k } \notin a _ { t } } \bigl ( 1 - p _ { t } ^ { ( k ) } \bigr ) ,\tag{13}
$$

and a complete trajectory is scored by its accumulated log-likelihood $S ( \mathbf { a } ) \mathbf { \Psi } = \sum _ { t = 1 } ^ { T }$ log $P _ { \theta } ( a _ { t } \ \mid \ s _ { t } )$ . At each step, every surviving partial trajectory in the current beam expands its frontier by enumerating the non-empty subsets of slots whose marginal probability $p _ { t } ^ { ( k ) }$ exceeds a pruning threshold τ (which keeps the branching factor bounded). Each child applies a deterministic state update that appends $e _ { t }$ to the history of every assigned resident and accumulates the corresponding log $P _ { \theta } { \left( a _ { t } \mid s _ { t } \right) }$ into its score, and the global frontier is then truncated to the top-B partial trajectories ranked by S(·). After all $T$ events have been processed, the trajectory with the highest accumulated score is returned as the final attribution.

## 6 LLM as a Generative Policy

While the scratch-trained architecture described in Section 5.3 effectively captures statistical dependencies, it is inherently limited by its reliance on dataset-specific sensor IDs. Such models tend to overfit to the fixed topology of the training environment and struggle to generalize to unseen interaction patterns or heterogeneous sensor layouts. To overcome this limitation, we propose an alternative policy implementation leveraging Large Language Models (LLMs). By grounding the data association task in natural language, we exploit the pre-trained world knowledge of LLMs to achieve robust generalization beyond mere statistical correlation. For instance, the model can leverage the semantic relationship between a motion sensor in the kitchen and cooking activities without requiring extensive training from scratch.

Figure 4 illustrates our proposed framework. We reformulate the problem from a multi-label classification task into a supervised conditional text generation task. The workflow begins by serializing the environmental priors and interaction history into a semantic prompt. This context conditions the LLM to directly generate the resident assignment in a concise JSON format. Through this paradigm, the model learns to associate events based on universal semantic logic rather than rigid ID matching This significantly enhances its ability to handle ambiguity and dynamic environmental changes.

![](images/b9095a9914a64585a7e44e43be20dc50001ed4ff0e89a3406f4a70077edd946c.jpg)  
Figure 4: Overview of the LLM-based policy architecture. The discrete MDP state is serialized into a textual prompt, and the model isfine-tuned to generate the target resident assignment directly.

## 6.1 Generative Policy Learning

To enable the model to comprehend the physical environment and generalize across heterogeneous sensor layouts, we augment the dynamic MDP state $s _ { t }$ with static environmental context c. This context includes invariant priors such as the house layout and sensor descriptions. Adhering to the generative paradigm, we serialize both the state $s _ { t }$ and context c into a composite textual prompt $\mathcal { X } _ { t }$ . The policy $\pi _ { \theta } .$ , parameterized by the LLM, learns to map this prompt to a target sequence $\mathcal { V } _ { t } = \{ y _ { t , 1 } , \dotsc , y _ { t , L } \}$ that explicitly encodes the resident assignment decision in a structured JSON format (e.g., {"subject": "User A, User B"}).

Consistent with the Behavior Cloning framework established in Section 5, we treat the Supervised Fine-Tuning (SFT) of the LLM as a direct policy imitation process. Adhering to the generative paradigm, we serialize both the state $s _ { t }$ and context c into a composite textual prompt $\mathcal { X } _ { t }$ . The policy $\pi _ { \theta }$ , parameterized by the LLM, learns to map this prompt to a target sequence $\mathcal { V } _ { t } = \{ y _ { t , 1 } , \dotsc , y _ { t , L } \}$ that explicitly encodes the resident assignment decision in a structured JSON format (e.g., {"subject": "User A, User B"}).

In this framework, the policy defines a conditional distribution over the target token sequence:

$$
\pi _ { \boldsymbol { \theta } } ( a _ { t } \mid s _ { t } ) \equiv P _ { \boldsymbol { \theta } } ( \mathcal { V } _ { t } \mid \mathcal { X } _ { t } ) .\tag{14}
$$

To make optimization tractable, we decompose this joint probability autoregressively:

$$
P _ { \theta } ( \mathcal { V } _ { t } \mid \mathcal { X } _ { t } ) = \prod _ { i = 1 } ^ { L } P _ { \theta } ( y _ { t , i } \mid \mathcal { X } _ { t } , y _ { t , < i } ) ,\tag{15}
$$

where $y _ { t , < i }$ denotes the history of generated tokens for the current response.

We optimize the model parameters θ by minimizing the negative log-likelihood (NLL) of the generated tokens, corresponding to the standard Next Token Prediction (NTP) objective. This objective is mathematically equivalent to minimizing the divergence between the learner and expert distributions as defined in classical Behavior Cloning:

$$
\mathcal { L } ( \theta ) = - \sum _ { t = 1 } ^ { T } \sum _ { i = 1 } ^ { L } \log P _ { \theta } ( y _ { t , i } \mid \mathcal { X } _ { t } , y _ { t , < i } ) .\tag{16}
$$

During training, we employ teacher forcing by providing the ground-truth prefix $y _ { t , < i }$ to predict the next token. Crucially, a loss mask is applied to the prompt tokens $\mathcal { X } _ { t }$ . This ensures that gradients are back-propagated exclusively from the prediction of the target resident assignment.

## 6.2 Prompt Engineering

To enable the language model to process sensor data, we employ a structured serialization strateg, mapping the numerical state $s _ { t }$ and context c into a structured textual input $\mathcal { X } _ { t } .$ Similar to general prompt designs, the schema consists of a static system prompt and a dynamic user prompt.

User Prompt The user prompt represents the dynamic state $s _ { t }$ . As shown in our prompt design, it serializes the current sensor reading $e _ { t }$ (including timestamp, sensor ID, and location) and the recent history of residents $h _ { < t }$ into a human-readable format. It further instructs the LLM to output the final assignment in a strict JSON format, so that our system can automatically parse the response and update the state. Taking the MARBLE dataset as an example, the final user prompt used in this study is shown as following:

User Prompt for MARBLE dataset used in our proposed method   
Observed users:   
• User\_1 (latest 1 events):   
[12:14:44] P4 ON   
You can assign this event to:   
1. An existing user (from the list above)   
2. A new user (User\_2) if this event belongs to someone not yet observed   
Current sensor event observation:   
{   
"time": "12:14:44",   
"event": "P1 ON",   
"sensor": "P1",   
"action": "ON",   
"location": "dining room"   
}   
Based on the historical patterns and location constraints, which user(s) triggered this event?   
Output your prediction in JSON format: {{"subject": "User\_X"}} or {{"subject": "User\_X, User\_Y"}} for multiple users.

System Prompt The system prompt encodes the static context $c , ,$ which provides static background knowledge that remains fixed across all inference steps. It defines the environment in which reasoning takes place, including: (1) the general background and objective of the task; (2) the spatial layout of the smart home and inter-room connectivity; (3) the mapping between sensor identifiers and physical objects; and (4) key behavioral and physical constraints that govern user mobility and spatial exclusivity. This structured global context serves as a static prior that grounds the model’s reasoning in the spatial, semantic, and physical realities of the environment. Taking the MARBLE dataset as an example, the final system prompt use in this study is shown as following on next page.

System Prompt for MARBLE dataset used in our proposed method

## [Background]

You are in a smart home where some sensors are installed. There are at most four people living in the house. They engage in daily activities at home. Their actions in the house can trigger the sensors and be recorded as sensor events. Patterns in sensor event sequences help you infer their movement and actions.

## [Objective]

Given the layout of the house and the description of the sensors, you need to analyse step by step the input sensor event sequences to determine their actions and tracks with the time.

## [House Layout]

Start at the Hall (bottom-left). The Hall has two openings only: north into the Medicine Area, and east into the Kitchen. There is no direct Hall access to the Living Room, Dining Room, or Office.

From the Hall → Medicine Area → Living Room (east from Medicine Area) → Office (open passage to the east).

From the Hall → Kitchen → Dining Room (fully open to the east).

Overall arrangement by bands:

top row: Medicine Area (left), Living Room (center), Office (right);

bottom row: Hall (left), Kitchen (center), Dining Room (right). No other direct connections exist.

Each room contains specific facilities and sensors. The Hall serves as the entrance and passage area, with no sensors. The Medicine Area includes a medicine cabinet equipped with a medicine magnetic sensor (R6) to detect door openings. The Living Room contains 4 couches, each with a pressure mat (P3, P7, P8, P9). Face to the couches, there is a television connected to a smart plug (E2). Next to the living room, there is an office with a dest and an office chair (P2). The Kitchen, located in the center, includes several sensors: a cooker smart plug (E1) for the stove, a magnetic sensor (R5) for the pot cabinet, a magnetic sensor (R7) for the refrigerator door, a magnetic sensor (R1) for the pantry, and a magnetic sensor (R2) for the cutlery drawer. The Dining Room features a dining table with four dining chairs (P1, P4, P5, P6) and is designed as an open dining space connected to the Kitchen.

## [Sensor Description]

SENSOR EXPLANATIONS (In column order)

<table><tr><td>Sensor ID</td><td>Sensor Type</td><td>Description</td></tr><tr><td>R1</td><td>Magnetic</td><td>pantry</td></tr><tr><td>R2</td><td>Magnetic</td><td>cutlery drawer</td></tr><tr><td>R5</td><td>Magnetic</td><td>pots drawer</td></tr><tr><td>R6</td><td>Magnetic</td><td>medicines cabinet</td></tr><tr><td>R7</td><td>Magnetic</td><td>fridge</td></tr><tr><td>E1</td><td>Smart Plug</td><td>stove</td></tr><tr><td>E2</td><td>Smart Plug</td><td>television</td></tr><tr><td>P1</td><td>Pressure Mat</td><td>dining room chair</td></tr><tr><td>P2</td><td>Pressure Mat</td><td>office chair</td></tr><tr><td>P3</td><td>Pressure Mat</td><td>living room couch</td></tr><tr><td>P4</td><td>Pressure Mat</td><td>dining room chair</td></tr><tr><td>P5</td><td>Pressure Mat</td><td>dining room chair</td></tr><tr><td>P6</td><td>Pressure Mat</td><td>dining room chair</td></tr><tr><td>P7</td><td>Pressure Mat</td><td>living room couch</td></tr><tr><td>P8 P9</td><td>Pressure Mat</td><td>living room couch</td></tr><tr><td></td><td>Pressure Mat</td><td>living room couch</td></tr></table>

## [Key Constraints]

When processing each sensor event step by step, you must carefully consider location constraints:

\- Always check each user’s recent activities to determine their current location status

\- Consider time sequence and movement feasibility when assigning events

\- If a user is already at an exclusive location (pressure mat, bathroom, etc.), they cannot be at a different exclusive location simultaneously

Based on the historical activity patterns of each user and the current sensor event, predict which user(s) triggered the current event.   
Output user indices (1-indexed) as a multi-label prediction.

## 6.3 Inference

At deployment, the LLM policy is decoded with standard greedy autoregressive generation: at each event step t, the prompt X is fed to the fine-tuned model, which emits a JSON response that is parsed into the assignment action a and used to update the state to $s _ { t + 1 }$ . Unlike the multi-label policy of Section 5, the LLM does not expose a closed-form per-slot probability vector $( p _ { t } ^ { ( 1 ) } , \dots , p _ { t } ^ { ( K ) } )$ ; recovering the joint probability of an arbitrary action $a _ { t }$ would require evaluating the likelihood of every candidate JSON string under the model, which scales poorly with the number of residents. We therefore do not apply the trajectory-level beam search of Section 5.5 to the LLM policy in this work, and treat the comparison between the two policies under their respective native decoders as a faithful reflection of how each is intended to be deployed.

## 7 Experimental Evaluation

## 7.1 Datasets

Among the mainstream multi-resident ambient sensor datasets are CASAS [36], ARAS [4], MARBLE [6], and the recently introduced MuRAL dataset [14]. Notably, the ARAS dataset does not include resident identity annotations for individual sensor events, making it unsuitable for evaluating the Data Association task. As a result, ARAS is excluded from our experimental evaluation.

## 7.1.1 CASAS ADLMR Dataset

CASAS ADLMR [36] is a two-person activity recognition dataset. It was collected over 26 sessions in a home equipped with 37 ambient sensors. These sensors are primarily ceiling-mounted PIR motion sensors, which are triggered by movement within approximately one meter, along with item sensors used to indicate whether objects such as medications remain in their original locations. For each session, a different pair of residents performed a series of 15 activities. Each event triggered by an sensor is further annotated with one or two corresponding resident IDs and their activities. Finally, we note that CASAS is dominated by densely deployed ceiling-mounted motion sensors, whose activations are difficult to describe faithfully in natural language (e.g., lacking object- or location-specific semantics). Therefore, we do not run any LLM-based methods on CASAS in our experiments.

## 7.1.2 MARBLE Dataset

The MARBLE dataset [6] consists of 76 sessions representing different scenarios, with each sessions potentially involving 1 to 4 residents. It was collected in an environment equipped with 16 ambient sensors, including 5 magnetic sensors, 2 smart plugs and 9 pressure mat. During data collection, residents were instructed to perform up to 13 different activities in a scripted sequence. Unlike the CASAS ADLMR datasets, the MARBLE dataset incorporates resident-specific wearable devices such as micro-localization, smartwatch, and phone usage sensors in addition to environmental sensors. However, since our method does not consider these devices, we use only the ambient sensor data in our experiments. In MARBLE, each sensor event is attributed to exactly one person, resulting in a one-to-one correspondence.

## 7.1.3 MuRAL Dataset

The MuRAL dataset [14] is a multi-resident smart home dataset comprising 21 sessions, each involving 2 to 4 participants and instrumented with 23 ambient sensors (5 smart plugs, 10 magnetic sensors, 6 PIR motion sensors) . Every sensor event in the dataset is annotated with the identities of the triggering residents, one of 27 predefined activity categories, and a natural language description. Compared to existing datasets such as CASAS and MARBLE, MuRAL offers distinct advantages in the data association task by capturing dynamic resident entry and exit, and by allowing unscripted, naturalistic behaviors. This design results in richer and more complex activity patterns, better reflecting real-world multi-resident scenarios.

## 7.2 Evaluation Metrics

## 7.2.1 Resident Assignment Accuracy (Acc)

Prior evaluations of data association often suffer from global bias and cascading error effects. Some studies [13, 33] assess performance only indirectly via downstream activity recognition, while others [17, 19, 24] evaluate identity assignments based on a global one-to-one mapping between predicted and ground truth labels in a small chunk. However, since our method can process the entire sequence of a session, our evaluation scope is significantly larger. Once this mapping is misaligned—especially in the early part of a session—all subsequent predictions may be incorrectly penalized, even if identity tracking later becomes accurate.

To mitigate this issue, we propose a window-based evaluation protocol that enables localized assessment of identity consistency. The key idea is to decouple the prediction phase from the evaluation phase: we first perform data association over the entire session to produce a complete sequence of predicted identity labels. We then slide a fixed-length window (by default, the window size w is 16, the stride is 1) across the session and assess identity tracking consistency within each window.

In each window, we extract the resident labels from both the ground truth and predicted sequences, and assign them symbolic codes $( \mathbf { A } , \mathbf { B } , \mathbf { C } , \ldots )$ based on their order of first appearance. This yields a one-to-one mapping between raw labels and symbolic identifiers within the window. We then relabel both the ground truth and prediction sequences using this mapping, and compute a per-event consistency score. The score for the window is the average consistency across all steps, and the final tracking score is the mean over all windows.

This evaluation strategy preserves temporal context from the full-session prediction, while preventing early errors from dominating the accuracy measurement. It provides a robust means to analyze cascading error, as performance degradation under increasing window sizes reflects how well a method maintains identity continuity over time.

## 7.2.2 Correct Run Length (CRL)

To complement the accuracy metric, we introduce Correct Run Length (CRL) as an additional evaluation measure. CRL directly reflects the quality of resident tracking by quantifying how long a model can continuously maintain correct identity assignments before the first error occurs. Specifically, for each sliding window and for each ground-truth resident, we count the length of the initial consecutive segment of correct predictions until the first mistake. These lengths are then averaged across all residents and all windows to produce the CRL score. A larger CRL indicates that the model sustains more stable and coherent tracking of residents.

To contextualize CRL, we also report, for reference, the average number ofevents per resident. This auxiliary value allows readers to interpret CRL in terms of the typical sequence length over which the tracking is evaluated, but it is not used as a primary metric.

## 7.2.3 BLEU

We adopted the Bilingual Evaluation Understudy (BLEU) [29] to assess the consistency of the seperated sequence. For any mixed sequence, we connect the separated sequence generated by the model using a delimiter (an <EOS> token) to form a candidate sequence. Similarly, we construct a reference sequence using the ground truth in the same way. Given a candidate sequence and a reference sequence, the BLEU metric considers the precision of N-grams, which is the proportion of N-gram phrases in the candidate sequence that appear in the reference sequence. This evaluation method focuses more on assessing whether the results of the model’s separation are coherent and consistent. In our subsequent experiments, we divide the mixed sequence into segments of length 30 and evaluate the BLEU score for each segment. Finally, we compute the average BLEU score across all segments.

## 7.2.4 Resident Count Error (RCE)

Since BLEU and Resdient Assignment Accuracy cannot directly reflect the model’s real-time assessment of the number of residents, we designed a dedicated error metric for the number of residents in order to systematically evaluated the estimation of the number of individuals in the results. This metric is introduced for the first time in related research, as previous methods lacked the ability to dynamically evaluate the number of residents. To compute this metric, we divide the mixed sequence into segments of 30 events. We chose 30 as the segment length which is sufficient to involve multiple residents while still keeping the local representation. For each segment, we count how many different residents the model assigns the L events to, obtaining a predicted value $\hat { K }$ . This predicted value is then compared with the ground truth $K ,$ , and the absolute error for each segment is calculated. Finally, the average of these absolute errors across all segments is computed.

The error metric can be expressed as:

$$
\mathrm { R C E } = \frac { 1 } { S } \sum _ { i = 1 } ^ { S } \left| K _ { i } - \hat { K } _ { i } \right|
$$

where $S$ is the total number of segments, $\hat { K } _ { i }$ is the predicted number of residents for the i-th segment, and $K _ { i }$ is the ground truth number of residents for the i-th segment.

## 7.3 Experimental Setup

We conduct experiments on the three benchmark datasets described above (CASAS [36], MARBLE [6], and MuRAL [14]) to compare our proposed method with sMRT [39], LAHAR [13] which is powered by GPT-4o, and the naive “All-to-One” baseline which assumes that all events belong to a single resident. To conduct these experiments, we randomly split the sessions of each datasets into training, validation, and test sets with a ratio of 7:1:2. For our proposed methods, we refer to the Transformer-onl policy trained from scratch via Behavior Cloning as BC, and the approach that uses an LLM as the policy as LLM+BC.

## 7.3.1 Setting of BC

The BC method utilizes a Transformer encoder, consisting of 3 layers with a model dimension $( d _ { \mathrm { m o d e l } } )$ of 128, 8 attention heads, and a feed-forward dimension of 512. Training is performed with a batch size of 512, a learning rate of $1 0 ^ { - 4 }$ , and a dropout rate of 0.2. These model architectures and training hyperparameters were chosen based on preliminary trials on the validation set, without extensive hyperparameter tuning, to demonstrate the robustness and generalizability of the proposed method. However, to accommodate the varying complexity and density of the datasets, we adjusted the training duration and the resident history window size (L) for each dataset. Specifically, the MARBLE dataset was trained for 200 epochs with a history window of $L = 1 1$ . In contrast, both the CASAS and MuRAL datasets were trained for 100 epochs, with history windows set to $L = 9$ and $L = 1$ , respectively. These hyperparameters were obtained via grid search on the validation set, and the search procedure and its impact will be presented in the following sections. As for inference, we adopt greedy decoding by default, corresponding to a beam size of 1, for a fairer comparison with prior work as well as with the LLM+BC variant under its native decoding setup, and apply a fixed multi-label decision threshold of $\tau = 0 . 5$ across all datasets. These settings are used as standard defaults without further tuning, in order to avoid dataset-specific optimization and to ensure fair and reproducible evaluation.

## 7.3.2 Setting of LLM+BC

We employ the Qwen3-4B-Instruct-2507 as the foundation model for all experiments. To enable parameter-efficient fine-tuning, we adopt Low-Rank Adaptation (LoRA) [25]. The LoRA configuration consists of a rank $r = 1 6 ,$ a scaling factor $\alpha = 3 2 .$ and a dropout rate of 0.05. This adaptation is applied to all linear projections within both the attention mechanisms and feedforward networks. Training is performed using Bfloat16 (bf16) precision to ensure computational efficiency. All experiments were conducted using the Hugging Face Transformers Trainer framework and the Hugging Face PEFT library for the LoRA implementation. The computational setup consisting of two NVIDIA A100 (80GB) GPUs. To leverage hardware parallelism, we employed the Accelerate library for Distribued Data Parallel (DDP) training. The model was fine-tuned for 1 epoch with a maximum sequence length of 1024 tokens given the length of the prompt is about 900 tokens. We set the per-device batch size to 6 with gradient accumulation steps of 4, resulting in an effective global batch size of $4 8 \ : ( 6 \times 2 \times 4 )$ Optimization was performed using a cosine learning rate scheduler with a peak learning rate of $2 \times 1 0 ^ { - 4 }$ and a warmup ratio of 0.1, regularized by a weight decay of 0.01. For inference, we configure the generation process to produce a maximum of 50 new tokens. We align our history context lengths L with those found in BC method, using a window of 1 for MuRAL and 11 for MARBLE.

## 7.4 Qualitative Examples

Table 1: Qualitative example of the resident assignment performed by our LLM+BC method on 33 sequential sensor events from Session 12 of the MuRAL dataset. Row colors indicate ground truth residents: green (A), blue (B), and orange (C). Red rows represent incorrect predictions.
<table><tr><td>Step</td><td>Time</td><td>Event</td><td>Facility</td><td>Predicted ID</td><td>GT ID</td><td>Correct</td></tr><tr><td>0</td><td>19:04:24</td><td>OPEN</td><td>Main Door</td><td>A, B</td><td>A, B</td><td>√</td></tr><tr><td>1</td><td>19:04:30</td><td>CLOSE</td><td>Main Door</td><td>A, B</td><td>A, B</td><td>√</td></tr><tr><td>2</td><td>19:04:35</td><td>ON</td><td>Living Room Couch</td><td>A</td><td>A</td><td>√</td></tr><tr><td>3</td><td>19:04:36</td><td>OPEN</td><td>Bedroom 1 Door</td><td>B</td><td>B</td><td>√</td></tr><tr><td>4</td><td>19:04:41</td><td>OFF</td><td>Living Room Couch</td><td>A</td><td>A</td><td>√</td></tr><tr><td>5</td><td>19:04:42</td><td>ON</td><td>Bathroom Global</td><td>B</td><td>A</td><td>x</td></tr><tr><td>6</td><td>19:04:42</td><td>CLOSE</td><td>Bedroom 1 Door</td><td>A</td><td>B</td><td></td></tr><tr><td>7</td><td>19:04:46</td><td>ON</td><td>Living Room Couch</td><td>A</td><td>B</td><td></td></tr><tr><td>8</td><td>19:04:49</td><td>CLOSE</td><td>Toilet Contact</td><td>B</td><td>A</td><td></td></tr><tr><td>9</td><td>19:04:52</td><td>OFF</td><td>Living Room Couch</td><td>A</td><td>B</td><td></td></tr><tr><td>10</td><td>19:05:00</td><td>OFF</td><td>Bathroom Global</td><td>B</td><td>A</td><td></td></tr><tr><td>11</td><td>19:05:26</td><td>ON</td><td>Bathroom Global</td><td>B</td><td>A</td><td></td></tr><tr><td>12</td><td>19:05:37</td><td>OPEN</td><td>Toilet Contact</td><td>B</td><td>A</td><td></td></tr><tr><td>13</td><td>19:05:44</td><td>ON</td><td>Bathroom Sink</td><td>B</td><td>A</td><td></td></tr><tr><td>14</td><td>19:05:52</td><td>OFF</td><td>Bathroom Sink</td><td>B</td><td>A</td><td></td></tr><tr><td>15</td><td>19:06:04</td><td>OFF</td><td>Bathroom Global</td><td>B</td><td>A</td><td></td></tr><tr><td>16</td><td>19:06:13</td><td>OPEN</td><td>Fridge Door</td><td>B</td><td>A</td><td></td></tr><tr><td>17</td><td>19:06:14</td><td>CLOSE</td><td>Fridge Door</td><td>B</td><td>A</td><td></td></tr><tr><td>18</td><td>19:07:58</td><td>OPEN</td><td>Dining Chair 2</td><td>B</td><td>A</td><td></td></tr><tr><td>19</td><td>19:08:06</td><td>ON</td><td>Living Room Couch</td><td>A</td><td>B</td><td></td></tr><tr><td>20</td><td>19:08:10</td><td>OFF</td><td>Living Room Couch</td><td>A</td><td>B</td><td></td></tr><tr><td>21</td><td>19:08:16</td><td>OPEN</td><td>Dining Chair 3</td><td>A</td><td>B</td><td>–</td></tr><tr><td>22</td><td>19:11:49</td><td>OPEN</td><td>Main Door</td><td>C</td><td>C</td><td>√</td></tr><tr><td>23</td><td>19:11:53</td><td>CLOSE</td><td>Main Door</td><td>C</td><td>C</td><td>√</td></tr><tr><td>24</td><td>19:12:06</td><td>OPEN</td><td>Bedroom 2 Door</td><td>C</td><td>C</td><td>√</td></tr><tr><td>25</td><td>19:12:09</td><td>CLOSE</td><td>Dining Chair 2</td><td>B</td><td>A</td><td>1</td></tr><tr><td>26</td><td>19:12:10</td><td>CLOSE</td><td>Dining Chair 3</td><td>A</td><td>B</td><td>1</td></tr><tr><td>27</td><td>19:12:12</td><td>CLOSE</td><td>Bedroom 2 Door</td><td>C</td><td>C</td><td>√</td></tr><tr><td>28</td><td>19:12:21</td><td>ON</td><td>Kitchen Sink</td><td>B</td><td>A</td><td>=</td></tr><tr><td>29</td><td>19:12:24</td><td>ON</td><td>Kitchen Counter</td><td>B</td><td>B</td><td>√</td></tr><tr><td>30</td><td>19:12:43</td><td>OFF</td><td>Kitchen Sink</td><td>B</td><td>A</td><td>x</td></tr><tr><td>31</td><td>19:12:51</td><td>ON</td><td>Stove</td><td>B</td><td>A</td><td>-</td></tr><tr><td>32</td><td>19:13:07</td><td>OPEN</td><td>Bedroom 2 Door</td><td>C</td><td>C</td><td>√</td></tr></table>

Table 1 illustrates the resident assignment performance of our proposed LLM+BC method, using a sequential excerpt of 33 sensor events from Session 12 of the MuRAL dataset.

While the system correctly initializes tracks for Residents A and B upon their simultaneous entry (Steps 0–3), a critical identity switch occurs at Step 5. We hypothesize that this early failure stems from the model’s rigid interpretation of temporal constraints. As shown in the log, the Living Room Couch turns OFF at 19:04:41 (Step 4), and the Bathroom turns ON merely one second later at 19:04:42 (Step 5). The LLM likely inferred that it was physically infeasible for Resident A to traverse from the living room to the bathroom within such a negligible time window. Consequently, to satisfy this apparent physical constraint, the model assigned the bathroom activity to Resident B, leading to a persistent identity swap for the subsequent events. In reality, the sensor timestamps did not perfectly align with physical movements; Resident A had placed a bag on the couch at Step 2 and immediately departed. The OFF event at Step 4 was merely a delayed deactivation caused by the sensor’s timeout mechanism, which the model failed to account for.

Table 2: Comparison ofData Association performances ofbaseline approaches and our approach on 3 datasets. Acc and CRL are calculated using a window size w = 16.
<table><tr><td>Dataset</td><td colspan="4">CASAS [36]</td><td colspan="4">MARBLE [6]</td><td colspan="4">MuRAL [14]</td></tr><tr><td>Metric</td><td>Acc</td><td>CRL</td><td>BLEU</td><td>RCE</td><td>Acc</td><td>CRL</td><td>BLEU</td><td>RCE</td><td>Acc CRL</td><td>BLEU</td><td></td><td>RCE</td></tr><tr><td>All-to-One</td><td>0.580</td><td>5.372</td><td>0.565</td><td>0.995</td><td>0.729</td><td>8.606</td><td>0.707</td><td>0.789</td><td>0.551</td><td>4.489</td><td>0.510</td><td>1.593</td></tr><tr><td>sMRT [39]</td><td>0.561</td><td>3.051</td><td>0.530</td><td>1.257</td><td>0.119</td><td>0.641</td><td>0.079</td><td>15.10</td><td>0.529</td><td>3.321</td><td>0.497</td><td>1.560</td></tr><tr><td>LAHAR [13]</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0.730</td><td>8.231</td><td>0.747</td><td>0.383</td><td>0.616</td><td>4.375</td><td>0.736</td><td>0.571</td></tr><tr><td>BC</td><td>0.757</td><td>6.234</td><td>0.727</td><td>0.021</td><td>0.792</td><td>8.710</td><td>0.797</td><td>0.136</td><td>0.728</td><td>5.476</td><td>0.774</td><td>0.437</td></tr><tr><td>LLM+BC</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0.824</td><td>8.783</td><td>0.841</td><td>0.136</td><td>0.773</td><td>6.034</td><td>0.799</td><td>0.261</td></tr></table>

A second challenge arises during the interaction in the kitchen area (Steps 25–31), illustrating the difficulty of tracking concurrent activities in shared zones. At Steps 25 and 26, distinct sensors on Dining Chairs 2 and 3 trigger CLOSE events almost simultaneously, indicating that two individuals stood up to move elsewhere. However, the model fails to register this as a dual transition. When activities subsequently occur in the kitchen (Sink, Counter, and Stove at Steps 28–31), the model attributes them solely to a single resident (Predicted ID: B). This suggests that the model overlooked the cue provided by the simultaneous chair events, failing to realize that both residents had entered the kitchen. Consequently, it collapses the multi-resident scenario into a single-resident track, highlighting the complexity of distinguishing identities when residents occupy the same confined functional area.

Despite these challenges with Residents A and B, the system demonstrates robustness in handling cardinality changes. When Resident C enters the facility at Step 22, the model successfully distinguishes this new individual from the existing entangled tracks. The events associated with Resident C (Steps 22–32) are tracked with perfect accuracy, indicating that the reasoning engine remains effective for distinct, unambiguous trajectories.

## 7.5 Quantitative Comparison against Baselines

Table 2 presents the performance comparison between our proposed methods and the baselines. A decisive advantage of our approaches, BC and LLM+BC, is first observed in the Resident Count Error (RCE), where they demonstrate a substantial improvement over the All-to-One and sMRT [39] baselines across all datasets. This contrast is most distinct on the MARBLE dataset: while the sMRT filter suffers from error covariance divergence due to sparse sensor data, resulting in a catastrophic RCE of 15.10, our methods maintain exceptional stability with an RCE as low as 0.136. This highlights the capability of our MDP-based models to dynamically estimate the number of active residents, whereas traditional filtering algorithms tend to erroneously assign events to independent identities under high uncertainty.

Significantly, our approach outperforms the recent LLM-driven baseline, LAHAR [13], despite a massive disparity in foundation model scale. While LAHAR relies on the state-of-the-art commercial model GPT-4o, our LLM+BC framework utilizes the compact, open-source qwen3-4B. Notwithstanding this resource constraint, our method achieves superior performance across all metrics. This advantage is particularly pronounced on the complex MuRAL dataset, where LLM+BC surpasses LAHAR by a wide margin in Accuracy (0.773 vs. 0.616) and reduces the RCE by over 50% (0.261 vs. 0.571). These results suggest that our structured MDP formulation provides more effective grounding for sensor data than relying solely on the generalized reasoning capabilities of extremely large language models.

Comparing our two proposed methods reveals the specific contribution of the semantic module. While pure BC is effective in simpler settings, the LLM+BC framework provides a critical advantage in complex scenarios like MuRAL. This is bes reflected in the Correct Run Length (CRL) and BLEU metrics. On MuRAL, incorporating the qwen3-4B model improves the CRL from 5.476 to 6.034. Since CRL measures the length of consecutive correct predictions, this increase signifies that the semantic reasoning of the LLM significantly enhances tracking stability, sustaining correct identity assignments for longer durations. Similarly, the improvement in BLEU score (from 0.774 to 0.799) indicates that the sequences generated by LLM+BC possess higher coherence and better align with the N-gram structure of ground truth behaviors.

## 7.6 Impact of Policy Training

Our proposed MDP-based framework relies critically on the learned policy model to make sequential resident-assignment decisions. To validate the importance and effectiveness of this policy component, we study how BC policy models at different training stages affect the final Data Association performance. Figure 5 plots the test-set F1 score of the policy model together with the Data Association evaluation metrics (Acc, BLEU, RCE, and CRL) obtained during inference for different training epochs on the three datasets.

![](images/95329c61d7c6591d35cd2fb8852ba7e0289e0fc3a876c905549642fada1226bb.jpg)  
Figure 5: Performance evolution ofthe policy network across training epochs on all datasets.

![](images/5534083abeabeda6affe2f427f7f9d6c47fcc6961f172743f83c7b3674c16fbe.jpg)  
(a) CASAS

![](images/4e43f4fa15a371d2677705655175041b9bba6b7283cc1c404f33f13f70ee7a85.jpg)  
(b) MARBLE

![](images/f70600a2296d84d68167eb50974e85fa33be61a27d896787469a71dcf2b8151f.jpg)  
(c) MuRAL  
Figure 6: Resident assignment accuracy as afunction ofthe evaluation window size (log scale) on the CASAS, MARBLE, and MuRAL datasets.

Across all datasets, we observe that the policy network’s classification performance generally improves as training progresses: the test-set F1 score exceeds 0.8 in all cases and approaches 0.9 on both CASAS and MuRAL. This trend suggests that the mapping from multi-resident contextual event representations to resident-assignment actions is learnable under the current dataset scale and model capacity. Moreover, as the test-set F1 score increases, the downstream Data Association metrics (Acc, BLEU, RCE, and CRL) improve in tandem, indicating a strong linkage between per-step policy accuracy and sequence-level association quality in the MDP inference process. Overall, these results support both our modeling formulation and the effectiveness of the BC-based training paradigm.

Beyond this overall trend, each dataset exhibits characteristic learning dynamics. On CASAS, policy F1 and BLEU generally increase with training, whereas RCE and CRL show noticeable non-monotonic fluctuations before epoch 15. We attribute these fluctuations to unstable early-stage policies (e.g., collapsing most events to a single resident), which can distort both resident-count estimates and run-length statistics. MARBLE shows faster early saturation in Acc and BLEU, likely because a large fraction of its sessions involve only one resident, making the underlying classification problem comparatively simpler. In contrast, CRL improves more gradually along with test-set F1, suggesting that sustaining longer streaks of correct identity assignments requires a more precisely trained policy, even when average accuracy has largely converged.

Overall, these results further confirm that improving the policy network translates into better sequence-level Data Association performance.

Table 3: Impact of beam-search inference on the Data Association performance of our method. "Greedy" denotes standard decoding with beam size 1, and $" B \bar { S } k "$ denotes beam search with beam size k.
<table><tr><td>Dataset</td><td>Metric</td><td>Greedy</td><td>BS2</td><td>BS4</td><td>BS8</td><td>BS 12</td></tr><tr><td rowspan="3">CASAS [36]</td><td>Acc CRL</td><td>0.757</td><td>0.758</td><td>0.760</td><td>0.759</td><td>0.759</td></tr><tr><td>BLEU</td><td>6.234</td><td>6.579</td><td>6.811</td><td>6.797</td><td>6.666</td></tr><tr><td>RCE</td><td>0.727 0.021</td><td>0.736 0.011</td><td>0.743 0.005</td><td>0.742</td><td>0.738 0.010</td></tr><tr><td rowspan="3">MARBLE [6]</td><td>Acc</td><td>0.792</td><td>0.797</td><td>0.795</td><td>0.005 0.795</td><td>0.794</td></tr><tr><td>CRL</td><td>8.710</td><td>8.765</td><td>8.761</td><td>8.760</td><td>8.756</td></tr><tr><td>BLEU</td><td>0.797</td><td>0.809</td><td>0.809</td><td>0.808</td><td>0.801</td></tr><tr><td rowspan="3">MuRAL [14]</td><td>RCE Acc</td><td>0.136</td><td>0.113</td><td>0.119</td><td>0.120</td><td>0.123</td></tr><tr><td>CRL</td><td>0.728</td><td>0.733</td><td>0.744</td><td>0.747</td><td>0.747</td></tr><tr><td></td><td>5.476</td><td>5.628</td><td>5.761</td><td>5.822</td><td>5.833</td></tr><tr><td rowspan="3"></td><td>BLEU</td><td>0.774</td><td>0.779</td><td>0.782</td><td>0.787</td><td>0.788</td></tr><tr><td>RCE</td><td>0.437</td><td>0.421</td><td>0.376</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>0.312</td><td>0.313</td></tr></table>

## 7.7 Impact of Error Accumulation

Data Association is inherently a sequential decision-making problem: a model assigns an identity to each incoming event conditioned on its current belief about all residents. As a result, mistakes are difficult to fully avoid and can compound over time. Once an early event is assigned to an incorrect resident, the resident states maintained by the model may deviate from the true states, and subsequent decisions are made under this shifted input distribution, leading to cascading errors.

To assess robustness to this error accumulation, we plot Acc as a function of the evaluation window size w (log scale) in Figure 6. Larger w corresponds to a longer temporal horizon over which the predicted identity pattern must remain consistent. Therefore, a faster degradation of Acc as w increases indicates higher sensitivity to compounding errors. Notably, on MARBLE the filtering-based baseline sMRT fails completely under long-horizon evaluation: its Acc drops from 0.69 at $w = 2$ to 0.08 at $w = 3 2$ . For readability and to keep the scale of Figure 6 informative for the remaining methods, we omit the sMRT curve from the plot of MARBLE.

Across all three datasets, all methods exhibit cascading errors: Acc generally decreases as the evaluation window size w increases. On MARBLE, the curves tend to saturate once w exceeds 16, which is consistent with the fact that each MARBLE session is relatively short (26.21 events on average). Beyond this point, enlarging the window provides limited additional temporal context to evaluate.

Within the range $w \in [ 2 , 3 2 ] ,$ , our BC and LLM+BC degrade more slowly than the baselines on all three datasets, indicating stronger robustness to error accumulation in the short-to-medium horizon. We also observe that BC on CASAS and MuRAL shows a noticeably steeper drop for $w > 3 2$ , suggesting that long-horizon performance can still be affected by compounding errors. On MuRAL, LLM+BC maintains a milder degradation trend than BC when $w > 3 2 .$ , indicating improved robustness under larger distribution shift. We conjecture that this benefit may stem from the LLM’s semantic priors, which help stabilize resident-state estimation when the model’s own prediction history becomes less reliable.

## 7.8 Inference Benefit of the MDP Formulation

As discussed in Section 5.5, one key advantage of formulating Data Association as an MDP, compared with earlier filteringbased approaches, is that it enables the use of trajectory-level inference strategies such as beam search to handle complex and ambiguous events at test time. To validate this hypothesis, we compared greedy decoding (B=1) with the beam-search using widths $B \in \{ 2 , 4 , 8 , 1 2 \}$ , holding the trained BC policy $\pi _ { \theta }$ fixed.

Table 3 reports the resulting Data Association metrics on all three datasets. Across all datasets, beam search consistently matches or improves upon greedy decoding under the same policy weights, with the clearest gains appearing in long-term metrics such as CRL and RCE. For example, CRL increases from 6.234 to 6.811 on CASAS and from 5.476 to 5.822 on MuRAL. By contrast, per-step accuracy remains relatively stable, which is expected because beam search re-ranks complete trajectories rather than isolated decisions. Performance generally saturates around B=4–8, with little additional benefit at $B { = } 1 2$ , suggesting that the relevant ambiguity is concentrated in only a small number of plausible trajectories. Overall, these results show that trajectory-level search provides a practical inference benefit without changing the policy or training data.

Figure 7 further shows that the gap between beam search and greedy decoding widens as the evaluation window increases, indicating that beam search better mitigates error accumulation over longer horizons. This trend is most evident on MuRAL, where greedy accuracy drops from 0.908 to 0.457 when the window size reaches 128. By maintaining multiple candidate trajectories instead of committing to a single early decision, beam search is less vulnerable to transient ambiguities and therefore preserves stronger long-range consistency. These results further support the role of trajectory-level search in our MDP-based formulation.

![](images/d131cad59cefbf78d30370db2efcf157834928e66277fa1277c8879526a05247.jpg)  
(a) CASAS

![](images/432ca885629cf6cb2103b8187c094e4140703d2b288753cfc47eedbe2912afd9.jpg)  
(b) MARBLE

![](images/db7b9ac07c1be845956572bd91786844093b1f2829441b622dbf7296ccf21df0.jpg)  
(c) MuRAL  
Figure 7: Resident assignment accuracy as a function of the evaluation window size (log scale) under greedy decoding and beam-search decoding on the CASAS, MARBLE, and MuRAL datasets. A slower decline as the window size increases indicates better resistance to long-horizon error accumulation.

Table 4: Qualitative example illustrating how beam search resolves an early ambiguity that greedy decoding commits to. The table shows thefirstfive sensor eventsfrom Session P20 ofthe CASAS ADLMR dataset, together with the ground-truth resident label, the prediction under greedy decoding, and the prediction under beam search (BS). Green prediction cells indicate correct assignments, red prediction cells indicate incorrect assignments, and the intensity ofblue in the probability columns reflects the magnitude of the predicted assignment probability.
<table><tr><td>Step</td><td>Event</td><td>GT ID</td><td>Greedy Pred.</td><td>Greedy Prob.</td><td>BS Pred.</td><td>BS Prob.</td></tr><tr><td>0</td><td>M15 ON</td><td>A</td><td>A, B</td><td>0.438</td><td>A</td><td>0.426</td></tr><tr><td>1</td><td>M16 ON</td><td>A</td><td>A, B</td><td>0.928</td><td>A</td><td>0.916</td></tr><tr><td>2</td><td>D07 OPEN</td><td>A</td><td>A, B</td><td>0.987</td><td>A</td><td>0.993</td></tr><tr><td>3</td><td>M17 ON</td><td>A</td><td>A, B</td><td>0.989</td><td>A</td><td>0.985</td></tr><tr><td>4</td><td>M03 ON</td><td>B</td><td>A</td><td>0.311</td><td>B</td><td>0.875</td></tr></table>

Table 4 provides a concrete example of how trajectory-level search resolves local ambiguity. At Step 0 (M15 ON), the event is ambiguous because it is the first observation in the session and no historical context is available. In addition, this sensor is statistically associated with situations in which two residents often co-occur. Accordingly, the policy predicts A with probability 0.865 and B with probability 0.507, while the remaining slots are close to zero. Under the multi-label factorization, this gives a probability of $0 . { \dot { 8 } } 6 5 \times 0 . 5 0 7 = 0 . 4 3 8$ for selecting both A and B, compared with $0 . 8 6 5 \times ( 1 - 0 . 5 0 7 ) = 0 . 4 2 6$ for selecting only A. The margin is therefore extremely small, indicating that the model has little confidence in the correct assignment of this event. Greedy decoding nevertheless forces a hard decision and commits to the joint assignment (A, B), which is then propagated through Steps 1–3 around the D07/M17 area, with confidence increasing to 0.989. The consequence becomes clear at Step 4 (M03 ON), where the sensor M03 is far from M17. Under the greedy trajectory, both A and B have already been tied to the M17 region by the earlier joint assignment, so the new event is inconsistent with the current resident states. Greedy decoding still explains it using one of the existing labels and therefore makes a low-confidence misassignment to A. Specifically, the slot probabilities are 0.537 for A and 0.421 for B, so the probability of selecting only A is $0 . 5 3 \bar { 7 } \times ( 1 - 0 . 4 2 \bar { 1 } ) = 0 . 3 \bar { 1 } 1$ . Beam search instead preserves alternative hypotheses. Under the better hypothesis, resident B has not yet been instantiated, whereas resident A remains associated with the M17 area and is therefore unlikely to trigger the distant M03 sensor. Assigning the event to a newly created B thus becomes the most plausible explanation, and beam search assigns M03 to B with high confidence (0.875). This example shows that beam search handles early ambiguity more effectively by delaying irreversible decisions until sufficient evidence becomes available.

## 7.9 Assignments of Multi-Resident Events

A multi-resident event refers to a single ambient sensor activation jointly triggered by multiple individuals, such as two colocated residents simultaneously activating the same motion sensor. Such events constitute 14.4% of the CASAS dataset and

Table 5: Performance comparison of BC and LLM+BC in assigning events to the correct number of residents on CASAS and MuRAL.
<table><tr><td rowspan="2">#Residents</td><td>Dataset</td><td>CASAS</td><td></td><td>MuRAL</td></tr><tr><td>Method</td><td>LLM+BC</td><td>BC</td><td>LLM+BC BC</td></tr><tr><td rowspan="3">Multi</td><td>Correct</td><td>1</td><td>0.354 0.413</td><td>0.418</td></tr><tr><td>Over-Assigned</td><td>1</td><td>0.000 0.000</td><td>0.010</td></tr><tr><td>Under-Assigned</td><td>1 0.646</td><td>0.587</td><td>0.572</td></tr><tr><td>Single</td><td>Over-Assigned</td><td>1</td><td>0.081</td><td>0.008 0.028</td></tr></table>

8.5% of the MuRAL dataset. To our knowledge, our MDP-based framework is the only Data Association method theoretically capable of mapping a single sensor event to a set of residents. We empirically validate this capability by evaluating the model’s precision in cardinality estimation, particularly focusing on its ability to correctly identify the number of residents involved in each event.

Admittedly, achieving high precision in this task remains a formidable challenge. In many instances, the raw binary sensor observations lack sufficient semantic distinctiveness to explicitly reveal the simultaneous presence of multiple residents, making it intrinsically difficult for any model to distinguish a multi-resident activation from a single-resident one. Consequently, as detailed in Table 5, the dominant failure mode across all settings is under-assignment, where the model defaults to a singleresident prediction. However, despite these inherent ambiguities, our framework successfully recovers the correct resident cardinality in a significant proportion of cases. These results, while leaving room for improvement, demonstrate the promising potential of our approach to handle complex, concurrent interactions that lie beyond the scope of traditional one-to-one association methods.

Specifically, on the CASAS dataset, the BC model correctly assigns 35.4% of multi-resident events to the correct set of users. On the MuRAL dataset, both BC and LLM+BC achieve a recognition rate of approximately 41%. Crucially, the comparison on MuRAL highlights the robustness of the LLM-enhanced approach. While the standard BC model tends to over-assign single-resident events as multi-resident ones (an error rate of 2.8%), LLM+BC significantly suppresses these false positives, reducing the over-assignment rate to just 0.8%. This indicates that while both methods possess similar sensitivity in detecting multi-resident activity, LLM+BC offers superior precision, leveraging semantic context to avoid hallucinating multiple residents when only one is present.

## 7.10 Cross-Environment Evaluation

Table 6: Cross-environment evaluation of the proposed LLM+BC method on MARBLE [6] and MuRAL [14] (train on one dataset and test on the other). Acc and CRL are computed with window size w = 16.
<table><tr><td>Dataset</td><td colspan="4">MARBLE [6]</td><td colspan="3">MuRAL [14]</td></tr><tr><td>Metric</td><td>Acc</td><td>CRL</td><td>BLEU</td><td>RCE</td><td>Acc CRL</td><td>BLEU</td><td>RCE</td></tr><tr><td>Without Training</td><td>0.562</td><td>3.804</td><td>0.486</td><td>1.544</td><td>0.460 2.488</td><td>0.419</td><td>1.233</td></tr><tr><td>Trained on MuRAL</td><td>0.765</td><td>8.853</td><td>0.760</td><td>0.727</td><td>0.773 6.034</td><td>0.799</td><td>0.261</td></tr><tr><td>Trained on MARBLE</td><td>0.824</td><td>8.783</td><td>0.841</td><td>0.136</td><td>0.695 5.184</td><td>0.753</td><td>0.613</td></tr></table>

A primary motivation for introducing our LLM+BC framework is to endow the overall method with cross-environment generalization capability. To validate this, we conducted a cross-evaluation experiment across different environments, as detailed in Table 6. Specifically, we train LLM+BC on one dataset (source environment), then freeze all model weights and swap the system prompt to the target environment prompt (i.e., the target dataset’s resident nomenclature and sensor topology), and finally evaluate the model on the target dataset without any additional fine-tuning. As a control, we also report a no-training ("Without Training") baseline that directly uses Qwen3-4B-Instruct-2507 as the policy in our MDP framework; its relatively low performance (e.g., 0.460 Acc on MuRAL) suggests that directly using a lightweight 4B-parameter LLM as the policy is insufficient for reliable data association, and that domain-specific fine-tuning is necessary for it to adapt to the sensor layouts and resident behaviors of a target smart home.

With this baseline in mind, we next examine cross-environment transfer when the policy is trained in a source environment and evaluated in a target environment. The results reveal a significant asymmetry in transferability: the model trained on the simpler MARBLE dataset generalizes reasonably well to the more complex MuRAL dataset, indicating that the fundamental association logic learned from MARBLE’s high-contrast and mutually exclusive patterns can transfer to more chaotic settings (achieving 0.695 Acc on MuRAL without in-domain training). However, the reverse transfer is much less effective. When trained on the complex MuRAL dataset and evaluated on MARBLE, the model’s behavior degenerates into a naive “All-to-one” baseline: despite an Accuracy of 0.765, the exceptionally high RCE (0.727) indicates that most events are indiscriminately assigned to a single resident identity. We hypothesize that this failure is caused by overfitting to the intricate and noisy distributions of MuRAL, which harms the policy’s ability to generalize to MARBLE’s simpler, mutually exclusive structure. Further adjusting the number of training epochs on MuRAL may alleviate this issue and lead to different cross-environment transfer results.

Table 7: Ablation study on the effect of metadata in LLM+BC across three datasets. LLM+BC w/o Metadata removes homespecific prompt information and sensor semantics, whereas LLM+BC w/ Metadata uses the full prompt. Acc and CRL are calculated with a window size ofw = 16.
<table><tr><td>Dataset</td><td colspan="4">CASAS [36]</td><td colspan="4">MARBLE [6]</td><td colspan="4">MuRAL [14]</td></tr><tr><td>Metric</td><td>Acc</td><td>CRL</td><td>BLEU</td><td>RCE</td><td>Acc</td><td>CRL</td><td>BLEU</td><td>RCE</td><td>Acc</td><td>CRL</td><td>BLEU</td><td>RCE</td></tr><tr><td>BC</td><td>0.757</td><td>6.234</td><td>0.727</td><td>0.021</td><td>0.792</td><td>8.710</td><td>0.797</td><td>0.136</td><td>0.728</td><td>5.476</td><td>0.774</td><td>0.437</td></tr><tr><td>LLM+BC w/o Metadata</td><td>0.761</td><td>6.135</td><td>0.725</td><td>0.024</td><td>0.789</td><td>8.037</td><td>0.786</td><td>0.182</td><td>0.756</td><td>5.942</td><td>0.777</td><td>0.632</td></tr><tr><td>LLM+BC w/ Metadata</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0.824</td><td>8.783</td><td>0.841</td><td>0.136</td><td>0.773</td><td>6.034</td><td>0.799</td><td>0.261</td></tr></table>

## 7.11 Impact of Metadata on LLMs

Although the LLM+BC framework demonstrates strong potential, it still relies heavily on environment-specific metadata, and such customized information may not always be available in practical deployments. It is therefore important to examine how the method behaves when these inputs are missing. To this end, we conduct an ablation study in which we remove the House Layout, Sensor Description, and Key Constraints sections from the original system prompt. We further strip home-specific information from the Background and Objective sections, yielding a simplified and unified prompt shared across all datasets. In addition, we anonymize the sensors in each dataset by assigning each sensor an arbitrary numeric identifier with no semantic meaning, and represent every sensor state change uniformly as ON or OFF. Using this prompt and data representation, we retrain the LLM on all three datasets. The results, shown in Table 7, indicate that without metadata, the performance of LLM+BC falls back to a level very close to that of BC. This result is consistent with the design of the ablation, as removing metadata effectively reduces the method from text-enhanced data-driven reasoning to a purely data-driven formulation. Conversely, providing metadata consistently improves performance on MARBLE and MuRAL, confirming that explicit semantic and structural context is a key factor behind the gains of the LLM-enhanced policy.

## 7.12 Robustness to Noise

In real deployments, sensor streams may contain a certain proportion of noisy events caused by sensor faults or other environmental disturbances. Compared with previous methods, our MDP formulation introduces an implicit skip action: when the policy assigns a probability below the multi-label decision threshold (set to 0.5 in our experiments) to every resident slot, the model makes no resident assignment for the current event and simply skips it. We hypothesize that this mechanism provides a natural source of robustness to noise. To test this hypothesis, we inject random noise events into the CASAS dataset at rates of 1%, 5%, and 10%. These noise events are constructed from existing sensors in the environment paired with randomly sampled state changes. During BC training, all resident slots for these noise events are labeled as 0, indicating that the noise belong to no resident. In Table 8, we compare performance degradation under different noise levels against the clean setting, and also report whether the model successfully skips the injected noise. We define Skip Rate as the proportion of noise events that are skipped, and False Skip Rate as the proportion of valid events that are mistakenly treated as noise and skipped. The results show that, compared with sMRT, our BC method remains highly stable under 1% noise injection, with almost no performance degradation, while skipping about 10% of the noise events. When the noise ratio increases to 5%, the Skip Rate rises to 33%, while the False Skip Rate remains only 0.7%, which helps prevent excessive performance degradation. At 10% noise, however, even skipping roughly half of the noise events is insufficient to prevent noticeable degradation, because a large amount of noise still enters the state history. In addition, the False Skip Rate rises sharply to 11%, meaning that a substantial portion of valid events is incorrectly discarded, which further harms state construction. Nevertheless, our MDP-based method provides a skip mechanism that is absent from prior approaches, enabling it to maintain stronger performance than previous methods under moderate levels of noise.

## 7.13 Simulated Downstream Health Evaluation via Activity Duration Analysis

Finally, we return to the broader motivation of this work: to extend smart home applications that were previously developed for single-resident ambient sensor environments into multi-resident settings in an algorithmic manner. To this end, we finally verify the practical potential of our framework for downstream health-related applications. Following Aramendi et al. [5], who assessed residents’ health status from activity durations in single-resident CASAS environments, we ask whether a similar pipeline can be extended to the corresponding two-resident CASAS setting using the sequences separated by our method. To isolate the effect of data association, we assume 100% HAR accuracy on the separated single-resident sequences, i.e., we directly use the ground-truth activity labels. This simplification avoids introducing additional noise from HAR, which lies outside the main scope of our contribution, and is also reasonable because Aramendi et al. reported over 98% HAR accuracy on single-resident CASAS sequences. We then compare predicted and ground-truth activity durations at the (resident, activity) level. On the ground-truth side, each activity duration is computed directly from the CASAS annotations. On the prediction side, we align the session-local resident labels produced by our method with the CASAS resident IDs using the first event of the session, and then apply this mapping to all subsequent events to reconstruct the predicted (r, a) segments. For each resident–activity pair, we finally compare the predicted duration against the ground truth.

Resident 1  
Table 8: Robustness of data association under injected noise on the CASAS dataset. Random noise events are added at different ratios. Skip denotes the proportion of injected noise events that are skipped by the model, while False Skip denotes the proportion of valid events that are incorrectly skipped.
<table><tr><td>Methods</td><td>Noise%</td><td>Acc</td><td>CRL</td><td>BLEU</td><td>RCE</td><td>Skip</td><td>False Skip</td></tr><tr><td rowspan="4">BC</td><td>0%</td><td>0.757</td><td>6.234</td><td>0.727</td><td>0.021</td><td>0.000</td><td>0.000</td></tr><tr><td>1%</td><td>0.755</td><td>6.201</td><td>0.721</td><td>0.021</td><td>0.103</td><td>0.001</td></tr><tr><td>5%</td><td>0.701</td><td>5.838</td><td>0.703</td><td>0.032</td><td>0.334</td><td>0.007</td></tr><tr><td>10%</td><td>0.581</td><td>4.690</td><td>0.591</td><td>0.043</td><td>0.506</td><td>0.112</td></tr><tr><td rowspan="2">sMRT</td><td>0%</td><td>0.561</td><td>3.051</td><td>0.530</td><td>1.257</td><td>1</td><td>1</td></tr><tr><td>1%</td><td>0.232</td><td>1.501</td><td>0.190</td><td>5.690</td><td>1</td><td>1</td></tr></table>

Table 9: Per-session activity durations (seconds) for Resident 1 and Resident 2. Each entry is reported as predicted / ground truth. T01–T15 denote the 15 scripted activities defined in the CASAS dataset.
<table><tr><td>Session</td><td>T01</td><td>T04</td><td>T06</td><td>T09</td><td>T10</td><td>T12</td><td>T13</td><td>T14</td></tr><tr><td>P19</td><td>146/170</td><td>17/17</td><td>0/123</td><td>411 /411</td><td>92 / 107</td><td>329/ 329</td><td>12 /13</td><td>111/112</td></tr><tr><td>P20</td><td>192 / 192</td><td>29/29</td><td>18/64</td><td>348 / 348</td><td>84/86</td><td>427 /434</td><td>226/49</td><td>59/95</td></tr><tr><td>P21</td><td>247 /247</td><td>25 / 25</td><td>26/48</td><td>406 / 406</td><td>30/74</td><td>211/211</td><td>5/29</td><td>58/58</td></tr><tr><td>P22</td><td>200 / 200</td><td>42/43</td><td>22/41</td><td>306 / 306</td><td>65 /65</td><td>330/330</td><td>14/2</td><td>86/187</td></tr><tr><td>P23</td><td>301/ 305</td><td>31/31</td><td>84/128</td><td>320/ 320</td><td>137 /137</td><td>0/246</td><td>273/0</td><td>24/81</td></tr><tr><td>P24</td><td>350/350</td><td>36/36</td><td>72/100</td><td>375 / 375</td><td>79/79</td><td>406/554</td><td>1/88</td><td>94/97</td></tr><tr><td>P25</td><td>311/379</td><td>65/65</td><td>112/153</td><td>67/392</td><td>0/81</td><td>138/505</td><td>290 /155</td><td>77 /106</td></tr><tr><td>P26</td><td>312 /312</td><td>50/50</td><td>54/85</td><td>347/347</td><td>13/89</td><td>436 /436</td><td>84/16</td><td>86/93</td></tr><tr><td>Our Err %</td><td>-4.5%</td><td>-0.3%</td><td>-47.7%</td><td>-11.2%</td><td>-30.4%</td><td>-25.2%</td><td>+157.1%</td><td>-28.2%</td></tr><tr><td>sMRT Err %</td><td>-72.7%</td><td>-45.4%</td><td>-78.9%</td><td>-55.3%</td><td>-87.0%</td><td>-53.7%</td><td>+12.2%</td><td>-98.7%</td></tr><tr><td colspan="9">Resident 2</td></tr><tr><td>Session</td><td>T02</td><td>T03</td><td>T05</td><td>T07</td><td>T08</td><td>T11</td><td>T13</td><td>T15</td></tr><tr><td>P19</td><td>64/73</td><td>20/20</td><td>67 / 70</td><td>17/ 55</td><td>0/340</td><td>119 / 140</td><td>244/329</td><td>113 / 116</td></tr><tr><td>P20</td><td>83 / 83</td><td>18 / 18</td><td>26/26</td><td>62/139</td><td>267 /267</td><td>96/98</td><td>351/351</td><td>67/98</td></tr><tr><td>P21</td><td>51/ 53</td><td>17 /17</td><td>86/86</td><td>28/41</td><td>345 / 345</td><td>125 / 131</td><td>173 /218</td><td>55/68</td></tr><tr><td>P22</td><td>103 / 103</td><td>36/42</td><td>20/48</td><td>48/52</td><td>140/217</td><td>66/66</td><td>238/337</td><td>115/184</td></tr><tr><td>P23</td><td>70/70</td><td>24 / 24</td><td>179 / 186</td><td>112 / 122</td><td>306 /314</td><td>11/127</td><td>0/273</td><td>37/84</td></tr><tr><td>P24</td><td>92/92</td><td>45/45</td><td>233 /238</td><td>104/143</td><td>0/328</td><td>88/91</td><td>426/600</td><td>29/111</td></tr><tr><td>P25</td><td>88/ 96</td><td>61 / 61</td><td>134 / 134</td><td>211/221</td><td>289/325</td><td>14/115</td><td>150/463</td><td>49/109</td></tr><tr><td>P26</td><td>191 / 197</td><td>34/47</td><td>38/38</td><td>165 / 177</td><td>283 /283</td><td>16/94</td><td>300 /459</td><td>81/94</td></tr><tr><td>Our Err %</td><td>-3.3%</td><td>-6.9%</td><td>-5.2%</td><td>-21.4%</td><td>-32.6%</td><td>-37.9%</td><td>-37.9%</td><td>-36.8%</td></tr><tr><td>sMRT Err %</td><td>-55.7%</td><td>-42.2%</td><td>-60.7%</td><td>-83.8%</td><td>-45.2%</td><td>-82.5%</td><td>-71.4%</td><td>-94.5%</td></tr></table>

Table 9 reports the per-session activity durations for each resident on the test sessions P19–P26, where T01–T15 denote the 15 scripted ADL tasks in the CASAS protocol. Overall, compared with the fragmented trajectories produced by sMRT, our method substantially reduces activity-duration estimation errors, with the clearest improvements appearing in T01–T05, which occur in the early part of every session before substantial error accumulation sets in. This improvement further suggests the practical potential of our method for supporting duration-based downstream health analysis.

However, the more noticeable remaining discrepancies tend to appear in later activities (T06–T15), where two factors compound the difficulty: first, error accumulation over time makes identity swaps more likely; second, these activities often involve close interactions and overlapping sensor activations between the two residents, which further increases the difficulty of distinguishing them. In particular, Resident 1’s T06 and Resident 2’s T07 are difficult to distinguish, and a similar pattern appears for Resident 2’s T08 and Resident 1’s T09 during the dinner-preparation stage. These activity pairs often occur in parallel, exhibit substantial spatial overlap, and involve many multi-resident sensor events. For example, in P19 and P24, Resident 2’s T08 is predicted entirely as Resident 1’s T09, indicating that the two residents’ trajectories have effectively been merged into a single track; this again reflects the difficulty of handling multi-resident events. A related source of error appears in T13, which is primarily carried out by Resident 2 but includes brief assistance from Resident 1. Because this activity contains a short but meaningful overlap between the two residents, the model may either absorb Resident 1’s contribution into Resident 2’s main trajectory or over-extend Resident 1’s involvement, thereby over-predicting Resident 1’s T13 duration. P23 provides a representative example of a longer-horizon failure mode. In this session, Resident 1’s T12 and T13 become fully entangled with Resident 2’s T11 and T13. At the onset of T11 and T12, the two residents are assigned in a swapped manner, and this identity reversal then persists through T13 and continues until T14 and T15. Thus, the dominant error is not an isolated duration mismatch within a single activity, but a sustained identity swap over the later part of the session that systematically exchanges the two residents’ trajectories. Taken together, these results suggest that the remaining duration errors are driven less by isolated activity recognition failures than by ambiguous multi-resident interactions and long-horizon identity swaps in the later parts of the session.

## 8 Limitations and Future Work

Our work establishes a technical foundation for extending smart-home ambient sensor applications from single-resident settings to multi-resident environments by formulating Data Association as an MDP and solving it through BC with both Transformerbased and LLM-based policies. Across all three datasets, the proposed framework consistently outperforms prior baselines in tracking consistency, resident-count estimation, overlapping-event attribution, robustness to noise, and cross-environment generalizability. These improvements appear to arise from two main factors: first, the MDP formulation captures global spatiotemporal consistency over assignment histories; second, the BC paradigm enables data-driven and effective policy learning without requiring carefully designed reward functions. Consequently, the framework mitigates the severe trajectory fragmentation often observed in filtering-based methods such as sMRT, while also allowing deferred decision-making, in which later observations help resolve earlier ambiguities during inference. Moreover, experiments with LLM-based policies indicate that the proposed training paradigm not only surpasses prior prompt-based methods such as LAHAR, but also retains useful association capability under reduced metadata conditions. Finally, the simulated downstream health-analysis experiment suggests that improved DA can substantially reduce activity-duration estimation errors in multi-resident settings, highlighting the broader potential of this framework as an enabling layer for future smart-home applications. Nevertheless, the results also reveal several important limitations that must be addressed before such methods can be robustly deployed at scale

## 8.1 Error Accumulation and Distribution Shift

Although our supervised Behavioral Cloning (BC) approach is effective, our experimental analysis reveals a susceptibility to error accumulation over long horizons. As demonstrated by the window-size analysis in Section 7.7, tracking performance degrades notably as the evaluation window increases. This indicates that early identity assignment mistakes are not isolated events; rather, they propagate through the maintained resident histories, causing the agent to drift further from the correct trajectory. Theoretically, this phenomenon stems from the covariate shift inherent in sequential decision-making. Standard BC relies on the assumption of independent and identically distributed (i.i.d.) data, training the policy exclusively on perfect expert trajectories. Consequently, when the agent encounters “corrupted” states caused by its own prediction errors during inference, it lacks the capability to recover. To mitigate this issue, we encourage future research to explore two complementary directions: training-side robustness and deployment-side lifecycle management.

From the training perspective, adopting Dataset Aggregation (DAgger) [35] offers a promising solution. Unlike static BC, DAgger allows the learner policy to execute actions during training, naturally generating deviant system states. By querying the expert for correct assignments given these corrupted histories, the policy explicitly learns to recover from tracking errors. This interactive process effectively trains the agent to re-associate events correctly even after misallocations, thereby enhancing the policy’s intrinsic robustness against distribution shift.

Beyond imitation-learning extensions, an important next step is to build on the strong BC baseline with long-horizon Reinforcement Learning. The BC-trained policy provides a stable warm start, after which the agent can sample its own trajectories and be optimized for trajectory-level association quality. This direction is also supported by our findings in Section 7.8: the gains from trajectory-level search suggest that preserving multiple candidate futures can already mitigate error accumulation at inference time. RL offers a natural way to move this benefit into training, shifting the objective from matching expert actions locally to improving the global consistency of the full attribution sequence. Rather than relying on dense heuristic rewards, a more principled design could use delayed rewards over blocks of events together with a value model that estimates the utility of intermediate states. A further extension is to enlarge the action space itself so that the policy is not restricted to irreversible event-to-resident assignments. Besides assigning the current event, the agent could invoke temporary skip actions or history-editing actions that revise earlier attributions when later evidence reveals an inconsistency. This idea is especially natural for the LLM-based policy, which could emit structured edit commands over the resident history. Overall, BC initialization, beam-search inference, and subsequent RL fine-tuning—potentially with an expanded action space for corrective decisions—form a natural path toward more stable long-horizon reasoning.

From the deployment perspective, future work may benefit from system-level lifecycle management inspired by classical Multi-Target Tracking (MTT) theory [9]. Maintaining resident states indefinitely can risk permanent divergence, similar to the kidnapped robot problem [37]. We believe an Error-Triggered Re-initialization mechanism may be useful: by monitoring uncertainty metrics, the system can detect out-of-distribution states and perform a hard reset to truncate error propagation. To reduce track fragmentation, one practical strategy may be to execute such a reset when the environment is inferred to be vacant, for example, when the house is inferred to be empty.

## 8.2 Multi-Resident Events

An important advantage of our proposed method is applying multi-label classifier, which, to our knowledge, is the first Data Association model that can assign a single ambient sensor event to multiple residents. However, as shown in Section 7.9, although our method can correctly recover the resident cardinality for a subset of multi-resident events, a substantial fraction is still under-assigned and treated as single-resident events.

We believe this limitation is primarily driven by the data rather than the model formulation. Truly concurrent events are both intrinsically ambiguous under binary ambient sensors and relatively rare in the available datasets (accounting for approximately 10% in MuRAL and CASAS), which limits the evidence available for learning and evaluation. Future work could therefore prioritize data augmentation via trajectory mixing to synthesize additional multi-resident overlaps and strengthen the model’s sensitivity to concurrent activations.

## 8.3 Datasets Limitations

Although our work has been comprehensively evaluated across three datasets with diverse characteristics, we must acknowledge the inherent domain gap between these controlled experimental setups and in-the-wild smart home environments. The primary discrepancy lies in sensor topology and deployment density. As introduced in Section 7.1, the CASAS environment features a high-density array of motion sensors mounted on the ceiling. Such a high-cost and technically demanding deployment differs drastically from the reality of consumer smart homes. Similarly, while MARBLE and MuRAL more closely resemble modern smart home configurations, they still utilize over 16 sensors and employ highly localized, custom deployment strategies designed specifically to isolate individuals (e.g., the seating area in MARBLE is densely equipped with eight distinct pressure mats to ensure the exclusive capture of a single occupant’s movements). Such fine-grained sensor layouts are exceedingly rare in typical households. Secondly, our method faces a steep barrier regarding annotation costs. Under the current supervised learning paradigm, our proposed approach relies heavily on event-level identity annotations. Acquiring such detailed labels in everyday home environments without professional volunteers is prohibitively expensive and highly impractical. Consequently, the actual generalization capability and deployment performance of our method in natural, resource-constrained households with sparse sensor configurations remain to be further validated in future work.

## 8.4 Cross-Environment Generalization and Reasoning

Although our LLM+BC framework exhibits promising cross-dataset transfer, relying on direct input-to-output mapping limits its robustness against severe domain shifts. Without explicit reasoning, the policy risks overfitting to superficial statistical correlations specific to the training sensors, rather than learning the invariant causal logic required for data association. We hypothesize that incorporating Chain-of-Thought (CoT) [41] is crucial for bridging this gap. To elicit the latent reasoning capabilities of LLMs and achieve true generalizability, future work should explore Reasoning-Aware Fine-tuning through Reasoning Distillation via Rejection Sampling [44] and Outcome-Supervised Reinforcement Learning [22].

## 8.5 Scalability and Real-time System Integration

Beyond the domain gaps discussed above, translating our theoretical framework into an industrial smart home foundation requires addressing additional practical considerations: scalability concerning the number of residents and integration into real-time streaming systems.

Regarding scalability, as the number of household members increases, the ambiguity of data association and the corresponding search space typically experience exponential growth. However, our MDP formulation is logically scalable: introducing new residents merely results in a linear expansion of the action space, which our Transformer/LLM-based policy can naturally accommodate through contextual sequence modeling. Computationally, beam search effectively serves to cap the processing overhead. By strictly bounding the number of maintained hypotheses to a constant beam width B, the system bypasses the combinatorial explosion of exhaustive search, ensuring that the inference complexity per step scales linearly rather than exponentially with the number of residents.

Regarding system integration, our method is designed to function as a low-level data processing middleware seamlessly plugged into real-time streaming smart home ecosystems. In practice, as asynchronous sensor events arrive, the system maintains a sliding temporal window and a state buffer. For unambiguous events, the module yields immediate assignments to ensure low latency. When facing highly ambiguous scenarios, it triggers a “deferred decision”, leveraging the buffer to cross-validate and prune hypotheses as new streaming data arrives. In practice, many downstream applications, such as health monitoring and behavioral analysis, do not require extremely strict latency and therefore align well with our framework, which can trade a small amount of delay for better long-horizon consistency. For applications that do require near-real-time responses, such as home security anomaly detection or smart-home automation, the Transformer-based policy can fall back to greedy online inference to ensure low latency. When an LLM is used as the policy, deployment is also increasingly practical: models at the 4B scale can already be run locally on commodity devices such as a Mac mini with sufficiently fast inference for system integration, allowing the policy to be exposed as an API within a smart-home pipeline.

## 9 Conclusions

In this paper, we presented a framework to address the challenge of Data Association in multi-resident smart home environments. We identified that existing approaches, whether relying on event-centric classification or Bayesian filtering, are fundamentally limited by the rigid first-order Markov assumption and the lack of a persistent global memory. These limitations frequently result in inconsistent assignments for silent residents and trajectory fragmentation in complex scenarios.

To overcome these structural bottlenecks, we reformulated the Data Association problem as a sequential decision-making task within a Markov Decision Process (MDP). This formulation allows us to explicitly represent the global assignment history and reason over longer behavioral context, although our experiments also show that robust long-horizon tracking remains challenging because early mistakes can accumulate under Behavior Cloning. By solving this MDP through Behavior Cloning, we adopt a fully data-driven approach that avoids handcrafted reward design and the tractability constraints of traditiona particle filters, but currently still depends on environment-specific supervision for reliable adaptation. Our framework was instantiated with two complementary policy architectures: a lightweight Transformer that handles association as a multi-label classification task, and a fine-tuned Large Language Model that reformulates the problem as conditional text generation to leverage pre-trained semantic knowledge.

Extensive evaluations on the CASAS, MARBLE, and MuRAL datasets demonstrate that our proposed method significantly outperforms state-of-the-art baselines on these benchmarks. Our results specifically highlight that the framework improves global consistency relative to prior methods and is particularly effective at resolving concurrent events involving multiple residents. Furthermore, the LLM-based policy shows encouraging cross-environment transfer under controlled benchmark settings, suggesting that leveraging semantic reasoning is a promising direction, while its robustness under sparse, outdated, or changing metadata and under real household deployment conditions remains unresolved.

Overall, we view this work as a technical foundation rather than a deployment-ready solution. Moving toward real-world smart-home deployment will require addressing long-horizon error accumulation, robustness to delayed and missing sensor data, dependence on static metadata, annotation cost, and validation in more natural household environments.

## References

[1] Giovanni Acampora, Diane J Cook, Parisa Rashidi, and Athanasios V Vasilakos. A survey on ambient intelligence in healthcare. Proceedings ofthe IEEE, 101(12):2470–2494, 2013.

[2] Md Afroz, Emmanuel Nyakwende, and Birendra Goswami. Intrusion detection in smart home environments: A machine learning approach. Transportation Research Procedia, 84:243–250, 2025.

[3] Simin Ahmadi-Karvigh, Ali Ghahramani, Burcin Becerik-Gerber, and Lucio Soibelman. Real-time activity recognition for energy efficiency in buildings. Applied energy, 211:146–160, 2018.

[4] Hande Alemdar, Halil Ertan, Ozlem Durmaz Incel, and Cem Ersoy. Aras human activity datasets in multiple homes with multiple residents. In 2013 7th International Conference on Pervasive Computing Technologies for Healthcare and Workshops, pages 232–235. IEEE, 2013.

[5] Ane Alberdi Aramendi, Alyssa Weakley, Asier Aztiria Goenaga, Maureen Schmitter-Edgecombe, and Diane J Cook. Automatic assessment of functional health decline in older adults based on smart home data. Journal ofbiomedical informatics, 81:119–130, 2018.

[6] Luca Arrotta, Claudio Bettini, and Gabriele Civitarese. The marble dataset: Multi-inhabitant activities of daily living combining wearable and environmental sensors data. In International Conference on Mobile and Ubiquitous Systems: Computing, Networking, and Services, pages 451–468, Cham, 2021. Springer, Springer International Publishing.

[7] Luca Arrotta, Gabriele Civitarese, Xi Chen, Julien Cumin, and Claudio Bettini. Multi-subject human activities: A survey of recognition and evaluation methods based on a formal framework. Expert Systems with Applications, page 126178, 2024.

[8] Asma Benmansour, Abdelhamid Bouchachia, and Mohammed Feham. Multioccupant activity recognition in pervasive smart home environments. ACM Computing Surveys (CSUR), 48(3):1–36, 2015.

[9] Samuel S Blackman and Robert Popoli. Design and analysis ofmodern tracking systems. Artech House, 1999.

[10] Andrew Chan, Joanne Cai, Linna Qian, Brendan Coutts, Steven Phan, Geoff Gregson, Michael Lipsett, Adriana M Ríos Rincón, et al. In-home positioning for remote home health monitoring in older adults: Systematic review. JMIR aging, 7(1):e57320, 2024.

[11] Rong Chen and Yu Tong. A two-stage method for solving multi-resident activity recognition in smart environments. Entropy, 16(4):2184–2203, 2014.

[12] Xi Chen, Julien Cumin, Fano Ramparany, and Dominique Vaufreydaz. Generative resident separation and multi-label classification for multi-person activity recognition. In 2024 IEEE International Conference on Pervasive Computing and Communications Workshops and other Affiliated Events (PerCom Workshops), pages 1–6, Biarritz, France, 2024. IEEE, IEEE.

[13] Xi Chen, Julien Cumin, Fano Ramparany, and Dominique Vaufreydaz. Towards llm-powered ambient sensor based multi-person human activity recognition. In 2024 IEEE 30th International Conference on Parallel and Distributed Systems (ICPADS), pages 609–616, Belgrade, Serbia, 2024. IEEE, IEEE.

[14] Xi Chen, Julien Cumin, Fano Ramparany, and Dominique Vaufreydaz. MuRAL: A Multi-Resident Ambient Sensor Dataset Annotated with Natural Language for Activities of Daily Living. HAL preprint hal-05048859, 2025. URL https://hal. science/hal-05048859.

[15] Yi-Ting Chiang, Kuo-Chung Hsu, Ching-Hu Lu, Li-Chen Fu, and Jane Yung-Jen Hsu. Interaction models for multiple-resident activity recognition in a smart home. In 2010 IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 3753–3758. IEEE, 2010.

[16] Diane J Cook, Aaron Crandall, Geetika Singla, and Brian Thomas. Detection of social interaction in smart spaces. Cybernetics and Systems: An International Journal, 41(2):90–104, 2010.

[17] Aaron S Crandall and Diane Cook. Attributing events to individuals in multi-inhabitant environments. In 2008 IET 4th International Conference on Intelligent Environments, pages 1–8. IET, 2008.

[18] Aaron S Crandall and Diane J Cook. Resident and caregiver: Handling multiple people in a smart care facility. In AAAI Fall symposium: AI in eldercare: new solutions to old problems, pages 39–47, 2008.

[19] Aaron S Crandall and Diane J Cook. Using a hidden markov model for resident identification. In 2010 Sixth International Conference on Intelligent Environments, pages 74–79. IEEE, 2010.

[20] Daniel Fährmann, Laura Martín, Luis Sánchez, and Naser Damer. Anomaly detection in smart environments: a comprehensive survey. IEEE access, 2024.

[21] Giorgos Giannios, Lampros Mpaltadoros, Vasilis Alepopoulos, Margarita Grammatikopoulou, Thanos G Stavropoulos, Spiros Nikolopoulos, Ioulietta Lazarou, Magda Tsolaki, and Ioannis Kompatsiaris. A semantic framework to detect problems in activities of daily living monitored through smart home sensors. Sensors, 24(4):1107, 2024.

[22] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638, 2025.

[23] SJ Hayward, Katherine van Lopik, Christopher Hinde, and Andrew A West. A survey of indoor location technologies, techniques and applications in industry. Internet ofThings, 20:100608, 2022.

[24] Kuo-Chung Hsu, Yi-Ting Chiang, Gu-Yang Lin, Ching-Hu Lu, Jane Yung-Jen Hsu, and Li-Chen Fu. Strategies for inference mechanism of conditional random fields for multiple-resident activity recognition in a smart home. In Trends in Applied Intelligent Systems: 23rd International Conference on Industrial Engineering and Other Applications of Applied Intelligent Systems, IEA/AIE 2010, Cordoba, Spain, June 1-4, 2010, Proceedings, Part I 23, pages 417–426. Springer, 2010.

[25] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.

[26] Behzad Lashkari, Yuxiang Chen, and Petr Musilek. Energy management for smart homes—state of the art. Applied Sciences, 9 (17):3459, 2019.

[27] Baihan Lin. Behavioral cloning and imitation learning. In Reinforcement Learning Methods in Speech and Language Technology, pages 63–67. Springer, 2024.

[28] Umar Mahmud, Shariq Hussain, and Tehmina Karamat. Towards an intelligent energy conservation approach for context-aware systems in smart environments. Frontiers in Computer Science, 6:1525382, 2024.

[29] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pages 311–318, USA, 2002. Association for Computational Linguistics.

[30] Martin L Puterman. Markov decision processes: discrete stochastic dynamic programming. John Wiley & Sons, 2014.

[31] Asif Rahim, Yanru Zhong, Tariq Ahmad, Sadique Ahmad, Paweł Pławiak, and Mohamed Hammad. Enhancing smart home security: anomaly detection and face recognition in smart home iot devices using logit-boosted cnn models. Sensors, 23(15): 6979, 2023.

[32] Md Motiur Rahman, Deepti Gupta, Smriti Bhatt, Shiva Shokouhmand, and Miad Faezipour. A comprehensive review of machine learning approaches for anomaly detection in smart homes: experimental analysis and future directions. Future Internet, 16(4): 139, 2024.

[33] Daniele Riboni and Flavia Murru. Unsupervised recognition of multi-resident activities in smart-homes. IEEE Access, 8: 201985–201994, 2020.

[34] Daniele Riboni, Claudio Bettini, Gabriele Civitarese, Zaffar Haider Janjua, and Rim Helaoui. Smartfaber: Recognizing fine grained abnormal behaviors for early detection of mild cognitive impairment. Artificial intelligence in medicine, 67:57–74, 2016.

[35] Stéphane Ross, Geoffrey Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings ofthefourteenth international conference on artificial intelligence and statistics, pages 627–635. JMLR Workshop and Conference Proceedings, 2011.

[36] Geetika Singla, Diane J Cook, and Maureen Schmitter-Edgecombe. Recognizing independent and joint activities among multiple residents in smart environments. Journal ofambient intelligence and humanized computing, 1:57–63, 2010.

[37] Sebastian Thrun, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.

[38] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[39] Tinghui Wang and Diane J Cook. smrt: Multi-resident tracking in smart homes with sensor vectorization. IEEE transactions on pattern analysis and machine intelligence, 43(8):2809–2821, 2020.

[40] Tinghui Wang and Diane J Cook. Multi-person activity recognition in continuously monitored smart homes. IEEE transactions on emerging topics in computing, 10(2):1130–1141, 2021.

[41] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

[42] Daniel H Wilson and Chris Atkeson. Simultaneous tracking and activity recognition (star) using many anonymous, binary sensors. In International conference on pervasive computing, pages 62–79. Springer, 2005.

[43] Zhouwen Wu, Xia Chen, Yujun Lin, Jinyu Wen, and Yin Chen. A smart home energy management system based on human activity recognition and deep reinforcement learning. Energy and Buildings, 325:114951, 2024.

[44] Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman. Star: Bootstrapping reasoning with reasoning. In Advances in Neural Information Processing Systems, volume 35, pages 15476–15488, 2022.

[45] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. A survey of large language models. arXiv preprint arXiv:2303.18223, 1(2):1–124, 2023.